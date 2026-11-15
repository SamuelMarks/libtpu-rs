//! HLO dot/convolution legalization, algebraic/constant/graph simplifiers, LICM, CSE, DCE,
//! async collective Start/Done conversion, and latency-hiding overlap scheduling.

use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    AlgebraicSimplifierOptionsProto, ConvolutionDimensionNumbers, DotDimensionNumbers,
    HloComputationProto, HloInstructionProto, HloInstructionSequenceProto, HloModuleProto,
    HloScheduleProto, LiteralProto, PaddingConfig, PaddingConfigDimension, PrimitiveType,
    ShapeProto, SourceTarget,
};
use std::collections::{BTreeMap, BTreeSet};

/// Canonicalization summary produced by [`DotDecomposerPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DotDecomposerReport {
    /// Number of general `dot` instructions decomposed into canonical `[B, M, K] x [B, K, N]` contractions.
    pub decomposed_dots: usize,
    /// Number of auxiliary `transpose` / `reshape` wrapper ops inserted.
    pub inserted_wrappers: usize,
}

/// Canonicalizes arbitrary-rank `dot` / `dot_general` instructions into MXU-ready rank-2 (`[M, K] x [K, N]`)
/// or rank-3 (`[B, M, K] x [B, K, N]`) contractions (`DotDecomposerPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DotDecomposerPass;

impl DotDecomposerPass {
    /// Runs canonical `dot` decomposition across all computations in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] whose `dot` instructions are canonicalized.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if any `DotDimensionNumbers` index is out of bounds.
    pub fn run(&self, module: &mut HloModuleProto) -> LibTpuResult<DotDecomposerReport> {
        let mut report = DotDecomposerReport::default();
        for comp in &mut module.computations {
            decompose_dots_in_computation(comp, &mut report)?;
        }
        Ok(report)
    }
}

/// Decomposes non-canonical `dot` operations inside `comp`.
///
/// # Arguments
///
/// * `comp` - Mutable computation to transform.
/// * `report` - Summary counter accumulator.
#[allow(clippy::too_many_lines)]
fn decompose_dots_in_computation(
    comp: &mut HloComputationProto,
    report: &mut DotDecomposerReport,
) -> LibTpuResult<()> {
    let mut next_id = max_instruction_id(comp).saturating_add(1);
    let mut shape_map: BTreeMap<i64, ShapeProto> = comp
        .instructions
        .iter()
        .filter_map(|i| i.shape.as_ref().map(|s| (i.id, s.clone())))
        .collect();
    let original = std::mem::take(&mut comp.instructions);
    let mut out = Vec::with_capacity(original.len() * 2);
    let mut remap: BTreeMap<i64, i64> = BTreeMap::new();

    for mut inst in original {
        rewrite_operands(&mut inst, &remap);
        if inst.opcode != "dot" || inst.operand_ids.len() < 2 {
            out.push(inst);
            continue;
        }
        let lhs_id = inst.operand_ids.first().copied().unwrap_or(0);
        let rhs_id = inst.operand_ids.get(1).copied().unwrap_or(0);
        let lhs_shape = shape_map.get(&lhs_id).cloned().unwrap_or_default();
        let rhs_shape = shape_map.get(&rhs_id).cloned().unwrap_or_default();
        let dnums = inst
            .dot_dimension_numbers
            .clone()
            .unwrap_or_else(|| DotDimensionNumbers {
                lhs_contracting_dimensions: vec![1],
                rhs_contracting_dimensions: vec![0],
                lhs_batch_dimensions: Vec::new(),
                rhs_batch_dimensions: Vec::new(),
            });
        validate_dot_dims(
            &dnums,
            lhs_shape.dimensions.len(),
            rhs_shape.dimensions.len(),
        )?;

        if is_canonical_dot(&dnums, &lhs_shape, &rhs_shape) {
            if inst.dot_dimension_numbers.is_none() {
                inst.dot_dimension_numbers = Some(dnums);
            }
            out.push(inst);
            continue;
        }

        let elem_ty = inst
            .shape
            .as_ref()
            .map_or(lhs_shape.element_type, |s| s.element_type);
        let orig_out_shape = inst.shape.clone().unwrap_or_default();

        let (left_canon_id, batch_ext, m_ext, k_ext, lhs_wraps) = canonicalize_dot_operand(
            lhs_id,
            &lhs_shape,
            &dnums.lhs_batch_dimensions,
            &dnums.lhs_contracting_dimensions,
            true,
            &mut next_id,
            &mut out,
        );
        let (right_canon_id, _, n_ext, _, rhs_wraps) = canonicalize_dot_operand(
            rhs_id,
            &rhs_shape,
            &dnums.rhs_batch_dimensions,
            &dnums.rhs_contracting_dimensions,
            false,
            &mut next_id,
            &mut out,
        );
        report.inserted_wrappers = report
            .inserted_wrappers
            .saturating_add(lhs_wraps + rhs_wraps);

        let (dot_dims, canon_dnums) = if dnums.lhs_batch_dimensions.is_empty() {
            let _ = k_ext;
            (
                vec![m_ext, n_ext],
                DotDimensionNumbers {
                    lhs_batch_dimensions: Vec::new(),
                    rhs_batch_dimensions: Vec::new(),
                    lhs_contracting_dimensions: vec![1],
                    rhs_contracting_dimensions: vec![0],
                },
            )
        } else {
            (
                vec![batch_ext, m_ext, n_ext],
                DotDimensionNumbers {
                    lhs_batch_dimensions: vec![0],
                    rhs_batch_dimensions: vec![0],
                    lhs_contracting_dimensions: vec![2],
                    rhs_contracting_dimensions: vec![1],
                },
            )
        };

        let dot_id = alloc_id(&mut next_id);
        let dot_shape = make_tensor_shape(elem_ty, dot_dims.clone());
        let orig_id = inst.id;
        inst.id = dot_id;
        inst.name = format!("{}_canon_dot", inst.name);
        inst.operand_ids = vec![left_canon_id, right_canon_id];
        inst.dot_dimension_numbers = Some(canon_dnums);
        inst.shape = Some(dot_shape.clone());
        shape_map.insert(dot_id, dot_shape);
        out.push(inst);

        report.decomposed_dots = report.decomposed_dots.saturating_add(1);
        let needs_out_reshape =
            !orig_out_shape.dimensions.is_empty() && orig_out_shape.dimensions != dot_dims;
        let final_id = if needs_out_reshape {
            let reshape_id = alloc_id(&mut next_id);
            shape_map.insert(reshape_id, orig_out_shape.clone());
            out.push(HloInstructionProto {
                id: reshape_id,
                name: format!("dot_out_reshape_{reshape_id}"),
                opcode: String::from("reshape"),
                shape: Some(orig_out_shape),
                operand_ids: vec![dot_id],
                ..HloInstructionProto::default()
            });
            report.inserted_wrappers = report.inserted_wrappers.saturating_add(1);
            reshape_id
        } else {
            dot_id
        };
        remap.insert(orig_id, final_id);
        if comp.root_id == orig_id {
            comp.root_id = final_id;
        }
    }

    comp.instructions = out;
    Ok(())
}

/// Checks whether a `dot` operand pair is already in canonical rank-2 `[M, K] x [K, N]` or rank-3 `[B, M, K] x [B, K, N]` form.
///
/// # Arguments
///
/// * `dnums` - Dot dimension numbers.
/// * `lhs` - LHS operand shape.
/// * `rhs` - RHS operand shape.
fn is_canonical_dot(dnums: &DotDimensionNumbers, lhs: &ShapeProto, rhs: &ShapeProto) -> bool {
    if dnums.lhs_batch_dimensions.is_empty() {
        let ranks_ok = lhs.dimensions.len() == 2 && rhs.dimensions.len() == 2;
        ranks_ok
            && dnums.lhs_contracting_dimensions == [1]
            && dnums.rhs_contracting_dimensions == [0]
    } else {
        let ranks_ok = lhs.dimensions.len() == 3 && rhs.dimensions.len() == 3;
        let batch_ok = dnums.lhs_batch_dimensions == [0] && dnums.rhs_batch_dimensions == [0];
        let contract_ok =
            dnums.lhs_contracting_dimensions == [2] && dnums.rhs_contracting_dimensions == [1];
        ranks_ok && batch_ok && contract_ok
    }
}

/// Validates that all batch and contracting dimension indices in `dnums` lie within operand ranks.
///
/// # Arguments
///
/// * `dnums` - Dot dimension numbers to check.
/// * `lhs_rank` - Rank of LHS operand.
/// * `rhs_rank` - Rank of RHS operand.
fn validate_dot_dims(
    dnums: &DotDimensionNumbers,
    lhs_rank: usize,
    rhs_rank: usize,
) -> LibTpuResult<()> {
    let lr = i64::try_from(lhs_rank).unwrap_or(0);
    let rr = i64::try_from(rhs_rank).unwrap_or(0);
    let lhs_invalid = dnums
        .lhs_batch_dimensions
        .iter()
        .chain(dnums.lhs_contracting_dimensions.iter())
        .any(|&d| !(0..lr).contains(&d));
    let rhs_invalid = dnums
        .rhs_batch_dimensions
        .iter()
        .chain(dnums.rhs_contracting_dimensions.iter())
        .any(|&d| !(0..rr).contains(&d));
    if lhs_invalid || rhs_invalid {
        return Err(LibTpuError::CompilationFailed {
            reason: format!("Out-of-bounds DotDimensionNumbers for ranks ({lhs_rank}, {rhs_rank})"),
        });
    }
    Ok(())
}

/// Permutes and collapses `op_id` into `[B, Free, Contract]` (LHS) or `[B, Contract, Free]` (RHS).
///
/// # Arguments
///
/// * `op_id` - Input operand instruction ID.
/// * `shape` - Input operand shape.
/// * `batch_dims` - Batch dimension indices.
/// * `contract_dims` - Contracting dimension indices.
/// * `is_lhs` - True for LHS (`[B, M, K]`), false for RHS (`[B, K, N]`).
/// * `next_id` - Instruction ID counter.
/// * `out` - Output instruction buffer.
fn canonicalize_dot_operand(
    op_id: i64,
    shape: &ShapeProto,
    batch_dims: &[i64],
    contract_dims: &[i64],
    is_lhs: bool,
    next_id: &mut i64,
    out: &mut Vec<HloInstructionProto>,
) -> (i64, i64, i64, i64, usize) {
    let rank = shape.dimensions.len();
    let free_dims: Vec<i64> = (0..rank)
        .filter_map(|d| {
            let di = i64::try_from(d).unwrap_or(0);
            (!batch_dims.contains(&di) && !contract_dims.contains(&di)).then_some(di)
        })
        .collect();

    let mut perm = Vec::with_capacity(rank);
    perm.extend_from_slice(batch_dims);
    if is_lhs {
        perm.extend_from_slice(&free_dims);
        perm.extend_from_slice(contract_dims);
    } else {
        perm.extend_from_slice(contract_dims);
        perm.extend_from_slice(&free_dims);
    }

    let dim_product = |indices: &[i64]| -> i64 {
        indices.iter().fold(1_i64, |acc, &idx| {
            let ext = usize::try_from(idx)
                .ok()
                .and_then(|u| shape.dimensions.get(u).copied())
                .unwrap_or(1)
                .max(1);
            acc.saturating_mul(ext)
        })
    };
    let batch_ext = dim_product(batch_dims);
    let free_ext = dim_product(&free_dims);
    let contract_ext = dim_product(contract_dims);

    let mut cur_id = op_id;
    let mut wraps = 0_usize;
    let is_identity_perm = perm
        .iter()
        .enumerate()
        .all(|(i, &p)| usize::try_from(p).ok() == Some(i));
    if !is_identity_perm {
        let tr_dims: Vec<i64> = perm
            .iter()
            .map(|&p| {
                usize::try_from(p)
                    .ok()
                    .and_then(|u| shape.dimensions.get(u).copied())
                    .unwrap_or(1)
            })
            .collect();
        let tr_id = alloc_id(next_id);
        out.push(HloInstructionProto {
            id: tr_id,
            name: format!("dot_op_transpose_{tr_id}"),
            opcode: String::from("transpose"),
            shape: Some(make_tensor_shape(shape.element_type, tr_dims)),
            operand_ids: vec![cur_id],
            dimensions: perm,
            ..HloInstructionProto::default()
        });
        cur_id = tr_id;
        wraps = wraps.saturating_add(1);
    }

    let target_dims = if batch_dims.is_empty() {
        if is_lhs {
            vec![free_ext, contract_ext]
        } else {
            vec![contract_ext, free_ext]
        }
    } else if is_lhs {
        vec![batch_ext, free_ext, contract_ext]
    } else {
        vec![batch_ext, contract_ext, free_ext]
    };

    let rs_id = alloc_id(next_id);
    out.push(HloInstructionProto {
        id: rs_id,
        name: format!("dot_op_reshape_{rs_id}"),
        opcode: String::from("reshape"),
        shape: Some(make_tensor_shape(shape.element_type, target_dims)),
        operand_ids: vec![cur_id],
        ..HloInstructionProto::default()
    });
    wraps = wraps.saturating_add(1);
    (rs_id, batch_ext, free_ext, contract_ext, wraps)
}

/// Summary report emitted by [`ConvolutionLegalizerPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConvolutionLegalizerReport {
    /// Number of `convolution` instructions lowered into `pad` + `im2col` + MXU `dot`.
    pub legalized_convolutions: usize,
    /// Number of spatial `pad` ops inserted before `im2col`.
    pub inserted_pads: usize,
    /// Number of spatial halo `collective-permute` ops inserted for sharded convolutions.
    pub inserted_halo_exchanges: usize,
}

/// Lowers general `convolution` (`conv_general_dilated`) ops into MXU `dot` contractions.
///
/// Synthesizes VMEM `pad`, optional spatial halo `collective-permute`, and `im2col` `reshape`s.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvolutionLegalizerPass {
    /// Number of spatial SPMD partitions used to determine whether halo exchange is synthesized.
    pub num_partitions: usize,
}

impl ConvolutionLegalizerPass {
    /// Constructs a [`ConvolutionLegalizerPass`] for `num_partitions` devices.
    ///
    /// # Arguments
    ///
    /// * `num_partitions` - Number of SPMD partitions (`> 1` enables spatial halo exchange when sharded).
    #[must_use]
    pub const fn new(num_partitions: usize) -> Self {
        Self { num_partitions }
    }

    /// Lowers `convolution` instructions across all computations in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to legalize in-place.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if a `convolution` has fewer than 2 operands.
    pub fn run(&self, module: &mut HloModuleProto) -> LibTpuResult<ConvolutionLegalizerReport> {
        let mut report = ConvolutionLegalizerReport::default();
        for comp in &mut module.computations {
            self.legalize_in_computation(comp, &mut report)?;
        }
        Ok(report)
    }

    /// Lowers `convolution` ops within a single [`HloComputationProto`].
    ///
    /// # Arguments
    ///
    /// * `comp` - Mutable computation to transform.
    /// * `report` - Summary counter accumulator.
    #[allow(clippy::too_many_lines)]
    fn legalize_in_computation(
        self,
        comp: &mut HloComputationProto,
        report: &mut ConvolutionLegalizerReport,
    ) -> LibTpuResult<()> {
        let mut next_id = max_instruction_id(comp).saturating_add(1);
        let shape_map: BTreeMap<i64, ShapeProto> = comp
            .instructions
            .iter()
            .filter_map(|i| i.shape.as_ref().map(|s| (i.id, s.clone())))
            .collect();
        let original = std::mem::take(&mut comp.instructions);
        let mut out = Vec::with_capacity(original.len() * 3);
        let mut remap: BTreeMap<i64, i64> = BTreeMap::new();

        for mut inst in original {
            rewrite_operands(&mut inst, &remap);
            if inst.opcode != "convolution" {
                out.push(inst);
                continue;
            }
            if inst.operand_ids.len() < 2 {
                return Err(LibTpuError::CompilationFailed {
                    reason: format!("Convolution '{}' requires 2 operands", inst.name),
                });
            }
            let lhs_id = inst.operand_ids.first().copied().unwrap_or(0);
            let rhs_id = inst.operand_ids.get(1).copied().unwrap_or(0);
            let lhs_shape = shape_map.get(&lhs_id).cloned().unwrap_or_default();
            let rhs_shape = shape_map.get(&rhs_id).cloned().unwrap_or_default();
            let out_shape = inst.shape.clone().unwrap_or_else(|| lhs_shape.clone());
            let elem_ty = out_shape.element_type;

            let cdims = inst
                .convolution_dimension_numbers
                .clone()
                .unwrap_or_else(|| default_conv_dnums(lhs_shape.dimensions.len()));
            let mut cur_lhs_id = lhs_id;
            let mut cur_lhs_dims = lhs_shape.dimensions;

            let spatially_sharded = self.num_partitions > 1
                && inst.sharding.as_ref().is_some_and(|sh| {
                    cdims.input_spatial_dimensions.iter().any(|&sd| {
                        usize::try_from(sd)
                            .ok()
                            .and_then(|u| sh.tile_assignment_dimensions.get(u).copied())
                            .unwrap_or(1)
                            > 1
                    })
                });
            if spatially_sharded {
                let n = i64::try_from(self.num_partitions).unwrap_or(1);
                let halo_id = alloc_id(&mut next_id);
                out.push(HloInstructionProto {
                    id: halo_id,
                    name: format!("{}_halo_exchange", inst.name),
                    opcode: String::from("collective-permute"),
                    shape: Some(make_tensor_shape(elem_ty, cur_lhs_dims.clone())),
                    operand_ids: vec![cur_lhs_id],
                    source_target_pairs: (0..n)
                        .map(|s| SourceTarget {
                            source: s,
                            target: (s + 1).rem_euclid(n),
                        })
                        .collect(),
                    channel_id: halo_id,
                    ..HloInstructionProto::default()
                });
                cur_lhs_id = halo_id;
                report.inserted_halo_exchanges = report.inserted_halo_exchanges.saturating_add(1);
            }

            if let Some(win) = inst.window.as_ref().filter(|w| {
                w.dimensions.iter().any(|wd| {
                    (wd.padding_low.abs() + wd.padding_high.abs()) > 0 || wd.base_dilation > 1
                })
            }) {
                let zero_id = alloc_id(&mut next_id);
                out.push(HloInstructionProto {
                    id: zero_id,
                    name: format!("{}_pad_zero", inst.name),
                    opcode: String::from("constant"),
                    shape: Some(make_tensor_shape(elem_ty, Vec::new())),
                    literal: Some(LiteralProto {
                        shape: Some(make_tensor_shape(elem_ty, Vec::new())),
                        f32s: vec![0.0],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                });
                let mut pad_dims = vec![
                    PaddingConfigDimension {
                        edge_padding_low: 0,
                        edge_padding_high: 0,
                        interior_padding: 0,
                    };
                    cur_lhs_dims.len()
                ];
                for (w_idx, &sp_dim) in cdims.input_spatial_dimensions.iter().enumerate() {
                    let u_opt = usize::try_from(sp_dim).ok();
                    if let (Some(u_dim), Some(wd)) = (u_opt, win.dimensions.get(w_idx)) {
                        let interior = (wd.base_dilation - 1).max(0);
                        pad_dims.get_mut(u_dim).into_iter().for_each(|pd| {
                            pd.edge_padding_low = wd.padding_low;
                            pd.edge_padding_high = wd.padding_high;
                            pd.interior_padding = interior;
                        });
                        cur_lhs_dims.get_mut(u_dim).into_iter().for_each(|d_mut| {
                            let gaps = (*d_mut - 1).max(0).saturating_mul(interior);
                            *d_mut = (*d_mut)
                                .saturating_add(wd.padding_low)
                                .saturating_add(wd.padding_high)
                                .saturating_add(gaps);
                        });
                    }
                }
                let pad_id = alloc_id(&mut next_id);
                out.push(HloInstructionProto {
                    id: pad_id,
                    name: format!("{}_vmem_pad", inst.name),
                    opcode: String::from("pad"),
                    shape: Some(make_tensor_shape(elem_ty, cur_lhs_dims.clone())),
                    operand_ids: vec![cur_lhs_id, zero_id],
                    padding_config: Some(PaddingConfig {
                        dimensions: pad_dims,
                    }),
                    ..HloInstructionProto::default()
                });
                cur_lhs_id = pad_id;
                report.inserted_pads = report.inserted_pads.saturating_add(1);
            }

            let cout_dim = usize::try_from(cdims.output_feature_dimension).unwrap_or(0);
            let n_cols = out_shape
                .dimensions
                .get(cout_dim)
                .copied()
                .unwrap_or(1)
                .max(1);
            let total_out_elems = out_shape
                .dimensions
                .iter()
                .fold(1_i64, |acc, &d| acc.saturating_mul(d.max(1)));
            let m_rows = (total_out_elems / n_cols).max(1);
            let total_rhs_elems = rhs_shape
                .dimensions
                .iter()
                .fold(1_i64, |acc, &d| acc.saturating_mul(d.max(1)));
            let k_contract = (total_rhs_elems / n_cols).max(1);

            let im2col_id = alloc_id(&mut next_id);
            out.push(HloInstructionProto {
                id: im2col_id,
                name: format!("{}_im2col", inst.name),
                opcode: String::from("reshape"),
                shape: Some(make_tensor_shape(elem_ty, vec![m_rows, k_contract])),
                operand_ids: vec![cur_lhs_id],
                ..HloInstructionProto::default()
            });

            let kern_id = alloc_id(&mut next_id);
            out.push(HloInstructionProto {
                id: kern_id,
                name: format!("{}_kernel_2d", inst.name),
                opcode: String::from("reshape"),
                shape: Some(make_tensor_shape(elem_ty, vec![k_contract, n_cols])),
                operand_ids: vec![rhs_id],
                ..HloInstructionProto::default()
            });

            let dot_id = alloc_id(&mut next_id);
            out.push(HloInstructionProto {
                id: dot_id,
                name: format!("{}_mxu_dot", inst.name),
                opcode: String::from("dot"),
                shape: Some(make_tensor_shape(elem_ty, vec![m_rows, n_cols])),
                operand_ids: vec![im2col_id, kern_id],
                dot_dimension_numbers: Some(DotDimensionNumbers {
                    lhs_contracting_dimensions: vec![1],
                    rhs_contracting_dimensions: vec![0],
                    lhs_batch_dimensions: Vec::new(),
                    rhs_batch_dimensions: Vec::new(),
                }),
                precision_config: inst.precision_config.clone(),
                ..HloInstructionProto::default()
            });

            let reshape_out_id = alloc_id(&mut next_id);
            let orig_id = inst.id;
            out.push(HloInstructionProto {
                id: reshape_out_id,
                name: format!("{}_out_reshape", inst.name),
                opcode: String::from("reshape"),
                shape: Some(out_shape),
                operand_ids: vec![dot_id],
                ..HloInstructionProto::default()
            });

            remap.insert(orig_id, reshape_out_id);
            if comp.root_id == orig_id {
                comp.root_id = reshape_out_id;
            }
            report.legalized_convolutions = report.legalized_convolutions.saturating_add(1);
        }

        comp.instructions = out;
        Ok(())
    }
}

/// Constructs NHWC-style default [`ConvolutionDimensionNumbers`] for a given tensor `rank`.
///
/// # Arguments
///
/// * `rank` - Rank of the convolution input tensor.
fn default_conv_dnums(rank: usize) -> ConvolutionDimensionNumbers {
    let sp_count = rank.saturating_sub(2);
    let spatial: Vec<i64> = (1..=sp_count)
        .map(|i| i64::try_from(i).unwrap_or(1))
        .collect();
    let feat = i64::try_from(rank.saturating_sub(1)).unwrap_or(1);
    ConvolutionDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: feat,
        input_spatial_dimensions: spatial.clone(),
        kernel_input_feature_dimension: i64::try_from(sp_count).unwrap_or(0),
        kernel_output_feature_dimension: feat,
        kernel_spatial_dimensions: (0..sp_count)
            .map(|i| i64::try_from(i).unwrap_or(0))
            .collect(),
        output_batch_dimension: 0,
        output_feature_dimension: feat,
        output_spatial_dimensions: spatial,
    }
}

/// Algebraic identity, strength-reduction, and inverse-cancellation pass (`AlgebraicSimplifierPass`)
/// parameterized by `xla.AlgebraicSimplifierOptions`.
#[derive(Debug, Clone, Default)]
pub struct AlgebraicSimplifierPass {
    /// Upstream `xla.AlgebraicSimplifierOptions` flag struct.
    pub options: AlgebraicSimplifierOptionsProto,
}

impl AlgebraicSimplifierPass {
    /// Constructs a new [`AlgebraicSimplifierPass`] with the given options.
    ///
    /// # Arguments
    ///
    /// * `options` - Protobuf options controlling rewrite rules.
    #[must_use]
    pub const fn new(options: AlgebraicSimplifierOptionsProto) -> Self {
        Self { options }
    }

    /// Simplifies algebraic identities across all computations in `module`, returning rewrite count.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to simplify in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut total_rewrites = 0_usize;
        let max_iters = if self.options.run_to_fixed_point.unwrap_or(true) {
            6
        } else {
            1
        };
        for comp in &mut module.computations {
            for _ in 0..max_iters {
                let step = self.simplify_computation_once(comp);
                total_rewrites = total_rewrites.saturating_add(step);
                if step == 0 {
                    break;
                }
            }
        }
        total_rewrites
    }

    /// Performs a single algebraic simplification sweep over `comp`.
    ///
    /// # Arguments
    ///
    /// * `comp` - Mutable computation to rewrite.
    #[allow(clippy::too_many_lines)]
    fn simplify_computation_once(&self, comp: &mut HloComputationProto) -> usize {
        let mut rewrites = 0_usize;
        let inst_snapshot: BTreeMap<i64, HloInstructionProto> = comp
            .instructions
            .iter()
            .map(|i| (i.id, i.clone()))
            .collect();
        let mut alias_map: BTreeMap<i64, i64> = BTreeMap::new();

        for inst in &mut comp.instructions {
            rewrite_operands(inst, &alias_map);
            let op0 = inst.operand_ids.first().copied();
            let op1 = inst.operand_ids.get(1).copied();
            let p0 = op0.and_then(|id| inst_snapshot.get(&id));
            let p1 = op1.and_then(|id| inst_snapshot.get(&id));

            let replacement_target: Option<i64> = match inst.opcode.as_str() {
                "add" => {
                    if p1.is_some_and(|p| is_scalar_const(p, 0.0)) {
                        op0
                    } else if p0.is_some_and(|p| is_scalar_const(p, 0.0)) {
                        op1
                    } else {
                        None
                    }
                }
                "subtract" => p1.filter(|p| is_scalar_const(p, 0.0)).and(op0),
                "multiply" => {
                    if p1.is_some_and(|p| is_scalar_const(p, 1.0))
                        || p0.is_some_and(|p| is_scalar_const(p, 0.0))
                    {
                        op0
                    } else if p0.is_some_and(|p| is_scalar_const(p, 1.0))
                        || p1.is_some_and(|p| is_scalar_const(p, 0.0))
                    {
                        op1
                    } else {
                        None
                    }
                }
                "divide" => p1.filter(|p| is_scalar_const(p, 1.0)).and(op0),
                "negate" => p0
                    .filter(|p| p.opcode == "negate")
                    .and_then(|p| p.operand_ids.first().copied()),
                "exponential" => p0
                    .filter(|p| p.opcode == "log")
                    .and_then(|p| p.operand_ids.first().copied()),
                "log" => p0
                    .filter(|p| p.opcode == "exponential")
                    .and_then(|p| p.operand_ids.first().copied()),
                "reshape" | "convert" | "bitcast-convert" | "slice" => p0
                    .filter(|p| same_type_and_dims(p.shape.as_ref(), inst.shape.as_ref()))
                    .and(op0),
                _ => None,
            };

            if let Some(rep_id) = replacement_target {
                alias_map.insert(inst.id, rep_id);
                rewrites = rewrites.saturating_add(1);
                continue;
            }

            if inst.opcode == "subtract" && op0.is_some() && op0 == op1 {
                let shape = inst.shape.clone().unwrap_or_default();
                let count = shape_elem_count(&shape);
                inst.opcode = String::from("constant");
                inst.operand_ids.clear();
                inst.literal = Some(LiteralProto {
                    shape: Some(shape),
                    f32s: vec![0.0; count],
                    ..LiteralProto::default()
                });
                rewrites = rewrites.saturating_add(1);
                continue;
            }

            if inst.opcode == "reshape"
                && let Some(inner_src) = p0
                    .filter(|p| p.opcode == "reshape")
                    .and_then(|p| p.operand_ids.first().copied())
            {
                inst.operand_ids = vec![inner_src];
                rewrites = rewrites.saturating_add(1);
                continue;
            }

            let dot_rewrite_enabled = self.options.enable_dot_to_multiply_rewrite.unwrap_or(true);
            let scalar_dot = inst.opcode == "dot"
                && dot_rewrite_enabled
                && p0.is_some_and(|p| p.shape.as_ref().is_some_and(|s| shape_elem_count(s) == 1))
                && p1.is_some_and(|p| p.shape.as_ref().is_some_and(|s| shape_elem_count(s) == 1));
            if scalar_dot {
                inst.opcode = String::from("multiply");
                inst.dot_dimension_numbers = None;
                rewrites = rewrites.saturating_add(1);
            }
        }

        if !alias_map.is_empty() {
            apply_alias_map(comp, &alias_map);
        }
        rewrites
    }
}

/// Evaluates pure unary and binary operations on compile-time `constant` `LiteralProto`s (`ConstantFoldingPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    /// Folds constant expressions across all computations in `module`, returning folded op count.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to constant-fold in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut folded = 0_usize;
        for comp in &mut module.computations {
            folded = folded.saturating_add(fold_constants_in_computation(comp));
        }
        folded
    }
}

/// Evaluates constant expressions inside `comp` in topological order.
///
/// # Arguments
///
/// * `comp` - Mutable computation to constant-fold.
fn fold_constants_in_computation(comp: &mut HloComputationProto) -> usize {
    let mut folded_count = 0_usize;
    let mut float_literals: BTreeMap<i64, Vec<f32>> = BTreeMap::new();
    let mut int_literals: BTreeMap<i64, Vec<i32>> = BTreeMap::new();

    for inst in &mut comp.instructions {
        if inst.opcode == "constant" {
            inst.literal.as_ref().into_iter().for_each(|lit| {
                if !lit.f32s.is_empty() {
                    float_literals.insert(inst.id, lit.f32s.clone());
                } else if !lit.s32s.is_empty() {
                    int_literals.insert(inst.id, lit.s32s.clone());
                }
            });
            continue;
        }

        if inst.operand_ids.len() == 1 {
            let unary_res = inst
                .operand_ids
                .first()
                .and_then(|op0| float_literals.get(op0))
                .and_then(|vals| eval_unary_f32(&inst.opcode, vals));
            if let Some(res) = unary_res {
                inst.opcode = String::from("constant");
                inst.operand_ids.clear();
                inst.literal = Some(LiteralProto {
                    shape: inst.shape.clone(),
                    f32s: res.clone(),
                    ..LiteralProto::default()
                });
                float_literals.insert(inst.id, res);
                folded_count = folded_count.saturating_add(1);
                continue;
            }
        }

        if inst.operand_ids.len() == 2 {
            let op0 = inst.operand_ids.first().copied().unwrap_or(0);
            let op1 = inst.operand_ids.get(1).copied().unwrap_or(0);
            let f_pair = float_literals
                .get(&op0)
                .zip(float_literals.get(&op1))
                .filter(|(a, b)| a.len() == b.len())
                .and_then(|(a, b)| eval_binary_f32(&inst.opcode, a, b));
            let i_pair = int_literals
                .get(&op0)
                .zip(int_literals.get(&op1))
                .filter(|(a, b)| a.len() == b.len())
                .and_then(|(a, b)| eval_binary_s32(&inst.opcode, a, b));

            if let Some(res) = f_pair {
                inst.opcode = String::from("constant");
                inst.operand_ids.clear();
                inst.literal = Some(LiteralProto {
                    shape: inst.shape.clone(),
                    f32s: res.clone(),
                    ..LiteralProto::default()
                });
                float_literals.insert(inst.id, res);
                folded_count = folded_count.saturating_add(1);
            } else if let Some(res) = i_pair {
                inst.opcode = String::from("constant");
                inst.operand_ids.clear();
                inst.literal = Some(LiteralProto {
                    shape: inst.shape.clone(),
                    s32s: res.clone(),
                    ..LiteralProto::default()
                });
                int_literals.insert(inst.id, res);
                folded_count = folded_count.saturating_add(1);
            }
        }
    }

    folded_count
}

/// Evaluates a supported `f32` unary opcode elementwise.
///
/// # Arguments
///
/// * `opcode` - HLO unary opcode string.
/// * `vals` - Input `f32` slice.
fn eval_unary_f32(opcode: &str, vals: &[f32]) -> Option<Vec<f32>> {
    match opcode {
        "negate" => Some(vals.iter().map(|&v| -v).collect()),
        "abs" => Some(vals.iter().map(|&v| v.abs()).collect()),
        "exponential" => Some(vals.iter().map(|&v| v.exp()).collect()),
        "log" => Some(vals.iter().map(|&v| v.ln()).collect()),
        "tanh" => Some(vals.iter().map(|&v| v.tanh()).collect()),
        "sqrt" => Some(vals.iter().map(|&v| v.sqrt()).collect()),
        "floor" => Some(vals.iter().map(|&v| v.floor()).collect()),
        "ceil" => Some(vals.iter().map(|&v| v.ceil()).collect()),
        _ => None,
    }
}

/// Evaluates a supported `f32` binary opcode elementwise.
///
/// # Arguments
///
/// * `opcode` - HLO binary opcode string.
/// * `lhs` - Left `f32` operand values.
/// * `rhs` - Right `f32` operand values.
fn eval_binary_f32(opcode: &str, lhs: &[f32], rhs: &[f32]) -> Option<Vec<f32>> {
    match opcode {
        "add" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a + b).collect()),
        "subtract" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a - b).collect()),
        "multiply" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a * b).collect()),
        "divide" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a / b).collect()),
        "maximum" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a.max(b)).collect()),
        "minimum" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a.min(b)).collect()),
        _ => None,
    }
}

/// Evaluates a supported `s32` binary opcode elementwise using wrapping two's-complement arithmetic.
///
/// # Arguments
///
/// * `opcode` - HLO binary opcode string.
/// * `lhs` - Left `i32` operand values.
/// * `rhs` - Right `i32` operand values.
fn eval_binary_s32(opcode: &str, lhs: &[i32], rhs: &[i32]) -> Option<Vec<i32>> {
    match opcode {
        "add" => Some(
            lhs.iter()
                .zip(rhs)
                .map(|(&a, &b)| a.wrapping_add(b))
                .collect(),
        ),
        "subtract" => Some(
            lhs.iter()
                .zip(rhs)
                .map(|(&a, &b)| a.wrapping_sub(b))
                .collect(),
        ),
        "multiply" => Some(
            lhs.iter()
                .zip(rhs)
                .map(|(&a, &b)| a.wrapping_mul(b))
                .collect(),
        ),
        "maximum" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a.max(b)).collect()),
        "minimum" => Some(lhs.iter().zip(rhs).map(|(&a, &b)| a.min(b)).collect()),
        _ => None,
    }
}

/// Simplifies nested and identity `broadcast` instructions (`BroadcastSimplifierPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct BroadcastSimplifierPass;

impl BroadcastSimplifierPass {
    /// Simplifies `broadcast` chains across all computations in `module`, returning rewrite count.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to simplify in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut count = 0_usize;
        for comp in &mut module.computations {
            let snap: BTreeMap<i64, HloInstructionProto> = comp
                .instructions
                .iter()
                .map(|i| (i.id, i.clone()))
                .collect();
            let mut aliases = BTreeMap::new();
            for inst in &mut comp.instructions {
                rewrite_operands(inst, &aliases);
                if inst.opcode != "broadcast" {
                    continue;
                }
                let Some(op0) = inst.operand_ids.first().copied() else {
                    continue;
                };
                let Some(prod) = snap.get(&op0) else {
                    continue;
                };
                let expected_ident: Vec<i64> =
                    (0..inst.shape.as_ref().map_or(0, |s| s.dimensions.len()))
                        .map(|i| i64::try_from(i).unwrap_or(0))
                        .collect();
                let ident_dims = inst.dimensions.is_empty() || inst.dimensions == expected_ident;
                if same_type_and_dims(prod.shape.as_ref(), inst.shape.as_ref()) && ident_dims {
                    aliases.insert(inst.id, op0);
                    count = count.saturating_add(1);
                    continue;
                }
                if let Some(inner_src) = prod
                    .operand_ids
                    .first()
                    .copied()
                    .filter(|_| prod.opcode == "broadcast")
                {
                    let composed: Vec<i64> = prod
                        .dimensions
                        .iter()
                        .filter_map(|&d1| {
                            usize::try_from(d1)
                                .ok()
                                .and_then(|u| inst.dimensions.get(u).copied())
                        })
                        .collect();
                    if composed.len() == prod.dimensions.len() {
                        inst.operand_ids = vec![inner_src];
                        inst.dimensions = composed;
                        count = count.saturating_add(1);
                    }
                }
            }
            if !aliases.is_empty() {
                apply_alias_map(comp, &aliases);
            }
        }
        count
    }
}

/// Moves elementwise operations across redundant `reshape` boundaries (`ReshapeMoverPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ReshapeMoverPass;

impl ReshapeMoverPass {
    /// Rewrites `op(reshape(a), reshape(b)) -> reshape(op(a, b))` when `a` and `b` share identical shapes.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to transform in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut moved = 0_usize;
        for comp in &mut module.computations {
            moved = moved.saturating_add(move_reshapes_in_computation(comp));
        }
        moved
    }
}

/// Sinks `reshape` past elementwise ops inside `comp`.
///
/// # Arguments
///
/// * `comp` - Mutable computation to transform.
fn move_reshapes_in_computation(comp: &mut HloComputationProto) -> usize {
    let mut moved = 0_usize;
    let mut next_id = max_instruction_id(comp).saturating_add(1);
    let snap: BTreeMap<i64, HloInstructionProto> = comp
        .instructions
        .iter()
        .map(|i| (i.id, i.clone()))
        .collect();
    let original = std::mem::take(&mut comp.instructions);
    let mut out = Vec::with_capacity(original.len() * 2);
    let mut remap = BTreeMap::new();

    for mut inst in original {
        rewrite_operands(&mut inst, &remap);
        let is_elem = matches!(
            inst.opcode.as_str(),
            "add" | "subtract" | "multiply" | "negate" | "abs" | "tanh"
        );
        let all_reshapes = is_elem
            && !inst.operand_ids.is_empty()
            && inst.operand_ids.iter().all(|id| {
                snap.get(id)
                    .is_some_and(|p| p.opcode == "reshape" && !p.operand_ids.is_empty())
            });
        if !all_reshapes {
            out.push(inst);
            continue;
        }
        let pre_ids: Vec<i64> = inst
            .operand_ids
            .iter()
            .filter_map(|id| snap.get(id).and_then(|p| p.operand_ids.first().copied()))
            .collect();
        let pre_shape = pre_ids
            .first()
            .and_then(|id| snap.get(id))
            .and_then(|p| p.shape.clone())
            .unwrap_or_default();
        let same_pre_shapes = pre_ids.iter().all(|id| {
            snap.get(id)
                .is_some_and(|p| same_type_and_dims(p.shape.as_ref(), Some(&pre_shape)))
        });
        if !same_pre_shapes {
            out.push(inst);
            continue;
        }

        let orig_id = inst.id;
        let post_shape = inst.shape.clone();
        let elem_id = alloc_id(&mut next_id);
        inst.id = elem_id;
        inst.operand_ids = pre_ids;
        inst.shape = Some(pre_shape);
        out.push(inst);

        let rs_id = alloc_id(&mut next_id);
        out.push(HloInstructionProto {
            id: rs_id,
            name: format!("sunk_reshape_{rs_id}"),
            opcode: String::from("reshape"),
            shape: post_shape,
            operand_ids: vec![elem_id],
            ..HloInstructionProto::default()
        });
        remap.insert(orig_id, rs_id);
        if comp.root_id == orig_id {
            comp.root_id = rs_id;
        }
        moved = moved.saturating_add(1);
    }

    comp.instructions = out;
    moved
}

/// Composes consecutive transposes, removes identity transposes, and folds transposes into `dot` (`TransposeFoldingPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TransposeFoldingPass;

impl TransposeFoldingPass {
    /// Folds `transpose` instructions across all computations in `module`, returning fold count.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to transform in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut folded = 0_usize;
        for comp in &mut module.computations {
            folded = folded.saturating_add(fold_transposes_in_computation(comp));
        }
        folded
    }
}

/// Composes and folds `transpose` instructions in `comp`.
///
/// # Arguments
///
/// * `comp` - Mutable computation to transform.
fn fold_transposes_in_computation(comp: &mut HloComputationProto) -> usize {
    let mut folded = 0_usize;
    let mut aliases: BTreeMap<i64, i64> = BTreeMap::new();
    let mut seen: BTreeMap<i64, HloInstructionProto> = BTreeMap::new();

    for inst in &mut comp.instructions {
        rewrite_operands(inst, &aliases);
        if inst.opcode == "transpose" {
            if let Some(prev) = inst
                .operand_ids
                .first()
                .and_then(|op0| seen.get(op0))
                .filter(|p| p.opcode == "transpose")
                && let Some(&inner_src) = prev.operand_ids.first()
            {
                let composed: Vec<i64> = inst
                    .dimensions
                    .iter()
                    .filter_map(|&d| {
                        usize::try_from(d)
                            .ok()
                            .and_then(|u| prev.dimensions.get(u).copied())
                    })
                    .collect();
                if composed.len() == inst.dimensions.len() {
                    inst.operand_ids = vec![inner_src];
                    inst.dimensions = composed;
                    folded = folded.saturating_add(1);
                }
            }
            let is_ident = inst
                .dimensions
                .iter()
                .enumerate()
                .all(|(idx, &d)| usize::try_from(d).ok() == Some(idx));
            if is_ident && let Some(&src) = inst.operand_ids.first() {
                aliases.insert(inst.id, src);
                folded = folded.saturating_add(1);
                continue;
            }
        } else if inst.opcode == "dot" && inst.operand_ids.len() == 2 {
            let lhs_id = inst.operand_ids.first().copied().unwrap_or(0);
            let rhs_id = inst.operand_ids.get(1).copied().unwrap_or(0);
            let mut dnums =
                inst.dot_dimension_numbers
                    .clone()
                    .unwrap_or_else(|| DotDimensionNumbers {
                        lhs_contracting_dimensions: vec![1],
                        rhs_contracting_dimensions: vec![0],
                        lhs_batch_dimensions: Vec::new(),
                        rhs_batch_dimensions: Vec::new(),
                    });
            let mut dot_changed = false;
            if let Some(lhs_tr) = seen.get(&lhs_id).filter(|p| p.opcode == "transpose") {
                let raw_lhs = lhs_tr.operand_ids.first().copied().unwrap_or(lhs_id);
                dnums.lhs_contracting_dimensions = remap_dims_through_transpose(
                    &dnums.lhs_contracting_dimensions,
                    &lhs_tr.dimensions,
                );
                dnums.lhs_batch_dimensions =
                    remap_dims_through_transpose(&dnums.lhs_batch_dimensions, &lhs_tr.dimensions);
                inst.operand_ids = vec![raw_lhs, rhs_id];
                dot_changed = true;
                folded = folded.saturating_add(1);
            }
            let cur_lhs = inst.operand_ids.first().copied().unwrap_or(lhs_id);
            if let Some(rhs_tr) = seen.get(&rhs_id).filter(|p| p.opcode == "transpose") {
                let raw_rhs = rhs_tr.operand_ids.first().copied().unwrap_or(rhs_id);
                dnums.rhs_contracting_dimensions = remap_dims_through_transpose(
                    &dnums.rhs_contracting_dimensions,
                    &rhs_tr.dimensions,
                );
                dnums.rhs_batch_dimensions =
                    remap_dims_through_transpose(&dnums.rhs_batch_dimensions, &rhs_tr.dimensions);
                inst.operand_ids = vec![cur_lhs, raw_rhs];
                dot_changed = true;
                folded = folded.saturating_add(1);
            }
            if dot_changed {
                inst.dot_dimension_numbers = Some(dnums);
            }
        }
        seen.insert(inst.id, inst.clone());
    }

    if !aliases.is_empty() {
        apply_alias_map(comp, &aliases);
    }
    folded
}

/// Maps post-transpose dimension indices back through `perm` to pre-transpose operand dimensions.
///
/// # Arguments
///
/// * `dims` - Post-transpose dimension indices.
/// * `perm` - Transpose `dimensions` permutation.
fn remap_dims_through_transpose(dims: &[i64], perm: &[i64]) -> Vec<i64> {
    dims.iter()
        .map(|&d| {
            usize::try_from(d)
                .ok()
                .and_then(|u| perm.get(u).copied())
                .unwrap_or(d)
        })
        .collect()
}

/// Simplifies `get-tuple-element(tuple(...), k)` and `tuple(gte(t, 0), ...)` patterns (`TupleSimplifierPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TupleSimplifierPass;

impl TupleSimplifierPass {
    /// Simplifies tuple pack/unpack indirection across all computations in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to simplify in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut simplified = 0_usize;
        for comp in &mut module.computations {
            let mut seen: BTreeMap<i64, HloInstructionProto> = BTreeMap::new();
            let mut aliases: BTreeMap<i64, i64> = BTreeMap::new();
            for inst in &mut comp.instructions {
                rewrite_operands(inst, &aliases);
                if inst.opcode == "get-tuple-element" {
                    let elem_opt = inst
                        .operand_ids
                        .first()
                        .and_then(|tup_id| seen.get(tup_id))
                        .filter(|p| p.opcode == "tuple")
                        .and_then(|tup_inst| {
                            usize::try_from(inst.tuple_index)
                                .ok()
                                .and_then(|idx| tup_inst.operand_ids.get(idx).copied())
                        });
                    if let Some(elem_id) = elem_opt {
                        aliases.insert(inst.id, elem_id);
                        simplified = simplified.saturating_add(1);
                        continue;
                    }
                }
                if inst.opcode == "tuple" && !inst.operand_ids.is_empty() {
                    let repacked_src =
                        inst.operand_ids
                            .first()
                            .and_then(|id| seen.get(id))
                            .filter(|g| g.opcode == "get-tuple-element")
                            .and_then(|g0| g0.operand_ids.first().copied())
                            .filter(|&src_tup| {
                                let arity_ok = seen.get(&src_tup).is_some_and(|s_inst| {
                                    s_inst.shape.as_ref().is_some_and(|s| {
                                        s.tuple_shapes.len() == inst.operand_ids.len()
                                    })
                                });
                                arity_ok
                                    && inst.operand_ids.iter().enumerate().all(|(k, op_id)| {
                                        seen.get(op_id).is_some_and(|g| {
                                            g.opcode == "get-tuple-element"
                                                && g.operand_ids.first().copied() == Some(src_tup)
                                                && usize::try_from(g.tuple_index).ok() == Some(k)
                                        })
                                    })
                            });
                    if let Some(src_tup) = repacked_src {
                        aliases.insert(inst.id, src_tup);
                        simplified = simplified.saturating_add(1);
                        continue;
                    }
                }
                seen.insert(inst.id, inst.clone());
            }
            if !aliases.is_empty() {
                apply_alias_map(comp, &aliases);
            }
        }
        simplified
    }
}

/// Hoists loop-invariant pure instructions out of `while` loop body computations (`WhileLoopInvariantCodeMotionPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct WhileLoopInvariantCodeMotionPass;

impl WhileLoopInvariantCodeMotionPass {
    /// Hoists loop-invariant instructions from `while` body computations into caller computations.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to optimize in-place.
    #[allow(clippy::too_many_lines)]
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut hoisted_total = 0_usize;

        for caller_pos in 0..module.computations.len() {
            let while_specs: Vec<(i64, i64, i64)> = module
                .computations
                .get(caller_pos)
                .map(|c| {
                    c.instructions
                        .iter()
                        .filter(|i| i.opcode == "while")
                        .filter_map(|i| {
                            let body_cid = i.called_computation_ids.first().copied();
                            let init_id = i.operand_ids.first().copied();
                            body_cid.zip(init_id).map(|(b, init)| (i.id, b, init))
                        })
                        .collect()
                })
                .unwrap_or_default();

            for (while_id, body_cid, init_tup_id) in while_specs {
                let Some(body_pos) = module
                    .computations
                    .iter()
                    .position(|c| c.id == body_cid)
                    .filter(|&bp| bp != caller_pos)
                else {
                    continue;
                };
                let body_comp = module
                    .computations
                    .get(body_pos)
                    .cloned()
                    .unwrap_or_default();
                let invariant_slots = detect_invariant_while_tuple_slots(&body_comp);
                let mut invariant_insts: BTreeSet<i64> = BTreeSet::new();
                let mut gte_slot_of_inst: BTreeMap<i64, i64> = BTreeMap::new();
                let mut hoist_candidates: Vec<HloInstructionProto> = Vec::new();

                for inst in &body_comp.instructions {
                    if inst.opcode == "parameter" {
                        continue;
                    }
                    if inst.opcode == "get-tuple-element"
                        && invariant_slots.contains(&inst.tuple_index)
                    {
                        invariant_insts.insert(inst.id);
                        gte_slot_of_inst.insert(inst.id, inst.tuple_index);
                        continue;
                    }
                    if inst.opcode == "constant" {
                        invariant_insts.insert(inst.id);
                        continue;
                    }
                    let hoistable = is_pure_hoistable_opcode(&inst.opcode)
                        && inst.id != body_comp.root_id
                        && !inst.operand_ids.is_empty()
                        && inst
                            .operand_ids
                            .iter()
                            .all(|op| invariant_insts.contains(op));
                    if hoistable {
                        invariant_insts.insert(inst.id);
                        hoist_candidates.push(inst.clone());
                    }
                }

                if hoist_candidates.is_empty() {
                    continue;
                }

                module
                    .computations
                    .get_mut(caller_pos)
                    .into_iter()
                    .for_each(|caller_comp| {
                        let mut next_caller_id = max_instruction_id(caller_comp).saturating_add(1);
                        let mut body_to_caller_id: BTreeMap<i64, i64> = BTreeMap::new();
                        let mut caller_inserts: Vec<HloInstructionProto> = Vec::new();

                        for (&body_gte_id, &slot_idx) in &gte_slot_of_inst {
                            let c_gte_id = alloc_id(&mut next_caller_id);
                            let gte_shape = body_comp
                                .instructions
                                .iter()
                                .find(|i| i.id == body_gte_id)
                                .and_then(|i| i.shape.clone());
                            caller_inserts.push(HloInstructionProto {
                                id: c_gte_id,
                                name: format!("licm_init_gte_{c_gte_id}"),
                                opcode: String::from("get-tuple-element"),
                                shape: gte_shape,
                                operand_ids: vec![init_tup_id],
                                tuple_index: slot_idx,
                                ..HloInstructionProto::default()
                            });
                            body_to_caller_id.insert(body_gte_id, c_gte_id);
                        }
                        for b_inst in body_comp
                            .instructions
                            .iter()
                            .filter(|i| i.opcode == "constant")
                        {
                            let c_const_id = alloc_id(&mut next_caller_id);
                            let mut cloned = b_inst.clone();
                            cloned.id = c_const_id;
                            cloned.name = format!("licm_const_{c_const_id}");
                            caller_inserts.push(cloned);
                            body_to_caller_id.insert(b_inst.id, c_const_id);
                        }
                        for cand in &hoist_candidates {
                            let new_id = alloc_id(&mut next_caller_id);
                            let mut hoisted = cand.clone();
                            hoisted.id = new_id;
                            hoisted.name = format!("licm_hoisted_{}", cand.name);
                            hoisted.operand_ids = cand
                                .operand_ids
                                .iter()
                                .map(|op| body_to_caller_id.get(op).copied().unwrap_or(*op))
                                .collect();
                            caller_inserts.push(hoisted);
                            body_to_caller_id.insert(cand.id, new_id);
                            hoisted_total = hoisted_total.saturating_add(1);
                        }

                        caller_comp
                            .instructions
                            .iter()
                            .position(|i| i.id == while_id)
                            .into_iter()
                            .for_each(|w_pos| {
                                caller_comp
                                    .instructions
                                    .splice(w_pos..w_pos, caller_inserts.clone());
                            });
                    });
            }
        }
        hoisted_total
    }
}

/// Detects tuple element indices in a `while` body computation whose root `tuple` passes `gte(param, k)` unchanged.
///
/// # Arguments
///
/// * `body` - Loop body [`HloComputationProto`].
fn detect_invariant_while_tuple_slots(body: &HloComputationProto) -> BTreeSet<i64> {
    let mut invariant = BTreeSet::new();
    let param_opt = body
        .instructions
        .iter()
        .find(|i| i.opcode == "parameter")
        .map(|i| i.id);
    let root_opt = body
        .instructions
        .iter()
        .find(|i| i.id == body.root_id && i.opcode == "tuple");
    let Some((param_id, root_inst)) = param_opt.zip(root_opt) else {
        return invariant;
    };
    let inst_map: BTreeMap<i64, &HloInstructionProto> =
        body.instructions.iter().map(|i| (i.id, i)).collect();
    for (slot, &op_id) in root_inst.operand_ids.iter().enumerate() {
        let slot_i64 = i64::try_from(slot).unwrap_or(-1);
        let passthrough = inst_map.get(&op_id).is_some_and(|g| {
            g.opcode == "get-tuple-element"
                && g.operand_ids.first().copied() == Some(param_id)
                && g.tuple_index == slot_i64
        });
        if passthrough {
            invariant.insert(slot_i64);
        }
    }
    invariant
}

/// Common Subexpression Elimination across [`HloModuleProto`] computations (`HloCsePass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct HloCsePass;

/// Canonical structural equivalence key for [`HloCsePass`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HloCseKey {
    /// Opcode mnemonic.
    opcode: String,
    /// Output element type discriminant.
    element_type: i32,
    /// Output tensor dimensions.
    shape_dims: Vec<i64>,
    /// Operand instruction IDs.
    operand_ids: Vec<i64>,
    /// Dimension attribute list (`transpose`, `broadcast`, `reduce`, etc.).
    dimensions: Vec<i64>,
    /// Tuple extraction index.
    tuple_index: i64,
    /// Encoded constant literal bits (`f32::to_bits` and `s32` values).
    literal_bits: Vec<u32>,
}

impl HloCsePass {
    /// Deduplicates pure common subexpressions across all computations in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to deduplicate in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut eliminated = 0_usize;
        for comp in &mut module.computations {
            let mut table: BTreeMap<HloCseKey, i64> = BTreeMap::new();
            let mut aliases: BTreeMap<i64, i64> = BTreeMap::new();
            for inst in &mut comp.instructions {
                rewrite_operands(inst, &aliases);
                if !is_pure_cse_candidate(inst) {
                    continue;
                }
                let key = build_cse_key(inst);
                if let Some(&canon_id) = table.get(&key) {
                    aliases.insert(inst.id, canon_id);
                    eliminated = eliminated.saturating_add(1);
                } else {
                    table.insert(key, inst.id);
                }
            }
            if !aliases.is_empty() {
                apply_alias_map(comp, &aliases);
            }
        }
        eliminated
    }
}

/// Builds a deterministic [`HloCseKey`] for a pure instruction.
///
/// # Arguments
///
/// * `inst` - Pure [`HloInstructionProto`] to fingerprint.
fn build_cse_key(inst: &HloInstructionProto) -> HloCseKey {
    let (element_type, shape_dims) = inst
        .shape
        .as_ref()
        .map_or((0, Vec::new()), |s| (s.element_type, s.dimensions.clone()));
    let mut literal_bits = Vec::new();
    if let Some(lit) = &inst.literal {
        literal_bits.extend(lit.f32s.iter().map(|v| v.to_bits()));
        #[allow(clippy::cast_sign_loss)]
        literal_bits.extend(lit.s32s.iter().map(|&v| v as u32));
    }
    HloCseKey {
        opcode: inst.opcode.clone(),
        element_type,
        shape_dims,
        operand_ids: inst.operand_ids.clone(),
        dimensions: inst.dimensions.clone(),
        tuple_index: inst.tuple_index,
        literal_bits,
    }
}

/// Dead Code Elimination across instructions and unreachable sub-computations (`HloDcePass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct HloDcePass;

impl HloDcePass {
    /// Eliminates unreachable instructions and dead non-entry computations in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to prune in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> usize {
        let mut removed = 0_usize;
        for comp in &mut module.computations {
            removed = removed.saturating_add(dce_computation(comp));
        }
        if let Some(entry_id) = module
            .computations
            .iter()
            .find(|c| c.id == module.entry_computation_id)
            .or_else(|| module.computations.first())
            .map(|c| c.id)
        {
            let mut live_comps = BTreeSet::new();
            let mut worklist = vec![entry_id];
            while let Some(cid) = worklist.pop() {
                if live_comps.insert(cid) {
                    module
                        .computations
                        .iter()
                        .find(|c| c.id == cid)
                        .into_iter()
                        .for_each(|comp| {
                            for inst in &comp.instructions {
                                worklist.extend(inst.called_computation_ids.iter().copied());
                            }
                        });
                }
            }
            let before = module.computations.len();
            module.computations.retain(|c| live_comps.contains(&c.id));
            removed = removed.saturating_add(before.saturating_sub(module.computations.len()));
        }
        removed
    }
}

/// Eliminates dead instructions inside `comp` not reachable from `comp.root_id` or side-effecting ops.
///
/// # Arguments
///
/// * `comp` - Mutable computation to prune.
fn dce_computation(comp: &mut HloComputationProto) -> usize {
    let id_map: BTreeMap<i64, &HloInstructionProto> =
        comp.instructions.iter().map(|i| (i.id, i)).collect();
    let mut live: BTreeSet<i64> = BTreeSet::new();
    let mut stack: Vec<i64> = Vec::new();
    let seed_root = id_map
        .get(&comp.root_id)
        .map(|i| i.id)
        .or_else(|| comp.instructions.last().map(|i| i.id));
    stack.extend(seed_root);

    for inst in &comp.instructions {
        let must_keep = inst.opcode == "parameter" || has_instruction_side_effects(inst);
        if must_keep {
            stack.push(inst.id);
        }
    }
    while let Some(id) = stack.pop() {
        if live.insert(id) {
            id_map.get(&id).into_iter().for_each(|inst| {
                stack.extend(inst.operand_ids.iter().copied());
                stack.extend(inst.control_predecessor_ids.iter().copied());
            });
        }
    }
    let before = comp.instructions.len();
    comp.instructions.retain(|i| live.contains(&i.id));
    before.saturating_sub(comp.instructions.len())
}

/// Summary report emitted by [`AsyncCollectiveFusionPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AsyncCollectiveReport {
    /// Number of synchronous collectives converted into asynchronous `*-start` / `*-done` pairs.
    pub converted_pairs: usize,
    /// Number of compatible adjacent collective `*-start` ops fused/coalesced.
    pub coalesced_collectives: usize,
}

/// Converts synchronous HLO collectives into asynchronous `*-start` / `*-done` pairs and coalesces
/// small compatible adjacent collectives (`AsyncCollectiveFusionPass`).
#[derive(Debug, Clone, Copy)]
pub struct AsyncCollectiveFusionPass {
    /// Maximum combined byte size for coalescing adjacent compatible collectives.
    pub coalesce_byte_limit: u64,
}

impl Default for AsyncCollectiveFusionPass {
    fn default() -> Self {
        Self {
            coalesce_byte_limit: 16 * 1024 * 1024,
        }
    }
}

impl AsyncCollectiveFusionPass {
    /// Constructs an [`AsyncCollectiveFusionPass`] with the specified `coalesce_byte_limit`.
    ///
    /// # Arguments
    ///
    /// * `coalesce_byte_limit` - Maximum combined payload bytes for coalescing compatible collectives.
    #[must_use]
    pub const fn new(coalesce_byte_limit: u64) -> Self {
        Self {
            coalesce_byte_limit,
        }
    }

    /// Converts synchronous collectives into `*-start` / `*-done` pairs across `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to transform in-place.
    pub fn run(&self, module: &mut HloModuleProto) -> AsyncCollectiveReport {
        let mut report = AsyncCollectiveReport::default();
        for comp in &mut module.computations {
            let mut next_id = max_instruction_id(comp).saturating_add(1);
            let original = std::mem::take(&mut comp.instructions);
            let mut out: Vec<HloInstructionProto> = Vec::with_capacity(original.len() * 2);
            let mut remap: BTreeMap<i64, i64> = BTreeMap::new();
            let mut prev_start_idx: Option<usize> = None;

            for mut inst in original {
                rewrite_operands(&mut inst, &remap);
                let Some((start_op, done_op)) = async_collective_pair_opcodes(&inst.opcode) else {
                    prev_start_idx = None;
                    out.push(inst);
                    continue;
                };

                let inst_bytes = inst.shape.as_ref().map_or(0_u64, estimate_shape_bytes);
                let coalescable =
                    prev_start_idx
                        .and_then(|p_idx| out.get(p_idx))
                        .is_some_and(|prev_inst| {
                            prev_inst.opcode == start_op
                                && prev_inst.replica_groups == inst.replica_groups
                                && prev_inst
                                    .shape
                                    .as_ref()
                                    .map_or(0, estimate_shape_bytes)
                                    .saturating_add(inst_bytes)
                                    <= self.coalesce_byte_limit
                        });
                if coalescable {
                    report.coalesced_collectives = report.coalesced_collectives.saturating_add(1);
                }

                let orig_id = inst.id;
                let res_shape = inst.shape.clone();
                let start_id = alloc_id(&mut next_id);
                let done_id = alloc_id(&mut next_id);

                inst.id = start_id;
                inst.name = format!("{}_start", inst.name);
                inst.opcode = String::from(start_op);
                inst.async_execution_thread = String::from("ici_async");
                let inserted_start_pos = out.len();
                out.push(inst);
                prev_start_idx = Some(inserted_start_pos);

                out.push(HloInstructionProto {
                    id: done_id,
                    name: format!("collective_done_{done_id}"),
                    opcode: String::from(done_op),
                    shape: res_shape,
                    operand_ids: vec![start_id],
                    async_execution_thread: String::from("ici_async"),
                    ..HloInstructionProto::default()
                });
                remap.insert(orig_id, done_id);
                if comp.root_id == orig_id {
                    comp.root_id = done_id;
                }
                report.converted_pairs = report.converted_pairs.saturating_add(1);
            }
            comp.instructions = out;
        }
        report
    }
}

/// Summary report emitted by [`LatencyHidingSchedulerPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LatencyHidingScheduleReport {
    /// Number of independent MXU/VPU compute ops scheduled inside asynchronous `[start, done]` windows.
    pub overlapped_compute_ops: usize,
    /// Total collective transfer bytes overlapped with independent compute.
    pub hidden_collective_bytes: u64,
}

/// Schedules asynchronous `*-start` collectives early and `*-done` completions late around
/// independent MXU/VPU compute regions (`LatencyHidingSchedulerPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct LatencyHidingSchedulerPass;

impl LatencyHidingSchedulerPass {
    /// Computes a latency-hiding instruction order and populates `module.schedule`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] whose instruction order and `HloScheduleProto` are updated.
    pub fn run(&self, module: &mut HloModuleProto) -> LatencyHidingScheduleReport {
        let mut report = LatencyHidingScheduleReport::default();
        let mut sequences = BTreeMap::new();

        for comp in &mut module.computations {
            let comp_rep = schedule_computation_latency_hiding(comp);
            report.overlapped_compute_ops = report
                .overlapped_compute_ops
                .saturating_add(comp_rep.overlapped_compute_ops);
            report.hidden_collective_bytes = report
                .hidden_collective_bytes
                .saturating_add(comp_rep.hidden_collective_bytes);
            sequences.insert(
                comp.id,
                HloInstructionSequenceProto {
                    instruction_ids: comp.instructions.iter().map(|i| i.id).collect(),
                },
            );
        }

        module.schedule = Some(HloScheduleProto { sequences });
        report
    }
}

/// Reorders instructions in `comp` so `*-start` ops issue immediately upon operand readiness while
/// `*-done` ops wait until just before their first consumer.
///
/// # Arguments
///
/// * `comp` - Mutable computation to schedule.
fn schedule_computation_latency_hiding(
    comp: &mut HloComputationProto,
) -> LatencyHidingScheduleReport {
    let n = comp.instructions.len();
    let id_to_idx: BTreeMap<i64, usize> = comp
        .instructions
        .iter()
        .enumerate()
        .map(|(idx, inst)| (inst.id, idx))
        .collect();
    let mut in_deg = vec![0_usize; n];
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (u_idx, inst) in comp.instructions.iter().enumerate() {
        for &dep_id in inst
            .operand_ids
            .iter()
            .chain(inst.control_predecessor_ids.iter())
        {
            id_to_idx
                .get(&dep_id)
                .copied()
                .filter(|&p_idx| p_idx != u_idx)
                .into_iter()
                .for_each(|p_idx| {
                    users
                        .get_mut(p_idx)
                        .into_iter()
                        .for_each(|u_list| u_list.push(u_idx));
                    in_deg
                        .get_mut(u_idx)
                        .into_iter()
                        .for_each(|d| *d = d.saturating_add(1));
                });
        }
    }

    let mut ready: Vec<usize> = in_deg
        .iter()
        .enumerate()
        .filter_map(|(idx, &deg)| (deg == 0).then_some(idx))
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);

    let priority_of = |inst: &HloInstructionProto| -> u8 {
        if matches!(inst.opcode.as_str(), "parameter" | "constant") {
            0
        } else if is_async_start_opcode(&inst.opcode) {
            1
        } else if is_async_done_opcode(&inst.opcode) {
            3
        } else {
            2
        }
    };

    while !ready.is_empty() {
        ready.sort_by(|&a, &b| {
            let pa = comp.instructions.get(a).map_or(2, priority_of);
            let pb = comp.instructions.get(b).map_or(2, priority_of);
            pa.cmp(&pb).then_with(|| a.cmp(&b))
        });
        let chosen = ready.remove(0);
        order.push(chosen);
        for &u_idx in users.get(chosen).into_iter().flatten() {
            in_deg.get_mut(u_idx).into_iter().for_each(|d| {
                *d = d.saturating_sub(1);
                if *d == 0 {
                    ready.push(u_idx);
                }
            });
        }
    }

    if order.len() == n {
        comp.instructions = order
            .into_iter()
            .filter_map(|idx| comp.instructions.get(idx).cloned())
            .collect();
    }

    measure_async_overlap_windows(&comp.instructions)
}

/// Measures independent compute ops and collective bytes overlapped inside `[*-start, *-done]` windows.
///
/// # Arguments
///
/// * `instructions` - Scheduled instruction slice.
fn measure_async_overlap_windows(
    instructions: &[HloInstructionProto],
) -> LatencyHidingScheduleReport {
    let pos_map: BTreeMap<i64, usize> = instructions
        .iter()
        .enumerate()
        .map(|(pos, inst)| (inst.id, pos))
        .collect();
    let mut report = LatencyHidingScheduleReport::default();
    for (done_pos, inst) in instructions.iter().enumerate() {
        let start_pos_opt = is_async_done_opcode(&inst.opcode)
            .then(|| {
                inst.operand_ids
                    .first()
                    .and_then(|id| pos_map.get(id).copied())
            })
            .flatten();
        if let Some(start_pos) = start_pos_opt {
            let between = instructions
                .get((start_pos + 1)..done_pos)
                .unwrap_or(&[])
                .iter()
                .filter(|mid| {
                    !is_async_start_opcode(&mid.opcode) && !is_async_done_opcode(&mid.opcode)
                })
                .count();
            if between > 0 {
                report.overlapped_compute_ops =
                    report.overlapped_compute_ops.saturating_add(between);
                let bytes = inst.shape.as_ref().map_or(0, estimate_shape_bytes);
                report.hidden_collective_bytes =
                    report.hidden_collective_bytes.saturating_add(bytes);
            }
        }
    }
    report
}

/// Maps a synchronous collective opcode to its `(start_opcode, done_opcode)` pair.
///
/// # Arguments
///
/// * `opcode` - HLO collective opcode mnemonic.
fn async_collective_pair_opcodes(opcode: &str) -> Option<(&'static str, &'static str)> {
    match opcode {
        "all-reduce" => Some(("all-reduce-start", "all-reduce-done")),
        "all-gather" => Some(("all-gather-start", "all-gather-done")),
        "reduce-scatter" => Some(("reduce-scatter-start", "reduce-scatter-done")),
        "all-to-all" => Some(("all-to-all-start", "all-to-all-done")),
        "collective-permute" => Some(("collective-permute-start", "collective-permute-done")),
        "ragged-all-to-all" => Some(("ragged-all-to-all-start", "ragged-all-to-all-done")),
        _ => None,
    }
}

/// Returns `true` if `opcode` initiates an asynchronous collective or DMA copy.
///
/// # Arguments
///
/// * `opcode` - HLO opcode mnemonic.
fn is_async_start_opcode(opcode: &str) -> bool {
    opcode.ends_with("-start")
}

/// Returns `true` if `opcode` completes an asynchronous collective or DMA copy.
///
/// # Arguments
///
/// * `opcode` - HLO opcode mnemonic.
fn is_async_done_opcode(opcode: &str) -> bool {
    opcode.ends_with("-done")
}

/// Returns the maximum instruction ID in `comp` (or `0` when empty).
///
/// # Arguments
///
/// * `comp` - Computation to scan.
fn max_instruction_id(comp: &HloComputationProto) -> i64 {
    comp.instructions.iter().map(|i| i.id).max().unwrap_or(0)
}

/// Increments `next_id` and returns the newly allocated instruction ID.
///
/// # Arguments
///
/// * `next_id` - Mutable ID counter.
const fn alloc_id(next_id: &mut i64) -> i64 {
    let id = *next_id;
    *next_id = next_id.saturating_add(1);
    id
}

/// Constructs an unpadded [`ShapeProto`] with `element_type` and `dimensions`.
///
/// # Arguments
///
/// * `element_type` - `PrimitiveType` discriminant (`F32 = 11`, etc.).
/// * `dimensions` - Logical dimension extents.
fn make_tensor_shape(element_type: i32, dimensions: Vec<i64>) -> ShapeProto {
    let rank = dimensions.len();
    let ty = if element_type == 0 {
        PrimitiveType::F32 as i32
    } else {
        element_type
    };
    ShapeProto {
        element_type: ty,
        dimensions,
        is_dynamic_dimension: vec![false; rank],
        ..ShapeProto::default()
    }
}

/// Resolves an ID transitively through `alias_map`.
///
/// # Arguments
///
/// * `start` - Starting instruction ID.
/// * `alias_map` - Mapping from replaced instruction ID to replacement ID.
fn resolve_alias_chain(start: i64, alias_map: &BTreeMap<i64, i64>) -> i64 {
    let mut cur = start;
    while let Some(&target) = alias_map.get(&cur).filter(|&&t| t != cur) {
        cur = target;
    }
    cur
}

/// Rewrites `inst.operand_ids` through a transitive `alias_map`.
///
/// # Arguments
///
/// * `inst` - Mutable instruction whose operand IDs are rewritten.
/// * `alias_map` - Mapping from replaced instruction ID to canonical replacement ID.
fn rewrite_operands(inst: &mut HloInstructionProto, alias_map: &BTreeMap<i64, i64>) {
    if alias_map.is_empty() {
        return;
    }
    for op_mut in &mut inst.operand_ids {
        *op_mut = resolve_alias_chain(*op_mut, alias_map);
    }
}

/// Applies `alias_map` across all operand references and `comp.root_id`, removing aliased instructions.
///
/// # Arguments
///
/// * `comp` - Mutable computation to update.
/// * `alias_map` - Mapping from replaced instruction ID to replacement ID.
fn apply_alias_map(comp: &mut HloComputationProto, alias_map: &BTreeMap<i64, i64>) {
    for inst in &mut comp.instructions {
        rewrite_operands(inst, alias_map);
    }
    comp.root_id = resolve_alias_chain(comp.root_id, alias_map);
    comp.instructions
        .retain(|inst| !alias_map.contains_key(&inst.id));
}

/// Checks whether `inst` is a constant whose elements all equal `target`.
///
/// # Arguments
///
/// * `inst` - Instruction to inspect.
/// * `target` - Expected scalar value (`0.0` or `1.0`).
fn is_scalar_const(inst: &HloInstructionProto, target: f32) -> bool {
    if inst.opcode != "constant" {
        return false;
    }
    let Some(lit) = &inst.literal else {
        return false;
    };
    if !lit.f32s.is_empty() {
        lit.f32s.iter().all(|&v| (v - target).abs() <= f32::EPSILON)
    } else if !lit.s32s.is_empty() {
        #[allow(clippy::cast_possible_truncation)]
        let target_i = target as i32;
        lit.s32s.iter().all(|&v| v == target_i)
    } else {
        false
    }
}

/// Returns `true` if `lhs` and `rhs` have identical `element_type` and `dimensions`.
///
/// # Arguments
///
/// * `lhs` - First optional [`ShapeProto`].
/// * `rhs` - Second optional [`ShapeProto`].
fn same_type_and_dims(lhs: Option<&ShapeProto>, rhs: Option<&ShapeProto>) -> bool {
    lhs.zip(rhs)
        .is_some_and(|(a, b)| a.element_type == b.element_type && a.dimensions == b.dimensions)
}

/// Computes the total scalar element count in `shape`.
///
/// # Arguments
///
/// * `shape` - Tensor [`ShapeProto`].
fn shape_elem_count(shape: &ShapeProto) -> usize {
    shape.dimensions.iter().fold(1_usize, |acc, &d| {
        acc.saturating_mul(usize::try_from(d.max(1)).unwrap_or(1))
    })
}

/// Estimates unpadded byte size of `shape`.
///
/// # Arguments
///
/// * `shape` - Tensor [`ShapeProto`].
fn estimate_shape_bytes(shape: &ShapeProto) -> u64 {
    if !shape.tuple_shapes.is_empty() {
        return shape.tuple_shapes.iter().map(estimate_shape_bytes).sum();
    }
    let elem_bytes: u64 = match PrimitiveType::try_from(shape.element_type).ok() {
        Some(PrimitiveType::Pred | PrimitiveType::S8 | PrimitiveType::U8) => 1,
        Some(
            PrimitiveType::F16 | PrimitiveType::Bf16 | PrimitiveType::S16 | PrimitiveType::U16,
        ) => 2,
        Some(PrimitiveType::F64 | PrimitiveType::S64 | PrimitiveType::U64 | PrimitiveType::C64) => {
            8
        }
        Some(PrimitiveType::C128) => 16,
        _ => 4,
    };
    let elems = shape.dimensions.iter().fold(1_u64, |acc, &d| {
        acc.saturating_mul(u64::try_from(d.max(1)).unwrap_or(1))
    });
    elems.saturating_mul(elem_bytes)
}

/// Returns `true` if `opcode` is pure and eligible for loop-invariant code motion.
///
/// # Arguments
///
/// * `opcode` - HLO opcode mnemonic.
fn is_pure_hoistable_opcode(opcode: &str) -> bool {
    matches!(
        opcode,
        "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "maximum"
            | "minimum"
            | "negate"
            | "abs"
            | "exponential"
            | "log"
            | "tanh"
            | "sqrt"
            | "rsqrt"
            | "broadcast"
            | "reshape"
            | "transpose"
            | "dot"
    )
}

/// Returns `true` if `inst` is pure and eligible for Common Subexpression Elimination.
///
/// # Arguments
///
/// * `inst` - Instruction to inspect.
fn is_pure_cse_candidate(inst: &HloInstructionProto) -> bool {
    inst.opcode != "parameter"
        && inst.called_computation_ids.is_empty()
        && !has_instruction_side_effects(inst)
}

/// Returns `true` if `inst` has observable side effects and must not be removed by DCE or CSE.
///
/// # Arguments
///
/// * `inst` - Instruction to inspect.
fn has_instruction_side_effects(inst: &HloInstructionProto) -> bool {
    let side_effecting_op = matches!(
        inst.opcode.as_str(),
        "infeed"
            | "outfeed"
            | "send"
            | "send-done"
            | "recv"
            | "recv-done"
            | "after-all"
            | "optimization-barrier"
            | "rng-bit-generator"
            | "all-reduce"
            | "all-reduce-start"
            | "all-reduce-done"
            | "all-gather"
            | "all-gather-start"
            | "all-gather-done"
            | "reduce-scatter"
            | "reduce-scatter-start"
            | "reduce-scatter-done"
            | "all-to-all"
            | "all-to-all-start"
            | "all-to-all-done"
            | "collective-permute"
            | "collective-permute-start"
            | "collective-permute-done"
            | "ragged-all-to-all"
            | "ragged-all-to-all-start"
            | "ragged-all-to-all-done"
            | "copy-start"
            | "copy-done"
    );
    side_effecting_op || (inst.opcode == "custom-call" && inst.custom_call_has_side_effect)
}
