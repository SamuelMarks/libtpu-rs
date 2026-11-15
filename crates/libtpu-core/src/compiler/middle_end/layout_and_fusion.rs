//! TPU physical layout assignment, MXU systolic tiling, operator fusion, and rematerialization.
//!
//! Assigns `(8, 128)` / `(4, 128)` / `(2, 128)` tiles and sub-32-bit lane packings, tiles `dot`
//! onto `128×128` / `256×256` MXUs, fuses producers/siblings, and rematerializes cheap chains.

use crate::compiler::passes::HardwareRegisterFileConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    FrontendAttributes, HloComputationProto, HloInstructionProto, HloInstructionSequenceProto,
    HloModuleProto, HloRematerializationSummaryProto, HloScheduleProto,
    JellyfishLayoutProfileProto, JellyfishOpLayoutProposedProto, LayoutProto, LiteralProto,
    PaddingConfig, PaddingConfigDimension, PrimitiveType, RematerializedInstructionEntryProto,
    ShapeProto, TileProto,
};
use crate::types::TpuGeneration;
use std::collections::{BTreeMap, BTreeSet};

/// Frontend attribute key storing serialized JSON [`MxuContractionSchedule`] on tiled `dot` ops.
pub const MXU_TILE_SCHEDULE_ATTR_KEY: &str = "xla.tpu.mxu_tile_schedule";

/// Summary report emitted by [`TpuLayoutAssignmentPass`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TpuLayoutAssignmentReport {
    /// Total number of instruction shapes assigned physical [`LayoutProto`]s.
    pub assigned_instructions: usize,
    /// Number of XLU `transpose` relayout ops inserted on `minor_to_major` conflicts.
    pub inserted_transposes: usize,
    /// Number of explicit tile-alignment `pad` ops inserted.
    pub inserted_pads: usize,
    /// Number of sub-32-bit `bitcast-convert` lane-packing ops inserted.
    pub inserted_bitcast_packs: usize,
    /// Proposed per-instruction input/output layout profile (`xla.jellyfish.LayoutProfileProto`).
    pub layout_profile: JellyfishLayoutProfileProto,
}

/// Assigns physical TPU [`LayoutProto`]s (`minor_to_major`, `(8, 128)`/`(4, 128)`/`(2, 128)` tiles,
/// and sub-32-bit compact lane packings) across an [`HloModuleProto`] (`TpuLayoutAssignmentPass`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TpuLayoutAssignmentPass {
    /// Whether to materialize explicit `pad` ops when a `dot` operand is unaligned to sublane tile height.
    pub materialize_tile_pads: bool,
    /// Whether to materialize explicit `bitcast-convert` lane-packing ops on sub-32-bit `dot` operands.
    pub materialize_sub32_bitcast_packs: bool,
}

impl TpuLayoutAssignmentPass {
    /// Constructs a [`TpuLayoutAssignmentPass`] with explicit relayout materialization flags.
    ///
    /// # Arguments
    ///
    /// * `materialize_tile_pads` - Insert explicit `pad` ops on unaligned `dot` operands.
    /// * `materialize_sub32_bitcast_packs` - Insert explicit `bitcast-convert` lane-packing ops on sub-32-bit `dot` operands.
    #[must_use]
    pub const fn new(materialize_tile_pads: bool, materialize_sub32_bitcast_packs: bool) -> Self {
        Self {
            materialize_tile_pads,
            materialize_sub32_bitcast_packs,
        }
    }

    /// Assigns physical TPU layouts across all computations and program shapes in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to assign physical layouts on.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::LayoutAssignmentError`] if a `constrain_layout` instruction carries
    /// an invalid `minor_to_major` permutation.
    pub fn run(&self, module: &mut HloModuleProto) -> LibTpuResult<TpuLayoutAssignmentReport> {
        let mut report = TpuLayoutAssignmentReport::default();

        for comp in &mut module.computations {
            self.assign_computation_layouts(comp, &mut report)?;
        }

        if let Some(prog_shape) = &mut module.host_program_shape {
            for p_shape in &mut prog_shape.parameters {
                assign_shape_tpu_layout(p_shape, None);
            }
            prog_shape
                .result
                .as_mut()
                .into_iter()
                .for_each(|r_shape| assign_shape_tpu_layout(r_shape, None));
        }

        Ok(report)
    }

    /// Assigns physical layouts and materializes relayout instructions in `comp`.
    ///
    /// # Arguments
    ///
    /// * `comp` - Mutable computation to process.
    /// * `report` - Mutable report accumulator.
    #[allow(clippy::too_many_lines)]
    fn assign_computation_layouts(
        self,
        comp: &mut HloComputationProto,
        report: &mut TpuLayoutAssignmentReport,
    ) -> LibTpuResult<()> {
        let mut next_id = comp
            .instructions
            .iter()
            .map(|i| i.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut assigned_layouts: BTreeMap<i64, LayoutProto> = BTreeMap::new();
        let mut shape_lookup: BTreeMap<i64, ShapeProto> = BTreeMap::new();
        for inst in &mut comp.instructions {
            if let Some(shape) = &mut inst.shape {
                if let Some(existing) = shape.layout.as_deref().filter(|_| inst.constrain_layout) {
                    validate_minor_to_major(&existing.minor_to_major, shape.dimensions.len())?;
                }
                let preserve = inst
                    .constrain_layout
                    .then(|| shape.layout.as_deref().cloned())
                    .flatten();
                assign_shape_tpu_layout(shape, preserve.as_ref());
                shape.layout.as_deref().into_iter().for_each(|lay| {
                    assigned_layouts.insert(inst.id, lay.clone());
                });
                shape_lookup.insert(inst.id, shape.clone());
                report.assigned_instructions = report.assigned_instructions.saturating_add(1);
            }
        }

        let original = std::mem::take(&mut comp.instructions);
        let mut out: Vec<HloInstructionProto> = Vec::with_capacity(original.len() * 2);

        for mut inst in original {
            let is_binary_dot = inst.opcode == "dot" && inst.operand_ids.len() == 2;
            if is_binary_dot {
                let mut new_ops = Vec::with_capacity(2);
                for &op_id in &inst.operand_ids {
                    let mut cur_op_id = op_id;
                    if let Some(op_shape) = shape_lookup.get(&op_id).cloned() {
                        let rank = op_shape.dimensions.len();
                        let canonical_m2m: Vec<i64> = (0..rank)
                            .rev()
                            .map(|d| i64::try_from(d).unwrap_or(0))
                            .collect();
                        let needs_transpose = rank >= 2
                            && assigned_layouts
                                .get(&op_id)
                                .is_some_and(|op_lay| op_lay.minor_to_major != canonical_m2m);
                        if needs_transpose {
                            let tr_id = next_id;
                            next_id = next_id.saturating_add(1);
                            let mut relayout_shape = op_shape.clone();
                            assign_shape_tpu_layout(&mut relayout_shape, None);
                            out.push(HloInstructionProto {
                                id: tr_id,
                                name: format!("xlu_relayout_transpose_{tr_id}"),
                                opcode: String::from("transpose"),
                                shape: Some(relayout_shape),
                                operand_ids: vec![cur_op_id],
                                dimensions: (0..rank)
                                    .map(|d| i64::try_from(d).unwrap_or(0))
                                    .collect(),
                                ..HloInstructionProto::default()
                            });
                            cur_op_id = tr_id;
                            report.inserted_transposes =
                                report.inserted_transposes.saturating_add(1);
                        }

                        let unaligned_minor =
                            op_shape.dimensions.last().copied().filter(|&ext| {
                                self.materialize_tile_pads && rank >= 2 && ext % 8 != 0
                            });
                        if let Some(minor_ext) = unaligned_minor {
                            let pad_rem = 8 - (minor_ext % 8);
                            let zero_id = next_id;
                            next_id = next_id.saturating_add(1);
                            let mut zero_shape = ShapeProto {
                                element_type: op_shape.element_type,
                                ..ShapeProto::default()
                            };
                            assign_shape_tpu_layout(&mut zero_shape, None);
                            out.push(HloInstructionProto {
                                id: zero_id,
                                name: format!("layout_pad_zero_{zero_id}"),
                                opcode: String::from("constant"),
                                shape: Some(zero_shape.clone()),
                                literal: Some(LiteralProto {
                                    shape: Some(zero_shape),
                                    f32s: vec![0.0],
                                    ..LiteralProto::default()
                                }),
                                ..HloInstructionProto::default()
                            });

                            let mut padded_dims = op_shape.dimensions.clone();
                            padded_dims.last_mut().into_iter().for_each(|last_mut| {
                                *last_mut = last_mut.saturating_add(pad_rem);
                            });
                            let mut pad_cfg_dims = vec![
                                PaddingConfigDimension {
                                    edge_padding_low: 0,
                                    edge_padding_high: 0,
                                    interior_padding: 0,
                                };
                                rank
                            ];
                            pad_cfg_dims.last_mut().into_iter().for_each(|last_pc| {
                                last_pc.edge_padding_high = pad_rem;
                            });
                            let mut padded_shape = ShapeProto {
                                element_type: op_shape.element_type,
                                dimensions: padded_dims,
                                is_dynamic_dimension: vec![false; rank],
                                ..ShapeProto::default()
                            };
                            assign_shape_tpu_layout(&mut padded_shape, None);
                            let pad_id = next_id;
                            next_id = next_id.saturating_add(1);
                            out.push(HloInstructionProto {
                                id: pad_id,
                                name: format!("layout_tile_pad_{pad_id}"),
                                opcode: String::from("pad"),
                                shape: Some(padded_shape),
                                operand_ids: vec![cur_op_id, zero_id],
                                padding_config: Some(PaddingConfig {
                                    dimensions: pad_cfg_dims,
                                }),
                                ..HloInstructionProto::default()
                            });
                            cur_op_id = pad_id;
                            report.inserted_pads = report.inserted_pads.saturating_add(1);
                        }

                        let bits = primitive_type_bit_width(op_shape.element_type);
                        if self.materialize_sub32_bitcast_packs && bits < 32 {
                            let pack_id = next_id;
                            next_id = next_id.saturating_add(1);
                            let mut packed_shape = op_shape;
                            assign_shape_tpu_layout(&mut packed_shape, None);
                            out.push(HloInstructionProto {
                                id: pack_id,
                                name: format!("sub32_lane_pack_{pack_id}"),
                                opcode: String::from("bitcast-convert"),
                                shape: Some(packed_shape),
                                operand_ids: vec![cur_op_id],
                                ..HloInstructionProto::default()
                            });
                            cur_op_id = pack_id;
                            report.inserted_bitcast_packs =
                                report.inserted_bitcast_packs.saturating_add(1);
                        }
                    }
                    new_ops.push(cur_op_id);
                }
                inst.operand_ids = new_ops;
            }

            let out_layout = inst
                .shape
                .as_ref()
                .and_then(|s| s.layout.as_deref().cloned());
            let in_layouts: Vec<LayoutProto> = inst
                .operand_ids
                .iter()
                .filter_map(|id| assigned_layouts.get(id).cloned())
                .collect();
            report.layout_profile.entries.insert(
                inst.id,
                JellyfishOpLayoutProposedProto {
                    output: out_layout,
                    inputs: in_layouts,
                },
            );
            out.push(inst);
        }

        comp.instructions = out;
        Ok(())
    }
}

/// Validates that `minor_to_major` is a permutation of `0..rank`.
///
/// # Arguments
///
/// * `minor_to_major` - Candidate minor-to-major dimension order.
/// * `rank` - Expected tensor rank.
fn validate_minor_to_major(minor_to_major: &[i64], rank: usize) -> LibTpuResult<()> {
    if minor_to_major.len() != rank {
        return Err(LibTpuError::LayoutAssignmentError {
            shape_summary: format!("rank={rank}, m2m={minor_to_major:?}"),
            reason: "minor_to_major length does not match tensor rank",
        });
    }
    let mut seen = BTreeSet::new();
    let r_i64 = i64::try_from(rank).unwrap_or(0);
    for &d in minor_to_major {
        let invalid_dim = !(0..r_i64).contains(&d) || !seen.insert(d);
        if invalid_dim {
            return Err(LibTpuError::LayoutAssignmentError {
                shape_summary: format!("rank={rank}, m2m={minor_to_major:?}"),
                reason: "minor_to_major is not a valid permutation of 0..rank",
            });
        }
    }
    Ok(())
}

/// Assigns a TPU tiled [`LayoutProto`] (`(8, 128)` / `(4, 128)` / `(2, 128)` + sub-32-bit packing tile)
/// recursively to `shape` (including tuple sub-shapes).
///
/// # Arguments
///
/// * `shape` - Mutable [`ShapeProto`] to populate with a physical TPU layout.
/// * `preserved` - Optional pre-validated layout whose `minor_to_major` order should be preserved.
pub fn assign_shape_tpu_layout(shape: &mut ShapeProto, preserved: Option<&LayoutProto>) {
    if !shape.tuple_shapes.is_empty() {
        for sub in &mut shape.tuple_shapes {
            assign_shape_tpu_layout(sub, None);
        }
        return;
    }

    let rank = shape.dimensions.len();
    let minor_to_major = preserved
        .filter(|p| p.minor_to_major.len() == rank)
        .map_or_else(
            || {
                (0..rank)
                    .rev()
                    .map(|idx| i64::try_from(idx).unwrap_or(0))
                    .collect::<Vec<i64>>()
            },
            |p| p.minor_to_major.clone(),
        );

    let bits = primitive_type_bit_width(shape.element_type);
    let mut tiles = Vec::new();
    if rank >= 2 {
        let second_minor_dim = minor_to_major
            .get(1)
            .and_then(|&d| usize::try_from(d).ok())
            .and_then(|u| shape.dimensions.get(u).copied())
            .unwrap_or(8);
        let sublane_tile = if second_minor_dim <= 2 {
            2_i64
        } else if second_minor_dim <= 4 {
            4_i64
        } else {
            8_i64
        };
        tiles.push(TileProto {
            dimensions: vec![sublane_tile, 128],
        });
        match bits {
            16 => tiles.push(TileProto {
                dimensions: vec![2, 1],
            }),
            8 => tiles.push(TileProto {
                dimensions: vec![4, 1],
            }),
            4 => tiles.push(TileProto {
                dimensions: vec![8, 1],
            }),
            _ => {}
        }
    } else if rank == 1 {
        tiles.push(TileProto {
            dimensions: vec![128],
        });
    }

    let prev_mem_space = shape.layout.as_ref().map_or(0, |l| l.memory_space);
    shape.layout = Some(Box::new(LayoutProto {
        minor_to_major,
        tiles,
        element_size_in_bits: bits,
        memory_space: prev_mem_space,
        tail_padding_alignment_in_elements: if rank >= 1 { 128 } else { 1 },
        ..LayoutProto::default()
    }));
}

/// Returns the bit width of an `xla.PrimitiveType` discriminant (`4`, `8`, `16`, `32`, `64`, `128`).
///
/// # Arguments
///
/// * `raw_type` - Numeric `PrimitiveType` discriminant.
#[must_use]
pub fn primitive_type_bit_width(raw_type: i32) -> i64 {
    match PrimitiveType::try_from(raw_type).ok() {
        Some(PrimitiveType::S4 | PrimitiveType::U4 | PrimitiveType::F4e2m1fn) => 4,
        Some(
            PrimitiveType::Pred
            | PrimitiveType::S8
            | PrimitiveType::U8
            | PrimitiveType::F8e5m2
            | PrimitiveType::F8e4m3
            | PrimitiveType::F8e4m3fn
            | PrimitiveType::F8e4m3b11fnuz
            | PrimitiveType::F8e3m4
            | PrimitiveType::F8e5m2fnuz
            | PrimitiveType::F8e4m3fnuz
            | PrimitiveType::F8e8m0fnu,
        ) => 8,
        Some(
            PrimitiveType::F16 | PrimitiveType::Bf16 | PrimitiveType::S16 | PrimitiveType::U16,
        ) => 16,
        Some(PrimitiveType::F64 | PrimitiveType::S64 | PrimitiveType::U64 | PrimitiveType::C64) => {
            64
        }
        Some(PrimitiveType::C128) => 128,
        _ => 32,
    }
}

/// Per-instruction systolic array tiling and accumulator chaining schedule produced by [`MxuTilingPass`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MxuContractionSchedule {
    /// Target `dot` instruction ID.
    pub instruction_id: i64,
    /// Systolic array edge width (`128` on `v2`–`v5p`, `256` on `v6e`/`v7x`).
    pub systolic_dim: i64,
    /// Number of hardware MXUs per `TensorCore` utilized (`1`, `2`, or `4`).
    pub mxus_per_core: u32,
    /// Number of LHS non-contracting (`M`) tiles.
    pub m_tiles: u64,
    /// Number of contracting (`K`) accumulator-chained tiles.
    pub k_tiles: u64,
    /// Number of RHS non-contracting (`N`) tiles.
    pub n_tiles: u64,
    /// Whether `$mreg` accumulator chaining across `k_tiles > 1` is active.
    pub accumulator_chaining: bool,
    /// Assigned MXU unit ordinal (`0..mxus_per_core`) for each `(m_tile, n_tile)` output block.
    pub block_mxu_assignments: Vec<u32>,
}

/// Summary report emitted by [`MxuTilingPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MxuTilingReport {
    /// Systolic array edge dimension (`128` or `256`).
    pub systolic_dim: i64,
    /// Number of MXUs per `TensorCore`.
    pub mxus_per_core: u32,
    /// Schedules generated for every `dot` contraction in the module.
    pub schedules: Vec<MxuContractionSchedule>,
}

/// Tiles matrix contractions onto `128×128` (`v2`–`v5p`) or `256×256` (`v6e`/`v7x`) MXU systolic
/// array blocks with `$mreg` accumulator chaining across multiple MXUs per `TensorCore` (`MxuTilingPass`).
#[derive(Debug, Clone, Copy)]
pub struct MxuTilingPass {
    /// Target Cloud TPU hardware generation.
    pub generation: TpuGeneration,
}

impl MxuTilingPass {
    /// Constructs an [`MxuTilingPass`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation determining systolic array width and MXU count.
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self { generation }
    }

    /// Returns `(systolic_dim, mxus_per_core)` for `self.generation`.
    #[must_use]
    pub const fn hardware_mxu_geometry(&self) -> (i64, u32) {
        match self.generation {
            TpuGeneration::V2 | TpuGeneration::V4Lite => (128, 1),
            TpuGeneration::V3 => (128, 2),
            TpuGeneration::V4 | TpuGeneration::V5e | TpuGeneration::V5p => (128, 4),
            TpuGeneration::V6e | TpuGeneration::V7x => (256, 4),
        }
    }

    /// Computes systolic array tiling and multi-MXU accumulator chaining schedules across `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] whose `dot` instructions receive MXU tile schedules.
    pub fn run(&self, module: &mut HloModuleProto) -> MxuTilingReport {
        let (systolic_dim, mxus_per_core) = self.hardware_mxu_geometry();
        let mut schedules = Vec::new();
        let tile_u64 = u64::try_from(systolic_dim.max(1)).unwrap_or(128);

        for comp in &mut module.computations {
            let shapes: BTreeMap<i64, ShapeProto> = comp
                .instructions
                .iter()
                .filter_map(|i| i.shape.as_ref().map(|s| (i.id, s.clone())))
                .collect();

            for inst in &mut comp.instructions {
                if inst.opcode != "dot" {
                    continue;
                }
                let lhs_shape = inst.operand_ids.first().and_then(|id| shapes.get(id));
                let rhs_shape = inst.operand_ids.get(1).and_then(|id| shapes.get(id));
                let (m_ext, k_ext, n_ext) = extract_mkn_extents(inst, lhs_shape, rhs_shape);
                let m_tiles = m_ext.div_ceil(tile_u64).max(1);
                let k_tiles = k_ext.div_ceil(tile_u64).max(1);
                let n_tiles = n_ext.div_ceil(tile_u64).max(1);

                let total_mn_blocks = usize::try_from(m_tiles.saturating_mul(n_tiles))
                    .unwrap_or(1)
                    .min(256);
                let block_mxu_assignments: Vec<u32> = (0..total_mn_blocks)
                    .map(|blk| (u32::try_from(blk).unwrap_or(0)) % mxus_per_core.max(1))
                    .collect();

                let sched = MxuContractionSchedule {
                    instruction_id: inst.id,
                    systolic_dim,
                    mxus_per_core,
                    m_tiles,
                    k_tiles,
                    n_tiles,
                    accumulator_chaining: k_tiles > 1,
                    block_mxu_assignments,
                };
                let json_str = serde_json::to_string(&sched).unwrap_or_default();
                let fa = inst
                    .frontend_attributes
                    .get_or_insert_with(FrontendAttributes::default);
                fa.map
                    .insert(String::from(MXU_TILE_SCHEDULE_ATTR_KEY), json_str);
                schedules.push(sched);
            }
        }

        MxuTilingReport {
            systolic_dim,
            mxus_per_core,
            schedules,
        }
    }
}

/// Extracts `(M, K, N)` extents from a `dot` instruction and its LHS/RHS shapes.
///
/// # Arguments
///
/// * `inst` - `dot` instruction.
/// * `lhs` - Optional LHS operand shape.
/// * `rhs` - Optional RHS operand shape.
fn extract_mkn_extents(
    inst: &HloInstructionProto,
    lhs: Option<&ShapeProto>,
    rhs: Option<&ShapeProto>,
) -> (u64, u64, u64) {
    let dnums = inst.dot_dimension_numbers.as_ref();
    let lhs_c = dnums.map_or(&[1_i64][..], |d| d.lhs_contracting_dimensions.as_slice());
    let lhs_b = dnums.map_or(&[][..], |d| d.lhs_batch_dimensions.as_slice());
    let rhs_c = dnums.map_or(&[0_i64][..], |d| d.rhs_contracting_dimensions.as_slice());
    let rhs_b = dnums.map_or(&[][..], |d| d.rhs_batch_dimensions.as_slice());

    let (mut m, mut k) = (1_u64, 1_u64);
    for (idx, &ext) in lhs
        .map_or(&[][..], |ls| ls.dimensions.as_slice())
        .iter()
        .enumerate()
    {
        let di = i64::try_from(idx).unwrap_or(0);
        let u_ext = u64::try_from(ext.max(1)).unwrap_or(1);
        if lhs_c.contains(&di) {
            k = k.saturating_mul(u_ext);
        } else if !lhs_b.contains(&di) {
            m = m.saturating_mul(u_ext);
        }
    }
    let mut n = 1_u64;
    for (idx, &ext) in rhs
        .map_or(&[][..], |rs| rs.dimensions.as_slice())
        .iter()
        .enumerate()
    {
        let di = i64::try_from(idx).unwrap_or(0);
        let u_ext = u64::try_from(ext.max(1)).unwrap_or(1);
        if !rhs_c.contains(&di) && !rhs_b.contains(&di) {
            n = n.saturating_mul(u_ext);
        }
    }
    (m, k, n)
}

/// Summary report emitted by [`TpuProducerConsumerFusionPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProducerConsumerFusionReport {
    /// Number of producers absorbed into consumer `fusion` computations.
    pub fused_producers: usize,
    /// Number of `fusion` computations created.
    pub created_fusion_computations: usize,
    /// Number of candidate fusions rejected due to VMEM scratchpad or physical VREG pressure.
    pub rejected_for_budget: usize,
}

/// Fuses elementwise/broadcast producers into consumers subject to VMEM scratchpad byte budget
/// and physical `VREG` live-range capacity (`TpuProducerConsumerFusionPass`).
#[derive(Debug, Clone, Copy)]
pub struct TpuProducerConsumerFusionPass {
    /// Maximum combined working-set bytes allowed in VMEM per fusion region.
    pub vmem_budget_bytes: u64,
    /// Maximum live 2D `VREG` registers allowed inside a single fusion kernel.
    pub max_live_vregs: u16,
}

impl TpuProducerConsumerFusionPass {
    /// Constructs a [`TpuProducerConsumerFusionPass`] calibrated to `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation determining VMEM and VREG capacity.
    #[must_use]
    pub fn for_generation(generation: TpuGeneration) -> Self {
        let reg_cfg = HardwareRegisterFileConfig::for_generation(generation);
        let vmem_budget_bytes = match generation {
            TpuGeneration::V2 | TpuGeneration::V3 | TpuGeneration::V4Lite => 16 * 1024 * 1024,
            TpuGeneration::V4 | TpuGeneration::V5e => 32 * 1024 * 1024,
            TpuGeneration::V5p | TpuGeneration::V6e => 64 * 1024 * 1024,
            TpuGeneration::V7x => 128 * 1024 * 1024,
        };
        Self {
            vmem_budget_bytes,
            max_live_vregs: u16::from(reg_cfg.num_vregs),
        }
    }

    /// Constructs a [`TpuProducerConsumerFusionPass`] with explicit VMEM and VREG limits.
    ///
    /// # Arguments
    ///
    /// * `vmem_budget_bytes` - Maximum VMEM scratchpad working set per fusion in bytes.
    /// * `max_live_vregs` - Maximum physical `VREG` count per fusion kernel.
    #[must_use]
    pub const fn new(vmem_budget_bytes: u64, max_live_vregs: u16) -> Self {
        Self {
            vmem_budget_bytes,
            max_live_vregs,
        }
    }

    /// Runs producer-consumer operator fusion on the entry computation of `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to fuse in-place.
    #[allow(clippy::too_many_lines)]
    pub fn run(&self, module: &mut HloModuleProto) -> ProducerConsumerFusionReport {
        let mut report = ProducerConsumerFusionReport::default();
        let Some(entry_pos) = resolve_entry_computation_index(module) else {
            return report;
        };

        let mut next_comp_id = module
            .computations
            .iter()
            .map(|c| c.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut entry = module
            .computations
            .get(entry_pos)
            .cloned()
            .unwrap_or_default();
        let mut new_subcomps: Vec<HloComputationProto> = Vec::new();
        let mut next_inst_id = entry
            .instructions
            .iter()
            .map(|i| i.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let use_counts = count_instruction_uses(&entry);
        let by_id: BTreeMap<i64, HloInstructionProto> = entry
            .instructions
            .iter()
            .map(|i| (i.id, i.clone()))
            .collect();
        let mut absorbed_ids: BTreeSet<i64> = BTreeSet::new();
        let mut rewritten_entry: Vec<HloInstructionProto> =
            Vec::with_capacity(entry.instructions.len());
        let mut root_remap: BTreeMap<i64, i64> = BTreeMap::new();

        for mut consumer in entry.instructions {
            for op_mut in &mut consumer.operand_ids {
                if let Some(&rep) = root_remap.get(op_mut) {
                    *op_mut = rep;
                }
            }
            if !is_fusable_consumer(&consumer.opcode) {
                rewritten_entry.push(consumer);
                continue;
            }

            let fused_prods: Vec<HloInstructionProto> = consumer
                .operand_ids
                .iter()
                .filter(|op_id| use_counts.get(op_id).copied().unwrap_or(0) == 1)
                .filter_map(|op_id| by_id.get(op_id))
                .filter(|prod| is_fusable_producer(&prod.opcode))
                .cloned()
                .collect();
            if fused_prods.is_empty() {
                rewritten_entry.push(consumer);
                continue;
            }

            let prod_bytes: u64 = fused_prods
                .iter()
                .filter_map(|p| p.shape.as_ref())
                .map(shape_bytes_256_aligned)
                .sum();
            let cons_bytes = consumer.shape.as_ref().map_or(256, shape_bytes_256_aligned);
            let est_vregs = u16::try_from(
                fused_prods
                    .len()
                    .saturating_add(consumer.operand_ids.len())
                    .saturating_add(1),
            )
            .unwrap_or(u16::MAX);
            let over_budget = prod_bytes.saturating_add(cons_bytes) > self.vmem_budget_bytes
                || est_vregs > self.max_live_vregs;
            if over_budget {
                report.rejected_for_budget = report.rejected_for_budget.saturating_add(1);
                rewritten_entry.push(consumer);
                continue;
            }

            let prod_id_set: BTreeSet<i64> = fused_prods.iter().map(|p| p.id).collect();
            absorbed_ids.extend(prod_id_set.iter().copied());

            let mut ext_operands: Vec<i64> = Vec::new();
            let candidate_ext = fused_prods
                .iter()
                .flat_map(|p| p.operand_ids.iter().copied())
                .chain(consumer.operand_ids.iter().copied());
            for raw_op in candidate_ext {
                let resolved = root_remap.get(&raw_op).copied().unwrap_or(raw_op);
                if !prod_id_set.contains(&resolved) && !ext_operands.contains(&resolved) {
                    ext_operands.push(resolved);
                }
            }

            let mut sub_insts: Vec<HloInstructionProto> = Vec::new();
            let mut sub_map: BTreeMap<i64, i64> = BTreeMap::new();
            let mut sub_local_id = 1_i64;
            for (p_idx, &ext_id) in ext_operands.iter().enumerate() {
                let pid = sub_local_id;
                sub_local_id = sub_local_id.saturating_add(1);
                let p_shape = by_id.get(&ext_id).and_then(|i| i.shape.clone());
                sub_insts.push(HloInstructionProto {
                    id: pid,
                    name: format!("fusion_param_{p_idx}"),
                    opcode: String::from("parameter"),
                    parameter_number: i64::try_from(p_idx).unwrap_or(0),
                    shape: p_shape,
                    ..HloInstructionProto::default()
                });
                sub_map.insert(ext_id, pid);
            }
            for prod in &fused_prods {
                let cloned_id = sub_local_id;
                sub_local_id = sub_local_id.saturating_add(1);
                let mut cloned = prod.clone();
                cloned.id = cloned_id;
                cloned.operand_ids = prod
                    .operand_ids
                    .iter()
                    .map(|op| {
                        let resolved = root_remap.get(op).copied().unwrap_or(*op);
                        sub_map.get(&resolved).copied().unwrap_or(resolved)
                    })
                    .collect();
                sub_insts.push(cloned);
                sub_map.insert(prod.id, cloned_id);
            }
            let root_local_id = sub_local_id;
            let mut cloned_cons = consumer.clone();
            cloned_cons.id = root_local_id;
            cloned_cons.operand_ids = consumer
                .operand_ids
                .iter()
                .map(|op| sub_map.get(op).copied().unwrap_or(*op))
                .collect();
            let fusion_kind = if matches!(consumer.opcode.as_str(), "dot" | "convolution") {
                "kOutput"
            } else {
                "kLoop"
            };
            let out_shape = consumer.shape.clone();
            let orig_cons_id = consumer.id;
            sub_insts.push(cloned_cons);

            let sub_cid = next_comp_id;
            next_comp_id = next_comp_id.saturating_add(1);
            new_subcomps.push(HloComputationProto {
                id: sub_cid,
                name: format!("fused_computation_{sub_cid}"),
                instructions: sub_insts,
                root_id: root_local_id,
                is_fusion_computation: true,
                ..HloComputationProto::default()
            });

            let fusion_inst_id = next_inst_id;
            next_inst_id = next_inst_id.saturating_add(1);
            rewritten_entry.push(HloInstructionProto {
                id: fusion_inst_id,
                name: format!("tpu_fusion_{fusion_inst_id}"),
                opcode: String::from("fusion"),
                fusion_kind: String::from(fusion_kind),
                shape: out_shape,
                operand_ids: ext_operands,
                called_computation_ids: vec![sub_cid],
                ..HloInstructionProto::default()
            });

            root_remap.insert(orig_cons_id, fusion_inst_id);
            if entry.root_id == orig_cons_id {
                entry.root_id = fusion_inst_id;
            }
            report.fused_producers = report.fused_producers.saturating_add(fused_prods.len());
            report.created_fusion_computations =
                report.created_fusion_computations.saturating_add(1);
        }

        rewritten_entry.retain(|inst| !absorbed_ids.contains(&inst.id));
        entry.instructions = rewritten_entry;
        module
            .computations
            .get_mut(entry_pos)
            .into_iter()
            .for_each(|slot| *slot = entry.clone());
        module.computations.extend(new_subcomps);
        report
    }
}

/// Summary report emitted by [`MultiOutputFusionPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiOutputFusionReport {
    /// Number of sibling instruction pairs merged into multi-output `tuple` fusions.
    pub multi_output_fusions: usize,
}

/// Merges sibling pure instructions sharing common HBM operand inputs into a single multi-output
/// `tuple` `fusion` computation (`MultiOutputFusionPass`).
#[derive(Debug, Clone, Copy)]
pub struct MultiOutputFusionPass {
    /// Maximum VMEM working-set bytes allowed for a merged multi-output fusion.
    pub vmem_budget_bytes: u64,
}

impl Default for MultiOutputFusionPass {
    fn default() -> Self {
        Self {
            vmem_budget_bytes: 32 * 1024 * 1024,
        }
    }
}

impl MultiOutputFusionPass {
    /// Constructs a [`MultiOutputFusionPass`] with `vmem_budget_bytes`.
    ///
    /// # Arguments
    ///
    /// * `vmem_budget_bytes` - Maximum combined VMEM bytes for merged sibling outputs.
    #[must_use]
    pub const fn new(vmem_budget_bytes: u64) -> Self {
        Self { vmem_budget_bytes }
    }

    /// Runs sibling multi-output fusion on the entry computation of `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to transform in-place.
    #[allow(clippy::too_many_lines)]
    pub fn run(&self, module: &mut HloModuleProto) -> MultiOutputFusionReport {
        let mut report = MultiOutputFusionReport::default();
        let Some(entry_pos) = resolve_entry_computation_index(module) else {
            return report;
        };

        let mut next_comp_id = module
            .computations
            .iter()
            .map(|c| c.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let mut entry = module
            .computations
            .get(entry_pos)
            .cloned()
            .unwrap_or_default();
        let mut next_inst_id = entry
            .instructions
            .iter()
            .map(|i| i.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut paired: BTreeSet<usize> = BTreeSet::new();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (i, inst_a) in entry.instructions.iter().enumerate() {
            if paired.contains(&i) || !is_fusable_producer(&inst_a.opcode) {
                continue;
            }
            for (j, inst_b) in entry.instructions.iter().enumerate().skip(i + 1) {
                if paired.contains(&j) || !is_fusable_producer(&inst_b.opcode) {
                    continue;
                }
                let shares_input = inst_a
                    .operand_ids
                    .iter()
                    .any(|op| inst_b.operand_ids.contains(op));
                let depends_on_a = inst_b.operand_ids.contains(&inst_a.id)
                    || entry
                        .instructions
                        .get((i + 1)..j)
                        .unwrap_or(&[])
                        .iter()
                        .any(|mid| {
                            mid.operand_ids.contains(&inst_a.id)
                                && inst_b.operand_ids.contains(&mid.id)
                        });
                let bytes_ab = inst_a
                    .shape
                    .as_ref()
                    .map_or(256, shape_bytes_256_aligned)
                    .saturating_add(inst_b.shape.as_ref().map_or(256, shape_bytes_256_aligned));
                if shares_input && !depends_on_a && bytes_ab <= self.vmem_budget_bytes {
                    paired.insert(i);
                    paired.insert(j);
                    pairs.push((i, j));
                    break;
                }
            }
        }

        if pairs.is_empty() {
            return report;
        }

        let by_id: BTreeMap<i64, HloInstructionProto> = entry
            .instructions
            .iter()
            .map(|i| (i.id, i.clone()))
            .collect();
        let first_to_second: BTreeMap<usize, usize> = pairs.iter().copied().collect();
        let second_set: BTreeSet<usize> = pairs.iter().map(|&(_, b)| b).collect();
        let mut new_subcomps = Vec::new();
        let mut out_insts = Vec::with_capacity(entry.instructions.len() + pairs.len() * 2);
        let mut remap: BTreeMap<i64, i64> = BTreeMap::new();

        for (idx, mut inst) in entry.instructions.iter().cloned().enumerate() {
            if second_set.contains(&idx) {
                continue;
            }
            for op_mut in &mut inst.operand_ids {
                if let Some(&rep) = remap.get(op_mut) {
                    *op_mut = rep;
                }
            }
            let Some(&b_idx) = first_to_second.get(&idx) else {
                out_insts.push(inst);
                continue;
            };
            let mut inst_b = entry.instructions.get(b_idx).cloned().unwrap_or_default();
            for op_mut in &mut inst_b.operand_ids {
                if let Some(&rep) = remap.get(op_mut) {
                    *op_mut = rep;
                }
            }

            let mut union_ops: Vec<i64> = Vec::new();
            for &op in inst.operand_ids.iter().chain(inst_b.operand_ids.iter()) {
                if !union_ops.contains(&op) {
                    union_ops.push(op);
                }
            }

            let mut sub_insts = Vec::new();
            let mut param_map = BTreeMap::new();
            let mut sid = 1_i64;
            for (p_ord, &ext_op) in union_ops.iter().enumerate() {
                let pid = sid;
                sid = sid.saturating_add(1);
                sub_insts.push(HloInstructionProto {
                    id: pid,
                    name: format!("mof_param_{p_ord}"),
                    opcode: String::from("parameter"),
                    parameter_number: i64::try_from(p_ord).unwrap_or(0),
                    shape: by_id.get(&ext_op).and_then(|x| x.shape.clone()),
                    ..HloInstructionProto::default()
                });
                param_map.insert(ext_op, pid);
            }

            let a_local = sid;
            sid = sid.saturating_add(1);
            let mut sub_a = inst.clone();
            sub_a.id = a_local;
            sub_a.operand_ids = inst
                .operand_ids
                .iter()
                .map(|op| param_map.get(op).copied().unwrap_or(*op))
                .collect();
            let shape_a = inst.shape.clone().unwrap_or_default();
            sub_insts.push(sub_a);

            let b_local = sid;
            sid = sid.saturating_add(1);
            let mut sub_b = inst_b.clone();
            sub_b.id = b_local;
            sub_b.operand_ids = inst_b
                .operand_ids
                .iter()
                .map(|op| param_map.get(op).copied().unwrap_or(*op))
                .collect();
            let shape_b = inst_b.shape.clone().unwrap_or_default();
            sub_insts.push(sub_b);

            let tup_local = sid;
            let tuple_shape = ShapeProto {
                element_type: PrimitiveType::Tuple as i32,
                tuple_shapes: vec![shape_a.clone(), shape_b.clone()],
                ..ShapeProto::default()
            };
            sub_insts.push(HloInstructionProto {
                id: tup_local,
                name: String::from("mof_root_tuple"),
                opcode: String::from("tuple"),
                shape: Some(tuple_shape.clone()),
                operand_ids: vec![a_local, b_local],
                ..HloInstructionProto::default()
            });

            let sub_cid = next_comp_id;
            next_comp_id = next_comp_id.saturating_add(1);
            new_subcomps.push(HloComputationProto {
                id: sub_cid,
                name: format!("multi_output_fusion_{sub_cid}"),
                instructions: sub_insts,
                root_id: tup_local,
                is_fusion_computation: true,
                ..HloComputationProto::default()
            });

            let mof_id = next_inst_id;
            next_inst_id = next_inst_id.saturating_add(1);
            out_insts.push(HloInstructionProto {
                id: mof_id,
                name: format!("mof_fusion_{mof_id}"),
                opcode: String::from("fusion"),
                fusion_kind: String::from("kLoop"),
                shape: Some(tuple_shape),
                operand_ids: union_ops,
                called_computation_ids: vec![sub_cid],
                ..HloInstructionProto::default()
            });

            let gte0_id = next_inst_id;
            next_inst_id = next_inst_id.saturating_add(1);
            out_insts.push(HloInstructionProto {
                id: gte0_id,
                name: format!("mof_gte0_{gte0_id}"),
                opcode: String::from("get-tuple-element"),
                shape: Some(shape_a),
                operand_ids: vec![mof_id],
                tuple_index: 0,
                ..HloInstructionProto::default()
            });

            let gte1_id = next_inst_id;
            next_inst_id = next_inst_id.saturating_add(1);
            out_insts.push(HloInstructionProto {
                id: gte1_id,
                name: format!("mof_gte1_{gte1_id}"),
                opcode: String::from("get-tuple-element"),
                shape: Some(shape_b),
                operand_ids: vec![mof_id],
                tuple_index: 1,
                ..HloInstructionProto::default()
            });

            remap.insert(inst.id, gte0_id);
            remap.insert(inst_b.id, gte1_id);
            entry.root_id = remap.get(&entry.root_id).copied().unwrap_or(entry.root_id);
            report.multi_output_fusions = report.multi_output_fusions.saturating_add(1);
        }

        entry.instructions = out_insts;
        module
            .computations
            .get_mut(entry_pos)
            .into_iter()
            .for_each(|slot| *slot = entry.clone());
        module.computations.extend(new_subcomps);
        report
    }
}

/// Recomputes cheap elementwise/broadcast values before late uses when peak live bytes exceed
/// `memory_limit_bytes` (`HloRematerializationPass`).
#[derive(Debug, Clone, Copy)]
pub struct HloRematerializationPass {
    /// Target peak live memory budget in bytes.
    pub memory_limit_bytes: u64,
}

impl HloRematerializationPass {
    /// Constructs an [`HloRematerializationPass`] with `memory_limit_bytes`.
    ///
    /// # Arguments
    ///
    /// * `memory_limit_bytes` - Target maximum peak live memory in bytes.
    #[must_use]
    pub const fn new(memory_limit_bytes: u64) -> Self {
        Self { memory_limit_bytes }
    }

    /// Runs live-range analysis and cheap-instruction rematerialization on `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] whose entry computation and schedule are updated.
    pub fn run(&self, module: &mut HloModuleProto) -> HloRematerializationSummaryProto {
        let Some(entry_pos) = resolve_entry_computation_index(module) else {
            return HloRematerializationSummaryProto {
                memory_limit_bytes: self.memory_limit_bytes,
                ..HloRematerializationSummaryProto::default()
            };
        };

        let mut comp = module
            .computations
            .get(entry_pos)
            .cloned()
            .unwrap_or_default();
        let peak_before = compute_peak_live_bytes(&comp);
        let entries = if peak_before > self.memory_limit_bytes {
            rematerialize_cheap_chains(&mut comp, self.memory_limit_bytes)
        } else {
            Vec::new()
        };

        let root_id = comp.root_id;
        let used_ids: BTreeSet<i64> = comp
            .instructions
            .iter()
            .flat_map(|i| i.operand_ids.iter().copied())
            .chain(std::iter::once(root_id))
            .collect();
        comp.instructions
            .retain(|i| i.opcode == "parameter" || used_ids.contains(&i.id));

        let peak_after = compute_peak_live_bytes(&comp);
        let comp_name = comp.name.clone();
        let comp_id = comp.id;
        let seq_ids: Vec<i64> = comp.instructions.iter().map(|i| i.id).collect();
        module
            .computations
            .get_mut(entry_pos)
            .into_iter()
            .for_each(|slot| *slot = comp.clone());
        let sched = module
            .schedule
            .get_or_insert_with(HloScheduleProto::default);
        sched.sequences.insert(
            comp_id,
            HloInstructionSequenceProto {
                instruction_ids: seq_ids,
            },
        );

        HloRematerializationSummaryProto {
            computation_name: comp_name,
            memory_limit_bytes: self.memory_limit_bytes,
            peak_memory_before_bytes: peak_before,
            peak_memory_after_bytes: peak_after,
            rematerialized_count: u32::try_from(entries.len()).unwrap_or(0),
            entries,
        }
    }
}

/// Clones cheap pure instructions immediately prior to late uses until peak live memory meets `limit_bytes`.
///
/// # Arguments
///
/// * `comp` - Mutable computation to transform.
/// * `limit_bytes` - Target peak live memory budget in bytes.
fn rematerialize_cheap_chains(
    comp: &mut HloComputationProto,
    limit_bytes: u64,
) -> Vec<RematerializedInstructionEntryProto> {
    let mut entries = Vec::new();
    let mut next_id = comp
        .instructions
        .iter()
        .map(|i| i.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    let mut idx = 0_usize;
    while idx < comp.instructions.len() {
        if compute_peak_live_bytes(comp) <= limit_bytes {
            break;
        }
        let cand = comp.instructions.get(idx).cloned().unwrap_or_default();
        if !is_rematerializable_opcode(&cand.opcode) {
            idx = idx.saturating_add(1);
            continue;
        }
        let last_user_pos = comp
            .instructions
            .iter()
            .enumerate()
            .skip(idx + 2)
            .filter(|(_, u)| u.operand_ids.contains(&cand.id))
            .map(|(p, _)| p)
            .max();
        let Some(u_pos) = last_user_pos else {
            idx = idx.saturating_add(1);
            continue;
        };

        let clone_id = next_id;
        next_id = next_id.saturating_add(1);
        let clone_name = format!("{}.remat.{clone_id}", cand.name);
        let saved = cand.shape.as_ref().map_or(256, shape_bytes_256_aligned);
        let mut cloned = cand.clone();
        cloned.id = clone_id;
        cloned.name.clone_from(&clone_name);

        comp.instructions
            .get_mut(u_pos)
            .into_iter()
            .for_each(|user_mut| {
                for op_mut in &mut user_mut.operand_ids {
                    if *op_mut == cand.id {
                        *op_mut = clone_id;
                    }
                }
            });
        comp.instructions.insert(u_pos, cloned);

        entries.push(RematerializedInstructionEntryProto {
            original_instruction_name: cand.name,
            clone_instruction_name: clone_name,
            original_bundle_pc: u32::try_from(idx).unwrap_or(0),
            rematerialized_bundle_pc: u32::try_from(u_pos).unwrap_or(0),
            saved_bytes: saved,
            recompute_cost_cycles: 4,
        });
        idx = idx.saturating_add(1);
    }
    entries
}

/// Resolves the index of the entry computation inside `module.computations`.
///
/// # Arguments
///
/// * `module` - [`HloModuleProto`] to query.
fn resolve_entry_computation_index(module: &HloModuleProto) -> Option<usize> {
    module
        .computations
        .iter()
        .position(|c| c.id == module.entry_computation_id)
        .or_else(|| (!module.computations.is_empty()).then_some(0))
}

/// Computes the peak simultaneously live tensor bytes across `comp.instructions`.
///
/// # Arguments
///
/// * `comp` - Computation whose sequential live memory watermark is evaluated.
#[must_use]
pub fn compute_peak_live_bytes(comp: &HloComputationProto) -> u64 {
    let n = comp.instructions.len();
    let mut last_use: BTreeMap<i64, usize> = BTreeMap::new();
    for (pos, inst) in comp.instructions.iter().enumerate() {
        last_use.insert(inst.id, pos);
        for &op_id in &inst.operand_ids {
            last_use.insert(op_id, pos);
        }
        if inst.id == comp.root_id {
            last_use.insert(inst.id, n.saturating_sub(1));
        }
    }
    let sizes: BTreeMap<i64, u64> = comp
        .instructions
        .iter()
        .map(|i| (i.id, i.shape.as_ref().map_or(256, shape_bytes_256_aligned)))
        .collect();

    let mut live_ids: BTreeSet<i64> = BTreeSet::new();
    let mut cur_bytes = 0_u64;
    let mut peak_bytes = 0_u64;

    for (pos, inst) in comp.instructions.iter().enumerate() {
        live_ids.insert(inst.id);
        cur_bytes = cur_bytes.saturating_add(sizes.get(&inst.id).copied().unwrap_or(256));
        peak_bytes = peak_bytes.max(cur_bytes);

        let expired: Vec<i64> = live_ids
            .iter()
            .copied()
            .filter(|id| last_use.get(id).copied().unwrap_or(pos) <= pos)
            .collect();
        for exp_id in expired {
            live_ids.remove(&exp_id);
            cur_bytes = cur_bytes.saturating_sub(sizes.get(&exp_id).copied().unwrap_or(256));
        }
    }

    peak_bytes
}

/// Counts consumer operand uses of every instruction in `comp`.
///
/// # Arguments
///
/// * `comp` - Computation to inspect.
fn count_instruction_uses(comp: &HloComputationProto) -> BTreeMap<i64, usize> {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for inst in &comp.instructions {
        for &op in &inst.operand_ids {
            let entry = counts.entry(op).or_insert(0);
            *entry = (*entry).saturating_add(1);
        }
    }
    if comp.root_id != 0 {
        let root_entry = counts.entry(comp.root_id).or_insert(0);
        *root_entry = (*root_entry).saturating_add(1);
    }
    counts
}

/// Returns `true` if `opcode` is a pure producer eligible to be absorbed into a TPU fusion.
///
/// # Arguments
///
/// * `opcode` - HLO opcode mnemonic.
fn is_fusable_producer(opcode: &str) -> bool {
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
            | "logistic"
            | "broadcast"
            | "convert"
            | "clamp"
            | "select"
    )
}

/// Returns `true` if `opcode` can root a loop or output fusion computation.
///
/// # Arguments
///
/// * `opcode` - HLO opcode mnemonic.
fn is_fusable_consumer(opcode: &str) -> bool {
    is_fusable_producer(opcode) || matches!(opcode, "dot" | "convolution" | "reduce")
}

/// Returns `true` if `opcode` is cheap and pure enough for HLO rematerialization.
///
/// # Arguments
///
/// * `opcode` - HLO opcode mnemonic.
fn is_rematerializable_opcode(opcode: &str) -> bool {
    matches!(opcode, "constant" | "iota" | "broadcast") || is_fusable_producer(opcode)
}

/// Computes the 256-byte-aligned physical byte footprint of `shape`.
///
/// # Arguments
///
/// * `shape` - Tensor [`ShapeProto`].
pub fn shape_bytes_256_aligned(shape: &ShapeProto) -> u64 {
    if !shape.tuple_shapes.is_empty() {
        return shape.tuple_shapes.iter().map(shape_bytes_256_aligned).sum();
    }
    let bits = u64::try_from(primitive_type_bit_width(shape.element_type).max(4)).unwrap_or(32);
    let elems = shape.dimensions.iter().fold(1_u64, |acc, &d| {
        acc.saturating_mul(u64::try_from(d.max(1)).unwrap_or(1))
    });
    let raw_bytes = elems.saturating_mul(bits).div_ceil(8).max(256);
    raw_bytes.div_ceil(256) * 256
}
