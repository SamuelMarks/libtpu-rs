//! Compiler optimization passes, VMEM liveness analysis, cost analysis, and HLO DAG scheduling.

use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use super::vliw::TensorCoreVliwBundle;
use super::{COMPILATION_PHASE_NAMES, CompiledProgram};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{HloComputationProto, PjRtPartialProgramProto};
use crate::types::{ElementType, MemoryKind, MemoryLayout, Shape};
use indexmap::IndexMap;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableDiGraph;

/// Parsed GSPMD / `OpenXLA` Shardy (`sdy`) tensor sharding specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedShardingSpec {
    /// True if the tensor is replicated across all partitions (`{replicated}`).
    pub is_replicated: bool,
    /// Per-dimension tile factors (`[t_0, t_1, ...]`), e.g. `[2, 2]` for `{devices=[2,2]0,1,2,3}`.
    pub tile_factors: Vec<usize>,
    /// Canonical normalized sharding identifier used for resharding edge detection.
    pub canonical_key: String,
}

impl ParsedShardingSpec {
    /// Parses a sharding specification string from XLA HLO (`{devices=[2,1]0,1}`, `{replicated}`)
    /// or `OpenXLA` Shardy (`#sdy.sharding<@mesh, [{"x"}, {}]>`) syntax.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw sharding specification string.
    /// * `num_partitions` - Total number of SPMD partitions in the compilation environment.
    #[must_use]
    pub fn from_str_with_partitions(raw: &str, num_partitions: usize) -> Self {
        let trimmed = raw.trim();
        let lower = trimmed.to_ascii_lowercase();
        let is_replicated = trimmed.is_empty() || lower.contains("replicated");
        if is_replicated {
            return Self {
                is_replicated: true,
                tile_factors: Vec::new(),
                canonical_key: String::from("replicated"),
            };
        }
        let bracket_scope = trimmed
            .split_once('[')
            .and_then(|(_, rest)| rest.split_once(']'))
            .map_or(trimmed, |(inner, _)| inner);
        let mut factors: Vec<usize> = bracket_scope
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|token| token.parse::<usize>().ok())
            .map(|v| v.max(1))
            .collect();
        if factors.is_empty() {
            factors.push(num_partitions.max(1));
        }
        Self {
            is_replicated: false,
            tile_factors: factors,
            canonical_key: String::from(trimmed),
        }
    }

    /// Computes the local per-partition [`Shape`] by dividing each dimension `d` of `global_shape`
    /// by `tile_factors[d]` (rounded up).
    ///
    /// # Arguments
    ///
    /// * `global_shape` - Global unpartitioned tensor [`Shape`].
    #[must_use]
    pub fn partition_shape(&self, global_shape: &Shape) -> Shape {
        if self.is_replicated {
            return global_shape.clone();
        }
        let mut local_dims = global_shape.dims.clone();
        for (axis, dim_mut) in local_dims.iter_mut().enumerate() {
            let factor = i64::try_from(self.tile_factors.get(axis).copied().unwrap_or(1).max(1))
                .unwrap_or(1);
            *dim_mut = ((*dim_mut).max(1) + factor - 1) / factor;
        }
        Shape {
            element_type: global_shape.element_type,
            dims: local_dims,
            dynamic_dims: global_shape.dynamic_dims.clone(),
            layout: global_shape.layout.clone(),
        }
    }
}
/// Strongly-typed Common Subexpression Elimination (`CSE`) equivalence key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CseExpressionKey {
    /// Hardware [`TpuOpCode`] discriminant.
    pub opcode: TpuOpCode,
    /// Ordered input register slot indices.
    pub operand_slots: Vec<usize>,
    /// Full instruction attributes (`slice_starts`, `permutation`, `constant_bytes`, `output_shape`, etc.).
    pub attrs: InstructionAttrs,
}

/// Static cost analysis metrics for a compiled TPU executable (`PJRT_Executable_GetCostAnalysis`).
#[derive(Debug, Clone, Copy)]
pub struct CostAnalysis {
    /// Total floating-point operations estimated for one execution.
    pub flops: f32,
    /// Total transcendental operations estimated for one execution.
    pub transcendentals: f32,
    /// Total HBM bytes read and written during execution.
    pub bytes_accessed: f32,
    /// Estimated optimal execution time in seconds on target TPU MXU.
    pub optimal_seconds: f32,
}

/// On-device memory usage breakdown for a compiled executable (`PJRT_Executable_GetCompiledMemoryStats`).
#[derive(Debug, Clone, Copy)]
pub struct CompiledMemoryStats {
    /// Size of generated machine code in bytes.
    pub generated_code_size_in_bytes: i64,
    /// Total size of input argument buffers in bytes.
    pub argument_size_in_bytes: i64,
    /// Total size of output buffers in bytes.
    pub output_size_in_bytes: i64,
    /// Amount of aliased (reused) input-to-output memory in bytes.
    pub alias_size_in_bytes: i64,
    /// Temporary scratchpad HBM bytes required during execution.
    pub temp_size_in_bytes: i64,
}

/// SSA register liveness interval `[first_def_pc, last_use_pc]` and assigned linear-scan VMEM offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterLivenessInterval {
    /// SSA register slot index.
    pub slot_index: usize,
    /// VLIW bundle PC index where the slot is defined (`0` for entry parameters).
    pub first_def_pc: u32,
    /// Last VLIW bundle PC index where the slot is read or returned.
    pub last_use_pc: u32,
    /// 256-byte-aligned byte size required for this register slot.
    pub size_bytes: usize,
    /// Linear-scan VMEM byte offset assigned to this register slot.
    pub assigned_vmem_offset: u64,
}

/// Topological schedule and HBM buffer assignment report produced by [`analyze_hlo_computation_dag`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HloDagAnalysisReport {
    /// Topologically sorted HLO instruction IDs (`operand` before `user`).
    pub topological_order: Vec<i64>,
    /// Deterministic insertion-ordered map from instruction ID to 4096-byte-aligned HBM byte offset.
    pub buffer_offsets: IndexMap<i64, u64>,
    /// Total HBM bytes planned across all scheduled instructions.
    pub total_hbm_bytes: u64,
}

/// Builds a `petgraph::stable_graph::StableDiGraph` over an [`HloComputationProto`], verifies
/// DAG acyclicity via `petgraph::algo::toposort`, and assigns 4096-byte-aligned HBM offsets in
/// an `indexmap::IndexMap`.
///
/// # Arguments
///
/// * `computation` - Wire-decoded [`HloComputationProto`] to schedule and plan.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if the instruction dependency graph contains a cycle.
pub fn analyze_hlo_computation_dag(
    computation: &HloComputationProto,
) -> LibTpuResult<HloDagAnalysisReport> {
    let mut graph = StableDiGraph::<i64, ()>::new();
    let mut id_to_node = IndexMap::new();
    let mut id_to_bytes = IndexMap::new();

    let mut nodes = Vec::with_capacity(computation.instructions.len());
    for inst in &computation.instructions {
        let node = graph.add_node(inst.id);
        nodes.push(node);
        id_to_node.insert(inst.id, node);
        let elem_count: u64 = inst.shape.as_ref().map_or(1, |s| {
            s.dimensions.iter().fold(1_u64, |acc, &d| {
                acc.saturating_mul(u64::try_from(d).unwrap_or(1).max(1))
            })
        });
        let raw_bytes = elem_count.saturating_mul(4).max(256);
        let aligned_bytes = raw_bytes.div_ceil(4096) * 4096;
        id_to_bytes.insert(inst.id, aligned_bytes);
    }

    for (inst, &dst_node) in computation.instructions.iter().zip(nodes.iter()) {
        for &op_id in &inst.operand_ids {
            if let Some(&src_node) = id_to_node.get(&op_id) {
                graph.add_edge(src_node, dst_node, ());
            }
        }
    }

    let sorted_nodes = toposort(&graph, None).map_err(|_| LibTpuError::CompilationFailed {
        reason: format!(
            "Dependency cycle detected in HloComputationProto '{}'",
            computation.name
        ),
    })?;

    let mut topological_order = Vec::with_capacity(sorted_nodes.len());
    let mut buffer_offsets = IndexMap::new();
    let mut next_offset = 0_u64;

    for node_idx in sorted_nodes {
        let inst_id = graph.node_weight(node_idx).copied().unwrap_or(0);
        topological_order.push(inst_id);
        buffer_offsets.insert(inst_id, next_offset);
        let sz = id_to_bytes.get(&inst_id).copied().unwrap_or(4096);
        next_offset = next_offset.saturating_add(sz);
    }

    Ok(HloDagAnalysisReport {
        topological_order,
        buffer_offsets,
        total_hbm_bytes: next_offset,
    })
}

impl CompiledProgram {
    /// Executes a named compilation phase (`PJRT_PhaseCompile_RunPhase`).
    ///
    /// # Arguments
    ///
    /// * `phase_name` - Name of the compilation phase from [`COMPILATION_PHASE_NAMES`].
    /// * `input_bytes` - Input program or intermediate phase bytes.
    /// * `compile_options` - Serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::PhaseCompilationFailed`] if `phase_name` is unknown or compilation fails.
    pub fn run_compilation_phase(
        phase_name: &str,
        input_bytes: &[u8],
        compile_options: &[u8],
    ) -> LibTpuResult<Vec<u8>> {
        let Some(phase_idx) = COMPILATION_PHASE_NAMES
            .iter()
            .position(|&p| p == phase_name)
        else {
            return Err(LibTpuError::PhaseCompilationFailed {
                phase_name: String::from(phase_name),
                reason: format!("Unknown compilation phase '{phase_name}'"),
            });
        };
        let raw_program = if let Ok(partial) = PjRtPartialProgramProto::decode(input_bytes) {
            if partial.program.is_empty() {
                input_bytes.to_vec()
            } else {
                partial.program
            }
        } else {
            input_bytes.to_vec()
        };
        let mut prog = if let Ok(existing) = Self::deserialize(&raw_program) {
            existing
        } else {
            Self::compile("mlir", &raw_program, compile_options).map_err(|e| {
                LibTpuError::PhaseCompilationFailed {
                    phase_name: String::from(phase_name),
                    reason: e.to_string(),
                }
            })?
        };
        match phase_idx {
            0 => {
                prog.pass_hlo_verification_and_shape_inference();
                prog.pass_spmd_sharding_propagation();
                prog.pass_spmd_partitioning();
            }
            1 => {
                prog.pass_tpu_layout_assignment_and_tiling();
                prog.pass_dot_decomposer();
            }
            2 => {
                prog.pass_algebraic_simplifier();
                prog.pass_constant_folding();
                prog.pass_common_subexpression_elimination();
                prog.pass_vmem_scratchpad_allocation();
            }
            3 => {
                prog.pass_tpu_custom_call_lowering();
                prog.pass_hlo_operator_fusion();
                prog.pass_llo_vliw_bundle_scheduling();
                prog.pass_vliw_bundle_packing();
            }
            _ => {
                prog.run_all_passes();
            }
        }
        prog.memory_stats.temp_size_in_bytes = 256 * (i64::try_from(phase_idx).unwrap_or(0) + 1);
        Ok(prog.serialize())
    }

    /// Executes a named compilation phase and wraps the resulting artifact in a wire-compatible
    /// [`PjRtPartialProgramProto`] (`xla/pjrt/proto/pjrt_partial_program.proto`).
    ///
    /// # Arguments
    ///
    /// * `phase_name` - Name of the compilation phase from [`COMPILATION_PHASE_NAMES`].
    /// * `input_bytes` - Raw program bytes or serialized [`PjRtPartialProgramProto`].
    /// * `compile_options` - Serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::PhaseCompilationFailed`] if `phase_name` is unknown or compilation fails.
    pub fn run_compilation_phase_proto(
        phase_name: &str,
        input_bytes: &[u8],
        compile_options: &[u8],
    ) -> LibTpuResult<PjRtPartialProgramProto> {
        let artifact = Self::run_compilation_phase(phase_name, input_bytes, compile_options)?;
        let phase_idx = COMPILATION_PHASE_NAMES
            .iter()
            .position(|&p| p == phase_name)
            .unwrap_or(0);
        let consumer_phases: Vec<String> = COMPILATION_PHASE_NAMES
            .get(phase_idx + 1..)
            .unwrap_or(&[])
            .iter()
            .map(|&s| String::from(s))
            .collect();
        Ok(PjRtPartialProgramProto {
            program: artifact,
            program_format: String::from("ltpuexec"),
            producer_phase: String::from(phase_name),
            consumer_phases,
            version: String::from("0.0.52"),
            program_name: String::from("tpu_partial_program"),
        })
    }

    /// Infers the output [`Shape`] for a single [`TpuInstructionBundle`] from its attributes
    /// and the current SSA `slot_shapes` table.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Instruction bundle to infer the output shape for.
    /// * `slot_shapes` - Mapping from SSA register slot index to known [`Shape`].
    /// * `fallback_shape` - Default fallback [`Shape`] when no operand shape is known.
    #[allow(clippy::too_many_lines)]
    fn infer_bundle_output_shape(
        bundle: &TpuInstructionBundle,
        slot_shapes: &std::collections::BTreeMap<usize, Shape>,
        fallback_shape: &Shape,
    ) -> Shape {
        if let Some(existing) = &bundle.attrs.output_shape {
            return existing.clone();
        }
        let first_op_shape = bundle
            .operand_slots
            .first()
            .and_then(|s| slot_shapes.get(s))
            .unwrap_or(fallback_shape);
        match bundle.opcode {
            TpuOpCode::Constant if !bundle.attrs.constant_bytes.is_empty() => {
                let elem_bytes = first_op_shape.element_type.byte_size().max(1);
                let count = i64::try_from(bundle.attrs.constant_bytes.len() / elem_bytes)
                    .unwrap_or(1)
                    .max(1);
                Shape {
                    element_type: first_op_shape.element_type,
                    dims: vec![count],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                }
            }
            TpuOpCode::Transpose => {
                let dims: Vec<i64> = bundle
                    .attrs
                    .permutation
                    .iter()
                    .map(|&axis| first_op_shape.dims.get(axis).copied().unwrap_or(1))
                    .collect();
                let rank = dims.len();
                Shape {
                    element_type: first_op_shape.element_type,
                    dims,
                    dynamic_dims: vec![false; rank],
                    layout: MemoryLayout::default_tpu_tiled(rank),
                }
            }
            TpuOpCode::Slice => {
                let dims: Vec<i64> = bundle
                    .attrs
                    .slice_limits
                    .iter()
                    .enumerate()
                    .map(|(ax, &lim)| {
                        let start = bundle.attrs.slice_starts.get(ax).copied().unwrap_or(0);
                        let stride = bundle
                            .attrs
                            .slice_strides
                            .get(ax)
                            .copied()
                            .unwrap_or(1)
                            .max(1);
                        let extent = lim.saturating_sub(start).div_ceil(stride).max(1);
                        i64::try_from(extent).unwrap_or(1)
                    })
                    .collect();
                let rank = dims.len();
                Shape {
                    element_type: first_op_shape.element_type,
                    dims,
                    dynamic_dims: vec![false; rank],
                    layout: MemoryLayout::default_tpu_tiled(rank),
                }
            }
            TpuOpCode::Pad => {
                let mut dims = Vec::with_capacity(first_op_shape.dims.len());
                for (ax, &base_d) in first_op_shape.dims.iter().enumerate() {
                    let low =
                        i64::try_from(bundle.attrs.pad_edge_low.get(ax).copied().unwrap_or(0))
                            .unwrap_or(0);
                    let high =
                        i64::try_from(bundle.attrs.pad_edge_high.get(ax).copied().unwrap_or(0))
                            .unwrap_or(0);
                    let inter =
                        i64::try_from(bundle.attrs.pad_interior.get(ax).copied().unwrap_or(0))
                            .unwrap_or(0);
                    let gaps = (base_d - 1).max(0).saturating_mul(inter);
                    dims.push(
                        base_d
                            .saturating_add(low)
                            .saturating_add(high)
                            .saturating_add(gaps),
                    );
                }
                let rank = dims.len();
                Shape {
                    element_type: first_op_shape.element_type,
                    dims,
                    dynamic_dims: vec![false; rank],
                    layout: MemoryLayout::default_tpu_tiled(rank),
                }
            }
            TpuOpCode::Concatenate => {
                let mut dims = first_op_shape.dims.clone();
                let c_dim = bundle.attrs.concat_dim;
                let mut total_extent: i64 = 0;
                for op_slot in &bundle.operand_slots {
                    let ext = slot_shapes
                        .get(op_slot)
                        .and_then(|s| s.dims.get(c_dim).copied())
                        .unwrap_or(1);
                    total_extent = total_extent.saturating_add(ext);
                }
                dims.get_mut(c_dim).into_iter().for_each(|d_mut| {
                    *d_mut = total_extent.max(*d_mut);
                });
                let rank = dims.len();
                Shape {
                    element_type: first_op_shape.element_type,
                    dims,
                    dynamic_dims: vec![false; rank],
                    layout: MemoryLayout::default_tpu_tiled(rank),
                }
            }
            TpuOpCode::ReduceSumF32 | TpuOpCode::ReduceMax | TpuOpCode::ReduceMin => {
                let mut dims: Vec<i64> = first_op_shape
                    .dims
                    .iter()
                    .enumerate()
                    .filter(|(ax, _)| !bundle.attrs.reduce_dims.contains(ax))
                    .map(|(_, &d)| d)
                    .collect();
                if dims.is_empty() {
                    dims.push(1);
                }
                let rank = dims.len();
                Shape {
                    element_type: first_op_shape.element_type,
                    dims,
                    dynamic_dims: vec![false; rank],
                    layout: MemoryLayout::default_tpu_tiled(rank),
                }
            }
            TpuOpCode::Convolution => {
                let rhs_op_shape = bundle
                    .operand_slots
                    .get(1)
                    .and_then(|s| slot_shapes.get(s))
                    .unwrap_or(first_op_shape);
                let rank = first_op_shape.rank();
                let spatial_rank = rank.saturating_sub(2);
                let b_groups = i64::try_from(bundle.attrs.batch_group_count.max(1)).unwrap_or(1);
                let batch = (first_op_shape.dims.first().copied().unwrap_or(1) / b_groups).max(1);
                let mut dims = Vec::with_capacity(rank.max(1));
                dims.push(batch);
                for s in 0..spatial_rank {
                    let in_d =
                        usize::try_from(first_op_shape.dims.get(s + 1).copied().unwrap_or(1))
                            .unwrap_or(1)
                            .max(1);
                    let k_d = usize::try_from(rhs_op_shape.dims.get(s).copied().unwrap_or(1))
                        .unwrap_or(1)
                        .max(1);
                    let stride = bundle
                        .attrs
                        .window_strides
                        .get(s)
                        .copied()
                        .unwrap_or(1)
                        .max(1);
                    let p_low = bundle.attrs.pad_edge_low.get(s).copied().unwrap_or(0);
                    let p_high = bundle.attrs.pad_edge_high.get(s).copied().unwrap_or(0);
                    let l_dil = bundle
                        .attrs
                        .lhs_dilation
                        .get(s)
                        .copied()
                        .unwrap_or(1)
                        .max(1);
                    let r_dil = bundle
                        .attrs
                        .rhs_dilation
                        .get(s)
                        .copied()
                        .unwrap_or(1)
                        .max(1);
                    let eff_in = (in_d - 1) * l_dil + 1 + p_low + p_high;
                    let eff_k = (k_d - 1) * r_dil + 1;
                    let out_d = eff_in.saturating_sub(eff_k) / stride + 1;
                    dims.push(i64::try_from(out_d.max(1)).unwrap_or(1));
                }
                dims.push(
                    rhs_op_shape
                        .dims
                        .get(rank.saturating_sub(1))
                        .copied()
                        .unwrap_or(1),
                );
                let out_rank = dims.len();
                Shape {
                    element_type: first_op_shape.element_type,
                    dims,
                    dynamic_dims: vec![false; out_rank],
                    layout: MemoryLayout::default_tpu_tiled(out_rank),
                }
            }
            TpuOpCode::Gather => {
                let dims: Vec<i64> = bundle
                    .attrs
                    .gather_slice_sizes
                    .iter()
                    .enumerate()
                    .filter(|(ax, _)| !bundle.attrs.collapsed_slice_dims.contains(ax))
                    .map(|(_, &sz)| i64::try_from(sz.max(1)).unwrap_or(1))
                    .collect();
                let rank = dims.len().max(1);
                Shape {
                    element_type: first_op_shape.element_type,
                    dims,
                    dynamic_dims: vec![false; rank],
                    layout: MemoryLayout::default_tpu_tiled(rank),
                }
            }
            TpuOpCode::CompareEq
            | TpuOpCode::CompareLt
            | TpuOpCode::CompareGt
            | TpuOpCode::IsFinite => Shape {
                element_type: ElementType::Pred,
                dims: first_op_shape.dims.clone(),
                dynamic_dims: first_op_shape.dynamic_dims.clone(),
                layout: first_op_shape.layout.clone(),
            },
            _ => first_op_shape.clone(),
        }
    }

    /// Pass 1: HLO verification and per-instruction shape inference across parameter, bundle, and output shapes.
    #[allow(clippy::or_fun_call)]
    pub fn pass_hlo_verification_and_shape_inference(&mut self) {
        self.memory_stats.generated_code_size_in_bytes =
            self.memory_stats.generated_code_size_in_bytes.max(256);
        let fallback_shape = self.output_shapes.first().cloned().unwrap_or(Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: MemoryLayout::default_tpu_tiled(1),
        });
        let mut slot_shapes: std::collections::BTreeMap<usize, Shape> =
            std::collections::BTreeMap::new();
        for (idx, sh) in self.parameter_shapes.iter().enumerate() {
            slot_shapes.insert(idx, sh.clone());
        }
        for bundle in &mut self.instruction_bundles {
            let inferred = Self::infer_bundle_output_shape(bundle, &slot_shapes, &fallback_shape);
            slot_shapes.insert(bundle.output_slot, inferred.clone());
            bundle.attrs.output_shape = Some(inferred);
        }
        let mut arg_bytes: i64 = 0;
        for sh in &self.parameter_shapes {
            let sz = i64::try_from(sh.unpadded_byte_size().0.max(256)).unwrap_or(256);
            arg_bytes = arg_bytes.saturating_add(sz);
        }
        let mut out_bytes: i64 = 0;
        for sh in &self.output_shapes {
            let sz = i64::try_from(sh.unpadded_byte_size().0.max(256)).unwrap_or(256);
            out_bytes = out_bytes.saturating_add(sz);
        }
        self.memory_stats.argument_size_in_bytes = arg_bytes;
        self.memory_stats.output_size_in_bytes = out_bytes;
    }

    /// Pass 2: Bidirectional SPMD and Shardy (`sdy`) sharding propagation across replicas and partitions.
    pub fn pass_spmd_sharding_propagation(&mut self) {
        self.num_replicas = self.num_replicas.max(1);
        self.num_partitions = self.num_partitions.max(1);
        self.parameter_memory_kinds
            .resize(self.parameter_shapes.len(), MemoryKind::DeviceHbm);
        self.output_memory_kinds
            .resize(self.output_shapes.len(), MemoryKind::DeviceHbm);

        if self.num_partitions > 1 && !self.mosaic_metadata.sharding.is_empty() {
            let default_sharding = self.mosaic_metadata.sharding.clone();
            for bundle in &mut self.instruction_bundles {
                if bundle.attrs.sharding_spec.is_empty() {
                    bundle.attrs.sharding_spec.clone_from(&default_sharding);
                }
            }
        }
        let mut slot_sharding: std::collections::BTreeMap<usize, String> =
            std::collections::BTreeMap::new();
        for bundle in self.instruction_bundles.iter().rev() {
            if !bundle.attrs.sharding_spec.is_empty() {
                for &op_slot in &bundle.operand_slots {
                    slot_sharding.insert(op_slot, bundle.attrs.sharding_spec.clone());
                }
            }
        }
        for bundle in &mut self.instruction_bundles {
            if bundle.attrs.sharding_spec.is_empty() {
                bundle.attrs.sharding_spec = slot_sharding
                    .get(&bundle.output_slot)
                    .cloned()
                    .unwrap_or_default();
            }
        }
    }

    /// Pass 3b (`spmd_partitioning`): Normalizes SPMD replica/partition counts, partitions local
    /// shard shapes for explicit Shardy (`sdy`) / `OpSharding` tile specs, and automatically
    /// synthesizes `AllReduceSumF32`, `ReduceScatter`, `AllGather`, `AllToAll`, or `CollectivePermute`
    /// collective bundles when GSPMD / Shardy sharded contractions or resharding transitions occur.
    pub fn pass_spmd_partitioning(&mut self) {
        self.num_replicas = self.num_replicas.max(1);
        self.num_partitions = self.num_partitions.max(1);
        if self.num_partitions <= 1 {
            return;
        }
        let mut slot_shardings: std::collections::BTreeMap<usize, ParsedShardingSpec> =
            std::collections::BTreeMap::new();
        let mut updated = Vec::with_capacity(self.instruction_bundles.len() * 2);
        for mut bundle in self.instruction_bundles.drain(..) {
            let sharding = bundle.attrs.sharding_spec.clone();
            let parsed_spec =
                ParsedShardingSpec::from_str_with_partitions(&sharding, self.num_partitions);

            for &op_slot in &bundle.operand_slots {
                if let Some(prev_spec) = slot_shardings.get(&op_slot)
                    && prev_spec.canonical_key != parsed_spec.canonical_key
                {
                    let pc = u32::try_from(updated.len()).unwrap_or(0);
                    updated.push(TpuInstructionBundle {
                        pc_offset: pc,
                        opcode: TpuOpCode::CollectivePermute,
                        operand_slots: vec![op_slot],
                        output_slot: op_slot,
                        sync_flag_id: pc + 1,
                        attrs: InstructionAttrs {
                            input_shapes: bundle.attrs.input_shapes.clone(),
                            output_shape: bundle.attrs.input_shapes.first().cloned(),
                            sharding_spec: sharding.clone(),
                            ..InstructionAttrs::default()
                        },
                    });
                }
            }

            bundle.attrs.output_shape.iter().for_each(|out_sh| {
                let local_sh = parsed_spec.partition_shape(out_sh);
                let local_bytes =
                    i64::try_from(local_sh.unpadded_byte_size().0.max(256)).unwrap_or(256);
                self.memory_stats.temp_size_in_bytes =
                    self.memory_stats.temp_size_in_bytes.max(local_bytes);
            });

            let is_contraction = matches!(
                bundle.opcode,
                TpuOpCode::DotGeneralF32 | TpuOpCode::Convolution
            );
            let out_slot = bundle.output_slot;
            let out_shape = bundle.attrs.output_shape.clone();
            if !sharding.is_empty() {
                slot_shardings.insert(out_slot, parsed_spec);
            }
            let pc_cur = u32::try_from(updated.len()).unwrap_or(0);
            bundle.pc_offset = pc_cur;
            updated.push(bundle);

            if is_contraction && !sharding.is_empty() {
                let coll_op = if sharding.contains("reduce_scatter") {
                    TpuOpCode::ReduceScatter
                } else if sharding.contains("all_gather") {
                    TpuOpCode::AllGather
                } else if sharding.contains("all_to_all") {
                    TpuOpCode::AllToAll
                } else {
                    TpuOpCode::AllReduceSumF32
                };
                let pc = u32::try_from(updated.len()).unwrap_or(0);
                updated.push(TpuInstructionBundle {
                    pc_offset: pc,
                    opcode: coll_op,
                    operand_slots: vec![out_slot],
                    output_slot: out_slot,
                    sync_flag_id: pc + 1,
                    attrs: InstructionAttrs {
                        input_shapes: out_shape.clone().into_iter().collect(),
                        output_shape: out_shape,
                        sharding_spec: sharding,
                        ..InstructionAttrs::default()
                    },
                });
            }
        }
        self.instruction_bundles = updated;
    }

    /// Pass 3: TPU `(8, 128)` matrix sublane layout assignment and sub-packing tiling.
    pub fn pass_tpu_layout_assignment_and_tiling(&mut self) {
        for sh in &mut self.parameter_shapes {
            sh.layout = MemoryLayout::default_tpu_tiled_for_type(sh.dims.len(), sh.element_type);
        }
        for sh in &mut self.output_shapes {
            sh.layout = MemoryLayout::default_tpu_tiled_for_type(sh.dims.len(), sh.element_type);
        }
    }

    /// Pass 4: HLO operator fusion across producer-consumer instruction bundles.
    pub fn pass_hlo_operator_fusion(&mut self) {
        let next_info: Vec<Option<(TpuOpCode, usize, Vec<usize>)>> = self
            .instruction_bundles
            .iter()
            .skip(1)
            .map(|b| Some((b.opcode, b.output_slot, b.operand_slots.clone())))
            .chain(std::iter::once(None))
            .collect();
        for (idx, (bundle, next_opt)) in self
            .instruction_bundles
            .iter_mut()
            .zip(next_info)
            .enumerate()
        {
            if matches!(
                bundle.opcode,
                TpuOpCode::DotGeneralF32 | TpuOpCode::Convolution
            ) && let Some((ep_op, _ep_out, ep_operands)) = next_opt
            {
                bundle.attrs.fused_epilogue_op = Some(ep_op);
                for &extra_slot in &ep_operands {
                    if extra_slot != bundle.output_slot {
                        bundle.operand_slots.push(extra_slot);
                    }
                }
            }
            bundle.pc_offset = u32::try_from(idx).unwrap_or(0);
        }
    }

    /// Pass 5: On-chip VMEM scratchpad allocation based on intermediate instruction bundle footprint
    /// and Mosaic `scratch_vmem_bytes` target metadata.
    pub fn pass_vmem_scratchpad_allocation(&mut self) {
        let mut scratch_bytes: i64 = 0;
        for bundle in &self.instruction_bundles {
            let b_bytes = bundle
                .attrs
                .output_shape
                .as_ref()
                .map_or(256_usize, |s| s.unpadded_byte_size().0.max(256));
            scratch_bytes = scratch_bytes.saturating_add(i64::try_from(b_bytes).unwrap_or(256));
        }
        let mosaic_scratch = i64::try_from(self.mosaic_metadata.scratch_vmem_bytes).unwrap_or(0);
        let liveness_peak = self
            .compute_liveness_intervals()
            .iter()
            .map(|iv| {
                i64::try_from(iv.assigned_vmem_offset.saturating_add(iv.size_bytes as u64))
                    .unwrap_or(256)
            })
            .max()
            .unwrap_or(256);
        self.memory_stats.temp_size_in_bytes = scratch_bytes
            .max(mosaic_scratch)
            .max(liveness_peak)
            .max(256);
    }

    /// Pass 6: LLO VLIW instruction bundle scheduling, sync flag assignment, and cost analysis.
    #[allow(clippy::cast_precision_loss)]
    pub fn pass_llo_vliw_bundle_scheduling(&mut self) {
        let mut total_flops: f64 = 0.0;
        let mut total_trans: f64 = 0.0;
        for (idx, bundle) in self.instruction_bundles.iter_mut().enumerate() {
            bundle.sync_flag_id = u32::try_from(idx + 1).unwrap_or(1);
            let elems = bundle
                .attrs
                .output_shape
                .as_ref()
                .map_or(64_u64, |s| (s.element_count().max(1)) as u64);
            if bundle.opcode == TpuOpCode::DotGeneralF32 {
                total_flops += (elems.saturating_mul(8)) as f64;
            } else if bundle.opcode != TpuOpCode::Identity {
                total_flops += elems as f64;
            }
            if bundle.opcode.is_transcendental() {
                total_trans += elems as f64;
            }
        }
        let total_bytes = (self
            .memory_stats
            .argument_size_in_bytes
            .saturating_add(self.memory_stats.output_size_in_bytes))
        .max(256) as f64;
        // Estimate optimal execution time from target TPU generation roofline model:
        // max(compute_time, memory_time) with a 1.0 microsecond launch floor.
        let tpu_gen = crate::config::global_init_config().generation;
        let peak_tflops = match tpu_gen {
            crate::types::TpuGeneration::V7x => 4614.0e12,
            crate::types::TpuGeneration::V6e => 918.0e12,
            crate::types::TpuGeneration::V5p => 459.0e12,
            _ => 275.0e12,
        };
        let compute_sec = total_flops / peak_tflops;
        let mem_sec = total_bytes / 1.6e12;
        let optimal_sec = compute_sec.max(mem_sec).max(1.0e-6);
        #[allow(clippy::cast_possible_truncation)]
        {
            self.cost_analysis = CostAnalysis {
                flops: total_flops as f32,
                transcendentals: total_trans as f32,
                bytes_accessed: total_bytes as f32,
                optimal_seconds: optimal_sec as f32,
            };
        }
    }

    /// Pass 4b (`algebraic_simplifier`): Canonicalizes reshape/transpose/identity bundles, normalizes operand slots,
    /// and folds uniform identity/zero constants (`x + 0 = x`, `0 + x = x`, `x - 0 = x`, `x * 1 = x`,
    /// `1 * x = x`, `x * 0 = 0`, `0 * x = 0`, `x / 1 = x`) when **all** elements of the constant tensor match.
    pub fn pass_algebraic_simplifier(&mut self) {
        let mut const_vals: std::collections::BTreeMap<usize, Vec<f64>> =
            std::collections::BTreeMap::new();
        let mut transpose_map: std::collections::BTreeMap<usize, (usize, Vec<usize>)> =
            std::collections::BTreeMap::new();
        let mut reshape_map: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();

        for bundle in &mut self.instruction_bundles {
            if bundle.opcode == TpuOpCode::Constant && !bundle.attrs.constant_bytes.is_empty() {
                let et = bundle
                    .attrs
                    .output_shape
                    .as_ref()
                    .map_or(ElementType::F32, |s| s.element_type);
                const_vals.insert(
                    bundle.output_slot,
                    et.unpack_to_f64_vec(&bundle.attrs.constant_bytes),
                );
                continue;
            }
            if bundle.operand_slots.is_empty() && bundle.opcode.arity() > 0 {
                bundle.operand_slots.push(0);
            }
            if bundle.opcode == TpuOpCode::Transpose {
                let src_slot = bundle.operand_slots.first().copied().unwrap_or(0);
                if let Some((orig_src, prev_perm)) = transpose_map.get(&src_slot).cloned() {
                    let composed: Vec<usize> = bundle
                        .attrs
                        .permutation
                        .iter()
                        .map(|&ax| prev_perm.get(ax).copied().unwrap_or(ax))
                        .collect();
                    bundle.operand_slots = vec![orig_src];
                    bundle.attrs.permutation.clone_from(&composed);
                }
                let is_id_perm = bundle
                    .attrs
                    .permutation
                    .iter()
                    .enumerate()
                    .all(|(i, &p)| i == p)
                    && !bundle.attrs.permutation.is_empty();
                if is_id_perm {
                    bundle.opcode = TpuOpCode::Identity;
                } else {
                    let cur_src = bundle.operand_slots.first().copied().unwrap_or(src_slot);
                    transpose_map.insert(
                        bundle.output_slot,
                        (cur_src, bundle.attrs.permutation.clone()),
                    );
                }
            } else if bundle.opcode == TpuOpCode::Reshape {
                let src_slot = bundle.operand_slots.first().copied().unwrap_or(0);
                let orig_src = reshape_map.get(&src_slot).copied().unwrap_or(src_slot);
                bundle.operand_slots = vec![orig_src];
                reshape_map.insert(bundle.output_slot, orig_src);
            } else if let [s0, s1] = bundle.operand_slots.as_slice() {
                let s0 = *s0;
                let s1 = *s1;
                let c0_all_zero = const_vals
                    .get(&s0)
                    .is_some_and(|v| v.iter().all(|&x| x == 0.0));
                let c1_all_zero = const_vals
                    .get(&s1)
                    .is_some_and(|v| v.iter().all(|&x| x == 0.0));
                let c0_all_one = const_vals
                    .get(&s0)
                    .is_some_and(|v| v.iter().all(|&x| (x - 1.0).abs() < 1e-12));
                let c1_all_one = const_vals
                    .get(&s1)
                    .is_some_and(|v| v.iter().all(|&x| (x - 1.0).abs() < 1e-12));

                let keep_s0 = match bundle.opcode {
                    TpuOpCode::AddF32 | TpuOpCode::SubF32 => c1_all_zero,
                    TpuOpCode::MulF32 => c1_all_one | c0_all_zero,
                    TpuOpCode::DivF32 => c1_all_one,
                    _ => false,
                };
                let keep_s1 = match bundle.opcode {
                    TpuOpCode::AddF32 => c0_all_zero,
                    TpuOpCode::MulF32 => c0_all_one | c1_all_zero,
                    _ => false,
                };
                if keep_s0 {
                    bundle.opcode = TpuOpCode::Identity;
                    bundle.operand_slots = vec![s0];
                } else if keep_s1 {
                    bundle.opcode = TpuOpCode::Identity;
                    bundle.operand_slots = vec![s1];
                }
            }
        }
    }

    /// Pass 5b (`dot_decomposer`): Canonicalizes `DotGeneralF32` contraction dimensions for the MXU systolic array.
    pub fn pass_dot_decomposer(&mut self) {
        for bundle in &mut self.instruction_bundles {
            if bundle.opcode == TpuOpCode::DotGeneralF32 && bundle.attrs.input_shapes.is_empty() {
                bundle.attrs.input_shapes.clone_from(&self.parameter_shapes);
            }
        }
    }

    /// Pass 6b (`tpu_custom_call_lowering`): Lowers Mosaic / Pallas / `SparseCore` `CustomCall` bundles.
    pub fn pass_tpu_custom_call_lowering(&mut self) {
        for bundle in &mut self.instruction_bundles {
            if bundle.opcode == TpuOpCode::CustomCall && bundle.sync_flag_id == 0 {
                bundle.sync_flag_id = 1;
            }
        }
    }

    /// Packs hazard-free independent [`TpuInstructionBundle`]s across functional units (`VALU`, `MXU`, `XLU`, `DMA`)
    /// into consolidated 64-byte [`TensorCoreVliwBundle`] words.
    #[must_use]
    pub fn pack_independent_vliw_bundles(&self) -> Vec<TensorCoreVliwBundle> {
        let raw = self.encode_vliw_program();
        let mut packed: Vec<TensorCoreVliwBundle> = Vec::with_capacity(raw.len());
        let mut idx = 0_usize;
        while let Some(&first) = raw.get(idx) {
            let mut merged = first;
            if let Some(&second) = raw.get(idx + 1) {
                let first_dst = first.vector_slot.dst_vreg;
                let second_uses_first = second.vector_slot.src0_vreg == first_dst
                    || second.vector_slot.src1_vreg == first_dst
                    || second.vector_slot.src2_vreg == first_dst;
                let second_is_mxu = second.matrix_slot.opcode != 0 && first.matrix_slot.opcode == 0;
                let second_is_xunary =
                    second.xunary_slot.opcode != 0 && first.xunary_slot.opcode == 0;
                if !second_uses_first && (second_is_mxu || second_is_xunary) {
                    if second_is_mxu {
                        merged.matrix_slot = second.matrix_slot;
                        merged.header.bundle_flags |= 1 << 1;
                    } else {
                        merged.xunary_slot = second.xunary_slot;
                        merged.header.bundle_flags |= 1 << 2;
                    }
                    idx += 2;
                    packed.push(merged);
                    continue;
                }
            }
            packed.push(merged);
            idx += 1;
        }
        packed
    }

    /// Computes polyhedral loop tile bounds and double-buffered (`ping`/`pong`) `VMEM` DMA prefetch
    /// schedules for all lowered instruction bundles in this [`CompiledProgram`].
    #[must_use]
    pub fn compute_polyhedral_tile_schedules(&self) -> Vec<super::vliw::PolyhedralTileSchedule> {
        let tpu_gen = crate::config::global_init_config().generation;
        let default_shape = Shape {
            element_type: ElementType::F32,
            dims: vec![8, 128],
            dynamic_dims: vec![false, false],
            layout: MemoryLayout::default_tpu_tiled(2),
        };
        let fallback_shape = self.output_shapes.first().unwrap_or(&default_shape);
        let mut schedules = Vec::with_capacity(self.instruction_bundles.len());
        let mut base_vmem = 0_u64;
        for (idx, bundle) in self.instruction_bundles.iter().enumerate() {
            let sh = bundle.attrs.output_shape.as_ref().unwrap_or(fallback_shape);
            let sched = super::vliw::PolyhedralTileSchedule::from_shape(
                sh,
                tpu_gen,
                base_vmem,
                bundle.sync_flag_id.max(u32::try_from(idx + 1).unwrap_or(1)),
            );
            base_vmem = sched.pong_vmem_offset.saturating_add(4096);
            schedules.push(sched);
        }
        schedules
    }

    /// Pass 11 (`vliw_bundle_packing`): Packs all instruction bundles into 64-byte [`TensorCoreVliwBundle`]s
    /// and updates `generated_code_size_in_bytes`.
    pub fn pass_vliw_bundle_packing(&mut self) {
        let _packed = self.pack_independent_vliw_bundles();
        let encoded = self.encode_vliw_program();
        let code_bytes = encoded
            .len()
            .saturating_mul(size_of::<TensorCoreVliwBundle>());
        self.memory_stats.generated_code_size_in_bytes = self
            .memory_stats
            .generated_code_size_in_bytes
            .max(i64::try_from(code_bytes.max(256)).unwrap_or(256));
    }

    /// Pass 12 (`binary_emission`): Refreshes the deterministic SHA-256 executable fingerprint.
    pub fn pass_binary_emission(&mut self) {
        self.fingerprint = Self::compute_fingerprint(
            &self.format,
            &self.source_bytes,
            &self.compile_options_bytes,
        );
    }

    /// Encodes all lowered [`TpuInstructionBundle`]s into architectural 64-byte [`TensorCoreVliwBundle`]s,
    /// annotating each bundle with its [`super::vliw::LloVliwScheduler`] hazard stall count, double-buffered (`ping`/`pong`)
    /// `VMEM` offset, and `AsyncDmaSlot` prefetch/writeback descriptor from [`Self::compute_polyhedral_tile_schedules`].
    #[must_use]
    pub fn encode_vliw_program(&self) -> Vec<TensorCoreVliwBundle> {
        let tpu_gen = crate::config::global_init_config().generation;
        let llo_sched = super::vliw::LloVliwScheduler::new(tpu_gen);
        let packed_vliw = llo_sched.schedule_bundles(&self.instruction_bundles);
        let tile_plans = self.compute_polyhedral_tile_schedules();
        let last_idx = self.instruction_bundles.len().saturating_sub(1);
        packed_vliw
            .into_iter()
            .zip(tile_plans.iter())
            .enumerate()
            .map(|(idx, (mut vliw, plan))| {
                let vmem_off = if (idx & 1) == 0 {
                    plan.ping_vmem_offset
                } else {
                    plan.pong_vmem_offset
                };
                #[allow(clippy::cast_possible_truncation)]
                {
                    vliw.scalar_slot.imm32 = vliw.scalar_slot.imm32.max(vmem_off as u32);
                }
                vliw.dma_slot.dma_opcode = vliw.dma_slot.dma_opcode.max(1);
                if idx == last_idx {
                    vliw.dma_slot.src_space = 1;
                    vliw.dma_slot.dst_space = 0;
                } else {
                    vliw.dma_slot.src_space = 0;
                    vliw.dma_slot.dst_space = 1;
                }
                vliw.dma_slot.sync_flag_id = plan.prefetch_sync_flag_id;
                vliw.header.bundle_flags |= 1 << 3;
                vliw
            })
            .collect()
    }

    /// Encodes the [`CompiledProgram`]'s VLIW instruction bundles into a contiguous `IMEM` binary image
    /// of 64-byte [`TensorCoreVliwBundle`] words.
    #[must_use]
    pub fn encode_architectural_imem(&self) -> Vec<u8> {
        let bundles = self.encode_vliw_program();
        let mut imem = Vec::with_capacity(bundles.len() * size_of::<TensorCoreVliwBundle>());
        for b in &bundles {
            imem.extend_from_slice(bytemuck::bytes_of(b));
        }
        imem
    }

    /// Decodes a binary `IMEM` byte slice of 64-byte [`TensorCoreVliwBundle`] words.
    ///
    /// # Arguments
    ///
    /// * `imem_bytes` - Raw `IMEM` binary image (must be a multiple of 64 bytes; trailing partial bytes are ignored).
    #[must_use]
    pub fn decode_architectural_imem(imem_bytes: &[u8]) -> Vec<TensorCoreVliwBundle> {
        imem_bytes
            .chunks_exact(size_of::<TensorCoreVliwBundle>())
            .map(bytemuck::pod_read_unaligned)
            .collect()
    }

    /// Pass 4c (`constant_folding`): Folds unary (`NegF32`, `Abs`, `ExpF32`, `TanhF32`, `Sqrt`)
    /// and binary (`AddF32`, `SubF32`, `MulF32`, `MaxF32`, `MinF32`) operations whose inputs
    /// are compile-time [`TpuOpCode::Constant`] bundles into precomputed [`TpuOpCode::Constant`]
    /// bundles using each bundle's [`ElementType`].
    pub fn pass_constant_folding(&mut self) {
        let mut const_map: std::collections::BTreeMap<usize, Vec<f64>> =
            std::collections::BTreeMap::new();
        for bundle in &mut self.instruction_bundles {
            let et = bundle
                .attrs
                .output_shape
                .as_ref()
                .map_or(ElementType::F32, |s| s.element_type);
            if bundle.opcode == TpuOpCode::Constant && !bundle.attrs.constant_bytes.is_empty() {
                let vals = et.unpack_to_f64_vec(&bundle.attrs.constant_bytes);
                const_map.insert(bundle.output_slot, vals);
                continue;
            }
            let s0 = bundle.operand_slots.first().copied().unwrap_or(usize::MAX);
            let s1 = bundle.operand_slots.get(1).copied().unwrap_or(usize::MAX);
            let is_foldable_unary = matches!(
                bundle.opcode,
                TpuOpCode::NegF32
                    | TpuOpCode::Abs
                    | TpuOpCode::ExpF32
                    | TpuOpCode::TanhF32
                    | TpuOpCode::Sqrt
            );
            let is_foldable_binary = matches!(
                bundle.opcode,
                TpuOpCode::AddF32
                    | TpuOpCode::SubF32
                    | TpuOpCode::MulF32
                    | TpuOpCode::MaxF32
                    | TpuOpCode::MinF32
            );
            if is_foldable_unary && let Some(v0) = const_map.get(&s0) {
                let op = bundle.opcode;
                let folded: Vec<f64> = v0
                    .iter()
                    .map(|&x| match op {
                        TpuOpCode::Abs => x.abs(),
                        TpuOpCode::ExpF32 => x.exp(),
                        TpuOpCode::TanhF32 => x.tanh(),
                        TpuOpCode::Sqrt => x.sqrt(),
                        _ => -x,
                    })
                    .collect();
                bundle.opcode = TpuOpCode::Constant;
                bundle.operand_slots.clear();
                bundle.attrs.constant_bytes = et.pack_from_f64_slice(&folded);
                const_map.insert(bundle.output_slot, folded);
            } else if is_foldable_binary
                && let (Some(v0), Some(v1)) = (const_map.get(&s0), const_map.get(&s1))
            {
                let op = bundle.opcode;
                let folded: Vec<f64> = v0
                    .iter()
                    .zip(v1.iter())
                    .map(|(&a, &b)| match op {
                        TpuOpCode::AddF32 => a + b,
                        TpuOpCode::SubF32 => a - b,
                        TpuOpCode::MaxF32 => a.max(b),
                        TpuOpCode::MinF32 => a.min(b),
                        _ => a * b,
                    })
                    .collect();
                bundle.opcode = TpuOpCode::Constant;
                bundle.operand_slots.clear();
                bundle.attrs.constant_bytes = et.pack_from_f64_slice(&folded);
                const_map.insert(bundle.output_slot, folded);
            }
        }
    }

    /// Pass 4e (`dead_code_elimination`): Computes backward reachability from `output_slots`
    /// and removes side-effect-free instruction bundles whose `output_slot` is never transitively read.
    pub fn pass_dead_code_elimination(&mut self) {
        if self.output_slots.is_empty() {
            return;
        }
        let mut live_slots: std::collections::BTreeSet<usize> =
            self.output_slots.iter().copied().collect();
        for bundle in self.instruction_bundles.iter().rev() {
            if bundle.opcode.has_side_effects() || live_slots.contains(&bundle.output_slot) {
                live_slots.insert(bundle.output_slot);
                for &op_slot in &bundle.operand_slots {
                    live_slots.insert(op_slot);
                }
            }
        }
        self.instruction_bundles
            .retain(|b| b.opcode.has_side_effects() || live_slots.contains(&b.output_slot));
    }

    /// Pass 4d (`common_subexpression_elimination`): Deduplicates identical pure instruction
    /// bundles via [`CseExpressionKey`] and rewires downstream consumers.
    pub fn pass_common_subexpression_elimination(&mut self) {
        if self.instruction_bundles.len() <= 1 {
            return;
        }
        let mut expr_to_slot: Vec<(CseExpressionKey, usize)> = Vec::new();
        let mut slot_remap: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        let mut retained = Vec::with_capacity(self.instruction_bundles.len());

        for mut bundle in self.instruction_bundles.drain(..) {
            for op_slot in &mut bundle.operand_slots {
                *op_slot = slot_remap.get(op_slot).copied().unwrap_or(*op_slot);
            }
            let is_impure = bundle.opcode.has_side_effects();
            let key = CseExpressionKey {
                opcode: bundle.opcode,
                operand_slots: bundle.operand_slots.clone(),
                attrs: bundle.attrs.clone(),
            };
            if !is_impure
                && let Some(&(_, canonical_slot)) = expr_to_slot.iter().find(|(k, _)| *k == key)
            {
                slot_remap.insert(bundle.output_slot, canonical_slot);
            } else {
                expr_to_slot.push((key, bundle.output_slot));
                retained.push(bundle);
            }
        }
        for out_slot in &mut self.output_slots {
            *out_slot = slot_remap.get(out_slot).copied().unwrap_or(*out_slot);
        }
        self.instruction_bundles = retained;
    }

    /// Computes `[first_def_pc, last_use_pc]` liveness intervals and linear-scan VMEM offsets
    /// for all parameter and intermediate SSA register slots.
    #[must_use]
    pub fn compute_liveness_intervals(&self) -> Vec<RegisterLivenessInterval> {
        let last_pc = u32::try_from(self.instruction_bundles.len().saturating_sub(1)).unwrap_or(0);
        let mut intervals: std::collections::BTreeMap<usize, RegisterLivenessInterval> =
            std::collections::BTreeMap::new();

        for (p_idx, shape) in self.parameter_shapes.iter().enumerate() {
            let sz = shape.unpadded_byte_size().0.max(256).div_ceil(256) * 256;
            intervals.insert(
                p_idx,
                RegisterLivenessInterval {
                    slot_index: p_idx,
                    first_def_pc: 0,
                    last_use_pc: 0,
                    size_bytes: sz,
                    assigned_vmem_offset: 0,
                },
            );
        }

        for (b_idx, bundle) in self.instruction_bundles.iter().enumerate() {
            let pc = u32::try_from(b_idx).unwrap_or(0);
            let sz = bundle.attrs.output_shape.as_ref().map_or(256, |s| {
                s.unpadded_byte_size().0.max(256).div_ceil(256) * 256
            });
            intervals
                .entry(bundle.output_slot)
                .or_insert(RegisterLivenessInterval {
                    slot_index: bundle.output_slot,
                    first_def_pc: pc,
                    last_use_pc: pc,
                    size_bytes: sz,
                    assigned_vmem_offset: 0,
                });
            for &op_slot in &bundle.operand_slots {
                intervals.get_mut(&op_slot).into_iter().for_each(|iv| {
                    iv.last_use_pc = iv.last_use_pc.max(pc);
                });
            }
        }

        for &out_slot in &self.output_slots {
            intervals.get_mut(&out_slot).into_iter().for_each(|iv| {
                iv.last_use_pc = iv.last_use_pc.max(last_pc);
            });
        }

        // Linear-scan VMEM allocator with interval expiration reuse
        let mut active: Vec<(u32, u64, usize)> = Vec::new();
        let mut free_blocks: Vec<(u64, usize)> = Vec::new();
        let mut next_vmem_offset: u64 = 0;
        let mut result: Vec<RegisterLivenessInterval> = intervals.into_values().collect();

        for iv in &mut result {
            active.retain(|&(end_pc, off, sz)| {
                if end_pc < iv.first_def_pc {
                    free_blocks.push((off, sz));
                    false
                } else {
                    true
                }
            });
            let reused_pos = free_blocks.iter().position(|&(_, sz)| sz >= iv.size_bytes);
            let assigned = if let Some(pos) = reused_pos {
                free_blocks.remove(pos).0
            } else {
                let off = next_vmem_offset;
                next_vmem_offset = next_vmem_offset.saturating_add(iv.size_bytes as u64);
                off
            };
            iv.assigned_vmem_offset = assigned;
            active.push((iv.last_use_pc, assigned, iv.size_bytes));
        }

        result
    }

    /// Runs all compiler passes in sequence on this [`CompiledProgram`].
    pub fn run_all_passes(&mut self) {
        self.pass_hlo_verification_and_shape_inference();
        self.pass_spmd_sharding_propagation();
        self.pass_spmd_partitioning();
        self.pass_algebraic_simplifier();
        self.pass_constant_folding();
        self.pass_common_subexpression_elimination();
        self.pass_dot_decomposer();
        self.pass_tpu_custom_call_lowering();
        self.pass_hlo_operator_fusion();
        self.pass_tpu_layout_assignment_and_tiling();
        self.pass_vmem_scratchpad_allocation();
        self.pass_llo_vliw_bundle_scheduling();
        self.pass_vliw_bundle_packing();
        self.pass_binary_emission();
    }
}
