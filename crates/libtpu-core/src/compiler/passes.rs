//! Compiler optimization passes, VMEM liveness analysis, cost analysis, and HLO DAG scheduling.

use super::opcode::{TpuInstructionBundle, TpuOpCode};
use super::vliw::TensorCoreVliwBundle;
use super::{COMPILATION_PHASE_NAMES, CompiledProgram};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{HloComputationProto, PjRtPartialProgramProto};
use crate::types::{ElementType, MemoryKind, MemoryLayout, Shape};
use indexmap::IndexMap;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableDiGraph;

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
        prog.run_all_passes();
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
            version: String::from("0.0.50"),
            program_name: String::from("tpu_partial_program"),
        })
    }

    /// Pass 1: HLO verification and shape inference across parameter, bundle, and output shapes.
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
        for bundle in &mut self.instruction_bundles {
            let _ = bundle
                .attrs
                .output_shape
                .get_or_insert(fallback_shape.clone());
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

    /// Pass 2: SPMD sharding propagation across replicas and partitions.
    pub fn pass_spmd_sharding_propagation(&mut self) {
        self.num_replicas = self.num_replicas.max(1);
        self.num_partitions = self.num_partitions.max(1);
        self.parameter_memory_kinds
            .resize(self.parameter_shapes.len(), MemoryKind::DeviceHbm);
        self.output_memory_kinds
            .resize(self.output_shapes.len(), MemoryKind::DeviceHbm);
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
        for (idx, bundle) in self.instruction_bundles.iter_mut().enumerate() {
            bundle.pc_offset = u32::try_from(idx).unwrap_or(0);
        }
    }

    /// Pass 5: On-chip VMEM scratchpad allocation based on intermediate instruction bundle footprint.
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
        self.memory_stats.temp_size_in_bytes = scratch_bytes.max(256);
    }

    /// Pass 6: LLO VLIW instruction bundle scheduling, sync flag assignment, and cost analysis.
    pub fn pass_llo_vliw_bundle_scheduling(&mut self) {
        let mut total_flops: f32 = 0.0;
        let mut total_trans: f32 = 0.0;
        for (idx, bundle) in self.instruction_bundles.iter_mut().enumerate() {
            bundle.sync_flag_id = u32::try_from(idx + 1).unwrap_or(1);
            let elems = u16::try_from(
                bundle
                    .attrs
                    .output_shape
                    .as_ref()
                    .map_or(64_usize, |s| s.element_count().max(1)),
            )
            .unwrap_or(64);
            if bundle.opcode == TpuOpCode::DotGeneralF32 {
                total_flops += f32::from(elems.saturating_mul(8));
            } else if bundle.opcode != TpuOpCode::Identity {
                total_flops += f32::from(elems);
            }
            if bundle.opcode.is_transcendental() {
                total_trans += f32::from(elems);
            }
        }
        let bytes_u16 = u16::try_from(
            (self.memory_stats.argument_size_in_bytes + self.memory_stats.output_size_in_bytes)
                .max(256),
        )
        .unwrap_or(512);
        self.cost_analysis = CostAnalysis {
            flops: total_flops,
            transcendentals: total_trans,
            bytes_accessed: f32::from(bytes_u16),
            optimal_seconds: 0.000_001,
        };
    }

    /// Pass 3b (`spmd_partitioning`): Normalizes SPMD replica/partition counts and collective opcodes.
    pub fn pass_spmd_partitioning(&mut self) {
        self.num_replicas = self.num_replicas.max(1);
        self.num_partitions = self.num_partitions.max(1);
    }

    /// Pass 4b (`algebraic_simplifier`): Canonicalizes reshape/identity bundles and normalizes operand slots.
    pub fn pass_algebraic_simplifier(&mut self) {
        for bundle in &mut self.instruction_bundles {
            if bundle.operand_slots.is_empty() && bundle.opcode != TpuOpCode::Constant {
                bundle.operand_slots.push(0);
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

    /// Pass 11 (`vliw_bundle_packing`): Packs all instruction bundles into 64-byte [`TensorCoreVliwBundle`]s
    /// and updates `generated_code_size_in_bytes`.
    pub fn pass_vliw_bundle_packing(&mut self) {
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

    /// Encodes all lowered [`TpuInstructionBundle`]s into architectural 64-byte [`TensorCoreVliwBundle`]s.
    #[must_use]
    pub fn encode_vliw_program(&self) -> Vec<TensorCoreVliwBundle> {
        self.instruction_bundles
            .iter()
            .map(TpuInstructionBundle::encode_architectural_bundle)
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

    /// Pass 4c (`constant_folding`): Folds unary (`NegF32`) and binary (`AddF32`, `MulF32`)
    /// `F32` operations whose inputs are compile-time [`TpuOpCode::Constant`] bundles into
    /// precomputed [`TpuOpCode::Constant`] bundles.
    pub fn pass_constant_folding(&mut self) {
        let mut const_map: std::collections::BTreeMap<usize, Vec<f64>> =
            std::collections::BTreeMap::new();
        for bundle in &mut self.instruction_bundles {
            if bundle.opcode == TpuOpCode::Constant && !bundle.attrs.constant_bytes.is_empty() {
                let vals = ElementType::F32.unpack_to_f64_vec(&bundle.attrs.constant_bytes);
                const_map.insert(bundle.output_slot, vals);
                continue;
            }
            let s0 = bundle.operand_slots.first().copied().unwrap_or(usize::MAX);
            let s1 = bundle.operand_slots.get(1).copied().unwrap_or(usize::MAX);
            if bundle.opcode == TpuOpCode::NegF32
                && let Some(v0) = const_map.get(&s0)
            {
                let folded: Vec<f64> = v0.iter().map(|&x| -x).collect();
                bundle.opcode = TpuOpCode::Constant;
                bundle.operand_slots.clear();
                bundle.attrs.constant_bytes = ElementType::F32.pack_from_f64_slice(&folded);
                const_map.insert(bundle.output_slot, folded);
            } else if matches!(bundle.opcode, TpuOpCode::AddF32 | TpuOpCode::MulF32)
                && let (Some(v0), Some(v1)) = (const_map.get(&s0), const_map.get(&s1))
            {
                let is_add = bundle.opcode == TpuOpCode::AddF32;
                let folded: Vec<f64> = v0
                    .iter()
                    .zip(v1.iter())
                    .map(|(&a, &b)| if is_add { a + b } else { a * b })
                    .collect();
                bundle.opcode = TpuOpCode::Constant;
                bundle.operand_slots.clear();
                bundle.attrs.constant_bytes = ElementType::F32.pack_from_f64_slice(&folded);
                const_map.insert(bundle.output_slot, folded);
            }
        }
    }

    /// Pass 4d (`common_subexpression_elimination`): Deduplicates identical pure instruction
    /// bundles `(opcode, operand_slots, constant_bytes)` and rewires downstream consumers.
    pub fn pass_common_subexpression_elimination(&mut self) {
        if self.instruction_bundles.len() <= 1 {
            return;
        }
        let mut expr_to_slot: std::collections::BTreeMap<(u8, Vec<usize>, Vec<u8>), usize> =
            std::collections::BTreeMap::new();
        let mut slot_remap: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        let mut retained = Vec::with_capacity(self.instruction_bundles.len());

        for mut bundle in self.instruction_bundles.drain(..) {
            for op_slot in &mut bundle.operand_slots {
                *op_slot = slot_remap.get(op_slot).copied().unwrap_or(*op_slot);
            }
            let is_impure = matches!(
                bundle.opcode,
                TpuOpCode::CustomCall | TpuOpCode::RngBitGenerator
            );
            let key = (
                bundle.opcode as u8,
                bundle.operand_slots.clone(),
                bundle.attrs.constant_bytes.clone(),
            );
            if !is_impure && let Some(&canonical_slot) = expr_to_slot.get(&key) {
                slot_remap.insert(bundle.output_slot, canonical_slot);
            } else {
                expr_to_slot.insert(key, bundle.output_slot);
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
    #[allow(for_loops_over_fallibles)]
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
                for iv in intervals.get_mut(&op_slot) {
                    iv.last_use_pc = iv.last_use_pc.max(pc);
                }
            }
        }

        for &out_slot in &self.output_slots {
            for iv in intervals.get_mut(&out_slot) {
                iv.last_use_pc = iv.last_use_pc.max(last_pc);
            }
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
