//! XLA-TPU `MemorySpaceAssignment`, spacetime heap simulation, and learned cost analysis.
//!
//! Colors tensors across `HBM=0`, `VMEM=1`, `SMEM=2`, and `CMEM=3`, schedules double-buffered
//! `ping`/`pong` prefetches, runs `GlobalDecreasingSizeBestFitHeapSimulator`, and evaluates costs.

use crate::compiler::middle_end::layout_and_fusion::{
    assign_shape_tpu_layout, primitive_type_bit_width, shape_bytes_256_aligned,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    BufferAllocationAssignedProto, BufferAllocationProto, BufferAssignmentAliasProto,
    BufferAssignmentPeakBuffersProto, BufferAssignmentProto, BufferDonationAliasEntryProto,
    CompiledMemoryStatsProto, ComputationStats, EmitterLearnedCostModelOptionsProto,
    ExpandedScopedAlternateMemoryMode, FrontendAttributes, HeapSimulatorEventKindProto,
    HeapSimulatorTraceEventProto, HeapSimulatorTraceProto, HloAliasEntryProto, HloComputationProto,
    HloInstructionProto, HloModuleProto, InputOutputBufferAliasConfigProto,
    JellyfishLearnedCostModelModeProto, JellyfishMemorySpaceProto,
    JellyfishMlOutputValidationStrategyProto, LogicalBufferLocationProto, LogicalBufferProto,
    MemoryBoundLoopOptimizerOptionsProto, MemoryUsageAllocationEntryProto, MemoryUsageReportProto,
    MemoryUsageSpaceEntryProto, MsaHloOperandFilterProto, MsaHloPositionMatcherProto,
    MsaSortOrderOverridesProto, MsaTensorOverridesProto, PreferredPrefetchOverridesProto,
    PrimitiveType, ProtoSchemaMessage, ShapeProto, SlicedPrefetchOptionsProto,
    msa_sort_order_override_options, msa_tensor_override, preferred_prefetch_override_options,
};
use crate::types::TpuGeneration;
use std::collections::BTreeMap;

/// Frontend attribute key recording `"ping"` or `"pong"` double-buffer slot on prefetched `copy-start`/`copy-done` ops.
pub const MSA_DOUBLE_BUFFER_SLOT_ATTR_KEY: &str = "xla.msa.double_buffer_slot";

/// Canonical XLA-TPU layout `memory_space` / `LogicalBufferProto.color` codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i64)]
pub enum TpuMsaMemorySpace {
    /// Default High-Bandwidth Memory (`0`).
    Hbm = 0,
    /// Alternate `TensorCore` Vector Scratchpad Memory (`1`).
    Vmem = 1,
    /// Scalar Control / Constant Scratchpad Memory (`2`).
    Smem = 2,
    /// TPU v4 On-Chip Common Memory (`3`).
    Cmem = 3,
}

impl TpuMsaMemorySpace {
    /// Returns the numeric XLA `LayoutProto.memory_space` color (`0..=3`).
    #[must_use]
    pub const fn color(self) -> i64 {
        self as i64
    }

    /// Maps this MSA color to [`JellyfishMemorySpaceProto`].
    #[must_use]
    pub const fn to_jellyfish_proto(self) -> JellyfishMemorySpaceProto {
        match self {
            Self::Hbm => JellyfishMemorySpaceProto::MemorySpaceHbm,
            Self::Vmem => JellyfishMemorySpaceProto::MemorySpaceVmem,
            Self::Smem => JellyfishMemorySpaceProto::MemorySpaceSmem,
            Self::Cmem => JellyfishMemorySpaceProto::MemorySpaceCmem,
        }
    }

    /// Byte alignment required for allocations in this memory space.
    #[must_use]
    pub const fn alignment_bytes(self) -> u64 {
        match self {
            Self::Hbm => 4096,
            Self::Vmem | Self::Cmem => 256,
            Self::Smem => 64,
        }
    }

    /// Converts a raw numeric color (`0..=3`) into a [`TpuMsaMemorySpace`].
    ///
    /// # Arguments
    ///
    /// * `color` - Raw layout memory space integer.
    #[must_use]
    pub const fn from_color(color: i64) -> Self {
        match color {
            1 => Self::Vmem,
            2 => Self::Smem,
            3 => Self::Cmem,
            _ => Self::Hbm,
        }
    }
}

/// Summary report emitted by [`MemorySpaceAssignmentPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemorySpaceAssignmentReport {
    /// Number of tensors placed in HBM (`color = 0`).
    pub hbm_count: usize,
    /// Number of tensors placed in VMEM (`color = 1`).
    pub vmem_count: usize,
    /// Number of tensors placed in SMEM (`color = 2`).
    pub smem_count: usize,
    /// Number of tensors placed in CMEM (`color = 3`).
    pub cmem_count: usize,
    /// Number of async `copy-start` / `copy-done` prefetch windows scheduled.
    pub scheduled_prefetches: usize,
    /// Number of `ping` (`0`) prefetch slots assigned.
    pub ping_buffers: usize,
    /// Number of `pong` (`1`) prefetch slots assigned.
    pub pong_buffers: usize,
    /// Effective usable VMEM scratchpad budget after reserving `scoped_vmem_limit_kib`.
    pub usable_vmem_budget_bytes: u64,
}

/// Configuration and proto-driven policy for [`MemorySpaceAssignmentPass`].
#[derive(Debug, Clone)]
pub struct MemorySpaceAssignmentPass {
    /// Target Cloud TPU generation determining VMEM, SMEM, and CMEM capacities.
    pub generation: TpuGeneration,
    /// Reserved scoped VMEM scratchpad in KiB (`xla_tpu_scoped_vmem_limit_kib`).
    pub scoped_vmem_limit_kib: u64,
    /// Optional explicit SMEM byte cap override (`None` uses hardware default).
    pub smem_limit_bytes: Option<u64>,
    /// Expanded scoped alternate memory mode (`ExpandedScopedAlternateMemoryMode::Value`).
    pub expanded_scoped_vmem_mode: i32,
    /// Explicit tensor pinning/prefetch overrides (`memory_space_assignment.proto`).
    pub tensor_overrides: MsaTensorOverridesProto,
    /// Sort-order priority overrides (`memory_space_assignment.proto`).
    pub sort_order_overrides: MsaSortOrderOverridesProto,
    /// Preferred prefetch timing overrides (`memory_space_assignment.proto`).
    pub preferred_prefetch_overrides: PreferredPrefetchOverridesProto,
    /// Sliced prefetch configuration (`memory_space_assignment.proto`).
    pub sliced_prefetch_options: SlicedPrefetchOptionsProto,
    /// Memory-bound loop optimizer configuration (`memory_space_assignment.proto`).
    pub loop_optimizer_options: MemoryBoundLoopOptimizerOptionsProto,
}

impl MemorySpaceAssignmentPass {
    /// Constructs a [`MemorySpaceAssignmentPass`] with hardware defaults for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation.
    /// * `scoped_vmem_limit_kib` - Reserved scoped VMEM in KiB for custom/Mosaic kernels.
    #[must_use]
    pub fn for_generation(generation: TpuGeneration, scoped_vmem_limit_kib: u64) -> Self {
        Self {
            generation,
            scoped_vmem_limit_kib,
            smem_limit_bytes: None,
            expanded_scoped_vmem_mode:
                crate::proto::descriptor::generated_prost_schemas::xla::memory_space_assignment::expanded_scoped_alternate_memory_mode::Value::Undefined
                    as i32,
            tensor_overrides: MsaTensorOverridesProto::default(),
            sort_order_overrides: MsaSortOrderOverridesProto::default(),
            preferred_prefetch_overrides: PreferredPrefetchOverridesProto::default(),
            sliced_prefetch_options: SlicedPrefetchOptionsProto::default(),
            loop_optimizer_options: MemoryBoundLoopOptimizerOptionsProto::default(),
        }
    }

    /// Returns `(total_vmem_bytes, total_smem_bytes, total_cmem_bytes)` for `self.generation`.
    #[must_use]
    pub const fn hardware_scratchpad_capacities(&self) -> (u64, u64, u64) {
        let _ = ExpandedScopedAlternateMemoryMode {};
        match self.generation {
            TpuGeneration::V2 | TpuGeneration::V3 | TpuGeneration::V4Lite => {
                (16 * 1024 * 1024, 1024 * 1024, 0)
            }
            TpuGeneration::V4 => (32 * 1024 * 1024, 1024 * 1024, 128 * 1024 * 1024),
            TpuGeneration::V5e => (32 * 1024 * 1024, 1024 * 1024, 0),
            TpuGeneration::V5p | TpuGeneration::V6e => (64 * 1024 * 1024, 2 * 1024 * 1024, 0),
            TpuGeneration::V7x => (128 * 1024 * 1024, 4 * 1024 * 1024, 0),
        }
    }

    /// Assigns `HBM` (`0`), `VMEM` (`1`), `SMEM` (`2`), and `CMEM` (`3`) memory spaces and
    /// schedules double-buffered (`ping`/`pong`) `copy-start`/`copy-done` prefetches on `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] whose instruction layouts and async copies are updated.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MemorySpaceAssignmentOom`] if `scoped_vmem_limit_kib` or a pinned
    /// `PinInAlternateMemoryAction` exceeds VMEM capacity, or if a strict `PreferredPrefetchOverride`
    /// cannot be satisfied.
    #[allow(clippy::too_many_lines)]
    pub fn run(&self, module: &mut HloModuleProto) -> LibTpuResult<MemorySpaceAssignmentReport> {
        let (vmem_cap, scalar_cap, common_cap) = self.hardware_scratchpad_capacities();
        let reserved_scoped = self.scoped_vmem_limit_kib.saturating_mul(1024);
        let effective_total_vmem = if self.expanded_scoped_vmem_mode == 2 {
            vmem_cap.saturating_add(reserved_scoped / 2)
        } else {
            vmem_cap
        };
        if reserved_scoped >= effective_total_vmem {
            return Err(LibTpuError::MemorySpaceAssignmentOom {
                memory_space: "vmem",
                requested_bytes: reserved_scoped,
                limit_bytes: effective_total_vmem,
            });
        }
        let usable_vmem = effective_total_vmem - reserved_scoped;

        self.validate_strict_prefetch_overrides(module)?;

        let mut report = MemorySpaceAssignmentReport {
            usable_vmem_budget_bytes: usable_vmem,
            ..MemorySpaceAssignmentReport::default()
        };

        for comp in &mut module.computations {
            self.assign_computation_memory_spaces(
                comp,
                usable_vmem,
                self.smem_limit_bytes.unwrap_or(scalar_cap),
                common_cap,
                &mut report,
            )?;
        }

        Ok(report)
    }

    /// Checks whether any `PreferredPrefetchOverride` with `fail_on_unsatisfied_override` fails to match.
    ///
    /// # Arguments
    ///
    /// * `module` - Module whose instructions are matched against prefetch filters.
    fn validate_strict_prefetch_overrides(&self, module: &HloModuleProto) -> LibTpuResult<()> {
        let global_strict = self
            .preferred_prefetch_overrides
            .fail_on_unsatisfied_override
            .unwrap_or(false);
        for ov in &self.preferred_prefetch_overrides.overrides {
            if !ov.fail_on_unsatisfied_override.unwrap_or(global_strict) {
                continue;
            }
            if let Some(filter) = &ov.hlo_operand_filter {
                let matched = module.computations.iter().any(|c| {
                    c.instructions
                        .iter()
                        .any(|inst| matches_operand_filter(inst, filter))
                });
                if !matched {
                    return Err(LibTpuError::CompilationFailed {
                        reason: String::from(
                            "Unsatisfied strict PreferredPrefetchOverride in MemorySpaceAssignmentPass",
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Colors tensors and inserts double-buffered `ping`/`pong` prefetch windows in `comp`.
    ///
    /// # Arguments
    ///
    /// * `comp` - Mutable computation to assign memory spaces on.
    /// * `usable_vmem` - Usable VMEM byte budget.
    /// * `scalar_budget` - Usable SMEM byte budget.
    /// * `common_budget` - Usable CMEM byte budget (`> 0` on TPU v4).
    /// * `report` - Summary counter accumulator.
    #[allow(clippy::too_many_lines)]
    fn assign_computation_memory_spaces(
        &self,
        comp: &mut HloComputationProto,
        usable_vmem: u64,
        scalar_budget: u64,
        common_budget: u64,
        report: &mut MemorySpaceAssignmentReport,
    ) -> LibTpuResult<()> {
        let mut use_counts: BTreeMap<i64, usize> = BTreeMap::new();
        for inst in &comp.instructions {
            for &op in &inst.operand_ids {
                let c = use_counts.entry(op).or_insert(0);
                *c = c.saturating_add(1);
            }
        }

        let mut order_indices: Vec<(i64, u64, usize)> = comp
            .instructions
            .iter()
            .enumerate()
            .map(|(idx, inst)| {
                let sz = inst.shape.as_ref().map_or(256, shape_bytes_256_aligned);
                let sort_prio = self.compute_sort_priority(inst, sz);
                (sort_prio, u64::MAX - sz, idx)
            })
            .collect();
        order_indices.sort_unstable();

        let mut vmem_free = usable_vmem;
        let mut scalar_free = scalar_budget;
        let mut common_free = common_budget;
        let mut assigned_space: BTreeMap<i64, TpuMsaMemorySpace> = BTreeMap::new();

        for &(_, _, idx) in &order_indices {
            comp.instructions.get_mut(idx).into_iter().try_for_each(
                |inst| -> LibTpuResult<()> {
                    let is_root = inst.id == comp.root_id;
                    let sz = inst.shape.as_ref().map_or(256, shape_bytes_256_aligned);
                    let rank = inst.shape.as_ref().map_or(0, |s| s.dimensions.len());

                    let explicit_override = self.evaluate_tensor_override(inst, sz);
                    let is_scalar_const =
                        rank == 0 && inst.opcode == "constant" && sz <= scalar_free;
                    let is_shared_cmem =
                        common_free >= sz && use_counts.get(&inst.id).copied().unwrap_or(0) >= 2;
                    let chosen = match explicit_override {
                        Some(TpuMsaMemorySpace::Vmem) => {
                            if sz > vmem_free {
                                return Err(LibTpuError::MemorySpaceAssignmentOom {
                                    memory_space: "vmem",
                                    requested_bytes: sz,
                                    limit_bytes: vmem_free,
                                });
                            }
                            vmem_free = vmem_free.saturating_sub(sz);
                            TpuMsaMemorySpace::Vmem
                        }
                        Some(space) => space,
                        None if inst.opcode == "parameter" || is_root => TpuMsaMemorySpace::Hbm,
                        None if is_scalar_const => {
                            scalar_free = scalar_free.saturating_sub(sz);
                            TpuMsaMemorySpace::Smem
                        }
                        None if is_shared_cmem => {
                            common_free = common_free.saturating_sub(sz);
                            TpuMsaMemorySpace::Cmem
                        }
                        None if sz <= vmem_free / 2 => {
                            vmem_free = vmem_free.saturating_sub(sz);
                            TpuMsaMemorySpace::Vmem
                        }
                        None => TpuMsaMemorySpace::Hbm,
                    };

                    inst.shape.as_mut().into_iter().for_each(|shape| {
                        if shape.layout.is_none() {
                            assign_shape_tpu_layout(shape, None);
                        }
                        set_shape_memory_space_recursive(shape, chosen.color());
                    });
                    assigned_space.insert(inst.id, chosen);
                    match chosen {
                        TpuMsaMemorySpace::Hbm => {
                            report.hbm_count = report.hbm_count.saturating_add(1);
                        }
                        TpuMsaMemorySpace::Vmem => {
                            report.vmem_count = report.vmem_count.saturating_add(1);
                        }
                        TpuMsaMemorySpace::Smem => {
                            report.smem_count = report.smem_count.saturating_add(1);
                        }
                        TpuMsaMemorySpace::Cmem => {
                            report.cmem_count = report.cmem_count.saturating_add(1);
                        }
                    }
                    Ok(())
                },
            )?;
        }

        let mut next_id = comp
            .instructions
            .iter()
            .map(|i| i.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let shape_by_id: BTreeMap<i64, ShapeProto> = comp
            .instructions
            .iter()
            .filter_map(|i| i.shape.as_ref().map(|s| (i.id, s.clone())))
            .collect();

        let original = std::mem::take(&mut comp.instructions);
        let mut prefetched_insts: Vec<HloInstructionProto> = Vec::with_capacity(original.len() * 2);
        let mut ping_pong_toggle = false;

        let eagerness = self.resolve_prefetch_eagerness();
        let max_prefetch_bytes = if self.sliced_prefetch_options.preferred_slice_size > 0 {
            self.sliced_prefetch_options
                .preferred_slice_size
                .max(usable_vmem / 2)
        } else {
            usable_vmem / 2
        };

        for mut inst in original {
            let wants_prefetch =
                matches!(inst.opcode.as_str(), "dot" | "fusion") && eagerness > 0.0;
            if wants_prefetch {
                let mut new_ops = Vec::with_capacity(inst.operand_ids.len());
                for &op_id in &inst.operand_ids {
                    let op_space = assigned_space
                        .get(&op_id)
                        .copied()
                        .unwrap_or(TpuMsaMemorySpace::Hbm);
                    let eligible_shape = shape_by_id.get(&op_id).filter(|s| {
                        let op_sz = shape_bytes_256_aligned(s);
                        op_space == TpuMsaMemorySpace::Hbm && op_sz <= max_prefetch_bytes
                    });

                    if let Some(src_shape) = eligible_shape {
                        let slot_label = if ping_pong_toggle { "pong" } else { "ping" };
                        if ping_pong_toggle {
                            report.pong_buffers = report.pong_buffers.saturating_add(1);
                        } else {
                            report.ping_buffers = report.ping_buffers.saturating_add(1);
                        }
                        ping_pong_toggle = !ping_pong_toggle;

                        let mut vmem_shape = src_shape.clone();
                        assign_shape_tpu_layout(&mut vmem_shape, None);
                        set_shape_memory_space_recursive(
                            &mut vmem_shape,
                            TpuMsaMemorySpace::Vmem.color(),
                        );

                        let copy_start_id = next_id;
                        next_id = next_id.saturating_add(1);
                        let copy_done_id = next_id;
                        next_id = next_id.saturating_add(1);

                        let mut fa = FrontendAttributes::default();
                        fa.map.insert(
                            String::from(MSA_DOUBLE_BUFFER_SLOT_ATTR_KEY),
                            String::from(slot_label),
                        );

                        let tuple_ctx_shape = ShapeProto {
                            element_type: PrimitiveType::Tuple as i32,
                            tuple_shapes: vec![
                                vmem_shape.clone(),
                                src_shape.clone(),
                                ShapeProto {
                                    element_type: PrimitiveType::U32 as i32,
                                    ..ShapeProto::default()
                                },
                            ],
                            ..ShapeProto::default()
                        };

                        prefetched_insts.push(HloInstructionProto {
                            id: copy_start_id,
                            name: format!("msa_prefetch_start_{slot_label}_{copy_start_id}"),
                            opcode: String::from("copy-start"),
                            shape: Some(tuple_ctx_shape),
                            operand_ids: vec![op_id],
                            frontend_attributes: Some(fa.clone()),
                            ..HloInstructionProto::default()
                        });
                        prefetched_insts.push(HloInstructionProto {
                            id: copy_done_id,
                            name: format!("msa_prefetch_done_{slot_label}_{copy_done_id}"),
                            opcode: String::from("copy-done"),
                            shape: Some(vmem_shape),
                            operand_ids: vec![copy_start_id],
                            frontend_attributes: Some(fa),
                            ..HloInstructionProto::default()
                        });
                        report.scheduled_prefetches = report.scheduled_prefetches.saturating_add(1);
                        report.vmem_count = report.vmem_count.saturating_add(1);
                        new_ops.push(copy_done_id);
                    } else {
                        new_ops.push(op_id);
                    }
                }
                inst.operand_ids = new_ops;
            }
            prefetched_insts.push(inst);
        }

        hoist_copy_starts_for_double_buffering(&mut prefetched_insts, eagerness);
        comp.instructions = prefetched_insts;
        Ok(())
    }

    /// Computes sort priority for `inst` honoring `self.sort_order_overrides`.
    ///
    /// # Arguments
    ///
    /// * `inst` - Instruction to score.
    /// * `byte_size` - Physical byte size of `inst`.
    fn compute_sort_priority(&self, inst: &HloInstructionProto, byte_size: u64) -> i64 {
        for ov in &self.sort_order_overrides.overrides {
            let matched = ov
                .hlo_position_matcher
                .as_ref()
                .is_none_or(|m| matches_position_matcher(inst, byte_size, m));
            let choice_opt = ov
                .override_options
                .as_ref()
                .and_then(|o| o.options.as_ref());
            if matched && let Some(choice) = choice_opt {
                return match choice {
                    msa_sort_order_override_options::Options::AssignFirst(true) => i64::MIN,
                    msa_sort_order_override_options::Options::AssignLast(true) => i64::MAX,
                    msa_sort_order_override_options::Options::AssignValue(v) => *v,
                    _ => 0,
                };
            }
        }
        0
    }

    /// Evaluates `self.tensor_overrides` on `inst` to determine any explicit pin/keep directive.
    ///
    /// # Arguments
    ///
    /// * `inst` - Instruction to evaluate.
    /// * `byte_size` - Physical byte size of `inst`.
    fn evaluate_tensor_override(
        &self,
        inst: &HloInstructionProto,
        byte_size: u64,
    ) -> Option<TpuMsaMemorySpace> {
        for ov in &self.tensor_overrides.overrides {
            let pos_ok = ov
                .hlo_position_matcher
                .as_ref()
                .is_none_or(|m| matches_position_matcher(inst, byte_size, m));
            let op_ok = ov
                .hlo_operand_filter
                .as_ref()
                .is_none_or(|f| matches_operand_filter(inst, f));
            if pos_ok && op_ok {
                return ov.action.as_ref().map(|act| match act {
                    msa_tensor_override::Action::PinInAlternateMemory(_)
                    | msa_tensor_override::Action::Prefetch(_) => TpuMsaMemorySpace::Vmem,
                    msa_tensor_override::Action::KeepInDefaultMemory(_) => TpuMsaMemorySpace::Hbm,
                });
            }
        }
        None
    }

    /// Resolves effective prefetch eagerness in `[0.0, 1.0]` from overrides and loop optimizer options.
    fn resolve_prefetch_eagerness(&self) -> f32 {
        if let Some(choice) = self
            .preferred_prefetch_overrides
            .overrides
            .iter()
            .find_map(|ov| {
                ov.override_options
                    .as_ref()
                    .and_then(|o| o.options.as_ref())
            })
        {
            return match choice {
                preferred_prefetch_override_options::Options::PrefetchEagerness(e) => {
                    e.clamp(0.0, 1.0)
                }
                preferred_prefetch_override_options::Options::LogicalTime(_)
                | preferred_prefetch_override_options::Options::AfterInstruction(_)
                | preferred_prefetch_override_options::Options::BeforeInstruction(_) => 0.75,
            };
        }
        if self.loop_optimizer_options.enabled == Some(false) {
            0.0
        } else {
            self.loop_optimizer_options
                .desired_copy_ratio
                .unwrap_or(0.5)
                .clamp(0.0, 1.0)
        }
    }
}

/// Moves `copy-start` instructions earlier (up to operand definition point) so `ping` and `pong`
/// DMA prefetches overlap prior compute instructions.
///
/// # Arguments
///
/// * `insts` - Mutable instruction sequence.
/// * `eagerness` - Prefetch eagerness in `[0.0, 1.0]`.
fn hoist_copy_starts_for_double_buffering(insts: &mut Vec<HloInstructionProto>, eagerness: f32) {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let max_back_steps = ((eagerness * 8.0).ceil() as usize).max(1);
    for idx in 0..insts.len() {
        let is_cs = insts.get(idx).is_some_and(|i| i.opcode == "copy-start");
        if !is_cs {
            continue;
        }
        let op_deps: Vec<i64> = insts
            .get(idx)
            .map(|i| i.operand_ids.clone())
            .unwrap_or_default();
        let earliest_valid = insts
            .get(..idx)
            .unwrap_or(&[])
            .iter()
            .enumerate()
            .filter(|(_, p)| op_deps.contains(&p.id))
            .map(|(p_idx, _)| p_idx + 1)
            .max()
            .unwrap_or(0);
        let target_pos = idx.saturating_sub(max_back_steps).max(earliest_valid);
        if target_pos < idx {
            let cs = insts.remove(idx);
            insts.insert(target_pos, cs);
        }
    }
}

/// Recursively sets `LayoutProto.memory_space` across `shape` and any tuple sub-shapes.
///
/// # Arguments
///
/// * `shape` - Mutable [`ShapeProto`] to color.
/// * `color` - Memory space color (`0..=3`).
fn set_shape_memory_space_recursive(shape: &mut ShapeProto, color: i64) {
    for sub in &mut shape.tuple_shapes {
        set_shape_memory_space_recursive(sub, color);
    }
    shape.layout.as_mut().into_iter().for_each(|lay| {
        lay.memory_space = color;
    });
}

/// Checks whether `inst` matches an [`MsaHloPositionMatcherProto`] filter.
///
/// # Arguments
///
/// * `inst` - Instruction to inspect.
/// * `byte_size` - Physical byte size of `inst`.
/// * `matcher` - Position matcher filter.
fn matches_position_matcher(
    inst: &HloInstructionProto,
    byte_size: u64,
    matcher: &MsaHloPositionMatcherProto,
) -> bool {
    let sz_i64 = i64::try_from(byte_size).unwrap_or(i64::MAX);
    let name_ok = matcher
        .instruction_name_regex
        .as_ref()
        .is_none_or(|pat| inst.name.contains(pat.as_str()));
    let op_ok = matcher
        .instruction_regex
        .as_ref()
        .is_none_or(|pat| inst.opcode.contains(pat.as_str()) || inst.name.contains(pat.as_str()));
    let min_ok = matcher.size_gte.is_none_or(|min_sz| sz_i64 >= min_sz);
    let max_ok = matcher.size_lte.is_none_or(|max_sz| sz_i64 <= max_sz);
    name_ok && op_ok && min_ok && max_ok
}

/// Checks whether `inst` matches an [`MsaHloOperandFilterProto`] filter.
///
/// # Arguments
///
/// * `inst` - Instruction to inspect.
/// * `filter` - Operand filter specification.
fn matches_operand_filter(inst: &HloInstructionProto, filter: &MsaHloOperandFilterProto) -> bool {
    let name_ok = filter
        .instruction_name_regex
        .as_ref()
        .is_none_or(|pat| inst.name.contains(pat.as_str()));
    let op_ok = filter
        .instruction_regex
        .as_ref()
        .is_none_or(|pat| inst.opcode.contains(pat.as_str()) || inst.name.contains(pat.as_str()));
    let num_ok = filter
        .operand_number
        .is_none_or(|op_num| usize::try_from(op_num).is_ok_and(|n| n < inst.operand_ids.len()));
    name_ok && op_ok && num_ok
}

/// Complete heap simulation output bundle emitted by [`GlobalDecreasingSizeBestFitHeapSimulator`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HeapSimulationResult {
    /// Complete XLA [`BufferAssignmentProto`] with byte offsets, aliases, traces, and peak buffers.
    pub buffer_assignment: BufferAssignmentProto,
    /// Per-memory-space allocation breakdown [`MemoryUsageReportProto`].
    pub memory_usage_report: MemoryUsageReportProto,
    /// Zero-copy input-output buffer donation summary [`InputOutputBufferAliasConfigProto`].
    pub alias_config: InputOutputBufferAliasConfigProto,
    /// Aggregate PJRT [`CompiledMemoryStatsProto`] including serialized `BufferAssignmentProto`.
    pub compiled_memory_stats: CompiledMemoryStatsProto,
}

/// Spacetime rectangle packed by [`GlobalDecreasingSizeBestFitHeapSimulator`].
#[derive(Debug, Clone, Copy)]
struct PlacedSpacetimeChunk {
    /// Logical buffer ID.
    logical_id: i64,
    /// Birth step index in the schedule.
    birth: usize,
    /// Death step index in the schedule (inclusive).
    death: usize,
    /// Byte offset inside the memory-space heap.
    offset: u64,
    /// Aligned byte size.
    size: u64,
    /// `PrimitiveType` discriminant.
    element_type: i32,
}

/// Candidate scratchpad buffer `(aligned_bytes, birth_step, death_step, buf_id, elem_type, name)`.
type HeapCandidateEntry = (u64, usize, usize, i64, i32, String);

/// `OpenXLA` `GlobalDecreasingSizeBestFitHeapSimulator` spacetime buffer allocator.
///
/// Computes exact byte offsets, honors `HloInputOutputAliasProto` & `HloBufferDonorProto`
/// zero-copy buffer donation, and produces [`BufferAssignmentProto`] + [`CompiledMemoryStatsProto`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalDecreasingSizeBestFitHeapSimulator {
    /// Estimated generated executable code size in bytes reported in [`CompiledMemoryStatsProto`].
    pub generated_code_size_bytes: i64,
}

impl GlobalDecreasingSizeBestFitHeapSimulator {
    /// Constructs a [`GlobalDecreasingSizeBestFitHeapSimulator`].
    ///
    /// # Arguments
    ///
    /// * `generated_code_size_bytes` - Executable code size reported in `CompiledMemoryStatsProto`.
    #[must_use]
    pub const fn new(generated_code_size_bytes: i64) -> Self {
        Self {
            generated_code_size_bytes,
        }
    }

    /// Runs global decreasing-size best-fit spacetime allocation over `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - [`HloModuleProto`] (mutated only to synchronize newly synthesized `input_output_alias` entries from `buffer_donor`).
    #[allow(clippy::too_many_lines)]
    pub fn run(&self, module: &mut HloModuleProto) -> HeapSimulationResult {
        let Some(entry) = module
            .computations
            .iter()
            .find(|c| c.id == module.entry_computation_id)
            .or_else(|| module.computations.first())
            .cloned()
        else {
            return HeapSimulationResult::default();
        };

        let inst_by_id: BTreeMap<i64, &HloInstructionProto> =
            entry.instructions.iter().map(|i| (i.id, i)).collect();
        let ordered_insts: Vec<&HloInstructionProto> = module
            .schedule
            .as_ref()
            .and_then(|s| s.sequences.get(&entry.id))
            .filter(|seq| seq.instruction_ids.len() == entry.instructions.len())
            .map_or_else(
                || entry.instructions.iter().collect(),
                |seq| {
                    seq.instruction_ids
                        .iter()
                        .filter_map(|id| inst_by_id.get(id).copied())
                        .collect()
                },
            );

        let n_steps = ordered_insts.len();
        let mut birth_step: BTreeMap<i64, usize> = BTreeMap::new();
        let mut death_step: BTreeMap<i64, usize> = BTreeMap::new();
        for (step, inst) in ordered_insts.iter().enumerate() {
            birth_step.insert(inst.id, step);
            death_step.insert(inst.id, step);
            for &op_id in &inst.operand_ids {
                death_step.insert(op_id, step);
            }
            if inst.id == entry.root_id {
                death_step.insert(inst.id, n_steps.saturating_sub(1));
            }
        }

        let root_inst = ordered_insts
            .iter()
            .copied()
            .find(|i| i.id == entry.root_id)
            .or_else(|| ordered_insts.last().copied());
        let donated_param = resolve_donated_parameter(module, &ordered_insts, root_inst);

        let mut logical_buffers: Vec<LogicalBufferProto> = Vec::with_capacity(ordered_insts.len());
        let mut buffer_allocations: Vec<BufferAllocationProto> = Vec::new();
        let mut buffer_aliases: Vec<BufferAssignmentAliasProto> = Vec::new();
        let mut alias_entries: Vec<BufferDonationAliasEntryProto> = Vec::new();
        let mut total_aliased_bytes = 0_u64;
        let mut argument_size_bytes = 0_i64;
        let mut output_size_bytes = 0_i64;
        let mut next_alloc_idx = 0_i64;

        let mut heap_candidates_by_color: BTreeMap<i64, Vec<HeapCandidateEntry>> = BTreeMap::new();
        let mut donated_alloc_idx_and_buf: Option<(i64, i64, u64, i64, bool)> = None;

        for inst in &ordered_insts {
            let color = inst
                .shape
                .as_ref()
                .and_then(|s| s.layout.as_ref())
                .map_or(0, |l| l.memory_space);
            let space = TpuMsaMemorySpace::from_color(color);
            let align = space.alignment_bytes();
            let raw_sz = inst.shape.as_ref().map_or(256, shape_bytes_256_aligned);
            let aligned_sz = raw_sz.div_ceil(align) * align;
            let sz_i64 = i64::try_from(aligned_sz).unwrap_or(0);
            let elem_ty = inst
                .shape
                .as_ref()
                .map_or(PrimitiveType::F32 as i32, |s| s.element_type);
            let is_root = Some(inst.id) == root_inst.map(|r| r.id);

            logical_buffers.push(LogicalBufferProto {
                id: inst.id,
                size: sz_i64,
                defined_at: Some(LogicalBufferLocationProto {
                    instruction_name: inst.name.clone(),
                    instruction_id: inst.id,
                    shape_index: Vec::new(),
                }),
                color,
            });

            if inst.opcode == "parameter" {
                argument_size_bytes = argument_size_bytes.saturating_add(sz_i64);
                let matched_donor =
                    donated_param.filter(|&(p_num, _)| p_num == inst.parameter_number);
                let alloc_idx = next_alloc_idx;
                next_alloc_idx = next_alloc_idx.saturating_add(1);
                if let Some((don_pnum, must_alias)) = matched_donor {
                    donated_alloc_idx_and_buf =
                        Some((alloc_idx, inst.id, aligned_sz, don_pnum, must_alias));
                }
                buffer_allocations.push(BufferAllocationProto {
                    index: alloc_idx,
                    size: sz_i64,
                    is_entry_computation_parameter: true,
                    parameter_number: inst.parameter_number,
                    is_parameter_aliased_with_output: matched_donor.is_some(),
                    maybe_live_out: matched_donor.is_some() || is_root,
                    color,
                    assigned: vec![BufferAllocationAssignedProto {
                        logical_buffer_id: inst.id,
                        offset: 0,
                        size: sz_i64,
                        element_type: elem_ty,
                    }],
                    ..BufferAllocationProto::default()
                });
                continue;
            }

            if is_root {
                output_size_bytes = output_size_bytes.saturating_add(sz_i64);
                if let Some((don_alloc_idx, don_buf_id, _, don_pnum, must_alias)) =
                    donated_alloc_idx_and_buf.filter(|&(_, _, b_sz, _, _)| b_sz == aligned_sz)
                {
                    buffer_allocations
                        .iter_mut()
                        .find(|a| a.index == don_alloc_idx)
                        .into_iter()
                        .for_each(|don_alloc| {
                            don_alloc.assigned.push(BufferAllocationAssignedProto {
                                logical_buffer_id: inst.id,
                                offset: 0,
                                size: sz_i64,
                                element_type: elem_ty,
                            });
                        });
                    buffer_aliases.push(BufferAssignmentAliasProto {
                        source_buffer_id: don_buf_id,
                        location: Some(LogicalBufferLocationProto {
                            instruction_name: inst.name.clone(),
                            instruction_id: inst.id,
                            shape_index: Vec::new(),
                        }),
                    });
                    alias_entries.push(BufferDonationAliasEntryProto {
                        output_tuple_index: 0,
                        parameter_number: u32::try_from(don_pnum.max(0)).unwrap_or(0),
                        parameter_tuple_index: 0,
                        must_alias,
                        aliased_size_bytes: aligned_sz,
                    });
                    total_aliased_bytes = total_aliased_bytes.saturating_add(aligned_sz);
                    continue;
                }
                let out_alloc_idx = next_alloc_idx;
                next_alloc_idx = next_alloc_idx.saturating_add(1);
                buffer_allocations.push(BufferAllocationProto {
                    index: out_alloc_idx,
                    size: sz_i64,
                    is_tuple: inst.opcode == "tuple",
                    maybe_live_out: true,
                    color,
                    assigned: vec![BufferAllocationAssignedProto {
                        logical_buffer_id: inst.id,
                        offset: 0,
                        size: sz_i64,
                        element_type: elem_ty,
                    }],
                    ..BufferAllocationProto::default()
                });
                continue;
            }

            if inst.opcode == "constant" {
                let c_idx = next_alloc_idx;
                next_alloc_idx = next_alloc_idx.saturating_add(1);
                buffer_allocations.push(BufferAllocationProto {
                    index: c_idx,
                    size: sz_i64,
                    is_constant: true,
                    color,
                    assigned: vec![BufferAllocationAssignedProto {
                        logical_buffer_id: inst.id,
                        offset: 0,
                        size: sz_i64,
                        element_type: elem_ty,
                    }],
                    ..BufferAllocationProto::default()
                });
                continue;
            }

            let b = birth_step.get(&inst.id).copied().unwrap_or(0);
            let d = death_step.get(&inst.id).copied().unwrap_or(b);
            heap_candidates_by_color.entry(color).or_default().push((
                aligned_sz,
                b,
                d,
                inst.id,
                elem_ty,
                inst.name.clone(),
            ));
        }

        let mut heap_traces: Vec<HeapSimulatorTraceProto> = Vec::new();
        let mut peak_buffers: Vec<BufferAssignmentPeakBuffersProto> = Vec::new();
        let mut hbm_temp_bytes = 0_i64;
        let mut vmem_temp_bytes = 0_i64;

        for (color, mut candidates) in heap_candidates_by_color {
            candidates.sort_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| (b.2.saturating_sub(b.1)).cmp(&(a.2.saturating_sub(a.1))))
                    .then_with(|| a.3.cmp(&b.3))
            });

            let space = TpuMsaMemorySpace::from_color(color);
            let align = space.alignment_bytes();
            let mut placed: Vec<PlacedSpacetimeChunk> = Vec::with_capacity(candidates.len());
            let mut heap_high_water = 0_u64;

            for &(sz, birth, death, buf_id, elem_ty, _) in &candidates {
                let offset = find_best_fit_spacetime_offset(&placed, birth, death, sz, align);
                heap_high_water = heap_high_water.max(offset.saturating_add(sz));
                placed.push(PlacedSpacetimeChunk {
                    logical_id: buf_id,
                    birth,
                    death,
                    offset,
                    size: sz,
                    element_type: elem_ty,
                });
            }

            let alloc_idx = next_alloc_idx;
            next_alloc_idx = next_alloc_idx.saturating_add(1);
            let heap_sz_i64 = i64::try_from(heap_high_water).unwrap_or(0);
            if color == 0 {
                hbm_temp_bytes = hbm_temp_bytes.saturating_add(heap_sz_i64);
            } else if color == 1 {
                vmem_temp_bytes = vmem_temp_bytes.saturating_add(heap_sz_i64);
            }

            let mut assigned: Vec<BufferAllocationAssignedProto> = placed
                .iter()
                .map(|c| BufferAllocationAssignedProto {
                    logical_buffer_id: c.logical_id,
                    offset: i64::try_from(c.offset).unwrap_or(0),
                    size: i64::try_from(c.size).unwrap_or(0),
                    element_type: c.element_type,
                })
                .collect();
            assigned.sort_by_key(|a| a.logical_buffer_id);

            buffer_allocations.push(BufferAllocationProto {
                index: alloc_idx,
                size: heap_sz_i64,
                color,
                assigned,
                ..BufferAllocationProto::default()
            });

            let name_lookup: BTreeMap<i64, String> = candidates
                .iter()
                .map(|(_, _, _, id, _, nm)| (*id, nm.clone()))
                .collect();
            let mut timed_events: Vec<(usize, u8, HeapSimulatorTraceEventProto)> = Vec::new();
            let mut offset_canon: BTreeMap<u64, i64> = BTreeMap::new();
            for chunk in &placed {
                let inst_name = name_lookup
                    .get(&chunk.logical_id)
                    .cloned()
                    .unwrap_or_default();
                if let Some(&canon_id) = offset_canon.get(&chunk.offset) {
                    timed_events.push((
                        chunk.birth,
                        0,
                        HeapSimulatorTraceEventProto {
                            kind: HeapSimulatorEventKindProto::ShareWith as i32,
                            buffer_id: chunk.logical_id,
                            computation_name: entry.name.clone(),
                            instruction_name: inst_name.clone(),
                            share_with_canonical_id: canon_id,
                        },
                    ));
                } else {
                    offset_canon.insert(chunk.offset, chunk.logical_id);
                    timed_events.push((
                        chunk.birth,
                        0,
                        HeapSimulatorTraceEventProto {
                            kind: HeapSimulatorEventKindProto::Alloc as i32,
                            buffer_id: chunk.logical_id,
                            computation_name: entry.name.clone(),
                            instruction_name: inst_name.clone(),
                            share_with_canonical_id: 0,
                        },
                    ));
                }
                timed_events.push((
                    chunk.death,
                    1,
                    HeapSimulatorTraceEventProto {
                        kind: HeapSimulatorEventKindProto::Free as i32,
                        buffer_id: chunk.logical_id,
                        computation_name: entry.name.clone(),
                        instruction_name: inst_name,
                        share_with_canonical_id: 0,
                    },
                ));
            }
            timed_events.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            heap_traces.push(HeapSimulatorTraceProto {
                events: timed_events.into_iter().map(|(_, _, ev)| ev).collect(),
                whole_module_simulation: true,
                buffer_allocation_index: alloc_idx,
            });

            peak_buffers.push(BufferAssignmentPeakBuffersProto {
                index: alloc_idx,
                logical_buffer_ids: placed.iter().map(|c| c.logical_id).collect(),
            });
        }

        let memory_usage_report = build_memory_usage_report(&buffer_allocations);
        let buffer_assignment = BufferAssignmentProto {
            logical_buffers,
            buffer_aliases,
            buffer_allocations,
            heap_simulator_traces: heap_traces,
            peak_buffers,
        };
        let serialized_ba = buffer_assignment.encode_proto();
        let aliased_i64 = i64::try_from(total_aliased_bytes).unwrap_or(0);
        let peak_hbm = argument_size_bytes
            .saturating_add(output_size_bytes)
            .saturating_sub(aliased_i64)
            .saturating_add(hbm_temp_bytes);

        let compiled_memory_stats = CompiledMemoryStatsProto {
            generated_code_size_in_bytes: self.generated_code_size_bytes,
            argument_size_in_bytes: argument_size_bytes,
            output_size_in_bytes: output_size_bytes,
            alias_size_in_bytes: aliased_i64,
            temp_size_in_bytes: hbm_temp_bytes.saturating_add(vmem_temp_bytes),
            host_generated_code_size_in_bytes: 0,
            host_argument_size_in_bytes: 0,
            host_output_size_in_bytes: 0,
            host_alias_size_in_bytes: 0,
            host_temp_size_in_bytes: 0,
            serialized_buffer_assignment: serialized_ba,
            peak_memory_in_bytes: peak_hbm,
            total_size_in_bytes: peak_hbm.saturating_add(vmem_temp_bytes),
            total_allocation_bytes: peak_hbm,
            indefinite_allocations: argument_size_bytes
                .saturating_add(output_size_bytes)
                .saturating_sub(aliased_i64),
            peak_unpadded_heap_bytes: hbm_temp_bytes,
        };

        let alias_config = InputOutputBufferAliasConfigProto {
            module_name: module.name.clone(),
            total_aliased_bytes,
            alias_entries,
        };

        HeapSimulationResult {
            buffer_assignment,
            memory_usage_report,
            alias_config,
            compiled_memory_stats,
        }
    }
}

/// Resolves `(parameter_number, must_alias)` for an eligible donated input parameter matching `root_inst` size.
///
/// # Arguments
///
/// * `module` - Mutable [`HloModuleProto`] whose `input_output_alias` is updated when `buffer_donor` matches.
/// * `ordered_insts` - Entry computation instructions.
/// * `root_inst` - Optional root output instruction.
fn resolve_donated_parameter(
    module: &mut HloModuleProto,
    ordered_insts: &[&HloInstructionProto],
    root_inst: Option<&HloInstructionProto>,
) -> Option<(i64, bool)> {
    if let Some(first) = module
        .input_output_alias
        .as_ref()
        .and_then(|io| io.entries.first())
    {
        return Some((first.parameter_number, first.kind == 2));
    }

    let root = root_inst?;
    let root_bytes = root.shape.as_ref().map_or(0, shape_bytes_256_aligned);
    let donors = module.buffer_donor.as_ref()?;
    for donor in &donors.entries {
        let matched = ordered_insts.iter().copied().find(|i| {
            i.opcode == "parameter"
                && i.parameter_number == donor.parameter_number
                && i.shape.as_ref().map_or(0, shape_bytes_256_aligned) == root_bytes
        });
        if matched.is_some() {
            let io = module
                .input_output_alias
                .get_or_insert_with(Default::default);
            io.entries.push(HloAliasEntryProto {
                output_shape_index: Vec::new(),
                parameter_number: donor.parameter_number,
                parameter_shape_index: donor.parameter_shape_index.clone(),
                kind: 1,
            });
            return Some((donor.parameter_number, false));
        }
    }
    None
}

/// Finds the smallest aligned byte offset whose `[offset, offset + size)` interval does not
/// overlap any temporally live chunk in `[birth, death]`, preferring the tightest best-fit gap.
///
/// # Arguments
///
/// * `placed` - Already placed spacetime chunks in the same memory space.
/// * `birth` - Candidate birth step (inclusive).
/// * `death` - Candidate death step (inclusive).
/// * `size` - Candidate aligned byte size.
/// * `align` - Required byte alignment.
fn find_best_fit_spacetime_offset(
    placed: &[PlacedSpacetimeChunk],
    birth: usize,
    death: usize,
    size: u64,
    align: u64,
) -> u64 {
    let mut interfering: Vec<(u64, u64)> = placed
        .iter()
        .filter(|c| !(c.death < birth || death < c.birth))
        .map(|c| (c.offset, c.offset.saturating_add(c.size)))
        .collect();
    if interfering.is_empty() {
        return 0;
    }
    interfering.sort_unstable();

    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(interfering.len());
    for (s, e) in interfering {
        if let Some(last) = merged.last_mut().filter(|l| s <= l.1) {
            last.1 = last.1.max(e);
        } else {
            merged.push((s, e));
        }
    }

    let mut best_gap_offset: Option<(u64, u64)> = None;
    let mut cursor = 0_u64;
    for &(occ_start, occ_end) in &merged {
        let aligned_cursor = cursor.div_ceil(align) * align;
        if aligned_cursor.saturating_add(size) <= occ_start {
            let slack = occ_start - (aligned_cursor + size);
            if best_gap_offset.is_none_or(|(best_slack, _)| slack < best_slack) {
                best_gap_offset = Some((slack, aligned_cursor));
            }
        }
        cursor = cursor.max(occ_end);
    }

    best_gap_offset.map_or_else(|| cursor.div_ceil(align) * align, |(_, off)| off)
}

/// Builds a [`MemoryUsageReportProto`] grouped by memory space color.
///
/// # Arguments
///
/// * `allocations` - Final [`BufferAllocationProto`] slice.
fn build_memory_usage_report(allocations: &[BufferAllocationProto]) -> MemoryUsageReportProto {
    let mut by_color: BTreeMap<i32, Vec<&BufferAllocationProto>> = BTreeMap::new();
    for alloc in allocations {
        let color_i32 = i32::try_from(alloc.color).unwrap_or(0);
        by_color.entry(color_i32).or_default().push(alloc);
    }

    let mut space_entries = Vec::with_capacity(by_color.len());
    for (color, mut list) in by_color {
        list.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.index.cmp(&b.index)));
        let total: i64 = list.iter().map(|a| a.size).sum();
        let denom = total.max(1);
        let mut cum = 0_i64;
        let mut entries = Vec::with_capacity(list.len());
        for a in list {
            cum = cum.saturating_add(a.size);
            #[allow(clippy::cast_precision_loss)]
            let pct = (cum as f64) / (denom as f64) * 100.0;
            entries.push(MemoryUsageAllocationEntryProto {
                index: a.index,
                size: a.size,
                cumulative_size: cum,
                cumulative_percentage: pct,
                defining_positions: a
                    .assigned
                    .iter()
                    .map(|asg| format!("buf#{}", asg.logical_buffer_id))
                    .collect(),
                hlo_text: format!("allocation_{}_color_{color}", a.index),
            });
        }
        space_entries.push(MemoryUsageSpaceEntryProto {
            total_bytes: total,
            memory_space_color: Some(color),
            allocation_entries: entries,
        });
    }

    MemoryUsageReportProto {
        memory_space_allocation_entries: space_entries,
    }
}

/// Detailed roofline + learned execution time and bandwidth report emitted by [`MiddleEndCostModelEvaluator`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MiddleEndCostReport {
    /// Aggregate FLOP and transcendental op counts (`xla.ComputationStats`).
    pub computation_stats: ComputationStats,
    /// Total HBM bytes read/written across the module.
    pub hbm_bytes_accessed: u64,
    /// Total VMEM bytes read/written across the module.
    pub vmem_bytes_accessed: u64,
    /// Total SMEM bytes read/written across the module.
    pub smem_bytes_accessed: u64,
    /// Total CMEM bytes read/written across the module.
    pub cmem_bytes_accessed: u64,
    /// Analytical hardware roofline lower-bound latency in nanoseconds.
    pub roofline_time_ns: f64,
    /// Final cost-model latency estimate in nanoseconds after applying `EmitterLearnedCostModelOptions`.
    pub estimated_time_ns: f64,
    /// Operational arithmetic intensity (`FLOPs / max(1, hbm_bytes_accessed)`).
    pub arithmetic_intensity_flops_per_hbm_byte: f64,
}

/// Evaluates [`HloModuleProto`] FLOPs, transcendentals, per-memory-space byte traffic, and
/// roofline + learned execution time under `xla.jellyfish.EmitterLearnedCostModelOptions`.
#[derive(Debug, Clone)]
pub struct MiddleEndCostModelEvaluator {
    /// Target Cloud TPU generation determining peak MXU TFLOP/s and HBM/VMEM GB/s.
    pub generation: TpuGeneration,
    /// Learned cost model options (`emitter_learned_cost_model_options.proto`).
    pub options: EmitterLearnedCostModelOptionsProto,
    /// Optional embedded profile DB table mapping computation/instruction name -> calibrated latency (ns).
    pub profile_db_ns: BTreeMap<String, f64>,
}

impl MiddleEndCostModelEvaluator {
    /// Constructs a [`MiddleEndCostModelEvaluator`] for `generation` and `options`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation.
    /// * `options` - Protobuf `EmitterLearnedCostModelOptions` flags.
    #[must_use]
    pub const fn new(
        generation: TpuGeneration,
        options: EmitterLearnedCostModelOptionsProto,
    ) -> Self {
        Self {
            generation,
            options,
            profile_db_ns: BTreeMap::new(),
        }
    }

    /// Evaluates cost metrics across all reachable computations in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - [`HloModuleProto`] to analyze.
    #[must_use]
    pub fn evaluate_module(&self, module: &HloModuleProto) -> MiddleEndCostReport {
        let mut flops = 0.0_f64;
        let mut transcendentals = 0.0_f64;
        let mut hbm_bytes = 0_u64;
        let mut vmem_bytes = 0_u64;
        let mut smem_bytes = 0_u64;
        let mut cmem_bytes = 0_u64;

        for comp in &module.computations {
            let shapes: BTreeMap<i64, &ShapeProto> = comp
                .instructions
                .iter()
                .filter_map(|i| i.shape.as_ref().map(|s| (i.id, s)))
                .collect();

            for inst in &comp.instructions {
                if matches!(inst.opcode.as_str(), "parameter" | "constant") {
                    continue;
                }
                let (inst_flops, inst_trans) = instruction_flops_and_transcendentals(inst, &shapes);
                flops += inst_flops;
                transcendentals += inst_trans;

                for op_sh in inst
                    .operand_ids
                    .iter()
                    .filter_map(|id| shapes.get(id).copied())
                {
                    accumulate_space_bytes(
                        op_sh,
                        &mut hbm_bytes,
                        &mut vmem_bytes,
                        &mut smem_bytes,
                        &mut cmem_bytes,
                    );
                }
                inst.shape.as_ref().into_iter().for_each(|out_sh| {
                    accumulate_space_bytes(
                        out_sh,
                        &mut hbm_bytes,
                        &mut vmem_bytes,
                        &mut smem_bytes,
                        &mut cmem_bytes,
                    );
                });
            }
        }

        let (peak_tflops, hbm_gbps, vmem_gbps) = hardware_roofline_rates(self.generation);
        let compute_ns = transcendentals.mul_add(8.0, flops) / (peak_tflops * 1_000.0);
        #[allow(clippy::cast_precision_loss)]
        let hbm_ns = (hbm_bytes as f64) / hbm_gbps;
        #[allow(clippy::cast_precision_loss)]
        let vmem_ns = ((vmem_bytes + cmem_bytes + smem_bytes) as f64) / vmem_gbps;
        let roofline_ns = compute_ns.max(hbm_ns).max(vmem_ns).max(1.0);

        let ml_predicted_ns = roofline_ns * 0.04_f64.mul_add(transcendentals / (flops + 1.0), 1.12);
        let db_hit = self.profile_db_ns.get(&module.name).copied();

        let mode = JellyfishLearnedCostModelModeProto::try_from(self.options.cost_model_mode)
            .unwrap_or(JellyfishLearnedCostModelModeProto::Invalid);
        let raw_estimated = match mode {
            JellyfishLearnedCostModelModeProto::OnlyDb => db_hit.unwrap_or(roofline_ns),
            JellyfishLearnedCostModelModeProto::DbWithFallbackToMlPrediction => {
                db_hit.unwrap_or(ml_predicted_ns)
            }
            JellyfishLearnedCostModelModeProto::OnlyMlPrediction => ml_predicted_ns,
            JellyfishLearnedCostModelModeProto::OnlyDataCollection
            | JellyfishLearnedCostModelModeProto::Invalid => roofline_ns,
        };

        let val_strat = JellyfishMlOutputValidationStrategyProto::try_from(
            self.options.ml_output_validation_strategy,
        )
        .unwrap_or(JellyfishMlOutputValidationStrategyProto::None);
        let validated_ns = match val_strat {
            JellyfishMlOutputValidationStrategyProto::NeverTrust => roofline_ns,
            JellyfishMlOutputValidationStrategyProto::NoNegativeCycles => raw_estimated.max(0.0),
            JellyfishMlOutputValidationStrategyProto::AlwaysTrust
            | JellyfishMlOutputValidationStrategyProto::None => raw_estimated,
        };

        #[allow(clippy::cast_precision_loss)]
        let ai = flops / ((hbm_bytes.max(1)) as f64);

        MiddleEndCostReport {
            computation_stats: ComputationStats {
                flop_count: flops,
                transcendental_count: transcendentals,
            },
            hbm_bytes_accessed: hbm_bytes,
            vmem_bytes_accessed: vmem_bytes,
            smem_bytes_accessed: smem_bytes,
            cmem_bytes_accessed: cmem_bytes,
            roofline_time_ns: roofline_ns,
            estimated_time_ns: validated_ns,
            arithmetic_intensity_flops_per_hbm_byte: ai,
        }
    }
}

/// Computes `(flops, transcendentals)` for a single [`HloInstructionProto`].
///
/// # Arguments
///
/// * `inst` - Instruction to evaluate.
/// * `shapes` - Map from instruction ID to [`ShapeProto`].
fn instruction_flops_and_transcendentals(
    inst: &HloInstructionProto,
    shapes: &BTreeMap<i64, &ShapeProto>,
) -> (f64, f64) {
    let out_elems = inst.shape.as_ref().map_or(1.0_f64, shape_elem_count_f64);

    if inst.opcode == "dot"
        && let Some(lhs_sh) = inst
            .operand_ids
            .first()
            .and_then(|id| shapes.get(id).copied())
    {
        let k_dims = inst
            .dot_dimension_numbers
            .as_ref()
            .map_or(&[1_i64][..], |d| d.lhs_contracting_dimensions.as_slice());
        let k_prod = k_dims.iter().fold(1.0_f64, |acc, &kd| {
            let ext = usize::try_from(kd)
                .ok()
                .and_then(|u| lhs_sh.dimensions.get(u).copied())
                .unwrap_or(1)
                .max(1);
            #[allow(clippy::cast_precision_loss)]
            let ext_f = ext as f64;
            acc * ext_f
        });
        return (2.0 * out_elems * k_prod, 0.0);
    }

    match inst.opcode.as_str() {
        "exponential" | "log" | "tanh" | "rsqrt" | "sqrt" | "sine" | "cosine" | "logistic"
        | "erf" | "power" => (0.0, out_elems),
        "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum" | "negate" | "abs"
        | "reduce" => (out_elems, 0.0),
        _ => (0.0, 0.0),
    }
}

/// Accumulates logical byte size of `shape` into the counter matching `shape.layout.memory_space`.
///
/// # Arguments
///
/// * `shape` - Tensor shape to account.
/// * `hbm` - HBM byte counter.
/// * `vmem` - VMEM byte counter.
/// * `smem` - SMEM byte counter.
/// * `cmem` - CMEM byte counter.
fn accumulate_space_bytes(
    shape: &ShapeProto,
    hbm: &mut u64,
    vmem: &mut u64,
    smem: &mut u64,
    cmem: &mut u64,
) {
    if !shape.tuple_shapes.is_empty() {
        for sub in &shape.tuple_shapes {
            accumulate_space_bytes(sub, hbm, vmem, smem, cmem);
        }
        return;
    }
    let bits = u64::try_from(primitive_type_bit_width(shape.element_type).max(4)).unwrap_or(32);
    let elems = shape.dimensions.iter().fold(1_u64, |acc, &d| {
        acc.saturating_mul(u64::try_from(d.max(1)).unwrap_or(1))
    });
    let bytes = elems.saturating_mul(bits).div_ceil(8);
    let color = shape.layout.as_ref().map_or(0, |l| l.memory_space);
    match color {
        1 => *vmem = vmem.saturating_add(bytes),
        2 => *smem = smem.saturating_add(bytes),
        3 => *cmem = cmem.saturating_add(bytes),
        _ => *hbm = hbm.saturating_add(bytes),
    }
}

/// Computes total scalar element count in `shape` as `f64`.
///
/// # Arguments
///
/// * `shape` - Tensor [`ShapeProto`].
fn shape_elem_count_f64(shape: &ShapeProto) -> f64 {
    shape.dimensions.iter().fold(1.0_f64, |acc, &d| {
        #[allow(clippy::cast_precision_loss)]
        let df = (d.max(1)) as f64;
        acc * df
    })
}

/// Returns `(peak_bf16_tflops_per_core, hbm_bandwidth_gbps, vmem_bandwidth_gbps)` for `generation`.
///
/// # Arguments
///
/// * `generation` - Target Cloud TPU generation.
const fn hardware_roofline_rates(generation: TpuGeneration) -> (f64, f64, f64) {
    match generation {
        TpuGeneration::V2 => (22.5, 300.0, 4_000.0),
        TpuGeneration::V3 => (61.5, 450.0, 6_000.0),
        TpuGeneration::V4Lite => (68.0, 400.0, 6_400.0),
        TpuGeneration::V4 => (137.5, 600.0, 12_800.0),
        TpuGeneration::V5e => (197.0, 819.0, 16_000.0),
        TpuGeneration::V5p => (229.5, 1_382.0, 20_480.0),
        TpuGeneration::V6e => (918.0, 1_638.0, 32_768.0),
        TpuGeneration::V7x => (2_307.0, 3_686.0, 65_536.0),
    }
}
