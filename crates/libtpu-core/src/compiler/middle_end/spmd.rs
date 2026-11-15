//! Shardy (`sdy`) bidirectional sharding propagation and GSPMD partitioner passes.
//!
//! Implements [`ShardyPropagationPass`] (priority-tiered forward/backward sharding propagation
//! with closed/open dimensions, sub-axes, and unreduced axes) and [`GspmdPartitionerPass`]
//! (local tensor shape partitioning and synthesis of `AllReduce`, `ReduceScatter`, `AllGather`,
//! `AllToAll`, `CollectivePermute`, and `RaggedAllToAll` HLO collectives with exact
//! [`ReplicaGroup`] and `channel_id` assignments).

use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    FrontendAttributes, HloComputationProto, HloInstructionProto, HloModuleProto, OpShardingProto,
    ReplicaGroup, ShapeProto, ShardyAxisRefProto, ShardyDimensionShardingProto,
    ShardyMeshAxisProto, ShardyMeshProto, ShardyTensorShardingProto, SourceTarget,
};
use std::collections::{BTreeMap, BTreeSet};

/// Frontend attribute key storing serialized JSON [`ShardyTensorShardingProto`] on an [`HloInstructionProto`].
pub const SDY_SHARDING_ATTR_KEY: &str = "xla.sdy.sharding";

/// Frontend attribute key requesting cyclic halo shift (`collective-permute`) during GSPMD partitioning.
pub const SPMD_HALO_SHIFT_ATTR_KEY: &str = "xla.spmd.halo_shift";

/// Frontend attribute key requesting dynamic `MoE` `ragged-all-to-all` redistribution during GSPMD partitioning.
pub const SPMD_RAGGED_RESHARD_ATTR_KEY: &str = "xla.spmd.ragged_reshard";

/// Summary report emitted by [`ShardyPropagationPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShardyPropagationReport {
    /// Total fixed-point propagation iterations executed across all priority tiers.
    pub iterations: usize,
    /// Number of distinct priority tiers evaluated in ascending priority order.
    pub priority_tiers_evaluated: usize,
    /// Total number of producer-consumer or consumer-producer edges that propagated shardings.
    pub propagated_edges: usize,
    /// Number of instructions annotated with non-empty `unreduced_axes`.
    pub unreduced_instructions: usize,
}

/// Bidirectional `OpenXLA` Shardy (`sdy`) sharding propagation pass (`ShardyPropagationPass`).
#[derive(Debug, Clone)]
pub struct ShardyPropagationPass {
    /// Logical device mesh over which shardings are validated and propagated.
    pub mesh: ShardyMeshProto,
    /// Maximum fixed-point iterations per priority tier.
    pub max_iterations_per_tier: usize,
}

impl ShardyPropagationPass {
    /// Constructs a new [`ShardyPropagationPass`] for the given [`ShardyMeshProto`].
    ///
    /// # Arguments
    ///
    /// * `mesh` - Logical Shardy device mesh definition.
    #[must_use]
    pub const fn new(mesh: ShardyMeshProto) -> Self {
        Self {
            mesh,
            max_iterations_per_tier: 8,
        }
    }

    /// Constructs a default 1D `"data"` mesh [`ShardyPropagationPass`] with `num_partitions` devices.
    ///
    /// # Arguments
    ///
    /// * `num_partitions` - Number of SPMD partitions along the `"data"` axis.
    #[must_use]
    pub fn with_linear_mesh(num_partitions: usize) -> Self {
        let size = i64::try_from(num_partitions.max(1)).unwrap_or(1);
        Self::new(ShardyMeshProto {
            name: String::from("mesh"),
            axes: vec![ShardyMeshAxisProto {
                name: String::from("data"),
                size,
            }],
            device_ids: (0..size).collect(),
        })
    }

    /// Runs bidirectional priority-ordered sharding propagation across all computations in `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] whose instruction, parameter, and output shardings are updated.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if any instruction references an unknown mesh axis
    /// or an out-of-range [`crate::proto::ShardySubAxisInfoProto`] factor.
    pub fn run(&self, module: &mut HloModuleProto) -> LibTpuResult<ShardyPropagationReport> {
        let axis_sizes = build_mesh_axis_map(&self.mesh)?;
        let mut report = ShardyPropagationReport::default();

        for comp in &mut module.computations {
            let comp_report = self.propagate_computation(comp, &axis_sizes)?;
            report.iterations = report.iterations.saturating_add(comp_report.iterations);
            report.priority_tiers_evaluated = report
                .priority_tiers_evaluated
                .max(comp_report.priority_tiers_evaluated);
            report.propagated_edges = report
                .propagated_edges
                .saturating_add(comp_report.propagated_edges);
            report.unreduced_instructions = report
                .unreduced_instructions
                .saturating_add(comp_report.unreduced_instructions);
        }

        sync_module_entry_shardings(module);
        Ok(report)
    }

    /// Propagates shardings within a single [`HloComputationProto`].
    ///
    /// # Arguments
    ///
    /// * `comp` - Mutable computation to propagate shardings across.
    /// * `axis_sizes` - Validated mapping from mesh axis name to axis extent.
    fn propagate_computation(
        &self,
        comp: &mut HloComputationProto,
        axis_sizes: &BTreeMap<String, i64>,
    ) -> LibTpuResult<ShardyPropagationReport> {
        let mut shardings: BTreeMap<i64, ShardyTensorShardingProto> = BTreeMap::new();
        let mut priorities: BTreeSet<i64> = BTreeSet::new();

        for inst in &comp.instructions {
            let rank = instruction_rank(inst);
            let parsed = extract_instruction_sdy_sharding(inst, &self.mesh.name, axis_sizes, rank)?;
            validate_sharding_against_mesh(&parsed, axis_sizes)?;
            for dim_sh in &parsed.dim_shardings {
                if !dim_sh.axes.is_empty() {
                    priorities.insert(dim_sh.priority);
                }
            }
            shardings.insert(inst.id, parsed);
        }
        if priorities.is_empty() {
            priorities.insert(0);
        }

        let mut report = ShardyPropagationReport {
            priority_tiers_evaluated: priorities.len(),
            ..ShardyPropagationReport::default()
        };

        for &max_priority in &priorities {
            for _ in 0..self.max_iterations_per_tier {
                report.iterations = report.iterations.saturating_add(1);
                let mut changed_in_iter = false;

                // Forward pass: operands -> consumer
                for inst in &comp.instructions {
                    let cur_sh = shardings.get(&inst.id).cloned().unwrap_or_default();
                    let updated =
                        propagate_forward_to_instruction(inst, &cur_sh, &shardings, max_priority);
                    if updated != cur_sh {
                        shardings.insert(inst.id, updated);
                        report.propagated_edges = report.propagated_edges.saturating_add(1);
                        changed_in_iter = true;
                    }
                }

                // Backward pass: consumer -> operands
                for inst in comp.instructions.iter().rev() {
                    let consumer_sh = shardings.get(&inst.id).cloned().unwrap_or_default();
                    for (op_pos, &op_id) in inst.operand_ids.iter().enumerate() {
                        let Some(cur_op_sh) = shardings.get(&op_id).cloned() else {
                            continue;
                        };
                        let updated_op = propagate_backward_to_operand(
                            inst,
                            op_pos,
                            &consumer_sh,
                            &cur_op_sh,
                            max_priority,
                        );
                        if updated_op != cur_op_sh {
                            shardings.insert(op_id, updated_op);
                            report.propagated_edges = report.propagated_edges.saturating_add(1);
                            changed_in_iter = true;
                        }
                    }
                }

                if !changed_in_iter {
                    break;
                }
            }
        }

        for inst in &mut comp.instructions {
            let sh = shardings.get(&inst.id).cloned().unwrap_or_default();
            if !sh.unreduced_axes.is_empty() {
                report.unreduced_instructions = report.unreduced_instructions.saturating_add(1);
            }
            write_sharding_to_instruction(inst, &sh, axis_sizes);
        }

        Ok(report)
    }
}

/// Summary report emitted by [`GspmdPartitionerPass`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GspmdPartitionReport {
    /// Number of instructions converted to local per-partition shapes.
    pub partitioned_instructions: usize,
    /// Number of synthesized `all-reduce` collectives.
    pub all_reduce_count: usize,
    /// Number of synthesized `reduce-scatter` collectives.
    pub reduce_scatter_count: usize,
    /// Number of synthesized `all-gather` collectives.
    pub all_gather_count: usize,
    /// Number of synthesized `all-to-all` collectives.
    pub all_to_all_count: usize,
    /// Number of synthesized `collective-permute` collectives.
    pub collective_permute_count: usize,
    /// Number of synthesized `ragged-all-to-all` collectives.
    pub ragged_all_to_all_count: usize,
}

/// GSPMD SPMD partitioner pass (`GspmdPartitionerPass`) lowering global `HloModuleProto` shapes
/// to per-partition local shapes and inserting cross-partition collectives.
#[derive(Debug, Clone)]
pub struct GspmdPartitionerPass {
    /// Logical Shardy device mesh governing `ReplicaGroup` device ordinals.
    pub mesh: ShardyMeshProto,
    /// Total number of SPMD partitions (`>= 1`).
    pub num_partitions: usize,
    /// Starting `channel_id` assigned to synthesized SPMD collectives.
    pub base_channel_id: i64,
}

impl GspmdPartitionerPass {
    /// Constructs a [`GspmdPartitionerPass`] for `mesh` and `num_partitions`.
    ///
    /// # Arguments
    ///
    /// * `mesh` - Logical device mesh used to compute axis-aligned [`ReplicaGroup`]s.
    /// * `num_partitions` - Total number of SPMD model-parallel partitions.
    #[must_use]
    pub fn new(mesh: ShardyMeshProto, num_partitions: usize) -> Self {
        Self {
            mesh,
            num_partitions: num_partitions.max(1),
            base_channel_id: 1,
        }
    }

    /// Partitions all computations in `module` into local per-partition shapes and synthesizes
    /// required SPMD collective instructions.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] to partition in-place.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if the mesh definition is invalid.
    pub fn run(&self, module: &mut HloModuleProto) -> LibTpuResult<GspmdPartitionReport> {
        let axis_sizes = build_mesh_axis_map(&self.mesh)?;
        let mut report = GspmdPartitionReport::default();
        let mut next_channel_id = self.base_channel_id.max(1);

        for comp in &mut module.computations {
            self.partition_computation(comp, &axis_sizes, &mut next_channel_id, &mut report)?;
        }

        if let Some(entry) = find_entry_computation(module) {
            let param_shapes: Vec<ShapeProto> = entry
                .instructions
                .iter()
                .filter(|inst| inst.opcode == "parameter")
                .filter_map(|inst| inst.shape.clone())
                .collect();
            let root_shape = find_root_instruction(entry).and_then(|i| i.shape.clone());
            if let Some(prog_shape) = &mut module.host_program_shape {
                let prev_res = prog_shape.result.take();
                prog_shape.parameters = param_shapes;
                prog_shape.result = root_shape.or(prev_res);
            }
        }

        Ok(report)
    }

    /// Partitions a single [`HloComputationProto`] and inserts resharding/reduction collectives.
    ///
    /// # Arguments
    ///
    /// * `comp` - Mutable computation to partition.
    /// * `axis_sizes` - Validated mesh axis sizes.
    /// * `next_channel_id` - Monotonic channel handle counter.
    /// * `report` - Mutable partition summary accumulator.
    #[allow(clippy::too_many_lines)]
    fn partition_computation(
        &self,
        comp: &mut HloComputationProto,
        axis_sizes: &BTreeMap<String, i64>,
        next_channel_id: &mut i64,
        report: &mut GspmdPartitionReport,
    ) -> LibTpuResult<()> {
        let mut next_inst_id = comp
            .instructions
            .iter()
            .map(|i| i.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut global_shapes: BTreeMap<i64, ShapeProto> = BTreeMap::new();
        let mut inst_shardings: BTreeMap<i64, ShardyTensorShardingProto> = BTreeMap::new();
        for inst in &comp.instructions {
            inst.shape.as_ref().into_iter().for_each(|sh| {
                global_shapes.insert(inst.id, sh.clone());
            });
            let rank = instruction_rank(inst);
            let parsed = extract_instruction_sdy_sharding(inst, &self.mesh.name, axis_sizes, rank)?;
            inst_shardings.insert(inst.id, parsed);
        }

        let original_instructions = std::mem::take(&mut comp.instructions);
        let mut new_instructions: Vec<HloInstructionProto> =
            Vec::with_capacity(original_instructions.len() * 2);
        let mut current_root_id = comp.root_id;

        for mut inst in original_instructions {
            let inst_id = inst.id;
            let dst_sharding = inst_shardings.get(&inst_id).cloned().unwrap_or_default();

            let is_dot_like = matches!(inst.opcode.as_str(), "dot" | "convolution");
            let mut updated_operands = Vec::with_capacity(inst.operand_ids.len());
            for &op_id in &inst.operand_ids {
                let src_sharding = inst_shardings.get(&op_id).cloned().unwrap_or_default();
                let op_global_shape = global_shapes.get(&op_id).cloned().unwrap_or_default();
                if is_dot_like {
                    updated_operands.push(op_id);
                    continue;
                }
                let rewritten_op_id = maybe_synthesize_edge_collective(
                    op_id,
                    &op_global_shape,
                    &src_sharding,
                    &dst_sharding,
                    &self.mesh,
                    axis_sizes,
                    self.num_partitions,
                    &mut next_inst_id,
                    next_channel_id,
                    &mut new_instructions,
                    report,
                );
                updated_operands.push(rewritten_op_id);
            }
            inst.operand_ids = updated_operands;

            inst.shape.as_mut().into_iter().for_each(|shape| {
                partition_shape_in_place(shape, &dst_sharding, axis_sizes);
                report.partitioned_instructions = report.partitioned_instructions.saturating_add(1);
            });

            let halo_shift = get_frontend_attr(&inst, SPMD_HALO_SHIFT_ATTR_KEY)
                .and_then(|v| v.parse::<i64>().ok());
            let ragged_reshard = get_frontend_attr(&inst, SPMD_RAGGED_RESHARD_ATTR_KEY)
                .is_some_and(|v| matches!(v, "true" | "1"));

            let local_shape = inst.shape.clone().unwrap_or_default();
            new_instructions.push(inst);
            let mut latest_id = inst_id;

            let multi_part = self.num_partitions > 1;
            if multi_part && !dst_sharding.unreduced_axes.is_empty() {
                let axis_name = dst_sharding
                    .unreduced_axes
                    .first()
                    .map_or("data", |a| a.name.as_str());
                let replica_groups =
                    build_replica_groups_for_axis(&self.mesh, axis_name, self.num_partitions);
                let scatter_dim = dst_sharding
                    .dim_shardings
                    .iter()
                    .position(|ds| ds.axes.iter().any(|a| a.name == axis_name));
                let ch_id = alloc_channel(next_channel_id);
                let coll_id = alloc_inst_id(&mut next_inst_id);
                if let Some(sdim) = scatter_dim {
                    report.reduce_scatter_count = report.reduce_scatter_count.saturating_add(1);
                    new_instructions.push(HloInstructionProto {
                        id: coll_id,
                        name: format!("spmd_reduce_scatter_{coll_id}"),
                        opcode: String::from("reduce-scatter"),
                        shape: Some(local_shape.clone()),
                        operand_ids: vec![latest_id],
                        dimensions: vec![i64::try_from(sdim).unwrap_or(0)],
                        replica_groups,
                        channel_id: ch_id,
                        use_global_device_ids: true,
                        ..HloInstructionProto::default()
                    });
                } else {
                    report.all_reduce_count = report.all_reduce_count.saturating_add(1);
                    new_instructions.push(HloInstructionProto {
                        id: coll_id,
                        name: format!("spmd_all_reduce_{coll_id}"),
                        opcode: String::from("all-reduce"),
                        shape: Some(local_shape.clone()),
                        operand_ids: vec![latest_id],
                        replica_groups,
                        channel_id: ch_id,
                        all_reduce_id: ch_id,
                        use_global_device_ids: true,
                        ..HloInstructionProto::default()
                    });
                }
                if current_root_id == latest_id {
                    current_root_id = coll_id;
                }
                remap_future_uses(&mut inst_shardings, inst_id, coll_id);
                latest_id = coll_id;
            }

            if let Some(shift) = halo_shift.filter(|_| multi_part) {
                let n = i64::try_from(self.num_partitions).unwrap_or(1);
                let pairs: Vec<SourceTarget> = (0..n)
                    .map(|src| SourceTarget {
                        source: src,
                        target: (src + shift).rem_euclid(n),
                    })
                    .collect();
                let chan_handle = alloc_channel(next_channel_id);
                let permute_inst_id = alloc_inst_id(&mut next_inst_id);
                report.collective_permute_count = report.collective_permute_count.saturating_add(1);
                new_instructions.push(HloInstructionProto {
                    id: permute_inst_id,
                    name: format!("spmd_collective_permute_{permute_inst_id}"),
                    opcode: String::from("collective-permute"),
                    shape: Some(local_shape.clone()),
                    operand_ids: vec![latest_id],
                    source_target_pairs: pairs,
                    channel_id: chan_handle,
                    ..HloInstructionProto::default()
                });
                if current_root_id == latest_id {
                    current_root_id = permute_inst_id;
                }
                latest_id = permute_inst_id;
            }

            if multi_part && ragged_reshard {
                let replica_groups =
                    build_replica_groups_for_axis(&self.mesh, "", self.num_partitions);
                let ch_id = alloc_channel(next_channel_id);
                let ra2a_id = alloc_inst_id(&mut next_inst_id);
                report.ragged_all_to_all_count = report.ragged_all_to_all_count.saturating_add(1);
                new_instructions.push(HloInstructionProto {
                    id: ra2a_id,
                    name: format!("spmd_ragged_all_to_all_{ra2a_id}"),
                    opcode: String::from("ragged-all-to-all"),
                    shape: Some(local_shape),
                    operand_ids: vec![latest_id],
                    replica_groups,
                    channel_id: ch_id,
                    use_global_device_ids: true,
                    ..HloInstructionProto::default()
                });
                if current_root_id == latest_id {
                    current_root_id = ra2a_id;
                }
            }
        }

        comp.instructions = new_instructions;
        comp.root_id = current_root_id;
        Ok(())
    }
}

/// Resolves the entry computation of `module` by `entry_computation_id` or falls back to the first computation.
///
/// # Arguments
///
/// * `module` - [`HloModuleProto`] to query.
fn find_entry_computation(module: &HloModuleProto) -> Option<&HloComputationProto> {
    module
        .computations
        .iter()
        .find(|c| c.id == module.entry_computation_id)
        .or_else(|| module.computations.first())
}

/// Resolves the root instruction of `comp` by `root_id` or falls back to the last instruction.
///
/// # Arguments
///
/// * `comp` - [`HloComputationProto`] to query.
fn find_root_instruction(comp: &HloComputationProto) -> Option<&HloInstructionProto> {
    comp.instructions
        .iter()
        .find(|i| i.id == comp.root_id)
        .or_else(|| comp.instructions.last())
}

/// Allocates a monotonically increasing instruction ID.
///
/// # Arguments
///
/// * `next_id` - Mutable counter holding the next available instruction ID.
const fn alloc_inst_id(next_id: &mut i64) -> i64 {
    let id = *next_id;
    *next_id = next_id.saturating_add(1);
    id
}

/// Allocates a monotonically increasing SPMD collective channel ID.
///
/// # Arguments
///
/// * `next_channel` - Mutable counter holding the next available channel ID.
const fn alloc_channel(next_channel: &mut i64) -> i64 {
    let ch = *next_channel;
    *next_channel = next_channel.saturating_add(1);
    ch
}

/// Copies a sharding entry under a newly inserted collective's instruction ID.
///
/// # Arguments
///
/// * `shardings` - Map from instruction ID to [`ShardyTensorShardingProto`].
/// * `old_id` - Original producer instruction ID.
/// * `new_id` - Newly synthesized collective instruction ID.
fn remap_future_uses(
    shardings: &mut BTreeMap<i64, ShardyTensorShardingProto>,
    old_id: i64,
    new_id: i64,
) {
    shardings
        .get(&old_id)
        .cloned()
        .into_iter()
        .for_each(|mut existing| {
            existing.unreduced_axes.clear();
            shardings.insert(new_id, existing);
        });
}

/// Synthesizes an `all-gather` or `all-to-all` resharding collective on a producer->consumer edge
/// when their dimension shardings differ and `num_partitions > 1`.
///
/// # Arguments
///
/// * `op_id` - Producer instruction ID.
/// * `op_global_shape` - Global unpartitioned shape of the producer tensor.
/// * `src_sharding` - Sharding of the producer instruction.
/// * `dst_sharding` - Sharding expected by the consumer instruction.
/// * `mesh` - Logical device mesh.
/// * `axis_sizes` - Validated mesh axis sizes.
/// * `num_partitions` - Total number of SPMD partitions.
/// * `next_inst_id` - Mutable instruction ID allocator.
/// * `next_channel_id` - Mutable channel ID allocator.
/// * `out_insts` - Instruction list receiving any synthesized collective.
/// * `report` - Mutable partition summary accumulator.
#[allow(clippy::too_many_arguments)]
fn maybe_synthesize_edge_collective(
    op_id: i64,
    op_global_shape: &ShapeProto,
    src_sharding: &ShardyTensorShardingProto,
    dst_sharding: &ShardyTensorShardingProto,
    mesh: &ShardyMeshProto,
    axis_sizes: &BTreeMap<String, i64>,
    num_partitions: usize,
    next_inst_id: &mut i64,
    next_channel_id: &mut i64,
    out_insts: &mut Vec<HloInstructionProto>,
    report: &mut GspmdPartitionReport,
) -> i64 {
    if num_partitions <= 1 || src_sharding.dim_shardings == dst_sharding.dim_shardings {
        return op_id;
    }

    let src_sharded_dim = src_sharding
        .dim_shardings
        .iter()
        .position(|ds| !ds.axes.is_empty());
    let dst_sharded_dim = dst_sharding
        .dim_shardings
        .iter()
        .position(|ds| !ds.axes.is_empty());

    match (src_sharded_dim, dst_sharded_dim) {
        (Some(s_dim), None) => {
            let axis_name = src_sharding
                .dim_shardings
                .get(s_dim)
                .and_then(|ds| ds.axes.first())
                .map_or("data", |a| a.name.as_str());
            let replica_groups = build_replica_groups_for_axis(mesh, axis_name, num_partitions);
            let ch_id = alloc_channel(next_channel_id);
            let ag_id = alloc_inst_id(next_inst_id);
            let mut gathered_shape = op_global_shape.clone();
            partition_shape_in_place(&mut gathered_shape, dst_sharding, axis_sizes);
            report.all_gather_count = report.all_gather_count.saturating_add(1);
            out_insts.push(HloInstructionProto {
                id: ag_id,
                name: format!("spmd_all_gather_{ag_id}"),
                opcode: String::from("all-gather"),
                shape: Some(gathered_shape),
                operand_ids: vec![op_id],
                dimensions: vec![i64::try_from(s_dim).unwrap_or(0)],
                replica_groups,
                channel_id: ch_id,
                use_global_device_ids: true,
                ..HloInstructionProto::default()
            });
            ag_id
        }
        (Some(s_dim), Some(d_dim)) if s_dim != d_dim => {
            let axis_name = src_sharding
                .dim_shardings
                .get(s_dim)
                .and_then(|ds| ds.axes.first())
                .map_or("data", |a| a.name.as_str());
            let replica_groups = build_replica_groups_for_axis(mesh, axis_name, num_partitions);
            let ch_id = alloc_channel(next_channel_id);
            let a2a_id = alloc_inst_id(next_inst_id);
            let mut resharded_shape = op_global_shape.clone();
            partition_shape_in_place(&mut resharded_shape, dst_sharding, axis_sizes);
            report.all_to_all_count = report.all_to_all_count.saturating_add(1);
            out_insts.push(HloInstructionProto {
                id: a2a_id,
                name: format!("spmd_all_to_all_{a2a_id}"),
                opcode: String::from("all-to-all"),
                shape: Some(resharded_shape),
                operand_ids: vec![op_id],
                dimensions: vec![i64::try_from(d_dim).unwrap_or(0)],
                replica_groups,
                channel_id: ch_id,
                use_global_device_ids: true,
                ..HloInstructionProto::default()
            });
            a2a_id
        }
        _ => op_id,
    }
}

/// Builds axis-aligned [`ReplicaGroup`] device lists for `axis_name` over `mesh`.
///
/// # Arguments
///
/// * `mesh` - Logical Shardy mesh definition.
/// * `axis_name` - Mesh axis name along which collective reduction/gathering occurs (empty for all partitions).
/// * `num_partitions` - Fallback partition count if `mesh` has no explicit `axes`.
#[must_use]
pub fn build_replica_groups_for_axis(
    mesh: &ShardyMeshProto,
    axis_name: &str,
    num_partitions: usize,
) -> Vec<ReplicaGroup> {
    let total_mesh_devs: i64 = mesh
        .axes
        .iter()
        .fold(1_i64, |acc, ax| acc.saturating_mul(ax.size.max(1)));
    let total = total_mesh_devs.max(i64::try_from(num_partitions.max(1)).unwrap_or(1));
    let devices: Vec<i64> = if mesh.device_ids.len() == usize::try_from(total).unwrap_or(0) {
        mesh.device_ids.clone()
    } else {
        (0..total).collect()
    };

    let Some(axis_pos) = mesh.axes.iter().position(|a| a.name == axis_name) else {
        return vec![ReplicaGroup {
            replica_ids: devices,
        }];
    };

    let axis_len = mesh.axes.get(axis_pos).map_or(total, |a| a.size.max(1));
    let inner_stride: i64 = mesh
        .axes
        .get(axis_pos + 1..)
        .unwrap_or(&[])
        .iter()
        .fold(1_i64, |acc, a| acc.saturating_mul(a.size.max(1)));
    let outer_count = total / (axis_len.saturating_mul(inner_stride)).max(1);

    let mut groups = Vec::new();
    for outer in 0..outer_count {
        for inner in 0..inner_stride {
            let ids: Vec<i64> = (0..axis_len)
                .filter_map(|k| {
                    let linear = outer
                        .saturating_mul(axis_len)
                        .saturating_mul(inner_stride)
                        .saturating_add(k.saturating_mul(inner_stride))
                        .saturating_add(inner);
                    usize::try_from(linear)
                        .ok()
                        .and_then(|idx| devices.get(idx).copied())
                })
                .collect();
            groups.push(ReplicaGroup { replica_ids: ids });
        }
    }
    groups
}

/// Builds and validates a map of mesh axis names to positive extents.
///
/// # Arguments
///
/// * `mesh` - Logical [`ShardyMeshProto`] to validate.
fn build_mesh_axis_map(mesh: &ShardyMeshProto) -> LibTpuResult<BTreeMap<String, i64>> {
    let mut map = BTreeMap::new();
    for axis in &mesh.axes {
        let invalid_axis = axis.name.trim().is_empty() || axis.size <= 0;
        if invalid_axis {
            return Err(LibTpuError::CompilationFailed {
                reason: format!(
                    "Invalid ShardyMeshProto axis '{}' with size {}",
                    axis.name, axis.size
                ),
            });
        }
        map.insert(axis.name.clone(), axis.size);
    }
    Ok(map)
}

/// Validates that every axis and sub-axis in `sharding` exists in `axis_sizes` and has legal bounds.
///
/// # Arguments
///
/// * `sharding` - Tensor sharding annotation to validate.
/// * `axis_sizes` - Map from mesh axis name to full axis size.
fn validate_sharding_against_mesh(
    sharding: &ShardyTensorShardingProto,
    axis_sizes: &BTreeMap<String, i64>,
) -> LibTpuResult<()> {
    if axis_sizes.is_empty() {
        return Ok(());
    }
    let all_refs = sharding
        .dim_shardings
        .iter()
        .flat_map(|d| d.axes.iter())
        .chain(sharding.replicated_axes.iter())
        .chain(sharding.unreduced_axes.iter());
    for ax_ref in all_refs {
        let Some(&full_sz) = axis_sizes.get(&ax_ref.name) else {
            return Err(LibTpuError::CompilationFailed {
                reason: format!(
                    "Shardy sharding references unknown mesh axis '{}'",
                    ax_ref.name
                ),
            });
        };
        if let Some(sub) = &ax_ref.sub_axis_info {
            let out_of_range = sub.pre_size <= 0
                || sub.size <= 0
                || sub.pre_size.saturating_mul(sub.size) > full_sz;
            if out_of_range {
                return Err(LibTpuError::CompilationFailed {
                    reason: format!(
                        "Invalid Shardy sub-axis pre_size={} size={} for axis '{}' (size={full_sz})",
                        sub.pre_size, sub.size, ax_ref.name
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Returns the tensor rank of an [`HloInstructionProto`].
///
/// # Arguments
///
/// * `inst` - Instruction whose output shape rank is queried.
fn instruction_rank(inst: &HloInstructionProto) -> usize {
    inst.shape.as_ref().map_or(0, |s| s.dimensions.len())
}

/// Reads a frontend attribute string value from `inst` by `key`.
///
/// # Arguments
///
/// * `inst` - Instruction whose `frontend_attributes` map is queried.
/// * `key` - Attribute key name.
fn get_frontend_attr<'a>(inst: &'a HloInstructionProto, key: &str) -> Option<&'a str> {
    inst.frontend_attributes
        .as_ref()
        .and_then(|fa| fa.map.get(key))
        .map(String::as_str)
}

/// Extracts or synthesizes a [`ShardyTensorShardingProto`] for `inst` of rank `rank`.
///
/// # Arguments
///
/// * `inst` - Instruction to inspect for `xla.sdy.sharding` or `OpShardingProto`.
/// * `mesh_name` - Default mesh symbol name.
/// * `axis_sizes` - Known mesh axes and sizes.
/// * `rank` - Expected tensor rank.
fn extract_instruction_sdy_sharding(
    inst: &HloInstructionProto,
    mesh_name: &str,
    axis_sizes: &BTreeMap<String, i64>,
    rank: usize,
) -> LibTpuResult<ShardyTensorShardingProto> {
    if let Some(raw_json) = get_frontend_attr(inst, SDY_SHARDING_ATTR_KEY) {
        use crate::proto::ProtoSchemaMessage as _;
        let mut decoded = ShardyTensorShardingProto::from_proto_json(raw_json).map_err(|err| {
            LibTpuError::CompilationFailed {
                reason: format!(
                    "Malformed {SDY_SHARDING_ATTR_KEY} JSON on '{}': {err}",
                    inst.name
                ),
            }
        })?;
        normalize_dim_sharding_rank(&mut decoded, rank);
        return Ok(decoded);
    }

    let mut dim_shardings = vec![
        ShardyDimensionShardingProto {
            axes: Vec::new(),
            is_closed: false,
            priority: 0,
        };
        rank
    ];
    if let Some(op_sh) = inst.sharding.as_ref().filter(|s| s.r#type == 3) {
        for (dim_idx, &factor) in op_sh
            .tile_assignment_dimensions
            .iter()
            .take(rank)
            .enumerate()
        {
            if factor > 1 {
                let axis_opt = axis_sizes
                    .iter()
                    .find(|&(_, &sz)| sz == factor)
                    .map(|(n, _)| n.clone())
                    .or_else(|| axis_sizes.keys().next().cloned());
                if let (Some(axis_name), Some(slot)) = (axis_opt, dim_shardings.get_mut(dim_idx)) {
                    slot.axes.push(ShardyAxisRefProto {
                        name: axis_name,
                        sub_axis_info: None,
                    });
                }
            }
        }
    }

    Ok(ShardyTensorShardingProto {
        mesh_name: String::from(mesh_name),
        dim_shardings,
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    })
}

/// Adjusts `sharding.dim_shardings` length so it matches `rank`.
///
/// # Arguments
///
/// * `sharding` - Mutable tensor sharding descriptor.
/// * `rank` - Target tensor rank.
fn normalize_dim_sharding_rank(sharding: &mut ShardyTensorShardingProto, rank: usize) {
    sharding.dim_shardings.resize(
        rank,
        ShardyDimensionShardingProto {
            axes: Vec::new(),
            is_closed: false,
            priority: 0,
        },
    );
}

/// Propagates shardings forward from `inst`'s operands onto `cur_sh` for shardings with `priority <= max_priority`.
///
/// # Arguments
///
/// * `inst` - Consumer instruction receiving forward-propagated shardings.
/// * `cur_sh` - Current sharding of `inst`.
/// * `shardings` - Current map of instruction ID to [`ShardyTensorShardingProto`].
/// * `max_priority` - Maximum numeric priority tier active in this phase.
fn propagate_forward_to_instruction(
    inst: &HloInstructionProto,
    cur_sh: &ShardyTensorShardingProto,
    shardings: &BTreeMap<i64, ShardyTensorShardingProto>,
    max_priority: i64,
) -> ShardyTensorShardingProto {
    let mut next = cur_sh.clone();

    if inst.opcode == "dot" {
        if let (Some(&lhs_id), Some(&rhs_id)) = (inst.operand_ids.first(), inst.operand_ids.get(1))
        {
            propagate_forward_dot(
                inst,
                shardings.get(&lhs_id),
                shardings.get(&rhs_id),
                &mut next,
                max_priority,
            );
        }
        return next;
    }

    if inst.opcode == "transpose" {
        if let Some(src_sh) = inst.operand_ids.first().and_then(|id| shardings.get(id)) {
            for (out_idx, &src_dim) in inst.dimensions.iter().enumerate() {
                usize::try_from(src_dim)
                    .ok()
                    .and_then(|i| src_sh.dim_shardings.get(i))
                    .into_iter()
                    .for_each(|src_ds| {
                        merge_open_dim_sharding(&mut next, out_idx, src_ds, max_priority);
                    });
            }
        }
        return next;
    }

    if is_rank_preserving_elementwise(&inst.opcode) {
        for op_sh in inst.operand_ids.iter().filter_map(|id| shardings.get(id)) {
            for (dim_idx, src_ds) in op_sh.dim_shardings.iter().enumerate() {
                merge_open_dim_sharding(&mut next, dim_idx, src_ds, max_priority);
            }
            for unred in &op_sh.unreduced_axes {
                if !next.unreduced_axes.iter().any(|u| u.name == unred.name) {
                    next.unreduced_axes.push(unred.clone());
                }
            }
        }
    }

    next
}

/// Propagates batch, non-contracting, and contracting `unreduced_axes` shardings across a `dot` instruction.
///
/// # Arguments
///
/// * `inst` - `dot` instruction.
/// * `lhs_sh` - Optional LHS operand sharding.
/// * `rhs_sh` - Optional RHS operand sharding.
/// * `next` - Mutable output sharding being updated.
/// * `max_priority` - Maximum active priority tier.
fn propagate_forward_dot(
    inst: &HloInstructionProto,
    lhs_sh: Option<&ShardyTensorShardingProto>,
    rhs_sh: Option<&ShardyTensorShardingProto>,
    next: &mut ShardyTensorShardingProto,
    max_priority: i64,
) {
    let (lhs_contract, rhs_contract, lhs_batch, rhs_batch) =
        inst.dot_dimension_numbers.as_ref().map_or_else(
            || (vec![1_i64], vec![0_i64], Vec::new(), Vec::new()),
            |d| {
                (
                    d.lhs_contracting_dimensions.clone(),
                    d.rhs_contracting_dimensions.clone(),
                    d.lhs_batch_dimensions.clone(),
                    d.rhs_batch_dimensions.clone(),
                )
            },
        );
    let lhs_rank = lhs_sh.map_or(0, |s| s.dim_shardings.len());
    let rhs_rank = rhs_sh.map_or(0, |s| s.dim_shardings.len());
    let mut out_dim_idx = 0_usize;

    for &b_dim in &lhs_batch {
        lhs_sh
            .and_then(|s| {
                usize::try_from(b_dim)
                    .ok()
                    .and_then(|i| s.dim_shardings.get(i))
            })
            .into_iter()
            .for_each(|src_ds| {
                merge_open_dim_sharding(next, out_dim_idx, src_ds, max_priority);
            });
        out_dim_idx = out_dim_idx.saturating_add(1);
    }
    for d in 0..lhs_rank {
        let d_i64 = i64::try_from(d).unwrap_or(-1);
        if !lhs_batch.contains(&d_i64) && !lhs_contract.contains(&d_i64) {
            lhs_sh
                .and_then(|s| s.dim_shardings.get(d))
                .into_iter()
                .for_each(|src_ds| {
                    merge_open_dim_sharding(next, out_dim_idx, src_ds, max_priority);
                });
            out_dim_idx = out_dim_idx.saturating_add(1);
        }
    }
    for d in 0..rhs_rank {
        let d_i64 = i64::try_from(d).unwrap_or(-1);
        if !rhs_batch.contains(&d_i64) && !rhs_contract.contains(&d_i64) {
            rhs_sh
                .and_then(|s| s.dim_shardings.get(d))
                .into_iter()
                .for_each(|src_ds| {
                    merge_open_dim_sharding(next, out_dim_idx, src_ds, max_priority);
                });
            out_dim_idx = out_dim_idx.saturating_add(1);
        }
    }
    for (&lc, &rc) in lhs_contract.iter().zip(rhs_contract.iter()) {
        let lhs_axes = lhs_sh
            .and_then(|s| {
                usize::try_from(lc)
                    .ok()
                    .and_then(|i| s.dim_shardings.get(i))
            })
            .map_or(&[][..], |ds| ds.axes.as_slice());
        let rhs_axes = rhs_sh
            .and_then(|s| {
                usize::try_from(rc)
                    .ok()
                    .and_then(|i| s.dim_shardings.get(i))
            })
            .map_or(&[][..], |ds| ds.axes.as_slice());
        for ax in lhs_axes {
            let shared_contract = rhs_axes.iter().any(|r| r.name == ax.name);
            let not_recorded = !next.unreduced_axes.iter().any(|u| u.name == ax.name);
            if shared_contract && not_recorded {
                next.unreduced_axes.push(ax.clone());
            }
        }
    }
}

/// Propagates shardings backward from `consumer` onto its `op_pos`-th operand.
///
/// # Arguments
///
/// * `consumer` - Consumer instruction carrying `consumer_sh`.
/// * `op_pos` - Operand index in `consumer.operand_ids`.
/// * `consumer_sh` - Current sharding of `consumer`.
/// * `cur_op_sh` - Current sharding of the operand.
/// * `max_priority` - Maximum numeric priority tier active in this phase.
fn propagate_backward_to_operand(
    consumer: &HloInstructionProto,
    op_pos: usize,
    consumer_sh: &ShardyTensorShardingProto,
    cur_op_sh: &ShardyTensorShardingProto,
    max_priority: i64,
) -> ShardyTensorShardingProto {
    let mut next = cur_op_sh.clone();

    if consumer.opcode == "transpose" && op_pos == 0 {
        for (out_idx, &src_dim) in consumer.dimensions.iter().enumerate() {
            if let (Ok(src_idx), Some(cons_ds)) = (
                usize::try_from(src_dim),
                consumer_sh.dim_shardings.get(out_idx),
            ) {
                merge_open_dim_sharding(&mut next, src_idx, cons_ds, max_priority);
            }
        }
        return next;
    }

    if is_rank_preserving_elementwise(&consumer.opcode) {
        for (dim_idx, cons_ds) in consumer_sh.dim_shardings.iter().enumerate() {
            merge_open_dim_sharding(&mut next, dim_idx, cons_ds, max_priority);
        }
    }

    next
}

/// Merges `candidate` axes into `target.dim_shardings[dim_idx]` if the target dimension is open,
/// `candidate.priority <= max_priority`, and none of the candidate axes overlap other dimensions.
///
/// # Arguments
///
/// * `target` - Mutable tensor sharding being updated.
/// * `dim_idx` - Target dimension index.
/// * `candidate` - Source dimension sharding candidate.
/// * `max_priority` - Maximum active priority tier.
fn merge_open_dim_sharding(
    target: &mut ShardyTensorShardingProto,
    dim_idx: usize,
    candidate: &ShardyDimensionShardingProto,
    max_priority: i64,
) {
    if candidate.axes.is_empty() || candidate.priority > max_priority {
        return;
    }
    let compatible_axes: Vec<ShardyAxisRefProto> = candidate
        .axes
        .iter()
        .filter(|ax_ref| {
            !target.dim_shardings.iter().enumerate().any(|(idx, ds)| {
                idx != dim_idx
                    && ds
                        .axes
                        .iter()
                        .any(|existing| axes_overlap(existing, ax_ref))
            })
        })
        .cloned()
        .collect();
    if compatible_axes.is_empty() {
        return;
    }
    target
        .dim_shardings
        .get_mut(dim_idx)
        .into_iter()
        .for_each(|dst_ds| {
            if dst_ds.is_closed {
                return;
            }
            if dst_ds.axes.is_empty() {
                dst_ds.axes.clone_from(&compatible_axes);
                dst_ds.priority = candidate.priority;
            } else {
                for ax in &compatible_axes {
                    if !dst_ds
                        .axes
                        .iter()
                        .any(|existing| axes_overlap(existing, ax))
                    {
                        dst_ds.axes.push(ax.clone());
                    }
                }
            }
        });
}

/// Checks whether two [`ShardyAxisRefProto`] references overlap on the same mesh axis or sub-axis slice.
///
/// # Arguments
///
/// * `lhs` - First axis or sub-axis reference.
/// * `rhs` - Second axis or sub-axis reference.
fn axes_overlap(lhs: &ShardyAxisRefProto, rhs: &ShardyAxisRefProto) -> bool {
    if lhs.name != rhs.name {
        return false;
    }
    match (&lhs.sub_axis_info, &rhs.sub_axis_info) {
        (Some(a), Some(b)) => {
            let a_end = a.pre_size.saturating_mul(a.size);
            let b_end = b.pre_size.saturating_mul(b.size);
            !(a_end <= b.pre_size || b_end <= a.pre_size)
        }
        _ => true,
    }
}

/// Returns `true` if `opcode` preserves operand rank and elementwise axis correspondence.
///
/// # Arguments
///
/// * `opcode` - HLO opcode mnemonic string.
fn is_rank_preserving_elementwise(opcode: &str) -> bool {
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
            | "copy"
            | "convert"
            | "bitcast-convert"
            | "clamp"
            | "select"
    )
}

/// Computes the sharding factor along each tensor dimension of `sharding`.
///
/// # Arguments
///
/// * `sharding` - Shardy tensor sharding descriptor.
/// * `axis_sizes` - Map from mesh axis name to axis size.
fn compute_dim_factors(
    sharding: &ShardyTensorShardingProto,
    axis_sizes: &BTreeMap<String, i64>,
) -> Vec<i64> {
    sharding
        .dim_shardings
        .iter()
        .map(|ds| {
            ds.axes.iter().fold(1_i64, |acc, ax| {
                let factor = ax.sub_axis_info.as_ref().map_or_else(
                    || axis_sizes.get(&ax.name).copied().unwrap_or(1),
                    |sub| sub.size.max(1),
                );
                acc.saturating_mul(factor.max(1))
            })
        })
        .collect()
}

/// Converts a [`ShardyTensorShardingProto`] into a wire-compatible [`OpShardingProto`].
///
/// # Arguments
///
/// * `sharding` - Shardy tensor sharding descriptor.
/// * `axis_sizes` - Map from mesh axis name to axis size.
#[must_use]
pub fn sdy_to_op_sharding_proto(
    sharding: &ShardyTensorShardingProto,
    axis_sizes: &BTreeMap<String, i64>,
) -> OpShardingProto {
    let factors = compute_dim_factors(sharding, axis_sizes);
    if factors.iter().all(|&f| f <= 1) {
        return OpShardingProto::default();
    }
    let total_devs = factors
        .iter()
        .fold(1_i64, |acc, &f| acc.saturating_mul(f.max(1)));
    OpShardingProto {
        r#type: 3,
        tile_assignment_dimensions: factors,
        tile_assignment_devices: (0..total_devs).collect(),
        ..OpShardingProto::default()
    }
}

/// Writes `sharding` onto `inst.sharding` and `inst.frontend_attributes[SDY_SHARDING_ATTR_KEY]`.
///
/// # Arguments
///
/// * `inst` - Mutable HLO instruction to annotate.
/// * `sharding` - Resolved Shardy tensor sharding descriptor.
/// * `axis_sizes` - Map from mesh axis name to axis size.
fn write_sharding_to_instruction(
    inst: &mut HloInstructionProto,
    sharding: &ShardyTensorShardingProto,
    axis_sizes: &BTreeMap<String, i64>,
) {
    use crate::proto::ProtoSchemaMessage as _;
    inst.sharding = Some(sdy_to_op_sharding_proto(sharding, axis_sizes));
    let json_str = sharding.to_proto_json().unwrap_or_default();
    let fa = inst
        .frontend_attributes
        .get_or_insert_with(FrontendAttributes::default);
    fa.map.insert(String::from(SDY_SHARDING_ATTR_KEY), json_str);
}

/// Divides each dimension of `shape` by the corresponding sharding factor in `sharding`.
///
/// # Arguments
///
/// * `shape` - Mutable tensor [`ShapeProto`] to convert from global to local per-partition extents.
/// * `sharding` - Shardy tensor sharding descriptor.
/// * `axis_sizes` - Map from mesh axis name to axis size.
fn partition_shape_in_place(
    shape: &mut ShapeProto,
    sharding: &ShardyTensorShardingProto,
    axis_sizes: &BTreeMap<String, i64>,
) {
    let factors = compute_dim_factors(sharding, axis_sizes);
    for (dim_mut, &factor) in shape.dimensions.iter_mut().zip(factors.iter()) {
        if factor > 1 {
            *dim_mut = (*dim_mut + factor - 1) / factor;
        }
    }
}

/// Synchronizes `module.spmd_parameters_shardings` and `module.spmd_output_sharding` from the entry computation.
///
/// # Arguments
///
/// * `module` - Mutable [`HloModuleProto`] whose entry parameter and root shardings are updated.
fn sync_module_entry_shardings(module: &mut HloModuleProto) {
    let Some(entry) = find_entry_computation(module) else {
        return;
    };
    let mut params: Vec<(i64, OpShardingProto)> = entry
        .instructions
        .iter()
        .filter(|i| i.opcode == "parameter")
        .map(|i| (i.parameter_number, i.sharding.clone().unwrap_or_default()))
        .collect();
    params.sort_by_key(|&(pnum, _)| pnum);
    let param_shardings: Vec<OpShardingProto> = params.into_iter().map(|(_, sh)| sh).collect();
    let root_sharding = find_root_instruction(entry).and_then(|i| i.sharding.clone());
    module.spmd_parameters_shardings = param_shardings;
    module.spmd_output_sharding = root_sharding;
}
