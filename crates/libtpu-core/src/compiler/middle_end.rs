//! XLA-TPU Middle-End Optimizer, Shardy/GSPMD Partitioner, Layout/Fusion & Memory Space Assignment.
//!
//! Organizes the complete Phase 5 middle-end pipeline over `xla::HloModuleProto`:
//! - [`spmd`][]: [`ShardyPropagationPass`] and [`GspmdPartitionerPass`].
//! - [`legalize_and_simplify`][]: [`DotDecomposerPass`], [`ConvolutionLegalizerPass`],
//!   [`AlgebraicSimplifierPass`], [`ConstantFoldingPass`], [`BroadcastSimplifierPass`],
//!   [`ReshapeMoverPass`], [`TransposeFoldingPass`], [`TupleSimplifierPass`],
//!   [`WhileLoopInvariantCodeMotionPass`], [`HloCsePass`], [`HloDcePass`],
//!   [`AsyncCollectiveFusionPass`], and [`LatencyHidingSchedulerPass`].
//! - [`layout_and_fusion`][]: [`TpuLayoutAssignmentPass`], [`MxuTilingPass`],
//!   [`TpuProducerConsumerFusionPass`], [`MultiOutputFusionPass`], and [`HloRematerializationPass`].
//! - [`memory_space_assignment`][]: [`MemorySpaceAssignmentPass`],
//!   [`GlobalDecreasingSizeBestFitHeapSimulator`], and [`MiddleEndCostModelEvaluator`].

pub mod layout_and_fusion;
pub mod legalize_and_simplify;
pub mod memory_space_assignment;
pub mod spmd;

pub use self::layout_and_fusion::{
    HloRematerializationPass, MXU_TILE_SCHEDULE_ATTR_KEY, MultiOutputFusionPass,
    MultiOutputFusionReport, MxuContractionSchedule, MxuTilingPass, MxuTilingReport,
    ProducerConsumerFusionReport, TpuLayoutAssignmentPass, TpuLayoutAssignmentReport,
    TpuProducerConsumerFusionPass, assign_shape_tpu_layout, compute_peak_live_bytes,
    primitive_type_bit_width, shape_bytes_256_aligned,
};
pub use self::legalize_and_simplify::{
    AlgebraicSimplifierPass, AsyncCollectiveFusionPass, AsyncCollectiveReport,
    BroadcastSimplifierPass, ConstantFoldingPass, ConvolutionLegalizerPass,
    ConvolutionLegalizerReport, DotDecomposerPass, DotDecomposerReport, HloCsePass, HloDcePass,
    LatencyHidingScheduleReport, LatencyHidingSchedulerPass, ReshapeMoverPass,
    TransposeFoldingPass, TupleSimplifierPass, WhileLoopInvariantCodeMotionPass,
};
pub use self::memory_space_assignment::{
    GlobalDecreasingSizeBestFitHeapSimulator, HeapSimulationResult,
    MSA_DOUBLE_BUFFER_SLOT_ATTR_KEY, MemorySpaceAssignmentPass, MemorySpaceAssignmentReport,
    MiddleEndCostModelEvaluator, MiddleEndCostReport, TpuMsaMemorySpace,
};
pub use self::spmd::{
    GspmdPartitionReport, GspmdPartitionerPass, SDY_SHARDING_ATTR_KEY, SPMD_HALO_SHIFT_ATTR_KEY,
    SPMD_RAGGED_RESHARD_ATTR_KEY, ShardyPropagationPass, ShardyPropagationReport,
    build_replica_groups_for_axis, sdy_to_op_sharding_proto,
};

use crate::error::LibTpuResult;
use crate::proto::{
    AlgebraicSimplifierOptionsProto, EmitterLearnedCostModelOptionsProto, HloModuleProto,
    HloRematerializationSummaryProto, ShardyMeshProto,
};
use crate::types::TpuGeneration;

/// Complete artifacts and pass reports produced by [`XlaTpuMiddleEndPipeline::run`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct XlaTpuMiddleEndArtifacts {
    /// Shardy bidirectional propagation report.
    pub shardy_report: ShardyPropagationReport,
    /// GSPMD partitioning and collective synthesis report.
    pub gspmd_report: GspmdPartitionReport,
    /// `DotDecomposerPass` report.
    pub dot_report: DotDecomposerReport,
    /// `ConvolutionLegalizerPass` report.
    pub conv_report: ConvolutionLegalizerReport,
    /// Total algebraic, constant, broadcast, reshape, transpose, tuple, LICM, CSE, and DCE rewrites.
    pub total_simplifier_rewrites: usize,
    /// Async collective Start/Done conversion report.
    pub async_collective_report: AsyncCollectiveReport,
    /// Latency-hiding collective/compute overlap schedule report.
    pub latency_schedule_report: LatencyHidingScheduleReport,
    /// Physical TPU layout assignment report.
    pub layout_report: TpuLayoutAssignmentReport,
    /// Generation-specific MXU systolic tiling report.
    pub mxu_tiling_report: MxuTilingReport,
    /// Producer-consumer fusion report.
    pub producer_consumer_fusion_report: ProducerConsumerFusionReport,
    /// Multi-output sibling fusion report.
    pub multi_output_fusion_report: MultiOutputFusionReport,
    /// HLO rematerialization summary.
    pub rematerialization_summary: HloRematerializationSummaryProto,
    /// Memory Space Assignment (`HBM`/`VMEM`/`SMEM`/`CMEM` + `ping`/`pong` prefetch) report.
    pub msa_report: MemorySpaceAssignmentReport,
    /// Global decreasing-size best-fit heap simulation and buffer donation output.
    pub heap_simulation: HeapSimulationResult,
    /// Roofline and learned cost model evaluation report.
    pub cost_report: MiddleEndCostReport,
}

/// End-to-end XLA-TPU Middle-End optimization pipeline orchestrating Phases 5.1, 5.2, and 5.3.
#[derive(Debug, Clone)]
pub struct XlaTpuMiddleEndPipeline {
    /// Target Cloud TPU generation.
    pub generation: TpuGeneration,
    /// Logical Shardy device mesh.
    pub mesh: ShardyMeshProto,
    /// Number of SPMD partitions (`>= 1`).
    pub num_partitions: usize,
    /// Reserved scoped VMEM scratchpad in KiB.
    pub scoped_vmem_limit_kib: u64,
    /// Peak live memory budget in bytes triggering [`HloRematerializationPass`].
    pub rematerialization_limit_bytes: u64,
    /// Algebraic simplifier options.
    pub algsimp_options: AlgebraicSimplifierOptionsProto,
    /// Learned cost model options.
    pub cost_model_options: EmitterLearnedCostModelOptionsProto,
}

impl XlaTpuMiddleEndPipeline {
    /// Constructs a default [`XlaTpuMiddleEndPipeline`] for `generation` and `num_partitions`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation.
    /// * `num_partitions` - Number of SPMD partitions (`>= 1`).
    #[must_use]
    pub fn for_generation(generation: TpuGeneration, num_partitions: usize) -> Self {
        let sdy = ShardyPropagationPass::with_linear_mesh(num_partitions);
        Self {
            generation,
            mesh: sdy.mesh,
            num_partitions: num_partitions.max(1),
            scoped_vmem_limit_kib: 0,
            rematerialization_limit_bytes: 256 * 1024 * 1024,
            algsimp_options: AlgebraicSimplifierOptionsProto::default(),
            cost_model_options: EmitterLearnedCostModelOptionsProto::default(),
        }
    }

    /// Executes the full Phase 5.1 -> 5.2 -> 5.3 XLA-TPU middle-end pipeline on `module`.
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`HloModuleProto`] transformed in-place from global HLO to scheduled,
    ///   tiled, fused, memory-colored, buffer-assigned TPU HLO.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError`] if sharding validation, dot/conv legalization,
    /// layout assignment, or memory space assignment fails.
    pub fn run(&self, module: &mut HloModuleProto) -> LibTpuResult<XlaTpuMiddleEndArtifacts> {
        // 5.1a Shardy propagation & GSPMD partitioning
        let shardy_report = ShardyPropagationPass::new(self.mesh.clone()).run(module)?;
        let gspmd_report =
            GspmdPartitionerPass::new(self.mesh.clone(), self.num_partitions).run(module)?;

        // 5.1b Dot & Convolution legalization
        let conv_report = ConvolutionLegalizerPass::new(self.num_partitions).run(module)?;
        let dot_report = DotDecomposerPass.run(module)?;

        // 5.1c Algebraic, constant, broadcast, reshape, transpose, tuple, LICM, CSE, and DCE passes
        let mut total_simplifier_rewrites = 0_usize;
        total_simplifier_rewrites = total_simplifier_rewrites
            .saturating_add(ConstantFoldingPass.run(module))
            .saturating_add(AlgebraicSimplifierPass::new(self.algsimp_options.clone()).run(module))
            .saturating_add(BroadcastSimplifierPass.run(module))
            .saturating_add(ReshapeMoverPass.run(module))
            .saturating_add(TransposeFoldingPass.run(module))
            .saturating_add(TupleSimplifierPass.run(module))
            .saturating_add(WhileLoopInvariantCodeMotionPass.run(module))
            .saturating_add(HloCsePass.run(module))
            .saturating_add(HloDcePass.run(module));

        // 5.1d Async collective conversion & latency-hiding overlap schedule
        let async_collective_report = AsyncCollectiveFusionPass::default().run(module);
        let latency_schedule_report = LatencyHidingSchedulerPass.run(module);

        // 5.2 Layout assignment, MXU systolic tiling, operator fusion & rematerialization
        let layout_report = TpuLayoutAssignmentPass::default().run(module)?;
        let mxu_tiling_report = MxuTilingPass::new(self.generation).run(module);
        let producer_consumer_fusion_report =
            TpuProducerConsumerFusionPass::for_generation(self.generation).run(module);
        let multi_output_fusion_report = MultiOutputFusionPass::default().run(module);
        let rematerialization_summary =
            HloRematerializationPass::new(self.rematerialization_limit_bytes).run(module);

        // 5.3 MemorySpaceAssignment, GlobalDecreasingSizeBestFitHeapSimulator & LearnedCostModel
        let msa_report =
            MemorySpaceAssignmentPass::for_generation(self.generation, self.scoped_vmem_limit_kib)
                .run(module)?;
        let heap_simulation = GlobalDecreasingSizeBestFitHeapSimulator::new(4096).run(module);
        let cost_report =
            MiddleEndCostModelEvaluator::new(self.generation, self.cost_model_options.clone())
                .evaluate_module(module);

        Ok(XlaTpuMiddleEndArtifacts {
            shardy_report,
            gspmd_report,
            dot_report,
            conv_report,
            total_simplifier_rewrites,
            async_collective_report,
            latency_schedule_report,
            layout_report,
            mxu_tiling_report,
            producer_consumer_fusion_report,
            multi_output_fusion_report,
            rematerialization_summary,
            msa_report,
            heap_simulation,
            cost_report,
        })
    }
}
