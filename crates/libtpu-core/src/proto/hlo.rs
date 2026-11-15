//! XLA HLO module, instruction, computation, and buffer assignment protobuf types.
//!
//! Re-exports `prost-build` generated structs from `third_party/tensorflow/compiler/xla/service/hlo.proto`,
//! `buffer_assignment.proto`, `shaped_slice.proto`, `metrics.proto`, `hlo_profile_printer_data.proto`,
//! and `protos/libtpu_ext/hlo_passes.proto`.

pub use super::descriptor::generated_prost_schemas::libtpu::ext::hlo::{
    BufferDonationAliasEntryProto, HloRematerializationSummaryProto,
    InputOutputBufferAliasConfigProto, RematerializedInstructionEntryProto,
};
pub use super::descriptor::generated_prost_schemas::xla::buffer_assignment::{
    AssignmentAlgorithmForComputationsWithoutOrderingProto, BufferAllocationSliceProto,
    BufferAssignmentAlgorithmProto,
};
pub use super::descriptor::generated_prost_schemas::xla::{
    AlgebraicSimplifierOptions as AlgebraicSimplifierOptionsProto, BufferAllocationProto,
    BufferAssignmentProto, CompilationLogEntry as CompilationLogEntryProto,
    CompilerBuildInfo as CompilerBuildInfoProto, ComputationStats,
    CrossProgramPrefetch as CrossProgramPrefetchProto,
    ExecutionOptions as XlaExecutionOptionsProto, HbmOomDetails as HbmOomDetailsProto,
    HeapSimulatorTrace as HeapSimulatorTraceProto, HloBufferDonorProto, HloComputationProto,
    HloInputOutputAliasProto, HloInstructionProto, HloModuleConfigProto, HloModuleProto,
    HloModuleProtoWithConfig, HloPassMetadata as HloPassMetadataProto,
    HloProfilePrinterData as HloProfilePrinterDataProto, HloProto, HloScheduleProto,
    HloSnapshot as HloSnapshotProto, JobInfo as JobInfoProto,
    KeyValueMetric as KeyValueMetricProto, LogicalBufferProto, MemoryUsageReportProto,
    NullableShapedSliceProto, PassMetrics as PassMetricsProto, ShapedSliceProto,
    StackFrameIndexProto, TagMetric as TagMetricProto,
};

/// Alias for `xla.HeapSimulatorTrace.Event.Kind` generated from `hlo.proto`.
pub type HeapSimulatorEventKindProto =
    super::descriptor::generated_prost_schemas::xla::heap_simulator_trace::event::Kind;

/// Alias for `xla.HloScheduleProto.InstructionSequence` generated from `hlo.proto`.
pub type HloInstructionSequenceProto =
    super::descriptor::generated_prost_schemas::xla::hlo_schedule_proto::InstructionSequence;

/// Alias for `xla.BufferAssignmentProto.BufferAlias` generated from `hlo.proto`.
pub type BufferAssignmentAliasProto =
    super::descriptor::generated_prost_schemas::xla::buffer_assignment_proto::BufferAlias;

/// Alias for `xla.BufferAssignmentProto.PeakBuffers` generated from `hlo.proto`.
pub type BufferAssignmentPeakBuffersProto =
    super::descriptor::generated_prost_schemas::xla::buffer_assignment_proto::PeakBuffers;

/// Alias for `xla.MemoryUsageReportProto.AllocationEntry` generated from `hlo.proto`.
pub type MemoryUsageAllocationEntryProto =
    super::descriptor::generated_prost_schemas::xla::memory_usage_report_proto::AllocationEntry;

/// Alias for `xla.MemoryUsageReportProto.AllocationEntryInMemorySpace` generated from `hlo.proto`.
pub type MemoryUsageSpaceEntryProto =
    super::descriptor::generated_prost_schemas::xla::memory_usage_report_proto::AllocationEntryInMemorySpace;

/// Alias for `xla.HloInstructionProto.SliceDimensions` generated from `hlo.proto`.
pub type HloSliceDimensionsProto =
    super::descriptor::generated_prost_schemas::xla::hlo_instruction_proto::SliceDimensions;

/// Alias for `xla.HloInputOutputAliasProto.AliasEntryProto` generated from `hlo.proto`.
pub type HloAliasEntryProto =
    super::descriptor::generated_prost_schemas::xla::hlo_input_output_alias_proto::AliasEntryProto;

/// Alias for `xla.HloBufferDonorProto.BufferDonorEntryProto` generated from `hlo.proto`.
pub type HloBufferDonorEntryProto =
    super::descriptor::generated_prost_schemas::xla::hlo_buffer_donor_proto::BufferDonorEntryProto;

/// Alias for `xla.StackFrameIndexProto.FileLocation` generated from `hlo.proto`.
pub type StackFrameFileLocationProto =
    super::descriptor::generated_prost_schemas::xla::stack_frame_index_proto::FileLocation;

/// Alias for `xla.StackFrameIndexProto.StackFrame` generated from `hlo.proto`.
pub type StackFrameEntryProto =
    super::descriptor::generated_prost_schemas::xla::stack_frame_index_proto::StackFrame;

/// Alias for `xla.LogicalBufferProto.Location` generated from `hlo.proto`.
pub type LogicalBufferLocationProto =
    super::descriptor::generated_prost_schemas::xla::logical_buffer_proto::Location;

/// Alias for `xla.HeapSimulatorTrace.Event` generated from `hlo.proto`.
pub type HeapSimulatorTraceEventProto =
    super::descriptor::generated_prost_schemas::xla::heap_simulator_trace::Event;

/// Alias for `xla.BufferAllocationProto.Assigned` generated from `hlo.proto`.
pub type BufferAllocationAssignedProto =
    super::descriptor::generated_prost_schemas::xla::buffer_allocation_proto::Assigned;

/// Alias for `xla.buffer_assignment.BufferIsolationConfig` generated from `buffer_assignment.proto`.
pub type BufferIsolationConfigProto =
    super::descriptor::generated_prost_schemas::xla::buffer_assignment::BufferIsolationConfig;
