//! XLA High-Level Optimizer (HLO) IR and buffer assignment protobuf messages.
//!
//! Synchronized with `xla/service/hlo.proto`, `xla/xla.proto`, and `xla/service/buffer_assignment.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use crate::proto::xla_data::{
    CholeskyOptions, ConvolutionDimensionNumbers, DeviceAssignmentProto, DotDimensionNumbers,
    ExecutionProfile, FrontendAttributes, GatherDimensionNumbers, LiteralProto, OpMetadata,
    OpShardingProto, PaddingConfig, ParameterReplicationProto, PrecisionConfig, ProgramShapeProto,
    ReplicaGroup, ScatterDimensionNumbers, ShapeProto, SourceTarget, StatisticsViz,
    TriangularSolveOptions, Window,
};
use prost::Message;

/// Wire-compatible `xla.HloInstructionProto.SliceDimensions` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloSliceDimensionsProto {
    /// Start index (inclusive, field `1`).
    #[prost(int64, tag = "1")]
    pub start: i64,
    /// Limit index (exclusive, field `2`).
    #[prost(int64, tag = "2")]
    pub limit: i64,
    /// Slice stride (field `3`).
    #[prost(int64, tag = "3")]
    pub stride: i64,
}

impl_proto_schema_message!(
    HloSliceDimensionsProto,
    "xla.HloInstructionProto.SliceDimensions"
);

/// Wire-compatible `xla.HloInputOutputAliasProto.AliasEntryProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloAliasEntryProto {
    /// Output shape index of the root instruction (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub output_shape_index: Vec<i64>,
    /// Entry computation parameter number (field `2`).
    #[prost(int64, tag = "2")]
    pub parameter_number: i64,
    /// Parameter shape index (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub parameter_shape_index: Vec<i64>,
    /// Alias kind (`0` = Undefined, `1` = `MayAlias`, `2` = `MustAlias`, field `4`).
    #[prost(int32, tag = "4")]
    pub kind: i32,
}

impl_proto_schema_message!(
    HloAliasEntryProto,
    "xla.HloInputOutputAliasProto.AliasEntryProto"
);

/// Wire-compatible `xla.HloInputOutputAliasProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloInputOutputAliasProto {
    /// Input-output buffer aliasing entries (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub entries: Vec<HloAliasEntryProto>,
}

impl_proto_schema_message!(HloInputOutputAliasProto, "xla.HloInputOutputAliasProto");

/// Wire-compatible `xla.HloBufferDonorProto.BufferDonorEntryProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloBufferDonorEntryProto {
    /// Entry computation parameter number (field `1`).
    #[prost(int64, tag = "1")]
    pub parameter_number: i64,
    /// Parameter shape index (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub parameter_shape_index: Vec<i64>,
}

impl_proto_schema_message!(
    HloBufferDonorEntryProto,
    "xla.HloBufferDonorProto.BufferDonorEntryProto"
);

/// Wire-compatible `xla.HloBufferDonorProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloBufferDonorProto {
    /// Donatable input buffer entries (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub entries: Vec<HloBufferDonorEntryProto>,
}

impl_proto_schema_message!(HloBufferDonorProto, "xla.HloBufferDonorProto");

/// Wire-compatible `xla.CrossProgramPrefetch` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CrossProgramPrefetchProto {
    /// Parameter number to prefetch across programs (field `1`).
    #[prost(int64, tag = "1")]
    pub parameter: i64,
    /// Tuple shape index within the parameter (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub index: Vec<i64>,
    /// Byte offset in alternate memory (`VMEM`/`CMEM`, field `3`).
    #[prost(int64, tag = "3")]
    pub offset: i64,
}

impl_proto_schema_message!(CrossProgramPrefetchProto, "xla.CrossProgramPrefetch");

/// Wire-compatible `xla.DynamicParameterBindingProto.Binding` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DynamicParameterBindingEntryProto {
    /// Dynamic size parameter number (field `1`).
    #[prost(int64, tag = "1")]
    pub dynamic_param_num: i64,
    /// Shape index into the dynamic size parameter (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub dynamic_param_index: Vec<i64>,
    /// Target bounded parameter number (field `3`).
    #[prost(int64, tag = "3")]
    pub target_param_num: i64,
    /// Shape index into the target parameter (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub target_param_index: Vec<i64>,
    /// Dynamic dimension index on the target parameter (field `5`).
    #[prost(int64, tag = "5")]
    pub target_param_dim_num: i64,
}

impl_proto_schema_message!(
    DynamicParameterBindingEntryProto,
    "xla.DynamicParameterBindingProto.Binding"
);

/// Wire-compatible `xla.DynamicParameterBindingProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DynamicParameterBindingProto {
    /// Dynamic parameter dimension bindings (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub entries: Vec<DynamicParameterBindingEntryProto>,
}

impl_proto_schema_message!(
    DynamicParameterBindingProto,
    "xla.DynamicParameterBindingProto"
);

/// Wire-compatible `xla.HloInstructionProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloInstructionProto {
    /// Instruction name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Instruction opcode string (e.g., `"add"`, `"multiply"`, `"dot"`, `"parameter"`, field `2`).
    #[prost(string, tag = "2")]
    pub opcode: String,
    /// Output shape of this instruction (field `3`).
    #[prost(message, optional, tag = "3")]
    pub shape: Option<ShapeProto>,
    /// Symbolization and profiling metadata (field `7`).
    #[prost(message, optional, tag = "7")]
    pub metadata: Option<OpMetadata>,
    /// Constant literal value (`opcode == "constant"`, field `8`).
    #[prost(message, optional, tag = "8")]
    pub literal: Option<LiteralProto>,
    /// Parameter number for `opcode == "parameter"` (field `9`).
    #[prost(int64, tag = "9")]
    pub parameter_number: i64,
    /// Fusion kind (`opcode == "fusion"`, field `11`).
    #[prost(string, tag = "11")]
    pub fusion_kind: String,
    /// Tuple element index (`opcode == "get-tuple-element"`, field `13`).
    #[prost(int64, tag = "13")]
    pub tuple_index: i64,
    /// Dimensions for `broadcast`, `transpose`, `reduce`, `reverse`, `concatenate`, `iota` (field `14`).
    #[prost(int64, repeated, packed = "true", tag = "14")]
    pub dimensions: Vec<i64>,
    /// Spatial window configuration for `convolution` and `reduce-window` (field `15`).
    #[prost(message, optional, tag = "15")]
    pub window: Option<Window>,
    /// Convolution dimension numbers (field `16`).
    #[prost(message, optional, tag = "16")]
    pub convolution_dimension_numbers: Option<ConvolutionDimensionNumbers>,
    /// Per-dimension `[start, limit, stride)` slice bounds (field `17`).
    #[prost(message, repeated, tag = "17")]
    pub slice_dimensions: Vec<HloSliceDimensionsProto>,
    /// Exponent bit width for `reduce-precision` (field `18`).
    #[prost(int32, tag = "18")]
    pub exponent_bits: i32,
    /// Mantissa bit width for `reduce-precision` (field `19`).
    #[prost(int32, tag = "19")]
    pub mantissa_bits: i32,
    /// Slice sizes for `dynamic-slice` (field `20`).
    #[prost(int64, repeated, packed = "true", tag = "20")]
    pub dynamic_slice_sizes: Vec<i64>,
    /// Edge and interior padding configuration for `pad` (field `21`).
    #[prost(message, optional, tag = "21")]
    pub padding_config: Option<PaddingConfig>,
    /// Outfeed configuration bytes (field `22`).
    #[prost(bytes = "vec", tag = "22")]
    pub outfeed_config: Vec<u8>,
    /// Random distribution enum (`xla.RandomDistribution`, field `23`).
    #[prost(int32, tag = "23")]
    pub distribution: i32,
    /// Numerical stability epsilon for batch norm (field `24`).
    #[prost(float, tag = "24")]
    pub epsilon: f32,
    /// Feature dimension index for batch norm (field `25`).
    #[prost(int64, tag = "25")]
    pub feature_index: i64,
    /// Collective or Send/Recv channel ID (field `26`).
    #[prost(int64, tag = "26")]
    pub channel_id: i64,
    /// Infeed configuration bytes (field `27`).
    #[prost(bytes = "vec", tag = "27")]
    pub infeed_config: Vec<u8>,
    /// Custom call target name (field `28`).
    #[prost(string, tag = "28")]
    pub custom_call_target: String,
    /// Outfeed tensor shape (field `29`).
    #[prost(message, optional, tag = "29")]
    pub outfeed_shape: Option<ShapeProto>,
    /// Contracting and batch dimensions for `dot` (field `30`).
    #[prost(message, optional, tag = "30")]
    pub dot_dimension_numbers: Option<DotDimensionNumbers>,
    /// Fast Fourier Transform type (`xla.FftType`, field `31`).
    #[prost(int32, tag = "31")]
    pub fft_type: i32,
    /// Fast Fourier Transform lengths (field `32`).
    #[prost(int64, repeated, packed = "true", tag = "32")]
    pub fft_length: Vec<i64>,
    /// Gather dimension numbers (field `33`).
    #[prost(message, optional, tag = "33")]
    pub gather_dimension_numbers: Option<GatherDimensionNumbers>,
    /// Gather slice sizes (field `34`).
    #[prost(int64, repeated, packed = "true", tag = "34")]
    pub gather_slice_sizes: Vec<i64>,
    /// Unique instruction ID within the module (field `35`).
    #[prost(int64, tag = "35")]
    pub id: i64,
    /// Operand instruction IDs (field `36`).
    #[prost(int64, repeated, packed = "true", tag = "36")]
    pub operand_ids: Vec<i64>,
    /// Control predecessor instruction IDs (field `37`).
    #[prost(int64, repeated, packed = "true", tag = "37")]
    pub control_predecessor_ids: Vec<i64>,
    /// Called sub-computation IDs (`reduce`, `while`, `conditional`, `call`, `scatter`, `sort`, field `38`).
    #[prost(int64, repeated, packed = "true", tag = "38")]
    pub called_computation_ids: Vec<i64>,
    /// SPMD sharding annotation (field `40`).
    #[prost(message, optional, tag = "40")]
    pub sharding: Option<OpShardingProto>,
    /// Backend-specific configuration payload (field `43`).
    #[prost(bytes = "vec", tag = "43")]
    pub backend_config: Vec<u8>,
    /// Triangular solve options (`opcode == "triangular-solve"`, field `45`).
    #[prost(message, optional, tag = "45")]
    pub triangular_solve_options: Option<TriangularSolveOptions>,
    /// Cholesky decomposition options (`opcode == "cholesky"`, field `46`).
    #[prost(message, optional, tag = "46")]
    pub cholesky_options: Option<CholeskyOptions>,
    /// Scatter dimension numbers (field `48`).
    #[prost(message, optional, tag = "48")]
    pub scatter_dimension_numbers: Option<ScatterDimensionNumbers>,
    /// Collective replica groups (field `49`).
    #[prost(message, repeated, tag = "49")]
    pub replica_groups: Vec<ReplicaGroup>,
    /// Convolution feature group count (field `50`).
    #[prost(int64, tag = "50")]
    pub feature_group_count: i64,
    /// Operand precision and accumulation algorithm (field `51`).
    #[prost(message, optional, tag = "51")]
    pub precision_config: Option<PrecisionConfig>,
    /// Collective-permute source/target pairs (field `52`).
    #[prost(message, repeated, tag = "52")]
    pub source_target_pairs: Vec<SourceTarget>,
    /// Convolution batch group count (field `58`).
    #[prost(int64, tag = "58")]
    pub batch_group_count: i64,
    /// Whether `sort` is stable (field `60`).
    #[prost(bool, tag = "60")]
    pub is_stable: bool,
    /// Parameter replication descriptor across data-parallel replicas (field `61`).
    #[prost(message, optional, tag = "61")]
    pub parameter_replication: Option<ParameterReplicationProto>,
    /// Comparison direction (`"EQ"`, `"NE"`, `"LT"`, `"LE"`, `"GT"`, `"GE"`, field `63`).
    #[prost(string, tag = "63")]
    pub comparison_direction: String,
    /// Frontend attributes map (field `68`).
    #[prost(message, optional, tag = "68")]
    pub frontend_attributes: Option<FrontendAttributes>,
    /// Whether replica group IDs represent global device IDs (field `71`).
    #[prost(bool, tag = "71")]
    pub use_global_device_ids: bool,
    /// Visualization statistics for HLO graph rendering (field `82`).
    #[prost(message, optional, tag = "82")]
    pub statistics_viz: Option<StatisticsViz>,
}

impl_proto_schema_message!(HloInstructionProto, "xla.HloInstructionProto");

/// Wire-compatible `xla.HloComputationProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloComputationProto {
    /// Computation name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Instructions in topological order (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub instructions: Vec<HloInstructionProto>,
    /// Optional program shape of this computation (field `4`).
    #[prost(message, optional, tag = "4")]
    pub program_shape: Option<ProgramShapeProto>,
    /// Computation ID (field `5`).
    #[prost(int64, tag = "5")]
    pub id: i64,
    /// Root instruction ID (field `6`).
    #[prost(int64, tag = "6")]
    pub root_id: i64,
    /// Whether this computation is a fused sub-computation (field `7`).
    #[prost(bool, tag = "7")]
    pub is_fusion_computation: bool,
    /// Execution thread name (field `8`).
    #[prost(string, tag = "8")]
    pub execution_thread: String,
}

impl_proto_schema_message!(HloComputationProto, "xla.HloComputationProto");

/// Wire-compatible `xla.HloModuleProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloModuleProto {
    /// Module name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Entry computation name (field `2`).
    #[prost(string, tag = "2")]
    pub entry_computation_name: String,
    /// Computations belonging to this module (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub computations: Vec<HloComputationProto>,
    /// Optional host program shape (field `4`).
    #[prost(message, optional, tag = "4")]
    pub host_program_shape: Option<ProgramShapeProto>,
    /// Module ID (field `5`).
    #[prost(int64, tag = "5")]
    pub id: i64,
    /// Entry computation ID (field `6`).
    #[prost(int64, tag = "6")]
    pub entry_computation_id: i64,
    /// Input-output buffer aliasing table (field `8`).
    #[prost(message, optional, tag = "8")]
    pub input_output_alias: Option<HloInputOutputAliasProto>,
    /// Dynamic parameter bindings (field `9`).
    #[prost(message, optional, tag = "9")]
    pub dynamic_parameter_binding: Option<DynamicParameterBindingProto>,
    /// Cross-program alternate-memory prefetches (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub cross_program_prefetches: Vec<CrossProgramPrefetchProto>,
    /// Whether the module contains dynamic shapes (field `11`).
    #[prost(bool, tag = "11")]
    pub is_dynamic: bool,
    /// SPMD output sharding (field `12`).
    #[prost(message, optional, tag = "12")]
    pub spmd_output_sharding: Option<OpShardingProto>,
    /// Per-parameter SPMD shardings (field `14`).
    #[prost(message, repeated, tag = "14")]
    pub spmd_parameters_shardings: Vec<OpShardingProto>,
    /// Static device assignment (field `15`).
    #[prost(message, optional, tag = "15")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Whether `AutoSharding` is enabled (field `16`).
    #[prost(bool, tag = "16")]
    pub use_auto_spmd_partitioning: bool,
    /// Input buffer donor table (field `18`).
    #[prost(message, optional, tag = "18")]
    pub buffer_donor: Option<HloBufferDonorProto>,
    /// Module-level frontend attributes (field `19`).
    #[prost(message, optional, tag = "19")]
    pub frontend_attributes: Option<FrontendAttributes>,
    /// Target device type string (field `21`).
    #[prost(string, tag = "21")]
    pub device_type: String,
}

impl_proto_schema_message!(HloModuleProto, "xla.HloModuleProto");

/// Wire-compatible `xla.HloModuleConfigProto` (`xla/xla.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleConfigProto {
    /// Entry computation parameter and result layout (field `1`).
    #[prost(message, optional, tag = "1")]
    pub entry_computation_layout: Option<ProgramShapeProto>,
    /// Deterministic PRNG compilation seed (field `2`).
    #[prost(uint64, tag = "2")]
    pub seed: u64,
    /// Launch identifier (field `3`).
    #[prost(int32, tag = "3")]
    pub launch_id: i32,
    /// Number of data-parallel replicas (field `4`).
    #[prost(int64, tag = "4")]
    pub replica_count: i64,
    /// Number of SPMD partitions (field `5`).
    #[prost(int64, tag = "5")]
    pub num_partitions: i64,
    /// Per-parameter collective broadcast flags (field `6`).
    #[prost(bool, repeated, packed = "true", tag = "6")]
    pub param_requires_broadcast_via_collectives: Vec<bool>,
    /// Whether SPMD partitioning is enabled (field `7`).
    #[prost(bool, tag = "7")]
    pub use_spmd_partitioning: bool,
    /// Whether `AutoSPMD` partitioning is enabled (field `8`).
    #[prost(bool, tag = "8")]
    pub use_auto_spmd_partitioning: bool,
    /// `AutoSPMD` mesh shape dimensions (field `9`).
    #[prost(int64, repeated, packed = "true", tag = "9")]
    pub auto_spmd_partitioning_mesh_shape: Vec<i64>,
    /// `AutoSPMD` mesh device IDs (field `10`).
    #[prost(int64, repeated, packed = "true", tag = "10")]
    pub auto_spmd_partitioning_mesh_ids: Vec<i64>,
    /// Whether identical HLO subcomputations are deduplicated (field `11`).
    #[prost(bool, tag = "11")]
    pub deduplicate_hlo: bool,
    /// Intra-op parallelism thread count (field `12`).
    #[prost(int64, tag = "12")]
    pub intra_op_parallelism_threads: i64,
    /// Target device type identifier (field `13`).
    #[prost(string, tag = "13")]
    pub device_type: String,
    /// Static device assignment (field `15`).
    #[prost(message, optional, tag = "15")]
    pub static_device_assignment: Option<DeviceAssignmentProto>,
    /// Alias passthrough parameters flag (field `17`).
    #[prost(bool, tag = "17")]
    pub alias_passthrough_params: bool,
    /// Matrix unit operand precision (`xla.PrecisionConfig.Precision`, field `29`).
    #[prost(int32, tag = "29")]
    pub matrix_unit_operand_precision: i32,
    /// Available device memory size in bytes (field `32`).
    #[prost(int64, tag = "32")]
    pub device_memory_size: i64,
    /// Whether the Shardy (`sdy`) partitioner is enabled (field `34`).
    #[prost(bool, tag = "34")]
    pub use_shardy_partitioner: bool,
}

impl_proto_schema_message!(HloModuleConfigProto, "xla.HloModuleConfigProto");

/// Wire-compatible `xla.HloModuleProtoWithConfig` (`xla/xla.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloModuleProtoWithConfig {
    /// Embedded `xla.HloModuleProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub hlo_module: Option<HloModuleProto>,
    /// Embedded `xla.HloModuleConfigProto` config (field `2`).
    #[prost(message, optional, tag = "2")]
    pub config: Option<HloModuleConfigProto>,
}

impl_proto_schema_message!(HloModuleProtoWithConfig, "xla.HloModuleProtoWithConfig");

/// Wire-compatible `xla.buffer_assignment.BufferAllocationSliceProto` (`xla/service/buffer_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAllocationSliceProto {
    /// Byte offset within the parent `BufferAllocationProto` (field `1`).
    #[prost(int64, tag = "1")]
    pub offset: i64,
    /// Size of the slice in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
    /// Parent allocation index (field `3`).
    #[prost(int64, tag = "3")]
    pub buffer_allocation_index: i64,
    /// Primitive element type (`xla.PrimitiveType`, field `4`).
    #[prost(int32, tag = "4")]
    pub element_type: i32,
}

impl_proto_schema_message!(
    BufferAllocationSliceProto,
    "xla.buffer_assignment.BufferAllocationSliceProto"
);

/// Wire-compatible `xla.BufferAllocationProto` (`xla/service/hlo.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAllocationProto {
    /// Allocation index (field `1`).
    #[prost(int64, tag = "1")]
    pub index: i64,
    /// Total size in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
    /// Whether this allocation is a thread-local buffer (field `3`).
    #[prost(bool, tag = "3")]
    pub is_thread_local: bool,
    /// Whether this allocation represents an entry computation parameter (field `5`).
    #[prost(bool, tag = "5")]
    pub is_entry_computation_parameter: bool,
    /// Parameter number if `is_entry_computation_parameter` is true (field `6`).
    #[prost(int64, tag = "6")]
    pub parameter_number: i64,
    /// Whether this allocation is live out of the module (output, field `7`).
    #[prost(bool, tag = "7")]
    pub maybe_live_out: bool,
    /// Memory space color (`0` = HBM, `1` = VMEM, `2` = SMEM, `3` = CMEM, `4` = `HostPinned`, field `8`).
    #[prost(int64, tag = "8")]
    pub color: i64,
    /// Assigned slices within this allocation (field `9`).
    #[prost(message, repeated, tag = "9")]
    pub assigned: Vec<BufferAllocationSliceProto>,
    /// Whether this allocation is a tuple buffer (field `11`).
    #[prost(bool, tag = "11")]
    pub is_tuple: bool,
    /// Whether this allocation is a constant buffer (field `12`).
    #[prost(bool, tag = "12")]
    pub is_constant: bool,
}

impl_proto_schema_message!(BufferAllocationProto, "xla.BufferAllocationProto");

/// Wire-compatible `xla.BufferAssignmentProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAssignmentProto {
    /// Physical buffer allocations across HBM, VMEM, SMEM, and CMEM (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub buffer_allocations: Vec<BufferAllocationProto>,
}

impl_proto_schema_message!(BufferAssignmentProto, "xla.BufferAssignmentProto");

/// Wire-compatible `xla.HloSnapshot` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloSnapshotProto {
    /// Captured `HloModuleProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub hlo: Option<HloModuleProto>,
    /// Captured input argument literals (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub arguments: Vec<LiteralProto>,
    /// Captured execution result literal (field `3`).
    #[prost(message, optional, tag = "3")]
    pub result: Option<LiteralProto>,
    /// Captured execution profile (field `4`).
    #[prost(message, optional, tag = "4")]
    pub execution_profile: Option<ExecutionProfile>,
}

impl_proto_schema_message!(HloSnapshotProto, "xla.HloSnapshot");

/// Wire-compatible `xla.buffer_assignment.BufferIsolationConfig` (`xla/service/buffer_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferIsolationConfigProto {
    /// Base byte offset added to all buffers (field `1`).
    #[prost(int64, tag = "1")]
    pub base_offset_bytes: i64,
    /// Number of buffers to isolate above the heap-simulator pack (field `2`).
    #[prost(int64, tag = "2")]
    pub isolation_fuel: i64,
    /// Extra padding in bytes between isolated buffers (field `3`).
    #[prost(int64, tag = "3")]
    pub isolation_padding_bytes: i64,
    /// Deterministic pseudo-random salt for buffer isolation order (field `4`).
    #[prost(uint64, tag = "4")]
    pub isolation_order_salt: u64,
    /// Memory space colors to isolate (field `5`).
    #[prost(int32, repeated, packed = "true", tag = "5")]
    pub isolation_colors: Vec<i32>,
}

impl_proto_schema_message!(
    BufferIsolationConfigProto,
    "xla.buffer_assignment.BufferIsolationConfig"
);

/// Wire-compatible `xla.buffer_assignment.BufferAssignmentAlgorithmProto` (`xla/service/buffer_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAssignmentAlgorithmProto {
    /// Optional fallback algorithm enum (`BufferAssignmentAlgorithmProto.Value`, field `1`).
    #[prost(int32, optional, tag = "1")]
    pub fallback_algorithm: Option<i32>,
}

impl_proto_schema_message!(
    BufferAssignmentAlgorithmProto,
    "xla.buffer_assignment.BufferAssignmentAlgorithmProto"
);

/// Wire-compatible `xla.buffer_assignment.AssignmentAlgorithmForComputationsWithoutOrderingProto` (`buffer_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AssignmentAlgorithmForComputationsWithoutOrderingProto {}

impl_proto_schema_message!(
    AssignmentAlgorithmForComputationsWithoutOrderingProto,
    "xla.buffer_assignment.AssignmentAlgorithmForComputationsWithoutOrderingProto"
);

/// Wire-compatible `xla.ShapedSliceProto` (`xla/service/shaped_slice.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShapedSliceProto {
    /// Buffer allocation slice (field `1`).
    #[prost(message, optional, tag = "1")]
    pub slice: Option<BufferAllocationSliceProto>,
    /// Logical tensor shape of the slice (field `2`).
    #[prost(message, optional, tag = "2")]
    pub shape: Option<ShapeProto>,
}

impl_proto_schema_message!(ShapedSliceProto, "xla.ShapedSliceProto");

/// Wire-compatible `xla.NullableShapedSliceProto` (`xla/service/shaped_slice.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct NullableShapedSliceProto {
    /// Optional shaped slice (`None` represents a null thunk argument, field `1`).
    #[prost(message, optional, tag = "1")]
    pub shaped_slice: Option<ShapedSliceProto>,
}

impl_proto_schema_message!(NullableShapedSliceProto, "xla.NullableShapedSliceProto");

/// Wire-compatible `xla.KeyValueMetric` (`xla/service/metrics.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct KeyValueMetricProto {
    /// Metric key name (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Metric integer value (field `2`).
    #[prost(int64, tag = "2")]
    pub value: i64,
}

impl_proto_schema_message!(KeyValueMetricProto, "xla.KeyValueMetric");

/// Wire-compatible `xla.PassMetrics` (`xla/service/metrics.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PassMetricsProto {
    /// Unique ID of the module on which the pass ran (field `1`).
    #[prost(uint64, tag = "1")]
    pub module_id: u64,
    /// Compiler pass name (field `2`).
    #[prost(string, tag = "2")]
    pub pass_name: String,
    /// Duration of the pass (field `3`).
    #[prost(message, optional, tag = "3")]
    pub pass_duration: Option<prost_types::Duration>,
    /// Opaque custom pass metrics (field `4`).
    #[prost(message, optional, tag = "4")]
    pub custom_metrics: Option<prost_types::Any>,
    /// Key-value pass metrics (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub kv_metrics: Vec<KeyValueMetricProto>,
}

impl_proto_schema_message!(PassMetricsProto, "xla.PassMetrics");

/// Wire-compatible `xla.JobInfo` (`xla/service/metrics.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct JobInfoProto {
    /// Job name (field `1`).
    #[prost(string, optional, tag = "1")]
    pub name: Option<String>,
    /// Cluster cell identifier (field `2`).
    #[prost(string, optional, tag = "2")]
    pub cell: Option<String>,
    /// User running the compilation job (field `3`).
    #[prost(string, optional, tag = "3")]
    pub user: Option<String>,
    /// Unique job UID (field `4`).
    #[prost(int64, optional, tag = "4")]
    pub uid: Option<i64>,
    /// Task index (field `5`).
    #[prost(int64, optional, tag = "5")]
    pub task_id: Option<i64>,
    /// Task unique ID across restarts (field `6`).
    #[prost(int64, optional, tag = "6")]
    pub task_uid: Option<i64>,
    /// Host OS process ID (field `7`).
    #[prost(int64, optional, tag = "7")]
    pub process_id: Option<i64>,
    /// Thread unique ID (field `8`).
    #[prost(int64, optional, tag = "8")]
    pub thread_id: Option<i64>,
}

impl_proto_schema_message!(JobInfoProto, "xla.JobInfo");

/// Wire-compatible `xla.TagMetric` (`xla/service/metrics.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TagMetricProto {
    /// Tag key (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Tag value (field `2`).
    #[prost(string, tag = "2")]
    pub value: String,
}

impl_proto_schema_message!(TagMetricProto, "xla.TagMetric");

/// Wire-compatible `xla.CompilerBuildInfo` (`xla/service/metrics.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilerBuildInfoProto {
    /// Changelist number of the compiler binary (field `1`).
    #[prost(int64, tag = "1")]
    pub compiler_changelist: i64,
    /// Build label of the compiler binary (field `2`).
    #[prost(string, tag = "2")]
    pub compiler_build_label: String,
    /// Baseline changelist number (field `3`).
    #[prost(int64, tag = "3")]
    pub baseline_changelist: i64,
    /// Client status enum (`0` = Unknown, `1` = Mint, `2` = Modified, field `4`).
    #[prost(int32, tag = "4")]
    pub client_status: i32,
}

impl_proto_schema_message!(CompilerBuildInfoProto, "xla.CompilerBuildInfo");

/// Wire-compatible `xla.CompilationLogEntry` (`xla/service/metrics.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilationLogEntryProto {
    /// Event timestamp (field `1`).
    #[prost(message, optional, tag = "1")]
    pub timestamp: Option<prost_types::Timestamp>,
    /// Compilation stage enum (`CompilationStage`, field `2`).
    #[prost(int32, tag = "2")]
    pub stage: i32,
    /// Stage duration (field `3`).
    #[prost(message, optional, tag = "3")]
    pub duration: Option<prost_types::Duration>,
    /// Task index (`-1` if unknown, field `4`).
    #[prost(int32, tag = "4")]
    pub task_index: i32,
    /// Per-pass metrics (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub pass_metrics: Vec<PassMetricsProto>,
    /// Module IDs compiled during this stage (field `6`).
    #[prost(uint64, repeated, packed = "true", tag = "6")]
    pub module_ids: Vec<u64>,
    /// Job metadata (field `7`).
    #[prost(message, optional, tag = "7")]
    pub job_info: Option<JobInfoProto>,
    /// HLO module name (field `8`).
    #[prost(string, tag = "8")]
    pub hlo_module_name: String,
    /// Metadata filter tags (field `9`).
    #[prost(message, repeated, tag = "9")]
    pub tag: Vec<TagMetricProto>,
    /// Compiler binary build information (field `10`).
    #[prost(message, optional, tag = "10")]
    pub compiler_build_info: Option<CompilerBuildInfoProto>,
}

impl_proto_schema_message!(CompilationLogEntryProto, "xla.CompilationLogEntry");

/// Wire-compatible `xla.CompilerPerfStats` (`xla/service/xla_compile_result.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilerPerfStatsProto {
    /// Compiler initialization duration (field `1`).
    #[prost(message, optional, tag = "1")]
    pub init_duration: Option<prost_types::Duration>,
    /// HLO verification duration (field `2`).
    #[prost(message, optional, tag = "2")]
    pub hlo_verification_duration: Option<prost_types::Duration>,
    /// Compilation prologue duration (field `3`).
    #[prost(message, optional, tag = "3")]
    pub compilation_prologue_duration: Option<prost_types::Duration>,
    /// Main compilation duration (field `4`).
    #[prost(message, optional, tag = "4")]
    pub compilation_duration: Option<prost_types::Duration>,
    /// Total end-to-end compilation duration (field `5`).
    #[prost(message, optional, tag = "5")]
    pub total_duration: Option<prost_types::Duration>,
}

impl_proto_schema_message!(CompilerPerfStatsProto, "xla.CompilerPerfStats");

/// Wire-compatible `xla.ProgramMemoryMetadata` (`xla/service/xla_compile_result.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProgramMemoryMetadataProto {
    /// Reserved HBM stack usage in bytes (field `1`).
    #[prost(int64, optional, tag = "1")]
    pub reserved_hbm_usage_bytes: Option<i64>,
}

impl_proto_schema_message!(ProgramMemoryMetadataProto, "xla.ProgramMemoryMetadata");

/// Wire-compatible `xla.CompilationResult` (`xla/service/xla_compile_result.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XlaCompilationResultProto {
    /// Compiled `HloModuleProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub hlo_module: Option<HloModuleProto>,
    /// Compilation timing breakdown (field `2`).
    #[prost(message, optional, tag = "2")]
    pub perf_stats: Option<CompilerPerfStatsProto>,
    /// Compilation status (field `3`).
    #[prost(message, optional, tag = "3")]
    pub status: Option<crate::proto::megascale::StatusProto>,
    /// Collected compilation counters (field `4`).
    #[prost(btree_map = "string, int64", tag = "4")]
    pub counters: std::collections::BTreeMap<String, i64>,
    /// Program memory metadata (field `5`).
    #[prost(message, optional, tag = "5")]
    pub program_memory_metadata: Option<ProgramMemoryMetadataProto>,
}

impl_proto_schema_message!(XlaCompilationResultProto, "xla.CompilationResult");

/// Wire-compatible `xla.ExecutionOptions` (`xla/xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XlaExecutionOptionsProto {
    /// Number of replicas (field `1`).
    #[prost(int32, tag = "1")]
    pub num_replicas: i32,
    /// Number of partitions (field `2`).
    #[prost(int32, tag = "2")]
    pub num_partitions: i32,
}

impl_proto_schema_message!(XlaExecutionOptionsProto, "xla.ExecutionOptions");

/// Wire-compatible `xla.HloInstructionProfile` (`hlo_profile_printer_data.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloInstructionProfileProto {
    /// Unique HLO instruction identifier (field `1`).
    #[prost(int64, tag = "1")]
    pub instruction_id: i64,
    /// HLO instruction mnemonic name (field `2`).
    #[prost(string, tag = "2")]
    pub instruction_name: String,
    /// Enclosing HLO computation name (field `3`).
    #[prost(string, tag = "3")]
    pub computation_name: String,
    /// Instruction execution category (e.g., `"convolution"`, `"dot"`, field `4`).
    #[prost(string, tag = "4")]
    pub category: String,
    /// Floating-point operation count (field `5`).
    #[prost(int64, tag = "5")]
    pub flop_count: i64,
    /// Transcendental operation count (field `6`).
    #[prost(int64, tag = "6")]
    pub transcendental_count: i64,
    /// Total memory bytes accessed (field `7`).
    #[prost(int64, tag = "7")]
    pub bytes_accessed: i64,
    /// Roofline optimal execution duration in seconds (field `8`).
    #[prost(double, tag = "8")]
    pub optimal_seconds: f64,
    /// Measured or simulated hardware clock cycles (field `9`).
    #[prost(int64, tag = "9")]
    pub clock_cycles: i64,
}

impl_proto_schema_message!(HloInstructionProfileProto, "xla.HloInstructionProfile");

/// Wire-compatible `xla.HloComputationProfile` (`hlo_profile_printer_data.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloComputationProfileProto {
    /// Unique HLO computation identifier (field `1`).
    #[prost(int64, tag = "1")]
    pub computation_id: i64,
    /// HLO computation name (field `2`).
    #[prost(string, tag = "2")]
    pub computation_name: String,
    /// Whether this computation is the module entry computation (field `3`).
    #[prost(bool, tag = "3")]
    pub is_entry_computation: bool,
    /// Per-instruction profile entries (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub instructions: Vec<HloInstructionProfileProto>,
    /// Total hardware clock cycles spent in this computation (field `5`).
    #[prost(int64, tag = "5")]
    pub total_cycles: i64,
}

impl_proto_schema_message!(HloComputationProfileProto, "xla.HloComputationProfile");

/// Wire-compatible `xla.HloProfilePrinterData` (`hlo_profile_printer_data.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HloProfilePrinterDataProto {
    /// Per-computation hardware profile records (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub computation_profiles: Vec<HloComputationProfileProto>,
    /// Total number of hardware profile counters (field `2`).
    #[prost(int64, tag = "2")]
    pub profile_counters_size: i64,
    /// Extra named profile counter values (field `3`).
    #[prost(btree_map = "string, int64", tag = "3")]
    pub extra_metrics: std::collections::BTreeMap<String, i64>,
    /// Entry computation name (field `4`).
    #[prost(string, tag = "4")]
    pub entry_computation: String,
}

impl_proto_schema_message!(HloProfilePrinterDataProto, "xla.HloProfilePrinterData");
