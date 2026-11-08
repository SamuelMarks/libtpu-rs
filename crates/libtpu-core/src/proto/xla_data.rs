//! XLA data model protobuf messages (`xla/xla_data.proto`).
//!
//! Covers shapes, physical layouts, tiles, program signatures, SPMD shardings, and device assignments.

use crate::proto::descriptor::impl_proto_schema_message;
use prost::Message;

/// Wire-compatible `xla.TileProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TileProto {
    /// Tile dimensions (e.g., `[8, 128]`), field `1`.
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub dimensions: Vec<i64>,
}

impl_proto_schema_message!(TileProto, "xla.TileProto");

/// Wire-compatible `xla.LayoutProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LayoutProto {
    /// Minor-to-major dimension ordering (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub minor_to_major: Vec<i64>,
    /// Tiles applied to the layout (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub tiles: Vec<TileProto>,
    /// Element size in bits (field `7`).
    #[prost(int64, tag = "7")]
    pub element_size_in_bits: i64,
    /// Memory space ID (`0` = default HBM, `1` = VMEM, `3` = pinned host, field `8`).
    #[prost(int64, tag = "8")]
    pub memory_space: i64,
    /// Primitive type used for sparse indices (field `11`).
    #[prost(int32, tag = "11")]
    pub index_primitive_type: i32,
    /// Primitive type used for sparse pointers (field `12`).
    #[prost(int32, tag = "12")]
    pub pointer_primitive_type: i32,
    /// Dynamic shape metadata prefix size in bytes (field `15`).
    #[prost(int64, tag = "15")]
    pub dynamic_shape_metadata_prefix_bytes: i64,
    /// Alignment of tail padding in elements (field `16`).
    #[prost(int64, tag = "16")]
    pub tail_padding_alignment_in_elements: i64,
}

impl_proto_schema_message!(LayoutProto, "xla.LayoutProto");

/// Wire-compatible `xla.ShapeProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShapeProto {
    /// Primitive element type (`PrimitiveType`, field `2`).
    #[prost(int32, tag = "2")]
    pub element_type: i32,
    /// Dimension extents (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub dimensions: Vec<i64>,
    /// Tuple subshapes if `element_type == TUPLE` (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub tuple_shapes: Vec<Self>,
    /// Optional physical layout (field `5`).
    #[prost(message, optional, tag = "5")]
    pub layout: Option<LayoutProto>,
    /// Dynamic dimension flags (field `6`).
    #[prost(bool, repeated, packed = "false", tag = "6")]
    pub is_dynamic_dimension: Vec<bool>,
}

impl_proto_schema_message!(ShapeProto, "xla.ShapeProto");

/// Wire-compatible `xla.ProgramShapeProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProgramShapeProto {
    /// Parameter shapes of the computation (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub parameters: Vec<ShapeProto>,
    /// Result shape of the computation (field `2`).
    #[prost(message, optional, tag = "2")]
    pub result: Option<ShapeProto>,
    /// Parameter names (field `3`).
    #[prost(string, repeated, tag = "3")]
    pub parameter_names: Vec<String>,
}

impl_proto_schema_message!(ProgramShapeProto, "xla.ProgramShapeProto");

/// Wire-compatible `xla.OpSharding` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct OpShardingProto {
    /// Sharding type (`0` = Replicated, `1` = Maximal, `2` = Tuple, `3` = Other/Tiled, `4` = Manual, field `1`).
    #[prost(int32, tag = "1")]
    pub sharding_type: i32,
    /// Shape for tuple shardings (field `2`).
    #[prost(message, optional, tag = "2")]
    pub tuple_shardings_shape: Option<ShapeProto>,
    /// Tile assignment dimensions (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub tile_assignment_dimensions: Vec<i64>,
    /// Tile assignment devices (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub tile_assignment_devices: Vec<i64>,
    /// Nested tuple shardings (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub tuple_shardings: Vec<Self>,
    /// Whether the last tile dimension is replicated across devices (field `6`).
    #[prost(bool, tag = "6")]
    pub replicate_on_last_tile_dim: bool,
    /// Compact iota reshape dimensions (field `9`).
    #[prost(int64, repeated, packed = "true", tag = "9")]
    pub iota_reshape_dims: Vec<i64>,
    /// Compact iota transpose permutation (field `10`).
    #[prost(int32, repeated, packed = "true", tag = "10")]
    pub iota_transpose_perm: Vec<i32>,
}

impl_proto_schema_message!(OpShardingProto, "xla.OpSharding");

/// Wire-compatible `xla.DeviceAssignmentProto.ComputationDevice` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ComputationDeviceProto {
    /// Device IDs assigned to each replica of this computation (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub replica_device_ids: Vec<i64>,
}

impl_proto_schema_message!(
    ComputationDeviceProto,
    "xla.DeviceAssignmentProto.ComputationDevice"
);

/// Wire-compatible `xla.DeviceAssignmentProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeviceAssignmentProto {
    /// Number of data-parallel replicas (field `1`).
    #[prost(int32, tag = "1")]
    pub replica_count: i32,
    /// Number of model-parallel computations/partitions (field `2`).
    #[prost(int32, tag = "2")]
    pub computation_count: i32,
    /// Per-computation replica device assignments (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub computation_devices: Vec<ComputationDeviceProto>,
}

impl_proto_schema_message!(DeviceAssignmentProto, "xla.DeviceAssignmentProto");

/// Wire-compatible `xla.SplitConfigProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SplitConfigProto {
    /// The dimension that is split (field `1`).
    #[prost(int64, tag = "1")]
    pub dimension: i64,
    /// Indices where each split point occurs (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub split_indices: Vec<i64>,
}

impl_proto_schema_message!(SplitConfigProto, "xla.SplitConfigProto");

/// Wire-compatible `xla.OpMetadata` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct OpMetadata {
    /// Framework op type name (field `1`).
    #[prost(string, tag = "1")]
    pub op_type: String,
    /// User-specified op name (field `2`).
    #[prost(string, tag = "2")]
    pub op_name: String,
    /// Source file path (field `3`).
    #[prost(string, tag = "3")]
    pub source_file: String,
    /// Source line number (field `4`).
    #[prost(int32, tag = "4")]
    pub source_line: i32,
    /// Deduplicated HLO op name (field `12`).
    #[prost(string, tag = "12")]
    pub deduplicated_name: String,
    /// Scheduling name (field `16`).
    #[prost(string, tag = "16")]
    pub scheduling_name: String,
}

impl_proto_schema_message!(OpMetadata, "xla.OpMetadata");

/// Wire-compatible `xla.ExecutionProfile` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutionProfile {
    /// Whether the executable was read from the compilation cache (field `1`).
    #[prost(bool, tag = "1")]
    pub compilation_cache_hit: bool,
    /// Time in milliseconds spent compiling (field `2`).
    #[prost(int64, tag = "2")]
    pub compile_time_ms: i64,
    /// Number of compute cycles spent (field `3`).
    #[prost(int64, tag = "3")]
    pub compute_cycle_count: i64,
    /// Compute time in nanoseconds without transfer (field `4`).
    #[prost(int64, tag = "4")]
    pub compute_time_ns: i64,
    /// Total compute and transfer time in nanoseconds (field `5`).
    #[prost(int64, tag = "5")]
    pub compute_and_transfer_time_ns: i64,
    /// Size of the binary code in the executable (field `6`).
    #[prost(int64, tag = "6")]
    pub executable_size_in_bytes: i64,
    /// Whether this profile was drawn from a profile cache (field `7`).
    #[prost(bool, tag = "7")]
    pub profile_cache_hit: bool,
    /// Whether a warm-up run was executed (field `8`).
    #[prost(bool, tag = "8")]
    pub warmup_run_executed: bool,
}

impl_proto_schema_message!(ExecutionProfile, "xla.ExecutionProfile");

/// Wire-compatible `xla.ExecutionHandle` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutionHandle {
    /// Handle identifier (field `1`).
    #[prost(int64, tag = "1")]
    pub handle: i64,
}

impl_proto_schema_message!(ExecutionHandle, "xla.ExecutionHandle");

/// Wire-compatible `xla.GlobalDataHandle` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GlobalDataHandle {
    /// Global data handle identifier (field `1`).
    #[prost(int64, tag = "1")]
    pub handle: i64,
}

impl_proto_schema_message!(GlobalDataHandle, "xla.GlobalDataHandle");

/// Wire-compatible `xla.DeviceHandle` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeviceHandle {
    /// Device handle identifier (field `1`).
    #[prost(int64, tag = "1")]
    pub handle: i64,
    /// Number of model-parallel virtual devices (field `2`).
    #[prost(int64, tag = "2")]
    pub device_count: i64,
}

impl_proto_schema_message!(DeviceHandle, "xla.DeviceHandle");

/// Wire-compatible `xla.ChannelHandle` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ChannelHandle {
    /// Channel identifier (field `1`).
    #[prost(int64, tag = "1")]
    pub handle: i64,
    /// Channel type (`0` = `Invalid`, `1` = `DeviceToDevice`, `2` = `DeviceToHost`, `3` = `HostToDevice`, field `2`).
    #[prost(int32, tag = "2")]
    pub channel_type: i32,
}

impl_proto_schema_message!(ChannelHandle, "xla.ChannelHandle");

/// Wire-compatible `xla.LiteralProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct LiteralProto {
    /// Literal tensor shape (field `1`).
    #[prost(message, optional, tag = "1")]
    pub shape: Option<ShapeProto>,
    /// Predicate values (`PRED`, field `2`).
    #[prost(bool, repeated, packed = "true", tag = "2")]
    pub preds: Vec<bool>,
    /// Unsigned 8-bit integer bytes (`U8`, field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub u8s: Vec<u8>,
    /// Signed 32-bit integers (`S32`, field `4`).
    #[prost(int32, repeated, packed = "true", tag = "4")]
    pub s32s: Vec<i32>,
    /// Signed 64-bit integers (`S64`, field `5`).
    #[prost(int64, repeated, packed = "true", tag = "5")]
    pub s64s: Vec<i64>,
    /// Unsigned 32-bit integers (`U32`, field `6`).
    #[prost(uint32, repeated, packed = "true", tag = "6")]
    pub u32s: Vec<u32>,
    /// Unsigned 64-bit integers (`U64`, field `7`).
    #[prost(uint64, repeated, packed = "true", tag = "7")]
    pub u64s: Vec<u64>,
    /// 32-bit IEEE floats (`F32`, field `8`).
    #[prost(float, repeated, packed = "true", tag = "8")]
    pub f32s: Vec<f32>,
    /// 64-bit IEEE doubles (`F64`, field `9`).
    #[prost(double, repeated, packed = "true", tag = "9")]
    pub f64s: Vec<f64>,
    /// Nested tuple element literals (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub tuple_literals: Vec<Self>,
    /// Little-endian `F16` bytes (field `11`).
    #[prost(bytes = "vec", tag = "11")]
    pub f16s: Vec<u8>,
    /// Interleaved real/imaginary `F32` pairs (`C64`, field `12`).
    #[prost(float, repeated, packed = "true", tag = "12")]
    pub c64s: Vec<f32>,
    /// Little-endian `BF16` bytes (field `13`).
    #[prost(bytes = "vec", tag = "13")]
    pub bf16s: Vec<u8>,
    /// Signed 8-bit integer bytes (`S8`, field `15`).
    #[prost(bytes = "vec", tag = "15")]
    pub s8s: Vec<u8>,
    /// Little-endian `U16` bytes (field `16`).
    #[prost(bytes = "vec", tag = "16")]
    pub u16s: Vec<u8>,
    /// Little-endian `S16` bytes (field `17`).
    #[prost(bytes = "vec", tag = "17")]
    pub s16s: Vec<u8>,
    /// Interleaved real/imaginary `F64` pairs (`C128`, field `18`).
    #[prost(double, repeated, packed = "true", tag = "18")]
    pub c128s: Vec<f64>,
    /// `F8E5M2` bytes (field `19`).
    #[prost(bytes = "vec", tag = "19")]
    pub f8e5m2s: Vec<u8>,
    /// `F8E4M3FN` bytes (field `20`).
    #[prost(bytes = "vec", tag = "20")]
    pub f8e4m3fns: Vec<u8>,
    /// Packed `S4` nibbles (field `21`).
    #[prost(bytes = "vec", tag = "21")]
    pub s4s: Vec<u8>,
    /// Packed `U4` nibbles (field `22`).
    #[prost(bytes = "vec", tag = "22")]
    pub u4s: Vec<u8>,
    /// `F8E4M3B11FNUZ` bytes (field `23`).
    #[prost(bytes = "vec", tag = "23")]
    pub f8e4m3b11fnuzs: Vec<u8>,
}

impl_proto_schema_message!(LiteralProto, "xla.LiteralProto");

/// Wire-compatible `xla.ParameterReplication` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ParameterReplicationProto {
    /// Per-leaf-buffer replication flags across data-parallel replicas (field `1`).
    #[prost(bool, repeated, packed = "true", tag = "1")]
    pub replicated_at_leaf_buffers: Vec<bool>,
}

impl_proto_schema_message!(ParameterReplicationProto, "xla.ParameterReplication");

/// Wire-compatible `xla.WindowDimension` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct WindowDimension {
    /// Window size in this dimension (field `1`).
    #[prost(int64, tag = "1")]
    pub size: i64,
    /// Window movement stride in this dimension (field `2`).
    #[prost(int64, tag = "2")]
    pub stride: i64,
    /// Low-end padding in this dimension (field `3`).
    #[prost(int64, tag = "3")]
    pub padding_low: i64,
    /// High-end padding in this dimension (field `4`).
    #[prost(int64, tag = "4")]
    pub padding_high: i64,
    /// Kernel window dilation factor (field `5`).
    #[prost(int64, tag = "5")]
    pub window_dilation: i64,
    /// Base input dilation factor (field `6`).
    #[prost(int64, tag = "6")]
    pub base_dilation: i64,
    /// Logical window reversal flag (field `7`).
    #[prost(bool, tag = "7")]
    pub window_reversal: bool,
}

impl_proto_schema_message!(WindowDimension, "xla.WindowDimension");

/// Wire-compatible `xla.Window` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct Window {
    /// Per-spatial-dimension window configurations (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub dimensions: Vec<WindowDimension>,
}

impl_proto_schema_message!(Window, "xla.Window");

/// Wire-compatible `xla.GatherDimensionNumbers` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GatherDimensionNumbers {
    /// Output dimensions corresponding to non-collapsed slice dimensions (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub offset_dims: Vec<i64>,
    /// Operand dimensions collapsed out of the slice (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub collapsed_slice_dims: Vec<i64>,
    /// Map from index vector component to operand dimension (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub start_index_map: Vec<i64>,
    /// Dimension in `start_indices` containing the index vector (field `4`).
    #[prost(int64, tag = "4")]
    pub index_vector_dim: i64,
    /// Batch dimensions in the operand tensor (field `5`).
    #[prost(int64, repeated, packed = "true", tag = "5")]
    pub operand_batching_dims: Vec<i64>,
    /// Batch dimensions in the `start_indices` tensor (field `6`).
    #[prost(int64, repeated, packed = "true", tag = "6")]
    pub start_indices_batching_dims: Vec<i64>,
}

impl_proto_schema_message!(GatherDimensionNumbers, "xla.GatherDimensionNumbers");

/// Wire-compatible `xla.ScatterDimensionNumbers` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ScatterDimensionNumbers {
    /// Dimensions in `updates` that are window dimensions (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub update_window_dims: Vec<i64>,
    /// Window dimensions inserted into `updates` (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub inserted_window_dims: Vec<i64>,
    /// Map from scatter index vector component to operand dimension (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub scatter_dims_to_operand_dims: Vec<i64>,
    /// Dimension in `scatter_indices` containing the index vector (field `4`).
    #[prost(int64, tag = "4")]
    pub index_vector_dim: i64,
    /// Batch dimensions in the input operand (field `5`).
    #[prost(int64, repeated, packed = "true", tag = "5")]
    pub input_batching_dims: Vec<i64>,
    /// Batch dimensions in `scatter_indices` (field `6`).
    #[prost(int64, repeated, packed = "true", tag = "6")]
    pub scatter_indices_batching_dims: Vec<i64>,
}

impl_proto_schema_message!(ScatterDimensionNumbers, "xla.ScatterDimensionNumbers");

/// Wire-compatible `xla.ConvolutionDimensionNumbers` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ConvolutionDimensionNumbers {
    /// Kernel input feature dimension (field `3`).
    #[prost(int64, tag = "3")]
    pub kernel_input_feature_dimension: i64,
    /// Kernel output feature dimension (field `4`).
    #[prost(int64, tag = "4")]
    pub kernel_output_feature_dimension: i64,
    /// Kernel spatial dimensions (field `6`).
    #[prost(int64, repeated, packed = "true", tag = "6")]
    pub kernel_spatial_dimensions: Vec<i64>,
    /// Input batch dimension (field `7`).
    #[prost(int64, tag = "7")]
    pub input_batch_dimension: i64,
    /// Input feature dimension (field `8`).
    #[prost(int64, tag = "8")]
    pub input_feature_dimension: i64,
    /// Output batch dimension (field `9`).
    #[prost(int64, tag = "9")]
    pub output_batch_dimension: i64,
    /// Output feature dimension (field `10`).
    #[prost(int64, tag = "10")]
    pub output_feature_dimension: i64,
    /// Input spatial dimensions (field `11`).
    #[prost(int64, repeated, packed = "true", tag = "11")]
    pub input_spatial_dimensions: Vec<i64>,
    /// Output spatial dimensions (field `12`).
    #[prost(int64, repeated, packed = "true", tag = "12")]
    pub output_spatial_dimensions: Vec<i64>,
}

impl_proto_schema_message!(
    ConvolutionDimensionNumbers,
    "xla.ConvolutionDimensionNumbers"
);

/// Wire-compatible `xla.DotDimensionNumbers` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DotDimensionNumbers {
    /// LHS contracting dimensions (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub lhs_contracting_dimensions: Vec<i64>,
    /// RHS contracting dimensions (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub rhs_contracting_dimensions: Vec<i64>,
    /// LHS batch dimensions (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub lhs_batch_dimensions: Vec<i64>,
    /// RHS batch dimensions (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub rhs_batch_dimensions: Vec<i64>,
}

impl_proto_schema_message!(DotDimensionNumbers, "xla.DotDimensionNumbers");

/// Wire-compatible `xla.PaddingConfig.PaddingConfigDimension` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PaddingConfigDimension {
    /// Low-end edge padding (field `1`).
    #[prost(int64, tag = "1")]
    pub edge_padding_low: i64,
    /// High-end edge padding (field `2`).
    #[prost(int64, tag = "2")]
    pub edge_padding_high: i64,
    /// Interior padding between elements (field `3`).
    #[prost(int64, tag = "3")]
    pub interior_padding: i64,
}

impl_proto_schema_message!(
    PaddingConfigDimension,
    "xla.PaddingConfig.PaddingConfigDimension"
);

/// Wire-compatible `xla.PaddingConfig` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PaddingConfig {
    /// Per-dimension padding configuration (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub dimensions: Vec<PaddingConfigDimension>,
}

impl_proto_schema_message!(PaddingConfig, "xla.PaddingConfig");

/// Wire-compatible `xla.ReplicaGroup` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ReplicaGroup {
    /// Replica or partition IDs belonging to this collective group (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub replica_ids: Vec<i64>,
}

impl_proto_schema_message!(ReplicaGroup, "xla.ReplicaGroup");

/// Wire-compatible `xla.SourceTarget` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SourceTarget {
    /// Source replica/partition ID (field `1`).
    #[prost(int64, tag = "1")]
    pub source: i64,
    /// Target replica/partition ID (field `2`).
    #[prost(int64, tag = "2")]
    pub target: i64,
}

impl_proto_schema_message!(SourceTarget, "xla.SourceTarget");

/// Wire-compatible `xla.PrecisionConfig` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PrecisionConfig {
    /// Per-operand precision settings (`0` = Default, `1` = High, `2` = Highest, field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub operand_precision: Vec<i32>,
    /// Dot accumulation algorithm enum (field `2`).
    #[prost(int32, tag = "2")]
    pub algorithm: i32,
}

impl_proto_schema_message!(PrecisionConfig, "xla.PrecisionConfig");

/// Wire-compatible `xla.FrontendAttributes` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct FrontendAttributes {
    /// Key-value attribute map passed from frontend to XLA backend (field `1`).
    #[prost(btree_map = "string, string", tag = "1")]
    pub map: std::collections::BTreeMap<String, String>,
}

impl_proto_schema_message!(FrontendAttributes, "xla.FrontendAttributes");

/// Wire-compatible `xla.Statistic` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct Statistic {
    /// Single-word statistic name (field `1`).
    #[prost(string, tag = "1")]
    pub stat_name: String,
    /// Statistic value in `[0.0, 100.0]` (field `2`).
    #[prost(double, tag = "2")]
    pub stat_val: f64,
}

impl_proto_schema_message!(Statistic, "xla.Statistic");

/// Wire-compatible `xla.StatisticsViz` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct StatisticsViz {
    /// Index of statistic to visualize (field `1`).
    #[prost(int64, tag = "1")]
    pub stat_index_to_visualize: i64,
    /// Tracked statistics (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub statistics: Vec<Statistic>,
}

impl_proto_schema_message!(StatisticsViz, "xla.StatisticsViz");

/// Wire-compatible `xla.TriangularSolveOptions` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TriangularSolveOptions {
    /// Solve `op(a) * x = b` (`true`) or `x * op(a) = b` (`false`, field `1`).
    #[prost(bool, tag = "1")]
    pub left_side: bool,
    /// Use lower (`true`) or upper (`false`) triangle of `a` (field `2`).
    #[prost(bool, tag = "2")]
    pub lower: bool,
    /// Assume unit diagonal elements on `a` (field `3`).
    #[prost(bool, tag = "3")]
    pub unit_diagonal: bool,
    /// Transpose mode for `a` (`0` = Invalid, `1` = `NoTranspose`, `2` = `Transpose`, `3` = `Adjoint`, field `4`).
    #[prost(int32, tag = "4")]
    pub transpose_a: i32,
}

impl_proto_schema_message!(TriangularSolveOptions, "xla.TriangularSolveOptions");

/// Wire-compatible `xla.CholeskyOptions` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CholeskyOptions {
    /// Compute lower (`L * L^T`, `true`) or upper (`U^T * U`, `false`) Cholesky factor (field `1`).
    #[prost(bool, tag = "1")]
    pub lower: bool,
}

impl_proto_schema_message!(CholeskyOptions, "xla.CholeskyOptions");
