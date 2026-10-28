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
