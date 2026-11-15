//! XLA core data, tensor shape, physical tile/layout, and SPMD sharding protobuf types.
//!
//! All types in this module are direct exports and aliases of `prost-build` generated structs
//! compiled from `third_party/tensorflow/compiler/xla/xla_data.proto`.

pub use super::descriptor::generated_prost_schemas::xla::{
    ChannelHandle, CholeskyOptions, ConvolutionDimensionNumbers, DeviceAssignmentProto,
    DeviceHandle, DotDimensionNumbers, ExecutionHandle, ExecutionProfile, FrontendAttributes,
    GatherDimensionNumbers, GlobalDataHandle, LayoutProto, LiteralProto, OpMetadata, PaddingConfig,
    PrecisionConfig, PrimitiveType, ProgramShapeProto, ReplicaGroup, ScatterDimensionNumbers,
    ShapeProto, SourceTarget, SplitConfigProto, Statistic, StatisticsViz, TileProto,
    TriangularSolveOptions, Window, WindowDimension,
};

/// Alias for `xla.OpSharding` generated from `xla_data.proto`.
pub type OpShardingProto = super::descriptor::generated_prost_schemas::xla::OpSharding;

/// Alias for `xla.DeviceAssignmentProto.ComputationDevice` generated from `xla_data.proto`.
pub type ComputationDeviceProto =
    super::descriptor::generated_prost_schemas::xla::device_assignment_proto::ComputationDevice;

/// Alias for `xla.ParameterReplication` generated from `xla_data.proto`.
pub type ParameterReplicationProto =
    super::descriptor::generated_prost_schemas::xla::ParameterReplication;

/// Alias for `xla.PaddingConfig.PaddingConfigDimension` generated from `xla_data.proto`.
pub type PaddingConfigDimension =
    super::descriptor::generated_prost_schemas::xla::padding_config::PaddingConfigDimension;
