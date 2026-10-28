//! TPU topology, compilation environment, and `SparseCore` embedding protobuf messages.
//!
//! Synchronized with `tpu_topology.proto`, `tpu_embedding_configuration.proto`, and `sparse_core_layout.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use crate::proto::hlo::HloModuleProtoWithConfig;
use crate::proto::xla_data::DeviceAssignmentProto;
use prost::Message;

/// Wire-compatible `tpu.TpuCompilationEnvironmentProto` (`tpu_compilation_environment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompilationEnvironmentProto {
    /// XLA TPU Megacore fusion toggle (field `1`).
    #[prost(bool, tag = "1")]
    pub xla_tpu_enable_megacore_fusion: bool,
    /// XLA TPU async collective fusion toggle (field `2`).
    #[prost(bool, tag = "2")]
    pub xla_tpu_enable_async_collective_fusion: bool,
    /// XLA TPU VMEM limit in KiB (field `3`).
    #[prost(int64, tag = "3")]
    pub xla_tpu_scoped_vmem_limit_kib: i64,
    /// Raw custom flag string overrides (field `4`).
    #[prost(string, tag = "4")]
    pub custom_flags: String,
}

impl_proto_schema_message!(
    TpuCompilationEnvironmentProto,
    "tpu.TpuCompilationEnvironmentProto"
);

/// Wire-compatible `tpu.TpuTopologyArgsProto` (`tpu_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuTopologyArgsProto {
    /// TPU platform generation enum (`TpuVersionEnum`, field `1`).
    #[prost(int32, tag = "1")]
    pub version: i32,
    /// Number of hosts in the pod slice (field `2`).
    #[prost(int32, tag = "2")]
    pub num_hosts: i32,
    /// Number of chips per host (field `3`).
    #[prost(int32, tag = "3")]
    pub chips_per_host: i32,
    /// Mesh dimensions `[x, y, z]` (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub mesh_bounds: Vec<i64>,
    /// Platform version string (field `5`).
    #[prost(string, tag = "5")]
    pub platform_version: String,
    /// Deterministic topology fingerprint (field `6`).
    #[prost(string, tag = "6")]
    pub fingerprint: String,
}

impl_proto_schema_message!(TpuTopologyArgsProto, "tpu.TpuTopologyArgsProto");

/// Wire-compatible `tensorflow.tpu.TPUHardwareFeature` (`tensorflow/core/protobuf/tpu/topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuHardwareFeatureProto {
    /// Embedding feature version enum (field `1`).
    #[prost(int32, tag = "1")]
    pub embedding_feature: i32,
    /// Number of embedding (`SparseCore`) devices per chip (field `2`).
    #[prost(int32, tag = "2")]
    pub num_embedding_devices_per_chip: i32,
}

impl_proto_schema_message!(TpuHardwareFeatureProto, "tensorflow.tpu.TPUHardwareFeature");

/// Wire-compatible `tensorflow.tpu.TopologyProto` (`tensorflow/core/protobuf/tpu/topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TopologyProto {
    /// Mesh shape `[x, y, z, core]` (field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub mesh_shape: Vec<i32>,
    /// Number of TensorFlow tasks/hosts in the cluster (field `2`).
    #[prost(int32, tag = "2")]
    pub num_tasks: i32,
    /// Number of TPU devices per task (field `3`).
    #[prost(int32, tag = "3")]
    pub num_tpu_devices_per_task: i32,
    /// Flattened `(x, y, z, core)` device coordinates (field `4`).
    #[prost(int32, repeated, packed = "true", tag = "4")]
    pub device_coordinates: Vec<i32>,
    /// Hardware feature capabilities (field `5`).
    #[prost(message, optional, tag = "5")]
    pub tpu_hardware_feature: Option<TpuHardwareFeatureProto>,
}

impl_proto_schema_message!(TopologyProto, "tensorflow.tpu.TopologyProto");

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingTableDescriptorProto {
    /// Table name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Vocabulary size (number of rows, field `2`).
    #[prost(int64, tag = "2")]
    pub vocabulary_size: i64,
    /// Embedding dimension width (field `3`).
    #[prost(int64, tag = "3")]
    pub dimension: i64,
    /// Number of features referencing this table (field `4`).
    #[prost(int64, tag = "4")]
    pub num_features: i64,
}

impl_proto_schema_message!(
    EmbeddingTableDescriptorProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor"
);

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingFeatureDescriptorProto {
    /// Feature name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Index of the corresponding table in `table_descriptor` (field `2`).
    #[prost(int32, tag = "2")]
    pub table_id: i32,
    /// Input activation shape dimensions (field `3`).
    #[prost(int32, repeated, packed = "true", tag = "3")]
    pub input_shape: Vec<i32>,
}

impl_proto_schema_message!(
    EmbeddingFeatureDescriptorProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor"
);

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingSpmdShardingProto {
    /// Whether SPMD sharding is enabled (field `1`).
    #[prost(bool, tag = "1")]
    pub enabled: bool,
    /// Number of cores per replica in SPMD mode (field `2`).
    #[prost(int32, tag = "2")]
    pub num_cores_per_replica: i32,
}

impl_proto_schema_message!(
    EmbeddingSpmdShardingProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding"
);

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration` (`xla/tpu/protobuf/tpu_embedding_configuration.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingConfigurationProto {
    /// Table descriptors (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub table_descriptor: Vec<EmbeddingTableDescriptorProto>,
    /// Mode (`1` = Inference, `2` = Training, `3` = `BackwardPassOnly`, field `2`).
    #[prost(int32, tag = "2")]
    pub mode: i32,
    /// Batch size per `TensorCore` (field `3`).
    #[prost(int32, tag = "3")]
    pub batch_size_per_tensor_core: i32,
    /// Number of hosts (field `4`).
    #[prost(int32, tag = "4")]
    pub num_hosts: i32,
    /// Number of `TensorCores` (field `5`).
    #[prost(int32, tag = "5")]
    pub num_tensor_cores: i32,
    /// Table sharding strategy (`0` = Unspecified, `1` = Mod, `2` = Div, field `6`).
    #[prost(int32, tag = "6")]
    pub sharding_strategy: i32,
    /// Whether `SparseCore` execution is pipelined with `TensorCore` (field `7`).
    #[prost(bool, tag = "7")]
    pub pipeline_execution_with_tensor_core: bool,
    /// Directory for embedding profile data dumps (field `9`).
    #[prost(string, tag = "9")]
    pub profile_data_directory: String,
    /// Feature descriptors (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub feature_descriptor: Vec<EmbeddingFeatureDescriptorProto>,
    /// Optional SPMD sharding configuration (field `11`).
    #[prost(message, optional, tag = "11")]
    pub spmd_sharding: Option<EmbeddingSpmdShardingProto>,
}

impl_proto_schema_message!(
    TpuEmbeddingConfigurationProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration"
);

/// Wire-compatible `tensorflow.tpu.SparseCoreTableLayout` (`tensorflow/core/tpu/kernels/sparse_core_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreTableLayoutProto {
    /// Table name (field `1`).
    #[prost(string, tag = "1")]
    pub table_name: String,
    /// Number of `SparseCore` shards across the slice (field `3`).
    #[prost(int32, tag = "3")]
    pub num_sparse_cores: i32,
    /// Number of partitions per chip (field `4`).
    #[prost(int32, tag = "4")]
    pub num_partitions: i32,
    /// Total rows per `SparseCore` shard (field `5`).
    #[prost(int32, tag = "5")]
    pub total_rows_per_sparse_core_shard: i32,
    /// Logical shard dimensions `[vocab, dim]` (field `6`).
    #[prost(int64, repeated, packed = "true", tag = "6")]
    pub unsharded_shape: Vec<i64>,
    /// Physical padded shape `[padded_vocab, padded_dim]` (field `7`).
    #[prost(int64, repeated, packed = "true", tag = "7")]
    pub unsharded_padded_shape: Vec<i64>,
}

impl_proto_schema_message!(
    SparseCoreTableLayoutProto,
    "tensorflow.tpu.SparseCoreTableLayout"
);

/// Wire-compatible `tensorflow.tpu.SparseCoreTableLayouts` (`tensorflow/core/tpu/kernels/sparse_core_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreTableLayoutsProto {
    /// Per-table `SparseCore` layout specifications (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub tables: Vec<SparseCoreTableLayoutProto>,
}

impl_proto_schema_message!(
    SparseCoreTableLayoutsProto,
    "tensorflow.tpu.SparseCoreTableLayouts"
);

/// Wire-compatible `tensorflow.tpu.CompilationResultProto` (`xla/tpu/protobuf/compilation_result.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilationResultProto {
    /// Abseil status code (`0` = OK, field `1`).
    #[prost(int32, tag = "1")]
    pub status_code: i32,
    /// Error message when `status_code != 0` (field `2`).
    #[prost(string, tag = "2")]
    pub status_error_message: String,
    /// HLO protocols produced by compilation (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub hlo_protos: Vec<HloModuleProtoWithConfig>,
}

impl_proto_schema_message!(
    CompilationResultProto,
    "tensorflow.tpu.CompilationResultProto"
);

/// Wire-compatible `tensorflow.tpu.TPUCompileMetadataProto` (`tensorflow/core/protobuf/tpu/compile_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompileMetadataProto {
    /// Number of replicas (field `3`).
    #[prost(int32, tag = "3")]
    pub num_replicas: i32,
    /// Number of cores per replica (field `4`).
    #[prost(int32, tag = "4")]
    pub num_cores_per_replica: i32,
    /// Static device assignment (field `8`).
    #[prost(message, optional, tag = "8")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Whether SPMD is enabled (field `15`).
    #[prost(bool, tag = "15")]
    pub use_spmd_for_xla_partitioning: bool,
}

impl_proto_schema_message!(
    TpuCompileMetadataProto,
    "tensorflow.tpu.TPUCompileMetadataProto"
);

/// Wire-compatible `tpu.TpuEmbeddingPartitionedConfigProto` produced by `TpuEmbeddingEngine_ExecutePartitioner`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingPartitionedConfigProto {
    /// Number of `SparseCore` partitions across the slice (field `1`).
    #[prost(int32, tag = "1")]
    pub num_partitions: i32,
    /// Partitioned table names (field `2`).
    #[prost(string, repeated, tag = "2")]
    pub table_names: Vec<String>,
    /// Original serialized `TpuEmbeddingConfigurationProto` bytes (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub raw_config_bytes: Vec<u8>,
}

impl_proto_schema_message!(
    TpuEmbeddingPartitionedConfigProto,
    "tpu.TpuEmbeddingPartitionedConfigProto"
);

/// Wire-compatible `tpu.TpuEmbeddingMemoryConfigProto` for `SparseCore` memory configuration.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingMemoryConfigProto {
    /// Total HBM bytes reserved for `SparseCore` lookup tables (field `1`).
    #[prost(uint64, tag = "1")]
    pub hbm_buffers_bytes: u64,
    /// Number of collated host memory segments (field `2`).
    #[prost(int32, tag = "2")]
    pub num_segments: i32,
    /// Collated per-host memory configuration blobs (field `3`).
    #[prost(bytes = "vec", repeated, tag = "3")]
    pub segment_blobs: Vec<Vec<u8>>,
}

impl_proto_schema_message!(
    TpuEmbeddingMemoryConfigProto,
    "tpu.TpuEmbeddingMemoryConfigProto"
);

/// Wire-compatible `tpu.TpuEmbeddingHostNetworkConfigProto` used by `TpuEmbeddingEngine_ConfigureHost`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingHostNetworkConfigProto {
    /// Host ordinal index (field `1`).
    #[prost(int32, tag = "1")]
    pub host_id: i32,
    /// Network endpoint address (field `2`).
    #[prost(string, tag = "2")]
    pub endpoint_address: String,
    /// Raw partition configuration payload (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub partition_payload: Vec<u8>,
}

impl_proto_schema_message!(
    TpuEmbeddingHostNetworkConfigProto,
    "tpu.TpuEmbeddingHostNetworkConfigProto"
);
