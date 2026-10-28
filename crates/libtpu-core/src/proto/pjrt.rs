//! PJRT compilation, execution, metadata, ABI versioning, and distributed topology messages.
//!
//! Synchronized with `xla/pjrt/proto/*.proto` and `xla/pjrt/distributed/protocol.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use crate::proto::xla_data::{DeviceAssignmentProto, ShapeProto};
use num_enum::{FromPrimitive, IntoPrimitive};
use prost::Message;
use std::collections::BTreeMap;

/// Wire-compatible `xla.OptionOverrideProto` (`xla/pjrt/proto/compile_options.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct OptionOverrideProto {
    /// String option override (field `1`).
    #[prost(string, tag = "1")]
    pub string_field: String,
    /// Boolean option override (field `2`).
    #[prost(bool, tag = "2")]
    pub bool_field: bool,
    /// Integer option override (field `3`).
    #[prost(int64, tag = "3")]
    pub int_field: i64,
    /// Floating-point option override (field `4`).
    #[prost(double, tag = "4")]
    pub double_field: f64,
}

impl_proto_schema_message!(OptionOverrideProto, "xla.OptionOverrideProto");

/// Wire-compatible `xla.ExecutableBuildOptionsProto` (`xla/pjrt/proto/compile_options.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutableBuildOptionsProto {
    /// Device ordinal (`-1` if unspecified, field `1`).
    #[prost(int64, tag = "1")]
    pub device_ordinal: i64,
    /// Result layout (field `2`).
    #[prost(message, optional, tag = "2")]
    pub result_layout: Option<ShapeProto>,
    /// Number of replicas (field `4`).
    #[prost(int64, tag = "4")]
    pub num_replicas: i64,
    /// Number of SPMD partitions (field `5`).
    #[prost(int64, tag = "5")]
    pub num_partitions: i64,
    /// Whether SPMD partitioning is enabled (field `6`).
    #[prost(bool, tag = "6")]
    pub use_spmd_partitioning: bool,
    /// Whether auto-SPMD partitioning is enabled (field `7`).
    #[prost(bool, tag = "7")]
    pub use_auto_spmd_partitioning: bool,
    /// Static device assignment (field `9`).
    #[prost(message, optional, tag = "9")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Per-output SPMD sharding propagation flags (field `12`).
    #[prost(bool, repeated, packed = "true", tag = "12")]
    pub allow_spmd_sharding_propagation_to_output: Vec<bool>,
    /// Per-parameter SPMD sharding propagation flags (field `18`).
    #[prost(bool, repeated, packed = "true", tag = "18")]
    pub allow_spmd_sharding_propagation_to_parameters: Vec<bool>,
    /// Whether the Shardy partitioner is enabled (field `19`).
    #[prost(bool, tag = "19")]
    pub use_shardy_partitioner: bool,
    /// Process index in multi-host jobs (field `22`).
    #[prost(int64, tag = "22")]
    pub process_index: i64,
    /// Total process count in multi-host jobs (field `23`).
    #[prost(int64, tag = "23")]
    pub process_count: i64,
}

impl_proto_schema_message!(
    ExecutableBuildOptionsProto,
    "xla.ExecutableBuildOptionsProto"
);

/// Wire-compatible `xla.CompileOptionsProto` (`xla/pjrt/proto/compile_options.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct CompileOptionsProto {
    /// Argument layouts (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub argument_layouts: Vec<ShapeProto>,
    /// Whether parameters are tupled (field `2`).
    #[prost(bool, tag = "2")]
    pub parameter_is_tupled_arguments: bool,
    /// Build options (field `3`).
    #[prost(message, optional, tag = "3")]
    pub executable_build_options: Option<ExecutableBuildOptionsProto>,
    /// Compile portable executable flag (field `4`).
    #[prost(bool, tag = "4")]
    pub compile_portable_executable: bool,
    /// Profile version (field `5`).
    #[prost(int64, tag = "5")]
    pub profile_version: i64,
    /// Serialized multi-slice configuration bytes (field `6`).
    #[prost(bytes = "vec", tag = "6")]
    pub serialized_multi_slice_config: Vec<u8>,
    /// Environment option overrides keyed by flag name (field `7`).
    #[prost(btree_map = "string, message", tag = "7")]
    pub env_option_overrides: BTreeMap<String, OptionOverrideProto>,
}

impl_proto_schema_message!(CompileOptionsProto, "xla.CompileOptionsProto");

/// Execution mode enum (`xla.ExecutionModeProto`) for [`ExecuteOptionsProto`].
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, FromPrimitive, IntoPrimitive)]
pub enum ExecutionModeProto {
    /// Unspecified execution mode (`0`).
    #[default]
    Unspecified = 0,
    /// Default runtime execution mode (`1`).
    Default = 1,
    /// Synchronous host-blocking execution (`2`).
    Synchronous = 2,
    /// Asynchronous non-blocking execution (`3`).
    Asynchronous = 3,
}

/// Wire-compatible `xla.ExecuteOptionsProto` (`xla/pjrt/proto/execute_options.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecuteOptionsProto {
    /// Whether arguments are passed as a single tuple (field `1`).
    #[prost(bool, tag = "1")]
    pub arguments_are_tupled: bool,
    /// Whether tuple results should be untupled into individual buffers (field `2`).
    #[prost(bool, tag = "2")]
    pub untuple_result: bool,
    /// Launch identifier for multi-host coordination (field `3`).
    #[prost(int32, tag = "3")]
    pub launch_id: i32,
    /// Whether strict input shape checking is enforced (field `4`).
    #[prost(bool, tag = "4")]
    pub strict_shape_checking: bool,
    /// Execution mode (`ExecutionModeProto`, field `6`).
    #[prost(int32, tag = "6")]
    pub execution_mode: i32,
    /// Input parameter indices that must not be donated (field `7`).
    #[prost(int32, repeated, packed = "true", tag = "7")]
    pub non_donatable_input_indices: Vec<i32>,
    /// PRNG execution seed (field `9`).
    #[prost(int64, tag = "9")]
    pub seed: i64,
}

impl_proto_schema_message!(ExecuteOptionsProto, "xla.ExecuteOptionsProto");

/// Wire-compatible `xla.CompiledMemoryStatsProto` (`xla/pjrt/proto/executable_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompiledMemoryStatsProto {
    /// Device generated code size in bytes (field `1`).
    #[prost(int64, tag = "1")]
    pub generated_code_size_in_bytes: i64,
    /// Device argument buffer size in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub argument_size_in_bytes: i64,
    /// Device output buffer size in bytes (field `3`).
    #[prost(int64, tag = "3")]
    pub output_size_in_bytes: i64,
    /// Device aliased buffer size in bytes (field `4`).
    #[prost(int64, tag = "4")]
    pub alias_size_in_bytes: i64,
    /// Device temporary scratchpad size in bytes (field `5`).
    #[prost(int64, tag = "5")]
    pub temp_size_in_bytes: i64,
    /// Host generated code size in bytes (field `7`).
    #[prost(int64, tag = "7")]
    pub host_generated_code_size_in_bytes: i64,
    /// Host argument buffer size in bytes (field `8`).
    #[prost(int64, tag = "8")]
    pub host_argument_size_in_bytes: i64,
    /// Host output buffer size in bytes (field `9`).
    #[prost(int64, tag = "9")]
    pub host_output_size_in_bytes: i64,
    /// Host aliased buffer size in bytes (field `10`).
    #[prost(int64, tag = "10")]
    pub host_alias_size_in_bytes: i64,
    /// Host temporary buffer size in bytes (field `11`).
    #[prost(int64, tag = "11")]
    pub host_temp_size_in_bytes: i64,
    /// Serialized `BufferAssignmentProto` bytes (field `13`).
    #[prost(bytes = "vec", tag = "13")]
    pub serialized_buffer_assignment: Vec<u8>,
    /// Peak device HBM usage in bytes (field `14`).
    #[prost(int64, tag = "14")]
    pub peak_memory_in_bytes: i64,
    /// Total device footprint in bytes (field `15`).
    #[prost(int64, tag = "15")]
    pub total_size_in_bytes: i64,
    /// Total cumulative allocation bytes (field `16`).
    #[prost(int64, tag = "16")]
    pub total_allocation_bytes: i64,
    /// Indefinite/live allocation bytes (field `17`).
    #[prost(int64, tag = "17")]
    pub indefinite_allocations: i64,
    /// Peak unpadded heap bytes (field `18`).
    #[prost(int64, tag = "18")]
    pub peak_unpadded_heap_bytes: i64,
}

impl_proto_schema_message!(CompiledMemoryStatsProto, "xla.CompiledMemoryStatsProto");

/// Wire-compatible `xla.PjRtExecutableMetadata` (`xla/pjrt/proto/executable_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtExecutableMetadataProto {
    /// Compiled memory footprint statistics (field `1`).
    #[prost(message, optional, tag = "1")]
    pub compiled_memory_stats: Option<CompiledMemoryStatsProto>,
    /// Platform-specific metadata payload (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub platform_specific_metadata: Vec<u8>,
}

impl_proto_schema_message!(PjRtExecutableMetadataProto, "xla.PjRtExecutableMetadata");

/// Wire-compatible `xla.PjRtPartialProgramProto` (`xla/pjrt/proto/pjrt_partial_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtPartialProgramProto {
    /// Intermediate program payload bytes (field `1`).
    #[prost(bytes = "vec", tag = "1")]
    pub program: Vec<u8>,
    /// Program format identifier (e.g., `"mlir"`, `"hlo"`, `"llo"`, field `2`).
    #[prost(string, tag = "2")]
    pub program_format: String,
    /// Name of the compilation phase that produced this artifact (field `3`).
    #[prost(string, tag = "3")]
    pub producer_phase: String,
    /// Valid next consumer compilation phase names (field `4`).
    #[prost(string, repeated, tag = "4")]
    pub consumer_phases: Vec<String>,
    /// Schema/compiler version string (field `5`).
    #[prost(string, tag = "5")]
    pub version: String,
    /// Program or module name (field `6`).
    #[prost(string, tag = "6")]
    pub program_name: String,
}

impl_proto_schema_message!(PjRtPartialProgramProto, "xla.PjRtPartialProgramProto");

/// Wire-compatible `xla.PjRtTopologyDescriptionProto` (`xla/pjrt/proto/topology_description.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtTopologyDescriptionProto {
    /// Platform ID (field `1`).
    #[prost(uint64, tag = "1")]
    pub platform_id: u64,
    /// Platform name (e.g., `"tpu"`, field `2`).
    #[prost(string, tag = "2")]
    pub platform_name: String,
    /// Platform version string (field `3`).
    #[prost(string, tag = "3")]
    pub platform_version: String,
    /// Whether this topology represents a subslice (field `4`).
    #[prost(bool, tag = "4")]
    pub is_subslice_topology: bool,
    /// Serialized platform-specific topology proto (`TpuTopologyArgsProto`, field `9`).
    #[prost(bytes = "vec", tag = "9")]
    pub platform_specific_topology: Vec<u8>,
}

impl_proto_schema_message!(
    PjRtTopologyDescriptionProto,
    "xla.PjRtTopologyDescriptionProto"
);

/// Wire-compatible `xla.PjRtRuntimeAbiVersionProto` (`pjrt_abi_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RuntimeAbiVersionProto {
    /// Platform ID (`xla::PjRtRuntimeAbiVersionProto::platform`, field `1`).
    #[prost(uint64, tag = "1")]
    pub platform_id: u64,
    /// Runtime ABI version payload (`xla::PjRtRuntimeAbiVersionProto::version`, field `2`).
    #[prost(string, tag = "2")]
    pub version_string: String,
    /// PJRT API minor version (field `3`).
    #[prost(int32, tag = "3")]
    pub pjrt_api_minor: i32,
}

/// Type alias for [`RuntimeAbiVersionProto`] matching upstream `PjRtRuntimeAbiVersionProto`.
pub type PjRtRuntimeAbiVersionProto = RuntimeAbiVersionProto;

impl_proto_schema_message!(RuntimeAbiVersionProto, "xla.PjRtRuntimeAbiVersionProto");

/// Wire-compatible `xla.PjRtExecutableAbiVersionProto` (`pjrt_abi_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutableAbiVersionProto {
    /// Platform ID (`xla::PjRtExecutableAbiVersionProto::platform`, field `1`).
    #[prost(uint64, tag = "1")]
    pub platform_id: u64,
    /// Compiler ABI version payload (`xla::PjRtExecutableAbiVersionProto::version`, field `2`).
    #[prost(string, tag = "2")]
    pub compiler_version: String,
    /// Target TPU generation code (field `3`).
    #[prost(int32, tag = "3")]
    pub target_generation: i32,
}

/// Type alias for [`ExecutableAbiVersionProto`] matching upstream `PjRtExecutableAbiVersionProto`.
pub type PjRtExecutableAbiVersionProto = ExecutableAbiVersionProto;

impl_proto_schema_message!(
    ExecutableAbiVersionProto,
    "xla.PjRtExecutableAbiVersionProto"
);

/// Wire-compatible `xla.DeviceProto` (`xla/pjrt/distributed/protocol.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DistributedDeviceProto {
    /// Local device ordinal on the host (field `1`).
    #[prost(int32, tag = "1")]
    pub local_device_ordinal: i32,
    /// Device name string (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Vendor string (e.g., `"Google"`, field `3`).
    #[prost(string, tag = "3")]
    pub vendor: String,
    /// Globally unique device ID assigned by the coordinator (field `4`).
    #[prost(int32, tag = "4")]
    pub global_device_id: i32,
    /// Fast-interconnect partition index (field `5`).
    #[prost(int32, tag = "5")]
    pub partition_index: i32,
    /// Compute capability string (field `6`).
    #[prost(string, tag = "6")]
    pub compute_capability: String,
    /// Core count per device (field `7`).
    #[prost(int32, tag = "7")]
    pub core_count: i32,
    /// Device kind string (e.g., `"Cloud TPU v5e"`, field `8`).
    #[prost(string, tag = "8")]
    pub device_kind: String,
    /// Human-readable `to_string` representation (field `9`).
    #[prost(string, tag = "9")]
    pub to_string: String,
    /// Human-readable `debug_string` representation (field `10`).
    #[prost(string, tag = "10")]
    pub debug_string: String,
    /// Fabric UUID (`"slice_uuid/clique_id"`, field `12`).
    #[prost(string, tag = "12")]
    pub fabric_uuid: String,
    /// NUMA node index (`-1` if unpinned, field `14`).
    #[prost(int32, tag = "14")]
    pub numa_node: i32,
    /// Device HBM memory limit in bytes (field `15`).
    #[prost(int64, tag = "15")]
    pub device_memory_bytes_limit: i64,
}

impl_proto_schema_message!(DistributedDeviceProto, "xla.DeviceProto");

/// Wire-compatible `xla.LocalTopologyProto` (`xla/pjrt/distributed/protocol.proto`) published by each host.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LocalTopologyProto {
    /// Communicating process index in `[0, num_processes)` (field `1`).
    #[prost(int32, tag = "1")]
    pub process_id: i32,
    /// Kernel boot UUID (`/proc/sys/kernel/random/boot_id`, field `2`).
    #[prost(string, tag = "2")]
    pub boot_id: String,
    /// Local devices controlled by `process_id` (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub devices: Vec<DistributedDeviceProto>,
    /// Partition index (field `4`).
    #[prost(int32, tag = "4")]
    pub partition_index: i32,
    /// Ordered list of DCN switch hierarchy node IDs (field `5`).
    #[prost(string, repeated, tag = "5")]
    pub network_nodes: Vec<String>,
}

impl_proto_schema_message!(LocalTopologyProto, "xla.LocalTopologyProto");

/// Wire-compatible `xla.GlobalTopologyProto` (`xla/pjrt/distributed/protocol.proto`) broadcast by the coordinator.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GlobalTopologyProto {
    /// Per-process topologies with globally assigned `global_device_id` ordinals (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub processes: Vec<LocalTopologyProto>,
}

impl_proto_schema_message!(GlobalTopologyProto, "xla.GlobalTopologyProto");
