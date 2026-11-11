//! Megascale multi-slice coordination, DCN topology, transport, and runtime error overlays.
//!
//! Synchronized with `megascale_*.proto`, `status.proto`, and `runtime_config.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use prost::Message;
use std::collections::BTreeMap;

/// Wire-compatible `tensorflow.StatusProto` (`xla/tsl/protobuf/status.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StatusProto {
    /// Canonical `tensorflow.error.Code` status code (field `1`).
    #[prost(int32, tag = "1")]
    pub code: i32,
    /// Human-readable error message (field `2`).
    #[prost(string, tag = "2")]
    pub message: String,
    /// Error payload dictionary keyed by type URL (field `3`).
    #[prost(btree_map = "string, bytes", tag = "3")]
    pub payload: BTreeMap<String, Vec<u8>>,
}

impl_proto_schema_message!(StatusProto, "tensorflow.StatusProto");

/// Wire-compatible `megascale.SliceDeviceListProto` (`xla/megascale/megascale.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SliceDeviceListProto {
    /// Zero-based slice index (field `1`).
    #[prost(int32, tag = "1")]
    pub slice_id: i32,
    /// Global device IDs belonging to this slice (field `2`).
    #[prost(int32, repeated, packed = "true", tag = "2")]
    pub device_ids: Vec<i32>,
}

impl_proto_schema_message!(SliceDeviceListProto, "megascale.SliceDeviceListProto");

/// Wire-compatible `xla.megascale.runtime.DCNTopology.SymmetricTree` (`megascale_dcn_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DcnSymmetricTreeProto {
    /// Branching factor at each hierarchy layer from top to bottom (field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub branching_per_layer: Vec<i32>,
}

impl_proto_schema_message!(
    DcnSymmetricTreeProto,
    "xla.megascale.runtime.DCNTopology.SymmetricTree"
);

/// Wire-compatible `xla.megascale.runtime.DCNTopology.TreeNode.SliceRange` (`megascale_dcn_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DcnSliceRangeProto {
    /// Inclusive starting slice ID (field `1`).
    #[prost(int32, tag = "1")]
    pub slice_id_start: i32,
    /// Exclusive ending slice ID (field `2`).
    #[prost(int32, tag = "2")]
    pub slice_id_end: i32,
}

impl_proto_schema_message!(
    DcnSliceRangeProto,
    "xla.megascale.runtime.DCNTopology.TreeNode.SliceRange"
);

/// Wire-compatible `xla.megascale.runtime.DCNTopology.TreeNode` (`megascale_dcn_topology.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct DcnTreeNodeProto {
    /// Optional human-readable switch/rack label (field `1`).
    #[prost(string, optional, tag = "1")]
    pub label: Option<String>,
    /// Child tree nodes (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub nodes: Vec<Self>,
    /// Leaf slice ranges attached to this node (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub slice_ranges: Vec<DcnSliceRangeProto>,
    /// Optional egress constraint factor in `[0.0, 1.0]` (field `4`).
    #[prost(float, optional, tag = "4")]
    pub egress_constraint: Option<f32>,
    /// Whether children execute ring transfers instead of shuffle transfers (field `5`).
    #[prost(bool, tag = "5")]
    pub ring_transfers: bool,
}

impl_proto_schema_message!(
    DcnTreeNodeProto,
    "xla.megascale.runtime.DCNTopology.TreeNode"
);

/// Wire-compatible `xla.megascale.runtime.DCNTopology` (`megascale_dcn_topology.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct DcnTopologyTreeProto {
    /// Symmetric hierarchical tree representation (field `1`).
    #[prost(message, optional, tag = "1")]
    pub symmetric_tree: Option<DcnSymmetricTreeProto>,
    /// General asymmetric tree representation (field `2`).
    #[prost(message, optional, tag = "2")]
    pub tree: Option<DcnTreeNodeProto>,
}

impl_proto_schema_message!(DcnTopologyTreeProto, "xla.megascale.runtime.DCNTopology");

/// Wire-compatible `megascale.MultiSliceConfigProto` (`xla/megascale/megascale.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct MegascaleMultiSliceConfigProto {
    /// Total number of slices in the Megascale pod cluster (field `1`).
    #[prost(int32, tag = "1")]
    pub num_slices: i32,
    /// Zero-based index of the current slice (field `2`).
    #[prost(int32, tag = "2")]
    pub slice_id: i32,
    /// Number of devices per slice (field `3`).
    #[prost(int32, tag = "3")]
    pub num_devices_per_slice: i32,
    /// DCN coordinator endpoint address (field `4`).
    #[prost(string, tag = "4")]
    pub coordinator_address: String,
    /// Per-slice device ID assignments (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub slice_devices: Vec<SliceDeviceListProto>,
    /// Endpoint network address table across all slices and hosts (field `6`).
    #[prost(message, optional, tag = "6")]
    pub endpoint_addresses: Option<EndpointAddressesProto>,
    /// Hierarchical DCN network topology tree (field `7`).
    #[prost(message, optional, tag = "7")]
    pub dcn_topology: Option<DcnTopologyTreeProto>,
}

impl_proto_schema_message!(
    MegascaleMultiSliceConfigProto,
    "megascale.MultiSliceConfigProto"
);

/// Wire-compatible `megascale.DCNTopologyProto` (`xla/megascale/dcn_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleDcnTopologyProto {
    /// Number of DCN slices (field `1`).
    #[prost(int32, tag = "1")]
    pub num_slices: i32,
    /// Number of hosts per slice (field `2`).
    #[prost(int32, tag = "2")]
    pub num_hosts_per_slice: i32,
    /// Inter-slice transport protocol (`1` = TCP, `2` = RoCEv2/Falcon, field `3`).
    #[prost(int32, tag = "3")]
    pub transport_type: i32,
}

impl_proto_schema_message!(MegascaleDcnTopologyProto, "megascale.DCNTopologyProto");

/// Wire-compatible `megascale.RuntimeMetricsProto` (`xla/megascale/megascale.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleRuntimeMetricsProto {
    /// Number of completed cross-slice barriers (field `1`).
    #[prost(uint64, tag = "1")]
    pub completed_barriers: u64,
    /// Total collective operations executed (field `2`).
    #[prost(uint64, tag = "2")]
    pub collective_ops_count: u64,
    /// Active Megascale client contexts (field `3`).
    #[prost(int32, tag = "3")]
    pub active_contexts: i32,
    /// Total DCN bytes transferred across slices (field `4`).
    #[prost(uint64, tag = "4")]
    pub bytes_transferred: u64,
}

impl_proto_schema_message!(
    MegascaleRuntimeMetricsProto,
    "megascale.RuntimeMetricsProto"
);

/// Wire-compatible `xla.megascale.runtime.HostNetworkAddress` (`xla/megascale/addresses.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HostNetworkAddressProto {
    /// Network socket address (`"ip:port"`, field `1`).
    #[prost(string, tag = "1")]
    pub address: String,
    /// Network interface name (e.g., `"eth0"`, field `2`).
    #[prost(string, tag = "2")]
    pub interface_name: String,
    /// Debugging hostname (field `3`).
    #[prost(string, tag = "3")]
    pub host_name_for_debugging: String,
    /// NUMA node affinity (field `4`).
    #[prost(int32, tag = "4")]
    pub numa_node: i32,
}

impl_proto_schema_message!(
    HostNetworkAddressProto,
    "xla.megascale.runtime.HostNetworkAddress"
);

/// Wire-compatible `xla.megascale.runtime.NetworkAddressMapping` (`xla/megascale/addresses.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct NetworkAddressMappingProto {
    /// Slice index (field `1`).
    #[prost(int32, tag = "1")]
    pub slice_id: i32,
    /// Host index within the slice (field `2`).
    #[prost(int32, tag = "2")]
    pub host_id: i32,
    /// Bound host network addresses (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub addresses: Vec<HostNetworkAddressProto>,
    /// Optional incarnation ID (field `4`).
    #[prost(uint64, optional, tag = "4")]
    pub incarnation_id: Option<u64>,
}

impl_proto_schema_message!(
    NetworkAddressMappingProto,
    "xla.megascale.runtime.NetworkAddressMapping"
);

/// Wire-compatible `xla.megascale.runtime.EndpointAddresses` (`xla/megascale/addresses.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EndpointAddressesProto {
    /// Address mappings across all `(slice_id, host_id)` endpoints (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub address_mappings: Vec<NetworkAddressMappingProto>,
}

impl_proto_schema_message!(
    EndpointAddressesProto,
    "xla.megascale.runtime.EndpointAddresses"
);

/// Wire-compatible `xla.megascale.runtime.MegascaleTransportConfigProto` (`runtime_config.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleTransportConfigProto {
    /// Number of parallel DCN channels (field `1`).
    #[prost(int32, tag = "1")]
    pub num_channels: i32,
    /// Maximum in-flight byte budget (field `2`).
    #[prost(int64, tag = "2")]
    pub max_inflight_bytes: i64,
    /// Operation timeout in milliseconds (field `3`).
    #[prost(int64, tag = "3")]
    pub timeout_ms: i64,
    /// Whether `zstd` payload compression is enabled (field `4`).
    #[prost(bool, tag = "4")]
    pub enable_zstd_compression: bool,
    /// `zstd` compression level (field `5`).
    #[prost(int32, tag = "5")]
    pub zstd_compression_level: i32,
}

impl_proto_schema_message!(
    MegascaleTransportConfigProto,
    "xla.megascale.runtime.MegascaleTransportConfigProto"
);

/// Wire-compatible `xla.megascale.runtime.external.MegaScaleAction.Input`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegaScaleActionInputProto {
    /// Producer action index in `MegaScaleActionGraph.actions` (field `1`).
    #[prost(int32, tag = "1")]
    pub action_index: i32,
}

impl_proto_schema_message!(
    MegaScaleActionInputProto,
    "xla.megascale.runtime.external.MegaScaleAction.Input"
);

/// Wire-compatible `xla.megascale.runtime.external.MegaScaleAction`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegaScaleActionProto {
    /// Action input dependencies (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub inputs: Vec<MegaScaleActionInputProto>,
    /// Action index within the graph (field `8`).
    #[prost(int32, tag = "8")]
    pub index: i32,
}

impl_proto_schema_message!(
    MegaScaleActionProto,
    "xla.megascale.runtime.external.MegaScaleAction"
);

/// Wire-compatible `xla.megascale.runtime.external.MegaScaleActionGraph`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegaScaleActionGraphProto {
    /// Unique identifier for the action graph (field `1`).
    #[prost(string, tag = "1")]
    pub id: String,
    /// Actions belonging to the graph (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub actions: Vec<MegaScaleActionProto>,
    /// Triggering HLO module name (field `7`).
    #[prost(string, tag = "7")]
    pub hlo_module: String,
}

impl_proto_schema_message!(
    MegaScaleActionGraphProto,
    "xla.megascale.runtime.external.MegaScaleActionGraph"
);

/// Wire-compatible `xla.megascale.runtime.external.ExecutorTelemetry.PendingGraphState`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PendingGraphStateProto {
    /// Action graph descriptor (field `1`).
    #[prost(message, optional, tag = "1")]
    pub graph: Option<MegaScaleActionGraphProto>,
    /// Launch identifier (field `2`).
    #[prost(int32, tag = "2")]
    pub launch_id: i32,
    /// Completed action indices (field `3`).
    #[prost(int32, repeated, packed = "true", tag = "3")]
    pub completed_actions: Vec<i32>,
    /// Pending action indices (field `4`).
    #[prost(int32, repeated, packed = "true", tag = "4")]
    pub pending_actions: Vec<i32>,
    /// Active action indices (field `6`).
    #[prost(int32, repeated, packed = "true", tag = "6")]
    pub active_actions: Vec<i32>,
}

impl_proto_schema_message!(
    PendingGraphStateProto,
    "xla.megascale.runtime.external.ExecutorTelemetry.PendingGraphState"
);

/// Wire-compatible `xla.megascale.runtime.external.ExecutorTelemetry`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutorTelemetryProto {
    /// Pending graph states on the executor (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub pending_graphs: Vec<PendingGraphStateProto>,
}

impl_proto_schema_message!(
    ExecutorTelemetryProto,
    "xla.megascale.runtime.external.ExecutorTelemetry"
);

/// Wire-compatible `xla.megascale.runtime.external.ProgramRunInfo`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProgramRunInfoProto {
    /// Core-local run identifier (field `1`).
    #[prost(uint64, tag = "1")]
    pub run_id: u64,
    /// Cross-slice launch identifier (field `2`).
    #[prost(int32, tag = "2")]
    pub launch_id: i32,
    /// HLO module/program name (field `3`).
    #[prost(string, tag = "3")]
    pub name: String,
    /// Executable fingerprint hex string (field `4`).
    #[prost(string, tag = "4")]
    pub fingerprint: String,
}

impl_proto_schema_message!(
    ProgramRunInfoProto,
    "xla.megascale.runtime.external.ProgramRunInfo"
);

/// Wire-compatible `xla.megascale.runtime.external.TpuCoreHLO`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCoreHloProto {
    /// Slice-local chip ID (field `1`).
    #[prost(int32, tag = "1")]
    pub chip_id: i32,
    /// Core index within the chip (field `2`).
    #[prost(int32, tag = "2")]
    pub core_idx: i32,
    /// Full HLO instruction text (field `3`).
    #[prost(string, tag = "3")]
    pub hlo: String,
    /// Running executable fingerprint (field `4`).
    #[prost(string, tag = "4")]
    pub executable_fingerprint: String,
    /// Current hardware program counter (field `6`).
    #[prost(int64, optional, tag = "6")]
    pub pc: Option<i64>,
    /// PC tag annotation (field `7`).
    #[prost(int64, optional, tag = "7")]
    pub tag: Option<i64>,
    /// Worker hostname (field `8`).
    #[prost(string, tag = "8")]
    pub hostname: String,
    /// Short HLO instruction name (field `9`).
    #[prost(string, tag = "9")]
    pub hlo_name: String,
    /// Running HLO module name (field `10`).
    #[prost(string, tag = "10")]
    pub module_name: String,
    /// Running launch ID (field `12`).
    #[prost(int32, tag = "12")]
    pub launch_id: i32,
    /// TPU core type enum (`TpuCoreTypeProto`, field `13`).
    #[prost(int32, tag = "13")]
    pub tpu_core_type: i32,
    /// Short computation name (field `15`).
    #[prost(string, tag = "15")]
    pub computation_name: String,
    /// Recently completed program runs (field `16`).
    #[prost(message, repeated, tag = "16")]
    pub recent_completions: Vec<ProgramRunInfoProto>,
    /// Whether the TPU core is actively executing (field `18`).
    #[prost(bool, optional, tag = "18")]
    pub active: Option<bool>,
    /// Errors encountered while sampling core state (field `19`).
    #[prost(message, repeated, tag = "19")]
    pub errors: Vec<StatusProto>,
}

impl_proto_schema_message!(TpuCoreHloProto, "xla.megascale.runtime.external.TpuCoreHLO");

/// Wire-compatible `xla.megascale.runtime.external.RapidEyeInfo`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RapidEyeInfoProto {
    /// Executor telemetry snapshot (field `1`).
    #[prost(message, optional, tag = "1")]
    pub telemetry: Option<ExecutorTelemetryProto>,
    /// Per-core HLO execution snapshots (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub tpu_core_hlos: Vec<TpuCoreHloProto>,
    /// Whether the snapshot was truncated due to size limits (field `3`).
    #[prost(bool, tag = "3")]
    pub is_truncated: bool,
    /// Number of slices in the workload topology (field `4`).
    #[prost(int64, tag = "4")]
    pub num_slices: i64,
    /// Number of tasks per slice (field `5`).
    #[prost(int64, tag = "5")]
    pub num_tasks_per_slice: i64,
    /// Local slice ID (field `6`).
    #[prost(int32, tag = "6")]
    pub slice_id: i32,
    /// Local task ID (field `7`).
    #[prost(int32, tag = "7")]
    pub task_id: i32,
    /// Shared workload incarnation ID (field `8`).
    #[prost(int64, tag = "8")]
    pub incarnation_id: i64,
    /// TPU platform version code (field `9`).
    #[prost(int32, tag = "9")]
    pub tpu_version: i32,
    /// Chip configuration name (field `10`).
    #[prost(string, tag = "10")]
    pub chip_config_name: String,
}

impl_proto_schema_message!(
    RapidEyeInfoProto,
    "xla.megascale.runtime.external.RapidEyeInfo"
);

/// Wire-compatible `xla.megascale.runtime.external.RuntimeStateSummary`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RuntimeStateSummaryProto {
    /// `RapidEye` hardware and executor telemetry summary (field `4`).
    #[prost(message, optional, tag = "4")]
    pub rapid_eye_info: Option<RapidEyeInfoProto>,
}

impl_proto_schema_message!(
    RuntimeStateSummaryProto,
    "xla.megascale.runtime.external.RuntimeStateSummary"
);

/// Wire-compatible `xla.megascale.runtime.external.MegaScaleRuntimeError`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegaScaleRuntimeErrorProto {
    /// Error type (`0` = `NO_ERROR`, `1` = `HANG_DETECTED`, `2` = `UNRECOVERABLE_ERROR`, `3` = `CANCELLED`, field `1`).
    #[prost(int32, tag = "1")]
    pub error_type: i32,
    /// Human-readable error message (field `2`).
    #[prost(string, tag = "2")]
    pub error_message: String,
    /// Estimated start time in milliseconds since epoch (field `3`).
    #[prost(uint64, tag = "3")]
    pub estimated_start_time_ms: u64,
    /// Runtime state summary captured at error time (field `4`).
    #[prost(message, optional, tag = "4")]
    pub runtime_state: Option<RuntimeStateSummaryProto>,
    /// Triggering launch ID (field `5`).
    #[prost(int32, tag = "5")]
    pub triggering_launch_id: i32,
    /// Reporting hostname (field `6`).
    #[prost(string, tag = "6")]
    pub hostname: String,
    /// Reporting job name (field `8`).
    #[prost(string, tag = "8")]
    pub job_name: String,
    /// Task ordinal index (field `9`).
    #[prost(int32, tag = "9")]
    pub task_id: i32,
    /// Event identifier (field `11`).
    #[prost(fixed64, tag = "11")]
    pub event_id: u64,
    /// Unrecoverable error sub-type (`0` = Unclassified, `1` = H2D, `2` = D2H, field `12`).
    #[prost(int32, tag = "12")]
    pub unrecoverable_error_type: i32,
    /// Runtime error identifier (field `13`).
    #[prost(fixed64, tag = "13")]
    pub runtime_error_id: u64,
    /// Hang error sub-type (`0` = Unspecified, `1` = Timeout, `2` = `NoProgress`, field `14`).
    #[prost(int32, tag = "14")]
    pub hang_error_type: i32,
}

impl_proto_schema_message!(
    MegaScaleRuntimeErrorProto,
    "xla.megascale.runtime.external.MegaScaleRuntimeError"
);

/// Wire-compatible `tensorflow.CoordinatedJob` (`xla/tsl/protobuf/coordination_config.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoordinatedJobProto {
    /// Job name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Total number of tasks in the job (field `2`).
    #[prost(int32, tag = "2")]
    pub num_tasks: i32,
}

impl_proto_schema_message!(CoordinatedJobProto, "tensorflow.CoordinatedJob");

/// Wire-compatible `tensorflow.CoordinationServiceConfig` (`xla/tsl/protobuf/coordination_config.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoordinationServiceConfigProto {
    /// Service implementation type (e.g., `"standalone"`, field `1`).
    #[prost(string, tag = "1")]
    pub service_type: String,
    /// Leader task name or address (field `2`).
    #[prost(string, tag = "2")]
    pub service_leader: String,
    /// Whether to enable the health check mechanism (field `3`).
    #[prost(bool, tag = "3")]
    pub enable_health_check: bool,
    /// Maximum wait time for all cluster members to register in milliseconds (field `4`).
    #[prost(int64, tag = "4")]
    pub cluster_register_timeout_in_ms: i64,
    /// Heartbeat timeout in milliseconds (field `5`).
    #[prost(int64, tag = "5")]
    pub heartbeat_timeout_in_ms: i64,
    /// Shutdown barrier timeout in milliseconds (field `7`).
    #[prost(int64, tag = "7")]
    pub shutdown_barrier_timeout_in_ms: i64,
    /// Whether agent destruction without shutdown is an error (field `8`).
    #[prost(bool, tag = "8")]
    pub agent_destruction_without_shutdown: bool,
    /// Recoverable job names (field `9`).
    #[prost(string, repeated, tag = "9")]
    pub recoverable_jobs: Vec<String>,
    /// Coordinated job specifications in the cluster (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub coordinated_job_list: Vec<CoordinatedJobProto>,
    /// Whether to allow new incarnation reconnects (field `11`).
    #[prost(bool, tag = "11")]
    pub allow_new_incarnation_to_reconnect: bool,
    /// Force disable coordination service (field `12`).
    #[prost(bool, tag = "12")]
    pub force_disable: bool,
    /// Poll for error from service at startup (field `13`).
    #[prost(bool, tag = "13")]
    pub poll_for_error_from_service_at_startup: bool,
    /// Synchronize agent registration on a barrier (field `14`).
    #[prost(bool, tag = "14")]
    pub cluster_register_with_barrier: bool,
}

impl_proto_schema_message!(
    CoordinationServiceConfigProto,
    "tensorflow.CoordinationServiceConfig"
);

/// Wire-compatible `tensorflow.CoordinatedTask` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoordinatedTaskProto {
    /// Job name (field `1`).
    #[prost(string, tag = "1")]
    pub job_name: String,
    /// Task index within the job (field `2`).
    #[prost(int32, tag = "2")]
    pub task_id: i32,
    /// Whether the task is recoverable upon restart (field `3`).
    #[prost(bool, tag = "3")]
    pub recoverable: bool,
}

impl_proto_schema_message!(CoordinatedTaskProto, "tensorflow.CoordinatedTask");

/// Wire-compatible `tensorflow.CoordinationServiceError` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoordinationServiceErrorProto {
    /// Whether the error was reported via agent API by the user (field `3`).
    #[prost(bool, tag = "3")]
    pub is_reported_error: bool,
    /// Task that originated the error (field `4`).
    #[prost(message, optional, tag = "4")]
    pub source_task: Option<CoordinatedTaskProto>,
}

impl_proto_schema_message!(
    CoordinationServiceErrorProto,
    "tensorflow.CoordinationServiceError"
);

/// Wire-compatible `tensorflow.BarrierError` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BarrierErrorProto {
    /// Barrier identifier (field `1`).
    #[prost(string, tag = "1")]
    pub barrier_id: String,
    /// Invocation counter for reused barrier IDs (field `2`).
    #[prost(int64, tag = "2")]
    pub counter: i64,
}

impl_proto_schema_message!(BarrierErrorProto, "tensorflow.BarrierError");

/// Wire-compatible `tensorflow.CoordinatedTaskStateInfo` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoordinatedTaskStateInfoProto {
    /// Task identifier (field `1`).
    #[prost(message, optional, tag = "1")]
    pub task: Option<CoordinatedTaskProto>,
    /// Task state enum (`CoordinatedTaskState`, field `2`).
    #[prost(int32, tag = "2")]
    pub state: i32,
    /// Error status code (`0` = OK, field `3`).
    #[prost(int32, tag = "3")]
    pub error_code: i32,
    /// Error message string (field `4`).
    #[prost(string, tag = "4")]
    pub error_message: String,
    /// Detailed coordination error payload (field `5`).
    #[prost(message, optional, tag = "5")]
    pub error_payload: Option<CoordinationServiceErrorProto>,
    /// Task incarnation ID (field `6`).
    #[prost(fixed64, tag = "6")]
    pub incarnation: u64,
}

impl_proto_schema_message!(
    CoordinatedTaskStateInfoProto,
    "tensorflow.CoordinatedTaskStateInfo"
);

/// Wire-compatible `tensorflow.DeviceInfo` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoordinationDeviceInfoProto {
    /// Device attributes packed as `google.protobuf.Any` (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub device: Vec<prost_types::Any>,
}

impl_proto_schema_message!(CoordinationDeviceInfoProto, "tensorflow.DeviceInfo");

/// Wire-compatible `tensorflow.RegisterTaskRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RegisterTaskRequestProto {
    /// Task incarnation ID (field `3`).
    #[prost(fixed64, tag = "3")]
    pub incarnation: u64,
    /// Requesting task identifier (field `5`).
    #[prost(message, optional, tag = "5")]
    pub source_task: Option<CoordinatedTaskProto>,
}

impl_proto_schema_message!(RegisterTaskRequestProto, "tensorflow.RegisterTaskRequest");

/// Wire-compatible `tensorflow.RegisterTaskResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RegisterTaskResponseProto {
    /// Leader incarnation ID (field `1`).
    #[prost(fixed64, tag = "1")]
    pub leader_incarnation: u64,
}

impl_proto_schema_message!(RegisterTaskResponseProto, "tensorflow.RegisterTaskResponse");

/// Wire-compatible `tensorflow.HeartbeatRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HeartbeatRequestProto {
    /// Task incarnation ID (field `3`).
    #[prost(fixed64, tag = "3")]
    pub incarnation: u64,
    /// Requesting task identifier (field `4`).
    #[prost(message, optional, tag = "4")]
    pub source_task: Option<CoordinatedTaskProto>,
}

impl_proto_schema_message!(HeartbeatRequestProto, "tensorflow.HeartbeatRequest");

/// Wire-compatible `tensorflow.HeartbeatResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HeartbeatResponseProto {
    /// Leader incarnation ID (field `1`).
    #[prost(fixed64, tag = "1")]
    pub leader_incarnation: u64,
}

impl_proto_schema_message!(HeartbeatResponseProto, "tensorflow.HeartbeatResponse");

/// Wire-compatible `tensorflow.PollForErrorRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PollForErrorRequestProto {
    /// Requesting task identifier (field `1`).
    #[prost(message, optional, tag = "1")]
    pub source_task: Option<CoordinatedTaskProto>,
}

impl_proto_schema_message!(PollForErrorRequestProto, "tensorflow.PollForErrorRequest");

/// Wire-compatible `tensorflow.PollForErrorResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PollForErrorResponseProto {}

impl_proto_schema_message!(PollForErrorResponseProto, "tensorflow.PollForErrorResponse");

/// Wire-compatible `tensorflow.WaitForAllTasksRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct WaitForAllTasksRequestProto {
    /// Requesting task identifier (field `5`).
    #[prost(message, optional, tag = "5")]
    pub source_task: Option<CoordinatedTaskProto>,
    /// Local device info published by the sender (field `6`).
    #[prost(message, optional, tag = "6")]
    pub device_info: Option<CoordinationDeviceInfoProto>,
}

impl_proto_schema_message!(
    WaitForAllTasksRequestProto,
    "tensorflow.WaitForAllTasksRequest"
);

/// Wire-compatible `tensorflow.WaitForAllTasksResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct WaitForAllTasksResponseProto {
    /// Leader incarnation ID (field `1`).
    #[prost(fixed64, tag = "1")]
    pub leader_incarnation: u64,
    /// Aggregated cluster device info (field `4`).
    #[prost(message, optional, tag = "4")]
    pub device_info: Option<CoordinationDeviceInfoProto>,
}

impl_proto_schema_message!(
    WaitForAllTasksResponseProto,
    "tensorflow.WaitForAllTasksResponse"
);

/// Wire-compatible `tensorflow.ShutdownTaskRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShutdownTaskRequestProto {
    /// Disconnecting task identifier (field `1`).
    #[prost(message, optional, tag = "1")]
    pub source_task: Option<CoordinatedTaskProto>,
}

impl_proto_schema_message!(ShutdownTaskRequestProto, "tensorflow.ShutdownTaskRequest");

/// Wire-compatible `tensorflow.ShutdownTaskResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShutdownTaskResponseProto {}

impl_proto_schema_message!(ShutdownTaskResponseProto, "tensorflow.ShutdownTaskResponse");

/// Wire-compatible `tensorflow.ResetTaskRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ResetTaskRequestProto {
    /// Task identifier to reset (field `1`).
    #[prost(message, optional, tag = "1")]
    pub source_task: Option<CoordinatedTaskProto>,
}

impl_proto_schema_message!(ResetTaskRequestProto, "tensorflow.ResetTaskRequest");

/// Wire-compatible `tensorflow.ResetTaskResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ResetTaskResponseProto {}

impl_proto_schema_message!(ResetTaskResponseProto, "tensorflow.ResetTaskResponse");

/// Wire-compatible `tensorflow.ReportErrorToTaskRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ReportErrorToTaskRequestProto {
    /// Error status code (field `1`).
    #[prost(int32, tag = "1")]
    pub error_code: i32,
    /// Error message string (field `2`).
    #[prost(string, tag = "2")]
    pub error_message: String,
    /// Detailed error payload (field `5`).
    #[prost(message, optional, tag = "5")]
    pub error_payload: Option<CoordinationServiceErrorProto>,
}

impl_proto_schema_message!(
    ReportErrorToTaskRequestProto,
    "tensorflow.ReportErrorToTaskRequest"
);

/// Wire-compatible `tensorflow.ReportErrorToTaskResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ReportErrorToTaskResponseProto {}

impl_proto_schema_message!(
    ReportErrorToTaskResponseProto,
    "tensorflow.ReportErrorToTaskResponse"
);

/// Wire-compatible `tensorflow.ReportErrorToServiceRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ReportErrorToServiceRequestProto {
    /// Error status code (field `1`).
    #[prost(int32, tag = "1")]
    pub error_code: i32,
    /// Error message string (field `2`).
    #[prost(string, tag = "2")]
    pub error_message: String,
    /// Task that originated the error (field `5`).
    #[prost(message, optional, tag = "5")]
    pub error_origin: Option<CoordinatedTaskProto>,
}

impl_proto_schema_message!(
    ReportErrorToServiceRequestProto,
    "tensorflow.ReportErrorToServiceRequest"
);

/// Wire-compatible `tensorflow.ReportErrorToServiceResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ReportErrorToServiceResponseProto {}

impl_proto_schema_message!(
    ReportErrorToServiceResponseProto,
    "tensorflow.ReportErrorToServiceResponse"
);

/// Wire-compatible `tensorflow.GetTaskStateRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetTaskStateRequestProto {
    /// Tasks whose state is queried (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub source_task: Vec<CoordinatedTaskProto>,
}

impl_proto_schema_message!(GetTaskStateRequestProto, "tensorflow.GetTaskStateRequest");

/// Wire-compatible `tensorflow.GetTaskStateResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetTaskStateResponseProto {
    /// Per-task state descriptors (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub task_state: Vec<CoordinatedTaskStateInfoProto>,
}

impl_proto_schema_message!(GetTaskStateResponseProto, "tensorflow.GetTaskStateResponse");

/// Wire-compatible `tensorflow.WatchJobStateRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct WatchJobStateRequestProto {
    /// Job name to watch (field `1`).
    #[prost(string, tag = "1")]
    pub job_name: String,
    /// Last seen state version number (`-1` for immediate fetch, field `2`).
    #[prost(int64, tag = "2")]
    pub version_number: i64,
}

impl_proto_schema_message!(WatchJobStateRequestProto, "tensorflow.WatchJobStateRequest");

/// Wire-compatible `tensorflow.WatchJobStateResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct WatchJobStateResponseProto {
    /// Task states for the watched job (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub task_state: Vec<CoordinatedTaskStateInfoProto>,
    /// Monotonically increasing job state version number (field `2`).
    #[prost(int64, tag = "2")]
    pub version_number: i64,
}

impl_proto_schema_message!(
    WatchJobStateResponseProto,
    "tensorflow.WatchJobStateResponse"
);

/// Wire-compatible `tensorflow.KeyValueEntry` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct KeyValueEntryProto {
    /// Hierarchical key path (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Raw value payload bytes (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub value: Vec<u8>,
}

impl_proto_schema_message!(KeyValueEntryProto, "tensorflow.KeyValueEntry");

/// Wire-compatible `tensorflow.InsertKeyValueRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct InsertKeyValueRequestProto {
    /// Key-value entry to insert (field `1`).
    #[prost(message, optional, tag = "1")]
    pub kv: Option<KeyValueEntryProto>,
    /// Whether existing values may be overwritten (field `2`).
    #[prost(bool, tag = "2")]
    pub allow_overwrite: bool,
}

impl_proto_schema_message!(
    InsertKeyValueRequestProto,
    "tensorflow.InsertKeyValueRequest"
);

/// Wire-compatible `tensorflow.InsertKeyValueResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct InsertKeyValueResponseProto {}

impl_proto_schema_message!(
    InsertKeyValueResponseProto,
    "tensorflow.InsertKeyValueResponse"
);

/// Wire-compatible `tensorflow.GetKeyValueRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetKeyValueRequestProto {
    /// Key to look up (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
}

impl_proto_schema_message!(GetKeyValueRequestProto, "tensorflow.GetKeyValueRequest");

/// Wire-compatible `tensorflow.GetKeyValueResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetKeyValueResponseProto {
    /// Retrieved key-value entry (field `1`).
    #[prost(message, optional, tag = "1")]
    pub kv: Option<KeyValueEntryProto>,
}

impl_proto_schema_message!(GetKeyValueResponseProto, "tensorflow.GetKeyValueResponse");

/// Wire-compatible `tensorflow.TryGetKeyValueRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TryGetKeyValueRequestProto {
    /// Key to look up non-blockingly (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
}

impl_proto_schema_message!(
    TryGetKeyValueRequestProto,
    "tensorflow.TryGetKeyValueRequest"
);

/// Wire-compatible `tensorflow.TryGetKeyValueResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TryGetKeyValueResponseProto {
    /// Retrieved key-value entry if present (field `1`).
    #[prost(message, optional, tag = "1")]
    pub kv: Option<KeyValueEntryProto>,
}

impl_proto_schema_message!(
    TryGetKeyValueResponseProto,
    "tensorflow.TryGetKeyValueResponse"
);

/// Wire-compatible `tensorflow.IncrementKeyValueRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IncrementKeyValueRequestProto {
    /// Key to increment atomically (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Signed increment delta (field `2`).
    #[prost(int64, tag = "2")]
    pub increment: i64,
}

impl_proto_schema_message!(
    IncrementKeyValueRequestProto,
    "tensorflow.IncrementKeyValueRequest"
);

/// Wire-compatible `tensorflow.IncrementKeyValueResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IncrementKeyValueResponseProto {
    /// Updated key-value entry after increment (field `1`).
    #[prost(message, optional, tag = "1")]
    pub kv: Option<KeyValueEntryProto>,
}

impl_proto_schema_message!(
    IncrementKeyValueResponseProto,
    "tensorflow.IncrementKeyValueResponse"
);

/// Wire-compatible `tensorflow.GetKeyValueDirRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetKeyValueDirRequestProto {
    /// Directory key prefix (field `1`).
    #[prost(string, tag = "1")]
    pub directory_key: String,
}

impl_proto_schema_message!(
    GetKeyValueDirRequestProto,
    "tensorflow.GetKeyValueDirRequest"
);

/// Wire-compatible `tensorflow.GetKeyValueDirResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetKeyValueDirResponseProto {
    /// Queried directory key prefix (field `1`).
    #[prost(string, tag = "1")]
    pub directory_key: String,
    /// All key-value entries matching the directory prefix (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub kv: Vec<KeyValueEntryProto>,
}

impl_proto_schema_message!(
    GetKeyValueDirResponseProto,
    "tensorflow.GetKeyValueDirResponse"
);

/// Wire-compatible `tensorflow.DeleteKeyValueRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeleteKeyValueRequestProto {
    /// Key or directory prefix to delete (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Whether to recursively delete all entries under `key` (field `2`).
    #[prost(bool, tag = "2")]
    pub is_directory: bool,
}

impl_proto_schema_message!(
    DeleteKeyValueRequestProto,
    "tensorflow.DeleteKeyValueRequest"
);

/// Wire-compatible `tensorflow.DeleteKeyValueResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeleteKeyValueResponseProto {}

impl_proto_schema_message!(
    DeleteKeyValueResponseProto,
    "tensorflow.DeleteKeyValueResponse"
);

/// Wire-compatible `tensorflow.BarrierRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BarrierRequestProto {
    /// Unique barrier identifier (field `1`).
    #[prost(string, tag = "1")]
    pub barrier_id: String,
    /// Barrier timeout in milliseconds (field `2`).
    #[prost(int64, tag = "2")]
    pub barrier_timeout_in_ms: i64,
    /// Participating tasks (empty implies all cluster tasks, field `3`).
    #[prost(message, repeated, tag = "3")]
    pub tasks: Vec<CoordinatedTaskProto>,
    /// Task making the barrier request (field `4`).
    #[prost(message, optional, tag = "4")]
    pub source_task: Option<CoordinatedTaskProto>,
    /// Invocation counter for reused barrier IDs (field `5`).
    #[prost(int64, tag = "5")]
    pub counter: i64,
}

impl_proto_schema_message!(BarrierRequestProto, "tensorflow.BarrierRequest");

/// Wire-compatible `tensorflow.BarrierResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BarrierResponseProto {
    /// Acknowledged service-side barrier counter (field `1`).
    #[prost(int64, tag = "1")]
    pub counter: i64,
}

impl_proto_schema_message!(BarrierResponseProto, "tensorflow.BarrierResponse");

/// Wire-compatible `tensorflow.GetAliveTasksRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetAliveTasksRequestProto {
    /// Task making the request (field `1`).
    #[prost(message, optional, tag = "1")]
    pub requesting_task: Option<CoordinatedTaskProto>,
    /// Candidate tasks to check for aliveness (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub tasks: Vec<CoordinatedTaskProto>,
}

impl_proto_schema_message!(GetAliveTasksRequestProto, "tensorflow.GetAliveTasksRequest");

/// Wire-compatible `tensorflow.GetAliveTasksResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetAliveTasksResponseProto {
    /// Subset of tasks currently alive (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub alive_tasks: Vec<CoordinatedTaskProto>,
    /// Incarnation IDs corresponding to `alive_tasks` (field `2`).
    #[prost(fixed64, repeated, packed = "true", tag = "2")]
    pub incarnations: Vec<u64>,
}

impl_proto_schema_message!(
    GetAliveTasksResponseProto,
    "tensorflow.GetAliveTasksResponse"
);

/// Wire-compatible `tensorflow.CancelBarrierRequest` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CancelBarrierRequestProto {
    /// Barrier identifier to cancel (field `1`).
    #[prost(string, tag = "1")]
    pub barrier_id: String,
    /// Task requesting cancellation (field `2`).
    #[prost(message, optional, tag = "2")]
    pub source_task: Option<CoordinatedTaskProto>,
    /// Barrier invocation counter (field `3`).
    #[prost(int64, tag = "3")]
    pub counter: i64,
}

impl_proto_schema_message!(CancelBarrierRequestProto, "tensorflow.CancelBarrierRequest");

/// Wire-compatible `tensorflow.CancelBarrierResponse` (`xla/tsl/protobuf/coordination_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CancelBarrierResponseProto {}

impl_proto_schema_message!(
    CancelBarrierResponseProto,
    "tensorflow.CancelBarrierResponse"
);

/// Wire-compatible `xla.megascale.runtime.MegascaleChannelSliceDescriptorProto` (`docs/reverse_engineering/protos/megascale_collectives.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleChannelSliceDescriptorProto {
    /// Collective channel identifier (field `1`).
    #[prost(uint32, tag = "1")]
    pub channel_id: u32,
    /// Source TPU slice index (field `2`).
    #[prost(int32, tag = "2")]
    pub src_slice_id: i32,
    /// Destination TPU slice index (field `3`).
    #[prost(int32, tag = "3")]
    pub dst_slice_id: i32,
    /// Payload size in bytes transferred across the DCN link (field `4`).
    #[prost(uint64, tag = "4")]
    pub payload_bytes: u64,
    /// Intra-slice ICI torus ring reduction stage index (field `5`).
    #[prost(uint32, tag = "5")]
    pub ici_ring_stage: u32,
}

impl_proto_schema_message!(
    MegascaleChannelSliceDescriptorProto,
    "xla.megascale.runtime.MegascaleChannelSliceDescriptorProto"
);

/// Wire-compatible `xla.megascale.runtime.MegascaleCollectiveScheduleProto` (`docs/reverse_engineering/protos/megascale_collectives.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleCollectiveScheduleProto {
    /// Collective operation name (field `1`).
    #[prost(string, tag = "1")]
    pub collective_name: String,
    /// Reduction kind (`0=Sum`, `1=Product`, `2=Min`, `3=Max`, field `2`).
    #[prost(int32, tag = "2")]
    pub reduction_kind: i32,
    /// Whether two-tier Intra-Slice ICI Torus + Inter-Slice DCN reduction is active (field `3`).
    #[prost(bool, tag = "3")]
    pub hierarchical_ici_then_dcn: bool,
    /// Per-channel slice transfer descriptors (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub channel_descriptors: Vec<MegascaleChannelSliceDescriptorProto>,
}

impl_proto_schema_message!(
    MegascaleCollectiveScheduleProto,
    "xla.megascale.runtime.MegascaleCollectiveScheduleProto"
);
