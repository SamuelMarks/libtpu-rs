//! TensorFlow / TSL `XPlane` hardware profiler trace protobuf messages.
//!
//! Synchronized with `tsl/profiler/protobuf/xplane.proto` (`XSpace`, `XPlane`, `XLine`, `XEvent`, `XStat`).

use crate::proto::descriptor::impl_proto_schema_message;
use prost::Message;
use std::collections::BTreeMap;

/// Wire-compatible `tensorflow.profiler.XStatMetadata` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XStatMetadataProto {
    /// Metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Stat name (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Stat description (field `3`).
    #[prost(string, tag = "3")]
    pub description: String,
}

impl_proto_schema_message!(XStatMetadataProto, "tensorflow.profiler.XStatMetadata");

/// Wire-compatible `tensorflow.profiler.XEventMetadata` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XEventMetadataProto {
    /// Metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Event name (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Additional metadata string (field `3`).
    #[prost(string, tag = "3")]
    pub metadata: String,
    /// Display name (field `4`).
    #[prost(string, tag = "4")]
    pub display_name: String,
    /// Constant event statistics shared across events with this metadata ID (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub stats: Vec<XStatProto>,
    /// Child event metadata IDs (field `6`).
    #[prost(int64, repeated, packed = "true", tag = "6")]
    pub child_id: Vec<i64>,
}

impl_proto_schema_message!(XEventMetadataProto, "tensorflow.profiler.XEventMetadata");

/// Wire-compatible `tensorflow.profiler.XStat` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XStatProto {
    /// Metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,
    /// Double stat value (field `2`).
    #[prost(double, tag = "2")]
    pub double_value: f64,
    /// Unsigned 64-bit integer stat value (field `3`).
    #[prost(uint64, tag = "3")]
    pub uint64_value: u64,
    /// Signed 64-bit integer stat value (field `4`).
    #[prost(int64, tag = "4")]
    pub int64_value: i64,
    /// String stat value (field `5`).
    #[prost(string, tag = "5")]
    pub str_value: String,
    /// Raw bytes stat value (field `6`).
    #[prost(bytes = "vec", tag = "6")]
    pub bytes_value: Vec<u8>,
    /// Referenced string metadata ID (field `7`).
    #[prost(uint64, tag = "7")]
    pub ref_value: u64,
}

impl_proto_schema_message!(XStatProto, "tensorflow.profiler.XStat");

/// Wire-compatible `tensorflow.profiler.XEvent` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XEventProto {
    /// Event metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,
    /// Offset in picoseconds from the line start timestamp (field `2`).
    #[prost(int64, tag = "2")]
    pub offset_ps: i64,
    /// Duration in picoseconds (field `3`).
    #[prost(int64, tag = "3")]
    pub duration_ps: i64,
    /// Attached event statistics (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub stats: Vec<XStatProto>,
    /// Number of aggregated event occurrences (field `5`).
    #[prost(int64, tag = "5")]
    pub num_occurrences: i64,
}

impl_proto_schema_message!(XEventProto, "tensorflow.profiler.XEvent");

/// Wire-compatible `tensorflow.profiler.XLine` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XLineProto {
    /// Line ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Line name (e.g., `"TensorCore 0"`, field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Start timestamp in nanoseconds (field `3`).
    #[prost(int64, tag = "3")]
    pub timestamp_ns: i64,
    /// Events recorded on this timeline (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub events: Vec<XEventProto>,
    /// Line duration in picoseconds (field `9`).
    #[prost(int64, tag = "9")]
    pub duration_ps: i64,
    /// Display ID (field `10`).
    #[prost(int64, tag = "10")]
    pub display_id: i64,
    /// Display name (field `11`).
    #[prost(string, tag = "11")]
    pub display_name: String,
}

impl_proto_schema_message!(XLineProto, "tensorflow.profiler.XLine");

/// Wire-compatible `tensorflow.profiler.XPlane` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XPlaneProto {
    /// Plane ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Plane name (e.g., `"/device:TPU:0"`, field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Timelines within this plane (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub lines: Vec<XLineProto>,
    /// Event metadata dictionary keyed by metadata ID (field `4`).
    #[prost(btree_map = "int64, message", tag = "4")]
    pub event_metadata: BTreeMap<i64, XEventMetadataProto>,
    /// Stat metadata dictionary keyed by metadata ID (field `5`).
    #[prost(btree_map = "int64, message", tag = "5")]
    pub stat_metadata: BTreeMap<i64, XStatMetadataProto>,
    /// Plane-level statistics (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub stats: Vec<XStatProto>,
}

impl_proto_schema_message!(XPlaneProto, "tensorflow.profiler.XPlane");

/// Wire-compatible `tensorflow.profiler.XSpace` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XSpaceProto {
    /// Planes in the trace (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub planes: Vec<XPlaneProto>,
    /// Errors recorded during profiling (field `2`).
    #[prost(string, repeated, tag = "2")]
    pub errors: Vec<String>,
    /// Warnings recorded during profiling (field `3`).
    #[prost(string, repeated, tag = "3")]
    pub warnings: Vec<String>,
    /// Hostnames participating in the trace (field `4`).
    #[prost(string, repeated, tag = "4")]
    pub hostnames: Vec<String>,
}

impl_proto_schema_message!(XSpaceProto, "tensorflow.profiler.XSpace");

/// Wire-compatible `tensorflow.ProfileOptions.TraceOptions` (`profiler_options.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProfileTraceOptionsProto {
    /// Filter mask for `TraceMe` events (field `1`).
    #[prost(uint64, tag = "1")]
    pub host_traceme_filter_mask: u64,
}

impl_proto_schema_message!(
    ProfileTraceOptionsProto,
    "tensorflow.ProfileOptions.TraceOptions"
);

/// Wire-compatible `tensorflow.ProfileOptions.AdvancedConfigValue` (`profiler_options.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProfileAdvancedConfigValueProto {
    /// String configuration value (field `1`).
    #[prost(string, tag = "1")]
    pub string_value: String,
    /// Boolean configuration value (field `2`).
    #[prost(bool, tag = "2")]
    pub bool_value: bool,
    /// Signed 64-bit integer configuration value (field `3`).
    #[prost(int64, tag = "3")]
    pub int64_value: i64,
}

impl_proto_schema_message!(
    ProfileAdvancedConfigValueProto,
    "tensorflow.ProfileOptions.AdvancedConfigValue"
);

/// Wire-compatible `tensorflow.ProfileOptions` (`profiler_options.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProfileOptionsProto {
    /// Whether to include dataset ops (field `1`).
    #[prost(bool, tag = "1")]
    pub include_dataset_ops: bool,
    /// Host tracer level (`1` = default, `0` = disable, field `2`).
    #[prost(uint32, tag = "2")]
    pub host_tracer_level: u32,
    /// Device tracer level (`1` = default, `0` = disable, field `3`).
    #[prost(uint32, tag = "3")]
    pub device_tracer_level: u32,
    /// Python tracer level (`0` = default disable, `1` = enable, field `4`).
    #[prost(uint32, tag = "4")]
    pub python_tracer_level: u32,
    /// Options version (field `5`).
    #[prost(uint32, tag = "5")]
    pub version: u32,
    /// Target device type enum (`DeviceType`, field `6`).
    #[prost(int32, tag = "6")]
    pub device_type: i32,
    /// Whether to serialize `HloProto` when XLA is used (field `7`).
    #[prost(bool, tag = "7")]
    pub enable_hlo_proto: bool,
    /// Start timestamp in nanoseconds since Unix epoch (field `8`).
    #[prost(uint64, tag = "8")]
    pub start_timestamp_ns: u64,
    /// Duration of profiling in milliseconds (field `9`).
    #[prost(uint64, tag = "9")]
    pub duration_ms: u64,
    /// Repository path for saving profile data (field `10`).
    #[prost(string, tag = "10")]
    pub repository_path: String,
    /// Low-level trace filter options (field `11`).
    #[prost(message, optional, tag = "11")]
    pub trace_options: Option<ProfileTraceOptionsProto>,
    /// Advanced key-value profiler configuration (field `12`).
    #[prost(btree_map = "string, message", tag = "12")]
    pub advanced_configuration: BTreeMap<String, ProfileAdvancedConfigValueProto>,
    /// Raise error when on-demand profiling fails (field `13`).
    #[prost(bool, tag = "13")]
    pub raise_error_on_start_failure: bool,
    /// Session identifier string (field `14`).
    #[prost(string, tag = "14")]
    pub session_id: String,
    /// Overridden hostname (field `15`).
    #[prost(string, tag = "15")]
    pub override_hostname: String,
    /// Whether programmatic replay multipass is enabled (field `16`).
    #[prost(bool, tag = "16")]
    pub enable_multipass: bool,
}

impl_proto_schema_message!(ProfileOptionsProto, "tensorflow.ProfileOptions");

/// Wire-compatible `tensorflow.RemoteProfilerSessionManagerOptions` (`profiler_options.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RemoteProfilerSessionManagerOptionsProto {
    /// Base profiler options (field `1`).
    #[prost(message, optional, tag = "1")]
    pub profiler_options: Option<ProfileOptionsProto>,
    /// Target gRPC service addresses (field `2`).
    #[prost(string, repeated, tag = "2")]
    pub service_addresses: Vec<String>,
    /// Session creation timestamp in nanoseconds (field `3`).
    #[prost(uint64, tag = "3")]
    pub session_creation_timestamp_ns: u64,
    /// Maximum session duration in milliseconds (field `4`).
    #[prost(uint64, tag = "4")]
    pub max_session_duration_ms: u64,
    /// Profiling start delay in milliseconds (field `5`).
    #[prost(uint64, tag = "5")]
    pub delay_ms: u64,
}

impl_proto_schema_message!(
    RemoteProfilerSessionManagerOptionsProto,
    "tensorflow.RemoteProfilerSessionManagerOptions"
);

/// Wire-compatible `tensorflow.ToolRequestOptions` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ToolRequestOptionsProto {
    /// Desired file extension for tool data (field `2`).
    #[prost(string, tag = "2")]
    pub output_formats: String,
    /// Whether to save tool data directly to repository (field `3`).
    #[prost(bool, tag = "3")]
    pub save_to_repo: bool,
}

impl_proto_schema_message!(ToolRequestOptionsProto, "tensorflow.ToolRequestOptions");

/// Wire-compatible `tensorflow.ProfileRequest` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProfileRequestProto {
    /// Tracing duration in milliseconds (field `1`).
    #[prost(uint64, tag = "1")]
    pub duration_ms: u64,
    /// Maximum events to return (field `2`).
    #[prost(uint64, tag = "2")]
    pub max_events: u64,
    /// Requested profiling tools (e.g., `"trace_viewer"`, field `3`).
    #[prost(string, repeated, tag = "3")]
    pub tools: Vec<String>,
    /// Per-tool options dictionary (field `8`).
    #[prost(btree_map = "string, message", tag = "8")]
    pub tool_options: BTreeMap<String, ToolRequestOptionsProto>,
    /// Profiler options (field `4`).
    #[prost(message, optional, tag = "4")]
    pub opts: Option<ProfileOptionsProto>,
    /// Repository root directory (field `5`).
    #[prost(string, tag = "5")]
    pub repository_root: String,
    /// Profiling session ID (field `6`).
    #[prost(string, tag = "6")]
    pub session_id: String,
    /// Hostname of the profiled system (field `7`).
    #[prost(string, tag = "7")]
    pub host_name: String,
    /// Whether to emit an `XSpace` payload in the response (field `9`).
    #[prost(bool, tag = "9")]
    pub emit_xspace: bool,
}

impl_proto_schema_message!(ProfileRequestProto, "tensorflow.ProfileRequest");

/// Wire-compatible `tensorflow.ProfileToolData` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProfileToolDataProto {
    /// File name associated with this tool data (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Serialized tool payload bytes (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub data: Vec<u8>,
}

impl_proto_schema_message!(ProfileToolDataProto, "tensorflow.ProfileToolData");

/// Wire-compatible `tensorflow.ProfileResponse` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct ProfileResponseProto {
    /// Tool data payloads (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub tool_data: Vec<ProfileToolDataProto>,
    /// Whether the trace was empty (field `7`).
    #[prost(bool, tag = "7")]
    pub empty_trace: bool,
    /// Captured `XSpace` structured trace (field `8`).
    #[prost(message, optional, tag = "8")]
    pub xspace: Option<XSpaceProto>,
}

impl_proto_schema_message!(ProfileResponseProto, "tensorflow.ProfileResponse");

/// Wire-compatible `tensorflow.TerminateRequest` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TerminateRequestProto {
    /// Session ID to terminate (field `1`).
    #[prost(string, tag = "1")]
    pub session_id: String,
}

impl_proto_schema_message!(TerminateRequestProto, "tensorflow.TerminateRequest");

/// Wire-compatible `tensorflow.TerminateResponse` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TerminateResponseProto {}

impl_proto_schema_message!(TerminateResponseProto, "tensorflow.TerminateResponse");

/// Wire-compatible `tensorflow.GetSnapshotRequest` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GetSnapshotRequestProto {
    /// Snapshot session identifier (field `1`).
    #[prost(string, tag = "1")]
    pub snapshot_session_id: String,
}

impl_proto_schema_message!(GetSnapshotRequestProto, "tensorflow.GetSnapshotRequest");

/// Wire-compatible `tensorflow.ContinuousProfilingResponse` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ContinuousProfilingResponseProto {}

impl_proto_schema_message!(
    ContinuousProfilingResponseProto,
    "tensorflow.ContinuousProfilingResponse"
);

/// Wire-compatible `tensorflow.StopContinuousProfilingRequest` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StopContinuousProfilingRequestProto {}

impl_proto_schema_message!(
    StopContinuousProfilingRequestProto,
    "tensorflow.StopContinuousProfilingRequest"
);

/// Wire-compatible `tensorflow.StopContinuousProfilingResponse` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StopContinuousProfilingResponseProto {}

impl_proto_schema_message!(
    StopContinuousProfilingResponseProto,
    "tensorflow.StopContinuousProfilingResponse"
);

/// Wire-compatible `tensorflow.MonitorRequest` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MonitorRequestProto {
    /// Monitoring duration in milliseconds (field `1`).
    #[prost(uint64, tag = "1")]
    pub duration_ms: u64,
    /// Monitoring detail level (`1` or `2`, field `2`).
    #[prost(int32, tag = "2")]
    pub monitoring_level: i32,
    /// Whether to include timestamps in output (field `3`).
    #[prost(bool, tag = "3")]
    pub timestamp: bool,
}

impl_proto_schema_message!(MonitorRequestProto, "tensorflow.MonitorRequest");

/// Wire-compatible `tensorflow.ProfilerServiceMonitorResult` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct ProfilerServiceMonitorResultProto {
    /// Formatted monitoring result string (field `1`).
    #[prost(string, tag = "1")]
    pub formatted_result: String,
    /// Device idle time percentage (field `2`).
    #[prost(double, tag = "2")]
    pub device_idle_time_percent: f64,
    /// Matrix unit (`MXU`) utilization percentage (field `3`).
    #[prost(double, tag = "3")]
    pub matrix_unit_utilization_percent: f64,
    /// Average step time in milliseconds (field `4`).
    #[prost(double, tag = "4")]
    pub step_time_ms_avg: f64,
}

impl_proto_schema_message!(
    ProfilerServiceMonitorResultProto,
    "tensorflow.ProfilerServiceMonitorResult"
);

/// Wire-compatible `tensorflow.MonitorResponse` (`profiler_service.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct MonitorResponseProto {
    /// Formatted monitoring summary string (field `1`).
    #[prost(string, tag = "1")]
    pub data: String,
    /// Structured monitoring metrics (field `10`).
    #[prost(message, optional, tag = "10")]
    pub monitor_result: Option<ProfilerServiceMonitorResultProto>,
}

impl_proto_schema_message!(MonitorResponseProto, "tensorflow.MonitorResponse");

/// Wire-compatible `tensorflow.profiler.TopologyLocation` (`profiler_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProfilerTopologyLocationProto {
    /// Mesh X coordinate (field `1`).
    #[prost(int32, tag = "1")]
    pub x: i32,
    /// Mesh Y coordinate (field `2`).
    #[prost(int32, tag = "2")]
    pub y: i32,
    /// Mesh Z coordinate (field `3`).
    #[prost(int32, tag = "3")]
    pub z: i32,
    /// Host X coordinate (field `4`).
    #[prost(int32, tag = "4")]
    pub host_x: i32,
    /// Host Y coordinate (field `5`).
    #[prost(int32, tag = "5")]
    pub host_y: i32,
    /// Host Z coordinate (field `6`).
    #[prost(int32, tag = "6")]
    pub host_z: i32,
    /// Chip index on the host (field `7`).
    #[prost(int32, tag = "7")]
    pub index_on_host: i32,
    /// Global chip ID (field `8`).
    #[prost(int32, tag = "8")]
    pub global_id: i32,
}

impl_proto_schema_message!(
    ProfilerTopologyLocationProto,
    "tensorflow.profiler.TopologyLocation"
);

/// Wire-compatible `tensorflow.profiler.ProfilerTopology` (`profiler_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProfilerTopologyProto {
    /// Total chip dimensions `(x, y, z)` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub num_chips: Option<ProfilerTopologyLocationProto>,
    /// Total host dimensions `(x, y, z)` (field `2`).
    #[prost(message, optional, tag = "2")]
    pub num_hosts: Option<ProfilerTopologyLocationProto>,
    /// Per-chip topology locations (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub chip_location: Vec<ProfilerTopologyLocationProto>,
}

impl_proto_schema_message!(
    ProfilerTopologyProto,
    "tensorflow.profiler.ProfilerTopology"
);

/// Wire-compatible `tensorflow.tpu.PerTpuStepDetails` (`tpu_input_pipeline.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct PerTpuStepDetailsProto {
    /// Training step number (field `1`).
    #[prost(int32, tag = "1")]
    pub step_number: i32,
    /// `TensorCore` compute time in milliseconds (field `2`).
    #[prost(double, tag = "2")]
    pub tc_compute_time_ms: f64,
    /// `SparseCore` compute time in milliseconds (field `3`).
    #[prost(double, tag = "3")]
    pub scv0_compute_time_ms: f64,
    /// Host-to-device infeed wait time in milliseconds (field `4`).
    #[prost(double, tag = "4")]
    pub infeed_time_ms: f64,
    /// Device-to-host outfeed wait time in milliseconds (field `5`).
    #[prost(double, tag = "5")]
    pub outfeed_time_ms: f64,
    /// Cross-replica `AllReduce` compute time in milliseconds (field `6`).
    #[prost(double, tag = "6")]
    pub all_reduce_compute_time_ms: f64,
    /// Cross-replica `AllReduce` synchronization time in milliseconds (field `7`).
    #[prost(double, tag = "7")]
    pub all_reduce_sync_time_ms: f64,
}

impl_proto_schema_message!(PerTpuStepDetailsProto, "tensorflow.tpu.PerTpuStepDetails");

/// Wire-compatible `tensorflow.tpu.TpuBottleneckAnalysis` (`tpu_input_pipeline.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct TpuBottleneckAnalysisProto {
    /// Percentage of step time spent waiting on input (field `1`).
    #[prost(double, tag = "1")]
    pub input_percent: f64,
    /// Percentage of step time spent waiting on output (field `2`).
    #[prost(double, tag = "2")]
    pub output_percent: f64,
    /// Percentage of step time where `TensorCore` is idle (field `3`).
    #[prost(double, tag = "3")]
    pub tc_idle_percent: f64,
    /// Bottleneck classification label (e.g., `"device"`, `"host"`, field `4`).
    #[prost(string, tag = "4")]
    pub input_classification: String,
    /// Human-readable bottleneck recommendation statement (field `5`).
    #[prost(string, tag = "5")]
    pub input_statement: String,
}

impl_proto_schema_message!(
    TpuBottleneckAnalysisProto,
    "tensorflow.tpu.TpuBottleneckAnalysis"
);

/// Wire-compatible `tensorflow.tpu.TpuInputPipelineAnalysisResult` (`tpu_input_pipeline.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct TpuInputPipelineAnalysisResultProto {
    /// Whether the profiled hardware is a Cloud TPU (field `1`).
    #[prost(bool, tag = "1")]
    pub hardware_is_tpu: bool,
    /// Average percentage of step time spent in infeed (field `2`).
    #[prost(double, tag = "2")]
    pub infeed_percent_average: f64,
    /// Minimum percentage of step time spent in infeed (field `3`).
    #[prost(double, tag = "3")]
    pub infeed_percent_minimum: f64,
    /// Maximum percentage of step time spent in infeed (field `4`).
    #[prost(double, tag = "4")]
    pub infeed_percent_maximum: f64,
    /// Standard deviation of infeed percentage (field `5`).
    #[prost(double, tag = "5")]
    pub infeed_percent_standard_deviation: f64,
    /// Average step duration in milliseconds (field `6`).
    #[prost(double, tag = "6")]
    pub step_time_ms_average: f64,
    /// Per-step timing breakdown records (field `7`).
    #[prost(message, repeated, tag = "7")]
    pub step_details: Vec<PerTpuStepDetailsProto>,
    /// Input pipeline bottleneck diagnosis (field `8`).
    #[prost(message, optional, tag = "8")]
    pub bottleneck_analysis: Option<TpuBottleneckAnalysisProto>,
}

impl_proto_schema_message!(
    TpuInputPipelineAnalysisResultProto,
    "tensorflow.tpu.TpuInputPipelineAnalysisResult"
);
