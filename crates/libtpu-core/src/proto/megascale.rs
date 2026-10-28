//! Megascale multi-slice coordination, DCN topology, transport, and runtime error overlays.
//!
//! Synchronized with `megascale_*.proto` and `runtime_config.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use prost::Message;

/// Wire-compatible `megascale.MultiSliceConfigProto` (`xla/megascale/megascale.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
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
