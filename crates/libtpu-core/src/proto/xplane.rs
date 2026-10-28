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
#[derive(Clone, PartialEq, Eq, Message)]
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
