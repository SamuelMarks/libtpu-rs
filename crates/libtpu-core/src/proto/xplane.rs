//! TensorFlow / TSL `XPlane` and `Xprof` hardware profiler protobuf types.
//!
//! Re-exports `prost-build` generated structs from `third_party/tensorflow/tsl/profiler/protobuf/`
//! (`xplane.proto`, `profiler_options.proto`, `profiler_service_monitor_result.proto`,
//! `profiler_topology.proto`), `perftools/accelerators/xprof/xprof_service.proto`,
//! and `protos/libtpu_ext/xplane_ext.proto`.

use super::descriptor::ProtoSchemaMessage as _;
use crate::error::LibTpuResult;

pub use super::descriptor::generated_prost_schemas::libtpu::ext::xplane::{
    ContinuousProfilingResponseProto, GetSnapshotRequestProto, MonitorRequestProto,
    MonitorResponseProto, PerTpuStepDetailsProto, ProfileRequestProto, ProfileResponseProto,
    ProfileToolDataProto, StopContinuousProfilingRequestProto,
    StopContinuousProfilingResponseProto, TerminateRequestProto, TerminateResponseProto,
    ToolRequestOptionsProto, TpuBottleneckAnalysisProto, TpuInputPipelineAnalysisResultProto,
};
pub use super::descriptor::generated_prost_schemas::tensorflow::profiler::{
    Topology as ProfilerTopologyProto, TopologyLocation as ProfilerTopologyLocationProto,
    XEvent as XEventProto, XEventMetadata as XEventMetadataProto, XLine as XLineProto,
    XPlane as XPlaneProto, XSpace as XSpaceProto, XStat as XStatProto,
    XStatMetadata as XStatMetadataProto, x_event, x_stat,
};
pub use super::descriptor::generated_prost_schemas::tensorflow::{
    ProfileOptions as ProfileOptionsProto,
    ProfilerServiceMonitorResult as ProfilerServiceMonitorResultProto,
    RemoteProfilerSessionManagerOptions as RemoteProfilerSessionManagerOptionsProto,
};
pub use super::descriptor::generated_prost_schemas::xprof::{
    XprofMonitorRequest, XprofMonitorResponse, XprofRequest, XprofResponse,
};

/// Alias for `tensorflow.ProfileOptions.TraceOptions` generated from `profiler_options.proto`.
pub type ProfileTraceOptionsProto =
    super::descriptor::generated_prost_schemas::tensorflow::profile_options::TraceOptions;

/// Alias for `tensorflow.ProfileOptions.AdvancedConfigValue` generated from `profiler_options.proto`.
pub type ProfileAdvancedConfigValueProto =
    super::descriptor::generated_prost_schemas::tensorflow::profile_options::AdvancedConfigValue;

impl ProfileRequestProto {
    /// Serializes this [`ProfileRequestProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`ProfileRequestProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ProfileRequestProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl ProfileResponseProto {
    /// Serializes this [`ProfileResponseProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`ProfileResponseProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ProfileResponseProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl TerminateRequestProto {
    /// Serializes this [`TerminateRequestProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`TerminateRequestProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TerminateRequestProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl TerminateResponseProto {
    /// Serializes this [`TerminateResponseProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`TerminateResponseProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TerminateResponseProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl GetSnapshotRequestProto {
    /// Serializes this [`GetSnapshotRequestProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`GetSnapshotRequestProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `GetSnapshotRequestProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl StopContinuousProfilingRequestProto {
    /// Serializes this [`StopContinuousProfilingRequestProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`StopContinuousProfilingRequestProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `StopContinuousProfilingRequestProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl StopContinuousProfilingResponseProto {
    /// Serializes this [`StopContinuousProfilingResponseProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`StopContinuousProfilingResponseProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `StopContinuousProfilingResponseProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl MonitorRequestProto {
    /// Serializes this [`MonitorRequestProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`MonitorRequestProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MonitorRequestProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl MonitorResponseProto {
    /// Serializes this [`MonitorResponseProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`MonitorResponseProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MonitorResponseProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl TpuInputPipelineAnalysisResultProto {
    /// Serializes this [`TpuInputPipelineAnalysisResultProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`TpuInputPipelineAnalysisResultProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuInputPipelineAnalysisResultProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}
