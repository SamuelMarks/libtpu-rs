//! `MegaScale` multi-slice DCN topology, transport, runtime error overlay, and TensorFlow
//! `CoordinationService` protobuf types.
//!
//! Re-exports `prost-build` generated structs from `platforms/xla/megascale/`,
//! `third_party/tensorflow/compiler/xla/tsl/protobuf/status.proto`,
//! `coordination_config.proto`, `coordination_service.proto`, and `protos/libtpu_ext/megascale_ext.proto`.

use super::descriptor::ProtoSchemaMessage as _;
use crate::error::LibTpuResult;

pub use super::descriptor::generated_prost_schemas::accel_ssw::deepsea::proto::{
    RoutingEntry as DeepseaRoutingEntryProto, RoutingTable as DeepseaRoutingTableProto,
    UnicastRoutingTable as DeepseaUnicastRoutingTableProto,
    UnicastTarget as DeepseaUnicastTargetProto,
    VirtualChannelControl as DeepseaVirtualChannelControlProto,
    routing_table as deepsea_routing_table,
};
pub use super::descriptor::generated_prost_schemas::libtpu::ext::megascale::{
    MegascaleChannelSliceDescriptorProto, MegascaleCollectiveScheduleProto,
    MegascaleCongestionControlConfigProto, MegascaleCongestionWindowStateProto,
    MegascaleDcnTopologyProto, MegascaleMultiSliceConfigProto, MegascaleRuntimeMetricsProto,
    MegascaleTransportConfigProto, SliceDeviceListProto,
};
pub use super::descriptor::generated_prost_schemas::superpod::routing::proto::{
    ChipCoordinate as SuperpodChipCoordinateProto,
    ChipLocationToIciConfig as SuperpodChipLocationToIciConfigProto,
    IciNetworkConfig as SuperpodIciNetworkConfigProto, Topology as SuperpodTopologyProto,
    ToroidalTopology as SuperpodToroidalTopologyProto,
    TwistedTorusShape as SuperpodTwistedTorusShapeProto, topology as superpod_topology,
    toroidal_topology as superpod_toroidal_topology,
};
pub use super::descriptor::generated_prost_schemas::tensorflow::{
    BarrierError as BarrierErrorProto, BarrierRequest as BarrierRequestProto,
    BarrierResponse as BarrierResponseProto, CancelBarrierRequest as CancelBarrierRequestProto,
    CancelBarrierResponse as CancelBarrierResponseProto, CoordinatedJob as CoordinatedJobProto,
    CoordinatedTask as CoordinatedTaskProto, CoordinatedTaskState as CoordinatedTaskStateProto,
    CoordinatedTaskStateInfo as CoordinatedTaskStateInfoProto,
    CoordinationServiceConfig as CoordinationServiceConfigProto,
    CoordinationServiceError as CoordinationServiceErrorProto,
    DeleteKeyValueRequest as DeleteKeyValueRequestProto,
    DeleteKeyValueResponse as DeleteKeyValueResponseProto,
    DeviceInfo as CoordinationDeviceInfoProto, GetAliveTasksRequest as GetAliveTasksRequestProto,
    GetAliveTasksResponse as GetAliveTasksResponseProto,
    GetKeyValueDirRequest as GetKeyValueDirRequestProto,
    GetKeyValueDirResponse as GetKeyValueDirResponseProto,
    GetKeyValueRequest as GetKeyValueRequestProto, GetKeyValueResponse as GetKeyValueResponseProto,
    GetTaskStateRequest as GetTaskStateRequestProto,
    GetTaskStateResponse as GetTaskStateResponseProto, HeartbeatRequest as HeartbeatRequestProto,
    HeartbeatResponse as HeartbeatResponseProto,
    IncrementKeyValueRequest as IncrementKeyValueRequestProto,
    IncrementKeyValueResponse as IncrementKeyValueResponseProto,
    InsertKeyValueRequest as InsertKeyValueRequestProto,
    InsertKeyValueResponse as InsertKeyValueResponseProto, KeyValueEntry as KeyValueEntryProto,
    PollForErrorRequest as PollForErrorRequestProto,
    PollForErrorResponse as PollForErrorResponseProto,
    RegisterTaskRequest as RegisterTaskRequestProto,
    RegisterTaskResponse as RegisterTaskResponseProto,
    ReportErrorToServiceRequest as ReportErrorToServiceRequestProto,
    ReportErrorToServiceResponse as ReportErrorToServiceResponseProto,
    ReportErrorToTaskRequest as ReportErrorToTaskRequestProto,
    ReportErrorToTaskResponse as ReportErrorToTaskResponseProto,
    ResetTaskRequest as ResetTaskRequestProto, ResetTaskResponse as ResetTaskResponseProto,
    ShutdownTaskRequest as ShutdownTaskRequestProto,
    ShutdownTaskResponse as ShutdownTaskResponseProto, StatusProto,
    TryGetKeyValueRequest as TryGetKeyValueRequestProto,
    TryGetKeyValueResponse as TryGetKeyValueResponseProto,
    WaitForAllTasksRequest as WaitForAllTasksRequestProto,
    WaitForAllTasksResponse as WaitForAllTasksResponseProto,
    WatchJobStateRequest as WatchJobStateRequestProto,
    WatchJobStateResponse as WatchJobStateResponseProto,
};
pub use super::descriptor::generated_prost_schemas::xla::megascale::runtime::{
    DcnTopology as DcnTopologyTreeProto, EndpointAddresses as EndpointAddressesProto,
    ExecutorTelemetry as ExecutorTelemetryProto, HostNetworkAddress as HostNetworkAddressProto,
    MegaScaleAction as MegaScaleActionProto, MegaScaleActionGraph as MegaScaleActionGraphProto,
    MegaScaleRuntimeError as MegaScaleRuntimeErrorProto, MultiSliceTopologyAndLocationProto,
    NetworkAddressMapping as NetworkAddressMappingProto, ProgramRunInfo as ProgramRunInfoProto,
    RapidEyeErrorDigestProto, RapidEyeInfo as RapidEyeInfoProto,
    RequestId as MegascaleRequestIdProto, RuntimeConfig as MegascaleRuntimeConfigProto,
    RuntimeStateSummary as RuntimeStateSummaryProto, SendRequest as MegascaleSendRequestProto,
    TpuCoreHlo as TpuCoreHloProto, TransferBufferRequest as MegascaleTransferBufferRequestProto,
    TransferBufferResponse as MegascaleTransferBufferResponseProto, dcn_topology,
    mega_scale_action, mega_scale_runtime_error, rapid_eye_error_digest_proto,
};

/// Alias for `xla.megascale.runtime.DCNTopology.SymmetricTree`.
pub type DcnSymmetricTreeProto = dcn_topology::SymmetricTree;

/// Alias for `xla.megascale.runtime.DCNTopology.TreeNode`.
pub type DcnTreeNodeProto = dcn_topology::TreeNode;

/// Alias for `xla.megascale.runtime.DCNTopology.TreeNode.SliceRange`.
pub type DcnSliceRangeProto = dcn_topology::tree_node::SliceRange;

/// Alias for `xla.megascale.runtime.MegaScaleAction.Input`.
pub type MegaScaleActionInputProto = mega_scale_action::Input;

/// Alias for `xla.megascale.runtime.ExecutorTelemetry.PendingGraphState`.
pub type PendingGraphStateProto = super::descriptor::generated_prost_schemas::xla::megascale::runtime::executor_telemetry::PendingGraphState;

impl MegascaleMultiSliceConfigProto {
    /// Encodes this [`MegascaleMultiSliceConfigProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Decodes a [`MegascaleMultiSliceConfigProto`] from Protobuf wire bytes produced by [`Self::encode`].
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MegascaleMultiSliceConfigProto` byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl MegascaleDcnTopologyProto {
    /// Encodes this [`MegascaleDcnTopologyProto`] to Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Decodes a [`MegascaleDcnTopologyProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MegascaleDcnTopologyProto` byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl MegascaleRuntimeMetricsProto {
    /// Encodes this [`MegascaleRuntimeMetricsProto`] to Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Decodes a [`MegascaleRuntimeMetricsProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MegascaleRuntimeMetricsProto` byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl MegascaleTransportConfigProto {
    /// Encodes this [`MegascaleTransportConfigProto`] to Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Decodes a [`MegascaleTransportConfigProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MegascaleTransportConfigProto` byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}
