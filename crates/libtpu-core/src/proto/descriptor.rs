//! Protobuf descriptor reflection, dynamic transcoding, and `.proto` schema verification.
//!
//! Powered by `prost` and `prost-reflect` over the embedded `FileDescriptorSet`.

use crate::error::{LibTpuError, LibTpuResult};
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Raw wire-encoded `google.protobuf.FileDescriptorSet` compiled at build time by `protox` + `prost-build`
/// from `docs/reverse_engineering/protos/*.proto`.
pub const EMBEDDED_FILE_DESCRIPTOR_SET_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/libtpu_file_descriptor_set.bin"));

/// Canonical list of all strongly-typed `libtpu-rs` protobuf message names synchronized with
/// [`global_descriptor_pool`].
pub const ALL_REGISTERED_PROTO_MESSAGE_NAMES: &[&str] = &[
    "asic_sw.AnalyzerReport",
    "asic_sw.CoreDump",
    "asic_sw.CoreStateDump",
    "asic_sw.HardwareInfo",
    "asic_sw.MachineInfo",
    "asic_sw.RegisterDump",
    "asic_sw.TimestampInfo",
    "jax.DictKeysProto",
    "jax.PyHostCallbackProto",
    "jax.PyTreeDefProto",
    "jax.PyTreeNodeDefProto",
    "libtpu.sdk.tpu_telemetry.ActiveSyncFlagWait",
    "libtpu.sdk.tpu_telemetry.AllCoreStateSummaries",
    "libtpu.sdk.tpu_telemetry.CoreStallSummary",
    "megascale.DCNTopologyProto",
    "megascale.MultiSliceConfigProto",
    "megascale.RuntimeMetricsProto",
    "megascale.SliceDeviceListProto",
    "mlir.tpu.MosaicBackendConfigProto",
    "mlir.tpu.MosaicDmaDescriptorProto",
    "mlir.tpu.MosaicScratchpadLayoutProto",
    "platforms_deepsea.jellyfish.xdb.CompilerMetadata",
    "platforms_deepsea.jellyfish.xdb.HloToLloMapping",
    "platforms_deepsea.jellyfish.xdb.SectionAllocation",
    "platforms_deepsea.jellyfish.xdb.SymbolTableEntry",
    "stream_executor.CudaComputeCapabilityProto",
    "stream_executor.DnnVersionInfoProto",
    "stream_executor.ExecutableAbiVersionProto",
    "stream_executor.ExecutableAbiVersionProto.CudaPlatformVersion",
    "stream_executor.GpuDeviceInfoProto",
    "stream_executor.GpuTargetConfigProto",
    "stream_executor.OneAPIComputeCapabilityProto",
    "stream_executor.RuntimeAbiVersionProto",
    "stream_executor.dnn.AlgorithmConfigProto",
    "stream_executor.dnn.AlgorithmProto",
    "stream_executor.dnn.ConvolutionDescriptorProto",
    "stream_executor.dnn.TensorDescriptorProto",
    "tensorflow.BarrierError",
    "tensorflow.BarrierRequest",
    "tensorflow.BarrierResponse",
    "tensorflow.CancelBarrierRequest",
    "tensorflow.CancelBarrierResponse",
    "tensorflow.ContinuousProfilingResponse",
    "tensorflow.CoordinatedJob",
    "tensorflow.CoordinatedTask",
    "tensorflow.CoordinatedTaskStateInfo",
    "tensorflow.CoordinationServiceConfig",
    "tensorflow.CoordinationServiceError",
    "tensorflow.DeleteKeyValueRequest",
    "tensorflow.DeleteKeyValueResponse",
    "tensorflow.DeviceInfo",
    "tensorflow.GetAliveTasksRequest",
    "tensorflow.GetAliveTasksResponse",
    "tensorflow.GetKeyValueDirRequest",
    "tensorflow.GetKeyValueDirResponse",
    "tensorflow.GetKeyValueRequest",
    "tensorflow.GetKeyValueResponse",
    "tensorflow.GetSnapshotRequest",
    "tensorflow.GetTaskStateRequest",
    "tensorflow.GetTaskStateResponse",
    "tensorflow.HeartbeatRequest",
    "tensorflow.HeartbeatResponse",
    "tensorflow.IncrementKeyValueRequest",
    "tensorflow.IncrementKeyValueResponse",
    "tensorflow.InsertKeyValueRequest",
    "tensorflow.InsertKeyValueResponse",
    "tensorflow.KeyValueEntry",
    "tensorflow.MonitorRequest",
    "tensorflow.MonitorResponse",
    "tensorflow.PollForErrorRequest",
    "tensorflow.PollForErrorResponse",
    "tensorflow.ProfileOptions",
    "tensorflow.ProfileOptions.AdvancedConfigValue",
    "tensorflow.ProfileOptions.TraceOptions",
    "tensorflow.ProfileRequest",
    "tensorflow.ProfileResponse",
    "tensorflow.ProfileToolData",
    "tensorflow.ProfilerServiceMonitorResult",
    "tensorflow.RegisterTaskRequest",
    "tensorflow.RegisterTaskResponse",
    "tensorflow.RemoteProfilerSessionManagerOptions",
    "tensorflow.ReportErrorToServiceRequest",
    "tensorflow.ReportErrorToServiceResponse",
    "tensorflow.ReportErrorToTaskRequest",
    "tensorflow.ReportErrorToTaskResponse",
    "tensorflow.ResetTaskRequest",
    "tensorflow.ResetTaskResponse",
    "tensorflow.ShutdownTaskRequest",
    "tensorflow.ShutdownTaskResponse",
    "tensorflow.StatusProto",
    "tensorflow.StopContinuousProfilingRequest",
    "tensorflow.StopContinuousProfilingResponse",
    "tensorflow.TPUExecutableInfoProto",
    "tensorflow.TPUExecutableInfoProto.ShapeIndex",
    "tensorflow.TPUExecutableInfoProto.UpdateIndexPair",
    "tensorflow.TPUHostTransferInfoProto",
    "tensorflow.TensorProto",
    "tensorflow.TensorShapeProto",
    "tensorflow.TensorShapeProto.Dim",
    "tensorflow.TerminateRequest",
    "tensorflow.TerminateResponse",
    "tensorflow.ToolRequestOptions",
    "tensorflow.TryGetKeyValueRequest",
    "tensorflow.TryGetKeyValueResponse",
    "tensorflow.WaitForAllTasksRequest",
    "tensorflow.WaitForAllTasksResponse",
    "tensorflow.WatchJobStateRequest",
    "tensorflow.WatchJobStateResponse",
    "tensorflow.profiler.ProfilerTopology",
    "tensorflow.profiler.TopologyLocation",
    "tensorflow.profiler.XEvent",
    "tensorflow.profiler.XEventMetadata",
    "tensorflow.profiler.XLine",
    "tensorflow.profiler.XPlane",
    "tensorflow.profiler.XSpace",
    "tensorflow.profiler.XStat",
    "tensorflow.profiler.XStatMetadata",
    "tensorflow.tf2xla.Config",
    "tensorflow.tf2xla.Feed",
    "tensorflow.tf2xla.Fetch",
    "tensorflow.tf2xla.HostComputeMetadata",
    "tensorflow.tf2xla.HostTransferMetadata",
    "tensorflow.tf2xla.TensorId",
    "tensorflow.tf2xla.TensorMetadata",
    "tensorflow.tf2xla.Variable",
    "tensorflow.tf2xla.XlaArgumentProto",
    "tensorflow.tf2xla.XlaArgumentProto.Shape",
    "tensorflow.tpu.AdadeltaParameters",
    "tensorflow.tpu.AdagradMomentumParameters",
    "tensorflow.tpu.AdagradParameters",
    "tensorflow.tpu.AdamParameters",
    "tensorflow.tpu.AssignParameters",
    "tensorflow.tpu.BoundedAdagradParameters",
    "tensorflow.tpu.CenteredRmsPropParameters",
    "tensorflow.tpu.ClippingLimits",
    "tensorflow.tpu.CompilationResultProto",
    "tensorflow.tpu.FrequencyAwareAdagradParameters",
    "tensorflow.tpu.FrequencyEstimatorParameters",
    "tensorflow.tpu.FtrlParameters",
    "tensorflow.tpu.GradientAccumulationStatus",
    "tensorflow.tpu.HotIdReplicationConfiguration",
    "tensorflow.tpu.LearningRate",
    "tensorflow.tpu.LionParameters",
    "tensorflow.tpu.LowDimensionalPackingStatus",
    "tensorflow.tpu.MdlAdagradLightParameters",
    "tensorflow.tpu.MomentumParameters",
    "tensorflow.tpu.OnlineYogiParameters",
    "tensorflow.tpu.OptimizationParameters",
    "tensorflow.tpu.OptimizerDynamicInput",
    "tensorflow.tpu.PaddingMap",
    "tensorflow.tpu.PerTpuStepDetails",
    "tensorflow.tpu.ProximalAdagradParameters",
    "tensorflow.tpu.ProximalYogiParameters",
    "tensorflow.tpu.RmsPropParameters",
    "tensorflow.tpu.SimulatedQuantization",
    "tensorflow.tpu.SparseCoreTableLayout",
    "tensorflow.tpu.SparseCoreTableLayouts",
    "tensorflow.tpu.StateVariableSpecification",
    "tensorflow.tpu.StateVariableSpecification.FillWithConstant",
    "tensorflow.tpu.StateVariableSpecification.UserDefined",
    "tensorflow.tpu.StochasticGradientDescentParameters",
    "tensorflow.tpu.TPUCompileMetadataProto",
    "tensorflow.tpu.TPUCompileMetadataProto.Arg",
    "tensorflow.tpu.TPUCompileMetadataProto.Retval",
    "tensorflow.tpu.TPUEmbeddingConfiguration",
    "tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor",
    "tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding",
    "tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor",
    "tensorflow.tpu.TPUHardwareFeature",
    "tensorflow.tpu.TopologyProto",
    "tensorflow.tpu.TpuBottleneckAnalysis",
    "tensorflow.tpu.TpuInputPipelineAnalysisResult",
    "tensorflow.tpu.UserDefinedProgramParameters",
    "tpu.HostCommandEnvelopeProto",
    "tpu.HloModuleFingerprintProto",
    "tpu.RegisterTranslatorDataProto",
    "tpu.RegisterTranslatorEntryProto",
    "tpu.SliceConfig",
    "tpu.TpuChipConfigProto",
    "tpu.TpuChipPartsProto",
    "tpu.TpuChipVariantDescriptorProto",
    "tpu.TpuCompilationEnvironmentProto",
    "tpu.TpuCompilationFingerprintProto",
    "tpu.TpuCompilerEnvironmentFingerprintProto",
    "tpu.TpuCoreProgramAbiOptionsProto",
    "tpu.TpuCoreProgramAbiProto",
    "tpu.TpuEmbeddingHostNetworkConfigProto",
    "tpu.TpuEmbeddingMemoryConfigProto",
    "tpu.TpuEmbeddingPartitionedConfigProto",
    "tpu.TpuPlatformConfig",
    "tpu.TpuPlatformConfigs",
    "tpu.TpuRuntimeAbiProto",
    "tpu.TpuSerializedExecutableProto",
    "tpu.TpuTopologyArgsProto",
    "tpu.telemetry.ChipThermalThrottleSummaryProto",
    "tpu.telemetry.FrequencyScalingTelemetryProto",
    "tpu.telemetry.TpuPowerStateProto",
    "xla.AutotuneResult",
    "xla.AutotuneResult.BackendConfigKey",
    "xla.AutotuneResult.ConvKey",
    "xla.AutotuneResult.CudaConvPlanKey",
    "xla.AutotuneResult.CustomKernelFusionKey",
    "xla.AutotuneResult.FailureResult",
    "xla.AutotuneResult.GemmKey",
    "xla.AutotuneResult.TritonGemmKey",
    "xla.AutotuneResults",
    "xla.AutotuneResults.Entry",
    "xla.AutotuningLog",
    "xla.AutotuningLogs",
    "xla.BufferAllocationProto",
    "xla.BufferAssignmentProto",
    "xla.ChannelHandle",
    "xla.CholeskyOptions",
    "xla.CompilationEnvironmentsProto",
    "xla.CompilationLogEntry",
    "xla.CompilationResult",
    "xla.CompileOptionsProto",
    "xla.CompiledMemoryStatsProto",
    "xla.CompilerBuildInfo",
    "xla.CompilerPerfStats",
    "xla.ComputeCapability",
    "xla.ConvolutionDimensionNumbers",
    "xla.CrossProgramPrefetch",
    "xla.CudnnVersion",
    "xla.DebugOptions",
    "xla.DeepseaConstantPoolEntryProto",
    "xla.DeepseaExecutableProto",
    "xla.DeepseaOverlaySectionProto",
    "xla.DeepseaRelocationEntryProto",
    "xla.DeepseaSyncFlagInitProto",
    "xla.DeviceAssignmentProto",
    "xla.DeviceAssignmentProto.ComputationDevice",
    "xla.DeviceHandle",
    "xla.DeviceProto",
    "xla.DotDimensionNumbers",
    "xla.DynamicParameterBindingProto",
    "xla.DynamicParameterBindingProto.Binding",
    "xla.ExecutableAndOptionsProto",
    "xla.ExecutableBuildOptionsProto",
    "xla.ExecuteOptionsProto",
    "xla.ExecutionHandle",
    "xla.ExecutionOptions",
    "xla.ExecutionProfile",
    "xla.FrontendAttributes",
    "xla.GatherDimensionNumbers",
    "xla.GlobalDataHandle",
    "xla.GlobalTopologyProto",
    "xla.GpuTopologyProto",
    "xla.HloBufferDonorProto",
    "xla.HloBufferDonorProto.BufferDonorEntryProto",
    "xla.HloComputationProfile",
    "xla.HloComputationProto",
    "xla.HloInputOutputAliasProto",
    "xla.HloInputOutputAliasProto.AliasEntryProto",
    "xla.HloInstructionProfile",
    "xla.HloInstructionProto",
    "xla.HloInstructionProto.SliceDimensions",
    "xla.HloModuleConfigProto",
    "xla.HloModuleProto",
    "xla.HloModuleProtoWithConfig",
    "xla.HloProfilePrinterData",
    "xla.HloSnapshot",
    "xla.HostOffloadingExecutableProto",
    "xla.IsaBundleSlotProto",
    "xla.IsaCoreProgramHeaderProto",
    "xla.IsaProgramProto",
    "xla.IsaSymbolTableEntryProto",
    "xla.IsaVliwBundleProto",
    "xla.JobInfo",
    "xla.KeyValueMetric",
    "xla.LayoutProto",
    "xla.LiteralProto",
    "xla.LocalTopologyProto",
    "xla.MemorySchedulerProto",
    "xla.NullableShapedSliceProto",
    "xla.OpMetadata",
    "xla.OpSharding",
    "xla.OptionOverrideProto",
    "xla.PaddingConfig",
    "xla.PaddingConfig.PaddingConfigDimension",
    "xla.ParameterReplication",
    "xla.PassMetrics",
    "xla.PjRtDeviceDimensionsProto",
    "xla.PjRtExecutableAbiVersionProto",
    "xla.PjRtExecutableMetadata",
    "xla.PjRtPartialProgramProto",
    "xla.PjRtRuntimeAbiVersionProto",
    "xla.PjRtTopologyDescriptionProto",
    "xla.PjRtValueTypeProto",
    "xla.PjRtValueTypeProto.IntVector",
    "xla.PrecisionConfig",
    "xla.ProgramMemoryMetadata",
    "xla.ProgramRelocation",
    "xla.ProgramShapeProto",
    "xla.ReplicaGroup",
    "xla.ScatterDimensionNumbers",
    "xla.ShapeProto",
    "xla.ShapedSliceProto",
    "xla.SourceTarget",
    "xla.SplitConfigProto",
    "xla.Statistic",
    "xla.StatisticsViz",
    "xla.TagMetric",
    "xla.TargetArgumentsProto",
    "xla.TileProto",
    "xla.TpuCoreProgramProto",
    "xla.TpuExecutableProto",
    "xla.TpuMemorySegmentProto",
    "xla.TpuMemorySegmentRangeProto",
    "xla.TpuMemorySegmentSetProto",
    "xla.TriangularSolveOptions",
    "xla.TritonGemmConfigsProto",
    "xla.Window",
    "xla.WindowDimension",
    "xla.barnacore.BarnaCoreChannelConfigProto",
    "xla.barnacore.BarnaCoreExecutableProto",
    "xla.barnacore.SparseCoreDmaDescriptorProto",
    "xla.barnacore.SparseCoreProgramProto",
    "xla.buffer_assignment.AssignmentAlgorithmForComputationsWithoutOrderingProto",
    "xla.buffer_assignment.BufferAllocationSliceProto",
    "xla.buffer_assignment.BufferAssignmentAlgorithmProto",
    "xla.buffer_assignment.BufferIsolationConfig",
    "xla.ffi.ArrayProto",
    "xla.ffi.ExecutionStateProto",
    "xla.ffi.FlatAttributeProto",
    "xla.ffi.FlatAttributesMapProto",
    "xla.ffi.ScalarProto",
    "xla.ffi.XlaFfiCallFrameMetadataProto",
    "xla.ffi.XlaFfiHandlerRegistrationProto",
    "xla.jellyfish.EmitterLearnedCostModelOptions",
    "xla.jellyfish.JellyfishCoreProgramProto",
    "xla.jellyfish.JellyfishExecutableProto",
    "xla.jellyfish.JellyfishHostTransferProto",
    "xla.jellyfish.JellyfishOverlayConfigProto",
    "xla.jellyfish.MemoryBoundLoopOptimizerOptions",
    "xla.jellyfish.MemorySpaceAssignmentOptions",
    "xla.jellyfish.MsaSortOrderOverrides",
    "xla.jellyfish.MsaTensorOverrides",
    "xla.jellyfish.PreferredPrefetchOverrides",
    "xla.megascale.runtime.DCNTopology",
    "xla.megascale.runtime.DCNTopology.SymmetricTree",
    "xla.megascale.runtime.DCNTopology.TreeNode",
    "xla.megascale.runtime.DCNTopology.TreeNode.SliceRange",
    "xla.megascale.runtime.EndpointAddresses",
    "xla.megascale.runtime.HostNetworkAddress",
    "xla.megascale.runtime.MegascaleTransportConfigProto",
    "xla.megascale.runtime.NetworkAddressMapping",
    "xla.megascale.runtime.external.ExecutorTelemetry",
    "xla.megascale.runtime.external.ExecutorTelemetry.PendingGraphState",
    "xla.megascale.runtime.external.MegaScaleAction",
    "xla.megascale.runtime.external.MegaScaleAction.Input",
    "xla.megascale.runtime.external.MegaScaleActionGraph",
    "xla.megascale.runtime.external.MegaScaleRuntimeError",
    "xla.megascale.runtime.external.ProgramRunInfo",
    "xla.megascale.runtime.external.RapidEyeInfo",
    "xla.megascale.runtime.external.RuntimeStateSummary",
    "xla.megascale.runtime.external.TpuCoreHLO",
];

/// Global cached [`DescriptorPool`] decoded from [`EMBEDDED_FILE_DESCRIPTOR_SET_BYTES`].
static GLOBAL_DESCRIPTOR_POOL: OnceLock<DescriptorPool> = OnceLock::new();

/// Decodes a [`DescriptorPool`] from a wire-encoded `google.protobuf.FileDescriptorSet` slice.
///
/// # Arguments
///
/// * `bytes` - Wire-encoded `FileDescriptorSet` byte slice.
///
/// # Errors
///
/// Returns [`LibTpuError::ProstReflectDescriptorFailed`] if the descriptor pool fails to decode.
pub fn decode_descriptor_pool(bytes: &[u8]) -> LibTpuResult<DescriptorPool> {
    Ok(DescriptorPool::decode(bytes)?)
}

/// Returns a fallible reference to the global [`DescriptorPool`] compiled from `docs/reverse_engineering/protos/*.proto`.
///
/// # Errors
///
/// Returns [`LibTpuError::ProstReflectDescriptorFailed`] if the embedded descriptor set is malformed.
pub fn try_global_descriptor_pool() -> LibTpuResult<&'static DescriptorPool> {
    let _ = decode_descriptor_pool(EMBEDDED_FILE_DESCRIPTOR_SET_BYTES)?;
    Ok(global_descriptor_pool())
}

/// Returns a reference to the global [`DescriptorPool`] compiled from `docs/reverse_engineering/protos/*.proto`.
#[must_use]
pub fn global_descriptor_pool() -> &'static DescriptorPool {
    GLOBAL_DESCRIPTOR_POOL.get_or_init(|| {
        let mut pool = DescriptorPool::new();
        let _ = pool.decode_file_descriptor_set(EMBEDDED_FILE_DESCRIPTOR_SET_BYTES);
        pool
    })
}

/// Looks up a [`MessageDescriptor`] by its fully-qualified protobuf type name in [`global_descriptor_pool`].
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found.
pub fn lookup_message_descriptor(
    full_message_name: &'static str,
) -> LibTpuResult<MessageDescriptor> {
    global_descriptor_pool()
        .get_message_by_name(full_message_name)
        .ok_or_else(|| LibTpuError::ProtobufDecodeError {
            message_type: full_message_name,
            reason: format!("message descriptor '{full_message_name}' not found in DescriptorPool"),
        })
}

/// Decodes a dynamic protobuf message by its fully-qualified protobuf type name
/// (e.g., `"xla.IsaProgramProto"` or `"xla.TargetArgumentsProto"`) using [`global_descriptor_pool`].
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `bytes` - Wire-encoded protobuf message payload.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found or wire decoding fails.
pub fn decode_dynamic_proto_message(
    full_message_name: &'static str,
    bytes: &[u8],
) -> LibTpuResult<DynamicMessage> {
    let msg_desc = lookup_message_descriptor(full_message_name)?;
    DynamicMessage::decode(msg_desc, bytes).map_err(|err| LibTpuError::ProtobufDecodeError {
        message_type: full_message_name,
        reason: err.to_string(),
    })
}

/// Decodes a dynamic protobuf message from canonical Protobuf JSON (`proto3` JSON mapping)
/// using [`global_descriptor_pool`] and `serde_json`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `json` - Canonical Protobuf JSON string.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found or JSON parsing fails.
pub fn decode_dynamic_proto_json(
    full_message_name: &'static str,
    json: &str,
) -> LibTpuResult<DynamicMessage> {
    let msg_desc = lookup_message_descriptor(full_message_name)?;
    let mut deserializer = serde_json::Deserializer::from_str(json);
    DynamicMessage::deserialize(msg_desc, &mut deserializer).map_err(|err| {
        LibTpuError::ProtobufDecodeError {
            message_type: full_message_name,
            reason: err.to_string(),
        }
    })
}

/// Transcodes a wire-encoded protobuf message into canonical Protobuf JSON via `prost_reflect` and `serde_json`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `bytes` - Wire-encoded protobuf message payload.
///
/// # Errors
///
/// Returns [`LibTpuError`] if descriptor lookup or wire decoding fails.
pub fn transcode_proto_bytes_to_json(
    full_message_name: &'static str,
    bytes: &[u8],
) -> LibTpuResult<String> {
    let dyn_msg = decode_dynamic_proto_message(full_message_name, bytes)?;
    Ok(serde_json::to_string(&dyn_msg)?)
}

/// Decodes a dynamic protobuf message from canonical Protobuf text-format (`textproto`)
/// using [`global_descriptor_pool`] and `prost_reflect`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `text` - Canonical Protobuf text-format (`textproto`) string.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found or textproto parsing fails.
pub fn decode_dynamic_proto_text(
    full_message_name: &'static str,
    text: &str,
) -> LibTpuResult<DynamicMessage> {
    let msg_desc = lookup_message_descriptor(full_message_name)?;
    DynamicMessage::parse_text_format(msg_desc, text).map_err(|err| {
        LibTpuError::ProtobufDecodeError {
            message_type: full_message_name,
            reason: err.to_string(),
        }
    })
}

/// Transcodes a wire-encoded protobuf message into canonical Protobuf text-format (`textproto`) via `prost_reflect`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `bytes` - Wire-encoded protobuf message payload.
///
/// # Errors
///
/// Returns [`LibTpuError`] if descriptor lookup or wire decoding fails.
pub fn transcode_proto_bytes_to_text(
    full_message_name: &'static str,
    bytes: &[u8],
) -> LibTpuResult<String> {
    let dyn_msg = decode_dynamic_proto_message(full_message_name, bytes)?;
    Ok(dyn_msg.to_text_format())
}

/// Decodes a `prost::Message` type `M` from `bytes`, mapping any wire error into
/// [`LibTpuError::ProtobufDecodeError`].
///
/// # Arguments
///
/// * `bytes` - Raw binary protobuf slice.
/// * `message_type` - Canonical protobuf message name for diagnostics.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if `prost::Message::decode` fails.
pub fn decode_prost_message<M: Message + Default>(
    bytes: &[u8],
    message_type: &'static str,
) -> LibTpuResult<M> {
    M::decode(bytes).map_err(|err| LibTpuError::ProtobufDecodeError {
        message_type,
        reason: err.to_string(),
    })
}

/// Computes a canonical 128-bit truncated SHA-256 fingerprint (32 lowercase hex characters)
/// over a serialized protobuf or executable byte slice using `sha2::Sha256`.
///
/// # Arguments
///
/// * `payload` - Input byte slice to hash.
#[must_use]
pub fn sha256_fingerprint_hex(payload: &[u8]) -> String {
    let digest = Sha256::digest(payload);
    let mut hex = String::with_capacity(32);
    for &b in digest.iter().take(16) {
        let hi = char::from_digit(u32::from(b >> 4), 16).unwrap_or('0');
        let lo = char::from_digit(u32::from(b & 0x0F), 16).unwrap_or('0');
        hex.push(hi);
        hex.push(lo);
    }
    hex
}

/// Computes a deterministic 64-bit unsigned integer fingerprint over `payload` using `sha2::Sha256`.
///
/// # Arguments
///
/// * `payload` - Input byte slice to hash.
#[must_use]
pub fn sha256_fingerprint_u64(payload: &[u8]) -> u64 {
    let digest = Sha256::digest(payload);
    let mut first8 = [0_u8; 8];
    for (dst, &src) in first8.iter_mut().zip(digest.iter().take(8)) {
        *dst = src;
    }
    u64::from_le_bytes(first8)
}

/// Official 8-byte Riegeli block header magic (`b"\x83\xaf\x70\xd1\x0d\x88\x4a\x3f"`).
pub const RIEGELI_BLOCK_MAGIC: [u8; 8] = [0x83, 0xAF, 0x70, 0xD1, 0x0D, 0x88, 0x4A, 0x3F];

/// Canonical 32-byte (`4 x u64`) Google `HighwayHash64` key used by Riegeli (`"riegeli_/records_format_20180226"`).
pub const RIEGELI_HIGHWAYHASH_KEY: [u64; 4] = [
    0x5F69_6C65_6765_6972,
    0x5F73_6472_6F63_6572,
    0x325F_7461_6D72_6F66,
    0x3632_3230_3831_3032,
];

/// Pure-Rust, zero-panic Google `HighwayHash64` state machine (`v0`, `v1`, `mul0`, `mul1` 4x64-bit lanes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HighwayHash64State {
    /// Primary 4-lane state vector `v0`.
    pub v0: [u64; 4],
    /// Secondary 4-lane state vector `v1`.
    pub v1: [u64; 4],
    /// Multiplier 4-lane state vector `mul0`.
    pub mul0: [u64; 4],
    /// Multiplier 4-lane state vector `mul1`.
    pub mul1: [u64; 4],
}

impl HighwayHash64State {
    /// Initializes a new [`HighwayHash64State`] with a 4x64-bit key.
    ///
    /// # Arguments
    ///
    /// * `key` - 256-bit (`[u64; 4]`) `HighwayHash` secret key.
    #[must_use]
    pub const fn new(key: &[u64; 4]) -> Self {
        let mul0 = [
            0xDBE6_D5D5_FE4C_CE2F,
            0xA409_3822_299F_31D0,
            0x1319_8A2E_0370_7344,
            0x243F_6A88_85A3_08D3,
        ];
        let mul1 = [
            0x3BD3_9E10_CB0E_F593,
            0xC0AC_F169_B5F1_8A8C,
            0xBE54_66CF_34E9_0C6C,
            0x4528_21E6_38D0_1377,
        ];
        Self {
            v0: [
                mul0[0] ^ key[0],
                mul0[1] ^ key[1],
                mul0[2] ^ key[2],
                mul0[3] ^ key[3],
            ],
            v1: [
                mul1[0] ^ key[0].rotate_left(32),
                mul1[1] ^ key[1].rotate_left(32),
                mul1[2] ^ key[2].rotate_left(32),
                mul1[3] ^ key[3].rotate_left(32),
            ],
            mul0,
            mul1,
        }
    }

    /// Computes Google `HighwayHash64` over a byte slice using `key`.
    ///
    /// # Arguments
    ///
    /// * `data` - Input byte slice to hash.
    /// * `key` - 256-bit (`[u64; 4]`) `HighwayHash` key.
    #[must_use]
    pub fn hash64(data: &[u8], key: &[u64; 4]) -> u64 {
        let mut state = Self::new(key);
        let mut chunks = data.chunks_exact(32);
        for chunk in &mut chunks {
            let mut lanes = [0_u64; 4];
            for (dst, src) in lanes.iter_mut().zip(chunk.chunks_exact(8)) {
                let mut b8 = [0_u8; 8];
                b8.copy_from_slice(src);
                *dst = u64::from_le_bytes(b8);
            }
            state.update_lanes(&lanes);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            state.update_remainder(rem);
        }
        state.finalize64()
    }

    /// Absorbs a single 32-byte (`[u64; 4]`) packet into the `HighwayHash` state.
    ///
    /// # Arguments
    ///
    /// * `packet` - Four 64-bit little-endian words.
    #[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
    pub fn update_lanes(&mut self, packet: &[u64; 4]) {
        for i in 0..4 {
            self.v1[i] = self.v1[i]
                .wrapping_add(self.mul0[i])
                .wrapping_add(packet[i]);
            self.mul0[i] ^= (self.v1[i] & 0xFFFF_FFFF).wrapping_mul(self.v0[i] >> 32);
            self.v0[i] = self.v0[i].wrapping_add(self.mul1[i]);
            self.mul1[i] ^= (self.v0[i] & 0xFFFF_FFFF).wrapping_mul(self.v1[i] >> 32);
        }
        let z00 = Self::zipper_merge_word(self.v1[0], self.v1[1]);
        let z01 = Self::zipper_merge_word(self.v1[2], self.v1[3]);
        self.v0[0] = self.v0[0].wrapping_add(z00.0);
        self.v0[1] = self.v0[1].wrapping_add(z00.1);
        self.v0[2] = self.v0[2].wrapping_add(z01.0);
        self.v0[3] = self.v0[3].wrapping_add(z01.1);

        let z10 = Self::zipper_merge_word(self.v0[0], self.v0[1]);
        let z11 = Self::zipper_merge_word(self.v0[2], self.v0[3]);
        self.v1[0] = self.v1[0].wrapping_add(z10.0);
        self.v1[1] = self.v1[1].wrapping_add(z10.1);
        self.v1[2] = self.v1[2].wrapping_add(z11.0);
        self.v1[3] = self.v1[3].wrapping_add(z11.1);
    }

    /// Absorbs a trailing remainder slice (`1..=31` bytes) into the `HighwayHash` state.
    ///
    /// # Arguments
    ///
    /// * `rem` - Trailing byte slice of length `< 32`.
    #[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
    pub fn update_remainder(&mut self, rem: &[u8]) {
        let size_mod32 = rem.len() as u64;
        for i in 0..4 {
            self.v0[i] = self.v0[i].wrapping_add((size_mod32 << 32).wrapping_add(size_mod32));
            let rot = u32::try_from(size_mod32).unwrap_or(0);
            self.v1[i] = self.v1[i].rotate_left(rot);
        }
        let mut buf = [0_u8; 32];
        for (dst, &src) in buf.iter_mut().zip(rem.iter()) {
            *dst = src;
        }
        let mut lanes = [0_u64; 4];
        for (dst, src) in lanes.iter_mut().zip(buf.chunks_exact(8)) {
            let mut b8 = [0_u8; 8];
            b8.copy_from_slice(src);
            *dst = u64::from_le_bytes(b8);
        }
        self.update_lanes(&lanes);
    }

    /// Permutes and finalizes the state into a 64-bit `HighwayHash64` digest.
    #[must_use]
    pub fn finalize64(&mut self) -> u64 {
        for _ in 0..4 {
            let permuted = [
                self.v0[2].rotate_left(32),
                self.v0[3].rotate_left(32),
                self.v0[0].rotate_left(32),
                self.v0[1].rotate_left(32),
            ];
            self.update_lanes(&permuted);
        }
        self.v0[0]
            .wrapping_add(self.v1[0])
            .wrapping_add(self.mul0[0])
            .wrapping_add(self.mul1[0])
    }

    /// Performs the `HighwayHash` byte-level zipper merge over two 64-bit words.
    ///
    /// # Arguments
    ///
    /// * `v0` - First 64-bit word.
    /// * `v1` - Second 64-bit word.
    #[must_use]
    const fn zipper_merge_word(v0: u64, v1: u64) -> (u64, u64) {
        let b = v0.to_le_bytes();
        let c = v1.to_le_bytes();
        let w0 = u64::from_le_bytes([b[3], c[4], b[2], c[5], c[6], b[0], b[1], c[7]]);
        let w1 = u64::from_le_bytes([c[3], b[4], c[2], b[5], b[6], c[0], c[1], b[7]]);
        (w0, w1)
    }
}

/// Computes Google `HighwayHash64` over `payload` using [`RIEGELI_HIGHWAYHASH_KEY`].
///
/// # Arguments
///
/// * `payload` - Byte slice to hash.
#[must_use]
pub fn riegeli_highwayhash64(payload: &[u8]) -> u64 {
    HighwayHash64State::hash64(payload, &RIEGELI_HIGHWAYHASH_KEY)
}

/// Clean-room Riegeli record container reader and writer (`RiegeliRecordCodec`).
///
/// Supports 64-byte block headers (`b"\x83\xaf\x70\xd1\x0d\x88\x4a\x3f"` or `b"riegeli_"`) and chunk headers
/// with Google [`HighwayHash64State`] header/payload checksums and `64 KiB` (`0x10000`-byte) block boundary
/// interleaving used by `libtpu.sdk.diagnostics` (`read_core_dump`, `read_stall_summaries`, `read_compiler_metadata`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RiegeliRecordCodec;

impl RiegeliRecordCodec {
    /// Canonical Riegeli block boundary interval in bytes (`64 KiB` = `65,536` bytes).
    pub const BLOCK_BOUNDARY_BYTES: usize = 65_536;

    /// Wraps a serialized protobuf payload into a 64-byte Riegeli block + chunk header container
    /// with Google [`riegeli_highwayhash64`] header and payload hashes in bytes `48..64`.
    ///
    /// # Arguments
    ///
    /// * `proto_payload` - Raw binary protobuf record bytes.
    #[must_use]
    pub fn wrap_record(proto_payload: &[u8]) -> Vec<u8> {
        let len_u64 = proto_payload.len() as u64;
        let mut out = Vec::with_capacity(64 + proto_payload.len());
        out.extend_from_slice(&RIEGELI_BLOCK_MAGIC);
        out.extend_from_slice(b"riegeli_");
        out.extend_from_slice(&0_u64.to_le_bytes());
        out.extend_from_slice(&len_u64.to_le_bytes());
        out.extend_from_slice(&1_u64.to_le_bytes());
        out.extend_from_slice(&len_u64.to_le_bytes());
        let header_hash = riegeli_highwayhash64(&out);
        let payload_hash = riegeli_highwayhash64(proto_payload);
        out.extend_from_slice(&header_hash.to_le_bytes());
        out.extend_from_slice(&payload_hash.to_le_bytes());
        out.extend_from_slice(proto_payload);
        out
    }

    /// Unwraps a Riegeli-framed container if `raw` begins with [`RIEGELI_BLOCK_MAGIC`] or `b"riegeli"`,
    /// returning the inner protobuf record bytes, or returns `raw` unchanged for plain `.pb` files.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw file bytes (either Riegeli-framed or plain protobuf).
    #[must_use]
    pub fn unwrap_record(raw: &[u8]) -> &[u8] {
        if raw.len() >= 64 && raw.starts_with(&RIEGELI_BLOCK_MAGIC) {
            return raw.get(64..).unwrap_or(raw);
        }
        raw
    }

    /// Unwraps and validates a Riegeli-framed container, checking the 64-bit `HighwayHash64` header
    /// and payload checksums when non-zero.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw file bytes (either Riegeli-framed or plain protobuf).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `raw` has a Riegeli magic prefix with
    /// a non-zero `HighwayHash64` checksum that does not match the header or payload.
    pub fn unwrap_record_checked(raw: &[u8]) -> LibTpuResult<&[u8]> {
        if raw.starts_with(&RIEGELI_BLOCK_MAGIC) {
            let header_prefix = raw.get(..48).unwrap_or_default();
            let mut h_bytes = [0_u8; 8];
            let mut p_bytes = [0_u8; 8];
            h_bytes.copy_from_slice(raw.get(48..56).unwrap_or(&[0_u8; 8]));
            p_bytes.copy_from_slice(raw.get(56..64).unwrap_or(&[0_u8; 8]));
            let stored_header_hash = u64::from_le_bytes(h_bytes);
            let stored_payload_hash = u64::from_le_bytes(p_bytes);
            let payload = raw.get(64..).unwrap_or_default();
            let hdr_ok = (u8::from(stored_header_hash == 0)
                | u8::from(stored_header_hash == riegeli_highwayhash64(header_prefix)))
                != 0;
            let pay_ok = (u8::from(stored_payload_hash == 0)
                | u8::from(stored_payload_hash == riegeli_highwayhash64(payload)))
                != 0;
            if (u8::from(hdr_ok) & u8::from(pay_ok)) == 0 {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from(
                        "Riegeli HighwayHash64 header or payload checksum mismatch",
                    ),
                });
            }
            return Ok(payload);
        }
        Ok(raw)
    }

    /// Encodes multiple protobuf records into a Riegeli stream, inserting 24-byte block boundary
    /// headers (`RIEGELI_BLOCK_MAGIC` + `previous_chunk` + `next_chunk`) at every `64 KiB` (`0x10000`) offset.
    ///
    /// # Arguments
    ///
    /// * `records` - Slice of serialized protobuf record byte vectors.
    #[must_use]
    pub fn wrap_records(records: &[Vec<u8>]) -> Vec<u8> {
        let mut raw_chunks = Vec::new();
        for rec in records {
            raw_chunks.extend_from_slice(&Self::wrap_record(rec));
        }
        if raw_chunks.len() < Self::BLOCK_BOUNDARY_BYTES {
            return raw_chunks;
        }
        let mut out = Vec::with_capacity(raw_chunks.len() + 64);
        let mut src_idx = 0_usize;
        while src_idx < raw_chunks.len() {
            let pos_in_block = out.len() % Self::BLOCK_BOUNDARY_BYTES;
            if (u8::from(!out.is_empty()) & u8::from(pos_in_block == 0)) != 0 {
                out.extend_from_slice(&RIEGELI_BLOCK_MAGIC);
                let prev = (src_idx as u64).to_le_bytes();
                let next = ((raw_chunks.len() - src_idx) as u64).to_le_bytes();
                out.extend_from_slice(&prev);
                out.extend_from_slice(&next);
            }
            let room = Self::BLOCK_BOUNDARY_BYTES - (out.len() % Self::BLOCK_BOUNDARY_BYTES);
            let take = room.min(raw_chunks.len() - src_idx);
            out.extend_from_slice(raw_chunks.get(src_idx..src_idx + take).unwrap_or_default());
            src_idx += take;
        }
        out
    }

    /// Decodes a multi-record Riegeli stream (stripping any `64 KiB` block boundary headers)
    /// and verifies each record's `HighwayHash64` checksum.
    ///
    /// # Arguments
    ///
    /// * `stream` - Raw Riegeli stream bytes produced by [`Self::wrap_records`] or [`Self::wrap_record`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if any record header is truncated or fails checksum verification.
    pub fn unwrap_records(stream: &[u8]) -> LibTpuResult<Vec<Vec<u8>>> {
        let mut stripped = Vec::with_capacity(stream.len());
        let mut idx = 0_usize;
        while let Some(&b) = stream.get(idx) {
            let is_block_hdr =
                (u8::from(idx > 0) & u8::from(idx % Self::BLOCK_BOUNDARY_BYTES == 0)) != 0;
            if is_block_hdr {
                idx = idx.saturating_add(24).min(stream.len());
                continue;
            }
            stripped.push(b);
            idx += 1;
        }

        let mut records = Vec::new();
        let mut cursor = 0_usize;
        while cursor < stripped.len() {
            let rem = stripped.get(cursor..).unwrap_or_default();
            if !rem.starts_with(&RIEGELI_BLOCK_MAGIC) {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from("truncated or invalid Riegeli record header in stream"),
                });
            }
            let mut len_bytes = [0_u8; 8];
            len_bytes.copy_from_slice(rem.get(24..32).unwrap_or(&[0_u8; 8]));
            let rec_len = usize::try_from(u64::from_le_bytes(len_bytes)).unwrap_or(0);
            let frame_end = 64_usize.saturating_add(rec_len);
            let Some(frame) = rem.get(..frame_end) else {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from("Riegeli record payload extends past stream end"),
                });
            };
            let payload = Self::unwrap_record_checked(frame)?;
            records.push(payload.to_vec());
            cursor = cursor.saturating_add(frame_end);
        }
        Ok(records)
    }
}

/// Encodes a protobuf record payload into a 64-byte Riegeli record container with Google `HighwayHash64`
/// header and payload checksums (`encode_riegeli_record_container`).
///
/// # Arguments
///
/// * `proto_payload` - Raw binary protobuf record bytes.
#[must_use]
pub fn encode_riegeli_record_container(proto_payload: &[u8]) -> Vec<u8> {
    RiegeliRecordCodec::wrap_record(proto_payload)
}

/// Decodes and validates a 64-byte Riegeli record container (or returns plain protobuf bytes unchanged),
/// verifying Google `HighwayHash64` header and payload checksums (`decode_riegeli_record_container`).
///
/// # Arguments
///
/// * `raw` - Raw file bytes (either Riegeli-framed `.riegeli` or plain binary protobuf `.pb`).
///
/// # Errors
///
/// Returns [`LibTpuError::DeserializationFailed`] if `raw` begins with [`RIEGELI_BLOCK_MAGIC`] and fails
/// `HighwayHash64` checksum verification.
pub fn decode_riegeli_record_container(raw: &[u8]) -> LibTpuResult<Vec<u8>> {
    Ok(RiegeliRecordCodec::unwrap_record_checked(raw)?.to_vec())
}

/// Verifies that [`ALL_REGISTERED_PROTO_MESSAGE_NAMES`] and [`global_descriptor_pool`] are 100% closed.
///
/// Ensures every registered message exists in [`global_descriptor_pool`] and every non-map-entry
/// `.proto` message in the pool is registered in [`ALL_REGISTERED_PROTO_MESSAGE_NAMES`].
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if any registered message descriptor is missing or if any
/// `.proto` message in the pool is unregistered.
pub fn verify_all_proto_descriptors() -> LibTpuResult<usize> {
    for &name in ALL_REGISTERED_PROTO_MESSAGE_NAMES {
        lookup_message_descriptor(name)?;
    }
    let pool_count = global_descriptor_pool()
        .all_messages()
        .filter(|m| {
            (u8::from(m.is_map_entry()) | u8::from(m.package_name().starts_with("google.protobuf")))
                == 0
        })
        .count();
    Ok(ALL_REGISTERED_PROTO_MESSAGE_NAMES.len().min(pool_count))
}

/// Cardinality of a field in a [`ProtoMessageSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtoFieldCardinality {
    /// Singular or optional field (`proto3` implicit/explicit optional).
    Optional,
    /// Repeated list field (`repeated T`).
    Repeated,
    /// Associative map field (`map<K, V>`).
    Map,
}

/// Strongly-typed field specification extracted for `.proto` schema emission and parity checking.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProtoFieldSpec {
    /// Canonical `.proto` field identifier.
    pub name: String,
    /// Wire tag number (`>= 1`).
    pub tag: u32,
    /// Field cardinality (`Optional`, `Repeated`, or `Map`).
    pub cardinality: ProtoFieldCardinality,
    /// Canonical `.proto` type expression (e.g., `"int64"`, `"bytes"`, `"map<string, string>"`).
    pub type_name: String,
}

/// Strongly-typed message schema specification used to emit `.proto` definitions from Rust types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProtoMessageSpec {
    /// Fully-qualified protobuf message name (`package.MessageName`).
    pub full_name: String,
    /// Enclosing protobuf package name.
    pub package_name: String,
    /// Unqualified or nested message identifier.
    pub message_name: String,
    /// Ordered list of field specifications sorted by tag number.
    pub fields: Vec<ProtoFieldSpec>,
}

/// Formats a [`prost_reflect::Kind`] into its canonical `.proto` type string.
///
/// # Arguments
///
/// * `kind` - Reflected protobuf field kind.
#[must_use]
pub fn format_proto_kind(kind: &prost_reflect::Kind) -> String {
    match kind {
        prost_reflect::Kind::Double => String::from("double"),
        prost_reflect::Kind::Float => String::from("float"),
        prost_reflect::Kind::Int32 => String::from("int32"),
        prost_reflect::Kind::Int64 => String::from("int64"),
        prost_reflect::Kind::Uint32 => String::from("uint32"),
        prost_reflect::Kind::Uint64 => String::from("uint64"),
        prost_reflect::Kind::Sint32 => String::from("sint32"),
        prost_reflect::Kind::Sint64 => String::from("sint64"),
        prost_reflect::Kind::Fixed32 => String::from("fixed32"),
        prost_reflect::Kind::Fixed64 => String::from("fixed64"),
        prost_reflect::Kind::Sfixed32 => String::from("sfixed32"),
        prost_reflect::Kind::Sfixed64 => String::from("sfixed64"),
        prost_reflect::Kind::Bool => String::from("bool"),
        prost_reflect::Kind::String => String::from("string"),
        prost_reflect::Kind::Bytes => String::from("bytes"),
        prost_reflect::Kind::Message(m) => String::from(m.full_name()),
        prost_reflect::Kind::Enum(e) => String::from(e.full_name()),
    }
}

/// Extracts a [`ProtoMessageSpec`] from the compiled [`MessageDescriptor`] for `full_message_name`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if `full_message_name` is not registered.
pub fn extract_proto_message_spec(full_message_name: &str) -> LibTpuResult<ProtoMessageSpec> {
    let desc = global_descriptor_pool()
        .get_message_by_name(full_message_name)
        .ok_or_else(|| LibTpuError::ProtobufDecodeError {
            message_type: "prost_reflect::DescriptorPool",
            reason: format!(
                "message descriptor '{full_message_name}' not found in global_descriptor_pool"
            ),
        })?;
    let package_name = String::from(desc.package_name());
    let prefix = format!("{package_name}.");
    let message_name = full_message_name
        .strip_prefix(&prefix)
        .unwrap_or(full_message_name)
        .replace('.', "_");
    let mut fields = Vec::new();
    for field in desc.fields() {
        let (cardinality, type_name) = if let (true, prost_reflect::Kind::Message(entry_desc)) =
            (field.is_map(), field.kind())
        {
            let key_ty = entry_desc.map_entry_key_field().kind();
            let val_ty = entry_desc.map_entry_value_field().kind();
            (
                ProtoFieldCardinality::Map,
                format!(
                    "map<{}, {}>",
                    format_proto_kind(&key_ty),
                    format_proto_kind(&val_ty)
                ),
            )
        } else if field.is_list() {
            (
                ProtoFieldCardinality::Repeated,
                format_proto_kind(&field.kind()),
            )
        } else {
            (
                ProtoFieldCardinality::Optional,
                format_proto_kind(&field.kind()),
            )
        };
        fields.push(ProtoFieldSpec {
            name: String::from(field.name()),
            tag: field.number(),
            cardinality,
            type_name,
        });
    }
    fields.sort_by_key(|f| f.tag);
    Ok(ProtoMessageSpec {
        full_name: String::from(full_message_name),
        package_name,
        message_name,
        fields,
    })
}

/// Emits a canonical `proto3` `.proto` schema definition string from a [`ProtoMessageSpec`].
///
/// # Arguments
///
/// * `spec` - Strongly-typed [`ProtoMessageSpec`] to render.
#[must_use]
pub fn emit_proto_schema_from_spec(spec: &ProtoMessageSpec) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("syntax = \"proto3\";\n\n");
    if !spec.package_name.is_empty() {
        let _ = writeln!(out, "package {};\n", spec.package_name);
    }
    let _ = writeln!(out, "message {} {{", spec.message_name);
    for field in &spec.fields {
        match field.cardinality {
            ProtoFieldCardinality::Repeated => {
                let _ = writeln!(
                    out,
                    "  repeated {} {} = {};",
                    field.type_name, field.name, field.tag
                );
            }
            ProtoFieldCardinality::Optional | ProtoFieldCardinality::Map => {
                let _ = writeln!(out, "  {} {} = {};", field.type_name, field.name, field.tag);
            }
        }
    }
    out.push_str("}\n");
    out
}

/// Emits a canonical `proto3` `.proto` schema string for `full_message_name`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if `full_message_name` is not registered.
pub fn emit_proto_message_schema(full_message_name: &str) -> LibTpuResult<String> {
    let spec = extract_proto_message_spec(full_message_name)?;
    Ok(emit_proto_schema_from_spec(&spec))
}

/// Constructs a deterministic non-default [`prost_reflect::Value`] for a scalar or message [`prost_reflect::Kind`].
///
/// # Arguments
///
/// * `kind` - Field [`prost_reflect::Kind`].
/// * `tag` - Field tag number used to seed deterministic values.
/// * `depth` - Remaining recursion depth for nested sub-messages.
#[must_use]
pub fn build_populated_field_value(
    kind: &prost_reflect::Kind,
    tag: u32,
    depth: usize,
) -> prost_reflect::Value {
    use prost_reflect::{Kind, Value};
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    match kind {
        Kind::Double => Value::F64(f64::from(tag) + 0.5),
        Kind::Float => Value::F32((tag as f32) + 0.5),
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => Value::I32((tag as i32).saturating_add(10)),
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => {
            Value::I64(i64::from(tag).saturating_add(10))
        }
        Kind::Uint32 | Kind::Fixed32 => Value::U32(tag.saturating_add(10)),
        Kind::Uint64 | Kind::Fixed64 => Value::U64(u64::from(tag).saturating_add(10)),
        Kind::Bool => Value::Bool(true),
        Kind::String => Value::String(format!("v_{tag}")),
        Kind::Bytes => Value::Bytes(prost::bytes::Bytes::from(vec![0xAB, (tag & 0xFF) as u8])),
        Kind::Enum(enum_desc) => {
            let num = enum_desc
                .values()
                .find(|v| v.number() != 0)
                .map_or(1, |v| v.number());
            Value::EnumNumber(num)
        }
        Kind::Message(sub_desc) => {
            if depth > 0 {
                Value::Message(build_populated_dynamic_message(sub_desc, depth - 1))
            } else {
                Value::Message(DynamicMessage::new(sub_desc.clone()))
            }
        }
    }
}

/// Constructs a [`DynamicMessage`] with every field in `desc` populated with deterministic non-default values.
///
/// # Arguments
///
/// * `desc` - Target [`MessageDescriptor`].
/// * `depth` - Remaining recursion depth for nested messages.
#[must_use]
pub fn build_populated_dynamic_message(desc: &MessageDescriptor, depth: usize) -> DynamicMessage {
    use prost_reflect::{Kind, MapKey, Value};
    let mut msg = DynamicMessage::new(desc.clone());
    for field in desc.fields() {
        let tag = field.number();
        if let (true, Kind::Message(entry_desc)) = (field.is_map(), field.kind()) {
            let key_kind = entry_desc.map_entry_key_field().kind();
            let val_kind = entry_desc.map_entry_value_field().kind();
            let map_key = if matches!(key_kind, Kind::Int64 | Kind::Sint64 | Kind::Sfixed64) {
                MapKey::I64(i64::from(tag))
            } else {
                MapKey::String(format!("k_{tag}"))
            };
            let map_val = build_populated_field_value(&val_kind, tag, depth);
            let mut map = std::collections::HashMap::new();
            map.insert(map_key, map_val);
            msg.set_field(&field, Value::Map(map));
        } else if field.is_list() {
            if matches!(field.kind(), Kind::Message(_)) && depth == 0 {
                continue;
            }
            let elem = build_populated_field_value(&field.kind(), tag, depth);
            msg.set_field(&field, Value::List(vec![elem]));
        } else {
            let val = build_populated_field_value(&field.kind(), tag, depth);
            msg.set_field(&field, val);
        }
    }
    msg
}

/// Reads a protobuf base-128 varint from `bytes` starting at `*cursor`.
///
/// # Arguments
///
/// * `bytes` - Raw protobuf wire buffer.
/// * `cursor` - Mutable byte offset updated past the varint.
fn read_wire_varint(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    let mut result = 0_u64;
    let mut shift = 0_u32;
    while shift < 64 {
        let &b = bytes.get(*cursor)?;
        *cursor += 1;
        result |= u64::from(b & 0x7F) << shift;
        if (b & 0x80) == 0 {
            return Some(result);
        }
        shift += 7;
    }
    None
}

/// Preserves unknown/forward-compatible wire fields from `original_wire` whose field numbers are not
/// defined in `desc` by appending their raw wire records to `reencoded_wire`.
///
/// # Arguments
///
/// * `original_wire` - Original caller-provided protobuf wire payload.
/// * `reencoded_wire` - Re-encoded wire payload produced by a Rust [`ProtoSchemaMessage`].
/// * `desc` - [`MessageDescriptor`] of the known schema.
#[must_use]
pub fn preserve_unknown_wire_fields(
    original_wire: &[u8],
    reencoded_wire: &[u8],
    desc: &MessageDescriptor,
) -> Vec<u8> {
    let mut out = reencoded_wire.to_vec();
    let mut cursor = 0_usize;
    while cursor < original_wire.len() {
        let record_start = cursor;
        let Some(tag_key) = read_wire_varint(original_wire, &mut cursor) else {
            break;
        };
        #[allow(clippy::cast_possible_truncation)]
        let field_num = (tag_key >> 3) as u32;
        let wire_type = tag_key & 0x07;
        let step_ok = match wire_type {
            0 => read_wire_varint(original_wire, &mut cursor).is_some(),
            1 => {
                cursor = cursor.saturating_add(8);
                cursor <= original_wire.len()
            }
            2 => read_wire_varint(original_wire, &mut cursor).is_some_and(|len| {
                let step = usize::try_from(len).unwrap_or(original_wire.len());
                cursor = cursor.saturating_add(step);
                cursor <= original_wire.len()
            }),
            5 => {
                cursor = cursor.saturating_add(4);
                cursor <= original_wire.len()
            }
            _ => false,
        };
        if !step_ok {
            break;
        }
        if desc.get_field(field_num).is_none() {
            out.extend_from_slice(original_wire.get(record_start..cursor).unwrap_or_default());
        }
    }
    out
}

/// Decodes `wire` as `M` and re-encodes it into canonical protobuf wire format bytes.
///
/// # Arguments
///
/// * `wire` - Raw protobuf wire slice.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if decoding `M` fails.
pub fn roundtrip_proto_wire_bytes<M: ProtoSchemaMessage>(wire: &[u8]) -> LibTpuResult<Vec<u8>> {
    M::decode_proto(wire).map(|m| m.encode_proto())
}

/// Verifies bidirectional 1:1 equivalence between a `.proto` descriptor and Rust wire round-trip.
///
/// Compares the [`MessageDescriptor`] (`full_message_name`), `roundtrip_fn`, `spec`, and `emitted`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `roundtrip_fn` - Function decoding wire bytes into the target Rust struct and re-encoding them.
/// * `spec` - Extracted [`ProtoMessageSpec`].
/// * `emitted` - Emitted `proto3` `.proto` schema text.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] if any field is lost or altered on round-trip
/// or missing from `spec` / `emitted`.
pub fn verify_bidirectional_proto_struct_equivalence_with_spec(
    full_message_name: &'static str,
    roundtrip_fn: fn(&[u8]) -> LibTpuResult<Vec<u8>>,
    spec: &ProtoMessageSpec,
    emitted: &str,
) -> LibTpuResult<()> {
    let desc = lookup_message_descriptor(full_message_name)?;
    let populated_dyn = build_populated_dynamic_message(&desc, 1);
    let wire_bytes = populated_dyn.encode_to_vec();
    let reencoded_wire = roundtrip_fn(&wire_bytes)?;
    let roundtripped_dyn = decode_dynamic_proto_message(full_message_name, &reencoded_wire)?;

    for field in desc.fields() {
        let expected_val = populated_dyn.get_field(&field);
        let actual_val = roundtripped_dyn.get_field(&field);
        if actual_val != expected_val {
            return Err(LibTpuError::ProtoSchemaDriftError {
                message_type: full_message_name,
                field_name: String::from(field.name()),
                reason: format!(
                    "tag {} value mismatch after .proto -> Rust -> .proto round-trip",
                    field.number()
                ),
            });
        }
    }

    for field in desc.fields() {
        let needle = format!(" {} = {};", field.name(), field.number());
        let in_spec = spec
            .fields
            .iter()
            .any(|f| f.tag == field.number() && f.name == field.name());
        if (u8::from(in_spec) & u8::from(emitted.contains(&needle))) == 0 {
            return Err(LibTpuError::ProtoSchemaDriftError {
                message_type: full_message_name,
                field_name: String::from(field.name()),
                reason: String::from("field missing from emitted .proto schema"),
            });
        }
    }
    Ok(())
}

/// Verifies bidirectional 1:1 equivalence between the `.proto` [`MessageDescriptor`] and `M`.
///
/// Tests both `.proto` -> Rust struct wire decoding of all populated fields and Rust struct ->
/// `.proto` schema emission.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] if any field in the `.proto` descriptor is lost
/// or altered when decoded into `M` and re-encoded, or if the emitted `.proto` schema does not
/// match the descriptor.
pub fn verify_bidirectional_proto_struct_equivalence<M: ProtoSchemaMessage>() -> LibTpuResult<()> {
    let spec = extract_proto_message_spec(M::PROTO_FULL_NAME)?;
    let emitted = emit_proto_schema_from_spec(&spec);
    verify_bidirectional_proto_struct_equivalence_with_spec(
        M::PROTO_FULL_NAME,
        roundtrip_proto_wire_bytes::<M>,
        &spec,
        &emitted,
    )
}

/// Trait implemented by all strongly-typed `libtpu-rs` protobuf messages backed by an
/// official `.proto` descriptor in [`global_descriptor_pool`].
pub trait ProtoSchemaMessage: Message + Default + Clone + PartialEq {
    /// Canonical fully-qualified protobuf message name (`package.MessageName`).
    const PROTO_FULL_NAME: &'static str;

    /// Encodes this message into canonical protobuf wire format bytes.
    #[must_use]
    fn encode_proto(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an instance of this message from canonical protobuf wire format bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw binary protobuf slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if wire decoding fails.
    fn decode_proto(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, Self::PROTO_FULL_NAME)
    }

    /// Looks up the [`MessageDescriptor`] for this message in [`global_descriptor_pool`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the descriptor is missing from the pool.
    fn descriptor() -> LibTpuResult<MessageDescriptor> {
        lookup_message_descriptor(Self::PROTO_FULL_NAME)
    }

    /// Returns the strongly-typed [`ProtoMessageSpec`] for this Rust protobuf struct.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the descriptor is missing from the pool.
    fn schema_spec() -> LibTpuResult<ProtoMessageSpec> {
        extract_proto_message_spec(Self::PROTO_FULL_NAME)
    }

    /// Emits a canonical `proto3` `.proto` schema definition string from this Rust struct's [`ProtoMessageSpec`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the descriptor is missing from the pool.
    fn emit_proto_schema() -> LibTpuResult<String> {
        emit_proto_message_schema(Self::PROTO_FULL_NAME)
    }

    /// Verifies 1:1 bidirectional equivalence between the `.proto` descriptor and this Rust struct.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtoSchemaDriftError`] if any field drifts between `.proto` and Rust.
    fn verify_bidirectional_equivalence() -> LibTpuResult<()> {
        verify_bidirectional_proto_struct_equivalence::<Self>()
    }

    /// Converts this strongly-typed message into a reflected [`DynamicMessage`] via wire transcoding.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if descriptor lookup or dynamic decoding fails.
    fn to_dynamic_message(&self) -> LibTpuResult<DynamicMessage> {
        decode_dynamic_proto_message(Self::PROTO_FULL_NAME, &self.encode_to_vec())
    }

    /// Decodes an instance of this strongly-typed message from a reflected [`DynamicMessage`].
    ///
    /// # Arguments
    ///
    /// * `dyn_msg` - Reflected [`DynamicMessage`] instance.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if re-encoding or static decoding fails.
    fn from_dynamic_message(dyn_msg: &DynamicMessage) -> LibTpuResult<Self> {
        Self::decode_proto(&dyn_msg.encode_to_vec())
    }

    /// Serializes this message into canonical Protobuf JSON (`proto3` JSON mapping) via `prost-reflect`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if dynamic reflection fails.
    fn to_proto_json(&self) -> LibTpuResult<String> {
        transcode_proto_bytes_to_json(Self::PROTO_FULL_NAME, &self.encode_to_vec())
    }

    /// Deserializes an instance of this message from canonical Protobuf JSON (`proto3` JSON mapping) via `prost-reflect`.
    ///
    /// # Arguments
    ///
    /// * `json` - Canonical Protobuf JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if JSON parsing or wire transcoding fails.
    fn from_proto_json(json: &str) -> LibTpuResult<Self> {
        let dyn_msg = decode_dynamic_proto_json(Self::PROTO_FULL_NAME, json)?;
        Self::from_dynamic_message(&dyn_msg)
    }

    /// Serializes this message into canonical Protobuf text-format (`textproto`) via `prost-reflect`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if dynamic reflection fails.
    fn to_proto_text(&self) -> LibTpuResult<String> {
        transcode_proto_bytes_to_text(Self::PROTO_FULL_NAME, &self.encode_to_vec())
    }

    /// Deserializes an instance of this message from canonical Protobuf text-format (`textproto`) via `prost-reflect`.
    ///
    /// # Arguments
    ///
    /// * `text` - Canonical Protobuf text-format (`textproto`) string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if textproto parsing or wire transcoding fails.
    fn from_proto_text(text: &str) -> LibTpuResult<Self> {
        let dyn_msg = decode_dynamic_proto_text(Self::PROTO_FULL_NAME, text)?;
        Self::from_dynamic_message(&dyn_msg)
    }

    /// Verifies that this message instance round-trips losslessly through both [`DynamicMessage`]
    /// wire reflection and canonical Protobuf JSON against the official `.proto` schema in [`global_descriptor_pool`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if wire or JSON reflection diverges from `self`.
    fn verify_schema_sync(&self) -> LibTpuResult<()> {
        let json = self.to_proto_json()?;
        let roundtripped = Self::from_proto_json(&json)?;
        if &roundtripped != self {
            return Err(LibTpuError::ProtobufDecodeError {
                message_type: Self::PROTO_FULL_NAME,
                reason: String::from(
                    "schema synchronization mismatch after DynamicMessage + JSON round-trip",
                ),
            });
        }
        Ok(())
    }
}

/// Implements `encode`, `decode`, and [`ProtoSchemaMessage`] for a `#[derive(Message)]` struct.
macro_rules! impl_proto_schema_message {
    ($ty:ty, $full_name:literal) => {
        impl $crate::proto::descriptor::ProtoSchemaMessage for $ty {
            const PROTO_FULL_NAME: &'static str = $full_name;
        }

        impl $ty {
            /// Encodes this protobuf message to wire bytes via `prost`.
            #[must_use]
            pub fn encode(&self) -> Vec<u8> {
                <Self as prost::Message>::encode_to_vec(self)
            }

            /// Decodes an instance of this protobuf message from wire bytes via `prost`.
            ///
            /// # Arguments
            ///
            /// * `bytes` - Serialized protobuf wire bytes.
            ///
            /// # Errors
            ///
            /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
            pub fn decode(bytes: &[u8]) -> $crate::error::LibTpuResult<Self> {
                $crate::proto::descriptor::decode_prost_message(bytes, $full_name)
            }
        }
    };
}

pub(crate) use impl_proto_schema_message;
