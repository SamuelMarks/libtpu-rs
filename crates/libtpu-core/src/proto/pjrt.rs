//! PJRT `CompileOptions`, `ExecuteOptions`, topology description, ABI version, and distributed
//! runtime protobuf types.
//!
//! Re-exports `prost-build` generated structs from `third_party/tensorflow/compiler/xla/pjrt/proto/`,
//! `third_party/tensorflow/compiler/xla/service/`, `third_party/tensorflow/compiler/xla/stream_executor/`,
//! and `protos/libtpu_ext/pjrt_ext.proto`.

use super::descriptor::ProtoSchemaMessage as _;
use crate::error::LibTpuResult;

pub use super::descriptor::generated_prost_schemas::libtpu::ext::pjrt::{
    DictKeysProto, DistributedDeviceProto, GlobalTopologyProto, LocalTopologyProto,
    PyHostCallbackProto, PyTreeDefProto, PyTreeNodeDefProto, ShardyAxisRefProto,
    ShardyDimensionShardingProto, ShardyMeshAxisProto, ShardyMeshProto, ShardySubAxisInfoProto,
    ShardyTensorShardingProto, StableHloPortableArtifactMetadataProto, VhloDialectVersionProto,
    XlaFfiExecutionContextMetadataProto, XlaFfiTypeIdRegistrationProto,
};
pub use super::descriptor::generated_prost_schemas::stream_executor::dnn::{
    AlgorithmConfigProto as DnnAlgorithmConfigProto, AlgorithmProto as DnnAlgorithmProto,
    ConvolutionDescriptorProto as DnnConvolutionDescriptorProto,
    TensorDescriptorProto as DnnTensorDescriptorProto,
};
pub use super::descriptor::generated_prost_schemas::stream_executor::{
    CudaComputeCapabilityProto,
    ExecutableAbiVersionProto as StreamExecutorExecutableAbiVersionProto, GpuDeviceInfoProto,
    GpuTargetConfigProto, OneApiComputeCapabilityProto,
    RuntimeAbiVersionProto as StreamExecutorRuntimeAbiVersionProto,
};
pub use super::descriptor::generated_prost_schemas::xla::{
    AutotuneResult as AutotuneResultProto, AutotuneResults as AutotuneResultsProto,
    AutotuningLog as AutotuningLogProto, AutotuningLogs as AutotuningLogsProto,
    CompilationEnvironmentsProto, CompileOptionsProto, CompiledMemoryStatsProto,
    ComputeCapability as ComputeCapabilityProto, CudnnVersion as CudnnVersionProto,
    DebugOptions as DebugOptionsProto, ExecutableAndOptionsProto, ExecutableBuildOptionsProto,
    ExecuteOptionsProto, ExecutionModeProto, GpuTopologyProto, OptionOverrideProto,
    PjRtDeviceDimensionsProto, PjRtExecutableAbiVersionProto,
    PjRtExecutableMetadata as PjRtExecutableMetadataProto, PjRtPartialProgramProto,
    PjRtRuntimeAbiVersionProto, PjRtTopologyDescriptionProto, PjRtValueTypeProto,
    TritonGemmConfigsProto, pj_rt_value_type_proto,
};

/// Alias for `PjRtRuntimeAbiVersionProto` generated from `pjrt_abi_version.proto`.
pub type RuntimeAbiVersionProto = PjRtRuntimeAbiVersionProto;

/// Alias for `PjRtExecutableAbiVersionProto` generated from `pjrt_abi_version.proto`.
pub type ExecutableAbiVersionProto = PjRtExecutableAbiVersionProto;

/// Alias for `stream_executor.ExecutableAbiVersionProto.CudaPlatformVersion`.
pub type CudaPlatformVersionProto = super::descriptor::generated_prost_schemas::stream_executor::executable_abi_version_proto::CudaPlatformVersion;

/// Alias for `xla.AutotuneResult.ConvKey`.
pub type AutotuneConvKeyProto =
    super::descriptor::generated_prost_schemas::xla::autotune_result::ConvKey;

/// Alias for `xla.AutotuneResult.GemmKey`.
pub type AutotuneGemmKeyProto =
    super::descriptor::generated_prost_schemas::xla::autotune_result::GemmKey;

/// Alias for `xla.AutotuneResult.CudaConvPlanKey`.
pub type AutotuneCudaConvPlanKeyProto =
    super::descriptor::generated_prost_schemas::xla::autotune_result::CudaConvPlanKey;

/// Alias for `xla.AutotuneResult.TritonGemmKey`.
pub type AutotuneTritonGemmKeyProto =
    super::descriptor::generated_prost_schemas::xla::autotune_result::TritonGemmKey;

/// Alias for `xla.AutotuneResult.CustomKernelFusionKey`.
pub type AutotuneCustomKernelFusionKeyProto =
    super::descriptor::generated_prost_schemas::xla::autotune_result::CustomKernelFusionKey;

/// Alias for `xla.AutotuneResult.BackendConfigKey`.
pub type AutotuneBackendConfigKeyProto =
    super::descriptor::generated_prost_schemas::xla::autotune_result::BackendConfigKey;

/// Alias for `xla.AutotuneResult.FailureResult`.
pub type AutotuneFailureResultProto =
    super::descriptor::generated_prost_schemas::xla::autotune_result::FailureResult;

/// Alias for `xla.AutotuneResults.Entry`.
pub type AutotuneResultsEntryProto =
    super::descriptor::generated_prost_schemas::xla::autotune_results::Entry;

/// Alias for `xla.PjRtValueTypeProto.IntVector`.
pub type PjRtValueIntVectorProto = pj_rt_value_type_proto::IntVector;

impl CompileOptionsProto {
    /// Decodes a [`CompileOptionsProto`] from `bytes`, returning [`CompileOptionsProto::default()`]
    /// if `bytes` is empty.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw serialized `CompileOptionsProto` byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if `bytes` is non-empty and malformed.
    pub fn decode_or_empty(bytes: &[u8]) -> LibTpuResult<Self> {
        if bytes.is_empty() {
            Ok(Self::default())
        } else {
            Self::decode_proto(bytes)
        }
    }
}

impl LocalTopologyProto {
    /// Serializes this [`LocalTopologyProto`] into Protobuf wire bytes for KV exchange.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`LocalTopologyProto`] from Protobuf wire bytes written to the coordination KV store.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `LocalTopologyProto` payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if `bytes` cannot be decoded.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl GlobalTopologyProto {
    /// Serializes this [`GlobalTopologyProto`] into Protobuf wire bytes for KV exchange.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`GlobalTopologyProto`] from Protobuf wire bytes written to the coordination KV store.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `GlobalTopologyProto` payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if `bytes` cannot be decoded.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}
