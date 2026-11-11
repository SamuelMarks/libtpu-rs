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

/// Wire-compatible `xla.CompilationEnvironmentsProto` (`xla/xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilationEnvironmentsProto {
    /// Serialized compilation environment payloads (field `1`).
    #[prost(bytes = "vec", repeated, tag = "1")]
    pub environments: Vec<Vec<u8>>,
}

impl_proto_schema_message!(
    CompilationEnvironmentsProto,
    "xla.CompilationEnvironmentsProto"
);

/// Wire-compatible `xla.DebugOptions` (`xla/xla.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DebugOptionsProto {
    /// Directory path to dump HLO / LLO artifacts (field `1`).
    #[prost(string, tag = "1")]
    pub xla_dump_to: String,
    /// Whether to dump HLO modules as text (field `2`).
    #[prost(bool, tag = "2")]
    pub xla_dump_hlo_as_text: bool,
    /// Whether to dump HLO modules as protobuf (field `3`).
    #[prost(bool, tag = "3")]
    pub xla_dump_hlo_as_proto: bool,
    /// Regex filter of HLO passes to dump (field `4`).
    #[prost(string, tag = "4")]
    pub xla_dump_hlo_pass_re: String,
    /// Regex filter of HLO modules to dump (field `5`).
    #[prost(string, tag = "5")]
    pub xla_dump_hlo_module_re: String,
    /// Whether to dump `HloSnapshot` inputs/outputs on execution (field `6`).
    #[prost(bool, tag = "6")]
    pub xla_dump_hlo_snapshots: bool,
    /// Backend optimization level (field `31`).
    #[prost(int32, tag = "31")]
    pub xla_backend_optimization_level: i32,
    /// HLO pass names explicitly disabled during compilation (field `32`).
    #[prost(string, repeated, tag = "32")]
    pub xla_disable_hlo_passes: Vec<String>,
    /// HLO pass names explicitly enabled exclusively during compilation (field `33`).
    #[prost(string, repeated, tag = "33")]
    pub xla_enable_hlo_passes_only: Vec<String>,
    /// Enable hardware NaN trap detection on Cloud TPU (field `64`).
    #[prost(bool, tag = "64")]
    pub xla_tpu_detect_nan: bool,
    /// Enable hardware Inf trap detection on Cloud TPU (field `65`).
    #[prost(bool, tag = "65")]
    pub xla_tpu_detect_inf: bool,
    /// Enable fast approximate math optimizations (field `66`).
    #[prost(bool, tag = "66")]
    pub xla_enable_fast_math: bool,
    /// Enable Triton GEMM emitter (field `67`).
    #[prost(bool, tag = "67")]
    pub xla_gpu_enable_triton_gemm: bool,
    /// Enable TPU asynchronous collective fusion (field `68`).
    #[prost(bool, tag = "68")]
    pub xla_tpu_enable_async_collective_fusion: bool,
    /// Enable TPU all-gather backward pipelining (field `69`).
    #[prost(bool, tag = "69")]
    pub xla_tpu_enable_ag_backward_pipelining: bool,
    /// Enable TPU reduce-scatter forward pipelining (field `70`).
    #[prost(bool, tag = "70")]
    pub xla_tpu_enable_rs_forward_pipelining: bool,
    /// Overlap `TensorCore` compute with ICI/DCN collectives (field `71`).
    #[prost(bool, tag = "71")]
    pub xla_tpu_overlap_compute_collective_tc: bool,
    /// Enable `OpenXLA` Shardy (`sdy`) SPMD partitioner on TPU (field `72`).
    #[prost(bool, tag = "72")]
    pub xla_tpu_use_shardy_partitioner: bool,
    /// Latency-hiding scheduler memory limit in bytes (field `73`).
    #[prost(int64, tag = "73")]
    pub xla_latency_hiding_scheduler_limit: i64,
}

impl_proto_schema_message!(DebugOptionsProto, "xla.DebugOptions");

/// Wire-compatible `xla.GpuTopologyProto` (`xla/service/gpu_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GpuTopologyProto {
    /// Number of partitions (field `1`).
    #[prost(int32, tag = "1")]
    pub num_partitions: i32,
    /// Number of hosts per partition (field `2`).
    #[prost(int32, tag = "2")]
    pub num_hosts_per_partition: i32,
    /// Number of devices per host (field `3`).
    #[prost(int32, tag = "3")]
    pub num_devices_per_host: i32,
}

impl_proto_schema_message!(GpuTopologyProto, "xla.GpuTopologyProto");

/// Wire-compatible `stream_executor.CudaComputeCapabilityProto` (`cuda_compute_capability.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CudaComputeCapabilityProto {
    /// CUDA major compute capability (field `1`).
    #[prost(int32, tag = "1")]
    pub major: i32,
    /// CUDA minor compute capability (field `2`).
    #[prost(int32, tag = "2")]
    pub minor: i32,
    /// Feature extension enum (`0` = None, `1` = `AcceleratedFeatures`, `2` = `FamilySpecificFeatures`, field `3`).
    #[prost(int32, tag = "3")]
    pub feature_extension: i32,
}

impl_proto_schema_message!(
    CudaComputeCapabilityProto,
    "stream_executor.CudaComputeCapabilityProto"
);

/// Wire-compatible `stream_executor.OneAPIComputeCapabilityProto` (`oneapi_compute_capability.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct OneApiComputeCapabilityProto {
    /// Intel `oneAPI` generation identifier (field `1`).
    #[prost(int32, tag = "1")]
    pub generation: i32,
}

impl_proto_schema_message!(
    OneApiComputeCapabilityProto,
    "stream_executor.OneAPIComputeCapabilityProto"
);

/// Wire-compatible `stream_executor.DnnVersionInfoProto` (`device_description.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DnnVersionInfoProto {
    /// Major version number (field `1`).
    #[prost(int32, tag = "1")]
    pub major: i32,
    /// Minor version number (field `2`).
    #[prost(int32, tag = "2")]
    pub minor: i32,
    /// Patch version number (field `3`).
    #[prost(int32, tag = "3")]
    pub patch: i32,
}

impl_proto_schema_message!(DnnVersionInfoProto, "stream_executor.DnnVersionInfoProto");

/// Wire-compatible `stream_executor.GpuDeviceInfoProto` (`device_description.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct GpuDeviceInfoProto {
    /// Maximum threads per block (field `1`).
    #[prost(int32, tag = "1")]
    pub threads_per_block_limit: i32,
    /// Threads per warp (field `2`).
    #[prost(int32, tag = "2")]
    pub threads_per_warp: i32,
    /// Shared memory per block in bytes (field `3`).
    #[prost(int32, tag = "3")]
    pub shared_memory_per_block: i32,
    /// Shared memory per core in bytes (field `4`).
    #[prost(int32, tag = "4")]
    pub shared_memory_per_core: i32,
    /// Maximum threads per core (field `5`).
    #[prost(int32, tag = "5")]
    pub threads_per_core_limit: i32,
    /// Streaming multiprocessor / core count (field `6`).
    #[prost(int32, tag = "6")]
    pub core_count: i32,
    /// FPUs per core (field `7`).
    #[prost(int64, tag = "7")]
    pub fpus_per_core: i64,
    /// Block dimension X limit (field `8`).
    #[prost(int32, tag = "8")]
    pub block_dim_limit_x: i32,
    /// Block dimension Y limit (field `9`).
    #[prost(int32, tag = "9")]
    pub block_dim_limit_y: i32,
    /// Block dimension Z limit (field `10`).
    #[prost(int32, tag = "10")]
    pub block_dim_limit_z: i32,
    /// Memory bandwidth in bytes/sec (field `11`).
    #[prost(int64, tag = "11")]
    pub memory_bandwidth: i64,
    /// L2 cache size in bytes (field `12`).
    #[prost(int64, tag = "12")]
    pub l2_cache_size: i64,
    /// Clock rate in GHz (field `13`).
    #[prost(float, tag = "13")]
    pub clock_rate_ghz: f32,
    /// Device memory size in bytes (field `14`).
    #[prost(int64, tag = "14")]
    pub device_memory_size: i64,
    /// Opt-in shared memory per block in bytes (field `15`).
    #[prost(int32, tag = "15")]
    pub shared_memory_per_block_optin: i32,
    /// CUDA compute capability (field `16`).
    #[prost(message, optional, tag = "16")]
    pub cuda_compute_capability: Option<CudaComputeCapabilityProto>,
    /// `oneAPI` compute capability (field `20`).
    #[prost(message, optional, tag = "20")]
    pub oneapi_compute_capability: Option<OneApiComputeCapabilityProto>,
}

impl_proto_schema_message!(GpuDeviceInfoProto, "stream_executor.GpuDeviceInfoProto");

/// Wire-compatible `stream_executor.RuntimeAbiVersionProto` (`runtime_abi_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StreamExecutorRuntimeAbiVersionProto {
    /// Platform name (field `1`).
    #[prost(string, tag = "1")]
    pub platform_name: String,
    /// Platform-specific serialized version bytes (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub platform_specific_version: Vec<u8>,
}

impl_proto_schema_message!(
    StreamExecutorRuntimeAbiVersionProto,
    "stream_executor.RuntimeAbiVersionProto"
);

/// Wire-compatible `stream_executor.ExecutableAbiVersionProto.CudaPlatformVersion` (`executable_abi_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CudaPlatformVersionProto {
    /// CUDA toolkit version bytes (field `1`).
    #[prost(bytes = "vec", tag = "1")]
    pub cuda_toolkit_version: Vec<u8>,
    /// cuDNN version bytes (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub cudnn_version: Vec<u8>,
    /// CUB version bytes (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub cub_version: Vec<u8>,
}

impl_proto_schema_message!(
    CudaPlatformVersionProto,
    "stream_executor.ExecutableAbiVersionProto.CudaPlatformVersion"
);

/// Wire-compatible `stream_executor.ExecutableAbiVersionProto` (`executable_abi_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StreamExecutorExecutableAbiVersionProto {
    /// Platform name (field `1`).
    #[prost(string, tag = "1")]
    pub platform_name: String,
    /// CUDA platform version payload (field `3`).
    #[prost(message, optional, tag = "3")]
    pub cuda_platform_version: Option<CudaPlatformVersionProto>,
}

impl_proto_schema_message!(
    StreamExecutorExecutableAbiVersionProto,
    "stream_executor.ExecutableAbiVersionProto"
);

/// Wire-compatible `stream_executor.dnn.AlgorithmProto` (`dnn.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DnnAlgorithmProto {
    /// Algorithm ID (field `1`).
    #[prost(int64, tag = "1")]
    pub algo_id: i64,
    /// Math type flag (field `2`).
    #[prost(bool, tag = "2")]
    pub math_type: bool,
    /// Workspace size in bytes (field `5`).
    #[prost(uint64, tag = "5")]
    pub workspace_size: u64,
}

impl_proto_schema_message!(DnnAlgorithmProto, "stream_executor.dnn.AlgorithmProto");

/// Wire-compatible `stream_executor.dnn.AlgorithmConfigProto` (`dnn.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DnnAlgorithmConfigProto {
    /// Primary algorithm (field `1`).
    #[prost(message, optional, tag = "1")]
    pub algorithm: Option<DnnAlgorithmProto>,
    /// Fallback algorithm without scratch memory (field `2`).
    #[prost(message, optional, tag = "2")]
    pub algorithm_no_scratch: Option<DnnAlgorithmProto>,
    /// Scratch size in bytes (field `3`).
    #[prost(int64, tag = "3")]
    pub scratch_size: i64,
}

impl_proto_schema_message!(
    DnnAlgorithmConfigProto,
    "stream_executor.dnn.AlgorithmConfigProto"
);

/// Wire-compatible `stream_executor.dnn.ConvolutionDescriptorProto` (`dnn.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DnnConvolutionDescriptorProto {
    /// Spatial paddings (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub paddings: Vec<i64>,
    /// Spatial strides (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub strides: Vec<i64>,
    /// Spatial dilations (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub dilations: Vec<i64>,
    /// Convolution group count (field `6`).
    #[prost(int32, tag = "6")]
    pub group_count: i32,
}

impl_proto_schema_message!(
    DnnConvolutionDescriptorProto,
    "stream_executor.dnn.ConvolutionDescriptorProto"
);

/// Wire-compatible `stream_executor.dnn.TensorDescriptorProto` (`dnn.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DnnTensorDescriptorProto {
    /// Tensor dimensions (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub dimensions: Vec<i64>,
    /// Data type enum (field `2`).
    #[prost(int32, tag = "2")]
    pub data_type: i32,
}

impl_proto_schema_message!(
    DnnTensorDescriptorProto,
    "stream_executor.dnn.TensorDescriptorProto"
);

/// Wire-compatible `xla.CudnnVersion` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CudnnVersionProto {
    /// Major version (field `4`).
    #[prost(int32, tag = "4")]
    pub major_version: i32,
    /// Minor version (field `5`).
    #[prost(int32, tag = "5")]
    pub minor_version: i32,
    /// Patch version (field `6`).
    #[prost(int32, tag = "6")]
    pub patch_version: i32,
}

impl_proto_schema_message!(CudnnVersionProto, "xla.CudnnVersion");

/// Wire-compatible `xla.ComputeCapability` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ComputeCapabilityProto {
    /// Major compute capability version (field `3`).
    #[prost(int32, tag = "3")]
    pub major_version: i32,
    /// Minor compute capability version (field `4`).
    #[prost(int32, tag = "4")]
    pub minor_version: i32,
}

impl_proto_schema_message!(ComputeCapabilityProto, "xla.ComputeCapability");

/// Wire-compatible `xla.AutotuneResult.ConvKey` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneConvKeyProto {
    /// Convolution algorithm ID (field `1`).
    #[prost(int64, tag = "1")]
    pub algorithm: i64,
    /// Whether `TensorCore` ops are enabled (field `2`).
    #[prost(bool, tag = "2")]
    pub tensor_ops_enabled: bool,
}

impl_proto_schema_message!(AutotuneConvKeyProto, "xla.AutotuneResult.ConvKey");

/// Wire-compatible `xla.AutotuneResult.GemmKey` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneGemmKeyProto {
    /// GEMM algorithm ID (field `1`).
    #[prost(int64, tag = "1")]
    pub algorithm: i64,
    /// Autotune workspace size in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub autotune_workspace_size: i64,
}

impl_proto_schema_message!(AutotuneGemmKeyProto, "xla.AutotuneResult.GemmKey");

/// Wire-compatible `xla.AutotuneResult.CudaConvPlanKey` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneCudaConvPlanKeyProto {
    /// Execution plan identifier (field `1`).
    #[prost(string, tag = "1")]
    pub exec_plan_id: String,
}

impl_proto_schema_message!(
    AutotuneCudaConvPlanKeyProto,
    "xla.AutotuneResult.CudaConvPlanKey"
);

/// Wire-compatible `xla.AutotuneResult.TritonGemmKey` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneTritonGemmKeyProto {
    /// Block size M (field `1`).
    #[prost(int64, tag = "1")]
    pub block_m: i64,
    /// Block size N (field `2`).
    #[prost(int64, tag = "2")]
    pub block_n: i64,
    /// Block size K (field `3`).
    #[prost(int64, tag = "3")]
    pub block_k: i64,
    /// Split-K factor (field `4`).
    #[prost(int64, tag = "4")]
    pub split_k: i64,
    /// Pipeline stages count (field `5`).
    #[prost(int64, tag = "5")]
    pub num_stages: i64,
    /// Warps count (field `6`).
    #[prost(int64, tag = "6")]
    pub num_warps: i64,
    /// CTAs per cluster count (field `7`).
    #[prost(int64, tag = "7")]
    pub num_ctas: i64,
    /// Whether TMA is allowed (field `8`).
    #[prost(bool, tag = "8")]
    pub is_tma_allowed: bool,
    /// Whether warp specialization is allowed (field `9`).
    #[prost(bool, tag = "9")]
    pub is_warp_specialization_allowed: bool,
    /// Waves per execution unit (field `10`).
    #[prost(int64, tag = "10")]
    pub waves_per_eu: i64,
    /// L2 tile grouping size (field `11`).
    #[prost(int64, tag = "11")]
    pub group_size: i64,
    /// AMD MFMA MN instruction size (field `12`).
    #[prost(int64, tag = "12")]
    pub mfma_size: i64,
}

impl_proto_schema_message!(
    AutotuneTritonGemmKeyProto,
    "xla.AutotuneResult.TritonGemmKey"
);

/// Wire-compatible `xla.AutotuneResult.CustomKernelFusionKey` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneCustomKernelFusionKeyProto {
    /// Custom kernel index (field `1`).
    #[prost(int64, tag = "1")]
    pub kernel_index: i64,
}

impl_proto_schema_message!(
    AutotuneCustomKernelFusionKeyProto,
    "xla.AutotuneResult.CustomKernelFusionKey"
);

/// Wire-compatible `xla.AutotuneResult.BackendConfigKey` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneBackendConfigKeyProto {
    /// Backend config name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Opaque backend config payload (field `2`).
    #[prost(message, optional, tag = "2")]
    pub config: Option<prost_types::Any>,
}

impl_proto_schema_message!(
    AutotuneBackendConfigKeyProto,
    "xla.AutotuneResult.BackendConfigKey"
);

/// Wire-compatible `xla.AutotuneResult.FailureResult` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneFailureResultProto {
    /// Failure kind enum (`FailureKind`, field `1`).
    #[prost(int32, tag = "1")]
    pub kind: i32,
    /// Failure diagnostic message (field `2`).
    #[prost(string, tag = "2")]
    pub msg: String,
    /// Reference conv key (field `11`).
    #[prost(message, optional, tag = "11")]
    pub reference_conv: Option<AutotuneConvKeyProto>,
    /// Reference GEMM key (field `12`).
    #[prost(message, optional, tag = "12")]
    pub reference_gemm: Option<AutotuneGemmKeyProto>,
    /// Corrupted buffer address (field `13`).
    #[prost(int64, tag = "13")]
    pub buffer_address: i64,
    /// Reference CUDA conv plan key (field `14`).
    #[prost(message, optional, tag = "14")]
    pub reference_cuda_conv_plan: Option<AutotuneCudaConvPlanKeyProto>,
    /// Reference DNN algorithm (field `15`).
    #[prost(message, optional, tag = "15")]
    pub reference_algorithm: Option<DnnAlgorithmProto>,
}

impl_proto_schema_message!(
    AutotuneFailureResultProto,
    "xla.AutotuneResult.FailureResult"
);

/// Wire-compatible `xla.AutotuneResult` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneResultProto {
    /// Conv key (field `5`).
    #[prost(message, optional, tag = "5")]
    pub conv: Option<AutotuneConvKeyProto>,
    /// GEMM key (field `6`).
    #[prost(message, optional, tag = "6")]
    pub gemm: Option<AutotuneGemmKeyProto>,
    /// Failure details if autotuning failed (field `7`).
    #[prost(message, optional, tag = "7")]
    pub failure: Option<AutotuneFailureResultProto>,
    /// Scratch memory bytes required (field `8`).
    #[prost(int64, tag = "8")]
    pub scratch_bytes: i64,
    /// Measured execution runtime (field `9`).
    #[prost(message, optional, tag = "9")]
    pub run_time: Option<prost_types::Duration>,
    /// CUDA conv plan key (field `15`).
    #[prost(message, optional, tag = "15")]
    pub cuda_conv_plan: Option<AutotuneCudaConvPlanKeyProto>,
    /// DNN algorithm key (field `16`).
    #[prost(message, optional, tag = "16")]
    pub algorithm: Option<DnnAlgorithmProto>,
    /// Triton GEMM key (field `17`).
    #[prost(message, optional, tag = "17")]
    pub triton: Option<AutotuneTritonGemmKeyProto>,
    /// Custom kernel fusion key (field `18`).
    #[prost(message, optional, tag = "18")]
    pub custom_kernel_fusion: Option<AutotuneCustomKernelFusionKeyProto>,
    /// Other backend config key (field `19`).
    #[prost(message, optional, tag = "19")]
    pub other: Option<AutotuneBackendConfigKeyProto>,
}

impl_proto_schema_message!(AutotuneResultProto, "xla.AutotuneResult");

/// Wire-compatible `xla.TritonGemmConfigsProto` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TritonGemmConfigsProto {
    /// Triton GEMM configurations (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub config: Vec<AutotuneTritonGemmKeyProto>,
}

impl_proto_schema_message!(TritonGemmConfigsProto, "xla.TritonGemmConfigsProto");

/// Wire-compatible `xla.AutotuningLog` (`autotuning.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuningLogProto {
    /// Autotuned HLO instruction packed as `google.protobuf.Any` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub instr: Option<prost_types::Any>,
    /// Candidate autotune results (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub results: Vec<AutotuneResultProto>,
    /// cuDNN version (field `3`).
    #[prost(message, optional, tag = "3")]
    pub cudnn_version: Option<CudnnVersionProto>,
    /// Compute capability (field `4`).
    #[prost(message, optional, tag = "4")]
    pub compute_capability: Option<ComputeCapabilityProto>,
    /// Device `PCIe` bus ID (field `5`).
    #[prost(string, tag = "5")]
    pub device_pci_bus_id: String,
    /// BLAS library version (field `6`).
    #[prost(string, tag = "6")]
    pub blas_version: String,
    /// Fusion name (field `7`).
    #[prost(string, tag = "7")]
    pub fusion_name: String,
    /// Fusion count (field `8`).
    #[prost(int64, tag = "8")]
    pub fusion_count: i64,
    /// Selected backend name (field `9`).
    #[prost(string, tag = "9")]
    pub selected_backend: String,
}

impl_proto_schema_message!(AutotuningLogProto, "xla.AutotuningLog");

/// Wire-compatible `xla.AutotuneResults.Entry` (`autotune_results.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneResultsEntryProto {
    /// Device description string (field `1`).
    #[prost(string, tag = "1")]
    pub device: String,
    /// HLO instruction text (field `2`).
    #[prost(string, tag = "2")]
    pub hlo: String,
    /// Best autotuning result (field `3`).
    #[prost(message, optional, tag = "3")]
    pub result: Option<AutotuneResultProto>,
    /// Entry format version (field `4`).
    #[prost(int32, tag = "4")]
    pub version: i32,
}

impl_proto_schema_message!(AutotuneResultsEntryProto, "xla.AutotuneResults.Entry");

/// Wire-compatible `xla.AutotuneResults` (`autotune_results.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuneResultsProto {
    /// Cache format version (field `1`).
    #[prost(int32, tag = "1")]
    pub version: i32,
    /// Cached autotuning entries (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub results: Vec<AutotuneResultsEntryProto>,
}

impl_proto_schema_message!(AutotuneResultsProto, "xla.AutotuneResults");

/// Wire-compatible `xla.AutotuningLogs` (`autotune_results.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AutotuningLogsProto {
    /// Autotuning log records (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub logs: Vec<AutotuningLogProto>,
}

impl_proto_schema_message!(AutotuningLogsProto, "xla.AutotuningLogs");

/// Wire-compatible `stream_executor.GpuTargetConfigProto` (`xla/stream_executor/device_description.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct GpuTargetConfigProto {
    /// GPU device hardware parameters (field `1`).
    #[prost(message, optional, tag = "1")]
    pub gpu_device_info: Option<GpuDeviceInfoProto>,
    /// Platform name string (field `4`).
    #[prost(string, tag = "4")]
    pub platform_name: String,
    /// DNN library version info (field `5`).
    #[prost(message, optional, tag = "5")]
    pub dnn_version_info: Option<DnnVersionInfoProto>,
    /// Ahead-of-time autotune results (field `6`).
    #[prost(message, optional, tag = "6")]
    pub autotune_results: Option<AutotuneResultsProto>,
    /// Device description string (field `7`).
    #[prost(string, tag = "7")]
    pub device_description_str: String,
    /// Runtime ABI version (field `8`).
    #[prost(message, optional, tag = "8")]
    pub runtime_abi_version: Option<StreamExecutorRuntimeAbiVersionProto>,
}

impl_proto_schema_message!(GpuTargetConfigProto, "stream_executor.GpuTargetConfigProto");

/// Wire-compatible `xla.PjRtDeviceDimensionsProto` (`pjrt_device_dimensions.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtDeviceDimensionsProto {
    /// Device mesh dimensions (field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub dimensions: Vec<i32>,
}

impl_proto_schema_message!(PjRtDeviceDimensionsProto, "xla.PjRtDeviceDimensionsProto");

/// Wire-compatible `xla.PjRtValueTypeProto.IntVector` (`pjrt_value_type.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtValueIntVectorProto {
    /// 64-bit integer vector elements (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub values: Vec<i64>,
}

impl_proto_schema_message!(PjRtValueIntVectorProto, "xla.PjRtValueTypeProto.IntVector");

/// Wire-compatible `xla.PjRtValueTypeProto` (`pjrt_value_type.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct PjRtValueTypeProto {
    /// String value (field `1`).
    #[prost(string, tag = "1")]
    pub string_value: String,
    /// Boolean value (field `2`).
    #[prost(bool, tag = "2")]
    pub bool_value: bool,
    /// Signed 64-bit integer value (field `3`).
    #[prost(int64, tag = "3")]
    pub int_value: i64,
    /// Integer vector value (field `4`).
    #[prost(message, optional, tag = "4")]
    pub int_vector: Option<PjRtValueIntVectorProto>,
    /// 32-bit float value (field `5`).
    #[prost(float, tag = "5")]
    pub float_value: f32,
}

impl_proto_schema_message!(PjRtValueTypeProto, "xla.PjRtValueTypeProto");

/// Wire-compatible `jax.PyHostCallbackProto` (`py_host_callback.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PyHostCallbackProto {
    /// Serialized `cloudpickle` Python callable bytes (field `1`).
    #[prost(bytes = "vec", tag = "1")]
    pub callable: Vec<u8>,
}

impl_proto_schema_message!(PyHostCallbackProto, "jax.PyHostCallbackProto");

/// Wire-compatible `jax.DictKeysProto` (`pytree.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DictKeysProto {
    /// Interned string IDs for dictionary keys (field `1`).
    #[prost(uint32, repeated, packed = "true", tag = "1")]
    pub str_id: Vec<u32>,
}

impl_proto_schema_message!(DictKeysProto, "jax.DictKeysProto");

/// Wire-compatible `jax.PyTreeNodeDefProto` (`pytree.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PyTreeNodeDefProto {
    /// Child node arity (field `1`).
    #[prost(uint32, tag = "1")]
    pub arity: u32,
    /// Node type enum (`PyTreeNodeType`, field `2`).
    #[prost(int32, tag = "2")]
    pub r#type: i32,
    /// Dictionary keys when `type == DICT` (field `3`).
    #[prost(message, optional, tag = "3")]
    pub dict_keys: Option<DictKeysProto>,
}

impl_proto_schema_message!(PyTreeNodeDefProto, "jax.PyTreeNodeDefProto");

/// Wire-compatible `jax.PyTreeDefProto` (`pytree.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PyTreeDefProto {
    /// Flattened `PyTree` node definitions (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub nodes: Vec<PyTreeNodeDefProto>,
    /// Interned string table for dictionary keys and namedtuples (field `2`).
    #[prost(string, repeated, tag = "2")]
    pub interned_strings: Vec<String>,
}

impl_proto_schema_message!(PyTreeDefProto, "jax.PyTreeDefProto");

/// Wire-compatible `xla.ExecutableBuildOptionsProto` (`xla/pjrt/proto/compile_options.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutableBuildOptionsProto {
    /// Device ordinal (`-1` if unspecified, field `1`).
    #[prost(int64, tag = "1")]
    pub device_ordinal: i64,
    /// Result layout (field `2`).
    #[prost(message, optional, tag = "2")]
    pub result_layout: Option<ShapeProto>,
    /// Debug options (field `3`).
    #[prost(message, optional, tag = "3")]
    pub debug_options: Option<DebugOptionsProto>,
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
    /// Whether HLO computations should be deduplicated (field `8`).
    #[prost(bool, tag = "8")]
    pub deduplicate_hlo: bool,
    /// Static device assignment (field `9`).
    #[prost(message, optional, tag = "9")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Whether passthrough parameters are aliased (field `10`).
    #[prost(bool, tag = "10")]
    pub alias_passthrough_params: bool,
    /// Whether only the backend compiler should run (field `11`).
    #[prost(bool, tag = "11")]
    pub run_backend_only: bool,
    /// Per-output SPMD sharding propagation flags (field `12`).
    #[prost(bool, repeated, packed = "true", tag = "12")]
    pub allow_spmd_sharding_propagation_to_output: Vec<bool>,
    /// Compilation environments container (field `13`).
    #[prost(message, optional, tag = "13")]
    pub comp_envs: Option<CompilationEnvironmentsProto>,
    /// Feedback-directed optimization profile bytes (field `14`).
    #[prost(bytes = "vec", tag = "14")]
    pub fdo_profile: Vec<u8>,
    /// Device memory size override in bytes (field `15`).
    #[prost(int64, tag = "15")]
    pub device_memory_size: i64,
    /// Auto-SPMD partitioning mesh shape dimensions (field `16`).
    #[prost(int64, repeated, packed = "true", tag = "16")]
    pub auto_spmd_partitioning_mesh_shape: Vec<i64>,
    /// Auto-SPMD partitioning mesh axis IDs (field `17`).
    #[prost(int64, repeated, packed = "true", tag = "17")]
    pub auto_spmd_partitioning_mesh_ids: Vec<i64>,
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
    /// Optimization effort level enum (`ExecutionOptions.EffortLevel`, field `24`).
    #[prost(int32, tag = "24")]
    pub optimization_level: i32,
    /// Memory fitting effort level enum (`ExecutionOptions.EffortLevel`, field `25`).
    #[prost(int32, tag = "25")]
    pub memory_fitting_level: i32,
    /// Optional GPU/accelerator topology descriptor (field `27`).
    #[prost(message, optional, tag = "27")]
    pub gpu_topology: Option<GpuTopologyProto>,
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
    /// Optional target configuration (field `8`).
    #[prost(message, optional, tag = "8")]
    pub target_config: Option<GpuTargetConfigProto>,
    /// Whether in-place MLIR module modification is allowed (field `9`).
    #[prost(bool, tag = "9")]
    pub allow_in_place_mlir_modification: bool,
    /// Matrix unit operand precision (`PrecisionConfig.Precision`, field `10`).
    #[prost(int32, tag = "10")]
    pub matrix_unit_operand_precision: i32,
    /// Optional compiler variant identifier (field `11`).
    #[prost(string, optional, tag = "11")]
    pub compiler_variant: Option<String>,
    /// Output indices with individually defined layouts (field `12`).
    #[prost(int32, repeated, packed = "true", tag = "12")]
    pub individually_defined_output_indices: Vec<i32>,
}

impl_proto_schema_message!(CompileOptionsProto, "xla.CompileOptionsProto");

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
    pub fn decode_or_empty(bytes: &[u8]) -> crate::error::LibTpuResult<Self> {
        if bytes.is_empty() {
            Ok(Self::default())
        } else {
            Self::decode(bytes)
        }
    }
}

/// Wire-compatible `xla.ExecutableAndOptionsProto` (`xla/pjrt/proto/compile_options.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct ExecutableAndOptionsProto {
    /// Serialized opaque executable bytes (field `1`).
    #[prost(bytes = "vec", tag = "1")]
    pub serialized_executable: Vec<u8>,
    /// Associated compilation options (field `2`).
    #[prost(message, optional, tag = "2")]
    pub compile_options: Option<CompileOptionsProto>,
    /// Producing PJRT client name (field `3`).
    #[prost(string, tag = "3")]
    pub pjrt_client_name: String,
}

impl_proto_schema_message!(ExecutableAndOptionsProto, "xla.ExecutableAndOptionsProto");

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
#[allow(clippy::struct_excessive_bools)]
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
    /// Whether host callbacks use major-to-minor data layout (field `8`).
    #[prost(bool, tag = "8")]
    pub use_major_to_minor_data_layout_for_callbacks: bool,
    /// PRNG execution seed (field `9`).
    #[prost(int64, tag = "9")]
    pub seed: i64,
    /// Whether output buffers should be allocated from an arena (field `10`).
    #[prost(bool, tag = "10")]
    pub use_output_arena: bool,
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
    /// Optional shared memory per block in bytes (field `13`).
    #[prost(int32, tag = "13")]
    pub shared_memory_per_block_optin: i32,
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

/// Wire-compatible `vhlo.VhloDialectVersionProto` (`vhlo_ops.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct VhloDialectVersionProto {
    /// Major dialect version (field `1`).
    #[prost(uint32, tag = "1")]
    pub major: u32,
    /// Minor dialect version (field `2`).
    #[prost(uint32, tag = "2")]
    pub minor: u32,
    /// Patch dialect version (field `3`).
    #[prost(uint32, tag = "3")]
    pub patch: u32,
    /// Formatted target version string (e.g., `"1.9.0"`, field `4`).
    #[prost(string, tag = "4")]
    pub target_version_string: String,
}

impl_proto_schema_message!(VhloDialectVersionProto, "vhlo.VhloDialectVersionProto");

/// Wire-compatible `vhlo.StableHloPortableArtifactMetadataProto` (`vhlo_ops.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StableHloPortableArtifactMetadataProto {
    /// Portable artifact module name (field `1`).
    #[prost(string, tag = "1")]
    pub artifact_name: String,
    /// Negotiated `VHLO` dialect version (field `2`).
    #[prost(message, optional, tag = "2")]
    pub vhlo_version: Option<VhloDialectVersionProto>,
    /// Producer MLIR/StableHLO compiler version string (field `3`).
    #[prost(string, tag = "3")]
    pub producer_mlir_version: String,
    /// Total bytecode size in bytes (field `4`).
    #[prost(uint64, tag = "4")]
    pub bytecode_size_bytes: u64,
    /// 32-character SHA-256 fingerprint of the portable artifact (field `5`).
    #[prost(string, tag = "5")]
    pub sha256_fingerprint: String,
}

impl_proto_schema_message!(
    StableHloPortableArtifactMetadataProto,
    "vhlo.StableHloPortableArtifactMetadataProto"
);

/// Wire-compatible `sdy.ShardyMeshAxisProto` (`shardy_mesh.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShardyMeshAxisProto {
    /// Logical mesh axis identifier (e.g., `"x"`, `"y"`, `"data"`, field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Axis size (number of partitions along this axis, field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
}

impl_proto_schema_message!(ShardyMeshAxisProto, "sdy.ShardyMeshAxisProto");

/// Wire-compatible `sdy.ShardySubAxisInfoProto` (`shardy_mesh.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShardySubAxisInfoProto {
    /// Product of sub-axes prior to this sub-axis (field `1`).
    #[prost(int64, tag = "1")]
    pub pre_size: i64,
    /// Size of this sub-axis factor (field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
}

impl_proto_schema_message!(ShardySubAxisInfoProto, "sdy.ShardySubAxisInfoProto");

/// Wire-compatible `sdy.ShardyAxisRefProto` (`shardy_mesh.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShardyAxisRefProto {
    /// Referenced mesh axis name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Optional sub-axis slicing descriptor (field `2`).
    #[prost(message, optional, tag = "2")]
    pub sub_axis_info: Option<ShardySubAxisInfoProto>,
}

impl_proto_schema_message!(ShardyAxisRefProto, "sdy.ShardyAxisRefProto");

/// Wire-compatible `sdy.ShardyDimensionShardingProto` (`shardy_mesh.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShardyDimensionShardingProto {
    /// Ordered mesh axes sharding this tensor dimension (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub axes: Vec<ShardyAxisRefProto>,
    /// Whether this dimension is closed to further sharding propagation (field `2`).
    #[prost(bool, tag = "2")]
    pub is_closed: bool,
    /// Optional propagation priority (`p0`, `p1`, ..., field `3`).
    #[prost(int64, tag = "3")]
    pub priority: i64,
}

impl_proto_schema_message!(
    ShardyDimensionShardingProto,
    "sdy.ShardyDimensionShardingProto"
);

/// Wire-compatible `sdy.ShardyTensorShardingProto` (`shardy_mesh.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShardyTensorShardingProto {
    /// Target `sdy.mesh` symbol name (field `1`).
    #[prost(string, tag = "1")]
    pub mesh_name: String,
    /// Per-dimension sharding specifications (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub dim_shardings: Vec<ShardyDimensionShardingProto>,
    /// Explicitly replicated mesh axes (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub replicated_axes: Vec<ShardyAxisRefProto>,
    /// Unreduced mesh axes requiring deferred all-reduce (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub unreduced_axes: Vec<ShardyAxisRefProto>,
}

impl_proto_schema_message!(ShardyTensorShardingProto, "sdy.ShardyTensorShardingProto");

/// Wire-compatible `sdy.ShardyMeshProto` (`shardy_mesh.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShardyMeshProto {
    /// Mesh symbol name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Ordered mesh axis definitions (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub axes: Vec<ShardyMeshAxisProto>,
    /// Optional explicit logical device ID ordering (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub device_ids: Vec<i64>,
}

impl_proto_schema_message!(ShardyMeshProto, "sdy.ShardyMeshProto");

/// Wire-compatible `xla.ffi.XlaFfiTypeIdRegistrationProto` (`xla_ffi_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XlaFfiTypeIdRegistrationProto {
    /// Registered custom C++/Rust type name (field `1`).
    #[prost(string, tag = "1")]
    pub type_name: String,
    /// Assigned unique `XLA_FFI_TypeId` value (field `2`).
    #[prost(int64, tag = "2")]
    pub type_id: i64,
    /// Size of the custom type in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub type_size_bytes: u64,
    /// Alignment of the custom type in bytes (field `4`).
    #[prost(uint64, tag = "4")]
    pub type_align_bytes: u64,
}

impl_proto_schema_message!(
    XlaFfiTypeIdRegistrationProto,
    "xla.ffi.XlaFfiTypeIdRegistrationProto"
);

/// Wire-compatible `xla.ffi.XlaFfiExecutionContextMetadataProto` (`xla_ffi_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XlaFfiExecutionContextMetadataProto {
    /// Execution run identifier (field `1`).
    #[prost(int32, tag = "1")]
    pub run_id: i32,
    /// Target device ordinal (field `2`).
    #[prost(int32, tag = "2")]
    pub device_ordinal: i32,
    /// Hardware stream/queue handle (field `3`).
    #[prost(uint64, tag = "3")]
    pub stream_handle: u64,
    /// Registered custom type descriptors in this execution context (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub registered_types: Vec<XlaFfiTypeIdRegistrationProto>,
}

impl_proto_schema_message!(
    XlaFfiExecutionContextMetadataProto,
    "xla.ffi.XlaFfiExecutionContextMetadataProto"
);
