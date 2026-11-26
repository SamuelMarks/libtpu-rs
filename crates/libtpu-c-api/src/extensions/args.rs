//! `#[repr(C)]` argument structures and layout assertions for TPU PJRT extensions (`0.0.53.1`).

use crate::pjrt_types::{
    PJRT_AbiVersion_Holder, PJRT_Buffer, PJRT_Client, PJRT_Collectives_Communicator,
    PJRT_DeviceDescription, PJRT_Error, PJRT_Event, PJRT_Executable,
    PJRT_ExecutableMetadata_Handle, PJRT_ExecuteContext, PJRT_Extension_Base,
    PJRT_Layouts_MemoryLayout, PJRT_Megascale_ClientContext, PJRT_Memory, PJRT_MemoryDescription,
    PJRT_MultiSlice_Config, PJRT_PhaseCompile_Compiler, PJRT_Program, PJRT_RawBuffer,
    PJRT_Relocatable, PJRT_TopologyDescription, PLUGIN_Profiler_ConsumeResult,
    PLUGIN_Profiler_Instance,
};

/// Trait implemented by every `#[repr(C)]` PJRT extension argument structure specifying its
/// minimum ABI-accepted `struct_size`.
pub trait ExtensionArgsStruct: Sized {
    /// Minimum valid `struct_size` in bytes accepted by extension dispatchers (`<= size_of::<Self>()`).
    const MIN_STRUCT_SIZE: usize = size_of::<Self>();
}

/// Macro implementing [`ExtensionArgsStruct`] with default `MIN_STRUCT_SIZE = size_of::<T>()` for a batch of structs.
macro_rules! impl_default_extension_args {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ExtensionArgsStruct for $ty {}
        )*
    };
}

/// Argument structure for `PLUGIN_Profiler_Error_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle to destroy.
    pub error: *mut PJRT_Error,
}

/// Argument structure for `PLUGIN_Profiler_Error_Message` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_Message_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle.
    pub error: *const PJRT_Error,
    /// Output error message pointer.
    pub message: *const std::os::raw::c_char,
    /// Output error message byte length.
    pub message_size: usize,
}

/// Argument structure for `PLUGIN_Profiler_Error_GetCode` (`32` bytes, `MIN_STRUCT_SIZE = 28`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_GetCode_Args {
    /// Size of this struct (`28` or `32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle.
    pub error: *const PJRT_Error,
    /// Output status code.
    pub code: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
}

impl ExtensionArgsStruct for PLUGIN_Profiler_Error_GetCode_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}

/// Argument structure for `PLUGIN_Profiler_Create` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Create_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Options string pointer.
    pub options: *const std::os::raw::c_char,
    /// Options string byte length.
    pub options_size: usize,
    /// Output profiler handle.
    pub profiler: *mut PLUGIN_Profiler_Instance,
}

/// Argument structure for `PLUGIN_Profiler_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle to destroy.
    pub profiler: *mut PLUGIN_Profiler_Instance,
}

/// Argument structure for `PLUGIN_Profiler_Start` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Start_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle to start.
    pub profiler: *mut PLUGIN_Profiler_Instance,
}

/// Argument structure for `PLUGIN_Profiler_Stop` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Stop_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle to stop.
    pub profiler: *mut PLUGIN_Profiler_Instance,
}

/// Argument structure for `PLUGIN_Profiler_CollectData` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_CollectData_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler instance handle (`PLUGIN_Profiler_Instance*`).
    pub profiler: *mut PLUGIN_Profiler_Instance,
    /// Caller buffer to receive serialized `tensorflow.profiler.XSpace` bytes (or `NULL` to query size).
    pub buffer: *mut u8,
    /// Output byte count (or input/output buffer capacity).
    pub size_in_bytes: *mut usize,
}

/// Argument structure for `PLUGIN_Profiler_Consume` (`24` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Consume_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Profiler instance handle (`PLUGIN_Profiler_Instance*`).
    pub profiler: *mut PLUGIN_Profiler_Instance,
    /// Output consumed trace snapshot handle (`PLUGIN_Profiler_ConsumeResult*`).
    pub result: *mut PLUGIN_Profiler_ConsumeResult,
}

/// Argument structure for `PLUGIN_Profiler_ConsumeResult_Destroy` (`16` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_ConsumeResult_Destroy_Args {
    /// Size of this struct (`16`).
    pub struct_size: usize,
    /// Consumed trace snapshot handle to destroy (`PLUGIN_Profiler_ConsumeResult*`).
    pub result: *mut PLUGIN_Profiler_ConsumeResult,
}

/// Argument structure for `PLUGIN_Profiler_Serialize` (`40` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Serialize_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Profiler instance handle (`PLUGIN_Profiler_Instance*`).
    pub profiler: *mut PLUGIN_Profiler_Instance,
    /// Consumed trace snapshot handle (`PLUGIN_Profiler_ConsumeResult*`).
    pub result: *const PLUGIN_Profiler_ConsumeResult,
    /// Output pointer to serialized `tensorflow.profiler.XSpace` bytes.
    pub buffer: *const u8,
    /// Output byte length of `buffer`.
    pub size_in_bytes: usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle to destroy (`PJRT_MultiSlice_Config*`).
    pub config: *mut PJRT_MultiSlice_Config,
}

/// Argument structure for `PJRT_MultiSlice_Config_NumSlices` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_NumSlices_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle (`const PJRT_MultiSlice_Config*`).
    pub config: *const PJRT_MultiSlice_Config,
    /// Output number of slices (`i32` / `usize`).
    pub num_slices: usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_NumDevicesPerSlice` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle (`const PJRT_MultiSlice_Config*`).
    pub config: *const PJRT_MultiSlice_Config,
    /// Output device count per slice.
    pub num_devices_per_slice_map: usize,
    /// Output number of slices.
    pub num_slices: usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_SliceId` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_SliceId_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle (`const PJRT_MultiSlice_Config*`).
    pub config: *const PJRT_MultiSlice_Config,
    /// Output local slice ID.
    pub slice_id: usize,
}

/// Argument structure for `PJRT_Collectives_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Collectives_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Collectives instance handle.
    pub collectives: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Collectives_CreateCommunicator` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Collectives_CreateCommunicator_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Collectives or client handle.
    pub collectives: *mut std::ffi::c_void,
    /// Clique device IDs pointer.
    pub device_ids: *const i32,
    /// Number of devices in clique.
    pub num_devices: usize,
    /// Output communicator handle (`PJRT_Collectives_Communicator*`).
    pub communicator: *mut PJRT_Collectives_Communicator,
}

/// Argument structure for `PJRT_Communicator_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle to destroy (`PJRT_Collectives_Communicator*`).
    pub communicator: *mut PJRT_Collectives_Communicator,
}

/// Argument structure for `PJRT_Communicator_AllGather` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_AllGather_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle (`PJRT_Collectives_Communicator*`).
    pub communicator: *mut PJRT_Collectives_Communicator,
    /// Send buffer handle (`PJRT_Buffer*`).
    pub send_buffer: *mut PJRT_Buffer,
    /// Receive buffer handle (`PJRT_Buffer*`).
    pub recv_buffer: *mut PJRT_Buffer,
    /// Element count.
    pub count: usize,
    /// Output completion event (`PJRT_Event*`).
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Communicator_CollectivePermute` and `AllToAll` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_CollectivePermute_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle (`PJRT_Collectives_Communicator*`).
    pub communicator: *mut PJRT_Collectives_Communicator,
    /// Send buffer handle (`PJRT_Buffer*`).
    pub send_buffer: *mut PJRT_Buffer,
    /// Receive buffer handle (`PJRT_Buffer*`).
    pub recv_buffer: *mut PJRT_Buffer,
    /// Target ranks pointer (`[src_rank, dst_rank, ...]` or target rank per rank).
    pub target_ranks: *const i32,
    /// Number of target rank entries or elements.
    pub count: usize,
    /// Output completion event (`PJRT_Event*`).
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Communicator_ToString` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_ToString_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle (`const PJRT_Collectives_Communicator*`).
    pub communicator: *const PJRT_Collectives_Communicator,
    /// Output string pointer.
    pub str_out: *const std::os::raw::c_char,
    /// Output string byte length.
    pub str_size: usize,
}

/// Argument structure for `PJRT_Client_RuntimeAbiVersion` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_RuntimeAbiVersion_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`const PJRT_Client*`).
    pub client: *const PJRT_Client,
    /// Output runtime ABI version handle (`PJRT_AbiVersion_Holder*`).
    pub abi_version: *mut PJRT_AbiVersion_Holder,
}

/// Argument structure for `PJRT_Executable_GetAbiVersion` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_GetAbiVersion_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`const PJRT_Executable*`).
    pub executable: *const PJRT_Executable,
    /// Output executable ABI version handle (`PJRT_AbiVersion_Holder*`).
    pub abi_version: *mut PJRT_AbiVersion_Holder,
}

/// Backward-compatible alias for `PJRT_AbiVersion_Get_Args` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client or executable handle.
    pub handle: *const std::ffi::c_void,
    /// Output ABI version handle (`PJRT_AbiVersion_Holder*`).
    pub abi_version_out: *mut PJRT_AbiVersion_Holder,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_Destroy` and `PJRT_ExecutableAbiVersion_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle to destroy (`PJRT_AbiVersion_Holder*`).
    pub abi_version: *mut PJRT_AbiVersion_Holder,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_IsCompatibleWith*` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_IsCompatible_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Runtime ABI version handle (`const PJRT_AbiVersion_Holder*`).
    pub abi_version: *const PJRT_AbiVersion_Holder,
    /// Other runtime or executable ABI version handle (`const PJRT_AbiVersion_Holder*`).
    pub other_abi_version: *const PJRT_AbiVersion_Holder,
    /// Output compatibility boolean flag (`0` = incompatible, `1` = compatible).
    pub is_compatible_out: usize,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_PlatformId` and `PJRT_ExecutableAbiVersion_PlatformId` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_PlatformId_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle (`const PJRT_AbiVersion_Holder*`).
    pub abi_version: *const PJRT_AbiVersion_Holder,
    /// Output 64-bit platform ID.
    pub platform_id_out: u64,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_FromProto` and `PJRT_ExecutableAbiVersion_FromProto` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_FromProto_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Serialized proto bytes pointer.
    pub serialized_proto: *const std::os::raw::c_char,
    /// Serialized proto byte length.
    pub serialized_proto_size: usize,
    /// Output ABI version handle (`PJRT_AbiVersion_Holder*`).
    pub abi_version_out: *mut PJRT_AbiVersion_Holder,
}

/// Argument structure for `PJRT_Megascale_*` 24-byte lifecycle operations (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Handle_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Megascale context or config handle.
    pub handle: *mut PJRT_Megascale_ClientContext,
}

/// Argument structure for `PJRT_Megascale_*` 32-byte query/create operations (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Query_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input Megascale or executable handle.
    pub handle: *const std::ffi::c_void,
    /// Output value, count, or created handle.
    pub out_value: usize,
}

/// Argument structure for `PJRT_Megascale_*` 40-byte config/metrics operations (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Data_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input Megascale or executable handle.
    pub handle: *mut std::ffi::c_void,
    /// Output or input payload pointer.
    pub out_first: *const u8,
    /// Output or input byte length.
    pub out_second: usize,
}

/// Dedicated argument structure for `PJRT_TpuExecutable_RunHloCostAnalysis` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_RunHloCostAnalysis_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`const PJRT_Executable*`).
    pub executable: *const PJRT_Executable,
    /// Output number of cost properties.
    pub num_properties_out: usize,
}

/// Dedicated argument structure for `PJRT_TpuExecutable_SetTpuCompilationEnv` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_SetTpuCompilationEnv_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *mut PJRT_Executable,
    /// Serialized `TpuCompilationEnvironmentProto` bytes pointer.
    pub serialized_env: *const u8,
    /// Byte length of `serialized_env` (on input) and total overrides count (on output).
    pub serialized_env_size: usize,
}

/// Argument structure for `PJRT_TpuExecutable_*` 48-byte serialized proto queries (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_ProtoQuery_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`const PJRT_Executable*`).
    pub executable: *const PJRT_Executable,
    /// Output serialized bytes pointer.
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output artifact holder handle (`PJRT_ExecutableMetadata_Handle*`).
    pub holder: *mut PJRT_ExecutableMetadata_Handle,
}

/// Argument structure for `PJRT_TpuExecutable_IsPredeterminedError` and `IsCancellationPlaceholderError` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_ErrorCheck_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Error*` handle to inspect.
    pub error: *const PJRT_Error,
    /// Output boolean result (`0` or `1`).
    pub is_matched: usize,
}

/// Argument structure for `PJRT_TpuTopology_*` 32-byte scalar queries (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ScalarQuery_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output count, ID, or boolean flag.
    pub out_value: usize,
}

/// Argument structure for `PJRT_TpuTopology_*` 40-byte coordinate/bounds queries (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_BoundsQuery_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output pointer to `int32_t[3]` coordinate bounds array (or null).
    pub coords_out: *mut i32,
    /// Output coordinate dimension count (`3`).
    pub num_coords: usize,
}

/// Argument structure for `PJRT_TpuTopology_*` 48-byte mapping/subslice queries (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_MappingQuery_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input ID or subslice dimension word.
    pub param0: usize,
    /// Output topology handle or primary coordinate word.
    pub out_first: *mut PJRT_TopologyDescription,
    /// Output secondary word.
    pub out_second: usize,
}

/// Binary slice configuration record returned by `PJRT_TpuTopology_GetSliceConfig` and `GetSliceConfigs` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_SliceConfig {
    /// Pointer to slice configuration name UTF-8 bytes (e.g., `"2x2x1"`).
    pub name: *const std::os::raw::c_char,
    /// Byte length of `name`.
    pub name_len: usize,
    /// Mesh dimensions along up to 4 axes (`[x, y, z, w]`).
    pub dimensions: [i32; 4],
    /// Torus wrap-around flags along up to 4 axes.
    pub wraps: [bool; 4],
    /// Number of active dimensions (`2` or `3`).
    pub num_dimensions: usize,
    /// Whether twisted torus routing is enabled.
    pub twist: bool,
    /// Trailing 7-byte alignment padding (`56` bytes total).
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_TpuTopology_ComputeSubsliceForDevices` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ComputeSubsliceForDevices_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology description handle (`const PJRT_TopologyDescription*`).
    pub topology: *const PJRT_TopologyDescription,
    /// Subslice dimension bounds array pointer (`const int32_t*`).
    pub subslice_dims: *const i32,
    /// Number of subslice dimensions.
    pub num_subslice_dims: usize,
    /// Device ID array pointer (`const int32_t*`).
    pub device_ids: *const i32,
    /// Number of devices in `device_ids`.
    pub num_devices: usize,
    /// Output subslice origin coordinates array pointer (`int32_t*`).
    pub subslice_origin_out: *mut i32,
}

/// Argument structure for `PJRT_Callback_InvokeCallback` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Invoke_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`PJRT_Client*`).
    pub client: *mut PJRT_Client,
    /// Callback type identifier.
    pub callback_type: usize,
    /// Callback payload pointer.
    pub payload: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_PhaseCompile_GetCompiler` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_GetCompiler_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input topology or client handle.
    pub client: *mut PJRT_Client,
    /// Output phase compiler handle (`PJRT_PhaseCompile_Compiler*`).
    pub compiler_out: *mut PJRT_PhaseCompile_Compiler,
}

/// Argument structure for `PJRT_PhaseCompile_DestroyCompiler` and `CBuffersDestroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler or C buffer handle to destroy.
    pub handle: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Relocatable_Destroy` (`24` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle to destroy (`PJRT_Relocatable*`).
    pub relocatable: *mut PJRT_Relocatable,
}

/// Argument structure for `PJRT_Relocatable_Name`, `BuildId`, and `CallBackendConfig` (`40` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_StringQuery_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle (`const PJRT_Relocatable*`).
    pub relocatable: *const PJRT_Relocatable,
    /// Output UTF-8 / byte pointer.
    pub value_ptr: *const std::os::raw::c_char,
    /// Output byte length.
    pub value_size: usize,
}

/// Argument structure for `PJRT_Relocatable_HasSideEffects` (`32` bytes, `min_struct_size = 25`, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_HasSideEffects_Args {
    /// Size of this struct (`25` or `32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle (`const PJRT_Relocatable*`).
    pub relocatable: *const PJRT_Relocatable,
    /// Output boolean flag (`true` if module has side effects).
    pub has_side_effects: bool,
    /// Alignment padding (`7` bytes).
    pub pad: [u8; 7],
}

impl ExtensionArgsStruct for PJRT_Relocatable_HasSideEffects_Args {
    const MIN_STRUCT_SIZE: usize = 25;
}

/// Argument structure for `PJRT_Relocatable_Serialize` (`56` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_Serialize_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle (`const PJRT_Relocatable*`).
    pub relocatable: *const PJRT_Relocatable,
    /// Output serialized bytes pointer.
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output opaque serialized relocatable backing handle.
    pub serialized_relocatable: *mut std::ffi::c_void,
    /// Output deleter callback for `serialized_relocatable`.
    pub serialized_relocatable_deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
}

/// Argument structure for `PJRT_Relocatable_ParameterElementTypes` and `OutputElementTypes` (`40` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_ElementTypes_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle (`const PJRT_Relocatable*`).
    pub relocatable: *const PJRT_Relocatable,
    /// Output array of `PJRT_Buffer_Type` (`const int32_t*`).
    pub element_types: *const i32,
    /// Output number of types.
    pub num_types: usize,
}

/// Argument structure for `PJRT_Relocatable_ParameterDimensions` and `OutputDimensions` (`48` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_Dimensions_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle (`const PJRT_Relocatable*`).
    pub relocatable: *const PJRT_Relocatable,
    /// Output number of parameter or output shapes.
    pub num_items: usize,
    /// Output flattened dimensions array (`const int64_t*`).
    pub dims: *const i64,
    /// Output per-item rank array (`const size_t*`).
    pub dim_sizes: *const usize,
}

/// Argument structure for `PJRT_Relocatable_ParameterLayouts` and `OutputLayouts` (`40` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_Layouts_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle (`const PJRT_Relocatable*`).
    pub relocatable: *const PJRT_Relocatable,
    /// Output number of layouts.
    pub num_layouts: usize,
    /// Output array of `PJRT_Layouts_MemoryLayout*`.
    pub layouts: *const *mut PJRT_Layouts_MemoryLayout,
}

/// Argument structure for `PJRT_Relocatable_ParameterOutputAliases` (`40` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Relocatable_ParameterOutputAliases_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Relocatable handle (`const PJRT_Relocatable*`).
    pub relocatable: *const PJRT_Relocatable,
    /// Output number of parameters.
    pub num_parameters: usize,
    /// Output alias indices (`const int64_t*`, `-1` when unaliased).
    pub parameter_output_aliases: *const i64,
}

/// Argument structure for `PJRT_PhaseCompile_CompileToRelocatable` (`88` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_CompileToRelocatable_Args {
    /// Size of this struct (`88`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler handle (`const PJRT_PhaseCompile_Compiler*`).
    pub phase_compiler: *const PJRT_PhaseCompile_Compiler,
    /// Optional `PJRT_Client*`.
    pub client: *mut PJRT_Client,
    /// Input `const PJRT_Program*`.
    pub program: *const PJRT_Program,
    /// Serialized compile options pointer.
    pub compile_options: *const std::os::raw::c_char,
    /// Serialized compile options byte length.
    pub compile_options_size: usize,
    /// Input topology (`const PJRT_TopologyDescription*`).
    pub topology: *const PJRT_TopologyDescription,
    /// Output array of `PJRT_Relocatable*` handles.
    pub relocatables: *const *mut PJRT_Relocatable,
    /// Output number of produced relocatable modules.
    pub num_relocatables: usize,
    /// Output backing buffer handle to free via `C_Buffers_Destroy`.
    pub relocatables_buffer: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_PhaseCompile_DeserializeRelocatable` (`48` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_DeserializeRelocatable_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler handle (`const PJRT_PhaseCompile_Compiler*`).
    pub phase_compiler: *const PJRT_PhaseCompile_Compiler,
    /// Serialized relocatable bytes pointer.
    pub serialized_relocatable: *const std::os::raw::c_char,
    /// Serialized relocatable byte length.
    pub serialized_relocatable_size: usize,
    /// Output deserialized `PJRT_Relocatable*`.
    pub relocatable: *mut PJRT_Relocatable,
}

/// Argument structure for `PJRT_PhaseCompile_LinkRelocatables` (`80` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_LinkRelocatables_Args {
    /// Size of this struct (`80`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler handle (`const PJRT_PhaseCompile_Compiler*`).
    pub phase_compiler: *const PJRT_PhaseCompile_Compiler,
    /// Input array of `PJRT_Relocatable*` handles.
    pub relocatables: *const *mut PJRT_Relocatable,
    /// Number of input relocatables.
    pub num_relocatables: usize,
    /// Input `const PJRT_Program*` entry wrapper (optional).
    pub program: *const PJRT_Program,
    /// Serialized compile options pointer.
    pub compile_options: *const std::os::raw::c_char,
    /// Serialized compile options byte length.
    pub compile_options_size: usize,
    /// Input topology (`const PJRT_TopologyDescription*`).
    pub topology: *const PJRT_TopologyDescription,
    /// Output linked `PJRT_Executable*`.
    pub executable: *mut PJRT_Executable,
}

/// Optional cross-host serialized descriptor callback (`void (*)(const char*, size_t, void*)`).
pub type PjrtCrossHostDescriptorCallbackFn =
    unsafe extern "C" fn(*const std::os::raw::c_char, usize, *mut std::ffi::c_void);

/// Argument structure for `PJRT_CrossHostTransfers_*` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CrossHostTransfers_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`PJRT_Client*`).
    pub client: *mut PJRT_Client,
    /// Array of `PJRT_Buffer*` handles (`*const *mut PJRT_Buffer`).
    pub buffers: *const *mut PJRT_Buffer,
    /// Number of buffers in `buffers`.
    pub num_buffers: usize,
    /// Transfer key / tag (`usize`).
    pub transfer_key: usize,
    /// Output completion event (`PJRT_Event*`).
    pub out_event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_ExecutableMetadata_Get` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Metadata_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`const PJRT_Executable*`).
    pub executable: *const PJRT_Executable,
    /// Output metadata handle (`PJRT_ExecutableMetadata_Handle*`).
    pub metadata_out: *mut PJRT_ExecutableMetadata_Handle,
}

/// Argument structure for `PJRT_ExecutableMetadata_Destroy` and `PJRT_MemoryDescriptions_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Metadata_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Metadata handle to destroy (`PJRT_ExecutableMetadata_Handle*`).
    pub metadata: *mut PJRT_ExecutableMetadata_Handle,
}

/// Argument structure for `PJRT_Layouts_PJRT_Executable_GetOutputLayouts` and `GetParameterLayouts` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_PJRT_Executable_Layouts_Args {
    /// Size of this struct (`40`, or legacy `32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *mut PJRT_Executable,
    /// Output number of layouts (`num_outputs` or `num_parameters`).
    pub num_layouts: usize,
    /// Output pointer to array of `PJRT_Layouts_MemoryLayout*` handles (written when `struct_size >= 40`).
    pub layouts: *const *mut PJRT_Layouts_MemoryLayout,
}

impl ExtensionArgsStruct for PJRT_Layouts_PJRT_Executable_Layouts_Args {
    const MIN_STRUCT_SIZE: usize = 32;
}

/// Argument structure for `PJRT_RawBuffer_*` 32-byte queries (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_Query_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_RawBuffer*` (or `PJRT_Buffer*` for `CreateRawAliasOfBuffer`).
    pub buffer: *mut std::ffi::c_void,
    /// Output raw buffer handle, pointer, size, or memory space.
    pub out_value: usize,
}

/// Argument structure for `PJRT_RawBuffer_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Raw buffer handle to destroy (`PJRT_RawBuffer*`).
    pub raw_buffer: *mut PJRT_RawBuffer,
}

/// Argument structure for `PJRT_Layouts_MemoryLayout_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_MemoryLayout_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory layout handle to destroy (`PJRT_Layouts_MemoryLayout*`).
    pub layout: *mut PJRT_Layouts_MemoryLayout,
}

/// Argument structure for `PJRT_Layouts_MemoryLayout_Serialize` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_MemoryLayout_Serialize_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory layout handle to serialize (`const PJRT_Layouts_MemoryLayout*`).
    pub layout: *const PJRT_Layouts_MemoryLayout,
    /// Output pointer to serialized layout bytes.
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output byte length of serialized layout.
    pub serialized_bytes_size: usize,
    /// Output opaque serialized layout backing handle.
    pub serialized_layout: *mut std::ffi::c_void,
    /// Output deleter callback for `serialized_layout`.
    pub serialized_layout_deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
}

/// Argument structure for `PJRT_Layouts_PJRT_Client_GetDefaultLayout` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`PJRT_Client*`).
    pub client: *mut PJRT_Client,
    /// Element type (`PJRT_Buffer_Type`).
    pub elem_type: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
    /// Dimension array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Output default memory layout handle (`PJRT_Layouts_MemoryLayout*`).
    pub layout: *mut PJRT_Layouts_MemoryLayout,
}

/// Argument structure for `PJRT_Layouts_PJRT_Buffer_MemoryLayout` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle (`PJRT_Buffer*`).
    pub buffer: *mut PJRT_Buffer,
    /// Output memory layout handle (`PJRT_Layouts_MemoryLayout*`).
    pub layout: *mut PJRT_Layouts_MemoryLayout,
}

/// Argument structure for `PJRT_Shardings_PJRT_Executable_ParameterShardings` and `OutputShardings` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Shardings_PJRT_Executable_Shardings_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *mut PJRT_Executable,
    /// Number of parameters or outputs.
    pub num_items: usize,
    /// Output array of serialized `xla.OpSharding` byte pointers.
    pub shardings: *const *const std::os::raw::c_char,
    /// Output array of serialized `xla.OpSharding` byte lengths.
    pub sharding_sizes: *const usize,
}

/// Argument structure for `PJRT_PhaseCompile_GetPhaseNames` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_GetPhaseNames_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler handle (`const PJRT_PhaseCompile_Compiler*`).
    pub phase_compiler: *const PJRT_PhaseCompile_Compiler,
    /// Output array of phase name pointers.
    pub phase_names: *const *const std::os::raw::c_char,
    /// Output array of phase name byte lengths.
    pub phase_names_sizes: *const usize,
    /// Output number of compilation phases.
    pub num_phase_names: usize,
}

/// Argument structure for `PJRT_Triton_Compile` (`88` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Triton_Compile_Args {
    /// Size of this struct (`88`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Triton MLIR module bytes pointer.
    pub module: *const std::os::raw::c_char,
    /// Triton MLIR module byte length.
    pub module_size: usize,
    /// Target architecture string pointer.
    pub arch_name: *const std::os::raw::c_char,
    /// Target architecture string byte length.
    pub arch_name_size: usize,
    /// Number of warps / sublanes.
    pub num_warps: i32,
    /// Number of CTAs / cores.
    pub num_ctas: i32,
    /// Number of software pipeline stages.
    pub num_stages: i32,
    /// Explicit 4-byte padding.
    pub pad: i32,
    /// Output compiled assembly/payload pointer.
    pub out_asm: *const std::os::raw::c_char,
    /// Output compiled assembly/payload byte size.
    pub out_asm_size: usize,
    /// Output shared memory / VMEM scratchpad size in bytes.
    pub out_smem_bytes: i64,
}

/// Collective reduction operation kind (`PJRT_Collectives_ReductionKind`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PJRT_Collectives_ReductionKind {
    /// Element-wise summation (`0`).
    #[default]
    Sum = 0,
    /// Element-wise product (`1`).
    Product = 1,
    /// Element-wise minimum (`2`).
    Min = 2,
    /// Element-wise maximum (`3`).
    Max = 3,
}

/// Collective operation group mode (`PJRT_Collectives_CollectiveOpKind`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PJRT_Collectives_CollectiveOpKind {
    /// Cross-module (SPMD partition) collective (`0`).
    #[default]
    CrossModule = 0,
    /// Cross-replica (data-parallel) collective (`1`).
    CrossReplica = 1,
}

/// Runtime callback category (`PJRT_Callback_Type`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PJRT_Callback_Type {
    /// Unknown callback type (`0`).
    #[default]
    Unknown = 0,
    /// TPU slice builder failure callback (`1`).
    TpuSliceBuilder = 1,
    /// Pre-fatal error notification callback (`2`).
    Prefatal = 2,
}

/// TPU slice builder failure classification (`PJRT_Callback_Tpu_SliceFailureType`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PJRT_Callback_Tpu_SliceFailureType {
    /// Unknown slice failure (`0`).
    #[default]
    SliceFailureUnknown = 0,
    /// Initialization error (`1`).
    InitError = 1,
    /// Worker disconnected (`2`).
    WorkerDisconnected = 2,
    /// Heartbeat check failure (`3`).
    HeartbeatCheckFailure = 3,
    /// Fast-reroute ICI failure (`4`).
    FastRerouteFailure = 4,
    /// Chip driver error (`5`).
    ChipDriverError = 5,
}

/// Payload passed to `PJRT_Callback_Type::TpuSliceBuilder` callbacks (`16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Tpu_SliceBuilderArgs {
    /// Size of this struct (`16`).
    pub struct_size: usize,
    /// Failure type (`PJRT_Callback_Tpu_SliceFailureType`).
    pub failure_type: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
}

/// Payload passed to `PJRT_Callback_Type::Prefatal` callbacks (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_PrefatalArgs {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Error code (`PJRT_Error_Code`).
    pub error_code: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
    /// Pointer to error message text.
    pub error_message: *const std::os::raw::c_char,
    /// Byte length of `error_message`.
    pub error_message_size: usize,
}

/// Argument structure for `PJRT_HostMemoryAllocator_Allocate` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_HostMemoryAllocator_Allocate_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`PJRT_Client*`).
    pub client: *mut PJRT_Client,
    /// Requested allocation size in bytes.
    pub size: usize,
    /// Preferred NUMA node index.
    pub numa_node: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
    /// Output allocated host memory pointer.
    pub ptr: *mut std::ffi::c_void,
    /// Output deleter callback function pointer.
    pub deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    /// Output deleter user argument.
    pub deleter_arg: *mut std::ffi::c_void,
}

/// Serialized executable metadata structure (`PJRT_ExecutableMetadata`, `16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecutableMetadata {
    /// Pointer to serialized `xla.PjRtExecutableMetadata` protobuf bytes.
    pub serialized_metadata: *const std::os::raw::c_char,
    /// Byte length of `serialized_metadata`.
    pub serialized_metadata_size: usize,
}

/// Argument structure for `PJRT_DeviceDescription_MemoryDescriptions` (`48` bytes, `min_struct_size = 32`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_MemoryDescriptions_Args {
    /// Size of this struct (`48`, or legacy `32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle (`const PJRT_DeviceDescription*`).
    pub device_description: *const PJRT_DeviceDescription,
    /// Output array of `const PJRT_MemoryDescription*` handles.
    pub memory_descriptions: *const *const PJRT_MemoryDescription,
    /// Output number of memory descriptions (written when `struct_size >= 40`).
    pub num_memory_descriptions: usize,
    /// Output index of default memory space (`0` = HBM, written when `struct_size >= 48`).
    pub default_memory_index: isize,
}

impl ExtensionArgsStruct for PJRT_DeviceDescription_MemoryDescriptions_Args {
    const MIN_STRUCT_SIZE: usize = 32;
}

/// Argument structure for `PJRT_MemoryDescription_Kind` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MemoryDescription_Kind_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory description handle (`const PJRT_MemoryDescription*`).
    pub memory_description: *const PJRT_MemoryDescription,
    /// Output memory space kind string pointer.
    pub kind: *const std::os::raw::c_char,
    /// Output memory space kind string byte length.
    pub kind_size: usize,
    /// Output numeric kind ID (`0` = device/HBM, `1` = vmem, `2` = smem, `3` = `pinned_host`).
    pub kind_id: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
}

/// Optional C transform hook signature for `PJRT_XlaTransform_Register`.
pub type PjrtXlaTransformCallbackFn =
    unsafe extern "C" fn(*const std::os::raw::c_char, usize, *mut std::ffi::c_void) -> i32;

/// Argument structure for `PJRT_XlaTransform_Register` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_Register_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transform pass name pointer.
    pub name: *const std::os::raw::c_char,
    /// Transform pass name byte length.
    pub name_size: usize,
    /// Optional C transform callback or output registration ID.
    pub transform_fn: usize,
}

/// Argument structure for `PJRT_XlaTransform_WaitForAll` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_WaitForAll_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`PJRT_Client*`).
    pub client: *mut PJRT_Client,
}

/// Argument structure for `PJRT_XlaTransform_Unregister` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_Unregister_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transform pass name pointer.
    pub name: *const std::os::raw::c_char,
    /// Transform pass name byte length.
    pub name_size: usize,
}

/// Argument structure for `PJRT_XlaTransform_List` (`40` bytes, `min_struct_size = 32`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_List_Args {
    /// Size of this struct (`32` or `40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output array of owned null-terminated transform pass name pointers.
    pub names_out: *const *const std::os::raw::c_char,
    /// Output number of registered transform passes.
    pub num_names_out: usize,
    /// Output array of transform pass byte lengths (written when `struct_size >= 40`).
    pub name_sizes_out: *const usize,
}

impl ExtensionArgsStruct for PJRT_XlaTransform_List_Args {
    const MIN_STRUCT_SIZE: usize = 32;
}

/// Argument structure for `PJRT_HostMemoryAllocator_Get` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_HostMemoryAllocator_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`PJRT_Client*`).
    pub client: *mut PJRT_Client,
    /// Output host memory allocator capacity / descriptor.
    pub allocator_out: usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_Serialize` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_Serialize_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice config handle (`const PJRT_MultiSlice_Config*`).
    pub config: *const PJRT_MultiSlice_Config,
    /// Output serialized proto byte pointer.
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output serialized proto byte length.
    pub serialized_bytes_size: usize,
    /// Output opaque serialized proto holder.
    pub serialized_config: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_AllReduce` and `ReduceScatter` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_AllReduce_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle (`PJRT_Collectives_Communicator*`).
    pub communicator: *mut PJRT_Collectives_Communicator,
    /// Send buffer handle (`PJRT_Buffer*`).
    pub send_buffer: *mut PJRT_Buffer,
    /// Receive buffer handle (`PJRT_Buffer*`).
    pub recv_buffer: *mut PJRT_Buffer,
    /// Reduction operation code (`PJRT_Collectives_ReductionKind`).
    pub reduction_op: i32,
    /// Alignment padding.
    pub pad: i32,
    /// Element count.
    pub count: usize,
    /// Output completion event (`PJRT_Event*`).
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_AbiVersion_RuntimeToProto` and `ExecutableToProto` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_ToProto_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle (`const PJRT_AbiVersion_Holder*`).
    pub abi_version: *const PJRT_AbiVersion_Holder,
    /// Output serialized proto pointer.
    pub serialized_proto: *const std::os::raw::c_char,
    /// Output serialized proto byte length.
    pub serialized_proto_size: usize,
    /// Output serialized proto holder.
    pub serialized_proto_holder: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_TpuExecutable_GetCompiledMemoryStats` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *mut PJRT_Executable,
    /// Generated code size in bytes.
    pub generated_code_size_in_bytes: i64,
    /// Argument size in bytes.
    pub argument_size_in_bytes: i64,
    /// Output size in bytes.
    pub output_size_in_bytes: i64,
    /// Alias size in bytes.
    pub alias_size_in_bytes: i64,
    /// Temporary buffer size in bytes.
    pub temp_size_in_bytes: i64,
}

/// C callback function pointer type for `PJRT_Callback_Extension`.
pub type PjrtRuntimeCallbackFn = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void);

/// Argument structure for `PJRT_Callback_RegisterCallback` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Register_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle (`PJRT_Client*`).
    pub client: *mut PJRT_Client,
    /// Callback type identifier.
    pub callback_type: i64,
    /// Callback function pointer.
    pub callback_fn: Option<PjrtRuntimeCallbackFn>,
    /// User context pointer.
    pub user_arg: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_PhaseCompile_RunPhase` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_RunPhase_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler handle (`PJRT_PhaseCompile_Compiler*`).
    pub phase_compiler: *mut PJRT_PhaseCompile_Compiler,
    /// Input program bytes pointer.
    pub input_bytes: *const std::os::raw::c_char,
    /// Input program byte length.
    pub input_size: usize,
    /// Phase name pointer.
    pub phase_name: *const std::os::raw::c_char,
    /// Output artifact byte count.
    pub output_status: usize,
}

/// Argument structure for `PJRT_RawBuffer_CopyRawHostToDevice` and `CopyRawDeviceToHost` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_CopyRaw_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Raw buffer handle (`PJRT_RawBuffer*`).
    pub buffer: *mut PJRT_RawBuffer,
    /// Host source or destination pointer.
    pub host_ptr: *mut std::ffi::c_void,
    /// Byte offset within the raw device buffer.
    pub offset: i64,
    /// Number of bytes to transfer.
    pub transfer_size: i64,
    /// Output completion event (`PJRT_Event*`).
    pub event: *mut PJRT_Event,
}

/// `OpenXLA` Typed FFI API version structure (`XLA_FFI_Api_Version`, `24` bytes, v0.2).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XLA_FFI_Api_Version {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Major version (`0`).
    pub major_version: i32,
    /// Minor version (`2`).
    pub minor_version: i32,
}

/// `OpenXLA` Typed FFI buffer descriptor passed inside [`XLA_FFI_CallFrame`] (`XLA_FFI_Buffer`, `40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XLA_FFI_Buffer {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Element data type (`PJRT_Buffer_Type`).
    pub dtype: i32,
    /// Rank of the buffer (`dims` length).
    pub rank: i32,
    /// Mutable device/host data pointer.
    pub data: *mut std::ffi::c_void,
    /// Pointer to `rank` dimension extents (`const int64_t*`).
    pub dims: *const i64,
}

/// `OpenXLA` Typed FFI call frame passed to `XLA_FFI_Handler` callbacks (`XLA_FFI_CallFrame`, `64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XLA_FFI_CallFrame {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Pointer to the host `XLA_FFI_Api` vtable.
    pub api: *const XLA_FFI_Api,
    /// Opaque execution context pointer.
    pub ctx: *mut std::ffi::c_void,
    /// Execution stage (`0 = Instantiate`, `1 = Prepare`, `2 = Initialize`, `3 = Execute`).
    pub stage: i32,
    /// Number of input [`XLA_FFI_Buffer`] arguments.
    pub num_args: u32,
    /// Array of `num_args` input [`XLA_FFI_Buffer`] pointers.
    pub args: *const *const XLA_FFI_Buffer,
    /// Number of output [`XLA_FFI_Buffer`] return slots.
    pub num_rets: usize,
    /// Array of `num_rets` output [`XLA_FFI_Buffer`] pointers.
    pub rets: *const *mut XLA_FFI_Buffer,
}

/// `OpenXLA` Typed FFI handler callback signature (`XLA_FFI_Error* (*)(XLA_FFI_CallFrame*)`).
pub type XLA_FFI_Handler = unsafe extern "C" fn(*mut XLA_FFI_CallFrame) -> *mut PJRT_Error;

/// `OpenXLA` Typed FFI vtable (`XLA_FFI_Api`, `56` bytes, v0.2).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XLA_FFI_Api {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Embedded `XLA_FFI_Api_Version` (`0.2`).
    pub api_version: XLA_FFI_Api_Version,
    /// Handler registration entrypoint (`PJRT_FFI_Register_Handler`).
    pub register_handler: unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
    /// Type ID registration entrypoint (`PJRT_FFI_TypeID_Register`).
    pub register_type_id: unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
}

// SAFETY: Contains only immutable function pointers and version integers.
unsafe impl Sync for XLA_FFI_Api {}

/// Custom type lifecycle callbacks for `PJRT_FFI_Type_Register` (`PJRT_FFI_Type_Info`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_FFI_Type_Info {
    /// Optional C deleter (`void (*)(void*)`).
    pub deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    /// Reserved serialization callback.
    pub serialize: *const std::ffi::c_void,
    /// Reserved deserialization callback.
    pub deserialize: *const std::ffi::c_void,
}

/// `#[repr(C)]` argument structure for `PJRT_FFI_TypeID_Register` (`48` bytes, `min_struct_size = 40`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_FFI_TypeID_Register_Args {
    /// Size of this struct (`40` or `48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Pointer to custom type name UTF-8 bytes.
    pub type_name: *const std::os::raw::c_char,
    /// Byte length of `type_name`.
    pub type_name_size: usize,
    /// Output inline `int64_t` registered type ID (`>= 1`).
    pub type_id: i64,
    /// Optional pointer to [`PJRT_FFI_Type_Info`] (`0.0.53.1`).
    pub type_info: *const PJRT_FFI_Type_Info,
}

impl ExtensionArgsStruct for PJRT_FFI_TypeID_Register_Args {
    const MIN_STRUCT_SIZE: usize = 40;
}

/// `#[repr(C)]` argument structure for `PJRT_FFI_UserData_Add` (`48` bytes, `min_struct_size = 40`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_FFI_UserData_Add_Args {
    /// Size of this struct (`40` or `48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Target `PJRT_ExecuteContext*` handle.
    pub context: *mut PJRT_ExecuteContext,
    /// Registered type ID (`int64_t`).
    pub type_id: i64,
    /// User data opaque pointer.
    pub data: *mut std::ffi::c_void,
    /// Optional C deleter invoked when context finishes (`void (*)(void*)`).
    pub deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
}

impl ExtensionArgsStruct for PJRT_FFI_UserData_Add_Args {
    const MIN_STRUCT_SIZE: usize = 40;
}

/// `#[repr(C)]` argument structure for `PJRT_FFI_Register_Handler` (`64` bytes, `min_struct_size = 52`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_FFI_Register_Handler_Args {
    /// Size of this struct (`52`, `56`, or `64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Custom-call target name UTF-8 pointer.
    pub target_name: *const std::os::raw::c_char,
    /// Byte length of `target_name`.
    pub target_name_size: usize,
    /// Typed `XLA_FFI_Handler` function pointer.
    pub handler: Option<XLA_FFI_Handler>,
    /// Platform name UTF-8 pointer (`"TPU"`).
    pub platform_name: *const std::os::raw::c_char,
    /// Byte length of `platform_name`.
    pub platform_name_size: usize,
    /// Bitmask of `XLA_FFI_Handler_TraitsBits` (`0` = default).
    pub traits: u32,
    /// Reserved alignment padding.
    pub pad: u32,
}

impl ExtensionArgsStruct for PJRT_FFI_Register_Handler_Args {
    const MIN_STRUCT_SIZE: usize = 52;
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcessCount` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcessCount_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output process count.
    pub process_count: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipsPerProcess` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipsPerProcess_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output chips per process count.
    pub chips_per_process: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_CoreCountPerChip` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_CoreCountPerChip_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output cores per chip count.
    pub core_count_per_chip: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipCount` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipCount_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output total chip count.
    pub chip_count: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_CoreCount` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_CoreCount_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output total core count.
    pub core_count: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_LogiDeviceCountPerProcess` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_LogiDeviceCountPerProcess_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output logical devices per process count.
    pub logi_device_count_per_process: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_LogiDeviceCount` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_LogiDeviceCount_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output total logical device count.
    pub logi_device_count: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_LogiDeviceCountPerChip` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_LogiDeviceCountPerChip_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output logical devices per chip count.
    pub logi_device_count_per_chip: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_CoreCountPerProcess` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_CoreCountPerProcess_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output cores per process count.
    pub core_count_per_process: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_HasLimitedIciConnectivity` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_HasLimitedIciConnectivity_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output boolean flag (`0` or `1`).
    pub has_limited_ici: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_IsSubsliceTopology` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_IsSubsliceTopology_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output boolean flag (`0` or `1`).
    pub is_subslice: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_IsValidSubslice` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_IsValidSubslice_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output boolean flag (`0` or `1`).
    pub is_valid_subslice: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcessBounds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcessBounds_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Optional output pointer to `int32_t[3]` coordinate bounds array.
    pub coords_out: *mut i32,
    /// Output dimension count (`3`).
    pub num_coords: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipsPerProcessBounds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipsPerProcessBounds_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Optional output pointer to `int32_t[3]` coordinate bounds array.
    pub coords_out: *mut i32,
    /// Output dimension count (`3`).
    pub num_coords: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipBounds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipBounds_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Optional output pointer to `int32_t[3]` coordinate bounds array.
    pub coords_out: *mut i32,
    /// Output dimension count (`3`).
    pub num_coords: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcessCoordFromId` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcessCoordFromId_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input process ID.
    pub process_id: usize,
    /// Optional output pointer to `int32_t[3]` coordinate array.
    pub coords_out: *mut i32,
    /// Output coordinate dimension count (`3`).
    pub num_coords: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipIdFromCoord` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipIdFromCoord_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input pointer to `const int32_t[3]` coordinates (or null for `(0, 0, 0)`).
    pub coords: *const i32,
    /// Output global chip ID.
    pub chip_id_out: usize,
    /// Output chip index on owning process.
    pub idx_on_proc_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input pointer to `const int32_t[3]` chip coordinates (or null for `(0, 0, 0)`).
    pub coords: *const i32,
    /// Input core index on chip (`out` receives logical device ID when `coords` is null).
    pub core_idx: usize,
    /// Output logical device ID.
    pub device_id_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input logical device ID.
    pub device_id: usize,
    /// Optional output pointer to `int32_t[3]` chip coordinates.
    pub coords_out: *mut i32,
    /// Output core index on chip.
    pub core_idx_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcIdAndIdxOnProcForChip` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input global chip ID.
    pub chip_id: usize,
    /// Output process ID.
    pub out_proc_id: usize,
    /// Output chip index on process.
    pub out_idx_on_proc: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input logical device ID.
    pub device_id: usize,
    /// Output process ID.
    pub out_proc_id: usize,
    /// Output device index on process.
    pub out_idx_on_proc: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcessIds` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcessIds_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Optional output pointer to `int32_t*` process IDs buffer.
    pub process_ids_out: *mut i32,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipsOnProcess` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipsOnProcess_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Target process ID.
    pub process_id: usize,
    /// Output chip count on process.
    pub num_chips: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_LogiDeviceIdsOnProcess` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Optional output pointer to `int32_t*` logical device IDs buffer.
    pub device_ids_out: *mut i32,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_GetSliceConfig` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_GetSliceConfig_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Optional output `PJRT_TpuTopology_SliceConfig*` pointer.
    pub slice_config_out: *mut PJRT_TpuTopology_SliceConfig,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_GetSliceConfigs` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_GetSliceConfigs_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Optional output `PJRT_TpuTopology_SliceConfig*` array pointer.
    pub slice_configs_out: *mut PJRT_TpuTopology_SliceConfig,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_GetDefaultPlatformConfig` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_GetDefaultPlatformConfig_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output default platform generation ID.
    pub platform_config: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_GetRoutingStrategy` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_GetRoutingStrategy_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output routing strategy code (`0 = DoradoShortestPath`).
    pub routing_strategy: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_CreateClientContext` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_CreateClientContextFromPjRtClient_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_Client*` handle.
    pub client: *mut PJRT_Client,
    /// Output `PJRT_Megascale_ClientContext*` handle.
    pub context_out: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_CreateDefaultClientContext` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_CreateDefaultClientContext_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional configuration flags.
    pub flags: usize,
    /// Output `PJRT_Megascale_ClientContext*` handle.
    pub context_out: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_DeleteClientContext` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_DeleteClientContext_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle to delete.
    pub context: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_InitializeClientContext` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_InitializeClientContext_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output initialization status (`1 = initialized`).
    pub status_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_UnblockPendingWork` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_UnblockPendingWork_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_CreateAotConfig` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_CreateAoTMegascaleConfig_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional `const PJRT_TopologyDescription*` handle (or null for default config).
    pub topology: *const PJRT_TopologyDescription,
    /// Output `PJRT_MultiSlice_Config*` handle.
    pub config_out: *mut PJRT_MultiSlice_Config,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_CreateMultiSliceConfig` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_CreateMultiSliceConfig_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional `const PJRT_TopologyDescription*` handle (or null for default config).
    pub topology: *const PJRT_TopologyDescription,
    /// Output `PJRT_MultiSlice_Config*` handle.
    pub config_out: *mut PJRT_MultiSlice_Config,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_DeleteMultiSliceConfig` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_DeleteMultiSliceConfig_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_MultiSlice_Config*` handle to delete.
    pub config: *mut PJRT_MultiSlice_Config,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_ConfigNumSlices` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_ConfigNumSlices_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Output slice count.
    pub num_slices: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_ConfigNumDevicesPerSlice` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_ConfigNumDevicesPerSlice_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Output devices per slice count.
    pub num_devices_per_slice: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_ConfigSliceId` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_ConfigSliceId_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Output local slice ID.
    pub slice_id: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_ConfigSerializedSliceKeyValue` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_ConfigSerializedSliceKeyValue_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Output serialized key-value byte length.
    pub serialized_kv_out: usize,
    /// Output key-value entry count.
    pub num_entries_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_ConfigSerialize` (`48` bytes).
pub type PJRT_Megascale_ConfigSerialize_Args = PJRT_MultiSlice_Config_Serialize_Args;

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_DeserializeMultiSliceConfig` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_DeserializeMultiSliceConfig_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional serialized bytes pointer (`const char*`).
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output `PJRT_MultiSlice_Config*` handle.
    pub config_out: *mut PJRT_MultiSlice_Config,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_Addresses` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Addresses_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output endpoint address count.
    pub num_addresses: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_Barrier` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Barrier_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output completed barrier sequence number.
    pub completed_barriers: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_WaitForBarrier` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_WaitForBarrier_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Barrier or context handle to await.
    pub handle: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_GetDcnTopology` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_GetDcnTopology_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output serialized `MegascaleDcnTopologyProto` byte length.
    pub serialized_size: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_GetSliceId` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_GetSliceId_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output active slice ID.
    pub slice_id: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_GetNumSlices` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_GetNumSlices_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output total slice count.
    pub num_slices: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_InitializeSlice` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_InitializeSlice_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output initialization status (`1 = initialized`).
    pub status_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_FinalizeSlice` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_FinalizeSlice_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub handle: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_RegisterErrorHandler` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_RegisterErrorHandler_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional C error callback function pointer.
    pub callback_fn: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    /// Output registered handler count or ID.
    pub handler_id_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_UnregisterErrorHandler` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_UnregisterErrorHandler_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Registered error handler handle or callback pointer.
    pub handle: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_CollectMetrics` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_CollectMetrics_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output pointer to serialized `MegascaleRuntimeMetricsProto`.
    pub serialized_metrics_ptr: *const u8,
    /// Output serialized byte length.
    pub serialized_metrics_size: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_ResetMetrics` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_ResetMetrics_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub handle: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_GetTransportType` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_GetTransportType_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub context: *mut PJRT_Megascale_ClientContext,
    /// Output transport type (`0 = ICI`, `1 = DCN`, `2 = RoCEv2`).
    pub transport_type_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_SetTransportType` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_SetTransportType_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Requested transport type (`0 = ICI`, `1 = DCN`, `2 = RoCEv2`).
    pub requested_transport_type: usize,
    /// Output active transport type.
    pub transport_type_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_AbortCollectives` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_AbortCollectives_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub handle: *mut PJRT_Megascale_ClientContext,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_LogError` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_LogError_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error payload pointer or handle.
    pub handle: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Megascale_DiscoverSlices` (`24` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_DiscoverSlices_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// `PJRT_Megascale_ClientContext*` handle.
    pub client_context: *mut PJRT_Megascale_ClientContext,
    /// Output discovered `PJRT_MultiSlice_Config*` handle.
    pub multi_slice_config: *mut PJRT_MultiSlice_Config,
}

/// Argument structure for `PJRT_Megascale_MultiSliceConfig_GetIncarnationId` (`24` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_MultiSliceConfig_GetIncarnationId_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Output slice incarnation identifier.
    pub incarnation_id: u64,
}

/// Argument structure for `PJRT_Megascale_MultiSliceConfig_GetLocalHostId` (`24` bytes, `min_struct_size = 20`, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args {
    /// Size of this struct (`20` or `24`).
    pub struct_size: usize,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Output local host identifier.
    pub local_host_id: i32,
    /// Alignment padding (`4` bytes).
    pub pad: i32,
}

impl ExtensionArgsStruct for PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args {
    const MIN_STRUCT_SIZE: usize = 20;
}

/// Argument structure for `PJRT_Megascale_MultiSliceConfig_GetNumHosts` (`24` bytes, `min_struct_size = 20`, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_MultiSliceConfig_GetNumHosts_Args {
    /// Size of this struct (`20` or `24`).
    pub struct_size: usize,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Output total host count.
    pub num_hosts: i32,
    /// Alignment padding (`4` bytes).
    pub pad: i32,
}

impl ExtensionArgsStruct for PJRT_Megascale_MultiSliceConfig_GetNumHosts_Args {
    const MIN_STRUCT_SIZE: usize = 20;
}

/// Argument structure for `PJRT_Megascale_GlobalBarrier` (`40` bytes, `0.0.53.1`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_GlobalBarrier_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// `const PJRT_MultiSlice_Config*` handle.
    pub config: *const PJRT_MultiSlice_Config,
    /// Barrier name UTF-8 pointer.
    pub barrier_id: *const std::os::raw::c_char,
    /// Barrier name byte length.
    pub barrier_id_size: usize,
    /// Timeout in milliseconds (`int64_t`).
    pub timeout_ms: i64,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_RawBuffer_CreateRawAliasOfBuffer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_Buffer*` handle.
    pub buffer: *mut PJRT_Buffer,
    /// Output `PJRT_RawBuffer*` handle.
    pub raw_buffer_out: *mut PJRT_RawBuffer,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_RawBuffer_GetHostPointer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_GetHostPointer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `const PJRT_RawBuffer*` handle.
    pub raw_buffer: *const PJRT_RawBuffer,
    /// Output device/host address (`*mut c_void`).
    pub host_pointer_out: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_RawBuffer_GetOnDeviceSizeInBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `const PJRT_RawBuffer*` handle.
    pub raw_buffer: *const PJRT_RawBuffer,
    /// Output on-device size in bytes.
    pub on_device_size_in_bytes: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_RawBuffer_GetMemorySpace` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_GetMemorySpace_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `const PJRT_RawBuffer*` handle.
    pub raw_buffer: *const PJRT_RawBuffer,
    /// Output `PJRT_Memory*` handle.
    pub memory_space_out: *mut PJRT_Memory,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetTargetArguments` (`48` bytes).
pub type PJRT_TpuExecutable_TargetArguments_Args = PJRT_TpuExecutable_ProtoQuery_Args;

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_CoreProgramAbiVersion` (`48` bytes).
pub type PJRT_TpuExecutable_CoreProgramAbiVersion_Args = PJRT_TpuExecutable_ProtoQuery_Args;

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetHloModuleWithConfig` (`48` bytes).
pub type PJRT_TpuExecutable_HloModule_Args = PJRT_TpuExecutable_ProtoQuery_Args;

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString` (`48` bytes).
pub type PJRT_TpuExecutable_Serialize_Args = PJRT_TpuExecutable_ProtoQuery_Args;

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetFingerprint` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_GetFingerprint_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`const PJRT_Executable*`).
    pub executable: *const PJRT_Executable,
    /// Output fingerprint UTF-8 bytes pointer.
    pub fingerprint_bytes: *const std::os::raw::c_char,
    /// Output fingerprint byte length.
    pub fingerprint_size: usize,
    /// Output artifact holder handle (`PJRT_ExecutableMetadata_Handle*`).
    pub holder: *mut PJRT_ExecutableMetadata_Handle,
}

impl_default_extension_args!(
    PLUGIN_Profiler_Error_Destroy_Args,
    PLUGIN_Profiler_Error_Message_Args,
    PLUGIN_Profiler_Create_Args,
    PLUGIN_Profiler_Destroy_Args,
    PLUGIN_Profiler_Start_Args,
    PLUGIN_Profiler_Stop_Args,
    PLUGIN_Profiler_CollectData_Args,
    PLUGIN_Profiler_Consume_Args,
    PLUGIN_Profiler_ConsumeResult_Destroy_Args,
    PLUGIN_Profiler_Serialize_Args,
    PJRT_MultiSlice_Config_Destroy_Args,
    PJRT_MultiSlice_Config_NumSlices_Args,
    PJRT_MultiSlice_Config_NumDevicesPerSlice_Args,
    PJRT_MultiSlice_Config_SliceId_Args,
    PJRT_MultiSlice_Config_Serialize_Args,
    PJRT_Collectives_Destroy_Args,
    PJRT_Collectives_CreateCommunicator_Args,
    PJRT_Communicator_Destroy_Args,
    PJRT_Communicator_AllGather_Args,
    PJRT_Communicator_CollectivePermute_Args,
    PJRT_Communicator_ToString_Args,
    PJRT_Communicator_AllReduce_Args,
    PJRT_Client_RuntimeAbiVersion_Args,
    PJRT_Executable_GetAbiVersion_Args,
    PJRT_AbiVersion_Get_Args,
    PJRT_AbiVersion_Destroy_Args,
    PJRT_AbiVersion_IsCompatible_Args,
    PJRT_AbiVersion_PlatformId_Args,
    PJRT_AbiVersion_FromProto_Args,
    PJRT_AbiVersion_ToProto_Args,
    PJRT_Megascale_Handle_Args,
    PJRT_Megascale_Query_Args,
    PJRT_Megascale_Data_Args,
    PJRT_TpuExecutable_RunHloCostAnalysis_Args,
    PJRT_TpuExecutable_SetTpuCompilationEnv_Args,
    PJRT_TpuExecutable_ProtoQuery_Args,
    PJRT_TpuExecutable_ErrorCheck_Args,
    PJRT_TpuExecutable_GetCompiledMemoryStats_Args,
    PJRT_TpuExecutable_GetFingerprint_Args,
    PJRT_TpuTopology_ScalarQuery_Args,
    PJRT_TpuTopology_BoundsQuery_Args,
    PJRT_TpuTopology_MappingQuery_Args,
    PJRT_TpuTopology_ComputeSubsliceForDevices_Args,
    PJRT_TpuTopology_ProcessCount_Args,
    PJRT_TpuTopology_ChipsPerProcess_Args,
    PJRT_TpuTopology_CoreCountPerChip_Args,
    PJRT_TpuTopology_ChipCount_Args,
    PJRT_TpuTopology_CoreCount_Args,
    PJRT_TpuTopology_LogiDeviceCountPerProcess_Args,
    PJRT_TpuTopology_LogiDeviceCount_Args,
    PJRT_TpuTopology_LogiDeviceCountPerChip_Args,
    PJRT_TpuTopology_CoreCountPerProcess_Args,
    PJRT_TpuTopology_HasLimitedIciConnectivity_Args,
    PJRT_TpuTopology_IsSubsliceTopology_Args,
    PJRT_TpuTopology_IsValidSubslice_Args,
    PJRT_TpuTopology_ProcessBounds_Args,
    PJRT_TpuTopology_ChipsPerProcessBounds_Args,
    PJRT_TpuTopology_ChipBounds_Args,
    PJRT_TpuTopology_ProcessCoordFromId_Args,
    PJRT_TpuTopology_ChipIdFromCoord_Args,
    PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args,
    PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args,
    PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args,
    PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args,
    PJRT_TpuTopology_ProcessIds_Args,
    PJRT_TpuTopology_ChipsOnProcess_Args,
    PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args,
    PJRT_TpuTopology_GetSliceConfig_Args,
    PJRT_TpuTopology_GetSliceConfigs_Args,
    PJRT_TpuTopology_GetDefaultPlatformConfig_Args,
    PJRT_TpuTopology_GetRoutingStrategy_Args,
    PJRT_Callback_Invoke_Args,
    PJRT_Callback_Register_Args,
    PJRT_PhaseCompile_GetCompiler_Args,
    PJRT_PhaseCompile_Destroy_Args,
    PJRT_PhaseCompile_RunPhase_Args,
    PJRT_PhaseCompile_GetPhaseNames_Args,
    PJRT_Relocatable_Destroy_Args,
    PJRT_Relocatable_StringQuery_Args,
    PJRT_Relocatable_Serialize_Args,
    PJRT_Relocatable_ElementTypes_Args,
    PJRT_Relocatable_Dimensions_Args,
    PJRT_Relocatable_Layouts_Args,
    PJRT_Relocatable_ParameterOutputAliases_Args,
    PJRT_PhaseCompile_CompileToRelocatable_Args,
    PJRT_PhaseCompile_DeserializeRelocatable_Args,
    PJRT_PhaseCompile_LinkRelocatables_Args,
    PJRT_CrossHostTransfers_Args,
    PJRT_Metadata_Get_Args,
    PJRT_Metadata_Destroy_Args,
    PJRT_RawBuffer_Query_Args,
    PJRT_RawBuffer_Destroy_Args,
    PJRT_RawBuffer_CopyRaw_Args,
    PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args,
    PJRT_RawBuffer_GetHostPointer_Args,
    PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args,
    PJRT_RawBuffer_GetMemorySpace_Args,
    PJRT_Layouts_MemoryLayout_Destroy_Args,
    PJRT_Layouts_MemoryLayout_Serialize_Args,
    PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args,
    PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args,
    PJRT_Shardings_PJRT_Executable_Shardings_Args,
    PJRT_Triton_Compile_Args,
    PJRT_HostMemoryAllocator_Allocate_Args,
    PJRT_HostMemoryAllocator_Get_Args,
    PJRT_MemoryDescription_Kind_Args,
    PJRT_XlaTransform_Register_Args,
    PJRT_XlaTransform_WaitForAll_Args,
    PJRT_XlaTransform_Unregister_Args,
    PJRT_Megascale_CreateClientContextFromPjRtClient_Args,
    PJRT_Megascale_CreateDefaultClientContext_Args,
    PJRT_Megascale_DeleteClientContext_Args,
    PJRT_Megascale_InitializeClientContext_Args,
    PJRT_Megascale_UnblockPendingWork_Args,
    PJRT_Megascale_CreateAoTMegascaleConfig_Args,
    PJRT_Megascale_CreateMultiSliceConfig_Args,
    PJRT_Megascale_DeleteMultiSliceConfig_Args,
    PJRT_Megascale_ConfigNumSlices_Args,
    PJRT_Megascale_ConfigNumDevicesPerSlice_Args,
    PJRT_Megascale_ConfigSliceId_Args,
    PJRT_Megascale_ConfigSerializedSliceKeyValue_Args,
    PJRT_Megascale_DeserializeMultiSliceConfig_Args,
    PJRT_Megascale_Addresses_Args,
    PJRT_Megascale_Barrier_Args,
    PJRT_Megascale_WaitForBarrier_Args,
    PJRT_Megascale_GetDcnTopology_Args,
    PJRT_Megascale_GetSliceId_Args,
    PJRT_Megascale_GetNumSlices_Args,
    PJRT_Megascale_InitializeSlice_Args,
    PJRT_Megascale_FinalizeSlice_Args,
    PJRT_Megascale_RegisterErrorHandler_Args,
    PJRT_Megascale_UnregisterErrorHandler_Args,
    PJRT_Megascale_CollectMetrics_Args,
    PJRT_Megascale_ResetMetrics_Args,
    PJRT_Megascale_GetTransportType_Args,
    PJRT_Megascale_SetTransportType_Args,
    PJRT_Megascale_AbortCollectives_Args,
    PJRT_Megascale_LogError_Args,
    PJRT_Megascale_DiscoverSlices_Args,
    PJRT_Megascale_MultiSliceConfig_GetIncarnationId_Args,
    PJRT_Megascale_GlobalBarrier_Args,
);

const _: () = assert!(size_of::<PLUGIN_Profiler_Error_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_Error_Message_Args>() == 40);
const _: () = assert!(size_of::<PLUGIN_Profiler_Error_GetCode_Args>() == 32);
const _: () = assert!(size_of::<PLUGIN_Profiler_Create_Args>() == 40);
const _: () = assert!(size_of::<PLUGIN_Profiler_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_Start_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_Stop_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_CollectData_Args>() == 40);
const _: () = assert!(size_of::<PLUGIN_Profiler_Consume_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_ConsumeResult_Destroy_Args>() == 16);
const _: () = assert!(size_of::<PLUGIN_Profiler_Serialize_Args>() == 40);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_NumSlices_Args>() == 32);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_NumDevicesPerSlice_Args>() == 40);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_SliceId_Args>() == 32);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_Serialize_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Collectives_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Collectives_CreateCommunicator_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Communicator_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Communicator_AllGather_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Communicator_CollectivePermute_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Communicator_ToString_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Communicator_AllReduce_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Client_RuntimeAbiVersion_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Executable_GetAbiVersion_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AbiVersion_Get_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AbiVersion_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_AbiVersion_IsCompatible_Args>() == 40);
const _: () = assert!(size_of::<PJRT_AbiVersion_PlatformId_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AbiVersion_FromProto_Args>() == 40);
const _: () = assert!(size_of::<PJRT_AbiVersion_ToProto_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Megascale_Handle_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_Query_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_Data_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuExecutable_RunHloCostAnalysis_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuExecutable_SetTpuCompilationEnv_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuExecutable_ProtoQuery_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuExecutable_ErrorCheck_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuExecutable_GetCompiledMemoryStats_Args>() == 64);
const _: () = assert!(size_of::<PJRT_TpuExecutable_GetFingerprint_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_ScalarQuery_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_BoundsQuery_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuTopology_MappingQuery_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_SliceConfig>() == 56);
const _: () = assert!(size_of::<PJRT_TpuTopology_ComputeSubsliceForDevices_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Callback_Invoke_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Callback_Register_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Callback_Tpu_SliceBuilderArgs>() == 16);
const _: () = assert!(size_of::<PJRT_Callback_PrefatalArgs>() == 32);
const _: () = assert!(size_of::<PJRT_PhaseCompile_GetCompiler_Args>() == 32);
const _: () = assert!(size_of::<PJRT_PhaseCompile_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_PhaseCompile_RunPhase_Args>() == 56);
const _: () = assert!(size_of::<PJRT_PhaseCompile_GetPhaseNames_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Relocatable_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Relocatable_StringQuery_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Relocatable_HasSideEffects_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Relocatable_Serialize_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Relocatable_ElementTypes_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Relocatable_Dimensions_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Relocatable_Layouts_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Relocatable_ParameterOutputAliases_Args>() == 40);
const _: () = assert!(size_of::<PJRT_PhaseCompile_CompileToRelocatable_Args>() == 88);
const _: () = assert!(size_of::<PJRT_PhaseCompile_DeserializeRelocatable_Args>() == 48);
const _: () = assert!(size_of::<PJRT_PhaseCompile_LinkRelocatables_Args>() == 80);
const _: () = assert!(size_of::<PJRT_CrossHostTransfers_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Metadata_Get_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Metadata_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_ExecutableMetadata>() == 16);
const _: () = assert!(size_of::<PJRT_Layouts_PJRT_Executable_Layouts_Args>() == 40);
const _: () = assert!(size_of::<PJRT_RawBuffer_Query_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_RawBuffer_CopyRaw_Args>() == 56);
const _: () = assert!(size_of::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_GetHostPointer_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_GetMemorySpace_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Layouts_MemoryLayout_Serialize_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Shardings_PJRT_Executable_Shardings_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Triton_Compile_Args>() == 88);
const _: () = assert!(size_of::<PJRT_HostMemoryAllocator_Allocate_Args>() == 64);
const _: () = assert!(size_of::<PJRT_HostMemoryAllocator_Get_Args>() == 32);
const _: () = assert!(size_of::<PJRT_DeviceDescription_MemoryDescriptions_Args>() == 48);
const _: () = assert!(size_of::<PJRT_MemoryDescription_Kind_Args>() == 48);
const _: () = assert!(size_of::<PJRT_XlaTransform_Register_Args>() == 40);
const _: () = assert!(size_of::<PJRT_XlaTransform_WaitForAll_Args>() == 24);
const _: () = assert!(size_of::<PJRT_XlaTransform_Unregister_Args>() == 32);
const _: () = assert!(size_of::<PJRT_XlaTransform_List_Args>() == 40);
const _: () = assert!(size_of::<XLA_FFI_Api_Version>() == 24);
const _: () = assert!(size_of::<XLA_FFI_Buffer>() == 40);
const _: () = assert!(size_of::<XLA_FFI_CallFrame>() == 64);
const _: () = assert!(size_of::<XLA_FFI_Api>() == 56);
const _: () = assert!(size_of::<PJRT_FFI_Type_Info>() == 24);
const _: () = assert!(size_of::<PJRT_FFI_TypeID_Register_Args>() == 48);
const _: () = assert!(size_of::<PJRT_FFI_UserData_Add_Args>() == 48);
const _: () = assert!(size_of::<PJRT_FFI_Register_Handler_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Megascale_DiscoverSlices_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_MultiSliceConfig_GetIncarnationId_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_MultiSliceConfig_GetNumHosts_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_GlobalBarrier_Args>() == 40);
