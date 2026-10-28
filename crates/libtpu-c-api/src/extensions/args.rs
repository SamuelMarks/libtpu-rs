//! `#[repr(C)]` argument structures, enums, and compile-time layout assertions for all 17 TPU PJRT extensions.

use crate::pjrt_types::{PJRT_Error, PJRT_Extension_Base};

/// Argument structure for `PLUGIN_Profiler_Error_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle to destroy.
    pub error: *mut std::ffi::c_void,
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
    pub error: *const std::ffi::c_void,
    /// Output error message pointer.
    pub message: *const std::os::raw::c_char,
    /// Output error message byte length.
    pub message_size: usize,
}

/// Argument structure for `PLUGIN_Profiler_Error_GetCode` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_GetCode_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle.
    pub error: *const std::ffi::c_void,
    /// Output status code.
    pub code: usize,
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
    pub profiler: *mut std::ffi::c_void,
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
    pub profiler: *mut std::ffi::c_void,
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
    pub profiler: *mut std::ffi::c_void,
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
    pub profiler: *mut std::ffi::c_void,
}

/// Argument structure for `PLUGIN_Profiler_CollectData` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_CollectData_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle word.
    pub profiler: usize,
    /// Caller buffer to receive serialized `tensorflow.profiler.XSpace` bytes (or `NULL` to query size).
    pub buffer: *mut u8,
    /// Output byte count (or input/output buffer capacity).
    pub size_in_bytes: *mut usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle to destroy.
    pub config: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_MultiSlice_Config_NumSlices` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_NumSlices_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle.
    pub config: *const std::ffi::c_void,
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
    /// Multi-slice configuration handle.
    pub config: *const std::ffi::c_void,
    /// Output slice device count map.
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
    /// Multi-slice configuration handle.
    pub config: *const std::ffi::c_void,
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
    /// Collectives instance handle.
    pub collectives: *mut std::ffi::c_void,
    /// Clique device IDs pointer.
    pub device_ids: *const i32,
    /// Number of devices in clique.
    pub num_devices: usize,
    /// Output communicator handle.
    pub communicator: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle to destroy.
    pub communicator: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_AllGather` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_AllGather_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle.
    pub communicator: *mut std::ffi::c_void,
    /// Send buffer pointer.
    pub send_buffer: *mut std::ffi::c_void,
    /// Receive buffer pointer.
    pub recv_buffer: *mut std::ffi::c_void,
    /// Element count.
    pub count: usize,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_CollectivePermute` and `AllToAll` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_CollectivePermute_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle.
    pub communicator: *mut std::ffi::c_void,
    /// Send buffer pointer.
    pub send_buffer: *mut std::ffi::c_void,
    /// Receive buffer pointer.
    pub recv_buffer: *mut std::ffi::c_void,
    /// Target ranks pointer.
    pub target_ranks: *const i32,
    /// Number of target ranks or elements.
    pub count: usize,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_ToString` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_ToString_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle.
    pub communicator: *const std::ffi::c_void,
    /// Output string pointer or length.
    pub str_out: usize,
    /// Output string length.
    pub str_size: usize,
}

/// Argument structure for `PJRT_Client_RuntimeAbiVersion` and `PJRT_Executable_GetAbiVersion` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client or executable handle.
    pub handle: *const std::ffi::c_void,
    /// Output ABI version handle or version number.
    pub abi_version_out: usize,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_Destroy` and `PJRT_ExecutableAbiVersion_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle to destroy.
    pub abi_version: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_IsCompatibleWith*` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_IsCompatible_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Runtime ABI version handle.
    pub abi_version: *const std::ffi::c_void,
    /// Other runtime or executable ABI version handle.
    pub other_abi_version: usize,
    /// Output compatibility boolean flag.
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
    /// ABI version handle.
    pub abi_version: *const std::ffi::c_void,
    /// Output 64-bit platform ID.
    pub platform_id_out: usize,
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
    /// Output ABI version handle.
    pub abi_version_out: usize,
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
    pub handle: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Megascale_*` 32-byte query/create operations (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Query_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input Megascale handle or slice parameter.
    pub handle: usize,
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
    /// Input Megascale handle.
    pub handle: usize,
    /// Output pointer or count.
    pub out_first: usize,
    /// Output byte length.
    pub out_second: usize,
}

/// Argument structure for `PJRT_TpuExecutable_*` 48-byte serialized proto queries (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_ProtoQuery_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const std::ffi::c_void,
    /// Output serialized bytes pointer or length.
    pub serialized_bytes: usize,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output holder handle.
    pub holder: *mut std::ffi::c_void,
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
    /// Output boolean result.
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
    /// Primary coordinate or output pointer word.
    pub out_first: usize,
    /// Secondary coordinate or count word.
    pub out_second: usize,
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
    pub topology: *const std::ffi::c_void,
    /// Input ID or coordinate word.
    pub param0: usize,
    /// Output process/chip ID word.
    pub out_first: usize,
    /// Output index on process/chip word.
    pub out_second: usize,
}

/// Argument structure for `PJRT_Callback_InvokeCallback` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Invoke_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
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
    pub client: *mut std::ffi::c_void,
    /// Output phase compiler handle.
    pub compiler_out: usize,
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

/// Argument structure for `PJRT_CrossHostTransfers_*` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CrossHostTransfers_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Buffers or shape specs array pointer.
    pub buffers: usize,
    /// Number of buffers.
    pub num_buffers: usize,
    /// Remote device or tag descriptor.
    pub remote_device: usize,
    /// Output completion event or count.
    pub out_event: usize,
}

/// Argument structure for `PJRT_ExecutableMetadata_Get` and `PJRT_MemoryDescriptions_Get` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Metadata_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable or device description handle.
    pub handle: *const std::ffi::c_void,
    /// Output metadata pointer or count.
    pub metadata_out: usize,
}

/// Argument structure for `PJRT_ExecutableMetadata_Destroy` and `PJRT_MemoryDescriptions_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Metadata_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Metadata handle to destroy.
    pub metadata: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_RawBuffer_*` 32-byte queries (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_Query_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_Buffer*` or `PJRT_RawBuffer*` handle.
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
    /// Raw buffer handle to destroy.
    pub raw_buffer: *mut std::ffi::c_void,
}

const _: () = assert!(size_of::<PLUGIN_Profiler_Error_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_Error_Message_Args>() == 40);
const _: () = assert!(size_of::<PLUGIN_Profiler_Error_GetCode_Args>() == 32);
const _: () = assert!(size_of::<PLUGIN_Profiler_Create_Args>() == 40);
const _: () = assert!(size_of::<PLUGIN_Profiler_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_Start_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_Stop_Args>() == 24);
const _: () = assert!(size_of::<PLUGIN_Profiler_CollectData_Args>() == 40);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_NumSlices_Args>() == 32);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_NumDevicesPerSlice_Args>() == 40);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_SliceId_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Collectives_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Collectives_CreateCommunicator_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Communicator_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Communicator_AllGather_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Communicator_CollectivePermute_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Communicator_ToString_Args>() == 40);
const _: () = assert!(size_of::<PJRT_AbiVersion_Get_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AbiVersion_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_AbiVersion_IsCompatible_Args>() == 40);
const _: () = assert!(size_of::<PJRT_AbiVersion_PlatformId_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AbiVersion_FromProto_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Megascale_Handle_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_Query_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_Data_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuExecutable_ProtoQuery_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuExecutable_ErrorCheck_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_ScalarQuery_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_BoundsQuery_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuTopology_MappingQuery_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Callback_Invoke_Args>() == 40);
const _: () = assert!(size_of::<PJRT_PhaseCompile_GetCompiler_Args>() == 32);
const _: () = assert!(size_of::<PJRT_PhaseCompile_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_CrossHostTransfers_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Metadata_Get_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Metadata_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_RawBuffer_Query_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_Destroy_Args>() == 24);

/// Argument structure for `PJRT_Layouts_MemoryLayout_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_MemoryLayout_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory layout handle to destroy.
    pub layout: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Layouts_MemoryLayout_Serialize` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_MemoryLayout_Serialize_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory layout handle to serialize.
    pub layout: *const std::ffi::c_void,
    /// Output pointer to serialized layout bytes.
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output byte length of serialized layout.
    pub serialized_bytes_size: usize,
    /// Output opaque serialized layout backing handle.
    pub serialized_layout: *mut std::ffi::c_void,
    /// Output deleter callback for `serialized_layout`.
    pub serialized_layout_deleter: *const std::ffi::c_void,
}

/// Argument structure for `PJRT_Layouts_PJRT_Client_GetDefaultLayout` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Element type (`PJRT_Buffer_Type`).
    pub elem_type: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
    /// Dimension array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Output default memory layout handle.
    pub layout: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Layouts_PJRT_Buffer_MemoryLayout` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut std::ffi::c_void,
    /// Output memory layout handle.
    pub layout: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Shardings_PJRT_Executable_ParameterShardings` and `OutputShardings` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Shardings_PJRT_Executable_Shardings_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *mut std::ffi::c_void,
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
    /// Phase compiler handle.
    pub phase_compiler: *const std::ffi::c_void,
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

/// Collective reduction operation kind (`PJRT_Collectives_ReductionKind` in `pjrt_c_api_collectives_extension.h`).
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

/// Collective operation group mode (`PJRT_Collectives_CollectiveOpKind` in `pjrt_c_api_collectives_extension.h`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PJRT_Collectives_CollectiveOpKind {
    /// Cross-module (SPMD partition) collective (`0`).
    #[default]
    CrossModule = 0,
    /// Cross-replica (data-parallel) collective (`1`).
    CrossReplica = 1,
}

/// Runtime callback category (`PJRT_Callback_Type` in `pjrt_c_api_callback_extension.h`).
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

/// TPU slice builder failure classification (`PJRT_Callback_Tpu_SliceFailureType` in `pjrt_c_api_callback_extension.h`).
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

/// Argument structure for `PJRT_HostMemoryAllocator_Allocate` (`56` bytes, `host_memory_allocator_extension.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_HostMemoryAllocator_Allocate_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Requested allocation size in bytes.
    pub size: usize,
    /// Preferred NUMA node index.
    pub numa_node: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
    /// Output allocated host memory pointer.
    pub ptr: *mut std::ffi::c_void,
    /// Output deleter callback function pointer.
    pub deleter: *const std::ffi::c_void,
    /// Output deleter user argument.
    pub deleter_arg: *mut std::ffi::c_void,
}

/// Serialized executable metadata structure (`PJRT_ExecutableMetadata` in `executable_metadata_extension.h`, `16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecutableMetadata {
    /// Pointer to serialized `xla.PjRtExecutableMetadata` protobuf bytes.
    pub serialized_metadata: *const std::os::raw::c_char,
    /// Byte length of `serialized_metadata`.
    pub serialized_metadata_size: usize,
}

/// Argument structure for `PJRT_DeviceDescription_MemoryDescriptions` (`48` bytes, `pjrt_c_api_memory_descriptions_extension.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_MemoryDescriptions_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle.
    pub device_description: *const std::ffi::c_void,
    /// Output array of `PJRT_MemoryDescription*` handles.
    pub memory_descriptions: *const *const std::ffi::c_void,
    /// Output number of memory descriptions (`4`: HBM, VMEM, SMEM, `pinned_host`).
    pub num_memory_descriptions: usize,
    /// Output index of default memory space (`0` = HBM, `-1` if none).
    pub default_memory_index: isize,
}

/// Argument structure for `PJRT_MemoryDescription_Kind` (`48` bytes, `pjrt_c_api_memory_descriptions_extension.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MemoryDescription_Kind_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory description handle.
    pub memory_description: *const std::ffi::c_void,
    /// Output memory space kind string pointer.
    pub kind: *const std::os::raw::c_char,
    /// Output memory space kind string byte length.
    pub kind_size: usize,
    /// Output numeric kind ID (`0` = device/HBM, `1` = vmem, `2` = smem, `3` = `pinned_host`).
    pub kind_id: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
}

const _: () = assert!(size_of::<PJRT_Callback_Tpu_SliceBuilderArgs>() == 16);
const _: () = assert!(size_of::<PJRT_Callback_PrefatalArgs>() == 32);
const _: () = assert!(size_of::<PJRT_HostMemoryAllocator_Allocate_Args>() == 64);
const _: () = assert!(size_of::<PJRT_ExecutableMetadata>() == 16);
const _: () = assert!(size_of::<PJRT_DeviceDescription_MemoryDescriptions_Args>() == 48);
const _: () = assert!(size_of::<PJRT_MemoryDescription_Kind_Args>() == 48);

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
    /// Transform callback or output registration ID.
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
    /// Client handle.
    pub client: *mut std::ffi::c_void,
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

/// Argument structure for `PJRT_XlaTransform_List` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_List_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output array of transform pass name pointers.
    pub names_out: *const *const std::os::raw::c_char,
    /// Output number of registered transform passes.
    pub num_names_out: usize,
}

/// Argument structure for `PJRT_HostMemoryAllocator_Get` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_HostMemoryAllocator_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Output host memory allocator handle or capacity.
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
    /// Multi-slice config handle.
    pub config: *const std::ffi::c_void,
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
    /// Communicator handle.
    pub communicator: *mut std::ffi::c_void,
    /// Send buffer pointer.
    pub send_buffer: *mut std::ffi::c_void,
    /// Receive buffer pointer.
    pub recv_buffer: *mut std::ffi::c_void,
    /// Reduction operation code (`CollectiveReductionOp`).
    pub reduction_op: i32,
    /// Alignment padding.
    pub pad: i32,
    /// Element count.
    pub count: usize,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_AbiVersion_RuntimeToProto` and `ExecutableToProto` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_ToProto_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle.
    pub abi_version: *const std::ffi::c_void,
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
    /// Executable handle.
    pub executable: *mut std::ffi::c_void,
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

/// Argument structure for `PJRT_Callback_RegisterCallback` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Register_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Callback type identifier.
    pub callback_type: i64,
    /// Callback function pointer.
    pub callback_fn: *const std::ffi::c_void,
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
    /// Phase compiler handle.
    pub phase_compiler: *mut std::ffi::c_void,
    /// Input program bytes pointer.
    pub input_bytes: *const std::os::raw::c_char,
    /// Input program byte length.
    pub input_size: usize,
    /// Phase name pointer.
    pub phase_name: *const std::os::raw::c_char,
    /// Output artifact count or size.
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
    /// Raw buffer handle.
    pub buffer: *mut std::ffi::c_void,
    /// Host pointer.
    pub host_ptr: *mut std::ffi::c_void,
    /// Byte offset within buffer.
    pub offset: i64,
    /// Number of bytes to transfer.
    pub transfer_size: i64,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

const _: () = assert!(size_of::<PJRT_XlaTransform_Register_Args>() == 40);
const _: () = assert!(size_of::<PJRT_XlaTransform_WaitForAll_Args>() == 24);
const _: () = assert!(size_of::<PJRT_XlaTransform_Unregister_Args>() == 32);
const _: () = assert!(size_of::<PJRT_XlaTransform_List_Args>() == 32);
const _: () = assert!(size_of::<PJRT_HostMemoryAllocator_Get_Args>() == 32);
const _: () = assert!(size_of::<PJRT_MultiSlice_Config_Serialize_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Communicator_AllReduce_Args>() == 64);
const _: () = assert!(size_of::<PJRT_AbiVersion_ToProto_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuExecutable_GetCompiledMemoryStats_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Callback_Register_Args>() == 48);
const _: () = assert!(size_of::<PJRT_PhaseCompile_RunPhase_Args>() == 56);
const _: () = assert!(size_of::<PJRT_RawBuffer_CopyRaw_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Layouts_MemoryLayout_Serialize_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Shardings_PJRT_Executable_Shardings_Args>() == 48);
const _: () = assert!(size_of::<PJRT_PhaseCompile_GetPhaseNames_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Triton_Compile_Args>() == 88);
