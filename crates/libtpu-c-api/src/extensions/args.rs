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
    /// Topology description handle.
    pub topology: *const std::ffi::c_void,
    /// Subslice dimension bounds array pointer (`int32_t*`).
    pub subslice_dims: *const i32,
    /// Number of subslice dimensions.
    pub num_subslice_dims: usize,
    /// Device ID array pointer (`int32_t*`).
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
const _: () = assert!(size_of::<PJRT_TpuTopology_SliceConfig>() == 56);
const _: () = assert!(size_of::<PJRT_TpuTopology_ComputeSubsliceForDevices_Args>() == 64);

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
    ///Rank of the buffer (`dims` length).
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

const _: () = assert!(size_of::<XLA_FFI_Api_Version>() == 24);
const _: () = assert!(size_of::<XLA_FFI_Buffer>() == 40);
const _: () = assert!(size_of::<XLA_FFI_CallFrame>() == 64);
const _: () = assert!(size_of::<XLA_FFI_Api>() == 56);

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcessCount` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcessCount_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
    /// Output boolean flag (`0` or `1`).
    pub is_valid_subslice: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcessBounds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcessBounds_Args {
    /// Size of this struct (`40` or `48` with Z/buffer).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// X bound or pointer to `int32_t[3]` coordinate array.
    pub out_first: usize,
    /// Y bound or coordinate array capacity.
    pub out_second: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipsPerProcessBounds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipsPerProcessBounds_Args {
    /// Size of this struct (`40` or `48` with Z/buffer).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// X bound or pointer to `int32_t[3]` coordinate array.
    pub out_first: usize,
    /// Y bound or coordinate array capacity.
    pub out_second: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ChipBounds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ChipBounds_Args {
    /// Size of this struct (`40` or `48` with Z/buffer).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// X bound or pointer to `int32_t[3]` coordinate array.
    pub out_first: usize,
    /// Y bound or coordinate array capacity.
    pub out_second: usize,
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
    pub topology: *const std::ffi::c_void,
    /// Input process ID.
    pub process_id: usize,
    /// Output X coordinate or pointer to `int32_t[3]` coordinate array.
    pub out_first: usize,
    /// Output Y coordinate or coordinate array length.
    pub out_second: usize,
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
    pub topology: *const std::ffi::c_void,
    /// Input X coordinate or pointer to `const int32_t[3]` coordinates.
    pub coord_or_ptr: usize,
    /// Output chip ID or Y coordinate.
    pub out_first: usize,
    /// Output index on process or chip ID.
    pub out_second: usize,
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
    pub topology: *const std::ffi::c_void,
    /// Input chip coordinate or pointer.
    pub param0: usize,
    /// Input core index or output logical device ID.
    pub out_first: usize,
    /// Output logical device ID or core index.
    pub out_second: usize,
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
    pub topology: *const std::ffi::c_void,
    /// Input logical device ID.
    pub device_id: usize,
    /// Output chip ID / coordinate pointer (`int32_t[3]*`).
    pub out_first: usize,
    /// Output core index on chip.
    pub out_second: usize,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
    /// Input logical device ID.
    pub device_id: usize,
    /// Output process ID.
    pub out_proc_id: usize,
    /// Output device index on process.
    pub out_idx_on_proc: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_ProcessIds` (`32` bytes, or `40` bytes with out-array).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ProcessIds_Args {
    /// Size of this struct (`32` or `40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// Output process count or pointer to `int32_t*` IDs buffer.
    pub out_value: usize,
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
    pub topology: *const std::ffi::c_void,
    /// Target process ID or output buffer pointer.
    pub process_id: usize,
    /// Output chip count on process.
    pub num_chips: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_LogiDeviceIdsOnProcess` (`32` bytes, or `40` bytes with out-array).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args {
    /// Size of this struct (`32` or `40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// Output logical device count or pointer to `int32_t*` IDs buffer.
    pub out_value: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_GetSliceConfig` (`32` bytes, or `40` bytes with `SliceConfig*`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_GetSliceConfig_Args {
    /// Size of this struct (`32` or `40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle or slice name pointer.
    pub topology: *const std::ffi::c_void,
    /// Output slice config status or `PJRT_TpuTopology_SliceConfig*` pointer.
    pub out_value: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuTopology_GetSliceConfigs` (`32` bytes, or `40` bytes with `SliceConfig*`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_GetSliceConfigs_Args {
    /// Size of this struct (`32` or `40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// Output slice config count or `PJRT_TpuTopology_SliceConfig*` array pointer.
    pub out_value: usize,
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
    pub topology: *const std::ffi::c_void,
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
    pub topology: *const std::ffi::c_void,
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
    pub client: usize,
    /// Output `PJRT_Megascale_ClientContext*` handle.
    pub context_out: usize,
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
    pub context_out: usize,
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
    pub context: *mut std::ffi::c_void,
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
    pub context: usize,
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
    pub context: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_CreateAotConfig` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_CreateAoTMegascaleConfig_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input topology or slice count.
    pub topology: usize,
    /// Output `PJRT_MultiSlice_Config*` handle.
    pub config_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_CreateMultiSliceConfig` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_CreateMultiSliceConfig_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input topology or slice parameters.
    pub topology: usize,
    /// Output `PJRT_MultiSlice_Config*` handle.
    pub config_out: usize,
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
    pub config: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_ConfigNumSlices` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_ConfigNumSlices_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_MultiSlice_Config*` handle.
    pub config: usize,
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
    /// `PJRT_MultiSlice_Config*` handle.
    pub config: usize,
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
    /// `PJRT_MultiSlice_Config*` handle.
    pub config: usize,
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
    /// `PJRT_MultiSlice_Config*` handle.
    pub config: usize,
    /// Output serialized key-value byte length or pointer.
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
    /// Serialized bytes pointer (`const char*`).
    pub serialized_bytes: usize,
    /// Output `PJRT_MultiSlice_Config*` handle.
    pub config_out: usize,
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
    pub context: usize,
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
    /// Barrier ID or context handle.
    pub barrier_id: usize,
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
    /// Barrier handle to await.
    pub handle: *mut std::ffi::c_void,
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
    pub context: usize,
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
    pub context: usize,
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
    pub context: usize,
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
    pub context: usize,
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
    pub handle: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_Megascale_RegisterErrorHandler` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_RegisterErrorHandler_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional C error callback function pointer (`void (*)(const void*, void*)` as `usize`).
    pub callback_fn: usize,
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
    pub context: usize,
    /// Output pointer to serialized `MegascaleRuntimeMetricsProto`.
    pub serialized_metrics_ptr: usize,
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
    pub handle: *mut std::ffi::c_void,
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
    pub context: usize,
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
    /// Requested transport type.
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
    pub handle: *mut std::ffi::c_void,
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

/// Dedicated `#[repr(C)]` argument structure for `PJRT_RawBuffer_CreateRawAliasOfBuffer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_Buffer*` handle.
    pub buffer: *mut std::ffi::c_void,
    /// Output `PJRT_RawBuffer*` handle.
    pub raw_buffer_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_RawBuffer_GetHostPointer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_GetHostPointer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_Buffer*` or `PJRT_RawBuffer*` handle.
    pub buffer: *mut std::ffi::c_void,
    /// Output device/host address (`uintptr_t`).
    pub host_pointer_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_RawBuffer_GetOnDeviceSizeInBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_Buffer*` or `PJRT_RawBuffer*` handle.
    pub buffer: *mut std::ffi::c_void,
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
    /// Input `PJRT_Buffer*` or `PJRT_RawBuffer*` handle.
    pub buffer: *mut std::ffi::c_void,
    /// Output `PJRT_Memory*` or address handle.
    pub memory_space_out: usize,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetTargetArguments` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_TargetArguments_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *const std::ffi::c_void,
    /// Output serialized `TargetArgumentsProto` bytes pointer.
    pub serialized_bytes: usize,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output artifact holder handle.
    pub holder: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_CoreProgramAbiVersion` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_CoreProgramAbiVersion_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *const std::ffi::c_void,
    /// Output serialized `TpuRuntimeAbiProto` bytes pointer.
    pub serialized_bytes: usize,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output artifact holder handle.
    pub holder: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetHloModuleWithConfig` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_HloModule_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *const std::ffi::c_void,
    /// Output serialized `HloModuleProtoWithConfig` bytes pointer.
    pub serialized_bytes: usize,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output artifact holder handle.
    pub holder: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_Serialize_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *const std::ffi::c_void,
    /// Output serialized bytes pointer.
    pub serialized_bytes: usize,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output artifact holder handle.
    pub holder: *mut std::ffi::c_void,
}

/// Dedicated `#[repr(C)]` argument structure for `PJRT_TpuExecutable_GetFingerprint` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_GetFingerprint_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle (`PJRT_Executable*`).
    pub executable: *const std::ffi::c_void,
    /// Output fingerprint UTF-8 bytes pointer.
    pub fingerprint_bytes: usize,
    /// Output fingerprint byte length.
    pub fingerprint_size: usize,
    /// Output artifact holder handle.
    pub holder: *mut std::ffi::c_void,
}

const _: () = assert!(size_of::<PJRT_TpuTopology_ProcessCount_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_ChipsPerProcess_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_CoreCountPerChip_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_ChipCount_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_CoreCount_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_LogiDeviceCountPerProcess_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_LogiDeviceCount_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_LogiDeviceCountPerChip_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_CoreCountPerProcess_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_HasLimitedIciConnectivity_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_IsSubsliceTopology_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_IsValidSubslice_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_ProcessBounds_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuTopology_ChipsPerProcessBounds_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuTopology_ChipBounds_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuTopology_ProcessCoordFromId_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_ChipIdFromCoord_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuTopology_ProcessIds_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_ChipsOnProcess_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_GetSliceConfig_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_GetSliceConfigs_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_GetDefaultPlatformConfig_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuTopology_GetRoutingStrategy_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_CreateClientContextFromPjRtClient_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_CreateDefaultClientContext_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_DeleteClientContext_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_InitializeClientContext_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_UnblockPendingWork_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_CreateAoTMegascaleConfig_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_CreateMultiSliceConfig_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_DeleteMultiSliceConfig_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_ConfigNumSlices_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_ConfigNumDevicesPerSlice_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_ConfigSliceId_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_ConfigSerializedSliceKeyValue_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Megascale_DeserializeMultiSliceConfig_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_Addresses_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_Barrier_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_WaitForBarrier_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_GetDcnTopology_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_GetSliceId_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_GetNumSlices_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_InitializeSlice_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_FinalizeSlice_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_RegisterErrorHandler_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_UnregisterErrorHandler_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_CollectMetrics_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Megascale_ResetMetrics_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_GetTransportType_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_SetTransportType_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Megascale_AbortCollectives_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Megascale_LogError_Args>() == 24);
const _: () = assert!(size_of::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_GetHostPointer_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args>() == 32);
const _: () = assert!(size_of::<PJRT_RawBuffer_GetMemorySpace_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TpuExecutable_TargetArguments_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuExecutable_CoreProgramAbiVersion_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuExecutable_HloModule_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuExecutable_Serialize_Args>() == 48);
const _: () = assert!(size_of::<PJRT_TpuExecutable_GetFingerprint_Args>() == 48);
