//! `#[repr(C)]` argument structures for `PJRT_Client_*` and `PJRT_Compile` entrypoints.

use crate::pjrt_types::{
    PJRT_AsyncHostToDeviceTransferManager, PJRT_Buffer, PJRT_Buffer_MemoryLayout, PJRT_Client,
    PJRT_Device, PJRT_Event, PJRT_Executable, PJRT_Extension_Base, PJRT_LoadedExecutable,
    PJRT_Memory, PJRT_ProcessInfo, PJRT_Program, PJRT_ShapeSpec, PJRT_TopologyDescription,
    PjrtArgsStruct,
};
use std::os::raw::c_char;

/// Argument structure for `PJRT_Client_Create` (`88` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_Create_Args {
    /// Size of this struct (`88`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Optional creation options array.
    pub create_options: *const std::ffi::c_void,
    /// Number of creation options.
    pub num_options: usize,
    /// Optional key-value get callback.
    pub kv_get_callback: *const std::ffi::c_void,
    /// Optional key-value get user arg.
    pub kv_get_user_arg: *mut std::ffi::c_void,
    /// Optional key-value put callback.
    pub kv_put_callback: *const std::ffi::c_void,
    /// Optional key-value put user arg.
    pub kv_put_user_arg: *mut std::ffi::c_void,
    /// Optional non-blocking key-value try-get callback.
    pub kv_try_get_callback: *const std::ffi::c_void,
    /// Optional non-blocking key-value try-get user arg.
    pub kv_try_get_user_arg: *mut std::ffi::c_void,
    /// Output created `PJRT_Client*`.
    pub client: *mut PJRT_Client,
}

/// Argument structure for `PJRT_Client_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_Destroy_Args {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle to destroy.
    pub client: *mut PJRT_Client,
}

/// Argument structure for `PJRT_Client_PlatformName` and `PJRT_Client_PlatformVersion` (`40` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_PlatformStr_Args {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Output string pointer.
    pub platform_str: *const c_char,
    /// Output string byte length.
    pub platform_str_size: usize,
}

/// Type alias for `PJRT_Client_PlatformName_Args` (`40` bytes).
pub type PJRT_Client_PlatformName_Args = PJRT_Client_PlatformStr_Args;

/// Type alias for `PJRT_Client_PlatformVersion_Args` (`40` bytes).
pub type PJRT_Client_PlatformVersion_Args = PJRT_Client_PlatformStr_Args;

/// Argument structure for `PJRT_Client_ProcessIndex` (`32` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_ProcessIndex_Args {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Output process index.
    pub process_index: i32,
}

/// Argument structure for `PJRT_Client_Devices` and `PJRT_Client_AddressableDevices` (`40` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_Devices_Args {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Output pointer to array of `PJRT_Device*`.
    pub devices: *const *mut PJRT_Device,
    /// Output number of devices.
    pub num_devices: usize,
}

/// Type alias for `PJRT_Client_AddressableDevices_Args` (`40` bytes).
pub type PJRT_Client_AddressableDevices_Args = PJRT_Client_Devices_Args;

/// Argument structure for `PJRT_Client_LookupDevice` (`40` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_LookupDevice_Args {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Requested device ID.
    pub id: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Output `PJRT_Device*`.
    pub device: *mut PJRT_Device,
}

/// Type alias for `PJRT_Client_LookupAddressableDevice_Args` (`40` bytes).
pub type PJRT_Client_LookupAddressableDevice_Args = PJRT_Client_LookupDevice_Args;

/// Argument structure for `PJRT_Client_AddressableMemories` (`40` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_AddressableMemories_Args {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Output pointer to array of `PJRT_Memory*`.
    pub addressable_memories: *const *mut PJRT_Memory,
    /// Output number of addressable memories.
    pub num_addressable_memories: usize,
}

/// Argument structure for `PJRT_Client_Compile` (`56` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_Compile_Args {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Input program descriptor.
    pub program: *const PJRT_Program,
    /// Pointer to serialized `CompileOptionsProto`.
    pub compile_options: *const c_char,
    /// Length of `compile_options` in bytes.
    pub compile_options_size: usize,
    /// Output `PJRT_LoadedExecutable*`.
    pub executable: *mut PJRT_LoadedExecutable,
}

/// Argument structure for `PJRT_Client_DefaultDeviceAssignment` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_DefaultDeviceAssignment_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Number of replicas.
    pub num_replicas: i32,
    /// Number of partitions.
    pub num_partitions: i32,
    /// Capacity of `default_assignment` array.
    pub default_assignment_size: usize,
    /// Caller buffer for `int` device IDs.
    pub default_assignment: *mut i32,
}

/// Argument structure for `PJRT_Client_BufferFromHostBuffer` (`120` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_BufferFromHostBuffer_Args {
    /// Size of this struct (`120`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Host data pointer.
    pub data: *const std::ffi::c_void,
    /// Element type (`i32`).
    pub element_type: i32,
    /// Explicit 4-byte padding.
    pub pad0: i32,
    /// Dimension extents pointer (`const int64_t*`).
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Optional byte strides pointer (`const int64_t*`).
    pub byte_strides: *const i64,
    /// Number of byte strides.
    pub num_byte_strides: usize,
    /// Host buffer semantics discriminant.
    pub host_buffer_semantics: i32,
    /// Explicit 4-byte padding.
    pub pad1: i32,
    /// Target `PJRT_Device*`.
    pub device: *mut PJRT_Device,
    /// Optional target `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
    /// Optional `PJRT_Buffer_MemoryLayout*`.
    pub device_layout: *const PJRT_Buffer_MemoryLayout,
    /// Output host buffer completion `PJRT_Event*`.
    pub done_with_host_buffer: *mut PJRT_Event,
    /// Output `buffer` (`PJRT_Buffer*`).
    pub out_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Compile` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Compile_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Input `PJRT_Program*`.
    pub program: *const PJRT_Program,
    /// Serialized compile options pointer.
    pub compile_options: *const c_char,
    /// Serialized compile options byte count.
    pub compile_options_size: usize,
    /// Optional client handle.
    pub client: *mut PJRT_Client,
    /// Output `PJRT_Executable*`.
    pub executable: *mut PJRT_Executable,
}

/// Argument structure for `PJRT_Client_CreateViewOfDeviceBuffer` (`88` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateViewOfDeviceBuffer_Args {
    /// Size of this struct (`88`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Device buffer pointer.
    pub device_buffer_ptr: *const std::ffi::c_void,
    /// Dimensions array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Element type (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Memory layout pointer.
    pub layout: *const PJRT_Buffer_MemoryLayout,
    /// Target `PJRT_Device*`.
    pub device: *mut PJRT_Device,
    /// Optional deletion callback.
    pub on_delete_callback: *const std::ffi::c_void,
    /// Output `PJRT_Buffer*`.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Client_TopologyDescription` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_TopologyDescription_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *const PJRT_Client,
    /// Output `PJRT_TopologyDescription*`.
    pub topology: *mut PJRT_TopologyDescription,
}

/// Argument structure for `PJRT_Client_CreateBuffersForAsyncHostToDevice` (`72` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
    /// Size of this struct (`72`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Shape specs array pointer.
    pub shape_specs: *const PJRT_ShapeSpec,
    /// Number of shape specs.
    pub num_shape_specs: usize,
    /// Optional device layouts array pointer.
    pub device_layouts: *const *mut PJRT_Buffer_MemoryLayout,
    /// Number of device layouts.
    pub num_device_layouts: usize,
    /// Target `PJRT_Memory*` (or `PJRT_Device*`).
    pub memory: *mut PJRT_Memory,
    /// Output `PJRT_AsyncHostToDeviceTransferManager*`.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
}

/// Argument structure for `PJRT_Client_DmaMap` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_DmaMap_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Host virtual address to map.
    pub data: *const std::ffi::c_void,
    /// Byte size to map.
    pub size: usize,
}

/// Argument structure for `PJRT_Client_DmaUnmap` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_DmaUnmap_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Previously mapped host virtual address.
    pub data: *const std::ffi::c_void,
}

/// Argument structure for `PJRT_Client_CreateUninitializedBuffer` (`80` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateUninitializedBuffer_Args {
    /// Size of this struct (`80`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Dimensions array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Element type (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Layout pointer.
    pub layout: *const PJRT_Buffer_MemoryLayout,
    /// Target `PJRT_Device*`.
    pub device: *mut PJRT_Device,
    /// Optional target `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
    /// Output `PJRT_Buffer*`.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Client_UpdateGlobalProcessInfo` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_UpdateGlobalProcessInfo_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Process info array pointer.
    pub processes: *const PJRT_ProcessInfo,
    /// Number of process info entries.
    pub num_processes: usize,
}

/// Argument structure for `PJRT_Client_CreateAliasBuffer` (`80` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateAliasBuffer_Args {
    /// Size of this struct (`80`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Target `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
    /// Dimensions array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Element type (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Optional memory layout pointer.
    pub layout: *const PJRT_Buffer_MemoryLayout,
    /// Output alias `PJRT_Buffer*`.
    pub alias_buffer: *mut PJRT_Buffer,
    /// Output fulfillment callback handle.
    pub fulfill_alias_buffer_cb: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Client_FulfillAliasBuffer` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_FulfillAliasBuffer_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Concrete `PJRT_Buffer*`.
    pub buffer: *mut PJRT_Buffer,
    /// Status code (`i32`).
    pub status_code: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Optional error message pointer.
    pub error_message: *const c_char,
    /// Error message byte length.
    pub error_message_size: usize,
    /// Fulfillment callback handle.
    pub fulfill_alias_buffer_cb: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Client_CreateErrorBuffer` (`112` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateErrorBuffer_Args {
    /// Size of this struct (`112`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Dimensions array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Element type (`i32`).
    pub element_type: i32,
    /// Error code (`i32`).
    pub error_code: i32,
    /// Error message pointer.
    pub error_message: *const c_char,
    /// Error message byte length.
    pub error_message_size: usize,
    /// Target `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
    /// Optional layout pointer.
    pub layout: *const PJRT_Buffer_MemoryLayout,
    /// Target `PJRT_Device*`.
    pub device: *mut PJRT_Device,
    /// Reserved context pointer 0.
    pub reserved: usize,
    /// Reserved context pointer 1.
    pub reserved1: usize,
    /// Output poisoned `PJRT_Buffer*`.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Client_Load` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_Load_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Input `PJRT_Executable*`.
    pub executable: *mut PJRT_Executable,
    /// Optional compile options pointer.
    pub compile_options: *const c_char,
    /// Byte length of `compile_options`.
    pub compile_options_size: usize,
    /// Output `PJRT_LoadedExecutable*`.
    pub loaded_executable: *mut PJRT_LoadedExecutable,
}

impl PjrtArgsStruct for PJRT_Client_Create_Args {}
impl PjrtArgsStruct for PJRT_Client_Destroy_Args {}
impl PjrtArgsStruct for PJRT_Client_PlatformStr_Args {}
impl PjrtArgsStruct for PJRT_Client_ProcessIndex_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_Client_Devices_Args {}
impl PjrtArgsStruct for PJRT_Client_LookupDevice_Args {}
impl PjrtArgsStruct for PJRT_Client_AddressableMemories_Args {}
impl PjrtArgsStruct for PJRT_Client_Compile_Args {}
impl PjrtArgsStruct for PJRT_Client_DefaultDeviceAssignment_Args {}
impl PjrtArgsStruct for PJRT_Client_BufferFromHostBuffer_Args {}
impl PjrtArgsStruct for PJRT_Compile_Args {}
impl PjrtArgsStruct for PJRT_Client_CreateViewOfDeviceBuffer_Args {
    const MIN_STRUCT_SIZE: usize = 72;
}
impl PjrtArgsStruct for PJRT_Client_TopologyDescription_Args {}
impl PjrtArgsStruct for PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {}
impl PjrtArgsStruct for PJRT_Client_DmaMap_Args {}
impl PjrtArgsStruct for PJRT_Client_DmaUnmap_Args {}
impl PjrtArgsStruct for PJRT_Client_CreateUninitializedBuffer_Args {}
impl PjrtArgsStruct for PJRT_Client_UpdateGlobalProcessInfo_Args {}
impl PjrtArgsStruct for PJRT_Client_CreateAliasBuffer_Args {}
impl PjrtArgsStruct for PJRT_Client_FulfillAliasBuffer_Args {}
impl PjrtArgsStruct for PJRT_Client_CreateErrorBuffer_Args {}
impl PjrtArgsStruct for PJRT_Client_Load_Args {}

const _: () = assert!(std::mem::offset_of!(PJRT_Client_Create_Args, create_options) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Create_Args, num_options) == 0x18);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Create_Args, kv_get_callback) == 0x20);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Create_Args, kv_put_callback) == 0x30);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Create_Args, kv_try_get_callback) == 0x40);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Create_Args, client) == 0x50);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Compile_Args, client) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Compile_Args, program) == 0x18);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Compile_Args, compile_options) == 0x20);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Compile_Args, compile_options_size) == 0x28);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_Compile_Args, executable) == 0x30);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, client) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, data) == 0x18);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, element_type) == 0x20);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, dims) == 0x28);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, num_dims) == 0x30);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, byte_strides) == 0x38);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, device) == 0x50);
const _: () = assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, memory) == 0x58);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Client_BufferFromHostBuffer_Args, out_buffer) == 0x70);

const _: () = assert!(size_of::<PJRT_Client_Create_Args>() == 88);
const _: () = assert!(size_of::<PJRT_Client_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Client_PlatformStr_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Client_ProcessIndex_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Client_Devices_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Client_LookupDevice_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Client_AddressableMemories_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Client_Compile_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Client_DefaultDeviceAssignment_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Client_BufferFromHostBuffer_Args>() == 120);
const _: () = assert!(size_of::<PJRT_Compile_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Client_CreateViewOfDeviceBuffer_Args>() == 88);
const _: () = assert!(size_of::<PJRT_Client_TopologyDescription_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Client_CreateBuffersForAsyncHostToDevice_Args>() == 72);
const _: () = assert!(size_of::<PJRT_Client_DmaMap_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Client_DmaUnmap_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Client_CreateUninitializedBuffer_Args>() == 80);
const _: () = assert!(size_of::<PJRT_Client_UpdateGlobalProcessInfo_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Client_CreateAliasBuffer_Args>() == 80);
const _: () = assert!(size_of::<PJRT_Client_FulfillAliasBuffer_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Client_CreateErrorBuffer_Args>() == 112);
const _: () = assert!(size_of::<PJRT_Client_Load_Args>() == 56);
