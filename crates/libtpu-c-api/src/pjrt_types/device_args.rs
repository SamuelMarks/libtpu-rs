//! `#[repr(C)]` argument structures for device, memory, and topology entrypoints.
//!
//! Covers `PJRT_DeviceDescription_*`, `PJRT_Device_*`, `PJRT_Memory_*`, and `PJRT_TopologyDescription_*`.

use crate::pjrt_types::{
    PJRT_AsyncTrackingEvent, PJRT_Buffer_MemoryLayout, PJRT_Device, PJRT_DeviceDescription,
    PJRT_Extension_Base, PJRT_Memory, PJRT_NamedValue, PJRT_SerializedTopology,
    PJRT_TopologyDescription, PjrtArgsStruct,
};
use std::os::raw::c_char;

/// Argument structure for `PJRT_DeviceDescription_Id` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_Id_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle.
    pub device_description: *const PJRT_DeviceDescription,
    /// Output device ID (`i32`).
    pub id: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Argument structure for `PJRT_DeviceDescription_ProcessIndex` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_ProcessIndex_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle.
    pub device_description: *const PJRT_DeviceDescription,
    /// Output process index (`i32`).
    pub process_index: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Argument structure for `PJRT_DeviceDescription_Attributes` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_Attributes_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle.
    pub device_description: *const PJRT_DeviceDescription,
    /// Output attributes array pointer.
    pub attributes: *const PJRT_NamedValue,
    /// Output number of attributes.
    pub num_attributes: usize,
}

/// Argument structure for `PJRT_DeviceDescription_Kind` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_Kind_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle.
    pub device_description: *const PJRT_DeviceDescription,
    /// Output kind string pointer.
    pub device_kind: *const c_char,
    /// Output kind string length.
    pub device_kind_size: usize,
}

/// Argument structure for `PJRT_DeviceDescription_DebugString` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_DebugString_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle.
    pub device_description: *const PJRT_DeviceDescription,
    /// Output debug string pointer.
    pub debug_string: *const c_char,
    /// Output debug string length.
    pub debug_string_size: usize,
}

/// Argument structure for `PJRT_DeviceDescription_ToString` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_DeviceDescription_ToString_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device description handle.
    pub device_description: *const PJRT_DeviceDescription,
    /// Output string pointer.
    pub to_string: *const c_char,
    /// Output string length.
    pub to_string_size: usize,
}

/// Argument structure for `PJRT_Device_GetDescription` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_GetDescription_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output `PJRT_DeviceDescription*`.
    pub device_description: *mut PJRT_DeviceDescription,
}

/// Argument structure for `PJRT_Device_IsAddressable` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_IsAddressable_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output `is_addressable` flag.
    pub is_addressable: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Device_LocalHardwareId` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_LocalHardwareId_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output `local_hardware_id` (`i32`).
    pub local_hardware_id: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Argument structure for `PJRT_Device_AddressableMemories` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_AddressableMemories_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output `PJRT_Memory*` array pointer.
    pub memories: *const *mut PJRT_Memory,
    /// Output number of memories.
    pub num_memories: usize,
}

/// Argument structure for `PJRT_Device_DefaultMemory` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_DefaultMemory_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output default `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
}

/// Argument structure for `PJRT_Device_MemoryStats` (`192` bytes, `MIN_STRUCT_SIZE = 185`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_MemoryStats_Args {
    /// Size of this struct (`192` aligned, `185` minimum).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output `bytes_in_use`.
    pub bytes_in_use: i64,
    /// Output `peak_bytes_in_use`.
    pub peak_bytes_in_use: i64,
    /// Output `peak_bytes_in_use_is_set`.
    pub peak_bytes_in_use_is_set: bool,
    /// Explicit padding.
    pub pad0: [u8; 7],
    /// Output `num_allocs`.
    pub num_allocs: i64,
    /// Output `num_allocs_is_set`.
    pub num_allocs_is_set: bool,
    /// Explicit padding.
    pub pad1: [u8; 7],
    /// Output `largest_alloc_size`.
    pub largest_alloc_size: i64,
    /// Output `largest_alloc_size_is_set`.
    pub largest_alloc_size_is_set: bool,
    /// Explicit padding.
    pub pad2: [u8; 7],
    /// Output `bytes_limit`.
    pub bytes_limit: i64,
    /// Output `bytes_limit_is_set`.
    pub bytes_limit_is_set: bool,
    /// Explicit padding.
    pub pad3: [u8; 7],
    /// Output `bytes_reserved`.
    pub bytes_reserved: i64,
    /// Output `bytes_reserved_is_set`.
    pub bytes_reserved_is_set: bool,
    /// Explicit padding.
    pub pad4: [u8; 7],
    /// Output `peak_bytes_reserved`.
    pub peak_bytes_reserved: i64,
    /// Output `peak_bytes_reserved_is_set`.
    pub peak_bytes_reserved_is_set: bool,
    /// Explicit padding.
    pub pad5: [u8; 7],
    /// Output `bytes_reservable_limit`.
    pub bytes_reservable_limit: i64,
    /// Output `bytes_reservable_limit_is_set`.
    pub bytes_reservable_limit_is_set: bool,
    /// Explicit padding.
    pub pad6: [u8; 7],
    /// Output `largest_free_block_bytes`.
    pub largest_free_block_bytes: i64,
    /// Output `largest_free_block_bytes_is_set`.
    pub largest_free_block_bytes_is_set: bool,
    /// Explicit padding.
    pub pad7: [u8; 7],
    /// Output `pool_bytes`.
    pub pool_bytes: i64,
    /// Output `pool_bytes_is_set`.
    pub pool_bytes_is_set: bool,
    /// Explicit padding.
    pub pad8: [u8; 7],
    /// Output `peak_pool_bytes`.
    pub peak_pool_bytes: i64,
    /// Output `peak_pool_bytes_is_set`.
    pub peak_pool_bytes_is_set: bool,
    /// Explicit padding.
    pub pad9: [u8; 7],
}

/// Argument structure for `PJRT_Device_PoisonExecution` (`72` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_PoisonExecution_Args {
    /// Size of this struct (`72`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *mut PJRT_Device,
    /// Launch ID (`i32`).
    pub launch_id: i32,
    /// Error code (`i32`).
    pub error_code: i32,
    /// Error message pointer.
    pub error_message: *const c_char,
    /// Error message byte length.
    pub error_message_size: usize,
    /// Reserved word 0.
    pub reserved0: usize,
    /// Reserved word 1.
    pub reserved1: usize,
    /// Output `poisoned` flag.
    pub poisoned: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Device_CreateAsyncTrackingEvent` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_CreateAsyncTrackingEvent_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *mut PJRT_Device,
    /// Description string pointer.
    pub description: *const c_char,
    /// Description string length.
    pub description_size: usize,
    /// Output `PJRT_AsyncTrackingEvent*`.
    pub event: *mut PJRT_AsyncTrackingEvent,
}

/// Argument structure for `PJRT_Device_GetAttributes` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_GetAttributes_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output `PJRT_NamedValue` array pointer.
    pub attributes: *const PJRT_NamedValue,
    /// Output number of attributes.
    pub num_attributes: usize,
    /// Reserved word 0.
    pub reserved0: usize,
    /// Reserved word 1.
    pub reserved1: usize,
}

/// Argument structure for `PJRT_Device_ClearMemoryStats` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_ClearMemoryStats_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *mut PJRT_Device,
}

/// Argument structure for `PJRT_Memory_Id` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Memory_Id_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory handle.
    pub memory: *const PJRT_Memory,
    /// Output memory ID (`i32`).
    pub id: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Argument structure for `PJRT_Memory_Kind` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Memory_Kind_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory handle.
    pub memory: *const PJRT_Memory,
    /// Output kind string pointer.
    pub kind: *const c_char,
    /// Output kind string length.
    pub kind_size: usize,
}

/// Argument structure for `PJRT_Memory_DebugString` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Memory_DebugString_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory handle.
    pub memory: *const PJRT_Memory,
    /// Output debug string pointer.
    pub debug_string: *const c_char,
    /// Output debug string length.
    pub debug_string_size: usize,
}

/// Argument structure for `PJRT_Memory_ToString` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Memory_ToString_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory handle.
    pub memory: *const PJRT_Memory,
    /// Output string pointer.
    pub to_string: *const c_char,
    /// Output string length.
    pub to_string_size: usize,
}

/// Argument structure for `PJRT_Memory_AddressableByDevices` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Memory_AddressableByDevices_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory handle.
    pub memory: *const PJRT_Memory,
    /// Output `PJRT_Device*` array pointer.
    pub devices: *const *mut PJRT_Device,
    /// Output number of devices.
    pub num_devices: usize,
}

/// Argument structure for `PJRT_Memory_Kind_Id` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Memory_Kind_Id_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory handle.
    pub memory: *const PJRT_Memory,
    /// Output `kind_id` (`i32`).
    pub kind_id: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Argument structure for `PJRT_TopologyDescription_Create` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Create_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology name pointer.
    pub topology_name: *const c_char,
    /// Topology name byte length.
    pub topology_name_size: usize,
    /// Optional create options array.
    pub create_options: *const PJRT_NamedValue,
    /// Number of creation options in `create_options`.
    pub num_options: usize,
    /// Output `PJRT_TopologyDescription*`.
    pub topology: *mut PJRT_TopologyDescription,
}

/// Argument structure for `PJRT_TopologyDescription_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle to destroy.
    pub topology: *mut PJRT_TopologyDescription,
}

/// Argument structure for `PJRT_TopologyDescription_PlatformName` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_PlatformName_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output platform name pointer.
    pub platform_name: *const c_char,
    /// Output platform name length.
    pub platform_name_size: usize,
}

/// Argument structure for `PJRT_TopologyDescription_PlatformVersion` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_PlatformVersion_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output platform version pointer.
    pub platform_version: *const c_char,
    /// Output platform version length.
    pub platform_version_size: usize,
}

/// Argument structure for `PJRT_TopologyDescription_GetDeviceDescriptions` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_GetDeviceDescriptions_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output `PJRT_DeviceDescription*` array pointer.
    pub descriptions: *const *mut PJRT_DeviceDescription,
    /// Output number of descriptions.
    pub num_descriptions: usize,
}

/// Argument structure for `PJRT_TopologyDescription_Serialize` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Serialize_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output serialized bytes pointer.
    pub serialized_bytes: *const c_char,
    /// Output serialized byte count.
    pub serialized_bytes_size: usize,
    /// Output `PJRT_SerializedTopology*`.
    pub serialized_topology: *mut PJRT_SerializedTopology,
    /// Output deleter callback (`void (*)(PJRT_SerializedTopology*)`).
    pub serialized_topology_deleter: Option<extern "C" fn(*mut std::ffi::c_void)>,
}

/// Argument structure for `PJRT_TopologyDescription_Attributes` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Attributes_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output `PJRT_NamedValue` array pointer.
    pub attributes: *const PJRT_NamedValue,
    /// Output number of attributes.
    pub num_attributes: usize,
}

/// Argument structure for `PJRT_TopologyDescription_Deserialize` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Deserialize_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Serialized topology bytes pointer.
    pub serialized_topology: *const c_char,
    /// Serialized topology byte length.
    pub serialized_topology_size: usize,
    /// Output `PJRT_TopologyDescription*`.
    pub topology: *mut PJRT_TopologyDescription,
}

/// Argument structure for `PJRT_TopologyDescription_Fingerprint` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Fingerprint_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output 64-bit fingerprint.
    pub fingerprint: u64,
}

/// Argument structure for `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` (`88` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
    /// Size of this struct (`88`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Dimensions array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Element type (`i32`).
    pub element_type: i32,
    /// Memory space kind ID (`i32`).
    pub memory_space_kind: i32,
    /// Output canonical layout pointer.
    pub layout: *mut PJRT_Buffer_MemoryLayout,
    /// Reserved word 0.
    pub reserved: usize,
    /// Reserved word 1.
    pub reserved1: usize,
    /// Reserved word 2.
    pub reserved2: usize,
    /// Reserved word 3.
    pub reserved3: usize,
}

/// Argument structure for `PJRT_TopologyDescription_GetMemorySpaceKindIds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_GetMemorySpaceKindIds_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology handle.
    pub topology: *const PJRT_TopologyDescription,
    /// Output `kind_ids` (`const int32_t*`) array.
    pub kind_ids: *const i32,
    /// Output number of kind IDs.
    pub num_kind_ids: usize,
}

impl PjrtArgsStruct for PJRT_DeviceDescription_Id_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_DeviceDescription_ProcessIndex_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_DeviceDescription_Attributes_Args {}
impl PjrtArgsStruct for PJRT_DeviceDescription_Kind_Args {}
impl PjrtArgsStruct for PJRT_DeviceDescription_DebugString_Args {}
impl PjrtArgsStruct for PJRT_DeviceDescription_ToString_Args {}
impl PjrtArgsStruct for PJRT_Device_GetDescription_Args {}
impl PjrtArgsStruct for PJRT_Device_IsAddressable_Args {
    const MIN_STRUCT_SIZE: usize = 25;
}
impl PjrtArgsStruct for PJRT_Device_LocalHardwareId_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_Device_AddressableMemories_Args {}
impl PjrtArgsStruct for PJRT_Device_DefaultMemory_Args {}
impl PjrtArgsStruct for PJRT_Device_MemoryStats_Args {
    const MIN_STRUCT_SIZE: usize = 185;
}
impl PjrtArgsStruct for PJRT_Device_PoisonExecution_Args {}
impl PjrtArgsStruct for PJRT_Device_CreateAsyncTrackingEvent_Args {}
impl PjrtArgsStruct for PJRT_Device_GetAttributes_Args {}
impl PjrtArgsStruct for PJRT_Device_ClearMemoryStats_Args {}
impl PjrtArgsStruct for PJRT_Memory_Id_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_Memory_Kind_Args {}
impl PjrtArgsStruct for PJRT_Memory_DebugString_Args {}
impl PjrtArgsStruct for PJRT_Memory_ToString_Args {}
impl PjrtArgsStruct for PJRT_Memory_AddressableByDevices_Args {}
impl PjrtArgsStruct for PJRT_Memory_Kind_Id_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_TopologyDescription_Create_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_Destroy_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_PlatformName_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_PlatformVersion_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_GetDeviceDescriptions_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_Serialize_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_Attributes_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_Deserialize_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_Fingerprint_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {}
impl PjrtArgsStruct for PJRT_TopologyDescription_GetMemorySpaceKindIds_Args {}

const _: () = assert!(std::mem::offset_of!(PJRT_Device_PoisonExecution_Args, device) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_Device_PoisonExecution_Args, launch_id) == 0x18);
const _: () = assert!(std::mem::offset_of!(PJRT_Device_PoisonExecution_Args, error_code) == 0x1c);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Device_PoisonExecution_Args, error_message) == 0x20);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Device_PoisonExecution_Args, error_message_size) == 0x28);
const _: () = assert!(std::mem::offset_of!(PJRT_Device_PoisonExecution_Args, poisoned) == 0x40);

const _: () = assert!(size_of::<PJRT_DeviceDescription_Id_Args>() == 32);
const _: () = assert!(size_of::<PJRT_DeviceDescription_ProcessIndex_Args>() == 32);
const _: () = assert!(size_of::<PJRT_DeviceDescription_Attributes_Args>() == 40);
const _: () = assert!(size_of::<PJRT_DeviceDescription_Kind_Args>() == 40);
const _: () = assert!(size_of::<PJRT_DeviceDescription_DebugString_Args>() == 40);
const _: () = assert!(size_of::<PJRT_DeviceDescription_ToString_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Device_GetDescription_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Device_IsAddressable_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Device_LocalHardwareId_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Device_AddressableMemories_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Device_DefaultMemory_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Device_MemoryStats_Args>() == 192);
const _: () = assert!(size_of::<PJRT_Device_PoisonExecution_Args>() == 72);
const _: () = assert!(size_of::<PJRT_Device_CreateAsyncTrackingEvent_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Device_GetAttributes_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Device_ClearMemoryStats_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Memory_Id_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Memory_Kind_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Memory_DebugString_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Memory_ToString_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Memory_AddressableByDevices_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Memory_Kind_Id_Args>() == 32);
const _: () = assert!(size_of::<PJRT_TopologyDescription_Create_Args>() == 56);
const _: () = assert!(size_of::<PJRT_TopologyDescription_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_TopologyDescription_PlatformName_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TopologyDescription_PlatformVersion_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TopologyDescription_GetDeviceDescriptions_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TopologyDescription_Serialize_Args>() == 56);
const _: () = assert!(size_of::<PJRT_TopologyDescription_Attributes_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TopologyDescription_Deserialize_Args>() == 40);
const _: () = assert!(size_of::<PJRT_TopologyDescription_Fingerprint_Args>() == 32);
const _: () =
    assert!(size_of::<PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args>() == 88);
const _: () = assert!(size_of::<PJRT_TopologyDescription_GetMemorySpaceKindIds_Args>() == 40);
