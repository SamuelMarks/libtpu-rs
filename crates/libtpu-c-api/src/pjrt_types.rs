//! `#[repr(C)]` PJRT C API structures matching `libtpu.so` (`PJRT_Api` v0.116, 1,144 bytes).
//!
//! Defines the exact binary layout of `PJRT_Api` (with all 138 function pointer slots named
//! explicitly), `PJRT_Api_Version`, `PJRT_Extension_Base`, opaque PJRT handles, and all 138
//! `#[repr(C)]` `PJRT_*_Args` structures with compile-time byte size assertions.

use libtpu_core::error::LibTpuError;
use libtpu_core::pjrt::{
    PjrtCoreAsyncTransferManager, PjrtCoreBuffer, PjrtCoreClient, PjrtCoreCopyToDeviceStream,
    PjrtCoreDevice, PjrtCoreDeviceDescription, PjrtCoreEvent, PjrtCoreExecutable,
    PjrtCoreLoadedExecutable, PjrtCoreMemory, PjrtCoreTopology,
};
use std::ffi::CString;
use std::os::raw::c_char;

/// Exact byte size of `PJRT_Api` in `libtpu-0.0.50` (`0x478` = 1,144 bytes).
pub const PJRT_API_STRUCT_SIZE: usize = 1144;

/// Exact byte size of `PJRT_Api_Version` (`0x18` = 24 bytes).
pub const PJRT_API_VERSION_STRUCT_SIZE: usize = 24;

/// Supported PJRT major API version (`0`).
pub const PJRT_API_MAJOR: i32 = 0;

/// Supported PJRT minor API version (`116` / `0x74`).
pub const PJRT_API_MINOR: i32 = 116;

/// Opaque C handle wrapping a [`LibTpuError`] and its cached null-terminated message.
#[derive(Debug)]
pub struct PJRT_Error {
    /// Underlying core error.
    pub error: LibTpuError,
    /// Cached C-compatible error message string.
    pub c_message: CString,
}

/// Opaque C handle wrapping a [`PjrtCoreClient`].
#[derive(Debug)]
pub struct PJRT_Client {
    /// Underlying core client state.
    pub inner: PjrtCoreClient,
    /// Cached platform name (`"tpu"`).
    pub platform_name: CString,
    /// Cached platform version string.
    pub platform_version: CString,
    /// Heap-allocated device wrappers owned by this client.
    pub device_ptrs: Vec<*mut PJRT_Device>,
    /// Heap-allocated addressable device pointers.
    pub addressable_device_ptrs: Vec<*mut PJRT_Device>,
    /// Heap-allocated memory space wrappers owned by this client.
    pub memory_ptrs: Vec<*mut PJRT_Memory>,
    /// Heap-allocated topology wrapper owned by this client.
    pub topology_ptr: *mut PJRT_TopologyDescription,
}

/// Opaque C handle wrapping a [`PjrtCoreDeviceDescription`].
#[derive(Debug)]
pub struct PJRT_DeviceDescription {
    /// Underlying core device description.
    pub inner: PjrtCoreDeviceDescription,
}

/// Opaque C handle wrapping a [`PjrtCoreDevice`].
#[derive(Debug)]
pub struct PJRT_Device {
    /// Underlying core device.
    pub inner: PjrtCoreDevice,
    /// Owned pointer to the device description wrapper.
    pub desc_ptr: *mut PJRT_DeviceDescription,
}

/// Opaque C handle wrapping a [`PjrtCoreMemory`].
#[derive(Debug)]
pub struct PJRT_Memory {
    /// Underlying core memory space.
    pub inner: PjrtCoreMemory,
}

/// Opaque C handle wrapping a [`PjrtCoreBuffer`].
#[derive(Debug)]
pub struct PJRT_Buffer {
    /// Underlying core buffer.
    pub inner: PjrtCoreBuffer,
    /// Owning device wrapper pointer.
    pub device_ptr: *mut PJRT_Device,
    /// Owning memory wrapper pointer.
    pub memory_ptr: *mut PJRT_Memory,
}

/// Opaque C handle wrapping a [`PjrtCoreExecutable`].
#[derive(Debug)]
pub struct PJRT_Executable {
    /// Underlying core executable.
    pub inner: PjrtCoreExecutable,
}

/// Opaque C handle wrapping a [`PjrtCoreLoadedExecutable`].
#[derive(Debug)]
pub struct PJRT_LoadedExecutable {
    /// Underlying core loaded executable.
    pub inner: PjrtCoreLoadedExecutable,
    /// Owned executable wrapper pointer.
    pub executable_ptr: *mut PJRT_Executable,
    /// Addressable device wrapper pointers owned by the parent client.
    pub device_ptrs: Vec<*mut PJRT_Device>,
}

/// Opaque C handle wrapping a [`PjrtCoreEvent`].
#[derive(Debug)]
pub struct PJRT_Event {
    /// Underlying core event.
    pub inner: PjrtCoreEvent,
}

/// Opaque C handle wrapping a [`PjrtCoreTopology`].
#[derive(Debug)]
pub struct PJRT_TopologyDescription {
    /// Underlying core topology.
    pub inner: PjrtCoreTopology,
    /// Owned device description pointers.
    pub desc_ptrs: Vec<*mut PJRT_DeviceDescription>,
}

/// Opaque C handle representing a `PJRT_ExecuteContext`.
#[derive(Debug)]
pub struct PJRT_ExecuteContext {
    /// Unique context identifier.
    pub context_id: u64,
}

/// Opaque C handle wrapping a [`PjrtCoreCopyToDeviceStream`].
#[derive(Debug)]
pub struct PJRT_CopyToDeviceStream {
    /// Underlying stream state.
    pub inner: PjrtCoreCopyToDeviceStream,
}

/// Opaque C handle wrapping a [`PjrtCoreAsyncTransferManager`].
#[derive(Debug)]
pub struct PJRT_AsyncHostToDeviceTransferManager {
    /// Underlying async transfer manager.
    pub inner: PjrtCoreAsyncTransferManager,
}

/// Base header for all PJRT extension structures (`PJRT_Extension_Base`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Extension_Base {
    /// Size of the extension structure in bytes.
    pub struct_size: usize,
    /// Extension type discriminator (`PJRT_Extension_Type`).
    pub ext_type: i32,
    /// Explicit 4-byte padding on 64-bit targets.
    pub pad: i32,
    /// Pointer to the next extension in the linked list, or null.
    pub next: *const Self,
}

// SAFETY: Static extension descriptors are immutable after compile-time initialization.
unsafe impl Sync for PJRT_Extension_Base {}

/// PJRT API version header (`PJRT_Api_Version`, 24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Api_Version {
    /// Size of this structure (`24`).
    pub struct_size: usize,
    /// Optional extension pointer (`NULL`).
    pub extension_start: *const PJRT_Extension_Base,
    /// Major version number (`0`).
    pub major_version: i32,
    /// Minor version number (`116`).
    pub minor_version: i32,
}

// SAFETY: Version header contains only static pointers and plain integers.
unsafe impl Sync for PJRT_Api_Version {}

/// Generic PJRT function pointer signature taking an opaque args pointer and returning `*mut PJRT_Error`.
pub type PjrtFn = unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error;

/// Void-returning PJRT function pointer signature (used by `PJRT_Error_Destroy` and `PJRT_Error_Message`).
pub type PjrtVoidFn = unsafe extern "C" fn(*mut std::ffi::c_void);

/// Exact 1,144-byte (`143 * 8`) `PJRT_Api` vtable matching `libtpu.so` v0.116 with all 138 slots named.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Api {
    /// Total byte size of `PJRT_Api` (`1144` / `0x478`).
    pub struct_size: usize,
    /// Pointer to the head of the 17-node TPU PJRT extension linked list.
    pub extension_start: *const PJRT_Extension_Base,
    /// Embedded 24-byte `PJRT_Api_Version` (`0.116`).
    pub pjrt_api_version: PJRT_Api_Version,
    /// Slot 5 (`0x028`): `PJRT_Error_Destroy`.
    pub pjrt_error_destroy: PjrtVoidFn,
    /// Slot 6 (`0x030`): `PJRT_Error_Message`.
    pub pjrt_error_message: PjrtVoidFn,
    /// Slot 7 (`0x038`): `PJRT_Error_GetCode`.
    pub pjrt_error_get_code: PjrtFn,
    /// Slot 8 (`0x040`): `PJRT_Plugin_Initialize`.
    pub pjrt_plugin_initialize: PjrtFn,
    /// Slot 9 (`0x048`): `PJRT_Plugin_Attributes`.
    pub pjrt_plugin_attributes: PjrtFn,
    /// Slot 10 (`0x050`): `PJRT_Event_Destroy`.
    pub pjrt_event_destroy: PjrtFn,
    /// Slot 11 (`0x058`): `PJRT_Event_IsReady`.
    pub pjrt_event_is_ready: PjrtFn,
    /// Slot 12 (`0x060`): `PJRT_Event_Error`.
    pub pjrt_event_error: PjrtFn,
    /// Slot 13 (`0x068`): `PJRT_Event_Await`.
    pub pjrt_event_await: PjrtFn,
    /// Slot 14 (`0x070`): `PJRT_Event_OnReady`.
    pub pjrt_event_on_ready: PjrtFn,
    /// Slot 15 (`0x078`): `PJRT_Client_Create`.
    pub pjrt_client_create: PjrtFn,
    /// Slot 16 (`0x080`): `PJRT_Client_Destroy`.
    pub pjrt_client_destroy: PjrtFn,
    /// Slot 17 (`0x088`): `PJRT_Client_PlatformName`.
    pub pjrt_client_platform_name: PjrtFn,
    /// Slot 18 (`0x090`): `PJRT_Client_ProcessIndex`.
    pub pjrt_client_process_index: PjrtFn,
    /// Slot 19 (`0x098`): `PJRT_Client_PlatformVersion`.
    pub pjrt_client_platform_version: PjrtFn,
    /// Slot 20 (`0x0a0`): `PJRT_Client_Devices`.
    pub pjrt_client_devices: PjrtFn,
    /// Slot 21 (`0x0a8`): `PJRT_Client_AddressableDevices`.
    pub pjrt_client_addressable_devices: PjrtFn,
    /// Slot 22 (`0x0b0`): `PJRT_Client_LookupDevice`.
    pub pjrt_client_lookup_device: PjrtFn,
    /// Slot 23 (`0x0b8`): `PJRT_Client_LookupAddressableDevice`.
    pub pjrt_client_lookup_addressable_device: PjrtFn,
    /// Slot 24 (`0x0c0`): `PJRT_Client_AddressableMemories`.
    pub pjrt_client_addressable_memories: PjrtFn,
    /// Slot 25 (`0x0c8`): `PJRT_Client_Compile`.
    pub pjrt_client_compile: PjrtFn,
    /// Slot 26 (`0x0d0`): `PJRT_Client_DefaultDeviceAssignment`.
    pub pjrt_client_default_device_assignment: PjrtFn,
    /// Slot 27 (`0x0d8`): `PJRT_Client_BufferFromHostBuffer`.
    pub pjrt_client_buffer_from_host_buffer: PjrtFn,
    /// Slot 28 (`0x0e0`): `PJRT_DeviceDescription_Id`.
    pub pjrt_device_description_id: PjrtFn,
    /// Slot 29 (`0x0e8`): `PJRT_DeviceDescription_ProcessIndex`.
    pub pjrt_device_description_process_index: PjrtFn,
    /// Slot 30 (`0x0f0`): `PJRT_DeviceDescription_Attributes`.
    pub pjrt_device_description_attributes: PjrtFn,
    /// Slot 31 (`0x0f8`): `PJRT_DeviceDescription_Kind`.
    pub pjrt_device_description_kind: PjrtFn,
    /// Slot 32 (`0x100`): `PJRT_DeviceDescription_DebugString`.
    pub pjrt_device_description_debug_string: PjrtFn,
    /// Slot 33 (`0x108`): `PJRT_DeviceDescription_ToString`.
    pub pjrt_device_description_to_string: PjrtFn,
    /// Slot 34 (`0x110`): `PJRT_Device_GetDescription`.
    pub pjrt_device_get_description: PjrtFn,
    /// Slot 35 (`0x118`): `PJRT_Device_IsAddressable`.
    pub pjrt_device_is_addressable: PjrtFn,
    /// Slot 36 (`0x120`): `PJRT_Device_LocalHardwareId`.
    pub pjrt_device_local_hardware_id: PjrtFn,
    /// Slot 37 (`0x128`): `PJRT_Device_AddressableMemories`.
    pub pjrt_device_addressable_memories: PjrtFn,
    /// Slot 38 (`0x130`): `PJRT_Device_DefaultMemory`.
    pub pjrt_device_default_memory: PjrtFn,
    /// Slot 39 (`0x138`): `PJRT_Device_MemoryStats`.
    pub pjrt_device_memory_stats: PjrtFn,
    /// Slot 40 (`0x140`): `PJRT_Memory_Id`.
    pub pjrt_memory_id: PjrtFn,
    /// Slot 41 (`0x148`): `PJRT_Memory_Kind`.
    pub pjrt_memory_kind: PjrtFn,
    /// Slot 42 (`0x150`): `PJRT_Memory_DebugString`.
    pub pjrt_memory_debug_string: PjrtFn,
    /// Slot 43 (`0x158`): `PJRT_Memory_ToString`.
    pub pjrt_memory_to_string: PjrtFn,
    /// Slot 44 (`0x160`): `PJRT_Memory_AddressableByDevices`.
    pub pjrt_memory_addressable_by_devices: PjrtFn,
    /// Slot 45 (`0x168`): `PJRT_Executable_Destroy`.
    pub pjrt_executable_destroy: PjrtFn,
    /// Slot 46 (`0x170`): `PJRT_Executable_Name`.
    pub pjrt_executable_name: PjrtFn,
    /// Slot 47 (`0x178`): `PJRT_Executable_NumReplicas`.
    pub pjrt_executable_num_replicas: PjrtFn,
    /// Slot 48 (`0x180`): `PJRT_Executable_NumPartitions`.
    pub pjrt_executable_num_partitions: PjrtFn,
    /// Slot 49 (`0x188`): `PJRT_Executable_NumOutputs`.
    pub pjrt_executable_num_outputs: PjrtFn,
    /// Slot 50 (`0x190`): `PJRT_Executable_SizeOfGeneratedCodeInBytes`.
    pub pjrt_executable_size_of_generated_code_in_bytes: PjrtFn,
    /// Slot 51 (`0x198`): `PJRT_Executable_GetCostAnalysis`.
    pub pjrt_executable_get_cost_analysis: PjrtFn,
    /// Slot 52 (`0x1a0`): `PJRT_Executable_OutputMemoryKinds`.
    pub pjrt_executable_output_memory_kinds: PjrtFn,
    /// Slot 53 (`0x1a8`): `PJRT_Executable_OptimizedProgram`.
    pub pjrt_executable_optimized_program: PjrtFn,
    /// Slot 54 (`0x1b0`): `PJRT_Executable_Serialize`.
    pub pjrt_executable_serialize: PjrtFn,
    /// Slot 55 (`0x1b8`): `PJRT_LoadedExecutable_Destroy`.
    pub pjrt_loaded_executable_destroy: PjrtFn,
    /// Slot 56 (`0x1c0`): `PJRT_LoadedExecutable_GetExecutable`.
    pub pjrt_loaded_executable_get_executable: PjrtFn,
    /// Slot 57 (`0x1c8`): `PJRT_LoadedExecutable_AddressableDevices`.
    pub pjrt_loaded_executable_addressable_devices: PjrtFn,
    /// Slot 58 (`0x1d0`): `PJRT_LoadedExecutable_Delete`.
    pub pjrt_loaded_executable_delete: PjrtFn,
    /// Slot 59 (`0x1d8`): `PJRT_LoadedExecutable_IsDeleted`.
    pub pjrt_loaded_executable_is_deleted: PjrtFn,
    /// Slot 60 (`0x1e0`): `PJRT_LoadedExecutable_Execute`.
    pub pjrt_loaded_executable_execute: PjrtFn,
    /// Slot 61 (`0x1e8`): `PJRT_Executable_DeserializeAndLoad`.
    pub pjrt_executable_deserialize_and_load: PjrtFn,
    /// Slot 62 (`0x1f0`): `PJRT_LoadedExecutable_Fingerprint`.
    pub pjrt_loaded_executable_fingerprint: PjrtFn,
    /// Slot 63 (`0x1f8`): `PJRT_Buffer_Destroy`.
    pub pjrt_buffer_destroy: PjrtFn,
    /// Slot 64 (`0x200`): `PJRT_Buffer_ElementType`.
    pub pjrt_buffer_element_type: PjrtFn,
    /// Slot 65 (`0x208`): `PJRT_Buffer_Dimensions`.
    pub pjrt_buffer_dimensions: PjrtFn,
    /// Slot 66 (`0x210`): `PJRT_Buffer_UnpaddedDimensions`.
    pub pjrt_buffer_unpadded_dimensions: PjrtFn,
    /// Slot 67 (`0x218`): `PJRT_Buffer_DynamicDimensionIndices`.
    pub pjrt_buffer_dynamic_dimension_indices: PjrtFn,
    /// Slot 68 (`0x220`): `PJRT_Buffer_GetMemoryLayout`.
    pub pjrt_buffer_get_memory_layout: PjrtFn,
    /// Slot 69 (`0x228`): `PJRT_Buffer_OnDeviceSizeInBytes`.
    pub pjrt_buffer_on_device_size_in_bytes: PjrtFn,
    /// Slot 70 (`0x230`): `PJRT_Buffer_Device`.
    pub pjrt_buffer_device: PjrtFn,
    /// Slot 71 (`0x238`): `PJRT_Buffer_Memory`.
    pub pjrt_buffer_memory: PjrtFn,
    /// Slot 72 (`0x240`): `PJRT_Buffer_Delete`.
    pub pjrt_buffer_delete: PjrtFn,
    /// Slot 73 (`0x248`): `PJRT_Buffer_IsDeleted`.
    pub pjrt_buffer_is_deleted: PjrtFn,
    /// Slot 74 (`0x250`): `PJRT_Buffer_CopyToDevice`.
    pub pjrt_buffer_copy_to_device: PjrtFn,
    /// Slot 75 (`0x258`): `PJRT_Buffer_ToHostBuffer`.
    pub pjrt_buffer_to_host_buffer: PjrtFn,
    /// Slot 76 (`0x260`): `PJRT_Buffer_IsOnCpu`.
    pub pjrt_buffer_is_on_cpu: PjrtFn,
    /// Slot 77 (`0x268`): `PJRT_Buffer_ReadyEvent`.
    pub pjrt_buffer_ready_event: PjrtFn,
    /// Slot 78 (`0x270`): `PJRT_Buffer_UnsafePointer`.
    pub pjrt_buffer_unsafe_pointer: PjrtFn,
    /// Slot 79 (`0x278`): `PJRT_Buffer_IncreaseExternalReferenceCount`.
    pub pjrt_buffer_increase_external_reference_count: PjrtFn,
    /// Slot 80 (`0x280`): `PJRT_Buffer_DecreaseExternalReferenceCount`.
    pub pjrt_buffer_decrease_external_reference_count: PjrtFn,
    /// Slot 81 (`0x288`): `PJRT_Buffer_OpaqueDeviceMemoryDataPointer`.
    pub pjrt_buffer_opaque_device_memory_data_pointer: PjrtFn,
    /// Slot 82 (`0x290`): `PJRT_CopyToDeviceStream_Destroy`.
    pub pjrt_copy_to_device_stream_destroy: PjrtFn,
    /// Slot 83 (`0x298`): `PJRT_CopyToDeviceStream_AddChunk`.
    pub pjrt_copy_to_device_stream_add_chunk: PjrtFn,
    /// Slot 84 (`0x2a0`): `PJRT_CopyToDeviceStream_TotalBytes`.
    pub pjrt_copy_to_device_stream_total_bytes: PjrtFn,
    /// Slot 85 (`0x2a8`): `PJRT_CopyToDeviceStream_GranuleSize`.
    pub pjrt_copy_to_device_stream_granule_size: PjrtFn,
    /// Slot 86 (`0x2b0`): `PJRT_CopyToDeviceStream_CurrentBytes`.
    pub pjrt_copy_to_device_stream_current_bytes: PjrtFn,
    /// Slot 87 (`0x2b8`): `PJRT_TopologyDescription_Create`.
    pub pjrt_topology_description_create: PjrtFn,
    /// Slot 88 (`0x2c0`): `PJRT_TopologyDescription_Destroy`.
    pub pjrt_topology_description_destroy: PjrtFn,
    /// Slot 89 (`0x2c8`): `PJRT_TopologyDescription_PlatformName`.
    pub pjrt_topology_description_platform_name: PjrtFn,
    /// Slot 90 (`0x2d0`): `PJRT_TopologyDescription_PlatformVersion`.
    pub pjrt_topology_description_platform_version: PjrtFn,
    /// Slot 91 (`0x2d8`): `PJRT_TopologyDescription_GetDeviceDescriptions`.
    pub pjrt_topology_description_get_device_descriptions: PjrtFn,
    /// Slot 92 (`0x2e0`): `PJRT_TopologyDescription_Serialize`.
    pub pjrt_topology_description_serialize: PjrtFn,
    /// Slot 93 (`0x2e8`): `PJRT_TopologyDescription_Attributes`.
    pub pjrt_topology_description_attributes: PjrtFn,
    /// Slot 94 (`0x2f0`): `PJRT_Compile`.
    pub pjrt_compile: PjrtFn,
    /// Slot 95 (`0x2f8`): `PJRT_Executable_OutputElementTypes`.
    pub pjrt_executable_output_element_types: PjrtFn,
    /// Slot 96 (`0x300`): `PJRT_Executable_OutputDimensions`.
    pub pjrt_executable_output_dimensions: PjrtFn,
    /// Slot 97 (`0x308`): `PJRT_Buffer_CopyToMemory`.
    pub pjrt_buffer_copy_to_memory: PjrtFn,
    /// Slot 98 (`0x310`): `PJRT_Client_CreateViewOfDeviceBuffer`.
    pub pjrt_client_create_view_of_device_buffer: PjrtFn,
    /// Slot 99 (`0x318`): `PJRT_Executable_Fingerprint`.
    pub pjrt_executable_fingerprint: PjrtFn,
    /// Slot 100 (`0x320`): `PJRT_Client_TopologyDescription`.
    pub pjrt_client_topology_description: PjrtFn,
    /// Slot 101 (`0x328`): `PJRT_Executable_GetCompiledMemoryStats`.
    pub pjrt_executable_get_compiled_memory_stats: PjrtFn,
    /// Slot 102 (`0x330`): `PJRT_Memory_Kind_Id`.
    pub pjrt_memory_kind_id: PjrtFn,
    /// Slot 103 (`0x338`): `PJRT_ExecuteContext_Create`.
    pub pjrt_execute_context_create: PjrtFn,
    /// Slot 104 (`0x340`): `PJRT_ExecuteContext_Destroy`.
    pub pjrt_execute_context_destroy: PjrtFn,
    /// Slot 105 (`0x348`): `PJRT_Buffer_CopyRawToHost`.
    pub pjrt_buffer_copy_raw_to_host: PjrtFn,
    /// Slot 106 (`0x350`): `PJRT_AsyncHostToDeviceTransferManager_Destroy`.
    pub pjrt_async_transfer_manager_destroy: PjrtFn,
    /// Slot 107 (`0x358`): `PJRT_AsyncHostToDeviceTransferManager_TransferData`.
    pub pjrt_async_transfer_manager_transfer_data: PjrtFn,
    /// Slot 108 (`0x360`): `PJRT_Client_CreateBuffersForAsyncHostToDevice`.
    pub pjrt_client_create_buffers_for_async_host_to_device: PjrtFn,
    /// Slot 109 (`0x368`): `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer`.
    pub pjrt_async_transfer_manager_retrieve_buffer: PjrtFn,
    /// Slot 110 (`0x370`): `PJRT_AsyncHostToDeviceTransferManager_Device`.
    pub pjrt_async_transfer_manager_device: PjrtFn,
    /// Slot 111 (`0x378`): `PJRT_AsyncHostToDeviceTransferManager_BufferCount`.
    pub pjrt_async_transfer_manager_buffer_count: PjrtFn,
    /// Slot 112 (`0x380`): `PJRT_AsyncHostToDeviceTransferManager_BufferSize`.
    pub pjrt_async_transfer_manager_buffer_size: PjrtFn,
    /// Slot 113 (`0x388`): `PJRT_AsyncHostToDeviceTransferManager_SetBufferError`.
    pub pjrt_async_transfer_manager_set_buffer_error: PjrtFn,
    /// Slot 114 (`0x390`): `PJRT_AsyncHostToDeviceTransferManager_AddMetadata`.
    pub pjrt_async_transfer_manager_add_metadata: PjrtFn,
    /// Slot 115 (`0x398`): `PJRT_Client_DmaMap`.
    pub pjrt_client_dma_map: PjrtFn,
    /// Slot 116 (`0x3a0`): `PJRT_Client_DmaUnmap`.
    pub pjrt_client_dma_unmap: PjrtFn,
    /// Slot 117 (`0x3a8`): `PJRT_Client_CreateAliasBuffer`.
    pub pjrt_client_create_alias_buffer: PjrtFn,
    /// Slot 118 (`0x3b0`): `PJRT_Client_FulfillAliasBuffer`.
    pub pjrt_client_fulfill_alias_buffer: PjrtFn,
    /// Slot 119 (`0x3b8`): `PJRT_TopologyDescription_Deserialize`.
    pub pjrt_topology_description_deserialize: PjrtFn,
    /// Slot 120 (`0x3c0`): `PJRT_Client_CreateErrorBuffer`.
    pub pjrt_client_create_error_buffer: PjrtFn,
    /// Slot 121 (`0x3c8`): `PJRT_Client_CreateUninitializedBuffer`.
    pub pjrt_client_create_uninitialized_buffer: PjrtFn,
    /// Slot 122 (`0x3d0`): `PJRT_LoadedExecutable_GetDeviceAssignment`.
    pub pjrt_loaded_executable_get_device_assignment: PjrtFn,
    /// Slot 123 (`0x3d8`): `PJRT_Client_UpdateGlobalProcessInfo`.
    pub pjrt_client_update_global_process_info: PjrtFn,
    /// Slot 124 (`0x3e0`): `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`.
    pub pjrt_async_transfer_manager_transfer_literal: PjrtFn,
    /// Slot 125 (`0x3e8`): `PJRT_Buffer_CopyRawToHostFuture`.
    pub pjrt_buffer_copy_raw_to_host_future: PjrtFn,
    /// Slot 126 (`0x3f0`): `PJRT_Device_PoisonExecution`.
    pub pjrt_device_poison_execution: PjrtFn,
    /// Slot 127 (`0x3f8`): `PJRT_Device_CreateAsyncTrackingEvent`.
    pub pjrt_device_create_async_tracking_event: PjrtFn,
    /// Slot 128 (`0x400`): `PJRT_AsyncTrackingEvent_Destroy`.
    pub pjrt_async_tracking_event_destroy: PjrtFn,
    /// Slot 129 (`0x408`): `PJRT_Executable_GetCompileOptions`.
    pub pjrt_executable_get_compile_options: PjrtFn,
    /// Slot 130 (`0x410`): `PJRT_Buffer_DonateWithControlDependency`.
    pub pjrt_buffer_donate_with_control_dependency: PjrtFn,
    /// Slot 131 (`0x418`): `PJRT_Event_Create`.
    pub pjrt_event_create: PjrtFn,
    /// Slot 132 (`0x420`): `PJRT_Event_Set`.
    pub pjrt_event_set: PjrtFn,
    /// Slot 133 (`0x428`): `PJRT_Device_GetAttributes`.
    pub pjrt_device_get_attributes: PjrtFn,
    /// Slot 134 (`0x430`): `PJRT_Client_Load`.
    pub pjrt_client_load: PjrtFn,
    /// Slot 135 (`0x438`): `PJRT_LoadedExecutable_AddressableDeviceLogicalIds`.
    pub pjrt_loaded_executable_addressable_device_logical_ids: PjrtFn,
    /// Slot 136 (`0x440`): `PJRT_Buffer_Bitcast`.
    pub pjrt_buffer_bitcast: PjrtFn,
    /// Slot 137 (`0x448`): `PJRT_Error_ForEachPayload`.
    pub pjrt_error_for_each_payload: PjrtFn,
    /// Slot 138 (`0x450`): `PJRT_TopologyDescription_Fingerprint`.
    pub pjrt_topology_description_fingerprint: PjrtFn,
    /// Slot 139 (`0x458`): `PJRT_Executable_ParameterMemoryKinds`.
    pub pjrt_executable_parameter_memory_kinds: PjrtFn,
    /// Slot 140 (`0x460`): `PJRT_Device_ClearMemoryStats`.
    pub pjrt_device_clear_memory_stats: PjrtFn,
    /// Slot 141 (`0x468`): `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace`.
    pub pjrt_topology_description_make_canonical_shape_for_memory_space: PjrtFn,
    /// Slot 142 (`0x470`): `PJRT_TopologyDescription_GetMemorySpaceKindIds`.
    pub pjrt_topology_description_get_memory_space_kind_ids: PjrtFn,
}

// SAFETY: `PJRT_Api` contains only static function pointers and immutable extension metadata.
unsafe impl Sync for PJRT_Api {}

const _: () = assert!(std::mem::size_of::<PJRT_Api_Version>() == PJRT_API_VERSION_STRUCT_SIZE);
const _: () = assert!(std::mem::size_of::<PJRT_Api>() == PJRT_API_STRUCT_SIZE);

/// Argument structure for `PJRT_Error_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Error_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle to destroy.
    pub error: *mut PJRT_Error,
}

/// Argument structure for `PJRT_Error_Message` (`40` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Error_Message_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle to inspect.
    pub error: *const PJRT_Error,
    /// Output pointer to UTF-8 message bytes.
    pub message: *const c_char,
    /// Output length of `message` in bytes.
    pub message_size: usize,
}

/// Argument structure for `PJRT_Error_GetCode` (`32` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Error_GetCode_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle to inspect.
    pub error: *const PJRT_Error,
    /// Output `PJRT_Error_Code` integer.
    pub code: i32,
}

/// Argument structure for `PJRT_Plugin_Initialize` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Plugin_Initialize_Args {
    /// Size of this struct (`16`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
}

/// Argument structure for `PJRT_Client_Create` (`72` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Client_Create_Args {
    /// Size of this struct.
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

/// Program descriptor passed to `PJRT_Client_Compile` (`PJRT_Program`).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Program {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Pointer to program code bytes.
    pub code: *const c_char,
    /// Length of `code` in bytes.
    pub code_size: usize,
    /// Pointer to format string (`"mlir"` or `"hlo"`).
    pub format: *const c_char,
    /// Length of `format` in bytes.
    pub format_size: usize,
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

/// Opaque C handle representing a `PJRT_Stream`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Stream {
    /// Stream identifier.
    pub stream_id: u64,
}

/// Opaque C handle representing a `PJRT_AsyncTrackingEvent`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncTrackingEvent {
    /// Event identifier.
    pub event_id: u64,
}

/// Opaque C handle representing a `PJRT_SerializedExecutable`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_SerializedExecutable {
    /// Pointer to serialized bytes.
    pub ptr: *mut u8,
    /// Number of serialized bytes.
    pub len: usize,
}

/// Opaque C handle representing a `PJRT_SerializedTopology`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_SerializedTopology {
    /// Pointer to serialized bytes.
    pub ptr: *mut u8,
    /// Number of serialized bytes.
    pub len: usize,
}

/// Key-value attribute structure (`PJRT_NamedValue`, `40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_NamedValue {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Attribute name pointer.
    pub name: *const c_char,
    /// Attribute name byte length.
    pub name_size: usize,
    /// Value type discriminant (`i32`) and padding (`i32`).
    pub value_type_and_pad: u64,
    /// Union value payload (`int64`, `float`, `bool`, or string/array pointer).
    pub value_payload: usize,
    /// Value array size.
    pub value_size: usize,
}

/// Send callback descriptor (`PJRT_SendCallbackInfo`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_SendCallbackInfo {
    /// Channel identifier.
    pub channel_id: i64,
    /// User callback context pointer.
    pub user_arg: *mut std::ffi::c_void,
    /// Send callback function pointer.
    pub send_callback: *const std::ffi::c_void,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
}

/// Receive callback descriptor (`PJRT_RecvCallbackInfo`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RecvCallbackInfo {
    /// Channel identifier.
    pub channel_id: i64,
    /// User callback context pointer.
    pub user_arg: *mut std::ffi::c_void,
    /// Receive callback function pointer.
    pub recv_callback: *const std::ffi::c_void,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
}

/// Execution options passed to `PJRT_LoadedExecutable_Execute` (`PJRT_ExecuteOptions`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecuteOptions {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Send callbacks per device.
    pub send_callbacks: *const *const PJRT_SendCallbackInfo,
    /// Receive callbacks per device.
    pub recv_callbacks: *const *const PJRT_RecvCallbackInfo,
    /// Number of send callbacks per device.
    pub num_send_ops: usize,
    /// Number of receive callbacks per device.
    pub num_recv_ops: usize,
    /// Launch identifier (`i32`) and pad (`i32`).
    pub launch_id: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Non-donatable input indices pointer.
    pub non_donatable_input_indices: *const i64,
    /// Number of non-donatable input indices.
    pub num_non_donatable_input_indices: usize,
    /// Optional execution context handle.
    pub context: *mut PJRT_ExecuteContext,
}

/// Memory layout descriptor (`PJRT_Buffer_MemoryLayout`, `48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_MemoryLayout {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Minor-to-major dimension ordering or byte strides pointer.
    pub minor_to_major: *const i64,
    /// Number of minor-to-major dimensions.
    pub minor_to_major_size: usize,
    /// Tile dimensions pointer.
    pub tile_dims: *const i64,
    /// Number of tile dimensions (`usize`) and layout type (`i32`).
    pub tile_dim_sizes: *const usize,
}

/// Transfer metadata structure (`PJRT_TransferMetadata`, `24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TransferMetadata {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Metadata flags.
    pub flags: u64,
}

/// Logical `(replica, partition)` device ID pair (`PJRT_LogicalDeviceIds`, `8` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LogicalDeviceIds {
    /// Replica index.
    pub replica: i32,
    /// Partition index.
    pub partition: i32,
}

/// Shape specification for async H2D transfer (`PJRT_ShapeSpec`, `40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ShapeSpec {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Dimensions array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Element type (`i32`) and padding (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Host chunk passed to `PJRT_CopyToDeviceStream_AddChunk` (`PJRT_Chunk`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Chunk {
    /// Pointer to chunk data bytes.
    pub data: *mut std::ffi::c_void,
    /// Size of chunk in bytes.
    pub size: usize,
    /// Optional deleter callback (`void (*)(void* data, void* deleter_arg)`).
    pub deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void)>,
    /// Optional deleter user argument.
    pub deleter_arg: *mut std::ffi::c_void,
}

/// Multi-host process status descriptor (`PJRT_ProcessInfo`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ProcessInfo {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Task index.
    pub task_id: i32,
    /// Process state discriminant.
    pub state: i32,
    /// Incarnation ID.
    pub incarnation_id: u64,
    /// Error code (`i32`) and pad (`i32`).
    pub error_code: i64,
}

/// Argument structure for `PJRT_Plugin_Attributes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Plugin_Attributes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output pointer to `PJRT_NamedValue` array.
    pub attributes: *const PJRT_NamedValue,
    /// Output number of attributes.
    pub num_attributes: usize,
}

/// Argument structure for `PJRT_Event_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle to destroy.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_IsReady` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_IsReady_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
    /// Output readiness flag (`bool`).
    pub is_ready: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Event_Error` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Error_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_Await` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Await_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_OnReady` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_OnReady_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
    /// Callback function pointer.
    pub callback: Option<unsafe extern "C" fn(*mut PJRT_Error, *mut std::ffi::c_void)>,
    /// User argument pointer.
    pub user_arg: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Client_DefaultDeviceAssignment` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_DefaultDeviceAssignment_Args {
    /// Size of this struct (`56`).
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
    /// Reserved field.
    pub reserved: usize,
}

/// Argument structure for `PJRT_Client_BufferFromHostBuffer` (`112` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_BufferFromHostBuffer_Args {
    /// Size of this struct (`112`).
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
    /// Output `buffer` (`PJRT_Buffer*`).
    pub out_buffer: *mut PJRT_Buffer,
}

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

/// Argument structure for `PJRT_Device_MemoryStats` (`112` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_MemoryStats_Args {
    /// Size of this struct (`112`).
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

/// Argument structure for `PJRT_Executable_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle to destroy.
    pub executable: *mut PJRT_Executable,
}

/// Argument structure for `PJRT_Executable_Name` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Name_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output name string pointer.
    pub executable_name: *const c_char,
    /// Output name string length.
    pub executable_name_size: usize,
}

/// Argument structure for `PJRT_Executable_NumReplicas` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_NumReplicas_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_replicas`.
    pub num_replicas: usize,
}

/// Argument structure for `PJRT_Executable_NumPartitions` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_NumPartitions_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_partitions`.
    pub num_partitions: usize,
}

/// Argument structure for `PJRT_Executable_NumOutputs` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_NumOutputs_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_outputs`.
    pub num_outputs: usize,
}

/// Argument structure for `PJRT_Executable_SizeOfGeneratedCodeInBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `size_in_bytes`.
    pub size_in_bytes: i64,
}

/// Argument structure for `PJRT_Executable_GetCostAnalysis` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_GetCostAnalysis_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `PJRT_NamedValue` array pointer.
    pub properties: *const PJRT_NamedValue,
    /// Output number of properties.
    pub num_properties: usize,
}

/// Argument structure for `PJRT_Executable_OutputMemoryKinds` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OutputMemoryKinds_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_outputs`.
    pub num_outputs: usize,
    /// Output `memory_kinds` string pointer array.
    pub memory_kinds: *const *const c_char,
    /// Output `memory_kind_sizes` array.
    pub memory_kind_sizes: *const usize,
}

/// Argument structure for `PJRT_Executable_OptimizedProgram` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OptimizedProgram_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `PJRT_Program*`.
    pub program: *mut PJRT_Program,
}

/// Argument structure for `PJRT_Executable_Serialize` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Serialize_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output serialized bytes pointer.
    pub serialized_bytes: *const c_char,
    /// Output serialized byte count.
    pub serialized_bytes_size: usize,
    /// Output `PJRT_SerializedExecutable*`.
    pub serialized_executable: *mut PJRT_SerializedExecutable,
}

/// Argument structure for `PJRT_LoadedExecutable_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle to destroy.
    pub executable: *mut PJRT_LoadedExecutable,
}

/// Argument structure for `PJRT_LoadedExecutable_GetExecutable` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_GetExecutable_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub loaded_executable: *const PJRT_LoadedExecutable,
    /// Output `PJRT_Executable*`.
    pub executable: *mut PJRT_Executable,
}

/// Argument structure for `PJRT_LoadedExecutable_AddressableDevices` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_AddressableDevices_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output `PJRT_Device*` array pointer.
    pub addressable_devices: *const *mut PJRT_Device,
    /// Output number of addressable devices.
    pub num_addressable_devices: usize,
}

/// Argument structure for `PJRT_LoadedExecutable_Delete` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Delete_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *mut PJRT_LoadedExecutable,
}

/// Argument structure for `PJRT_LoadedExecutable_IsDeleted` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_IsDeleted_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output `is_deleted` flag.
    pub is_deleted: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_LoadedExecutable_Execute` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Execute_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *mut PJRT_LoadedExecutable,
    /// Optional `PJRT_ExecuteOptions*`.
    pub options: *const PJRT_ExecuteOptions,
    /// Input `PJRT_Buffer* const* const*` argument lists.
    pub argument_lists: *const *const *const PJRT_Buffer,
    /// Number of devices (`usize`).
    pub num_devices: usize,
    /// Output `PJRT_Buffer** const*` lists.
    pub output_lists: *const *mut *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Executable_DeserializeAndLoad` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_DeserializeAndLoad_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Serialized executable bytes pointer.
    pub serialized_executable: *const c_char,
    /// Serialized executable byte count.
    pub serialized_executable_size: usize,
    /// Output `PJRT_LoadedExecutable*`.
    pub loaded_executable: *mut PJRT_LoadedExecutable,
    /// Optional overridden compile options pointer.
    pub overridden_serialized_compile_options: *const c_char,
}

/// Argument structure for `PJRT_LoadedExecutable_Fingerprint` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Fingerprint_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output fingerprint string pointer.
    pub executable_fingerprint: *const c_char,
    /// Output fingerprint string length.
    pub executable_fingerprint_size: usize,
}

/// Argument structure for `PJRT_Buffer_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle to destroy.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_ElementType` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_ElementType_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output element type (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Argument structure for `PJRT_Buffer_Dimensions` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Dimensions_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output dimensions pointer (`const int64_t*`).
    pub dims: *const i64,
    /// Output number of dimensions.
    pub num_dims: usize,
}

/// Argument structure for `PJRT_Buffer_UnpaddedDimensions` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_UnpaddedDimensions_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output unpadded dimensions pointer (`const int64_t*`).
    pub unpadded_dims: *const i64,
    /// Output number of dimensions.
    pub num_dims: usize,
}

/// Argument structure for `PJRT_Buffer_DynamicDimensionIndices` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_DynamicDimensionIndices_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output dynamic dimension indices pointer.
    pub dynamic_dimension_indices: *const usize,
    /// Output number of dynamic dimensions.
    pub num_dynamic_dimensions: usize,
}

/// Argument structure for `PJRT_Buffer_GetMemoryLayout` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_GetMemoryLayout_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output memory layout fields.
    pub layout_words: [usize; 5],
}

/// Argument structure for `PJRT_Buffer_OnDeviceSizeInBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_OnDeviceSizeInBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `on_device_size_in_bytes`.
    pub on_device_size_in_bytes: usize,
}

/// Argument structure for `PJRT_Buffer_Device` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Device_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `PJRT_Device*`.
    pub device: *mut PJRT_Device,
}

/// Argument structure for `PJRT_Buffer_Memory` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Memory_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
}

/// Argument structure for `PJRT_Buffer_Delete` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Delete_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_IsDeleted` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_IsDeleted_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `is_deleted` flag.
    pub is_deleted: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Buffer_CopyToDevice` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyToDevice_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Destination `PJRT_Device*`.
    pub dst_device: *mut PJRT_Device,
    /// Output destination `PJRT_Buffer*`.
    pub dst_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_ToHostBuffer` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_ToHostBuffer_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source `PJRT_Buffer*`.
    pub src: *const PJRT_Buffer,
    /// Optional host layout pointer.
    pub host_layout: *const PJRT_Buffer_MemoryLayout,
    /// Destination host buffer pointer (or `NULL` to query size).
    pub dst: *mut std::ffi::c_void,
    /// Destination byte size.
    pub dst_size: usize,
    /// Output completion `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Buffer_IsOnCpu` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_IsOnCpu_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `is_on_cpu` flag.
    pub is_on_cpu: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Buffer_ReadyEvent` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_ReadyEvent_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Buffer_UnsafePointer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_UnsafePointer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output raw device pointer (`usize`).
    pub buffer_pointer: usize,
}

/// Argument structure for `PJRT_Buffer_IncreaseExternalReferenceCount` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_IncreaseExternalReferenceCount_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_DecreaseExternalReferenceCount` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_DecreaseExternalReferenceCount_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_OpaqueDeviceMemoryDataPointer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output opaque device memory pointer.
    pub device_memory_ptr: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_CopyToDeviceStream_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle to destroy.
    pub stream: *mut PJRT_CopyToDeviceStream,
}

/// Argument structure for `PJRT_CopyToDeviceStream_AddChunk` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_AddChunk_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *mut PJRT_CopyToDeviceStream,
    /// Input `PJRT_Chunk*`.
    pub chunk: *const PJRT_Chunk,
    /// Output `transfer_complete` (`PJRT_Event*`).
    pub transfer_complete: *mut PJRT_Event,
}

/// Argument structure for `PJRT_CopyToDeviceStream_TotalBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_TotalBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *const PJRT_CopyToDeviceStream,
    /// Output `total_bytes`.
    pub total_bytes: i64,
}

/// Argument structure for `PJRT_CopyToDeviceStream_GranuleSize` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_GranuleSize_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *const PJRT_CopyToDeviceStream,
    /// Output `granule_size_in_bytes`.
    pub granule_size_in_bytes: i64,
}

/// Argument structure for `PJRT_CopyToDeviceStream_CurrentBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_CurrentBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *const PJRT_CopyToDeviceStream,
    /// Output `current_bytes`.
    pub current_bytes: i64,
}

/// Argument structure for `PJRT_TopologyDescription_Create` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Create_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Topology name pointer.
    pub topology_name: *const c_char,
    /// Topology name byte length.
    pub topology_name_size: usize,
    /// Optional create options array.
    pub create_options: *const PJRT_NamedValue,
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

/// Argument structure for `PJRT_TopologyDescription_Serialize` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Serialize_Args {
    /// Size of this struct (`48`).
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

/// Argument structure for `PJRT_Compile` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Compile_Args {
    /// Size of this struct (`56`).
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
    /// Output `PJRT_Executable*`.
    pub executable: *mut PJRT_Executable,
}

/// Argument structure for `PJRT_Executable_OutputElementTypes` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OutputElementTypes_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `int32_t*` element types array.
    pub output_types: *const i32,
    /// Output number of output types.
    pub num_output_types: usize,
}

/// Argument structure for `PJRT_Executable_OutputDimensions` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OutputDimensions_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output number of outputs.
    pub num_outputs: usize,
    /// Output flattened `const int64_t*` dimensions.
    pub dims: *const i64,
    /// Output `const size_t*` dimension sizes per output.
    pub dim_sizes: *const usize,
}

/// Argument structure for `PJRT_Buffer_CopyToMemory` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyToMemory_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Destination `PJRT_Memory*`.
    pub dst_memory: *mut PJRT_Memory,
    /// Output destination `PJRT_Buffer*`.
    pub dst_buffer: *mut PJRT_Buffer,
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

/// Argument structure for `PJRT_Executable_Fingerprint` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Fingerprint_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output fingerprint string pointer.
    pub executable_fingerprint: *const c_char,
    /// Output fingerprint string length.
    pub executable_fingerprint_size: usize,
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

/// Argument structure for `PJRT_Executable_GetCompiledMemoryStats` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_GetCompiledMemoryStats_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `generated_code_size_in_bytes`.
    pub generated_code_size_in_bytes: i64,
    /// Output `argument_size_in_bytes`.
    pub argument_size_in_bytes: i64,
    /// Output `output_size_in_bytes`.
    pub output_size_in_bytes: i64,
    /// Output `alias_size_in_bytes`.
    pub alias_size_in_bytes: i64,
    /// Output `temp_size_in_bytes`.
    pub temp_size_in_bytes: i64,
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

/// Argument structure for `PJRT_ExecuteContext_Create` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecuteContext_Create_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output `PJRT_ExecuteContext*`.
    pub context: *mut PJRT_ExecuteContext,
}

/// Argument structure for `PJRT_ExecuteContext_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecuteContext_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Context handle to destroy.
    pub context: *mut PJRT_ExecuteContext,
}

/// Argument structure for `PJRT_Buffer_CopyRawToHost` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyRawToHost_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Destination host pointer.
    pub dst: *mut std::ffi::c_void,
    /// Byte offset into source buffer.
    pub offset: i64,
    /// Number of bytes to copy.
    pub transfer_size: i64,
    /// Output completion `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle to destroy.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_TransferData` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Target buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Source host data pointer.
    pub data: *const std::ffi::c_void,
    /// Byte offset into target buffer.
    pub offset: i64,
    /// Number of bytes to transfer.
    pub transfer_size: i64,
    /// Output `done_with_h2d_transfer` (`PJRT_Event*`).
    pub done_with_h2d_transfer: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Client_CreateBuffersForAsyncHostToDevice` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Shape specs array pointer.
    pub shape_specs: *const PJRT_ShapeSpec,
    /// Number of shape specs.
    pub num_shape_specs: usize,
    /// Output `PJRT_AsyncHostToDeviceTransferManager*`.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Output `PJRT_Buffer*`.
    pub buffer_out: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_Device` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_Device_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *const PJRT_AsyncHostToDeviceTransferManager,
    /// Output `PJRT_Device*`.
    pub device_out: *mut PJRT_Device,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_BufferCount` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *const PJRT_AsyncHostToDeviceTransferManager,
    /// Output `buffer_count`.
    pub buffer_count: usize,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_BufferSize` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *const PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Output `buffer_size`.
    pub buffer_size: usize,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_SetBufferError` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad0: i32,
    /// Error code (`i32`).
    pub error_code: i32,
    /// Explicit padding.
    pub pad1: i32,
    /// Error message pointer.
    pub error_message: *const c_char,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_AddMetadata` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Metadata array pointer.
    pub transfer_metadata: *const PJRT_TransferMetadata,
    /// Number of metadata entries.
    pub num_metadata: usize,
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

/// Argument structure for `PJRT_Client_CreateUninitializedBuffer` (`72` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateUninitializedBuffer_Args {
    /// Size of this struct (`72`).
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
    /// Output `PJRT_Buffer*`.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Client_UpdateGlobalProcessInfo` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_UpdateGlobalProcessInfo_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Process info array pointer.
    pub processes: *const PJRT_ProcessInfo,
    /// Number of process info entries.
    pub num_processes: usize,
}

/// Argument structure for `PJRT_TopologyDescription_Deserialize` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_Deserialize_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Serialized topology bytes pointer.
    pub serialized_topology: *const c_char,
    /// Serialized topology byte length.
    pub serialized_topology_size: usize,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Output `PJRT_TopologyDescription*`.
    pub topology: *mut PJRT_TopologyDescription,
}

/// Argument structure for `PJRT_Client_CreateAliasBuffer` (`72` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateAliasBuffer_Args {
    /// Size of this struct (`72`).
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
    /// Output alias `PJRT_Buffer*`.
    pub alias_buffer: *mut PJRT_Buffer,
    /// Output fulfillment callback handle.
    pub fulfill_alias_buffer_cb: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Client_FulfillAliasBuffer` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_FulfillAliasBuffer_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Concrete `PJRT_Buffer*`.
    pub buffer: *mut PJRT_Buffer,
    /// Fulfillment callback handle.
    pub fulfill_alias_buffer_cb: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_LoadedExecutable_GetDeviceAssignment` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_GetDeviceAssignment_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output serialized bytes pointer.
    pub serialized_bytes: *const c_char,
    /// Output serialized byte count.
    pub serialized_bytes_size: usize,
    /// Output serialized device assignment handle.
    pub serialized_device_assignment: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Client_CreateErrorBuffer` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_CreateErrorBuffer_Args {
    /// Size of this struct (`64`).
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
    /// Target `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
    /// Output poisoned `PJRT_Buffer*`.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Host literal data pointer.
    pub data: *const std::ffi::c_void,
    /// Shape dimensions pointer.
    pub shape_dims: *const i64,
    /// Number of shape dimensions.
    pub shape_num_dims: usize,
    /// Output `done_with_h2d_transfer` (`PJRT_Event*`).
    pub done_with_h2d_transfer: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Buffer_CopyRawToHostFuture` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyRawToHostFuture_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Byte offset.
    pub offset: i64,
    /// Transfer size in bytes.
    pub transfer_size: i64,
    /// Output completion `PJRT_Event*`.
    pub event: *mut PJRT_Event,
    /// Output callback data.
    pub callback_data: *mut std::ffi::c_void,
    /// Output future ready callback.
    pub future_ready_callback: *const std::ffi::c_void,
}

/// Argument structure for `PJRT_Device_PoisonExecution` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_PoisonExecution_Args {
    /// Size of this struct (`48`).
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

/// Argument structure for `PJRT_AsyncTrackingEvent_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncTrackingEvent_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Async tracking event handle to destroy.
    pub event: *mut PJRT_AsyncTrackingEvent,
}

/// Argument structure for `PJRT_Executable_GetCompileOptions` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_GetCompileOptions_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output serialized compile options bytes pointer.
    pub serialized_bytes: *const c_char,
    /// Output serialized byte count.
    pub serialized_bytes_size: usize,
    /// Output serialized compile options handle.
    pub serialized_compile_options: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Buffer_DonateWithControlDependency` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_DonateWithControlDependency_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
    /// Dependency callback data.
    pub callback_data: *mut std::ffi::c_void,
    /// Dependency callback pointer.
    pub dependency_callback: *const std::ffi::c_void,
    /// Output donated `PJRT_Buffer*`.
    pub out_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Event_Create` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Create_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output created `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_Set` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Set_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
    /// Status code (`i32`).
    pub error_code: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Error message pointer.
    pub error_message: *const c_char,
}

/// Argument structure for `PJRT_Device_GetAttributes` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Device_GetAttributes_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Device handle.
    pub device: *const PJRT_Device,
    /// Output `PJRT_NamedValue` array pointer.
    pub attributes: *const PJRT_NamedValue,
    /// Output number of attributes.
    pub num_attributes: usize,
}

/// Argument structure for `PJRT_Client_Load` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Client_Load_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Input `PJRT_Executable*`.
    pub executable: *mut PJRT_Executable,
    /// Optional compile options pointer.
    pub compile_options: *const c_char,
    /// Output `PJRT_LoadedExecutable*`.
    pub loaded_executable: *mut PJRT_LoadedExecutable,
}

/// Argument structure for `PJRT_LoadedExecutable_AddressableDeviceLogicalIds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output `PJRT_LogicalDeviceIds*` array.
    pub addressable_device_logical_ids: *const PJRT_LogicalDeviceIds,
    /// Output number of logical IDs.
    pub num_addressable_device_logical_ids: usize,
}

/// Argument structure for `PJRT_Buffer_Bitcast` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Bitcast_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *mut PJRT_Buffer,
    /// New element type (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
    /// New dimensions pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// New device layout pointer.
    pub device_layout: *const PJRT_Buffer_MemoryLayout,
    /// Output bitcast `PJRT_Buffer*`.
    pub out_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Error_ForEachPayload` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Error_ForEachPayload_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle.
    pub error: *const PJRT_Error,
    /// Payload visitor callback pointer.
    pub visitor: Option<
        unsafe extern "C" fn(*const c_char, usize, *const c_char, usize, *mut std::ffi::c_void),
    >,
    /// User argument pointer.
    pub user_arg: *mut std::ffi::c_void,
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

/// Argument structure for `PJRT_Executable_ParameterMemoryKinds` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_ParameterMemoryKinds_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_parameters`.
    pub num_parameters: usize,
    /// Output `memory_kinds` string array pointer.
    pub memory_kinds: *const *const c_char,
    /// Output `memory_kind_sizes` array pointer.
    pub memory_kind_sizes: *const usize,
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

/// Argument structure for `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
    /// Size of this struct (`64`).
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
    /// Reserved word.
    pub reserved: usize,
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

const _: () = assert!(std::mem::size_of::<PJRT_Error_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Error_Message_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Error_GetCode_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Plugin_Initialize_Args>() == 16);
const _: () = assert!(std::mem::size_of::<PJRT_Plugin_Attributes_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Event_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Event_IsReady_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Event_Error_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Event_Await_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Event_OnReady_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_Create_Args>() == 72);
const _: () = assert!(std::mem::size_of::<PJRT_Client_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Client_PlatformStr_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_ProcessIndex_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Client_Devices_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_LookupDevice_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_AddressableMemories_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_Compile_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Client_DefaultDeviceAssignment_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Client_BufferFromHostBuffer_Args>() == 112);
const _: () = assert!(std::mem::size_of::<PJRT_DeviceDescription_Id_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_DeviceDescription_ProcessIndex_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_DeviceDescription_Attributes_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_DeviceDescription_Kind_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_DeviceDescription_DebugString_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_DeviceDescription_ToString_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Device_GetDescription_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Device_IsAddressable_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Device_LocalHardwareId_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Device_AddressableMemories_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Device_DefaultMemory_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Device_MemoryStats_Args>() == 112);
const _: () = assert!(std::mem::size_of::<PJRT_Memory_Id_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Memory_Kind_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Memory_DebugString_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Memory_ToString_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Memory_AddressableByDevices_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_Name_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_NumReplicas_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_NumPartitions_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_NumOutputs_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_GetCostAnalysis_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_OutputMemoryKinds_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_OptimizedProgram_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_Serialize_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_GetExecutable_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_AddressableDevices_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_Delete_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_IsDeleted_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_Execute_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_DeserializeAndLoad_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_Fingerprint_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_ElementType_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_Dimensions_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_UnpaddedDimensions_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_DynamicDimensionIndices_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_GetMemoryLayout_Args>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_OnDeviceSizeInBytes_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_Device_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_Memory_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_Delete_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_IsDeleted_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_CopyToDevice_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_ToHostBuffer_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_IsOnCpu_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_ReadyEvent_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_UnsafePointer_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_IncreaseExternalReferenceCount_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_DecreaseExternalReferenceCount_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_CopyToDeviceStream_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_CopyToDeviceStream_AddChunk_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_CopyToDeviceStream_TotalBytes_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_CopyToDeviceStream_GranuleSize_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_CopyToDeviceStream_CurrentBytes_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_Create_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_PlatformName_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_PlatformVersion_Args>() == 40);
const _: () =
    assert!(std::mem::size_of::<PJRT_TopologyDescription_GetDeviceDescriptions_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_Serialize_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_Attributes_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Compile_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_OutputElementTypes_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_OutputDimensions_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_CopyToMemory_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_CreateViewOfDeviceBuffer_Args>() == 88);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_Fingerprint_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_TopologyDescription_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_GetCompiledMemoryStats_Args>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_Memory_Kind_Id_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_ExecuteContext_Create_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_ExecuteContext_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_CopyRawToHost_Args>() == 56);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_Destroy_Args>() == 24);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_TransferData_Args>() == 64);
const _: () =
    assert!(std::mem::size_of::<PJRT_Client_CreateBuffersForAsyncHostToDevice_Args>() == 48);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args>() == 40);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_Device_Args>() == 32);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args>() == 32);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args>() == 40);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args>() == 48);
const _: () =
    assert!(std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_DmaMap_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_DmaUnmap_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Client_CreateUninitializedBuffer_Args>() == 72);
const _: () = assert!(std::mem::size_of::<PJRT_Client_UpdateGlobalProcessInfo_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_Deserialize_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Client_CreateAliasBuffer_Args>() == 72);
const _: () = assert!(std::mem::size_of::<PJRT_Client_FulfillAliasBuffer_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_LoadedExecutable_GetDeviceAssignment_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Client_CreateErrorBuffer_Args>() == 64);
const _: () = assert!(
    std::mem::size_of::<PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args>() == 64
);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_CopyRawToHostFuture_Args>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_Device_PoisonExecution_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Device_CreateAsyncTrackingEvent_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_AsyncTrackingEvent_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_GetCompileOptions_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_DonateWithControlDependency_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Event_Create_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Event_Set_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Device_GetAttributes_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Client_Load_Args>() == 48);
const _: () =
    assert!(std::mem::size_of::<PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Buffer_Bitcast_Args>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_Error_ForEachPayload_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_TopologyDescription_Fingerprint_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Executable_ParameterMemoryKinds_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Device_ClearMemoryStats_Args>() == 24);
const _: () = assert!(
    std::mem::size_of::<PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args>() == 64
);
const _: () =
    assert!(std::mem::size_of::<PJRT_TopologyDescription_GetMemorySpaceKindIds_Args>() == 40);
