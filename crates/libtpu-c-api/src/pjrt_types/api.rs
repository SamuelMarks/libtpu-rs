//! `#[repr(C)]` `PJRT_Api` vtable (`v0.116`, 1,144 bytes), `PJRT_Api_Version`, and `PJRT_Extension_Base`.

use crate::pjrt_types::PJRT_Error;

/// Exact byte size of `PJRT_Api` in `libtpu-0.0.50` (`0x478` = 1,144 bytes).
pub const PJRT_API_STRUCT_SIZE: usize = 1144;

/// Exact byte size of `PJRT_Api_Version` (`0x18` = 24 bytes).
pub const PJRT_API_VERSION_STRUCT_SIZE: usize = 24;

/// Supported PJRT major API version (`0`).
pub const PJRT_API_MAJOR: i32 = 0;

/// Supported PJRT minor API version (`116` / `0x74`).
pub const PJRT_API_MINOR: i32 = 116;

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
    /// Slot 117 (`0x3a8`): `PJRT_Client_CreateUninitializedBuffer`.
    pub pjrt_client_create_uninitialized_buffer: PjrtFn,
    /// Slot 118 (`0x3b0`): `PJRT_Client_UpdateGlobalProcessInfo`.
    pub pjrt_client_update_global_process_info: PjrtFn,
    /// Slot 119 (`0x3b8`): `PJRT_TopologyDescription_Deserialize`.
    pub pjrt_topology_description_deserialize: PjrtFn,
    /// Slot 120 (`0x3c0`): `PJRT_Client_CreateAliasBuffer`.
    pub pjrt_client_create_alias_buffer: PjrtFn,
    /// Slot 121 (`0x3c8`): `PJRT_Client_FulfillAliasBuffer`.
    pub pjrt_client_fulfill_alias_buffer: PjrtFn,
    /// Slot 122 (`0x3d0`): `PJRT_LoadedExecutable_GetDeviceAssignment`.
    pub pjrt_loaded_executable_get_device_assignment: PjrtFn,
    /// Slot 123 (`0x3d8`): `PJRT_Client_CreateErrorBuffer`.
    pub pjrt_client_create_error_buffer: PjrtFn,
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

const _: () = assert!(size_of::<PJRT_Api_Version>() == PJRT_API_VERSION_STRUCT_SIZE);
const _: () = assert!(size_of::<PJRT_Api>() == PJRT_API_STRUCT_SIZE);

const _: () = assert!(std::mem::offset_of!(PJRT_Api, struct_size) == 0x000);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, extension_start) == 0x008);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_api_version) == 0x010);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_error_destroy) == 0x028);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_error_message) == 0x030);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_error_get_code) == 0x038);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_plugin_initialize) == 0x040);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_plugin_attributes) == 0x048);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_event_destroy) == 0x050);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_event_is_ready) == 0x058);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_event_error) == 0x060);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_event_await) == 0x068);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_event_on_ready) == 0x070);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_create) == 0x078);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_destroy) == 0x080);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_platform_name) == 0x088);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_process_index) == 0x090);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_platform_version) == 0x098);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_devices) == 0x0a0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_addressable_devices) == 0x0a8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_lookup_device) == 0x0b0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_lookup_addressable_device) == 0x0b8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_addressable_memories) == 0x0c0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_compile) == 0x0c8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_default_device_assignment) == 0x0d0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_buffer_from_host_buffer) == 0x0d8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_description_id) == 0x0e0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_description_process_index) == 0x0e8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_description_attributes) == 0x0f0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_description_kind) == 0x0f8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_description_debug_string) == 0x100);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_description_to_string) == 0x108);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_get_description) == 0x110);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_is_addressable) == 0x118);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_local_hardware_id) == 0x120);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_addressable_memories) == 0x128);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_default_memory) == 0x130);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_memory_stats) == 0x138);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_memory_id) == 0x140);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_memory_kind) == 0x148);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_memory_debug_string) == 0x150);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_memory_to_string) == 0x158);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_memory_addressable_by_devices) == 0x160);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_destroy) == 0x168);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_name) == 0x170);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_num_replicas) == 0x178);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_num_partitions) == 0x180);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_num_outputs) == 0x188);
const _: () = assert!(
    std::mem::offset_of!(PJRT_Api, pjrt_executable_size_of_generated_code_in_bytes) == 0x190
);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_get_cost_analysis) == 0x198);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_output_memory_kinds) == 0x1a0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_optimized_program) == 0x1a8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_serialize) == 0x1b0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_destroy) == 0x1b8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_get_executable) == 0x1c0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_addressable_devices) == 0x1c8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_delete) == 0x1d0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_is_deleted) == 0x1d8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_execute) == 0x1e0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_deserialize_and_load) == 0x1e8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_fingerprint) == 0x1f0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_destroy) == 0x1f8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_element_type) == 0x200);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_dimensions) == 0x208);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_unpadded_dimensions) == 0x210);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_dynamic_dimension_indices) == 0x218);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_get_memory_layout) == 0x220);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_on_device_size_in_bytes) == 0x228);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_device) == 0x230);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_memory) == 0x238);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_delete) == 0x240);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_is_deleted) == 0x248);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_copy_to_device) == 0x250);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_to_host_buffer) == 0x258);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_is_on_cpu) == 0x260);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_ready_event) == 0x268);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_unsafe_pointer) == 0x270);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_increase_external_reference_count) == 0x278);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_decrease_external_reference_count) == 0x280);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_opaque_device_memory_data_pointer) == 0x288);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_copy_to_device_stream_destroy) == 0x290);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_copy_to_device_stream_add_chunk) == 0x298);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_copy_to_device_stream_total_bytes) == 0x2a0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_copy_to_device_stream_granule_size) == 0x2a8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_copy_to_device_stream_current_bytes) == 0x2b0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_create) == 0x2b8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_destroy) == 0x2c0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_platform_name) == 0x2c8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_platform_version) == 0x2d0);
const _: () = assert!(
    std::mem::offset_of!(PJRT_Api, pjrt_topology_description_get_device_descriptions) == 0x2d8
);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_serialize) == 0x2e0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_attributes) == 0x2e8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_compile) == 0x2f0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_output_element_types) == 0x2f8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_output_dimensions) == 0x300);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_copy_to_memory) == 0x308);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_create_view_of_device_buffer) == 0x310);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_fingerprint) == 0x318);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_topology_description) == 0x320);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_get_compiled_memory_stats) == 0x328);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_memory_kind_id) == 0x330);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_execute_context_create) == 0x338);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_execute_context_destroy) == 0x340);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_copy_raw_to_host) == 0x348);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_destroy) == 0x350);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_transfer_data) == 0x358);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_Api,
        pjrt_client_create_buffers_for_async_host_to_device
    ) == 0x360
);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_retrieve_buffer) == 0x368);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_device) == 0x370);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_buffer_count) == 0x378);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_buffer_size) == 0x380);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_set_buffer_error) == 0x388);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_add_metadata) == 0x390);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_dma_map) == 0x398);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_dma_unmap) == 0x3a0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_create_uninitialized_buffer) == 0x3a8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_update_global_process_info) == 0x3b0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_deserialize) == 0x3b8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_create_alias_buffer) == 0x3c0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_fulfill_alias_buffer) == 0x3c8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_loaded_executable_get_device_assignment) == 0x3d0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_create_error_buffer) == 0x3d8);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_transfer_manager_transfer_literal) == 0x3e0);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_copy_raw_to_host_future) == 0x3e8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_poison_execution) == 0x3f0);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_create_async_tracking_event) == 0x3f8);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_async_tracking_event_destroy) == 0x400);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_get_compile_options) == 0x408);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_donate_with_control_dependency) == 0x410);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_event_create) == 0x418);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_event_set) == 0x420);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_get_attributes) == 0x428);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_client_load) == 0x430);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_Api,
        pjrt_loaded_executable_addressable_device_logical_ids
    ) == 0x438
);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_buffer_bitcast) == 0x440);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_error_for_each_payload) == 0x448);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_topology_description_fingerprint) == 0x450);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Api, pjrt_executable_parameter_memory_kinds) == 0x458);
const _: () = assert!(std::mem::offset_of!(PJRT_Api, pjrt_device_clear_memory_stats) == 0x460);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_Api,
        pjrt_topology_description_make_canonical_shape_for_memory_space
    ) == 0x468
);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_Api,
        pjrt_topology_description_get_memory_space_kind_ids
    ) == 0x470
);

const _: () = assert!(std::mem::offset_of!(PJRT_Extension_Base, struct_size) == 0x00);
const _: () = assert!(std::mem::offset_of!(PJRT_Extension_Base, ext_type) == 0x08);
const _: () = assert!(std::mem::offset_of!(PJRT_Extension_Base, next) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_Api_Version, struct_size) == 0x00);
const _: () = assert!(std::mem::offset_of!(PJRT_Api_Version, extension_start) == 0x08);
const _: () = assert!(std::mem::offset_of!(PJRT_Api_Version, major_version) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_Api_Version, minor_version) == 0x14);
