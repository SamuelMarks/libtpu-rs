//! `PJRT_Api` static vtable assembly and `GetPjrtApi` entrypoint.

use crate::extensions::EXT_01_XLA_TRANSFORM;
use crate::pjrt_buffer_api::{
    PJRT_AsyncHostToDeviceTransferManager_AddMetadata,
    PJRT_AsyncHostToDeviceTransferManager_BufferCount,
    PJRT_AsyncHostToDeviceTransferManager_BufferSize,
    PJRT_AsyncHostToDeviceTransferManager_Destroy, PJRT_AsyncHostToDeviceTransferManager_Device,
    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
    PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
    PJRT_AsyncHostToDeviceTransferManager_TransferData,
    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral, PJRT_Buffer_Bitcast,
    PJRT_Buffer_CopyRawToHost, PJRT_Buffer_CopyRawToHostFuture, PJRT_Buffer_CopyToDevice,
    PJRT_Buffer_CopyToMemory, PJRT_Buffer_DecreaseExternalReferenceCount, PJRT_Buffer_Delete,
    PJRT_Buffer_Destroy, PJRT_Buffer_Device, PJRT_Buffer_Dimensions,
    PJRT_Buffer_DonateWithControlDependency, PJRT_Buffer_DynamicDimensionIndices,
    PJRT_Buffer_ElementType, PJRT_Buffer_GetMemoryLayout,
    PJRT_Buffer_IncreaseExternalReferenceCount, PJRT_Buffer_IsDeleted, PJRT_Buffer_IsOnCpu,
    PJRT_Buffer_Memory, PJRT_Buffer_OnDeviceSizeInBytes, PJRT_Buffer_OpaqueDeviceMemoryDataPointer,
    PJRT_Buffer_ReadyEvent, PJRT_Buffer_ToHostBuffer, PJRT_Buffer_UnpaddedDimensions,
    PJRT_Buffer_UnsafePointer, PJRT_CopyToDeviceStream_AddChunk,
    PJRT_CopyToDeviceStream_CurrentBytes, PJRT_CopyToDeviceStream_Destroy,
    PJRT_CopyToDeviceStream_GranuleSize, PJRT_CopyToDeviceStream_TotalBytes,
};
use crate::pjrt_client_api::{
    PJRT_Client_AddressableDevices, PJRT_Client_AddressableMemories,
    PJRT_Client_BufferFromHostBuffer, PJRT_Client_Compile, PJRT_Client_Create,
    PJRT_Client_CreateAliasBuffer, PJRT_Client_CreateBuffersForAsyncHostToDevice,
    PJRT_Client_CreateErrorBuffer, PJRT_Client_CreateUninitializedBuffer,
    PJRT_Client_CreateViewOfDeviceBuffer, PJRT_Client_DefaultDeviceAssignment, PJRT_Client_Destroy,
    PJRT_Client_Devices, PJRT_Client_DmaMap, PJRT_Client_DmaUnmap, PJRT_Client_FulfillAliasBuffer,
    PJRT_Client_Load, PJRT_Client_LookupAddressableDevice, PJRT_Client_LookupDevice,
    PJRT_Client_PlatformName, PJRT_Client_PlatformVersion, PJRT_Client_ProcessIndex,
    PJRT_Client_TopologyDescription, PJRT_Client_UpdateGlobalProcessInfo,
    PJRT_ExecuteContext_Create, PJRT_ExecuteContext_Destroy, PJRT_TopologyDescription_Attributes,
    PJRT_TopologyDescription_Create, PJRT_TopologyDescription_Deserialize,
    PJRT_TopologyDescription_Destroy, PJRT_TopologyDescription_Fingerprint,
    PJRT_TopologyDescription_GetDeviceDescriptions, PJRT_TopologyDescription_GetMemorySpaceKindIds,
    PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace,
    PJRT_TopologyDescription_PlatformName, PJRT_TopologyDescription_PlatformVersion,
    PJRT_TopologyDescription_Serialize,
};
use crate::pjrt_device_api::{
    PJRT_AsyncTrackingEvent_Destroy, PJRT_Device_AddressableMemories, PJRT_Device_ClearMemoryStats,
    PJRT_Device_CreateAsyncTrackingEvent, PJRT_Device_DefaultMemory, PJRT_Device_GetAttributes,
    PJRT_Device_GetDescription, PJRT_Device_IsAddressable, PJRT_Device_LocalHardwareId,
    PJRT_Device_MemoryStats, PJRT_Device_PoisonExecution, PJRT_DeviceDescription_Attributes,
    PJRT_DeviceDescription_DebugString, PJRT_DeviceDescription_Id, PJRT_DeviceDescription_Kind,
    PJRT_DeviceDescription_ProcessIndex, PJRT_DeviceDescription_ToString,
    PJRT_Memory_AddressableByDevices, PJRT_Memory_DebugString, PJRT_Memory_Id, PJRT_Memory_Kind,
    PJRT_Memory_Kind_Id, PJRT_Memory_ToString,
};
use crate::pjrt_error_api::{
    PJRT_Error_Destroy, PJRT_Error_ForEachPayload, PJRT_Error_GetCode, PJRT_Error_Message,
    PJRT_Event_Await, PJRT_Event_Create, PJRT_Event_Destroy, PJRT_Event_Error, PJRT_Event_IsReady,
    PJRT_Event_OnReady, PJRT_Event_Set, PJRT_Plugin_Attributes, PJRT_Plugin_Initialize,
};
use crate::pjrt_executable_api::{
    PJRT_Compile, PJRT_Executable_DeserializeAndLoad, PJRT_Executable_Destroy,
    PJRT_Executable_Fingerprint, PJRT_Executable_GetCompileOptions,
    PJRT_Executable_GetCompiledMemoryStats, PJRT_Executable_GetCostAnalysis, PJRT_Executable_Name,
    PJRT_Executable_NumOutputs, PJRT_Executable_NumPartitions, PJRT_Executable_NumReplicas,
    PJRT_Executable_OptimizedProgram, PJRT_Executable_OutputDimensions,
    PJRT_Executable_OutputElementTypes, PJRT_Executable_OutputMemoryKinds,
    PJRT_Executable_ParameterMemoryKinds, PJRT_Executable_Serialize,
    PJRT_Executable_SizeOfGeneratedCodeInBytes, PJRT_LoadedExecutable_AddressableDeviceLogicalIds,
    PJRT_LoadedExecutable_AddressableDevices, PJRT_LoadedExecutable_Delete,
    PJRT_LoadedExecutable_Destroy, PJRT_LoadedExecutable_Execute,
    PJRT_LoadedExecutable_Fingerprint, PJRT_LoadedExecutable_GetDeviceAssignment,
    PJRT_LoadedExecutable_GetExecutable, PJRT_LoadedExecutable_IsDeleted,
};
use crate::pjrt_types::{
    PJRT_API_MAJOR, PJRT_API_MINOR, PJRT_API_STRUCT_SIZE, PJRT_API_VERSION_STRUCT_SIZE, PJRT_Api,
    PJRT_Api_Version,
};
use libtpu_core::types::{ByteSize, ChipCoordinates, DeviceAddress};
use std::ptr;

/// Static 1,144-byte `PJRT_Api` vtable returned by [`GetPjrtApi`].
pub(crate) static GLOBAL_PJRT_API: PJRT_Api = PJRT_Api {
    struct_size: PJRT_API_STRUCT_SIZE,
    extension_start: ptr::addr_of!(EXT_01_XLA_TRANSFORM.base),
    pjrt_api_version: PJRT_Api_Version {
        struct_size: PJRT_API_VERSION_STRUCT_SIZE,
        extension_start: ptr::null(),
        major_version: PJRT_API_MAJOR,
        minor_version: PJRT_API_MINOR,
    },
    pjrt_error_destroy: PJRT_Error_Destroy,
    pjrt_error_message: PJRT_Error_Message,
    pjrt_error_get_code: PJRT_Error_GetCode,
    pjrt_plugin_initialize: PJRT_Plugin_Initialize,
    pjrt_plugin_attributes: PJRT_Plugin_Attributes,
    pjrt_event_destroy: PJRT_Event_Destroy,
    pjrt_event_is_ready: PJRT_Event_IsReady,
    pjrt_event_error: PJRT_Event_Error,
    pjrt_event_await: PJRT_Event_Await,
    pjrt_event_on_ready: PJRT_Event_OnReady,
    pjrt_client_create: PJRT_Client_Create,
    pjrt_client_destroy: PJRT_Client_Destroy,
    pjrt_client_platform_name: PJRT_Client_PlatformName,
    pjrt_client_process_index: PJRT_Client_ProcessIndex,
    pjrt_client_platform_version: PJRT_Client_PlatformVersion,
    pjrt_client_devices: PJRT_Client_Devices,
    pjrt_client_addressable_devices: PJRT_Client_AddressableDevices,
    pjrt_client_lookup_device: PJRT_Client_LookupDevice,
    pjrt_client_lookup_addressable_device: PJRT_Client_LookupAddressableDevice,
    pjrt_client_addressable_memories: PJRT_Client_AddressableMemories,
    pjrt_client_compile: PJRT_Client_Compile,
    pjrt_client_default_device_assignment: PJRT_Client_DefaultDeviceAssignment,
    pjrt_client_buffer_from_host_buffer: PJRT_Client_BufferFromHostBuffer,
    pjrt_device_description_id: PJRT_DeviceDescription_Id,
    pjrt_device_description_process_index: PJRT_DeviceDescription_ProcessIndex,
    pjrt_device_description_attributes: PJRT_DeviceDescription_Attributes,
    pjrt_device_description_kind: PJRT_DeviceDescription_Kind,
    pjrt_device_description_debug_string: PJRT_DeviceDescription_DebugString,
    pjrt_device_description_to_string: PJRT_DeviceDescription_ToString,
    pjrt_device_get_description: PJRT_Device_GetDescription,
    pjrt_device_is_addressable: PJRT_Device_IsAddressable,
    pjrt_device_local_hardware_id: PJRT_Device_LocalHardwareId,
    pjrt_device_addressable_memories: PJRT_Device_AddressableMemories,
    pjrt_device_default_memory: PJRT_Device_DefaultMemory,
    pjrt_device_memory_stats: PJRT_Device_MemoryStats,
    pjrt_memory_id: PJRT_Memory_Id,
    pjrt_memory_kind: PJRT_Memory_Kind,
    pjrt_memory_debug_string: PJRT_Memory_DebugString,
    pjrt_memory_to_string: PJRT_Memory_ToString,
    pjrt_memory_addressable_by_devices: PJRT_Memory_AddressableByDevices,
    pjrt_executable_destroy: PJRT_Executable_Destroy,
    pjrt_executable_name: PJRT_Executable_Name,
    pjrt_executable_num_replicas: PJRT_Executable_NumReplicas,
    pjrt_executable_num_partitions: PJRT_Executable_NumPartitions,
    pjrt_executable_num_outputs: PJRT_Executable_NumOutputs,
    pjrt_executable_size_of_generated_code_in_bytes: PJRT_Executable_SizeOfGeneratedCodeInBytes,
    pjrt_executable_get_cost_analysis: PJRT_Executable_GetCostAnalysis,
    pjrt_executable_output_memory_kinds: PJRT_Executable_OutputMemoryKinds,
    pjrt_executable_optimized_program: PJRT_Executable_OptimizedProgram,
    pjrt_executable_serialize: PJRT_Executable_Serialize,
    pjrt_loaded_executable_destroy: PJRT_LoadedExecutable_Destroy,
    pjrt_loaded_executable_get_executable: PJRT_LoadedExecutable_GetExecutable,
    pjrt_loaded_executable_addressable_devices: PJRT_LoadedExecutable_AddressableDevices,
    pjrt_loaded_executable_delete: PJRT_LoadedExecutable_Delete,
    pjrt_loaded_executable_is_deleted: PJRT_LoadedExecutable_IsDeleted,
    pjrt_loaded_executable_execute: PJRT_LoadedExecutable_Execute,
    pjrt_executable_deserialize_and_load: PJRT_Executable_DeserializeAndLoad,
    pjrt_loaded_executable_fingerprint: PJRT_LoadedExecutable_Fingerprint,
    pjrt_buffer_destroy: PJRT_Buffer_Destroy,
    pjrt_buffer_element_type: PJRT_Buffer_ElementType,
    pjrt_buffer_dimensions: PJRT_Buffer_Dimensions,
    pjrt_buffer_unpadded_dimensions: PJRT_Buffer_UnpaddedDimensions,
    pjrt_buffer_dynamic_dimension_indices: PJRT_Buffer_DynamicDimensionIndices,
    pjrt_buffer_get_memory_layout: PJRT_Buffer_GetMemoryLayout,
    pjrt_buffer_on_device_size_in_bytes: PJRT_Buffer_OnDeviceSizeInBytes,
    pjrt_buffer_device: PJRT_Buffer_Device,
    pjrt_buffer_memory: PJRT_Buffer_Memory,
    pjrt_buffer_delete: PJRT_Buffer_Delete,
    pjrt_buffer_is_deleted: PJRT_Buffer_IsDeleted,
    pjrt_buffer_copy_to_device: PJRT_Buffer_CopyToDevice,
    pjrt_buffer_to_host_buffer: PJRT_Buffer_ToHostBuffer,
    pjrt_buffer_is_on_cpu: PJRT_Buffer_IsOnCpu,
    pjrt_buffer_ready_event: PJRT_Buffer_ReadyEvent,
    pjrt_buffer_unsafe_pointer: PJRT_Buffer_UnsafePointer,
    pjrt_buffer_increase_external_reference_count: PJRT_Buffer_IncreaseExternalReferenceCount,
    pjrt_buffer_decrease_external_reference_count: PJRT_Buffer_DecreaseExternalReferenceCount,
    pjrt_buffer_opaque_device_memory_data_pointer: PJRT_Buffer_OpaqueDeviceMemoryDataPointer,
    pjrt_copy_to_device_stream_destroy: PJRT_CopyToDeviceStream_Destroy,
    pjrt_copy_to_device_stream_add_chunk: PJRT_CopyToDeviceStream_AddChunk,
    pjrt_copy_to_device_stream_total_bytes: PJRT_CopyToDeviceStream_TotalBytes,
    pjrt_copy_to_device_stream_granule_size: PJRT_CopyToDeviceStream_GranuleSize,
    pjrt_copy_to_device_stream_current_bytes: PJRT_CopyToDeviceStream_CurrentBytes,
    pjrt_topology_description_create: PJRT_TopologyDescription_Create,
    pjrt_topology_description_destroy: PJRT_TopologyDescription_Destroy,
    pjrt_topology_description_platform_name: PJRT_TopologyDescription_PlatformName,
    pjrt_topology_description_platform_version: PJRT_TopologyDescription_PlatformVersion,
    pjrt_topology_description_get_device_descriptions:
        PJRT_TopologyDescription_GetDeviceDescriptions,
    pjrt_topology_description_serialize: PJRT_TopologyDescription_Serialize,
    pjrt_topology_description_attributes: PJRT_TopologyDescription_Attributes,
    pjrt_compile: PJRT_Compile,
    pjrt_executable_output_element_types: PJRT_Executable_OutputElementTypes,
    pjrt_executable_output_dimensions: PJRT_Executable_OutputDimensions,
    pjrt_buffer_copy_to_memory: PJRT_Buffer_CopyToMemory,
    pjrt_client_create_view_of_device_buffer: PJRT_Client_CreateViewOfDeviceBuffer,
    pjrt_executable_fingerprint: PJRT_Executable_Fingerprint,
    pjrt_client_topology_description: PJRT_Client_TopologyDescription,
    pjrt_executable_get_compiled_memory_stats: PJRT_Executable_GetCompiledMemoryStats,
    pjrt_memory_kind_id: PJRT_Memory_Kind_Id,
    pjrt_execute_context_create: PJRT_ExecuteContext_Create,
    pjrt_execute_context_destroy: PJRT_ExecuteContext_Destroy,
    pjrt_buffer_copy_raw_to_host: PJRT_Buffer_CopyRawToHost,
    pjrt_async_transfer_manager_destroy: PJRT_AsyncHostToDeviceTransferManager_Destroy,
    pjrt_async_transfer_manager_transfer_data: PJRT_AsyncHostToDeviceTransferManager_TransferData,
    pjrt_client_create_buffers_for_async_host_to_device:
        PJRT_Client_CreateBuffersForAsyncHostToDevice,
    pjrt_async_transfer_manager_retrieve_buffer:
        PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
    pjrt_async_transfer_manager_device: PJRT_AsyncHostToDeviceTransferManager_Device,
    pjrt_async_transfer_manager_buffer_count: PJRT_AsyncHostToDeviceTransferManager_BufferCount,
    pjrt_async_transfer_manager_buffer_size: PJRT_AsyncHostToDeviceTransferManager_BufferSize,
    pjrt_async_transfer_manager_set_buffer_error:
        PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
    pjrt_async_transfer_manager_add_metadata: PJRT_AsyncHostToDeviceTransferManager_AddMetadata,
    pjrt_client_dma_map: PJRT_Client_DmaMap,
    pjrt_client_dma_unmap: PJRT_Client_DmaUnmap,
    pjrt_client_create_uninitialized_buffer: PJRT_Client_CreateUninitializedBuffer,
    pjrt_client_update_global_process_info: PJRT_Client_UpdateGlobalProcessInfo,
    pjrt_topology_description_deserialize: PJRT_TopologyDescription_Deserialize,
    pjrt_client_create_alias_buffer: PJRT_Client_CreateAliasBuffer,
    pjrt_client_fulfill_alias_buffer: PJRT_Client_FulfillAliasBuffer,
    pjrt_loaded_executable_get_device_assignment: PJRT_LoadedExecutable_GetDeviceAssignment,
    pjrt_client_create_error_buffer: PJRT_Client_CreateErrorBuffer,
    pjrt_async_transfer_manager_transfer_literal:
        PJRT_AsyncHostToDeviceTransferManager_TransferLiteral,
    pjrt_buffer_copy_raw_to_host_future: PJRT_Buffer_CopyRawToHostFuture,
    pjrt_device_poison_execution: PJRT_Device_PoisonExecution,
    pjrt_device_create_async_tracking_event: PJRT_Device_CreateAsyncTrackingEvent,
    pjrt_async_tracking_event_destroy: PJRT_AsyncTrackingEvent_Destroy,
    pjrt_executable_get_compile_options: PJRT_Executable_GetCompileOptions,
    pjrt_buffer_donate_with_control_dependency: PJRT_Buffer_DonateWithControlDependency,
    pjrt_event_create: PJRT_Event_Create,
    pjrt_event_set: PJRT_Event_Set,
    pjrt_device_get_attributes: PJRT_Device_GetAttributes,
    pjrt_client_load: PJRT_Client_Load,
    pjrt_loaded_executable_addressable_device_logical_ids:
        PJRT_LoadedExecutable_AddressableDeviceLogicalIds,
    pjrt_buffer_bitcast: PJRT_Buffer_Bitcast,
    pjrt_error_for_each_payload: PJRT_Error_ForEachPayload,
    pjrt_topology_description_fingerprint: PJRT_TopologyDescription_Fingerprint,
    pjrt_executable_parameter_memory_kinds: PJRT_Executable_ParameterMemoryKinds,
    pjrt_device_clear_memory_stats: PJRT_Device_ClearMemoryStats,
    pjrt_topology_description_make_canonical_shape_for_memory_space:
        PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace,
    pjrt_topology_description_get_memory_space_kind_ids:
        PJRT_TopologyDescription_GetMemorySpaceKindIds,
};

/// Primary PJRT C API entrypoint exported by `libtpu.so` (`GetPjrtApi`).
#[unsafe(no_mangle)]
pub extern "C" fn GetPjrtApi() -> *const PJRT_Api {
    ptr::addr_of!(GLOBAL_PJRT_API)
}

/// Returns the ABI version tuple and core primitive defaults.
#[must_use]
pub const fn abi_version_tuple() -> (i32, i32, DeviceAddress, ByteSize, ChipCoordinates) {
    (
        PJRT_API_MAJOR,
        PJRT_API_MINOR,
        DeviceAddress(0),
        ByteSize(PJRT_API_STRUCT_SIZE),
        ChipCoordinates::new(1, 1, 1),
    )
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_helpers::{
        PJRT_CheckMatchingStructSizes, pjrt_serialized_artifact_deleter,
        pjrt_serialized_noop_deleter, pjrt_slot_dispatch_count, to_c_string,
    };
    use crate::pjrt_types::{
        PJRT_CopyToDeviceStream, PJRT_Error, PJRT_Event, PJRT_ExecuteContext,
        PJRT_Plugin_Initialize_Args, PJRT_TopologyDescription,
    };
    use crate::tpu_driver_types::LibtpuSdk_Client;
    use libtpu_core::driver::LibtpuSdkCoreClient;
    use libtpu_core::pjrt::{PjrtCoreCopyToDeviceStream, PjrtCoreEvent};

    #[test]
    fn test_pjrt_vtable_slots_and_struct_size_validation() {
        let tup = abi_version_tuple();
        assert_eq!(tup.0, 0);
        assert_eq!(tup.1, 116);
        let cstr_with_nul = to_c_string("a\0b");
        assert_eq!(cstr_with_nul.as_bytes(), b"ab");

        let api_ptr = GetPjrtApi();
        assert!(!api_ptr.is_null());
        // SAFETY: `api_ptr` points to `GLOBAL_PJRT_API`.
        let api = unsafe { *api_ptr };
        assert_eq!(api.struct_size, 1144);
        assert_eq!(api.pjrt_api_version.struct_size, 24);
        assert_eq!(api.pjrt_api_version.major_version, 0);
        assert_eq!(api.pjrt_api_version.minor_version, 116);
        assert!(!api.extension_start.is_null());
        assert_ne!(format!("{:?}", api.pjrt_api_version), "");

        let mut count = 0;
        let mut curr = api.extension_start;
        while !curr.is_null() {
            count += 1;
            // SAFETY: Static extension chain.
            let ext = unsafe { *curr };
            assert_ne!(format!("{ext:?}"), "");
            curr = ext.next;
        }
        assert_eq!(count, 17);

        let ev = PJRT_Event {
            inner: PjrtCoreEvent::ready_ok(),
        };
        let ctx = PJRT_ExecuteContext { context_id: 1 };
        let stm = PJRT_CopyToDeviceStream {
            inner: PjrtCoreCopyToDeviceStream::new(256, 128),
        };
        let sdk_cli = LibtpuSdk_Client {
            inner: LibtpuSdkCoreClient::new(),
            host_names: Vec::new(),
            coords_entries: Vec::new(),
            metrics: Vec::new(),
        };
        assert_ne!(format!("{ev:?}{ctx:?}{stm:?}{sdk_cli:?}"), "");

        // SAFETY: Exercising PJRT C API vtable slots with zeroed argument structures and struct size checks.
        unsafe {
            for slot_fn in [
                api.pjrt_plugin_attributes,
                api.pjrt_event_destroy,
                api.pjrt_event_is_ready,
                api.pjrt_event_error,
                api.pjrt_event_await,
                api.pjrt_event_on_ready,
                api.pjrt_client_lookup_addressable_device,
                api.pjrt_client_default_device_assignment,
                api.pjrt_client_buffer_from_host_buffer,
                api.pjrt_device_description_id,
                api.pjrt_device_description_process_index,
                api.pjrt_device_description_attributes,
                api.pjrt_device_description_kind,
                api.pjrt_device_description_debug_string,
                api.pjrt_device_description_to_string,
                api.pjrt_device_get_description,
                api.pjrt_device_is_addressable,
                api.pjrt_device_local_hardware_id,
                api.pjrt_device_addressable_memories,
                api.pjrt_device_default_memory,
                api.pjrt_device_memory_stats,
                api.pjrt_memory_id,
                api.pjrt_memory_kind,
                api.pjrt_memory_debug_string,
                api.pjrt_memory_to_string,
                api.pjrt_memory_addressable_by_devices,
                api.pjrt_executable_destroy,
                api.pjrt_executable_name,
                api.pjrt_executable_num_replicas,
                api.pjrt_executable_num_partitions,
                api.pjrt_executable_num_outputs,
                api.pjrt_executable_size_of_generated_code_in_bytes,
                api.pjrt_executable_get_cost_analysis,
                api.pjrt_executable_output_memory_kinds,
                api.pjrt_executable_optimized_program,
                api.pjrt_executable_serialize,
                api.pjrt_loaded_executable_destroy,
                api.pjrt_loaded_executable_get_executable,
                api.pjrt_loaded_executable_addressable_devices,
                api.pjrt_loaded_executable_delete,
                api.pjrt_loaded_executable_is_deleted,
                api.pjrt_loaded_executable_execute,
                api.pjrt_executable_deserialize_and_load,
                api.pjrt_loaded_executable_fingerprint,
                api.pjrt_buffer_destroy,
                api.pjrt_buffer_element_type,
                api.pjrt_buffer_dimensions,
                api.pjrt_buffer_unpadded_dimensions,
                api.pjrt_buffer_dynamic_dimension_indices,
                api.pjrt_buffer_get_memory_layout,
                api.pjrt_buffer_on_device_size_in_bytes,
                api.pjrt_buffer_device,
                api.pjrt_buffer_memory,
                api.pjrt_buffer_delete,
                api.pjrt_buffer_is_deleted,
                api.pjrt_buffer_copy_to_device,
                api.pjrt_buffer_to_host_buffer,
                api.pjrt_buffer_is_on_cpu,
                api.pjrt_buffer_ready_event,
                api.pjrt_buffer_unsafe_pointer,
                api.pjrt_buffer_increase_external_reference_count,
                api.pjrt_buffer_decrease_external_reference_count,
                api.pjrt_buffer_opaque_device_memory_data_pointer,
                api.pjrt_copy_to_device_stream_destroy,
                api.pjrt_copy_to_device_stream_add_chunk,
                api.pjrt_copy_to_device_stream_total_bytes,
                api.pjrt_copy_to_device_stream_granule_size,
                api.pjrt_copy_to_device_stream_current_bytes,
                api.pjrt_topology_description_create,
                api.pjrt_topology_description_destroy,
                api.pjrt_topology_description_platform_name,
                api.pjrt_topology_description_platform_version,
                api.pjrt_topology_description_get_device_descriptions,
                api.pjrt_topology_description_serialize,
                api.pjrt_topology_description_attributes,
                api.pjrt_compile,
                api.pjrt_executable_output_element_types,
                api.pjrt_executable_output_dimensions,
                api.pjrt_buffer_copy_to_memory,
                api.pjrt_client_create_view_of_device_buffer,
                api.pjrt_executable_fingerprint,
                api.pjrt_client_topology_description,
                api.pjrt_executable_get_compiled_memory_stats,
                api.pjrt_memory_kind_id,
                api.pjrt_execute_context_create,
                api.pjrt_execute_context_destroy,
                api.pjrt_buffer_copy_raw_to_host,
                api.pjrt_async_transfer_manager_destroy,
                api.pjrt_async_transfer_manager_transfer_data,
                api.pjrt_client_create_buffers_for_async_host_to_device,
                api.pjrt_async_transfer_manager_retrieve_buffer,
                api.pjrt_async_transfer_manager_device,
                api.pjrt_async_transfer_manager_buffer_count,
                api.pjrt_async_transfer_manager_buffer_size,
                api.pjrt_async_transfer_manager_set_buffer_error,
                api.pjrt_async_transfer_manager_add_metadata,
                api.pjrt_client_dma_map,
                api.pjrt_client_dma_unmap,
                api.pjrt_client_create_uninitialized_buffer,
                api.pjrt_client_update_global_process_info,
                api.pjrt_topology_description_deserialize,
                api.pjrt_client_create_alias_buffer,
                api.pjrt_client_fulfill_alias_buffer,
                api.pjrt_loaded_executable_get_device_assignment,
                api.pjrt_client_create_error_buffer,
                api.pjrt_async_transfer_manager_transfer_literal,
                api.pjrt_buffer_copy_raw_to_host_future,
                api.pjrt_device_poison_execution,
                api.pjrt_device_create_async_tracking_event,
                api.pjrt_async_tracking_event_destroy,
                api.pjrt_executable_get_compile_options,
                api.pjrt_buffer_donate_with_control_dependency,
                api.pjrt_event_create,
                api.pjrt_event_set,
                api.pjrt_device_get_attributes,
                api.pjrt_client_load,
                api.pjrt_loaded_executable_addressable_device_logical_ids,
                api.pjrt_buffer_bitcast,
                api.pjrt_error_for_each_payload,
                api.pjrt_topology_description_fingerprint,
                api.pjrt_executable_parameter_memory_kinds,
                api.pjrt_device_clear_memory_stats,
                api.pjrt_topology_description_make_canonical_shape_for_memory_space,
                api.pjrt_topology_description_get_memory_space_kind_ids,
            ] {
                let mut dummy_args = [0_usize; 28];
                dummy_args[0] = 224;
                assert!(slot_fn(dummy_args.as_mut_ptr().cast()).is_null());
                if slot_fn as usize == api.pjrt_event_create as usize && dummy_args[2] != 0 {
                    drop(Box::from_raw(
                        dummy_args[2] as *mut crate::pjrt_types::PJRT_Event,
                    ));
                }
                if slot_fn as usize == api.pjrt_execute_context_create as usize
                    && dummy_args[2] != 0
                {
                    drop(Box::from_raw(
                        dummy_args[2] as *mut crate::pjrt_types::PJRT_ExecuteContext,
                    ));
                }
                if slot_fn as usize == api.pjrt_device_create_async_tracking_event as usize
                    && dummy_args[5] != 0
                {
                    drop(Box::from_raw(
                        dummy_args[5] as *mut crate::pjrt_types::PJRT_AsyncTrackingEvent,
                    ));
                }
                if slot_fn as usize == api.pjrt_topology_description_create as usize
                    || slot_fn as usize == api.pjrt_topology_description_deserialize as usize
                {
                    let topo_val =
                        if slot_fn as usize == api.pjrt_topology_description_create as usize {
                            dummy_args[6]
                        } else {
                            dummy_args[5]
                        };
                    if topo_val != 0 {
                        let mut del_tp = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                            struct_size: size_of::<
                                crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args,
                            >(),
                            extension_start: ptr::null_mut(),
                            topology: topo_val as *mut PJRT_TopologyDescription,
                        };
                        let _ = PJRT_TopologyDescription_Destroy(ptr::addr_of_mut!(del_tp).cast());
                    }
                }
                if (slot_fn as usize == api.pjrt_executable_serialize as usize
                    || slot_fn as usize == api.pjrt_topology_description_serialize as usize
                    || slot_fn as usize
                        == api.pjrt_loaded_executable_get_device_assignment as usize
                    || slot_fn as usize == api.pjrt_executable_get_compile_options as usize)
                    && dummy_args[5] != 0
                {
                    pjrt_serialized_artifact_deleter(dummy_args[5] as *mut std::ffi::c_void);
                }
            }

            let mut small_args = [8_usize; 16];
            for strict_fn in [
                PJRT_Client_BufferFromHostBuffer
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_LoadedExecutable_Execute,
                PJRT_Buffer_ToHostBuffer,
                PJRT_Buffer_Dimensions,
                PJRT_Buffer_ElementType,
                PJRT_Buffer_OnDeviceSizeInBytes,
                PJRT_Buffer_Destroy,
                PJRT_LoadedExecutable_Destroy,
            ] {
                drop(Box::from_raw(strict_fn(small_args.as_mut_ptr().cast())));
            }

            let mut zero_sz: usize = 0;
            let e_chk = PJRT_CheckMatchingStructSizes(ptr::null_mut());
            drop(Box::from_raw(e_chk));
            let e_chk_sz = PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(zero_sz).cast());
            drop(Box::from_raw(e_chk_sz));
            let mut init_args = PJRT_Plugin_Initialize_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
            };
            assert!(PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(init_args).cast()).is_null());
            pjrt_serialized_noop_deleter(ptr::null_mut());
            assert!(pjrt_slot_dispatch_count() > 0);
        }
    }
}
