//! `PJRT_Api` static vtable assembly and `GetPjrtApi` entrypoint.

use crate::extensions::EXT_01_FFI;
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
    extension_start: ptr::addr_of!(EXT_01_FFI.base),
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

/// Per-minor-version negotiated `PJRT_Api` cache bounded to minor versions `-1..=256`.
static NEGOTIATED_PJRT_API_CACHE: std::sync::OnceLock<
    parking_lot::Mutex<std::collections::BTreeMap<i32, &'static PJRT_Api>>,
> = std::sync::OnceLock::new();

/// Returns the process-wide negotiated [`PJRT_Api`] cache.
fn negotiated_pjrt_api_cache()
-> &'static parking_lot::Mutex<std::collections::BTreeMap<i32, &'static PJRT_Api>> {
    NEGOTIATED_PJRT_API_CACHE
        .get_or_init(|| parking_lot::Mutex::new(std::collections::BTreeMap::new()))
}

/// Constructs a `PJRT_Api` vtable tailored to `minor` version negotiation.
///
/// # Arguments
///
/// * `minor` - Negotiated PJRT minor API version requested by the host framework.
#[must_use]
pub(crate) fn build_negotiated_pjrt_api(minor: i32) -> PJRT_Api {
    let mut copy = GLOBAL_PJRT_API;
    copy.pjrt_api_version.minor_version = minor;
    if minor < PJRT_API_MINOR {
        copy.extension_start = ptr::addr_of!(crate::extensions::NEGOTIATED_EXT_PROFILER).cast();
    }
    copy
}

/// Resolves the `PJRT_Api` vtable pointer for an optional `LIBTPU_PJRT_API_MINOR_VERSION` string,
/// caching distinct negotiated minor versions within `-1..=256`.
///
/// # Arguments
///
/// * `minor_env` - Optional string value of `LIBTPU_PJRT_API_MINOR_VERSION`.
#[must_use]
pub(crate) fn resolve_pjrt_api_for_env(minor_env: Option<&str>) -> *const PJRT_Api {
    if let Some(parsed) = minor_env.and_then(|raw| raw.parse::<i32>().ok()) {
        let minor = parsed.clamp(-1, 256);
        if minor != PJRT_API_MINOR {
            let mut cache = negotiated_pjrt_api_cache().lock();
            let entry: &'static PJRT_Api = cache
                .entry(minor)
                .or_insert_with(|| Box::leak(Box::new(build_negotiated_pjrt_api(minor))));
            return ptr::from_ref(entry);
        }
    }
    ptr::addr_of!(GLOBAL_PJRT_API)
}

/// Primary PJRT C API entrypoint exported by `libtpu.so` (`GetPjrtApi`).
#[unsafe(no_mangle)]
pub extern "C" fn GetPjrtApi() -> *const PJRT_Api {
    crate::pjrt_helpers::ffi_catch_unwind(std::ptr::null(), || {
        resolve_pjrt_api_for_env(
            std::env::var("LIBTPU_PJRT_API_MINOR_VERSION")
                .ok()
                .as_deref(),
        )
    })
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

    /// No-op deletion callback used when testing `PJRT_Client_CreateViewOfDeviceBuffer`.
    ///
    /// # Arguments
    ///
    /// * `_ctx` - Opaque callback context pointer.
    ///
    /// # Safety
    ///
    /// Safe for any `_ctx` pointer value.
    unsafe extern "C" fn view_del_cb(_ctx: *mut std::ffi::c_void) {}

    #[test]
    fn test_pjrt_vtable_slots_and_struct_size_validation() {
        let tup = abi_version_tuple();
        assert_eq!(tup.0, 0);
        assert_eq!(tup.1, 117);
        assert_eq!(
            crate::pjrt_types::select_min_struct_size(None, 120, 96),
            120
        );
        assert_eq!(
            crate::pjrt_types::select_min_struct_size(Some("bad"), 120, 96),
            120
        );
        assert_eq!(
            crate::pjrt_types::select_min_struct_size(Some("117"), 120, 96),
            120
        );
        assert_eq!(
            crate::pjrt_types::select_min_struct_size(Some("54"), 120, 96),
            96
        );
        assert_eq!(
            build_negotiated_pjrt_api(54).pjrt_api_version.minor_version,
            54
        );
        assert_eq!(
            build_negotiated_pjrt_api(120)
                .pjrt_api_version
                .minor_version,
            120
        );
        assert_eq!(
            resolve_pjrt_api_for_env(None),
            ptr::addr_of!(GLOBAL_PJRT_API)
        );
        assert_eq!(
            resolve_pjrt_api_for_env(Some("invalid")),
            ptr::addr_of!(GLOBAL_PJRT_API)
        );
        assert_eq!(
            resolve_pjrt_api_for_env(Some("117")),
            ptr::addr_of!(GLOBAL_PJRT_API)
        );
        let neg_ptr_54 = resolve_pjrt_api_for_env(Some("54"));
        let neg_ptr_89 = resolve_pjrt_api_for_env(Some("89"));
        let neg_ptr_54_again = resolve_pjrt_api_for_env(Some("54"));
        let neg_ptr_clamped = resolve_pjrt_api_for_env(Some("999"));
        assert!(!neg_ptr_54.is_null());
        assert!(!neg_ptr_89.is_null());
        assert_eq!(neg_ptr_54, neg_ptr_54_again);
        assert_ne!(neg_ptr_54, neg_ptr_89);
        // SAFETY: `neg_ptr_54`, `neg_ptr_89`, and `neg_ptr_clamped` point to leaked `'static PJRT_Api` instances.
        unsafe {
            assert_eq!((*neg_ptr_54).pjrt_api_version.minor_version, 54);
            assert_eq!((*neg_ptr_89).pjrt_api_version.minor_version, 89);
            assert_eq!((*neg_ptr_clamped).pjrt_api_version.minor_version, 256);
        }
        let cstr_with_nul = to_c_string("a\0b");
        assert_eq!(cstr_with_nul.as_bytes(), b"ab");

        let api_ptr = GetPjrtApi();
        assert!(!api_ptr.is_null());
        // SAFETY: `api_ptr` points to `GLOBAL_PJRT_API`.
        let api = unsafe { *api_ptr };
        assert_eq!(api.struct_size, 1144);
        assert_eq!(api.pjrt_api_version.struct_size, 24);
        assert_eq!(api.pjrt_api_version.major_version, 0);
        assert_eq!(api.pjrt_api_version.minor_version, 117);
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
        assert_eq!(count, 18);

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
            metric_names: Vec::new(),
            metrics: Vec::new(),
            c_debug_callbacks: parking_lot::Mutex::new(std::collections::BTreeMap::new()),
        };
        assert_ne!(format!("{ev:?}{ctx:?}{stm:?}{sdk_cli:?}"), "");

        // SAFETY: Layer 1 (Negative / Error Contract Tests) & Layer 2 (Positive / Real-Handle Lifecycle Tests).
        unsafe {
            let assert_pjrt_err_code = |err_ptr: *mut PJRT_Error, expected_code: i32| {
                assert!(!err_ptr.is_null());
                let mut code_args = crate::pjrt_types::PJRT_Error_GetCode_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Error_GetCode_Args>(),
                    extension_start: ptr::null_mut(),
                    error: err_ptr,
                    code: 0,
                };
                assert!((api.pjrt_error_get_code)(ptr::addr_of_mut!(code_args).cast()).is_null());
                assert_eq!(code_args.code, expected_code);
                let mut msg_args = crate::pjrt_types::PJRT_Error_Message_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Error_Message_Args>(),
                    extension_start: ptr::null_mut(),
                    error: err_ptr,
                    message: ptr::null(),
                    message_size: 0,
                };
                (api.pjrt_error_message)(ptr::addr_of_mut!(msg_args).cast());
                assert!(!msg_args.message.is_null() && msg_args.message_size > 0);
                let mut del_err = crate::pjrt_types::PJRT_Error_Destroy_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Error_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    error: err_ptr,
                };
                (api.pjrt_error_destroy)(ptr::addr_of_mut!(del_err).cast());
            };

            // Layer 1: Null args, undersized struct_size, null handles, and out-of-range device IDs
            assert_pjrt_err_code((api.pjrt_client_create)(ptr::null_mut()), 3);
            let mut undersized = [8_usize; 16];
            for strict_fn in [
                api.pjrt_client_buffer_from_host_buffer,
                api.pjrt_loaded_executable_execute,
                api.pjrt_buffer_to_host_buffer,
                api.pjrt_buffer_dimensions,
                api.pjrt_buffer_element_type,
                api.pjrt_buffer_on_device_size_in_bytes,
                api.pjrt_buffer_destroy,
                api.pjrt_loaded_executable_destroy,
            ] {
                assert_pjrt_err_code(strict_fn(undersized.as_mut_ptr().cast()), 3);
            }
            let mut null_unsafe_ptr = crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                buffer_pointer: 0,
            };
            assert_pjrt_err_code(
                (api.pjrt_buffer_unsafe_pointer)(ptr::addr_of_mut!(null_unsafe_ptr).cast()),
                3,
            );
            let mut null_opaque_ptr =
                crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    buffer: ptr::null_mut(),
                    device_memory_ptr: ptr::null_mut(),
                };
            assert_pjrt_err_code(
                (api.pjrt_buffer_opaque_device_memory_data_pointer)(
                    ptr::addr_of_mut!(null_opaque_ptr).cast(),
                ),
                3,
            );
            let mut null_dma = crate::pjrt_types::PJRT_Client_DmaMap_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_DmaMap_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                data: ptr::null(),
                size: 4096,
            };
            assert_pjrt_err_code(
                (api.pjrt_client_dma_map)(ptr::addr_of_mut!(null_dma).cast()),
                3,
            );

            let mut zero_sz: usize = 0;
            assert_pjrt_err_code(PJRT_CheckMatchingStructSizes(ptr::null_mut()), 3);
            assert_pjrt_err_code(
                PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(zero_sz).cast()),
                3,
            );
            let mut init_args = PJRT_Plugin_Initialize_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
            };
            assert!((api.pjrt_plugin_initialize)(ptr::addr_of_mut!(init_args).cast()).is_null());
            assert!(PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(init_args).cast()).is_null());
            pjrt_serialized_noop_deleter(ptr::null_mut());
            assert!(pjrt_slot_dispatch_count() > 0);

            // Layer 2: Real-handle TopologyDescription, Client, Device, Memory, Buffer, Event, Executable & Execute
            let mut topo_create = crate::pjrt_types::PJRT_TopologyDescription_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(),
                extension_start: ptr::null_mut(),
                topology_name: c"tpu".as_ptr(),
                topology_name_size: 3,
                create_options: ptr::null(),
                num_options: 0,
                topology: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_topology_description_create)(ptr::addr_of_mut!(topo_create).cast())
                    .is_null()
            );
            let topo_handle: *mut PJRT_TopologyDescription = topo_create.topology;
            assert!(!topo_handle.is_null());
            let mut topo_ser = crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args>(
                ),
                extension_start: ptr::null_mut(),
                topology: topo_handle,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_topology: ptr::null_mut(),
                serialized_topology_deleter: None,
            };
            assert!(
                (api.pjrt_topology_description_serialize)(ptr::addr_of_mut!(topo_ser).cast())
                    .is_null()
            );
            assert!(topo_ser.serialized_bytes_size > 0);
            pjrt_serialized_artifact_deleter(topo_ser.serialized_topology.cast());
            let mut topo_del = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                topology: topo_handle,
            };
            assert!(
                (api.pjrt_topology_description_destroy)(ptr::addr_of_mut!(topo_del).cast())
                    .is_null()
            );

            let mut cli_create = crate::pjrt_types::PJRT_Client_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Create_Args>(),
                extension_start: ptr::null_mut(),
                create_options: ptr::null(),
                num_options: 0,
                kv_get_callback: ptr::null(),
                kv_get_user_arg: ptr::null_mut(),
                kv_put_callback: None,
                kv_put_user_arg: ptr::null_mut(),
                kv_try_get_callback: ptr::null(),
                kv_try_get_user_arg: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            assert!((api.pjrt_client_create)(ptr::addr_of_mut!(cli_create).cast()).is_null());
            let client = cli_create.client;
            assert!(!client.is_null());

            // Out-of-range device lookup returns NotFound (5)
            let mut bad_lookup = crate::pjrt_types::PJRT_Client_LookupDevice_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_LookupDevice_Args>(),
                extension_start: ptr::null_mut(),
                client,
                id: 9999,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert_pjrt_err_code(
                (api.pjrt_client_lookup_device)(ptr::addr_of_mut!(bad_lookup).cast()),
                5,
            );

            let mut cli_devs = crate::pjrt_types::PJRT_Client_Devices_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Devices_Args>(),
                extension_start: ptr::null_mut(),
                client,
                devices: ptr::null(),
                num_devices: 0,
            };
            assert!((api.pjrt_client_devices)(ptr::addr_of_mut!(cli_devs).cast()).is_null());
            assert!(cli_devs.num_devices >= 1);
            let dev0 = *cli_devs.devices;

            let mut dev_mem = crate::pjrt_types::PJRT_Device_DefaultMemory_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Device_DefaultMemory_Args>(),
                extension_start: ptr::null_mut(),
                device: dev0,
                memory: ptr::null_mut(),
            };
            assert!((api.pjrt_device_default_memory)(ptr::addr_of_mut!(dev_mem).cast()).is_null());
            assert!(!dev_mem.memory.is_null());

            // Real Event & ExecuteContext lifecycle
            let mut ev_create = crate::pjrt_types::PJRT_Event_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Create_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!((api.pjrt_event_create)(ptr::addr_of_mut!(ev_create).cast()).is_null());
            let mut ev_set = crate::pjrt_types::PJRT_Event_Set_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Set_Args>(),
                extension_start: ptr::null_mut(),
                event: ev_create.event,
                error_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
            };
            assert!((api.pjrt_event_set)(ptr::addr_of_mut!(ev_set).cast()).is_null());
            let mut ev_ready = crate::pjrt_types::PJRT_Event_IsReady_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_IsReady_Args>(),
                extension_start: ptr::null_mut(),
                event: ev_create.event,
                is_ready: false,
                pad: [0; 7],
            };
            assert!((api.pjrt_event_is_ready)(ptr::addr_of_mut!(ev_ready).cast()).is_null());
            assert!(ev_ready.is_ready);
            let mut ev_del0 = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ev_create.event,
            };
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del0).cast()).is_null());

            let mut ctx_create = crate::pjrt_types::PJRT_ExecuteContext_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_ExecuteContext_Create_Args>(),
                extension_start: ptr::null_mut(),
                context: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_execute_context_create)(ptr::addr_of_mut!(ctx_create).cast()).is_null()
            );
            let mut ctx_del = crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                context: ctx_create.context,
            };
            assert!(
                (api.pjrt_execute_context_destroy)(ptr::addr_of_mut!(ctx_del).cast()).is_null()
            );

            // Real H2D -> Compile -> Execute -> D2H numerical check through vtable slots
            let lhs_vals = [10.0_f32, 20.0, 30.0, 40.0];
            let rhs_vals = [1.0_f32, 2.0, 3.0, 4.0];
            let dims = [4_i64];
            let make_buf = |vals: &[f32; 4]| -> *mut crate::pjrt_types::PJRT_Buffer {
                let mut h2d = crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    client,
                    data: vals.as_ptr().cast(),
                    element_type: 11,
                    pad0: 0,
                    dims: dims.as_ptr(),
                    num_dims: 1,
                    byte_strides: ptr::null(),
                    num_byte_strides: 0,
                    host_buffer_semantics: 0,
                    pad1: 0,
                    device: dev0,
                    memory: dev_mem.memory,
                    device_layout: ptr::null(),
                    done_with_host_buffer: ptr::null_mut(),
                    out_buffer: ptr::null_mut(),
                };
                assert!(
                    (api.pjrt_client_buffer_from_host_buffer)(ptr::addr_of_mut!(h2d).cast())
                        .is_null()
                );
                let mut ev_del = crate::pjrt_types::PJRT_Event_Destroy_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    event: h2d.done_with_host_buffer,
                };
                assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del).cast()).is_null());
                h2d.out_buffer
            };
            let buf_lhs = make_buf(&lhs_vals);
            let buf_rhs = make_buf(&rhs_vals);

            let mut uptr_args = crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_lhs,
                buffer_pointer: 0,
            };
            assert!(
                (api.pjrt_buffer_unsafe_pointer)(ptr::addr_of_mut!(uptr_args).cast()).is_null()
            );
            assert!(uptr_args.buffer_pointer > 0);

            let mlir_src = b"module @vtable_add { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }";
            let mut prog = crate::pjrt_types::PJRT_Program {
                struct_size: size_of::<crate::pjrt_types::PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: mlir_src.as_ptr().cast(),
                code_size: mlir_src.len(),
                format: c"mlir".as_ptr(),
                format_size: 4,
            };
            let mut comp_args = crate::pjrt_types::PJRT_Client_Compile_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Compile_Args>(),
                extension_start: ptr::null_mut(),
                client,
                program: ptr::addr_of_mut!(prog),
                compile_options: ptr::null(),
                compile_options_size: 0,
                executable: ptr::null_mut(),
            };
            assert!((api.pjrt_client_compile)(ptr::addr_of_mut!(comp_args).cast()).is_null());
            let loaded_exec = comp_args.executable;
            assert!(!loaded_exec.is_null());

            let arg_row: [*const crate::pjrt_types::PJRT_Buffer; 2] =
                [buf_lhs.cast_const(), buf_rhs.cast_const()];
            let arg_lists = [arg_row.as_ptr()];
            let mut out_buf: *mut crate::pjrt_types::PJRT_Buffer = ptr::null_mut();
            let mut out_lists = [ptr::addr_of_mut!(out_buf)];
            let mut exec_opts = crate::pjrt_types::PJRT_ExecuteOptions {
                struct_size: size_of::<crate::pjrt_types::PJRT_ExecuteOptions>(),
                extension_start: ptr::null_mut(),
                send_callbacks: ptr::null_mut(),
                recv_callbacks: ptr::null_mut(),
                num_send_ops: 0,
                num_recv_ops: 0,
                launch_id: 1,
                pad: 0,
                non_donatable_input_indices: ptr::null(),
                num_non_donatable_input_indices: 0,
                context: ptr::null_mut(),
            };
            let mut exec_args = crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args>(),
                extension_start: ptr::null_mut(),
                executable: loaded_exec,
                options: ptr::addr_of_mut!(exec_opts),
                argument_lists: arg_lists.as_ptr(),
                num_devices: 1,
                num_args: 2,
                output_lists: out_lists.as_mut_ptr(),
                device_complete_events: ptr::null_mut(),
                execute_device: dev0,
            };
            assert!(
                (api.pjrt_loaded_executable_execute)(ptr::addr_of_mut!(exec_args).cast()).is_null()
            );
            assert!(!out_buf.is_null());

            let mut host_out = [0.0_f32; 4];
            let mut d2h = crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                src: out_buf,
                host_layout: ptr::null_mut(),
                dst: host_out.as_mut_ptr().cast(),
                dst_size: size_of_val(&host_out),
                event: ptr::null_mut(),
            };
            assert!((api.pjrt_buffer_to_host_buffer)(ptr::addr_of_mut!(d2h).cast()).is_null());
            let mut ev_del_d2h = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: d2h.event,
            };
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del_d2h).cast()).is_null());
            assert_eq!(host_out, [11.0_f32, 22.0, 33.0, 44.0]);

            // Exercise remaining Layer 2 PJRT_Executable / PJRT_LoadedExecutable / PJRT_Client / Topology methods
            let mut get_ex = crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args>(
                ),
                extension_start: ptr::null_mut(),
                loaded_executable: loaded_exec,
                executable: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_loaded_executable_get_executable)(ptr::addr_of_mut!(get_ex).cast())
                    .is_null()
            );
            let exec_ptr = get_ex.executable;

            let mut ex_co = crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec_ptr,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_compile_options: ptr::null_mut(),
                serialized_compile_options_deleter: None,
            };
            assert!(
                (api.pjrt_executable_get_compile_options)(ptr::addr_of_mut!(ex_co).cast())
                    .is_null()
            );
            pjrt_serialized_artifact_deleter(ex_co.serialized_compile_options);
            ex_co.executable = ptr::null_mut();
            assert!(
                (api.pjrt_executable_get_compile_options)(ptr::addr_of_mut!(ex_co).cast())
                    .is_null()
            );
            pjrt_serialized_artifact_deleter(ex_co.serialized_compile_options);

            let mut ex_pmk = crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args>(
                ),
                extension_start: ptr::null_mut(),
                executable: exec_ptr,
                num_parameters: 0,
                memory_kinds: ptr::null(),
                memory_kind_sizes: ptr::null(),
            };
            assert!(
                (api.pjrt_executable_parameter_memory_kinds)(ptr::addr_of_mut!(ex_pmk).cast())
                    .is_null()
            );
            ex_pmk.executable = ptr::null_mut();
            assert!(
                (api.pjrt_executable_parameter_memory_kinds)(ptr::addr_of_mut!(ex_pmk).cast())
                    .is_null()
            );

            let mut lex_fp = crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args>(),
                extension_start: ptr::null_mut(),
                executable: loaded_exec,
                executable_fingerprint: ptr::null(),
                executable_fingerprint_size: 0,
            };
            assert!(
                (api.pjrt_loaded_executable_fingerprint)(ptr::addr_of_mut!(lex_fp).cast())
                    .is_null()
            );
            lex_fp.executable = ptr::null_mut();
            assert!(
                (api.pjrt_loaded_executable_fingerprint)(ptr::addr_of_mut!(lex_fp).cast())
                    .is_null()
            );

            let mut lex_lids =
                crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    executable: loaded_exec,
                    addressable_device_logical_ids: ptr::null(),
                    num_addressable_device_logical_ids: 0,
                };
            assert!(
                (api.pjrt_loaded_executable_addressable_device_logical_ids)(
                    ptr::addr_of_mut!(lex_lids).cast()
                )
                .is_null()
            );
            lex_lids.executable = ptr::null_mut();
            assert!(
                (api.pjrt_loaded_executable_addressable_device_logical_ids)(
                    ptr::addr_of_mut!(lex_lids).cast()
                )
                .is_null()
            );

            let mut lex_da = crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: ptr::null_mut(),
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_device_assignment: ptr::null_mut(),
                serialized_device_assignment_deleter: None,
            };
            assert!(
                (api.pjrt_loaded_executable_get_device_assignment)(
                    ptr::addr_of_mut!(lex_da).cast()
                )
                .is_null()
            );
            pjrt_serialized_artifact_deleter(lex_da.serialized_device_assignment);

            let mut ex_del = crate::pjrt_types::PJRT_Executable_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec_ptr,
            };
            assert!((api.pjrt_executable_destroy)(ptr::addr_of_mut!(ex_del).cast()).is_null());

            // DMA Map / Unmap error & success branches
            let mut dma_null = crate::pjrt_types::PJRT_Client_DmaMap_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_DmaMap_Args>(),
                extension_start: ptr::null_mut(),
                client,
                data: ptr::null(),
                size: 4096,
            };
            assert_pjrt_err_code(
                (api.pjrt_client_dma_map)(ptr::addr_of_mut!(dma_null).cast()),
                3,
            );
            let mut dma_unmap_null = crate::pjrt_types::PJRT_Client_DmaUnmap_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_DmaUnmap_Args>(),
                extension_start: ptr::null_mut(),
                client,
                data: ptr::null(),
            };
            assert_pjrt_err_code(
                (api.pjrt_client_dma_unmap)(ptr::addr_of_mut!(dma_unmap_null).cast()),
                3,
            );
            let mut dma_unmap_null_cli = crate::pjrt_types::PJRT_Client_DmaUnmap_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_DmaUnmap_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                data: host_out.as_ptr().cast(),
            };
            assert_pjrt_err_code(
                (api.pjrt_client_dma_unmap)(ptr::addr_of_mut!(dma_unmap_null_cli).cast()),
                3,
            );

            // ErrorBuffer with null message & CreateUninitializedBuffer
            let mut err_buf_null_msg = crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client,
                error_code: 3,
                reserved: 0,
                error_message: ptr::null(),
                error_message_size: 0,
                dims: dims.as_ptr(),
                num_dims: 1,
                element_type: 11,
                reserved1: 0,
                layout: ptr::null(),
                device: dev0,
                memory: ptr::null_mut(),
                buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_create_error_buffer)(ptr::addr_of_mut!(err_buf_null_msg).cast())
                    .is_null()
            );
            let mut uninit_args = crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args>(
                ),
                extension_start: ptr::null_mut(),
                client,
                dims: dims.as_ptr(),
                num_dims: 1,
                element_type: 11,
                pad: 0,
                layout: ptr::null(),
                device: dev0,
                memory: ptr::null_mut(),
                buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_create_uninitialized_buffer)(
                    ptr::addr_of_mut!(uninit_args).cast()
                )
                .is_null()
            );

            for b in [
                buf_lhs,
                buf_rhs,
                out_buf,
                err_buf_null_msg.buffer,
                uninit_args.buffer,
            ] {
                let mut buf_del = crate::pjrt_types::PJRT_Buffer_Destroy_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    buffer: b,
                };
                assert!((api.pjrt_buffer_destroy)(ptr::addr_of_mut!(buf_del).cast()).is_null());
            }

            // TopologyDescription Deserialize / MakeCanonicalShapeForMemorySpace / GetMemorySpaceKindIds + null-handle contracts
            let mut topo_ser = crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args>(
                ),
                extension_start: ptr::null_mut(),
                topology: ptr::null(),
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_topology: ptr::null_mut(),
                serialized_topology_deleter: None,
            };
            assert!(
                (api.pjrt_topology_description_serialize)(ptr::addr_of_mut!(topo_ser).cast())
                    .is_null()
            );
            let mut topo_deser = crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args>(
                ),
                extension_start: ptr::null_mut(),
                serialized_topology: topo_ser.serialized_bytes,
                serialized_topology_size: topo_ser.serialized_bytes_size,
                topology: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_topology_description_deserialize)(ptr::addr_of_mut!(topo_deser).cast())
                    .is_null()
            );
            assert!(!topo_deser.topology.is_null());
            pjrt_serialized_artifact_deleter(topo_ser.serialized_topology.cast());
            topo_deser.serialized_topology = ptr::null();
            topo_deser.serialized_topology_size = 0;
            let mut topo_deser_empty = topo_deser;
            topo_deser_empty.topology = ptr::null_mut();
            assert!(
                (api.pjrt_topology_description_deserialize)(
                    ptr::addr_of_mut!(topo_deser_empty).cast()
                )
                .is_null()
            );

            let mut canon_layout = crate::pjrt_types::PJRT_Buffer_MemoryLayout {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_MemoryLayout>(),
                extension_start: ptr::null_mut(),
                minor_to_major: ptr::null(),
                minor_to_major_size: 0,
                tile_dims: ptr::null(),
                tile_dim_sizes: ptr::null(),
            };
            let mut mcs = crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args>(),
                extension_start: ptr::null_mut(),
                topology: topo_deser.topology,
                dims: dims.as_ptr(),
                num_dims: 1,
                element_type: 11,
                memory_space_kind: 0,
                layout: ptr::addr_of_mut!(canon_layout),
                reserved: 0,
                reserved1: 0,
                reserved2: 0,
                reserved3: 0,
            };
            assert!(
                (api.pjrt_topology_description_make_canonical_shape_for_memory_space)(
                    ptr::addr_of_mut!(mcs).cast()
                )
                .is_null()
            );
            mcs.layout = ptr::null_mut();
            assert!(
                (api.pjrt_topology_description_make_canonical_shape_for_memory_space)(
                    ptr::addr_of_mut!(mcs).cast()
                )
                .is_null()
            );

            let mut gmk = crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args,
                >(),
                extension_start: ptr::null_mut(),
                topology: topo_deser.topology,
                kind_ids: ptr::null(),
                num_kind_ids: 0,
            };
            assert!(
                (api.pjrt_topology_description_get_memory_space_kind_ids)(
                    ptr::addr_of_mut!(gmk).cast()
                )
                .is_null()
            );
            assert!(gmk.num_kind_ids > 0);

            for t_del_p in [
                topo_deser.topology,
                topo_deser_empty.topology,
                ptr::null_mut(),
            ] {
                let mut td = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    topology: t_del_p,
                };
                assert!(
                    (api.pjrt_topology_description_destroy)(ptr::addr_of_mut!(td).cast()).is_null()
                );
            }

            // CreateViewOfDeviceBuffer with real device & callback + null client
            let mut view_args = crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args>(
                ),
                extension_start: ptr::null_mut(),
                client,
                device_buffer_ptr: host_out.as_ptr().cast(),
                dims: dims.as_ptr(),
                num_dims: 1,
                element_type: 11,
                pad: 0,
                layout: ptr::null(),
                device: dev0,
                on_delete_callback: Some(view_del_cb),
                buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_create_view_of_device_buffer)(ptr::addr_of_mut!(view_args).cast())
                    .is_null()
            );
            assert!(!view_args.buffer.is_null());
            let mut v_del = crate::pjrt_types::PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: view_args.buffer,
            };
            assert!((api.pjrt_buffer_destroy)(ptr::addr_of_mut!(v_del).cast()).is_null());
            view_args.client = ptr::null_mut();
            assert!(
                (api.pjrt_client_create_view_of_device_buffer)(ptr::addr_of_mut!(view_args).cast())
                    .is_null()
            );

            // Layer 1 typed null-handle contract checks on remaining PJRT vtable entrypoints
            let mut null_lkp = crate::pjrt_types::PJRT_Client_LookupDevice_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_LookupDevice_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                id: 0,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_lookup_addressable_device)(ptr::addr_of_mut!(null_lkp).cast())
                    .is_null()
            );
            let mut null_default_assign =
                crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    client: ptr::null_mut(),
                    num_replicas: 1,
                    num_partitions: 1,
                    default_assignment_size: 1,
                    default_assignment: ptr::null_mut(),
                };
            assert!(
                (api.pjrt_client_default_device_assignment)(
                    ptr::addr_of_mut!(null_default_assign).cast()
                )
                .is_null()
            );
            let mut null_ugp = crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args>(
                ),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                processes: ptr::null_mut(),
                num_processes: 0,
            };
            assert!(
                (api.pjrt_client_update_global_process_info)(ptr::addr_of_mut!(null_ugp).cast())
                    .is_null()
            );
            let mut null_load = crate::pjrt_types::PJRT_Client_Load_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Load_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                executable: ptr::null_mut(),
                compile_options: ptr::null(),
                compile_options_size: 0,
                loaded_executable: ptr::null_mut(),
            };
            assert!((api.pjrt_client_load)(ptr::addr_of_mut!(null_load).cast()).is_null());
            let mut null_ectx = crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                context: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_execute_context_destroy)(ptr::addr_of_mut!(null_ectx).cast()).is_null()
            );
            let mut null_b_d2h = crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                src: ptr::null_mut(),
                host_layout: ptr::null_mut(),
                dst: ptr::null_mut(),
                dst_size: 0,
                event: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_buffer_to_host_buffer)(ptr::addr_of_mut!(null_b_d2h).cast()).is_null()
            );
            let mut null_b_dims = crate::pjrt_types::PJRT_Buffer_Dimensions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                dims: ptr::null(),
                num_dims: 0,
            };
            assert!((api.pjrt_buffer_dimensions)(ptr::addr_of_mut!(null_b_dims).cast()).is_null());
            let mut null_b_et = crate::pjrt_types::PJRT_Buffer_ElementType_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                element_type: 0,
                pad: 0,
            };
            assert!((api.pjrt_buffer_element_type)(ptr::addr_of_mut!(null_b_et).cast()).is_null());
            let mut null_b_sz = crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                on_device_size_in_bytes: 0,
            };
            assert!(
                (api.pjrt_buffer_on_device_size_in_bytes)(ptr::addr_of_mut!(null_b_sz).cast())
                    .is_null()
            );
            let mut null_b_c2m = crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                dst_memory: ptr::null_mut(),
                dst_buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_buffer_copy_to_memory)(ptr::addr_of_mut!(null_b_c2m).cast()).is_null()
            );
            let mut null_b_cr2h = crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                dst: ptr::null_mut(),
                offset: 0,
                transfer_size: 0,
                event: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_buffer_copy_raw_to_host)(ptr::addr_of_mut!(null_b_cr2h).cast()).is_null()
            );
            drop(Box::from_raw(null_b_cr2h.event));
            let mut null_b_cr2hf = crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                offset: 0,
                transfer_size: 0,
                event: ptr::null_mut(),
                callback_data: ptr::null_mut(),
                future_ready_callback: ptr::null(),
            };
            assert!(
                (api.pjrt_buffer_copy_raw_to_host_future)(ptr::addr_of_mut!(null_b_cr2hf).cast())
                    .is_null()
            );
            drop(Box::from_raw(null_b_cr2hf.event));
            let mut null_b_don = crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                callback_data: ptr::null_mut(),
                dependency_callback: ptr::null(),
                out_buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_buffer_donate_with_control_dependency)(
                    ptr::addr_of_mut!(null_b_don).cast()
                )
                .is_null()
            );
            let mut null_b_bc = crate::pjrt_types::PJRT_Buffer_Bitcast_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Bitcast_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                element_type: 11,
                pad: 0,
                dims: ptr::null(),
                num_dims: 0,
                device_layout: ptr::null(),
                out_buffer: ptr::null_mut(),
            };
            assert!((api.pjrt_buffer_bitcast)(ptr::addr_of_mut!(null_b_bc).cast()).is_null());
            let mut lex_del = crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: loaded_exec,
            };
            assert!(
                (api.pjrt_loaded_executable_destroy)(ptr::addr_of_mut!(lex_del).cast()).is_null()
            );
            let mut cli_del = crate::pjrt_types::PJRT_Client_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                client,
            };
            assert!((api.pjrt_client_destroy)(ptr::addr_of_mut!(cli_del).cast()).is_null());
        }
    }

    #[test]
    fn test_pjrt_core_null_and_edge_contracts() {
        // SAFETY: Testing PJRT C-API vtable entrypoints with valid `#[repr(C)]` argument structs.
        unsafe {
            let api = &*GetPjrtApi();

            let mut pattr = crate::pjrt_types::PJRT_Plugin_Attributes_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Plugin_Attributes_Args>(),
                extension_start: ptr::null_mut(),
                attributes: ptr::null(),
                num_attributes: 0,
            };
            assert!((api.pjrt_plugin_attributes)(ptr::addr_of_mut!(pattr).cast()).is_null());
            assert!(pattr.num_attributes > 0);

            let mut dkind = crate::pjrt_types::PJRT_DeviceDescription_Kind_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_Kind_Args>(),
                extension_start: ptr::null_mut(),
                device_description: ptr::null(),
                device_kind: ptr::null(),
                device_kind_size: 0,
            };
            assert!((api.pjrt_device_description_kind)(ptr::addr_of_mut!(dkind).cast()).is_null());
            assert_eq!(dkind.device_kind_size, 11);

            let mut ev_err = crate::pjrt_types::PJRT_Event_Error_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Error_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!((api.pjrt_event_error)(ptr::addr_of_mut!(ev_err).cast()).is_null());

            let mut ev_aw = crate::pjrt_types::PJRT_Event_Await_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Await_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!((api.pjrt_event_await)(ptr::addr_of_mut!(ev_aw).cast()).is_null());

            let mut ev_rdy = crate::pjrt_types::PJRT_Event_OnReady_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_OnReady_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
                callback: None,
                user_arg: ptr::null_mut(),
            };
            assert!((api.pjrt_event_on_ready)(ptr::addr_of_mut!(ev_rdy).cast()).is_null());

            let mut ev_set = crate::pjrt_types::PJRT_Event_Set_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Set_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
                error_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
            };
            assert!((api.pjrt_event_set)(ptr::addr_of_mut!(ev_set).cast()).is_null());

            let mut err_pay = crate::pjrt_types::PJRT_Error_ForEachPayload_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Error_ForEachPayload_Args>(),
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                visitor: None,
                user_arg: ptr::null_mut(),
            };
            assert!((api.pjrt_error_for_each_payload)(ptr::addr_of_mut!(err_pay).cast()).is_null());

            let mut pois = crate::pjrt_types::PJRT_Device_PoisonExecution_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Device_PoisonExecution_Args>(),
                extension_start: ptr::null_mut(),
                device: ptr::null_mut(),
                launch_id: 0,
                error_code: 0,
                error_message: ptr::null(),
                error_message_size: 0,
                reserved0: 0,
                reserved1: 0,
                poisoned: false,
                pad: [0; 7],
            };
            assert!((api.pjrt_device_poison_execution)(ptr::addr_of_mut!(pois).cast()).is_null());
            assert!(pois.poisoned);

            let mut tracking_event_del = crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_async_tracking_event_destroy)(
                    ptr::addr_of_mut!(tracking_event_del).cast()
                )
                .is_null()
            );

            let neg_dims = [-1_i64];
            let mut mcs_bad =
                crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    topology: ptr::null(),
                    dims: neg_dims.as_ptr(),
                    num_dims: 1,
                    element_type: 11,
                    memory_space_kind: 0,
                    layout: ptr::null_mut(),
                    reserved: 0,
                    reserved1: 0,
                    reserved2: 0,
                    reserved3: 0,
                };
            assert!(
                (api.pjrt_topology_description_make_canonical_shape_for_memory_space)(
                    ptr::addr_of_mut!(mcs_bad).cast()
                )
                .is_null()
            );

            let mut st_del = crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                stream: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_copy_to_device_stream_destroy)(ptr::addr_of_mut!(st_del).cast())
                    .is_null()
            );
            let mut st_add = crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args>(),
                extension_start: ptr::null_mut(),
                stream: ptr::null_mut(),
                chunk: ptr::null_mut(),
                transfer_complete: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_copy_to_device_stream_add_chunk)(ptr::addr_of_mut!(st_add).cast())
                    .is_null()
            );
            let mut ev_del = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: st_add.transfer_complete,
            };
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del).cast()).is_null());

            let mut st_gran = crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args>(
                ),
                extension_start: ptr::null_mut(),
                stream: ptr::null(),
                granule_size_in_bytes: 0,
            };
            assert!(
                (api.pjrt_copy_to_device_stream_granule_size)(ptr::addr_of_mut!(st_gran).cast())
                    .is_null()
            );
            let mut st_cur = crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args>(
                ),
                extension_start: ptr::null_mut(),
                stream: ptr::null(),
                current_bytes: -1,
            };
            assert!(
                (api.pjrt_copy_to_device_stream_current_bytes)(ptr::addr_of_mut!(st_cur).cast())
                    .is_null()
            );

            let mut atm_del =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: ptr::null_mut(),
                };
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_Destroy(
                    ptr::addr_of_mut!(atm_del).cast()
                )
                .is_null()
            );
            let mut atm_ret =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: ptr::null_mut(),
                    buffer_index: 0,
                    pad: 0,
                    buffer_out: ptr::null_mut(),
                };
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
                    ptr::addr_of_mut!(atm_ret).cast()
                )
                .is_null()
            );
            let mut bdel = crate::pjrt_types::PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: atm_ret.buffer_out,
            };
            assert!((api.pjrt_buffer_destroy)(ptr::addr_of_mut!(bdel).cast()).is_null());

            let mut atm_sz =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: ptr::null_mut(),
                    buffer_index: 0,
                    pad: 0,
                    buffer_size: 0,
                };
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_BufferSize(
                    ptr::addr_of_mut!(atm_sz).cast()
                )
                .is_null()
            );
            let mut atm_set_error =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: ptr::null_mut(),
                    buffer_index: 0,
                    error_code: 3,
                    error_message: ptr::null(),
                    pad0: 0,
                    pad1: 0,
                };
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
                    ptr::addr_of_mut!(atm_set_error).cast()
                )
                .is_null()
            );
            let mut atm_lit =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: ptr::null_mut(),
                    buffer_index: 0,
                    pad: 0,
                    data: ptr::null(),
                    shape_dims: ptr::null(),
                    shape_num_dims: 0,
                    shape_element_type: 11,
                    pad1: 0,
                    shape_layout: ptr::null(),
                    done_with_h2d_transfer: ptr::null_mut(),
                };
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
                    ptr::addr_of_mut!(atm_lit).cast()
                )
                .is_null()
            );
            ev_del.event = atm_lit.done_with_h2d_transfer;
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del).cast()).is_null());

            let mut bfhb_null = crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                data: ptr::null(),
                element_type: 11,
                pad0: 0,
                dims: ptr::null(),
                num_dims: 0,
                byte_strides: ptr::null(),
                num_byte_strides: 0,
                host_buffer_semantics: 0,
                pad1: 0,
                device: ptr::null_mut(),
                memory: ptr::null_mut(),
                device_layout: ptr::null(),
                done_with_host_buffer: ptr::null_mut(),
                out_buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_buffer_from_host_buffer)(ptr::addr_of_mut!(bfhb_null).cast())
                    .is_null()
            );
            bdel.buffer = bfhb_null.out_buffer;
            assert!((api.pjrt_buffer_destroy)(ptr::addr_of_mut!(bdel).cast()).is_null());
            ev_del.event = bfhb_null.done_with_host_buffer;
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del).cast()).is_null());

            let mut ah2d_null =
                crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    client: ptr::null_mut(),
                    shape_specs: ptr::null(),
                    num_shape_specs: 0,
                    device_layouts: ptr::null_mut(),
                    num_device_layouts: 0,
                    memory: ptr::null_mut(),
                    transfer_manager: ptr::null_mut(),
                };
            assert!(
                (api.pjrt_client_create_buffers_for_async_host_to_device)(
                    ptr::addr_of_mut!(ah2d_null).cast()
                )
                .is_null()
            );
            atm_del.transfer_manager = ah2d_null.transfer_manager;
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_Destroy(
                    ptr::addr_of_mut!(atm_del).cast()
                )
                .is_null()
            );

            let mut fab_null = crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                buffer: ptr::null_mut(),
                status_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
                fulfill_alias_buffer_cb: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_fulfill_alias_buffer)(ptr::addr_of_mut!(fab_null).cast())
                    .is_null()
            );

            let mut err_buf_msg = crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                error_code: 3,
                reserved: 0,
                error_message: c"bad".as_ptr(),
                error_message_size: 3,
                dims: ptr::null(),
                num_dims: 0,
                element_type: 11,
                reserved1: 0,
                layout: ptr::null(),
                device: ptr::null_mut(),
                memory: ptr::null_mut(),
                buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_create_error_buffer)(ptr::addr_of_mut!(err_buf_msg).cast())
                    .is_null()
            );
            let mut bdel_only = crate::pjrt_types::PJRT_Buffer_Delete_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Delete_Args>(),
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
            };
            assert!((api.pjrt_buffer_delete)(ptr::addr_of_mut!(bdel_only).cast()).is_null());
            let mut crth = crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>(),
                extension_start: ptr::null_mut(),
                buffer: err_buf_msg.buffer,
                dst: ptr::null_mut(),
                offset: 0,
                transfer_size: 0,
                event: ptr::null_mut(),
            };
            assert!((api.pjrt_buffer_copy_raw_to_host)(ptr::addr_of_mut!(crth).cast()).is_null());
            ev_del.event = crth.event;
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del).cast()).is_null());
            bdel.buffer = err_buf_msg.buffer;
            assert!((api.pjrt_buffer_destroy)(ptr::addr_of_mut!(bdel).cast()).is_null());

            let mut ex_exec = crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args>(),
                extension_start: ptr::null_mut(),
                executable: ptr::null_mut(),
                options: ptr::null_mut(),
                argument_lists: ptr::null(),
                num_devices: 0,
                num_args: 0,
                output_lists: ptr::null(),
                device_complete_events: ptr::null_mut(),
                execute_device: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_loaded_executable_execute)(ptr::addr_of_mut!(ex_exec).cast()).is_null()
            );
            let mut opt_prog_buf = crate::pjrt_types::PJRT_Program {
                struct_size: size_of::<crate::pjrt_types::PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: ptr::null_mut(),
                code_size: 0,
                format: ptr::null(),
                format_size: 0,
            };
            let mut ex_opt = crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args>(),
                extension_start: ptr::null_mut(),
                executable: ptr::null(),
                program: ptr::addr_of_mut!(opt_prog_buf),
            };
            assert!(
                (api.pjrt_executable_optimized_program)(ptr::addr_of_mut!(ex_opt).cast()).is_null()
            );
            let mut opt_code_vec = vec![0_u8; opt_prog_buf.code_size];
            opt_prog_buf.code = opt_code_vec.as_mut_ptr().cast();
            ex_opt.program = ptr::addr_of_mut!(opt_prog_buf);
            assert!(
                (api.pjrt_executable_optimized_program)(ptr::addr_of_mut!(ex_opt).cast()).is_null()
            );

            let mut ex_ser = crate::pjrt_types::PJRT_Executable_Serialize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Serialize_Args>(),
                extension_start: ptr::null_mut(),
                executable: ptr::null(),
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_executable: ptr::null_mut(),
                serialized_executable_deleter: None,
            };
            assert!((api.pjrt_executable_serialize)(ptr::addr_of_mut!(ex_ser).cast()).is_null());
            pjrt_serialized_artifact_deleter(ex_ser.serialized_executable.cast());

            let mut ex_deser = crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args>(
                ),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                serialized_executable: ptr::null(),
                serialized_executable_size: 0,
                overridden_serialized_compile_options: ptr::null(),
                overridden_serialized_compile_options_size: 0,
                loaded_executable: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_executable_deserialize_and_load)(ptr::addr_of_mut!(ex_deser).cast())
                    .is_null()
            );

            let mut lex_delete = crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args>(),
                extension_start: ptr::null_mut(),
                executable: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_loaded_executable_delete)(ptr::addr_of_mut!(lex_delete).cast()).is_null()
            );

            let mut ex_del_null = crate::pjrt_types::PJRT_Executable_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: ptr::null_mut(),
            };
            assert!((api.pjrt_executable_destroy)(ptr::addr_of_mut!(ex_del_null).cast()).is_null());

            // Live client + buffer + transfer_manager edge branches
            let mut cc = crate::pjrt_types::PJRT_Client_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Create_Args>(),
                extension_start: ptr::null_mut(),
                create_options: ptr::null(),
                num_options: 0,
                kv_get_callback: ptr::null(),
                kv_get_user_arg: ptr::null_mut(),
                kv_put_callback: None,
                kv_put_user_arg: ptr::null_mut(),
                kv_try_get_callback: ptr::null(),
                kv_try_get_user_arg: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            assert!((api.pjrt_client_create)(ptr::addr_of_mut!(cc).cast()).is_null());
            let live_cli = cc.client;
            let live_dev0 = *(*live_cli)
                .addressable_device_ptrs
                .first()
                .unwrap_or(&ptr::null_mut());
            let vdims = [4_i64];
            let vhost = [1.0_f32; 4];
            let mut view_nocb = crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args>(
                ),
                extension_start: ptr::null_mut(),
                client: live_cli,
                device_buffer_ptr: vhost.as_ptr().cast(),
                dims: vdims.as_ptr(),
                num_dims: 1,
                element_type: 11,
                pad: 0,
                layout: ptr::null(),
                device: live_dev0,
                on_delete_callback: None,
                buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_client_create_view_of_device_buffer)(ptr::addr_of_mut!(view_nocb).cast())
                    .is_null()
            );
            let live_buf = view_nocb.buffer;

            let mut c2d_null = crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args>(),
                extension_start: ptr::null_mut(),
                buffer: live_buf,
                dst_device: ptr::null_mut(),
                dst_buffer: ptr::null_mut(),
            };
            assert!((api.pjrt_buffer_copy_to_device)(ptr::addr_of_mut!(c2d_null).cast()).is_null());
            assert!(c2d_null.dst_buffer.is_null());

            let mut copy_to_mem_null = crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args>(),
                extension_start: ptr::null_mut(),
                buffer: live_buf,
                dst_memory: ptr::null_mut(),
                dst_buffer: ptr::null_mut(),
            };
            assert!(
                (api.pjrt_buffer_copy_to_memory)(ptr::addr_of_mut!(copy_to_mem_null).cast())
                    .is_null()
            );
            assert!(copy_to_mem_null.dst_buffer.is_null());

            let spec = crate::pjrt_types::PJRT_ShapeSpec {
                struct_size: size_of::<crate::pjrt_types::PJRT_ShapeSpec>(),
                extension_start: ptr::null_mut(),
                dims: vdims.as_ptr(),
                num_dims: 1,
                element_type: 11,
                pad: 0,
            };
            let mut ah2d_live =
                crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    client: live_cli,
                    shape_specs: ptr::addr_of!(spec),
                    num_shape_specs: 1,
                    device_layouts: ptr::null_mut(),
                    num_device_layouts: 0,
                    memory: ptr::null_mut(),
                    transfer_manager: ptr::null_mut(),
                };
            assert!(
                (api.pjrt_client_create_buffers_for_async_host_to_device)(
                    ptr::addr_of_mut!(ah2d_live).cast()
                )
                .is_null()
            );
            let live_tm = ah2d_live.transfer_manager;

            let mut td_null =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: live_tm,
                    buffer_index: 0,
                    pad: 0,
                    data: ptr::null(),
                    offset: 0,
                    transfer_size: 0,
                    is_last_transfer: true,
                    pad1: [0; 7],
                    done_with_h2d_transfer: ptr::null_mut(),
                };
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_TransferData(
                    ptr::addr_of_mut!(td_null).cast()
                )
                .is_null()
            );
            ev_del.event = td_null.done_with_h2d_transfer;
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del).cast()).is_null());

            let mut am_null =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: live_tm,
                    transfer_metadata: ptr::null(),
                    num_metadata: 0,
                };
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
                    ptr::addr_of_mut!(am_null).cast()
                )
                .is_null()
            );

            td_null.transfer_manager = ptr::null_mut();
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_TransferData(
                    ptr::addr_of_mut!(td_null).cast()
                )
                .is_null()
            );
            ev_del.event = td_null.done_with_h2d_transfer;
            assert!((api.pjrt_event_destroy)(ptr::addr_of_mut!(ev_del).cast()).is_null());

            am_null.transfer_manager = ptr::null_mut();
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
                    ptr::addr_of_mut!(am_null).cast()
                )
                .is_null()
            );

            let bad_utf8_name = [0xff_u8, 0xfe_u8];
            for (tname, tlen) in [
                (ptr::null::<std::os::raw::c_char>(), 0_usize),
                (c"x".as_ptr(), 0_usize),
                (bad_utf8_name.as_ptr().cast(), bad_utf8_name.len()),
                (c"unknown_gen".as_ptr(), 11_usize),
            ] {
                let mut tc = crate::pjrt_types::PJRT_TopologyDescription_Create_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    topology_name: tname,
                    topology_name_size: tlen,
                    create_options: ptr::null(),
                    num_options: 0,
                    topology: ptr::null_mut(),
                };
                assert!(
                    (api.pjrt_topology_description_create)(ptr::addr_of_mut!(tc).cast()).is_null()
                );
                let mut td = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    topology: tc.topology,
                };
                assert!(
                    (api.pjrt_topology_description_destroy)(ptr::addr_of_mut!(td).cast()).is_null()
                );
            }

            atm_del.transfer_manager = live_tm;
            assert!(
                crate::PJRT_AsyncHostToDeviceTransferManager_Destroy(
                    ptr::addr_of_mut!(atm_del).cast()
                )
                .is_null()
            );
            bdel.buffer = live_buf;
            assert!((api.pjrt_buffer_destroy)(ptr::addr_of_mut!(bdel).cast()).is_null());
            let mut cdel = crate::pjrt_types::PJRT_Client_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                client: live_cli,
            };
            assert!((api.pjrt_client_destroy)(ptr::addr_of_mut!(cdel).cast()).is_null());
        }
    }
}
