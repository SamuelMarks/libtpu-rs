/**
 * @file libtpu.h
 * @brief Complete C11 ABI header for `libtpu-rs` exporting `PJRT_Api` (138
 * slots), `LibtpuSdk_Api` (17 slots), `TfTpu_*`, `TpuTopology_*`,
 * `TpuProfiler_*`, `TpuEmbeddingEngine_*`, `TpuDriver_*`, and `libtpu_rs_*`
 * entrypoints.
 */

#ifndef LIBTPU_H_
#define LIBTPU_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque PJRT error handle (`PJRT_Error`). */
typedef struct PJRT_Error PJRT_Error;
/** @brief Opaque PJRT client handle (`PJRT_Client`). */
typedef struct PJRT_Client PJRT_Client;
/** @brief Opaque PJRT device handle (`PJRT_Device`). */
typedef struct PJRT_Device PJRT_Device;
/** @brief Opaque PJRT device description handle (`PJRT_DeviceDescription`). */
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
/** @brief Opaque PJRT memory space handle (`PJRT_Memory`). */
typedef struct PJRT_Memory PJRT_Memory;
/** @brief Opaque PJRT executable handle (`PJRT_Executable`). */
typedef struct PJRT_Executable PJRT_Executable;
/** @brief Opaque PJRT loaded executable handle (`PJRT_LoadedExecutable`). */
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
/** @brief Opaque PJRT buffer handle (`PJRT_Buffer`). */
typedef struct PJRT_Buffer PJRT_Buffer;
/** @brief Opaque PJRT event handle (`PJRT_Event`). */
typedef struct PJRT_Event PJRT_Event;
/** @brief Opaque PJRT topology description handle (`PJRT_TopologyDescription`).
 */
typedef struct PJRT_TopologyDescription PJRT_TopologyDescription;
/** @brief Opaque PJRT copy-to-device stream handle (`PJRT_CopyToDeviceStream`).
 */
typedef struct PJRT_CopyToDeviceStream PJRT_CopyToDeviceStream;
/** @brief Opaque PJRT async H2D transfer manager handle
 * (`PJRT_AsyncHostToDeviceTransferManager`). */
typedef struct PJRT_AsyncHostToDeviceTransferManager
    PJRT_AsyncHostToDeviceTransferManager;
/** @brief Opaque PJRT execution context handle (`PJRT_ExecuteContext`). */
typedef struct PJRT_ExecuteContext PJRT_ExecuteContext;
/** @brief Opaque PJRT async tracking event handle (`PJRT_AsyncTrackingEvent`).
 */
typedef struct PJRT_AsyncTrackingEvent PJRT_AsyncTrackingEvent;
/** @brief Opaque PJRT serialized executable handle
 * (`PJRT_SerializedExecutable`). */
typedef struct PJRT_SerializedExecutable PJRT_SerializedExecutable;
/** @brief Opaque PJRT serialized topology handle (`PJRT_SerializedTopology`).
 */
typedef struct PJRT_SerializedTopology PJRT_SerializedTopology;
/** @brief Opaque LibtpuSdk error handle (`LibtpuSdk_Error`). */
typedef struct LibtpuSdk_Error LibtpuSdk_Error;
/** @brief Opaque LibtpuSdk client handle (`LibtpuSdk_Client`). */
typedef struct LibtpuSdk_Client LibtpuSdk_Client;
/** @brief Opaque LibtpuSdk metric handle (`LibtpuSdk_Metric`). */
typedef struct LibtpuSdk_Metric LibtpuSdk_Metric;
/** @brief Opaque LibtpuSdk runtime status handle (`LibtpuSdk_RuntimeStatus`).
 */
typedef struct LibtpuSdk_RuntimeStatus LibtpuSdk_RuntimeStatus;
/** @brief Opaque TPU status handle (`TpuStatus`). */
typedef struct TpuStatus TpuStatus;
/** @brief Opaque TPU profiler handle (`TpuProfiler`). */
typedef struct TpuProfiler TpuProfiler;
/** @brief Opaque TPU topology handle (`TpuTopology`). */
typedef struct TpuTopology TpuTopology;
/** @brief Opaque TPU core location handle (`TpuCoreLocation`). */
typedef struct TpuCoreLocation TpuCoreLocation;
/** @brief Opaque TPU host location handle (`TpuHostLocation`). */
typedef struct TpuHostLocation TpuHostLocation;
/** @brief Opaque TPU mesh state handle (`TpuMeshState`). */
typedef struct TpuMeshState TpuMeshState;
/** @brief Opaque TPU ordinal selector handle (`TfTpuOrdinalSelector`). */
typedef struct TfTpuOrdinalSelector TfTpuOrdinalSelector;
/** @brief Opaque TPU embedding engine state handle (`TpuEmbeddingEngineState`).
 */
typedef struct TpuEmbeddingEngineState TpuEmbeddingEngineState;
/** @brief Opaque TPU driver handle (`TpuDriver`). */
typedef struct TpuDriver TpuDriver;

/** @brief Void-returning PJRT C API vtable function pointer (`PjrtVoidFn`). */
typedef void (*PjrtVoidFn)(void *args);
/** @brief Error-returning PJRT C API vtable function pointer (`PjrtApiFn`). */
typedef PJRT_Error *(*PjrtApiFn)(void *args);
/** @brief Void-returning LibtpuSdk C API vtable function pointer
 * (`LibtpuSdkVoidFn`). */
typedef void (*LibtpuSdkVoidFn)(void *args);
/** @brief Error-returning LibtpuSdk C API vtable function pointer
 * (`LibtpuSdkFn`). */
typedef LibtpuSdk_Error *(*LibtpuSdkFn)(void *args);

/**
 * @brief Base header for PJRT extension structures (`PJRT_Extension_Base`).
 */
typedef struct PJRT_Extension_Base {
  /** @brief Size of the extension structure in bytes (`struct_size`). */
  size_t struct_size;
  /** @brief Extension type discriminator (`ext_type`). */
  int32_t ext_type;
  /** @brief Explicit 4-byte alignment padding (`pad`). */
  int32_t pad;
  /** @brief Pointer to the next extension in the linked list (`next`). */
  const struct PJRT_Extension_Base *next;
} PJRT_Extension_Base;

/**
 * @brief PJRT API version structure (`PJRT_Api_Version`, 24 bytes).
 */
typedef struct PJRT_Api_Version {
  /** @brief Size of this structure (`24`). */
  size_t struct_size;
  /** @brief Optional extension chain pointer (`NULL`). */
  const PJRT_Extension_Base *extension_start;
  /** @brief Major PJRT version (`0`). */
  int32_t major_version;
  /** @brief Minor PJRT version (`116`). */
  int32_t minor_version;
} PJRT_Api_Version;

/**
 * @brief Complete 1,144-byte `PJRT_Api` vtable returned by `GetPjrtApi()`.
 */
typedef struct PJRT_Api {
  /** @brief Total byte size of `PJRT_Api` (`1144`). */
  size_t struct_size;
  /** @brief Pointer to the first `PJRT_Extension_Base` in the 17-node TPU
   * extension chain. */
  const PJRT_Extension_Base *extension_start;
  /** @brief Embedded 24-byte `PJRT_Api_Version` (`0.116`). */
  PJRT_Api_Version pjrt_api_version;
  /** @brief Slot 1 (`0x028`): `PJRT_Error_Destroy`. */
  PjrtVoidFn pjrt_error_destroy;
  /** @brief Slot 2 (`0x030`): `PJRT_Error_Message`. */
  PjrtVoidFn pjrt_error_message;
  /** @brief Slot 3 (`0x038`): `PJRT_Error_GetCode`. */
  PjrtApiFn pjrt_error_get_code;
  /** @brief Slot 4 (`0x040`): `PJRT_Plugin_Initialize`. */
  PjrtApiFn pjrt_plugin_initialize;
  /** @brief Slot 5 (`0x048`): `PJRT_Plugin_Attributes`. */
  PjrtApiFn pjrt_plugin_attributes;
  /** @brief Slot 6 (`0x050`): `PJRT_Event_Destroy`. */
  PjrtApiFn pjrt_event_destroy;
  /** @brief Slot 7 (`0x058`): `PJRT_Event_IsReady`. */
  PjrtApiFn pjrt_event_is_ready;
  /** @brief Slot 8 (`0x060`): `PJRT_Event_Error`. */
  PjrtApiFn pjrt_event_error;
  /** @brief Slot 9 (`0x068`): `PJRT_Event_Await`. */
  PjrtApiFn pjrt_event_await;
  /** @brief Slot 10 (`0x070`): `PJRT_Event_OnReady`. */
  PjrtApiFn pjrt_event_on_ready;
  /** @brief Slot 11 (`0x078`): `PJRT_Client_Create`. */
  PjrtApiFn pjrt_client_create;
  /** @brief Slot 12 (`0x080`): `PJRT_Client_Destroy`. */
  PjrtApiFn pjrt_client_destroy;
  /** @brief Slot 13 (`0x088`): `PJRT_Client_PlatformName`. */
  PjrtApiFn pjrt_client_platform_name;
  /** @brief Slot 14 (`0x090`): `PJRT_Client_ProcessIndex`. */
  PjrtApiFn pjrt_client_process_index;
  /** @brief Slot 15 (`0x098`): `PJRT_Client_PlatformVersion`. */
  PjrtApiFn pjrt_client_platform_version;
  /** @brief Slot 16 (`0x0a0`): `PJRT_Client_Devices`. */
  PjrtApiFn pjrt_client_devices;
  /** @brief Slot 17 (`0x0a8`): `PJRT_Client_AddressableDevices`. */
  PjrtApiFn pjrt_client_addressable_devices;
  /** @brief Slot 18 (`0x0b0`): `PJRT_Client_LookupDevice`. */
  PjrtApiFn pjrt_client_lookup_device;
  /** @brief Slot 19 (`0x0b8`): `PJRT_Client_LookupAddressableDevice`. */
  PjrtApiFn pjrt_client_lookup_addressable_device;
  /** @brief Slot 20 (`0x0c0`): `PJRT_Client_AddressableMemories`. */
  PjrtApiFn pjrt_client_addressable_memories;
  /** @brief Slot 21 (`0x0c8`): `PJRT_Client_Compile`. */
  PjrtApiFn pjrt_client_compile;
  /** @brief Slot 22 (`0x0d0`): `PJRT_Client_DefaultDeviceAssignment`. */
  PjrtApiFn pjrt_client_default_device_assignment;
  /** @brief Slot 23 (`0x0d8`): `PJRT_Client_BufferFromHostBuffer`. */
  PjrtApiFn pjrt_client_buffer_from_host_buffer;
  /** @brief Slot 24 (`0x0e0`): `PJRT_DeviceDescription_Id`. */
  PjrtApiFn pjrt_device_description_id;
  /** @brief Slot 25 (`0x0e8`): `PJRT_DeviceDescription_ProcessIndex`. */
  PjrtApiFn pjrt_device_description_process_index;
  /** @brief Slot 26 (`0x0f0`): `PJRT_DeviceDescription_Attributes`. */
  PjrtApiFn pjrt_device_description_attributes;
  /** @brief Slot 27 (`0x0f8`): `PJRT_DeviceDescription_Kind`. */
  PjrtApiFn pjrt_device_description_kind;
  /** @brief Slot 28 (`0x100`): `PJRT_DeviceDescription_DebugString`. */
  PjrtApiFn pjrt_device_description_debug_string;
  /** @brief Slot 29 (`0x108`): `PJRT_DeviceDescription_ToString`. */
  PjrtApiFn pjrt_device_description_to_string;
  /** @brief Slot 30 (`0x110`): `PJRT_Device_GetDescription`. */
  PjrtApiFn pjrt_device_get_description;
  /** @brief Slot 31 (`0x118`): `PJRT_Device_IsAddressable`. */
  PjrtApiFn pjrt_device_is_addressable;
  /** @brief Slot 32 (`0x120`): `PJRT_Device_LocalHardwareId`. */
  PjrtApiFn pjrt_device_local_hardware_id;
  /** @brief Slot 33 (`0x128`): `PJRT_Device_AddressableMemories`. */
  PjrtApiFn pjrt_device_addressable_memories;
  /** @brief Slot 34 (`0x130`): `PJRT_Device_DefaultMemory`. */
  PjrtApiFn pjrt_device_default_memory;
  /** @brief Slot 35 (`0x138`): `PJRT_Device_MemoryStats`. */
  PjrtApiFn pjrt_device_memory_stats;
  /** @brief Slot 36 (`0x140`): `PJRT_Memory_Id`. */
  PjrtApiFn pjrt_memory_id;
  /** @brief Slot 37 (`0x148`): `PJRT_Memory_Kind`. */
  PjrtApiFn pjrt_memory_kind;
  /** @brief Slot 38 (`0x150`): `PJRT_Memory_DebugString`. */
  PjrtApiFn pjrt_memory_debug_string;
  /** @brief Slot 39 (`0x158`): `PJRT_Memory_ToString`. */
  PjrtApiFn pjrt_memory_to_string;
  /** @brief Slot 40 (`0x160`): `PJRT_Memory_AddressableByDevices`. */
  PjrtApiFn pjrt_memory_addressable_by_devices;
  /** @brief Slot 41 (`0x168`): `PJRT_Executable_Destroy`. */
  PjrtApiFn pjrt_executable_destroy;
  /** @brief Slot 42 (`0x170`): `PJRT_Executable_Name`. */
  PjrtApiFn pjrt_executable_name;
  /** @brief Slot 43 (`0x178`): `PJRT_Executable_NumReplicas`. */
  PjrtApiFn pjrt_executable_num_replicas;
  /** @brief Slot 44 (`0x180`): `PJRT_Executable_NumPartitions`. */
  PjrtApiFn pjrt_executable_num_partitions;
  /** @brief Slot 45 (`0x188`): `PJRT_Executable_NumOutputs`. */
  PjrtApiFn pjrt_executable_num_outputs;
  /** @brief Slot 46 (`0x190`): `PJRT_Executable_SizeOfGeneratedCodeInBytes`. */
  PjrtApiFn pjrt_executable_size_of_generated_code_in_bytes;
  /** @brief Slot 47 (`0x198`): `PJRT_Executable_GetCostAnalysis`. */
  PjrtApiFn pjrt_executable_get_cost_analysis;
  /** @brief Slot 48 (`0x1a0`): `PJRT_Executable_OutputMemoryKinds`. */
  PjrtApiFn pjrt_executable_output_memory_kinds;
  /** @brief Slot 49 (`0x1a8`): `PJRT_Executable_OptimizedProgram`. */
  PjrtApiFn pjrt_executable_optimized_program;
  /** @brief Slot 50 (`0x1b0`): `PJRT_Executable_Serialize`. */
  PjrtApiFn pjrt_executable_serialize;
  /** @brief Slot 51 (`0x1b8`): `PJRT_LoadedExecutable_Destroy`. */
  PjrtApiFn pjrt_loaded_executable_destroy;
  /** @brief Slot 52 (`0x1c0`): `PJRT_LoadedExecutable_GetExecutable`. */
  PjrtApiFn pjrt_loaded_executable_get_executable;
  /** @brief Slot 53 (`0x1c8`): `PJRT_LoadedExecutable_AddressableDevices`. */
  PjrtApiFn pjrt_loaded_executable_addressable_devices;
  /** @brief Slot 54 (`0x1d0`): `PJRT_LoadedExecutable_Delete`. */
  PjrtApiFn pjrt_loaded_executable_delete;
  /** @brief Slot 55 (`0x1d8`): `PJRT_LoadedExecutable_IsDeleted`. */
  PjrtApiFn pjrt_loaded_executable_is_deleted;
  /** @brief Slot 56 (`0x1e0`): `PJRT_LoadedExecutable_Execute`. */
  PjrtApiFn pjrt_loaded_executable_execute;
  /** @brief Slot 57 (`0x1e8`): `PJRT_Executable_DeserializeAndLoad`. */
  PjrtApiFn pjrt_executable_deserialize_and_load;
  /** @brief Slot 58 (`0x1f0`): `PJRT_LoadedExecutable_Fingerprint`. */
  PjrtApiFn pjrt_loaded_executable_fingerprint;
  /** @brief Slot 59 (`0x1f8`): `PJRT_Buffer_Destroy`. */
  PjrtApiFn pjrt_buffer_destroy;
  /** @brief Slot 60 (`0x200`): `PJRT_Buffer_ElementType`. */
  PjrtApiFn pjrt_buffer_element_type;
  /** @brief Slot 61 (`0x208`): `PJRT_Buffer_Dimensions`. */
  PjrtApiFn pjrt_buffer_dimensions;
  /** @brief Slot 62 (`0x210`): `PJRT_Buffer_UnpaddedDimensions`. */
  PjrtApiFn pjrt_buffer_unpadded_dimensions;
  /** @brief Slot 63 (`0x218`): `PJRT_Buffer_DynamicDimensionIndices`. */
  PjrtApiFn pjrt_buffer_dynamic_dimension_indices;
  /** @brief Slot 64 (`0x220`): `PJRT_Buffer_GetMemoryLayout`. */
  PjrtApiFn pjrt_buffer_get_memory_layout;
  /** @brief Slot 65 (`0x228`): `PJRT_Buffer_OnDeviceSizeInBytes`. */
  PjrtApiFn pjrt_buffer_on_device_size_in_bytes;
  /** @brief Slot 66 (`0x230`): `PJRT_Buffer_Device`. */
  PjrtApiFn pjrt_buffer_device;
  /** @brief Slot 67 (`0x238`): `PJRT_Buffer_Memory`. */
  PjrtApiFn pjrt_buffer_memory;
  /** @brief Slot 68 (`0x240`): `PJRT_Buffer_Delete`. */
  PjrtApiFn pjrt_buffer_delete;
  /** @brief Slot 69 (`0x248`): `PJRT_Buffer_IsDeleted`. */
  PjrtApiFn pjrt_buffer_is_deleted;
  /** @brief Slot 70 (`0x250`): `PJRT_Buffer_CopyToDevice`. */
  PjrtApiFn pjrt_buffer_copy_to_device;
  /** @brief Slot 71 (`0x258`): `PJRT_Buffer_ToHostBuffer`. */
  PjrtApiFn pjrt_buffer_to_host_buffer;
  /** @brief Slot 72 (`0x260`): `PJRT_Buffer_IsOnCpu`. */
  PjrtApiFn pjrt_buffer_is_on_cpu;
  /** @brief Slot 73 (`0x268`): `PJRT_Buffer_ReadyEvent`. */
  PjrtApiFn pjrt_buffer_ready_event;
  /** @brief Slot 74 (`0x270`): `PJRT_Buffer_UnsafePointer`. */
  PjrtApiFn pjrt_buffer_unsafe_pointer;
  /** @brief Slot 75 (`0x278`): `PJRT_Buffer_IncreaseExternalReferenceCount`. */
  PjrtApiFn pjrt_buffer_increase_external_reference_count;
  /** @brief Slot 76 (`0x280`): `PJRT_Buffer_DecreaseExternalReferenceCount`. */
  PjrtApiFn pjrt_buffer_decrease_external_reference_count;
  /** @brief Slot 77 (`0x288`): `PJRT_Buffer_OpaqueDeviceMemoryDataPointer`. */
  PjrtApiFn pjrt_buffer_opaque_device_memory_data_pointer;
  /** @brief Slot 78 (`0x290`): `PJRT_CopyToDeviceStream_Destroy`. */
  PjrtApiFn pjrt_copy_to_device_stream_destroy;
  /** @brief Slot 79 (`0x298`): `PJRT_CopyToDeviceStream_AddChunk`. */
  PjrtApiFn pjrt_copy_to_device_stream_add_chunk;
  /** @brief Slot 80 (`0x2a0`): `PJRT_CopyToDeviceStream_TotalBytes`. */
  PjrtApiFn pjrt_copy_to_device_stream_total_bytes;
  /** @brief Slot 81 (`0x2a8`): `PJRT_CopyToDeviceStream_GranuleSize`. */
  PjrtApiFn pjrt_copy_to_device_stream_granule_size;
  /** @brief Slot 82 (`0x2b0`): `PJRT_CopyToDeviceStream_CurrentBytes`. */
  PjrtApiFn pjrt_copy_to_device_stream_current_bytes;
  /** @brief Slot 83 (`0x2b8`): `PJRT_TopologyDescription_Create`. */
  PjrtApiFn pjrt_topology_description_create;
  /** @brief Slot 84 (`0x2c0`): `PJRT_TopologyDescription_Destroy`. */
  PjrtApiFn pjrt_topology_description_destroy;
  /** @brief Slot 85 (`0x2c8`): `PJRT_TopologyDescription_PlatformName`. */
  PjrtApiFn pjrt_topology_description_platform_name;
  /** @brief Slot 86 (`0x2d0`): `PJRT_TopologyDescription_PlatformVersion`. */
  PjrtApiFn pjrt_topology_description_platform_version;
  /** @brief Slot 87 (`0x2d8`):
   * `PJRT_TopologyDescription_GetDeviceDescriptions`. */
  PjrtApiFn pjrt_topology_description_get_device_descriptions;
  /** @brief Slot 88 (`0x2e0`): `PJRT_TopologyDescription_Serialize`. */
  PjrtApiFn pjrt_topology_description_serialize;
  /** @brief Slot 89 (`0x2e8`): `PJRT_TopologyDescription_Attributes`. */
  PjrtApiFn pjrt_topology_description_attributes;
  /** @brief Slot 90 (`0x2f0`): `PJRT_Compile`. */
  PjrtApiFn pjrt_compile;
  /** @brief Slot 91 (`0x2f8`): `PJRT_Executable_OutputElementTypes`. */
  PjrtApiFn pjrt_executable_output_element_types;
  /** @brief Slot 92 (`0x300`): `PJRT_Executable_OutputDimensions`. */
  PjrtApiFn pjrt_executable_output_dimensions;
  /** @brief Slot 93 (`0x308`): `PJRT_Buffer_CopyToMemory`. */
  PjrtApiFn pjrt_buffer_copy_to_memory;
  /** @brief Slot 94 (`0x310`): `PJRT_Client_CreateViewOfDeviceBuffer`. */
  PjrtApiFn pjrt_client_create_view_of_device_buffer;
  /** @brief Slot 95 (`0x318`): `PJRT_Executable_Fingerprint`. */
  PjrtApiFn pjrt_executable_fingerprint;
  /** @brief Slot 96 (`0x320`): `PJRT_Client_TopologyDescription`. */
  PjrtApiFn pjrt_client_topology_description;
  /** @brief Slot 97 (`0x328`): `PJRT_Executable_GetCompiledMemoryStats`. */
  PjrtApiFn pjrt_executable_get_compiled_memory_stats;
  /** @brief Slot 98 (`0x330`): `PJRT_Memory_Kind_Id`. */
  PjrtApiFn pjrt_memory_kind_id;
  /** @brief Slot 99 (`0x338`): `PJRT_ExecuteContext_Create`. */
  PjrtApiFn pjrt_execute_context_create;
  /** @brief Slot 100 (`0x340`): `PJRT_ExecuteContext_Destroy`. */
  PjrtApiFn pjrt_execute_context_destroy;
  /** @brief Slot 101 (`0x348`): `PJRT_Buffer_CopyRawToHost`. */
  PjrtApiFn pjrt_buffer_copy_raw_to_host;
  /** @brief Slot 102 (`0x350`):
   * `PJRT_AsyncHostToDeviceTransferManager_Destroy`. */
  PjrtApiFn pjrt_async_transfer_manager_destroy;
  /** @brief Slot 103 (`0x358`):
   * `PJRT_AsyncHostToDeviceTransferManager_TransferData`. */
  PjrtApiFn pjrt_async_transfer_manager_transfer_data;
  /** @brief Slot 104 (`0x360`):
   * `PJRT_Client_CreateBuffersForAsyncHostToDevice`. */
  PjrtApiFn pjrt_client_create_buffers_for_async_host_to_device;
  /** @brief Slot 105 (`0x368`):
   * `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer`. */
  PjrtApiFn pjrt_async_transfer_manager_retrieve_buffer;
  /** @brief Slot 106 (`0x370`): `PJRT_AsyncHostToDeviceTransferManager_Device`.
   */
  PjrtApiFn pjrt_async_transfer_manager_device;
  /** @brief Slot 107 (`0x378`):
   * `PJRT_AsyncHostToDeviceTransferManager_BufferCount`. */
  PjrtApiFn pjrt_async_transfer_manager_buffer_count;
  /** @brief Slot 108 (`0x380`):
   * `PJRT_AsyncHostToDeviceTransferManager_BufferSize`. */
  PjrtApiFn pjrt_async_transfer_manager_buffer_size;
  /** @brief Slot 109 (`0x388`):
   * `PJRT_AsyncHostToDeviceTransferManager_SetBufferError`. */
  PjrtApiFn pjrt_async_transfer_manager_set_buffer_error;
  /** @brief Slot 110 (`0x390`):
   * `PJRT_AsyncHostToDeviceTransferManager_AddMetadata`. */
  PjrtApiFn pjrt_async_transfer_manager_add_metadata;
  /** @brief Slot 111 (`0x398`): `PJRT_Client_DmaMap`. */
  PjrtApiFn pjrt_client_dma_map;
  /** @brief Slot 112 (`0x3a0`): `PJRT_Client_DmaUnmap`. */
  PjrtApiFn pjrt_client_dma_unmap;
  /** @brief Slot 113 (`0x3a8`): `PJRT_Client_CreateUninitializedBuffer`. */
  PjrtApiFn pjrt_client_create_uninitialized_buffer;
  /** @brief Slot 114 (`0x3b0`): `PJRT_Client_UpdateGlobalProcessInfo`. */
  PjrtApiFn pjrt_client_update_global_process_info;
  /** @brief Slot 115 (`0x3b8`): `PJRT_TopologyDescription_Deserialize`. */
  PjrtApiFn pjrt_topology_description_deserialize;
  /** @brief Slot 116 (`0x3c0`): `PJRT_Client_CreateAliasBuffer`. */
  PjrtApiFn pjrt_client_create_alias_buffer;
  /** @brief Slot 117 (`0x3c8`): `PJRT_Client_FulfillAliasBuffer`. */
  PjrtApiFn pjrt_client_fulfill_alias_buffer;
  /** @brief Slot 118 (`0x3d0`): `PJRT_LoadedExecutable_GetDeviceAssignment`. */
  PjrtApiFn pjrt_loaded_executable_get_device_assignment;
  /** @brief Slot 119 (`0x3d8`): `PJRT_Client_CreateErrorBuffer`. */
  PjrtApiFn pjrt_client_create_error_buffer;
  /** @brief Slot 120 (`0x3e0`):
   * `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`. */
  PjrtApiFn pjrt_async_transfer_manager_transfer_literal;
  /** @brief Slot 121 (`0x3e8`): `PJRT_Buffer_CopyRawToHostFuture`. */
  PjrtApiFn pjrt_buffer_copy_raw_to_host_future;
  /** @brief Slot 122 (`0x3f0`): `PJRT_Device_PoisonExecution`. */
  PjrtApiFn pjrt_device_poison_execution;
  /** @brief Slot 123 (`0x3f8`): `PJRT_Device_CreateAsyncTrackingEvent`. */
  PjrtApiFn pjrt_device_create_async_tracking_event;
  /** @brief Slot 124 (`0x400`): `PJRT_AsyncTrackingEvent_Destroy`. */
  PjrtApiFn pjrt_async_tracking_event_destroy;
  /** @brief Slot 125 (`0x408`): `PJRT_Executable_GetCompileOptions`. */
  PjrtApiFn pjrt_executable_get_compile_options;
  /** @brief Slot 126 (`0x410`): `PJRT_Buffer_DonateWithControlDependency`. */
  PjrtApiFn pjrt_buffer_donate_with_control_dependency;
  /** @brief Slot 127 (`0x418`): `PJRT_Event_Create`. */
  PjrtApiFn pjrt_event_create;
  /** @brief Slot 128 (`0x420`): `PJRT_Event_Set`. */
  PjrtApiFn pjrt_event_set;
  /** @brief Slot 129 (`0x428`): `PJRT_Device_GetAttributes`. */
  PjrtApiFn pjrt_device_get_attributes;
  /** @brief Slot 130 (`0x430`): `PJRT_Client_Load`. */
  PjrtApiFn pjrt_client_load;
  /** @brief Slot 131 (`0x438`):
   * `PJRT_LoadedExecutable_AddressableDeviceLogicalIds`. */
  PjrtApiFn pjrt_loaded_executable_addressable_device_logical_ids;
  /** @brief Slot 132 (`0x440`): `PJRT_Buffer_Bitcast`. */
  PjrtApiFn pjrt_buffer_bitcast;
  /** @brief Slot 133 (`0x448`): `PJRT_Error_ForEachPayload`. */
  PjrtApiFn pjrt_error_for_each_payload;
  /** @brief Slot 134 (`0x450`): `PJRT_TopologyDescription_Fingerprint`. */
  PjrtApiFn pjrt_topology_description_fingerprint;
  /** @brief Slot 135 (`0x458`): `PJRT_Executable_ParameterMemoryKinds`. */
  PjrtApiFn pjrt_executable_parameter_memory_kinds;
  /** @brief Slot 136 (`0x460`): `PJRT_Device_ClearMemoryStats`. */
  PjrtApiFn pjrt_device_clear_memory_stats;
  /** @brief Slot 137 (`0x468`):
   * `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace`. */
  PjrtApiFn pjrt_topology_description_make_canonical_shape_for_memory_space;
  /** @brief Slot 138 (`0x470`):
   * `PJRT_TopologyDescription_GetMemorySpaceKindIds`. */
  PjrtApiFn pjrt_topology_description_get_memory_space_kind_ids;
} PJRT_Api;

/**
 * @brief Argument structure for `PJRT_Error_Destroy` (`24` bytes).
 */
typedef struct PJRT_Error_Destroy_Args {
  /** @brief Size of this struct (`24`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Error handle to destroy. */
  PJRT_Error *error;
} PJRT_Error_Destroy_Args;

/**
 * @brief Argument structure for `PJRT_Error_Message` (`40` bytes).
 */
typedef struct PJRT_Error_Message_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Error handle to query. */
  const PJRT_Error *error;
  /** @brief Output pointer to UTF-8 message bytes. */
  const char *message;
  /** @brief Output byte length of `message`. */
  size_t message_size;
} PJRT_Error_Message_Args;

/**
 * @brief Argument structure for `PJRT_Error_GetCode` (`32` bytes).
 */
typedef struct PJRT_Error_GetCode_Args {
  /** @brief Size of this struct (`32`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Error handle to query. */
  const PJRT_Error *error;
  /** @brief Output integer status code (`PJRT_Error_Code`). */
  int32_t code;
  /** @brief Explicit 4-byte alignment padding. */
  int32_t pad;
} PJRT_Error_GetCode_Args;

/**
 * @brief Program representation passed to `PJRT_Client_Compile` (`48` bytes).
 */
typedef struct PJRT_Program {
  /** @brief Size of this struct (`48`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Pointer to program code bytes (`MLIR` or `HloModuleProto`). */
  const char *code;
  /** @brief Length of `code` in bytes. */
  size_t code_size;
  /** @brief Format string pointer (`"mlir"`, `"hlo"`, `"hlo_with_config"`). */
  const char *format;
  /** @brief Length of `format` string in bytes. */
  size_t format_size;
} PJRT_Program;

/**
 * @brief Argument structure for `PJRT_Client_Create` (`88` bytes).
 */
typedef struct PJRT_Client_Create_Args {
  /** @brief Size of this struct (`88`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Optional pointer to `PJRT_NamedValue` creation options. */
  const void *create_options;
  /** @brief Number of creation options. */
  size_t num_options;
  /** @brief Optional key-value get callback. */
  const void *kv_get_callback;
  /** @brief User argument for `kv_get_callback`. */
  void *kv_get_user_arg;
  /** @brief Optional key-value put callback. */
  const void *kv_put_callback;
  /** @brief User argument for `kv_put_callback`. */
  void *kv_put_user_arg;
  /** @brief Optional non-blocking key-value try-get callback. */
  const void *kv_try_get_callback;
  /** @brief User argument for `kv_try_get_callback`. */
  void *kv_try_get_user_arg;
  /** @brief Output created `PJRT_Client*`. */
  PJRT_Client *client;
} PJRT_Client_Create_Args;

/**
 * @brief Argument structure for `PJRT_Client_Destroy` (`24` bytes).
 */
typedef struct PJRT_Client_Destroy_Args {
  /** @brief Size of this struct (`24`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Client handle to destroy. */
  PJRT_Client *client;
} PJRT_Client_Destroy_Args;

/**
 * @brief Argument structure for `PJRT_Client_Devices` and
 * `PJRT_Client_AddressableDevices` (`40` bytes).
 */
typedef struct PJRT_Client_Devices_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Client handle. */
  PJRT_Client *client;
  /** @brief Output pointer to `PJRT_Device*` array. */
  PJRT_Device *const *devices;
  /** @brief Output number of devices. */
  size_t num_devices;
} PJRT_Client_Devices_Args;

/**
 * @brief Argument structure for `PJRT_Client_LookupDevice` (`40` bytes).
 */
typedef struct PJRT_Client_LookupDevice_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Client handle. */
  PJRT_Client *client;
  /** @brief Target global device ID. */
  int32_t id;
  /** @brief Explicit 4-byte alignment padding. */
  int32_t pad;
  /** @brief Output `PJRT_Device*`. */
  PJRT_Device *device;
} PJRT_Client_LookupDevice_Args;

/**
 * @brief Argument structure for `PJRT_Client_Compile` (`56` bytes).
 */
typedef struct PJRT_Client_Compile_Args {
  /** @brief Size of this struct (`56`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Client handle. */
  PJRT_Client *client;
  /** @brief Input `PJRT_Program*`. */
  const PJRT_Program *program;
  /** @brief Serialized `CompileOptionsProto` bytes pointer. */
  const char *compile_options;
  /** @brief Byte length of `compile_options`. */
  size_t compile_options_size;
  /** @brief Output compiled `PJRT_LoadedExecutable*`. */
  PJRT_LoadedExecutable *executable;
} PJRT_Client_Compile_Args;

/**
 * @brief Argument structure for `PJRT_Client_BufferFromHostBuffer` (`120`
 * bytes).
 */
typedef struct PJRT_Client_BufferFromHostBuffer_Args {
  /** @brief Size of this struct (`120`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Client handle. */
  PJRT_Client *client;
  /** @brief Host data pointer. */
  const void *data;
  /** @brief Element type (`PJRT_Buffer_Type`). */
  int32_t type;
  /** @brief Explicit 4-byte alignment padding. */
  int32_t pad0;
  /** @brief Dimension sizes array pointer. */
  const int64_t *dims;
  /** @brief Number of dimensions. */
  size_t num_dims;
  /** @brief Optional byte strides array pointer. */
  const int64_t *byte_strides;
  /** @brief Number of byte strides. */
  size_t num_byte_strides;
  /** @brief Host buffer semantics. */
  int32_t host_buffer_semantics;
  /** @brief Explicit 4-byte alignment padding. */
  int32_t pad1;
  /** @brief Target `PJRT_Device*`. */
  PJRT_Device *device;
  /** @brief Optional target `PJRT_Memory*`. */
  PJRT_Memory *memory;
  /** @brief Optional device memory layout pointer. */
  const void *device_layout;
  /** @brief Output host buffer completion `PJRT_Event*`. */
  PJRT_Event *done_with_host_buffer;
  /** @brief Output created `PJRT_Buffer*`. */
  PJRT_Buffer *buffer;
} PJRT_Client_BufferFromHostBuffer_Args;

/**
 * @brief Argument structure for `PJRT_Executable_GetCostAnalysis` (`40` bytes).
 */
typedef struct PJRT_Executable_GetCostAnalysis_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Executable handle. */
  const PJRT_Executable *executable;
  /** @brief Output pointer to `PJRT_NamedValue` array. */
  const void *properties;
  /** @brief Output number of properties. */
  size_t num_properties;
} PJRT_Executable_GetCostAnalysis_Args;

/**
 * @brief Argument structure for `PJRT_Executable_OutputMemoryKinds` (`48`
 * bytes).
 */
typedef struct PJRT_Executable_OutputMemoryKinds_Args {
  /** @brief Size of this struct (`48`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Executable handle. */
  const PJRT_Executable *executable;
  /** @brief Output number of outputs. */
  size_t num_outputs;
  /** @brief Output array of memory kind C string pointers. */
  const char *const *memory_kinds;
  /** @brief Output array of memory kind string lengths. */
  const size_t *memory_kind_sizes;
} PJRT_Executable_OutputMemoryKinds_Args;

/**
 * @brief Argument structure for `PJRT_Device_GetAttributes` (`56` bytes).
 */
typedef struct PJRT_Device_GetAttributes_Args {
  /** @brief Size of this struct (`56`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Device handle. */
  const PJRT_Device *device;
  /** @brief Output pointer to `PJRT_NamedValue` array. */
  const void *attributes;
  /** @brief Output number of attributes. */
  size_t num_attributes;
  /** @brief Reserved word 0. */
  size_t reserved0;
  /** @brief Reserved word 1. */
  size_t reserved1;
} PJRT_Device_GetAttributes_Args;

/**
 * @brief Argument structure for `PJRT_Event_Create` (`24` bytes).
 */
typedef struct PJRT_Event_Create_Args {
  /** @brief Size of this struct (`24`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Output created `PJRT_Event*`. */
  PJRT_Event *event;
} PJRT_Event_Create_Args;

/**
 * @brief Argument structure for `PJRT_Event_Destroy` (`24` bytes).
 */
typedef struct PJRT_Event_Destroy_Args {
  /** @brief Size of this struct (`24`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Event handle to destroy. */
  PJRT_Event *event;
} PJRT_Event_Destroy_Args;

/**
 * @brief Argument structure for `PJRT_Event_IsReady` (`32` bytes).
 */
typedef struct PJRT_Event_IsReady_Args {
  /** @brief Size of this struct (`32`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Event handle. */
  PJRT_Event *event;
  /** @brief Output readiness flag (`bool`). */
  bool is_ready;
  /** @brief Explicit 7-byte alignment padding. */
  uint8_t pad[7];
} PJRT_Event_IsReady_Args;

/**
 * @brief Argument structure for `PJRT_Event_Await` (`24` bytes).
 */
typedef struct PJRT_Event_Await_Args {
  /** @brief Size of this struct (`24`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Event handle. */
  PJRT_Event *event;
} PJRT_Event_Await_Args;

/**
 * @brief Argument structure for `PJRT_Event_Set` (`48` bytes).
 */
typedef struct PJRT_Event_Set_Args {
  /** @brief Size of this struct (`48`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Event handle. */
  PJRT_Event *event;
  /** @brief Status code (`0` = OK). */
  int32_t error_code;
  /** @brief Explicit 4-byte alignment padding. */
  int32_t pad;
  /** @brief Optional error message pointer. */
  const char *error_message;
  /** @brief Error message byte length. */
  size_t error_message_size;
} PJRT_Event_Set_Args;

/**
 * @brief Callback function pointer for `PJRT_Event_OnReady`
 * (`PJRT_Event_OnReadyCallback`).
 */
typedef void (*PJRT_Event_OnReadyCallback)(PJRT_Error *error, void *user_arg);

/**
 * @brief Argument structure for `PJRT_Event_OnReady` (`40` bytes).
 */
typedef struct PJRT_Event_OnReady_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Event handle. */
  PJRT_Event *event;
  /** @brief Readiness callback function pointer. */
  PJRT_Event_OnReadyCallback callback;
  /** @brief Opaque user argument passed to `callback`. */
  void *user_arg;
} PJRT_Event_OnReady_Args;

/**
 * @brief Argument structure for `PJRT_Executable_OptimizedProgram` (`32`
 * bytes).
 */
typedef struct PJRT_Executable_OptimizedProgram_Args {
  /** @brief Size of this struct (`32`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Executable handle. */
  const PJRT_Executable *executable;
  /** @brief Output `PJRT_Program*` buffer. */
  PJRT_Program *program;
} PJRT_Executable_OptimizedProgram_Args;

/**
 * @brief Argument structure for `PJRT_Buffer_GetMemoryLayout` (`104` bytes).
 */
typedef struct PJRT_Buffer_GetMemoryLayout_Args {
  /** @brief Size of this struct (`104`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Buffer handle. */
  const PJRT_Buffer *buffer;
  /** @brief Inline `PJRT_Buffer_MemoryLayout` words (`10` words = `80` bytes).
   */
  size_t layout_words[10];
} PJRT_Buffer_GetMemoryLayout_Args;

/**
 * @brief Argument structure for `PJRT_Executable_Serialize` (`56` bytes).
 */
typedef struct PJRT_Executable_Serialize_Args {
  /** @brief Size of this struct (`56`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Executable handle. */
  const PJRT_Executable *executable;
  /** @brief Output serialized bytes pointer. */
  const char *serialized_bytes;
  /** @brief Output serialized byte length. */
  size_t serialized_bytes_size;
  /** @brief Output `PJRT_SerializedExecutable*`. */
  PJRT_SerializedExecutable *serialized_executable;
  /** @brief Output deleter callback (`void (*)(PJRT_SerializedExecutable*)`).
   */
  void (*serialized_executable_deleter)(PJRT_SerializedExecutable *exec);
} PJRT_Executable_Serialize_Args;

/**
 * @brief Argument structure for `PJRT_LoadedExecutable_Destroy` (`24` bytes).
 */
typedef struct PJRT_LoadedExecutable_Destroy_Args {
  /** @brief Size of this struct (`24`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Loaded executable handle to destroy. */
  PJRT_LoadedExecutable *executable;
} PJRT_LoadedExecutable_Destroy_Args;

/**
 * @brief Argument structure for `PJRT_LoadedExecutable_GetExecutable` (`32`
 * bytes).
 */
typedef struct PJRT_LoadedExecutable_GetExecutable_Args {
  /** @brief Size of this struct (`32`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Loaded executable handle. */
  const PJRT_LoadedExecutable *loaded_executable;
  /** @brief Output `PJRT_Executable*`. */
  PJRT_Executable *executable;
} PJRT_LoadedExecutable_GetExecutable_Args;

/**
 * @brief Argument structure for `PJRT_LoadedExecutable_Execute` (`80` bytes).
 */
typedef struct PJRT_LoadedExecutable_Execute_Args {
  /** @brief Size of this struct (`80`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Loaded executable handle. */
  PJRT_LoadedExecutable *executable;
  /** @brief Optional `PJRT_ExecuteOptions*`. */
  const void *options;
  /** @brief Per-device input buffer lists (`[num_devices][num_args]`). */
  const PJRT_Buffer *const *const *argument_lists;
  /** @brief Number of devices to execute across. */
  size_t num_devices;
  /** @brief Number of input arguments per device. */
  size_t num_args;
  /** @brief Per-device output buffer lists (`[num_devices][num_outputs]`). */
  PJRT_Buffer **const *output_lists;
  /** @brief Optional per-device completion events array (`PJRT_Event**`). */
  PJRT_Event **device_complete_events;
  /** @brief Optional single target `PJRT_Device*` for portable execution. */
  PJRT_Device *execute_device;
} PJRT_LoadedExecutable_Execute_Args;

/**
 * @brief Argument structure for `PJRT_Executable_DeserializeAndLoad` (`64`
 * bytes).
 */
typedef struct PJRT_Executable_DeserializeAndLoad_Args {
  /** @brief Size of this struct (`64`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Client handle. */
  PJRT_Client *client;
  /** @brief Serialized executable bytes pointer. */
  const char *serialized_executable;
  /** @brief Byte length of `serialized_executable`. */
  size_t serialized_executable_size;
  /** @brief Output loaded `PJRT_LoadedExecutable*`. */
  PJRT_LoadedExecutable *loaded_executable;
  /** @brief Optional overridden compile options pointer. */
  const char *overridden_serialized_compile_options;
  /** @brief Byte length of `overridden_serialized_compile_options`. */
  size_t overridden_serialized_compile_options_size;
} PJRT_Executable_DeserializeAndLoad_Args;

/**
 * @brief Argument structure for `PJRT_LoadedExecutable_Fingerprint` (`40`
 * bytes).
 */
typedef struct PJRT_LoadedExecutable_Fingerprint_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Loaded executable handle. */
  PJRT_LoadedExecutable *executable;
  /** @brief Output fingerprint string pointer. */
  const char *executable_fingerprint;
  /** @brief Output fingerprint string length. */
  size_t executable_fingerprint_size;
} PJRT_LoadedExecutable_Fingerprint_Args;

/**
 * @brief Argument structure for `PJRT_Buffer_Destroy` (`24` bytes).
 */
typedef struct PJRT_Buffer_Destroy_Args {
  /** @brief Size of this struct (`24`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Buffer handle to destroy. */
  PJRT_Buffer *buffer;
} PJRT_Buffer_Destroy_Args;

/**
 * @brief Argument structure for `PJRT_Buffer_ElementType` (`32` bytes).
 */
typedef struct PJRT_Buffer_ElementType_Args {
  /** @brief Size of this struct (`32`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Buffer handle. */
  const PJRT_Buffer *buffer;
  /** @brief Output element type (`PJRT_Buffer_Type`). */
  int32_t type;
  /** @brief Explicit 4-byte alignment padding. */
  int32_t pad;
} PJRT_Buffer_ElementType_Args;

/**
 * @brief Argument structure for `PJRT_Buffer_Dimensions` (`40` bytes).
 */
typedef struct PJRT_Buffer_Dimensions_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Buffer handle. */
  const PJRT_Buffer *buffer;
  /** @brief Output dimension sizes array pointer. */
  const int64_t *dims;
  /** @brief Output number of dimensions. */
  size_t num_dims;
} PJRT_Buffer_Dimensions_Args;

/**
 * @brief Argument structure for `PJRT_Buffer_CopyToDevice` (`40` bytes).
 */
typedef struct PJRT_Buffer_CopyToDevice_Args {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Source buffer handle. */
  PJRT_Buffer *buffer;
  /** @brief Destination device handle. */
  PJRT_Device *dst_device;
  /** @brief Output copied `PJRT_Buffer*`. */
  PJRT_Buffer *dst_buffer;
} PJRT_Buffer_CopyToDevice_Args;

/**
 * @brief Argument structure for `PJRT_Buffer_ToHostBuffer` (`56` bytes).
 */
typedef struct PJRT_Buffer_ToHostBuffer_Args {
  /** @brief Size of this struct (`56`). */
  size_t struct_size;
  /** @brief Extension pointer (`NULL`). */
  PJRT_Extension_Base *extension_start;
  /** @brief Source `PJRT_Buffer*`. */
  const PJRT_Buffer *src;
  /** @brief Optional host layout pointer. */
  const void *host_layout;
  /** @brief Destination host buffer pointer (or `NULL` to query size). */
  void *dst;
  /** @brief Destination host buffer byte size. */
  size_t dst_size;
  /** @brief Output completion `PJRT_Event*`. */
  PJRT_Event *event;
} PJRT_Buffer_ToHostBuffer_Args;

/**
 * @brief Complete 144-byte `LibtpuSdk_Api` vtable returned by
 * `GetLibtpuSdkApi()`.
 */
typedef struct LibtpuSdk_Api {
  /** @brief Packed version header (`0x0000000100000000`). */
  uint64_t version_header;
  /** @brief Slot 1 (`0x08`): `LibtpuSdk_Error_GetMessage` (returns `void`). */
  LibtpuSdkVoidFn error_get_message;
  /** @brief Slot 2 (`0x10`): `LibtpuSdk_Error_Destroy` (returns `void`). */
  LibtpuSdkVoidFn error_destroy;
  /** @brief Slot 3 (`0x18`): `LibtpuSdk_Error_GetCode`. */
  LibtpuSdkFn error_get_code;
  /** @brief Slot 4 (`0x20`): `LibtpuSdk_Client_Create`. */
  LibtpuSdkFn client_create;
  /** @brief Slot 5 (`0x28`): `LibtpuSdk_Client_Destroy`. */
  LibtpuSdkFn client_destroy;
  /** @brief Slot 6 (`0x30`): `LibtpuSdk_GetChipCoordinates`. */
  LibtpuSdkFn get_chip_coordinates;
  /** @brief Slot 7 (`0x38`): `LibtpuSdk_GetHostName`. */
  LibtpuSdkFn get_host_name;
  /** @brief Slot 8 (`0x40`): `LibtpuSdk_GetChipIndex`. */
  LibtpuSdkFn get_chip_index;
  /** @brief Slot 9 (`0x48`): `LibtpuSdk_GetCartesianCoordinates`. */
  LibtpuSdkFn get_cartesian_coordinates;
  /** @brief Slot 10 (`0x50`): `LibtpuSdk_GetMetric`. */
  LibtpuSdkFn get_metric;
  /** @brief Slot 11 (`0x58`): `LibtpuSdk_GetMetricDescription`. */
  LibtpuSdkFn get_metric_description;
  /** @brief Slot 12 (`0x60`): `LibtpuSdk_GetMetricValues`. */
  LibtpuSdkFn get_metric_values;
  /** @brief Slot 13 (`0x68`): `LibtpuSdk_GetRuntimeStatus`. */
  LibtpuSdkFn get_runtime_status;
  /** @brief Slot 14 (`0x70`): `LibtpuSdk_RuntimeStatus_GetCoreStateSummary`. */
  LibtpuSdkFn runtime_status_get_core_state_summary;
  /** @brief Slot 15 (`0x78`): `LibtpuSdk_RuntimeStatus_Destroy`. */
  LibtpuSdkFn runtime_status_destroy;
  /** @brief Slot 16 (`0x80`): `LibtpuSdk_RegisterDebugCallback`. */
  LibtpuSdkFn register_debug_callback;
  /** @brief Slot 17 (`0x88`): `LibtpuSdk_UnregisterDebugCallback`. */
  LibtpuSdkFn unregister_debug_callback;
} LibtpuSdk_Api;

/**
 * @brief Argument structure for `LibtpuSdk_Client_Create` and
 * `LibtpuSdk_Client_Destroy` (`8` bytes).
 */
typedef struct LibtpuSdk_Client_Args {
  /** @brief In/out `LibtpuSdk_Client*` handle. */
  LibtpuSdk_Client *client;
} LibtpuSdk_Client_Args;

/**
 * @brief C-ABI structure representing a chip coordinate entry in
 * `LibtpuSdk_GetChipCoordinates` (`48` bytes).
 */
typedef struct LibtpuSdk_ChipLocationToCoordinates {
  /** @brief Pointer to null-terminated UTF-8 hostname string. */
  const char *host_name_ptr;
  /** @brief Byte length of hostname string. */
  size_t host_name_len;
  /** @brief Zero-based chip index on the host tray. */
  int32_t chip_index;
  /** @brief Explicit 4-byte alignment padding. */
  int32_t pad;
  /** @brief 3D Cartesian coordinates `[x, y, z]`. */
  int64_t coordinates[3];
} LibtpuSdk_ChipLocationToCoordinates;

/**
 * @brief Argument structure for `LibtpuSdk_GetChipCoordinates` (`24` bytes).
 */
typedef struct LibtpuSdk_GetChipCoordinates_Args {
  /** @brief Input `LibtpuSdk_Client*`. */
  LibtpuSdk_Client *client;
  /** @brief Output pointer to `LibtpuSdk_ChipLocationToCoordinates` array. */
  const LibtpuSdk_ChipLocationToCoordinates *coords;
  /** @brief Output number of chip coordinate entries. */
  size_t num_coords;
} LibtpuSdk_GetChipCoordinates_Args;

/**
 * @brief Argument structure for `LibtpuSdk_GetMetric` (`24` bytes).
 */
typedef struct LibtpuSdk_GetMetric_Args {
  /** @brief Input `LibtpuSdk_Client*`. */
  LibtpuSdk_Client *client;
  /** @brief Null-terminated metric name string. */
  const char *metric_name;
  /** @brief Output `LibtpuSdk_Metric*` handle. */
  LibtpuSdk_Metric *metric;
} LibtpuSdk_GetMetric_Args;

/**
 * @brief Argument structure for `LibtpuSdk_GetMetricValues` (`24` bytes).
 */
typedef struct LibtpuSdk_GetMetricValues_Args {
  /** @brief Input `const LibtpuSdk_Metric*` handle. */
  const LibtpuSdk_Metric *metric;
  /** @brief Output pointer to array of metric values. */
  const char *const *values;
  /** @brief Output number of values. */
  size_t num_values;
} LibtpuSdk_GetMetricValues_Args;

/**
 * @brief Argument structure for `LibtpuSdk_GetRuntimeStatus` (`24` bytes).
 */
typedef struct LibtpuSdk_GetRuntimeStatus_Args {
  /** @brief Input `LibtpuSdk_Client*`. */
  LibtpuSdk_Client *client;
  /** @brief Flag requesting HLO metadata. */
  bool include_hlo_info;
  /** @brief Explicit 7-byte alignment padding. */
  uint8_t pad[7];
  /** @brief Output `LibtpuSdk_RuntimeStatus*`. */
  LibtpuSdk_RuntimeStatus *status;
} LibtpuSdk_GetRuntimeStatus_Args;

/**
 * @brief Argument structure for `LibtpuSdk_RuntimeStatus_GetCoreStateSummary`
 * (`24` bytes).
 */
typedef struct LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args {
  /** @brief Input `LibtpuSdk_RuntimeStatus*`. */
  const LibtpuSdk_RuntimeStatus *status;
  /** @brief Output summary string pointer. */
  const char *summary;
  /** @brief Output summary string byte length. */
  size_t summary_size;
} LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args;

/**
 * @brief Argument structure for `LibtpuSdk_RuntimeStatus_Destroy` (`8` bytes).
 */
typedef struct LibtpuSdk_RuntimeStatus_Destroy_Args {
  /** @brief `LibtpuSdk_RuntimeStatus*` to destroy. */
  LibtpuSdk_RuntimeStatus *status;
} LibtpuSdk_RuntimeStatus_Destroy_Args;

/**
 * @brief Argument structure for `LibtpuSdk_RegisterDebugCallback` (`32` bytes).
 */
typedef struct LibtpuSdk_RegisterDebugCallback_Args {
  /** @brief Input `LibtpuSdk_Client*`. */
  LibtpuSdk_Client *client;
  /** @brief Callback function pointer (`void (*)(void*)`). */
  const void *callback;
  /** @brief Opaque user context pointer. */
  void *user_data;
  /** @brief Output registered callback ID (`int64_t`). */
  int64_t callback_id;
} LibtpuSdk_RegisterDebugCallback_Args;

/**
 * @brief Argument structure for `LibtpuSdk_UnregisterDebugCallback` (`16`
 * bytes).
 */
typedef struct LibtpuSdk_UnregisterDebugCallback_Args {
  /** @brief Input `LibtpuSdk_Client*`. */
  LibtpuSdk_Client *client;
  /** @brief Callback ID to unregister. */
  int64_t callback_id;
} LibtpuSdk_UnregisterDebugCallback_Args;

/**
 * @brief Invokes all registered debug callbacks on `client`
 * (`LibtpuSdk_TriggerDebugCallbacks`).
 * @param client Pointer to `LibtpuSdk_Client`.
 * @return Number of invoked callbacks (`0` if `client` is `NULL`).
 */
size_t LibtpuSdk_TriggerDebugCallbacks(const LibtpuSdk_Client *client);

/**
 * @brief Status structure returned by `libtpu_rs_*` C FFI functions.
 */
typedef struct LibTpuCStatus {
  /** @brief Status code (`0` = OK, non-zero = `PjrtErrorCode`). */
  int32_t code;
  /** @brief Heap-allocated null-terminated error message (or `NULL` when `code
   * == 0`). */
  char *message;
} LibTpuCStatus;

/**
 * @brief Returns a pointer to the static 1,144-byte `PJRT_Api` vtable.
 * @return Pointer to the static `PJRT_Api` table.
 */
const PJRT_Api *GetPjrtApi(void);

/**
 * @brief Returns a pointer to the static 144-byte `LibtpuSdk_Api` vtable.
 * @return Pointer to the static `LibtpuSdk_Api` table.
 */
const LibtpuSdk_Api *GetLibtpuSdkApi(void);

/**
 * @brief Returns a pointer to the global `TpuTopology` singleton
 * (`TpuUtil_GetTopologyPtr`).
 * @return Pointer to the singleton `TpuTopology`.
 */
const TpuTopology *TpuUtil_GetTopologyPtr(void);

/**
 * @brief Returns the XLA minor-dimension alignment pad size
 * (`TpuUtil_GetXlaPadSizeFromTpuTopology`).
 * @return XLA pad size (`128`).
 */
int32_t TpuUtil_GetXlaPadSizeFromTpuTopology(void);

/**
 * @brief Returns the number of logical devices per host
 * (`TpuTopology_LogicalDevicesPerHost`).
 * @param topo Pointer to `TpuTopology`.
 * @param core_type Core type discriminant (`0` = `kTensorCore`, `1` =
 * `kSparseCoreV0`).
 * @return Number of logical devices per host.
 */
int32_t TpuTopology_LogicalDevicesPerHost(const TpuTopology *topo,
                                          int32_t core_type);

/**
 * @brief Returns the X chip mesh bound (`TpuTopology_ChipBounds_X`).
 * @param topo Pointer to `TpuTopology`.
 * @return X mesh bound.
 */
int32_t TpuTopology_ChipBounds_X(const TpuTopology *topo);

/**
 * @brief Returns the Y chip mesh bound (`TpuTopology_ChipBounds_Y`).
 * @param topo Pointer to `TpuTopology`.
 * @return Y mesh bound.
 */
int32_t TpuTopology_ChipBounds_Y(const TpuTopology *topo);

/**
 * @brief Returns the Z chip mesh bound (`TpuTopology_ChipBounds_Z`).
 * @param topo Pointer to `TpuTopology`.
 * @return Z mesh bound.
 */
int32_t TpuTopology_ChipBounds_Z(const TpuTopology *topo);

/**
 * @brief Returns the `TpuVersionEnum` integer of the topology
 * (`TpuTopology_Version`).
 * @param topo Pointer to `TpuTopology`.
 * @return TPU version integer.
 */
int32_t TpuTopology_Version(const TpuTopology *topo);

/**
 * @brief Parameter structure for `TFNPD_InitPlugin` (`40` bytes).
 */
typedef struct TFNPD_PluginParams {
  /** @brief Size of this struct (`40`). */
  size_t struct_size;
  /** @brief Private pointer (`NULL`). */
  void *priv_data;
  /** @brief Output device type C string (`"TPU"`). */
  const char *device_type;
  /** @brief Output compilation device name C string (`"TPU_REPLICATED_CORE"`).
   */
  const char *compilation_device_name;
  /** @brief Output priority (`55`). */
  int32_t priority;
  /** @brief Output pluggable device flag (`true`). */
  bool is_pluggable_device;
  /** @brief Output on-demand compile flag (`true`). */
  bool use_pjrt_on_demand_compile;
  /** @brief Explicit 2-byte padding. */
  uint16_t pad;
} TFNPD_PluginParams;

/**
 * @brief Parameter structure for `TfTpu_GetTpuPartitionedCallParams` (`24`
 * bytes).
 */
typedef struct TpuPartitionedCall_Params {
  /** @brief Input shape optimization flag. */
  bool input_shape_opt;
  /** @brief Tensor packing flag. */
  bool group_tensors_for_packing;
  /** @brief Explicit 2-byte padding. */
  uint16_t pad0;
  /** @brief Minimum input tensors for packing. */
  int32_t minimum_input_tensors_packing;
  /** @brief Minimum output tensors for packing. */
  int32_t minimum_output_tensors_packing;
  /** @brief Automatic XLA input sharding flag. */
  bool enable_auto_xla_input_sharding;
  /** @brief Explicit 3-byte padding. */
  uint8_t pad1[3];
  /** @brief Automatic XLA input sharding dimension. */
  int32_t auto_xla_input_sharding_dim;
  /** @brief Scoped allocator flag. */
  bool scoped_allocator_for_tpu_anned_call;
  /** @brief Explicit 3-byte padding. */
  uint8_t pad2[3];
} TpuPartitionedCall_Params;

/**
 * @brief XLA shape structure for `TpuTransferManager_GetInfeedLayout` (`96`
 * bytes).
 */
typedef struct XLA_Shape {
  /** @brief Element type (`int32_t`). */
  int32_t element_type;
  /** @brief Rank (`int32_t`). */
  int32_t rank;
  /** @brief Dimensions array pointer. */
  const int64_t *dimensions;
  /** @brief Minor-to-major ordering (`int64_t[8]`). */
  int64_t minor_to_major[8];
  /** @brief Tile dimensions (`int64_t[2]`). */
  int64_t tile_dims[2];
} XLA_Shape;

/**
 * @brief Parameter structure for `TpuEmbeddingEngine_ExecutePartitioner` (`56`
 * bytes).
 */
typedef struct TpuEmbeddingEngine_Params {
  /** @brief Size of this struct (`56`). */
  size_t struct_size;
  /** @brief Reserved private pointer (`NULL`). */
  void *priv_data;
  /** @brief Input config or table bytes pointer. */
  const char *input_bytes;
  /** @brief Input byte length or parameter count. */
  size_t input_size;
  /** @brief Output size pointer. */
  size_t *output_size;
  /** @brief Output buffer pointer. */
  char **output_bytes;
  /** @brief Optional status pointer. */
  TpuStatus *status;
} TpuEmbeddingEngine_Params;

/**
 * @brief Initializes the legacy TensorFlow TPU driver (`TfTpu_Initialize`).
 * @param init_library Whether to initialize the underlying driver library.
 * @param num_args Number of CLI string arguments in `args`.
 * @param args Optional array of CLI string pointers.
 */
void TfTpu_Initialize(bool init_library, int32_t num_args,
                      const char *const *args);

/**
 * @brief Initializes the TensorFlow NextPluggableDevice plugin
 * (`TFNPD_InitPlugin`).
 * @param params Pointer to `TFNPD_PluginParams` structure.
 * @param status Optional `TpuStatus*` pointer.
 */
void TFNPD_InitPlugin(TFNPD_PluginParams *params, TpuStatus *status);

/**
 * @brief Queries TPU partitioned call runtime parameters
 * (`TfTpu_GetTpuPartitionedCallParams`).
 * @param params Pointer to `TpuPartitionedCall_Params` structure.
 * @return `0` on success, `-1` if `params` is null.
 */
int32_t TfTpu_GetTpuPartitionedCallParams(TpuPartitionedCall_Params *params);

/**
 * @brief Computes on-device TPU infeed layout
 * (`TpuTransferManager_GetInfeedLayout`).
 * @param host_shape Input host `XLA_Shape` pointer.
 * @param infeed_shape Output on-device `XLA_Shape` pointer.
 * @return `0` on success, `-1` if either pointer is null.
 */
int32_t TpuTransferManager_GetInfeedLayout(const XLA_Shape *host_shape,
                                           XLA_Shape *infeed_shape);

/**
 * @brief Queries SparseCore max IDs and uniques per partition
 * (`SparseCore_GetMaxIdsAndUniques`).
 * @param table_name Null-terminated embedding table name.
 * @param num_samples_per_sparse_core Batch samples per SparseCore.
 * @param feature_width Embedding feature width.
 * @param max_ids_per_partition Output pointer receiving max IDs per partition.
 * @param max_unique_ids_per_partition Output pointer receiving max unique IDs
 * per partition.
 * @param status Optional `TpuStatus*` pointer.
 * @return `128` on success, `-1` if any required pointer is null.
 */
int64_t SparseCore_GetMaxIdsAndUniques(const char *table_name,
                                       int64_t num_samples_per_sparse_core,
                                       int64_t feature_width,
                                       int64_t *max_ids_per_partition,
                                       int64_t *max_unique_ids_per_partition,
                                       TpuStatus *status);

/**
 * @brief Executes the TPU embedding partitioner
 * (`TpuEmbeddingEngine_ExecutePartitioner`).
 * @param params Pointer to `TpuEmbeddingEngine_Params` structure.
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t
TpuEmbeddingEngine_ExecutePartitioner(TpuEmbeddingEngine_Params *params);

/**
 * @brief TensorFlow kernel initialization hook (`TF_InitKernel`).
 * @return `0` on success.
 */
int32_t TF_InitKernel(void);

/**
 * @brief Configures global TPU topology array state
 * (`SetGlobalTPUArrayOp_DoWork`).
 * @param topology_len Length of `topology_proto` in bytes.
 * @param topology_proto Serialized `TpuTopologyArgsProto` bytes.
 * @param status Optional `TpuStatus*` pointer.
 * @return `0` on success, `-1` if `topology_proto` is null when `topology_len >
 * 0`.
 */
int32_t SetGlobalTPUArrayOp_DoWork(size_t topology_len,
                                   const char *topology_proto,
                                   TpuStatus *status);

/**
 * @brief Allocates a `TfTpuOrdinalSelector` (`TfTpuOrdinalSelector_Create`).
 * @param selector_out Output pointer receiving the allocated
 * `TfTpuOrdinalSelector*`.
 * @param num_cores Number of cores to cycle across.
 */
void TfTpuOrdinalSelector_Create(TfTpuOrdinalSelector **selector_out,
                                 int32_t num_cores);

/**
 * @brief Destroys a `TfTpuOrdinalSelector` (`TfTpuOrdinalSelector_Destroy`).
 * @param selector Pointer to `TfTpuOrdinalSelector` to free.
 */
void TfTpuOrdinalSelector_Destroy(TfTpuOrdinalSelector *selector);

/**
 * @brief Selects the next device ordinal (`TfTpuOrdinalSelector_GetOrdinal`).
 * @param selector Pointer to `TfTpuOrdinalSelector`.
 * @param seed Optional seed pointer.
 * @param req_id Optional output request ID pointer.
 * @return Selected device ordinal (`0` if `selector` is null).
 */
int64_t TfTpuOrdinalSelector_GetOrdinal(TfTpuOrdinalSelector *selector,
                                        uint64_t *seed, int64_t *req_id);

/**
 * @brief Dequeues a reserved ordinal from the core selector
 * (`TfTpuOrdinalSelector_DequeueFromCoreSelector`).
 * @param selector Pointer to `TfTpuOrdinalSelector`.
 * @param device_ordinal Device ordinal.
 * @param req_id Request ID.
 */
void TfTpuOrdinalSelector_DequeueFromCoreSelector(
    TfTpuOrdinalSelector *selector, int32_t device_ordinal, int64_t req_id);

/**
 * @brief Allocates a `TpuStatus` with explicit code and message
 * (`TpuStatus_Create`).
 * @param code Status code (`0` = OK).
 * @param msg Optional null-terminated message string.
 * @return Heap-allocated `TpuStatus*`.
 */
TpuStatus *TpuStatus_Create(int32_t code, const char *msg);

/**
 * @brief Mutates an existing `TpuStatus` (`TpuStatus_Set`).
 * @param status Pointer to `TpuStatus`.
 * @param code Status code.
 * @param msg Message buffer pointer.
 * @param len Length of `msg` in bytes (or negative for null-terminated).
 */
void TpuStatus_Set(TpuStatus *status, int32_t code, const char *msg,
                   int32_t len);

/**
 * @brief Returns the number of logical devices per chip
 * (`TpuTopology_LogicalDevicesPerChip`).
 * @param topo Pointer to `TpuTopology`.
 * @param core_type Core type discriminant.
 * @return Number of logical devices per chip.
 */
int32_t TpuTopology_LogicalDevicesPerChip(const TpuTopology *topo,
                                          int32_t core_type);

/**
 * @brief Returns the total number of hosts in the TPU slice
 * (`TpuTopology_HostCount`).
 * @param topo Pointer to `TpuTopology`.
 * @return Host count.
 */
int32_t TpuTopology_HostCount(const TpuTopology *topo);

/**
 * @brief Returns the number of physical chips per host
 * (`TpuTopology_ChipsPerHost`).
 * @param topo Pointer to `TpuTopology`.
 * @return Chips per host.
 */
int32_t TpuTopology_ChipsPerHost(const TpuTopology *topo);

/**
 * @brief Checks whether chip `(x, y, z)` exists in the topology
 * (`TpuTopology_HasChip`).
 * @param topo Pointer to `TpuTopology`.
 * @param x X chip coordinate.
 * @param y Y chip coordinate.
 * @param z Z chip coordinate.
 * @return `true` if the chip exists, `false` otherwise.
 */
bool TpuTopology_HasChip(const TpuTopology *topo, int32_t x, int32_t y,
                         int32_t z);

/**
 * @brief Looks up a `TpuCoreLocation` by global core ID
 * (`TpuTopology_CoreForId`).
 * @param topo Pointer to `TpuTopology`.
 * @param core_type Core type discriminant.
 * @param id Global core ID.
 * @return Pointer to `TpuCoreLocation` or `NULL`.
 */
const TpuCoreLocation *TpuTopology_CoreForId(const TpuTopology *topo,
                                             int32_t core_type, int32_t id);

/**
 * @brief Looks up a `TpuCoreLocation` by `(core_type, x, y, z, index)`
 * (`TpuTopology_Core`).
 * @param topo Pointer to `TpuTopology`.
 * @param core_type Core type discriminant.
 * @param x X chip coordinate.
 * @param y Y chip coordinate.
 * @param z Z chip coordinate.
 * @param index Core index on chip.
 * @return Pointer to `TpuCoreLocation` or `NULL`.
 */
const TpuCoreLocation *TpuTopology_Core(const TpuTopology *topo,
                                        int32_t core_type, int32_t x, int32_t y,
                                        int32_t z, int32_t index);

/**
 * @brief Returns the total number of cores in the topology
 * (`TpuTopology_NumCores`).
 * @param topo Pointer to `TpuTopology`.
 * @param core_type Core type discriminant.
 * @return Total core count.
 */
int32_t TpuTopology_NumCores(const TpuTopology *topo, int32_t core_type);

/**
 * @brief Copies pointers to all `TpuCoreLocation` entries into `cores_out`
 * (`TpuTopology_Cores`).
 * @param topo Pointer to `TpuTopology`.
 * @param core_type Core type discriminant.
 * @param cores_out Caller-allocated array of `const TpuCoreLocation*`.
 */
void TpuTopology_Cores(const TpuTopology *topo, int32_t core_type,
                       const TpuCoreLocation **cores_out);

/**
 * @brief Returns the host ID for host coordinates `(x, y, z)`
 * (`TpuTopology_IdForHost`).
 * @param topo Pointer to `TpuTopology`.
 * @param x Host X coordinate.
 * @param y Host Y coordinate.
 * @param z Host Z coordinate.
 * @return Host ID (`>= 0`) or `-1` if out of bounds.
 */
int32_t TpuTopology_IdForHost(const TpuTopology *topo, int32_t x, int32_t y,
                              int32_t z);

/**
 * @brief Queries available core count (`TpuTopology_AvailableCoreCount`).
 * @param mesh_state Pointer to `TpuMeshState`.
 * @param core_type Core type discriminant (`0` = TensorCore, `1` = SparseCore).
 * @return Available core count (`-1` on invalid `core_type`).
 */
int32_t TpuTopology_AvailableCoreCount(const TpuMeshState *mesh_state,
                                       int32_t core_type);

/**
 * @brief Queries available cores per chip
 * (`TpuTopology_AvailableCoresPerChip`).
 * @param core_type Core type discriminant (`0` = TensorCore, `1` = SparseCore).
 * @return Cores per chip (`-1` on invalid `core_type`).
 */
int32_t TpuTopology_AvailableCoresPerChip(int32_t core_type);

/**
 * @brief Queries SparseCore count per logical device
 * (`TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice`).
 * @param core_type Core type discriminant (`0` = TensorCore, `1` = SparseCore).
 * @return SparseCore count (`-1` on invalid `core_type`).
 */
int32_t
TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice(int32_t core_type);

/**
 * @brief Writes the `(x, y, z)` chip coordinates of `loc`
 * (`TpuCoreLocation_ChipCoordinates`).
 * @param loc Pointer to `TpuCoreLocation`.
 * @param x Optional output pointer for X coordinate.
 * @param y Optional output pointer for Y coordinate.
 * @param z Optional output pointer for Z coordinate.
 */
void TpuCoreLocation_ChipCoordinates(const TpuCoreLocation *loc, int32_t *x,
                                     int32_t *y, int32_t *z);

/**
 * @brief Writes the `(x, y, z)` host coordinates of `loc`
 * (`TpuCoreLocation_HostCoordinates`).
 * @param loc Pointer to `TpuCoreLocation`.
 * @param x Optional output pointer for X coordinate.
 * @param y Optional output pointer for Y coordinate.
 * @param z Optional output pointer for Z coordinate.
 */
void TpuCoreLocation_HostCoordinates(const TpuCoreLocation *loc, int32_t *x,
                                     int32_t *y, int32_t *z);

/**
 * @brief Returns the core index on chip (`TpuCoreLocation_Index`).
 * @param loc Pointer to `TpuCoreLocation`.
 * @return Core index on chip.
 */
int32_t TpuCoreLocation_Index(const TpuCoreLocation *loc);

/**
 * @brief Returns the global core ID (`TpuCoreLocation_Id`).
 * @param loc Pointer to `TpuCoreLocation`.
 * @return Global core ID.
 */
int32_t TpuCoreLocation_Id(const TpuCoreLocation *loc);

/**
 * @brief Returns the host ID (`TpuHostLocation_Id`).
 * @param host Pointer to `TpuHostLocation`.
 * @return Host ID.
 */
int32_t TpuHostLocation_Id(const TpuHostLocation *host);

/**
 * @brief Returns the number of cores on `host` (`TpuHostLocation_NumCores`).
 * @param host Pointer to `TpuHostLocation`.
 * @param core_type Core type discriminant.
 * @return Number of cores on host.
 */
int32_t TpuHostLocation_NumCores(const TpuHostLocation *host,
                                 int32_t core_type);

/**
 * @brief Copies core pointers on `host` into `cores_out`
 * (`TpuHostLocation_Cores`).
 * @param host Pointer to `TpuHostLocation`.
 * @param core_type Core type discriminant.
 * @param cores_out Caller-allocated array of `const TpuCoreLocation*`.
 */
void TpuHostLocation_Cores(const TpuHostLocation *host, int32_t core_type,
                           const TpuCoreLocation **cores_out);

/**
 * @brief Allocates a `TpuMeshState` (`TpuMeshState_Create`).
 * @return Heap-allocated `TpuMeshState*`.
 */
TpuMeshState *TpuMeshState_Create(void);

/**
 * @brief Frees a `TpuMeshState` (`TpuMeshState_Free`).
 * @param state Pointer to `TpuMeshState` to free.
 */
void TpuMeshState_Free(TpuMeshState *state);

/**
 * @brief Returns the internal mesh common state pointer
 * (`TpuMeshState_MeshCommonState`).
 * @param state Pointer to `TpuMeshState`.
 * @return Mesh common state pointer.
 */
const void *TpuMeshState_MeshCommonState(const TpuMeshState *state);

/**
 * @brief Creates a `TpuEmbeddingEngineState`
 * (`TpuEmbeddingEngineState_Create`).
 * @return Heap-allocated `TpuEmbeddingEngineState*`.
 */
TpuEmbeddingEngineState *TpuEmbeddingEngineState_Create(void);

/**
 * @brief Frees a `TpuEmbeddingEngineState` (`TpuEmbeddingEngineState_Free`).
 * @param state Pointer to `TpuEmbeddingEngineState` to free.
 */
void TpuEmbeddingEngineState_Free(TpuEmbeddingEngineState *state);

/**
 * @brief Returns internal state pointer (`TpuEmbeddingEngineState_GetState`).
 * @param state Pointer to `TpuEmbeddingEngineState`.
 * @return Internal state pointer.
 */
const void *
TpuEmbeddingEngineState_GetState(const TpuEmbeddingEngineState *state);

/**
 * @brief Configures HBM memory for embedding tables
 * (`TpuEmbeddingEngine_ConfigureMemory`).
 * @param params Pointer to `TpuEmbeddingEngine_ConfigureMemory_Params` (or
 * generic `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_ConfigureMemory(void *params);

/**
 * @brief Collates HBM memory across hosts (`TpuEmbeddingEngine_CollateMemory`).
 * @param params Pointer to `TpuEmbeddingEngine_CollateMemory_Params` (or
 * generic `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_CollateMemory(void *params);

/**
 * @brief Configures host embedding communication
 * (`TpuEmbeddingEngine_ConfigureHost`).
 * @param params Pointer to `TpuEmbeddingEngine_ConfigureHost_Params` (or
 * generic `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_ConfigureHost(void *params);

/**
 * @brief Connects inter-host embedding channels
 * (`TpuEmbeddingEngine_ConnectHosts`).
 * @param params Pointer to `TpuEmbeddingEngine_ConnectHosts_Params` (or generic
 * `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_ConnectHosts(void *params);

/**
 * @brief Finalizes embedding engine configuration
 * (`TpuEmbeddingEngine_Finalize`).
 * @param params Pointer to `TpuEmbeddingEngine_Finalize_Params` (or generic
 * `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_Finalize(void *params);

/**
 * @brief Checks whether embedding engine is initialized
 * (`TpuEmbeddingEngine_IsInitialized`).
 * @param params Pointer to `TpuEmbeddingEngine_IsInitialized_Params` (or
 * generic `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_IsInitialized(void *params);

/**
 * @brief Writes embedding parameters (`TpuEmbeddingEngine_WriteParameters`).
 * @param params Pointer to `TpuEmbeddingEngine_WriteParameters_Params` (or
 * generic `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_WriteParameters(void *params);

/**
 * @brief Reads embedding parameters (`TpuEmbeddingEngine_ReadParameters`).
 * @param params Pointer to `TpuEmbeddingEngine_ReadParameters_Params` (or
 * generic `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_ReadParameters(void *params);

/**
 * @brief Creates tensor batch fixed state
 * (`TpuEmbeddingTensorBatchFixedState_Create`).
 * @param status Optional `TpuStatus*`.
 * @return Heap-allocated `TpuEmbeddingEngineState*`.
 */
TpuEmbeddingEngineState *
TpuEmbeddingTensorBatchFixedState_Create(TpuStatus *status);

/**
 * @brief Destroys tensor batch fixed state
 * (`TpuEmbeddingTensorBatchFixedState_Destroy`).
 * @param state Pointer to `TpuEmbeddingEngineState` to free.
 */
void TpuEmbeddingTensorBatchFixedState_Destroy(TpuEmbeddingEngineState *state);

/**
 * @brief Enqueues sparse embedding lookup batch
 * (`TpuEmbeddingEngine_EnqueueTensorBatch`).
 * @param params Pointer to `TpuEmbeddingEngine_EnqueueTensorBatch_Params` (or
 * generic `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_EnqueueTensorBatch(void *params);

/**
 * @brief Builds XLA computation for receiving embedding activations
 * (`TpuEmbeddingEngine_RecvActivationsComputation`).
 * @param params Pointer to `TpuEmbeddingEngine_Computation_Params` (or generic
 * `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_RecvActivationsComputation(void *params);

/**
 * @brief Builds XLA computation for receiving deduplication data
 * (`TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation`).
 * @param params Pointer to `TpuEmbeddingEngine_Computation_Params` (or generic
 * `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t
TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation(void *params);

/**
 * @brief Builds XLA computation for sending embedding gradients
 * (`TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation`).
 * @param params Pointer to `TpuEmbeddingEngine_Computation_Params` (or generic
 * `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(void *params);

/**
 * @brief Builds XLA computation for deduplication data byte size
 * (`TpuEmbeddingEngine_DedupDataSizeComputation`).
 * @param params Pointer to `TpuEmbeddingEngine_Computation_Params` (or generic
 * `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_DedupDataSizeComputation(void *params);

/**
 * @brief Builds XLA computation for deduplication tuple mask
 * (`TpuEmbeddingEngine_DedupDataTupleMaskComputation`).
 * @param params Pointer to `TpuEmbeddingEngine_Computation_Params` (or generic
 * `void*`).
 * @return `0` on success, `-1` on invalid struct size.
 */
int32_t TpuEmbeddingEngine_DedupDataTupleMaskComputation(void *params);

/**
 * @brief Frees a heap-allocated `char*` array
 * (`TpuConfigurationApi_FreeCharArray`).
 * @param ptr Pointer previously returned by a configuration or embedding API.
 */
void TpuConfigurationApi_FreeCharArray(char *ptr);

/**
 * @brief Function pointer table stored at offset `0x00` of `TpuDriver` (`224`
 * bytes, 28 slots).
 */
typedef struct TpuDriverFn {
  /** @brief Slot 0 (`0x00`): `TpuDriver_Open`. */
  const void *open;
  /** @brief Slot 1 (`0x08`): `TpuDriver_Close`. */
  const void *close;
  /** @brief Slot 2 (`0x10`): `TpuDriver_ComputeLinearizedBytesFromShape`. */
  const void *compute_linearized_bytes_from_shape;
  /** @brief Slot 3 (`0x18`): `TpuDriver_LinearizeShape`. */
  const void *linearize_shape;
  /** @brief Slot 4 (`0x20`): `TpuDriver_DelinearizeShape`. */
  const void *delinearize_shape;
  /** @brief Slot 5 (`0x28`): `TpuDriver_CompileProgram`. */
  const void *compile_program;
  /** @brief Slot 6 (`0x30`): `TpuDriver_CompileProgramFromText`. */
  const void *compile_program_from_text;
  /** @brief Slot 7 (`0x38`): `TpuDriver_GetCompiledProgramShape`. */
  const void *get_compiled_program_shape;
  /** @brief Slot 8 (`0x40`): `TpuDriver_LoadProgram`. */
  const void *load_program;
  /** @brief Slot 9 (`0x48`): `TpuDriver_UnloadProgram`. */
  const void *unload_program;
  /** @brief Slot 10 (`0x50`): `TpuDriver_ExecuteProgram`. */
  const void *execute_program;
  /** @brief Slot 11 (`0x58`): `TpuDriver_AllocateTuple`. */
  const void *allocate_tuple;
  /** @brief Slot 12 (`0x60`): `TpuDriver_Allocate`. */
  const void *allocate;
  /** @brief Slot 13 (`0x68`): `TpuDriver_AllocateShape`. */
  const void *allocate_shape;
  /** @brief Slot 14 (`0x70`): `TpuDriver_Deallocate`. */
  const void *deallocate;
  /** @brief Slot 15 (`0x78`): `TpuDriver_TransferToDevice`. */
  const void *transfer_to_device;
  /** @brief Slot 16 (`0x80`): `TpuDriver_TransferFromDevice`. */
  const void *transfer_from_device;
  /** @brief Slot 17 (`0x88`): `TpuDriver_TransferFromDeviceToDevice`. */
  const void *transfer_from_device_to_device;
  /** @brief Slot 18 (`0x90`): `TpuDriver_CreateEvent`. */
  const void *create_event;
  /** @brief Slot 19 (`0x98`): `TpuDriver_FreeEvent`. */
  const void *free_event;
  /** @brief Slot 20 (`0xa0`): `TpuDriver_AddCallback`. */
  const void *add_callback;
  /** @brief Slot 21 (`0xa8`): `TpuDriver_WaitForEvent`. */
  const void *wait_for_event;
  /** @brief Slot 22 (`0xb0`): `TpuDriver_FreeStatus`. */
  const void *free_status;
  /** @brief Slot 23 (`0xb8`): `TpuDriver_Reset`. */
  const void *reset;
  /** @brief Slot 24 (`0xc0`): `TpuDriver_QuerySystemInfo`. */
  const void *query_system_info;
  /** @brief Slot 25 (`0xc8`): `TpuDriver_FreeSystemInfo`. */
  const void *free_system_info;
  /** @brief Slot 26 (`0xd0`): `TpuDriver_FreeCompiledProgramShape`. */
  const void *free_compiled_program_shape;
  /** @brief Slot 27 (`0xd8`): `TpuDriver_Version`. */
  const void *version;
} TpuDriverFn;

/** @brief Handle representing a `TpuDriver` instance with `fn_table` at offset
 * `0x00`. */
typedef struct TpuDriver {
  /** @brief Pointer to the static 28-slot `TpuDriverFn` vtable. */
  const TpuDriverFn *fn_table;
} TpuDriver;

/** @brief Opaque handle representing a `TpuDriver` synchronization event. */
typedef struct TpuEvent TpuEvent;

/** @brief Opaque handle representing a `TpuDriver` system info structure. */
typedef struct TpuSystemInfo TpuSystemInfo;

/** @brief Opaque handle representing a compiled program in `TpuDriver`. */
typedef struct TpuCompiledProgramHandle TpuCompiledProgramHandle;

/** @brief Opaque handle representing a loaded program in `TpuDriver`. */
typedef struct TpuLoadedProgramHandle TpuLoadedProgramHandle;

/** @brief Opaque handle representing a device buffer in `TpuDriver`. */
typedef struct TpuBufferHandle TpuBufferHandle;

/**
 * @brief Serialized HLO program descriptor passed to `TpuDriver_Compile` (`16`
 * bytes).
 */
typedef struct HloProto {
  /** @brief Pointer to serialized `HloModuleProto` or HLO text bytes. */
  const void *buffer;
  /** @brief Byte length of `buffer`. */
  int64_t size;
} HloProto;

/**
 * @brief Logical device assignment passed to `TpuDriver_ExecuteProgram` (`8`
 * bytes).
 */
typedef struct DeviceAssignment {
  /** @brief Number of replicas. */
  int32_t replica_count;
  /** @brief Number of computations (partitions). */
  int32_t computation_count;
} DeviceAssignment;

/**
 * @brief Populates a `TpuDriverFn` function table (`TpuDriver_Initialize`).
 * @param driver_fn Output pointer to `TpuDriverFn` to populate.
 * @param initialize Whether to initialize the global TPU runtime.
 */
void TpuDriver_Initialize(TpuDriverFn *driver_fn, bool initialize);

/**
 * @brief Opens a `TpuDriver` instance (`TpuDriver_Open`).
 * @param worker Null-terminated worker address (e.g., `"local://"`).
 * @return Heap-allocated `TpuDriver*` handle.
 */
TpuDriver *TpuDriver_Open(const char *worker);

/**
 * @brief Closes and frees a `TpuDriver` instance (`TpuDriver_Close`).
 * @param driver Pointer returned by `TpuDriver_Open`.
 */
void TpuDriver_Close(TpuDriver *driver);

/**
 * @brief Allocates a device buffer on a `TpuDriver` (`TpuDriver_Allocate`).
 * @param driver Pointer to `TpuDriver`.
 * @param core_id Target core ordinal.
 * @param memory_region Memory region (`0` = HBM, `1` = VMEM).
 * @param num_bytes Allocation size in bytes.
 * @param num_ains Number of prerequisite events.
 * @param ains Array of prerequisite `TpuEvent*` pointers.
 * @return Allocated `TpuBufferHandle*`.
 */
TpuBufferHandle *TpuDriver_Allocate(TpuDriver *driver, int32_t core_id,
                                    int32_t memory_region, int64_t num_bytes,
                                    int32_t num_ains, TpuEvent *const *ains);

/**
 * @brief Deallocates a device buffer on a `TpuDriver` (`TpuDriver_Deallocate`).
 * @param driver Pointer to `TpuDriver`.
 * @param buf Buffer handle returned by `TpuDriver_Allocate`.
 * @param num_ains Number of prerequisite events.
 * @param ains Array of prerequisite `TpuEvent*` pointers.
 * @return Completion `TpuEvent*`.
 */
TpuEvent *TpuDriver_Deallocate(TpuDriver *driver, TpuBufferHandle *buf,
                               int32_t num_ains, TpuEvent *const *ains);

/**
 * @brief Frees a `TpuEvent` (`TpuDriver_FreeEvent`).
 * @param event Event handle to free.
 */
void TpuDriver_FreeEvent(TpuEvent *event);

/**
 * @brief Allocates a new OK `TpuStatus`.
 * @return Heap-allocated `TpuStatus*`.
 */
TpuStatus *TpuStatus_New(void);

/**
 * @brief Frees a `TpuStatus` allocated by `TpuStatus_New`.
 * @param status Pointer to the `TpuStatus` to free.
 */
void TpuStatus_Free(TpuStatus *status);

/**
 * @brief Checks whether `status` represents success (`code == 0`).
 * @param status Pointer to `TpuStatus`.
 * @return `true` if `status` is OK, `false` otherwise.
 */
bool TpuStatus_Ok(const TpuStatus *status);

/**
 * @brief Returns the numeric status code of `status`.
 * @param status Pointer to `TpuStatus`.
 * @return Integer status code (`0` = OK).
 */
int32_t TpuStatus_Code(const TpuStatus *status);

/**
 * @brief Returns the null-terminated error message of `status`.
 * @param status Pointer to `TpuStatus`.
 * @return Pointer to null-terminated C string.
 */
const char *TpuStatus_Message(const TpuStatus *status);

/**
 * @brief Creates a `TpuProfiler` session.
 * @param profiler_out Output pointer receiving the created `TpuProfiler*`.
 * @param status Optional `TpuStatus*` to record any error.
 */
void TpuProfiler_Create(TpuProfiler **profiler_out, TpuStatus *status);

/**
 * @brief Destroys a `TpuProfiler` session.
 * @param profiler Pointer to `TpuProfiler` to destroy.
 */
void TpuProfiler_Destroy(TpuProfiler *profiler);

/**
 * @brief Starts trace collection on `profiler`.
 * @param profiler Pointer to `TpuProfiler`.
 * @param status Optional `TpuStatus*` to record any error.
 */
void TpuProfiler_Start(TpuProfiler *profiler, TpuStatus *status);

/**
 * @brief Stops trace collection on `profiler`.
 * @param profiler Pointer to `TpuProfiler`.
 * @param status Optional `TpuStatus*` to record any error.
 */
void TpuProfiler_Stop(TpuProfiler *profiler, TpuStatus *status);

/**
 * @brief Collects serialized `XSpace` trace bytes from `profiler`.
 * @param profiler Pointer to `TpuProfiler`.
 * @param status Optional `TpuStatus*` to record any error.
 * @param buffer Destination byte buffer (or `NULL` to query byte size).
 * @param size_in_bytes In/out pointer receiving the serialized byte length.
 */
void TpuProfiler_CollectData(TpuProfiler *profiler, TpuStatus *status,
                             uint8_t *buffer, size_t *size_in_bytes);

/**
 * @brief Frees the error message inside a `LibTpuCStatus`.
 * @param status Status structure whose `message` buffer should be freed.
 */
void libtpu_rs_status_free(LibTpuCStatus status);

/**
 * @brief Creates a new `PJRT_Client` instance.
 * @param client_out Output pointer receiving the created `PJRT_Client*`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus libtpu_rs_client_create(PJRT_Client **client_out);

/**
 * @brief Destroys a `PJRT_Client` instance.
 * @param client Pointer to `PJRT_Client` to destroy.
 */
void libtpu_rs_client_destroy(PJRT_Client *client);

/**
 * @brief Queries the number of addressable TPU devices on `client`.
 * @param client Pointer to `PJRT_Client`.
 * @return Number of addressable devices, or `-1` if `client` is null.
 */
int32_t libtpu_rs_client_device_count(const PJRT_Client *client);

/**
 * @brief Allocates a `PJRT_Buffer` on `device_id` and transfers `host_data` to
 * HBM.
 * @param client Pointer to `PJRT_Client`.
 * @param device_id Target logical TPU device ID.
 * @param element_type Raw `PJRT_Buffer_Type` integer (`11` = `F32`, `4` =
 * `S32`).
 * @param dims Pointer to dimension extents array.
 * @param num_dims Number of dimensions.
 * @param host_data Pointer to source host bytes.
 * @param host_data_len Number of bytes in `host_data`.
 * @param buffer_out Output pointer receiving the allocated `PJRT_Buffer*`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus
libtpu_rs_buffer_from_host(PJRT_Client *client, uint32_t device_id,
                           int32_t element_type, const int64_t *dims,
                           size_t num_dims, const uint8_t *host_data,
                           size_t host_data_len, PJRT_Buffer **buffer_out);

/**
 * @brief Reads a `PJRT_Buffer` from HBM into `host_out`.
 * @param buffer Pointer to source `PJRT_Buffer`.
 * @param host_out Destination host byte buffer.
 * @param host_out_len Number of bytes to read into `host_out`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus libtpu_rs_buffer_to_host(const PJRT_Buffer *buffer,
                                       uint8_t *host_out, size_t host_out_len);

/**
 * @brief Deletes and frees a `PJRT_Buffer`.
 * @param buffer Pointer to `PJRT_Buffer` to destroy.
 */
void libtpu_rs_buffer_destroy(PJRT_Buffer *buffer);

/**
 * @brief Compiles and loads an HLO or `StableHLO` program.
 * @param client Pointer to `PJRT_Client`.
 * @param format Null-terminated program format string (`"mlir"` or `"hlo"`).
 * @param code Pointer to program source bytes.
 * @param code_len Length of `code` in bytes.
 * @param exec_out Output pointer receiving the `PJRT_LoadedExecutable*`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus libtpu_rs_compile(PJRT_Client *client, const char *format,
                                const uint8_t *code, size_t code_len,
                                PJRT_LoadedExecutable **exec_out);

/**
 * @brief Executes a `PJRT_LoadedExecutable` on `device_id`.
 * @param executable Pointer to `PJRT_LoadedExecutable`.
 * @param device_id Target logical TPU device ID.
 * @param inputs Array of input `const PJRT_Buffer*` pointers.
 * @param num_inputs Number of input buffers.
 * @param output_out Output pointer receiving the primary output `PJRT_Buffer*`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus libtpu_rs_execute(const PJRT_LoadedExecutable *executable,
                                uint32_t device_id,
                                const PJRT_Buffer *const *inputs,
                                size_t num_inputs, PJRT_Buffer **output_out);

/**
 * @brief Destroys a `PJRT_LoadedExecutable`.
 * @param executable Pointer to `PJRT_LoadedExecutable` to destroy.
 */
void libtpu_rs_executable_destroy(PJRT_LoadedExecutable *executable);

/**
 * @brief Queries the rank and dimension extents of a `PJRT_Buffer`.
 * @param buffer Pointer to `PJRT_Buffer`.
 * @param out_dims Optional output buffer for `int64_t` dimensions (may be
 * `NULL`).
 * @param max_dims Capacity of `out_dims` in elements.
 * @param out_num_dims Output pointer receiving the rank of `buffer`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus libtpu_rs_buffer_dimensions(const PJRT_Buffer *buffer,
                                          int64_t *out_dims, size_t max_dims,
                                          size_t *out_num_dims);

/**
 * @brief Queries the raw `PJRT_Buffer_Type` integer of a `PJRT_Buffer`.
 * @param buffer Pointer to `PJRT_Buffer`.
 * @return Element type integer (`11` = `F32`, `4` = `S32`, `0` if `buffer` is
 * `NULL`).
 */
int32_t libtpu_rs_buffer_element_type(const PJRT_Buffer *buffer);

/**
 * @brief Copies a `PJRT_Buffer` to `dst_device_id` on `client`.
 * @param client Pointer to `PJRT_Client`.
 * @param buffer Source `PJRT_Buffer*`.
 * @param dst_device_id Destination logical TPU device ID.
 * @param out_buffer Output pointer receiving the copied `PJRT_Buffer*`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus libtpu_rs_buffer_copy_to_device(PJRT_Client *client,
                                              const PJRT_Buffer *buffer,
                                              uint32_t dst_device_id,
                                              PJRT_Buffer **out_buffer);

/**
 * @brief Executes a `PJRT_LoadedExecutable` with arbitrary `num_inputs` and
 * `num_outputs`.
 * @param executable Pointer to `PJRT_LoadedExecutable`.
 * @param device_id Target logical TPU device ID.
 * @param inputs Array of input `const PJRT_Buffer*` pointers.
 * @param num_inputs Number of input buffers.
 * @param outputs_out Caller-allocated array of `PJRT_Buffer*` of length
 * `num_outputs`.
 * @param num_outputs Number of output buffer slots in `outputs_out`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus libtpu_rs_execute_multi(const PJRT_LoadedExecutable *executable,
                                      uint32_t device_id,
                                      const PJRT_Buffer *const *inputs,
                                      size_t num_inputs,
                                      PJRT_Buffer **outputs_out,
                                      size_t num_outputs);

/**
 * @brief Serializes a `PJRT_LoadedExecutable` into `out_bytes`.
 * @param executable Pointer to `PJRT_LoadedExecutable`.
 * @param out_bytes Optional output buffer (pass `NULL` to query size).
 * @param max_len Capacity of `out_bytes` in bytes.
 * @param out_len Output pointer receiving the serialized byte length.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus
libtpu_rs_executable_serialize(const PJRT_LoadedExecutable *executable,
                               uint8_t *out_bytes, size_t max_len,
                               size_t *out_len);

/**
 * @brief Deserializes and loads a `PJRT_LoadedExecutable` on `client`.
 * @param client Pointer to `PJRT_Client`.
 * @param bytes Pointer to serialized executable bytes.
 * @param bytes_len Length of `bytes` in bytes.
 * @param exec_out Output pointer receiving the loaded `PJRT_LoadedExecutable*`.
 * @return `LibTpuCStatus` indicating success or failure.
 */
LibTpuCStatus
libtpu_rs_executable_deserialize(PJRT_Client *client, const uint8_t *bytes,
                                 size_t bytes_len,
                                 PJRT_LoadedExecutable **exec_out);

/**
 * @brief Copies the 32-character hex fingerprint of `executable` into
 * `out_buf`.
 * @param executable Pointer to `PJRT_LoadedExecutable`.
 * @param out_buf Optional destination character buffer.
 * @param buf_len Capacity of `out_buf` in bytes.
 * @return Length of the fingerprint in bytes (`32`, or `0` if `executable` is
 * `NULL`).
 */
size_t libtpu_rs_executable_fingerprint(const PJRT_LoadedExecutable *executable,
                                        char *out_buf, size_t buf_len);

/**
 * @brief Returns the estimated FLOPs of `executable`.
 * @param executable Pointer to `PJRT_LoadedExecutable`.
 * @return Estimated FLOPs (`0.0f` if `executable` is `NULL`).
 */
float libtpu_rs_executable_flops(const PJRT_LoadedExecutable *executable);

#ifdef __cplusplus
}
#endif

#endif /* LIBTPU_H_ */
