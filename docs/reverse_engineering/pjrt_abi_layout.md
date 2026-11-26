# `PJRT_Api` & `LibtpuSdk_Api` Binary Layout Specification (`libtpu-0.0.53.1`)

## 1. `PJRT_Api` Header & Version Layout
Disassembly of `pjrt::CreatePjrtApi` (`0x06f72ee0`) in `libtpu.so` reveals the exact binary header of the static `PJRT_Api` structure (`_ZZN4pjrt10tpu_plugin13GetTpuPjrtApiEvE8pjrt_api` at `0x14294e00`):

- **Total `PJRT_Api` Size (`struct_size`)**: `0x478` = **1,144 bytes** (143 64-bit words)
- **`PJRT_Api_Version`**:
  - `struct_size` (offset `0x10`): `0x18` = **24 bytes**
  - `extension_start` (offset `0x18`): `NULL` (`0x0`)
  - `major_version` (offset `0x20`, `int32_t`): **`0`**
  - `minor_version` (offset `0x24`, `int32_t`): **`117`** (`0x75`)

## 2. Complete `PJRT_Api` Function Pointer Vtable (Offsets `0x000` .. `0x470`) & `Args` Struct Layouts
| Offset | Slot # | Field Name | Internal Implementation Symbol in `libtpu.so` | `Args` Struct Size (`bytes`) |
| :--- | ---: | :--- | :--- | ---: |
| `0x000` | 0 | `struct_size` | `1144` (`0x478`) | — |
| `0x008` | 1 | `extension_start` | Pointer to `ffi_extension` (head of 18-node linked list) | — |
| `0x010` | 2..4 | `pjrt_api_version` | `{ struct_size: 24, extension_start: NULL, major: 0, minor: 117 }` | 24 |
| `0x028` | 5 | `PJRT_Error_Destroy` | `pjrt::PJRT_Error_Destroy` (`0x06f5e950`) | 24 |
| `0x030` | 6 | `PJRT_Error_Message` | `pjrt::PJRT_Error_Message` (`0x06f5ea30`) | 40 |
| `0x038` | 7 | `PJRT_Error_GetCode` | `pjrt::PJRT_Error_GetCode` (`0x06f5eb20`) | 32 |
| `0x040` | 8 | `PJRT_Plugin_Initialize` | `pjrt::tpu_plugin::PJRT_Plugin_Initialize` (`0x06078ff0`) | 16 |
| `0x048` | 9 | `PJRT_Plugin_Attributes` | `pjrt::PJRT_Plugin_Attributes_Xla` (`0x06f5ebf0`) | 32 |
| `0x050` | 10 | `PJRT_Event_Destroy` | `pjrt::PJRT_Event_Destroy` (`0x06f6e880`) | 24 |
| `0x058` | 11 | `PJRT_Event_IsReady` | `pjrt::PJRT_Event_IsReady` (`0x06f6e930`) | 32 |
| `0x060` | 12 | `PJRT_Event_Error` | `pjrt::PJRT_Event_Error` (`0x06f6eaf0`) | 24 |
| `0x068` | 13 | `PJRT_Event_Await` | `pjrt::PJRT_Event_Await` (`0x06f6e9c0`) | 24 |
| `0x070` | 14 | `PJRT_Event_OnReady` | `pjrt::PJRT_Event_OnReady` (`0x06f6ebb0`) | 40 |
| `0x078` | 15 | `PJRT_Client_Create` | `pjrt::tpu_plugin::PJRT_Client_Create` (`0x060779c0`) | 88 |
| `0x080` | 16 | `PJRT_Client_Destroy` | `pjrt::PJRT_Client_Destroy` (`0x06f5ec50`) | 24 |
| `0x088` | 17 | `PJRT_Client_PlatformName` | `pjrt::PJRT_Client_PlatformName` (`0x06f5eea0`) | 40 |
| `0x090` | 18 | `PJRT_Client_ProcessIndex` | `pjrt::PJRT_Client_ProcessIndex` (`0x06f5ee50`) | 32 |
| `0x098` | 19 | `PJRT_Client_PlatformVersion` | `pjrt::PJRT_Client_PlatformVersion` (`0x06f5ef00`) | 40 |
| `0x0a0` | 20 | `PJRT_Client_Devices` | `pjrt::PJRT_Client_Devices` (`0x06f5efe0`) | 40 |
| `0x0a8` | 21 | `PJRT_Client_AddressableDevices` | `pjrt::PJRT_Client_AddressableDevices` (`0x06f5f040`) | 40 |
| `0x0b0` | 22 | `PJRT_Client_LookupDevice` | `pjrt::PJRT_Client_LookupDevice` (`0x06f5f0a0`) | 40 |
| `0x0b8` | 23 | `PJRT_Client_LookupAddressableDevice` | `pjrt::PJRT_Client_LookupAddressableDevice` (`0x06f5f2b0`) | 40 |
| `0x0c0` | 24 | `PJRT_Client_AddressableMemories` | `pjrt::PJRT_Client_AddressableMemories` (`0x06f5f660`) | 40 |
| `0x0c8` | 25 | `PJRT_Client_Compile` | `pjrt::PJRT_Client_Compile` (`0x06f60fc0`) | 56 |
| `0x0d0` | 26 | `PJRT_Client_DefaultDeviceAssignment` | `pjrt::PJRT_Client_DefaultDeviceAssignment` (`0x06f62790`) | 56 |
| `0x0d8` | 27 | `PJRT_Client_BufferFromHostBuffer` | `pjrt::PJRT_Client_BufferFromHostBuffer` (`0x06f63db0`) | 112 |
| `0x0e0` | 28 | `PJRT_DeviceDescription_Id` | `pjrt::PJRT_DeviceDescription_Id` (`0x06f64b40`) | 32 |
| `0x0e8` | 29 | `PJRT_DeviceDescription_ProcessIndex` | `pjrt::PJRT_DeviceDescription_ProcessIndex` (`0x06f64b90`) | 32 |
| `0x0f0` | 30 | `PJRT_DeviceDescription_Attributes` | `pjrt::PJRT_DeviceDescription_Attributes` (`0x06f64be0`) | 40 |
| `0x0f8` | 31 | `PJRT_DeviceDescription_Kind` | `pjrt::PJRT_DeviceDescription_Kind` (`0x06f64c40`) | 40 |
| `0x100` | 32 | `PJRT_DeviceDescription_DebugString` | `pjrt::PJRT_DeviceDescription_DebugString` (`0x06f64cb0`) | 40 |
| `0x108` | 33 | `PJRT_DeviceDescription_ToString` | `pjrt::PJRT_DeviceDescription_ToString` (`0x06f65040`) | 40 |
| `0x110` | 34 | `PJRT_Device_GetDescription` | `pjrt::PJRT_Device_GetDescription` (`0x06f65120`) | 32 |
| `0x118` | 35 | `PJRT_Device_IsAddressable` | `pjrt::PJRT_Device_IsAddressable` (`0x06f65170`) | 32 |
| `0x120` | 36 | `PJRT_Device_LocalHardwareId` | `pjrt::PJRT_Device_LocalHardwareId` (`0x06f651c0`) | 32 |
| `0x128` | 37 | `PJRT_Device_AddressableMemories` | `pjrt::PJRT_Device_AddressableMemories` (`0x06f65210`) | 40 |
| `0x130` | 38 | `PJRT_Device_DefaultMemory` | `pjrt::PJRT_Device_DefaultMemory` (`0x06f65270`) | 32 |
| `0x138` | 39 | `PJRT_Device_MemoryStats` | `pjrt::PJRT_Device_MemoryStats` (`0x06f65310`) | 192 |
| `0x140` | 40 | `PJRT_Memory_Id` | `pjrt::PJRT_Memory_Id` (`0x06f654e0`) | 32 |
| `0x148` | 41 | `PJRT_Memory_Kind` | `pjrt::PJRT_Memory_Kind` (`0x06f65540`) | 40 |
| `0x150` | 42 | `PJRT_Memory_DebugString` | `pjrt::PJRT_Memory_DebugString` (`0x06f65610`) | 40 |
| `0x158` | 43 | `PJRT_Memory_ToString` | `pjrt::PJRT_Memory_ToString` (`0x06f65680`) | 40 |
| `0x160` | 44 | `PJRT_Memory_AddressableByDevices` | `pjrt::PJRT_Memory_AddressableByDevices` (`0x06f656f0`) | 40 |
| `0x168` | 45 | `PJRT_Executable_Destroy` | `pjrt::PJRT_Executable_Destroy` (`0x06f657e0`) | 24 |
| `0x170` | 46 | `PJRT_Executable_Name` | `pjrt::PJRT_Executable_Name` (`0x06f65e50`) | 40 |
| `0x178` | 47 | `PJRT_Executable_NumReplicas` | `pjrt::PJRT_Executable_NumReplicas` (`0x06f65eb0`) | 32 |
| `0x180` | 48 | `PJRT_Executable_NumPartitions` | `pjrt::PJRT_Executable_NumPartitions` (`0x06f65f10`) | 32 |
| `0x188` | 49 | `PJRT_Executable_NumOutputs` | `pjrt::PJRT_Executable_NumOutputs` (`0x06f66060`) | 32 |
| `0x190` | 50 | `PJRT_Executable_SizeOfGeneratedCodeInBytes` | `pjrt::PJRT_Executable_SizeOfGeneratedCodeInBytes` (`0x06f66800`) | 32 |
| `0x198` | 51 | `PJRT_Executable_GetCostAnalysis` | `pjrt::PJRT_Executable_GetCostAnalysis` (`0x06f67040`) | 40 |
| `0x1a0` | 52 | `PJRT_Executable_OutputMemoryKinds` | `pjrt::PJRT_Executable_OutputMemoryKinds` (`0x06f68950`) | 48 |
| `0x1a8` | 53 | `PJRT_Executable_OptimizedProgram` | `pjrt::PJRT_Executable_OptimizedProgram` (`0x06f66860`) | 32 |
| `0x1b0` | 54 | `PJRT_Executable_Serialize` | `pjrt::PJRT_Executable_Serialize` (`0x06f6b670`) | 48 |
| `0x1b8` | 55 | `PJRT_LoadedExecutable_Destroy` | `pjrt::PJRT_LoadedExecutable_Destroy` (`0x06f65d70`) | 24 |
| `0x1c0` | 56 | `PJRT_LoadedExecutable_GetExecutable` | `pjrt::PJRT_LoadedExecutable_GetExecutable` (`0x06f6c2f0`) | 32 |
| `0x1c8` | 57 | `PJRT_LoadedExecutable_AddressableDevices` | `pjrt::PJRT_LoadedExecutable_AddressableDevices` (`0x06f65f70`) | 40 |
| `0x1d0` | 58 | `PJRT_LoadedExecutable_Delete` | `pjrt::PJRT_LoadedExecutable_Delete` (`0x06f68e70`) | 24 |
| `0x1d8` | 59 | `PJRT_LoadedExecutable_IsDeleted` | `pjrt::PJRT_LoadedExecutable_IsDeleted` (`0x06f68ec0`) | 32 |
| `0x1e0` | 60 | `PJRT_LoadedExecutable_Execute` | `pjrt::PJRT_LoadedExecutable_Execute` (`0x06f68f10`) | 56 |
| `0x1e8` | 61 | `PJRT_Executable_DeserializeAndLoad` | `pjrt::PJRT_Executable_DeserializeAndLoad` (`0x06f6bf30`) | 56 |
| `0x1f0` | 62 | `PJRT_LoadedExecutable_Fingerprint` | `pjrt::PJRT_LoadedExecutable_Fingerprint` (`0x06f5f5e0`) | 40 |
| `0x1f8` | 63 | `PJRT_Buffer_Destroy` | `pjrt::PJRT_Buffer_Destroy` (`0x06f6c360`) | 24 |
| `0x200` | 64 | `PJRT_Buffer_ElementType` | `pjrt::PJRT_Buffer_ElementType` (`0x06f6c540`) | 32 |
| `0x208` | 65 | `PJRT_Buffer_Dimensions` | `pjrt::PJRT_Buffer_Dimensions` (`0x06f6c5a0`) | 40 |
| `0x210` | 66 | `PJRT_Buffer_UnpaddedDimensions` | `pjrt::PJRT_Buffer_UnpaddedDimensions` (`0x06f6c610`) | 40 |
| `0x218` | 67 | `PJRT_Buffer_DynamicDimensionIndices` | `pjrt::PJRT_Buffer_DynamicDimensionIndices` (`0x06f6c7c0`) | 40 |
| `0x220` | 68 | `PJRT_Buffer_GetMemoryLayout` | `pjrt::PJRT_Buffer_GetMemoryLayout` (`0x06f6c8e0`) | 64 |
| `0x228` | 69 | `PJRT_Buffer_OnDeviceSizeInBytes` | `pjrt::PJRT_Buffer_OnDeviceSizeInBytes` (`0x06f6cd50`) | 32 |
| `0x230` | 70 | `PJRT_Buffer_Device` | `pjrt::PJRT_Buffer_Device` (`0x06f6ce00`) | 32 |
| `0x238` | 71 | `PJRT_Buffer_Memory` | `pjrt::PJRT_Buffer_Memory` (`0x06f6cf10`) | 32 |
| `0x240` | 72 | `PJRT_Buffer_Delete` | `pjrt::PJRT_Buffer_Delete` (`0x06f6d120`) | 24 |
| `0x248` | 73 | `PJRT_Buffer_IsDeleted` | `pjrt::PJRT_Buffer_IsDeleted` (`0x06f6d170`) | 32 |
| `0x250` | 74 | `PJRT_Buffer_CopyToDevice` | `pjrt::PJRT_Buffer_CopyToDevice` (`0x06f6d570`) | 40 |
| `0x258` | 75 | `PJRT_Buffer_ToHostBuffer` | `pjrt::PJRT_Buffer_ToHostBuffer` (`0x06f6d820`) | 56 |
| `0x260` | 76 | `PJRT_Buffer_IsOnCpu` | `pjrt::PJRT_Buffer_IsOnCpu` (`0x06f6de10`) | 32 |
| `0x268` | 77 | `PJRT_Buffer_ReadyEvent` | `pjrt::PJRT_Buffer_ReadyEvent` (`0x06f6de70`) | 32 |
| `0x270` | 78 | `PJRT_Buffer_UnsafePointer` | `pjrt::PJRT_Buffer_UnsafePointer` (`0x06f6def0`) | 32 |
| `0x278` | 79 | `PJRT_Buffer_IncreaseExternalReferenceCount` | `pjrt::PJRT_Buffer_IncreaseExternalReferenceCount` (`0x06f6dfa0`) | 24 |
| `0x280` | 80 | `PJRT_Buffer_DecreaseExternalReferenceCount` | `pjrt::PJRT_Buffer_DecreaseExternalReferenceCount` (`0x06f6e1a0`) | 24 |
| `0x288` | 81 | `PJRT_Buffer_OpaqueDeviceMemoryDataPointer` | `pjrt::PJRT_Buffer_OpaqueDeviceMemoryDataPointer` (`0x06f6e300`) | 32 |
| `0x290` | 82 | `PJRT_CopyToDeviceStream_Destroy` | `pjrt::PJRT_CopyToDeviceStream_Destroy` (`0x06f6e640`) | 24 |
| `0x298` | 83 | `PJRT_CopyToDeviceStream_AddChunk` | `pjrt::PJRT_CopyToDeviceStream_AddChunk` (`0x06f6e6b0`) | 40 |
| `0x2a0` | 84 | `PJRT_CopyToDeviceStream_TotalBytes` | `pjrt::PJRT_CopyToDeviceStream_TotalBytes` (`0x06f6e770`) | 32 |
| `0x2a8` | 85 | `PJRT_CopyToDeviceStream_GranuleSize` | `pjrt::PJRT_CopyToDeviceStream_GranuleSize` (`0x06f6e7c0`) | 32 |
| `0x2b0` | 86 | `PJRT_CopyToDeviceStream_CurrentBytes` | `pjrt::PJRT_CopyToDeviceStream_CurrentBytes` (`0x06f6e810`) | 32 |
| `0x2b8` | 87 | `PJRT_TopologyDescription_Create` | `pjrt::tpu_plugin::PJRT_TopologyDescription_Create` (`0x06078e40`) | 48 |
| `0x2c0` | 88 | `PJRT_TopologyDescription_Destroy` | `pjrt::PJRT_TopologyDescription_Destroy` (`0x06f6eee0`) | 24 |
| `0x2c8` | 89 | `PJRT_TopologyDescription_PlatformName` | `pjrt::PJRT_TopologyDescription_PlatformName` (`0x06f6f080`) | 40 |
| `0x2d0` | 90 | `PJRT_TopologyDescription_PlatformVersion` | `pjrt::PJRT_TopologyDescription_PlatformVersion` (`0x06f6f0e0`) | 40 |
| `0x2d8` | 91 | `PJRT_TopologyDescription_GetDeviceDescriptions` | `pjrt::PJRT_TopologyDescription_GetDeviceDescriptions` (`0x06f6f140`) | 40 |
| `0x2e0` | 92 | `PJRT_TopologyDescription_Serialize` | `pjrt::PJRT_TopologyDescription_Serialize` (`0x06f6f1a0`) | 48 |
| `0x2e8` | 93 | `PJRT_TopologyDescription_Attributes` | `pjrt::PJRT_TopologyDescription_Attributes` (`0x06f6f430`) | 40 |
| `0x2f0` | 94 | `PJRT_Compile` | `pjrt::PJRT_Compile` (`0x06f6f8a0`) | 56 |
| `0x2f8` | 95 | `PJRT_Executable_OutputElementTypes` | `pjrt::PJRT_Executable_OutputElementTypes` (`0x06f67a00`) | 40 |
| `0x300` | 96 | `PJRT_Executable_OutputDimensions` | `pjrt::PJRT_Executable_OutputDimensions` (`0x06f67e40`) | 48 |
| `0x308` | 97 | `PJRT_Buffer_CopyToMemory` | `pjrt::PJRT_Buffer_CopyToMemory` (`0x06f6d6d0`) | 40 |
| `0x310` | 98 | `PJRT_Client_CreateViewOfDeviceBuffer` | `pjrt::PJRT_Client_CreateViewOfDeviceBuffer` (`0x06f64850`) | 88 |
| `0x318` | 99 | `PJRT_Executable_Fingerprint` | `pjrt::PJRT_Executable_Fingerprint` (`0x06f66f90`) | 40 |
| `0x320` | 100 | `PJRT_Client_TopologyDescription` | `pjrt::PJRT_Client_TopologyDescription` (`0x06f5ef60`) | 32 |
| `0x328` | 101 | `PJRT_Executable_GetCompiledMemoryStats` | `pjrt::PJRT_Executable_GetCompiledMemoryStats` (`0x06f6bb20`) | 64 |
| `0x330` | 102 | `PJRT_Memory_Kind_Id` | `pjrt::PJRT_Memory_Kind_Id` (`0x06f655b0`) | 32 |
| `0x338` | 103 | `PJRT_ExecuteContext_Create` | `pjrt::tpu_plugin::PJRT_ExecuteContext_Create` (`0x06078e10`) | 24 |
| `0x340` | 104 | `PJRT_ExecuteContext_Destroy` | `pjrt::PJRT_ExecuteContext_Destroy` (`0x06f65750`) | 24 |
| `0x348` | 105 | `PJRT_Buffer_CopyRawToHost` | `pjrt::PJRT_Buffer_CopyRawToHost` (`0x06f6d1d0`) | 56 |
| `0x350` | 106 | `PJRT_AsyncHostToDeviceTransferManager_Destroy` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_Destroy` (`0x06f5fff0`) | 24 |
| `0x358` | 107 | `PJRT_AsyncHostToDeviceTransferManager_TransferData` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_TransferData` (`0x06f60060`) | 64 |
| `0x360` | 108 | `PJRT_Client_CreateBuffersForAsyncHostToDevice` | `pjrt::PJRT_Client_CreateBuffersForAsyncHostToDevice` (`0x06f5f6c0`) | 48 |
| `0x368` | 109 | `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer` (`0x06f60a90`) | 40 |
| `0x370` | 110 | `PJRT_AsyncHostToDeviceTransferManager_Device` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_Device` (`0x06f60b40`) | 32 |
| `0x378` | 111 | `PJRT_AsyncHostToDeviceTransferManager_BufferCount` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferCount` (`0x06f60c50`) | 32 |
| `0x380` | 112 | `PJRT_AsyncHostToDeviceTransferManager_BufferSize` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferSize` (`0x06f60ca0`) | 40 |
| `0x388` | 113 | `PJRT_AsyncHostToDeviceTransferManager_SetBufferError` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_SetBufferError` (`0x06f60d00`) | 48 |
| `0x390` | 114 | `PJRT_AsyncHostToDeviceTransferManager_AddMetadata` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_AddMetadata` (`0x06f60dc0`) | 40 |
| `0x398` | 115 | `PJRT_Client_DmaMap` | `pjrt::PJRT_Client_DmaMap` (`0x06f5ff30`) | 40 |
| `0x3a0` | 116 | `PJRT_Client_DmaUnmap` | `pjrt::PJRT_Client_DmaUnmap` (`0x06f5ff90`) | 32 |
| `0x3a8` | 117 | `PJRT_Client_CreateUninitializedBuffer` | `pjrt::PJRT_Client_CreateUninitializedBuffer` (`0x06f62e70`) | 72 |
| `0x3b0` | 118 | `PJRT_Client_UpdateGlobalProcessInfo` | `pjrt::PJRT_Client_UpdateGlobalProcessInfo` (`0x06f5f360`) | 32 |
| `0x3b8` | 119 | `PJRT_TopologyDescription_Deserialize` | `pjrt::PJRT_TopologyDescription_Deserialize` (`0x06f6ff80`) | 48 |
| `0x3c0` | 120 | `PJRT_Client_CreateAliasBuffer` | `pjrt::PJRT_Client_CreateAliasBuffer` (`0x06f636b0`) | 72 |
| `0x3c8` | 121 | `PJRT_Client_FulfillAliasBuffer` | `pjrt::PJRT_Client_FulfillAliasBuffer` (`0x06f63ae0`) | 40 |
| `0x3d0` | 122 | `PJRT_LoadedExecutable_GetDeviceAssignment` | `pjrt::PJRT_LoadedExecutable_GetDeviceAssignment` (`0x06f70250`) | 48 |
| `0x3d8` | 123 | `PJRT_Client_CreateErrorBuffer` | `pjrt::PJRT_Client_CreateErrorBuffer` (`0x06f63080`) | 64 |
| `0x3e0` | 124 | `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral` (`0x06f602b0`) | 64 |
| `0x3e8` | 125 | `PJRT_Buffer_CopyRawToHostFuture` | `pjrt::PJRT_Buffer_CopyRawToHostFuture` (`0x06f6d2d0`) | 64 |
| `0x3f0` | 126 | `PJRT_Device_PoisonExecution` | `pjrt::PJRT_Device_PoisonExecution` (`0x06f60650`) | 48 |
| `0x3f8` | 127 | `PJRT_Device_CreateAsyncTrackingEvent` | `pjrt::PJRT_Device_CreateAsyncTrackingEvent` (`0x06f609a0`) | 48 |
| `0x400` | 128 | `PJRT_AsyncTrackingEvent_Destroy` | `pjrt::PJRT_AsyncTrackingEvent_Destroy` (`0x06f60a20`) | 24 |
| `0x408` | 129 | `PJRT_Executable_GetCompileOptions` | `pjrt::PJRT_Executable_GetCompileOptions` (`0x06f6b790`) | 48 |
| `0x410` | 130 | `PJRT_Buffer_DonateWithControlDependency` | `pjrt::PJRT_Buffer_DonateWithControlDependency` (`0x06f6e3d0`) | 48 |
| `0x418` | 131 | `PJRT_Event_Create` | `pjrt::PJRT_Event_Create` (`0x06f6ed50`) | 24 |
| `0x420` | 132 | `PJRT_Event_Set` | `pjrt::PJRT_Event_Set` (`0x06f6ee40`) | 40 |
| `0x428` | 133 | `PJRT_Device_GetAttributes` | `pjrt::PJRT_Device_GetAttributes` (`0x06f726b0`) | 40 |
| `0x430` | 134 | `PJRT_Client_Load` | `pjrt::PJRT_Client_Load` (`0x06f620d0`) | 48 |
| `0x438` | 135 | `PJRT_LoadedExecutable_AddressableDeviceLogicalIds` | `pjrt::PJRT_LoadedExecutable_AddressableDeviceLogicalIds` (`0x06f65fd0`) | 40 |
| `0x440` | 136 | `PJRT_Buffer_Bitcast` | `pjrt::PJRT_Buffer_Bitcast` (`0x06f62400`) | 64 |
| `0x448` | 137 | `PJRT_Error_ForEachPayload` | `pjrt::PJRT_Error_ForEachPayload` (`0x06f5eb70`) | 40 |
| `0x450` | 138 | `PJRT_TopologyDescription_Fingerprint` | `pjrt::PJRT_TopologyDescription_Fingerprint` (`0x06f6f380`) | 32 |
| `0x458` | 139 | `PJRT_Executable_ParameterMemoryKinds` | `pjrt::PJRT_Executable_ParameterMemoryKinds` (`0x06f68430`) | 48 |
| `0x460` | 140 | `PJRT_Device_ClearMemoryStats` | `pjrt::PJRT_Device_ClearMemoryStats` (`0x06f605f0`) | 24 |
| `0x468` | 141 | `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` | `pjrt::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` (`0x06f6f490`) | 64 |
| `0x470` | 142 | `PJRT_TopologyDescription_GetMemorySpaceKindIds` | `pjrt::PJRT_TopologyDescription_GetMemorySpaceKindIds` (`0x06f6f840`) | 40 |

## 3. PJRT Extension Chain (`PJRT_Extension_Base` Linked List)
`GetTpuPjrtApi()` (`0x060788b0`) links 18 extension nodes starting at `extension_start` (each beginning with `PJRT_Extension_Base { size_t struct_size; PJRT_Extension_Type type; PJRT_Extension_Base* next; }` = 24 bytes):

| Node | Variable Name in `libtpu.so` | `struct_size` | `PJRT_Extension_Type` (`offset 0x08`) | Function Pointer Slots (`offset 0x18..`) |
| ---: | :--- | ---: | ---: | :--- |
| 1 | `ffi_extension` | `48` (`0x30`) | `5` (`0x05`) | `0x18`: `type_register`, `0x20`: `user_data_add`, `0x28`: `register_handler` |
| 2 | `xla_transform_extension` | `56` (`0x38`) | `24` (`0x18`) | `0x18`: `register_xla_transform`, `0x20`: `wait_for_all_transforms`, `0x28`: `unregister_xla_transform`, `0x30`: `list_xla_transforms` |
| 3 | `host_memory_allocator_extension` | `32` (`0x20`) | `23` (`0x17`) | `0x18`: `get_host_memory_allocator` |
| 4 | `multi_slice_extension` | `64` (`0x40`) | `22` (`0x16`) | `0x18`: `config_destroy`, `0x20`: `config_num_slices`, `0x28`: `config_num_devices_per_slice`, `0x30`: `config_slice_id`, `0x38`: `config_serialize` |
| 5 | `collectives_extension` | `96` (`0x60`) | `21` (`0x15`) | `0x18`: `collectives_destroy`, `0x20`: `create_communicator`, `0x28`: `communicator_destroy`, `0x30`: `communicator_all_reduce`, `0x38`: `communicator_reduce_scatter`, `0x40`: `communicator_all_gather`, `0x48`: `communicator_collective_permute`, `0x50`: `communicator_all_to_all`, `0x58`: `communicator_to_string` |
| 6 | `abi_version_extension` | `120` (`0x78`) | `20` (`0x14`) | 12 ABI version handles, compatibility checks, platform IDs, and proto codecs (`0x18..0x70`) |
| 7 | `shardings_extension` | `40` (`0x28`) | `19` (`0x13`) | `0x18`: `parameter_shardings`, `0x20`: `output_shardings` |
| 8 | `megascale_extension` | `288` (`0x120`) | `18` (`0x12`) | 33 slots (`0x18..0x118`, with `0x40..0x60` reserved NULL; adds `discover_slices`, `multi_slice_config_get_incarnation_id`, `multi_slice_config_get_local_host_id`, `multi_slice_config_get_num_hosts`, `global_barrier`) |
| 9 | `tpu_executable_extension` | `96` (`0x60`) | `17` (`0x11`) | 9 slots (`0x18..0x58`, `0x28` reserved NULL) |
| 10 | `tpu_topology_extension` | `280` (`0x118`) | `16` (`0x10`) | 32 function pointers (`0x18..0x110`) |
| 11 | `callback_extension` | `40` (`0x28`) | `14` (`0x0e`) | `0x18`: `register_callback`, `0x20`: `invoke_callback` |
| 12 | `phase_compile_extension` | `192` (`0xc0`) | `9` (`0x09`) | 21 slots (`0x18..0xb8`, adding `PJRT_Relocatable_*` and `CompileToRelocatable` / `DeserializeRelocatable` / `LinkRelocatables`) |
| 13 | `cross_host_transfers_extension` | `56` (`0x38`) | `12` (`0x0c`) | 4 function pointers (`0x18..0x30`) |
| 14 | `executable_metadata_extension` | `40` (`0x28`) | `13` (`0x0d`) | `0x18`: `get_executable_metadata`, `0x20`: `destroy_executable_metadata` |
| 15 | `mem_desc_extension` | `40` (`0x28`) | `6` (`0x06`) | `0x18`: `get_memory_descriptions`, `0x20`: `destroy_memory_descriptions` |
| 16 | `layouts_extension` | `80` (`0x50`) | `4` (`0x04`) | 7 function pointers (`0x18..0x48`) |
| 17 | `raw_buffer_extension` | `80` (`0x50`) | `8` (`0x08`) | 7 function pointers (`0x18..0x48`) |
| 18 | `profiler_extension` | `40` (`0x28`) | `1` (`0x01`) | `0x18`: `profiler_api` (`*const PLUGIN_Profiler_Api`, `104` bytes / 11 functions at `0x13e99190`), `0x20`: `traceme_context_id` (`int64_t`) |

## 4. `LibtpuSdk_Api` Vtable & Argument Structs (`GetLibtpuSdkApi` @ `0x07fccb78`)
`CreateLibtpuSdkApi` (`0x07fcc980`) allocates a `0x90` (144-byte) structure containing:
- Offset `0x00`: Version header (`0x0000000100000000` -> `major = 0`, `minor = 1`)
- Offset `0x08`: `LibtpuSdk_Error_GetMessage(LibtpuSdk_Error_GetMessage_Args*)` — `{ const LibtpuSdk_Error* error; const char* message; size_t message_size; }` (24 bytes, returns `void`)
- Offset `0x10`: `LibtpuSdk_Error_Destroy(LibtpuSdk_Error_Destroy_Args*)` — `{ LibtpuSdk_Error* error; }` (8 bytes, returns `void`)
- Offset `0x18`: `LibtpuSdk_Error_GetCode(LibtpuSdk_Error_GetCode_Args*)` — `{ const LibtpuSdk_Error* error; int32_t code; }` (16 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x20`: `LibtpuSdk_Client_Create(LibtpuSdk_Client_Create_Args*)` — `{ LibtpuSdk_Client* client; }` (8 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x28`: `LibtpuSdk_Client_Destroy(LibtpuSdk_Client_Destroy_Args*)` — `{ LibtpuSdk_Client* client; }` (8 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x30`: `LibtpuSdk_GetChipCoordinates(LibtpuSdk_GetChipCoordinates_Args*)` — `{ LibtpuSdk_Client* client; const LibtpuSdk_ChipLocationToCoordinates* coords; size_t num_coords; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x38`: `LibtpuSdk_GetHostName(LibtpuSdk_GetHostName_Args*)` — `{ const LibtpuSdk_ChipLocationToCoordinates* coord; const char* host_name; size_t host_name_size; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x40`: `LibtpuSdk_GetChipIndex(LibtpuSdk_GetChipIndex_Args*)` — `{ const LibtpuSdk_ChipLocationToCoordinates* coord; int32_t chip_index; }` (16 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x48`: `LibtpuSdk_GetCartesianCoordinates(LibtpuSdk_GetCartesianCoordinates_Args*)` — `{ const LibtpuSdk_ChipLocationToCoordinates* coord; const int64_t* coordinates; size_t num_coordinates; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x50`: `LibtpuSdk_GetMetric(LibtpuSdk_GetMetric_Args*)` — `{ LibtpuSdk_Client* client; const char* metric_name; LibtpuSdk_Metric* metric; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x58`: `LibtpuSdk_GetMetricDescription(LibtpuSdk_GetMetricDescription_Args*)` — `{ const LibtpuSdk_Metric* metric; const char* description; size_t description_size; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x60`: `LibtpuSdk_GetMetricValues(LibtpuSdk_GetMetricValues_Args*)` — `{ const LibtpuSdk_Metric* metric; const char* const* values; size_t num_values; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x68`: `LibtpuSdk_GetRuntimeStatus(LibtpuSdk_GetRuntimeStatus_Args*)` — `{ LibtpuSdk_Client* client; bool include_hlo_info; LibtpuSdk_RuntimeStatus* status; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x70`: `LibtpuSdk_RuntimeStatus_GetCoreStateSummary(LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args*)` — `{ const LibtpuSdk_RuntimeStatus* status; const char* summary; size_t summary_size; }` (24 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x78`: `LibtpuSdk_RuntimeStatus_Destroy(LibtpuSdk_RuntimeStatus_Destroy_Args*)` — `{ LibtpuSdk_RuntimeStatus* status; }` (8 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x80`: `LibtpuSdk_RegisterDebugCallback(LibtpuSdk_RegisterDebugCallback_Args*)` — `{ LibtpuSdk_Client* client; void* callback; void* user_data; int64_t callback_id; }` (32 bytes, returns `LibtpuSdk_Error*`)
- Offset `0x88`: `LibtpuSdk_UnregisterDebugCallback(LibtpuSdk_UnregisterDebugCallback_Args*)` — `{ LibtpuSdk_Client* client; int64_t callback_id; }` (16 bytes, returns `LibtpuSdk_Error*`)

## 5. Field-by-Field Byte Offsets of `PJRT_*_Args` Structures

Every `PJRT_*_Args` structure in `libtpu.so` begins with the 16-byte header:
- `0x00`: `size_t struct_size`
- `0x08`: `PJRT_Extension_Base* extension_start` (`NULL` unless chained)

Followed by input fields and output fields at 8-byte aligned offsets:
- **24-byte `Args` (`struct_size == 24`)**:
  - `0x10`: single handle pointer (`error`, `event`, `client`, `executable`, `buffer`, `stream`, `topology`, `context`, `transfer_manager`, `async_tracking_event`, or `device`).
- **32-byte `Args` (`struct_size == 32`)**:
  - `0x10`: input handle pointer (`error`, `event`, `client`, `device_description`, `device`, `memory`, `executable`, `buffer`, `stream`, `topology`, `transfer_manager`, `data_ptr`, or `processes`).
  - `0x18`: 64-bit output or scalar (`code`, `is_ready`, `process_index`, `id`, `is_addressable`, `local_hardware_id`, `default_memory`, `num_replicas`, `num_partitions`, `num_outputs`, `size_in_bytes`, `is_deleted`, `type`, `on_device_size_in_bytes`, `is_on_cpu`, `unsafe_pointer`, `opaque_device_memory_data_pointer`, `total_bytes`, `granule_size_in_bytes`, `current_bytes`, `fingerprint`, `kind_id`, `buffer_count`, `num_processes`, or `size`).
- **40-byte `Args` (`struct_size == 40`)**:
  - `0x10`: input handle pointer.
  - `0x18`: second input/output field (`message` / `platform_name` / `devices` / `attributes` / `dims` / `kind` / `dst_device` / `dst_memory` / `chunk` / `callback` / `buffer_index` / `error_code` / `payload_visitor`).
  - `0x20`: third input/output field (`message_size` / `num_devices` / `num_attributes` / `num_dims` / `kind_size` / `dst_buffer` / `transfer_complete` / `user_arg` / `buffer_out` / `buffer_size` / `error_message`).
- **48-byte `Args` (`struct_size == 48`)**:
  - `0x10`: input handle (`executable`, `topology`, `client`, `device`, `buffer`, `transfer_manager`).
  - `0x18`: second field (`num_outputs` / `serialized_bytes` / `topology_name` / `shape_specs` / `buffer_index` / `launch_id` / `description` / `dependency_event`).
  - `0x20`: third field (`memory_kinds` / `serialized_bytes_size` / `topology_name_size` / `num_shape_specs` / `error_code` / `description_size` / `callback_data`).
  - `0x28`: fourth field (`memory_kind_sizes` / `serialized_executable` / `create_options` / `device` / `error_message` / `event` / `donated_buffer`).
  - `0x30` (when 56 bytes): fifth field (`serialized_executable_deleter` / `transfer_manager_out` / `poisoned`).
- **56-byte `Args` (`struct_size == 56`)**:
  - `PJRT_Client_Compile_Args`: `0x10` `client`, `0x18` `program`, `0x20` `compile_options`, `0x28` `compile_options_size`, `0x30` `executable`.
  - `PJRT_Compile_Args`: `0x10` `topology`, `0x18` `program`, `0x20` `compile_options`, `0x28` `compile_options_size`, `0x30` `client`, `0x38` (or packed 56B) `executable`.
  - `PJRT_Client_DefaultDeviceAssignment_Args`: `0x10` `client`, `0x18` `num_replicas` (`i32`) + `num_partitions` (`i32`), `0x20` `default_assignment_size`, `0x28` `default_assignment`, `0x30` `reserved`.
  - `PJRT_LoadedExecutable_Execute_Args`: `0x10` `executable`, `0x18` `options`, `0x20` `argument_lists`, `0x28` `num_devices`, `0x30` `num_args`, `0x38` `output_lists`, `0x40` `device_complete_events`.
  - `PJRT_Executable_DeserializeAndLoad_Args`: `0x10` `client`, `0x18` `serialized_executable`, `0x20` `serialized_executable_size`, `0x28` `loaded_executable`, `0x30` `overridden_serialized_compile_options`.
  - `PJRT_Buffer_ToHostBuffer_Args`: `0x10` `src`, `0x18` `host_layout`, `0x20` `dst`, `0x28` `dst_size`, `0x30` `event`.
  - `PJRT_Buffer_CopyRawToHost_Args`: `0x10` `buffer`, `0x18` `dst`, `0x20` `offset`, `0x28` `transfer_size`, `0x30` `event`.
- **64-byte `Args` (`struct_size == 64`)**:
  - `PJRT_Buffer_GetMemoryLayout_Args`: `0x10` `buffer`, `0x18..0x40` `layout` (`PJRT_Buffer_MemoryLayout`, `40` bytes).
  - `PJRT_Executable_GetCompiledMemoryStats_Args`: `0x10` `executable`, `0x18..0x40` memory stat counters (`generated_code_size_in_bytes`, `argument_size_in_bytes`, `output_size_in_bytes`, `alias_size_in_bytes`, `temp_size_in_bytes`).
  - `PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`: `0x10` `transfer_manager`, `0x18` `buffer_index`, `0x20` `data`, `0x28` `offset`, `0x30` `transfer_size`, `0x38` `is_last_transfer`, `0x40` (or packed 64B) `done_with_h2d_transfer`.
  - `PJRT_Client_CreateErrorBuffer_Args`: `0x10` `client`, `0x18` `dims`, `0x20` `num_dims`, `0x28` `element_type`, `0x30` `error_code` + `error_message`, `0x38` `buffer`.
  - `PJRT_Buffer_Bitcast_Args`: `0x10` `buffer`, `0x18` `element_type`, `0x20` `dims`, `0x28` `num_dims`, `0x30` `device_layout`, `0x38` `out_buffer`.
- **72-byte `Args` (`struct_size == 72`)**:
  - `PJRT_Client_Create_Args`: `0x10` `create_options`, `0x18` `num_options`, `0x20` `kv_get_callback`, `0x28` `kv_get_user_arg`, `0x30` `kv_put_callback`, `0x38` `kv_put_user_arg`, `0x40` `client`.
  - `PJRT_Client_CreateUninitializedBuffer_Args`: `0x10` `client`, `0x18` `dims`, `0x20` `num_dims`, `0x28` `element_type`, `0x30` `device_layout`, `0x38` `device`, `0x40` `buffer`.
  - `PJRT_Client_CreateAliasBuffer_Args`: `0x10` `client`, `0x18` `memory`, `0x20` `dims`, `0x28` `num_dims`, `0x30` `element_type`, `0x38` `alias_buffer`, `0x40` `fulfill_alias_buffer_cb`.
- **88-byte `Args` (`struct_size == 88`)**:
  - `PJRT_Client_CreateViewOfDeviceBuffer_Args`: `0x10` `client`, `0x18` `device_buffer_ptr`, `0x20` `dims`, `0x28` `num_dims`, `0x30` `element_type`, `0x38` `layout`, `0x40` `device`, `0x48` `on_delete_callback`, `0x50` `buffer`.
- **112-byte `Args` (`struct_size == 112`)**:
  - `PJRT_Client_BufferFromHostBuffer_Args`: `0x10` `client`, `0x18` `data`, `0x20` `element_type`, `0x28` `dims`, `0x30` `num_dims`, `0x38` `byte_strides`, `0x40` `num_byte_strides`, `0x48` `host_buffer_semantics`, `0x50` `device`, `0x58` `memory`, `0x60` `device_layout`, `0x68` `done_with_host_buffer` / `buffer`.
  - `PJRT_Device_MemoryStats_Args`: `0x10` `device`, `0x18` `bytes_in_use`, `0x20` `peak_bytes_in_use`, `0x28` `peak_bytes_in_use_is_set`, `0x30` `num_allocs`, `0x38` `num_allocs_is_set`, `0x40` `largest_alloc_size`, `0x48` `largest_alloc_size_is_set`, `0x50` `bytes_limit`, `0x58` `bytes_limit_is_set`, `0x60` `bytes_reserved`, `0x68` `bytes_reserved_is_set`.

## 6. Extension & Profiler `Args` Struct Layouts (162 Active Functions Across 18 Extensions)
Every extension function in the 18-node `PJRT_Extension_Base` chain takes a `#[repr(C)]` `Args` struct beginning with `size_t struct_size` (`0x00`) and `PJRT_Extension_Base* extension_start` (`0x08`, or `void* priv` for `PLUGIN_Profiler_Api` which is `104` bytes with `struct_size`, `priv`, `error_destroy`, `error_message`, `error_get_code`, `create`, `destroy`, `start`, `stop`, `collect_data`):
- **Node 1 (`PJRT_XlaTransform_Extension`, `ext_type = 29`, `56` B)**: `Register` (`40` B: `name`, `name_size`, `transform_fn`), `WaitForAll` (`24` B: `client`), `Unregister` (`32` B: `name`, `name_size`), `List` (`32` B: `names_out`, `num_names_out`).
- **Node 2 (`PJRT_HostMemoryAllocator_Extension`, `ext_type = 28`, `32` B)**: `Get` (`32` B: `client`, `allocator_out`).
- **Node 3 (`PJRT_MultiSlice_Extension`, `ext_type = 26`, `64` B)**: `Config_Destroy` (`24` B: `config`), `Config_NumSlices` (`32` B: `config`, `num_slices_out`), `Config_NumDevicesPerSlice` (`40` B: `config`, `slice_ids`, `num_devices_out`), `Config_SliceId` (`32` B: `config`, `slice_id_out`), `Config_Serialize` (`48` B: `config`, `bytes_out`, `size_out`, `deleter_out`).
- **Node 4 (`PJRT_Collectives_Extension`, `ext_type = 25`, `96` B)**: `Destroy`, `CreateCommunicator`, `Communicator_Destroy`, `AllReduce`, `ReduceScatter`, `AllGather`, `CollectivePermute`, `AllToAll`, `ToString` (`24..64` B).
- **Node 5 (`PJRT_AbiVersion_Extension`, `ext_type = 20`, `120` B)**: 12 functions (`24..48` B) managing `PJRT_RuntimeAbiVersion` and `PJRT_ExecutableAbiVersion` handles, compatibility checks, platform IDs, and proto serialization.
- **Node 6 (`PJRT_Shardings_Extension`, `ext_type = 19`, `40` B)**: `ParameterShardings` (`48` B) and `OutputShardings` (`48` B) returning serialized `xla.OpSharding` protobuf arrays.
- **Node 7 (`PJRT_Megascale_Extension`, `ext_type = 18`, `248` B)**: 28 functions (`24..56` B) managing client context, AOT/default/multi-slice configs, coordinator addresses, barriers, error handlers, DCN topology, metrics, transport type, and collective aborts.
- **Node 8 (`PJRT_TpuExecutable_Extension`, `ext_type = 17`, `96` B)**: 9 functions (`24..64` B) extracting target arguments, `HloModuleProtoWithConfig`, compiled memory stats, HLO cost analysis, `TpuCompilationEnvironmentProto` fields, and predetermined/cancellation error flags.
- **Node 9 (`PJRT_TpuTopology_Extension`, `ext_type = 16`, `272` B)**: 31 functions (`24..64` B) querying subslice topologies, host bounds, enhanced barrier/ICI reachability, process/chip/core/logical-device counts and coordinate mappings, routing strategy, and slice/platform configs.
- **Node 10 (`PJRT_Callback_Extension`, `ext_type = 14`, `40` B)**: `RegisterCallback` (`48` B: `client`, `callback_type`, `callback_fn`, `user_arg`, `callback_id_out`) and `InvokeCallback` (`40` B: `client`, `callback_id`, `args`).
- **Node 11 (`PJRT_PhaseCompile_Extension`, `ext_type = 9`, `64` B)**: `GetCompiler` (`32` B), `DestroyCompiler` (`24` B), `RunPhase` (`56` B), `GetPhaseNames` (`40` B), `CBuffersDestroy` (`32` B).
- **Node 12 (`PJRT_CrossHostTransfers_Extension`, `ext_type = 12`, `56` B)**: `MakeReceiveBuffers`, `CopyToRemoteDevice`, `ReceiveBuffers`, `SendBuffers` (`48..56` B).
- **Node 13 (`PJRT_ExecutableMetadata_Extension`, `ext_type = 11`, `40` B)**: `Get` (`32` B) and `Destroy` (`24` B).
- **Node 14 (`PJRT_MemoryDescriptions_Extension`, `ext_type = 7`, `40` B)**: `Get` (`40` B) and `Destroy` (`24` B).
- **Node 15 (`PJRT_Layouts_Extension`, `ext_type = 6`, `80` B)**: `MemoryLayout_Destroy` (`24` B), `MemoryLayout_Serialize` (`48` B), `Client_GetDefaultLayout` (`56` B), `Buffer_GetMemoryLayout` (`32` B), `Topology_GetDefaultLayout` (`56` B), `Executable_GetOutputLayouts` (`40` B), `Executable_GetParameterLayouts` (`40` B).
- **Node 16 (`PJRT_RawBuffer_Extension`, `ext_type = 8`, `80` B)**: `CreateRawAliasOfBuffer` (`40` B), `Destroy` (`24` B), `GetHostPointer` (`32` B), `GetOnDeviceSizeInBytes` (`32` B), `GetMemorySpace` (`32` B), `CopyRawHostToDevice` (`56` B), `CopyRawDeviceToHost` (`56` B).
- **Node 17 (`PJRT_Profiler_Extension`, `ext_type = 1`, `40` B) & `PLUGIN_Profiler_Api` (`80` B)**: `Error_Destroy` (`24` B), `Error_Message` (`40` B), `Error_GetCode` (`32` B), `Create` (`40` B), `Destroy` (`24` B), `Start` (`24` B), `Stop` (`24` B), `CollectData` (`40` B: `profiler`, `buffer`, `size_in_bytes` producing wire-compatible `tensorflow.profiler.XSpace` bytes).


