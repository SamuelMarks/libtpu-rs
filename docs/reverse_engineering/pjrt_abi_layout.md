# `PJRT_Api` & `LibtpuSdk_Api` Binary Layout Specification (`libtpu-0.0.50`)

## 1. `PJRT_Api` Header & Version Layout
Disassembly of `pjrt::CreatePjrtApi` (`0x0c5a55e0`) in `libtpu.so` reveals the exact binary header of the static `PJRT_Api` structure (`_ZZN4pjrt10tpu_plugin13GetTpuPjrtApiEvE8pjrt_api` at `0x1c8edd80`):

- **Total `PJRT_Api` Size (`struct_size`)**: `0x478` = **1,144 bytes** (143 64-bit words)
- **`PJRT_Api_Version`**:
  - `struct_size` (offset `0x10`): `0x18` = **24 bytes**
  - `extension_start` (offset `0x18`): `NULL` (`0x0`)
  - `major_version` (offset `0x20`, `int32_t`): **`0`**
  - `minor_version` (offset `0x24`, `int32_t`): **`116`** (`0x74`)

## 2. Complete `PJRT_Api` Function Pointer Vtable (Offsets `0x000` .. `0x470`) & `Args` Struct Layouts
| Offset | Slot # | Field Name | Internal Implementation Symbol in `libtpu.so` | `Args` Struct Size (`bytes`) |
| :--- | ---: | :--- | :--- | ---: |
| `0x000` | 0 | `struct_size` | `1144` (`0x478`) | — |
| `0x008` | 1 | `extension_start` | Pointer to `xla_transform_extension` (head of 17-node linked list) | — |
| `0x010` | 2..4 | `pjrt_api_version` | `{ struct_size: 24, extension_start: NULL, major: 0, minor: 116 }` | 24 |
| `0x028` | 5 | `PJRT_Error_Destroy` | `pjrt::PJRT_Error_Destroy` (`0x0c591220`) | 24 |
| `0x030` | 6 | `PJRT_Error_Message` | `pjrt::PJRT_Error_Message` (`0x0c591300`) | 40 |
| `0x038` | 7 | `PJRT_Error_GetCode` | `pjrt::PJRT_Error_GetCode` (`0x0c5913f0`) | 32 |
| `0x040` | 8 | `PJRT_Plugin_Initialize` | `pjrt::tpu_plugin::PJRT_Plugin_Initialize` (`0x0b4fbbd0`) | 16 |
| `0x048` | 9 | `PJRT_Plugin_Attributes` | `pjrt::PJRT_Plugin_Attributes_Xla` (`0x0c5914c0`) | 32 |
| `0x050` | 10 | `PJRT_Event_Destroy` | `pjrt::PJRT_Event_Destroy` (`0x0c5a12a0`) | 24 |
| `0x058` | 11 | `PJRT_Event_IsReady` | `pjrt::PJRT_Event_IsReady` (`0x0c5a1350`) | 32 |
| `0x060` | 12 | `PJRT_Event_Error` | `pjrt::PJRT_Event_Error` (`0x0c5a1510`) | 24 |
| `0x068` | 13 | `PJRT_Event_Await` | `pjrt::PJRT_Event_Await` (`0x0c5a13e0`) | 24 |
| `0x070` | 14 | `PJRT_Event_OnReady` | `pjrt::PJRT_Event_OnReady` (`0x0c5a15d0`) | 40 |
| `0x078` | 15 | `PJRT_Client_Create` | `pjrt::tpu_plugin::PJRT_Client_Create` (`0x0b4fa500`) | 72 |
| `0x080` | 16 | `PJRT_Client_Destroy` | `pjrt::PJRT_Client_Destroy` (`0x0c591520`) | 24 |
| `0x088` | 17 | `PJRT_Client_PlatformName` | `pjrt::PJRT_Client_PlatformName` (`0x0c591770`) | 40 |
| `0x090` | 18 | `PJRT_Client_ProcessIndex` | `pjrt::PJRT_Client_ProcessIndex` (`0x0c591720`) | 32 |
| `0x098` | 19 | `PJRT_Client_PlatformVersion` | `pjrt::PJRT_Client_PlatformVersion` (`0x0c5917d0`) | 40 |
| `0x0a0` | 20 | `PJRT_Client_Devices` | `pjrt::PJRT_Client_Devices` (`0x0c5918b0`) | 40 |
| `0x0a8` | 21 | `PJRT_Client_AddressableDevices` | `pjrt::PJRT_Client_AddressableDevices` (`0x0c591910`) | 40 |
| `0x0b0` | 22 | `PJRT_Client_LookupDevice` | `pjrt::PJRT_Client_LookupDevice` (`0x0c591970`) | 40 |
| `0x0b8` | 23 | `PJRT_Client_LookupAddressableDevice` | `pjrt::PJRT_Client_LookupAddressableDevice` (`0x0c591ba0`) | 40 |
| `0x0c0` | 24 | `PJRT_Client_AddressableMemories` | `pjrt::PJRT_Client_AddressableMemories` (`0x0c591f70`) | 40 |
| `0x0c8` | 25 | `PJRT_Client_Compile` | `pjrt::PJRT_Client_Compile` (`0x0c593890`) | 56 |
| `0x0d0` | 26 | `PJRT_Client_DefaultDeviceAssignment` | `pjrt::PJRT_Client_DefaultDeviceAssignment` (`0x0c595000`) | 56 |
| `0x0d8` | 27 | `PJRT_Client_BufferFromHostBuffer` | `pjrt::PJRT_Client_BufferFromHostBuffer` (`0x0c596420`) | 112 |
| `0x0e0` | 28 | `PJRT_DeviceDescription_Id` | `pjrt::PJRT_DeviceDescription_Id` (`0x0c5971f0`) | 32 |
| `0x0e8` | 29 | `PJRT_DeviceDescription_ProcessIndex` | `pjrt::PJRT_DeviceDescription_ProcessIndex` (`0x0c597240`) | 32 |
| `0x0f0` | 30 | `PJRT_DeviceDescription_Attributes` | `pjrt::PJRT_DeviceDescription_Attributes` (`0x0c597290`) | 40 |
| `0x0f8` | 31 | `PJRT_DeviceDescription_Kind` | `pjrt::PJRT_DeviceDescription_Kind` (`0x0c5972f0`) | 40 |
| `0x100` | 32 | `PJRT_DeviceDescription_DebugString` | `pjrt::PJRT_DeviceDescription_DebugString` (`0x0c597360`) | 40 |
| `0x108` | 33 | `PJRT_DeviceDescription_ToString` | `pjrt::PJRT_DeviceDescription_ToString` (`0x0c5976d0`) | 40 |
| `0x110` | 34 | `PJRT_Device_GetDescription` | `pjrt::PJRT_Device_GetDescription` (`0x0c5977b0`) | 32 |
| `0x118` | 35 | `PJRT_Device_IsAddressable` | `pjrt::PJRT_Device_IsAddressable` (`0x0c597800`) | 32 |
| `0x120` | 36 | `PJRT_Device_LocalHardwareId` | `pjrt::PJRT_Device_LocalHardwareId` (`0x0c597850`) | 32 |
| `0x128` | 37 | `PJRT_Device_AddressableMemories` | `pjrt::PJRT_Device_AddressableMemories` (`0x0c5978a0`) | 40 |
| `0x130` | 38 | `PJRT_Device_DefaultMemory` | `pjrt::PJRT_Device_DefaultMemory` (`0x0c597900`) | 32 |
| `0x138` | 39 | `PJRT_Device_MemoryStats` | `pjrt::PJRT_Device_MemoryStats` (`0x0c5979a0`) | 112 |
| `0x140` | 40 | `PJRT_Memory_Id` | `pjrt::PJRT_Memory_Id` (`0x0c597b70`) | 32 |
| `0x148` | 41 | `PJRT_Memory_Kind` | `pjrt::PJRT_Memory_Kind` (`0x0c597bd0`) | 40 |
| `0x150` | 42 | `PJRT_Memory_DebugString` | `pjrt::PJRT_Memory_DebugString` (`0x0c597ca0`) | 40 |
| `0x158` | 43 | `PJRT_Memory_ToString` | `pjrt::PJRT_Memory_ToString` (`0x0c597d10`) | 40 |
| `0x160` | 44 | `PJRT_Memory_AddressableByDevices` | `pjrt::PJRT_Memory_AddressableByDevices` (`0x0c597d80`) | 40 |
| `0x168` | 45 | `PJRT_Executable_Destroy` | `pjrt::PJRT_Executable_Destroy` (`0x0c597e70`) | 24 |
| `0x170` | 46 | `PJRT_Executable_Name` | `pjrt::PJRT_Executable_Name` (`0x0c5984c0`) | 40 |
| `0x178` | 47 | `PJRT_Executable_NumReplicas` | `pjrt::PJRT_Executable_NumReplicas` (`0x0c598520`) | 32 |
| `0x180` | 48 | `PJRT_Executable_NumPartitions` | `pjrt::PJRT_Executable_NumPartitions` (`0x0c598580`) | 32 |
| `0x188` | 49 | `PJRT_Executable_NumOutputs` | `pjrt::PJRT_Executable_NumOutputs` (`0x0c5986a0`) | 32 |
| `0x190` | 50 | `PJRT_Executable_SizeOfGeneratedCodeInBytes` | `pjrt::PJRT_Executable_SizeOfGeneratedCodeInBytes` (`0x0c598e70`) | 32 |
| `0x198` | 51 | `PJRT_Executable_GetCostAnalysis` | `pjrt::PJRT_Executable_GetCostAnalysis` (`0x0c599740`) | 40 |
| `0x1a0` | 52 | `PJRT_Executable_OutputMemoryKinds` | `pjrt::PJRT_Executable_OutputMemoryKinds` (`0x0c59b100`) | 48 |
| `0x1a8` | 53 | `PJRT_Executable_OptimizedProgram` | `pjrt::PJRT_Executable_OptimizedProgram` (`0x0c598ed0`) | 32 |
| `0x1b0` | 54 | `PJRT_Executable_Serialize` | `pjrt::PJRT_Executable_Serialize` (`0x0c59e150`) | 48 |
| `0x1b8` | 55 | `PJRT_LoadedExecutable_Destroy` | `pjrt::PJRT_LoadedExecutable_Destroy` (`0x0c598400`) | 24 |
| `0x1c0` | 56 | `PJRT_LoadedExecutable_GetExecutable` | `pjrt::PJRT_LoadedExecutable_GetExecutable` (`0x0c59edb0`) | 32 |
| `0x1c8` | 57 | `PJRT_LoadedExecutable_AddressableDevices` | `pjrt::PJRT_LoadedExecutable_AddressableDevices` (`0x0c5985e0`) | 40 |
| `0x1d0` | 58 | `PJRT_LoadedExecutable_Delete` | `pjrt::PJRT_LoadedExecutable_Delete` (`0x0c59b660`) | 24 |
| `0x1d8` | 59 | `PJRT_LoadedExecutable_IsDeleted` | `pjrt::PJRT_LoadedExecutable_IsDeleted` (`0x0c59b6b0`) | 32 |
| `0x1e0` | 60 | `PJRT_LoadedExecutable_Execute` | `pjrt::PJRT_LoadedExecutable_Execute` (`0x0c59b700`) | 56 |
| `0x1e8` | 61 | `PJRT_Executable_DeserializeAndLoad` | `pjrt::PJRT_Executable_DeserializeAndLoad` (`0x0c59ea10`) | 56 |
| `0x1f0` | 62 | `PJRT_LoadedExecutable_Fingerprint` | `pjrt::PJRT_LoadedExecutable_Fingerprint` (`0x0c591ef0`) | 40 |
| `0x1f8` | 63 | `PJRT_Buffer_Destroy` | `pjrt::PJRT_Buffer_Destroy` (`0x0c59ee20`) | 24 |
| `0x200` | 64 | `PJRT_Buffer_ElementType` | `pjrt::PJRT_Buffer_ElementType` (`0x0c59f000`) | 32 |
| `0x208` | 65 | `PJRT_Buffer_Dimensions` | `pjrt::PJRT_Buffer_Dimensions` (`0x0c59f060`) | 40 |
| `0x210` | 66 | `PJRT_Buffer_UnpaddedDimensions` | `pjrt::PJRT_Buffer_UnpaddedDimensions` (`0x0c59f0d0`) | 40 |
| `0x218` | 67 | `PJRT_Buffer_DynamicDimensionIndices` | `pjrt::PJRT_Buffer_DynamicDimensionIndices` (`0x0c59f280`) | 40 |
| `0x220` | 68 | `PJRT_Buffer_GetMemoryLayout` | `pjrt::PJRT_Buffer_GetMemoryLayout` (`0x0c59f3a0`) | 64 |
| `0x228` | 69 | `PJRT_Buffer_OnDeviceSizeInBytes` | `pjrt::PJRT_Buffer_OnDeviceSizeInBytes` (`0x0c59f820`) | 32 |
| `0x230` | 70 | `PJRT_Buffer_Device` | `pjrt::PJRT_Buffer_Device` (`0x0c59f8d0`) | 32 |
| `0x238` | 71 | `PJRT_Buffer_Memory` | `pjrt::PJRT_Buffer_Memory` (`0x0c59f9e0`) | 32 |
| `0x240` | 72 | `PJRT_Buffer_Delete` | `pjrt::PJRT_Buffer_Delete` (`0x0c59fbf0`) | 24 |
| `0x248` | 73 | `PJRT_Buffer_IsDeleted` | `pjrt::PJRT_Buffer_IsDeleted` (`0x0c59fc40`) | 32 |
| `0x250` | 74 | `PJRT_Buffer_CopyToDevice` | `pjrt::PJRT_Buffer_CopyToDevice` (`0x0c5a0040`) | 40 |
| `0x258` | 75 | `PJRT_Buffer_ToHostBuffer` | `pjrt::PJRT_Buffer_ToHostBuffer` (`0x0c5a02f0`) | 56 |
| `0x260` | 76 | `PJRT_Buffer_IsOnCpu` | `pjrt::PJRT_Buffer_IsOnCpu` (`0x0c5a08d0`) | 32 |
| `0x268` | 77 | `PJRT_Buffer_ReadyEvent` | `pjrt::PJRT_Buffer_ReadyEvent` (`0x0c5a0930`) | 32 |
| `0x270` | 78 | `PJRT_Buffer_UnsafePointer` | `pjrt::PJRT_Buffer_UnsafePointer` (`0x0c5a09b0`) | 32 |
| `0x278` | 79 | `PJRT_Buffer_IncreaseExternalReferenceCount` | `pjrt::PJRT_Buffer_IncreaseExternalReferenceCount` (`0x0c5a0a60`) | 24 |
| `0x280` | 80 | `PJRT_Buffer_DecreaseExternalReferenceCount` | `pjrt::PJRT_Buffer_DecreaseExternalReferenceCount` (`0x0c5a0c30`) | 24 |
| `0x288` | 81 | `PJRT_Buffer_OpaqueDeviceMemoryDataPointer` | `pjrt::PJRT_Buffer_OpaqueDeviceMemoryDataPointer` (`0x0c5a0d20`) | 32 |
| `0x290` | 82 | `PJRT_CopyToDeviceStream_Destroy` | `pjrt::PJRT_CopyToDeviceStream_Destroy` (`0x0c5a1060`) | 24 |
| `0x298` | 83 | `PJRT_CopyToDeviceStream_AddChunk` | `pjrt::PJRT_CopyToDeviceStream_AddChunk` (`0x0c5a10d0`) | 40 |
| `0x2a0` | 84 | `PJRT_CopyToDeviceStream_TotalBytes` | `pjrt::PJRT_CopyToDeviceStream_TotalBytes` (`0x0c5a1190`) | 32 |
| `0x2a8` | 85 | `PJRT_CopyToDeviceStream_GranuleSize` | `pjrt::PJRT_CopyToDeviceStream_GranuleSize` (`0x0c5a11e0`) | 32 |
| `0x2b0` | 86 | `PJRT_CopyToDeviceStream_CurrentBytes` | `pjrt::PJRT_CopyToDeviceStream_CurrentBytes` (`0x0c5a1230`) | 32 |
| `0x2b8` | 87 | `PJRT_TopologyDescription_Create` | `pjrt::tpu_plugin::PJRT_TopologyDescription_Create` (`0x0b4fba20`) | 48 |
| `0x2c0` | 88 | `PJRT_TopologyDescription_Destroy` | `pjrt::PJRT_TopologyDescription_Destroy` (`0x0c5a1900`) | 24 |
| `0x2c8` | 89 | `PJRT_TopologyDescription_PlatformName` | `pjrt::PJRT_TopologyDescription_PlatformName` (`0x0c5a1aa0`) | 40 |
| `0x2d0` | 90 | `PJRT_TopologyDescription_PlatformVersion` | `pjrt::PJRT_TopologyDescription_PlatformVersion` (`0x0c5a1b00`) | 40 |
| `0x2d8` | 91 | `PJRT_TopologyDescription_GetDeviceDescriptions` | `pjrt::PJRT_TopologyDescription_GetDeviceDescriptions` (`0x0c5a1b60`) | 40 |
| `0x2e0` | 92 | `PJRT_TopologyDescription_Serialize` | `pjrt::PJRT_TopologyDescription_Serialize` (`0x0c5a1bc0`) | 48 |
| `0x2e8` | 93 | `PJRT_TopologyDescription_Attributes` | `pjrt::PJRT_TopologyDescription_Attributes` (`0x0c5a1e50`) | 40 |
| `0x2f0` | 94 | `PJRT_Compile` | `pjrt::PJRT_Compile` (`0x0c5a2280`) | 56 |
| `0x2f8` | 95 | `PJRT_Executable_OutputElementTypes` | `pjrt::PJRT_Executable_OutputElementTypes` (`0x0c59a140`) | 40 |
| `0x300` | 96 | `PJRT_Executable_OutputDimensions` | `pjrt::PJRT_Executable_OutputDimensions` (`0x0c59a5b0`) | 48 |
| `0x308` | 97 | `PJRT_Buffer_CopyToMemory` | `pjrt::PJRT_Buffer_CopyToMemory` (`0x0c5a01a0`) | 40 |
| `0x310` | 98 | `PJRT_Client_CreateViewOfDeviceBuffer` | `pjrt::PJRT_Client_CreateViewOfDeviceBuffer` (`0x0c596ed0`) | 88 |
| `0x318` | 99 | `PJRT_Executable_Fingerprint` | `pjrt::PJRT_Executable_Fingerprint` (`0x0c599690`) | 40 |
| `0x320` | 100 | `PJRT_Client_TopologyDescription` | `pjrt::PJRT_Client_TopologyDescription` (`0x0c591830`) | 32 |
| `0x328` | 101 | `PJRT_Executable_GetCompiledMemoryStats` | `pjrt::PJRT_Executable_GetCompiledMemoryStats` (`0x0c59e5f0`) | 64 |
| `0x330` | 102 | `PJRT_Memory_Kind_Id` | `pjrt::PJRT_Memory_Kind_Id` (`0x0c597c40`) | 32 |
| `0x338` | 103 | `PJRT_ExecuteContext_Create` | `pjrt::tpu_plugin::PJRT_ExecuteContext_Create` (`0x0b4fb9f0`) | 24 |
| `0x340` | 104 | `PJRT_ExecuteContext_Destroy` | `pjrt::PJRT_ExecuteContext_Destroy` (`0x0c597de0`) | 24 |
| `0x348` | 105 | `PJRT_Buffer_CopyRawToHost` | `pjrt::PJRT_Buffer_CopyRawToHost` (`0x0c59fca0`) | 56 |
| `0x350` | 106 | `PJRT_AsyncHostToDeviceTransferManager_Destroy` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_Destroy` (`0x0c5928d0`) | 24 |
| `0x358` | 107 | `PJRT_AsyncHostToDeviceTransferManager_TransferData` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_TransferData` (`0x0c592940`) | 64 |
| `0x360` | 108 | `PJRT_Client_CreateBuffersForAsyncHostToDevice` | `pjrt::PJRT_Client_CreateBuffersForAsyncHostToDevice` (`0x0c591fd0`) | 48 |
| `0x368` | 109 | `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer` (`0x0c593370`) | 40 |
| `0x370` | 110 | `PJRT_AsyncHostToDeviceTransferManager_Device` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_Device` (`0x0c593420`) | 32 |
| `0x378` | 111 | `PJRT_AsyncHostToDeviceTransferManager_BufferCount` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferCount` (`0x0c593530`) | 32 |
| `0x380` | 112 | `PJRT_AsyncHostToDeviceTransferManager_BufferSize` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferSize` (`0x0c593580`) | 40 |
| `0x388` | 113 | `PJRT_AsyncHostToDeviceTransferManager_SetBufferError` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_SetBufferError` (`0x0c5935e0`) | 48 |
| `0x390` | 114 | `PJRT_AsyncHostToDeviceTransferManager_AddMetadata` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_AddMetadata` (`0x0c5936a0`) | 40 |
| `0x398` | 115 | `PJRT_Client_DmaMap` | `pjrt::PJRT_Client_DmaMap` (`0x0c592810`) | 40 |
| `0x3a0` | 116 | `PJRT_Client_DmaUnmap` | `pjrt::PJRT_Client_DmaUnmap` (`0x0c592870`) | 32 |
| `0x3a8` | 117 | `PJRT_Client_CreateUninitializedBuffer` | `pjrt::PJRT_Client_CreateUninitializedBuffer` (`0x0c595690`) | 72 |
| `0x3b0` | 118 | `PJRT_Client_UpdateGlobalProcessInfo` | `pjrt::PJRT_Client_UpdateGlobalProcessInfo` (`0x0c591c50`) | 32 |
| `0x3b8` | 119 | `PJRT_TopologyDescription_Deserialize` | `pjrt::PJRT_TopologyDescription_Deserialize` (`0x0c5a2930`) | 48 |
| `0x3c0` | 120 | `PJRT_Client_CreateAliasBuffer` | `pjrt::PJRT_Client_CreateAliasBuffer` (`0x0c595d20`) | 72 |
| `0x3c8` | 121 | `PJRT_Client_FulfillAliasBuffer` | `pjrt::PJRT_Client_FulfillAliasBuffer` (`0x0c596150`) | 40 |
| `0x3d0` | 122 | `PJRT_LoadedExecutable_GetDeviceAssignment` | `pjrt::PJRT_LoadedExecutable_GetDeviceAssignment` (`0x0c5a2be0`) | 48 |
| `0x3d8` | 123 | `PJRT_Client_CreateErrorBuffer` | `pjrt::PJRT_Client_CreateErrorBuffer` (`0x0c5958a0`) | 64 |
| `0x3e0` | 124 | `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral` | `pjrt::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral` (`0x0c592b90`) | 64 |
| `0x3e8` | 125 | `PJRT_Buffer_CopyRawToHostFuture` | `pjrt::PJRT_Buffer_CopyRawToHostFuture` (`0x0c59fda0`) | 64 |
| `0x3f0` | 126 | `PJRT_Device_PoisonExecution` | `pjrt::PJRT_Device_PoisonExecution` (`0x0c592f30`) | 48 |
| `0x3f8` | 127 | `PJRT_Device_CreateAsyncTrackingEvent` | `pjrt::PJRT_Device_CreateAsyncTrackingEvent` (`0x0c593280`) | 48 |
| `0x400` | 128 | `PJRT_AsyncTrackingEvent_Destroy` | `pjrt::PJRT_AsyncTrackingEvent_Destroy` (`0x0c593300`) | 24 |
| `0x408` | 129 | `PJRT_Executable_GetCompileOptions` | `pjrt::PJRT_Executable_GetCompileOptions` (`0x0c59e270`) | 48 |
| `0x410` | 130 | `PJRT_Buffer_DonateWithControlDependency` | `pjrt::PJRT_Buffer_DonateWithControlDependency` (`0x0c5a0df0`) | 48 |
| `0x418` | 131 | `PJRT_Event_Create` | `pjrt::PJRT_Event_Create` (`0x0c5a1770`) | 24 |
| `0x420` | 132 | `PJRT_Event_Set` | `pjrt::PJRT_Event_Set` (`0x0c5a1860`) | 40 |
| `0x428` | 133 | `PJRT_Device_GetAttributes` | `pjrt::PJRT_Device_GetAttributes` (`0x0c5a4f80`) | 40 |
| `0x430` | 134 | `PJRT_Client_Load` | `pjrt::PJRT_Client_Load` (`0x0c594980`) | 48 |
| `0x438` | 135 | `PJRT_LoadedExecutable_AddressableDeviceLogicalIds` | `pjrt::PJRT_LoadedExecutable_AddressableDeviceLogicalIds` (`0x0c598640`) | 40 |
| `0x440` | 136 | `PJRT_Buffer_Bitcast` | `pjrt::PJRT_Buffer_Bitcast` (`0x0c594ca0`) | 64 |
| `0x448` | 137 | `PJRT_Error_ForEachPayload` | `pjrt::PJRT_Error_ForEachPayload` (`0x0c591440`) | 40 |
| `0x450` | 138 | `PJRT_TopologyDescription_Fingerprint` | `pjrt::PJRT_TopologyDescription_Fingerprint` (`0x0c5a1da0`) | 32 |
| `0x458` | 139 | `PJRT_Executable_ParameterMemoryKinds` | `pjrt::PJRT_Executable_ParameterMemoryKinds` (`0x0c59aba0`) | 48 |
| `0x460` | 140 | `PJRT_Device_ClearMemoryStats` | `pjrt::PJRT_Device_ClearMemoryStats` (`0x0c592ed0`) | 24 |
| `0x468` | 141 | `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` | `pjrt::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` (`0x0c5a1eb0`) | 64 |
| `0x470` | 142 | `PJRT_TopologyDescription_GetMemorySpaceKindIds` | `pjrt::PJRT_TopologyDescription_GetMemorySpaceKindIds` (`0x0c5a2220`) | 40 |

## 3. PJRT Extension Chain (`PJRT_Extension_Base` Linked List)
`GetTpuPjrtApi()` (`0x0b4fb4e0`) links 17 extension nodes starting at `extension_start` (each beginning with `PJRT_Extension_Base { size_t struct_size; PJRT_Extension_Type type; PJRT_Extension_Base* next; }` = 24 bytes):

| Node | Variable Name in `libtpu.so` | `struct_size` | `PJRT_Extension_Type` (`offset 0x08`) | Function Pointer Slots (`offset 0x18..`) |
| ---: | :--- | ---: | ---: | :--- |
| 1 | `xla_transform_extension` | `56` (`0x38`) | `29` (`0x1d`) | `0x18`: `register_xla_transform`, `0x20`: `wait_for_all_transforms`, `0x28`: `unregister_xla_transform`, `0x30`: `list_xla_transforms` |
| 2 | `host_memory_allocator_extension` | `32` (`0x20`) | `28` (`0x1c`) | `0x18`: `get_host_memory_allocator` |
| 3 | `multi_slice_extension` | `64` (`0x40`) | `26` (`0x1a`) | `0x18`: `config_destroy`, `0x20`: `config_num_slices`, `0x28`: `config_num_devices_per_slice`, `0x30`: `config_slice_id`, `0x38`: `config_serialize` |
| 4 | `collectives_extension` | `96` (`0x60`) | `25` (`0x19`) | `0x18`: `collectives_destroy`, `0x20`: `create_communicator`, `0x28`: `communicator_destroy`, `0x30`: `communicator_all_reduce`, `0x38`: `communicator_reduce_scatter`, `0x40`: `communicator_all_gather`, `0x48`: `communicator_collective_permute`, `0x50`: `communicator_all_to_all`, `0x58`: `communicator_to_string` |
| 5 | `abi_version_extension` | `120` (`0x78`) | `20` (`0x14`) | `0x18`: `client_runtime_abi_version`, `0x20`: `executable_get_abi_version`, `0x28`: `runtime_abi_version_destroy`, `0x30`: `runtime_abi_version_is_compatible_with_runtime`, `0x38`: `runtime_abi_version_is_compatible_with_executable`, `0x40`: `runtime_abi_version_to_proto`, `0x48`: `runtime_abi_version_platform_id`, `0x50`: `executable_abi_version_destroy`, `0x58`: `executable_abi_version_to_proto`, `0x60`: `executable_abi_version_platform_id`, `0x68`: `runtime_abi_version_from_proto`, `0x70`: `executable_abi_version_from_proto` |
| 6 | `shardings_extension` | `40` (`0x28`) | `19` (`0x13`) | `0x18`: `parameter_shardings`, `0x20`: `output_shardings` |
| 7 | `megascale_extension` | `248` (`0xf8`) | `18` (`0x12`) | 28 function pointers (`0x18..0xf0`): `create_client_context`, `delete_client_context`, `create_default_config`, `create_aot_config`, `create_multi_slice_config`, `delete_multi_slice_config`, `multi_slice_config_num_slices`, `multi_slice_config_num_devices_per_slice`, `multi_slice_config_slice_id`, `multi_slice_config_serialized_slice_key_value`, `multi_slice_config_serialize`, `deserialize_multi_slice_config`, `megascale_addresses`, `barrier`, `log_error`, `get_megascale_dcn_topology`, `get_slice_id`, `get_num_slices`, `initialize_slice`, `finalize_slice`, `register_error_handler`, `unregister_error_handler`, `collect_metrics`, `reset_metrics`, `get_transport_type`, `set_transport_type`, `wait_for_barrier`, `abort_collectives` |
| 8 | `tpu_executable_extension` | `96` (`0x60`) | `17` (`0x11`) | `0x18`: `get_target_arguments`, `0x20`: `get_hlo_module_with_config`, `0x28`: `reserved (NULL)`, `0x30`: `get_compiled_memory_stats`, `0x38`: `run_hlo_cost_analysis`, `0x40`: `set_tpu_compilation_env`, `0x48`: `get_tpu_compilation_env_field_as_string`, `0x50`: `is_tpu_predetermined_error`, `0x58`: `is_tpu_cancellation_placeholder_error` |
| 9 | `tpu_topology_extension` | `272` (`0x110`) | `16` (`0x10`) | 31 function pointers (`0x18..0x108`): `subslice`, `is_subslice_topology`, `subslice_device_id_from_full_device_id`, `replace_host_bounds`, `is_enhanced_barrier_enabled`, `has_limited_ici_connectivity`, `is_reachable_over_limited_ici`, `process_count`, `chips_per_process`, `core_count_per_chip`, `chip_count`, `core_count`, `logi_device_count_per_process`, `logi_device_count`, `logi_device_count_per_chip`, `core_count_per_process`, `process_ids`, `logi_device_ids_on_process`, `proc_id_and_idx_on_proc_for_chip`, `proc_id_and_idx_on_proc_for_logi_device`, `process_coord_from_id`, `chip_id_from_coord`, `logi_device_id_from_chip_coord_and_idx`, `chip_coord_and_idx_for_logi_device`, `chips_per_process_bounds`, `chip_bounds`, `process_bounds`, `get_routing_strategy`, `get_slice_config`, `get_slice_configs`, `get_default_platform_config` |
| 10 | `callback_extension` | `40` (`0x28`) | `14` (`0x0e`) | `0x18`: `register_callback`, `0x20`: `invoke_callback` |
| 11 | `phase_compile_extension` | `64` (`0x40`) | `9` (`0x09`) | `0x18`: `phase_compile_get_compiler`, `0x20`: `phase_compile_destroy_compiler`, `0x28`: `phase_compile_run_phase`, `0x30`: `phase_compile_get_phase_names`, `0x38`: `phase_compile_c_buffers_destroy` |
| 12 | `cross_host_transfers_extension` | `56` (`0x38`) | `12` (`0x0c`) | `0x18`: `make_cross_host_receive_buffers`, `0x20`: `copy_to_remote_device`, `0x28`: `cross_host_receive_buffers`, `0x30`: `cross_host_send_buffers` |
| 13 | `executable_metadata_extension` | `40` (`0x28`) | `11` (`0x0b`) | `0x18`: `get_executable_metadata`, `0x20`: `destroy_executable_metadata` |
| 14 | `mem_desc_extension` | `40` (`0x28`) | `7` (`0x07`) | `0x18`: `get_memory_descriptions`, `0x20`: `destroy_memory_descriptions` |
| 15 | `layouts_extension` | `80` (`0x50`) | `6` (`0x06`) | `0x18`: `memory_layout_destroy`, `0x20`: `memory_layout_serialize`, `0x28`: `client_get_default_layout`, `0x30`: `buffer_memory_layout`, `0x38`: `topology_get_default_layout`, `0x40`: `executable_get_output_layouts`, `0x48`: `executable_get_parameter_layouts` |
| 16 | `raw_buffer_extension` | `80` (`0x50`) | `8` (`0x08`) | `0x18`: `create_raw_alias_of_buffer`, `0x20`: `destroy`, `0x28`: `get_host_pointer`, `0x30`: `get_on_device_size_in_bytes`, `0x38`: `get_memory_space`, `0x40`: `copy_raw_host_to_device`, `0x48`: `copy_raw_device_to_host` |
| 17 | `profiler_extension` | `40` (`0x28`) | `1` (`0x01`) | `0x18`: `profiler_api` (`*const PLUGIN_Profiler_Api`), `0x20`: `traceme_context_id` (`int64_t`) |

## 4. `LibtpuSdk_Api` Vtable & Argument Structs (`GetLibtpuSdkApi` @ `0x0d3e07d8`)
`CreateLibtpuSdkApi` (`0x0d3e05e0`) allocates a `0x90` (144-byte) structure containing:
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

## 6. Extension & Profiler `Args` Struct Layouts (113 Functions Across 17 Extensions)
Every extension function in the 17-node `PJRT_Extension_Base` chain takes a `#[repr(C)]` `Args` struct beginning with `size_t struct_size` (`0x00`) and `PJRT_Extension_Base* extension_start` (`0x08`, or `void* priv` for `PLUGIN_Profiler_Api` which is `80` bytes with `struct_size`, `priv`, `error_destroy`, `error_message`, `error_get_code`, `create`, `destroy`, `start`, `stop`, `collect_data`):
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


