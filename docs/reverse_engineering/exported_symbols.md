# `libtpu.so` Exported ELF Symbol Specification (`libtpu-0.0.50`)

## 1. ELF Header & Dynamic Linking Profile
- **Artifact**: `libtpu/libtpu.so` (`630,899,976` bytes, SHA-256 `ef2f7af1f1ca0f27db6a6a3adffdd2b275a3215b121c8f9bc3f6ce842fc7c7`)
- **Companion SDK Artifact**: `libtpu/sdk.so` (`37,821,432` bytes, SHA-256 `6d65c03e2b584eaa3b546b6b64931c4b153ad5c4e2b404f578ea548c0fb6d2a4`)
- **Format**: `ELF64` (`e_type = ET_DYN (3)`, `e_machine = EM_X86_64 (62)`)
- **Calling Convention**: System V AMD64 ABI (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9` for the first 6 integer/pointer arguments; `rax` for return value)
- **Program Headers**: 12
- **Section Headers**: 51 (including `.dynsym`, `.dynstr`, `.symtab` with 950,706 internal symbols, `.strtab`, `protodesc_cold`, `filewrapper_toc`, `.rodata`)
- **`DT_SONAME`**: `None`
- **`DT_NEEDED`**:
  - `libm.so.6`
  - `libpthread.so.0`
  - `libdl.so.2`
  - `librt.so.1`
  - `libc.so.6`
  - `ld-linux-x86-64.so.2`

## 2. Complete Exported Dynamic Function Symbols (`.dynsym`) & C Signatures

`libtpu.so` exports 75 global function symbols (`STT_FUNC`) and 8 linker section boundary markers (`STT_NOTYPE`):

### 2.1 PJRT & SDK Entrypoints
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `GetPjrtApi` | `0x0b4f9b90` | `const PJRT_Api* GetPjrtApi(void)` | Returns pointer to the 1,144-byte `PJRT_Api` vtable |
| `GetLibtpuSdkApi` | `0x0d3e07d8` | `const LibtpuSdk_Api* GetLibtpuSdkApi(void)` | Returns pointer to the 144-byte `LibtpuSdk_Api` vtable |

### 2.2 Runtime Initialization & TensorFlow / StreamExecutor Hooks
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TfTpu_Initialize` | `0x0b54a380` | `void TfTpu_Initialize(bool init_library, int num_args, const char* const* args)` | Initializes the TPU driver and applies CLI init flags |
| `TF_InitKernel` | `0x0b54a390` | `void TF_InitKernel(void)` | TensorFlow kernel initialization hook |
| `TFNPD_InitPlugin` | `0x0b576f90` | `void TFNPD_InitPlugin(TFNPD_PluginParams* params, TpuStatus* status)` | TensorFlow NextPluggableDevice plugin entrypoint |
| `SetGlobalTPUArrayOp_DoWork` | `0x0b64db50` | `void SetGlobalTPUArrayOp_DoWork(size_t topology_len, const char* topology_proto, TpuStatus* status)` | Configures global TPU topology array state |
| `TfTpu_GetTpuPartitionedCallParams` | `0x0b64dc80` | `void TfTpu_GetTpuPartitionedCallParams(TpuPartitionedCall_Params* params)` | Queries TPU partitioned call runtime parameters |
| `TpuConfigurationApi_FreeCharArray` | `0x0b64dc50` | `void TpuConfigurationApi_FreeCharArray(char* ptr)` | Frees heap-allocated `char*` arrays returned by configuration APIs |
| `TpuTransferManager_GetInfeedLayout` | `0x0c3da320` | `void TpuTransferManager_GetInfeedLayout(const XLA_Shape* host_shape, XLA_Shape* infeed_shape)` | Computes on-device infeed layout for host tensor shapes |
| `SparseCore_GetMaxIdsAndUniques` | `0x0c4bc420` | `void SparseCore_GetMaxIdsAndUniques(const char* table_name, int64_t num_samples_per_sparse_core, int64_t feature_width, int64_t* max_ids_per_partition, int64_t* max_unique_ids_per_partition, TpuStatus* status)` | Calculates SparseCore max IDs and unique IDs per partition |

### 2.3 Ordinal Selector C API (`TfTpuOrdinalSelector_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TfTpuOrdinalSelector_Create` | `0x0b64b270` | `void TfTpuOrdinalSelector_Create(TfTpuOrdinalSelector** selector_out, int32_t num_cores)` | Allocates a round-robin / load-balanced TPU core ordinal selector |
| `TfTpuOrdinalSelector_Destroy` | `0x0b64b3e0` | `void TfTpuOrdinalSelector_Destroy(TfTpuOrdinalSelector* selector)` | Frees an ordinal selector instance |
| `TfTpuOrdinalSelector_GetOrdinal` | `0x0b64b4a0` | `int32_t TfTpuOrdinalSelector_GetOrdinal(TfTpuOrdinalSelector* selector, uint64_t* seed, int64_t* req_id)` | Selects the next device ordinal for a given seed/request |
| `TfTpuOrdinalSelector_DequeueFromCoreSelector` | `0x0b64b4c0` | `void TfTpuOrdinalSelector_DequeueFromCoreSelector(TfTpuOrdinalSelector* selector, int32_t device_ordinal, int64_t req_id)` | Dequeues a reserved ordinal from the core selector |

### 2.4 Status C API (`TpuStatus_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuStatus_New` | `0x0c3da8d0` | `TpuStatus* TpuStatus_New(void)` | Allocates an OK `TpuStatus` |
| `TpuStatus_Create` | `0x0c3da8e0` | `TpuStatus* TpuStatus_Create(int32_t code, const char* msg)` | Allocates a `TpuStatus` with explicit code and message |
| `TpuStatus_Set` | `0x0c3da930` | `void TpuStatus_Set(TpuStatus* status, int32_t code, const char* msg, int32_t len)` | Mutates a `TpuStatus` with error code and message |
| `TpuStatus_Free` | `0x0c3da918` | `void TpuStatus_Free(TpuStatus* status)` | Destroys a `TpuStatus` instance |
| `TpuStatus_Message` | `0x0c3da920` | `const char* TpuStatus_Message(const TpuStatus* status)` | Returns `const char*` error message |
| `TpuStatus_Code` | `0x0c3daa18` | `int32_t TpuStatus_Code(const TpuStatus* status)` | Returns integer `absl::StatusCode` (`0` = OK) |
| `TpuStatus_Ok` | `0x0c3daa20` | `bool TpuStatus_Ok(const TpuStatus* status)` | Returns `bool` (`true` if status code == 0) |

### 2.5 Topology, HostLocation & CoreLocation C API (`TpuTopology_*`, `TpuHostLocation_*`, `TpuCoreLocation_*`, `TpuUtil_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuUtil_GetTopologyPtr` | `0x0c3da7e0` | `const TpuTopology* TpuUtil_GetTopologyPtr(void)` | Returns singleton `const TpuTopology*` |
| `TpuUtil_GetXlaPadSizeFromTpuTopology` | `0x0c3da8b0` | `int32_t TpuUtil_GetXlaPadSizeFromTpuTopology(void)` | Returns XLA minor-dimension alignment pad size |
| `TpuTopology_LogicalDevicesPerHost` | `0x0bbbec40` | `int32_t TpuTopology_LogicalDevicesPerHost(const TpuTopology* topo, int32_t core_type)` | Number of logical devices per host |
| `TpuTopology_LogicalDevicesPerChip` | `0x0bbbec60` | `int32_t TpuTopology_LogicalDevicesPerChip(const TpuTopology* topo, int32_t core_type)` | Number of logical devices per chip |
| `TpuTopology_HostCount` | `0x0bbbec78` | `int32_t TpuTopology_HostCount(const TpuTopology* topo)` | Total number of hosts in the TPU slice |
| `TpuTopology_ChipsPerHost` | `0x0bbbec7c` | `int32_t TpuTopology_ChipsPerHost(const TpuTopology* topo)` | Number of physical TPU chips per host |
| `TpuTopology_ChipBounds_X` | `0x0bbbec80` | `int32_t TpuTopology_ChipBounds_X(const TpuTopology* topo)` | X-dimension bound of the chip mesh |
| `TpuTopology_ChipBounds_Y` | `0x0bbbec84` | `int32_t TpuTopology_ChipBounds_Y(const TpuTopology* topo)` | Y-dimension bound of the chip mesh |
| `TpuTopology_ChipBounds_Z` | `0x0bbbec88` | `int32_t TpuTopology_ChipBounds_Z(const TpuTopology* topo)` | Z-dimension bound of the chip mesh |
| `TpuTopology_HasChip` | `0x0bbbec90` | `bool TpuTopology_HasChip(const TpuTopology* topo, int32_t x, int32_t y, int32_t z)` | Checks whether chip `(x, y, z)` exists in the topology |
| `TpuTopology_CoreForId` | `0x0bbbecc8` | `const TpuCoreLocation* TpuTopology_CoreForId(const TpuTopology* topo, int32_t core_type, int32_t id)` | Looks up `TpuCoreLocation` by global core ID |
| `TpuTopology_Core` | `0x0bbbecd0` | `const TpuCoreLocation* TpuTopology_Core(const TpuTopology* topo, int32_t core_type, int32_t x, int32_t y, int32_t z, int32_t index)` | Looks up `TpuCoreLocation` by `(core_type, x, y, z, index)` |
| `TpuTopology_NumCores` | `0x0bbbed10` | `int32_t TpuTopology_NumCores(const TpuTopology* topo, int32_t core_type)` | Returns total number of cores of a given `TpuCoreTypeEnum` |
| `TpuTopology_Cores` | `0x0bbbed20` | `void TpuTopology_Cores(const TpuTopology* topo, int32_t core_type, const TpuCoreLocation** cores_out)` | Copies pointers to all `TpuCoreLocation`s into caller array |
| `TpuTopology_IdForHost` | `0x0bbbedf0` | `int32_t TpuTopology_IdForHost(const TpuTopology* topo, int32_t x, int32_t y, int32_t z)` | Returns host ID for host coordinates `(x, y, z)` |
| `TpuTopology_Version` | `0x0bbbee30` | `int32_t TpuTopology_Version(const TpuTopology* topo)` | Returns `TpuVersionEnum` |
| `TpuTopology_AvailableCoreCount` | `0x0c3da470` | `int32_t TpuTopology_AvailableCoreCount(const TpuMeshState* mesh_state, int32_t core_type)` | Queries available core count for mesh state |
| `TpuTopology_AvailableCoresPerChip` | `0x0c3da5b0` | `int32_t TpuTopology_AvailableCoresPerChip(int32_t core_type)` | Queries available cores per chip for `TpuCoreTypeEnum` |
| `TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice` | `0x0c3da660` | `int32_t TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice(int32_t core_type)` | Queries SparseCore count per logical device |
| `TpuCoreLocation_ChipCoordinates` | `0x0bbbee50` | `void TpuCoreLocation_ChipCoordinates(const TpuCoreLocation* loc, int32_t* x, int32_t* y, int32_t* z)` | Returns `(x, y, z)` chip coordinates of a core |
| `TpuCoreLocation_HostCoordinates` | `0x0bbbee70` | `void TpuCoreLocation_HostCoordinates(const TpuCoreLocation* loc, int32_t* x, int32_t* y, int32_t* z)` | Returns `(x, y, z)` host coordinates of a core |
| `TpuCoreLocation_Index` | `0x0bbbeeb0` | `int32_t TpuCoreLocation_Index(const TpuCoreLocation* loc)` | Returns core index on chip |
| `TpuCoreLocation_Id` | `0x0bbbeeb8` | `int32_t TpuCoreLocation_Id(const TpuCoreLocation* loc)` | Returns global core ID |
| `TpuHostLocation_Id` | `0x0bbbeec0` | `int32_t TpuHostLocation_Id(const TpuCoreLocation* host)` | Returns host ID |
| `TpuHostLocation_NumCores` | `0x0bbbeed0` | `int32_t TpuHostLocation_NumCores(const TpuCoreLocation* host, int32_t core_type)` | Returns number of cores on host |
| `TpuHostLocation_Cores` | `0x0bbbeee0` | `void TpuHostLocation_Cores(const TpuCoreLocation* host, int32_t core_type, const TpuCoreLocation** cores_out)` | Copies core pointers on host into caller array |

### 2.6 MeshState & Profiler C API (`TpuMeshState_*`, `TpuProfiler_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuMeshState_Create` | `0x1b62fb00` | `TpuMeshState* TpuMeshState_Create(void)` | Allocates a `TpuMeshState` handle |
| `TpuMeshState_Free` | `0x0c4bc3c0` | `void TpuMeshState_Free(TpuMeshState* state)` | Frees a `TpuMeshState` handle |
| `TpuMeshState_MeshCommonState` | `0x0c4bc410` | `const void* TpuMeshState_MeshCommonState(const TpuMeshState* state)` | Returns internal mesh common state pointer |
| `TpuProfiler_Create` | `0x0bbbefb0` | `void TpuProfiler_Create(TpuProfiler** profiler_out, TpuStatus* status)` | Creates a `TpuProfiler` session |
| `TpuProfiler_Destroy` | `0x0bbbf1d0` | `void TpuProfiler_Destroy(TpuProfiler* profiler)` | Destroys a `TpuProfiler` session |
| `TpuProfiler_Start` | `0x0bbbf290` | `void TpuProfiler_Start(TpuProfiler* profiler, TpuStatus* status)` | Starts hardware & runtime trace collection |
| `TpuProfiler_Stop` | `0x0bbbf460` | `void TpuProfiler_Stop(TpuProfiler* profiler, TpuStatus* status)` | Stops hardware & runtime trace collection |
| `TpuProfiler_CollectData` | `0x0bbbf610` | `void TpuProfiler_CollectData(TpuProfiler* profiler, TpuStatus* status, uint8_t* buffer, size_t* size_in_bytes)` | Serializes collected `XSpace` protobuf into caller buffer |

### 2.7 TPU Embedding Engine C API (`TpuEmbeddingEngine_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuEmbeddingEngineState_Create` | `0x1b62f780` | `TpuEmbeddingEngineState* TpuEmbeddingEngineState_Create(void)` | Creates embedding engine state |
| `TpuEmbeddingEngineState_Free` | `0x0c4959c0` | `void TpuEmbeddingEngineState_Free(TpuEmbeddingEngineState* state)` | Frees embedding engine state |
| `TpuEmbeddingEngineState_GetState` | `0x0c495a00` | `const void* TpuEmbeddingEngineState_GetState(const TpuEmbeddingEngineState* state)` | Returns internal state pointer |
| `TpuEmbeddingEngine_ExecutePartitioner` | `0x0c3dbc90` | `void TpuEmbeddingEngine_ExecutePartitioner(TpuEmbeddingEngine_ExecutePartitioner_Params* params)` | Partitions embedding configuration across TPU shards |
| `TpuEmbeddingEngine_ConfigureMemory` | `0x0c3dbe70` | `void TpuEmbeddingEngine_ConfigureMemory(TpuEmbeddingEngine_ConfigureMemory_Params* params)` | Allocates HBM for embedding tables |
| `TpuEmbeddingEngine_CollateMemory` | `0x0c3dc090` | `void TpuEmbeddingEngine_CollateMemory(TpuEmbeddingEngine_CollateMemory_Params* params)` | Collates HBM addresses across hosts |
| `TpuEmbeddingEngine_ConfigureHost` | `0x0c3dc410` | `void TpuEmbeddingEngine_ConfigureHost(TpuEmbeddingEngine_ConfigureHost_Params* params)` | Configures host-side embedding subsystem |
| `TpuEmbeddingEngine_ConnectHosts` | `0x0c3dc790` | `void TpuEmbeddingEngine_ConnectHosts(TpuEmbeddingEngine_ConnectHosts_Params* params)` | Establishes inter-host ICI/gRPC embedding channels |
| `TpuEmbeddingEngine_Finalize` | `0x0c3dca80` | `void TpuEmbeddingEngine_Finalize(TpuEmbeddingEngine_Finalize_Params* params)` | Finalizes embedding engine initialization |
| `TpuEmbeddingEngine_IsInitialized` | `0x0c3dccb0` | `void TpuEmbeddingEngine_IsInitialized(TpuEmbeddingEngine_IsInitialized_Params* params)` | Checks whether embedding engine is initialized |
| `TpuEmbeddingEngine_WriteParameters` | `0x0c3dce40` | `void TpuEmbeddingEngine_WriteParameters(TpuEmbeddingEngine_WriteParameters_Params* params)` | Writes embedding table weights to HBM |
| `TpuEmbeddingEngine_ReadParameters` | `0x0c3dd250` | `void TpuEmbeddingEngine_ReadParameters(TpuEmbeddingEngine_ReadParameters_Params* params)` | Reads embedding table weights from HBM |
| `TpuEmbeddingTensorBatchFixedState_Create` | `0x0c3dd780` | `TpuEmbeddingEngineState* TpuEmbeddingTensorBatchFixedState_Create(TpuStatus* status)` | Creates fixed state for tensor batch enqueue |
| `TpuEmbeddingTensorBatchFixedState_Destroy` | `0x0c3ddc30` | `void TpuEmbeddingTensorBatchFixedState_Destroy(TpuEmbeddingEngineState* state)` | Destroys fixed state for tensor batch enqueue |
| `TpuEmbeddingEngine_EnqueueTensorBatch` | `0x0c3df6e0` | `void TpuEmbeddingEngine_EnqueueTensorBatch(TpuEmbeddingEngine_EnqueueTensorBatch_Params* params)` | Enqueues sparse embedding lookup batch |
| `TpuEmbeddingEngine_RecvActivationsComputation` | `0x0c496070` | `void TpuEmbeddingEngine_RecvActivationsComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for receiving embedding activations |
| `TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation` | `0x0c4969a0` | `void TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for receiving deduplication data |
| `TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation` | `0x0c4972e0` | `void TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for sending embedding gradients |
| `TpuEmbeddingEngine_DedupDataSizeComputation` | `0x0c497cd0` | `void TpuEmbeddingEngine_DedupDataSizeComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for deduplication data byte size |
| `TpuEmbeddingEngine_DedupDataTupleMaskComputation` | `0x0c498120` | `void TpuEmbeddingEngine_DedupDataTupleMaskComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for deduplication tuple mask |

### 2.8 TPU Driver C API (`TpuDriver_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuDriver_Open` | `0x0c3d9a10` | `TpuDriver* TpuDriver_Open(const char* worker)` | Opens a `TpuDriver` instance with populated `TpuDriverFn*` vtable |
| `TpuDriver_Close` | `0x0c3d9ae0` | `void TpuDriver_Close(TpuDriver* driver)` | Closes and frees a `TpuDriver` instance |

## 3. Disassembled `#[repr(C)]` Parameter Structure Layouts

### 3.1 `TFNPD_PluginParams` (`0x0b576f90`)
- `0x00`: `size_t struct_size` (`40` bytes)
- `0x08`: `void* priv`
- `0x10`: `const char* device_type` (written as `"TPU"`)
- `0x18`: `const char* compilation_device_name` (written as `"TPU_REPLICATED_CORE"`)
- `0x20`: `int32_t priority` (written as `0x37` = `55`)
- `0x24`: `bool is_pluggable_device` (`true`), `bool use_pjrt_on_demand_compile` (`true`), `uint16_t pad`

### 3.2 `TpuPartitionedCall_Params` (`0x0b64dc80`)
- `0x00`: `bool input_shape_opt`
- `0x01`: `bool group_tensors_for_packing`
- `0x02`: `uint16_t pad0`
- `0x04`: `int32_t minimum_input_tensors_packing`
- `0x08`: `int32_t minimum_output_tensors_packing`
- `0x0c`: `bool enable_auto_xla_input_sharding`
- `0x0d`: `uint8_t pad1[3]`
- `0x10`: `int32_t auto_xla_input_sharding_dim`
- `0x14`: `bool scoped_allocator_for_tpu_anned_call`

### 3.3 `XLA_Shape` (`0x0c3da410`)
- `0x00`: `int32_t element_type`
- `0x04`: `int32_t rank`
- `0x08`: `const int64_t* dimensions`
- `0x10`: `int64_t minor_to_major[8]`
- `0x50`: `int64_t tile_dims[2]`

### 3.4 `TpuSerializedProto` & `TpuEmbeddingEngine_*_Params` (`0x0c3dbd80..0x0c498120`)
All `TpuEmbeddingEngine_*` functions accept a pointer to a `#[repr(C)]` parameter struct starting with `size_t struct_size` (`0x00`) and `void* priv` (`0x08`):
- `TpuEmbeddingEngine_ExecutePartitioner_Params` (`56` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `const char* tpu_embedding_config_bytes`
  - `0x18`: `size_t tpu_embedding_config_size`
  - `0x20`: `size_t* common_config_size`
  - `0x28`: `char** common_config`
  - `0x30`: `TpuStatus* status`
- `TpuEmbeddingEngine_ConfigureMemory_Params` (`64` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `int32_t num_inputs` (with 4-byte padding)
  - `0x18`: `const char* common_config_bytes`
  - `0x20`: `size_t common_config_size`
  - `0x28`: `size_t* memory_config_size`
  - `0x30`: `char** memory_config`
  - `0x38`: `TpuStatus* status`
- `TpuEmbeddingEngine_CollateMemory_Params` (`56` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `size_t memory_configs_Count`
  - `0x18`: `const TpuSerializedProto* memory_configs`
  - `0x20`: `size_t* merged_memory_config_size`
  - `0x28`: `char** merged_memory_config`
  - `0x30`: `TpuStatus* status`
- `TpuEmbeddingEngine_ConfigureHost_Params` (`96` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `int32_t num_inputs` (with 4-byte padding)
  - `0x18`: `const char* common_config_bytes`
  - `0x20`: `size_t common_config_size`
  - `0x28`: `const char* memory_config_bytes`
  - `0x30`: `size_t memory_config_size`
  - `0x38`: `const char* tpu_embedding_config_bytes`
  - `0x40`: `size_t tpu_embedding_config_size`
  - `0x48`: `size_t* network_config_size`
  - `0x50`: `char** network_config`
  - `0x58`: `TpuStatus* status`
- `TpuEmbeddingEngine_ConnectHosts_Params` (`40` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `size_t network_configs_count`
  - `0x18`: `const TpuSerializedProto* network_configs`
  - `0x20`: `TpuStatus* status`
- `TpuEmbeddingEngine_Finalize_Params` (`64` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `const void* mesh_common_state`
  - `0x18`: `const char* common_config_bytes`
  - `0x20`: `size_t common_config_size`
  - `0x28`: `const char* memory_config_bytes`
  - `0x30`: `size_t memory_config_size`
  - `0x38`: `TpuStatus* status`
- `TpuEmbeddingEngine_IsInitialized_Params` (`48` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `const char* config_string_bytes`
  - `0x18`: `size_t config_string_size`
  - `0x20`: `bool* is_tpu_embedding_initialized`
  - `0x28`: `TpuStatus* status`
- `TpuEmbeddingEngine_WriteParameters_Params` / `ReadParameters_Params` (`72` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `const char* table_name`
  - `0x18`: `size_t table_name_len`
  - `0x20`: `const float* parameters_in` (or `float* parameters_out`)
  - `0x28`: `size_t num_parameters`
  - `0x30`: `size_t* num_parameters_written_or_read`
  - `0x38`: `TpuStatus* status`
  - `0x40`: `TpuEmbeddingEngineState* state`
- `TpuEmbeddingEngine_EnqueueTensorBatch_Params` (`56` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `int32_t mode_override`
  - `0x14`: `int32_t local_device_ordinal`
  - `0x18`: `TpuEmbeddingEngineState* fixed_state`
  - `0x20`: `const int32_t* sample_indices`
  - `0x28`: `size_t num_samples`
  - `0x30`: `TpuStatus* status`
- `TpuEmbeddingEngine_Computation_Params` (`88` bytes):
  - `0x00`: `size_t struct_size`
  - `0x08`: `void* priv`
  - `0x10`: `const char* tpu_embedding_config_bytes`
  - `0x18`: `size_t tpu_embedding_config_size`
  - `0x20`: `const char* embedding_partitions_bytes`
  - `0x28`: `size_t embedding_partitions_size`
  - `0x30`: `const char* hbm_buffers_config_bytes`
  - `0x38`: `size_t hbm_buffers_config_size`
  - `0x40`: `const char* tpu_topology_bytes`
  - `0x48`: `size_t tpu_topology_size`
  - `0x50`: `TpuSerializedProto* xla_computation_out` (`{ char* bytes; size_t size; }`)
  - `0x58`: `TpuStatus* status`

