# `libtpu.so` Exported ELF Symbol Specification (`libtpu-0.0.53.1`)

## 1. ELF Header & Dynamic Linking Profile
- **Artifact**: `libtpu/libtpu.so` (`451,904,656` bytes, SHA-256 `9b098d865d7ed58592646df3d7ebe988dadfb3f444011ddae99b230bbfcb3dc8`)
- **Companion SDK Artifacts**:
  - Standard GIL (`cp311`–`cp315`): `libtpu/sdk.so` (`37,925,248` bytes, SHA-256 `630c772352b470e1f32b7459ac3d60790f8d4d5000747390ebdeb743c1f33186`)
  - Free-threaded (`cp314t` / `cp315t`): `libtpu/sdk.so` (`37,930,208` bytes, SHA-256 `7fb29a9b394037ac61d984948afa55147c49248cf75cd6e518c445e4a80284f6`)
- **Format**: `ELF64` (`e_type = ET_DYN (3)`, `e_machine = EM_X86_64 (62)`)
- **Calling Convention**: System V AMD64 ABI (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9` for the first 6 integer/pointer arguments; `rax` for return value)
- **Program Headers**: 12
- **Section Headers**: 51 (including `.dynsym`, `.dynstr`, `.symtab` with 752,460 internal symbols, `.strtab`, `protodesc_cold`, `filewrapper_toc`, `.rodata`)
- **`DT_SONAME`**: `None`
- **`DT_NEEDED`**:
  - `libm.so.6`
  - `libpthread.so.0`
  - `libdl.so.2`
  - `librt.so.1`
  - `libc.so.6`
  - `ld-linux-x86-64.so.2`

## 2. Complete Exported Dynamic Function Symbols (`.dynsym`) & C Signatures

`libtpu.so` exports 75 global function symbols (`STT_FUNC`) and 8 linker section boundary markers (`STT_NOTYPE`) (with `libtpu-rs` also retaining the 3 legacy `TpuDriver_*` symbols for backward compatibility):

### 2.1 PJRT & SDK Entrypoints
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `GetPjrtApi` | `0x06077048` | `const PJRT_Api* GetPjrtApi(void)` | Returns pointer to the 1,144-byte `PJRT_Api` vtable |
| `GetLibtpuSdkApi` | `0x07fccb78` | `const LibtpuSdk_Api* GetLibtpuSdkApi(void)` | Returns pointer to the 144-byte `LibtpuSdk_Api` vtable |

### 2.2 Runtime Initialization & TensorFlow / StreamExecutor Hooks
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TfTpu_Initialize` | `0x061f63d0` | `void TfTpu_Initialize(bool init_library, int num_args, const char* const* args)` | Initializes the TPU driver and applies CLI init flags |
| `TF_InitKernel` | `0x061f63e0` | `void TF_InitKernel(void)` | TensorFlow kernel initialization hook |
| `TFNPD_InitPlugin` | `0x06222e90` | `void TFNPD_InitPlugin(TFNPD_PluginParams* params, TpuStatus* status)` | TensorFlow NextPluggableDevice plugin entrypoint |
| `SetGlobalTPUArrayOp_DoWork` | `0x0622a1d0` | `void SetGlobalTPUArrayOp_DoWork(size_t topology_len, const char* topology_proto, TpuStatus* status)` | Configures global TPU topology array state |
| `TfTpu_GetTpuPartitionedCallParams` | `0x0622a2e0` | `void TfTpu_GetTpuPartitionedCallParams(TpuPartitionedCall_Params* params)` | Queries TPU partitioned call runtime parameters |
| `TpuConfigurationApi_FreeCharArray` | `0x0622a2d0` | `void TpuConfigurationApi_FreeCharArray(char* ptr)` | Frees heap-allocated `char*` arrays returned by configuration APIs |
| `TpuTransferManager_GetInfeedLayout` | `0x06da2eb0` | `void TpuTransferManager_GetInfeedLayout(const XLA_Shape* host_shape, XLA_Shape* infeed_shape)` | Computes on-device infeed layout for host tensor shapes |
| `SparseCore_GetMaxIdsAndUniques` | `0x06e86a30` | `void SparseCore_GetMaxIdsAndUniques(const char* table_name, int64_t num_samples_per_sparse_core, int64_t feature_width, int64_t* max_ids_per_partition, int64_t* max_unique_ids_per_partition, TpuStatus* status)` | Calculates SparseCore max IDs and unique IDs per partition |

### 2.3 Ordinal Selector C API (`TfTpuOrdinalSelector_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TfTpuOrdinalSelector_Create` | `0x062277d0` | `void TfTpuOrdinalSelector_Create(TfTpuOrdinalSelector** selector_out, int32_t num_cores)` | Allocates a round-robin / load-balanced TPU core ordinal selector |
| `TfTpuOrdinalSelector_Destroy` | `0x06227940` | `void TfTpuOrdinalSelector_Destroy(TfTpuOrdinalSelector* selector)` | Frees an ordinal selector instance |
| `TfTpuOrdinalSelector_GetOrdinal` | `0x06227a00` | `int32_t TfTpuOrdinalSelector_GetOrdinal(TfTpuOrdinalSelector* selector, uint64_t* seed, int64_t* req_id)` | Selects the next device ordinal for a given seed/request |
| `TfTpuOrdinalSelector_DequeueFromCoreSelector` | `0x06227a20` | `void TfTpuOrdinalSelector_DequeueFromCoreSelector(TfTpuOrdinalSelector* selector, int32_t device_ordinal, int64_t req_id)` | Dequeues a reserved ordinal from the core selector |

### 2.4 Status C API (`TpuStatus_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuStatus_New` | `0x06da3440` | `TpuStatus* TpuStatus_New(void)` | Allocates an OK `TpuStatus` |
| `TpuStatus_Create` | `0x06da3450` | `TpuStatus* TpuStatus_Create(int32_t code, const char* msg)` | Allocates a `TpuStatus` with explicit code and message |
| `TpuStatus_Set` | `0x06da34a0` | `void TpuStatus_Set(TpuStatus* status, int32_t code, const char* msg, int32_t len)` | Mutates a `TpuStatus` with error code and message |
| `TpuStatus_Free` | `0x06da3488` | `void TpuStatus_Free(TpuStatus* status)` | Destroys a `TpuStatus` instance |
| `TpuStatus_Message` | `0x06da3490` | `const char* TpuStatus_Message(const TpuStatus* status)` | Returns `const char*` error message |
| `TpuStatus_Code` | `0x06da3588` | `int32_t TpuStatus_Code(const TpuStatus* status)` | Returns integer `absl::StatusCode` (`0` = OK) |
| `TpuStatus_Ok` | `0x06da3590` | `bool TpuStatus_Ok(const TpuStatus* status)` | Returns `bool` (`true` if status code == 0) |

### 2.5 Topology, HostLocation & CoreLocation C API (`TpuTopology_*`, `TpuHostLocation_*`, `TpuCoreLocation_*`, `TpuUtil_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuUtil_GetTopologyPtr` | `0x06da3340` | `const TpuTopology* TpuUtil_GetTopologyPtr(void)` | Returns singleton `const TpuTopology*` |
| `TpuUtil_GetXlaPadSizeFromTpuTopology` | `0x06da3420` | `int32_t TpuUtil_GetXlaPadSizeFromTpuTopology(void)` | Returns XLA minor-dimension alignment pad size |
| `TpuTopology_LogicalDevicesPerHost` | `0x0679f2e0` | `int32_t TpuTopology_LogicalDevicesPerHost(const TpuTopology* topo, int32_t core_type)` | Number of logical devices per host |
| `TpuTopology_LogicalDevicesPerChip` | `0x0679f300` | `int32_t TpuTopology_LogicalDevicesPerChip(const TpuTopology* topo, int32_t core_type)` | Number of logical devices per chip |
| `TpuTopology_HostCount` | `0x0679f318` | `int32_t TpuTopology_HostCount(const TpuTopology* topo)` | Total number of hosts in the TPU slice |
| `TpuTopology_ChipsPerHost` | `0x0679f31c` | `int32_t TpuTopology_ChipsPerHost(const TpuTopology* topo)` | Number of physical TPU chips per host |
| `TpuTopology_ChipBounds_X` | `0x0679f320` | `int32_t TpuTopology_ChipBounds_X(const TpuTopology* topo)` | X-dimension bound of the chip mesh |
| `TpuTopology_ChipBounds_Y` | `0x0679f324` | `int32_t TpuTopology_ChipBounds_Y(const TpuTopology* topo)` | Y-dimension bound of the chip mesh |
| `TpuTopology_ChipBounds_Z` | `0x0679f328` | `int32_t TpuTopology_ChipBounds_Z(const TpuTopology* topo)` | Z-dimension bound of the chip mesh |
| `TpuTopology_HasChip` | `0x0679f330` | `bool TpuTopology_HasChip(const TpuTopology* topo, int32_t x, int32_t y, int32_t z)` | Checks whether chip `(x, y, z)` exists in the topology |
| `TpuTopology_CoreForId` | `0x0679f368` | `const TpuCoreLocation* TpuTopology_CoreForId(const TpuTopology* topo, int32_t core_type, int32_t id)` | Looks up `TpuCoreLocation` by global core ID |
| `TpuTopology_Core` | `0x0679f370` | `const TpuCoreLocation* TpuTopology_Core(const TpuTopology* topo, int32_t core_type, int32_t x, int32_t y, int32_t z, int32_t index)` | Looks up `TpuCoreLocation` by `(core_type, x, y, z, index)` |
| `TpuTopology_NumCores` | `0x0679f3b0` | `int32_t TpuTopology_NumCores(const TpuTopology* topo, int32_t core_type)` | Returns total number of cores of a given `TpuCoreTypeEnum` |
| `TpuTopology_Cores` | `0x0679f3c0` | `void TpuTopology_Cores(const TpuTopology* topo, int32_t core_type, const TpuCoreLocation** cores_out)` | Copies pointers to all `TpuCoreLocation`s into caller array |
| `TpuTopology_IdForHost` | `0x0679f490` | `int32_t TpuTopology_IdForHost(const TpuTopology* topo, int32_t x, int32_t y, int32_t z)` | Returns host ID for host coordinates `(x, y, z)` |
| `TpuTopology_Version` | `0x0679f4d0` | `int32_t TpuTopology_Version(const TpuTopology* topo)` | Returns `TpuVersionEnum` |
| `TpuTopology_AvailableCoreCount` | `0x06da3000` | `int32_t TpuTopology_AvailableCoreCount(const TpuMeshState* mesh_state, int32_t core_type)` | Queries available core count for mesh state |
| `TpuTopology_AvailableCoresPerChip` | `0x06da3130` | `int32_t TpuTopology_AvailableCoresPerChip(int32_t core_type)` | Queries available cores per chip for `TpuCoreTypeEnum` |
| `TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice` | `0x06da31d0` | `int32_t TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice(int32_t core_type)` | Queries SparseCore count per logical device |
| `TpuCoreLocation_ChipCoordinates` | `0x0679f4f0` | `void TpuCoreLocation_ChipCoordinates(const TpuCoreLocation* loc, int32_t* x, int32_t* y, int32_t* z)` | Returns `(x, y, z)` chip coordinates of a core |
| `TpuCoreLocation_HostCoordinates` | `0x0679f510` | `void TpuCoreLocation_HostCoordinates(const TpuCoreLocation* loc, int32_t* x, int32_t* y, int32_t* z)` | Returns `(x, y, z)` host coordinates of a core |
| `TpuCoreLocation_Index` | `0x0679f550` | `int32_t TpuCoreLocation_Index(const TpuCoreLocation* loc)` | Returns core index on chip |
| `TpuCoreLocation_Id` | `0x0679f558` | `int32_t TpuCoreLocation_Id(const TpuCoreLocation* loc)` | Returns global core ID |
| `TpuHostLocation_Id` | `0x0679f560` | `int32_t TpuHostLocation_Id(const TpuCoreLocation* host)` | Returns host ID |
| `TpuHostLocation_NumCores` | `0x0679f570` | `int32_t TpuHostLocation_NumCores(const TpuCoreLocation* host, int32_t core_type)` | Returns number of cores on host |
| `TpuHostLocation_Cores` | `0x0679f580` | `void TpuHostLocation_Cores(const TpuCoreLocation* host, int32_t core_type, const TpuCoreLocation** cores_out)` | Copies core pointers on host into caller array |

### 2.6 MeshState & Profiler C API (`TpuMeshState_*`, `TpuProfiler_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuMeshState_Create` | `0x131da720` | `TpuMeshState* TpuMeshState_Create(void)` | Allocates a `TpuMeshState` handle |
| `TpuMeshState_Free` | `0x06e869d0` | `void TpuMeshState_Free(TpuMeshState* state)` | Frees a `TpuMeshState` handle |
| `TpuMeshState_MeshCommonState` | `0x06e86a20` | `const void* TpuMeshState_MeshCommonState(const TpuMeshState* state)` | Returns internal mesh common state pointer |
| `TpuProfiler_Create` | `0x0679f650` | `void TpuProfiler_Create(TpuProfiler** profiler_out, TpuStatus* status)` | Creates a `TpuProfiler` session |
| `TpuProfiler_Destroy` | `0x0679f870` | `void TpuProfiler_Destroy(TpuProfiler* profiler)` | Destroys a `TpuProfiler` session |
| `TpuProfiler_Start` | `0x0679f930` | `void TpuProfiler_Start(TpuProfiler* profiler, TpuStatus* status)` | Starts hardware & runtime trace collection |
| `TpuProfiler_Stop` | `0x0679faf0` | `void TpuProfiler_Stop(TpuProfiler* profiler, TpuStatus* status)` | Stops hardware & runtime trace collection |
| `TpuProfiler_CollectData` | `0x0679fc90` | `void TpuProfiler_CollectData(TpuProfiler* profiler, TpuStatus* status, uint8_t* buffer, size_t* size_in_bytes)` | Serializes collected `XSpace` protobuf into caller buffer |

### 2.7 TPU Embedding Engine C API (`TpuEmbeddingEngine_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuEmbeddingEngineState_Create` | `0x131da3a0` | `TpuEmbeddingEngineState* TpuEmbeddingEngineState_Create(void)` | Creates embedding engine state |
| `TpuEmbeddingEngineState_Free` | `0x06e622f0` | `void TpuEmbeddingEngineState_Free(TpuEmbeddingEngineState* state)` | Frees embedding engine state |
| `TpuEmbeddingEngineState_GetState` | `0x06e62330` | `const void* TpuEmbeddingEngineState_GetState(const TpuEmbeddingEngineState* state)` | Returns internal state pointer |
| `TpuEmbeddingEngine_ExecutePartitioner` | `0x06da4830` | `void TpuEmbeddingEngine_ExecutePartitioner(TpuEmbeddingEngine_ExecutePartitioner_Params* params)` | Partitions embedding configuration across TPU shards |
| `TpuEmbeddingEngine_ConfigureMemory` | `0x06da4a10` | `void TpuEmbeddingEngine_ConfigureMemory(TpuEmbeddingEngine_ConfigureMemory_Params* params)` | Allocates HBM for embedding tables |
| `TpuEmbeddingEngine_CollateMemory` | `0x06da4c30` | `void TpuEmbeddingEngine_CollateMemory(TpuEmbeddingEngine_CollateMemory_Params* params)` | Collates HBM addresses across hosts |
| `TpuEmbeddingEngine_ConfigureHost` | `0x06da4fb0` | `void TpuEmbeddingEngine_ConfigureHost(TpuEmbeddingEngine_ConfigureHost_Params* params)` | Configures host-side embedding subsystem |
| `TpuEmbeddingEngine_ConnectHosts` | `0x06da5330` | `void TpuEmbeddingEngine_ConnectHosts(TpuEmbeddingEngine_ConnectHosts_Params* params)` | Establishes inter-host ICI/gRPC embedding channels |
| `TpuEmbeddingEngine_Finalize` | `0x06da5620` | `void TpuEmbeddingEngine_Finalize(TpuEmbeddingEngine_Finalize_Params* params)` | Finalizes embedding engine initialization |
| `TpuEmbeddingEngine_IsInitialized` | `0x06da5850` | `void TpuEmbeddingEngine_IsInitialized(TpuEmbeddingEngine_IsInitialized_Params* params)` | Checks whether embedding engine is initialized |
| `TpuEmbeddingEngine_WriteParameters` | `0x06da59d0` | `void TpuEmbeddingEngine_WriteParameters(TpuEmbeddingEngine_WriteParameters_Params* params)` | Writes embedding table weights to HBM |
| `TpuEmbeddingEngine_ReadParameters` | `0x06da5de0` | `void TpuEmbeddingEngine_ReadParameters(TpuEmbeddingEngine_ReadParameters_Params* params)` | Reads embedding table weights from HBM |
| `TpuEmbeddingTensorBatchFixedState_Create` | `0x06da6310` | `TpuEmbeddingEngineState* TpuEmbeddingTensorBatchFixedState_Create(TpuStatus* status)` | Creates fixed state for tensor batch enqueue |
| `TpuEmbeddingTensorBatchFixedState_Destroy` | `0x06da67c0` | `void TpuEmbeddingTensorBatchFixedState_Destroy(TpuEmbeddingEngineState* state)` | Destroys fixed state for tensor batch enqueue |
| `TpuEmbeddingEngine_EnqueueTensorBatch` | `0x06da8280` | `void TpuEmbeddingEngine_EnqueueTensorBatch(TpuEmbeddingEngine_EnqueueTensorBatch_Params* params)` | Enqueues sparse embedding lookup batch |
| `TpuEmbeddingEngine_RecvActivationsComputation` | `0x06e62980` | `void TpuEmbeddingEngine_RecvActivationsComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for receiving embedding activations |
| `TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation` | `0x06e632b0` | `void TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for receiving deduplication data |
| `TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation` | `0x06e63bf0` | `void TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for sending embedding gradients |
| `TpuEmbeddingEngine_DedupDataSizeComputation` | `0x06e645e0` | `void TpuEmbeddingEngine_DedupDataSizeComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for deduplication data byte size |
| `TpuEmbeddingEngine_DedupDataTupleMaskComputation` | `0x06e64a30` | `void TpuEmbeddingEngine_DedupDataTupleMaskComputation(TpuEmbeddingEngine_Computation_Params* params)` | Builds XLA computation for deduplication tuple mask |

### 2.8 TPU Driver C API (`TpuDriver_*`)
| Symbol Name | Virtual Address | C Signature | Description |
| :--- | :--- | :--- | :--- |
| `TpuDriver_Initialize` | `0x0c633480` | `void TpuDriver_Initialize(TpuDriverFn* driver_fn, bool initialize)` | Populates a `TpuDriverFn` vtable and optionally initializes the driver |
| `TpuDriver_Open` | `0x0c633600` | `TpuDriver* TpuDriver_Open(const char* worker)` | Opens a `TpuDriver` instance with populated `TpuDriverFn*` vtable |
| `TpuDriver_Close` | `0x0c6337b0` | `void TpuDriver_Close(TpuDriver* driver)` | Closes and frees a `TpuDriver` instance |

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

