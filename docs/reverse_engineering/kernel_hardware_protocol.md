# Kernel, Hardware, Ioctl & Protobuf Protocol Specification (`libtpu-0.0.50`)

## 1. Kernel Driver Backends (`asic_sw::driver`)
`libtpu.so` contains two user-space PCIe device access backends in namespace `asic_sw::driver`:
1. **Gasket / Accel Kernel Driver (`KernelPcieEndpoint`, `KernelDmaMapper`, `GasketEventfdRegistrator`, `GasketDeviceInspector`)**:
   - Device paths: `/dev/accel0` .. `/dev/accel63`
   - Sysfs paths: `/sys/class/accel/accel<N>/` (`bar_size`, `mmap_offset`, `pci_address`, `driver_version`, `numa_node`, `reset_count`, `write_open_count`, `status`, `is_device_owned`, `device_owner`, `framework_version`, `user_mem_ranges`, `interrupt_counts`, `chip_model`).
2. **VFIO User-Space PCIe Driver (`VfioDeviceAccess`, `VfioPcieEndpoint`, `VfioKernelDmaMapper`, `VfioInterruptEventfdRegistrator`, `VfioDeviceInspector`)**:
   - Container path: `/dev/vfio/vfio` (`kVfioDevicePath` @ `0x0953670c`)
   - IOMMU group paths: `/dev/vfio/<iommu_group>`
   - Sysfs PCIe discovery paths: `/sys/bus/pci/devices/<bdf>/iommu_group`, `physfn`, `resource`, `sriov_vf`.

## 2. Exact `ioctl` Command Numbers Extracted from Machine Code

### 2.1 Gasket (`0xdc` Magic) Ioctls
| Method in `libtpu.so` | Address | Ioctl Hex | Decoded `_IO*` Macro | Payload Size | Purpose |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `KernelDmaMapper::GetNumberSupportedSimpleAddresses` | `0x0b627090` | `0xc020dc05` | `_IOWR(0xdc, 0x05, gasket_page_table_ioctl)` | 32 bytes | Queries simple page table size |
| `KernelDmaMapper::SetNumberSimpleAddresses` | `0x0b627140` | `0x4020dc07` | `_IOW(0xdc, 0x07, gasket_page_table_ioctl)` | 32 bytes | Partitions simple vs extended page tables |
| `KernelDmaMapper::MapMemory` | `0x0b626e90` | `0x4020dc08` | `_IOW(0xdc, 0x08, gasket_page_table_ioctl)` | 32 bytes | Maps host virtual pages to device DMA address |
| `KernelDmaMapper::UnmapMemory` | `0x0b626f10` | `0x4020dc09` | `_IOW(0xdc, 0x09, gasket_page_table_ioctl)` | 32 bytes | Unmaps host virtual pages from device DMA address |
| `GasketEventfdRegistrator::RegisterEventFd` | `0x0b627950` | `0x4020dc0b` | `_IOW(0xdc, 0x0b, gasket_interrupt_eventfd)` | 32 bytes | Binds an `eventfd` to a TPU interrupt index |
| `GasketEventfdRegistrator::UnregisterEventFd` | `0x0b6279a0` | `0x4008dc0c` | `_IOW(0xdc, 0x0c, uint64_t)` | 8 bytes | Clears an interrupt `eventfd` binding |
| `KernelDmaMapper::MapDmaBufMemory` | `0x0b626ed0` | `0x4018dc0d` | `_IOW(0xdc, 0x0d, gasket_page_table_dmabuf)` | 24 bytes | Maps a `dma_buf` fd into TPU DMA space |

### 2.2 VFIO (`0x3b` / `';'` Magic) Ioctls
| Method in `libtpu.so` | Address | Ioctl Hex | Standard VFIO Constant | Payload Size | Purpose |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `VfioInterruptEventfdRegistrator::RegisterEventFd` | `0x19f9e340` | `0x00003b6e` | `VFIO_DEVICE_SET_IRQS` | 24 bytes | Configures MSI-X interrupt signaling with `eventfd` (`flags = 0x24`) |
| `VfioKernelDmaMapper::MapMemory` | `0x19fa5b30` | `0x00003b71` | `VFIO_IOMMU_MAP_DMA` | 32 bytes | Maps host VMA into IOMMU IOVA (`flags = 0x3` READ\|WRITE, page shift `<< 12`) |
| `VfioKernelDmaMapper::UnmapMemory` | `0x19fa5ba0` | `0x00003b72` | `VFIO_IOMMU_UNMAP_DMA` | 24 bytes | Unmaps IOMMU IOVA range (page shift `<< 12`) |

## 3. TPU Chip Generations, PCIe IDs & Internal Codenames (`asic_sw::deepsea`)
All Google Cloud TPU PCIe endpoints share PCIe Vendor ID **`0x1AE0`** (`Google, Inc.`):
- **TPU v2 (`Jellyfish`, Device ID `0x0027`, rev `0x00`)**: 2 TensorCores/chip, 8 GiB HBM per core (16 GiB/chip), 1 MXU per core (`128x128`).
- **TPU v3 (`Dragonfish`, Device ID `0x0027`, rev `0x01`)**: 2 TensorCores/chip, 16 GiB HBM per core (32 GiB/chip), 2 MXUs per core (`128x128`).
- **TPU v4 (`Pufferfish` / `pxc` / `pfc` / `plc`, Device ID `0x0056`)**: 2 TensorCores/chip (Megacore unified), 32 GiB HBM/chip, SparseCore v1 (4 per chip), 3D torus optical ICI.
- **TPU v5e / v5lite (`Viperfish` / `vxc` / `vlc`, Device ID `0x0063`)**: 1 TensorCore/chip, 16 GiB HBM/chip, 4 MXUs (`128x128`), 2D/3D ICI.
- **TPU v5p (`Viperfish` / `vxc` / `vfc`, Device ID `0x0062`)**: 2 TensorCores/chip (Megacore), 95 GiB HBM/chip, 4 SparseCores/chip, 3D torus ICI.
- **TPU v6e / Trillium (`Ghostfish` / `gxc` / `glc`, Device ID `0x006F`)**: 1 TensorCore/chip, 32 GiB HBM/chip, `256x256` MXU, SparseCore v2, 2x peak FLOPs and HBM bandwidth over v5e.
- **TPU7x (`Ironwood` / `iwc`, Device ID `0x0076`)**: 2 TensorCores/chip, 192 GiB HBM3e/chip, `256x256` MXU, SparseCore v3, 3D optical ICI.

## 4. BAR0 / BAR2 MMIO Control & Status Register (CSR) Map
- **BAR0 (`0x0000_0000` .. `0x003F_FFFF`)**:
  - `0x0000_1000`: `BAR0_CSR_TENSORCORE_RUN_CONTROL` (`0x1` = Start/Run, `0x2` = Halt, `0x4` = Reset)
  - `0x0000_1008`: `BAR0_CSR_TENSORCORE_PC` (64-bit instruction bundle base address)
  - `0x0000_1010`: `BAR0_CSR_TENSORCORE_STATUS` (`0x0` = Idle, `0x1` = Running, `0x2` = Completed)
  - `0x0000_2000`: `BAR0_CSR_SYNC_FLAG_BASE` (4,096 32-bit architectural `SyncFlag` counters)
  - `0x0000_4000`: `BAR0_CSR_SPARSECORE_RUN_CONTROL`
  - `0x0000_8000`: `BAR0_CSR_HBM_DMA_RING_HEAD` / `0x0000_8008`: `BAR0_CSR_HBM_DMA_RING_TAIL`
  - `0x0000_9000`: `BAR0_CSR_ICI_ROUTER_LINK_STATE`
  - `0x0000_A000`: `BAR0_CSR_TELEMETRY_THERMAL_MILLICELSIUS`, `0x0000_A008`: `BAR0_CSR_TELEMETRY_POWER_MILLIWATTS`, `0x0000_A010`: `BAR0_CSR_TELEMETRY_DUTY_CYCLE_PCT`
- **BAR2 (`0x0000_0000` ..)**: Direct HBM aperture / VMEM window mapping.

## 5. Embedded Protobuf Wire Schemas (`protodesc_cold`)
- **`xla.TileProto`**: tag `1` (`dimensions`, repeated `int64`).
- **`xla.LayoutProto`**: tag `1` (`minor_to_major`, repeated `int64`), tag `2` (`tiles`, repeated `TileProto`), tag `3` (`element_size_in_bits`, `int64`), tag `4` (`memory_space`, `int64`).
- **`xla.ShapeProto`**: tag `1` (`element_type`, `int32`), tag `2` (`dimensions`, repeated `int64`), tag `3` (`tuple_shapes`, repeated `ShapeProto`), tag `4` (`layout`, `LayoutProto`), tag `5` (`is_dynamic_dimension`, repeated `bool`).
- **`xla.OpSharding`**: tag `1` (`type`, `int32`), tag `2` (`tile_shape`, `ShapeProto`), tag `3` (`tile_assignment_dimensions`, repeated `int64`), tag `4` (`tile_assignment_devices`, repeated `int64`), tag `5` (`tuple_shardings`, repeated `OpSharding`), tag `6` (`replicate_on_last_tile_dim`, `bool`).
- **`xla.HloInstructionProto` / `xla.HloComputationProto` / `xla.HloModuleProto`**: wire-compatible XLA HLO DAG representation with instruction `name`, `opcode`, `shape`, `operand_ids`, `literal_bytes`, `dimensions`, and `id`.
- **`xla.ExecutableBuildOptionsProto` & `xla.CompileOptionsProto`**: `num_replicas`, `num_partitions`, `device_ordinal`, `result_layout`, `parameter_is_tupled_arguments`, and `env_option_overrides`.
- **`xla.DeviceAssignmentProto` & `xla.ComputationDeviceProto`**: `replica_count`, `computation_count`, and per-computation `replica_device_ids`.
- **`tpu.TpuTopologyArgsProto`, `tpu.TpuCompilationEnvironmentProto`, `tpu.TpuSerializedExecutableProto`, `tpu.TpuEmbeddingConfigurationProto`**: TPU mesh bounds, compilation flags, serialized VLIW executable payload, and SparseCore embedding table definitions.
- **`tensorflow.profiler.XSpace` (`XPlane`, `XLine`, `XEvent`, `XEventMetadata`, `XStat`, `XStatMetadata`)**: TensorBoard / `jax.profiler` trace protobuf hierarchy.

## 6. Environment Variables Read by `libtpu.so`
- **Initialization & Library Resolution**:
  - `TPU_LIBRARY_PATH`
  - `LIBTPU_INIT_ARGS`
  - `TPU_STDERR_LOG_LEVEL`, `TPU_MIN_LOG_LEVEL`, `TPU_MAX_LOG_SIZE_MB`, `TPU_VMODULE`
- **Topology & Device Discovery**:
  - `TPU_ACCELERATOR_TYPE`
  - `TPU_CHIPS_PER_HOST_BOUNDS`, `TPU_CHIPS_PER_PROCESS_BOUNDS`, `TPU_HOST_BOUNDS`, `TPU_PROCESS_BOUNDS`
  - `TPU_VISIBLE_DEVICES`, `TPU_VISIBLE_CHIPS`, `TPU_VISIBLE_DEVICE_PATHS`
  - `TPU_PROCESS_ADDRESSES`, `TPU_PROCESS_PORT`, `TPU_WORKER_HOSTNAMES`, `TPU_HOSTNAME_OVERRIDE`
  - `TPU_MEGACORE`, `TPU_TOPOLOGY_ALT`, `TPU_TOPOLOGY_WRAP`
- **DMA & Memory Management**:
  - `TPU_PREMAPPED_BUFFER_SIZE`, `TPU_PREMAPPED_BUFFER_PARTITION_COUNT`, `TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES`
  - `TPU_HBM_REPORT_ENABLE`, `TPU_HBM_REPORT_FREQUENCY`
- **Megascale & Multi-Slice**:
  - `MEGASCALE_COORDINATOR_ADDRESS`, `MEGASCALE_NUM_SLICES`, `MEGASCALE_SLICE_ID`, `MEGASCALE_TRANSPORT_TYPE`, `MEGASCALE_DEBUG_PORT`
- **Telemetry & Checkpointing**:
  - `TPU_RUNTIME_METRICS_ADDRESS`, `TPU_RUNTIME_METRICS_PORTS`, `TPU_RUNTIME_PRIMARY_METRIC_PORT`
  - `LIBTPU_CHECKPOINTING_ENABLED`, `LIBTPU_CHECKPOINTING_SAVE_PARALLEL`, `LIBTPU_CHECKPOINTING_IN_PLACE_RESTORE_ENABLE`
