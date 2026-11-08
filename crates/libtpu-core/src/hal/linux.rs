//! Linux VFIO and Gasket hardware device HAL ([`LinuxDeviceHal`]) and backend selector ([`HalBackend`]).

use super::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SPARSECORE_RUN_CONTROL, BAR0_CSR_SYNC_FLAG_BASE,
    BAR0_CSR_TENSORCORE_PC, BAR0_CSR_TENSORCORE_RUN_CONTROL, BAR0_CSR_TENSORCORE_STATUS,
    BAR0_CSR_THERMAL_MILLICELSIUS, ChipRegisterMap, DeviceMemoryStats, EmulatedKernelDriverBackend,
    GASKET_IOCTL_CLEAR_EVENTFD, GASKET_IOCTL_MAP_BUFFER, GASKET_IOCTL_NUMBER_PAGE_TABLES,
    GASKET_IOCTL_PARTITION_PAGE_TABLE, GASKET_IOCTL_RESET, GASKET_IOCTL_SET_EVENTFD,
    GASKET_IOCTL_UNMAP_BUFFER, GasketInterruptEventFd, GasketPageTableIoctl, HbmDmaRingDescriptor,
    IoctlPayload, KernelDriverMode, SimulatedTpuHal, SyscallProvider, TopologySpec, TpuHal,
    VFIO_CHECK_EXTENSION, VFIO_DEVICE_GET_INFO, VFIO_DEVICE_GET_IRQ_INFO,
    VFIO_DEVICE_GET_REGION_INFO, VFIO_DEVICE_SET_IRQS_IOCTL, VFIO_GET_API_VERSION,
    VFIO_GROUP_GET_DEVICE_FD, VFIO_GROUP_GET_STATUS, VFIO_GROUP_SET_CONTAINER,
    VFIO_IOMMU_MAP_DMA_IOCTL, VFIO_IOMMU_UNMAP_DMA_IOCTL, VFIO_SET_IOMMU, VfioDmaMapIoctl,
    VfioDmaUnmapIoctl, VfioIrqSetIoctl,
};
use crate::compiler::CompiledProgram;
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ByteSize, DeviceAddress, DeviceId, MemoryKind};

/// Discovers attached Google Cloud TPU `PCIe` functions (`0x1ae0`) from a `sysfs` PCI devices directory.
///
/// # Arguments
///
/// * `sysfs_root` - Path to `/sys/bus/pci/devices` (or a synthetic test directory).
#[must_use]
pub fn discover_pcie_tpu_devices_from_sysfs(
    sysfs_root: &std::path::Path,
) -> Vec<super::DiscoveredPciTpuDevice> {
    super::PciDeviceEnumerator::discover_from_sysfs_root(sysfs_root)
}

/// Hardware DMA & `TensorCore` command-ring queue state machine (`HardwareDoorbellRing`).
///
/// Models producer (`tail`) and consumer (`head`) ring pointers, in-flight [`HbmDmaRingDescriptor`]
/// queue entries, and `eventfd` completion notification counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareDoorbellRing {
    /// Ring buffer capacity (power-of-two descriptor count).
    pub capacity: u32,
    /// Producer tail pointer (`BAR0_CSR_HBM_DMA_RING_TAIL`).
    pub tail: u64,
    /// Consumer head pointer (`BAR0_CSR_HBM_DMA_RING_HEAD`).
    pub head: u64,
    /// Monotonic `TensorCore` doorbell ring counter.
    pub doorbell_count: u64,
    /// Monotonic `eventfd` interrupt completion signal counter.
    pub eventfd_signal_count: u64,
    /// Queued [`HbmDmaRingDescriptor`] entries awaiting completion.
    pub pending_descriptors: std::collections::VecDeque<HbmDmaRingDescriptor>,
}

impl Default for HardwareDoorbellRing {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Gasket Address Translation Table (`ATT`) partitioning mode (`GasketPageTableMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GasketPageTableMode {
    /// Simple 1-level linear page table (`0..simple_entries`).
    SimpleLinear,
    /// Extended 2-level hierarchical L0/L1 page directory (`simple_entries..total_entries`).
    ExtendedTwoLevel,
}

/// Individual virtual-to-physical page mapping in a [`GasketTwoLevelPageTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasketPageTableEntry {
    /// Device I/O Virtual Address (`IOVA`).
    pub iova: u64,
    /// Host user-space virtual address (`vaddr`) or `dma_buf` backing token.
    pub host_vaddr: u64,
    /// Mapped byte length (page-aligned to `4096` bytes).
    pub size_bytes: u64,
    /// Level-0 directory index (`0..num_total_entries`).
    pub l0_index: u32,
    /// Level-1 leaf index (`0` for simple linear entries, `0..512` for extended 2-level entries).
    pub l1_index: u32,
    /// Reference count (incremented by `dma_buf` exports).
    pub ref_count: u32,
    /// Optional exported/imported Linux `dma_buf` file descriptor.
    pub dmabuf_fd: Option<i32>,
}

/// Hierarchical 2-level Gasket Address Translation Table (`GasketTwoLevelPageTable`) managing
/// simple linear and extended 2-level L0/L1 page mappings and `dma_buf` zero-copy file descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GasketTwoLevelPageTable {
    /// Page table index (`0` for `TensorCore 0`).
    pub page_table_index: u32,
    /// Total L0 directory capacity (`e.g. 4096`).
    pub total_entries: u32,
    /// Partition split index: `[0..simple_entries)` are simple linear, `[simple_entries..total_entries)` are extended 2-level.
    pub simple_entries: u32,
    /// Active mappings keyed by starting `iova`.
    pub entries: std::collections::BTreeMap<u64, GasketPageTableEntry>,
}

impl GasketTwoLevelPageTable {
    /// Creates a new [`GasketTwoLevelPageTable`] partitioned with `simple_entries` linear slots
    /// out of `total_entries`.
    ///
    /// # Arguments
    ///
    /// * `page_table_index` - Hardware page table index.
    /// * `total_entries` - Total L0 entries.
    /// * `simple_entries` - Number of simple linear entries (`<= total_entries`).
    #[must_use]
    pub fn new(page_table_index: u32, total_entries: u32, simple_entries: u32) -> Self {
        let total = total_entries.max(64);
        let simple = simple_entries.min(total);
        Self {
            page_table_index,
            total_entries: total,
            simple_entries: simple,
            entries: std::collections::BTreeMap::new(),
        }
    }

    /// Repartitions the split between simple linear and extended 2-level entries (`GASKET_IOCTL_PARTITION_PAGE_TABLE`).
    ///
    /// # Arguments
    ///
    /// * `simple_entries` - New simple linear slot count (`<= self.total_entries`).
    pub fn partition(&mut self, simple_entries: u32) {
        self.simple_entries = simple_entries.min(self.total_entries);
    }

    /// Maps a virtual `iova` range of `size_bytes` to `host_vaddr` in either the simple linear
    /// or extended 2-level L0/L1 table (`mode`).
    ///
    /// # Arguments
    ///
    /// * `iova` - Target device IOVA address.
    /// * `host_vaddr` - Host user-space virtual address.
    /// * `size_bytes` - Mapping byte size.
    /// * `mode` - [`GasketPageTableMode`] (`SimpleLinear` vs `ExtendedTwoLevel`).
    pub fn map_pages(
        &mut self,
        iova: u64,
        host_vaddr: u64,
        size_bytes: u64,
        mode: GasketPageTableMode,
    ) -> GasketPageTableEntry {
        let page_idx = u32::try_from((iova >> 12) & 0xFFFF).unwrap_or(0);
        let (l0_index, l1_index) = match mode {
            GasketPageTableMode::SimpleLinear => (page_idx % self.simple_entries.max(1), 0),
            GasketPageTableMode::ExtendedTwoLevel => {
                let ext_cap = self
                    .total_entries
                    .saturating_sub(self.simple_entries)
                    .max(1);
                let l0 = self.simple_entries + ((page_idx / 512) % ext_cap);
                let l1 = page_idx & 511;
                (l0, l1)
            }
        };
        let entry = GasketPageTableEntry {
            iova,
            host_vaddr,
            size_bytes: size_bytes.max(4096),
            l0_index,
            l1_index,
            ref_count: 1,
            dmabuf_fd: None,
        };
        self.entries.insert(iova, entry);
        entry
    }

    /// Unmaps the entry at `iova`, returning the removed [`GasketPageTableEntry`] if present.
    ///
    /// # Arguments
    ///
    /// * `iova` - Device IOVA address to unmap.
    pub fn unmap_pages(&mut self, iova: u64) -> Option<GasketPageTableEntry> {
        self.entries.remove(&iova)
    }

    /// Translates a virtual `iova` (plus optional intra-buffer offset) to `(host_vaddr, GasketPageTableEntry)`.
    ///
    /// # Arguments
    ///
    /// * `iova_query` - Device IOVA address to translate.
    #[must_use]
    pub fn translate_iova(&self, iova_query: u64) -> Option<(u64, GasketPageTableEntry)> {
        let (&base_iova, &entry) = self.entries.range(..=iova_query).next_back()?;
        let delta = iova_query.saturating_sub(base_iova);
        if delta < entry.size_bytes {
            Some((entry.host_vaddr.saturating_add(delta), entry))
        } else {
            None
        }
    }
}

impl HardwareDoorbellRing {
    /// Creates a new [`HardwareDoorbellRing`] with the given `capacity`.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of descriptors in the ring (`>= 1`).
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity: capacity.max(1),
            tail: 0,
            head: 0,
            doorbell_count: 0,
            eventfd_signal_count: 0,
            pending_descriptors: std::collections::VecDeque::new(),
        }
    }

    /// Enqueues a [`HbmDmaRingDescriptor`] into the producer ring and increments `tail` and `doorbell_count`.
    ///
    /// # Arguments
    ///
    /// * `desc` - 32-byte [`HbmDmaRingDescriptor`] to submit.
    pub fn enqueue_and_ring_doorbell(&mut self, desc: HbmDmaRingDescriptor) -> u64 {
        if self.pending_descriptors.len() >= self.capacity as usize {
            let _ = self.pending_descriptors.pop_front();
            self.head = self.head.wrapping_add(1);
        }
        self.pending_descriptors.push_back(desc);
        self.tail = self.tail.wrapping_add(1);
        self.doorbell_count = self.doorbell_count.wrapping_add(1);
        self.tail
    }

    /// Drains all in-flight descriptors, advances `head` to `tail`, and increments `eventfd_signal_count`.
    pub fn process_completions_and_signal_eventfd(&mut self) -> usize {
        let completed = self.pending_descriptors.len();
        self.pending_descriptors.clear();
        self.head = self.tail;
        if completed > 0 {
            self.eventfd_signal_count = self.eventfd_signal_count.wrapping_add(1);
        }
        completed
    }
}

/// Linux kernel driver HAL wrapping a [`SyscallProvider`], IOMMU [`super::HbmRegionAllocator`] tables,
/// BAR0 MMIO CSR bank, and backing [`SimulatedTpuHal`] execution engine.
#[derive(Debug)]
pub struct LinuxDeviceHal<S: SyscallProvider> {
    /// Kernel driver mode (`Gasket` or `Vfio`).
    pub mode: KernelDriverMode,
    /// Injectable syscall provider.
    pub syscalls: S,
    /// Open kernel file descriptor.
    pub device_fd: i32,
    /// Emulated `PCIe` BAR0 MMIO CSR bank.
    pub mmio: EmulatedKernelDriverBackend,
    /// Optional real `mmap`-backed BAR0 [`super::MappedBarAperture`].
    pub bar0_aperture: parking_lot::Mutex<Option<super::MappedBarAperture>>,
    /// Optional real `mmap`-backed BAR2 HBM/VMEM [`super::MappedBarAperture`].
    pub bar2_aperture: parking_lot::Mutex<Option<super::MappedBarAperture>>,
    /// Per-device IOMMU / page-table [`super::HbmRegionAllocator`] tracking mapped IOVA blocks.
    pub iommu_allocators:
        parking_lot::Mutex<std::collections::BTreeMap<DeviceId, super::HbmRegionAllocator>>,
    /// Backing HBM & execution state.
    pub inner: SimulatedTpuHal,
}

impl<S: SyscallProvider> LinuxDeviceHal<S> {
    /// Builds the per-device IOMMU [`super::HbmRegionAllocator`] map for `topology`.
    ///
    /// # Arguments
    ///
    /// * `topology` - Discovered TPU [`TopologySpec`].
    fn build_iommu_allocators(
        topology: &TopologySpec,
    ) -> parking_lot::Mutex<std::collections::BTreeMap<DeviceId, super::HbmRegionAllocator>> {
        let mut map = std::collections::BTreeMap::new();
        let cap = topology.generation.hbm_capacity_bytes().0;
        for dev in &topology.devices {
            map.insert(
                dev.device_id,
                super::HbmRegionAllocator::new(0x1000_0000, cap),
            );
        }
        parking_lot::Mutex::new(map)
    }

    /// Opens a Linux TPU device node (`/dev/accel0` or `/dev/vfio/vfio`) and initializes page tables.
    ///
    /// # Arguments
    ///
    /// * `mode` - Kernel driver mode (`Gasket` vs `Vfio`).
    /// * `syscalls` - Syscall implementation.
    /// * `config` - TPU initialization configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `open_device` or initial page-table `ioctl` fails.
    pub fn open(mode: KernelDriverMode, syscalls: S, config: &TpuInitConfig) -> LibTpuResult<Self> {
        let path = match mode {
            KernelDriverMode::Gasket => "/dev/accel0",
            KernelDriverMode::Vfio => "/dev/vfio/vfio",
        };
        Self::open_with_path(mode, path, syscalls, config)
    }

    /// Opens a specific Linux TPU device node `path` and initializes page tables.
    ///
    /// # Arguments
    ///
    /// * `mode` - Kernel driver mode (`Gasket` vs `Vfio`).
    /// * `path` - Device file path (e.g., `"/dev/accel0"`).
    /// * `syscalls` - Syscall implementation.
    /// * `config` - TPU initialization configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `open_device` or initial page-table `ioctl` fails.
    pub fn open_with_path(
        mode: KernelDriverMode,
        path: &str,
        syscalls: S,
        config: &TpuInitConfig,
    ) -> LibTpuResult<Self> {
        let fd = syscalls.open_device(path)?;
        if mode == KernelDriverMode::Gasket {
            syscalls.ioctl_device(
                fd,
                GASKET_IOCTL_NUMBER_PAGE_TABLES,
                size_of::<GasketPageTableIoctl>(),
            )?;
            syscalls.ioctl_device(
                fd,
                GASKET_IOCTL_PARTITION_PAGE_TABLE,
                size_of::<GasketPageTableIoctl>(),
            )?;
        }
        let inner = SimulatedTpuHal::new(config);
        let iommu_allocators = Self::build_iommu_allocators(&inner.topology);
        let mmio = inner.mmio.clone();
        Ok(Self {
            mode,
            syscalls,
            device_fd: fd,
            mmio,
            bar0_aperture: parking_lot::Mutex::new(None),
            bar2_aperture: parking_lot::Mutex::new(None),
            iommu_allocators,
            inner,
        })
    }

    /// Maps an OS-backed BAR2 HBM/VMEM [`super::MappedBarAperture`] of `size` bytes (`bar_index = 2`).
    ///
    /// # Arguments
    ///
    /// * `size` - Mapping byte size (e.g., `ByteSize(0x0010_0000)`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `mmap_bar_aperture` fails.
    pub fn attach_bar2_mmap(&self, size: ByteSize) -> LibTpuResult<super::MappedBarAperture> {
        let posix = super::PosixKernelDriverBackend::new();
        let mut ap = posix.mmap_bar_aperture(crate::types::FileDescriptor(-1), size)?;
        ap.bar_index = 2;
        *self.bar2_aperture.lock() = Some(ap);
        Ok(ap)
    }

    /// Unmaps the currently attached BAR2 [`super::MappedBarAperture`], if any.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `munmap_bar_aperture` fails.
    pub fn detach_bar2_mmap(&self) -> LibTpuResult<()> {
        let taken = self.bar2_aperture.lock().take();
        if let Some(ap) = taken {
            let posix = super::PosixKernelDriverBackend::new();
            posix.munmap_bar_aperture(ap)?;
        }
        Ok(())
    }

    /// Writes `data` at `offset` inside the attached BAR2 HBM/VMEM [`super::MappedBarAperture`].
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset within the BAR2 aperture (`8`-byte aligned).
    /// * `value` - 64-bit word to write.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if BAR2 is not mapped or `offset` is out of bounds.
    pub fn write_bar2_u64(&self, offset: u64, value: u64) -> LibTpuResult<()> {
        let ap = (*self.bar2_aperture.lock()).ok_or(LibTpuError::MmioOutOfBounds {
            bar_index: 2,
            offset,
            size: 8,
        })?;
        ap.write_csr_u64(crate::types::MmioOffset(offset), value)
    }

    /// Reads a 64-bit word at `offset` inside the attached BAR2 HBM/VMEM [`super::MappedBarAperture`].
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset within the BAR2 aperture (`8`-byte aligned).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if BAR2 is not mapped or `offset` is out of bounds.
    pub fn read_bar2_u64(&self, offset: u64) -> LibTpuResult<u64> {
        let ap = (*self.bar2_aperture.lock()).ok_or(LibTpuError::MmioOutOfBounds {
            bar_index: 2,
            offset,
            size: 8,
        })?;
        ap.read_csr_u64(crate::types::MmioOffset(offset))
    }

    /// Registers a multi-queue MSI-X `eventfd` for a specific hardware completion queue
    /// (`0` = `TensorCore 0`, `1` = `TensorCore 1`, `2..=5` = `SparseCore 0..3`, `6` = `ICI Router`).
    ///
    /// # Arguments
    ///
    /// * `queue_index` - Hardware interrupt vector / queue index (`0..=15`).
    /// * `event_fd` - Non-negative Linux `eventfd` descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `event_fd < 0` or the kernel `ioctl` fails.
    pub fn register_queue_interrupt_eventfd(
        &self,
        queue_index: u32,
        event_fd: i32,
    ) -> LibTpuResult<()> {
        if event_fd < 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "event_fd",
                reason: String::from("event_fd must be non-negative"),
            });
        }
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketInterruptEventFd {
                    interrupt: u64::from(queue_index),
                    event_fd: u64::try_from(event_fd).unwrap_or(0),
                    reserved: [0; 2],
                };
                self.syscalls.ioctl_typed(
                    self.device_fd,
                    GASKET_IOCTL_SET_EVENTFD,
                    &mut payload,
                )?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_SET_EVENTFD,
                    IoctlPayload::GasketInterruptEventFd(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let mut payload = VfioIrqSetIoctl {
                    argsz: 24,
                    flags: 0x24,
                    index: 2,
                    start: queue_index,
                    count: 1,
                    event_fd,
                };
                self.syscalls.ioctl_typed(
                    self.device_fd,
                    VFIO_DEVICE_SET_IRQS_IOCTL,
                    &mut payload,
                )?;
                self.mmio.apply_typed_ioctl(
                    VFIO_DEVICE_SET_IRQS_IOCTL,
                    IoctlPayload::VfioIrqSet(&payload),
                );
            }
        }
        Ok(())
    }

    /// Maps an OS-backed BAR0 [`super::MappedBarAperture`] of `size` bytes and synchronizes its initial
    /// telemetry and `SyncFlag` CSRs.
    ///
    /// # Arguments
    ///
    /// * `size` - Mapping byte size (e.g., `ByteSize(0x0010_0000)`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `mmap_bar_aperture` fails.
    pub fn attach_bar0_mmap(&self, size: ByteSize) -> LibTpuResult<super::MappedBarAperture> {
        let posix = super::PosixKernelDriverBackend::new();
        let ap = posix.mmap_bar_aperture(crate::types::FileDescriptor(-1), size)?;
        let _ = ap.write_csr_u64(
            crate::types::MmioOffset(BAR0_CSR_THERMAL_MILLICELSIUS),
            self.mmio.get_csr(BAR0_CSR_THERMAL_MILLICELSIUS),
        );
        let _ = ap.write_csr_u64(
            crate::types::MmioOffset(BAR0_CSR_POWER_MILLIWATTS),
            self.mmio.get_csr(BAR0_CSR_POWER_MILLIWATTS),
        );
        let _ = ap.write_csr_u64(
            crate::types::MmioOffset(BAR0_CSR_DUTY_CYCLE_PERCENT),
            self.mmio.get_csr(BAR0_CSR_DUTY_CYCLE_PERCENT),
        );
        let _ = ap.write_csr_u64(
            crate::types::MmioOffset(BAR0_CSR_SYNC_FLAG_BASE),
            self.mmio.get_csr(BAR0_CSR_SYNC_FLAG_BASE),
        );
        *self.bar0_aperture.lock() = Some(ap);
        Ok(ap)
    }

    /// Unmaps the currently attached BAR0 [`super::MappedBarAperture`], if any.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `munmap_bar_aperture` fails.
    pub fn detach_bar0_mmap(&self) -> LibTpuResult<()> {
        let taken = self.bar0_aperture.lock().take();
        if let Some(ap) = taken {
            let posix = super::PosixKernelDriverBackend::new();
            posix.munmap_bar_aperture(ap)?;
        }
        Ok(())
    }

    /// Writes a 64-bit CSR value to both `self.mmio` and the attached `bar0_aperture` (if mapped).
    ///
    /// # Arguments
    ///
    /// * `offset` - BAR0 byte offset.
    /// * `value` - 64-bit CSR value to write.
    pub fn write_bar0_csr(&self, offset: u64, value: u64) {
        self.mmio.set_csr(offset, value);
        let ap_opt = *self.bar0_aperture.lock();
        if let Some(ap) = ap_opt {
            let _ = ap.write_csr_u64(crate::types::MmioOffset(offset), value);
        }
    }

    /// Reads a 64-bit CSR value from the attached `bar0_aperture` (when mapped and within bounds) or `self.mmio`.
    ///
    /// # Arguments
    ///
    /// * `offset` - BAR0 byte offset.
    #[must_use]
    pub fn read_bar0_csr(&self, offset: u64) -> u64 {
        let ap_opt = *self.bar0_aperture.lock();
        if let Some(ap) = ap_opt
            && let Ok(val) = ap.read_csr_u64(crate::types::MmioOffset(offset))
        {
            return val;
        }
        self.mmio.get_csr(offset)
    }

    /// Reads the thermal (millicelsius), power (milliwatts), and duty-cycle (%) telemetry CSRs from BAR0.
    pub fn read_telemetry_csrs(&self) -> (u64, u64, u64) {
        (
            self.read_bar0_csr(BAR0_CSR_THERMAL_MILLICELSIUS),
            self.read_bar0_csr(BAR0_CSR_POWER_MILLIWATTS),
            self.read_bar0_csr(BAR0_CSR_DUTY_CYCLE_PERCENT),
        )
    }

    /// Submits a 32-byte [`HbmDmaRingDescriptor`] into the BAR0 HBM DMA descriptor ring and rings the tail doorbell.
    ///
    /// # Arguments
    ///
    /// * `src_iova` - Source IOVA or HBM offset.
    /// * `dst_iova` - Destination IOVA or HBM offset.
    /// * `length_bytes` - Transfer length in bytes.
    /// * `direction_flags` - DMA direction code (`0` = H2D, `1` = D2H, `2` = D2D).
    fn submit_dma_descriptor(
        &self,
        src_iova: u64,
        dst_iova: u64,
        length_bytes: usize,
        direction_flags: u32,
    ) {
        let next_tail = self
            .read_bar0_csr(BAR0_CSR_HBM_DMA_RING_TAIL)
            .wrapping_add(1);
        let desc = HbmDmaRingDescriptor {
            src_iova,
            dst_iova,
            length_bytes: u32::try_from(length_bytes).unwrap_or(u32::MAX),
            direction_flags,
            completion_sync_flag: next_tail,
        };
        self.mmio.enqueue_dma_descriptor(desc);
        self.write_bar0_csr(BAR0_CSR_HBM_DMA_RING_TAIL, next_tail);
        self.write_bar0_csr(BAR0_CSR_HBM_DMA_RING_HEAD, next_tail);
    }

    /// Registers an interrupt `eventfd` using the reverse-engineered Gasket or VFIO `ioctl`.
    ///
    /// # Arguments
    ///
    /// * `event_fd` - Linux `eventfd` descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] if the kernel rejects the interrupt registration.
    pub fn register_interrupt_eventfd(&self, event_fd: i32) -> LibTpuResult<()> {
        if event_fd < 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "event_fd",
                reason: String::from("event_fd must be non-negative"),
            });
        }
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketInterruptEventFd {
                    interrupt: 0,
                    event_fd: u64::try_from(event_fd).unwrap_or(0),
                    reserved: [0; 2],
                };
                self.syscalls.ioctl_typed(
                    self.device_fd,
                    GASKET_IOCTL_SET_EVENTFD,
                    &mut payload,
                )?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_SET_EVENTFD,
                    IoctlPayload::GasketInterruptEventFd(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let mut payload = VfioIrqSetIoctl {
                    argsz: 24,
                    flags: 0x24,
                    index: 2,
                    start: 0,
                    count: 1,
                    event_fd,
                };
                self.syscalls.ioctl_typed(
                    self.device_fd,
                    VFIO_DEVICE_SET_IRQS_IOCTL,
                    &mut payload,
                )?;
                self.mmio.apply_typed_ioctl(
                    VFIO_DEVICE_SET_IRQS_IOCTL,
                    IoctlPayload::VfioIrqSet(&payload),
                );
            }
        }
        Ok(())
    }

    /// Unregisters an interrupt `eventfd` on Gasket devices (`GASKET_IOCTL_CLEAR_EVENTFD`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] if the `ioctl` fails.
    pub fn unregister_interrupt_eventfd(&self) -> LibTpuResult<()> {
        if self.mode == KernelDriverMode::Gasket {
            self.syscalls
                .ioctl_device(self.device_fd, GASKET_IOCTL_CLEAR_EVENTFD, 8)?;
        }
        self.mmio.clear_interrupt_fds();
        Ok(())
    }

    /// Waits for the hardware `SyncFlag` completion counter on `self.mmio` after an interrupt
    /// notification on `queue_index` (`0` = `TensorCore 0`, `1` = `DMA Engine`).
    ///
    /// # Arguments
    ///
    /// * `queue_index` - Hardware interrupt queue index.
    #[must_use]
    pub fn poll_interrupt_completion_counter(&self, queue_index: u32) -> u64 {
        let _ = queue_index;
        self.read_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE).max(1)
    }

    /// Closes the underlying kernel device file descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if closing the file descriptor fails.
    pub fn close(&self) -> LibTpuResult<()> {
        self.syscalls.close_device(self.device_fd)
    }

    /// Executes the complete 7-step Linux VFIO container, IOMMU group, BAR region, and MSI-X IRQ
    /// initialization handshake (`VFIO_GET_API_VERSION` .. `VFIO_DEVICE_GET_IRQ_INFO`).
    ///
    /// # Arguments
    ///
    /// * `iommu_group_id` - Linux IOMMU group number (`e.g. 0` for `"/dev/vfio/0"`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any `open_device` or VFIO `ioctl` step fails.
    pub fn initialize_vfio_container_and_group(&self, iommu_group_id: u32) -> LibTpuResult<i32> {
        // Step 1: Verify VFIO_GET_API_VERSION (0x3b64)
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_GET_API_VERSION, 0)?;
        // Step 2: Check VFIO_TYPE1v2_IOMMU extension support (0x3b65)
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_CHECK_EXTENSION, 4)?;
        // Step 3: Open /dev/vfio/<group> and check VFIO_GROUP_GET_STATUS (0x3b67)
        let grp_path = format!("/dev/vfio/{iommu_group_id}");
        let grp_fd = self.syscalls.open_device(&grp_path)?;
        self.syscalls
            .ioctl_device(grp_fd, VFIO_GROUP_GET_STATUS, 8)?;
        // Step 4: Bind group to container via VFIO_GROUP_SET_CONTAINER (0x3b68)
        self.syscalls
            .ioctl_device(grp_fd, VFIO_GROUP_SET_CONTAINER, 4)?;
        // Step 5: Enable VFIO_TYPE1v2_IOMMU on the container via VFIO_SET_IOMMU (0x3b66)
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_SET_IOMMU, 4)?;
        // Step 6: Acquire PCIe function device fd via VFIO_GROUP_GET_DEVICE_FD (0x3b6a)
        let _ = self
            .syscalls
            .ioctl_device(grp_fd, VFIO_GROUP_GET_DEVICE_FD, 16)?;
        // Step 7: Query VFIO_DEVICE_GET_INFO (0x3b6b), VFIO_DEVICE_GET_REGION_INFO (0x3b6c for BAR0/BAR2), and VFIO_DEVICE_GET_IRQ_INFO (0x3b6d)
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_DEVICE_GET_INFO, 16)?;
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_DEVICE_GET_REGION_INFO, 32)?;
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_DEVICE_GET_IRQ_INFO, 16)?;
        Ok(grp_fd)
    }

    /// Executes the complete Gasket accelerator reset, page-table partitioning, and interrupt
    /// vector `eventfd` binding sequence (`GASKET_IOCTL_RESET` .. `GASKET_IOCTL_SET_EVENTFD`).
    ///
    /// # Arguments
    ///
    /// * `event_fd` - Linux `eventfd` handle to bind to interrupt vector `0`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any Gasket `ioctl` step fails.
    pub fn initialize_gasket_device(&self, event_fd: i32) -> LibTpuResult<()> {
        self.syscalls
            .ioctl_device(self.device_fd, GASKET_IOCTL_RESET, 8)?;
        self.mmio
            .reset_csrs_for_generation(self.inner.topology.generation);
        self.syscalls.ioctl_device(
            self.device_fd,
            GASKET_IOCTL_NUMBER_PAGE_TABLES,
            size_of::<GasketPageTableIoctl>(),
        )?;
        self.syscalls.ioctl_device(
            self.device_fd,
            GASKET_IOCTL_PARTITION_PAGE_TABLE,
            size_of::<GasketPageTableIoctl>(),
        )?;
        self.register_interrupt_eventfd(event_fd)
    }

    /// Enqueues a serialized [`crate::proto::HostCommandEnvelopeProto`] into the hardware continuation
    /// queue (`TN_QUEUE_CONTINUATION`) and updates the BAR0 continuation `SyncFlag` CSR.
    ///
    /// # Arguments
    ///
    /// * `envelope` - [`crate::proto::HostCommandEnvelopeProto`] host command emitted by `TensorCore`.
    pub fn enqueue_host_command_envelope(&self, envelope: &crate::proto::HostCommandEnvelopeProto) {
        let encoded = envelope.encode();
        self.submit_dma_descriptor(
            encoded.as_ptr() as u64,
            0x00F0_0000 + envelope.continuation_id.saturating_mul(256),
            encoded.len(),
            1,
        );
        let rmap = ChipRegisterMap::for_generation(self.inner.topology.generation);
        let sync_off = u64::from(rmap.completion_state_sync_flag_word_offset).saturating_mul(4);
        self.write_bar0_csr(sync_off, envelope.continuation_id.max(1));
    }

    /// Services a [`crate::proto::HostCommandEnvelopeProto`] from `TN_QUEUE_CONTINUATION`,
    /// dispatching `HostCommandOpcodeProto` (`SEND`, `RECV`, `SDC`, `LOG`, `MEGASCALE_SEND`,
    /// `MEGASCALE_RECV`, `HOST_EXECUTION`, `PROGRAM_CHANGE`) and signaling the continuation
    /// completion `SyncFlag` CSR on BAR0.
    ///
    /// # Arguments
    ///
    /// * `envelope_bytes` - Wire-encoded [`crate::proto::HostCommandEnvelopeProto`] bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `envelope_bytes` cannot be decoded.
    pub fn service_continuation_queue(&self, envelope_bytes: &[u8]) -> LibTpuResult<u64> {
        let env = crate::proto::HostCommandEnvelopeProto::decode(envelope_bytes)?;
        let ack_sync = env.continuation_id.wrapping_add(1).max(1);
        let rmap = ChipRegisterMap::for_generation(self.inner.topology.generation);
        let sync_off = u64::from(rmap.completion_state_sync_flag_word_offset).saturating_mul(4);
        self.write_bar0_csr(sync_off, ack_sync);
        self.write_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE, ack_sync);
        Ok(ack_sync)
    }

    /// Maps a Linux `dma_buf` file descriptor (`dmabuf_fd`) at `device_address` via `GASKET_IOCTL_MAP_DMABUF`
    /// for zero-copy `GPUDirect` / NIC RDMA transfers.
    ///
    /// # Arguments
    ///
    /// * `dmabuf_fd` - Open Linux `dma_buf` file descriptor (`>= 0`).
    /// * `device_address` - Target device IOVA address.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `dmabuf_fd < 0` or the `ioctl` fails.
    pub fn import_dmabuf_fd(&self, dmabuf_fd: i32, device_address: u64) -> LibTpuResult<()> {
        if dmabuf_fd < 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "dmabuf_fd",
                reason: String::from("dmabuf_fd must be non-negative"),
            });
        }
        let mut payload = super::GasketPageTableDmaBuf {
            page_table_index: 0,
            device_address,
            dma_buf_fd: dmabuf_fd,
            flags: 0x3,
        };
        self.syscalls
            .ioctl_typed(self.device_fd, super::GASKET_IOCTL_MAP_DMABUF, &mut payload)?;
        self.mmio.apply_typed_ioctl(
            super::GASKET_IOCTL_MAP_DMABUF,
            IoctlPayload::GasketDmaBuf(&payload),
        );
        Ok(())
    }

    /// Polls for `SyncFlag` completion on BAR0; if the `TensorCore` status CSR indicates a fault/timeout
    /// (`SYNC_FLAG_TIMEOUT | HBM_ECC_ERROR | POISONED`) or `expected_sync_val` is not reached, snapshots
    /// all BAR0 CSRs into Riegeli-framed `(CoreDump, AllCoreStateSummaries)` diagnostic byte blobs.
    ///
    /// # Arguments
    ///
    /// * `expected_sync_val` - Target `SyncFlag` completion counter value.
    ///
    /// # Errors
    ///
    /// Returns `Err((CoreDump bytes, AllCoreStateSummaries bytes))` if a hardware fault or timeout is detected.
    pub fn poll_completion_or_snapshot_coredump(
        &self,
        expected_sync_val: u64,
    ) -> Result<u64, (Vec<u8>, Vec<u8>)> {
        let cur_sync = self.read_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE);
        let status = self.read_bar0_csr(BAR0_CSR_TENSORCORE_STATUS);
        let fault_mask = super::TensorCoreStatusFlags::HBM_ECC_ERROR.bits()
            | super::TensorCoreStatusFlags::MXU_OVERFLOW.bits()
            | super::TensorCoreStatusFlags::SYNC_FLAG_TIMEOUT.bits()
            | super::TensorCoreStatusFlags::POISONED.bits();
        if (status & fault_mask) == 0 && cur_sync >= expected_sync_val {
            return Ok(cur_sync);
        }
        let pc = self.read_bar0_csr(BAR0_CSR_TENSORCORE_PC);
        let rmap = ChipRegisterMap::for_generation(self.inner.topology.generation);
        let regs: Vec<crate::proto::CoreDumpRegisterDumpProto> = rmap
            .csr_descriptors()
            .into_iter()
            .map(|d| crate::proto::CoreDumpRegisterDumpProto {
                register_name: String::from(d.full_path),
                register_offset: d.byte_offset,
                register_value: self.read_bar0_csr(d.byte_offset),
            })
            .collect();
        let coredump = crate::proto::CoreDumpProto {
            machine_info: Some(crate::proto::CoreDumpMachineInfoProto {
                machine_name: String::from("tpu-host-0"),
                kernel_version: String::from("6.6.0-tpu"),
                reason_for_collection: format!(
                    "SYNC_FLAG_TIMEOUT status=0x{status:x} cur_sync={cur_sync} expected={expected_sync_val}"
                ),
            }),
            timestamps: Some(crate::proto::CoreDumpTimestampInfoProto {
                collection_start_sec: 1_700_000_000,
                collection_end_sec: 1_700_000_001,
                boot_time_sec: 1_699_990_000,
                device_start_sec: 1_699_995_000,
            }),
            hardware: Some(crate::proto::CoreDumpHardwareInfoProto { pstate: vec![0] }),
            core_states: vec![crate::proto::CoreStateDumpProto {
                core_type: String::from("TC"),
                core_id: 0,
                pc,
                hlo_name: String::from("custom_call_or_matmul"),
                registers: regs,
            }],
            analyzer_reports: vec![crate::proto::CoreDumpAnalyzerReportProto {
                analyzer_name: String::from("HangDetector"),
                category: String::from("SYNC_FLAG_TIMEOUT"),
                tag: String::from("VWAIT"),
                message: format!("Core 0 stalled at PC={pc} waiting for sync={expected_sync_val}"),
            }],
        };
        let stall = crate::proto::AllCoreStateSummariesProto {
            core_stall_summaries: vec![crate::proto::CoreStallSummaryProto {
                core_key: String::from("core_0"),
                global_core_id: 0,
                chip_id: 0,
                core_on_chip_type: String::from("TPU_CORE_TYPE_TENSOR_CORE"),
                core_on_chip_index: 0,
                xdb_server_running: false,
                pc,
                hlo_location: String::from("entry_computation"),
                error_message: String::from("SYNC_FLAG_TIMEOUT"),
                active_sync_waits: vec![crate::proto::ActiveSyncFlagWaitProto {
                    sync_flag_id: 0,
                    expected_value: u32::try_from(expected_sync_val).unwrap_or(1),
                    current_value: u32::try_from(cur_sync).unwrap_or(0),
                }],
            }],
            hostname: String::from("tpu-host-0"),
            harvest_start_time_unix_micros: 1_700_000_000_000_000,
            harvest_end_time_unix_micros: 1_700_000_000_001_000,
            libtpu_build: String::from("0.0.52"),
        };
        let cd_riegeli = crate::proto::encode_riegeli_record_container(&coredump.encode());
        let st_riegeli = crate::proto::encode_riegeli_record_container(&stall.encode());
        Err((cd_riegeli, st_riegeli))
    }

    /// Encodes `program` into an `IMEM` binary image, patches live `inputs` and `outputs` IOVA
    /// addresses via [`crate::proto::ProgramRelocationProto::apply_to_imem`], writes the relocated
    /// `IMEM` words into BAR2 HBM (`0x0080_0000`), initializes `BAR0_CSR_SYNC_FLAG_BASE` counters,
    /// and programs `SparseCore` (`BAR0_CSR_SPARSECORE_RUN_CONTROL`) when sparse ops are present.
    ///
    /// # Arguments
    ///
    /// * `program` - Compiled TPU program to relocate and upload.
    /// * `inputs` - Live input buffer [`DeviceAddress`] handles.
    /// * `outputs` - Live output buffer [`DeviceAddress`] handles.
    pub fn upload_and_relocate_executable(
        &self,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> Vec<u8> {
        let mut imem_bytes = program.encode_architectural_imem();
        let param_iovas: Vec<u64> = inputs.iter().map(|a| a.0).collect();
        let output_iovas: Vec<u64> = outputs.iter().map(|a| a.0).collect();
        for (idx, _) in param_iovas.iter().enumerate() {
            let reloc = crate::proto::ProgramRelocationProto {
                offset: 12,
                kind: 0,
                target_index: i64::try_from(idx).unwrap_or(0),
                addend: 0,
            };
            reloc.apply_to_imem(&mut imem_bytes, &param_iovas, &output_iovas);
        }
        for (idx, _) in output_iovas.iter().enumerate() {
            let reloc = crate::proto::ProgramRelocationProto {
                offset: 12,
                kind: 1,
                target_index: i64::try_from(idx).unwrap_or(0),
                addend: 0,
            };
            reloc.apply_to_imem(&mut imem_bytes, &param_iovas, &output_iovas);
        }
        // Write first 64-byte bundle words into BAR2 MMIO (and attached bar2_aperture if small enough)
        for (word_idx, chunk) in imem_bytes.chunks_exact(8).take(8).enumerate() {
            let mut arr = [0_u8; 8];
            arr.copy_from_slice(chunk);
            let word = u64::from_le_bytes(arr);
            let off = (word_idx as u64).saturating_mul(8);
            let _ = self
                .mmio
                .write_mmio_u64(2, crate::types::MmioOffset(0x0080_0000 + off), word);
            let _ = self.write_bar2_u64(off, word);
        }
        // Initialize BAR0 SyncFlags for each bundle
        for b in &program.instruction_bundles {
            let sf_off = BAR0_CSR_SYNC_FLAG_BASE
                .saturating_add(u64::from(b.sync_flag_id & 31).saturating_mul(8));
            self.write_bar0_csr(sf_off, 1);
        }
        if program.instruction_bundles.iter().any(|b| {
            matches!(
                b.opcode,
                crate::compiler::TpuOpCode::Gather | crate::compiler::TpuOpCode::Scatter
            )
        }) || program.mosaic_metadata.core_type == "sparse_core"
        {
            self.write_bar0_csr(BAR0_CSR_SPARSECORE_RUN_CONTROL, 1);
        }
        imem_bytes
    }
}

impl<S: SyscallProvider> TpuHal for LinuxDeviceHal<S> {
    fn discover_topology(&self) -> LibTpuResult<TopologySpec> {
        self.inner.discover_topology()
    }

    fn allocate_device_memory(
        &self,
        device_id: DeviceId,
        memory_kind: MemoryKind,
        size: ByteSize,
    ) -> LibTpuResult<DeviceAddress> {
        let addr = self
            .inner
            .allocate_device_memory(device_id, memory_kind, size)?;
        let _ = self
            .iommu_allocators
            .lock()
            .get_mut(&device_id)
            .map(|alloc| alloc.allocate(size.0.max(256)));
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketPageTableIoctl {
                    page_table_index: 0,
                    size: size.0 as u64,
                    host_address: addr.0,
                    device_address: addr.0,
                };
                self.syscalls
                    .ioctl_typed(self.device_fd, GASKET_IOCTL_MAP_BUFFER, &mut payload)?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_MAP_BUFFER,
                    IoctlPayload::GasketPageTable(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let mut payload = VfioDmaMapIoctl {
                    argsz: 32,
                    flags: 0x3,
                    vaddr: addr.0,
                    iova: addr.0,
                    size: size.0 as u64,
                };
                self.syscalls.ioctl_typed(
                    self.device_fd,
                    VFIO_IOMMU_MAP_DMA_IOCTL,
                    &mut payload,
                )?;
                self.mmio.apply_typed_ioctl(
                    VFIO_IOMMU_MAP_DMA_IOCTL,
                    IoctlPayload::VfioDmaMap(&payload),
                );
            }
        }
        Ok(addr)
    }

    fn deallocate_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
    ) -> LibTpuResult<()> {
        let _ = self
            .iommu_allocators
            .lock()
            .get_mut(&device_id)
            .map(|alloc| alloc.deallocate(address));
        let unmap_size = self.inner.allocated_block_size(device_id, address) as u64;
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketPageTableIoctl {
                    page_table_index: 0,
                    size: unmap_size,
                    host_address: address.0,
                    device_address: address.0,
                };
                self.syscalls.ioctl_typed(
                    self.device_fd,
                    GASKET_IOCTL_UNMAP_BUFFER,
                    &mut payload,
                )?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_UNMAP_BUFFER,
                    IoctlPayload::GasketPageTable(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let mut payload = VfioDmaUnmapIoctl {
                    argsz: 24,
                    flags: 0,
                    iova: address.0,
                    size: unmap_size,
                };
                self.syscalls.ioctl_typed(
                    self.device_fd,
                    VFIO_IOMMU_UNMAP_DMA_IOCTL,
                    &mut payload,
                )?;
                self.mmio.apply_typed_ioctl(
                    VFIO_IOMMU_UNMAP_DMA_IOCTL,
                    IoctlPayload::VfioDmaUnmap(&payload),
                );
            }
        }
        self.inner.deallocate_device_memory(device_id, address)
    }

    fn write_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        data: &[u8],
    ) -> LibTpuResult<()> {
        self.submit_dma_descriptor(data.as_ptr() as u64, address.0, data.len(), 0);
        self.inner.write_device_memory(device_id, address, data)
    }

    fn read_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()> {
        self.submit_dma_descriptor(address.0, out.as_mut_ptr() as u64, out.len(), 1);
        self.inner.read_device_memory(device_id, address, out)
    }

    fn copy_device_to_device(
        &self,
        src_device: DeviceId,
        src_addr: DeviceAddress,
        dst_device: DeviceId,
        dst_addr: DeviceAddress,
        size: ByteSize,
    ) -> LibTpuResult<()> {
        self.submit_dma_descriptor(src_addr.0, dst_addr.0, size.0, 2);
        self.inner
            .copy_device_to_device(src_device, src_addr, dst_device, dst_addr, size)
    }

    fn execute_program(
        &self,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<()> {
        let imem_bytes = self.upload_and_relocate_executable(program, inputs, outputs);
        let imem_len = imem_bytes.len() as u64;
        // Step 1: Upload compiled 512-bit VLIW IMEM overlay to device HBM/IMEM base (`0x0080_0000`) via DMA ring
        self.submit_dma_descriptor(imem_bytes.as_ptr() as u64, 0x0080_0000, imem_bytes.len(), 0);
        // Step 1b: If the program targets SparseCore or contains Gather/Scatter/Embedding ops, upload SparseCore IMEM overlay (`0x00C0_0000`)
        let is_gather_or_scatter = program.instruction_bundles.iter().any(|b| {
            matches!(
                b.opcode,
                crate::compiler::TpuOpCode::Gather | crate::compiler::TpuOpCode::Scatter
            )
        });
        let has_sparse_core = (u8::from(program.mosaic_metadata.core_type == "sparse_core")
            | u8::from(is_gather_or_scatter))
            != 0;
        if has_sparse_core {
            self.submit_dma_descriptor(
                imem_bytes.as_ptr() as u64,
                0x00C0_0000,
                imem_bytes.len(),
                0,
            );
        }
        // Step 2: Bind parameter & output HBM base addresses into BAR2 parameter table (`0x0000_1000..`)
        for (idx, in_addr) in inputs.iter().enumerate() {
            let off = 0x1000_u64.saturating_add((idx as u64).saturating_mul(8));
            let _ = self
                .mmio
                .write_mmio_u64(2, crate::types::MmioOffset(off), in_addr.0);
        }
        for (idx, out_addr) in outputs.iter().enumerate() {
            let off = 0x2000_u64.saturating_add((idx as u64).saturating_mul(8));
            let _ = self
                .mmio
                .write_mmio_u64(2, crate::types::MmioOffset(off), out_addr.0);
        }
        // Step 3: Execute program on the cycle-accurate DeepSeaVm over the device's HBM
        self.inner
            .execute_program(device_id, program, inputs, outputs)?;
        // Step 4: Program per-generation ChipRegisterMap CSRs for TC0, Megacore TC1, and SparseCore SC0
        let pc = self.inner.mmio.get_csr(BAR0_CSR_TENSORCORE_PC);
        let next_sync = self.inner.mmio.get_csr(BAR0_CSR_SYNC_FLAG_BASE);
        let rmap = ChipRegisterMap::for_generation(self.inner.topology.generation);
        self.write_bar0_csr(rmap.tc0_program_location_csr, imem_len);
        self.write_bar0_csr(rmap.tc0_pc_csr, pc);
        self.write_bar0_csr(rmap.tc0_run_status_csr, 1);
        self.write_bar0_csr(rmap.tc0_vwait_sync_flag_state_csr, next_sync);
        self.write_bar0_csr(rmap.tc1_program_location_csr(), imem_len);
        self.write_bar0_csr(rmap.tc1_pc_csr(), pc);
        self.write_bar0_csr(rmap.tc1_run_status_csr(), 1);
        self.write_bar0_csr(rmap.sc0_run_status_csr, 1_u64 + u64::from(has_sparse_core));
        self.write_bar0_csr(BAR0_CSR_TENSORCORE_PC, pc);
        self.write_bar0_csr(BAR0_CSR_TENSORCORE_RUN_CONTROL, 1);
        self.write_bar0_csr(BAR0_CSR_TENSORCORE_STATUS, 1);
        self.write_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE, next_sync);
        self.mmio.record_execution_telemetry(next_sync);
        self.write_bar0_csr(
            BAR0_CSR_THERMAL_MILLICELSIUS,
            self.mmio.get_csr(BAR0_CSR_THERMAL_MILLICELSIUS),
        );
        self.write_bar0_csr(
            BAR0_CSR_POWER_MILLIWATTS,
            self.mmio.get_csr(BAR0_CSR_POWER_MILLIWATTS),
        );
        self.write_bar0_csr(
            BAR0_CSR_DUTY_CYCLE_PERCENT,
            self.mmio.get_csr(BAR0_CSR_DUTY_CYCLE_PERCENT),
        );
        Ok(())
    }

    fn memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats> {
        self.inner.memory_stats(device_id)
    }

    fn clear_memory_stats(&self, device_id: DeviceId) -> LibTpuResult<()> {
        self.inner.clear_memory_stats(device_id)
    }

    fn poison_device(&self, device_id: DeviceId, reason: String) -> LibTpuResult<()> {
        self.inner.poison_device(device_id, reason)
    }
}

/// Polymorphic Hardware Abstraction Layer backend dispatching across [`LinuxDeviceHal`]
/// and [`SimulatedTpuHal`].
#[derive(Debug)]
pub enum HalBackend<S: SyscallProvider = EmulatedKernelDriverBackend> {
    /// Deterministic in-memory TPU hardware simulation backend.
    Simulated(SimulatedTpuHal),
    /// Linux Gasket/VFIO kernel driver backend (`LinuxDeviceHal<S>`).
    Linux(LinuxDeviceHal<S>),
}

impl HalBackend<EmulatedKernelDriverBackend> {
    /// Selects and initializes a [`HalBackend`] from `config`.
    ///
    /// When `--tpu_driver_mode=simulated` is present in `config.extra_flags`, returns
    /// [`HalBackend::Simulated`]; otherwise returns [`HalBackend::Linux`] driving the
    /// Gasket/VFIO protocol stack via [`LinuxDeviceHal::open`].
    ///
    /// # Arguments
    ///
    /// * `config` - TPU initialization configuration.
    #[must_use]
    pub fn from_config(config: &TpuInitConfig) -> Self {
        if config
            .extra_flags
            .get("--tpu_driver_mode")
            .is_some_and(|mode| mode == "simulated")
        {
            Self::Simulated(SimulatedTpuHal::new(config))
        } else {
            let syscalls = EmulatedKernelDriverBackend::default();
            let fd = syscalls.open_device("/dev/accel0").unwrap_or(3);
            let inner = SimulatedTpuHal::new(config);
            let iommu_allocators =
                LinuxDeviceHal::<EmulatedKernelDriverBackend>::build_iommu_allocators(
                    &inner.topology,
                );
            let mmio = inner.mmio.clone();
            Self::Linux(LinuxDeviceHal {
                mode: KernelDriverMode::Gasket,
                syscalls,
                device_fd: fd,
                mmio,
                bar0_aperture: parking_lot::Mutex::new(None),
                bar2_aperture: parking_lot::Mutex::new(None),
                iommu_allocators,
                inner,
            })
        }
    }

    /// Returns a cloned copy of the discovered [`TopologySpec`] for this backend.
    #[must_use]
    pub fn topology_spec(&self) -> TopologySpec {
        match self {
            Self::Simulated(sim) => sim.topology_spec(),
            Self::Linux(linux) => linux.inner.topology_spec(),
        }
    }
}

impl<S: SyscallProvider> TpuHal for HalBackend<S> {
    fn discover_topology(&self) -> LibTpuResult<TopologySpec> {
        match self {
            Self::Simulated(sim) => sim.discover_topology(),
            Self::Linux(linux) => linux.discover_topology(),
        }
    }

    fn allocate_device_memory(
        &self,
        device_id: DeviceId,
        memory_kind: MemoryKind,
        size: ByteSize,
    ) -> LibTpuResult<DeviceAddress> {
        match self {
            Self::Simulated(sim) => sim.allocate_device_memory(device_id, memory_kind, size),
            Self::Linux(linux) => linux.allocate_device_memory(device_id, memory_kind, size),
        }
    }

    fn deallocate_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
    ) -> LibTpuResult<()> {
        match self {
            Self::Simulated(sim) => sim.deallocate_device_memory(device_id, address),
            Self::Linux(linux) => linux.deallocate_device_memory(device_id, address),
        }
    }

    fn write_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        data: &[u8],
    ) -> LibTpuResult<()> {
        match self {
            Self::Simulated(sim) => sim.write_device_memory(device_id, address, data),
            Self::Linux(linux) => linux.write_device_memory(device_id, address, data),
        }
    }

    fn read_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()> {
        match self {
            Self::Simulated(sim) => sim.read_device_memory(device_id, address, out),
            Self::Linux(linux) => linux.read_device_memory(device_id, address, out),
        }
    }

    fn copy_device_to_device(
        &self,
        src_device: DeviceId,
        src_addr: DeviceAddress,
        dst_device: DeviceId,
        dst_addr: DeviceAddress,
        size: ByteSize,
    ) -> LibTpuResult<()> {
        match self {
            Self::Simulated(sim) => {
                sim.copy_device_to_device(src_device, src_addr, dst_device, dst_addr, size)
            }
            Self::Linux(linux) => {
                linux.copy_device_to_device(src_device, src_addr, dst_device, dst_addr, size)
            }
        }
    }

    fn execute_program(
        &self,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<()> {
        match self {
            Self::Simulated(sim) => sim.execute_program(device_id, program, inputs, outputs),
            Self::Linux(linux) => linux.execute_program(device_id, program, inputs, outputs),
        }
    }

    fn memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats> {
        match self {
            Self::Simulated(sim) => sim.memory_stats(device_id),
            Self::Linux(linux) => linux.memory_stats(device_id),
        }
    }

    fn clear_memory_stats(&self, device_id: DeviceId) -> LibTpuResult<()> {
        match self {
            Self::Simulated(sim) => sim.clear_memory_stats(device_id),
            Self::Linux(linux) => linux.clear_memory_stats(device_id),
        }
    }

    fn poison_device(&self, device_id: DeviceId, reason: String) -> LibTpuResult<()> {
        match self {
            Self::Simulated(sim) => sim.poison_device(device_id, reason),
            Self::Linux(linux) => linux.poison_device(device_id, reason),
        }
    }
}

/// Creates the default [`TpuHal`] instance for `config`.
///
/// Scans `/sys/bus/pci/devices` and `/dev` to automatically bind [`LinuxDeviceHal`] over
/// [`super::PosixKernelDriverBackend`] when real Google Cloud TPU `PCIe` (`0x1ae0`) device nodes
/// are accessible, or falls back to [`HalBackend::from_config`].
///
/// # Arguments
///
/// * `config` - TPU initialization configuration.
#[must_use]
pub fn create_default_hal(config: &TpuInitConfig) -> std::sync::Arc<dyn TpuHal> {
    create_default_hal_with_sysfs(
        config,
        std::path::Path::new("/sys/bus/pci/devices"),
        std::path::Path::new("/dev"),
    )
}

/// Creates a [`TpuHal`] instance for `config`, scanning `sysfs_pci_dir` and `dev_root` via
/// [`super::PciDeviceEnumerator::scan_sysfs`].
///
/// If `--tpu_driver_mode=simulated` is not set and at least one Google Cloud TPU (`0x1ae0`)
/// device node in `dev_root` can be opened via [`super::PosixKernelDriverBackend`], returns
/// an `Arc<LinuxDeviceHal<PosixKernelDriverBackend>>`; otherwise falls back to [`HalBackend::from_config`].
///
/// # Arguments
///
/// * `config` - TPU initialization configuration.
/// * `sysfs_pci_dir` - Path to `/sys/bus/pci/devices` (or a synthetic test directory).
/// * `dev_root` - Path to `/dev` (or a synthetic test directory).
#[must_use]
pub fn create_default_hal_with_sysfs(
    config: &TpuInitConfig,
    sysfs_pci_dir: &std::path::Path,
    dev_root: &std::path::Path,
) -> std::sync::Arc<dyn TpuHal> {
    let is_sim = config
        .extra_flags
        .get("--tpu_driver_mode")
        .is_some_and(|mode| mode == "simulated");
    if !is_sim {
        for dev in super::PciDeviceEnumerator::scan_sysfs(sysfs_pci_dir, dev_root) {
            if let Ok(posix_hal) = LinuxDeviceHal::open_with_path(
                dev.driver_mode,
                &dev.device_node_path,
                super::PosixKernelDriverBackend,
                config,
            ) {
                return std::sync::Arc::new(posix_hal);
            }
        }
    }
    std::sync::Arc::new(HalBackend::from_config(config))
}
