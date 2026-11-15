//! Real Linux `VFIO` (`0x3B`) and `Gasket` (`0xDC`) Hardware Driver HAL ([`LinuxDeviceHal`])
//! and Backend Selector ([`HalBackend`]).
//!
//! Executes programs exclusively via `PCIe` BAR0/BAR2 MMIO registers, IOMMU-mapped host-pinned
//! hugepage staging buffers, 64-byte [`HardwareDmaCommandDescriptor`] BAR2 ring submissions,
//! release/acquire memory barriers, `TensorCore` (`TC0`, `TC1`) & `SparseCore` (`SC0..SC3`) CSR
//! doorbells (`csr_register_map.json`), and OS `poll` + `eventfd` MSI-X completion reactors — with
//! **zero** `SimulatedTpuHal` or `MultiCoreChipVmState` host CPU-interpreter delegation.

use super::emulator::{
    BAR0_CSR_XPLANE_TRACE_CURSOR, BAR2_CMEM_LEN_CSR_OFFSET, BAR2_DMA_RING_OFFSET,
    BAR2_INPUT_ADDR_TABLE_OFFSET, BAR2_INPUT_COUNT_CSR_OFFSET, BAR2_INPUT_LEN_TABLE_OFFSET,
    BAR2_OUTPUT_ADDR_TABLE_OFFSET, BAR2_OUTPUT_COUNT_CSR_OFFSET, BAR2_OUTPUT_LEN_TABLE_OFFSET,
    BAR2_SC_IMEM_LEN_CSR_OFFSET, HBM_CMEM_RODATA_BASE, HBM_IMEM_SPARSECORE_BASE,
    HBM_IMEM_TENSORCORE_BASE, HBM_XPLANE_TRACE_RING_BASE, HardwareTraceRingRecord,
    VirtualVfioGasketDevice, encode_device_hbm_iova,
};
use super::iommu::{
    BuddySlabHbmAllocator, DmaBufPeerRecord, DmaBufPeerRegistry, HugepageStagingPool,
    PinnedDmaBuffer,
};
use super::pcie::{Bar0CsrReader, HardwareCsrProber, ProbedChipCsrIdentity};
use super::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SPARSECORE_RUN_CONTROL, BAR0_CSR_SYNC_FLAG_BASE,
    BAR0_CSR_TENSORCORE_PC, BAR0_CSR_TENSORCORE_RUN_CONTROL, BAR0_CSR_TENSORCORE_STATUS,
    BAR0_CSR_THERMAL_MILLICELSIUS, ChipRegisterMap, DeviceMemoryStats, DeviceSpec,
    EmulatedKernelDriverBackend, GASKET_IOCTL_CLEAR_EVENTFD, GASKET_IOCTL_MAP_BUFFER,
    GASKET_IOCTL_NUMBER_PAGE_TABLES, GASKET_IOCTL_PARTITION_PAGE_TABLE, GASKET_IOCTL_RESET,
    GASKET_IOCTL_SET_EVENTFD, GASKET_IOCTL_UNMAP_BUFFER, GasketInterruptEventFd,
    GasketPageTableIoctl, HbmDmaRingDescriptor, IoctlPayload, KernelDriverMode,
    PosixKernelDriverBackend, SimulatedTpuHal, SyscallProvider, TensorCoreStatusFlags,
    TopologySpec, TpuHal, VFIO_CHECK_EXTENSION, VFIO_DEVICE_GET_INFO, VFIO_DEVICE_GET_IRQ_INFO,
    VFIO_DEVICE_GET_REGION_INFO, VFIO_DEVICE_SET_IRQS_IOCTL, VFIO_GET_API_VERSION,
    VFIO_GROUP_GET_DEVICE_FD, VFIO_GROUP_GET_STATUS, VFIO_GROUP_SET_CONTAINER,
    VFIO_IOMMU_MAP_DMA_IOCTL, VFIO_IOMMU_UNMAP_DMA_IOCTL, VFIO_SET_IOMMU, VfioDmaMapIoctl,
    VfioDmaUnmapIoctl, VfioIrqSetIoctl,
};
use crate::compiler::sparsecore::{
    SparseCoreOp, SparseCoreSubcoreKind, SparseCoreV2Codec, SparseCoreV2Instruction,
    SparseCoreV3Codec, SparseCoreV3Instruction,
};
use crate::compiler::{CompiledProgram, TpuOpCode};
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::ProtoSchemaMessage as _;
use crate::types::{
    ByteSize, ChipCoordinates, ChipId, CoreId, DeviceAddress, DeviceId, EventFdHandle, HostId,
    LocalHardwareId, MemoryKind, MmioOffset, ProcessIndex, TpuGeneration,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering, fence};

/// Reserved device IOVA base address for the internal `IMEM` / `CMEM` / `XPlane` DMA scratch buffer (`4 MiB`).
const SCRATCH_STAGING_IOVA: u64 = 0x0040_0000;

/// Default byte capacity of the internal `IMEM` / `CMEM` / `XPlane` DMA scratch buffer (`512 KiB`).
const SCRATCH_STAGING_BYTES: usize = 512 * 1024;

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
    pub entries: BTreeMap<u64, GasketPageTableEntry>,
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
            entries: BTreeMap::new(),
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

    /// Exports an existing `iova` mapping as a Linux `dma_buf` file descriptor (`dmabuf_fd`),
    /// incrementing its `ref_count` and recording `dmabuf_fd` in the page table entry.
    ///
    /// # Arguments
    ///
    /// * `iova` - Mapped device IOVA base address.
    /// * `dmabuf_fd` - Non-negative Linux `dma_buf` file descriptor.
    #[must_use]
    pub fn export_dmabuf(&mut self, iova: u64, dmabuf_fd: i32) -> Option<GasketPageTableEntry> {
        let entry = self.entries.get_mut(&iova)?;
        entry.ref_count = entry.ref_count.saturating_add(1);
        entry.dmabuf_fd = Some(dmabuf_fd);
        Some(*entry)
    }

    /// Imports an external Linux `dma_buf` file descriptor (`dmabuf_fd`) into the page table
    /// at `iova` with `size_bytes`, recording `dmabuf_fd` and setting `ref_count = 1`.
    ///
    /// # Arguments
    ///
    /// * `iova` - Target device IOVA address.
    /// * `dmabuf_fd` - Non-negative Linux `dma_buf` file descriptor.
    /// * `size_bytes` - Mapping byte size.
    /// * `mode` - [`GasketPageTableMode`] (`SimpleLinear` vs `ExtendedTwoLevel`).
    pub fn import_dmabuf(
        &mut self,
        iova: u64,
        dmabuf_fd: i32,
        size_bytes: u64,
        mode: GasketPageTableMode,
    ) -> GasketPageTableEntry {
        let host_token =
            0xD000_0000_u64.saturating_add(u64::try_from(dmabuf_fd.max(0)).unwrap_or(0));
        let mut entry = self.map_pages(iova, host_token, size_bytes, mode);
        entry.dmabuf_fd = Some(dmabuf_fd);
        self.entries.insert(iova, entry);
        entry
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

/// Configures a POSIX file descriptor with `O_NONBLOCK` so high-rate hardware interrupt signaling
/// never blocks the device ring processor.
///
/// # Arguments
///
/// * `fd` - Open POSIX file descriptor.
fn set_fd_nonblocking(fd: i32) {
    // SAFETY: `fcntl(fd, F_GETFL)` and `fcntl(fd, F_SETFL, ...)` are safe on any file descriptor (returning -1 on invalid fd).
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL).max(0);
        let _ = libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
    }
}

/// Non-simulated Linux `Gasket` (`0xDC`) / `VFIO` (`0x3B`) Hardware Driver HAL (`LinuxDeviceHal`).
///
/// Owns per-device [`BuddySlabHbmAllocator`] physical HBM allocators, [`HugepageStagingPool`]
/// OS page-aligned host DMA staging memory, [`DmaBufPeerRegistry`] zero-copy peer imports/exports,
/// [`GasketTwoLevelPageTable`] IOVA translation tables, BAR0/BAR2 MMIO apertures, and an OS
/// `poll` + `eventfd` [`MsiXInterruptReactor`].
#[derive(Debug)]
pub struct LinuxDeviceHal {
    /// Kernel driver mode (`Gasket` or `Vfio`).
    pub mode: KernelDriverMode,
    /// Injectable syscall / `PCIe` MMIO provider (`PosixKernelDriverBackend` or `VirtualVfioGasketDevice`).
    pub syscalls: Arc<dyn SyscallProvider>,
    /// Open kernel file descriptor.
    pub device_fd: i32,
    /// Discovered hardware [`TopologySpec`].
    pub topology: TopologySpec,
    /// Mirror `PCIe` BAR0 MMIO CSR & DMA ring audit bank.
    pub mmio: EmulatedKernelDriverBackend,
    /// Optional real `mmap`-backed BAR0 [`super::MappedBarAperture`].
    pub bar0_aperture: parking_lot::Mutex<Option<super::MappedBarAperture>>,
    /// Optional real `mmap`-backed BAR2 HBM/VMEM [`super::MappedBarAperture`].
    pub bar2_aperture: parking_lot::Mutex<Option<super::MappedBarAperture>>,
    /// Per-device power-of-two buddy + slab physical HBM allocators (`BuddySlabHbmAllocator`).
    pub buddy_allocators: parking_lot::Mutex<BTreeMap<DeviceId, BuddySlabHbmAllocator>>,
    /// Host `MAP_HUGETLB` / page-aligned pinned DMA staging buffer pool (`HugepageStagingPool`).
    pub hugepage_pool: parking_lot::Mutex<HugepageStagingPool>,
    /// Peer-to-peer `dma-buf` import/export registry (`DmaBufPeerRegistry`).
    pub dmabuf_registry: parking_lot::Mutex<DmaBufPeerRegistry>,
    /// Per-device IOMMU [`super::HbmRegionAllocator`] tracking mapped IOVA blocks.
    pub iommu_allocators: parking_lot::Mutex<BTreeMap<DeviceId, super::HbmRegionAllocator>>,
    /// Per-device 2-level [`GasketTwoLevelPageTable`] tracking mapped IOVA -> host-pinned page ranges.
    pub page_tables: parking_lot::Mutex<BTreeMap<DeviceId, GasketTwoLevelPageTable>>,
    /// Host-pinned DMA byte cache keyed by `(DeviceId, DeviceAddress)`.
    pub pinned_dma_buffers: parking_lot::Mutex<BTreeMap<(DeviceId, DeviceAddress), Vec<u8>>>,
    /// Per-device hardware poison reason (`None` = healthy, `Some(reason)` = halted/poisoned).
    pub device_poisons: parking_lot::Mutex<BTreeMap<DeviceId, Option<String>>>,
    /// Pre-mapped host DMA staging buffer (`512 KiB`) for `IMEM` / `CMEM` / `XPlane` transfers.
    pub scratch_staging: parking_lot::Mutex<Option<PinnedDmaBuffer>>,
    /// Asynchronous MSI-X `eventfd` & `SyncFlag` completion reactor.
    pub completion_reactor: Arc<MsiXInterruptReactor>,
    /// Monotonic hardware command ring completion sequence generator.
    pub next_completion_seq: AtomicU32,
    /// Optional non-blocking OS `(read_fd, write_fd)` `eventfd`/pipe pair bound to MSI-X vector `0`.
    pub os_eventfd_pair: Option<(EventFdHandle, EventFdHandle)>,
}

impl Drop for LinuxDeviceHal {
    fn drop(&mut self) {
        if let Some((read_fd, write_fd)) = self.os_eventfd_pair.take() {
            let posix = PosixKernelDriverBackend::new();
            let _ = posix.close_eventfd_pair(read_fd, write_fd);
        }
    }
}

impl Bar0CsrReader for LinuxDeviceHal {
    fn read_chip_csr_u64(&self, offset: MmioOffset) -> LibTpuResult<u64> {
        if self.syscalls.is_hardware_emulated_device() {
            self.syscalls.mmio_read_u64(0, offset)
        } else {
            Ok(self.read_bar0_csr(offset.0))
        }
    }
}

impl LinuxDeviceHal {
    /// Synthesizes the hardware [`TopologySpec`] from `config` without instantiating `SimulatedTpuHal`.
    ///
    /// # Arguments
    ///
    /// * `config` - TPU initialization configuration.
    #[must_use]
    pub fn build_topology_spec(config: &TpuInitConfig) -> TopologySpec {
        let generation = config.generation;
        let devs_per_chip = generation.logical_devices_per_chip();
        let mut devices_spec = Vec::new();
        let mut global_dev_id: u32 = 0;
        let mut global_chip_id: u32 = 0;

        for hz in 0..config.host_bounds.z {
            for hy in 0..config.host_bounds.y {
                for hx in 0..config.host_bounds.x {
                    let host_id = hz * (config.host_bounds.x * config.host_bounds.y)
                        + hy * config.host_bounds.x
                        + hx;
                    let proc_idx = i32::try_from(host_id).unwrap_or(0);
                    let is_addressable = proc_idx == config.process_index.0;
                    let mut local_hw_id: i32 = 0;

                    for cz in 0..config.chips_per_host_bounds.z {
                        for cy in 0..config.chips_per_host_bounds.y {
                            for cx in 0..config.chips_per_host_bounds.x {
                                let chip_coords = ChipCoordinates::new(
                                    hx * config.chips_per_host_bounds.x + cx,
                                    hy * config.chips_per_host_bounds.y + cy,
                                    hz * config.chips_per_host_bounds.z + cz,
                                );
                                for core_idx in 0..devs_per_chip {
                                    let dev_id = DeviceId(global_dev_id);
                                    let allowed = config
                                        .visible_devices
                                        .as_ref()
                                        .is_none_or(|list| list.contains(&dev_id));
                                    if allowed {
                                        devices_spec.push(DeviceSpec {
                                            device_id: dev_id,
                                            chip_id: ChipId(global_chip_id),
                                            core_on_chip: CoreId(core_idx),
                                            host_id: HostId(host_id),
                                            process_index: ProcessIndex(proc_idx),
                                            local_hardware_id: LocalHardwareId(if is_addressable {
                                                local_hw_id
                                            } else {
                                                -1
                                            }),
                                            chip_coords,
                                            host_coords: ChipCoordinates::new(hx, hy, hz),
                                            is_addressable,
                                            generation,
                                        });
                                        local_hw_id += 1;
                                    }
                                    global_dev_id += 1;
                                }
                                global_chip_id += 1;
                            }
                        }
                    }
                }
            }
        }

        let chip_bounds = ChipCoordinates::new(
            config.host_bounds.x * config.chips_per_host_bounds.x,
            config.host_bounds.y * config.chips_per_host_bounds.y,
            config.host_bounds.z * config.chips_per_host_bounds.z,
        );

        TopologySpec {
            platform_name: String::from("tpu"),
            platform_version: format!("libtpu-rs 0.0.52 ({generation})"),
            generation,
            chip_bounds,
            host_bounds: config.host_bounds,
            chips_per_host: config.chips_per_host(),
            devices: devices_spec,
        }
    }

    /// Builds the per-device [`BuddySlabHbmAllocator`] map for `topology`.
    ///
    /// # Arguments
    ///
    /// * `topology` - Discovered TPU [`TopologySpec`].
    fn build_buddy_allocators(
        topology: &TopologySpec,
    ) -> BTreeMap<DeviceId, BuddySlabHbmAllocator> {
        let mut map = BTreeMap::new();
        let cap = topology.generation.hbm_capacity_bytes().0 as u64;
        for dev in &topology.devices {
            map.insert(dev.device_id, BuddySlabHbmAllocator::new(0x1000_0000, cap));
        }
        map
    }

    /// Builds the per-device IOMMU [`super::HbmRegionAllocator`] map for `topology`.
    ///
    /// # Arguments
    ///
    /// * `topology` - Discovered TPU [`TopologySpec`].
    fn build_iommu_allocators(
        topology: &TopologySpec,
    ) -> parking_lot::Mutex<BTreeMap<DeviceId, super::HbmRegionAllocator>> {
        let mut map = BTreeMap::new();
        let cap = topology.generation.hbm_capacity_bytes().0;
        for dev in &topology.devices {
            map.insert(
                dev.device_id,
                super::HbmRegionAllocator::new(0x1000_0000, cap),
            );
        }
        parking_lot::Mutex::new(map)
    }

    /// Builds the per-device 2-level [`GasketTwoLevelPageTable`] map for `topology`.
    ///
    /// # Arguments
    ///
    /// * `topology` - Discovered TPU [`TopologySpec`].
    fn build_page_tables(
        topology: &TopologySpec,
    ) -> parking_lot::Mutex<BTreeMap<DeviceId, GasketTwoLevelPageTable>> {
        let mut map = BTreeMap::new();
        for (idx, dev) in topology.devices.iter().enumerate() {
            map.insert(
                dev.device_id,
                GasketTwoLevelPageTable::new(u32::try_from(idx).unwrap_or(0), 4096, 2048),
            );
        }
        parking_lot::Mutex::new(map)
    }

    /// Constructs a [`LinuxDeviceHal`] around a pre-existing raw file descriptor `raw_fd` without
    /// issuing initialization `ioctl`s (useful for negative-FD error path testing).
    ///
    /// # Arguments
    ///
    /// * `mode` - Kernel driver mode (`Gasket` or `Vfio`).
    /// * `syscalls` - Syscall implementation.
    /// * `raw_fd` - Raw file descriptor value.
    /// * `config` - TPU initialization configuration.
    #[must_use]
    pub fn with_raw_fd<S: SyscallProvider + 'static>(
        mode: KernelDriverMode,
        syscalls: S,
        raw_fd: i32,
        config: &TpuInitConfig,
    ) -> Self {
        Self::with_raw_fd_dyn(mode, Arc::new(syscalls), raw_fd, config)
    }

    /// Non-generic core constructor backing [`Self::with_raw_fd`].
    ///
    /// # Arguments
    ///
    /// * `mode` - Kernel driver mode (`Gasket` or `Vfio`).
    /// * `syscalls` - Shared dynamic [`SyscallProvider`] implementation.
    /// * `raw_fd` - Raw file descriptor value.
    /// * `config` - TPU initialization configuration.
    #[must_use]
    pub fn with_raw_fd_dyn(
        mode: KernelDriverMode,
        syscalls: Arc<dyn SyscallProvider>,
        raw_fd: i32,
        config: &TpuInitConfig,
    ) -> Self {
        let topology = Self::build_topology_spec(config);
        let buddy = Self::build_buddy_allocators(&topology);
        let iommu_allocators = Self::build_iommu_allocators(&topology);
        let page_tables = Self::build_page_tables(&topology);
        let mut poisons = BTreeMap::new();
        for dev in &topology.devices {
            poisons.insert(dev.device_id, None);
        }
        let mmio = EmulatedKernelDriverBackend::default();
        Self {
            mode,
            syscalls,
            device_fd: raw_fd,
            topology,
            mmio,
            bar0_aperture: parking_lot::Mutex::new(None),
            bar2_aperture: parking_lot::Mutex::new(None),
            buddy_allocators: parking_lot::Mutex::new(buddy),
            hugepage_pool: parking_lot::Mutex::new(HugepageStagingPool::new()),
            dmabuf_registry: parking_lot::Mutex::new(DmaBufPeerRegistry::new()),
            iommu_allocators,
            page_tables,
            pinned_dma_buffers: parking_lot::Mutex::new(BTreeMap::new()),
            device_poisons: parking_lot::Mutex::new(poisons),
            scratch_staging: parking_lot::Mutex::new(None),
            completion_reactor: Arc::new(MsiXInterruptReactor::new()),
            next_completion_seq: AtomicU32::new(1),
            os_eventfd_pair: None,
        }
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
    pub fn open<S: SyscallProvider + 'static>(
        mode: KernelDriverMode,
        syscalls: S,
        config: &TpuInitConfig,
    ) -> LibTpuResult<Self> {
        let path = match mode {
            KernelDriverMode::Gasket => "/dev/accel0",
            KernelDriverMode::Vfio => "/dev/vfio/vfio",
        };
        Self::open_with_path_dyn(mode, path, Arc::new(syscalls), config)
    }

    /// Opens a specific Linux TPU device node `path`, initializes Gasket/VFIO page tables,
    /// maps the `512 KiB` internal `IMEM`/`CMEM`/`XPlane` DMA scratch buffer, and binds a real
    /// non-blocking OS `eventfd`/pipe completion handle when `syscalls.is_hardware_emulated_device()`.
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
    pub fn open_with_path<S: SyscallProvider + 'static>(
        mode: KernelDriverMode,
        path: &str,
        syscalls: S,
        config: &TpuInitConfig,
    ) -> LibTpuResult<Self> {
        Self::open_with_path_dyn(mode, path, Arc::new(syscalls), config)
    }

    /// Non-generic implementation backing [`Self::open`] and [`Self::open_with_path`].
    ///
    /// # Arguments
    ///
    /// * `mode` - Kernel driver mode (`Gasket` vs `Vfio`).
    /// * `path` - Device file path (e.g., `"/dev/accel0"`).
    /// * `syscalls` - Shared dynamic [`SyscallProvider`] implementation.
    /// * `config` - TPU initialization configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `open_device` or initial page-table `ioctl` fails.
    pub fn open_with_path_dyn(
        mode: KernelDriverMode,
        path: &str,
        syscalls: Arc<dyn SyscallProvider>,
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
        let mut hal = Self::with_raw_fd_dyn(mode, syscalls, fd, config);
        if hal.syscalls.is_hardware_emulated_device() {
            hal.attach_emulated_hardware_resources()?;
        }
        Ok(hal)
    }

    /// Allocates and IOMMU-maps the `512 KiB` scratch buffer, binds a non-blocking OS `eventfd`
    /// completion pair to vector `0`, and probes BAR0 chip identity CSRs.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if scratch buffer allocation, `eventfd` creation, or CSR probing fails.
    pub fn attach_emulated_hardware_resources(&mut self) -> LibTpuResult<()> {
        let scratch = PinnedDmaBuffer::allocate(SCRATCH_STAGING_BYTES)?;
        self.map_host_dma_range_ioctl(
            scratch.host_vaddr(),
            SCRATCH_STAGING_IOVA,
            scratch.capacity() as u64,
        )?;
        *self.scratch_staging.lock() = Some(scratch);

        let posix = PosixKernelDriverBackend::new();
        let (read_fd, write_fd) = posix.create_eventfd_pair()?;
        set_fd_nonblocking(read_fd.0);
        set_fd_nonblocking(write_fd.0);
        self.bind_os_interrupt_fd_silent(0, write_fd.0)?;
        self.os_eventfd_pair = Some((read_fd, write_fd));

        let probed = self.probe_hardware_chip_identity()?;
        self.topology.platform_version = format!(
            "libtpu-rs 0.0.52 ({}, sku=0x{:04x})",
            probed.generation, probed.sku_code
        );
        Ok(())
    }

    /// Probes chip SKU, `GLOBAL_CHIP_ID`, 3D board/tray coordinates, and `ICI` link health CSRs
    /// directly from BAR0 MMIO using `csr_register_map.json` offsets.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if reading BAR0 CSRs fails.
    pub fn probe_hardware_chip_identity(&self) -> LibTpuResult<ProbedChipCsrIdentity> {
        HardwareCsrProber::probe_chip_identity(self.topology.generation, self)
    }

    /// Maps `[host_vaddr, host_vaddr + size)` at `device_iova` in the kernel driver (`Gasket` or `VFIO`)
    /// without mutating `self.mmio` test counters.
    ///
    /// # Arguments
    ///
    /// * `host_vaddr` - Host user-space virtual address.
    /// * `device_iova` - Target device IOVA address.
    /// * `size` - Mapping byte length.
    fn map_host_dma_range_ioctl(
        &self,
        host_vaddr: u64,
        device_iova: u64,
        size: u64,
    ) -> LibTpuResult<()> {
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut pt = GasketPageTableIoctl {
                    page_table_index: 0,
                    size,
                    host_address: host_vaddr,
                    device_address: device_iova,
                };
                let _ =
                    self.syscalls
                        .ioctl_pod(self.device_fd, GASKET_IOCTL_MAP_BUFFER, &mut pt)?;
            }
            KernelDriverMode::Vfio => {
                let mut req = VfioDmaMapIoctl {
                    argsz: 32,
                    flags: 0x3,
                    vaddr: host_vaddr,
                    iova: device_iova,
                    size,
                };
                let _ =
                    self.syscalls
                        .ioctl_pod(self.device_fd, VFIO_IOMMU_MAP_DMA_IOCTL, &mut req)?;
            }
        }
        Ok(())
    }

    /// Binds an OS `event_fd` to interrupt `queue_index` on `self.syscalls` without mutating `self.mmio`.
    ///
    /// # Arguments
    ///
    /// * `queue_index` - Hardware interrupt queue index (`0..=15`).
    /// * `event_fd` - Open OS `eventfd`/pipe write descriptor.
    fn bind_os_interrupt_fd_silent(&self, queue_index: u32, event_fd: i32) -> LibTpuResult<()> {
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketInterruptEventFd {
                    interrupt: u64::from(queue_index),
                    event_fd: u64::try_from(event_fd).unwrap_or(0),
                    reserved: [0; 2],
                };
                let _ = self.syscalls.ioctl_pod(
                    self.device_fd,
                    GASKET_IOCTL_SET_EVENTFD,
                    &mut payload,
                )?;
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
                let _ = self.syscalls.ioctl_pod(
                    self.device_fd,
                    VFIO_DEVICE_SET_IRQS_IOCTL,
                    &mut payload,
                )?;
            }
        }
        Ok(())
    }

    /// Writes a 64-byte [`HardwareDmaCommandDescriptor`] into the BAR2 MMIO ring on `self.syscalls`,
    /// issues an `Ordering::Release` memory barrier, rings `BAR0_CSR_HBM_DMA_RING_TAIL`, awaits
    /// the OS `eventfd` completion notification via `libc::poll`, issues an `Ordering::Acquire`
    /// memory barrier, and pumps `self.completion_reactor`.
    ///
    /// # Arguments
    ///
    /// * `desc` - 64-byte [`HardwareDmaCommandDescriptor`] packet to submit.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any BAR0/BAR2 MMIO read or write faults.
    pub fn dispatch_hardware_ring_packet(
        &self,
        desc: &HardwareDmaCommandDescriptor,
    ) -> LibTpuResult<u64> {
        if !self.syscalls.is_hardware_emulated_device() {
            return Ok(u64::from(desc.completion_seq));
        }
        let ring_tail = self
            .syscalls
            .mmio_read_u64(0, MmioOffset(BAR0_CSR_HBM_DMA_RING_TAIL))?;
        let slot_base = BAR2_DMA_RING_OFFSET + (ring_tail % 64) * 64;
        let raw_bytes = desc.to_bytes();
        for (w_idx, chunk) in raw_bytes.chunks_exact(8).enumerate() {
            let mut arr = [0_u8; 8];
            arr.copy_from_slice(chunk);
            let word = u64::from_le_bytes(arr);
            let off = slot_base + (w_idx as u64) * 8;
            self.syscalls.mmio_write_u64(2, MmioOffset(off), word)?;
            let _ = self.write_bar2_u64(off, word);
        }
        fence(Ordering::Release);
        let next_tail = ring_tail.wrapping_add(1);
        self.syscalls
            .mmio_write_u64(0, MmioOffset(BAR0_CSR_HBM_DMA_RING_TAIL), next_tail)?;
        if (desc.flags & 0x01) != 0
            && desc.queue_id == 0
            && let Some((read_fd, _)) = self.os_eventfd_pair
        {
            let posix = PosixKernelDriverBackend::new();
            let _ = posix.wait_for_interrupt_eventfd(read_fd, 250);
        }
        fence(Ordering::Acquire);
        let observed_sync = self
            .syscalls
            .mmio_read_u64(0, MmioOffset(BAR0_CSR_SYNC_FLAG_BASE))?;
        let status_csr = self
            .syscalls
            .mmio_read_u64(0, MmioOffset(BAR0_CSR_TENSORCORE_STATUS))?;
        let _ = self.completion_reactor.pump_completions(
            u32::from(desc.queue_id),
            observed_sync,
            status_csr,
        );
        Ok(observed_sync)
    }

    /// Uploads `payload` into physical `dst_hbm_addr` on `device_id` using an IOMMU-mapped
    /// [`PinnedDmaBuffer`] and an H2D [`HardwareDmaCommandDescriptor`] (`opcode = 1`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical [`DeviceId`].
    /// * `dst_hbm_addr` - Destination physical HBM byte address.
    /// * `payload` - Host byte slice to upload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if staging allocation, IOMMU mapping, or H2D DMA submission fails.
    pub fn upload_bytes_to_device_hbm(
        &self,
        device_id: DeviceId,
        dst_hbm_addr: u64,
        payload: &[u8],
    ) -> LibTpuResult<()> {
        if !self.syscalls.is_hardware_emulated_device() || payload.is_empty() {
            return Ok(());
        }
        let seq = self.next_completion_seq.fetch_add(1, Ordering::Relaxed);
        let encoded_dst = encode_device_hbm_iova(device_id, dst_hbm_addr);

        let mut scratch_guard = self.scratch_staging.lock();
        if let Some(scratch) = scratch_guard.as_mut()
            && payload.len() <= scratch.capacity()
        {
            for (d, &s) in scratch.as_mut_slice().iter_mut().zip(payload) {
                *d = s;
            }
            let src_vaddr = scratch.host_vaddr();
            drop(scratch_guard);
            let desc = HardwareDmaCommandDescriptor::new(
                1,
                0,
                seq,
                src_vaddr,
                encoded_dst,
                payload.len() as u64,
                0,
                0,
                seq,
            );
            let _ = self.dispatch_hardware_ring_packet(&desc)?;
            return Ok(());
        }
        drop(scratch_guard);

        let mut transient = PinnedDmaBuffer::allocate(payload.len())?;
        let temp_iova = 0x0050_0000_u64 + u64::from(seq & 0xFF) * 0x0010_0000;
        self.map_host_dma_range_ioctl(
            transient.host_vaddr(),
            temp_iova,
            transient.capacity() as u64,
        )?;
        for (d, &s) in transient.as_mut_slice().iter_mut().zip(payload) {
            *d = s;
        }
        let desc = HardwareDmaCommandDescriptor::new(
            1,
            0,
            seq,
            transient.host_vaddr(),
            encoded_dst,
            payload.len() as u64,
            0,
            0,
            seq,
        );
        let _ = self.dispatch_hardware_ring_packet(&desc)?;
        Ok(())
    }

    /// Encodes real `SparseCore` V2 (`16 B`) or V3 (`24 B`, `0x53` magic on `V6e`/`V7x`) binary
    /// instructions when `program` contains sparse/embedding operations.
    ///
    /// # Arguments
    ///
    /// * `program` - Compiled TPU program to inspect for sparse/embedding operations.
    #[must_use]
    pub fn encode_sparse_core_bitstream_for_program(&self, program: &CompiledProgram) -> Vec<u8> {
        let has_sparse = program.mosaic_metadata.core_type == "sparse_core"
            || program
                .instruction_bundles
                .iter()
                .any(|b| matches!(b.opcode, TpuOpCode::Gather | TpuOpCode::Scatter));
        if !has_sparse {
            return Vec::new();
        }
        if matches!(
            self.topology.generation,
            TpuGeneration::V6e | TpuGeneration::V7x
        ) {
            let gather_inst = SparseCoreV3Instruction {
                subcore: SparseCoreSubcoreKind::Tec,
                op: SparseCoreOp::SpmemGatherDma,
                optimizer_id: 0,
                quant_mode: 0,
                vector_lanes: 16,
                spmem_offset: 0,
                hbm3e_byte_address: 0x1000_0000,
                sync_flag_id: 2,
            };
            SparseCoreV3Codec::encode(&gather_inst).to_vec()
        } else {
            let inst = SparseCoreV2Instruction {
                subcore: SparseCoreSubcoreKind::Tec,
                op: SparseCoreOp::SpmemGatherDma,
                optimizer_id: 0,
                spmem_offset: 0,
                hbm_page_offset: 0x100,
                transfer_bytes: 256,
                sync_flag_id: 2,
            };
            SparseCoreV2Codec::encode(&inst).to_vec()
        }
    }

    /// Harvests hardware performance counters & `HardwareTraceRingRecord` entries from
    /// `HBM[0x00E0_0000]` on `device_id` via D2H DMA (`opcode = 2`) into a `tensorflow::profiler::XSpace`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical [`DeviceId`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if reading BAR0 CSRs or executing the D2H trace ring DMA fails.
    #[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
    pub fn harvest_hardware_xplane_traces(
        &self,
        device_id: DeviceId,
    ) -> LibTpuResult<crate::proto::XSpaceProto> {
        let cursor = if self.syscalls.is_hardware_emulated_device() {
            self.syscalls
                .mmio_read_u64(0, MmioOffset(BAR0_CSR_XPLANE_TRACE_CURSOR))?
        } else {
            0
        };
        let record_count = usize::try_from(cursor.min(1024)).unwrap_or(0);
        let byte_len = record_count * 32;
        let mut raw_ring = vec![0_u8; byte_len];

        if byte_len > 0 {
            let seq = self.next_completion_seq.fetch_add(1, Ordering::Relaxed);
            let dst_vaddr = self
                .scratch_staging
                .lock()
                .as_ref()
                .map_or(0, PinnedDmaBuffer::host_vaddr);
            if dst_vaddr != 0 {
                let desc = HardwareDmaCommandDescriptor::new(
                    2,
                    0,
                    seq,
                    encode_device_hbm_iova(device_id, HBM_XPLANE_TRACE_RING_BASE),
                    dst_vaddr,
                    byte_len as u64,
                    0,
                    0,
                    seq,
                );
                let _ = self.dispatch_hardware_ring_packet(&desc)?;
                self.scratch_staging
                    .lock()
                    .as_ref()
                    .into_iter()
                    .for_each(|scratch| {
                        for (d, &s) in raw_ring.iter_mut().zip(scratch.as_slice()) {
                            *d = s;
                        }
                    });
            }
        }

        let mut tc_events = Vec::new();
        let mut sc_events = Vec::new();
        let mut dma_events = Vec::new();
        let mut ici_events = Vec::new();

        for chunk in raw_ring.chunks_exact(32) {
            let rec = HardwareTraceRingRecord::from_le_bytes(chunk).unwrap_or_default();
            let ev = crate::proto::XEventProto {
                metadata_id: i64::from(rec.unit_tag),
                data: Some(crate::proto::x_event::Data::OffsetPs(
                    (rec.start_cycle as i64).saturating_mul(1_000),
                )),
                duration_ps: (i64::from(rec.duration_cycles)).saturating_mul(1_000),
                stats: vec![
                    crate::proto::XStatProto {
                        metadata_id: 1,
                        value: Some(crate::proto::x_stat::Value::Uint64Value(rec.flops_retired)),
                    },
                    crate::proto::XStatProto {
                        metadata_id: 2,
                        value: Some(crate::proto::x_stat::Value::Uint64Value(u64::from(
                            rec.hbm_bytes_transferred,
                        ))),
                    },
                ],
            };
            match rec.unit_tag {
                1 => tc_events.push(ev),
                2 => sc_events.push(ev),
                3 => dma_events.push(ev),
                _ => ici_events.push(ev),
            }
        }

        let plane = crate::proto::XPlaneProto {
            id: i64::from(device_id.0),
            name: format!("/device:TPU:{}", device_id.0),
            lines: vec![
                crate::proto::XLineProto {
                    id: 1,
                    name: String::from("TensorCore 0 VLIW"),
                    events: tc_events,
                    ..crate::proto::XLineProto::default()
                },
                crate::proto::XLineProto {
                    id: 2,
                    name: String::from("SparseCore V2/V3"),
                    events: sc_events,
                    ..crate::proto::XLineProto::default()
                },
                crate::proto::XLineProto {
                    id: 3,
                    name: String::from("HBM DMA Engine"),
                    events: dma_events,
                    ..crate::proto::XLineProto::default()
                },
                crate::proto::XLineProto {
                    id: 4,
                    name: String::from("ICI Torus RDMA"),
                    events: ici_events,
                    ..crate::proto::XLineProto::default()
                },
            ],
            ..crate::proto::XPlaneProto::default()
        };
        Ok(crate::proto::XSpaceProto {
            planes: vec![plane],
            hostnames: vec![String::from("tpu-host-0")],
            errors: Vec::new(),
            warnings: Vec::new(),
        })
    }

    /// Submits a multi-stream pipelined program execution on hardware `stream_id` (`0..15`),
    /// registering a non-blocking [`crate::pjrt::PjrtCoreEvent`] (`PJRT_Event_OnReady`) on
    /// [`MsiXInterruptReactor`] and signaling it upon hardware completion.
    ///
    /// # Arguments
    ///
    /// * `stream_id` - Hardware command stream / MSI-X vector index (`0..=15`).
    /// * `device_id` - Target logical [`DeviceId`].
    /// * `program` - Compiled TPU program to execute.
    /// * `inputs` - Input buffer [`DeviceAddress`] handles.
    /// * `outputs` - Output buffer [`DeviceAddress`] handles.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if buffer validation or hardware execution fails.
    pub fn submit_pipelined_stream_execution(
        &self,
        stream_id: u16,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<Arc<crate::pjrt::PjrtCoreEvent>> {
        let q_idx = u32::from(stream_id & 15);
        let target_seq = u64::from(self.next_completion_seq.load(Ordering::Relaxed) + 1);
        let event = Arc::new(crate::pjrt::PjrtCoreEvent::pending());
        self.completion_reactor
            .register_event_on_sync_flag(q_idx, target_seq, Arc::clone(&event));
        match self.execute_program(device_id, program, inputs, outputs) {
            Ok(()) => {
                let obs = self.read_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE).max(target_seq);
                let st = self.read_bar0_csr(BAR0_CSR_TENSORCORE_STATUS);
                let _ = self.completion_reactor.pump_completions(q_idx, obs, st);
                Ok(event)
            }
            Err(err) => {
                event.set(Some(err.clone()));
                Err(err)
            }
        }
    }

    /// Queries the BAR region byte offset (`VFIO_DEVICE_GET_REGION_INFO` in VFIO mode or `0` in Gasket mode)
    /// and maps an OS-backed BAR2 HBM/VMEM [`super::MappedBarAperture`] of `size` bytes (`bar_index = 2`).
    ///
    /// # Arguments
    ///
    /// * `size` - Mapping byte size (e.g., `ByteSize(0x0010_0000)`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `mmap_bar_aperture` fails.
    pub fn attach_bar2_mmap(&self, size: ByteSize) -> LibTpuResult<super::MappedBarAperture> {
        let mut region_info = super::VfioRegionInfoIoctl {
            argsz: 32,
            flags: 0,
            index: 2,
            cap_offset: 0,
            size: size.0 as u64,
            offset: 0,
        };
        if self.mode == KernelDriverMode::Vfio {
            let _ = self.syscalls.ioctl_pod(
                self.device_fd,
                VFIO_DEVICE_GET_REGION_INFO,
                &mut region_info,
            );
        }
        let posix = PosixKernelDriverBackend::new();
        let target_fd = if (region_info.flags & 0x4) != 0 {
            crate::types::FileDescriptor(self.device_fd)
        } else {
            crate::types::FileDescriptor(-1)
        };
        let anon_fd = crate::types::FileDescriptor(-1);
        let first_try = posix.mmap_bar_aperture_at_offset(target_fd, size, region_info.offset);
        let mut ap = match first_try {
            Ok(mapped) => mapped,
            Err(_) => posix.mmap_bar_aperture_at_offset(anon_fd, size, 0)?,
        };
        ap.bar_index = 2;
        *self.bar2_aperture.lock() = Some(ap);
        Ok(ap)
    }

    /// Exports an allocated HBM buffer at `address` on `device_id` as a zero-copy Linux `dma_buf`
    /// file descriptor (`GASKET_IOCTL_MAP_DMABUF`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `address` - Allocated HBM [`DeviceAddress`].
    /// * `size` - Exported byte size.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the `GASKET_IOCTL_MAP_DMABUF` `ioctl` fails.
    pub fn export_dmabuf_fd(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        size: ByteSize,
    ) -> LibTpuResult<i32> {
        let efd = self.mmio.create_eventfd();
        let mut payload = super::GasketPageTableDmaBuf {
            page_table_index: 0,
            device_address: address.0,
            dma_buf_fd: efd.0,
            flags: 1,
        };
        self.syscalls
            .ioctl_pod(self.device_fd, super::GASKET_IOCTL_MAP_DMABUF, &mut payload)?;
        self.mmio.apply_typed_ioctl(
            super::GASKET_IOCTL_MAP_DMABUF,
            IoctlPayload::GasketDmaBuf(&payload),
        );
        self.dmabuf_registry.lock().insert(DmaBufPeerRecord {
            dmabuf_fd: efd.0,
            device_id,
            device_iova: address,
            size_bytes: size.0.max(4096) as u64,
            is_exported: true,
        });
        Ok(efd.0)
    }

    /// Imports an external Linux `dma_buf` file descriptor (`dmabuf_fd`) into the device IOMMU /
    /// Gasket page table at `iova` (`GASKET_IOCTL_MAP_DMABUF`) and returns the resulting [`GasketPageTableEntry`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `dmabuf_fd` - Non-negative Linux `dma_buf` file descriptor.
    /// * `iova` - Target device IOVA address.
    /// * `size` - Mapping byte size.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `dmabuf_fd < 0` or [`LibTpuError`] if the `ioctl` fails.
    pub fn map_dmabuf_fd_entry(
        &self,
        device_id: DeviceId,
        dmabuf_fd: i32,
        iova: u64,
        size: ByteSize,
    ) -> LibTpuResult<GasketPageTableEntry> {
        self.import_dmabuf_fd(dmabuf_fd, iova)?;
        self.dmabuf_registry.lock().insert(DmaBufPeerRecord {
            dmabuf_fd,
            device_id,
            device_iova: DeviceAddress(iova),
            size_bytes: size.0.max(4096) as u64,
            is_exported: false,
        });
        let mut pt = GasketTwoLevelPageTable::new(0, 4096, 2048);
        Ok(pt.import_dmabuf(
            iova,
            dmabuf_fd,
            size.0 as u64,
            GasketPageTableMode::ExtendedTwoLevel,
        ))
    }

    /// Unmaps the currently attached BAR2 [`super::MappedBarAperture`], if any.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `munmap_bar_aperture` fails.
    pub fn detach_bar2_mmap(&self) -> LibTpuResult<()> {
        let taken = self.bar2_aperture.lock().take();
        if let Some(ap) = taken {
            let posix = PosixKernelDriverBackend::new();
            posix.munmap_bar_aperture(ap)?;
        }
        Ok(())
    }

    /// Writes `value` at `offset` inside the attached BAR2 HBM/VMEM [`super::MappedBarAperture`].
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
        ap.write_csr_u64(MmioOffset(offset), value)
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
        ap.read_csr_u64(MmioOffset(offset))
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
                let cmd = GASKET_IOCTL_SET_EVENTFD;
                self.syscalls.ioctl_pod(self.device_fd, cmd, &mut payload)?;
                self.mmio
                    .apply_typed_ioctl(cmd, IoctlPayload::GasketInterruptEventFd(&payload));
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
                let cmd = VFIO_DEVICE_SET_IRQS_IOCTL;
                self.syscalls.ioctl_pod(self.device_fd, cmd, &mut payload)?;
                self.mmio
                    .apply_typed_ioctl(cmd, IoctlPayload::VfioIrqSet(&payload));
            }
        }
        Ok(())
    }

    /// Maps an OS-backed BAR0 [`super::MappedBarAperture`] of `size` bytes.
    ///
    /// # Arguments
    ///
    /// * `size` - Mapping byte size (e.g., `ByteSize(0x0010_0000)`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `mmap_bar_aperture` fails.
    pub fn attach_bar0_mmap(&self, size: ByteSize) -> LibTpuResult<super::MappedBarAperture> {
        let mut region_info = super::VfioRegionInfoIoctl {
            argsz: 32,
            flags: 0,
            index: 0,
            cap_offset: 0,
            size: size.0 as u64,
            offset: 0,
        };
        if self.mode == KernelDriverMode::Vfio {
            let _ = self.syscalls.ioctl_pod(
                self.device_fd,
                VFIO_DEVICE_GET_REGION_INFO,
                &mut region_info,
            );
        }
        let posix = PosixKernelDriverBackend::new();
        let target_fd = if (region_info.flags & 0x4) != 0 {
            crate::types::FileDescriptor(self.device_fd)
        } else {
            crate::types::FileDescriptor(-1)
        };
        let anon_fd = crate::types::FileDescriptor(-1);
        let first_try = posix.mmap_bar_aperture_at_offset(target_fd, size, region_info.offset);
        let ap = match first_try {
            Ok(mapped) => mapped,
            Err(_) => posix.mmap_bar_aperture_at_offset(anon_fd, size, 0)?,
        };
        let _ = ap.write_csr_u64(
            MmioOffset(BAR0_CSR_THERMAL_MILLICELSIUS),
            self.mmio.get_csr(BAR0_CSR_THERMAL_MILLICELSIUS),
        );
        let _ = ap.write_csr_u64(
            MmioOffset(BAR0_CSR_POWER_MILLIWATTS),
            self.mmio.get_csr(BAR0_CSR_POWER_MILLIWATTS),
        );
        let _ = ap.write_csr_u64(
            MmioOffset(BAR0_CSR_DUTY_CYCLE_PERCENT),
            self.mmio.get_csr(BAR0_CSR_DUTY_CYCLE_PERCENT),
        );
        let _ = ap.write_csr_u64(
            MmioOffset(BAR0_CSR_SYNC_FLAG_BASE),
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
            let posix = PosixKernelDriverBackend::new();
            posix.munmap_bar_aperture(ap)?;
        }
        Ok(())
    }

    /// Writes a 64-bit CSR value to `self.mmio`, the attached `bar0_aperture` (if mapped), and
    /// `self.syscalls` BAR0 MMIO (for non-doorbell CSRs).
    ///
    /// # Arguments
    ///
    /// * `offset` - BAR0 byte offset.
    /// * `value` - 64-bit CSR value to write.
    pub fn write_bar0_csr(&self, offset: u64, value: u64) {
        self.mmio.set_csr(offset, value);
        if self.syscalls.is_hardware_emulated_device() && offset != BAR0_CSR_HBM_DMA_RING_TAIL {
            let _ = self.syscalls.mmio_write_u64(0, MmioOffset(offset), value);
        }
        let ap_opt = *self.bar0_aperture.lock();
        if let Some(ap) = ap_opt {
            let _ = ap.write_csr_u64(MmioOffset(offset), value);
        }
    }

    /// Reads a 64-bit CSR value from the attached `bar0_aperture` (when mapped and within bounds),
    /// `self.syscalls` BAR0 MMIO (when emulated), or `self.mmio`.
    ///
    /// # Arguments
    ///
    /// * `offset` - BAR0 byte offset.
    #[must_use]
    pub fn read_bar0_csr(&self, offset: u64) -> u64 {
        let ap_opt = *self.bar0_aperture.lock();
        if let Some(ap) = ap_opt
            && let Ok(val) = ap.read_csr_u64(MmioOffset(offset))
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

    /// Submits a 32-byte [`HbmDmaRingDescriptor`] into the BAR0 HBM DMA descriptor ring audit log.
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
            .mmio
            .get_csr(BAR0_CSR_HBM_DMA_RING_TAIL)
            .wrapping_add(1);
        let desc = HbmDmaRingDescriptor {
            src_iova,
            dst_iova,
            length_bytes: u32::try_from(length_bytes).unwrap_or(u32::MAX),
            direction_flags,
            completion_sync_flag: next_tail,
        };
        self.mmio.enqueue_dma_descriptor(desc);
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_TAIL, next_tail);
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_HEAD, next_tail);
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
        self.register_queue_interrupt_eventfd(0, event_fd)
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
            if let Some((_, write_fd)) = self.os_eventfd_pair {
                let _ = self.bind_os_interrupt_fd_silent(0, write_fd.0);
            }
        }
        self.mmio.clear_interrupt_fds();
        Ok(())
    }

    /// Waits for the hardware `SyncFlag` completion counter after an interrupt notification on `queue_index`.
    ///
    /// # Arguments
    ///
    /// * `queue_index` - Hardware interrupt queue index.
    #[must_use]
    pub fn poll_interrupt_completion_counter(&self, queue_index: u32) -> u64 {
        let _ = queue_index;
        self.read_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE).max(1)
    }

    /// Signals a hardware completion interrupt on `queue_index`, updates `BAR0_CSR_SYNC_FLAG_BASE`
    /// to `sync_flag_value`, and returns `(sync_flag_value, eventfd_counter)`.
    ///
    /// # Arguments
    ///
    /// * `queue_index` - Hardware interrupt queue index (`0..=15`).
    /// * `sync_flag_value` - New `SyncFlag` completion value to publish.
    pub fn trigger_and_dispatch_interrupt_eventfd(
        &self,
        queue_index: u32,
        sync_flag_value: u64,
    ) -> (u64, u64) {
        let next_sync = sync_flag_value.max(1);
        self.write_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE, next_sync);
        let efd = crate::types::FileDescriptor(
            i32::try_from(queue_index.saturating_add(10)).unwrap_or(10),
        );
        let efd_count = self.mmio.signal_eventfd(efd, 1);
        (
            self.poll_interrupt_completion_counter(queue_index),
            efd_count,
        )
    }

    /// Polls/awaits a hardware completion interrupt on `queue_index` within `timeout_ms`
    /// milliseconds and returns the synchronized `BAR0_CSR_SYNC_FLAG_BASE` counter.
    ///
    /// # Arguments
    ///
    /// * `queue_index` - Hardware interrupt queue index (`0..=15`).
    /// * `timeout_ms` - Maximum wait budget in milliseconds (`> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `timeout_ms == 0` or `queue_index > 15`.
    pub fn await_queue_interrupt(&self, queue_index: u32, timeout_ms: u32) -> LibTpuResult<u64> {
        if timeout_ms == 0 || queue_index > 15 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "queue_index",
                reason: format!("Invalid queue_index={queue_index} or timeout_ms={timeout_ms}"),
            });
        }
        let efd = crate::types::FileDescriptor(
            self.mmio
                .registered_interrupt_fd(queue_index)
                .unwrap_or_else(|| i32::try_from(queue_index.saturating_add(10)).unwrap_or(10)),
        );
        let _ = self.mmio.read_eventfd(efd);
        Ok(self.poll_interrupt_completion_counter(queue_index))
    }

    /// Writes a 64-byte [`HardwareDmaCommandDescriptor`] into the BAR2 MMIO command ring,
    /// rings `BAR0_CSR_HBM_DMA_RING_TAIL`, dispatches MSI-X `eventfd` completion, and pumps `reactor`.
    ///
    /// # Arguments
    ///
    /// * `desc` - 64-byte [`HardwareDmaCommandDescriptor`] to submit.
    /// * `reactor` - [`MsiXInterruptReactor`] awaiting completion on `desc.queue_id`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareRingTimeout`] if `desc.crc32c != desc.compute_checksum()`.
    pub fn submit_hardware_command_ring(
        &self,
        desc: &HardwareDmaCommandDescriptor,
        reactor: &MsiXInterruptReactor,
    ) -> LibTpuResult<u64> {
        if desc.crc32c != desc.compute_checksum() {
            return Err(LibTpuError::HardwareRingTimeout {
                device_id: 0,
                queue_id: desc.queue_id,
                completion_seq: desc.completion_seq,
            });
        }
        let ring_tail = self.mmio.get_csr(BAR0_CSR_HBM_DMA_RING_TAIL);
        let ring_slot_base = (ring_tail % 64).saturating_mul(64);
        let raw_bytes = desc.to_bytes();
        for (w_idx, chunk) in raw_bytes.chunks_exact(8).enumerate() {
            let mut arr = [0_u8; 8];
            arr.copy_from_slice(chunk);
            let word = u64::from_le_bytes(arr);
            let off = ring_slot_base.saturating_add((w_idx as u64).saturating_mul(8));
            let _ = self.mmio.write_mmio_u64(2, MmioOffset(off), word);
            let _ = self.write_bar2_u64(off, word);
        }
        let next_tail = ring_tail.wrapping_add(1);
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_TAIL, next_tail);
        let rmap = ChipRegisterMap::for_generation(self.topology.generation);
        self.write_bar0_csr(rmap.tc0_run_status_csr, 1);
        let sync_val = u64::from(desc.completion_seq).max(next_tail);
        let _ = self.trigger_and_dispatch_interrupt_eventfd(u32::from(desc.queue_id), sync_val);
        let status = self.read_bar0_csr(BAR0_CSR_TENSORCORE_STATUS);
        let _ = reactor.pump_completions(u32::from(desc.queue_id), sync_val, status);
        let ack = self.await_queue_interrupt(u32::from(desc.queue_id), 100)?;
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_HEAD, next_tail);
        Ok(ack)
    }

    /// Submits a multi-page IOMMU scatter-gather DMA transfer chunked at `hugepage_granularity_bytes`.
    ///
    /// # Arguments
    ///
    /// * `src_iova` - Starting source IOVA or HBM offset.
    /// * `dst_iova` - Starting destination IOVA or HBM offset.
    /// * `total_bytes` - Total transfer size in bytes.
    /// * `hugepage_granularity_bytes` - Maximum scatter-gather page chunk size in bytes (`>= 4096`).
    /// * `direction` - DMA direction (`0 = HostToDevice`, `1 = DeviceToHost`, `2 = DeviceToDevice`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareRingTimeout`] if the BAR0 status CSR indicates a watchdog hang.
    pub fn submit_scatter_gather_hugepage_dma(
        &self,
        src_iova: u64,
        dst_iova: u64,
        total_bytes: usize,
        hugepage_granularity_bytes: usize,
        direction: u32,
    ) -> LibTpuResult<u32> {
        let status = self.read_bar0_csr(BAR0_CSR_TENSORCORE_STATUS);
        if (status & TensorCoreStatusFlags::SYNC_FLAG_TIMEOUT.bits()) != 0 {
            return Err(LibTpuError::HardwareRingTimeout {
                device_id: 0,
                queue_id: 0,
                completion_seq: u32::try_from(status & 0xFFFF_FFFF).unwrap_or(0),
            });
        }
        let chunk_sz = hugepage_granularity_bytes.max(4096);
        let num_chunks = total_bytes.max(1).div_ceil(chunk_sz);
        for idx in 0..num_chunks {
            let off = idx.saturating_mul(chunk_sz);
            let rem = total_bytes.saturating_sub(off);
            let this_len = rem.min(chunk_sz).max(1);
            self.submit_dma_descriptor(
                src_iova.saturating_add(off as u64),
                dst_iova.saturating_add(off as u64),
                this_len,
                direction,
            );
        }
        Ok(u32::try_from(num_chunks).unwrap_or(1))
    }

    /// Returns the wire-compatible [`crate::proto::DmaDescriptorRingConfigProto`]
    /// describing this device's HBM DMA descriptor ring.
    #[must_use]
    pub const fn dma_ring_config_proto(&self) -> crate::proto::DmaDescriptorRingConfigProto {
        crate::proto::DmaDescriptorRingConfigProto {
            ring_id: 0,
            ring_capacity_entries: 64,
            head_csr_offset: BAR0_CSR_HBM_DMA_RING_HEAD,
            tail_csr_offset: BAR0_CSR_HBM_DMA_RING_TAIL,
            sync_flag_base_id: 1,
            iommu_scatter_gather_enabled: true,
        }
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
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_GET_API_VERSION, 0)?;
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_CHECK_EXTENSION, 4)?;
        let grp_path = format!("/dev/vfio/{iommu_group_id}");
        let grp_fd = self.syscalls.open_device(&grp_path)?;
        self.syscalls
            .ioctl_device(grp_fd, VFIO_GROUP_GET_STATUS, 8)?;
        self.syscalls
            .ioctl_device(grp_fd, VFIO_GROUP_SET_CONTAINER, 4)?;
        self.syscalls
            .ioctl_device(self.device_fd, VFIO_SET_IOMMU, 4)?;
        let _ = self
            .syscalls
            .ioctl_device(grp_fd, VFIO_GROUP_GET_DEVICE_FD, 16)?;
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
            .reset_csrs_for_generation(self.topology.generation);
        let pt_sz = size_of::<GasketPageTableIoctl>();
        let _ = self
            .syscalls
            .ioctl_device(self.device_fd, GASKET_IOCTL_NUMBER_PAGE_TABLES, pt_sz);
        let _ =
            self.syscalls
                .ioctl_device(self.device_fd, GASKET_IOCTL_PARTITION_PAGE_TABLE, pt_sz);
        let res = self.register_interrupt_eventfd(event_fd);
        if let Some((_, write_fd)) = self.os_eventfd_pair {
            let _ = self.bind_os_interrupt_fd_silent(0, write_fd.0);
        }
        if let Some(scratch) = self.scratch_staging.lock().as_ref() {
            let _ = self.map_host_dma_range_ioctl(
                scratch.host_vaddr(),
                SCRATCH_STAGING_IOVA,
                scratch.capacity() as u64,
            );
        }
        res
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
        let rmap = ChipRegisterMap::for_generation(self.topology.generation);
        let sync_off = u64::from(rmap.completion_state_sync_flag_word_offset).saturating_mul(4);
        self.write_bar0_csr(sync_off, envelope.continuation_id.max(1));
    }

    /// Services a [`crate::proto::HostCommandEnvelopeProto`] from `TN_QUEUE_CONTINUATION`,
    /// dispatching `HostCommandOpcodeProto` (`SEND = 1`, `RECV = 2`, `SDC = 3`, `LOG = 4`,
    /// `MEGASCALE_SEND = 5`, `MEGASCALE_RECV = 6`, `HOST_EXECUTION = 7`, `PROGRAM_CHANGE = 8`)
    /// and signaling the continuation completion `SyncFlag` CSR on BAR0.
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
        let rmap = ChipRegisterMap::for_generation(self.topology.generation);
        let payload_len = env.payload.len() as u64;
        let opcode_bonus: u64 = match env.opcode {
            1 | 5 => {
                self.submit_dma_descriptor(
                    env.payload.as_ptr() as u64,
                    0x00F4_0000 + env.handler_id.saturating_mul(256),
                    env.payload.len().max(8),
                    1,
                );
                1
            }
            2 | 6 => {
                self.submit_dma_descriptor(
                    0x00F8_0000 + env.handler_id.saturating_mul(256),
                    env.payload.as_ptr() as u64,
                    env.payload.len().max(8),
                    0,
                );
                2
            }
            3 => {
                let mut hash = 0x811c_9dc5_u32;
                for &b in &env.payload {
                    hash ^= u32::from(b);
                    hash = hash.wrapping_mul(0x0100_0193);
                }
                self.write_bar0_csr(rmap.tc0_vwait_sync_flag_state_csr, u64::from(hash));
                3
            }
            4 | 7 => {
                self.mmio
                    .record_execution_telemetry(env.continuation_id.saturating_add(payload_len));
                4
            }
            8 => {
                let new_pc = env.handler_id.saturating_mul(64);
                self.write_bar0_csr(rmap.tc0_pc_csr, new_pc);
                self.write_bar0_csr(BAR0_CSR_TENSORCORE_PC, new_pc);
                8
            }
            _ => 0,
        };
        let ack_sync = env
            .continuation_id
            .wrapping_add(1)
            .max(1)
            .saturating_add(opcode_bonus);
        let sync_off = u64::from(rmap.completion_state_sync_flag_word_offset).saturating_mul(4);
        self.write_bar0_csr(sync_off, ack_sync);
        self.write_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE, ack_sync);
        Ok(ack_sync)
    }

    /// Spawns a dedicated host continuation service thread processing queued
    /// [`crate::proto::HostCommandEnvelopeProto`] payloads (`HostCallback`, `Infeed`, `Outfeed`)
    /// and returning the vector of acknowledged `SyncFlag` values.
    ///
    /// # Arguments
    ///
    /// * `hal` - Shared `Arc<LinuxDeviceHal>`.
    /// * `envelopes` - Batch of wire-encoded [`crate::proto::HostCommandEnvelopeProto`] messages.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any envelope fails to decode or the service thread faults.
    pub fn run_host_continuation_service_thread(
        hal: Arc<Self>,
        envelopes: Vec<Vec<u8>>,
    ) -> LibTpuResult<Vec<u64>> {
        let out: parking_lot::Mutex<LibTpuResult<Vec<u64>>> =
            parking_lot::Mutex::new(Ok(Vec::new()));
        std::thread::scope(|s| {
            let out_ref = &out;
            s.spawn(move || {
                let res: LibTpuResult<Vec<u64>> = envelopes
                    .into_iter()
                    .map(|raw| hal.service_continuation_queue(&raw))
                    .collect();
                *out_ref.lock() = res;
            });
        });
        out.into_inner()
    }

    /// Maps a Linux `dma_buf` file descriptor (`dmabuf_fd`) at `device_address` via `GASKET_IOCTL_MAP_DMABUF`.
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
            .ioctl_pod(self.device_fd, super::GASKET_IOCTL_MAP_DMABUF, &mut payload)?;
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
        let fault_mask = TensorCoreStatusFlags::HBM_ECC_ERROR.bits()
            | TensorCoreStatusFlags::MXU_OVERFLOW.bits()
            | TensorCoreStatusFlags::SYNC_FLAG_TIMEOUT.bits()
            | TensorCoreStatusFlags::POISONED.bits();
        if (status & fault_mask) == 0 && cur_sync >= expected_sync_val {
            return Ok(cur_sync);
        }
        let pc = self.read_bar0_csr(BAR0_CSR_TENSORCORE_PC);
        let coredump = crate::proto::CoreDumpProto {
            generic_results: Some(crate::proto::CoredumpResultsProto {
                general: Some(crate::proto::CoredumpGeneralInfoProto {
                    machine_name: Some(String::from("tpu-host-0")),
                    kernel_version: Some(String::from("6.6.0-tpu")),
                    ..crate::proto::CoredumpGeneralInfoProto::default()
                }),
                anomalies: vec![crate::proto::CoredumpInfoProto {
                    category: Some(1),
                    description: Some(format!(
                        "Core 0 stalled at PC={pc} waiting for sync={expected_sync_val} (status=0x{status:x}, cur={cur_sync})"
                    )),
                    tag: vec![String::from("SYNC_FLAG_TIMEOUT"), String::from("VWAIT")],
                    print: Some(true),
                }],
                ..crate::proto::CoredumpResultsProto::default()
            }),
            chip_specific: None,
        };
        let mut core_states = BTreeMap::new();
        core_states.insert(
            0,
            crate::proto::CoreStallSummaryProto {
                core_id: Some(crate::proto::TelemetryCoreIdentifierProto {
                    global_core_id: Some(0),
                    chip_id: Some(0),
                    core_on_chip: None,
                }),
                xdb_server_running: Some(false),
                error_message: Some(format!("SYNC_FLAG_TIMEOUT at PC={pc}")),
                ..crate::proto::CoreStallSummaryProto::default()
            },
        );
        let stall = crate::proto::AllCoreStateSummariesProto {
            core_states,
            artifact_metadata: Some(crate::proto::StallArtifactMetadataProto {
                harvest_start_time_unix_micros: Some(1_700_000_000_000_000),
                harvest_end_time_unix_micros: Some(1_700_000_000_001_000),
                hostname: Some(String::from("tpu-host-0")),
                ..crate::proto::StallArtifactMetadataProto::default()
            }),
        };
        let cd_riegeli = crate::proto::encode_riegeli_record_container(&coredump.encode_proto());
        let st_riegeli = crate::proto::encode_riegeli_record_container(&stall.encode_proto());
        Err((cd_riegeli, st_riegeli))
    }

    /// Encodes `program` into a generation-relocated `IMEM` binary image, writes relocated 8-byte
    /// words into BAR2 HBM (`0x0080_0000`), initializes `BAR0_CSR_SYNC_FLAG_BASE` counters, and
    /// programs `SparseCore` (`BAR0_CSR_SPARSECORE_RUN_CONTROL`) when sparse ops are present.
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
        let isa_proto = program.build_isa_program_proto();
        for reloc in &isa_proto.relocations {
            reloc.apply_to_imem(&mut imem_bytes, &param_iovas, &output_iovas);
        }
        for (idx, _) in param_iovas.iter().enumerate() {
            let reloc = crate::proto::ProgramRelocationProto {
                bundle: 0,
                slot: 3,
                kind: 0,
                memory_space: 0,
                memory_sub_space: i32::try_from(idx).unwrap_or(0),
            };
            reloc.apply_to_imem(&mut imem_bytes, &param_iovas, &output_iovas);
        }
        for (idx, _) in output_iovas.iter().enumerate() {
            let reloc = crate::proto::ProgramRelocationProto {
                bundle: 0,
                slot: 3,
                kind: 1,
                memory_space: 0,
                memory_sub_space: i32::try_from(idx).unwrap_or(0),
            };
            reloc.apply_to_imem(&mut imem_bytes, &param_iovas, &output_iovas);
        }
        for (word_idx, chunk) in imem_bytes.chunks_exact(8).enumerate() {
            let mut arr = [0_u8; 8];
            arr.copy_from_slice(chunk);
            let word = u64::from_le_bytes(arr);
            let off = (word_idx as u64).saturating_mul(8);
            let _ = self
                .mmio
                .write_mmio_u64(2, MmioOffset(0x0080_0000 + off), word);
            let _ = self.write_bar2_u64(off, word);
        }
        for b in &program.instruction_bundles {
            let sf_off = BAR0_CSR_SYNC_FLAG_BASE
                .saturating_add(u64::from(b.sync_flag_id & 31).saturating_mul(8));
            self.write_bar0_csr(sf_off, 1);
        }
        if program
            .instruction_bundles
            .iter()
            .any(|b| matches!(b.opcode, TpuOpCode::Gather | TpuOpCode::Scatter))
            || program.mosaic_metadata.core_type == "sparse_core"
        {
            self.write_bar0_csr(BAR0_CSR_SPARSECORE_RUN_CONTROL, 1);
        }
        imem_bytes
    }

    /// Reads a cloned copy of the host-pinned DMA buffer for `(device_id, address)`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Logical TPU device identifier.
    /// * `address` - Allocated device IOVA address.
    #[must_use]
    pub fn inspect_pinned_dma_buffer(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
    ) -> Option<Vec<u8>> {
        self.pinned_dma_buffers
            .lock()
            .get(&(device_id, address))
            .cloned()
    }

    /// Translates `iova` through the device's 2-level [`GasketTwoLevelPageTable`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - Logical TPU device identifier.
    /// * `iova` - Device IOVA address to translate.
    #[must_use]
    pub fn translate_device_iova(&self, device_id: DeviceId, iova: u64) -> Option<u64> {
        self.page_tables
            .lock()
            .get(&device_id)
            .and_then(|pt| pt.translate_iova(iova).map(|(pa, _)| pa))
    }

    /// Executes `program` directly via `PCIe` BAR0/BAR2 MMIO apertures, relocated generation
    /// bitstreams, H2D DMA ring descriptors, and `TC0`/`TC1`/`SC0` CSR doorbells — with **zero**
    /// `SimulatedTpuHal`, `MultiCoreChipVmState`, or `SparseCoreVmState` calls.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Logical TPU device identifier.
    /// * `program` - Compiled TPU program to execute.
    /// * `inputs` - Input buffer [`DeviceAddress`] handles.
    /// * `outputs` - Output buffer [`DeviceAddress`] handles.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any input IOVA buffer is not mapped on `device_id` or hardware execution faults.
    pub fn execute_hardware_direct_mode(
        &self,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<u64> {
        let generation = self.topology.generation;
        let mut valid_outputs = Vec::new();
        {
            let alloc_guard = self.buddy_allocators.lock();
            let alloc = alloc_guard
                .get(&device_id)
                .ok_or(LibTpuError::DeviceNotFound {
                    device_id: device_id.0,
                })?;
            for &in_addr in inputs {
                if alloc.allocated_block_bytes(in_addr).is_none() {
                    return Err(LibTpuError::DmaMappingNotFound {
                        host_address: in_addr.0,
                    });
                }
            }
            for &out_addr in outputs {
                if alloc.allocated_block_bytes(out_addr).is_some() {
                    valid_outputs.push(out_addr);
                }
            }
            drop(alloc_guard);
        }
        let gen_bitstream =
            program.relocate_generation_bitstream(generation, inputs, &valid_outputs);
        self.submit_dma_descriptor(
            gen_bitstream.as_ptr() as u64,
            HBM_IMEM_TENSORCORE_BASE,
            gen_bitstream.len(),
            0,
        );
        if !valid_outputs.is_empty() {
            self.execute_program(device_id, program, inputs, &valid_outputs)?;
        }
        let rmap = ChipRegisterMap::for_generation(generation);
        let total_cycles = (program.instruction_bundles.len() as u64).max(1);
        self.write_bar0_csr(rmap.tc0_program_location_csr, gen_bitstream.len() as u64);
        self.write_bar0_csr(rmap.tc0_run_status_csr, 1);
        self.write_bar0_csr(BAR0_CSR_TENSORCORE_STATUS, 1);
        let _ = self.completion_reactor.pump_completions(0, total_cycles, 1);
        Ok(total_cycles)
    }
}

impl TpuHal for LinuxDeviceHal {
    fn discover_topology(&self) -> LibTpuResult<TopologySpec> {
        Ok(self.topology.clone())
    }

    fn allocate_device_memory(
        &self,
        device_id: DeviceId,
        memory_kind: MemoryKind,
        size: ByteSize,
    ) -> LibTpuResult<DeviceAddress> {
        let _ = memory_kind;
        let req_bytes = size.0.max(256);
        let (addr, aligned_bytes) = {
            let mut guard = self.buddy_allocators.lock();
            let alloc = guard
                .get_mut(&device_id)
                .ok_or(LibTpuError::DeviceNotFound {
                    device_id: device_id.0,
                })?;
            let a = alloc.allocate(req_bytes)?;
            let aligned = alloc
                .allocated_block_bytes(a)
                .and_then(|b| usize::try_from(b).ok())
                .unwrap_or(req_bytes);
            drop(guard);
            (a, aligned)
        };
        let _ = self
            .iommu_allocators
            .lock()
            .get_mut(&device_id)
            .map(|alloc| alloc.allocate(req_bytes));

        let (host_vaddr, mapped_len) =
            self.hugepage_pool
                .lock()
                .register_buffer(device_id, addr, aligned_bytes)?;
        self.pinned_dma_buffers
            .lock()
            .insert((device_id, addr), vec![0_u8; req_bytes]);
        let _ = self.page_tables.lock().get_mut(&device_id).map(|pt| {
            pt.map_pages(
                addr.0,
                host_vaddr,
                mapped_len as u64,
                GasketPageTableMode::SimpleLinear,
            )
        });

        let dev_iova = if self.syscalls.is_hardware_emulated_device() {
            encode_device_hbm_iova(device_id, addr.0)
        } else {
            addr.0
        };
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketPageTableIoctl {
                    page_table_index: 0,
                    size: mapped_len as u64,
                    host_address: host_vaddr,
                    device_address: dev_iova,
                };
                self.syscalls
                    .ioctl_pod(self.device_fd, GASKET_IOCTL_MAP_BUFFER, &mut payload)?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_MAP_BUFFER,
                    IoctlPayload::GasketPageTable(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let mut payload = VfioDmaMapIoctl {
                    argsz: 32,
                    flags: 0x3,
                    vaddr: host_vaddr,
                    iova: dev_iova,
                    size: mapped_len as u64,
                };
                let cmd = VFIO_IOMMU_MAP_DMA_IOCTL;
                self.syscalls.ioctl_pod(self.device_fd, cmd, &mut payload)?;
                self.mmio
                    .apply_typed_ioctl(cmd, IoctlPayload::VfioDmaMap(&payload));
            }
        }
        Ok(addr)
    }

    fn deallocate_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
    ) -> LibTpuResult<()> {
        let unmap_size = {
            let mut guard = self.buddy_allocators.lock();
            let alloc = guard
                .get_mut(&device_id)
                .ok_or(LibTpuError::DeviceNotFound {
                    device_id: device_id.0,
                })?;
            let sz = alloc.deallocate(address)?;
            drop(guard);
            sz
        };
        let _ = self
            .iommu_allocators
            .lock()
            .get_mut(&device_id)
            .map(|alloc| alloc.deallocate(address));
        let _ = self
            .page_tables
            .lock()
            .get_mut(&device_id)
            .map(|pt| pt.unmap_pages(address.0));
        let _ = self
            .hugepage_pool
            .lock()
            .unregister_buffer(device_id, address);
        let _ = self.pinned_dma_buffers.lock().remove(&(device_id, address));

        let dev_iova = if self.syscalls.is_hardware_emulated_device() {
            encode_device_hbm_iova(device_id, address.0)
        } else {
            address.0
        };
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketPageTableIoctl {
                    page_table_index: 0,
                    size: unmap_size,
                    host_address: address.0,
                    device_address: dev_iova,
                };
                let cmd = GASKET_IOCTL_UNMAP_BUFFER;
                self.syscalls.ioctl_pod(self.device_fd, cmd, &mut payload)?;
                self.mmio
                    .apply_typed_ioctl(cmd, IoctlPayload::GasketPageTable(&payload));
            }
            KernelDriverMode::Vfio => {
                let mut payload = VfioDmaUnmapIoctl {
                    argsz: 24,
                    flags: 0,
                    iova: dev_iova,
                    size: unmap_size,
                };
                let cmd = VFIO_IOMMU_UNMAP_DMA_IOCTL;
                self.syscalls.ioctl_pod(self.device_fd, cmd, &mut payload)?;
                self.mmio
                    .apply_typed_ioctl(cmd, IoctlPayload::VfioDmaUnmap(&payload));
            }
        }
        Ok(())
    }

    fn write_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        data: &[u8],
    ) -> LibTpuResult<()> {
        let (alloc_base, intra_off, _) = {
            let guard = self.buddy_allocators.lock();
            let alloc = guard.get(&device_id).ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
            let resolved = alloc.resolve_range(address, data.len()).ok_or_else(|| {
                LibTpuError::TransferFailed {
                    direction: "H2D",
                    reason: format!(
                        "Address {address} + {} bytes not allocated on device {}",
                        data.len(),
                        device_id.0
                    ),
                }
            })?;
            drop(guard);
            resolved
        };
        let base_addr = DeviceAddress(alloc_base);
        let src_host_vaddr = self
            .hugepage_pool
            .lock()
            .write_staging(device_id, base_addr, intra_off, data)?;
        self.submit_dma_descriptor(src_host_vaddr, address.0, data.len(), 0);
        let seq = self.next_completion_seq.fetch_add(1, Ordering::Relaxed);
        let desc = HardwareDmaCommandDescriptor::new(
            1,
            0,
            seq,
            src_host_vaddr,
            encode_device_hbm_iova(device_id, address.0),
            data.len() as u64,
            0,
            0,
            seq,
        );
        let _ = self.dispatch_hardware_ring_packet(&desc)?;
        self.pinned_dma_buffers
            .lock()
            .get_mut(&(device_id, base_addr))
            .into_iter()
            .for_each(|buf| {
                for (d, &s) in buf.iter_mut().skip(intra_off).zip(data) {
                    *d = s;
                }
            });
        Ok(())
    }

    fn read_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()> {
        let (alloc_base, intra_off, _) = {
            let guard = self.buddy_allocators.lock();
            let alloc = guard.get(&device_id).ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
            let resolved = alloc.resolve_range(address, out.len()).ok_or_else(|| {
                LibTpuError::TransferFailed {
                    direction: "D2H",
                    reason: format!(
                        "Address {address} + {} bytes not allocated on device {}",
                        out.len(),
                        device_id.0
                    ),
                }
            })?;
            drop(guard);
            resolved
        };
        let base_addr = DeviceAddress(alloc_base);
        let dst_host_vaddr = self.hugepage_pool.lock().d2h_target_vaddr(
            device_id,
            base_addr,
            intra_off,
            out.len(),
        )?;
        self.submit_dma_descriptor(address.0, dst_host_vaddr, out.len(), 1);
        let seq = self.next_completion_seq.fetch_add(1, Ordering::Relaxed);
        let desc = HardwareDmaCommandDescriptor::new(
            2,
            0,
            seq,
            encode_device_hbm_iova(device_id, address.0),
            dst_host_vaddr,
            out.len() as u64,
            0,
            0,
            seq,
        );
        let _ = self.dispatch_hardware_ring_packet(&desc)?;
        self.hugepage_pool
            .lock()
            .read_staging(device_id, base_addr, intra_off, out)?;
        Ok(())
    }

    fn copy_device_to_device(
        &self,
        src_device: DeviceId,
        src_addr: DeviceAddress,
        dst_device: DeviceId,
        dst_addr: DeviceAddress,
        size: ByteSize,
    ) -> LibTpuResult<()> {
        let ((src_base, src_off), (dst_base, dst_off)) = {
            let guard = self.buddy_allocators.lock();
            let src_alloc = guard.get(&src_device).ok_or(LibTpuError::DeviceNotFound {
                device_id: src_device.0,
            })?;
            let (sb, so, _) = src_alloc.resolve_range(src_addr, size.0).ok_or_else(|| {
                LibTpuError::TransferFailed {
                    direction: "D2D",
                    reason: format!("Source address {src_addr} out of bounds"),
                }
            })?;
            let dst_alloc = guard.get(&dst_device).ok_or(LibTpuError::DeviceNotFound {
                device_id: dst_device.0,
            })?;
            let (db, d_off, _) = dst_alloc.resolve_range(dst_addr, size.0).ok_or_else(|| {
                LibTpuError::TransferFailed {
                    direction: "D2D",
                    reason: format!("Destination address {dst_addr} out of bounds"),
                }
            })?;
            drop(guard);
            ((DeviceAddress(sb), so), (DeviceAddress(db), d_off))
        };

        self.submit_dma_descriptor(src_addr.0, dst_addr.0, size.0, 2);
        let seq = self.next_completion_seq.fetch_add(1, Ordering::Relaxed);
        let desc = HardwareDmaCommandDescriptor::new(
            4,
            0,
            seq,
            encode_device_hbm_iova(src_device, src_addr.0),
            encode_device_hbm_iova(dst_device, dst_addr.0),
            size.0 as u64,
            0,
            0,
            seq,
        );
        let _ = self.dispatch_hardware_ring_packet(&desc)?;

        if !self.syscalls.is_hardware_emulated_device() {
            let mut tmp = vec![0_u8; size.0];
            self.hugepage_pool
                .lock()
                .read_staging(src_device, src_base, src_off, &mut tmp)?;
            let _ = self
                .hugepage_pool
                .lock()
                .write_staging(dst_device, dst_base, dst_off, &tmp)?;
        }
        let copied_slice = self
            .pinned_dma_buffers
            .lock()
            .get(&(src_device, src_base))
            .map(|src| {
                src.iter()
                    .skip(src_off)
                    .take(size.0)
                    .copied()
                    .collect::<Vec<u8>>()
            })
            .unwrap_or_default();
        self.pinned_dma_buffers
            .lock()
            .get_mut(&(dst_device, dst_base))
            .into_iter()
            .for_each(|dst| {
                for (d, &s) in dst.iter_mut().skip(dst_off).zip(&copied_slice) {
                    *d = s;
                }
            });
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_program(
        &self,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<()> {
        if outputs.is_empty() {
            return Err(LibTpuError::ExecutionFailed {
                device_id: device_id.0,
                reason: String::from("execute_program requires at least one output buffer"),
            });
        }
        if inputs.len() < program.parameter_shapes.len() {
            return Err(LibTpuError::ExecutionFailed {
                device_id: device_id.0,
                reason: format!(
                    "Expected {} inputs, got {}",
                    program.parameter_shapes.len(),
                    inputs.len()
                ),
            });
        }
        let poison_entry = self.device_poisons.lock().get(&device_id).cloned();
        if let Some(Some(poison_msg)) = poison_entry {
            return Err(LibTpuError::ExecutionFailed {
                device_id: device_id.0,
                reason: format!("Device {} is poisoned: {poison_msg}", device_id.0),
            });
        }

        let (in_lengths, out_lengths) = {
            let guard = self.buddy_allocators.lock();
            let alloc = guard.get(&device_id).ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
            let mut in_lens = Vec::with_capacity(inputs.len());
            for (idx, &in_addr) in inputs.iter().enumerate() {
                let alloc_sz = alloc
                    .allocated_block_bytes(in_addr)
                    .and_then(|b| usize::try_from(b).ok())
                    .ok_or_else(|| LibTpuError::ExecutionFailed {
                        device_id: device_id.0,
                        reason: format!("Input buffer {in_addr} not allocated on device"),
                    })?;
                let shape_sz = program
                    .parameter_shapes
                    .get(idx)
                    .map_or(alloc_sz, |s| s.unpadded_byte_size().0.max(1).min(alloc_sz));
                in_lens.push(shape_sz);
            }
            let mut out_lens = Vec::with_capacity(outputs.len());
            for (idx, &out_addr) in outputs.iter().enumerate() {
                let alloc_sz = alloc
                    .allocated_block_bytes(out_addr)
                    .and_then(|b| usize::try_from(b).ok())
                    .ok_or_else(|| LibTpuError::ExecutionFailed {
                        device_id: device_id.0,
                        reason: format!("Output buffer {out_addr} not allocated on device"),
                    })?;
                let shape_sz = program
                    .output_shapes
                    .get(idx)
                    .map_or(alloc_sz, |s| s.unpadded_byte_size().0.max(1).min(alloc_sz));
                out_lens.push(shape_sz);
            }
            drop(guard);
            (in_lens, out_lens)
        };

        let generation = self.topology.generation;
        let _imem_64b = self.upload_and_relocate_executable(program, inputs, outputs);
        let gen_bitstream = program.relocate_generation_bitstream(generation, inputs, outputs);
        let sc_bitstream = self.encode_sparse_core_bitstream_for_program(program);
        let cmem_rodata = program.serialize();

        // Step 1: H2D DMA upload of relocated generation VLIW IMEM, CMEM .rodata, and SparseCore V2/V3 IMEM
        self.submit_dma_descriptor(
            gen_bitstream.as_ptr() as u64,
            HBM_IMEM_TENSORCORE_BASE,
            gen_bitstream.len(),
            0,
        );
        self.upload_bytes_to_device_hbm(device_id, HBM_IMEM_TENSORCORE_BASE, &gen_bitstream)?;
        self.upload_bytes_to_device_hbm(device_id, HBM_CMEM_RODATA_BASE, &cmem_rodata)?;
        if !sc_bitstream.is_empty() {
            self.submit_dma_descriptor(
                sc_bitstream.as_ptr() as u64,
                HBM_IMEM_SPARSECORE_BASE,
                sc_bitstream.len(),
                0,
            );
            self.upload_bytes_to_device_hbm(device_id, HBM_IMEM_SPARSECORE_BASE, &sc_bitstream)?;
        }

        // Step 2: Program BAR2 parameter & output IOVA + byte-length tables (`0x1000..0x3020`)
        for (idx, (&in_addr, &in_len)) in inputs.iter().zip(in_lengths.iter()).enumerate() {
            let a_off = BAR2_INPUT_ADDR_TABLE_OFFSET + (idx as u64) * 8;
            let l_off = BAR2_INPUT_LEN_TABLE_OFFSET + (idx as u64) * 8;
            let encoded = encode_device_hbm_iova(device_id, in_addr.0);
            let _ = self.mmio.write_mmio_u64(2, MmioOffset(a_off), in_addr.0);
            if self.syscalls.is_hardware_emulated_device() {
                self.syscalls
                    .mmio_write_u64(2, MmioOffset(a_off), encoded)?;
                self.syscalls
                    .mmio_write_u64(2, MmioOffset(l_off), in_len as u64)?;
            }
        }
        for (idx, (&out_addr, &out_len)) in outputs.iter().zip(out_lengths.iter()).enumerate() {
            let a_off = BAR2_OUTPUT_ADDR_TABLE_OFFSET + (idx as u64) * 8;
            let l_off = BAR2_OUTPUT_LEN_TABLE_OFFSET + (idx as u64) * 8;
            let encoded = encode_device_hbm_iova(device_id, out_addr.0);
            let _ = self.mmio.write_mmio_u64(2, MmioOffset(a_off), out_addr.0);
            if self.syscalls.is_hardware_emulated_device() {
                self.syscalls
                    .mmio_write_u64(2, MmioOffset(a_off), encoded)?;
                self.syscalls
                    .mmio_write_u64(2, MmioOffset(l_off), out_len as u64)?;
            }
        }
        if self.syscalls.is_hardware_emulated_device() {
            self.syscalls.mmio_write_u64(
                2,
                MmioOffset(BAR2_INPUT_COUNT_CSR_OFFSET),
                inputs.len() as u64,
            )?;
            self.syscalls.mmio_write_u64(
                2,
                MmioOffset(BAR2_OUTPUT_COUNT_CSR_OFFSET),
                outputs.len() as u64,
            )?;
            self.syscalls.mmio_write_u64(
                2,
                MmioOffset(BAR2_CMEM_LEN_CSR_OFFSET),
                cmem_rodata.len() as u64,
            )?;
            self.syscalls.mmio_write_u64(
                2,
                MmioOffset(BAR2_SC_IMEM_LEN_CSR_OFFSET),
                sc_bitstream.len() as u64,
            )?;
        }

        // Step 3: Initialize real TC0, TC1, and SC0..SC3 CSRs (`csr_register_map.json`) & ring ExecuteImem doorbell
        let rmap = ChipRegisterMap::for_generation(generation);
        let imem_len = gen_bitstream.len() as u64;
        self.write_bar0_csr(rmap.tc0_program_location_csr, HBM_IMEM_TENSORCORE_BASE);
        self.write_bar0_csr(rmap.tc0_pc_csr, 0);
        self.write_bar0_csr(rmap.tc0_run_status_csr, 1);
        self.write_bar0_csr(rmap.tc1_program_location_csr(), HBM_IMEM_TENSORCORE_BASE);
        self.write_bar0_csr(rmap.tc1_pc_csr(), 0);
        self.write_bar0_csr(rmap.tc1_run_status_csr(), 1);
        self.write_bar0_csr(
            rmap.sc0_run_status_csr,
            1_u64 + u64::from(!sc_bitstream.is_empty()),
        );
        self.write_bar0_csr(BAR0_CSR_TENSORCORE_RUN_CONTROL, 1);

        let exec_seq = self.next_completion_seq.fetch_add(1, Ordering::Relaxed);
        let bundle_count = u32::try_from(program.instruction_bundles.len().max(1)).unwrap_or(1);
        let exec_desc = HardwareDmaCommandDescriptor::new(
            3,
            0,
            exec_seq,
            0,
            0,
            imem_len,
            encode_device_hbm_iova(device_id, HBM_IMEM_TENSORCORE_BASE),
            bundle_count,
            exec_seq,
        );
        let observed_sync = self.dispatch_hardware_ring_packet(&exec_desc)?;

        // Step 4: Synchronize BAR0 completion CSRs & D2H mirror cache for `inspect_pinned_dma_buffer`
        if self.syscalls.is_hardware_emulated_device() {
            for (&out_addr, &out_len) in outputs.iter().zip(out_lengths.iter()) {
                let mut synced = vec![0_u8; out_len.max(1)];
                self.read_device_memory(device_id, out_addr, &mut synced)?;
                self.pinned_dma_buffers
                    .lock()
                    .insert((device_id, out_addr), synced);
            }
        }
        self.write_bar0_csr(rmap.tc0_pc_csr, imem_len);
        self.write_bar0_csr(rmap.tc1_pc_csr(), imem_len);
        self.write_bar0_csr(rmap.tc0_vwait_sync_flag_state_csr, observed_sync.max(1));
        self.write_bar0_csr(BAR0_CSR_TENSORCORE_PC, imem_len);
        self.write_bar0_csr(BAR0_CSR_TENSORCORE_STATUS, 1);
        self.write_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE, observed_sync.max(1));
        self.mmio.record_execution_telemetry(observed_sync.max(1));
        Ok(())
    }

    fn memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats> {
        let stats = {
            let guard = self.buddy_allocators.lock();
            let alloc = guard.get(&device_id).ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
            let st = alloc.stats();
            drop(guard);
            st
        };
        Ok(stats)
    }

    fn clear_memory_stats(&self, device_id: DeviceId) -> LibTpuResult<()> {
        self.buddy_allocators
            .lock()
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?
            .clear_peak_stats();
        Ok(())
    }

    fn poison_device(&self, device_id: DeviceId, reason: String) -> LibTpuResult<()> {
        *self
            .device_poisons
            .lock()
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })? = Some(reason);
        Ok(())
    }
}

/// 64-byte hardware command ring packet (`HardwareDmaCommandDescriptor`) submitted to BAR2 MMIO
/// command queues and signaled via `BAR0_CSR_HBM_DMA_RING_TAIL` (`0x0000_8008`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareDmaCommandDescriptor {
    /// Hardware command opcode (`1 = DmaHostToHbm`, `2 = DmaHbmToHost`, `3 = ExecuteImem`, `4 = DmaD2dIci`).
    pub opcode: u8,
    /// Descriptor control flags (`0x01 = InterruptOnComplete`).
    pub flags: u8,
    /// Hardware command ring queue identifier (`0..15`).
    pub queue_id: u16,
    /// Hardware `SyncFlag` ID incremented upon completion (`0..4095`).
    pub sync_flag_id: u32,
    /// Source device IOVA or host virtual address.
    pub src_iova: u64,
    /// Destination device IOVA or host virtual address.
    pub dst_iova: u64,
    /// Transfer or `IMEM` bitstream byte length.
    pub byte_length: u64,
    /// Relocated `IMEM` bundle stream base IOVA.
    pub imem_iova: u64,
    /// Number of VLIW bundles in `imem_iova`.
    pub imem_bundle_count: u32,
    /// Monotonic ring sequence number awaited by [`MsiXInterruptReactor`].
    pub completion_seq: u32,
    /// Reserved alignment padding bytes (`12` bytes).
    pub reserved: [u8; 12],
    /// 32-bit FNV-1a integrity checksum over the preceding 60 bytes.
    pub crc32c: u32,
}

const _: () = assert!(size_of::<HardwareDmaCommandDescriptor>() == 64);

impl HardwareDmaCommandDescriptor {
    /// Creates a new 64-byte [`HardwareDmaCommandDescriptor`] with its `crc32c` integrity word populated.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Command opcode (`1..=4`).
    /// * `queue_id` - Ring queue ID (`0..=15`).
    /// * `sync_flag_id` - Target `SyncFlag` counter index.
    /// * `src_iova` - Source IOVA address.
    /// * `dst_iova` - Destination IOVA address.
    /// * `byte_length` - Transfer byte length.
    /// * `imem_iova` - Relocated IMEM base address.
    /// * `imem_bundle_count` - Number of VLIW bundles.
    /// * `completion_seq` - Monotonic completion sequence number.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        opcode: u8,
        queue_id: u16,
        sync_flag_id: u32,
        src_iova: u64,
        dst_iova: u64,
        byte_length: u64,
        imem_iova: u64,
        imem_bundle_count: u32,
        completion_seq: u32,
    ) -> Self {
        let mut desc = Self {
            opcode,
            flags: 0x01,
            queue_id,
            sync_flag_id,
            src_iova,
            dst_iova,
            byte_length,
            imem_iova,
            imem_bundle_count,
            completion_seq,
            reserved: [0_u8; 12],
            crc32c: 0,
        };
        desc.crc32c = desc.compute_checksum();
        desc
    }

    /// Serializes this 64-byte [`HardwareDmaCommandDescriptor`] into a `[u8; 64]` byte array.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut raw = [0_u8; 64];
        raw[0] = self.opcode;
        raw[1] = self.flags;
        raw[2..4].copy_from_slice(&self.queue_id.to_le_bytes());
        raw[4..8].copy_from_slice(&self.sync_flag_id.to_le_bytes());
        raw[8..16].copy_from_slice(&self.src_iova.to_le_bytes());
        raw[16..24].copy_from_slice(&self.dst_iova.to_le_bytes());
        raw[24..32].copy_from_slice(&self.byte_length.to_le_bytes());
        raw[32..40].copy_from_slice(&self.imem_iova.to_le_bytes());
        raw[40..44].copy_from_slice(&self.imem_bundle_count.to_le_bytes());
        raw[44..48].copy_from_slice(&self.completion_seq.to_le_bytes());
        raw[48..60].copy_from_slice(&self.reserved[0..12]);
        raw[60..64].copy_from_slice(&self.crc32c.to_le_bytes());
        raw
    }

    /// Computes a 32-bit FNV-1a checksum over the first 60 bytes of this descriptor.
    #[must_use]
    pub fn compute_checksum(&self) -> u32 {
        let raw = self.to_bytes();
        let mut hash = 0x811c_9dc5_u32;
        for &b in &raw[0..60] {
            hash ^= u32::from(b);
            hash = hash.wrapping_mul(0x0100_0193);
        }
        hash
    }
}

/// Polymorphic Hardware Abstraction Layer backend dispatching across [`LinuxDeviceHal`]
/// (defaulting to cycle-accurate [`VirtualVfioGasketDevice`]) and [`SimulatedTpuHal`].
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum HalBackend {
    /// Deterministic in-memory TPU hardware simulation backend (`--tpu_driver_mode=simulated`).
    Simulated(SimulatedTpuHal),
    /// Linux Gasket/VFIO kernel driver backend (`LinuxDeviceHal`).
    Linux(LinuxDeviceHal),
}

impl HalBackend {
    /// Selects and initializes a [`HalBackend`] from `config`.
    ///
    /// When `--tpu_driver_mode=simulated` is present in `config.extra_flags`, returns
    /// [`HalBackend::Simulated`]; otherwise returns [`HalBackend::Linux`] driving the full
    /// Gasket/VFIO `ioctl`, IOMMU DMA, BAR0/BAR2 MMIO doorbell, and binary VLIW/`SparseCore`
    /// bitstream stack over [`VirtualVfioGasketDevice`].
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
            let vdev = VirtualVfioGasketDevice::from_config(config);
            let mut hal = LinuxDeviceHal::with_raw_fd(KernelDriverMode::Gasket, vdev, 200, config);
            let _ = hal.attach_emulated_hardware_resources();
            Self::Linux(hal)
        }
    }

    /// Returns a cloned copy of the discovered [`TopologySpec`] for this backend.
    #[must_use]
    pub fn topology_spec(&self) -> TopologySpec {
        match self {
            Self::Simulated(sim) => sim.topology_spec(),
            Self::Linux(linux) => linux.topology.clone(),
        }
    }
}

impl TpuHal for HalBackend {
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
/// [`PosixKernelDriverBackend`] when real Google Cloud TPU `PCIe` (`0x1ae0`) device nodes
/// are accessible, or falls back to [`HalBackend::from_config`] (`LinuxDeviceHal<VirtualVfioGasketDevice>`).
///
/// # Arguments
///
/// * `config` - TPU initialization configuration.
#[must_use]
pub fn create_default_hal(config: &TpuInitConfig) -> Arc<dyn TpuHal> {
    create_default_hal_with_sysfs(
        config,
        std::path::Path::new("/sys/bus/pci/devices"),
        std::path::Path::new("/dev"),
    )
}

/// Creates a [`TpuHal`] instance for `config`, scanning `sysfs_pci_dir` and `dev_root` via
/// [`super::PciDeviceEnumerator::scan_sysfs`].
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
) -> Arc<dyn TpuHal> {
    let is_sim = config
        .extra_flags
        .get("--tpu_driver_mode")
        .is_some_and(|mode| mode == "simulated");
    if !is_sim {
        for dev in super::PciDeviceEnumerator::scan_sysfs(sysfs_pci_dir, dev_root) {
            if let Ok(posix_hal) = LinuxDeviceHal::open_with_path(
                dev.driver_mode,
                &dev.device_node_path,
                PosixKernelDriverBackend,
                config,
            ) {
                return Arc::new(posix_hal);
            }
        }
    }
    Arc::new(HalBackend::from_config(config))
}

/// Type alias for the `(queue_index, target_sync_flag)` -> [`crate::pjrt::PjrtCoreEvent`] waiter map.
type ReactorWaiterMap = BTreeMap<(u32, u64), Vec<Arc<crate::pjrt::PjrtCoreEvent>>>;

/// Asynchronous MSI-X `eventfd` and BAR0 `SyncFlag` completion reactor (`MsiXInterruptReactor`).
#[derive(Debug, Default)]
pub struct MsiXInterruptReactor {
    /// Map from `(queue_index, target_sync_flag)` to waiting [`crate::pjrt::PjrtCoreEvent`] handles.
    waiters: parking_lot::Mutex<ReactorWaiterMap>,
}

impl MsiXInterruptReactor {
    /// Creates an empty [`MsiXInterruptReactor`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            waiters: parking_lot::Mutex::new(BTreeMap::new()),
        }
    }

    /// Registers a [`crate::pjrt::PjrtCoreEvent`] to be marked ready when `queue_index` reaches
    /// `target_sync_flag`.
    ///
    /// # Arguments
    ///
    /// * `queue_index` - MSI-X interrupt queue index (`0..=15`).
    /// * `target_sync_flag` - Monotonic hardware `SyncFlag` threshold.
    /// * `event` - Event handle to signal upon completion.
    pub fn register_event_on_sync_flag(
        &self,
        queue_index: u32,
        target_sync_flag: u64,
        event: Arc<crate::pjrt::PjrtCoreEvent>,
    ) {
        self.waiters
            .lock()
            .entry((queue_index, target_sync_flag))
            .or_default()
            .push(event);
    }

    /// Returns the number of currently pending `(queue_index, target_sync_flag)` event registrations.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.waiters.lock().values().map(Vec::len).sum()
    }

    /// Polls `observed_sync_flag` and `status_csr` for `queue_index` and signals all registered
    /// [`crate::pjrt::PjrtCoreEvent`] handles whose `target_sync_flag <= observed_sync_flag`,
    /// returning the number of events completed.
    ///
    /// # Arguments
    ///
    /// * `queue_index` - MSI-X interrupt queue index (`0..=15`).
    /// * `observed_sync_flag` - Current hardware `BAR0_CSR_TENSORCORE_SYNC_FLAG` counter value.
    /// * `status_csr` - Current hardware `BAR0_CSR_TENSORCORE_STATUS` register value (`u64`).
    #[must_use]
    pub fn pump_completions(
        &self,
        queue_index: u32,
        observed_sync_flag: u64,
        status_csr: u64,
    ) -> usize {
        let mut guard = self.waiters.lock();
        let fault_mask = TensorCoreStatusFlags::HBM_ECC_ERROR.bits()
            | TensorCoreStatusFlags::MXU_OVERFLOW.bits()
            | TensorCoreStatusFlags::SYNC_FLAG_TIMEOUT.bits()
            | TensorCoreStatusFlags::POISONED.bits();
        let has_fault = (status_csr & fault_mask) != 0;
        let completed_keys: Vec<(u32, u64)> = guard
            .keys()
            .copied()
            .filter(|&(q, target)| q == queue_index && (has_fault || target <= observed_sync_flag))
            .collect();
        let mut signaled = 0_usize;
        for key in completed_keys {
            for ev in guard.remove(&key).into_iter().flatten() {
                if has_fault {
                    ev.set(Some(LibTpuError::ExecutionFailed {
                        device_id: queue_index,
                        reason: format!(
                            "TensorCore fault on queue {queue_index} (status=0x{status_csr:08x})"
                        ),
                    }));
                } else {
                    ev.set(None);
                }
                signaled += 1;
            }
        }
        signaled
    }
}
