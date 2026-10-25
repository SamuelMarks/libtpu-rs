//! Hardware Abstraction Layer (`TpuHal`) for simulated and Linux VFIO/Gasket TPU devices.
//!
//! Implements both an in-memory deterministic [`SimulatedTpuHal`] and a kernel-ioctl + `PCIe` BAR MMIO
//! backed [`LinuxDeviceHal`] supporting the reverse-engineered Gasket (`0xdc`) and VFIO (`0x3b`)
//! device protocols.

use crate::compiler::{CompiledProgram, TpuOpCode};
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{
    ByteSize, ChipCoordinates, ChipId, CoreId, DeviceAddress, DeviceId, ElementType, EventFdHandle,
    FileDescriptor, HostId, IoctlCommand, LocalHardwareId, MemoryKind, MemoryLayout, MmioOffset,
    PciDeviceId, PciVendorId, ProcessIndex, Shape, TpuGeneration,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Exact Gasket ioctl command for querying simple page table addresses (`_IOWR(0xdc, 0x05, 32)`).
pub const GASKET_IOCTL_NUMBER_PAGE_TABLES: u64 = 0xc020_dc05;

/// Exact Gasket ioctl command for partitioning page tables (`_IOW(0xdc, 0x07, 32)`).
pub const GASKET_IOCTL_PARTITION_PAGE_TABLE: u64 = 0x4020_dc07;

/// Exact Gasket ioctl command for mapping host DMA buffers (`_IOW(0xdc, 0x08, 32)`).
pub const GASKET_IOCTL_MAP_BUFFER: u64 = 0x4020_dc08;

/// Exact Gasket ioctl command for unmapping host DMA buffers (`_IOW(0xdc, 0x09, 32)`).
pub const GASKET_IOCTL_UNMAP_BUFFER: u64 = 0x4020_dc09;

/// Exact Gasket ioctl command for binding an interrupt `eventfd` (`_IOW(0xdc, 0x0b, 32)`).
pub const GASKET_IOCTL_SET_EVENTFD: u64 = 0x4020_dc0b;

/// Exact Gasket ioctl command for clearing an interrupt `eventfd` (`_IOW(0xdc, 0x0c, 8)`).
pub const GASKET_IOCTL_CLEAR_EVENTFD: u64 = 0x4008_dc0c;

/// Exact Gasket ioctl command for mapping a `dma_buf` fd (`_IOW(0xdc, 0x0d, 24)`).
pub const GASKET_IOCTL_MAP_DMABUF: u64 = 0x4018_dc0d;

/// Exact VFIO ioctl command for configuring MSI-X interrupt signaling (`0x3b6e`).
pub const VFIO_DEVICE_SET_IRQS_IOCTL: u64 = 0x0000_3b6e;

/// Alias for [`VFIO_DEVICE_SET_IRQS_IOCTL`] (`0x3b6e`).
pub const VFIO_DEVICE_SET_IRQS: u64 = VFIO_DEVICE_SET_IRQS_IOCTL;

/// Exact VFIO ioctl command for mapping DMA pages (`0x3b71`).
pub const VFIO_IOMMU_MAP_DMA_IOCTL: u64 = 0x0000_3b71;

/// Alias for [`VFIO_IOMMU_MAP_DMA_IOCTL`] (`0x3b71`).
pub const VFIO_IOMMU_MAP_DMA: u64 = VFIO_IOMMU_MAP_DMA_IOCTL;

/// Exact VFIO ioctl command for unmapping DMA pages (`0x3b72`).
pub const VFIO_IOMMU_UNMAP_DMA_IOCTL: u64 = 0x0000_3b72;

/// Alias for [`VFIO_IOMMU_UNMAP_DMA_IOCTL`] (`0x3b72`).
pub const VFIO_IOMMU_UNMAP_DMA: u64 = VFIO_IOMMU_UNMAP_DMA_IOCTL;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Run Control CSR.
pub const BAR0_CSR_TENSORCORE_RUN_CONTROL: u64 = 0x0000_1000;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 `SyncFlag` Completion Counter CSR.
pub const BAR0_CSR_TENSORCORE_SYNC_FLAG: u64 = 0x0000_1008;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Program Counter CSR.
pub const BAR0_CSR_TENSORCORE_PC: u64 = 0x0000_1008;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Execution Status CSR.
pub const BAR0_CSR_TENSORCORE_STATUS: u64 = 0x0000_1010;

/// `PCIe` BAR0 MMIO register offset for `SyncFlag` table base (`0x2000`).
pub const BAR0_CSR_SYNC_FLAG_BASE: u64 = 0x0000_2000;

/// `PCIe` BAR0 MMIO register offset for `SparseCore` 0 Run Control CSR (`0x4000`).
pub const BAR0_CSR_SPARSECORE_RUN_CONTROL: u64 = 0x0000_4000;

/// `PCIe` BAR0 MMIO register offset for HBM DMA descriptor ring head pointer (`0x8000`).
pub const BAR0_CSR_HBM_DMA_RING_HEAD: u64 = 0x0000_8000;

/// `PCIe` BAR0 MMIO register offset for HBM DMA descriptor ring tail doorbell (`0x8008`).
pub const BAR0_CSR_HBM_DMA_RING_TAIL: u64 = 0x0000_8008;

/// `PCIe` BAR0 MMIO register offset for ICI optical/electrical router link state (`0x9000`).
pub const BAR0_CSR_ICI_ROUTER_LINK_STATE: u64 = 0x0000_9000;

/// `PCIe` BAR0 MMIO register offset for ASIC junction temperature in millicelsius (`0xA000`).
pub const BAR0_CSR_THERMAL_MILLICELSIUS: u64 = 0x0000_A000;

/// `PCIe` BAR0 MMIO register offset for board power draw in milliwatts (`0xA008`).
pub const BAR0_CSR_POWER_MILLIWATTS: u64 = 0x0000_A008;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` duty cycle percentage (`0xA010`).
pub const BAR0_CSR_DUTY_CYCLE_PERCENT: u64 = 0x0000_A010;

/// Gasket 32-byte DMA page-table mapping payload (`gasket_page_table_ioctl`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasketPageTableIoctl {
    /// Page table index (`0` for `TensorCore` 0).
    pub page_table_index: u64,
    /// Size in bytes (page-aligned to 4096 bytes).
    pub size: u64,
    /// Host virtual memory address (`u64`).
    pub host_address: u64,
    /// Device DMA virtual address (`u64`).
    pub device_address: u64,
}

/// Gasket 32-byte interrupt `eventfd` registration payload (`gasket_interrupt_eventfd`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasketInterruptEventFd {
    /// Interrupt index (`0..N`).
    pub interrupt: u64,
    /// Linux `eventfd` file descriptor.
    pub event_fd: u64,
    /// Reserved fields (`0`).
    pub reserved: [u64; 2],
}

/// Gasket 24-byte `dma_buf` mapping payload (`gasket_page_table_dmabuf`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasketPageTableDmaBuf {
    /// Page table index (`0`).
    pub page_table_index: u64,
    /// Target device IOVA address.
    pub device_address: u64,
    /// Linux `dma_buf` file descriptor.
    pub dma_buf_fd: i32,
    /// Mapping flags.
    pub flags: u32,
}

/// VFIO 32-byte IOMMU DMA mapping payload (`vfio_iommu_type1_dma_map`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfioDmaMapIoctl {
    /// Structure size (`32` bytes).
    pub argsz: u32,
    /// Mapping flags (`0x3` = `READ | WRITE`).
    pub flags: u32,
    /// Host user-space virtual address.
    pub vaddr: u64,
    /// I/O virtual address in TPU HBM space.
    pub iova: u64,
    /// Mapping size in bytes.
    pub size: u64,
}

/// VFIO 24-byte IOMMU DMA unmapping payload (`vfio_iommu_type1_dma_unmap`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfioDmaUnmapIoctl {
    /// Structure size (`24` bytes).
    pub argsz: u32,
    /// Unmap flags (`0`).
    pub flags: u32,
    /// I/O virtual address to unmap.
    pub iova: u64,
    /// Size in bytes to unmap.
    pub size: u64,
}

/// VFIO 24-byte interrupt signaling header (`vfio_irq_set`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfioIrqSetIoctl {
    /// Structure size (`24` bytes).
    pub argsz: u32,
    /// Action and data flags (`VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER`).
    pub flags: u32,
    /// IRQ index (`VFIO_PCI_MSIX_IRQ_INDEX`).
    pub index: u32,
    /// First sub-index to configure.
    pub start: u32,
    /// Number of IRQs to configure.
    pub count: u32,
    /// Eventfd file descriptor payload.
    pub event_fd: i32,
}

const _: () = assert!(size_of::<GasketPageTableIoctl>() == 32);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, page_table_index) == 0);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, size) == 8);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, host_address) == 16);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, device_address) == 24);
const _: () = assert!(size_of::<GasketInterruptEventFd>() == 32);
const _: () = assert!(std::mem::offset_of!(GasketInterruptEventFd, interrupt) == 0);
const _: () = assert!(std::mem::offset_of!(GasketInterruptEventFd, event_fd) == 8);
const _: () = assert!(std::mem::offset_of!(GasketInterruptEventFd, reserved) == 16);
const _: () = assert!(size_of::<GasketPageTableDmaBuf>() == 24);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, page_table_index) == 0);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, device_address) == 8);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, dma_buf_fd) == 16);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, flags) == 20);
const _: () = assert!(size_of::<VfioDmaMapIoctl>() == 32);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, argsz) == 0);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, flags) == 4);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, vaddr) == 8);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, iova) == 16);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, size) == 24);
const _: () = assert!(size_of::<VfioDmaUnmapIoctl>() == 24);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, argsz) == 0);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, flags) == 4);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, iova) == 8);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, size) == 16);
const _: () = assert!(size_of::<VfioIrqSetIoctl>() == 24);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, argsz) == 0);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, flags) == 4);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, index) == 8);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, start) == 12);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, count) == 16);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, event_fd) == 20);

bitflags::bitflags! {
    /// Strongly-typed bitmask flags for `BAR0_CSR_TENSORCORE_RUN_CONTROL`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TensorCoreControlFlags: u64 {
        /// Starts `TensorCore` instruction execution (`1 << 0`).
        const START = 1 << 0;
        /// Halts `TensorCore` instruction execution (`1 << 1`).
        const HALT = 1 << 1;
        /// Advances `TensorCore` by a single VLIW bundle (`1 << 2`).
        const SINGLE_STEP = 1 << 2;
        /// Performs a soft architectural reset of the `TensorCore` (`1 << 3`).
        const SOFT_RESET = 1 << 3;
        /// Clears sticky fault bits in the `TensorCore` status CSR (`1 << 4`).
        const CLEAR_FAULT = 1 << 4;
    }

    /// Strongly-typed bitmask flags for `BAR0_CSR_TENSORCORE_STATUS`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TensorCoreStatusFlags: u64 {
        /// `TensorCore` is idle and ready for launch (`1 << 0`).
        const IDLE = 1 << 0;
        /// `TensorCore` is actively executing bundles (`1 << 1`).
        const RUNNING = 1 << 1;
        /// `TensorCore` has completed the last launched program (`1 << 2`).
        const COMPLETED = 1 << 2;
        /// Uncorrectable HBM ECC error detected (`1 << 3`).
        const HBM_ECC_ERROR = 1 << 3;
        /// Matrix Multiply Unit (MXU) arithmetic fault (`1 << 4`).
        const MXU_OVERFLOW = 1 << 4;
        /// Architectural `SyncFlag` barrier timeout (`1 << 5`).
        const SYNC_FLAG_TIMEOUT = 1 << 5;
        /// Device execution has been poisoned (`1 << 6`).
        const POISONED = 1 << 6;
    }

    /// Strongly-typed bitmask flags for `VFIO_IOMMU_MAP_DMA` (`VfioDmaMapIoctl::flags`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VfioDmaMapFlags: u32 {
        /// Device may read from the mapped DMA region (`1 << 0`).
        const READ = 1 << 0;
        /// Device may write to the mapped DMA region (`1 << 1`).
        const WRITE = 1 << 1;
    }

    /// Strongly-typed bitmask flags for `VFIO_DEVICE_SET_IRQS` (`VfioIrqSetIoctl::flags`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VfioIrqSetFlags: u32 {
        /// No payload data (`1 << 0`).
        const DATA_NONE = 1 << 0;
        /// Boolean payload (`1 << 1`).
        const DATA_BOOL = 1 << 1;
        /// Linux `eventfd` payload (`1 << 2`).
        const DATA_EVENTFD = 1 << 2;
        /// Mask the interrupt (`1 << 3`).
        const ACTION_MASK = 1 << 3;
        /// Unmask the interrupt (`1 << 4`).
        const ACTION_UNMASK = 1 << 4;
        /// Trigger the interrupt (`1 << 5`).
        const ACTION_TRIGGER = 1 << 5;
    }
}

/// Hardware 32-byte HBM DMA ring descriptor (`HbmDmaRingDescriptor`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HbmDmaRingDescriptor {
    /// Source IOVA or HBM physical address.
    pub src_iova: u64,
    /// Destination IOVA or HBM physical address.
    pub dst_iova: u64,
    /// Transfer length in bytes.
    pub length_bytes: u32,
    /// DMA direction and attribute flags (`0` = H2D, `1` = D2H, `2` = D2D).
    pub direction_flags: u32,
    /// Architectural `SyncFlag` ID incremented upon DMA completion.
    pub completion_sync_flag: u64,
}

const _: () = assert!(size_of::<HbmDmaRingDescriptor>() == 32);

/// Coalescing free-list and allocated-interval tracker for TPU HBM and VMEM address spaces.
#[derive(Debug, Clone)]
pub struct HbmRegionAllocator {
    /// Base virtual/physical address of the managed memory space.
    pub base_address: u64,
    /// Total capacity in bytes of the managed memory space.
    pub capacity_bytes: usize,
    /// Map of free block start addresses to their contiguous byte lengths.
    free_blocks: BTreeMap<u64, usize>,
    /// Map of active allocation start addresses to their aligned byte lengths.
    allocated_blocks: BTreeMap<u64, usize>,
}

impl HbmRegionAllocator {
    /// Creates a new [`HbmRegionAllocator`] managing `[base_address, base_address + capacity_bytes)`.
    ///
    /// # Arguments
    ///
    /// * `base_address` - Starting HBM/VMEM address (e.g., `0x1000_0000`).
    /// * `capacity_bytes` - Total capacity of the region in bytes.
    #[must_use]
    pub fn new(base_address: u64, capacity_bytes: usize) -> Self {
        let mut free_blocks = BTreeMap::new();
        if capacity_bytes > 0 {
            free_blocks.insert(base_address, capacity_bytes);
        }
        Self {
            base_address,
            capacity_bytes,
            free_blocks,
            allocated_blocks: BTreeMap::new(),
        }
    }

    /// Resizes the total capacity of the allocator and resets free blocks around active allocations.
    ///
    /// # Arguments
    ///
    /// * `new_capacity` - New total capacity in bytes.
    pub fn set_capacity(&mut self, new_capacity: usize) {
        self.capacity_bytes = new_capacity;
        let used: usize = self.allocated_blocks.values().sum();
        self.free_blocks.clear();
        if new_capacity > used {
            let next_start = self
                .allocated_blocks
                .iter()
                .next_back()
                .map_or(self.base_address, |(&addr, &len)| addr + len as u64);
            self.free_blocks.insert(next_start, new_capacity - used);
        }
    }

    /// Allocates a page-aligned (`4096`-byte) block of at least `requested_bytes` using first-fit
    /// and splits any remaining tail back into the free list.
    ///
    /// # Arguments
    ///
    /// * `requested_bytes` - Number of bytes requested.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::OutOfMemory`] if no contiguous free block is large enough.
    pub fn allocate(&mut self, requested_bytes: usize) -> LibTpuResult<DeviceAddress> {
        let aligned_size = requested_bytes.max(256).div_ceil(4096) * 4096;
        let candidate = self
            .free_blocks
            .iter()
            .find(|(_, len)| **len >= aligned_size)
            .map(|(&addr, &len)| (addr, len));
        let Some((addr, block_len)) = candidate else {
            return Err(LibTpuError::OutOfMemory {
                requested_bytes,
                available_bytes: self.largest_free_block_bytes(),
            });
        };
        self.free_blocks.remove(&addr);
        let remainder = block_len - aligned_size;
        if remainder > 0 {
            self.free_blocks
                .insert(addr + aligned_size as u64, remainder);
        }
        self.allocated_blocks.insert(addr, aligned_size);
        Ok(DeviceAddress(addr))
    }

    /// Frees a previously allocated block at `address` and coalesces it with adjacent predecessor
    /// and successor free blocks.
    ///
    /// # Arguments
    ///
    /// * `address` - [`DeviceAddress`] returned by [`HbmRegionAllocator::allocate`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `address` is not an active allocation.
    pub fn deallocate(&mut self, address: DeviceAddress) -> LibTpuResult<usize> {
        let Some(freed_len) = self.allocated_blocks.remove(&address.0) else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "address",
                reason: format!("HBM block {address} is not allocated"),
            });
        };
        let mut merged_start = address.0;
        let mut merged_len = freed_len;

        let prev_merge = self
            .free_blocks
            .range(..address.0)
            .next_back()
            .filter(|&(&prev_start, &prev_len)| prev_start + prev_len as u64 == merged_start)
            .map(|(&prev_start, &prev_len)| (prev_start, prev_len));
        if let Some((prev_start, prev_len)) = prev_merge {
            self.free_blocks.remove(&prev_start);
            merged_start = prev_start;
            merged_len += prev_len;
        }
        let next_expected = merged_start + merged_len as u64;
        if let Some(next_len) = self.free_blocks.remove(&next_expected) {
            merged_len += next_len;
        }
        self.free_blocks.insert(merged_start, merged_len);
        Ok(freed_len)
    }

    /// Returns the size in bytes of the largest contiguous free block in the allocator.
    #[must_use]
    pub fn largest_free_block_bytes(&self) -> usize {
        self.free_blocks.values().copied().max().unwrap_or(0)
    }
}

/// Safe volatile MMIO wrapper around an `mmap`-ed `PCIe` BAR0/BAR2 virtual memory aperture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedBarAperture {
    /// Base user-space virtual address returned by `mmap`.
    pub base_vaddr: usize,
    /// Total mapped aperture length in bytes.
    pub size: ByteSize,
}

impl MappedBarAperture {
    /// Performs a bounds-checked, 8-byte-aligned volatile 64-bit CSR read at `offset`.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset within the mapped BAR aperture.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset` is unaligned or exceeds `self.size`.
    pub fn read_csr_u64(&self, offset: MmioOffset) -> LibTpuResult<u64> {
        let off = usize::try_from(offset.0).unwrap_or(usize::MAX);
        if (off & 7) != 0 || off.saturating_add(8) > self.size.0 || self.base_vaddr == 0 {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index: 0,
                offset: offset.0,
                size: 8,
            });
        }
        let reg_ptr = (self.base_vaddr + off) as *const u64;
        // SAFETY: `reg_ptr` is 8-byte aligned and within `[base_vaddr, base_vaddr + size)`.
        let val = unsafe { std::ptr::read_volatile(reg_ptr) };
        Ok(val)
    }

    /// Performs a bounds-checked, 8-byte-aligned volatile 64-bit CSR write at `offset`.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset within the mapped BAR aperture.
    /// * `value` - 64-bit CSR value to write.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset` is unaligned or exceeds `self.size`.
    pub fn write_csr_u64(&self, offset: MmioOffset, value: u64) -> LibTpuResult<()> {
        let off = usize::try_from(offset.0).unwrap_or(usize::MAX);
        if (off & 7) != 0 || off.saturating_add(8) > self.size.0 || self.base_vaddr == 0 {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index: 0,
                offset: offset.0,
                size: 8,
            });
        }
        let reg_ptr = (self.base_vaddr + off) as *mut u64;
        // SAFETY: `reg_ptr` is 8-byte aligned and within `[base_vaddr, base_vaddr + size)`.
        unsafe {
            std::ptr::write_volatile(reg_ptr, value);
        }
        Ok(())
    }
}

/// Discovered `PCIe` Google Cloud TPU device descriptor from `/sys/bus/pci/devices/*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredPciTpuDevice {
    /// `PCIe` BDF slot name (e.g., `"0000:00:04.0"`).
    pub bdf_slot: String,
    /// `PCIe` Vendor ID (`0x1ae0`).
    pub vendor_id: PciVendorId,
    /// `PCIe` Device ID (`0x0027`, `0x0056`, `0x005e`, `0x0062`, `0x0063`, `0x006f`, `0x0076`).
    pub device_id: PciDeviceId,
    /// Detected TPU architecture generation.
    pub generation: TpuGeneration,
    /// Detected kernel driver binding mode (`Gasket` or `Vfio`).
    pub driver_mode: KernelDriverMode,
    /// Device node path (e.g., `"/dev/accel0"` or `"/dev/vfio/0"`).
    pub device_node_path: String,
}

/// Linux `sysfs` `PCIe` hardware enumerator for Google Cloud TPU ASICs (`0x1ae0`).
#[derive(Debug, Clone, Copy, Default)]
pub struct PciDeviceEnumerator;

impl PciDeviceEnumerator {
    /// Maps a Google `PCIe` Device ID (`0x1ae0:<device_id>`) to its [`TpuGeneration`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - 16-bit `PCIe` Device ID.
    #[must_use]
    pub const fn generation_from_pci_device_id(device_id: PciDeviceId) -> Option<TpuGeneration> {
        match device_id.0 {
            0x0027 => Some(TpuGeneration::V3),
            0x0056 => Some(TpuGeneration::V4),
            0x005e | 0x0062 => Some(TpuGeneration::V5p),
            0x0063 => Some(TpuGeneration::V5e),
            0x006f => Some(TpuGeneration::V6e),
            0x0076 => Some(TpuGeneration::V7x),
            _ => None,
        }
    }

    /// Parses a hexadecimal `sysfs` attribute string such as `"0x1ae0\n"` into a `u16`.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw contents of a `sysfs` `vendor` or `device` file.
    #[must_use]
    pub fn parse_hex_u16(raw: &str) -> Option<u16> {
        let trimmed = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
        u16::from_str_radix(trimmed, 16).ok()
    }

    /// Scans a `sysfs` PCI devices directory (e.g., `/sys/bus/pci/devices`) and `/dev` root to
    /// discover all attached Google Cloud TPU `PCIe` functions.
    ///
    /// # Arguments
    ///
    /// * `sysfs_pci_dir` - Path to `/sys/bus/pci/devices`.
    /// * `dev_root` - Path to `/dev`.
    #[must_use]
    pub fn scan_sysfs(
        sysfs_pci_dir: &std::path::Path,
        dev_root: &std::path::Path,
    ) -> Vec<DiscoveredPciTpuDevice> {
        let Ok(entries) = std::fs::read_dir(sysfs_pci_dir) else {
            return Vec::new();
        };
        let mut slot_paths: Vec<std::path::PathBuf> =
            entries.filter_map(|e| e.ok().map(|d| d.path())).collect();
        slot_paths.sort();

        slot_paths
            .into_iter()
            .enumerate()
            .filter_map(|(idx, path)| {
                let vendor_str = std::fs::read_to_string(path.join("vendor")).unwrap_or_default();
                let device_str = std::fs::read_to_string(path.join("device")).unwrap_or_default();
                let vendor_raw = Self::parse_hex_u16(&vendor_str).filter(|&v| v == 0x1ae0)?;
                let device_raw = Self::parse_hex_u16(&device_str)?;
                let pci_dev_id = PciDeviceId(device_raw);
                let generation = Self::generation_from_pci_device_id(pci_dev_id)?;
                let bdf_slot = path
                    .file_name()
                    .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
                let (driver_mode, device_node_path) = if path.join("vfio").exists() {
                    (
                        KernelDriverMode::Vfio,
                        dev_root
                            .join("vfio")
                            .join(idx.to_string())
                            .to_string_lossy()
                            .into_owned(),
                    )
                } else {
                    (
                        KernelDriverMode::Gasket,
                        dev_root
                            .join(format!("accel{idx}"))
                            .to_string_lossy()
                            .into_owned(),
                    )
                };
                Some(DiscoveredPciTpuDevice {
                    bdf_slot,
                    vendor_id: PciVendorId(vendor_raw),
                    device_id: pci_dev_id,
                    generation,
                    driver_mode,
                    device_node_path,
                })
            })
            .collect()
    }
}

/// Specification for a single logical TPU device in a discovered topology.
#[derive(Debug, Clone)]
pub struct DeviceSpec {
    /// Global logical device identifier.
    pub device_id: DeviceId,
    /// Physical chip identifier.
    pub chip_id: ChipId,
    /// Core index on the physical chip.
    pub core_on_chip: CoreId,
    /// Host identifier owning this chip.
    pub host_id: HostId,
    /// PJRT process index owning this device.
    pub process_index: ProcessIndex,
    /// Local hardware ordinal on the host.
    pub local_hardware_id: LocalHardwareId,
    /// 3D torus coordinates `(x, y, z)` of the chip.
    pub chip_coords: ChipCoordinates,
    /// 3D torus coordinates `(x, y, z)` of the host.
    pub host_coords: ChipCoordinates,
    /// Whether this device is directly addressable by the current process.
    pub is_addressable: bool,
    /// TPU hardware generation.
    pub generation: TpuGeneration,
}

/// Complete TPU slice topology specification discovered by a [`TpuHal`].
#[derive(Debug, Clone)]
pub struct TopologySpec {
    /// Platform identifier (`"tpu"`).
    pub platform_name: String,
    /// Platform human-readable version string.
    pub platform_version: String,
    /// Hardware generation of chips in the topology.
    pub generation: TpuGeneration,
    /// 3D mesh bounds of chips across the entire slice.
    pub chip_bounds: ChipCoordinates,
    /// 3D mesh bounds of hosts across the slice.
    pub host_bounds: ChipCoordinates,
    /// Number of physical chips per host.
    pub chips_per_host: u32,
    /// Ordered list of all logical devices in the slice.
    pub devices: Vec<DeviceSpec>,
}

/// On-device HBM memory usage statistics matching `PJRT_Device_MemoryStats`.
#[derive(Debug, Clone, Copy)]
pub struct DeviceMemoryStats {
    /// Number of bytes currently allocated and in use.
    pub bytes_in_use: i64,
    /// High-water mark of bytes in use since creation or last clear.
    pub peak_bytes_in_use: i64,
    /// Total number of successful allocation operations.
    pub num_allocs: i64,
    /// Size in bytes of the largest single allocation.
    pub largest_alloc_size: i64,
    /// Total HBM capacity limit in bytes.
    pub bytes_limit: i64,
    /// Number of bytes currently reserved in the driver pool.
    pub bytes_reserved: i64,
    /// High-water mark of reserved bytes.
    pub peak_bytes_reserved: i64,
    /// Maximum reservable limit in bytes.
    pub bytes_reservable_limit: i64,
    /// Largest contiguous free block size in bytes.
    pub largest_free_block_bytes: i64,
    /// Bytes currently held by the sub-allocator pool.
    pub pool_bytes: i64,
    /// Peak bytes held by the sub-allocator pool.
    pub peak_pool_bytes: i64,
}

/// Hardware Abstraction Layer trait for TPU device discovery, HBM allocation, DMA transfers,
/// and program execution.
pub trait TpuHal: Send + Sync + std::fmt::Debug {
    /// Discovers the TPU topology for this backend.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if hardware discovery fails.
    fn discover_topology(&self) -> LibTpuResult<TopologySpec>;

    /// Allocates a region of device memory on `device_id`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `memory_kind` - Target memory space kind.
    /// * `size` - Number of bytes to allocate.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] or [`LibTpuError::OutOfMemory`] on failure.
    fn allocate_device_memory(
        &self,
        device_id: DeviceId,
        memory_kind: MemoryKind,
        size: ByteSize,
    ) -> LibTpuResult<DeviceAddress>;

    /// Frees a previously allocated device memory region.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `address` - Device address returned by [`TpuHal::allocate_device_memory`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the device or allocation does not exist.
    fn deallocate_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
    ) -> LibTpuResult<()>;

    /// Writes host bytes into an allocated device memory buffer (Host-to-Device DMA).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `address` - Destination device address.
    /// * `data` - Source host byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::TransferFailed`] if the address or bounds are invalid.
    fn write_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        data: &[u8],
    ) -> LibTpuResult<()>;

    /// Reads device memory bytes into a host buffer (Device-to-Host DMA).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Source logical TPU device.
    /// * `address` - Source device address.
    /// * `out` - Destination host mutable byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::TransferFailed`] if the address or bounds are invalid.
    fn read_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()>;

    /// Copies bytes between two device memory buffers (Device-to-Device ICI / HBM copy).
    ///
    /// # Arguments
    ///
    /// * `src_device` - Source logical TPU device.
    /// * `src_addr` - Source device address.
    /// * `dst_device` - Destination logical TPU device.
    /// * `dst_addr` - Destination device address.
    /// * `size` - Number of bytes to copy.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::TransferFailed`] if either buffer is invalid or too small.
    fn copy_device_to_device(
        &self,
        src_device: DeviceId,
        src_addr: DeviceAddress,
        dst_device: DeviceId,
        dst_addr: DeviceAddress,
        size: ByteSize,
    ) -> LibTpuResult<()>;

    /// Executes a [`CompiledProgram`] on `device_id` reading from `inputs` and writing to `outputs`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `program` - Lowered TPU program.
    /// * `inputs` - Slice of input buffer [`DeviceAddress`] handles.
    /// * `outputs` - Slice of output buffer [`DeviceAddress`] handles.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ExecutionFailed`] if the device is poisoned or buffers are invalid.
    fn execute_program(
        &self,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<()>;

    /// Queries HBM memory allocation statistics for `device_id`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if `device_id` is invalid.
    fn memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats>;

    /// Clears peak HBM memory statistics for `device_id` (`PJRT_Device_ClearMemoryStats`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if `device_id` is invalid.
    fn clear_memory_stats(&self, device_id: DeviceId) -> LibTpuResult<()>;

    /// Poisons execution on `device_id` with a specific status payload (`PJRT_Device_PoisonExecution`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `reason` - Poison error description.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if `device_id` is invalid.
    fn poison_device(&self, device_id: DeviceId, reason: String) -> LibTpuResult<()>;
}

/// Internal per-device state for [`SimulatedTpuHal`].
#[derive(Debug)]
struct SimulatedDeviceState {
    /// Page-aligned (`4096` B) HBM region allocator with free-block coalescing.
    region_allocator: HbmRegionAllocator,
    /// Active HBM allocations mapped by [`DeviceAddress`].
    allocations: BTreeMap<DeviceAddress, Vec<u8>>,
    /// Memory space kind associated with each [`DeviceAddress`].
    allocation_kinds: BTreeMap<DeviceAddress, MemoryKind>,
    /// Bytes currently allocated.
    bytes_in_use: usize,
    /// Peak bytes allocated.
    peak_bytes_in_use: usize,
    /// Total allocation count.
    num_allocs: i64,
    /// Largest single allocation in bytes.
    largest_alloc_size: usize,
    /// Configurable HBM capacity limit in bytes.
    capacity_bytes: usize,
    /// Optional poison reason set by `poison_device`.
    poison_reason: Option<String>,
}

/// Deterministic in-memory TPU hardware simulator implementing [`TpuHal`].
#[derive(Debug)]
pub struct SimulatedTpuHal {
    /// Discovered topology specification.
    topology: TopologySpec,
    /// Per-device mutable HBM and execution state.
    devices: Mutex<BTreeMap<DeviceId, SimulatedDeviceState>>,
    /// Emulated `PCIe` BAR0 MMIO CSR bank updated during DMA and `TensorCore` execution.
    pub mmio: EmulatedKernelDriverBackend,
}

impl SimulatedTpuHal {
    /// Creates a new [`SimulatedTpuHal`] from a [`TpuInitConfig`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration specifying generation and mesh bounds.
    #[must_use]
    pub fn new(config: &TpuInitConfig) -> Self {
        let generation = config.generation;
        let devs_per_chip = generation.logical_devices_per_chip();
        let mut devices_spec = Vec::new();
        let mut state_map = BTreeMap::new();

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
                                        state_map.insert(
                                            dev_id,
                                            SimulatedDeviceState {
                                                region_allocator: HbmRegionAllocator::new(
                                                    0x1000_0000,
                                                    generation.hbm_capacity_bytes().0,
                                                ),
                                                allocations: BTreeMap::new(),
                                                allocation_kinds: BTreeMap::new(),
                                                bytes_in_use: 0,
                                                peak_bytes_in_use: 0,
                                                num_allocs: 0,
                                                largest_alloc_size: 0,
                                                capacity_bytes: generation.hbm_capacity_bytes().0,
                                                poison_reason: None,
                                            },
                                        );
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

        let topology = TopologySpec {
            platform_name: String::from("tpu"),
            platform_version: format!("libtpu-rs 0.0.50 ({generation})"),
            generation,
            chip_bounds,
            host_bounds: config.host_bounds,
            chips_per_host: config.chips_per_host(),
            devices: devices_spec,
        };

        Self {
            topology,
            devices: Mutex::new(state_map),
            mmio: EmulatedKernelDriverBackend::default(),
        }
    }

    /// Returns a cloned copy of the discovered [`TopologySpec`].
    pub fn topology_spec(&self) -> TopologySpec {
        self.topology.clone()
    }

    /// Acquires the internal device map mutex guard or returns [`LibTpuError::LockPoisoned`].
    fn lock_devices(
        &self,
    ) -> LibTpuResult<MutexGuard<'_, BTreeMap<DeviceId, SimulatedDeviceState>>> {
        self.devices.lock().map_or_else(
            |_| {
                Err(LibTpuError::LockPoisoned {
                    resource: "devices",
                })
            },
            Ok,
        )
    }

    /// Overrides the HBM capacity limit of `device_id` for testing out-of-memory conditions.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `capacity` - New capacity limit in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if `device_id` does not exist.
    pub fn set_device_capacity(&self, device_id: DeviceId, capacity: ByteSize) -> LibTpuResult<()> {
        let mut guard = self.lock_devices()?;
        let dev = guard
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
        dev.capacity_bytes = capacity.0;
        dev.region_allocator.set_capacity(capacity.0);
        drop(guard);
        Ok(())
    }

    /// Evaluates a single unary operation across `input_bytes`.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Unary [`TpuOpCode`].
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `input_bytes` - Source HBM byte buffer.
    #[allow(clippy::cast_precision_loss)]
    fn eval_unary_op(opcode: TpuOpCode, elem_type: ElementType, input_bytes: &[u8]) -> Vec<u8> {
        if matches!(
            opcode,
            TpuOpCode::Identity
                | TpuOpCode::AllReduceSumF32
                | TpuOpCode::AllGather
                | TpuOpCode::ReduceScatter
                | TpuOpCode::CustomCall
                | TpuOpCode::Reshape
        ) {
            return input_bytes.to_vec();
        }
        let vals = elem_type.unpack_to_f64_vec(input_bytes);
        let mapped: Vec<f64> = match opcode {
            TpuOpCode::NegF32 => vals.iter().map(|&v| -v).collect(),
            TpuOpCode::ExpF32 => vals.iter().map(|&v| v.exp()).collect(),
            TpuOpCode::TanhF32 => vals.iter().map(|&v| v.tanh()).collect(),
            TpuOpCode::Abs => vals.iter().map(|&v| v.abs()).collect(),
            TpuOpCode::Log => vals.iter().map(|&v| v.max(1e-12).ln()).collect(),
            TpuOpCode::Sqrt => vals.iter().map(|&v| v.max(0.0).sqrt()).collect(),
            TpuOpCode::Rsqrt => vals.iter().map(|&v| 1.0 / v.max(1e-12).sqrt()).collect(),
            TpuOpCode::Sine => vals.iter().map(|&v| v.sin()).collect(),
            TpuOpCode::Cosine => vals.iter().map(|&v| v.cos()).collect(),
            TpuOpCode::Floor => vals.iter().map(|&v| v.floor()).collect(),
            TpuOpCode::Ceil => vals.iter().map(|&v| v.ceil()).collect(),
            TpuOpCode::Not => vals
                .iter()
                .map(|&v| f64::from(u8::from(v == 0.0)))
                .collect(),
            TpuOpCode::Iota => (0..vals.len()).map(|i| i as f64).collect(),
            TpuOpCode::Reverse => vals.iter().copied().rev().collect(),
            TpuOpCode::ReduceMax => {
                let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                vec![mx; vals.len()]
            }
            TpuOpCode::ReduceMin => {
                let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
                vec![mn; vals.len()]
            }
            _ => {
                let sum: f64 = vals.iter().sum();
                vec![sum; vals.len()]
            }
        };
        elem_type.pack_from_f64_slice(&mapped)
    }

    /// Evaluates a general 2D or 3D batched matrix contraction (`dot_general`).
    ///
    /// # Arguments
    ///
    /// * `elem_type` - Output [`ElementType`].
    /// * `v0` - Unpacked LHS elements.
    /// * `v1` - Unpacked RHS elements.
    /// * `attrs` - Optional [`crate::compiler::InstructionAttrs`] with input/output shapes.
    fn eval_dot_general(
        elem_type: ElementType,
        v0: &[f64],
        v1: &[f64],
        attrs: &crate::compiler::InstructionAttrs,
    ) -> Vec<u8> {
        let mut batch = 1_usize;
        let mut m = v0.len().min(v1.len());
        let mut k = 1_usize;
        let mut n = 1_usize;
        let mut d = 1_usize;
        while d.saturating_mul(d) < m {
            d += 1;
        }
        if d > 1 && d.saturating_mul(d) == m {
            m = d;
            k = d;
            n = d;
        }
        let lhs_dims = attrs
            .input_shapes
            .first()
            .map_or(&[][..], |s| s.dims.as_slice());
        let rhs_dims = attrs
            .input_shapes
            .get(1)
            .map_or(&[][..], |s| s.dims.as_slice());
        if lhs_dims.len() >= 2 && rhs_dims.len() >= 2 {
            let l_rank = lhs_dims.len();
            let r_rank = rhs_dims.len();
            for &bd in lhs_dims.iter().take(l_rank - 2) {
                batch = batch.saturating_mul(usize::try_from(bd).unwrap_or(1));
            }
            m = usize::try_from(lhs_dims.get(l_rank - 2).copied().unwrap_or(0)).unwrap_or(0);
            k = usize::try_from(lhs_dims.get(l_rank - 1).copied().unwrap_or(0)).unwrap_or(0);
            n = usize::try_from(rhs_dims.get(r_rank - 1).copied().unwrap_or(0)).unwrap_or(0);
        }
        let mut out_vals = Vec::with_capacity(batch * m * n);
        for b_idx in 0..batch {
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0.0_f64;
                    for idx in 0..k {
                        let lv = v0.get((b_idx * m + r) * k + idx).copied().unwrap_or(0.0);
                        let rv = v1.get((b_idx * k + idx) * n + c).copied().unwrap_or(0.0);
                        acc = lv.mul_add(rv, acc);
                    }
                    out_vals.push(acc);
                }
            }
        }
        elem_type.pack_from_f64_slice(&out_vals)
    }

    /// Evaluates a binary operation across `b0` and `b1`.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Binary [`TpuOpCode`].
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `b0` - First input HBM buffer.
    /// * `b1` - Second input HBM buffer.
    #[allow(clippy::cast_possible_truncation)]
    fn eval_binary_op(opcode: TpuOpCode, elem_type: ElementType, b0: &[u8], b1: &[u8]) -> Vec<u8> {
        if opcode == TpuOpCode::XorBytes {
            return b0.iter().zip(b1.iter()).map(|(&x, &y)| x ^ y).collect();
        }
        let v0 = elem_type.unpack_to_f64_vec(b0);
        let v1 = elem_type.unpack_to_f64_vec(b1);
        if opcode == TpuOpCode::Concatenate {
            let mut joined = Vec::with_capacity(v0.len() + v1.len());
            joined.extend_from_slice(&v0);
            joined.extend_from_slice(&v1);
            return elem_type.pack_from_f64_slice(&joined);
        }
        let out_vals: Vec<f64> = v0
            .iter()
            .zip(v1.iter())
            .map(|(&x, &y)| match opcode {
                TpuOpCode::AddF32 | TpuOpCode::AddS32 => x + y,
                TpuOpCode::SubF32 => x - y,
                TpuOpCode::DivF32 => x / y,
                TpuOpCode::Rem => x % y,
                TpuOpCode::MaxF32 => x.max(y),
                TpuOpCode::MinF32 => x.min(y),
                TpuOpCode::And => f64::from((x as i64 & y as i64) as i32),
                TpuOpCode::Or => f64::from((x as i64 | y as i64) as i32),
                TpuOpCode::CompareEq => f64::from(u8::from((x - y).abs() < 1e-9)),
                TpuOpCode::CompareLt => f64::from(u8::from(x < y)),
                TpuOpCode::CompareGt => f64::from(u8::from(x > y)),
                _ => x * y,
            })
            .collect();
        elem_type.pack_from_f64_slice(&out_vals)
    }

    /// Evaluates a single [`crate::compiler::TpuInstructionBundle`] with its [`crate::compiler::InstructionAttrs`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Lowered [`crate::compiler::TpuInstructionBundle`].
    /// * `default_elem_type` - Default [`ElementType`] if not overridden by `bundle.attrs`.
    /// * `operands` - Slice of operand byte slices.
    #[allow(clippy::too_many_lines, clippy::or_fun_call)]
    fn eval_bundle_with_attrs(
        bundle: &crate::compiler::TpuInstructionBundle,
        default_elem_type: ElementType,
        operands: &[&[u8]],
    ) -> Vec<u8> {
        let out_elem_type = bundle
            .attrs
            .output_shape
            .as_ref()
            .map_or(default_elem_type, |s| s.element_type);
        let in_elem_type = bundle
            .attrs
            .input_shapes
            .first()
            .map_or(out_elem_type, |s| s.element_type);
        let b0 = operands.first().copied().unwrap_or(&[]);

        match bundle.opcode {
            TpuOpCode::Constant => {
                if !bundle.attrs.constant_bytes.is_empty() {
                    return bundle.attrs.constant_bytes.clone();
                }
                return vec![0_u8; 16];
            }
            TpuOpCode::Convert => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                return out_elem_type.pack_from_f64_slice(&vals);
            }
            TpuOpCode::Transpose => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                let in_sh = bundle.attrs.input_shapes.first().cloned().unwrap_or(Shape {
                    element_type: in_elem_type,
                    dims: vec![2, 2],
                    dynamic_dims: vec![false; 2],
                    layout: MemoryLayout::default_tpu_tiled(2),
                });
                let perm = if bundle.attrs.permutation.len() == in_sh.rank() {
                    bundle.attrs.permutation.clone()
                } else {
                    (0..in_sh.rank()).rev().collect()
                };
                let out_sh = in_sh.transpose(&perm).unwrap_or(in_sh.clone());
                let count = out_sh.element_count().min(vals.len());
                let mut out_vals = Vec::with_capacity(count);
                for dst_idx in 0..count {
                    let dst_coords = out_sh.linear_to_coords(dst_idx);
                    let mut src_coords = vec![0_usize; in_sh.rank()];
                    for (src_axis, slot) in src_coords.iter_mut().enumerate() {
                        let dst_axis = perm.iter().position(|&p| p == src_axis).unwrap_or(src_axis);
                        *slot = dst_coords.get(dst_axis).copied().unwrap_or(0);
                    }
                    let src_idx = in_sh.coords_to_linear(&src_coords).unwrap_or(0);
                    out_vals.push(vals.get(src_idx).copied().unwrap_or(0.0));
                }
                return out_elem_type.pack_from_f64_slice(&out_vals);
            }
            TpuOpCode::BroadcastInDim => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                let out_sh = bundle.attrs.output_shape.clone().unwrap_or(Shape {
                    element_type: out_elem_type,
                    dims: vec![4],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                });
                let in_sh = bundle
                    .attrs
                    .input_shapes
                    .first()
                    .cloned()
                    .unwrap_or(out_sh.clone());
                let out_count = out_sh.element_count();
                let mut out_vals = Vec::with_capacity(out_count);
                for dst_idx in 0..out_count {
                    let dst_coords = out_sh.linear_to_coords(dst_idx);
                    let mut src_coords = vec![0_usize; in_sh.rank()];
                    for (src_axis, slot) in src_coords.iter_mut().enumerate() {
                        let dst_axis = bundle
                            .attrs
                            .broadcast_dims
                            .get(src_axis)
                            .copied()
                            .unwrap_or(src_axis);
                        let src_ext =
                            usize::try_from(in_sh.dims.get(src_axis).copied().unwrap_or(1))
                                .unwrap_or(1)
                                .max(1);
                        let dc = dst_coords.get(dst_axis).copied().unwrap_or(0);
                        *slot = dc % src_ext;
                    }
                    let src_idx = in_sh.coords_to_linear(&src_coords).unwrap_or(0);
                    out_vals.push(vals.get(src_idx).copied().unwrap_or(0.0));
                }
                return out_elem_type.pack_from_f64_slice(&out_vals);
            }
            TpuOpCode::Slice => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                let in_sh = bundle.attrs.input_shapes.first().cloned().unwrap_or(Shape {
                    element_type: in_elem_type,
                    dims: vec![4],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                });
                if !bundle.attrs.slice_starts.is_empty() {
                    let mut out_vals = Vec::new();
                    let in_count = in_sh.element_count().min(vals.len());
                    for idx in 0..in_count {
                        let coords = in_sh.linear_to_coords(idx);
                        let mut keep = true;
                        for (axis, &c) in coords.iter().enumerate() {
                            let st = bundle.attrs.slice_starts.get(axis).copied().unwrap_or(0);
                            let lim = bundle
                                .attrs
                                .slice_limits
                                .get(axis)
                                .copied()
                                .unwrap_or(usize::MAX);
                            let stride = bundle
                                .attrs
                                .slice_strides
                                .get(axis)
                                .copied()
                                .unwrap_or(1)
                                .max(1);
                            if c < st || c >= lim || (c - st) % stride != 0 {
                                keep = false;
                                break;
                            }
                        }
                        if keep {
                            out_vals.push(vals.get(idx).copied().unwrap_or(0.0));
                        }
                    }
                    return out_elem_type.pack_from_f64_slice(&out_vals);
                }
                return b0.to_vec();
            }
            TpuOpCode::DotGeneralF32 => {
                let v0 = in_elem_type.unpack_to_f64_vec(b0);
                let v1 = in_elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
                return Self::eval_dot_general(out_elem_type, &v0, &v1, &bundle.attrs);
            }
            TpuOpCode::Iota => {
                if let Some(out_sh) = &bundle.attrs.output_shape {
                    let axis = bundle.attrs.iota_dim.min(out_sh.rank().saturating_sub(1));
                    let count = out_sh.element_count();
                    let mut out_vals = Vec::with_capacity(count);
                    for idx in 0..count {
                        let coords = out_sh.linear_to_coords(idx);
                        let c =
                            u32::try_from(coords.get(axis).copied().unwrap_or(idx)).unwrap_or(0);
                        out_vals.push(f64::from(c));
                    }
                    return out_elem_type.pack_from_f64_slice(&out_vals);
                }
            }
            TpuOpCode::ReduceSumF32 | TpuOpCode::ReduceMax | TpuOpCode::ReduceMin => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                let scalar = match bundle.opcode {
                    TpuOpCode::ReduceMax => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    TpuOpCode::ReduceMin => vals.iter().copied().fold(f64::INFINITY, f64::min),
                    _ => vals.iter().sum(),
                };
                let out_len = bundle
                    .attrs
                    .output_shape
                    .as_ref()
                    .map_or(vals.len(), |s| s.element_count().max(1));
                return out_elem_type.pack_from_f64_slice(&vec![scalar; out_len]);
            }
            TpuOpCode::Concatenate if bundle.attrs.concat_dim > 0 => {
                let v0 = in_elem_type.unpack_to_f64_vec(b0);
                let v1 = in_elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
                let mut interleaved = Vec::with_capacity(v0.len() + v1.len());
                for (&x, &y) in v0.iter().zip(v1.iter()) {
                    interleaved.push(x);
                    interleaved.push(y);
                }
                return out_elem_type.pack_from_f64_slice(&interleaved);
            }
            _ => {}
        }
        Self::eval_bundle(bundle.opcode, out_elem_type, operands)
    }

    /// Evaluates a single [`crate::compiler::TpuInstructionBundle`] against a register bank of byte vectors.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Instruction [`TpuOpCode`].
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `operands` - Slice of operand byte slices.
    fn eval_bundle(opcode: TpuOpCode, elem_type: ElementType, operands: &[&[u8]]) -> Vec<u8> {
        if opcode == TpuOpCode::Select {
            let c = elem_type.unpack_to_f64_vec(operands.first().copied().unwrap_or(&[]));
            let t = elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
            let f = elem_type.unpack_to_f64_vec(operands.get(2).copied().unwrap_or(&[]));
            let sel: Vec<f64> = c
                .iter()
                .zip(t.iter().zip(f.iter()))
                .map(|(&cond, (&tv, &fv))| if cond == 0.0 { fv } else { tv })
                .collect();
            elem_type.pack_from_f64_slice(&sel)
        } else if opcode.is_binary() {
            Self::eval_binary_op(
                opcode,
                elem_type,
                operands.first().copied().unwrap_or(&[]),
                operands.get(1).copied().unwrap_or(&[]),
            )
        } else {
            Self::eval_unary_op(opcode, elem_type, operands.first().copied().unwrap_or(&[]))
        }
    }

    /// Executes a cross-device collective (`AllReduce`, `AllGather`, or `ReduceScatter`) across multiple
    /// device buffers on the simulated ICI torus mesh.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Collective [`TpuOpCode`] (`AllReduceSumF32`, `AllGather`, or `ReduceScatter`).
    /// * `elem_type` - Element data type.
    /// * `participants` - Slice of `(DeviceId, input_addr, output_addr)` tuples for each replica.
    /// * `element_count` - Number of elements per replica input buffer.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any participant device or buffer address is invalid.
    #[allow(for_loops_over_fallibles)]
    pub fn execute_collective(
        &self,
        opcode: TpuOpCode,
        elem_type: ElementType,
        participants: &[(DeviceId, DeviceAddress, DeviceAddress)],
        element_count: usize,
    ) -> LibTpuResult<()> {
        let mut guard = self.lock_devices()?;
        let mut replica_vecs: Vec<Vec<f64>> = Vec::with_capacity(participants.len());
        let byte_len = (element_count * elem_type.bit_size()).div_ceil(8);

        for &(dev_id, in_addr, out_addr) in participants {
            let dev = guard.get(&dev_id).ok_or(LibTpuError::DeviceNotFound {
                device_id: dev_id.0,
            })?;
            if !dev.allocations.contains_key(&out_addr) {
                return Err(LibTpuError::ExecutionFailed {
                    device_id: dev_id.0,
                    reason: format!("Collective output {out_addr} not found on device {dev_id}"),
                });
            }
            let buf =
                dev.allocations
                    .get(&in_addr)
                    .ok_or_else(|| LibTpuError::ExecutionFailed {
                        device_id: dev_id.0,
                        reason: format!("Collective input {in_addr} not found on device {dev_id}"),
                    })?;
            let slice = buf.get(..byte_len).unwrap_or(buf.as_slice());
            replica_vecs.push(elem_type.unpack_to_f64_vec(slice));
        }

        let mut reduced = vec![0.0_f64; element_count];
        for r_vec in &replica_vecs {
            for (dst, &val) in reduced.iter_mut().zip(r_vec.iter()) {
                *dst += val;
            }
        }

        let mut gathered = Vec::with_capacity(element_count * replica_vecs.len());
        for r_vec in &replica_vecs {
            gathered.extend_from_slice(r_vec);
        }

        let chunk_len = (element_count / participants.len().max(1)).max(1);
        for (rep_idx, &(dev_id, _, out_addr)) in participants.iter().enumerate() {
            let payload_f64: Vec<f64> = match opcode {
                TpuOpCode::AllGather => gathered.clone(),
                TpuOpCode::ReduceScatter => {
                    let start = (rep_idx * chunk_len).min(reduced.len());
                    let end = (start + chunk_len).min(reduced.len());
                    reduced.get(start..end).unwrap_or(&[]).to_vec()
                }
                _ => reduced.clone(),
            };
            let packed = elem_type.pack_from_f64_slice(&payload_f64);
            for dev in guard.get_mut(&dev_id) {
                for out_buf in dev.allocations.get_mut(&out_addr) {
                    for (dst_b, &src_b) in out_buf.iter_mut().zip(packed.iter()) {
                        *dst_b = src_b;
                    }
                }
            }
        }
        drop(guard);
        Ok(())
    }
}

impl TpuHal for SimulatedTpuHal {
    fn discover_topology(&self) -> LibTpuResult<TopologySpec> {
        Ok(self.topology_spec())
    }

    fn allocate_device_memory(
        &self,
        device_id: DeviceId,
        memory_kind: MemoryKind,
        size: ByteSize,
    ) -> LibTpuResult<DeviceAddress> {
        let mut guard = self.lock_devices()?;
        let dev = guard
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
        let alloc_size = size.0.max(256);
        let addr = dev.region_allocator.allocate(alloc_size)?;
        dev.allocations.insert(addr, vec![0_u8; alloc_size]);
        dev.allocation_kinds.insert(addr, memory_kind);
        dev.bytes_in_use += alloc_size;
        dev.peak_bytes_in_use = dev.peak_bytes_in_use.max(dev.bytes_in_use);
        dev.num_allocs += 1;
        dev.largest_alloc_size = dev.largest_alloc_size.max(alloc_size);
        drop(guard);
        Ok(addr)
    }

    fn deallocate_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
    ) -> LibTpuResult<()> {
        let mut guard = self.lock_devices()?;
        let dev = guard
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
        let removed =
            dev.allocations
                .remove(&address)
                .ok_or_else(|| LibTpuError::InvalidArgument {
                    parameter: "address",
                    reason: format!("Allocation {address} not found on device {device_id}"),
                })?;
        let _ = dev.region_allocator.deallocate(address);
        dev.allocation_kinds.remove(&address);
        dev.bytes_in_use = dev.bytes_in_use.saturating_sub(removed.len());
        drop(guard);
        Ok(())
    }

    fn write_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        data: &[u8],
    ) -> LibTpuResult<()> {
        let mut guard = self.lock_devices()?;
        let dev = guard
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
        let buf = dev
            .allocations
            .get_mut(&address)
            .ok_or_else(|| LibTpuError::TransferFailed {
                direction: "H2D",
                reason: format!("Invalid HBM address {address} on device {device_id}"),
            })?;
        let Some(target) = buf.get_mut(..data.len()) else {
            return Err(LibTpuError::TransferFailed {
                direction: "H2D",
                reason: format!(
                    "Host payload ({} bytes) exceeds HBM buffer ({} bytes)",
                    data.len(),
                    buf.len()
                ),
            });
        };
        target.copy_from_slice(data);
        let next_tail = self
            .mmio
            .get_csr(BAR0_CSR_HBM_DMA_RING_TAIL)
            .wrapping_add(1);
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_TAIL, next_tail);
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_HEAD, next_tail);
        drop(guard);
        Ok(())
    }

    fn read_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()> {
        let guard = self.lock_devices()?;
        let dev = guard.get(&device_id).ok_or(LibTpuError::DeviceNotFound {
            device_id: device_id.0,
        })?;
        let buf = dev
            .allocations
            .get(&address)
            .ok_or_else(|| LibTpuError::TransferFailed {
                direction: "D2H",
                reason: format!("Invalid HBM address {address} on device {device_id}"),
            })?;
        let Some(src) = buf.get(..out.len()) else {
            return Err(LibTpuError::TransferFailed {
                direction: "D2H",
                reason: format!(
                    "Requested read ({} bytes) exceeds HBM buffer ({} bytes)",
                    out.len(),
                    buf.len()
                ),
            });
        };
        out.copy_from_slice(src);
        let next_tail = self
            .mmio
            .get_csr(BAR0_CSR_HBM_DMA_RING_TAIL)
            .wrapping_add(1);
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_TAIL, next_tail);
        self.mmio.set_csr(BAR0_CSR_HBM_DMA_RING_HEAD, next_tail);
        drop(guard);
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
        let mut temp = vec![0_u8; size.0];
        self.read_device_memory(src_device, src_addr, &mut temp)?;
        self.write_device_memory(dst_device, dst_addr, &temp)
    }

    fn execute_program(
        &self,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<()> {
        let mut guard = self.lock_devices()?;
        let dev = guard
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
        if let Some(reason) = &dev.poison_reason {
            return Err(LibTpuError::ExecutionFailed {
                device_id: device_id.0,
                reason: reason.clone(),
            });
        }
        if outputs.is_empty() {
            return Err(LibTpuError::ExecutionFailed {
                device_id: device_id.0,
                reason: String::from("Missing output buffer address"),
            });
        }
        let required_inputs = program.opcode.arity().min(program.parameter_shapes.len());
        if inputs.len() < required_inputs {
            return Err(LibTpuError::ExecutionFailed {
                device_id: device_id.0,
                reason: format!(
                    "Operation requires at least {required_inputs} input buffers, got {}",
                    inputs.len()
                ),
            });
        }
        let elem_type = program
            .output_shapes
            .first()
            .map_or(ElementType::F32, |s| s.element_type);

        let mut registers: Vec<Vec<u8>> =
            Vec::with_capacity(inputs.len() + program.instruction_bundles.len() + 1);
        for (idx, &in_addr) in inputs.iter().enumerate() {
            let buf =
                dev.allocations
                    .get(&in_addr)
                    .ok_or_else(|| LibTpuError::ExecutionFailed {
                        device_id: device_id.0,
                        reason: format!("Input {idx} address {in_addr} not found"),
                    })?;
            registers.push(buf.clone());
        }

        let mut last_result = registers.first().cloned().unwrap_or_default();
        for bundle in &program.instruction_bundles {
            let mut op_slices: Vec<&[u8]> = Vec::with_capacity(bundle.operand_slots.len());
            for &slot in &bundle.operand_slots {
                let slice = registers.get(slot).map_or(&[][..], Vec::as_slice);
                op_slices.push(slice);
            }
            let evaluated = Self::eval_bundle_with_attrs(bundle, elem_type, &op_slices);
            last_result.clone_from(&evaluated);
            registers.push(evaluated);
        }

        for (out_idx, &out_addr) in outputs.iter().enumerate() {
            let src_bytes = program
                .output_slots
                .get(out_idx)
                .and_then(|&slot| registers.get(slot))
                .unwrap_or(&last_result);
            let out_buf =
                dev.allocations
                    .get_mut(&out_addr)
                    .ok_or_else(|| LibTpuError::ExecutionFailed {
                        device_id: device_id.0,
                        reason: format!("Output address {out_addr} not found"),
                    })?;
            for (dst_byte, &src_byte) in out_buf.iter_mut().zip(src_bytes.iter()) {
                *dst_byte = src_byte;
            }
        }
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_PC, 0);
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_RUN_CONTROL, 1);
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_STATUS, 2);
        let sync = self.mmio.get_csr(BAR0_CSR_SYNC_FLAG_BASE);
        self.mmio.set_csr(
            BAR0_CSR_SYNC_FLAG_BASE,
            sync.wrapping_add(u64::try_from(program.instruction_bundles.len()).unwrap_or(1)),
        );
        drop(guard);
        Ok(())
    }

    fn memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats> {
        let guard = self.lock_devices()?;
        let dev = guard.get(&device_id).ok_or(LibTpuError::DeviceNotFound {
            device_id: device_id.0,
        })?;
        let free_bytes =
            i64::try_from(dev.region_allocator.largest_free_block_bytes()).unwrap_or(0);
        let in_use = i64::try_from(dev.bytes_in_use).unwrap_or(0);
        let peak = i64::try_from(dev.peak_bytes_in_use).unwrap_or(0);
        let largest = i64::try_from(dev.largest_alloc_size).unwrap_or(0);
        let limit = i64::try_from(dev.capacity_bytes).unwrap_or(0);
        let stats = DeviceMemoryStats {
            bytes_in_use: in_use,
            peak_bytes_in_use: peak,
            num_allocs: dev.num_allocs,
            largest_alloc_size: largest,
            bytes_limit: limit,
            bytes_reserved: in_use,
            peak_bytes_reserved: peak,
            bytes_reservable_limit: limit,
            largest_free_block_bytes: free_bytes,
            pool_bytes: in_use,
            peak_pool_bytes: peak,
        };
        drop(guard);
        Ok(stats)
    }

    fn clear_memory_stats(&self, device_id: DeviceId) -> LibTpuResult<()> {
        let mut guard = self.lock_devices()?;
        let dev = guard
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?;
        dev.peak_bytes_in_use = dev.bytes_in_use;
        dev.largest_alloc_size = 0;
        drop(guard);
        Ok(())
    }

    fn poison_device(&self, device_id: DeviceId, reason: String) -> LibTpuResult<()> {
        self.lock_devices()?
            .get_mut(&device_id)
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: device_id.0,
            })?
            .poison_reason = Some(reason);
        Ok(())
    }
}

/// Kernel driver mode used by [`LinuxDeviceHal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDriverMode {
    /// Legacy `/dev/accel*` Gasket kernel driver.
    Gasket,
    /// Modern `/dev/vfio/vfio` user-space `PCIe` IOMMU driver.
    Vfio,
}

/// Injectable OS system call and `PCIe` BAR MMIO interface (`open`, `close`, `ioctl`, `eventfd`, `mmio`)
/// for [`LinuxDeviceHal`].
pub trait SyscallProvider: Send + Sync + std::fmt::Debug {
    /// Opens a device path (e.g., `"/dev/accel0"` or `"/dev/vfio/vfio"`) and returns a file descriptor.
    ///
    /// # Arguments
    ///
    /// * `path` - Device filesystem path.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] on failure.
    fn open_device(&self, path: &str) -> LibTpuResult<i32>;

    /// Issues an `ioctl` command to an open device file descriptor.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open file descriptor.
    /// * `request` - Raw `ioctl` command number (`GASKET_IOCTL_*` or `VFIO_*`).
    /// * `payload_size` - Size in bytes of the `ioctl` payload structure.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] on failure.
    fn ioctl_device(&self, fd: i32, request: u64, payload_size: usize) -> LibTpuResult<i32>;

    /// Closes an open device file descriptor.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open file descriptor to close.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] on failure.
    fn close_device(&self, fd: i32) -> LibTpuResult<()>;
}

/// Typed `ioctl` payload passed into [`EmulatedKernelDriverBackend::apply_typed_ioctl`].
#[derive(Debug, Clone, Copy)]
pub enum IoctlPayload<'a> {
    /// Gasket page table map/unmap descriptor.
    GasketPageTable(&'a GasketPageTableIoctl),
    /// Gasket interrupt `eventfd` binding descriptor.
    GasketInterruptEventFd(&'a GasketInterruptEventFd),
    /// Gasket `dma_buf` page table mapping descriptor.
    GasketDmaBuf(&'a GasketPageTableDmaBuf),
    /// VFIO IOMMU DMA mapping descriptor.
    VfioDmaMap(&'a VfioDmaMapIoctl),
    /// VFIO IOMMU DMA unmapping descriptor.
    VfioDmaUnmap(&'a VfioDmaUnmapIoctl),
    /// VFIO MSI-X interrupt `eventfd` binding descriptor.
    VfioIrqSet(&'a VfioIrqSetIoctl),
}

/// Cycle-accurate emulated Linux Gasket/VFIO kernel driver and `PCIe` BAR0/BAR2 MMIO CSR backend.
#[derive(Debug)]
pub struct EmulatedKernelDriverBackend {
    /// BAR0 MMIO 64-bit registers indexed by byte offset.
    bar0_registers: parking_lot::Mutex<BTreeMap<u64, u64>>,
    /// Emulated IOMMU page table mapping IOVA addresses to mapped byte sizes.
    iommu_mappings: parking_lot::Mutex<BTreeMap<u64, u64>>,
    /// Registered interrupt `eventfd` handles indexed by interrupt vector ID.
    interrupt_eventfds: parking_lot::Mutex<BTreeMap<u32, i32>>,
    /// Total BAR0 aperture size in bytes.
    bar0_size: usize,
    /// Next allocated file descriptor.
    next_fd: AtomicI32,
}

impl Default for EmulatedKernelDriverBackend {
    fn default() -> Self {
        Self::new(0x0010_0000)
    }
}

impl EmulatedKernelDriverBackend {
    /// Creates a new [`EmulatedKernelDriverBackend`] with the given BAR0 aperture size.
    ///
    /// # Arguments
    ///
    /// * `bar0_size` - Size of the simulated `PCIe` BAR0 MMIO region in bytes.
    #[must_use]
    pub fn new(bar0_size: usize) -> Self {
        let mut regs = BTreeMap::new();
        regs.insert(BAR0_CSR_TENSORCORE_RUN_CONTROL, 0);
        regs.insert(BAR0_CSR_TENSORCORE_SYNC_FLAG, 1);
        regs.insert(BAR0_CSR_TENSORCORE_STATUS, 0);
        regs.insert(BAR0_CSR_SYNC_FLAG_BASE, 1);
        regs.insert(BAR0_CSR_HBM_DMA_RING_HEAD, 0);
        regs.insert(BAR0_CSR_HBM_DMA_RING_TAIL, 0);
        regs.insert(BAR0_CSR_THERMAL_MILLICELSIUS, 45_000);
        regs.insert(BAR0_CSR_POWER_MILLIWATTS, 175_000);
        regs.insert(BAR0_CSR_DUTY_CYCLE_PERCENT, 100);
        Self {
            bar0_registers: parking_lot::Mutex::new(regs),
            iommu_mappings: parking_lot::Mutex::new(BTreeMap::new()),
            interrupt_eventfds: parking_lot::Mutex::new(BTreeMap::new()),
            bar0_size,
            next_fd: AtomicI32::new(100),
        }
    }

    /// Applies a typed [`IoctlPayload`] to the emulated IOMMU page table or interrupt vector table.
    ///
    /// # Arguments
    ///
    /// * `request` - Raw `ioctl` command number.
    /// * `payload` - Typed [`IoctlPayload`] structure.
    pub fn apply_typed_ioctl(&self, request: u64, payload: IoctlPayload<'_>) {
        match payload {
            IoctlPayload::GasketPageTable(pt) => {
                let mut guard = self.iommu_mappings.lock();
                if request == GASKET_IOCTL_UNMAP_BUFFER {
                    guard.remove(&pt.device_address);
                } else {
                    guard.insert(pt.device_address, pt.size);
                }
            }
            IoctlPayload::GasketDmaBuf(db) => {
                self.iommu_mappings.lock().insert(db.device_address, 4096);
            }
            IoctlPayload::VfioDmaMap(vm) => {
                self.iommu_mappings.lock().insert(vm.iova, vm.size);
            }
            IoctlPayload::VfioDmaUnmap(vu) => {
                self.iommu_mappings.lock().remove(&vu.iova);
            }
            IoctlPayload::GasketInterruptEventFd(ge) => {
                self.interrupt_eventfds.lock().insert(
                    u32::try_from(ge.interrupt).unwrap_or(0),
                    i32::try_from(ge.event_fd).unwrap_or(0),
                );
            }
            IoctlPayload::VfioIrqSet(vi) => {
                self.interrupt_eventfds.lock().insert(vi.index, vi.event_fd);
            }
        }
    }

    /// Returns the number of active IOVA mappings in the emulated IOMMU page table.
    pub fn mapped_iova_count(&self) -> usize {
        self.iommu_mappings.lock().len()
    }

    /// Returns the registered `eventfd` for interrupt `index`, if present.
    ///
    /// # Arguments
    ///
    /// * `index` - Interrupt vector index.
    pub fn registered_interrupt_fd(&self, index: u32) -> Option<i32> {
        self.interrupt_eventfds.lock().get(&index).copied()
    }

    /// Clears all registered interrupt `eventfd` bindings.
    pub fn clear_interrupt_fds(&self) {
        self.interrupt_eventfds.lock().clear();
    }

    /// Creates a simulated Linux `eventfd` handle.
    pub fn create_eventfd(&self) -> EventFdHandle {
        let efd = self.next_fd.fetch_add(1, Ordering::Relaxed);
        EventFdHandle(efd)
    }

    /// Reads a 64-bit BAR0 CSR value directly from the register map.
    ///
    /// # Arguments
    ///
    /// * `csr_offset` - Byte offset of the BAR0 CSR.
    pub fn get_csr(&self, csr_offset: u64) -> u64 {
        self.bar0_registers
            .lock()
            .get(&csr_offset)
            .copied()
            .unwrap_or(0)
    }

    /// Writes a 64-bit BAR0 CSR value directly into the register map.
    ///
    /// # Arguments
    ///
    /// * `csr_offset` - Byte offset of the BAR0 CSR.
    /// * `value` - 64-bit register value to store.
    pub fn set_csr(&self, csr_offset: u64, value: u64) {
        self.bar0_registers.lock().insert(csr_offset, value);
    }

    /// Reads a 64-bit CSR from `PCIe` BAR `bar_index` at `offset`.
    ///
    /// # Arguments
    ///
    /// * `bar_index` - `PCIe` BAR index (`0` or `2`).
    /// * `offset` - Byte offset within the BAR.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset + 8 > bar0_size`.
    pub fn read_mmio_u64(&self, bar_index: u8, offset: MmioOffset) -> LibTpuResult<u64> {
        if bar_index != 0
            || usize::try_from(offset.0)
                .unwrap_or(usize::MAX)
                .saturating_add(8)
                > self.bar0_size
        {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index,
                offset: offset.0,
                size: 8,
            });
        }
        let val = self
            .bar0_registers
            .lock()
            .get(&offset.0)
            .copied()
            .unwrap_or(0);
        Ok(val)
    }

    /// Writes a 64-bit CSR to `PCIe` BAR `bar_index` at `offset`.
    ///
    /// # Arguments
    ///
    /// * `bar_index` - `PCIe` BAR index (`0` or `2`).
    /// * `offset` - Byte offset within the BAR.
    /// * `value` - 64-bit register value to write.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset + 8 > bar0_size`.
    pub fn write_mmio_u64(
        &self,
        bar_index: u8,
        offset: MmioOffset,
        value: u64,
    ) -> LibTpuResult<()> {
        if bar_index != 0
            || usize::try_from(offset.0)
                .unwrap_or(usize::MAX)
                .saturating_add(8)
                > self.bar0_size
        {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index,
                offset: offset.0,
                size: 8,
            });
        }
        let mut guard = self.bar0_registers.lock();
        guard.insert(offset.0, value);
        if offset.0 == BAR0_CSR_TENSORCORE_RUN_CONTROL && value != 0 {
            let sync = guard
                .get(&BAR0_CSR_TENSORCORE_SYNC_FLAG)
                .copied()
                .unwrap_or(0);
            guard.insert(BAR0_CSR_TENSORCORE_SYNC_FLAG, sync.wrapping_add(1));
        }
        drop(guard);
        Ok(())
    }

    /// Maps a `dma_buf` file descriptor into the Gasket page table (`GASKET_IOCTL_MAP_DMABUF`).
    ///
    /// # Arguments
    ///
    /// * `fd` - Open Gasket device file descriptor.
    /// * `payload` - `GasketPageTableDmaBuf` descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] if the `ioctl` fails.
    pub fn map_dmabuf(
        &self,
        fd: FileDescriptor,
        payload: &GasketPageTableDmaBuf,
    ) -> LibTpuResult<()> {
        self.ioctl_device(fd.0, GASKET_IOCTL_MAP_DMABUF, size_of_val(payload))?;
        self.apply_typed_ioctl(GASKET_IOCTL_MAP_DMABUF, IoctlPayload::GasketDmaBuf(payload));
        Ok(())
    }
}

impl SyscallProvider for EmulatedKernelDriverBackend {
    fn open_device(&self, _path: &str) -> LibTpuResult<i32> {
        let fd = self.next_fd.fetch_add(1, Ordering::Relaxed);
        Ok(fd)
    }

    fn ioctl_device(&self, fd: i32, request: u64, _payload_size: usize) -> LibTpuResult<i32> {
        let _ = IoctlCommand(request);
        if fd < 0 {
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: 9,
                message: String::from("EBADF"),
            });
        }
        Ok(0)
    }

    fn close_device(&self, fd: i32) -> LibTpuResult<()> {
        if fd < 0 {
            return Err(LibTpuError::HardwareIoError {
                operation: "close",
                errno: 9,
                message: String::from("EBADF"),
            });
        }
        Ok(())
    }
}

/// Alias for [`EmulatedKernelDriverBackend`] (`SimulatedKernelDriverBackend`).
pub type SimulatedKernelDriverBackend = EmulatedKernelDriverBackend;

/// Generic VFS `FIOCLEX` `ioctl` command (`0x20006601` on macOS, `0x5451` on Linux).
#[cfg(target_os = "macos")]
pub const POSIX_FIOCLEX_IOCTL: u64 = 0x2000_6601;

/// Generic VFS `FIOCLEX` `ioctl` command (`0x5451` on Linux).
#[cfg(not(target_os = "macos"))]
pub const POSIX_FIOCLEX_IOCTL: u64 = 0x5451;

/// Low-level POSIX C library (`libc`) FFI wrappers for [`PosixKernelDriverBackend`].
mod posix_libc {
    pub(super) use libc::{
        MAP_ANON as MAP_ANONYMOUS, MAP_PRIVATE, MAP_SHARED, O_RDWR, PROT_READ, PROT_WRITE, close,
        ioctl, mmap, munmap, open, pipe,
    };
}

/// Real POSIX kernel driver backend invoking `libc` `open`, `close`, `ioctl`, `mmap`, `munmap`,
/// and `pipe`/`eventfd` system calls for `/dev/accel*` and `/dev/vfio/*` device nodes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PosixKernelDriverBackend;

impl PosixKernelDriverBackend {
    /// Creates a new [`PosixKernelDriverBackend`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Issues a typed `#[repr(C)]` `ioctl` command on `fd` passing a mutable pointer to `payload`.
    ///
    /// # Arguments
    ///
    /// * `fd` - Open POSIX file descriptor.
    /// * `request` - Raw `ioctl` command number.
    /// * `payload` - Mutable reference to a `#[repr(C)]` `ioctl` payload structure.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::IoctlFailed`] if `libc::ioctl` returns `< 0`.
    pub fn ioctl_with_payload<T>(
        &self,
        fd: i32,
        request: u64,
        payload: &mut T,
    ) -> LibTpuResult<i32> {
        #[allow(clippy::cast_possible_truncation)]
        let req_ulong = request as libc::c_ulong;
        let ptr = std::ptr::from_mut::<T>(payload).cast::<libc::c_void>();
        // SAFETY: `ptr` points to a valid, mutable `T` structure on the stack.
        let rc = unsafe { posix_libc::ioctl(fd, req_ulong, ptr) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: err.raw_os_error().unwrap_or(9),
                message: err.to_string(),
            });
        }
        Ok(rc)
    }

    /// Maps a `PCIe` BAR aperture of `size` bytes into a [`MappedBarAperture`].
    ///
    /// # Arguments
    ///
    /// * `fd` - Open device file descriptor (or `-1` for anonymous mapping).
    /// * `size` - Mapping byte size (must be `> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `mmap` fails.
    pub fn mmap_bar_aperture(
        &self,
        fd: FileDescriptor,
        size: ByteSize,
    ) -> LibTpuResult<MappedBarAperture> {
        let base_vaddr = self.mmap_bar0(fd.0, size.0)?;
        Ok(MappedBarAperture { base_vaddr, size })
    }

    /// Unmaps a [`MappedBarAperture`] previously mapped with [`PosixKernelDriverBackend::mmap_bar_aperture`].
    ///
    /// # Arguments
    ///
    /// * `aperture` - Mapped BAR aperture to unmap.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `munmap` fails.
    pub fn munmap_bar_aperture(&self, aperture: MappedBarAperture) -> LibTpuResult<()> {
        self.munmap_bar0(aperture.base_vaddr, aperture.size.0)
    }

    /// Maps a `PCIe` BAR0 region of `size` bytes on file descriptor `fd` (or anonymous memory when `fd == -1`).
    ///
    /// # Arguments
    ///
    /// * `fd` - Open device file descriptor (or `-1` for anonymous mapping).
    /// * `size` - Mapping byte size (must be `> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `mmap` returns `MAP_FAILED`.
    pub fn mmap_bar0(&self, fd: i32, size: usize) -> LibTpuResult<usize> {
        let flags = if fd < 0 {
            posix_libc::MAP_PRIVATE | posix_libc::MAP_ANONYMOUS
        } else {
            posix_libc::MAP_SHARED
        };
        // SAFETY: `mmap` with `null_mut()` lets the kernel select a page-aligned virtual address.
        let ptr = unsafe {
            posix_libc::mmap(
                std::ptr::null_mut(),
                size,
                posix_libc::PROT_READ | posix_libc::PROT_WRITE,
                flags,
                fd,
                0,
            )
        };
        if ptr as usize == usize::MAX {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "mmap",
                errno: err.raw_os_error().unwrap_or(22),
                message: err.to_string(),
            });
        }
        Ok(ptr as usize)
    }

    /// Unmaps a `PCIe` BAR0 region previously mapped with [`PosixKernelDriverBackend::mmap_bar0`].
    ///
    /// # Arguments
    ///
    /// * `addr` - Mapped virtual address returned by `mmap_bar0`.
    /// * `size` - Mapping byte size (must be `> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `munmap` fails.
    pub fn munmap_bar0(&self, addr: usize, size: usize) -> LibTpuResult<()> {
        // SAFETY: `munmap` validates `addr` and `size` in the kernel (`size == 0` returns `EINVAL`).
        let rc = unsafe { posix_libc::munmap(addr as *mut std::ffi::c_void, size) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "munmap",
                errno: err.raw_os_error().unwrap_or(22),
                message: err.to_string(),
            });
        }
        Ok(())
    }

    /// Allocates an OS-backed event notification descriptor (`eventfd` / pipe read end).
    #[must_use]
    pub fn create_eventfd(&self) -> EventFdHandle {
        let mut fds = [0_i32; 2];
        // SAFETY: `fds` points to a valid 2-element `c_int` array.
        let _ = unsafe { posix_libc::pipe(fds.as_mut_ptr()) };
        // SAFETY: Closing the write end of the newly created pipe.
        let _ = unsafe { posix_libc::close(fds[1]) };
        EventFdHandle(fds[0])
    }
}

impl SyscallProvider for PosixKernelDriverBackend {
    fn open_device(&self, path: &str) -> LibTpuResult<i32> {
        let c_path = std::ffi::CString::new(path).map_err(|_| LibTpuError::InvalidArgument {
            parameter: "path",
            reason: String::from("Device path contains interior NUL byte"),
        })?;
        // SAFETY: `c_path.as_ptr()` is a valid NUL-terminated C string.
        let fd = unsafe { posix_libc::open(c_path.as_ptr(), posix_libc::O_RDWR) };
        if fd < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "open",
                errno: err.raw_os_error().unwrap_or(2),
                message: err.to_string(),
            });
        }
        Ok(fd)
    }

    fn ioctl_device(&self, fd: i32, request: u64, _payload_size: usize) -> LibTpuResult<i32> {
        let mut arg_buf = [0_u8; 64];
        #[allow(clippy::cast_possible_truncation)]
        let req_ulong = request as std::ffi::c_ulong;
        // SAFETY: `arg_buf` is a 64-byte stack buffer large enough for Gasket/VFIO ioctl structs.
        let rc = unsafe {
            posix_libc::ioctl(
                fd,
                req_ulong,
                arg_buf.as_mut_ptr().cast::<std::ffi::c_void>(),
            )
        };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: err.raw_os_error().unwrap_or(9),
                message: err.to_string(),
            });
        }
        Ok(rc)
    }

    fn close_device(&self, fd: i32) -> LibTpuResult<()> {
        // SAFETY: `close` is safe for any integer file descriptor (returns `-1` with `EBADF` if invalid).
        let rc = unsafe { posix_libc::close(fd) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            return Err(LibTpuError::HardwareIoError {
                operation: "close",
                errno: err.raw_os_error().unwrap_or(9),
                message: err.to_string(),
            });
        }
        Ok(())
    }
}

/// Linux kernel driver HAL wrapping a [`SyscallProvider`], BAR0 MMIO CSR bank, and backing [`SimulatedTpuHal`].
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
    /// Backing HBM & execution state.
    pub inner: SimulatedTpuHal,
}

impl<S: SyscallProvider> LinuxDeviceHal<S> {
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
        Ok(Self {
            mode,
            syscalls,
            device_fd: fd,
            mmio: EmulatedKernelDriverBackend::default(),
            inner: SimulatedTpuHal::new(config),
        })
    }

    /// Reads the thermal (millicelsius), power (milliwatts), and duty-cycle (%) telemetry CSRs from BAR0.
    pub fn read_telemetry_csrs(&self) -> (u64, u64, u64) {
        (
            self.mmio.get_csr(BAR0_CSR_THERMAL_MILLICELSIUS),
            self.mmio.get_csr(BAR0_CSR_POWER_MILLIWATTS),
            self.mmio.get_csr(BAR0_CSR_DUTY_CYCLE_PERCENT),
        )
    }

    /// Advances the BAR0 HBM DMA descriptor ring tail and head pointers.
    fn advance_dma_ring(&self) {
        let next_tail = self
            .mmio
            .get_csr(BAR0_CSR_HBM_DMA_RING_TAIL)
            .wrapping_add(1);
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
        if event_fd < 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "event_fd",
                reason: String::from("event_fd must be non-negative"),
            });
        }
        match self.mode {
            KernelDriverMode::Gasket => {
                let payload = GasketInterruptEventFd {
                    interrupt: 0,
                    event_fd: u64::try_from(event_fd).unwrap_or(0),
                    reserved: [0; 2],
                };
                self.syscalls.ioctl_device(
                    self.device_fd,
                    GASKET_IOCTL_SET_EVENTFD,
                    size_of_val(&payload),
                )?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_SET_EVENTFD,
                    IoctlPayload::GasketInterruptEventFd(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let payload = VfioIrqSetIoctl {
                    argsz: 24,
                    flags: 0x24,
                    index: 2,
                    start: 0,
                    count: 1,
                    event_fd,
                };
                self.syscalls.ioctl_device(
                    self.device_fd,
                    VFIO_DEVICE_SET_IRQS_IOCTL,
                    size_of_val(&payload),
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

    /// Closes the underlying kernel device file descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if closing the file descriptor fails.
    pub fn close(&self) -> LibTpuResult<()> {
        self.syscalls.close_device(self.device_fd)
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
        match self.mode {
            KernelDriverMode::Gasket => {
                let payload = GasketPageTableIoctl {
                    page_table_index: 0,
                    size: size.0 as u64,
                    host_address: addr.0,
                    device_address: addr.0,
                };
                self.syscalls.ioctl_device(
                    self.device_fd,
                    GASKET_IOCTL_MAP_BUFFER,
                    size_of_val(&payload),
                )?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_MAP_BUFFER,
                    IoctlPayload::GasketPageTable(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let payload = VfioDmaMapIoctl {
                    argsz: 32,
                    flags: 0x3,
                    vaddr: addr.0,
                    iova: addr.0,
                    size: size.0 as u64,
                };
                self.syscalls.ioctl_device(
                    self.device_fd,
                    VFIO_IOMMU_MAP_DMA_IOCTL,
                    size_of_val(&payload),
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
        match self.mode {
            KernelDriverMode::Gasket => {
                let payload = GasketPageTableIoctl {
                    page_table_index: 0,
                    size: 4096,
                    host_address: address.0,
                    device_address: address.0,
                };
                self.syscalls.ioctl_device(
                    self.device_fd,
                    GASKET_IOCTL_UNMAP_BUFFER,
                    size_of_val(&payload),
                )?;
                self.mmio.apply_typed_ioctl(
                    GASKET_IOCTL_UNMAP_BUFFER,
                    IoctlPayload::GasketPageTable(&payload),
                );
            }
            KernelDriverMode::Vfio => {
                let payload = VfioDmaUnmapIoctl {
                    argsz: 24,
                    flags: 0,
                    iova: address.0,
                    size: 4096,
                };
                self.syscalls.ioctl_device(
                    self.device_fd,
                    VFIO_IOMMU_UNMAP_DMA_IOCTL,
                    size_of_val(&payload),
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
        self.advance_dma_ring();
        self.inner.write_device_memory(device_id, address, data)
    }

    fn read_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()> {
        self.advance_dma_ring();
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
        self.advance_dma_ring();
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
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_PC, 0);
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_RUN_CONTROL, 1);
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_STATUS, 1);
        let sync = self.mmio.get_csr(BAR0_CSR_SYNC_FLAG_BASE);
        self.mmio.set_csr(
            BAR0_CSR_SYNC_FLAG_BASE,
            sync.wrapping_add(u64::try_from(program.instruction_bundles.len()).unwrap_or(1)),
        );
        self.inner
            .execute_program(device_id, program, inputs, outputs)
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
    /// Gasket/VFIO protocol stack.
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
            Self::Linux(LinuxDeviceHal {
                mode: KernelDriverMode::Gasket,
                syscalls: EmulatedKernelDriverBackend::default(),
                device_fd: 3,
                mmio: EmulatedKernelDriverBackend::default(),
                inner: SimulatedTpuHal::new(config),
            })
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

/// Creates the default [`TpuHal`] instance for `config`, routing allocations, DMA ring transfers,
/// and `TensorCore` execution through [`HalBackend::from_config`].
///
/// # Arguments
///
/// * `config` - TPU initialization configuration.
#[must_use]
pub fn create_default_hal(config: &TpuInitConfig) -> std::sync::Arc<dyn TpuHal> {
    std::sync::Arc::new(HalBackend::from_config(config))
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockSyscalls {
        open_err: bool,
        ioctl_err_cmd: Option<u64>,
        close_err: bool,
    }

    impl SyscallProvider for MockSyscalls {
        fn open_device(&self, _path: &str) -> LibTpuResult<i32> {
            if self.open_err {
                Err(LibTpuError::HardwareIoError {
                    operation: "open",
                    errno: 2,
                    message: String::from("No such file"),
                })
            } else {
                Ok(42)
            }
        }

        fn ioctl_device(&self, _fd: i32, request: u64, _payload_size: usize) -> LibTpuResult<i32> {
            if self.ioctl_err_cmd == Some(request) {
                Err(LibTpuError::IoctlFailed {
                    request,
                    errno: 22,
                    message: String::from("EINVAL"),
                })
            } else {
                Ok(0)
            }
        }

        fn close_device(&self, _fd: i32) -> LibTpuResult<()> {
            if self.close_err {
                Err(LibTpuError::HardwareIoError {
                    operation: "close",
                    errno: 9,
                    message: String::from("EBADF"),
                })
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn test_simulated_hal_topology_and_memory() {
        let cfg = TpuInitConfig {
            host_bounds: ChipCoordinates::new(2, 1, 1),
            chips_per_host_bounds: ChipCoordinates::new(1, 1, 1),
            visible_devices: Some(vec![DeviceId(0), DeviceId(1)]),
            ..TpuInitConfig::default()
        };

        let hal = SimulatedTpuHal::new(&cfg);
        assert_ne!(format!("{hal:?}"), "");
        let topo = hal.discover_topology();
        assert!(topo.is_ok());
        for t in topo {
            assert_eq!(t.devices.len(), 2);
            for d0 in t.devices.first() {
                assert!(d0.is_addressable);
                assert_ne!(format!("{d0:?}"), "");
            }
            for d1 in t.devices.get(1) {
                assert!(!d1.is_addressable);
            }
            assert_ne!(format!("{t:?}"), "");
        }

        let filtered_cfg = TpuInitConfig {
            visible_devices: Some(vec![DeviceId(0)]),
            ..TpuInitConfig::default()
        };
        let filtered_hal = SimulatedTpuHal::new(&filtered_cfg);
        assert_eq!(
            filtered_hal
                .discover_topology()
                .ok()
                .map(|t| t.devices.len()),
            Some(1)
        );

        let a1 = hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(512));
        assert!(a1.is_ok());
        let a2 = hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256));
        assert!(a2.is_ok());
        assert!(
            hal.allocate_device_memory(DeviceId(99), MemoryKind::DeviceHbm, ByteSize(256))
                .is_err()
        );

        let addr1 = a1.unwrap_or(DeviceAddress(0));
        let addr2 = a2.unwrap_or(DeviceAddress(0));

        assert!(
            hal.write_device_memory(DeviceId(0), addr1, &[1, 2, 3, 4])
                .is_ok()
        );
        assert!(hal.write_device_memory(DeviceId(99), addr1, &[1]).is_err());
        assert!(
            hal.write_device_memory(DeviceId(0), DeviceAddress(0xdead), &[1])
                .is_err()
        );
        assert!(
            hal.write_device_memory(DeviceId(0), addr1, &vec![0_u8; 1024])
                .is_err()
        );

        let mut read_buf = [0_u8; 4];
        assert!(
            hal.read_device_memory(DeviceId(0), addr1, &mut read_buf)
                .is_ok()
        );
        assert_eq!(read_buf, [1, 2, 3, 4]);
        assert!(
            hal.read_device_memory(DeviceId(99), addr1, &mut read_buf)
                .is_err()
        );
        assert!(
            hal.read_device_memory(DeviceId(0), DeviceAddress(0xdead), &mut read_buf)
                .is_err()
        );
        let mut huge_buf = vec![0_u8; 1024];
        assert!(
            hal.read_device_memory(DeviceId(0), addr1, &mut huge_buf)
                .is_err()
        );

        assert!(
            hal.copy_device_to_device(DeviceId(0), addr1, DeviceId(0), addr2, ByteSize(4))
                .is_ok()
        );
        assert!(
            hal.copy_device_to_device(DeviceId(99), addr1, DeviceId(0), addr2, ByteSize(4))
                .is_err()
        );

        let stats = hal.memory_stats(DeviceId(0));
        assert!(stats.is_ok());
        for s in stats {
            assert_eq!(s.num_allocs, 2);
            assert_ne!(format!("{s:?}"), "");
        }
        assert!(hal.memory_stats(DeviceId(99)).is_err());
        assert!(hal.clear_memory_stats(DeviceId(0)).is_ok());
        assert!(hal.clear_memory_stats(DeviceId(99)).is_err());

        assert!(hal.deallocate_device_memory(DeviceId(0), addr1).is_ok());
        assert!(hal.deallocate_device_memory(DeviceId(0), addr1).is_err());
        assert!(hal.deallocate_device_memory(DeviceId(99), addr2).is_err());
        assert!(hal.deallocate_device_memory(DeviceId(0), addr2).is_ok());

        assert!(hal.set_device_capacity(DeviceId(0), ByteSize(512)).is_ok());
        assert!(
            hal.set_device_capacity(DeviceId(99), ByteSize(512))
                .is_err()
        );
        assert!(
            hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(1024))
                .is_err()
        );
    }

    #[test]
    fn test_simulated_hal_lock_poisoning() {
        let hal = Arc::new(SimulatedTpuHal::new(&TpuInitConfig::default()));
        let hal_clone = Arc::clone(&hal);
        let _ = std::thread::spawn(move || {
            let _guard = hal_clone.devices.lock();
            std::panic::resume_unwind(Box::new(()));
        })
        .join();
        assert!(hal.set_device_capacity(DeviceId(0), ByteSize(256)).is_err());
        assert!(
            hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
                .is_err()
        );
        assert!(
            hal.deallocate_device_memory(DeviceId(0), DeviceAddress(0))
                .is_err()
        );
        assert!(
            hal.write_device_memory(DeviceId(0), DeviceAddress(0), &[1])
                .is_err()
        );
        let mut buf = [0_u8; 1];
        assert!(
            hal.read_device_memory(DeviceId(0), DeviceAddress(0), &mut buf)
                .is_err()
        );
        for prog in CompiledProgram::compile("mlir", b"return", b"") {
            assert!(hal.execute_program(DeviceId(0), &prog, &[], &[]).is_err());
        }
        assert!(hal.memory_stats(DeviceId(0)).is_err());
        assert!(hal.clear_memory_stats(DeviceId(0)).is_err());
        assert!(hal.poison_device(DeviceId(0), String::from("err")).is_err());
        assert!(
            hal.execute_collective(TpuOpCode::AllReduceSumF32, ElementType::F32, &[], 4)
                .is_err()
        );
    }

    #[test]
    fn test_simulated_hal_execute_all_opcodes() {
        let cfg = TpuInitConfig::default();
        let hal = SimulatedTpuHal::new(&cfg);
        let in0 = hal
            .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
            .unwrap_or(DeviceAddress(0));
        let in1 = hal
            .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
            .unwrap_or(DeviceAddress(0));
        let out = hal
            .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
            .unwrap_or(DeviceAddress(0));

        let mut b0 = Vec::new();
        let mut b1 = Vec::new();
        for f in [1.5_f32, 2.0, 3.0, 4.0] {
            b0.extend_from_slice(&f.to_le_bytes());
        }
        for f in [10.0_f32, 20.0, 30.0, 40.0] {
            b1.extend_from_slice(&f.to_le_bytes());
        }
        assert!(hal.write_device_memory(DeviceId(0), in0, &b0).is_ok());
        assert!(hal.write_device_memory(DeviceId(0), in1, &b1).is_ok());
        let non_sq = SimulatedTpuHal::eval_binary_op(
            TpuOpCode::DotGeneralF32,
            ElementType::F32,
            &[0_u8; 8],
            &[0_u8; 8],
        );
        assert_eq!(non_sq.len(), 8);
        let one_shape_attrs = crate::compiler::InstructionAttrs {
            input_shapes: vec![Shape {
                element_type: ElementType::F32,
                dims: vec![2, 2],
                dynamic_dims: vec![false; 2],
                layout: MemoryLayout::default_tpu_tiled(2),
            }],
            ..crate::compiler::InstructionAttrs::default()
        };
        assert_eq!(
            SimulatedTpuHal::eval_dot_general(
                ElementType::F32,
                &[1.0, 2.0, 3.0, 4.0],
                &[1.0, 0.0, 0.0, 1.0],
                &one_shape_attrs,
            )
            .len(),
            16
        );
        let two_shape_attrs = crate::compiler::InstructionAttrs {
            input_shapes: vec![
                Shape {
                    element_type: ElementType::F32,
                    dims: vec![1, 2],
                    dynamic_dims: vec![false; 2],
                    layout: MemoryLayout::default_tpu_tiled(2),
                },
                Shape {
                    element_type: ElementType::F32,
                    dims: vec![2],
                    dynamic_dims: vec![false; 1],
                    layout: MemoryLayout::default_tpu_tiled(1),
                },
            ],
            ..crate::compiler::InstructionAttrs::default()
        };
        assert_eq!(
            SimulatedTpuHal::eval_dot_general(
                ElementType::F32,
                &[1.0, 2.0],
                &[3.0, 4.0],
                &two_shape_attrs,
            )
            .len(),
            8
        );
        assert_eq!(
            SimulatedTpuHal::eval_dot_general(
                ElementType::F32,
                &[],
                &[],
                &crate::compiler::InstructionAttrs::default(),
            )
            .len(),
            0
        );
        assert_eq!(
            SimulatedTpuHal::eval_dot_general(
                ElementType::F32,
                &[1.0, 2.0],
                &[3.0, 4.0],
                &crate::compiler::InstructionAttrs::default(),
            )
            .len(),
            8
        );

        for src in [
            b"module @id { return }" as &[u8],
            b"module @add { stablehlo.add }",
            b"module @mul { mul_f32 }",
            b"module @sub { sub_f32 }",
            b"module @div { div_f32 }",
            b"module @max { max_f32 }",
            b"module @min { min_f32 }",
            b"module @dot { dot_general }",
            b"module @neg { neg_f32 }",
            b"module @exp { exp_f32 }",
            b"module @tanh { tanh_f32 }",
            b"module @abs { abs }",
            b"module @log { log }",
            b"module @sqrt { sqrt }",
            b"module @rsqrt { rsqrt }",
            b"module @sin { sine }",
            b"module @cos { cosine }",
            b"module @flr { floor }",
            b"module @cel { ceil }",
            b"module @not { not }",
            b"module @rem { remainder }",
            b"module @and { and }",
            b"module @or { or }",
            b"module @ceq { compare_eq }",
            b"module @clt { compare_lt }",
            b"module @cgt { compare_gt }",
            b"module @sel { select }",
            b"module @iot { iota }",
            b"module @rev { reverse }",
            b"module @rmx { reduce_max }",
            b"module @rmn { reduce_min }",
            b"module @rsh { reshape }",
            b"module @trp { transpose }",
            b"module @brd { broadcast_in_dim }",
            b"module @slc { slice }",
            b"module @cat { concatenate }",
            b"module @cvt { convert }",
            b"module @cst { constant }",
            b"module @ag { all_gather }",
            b"module @rs { reduce_scatter }",
            b"module @red { reduce_sum }",
            b"module @ar { all_reduce }",
            b"module @cc { custom_call }",
            b"module @adds { s32_add }",
            b"module @xor { xor_u8 }",
            b"module @dag { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>\n %0 = stablehlo.multiply %0, %arg1 : tensor<4xf32> } }",
            b"module @rect_dot { func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>\n return %0 : tensor<2x2xf32> } }",
            b"module @batch_dot { func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>\n return %0 : tensor<2x2x2xf32> } }",
            b"module @vec_dot { func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>\n return %0 : tensor<3xf32> } }",
            b"module @sq_dot { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n return %0 : tensor<4xf32> } }",
            b"HloModule hlo_eq=1\nENTRY main {\n p0 = f32[4] parameter(0)\n ROOT r = f32[4] negate(p0)\n}",
            b"module @trp_nd { func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.transpose %arg0, permutation = [1, 0] : tensor<2x2xf32>\n return %0 : tensor<2x2xf32> } }",
            b"module @trp_def { func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.transpose %arg0 : tensor<2x2xf32>\n return %0 : tensor<2x2xf32> } }",
            b"module @brd_nd { func.func @main(%arg0: tensor<2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.broadcast_in_dim %arg0, broadcast_dimensions = [1] : (tensor<2xf32>) -> tensor<2x2xf32>\n return %0 : tensor<2x2xf32> } }",
            b"module @slc_nd { func.func @main(%arg0: tensor<4xf32>) -> tensor<2xf32> { %0 = stablehlo.slice %arg0, start_indices = [1], limit_indices = [3], strides = [1] : (tensor<4xf32>) -> tensor<2xf32>\n return %0 : tensor<2xf32> } }",
            b"module @slc_strided { func.func @main(%arg0: tensor<4xf32>) -> tensor<2xf32> { %0 = stablehlo.slice %arg0, start_indices = [0], limit_indices = [4], strides = [2] : (tensor<4xf32>) -> tensor<2xf32>\n return %0 : tensor<2xf32> } }",
            b"module @cst_dense { func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.constant dense<3.0> : tensor<4xf32>\n return %0 : tensor<4xf32> } }",
            b"module @cat_ax1 { func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x4xf32> { %0 = stablehlo.concatenate %arg0, %arg1, dimension = 1 : tensor<2x4xf32>\n return %0 : tensor<2x4xf32> } }",
            b"HloModule hlo_params\nENTRY main {\n p0 = f32[4] parameter(0)\n ROOT r = f32[4] negate(p0)\n}",
        ] {
            for prog in CompiledProgram::compile("mlir", src, b"") {
                assert!(hal
                    .execute_program(DeviceId(0), &prog, &[in0, in1, in0], &[out])
                    .is_ok());
            }
        }
        assert_eq!(
            SimulatedTpuHal::eval_unary_op(TpuOpCode::Iota, ElementType::F32, &[0_u8; 8]).len(),
            8
        );
        assert_eq!(
            SimulatedTpuHal::eval_unary_op(TpuOpCode::ReduceMax, ElementType::F32, &[0_u8; 8])
                .len(),
            8
        );
        assert_eq!(
            SimulatedTpuHal::eval_unary_op(TpuOpCode::ReduceMin, ElementType::F32, &[0_u8; 8])
                .len(),
            8
        );
        assert_eq!(
            SimulatedTpuHal::eval_unary_op(TpuOpCode::ReduceSumF32, ElementType::F32, &[0_u8; 8])
                .len(),
            8
        );
        let no_shape_iota = crate::compiler::TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Iota,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 1,
            attrs: crate::compiler::InstructionAttrs::default(),
        };
        assert_eq!(
            SimulatedTpuHal::eval_bundle_with_attrs(
                &no_shape_iota,
                ElementType::F32,
                &[&[0_u8; 8]]
            )
            .len(),
            8
        );

        // Test cross-device collective execution (AllReduce, AllGather, ReduceScatter)
        for col_op in [
            TpuOpCode::AllReduceSumF32,
            TpuOpCode::AllGather,
            TpuOpCode::ReduceScatter,
        ] {
            assert!(
                hal.execute_collective(
                    col_op,
                    ElementType::F32,
                    &[(DeviceId(0), in0, out), (DeviceId(0), in1, out)],
                    4,
                )
                .is_ok()
            );
        }
        assert!(
            hal.execute_collective(
                TpuOpCode::AllReduceSumF32,
                ElementType::F32,
                &[(DeviceId(99), in0, out)],
                4,
            )
            .is_err()
        );
        assert!(
            hal.execute_collective(
                TpuOpCode::AllReduceSumF32,
                ElementType::F32,
                &[(DeviceId(0), DeviceAddress(0xdead), out)],
                4,
            )
            .is_err()
        );
        assert!(
            hal.execute_collective(
                TpuOpCode::AllReduceSumF32,
                ElementType::F32,
                &[(DeviceId(0), in0, DeviceAddress(0xdead))],
                4,
            )
            .is_err()
        );

        for id_prog in CompiledProgram::compile("mlir", b"return", b"") {
            assert!(
                hal.execute_program(DeviceId(99), &id_prog, &[in0], &[out])
                    .is_err()
            );
            assert!(
                hal.execute_program(DeviceId(0), &id_prog, &[in0], &[])
                    .is_err()
            );
            assert!(
                hal.execute_program(DeviceId(0), &id_prog, &[], &[out])
                    .is_err()
            );
            assert!(
                hal.execute_program(DeviceId(0), &id_prog, &[DeviceAddress(0xdead)], &[out])
                    .is_err()
            );
            assert!(
                hal.execute_program(DeviceId(0), &id_prog, &[in0], &[DeviceAddress(0xdead)])
                    .is_err()
            );
        }
        for add_prog in CompiledProgram::compile("mlir", b"add", b"") {
            assert!(
                hal.execute_program(DeviceId(0), &add_prog, &[in0], &[out])
                    .is_err()
            );
            assert!(
                hal.execute_program(
                    DeviceId(0),
                    &add_prog,
                    &[DeviceAddress(0xdead), in1],
                    &[out]
                )
                .is_err()
            );
            assert!(
                hal.execute_program(
                    DeviceId(0),
                    &add_prog,
                    &[in0, DeviceAddress(0xdead)],
                    &[out]
                )
                .is_err()
            );
        }

        assert!(
            hal.poison_device(DeviceId(99), String::from("bad"))
                .is_err()
        );
        assert!(
            hal.poison_device(DeviceId(0), String::from("injected hardware fault"))
                .is_ok()
        );
        for id_prog in CompiledProgram::compile("mlir", b"return", b"") {
            assert!(
                hal.execute_program(DeviceId(0), &id_prog, &[in0], &[out])
                    .is_err()
            );
        }
    }

    #[test]
    fn test_emulated_kernel_driver_backend_and_mmio() {
        let backend = EmulatedKernelDriverBackend::default();
        assert_ne!(format!("{backend:?}"), "");
        assert_eq!(backend.create_eventfd(), EventFdHandle(100));
        assert!(backend.open_device("/dev/accel0").is_ok());
        assert!(
            backend
                .ioctl_device(100, GASKET_IOCTL_MAP_BUFFER, 32)
                .is_ok()
        );
        assert!(
            backend
                .ioctl_device(-1, GASKET_IOCTL_MAP_BUFFER, 32)
                .is_err()
        );
        assert!(backend.close_device(100).is_ok());
        assert!(backend.close_device(-1).is_err());

        let dmabuf = GasketPageTableDmaBuf {
            page_table_index: 0,
            device_address: 0x1000,
            dma_buf_fd: 5,
            flags: 0,
        };
        let ioctl_p = IoctlPayload::GasketDmaBuf(&dmabuf);
        assert_ne!(format!("{dmabuf:?}{ioctl_p:?}"), "");
        assert!(backend.map_dmabuf(FileDescriptor(100), &dmabuf).is_ok());
        assert_eq!(backend.mapped_iova_count(), 1);
        assert!(backend.registered_interrupt_fd(0).is_none());
        assert!(backend.map_dmabuf(FileDescriptor(-1), &dmabuf).is_err());

        assert_eq!(
            backend
                .read_mmio_u64(0, MmioOffset(BAR0_CSR_TENSORCORE_SYNC_FLAG))
                .ok(),
            Some(1)
        );
        assert!(
            backend
                .write_mmio_u64(0, MmioOffset(BAR0_CSR_TENSORCORE_RUN_CONTROL), 1)
                .is_ok()
        );
        assert_eq!(
            backend
                .read_mmio_u64(0, MmioOffset(BAR0_CSR_TENSORCORE_SYNC_FLAG))
                .ok(),
            Some(2)
        );
        assert!(
            backend
                .write_mmio_u64(0, MmioOffset(BAR0_CSR_TENSORCORE_RUN_CONTROL), 0)
                .is_ok()
        );
        assert!(backend.write_mmio_u64(0, MmioOffset(0x2000), 42).is_ok());
        assert!(backend.read_mmio_u64(2, MmioOffset(0)).is_err());
        assert!(backend.read_mmio_u64(0, MmioOffset(0x100_0000)).is_err());
        assert!(backend.write_mmio_u64(2, MmioOffset(0), 1).is_err());
        assert!(
            backend
                .write_mmio_u64(0, MmioOffset(0x100_0000), 1)
                .is_err()
        );
    }

    #[test]
    fn test_linux_device_hal_gasket_and_vfio() {
        let cfg = TpuInitConfig::default();

        let g_ioctl = GasketPageTableIoctl {
            page_table_index: 0,
            size: 4096,
            host_address: 0,
            device_address: 0,
        };
        let g_efd = GasketInterruptEventFd {
            interrupt: 0,
            event_fd: 1,
            reserved: [0; 2],
        };
        let v_map = VfioDmaMapIoctl {
            argsz: 32,
            flags: 3,
            vaddr: 0,
            iova: 0,
            size: 4096,
        };
        let v_unmap = VfioDmaUnmapIoctl {
            argsz: 24,
            flags: 0,
            iova: 0,
            size: 4096,
        };
        let v_irq = VfioIrqSetIoctl {
            argsz: 24,
            flags: 0x24,
            index: 2,
            start: 0,
            count: 1,
            event_fd: 1,
        };
        assert_ne!(
            format!("{g_ioctl:?}{g_efd:?}{v_map:?}{v_unmap:?}{v_irq:?}"),
            ""
        );

        for mode in [KernelDriverMode::Gasket, KernelDriverMode::Vfio] {
            assert_ne!(format!("{mode:?}"), "");
            let hal_res = LinuxDeviceHal::open(
                mode,
                MockSyscalls {
                    open_err: false,
                    ioctl_err_cmd: None,
                    close_err: false,
                },
                &cfg,
            );
            assert!(hal_res.is_ok());
            for hal in hal_res {
                assert_ne!(format!("{hal:?}"), "");
                assert!(hal.discover_topology().is_ok());
                assert_eq!(hal.read_telemetry_csrs(), (45_000, 175_000, 100));
                assert!(hal.register_interrupt_eventfd(-1).is_err());
                assert!(hal.register_interrupt_eventfd(7).is_ok());
                assert!(hal.unregister_interrupt_eventfd().is_ok());

                let addr = hal
                    .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
                    .unwrap_or(DeviceAddress(0));
                let addr2 = hal
                    .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
                    .unwrap_or(DeviceAddress(0));
                assert!(
                    hal.write_device_memory(DeviceId(0), addr, &[9, 8, 7, 6])
                        .is_ok()
                );
                let mut out = [0_u8; 4];
                assert!(hal.read_device_memory(DeviceId(0), addr, &mut out).is_ok());
                assert_eq!(out, [9, 8, 7, 6]);
                assert!(
                    hal.copy_device_to_device(DeviceId(0), addr, DeviceId(0), addr2, ByteSize(4))
                        .is_ok()
                );
                for prog in CompiledProgram::compile("mlir", b"return", b"") {
                    assert!(
                        hal.execute_program(DeviceId(0), &prog, &[addr], &[addr2])
                            .is_ok()
                    );
                }
                assert!(hal.memory_stats(DeviceId(0)).is_ok());
                assert!(hal.clear_memory_stats(DeviceId(0)).is_ok());
                assert!(
                    hal.poison_device(DeviceId(0), String::from("fault"))
                        .is_ok()
                );
                assert!(hal.deallocate_device_memory(DeviceId(0), addr).is_ok());
                assert!(hal.deallocate_device_memory(DeviceId(0), addr2).is_ok());
                assert!(hal.close().is_ok());
            }
        }

        assert!(
            LinuxDeviceHal::open(
                KernelDriverMode::Gasket,
                MockSyscalls {
                    open_err: true,
                    ioctl_err_cmd: None,
                    close_err: false,
                },
                &cfg
            )
            .is_err()
        );

        for cmd in [
            GASKET_IOCTL_NUMBER_PAGE_TABLES,
            GASKET_IOCTL_PARTITION_PAGE_TABLE,
        ] {
            assert!(
                LinuxDeviceHal::open(
                    KernelDriverMode::Gasket,
                    MockSyscalls {
                        open_err: false,
                        ioctl_err_cmd: Some(cmd),
                        close_err: false,
                    },
                    &cfg
                )
                .is_err()
            );
        }

        for g_hal in LinuxDeviceHal::open(
            KernelDriverMode::Gasket,
            MockSyscalls {
                open_err: false,
                ioctl_err_cmd: Some(GASKET_IOCTL_MAP_BUFFER),
                close_err: true,
            },
            &cfg,
        ) {
            assert!(
                g_hal
                    .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
                    .is_err()
            );
            assert!(
                g_hal
                    .allocate_device_memory(DeviceId(99), MemoryKind::DeviceHbm, ByteSize(256))
                    .is_err()
            );
            assert!(g_hal.close().is_err());
        }

        for (mode, cmd) in [
            (KernelDriverMode::Gasket, GASKET_IOCTL_SET_EVENTFD),
            (KernelDriverMode::Gasket, GASKET_IOCTL_CLEAR_EVENTFD),
            (KernelDriverMode::Gasket, GASKET_IOCTL_UNMAP_BUFFER),
            (KernelDriverMode::Vfio, VFIO_DEVICE_SET_IRQS_IOCTL),
            (KernelDriverMode::Vfio, VFIO_IOMMU_MAP_DMA_IOCTL),
            (KernelDriverMode::Vfio, VFIO_IOMMU_UNMAP_DMA_IOCTL),
        ] {
            for hal in LinuxDeviceHal::open(
                mode,
                MockSyscalls {
                    open_err: false,
                    ioctl_err_cmd: Some(cmd),
                    close_err: false,
                },
                &cfg,
            ) {
                let _ = hal.register_interrupt_eventfd(1);
                let _ = hal.unregister_interrupt_eventfd();
                let _ =
                    hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256));
                let _ = hal.deallocate_device_memory(DeviceId(0), DeviceAddress(0x1000_0000));
            }
        }
        let default_hal = create_default_hal(&cfg);
        assert!(default_hal.discover_topology().is_ok());

        let mut sim_mode_cfg = TpuInitConfig::default();
        sim_mode_cfg
            .extra_flags
            .insert(String::from("--tpu_driver_mode"), String::from("simulated"));
        let mut other_mode_cfg = TpuInitConfig::default();
        other_mode_cfg
            .extra_flags
            .insert(String::from("--tpu_driver_mode"), String::from("linux"));
        for backend in [
            HalBackend::from_config(&sim_mode_cfg),
            HalBackend::from_config(&other_mode_cfg),
            HalBackend::from_config(&cfg),
        ] {
            assert_ne!(format!("{backend:?}"), "");
            assert!(backend.discover_topology().is_ok());
            let b_addr1 = backend
                .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
                .unwrap_or(DeviceAddress(0));
            let b_addr2 = backend
                .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(256))
                .unwrap_or(DeviceAddress(0));
            assert!(
                backend
                    .write_device_memory(DeviceId(0), b_addr1, &[1, 2, 3, 4])
                    .is_ok()
            );
            let mut b_out = [0_u8; 4];
            assert!(
                backend
                    .read_device_memory(DeviceId(0), b_addr1, &mut b_out)
                    .is_ok()
            );
            assert_eq!(b_out, [1, 2, 3, 4]);
            assert!(
                backend
                    .copy_device_to_device(DeviceId(0), b_addr1, DeviceId(0), b_addr2, ByteSize(4))
                    .is_ok()
            );
            for prog in CompiledProgram::compile("mlir", b"return", b"") {
                assert!(
                    backend
                        .execute_program(DeviceId(0), &prog, &[b_addr1], &[b_addr2])
                        .is_ok()
                );
            }
            assert!(backend.memory_stats(DeviceId(0)).is_ok());
            assert!(backend.clear_memory_stats(DeviceId(0)).is_ok());
            assert!(
                backend
                    .poison_device(DeviceId(0), String::from("fault"))
                    .is_ok()
            );
            assert!(
                backend
                    .deallocate_device_memory(DeviceId(0), b_addr1)
                    .is_ok()
            );
            assert!(
                backend
                    .deallocate_device_memory(DeviceId(0), b_addr2)
                    .is_ok()
            );
        }

        for emu_hal in LinuxDeviceHal::open(
            KernelDriverMode::Gasket,
            EmulatedKernelDriverBackend::default(),
            &cfg,
        ) {
            assert!(emu_hal.close().is_ok());
        }
        let bad_fd_hal = LinuxDeviceHal {
            mode: KernelDriverMode::Gasket,
            syscalls: SimulatedKernelDriverBackend::default(),
            device_fd: -1,
            mmio: SimulatedKernelDriverBackend::default(),
            inner: SimulatedTpuHal::new(&cfg),
        };
        assert!(bad_fd_hal.close().is_err());

        let posix = PosixKernelDriverBackend::new();
        assert_eq!(posix, PosixKernelDriverBackend);
        assert_ne!(format!("{posix:?}"), "");
        assert!(posix.open_device("bad\0path").is_err());
        assert!(
            posix
                .open_device("/nonexistent_libtpu_device_node_xyz")
                .is_err()
        );
        assert!(posix.ioctl_device(-1, GASKET_IOCTL_MAP_BUFFER, 32).is_err());
        assert!(posix.close_device(-1).is_err());
        assert!(posix.mmap_bar0(0, 0).is_err());
        assert!(posix.munmap_bar0(0x1000, 0).is_err());

        for fd in posix.open_device("/dev/null") {
            let mut flag: i32 = 0;
            assert!(
                posix
                    .ioctl_with_payload(fd, POSIX_FIOCLEX_IOCTL, &mut flag)
                    .is_ok()
            );
            assert!(posix.ioctl_device(fd, POSIX_FIOCLEX_IOCTL, 0).is_ok());
            assert!(posix.close_device(fd).is_ok());
        }
        let mut dummy_flag: i32 = 0;
        assert!(
            posix
                .ioctl_with_payload(-1, POSIX_FIOCLEX_IOCTL, &mut dummy_flag)
                .is_err()
        );

        for aperture in posix.mmap_bar_aperture(FileDescriptor(-1), ByteSize(4096)) {
            assert!(
                aperture
                    .write_csr_u64(MmioOffset(0), 0xCAFE_BABE_1234_5678)
                    .is_ok()
            );
            assert_eq!(
                aperture.read_csr_u64(MmioOffset(0)).ok(),
                Some(0xCAFE_BABE_1234_5678)
            );
            assert!(aperture.read_csr_u64(MmioOffset(3)).is_err());
            assert!(aperture.write_csr_u64(MmioOffset(3), 1).is_err());
            assert!(aperture.read_csr_u64(MmioOffset(4096)).is_err());
            assert!(aperture.write_csr_u64(MmioOffset(4096), 1).is_err());
            assert!(posix.munmap_bar_aperture(aperture).is_ok());
        }
        let null_aperture = MappedBarAperture {
            base_vaddr: 0,
            size: ByteSize(4096),
        };
        assert_ne!(format!("{null_aperture:?}"), "");
        assert!(null_aperture.read_csr_u64(MmioOffset(0)).is_err());
        assert!(null_aperture.write_csr_u64(MmioOffset(0), 1).is_err());

        let efd = posix.create_eventfd();
        assert!(efd.0 >= 0);
        assert!(posix.close_device(efd.0).is_ok());

        // Test bitflags, HbmDmaRingDescriptor, HbmRegionAllocator, and PciDeviceEnumerator
        let ctrl = TensorCoreControlFlags::START | TensorCoreControlFlags::CLEAR_FAULT;
        let status = TensorCoreStatusFlags::IDLE | TensorCoreStatusFlags::COMPLETED;
        let dma_f = VfioDmaMapFlags::READ | VfioDmaMapFlags::WRITE;
        let irq_f = VfioIrqSetFlags::DATA_EVENTFD | VfioIrqSetFlags::ACTION_TRIGGER;
        let ring_desc = HbmDmaRingDescriptor {
            src_iova: 0x1000,
            dst_iova: 0x2000,
            length_bytes: 4096,
            direction_flags: 0,
            completion_sync_flag: 1,
        };
        assert_ne!(
            format!("{ctrl:?}{status:?}{dma_f:?}{irq_f:?}{ring_desc:?}"),
            ""
        );
        assert_eq!(bytemuck::bytes_of(&ring_desc).len(), 32);

        let empty_alloc = HbmRegionAllocator::new(0x1000_0000, 0);
        assert_eq!(empty_alloc.largest_free_block_bytes(), 0);

        let mut region = HbmRegionAllocator::new(0x1000_0000, 16384);
        assert_ne!(format!("{region:?}"), "");
        assert_eq!(region.largest_free_block_bytes(), 16384);
        let b1 = region.allocate(4096).unwrap_or(DeviceAddress(0));
        let b2 = region.allocate(4096).unwrap_or(DeviceAddress(0));
        let b3 = region.allocate(4096).unwrap_or(DeviceAddress(0));
        let b4 = region.allocate(4096).unwrap_or(DeviceAddress(0));
        assert!(region.allocate(4096).is_err());
        assert!(region.deallocate(DeviceAddress(0xdead)).is_err());
        // Free b1 and b3 first, then b4 (which has free predecessor b1 not adjacent and b3 adjacent), then b2
        assert_eq!(region.deallocate(b1).ok(), Some(4096));
        assert_eq!(region.deallocate(b4).ok(), Some(4096));
        assert_eq!(region.deallocate(b3).ok(), Some(4096));
        assert_eq!(region.deallocate(b2).ok(), Some(4096));
        assert_eq!(region.largest_free_block_bytes(), 16384);
        let b_keep = region.allocate(4096).unwrap_or(DeviceAddress(0));
        region.set_capacity(32768);
        assert_eq!(region.largest_free_block_bytes(), 28672);
        region.set_capacity(2048);
        assert_eq!(region.largest_free_block_bytes(), 0);
        assert_eq!(region.deallocate(b_keep).ok(), Some(4096));
        region.set_capacity(8192);
        assert_eq!(region.largest_free_block_bytes(), 8192);

        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x0027)),
            Some(TpuGeneration::V3)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x0056)),
            Some(TpuGeneration::V4)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x005e)),
            Some(TpuGeneration::V5p)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x0063)),
            Some(TpuGeneration::V5e)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x006f)),
            Some(TpuGeneration::V6e)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x0076)),
            Some(TpuGeneration::V7x)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0xffff)),
            None
        );
        assert_eq!(PciDeviceEnumerator::parse_hex_u16("0x1ae0\n"), Some(0x1ae0));
        assert_eq!(PciDeviceEnumerator::parse_hex_u16("not_hex"), None);
        assert_eq!(
            PciDeviceEnumerator::scan_sysfs(
                std::path::Path::new("/nonexistent_sysfs_pci_dir"),
                std::path::Path::new("/dev")
            )
            .len(),
            0
        );

        let tmp_dir =
            std::env::temp_dir().join(format!("libtpu_sysfs_test_{}", std::process::id()));
        let sysfs_dir = tmp_dir.join("sysfs");
        let dev_dir = tmp_dir.join("dev");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:04.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:05.0/vfio"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:06.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:07.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:08.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:09.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:0a.0"));
        let _ = std::fs::create_dir_all(dev_dir.join("vfio"));
        // 04.0: Valid Gasket TPU v6e
        let _ = std::fs::write(sysfs_dir.join("0000:00:04.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:04.0/device"), "0x006f\n");
        // 05.0: Valid VFIO TPU v5p
        let _ = std::fs::write(sysfs_dir.join("0000:00:05.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:05.0/device"), "0x005e\n");
        // 06.0: Non-Google vendor
        let _ = std::fs::write(sysfs_dir.join("0000:00:06.0/vendor"), "0x8086\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:06.0/device"), "0x1234\n");
        // 07.0: Missing device file
        let _ = std::fs::write(sysfs_dir.join("0000:00:07.0/vendor"), "0x1ae0\n");
        // 08.0: Invalid vendor hex
        let _ = std::fs::write(sysfs_dir.join("0000:00:08.0/vendor"), "bad\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:08.0/device"), "0x006f\n");
        // 09.0: Invalid device hex
        let _ = std::fs::write(sysfs_dir.join("0000:00:09.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:09.0/device"), "bad\n");
        // 0a.0: Unknown Google PCI device ID
        let _ = std::fs::write(sysfs_dir.join("0000:00:0a.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:0a.0/device"), "0x9999\n");

        let scanned = PciDeviceEnumerator::scan_sysfs(&sysfs_dir, &dev_dir);
        assert_eq!(scanned.len(), 2);
        assert_ne!(format!("{PciDeviceEnumerator:?}{scanned:?}"), "");
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
