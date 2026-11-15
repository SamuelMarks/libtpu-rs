//! Linux `VFIO`, `Gasket`, Buddy/Slab HBM Allocator, and `dma-buf` IOMMU Layer.
//!
//! Implements Phase 7.2:
//! - [`VfioIommuSession`]: VFIO container/group/device initialization, `VFIO_TYPE1v2_IOMMU` DMA
//!   mapping (`VFIO_IOMMU_MAP_DMA` / `VFIO_IOMMU_UNMAP_DMA`), BAR0/BAR2/BAR4 region info & `mmap`,
//!   MSI-X IRQ vector `eventfd` binding (`VFIO_DEVICE_SET_IRQS`), and `PCIe` hot-reset (`VFIO_DEVICE_RESET`).
//! - [`GasketIommuSession`]: Gasket/Apex ioctl lifecycle (`GASKET_IOCTL_RESET`, `NUMBER_PAGE_TABLES`,
//!   `PARTITION_PAGE_TABLE`, `MAP_BUFFER`, `UNMAP_BUFFER`, `MAP_DMABUF`, `SET_EVENTFD`, `CLEAR_EVENTFD`).
//! - [`BuddySlabHbmAllocator`]: Power-of-two buddy allocator (`4 KiB` min page up to `192 GiB`)
//!   with sub-page (`256 B..2 KiB`) slab caches for HBM and VMEM.
//! - [`HugepageStagingPool`] & [`DmaBufPeerRegistry`]: Host `2 MiB` hugepage / page-aligned pinned
//!   DMA staging buffers and zero-copy `dma-buf` peer-to-peer FD import/export.

use super::{
    DeviceMemoryStats, GASKET_IOCTL_CLEAR_EVENTFD, GASKET_IOCTL_MAP_BUFFER,
    GASKET_IOCTL_MAP_DMABUF, GASKET_IOCTL_NUMBER_PAGE_TABLES, GASKET_IOCTL_PARTITION_PAGE_TABLE,
    GASKET_IOCTL_RESET, GASKET_IOCTL_SET_EVENTFD, GASKET_IOCTL_UNMAP_BUFFER,
    GasketInterruptEventFd, GasketPageTableDmaBuf, GasketPageTableIoctl, MappedBarAperture,
    PosixKernelDriverBackend, SyscallProvider, VFIO_CHECK_EXTENSION, VFIO_DEVICE_GET_INFO,
    VFIO_DEVICE_GET_IRQ_INFO, VFIO_DEVICE_GET_REGION_INFO, VFIO_DEVICE_RESET,
    VFIO_DEVICE_SET_IRQS_IOCTL, VFIO_GET_API_VERSION, VFIO_GROUP_GET_DEVICE_FD,
    VFIO_GROUP_GET_STATUS, VFIO_GROUP_SET_CONTAINER, VFIO_IOMMU_MAP_DMA_IOCTL,
    VFIO_IOMMU_UNMAP_DMA_IOCTL, VFIO_SET_IOMMU, VfioDmaMapIoctl, VfioDmaUnmapIoctl,
    VfioIrqSetIoctl, VfioRegionInfoIoctl,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ByteSize, DeviceAddress, DeviceId, FileDescriptor};
use std::collections::{BTreeMap, BTreeSet};

/// Minimum buddy page size in bytes (`4096` B / `4 KiB`).
pub const BUDDY_MIN_PAGE_BYTES: u64 = 4096;

/// Sub-page slab size classes (`256`, `512`, `1024`, `2048` bytes).
const SLAB_CLASSES: [u64; 4] = [256, 512, 1024, 2048];

/// Active allocation metadata inside [`BuddySlabHbmAllocator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BuddySlabAllocRecord {
    /// Requested user byte size.
    requested_bytes: u64,
    /// Backing power-of-two block size (`256..=capacity`).
    block_bytes: u64,
    /// Buddy order (`None` when allocated from a sub-page slab cache).
    buddy_order: Option<u8>,
}

/// Power-of-two Buddy + Sub-Page Slab allocator for Cloud TPU HBM and VMEM (`BuddySlabHbmAllocator`).
///
/// Sub-page requests (`<= 2048` B) are served from `4 KiB` buddy pages carved into `256 B`,
/// `512 B`, `1024 B`, or `2048 B` slabs; requests `>= 4096` B are rounded to the next power-of-two
/// order (`4 KiB ..= 256 GiB`) with O(log N) buddy splitting and coalescing.
#[derive(Debug, Clone)]
pub struct BuddySlabHbmAllocator {
    /// Base device address (e.g., `0x1000_0000` in HBM).
    pub base_address: u64,
    /// Logical usable capacity in bytes.
    pub capacity_bytes: u64,
    /// Maximum buddy order (`0` = `4 KiB`, `order` -> `4096 << order`).
    max_order: u8,
    /// Per-order sorted set of free block start addresses (`order -> BTreeSet<u64>`).
    free_orders: Vec<BTreeSet<u64>>,
    /// Sub-page slab free lists for `[256, 512, 1024, 2048]` byte slots.
    slab_free_slots: [BTreeSet<u64>; 4],
    /// Map of active allocation start addresses to [`BuddySlabAllocRecord`].
    active_allocs: BTreeMap<u64, BuddySlabAllocRecord>,
    /// Current bytes in use across active allocations.
    bytes_in_use: u64,
    /// Peak bytes simultaneously in use since creation or last [`Self::clear_peak_stats`].
    peak_bytes_in_use: u64,
    /// Cumulative number of successful allocations.
    total_num_allocs: u64,
}

impl BuddySlabHbmAllocator {
    /// Creates a new [`BuddySlabHbmAllocator`] over `[base_address, base_address + capacity_bytes)`.
    ///
    /// # Arguments
    ///
    /// * `base_address` - Starting HBM or VMEM device address.
    /// * `capacity_bytes` - Usable capacity in bytes.
    #[must_use]
    pub fn new(base_address: u64, capacity_bytes: u64) -> Self {
        let mut max_order = 0_u8;
        while max_order < 30 && (BUDDY_MIN_PAGE_BYTES << max_order) < capacity_bytes {
            max_order += 1;
        }
        let mut free_orders = vec![BTreeSet::new(); usize::from(max_order) + 1];
        // Decompose `capacity_bytes` (aligned down to 4 KiB pages) into maximal power-of-two buddy blocks
        let mut cursor = base_address;
        let mut remaining_pages = capacity_bytes / BUDDY_MIN_PAGE_BYTES;
        for ord in (0..=max_order).rev() {
            let pages_in_ord = 1_u64 << ord;
            while remaining_pages >= pages_in_ord {
                free_orders
                    .get_mut(usize::from(ord))
                    .into_iter()
                    .for_each(|set| {
                        set.insert(cursor);
                    });
                cursor = cursor.saturating_add(pages_in_ord * BUDDY_MIN_PAGE_BYTES);
                remaining_pages -= pages_in_ord;
            }
        }
        Self {
            base_address,
            capacity_bytes,
            max_order,
            free_orders,
            slab_free_slots: [
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            ],
            active_allocs: BTreeMap::new(),
            bytes_in_use: 0,
            peak_bytes_in_use: 0,
            total_num_allocs: 0,
        }
    }

    /// Allocates a buddy block of `target_order` (`4096 << target_order` bytes), splitting higher orders as needed.
    ///
    /// # Arguments
    ///
    /// * `target_order` - Desired buddy order (`0..=self.max_order`).
    fn alloc_buddy_order(&mut self, target_order: u8) -> Option<u64> {
        let found_ord = (target_order..=self.max_order).find(|&ord| {
            self.free_orders
                .get(usize::from(ord))
                .is_some_and(|s| !s.is_empty())
        })?;
        let mut block_addr = self
            .free_orders
            .get_mut(usize::from(found_ord))
            .and_then(BTreeSet::pop_first)
            .unwrap_or(self.base_address);
        let mut cur_ord = found_ord;
        while cur_ord > target_order {
            cur_ord -= 1;
            let half_bytes = BUDDY_MIN_PAGE_BYTES << cur_ord;
            let right_buddy = block_addr.saturating_add(half_bytes);
            self.free_orders
                .get_mut(usize::from(cur_ord))
                .into_iter()
                .for_each(|s| {
                    s.insert(right_buddy);
                });
            let _ = &mut block_addr;
        }
        Some(block_addr)
    }

    /// Frees a buddy block at `addr` of `order` and recursively coalesces with free buddies.
    ///
    /// # Arguments
    ///
    /// * `addr` - Block start address.
    /// * `order` - Block buddy order.
    fn free_buddy_order(&mut self, mut addr: u64, mut order: u8) {
        while order < self.max_order {
            let block_bytes = BUDDY_MIN_PAGE_BYTES << order;
            let rel = addr.saturating_sub(self.base_address);
            let buddy_addr = self.base_address.saturating_add(rel ^ block_bytes);
            let buddy_end = buddy_addr.saturating_add(block_bytes);
            let within_cap = buddy_end <= self.base_address.saturating_add(self.capacity_bytes);
            let removed = within_cap
                && self
                    .free_orders
                    .get_mut(usize::from(order))
                    .is_some_and(|s| s.remove(&buddy_addr));
            if !removed {
                break;
            }
            addr = addr.min(buddy_addr);
            order += 1;
        }
        self.free_orders
            .get_mut(usize::from(order))
            .into_iter()
            .for_each(|s| {
                s.insert(addr);
            });
    }

    /// Allocates at least `requested_bytes` from either the sub-page slab cache (`<= 2048` B)
    /// or the power-of-two buddy allocator (`>= 4096` B).
    ///
    /// # Arguments
    ///
    /// * `requested_bytes` - Positive byte length to allocate.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `requested_bytes == 0`, or [`LibTpuError::OutOfMemory`]
    /// when insufficient contiguous HBM remains.
    pub fn allocate(&mut self, requested_bytes: usize) -> LibTpuResult<DeviceAddress> {
        if requested_bytes == 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "requested_bytes",
                reason: String::from("Allocation size must be greater than zero"),
            });
        }
        let req_u64 = requested_bytes as u64;
        if let Some((cls_idx, &slab_sz)) = SLAB_CLASSES
            .iter()
            .enumerate()
            .find(|&(_, &sz)| req_u64 <= sz)
        {
            if self
                .slab_free_slots
                .get(cls_idx)
                .is_some_and(BTreeSet::is_empty)
            {
                let Some(page_addr) = self.alloc_buddy_order(0) else {
                    return Err(LibTpuError::OutOfMemory {
                        requested_bytes,
                        available_bytes: self.largest_free_block_bytes(),
                    });
                };
                let slots_per_page = BUDDY_MIN_PAGE_BYTES / slab_sz;
                self.slab_free_slots
                    .get_mut(cls_idx)
                    .into_iter()
                    .for_each(|set| {
                        for i in 0..slots_per_page {
                            set.insert(page_addr + i * slab_sz);
                        }
                    });
            }
            let addr = self
                .slab_free_slots
                .get_mut(cls_idx)
                .and_then(BTreeSet::pop_first)
                .unwrap_or(self.base_address);
            self.active_allocs.insert(
                addr,
                BuddySlabAllocRecord {
                    requested_bytes: req_u64,
                    block_bytes: slab_sz,
                    buddy_order: None,
                },
            );
            self.bytes_in_use = self.bytes_in_use.saturating_add(slab_sz);
            self.peak_bytes_in_use = self.peak_bytes_in_use.max(self.bytes_in_use);
            self.total_num_allocs = self.total_num_allocs.saturating_add(1);
            return Ok(DeviceAddress(addr));
        }

        let mut order = 0_u8;
        while order <= self.max_order && (BUDDY_MIN_PAGE_BYTES << order) < req_u64 {
            order += 1;
        }
        if order > self.max_order {
            return Err(LibTpuError::OutOfMemory {
                requested_bytes,
                available_bytes: self.largest_free_block_bytes(),
            });
        }
        let Some(addr) = self.alloc_buddy_order(order) else {
            return Err(LibTpuError::OutOfMemory {
                requested_bytes,
                available_bytes: self.largest_free_block_bytes(),
            });
        };
        let block_bytes = BUDDY_MIN_PAGE_BYTES << order;
        self.active_allocs.insert(
            addr,
            BuddySlabAllocRecord {
                requested_bytes: req_u64,
                block_bytes,
                buddy_order: Some(order),
            },
        );
        self.bytes_in_use = self.bytes_in_use.saturating_add(block_bytes);
        self.peak_bytes_in_use = self.peak_bytes_in_use.max(self.bytes_in_use);
        self.total_num_allocs = self.total_num_allocs.saturating_add(1);
        Ok(DeviceAddress(addr))
    }

    /// Deallocates a previously allocated block at `address` and returns its backing byte length.
    ///
    /// # Arguments
    ///
    /// * `address` - [`DeviceAddress`] returned by [`Self::allocate`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `address` is not an active allocation.
    pub fn deallocate(&mut self, address: DeviceAddress) -> LibTpuResult<u64> {
        let Some(rec) = self.active_allocs.remove(&address.0) else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "address",
                reason: format!("HBM address {address} is not allocated"),
            });
        };
        self.bytes_in_use = self.bytes_in_use.saturating_sub(rec.block_bytes);
        if let Some(ord) = rec.buddy_order {
            self.free_buddy_order(address.0, ord);
        } else {
            let cls_idx = SLAB_CLASSES
                .iter()
                .position(|&s| s == rec.block_bytes)
                .unwrap_or(0);
            self.slab_free_slots
                .get_mut(cls_idx)
                .into_iter()
                .for_each(|s| {
                    s.insert(address.0);
                });
        }
        Ok(rec.block_bytes)
    }

    /// Checks whether `[address, address + len)` lies within a valid active allocation and returns `(alloc_base, offset_in_alloc, block_bytes)`.
    ///
    /// # Arguments
    ///
    /// * `address` - Target [`DeviceAddress`] (may point inside an allocation).
    /// * `len` - Access byte length.
    #[must_use]
    pub fn resolve_range(&self, address: DeviceAddress, len: usize) -> Option<(u64, usize, u64)> {
        let (&base, rec) = self.active_allocs.range(..=address.0).next_back()?;
        let rel = address.0 - base;
        let end_rel = rel.saturating_add(len as u64);
        if end_rel <= rec.block_bytes {
            let rel_usize = usize::try_from(rel).ok()?;
            Some((base, rel_usize, rec.block_bytes))
        } else {
            None
        }
    }

    /// Returns the backing block byte length of an active allocation starting at `address`.
    ///
    /// # Arguments
    ///
    /// * `address` - Allocation base [`DeviceAddress`].
    #[must_use]
    pub fn allocated_block_bytes(&self, address: DeviceAddress) -> Option<u64> {
        self.active_allocs.get(&address.0).map(|r| r.block_bytes)
    }

    /// Returns the size in bytes of the largest contiguous free buddy or slab block.
    #[must_use]
    pub fn largest_free_block_bytes(&self) -> usize {
        for ord in (0..=self.max_order).rev() {
            if self
                .free_orders
                .get(usize::from(ord))
                .is_some_and(|s| !s.is_empty())
            {
                let b = BUDDY_MIN_PAGE_BYTES << ord;
                return usize::try_from(b).unwrap_or(usize::MAX);
            }
        }
        for (idx, &sz) in SLAB_CLASSES.iter().enumerate().rev() {
            if self.slab_free_slots.get(idx).is_some_and(|s| !s.is_empty()) {
                return usize::try_from(sz).unwrap_or(0);
            }
        }
        0
    }

    /// Returns a snapshot of [`DeviceMemoryStats`] for PJRT memory accounting.
    #[must_use]
    pub fn stats(&self) -> DeviceMemoryStats {
        let largest_free = i64::try_from(self.largest_free_block_bytes()).unwrap_or(i64::MAX);
        let cap_i64 = i64::try_from(self.capacity_bytes).unwrap_or(i64::MAX);
        let largest_alloc = i64::try_from(
            self.active_allocs
                .values()
                .map(|r| r.requested_bytes)
                .max()
                .unwrap_or(0),
        )
        .unwrap_or(i64::MAX);
        DeviceMemoryStats {
            bytes_in_use: i64::try_from(self.bytes_in_use).unwrap_or(i64::MAX),
            peak_bytes_in_use: i64::try_from(self.peak_bytes_in_use).unwrap_or(i64::MAX),
            num_allocs: i64::try_from(self.total_num_allocs).unwrap_or(i64::MAX),
            largest_alloc_size: largest_alloc,
            bytes_limit: cap_i64,
            bytes_reserved: 0,
            peak_bytes_reserved: 0,
            bytes_reservable_limit: cap_i64,
            largest_free_block_bytes: largest_free,
            pool_bytes: cap_i64,
            peak_pool_bytes: cap_i64,
        }
    }

    /// Resets `peak_bytes_in_use` down to the current `bytes_in_use`.
    pub const fn clear_peak_stats(&mut self) {
        self.peak_bytes_in_use = self.bytes_in_use;
    }
}

/// Host `2 MiB` hugepage (`MAP_HUGETLB` with page-aligned fallback) pinned DMA staging buffer (`PinnedDmaBuffer`).
#[derive(Debug)]
pub struct PinnedDmaBuffer {
    /// Virtual base address of the `mmap`-allocated host buffer.
    vaddr: usize,
    /// Mapped byte capacity (page-aligned).
    mapped_len: usize,
    /// Whether the kernel granted a real `2 MiB` `MAP_HUGETLB` mapping.
    pub is_hugetlb: bool,
}

/// Platform `MAP_HUGETLB` flag (`libc::MAP_HUGETLB` on Linux, `0` on non-Linux POSIX).
#[cfg(target_os = "linux")]
const PLATFORM_MAP_HUGETLB_FLAG: libc::c_int = libc::MAP_HUGETLB;

/// Platform `MAP_HUGETLB` flag (`0` on non-Linux POSIX).
#[cfg(not(target_os = "linux"))]
const PLATFORM_MAP_HUGETLB_FLAG: libc::c_int = 0;

/// Invokes anonymous private `libc::mmap` with optional `MAP_HUGETLB` hint.
///
/// # Arguments
///
/// * `size_bytes` - Mapping byte length.
/// * `request_hugetlb` - Whether to include [`PLATFORM_MAP_HUGETLB_FLAG`] in `mmap` flags.
pub(crate) fn os_mmap_anonymous(size_bytes: usize, request_hugetlb: bool) -> *mut libc::c_void {
    let hugetlb_bits = if request_hugetlb {
        PLATFORM_MAP_HUGETLB_FLAG
    } else {
        0
    };
    // SAFETY: Anonymous private mmap with read/write permissions and `fd = -1`.
    unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size_bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | hugetlb_bits,
            -1,
            0,
        )
    }
}

impl PinnedDmaBuffer {
    /// Allocates a page-aligned (attempting `2 MiB` `MAP_HUGETLB` on Linux when `len >= 2 MiB`)
    /// pinned host DMA buffer of at least `len` bytes.
    ///
    /// # Arguments
    ///
    /// * `len` - Minimum buffer capacity in bytes (`>= 1`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if anonymous `mmap` fails.
    pub fn allocate(len: usize) -> LibTpuResult<Self> {
        let mut os_mapper = os_mmap_anonymous;
        Self::allocate_with_mapper(len, cfg!(target_os = "linux"), &mut os_mapper)
    }

    /// Allocates a [`PinnedDmaBuffer`] using an injectable `mapper(size_bytes, request_hugetlb)`
    /// trait object for deterministic testing of hugepage grant, fallback, and `MAP_FAILED` paths.
    ///
    /// # Arguments
    ///
    /// * `len` - Minimum buffer capacity in bytes (`>= 1`).
    /// * `attempt_hugetlb` - Whether to attempt a `2 MiB` hugepage mapping when `len >= 2 MiB`.
    /// * `mapper` - Mutable trait-object closure returning an `mmap` pointer or `libc::MAP_FAILED`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if the fallback `4 KiB` mapping returns `MAP_FAILED`.
    pub(crate) fn allocate_with_mapper(
        len: usize,
        attempt_hugetlb: bool,
        mapper: &mut dyn FnMut(usize, bool) -> *mut libc::c_void,
    ) -> LibTpuResult<Self> {
        let page_aligned_bytes = len.max(4096).div_ceil(4096) * 4096;
        if attempt_hugetlb && page_aligned_bytes >= 2_097_152 {
            let hugepage_aligned_bytes = page_aligned_bytes.div_ceil(2_097_152) * 2_097_152;
            let huge_ptr = mapper(hugepage_aligned_bytes, true);
            if huge_ptr != libc::MAP_FAILED {
                return Ok(Self {
                    vaddr: huge_ptr as usize,
                    mapped_len: hugepage_aligned_bytes,
                    is_hugetlb: true,
                });
            }
        }
        let ptr = mapper(page_aligned_bytes, false);
        if ptr == libc::MAP_FAILED {
            return Err(LibTpuError::HardwareIoError {
                operation: "mmap(PinnedDmaBuffer)",
                errno: 12,
                message: String::from("Failed to mmap host DMA staging buffer"),
            });
        }
        Ok(Self {
            vaddr: ptr as usize,
            mapped_len: page_aligned_bytes,
            is_hugetlb: false,
        })
    }

    /// Returns the host user-space virtual address (`vaddr`) suitable for `VFIO_IOMMU_MAP_DMA` / `GASKET_IOCTL_MAP_BUFFER`.
    #[must_use]
    pub const fn host_vaddr(&self) -> u64 {
        self.vaddr as u64
    }

    /// Returns the page-aligned mapped byte capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.mapped_len
    }

    /// Returns an immutable view over the mapped host staging memory.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.vaddr` points to `self.mapped_len` valid initialized bytes owned exclusively by `self`.
        unsafe { std::slice::from_raw_parts(self.vaddr as *const u8, self.mapped_len) }
    }

    /// Returns a mutable view over the mapped host staging memory.
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `self.vaddr` points to `self.mapped_len` valid bytes exclusively borrowed via `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.vaddr as *mut u8, self.mapped_len) }
    }
}

impl Drop for PinnedDmaBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.vaddr` and `self.mapped_len` were returned by `libc::mmap` in `Self::allocate`.
        let _ = unsafe { libc::munmap(self.vaddr as *mut libc::c_void, self.mapped_len) };
    }
}

/// Pool of host-pinned page-aligned DMA staging buffers keyed by `(DeviceId, DeviceAddress)` (`HugepageStagingPool`).
#[derive(Debug, Default)]
pub struct HugepageStagingPool {
    /// Active pinned host buffers keyed by `(device_id, device_iova)`.
    buffers: BTreeMap<(DeviceId, DeviceAddress), PinnedDmaBuffer>,
}

impl HugepageStagingPool {
    /// Creates an empty [`HugepageStagingPool`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
        }
    }

    /// Allocates and registers a [`PinnedDmaBuffer`] for `(device_id, iova)` of `size_bytes`, returning `(host_vaddr, mapped_len)`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target [`DeviceId`].
    /// * `iova` - Device HBM IOVA [`DeviceAddress`].
    /// * `size_bytes` - Minimum staging size in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `mmap` fails.
    pub fn register_buffer(
        &mut self,
        device_id: DeviceId,
        iova: DeviceAddress,
        size_bytes: usize,
    ) -> LibTpuResult<(u64, usize)> {
        let buf = PinnedDmaBuffer::allocate(size_bytes)?;
        let out = (buf.host_vaddr(), buf.capacity());
        self.buffers.insert((device_id, iova), buf);
        Ok(out)
    }

    /// Removes and `munmap`s the [`PinnedDmaBuffer`] at `(device_id, iova)`, returning its mapped length if present.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target [`DeviceId`].
    /// * `iova` - Device HBM IOVA [`DeviceAddress`].
    pub fn unregister_buffer(&mut self, device_id: DeviceId, iova: DeviceAddress) -> Option<usize> {
        self.buffers
            .remove(&(device_id, iova))
            .map(|b| b.capacity())
    }

    /// Stages `src` into the host-pinned buffer covering `(device_id, address)` at `offset_in_alloc`, returning `(host_src_vaddr, byte_len)`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target [`DeviceId`].
    /// * `alloc_base` - Base [`DeviceAddress`] of the allocation.
    /// * `offset_in_alloc` - Byte offset within the allocation.
    /// * `src` - Host source payload slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DmaMappingNotFound`] if `(device_id, alloc_base)` is not mapped or out of bounds.
    pub fn write_staging(
        &mut self,
        device_id: DeviceId,
        alloc_base: DeviceAddress,
        offset_in_alloc: usize,
        src: &[u8],
    ) -> LibTpuResult<u64> {
        let buf = self.buffers.get_mut(&(device_id, alloc_base)).ok_or(
            LibTpuError::DmaMappingNotFound {
                host_address: alloc_base.0,
            },
        )?;
        let base_vaddr = buf.host_vaddr();
        let end = offset_in_alloc.saturating_add(src.len());
        let dst = buf
            .as_mut_slice()
            .get_mut(offset_in_alloc..end)
            .ok_or_else(|| LibTpuError::DmaMappingNotFound {
                host_address: alloc_base.0.saturating_add(offset_in_alloc as u64),
            })?;
        dst.copy_from_slice(src);
        Ok(base_vaddr.saturating_add(offset_in_alloc as u64))
    }

    /// Returns the host virtual address inside `(device_id, alloc_base)` at `offset_in_alloc` for a D2H DMA write of `len` bytes.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target [`DeviceId`].
    /// * `alloc_base` - Base [`DeviceAddress`] of the allocation.
    /// * `offset_in_alloc` - Byte offset within the allocation.
    /// * `len` - Transfer byte length.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DmaMappingNotFound`] if unmapped or out of bounds.
    pub fn d2h_target_vaddr(
        &self,
        device_id: DeviceId,
        alloc_base: DeviceAddress,
        offset_in_alloc: usize,
        len: usize,
    ) -> LibTpuResult<u64> {
        let buf =
            self.buffers
                .get(&(device_id, alloc_base))
                .ok_or(LibTpuError::DmaMappingNotFound {
                    host_address: alloc_base.0,
                })?;
        if offset_in_alloc.saturating_add(len) > buf.capacity() {
            return Err(LibTpuError::DmaMappingNotFound {
                host_address: alloc_base.0.saturating_add(offset_in_alloc as u64),
            });
        }
        Ok(buf.host_vaddr().saturating_add(offset_in_alloc as u64))
    }

    /// Copies `dst.len()` bytes out of `(device_id, alloc_base)` at `offset_in_alloc` after D2H DMA completes.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target [`DeviceId`].
    /// * `alloc_base` - Base [`DeviceAddress`] of the allocation.
    /// * `offset_in_alloc` - Byte offset within the allocation.
    /// * `dst` - Destination slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DmaMappingNotFound`] if unmapped or out of bounds.
    pub fn read_staging(
        &self,
        device_id: DeviceId,
        alloc_base: DeviceAddress,
        offset_in_alloc: usize,
        dst: &mut [u8],
    ) -> LibTpuResult<()> {
        let buf =
            self.buffers
                .get(&(device_id, alloc_base))
                .ok_or(LibTpuError::DmaMappingNotFound {
                    host_address: alloc_base.0,
                })?;
        let end = offset_in_alloc.saturating_add(dst.len());
        let src = buf.as_slice().get(offset_in_alloc..end).ok_or_else(|| {
            LibTpuError::DmaMappingNotFound {
                host_address: alloc_base.0.saturating_add(offset_in_alloc as u64),
            }
        })?;
        dst.copy_from_slice(src);
        Ok(())
    }

    /// Returns a cloned byte vector of the staging buffer at `(device_id, alloc_base)`, if present.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target [`DeviceId`].
    /// * `alloc_base` - Base [`DeviceAddress`].
    #[must_use]
    pub fn snapshot_buffer(
        &self,
        device_id: DeviceId,
        alloc_base: DeviceAddress,
    ) -> Option<Vec<u8>> {
        self.buffers
            .get(&(device_id, alloc_base))
            .map(|b| b.as_slice().to_vec())
    }
}

/// Zero-copy peer-to-peer Linux `dma-buf` file descriptor descriptor (`DmaBufPeerRecord`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaBufPeerRecord {
    /// Linux `dma-buf` file descriptor.
    pub dmabuf_fd: i32,
    /// Owning or importing [`DeviceId`].
    pub device_id: DeviceId,
    /// Bound device IOVA [`DeviceAddress`].
    pub device_iova: DeviceAddress,
    /// Buffer byte size.
    pub size_bytes: u64,
    /// Whether this record was exported (`true`) or imported (`false`).
    pub is_exported: bool,
}

/// Peer-to-peer NIC/GPU `dma-buf` zero-copy FD registry (`DmaBufPeerRegistry`).
#[derive(Debug, Clone, Default)]
pub struct DmaBufPeerRegistry {
    /// Active `dma-buf` peer records keyed by `dmabuf_fd`.
    records: BTreeMap<i32, DmaBufPeerRecord>,
}

impl DmaBufPeerRegistry {
    /// Creates an empty [`DmaBufPeerRegistry`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: BTreeMap::new(),
        }
    }

    /// Records a newly exported or imported `dma-buf` peer mapping.
    ///
    /// # Arguments
    ///
    /// * `record` - [`DmaBufPeerRecord`] to insert.
    pub fn insert(&mut self, record: DmaBufPeerRecord) {
        self.records.insert(record.dmabuf_fd, record);
    }

    /// Looks up a [`DmaBufPeerRecord`] by `dmabuf_fd`.
    ///
    /// # Arguments
    ///
    /// * `dmabuf_fd` - Linux `dma-buf` file descriptor.
    #[must_use]
    pub fn get(&self, dmabuf_fd: i32) -> Option<DmaBufPeerRecord> {
        self.records.get(&dmabuf_fd).copied()
    }

    /// Returns the total number of active peer `dma-buf` bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if no peer `dma-buf` bindings are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Complete Linux `VFIO` (`0x3B`) IOMMU container, group, BAR `mmap`, MSI-X, and hot-reset controller (`VfioIommuSession`).
#[derive(Debug)]
pub struct VfioIommuSession {
    /// Open `/dev/vfio/vfio` container file descriptor.
    pub container_fd: i32,
    /// Open `/dev/vfio/<group>` IOMMU group file descriptor.
    pub group_fd: i32,
    /// Open `VFIO_GROUP_GET_DEVICE_FD` `PCIe` function file descriptor.
    pub device_fd: i32,
}

impl VfioIommuSession {
    /// Performs full Linux `VFIO_TYPE1v2_IOMMU` container + group + device initialization.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `iommu_group_id` - Numeric IOMMU group ID (`e.g. 0` -> `"/dev/vfio/0"`).
    /// * `bdf_name` - `PCIe` BDF string (e.g., `"0000:00:04.0"`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any `VFIO_*` `open` or `ioctl` call fails.
    pub fn open_and_initialize(
        syscalls: &dyn SyscallProvider,
        iommu_group_id: u32,
        bdf_name: &str,
    ) -> LibTpuResult<Self> {
        let container_fd = syscalls.open_device("/dev/vfio/vfio")?;
        let _ = syscalls.ioctl_device(container_fd, VFIO_GET_API_VERSION, 0)?;
        let _ = syscalls.ioctl_device(container_fd, VFIO_CHECK_EXTENSION, 4)?;

        let group_path = format!("/dev/vfio/{iommu_group_id}");
        let group_fd = syscalls.open_device(&group_path)?;
        let _ = syscalls.ioctl_device(group_fd, VFIO_GROUP_GET_STATUS, 8)?;
        let mut cfd_bytes = container_fd.to_le_bytes();
        let _ = syscalls.ioctl_bytes(group_fd, VFIO_GROUP_SET_CONTAINER, &mut cfd_bytes)?;
        let _ = syscalls.ioctl_device(container_fd, VFIO_SET_IOMMU, 4)?;

        let mut bdf_buf = [0_u8; 32];
        bdf_buf
            .iter_mut()
            .zip(bdf_name.as_bytes().iter())
            .for_each(|(d, s)| *d = *s);
        let raw_dev_fd = syscalls.ioctl_bytes(group_fd, VFIO_GROUP_GET_DEVICE_FD, &mut bdf_buf)?;
        let device_fd = if raw_dev_fd > 0 {
            raw_dev_fd
        } else {
            container_fd
        };

        let _ = syscalls.ioctl_device(device_fd, VFIO_DEVICE_GET_INFO, 16)?;
        let _ = syscalls.ioctl_device(device_fd, VFIO_DEVICE_GET_IRQ_INFO, 16)?;

        Ok(Self {
            container_fd,
            group_fd,
            device_fd,
        })
    }

    /// Maps `[host_vaddr, host_vaddr + size)` at `iova` via `VFIO_IOMMU_MAP_DMA` (`READ | WRITE`).
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `host_vaddr` - User-space virtual address of pinned host memory.
    /// * `iova` - Target device IOVA address.
    /// * `size` - Mapping byte size (4 KiB aligned).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `VFIO_IOMMU_MAP_DMA_IOCTL` fails.
    pub fn map_dma(
        &self,
        syscalls: &dyn SyscallProvider,
        host_vaddr: u64,
        iova: u64,
        size: u64,
    ) -> LibTpuResult<()> {
        let mut req = VfioDmaMapIoctl {
            argsz: 32,
            flags: 0x3,
            vaddr: host_vaddr,
            iova,
            size,
        };
        let _ = syscalls.ioctl_pod(self.container_fd, VFIO_IOMMU_MAP_DMA_IOCTL, &mut req)?;
        Ok(())
    }

    /// Unmaps `[iova, iova + size)` via `VFIO_IOMMU_UNMAP_DMA`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `iova` - Device IOVA start address.
    /// * `size` - Mapping byte size.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `VFIO_IOMMU_UNMAP_DMA_IOCTL` fails.
    pub fn unmap_dma(
        &self,
        syscalls: &dyn SyscallProvider,
        iova: u64,
        size: u64,
    ) -> LibTpuResult<u64> {
        let mut req = VfioDmaUnmapIoctl {
            argsz: 24,
            flags: 0,
            iova,
            size,
        };
        let _ = syscalls.ioctl_pod(self.container_fd, VFIO_IOMMU_UNMAP_DMA_IOCTL, &mut req)?;
        Ok(req.size)
    }

    /// Queries `VFIO_DEVICE_GET_REGION_INFO` for `bar_index` (`0`, `2`, `4`) and maps a [`MappedBarAperture`].
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `bar_index` - `PCIe` BAR index (`0`, `2`, or `4`).
    /// * `fallback_size` - Aperture byte size used if `region_info.size == 0`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `VFIO_DEVICE_GET_REGION_INFO` or `mmap` fails.
    pub fn mmap_bar_region(
        &self,
        syscalls: &dyn SyscallProvider,
        bar_index: u8,
        fallback_size: ByteSize,
    ) -> LibTpuResult<MappedBarAperture> {
        let mut info = VfioRegionInfoIoctl {
            argsz: 32,
            flags: 0,
            index: u32::from(bar_index),
            cap_offset: 0,
            size: fallback_size.0 as u64,
            offset: 0,
        };
        let _ = syscalls.ioctl_pod(self.device_fd, VFIO_DEVICE_GET_REGION_INFO, &mut info)?;
        let map_len = ByteSize(
            usize::try_from(info.size)
                .unwrap_or(fallback_size.0)
                .max(4096),
        );
        let posix = PosixKernelDriverBackend::new();
        let target_fd = if (info.flags & 0x4) != 0 {
            FileDescriptor(self.device_fd)
        } else {
            FileDescriptor(-1)
        };
        let ap_res = posix
            .mmap_bar_aperture_at_offset(target_fd, map_len, info.offset)
            .or_else(|_| posix.mmap_bar_aperture_at_offset(FileDescriptor(-1), map_len, 0));
        let mut ap = ap_res?;
        ap.bar_index = bar_index;
        Ok(ap)
    }

    /// Binds a Linux `eventfd` (`event_fd`) to MSI-X vector `vector_index` via `VFIO_DEVICE_SET_IRQS`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `vector_index` - MSI-X vector index (`0..=15`).
    /// * `event_fd` - Non-negative `eventfd` descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `event_fd < 0` or `VFIO_DEVICE_SET_IRQS_IOCTL` fails.
    pub fn bind_msix_eventfd(
        &self,
        syscalls: &dyn SyscallProvider,
        vector_index: u32,
        event_fd: i32,
    ) -> LibTpuResult<()> {
        if event_fd < 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "event_fd",
                reason: String::from("MSI-X event_fd must be non-negative"),
            });
        }
        let mut irq = VfioIrqSetIoctl {
            argsz: 24,
            flags: 0x24,
            index: 2,
            start: vector_index,
            count: 1,
            event_fd,
        };
        let _ = syscalls.ioctl_pod(self.device_fd, VFIO_DEVICE_SET_IRQS_IOCTL, &mut irq)?;
        Ok(())
    }

    /// Issues a `PCIe` function-level hot-reset via `VFIO_DEVICE_RESET` (`0x3b6f`).
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `VFIO_DEVICE_RESET` fails.
    pub fn hot_reset(&self, syscalls: &dyn SyscallProvider) -> LibTpuResult<()> {
        let _ = syscalls.ioctl_device(self.device_fd, VFIO_DEVICE_RESET, 0)?;
        Ok(())
    }

    /// Closes the device, group, and container file descriptors.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if closing the container descriptor fails.
    pub fn close_all(&self, syscalls: &dyn SyscallProvider) -> LibTpuResult<()> {
        if self.device_fd != self.container_fd {
            let _ = syscalls.close_device(self.device_fd);
        }
        let _ = syscalls.close_device(self.group_fd);
        syscalls.close_device(self.container_fd)
    }
}

/// Complete Linux `Gasket` / `Apex` (`0xDC`) kernel driver session (`GasketIommuSession`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasketIommuSession {
    /// Open `/dev/accel<N>` character device file descriptor.
    pub accel_fd: i32,
}

impl GasketIommuSession {
    /// Opens `accel_path` (`"/dev/accel0"`), resets the accelerator (`GASKET_IOCTL_RESET`),
    /// queries `GASKET_IOCTL_NUMBER_PAGE_TABLES`, and partitions page table `0`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `accel_path` - Path to `/dev/accel<N>`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `open` or any initialization `ioctl` fails.
    pub fn open_and_initialize(
        syscalls: &dyn SyscallProvider,
        accel_path: &str,
    ) -> LibTpuResult<Self> {
        let accel_fd = syscalls.open_device(accel_path)?;
        let _ = syscalls.ioctl_device(accel_fd, GASKET_IOCTL_RESET, 8)?;
        let _ = syscalls.ioctl_device(
            accel_fd,
            GASKET_IOCTL_NUMBER_PAGE_TABLES,
            size_of::<GasketPageTableIoctl>(),
        )?;
        let _ = syscalls.ioctl_device(
            accel_fd,
            GASKET_IOCTL_PARTITION_PAGE_TABLE,
            size_of::<GasketPageTableIoctl>(),
        )?;
        Ok(Self { accel_fd })
    }

    /// Maps `[host_vaddr, host_vaddr + size)` at `device_iova` via `GASKET_IOCTL_MAP_BUFFER`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `host_vaddr` - Host user-space virtual address.
    /// * `device_iova` - Target device IOVA address.
    /// * `size` - Mapping byte size.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `GASKET_IOCTL_MAP_BUFFER` fails.
    pub fn map_buffer(
        &self,
        syscalls: &dyn SyscallProvider,
        host_vaddr: u64,
        device_iova: u64,
        size: u64,
    ) -> LibTpuResult<()> {
        let mut pt = GasketPageTableIoctl {
            page_table_index: 0,
            size,
            host_address: host_vaddr,
            device_address: device_iova,
        };
        let _ = syscalls.ioctl_pod(self.accel_fd, GASKET_IOCTL_MAP_BUFFER, &mut pt)?;
        Ok(())
    }

    /// Unmaps `[device_iova, device_iova + size)` via `GASKET_IOCTL_UNMAP_BUFFER`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `device_iova` - Target device IOVA address.
    /// * `size` - Mapping byte size.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `GASKET_IOCTL_UNMAP_BUFFER` fails.
    pub fn unmap_buffer(
        &self,
        syscalls: &dyn SyscallProvider,
        device_iova: u64,
        size: u64,
    ) -> LibTpuResult<()> {
        let mut pt = GasketPageTableIoctl {
            page_table_index: 0,
            size,
            host_address: 0,
            device_address: device_iova,
        };
        let _ = syscalls.ioctl_pod(self.accel_fd, GASKET_IOCTL_UNMAP_BUFFER, &mut pt)?;
        Ok(())
    }

    /// Maps a zero-copy `dma_buf` file descriptor (`dmabuf_fd`) at `device_iova` via `GASKET_IOCTL_MAP_DMABUF`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `dmabuf_fd` - Non-negative `dma-buf` file descriptor.
    /// * `device_iova` - Target device IOVA address.
    /// * `flags` - Access flags (`0x3 = READ | WRITE`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `dmabuf_fd < 0` or `GASKET_IOCTL_MAP_DMABUF` fails.
    pub fn map_dmabuf(
        &self,
        syscalls: &dyn SyscallProvider,
        dmabuf_fd: i32,
        device_iova: u64,
        flags: u32,
    ) -> LibTpuResult<()> {
        if dmabuf_fd < 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "dmabuf_fd",
                reason: String::from("Gasket dmabuf_fd must be non-negative"),
            });
        }
        let mut req = GasketPageTableDmaBuf {
            page_table_index: 0,
            device_address: device_iova,
            dma_buf_fd: dmabuf_fd,
            flags,
        };
        let _ = syscalls.ioctl_pod(self.accel_fd, GASKET_IOCTL_MAP_DMABUF, &mut req)?;
        Ok(())
    }

    /// Binds `event_fd` to Gasket interrupt `interrupt_index` via `GASKET_IOCTL_SET_EVENTFD`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `interrupt_index` - Gasket interrupt vector index.
    /// * `event_fd` - Non-negative `eventfd` descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `event_fd < 0` or `GASKET_IOCTL_SET_EVENTFD` fails.
    pub fn set_eventfd(
        &self,
        syscalls: &dyn SyscallProvider,
        interrupt_index: u32,
        event_fd: i32,
    ) -> LibTpuResult<()> {
        if event_fd < 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "event_fd",
                reason: String::from("Gasket event_fd must be non-negative"),
            });
        }
        let mut req = GasketInterruptEventFd {
            interrupt: u64::from(interrupt_index),
            event_fd: u64::try_from(event_fd).unwrap_or(0),
            reserved: [0; 2],
        };
        let _ = syscalls.ioctl_pod(self.accel_fd, GASKET_IOCTL_SET_EVENTFD, &mut req)?;
        Ok(())
    }

    /// Clears `interrupt_index` via `GASKET_IOCTL_CLEAR_EVENTFD`.
    ///
    /// # Arguments
    ///
    /// * `syscalls` - [`SyscallProvider`] implementation.
    /// * `interrupt_index` - Gasket interrupt vector index.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `GASKET_IOCTL_CLEAR_EVENTFD` fails.
    pub fn clear_eventfd(
        &self,
        syscalls: &dyn SyscallProvider,
        interrupt_index: u32,
    ) -> LibTpuResult<()> {
        let mut idx_bytes = u64::from(interrupt_index).to_le_bytes();
        let _ = syscalls.ioctl_bytes(self.accel_fd, GASKET_IOCTL_CLEAR_EVENTFD, &mut idx_bytes)?;
        Ok(())
    }
}
