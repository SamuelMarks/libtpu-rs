//! HBM/VMEM region allocation, BAR MMIO apertures, and device memory statistics.
//!
//! Provides [`HbmRegionAllocator`], bounds-checked volatile `PCIe` BAR MMIO apertures
//! ([`MappedBarAperture`]), and device memory usage statistics ([`DeviceMemoryStats`]).

use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ByteSize, DeviceAddress, MmioOffset};
use std::collections::BTreeMap;

/// Type alias matching the architectural MMIO aperture name (`MmioAperture`).
pub type MmioAperture = MappedBarAperture;

/// Type alias for [`HbmRegionAllocator`] used for on-chip Vector Memory (VMEM) region allocation (`VmemRegionAllocator`).
pub type VmemRegionAllocator = HbmRegionAllocator;

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

    /// Performs a bounds-checked, 4-byte-aligned volatile 32-bit CSR read at `offset`.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset within the mapped BAR aperture.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset` is unaligned or exceeds `self.size`.
    pub fn read_csr_u32(&self, offset: MmioOffset) -> LibTpuResult<u32> {
        let off = usize::try_from(offset.0).unwrap_or(usize::MAX);
        if (off & 3) != 0 || off.saturating_add(4) > self.size.0 || self.base_vaddr == 0 {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index: 0,
                offset: offset.0,
                size: 4,
            });
        }
        let reg_ptr = (self.base_vaddr + off) as *const u32;
        // SAFETY: `reg_ptr` is 4-byte aligned and within `[base_vaddr, base_vaddr + size)`.
        let val = unsafe { std::ptr::read_volatile(reg_ptr) };
        Ok(val)
    }

    /// Performs a bounds-checked, 4-byte-aligned volatile 32-bit CSR write at `offset`.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset within the mapped BAR aperture.
    /// * `value` - 32-bit CSR value to write.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MmioOutOfBounds`] if `offset` is unaligned or exceeds `self.size`.
    pub fn write_csr_u32(&self, offset: MmioOffset, value: u32) -> LibTpuResult<()> {
        let off = usize::try_from(offset.0).unwrap_or(usize::MAX);
        if (off & 3) != 0 || off.saturating_add(4) > self.size.0 || self.base_vaddr == 0 {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index: 0,
                offset: offset.0,
                size: 4,
            });
        }
        let reg_ptr = (self.base_vaddr + off) as *mut u32;
        // SAFETY: `reg_ptr` is 4-byte aligned and within `[base_vaddr, base_vaddr + size)`.
        unsafe {
            std::ptr::write_volatile(reg_ptr, value);
        }
        Ok(())
    }
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
