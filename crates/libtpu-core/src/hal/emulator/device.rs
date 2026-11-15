//! Virtual `PCIe` / `VFIO` / `Gasket` Kernel Device Emulator (`VirtualVfioGasketDevice`).
//!
//! Speaks authentic Linux `VFIO` and `Gasket` `ioctl` structures, maintains an IOMMU `IOVA -> host_vaddr`
//! page table, traps `BAR0` and `BAR2` MMIO writes, executes real host-to-HBM and HBM-to-host DMA
//! transfers, decodes and executes real generation-specific binary VLIW & `SparseCore` bitstreams
//! from `SparseHbmStore` via [`ArchitecturalBitStreamEngine`], records `HardwareTraceRingRecord`s
//! at `HBM[0x00E0_0000]`, and signals real OS `eventfd`/pipe descriptors upon completion.

use super::super::pcie::{Bar0CsrReader, HardwareCsrProber};
use super::super::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SPARSECORE_RUN_CONTROL, BAR0_CSR_SYNC_FLAG_BASE,
    BAR0_CSR_TENSORCORE_PC, BAR0_CSR_TENSORCORE_RUN_CONTROL, BAR0_CSR_TENSORCORE_STATUS,
    BAR0_CSR_TENSORCORE_SYNC_FLAG, BAR0_CSR_THERMAL_MILLICELSIUS, ChipRegisterMap,
    GASKET_IOCTL_CLEAR_EVENTFD, GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS, GASKET_IOCTL_MAP_BUFFER,
    GASKET_IOCTL_MAP_DMABUF, GASKET_IOCTL_NUMBER_PAGE_TABLES, GASKET_IOCTL_PAGE_TABLE_SIZE,
    GASKET_IOCTL_PARTITION_PAGE_TABLE, GASKET_IOCTL_RESET, GASKET_IOCTL_SET_EVENTFD,
    GASKET_IOCTL_UNMAP_BUFFER, GasketInterruptEventFd, GasketPageTableDmaBuf, GasketPageTableIoctl,
    HardwareDmaCommandDescriptor, PosixKernelDriverBackend, SyscallProvider, TensorCoreStatusFlags,
    VFIO_CHECK_EXTENSION, VFIO_DEVICE_GET_INFO, VFIO_DEVICE_GET_IRQ_INFO,
    VFIO_DEVICE_GET_REGION_INFO, VFIO_DEVICE_RESET, VFIO_DEVICE_SET_IRQS_IOCTL,
    VFIO_GET_API_VERSION, VFIO_GROUP_GET_DEVICE_FD, VFIO_GROUP_GET_STATUS,
    VFIO_GROUP_SET_CONTAINER, VFIO_IOMMU_MAP_DMA_IOCTL, VFIO_IOMMU_UNMAP_DMA_IOCTL, VFIO_SET_IOMMU,
    VfioDmaMapIoctl, VfioDmaUnmapIoctl, VfioIrqSetIoctl, VfioRegionInfoIoctl,
};
use super::core_engine::{
    ArchitecturalBitStreamEngine, HBM_IMEM_TENSORCORE_BASE, HBM_XPLANE_TRACE_RING_BASE,
    HardwareTraceRingRecord, SparseHbmStore, VirtualIciTorusFabric,
};
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ChipCoordinates, DeviceId, EventFdHandle, MmioOffset, TpuGeneration};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

/// `BAR2` byte offset of the 64-entry × 64-byte [`HardwareDmaCommandDescriptor`] submission ring (`0x0000`).
pub const BAR2_DMA_RING_OFFSET: u64 = 0x0000_0000;

/// `BAR2` byte offset of the 64-entry input buffer HBM IOVA table (`0x1000`).
pub const BAR2_INPUT_ADDR_TABLE_OFFSET: u64 = 0x0000_1000;

/// `BAR2` byte offset of the 64-entry input buffer byte-length table (`0x1800`).
pub const BAR2_INPUT_LEN_TABLE_OFFSET: u64 = 0x0000_1800;

/// `BAR2` byte offset of the 64-entry output buffer HBM IOVA table (`0x2000`).
pub const BAR2_OUTPUT_ADDR_TABLE_OFFSET: u64 = 0x0000_2000;

/// `BAR2` byte offset of the 64-entry output buffer byte-length table (`0x2800`).
pub const BAR2_OUTPUT_LEN_TABLE_OFFSET: u64 = 0x0000_2800;

/// `BAR2` byte offset storing the active input descriptor count (`0x3000`).
pub const BAR2_INPUT_COUNT_CSR_OFFSET: u64 = 0x0000_3000;

/// `BAR2` byte offset storing the active output descriptor count (`0x3008`).
pub const BAR2_OUTPUT_COUNT_CSR_OFFSET: u64 = 0x0000_3008;

/// `BAR2` byte offset storing the uploaded `.rodata` `CMEM` descriptor byte length (`0x3010`).
pub const BAR2_CMEM_LEN_CSR_OFFSET: u64 = 0x0000_3010;

/// `BAR2` byte offset storing the uploaded `SparseCore` V2/V3 bitstream byte length (`0x3018`).
pub const BAR2_SC_IMEM_LEN_CSR_OFFSET: u64 = 0x0000_3018;

/// `BAR0` CSR offset publishing the current `XPlane` hardware trace ring write cursor (`0x9040`).
pub const BAR0_CSR_XPLANE_TRACE_CURSOR: u64 = 0x0000_9040;

/// Packs `(device_id, hbm_phys_addr)` into a 64-bit hardware IOVA (`bits 48..64 = device_id`, `bits 0..48 = hbm_phys_addr`).
///
/// # Arguments
///
/// * `device_id` - Logical chip [`DeviceId`].
/// * `hbm_phys_addr` - Physical HBM byte offset (`0 .. 256 TiB`).
#[must_use]
pub const fn encode_device_hbm_iova(device_id: DeviceId, hbm_phys_addr: u64) -> u64 {
    ((device_id.0 as u64) << 48) | (hbm_phys_addr & 0x0000_FFFF_FFFF_FFFF)
}

/// Unpacks an encoded 64-bit hardware IOVA into `(DeviceId, hbm_phys_addr)`.
///
/// # Arguments
///
/// * `encoded_iova` - 64-bit hardware IOVA encoded by [`encode_device_hbm_iova`].
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn unpack_device_hbm_iova(encoded_iova: u64) -> (DeviceId, u64) {
    let dev = (encoded_iova >> 48) as u32;
    let phys = encoded_iova & 0x0000_FFFF_FFFF_FFFF;
    (DeviceId(dev), phys)
}

/// Active IOMMU DMA mapping entry inside [`VirtualVfioGasketDevice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VirtualIommuEntry {
    /// Host user-space virtual address (`vaddr`).
    host_vaddr: u64,
    /// Mapped byte size.
    size_bytes: u64,
}

/// Internal mutable hardware state of [`VirtualVfioGasketDevice`].
#[derive(Debug)]
struct VirtualDeviceState {
    /// Cloud TPU hardware generation.
    generation: TpuGeneration,
    /// Open virtual file descriptors.
    open_fds: BTreeSet<i32>,
    /// Whether `VFIO_TYPE1v2_IOMMU` has been enabled on the container.
    vfio_iommu_enabled: bool,
    /// Active IOMMU mappings keyed by `device_iova` AND `host_vaddr` for DMA validation.
    iommu_by_iova: BTreeMap<u64, VirtualIommuEntry>,
    /// Active host staging regions keyed by `host_vaddr -> size_bytes`.
    iommu_by_vaddr: BTreeMap<u64, u64>,
    /// Bound OS `eventfd` / pipe write descriptors keyed by interrupt vector (`0..16`).
    interrupt_fds: BTreeMap<u32, i32>,
    /// `BAR0` 64-bit CSR register file keyed by byte offset.
    bar0_csrs: BTreeMap<u64, u64>,
    /// `BAR2` 64-bit MMIO aperture words keyed by byte offset (`ring`, `input/output tables`, etc.).
    bar2_words: BTreeMap<u64, u64>,
    /// Per-device `4 KiB`-paged physical HBM stores.
    chip_hbms: BTreeMap<DeviceId, Arc<Mutex<SparseHbmStore>>>,
    /// Shared 3D twisted-torus `ICI` RDMA switch fabric.
    ici_fabric: Arc<VirtualIciTorusFabric>,
    /// Monotonic hardware cycle counter.
    cycle_counter: u64,
    /// Monotonic write cursor into `HBM[HBM_XPLANE_TRACE_RING_BASE]` (`0..1024` modulo ring).
    trace_cursor: u32,
}

impl VirtualDeviceState {
    /// Resets `bar0_csrs` to authentic `csr_register_map.json` power-on reset values for `generation`.
    fn reset_generation_csrs(&mut self) {
        let rmap = ChipRegisterMap::for_generation(self.generation);
        self.bar0_csrs.clear();
        let sku = self.generation.pcie_device_id();
        let global_id_word =
            HardwareCsrProber::pack_global_chip_id_csr(0, ChipCoordinates::new(0, 0, 0), 0, sku);
        self.bar0_csrs
            .insert(rmap.global_chip_id_csr, global_id_word);

        let port_count: u8 = match self.generation {
            TpuGeneration::V2 | TpuGeneration::V3 | TpuGeneration::V5e | TpuGeneration::V6e => 4,
            TpuGeneration::V4 | TpuGeneration::V4Lite | TpuGeneration::V5p | TpuGeneration::V7x => {
                6
            }
        };
        let lane_rate: u16 = match self.generation {
            TpuGeneration::V2 => 50,
            TpuGeneration::V3 | TpuGeneration::V4 | TpuGeneration::V4Lite => 100,
            TpuGeneration::V5e => 200,
            TpuGeneration::V5p | TpuGeneration::V6e => 400,
            TpuGeneration::V7x => 800,
        };
        for port in 0..port_count {
            let off = HardwareCsrProber::ici_port_status_csr_offset(self.generation, port);
            let word = HardwareCsrProber::pack_ici_port_status_csr(
                true,
                lane_rate,
                u16::from(port) + 1,
                0,
            );
            self.bar0_csrs.insert(off.0, word);
        }

        self.bar0_csrs.insert(BAR0_CSR_TENSORCORE_RUN_CONTROL, 0);
        self.bar0_csrs.insert(BAR0_CSR_TENSORCORE_PC, 0);
        self.bar0_csrs.insert(BAR0_CSR_TENSORCORE_SYNC_FLAG, 1);
        self.bar0_csrs.insert(BAR0_CSR_TENSORCORE_STATUS, 0);
        self.bar0_csrs.insert(BAR0_CSR_SPARSECORE_RUN_CONTROL, 0);
        self.bar0_csrs.insert(BAR0_CSR_SYNC_FLAG_BASE, 1);
        self.bar0_csrs.insert(BAR0_CSR_HBM_DMA_RING_HEAD, 0);
        self.bar0_csrs.insert(BAR0_CSR_HBM_DMA_RING_TAIL, 0);
        self.bar0_csrs.insert(BAR0_CSR_XPLANE_TRACE_CURSOR, 0);
        self.bar0_csrs.insert(BAR0_CSR_THERMAL_MILLICELSIUS, 45_000);
        self.bar0_csrs.insert(BAR0_CSR_POWER_MILLIWATTS, 175_000);
        self.bar0_csrs.insert(BAR0_CSR_DUTY_CYCLE_PERCENT, 100);
        self.bar0_csrs.insert(rmap.tc0_program_location_csr, 0);
        self.bar0_csrs.insert(rmap.tc0_pc_csr, 0);
        self.bar0_csrs.insert(rmap.tc0_run_status_csr, 0);
        self.bar0_csrs.insert(rmap.tc0_vwait_sync_flag_state_csr, 1);
        self.bar0_csrs.insert(rmap.sc0_run_status_csr, 0);
    }

    /// Returns or lazily allocates the [`SparseHbmStore`] for `device_id` and attaches it to `ici_fabric`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Logical [`DeviceId`].
    fn ensure_chip_hbm(&mut self, device_id: DeviceId) -> Arc<Mutex<SparseHbmStore>> {
        if let Some(existing) = self.chip_hbms.get(&device_id) {
            return Arc::clone(existing);
        }
        let store = Arc::new(Mutex::new(SparseHbmStore::new()));
        self.ici_fabric.register_chip(device_id, Arc::clone(&store));
        self.chip_hbms.insert(device_id, Arc::clone(&store));
        store
    }

    /// Resolves a host virtual address range `[addr, addr + len)` through registered IOMMU mappings.
    ///
    /// # Arguments
    ///
    /// * `iova_or_vaddr` - Device IOVA or host virtual address.
    /// * `len` - Transfer byte length.
    fn resolve_host_dma_ptr(&self, iova_or_vaddr: u64, len: usize) -> Option<usize> {
        let len_u64 = len as u64;
        if let Some((&base_vaddr, &cap)) = self.iommu_by_vaddr.range(..=iova_or_vaddr).next_back() {
            let off = iova_or_vaddr - base_vaddr;
            if off.saturating_add(len_u64) <= cap {
                return usize::try_from(iova_or_vaddr).ok();
            }
        }
        if let Some((&base_iova, entry)) = self.iommu_by_iova.range(..=iova_or_vaddr).next_back() {
            let off = iova_or_vaddr - base_iova;
            if off.saturating_add(len_u64) <= entry.size_bytes && entry.host_vaddr != 0 {
                return usize::try_from(entry.host_vaddr.saturating_add(off)).ok();
            }
        }
        None
    }

    /// Decodes a 64-byte [`HardwareDmaCommandDescriptor`] at `slot_index` (`0..64`) from `bar2_words`.
    ///
    /// # Arguments
    ///
    /// * `slot_index` - Ring descriptor slot index.
    #[allow(clippy::cast_possible_truncation)]
    fn read_bar2_ring_descriptor(&self, slot_index: u64) -> HardwareDmaCommandDescriptor {
        let base = BAR2_DMA_RING_OFFSET + (slot_index % 64) * 64;
        let w0 = self.bar2_words.get(&base).copied().unwrap_or(0);
        let w1 = self.bar2_words.get(&(base + 8)).copied().unwrap_or(0);
        let w2 = self.bar2_words.get(&(base + 16)).copied().unwrap_or(0);
        let w3 = self.bar2_words.get(&(base + 24)).copied().unwrap_or(0);
        let w4 = self.bar2_words.get(&(base + 32)).copied().unwrap_or(0);
        let w5 = self.bar2_words.get(&(base + 40)).copied().unwrap_or(0);
        let w6 = self.bar2_words.get(&(base + 48)).copied().unwrap_or(0);
        let w7 = self.bar2_words.get(&(base + 56)).copied().unwrap_or(0);

        let b6 = w6.to_le_bytes();
        let b7 = w7.to_le_bytes();
        HardwareDmaCommandDescriptor {
            opcode: (w0 & 0xFF) as u8,
            flags: ((w0 >> 8) & 0xFF) as u8,
            queue_id: ((w0 >> 16) & 0xFFFF) as u16,
            sync_flag_id: (w0 >> 32) as u32,
            src_iova: w1,
            dst_iova: w2,
            byte_length: w3,
            imem_iova: w4,
            imem_bundle_count: (w5 & 0xFFFF_FFFF) as u32,
            completion_seq: (w5 >> 32) as u32,
            reserved: [
                b6[0], b6[1], b6[2], b6[3], b6[4], b6[5], b6[6], b6[7], b7[0], b7[1], b7[2], b7[3],
            ],
            crc32c: u32::from_le_bytes([b7[4], b7[5], b7[6], b7[7]]),
        }
    }

    /// Signals a registered OS `eventfd`/pipe write handle on `queue_id`, if present.
    ///
    /// # Arguments
    ///
    /// * `queue_id` - Hardware MSI-X / Gasket interrupt vector index.
    fn signal_queue_os_eventfd(&self, queue_id: u32) {
        if let Some(&efd) = self.interrupt_fds.get(&queue_id) {
            let posix = PosixKernelDriverBackend::new();
            let _ = posix.signal_eventfd(EventFdHandle(efd), 1);
        }
    }

    /// Processes all pending [`HardwareDmaCommandDescriptor`] entries in `[head, new_tail)`.
    ///
    /// # Arguments
    ///
    /// * `new_tail` - Newly written `BAR0_CSR_HBM_DMA_RING_TAIL` producer pointer.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn process_dma_ring_doorbell(&mut self, new_tail: u64) -> LibTpuResult<()> {
        let status = self
            .bar0_csrs
            .get(&BAR0_CSR_TENSORCORE_STATUS)
            .copied()
            .unwrap_or(0);
        let fault_mask = TensorCoreStatusFlags::HBM_ECC_ERROR.bits()
            | TensorCoreStatusFlags::SYNC_FLAG_TIMEOUT.bits()
            | TensorCoreStatusFlags::POISONED.bits();
        if (status & fault_mask) != 0 {
            return Err(LibTpuError::HardwareRingTimeout {
                device_id: 0,
                queue_id: 0,
                completion_seq: (status & 0xFFFF_FFFF) as u32,
            });
        }

        let mut head = self
            .bar0_csrs
            .get(&BAR0_CSR_HBM_DMA_RING_HEAD)
            .copied()
            .unwrap_or(0);
        while head < new_tail {
            let desc = self.read_bar2_ring_descriptor(head);
            if desc.opcode != 0 && desc.crc32c != desc.compute_checksum() {
                self.bar0_csrs.insert(
                    BAR0_CSR_TENSORCORE_STATUS,
                    status | TensorCoreStatusFlags::POISONED.bits(),
                );
                return Err(LibTpuError::HardwareRingTimeout {
                    device_id: 0,
                    queue_id: desc.queue_id,
                    completion_seq: desc.completion_seq,
                });
            }

            let byte_len = usize::try_from(desc.byte_length).unwrap_or(0);
            match desc.opcode {
                1 => {
                    // Host-to-Device (H2D) DMA over IOMMU
                    let (dst_dev, dst_hbm) = unpack_device_hbm_iova(desc.dst_iova);
                    let hbm = self.ensure_chip_hbm(dst_dev);
                    if byte_len > 0 {
                        let host_ptr = self.resolve_host_dma_ptr(desc.src_iova, byte_len).ok_or(
                            LibTpuError::DmaMappingNotFound {
                                host_address: desc.src_iova,
                            },
                        )?;
                        // SAFETY: `host_ptr` was validated against an active `PinnedDmaBuffer` mapping in `iommu_by_vaddr` / `iommu_by_iova`.
                        let host_slice =
                            unsafe { std::slice::from_raw_parts(host_ptr as *const u8, byte_len) };
                        hbm.lock().write_bytes(dst_hbm, host_slice);
                    }
                    let start_c = self.cycle_counter;
                    let dma_cycles = ((desc.byte_length / 4096) + 1).max(1);
                    self.cycle_counter = self.cycle_counter.saturating_add(dma_cycles);
                    let rec = HardwareTraceRingRecord {
                        start_cycle: start_c,
                        duration_cycles: u32::try_from(dma_cycles).unwrap_or(1),
                        pc_offset: 0,
                        flops_retired: 0,
                        hbm_bytes_transferred: u32::try_from(byte_len).unwrap_or(u32::MAX),
                        unit_tag: 3,
                    };
                    let ring_off =
                        HBM_XPLANE_TRACE_RING_BASE + u64::from(self.trace_cursor & 1023) * 32;
                    hbm.lock().write_bytes(ring_off, &rec.to_le_bytes());
                    self.trace_cursor = self.trace_cursor.wrapping_add(1);
                }
                2 => {
                    // Device-to-Host (D2H) DMA over IOMMU
                    let (src_dev, src_hbm) = unpack_device_hbm_iova(desc.src_iova);
                    let hbm = self.ensure_chip_hbm(src_dev);
                    if byte_len > 0 {
                        let host_ptr = self.resolve_host_dma_ptr(desc.dst_iova, byte_len).ok_or(
                            LibTpuError::DmaMappingNotFound {
                                host_address: desc.dst_iova,
                            },
                        )?;
                        // SAFETY: `host_ptr` was validated against an active `PinnedDmaBuffer` mapping in `iommu_by_vaddr` / `iommu_by_iova`.
                        let host_dst = unsafe {
                            std::slice::from_raw_parts_mut(host_ptr as *mut u8, byte_len)
                        };
                        hbm.lock().read_into(src_hbm, host_dst);
                    }
                    let start_c = self.cycle_counter;
                    let dma_cycles = ((desc.byte_length / 4096) + 1).max(1);
                    self.cycle_counter = self.cycle_counter.saturating_add(dma_cycles);
                    let rec = HardwareTraceRingRecord {
                        start_cycle: start_c,
                        duration_cycles: u32::try_from(dma_cycles).unwrap_or(1),
                        pc_offset: 0,
                        flops_retired: 0,
                        hbm_bytes_transferred: u32::try_from(byte_len).unwrap_or(u32::MAX),
                        unit_tag: 3,
                    };
                    let ring_off =
                        HBM_XPLANE_TRACE_RING_BASE + u64::from(self.trace_cursor & 1023) * 32;
                    hbm.lock().write_bytes(ring_off, &rec.to_le_bytes());
                    self.trace_cursor = self.trace_cursor.wrapping_add(1);
                }
                3 => {
                    // ExecuteImem: fetch & decode real binary VLIW + SparseCore bitstreams from HBM
                    let (exec_dev, imem_phys) = unpack_device_hbm_iova(desc.imem_iova);
                    let imem_base = if imem_phys == 0 {
                        HBM_IMEM_TENSORCORE_BASE
                    } else {
                        imem_phys
                    };
                    let hbm = self.ensure_chip_hbm(exec_dev);
                    let in_cnt = usize::try_from(
                        self.bar2_words
                            .get(&BAR2_INPUT_COUNT_CSR_OFFSET)
                            .copied()
                            .unwrap_or(0),
                    )
                    .unwrap_or(0)
                    .min(64);
                    let out_cnt = usize::try_from(
                        self.bar2_words
                            .get(&BAR2_OUTPUT_COUNT_CSR_OFFSET)
                            .copied()
                            .unwrap_or(0),
                    )
                    .unwrap_or(0)
                    .min(64);
                    let mut in_descs = Vec::with_capacity(in_cnt);
                    for idx in 0..in_cnt {
                        let a_off = BAR2_INPUT_ADDR_TABLE_OFFSET + (idx as u64) * 8;
                        let l_off = BAR2_INPUT_LEN_TABLE_OFFSET + (idx as u64) * 8;
                        let (_, phys_a) = unpack_device_hbm_iova(
                            self.bar2_words.get(&a_off).copied().unwrap_or(0),
                        );
                        let len_a =
                            usize::try_from(self.bar2_words.get(&l_off).copied().unwrap_or(0))
                                .unwrap_or(0);
                        in_descs.push((phys_a, len_a));
                    }
                    let mut out_descs = Vec::with_capacity(out_cnt);
                    for idx in 0..out_cnt {
                        let a_off = BAR2_OUTPUT_ADDR_TABLE_OFFSET + (idx as u64) * 8;
                        let l_off = BAR2_OUTPUT_LEN_TABLE_OFFSET + (idx as u64) * 8;
                        let (_, phys_a) = unpack_device_hbm_iova(
                            self.bar2_words.get(&a_off).copied().unwrap_or(0),
                        );
                        let len_a =
                            usize::try_from(self.bar2_words.get(&l_off).copied().unwrap_or(0))
                                .unwrap_or(0);
                        out_descs.push((phys_a, len_a));
                    }
                    let cmem_len = usize::try_from(
                        self.bar2_words
                            .get(&BAR2_CMEM_LEN_CSR_OFFSET)
                            .copied()
                            .unwrap_or(0),
                    )
                    .unwrap_or(0);
                    let sc_len = usize::try_from(
                        self.bar2_words
                            .get(&BAR2_SC_IMEM_LEN_CSR_OFFSET)
                            .copied()
                            .unwrap_or(0),
                    )
                    .unwrap_or(0);

                    let fabric = Arc::clone(&self.ici_fabric);
                    let mut cursor_copy = self.trace_cursor;
                    let summary = ArchitecturalBitStreamEngine::execute_from_hbm(
                        self.generation,
                        exec_dev,
                        &hbm,
                        Some(&fabric),
                        imem_base,
                        byte_len,
                        cmem_len,
                        sc_len,
                        &in_descs,
                        &out_descs,
                        self.cycle_counter,
                        &mut cursor_copy,
                    )?;
                    self.trace_cursor = cursor_copy;
                    self.cycle_counter = self.cycle_counter.saturating_add(summary.cycles_elapsed);

                    let rmap = ChipRegisterMap::for_generation(self.generation);
                    self.bar0_csrs
                        .insert(rmap.tc0_program_location_csr, summary.final_pc_bytes);
                    self.bar0_csrs
                        .insert(rmap.tc0_pc_csr, summary.final_pc_bytes);
                    self.bar0_csrs.insert(rmap.tc0_run_status_csr, 2);
                    self.bar0_csrs
                        .insert(rmap.tc1_program_location_csr(), summary.final_pc_bytes);
                    self.bar0_csrs
                        .insert(rmap.tc1_pc_csr(), summary.final_pc_bytes);
                    self.bar0_csrs.insert(rmap.tc1_run_status_csr(), 2);
                    if summary.sc_instructions_retired > 0 {
                        self.bar0_csrs.insert(rmap.sc0_run_status_csr, 2);
                    }
                    self.bar0_csrs
                        .insert(BAR0_CSR_TENSORCORE_PC, summary.final_pc_bytes);
                    self.bar0_csrs.insert(BAR0_CSR_TENSORCORE_STATUS, 2);
                }
                4 => {
                    // Device-to-Device (D2D) / 3D Twisted-Torus ICI RDMA DMA
                    let (src_dev, src_hbm) = unpack_device_hbm_iova(desc.src_iova);
                    let (dst_dev, dst_hbm) = unpack_device_hbm_iova(desc.dst_iova);
                    let src_store = self.ensure_chip_hbm(src_dev);
                    let _dst_store = self.ensure_chip_hbm(dst_dev);
                    self.ici_fabric
                        .rdma_copy(src_dev, src_hbm, dst_dev, dst_hbm, byte_len)?;
                    let start_c = self.cycle_counter;
                    self.cycle_counter = self.cycle_counter.saturating_add(2);
                    let rec = HardwareTraceRingRecord {
                        start_cycle: start_c,
                        duration_cycles: 2,
                        pc_offset: 0,
                        flops_retired: 0,
                        hbm_bytes_transferred: u32::try_from(byte_len).unwrap_or(u32::MAX),
                        unit_tag: 4,
                    };
                    let ring_off =
                        HBM_XPLANE_TRACE_RING_BASE + u64::from(self.trace_cursor & 1023) * 32;
                    src_store.lock().write_bytes(ring_off, &rec.to_le_bytes());
                    self.trace_cursor = self.trace_cursor.wrapping_add(1);
                }
                _ => {}
            }

            let next_sync = u64::from(desc.completion_seq)
                .max(u64::from(desc.sync_flag_id))
                .max(head + 1);
            let cur_sync = self
                .bar0_csrs
                .get(&BAR0_CSR_SYNC_FLAG_BASE)
                .copied()
                .unwrap_or(1);
            let merged_sync = cur_sync.max(next_sync);
            self.bar0_csrs.insert(BAR0_CSR_SYNC_FLAG_BASE, merged_sync);
            self.bar0_csrs
                .insert(BAR0_CSR_TENSORCORE_SYNC_FLAG, merged_sync);
            self.bar0_csrs
                .insert(BAR0_CSR_XPLANE_TRACE_CURSOR, u64::from(self.trace_cursor));
            if (desc.flags & 0x01) != 0 {
                self.signal_queue_os_eventfd(u32::from(desc.queue_id));
            }
            head = head.wrapping_add(1);
            self.bar0_csrs.insert(BAR0_CSR_HBM_DMA_RING_HEAD, head);
        }
        Ok(())
    }
}

/// Standalone Virtual `PCIe` / `VFIO` (`0x3B`) / `Gasket` (`0xDC`) Cloud TPU Device (`VirtualVfioGasketDevice`).
#[derive(Debug, Clone)]
pub struct VirtualVfioGasketDevice {
    /// Thread-safe hardware register, IOMMU, and HBM state.
    state: Arc<Mutex<VirtualDeviceState>>,
    /// Monotonic virtual file descriptor allocator (`>= 200`).
    next_fd: Arc<AtomicI32>,
}

impl Default for VirtualVfioGasketDevice {
    fn default() -> Self {
        Self::new(TpuGeneration::V5e, 1, VirtualIciTorusFabric::new())
    }
}

/// Writes a little-endian `u32` into `payload` starting at `offset` without panicking on short buffers.
///
/// # Arguments
///
/// * `payload` - Destination `ioctl` byte slice.
/// * `offset` - Starting byte offset.
/// * `val` - 32-bit unsigned value to write.
fn write_u32_le(payload: &mut [u8], offset: usize, val: u32) {
    for (d, s) in payload.iter_mut().skip(offset).zip(val.to_le_bytes()) {
        *d = s;
    }
}

/// Decodes a `Pod` struct `T` from the prefix of `payload` (zero-padding if `payload` is shorter than `size_of::<T>()`).
///
/// # Arguments
///
/// * `payload` - Source `ioctl` byte slice.
fn pod_from_prefix<T: bytemuck::Pod>(payload: &[u8]) -> T {
    let mut out = T::zeroed();
    for (d, &s) in bytemuck::bytes_of_mut(&mut out).iter_mut().zip(payload) {
        *d = s;
    }
    out
}

impl VirtualVfioGasketDevice {
    /// Dispatches `request` and `payload` under the internal state lock.
    ///
    /// # Arguments
    ///
    /// * `st` - Mutable [`VirtualDeviceState`] guard.
    /// * `next_fd` - Monotonic virtual file descriptor allocator.
    /// * `request` - Raw `VFIO` (`0x3B`) or `Gasket` (`0xDC`) `ioctl` request code.
    /// * `payload` - Mutable user-space `ioctl` payload slice.
    #[allow(clippy::too_many_lines)]
    fn dispatch_ioctl_inner(
        st: &mut VirtualDeviceState,
        next_fd: &AtomicI32,
        request: u64,
        payload: &mut [u8],
    ) -> LibTpuResult<i32> {
        match request {
            VFIO_CHECK_EXTENSION => Ok(1),
            VFIO_SET_IOMMU => {
                st.vfio_iommu_enabled = true;
                Ok(0)
            }
            VFIO_GROUP_GET_STATUS => {
                write_u32_le(payload, 4, 1);
                Ok(0)
            }
            VFIO_GROUP_GET_DEVICE_FD => {
                let dev_fd = next_fd.fetch_add(1, Ordering::Relaxed);
                st.open_fds.insert(dev_fd);
                Ok(dev_fd)
            }
            VFIO_DEVICE_GET_INFO => {
                write_u32_le(payload, 4, 3);
                write_u32_le(payload, 8, 6);
                write_u32_le(payload, 12, 16);
                Ok(0)
            }
            VFIO_DEVICE_GET_REGION_INFO => {
                let mut info: VfioRegionInfoIoctl = pod_from_prefix(payload);
                info.flags = 0x7;
                info.size = match info.index {
                    0 => 0x0010_0000,
                    2 => 0x0400_0000,
                    _ => 0x0001_0000,
                };
                info.offset = u64::from(info.index) << 32;
                for (d, &s) in payload.iter_mut().zip(bytemuck::bytes_of(&info)) {
                    *d = s;
                }
                Ok(0)
            }
            VFIO_DEVICE_GET_IRQ_INFO => {
                write_u32_le(payload, 4, 9);
                write_u32_le(payload, 12, 16);
                Ok(0)
            }
            VFIO_DEVICE_SET_IRQS_IOCTL => {
                let irq: VfioIrqSetIoctl = pod_from_prefix(payload);
                if irq.event_fd >= 0 {
                    st.interrupt_fds.insert(irq.start, irq.event_fd);
                } else {
                    st.interrupt_fds.remove(&irq.start);
                }
                Ok(0)
            }
            VFIO_DEVICE_RESET | GASKET_IOCTL_RESET => {
                st.reset_generation_csrs();
                Ok(0)
            }
            VFIO_IOMMU_MAP_DMA_IOCTL => {
                let map: VfioDmaMapIoctl = pod_from_prefix(payload);
                st.iommu_by_iova.insert(
                    map.iova,
                    VirtualIommuEntry {
                        host_vaddr: map.vaddr,
                        size_bytes: map.size,
                    },
                );
                if map.vaddr != 0 {
                    st.iommu_by_vaddr.insert(map.vaddr, map.size);
                }
                Ok(0)
            }
            VFIO_IOMMU_UNMAP_DMA_IOCTL => {
                let unmap: VfioDmaUnmapIoctl = pod_from_prefix(payload);
                if let Some(removed) = st.iommu_by_iova.remove(&unmap.iova) {
                    st.iommu_by_vaddr.remove(&removed.host_vaddr);
                }
                Ok(0)
            }
            VFIO_GET_API_VERSION
            | VFIO_GROUP_SET_CONTAINER
            | GASKET_IOCTL_NUMBER_PAGE_TABLES
            | GASKET_IOCTL_PARTITION_PAGE_TABLE
            | GASKET_IOCTL_PAGE_TABLE_SIZE
            | GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS => Ok(0),
            GASKET_IOCTL_MAP_BUFFER => {
                let pt: GasketPageTableIoctl = pod_from_prefix(payload);
                st.iommu_by_iova.insert(
                    pt.device_address,
                    VirtualIommuEntry {
                        host_vaddr: pt.host_address,
                        size_bytes: pt.size,
                    },
                );
                if pt.host_address != 0 {
                    st.iommu_by_vaddr.insert(pt.host_address, pt.size);
                }
                Ok(0)
            }
            GASKET_IOCTL_UNMAP_BUFFER => {
                let pt: GasketPageTableIoctl = pod_from_prefix(payload);
                if let Some(removed) = st.iommu_by_iova.remove(&pt.device_address) {
                    st.iommu_by_vaddr.remove(&removed.host_vaddr);
                }
                Ok(0)
            }
            GASKET_IOCTL_MAP_DMABUF => {
                let dmabuf: GasketPageTableDmaBuf = pod_from_prefix(payload);
                st.iommu_by_iova.insert(
                    dmabuf.device_address,
                    VirtualIommuEntry {
                        host_vaddr: 0,
                        size_bytes: 4096,
                    },
                );
                Ok(0)
            }
            GASKET_IOCTL_SET_EVENTFD => {
                let efd: GasketInterruptEventFd = pod_from_prefix(payload);
                let irq_idx = u32::try_from(efd.interrupt).unwrap_or(0);
                let os_fd = i32::try_from(efd.event_fd).unwrap_or(-1);
                if os_fd >= 0 {
                    st.interrupt_fds.insert(irq_idx, os_fd);
                }
                Ok(0)
            }
            GASKET_IOCTL_CLEAR_EVENTFD => {
                if let Some(&[b0, b1, b2, b3, b4, b5, b6, b7]) = payload.first_chunk::<8>() {
                    let idx = u32::try_from(u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]))
                        .unwrap_or(0);
                    st.interrupt_fds.remove(&idx);
                } else {
                    st.interrupt_fds.clear();
                }
                Ok(0)
            }
            _ => Err(LibTpuError::IoctlFailed {
                request,
                errno: 25,
                message: format!("Unsupported VirtualVfioGasketDevice ioctl 0x{request:x}"),
            }),
        }
    }

    /// Creates a new [`VirtualVfioGasketDevice`] for `generation` with `chip_count` chips attached
    /// to `ici_fabric`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Emulated Cloud TPU hardware generation (`V2..=V7x`).
    /// * `chip_count` - Number of logical chips (`>= 1`).
    /// * `ici_fabric` - Shared [`VirtualIciTorusFabric`] for multi-chip RDMA & collectives.
    #[must_use]
    pub fn new(
        generation: TpuGeneration,
        chip_count: usize,
        ici_fabric: Arc<VirtualIciTorusFabric>,
    ) -> Self {
        let mut st = VirtualDeviceState {
            generation,
            open_fds: BTreeSet::new(),
            vfio_iommu_enabled: false,
            iommu_by_iova: BTreeMap::new(),
            iommu_by_vaddr: BTreeMap::new(),
            interrupt_fds: BTreeMap::new(),
            bar0_csrs: BTreeMap::new(),
            bar2_words: BTreeMap::new(),
            chip_hbms: BTreeMap::new(),
            ici_fabric,
            cycle_counter: 0,
            trace_cursor: 0,
        };
        st.reset_generation_csrs();
        for idx in 0..chip_count.max(1) {
            let dev_id = DeviceId(u32::try_from(idx).unwrap_or(0));
            let _ = st.ensure_chip_hbm(dev_id);
        }
        Self {
            state: Arc::new(Mutex::new(st)),
            next_fd: Arc::new(AtomicI32::new(200)),
        }
    }

    /// Creates a [`VirtualVfioGasketDevice`] configured from `config` (`generation` & logical device count).
    ///
    /// # Arguments
    ///
    /// * `config` - [`TpuInitConfig`] initialization parameters.
    #[must_use]
    pub fn from_config(config: &TpuInitConfig) -> Self {
        let chip_gen = config.generation;
        let total_hosts = (config.host_bounds.x as usize)
            .saturating_mul(config.host_bounds.y as usize)
            .saturating_mul(config.host_bounds.z as usize)
            .max(1);
        let devs_per_chip = chip_gen.logical_devices_per_chip() as usize;
        let total_devs = total_hosts
            .saturating_mul(config.chips_per_host() as usize)
            .saturating_mul(devs_per_chip)
            .max(1);
        Self::new(chip_gen, total_devs, VirtualIciTorusFabric::new())
    }

    /// Returns a shared handle to `device_id`'s [`SparseHbmStore`] for white-box hardware verification.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Logical [`DeviceId`].
    #[must_use]
    pub fn chip_hbm(&self, device_id: DeviceId) -> Arc<Mutex<SparseHbmStore>> {
        self.state.lock().ensure_chip_hbm(device_id)
    }

    /// Returns a shared handle to the underlying [`VirtualIciTorusFabric`].
    #[must_use]
    pub fn ici_fabric(&self) -> Arc<VirtualIciTorusFabric> {
        Arc::clone(&self.state.lock().ici_fabric)
    }

    /// Returns the number of active IOMMU DMA mappings currently programmed into the device.
    #[must_use]
    pub fn active_iommu_mapping_count(&self) -> usize {
        self.state.lock().iommu_by_iova.len()
    }

    /// Returns the current hardware cycle counter.
    #[must_use]
    pub fn cycle_counter(&self) -> u64 {
        self.state.lock().cycle_counter
    }
}

impl Bar0CsrReader for VirtualVfioGasketDevice {
    fn read_chip_csr_u64(&self, offset: MmioOffset) -> LibTpuResult<u64> {
        self.mmio_read_u64(0, offset)
    }
}

impl SyscallProvider for VirtualVfioGasketDevice {
    fn open_device(&self, path: &str) -> LibTpuResult<i32> {
        if path.is_empty() || path.contains('\0') {
            return Err(LibTpuError::InvalidArgument {
                parameter: "path",
                reason: String::from("Invalid virtual device node path"),
            });
        }
        let fd = self.next_fd.fetch_add(1, Ordering::Relaxed);
        self.state.lock().open_fds.insert(fd);
        Ok(fd)
    }

    fn ioctl_device(&self, fd: i32, request: u64, payload_size: usize) -> LibTpuResult<i32> {
        let mut scratch = vec![0_u8; payload_size.max(32)];
        self.ioctl_bytes(fd, request, &mut scratch)
    }

    fn ioctl_bytes(&self, fd: i32, request: u64, payload: &mut [u8]) -> LibTpuResult<i32> {
        if fd < 0 {
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: 9,
                message: String::from("VirtualVfioGasketDevice::ioctl_bytes negative fd"),
            });
        }
        Self::dispatch_ioctl_inner(&mut self.state.lock(), &self.next_fd, request, payload)
    }

    fn close_device(&self, fd: i32) -> LibTpuResult<()> {
        if fd < 0 {
            return Err(LibTpuError::HardwareIoError {
                operation: "close(VirtualVfioGasketDevice)",
                errno: 9,
                message: String::from("Negative virtual device fd"),
            });
        }
        let _ = self.state.lock().open_fds.remove(&fd);
        Ok(())
    }

    fn mmio_read_u64(&self, bar_index: u8, offset: MmioOffset) -> LibTpuResult<u64> {
        let st = self.state.lock();
        match bar_index {
            0 => Ok(st.bar0_csrs.get(&offset.0).copied().unwrap_or(0)),
            2 => Ok(st.bar2_words.get(&offset.0).copied().unwrap_or(0)),
            4 => Ok(0),
            _ => Err(LibTpuError::InvalidArgument {
                parameter: "bar_index",
                reason: format!("Unsupported PCIe BAR index {bar_index}"),
            }),
        }
    }

    fn mmio_write_u64(&self, bar_index: u8, offset: MmioOffset, value: u64) -> LibTpuResult<()> {
        match bar_index {
            0 => {
                let mut st = self.state.lock();
                st.bar0_csrs.insert(offset.0, value);
                if offset.0 == BAR0_CSR_HBM_DMA_RING_TAIL {
                    st.process_dma_ring_doorbell(value)?;
                }
                drop(st);
                Ok(())
            }
            2 => {
                self.state.lock().bar2_words.insert(offset.0, value);
                Ok(())
            }
            4 => Ok(()),
            _ => Err(LibTpuError::InvalidArgument {
                parameter: "bar_index",
                reason: format!("Unsupported PCIe BAR index {bar_index}"),
            }),
        }
    }

    fn is_hardware_emulated_device(&self) -> bool {
        true
    }
}
