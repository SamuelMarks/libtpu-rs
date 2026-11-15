//! Exhaustive Phase 7 test suite covering `pcie`, `iommu`, `emulator` (`VirtualVfioGasketDevice`
//! & `ArchitecturalBitStreamEngine`), and non-simulated `LinuxDeviceHal`.

use super::emulator::{
    ArchitecturalBitStreamEngine, BAR2_DMA_RING_OFFSET, BAR2_INPUT_ADDR_TABLE_OFFSET,
    BAR2_INPUT_COUNT_CSR_OFFSET, BAR2_INPUT_LEN_TABLE_OFFSET, BAR2_OUTPUT_ADDR_TABLE_OFFSET,
    BAR2_OUTPUT_COUNT_CSR_OFFSET, BAR2_OUTPUT_LEN_TABLE_OFFSET, BAR2_SC_IMEM_LEN_CSR_OFFSET,
    HBM_IMEM_SPARSECORE_BASE, HBM_IMEM_TENSORCORE_BASE, HBM_XPLANE_TRACE_RING_BASE,
    HardwareTraceRingRecord, SparseHbmStore, VirtualIciTorusFabric, VirtualVfioGasketDevice,
    encode_device_hbm_iova, unpack_device_hbm_iova,
};
use super::iommu::{
    BuddySlabHbmAllocator, DmaBufPeerRecord, DmaBufPeerRegistry, GasketIommuSession,
    HugepageStagingPool, PinnedDmaBuffer, VfioIommuSession,
};
use super::pcie::{Bar0CsrReader, HardwareCsrProber, PcieBarResource, SysfsPcieEnumerator};
use super::{
    BAR0_CSR_HBM_DMA_RING_TAIL, BAR0_CSR_TENSORCORE_STATUS, ChipRegisterMap,
    GASKET_IOCTL_CLEAR_EVENTFD, GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS, GASKET_IOCTL_PAGE_TABLE_SIZE,
    HalBackend, HardwareDmaCommandDescriptor, KernelDriverMode, LinuxDeviceHal,
    PosixKernelDriverBackend, SyscallProvider, TensorCoreStatusFlags, TpuHal,
    VFIO_DEVICE_SET_IRQS_IOCTL, VfioIrqSetIoctl,
};
use crate::compiler::isa::encode_bundle_for_generation;
use crate::compiler::sparsecore::{
    SparseCoreOp, SparseCoreSubcoreKind, SparseCoreV2Codec, SparseCoreV2Instruction,
    SparseCoreV3Codec, SparseCoreV3Instruction,
};
use crate::compiler::{CompiledProgram, LloBundle};
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::HostCommandEnvelopeProto;
use crate::types::{
    ByteSize, ChipCoordinates, DeviceAddress, DeviceId, ElementType, FileDescriptor, MemoryKind,
    MmioOffset, TpuGeneration,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Mock `Bar0CsrReader` capable of injecting CSR values or read errors.
#[derive(Debug)]
struct TestCsrReader {
    /// Map of CSR byte offset to 64-bit value.
    csrs: BTreeMap<u64, u64>,
    /// Optional offset that triggers a simulated hardware read error.
    fail_on_offset: Option<u64>,
}

impl Bar0CsrReader for TestCsrReader {
    fn read_chip_csr_u64(&self, offset: MmioOffset) -> LibTpuResult<u64> {
        if self.fail_on_offset == Some(offset.0) {
            return Err(LibTpuError::HardwareIoError {
                operation: "TestCsrReader::read_chip_csr_u64",
                errno: 5,
                message: String::from("Injected BAR0 CSR read error"),
            });
        }
        Ok(self.csrs.get(&offset.0).copied().unwrap_or(0))
    }
}

/// Mock `SyscallProvider` returning `0` from `VFIO_GROUP_GET_DEVICE_FD` to cover fallback branch.
#[derive(Debug)]
struct ZeroDevFdSyscalls;

impl SyscallProvider for ZeroDevFdSyscalls {
    fn open_device(&self, _path: &str) -> LibTpuResult<i32> {
        Ok(42)
    }

    fn ioctl_device(&self, _fd: i32, _request: u64, _payload_size: usize) -> LibTpuResult<i32> {
        Ok(0)
    }

    fn ioctl_bytes(&self, _fd: i32, _request: u64, _payload: &mut [u8]) -> LibTpuResult<i32> {
        Ok(0)
    }

    fn close_device(&self, _fd: i32) -> LibTpuResult<()> {
        Ok(())
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase7_sysfs_pcie_enumerator_and_hardware_csr_prober_cover_all_branches() -> LibTpuResult<()> {
    let bar_populated = PcieBarResource {
        bar_index: 2,
        phys_start: 0x2000_0000,
        phys_end: 0x23ff_ffff,
        flags: 0x0014_220c,
        size_bytes: 0x0400_0000,
    };
    assert!(bar_populated.is_mmio_populated());
    assert!(bar_populated.is_prefetchable());

    let bar_empty = PcieBarResource {
        bar_index: 1,
        phys_start: 0,
        phys_end: 0,
        flags: 0,
        size_bytes: 0,
    };
    assert!(!bar_empty.is_mmio_populated());
    assert!(!bar_empty.is_prefetchable());

    let id_cases = [
        (0x0027, 0, 0, Some(TpuGeneration::V2)),
        (0x0027, 1, 0, Some(TpuGeneration::V3)),
        (0x0056, 0, 0, Some(TpuGeneration::V4)),
        (0x0056, 0, 0x004e, Some(TpuGeneration::V4Lite)),
        (0x005e, 0, 0, Some(TpuGeneration::V4Lite)),
        (0x0063, 0, 0, Some(TpuGeneration::V5e)),
        (0x0062, 0, 0, Some(TpuGeneration::V5p)),
        (0x006f, 0, 0, Some(TpuGeneration::V6e)),
        (0x0073, 0, 0, Some(TpuGeneration::V7x)),
        (0x0076, 0, 0, Some(TpuGeneration::V7x)),
        (0x007f, 0, 0, Some(TpuGeneration::V7x)),
        (0x9999, 0, 0, None),
    ];
    for (dev_id, rev, subsys, expected) in id_cases {
        assert_eq!(
            SysfsPcieEnumerator::classify_pci_ids(dev_id, rev, subsys),
            expected
        );
    }

    let resource_text = concat!(
        "0x00000000a0000000 0x00000000a00fffff 0x0000000000140204\n",
        "0x0000000000000000 0x0000000000000000 0x0000000000000000\n",
        "0x0000001000000000 0x0000001003ffffff 0x000000000014220c\n",
        "0x0000000000000200 0x0000000000000100 0x0000000000000200\n",
        "0x00000000a0200000 0x00000000a020ffff 0x0000000000040200\n",
        "malformed line\n",
    );
    assert!(SysfsPcieEnumerator::parse_bar_resource(resource_text, 0).is_mmio_populated());
    assert!(!SysfsPcieEnumerator::parse_bar_resource(resource_text, 1).is_mmio_populated());
    assert!(SysfsPcieEnumerator::parse_bar_resource(resource_text, 2).is_prefetchable());
    assert!(!SysfsPcieEnumerator::parse_bar_resource(resource_text, 3).is_mmio_populated());
    assert!(SysfsPcieEnumerator::parse_bar_resource(resource_text, 4).is_mmio_populated());
    assert!(!SysfsPcieEnumerator::parse_bar_resource(resource_text, 5).is_mmio_populated());
    assert!(!SysfsPcieEnumerator::parse_bar_resource(resource_text, 99).is_mmio_populated());

    let missing_dir = std::path::Path::new("/nonexistent_sysfs_phase7_test_dir_xyz");
    assert!(SysfsPcieEnumerator::enumerate_devices(missing_dir, missing_dir).is_err());

    let tmp_base = std::env::temp_dir().join(format!("libtpu_phase7_sysfs_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_base);
    let sysfs_pci = tmp_base.join("sys_pci");
    let dev_root = tmp_base.join("dev");
    std::fs::create_dir_all(dev_root.join("vfio"))?;
    std::fs::write(dev_root.join("vfio").join("7"), b"")?;
    std::fs::write(dev_root.join("accel0"), b"")?;

    std::fs::create_dir_all(sysfs_pci.join("not_a_bdf"))?;
    let non_google = sysfs_pci.join("0000:00:01.0");
    std::fs::create_dir_all(&non_google)?;
    std::fs::write(
        non_google.join("vendor"),
        "0x8086
",
    )?;

    let unknown_dev = sysfs_pci.join("0000:00:02.0");
    std::fs::create_dir_all(&unknown_dev)?;
    std::fs::write(
        unknown_dev.join("vendor"),
        "0x1ae0
",
    )?;
    std::fs::write(
        unknown_dev.join("device"),
        "0x9999
",
    )?;

    let no_dev_file = sysfs_pci.join("0000:00:03.0");
    std::fs::create_dir_all(&no_dev_file)?;
    std::fs::write(
        no_dev_file.join("vendor"),
        "0x1ae0
",
    )?;

    let vfio_fn = sysfs_pci.join("0000:00:04.0");
    std::fs::create_dir_all(&vfio_fn)?;
    std::fs::write(
        vfio_fn.join("vendor"),
        "0x1ae0
",
    )?;
    std::fs::write(
        vfio_fn.join("device"),
        "0x007f
",
    )?;
    std::fs::write(
        vfio_fn.join("numa_node"),
        "1
",
    )?;
    std::fs::write(
        vfio_fn.join("local_cpulist"),
        "0-3
",
    )?;
    std::fs::write(
        vfio_fn.join("current_link_speed"),
        "32.0 GT/s PCIe
",
    )?;
    std::fs::write(
        vfio_fn.join("current_link_width"),
        "16
",
    )?;
    std::fs::write(vfio_fn.join("resource"), resource_text)?;
    let drv_target = tmp_base.join("drivers").join("vfio-pci");
    let grp_target = tmp_base.join("iommu_groups").join("7");
    std::fs::create_dir_all(&drv_target)?;
    std::fs::create_dir_all(&grp_target)?;
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&drv_target, vfio_fn.join("driver"))?;
        std::os::unix::fs::symlink(&grp_target, vfio_fn.join("iommu_group"))?;
    }

    let gasket_fn = sysfs_pci.join("0000:00:05.0");
    std::fs::create_dir_all(gasket_fn.join("accel").join("accel0"))?;
    std::fs::write(
        gasket_fn.join("vendor"),
        "0x1ae0
",
    )?;
    std::fs::write(
        gasket_fn.join("device"),
        "0x0063
",
    )?;
    std::fs::write(
        gasket_fn.join("numa_node"),
        "-1
",
    )?;

    let gasket_fallback_fn = sysfs_pci.join("0000:00:06.0");
    std::fs::create_dir_all(&gasket_fallback_fn)?;
    std::fs::write(
        gasket_fallback_fn.join("vendor"),
        "0x1ae0
",
    )?;
    std::fs::write(
        gasket_fallback_fn.join("device"),
        "0x0056
",
    )?;

    let enumerated = SysfsPcieEnumerator::enumerate_devices(&sysfs_pci, &dev_root)?;
    assert_eq!(enumerated.len(), 3);
    if let Some(first) = enumerated.first() {
        assert_eq!(first.generation, TpuGeneration::V7x);
        assert_eq!(first.driver_mode, KernelDriverMode::Vfio);
        assert_eq!(first.iommu_group, Some(7));
        assert_eq!(first.numa_node, 1);
        assert!(first.bar0.size_bytes > 0);
        assert!(first.bar2.size_bytes > 0);
        assert!(first.bar4.size_bytes > 0);
    }
    if let Some(second) = enumerated.get(1) {
        assert_eq!(second.generation, TpuGeneration::V5e);
        assert_eq!(second.numa_node, -1);
    }
    let _ = std::fs::remove_dir_all(&tmp_base);

    let posix = PosixKernelDriverBackend::new();
    let bar0_mmap = posix.mmap_bar_aperture(FileDescriptor(-1), ByteSize(4096))?;
    let rmap_v5e = ChipRegisterMap::for_generation(TpuGeneration::V5e);
    let packed_id =
        HardwareCsrProber::pack_global_chip_id_csr(11, ChipCoordinates::new(2, 3, 1), 4, 0x0063);
    bar0_mmap.write_csr_u64(MmioOffset(0x100), packed_id)?;
    assert_eq!(bar0_mmap.read_chip_csr_u64(MmioOffset(0x100))?, packed_id);
    posix.munmap_bar_aperture(bar0_mmap)?;

    let mut id_reader = TestCsrReader {
        csrs: BTreeMap::new(),
        fail_on_offset: None,
    };
    id_reader
        .csrs
        .insert(rmap_v5e.global_chip_id_csr, packed_id);
    let probed_mmap = HardwareCsrProber::probe_chip_identity(TpuGeneration::V5e, &id_reader)?;
    assert_eq!(probed_mmap.global_chip_id, 11);
    assert_eq!(probed_mmap.coordinates, ChipCoordinates::new(2, 3, 1));
    assert_eq!(probed_mmap.tray_slot_index, 4);
    assert_eq!(probed_mmap.sku_code, 0x0063);

    for chip_gen in [
        TpuGeneration::V2,
        TpuGeneration::V3,
        TpuGeneration::V4,
        TpuGeneration::V4Lite,
        TpuGeneration::V5e,
        TpuGeneration::V5p,
        TpuGeneration::V6e,
        TpuGeneration::V7x,
    ] {
        let mut reader = TestCsrReader {
            csrs: BTreeMap::new(),
            fail_on_offset: None,
        };
        let port0 = HardwareCsrProber::ici_port_status_csr_offset(chip_gen, 0);
        reader.csrs.insert(
            port0.0,
            HardwareCsrProber::pack_ici_port_status_csr(true, 400, 9, 3),
        );
        let ident = HardwareCsrProber::probe_chip_identity(chip_gen, &reader)?;
        assert_eq!(ident.sku_code, chip_gen.pcie_device_id());
        assert!(
            ident
                .ici_ports
                .first()
                .is_some_and(|p| p.link_up && p.fec_corrected_errors == 3)
        );

        reader.fail_on_offset = Some(port0.0);
        assert!(HardwareCsrProber::probe_chip_identity(chip_gen, &reader).is_err());
    }
    Ok(())
}

#[test]
fn phase7_iommu_buddy_slab_hugepage_dmabuf_vfio_and_gasket_sessions() -> LibTpuResult<()> {
    let mut alloc = BuddySlabHbmAllocator::new(0x1000_0000, 64 * 1024);
    assert!(alloc.allocate(0).is_err());
    assert!(alloc.allocate(1024 * 1024).is_err());

    let s256 = alloc.allocate(128)?;
    let s512 = alloc.allocate(400)?;
    let s1024 = alloc.allocate(900)?;
    let s2048 = alloc.allocate(1800)?;
    assert_eq!(alloc.allocated_block_bytes(s256), Some(256));
    assert!(
        alloc
            .resolve_range(DeviceAddress(s256.0 + 64), 64)
            .is_some()
    );
    assert!(
        alloc
            .resolve_range(DeviceAddress(s256.0 + 200), 128)
            .is_none()
    );
    assert!(
        alloc
            .resolve_range(DeviceAddress(0x0000_1000), 16)
            .is_none()
    );

    let _ = alloc.deallocate(s256)?;
    let s256_reused = alloc.allocate(200)?;
    assert_eq!(s256_reused, s256);

    let b4k = alloc.allocate(4096)?;
    let b8k = alloc.allocate(8192)?;
    assert!(alloc.stats().bytes_in_use > 0);
    alloc.clear_peak_stats();
    let _ = alloc.deallocate(b4k)?;
    let _ = alloc.deallocate(b8k)?;
    let _ = alloc.deallocate(s256_reused)?;
    let _ = alloc.deallocate(s512)?;
    let _ = alloc.deallocate(s1024)?;
    let _ = alloc.deallocate(s2048)?;
    assert!(alloc.deallocate(DeviceAddress(0xDEAD_0000)).is_err());

    let mut tiny_alloc = BuddySlabHbmAllocator::new(0x2000_0000, 8192);
    let slab_a = tiny_alloc.allocate(256)?;
    let page_rest = tiny_alloc.allocate(4096)?;
    let _ = tiny_alloc.deallocate(slab_a)?;
    assert_eq!(tiny_alloc.largest_free_block_bytes(), 256);
    let mut exact_alloc = BuddySlabHbmAllocator::new(0x3000_0000, 4096);
    let full = exact_alloc.allocate(4096)?;
    assert_eq!(exact_alloc.largest_free_block_bytes(), 0);
    let _ = exact_alloc.deallocate(full)?;
    let _ = tiny_alloc.deallocate(page_rest)?;

    let large_pinned = PinnedDmaBuffer::allocate(2_097_152)?;
    assert!(large_pinned.capacity() >= 2_097_152);
    drop(large_pinned);

    let mut pool = HugepageStagingPool::new();
    let base_a = DeviceAddress(0x1000_0000);
    assert!(pool.write_staging(DeviceId(0), base_a, 0, &[1, 2]).is_err());
    assert!(pool.d2h_target_vaddr(DeviceId(0), base_a, 0, 4).is_err());
    let mut tmp_read = [0_u8; 4];
    assert!(
        pool.read_staging(DeviceId(0), base_a, 0, &mut tmp_read)
            .is_err()
    );

    let (_vaddr, cap) = pool.register_buffer(DeviceId(0), base_a, 4096)?;
    assert!(
        pool.write_staging(DeviceId(0), base_a, cap, &[1, 2])
            .is_err()
    );
    assert!(pool.d2h_target_vaddr(DeviceId(0), base_a, cap, 8).is_err());
    assert!(
        pool.read_staging(DeviceId(0), base_a, cap, &mut tmp_read)
            .is_err()
    );
    let _ = pool.write_staging(DeviceId(0), base_a, 16, &[10, 20, 30, 40])?;
    pool.read_staging(DeviceId(0), base_a, 16, &mut tmp_read)?;
    assert_eq!(tmp_read, [10, 20, 30, 40]);
    assert!(pool.snapshot_buffer(DeviceId(0), base_a).is_some());
    assert!(pool.unregister_buffer(DeviceId(0), base_a).is_some());
    Ok(())
}

#[test]
fn phase7_iommu_dmabuf_vfio_and_gasket_sessions() -> LibTpuResult<()> {
    let base_a = DeviceAddress(0x1000_0000);
    let mut reg = DmaBufPeerRegistry::new();
    assert!(reg.is_empty());
    reg.insert(DmaBufPeerRecord {
        dmabuf_fd: 77,
        device_id: DeviceId(0),
        device_iova: base_a,
        size_bytes: 4096,
        is_exported: true,
    });
    assert!(!reg.is_empty());
    assert_eq!(reg.len(), 1);
    assert!(reg.get(77).is_some());

    let vdev = VirtualVfioGasketDevice::default();
    let vfio = VfioIommuSession::open_and_initialize(&vdev, 0, "0000:00:04.0")?;
    let pinned = PinnedDmaBuffer::allocate(4096)?;
    vfio.map_dma(&vdev, pinned.host_vaddr(), 0x5000_0000, 4096)?;
    assert_eq!(vdev.active_iommu_mapping_count(), 1);
    let _ = vfio.unmap_dma(&vdev, 0x5000_0000, 4096)?;
    assert_eq!(vdev.active_iommu_mapping_count(), 0);
    let bar0 = vfio.mmap_bar_region(&vdev, 0, ByteSize(4096))?;
    let bar2 = vfio.mmap_bar_region(&vdev, 2, ByteSize(4096))?;
    let bar4 = vfio.mmap_bar_region(&vdev, 4, ByteSize(4096))?;
    let posix = PosixKernelDriverBackend::new();
    posix.munmap_bar_aperture(bar0)?;
    posix.munmap_bar_aperture(bar2)?;
    posix.munmap_bar_aperture(bar4)?;
    assert!(vfio.bind_msix_eventfd(&vdev, 0, -1).is_err());
    vfio.bind_msix_eventfd(&vdev, 0, 12)?;
    vfio.hot_reset(&vdev)?;
    vfio.close_all(&vdev)?;

    let vfio_zero = VfioIommuSession::open_and_initialize(&ZeroDevFdSyscalls, 0, "0000:00:04.0")?;
    let bar_z = vfio_zero.mmap_bar_region(&ZeroDevFdSyscalls, 0, ByteSize(4096))?;
    posix.munmap_bar_aperture(bar_z)?;
    vfio_zero.close_all(&ZeroDevFdSyscalls)?;

    let gasket = GasketIommuSession::open_and_initialize(&vdev, "/dev/accel0")?;
    gasket.map_buffer(&vdev, pinned.host_vaddr(), 0x6000_0000, 4096)?;
    gasket.unmap_buffer(&vdev, 0x6000_0000, 4096)?;
    assert!(gasket.map_dmabuf(&vdev, -1, 0x6000_0000, 3).is_err());
    gasket.map_dmabuf(&vdev, 9, 0x6000_0000, 3)?;
    assert!(gasket.set_eventfd(&vdev, 0, -1).is_err());
    gasket.set_eventfd(&vdev, 0, 15)?;
    gasket.clear_eventfd(&vdev, 0)?;
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase7_emulator_bitstream_engine_and_virtual_device_faults() -> LibTpuResult<()> {
    let mut store = SparseHbmStore::new();
    let cross_page = vec![0xAB_u8; 9000];
    let ironwood_high_addr = 120_u64 * 1024 * 1024 * 1024 + 4000;
    store.write_bytes(ironwood_high_addr, &cross_page);
    assert_eq!(store.read_bytes(ironwood_high_addr, 9000), cross_page);
    assert_eq!(store.read_bytes(0x0020_0000, 32), vec![0_u8; 32]);

    assert!(HardwareTraceRingRecord::from_le_bytes(&[0_u8; 16]).is_none());
    let rec = HardwareTraceRingRecord {
        start_cycle: 10,
        duration_cycles: 4,
        pc_offset: 64,
        flops_retired: 2048,
        hbm_bytes_transferred: 512,
        unit_tag: 1,
    };
    assert_eq!(
        HardwareTraceRingRecord::from_le_bytes(&rec.to_le_bytes()),
        Some(rec)
    );

    let fabric = VirtualIciTorusFabric::new();
    let hbm0 = Arc::new(Mutex::new(SparseHbmStore::new()));
    let hbm1 = Arc::new(Mutex::new(SparseHbmStore::new()));
    assert!(
        fabric
            .rdma_copy(DeviceId(0), 0, DeviceId(1), 0, 16)
            .is_err()
    );
    fabric.register_chip(DeviceId(0), Arc::clone(&hbm0));
    assert!(
        fabric
            .rdma_copy(DeviceId(0), 0, DeviceId(1), 0, 16)
            .is_err()
    );
    fabric.register_chip(DeviceId(1), Arc::clone(&hbm1));
    hbm0.lock().write_bytes(0x1000, &[1, 2, 3, 4]);
    fabric.rdma_copy(DeviceId(0), 0x1000, DeviceId(1), 0x2000, 4)?;
    assert_eq!(hbm1.lock().read_bytes(0x2000, 4), vec![1, 2, 3, 4]);

    let a_bytes = ElementType::F32.pack_from_f64_slice(&[2.0, 8.0]);
    let b_bytes = ElementType::F32.pack_from_f64_slice(&[5.0, 3.0]);
    let _ = fabric.post_and_reduce_collective(10, DeviceId(0), 14, ElementType::F32, &a_bytes);
    let sum_out =
        fabric.post_and_reduce_collective(10, DeviceId(1), 14, ElementType::F32, &b_bytes);
    assert_eq!(
        ElementType::F32.unpack_to_f64_vec(&sum_out),
        vec![7.0, 11.0]
    );
    let _ = fabric.post_and_reduce_collective(11, DeviceId(0), 16, ElementType::F32, &a_bytes);
    let max_out =
        fabric.post_and_reduce_collective(11, DeviceId(1), 16, ElementType::F32, &b_bytes);
    assert_eq!(ElementType::F32.unpack_to_f64_vec(&max_out), vec![5.0, 8.0]);
    let _ = fabric.post_and_reduce_collective(12, DeviceId(0), 17, ElementType::F32, &a_bytes);
    let min_out =
        fabric.post_and_reduce_collective(12, DeviceId(1), 17, ElementType::F32, &b_bytes);
    assert_eq!(ElementType::F32.unpack_to_f64_vec(&min_out), vec![2.0, 3.0]);

    let mut cursor = 0_u32;
    assert!(
        ArchitecturalBitStreamEngine::execute_from_hbm(
            TpuGeneration::V7x,
            DeviceId(0),
            &hbm0,
            Some(&fabric),
            HBM_IMEM_TENSORCORE_BASE,
            7,
            0,
            0,
            &[],
            &[],
            0,
            &mut cursor,
        )
        .is_err()
    );

    let mut raw_bundle = LloBundle {
        generation: TpuGeneration::V7x,
        ..LloBundle::default()
    };
    raw_bundle.vector_slots[0].opcode = 1;
    raw_bundle.vector_slots[0].dst_vreg = 2;
    raw_bundle.vector_slots[0].src0_vreg = 0;
    raw_bundle.vector_slots[0].src1_vreg = 1;
    raw_bundle.vector_slots[0].element_type = 11;
    raw_bundle.dma_slot.dma_opcode = 14;
    raw_bundle.dma_slot.sync_flag_id = 5;
    raw_bundle.dma_slot.collective_op = 14;
    let encoded_v7x = encode_bundle_for_generation(TpuGeneration::V7x, &raw_bundle);
    hbm0.lock()
        .write_bytes(HBM_IMEM_TENSORCORE_BASE, &encoded_v7x);
    hbm0.lock().write_bytes(
        0x1000_0000,
        &ElementType::F32.pack_from_f64_slice(&[10.0, 20.0, 30.0, 40.0]),
    );
    hbm0.lock()
        .write_bytes(0x1000_1000, &ElementType::S32.pack_from_i64_slice(&[1, 0]));
    let sc_v3_raw = SparseCoreV3Codec::encode(&SparseCoreV3Instruction {
        subcore: SparseCoreSubcoreKind::Tec,
        op: SparseCoreOp::SpmemGatherDma,
        optimizer_id: 0,
        quant_mode: 0,
        vector_lanes: 16,
        spmem_offset: 0,
        hbm3e_byte_address: 0x1000_0000,
        sync_flag_id: 6,
    });
    hbm0.lock()
        .write_bytes(HBM_IMEM_SPARSECORE_BASE, &sc_v3_raw);

    let summary = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V7x,
        DeviceId(0),
        &hbm0,
        Some(&fabric),
        HBM_IMEM_TENSORCORE_BASE,
        encoded_v7x.len(),
        0,
        sc_v3_raw.len(),
        &[(0x1000_0000, 16), (0x1000_1000, 8)],
        &[(0x1000_2000, 16)],
        0,
        &mut cursor,
    )?;
    assert_eq!(summary.tc_bundles_retired, 1);
    assert_eq!(summary.sc_instructions_retired, 1);
    let gathered_f = ElementType::F32.unpack_to_f64_vec(&hbm0.lock().read_bytes(0x1000_2000, 16));
    assert_eq!(gathered_f, vec![30.0, 40.0, 10.0, 20.0]);

    let sc_v2_raw = SparseCoreV2Codec::encode(&SparseCoreV2Instruction {
        subcore: SparseCoreSubcoreKind::Tec,
        op: SparseCoreOp::SpmemGatherDma,
        optimizer_id: 0,
        spmem_offset: 0,
        hbm_page_offset: 0,
        transfer_bytes: 64,
        sync_flag_id: 3,
    });
    hbm0.lock()
        .write_bytes(HBM_IMEM_SPARSECORE_BASE, &sc_v2_raw);
    let mut mxu_bundle = LloBundle {
        generation: TpuGeneration::V5e,
        ..LloBundle::default()
    };
    mxu_bundle.matrix_slots[0].opcode = 1;
    mxu_bundle.matrix_slots[0].acc_vreg = 2;
    let enc_v5e = encode_bundle_for_generation(TpuGeneration::V5e, &mxu_bundle);
    hbm0.lock().write_bytes(HBM_IMEM_TENSORCORE_BASE, &enc_v5e);
    let summary_v2 = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V5e,
        DeviceId(0),
        &hbm0,
        None,
        HBM_IMEM_TENSORCORE_BASE,
        enc_v5e.len(),
        0,
        sc_v2_raw.len(),
        &[],
        &[(0, 0)],
        10,
        &mut cursor,
    )?;
    assert_eq!(summary_v2.sc_instructions_retired, 1);

    assert!(
        ArchitecturalBitStreamEngine::execute_from_hbm(
            TpuGeneration::V5e,
            DeviceId(0),
            &hbm0,
            None,
            HBM_IMEM_TENSORCORE_BASE,
            enc_v5e.len(),
            0,
            13,
            &[],
            &[],
            0,
            &mut cursor,
        )
        .is_err()
    );

    let mut corrupted = enc_v5e;
    if let Some(first) = corrupted.first_mut() {
        *first ^= 0xFF;
    }
    hbm0.lock()
        .write_bytes(HBM_IMEM_TENSORCORE_BASE, &corrupted);
    assert!(
        ArchitecturalBitStreamEngine::execute_from_hbm(
            TpuGeneration::V5e,
            DeviceId(0),
            &hbm0,
            None,
            HBM_IMEM_TENSORCORE_BASE,
            corrupted.len(),
            0,
            0,
            &[],
            &[],
            0,
            &mut cursor,
        )
        .is_err()
    );

    let encoded_iova = encode_device_hbm_iova(DeviceId(5), 0x1234_5678);
    assert_eq!(
        unpack_device_hbm_iova(encoded_iova),
        (DeviceId(5), 0x1234_5678)
    );

    let vdev = VirtualVfioGasketDevice::default();
    assert!(vdev.open_device("").is_err());
    assert!(vdev.open_device("bad\0path").is_err());
    let fd = vdev.open_device("/dev/accel0")?;
    assert!(
        vdev.ioctl_bytes(-1, GASKET_IOCTL_PAGE_TABLE_SIZE, &mut [0; 8])
            .is_err()
    );
    assert!(
        vdev.ioctl_device(fd, GASKET_IOCTL_PAGE_TABLE_SIZE, 8)
            .is_ok()
    );
    assert!(
        vdev.ioctl_device(fd, GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS, 8)
            .is_ok()
    );
    assert!(
        vdev.ioctl_bytes(fd, GASKET_IOCTL_CLEAR_EVENTFD, &mut [0; 4])
            .is_ok()
    );
    let mut unbind_irq = VfioIrqSetIoctl {
        argsz: 24,
        flags: 0x24,
        index: 2,
        start: 0,
        count: 1,
        event_fd: -1,
    };
    let _ = vdev.ioctl_typed(fd, VFIO_DEVICE_SET_IRQS_IOCTL, &mut unbind_irq)?;
    assert!(vdev.ioctl_device(fd, 0xFFFF_FFFF, 8).is_err());
    assert!(vdev.close_device(-1).is_err());
    vdev.close_device(fd)?;
    assert_eq!(vdev.mmio_read_u64(4, MmioOffset(0))?, 0);
    vdev.mmio_write_u64(4, MmioOffset(0), 1)?;
    assert!(vdev.mmio_read_u64(9, MmioOffset(0)).is_err());
    assert!(vdev.mmio_write_u64(9, MmioOffset(0), 1).is_err());
    let _ = vdev.chip_hbm(DeviceId(0));
    let _ = vdev.ici_fabric();
    let _ = vdev.cycle_counter();

    let bad_h2d = HardwareDmaCommandDescriptor::new(1, 0, 1, 0xDEAD_0000, 0x1000_0000, 64, 0, 0, 1);
    for (i, ch) in bad_h2d.to_bytes().chunks_exact(8).enumerate() {
        let mut arr = [0_u8; 8];
        arr.copy_from_slice(ch);
        vdev.mmio_write_u64(
            2,
            MmioOffset(BAR2_DMA_RING_OFFSET + (i as u64) * 8),
            u64::from_le_bytes(arr),
        )?;
    }
    assert!(
        vdev.mmio_write_u64(0, MmioOffset(BAR0_CSR_HBM_DMA_RING_TAIL), 1)
            .is_err()
    );

    let mut bad_crc = bad_h2d;
    bad_crc.crc32c ^= 1;
    for (i, ch) in bad_crc.to_bytes().chunks_exact(8).enumerate() {
        let mut arr = [0_u8; 8];
        arr.copy_from_slice(ch);
        vdev.mmio_write_u64(
            2,
            MmioOffset(BAR2_DMA_RING_OFFSET + (i as u64) * 8),
            u64::from_le_bytes(arr),
        )?;
    }
    assert!(
        vdev.mmio_write_u64(0, MmioOffset(BAR0_CSR_HBM_DMA_RING_TAIL), 1)
            .is_err()
    );
    assert_ne!(
        vdev.mmio_read_u64(0, MmioOffset(BAR0_CSR_TENSORCORE_STATUS))?
            & TensorCoreStatusFlags::POISONED.bits(),
        0
    );
    assert!(
        vdev.mmio_write_u64(0, MmioOffset(BAR0_CSR_HBM_DMA_RING_TAIL), 2)
            .is_err()
    );
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase7_non_simulated_linux_device_hal_full_hardware_execution_across_generations()
-> LibTpuResult<()> {
    for (chip_gen, mode) in [
        (TpuGeneration::V2, KernelDriverMode::Gasket),
        (TpuGeneration::V4, KernelDriverMode::Vfio),
        (TpuGeneration::V5e, KernelDriverMode::Gasket),
        (TpuGeneration::V6e, KernelDriverMode::Vfio),
        (TpuGeneration::V7x, KernelDriverMode::Gasket),
    ] {
        let cfg = TpuInitConfig {
            generation: chip_gen,
            ..TpuInitConfig::default()
        };
        let vdev = VirtualVfioGasketDevice::from_config(&cfg);
        let hal = Arc::new(LinuxDeviceHal::open(mode, vdev.clone(), &cfg)?);
        let ident = hal.probe_hardware_chip_identity()?;
        assert_eq!(ident.generation, chip_gen);

        let a = hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16))?;
        let b = hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16))?;
        let out = hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16))?;
        let d2d_dst =
            hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16))?;

        let lhs = ElementType::F32.pack_from_f64_slice(&[1.5, 2.5, 3.5, 4.5]);
        let rhs = ElementType::F32.pack_from_f64_slice(&[10.0, 20.0, 30.0, 40.0]);
        hal.write_device_memory(DeviceId(0), a, &lhs)?;
        hal.write_device_memory(DeviceId(0), b, &rhs)?;

        let add_mlir = br"
module @hw_add {
  func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
";
        let prog = CompiledProgram::compile("mlir", add_mlir, &[])?;
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);
        let ev = hal.submit_pipelined_stream_execution(0, DeviceId(0), &prog, &[a, b], &[out])?;
        ev.on_ready(move |err| {
            if err.is_none() {
                fired_clone.store(true, Ordering::SeqCst);
            }
        });
        assert!(fired.load(Ordering::SeqCst));

        let mut host_out = [0_u8; 16];
        hal.read_device_memory(DeviceId(0), out, &mut host_out)?;
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&host_out),
            vec![11.5, 22.5, 33.5, 44.5]
        );

        hal.copy_device_to_device(DeviceId(0), out, DeviceId(0), d2d_dst, ByteSize(16))?;
        let mut d2d_out = [0_u8; 16];
        hal.read_device_memory(DeviceId(0), d2d_dst, &mut d2d_out)?;
        assert_eq!(d2d_out, host_out);

        let mut sc_prog = prog.clone();
        sc_prog.mosaic_metadata.core_type = String::from("sparse_core");
        hal.execute_program(DeviceId(0), &sc_prog, &[a, b], &[out])?;
        let _ = hal.execute_hardware_direct_mode(DeviceId(0), &sc_prog, &[a, b], &[out])?;

        let big_payload = vec![0x5A_u8; 600 * 1024];
        hal.upload_bytes_to_device_hbm(DeviceId(0), 0x0200_0000, &big_payload)?;
        assert_eq!(
            vdev.chip_hbm(DeviceId(0))
                .lock()
                .read_bytes(0x0200_0000, 16),
            vec![0x5A_u8; 16]
        );

        let xspace = hal.harvest_hardware_xplane_traces(DeviceId(0))?;
        assert_eq!(xspace.planes.len(), 1);
        assert!(
            xspace
                .planes
                .first()
                .is_some_and(|p| p.lines.iter().any(|l| !l.events.is_empty()))
        );

        let env1 = HostCommandEnvelopeProto {
            continuation_id: 3,
            opcode: 1,
            handler_id: 0,
            payload: vec![1, 2, 3, 4],
            ..HostCommandEnvelopeProto::default()
        };
        let env2 = HostCommandEnvelopeProto {
            continuation_id: 4,
            opcode: 7,
            handler_id: 1,
            payload: vec![9, 9],
            ..HostCommandEnvelopeProto::default()
        };
        let acks = LinuxDeviceHal::run_host_continuation_service_thread(
            Arc::clone(&hal),
            vec![env1.encode(), env2.encode()],
        )?;
        assert_eq!(acks.len(), 2);

        assert!(
            hal.execute_program(DeviceId(0), &prog, &[a, b], &[])
                .is_err()
        );
        assert!(
            hal.execute_program(DeviceId(0), &prog, &[a], &[out])
                .is_err()
        );
        assert!(
            hal.execute_program(DeviceId(0), &prog, &[DeviceAddress(0xBAD0), b], &[out])
                .is_err()
        );
        assert!(
            hal.execute_program(DeviceId(0), &prog, &[a, b], &[DeviceAddress(0xBAD0)])
                .is_err()
        );
        assert!(
            hal.submit_pipelined_stream_execution(1, DeviceId(0), &prog, &[a], &[out])
                .is_err()
        );

        let st = hal.memory_stats(DeviceId(0))?;
        assert!(st.bytes_in_use > 0);
        hal.clear_memory_stats(DeviceId(0))?;
        assert!(hal.memory_stats(DeviceId(99)).is_err());
        assert!(hal.clear_memory_stats(DeviceId(99)).is_err());
        assert!(
            hal.poison_device(DeviceId(99), String::from("bad"))
                .is_err()
        );

        hal.poison_device(DeviceId(0), String::from("ECC double-bit fault"))?;
        assert!(
            hal.execute_program(DeviceId(0), &prog, &[a, b], &[out])
                .is_err()
        );

        hal.deallocate_device_memory(DeviceId(0), a)?;
        hal.deallocate_device_memory(DeviceId(0), b)?;
        hal.deallocate_device_memory(DeviceId(0), out)?;
        hal.deallocate_device_memory(DeviceId(0), d2d_dst)?;

        let _ = BAR2_INPUT_ADDR_TABLE_OFFSET;
        let _ = BAR2_INPUT_COUNT_CSR_OFFSET;
        let _ = BAR2_INPUT_LEN_TABLE_OFFSET;
        let _ = BAR2_OUTPUT_ADDR_TABLE_OFFSET;
        let _ = BAR2_OUTPUT_COUNT_CSR_OFFSET;
        let _ = BAR2_OUTPUT_LEN_TABLE_OFFSET;
        let _ = BAR2_SC_IMEM_LEN_CSR_OFFSET;
        let _ = HBM_XPLANE_TRACE_RING_BASE;
    }

    let default_backend = HalBackend::from_config(&TpuInitConfig::default());
    assert_eq!(default_backend.topology_spec().platform_name, "tpu");
    let mut sim_cfg = TpuInitConfig::default();
    sim_cfg
        .extra_flags
        .insert(String::from("--tpu_driver_mode"), String::from("simulated"));
    let sim_backend = HalBackend::from_config(&sim_cfg);
    assert_eq!(sim_backend.topology_spec().platform_name, "tpu");
    Ok(())
}

/// Programmable fault-injecting `SyscallProvider` wrapping `VirtualVfioGasketDevice` and failing
/// on the `fail_at_step`-th syscall/MMIO operation (`1`-indexed).
#[derive(Debug)]
struct StepFaultSyscalls {
    /// Backing `VirtualVfioGasketDevice`.
    inner: VirtualVfioGasketDevice,
    /// Monotonic operation counter.
    step: std::sync::atomic::AtomicU32,
    /// 1-based step index at which to inject an error (`0` = never fail).
    fail_at_step: u32,
}

impl StepFaultSyscalls {
    /// Creates a new `StepFaultSyscalls` failing at `fail_at_step`.
    ///
    /// # Arguments
    ///
    /// * `fail_at_step` - 1-based operation index at which to fail.
    fn new(fail_at_step: u32) -> Self {
        Self {
            inner: VirtualVfioGasketDevice::default(),
            step: std::sync::atomic::AtomicU32::new(0),
            fail_at_step,
        }
    }

    /// Increments the step counter and returns `true` when `step == fail_at_step`.
    fn should_fail(&self) -> bool {
        let cur = self.step.fetch_add(1, Ordering::Relaxed) + 1;
        self.fail_at_step != 0 && cur == self.fail_at_step
    }
}

impl SyscallProvider for StepFaultSyscalls {
    fn open_device(&self, path: &str) -> LibTpuResult<i32> {
        if self.should_fail() {
            return Err(LibTpuError::HardwareIoError {
                operation: "StepFaultSyscalls::open_device",
                errno: 13,
                message: String::from("injected open fault"),
            });
        }
        self.inner.open_device(path)
    }

    fn ioctl_device(&self, fd: i32, request: u64, payload_size: usize) -> LibTpuResult<i32> {
        if self.should_fail() {
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: 5,
                message: String::from("injected ioctl_device fault"),
            });
        }
        self.inner.ioctl_device(fd, request, payload_size)
    }

    fn ioctl_bytes(&self, fd: i32, request: u64, payload: &mut [u8]) -> LibTpuResult<i32> {
        if self.should_fail() {
            return Err(LibTpuError::IoctlFailed {
                request,
                errno: 5,
                message: String::from("injected ioctl_bytes fault"),
            });
        }
        self.inner.ioctl_bytes(fd, request, payload)
    }

    fn close_device(&self, fd: i32) -> LibTpuResult<()> {
        if self.should_fail() {
            return Err(LibTpuError::HardwareIoError {
                operation: "StepFaultSyscalls::close_device",
                errno: 9,
                message: String::from("injected close fault"),
            });
        }
        self.inner.close_device(fd)
    }

    fn mmio_read_u64(&self, bar_index: u8, offset: MmioOffset) -> LibTpuResult<u64> {
        if self.should_fail() {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index,
                offset: offset.0,
                size: 8,
            });
        }
        self.inner.mmio_read_u64(bar_index, offset)
    }

    fn mmio_write_u64(&self, bar_index: u8, offset: MmioOffset, value: u64) -> LibTpuResult<()> {
        if self.should_fail() {
            return Err(LibTpuError::MmioOutOfBounds {
                bar_index,
                offset: offset.0,
                size: 8,
            });
        }
        self.inner.mmio_write_u64(bar_index, offset, value)
    }

    fn is_hardware_emulated_device(&self) -> bool {
        true
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase7_100_percent_branch_and_line_closure() -> LibTpuResult<()> {
    // 1. simulated.rs & syscall.rs default trait methods
    let sim = super::SimulatedTpuHal::new(&TpuInitConfig::default());
    assert_eq!(sim.allocated_block_size(DeviceId(0), DeviceAddress(0)), 256);
    assert_eq!(ZeroDevFdSyscalls.mmio_read_u64(0, MmioOffset(0))?, 0);
    ZeroDevFdSyscalls.mmio_write_u64(0, MmioOffset(0), 1)?;

    // 2. pcie.rs remaining branches (`start == 0 && end != 0`, `"0X..."`, plain `iommu_group`, `vfio_candidate`, `link_up = false`)
    assert!(
        SysfsPcieEnumerator::parse_bar_resource(
            "0x0000000000000000 0x0000000000000fff 0x0000000000000200\n",
            0,
        )
        .is_mmio_populated()
    );
    let _ = HardwareCsrProber::pack_ici_port_status_csr(false, 0, 0, 0);

    let tmp_pcie =
        std::env::temp_dir().join(format!("libtpu_phase7_pcie_extra_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_pcie);
    let sys_p = tmp_pcie.join("sys");
    let dev_p = tmp_pcie.join("dev");
    std::fs::create_dir_all(dev_p.join("vfio"))?;
    std::fs::write(dev_p.join("vfio").join("8"), b"")?;
    std::fs::write(dev_p.join("vfio").join("9"), b"")?;
    std::fs::write(dev_p.join("accel9"), b"")?;

    let fn_vfio_only = sys_p.join("0000:00:08.0");
    std::fs::create_dir_all(&fn_vfio_only)?;
    std::fs::write(fn_vfio_only.join("vendor"), "0X1AE0\n")?;
    std::fs::write(fn_vfio_only.join("device"), "0X0063\n")?;
    std::fs::write(fn_vfio_only.join("iommu_group"), "8\n")?;

    let fn_both = sys_p.join("0000:00:09.0");
    std::fs::create_dir_all(fn_both.join("accel").join("accel9"))?;
    std::fs::write(fn_both.join("vendor"), "0x1ae0\n")?;
    std::fs::write(fn_both.join("device"), "0x0063\n")?;
    std::fs::write(fn_both.join("iommu_group"), "9\n")?;

    let extra_enum = SysfsPcieEnumerator::enumerate_devices(&sys_p, &dev_p)?;
    assert_eq!(extra_enum.len(), 2);
    let _ = std::fs::remove_dir_all(&tmp_pcie);

    // 3. iommu.rs remaining branches (`max_order == 30`, non-pow2 `within_cap == false`, slab/buddy OOM, `MAP_FAILED`, step-faulting ioctls)
    let _huge_cap_buddy = BuddySlabHbmAllocator::new(0x1000_0000, 8_u64 << 40);
    let mut three_page = BuddySlabHbmAllocator::new(0x1000_0000, 12 * 1024);
    let p0 = three_page.allocate(4096)?;
    let p1 = three_page.allocate(4096)?;
    let p2 = three_page.allocate(4096)?;
    let _ = three_page.deallocate(p2)?;
    let _ = three_page.deallocate(p1)?;
    let _ = three_page.deallocate(p0)?;

    let mut one_page = BuddySlabHbmAllocator::new(0x1000_0000, 4096);
    let only_pg = one_page.allocate(4096)?;
    assert!(one_page.allocate(128).is_err());
    assert!(one_page.allocate(4096).is_err());
    let _ = one_page.deallocate(only_pg)?;

    assert!(PinnedDmaBuffer::allocate(isize::MAX as usize).is_err());

    for fail_step in 1..=12 {
        let fault = StepFaultSyscalls::new(fail_step);
        let _ = VfioIommuSession::open_and_initialize(&fault, 0, "0000:00:04.0");
        let fault_g = StepFaultSyscalls::new(fail_step);
        let _ = GasketIommuSession::open_and_initialize(&fault_g, "/dev/accel0");
    }
    let ok_vfio =
        VfioIommuSession::open_and_initialize(&StepFaultSyscalls::new(0), 0, "0000:00:04.0")?;
    assert!(
        ok_vfio
            .map_dma(&StepFaultSyscalls::new(1), 0x1000, 0x2000, 4096)
            .is_err()
    );
    assert!(
        ok_vfio
            .unmap_dma(&StepFaultSyscalls::new(1), 0x2000, 4096)
            .is_err()
    );
    assert!(
        ok_vfio
            .mmap_bar_region(&StepFaultSyscalls::new(1), 0, ByteSize(4096))
            .is_err()
    );
    assert!(
        ok_vfio
            .bind_msix_eventfd(&StepFaultSyscalls::new(1), 0, 1)
            .is_err()
    );
    assert!(ok_vfio.hot_reset(&StepFaultSyscalls::new(1)).is_err());

    let ok_gasket =
        GasketIommuSession::open_and_initialize(&StepFaultSyscalls::new(0), "/dev/accel0")?;
    assert!(
        ok_gasket
            .map_buffer(&StepFaultSyscalls::new(1), 0x1000, 0x2000, 4096)
            .is_err()
    );
    assert!(
        ok_gasket
            .unmap_buffer(&StepFaultSyscalls::new(1), 0x2000, 4096)
            .is_err()
    );
    assert!(
        ok_gasket
            .map_dmabuf(&StepFaultSyscalls::new(1), 5, 0x2000, 3)
            .is_err()
    );
    assert!(
        ok_gasket
            .set_eventfd(&StepFaultSyscalls::new(1), 0, 5)
            .is_err()
    );
    assert!(
        ok_gasket
            .clear_eventfd(&StepFaultSyscalls::new(1), 0)
            .is_err()
    );

    // 4. core_engine.rs remaining branches (`imem_byte_len == 0`, bad `cmem_blob`, `x0.opcode != 0`, scalar-only, `collective_op` without tile/fabric, SparseCoreV3 non-gather & empty slices, empty vreg output fallback)
    let hbm = Arc::new(Mutex::new(SparseHbmStore::new()));
    let fabric = VirtualIciTorusFabric::new();
    let mut cur = 0_u32;
    assert!(
        ArchitecturalBitStreamEngine::execute_from_hbm(
            TpuGeneration::V5e,
            DeviceId(0),
            &hbm,
            None,
            HBM_IMEM_TENSORCORE_BASE,
            0,
            0,
            0,
            &[],
            &[],
            0,
            &mut cur,
        )
        .is_err()
    );

    let mut xlu_bundle = LloBundle {
        generation: TpuGeneration::V6e,
        ..LloBundle::default()
    };
    xlu_bundle.xlu_slots[0].opcode = 1;
    xlu_bundle.xlu_slots[0].dst_vreg = 3;
    xlu_bundle.dma_slot.collective_op = 14;
    let mut scalar_bundle = LloBundle {
        generation: TpuGeneration::V6e,
        stall_cycles: 2,
        ..LloBundle::default()
    };
    scalar_bundle.dma_slot.collective_op = 14;
    let mut two_bundles = encode_bundle_for_generation(TpuGeneration::V6e, &xlu_bundle);
    two_bundles.extend_from_slice(&encode_bundle_for_generation(
        TpuGeneration::V6e,
        &scalar_bundle,
    ));
    hbm.lock()
        .write_bytes(HBM_IMEM_TENSORCORE_BASE, &two_bundles);
    hbm.lock()
        .write_bytes(super::emulator::HBM_CMEM_RODATA_BASE, b"bad_cmem");

    let sc_non_gather = SparseCoreV3Codec::encode(&SparseCoreV3Instruction {
        subcore: SparseCoreSubcoreKind::Tac,
        op: SparseCoreOp::SyncBarrier,
        optimizer_id: 0,
        quant_mode: 0,
        vector_lanes: 8,
        spmem_offset: 0,
        hbm3e_byte_address: 0,
        sync_flag_id: 1,
    });
    let sc_gather = SparseCoreV3Codec::encode(&SparseCoreV3Instruction {
        subcore: SparseCoreSubcoreKind::Tec,
        op: SparseCoreOp::SpmemGatherDma,
        optimizer_id: 0,
        quant_mode: 0,
        vector_lanes: 8,
        spmem_offset: 0,
        hbm3e_byte_address: 0,
        sync_flag_id: 1,
    });
    let mut sc_both = Vec::new();
    sc_both.extend_from_slice(&sc_non_gather);
    sc_both.extend_from_slice(&sc_gather);
    hbm.lock().write_bytes(HBM_IMEM_SPARSECORE_BASE, &sc_both);

    // Run once with `fabric = None` and empty inputs, then with `fabric = Some(&fabric)` and 2-byte (< 4B) inputs
    let _ = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V6e,
        DeviceId(0),
        &hbm,
        None,
        HBM_IMEM_TENSORCORE_BASE,
        two_bundles.len(),
        8,
        sc_both.len(),
        &[(0, 16), (0x1000, 0)],
        &[(0, 16), (0x2000, 0), (0x3000, 16)],
        0,
        &mut cur,
    )?;
    let _ = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V6e,
        DeviceId(0),
        &hbm,
        Some(&fabric),
        HBM_IMEM_TENSORCORE_BASE,
        two_bundles.len(),
        0,
        sc_both.len(),
        &[(0x1000, 2), (0x1100, 2)],
        &[(0x3000, 16)],
        0,
        &mut cur,
    )?;
    let _ = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V6e,
        DeviceId(0),
        &hbm,
        Some(&fabric),
        HBM_IMEM_TENSORCORE_BASE,
        two_bundles.len(),
        0,
        sc_both.len(),
        &[(0x1000, 8), (0x1100, 2)],
        &[(0x3000, 16)],
        0,
        &mut cur,
    )?;

    // 5. device.rs remaining branches (`read_chip_csr_u64`, `mmio_read_u64(2)`, `iommu_by_iova` resolution, zero-length H2D/D2H, `imem_phys == 0` fault, `opcode = 0` & `99`, `flags = 0`, zero host_vaddr ioctls)
    let vdev = VirtualVfioGasketDevice::default();
    let _ = vdev.read_chip_csr_u64(MmioOffset(0))?;
    assert_eq!(vdev.mmio_read_u64(2, MmioOffset(0))?, 0);
    let dev_fd = vdev.open_device("/dev/accel0")?;
    let host_buf = PinnedDmaBuffer::allocate(4096)?;

    // Map with `host_address == 0` and with distinct `device_address = 0x7000_0000` -> `host_buf.host_vaddr()`
    let mut zero_gasket = super::GasketPageTableIoctl {
        page_table_index: 0,
        size: 4096,
        host_address: 0,
        device_address: 0x8000_0000,
    };
    let _ = vdev.ioctl_typed(dev_fd, super::GASKET_IOCTL_MAP_BUFFER, &mut zero_gasket)?;
    let mut iova_gasket = super::GasketPageTableIoctl {
        page_table_index: 0,
        size: 4096,
        host_address: host_buf.host_vaddr(),
        device_address: 0x7000_0000,
    };
    let _ = vdev.ioctl_typed(dev_fd, super::GASKET_IOCTL_MAP_BUFFER, &mut iova_gasket)?;
    let mut unmapped_gasket = super::GasketPageTableIoctl {
        page_table_index: 0,
        size: 4096,
        host_address: 0,
        device_address: 0x9999_0000,
    };
    let _ = vdev.ioctl_typed(
        dev_fd,
        super::GASKET_IOCTL_UNMAP_BUFFER,
        &mut unmapped_gasket,
    )?;

    let mut zero_vfio = super::VfioDmaMapIoctl {
        argsz: 32,
        flags: 3,
        vaddr: 0,
        iova: 0x8100_0000,
        size: 4096,
    };
    let _ = vdev.ioctl_typed(dev_fd, super::VFIO_IOMMU_MAP_DMA_IOCTL, &mut zero_vfio)?;
    let mut unmapped_vfio = super::VfioDmaUnmapIoctl {
        argsz: 24,
        flags: 0,
        iova: 0x9999_0000,
        size: 4096,
    };
    let _ = vdev.ioctl_typed(
        dev_fd,
        super::VFIO_IOMMU_UNMAP_DMA_IOCTL,
        &mut unmapped_vfio,
    )?;
    let mut neg_efd = super::GasketInterruptEventFd {
        interrupt: 3,
        event_fd: u64::MAX,
        reserved: [0; 2],
    };
    let _ = vdev.ioctl_typed(dev_fd, super::GASKET_IOCTL_SET_EVENTFD, &mut neg_efd)?;

    // Helper closure to submit a single ring descriptor at current tail
    let submit_one = |mut pkt: HardwareDmaCommandDescriptor, ring_slot: u64| -> LibTpuResult<()> {
        pkt.crc32c = pkt.compute_checksum();
        for (idx, ch) in pkt.to_bytes().chunks_exact(8).enumerate() {
            let mut arr = [0_u8; 8];
            arr.copy_from_slice(ch);
            vdev.mmio_write_u64(
                2,
                MmioOffset(BAR2_DMA_RING_OFFSET + (ring_slot % 64) * 64 + (idx as u64) * 8),
                u64::from_le_bytes(arr),
            )?;
        }
        vdev.mmio_write_u64(0, MmioOffset(BAR0_CSR_HBM_DMA_RING_TAIL), ring_slot + 1)
    };

    // slot 0: opcode=1 zero-len (`byte_len == 0`) with `flags = 0`
    let mut d0 = HardwareDmaCommandDescriptor::new(1, 9, 1, 0, 0x1000_0000, 0, 0, 0, 1);
    d0.flags = 0;
    submit_one(d0, 0)?;
    // slot 1: opcode=1 via `iommu_by_iova` (`src_iova = 0x7000_0000`) on unbound queue 9
    submit_one(
        HardwareDmaCommandDescriptor::new(1, 9, 2, 0x7000_0000, 0x1000_0000, 64, 0, 0, 2),
        1,
    )?;
    // slot 2: opcode=2 zero-len (`byte_len == 0`)
    submit_one(
        HardwareDmaCommandDescriptor::new(2, 9, 3, 0x1000_0000, 0, 0, 0, 0, 3),
        2,
    )?;
    // slot 3: opcode=0 (NOP) & slot 4: opcode=99 (unknown no-op)
    submit_one(
        HardwareDmaCommandDescriptor::new(0, 9, 4, 0, 0, 0, 0, 0, 4),
        3,
    )?;
    submit_one(
        HardwareDmaCommandDescriptor::new(99, 9, 5, 0, 0, 0, 0, 0, 5),
        4,
    )?;
    // Negative interrupt_fd (`-5`) in `signal_queue_os_eventfd` doesn't insert, already covered;
    // Now test failing descriptors (`off + len > cap`, `host_vaddr == 0`, D2H unmapped, ExecuteImem `imem_phys == 0` invalid len)
    assert!(
        submit_one(
            HardwareDmaCommandDescriptor::new(
                1,
                0,
                6,
                host_buf.host_vaddr(),
                0x1000_0000,
                8192,
                0,
                0,
                6,
            ),
            5,
        )
        .is_err()
    );
    assert!(
        submit_one(
            HardwareDmaCommandDescriptor::new(1, 0, 7, 0x8000_0000, 0x1000_0000, 64, 0, 0, 7),
            5,
        )
        .is_err()
    );
    assert!(
        submit_one(
            HardwareDmaCommandDescriptor::new(2, 0, 8, 0x1000_0000, 0xDEAD_0000, 64, 0, 0, 8),
            5,
        )
        .is_err()
    );
    assert!(
        submit_one(
            HardwareDmaCommandDescriptor::new(3, 0, 9, 0, 0, 7, 0, 1, 9),
            5,
        )
        .is_err()
    );
    //Also resolve_host_dma_ptr len == 0 directly via a 0-byte map or check
    assert!(
        submit_one(
            HardwareDmaCommandDescriptor::new(1, 0, 10, 0x7000_0000, 0x1000_0000, 8192, 0, 0, 10),
            5,
        )
        .is_err()
    );

    // 6. linux.rs remaining branches
    let vis_cfg = TpuInitConfig {
        chips_per_host_bounds: ChipCoordinates::new(2, 1, 1),
        visible_devices: Some(vec![DeviceId(0)]),
        ..TpuInitConfig::default()
    };
    let non_emu =
        LinuxDeviceHal::with_raw_fd(KernelDriverMode::Gasket, ZeroDevFdSyscalls, 42, &vis_cfg);
    assert_eq!(non_emu.topology.devices.len(), 1);
    assert_eq!(non_emu.read_chip_csr_u64(MmioOffset(0))?, 0);
    let empty_xs = non_emu.harvest_hardware_xplane_traces(DeviceId(0))?;
    assert_eq!(empty_xs.planes.len(), 1);

    let emu_hal = LinuxDeviceHal::open(
        KernelDriverMode::Gasket,
        VirtualVfioGasketDevice::default(),
        &TpuInitConfig::default(),
    )?;
    // Harvest before any execution (`cursor == 0`) & with ICI RDMA (`unit_tag = 4`)
    let _ = emu_hal.harvest_hardware_xplane_traces(DeviceId(0))?;
    emu_hal.unregister_interrupt_eventfd()?;
    emu_hal.initialize_gasket_device(12)?;
    emu_hal.upload_bytes_to_device_hbm(DeviceId(0), 0x0100_0000, &[])?;

    let add_mlir = br"
module @hw_add2 {
  func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
";
    let prog = CompiledProgram::compile("mlir", add_mlir, &[])?;
    let a = emu_hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16))?;
    let b = emu_hal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16))?;
    let _ = emu_hal.execute_hardware_direct_mode(DeviceId(0), &prog, &[a, b], &[])?;
    assert!(
        emu_hal
            .execute_hardware_direct_mode(DeviceId(99), &prog, &[a, b], &[])
            .is_err()
    );

    // Clear `scratch_staging` to hit `None` / `dst_vaddr == 0` branches in `upload_bytes_to_device_hbm` & `harvest_hardware_xplane_traces`
    *emu_hal.scratch_staging.lock() = None;
    emu_hal.upload_bytes_to_device_hbm(DeviceId(0), 0x0100_0000, &[1, 2, 3, 4])?;
    let _ = emu_hal.harvest_hardware_xplane_traces(DeviceId(0))?;
    emu_hal.initialize_gasket_device(12)?;

    // DeviceNotFound & D2D out-of-bounds error branches on `LinuxDeviceHal`
    assert!(
        emu_hal
            .allocate_device_memory(DeviceId(99), MemoryKind::DeviceHbm, ByteSize(16))
            .is_err()
    );
    assert!(emu_hal.deallocate_device_memory(DeviceId(99), a).is_err());
    assert!(emu_hal.write_device_memory(DeviceId(99), a, &[1]).is_err());
    let mut rbuf = [0_u8; 4];
    assert!(
        emu_hal
            .read_device_memory(DeviceId(99), a, &mut rbuf)
            .is_err()
    );
    assert!(
        emu_hal
            .copy_device_to_device(DeviceId(99), a, DeviceId(0), b, ByteSize(4))
            .is_err()
    );
    assert!(
        emu_hal
            .copy_device_to_device(DeviceId(0), a, DeviceId(99), b, ByteSize(4))
            .is_err()
    );
    assert!(
        emu_hal
            .copy_device_to_device(
                DeviceId(0),
                a,
                DeviceId(0),
                DeviceAddress(0xDEAD_0000),
                ByteSize(4),
            )
            .is_err()
    );
    assert!(
        emu_hal
            .execute_program(DeviceId(99), &prog, &[a, b], &[b])
            .is_err()
    );

    // Step-faulting `StepFaultSyscalls` across `LinuxDeviceHal::open` (Gasket & VFIO) & `execute_program` & `harvest_hardware_xplane_traces`
    for step in 1..=25 {
        if let Ok(fhal) = LinuxDeviceHal::open(
            KernelDriverMode::Gasket,
            StepFaultSyscalls::new(step),
            &TpuInitConfig::default(),
        ) {
            if let (Ok(fa), Ok(fb), Ok(fo)) = (
                fhal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16)),
                fhal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16)),
                fhal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16)),
            ) {
                let _ = fhal.write_device_memory(DeviceId(0), fa, &[0_u8; 16]);
                let _ = fhal.execute_program(DeviceId(0), &prog, &[fa, fb], &[fo]);
                let _ = fhal.harvest_hardware_xplane_traces(DeviceId(0));
            }
        }
        let _ = LinuxDeviceHal::open(
            KernelDriverMode::Vfio,
            StepFaultSyscalls::new(step),
            &TpuInitConfig::default(),
        );
    }
    for step in 26..=65 {
        if let Ok(fhal) = LinuxDeviceHal::open(
            KernelDriverMode::Gasket,
            StepFaultSyscalls::new(step),
            &TpuInitConfig::default(),
        ) && let (Ok(fa), Ok(fb), Ok(fo)) = (
            fhal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16)),
            fhal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16)),
            fhal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(16)),
        ) {
            let _ = fhal.execute_program(DeviceId(0), &prog, &[fa, fb], &[fo]);
            let _ = fhal.harvest_hardware_xplane_traces(DeviceId(0));
        }
    }

    // 7. Remaining branch arms across `core_engine.rs`, `device.rs`, `pcie.rs`, and `linux.rs`
    let nop_coll_hbm = Arc::new(Mutex::new(SparseHbmStore::new()));
    let nop_fabric = VirtualIciTorusFabric::new();
    let mut nop_coll = LloBundle {
        generation: TpuGeneration::V6e,
        ..LloBundle::default()
    };
    nop_coll.dma_slot.dma_opcode = 14;
    nop_coll.dma_slot.collective_op = 14;
    nop_coll.dma_slot.sync_flag_id = 3;
    let nop_v6e_bytes = encode_bundle_for_generation(TpuGeneration::V6e, &nop_coll);
    nop_coll_hbm
        .lock()
        .write_bytes(HBM_IMEM_TENSORCORE_BASE, &nop_v6e_bytes);
    let sc_v3_gather = SparseCoreV3Codec::encode(&SparseCoreV3Instruction {
        subcore: SparseCoreSubcoreKind::Tec,
        op: SparseCoreOp::SpmemGatherDma,
        optimizer_id: 0,
        quant_mode: 0,
        vector_lanes: 8,
        spmem_offset: 0,
        hbm3e_byte_address: 0x1000_0000,
        sync_flag_id: 4,
    });
    nop_coll_hbm
        .lock()
        .write_bytes(HBM_IMEM_SPARSECORE_BASE, &sc_v3_gather);
    let mut c7 = 0_u32;
    // `fabric = None` on `collective_op >= 14`
    let _ = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V6e,
        DeviceId(0),
        &nop_coll_hbm,
        None,
        HBM_IMEM_TENSORCORE_BASE,
        nop_v6e_bytes.len(),
        0,
        sc_v3_gather.len(),
        &[],
        &[(0x1000_4000, 16)],
        0,
        &mut c7,
    )?;
    // `fabric = Some(&nop_fabric)` with empty `inputs` (`vreg_tiles` empty -> `None` & `unwrap_or_else`)
    let _ = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V6e,
        DeviceId(0),
        &nop_coll_hbm,
        Some(&nop_fabric),
        HBM_IMEM_TENSORCORE_BASE,
        nop_v6e_bytes.len(),
        0,
        sc_v3_gather.len(),
        &[],
        &[(0x1000_4000, 16)],
        0,
        &mut c7,
    )?;
    // `V6e` with 16-byte `SparseCoreV2Codec` stream (`sc_imem_byte_len % 24 != 0`)
    let sc_v2_raw = SparseCoreV2Codec::encode(&SparseCoreV2Instruction {
        subcore: SparseCoreSubcoreKind::Tec,
        op: SparseCoreOp::SpmemGatherDma,
        optimizer_id: 0,
        spmem_offset: 0,
        hbm_page_offset: 0,
        transfer_bytes: 64,
        sync_flag_id: 5,
    });
    nop_coll_hbm
        .lock()
        .write_bytes(HBM_IMEM_SPARSECORE_BASE, &sc_v2_raw);
    let _ = ArchitecturalBitStreamEngine::execute_from_hbm(
        TpuGeneration::V6e,
        DeviceId(0),
        &nop_coll_hbm,
        None,
        HBM_IMEM_TENSORCORE_BASE,
        nop_v6e_bytes.len(),
        0,
        sc_v2_raw.len(),
        &[],
        &[],
        0,
        &mut c7,
    )?;

    // `device.rs`: negative `event_fd` (`-1`) on MSI-X queue 2 + doorbell on queue 2
    let neg_efd_dev = VirtualVfioGasketDevice::default();
    let vfio_unbind_fd = neg_efd_dev.open_device("/dev/vfio/vfio")?;
    let mut unbind_irq = VfioIrqSetIoctl {
        argsz: 24,
        flags: 0x24,
        index: 2,
        start: 2,
        count: 1,
        event_fd: -1,
    };
    let _ = neg_efd_dev.ioctl_typed(vfio_unbind_fd, VFIO_DEVICE_SET_IRQS_IOCTL, &mut unbind_irq)?;
    let neg_hal = LinuxDeviceHal::with_raw_fd(
        KernelDriverMode::Vfio,
        neg_efd_dev,
        vfio_unbind_fd,
        &TpuInitConfig::default(),
    );
    // `os_eventfd_pair` is `None` (`queue_id = 0`, `flags = 1`), `flags = 0`, and `queue_id = 1`
    let _ = neg_hal.dispatch_hardware_ring_packet(&HardwareDmaCommandDescriptor::new(
        0, 0, 1, 0, 0, 0, 0, 0, 101,
    ))?;
    let mut no_irq_pkt = HardwareDmaCommandDescriptor::new(0, 0, 1, 0, 0, 0, 0, 0, 102);
    no_irq_pkt.flags = 0;
    no_irq_pkt.crc32c = no_irq_pkt.compute_checksum();
    let _ = emu_hal.dispatch_hardware_ring_packet(&no_irq_pkt)?;
    let _ = neg_hal.dispatch_hardware_ring_packet(&HardwareDmaCommandDescriptor::new(
        0, 1, 2, 0, 0, 0, 0, 0, 103,
    ))?;
    emu_hal.write_bar0_csr(BAR0_CSR_HBM_DMA_RING_TAIL, 0);

    // `pcie.rs`: `iommu_group_id` fallback file and `vfio_candidate.exists()` with/without `accel`
    let tmp_pcie = std::env::temp_dir().join(format!("libtpu_pcie_cov_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_pcie);
    let sys_pci = tmp_pcie.join("pci");
    let dev_dir = tmp_pcie.join("dev");
    let bdf5 = sys_pci.join("0000:00:05.0");
    std::fs::create_dir_all(&bdf5)?;
    std::fs::create_dir_all(dev_dir.join("vfio"))?;
    std::fs::write(bdf5.join("vendor"), "0x1ae0\n")?;
    std::fs::write(bdf5.join("device"), "0x006f\n")?;
    std::fs::write(bdf5.join("iommu_group_id"), "7\n")?;
    std::fs::write(dev_dir.join("vfio").join("7"), "")?;
    let scanned_vfio = SysfsPcieEnumerator::enumerate_devices(&sys_pci, &dev_dir)?;
    assert_eq!(
        scanned_vfio.first().map(|f| f.driver_mode),
        Some(KernelDriverMode::Vfio)
    );
    std::fs::write(dev_dir.join("accel0"), "")?;
    let scanned_gasket = SysfsPcieEnumerator::enumerate_devices(&sys_pci, &dev_dir)?;
    assert_eq!(
        scanned_gasket.first().map(|f| f.driver_mode),
        Some(KernelDriverMode::Gasket)
    );
    let _ = std::fs::remove_dir_all(&tmp_pcie);

    Ok(())
}
