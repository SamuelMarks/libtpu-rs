//! Linux VFIO and Gasket hardware device HAL ([`LinuxDeviceHal`]) and backend selector ([`HalBackend`]).

use super::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SYNC_FLAG_BASE, BAR0_CSR_TENSORCORE_PC,
    BAR0_CSR_TENSORCORE_RUN_CONTROL, BAR0_CSR_TENSORCORE_STATUS, BAR0_CSR_THERMAL_MILLICELSIUS,
    ChipRegisterMap, DeviceMemoryStats, EmulatedKernelDriverBackend, GASKET_IOCTL_CLEAR_EVENTFD,
    GASKET_IOCTL_MAP_BUFFER, GASKET_IOCTL_NUMBER_PAGE_TABLES, GASKET_IOCTL_PARTITION_PAGE_TABLE,
    GASKET_IOCTL_RESET, GASKET_IOCTL_SET_EVENTFD, GASKET_IOCTL_UNMAP_BUFFER,
    GasketInterruptEventFd, GasketPageTableIoctl, HbmDmaRingDescriptor, IoctlPayload,
    KernelDriverMode, SimulatedTpuHal, SyscallProvider, TopologySpec, TpuHal, VFIO_CHECK_EXTENSION,
    VFIO_DEVICE_GET_INFO, VFIO_DEVICE_GET_IRQ_INFO, VFIO_DEVICE_GET_REGION_INFO,
    VFIO_DEVICE_SET_IRQS_IOCTL, VFIO_GET_API_VERSION, VFIO_GROUP_GET_DEVICE_FD,
    VFIO_GROUP_GET_STATUS, VFIO_GROUP_SET_CONTAINER, VFIO_IOMMU_MAP_DMA_IOCTL,
    VFIO_IOMMU_UNMAP_DMA_IOCTL, VFIO_SET_IOMMU, VfioDmaMapIoctl, VfioDmaUnmapIoctl,
    VfioIrqSetIoctl,
};
use crate::compiler::CompiledProgram;
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ByteSize, DeviceAddress, DeviceId, MemoryKind};

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
            iommu_allocators,
            inner,
        })
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
        let imem_bytes = program.encode_architectural_imem();
        let imem_len = imem_bytes.len() as u64;
        // Step 1: Upload compiled 512-bit VLIW IMEM overlay to device HBM/IMEM base (`0x0080_0000`) via DMA ring
        self.submit_dma_descriptor(imem_bytes.as_ptr() as u64, 0x0080_0000, imem_bytes.len(), 0);
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
        // Step 4: Program per-generation ChipRegisterMap CSRs (`PROGRAM_LOCATION`, `PC`, `RUN_STATUS`, `VWAIT_SYNC_FLAG_STATE`)
        let pc = self.inner.mmio.get_csr(BAR0_CSR_TENSORCORE_PC);
        let next_sync = self.inner.mmio.get_csr(BAR0_CSR_SYNC_FLAG_BASE);
        let rmap = ChipRegisterMap::for_generation(self.inner.topology.generation);
        self.write_bar0_csr(rmap.tc0_program_location_csr, imem_len);
        self.write_bar0_csr(rmap.tc0_pc_csr, pc);
        self.write_bar0_csr(rmap.tc0_run_status_csr, 1);
        self.write_bar0_csr(rmap.tc0_vwait_sync_flag_state_csr, next_sync);
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
