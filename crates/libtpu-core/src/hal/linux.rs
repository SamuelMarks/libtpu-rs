//! Linux VFIO and Gasket hardware device HAL ([`LinuxDeviceHal`]) and backend selector ([`HalBackend`]).

use super::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SYNC_FLAG_BASE, BAR0_CSR_TENSORCORE_PC,
    BAR0_CSR_TENSORCORE_RUN_CONTROL, BAR0_CSR_TENSORCORE_STATUS, BAR0_CSR_THERMAL_MILLICELSIUS,
    ChipRegisterMap, DeepSeaVm, DeviceMemoryStats, EmulatedKernelDriverBackend,
    GASKET_IOCTL_CLEAR_EVENTFD, GASKET_IOCTL_MAP_BUFFER, GASKET_IOCTL_NUMBER_PAGE_TABLES,
    GASKET_IOCTL_PARTITION_PAGE_TABLE, GASKET_IOCTL_SET_EVENTFD, GASKET_IOCTL_UNMAP_BUFFER,
    GasketInterruptEventFd, GasketPageTableIoctl, HbmDmaRingDescriptor, IoctlPayload,
    KernelDriverMode, SimulatedTpuHal, SyscallProvider, TopologySpec, TpuHal,
    VFIO_DEVICE_SET_IRQS_IOCTL, VFIO_IOMMU_MAP_DMA_IOCTL, VFIO_IOMMU_UNMAP_DMA_IOCTL,
    VfioDmaMapIoctl, VfioDmaUnmapIoctl, VfioIrqSetIoctl,
};
use crate::compiler::CompiledProgram;
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ByteSize, DeviceAddress, DeviceId, MemoryKind};

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
        match self.mode {
            KernelDriverMode::Gasket => {
                let mut payload = GasketPageTableIoctl {
                    page_table_index: 0,
                    size: 4096,
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
                    size: 4096,
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
        let imem = program.encode_architectural_imem();
        let mut vm = DeepSeaVm::new();
        vm.execute_imem_bytes(&imem);
        let rmap = ChipRegisterMap::for_generation(self.inner.topology.generation);
        self.mmio
            .set_csr(rmap.tc0_program_location_csr, imem.len() as u64);
        self.mmio.set_csr(rmap.tc0_pc_csr, u64::from(vm.pc));
        self.mmio.set_csr(rmap.tc0_run_status_csr, 1);
        self.mmio
            .set_csr(rmap.tc0_vwait_sync_flag_state_csr, vm.cycles_executed);
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_PC, u64::from(vm.pc));
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_RUN_CONTROL, 1);
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_STATUS, 1);
        let sync = self.mmio.get_csr(BAR0_CSR_SYNC_FLAG_BASE);
        self.mmio.set_csr(
            BAR0_CSR_SYNC_FLAG_BASE,
            sync.wrapping_add(vm.cycles_executed),
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
