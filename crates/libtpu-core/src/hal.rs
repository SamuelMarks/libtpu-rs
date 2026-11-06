//! Hardware Abstraction Layer (`TpuHal`) for simulated and Linux VFIO/Gasket TPU devices.
//!
//! Implements both an in-memory deterministic [`SimulatedTpuHal`] and a kernel-ioctl + `PCIe` BAR MMIO
//! backed [`LinuxDeviceHal`] supporting the reverse-engineered Gasket (`0xdc`) and VFIO (`0x3b`)
//! device protocols.

/// Tensor operation evaluation routines (`MXU`, `VALU`, `EUP`, and ICI collectives).
pub mod eval;
/// Linux VFIO and Gasket hardware device HAL ([`LinuxDeviceHal`]) and backend selector.
pub mod linux;
/// HBM/VMEM region allocation, BAR MMIO apertures, and device memory statistics.
pub mod memory;
/// Linux `sysfs` `PCIe` hardware discovery and TPU topology descriptors.
pub mod pci;
/// Gasket/VFIO `ioctl` constants, `DeepSea` BAR0 register maps, and CSR bitflags.
pub mod registers;
/// Deterministic in-memory TPU hardware simulator ([`SimulatedTpuHal`]).
pub mod simulated;
/// Kernel driver `ioctl` and BAR0 MMIO traits and backend implementations.
pub mod syscall;
/// Architectural `TensorCore` VLIW execution state machine ([`TpuCoreVmState`]).
pub mod vm;

pub use self::eval::{
    CustomCallHandlerFn, MAX_WHILE_ITERATIONS, invoke_custom_call_handler,
    register_custom_call_handler, unregister_custom_call_handler,
};
pub use self::linux::{
    HalBackend, HardwareDoorbellRing, LinuxDeviceHal, create_default_hal,
    discover_pcie_tpu_devices_from_sysfs,
};
pub use self::memory::{
    DeviceMemoryStats, HbmRegionAllocator, MappedBarAperture, MmioAperture, VmemRegionAllocator,
};
pub use self::pci::{
    DeviceSpec, DiscoveredPciTpuDevice, KernelDriverMode, PciDeviceEnumerator, PciSysfsScanner,
    TopologySpec, scan_sysfs_pci_tpu_devices,
};
pub use self::registers::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_ICI_ROUTER_LINK_STATE, BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SPARSECORE_RUN_CONTROL,
    BAR0_CSR_SYNC_FLAG_BASE, BAR0_CSR_TENSORCORE_PC, BAR0_CSR_TENSORCORE_RUN_CONTROL,
    BAR0_CSR_TENSORCORE_STATUS, BAR0_CSR_TENSORCORE_SYNC_FLAG, BAR0_CSR_THERMAL_MILLICELSIUS,
    ChipRegisterMap, CsrAccessPermission, CsrBitfieldSpec, CsrRegisterDescriptor,
    DmaRingDescriptor, GASKET_IOCTL_CLEAR_EVENTFD, GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS,
    GASKET_IOCTL_MAP_BUFFER, GASKET_IOCTL_MAP_DMABUF, GASKET_IOCTL_NUMBER_PAGE_TABLES,
    GASKET_IOCTL_PAGE_TABLE_SIZE, GASKET_IOCTL_PARTITION_PAGE_TABLE, GASKET_IOCTL_RESET,
    GASKET_IOCTL_SET_EVENTFD, GASKET_IOCTL_SIMPLE_PAGE_TABLE_SIZE, GASKET_IOCTL_UNMAP_BUFFER,
    GasketInterruptEventFd, GasketPageTableDmaBuf, GasketPageTableIoctl, HbmDmaRingDescriptor,
    IciLinkStateFlags, TensorCoreControlFlags, TensorCoreStatusFlags, VFIO_CHECK_EXTENSION,
    VFIO_DEVICE_GET_INFO, VFIO_DEVICE_GET_IRQ_INFO, VFIO_DEVICE_GET_REGION_INFO, VFIO_DEVICE_RESET,
    VFIO_DEVICE_SET_IRQS, VFIO_DEVICE_SET_IRQS_IOCTL, VFIO_GET_API_VERSION,
    VFIO_GROUP_GET_DEVICE_FD, VFIO_GROUP_GET_STATUS, VFIO_GROUP_SET_CONTAINER, VFIO_IOMMU_MAP_DMA,
    VFIO_IOMMU_MAP_DMA_IOCTL, VFIO_IOMMU_UNMAP_DMA, VFIO_IOMMU_UNMAP_DMA_IOCTL, VFIO_SET_IOMMU,
    VfioDmaMapFlags, VfioDmaMapIoctl, VfioDmaUnmapIoctl, VfioIrqSetFlags, VfioIrqSetIoctl,
};
pub use self::simulated::SimulatedTpuHal;
pub use self::syscall::{
    DiscoveredPciDevice, EmulatedKernelDriverBackend, IoctlPayload, KernelDriverOps,
    MockKernelSyscalls, POSIX_FIOCLEX_IOCTL, PosixKernelDriverBackend, RealLinuxSyscalls,
    SimulatedKernelDriverBackend, SyscallProvider, VirtualTpuKernelDevice,
    discover_sysfs_tpu_pci_devices,
};
pub use self::vm::{DeepSeaVm, SparseCoreOptimizer, SparseCoreVmState, TpuCoreVmState};

use crate::compiler::CompiledProgram;
use crate::error::LibTpuResult;
use crate::types::{ByteSize, DeviceAddress, DeviceId, MemoryKind};

/// Hardware Abstraction Layer trait for TPU device discovery, HBM allocation, DMA transfers,
/// and program execution.
pub trait TpuHal: Send + Sync + std::fmt::Debug {
    /// Discovers the TPU topology for this backend.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError`] if hardware discovery fails.
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
    /// Returns [`crate::error::LibTpuError::DeviceNotFound`] or [`crate::error::LibTpuError::OutOfMemory`] on failure.
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
    /// Returns [`crate::error::LibTpuError`] if the device or allocation does not exist.
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
    /// Returns [`crate::error::LibTpuError::TransferFailed`] if the address or bounds are invalid.
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
    /// Returns [`crate::error::LibTpuError::TransferFailed`] if the address or bounds are invalid.
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
    /// Returns [`crate::error::LibTpuError::TransferFailed`] if either buffer is invalid or too small.
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
    /// Returns [`crate::error::LibTpuError::ExecutionFailed`] if the device is poisoned or buffers are invalid.
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
    /// Returns [`crate::error::LibTpuError::DeviceNotFound`] if `device_id` is invalid.
    fn memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats>;

    /// Clears peak HBM memory statistics for `device_id` (`PJRT_Device_ClearMemoryStats`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::DeviceNotFound`] if `device_id` is invalid.
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
    /// Returns [`crate::error::LibTpuError::DeviceNotFound`] if `device_id` is invalid.
    fn poison_device(&self, device_id: DeviceId, reason: String) -> LibTpuResult<()>;
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::compiler::TpuOpCode;
    use crate::config::TpuInitConfig;
    use crate::error::LibTpuError;
    use crate::types::{
        ChipCoordinates, ElementType, EventFdHandle, FileDescriptor, MemoryLayout, MmioOffset,
        PciDeviceId, Shape, TpuGeneration,
    };
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

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
        if let Ok(t) = topo {
            assert_eq!(t.devices.len(), 2);
            if let Some(d0) = t.devices.first() {
                assert!(d0.is_addressable);
                assert_ne!(format!("{d0:?}"), "");
            }
            if let Some(d1) = t.devices.get(1) {
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
        if let Ok(s) = stats {
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
        hal.devices_poisoned.store(true, Ordering::Release);
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
        if let Ok(prog) = CompiledProgram::compile("mlir", b"return", b"") {
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
        let nd_dot_attrs = crate::compiler::InstructionAttrs {
            dot_lhs_batch_dims: vec![0],
            dot_rhs_batch_dims: vec![0],
            dot_lhs_contracting_dims: vec![2],
            dot_rhs_contracting_dims: vec![1],
            input_shapes: vec![
                Shape {
                    element_type: ElementType::F32,
                    dims: vec![1, 2, 2],
                    dynamic_dims: vec![false; 3],
                    layout: MemoryLayout::default_tpu_tiled(3),
                },
                Shape {
                    element_type: ElementType::F32,
                    dims: vec![1, 2, 2],
                    dynamic_dims: vec![false; 3],
                    layout: MemoryLayout::default_tpu_tiled(3),
                },
            ],
            output_shape: None,
            ..crate::compiler::InstructionAttrs::default()
        };
        assert_eq!(
            SimulatedTpuHal::eval_dot_general(
                ElementType::F32,
                &[1.0, 2.0, 3.0, 4.0],
                &[1.0, 0.0, 0.0, 1.0],
                &nd_dot_attrs,
            )
            .len(),
            16
        );
        assert_eq!(
            SimulatedTpuHal::eval_pad_with_attrs(
                ElementType::F32,
                &[1.0, 2.0],
                0.5,
                &crate::compiler::InstructionAttrs::default(),
            )
            .len(),
            8
        );
        let pad_bundle = crate::compiler::TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Pad,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: crate::compiler::InstructionAttrs {
                pad_edge_low: vec![1],
                pad_edge_high: vec![2],
                pad_interior: vec![1],
                input_shapes: vec![Shape {
                    element_type: ElementType::F32,
                    dims: vec![2],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                }],
                output_shape: None,
                ..crate::compiler::InstructionAttrs::default()
            },
        };
        let pad_in = ElementType::F32.pack_from_f64_slice(&[10.0, 20.0]);
        let pad_v = ElementType::F32.pack_from_f64_slice(&[9.0]);
        let pad_res = ElementType::F32.unpack_to_f64_vec(&SimulatedTpuHal::eval_bundle_with_attrs(
            &pad_bundle,
            ElementType::F32,
            &[&pad_in, &pad_v],
        ));
        assert_eq!(pad_res, vec![9.0, 10.0, 9.0, 20.0, 9.0, 9.0]);

        for r_op in [
            TpuOpCode::ReduceSumF32,
            TpuOpCode::ReduceMax,
            TpuOpCode::ReduceMin,
        ] {
            let red_bundle = crate::compiler::TpuInstructionBundle {
                pc_offset: 0,
                opcode: r_op,
                operand_slots: vec![0],
                output_slot: 1,
                sync_flag_id: 1,
                attrs: crate::compiler::InstructionAttrs {
                    reduce_dims: vec![1],
                    input_shapes: vec![Shape {
                        element_type: ElementType::F32,
                        dims: vec![2, 2],
                        dynamic_dims: vec![false; 2],
                        layout: MemoryLayout::default_tpu_tiled(2),
                    }],
                    output_shape: Some(Shape {
                        element_type: ElementType::F32,
                        dims: vec![2],
                        dynamic_dims: vec![false],
                        layout: MemoryLayout::default_tpu_tiled(1),
                    }),
                    ..crate::compiler::InstructionAttrs::default()
                },
            };
            let red_in = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(
                SimulatedTpuHal::eval_bundle_with_attrs(&red_bundle, ElementType::F32, &[&red_in])
                    .len(),
                8
            );
        }

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
            b"module @erf { erf }",
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
            b"module @dag { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
 %0 = stablehlo.multiply %0, %arg1 : tensor<4xf32> } }",
            b"module @rect_dot { func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
 return %0 : tensor<2x2xf32> } }",
            b"module @batch_dot { func.func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
 return %0 : tensor<2x2x2xf32> } }",
            b"module @vec_dot { func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
 return %0 : tensor<3xf32> } }",
            b"module @sq_dot { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
 return %0 : tensor<4xf32> } }",
            b"HloModule hlo_eq=1
ENTRY main {
 p0 = f32[4] parameter(0)
 ROOT r = f32[4] negate(p0)
}",
            b"module @trp_nd { func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.transpose %arg0, permutation = [1, 0] : tensor<2x2xf32>
 return %0 : tensor<2x2xf32> } }",
            b"module @trp_def { func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.transpose %arg0 : tensor<2x2xf32>
 return %0 : tensor<2x2xf32> } }",
            b"module @brd_nd { func.func @main(%arg0: tensor<2xf32>) -> tensor<2x2xf32> { %0 = stablehlo.broadcast_in_dim %arg0, broadcast_dimensions = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
 return %0 : tensor<2x2xf32> } }",
            b"module @slc_nd { func.func @main(%arg0: tensor<4xf32>) -> tensor<2xf32> { %0 = stablehlo.slice %arg0, start_indices = [1], limit_indices = [3], strides = [1] : (tensor<4xf32>) -> tensor<2xf32>
 return %0 : tensor<2xf32> } }",
            b"module @slc_strided { func.func @main(%arg0: tensor<4xf32>) -> tensor<2xf32> { %0 = stablehlo.slice %arg0, start_indices = [0], limit_indices = [4], strides = [2] : (tensor<4xf32>) -> tensor<2xf32>
 return %0 : tensor<2xf32> } }",
            b"module @cst_dense { func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.constant dense<3.0> : tensor<4xf32>
 return %0 : tensor<4xf32> } }",
            b"module @cat_ax1 { func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x4xf32> { %0 = stablehlo.concatenate %arg0, %arg1, dimension = 1 : tensor<2x4xf32>
 return %0 : tensor<2x4xf32> } }",
            b"HloModule hlo_params
ENTRY main {
 p0 = f32[4] parameter(0)
 ROOT r = f32[4] negate(p0)
}",
        ] {
            if let Ok(prog) = CompiledProgram::compile("mlir", src, b"") {
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
        for u_op in [
            TpuOpCode::PopulationCount,
            TpuOpCode::CountLeadingZeros,
            TpuOpCode::Sort,
            TpuOpCode::RngBitGenerator,
            TpuOpCode::ReduceWindow,
            TpuOpCode::Expm1,
            TpuOpCode::Log1p,
            TpuOpCode::Logistic,
            TpuOpCode::Tan,
            TpuOpCode::Cbrt,
            TpuOpCode::IsFinite,
            TpuOpCode::RoundNearestEven,
            TpuOpCode::RoundNearestAfz,
            TpuOpCode::UniformQuantize,
            TpuOpCode::UniformDequantize,
            TpuOpCode::TopK,
            TpuOpCode::Cholesky,
            TpuOpCode::Fft,
            TpuOpCode::CollectivePermute,
            TpuOpCode::AllToAll,
        ] {
            assert_eq!(
                SimulatedTpuHal::eval_unary_op(
                    u_op,
                    ElementType::S32,
                    &[1_u8, 0, 0, 0, 2, 0, 0, 0]
                )
                .len(),
                8
            );
        }
        // Test Sign (-1, 0, +1) and 2x2 Cholesky
        let sign_in = ElementType::F32.pack_from_f64_slice(&[-2.0, 0.0, 3.0]);
        let sign_out = ElementType::F32.unpack_to_f64_vec(&SimulatedTpuHal::eval_unary_op(
            TpuOpCode::Sign,
            ElementType::F32,
            &sign_in,
        ));
        assert_eq!(sign_out, vec![-1.0, 0.0, 1.0]);
        let chol_in = ElementType::F32.pack_from_f64_slice(&[4.0, 2.0, 2.0, 5.0]);
        let chol_out = ElementType::F32.unpack_to_f64_vec(&SimulatedTpuHal::eval_unary_op(
            TpuOpCode::Cholesky,
            ElementType::F32,
            &chol_in,
        ));
        assert_eq!(chol_out, vec![2.0, 0.0, 1.0, 2.0]);

        // Test C64/C128 complex multiply, divide, and DotGeneral
        // (1 + 2i) * (3 + 4i) = (3 - 8) + (4 + 6)i = -5 + 10i
        let c64_a = ElementType::C64.pack_from_f64_slice(&[1.0, 2.0, 2.0, 1.0]);
        let c64_b = ElementType::C64.pack_from_f64_slice(&[3.0, 4.0, 0.0, 0.0]);
        let c64_mul_bytes =
            SimulatedTpuHal::eval_binary_op(TpuOpCode::MulF32, ElementType::C64, &c64_a, &c64_b);
        let c64_mul = ElementType::C64.unpack_to_f64_vec(&c64_mul_bytes);
        assert_eq!(c64_mul, vec![-5.0, 10.0, 0.0, 0.0]);
        let c64_div = ElementType::C64.unpack_to_f64_vec(&SimulatedTpuHal::eval_binary_op(
            TpuOpCode::DivF32,
            ElementType::C64,
            &c64_mul_bytes,
            &c64_b,
        ));
        assert_eq!(c64_div, vec![1.0, 2.0, 0.0, 0.0]);
        let c128_dot = SimulatedTpuHal::eval_dot_general(
            ElementType::C128,
            &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0],
            &crate::compiler::InstructionAttrs::default(),
        );
        let c128_dot_unpacked = ElementType::C128.unpack_to_f64_vec(&c128_dot);
        assert_eq!(
            c128_dot_unpacked,
            vec![-5.0, 10.0, 0.0, 0.0, 0.0, 0.0, -5.0, 10.0]
        );
        let c64_dot_1d = SimulatedTpuHal::eval_dot_general(
            ElementType::C64,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            &crate::compiler::InstructionAttrs::default(),
        );
        assert_eq!(
            ElementType::C64.unpack_to_f64_vec(&c64_dot_1d),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // Test N-D Concatenate along axis 1 with explicit input_shapes and output_shape
        let cat_nd_bundle = crate::compiler::TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Concatenate,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: crate::compiler::InstructionAttrs {
                concat_dim: 1,
                input_shapes: vec![
                    Shape {
                        element_type: ElementType::F32,
                        dims: vec![2, 1],
                        dynamic_dims: vec![false, false],
                        layout: MemoryLayout::default_tpu_tiled(2),
                    },
                    Shape {
                        element_type: ElementType::F32,
                        dims: vec![2, 1],
                        dynamic_dims: vec![false, false],
                        layout: MemoryLayout::default_tpu_tiled(2),
                    },
                ],
                output_shape: Some(Shape {
                    element_type: ElementType::F32,
                    dims: vec![2, 2],
                    dynamic_dims: vec![false, false],
                    layout: MemoryLayout::default_tpu_tiled(2),
                }),
                ..crate::compiler::InstructionAttrs::default()
            },
        };
        let cat_in0 = ElementType::F32.pack_from_f64_slice(&[1.0, 3.0]);
        let cat_in1 = ElementType::F32.pack_from_f64_slice(&[2.0, 4.0]);
        let cat_nd_out =
            ElementType::F32.unpack_to_f64_vec(&SimulatedTpuHal::eval_bundle_with_attrs(
                &cat_nd_bundle,
                ElementType::F32,
                &[&cat_in0, &cat_in1],
            ));
        assert_eq!(cat_nd_out, vec![1.0, 2.0, 3.0, 4.0]);

        // Test Phase 4 numerical algorithms: Complex FFT/IFFT, Threefry2x32, Philox4x32-10, BatchNorm, SelectAndScatter, Complex/Real/Imag, While/Conditional/Call
        let tf_out = SimulatedTpuHal::threefry2x32([1, 2], [3, 4]);
        assert_eq!(tf_out, SimulatedTpuHal::threefry2x32([1, 2], [3, 4]));
        assert_ne!(tf_out, [0, 0]);
        let ph_out = SimulatedTpuHal::philox4x32_10([1, 2], [3, 4, 5, 6]);
        assert_eq!(ph_out, SimulatedTpuHal::philox4x32_10([1, 2], [3, 4, 5, 6]));
        assert_ne!(ph_out, [0, 0, 0, 0]);

        let c64_forward = SimulatedTpuHal::eval_fft_with_mode(
            ElementType::C64,
            &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
            "FFT",
        );
        let c64_fft_vals = ElementType::C64.unpack_to_f64_vec(&c64_forward);
        let c64_inverse =
            SimulatedTpuHal::eval_fft_with_mode(ElementType::C64, &c64_fft_vals, "IFFT");
        let c64_roundtrip = ElementType::C64.unpack_to_f64_vec(&c64_inverse);
        assert!((c64_roundtrip.first().copied().unwrap_or(0.0) - 1.0).abs() < 1e-4);
        let f32_inverse =
            SimulatedTpuHal::eval_fft_with_mode(ElementType::F32, &[1.0, 2.0, 3.0, 4.0], "IRFFT");
        assert_eq!(f32_inverse.len(), 16);
        let f32_real_fft =
            SimulatedTpuHal::eval_fft_with_mode(ElementType::F32, &[1.0, 2.0, 3.0, 4.0], "RFFT");
        assert_eq!(f32_real_fft.len(), 16);

        let fft_bundle = crate::compiler::TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Fft,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 1,
            attrs: crate::compiler::InstructionAttrs {
                custom_call_target: String::from("FFT"),
                ..crate::compiler::InstructionAttrs::default()
            },
        };
        assert_eq!(
            SimulatedTpuHal::eval_bundle_with_attrs(&fft_bundle, ElementType::F32, &[&cat_in0])
                .len(),
            8
        );

        for bn_op in [
            TpuOpCode::BatchNormInference,
            TpuOpCode::BatchNormTraining,
            TpuOpCode::BatchNormGrad,
            TpuOpCode::SelectAndScatter,
        ] {
            let bn_bundle = crate::compiler::TpuInstructionBundle {
                pc_offset: 0,
                opcode: bn_op,
                operand_slots: vec![0, 1, 2],
                output_slot: 3,
                sync_flag_id: 1,
                attrs: crate::compiler::InstructionAttrs::default(),
            };
            assert_eq!(
                SimulatedTpuHal::eval_bundle_with_attrs(
                    &bn_bundle,
                    ElementType::F32,
                    &[&cat_in0, &cat_in1, &cat_in0]
                )
                .len(),
                8
            );
        }

        let cplx_bytes = SimulatedTpuHal::eval_bundle(
            TpuOpCode::Complex,
            ElementType::C64,
            &[&cat_in0, &cat_in1],
        );
        assert_eq!(cplx_bytes.len(), 16);
        assert_eq!(
            SimulatedTpuHal::eval_unary_op(TpuOpCode::Real, ElementType::F32, &cplx_bytes).len(),
            8
        );
        assert_eq!(
            SimulatedTpuHal::eval_unary_op(TpuOpCode::Imag, ElementType::F32, &cplx_bytes).len(),
            8
        );
        for new_u_op in [
            TpuOpCode::ReplicaId,
            TpuOpCode::PartitionId,
            TpuOpCode::AfterAll,
            TpuOpCode::BitcastConvert,
            TpuOpCode::OptimizationBarrier,
            TpuOpCode::Tuple,
            TpuOpCode::GetTupleElement,
            TpuOpCode::Infeed,
            TpuOpCode::Outfeed,
            TpuOpCode::Send,
            TpuOpCode::Recv,
            TpuOpCode::While,
            TpuOpCode::Conditional,
            TpuOpCode::Call,
        ] {
            assert_ne!(
                SimulatedTpuHal::eval_unary_op(new_u_op, ElementType::F32, &cat_in0).len(),
                0
            );
        }

        let while_bundle = crate::compiler::TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::While,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 1,
            attrs: crate::compiler::InstructionAttrs {
                sub_computations: vec![vec![crate::compiler::TpuInstructionBundle {
                    pc_offset: 0,
                    opcode: TpuOpCode::NegF32,
                    operand_slots: vec![0],
                    output_slot: 0,
                    sync_flag_id: 1,
                    attrs: crate::compiler::InstructionAttrs::default(),
                }]],
                ..crate::compiler::InstructionAttrs::default()
            },
        };
        let while_out = ElementType::F32.unpack_to_f64_vec(
            &SimulatedTpuHal::eval_bundle_with_attrs(&while_bundle, ElementType::F32, &[&cat_in0]),
        );
        assert_eq!(while_out, vec![-1.0, -3.0]);

        let conv_bundle = crate::compiler::TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Convolution,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: crate::compiler::InstructionAttrs {
                window_strides: vec![1],
                rhs_dilation: vec![1],
                ..crate::compiler::InstructionAttrs::default()
            },
        };
        assert_eq!(
            SimulatedTpuHal::eval_bundle_with_attrs(
                &conv_bundle,
                ElementType::F32,
                &[&cat_in0, &cat_in1]
            )
            .len(),
            8
        );

        let branch_bundle = crate::compiler::TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Conditional,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: crate::compiler::InstructionAttrs {
                sub_computations: vec![
                    vec![crate::compiler::TpuInstructionBundle {
                        pc_offset: 0,
                        opcode: TpuOpCode::NegF32,
                        operand_slots: vec![0],
                        output_slot: 0,
                        sync_flag_id: 1,
                        attrs: crate::compiler::InstructionAttrs::default(),
                    }],
                    vec![crate::compiler::TpuInstructionBundle {
                        pc_offset: 0,
                        opcode: TpuOpCode::Identity,
                        operand_slots: vec![0],
                        output_slot: 0,
                        sync_flag_id: 1,
                        attrs: crate::compiler::InstructionAttrs::default(),
                    }],
                ],
                ..crate::compiler::InstructionAttrs::default()
            },
        };
        let cond_true =
            ElementType::F32.unpack_to_f64_vec(&SimulatedTpuHal::eval_bundle_with_attrs(
                &branch_bundle,
                ElementType::F32,
                &[&cat_in0, &cat_in1],
            ));
        assert_eq!(cond_true, vec![-2.0, -4.0]);
        let zero_pred = ElementType::F32.pack_from_f64_slice(&[0.0, 0.0]);
        let cond_false =
            ElementType::F32.unpack_to_f64_vec(&SimulatedTpuHal::eval_bundle_with_attrs(
                &branch_bundle,
                ElementType::F32,
                &[&zero_pred, &cat_in1],
            ));
        assert_eq!(cond_false, vec![2.0, 4.0]);

        for b_op in [
            TpuOpCode::ShiftLeft,
            TpuOpCode::ShiftRightArithmetic,
            TpuOpCode::ShiftRightLogical,
            TpuOpCode::Convolution,
            TpuOpCode::Gather,
            TpuOpCode::DynamicSlice,
            TpuOpCode::Atan2,
            TpuOpCode::Pow,
            TpuOpCode::Pad,
            TpuOpCode::TriangularSolve,
        ] {
            assert_eq!(
                SimulatedTpuHal::eval_binary_op(b_op, ElementType::F32, &[0_u8; 8], &[0_u8; 8])
                    .len(),
                8
            );
        }
        // Test 2x2 TriangularSolve (both non-zero and zero diagonal branches) and non-2x2 non-zero diagonal
        let tri_l = ElementType::F32.pack_from_f64_slice(&[2.0, 0.0, 1.0, 2.0]);
        let tri_b = ElementType::F32.pack_from_f64_slice(&[4.0, 6.0, 4.0, 7.0]);
        assert_eq!(
            SimulatedTpuHal::eval_binary_op(
                TpuOpCode::TriangularSolve,
                ElementType::F32,
                &tri_l,
                &tri_b
            )
            .len(),
            16
        );
        assert_eq!(
            SimulatedTpuHal::eval_binary_op(
                TpuOpCode::TriangularSolve,
                ElementType::F32,
                &[0_u8; 16],
                &tri_b
            )
            .len(),
            16
        );
        let tri_diag_l = ElementType::F32.pack_from_f64_slice(&[2.0, 0.0]);
        let tri_diag_b = ElementType::F32.pack_from_f64_slice(&[4.0, 6.0]);
        assert_eq!(
            SimulatedTpuHal::eval_binary_op(
                TpuOpCode::TriangularSolve,
                ElementType::F32,
                &tri_diag_l,
                &tri_diag_b
            )
            .len(),
            8
        );

        for t_op in [
            TpuOpCode::Scatter,
            TpuOpCode::DynamicUpdateSlice,
            TpuOpCode::Clamp,
        ] {
            assert_eq!(
                SimulatedTpuHal::eval_bundle(
                    t_op,
                    ElementType::F32,
                    &[&[0_u8; 8], &[0_u8; 8], &[0_u8; 8]]
                )
                .len(),
                8
            );
        }
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
            TpuOpCode::CollectivePermute,
            TpuOpCode::AllToAll,
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

        if let Ok(id_prog) = CompiledProgram::compile("mlir", b"return", b"") {
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
        if let Ok(add_prog) = CompiledProgram::compile("mlir", b"add", b"") {
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
        if let Ok(id_prog) = CompiledProgram::compile("mlir", b"return", b"") {
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
        assert!(backend.write_mmio_u64(2, MmioOffset(0), 77).is_ok());
        assert_eq!(backend.read_mmio_u64(2, MmioOffset(0)).ok(), Some(77));
        assert!(backend.read_mmio_u64(1, MmioOffset(0)).is_err());
        assert!(backend.read_mmio_u64(0, MmioOffset(0x100_0000)).is_err());
        assert!(backend.write_mmio_u64(1, MmioOffset(0), 1).is_err());
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
            if let Ok(hal) = hal_res {
                assert_ne!(format!("{hal:?}"), "");
                assert!(hal.discover_topology().is_ok());
                assert_eq!(hal.read_telemetry_csrs(), (45_000, 175_000, 100));
                assert!(hal.register_interrupt_eventfd(-1).is_err());
                assert!(hal.register_interrupt_eventfd(7).is_ok());
                assert!(hal.unregister_interrupt_eventfd().is_ok());
                if mode == KernelDriverMode::Vfio {
                    assert!(hal.initialize_vfio_container_and_group(0).is_ok());
                } else {
                    assert!(hal.initialize_gasket_device(7).is_ok());
                }

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
                assert_eq!(hal.mmio.dma_ring_len(), 3);
                if let Ok(prog) = CompiledProgram::compile("mlir", b"return", b"") {
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

        if let Ok(g_hal) = LinuxDeviceHal::open(
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
            if let Ok(hal) = LinuxDeviceHal::open(
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
            if let Ok(prog) = CompiledProgram::compile("mlir", b"return", b"") {
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

        if let Ok(emu_hal) = LinuxDeviceHal::open(
            KernelDriverMode::Gasket,
            EmulatedKernelDriverBackend::default(),
            &cfg,
        ) {
            assert!(emu_hal.detach_bar0_mmap().is_ok());
            assert!(emu_hal.attach_bar0_mmap(ByteSize(0x0010_0000)).is_ok());
            emu_hal.write_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE, 7);
            assert_eq!(emu_hal.read_bar0_csr(BAR0_CSR_SYNC_FLAG_BASE), 7);
            assert_eq!(emu_hal.read_bar0_csr(0x00FF_FFF8), 0);
            assert!(emu_hal.poll_interrupt_completion_counter(0) >= 1);
            let _ = emu_hal.read_telemetry_csrs();
            assert!(emu_hal.detach_bar0_mmap().is_ok());
            assert!(emu_hal.close().is_ok());
        }
        let bad_fd_hal = LinuxDeviceHal {
            mode: KernelDriverMode::Gasket,
            syscalls: SimulatedKernelDriverBackend::default(),
            device_fd: -1,
            mmio: SimulatedKernelDriverBackend::default(),
            bar0_aperture: parking_lot::Mutex::new(None),
            iommu_allocators: parking_lot::Mutex::new(std::collections::BTreeMap::new()),
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

        if let Ok(fd) = posix.open_device("/dev/null") {
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

        if let Ok(aperture) = posix.mmap_bar_aperture(FileDescriptor(-1), ByteSize(4096)) {
            assert!(
                aperture
                    .write_csr_u64(MmioOffset(0), 0xCAFE_BABE_1234_5678)
                    .is_ok()
            );
            assert_eq!(
                aperture.read_csr_u64(MmioOffset(0)).ok(),
                Some(0xCAFE_BABE_1234_5678)
            );
            assert!(aperture.write_csr_u32(MmioOffset(8), 0x1234_5678).is_ok());
            assert_eq!(aperture.read_csr_u32(MmioOffset(8)).ok(), Some(0x1234_5678));
            assert!(aperture.read_csr_u64(MmioOffset(3)).is_err());
            assert!(aperture.write_csr_u64(MmioOffset(3), 1).is_err());
            assert!(aperture.read_csr_u64(MmioOffset(4096)).is_err());
            assert!(aperture.write_csr_u64(MmioOffset(4096), 1).is_err());
            assert!(aperture.read_csr_u32(MmioOffset(3)).is_err());
            assert!(aperture.write_csr_u32(MmioOffset(3), 1).is_err());
            assert!(aperture.read_csr_u32(MmioOffset(4096)).is_err());
            assert!(aperture.write_csr_u32(MmioOffset(4096), 1).is_err());
            assert!(posix.munmap_bar_aperture(aperture).is_ok());
        }
        let null_aperture = MappedBarAperture {
            bar_index: 2,
            base_vaddr: 0,
            size: ByteSize(4096),
        };
        assert_ne!(format!("{null_aperture:?}"), "");
        assert!(null_aperture.read_csr_u64(MmioOffset(0)).is_err());
        assert!(null_aperture.write_csr_u64(MmioOffset(0), 1).is_err());
        assert!(null_aperture.read_csr_u32(MmioOffset(0)).is_err());
        assert!(null_aperture.write_csr_u32(MmioOffset(0), 1).is_err());
        let unaligned_aperture = MappedBarAperture {
            bar_index: 0,
            base_vaddr: 1,
            size: ByteSize(4096),
        };
        assert!(unaligned_aperture.read_csr_u64(MmioOffset(0)).is_err());
        assert!(unaligned_aperture.write_csr_u64(MmioOffset(0), 1).is_err());
        assert!(unaligned_aperture.read_csr_u32(MmioOffset(0)).is_err());
        assert!(unaligned_aperture.write_csr_u32(MmioOffset(0), 1).is_err());

        let efd = posix.create_eventfd();
        assert!(efd.0 >= 0);
        assert!(posix.close_device(efd.0).is_ok());
        if let Ok((r_fd, w_fd)) = posix.create_eventfd_pair() {
            assert!(posix.close_eventfd_pair(r_fd, w_fd).is_ok());
            assert!(posix.close_eventfd_pair(EventFdHandle(-1), w_fd).is_err());
        }

        assert!(posix.allocate_anon_bar0_mmap(ByteSize(0)).is_err());
        assert!(posix.allocate_anon_bar0_mmap(ByteSize(usize::MAX)).is_err());
        if let Ok(mut mmap_buf) = posix.allocate_anon_bar0_mmap(ByteSize(4096)) {
            assert_eq!(mmap_buf.len(), 4096);
            if let Some(first_byte) = mmap_buf.first_mut() {
                *first_byte = 0xAB;
            }
            assert_eq!(mmap_buf.first().copied(), Some(0xAB));
        }

        // Test bitflags, HbmDmaRingDescriptor, HbmRegionAllocator, and PciDeviceEnumerator
        let ctrl = TensorCoreControlFlags::START | TensorCoreControlFlags::CLEAR_FAULT;
        let status = TensorCoreStatusFlags::IDLE | TensorCoreStatusFlags::COMPLETED;
        let dma_f = VfioDmaMapFlags::READ | VfioDmaMapFlags::WRITE;
        let irq_f = VfioIrqSetFlags::DATA_EVENTFD | VfioIrqSetFlags::ACTION_TRIGGER;
        let ici_f = IciLinkStateFlags::ALL_PORTS_HEALTHY;
        assert!(ici_f.contains(IciLinkStateFlags::PORT_0_UP));
        let ring_desc = HbmDmaRingDescriptor {
            src_iova: 0x1000,
            dst_iova: 0x2000,
            length_bytes: 4096,
            direction_flags: 0,
            completion_sync_flag: 1,
        };
        assert_ne!(
            format!("{ctrl:?}{status:?}{dma_f:?}{irq_f:?}{ici_f:?}{ring_desc:?}"),
            ""
        );
        assert_eq!(bytemuck::bytes_of(&ring_desc).len(), 32);
        assert_eq!(bytemuck::bytes_of(&g_ioctl).len(), 32);
        assert_eq!(bytemuck::bytes_of(&g_efd).len(), 32);
        assert_eq!(bytemuck::bytes_of(&v_map).len(), 32);
        assert_eq!(bytemuck::bytes_of(&v_unmap).len(), 24);
        assert_eq!(bytemuck::bytes_of(&v_irq).len(), 24);

        let empty_alloc = HbmRegionAllocator::new(0x1000_0000, 0);
        assert_eq!(empty_alloc.largest_free_block_bytes(), 0);

        let mut region = HbmRegionAllocator::new(0x1000_0000, 16384);
        assert_ne!(format!("{region:?}"), "");
        assert_eq!(region.largest_free_block_bytes(), 16384);
        assert!(region.allocate(0).is_err());
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
        let b_second = region.allocate(4096).unwrap_or(DeviceAddress(0));
        assert_eq!(region.allocated_block_size(b_second), Some(4096));
        region.set_capacity(32768);
        assert_eq!(region.largest_free_block_bytes(), 24576);
        assert_eq!(region.deallocate(b_keep).ok(), Some(4096));
        region.set_capacity(32768);
        assert_eq!(region.largest_free_block_bytes(), 24576);
        region.set_capacity(2048);
        assert_eq!(region.largest_free_block_bytes(), 2048);
        assert_eq!(region.deallocate(b_second).ok(), Some(4096));
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
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x0073)),
            Some(TpuGeneration::V7x)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0x0076)),
            Some(TpuGeneration::V7x)
        );
        assert_eq!(
            PciDeviceEnumerator::generation_from_pci_device_id(PciDeviceId(0xffff)),
            None
        );
        assert_eq!(
            PciDeviceEnumerator::parse_hex_u16(
                "0x1ae0
"
            ),
            Some(0x1ae0)
        );
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
        let accel_root = tmp_dir.join("accel");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:04.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:05.0/vfio"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:06.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:07.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:08.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:09.0"));
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:0a.0"));
        let _ = std::fs::create_dir_all(dev_dir.join("vfio"));
        let _ = std::fs::create_dir_all(accel_root.join("accel0"));
        // 04.0: Valid Gasket TPU v6e with iommu_group, numa_node, and accel0 telemetry
        let _ = std::fs::write(sysfs_dir.join("0000:00:04.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:04.0/device"), "0x006f\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:04.0/iommu_group"), "7\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:04.0/numa_node"), "1\n");
        let _ = std::fs::write(accel_root.join("accel0/bar_size"), "2097152\n");
        let _ = std::fs::write(accel_root.join("accel0/driver_version"), "2.4.0\n");
        let _ = std::fs::write(accel_root.join("accel0/chip_model"), "trillium_v6e\n");
        let _ = std::fs::write(accel_root.join("accel0/is_device_owned"), "1\n");
        // 05.0: Valid VFIO TPU v5p
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:05.0/vendor"),
            "0x1ae0
",
        );
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:05.0/device"),
            "0x005e
",
        );
        // 06.0: Non-Google vendor
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:06.0/vendor"),
            "0x8086
",
        );
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:06.0/device"),
            "0x1234
",
        );
        // 07.0: Missing device file
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:07.0/vendor"),
            "0x1ae0
",
        );
        // 08.0: Invalid vendor hex
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:08.0/vendor"),
            "bad
",
        );
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:08.0/device"),
            "0x006f
",
        );
        // 09.0: Invalid device hex
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:09.0/vendor"),
            "0x1ae0
",
        );
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:09.0/device"),
            "bad
",
        );
        // 0a.0: Unknown Google PCI device ID
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:0a.0/vendor"),
            "0x1ae0
",
        );
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:0a.0/device"),
            "0x9999
",
        );

        let scanned = PciDeviceEnumerator::scan_sysfs_with_accel(&sysfs_dir, &dev_dir, &accel_root);
        assert_eq!(scanned.len(), 2);
        let root_scanned = PciSysfsScanner::discover_from_sysfs_root(&sysfs_dir);
        assert_eq!(root_scanned.len(), 2);
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:0b.0"));
        let _ = std::fs::write(sysfs_dir.join("0000:00:0b.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:0b.0/device"), "0x0027\n");
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:0c.0"));
        let _ = std::fs::write(sysfs_dir.join("0000:00:0c.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:0c.0/device"), "0x0056\n");
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:0d.0"));
        let _ = std::fs::write(sysfs_dir.join("0000:00:0d.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:0d.0/device"), "0x0062\n");
        let _ = std::fs::create_dir_all(sysfs_dir.join("0000:00:0e.0"));
        let _ = std::fs::write(sysfs_dir.join("0000:00:0e.0/vendor"), "0x1ae0\n");
        let _ = std::fs::write(sysfs_dir.join("0000:00:0e.0/device"), "0x0075\n");
        if let Ok(sysfs_devs) = discover_sysfs_tpu_pci_devices(&sysfs_dir) {
            assert_ne!(sysfs_devs, [] as [DiscoveredPciDevice; 0]);
            if let Some(first_dev) = sysfs_devs.first() {
                assert_eq!(first_dev.clone(), *first_dev);
                assert_ne!(format!("{first_dev:?}"), "");
            }
        }
        assert!(
            discover_sysfs_tpu_pci_devices(std::path::Path::new("/nonexistent_sysfs_dir")).is_err()
        );
        assert_ne!(format!("{PciDeviceEnumerator:?}{scanned:?}"), "");
        let _ = std::fs::remove_dir_all(&tmp_dir);

        for tpu_gen in [
            TpuGeneration::V2,
            TpuGeneration::V3,
            TpuGeneration::V4,
            TpuGeneration::V4Lite,
            TpuGeneration::V5e,
            TpuGeneration::V5p,
            TpuGeneration::V6e,
            TpuGeneration::V7x,
        ] {
            let rmap = ChipRegisterMap::for_generation(tpu_gen);
            assert_eq!(rmap.generation, tpu_gen);
            assert_ne!(rmap.chip_codename, "");
            assert!(rmap.hardware_abi_version >= 65_537);
            assert_eq!(rmap.clone(), rmap);
            assert_ne!(format!("{rmap:?}"), "");
            let descs = rmap.csr_descriptors();
            assert!(descs.len() >= 6);
            assert!(ChipRegisterMap::lookup_csr_by_name(tpu_gen, "RUN_STATUS").is_some());
            assert!(
                ChipRegisterMap::lookup_csr_by_name(tpu_gen, "tc_0.tcs_core_state.PC").is_some()
            );
            assert!(ChipRegisterMap::lookup_csr_by_name(tpu_gen, "NONEXISTENT_CSR").is_none());
            assert!(
                ChipRegisterMap::lookup_csr_by_offset(tpu_gen, rmap.tc0_run_status_csr).is_some()
            );
            assert!(ChipRegisterMap::lookup_csr_by_offset(tpu_gen, 0xDEAD_BEEF).is_none());
            if let Some(d0) = descs.first() {
                assert_eq!(d0.access, CsrAccessPermission::WriteOnly);
                assert_eq!(d0.clone(), *d0);
                assert_eq!(d0.apply_write_value(0, 99), 99);
                let bf0 = d0.bitfields.first().copied();
                assert_eq!(bf0.clone(), bf0);
                if let Some(bf) = bf0 {
                    assert_eq!(bf.unpack(bf.pack(0, 42)), 42);
                }
                assert_eq!(
                    d0.unpack_bitfield(d0.pack_bitfield(0, "VALUE", 55).unwrap_or(0), "VALUE"),
                    Some(55)
                );
                assert!(d0.unpack_bitfield(0, "MISSING_BF").is_none());
                assert!(d0.pack_bitfield(0, "MISSING_BF", 1).is_none());
                assert_ne!(
                    format!("{d0:?}{bf0:?}{:?}", CsrAccessPermission::WriteOnly),
                    ""
                );
            }
            if let Some(run_desc) = rmap.lookup_by_name("RUN_STATUS") {
                assert_eq!(run_desc.apply_write_value(7, 99), 7);
                let packed_state = run_desc.pack_bitfield(0, "STATE", 3).unwrap_or(0);
                assert_eq!(run_desc.unpack_bitfield(packed_state, "STATE"), Some(3));
            }
            if let Some(w1c_desc) = rmap.lookup_by_name("VMEM_FAULT_ADDR") {
                assert_eq!(w1c_desc.apply_write_value(0b1111, 0b0101), 0b1010);
            }
        }
        let emu_mmio = EmulatedKernelDriverBackend::default();
        emu_mmio.reset_csrs_for_generation(TpuGeneration::V5e);
        let v5e_map = ChipRegisterMap::for_generation(TpuGeneration::V5e);
        assert_eq!(
            emu_mmio.write_csr_checked(TpuGeneration::V5e, v5e_map.tc0_run_status_csr, 99),
            1
        );
        assert_eq!(
            emu_mmio.write_csr_checked(TpuGeneration::V5e, v5e_map.tc0_pc_csr, 128),
            128
        );
        assert_eq!(
            emu_mmio.write_csr_checked(TpuGeneration::V5e, 0xDEAD_BEEF, 77),
            77
        );

        if let Ok((r_fd, w_fd)) = PosixKernelDriverBackend.create_eventfd_pair() {
            assert!(PosixKernelDriverBackend.signal_eventfd(w_fd, 7).is_ok());
            assert_eq!(
                PosixKernelDriverBackend.wait_for_interrupt_eventfd(r_fd, 50),
                Ok(7)
            );
            assert!(
                PosixKernelDriverBackend
                    .wait_for_interrupt_eventfd(r_fd, 0)
                    .is_err()
            );
            let _ = PosixKernelDriverBackend.close_eventfd_pair(r_fd, w_fd);
        }
        assert!(
            PosixKernelDriverBackend
                .signal_eventfd(EventFdHandle(-1), 1)
                .is_err()
        );
        if let Ok((r_short, w_short)) = PosixKernelDriverBackend.create_eventfd_pair() {
            let _ = PosixKernelDriverBackend.close_device(w_short.0);
            assert!(
                PosixKernelDriverBackend
                    .wait_for_interrupt_eventfd(r_short, 10)
                    .is_err()
            );
            let _ = PosixKernelDriverBackend.close_device(r_short.0);
        }

        let mut vm = TpuCoreVmState::default();
        let tile_f32 = ElementType::F32.pack_from_f64_slice(&[2.0, 4.0, 6.0, 8.0]);
        vm.vreg_tiles.insert(0, tile_f32.clone());
        vm.vreg_tiles.insert(1, tile_f32.clone());
        vm.vmem_allocations.insert(32, tile_f32);
        assert!(
            PosixKernelDriverBackend
                .mmap_bar_aperture(FileDescriptor(-1), ByteSize(0))
                .is_err()
        );
        assert_eq!(vm.cycles_executed, 0);
        assert!(vm.wait_sync_flag(0, 1, 1).is_err());
        if let Ok(prog) = CompiledProgram::compile("mlir", b"dot_general", b"") {
            let mut m_bundles = prog.encode_vliw_program();
            if let Some(b0) = m_bundles.first_mut() {
                b0.matrix_slot.lhs_vreg = 0;
                b0.matrix_slot.rhs_vreg = 1;
                b0.matrix_slot.acc_vreg = 2;
                b0.dma_slot.dma_opcode = 1;
                b0.dma_slot.src_space = 1;
            }
            vm.execute_vliw_stream(&m_bundles);
            vm.mxu_weights = vec![1.0, 1.0, 1.0, 1.0];
            vm.execute_vliw_stream(&m_bundles);
            if let Some(ds_exec) = prog.to_tpu_executable_proto().ds_executable {
                vm.execute_deepsea_proto(&ds_exec);
            }
        }
        if let Ok(prog_exp) = CompiledProgram::compile("mlir", b"exp_f32", b"") {
            let mut bundles = prog_exp.encode_vliw_program();
            vm.preg[0] = true;
            for (idx, sc_op) in [2_u8, 3, 4, 5, 6].into_iter().enumerate() {
                if let Some(b0) = bundles.first().copied() {
                    let mut b_mod = b0;
                    b_mod.scalar_slot.opcode = sc_op;
                    b_mod.scalar_slot.src0_sreg = 0;
                    b_mod.scalar_slot.dst_sreg = u8::try_from(idx + 1).unwrap_or(1);
                    b_mod.scalar_slot.imm32 = 1;
                    bundles.push(b_mod);
                }
            }
            if let Some(b0) = bundles.first().copied() {
                let mut b_br_false = b0;
                b_br_false.scalar_slot.opcode = 6;
                b_br_false.scalar_slot.src0_sreg = 10;
                b_br_false.scalar_slot.dst_sreg = 10;
                b_br_false.scalar_slot.imm32 = 0;
                bundles.push(b_br_false);
            }
            for v_op in [
                TpuOpCode::AddF32 as u16,
                TpuOpCode::SubF32 as u16,
                TpuOpCode::MulF32 as u16,
                TpuOpCode::DivF32 as u16,
                TpuOpCode::MaxF32 as u16,
                TpuOpCode::MinF32 as u16,
                TpuOpCode::NegF32 as u16,
                TpuOpCode::Abs as u16,
                TpuOpCode::Identity as u16,
                TpuOpCode::Select as u16,
                TpuOpCode::Clamp as u16,
                TpuOpCode::CompareEq as u16,
                TpuOpCode::CompareLt as u16,
                TpuOpCode::CompareGt as u16,
                TpuOpCode::And as u16,
                TpuOpCode::Or as u16,
                TpuOpCode::XorBytes as u16,
                TpuOpCode::Not as u16,
                TpuOpCode::ShiftLeft as u16,
                TpuOpCode::ShiftRightLogical as u16,
                TpuOpCode::ShiftRightArithmetic as u16,
                TpuOpCode::PopulationCount as u16,
                TpuOpCode::CountLeadingZeros as u16,
                TpuOpCode::UniformQuantize as u16,
                TpuOpCode::UniformDequantize as u16,
                TpuOpCode::BitcastConvert as u16,
                999_u16,
            ] {
                if let Some(b0) = bundles.first().copied() {
                    let mut b_mod = b0;
                    b_mod.vector_slot.opcode = v_op;
                    b_mod.vector_slot.lane_mask = 0b0101_0101;
                    bundles.push(b_mod);
                }
            }
            vm.vreg_words[0] = 2.0_f32.to_bits();
            vm.vreg_words[1] = 4.0_f32.to_bits();
            if let Some(b0) = bundles.first().copied() {
                let mut b_div_nonzero = b0;
                b_div_nonzero.vector_slot.opcode = TpuOpCode::DivF32 as u16;
                b_div_nonzero.vector_slot.src0_vreg = 0;
                b_div_nonzero.vector_slot.src1_vreg = 1;
                bundles.push(b_div_nonzero);
            }
            for (x_op, p_mode, imm) in [
                (TpuOpCode::ExpF32 as u8, 0_u8, 0_u16),
                (TpuOpCode::TanhF32 as u8, 0, 0),
                (TpuOpCode::Log as u8, 0, 0),
                (TpuOpCode::Sqrt as u8, 0, 0),
                (TpuOpCode::Rsqrt as u8, 0, 0),
                (TpuOpCode::Sine as u8, 0, 0),
                (TpuOpCode::Cosine as u8, 0, 0),
                (TpuOpCode::Transpose as u8, 2, 0),
                (TpuOpCode::Transpose as u8, 0, 0),
                (TpuOpCode::Reverse as u8, 0, 0),
                (TpuOpCode::Iota as u8, 0, 4),
                (TpuOpCode::DynamicSlice as u8, 0, 1),
                (TpuOpCode::BroadcastInDim as u8, 0, 8),
                (TpuOpCode::Slice as u8, 1, 3),
                (TpuOpCode::Slice as u8, 0, 0),
                (TpuOpCode::Pad as u8, 1, 1),
                (TpuOpCode::Concatenate as u8, 0, 1),
                (TpuOpCode::Gather as u8, 0, 1),
                (TpuOpCode::Scatter as u8, 0, 1),
                (250_u8, 0, 0),
            ] {
                if let Some(b0) = bundles.first().copied() {
                    let mut b_mod = b0;
                    b_mod.xunary_slot.opcode = x_op;
                    b_mod.xunary_slot.perm_mode = p_mode;
                    b_mod.xunary_slot.imm16 = imm;
                    b_mod.xunary_slot.src_vreg = 0;
                    b_mod.xunary_slot.dst_vreg = 2;
                    bundles.push(b_mod);
                }
            }
            // Test rectangular & batched [2, 2, 2] x [2, 2, 4] -> [2, 2, 4] MXU across BF16, F32, INT8, FP8, S4
            let rect_lhs = ElementType::F32.pack_from_f64_slice(&[1.5_f64; 4]);
            let rect_rhs = ElementType::F32.pack_from_f64_slice(&[2.0_f64; 2 * 2 * 4]);
            vm.vreg_tiles.insert(10, rect_lhs);
            vm.vreg_tiles.insert(11, rect_rhs);
            vm.mxu_weights.clear();
            for p_mode in 0..=4_u8 {
                if let Some(b0) = bundles.first().copied() {
                    let mut b_mxu = b0;
                    b_mxu.matrix_slot.opcode = 1;
                    b_mxu.matrix_slot.precision_mode = p_mode;
                    b_mxu.matrix_slot.lhs_vreg = 10;
                    b_mxu.matrix_slot.rhs_vreg = 11;
                    b_mxu.matrix_slot.acc_vreg = 12;
                    b_mxu.matrix_slot.flags = (2 << 8) | 4;
                    b_mxu.matrix_slot.k_tiles = (2 << 8) | 2;
                    vm.step_vliw_bundle(&b_mxu);
                    let out_tile = vm.vreg_tiles.get(&12).cloned().unwrap_or_default();
                    let unpacked = ElementType::F32.unpack_to_f64_vec(&out_tile);
                    assert_eq!(unpacked.len(), 2 * 2 * 4);
                }
            }
            vm.execute_vliw_stream(&bundles);
        }
        assert!(vm.wait_sync_flag(0, 1, 1).is_ok());
        assert!(vm.cycles_executed >= 1);
        assert!(vm.mxu_accum.iter().any(|&v| v >= 1.0));
        assert_ne!(format!("{vm:?}"), "");
        let vm_cloned = vm.clone();
        assert_eq!(vm_cloned.cycles_executed, vm.cycles_executed);

        let hcmd = crate::proto::HostCommandEnvelopeProto {
            opcode: 1,
            continuation_id: 77,
            device_id: 0,
            core_index: 0,
            handler_id: 1,
            payload: Vec::new(),
        };
        vm.dispatch_host_command(&hcmd);
        assert_eq!(vm.sreg[0], 77);
        let hcmd2 = crate::proto::HostCommandEnvelopeProto {
            opcode: 2,
            continuation_id: 78,
            device_id: 0,
            core_index: 0,
            handler_id: 1,
            payload: vec![1, 2, 3, 4],
        };
        vm.dispatch_host_command(&hcmd2);
        assert_eq!(vm.sreg[0], 78);

        if let Ok(mut prog) =
            CompiledProgram::compile("mlir", b"module @vm_exec { stablehlo.add }", b"")
        {
            let in_bytes = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
            let imem_bin = prog.encode_architectural_imem();
            assert!(
                vm.execute_imem_binary(&imem_bin, vec![in_bytes.clone(), in_bytes.clone()], 1)
                    .is_ok()
            );
            assert!(vm.execute_imem_binary(&[], vec![], 1).is_err());
            assert!(vm.execute_imem_binary(&[0_u8; 17], vec![], 1).is_err());
            let mut bad_imem = imem_bin;
            if let Some(b) = bad_imem.get_mut(16) {
                *b = 250;
            }
            assert!(vm.execute_imem_binary(&bad_imem, vec![], 1).is_err());
            let exec_res = vm.execute(&prog, vec![in_bytes.clone(), in_bytes.clone()]);
            assert!(exec_res.is_ok());
            prog.output_slots = vec![99];
            assert!(vm.execute(&prog, vec![in_bytes.clone()]).is_err());
            prog.output_slots.clear();
            assert!(vm.execute(&prog, vec![in_bytes]).is_err());
        }

        // Test SparseCoreVmState gather and SGD / Adagrad / Adam / Ftrl / RmsProp / Momentum / Lion scatter updates
        let mut sc_vm = SparseCoreVmState::new(0, 4, 2, 1.0);
        let gathered = sc_vm.sparse_gather(&[1, 3]);
        assert_eq!(gathered, vec![1.0, 1.0, 1.0, 1.0]);
        for opt in [
            SparseCoreOptimizer::Sgd { learning_rate: 0.1 },
            SparseCoreOptimizer::Adagrad {
                learning_rate: 0.1,
                epsilon: 1e-6,
            },
            SparseCoreOptimizer::Adam {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            SparseCoreOptimizer::Ftrl {
                learning_rate: 0.1,
                learning_rate_power: -0.5,
                l1_regularization: 10.0,
                l2_regularization: 0.01,
            },
            SparseCoreOptimizer::Ftrl {
                learning_rate: 0.1,
                learning_rate_power: -0.5,
                l1_regularization: 0.0,
                l2_regularization: 0.01,
            },
            SparseCoreOptimizer::RmsProp {
                learning_rate: 0.01,
                decay: 0.9,
                momentum: 0.9,
                epsilon: 1e-6,
            },
            SparseCoreOptimizer::Momentum {
                learning_rate: 0.05,
                momentum: 0.9,
            },
            SparseCoreOptimizer::Lion {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.99,
                weight_decay: 0.01,
            },
        ] {
            sc_vm.sparse_scatter_update(&[1], &[0.5, -0.5], opt);
        }
        assert!(sc_vm.steps_executed >= 8);
        assert_eq!(sc_vm.clone(), sc_vm);
        assert_ne!(
            format!(
                "{sc_vm:?}{:?}",
                SparseCoreOptimizer::Sgd { learning_rate: 0.1 }.clone()
            ),
            ""
        );

        // Test 3x3 Cholesky and TriangularSolve
        let chol_3x3 =
            SimulatedTpuHal::eval_cholesky(&[4.0, 2.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 9.0]);
        assert_eq!(chol_3x3.len(), 9);
        assert!((chol_3x3.first().copied().unwrap_or(0.0) - 2.0).abs() < 1e-6);
        assert!((chol_3x3.get(8).copied().unwrap_or(0.0) - 3.0).abs() < 1e-6);
        let tri_3x3 = SimulatedTpuHal::eval_triangular_solve(
            &[2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0],
            &[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 9.0, 0.0, 0.0],
        );
        assert_eq!(tri_3x3.len(), 9);
        assert!((tri_3x3.first().copied().unwrap_or(0.0) - 1.0).abs() < 1e-6);
        assert!((tri_3x3.get(3).copied().unwrap_or(0.0) - 1.0).abs() < 1e-6);
        assert!((tri_3x3.get(6).copied().unwrap_or(0.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_exact_integer_alu_and_nd_tensor_semantics() {
        use crate::compiler::{InstructionAttrs, TpuInstructionBundle};
        use crate::types::{CollectiveReductionOp, MemoryLayout};

        // 1. Exact 64-bit S64 & U64 arithmetic above 2^53 (where f64 loses precision)
        let a_s64 = ElementType::S64.pack_from_i64_slice(&[0x7FFF_FFFF_FFFF_FFFE_i64, -10]);
        let b_s64 = ElementType::S64.pack_from_i64_slice(&[1_i64, 3]);
        let sum_s64 =
            SimulatedTpuHal::eval_binary_op(TpuOpCode::AddS32, ElementType::S64, &a_s64, &b_s64);
        assert_eq!(
            ElementType::S64.unpack_to_i64_vec(&sum_s64),
            vec![0x7FFF_FFFF_FFFF_FFFF_i64, -7]
        );
        let shift_s64 = SimulatedTpuHal::eval_binary_op(
            TpuOpCode::ShiftLeft,
            ElementType::S64,
            &ElementType::S64.pack_from_i64_slice(&[1_i64, 3_i64]),
            &ElementType::S64.pack_from_i64_slice(&[48_i64, 40_i64]),
        );
        assert_eq!(
            ElementType::S64.unpack_to_i64_vec(&shift_s64),
            vec![1_i64 << 48, 3_i64 << 40]
        );
        for bin_op in [
            TpuOpCode::SubF32,
            TpuOpCode::MulF32,
            TpuOpCode::DivF32,
            TpuOpCode::Rem,
            TpuOpCode::MaxF32,
            TpuOpCode::MinF32,
            TpuOpCode::And,
            TpuOpCode::Or,
            TpuOpCode::ShiftRightArithmetic,
            TpuOpCode::ShiftRightLogical,
            TpuOpCode::CompareEq,
            TpuOpCode::CompareLt,
            TpuOpCode::CompareGt,
        ] {
            let _ = SimulatedTpuHal::eval_binary_op(bin_op, ElementType::S64, &a_s64, &b_s64);
            let _ = SimulatedTpuHal::eval_binary_op(bin_op, ElementType::U64, &a_s64, &b_s64);
            let _ = SimulatedTpuHal::eval_binary_op(bin_op, ElementType::U16, &[4, 0], &[2, 0]);
        }
        let zero_s64 = ElementType::S64.pack_from_i64_slice(&[0_i64, 0]);
        let _ =
            SimulatedTpuHal::eval_binary_op(TpuOpCode::DivF32, ElementType::S64, &a_s64, &zero_s64);
        let _ =
            SimulatedTpuHal::eval_binary_op(TpuOpCode::Rem, ElementType::S64, &a_s64, &zero_s64);

        for un_op in [
            TpuOpCode::ReplicaId,
            TpuOpCode::NegF32,
            TpuOpCode::Abs,
            TpuOpCode::Sign,
            TpuOpCode::Not,
            TpuOpCode::PopulationCount,
            TpuOpCode::CountLeadingZeros,
            TpuOpCode::Sort,
            TpuOpCode::TopK,
            TpuOpCode::Iota,
            TpuOpCode::Reverse,
            TpuOpCode::ReduceMax,
            TpuOpCode::ReduceMin,
            TpuOpCode::ReduceSumF32,
            TpuOpCode::ExpF32,
        ] {
            let _ = SimulatedTpuHal::eval_unary_op(un_op, ElementType::S64, &a_s64);
            let _ = SimulatedTpuHal::eval_unary_op(un_op, ElementType::U64, &a_s64);
            let _ = SimulatedTpuHal::eval_unary_op(un_op, ElementType::S16, &[1, 0, 2, 0]);
            let _ = SimulatedTpuHal::eval_unary_op(un_op, ElementType::Pred, &[0, 1]);
        }

        // Integer Select & Clamp
        let _ = SimulatedTpuHal::eval_bundle(
            TpuOpCode::Select,
            ElementType::S64,
            &[&zero_s64, &a_s64, &b_s64],
        );
        let _ = SimulatedTpuHal::eval_bundle(
            TpuOpCode::Clamp,
            ElementType::S64,
            &[&b_s64, &a_s64, &a_s64],
        );
        let _ = SimulatedTpuHal::eval_bundle(
            TpuOpCode::Clamp,
            ElementType::U64,
            &[&b_s64, &a_s64, &a_s64],
        );

        // 2. BitcastConvert & integer Convert
        let f32_bytes = 1.0_f32.to_le_bytes();
        let bitcast_bundle = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::BitcastConvert,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 1,
            attrs: InstructionAttrs {
                input_shapes: vec![Shape {
                    element_type: ElementType::F32,
                    dims: vec![1],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                }],
                output_shape: Some(Shape {
                    element_type: ElementType::S32,
                    dims: vec![1],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                }),
                ..InstructionAttrs::default()
            },
        };
        let bc_out = SimulatedTpuHal::eval_bundle_with_attrs(
            &bitcast_bundle,
            ElementType::S32,
            &[&f32_bytes],
        );
        assert_eq!(bc_out, 0x3F80_0000_u32.to_le_bytes());

        let int_conv_bundle = TpuInstructionBundle {
            opcode: TpuOpCode::Convert,
            attrs: InstructionAttrs {
                input_shapes: vec![Shape {
                    element_type: ElementType::S64,
                    dims: vec![2],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                }],
                output_shape: Some(Shape {
                    element_type: ElementType::S32,
                    dims: vec![2],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                }),
                ..InstructionAttrs::default()
            },
            ..bitcast_bundle
        };
        let conv_int_out = SimulatedTpuHal::eval_bundle_with_attrs(
            &int_conv_bundle,
            ElementType::S32,
            &[&ElementType::S64.pack_from_i64_slice(&[42, -7])],
        );
        assert_eq!(
            ElementType::S32.unpack_to_i64_vec(&conv_int_out),
            vec![42, -7]
        );

        // 3. N-D Concatenate, DynamicSlice, DynamicUpdateSlice
        let sh_2x2 = Shape {
            element_type: ElementType::F32,
            dims: vec![2, 2],
            dynamic_dims: vec![false, false],
            layout: MemoryLayout::default_tpu_tiled(2),
        };
        let sh_2x4 = Shape {
            element_type: ElementType::F32,
            dims: vec![2, 4],
            dynamic_dims: vec![false, false],
            layout: MemoryLayout::default_tpu_tiled(2),
        };
        let sh_1x1 = Shape {
            element_type: ElementType::F32,
            dims: vec![1, 1],
            dynamic_dims: vec![false, false],
            layout: MemoryLayout::default_tpu_tiled(2),
        };
        let m0 = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
        let m1 = ElementType::F32.pack_from_f64_slice(&[5.0, 6.0, 7.0, 8.0]);
        let cat_nd = SimulatedTpuHal::eval_concatenate_nd(
            ElementType::F32,
            ElementType::F32,
            &[&m0, &m1],
            &InstructionAttrs {
                concat_dim: 1,
                input_shapes: vec![sh_2x2.clone(), sh_2x2.clone()],
                output_shape: Some(sh_2x4),
                ..InstructionAttrs::default()
            },
        );
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&cat_nd),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );

        let idx_1 = ElementType::S32.pack_from_i64_slice(&[1]);
        let dyn_slice_out = SimulatedTpuHal::eval_dynamic_slice_nd(
            ElementType::F32,
            ElementType::F32,
            &[&m0, &idx_1, &idx_1],
            &InstructionAttrs {
                input_shapes: vec![
                    sh_2x2.clone(),
                    Shape {
                        element_type: ElementType::S32,
                        dims: vec![1],
                        dynamic_dims: vec![false],
                        layout: MemoryLayout::default_tpu_tiled(1),
                    },
                    Shape {
                        element_type: ElementType::S32,
                        dims: vec![1],
                        dynamic_dims: vec![false],
                        layout: MemoryLayout::default_tpu_tiled(1),
                    },
                ],
                output_shape: Some(sh_1x1.clone()),
                ..InstructionAttrs::default()
            },
        );
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&dyn_slice_out),
            vec![4.0]
        );

        let upd_val = ElementType::F32.pack_from_f64_slice(&[99.0]);
        let dyn_update_out = SimulatedTpuHal::eval_dynamic_update_slice_nd(
            ElementType::F32,
            ElementType::F32,
            &[&m0, &upd_val, &idx_1, &idx_1],
            &InstructionAttrs {
                input_shapes: vec![sh_2x2.clone(), sh_1x1],
                output_shape: Some(sh_2x2.clone()),
                ..InstructionAttrs::default()
            },
        );
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&dyn_update_out),
            vec![1.0, 2.0, 3.0, 99.0]
        );

        // 4. N-D Convolution & ReduceWindow & Gather & Scatter
        let activation_sh = Shape {
            element_type: ElementType::F32,
            dims: vec![1, 3, 1],
            dynamic_dims: vec![false; 3],
            layout: MemoryLayout::default_tpu_tiled(3),
        };
        let kernel_sh = Shape {
            element_type: ElementType::F32,
            dims: vec![2, 1, 1],
            dynamic_dims: vec![false; 3],
            layout: MemoryLayout::default_tpu_tiled(3),
        };
        let conv_nd = SimulatedTpuHal::eval_convolution_nd(
            ElementType::F32,
            &[1.0, 2.0, 3.0],
            &[1.0, 1.0],
            &InstructionAttrs {
                input_shapes: vec![activation_sh, kernel_sh],
                window_strides: vec![1],
                pad_edge_low: vec![0],
                pad_edge_high: vec![0],
                lhs_dilation: vec![1],
                rhs_dilation: vec![1],
                feature_group_count: 1,
                batch_group_count: 1,
                ..InstructionAttrs::default()
            },
        );
        assert_eq!(ElementType::F32.unpack_to_f64_vec(&conv_nd), vec![3.0, 5.0]);

        for red_kind in [
            CollectiveReductionOp::Sum,
            CollectiveReductionOp::Max,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Product,
        ] {
            let rw_nd = SimulatedTpuHal::eval_reduce_window_nd(
                ElementType::F32,
                &[1.0, 2.0, 3.0, 4.0],
                None,
                &InstructionAttrs {
                    input_shapes: vec![sh_2x2.clone()],
                    window_dims: vec![2, 2],
                    window_strides: vec![1, 1],
                    pad_edge_low: vec![0, 0],
                    pad_edge_high: vec![0, 0],
                    reduce_op_kind: red_kind,
                    ..InstructionAttrs::default()
                },
            );
            assert_eq!(ElementType::F32.unpack_to_f64_vec(&rw_nd).len(), 1);
        }

        let gat_nd = SimulatedTpuHal::eval_gather_nd(
            ElementType::F32,
            ElementType::F32,
            &[&m0, &idx_1],
            &InstructionAttrs {
                input_shapes: vec![
                    sh_2x2.clone(),
                    Shape {
                        element_type: ElementType::S32,
                        dims: vec![1],
                        dynamic_dims: vec![false],
                        layout: MemoryLayout::default_tpu_tiled(1),
                    },
                ],
                gather_slice_sizes: vec![1, 2],
                ..InstructionAttrs::default()
            },
        );
        assert_eq!(ElementType::F32.unpack_to_f64_vec(&gat_nd), vec![3.0, 4.0]);
        let rw_bundle_out = SimulatedTpuHal::eval_bundle_with_attrs(
            &TpuInstructionBundle {
                pc_offset: 0,
                opcode: TpuOpCode::ReduceWindow,
                operand_slots: vec![0, 1],
                output_slot: 2,
                sync_flag_id: 1,
                attrs: InstructionAttrs {
                    window_dims: vec![1, 1],
                    window_strides: vec![1, 1],
                    ..InstructionAttrs::default()
                },
            },
            ElementType::F32,
            &[&m0, &idx_1],
        );
        assert_eq!(rw_bundle_out.len(), 16);

        let row_upd = ElementType::F32.pack_from_f64_slice(&[10.0, 20.0]);
        for red_kind in [
            CollectiveReductionOp::Sum,
            CollectiveReductionOp::Max,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Product,
        ] {
            let scat_nd = SimulatedTpuHal::eval_scatter_nd(
                ElementType::F32,
                ElementType::F32,
                &[&m0, &idx_1, &row_upd],
                &InstructionAttrs {
                    input_shapes: vec![
                        sh_2x2.clone(),
                        Shape {
                            element_type: ElementType::S32,
                            dims: vec![1],
                            dynamic_dims: vec![false],
                            layout: MemoryLayout::default_tpu_tiled(1),
                        },
                        Shape {
                            element_type: ElementType::F32,
                            dims: vec![1, 2],
                            dynamic_dims: vec![false, false],
                            layout: MemoryLayout::default_tpu_tiled(2),
                        },
                    ],
                    scatter_update_window_dims: vec![1],
                    reduce_op_kind: red_kind,
                    ..InstructionAttrs::default()
                },
            );
            assert_eq!(scat_nd.len(), 16);
        }

        let scat_overwrite = SimulatedTpuHal::eval_scatter_nd(
            ElementType::F32,
            ElementType::F32,
            &[&m0, &idx_1, &row_upd],
            &InstructionAttrs {
                input_shapes: vec![sh_2x2],
                ..InstructionAttrs::default()
            },
        );
        assert_eq!(scat_overwrite.len(), 16);

        // 5. Iterative While loop & multi-branch Conditional & CustomCall registry / linalg
        let while_bundle = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::While,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 1,
            attrs: InstructionAttrs {
                sub_computations: vec![
                    vec![TpuInstructionBundle {
                        pc_offset: 0,
                        opcode: TpuOpCode::Not,
                        operand_slots: vec![0],
                        output_slot: 0,
                        sync_flag_id: 1,
                        attrs: InstructionAttrs::default(),
                    }],
                    vec![TpuInstructionBundle {
                        pc_offset: 0,
                        opcode: TpuOpCode::Abs,
                        operand_slots: vec![0],
                        output_slot: 0,
                        sync_flag_id: 1,
                        attrs: InstructionAttrs::default(),
                    }],
                ],
                ..InstructionAttrs::default()
            },
        };
        let w_out = SimulatedTpuHal::eval_bundle_with_attrs(
            &while_bundle,
            ElementType::F32,
            &[&[0, 0, 0, 0]],
        );
        assert_eq!(w_out.len(), 4);
        let w_break_out = SimulatedTpuHal::eval_bundle_with_attrs(
            &while_bundle,
            ElementType::F32,
            &[&[0, 0, 128, 63]],
        );
        assert_eq!(w_break_out.len(), 4);

        let case_bundle = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Conditional,
            operand_slots: vec![0, 1, 2, 3],
            output_slot: 4,
            sync_flag_id: 1,
            attrs: InstructionAttrs {
                input_shapes: vec![Shape {
                    element_type: ElementType::S32,
                    dims: vec![1],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                }],
                sub_computations: vec![vec![], vec![], vec![]],
                ..InstructionAttrs::default()
            },
        };
        let _ = SimulatedTpuHal::eval_bundle_with_attrs(
            &case_bundle,
            ElementType::F32,
            &[&idx_1, &m0, &m1, &m0],
        );

        register_custom_call_handler("custom_double", |inputs, _attrs| {
            let b = inputs.first().copied().unwrap_or(&[]);
            let vals: Vec<f64> = ElementType::F32
                .unpack_to_f64_vec(b)
                .into_iter()
                .map(|x| x * 2.0)
                .collect();
            Ok(ElementType::F32.pack_from_f64_slice(&vals))
        });
        let cc_bundle = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::CustomCall,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 1,
            attrs: InstructionAttrs {
                custom_call_target: String::from("custom_double"),
                ..InstructionAttrs::default()
            },
        };
        let cc_out = SimulatedTpuHal::eval_bundle_with_attrs(&cc_bundle, ElementType::F32, &[&m0]);
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&cc_out),
            vec![2.0, 4.0, 6.0, 8.0]
        );
        assert!(unregister_custom_call_handler("custom_double"));
        assert!(!unregister_custom_call_handler("custom_double"));

        for linalg_target in ["Qr", "Eigh", "Lu"] {
            let l_out = SimulatedTpuHal::eval_bundle_with_attrs(
                &TpuInstructionBundle {
                    opcode: TpuOpCode::CustomCall,
                    attrs: InstructionAttrs {
                        custom_call_target: String::from(linalg_target),
                        ..InstructionAttrs::default()
                    },
                    ..cc_bundle.clone()
                },
                ElementType::F32,
                &[&m0],
            );
            assert_eq!(l_out.len(), m0.len());
        }

        // Bitwidth-aware CountLeadingZeros, PopulationCount, shifts, and S64 Clamp/Select
        let s8_in = ElementType::S8.pack_from_i64_slice(&[1, 0]);
        let s8_clz = ElementType::S8.unpack_to_i64_vec(&SimulatedTpuHal::eval_unary_op(
            TpuOpCode::CountLeadingZeros,
            ElementType::S8,
            &s8_in,
        ));
        assert_eq!(s8_clz, vec![7, 8]);
        let s16_in = ElementType::S16.pack_from_i64_slice(&[1, 255]);
        let s16_clz = ElementType::S16.unpack_to_i64_vec(&SimulatedTpuHal::eval_unary_op(
            TpuOpCode::CountLeadingZeros,
            ElementType::S16,
            &s16_in,
        ));
        assert_eq!(s16_clz, vec![15, 8]);
        let s16_pop = ElementType::S16.unpack_to_i64_vec(&SimulatedTpuHal::eval_unary_op(
            TpuOpCode::PopulationCount,
            ElementType::S16,
            &s16_in,
        ));
        assert_eq!(s16_pop, vec![1, 8]);
        let s16_sh = ElementType::S16.pack_from_i64_slice(&[4, 4]);
        let s16_shl = ElementType::S16.unpack_to_i64_vec(&SimulatedTpuHal::eval_binary_op(
            TpuOpCode::ShiftLeft,
            ElementType::S16,
            &s16_in,
            &s16_sh,
        ));
        assert_eq!(s16_shl, vec![16, 4080]);
        let s16_lshr = ElementType::S16.unpack_to_i64_vec(&SimulatedTpuHal::eval_binary_op(
            TpuOpCode::ShiftRightLogical,
            ElementType::S16,
            &s16_in,
            &s16_sh,
        ));
        assert_eq!(s16_lshr, vec![0, 15]);

        let s64_c = ElementType::S64.pack_from_i64_slice(&[1, 0]);
        let s64_t = ElementType::S64.pack_from_i64_slice(&[0x7FFF_FFFF_FFFF_FFFE, 10]);
        let s64_f = ElementType::S64.pack_from_i64_slice(&[20, 0x7FFF_FFFF_FFFF_FFFD]);
        let selected_s64 = ElementType::S64.unpack_to_i64_vec(&SimulatedTpuHal::eval_bundle(
            TpuOpCode::Select,
            ElementType::S64,
            &[&s64_c, &s64_t, &s64_f],
        ));
        assert_eq!(
            selected_s64,
            vec![0x7FFF_FFFF_FFFF_FFFE, 0x7FFF_FFFF_FFFF_FFFD]
        );
        let s64_clamped = ElementType::S64.unpack_to_i64_vec(&SimulatedTpuHal::eval_bundle(
            TpuOpCode::Clamp,
            ElementType::S64,
            &[&s64_c, &s64_t, &s64_f],
        ));
        assert_eq!(s64_clamped, vec![20, 10]);

        // Sysfs PCIe scanner & resource BAR parser
        assert_eq!(
            PciDeviceEnumerator::parse_sysfs_resource_bar_size(
                "0x00000000fe000000 0x00000000fe0fffff 0x00040200\n0x0 0x0 0x0\n0x0000001000000000 0x00000010ffffffff 0x0014220c\n",
                0
            ),
            Some(0x0010_0000)
        );
        assert_eq!(
            PciDeviceEnumerator::parse_sysfs_resource_bar_size("0x0 0x0 0x0\n", 0),
            None
        );
        assert!(
            scan_sysfs_pci_tpu_devices(std::path::Path::new("/nonexistent/sysfs/path")).is_err()
        );
        let tmp_sysfs =
            std::env::temp_dir().join(format!("libtpu_pci_bdf_test_{}", std::process::id()));
        let bdf_dir = tmp_sysfs.join("0000:00:04.0");
        let _ = std::fs::create_dir_all(&bdf_dir);
        let _ = std::fs::write(bdf_dir.join("vendor"), "0x1ae0\n");
        let _ = std::fs::write(bdf_dir.join("device"), "0x006f\n");
        let _ = std::fs::write(
            bdf_dir.join("resource"),
            "0x00000000fe000000 0x00000000fe0fffff 0x00040200\n0x0 0x0 0x0\n0x0000001000000000 0x00000010ffffffff 0x0014220c\n",
        );
        let bdfs = scan_sysfs_pci_tpu_devices(&tmp_sysfs);
        assert!(bdfs.is_ok());
        assert_eq!(bdfs.unwrap_or_default().len(), 1);
        let disc = discover_pcie_tpu_devices_from_sysfs(&tmp_sysfs);
        assert_eq!(disc.len(), 1);
        let _ = std::fs::remove_dir_all(&tmp_sysfs);

        let db_ring = HardwareDoorbellRing::default();
        assert_eq!(db_ring.capacity, 64);
        let mut small_ring = HardwareDoorbellRing::new(1);
        assert_eq!(
            small_ring.enqueue_and_ring_doorbell(HbmDmaRingDescriptor::default()),
            1
        );
        assert_eq!(
            small_ring.enqueue_and_ring_doorbell(HbmDmaRingDescriptor::default()),
            2
        );
        assert_eq!(small_ring.process_completions_and_signal_eventfd(), 1);
        assert_eq!(small_ring.process_completions_and_signal_eventfd(), 0);
        assert_eq!(small_ring.clone(), small_ring);
        assert_ne!(format!("{small_ring:?}"), "");

        let mut vm_st = TpuCoreVmState::new();
        vm_st.vreg_words[0] = 0x1234_5678;
        assert_eq!(vm_st.inspect_vreg_2d_lanes(0)[0][0], 0x1234_5678);
        vm_st
            .vreg_tiles
            .insert(1, vec![0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]);
        let v1_lanes = vm_st.inspect_vreg_2d_lanes(1);
        assert_eq!(v1_lanes[0][0], 1);
        assert_eq!(v1_lanes[0][1], 2);
        assert_eq!(vm_st.inspect_gmr_tile(0), [0.0; 4]);

        let sort_keys = ElementType::F32.pack_from_f64_slice(&[3.0, 1.0, 4.0, 2.0]);
        let sort_vals = ElementType::F32.pack_from_f64_slice(&[30.0, 10.0, 40.0, 20.0]);
        let mut sort_attrs = InstructionAttrs {
            input_shapes: vec![Shape {
                element_type: ElementType::F32,
                dims: vec![2, 2],
                dynamic_dims: vec![false, false],
                layout: MemoryLayout::default_tpu_tiled(2),
            }],
            reduce_dims: vec![1],
            comparison_direction: String::from("GT"),
            tuple_index: 1,
            ..InstructionAttrs::default()
        };
        let sorted_v = SimulatedTpuHal::eval_sort_nd(
            ElementType::F32,
            ElementType::F32,
            &[&sort_keys, &sort_vals],
            &sort_attrs,
        );
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&sorted_v),
            vec![30.0, 10.0, 40.0, 20.0]
        );
        sort_attrs.comparison_direction = String::from("LT");
        sort_attrs.tuple_index = 0;
        let sorted_k = SimulatedTpuHal::eval_sort_nd(
            ElementType::F32,
            ElementType::F32,
            &[&sort_keys, &sort_vals],
            &sort_attrs,
        );
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&sorted_k),
            vec![1.0, 3.0, 2.0, 4.0]
        );

        let sas_op = ElementType::F32.pack_from_f64_slice(&[4.0, 1.0, 2.0, 8.0]);
        let sas_src = ElementType::F32.pack_from_f64_slice(&[5.0, 7.0]);
        let sas_init = ElementType::F32.pack_from_f64_slice(&[0.0]);
        for red_kind in [
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Max,
            CollectiveReductionOp::Product,
            CollectiveReductionOp::Sum,
        ] {
            let sas_out = SimulatedTpuHal::eval_select_and_scatter_nd(
                ElementType::F32,
                &[&sas_op, &sas_src, &sas_init],
                &InstructionAttrs {
                    reduce_op_kind: red_kind,
                    window_dims: vec![2],
                    window_strides: vec![2],
                    ..InstructionAttrs::default()
                },
            );
            assert_eq!(sas_out.len(), sas_op.len());
        }
    }
}
