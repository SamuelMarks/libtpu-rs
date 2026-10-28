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

pub use self::linux::{HalBackend, LinuxDeviceHal, create_default_hal};
pub use self::memory::{
    DeviceMemoryStats, HbmRegionAllocator, MappedBarAperture, MmioAperture, VmemRegionAllocator,
};
pub use self::pci::{
    DeviceSpec, DiscoveredPciTpuDevice, KernelDriverMode, PciDeviceEnumerator, PciSysfsScanner,
    TopologySpec,
};
pub use self::registers::{
    BAR0_CSR_DUTY_CYCLE_PERCENT, BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL,
    BAR0_CSR_ICI_ROUTER_LINK_STATE, BAR0_CSR_POWER_MILLIWATTS, BAR0_CSR_SPARSECORE_RUN_CONTROL,
    BAR0_CSR_SYNC_FLAG_BASE, BAR0_CSR_TENSORCORE_PC, BAR0_CSR_TENSORCORE_RUN_CONTROL,
    BAR0_CSR_TENSORCORE_STATUS, BAR0_CSR_TENSORCORE_SYNC_FLAG, BAR0_CSR_THERMAL_MILLICELSIUS,
    ChipRegisterMap, DmaRingDescriptor, GASKET_IOCTL_CLEAR_EVENTFD,
    GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS, GASKET_IOCTL_MAP_BUFFER, GASKET_IOCTL_MAP_DMABUF,
    GASKET_IOCTL_NUMBER_PAGE_TABLES, GASKET_IOCTL_PAGE_TABLE_SIZE,
    GASKET_IOCTL_PARTITION_PAGE_TABLE, GASKET_IOCTL_RESET, GASKET_IOCTL_SET_EVENTFD,
    GASKET_IOCTL_SIMPLE_PAGE_TABLE_SIZE, GASKET_IOCTL_UNMAP_BUFFER, GasketInterruptEventFd,
    GasketPageTableDmaBuf, GasketPageTableIoctl, HbmDmaRingDescriptor, IciLinkStateFlags,
    TensorCoreControlFlags, TensorCoreStatusFlags, VFIO_CHECK_EXTENSION, VFIO_DEVICE_GET_INFO,
    VFIO_DEVICE_GET_IRQ_INFO, VFIO_DEVICE_GET_REGION_INFO, VFIO_DEVICE_RESET, VFIO_DEVICE_SET_IRQS,
    VFIO_DEVICE_SET_IRQS_IOCTL, VFIO_GET_API_VERSION, VFIO_GROUP_GET_DEVICE_FD,
    VFIO_GROUP_GET_STATUS, VFIO_GROUP_SET_CONTAINER, VFIO_IOMMU_MAP_DMA, VFIO_IOMMU_MAP_DMA_IOCTL,
    VFIO_IOMMU_UNMAP_DMA, VFIO_IOMMU_UNMAP_DMA_IOCTL, VFIO_SET_IOMMU, VfioDmaMapFlags,
    VfioDmaMapIoctl, VfioDmaUnmapIoctl, VfioIrqSetFlags, VfioIrqSetIoctl,
};
pub use self::simulated::SimulatedTpuHal;
pub use self::syscall::{
    EmulatedKernelDriverBackend, IoctlPayload, KernelDriverOps, MockKernelSyscalls,
    POSIX_FIOCLEX_IOCTL, PosixKernelDriverBackend, RealLinuxSyscalls, SimulatedKernelDriverBackend,
    SyscallProvider, VirtualTpuKernelDevice,
};
pub use self::vm::{DeepSeaVm, TpuCoreVmState};

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
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines
)]
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
                assert_eq!(hal.mmio.dma_ring_len(), 3);
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
            base_vaddr: 0,
            size: ByteSize(4096),
        };
        assert_ne!(format!("{null_aperture:?}"), "");
        assert!(null_aperture.read_csr_u64(MmioOffset(0)).is_err());
        assert!(null_aperture.write_csr_u64(MmioOffset(0), 1).is_err());
        assert!(null_aperture.read_csr_u32(MmioOffset(0)).is_err());
        assert!(null_aperture.write_csr_u32(MmioOffset(0), 1).is_err());

        let efd = posix.create_eventfd();
        assert!(efd.0 >= 0);
        assert!(posix.close_device(efd.0).is_ok());

        assert!(posix.allocate_anon_bar0_mmap(ByteSize(0)).is_err());
        for mut mmap_buf in posix.allocate_anon_bar0_mmap(ByteSize(4096)) {
            assert_eq!(mmap_buf.len(), 4096);
            for first_byte in mmap_buf.first_mut() {
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
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:04.0/vendor"),
            "0x1ae0
",
        );
        let _ = std::fs::write(
            sysfs_dir.join("0000:00:04.0/device"),
            "0x006f
",
        );
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

        let scanned = PciDeviceEnumerator::scan_sysfs(&sysfs_dir, &dev_dir);
        assert_eq!(scanned.len(), 2);
        let root_scanned = PciSysfsScanner::discover_from_sysfs_root(&sysfs_dir);
        assert_eq!(root_scanned.len(), 2);
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
        }

        let mut vm = TpuCoreVmState::default();
        assert!(
            PosixKernelDriverBackend
                .mmap_bar_aperture(FileDescriptor(-1), ByteSize(0))
                .is_err()
        );
        assert_eq!(vm.cycles_executed, 0);
        assert!(vm.wait_sync_flag(0, 1, 1).is_err());
        for prog in CompiledProgram::compile("mlir", b"dot_general", b"") {
            vm.execute_vliw_stream(&prog.encode_vliw_program());
            if let Some(ds_exec) = prog.to_tpu_executable_proto().ds_executable {
                vm.execute_deepsea_proto(&ds_exec);
            }
        }
        for prog_exp in CompiledProgram::compile("mlir", b"exp_f32", b"") {
            let mut bundles = prog_exp.encode_vliw_program();
            for (idx, sc_op) in [2_u8, 3, 4, 5].into_iter().enumerate() {
                if let Some(b0) = bundles.first().copied() {
                    let mut b_mod = b0;
                    b_mod.scalar_slot.opcode = sc_op;
                    b_mod.scalar_slot.dst_sreg = u8::try_from(idx + 1).unwrap_or(1);
                    bundles.push(b_mod);
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
    }
}
