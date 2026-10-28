//! Deterministic in-memory TPU hardware simulator ([`SimulatedTpuHal`]) implementing [`crate::hal::TpuHal`].

use super::{
    BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL, BAR0_CSR_SYNC_FLAG_BASE,
    BAR0_CSR_TENSORCORE_PC, BAR0_CSR_TENSORCORE_RUN_CONTROL, BAR0_CSR_TENSORCORE_STATUS,
    ChipRegisterMap, DeviceMemoryStats, DeviceSpec, EmulatedKernelDriverBackend,
    HbmRegionAllocator, TopologySpec, TpuCoreVmState, TpuHal,
};
use crate::compiler::CompiledProgram;
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{
    ByteSize, ChipCoordinates, ChipId, CoreId, DeviceAddress, DeviceId, HostId, LocalHardwareId,
    MemoryKind, ProcessIndex,
};
use parking_lot::{Mutex, MutexGuard};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

/// Internal per-device state for [`SimulatedTpuHal`].
#[derive(Debug)]
pub(super) struct SimulatedDeviceState {
    /// Page-aligned (`4096` B) HBM region allocator with free-block coalescing.
    pub(super) region_allocator: HbmRegionAllocator,
    /// Active HBM allocations mapped by [`DeviceAddress`].
    pub(super) allocations: BTreeMap<DeviceAddress, Vec<u8>>,
    /// Memory space kind associated with each [`DeviceAddress`].
    pub(super) allocation_kinds: BTreeMap<DeviceAddress, MemoryKind>,
    /// Bytes currently allocated.
    pub(super) bytes_in_use: usize,
    /// Peak bytes allocated.
    pub(super) peak_bytes_in_use: usize,
    /// Total allocation count.
    pub(super) num_allocs: i64,
    /// Largest single allocation in bytes.
    pub(super) largest_alloc_size: usize,
    /// Configurable HBM capacity limit in bytes.
    pub(super) capacity_bytes: usize,
    /// Optional poison reason set by `poison_device`.
    pub(super) poison_reason: Option<String>,
    /// Architectural `TensorCore` VLIW execution state machine.
    pub(super) vm: TpuCoreVmState,
}

/// Deterministic in-memory TPU hardware simulator implementing [`TpuHal`].
#[derive(Debug)]
pub struct SimulatedTpuHal {
    /// Discovered topology specification.
    pub(super) topology: TopologySpec,
    /// Per-device mutable HBM and execution state protected by `parking_lot::Mutex`.
    pub(super) devices: Mutex<BTreeMap<DeviceId, SimulatedDeviceState>>,
    /// Atomic flag allowing deterministic simulation of a poisoned device map lock.
    pub(super) devices_poisoned: std::sync::atomic::AtomicBool,
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
                                                vm: TpuCoreVmState::new(),
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
            devices_poisoned: std::sync::atomic::AtomicBool::new(false),
            mmio: EmulatedKernelDriverBackend::default(),
        }
    }

    /// Returns a cloned copy of the discovered [`TopologySpec`].
    pub fn topology_spec(&self) -> TopologySpec {
        self.topology.clone()
    }

    /// Acquires the internal `parking_lot::Mutex` device map guard or returns [`LibTpuError::LockPoisoned`].
    pub(super) fn lock_devices(
        &self,
    ) -> LibTpuResult<MutexGuard<'_, BTreeMap<DeviceId, SimulatedDeviceState>>> {
        if self.devices_poisoned.load(Ordering::Acquire) {
            return Err(LibTpuError::LockPoisoned {
                resource: "devices",
            });
        }
        Ok(self.devices.lock())
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
        let mut input_buffers: Vec<Vec<u8>> = Vec::with_capacity(inputs.len());
        for (idx, &in_addr) in inputs.iter().enumerate() {
            let buf =
                dev.allocations
                    .get(&in_addr)
                    .ok_or_else(|| LibTpuError::ExecutionFailed {
                        device_id: device_id.0,
                        reason: format!("Input {idx} address {in_addr} not found"),
                    })?;
            input_buffers.push(buf.clone());
        }

        let evaluated_outputs =
            dev.vm
                .execute_compiled_program(program, input_buffers, outputs.len());

        for (&out_addr, src_bytes) in outputs.iter().zip(evaluated_outputs.iter()) {
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
        let next_sync =
            sync.wrapping_add(u64::try_from(program.instruction_bundles.len()).unwrap_or(1));
        self.mmio.set_csr(BAR0_CSR_SYNC_FLAG_BASE, next_sync);
        let reg_map = ChipRegisterMap::for_generation(self.topology.generation);
        self.mmio.set_csr(reg_map.tc0_pc_csr, 0);
        self.mmio.set_csr(reg_map.tc0_run_status_csr, 2);
        self.mmio
            .set_csr(reg_map.tc0_vwait_sync_flag_state_csr, next_sync);
        self.mmio.set_csr(reg_map.sc0_run_status_csr, 2);
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
