//! Deterministic in-memory TPU hardware simulator ([`SimulatedTpuHal`]) implementing [`crate::hal::TpuHal`].

use super::{
    BAR0_CSR_HBM_DMA_RING_HEAD, BAR0_CSR_HBM_DMA_RING_TAIL, BAR0_CSR_SYNC_FLAG_BASE,
    BAR0_CSR_TENSORCORE_PC, BAR0_CSR_TENSORCORE_RUN_CONTROL, BAR0_CSR_TENSORCORE_STATUS,
    ChipRegisterMap, DeviceMemoryStats, DeviceSpec, EmulatedKernelDriverBackend,
    HbmRegionAllocator, MultiCoreChipVmState, TopologySpec, TpuCoreVmState, TpuHal,
};
use crate::compiler::CompiledProgram;
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::runtime_state::{SharedTpuRuntimeState, global_shared_runtime_state};
use crate::types::{
    ByteSize, ChipCoordinates, ChipId, CoreId, CoreIndex, DeviceAddress, DeviceId, HostId,
    LocalHardwareId, MemoryKind, ProcessIndex,
};
use parking_lot::{Mutex, MutexGuard};
use std::collections::BTreeMap;
use std::sync::Arc;
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
    /// Multi-core chip execution state machine (`tc_0`, `tc_1`, `scs_0..scs_3`).
    pub(super) chip_vm: MultiCoreChipVmState,
}

/// Monotonic counter assigning a unique namespace tag to each [`SimulatedTpuHal`] instance.
static NEXT_SIMULATED_HAL_INSTANCE_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Deterministic in-memory TPU hardware simulator implementing [`TpuHal`].
#[derive(Debug)]
pub struct SimulatedTpuHal {
    /// Unique simulator instance tag preventing cross-HAL address collisions in [`SharedTpuRuntimeState`].
    hal_instance_id: u64,
    /// Discovered topology specification.
    pub(super) topology: TopologySpec,
    /// Per-device mutable HBM and execution state protected by `parking_lot::Mutex`.
    pub(super) devices: Mutex<BTreeMap<DeviceId, SimulatedDeviceState>>,
    /// Atomic flag allowing deterministic simulation of a poisoned device map lock.
    pub(super) devices_poisoned: std::sync::atomic::AtomicBool,
    /// Emulated `PCIe` BAR0 MMIO CSR bank updated during DMA and `TensorCore` execution.
    pub mmio: EmulatedKernelDriverBackend,
    /// Synchronized runtime telemetry, live buffer registry, and `XSpace` trace state.
    runtime_state: Arc<SharedTpuRuntimeState>,
}

impl Drop for SimulatedTpuHal {
    fn drop(&mut self) {
        let tag = self.hal_instance_id.wrapping_shl(44);
        for (&dev_id, dev) in self.devices.get_mut() {
            for (&addr, bytes) in &dev.allocations {
                self.runtime_state.record_hbm_dealloc(
                    dev_id,
                    DeviceAddress(addr.0 ^ tag),
                    u64::try_from(bytes.len()).unwrap_or(0),
                );
            }
        }
    }
}

impl SimulatedTpuHal {
    /// Creates a new [`SimulatedTpuHal`] from a [`TpuInitConfig`] attached to [`global_shared_runtime_state`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration specifying generation and mesh bounds.
    #[must_use]
    pub fn new(config: &TpuInitConfig) -> Self {
        Self::with_runtime_state(config, Arc::clone(global_shared_runtime_state()))
    }

    /// Creates a new [`SimulatedTpuHal`] bound to an explicit [`SharedTpuRuntimeState`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration specifying generation and mesh bounds.
    /// * `runtime_state` - Shared telemetry and `XSpace` trace sink.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_runtime_state(
        config: &TpuInitConfig,
        runtime_state: Arc<SharedTpuRuntimeState>,
    ) -> Self {
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
                                                chip_vm: MultiCoreChipVmState::new(
                                                    generation,
                                                    generation.hbm_capacity_bytes().0,
                                                ),
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

        let hbm_cap_u64 = u64::try_from(generation.hbm_capacity_bytes().0).unwrap_or(0);
        for &dev_id in state_map.keys() {
            runtime_state.register_device_capacity(dev_id, hbm_cap_u64);
        }

        let topology = TopologySpec {
            platform_name: String::from("tpu"),
            platform_version: format!("libtpu-rs {} ({generation})", crate::LIBTPU_PYPI_VERSION),
            generation,
            chip_bounds,
            host_bounds: config.host_bounds,
            chips_per_host: config.chips_per_host(),
            devices: devices_spec,
        };

        Self {
            hal_instance_id: NEXT_SIMULATED_HAL_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            topology,
            devices: Mutex::new(state_map),
            devices_poisoned: std::sync::atomic::AtomicBool::new(false),
            mmio: EmulatedKernelDriverBackend::default(),
            runtime_state,
        }
    }

    /// Computes a [`SharedTpuRuntimeState`]-unique telemetry key for `address` on this HAL instance.
    ///
    /// # Arguments
    ///
    /// * `address` - Local HBM [`DeviceAddress`].
    #[must_use]
    const fn telemetry_address(&self, address: DeviceAddress) -> DeviceAddress {
        DeviceAddress(address.0 ^ self.hal_instance_id.wrapping_shl(44))
    }

    /// Returns a reference to the attached [`SharedTpuRuntimeState`].
    #[must_use]
    pub const fn runtime_state(&self) -> &Arc<SharedTpuRuntimeState> {
        &self.runtime_state
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
        self.runtime_state
            .register_device_capacity(device_id, u64::try_from(capacity.0).unwrap_or(0));
        Ok(())
    }

    /// Queries the byte length of an allocated HBM region at `address` on `device_id`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `address` - Allocated HBM [`DeviceAddress`].
    #[must_use]
    pub fn allocated_block_size(&self, device_id: DeviceId, address: DeviceAddress) -> usize {
        self.lock_devices()
            .ok()
            .and_then(|guard| {
                guard
                    .get(&device_id)
                    .and_then(|dev| dev.allocations.get(&address).map(Vec::len))
            })
            .unwrap_or(256)
    }

    /// Writes `data` into `address` on `device_id` without incrementing public H2D counters.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `address` - Destination HBM [`DeviceAddress`].
    /// * `data` - Source payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `device_id` or `address` is invalid or `data` exceeds capacity.
    fn write_hbm_raw(
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

    /// Reads `out.len()` bytes from `address` on `device_id` without incrementing public D2H counters.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Source logical TPU device.
    /// * `address` - Source HBM [`DeviceAddress`].
    /// * `out` - Destination host byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `device_id` or `address` is invalid or `out` exceeds buffer size.
    fn read_hbm_raw(
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
        let cap_u64 = u64::try_from(dev.capacity_bytes).unwrap_or(0);
        drop(guard);
        self.runtime_state.record_hbm_alloc(
            device_id,
            self.telemetry_address(addr),
            u64::try_from(alloc_size).unwrap_or(0),
            cap_u64,
        );
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
        let removed_len = removed.len();
        let _ = dev.region_allocator.deallocate(address);
        dev.allocation_kinds.remove(&address);
        dev.bytes_in_use = dev.bytes_in_use.saturating_sub(removed_len);
        drop(guard);
        self.runtime_state.record_hbm_dealloc(
            device_id,
            self.telemetry_address(address),
            u64::try_from(removed_len).unwrap_or(0),
        );
        Ok(())
    }

    fn write_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        data: &[u8],
    ) -> LibTpuResult<()> {
        self.write_hbm_raw(device_id, address, data)?;
        self.runtime_state
            .record_dma_h2d(u64::try_from(data.len()).unwrap_or(0));
        Ok(())
    }

    fn read_device_memory(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()> {
        self.read_hbm_raw(device_id, address, out)?;
        self.runtime_state
            .record_dma_d2h(u64::try_from(out.len()).unwrap_or(0));
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
        self.read_hbm_raw(src_device, src_addr, &mut temp)?;
        self.write_hbm_raw(dst_device, dst_addr, &temp)?;
        self.runtime_state
            .record_dma_d2d(u64::try_from(size.0).unwrap_or(0));
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
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

        let _ =
            dev.chip_vm
                .execute_spmd_and_sparse_program(program, &input_buffers, &input_buffers)?;
        let evaluated_outputs =
            dev.vm
                .execute_compiled_program(program, input_buffers, outputs.len());
        let vm_pc = dev.vm.pc;
        let vm_cycles = dev.vm.cycles_executed;

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
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_PC, u64::from(vm_pc));
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_RUN_CONTROL, 1);
        self.mmio.set_csr(BAR0_CSR_TENSORCORE_STATUS, 2);
        let sync = self.mmio.get_csr(BAR0_CSR_SYNC_FLAG_BASE);
        let next_sync =
            sync.wrapping_add(u64::try_from(program.instruction_bundles.len()).unwrap_or(1));
        self.mmio.set_csr(BAR0_CSR_SYNC_FLAG_BASE, next_sync);
        self.mmio.record_execution_telemetry(vm_cycles);
        let reg_map = ChipRegisterMap::for_generation(self.topology.generation);
        let imem_len = u64::try_from(program.instruction_bundles.len() * 64).unwrap_or(64);
        self.mmio
            .set_csr(reg_map.tc0_program_location_csr, imem_len);
        self.mmio.set_csr(reg_map.tc0_pc_csr, u64::from(vm_pc));
        self.mmio.set_csr(reg_map.tc0_run_status_csr, 2);
        self.mmio
            .set_csr(reg_map.tc0_vwait_sync_flag_state_csr, next_sync);
        // Megacore dual-TensorCore (TC0 + TC1) synchronization
        self.mmio
            .set_csr(reg_map.tc1_program_location_csr(), imem_len);
        self.mmio.set_csr(reg_map.tc1_pc_csr(), u64::from(vm_pc));
        self.mmio.set_csr(reg_map.tc1_run_status_csr(), 2);
        self.mmio.set_csr(reg_map.sc0_run_status_csr, 2);
        drop(guard);

        let bundles_retired = u64::try_from(program.instruction_bundles.len().max(1)).unwrap_or(1);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let flops = program.cost_analysis.flops.max(0.0) as u64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bytes_accessed = program.cost_analysis.bytes_accessed.max(0.0) as u64;
        let has_sc = program.instruction_bundles.iter().any(|b| {
            b.opcode == crate::compiler::TpuOpCode::CustomCall
                && (b.attrs.custom_call_target.contains("SparseCore")
                    || b.attrs.custom_call_target.contains("TpuEmbedding"))
        });
        let core_idx = CoreIndex(u8::try_from(device_id.0 % 2).unwrap_or(0));
        self.runtime_state.record_execution(
            device_id,
            core_idx,
            has_sc,
            &program.name,
            bundles_retired,
            vm_cycles,
            flops,
            bytes_accessed,
        );
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
        self.runtime_state.reset_device_peak_hbm(device_id);
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
