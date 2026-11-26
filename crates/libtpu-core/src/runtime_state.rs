//! Process-wide synchronized telemetry, HBM, callback, and `XSpace` trace state ([`SharedTpuRuntimeState`]).
//!
//! Unifies live per-device HBM memory accounting, DMA byte/event counters, compilation and
//! execution telemetry, per-core (`TensorCore` and `SparseCore`) program histories, registered
//! SDK debug callbacks, and active `XSpace` profiler trace spans across [`crate::pjrt::PjrtCoreClient`],
//! [`crate::hal::SimulatedTpuHal`], [`crate::driver::TpuCoreDriver`],
//! [`crate::driver::LibtpuSdkCoreClient`], [`crate::driver::TpuCoreProfiler`], and Python/C bindings.

use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{CoreIndex, DeviceAddress, DeviceId, TpuCoreTypeEnum};
use core::ffi::c_void;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// Maximum number of recent [`CoreProgramSummary`] records retained per `(DeviceId, CoreIndex)`.
const MAX_CORE_PROGRAM_SUMMARIES: usize = 16;

/// Maximum number of live [`RecordedTraceSpan`] records retained in the `XSpace` trace ring buffer.
const MAX_TRACE_RING_EVENTS: usize = 512;

/// Live HBM allocation and capacity snapshot for a single TPU device inside [`SharedTpuRuntimeState`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SharedDeviceHbmSnapshot {
    /// Bytes currently allocated in device HBM.
    pub bytes_in_use: u64,
    /// High-water mark of bytes simultaneously allocated in device HBM.
    pub peak_bytes_in_use: u64,
    /// Configured total HBM capacity in bytes for this device.
    pub capacity_bytes: u64,
    /// Cumulative number of HBM buffer allocations performed on this device.
    pub num_allocs: u64,
    /// Active live HBM buffer byte sizes indexed by [`DeviceAddress`].
    pub active_buffers: BTreeMap<DeviceAddress, u64>,
}

/// Summary of a queued or completed hardware program on a `TensorCore` or `SparseCore`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreProgramSummary {
    /// Target logical TPU [`DeviceId`].
    pub device_id: DeviceId,
    /// Core index within the TPU chip (`0`, `1`, ...).
    pub core_index: CoreIndex,
    /// Hardware core category (`TensorCore` or `SparseCore`).
    pub core_type: TpuCoreTypeEnum,
    /// Compiled module or kernel name.
    pub program_name: String,
    /// Monotonic execution launch identifier.
    pub launch_id: u64,
    /// Number of VLIW or `SparseCore` instruction bundles retired.
    pub bundles_retired: u64,
    /// Estimated floating-point operations retired by the launch.
    pub flops: u64,
    /// Simulated or hardware cycle count elapsed during the launch.
    pub cycles: u64,
    /// Whether execution has completed (`true`) or remains queued (`false`).
    pub completed: bool,
}

/// Registered `LibtpuSdk` debug callback descriptor stored in [`SharedTpuRuntimeState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugCallbackEntry {
    /// Unique positive callback registration handle (`>= 1`).
    pub callback_id: i64,
    /// Raw C/Python function pointer or callable token address.
    pub callback_addr: usize,
    /// Opaque caller-supplied `user_data` pointer address.
    pub user_data_addr: usize,
}

/// Target timeline `XPlane` category for a [`RecordedTraceSpan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePlaneKind {
    /// Host CPU runtime thread plane (`/host:CPU` / `Host Threads`).
    HostCpu,
    /// TPU device hardware plane (`/device:TPU:0`).
    TpuDevice,
}

/// Live profiler trace span recorded into [`SharedTpuRuntimeState`] while a profiler session is active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedTraceSpan {
    /// Target `XPlane` (`HostCpu` or `TpuDevice`).
    pub plane: TracePlaneKind,
    /// Timeline lane name (e.g., `"Host Threads"`, `"TensorCore 0"`, `"DMA Engine"`, `"ICI Router"`).
    pub lane_name: String,
    /// Operation or transfer display name.
    pub event_name: String,
    /// Canonical `XEventMetadata` identifier (`1` = HLO op, `2` = `SparseCore`, `3` = DMA, `4` = ICI).
    pub metadata_id: i64,
    /// Relative event offset in picoseconds within the session.
    pub offset_ps: i64,
    /// Event duration in picoseconds.
    pub duration_ps: i64,
    /// Bytes transferred or accessed during the event.
    pub bytes: u64,
    /// Floating-point operations retired during the event.
    pub flops: u64,
}

/// Callback invoker signature used by [`SharedTpuRuntimeState::trigger_debug_callbacks`].
pub type DebugCallbackInvoker = fn(DebugCallbackEntry, i32, &str);

/// Process-wide synchronized runtime telemetry, buffer registry, debug callback table, and `XSpace` ring.
#[derive(Debug)]
pub struct SharedTpuRuntimeState {
    /// Per-device HBM usage, peak usage, capacity, allocation count, and live buffer registry.
    devices_hbm: Mutex<BTreeMap<DeviceId, SharedDeviceHbmSnapshot>>,
    /// Cumulative host-to-device (`H2D`) DMA bytes transferred.
    dma_h2d_bytes: AtomicU64,
    /// Cumulative device-to-host (`D2H`) DMA bytes transferred.
    dma_d2h_bytes: AtomicU64,
    /// Cumulative device-to-device (`D2D`) DMA bytes transferred.
    dma_d2d_bytes: AtomicU64,
    /// Cumulative count of H2D, D2H, and D2D DMA transfer events.
    dma_transfer_events: AtomicU64,
    /// Cumulative count of HLO / `StableHLO` compilations completed.
    compile_count: AtomicU64,
    /// Cumulative wall-clock compilation latency in nanoseconds.
    compile_time_ns: AtomicU64,
    /// Cumulative count of TPU program executions completed.
    execute_count: AtomicU64,
    /// Cumulative simulated or hardware execution cycles across launches.
    execute_cycles: AtomicU64,
    /// Cumulative floating-point operations executed across launches.
    execute_flops: AtomicU64,
    /// Recent queued/completed program summaries keyed by `(DeviceId, CoreIndex)`.
    core_programs: Mutex<BTreeMap<(DeviceId, CoreIndex), Vec<CoreProgramSummary>>>,
    /// Registered SDK debug callbacks keyed by `callback_id` (`i64`).
    debug_callbacks: Mutex<BTreeMap<i64, DebugCallbackEntry>>,
    /// Monotonic sequence generator for `callback_id` (`>= 1`).
    next_callback_id: AtomicI64,
    /// Cumulative count of dispatched debug callback notifications.
    debug_events_triggered: AtomicU64,
    /// Reference count of currently active `TpuCoreProfiler` / `PLUGIN_Profiler` sessions.
    active_profiler_sessions: Mutex<u32>,
    /// Monotonic picosecond cursor assigning deterministic non-overlapping trace offsets.
    next_trace_offset_ps: AtomicI64,
    /// Ring buffer of live [`RecordedTraceSpan`] events captured during active profiler sessions.
    trace_events: Mutex<Vec<RecordedTraceSpan>>,
    /// Optional C-ABI [`ExternalRuntimeSyncVtable`] pointer bridging `libtpu.so` and `libtpu._C`/`sdk.so`.
    peer_sync_vtable: AtomicUsize,
}

impl Default for SharedTpuRuntimeState {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedTpuRuntimeState {
    /// Creates a new isolated [`SharedTpuRuntimeState`] with zeroed counters and empty registries.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            devices_hbm: Mutex::new(BTreeMap::new()),
            dma_h2d_bytes: AtomicU64::new(0),
            dma_d2h_bytes: AtomicU64::new(0),
            dma_d2d_bytes: AtomicU64::new(0),
            dma_transfer_events: AtomicU64::new(0),
            compile_count: AtomicU64::new(0),
            compile_time_ns: AtomicU64::new(0),
            execute_count: AtomicU64::new(0),
            execute_cycles: AtomicU64::new(0),
            execute_flops: AtomicU64::new(0),
            core_programs: Mutex::new(BTreeMap::new()),
            debug_callbacks: Mutex::new(BTreeMap::new()),
            next_callback_id: AtomicI64::new(1),
            debug_events_triggered: AtomicU64::new(0),
            active_profiler_sessions: Mutex::new(0),
            next_trace_offset_ps: AtomicI64::new(0),
            trace_events: Mutex::new(Vec::new()),
            peer_sync_vtable: AtomicUsize::new(0),
        }
    }

    /// Binds an [`ExternalRuntimeSyncVtable`] exported by another in-process `cdylib` (`libtpu.so`),
    /// ignoring null or self-referential `ctx` pointers.
    ///
    /// # Arguments
    ///
    /// * `vtable` - Pointer to a `'static`-lived [`ExternalRuntimeSyncVtable`].
    ///
    /// # Safety
    ///
    /// When non-null, `vtable` must point to a valid `'static` [`ExternalRuntimeSyncVtable`].
    pub unsafe fn bind_peer_sync_vtable(&self, vtable: *const ExternalRuntimeSyncVtable) {
        if vtable.is_null() {
            self.peer_sync_vtable.store(0, Ordering::Release);
            return;
        }
        // SAFETY: Caller guarantees `vtable` is valid for `'static`.
        let vt = unsafe { &*vtable };
        let self_addr = core::ptr::from_ref(self) as usize;
        if (vt.ctx as usize) != self_addr {
            self.peer_sync_vtable
                .store(vtable as usize, Ordering::Release);
        }
    }

    /// Returns the bound peer [`ExternalRuntimeSyncVtable`], if configured.
    #[must_use]
    fn peer_vtable(&self) -> Option<&'static ExternalRuntimeSyncVtable> {
        let addr = self.peer_sync_vtable.load(Ordering::Acquire);
        if addr == 0 {
            None
        } else {
            // SAFETY: Non-zero addresses stored via `bind_peer_sync_vtable` point to `'static` vtables.
            Some(unsafe { &*(addr as *const ExternalRuntimeSyncVtable) })
        }
    }

    /// Registers or updates the HBM capacity of `device_id` without discarding active allocations.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU [`DeviceId`].
    /// * `capacity_bytes` - Device HBM capacity in bytes.
    pub fn register_device_capacity(&self, device_id: DeviceId, capacity_bytes: u64) {
        self.devices_hbm
            .lock()
            .entry(device_id)
            .or_default()
            .capacity_bytes = capacity_bytes;
    }

    /// Records a live HBM buffer allocation on `device_id` at `address` of `size_bytes` bytes.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU [`DeviceId`].
    /// * `address` - Allocated HBM [`DeviceAddress`].
    /// * `size_bytes` - Size of the allocated region in bytes.
    /// * `capacity_bytes` - Total HBM capacity of `device_id` in bytes.
    pub fn record_hbm_alloc(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        size_bytes: u64,
        capacity_bytes: u64,
    ) {
        let mut guard = self.devices_hbm.lock();
        let entry = guard.entry(device_id).or_default();
        if capacity_bytes > 0 {
            entry.capacity_bytes = capacity_bytes;
        }
        if let Some(prev) = entry.active_buffers.insert(address, size_bytes) {
            entry.bytes_in_use = entry.bytes_in_use.saturating_sub(prev);
        }
        entry.bytes_in_use = entry.bytes_in_use.saturating_add(size_bytes);
        entry.peak_bytes_in_use = entry.peak_bytes_in_use.max(entry.bytes_in_use);
        entry.num_allocs = entry.num_allocs.saturating_add(1);
        drop(guard);
        if let Some(vt) = self.peer_vtable() {
            (vt.on_hbm_alloc)(vt.ctx, device_id.0, address.0, size_bytes, capacity_bytes);
        }
    }

    /// Records an HBM buffer deallocation on `device_id` at `address`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU [`DeviceId`].
    /// * `address` - Freed HBM [`DeviceAddress`].
    /// * `fallback_bytes` - Fallback byte count subtracted when `address` was allocated prior to attach.
    pub fn record_hbm_dealloc(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        fallback_bytes: u64,
    ) {
        let mut guard = self.devices_hbm.lock();
        if let Some(entry) = guard.get_mut(&device_id) {
            let freed = entry
                .active_buffers
                .remove(&address)
                .unwrap_or(fallback_bytes);
            entry.bytes_in_use = entry.bytes_in_use.saturating_sub(freed);
        }
        drop(guard);
        if let Some(vt) = self.peer_vtable() {
            (vt.on_hbm_dealloc)(vt.ctx, device_id.0, address.0, fallback_bytes);
        }
    }

    /// Resets `peak_bytes_in_use` on `device_id` down to its current `bytes_in_use`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU [`DeviceId`].
    pub fn reset_device_peak_hbm(&self, device_id: DeviceId) {
        let mut guard = self.devices_hbm.lock();
        if let Some(entry) = guard.get_mut(&device_id) {
            entry.peak_bytes_in_use = entry.bytes_in_use;
        }
    }

    /// Returns a cloned [`SharedDeviceHbmSnapshot`] for `device_id`, if registered.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU [`DeviceId`].
    #[must_use]
    pub fn device_hbm_snapshot(&self, device_id: DeviceId) -> Option<SharedDeviceHbmSnapshot> {
        self.all_device_hbm_snapshots().remove(&device_id)
    }

    /// Returns per-device `(bytes_in_use, peak_bytes_in_use, capacity_bytes, num_allocs)` snapshots
    /// in deterministic `DeviceId` order.
    #[must_use]
    pub fn all_device_hbm_snapshots(&self) -> BTreeMap<DeviceId, SharedDeviceHbmSnapshot> {
        if let Some(vt) = self.peer_vtable() {
            let mut rows = [[0_u64; 5]; 64];
            // SAFETY: `rows` points to 64 initialized `[u64; 5]` elements.
            let count =
                unsafe { (vt.read_device_hbm_table)(vt.ctx, rows.as_mut_ptr(), rows.len()) };
            let mut merged = BTreeMap::new();
            for row in rows.iter().take(count) {
                merged.insert(
                    DeviceId(u32::try_from(row[0]).unwrap_or(0)),
                    SharedDeviceHbmSnapshot {
                        bytes_in_use: row[1],
                        peak_bytes_in_use: row[2],
                        capacity_bytes: row[3],
                        num_allocs: row[4],
                        active_buffers: BTreeMap::new(),
                    },
                );
            }
            return merged;
        }
        self.devices_hbm.lock().clone()
    }

    /// Aggregates `(bytes_in_use, peak_bytes_in_use, capacity_bytes, num_allocs, active_buffers)`
    /// across all tracked TPU devices.
    #[must_use]
    pub fn aggregate_hbm_totals(&self) -> (u64, u64, u64, u64, usize) {
        if self.peer_vtable().is_some() {
            let snaps = self.all_device_hbm_snapshots();
            let mut in_use = 0_u64;
            let mut peak = 0_u64;
            let mut cap = 0_u64;
            let mut allocs = 0_u64;
            for snap in snaps.values() {
                in_use = in_use.saturating_add(snap.bytes_in_use);
                peak = peak.saturating_add(snap.peak_bytes_in_use);
                cap = cap.saturating_add(snap.capacity_bytes);
                allocs = allocs.saturating_add(snap.num_allocs);
            }
            return (in_use, peak, cap, allocs, snaps.len());
        }
        let guard = self.devices_hbm.lock();
        let mut in_use = 0_u64;
        let mut peak = 0_u64;
        let mut cap = 0_u64;
        let mut allocs = 0_u64;
        let mut active = 0_usize;
        for snap in guard.values() {
            in_use = in_use.saturating_add(snap.bytes_in_use);
            peak = peak.saturating_add(snap.peak_bytes_in_use);
            cap = cap.saturating_add(snap.capacity_bytes);
            allocs = allocs.saturating_add(snap.num_allocs);
            active = active.saturating_add(snap.active_buffers.len());
        }
        drop(guard);
        (in_use, peak, cap, allocs, active)
    }

    /// Records a host-to-device (`H2D`) DMA transfer and emits a profiler span when active.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of payload bytes transferred from host to HBM.
    pub fn record_dma_h2d(&self, bytes: u64) {
        self.dma_h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.dma_transfer_events.fetch_add(1, Ordering::Relaxed);
        if self.is_profiling_active() {
            self.push_trace_span(
                TracePlaneKind::TpuDevice,
                "DMA Engine",
                "dma_h2d",
                3,
                250_000,
                bytes,
                0,
            );
        }
        if let Some(vt) = self.peer_vtable() {
            (vt.on_dma)(vt.ctx, 0, bytes);
        }
    }

    /// Records a device-to-host (`D2H`) DMA transfer and emits a profiler span when active.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of payload bytes transferred from HBM to host.
    pub fn record_dma_d2h(&self, bytes: u64) {
        self.dma_d2h_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.dma_transfer_events.fetch_add(1, Ordering::Relaxed);
        if self.is_profiling_active() {
            self.push_trace_span(
                TracePlaneKind::TpuDevice,
                "DMA Engine",
                "dma_d2h",
                3,
                250_000,
                bytes,
                0,
            );
        }
        if let Some(vt) = self.peer_vtable() {
            (vt.on_dma)(vt.ctx, 1, bytes);
        }
    }

    /// Records a device-to-device (`D2D`) or cross-host DMA transfer and emits a profiler span when active.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of payload bytes copied between device buffers.
    pub fn record_dma_d2d(&self, bytes: u64) {
        self.dma_d2d_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.dma_transfer_events.fetch_add(1, Ordering::Relaxed);
        if self.is_profiling_active() {
            self.push_trace_span(
                TracePlaneKind::TpuDevice,
                "DMA Engine",
                "dma_d2d",
                3,
                200_000,
                bytes,
                0,
            );
        }
        if let Some(vt) = self.peer_vtable() {
            (vt.on_dma)(vt.ctx, 2, bytes);
        }
    }

    /// Returns `(h2d_bytes, d2h_bytes, d2d_bytes, transfer_events)` DMA telemetry counters.
    #[must_use]
    pub fn dma_counters(&self) -> (u64, u64, u64, u64) {
        if let Some(vt) = self.peer_vtable() {
            let mut raw = [0_u64; 9];
            (vt.read_counters)(vt.ctx, &raw mut raw);
            return (raw[0], raw[1], raw[2], raw[3]);
        }
        (
            self.dma_h2d_bytes.load(Ordering::Relaxed),
            self.dma_d2h_bytes.load(Ordering::Relaxed),
            self.dma_d2d_bytes.load(Ordering::Relaxed),
            self.dma_transfer_events.load(Ordering::Relaxed),
        )
    }

    /// Records a completed HLO / `StableHLO` compilation and emits a host profiler span when active.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the compiled module.
    /// * `elapsed_ns` - Compilation latency in nanoseconds.
    pub fn record_compilation(&self, module_name: &str, elapsed_ns: u64) {
        self.compile_count.fetch_add(1, Ordering::Relaxed);
        self.compile_time_ns
            .fetch_add(elapsed_ns.max(1), Ordering::Relaxed);
        if self.is_profiling_active() {
            let dur_ps =
                i64::try_from(elapsed_ns.max(1).saturating_mul(1_000)).unwrap_or(1_000_000);
            self.push_trace_span(
                TracePlaneKind::HostCpu,
                "Host Threads",
                &format!("PJRT_Compile({module_name})"),
                1,
                dur_ps,
                0,
                0,
            );
        }
        if let Some(vt) = self.peer_vtable() {
            // SAFETY: `module_name` UTF-8 bytes are valid for the duration of the synchronous call.
            unsafe {
                (vt.on_compile)(vt.ctx, module_name.as_ptr(), module_name.len(), elapsed_ns);
            }
        }
    }

    /// Returns `(compile_count, cumulative_compile_time_ns)`.
    #[must_use]
    pub fn compilation_counters(&self) -> (u64, u64) {
        if let Some(vt) = self.peer_vtable() {
            let mut raw = [0_u64; 9];
            (vt.read_counters)(vt.ctx, &raw mut raw);
            return (raw[4], raw[5]);
        }
        (
            self.compile_count.load(Ordering::Relaxed),
            self.compile_time_ns.load(Ordering::Relaxed),
        )
    }

    /// Records a completed TPU kernel execution, updates per-core program history, and emits
    /// `Host Threads` + `/device:TPU:0` trace spans when a profiler session is active.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU [`DeviceId`].
    /// * `core_index` - Core index on the chip.
    /// * `has_sparse_core` - Whether the launch also retired `SparseCore` bundles.
    /// * `program_name` - Name of the executed [`crate::compiler::CompiledProgram`].
    /// * `bundles_retired` - Number of VLIW bundles retired.
    /// * `cycles` - Simulated or hardware cycles elapsed.
    /// * `flops` - Estimated floating-point operations retired.
    /// * `bytes_accessed` - Total HBM/VMEM bytes read and written.
    #[allow(clippy::too_many_arguments)]
    pub fn record_execution(
        &self,
        device_id: DeviceId,
        core_index: CoreIndex,
        has_sparse_core: bool,
        program_name: &str,
        bundles_retired: u64,
        cycles: u64,
        flops: u64,
        bytes_accessed: u64,
    ) {
        let launch_id = self.execute_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.execute_cycles
            .fetch_add(cycles.max(1), Ordering::Relaxed);
        self.execute_flops.fetch_add(flops, Ordering::Relaxed);

        {
            let mut guard = self.core_programs.lock();
            let tc_list = guard.entry((device_id, core_index)).or_default();
            if tc_list.len() >= MAX_CORE_PROGRAM_SUMMARIES {
                tc_list.remove(0);
            }
            tc_list.push(CoreProgramSummary {
                device_id,
                core_index,
                core_type: TpuCoreTypeEnum::TensorCore,
                program_name: String::from(program_name),
                launch_id,
                bundles_retired,
                flops,
                cycles: cycles.max(1),
                completed: true,
            });
            if has_sparse_core {
                let sc_idx = CoreIndex(core_index.0.saturating_add(2));
                let sc_list = guard.entry((device_id, sc_idx)).or_default();
                if sc_list.len() >= MAX_CORE_PROGRAM_SUMMARIES {
                    sc_list.remove(0);
                }
                sc_list.push(CoreProgramSummary {
                    device_id,
                    core_index: sc_idx,
                    core_type: TpuCoreTypeEnum::SparseCore,
                    program_name: String::from(program_name),
                    launch_id,
                    bundles_retired: 1,
                    flops: 0,
                    cycles: cycles.max(1),
                    completed: true,
                });
            }
            drop(guard);
        }

        if self.is_profiling_active() {
            let dur_ps = i64::try_from(cycles.max(1).saturating_mul(1_000)).unwrap_or(1_000_000);
            self.push_trace_span(
                TracePlaneKind::HostCpu,
                "Host Threads",
                &format!("PJRT_Execute({program_name})"),
                1,
                dur_ps / 2 + 1,
                bytes_accessed,
                0,
            );
            self.push_trace_span(
                TracePlaneKind::TpuDevice,
                &format!("TensorCore {}", core_index.0),
                program_name,
                1,
                dur_ps,
                bytes_accessed,
                flops,
            );
            if has_sparse_core {
                self.push_trace_span(
                    TracePlaneKind::TpuDevice,
                    "SparseCore 0",
                    program_name,
                    2,
                    dur_ps / 2 + 1,
                    bytes_accessed / 2,
                    0,
                );
            }
        }
        if let Some(vt) = self.peer_vtable() {
            // SAFETY: `program_name` UTF-8 bytes are valid for the duration of the synchronous call.
            unsafe {
                (vt.on_execute)(
                    vt.ctx,
                    device_id.0,
                    core_index.0,
                    has_sparse_core,
                    program_name.as_ptr(),
                    program_name.len(),
                    bundles_retired,
                    cycles,
                    flops,
                    bytes_accessed,
                );
            }
        }
    }

    /// Records an ICI / Megascale collective operation and emits an `ICI Router` trace span when active.
    ///
    /// # Arguments
    ///
    /// * `op_name` - Collective operation name (`"all_reduce"`, `"reduce_scatter"`, etc.).
    /// * `bytes` - Payload byte count routed across ranks.
    pub fn record_collective(&self, op_name: &str, bytes: u64) {
        self.dma_d2d_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.dma_transfer_events.fetch_add(1, Ordering::Relaxed);
        if self.is_profiling_active() {
            self.push_trace_span(
                TracePlaneKind::TpuDevice,
                "ICI Router",
                op_name,
                4,
                200_000,
                bytes,
                0,
            );
        }
    }

    /// Returns `(execute_count, execute_cycles, execute_flops)`.
    #[must_use]
    pub fn execution_counters(&self) -> (u64, u64, u64) {
        if let Some(vt) = self.peer_vtable() {
            let mut raw = [0_u64; 9];
            (vt.read_counters)(vt.ctx, &raw mut raw);
            return (raw[6], raw[7], raw[8]);
        }
        (
            self.execute_count.load(Ordering::Relaxed),
            self.execute_cycles.load(Ordering::Relaxed),
            self.execute_flops.load(Ordering::Relaxed),
        )
    }

    /// Returns all recorded [`CoreProgramSummary`] entries across cores in deterministic order.
    #[must_use]
    pub fn recent_core_programs(&self) -> Vec<CoreProgramSummary> {
        self.core_programs
            .lock()
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect()
    }

    /// Registers a `LibtpuSdk` debug callback and returns its unique positive `i64` handle.
    ///
    /// # Arguments
    ///
    /// * `callback_addr` - Raw function pointer or callable token address.
    /// * `user_data_addr` - Opaque caller-supplied `user_data` pointer address.
    #[must_use]
    pub fn register_debug_callback(&self, callback_addr: usize, user_data_addr: usize) -> i64 {
        let callback_id = self.next_callback_id.fetch_add(1, Ordering::Relaxed);
        self.debug_callbacks.lock().insert(
            callback_id,
            DebugCallbackEntry {
                callback_id,
                callback_addr,
                user_data_addr,
            },
        );
        callback_id
    }

    /// Unregisters a previously registered `LibtpuSdk` debug callback by `callback_id`.
    ///
    /// # Arguments
    ///
    /// * `callback_id` - Callback registration handle returned by [`Self::register_debug_callback`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `callback_id` is not currently registered.
    pub fn unregister_debug_callback(&self, callback_id: i64) -> LibTpuResult<()> {
        self.debug_callbacks
            .lock()
            .remove(&callback_id)
            .map(|_| ())
            .ok_or_else(|| LibTpuError::InvalidArgument {
                parameter: "callback_handle",
                reason: format!("Debug callback handle {callback_id} is not registered"),
            })
    }

    /// Returns all currently registered [`DebugCallbackEntry`] records sorted by `callback_id`.
    #[must_use]
    pub fn registered_debug_callbacks(&self) -> Vec<DebugCallbackEntry> {
        self.debug_callbacks.lock().values().copied().collect()
    }

    /// Invokes all registered debug callbacks with `(event_code, message)` using `invoker` (when provided)
    /// and returns the number of callbacks notified.
    ///
    /// # Arguments
    ///
    /// * `event_code` - Numeric diagnostic/alert event code.
    /// * `message` - Human-readable diagnostic message.
    /// * `invoker` - Optional callback trampoline invoked outside the internal mutex lock.
    #[must_use]
    pub fn trigger_debug_callbacks(
        &self,
        event_code: i32,
        message: &str,
        invoker: Option<DebugCallbackInvoker>,
    ) -> usize {
        let snapshot: Vec<DebugCallbackEntry> =
            self.debug_callbacks.lock().values().copied().collect();
        self.debug_events_triggered.fetch_add(
            u64::try_from(snapshot.len()).unwrap_or(0),
            Ordering::Relaxed,
        );
        if let Some(invoke_fn) = invoker {
            for entry in &snapshot {
                invoke_fn(*entry, event_code, message);
            }
        }
        snapshot.len()
    }

    /// Returns the cumulative count of dispatched debug callback notifications.
    #[must_use]
    pub fn debug_events_triggered(&self) -> u64 {
        self.debug_events_triggered.load(Ordering::Relaxed)
    }

    /// Marks the start of a `TpuCoreProfiler` or `PLUGIN_Profiler` session so live trace spans are captured.
    pub fn begin_profiler_session(&self) {
        let mut guard = self.active_profiler_sessions.lock();
        *guard = guard.saturating_add(1);
    }

    /// Marks the end of an active `TpuCoreProfiler` or `PLUGIN_Profiler` session.
    pub fn end_profiler_session(&self) {
        let mut guard = self.active_profiler_sessions.lock();
        *guard = guard.saturating_sub(1);
    }

    /// Returns `true` when at least one profiler session is currently active.
    #[must_use]
    pub fn is_profiling_active(&self) -> bool {
        *self.active_profiler_sessions.lock() > 0
    }

    /// Appends a [`RecordedTraceSpan`] directly into the trace ring buffer.
    ///
    /// # Arguments
    ///
    /// * `plane` - Target [`TracePlaneKind`].
    /// * `lane_name` - Timeline lane display name.
    /// * `event_name` - Event operation name.
    /// * `metadata_id` - `XEventMetadata` identifier (`1..=4`).
    /// * `duration_ps` - Duration in picoseconds.
    /// * `bytes` - Bytes accessed or transferred.
    /// * `flops` - Floating-point operations retired.
    #[allow(clippy::too_many_arguments)]
    pub fn push_trace_span(
        &self,
        plane: TracePlaneKind,
        lane_name: &str,
        event_name: &str,
        metadata_id: i64,
        duration_ps: i64,
        bytes: u64,
        flops: u64,
    ) {
        let clean_dur = duration_ps.max(1_000);
        let offset_ps = self
            .next_trace_offset_ps
            .fetch_add(clean_dur, Ordering::Relaxed);
        let mut guard = self.trace_events.lock();
        if guard.len() >= MAX_TRACE_RING_EVENTS {
            guard.remove(0);
        }
        guard.push(RecordedTraceSpan {
            plane,
            lane_name: String::from(lane_name),
            event_name: String::from(event_name),
            metadata_id,
            offset_ps,
            duration_ps: clean_dur,
            bytes,
            flops,
        });
    }

    /// Returns a cloned snapshot of all recorded [`RecordedTraceSpan`] events in chronological order.
    #[must_use]
    pub fn snapshot_trace_spans(&self) -> Vec<RecordedTraceSpan> {
        self.trace_events.lock().clone()
    }

    /// Clears all recorded [`RecordedTraceSpan`] events from the trace ring buffer.
    pub fn clear_trace_spans(&self) {
        self.trace_events.lock().clear();
        self.next_trace_offset_ps.store(0, Ordering::Relaxed);
    }
}

/// `#[repr(C)]` synchronization vtable unifying telemetry across `libtpu.so` and `libtpu._C`/`sdk.so`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExternalRuntimeSyncVtable {
    /// Opaque pointer to the target [`SharedTpuRuntimeState`].
    pub ctx: *const c_void,
    /// Records an HBM allocation on `(device_id, address, size_bytes, capacity_bytes)`.
    pub on_hbm_alloc: extern "C" fn(*const c_void, u32, u64, u64, u64),
    /// Records an HBM deallocation on `(device_id, address, fallback_bytes)`.
    pub on_hbm_dealloc: extern "C" fn(*const c_void, u32, u64, u64),
    /// Records a DMA transfer (`direction`: `0`=H2D, `1`=D2H, `2`=D2D).
    pub on_dma: extern "C" fn(*const c_void, u8, u64),
    /// Records a completed compilation (`module_ptr`, `module_len`, `elapsed_ns`).
    pub on_compile: unsafe extern "C" fn(*const c_void, *const u8, usize, u64),
    /// Records a completed kernel execution.
    pub on_execute:
        unsafe extern "C" fn(*const c_void, u32, u8, bool, *const u8, usize, u64, u64, u64, u64),
    /// Reads `[h2d, d2h, d2d, dma_ev, comp_cnt, comp_ns, exec_cnt, exec_cyc, exec_flops]`.
    pub read_counters: extern "C" fn(*const c_void, *mut [u64; 9]),
    /// Reads up to `max_rows` rows of `[dev_id, bytes_in_use, peak_bytes, cap_bytes, num_allocs]`.
    pub read_device_hbm_table: unsafe extern "C" fn(*const c_void, *mut [u64; 5], usize) -> usize,
}

// SAFETY: `ExternalRuntimeSyncVtable` wraps a `'static` `SharedTpuRuntimeState` which is `Send + Sync`.
unsafe impl Send for ExternalRuntimeSyncVtable {}
// SAFETY: `ExternalRuntimeSyncVtable` wraps a `'static` `SharedTpuRuntimeState` which is `Send + Sync`.
unsafe impl Sync for ExternalRuntimeSyncVtable {}

/// C-ABI callback forwarding HBM allocations into `ctx`'s [`SharedTpuRuntimeState`].
///
/// # Arguments
///
/// * `ctx` - Pointer to target [`SharedTpuRuntimeState`].
/// * `dev` - Logical device ID.
/// * `addr` - Device HBM address.
/// * `size` - Allocated byte size.
/// * `cap` - Total device HBM capacity in bytes.
extern "C" fn sync_on_hbm_alloc(ctx: *const c_void, dev: u32, addr: u64, size: u64, cap: u64) {
    // SAFETY: `ctx` is initialized from a `'static` `&SharedTpuRuntimeState`.
    let st = unsafe { &*(ctx.cast::<SharedTpuRuntimeState>()) };
    st.record_hbm_alloc(DeviceId(dev), DeviceAddress(addr), size, cap);
}

/// C-ABI callback forwarding HBM deallocations into `ctx`'s [`SharedTpuRuntimeState`].
///
/// # Arguments
///
/// * `ctx` - Pointer to target [`SharedTpuRuntimeState`].
/// * `dev` - Logical device ID.
/// * `addr` - Device HBM address.
/// * `fallback` - Fallback byte count.
extern "C" fn sync_on_hbm_dealloc(ctx: *const c_void, dev: u32, addr: u64, fallback: u64) {
    // SAFETY: `ctx` is initialized from a `'static` `&SharedTpuRuntimeState`.
    let st = unsafe { &*(ctx.cast::<SharedTpuRuntimeState>()) };
    st.record_hbm_dealloc(DeviceId(dev), DeviceAddress(addr), fallback);
}

/// C-ABI callback forwarding DMA byte counters into `ctx`'s [`SharedTpuRuntimeState`].
///
/// # Arguments
///
/// * `ctx` - Pointer to target [`SharedTpuRuntimeState`].
/// * `direction` - `0` for H2D, `1` for D2H, otherwise D2D.
/// * `bytes` - Transferred byte count.
extern "C" fn sync_on_dma(ctx: *const c_void, direction: u8, bytes: u64) {
    // SAFETY: `ctx` is initialized from a `'static` `&SharedTpuRuntimeState`.
    let st = unsafe { &*(ctx.cast::<SharedTpuRuntimeState>()) };
    match direction {
        0 => st.record_dma_h2d(bytes),
        1 => st.record_dma_d2h(bytes),
        _ => st.record_dma_d2d(bytes),
    }
}

/// C-ABI callback forwarding compilation telemetry into `ctx`'s [`SharedTpuRuntimeState`].
///
/// # Arguments
///
/// * `ctx` - Pointer to target [`SharedTpuRuntimeState`].
/// * `name_ptr` - UTF-8 module name pointer.
/// * `name_len` - Byte length of `name_ptr`.
/// * `elapsed_ns` - Compilation latency in nanoseconds.
///
/// # Safety
///
/// `name_ptr` must be valid for `name_len` bytes.
unsafe extern "C" fn sync_on_compile(
    ctx: *const c_void,
    name_ptr: *const u8,
    name_len: usize,
    elapsed_ns: u64,
) {
    // SAFETY: Caller guarantees `ctx` and `(name_ptr, name_len)` are valid.
    let (st, bytes) = unsafe {
        (
            &*(ctx.cast::<SharedTpuRuntimeState>()),
            core::slice::from_raw_parts(name_ptr, name_len),
        )
    };
    let name = core::str::from_utf8(bytes).unwrap_or("module");
    st.record_compilation(name, elapsed_ns);
}

/// C-ABI callback forwarding kernel execution telemetry into `ctx`'s [`SharedTpuRuntimeState`].
///
/// # Arguments
///
/// * `ctx` - Pointer to target [`SharedTpuRuntimeState`].
/// * `dev` - Logical device ID.
/// * `core` - Core index on chip.
/// * `has_sc` - Whether `SparseCore` bundles retired.
/// * `name_ptr` - UTF-8 program name pointer.
/// * `name_len` - Byte length of `name_ptr`.
/// * `bundles` - Retired VLIW bundles.
/// * `cycles` - Simulated/hardware cycles.
/// * `flops` - Retired FLOPs.
/// * `bytes` - HBM/VMEM bytes accessed.
///
/// # Safety
///
/// `name_ptr` must be valid for `name_len` bytes.
unsafe extern "C" fn sync_on_execute(
    ctx: *const c_void,
    dev: u32,
    core: u8,
    has_sc: bool,
    name_ptr: *const u8,
    name_len: usize,
    bundles: u64,
    cycles: u64,
    flops: u64,
    bytes: u64,
) {
    // SAFETY: Caller guarantees `ctx` and `(name_ptr, name_len)` are valid.
    let (st, raw_name) = unsafe {
        (
            &*(ctx.cast::<SharedTpuRuntimeState>()),
            core::slice::from_raw_parts(name_ptr, name_len),
        )
    };
    let name = core::str::from_utf8(raw_name).unwrap_or("program");
    st.record_execution(
        DeviceId(dev),
        CoreIndex(core),
        has_sc,
        name,
        bundles,
        cycles,
        flops,
        bytes,
    );
}

/// C-ABI callback reading the 9 scalar telemetry counters from `ctx`'s [`SharedTpuRuntimeState`].
///
/// # Arguments
///
/// * `ctx` - Pointer to target [`SharedTpuRuntimeState`].
/// * `out` - Output `[u64; 9]` array pointer.
extern "C" fn sync_read_counters(ctx: *const c_void, out: *mut [u64; 9]) {
    // SAFETY: `ctx` and `out` are valid non-null pointers.
    let (st, dst) = unsafe { (&*(ctx.cast::<SharedTpuRuntimeState>()), &mut *out) };
    let (h2d, d2h, d2d, ev) = st.dma_counters();
    let (cc, cn) = st.compilation_counters();
    let (ec, ecyc, eflops) = st.execution_counters();
    *dst = [h2d, d2h, d2d, ev, cc, cn, ec, ecyc, eflops];
}

/// C-ABI callback reading per-device HBM snapshots from `ctx`'s [`SharedTpuRuntimeState`].
///
/// # Arguments
///
/// * `ctx` - Pointer to target [`SharedTpuRuntimeState`].
/// * `out_rows` - Output buffer of `[u64; 5]` rows.
/// * `max_rows` - Capacity of `out_rows` in rows.
///
/// # Safety
///
/// `out_rows` must point to at least `max_rows` writable `[u64; 5]` elements.
unsafe extern "C" fn sync_read_device_hbm_table(
    ctx: *const c_void,
    out_rows: *mut [u64; 5],
    max_rows: usize,
) -> usize {
    // SAFETY: Caller guarantees `ctx` and `(out_rows, max_rows)` are valid.
    let (st, slice) = unsafe {
        (
            &*(ctx.cast::<SharedTpuRuntimeState>()),
            core::slice::from_raw_parts_mut(out_rows, max_rows),
        )
    };
    let guard = st.devices_hbm.lock();
    let mut written = 0_usize;
    for (dst, (&dev_id, snap)) in slice.iter_mut().zip(guard.iter()) {
        *dst = [
            u64::from(dev_id.0),
            snap.bytes_in_use,
            snap.peak_bytes_in_use,
            snap.capacity_bytes,
            snap.num_allocs,
        ];
        written += 1;
    }
    drop(guard);
    written
}

/// Builds an [`ExternalRuntimeSyncVtable`] targeting `state`.
///
/// # Arguments
///
/// * `state` - `'static` reference to the target [`SharedTpuRuntimeState`].
#[must_use]
pub const fn build_runtime_sync_vtable(
    state: &'static SharedTpuRuntimeState,
) -> ExternalRuntimeSyncVtable {
    ExternalRuntimeSyncVtable {
        ctx: core::ptr::from_ref(state).cast::<c_void>(),
        on_hbm_alloc: sync_on_hbm_alloc,
        on_hbm_dealloc: sync_on_hbm_dealloc,
        on_dma: sync_on_dma,
        on_compile: sync_on_compile,
        on_execute: sync_on_execute,
        read_counters: sync_read_counters,
        read_device_hbm_table: sync_read_device_hbm_table,
    }
}

/// Process-wide singleton [`SharedTpuRuntimeState`] shared across PJRT, HAL, Driver, SDK, and `PyO3`.
static GLOBAL_SHARED_RUNTIME_STATE: OnceLock<Arc<SharedTpuRuntimeState>> = OnceLock::new();

/// Process-wide singleton [`ExternalRuntimeSyncVtable`] for [`global_shared_runtime_state`].
static GLOBAL_RUNTIME_SYNC_VTABLE: OnceLock<ExternalRuntimeSyncVtable> = OnceLock::new();

/// Returns a reference to the process-wide [`Arc<SharedTpuRuntimeState>`].
#[must_use]
pub fn global_shared_runtime_state() -> &'static Arc<SharedTpuRuntimeState> {
    GLOBAL_SHARED_RUNTIME_STATE.get_or_init(|| Arc::new(SharedTpuRuntimeState::new()))
}

/// Returns a `'static` pointer to the [`ExternalRuntimeSyncVtable`] for [`global_shared_runtime_state`].
#[must_use]
pub fn global_runtime_sync_vtable() -> &'static ExternalRuntimeSyncVtable {
    GLOBAL_RUNTIME_SYNC_VTABLE
        .get_or_init(|| build_runtime_sync_vtable(global_shared_runtime_state().as_ref()))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Counter tracking debug callback invocations during tests.
    static INVOKED_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Test debug callback invoker incrementing [`INVOKED_COUNT`].
    fn count_invoker(entry: DebugCallbackEntry, code: i32, msg: &str) {
        if entry.callback_id > 0 && code == 7 && !msg.is_empty() {
            INVOKED_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn test_shared_runtime_state_hbm_dma_compile_execute_callbacks_and_traces() -> LibTpuResult<()>
    {
        let state = SharedTpuRuntimeState::default();
        let dev0 = DeviceId(0);
        let dev1 = DeviceId(1);
        state.register_device_capacity(dev0, 1_048_576);
        assert_eq!(state.device_hbm_snapshot(DeviceId(99)), None);

        // Allocate, overwrite same address, allocate second address, reset peak, dealloc known & fallback
        state.record_hbm_alloc(dev0, DeviceAddress(0x1000), 4096, 1_048_576);
        state.record_hbm_alloc(dev0, DeviceAddress(0x1000), 8192, 0);
        state.record_hbm_alloc(dev0, DeviceAddress(0x2000), 2048, 1_048_576);
        state.record_hbm_alloc(dev1, DeviceAddress(0x3000), 1024, 524_288);
        let (in_use, peak, cap, allocs, active) = state.aggregate_hbm_totals();
        assert_eq!(in_use, 8192 + 2048 + 1024);
        assert_eq!(peak, 8192 + 2048 + 1024);
        assert_eq!(cap, 1_048_576 + 524_288);
        assert_eq!(allocs, 4);
        assert_eq!(active, 3);
        assert_eq!(state.all_device_hbm_snapshots().len(), 2);

        state.record_hbm_dealloc(dev0, DeviceAddress(0x2000), 0);
        state.record_hbm_dealloc(dev0, DeviceAddress(0x9999), 512);
        state.record_hbm_dealloc(DeviceId(77), DeviceAddress(0x1000), 256);
        state.reset_device_peak_hbm(dev0);
        state.reset_device_peak_hbm(DeviceId(77));
        let snap0 = state
            .device_hbm_snapshot(dev0)
            .ok_or(LibTpuError::DeviceNotFound { device_id: 0 })?;
        assert_eq!(snap0.bytes_in_use, 8192 - 512);
        assert_eq!(snap0.peak_bytes_in_use, snap0.bytes_in_use);

        // Inactive profiling: DMA, compile, execute, collective do not push trace spans
        assert!(!state.is_profiling_active());
        state.record_dma_h2d(128);
        state.record_dma_d2h(64);
        state.record_dma_d2d(32);
        state.record_compilation("inactive_mod", 500);
        state.record_execution(dev0, CoreIndex(0), false, "inactive_exec", 2, 10, 256, 512);
        state.record_collective("all_reduce", 64);
        assert_eq!(state.snapshot_trace_spans(), [] as [RecordedTraceSpan; 0]);
        assert_eq!(state.dma_counters(), (128, 64, 96, 4));
        assert_eq!(state.compilation_counters(), (1, 500));
        assert_eq!(state.execution_counters(), (1, 10, 256));

        // Active profiling + ring caps
        state.begin_profiler_session();
        assert!(state.is_profiling_active());
        state.record_dma_h2d(256);
        state.record_dma_d2h(256);
        state.record_dma_d2d(256);
        state.record_compilation("active_mod", 1_500);
        state.record_collective("reduce_scatter", 128);
        state.record_execution(dev1, CoreIndex(1), false, "tc_only_step", 2, 10, 512, 1024);
        for step in 0..(MAX_CORE_PROGRAM_SUMMARIES + 2) {
            state.record_execution(
                dev0,
                CoreIndex(0),
                true,
                &format!("step_{step}"),
                4,
                20,
                1024,
                4096,
            );
        }
        let progs = state.recent_core_programs();
        assert_eq!(progs.len(), MAX_CORE_PROGRAM_SUMMARIES * 2 + 1);

        for idx in 0..(MAX_TRACE_RING_EVENTS + 4) {
            state.push_trace_span(
                TracePlaneKind::TpuDevice,
                "TensorCore 0",
                &format!("ev_{idx}"),
                1,
                1_000,
                64,
                128,
            );
        }
        assert_eq!(state.snapshot_trace_spans().len(), MAX_TRACE_RING_EVENTS);
        state.clear_trace_spans();
        assert_eq!(state.snapshot_trace_spans(), [] as [RecordedTraceSpan; 0]);
        state.end_profiler_session();
        state.end_profiler_session();
        assert!(!state.is_profiling_active());

        let cb1 = state.register_debug_callback(0x1111, 0x2222);
        let cb2 = state.register_debug_callback(0x3333, 0x4444);
        assert_eq!(state.registered_debug_callbacks().len(), 2);
        assert_eq!(state.trigger_debug_callbacks(7, "stall_check", None), 2);
        assert_eq!(
            state.trigger_debug_callbacks(7, "stall_check", Some(count_invoker)),
            2
        );
        assert_eq!(INVOKED_COUNT.load(Ordering::Relaxed), 2);
        assert_eq!(state.debug_events_triggered(), 4);
        state.unregister_debug_callback(cb1)?;
        assert!(state.unregister_debug_callback(cb1).is_err());
        state.unregister_debug_callback(cb2)?;
        assert_eq!(
            state.registered_debug_callbacks(),
            [] as [DebugCallbackEntry; 0]
        );

        let global = global_shared_runtime_state();
        assert!(Arc::strong_count(global) >= 1);
        Ok(())
    }

    #[test]
    fn test_external_runtime_sync_vtable() {
        let dev0 = DeviceId(0);
        let global = global_shared_runtime_state();
        let global_vt = global_runtime_sync_vtable();
        // SAFETY: `global_vt` is a `'static` vtable; self-bind is ignored.
        unsafe {
            global.bind_peer_sync_vtable(core::ptr::from_ref(global_vt));
        }

        let primary: &'static SharedTpuRuntimeState =
            Box::leak(Box::new(SharedTpuRuntimeState::new()));
        let primary_vt: &'static ExternalRuntimeSyncVtable =
            Box::leak(Box::new(build_runtime_sync_vtable(primary)));
        let secondary = SharedTpuRuntimeState::new();
        // SAFETY: `primary_vt` is leaked for `'static` lifetime.
        unsafe {
            secondary.bind_peer_sync_vtable(core::ptr::from_ref(primary_vt));
        }
        secondary.record_hbm_alloc(dev0, DeviceAddress(0x5000), 4096, 65536);
        assert_eq!(primary.aggregate_hbm_totals().0, 4096);
        assert_eq!(secondary.aggregate_hbm_totals().0, 4096);
        assert!(secondary.device_hbm_snapshot(dev0).is_some());
        secondary.record_hbm_dealloc(dev0, DeviceAddress(0x5000), 0);
        assert_eq!(primary.aggregate_hbm_totals().0, 0);
        secondary.record_dma_h2d(100);
        secondary.record_dma_d2h(200);
        secondary.record_dma_d2d(300);
        secondary.record_compilation("bridged_mod", 2_000);
        secondary.record_execution(dev0, CoreIndex(0), false, "bridged_prog", 1, 5, 64, 128);
        assert_eq!(secondary.dma_counters(), (100, 200, 300, 3));
        assert_eq!(secondary.compilation_counters(), (1, 2_000));
        assert_eq!(secondary.execution_counters(), (1, 5, 64));
        // SAFETY: Null pointer unbinds the peer vtable cleanly.
        unsafe {
            secondary.bind_peer_sync_vtable(core::ptr::null());
        }
    }
}
