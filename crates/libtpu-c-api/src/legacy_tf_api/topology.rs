//! Legacy TensorFlow TPU topology and mesh state C API entrypoints.
//!
//! Exports `TpuUtil_*`, `TpuTopology_*`, `TpuCoreLocation_*`, `TpuHostLocation_*`, and `TpuMeshState_*`.

use super::*;

/// Builds a [`TpuTopology`] from a [`TpuInitConfig`].
///
/// # Arguments
///
/// * `cfg` - Runtime configuration describing generation and host/chip bounds.
fn build_topology_from_config(cfg: &TpuInitConfig) -> TpuTopology {
    let hal = SimulatedTpuHal::new(cfg);
    let spec = hal.topology_spec();
    let core_locations = spec
        .devices
        .iter()
        .map(|d| TpuCoreLocation {
            core_id: i32::try_from(d.device_id.0).unwrap_or(0),
            index_on_chip: i32::try_from(d.core_on_chip.0).unwrap_or(0),
            chip_x: i32::try_from(d.chip_coords.x).unwrap_or(0),
            chip_y: i32::try_from(d.chip_coords.y).unwrap_or(0),
            chip_z: i32::try_from(d.chip_coords.z).unwrap_or(0),
        })
        .collect();
    TpuTopology {
        hal,
        core_locations,
    }
}

/// Global mutable `TpuTopology` pointer synchronized with `global_init_config`.
static GLOBAL_TOPOLOGY: OnceLock<parking_lot::Mutex<usize>> = OnceLock::new();

/// Returns a reference to the global [`TpuTopology`] pointer mutex.
fn global_topology_mutex() -> &'static parking_lot::Mutex<usize> {
    GLOBAL_TOPOLOGY.get_or_init(|| {
        let cfg = libtpu_core::config::global_init_config();
        let ptr_val = Box::into_raw(Box::new(build_topology_from_config(&cfg))) as usize;
        parking_lot::Mutex::new(ptr_val)
    })
}

/// Rebuilds the global [`TpuTopology`] from [`libtpu_core::config::global_init_config`].
pub(super) fn refresh_global_topology() {
    let cfg = libtpu_core::config::global_init_config();
    let ptr_val = Box::into_raw(Box::new(build_topology_from_config(&cfg))) as usize;
    *global_topology_mutex().lock() = ptr_val;
}

/// Returns a static reference to the current global [`TpuTopology`].
pub(super) fn global_topology() -> &'static TpuTopology {
    let addr = *global_topology_mutex().lock();
    // SAFETY: `addr` points to a heap-allocated `TpuTopology` with static lifetime.
    unsafe { &*(addr as *const TpuTopology) }
}

/// Resolves `topo` to either `&*topo` (if non-null) or `global_topology()`.
///
/// # Arguments
///
/// * `topo` - Optional [`TpuTopology`] pointer.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[inline]
unsafe fn resolve_topology<'a>(topo: *const TpuTopology) -> &'a TpuTopology {
    if topo.is_null() {
        global_topology()
    } else {
        // SAFETY: `topo` is non-null.
        unsafe { &*topo }
    }
}

/// Returns a pointer to the singleton [`TpuTopology`] (`TpuUtil_GetTopologyPtr`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuUtil_GetTopologyPtr() -> *const TpuTopology {
    ptr::from_ref(global_topology())
}

/// Returns the XLA minor-dimension alignment pad size (`TpuUtil_GetXlaPadSizeFromTpuTopology`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuUtil_GetXlaPadSizeFromTpuTopology() -> i32 {
    global_topology()
        .hal
        .topology_spec()
        .generation
        .xla_pad_size()
}

/// Returns the number of logical devices per host (`TpuTopology_LogicalDevicesPerHost`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_LogicalDevicesPerHost(
    topo: *const TpuTopology,
    core_type: i32,
) -> i32 {
    // SAFETY: `topo` is null or valid.
    unsafe { TpuTopology_NumCores(topo, core_type) }
}

/// Returns the number of logical devices per chip (`TpuTopology_LogicalDevicesPerChip`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_LogicalDevicesPerChip(
    topo: *const TpuTopology,
    core_type: i32,
) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let tc_per_chip =
        i32::try_from(t.hal.topology_spec().generation.logical_devices_per_chip()).unwrap_or(1);
    if core_type == 1 {
        let sc_per_dev =
            i32::try_from(t.hal.topology_spec().generation.sparse_cores_per_device()).unwrap_or(0);
        tc_per_chip.saturating_mul(sc_per_dev.max(1))
    } else {
        tc_per_chip
    }
}

/// Returns the total host count in the topology (`TpuTopology_HostCount`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_HostCount(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let bounds = t.hal.topology_spec().host_bounds;
    i32::try_from((bounds.x * bounds.y * bounds.z).max(1)).unwrap_or(1)
}

/// Returns the number of chips per host (`TpuTopology_ChipsPerHost`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_ChipsPerHost(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chips_per_host).unwrap_or(1)
}

/// Returns the X mesh bound (`TpuTopology_ChipBounds_X`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_ChipBounds_X(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chip_bounds.x).unwrap_or(1)
}

/// Returns the Y mesh bound (`TpuTopology_ChipBounds_Y`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_ChipBounds_Y(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chip_bounds.y).unwrap_or(1)
}

/// Returns the Z mesh bound (`TpuTopology_ChipBounds_Z`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_ChipBounds_Z(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chip_bounds.z).unwrap_or(1)
}

/// Checks whether chip `(x, y, z)` exists in the topology (`TpuTopology_HasChip`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `x` - Chip X coordinate.
/// * `y` - Chip Y coordinate.
/// * `z` - Chip Z coordinate.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_HasChip(
    topo: *const TpuTopology,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let bounds = t.hal.topology_spec().chip_bounds;
    let Ok(ux) = u32::try_from(x) else {
        return false;
    };
    let Ok(uy) = u32::try_from(y) else {
        return false;
    };
    let Ok(uz) = u32::try_from(z) else {
        return false;
    };
    ux < bounds.x && uy < bounds.y && uz < bounds.z
}

/// Looks up a [`TpuCoreLocation`] by core ID (`TpuTopology_CoreForId`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `_core_type` - Core type discriminant.
/// * `id` - Core ID.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_CoreForId(
    topo: *const TpuTopology,
    _core_type: i32,
    id: i32,
) -> *const TpuCoreLocation {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let idx = usize::try_from(id).unwrap_or(usize::MAX);
    t.core_locations.get(idx).map_or(ptr::null(), ptr::from_ref)
}

/// Looks up a [`TpuCoreLocation`] by coordinates and index (`TpuTopology_Core`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `_core_type` - Core type discriminant.
/// * `x` - Chip X coordinate.
/// * `y` - Chip Y coordinate.
/// * `z` - Chip Z coordinate.
/// * `index` - Core index on chip.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_Core(
    topo: *const TpuTopology,
    _core_type: i32,
    x: i32,
    y: i32,
    z: i32,
    index: i32,
) -> *const TpuCoreLocation {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    t.core_locations
        .iter()
        .find(|c| (c.chip_x, c.chip_y, c.chip_z, c.index_on_chip) == (x, y, z, index))
        .map_or(ptr::null(), ptr::from_ref)
}

/// Returns the total number of cores in the topology (`TpuTopology_NumCores`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_NumCores(topo: *const TpuTopology, core_type: i32) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let tc_count = i32::try_from(t.core_locations.len()).unwrap_or(0);
    if core_type == 1 {
        let sc_per_dev =
            i32::try_from(t.hal.topology_spec().generation.sparse_cores_per_device()).unwrap_or(0);
        tc_count.saturating_mul(sc_per_dev.max(1))
    } else {
        tc_count
    }
}

/// Copies pointers to all [`TpuCoreLocation`] entries into `cores_out` (`TpuTopology_Cores`).
///
/// # Arguments
///
/// * `topo` - Topology handle.
/// * `_core_type` - Core type enum.
/// * `cores_out` - Output array of `*const TpuCoreLocation`.
///
/// # Safety
///
/// `cores_out` must be null or point to an array of at least `TpuTopology_NumCores` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_Cores(
    topo: *const TpuTopology,
    _core_type: i32,
    cores_out: *mut *const TpuCoreLocation,
) {
    if cores_out.is_null() {
        return;
    }
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    for (idx, loc) in t.core_locations.iter().enumerate() {
        // SAFETY: Caller guarantees capacity of `cores_out`.
        unsafe {
            *cores_out.add(idx) = ptr::from_ref(loc);
        }
    }
}

/// Returns host ID for host coordinates (`TpuTopology_IdForHost`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `x` - Host X coordinate.
/// * `y` - Host Y coordinate.
/// * `z` - Host Z coordinate.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_IdForHost(
    topo: *const TpuTopology,
    x: i32,
    y: i32,
    z: i32,
) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let hb = t.hal.topology_spec().host_bounds;
    let hx = i32::try_from(hb.x.max(1)).unwrap_or(1);
    let hy = i32::try_from(hb.y.max(1)).unwrap_or(1);
    x.max(0) + y.max(0) * hx + z.max(0) * hx * hy
}

/// Returns the `TpuVersionEnum` integer of the topology (`TpuTopology_Version`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_Version(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    t.hal.topology_spec().generation as i32
}

/// Queries available core count (`TpuTopology_AvailableCoreCount`).
///
/// # Arguments
///
/// * `_mesh_state` - Optional [`TpuMeshState`] pointer.
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuTopology_AvailableCoreCount(
    _mesh_state: *const TpuMeshState,
    core_type: i32,
) -> i32 {
    // SAFETY: Passing null uses `global_topology()`.
    unsafe { TpuTopology_NumCores(ptr::null(), core_type) }
}

/// Queries available cores per chip (`TpuTopology_AvailableCoresPerChip`).
///
/// # Arguments
///
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuTopology_AvailableCoresPerChip(core_type: i32) -> i32 {
    // SAFETY: Passing null uses `global_topology()`.
    unsafe { TpuTopology_LogicalDevicesPerChip(ptr::null(), core_type) }
}

/// Queries `SparseCores` per logical device (`TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice`).
///
/// # Arguments
///
/// * `_core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice(_core_type: i32) -> i32 {
    i32::try_from(
        global_topology()
            .hal
            .topology_spec()
            .generation
            .sparse_cores_per_device(),
    )
    .unwrap_or(0)
}

/// Writes the `(x, y, z)` chip coordinates of `core` (`TpuCoreLocation_ChipCoordinates`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
/// * `x_out` - Output X coordinate pointer.
/// * `y_out` - Output Y coordinate pointer.
/// * `z_out` - Output Z coordinate pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuCoreLocation_ChipCoordinates(
    core: *const TpuCoreLocation,
    x_out: *mut i32,
    y_out: *mut i32,
    z_out: *mut i32,
) {
    if any_null(&[
        core as usize,
        x_out as usize,
        y_out as usize,
        z_out as usize,
    ]) {
        return;
    }
    // SAFETY: Validated non-null above.
    unsafe {
        *x_out = (*core).chip_x;
        *y_out = (*core).chip_y;
        *z_out = (*core).chip_z;
    }
}

/// Writes the `(x, y, z)` host coordinates of `core` (`TpuCoreLocation_HostCoordinates`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
/// * `x_out` - Output X coordinate pointer.
/// * `y_out` - Output Y coordinate pointer.
/// * `z_out` - Output Z coordinate pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuCoreLocation_HostCoordinates(
    core: *const TpuCoreLocation,
    x_out: *mut i32,
    y_out: *mut i32,
    z_out: *mut i32,
) {
    if any_null(&[
        core as usize,
        x_out as usize,
        y_out as usize,
        z_out as usize,
    ]) {
        return;
    }
    let hb = global_topology().hal.topology_spec().host_bounds;
    // SAFETY: Validated non-null above.
    unsafe {
        *x_out = (*core).chip_x % i32::try_from(hb.x.max(1)).unwrap_or(1);
        *y_out = (*core).chip_y % i32::try_from(hb.y.max(1)).unwrap_or(1);
        *z_out = (*core).chip_z % i32::try_from(hb.z.max(1)).unwrap_or(1);
    }
}

/// Returns the core index on chip (`TpuCoreLocation_Index`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
///
/// # Safety
///
/// `core` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuCoreLocation_Index(core: *const TpuCoreLocation) -> i32 {
    if core.is_null() {
        return -1;
    }
    // SAFETY: `core` is non-null.
    unsafe { (*core).index_on_chip }
}

/// Returns the global core ID (`TpuCoreLocation_Id`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
///
/// # Safety
///
/// `core` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuCoreLocation_Id(core: *const TpuCoreLocation) -> i32 {
    if core.is_null() {
        return -1;
    }
    // SAFETY: `core` is non-null.
    unsafe { (*core).core_id }
}

/// Returns the host ID of a host location (`TpuHostLocation_Id`).
///
/// # Arguments
///
/// * `host` - Pointer to [`TpuHostLocation`] (or null).
///
/// # Safety
///
/// `host` must be null or a valid pointer to [`TpuHostLocation`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuHostLocation_Id(host: *const TpuHostLocation) -> i32 {
    if host.is_null() {
        0
    } else {
        // SAFETY: `host` is non-null.
        unsafe { (*host).host_id }
    }
}

/// Returns the number of cores on a host location (`TpuHostLocation_NumCores`).
///
/// # Arguments
///
/// * `host` - Pointer to [`TpuHostLocation`] (or null).
/// * `core_type` - Core type discriminant.
///
/// # Safety
///
/// `host` must be null or a valid pointer to [`TpuHostLocation`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuHostLocation_NumCores(
    host: *const TpuHostLocation,
    core_type: i32,
) -> i32 {
    if !host.is_null() {
        // SAFETY: `host` is non-null.
        let n = unsafe { (*host).num_cores };
        if n > 0 {
            return n;
        }
    }
    // SAFETY: Passing null uses `global_topology()`.
    unsafe { TpuTopology_NumCores(ptr::null(), core_type) }
}

/// Copies core pointers on a host location (`TpuHostLocation_Cores`).
///
/// # Arguments
///
/// * `_host` - Host location handle.
/// * `core_type` - Core type enum.
/// * `cores_out` - Output array of `*const TpuCoreLocation`.
///
/// # Safety
///
/// `cores_out` must be null or point to an array of sufficient size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuHostLocation_Cores(
    _host: *const TpuHostLocation,
    core_type: i32,
    cores_out: *mut *const TpuCoreLocation,
) {
    // SAFETY: Delegates to `TpuTopology_Cores`.
    unsafe {
        TpuTopology_Cores(ptr::null(), core_type, cores_out);
    }
}

/// Creates a [`TpuMeshState`] (`TpuMeshState_Create`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuMeshState_Create() -> *mut TpuMeshState {
    Box::into_raw(Box::new(TpuMeshState {
        mesh_common_state: 1,
    }))
}

/// Frees a [`TpuMeshState`] (`TpuMeshState_Free`).
///
/// # Arguments
///
/// * `state` - Pointer to [`TpuMeshState`].
///
/// # Safety
///
/// `state` must be null or allocated by `TpuMeshState_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuMeshState_Free(state: *mut TpuMeshState) {
    if !state.is_null() {
        // SAFETY: `state` is non-null.
        unsafe {
            drop(Box::from_raw(state));
        }
    }
}

/// Returns the internal mesh common state pointer (`TpuMeshState_MeshCommonState`).
///
/// # Arguments
///
/// * `state` - Pointer to [`TpuMeshState`].
///
/// # Safety
///
/// `state` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuMeshState_MeshCommonState(
    state: *const TpuMeshState,
) -> *const std::ffi::c_void {
    if state.is_null() {
        return ptr::null();
    }
    state.cast::<std::ffi::c_void>()
}

/// Returns the first available [`TpuCoreLocation`] in `topo` (`TpuTopology_AnAvailableCore`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_AnAvailableCore(
    topo: *const TpuTopology,
    core_type: i32,
) -> *const TpuCoreLocation {
    // SAFETY: Delegates to `TpuTopology_CoreForId` with core 0.
    unsafe { TpuTopology_CoreForId(topo, core_type, 0) }
}

/// Alias for [`TpuTopology_Cores`] (`TpuTopology_AvailableCores`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant.
/// * `cores_out` - Output array of `*const TpuCoreLocation`.
///
/// # Safety
///
/// `topo` and `cores_out` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_AvailableCores(
    topo: *const TpuTopology,
    core_type: i32,
    cores_out: *mut *const TpuCoreLocation,
) {
    // SAFETY: Delegates directly to `TpuTopology_Cores`.
    unsafe { TpuTopology_Cores(topo, core_type, cores_out) }
}

/// Alias for [`TpuTopology_NumCores`] (`TpuTopology_AvailableCoresCount`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_AvailableCoresCount(
    topo: *const TpuTopology,
    core_type: i32,
) -> i32 {
    // SAFETY: Delegates directly to `TpuTopology_NumCores`.
    unsafe { TpuTopology_NumCores(topo, core_type) }
}

/// Alias for [`TpuCoreLocation_ChipCoordinates`] (`TpuTopology_ChipCoordinates`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
/// * `x_out` - Output X coordinate pointer.
/// * `y_out` - Output Y coordinate pointer.
/// * `z_out` - Output Z coordinate pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_ChipCoordinates(
    core: *const TpuCoreLocation,
    x_out: *mut i32,
    y_out: *mut i32,
    z_out: *mut i32,
) {
    // SAFETY: Delegates directly to `TpuCoreLocation_ChipCoordinates`.
    unsafe { TpuCoreLocation_ChipCoordinates(core, x_out, y_out, z_out) }
}

/// Returns the total number of chips in `topo` (`TpuTopology_ChipCount`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_ChipCount(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let hosts = unsafe { TpuTopology_HostCount(topo) };
    // SAFETY: `topo` is null or valid.
    let chips_per_host = unsafe { TpuTopology_ChipsPerHost(topo) };
    hosts.saturating_mul(chips_per_host)
}

/// Alias for [`TpuTopology_NumCores`] (`TpuTopology_CoreCount`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_CoreCount(topo: *const TpuTopology, core_type: i32) -> i32 {
    // SAFETY: Delegates directly to `TpuTopology_NumCores`.
    unsafe { TpuTopology_NumCores(topo, core_type) }
}

/// Alias for [`TpuCoreLocation_Id`] (`TpuTopology_CoreId`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
///
/// # Safety
///
/// `core` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_CoreId(core: *const TpuCoreLocation) -> i32 {
    // SAFETY: Delegates directly to `TpuCoreLocation_Id`.
    unsafe { TpuCoreLocation_Id(core) }
}

/// Alias for [`TpuCoreLocation_Index`] (`TpuTopology_CoreIndexOnChip`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
///
/// # Safety
///
/// `core` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_CoreIndexOnChip(core: *const TpuCoreLocation) -> i32 {
    // SAFETY: Delegates directly to `TpuCoreLocation_Index`.
    unsafe { TpuCoreLocation_Index(core) }
}

/// Alias for [`TpuTopology_Cores`] (`TpuTopology_CoreLocations`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant.
/// * `cores_out` - Output array of `*const TpuCoreLocation`.
///
/// # Safety
///
/// `topo` and `cores_out` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_CoreLocations(
    topo: *const TpuTopology,
    core_type: i32,
    cores_out: *mut *const TpuCoreLocation,
) {
    // SAFETY: Delegates directly to `TpuTopology_Cores`.
    unsafe { TpuTopology_Cores(topo, core_type, cores_out) }
}

/// Returns the current host ID (`TpuTopology_CurrentHost`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_CurrentHost(topo: *const TpuTopology) -> i32 {
    // SAFETY: Delegates to `TpuTopology_IdForHost` at origin `(0, 0, 0)`.
    unsafe { TpuTopology_IdForHost(topo, 0, 0, 0) }
}

/// Returns the X host bound (`TpuTopology_HostBounds_X`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_HostBounds_X(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().host_bounds.x).unwrap_or(1)
}

/// Returns the Y host bound (`TpuTopology_HostBounds_Y`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_HostBounds_Y(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().host_bounds.y).unwrap_or(1)
}

/// Returns the Z host bound (`TpuTopology_HostBounds_Z`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_HostBounds_Z(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().host_bounds.z).unwrap_or(1)
}

/// Alias for [`TpuHostLocation_Cores`] (`TpuTopology_HostLocations`).
///
/// # Arguments
///
/// * `host` - Pointer to [`TpuHostLocation`] (or null).
/// * `core_type` - Core type discriminant.
/// * `cores_out` - Output array of `*const TpuCoreLocation`.
///
/// # Safety
///
/// `host` and `cores_out` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTopology_HostLocations(
    host: *const TpuHostLocation,
    core_type: i32,
    cores_out: *mut *const TpuCoreLocation,
) {
    // SAFETY: Delegates directly to `TpuHostLocation_Cores`.
    unsafe { TpuHostLocation_Cores(host, core_type, cores_out) }
}
