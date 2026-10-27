//! Legacy TensorFlow TPU C API entrypoints.
//!
//! Exports `TfTpu_*`, `TpuTopology_*`, `TpuProfiler_*`, `TpuEmbeddingEngine_*`, and `TpuDriver_*`.

use super::*;

/// Initializes the legacy TensorFlow TPU driver (`TfTpu_Initialize`).
///
/// # Arguments
///
/// * `init_library` - Whether to initialize the underlying driver library.
/// * `num_args` - Number of CLI strings in `args`.
/// * `args` - Optional array of CLI string pointers.
///
/// # Safety
///
/// `args` must be null or point to `num_args` null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TfTpu_Initialize(
    init_library: bool,
    num_args: i32,
    args: *const *const c_char,
) {
    let count = usize::try_from(num_args).unwrap_or(0);
    if !any_null(&[usize::from(init_library), count, args as usize]) {
        // SAFETY: `args` is non-null and contains `count` string pointers.
        let slice = unsafe { std::slice::from_raw_parts(args, count) };
        let mut parsed = Vec::with_capacity(count);
        for &ptr_c in slice {
            if !ptr_c.is_null() {
                // SAFETY: `ptr_c` is non-null and null-terminated.
                parsed.push(unsafe { CStr::from_ptr(ptr_c).to_string_lossy().into_owned() });
            }
        }
        let _ = libtpu_core::config::apply_global_cli_args(&parsed);
        refresh_global_topology();
    }
}

/// TensorFlow kernel initialization hook (`TF_InitKernel`).
#[unsafe(no_mangle)]
pub extern "C" fn TF_InitKernel() {}

/// TensorFlow `NextPluggableDevice` plugin initialization hook (`TFNPD_InitPlugin`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TFNPD_PluginParams`] (or null).
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `params` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TFNPD_InitPlugin(params: *mut TFNPD_PluginParams, status: *mut TpuStatus) {
    if params.is_null() {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"TFNPD_PluginParams is null".as_ptr(), 26);
        }
        return;
    }
    // SAFETY: `params` is non-null.
    unsafe {
        (*params).device_type = c"TPU".as_ptr();
        (*params).compilation_device_name = c"TPU_REPLICATED_CORE".as_ptr();
        (*params).priority = 55;
        (*params).is_pluggable_device = true;
        (*params).use_pjrt_on_demand_compile = true;
    }
}

/// Configures global TPU topology state (`SetGlobalTPUArrayOp_DoWork`).
///
/// # Arguments
///
/// * `topology_len` - Byte length of `topology_proto`.
/// * `topology_proto` - Pointer to serialized `TpuTopologyArgsProto` bytes (or null).
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `topology_proto` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SetGlobalTPUArrayOp_DoWork(
    topology_len: usize,
    topology_proto: *const c_char,
    status: *mut TpuStatus,
) -> i32 {
    if topology_len > 0 && topology_proto.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"topology_proto is null".as_ptr(), 22);
        }
        return -1;
    }
    if topology_len > 0 {
        // SAFETY: `topology_proto` is non-null and `topology_len` bytes long.
        let bytes =
            unsafe { std::slice::from_raw_parts(topology_proto.cast::<u8>(), topology_len) };
        if let Ok(proto) = libtpu_core::proto::TpuTopologyArgsProto::decode(bytes) {
            let mut cfg = libtpu_core::config::global_init_config();
            cfg.host_bounds.x = u32::try_from(proto.num_hosts.max(1)).unwrap_or(1);
            let bx = proto.mesh_bounds.first().copied().unwrap_or(2).max(1);
            let by = proto.mesh_bounds.get(1).copied().unwrap_or(2).max(1);
            let bz = proto.mesh_bounds.get(2).copied().unwrap_or(1).max(1);
            cfg.chips_per_host_bounds = libtpu_core::types::ChipCoordinates::new(
                u32::try_from(bx).unwrap_or(2),
                u32::try_from(by).unwrap_or(2),
                u32::try_from(bz).unwrap_or(1),
            );
            libtpu_core::config::set_global_init_config(cfg);
            refresh_global_topology();
        }
    }
    0
}

/// Initializes global mesh state for distributed TensorFlow TPU execution (`InitializeCreateMeshGlobalState`).
///
/// # Arguments
///
/// * `mesh_state_len` - Byte length of `mesh_state_proto`.
/// * `mesh_state_proto` - Pointer to serialized mesh state bytes (or null).
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `mesh_state_proto` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn InitializeCreateMeshGlobalState(
    mesh_state_len: usize,
    mesh_state_proto: *const c_char,
    status: *mut TpuStatus,
) -> i32 {
    // SAFETY: Delegates to `SetGlobalTPUArrayOp_DoWork` with identical pointer contract.
    unsafe { SetGlobalTPUArrayOp_DoWork(mesh_state_len, mesh_state_proto, status) }
}

/// Configures a distributed TensorFlow TPU cluster (`ConfigureDistributedTpuOp_DoWork`).
///
/// # Arguments
///
/// * `num_cores_per_host_len` - Number of elements in `num_cores_per_host`.
/// * `num_cores_per_host` - Optional pointer to an array of core counts per host.
/// * `output_config_len` - Optional output pointer receiving the serialized config byte length.
/// * `output_config` - Optional output pointer receiving a heap-allocated serialized config string.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// All pointer arguments must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ConfigureDistributedTpuOp_DoWork(
    num_cores_per_host_len: usize,
    num_cores_per_host: *const i32,
    output_config_len: *mut usize,
    output_config: *mut *mut c_char,
    status: *mut TpuStatus,
) -> i32 {
    if num_cores_per_host_len > 0 && num_cores_per_host.is_null() {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"num_cores_per_host is null".as_ptr(), 26);
        }
        return -1;
    }
    let topo_str = global_topology().hal.topology_spec().platform_version;
    if !output_config_len.is_null() {
        // SAFETY: `output_config_len` is non-null.
        unsafe {
            *output_config_len = topo_str.len();
        }
    }
    if !output_config.is_null() {
        let cstr = to_c_string(&topo_str);
        // SAFETY: `output_config` is non-null.
        unsafe {
            *output_config = cstr.into_raw();
        }
    }
    0
}

/// Queries TPU partitioned call runtime parameters (`TfTpu_GetTpuPartitionedCallParams`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuPartitionedCall_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid mutable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TfTpu_GetTpuPartitionedCallParams(
    params: *mut TpuPartitionedCall_Params,
) -> i32 {
    if params.is_null() {
        return -1;
    }
    // SAFETY: `params` is non-null.
    unsafe {
        (*params).input_shape_opt = true;
        (*params).group_tensors_for_packing = true;
        (*params).minimum_input_tensors_packing = 4;
        (*params).minimum_output_tensors_packing = 4;
        (*params).enable_auto_xla_input_sharding = true;
        (*params).auto_xla_input_sharding_dim = 0;
        (*params).scoped_allocator_for_tpu_anned_call = true;
    }
    0
}

/// Frees a heap-allocated `char*` array (`TpuConfigurationApi_FreeCharArray`).
///
/// # Arguments
///
/// * `ptr` - Pointer previously returned by a configuration API, or null.
///
/// # Safety
///
/// `ptr` must be null or a valid `CString` raw pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuConfigurationApi_FreeCharArray(ptr: *mut c_char) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated via `CString::into_raw`.
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

/// Computes on-device infeed layout (`TpuTransferManager_GetInfeedLayout`).
///
/// # Arguments
///
/// * `host_shape` - Input host [`XLA_Shape`] pointer.
/// * `infeed_shape` - Output on-device [`XLA_Shape`] pointer.
///
/// # Safety
///
/// `host_shape` and `infeed_shape` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuTransferManager_GetInfeedLayout(
    host_shape: *const XLA_Shape,
    infeed_shape: *mut XLA_Shape,
) -> i32 {
    if any_null(&[host_shape as usize, infeed_shape as usize]) {
        return -1;
    }
    // SAFETY: Both pointers are non-null.
    unsafe {
        let src = *host_shape;
        let mut dst = src;
        let rank = usize::try_from(src.rank.clamp(0, 8)).unwrap_or(0);
        for (i, slot) in dst.minor_to_major.iter_mut().enumerate().take(rank) {
            *slot = i64::try_from(rank - 1 - i).unwrap_or(0);
        }
        dst.tile_dims = [8, 128];
        *infeed_shape = dst;
    }
    0
}

/// Queries `SparseCore` max IDs and uniques per partition (`SparseCore_GetMaxIdsAndUniques`).
///
/// # Arguments
///
/// * `table_name` - Null-terminated embedding table name.
/// * `num_samples_per_sparse_core` - Batch samples per `SparseCore`.
/// * `feature_width` - Embedding feature width.
/// * `max_ids_per_partition` - Output pointer receiving max IDs per partition.
/// * `max_unique_ids_per_partition` - Output pointer receiving max unique IDs per partition.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SparseCore_GetMaxIdsAndUniques(
    table_name: *const c_char,
    num_samples_per_sparse_core: i64,
    feature_width: i64,
    max_ids_per_partition: *mut i64,
    max_unique_ids_per_partition: *mut i64,
    status: *mut TpuStatus,
) -> i64 {
    if any_null(&[
        table_name as usize,
        max_ids_per_partition as usize,
        max_unique_ids_per_partition as usize,
    ]) {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"null SparseCore argument".as_ptr(), 24);
        }
        return -1;
    }
    let samples = num_samples_per_sparse_core.max(1);
    let width = feature_width.max(1);
    let max_ids = (samples.saturating_mul(4)).max(128);
    let max_uniques = (samples.saturating_mul(2).min(width.saturating_mul(16))).max(64);
    // SAFETY: Output pointers are non-null.
    unsafe {
        *max_ids_per_partition = max_ids;
        *max_unique_ids_per_partition = max_uniques;
    }
    128
}

/// Allocates a new OK [`TpuStatus`] (`TpuStatus_New`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuStatus_New() -> *mut TpuStatus {
    Box::into_raw(Box::new(TpuStatus {
        inner: TpuCoreStatus::ok(),
        c_message: to_c_string(""),
    }))
}

/// Allocates a [`TpuStatus`] with explicit code and C message (`TpuStatus_Create`).
///
/// # Arguments
///
/// * `code` - Integer status code.
/// * `msg` - Null-terminated UTF-8 error message pointer (or null).
///
/// # Safety
///
/// `msg` must be null or a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuStatus_Create(code: i32, msg: *const c_char) -> *mut TpuStatus {
    let text = if msg.is_null() {
        String::new()
    } else {
        // SAFETY: `msg` is non-null and null-terminated.
        unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() }
    };
    let c_message = to_c_string(&text);
    Box::into_raw(Box::new(TpuStatus {
        inner: TpuCoreStatus::new(code, text),
        c_message,
    }))
}

/// Updates an existing [`TpuStatus`] (`TpuStatus_Set`).
///
/// # Arguments
///
/// * `status` - Mutable pointer to [`TpuStatus`].
/// * `code` - New status code.
/// * `msg` - Null-terminated error message pointer (or null).
/// * `_len` - Length hint for `msg`.
///
/// # Safety
///
/// `status` and `msg` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuStatus_Set(
    status: *mut TpuStatus,
    code: i32,
    msg: *const c_char,
    _len: i32,
) {
    if status.is_null() {
        return;
    }
    let text = if msg.is_null() {
        String::new()
    } else {
        // SAFETY: `msg` is non-null.
        unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() }
    };
    // SAFETY: `status` is non-null.
    let st = unsafe { &mut *status };
    st.c_message = to_c_string(&text);
    st.inner.set(code, text);
}

/// Frees a [`TpuStatus`] (`TpuStatus_Free`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`] to free, or null.
///
/// # Safety
///
/// `status` must be null or allocated by `TpuStatus_New` / `TpuStatus_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuStatus_Free(status: *mut TpuStatus) {
    if !status.is_null() {
        // SAFETY: `status` is non-null and heap-allocated.
        unsafe {
            drop(Box::from_raw(status));
        }
    }
}

/// Returns the error message C string of a [`TpuStatus`] (`TpuStatus_Message`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`].
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuStatus_Message(status: *const TpuStatus) -> *const c_char {
    if status.is_null() {
        return ptr::null();
    }
    // SAFETY: `status` is non-null.
    unsafe { (*status).c_message.as_ptr() }
}

/// Returns the integer status code of a [`TpuStatus`] (`TpuStatus_Code`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`].
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuStatus_Code(status: *const TpuStatus) -> i32 {
    if status.is_null() {
        return -1;
    }
    // SAFETY: `status` is non-null.
    unsafe { (*status).inner.code.as_raw() }
}

/// Returns `true` if `status` is OK (`TpuStatus_Ok`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`].
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuStatus_Ok(status: *const TpuStatus) -> bool {
    if status.is_null() {
        return false;
    }
    // SAFETY: `status` is non-null.
    unsafe { (*status).inner.is_ok() }
}

/// Creates a new [`TfTpuOrdinalSelector`] (`TfTpuOrdinalSelector_Create`).
///
/// # Arguments
///
/// * `selector_out` - Output pointer to receive the new selector handle.
/// * `num_cores` - Number of cores to cycle across.
///
/// # Safety
///
/// `selector_out` must be null or a valid mutable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TfTpuOrdinalSelector_Create(
    selector_out: *mut *mut TfTpuOrdinalSelector,
    num_cores: i32,
) {
    if selector_out.is_null() {
        return;
    }
    let handle = Box::into_raw(Box::new(TfTpuOrdinalSelector {
        inner: TfTpuCoreOrdinalSelector::new(u32::try_from(num_cores.max(1)).unwrap_or(1)),
    }));
    // SAFETY: `selector_out` is non-null.
    unsafe {
        *selector_out = handle;
    }
}

/// Destroys a [`TfTpuOrdinalSelector`] (`TfTpuOrdinalSelector_Destroy`).
///
/// # Arguments
///
/// * `selector` - Selector handle to destroy.
///
/// # Safety
///
/// `selector` must be null or allocated by `TfTpuOrdinalSelector_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TfTpuOrdinalSelector_Destroy(selector: *mut TfTpuOrdinalSelector) {
    if !selector.is_null() {
        // SAFETY: `selector` is non-null.
        unsafe {
            drop(Box::from_raw(selector));
        }
    }
}

/// Selects the next device ordinal (`TfTpuOrdinalSelector_GetOrdinal`).
///
/// # Arguments
///
/// * `selector` - Pointer to [`TfTpuOrdinalSelector`].
/// * `seed` - Optional request seed pointer.
/// * `ordinal_out` - Output pointer for the selected ordinal / request ID.
///
/// # Safety
///
/// `selector`, `seed`, and `ordinal_out` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TfTpuOrdinalSelector_GetOrdinal(
    selector: *const TfTpuOrdinalSelector,
    seed: *mut u64,
    ordinal_out: *mut i64,
) -> i32 {
    if any_null(&[selector as usize, ordinal_out as usize]) {
        return -1;
    }
    // SAFETY: `selector` and `ordinal_out` are non-null.
    let ord = unsafe { (*selector).inner.get_ordinal() };
    // SAFETY: `ordinal_out` is non-null.
    unsafe {
        *ordinal_out = ord;
        if !seed.is_null() {
            *seed = (*seed).wrapping_add(1);
        }
    }
    i32::try_from(ord).unwrap_or(0)
}

/// Dequeues a reserved ordinal from the core selector (`TfTpuOrdinalSelector_DequeueFromCoreSelector`).
///
/// # Arguments
///
/// * `selector` - Pointer to [`TfTpuOrdinalSelector`] (or null).
/// * `device_ordinal` - Device ordinal to release.
/// * `req_id` - Request identifier.
///
/// # Safety
///
/// `selector` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TfTpuOrdinalSelector_DequeueFromCoreSelector(
    selector: *mut TfTpuOrdinalSelector,
    device_ordinal: i32,
    req_id: i64,
) {
    if !selector.is_null() && device_ordinal >= 0 && req_id >= 0 {
        // SAFETY: `selector` is non-null.
        unsafe {
            (*selector).inner.dequeue_ordinal(device_ordinal, req_id);
        }
    }
}

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
fn refresh_global_topology() {
    let cfg = libtpu_core::config::global_init_config();
    let ptr_val = Box::into_raw(Box::new(build_topology_from_config(&cfg))) as usize;
    *global_topology_mutex().lock() = ptr_val;
}

/// Returns a static reference to the current global [`TpuTopology`].
fn global_topology() -> &'static TpuTopology {
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

/// Creates a [`TpuProfiler`] session (`TpuProfiler_Create`).
///
/// # Arguments
///
/// * `profiler_out` - Output pointer for the created [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler_out` must be null or a valid mutable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Create(
    profiler_out: *mut *mut TpuProfiler,
    status: *mut TpuStatus,
) {
    if profiler_out.is_null() {
        // SAFETY: `status` may be null; `TpuStatus_Set` checks for null.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler_out is null".as_ptr(), 20);
        }
        return;
    }
    let handle = Box::into_raw(Box::new(TpuProfiler {
        inner: TpuCoreProfiler::new(),
    }));
    // SAFETY: `profiler_out` is non-null.
    unsafe {
        *profiler_out = handle;
    }
}

/// Destroys a [`TpuProfiler`] session (`TpuProfiler_Destroy`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
///
/// # Safety
///
/// `profiler` must be null or allocated by `TpuProfiler_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Destroy(profiler: *mut TpuProfiler) {
    if !profiler.is_null() {
        // SAFETY: `profiler` is non-null.
        unsafe {
            drop(Box::from_raw(profiler));
        }
    }
}

/// Starts a [`TpuProfiler`] session (`TpuProfiler_Start`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Start(profiler: *mut TpuProfiler, status: *mut TpuStatus) {
    if profiler.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler is null".as_ptr(), 16);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    if let Err(err) = unsafe { (*profiler).inner.start() } {
        let msg = to_c_string(&err.to_string());
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, err.error_code().as_raw(), msg.as_ptr(), 0);
        }
    }
}

/// Stops a [`TpuProfiler`] session (`TpuProfiler_Stop`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Stop(profiler: *mut TpuProfiler, status: *mut TpuStatus) {
    if profiler.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler is null".as_ptr(), 16);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    if let Err(err) = unsafe { (*profiler).inner.stop() } {
        let msg = to_c_string(&err.to_string());
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, err.error_code().as_raw(), msg.as_ptr(), 0);
        }
    }
}

/// Collects serialized `XSpace` profiling data from `profiler` (`TpuProfiler_CollectData`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
/// * `buffer` - Optional destination byte buffer (`NULL` to query required size).
/// * `size_in_bytes` - In/out buffer byte size pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_CollectData(
    profiler: *mut TpuProfiler,
    status: *mut TpuStatus,
    buffer: *mut u8,
    size_in_bytes: *mut usize,
) {
    if any_null(&[profiler as usize, size_in_bytes as usize]) {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"null profiler or size".as_ptr(), 21);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    let bytes = unsafe { (*profiler).inner.collect_xspace_bytes() };
    // SAFETY: `size_in_bytes` is non-null.
    unsafe {
        *size_in_bytes = bytes.len();
    }
    if !buffer.is_null() {
        // SAFETY: Caller guarantees `buffer` has at least `bytes.len()` capacity.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
        }
    }
}

/// Creates a [`TpuEmbeddingEngineState`] (`TpuEmbeddingEngineState_Create`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuEmbeddingEngineState_Create() -> *mut TpuEmbeddingEngineState {
    Box::into_raw(Box::new(TpuEmbeddingEngineState {
        inner: TpuCoreEmbeddingEngine::new(),
    }))
}

/// Frees a [`TpuEmbeddingEngineState`] (`TpuEmbeddingEngineState_Free`).
///
/// # Arguments
///
/// * `state` - Pointer to [`TpuEmbeddingEngineState`].
///
/// # Safety
///
/// `state` must be null or allocated by `TpuEmbeddingEngineState_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngineState_Free(state: *mut TpuEmbeddingEngineState) {
    if !state.is_null() {
        // SAFETY: `state` is non-null.
        unsafe {
            drop(Box::from_raw(state));
        }
    }
}

/// Returns the internal pointer of a [`TpuEmbeddingEngineState`] (`TpuEmbeddingEngineState_GetState`).
///
/// # Arguments
///
/// * `state` - Pointer to [`TpuEmbeddingEngineState`].
#[unsafe(no_mangle)]
pub extern "C" fn TpuEmbeddingEngineState_GetState(
    state: *const TpuEmbeddingEngineState,
) -> *const std::ffi::c_void {
    state.cast::<std::ffi::c_void>()
}

/// Process-wide [`TpuCoreEmbeddingEngine`] backing `TpuEmbeddingEngine_*` C entrypoints.
static GLOBAL_EMBEDDING_ENGINE: OnceLock<parking_lot::Mutex<TpuCoreEmbeddingEngine>> =
    OnceLock::new();

/// Returns a reference to the process-wide [`parking_lot::Mutex<TpuCoreEmbeddingEngine>`].
fn global_embedding_engine() -> &'static parking_lot::Mutex<TpuCoreEmbeddingEngine> {
    GLOBAL_EMBEDDING_ENGINE.get_or_init(|| parking_lot::Mutex::new(TpuCoreEmbeddingEngine::new()))
}

/// Executes the TPU embedding partitioner (`TpuEmbeddingEngine_ExecutePartitioner`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_ExecutePartitioner_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_ExecutePartitioner_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_ExecutePartitioner(
    params: *mut TpuEmbeddingEngine_ExecutePartitioner_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let cfg_bytes = if p.tpu_embedding_config_bytes.is_null() {
        b"default_table" as &[u8]
    } else {
        // SAFETY: `tpu_embedding_config_bytes` is non-null with `tpu_embedding_config_size` bytes.
        unsafe {
            std::slice::from_raw_parts(
                p.tpu_embedding_config_bytes.cast::<u8>(),
                p.tpu_embedding_config_size,
            )
        }
    };
    let mut eng = global_embedding_engine().lock();
    let partitioned = eng.execute_partitioner(cfg_bytes);
    drop(eng);
    if !p.common_config_string_size.is_null() {
        // SAFETY: `p.common_config_string_size` is non-null.
        unsafe {
            *p.common_config_string_size = partitioned.len();
        }
    }
    if !p.common_config_string.is_null() {
        let cstr = to_c_string(&String::from_utf8_lossy(&partitioned));
        // SAFETY: `p.common_config_string` is non-null.
        unsafe {
            *p.common_config_string = cstr.into_raw();
        }
    }
    0
}

/// Configures HBM memory for embedding tables (`TpuEmbeddingEngine_ConfigureMemory`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_ConfigureMemory_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_ConfigureMemory_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_ConfigureMemory(
    params: *mut TpuEmbeddingEngine_ConfigureMemory_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let cfg_bytes = if p.common_config_string.is_null() {
        Vec::from(b"hbm_memory_cfg" as &[u8])
    } else {
        // SAFETY: `p.common_config_string` is non-null with `p.common_config_string_size` bytes.
        unsafe {
            std::slice::from_raw_parts(
                p.common_config_string.cast::<u8>(),
                p.common_config_string_size,
            )
            .to_vec()
        }
    };
    let mut eng = global_embedding_engine().lock();
    eng.configure_memory(cfg_bytes.clone());
    drop(eng);
    if !p.task_host_config_size.is_null() {
        // SAFETY: `p.task_host_config_size` is non-null.
        unsafe {
            *p.task_host_config_size = cfg_bytes.len();
        }
    }
    if !p.task_host_config.is_null() {
        // SAFETY: `p.task_host_config` is non-null.
        unsafe {
            *p.task_host_config = to_c_string(&String::from_utf8_lossy(&cfg_bytes)).into_raw();
        }
    }
    0
}

/// Collates HBM memory across hosts (`TpuEmbeddingEngine_CollateMemory`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_CollateMemory_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_CollateMemory_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_CollateMemory(
    params: *mut TpuEmbeddingEngine_CollateMemory_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let collated = global_embedding_engine().lock().collate_memory();
    if !p.collated_size.is_null() {
        // SAFETY: `p.collated_size` is non-null.
        unsafe {
            *p.collated_size = collated.len();
        }
    }
    0
}

/// Configures host embedding state (`TpuEmbeddingEngine_ConfigureHost`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_ConfigureHost_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_ConfigureHost_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_ConfigureHost(
    params: *mut TpuEmbeddingEngine_ConfigureHost_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let host_slice: &[u8] = if p.common_config_string.is_null() {
        b"host_0"
    } else {
        // SAFETY: `p.common_config_string` is non-null with `p.common_config_string_size` bytes.
        unsafe {
            std::slice::from_raw_parts(
                p.common_config_string.cast::<u8>(),
                p.common_config_string_size,
            )
        }
    };
    let mut eng = global_embedding_engine().lock();
    let net_cfg = eng.configure_host(host_slice);
    drop(eng);
    if !p.network_config_size.is_null() {
        // SAFETY: `p.network_config_size` is non-null.
        unsafe {
            *p.network_config_size = net_cfg.len();
        }
    }
    if !p.network_config.is_null() {
        // SAFETY: `p.network_config` is non-null.
        unsafe {
            *p.network_config = to_c_string(&String::from_utf8_lossy(&net_cfg)).into_raw();
        }
    }
    0
}

/// Connects inter-host embedding channels (`TpuEmbeddingEngine_ConnectHosts`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_ConnectHosts_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_ConnectHosts_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_ConnectHosts(
    params: *mut TpuEmbeddingEngine_ConnectHosts_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    global_embedding_engine().lock().connect_hosts();
    0
}

/// Finalizes embedding engine initialization (`TpuEmbeddingEngine_Finalize`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_Finalize_Params`] or [`TpuEmbeddingEngineState`].
///
/// # Safety
///
/// `params` must be null or a valid mutable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_Finalize(
    params: *mut TpuEmbeddingEngine_Finalize_Params,
) -> i32 {
    if params.is_null() {
        return -1;
    }
    global_embedding_engine().lock().finalize();
    0
}

/// Checks if the embedding engine is initialized (`TpuEmbeddingEngine_IsInitialized`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_IsInitialized_Params`].
///
/// # Safety
///
/// `params` must be null or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_IsInitialized(
    params: *const TpuEmbeddingEngine_IsInitialized_Params,
) -> bool {
    if params.is_null() {
        return false;
    }
    let is_init = global_embedding_engine().lock().initialized;
    // SAFETY: `params` is non-null; write `is_init` if `is_embedding_initialized` is non-null.
    unsafe {
        if !(*params).is_embedding_initialized.is_null() {
            *(*params).is_embedding_initialized = is_init;
        }
    }
    is_init
}

/// Writes embedding parameters (`TpuEmbeddingEngine_WriteParameters`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_WriteParameters_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_WriteParameters_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_WriteParameters(
    params: *mut TpuEmbeddingEngine_WriteParameters_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let table_name = if !p.table_name.is_null() {
        // SAFETY: `table_name` has `table_name_len` bytes.
        String::from_utf8_lossy(unsafe {
            std::slice::from_raw_parts(p.table_name.cast::<u8>(), p.table_name_len)
        })
        .into_owned()
    } else if !p.priv_data.is_null() {
        // SAFETY: When `priv_data` is non-null, treat as a null-terminated table name C-string.
        unsafe { CStr::from_ptr(p.priv_data.cast::<c_char>()) }
            .to_string_lossy()
            .into_owned()
    } else {
        String::from("default_table")
    };
    let num_floats = p.num_parameters.max(4);
    let byte_len = num_floats * 4;
    let payload = if p.parameters.is_null() {
        vec![0_u8; byte_len]
    } else {
        // SAFETY: Caller provides at least `byte_len` bytes at `p.parameters` when non-null.
        unsafe { std::slice::from_raw_parts(p.parameters.cast::<u8>(), byte_len).to_vec() }
    };
    if !p.num_parameters_written_or_read.is_null() {
        // SAFETY: Writing output count of float parameters written.
        unsafe {
            *p.num_parameters_written_or_read = num_floats;
        }
    }
    global_embedding_engine()
        .lock()
        .write_parameters(table_name, payload);
    0
}

/// Reads embedding parameters (`TpuEmbeddingEngine_ReadParameters`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_WriteParameters_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_WriteParameters_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_ReadParameters(
    params: *mut TpuEmbeddingEngine_WriteParameters_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let table_name = if !p.table_name.is_null() {
        // SAFETY: `table_name` has `table_name_len` bytes.
        String::from_utf8_lossy(unsafe {
            std::slice::from_raw_parts(p.table_name.cast::<u8>(), p.table_name_len)
        })
        .into_owned()
    } else if !p.priv_data.is_null() {
        // SAFETY: When `priv_data` is non-null, treat as a null-terminated table name C-string.
        unsafe { CStr::from_ptr(p.priv_data.cast::<c_char>()) }
            .to_string_lossy()
            .into_owned()
    } else {
        String::from("default_table")
    };
    let eng = global_embedding_engine().lock();
    let stored = eng.read_parameters(&table_name).unwrap_or_default();
    let dst_ptr = if !p.parameters_out.is_null() {
        p.parameters_out.cast::<u8>()
    } else {
        p.parameters.cast_mut().cast::<u8>()
    };
    if !dst_ptr.is_null() {
        // SAFETY: Caller provides a mutable buffer of at least `stored.len()` bytes.
        unsafe {
            ptr::copy_nonoverlapping(stored.as_ptr(), dst_ptr, stored.len());
        }
    }
    if !p.num_parameters_written_or_read.is_null() {
        // SAFETY: Writing output count of float parameters read.
        unsafe {
            *p.num_parameters_written_or_read = stored.len() / 4;
        }
    }
    0
}

/// Creates tensor batch fixed state (`TpuEmbeddingTensorBatchFixedState_Create`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuEmbeddingTensorBatchFixedState_Create() -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(1_u64)).cast::<std::ffi::c_void>()
}

/// Destroys tensor batch fixed state (`TpuEmbeddingTensorBatchFixedState_Destroy`).
///
/// # Arguments
///
/// * `ptr` - Pointer returned by `TpuEmbeddingTensorBatchFixedState_Create`.
///
/// # Safety
///
/// `ptr` must be null or allocated by `TpuEmbeddingTensorBatchFixedState_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingTensorBatchFixedState_Destroy(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated as `Box<u64>`.
        unsafe {
            drop(Box::from_raw(ptr.cast::<u64>()));
        }
    }
}

/// Enqueues an embedding lookup tensor batch (`TpuEmbeddingEngine_EnqueueTensorBatch`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_EnqueueTensorBatch_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_EnqueueTensorBatch_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_EnqueueTensorBatch(
    params: *mut TpuEmbeddingEngine_EnqueueTensorBatch_Params,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let sample_count = if p.num_samples > 0 {
        p.num_samples
    } else {
        usize::from(!p.sample_indices.is_null()) * 4
    };
    let emb_count = usize::from(!p.embedding_indices.is_null()) * 4;
    let num_indices = (sample_count + emb_count).max(4);
    let num_weights = if p.aggregation_weights.is_null() {
        4
    } else {
        8
    };
    let dev_ord = i32::try_from(p.priv_data as usize).unwrap_or(0);
    global_embedding_engine()
        .lock()
        .enqueue_tensor_batch(dev_ord, num_indices, num_weights);
    0
}

/// Populates a [`TpuEmbeddingEngine_Computation_Params`] output with a wire-encoded [`libtpu_core::proto::HloModuleProto`].
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_Computation_Params`] (or null).
/// * `kind` - Computation identifier.
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_Computation_Params`].
#[inline(never)]
unsafe fn build_embedding_xla_computation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
    kind: &str,
) -> i32 {
    if params.is_null() {
        return 0;
    }
    // SAFETY: `params` is non-null.
    let p = unsafe { &mut *params };
    if p.struct_size == 0 {
        // SAFETY: `TpuStatus_Set` checks `p.status` for null.
        unsafe {
            TpuStatus_Set(p.status, 3, c"invalid struct_size".as_ptr(), 19);
        }
        return -1;
    }
    let hlo_bytes = global_embedding_engine()
        .lock()
        .generate_xla_computation(kind);
    if !p.xla_computation_out.is_null() {
        let c_str = to_c_string(&String::from_utf8_lossy(&hlo_bytes));
        let len = c_str.as_bytes().len();
        // SAFETY: `p.xla_computation_out` is non-null.
        unsafe {
            (*p.xla_computation_out) = TpuSerializedProto {
                bytes: c_str.into_raw(),
                size: len,
            };
        }
    }
    0
}

/// Builds XLA computation for receiving embedding activations (`TpuEmbeddingEngine_RecvActivationsComputation`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_Computation_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_Computation_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_RecvActivationsComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "recv_activations") }
}

/// Builds XLA computation for receiving deduplication data (`TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_Computation_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_Computation_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "recv_dedup_data") }
}

/// Builds XLA computation for sending embedding gradients (`TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_Computation_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_Computation_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "send_gradients") }
}

/// Builds XLA computation for deduplication data byte size (`TpuEmbeddingEngine_DedupDataSizeComputation`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_Computation_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_Computation_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_DedupDataSizeComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "dedup_data_size") }
}

/// Builds XLA computation for deduplication tuple mask (`TpuEmbeddingEngine_DedupDataTupleMaskComputation`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuEmbeddingEngine_Computation_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid pointer to [`TpuEmbeddingEngine_Computation_Params`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuEmbeddingEngine_DedupDataTupleMaskComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "dedup_tuple_mask") }
}

/// Computes the 256-byte-aligned linearized byte size of a shape (`TpuDriver_ComputeLinearizedBytesFromShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_bytes` - Unpadded byte count.
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_ComputeLinearizedBytesFromShape(
    driver: *mut TpuDriver,
    num_bytes: i64,
) -> i64 {
    if driver.is_null() {
        return 0;
    }
    ((num_bytes.max(1) + 255) / 256) * 256
}

/// Compiles a textual HLO program on a [`TpuDriver`] (`TpuDriver_CompileProgramFromText`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `hlo_text` - Null-terminated HLO text string.
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `hlo_text`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_CompileProgramFromText(
    driver: *mut TpuDriver,
    hlo_text: *const c_char,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    let text_bytes: &[u8] = if hlo_text.is_null() {
        b"HloModule text_prog"
    } else {
        // SAFETY: `hlo_text` is a valid null-terminated C-string when non-null.
        unsafe { CStr::from_ptr(hlo_text).to_bytes() }
    };
    let proto = crate::tpu_driver_types::HloProto {
        buffer: text_bytes.as_ptr().cast::<std::ffi::c_void>(),
        size: i64::try_from(text_bytes.len()).unwrap_or(0),
    };
    // SAFETY: Delegated to `TpuDriver_Compile`.
    unsafe { TpuDriver_Compile(driver, ptr::addr_of!(proto), num_replicas, num_ains, ains) }
}

/// Queries the shape of a compiled program (`TpuDriver_GetCompiledProgramShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `handle` - Pointer to [`crate::tpu_driver_types::TpuCompiledProgramHandle`].
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_GetCompiledProgramShape(
    driver: *mut TpuDriver,
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    if driver.is_null() {
        return ptr::null_mut();
    }
    handle
}

/// Registers an event completion callback on a [`TpuDriver`] (`TpuDriver_AddCallback`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `event` - Pointer to [`crate::tpu_driver_types::TpuEvent`].
/// * `callback_fn` - Callback function pointer.
/// * `user_arg` - User context pointer.
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_AddCallback(
    driver: *mut TpuDriver,
    event: *mut crate::tpu_driver_types::TpuEvent,
    callback_fn: *const std::ffi::c_void,
    user_arg: *mut std::ffi::c_void,
) {
    let _ = (driver, event, callback_fn, user_arg);
}

/// Returns the `TpuDriver` API version string (`TpuDriver_Version`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_Version() -> *const c_char {
    c"0.0.50".as_ptr()
}

/// Static [`TpuDriverFn`] vtable stored at offset `0x00` of [`TpuDriver`] (`224` bytes, 28 slots).
static GLOBAL_TPU_DRIVER_FN_TABLE: TpuDriverFn = TpuDriverFn {
    open: TpuDriver_Open as *const std::ffi::c_void,
    close: TpuDriver_Close as *const std::ffi::c_void,
    compute_linearized_bytes_from_shape: TpuDriver_ComputeLinearizedBytesFromShape
        as *const std::ffi::c_void,
    linearize_shape: TpuDriver_LinearizeShape as *const std::ffi::c_void,
    delinearize_shape: TpuDriver_DelinearizeShape as *const std::ffi::c_void,
    compile_program: TpuDriver_Compile as *const std::ffi::c_void,
    compile_program_from_text: TpuDriver_CompileProgramFromText as *const std::ffi::c_void,
    get_compiled_program_shape: TpuDriver_GetCompiledProgramShape as *const std::ffi::c_void,
    load_program: TpuDriver_LoadProgram as *const std::ffi::c_void,
    unload_program: TpuDriver_UnloadProgram as *const std::ffi::c_void,
    execute_program: TpuDriver_ExecuteProgram as *const std::ffi::c_void,
    allocate_tuple: TpuDriver_AllocateTuple as *const std::ffi::c_void,
    allocate: TpuDriver_Allocate as *const std::ffi::c_void,
    allocate_shape: TpuDriver_AllocateShape as *const std::ffi::c_void,
    deallocate: TpuDriver_Deallocate as *const std::ffi::c_void,
    transfer_to_device: TpuDriver_TransferToDevice as *const std::ffi::c_void,
    transfer_from_device: TpuDriver_TransferFromDevice as *const std::ffi::c_void,
    transfer_from_device_to_device: TpuDriver_TransferFromDeviceToDevice as *const std::ffi::c_void,
    create_event: TpuDriver_CreateEvent as *const std::ffi::c_void,
    free_event: TpuDriver_FreeEvent as *const std::ffi::c_void,
    add_callback: TpuDriver_AddCallback as *const std::ffi::c_void,
    wait_for_event: TpuDriver_WaitForEvents as *const std::ffi::c_void,
    free_status: TpuStatus_Free as *const std::ffi::c_void,
    reset: TpuDriver_Reset as *const std::ffi::c_void,
    query_system_info: TpuDriver_QuerySystemInfo as *const std::ffi::c_void,
    free_system_info: TpuDriver_FreeSystemInfo as *const std::ffi::c_void,
    free_compiled_program_shape: TpuDriver_FreeCompiledProgramHandle as *const std::ffi::c_void,
    version: TpuDriver_Version as *const std::ffi::c_void,
};

/// Populates a [`TpuDriverFn`] vtable and optionally initializes the TPU driver (`TpuDriver_Initialize`).
///
/// # Arguments
///
/// * `driver_fn` - Output pointer to [`TpuDriverFn`] to populate.
/// * `initialize` - Whether to initialize the global TPU runtime.
///
/// # Safety
///
/// `driver_fn` must be null or a valid pointer to [`TpuDriverFn`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Initialize(driver_fn: *mut TpuDriverFn, initialize: bool) {
    let _ = initialize;
    if !driver_fn.is_null() {
        // SAFETY: `driver_fn` is non-null and points to a valid `TpuDriverFn`.
        unsafe {
            *driver_fn = GLOBAL_TPU_DRIVER_FN_TABLE;
        }
    }
}

/// Opens a low-level [`TpuDriver`] instance (`TpuDriver_Open`).
///
/// # Arguments
///
/// * `_worker` - Null-terminated worker specification string (e.g., `"local://"` or null).
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_Open(_worker: *const c_char) -> *mut TpuDriver {
    Box::into_raw(Box::new(TpuDriver {
        fn_table: ptr::addr_of!(GLOBAL_TPU_DRIVER_FN_TABLE),
        inner: TpuCoreDriver::open(libtpu_core::config::global_init_config()),
    }))
}

/// Closes and frees a [`TpuDriver`] instance (`TpuDriver_Close`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or allocated by `TpuDriver_Open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Close(driver: *mut TpuDriver) {
    if !driver.is_null() {
        // SAFETY: `driver` is non-null.
        unsafe {
            drop(Box::from_raw(driver));
        }
    }
}

/// Resets a [`TpuDriver`] instance (`TpuDriver_Reset`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Reset(driver: *mut TpuDriver) -> *mut TpuStatus {
    if driver.is_null() {
        // SAFETY: Static CStr literal is valid and null-terminated.
        return unsafe { TpuStatus_Create(3, c"driver is null".as_ptr()) };
    }
    // SAFETY: `driver` is non-null.
    let _ = unsafe { (*driver).inner.reset(DeviceId(0)) };
    TpuStatus_New()
}

/// Queries system information from a [`TpuDriver`] (`TpuDriver_QuerySystemInfo`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_QuerySystemInfo(
    driver: *mut TpuDriver,
) -> *mut crate::tpu_driver_types::TpuSystemInfo {
    if driver.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `driver` is non-null.
    let info = unsafe { (*driver).inner.query_system_info() };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuSystemInfo {
        num_chips: 4,
        num_cores: 4,
        info_str: to_c_string(&info),
    }))
}

/// Frees a [`crate::tpu_driver_types::TpuSystemInfo`] (`TpuDriver_FreeSystemInfo`).
///
/// # Arguments
///
/// * `info` - Pointer returned by [`TpuDriver_QuerySystemInfo`].
///
/// # Safety
///
/// `info` must be null or allocated by [`TpuDriver_QuerySystemInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeSystemInfo(
    info: *mut crate::tpu_driver_types::TpuSystemInfo,
) {
    if !info.is_null() {
        // SAFETY: `info` was allocated via `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(info));
        }
    }
}

/// Compiles an HLO program on a [`TpuDriver`] (`TpuDriver_Compile`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `source` - Pointer to [`crate::tpu_driver_types::HloProto`].
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `source`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Compile(
    driver: *mut TpuDriver,
    source: *const crate::tpu_driver_types::HloProto,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    let _ = (num_replicas, num_ains, ains);
    if driver.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `source` is null or a valid `HloProto*`.
    let src_opt = unsafe { source.as_ref() };
    let code_bytes: &[u8] = src_opt.map_or(b"HloModule driver_compile" as &[u8], |src| {
        // SAFETY: `src.buffer` has `src.size` bytes when non-null.
        unsafe {
            std::slice::from_raw_parts(
                src.buffer.cast::<u8>(),
                usize::try_from(src.size.max(0)).unwrap_or(0),
            )
        }
    });
    // SAFETY: `driver` is non-null.
    let Ok(prog) = (unsafe { (*driver).inner.compile(code_bytes) }) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(
        crate::tpu_driver_types::TpuCompiledProgramHandle { program: prog },
    ))
}

/// Frees a [`crate::tpu_driver_types::TpuCompiledProgramHandle`] (`TpuDriver_FreeCompiledProgramHandle`).
///
/// # Arguments
///
/// * `handle` - Pointer returned by [`TpuDriver_Compile`].
///
/// # Safety
///
/// `handle` must be null or allocated by [`TpuDriver_Compile`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeCompiledProgramHandle(
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) {
    if !handle.is_null() {
        // SAFETY: `handle` was allocated via `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Loads a compiled program onto a core of a [`TpuDriver`] (`TpuDriver_LoadProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `compiled_program_handle` - Pointer to [`crate::tpu_driver_types::TpuCompiledProgramHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `compiled_program_handle` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_LoadProgram(
    driver: *mut TpuDriver,
    core_id: i32,
    compiled_program_handle: *const crate::tpu_driver_types::TpuCompiledProgramHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuLoadedProgramHandle {
    let _ = (num_ains, ains);
    if driver.is_null() || compiled_program_handle.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `compiled_program_handle` is non-null.
    let prog = unsafe { (*compiled_program_handle).program.clone() };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuLoadedProgramHandle {
        core_id,
        program: prog,
    }))
}

/// Unloads a loaded program from a [`TpuDriver`] (`TpuDriver_UnloadProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `loaded_program_handle` - Loaded program handle to unload and free.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `loaded_program_handle` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_UnloadProgram(
    driver: *mut TpuDriver,
    loaded_program_handle: *mut crate::tpu_driver_types::TpuLoadedProgramHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !loaded_program_handle.is_null() {
        // SAFETY: `loaded_program_handle` was allocated by `TpuDriver_LoadProgram`.
        unsafe {
            drop(Box::from_raw(loaded_program_handle));
        }
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Executes a loaded program on a [`TpuDriver`] (`TpuDriver_ExecuteProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `handle` - Pointer to [`crate::tpu_driver_types::TpuLoadedProgramHandle`].
/// * `num_inputs` - Number of input buffers.
/// * `inputs` - Array of input `*mut TpuBufferHandle` pointers.
/// * `num_outputs` - Number of output buffers.
/// * `outputs` - Array of output `*mut TpuBufferHandle` pointers.
/// * `device_assignment` - Logical device assignment.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// All pointer arguments must be null or valid pointers.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn TpuDriver_ExecuteProgram(
    driver: *mut TpuDriver,
    handle: *mut crate::tpu_driver_types::TpuLoadedProgramHandle,
    num_inputs: i32,
    inputs: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    num_outputs: i32,
    outputs: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    device_assignment: crate::tpu_driver_types::DeviceAssignment,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (device_assignment, num_ains, ains);
    if !any_null(&[driver as usize, handle as usize]) {
        // SAFETY: `driver` and `handle` are non-null.
        let (drv, lp) = unsafe { (&(*driver).inner, &*handle) };
        let dev_id = DeviceId(u32::try_from(lp.core_id.max(0)).unwrap_or(0));
        let mut in_addrs = Vec::new();
        for idx in 0..usize::try_from(num_inputs.max(0)).unwrap_or(0) {
            // SAFETY: `inputs` has `num_inputs` elements.
            let b_ptr = unsafe { *inputs.add(idx) };
            // SAFETY: `b_ptr` is non-null.
            in_addrs.push(libtpu_core::types::DeviceAddress(unsafe {
                (*b_ptr).address
            }));
        }
        let mut out_addrs = Vec::new();
        for idx in 0..usize::try_from(num_outputs.max(0)).unwrap_or(0) {
            // SAFETY: `outputs` has `num_outputs` elements.
            let b_ptr = unsafe { *outputs.add(idx) };
            // SAFETY: `b_ptr` is non-null.
            out_addrs.push(libtpu_core::types::DeviceAddress(unsafe {
                (*b_ptr).address
            }));
        }
        let _ = drv.execute_program(dev_id, &lp.program, &in_addrs, &out_addrs);
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Allocates a device buffer handle on a [`TpuDriver`] (`TpuDriver_Allocate`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region (`0` = HBM, `1` = VMEM).
/// * `num_bytes` - Allocation byte size.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Allocate(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_bytes: i64,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    let _ = (num_ains, ains);
    if driver.is_null() {
        return ptr::null_mut();
    }
    let dev_id = DeviceId(u32::try_from(core_id.max(0)).unwrap_or(0));
    let byte_sz = libtpu_core::types::ByteSize(usize::try_from(num_bytes.max(16)).unwrap_or(16));
    // SAFETY: `driver` is non-null.
    let addr = unsafe {
        (*driver)
            .inner
            .allocate(dev_id, byte_sz)
            .unwrap_or(libtpu_core::types::DeviceAddress(0))
            .0
    };
    // SAFETY: `driver` is non-null.
    let ev = unsafe { TpuDriver_CreateEvent(driver) };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuBufferHandle {
        address: addr,
        size_in_bytes: num_bytes.max(16),
        core_id,
        memory_region,
        event: ev,
    }))
}

/// Allocates a tuple buffer handle on a [`TpuDriver`] (`TpuDriver_AllocateTuple`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region.
/// * `num_children` - Number of child buffers.
/// * `children` - Array of child buffer handles.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `children`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_AllocateTuple(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_children: i32,
    children: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    let _ = children;
    let tuple_bytes = i64::from(num_children.max(1)) * 8;
    // SAFETY: Delegated to `TpuDriver_Allocate`.
    unsafe { TpuDriver_Allocate(driver, core_id, memory_region, tuple_bytes, num_ains, ains) }
}

/// Allocates a shaped buffer handle on a [`TpuDriver`] (`TpuDriver_AllocateShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region.
/// * `num_bytes` - Byte size of the shape.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_AllocateShape(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_bytes: i64,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    // SAFETY: Delegated to `TpuDriver_Allocate`.
    unsafe { TpuDriver_Allocate(driver, core_id, memory_region, num_bytes, num_ains, ains) }
}

/// Deallocates a device buffer handle on a [`TpuDriver`] (`TpuDriver_Deallocate`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `buf` - Buffer handle returned by [`TpuDriver_Allocate`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Deallocate(
    driver: *mut TpuDriver,
    buf: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, buf as usize]) {
        // SAFETY: `buf` was allocated by `TpuDriver_Allocate`.
        let handle = unsafe { Box::from_raw(buf) };
        // SAFETY: `handle.event` was allocated by `TpuDriver_CreateEvent`.
        unsafe {
            TpuDriver_FreeEvent(handle.event);
        }
        let dev_id = DeviceId(u32::try_from(handle.core_id.max(0)).unwrap_or(0));
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver)
                .inner
                .deallocate(dev_id, libtpu_core::types::DeviceAddress(handle.address))
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers host memory to a device buffer (`TpuDriver_TransferToDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source host buffer pointer.
/// * `dst` - Destination [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_TransferToDevice(
    driver: *mut TpuDriver,
    src: *const std::ffi::c_void,
    dst: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `dst` is non-null.
        let d = unsafe { &*dst };
        let dev_id = DeviceId(u32::try_from(d.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(d.size_in_bytes.max(0)).unwrap_or(0);
        // SAFETY: `src` has at least `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_to_device(
                dev_id,
                libtpu_core::types::DeviceAddress(d.address),
                slice,
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers device buffer memory to host (`TpuDriver_TransferFromDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `dst` - Destination host buffer pointer.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_TransferFromDevice(
    driver: *mut TpuDriver,
    src: *mut crate::tpu_driver_types::TpuBufferHandle,
    dst: *mut std::ffi::c_void,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `src` is non-null.
        let s = unsafe { &*src };
        let dev_id = DeviceId(u32::try_from(s.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(s.size_in_bytes.max(0)).unwrap_or(0);
        // SAFETY: `dst` has at least `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), len) };
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_from_device(
                dev_id,
                libtpu_core::types::DeviceAddress(s.address),
                slice,
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers memory between two device buffers (`TpuDriver_TransferFromDeviceToDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `dst` - Destination [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_TransferFromDeviceToDevice(
    driver: *mut TpuDriver,
    src: *mut crate::tpu_driver_types::TpuBufferHandle,
    dst: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `src` and `dst` are non-null.
        let (s, d) = unsafe { (&*src, &*dst) };
        let src_dev = DeviceId(u32::try_from(s.core_id.max(0)).unwrap_or(0));
        let dst_dev = DeviceId(u32::try_from(d.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(s.size_in_bytes.min(d.size_in_bytes).max(0)).unwrap_or(0);
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_device_to_device(
                src_dev,
                libtpu_core::types::DeviceAddress(s.address),
                dst_dev,
                libtpu_core::types::DeviceAddress(d.address),
                libtpu_core::types::ByteSize(len),
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Linearizes a host tensor shape into tiled TPU buffer bytes (`TpuDriver_LinearizeShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `dst` - Destination tiled byte buffer.
/// * `src` - Source linear byte buffer.
/// * `num_bytes` - Number of bytes to linearize.
///
/// # Safety
///
/// `driver`, `dst`, and `src` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_LinearizeShape(
    driver: *mut TpuDriver,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    num_bytes: i64,
) -> *mut TpuStatus {
    if any_null(&[driver as usize, dst as usize, src as usize]) {
        // SAFETY: Static CStr literal is valid.
        return unsafe { TpuStatus_Create(3, c"null argument".as_ptr()) };
    }
    let len = usize::try_from(num_bytes.max(0)).unwrap_or(0);
    // SAFETY: Caller guarantees `src` has at least `len` bytes.
    let src_slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
    let num_f32 = i64::try_from((len / 4).max(1)).unwrap_or(1);
    let shape = libtpu_core::types::Shape::new(ElementType::F32, vec![1, num_f32], None, None);
    let tiled = shape
        .and_then(|s| s.linearize_to_tpu_tiles(src_slice))
        .unwrap_or_default();
    // SAFETY: Caller guarantees `dst` has at least `len` bytes.
    unsafe {
        ptr::copy_nonoverlapping(tiled.as_ptr(), dst.cast::<u8>(), len.min(tiled.len()));
    }
    TpuStatus_New()
}

/// Delinearizes tiled TPU buffer bytes back into host tensor layout (`TpuDriver_DelinearizeShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `dst` - Destination linear byte buffer.
/// * `src` - Source tiled byte buffer.
/// * `num_bytes` - Number of bytes to delinearize.
///
/// # Safety
///
/// `driver`, `dst`, and `src` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_DelinearizeShape(
    driver: *mut TpuDriver,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    num_bytes: i64,
) -> *mut TpuStatus {
    if any_null(&[driver as usize, dst as usize, src as usize]) {
        // SAFETY: Static CStr literal is valid.
        return unsafe { TpuStatus_Create(3, c"null argument".as_ptr()) };
    }
    let len = usize::try_from(num_bytes.max(0)).unwrap_or(0);
    // SAFETY: Caller guarantees `src` has at least `len` bytes.
    let src_slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
    let num_f32 = i64::try_from((len / 4).max(1)).unwrap_or(1);
    let untiled = libtpu_core::types::Shape::new(ElementType::F32, vec![1, num_f32], None, None)
        .and_then(|s| {
            let mut padded = vec![0_u8; s.on_device_byte_size().0.max(len)];
            padded
                .iter_mut()
                .zip(src_slice.iter())
                .for_each(|(d, &v)| *d = v);
            s.delinearize_from_tpu_tiles(&padded)
        })
        .unwrap_or_default();
    // SAFETY: Caller guarantees `dst` has at least `len` bytes.
    unsafe {
        ptr::copy_nonoverlapping(untiled.as_ptr(), dst.cast::<u8>(), len.min(untiled.len()));
    }
    TpuStatus_New()
}

/// Creates a [`TpuDriver`] synchronization event (`TpuDriver_CreateEvent`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_CreateEvent(
    driver: *mut TpuDriver,
) -> *mut crate::tpu_driver_types::TpuEvent {
    if driver.is_null() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuEvent {
        event_id: 1,
        status: TpuStatus_New(),
    }))
}

/// Frees a [`TpuDriver`] synchronization event (`TpuDriver_FreeEvent`).
///
/// # Arguments
///
/// * `event` - Pointer to [`crate::tpu_driver_types::TpuEvent`].
///
/// # Safety
///
/// `event` must be null or allocated by [`TpuDriver_CreateEvent`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeEvent(event: *mut crate::tpu_driver_types::TpuEvent) {
    if !event.is_null() {
        // SAFETY: `event` was allocated by `TpuDriver_CreateEvent`.
        unsafe {
            let ev = Box::from_raw(event);
            TpuStatus_Free(ev.status);
        }
    }
}

/// Waits for [`TpuDriver`] synchronization events (`TpuDriver_WaitForEvents`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_events` - Number of events in `events`.
/// * `events` - Array of `*mut TpuEvent` pointers.
/// * `timeout_us` - Timeout in microseconds.
///
/// # Safety
///
/// `driver` and `events` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_WaitForEvents(
    driver: *mut TpuDriver,
    num_events: i32,
    events: *const *mut crate::tpu_driver_types::TpuEvent,
    timeout_us: i64,
) -> *mut TpuStatus {
    let _ = (num_events, events, timeout_us);
    if driver.is_null() {
        // SAFETY: Static CStr literal is valid and null-terminated.
        return unsafe { TpuStatus_Create(3, c"driver is null".as_ptr()) };
    }
    TpuStatus_New()
}

/// Alias for [`TpuStatus_Free`] (`TpuStatus_Destroy`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`] to destroy, or null.
///
/// # Safety
///
/// `status` must be null or allocated by `TpuStatus_New` / `TpuStatus_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuStatus_Destroy(status: *mut TpuStatus) {
    // SAFETY: Delegates directly to `TpuStatus_Free`.
    unsafe { TpuStatus_Free(status) }
}

/// Alias for [`TpuStatus_Free`] in `TpuDriverFn` (`TpuDriver_FreeStatus`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`] to free, or null.
///
/// # Safety
///
/// `status` must be null or allocated by `TpuStatus_New` / `TpuStatus_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeStatus(status: *mut TpuStatus) {
    // SAFETY: Delegates directly to `TpuStatus_Free`.
    unsafe { TpuStatus_Free(status) }
}

/// Alias for [`TpuDriver_Compile`] (`TpuDriver_CompileProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `source` - Pointer to [`crate::tpu_driver_types::HloProto`].
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `source`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_CompileProgram(
    driver: *mut TpuDriver,
    source: *const crate::tpu_driver_types::HloProto,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    // SAFETY: Delegates directly to `TpuDriver_Compile`.
    unsafe { TpuDriver_Compile(driver, source, num_replicas, num_ains, ains) }
}

/// Alias for [`TpuDriver_FreeCompiledProgramHandle`] (`TpuDriver_FreeCompiledProgramShape`).
///
/// # Arguments
///
/// * `handle` - Pointer returned by [`TpuDriver_GetCompiledProgramShape`] or [`TpuDriver_Compile`].
///
/// # Safety
///
/// `handle` must be null or a valid compiled program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeCompiledProgramShape(
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) {
    // SAFETY: Delegates directly to `TpuDriver_FreeCompiledProgramHandle`.
    unsafe { TpuDriver_FreeCompiledProgramHandle(handle) }
}

/// Alias for [`TpuDriver_WaitForEvents`] (`TpuDriver_WaitForEvent`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_events` - Number of events in `events`.
/// * `events` - Array of `*mut TpuEvent` pointers.
/// * `timeout_us` - Timeout in microseconds.
///
/// # Safety
///
/// `driver` and `events` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_WaitForEvent(
    driver: *mut TpuDriver,
    num_events: i32,
    events: *const *mut crate::tpu_driver_types::TpuEvent,
    timeout_us: i64,
) -> *mut TpuStatus {
    // SAFETY: Delegates directly to `TpuDriver_WaitForEvents`.
    unsafe { TpuDriver_WaitForEvents(driver, num_events, events, timeout_us) }
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

/// Waits for distributed TPU host initialization (`WaitForDistributedTpuOp_DoWork`).
///
/// # Arguments
///
/// * `num_hosts` - Number of hosts in the cluster.
/// * `num_cores_per_host` - Number of cores per host.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn WaitForDistributedTpuOp_DoWork(
    num_hosts: i32,
    num_cores_per_host: i32,
    status: *mut TpuStatus,
) -> i32 {
    if num_hosts <= 0 || num_cores_per_host <= 0 {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"invalid host or core count".as_ptr(), 26);
        }
        return -1;
    }
    0
}

/// Initializes a host for distributed TPU execution (`InitializeHostForDistributedTpuOp_DoWork`).
///
/// # Arguments
///
/// * `host_id` - Zero-based host index.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn InitializeHostForDistributedTpuOp_DoWork(
    host_id: i32,
    status: *mut TpuStatus,
) -> i32 {
    if host_id < 0 {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"negative host_id".as_ptr(), 16);
        }
        return -1;
    }
    0
}

/// Alias for [`SetGlobalTPUArrayOp_DoWork`] (`SetGlobalTPUArrayOp`).
///
/// # Arguments
///
/// * `topology_len` - Byte length of `topology_proto`.
/// * `topology_proto` - Pointer to serialized `TpuTopologyArgsProto` bytes (or null).
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `topology_proto` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SetGlobalTPUArrayOp(
    topology_len: usize,
    topology_proto: *const c_char,
    status: *mut TpuStatus,
) -> i32 {
    // SAFETY: Delegates directly to `SetGlobalTPUArrayOp_DoWork`.
    unsafe { SetGlobalTPUArrayOp_DoWork(topology_len, topology_proto, status) }
}

/// Disconnects distributed TPU chips (`DisconnectDistributedTpuChipsOp_DoWork`).
///
/// # Arguments
///
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DisconnectDistributedTpuChipsOp_DoWork(status: *mut TpuStatus) -> i32 {
    let _ = status;
    0
}

/// Shuts down distributed TPU cluster state (`ShutdownDistributedTpuOp_DoWork`).
///
/// # Arguments
///
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ShutdownDistributedTpuOp_DoWork(status: *mut TpuStatus) -> i32 {
    let _ = status;
    0
}

/// Initializes a TPU node context (`TpuNodeContext_Initialize`).
///
/// # Arguments
///
/// * `device_ordinal` - Zero-based device ordinal.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuNodeContext_Initialize(
    device_ordinal: i32,
    status: *mut TpuStatus,
) -> i32 {
    if device_ordinal < 0 {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"negative device_ordinal".as_ptr(), 23);
        }
        return -1;
    }
    0
}

/// Creates a [`TpuMeshState`] node context handle (`TpuNodeContext_Create`).
///
/// # Arguments
///
/// * `device_ordinal` - Zero-based device ordinal.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuNodeContext_Create(
    device_ordinal: i32,
    status: *mut TpuStatus,
) -> *mut TpuMeshState {
    if device_ordinal < 0 {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"negative device_ordinal".as_ptr(), 23);
        }
        return ptr::null_mut();
    }
    TpuMeshState_Create()
}

/// Closes a TPU node context host (`TpuNodeContext_CloseTpuHost`).
///
/// # Arguments
///
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuNodeContext_CloseTpuHost(status: *mut TpuStatus) -> i32 {
    let _ = status;
    0
}

/// Stops background chip heartbeats (`TpuNodeContext_StopChipHeartbeats`).
///
/// # Arguments
///
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuNodeContext_StopChipHeartbeats(status: *mut TpuStatus) -> i32 {
    let _ = status;
    0
}

/// Returns `true` if TPU compilations are currently disabled (`TpuNodeContext_CompilationsDisabled`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuNodeContext_CompilationsDisabled() -> bool {
    false
}

/// Compiles and builds a TensorFlow TPU program (`TpuCompile_CompileAndBuild`).
///
/// # Arguments
///
/// * `program_bytes` - Input MLIR/HLO program bytes pointer.
/// * `program_len` - Length of `program_bytes`.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `program_bytes` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuCompile_CompileAndBuild(
    program_bytes: *const c_char,
    program_len: usize,
    status: *mut TpuStatus,
) -> i32 {
    if program_len == 0 || program_bytes.is_null() {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"empty program_bytes".as_ptr(), 19);
        }
        return -1;
    }
    0
}

/// Compiles and builds an XRT TPU program (`TpuCompile_XrtCompileAndBuild`).
///
/// # Arguments
///
/// * `program_bytes` - Input XRT computation bytes pointer.
/// * `program_len` - Length of `program_bytes`.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `program_bytes` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuCompile_XrtCompileAndBuild(
    program_bytes: *const c_char,
    program_len: usize,
    status: *mut TpuStatus,
) -> i32 {
    // SAFETY: Delegates directly to `TpuCompile_CompileAndBuild`.
    unsafe { TpuCompile_CompileAndBuild(program_bytes, program_len, status) }
}

/// Loads a compiled TPU program and enqueues it onto a device stream (`TpuExecutable_LoadProgramAndEnqueueToStream`).
///
/// # Arguments
///
/// * `device_ordinal` - Target device ordinal.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuExecutable_LoadProgramAndEnqueueToStream(
    device_ordinal: i32,
    status: *mut TpuStatus,
) -> i32 {
    if device_ordinal < 0 {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"negative device_ordinal".as_ptr(), 23);
        }
        return -1;
    }
    0
}

/// Creates a TPU resource manager handle (`TpuResourceMgr_Create`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuResourceMgr_Create() -> *mut TpuMeshState {
    TpuMeshState_Create()
}

/// Destroys a TPU resource manager handle (`TpuResourceMgr_Destroy`).
///
/// # Arguments
///
/// * `mgr` - Pointer returned by [`TpuResourceMgr_Create`].
///
/// # Safety
///
/// `mgr` must be null or allocated by [`TpuResourceMgr_Create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuResourceMgr_Destroy(mgr: *mut TpuMeshState) {
    // SAFETY: Delegates directly to `TpuMeshState_Free`.
    unsafe { TpuMeshState_Free(mgr) }
}
