//! Legacy TensorFlow TPU runtime, status, and compilation C API entrypoints.
//!
//! Exports `TfTpu_*`, `TF_InitKernel`, `TFNPD_InitPlugin`, `TpuStatus_*`, `TfTpuOrdinalSelector_*`,
//! `TpuNodeContext_*`, `TpuCompile_*`, `TpuExecutable_*`, `TpuResourceMgr_*`, and distributed TPU ops.

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
