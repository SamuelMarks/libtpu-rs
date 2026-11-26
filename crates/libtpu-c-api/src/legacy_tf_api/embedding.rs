//! Legacy TensorFlow TPU embedding engine C API entrypoints.
//!
//! Exports `TpuEmbeddingEngine*` and `TpuEmbeddingTensorBatchFixedState_*`.

use super::*;

/// Creates a [`TpuEmbeddingEngineState`] (`TpuEmbeddingEngineState_Create`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuEmbeddingEngineState_Create() -> *mut TpuEmbeddingEngineState {
    crate::pjrt_helpers::ffi_catch_unwind(std::ptr::null_mut(), || {
        Box::into_raw(Box::new(TpuEmbeddingEngineState {
            inner: TpuCoreEmbeddingEngine::new(),
        }))
    })
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
    crate::pjrt_helpers::ffi_catch_unwind((), || {
        if !state.is_null() {
            // SAFETY: `state` is non-null.
            unsafe {
                drop(Box::from_raw(state));
            }
        }
    });
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
    crate::pjrt_helpers::ffi_catch_unwind(std::ptr::null(), || state.cast::<std::ffi::c_void>())
}

/// Process-wide [`TpuCoreEmbeddingEngine`] backing `TpuEmbeddingEngine_*` C entrypoints.
static GLOBAL_EMBEDDING_ENGINE: OnceLock<parking_lot::Mutex<TpuCoreEmbeddingEngine>> =
    OnceLock::new();

/// Returns a reference to the process-wide [`parking_lot::Mutex<TpuCoreEmbeddingEngine>`].
fn global_embedding_engine() -> &'static parking_lot::Mutex<TpuCoreEmbeddingEngine> {
    GLOBAL_EMBEDDING_ENGINE.get_or_init(|| parking_lot::Mutex::new(TpuCoreEmbeddingEngine::new()))
}

/// Queries the registered float parameter count for `table_name` in [`GLOBAL_EMBEDDING_ENGINE`].
///
/// # Arguments
///
/// * `table_name` - Name of the embedding table.
pub(super) fn lookup_registered_table_vocab_size(table_name: &str) -> Option<usize> {
    global_embedding_engine()
        .lock()
        .tables
        .get(table_name)
        .map(|bytes| bytes.len() / 4)
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
        let Ok(partitioned) = eng.execute_partitioner(cfg_bytes) else {
            // SAFETY: `TpuStatus_Set` checks `p.status` for null.
            unsafe {
                TpuStatus_Set(p.status, 3, c"empty embedding config".as_ptr(), 22);
            }
            return -1;
        };
        drop(eng);
        if !p.common_config_string_size.is_null() {
            // SAFETY: `p.common_config_string_size` is non-null.
            unsafe {
                *p.common_config_string_size = partitioned.len();
            }
        }
        if !p.common_config_string.is_null() {
            // SAFETY: `p.common_config_string` is non-null; `alloc_c_byte_buf` preserves arbitrary binary protobuf bytes.
            unsafe {
                *p.common_config_string = crate::pjrt_helpers::alloc_c_byte_buf(&partitioned);
            }
        }
        0
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
        let Ok(mem_out) = eng.configure_memory(&cfg_bytes) else {
            // SAFETY: `TpuStatus_Set` checks `p.status` for null.
            unsafe {
                TpuStatus_Set(p.status, 3, c"empty memory config".as_ptr(), 19);
            }
            return -1;
        };
        drop(eng);
        if !p.task_host_config_size.is_null() {
            // SAFETY: `p.task_host_config_size` is non-null.
            unsafe {
                *p.task_host_config_size = mem_out.len();
            }
        }
        if !p.task_host_config.is_null() {
            // SAFETY: `p.task_host_config` is non-null.
            unsafe {
                *p.task_host_config = crate::pjrt_helpers::alloc_c_byte_buf(&mem_out);
            }
        }
        0
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
        if p.memory_configs_size > 0 && p.memory_configs.is_null() {
            // SAFETY: `TpuStatus_Set` checks `p.status` for null.
            unsafe {
                TpuStatus_Set(p.status, 3, c"memory_configs is null".as_ptr(), 22);
            }
            return -1;
        }
        let mut owned_slices: Vec<Vec<u8>> = Vec::with_capacity(p.memory_configs_size);
        if p.memory_configs_size > 0 {
            // SAFETY: `p.memory_configs` was verified non-null above for `p.memory_configs_size > 0`.
            let descriptors =
                unsafe { std::slice::from_raw_parts(p.memory_configs, p.memory_configs_size) };
            for desc in descriptors {
                if desc.size > 0 && desc.bytes.is_null() {
                    // SAFETY: `TpuStatus_Set` checks `p.status` for null.
                    unsafe {
                        TpuStatus_Set(p.status, 3, c"memory_config bytes is null".as_ptr(), 27);
                    }
                    return -1;
                }
                if desc.size == 0 {
                    owned_slices.push(Vec::new());
                } else {
                    // SAFETY: `desc.bytes` is non-null with `desc.size` readable bytes.
                    let slice =
                        unsafe { std::slice::from_raw_parts(desc.bytes.cast::<u8>(), desc.size) };
                    owned_slices.push(slice.to_vec());
                }
            }
        }
        let refs: Vec<&[u8]> = owned_slices.iter().map(Vec::as_slice).collect();
        let Ok(collated) = global_embedding_engine().lock().collate_memory(&refs) else {
            // SAFETY: `TpuStatus_Set` checks `p.status` for null.
            unsafe {
                TpuStatus_Set(p.status, 3, c"collate_memory failed".as_ptr(), 21);
            }
            return -1;
        };
        if !p.collated_size.is_null() {
            // SAFETY: `p.collated_size` is non-null.
            unsafe {
                *p.collated_size = collated.len();
            }
        }
        if !p.merged_memory_config.is_null() {
            // SAFETY: `p.merged_memory_config` is non-null.
            unsafe {
                *p.merged_memory_config = crate::pjrt_helpers::alloc_c_byte_buf(&collated);
            }
        }
        0
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
        let Ok(net_cfg) = eng.configure_host(host_slice) else {
            // SAFETY: `TpuStatus_Set` checks `p.status` for null.
            unsafe {
                TpuStatus_Set(p.status, 3, c"empty host config".as_ptr(), 17);
            }
            return -1;
        };
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
                *p.network_config = crate::pjrt_helpers::alloc_c_byte_buf(&net_cfg);
            }
        }
        0
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
        if p.network_configs_size > 0 && p.network_configs.is_null() {
            // SAFETY: `TpuStatus_Set` checks `p.status` for null.
            unsafe {
                TpuStatus_Set(p.status, 3, c"network_configs is null".as_ptr(), 23);
            }
            return -1;
        }
        let mut owned_nets: Vec<Vec<u8>> = Vec::with_capacity(p.network_configs_size);
        if p.network_configs_size > 0 {
            // SAFETY: `p.network_configs` was verified non-null above for `p.network_configs_size > 0`.
            let descriptors =
                unsafe { std::slice::from_raw_parts(p.network_configs, p.network_configs_size) };
            for desc in descriptors {
                if desc.size > 0 && desc.bytes.is_null() {
                    // SAFETY: `TpuStatus_Set` checks `p.status` for null.
                    unsafe {
                        TpuStatus_Set(p.status, 3, c"network_config bytes is null".as_ptr(), 28);
                    }
                    return -1;
                }
                if desc.size == 0 {
                    owned_nets.push(Vec::new());
                } else {
                    // SAFETY: `desc.bytes` is non-null with `desc.size` readable bytes.
                    let slice =
                        unsafe { std::slice::from_raw_parts(desc.bytes.cast::<u8>(), desc.size) };
                    owned_nets.push(slice.to_vec());
                }
            }
        }
        let refs: Vec<&[u8]> = owned_nets.iter().map(Vec::as_slice).collect();
        if global_embedding_engine()
            .lock()
            .connect_hosts(&refs)
            .is_err()
        {
            // SAFETY: `TpuStatus_Set` checks `p.status` for null.
            unsafe {
                TpuStatus_Set(p.status, 3, c"connect_hosts failed".as_ptr(), 20);
            }
            return -1;
        }
        0
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
        if params.is_null() {
            return -1;
        }
        global_embedding_engine().lock().finalize();
        0
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(false, || {
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
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
    })
}

/// Creates tensor batch fixed state (`TpuEmbeddingTensorBatchFixedState_Create`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuEmbeddingTensorBatchFixedState_Create() -> *mut std::ffi::c_void {
    crate::pjrt_helpers::ffi_catch_unwind(std::ptr::null_mut(), || {
        Box::into_raw(Box::new(1_u64)).cast::<std::ffi::c_void>()
    })
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
    crate::pjrt_helpers::ffi_catch_unwind((), || {
        if !ptr.is_null() {
            // SAFETY: `ptr` was allocated as `Box<u64>`.
            unsafe {
                drop(Box::from_raw(ptr.cast::<u64>()));
            }
        }
    });
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
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
    })
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
        let len = hlo_bytes.len();
        let raw_buf = crate::pjrt_helpers::alloc_c_byte_buf(&hlo_bytes);
        // SAFETY: `p.xla_computation_out` is non-null.
        unsafe {
            (*p.xla_computation_out) = TpuSerializedProto {
                bytes: raw_buf,
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
        // SAFETY: Delegated to `build_embedding_xla_computation`.
        unsafe { build_embedding_xla_computation(params, "recv_activations") }
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
        // SAFETY: Delegated to `build_embedding_xla_computation`.
        unsafe { build_embedding_xla_computation(params, "recv_dedup_data") }
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
        // SAFETY: Delegated to `build_embedding_xla_computation`.
        unsafe { build_embedding_xla_computation(params, "send_gradients") }
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
        // SAFETY: Delegated to `build_embedding_xla_computation`.
        unsafe { build_embedding_xla_computation(params, "dedup_data_size") }
    })
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
    crate::pjrt_helpers::ffi_catch_unwind(-1, || {
        // SAFETY: Delegated to `build_embedding_xla_computation`.
        unsafe { build_embedding_xla_computation(params, "dedup_tuple_mask") }
    })
}
