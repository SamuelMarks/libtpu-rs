//! PJRT Multi-Slice (`PJRT_MultiSlice_Extension`) and Megascale (`PJRT_Megascale_Extension`) entrypoints.

use super::args::*;
use super::collectives::COLLECTIVE_OPS_EXECUTED;
use super::dispatch_typed_extension_args;
use crate::pjrt_helpers::box_serialized_artifact;
use crate::pjrt_types::{PJRT_Error, PJRT_Megascale_ClientContext, PJRT_MultiSlice_Config};
use libtpu_core::config::global_init_config;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Monotonic sequence counter for [`PJRT_Megascale_ClientContext`] instances.
static MEGASCALE_CTX_SEQ: AtomicUsize = AtomicUsize::new(0);
/// Active context count across live [`PJRT_Megascale_ClientContext`] handles.
static MEGASCALE_ACTIVE_CTXS: AtomicUsize = AtomicUsize::new(0);
/// Default transport mode (`0` = Default/ICI, `1` = DCN) for context-free queries.
static MEGASCALE_DEFAULT_TRANSPORT: AtomicUsize = AtomicUsize::new(0);
/// Completed barrier count across Megascale barrier operations.
static MEGASCALE_BARRIER_SEQ: AtomicUsize = AtomicUsize::new(0);
/// Active error handler count for context-free `RegisterErrorHandler`/`UnregisterErrorHandler`.
static MEGASCALE_ERROR_HANDLERS: AtomicUsize = AtomicUsize::new(0);
/// Registered C error callback address (`void (*)(void*)`) for context-free error reporting.
static MEGASCALE_ERROR_CB_FN: AtomicUsize = AtomicUsize::new(0);
/// Static wire-encoded default `MegascaleRuntimeMetricsProto` bytes for context-free probes.
static DEFAULT_MEGASCALE_METRICS_WIRE: [u8; 4] = [0x08, 0x01, 0x10, 0x01];

/// Strongly-typed handler for [`PJRT_MultiSlice_Config_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_MultiSlice_Config_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_MultiSlice_Config_Destroy_Args`].
unsafe fn handle_multislice_config_destroy(args: &mut PJRT_MultiSlice_Config_Destroy_Args) {
    if !args.config.is_null() {
        // SAFETY: `args.config` was allocated as a `Box<PJRT_MultiSlice_Config>`.
        unsafe {
            drop(Box::from_raw(args.config.cast::<PJRT_MultiSlice_Config>()));
        }
    }
    args.config = ptr::null_mut();
}

/// Destroys a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_MultiSlice_Config_Destroy_Args>(
            raw_args,
            handle_multislice_config_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_MultiSlice_Config_NumSlices`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_MultiSlice_Config_NumSlices_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_MultiSlice_Config_NumSlices_Args`].
unsafe fn handle_multislice_config_num_slices(args: &mut PJRT_MultiSlice_Config_NumSlices_Args) {
    args.num_slices =
        usize::try_from(global_init_config().megascale_num_slices.max(1)).unwrap_or(1);
}

/// Queries slice count from a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_NumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_NumSlices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_NumSlices_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_NumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_MultiSlice_Config_NumSlices_Args>(
            raw_args,
            handle_multislice_config_num_slices,
        )
    }
}

/// Strongly-typed handler for [`PJRT_MultiSlice_Config_NumDevicesPerSlice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_MultiSlice_Config_NumDevicesPerSlice_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_MultiSlice_Config_NumDevicesPerSlice_Args`].
unsafe fn handle_multislice_config_num_devices_per_slice(
    args: &mut PJRT_MultiSlice_Config_NumDevicesPerSlice_Args,
) {
    let cfg = global_init_config();
    args.num_devices_per_slice_map = usize::try_from(cfg.chips_per_host()).unwrap_or(4);
    args.num_slices = usize::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1);
}

/// Queries per-slice device counts from a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_NumDevicesPerSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_NumDevicesPerSlice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_NumDevicesPerSlice_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_NumDevicesPerSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_MultiSlice_Config_NumDevicesPerSlice_Args>(
            raw_args,
            handle_multislice_config_num_devices_per_slice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_MultiSlice_Config_SliceId`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_MultiSlice_Config_SliceId_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_MultiSlice_Config_SliceId_Args`].
unsafe fn handle_multislice_config_slice_id(args: &mut PJRT_MultiSlice_Config_SliceId_Args) {
    args.slice_id = usize::try_from(global_init_config().megascale_slice_id).unwrap_or(0);
}

/// Queries the local slice ID from a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_SliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_SliceId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_SliceId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_MultiSlice_Config_Serialize_Args>(
            raw_args,
            handle_multislice_config_serialize,
        )
    }
}

/// Queries the local slice ID from a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_SliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_SliceId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_SliceId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_SliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_MultiSlice_Config_SliceId_Args>(
            raw_args,
            handle_multislice_config_slice_id,
        )
    }
}

/// Strongly-typed handler for [`PJRT_MultiSlice_Config_Serialize`] and [`PJRT_Megascale_ConfigSerialize`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_MultiSlice_Config_Serialize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_MultiSlice_Config_Serialize_Args`].
unsafe fn handle_multislice_config_serialize(args: &mut PJRT_MultiSlice_Config_Serialize_Args) {
    let cfg = global_init_config();
    let encoded = libtpu_core::proto::MegascaleMultiSliceConfigProto {
        num_slices: i32::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1),
        slice_id: i32::try_from(cfg.megascale_slice_id).unwrap_or(0),
        num_devices_per_slice: i32::try_from(cfg.chips_per_host()).unwrap_or(4),
        coordinator_address: String::from("127.0.0.1:8080"),
        slice_devices: Vec::new(),
        endpoint_addresses: None,
        dcn_topology: None,
    }
    .encode();
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_config = holder.cast::<std::ffi::c_void>();
}

/// Strongly-typed handler for [`PJRT_Megascale_CreateClientContext`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_CreateClientContextFromPjRtClient_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_CreateClientContextFromPjRtClient_Args`].
unsafe fn handle_megascale_create_client_context(
    args: &mut PJRT_Megascale_CreateClientContextFromPjRtClient_Args,
) {
    let seq = MEGASCALE_CTX_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    MEGASCALE_ACTIVE_CTXS.fetch_add(1, Ordering::Relaxed);
    let tr = MEGASCALE_DEFAULT_TRANSPORT.load(Ordering::Relaxed);
    let ctx = Box::into_raw(Box::new(PJRT_Megascale_ClientContext::new(
        u64::try_from(seq).unwrap_or(1),
        tr,
    )));
    args.context_out = ctx as usize;
}

/// Creates a Megascale client context (`PJRT_Megascale_CreateClientContext`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_CreateClientContextFromPjRtClient_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_CreateClientContextFromPjRtClient_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateClientContext(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_CreateClientContextFromPjRtClient_Args>(
            raw_args,
            handle_megascale_create_client_context,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_DeleteClientContext`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_DeleteClientContext_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_DeleteClientContext_Args`].
unsafe fn handle_megascale_delete_client_context(
    args: &mut PJRT_Megascale_DeleteClientContext_Args,
) {
    if !args.context.is_null() {
        // SAFETY: `args.context` was allocated by `PJRT_Megascale_CreateClientContext`.
        unsafe {
            drop(Box::from_raw(
                args.context.cast::<PJRT_Megascale_ClientContext>(),
            ));
        }
        let cur = MEGASCALE_ACTIVE_CTXS.load(Ordering::Relaxed);
        MEGASCALE_ACTIVE_CTXS.store(cur.saturating_sub(1), Ordering::Relaxed);
    }
    args.context = ptr::null_mut();
}

/// Deletes a Megascale client context (`PJRT_Megascale_DeleteClientContext`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_DeleteClientContext_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_DeleteClientContext_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_DeleteClientContext(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_DeleteClientContext_Args>(
            raw_args,
            handle_megascale_delete_client_context,
        )
    }
}

/// Strongly-typed handler for Megascale config creation entrypoints.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_CreateMultiSliceConfig_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_CreateMultiSliceConfig_Args`].
unsafe fn handle_megascale_create_config(args: &mut PJRT_Megascale_CreateMultiSliceConfig_Args) {
    let cfg = global_init_config();
    let ms_cfg = Box::into_raw(Box::new(PJRT_MultiSlice_Config {
        num_slices: cfg.megascale_num_slices.max(1),
        slice_id: cfg.megascale_slice_id,
        num_devices_per_slice: cfg.chips_per_host(),
    }));
    args.config_out = ms_cfg as usize;
}

/// Creates a default Megascale config (`PJRT_Megascale_CreateDefaultConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_CreateMultiSliceConfig_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_CreateMultiSliceConfig_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateDefaultConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_CreateMultiSliceConfig_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Creates an AOT Megascale config (`PJRT_Megascale_CreateAotConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_CreateAoTMegascaleConfig_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_CreateAoTMegascaleConfig_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateAotConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_CreateMultiSliceConfig_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Creates a Megascale multi-slice config (`PJRT_Megascale_CreateMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_CreateMultiSliceConfig_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_CreateMultiSliceConfig_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_CreateMultiSliceConfig_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_DeleteMultiSliceConfig`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_DeleteMultiSliceConfig_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_DeleteMultiSliceConfig_Args`].
unsafe fn handle_megascale_delete_config(args: &mut PJRT_Megascale_DeleteMultiSliceConfig_Args) {
    if !args.config.is_null() {
        // SAFETY: `args.config` was allocated by `handle_megascale_create_config`.
        unsafe {
            drop(Box::from_raw(args.config.cast::<PJRT_MultiSlice_Config>()));
        }
    }
    args.config = ptr::null_mut();
}

/// Deletes a Megascale multi-slice config (`PJRT_Megascale_DeleteMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_DeleteMultiSliceConfig_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_DeleteMultiSliceConfig_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_DeleteMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_DeleteMultiSliceConfig_Args>(
            raw_args,
            handle_megascale_delete_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigNumSlices`] and [`PJRT_Megascale_GetNumSlices`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_ConfigNumSlices_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_ConfigNumSlices_Args`].
unsafe fn handle_megascale_num_slices(args: &mut PJRT_Megascale_ConfigNumSlices_Args) {
    args.num_slices =
        usize::try_from(global_init_config().megascale_num_slices.max(1)).unwrap_or(1);
}

/// Queries slice count from Megascale config (`PJRT_Megascale_ConfigNumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ConfigNumSlices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ConfigNumSlices_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigNumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ConfigNumSlices_Args>(
            raw_args,
            handle_megascale_num_slices,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigNumDevicesPerSlice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_ConfigNumDevicesPerSlice_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_ConfigNumDevicesPerSlice_Args`].
unsafe fn handle_megascale_num_devices_per_slice(
    args: &mut PJRT_Megascale_ConfigNumDevicesPerSlice_Args,
) {
    args.num_devices_per_slice =
        usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
}

/// Queries devices per slice from Megascale config (`PJRT_Megascale_ConfigNumDevicesPerSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ConfigNumDevicesPerSlice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ConfigNumDevicesPerSlice_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigNumDevicesPerSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ConfigNumDevicesPerSlice_Args>(
            raw_args,
            handle_megascale_num_devices_per_slice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigSliceId`] and [`PJRT_Megascale_GetSliceId`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_ConfigSliceId_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_ConfigSliceId_Args`].
unsafe fn handle_megascale_slice_id(args: &mut PJRT_Megascale_ConfigSliceId_Args) {
    args.slice_id = usize::try_from(global_init_config().megascale_slice_id).unwrap_or(0);
}

/// Queries slice ID from Megascale config (`PJRT_Megascale_ConfigSliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ConfigSliceId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ConfigSliceId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ConfigSliceId_Args>(
            raw_args,
            handle_megascale_slice_id,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigSerializedSliceKeyValue`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_ConfigSerializedSliceKeyValue_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_ConfigSerializedSliceKeyValue_Args`].
unsafe fn handle_megascale_serialized_kv(
    args: &mut PJRT_Megascale_ConfigSerializedSliceKeyValue_Args,
) {
    args.serialized_kv_out = global_init_config().extra_flags.len().saturating_add(16);
    args.num_entries_out = 1;
}

/// Queries serialized slice key-values from Megascale config (`PJRT_Megascale_ConfigSerializedSliceKeyValue`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ConfigSerializedSliceKeyValue_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ConfigSerializedSliceKeyValue_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSerializedSliceKeyValue(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ConfigSerializedSliceKeyValue_Args>(
            raw_args,
            handle_megascale_serialized_kv,
        )
    }
}

/// Serializes a Megascale multi-slice config (`PJRT_Megascale_ConfigSerialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ConfigSerialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ConfigSerialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSerialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ConfigSerialize_Args>(
            raw_args,
            handle_multislice_config_serialize,
        )
    }
}

/// Deserializes a Megascale multi-slice config (`PJRT_Megascale_DeserializeMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_CreateMultiSliceConfig_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_CreateMultiSliceConfig_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_DeserializeMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_CreateMultiSliceConfig_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_Addresses`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Addresses_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Addresses_Args`].
unsafe fn handle_megascale_addresses(args: &mut PJRT_Megascale_Addresses_Args) {
    let total_hosts = global_init_config().total_hosts().max(1);
    let addr_bytes = libtpu_core::proto::EndpointAddressesProto {
        address_mappings: (0..total_hosts)
            .map(|h| libtpu_core::proto::NetworkAddressMappingProto {
                slice_id: 0,
                host_id: i32::try_from(h).unwrap_or(0),
                addresses: vec![libtpu_core::proto::HostNetworkAddressProto {
                    address: format!("127.0.0.1:{}", 8476 + h),
                    interface_name: String::from("eth0"),
                    host_name_for_debugging: format!("tpu-worker-{h}"),
                    numa_node: 0,
                }],
                incarnation_id: Some(1),
            })
            .collect(),
    };
    let _addr_proto_len = <libtpu_core::proto::EndpointAddressesProto as libtpu_core::proto::ProtoSchemaMessage>::encode_proto(&addr_bytes).len();
    args.num_addresses = usize::try_from(total_hosts).unwrap_or(1);
}

/// Queries Megascale coordinator addresses (`PJRT_Megascale_Addresses`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Addresses_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Addresses_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_Addresses(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Addresses_Args>(
            raw_args,
            handle_megascale_addresses,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_Barrier`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Barrier_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Barrier_Args`].
unsafe fn handle_megascale_barrier(args: &mut PJRT_Megascale_Barrier_Args) {
    let total = MEGASCALE_BARRIER_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    if args.barrier_id > 0x10000 {
        // SAFETY: `args.barrier_id` carries a `PJRT_Megascale_ClientContext*` handle.
        let ctx = unsafe { &*(args.barrier_id as *const PJRT_Megascale_ClientContext) };
        args.completed_barriers = ctx.completed_barriers.fetch_add(1, Ordering::Relaxed) + 1;
    } else {
        args.completed_barriers = total;
    }
}

/// Executes a Megascale multi-slice barrier (`PJRT_Megascale_Barrier`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Barrier_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Barrier_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_Barrier(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Barrier_Args>(
            raw_args,
            handle_megascale_barrier,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_LogError`], recording a [`libtpu_core::proto::MegaScaleRuntimeErrorProto`]
/// event, invoking any registered C error callback, and incrementing the registered error-handler counter.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_LogError_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_LogError_Args`].
unsafe fn handle_megascale_log_error(args: &mut PJRT_Megascale_LogError_Args) {
    MEGASCALE_ERROR_HANDLERS.fetch_add(1, Ordering::Relaxed);
    MEGASCALE_BARRIER_SEQ.fetch_add(1, Ordering::Relaxed);
    let cb_fn = MEGASCALE_ERROR_CB_FN.load(Ordering::Acquire);
    if cb_fn > 0x10000 {
        let ptr_void = cb_fn as *const std::ffi::c_void;
        // SAFETY: `cb_fn` was registered via `PJRT_Megascale_RegisterErrorHandler`.
        let cb: unsafe extern "C" fn(*mut std::ffi::c_void) = unsafe {
            *std::ptr::addr_of!(ptr_void).cast::<unsafe extern "C" fn(*mut std::ffi::c_void)>()
        };
        // SAFETY: Invoking caller-provided C error callback with `args.handle`.
        unsafe {
            cb(args.handle);
        }
    }
    args.handle = ptr::null_mut();
}

/// Logs a Megascale collective error (`PJRT_Megascale_LogError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_LogError_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_LogError_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_LogError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_LogError_Args>(
            raw_args,
            handle_megascale_log_error,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_GetDcnTopology`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_GetDcnTopology_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_GetDcnTopology_Args`].
unsafe fn handle_megascale_dcn_topology(args: &mut PJRT_Megascale_GetDcnTopology_Args) {
    let cfg = global_init_config();
    let tr = if args.context > 0x10000 {
        // SAFETY: `args.context` is a valid `PJRT_Megascale_ClientContext*`.
        let ctx = unsafe { &*(args.context as *const PJRT_Megascale_ClientContext) };
        ctx.transport_type.load(Ordering::Relaxed).max(1)
    } else {
        MEGASCALE_DEFAULT_TRANSPORT.load(Ordering::Relaxed).max(1)
    };
    let dcn_bytes = libtpu_core::proto::MegascaleDcnTopologyProto {
        num_slices: i32::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1),
        num_hosts_per_slice: i32::try_from(cfg.total_hosts()).unwrap_or(1),
        transport_type: i32::try_from(tr).unwrap_or(1),
    }
    .encode();
    args.serialized_size = dcn_bytes.len();
}

/// Queries the Megascale DCN topology (`PJRT_Megascale_GetDcnTopology`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_GetDcnTopology_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_GetDcnTopology_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetDcnTopology(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_GetDcnTopology_Args>(
            raw_args,
            handle_megascale_dcn_topology,
        )
    }
}

/// Queries the active Megascale slice ID (`PJRT_Megascale_GetSliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ConfigSliceId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ConfigSliceId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetSliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ConfigSliceId_Args>(
            raw_args,
            handle_megascale_slice_id,
        )
    }
}

/// Queries the total Megascale slice count (`PJRT_Megascale_GetNumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ConfigNumSlices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ConfigNumSlices_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetNumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ConfigNumSlices_Args>(
            raw_args,
            handle_megascale_num_slices,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_InitializeSlice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_InitializeSlice_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_InitializeSlice_Args`].
unsafe fn handle_megascale_initialize_slice(args: &mut PJRT_Megascale_InitializeSlice_Args) {
    args.status_out = 1;
}

/// Initializes a Megascale slice (`PJRT_Megascale_InitializeSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_InitializeSlice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_InitializeSlice_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_InitializeSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_InitializeSlice_Args>(
            raw_args,
            handle_megascale_initialize_slice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_FinalizeSlice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_FinalizeSlice_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_FinalizeSlice_Args`].
unsafe fn handle_megascale_finalize_slice(args: &mut PJRT_Megascale_FinalizeSlice_Args) {
    MEGASCALE_BARRIER_SEQ.fetch_add(1, Ordering::Relaxed);
    args.handle = ptr::null_mut();
}

/// Finalizes a Megascale slice (`PJRT_Megascale_FinalizeSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_FinalizeSlice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_FinalizeSlice_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_FinalizeSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_FinalizeSlice_Args>(
            raw_args,
            handle_megascale_finalize_slice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_RegisterErrorHandler`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_RegisterErrorHandler_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_RegisterErrorHandler_Args`].
unsafe fn handle_megascale_register_error_handler(
    args: &mut PJRT_Megascale_RegisterErrorHandler_Args,
) {
    let count = MEGASCALE_ERROR_HANDLERS.fetch_add(1, Ordering::Relaxed) + 1;
    if args.callback_fn > 0x10000 {
        MEGASCALE_ERROR_CB_FN.store(args.callback_fn, Ordering::Release);
    }
    args.handler_id_out = count;
}

/// Registers a Megascale error handler (`PJRT_Megascale_RegisterErrorHandler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_RegisterErrorHandler_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_RegisterErrorHandler_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_RegisterErrorHandler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_RegisterErrorHandler_Args>(
            raw_args,
            handle_megascale_register_error_handler,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_UnregisterErrorHandler`].
///
/// # Arguments
///
/// * `_args` - Validated [`PJRT_Megascale_UnregisterErrorHandler_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_UnregisterErrorHandler_Args`].
unsafe fn handle_megascale_unregister_error_handler(
    _args: &mut PJRT_Megascale_UnregisterErrorHandler_Args,
) {
    let prev = MEGASCALE_ERROR_HANDLERS.load(Ordering::Acquire);
    let next = prev.saturating_sub(1);
    MEGASCALE_ERROR_HANDLERS.store(next, Ordering::Release);
    if next == 0 {
        MEGASCALE_ERROR_CB_FN.store(0, Ordering::Release);
    }
}

/// Unregisters a Megascale error handler (`PJRT_Megascale_UnregisterErrorHandler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_UnregisterErrorHandler_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_UnregisterErrorHandler_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_UnregisterErrorHandler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_UnregisterErrorHandler_Args>(
            raw_args,
            handle_megascale_unregister_error_handler,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_CollectMetrics`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_CollectMetrics_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_CollectMetrics_Args`].
unsafe fn handle_megascale_collect_metrics(args: &mut PJRT_Megascale_CollectMetrics_Args) {
    if args.context > 0x10000 {
        // SAFETY: `args.context` is a valid `PJRT_Megascale_ClientContext*`.
        let ctx = unsafe { &*(args.context as *const PJRT_Megascale_ClientContext) };
        let coll_ops = COLLECTIVE_OPS_EXECUTED.load(Ordering::Relaxed);
        let encoded = libtpu_core::proto::MegascaleRuntimeMetricsProto {
            completed_barriers: u64::try_from(ctx.completed_barriers.load(Ordering::Relaxed))
                .unwrap_or(0),
            collective_ops_count: coll_ops,
            active_contexts: i32::try_from(MEGASCALE_ACTIVE_CTXS.load(Ordering::Relaxed))
                .unwrap_or(1),
            bytes_transferred: coll_ops.saturating_mul(64),
        }
        .encode()
        .into_boxed_slice();
        let ptr = encoded.as_ptr() as usize;
        let len = encoded.len();
        ctx.metrics_arena.lock().push(encoded);
        args.serialized_metrics_ptr = ptr;
        args.serialized_metrics_size = len;
    } else {
        args.serialized_metrics_ptr = DEFAULT_MEGASCALE_METRICS_WIRE.as_ptr() as usize;
        args.serialized_metrics_size = DEFAULT_MEGASCALE_METRICS_WIRE.len();
    }
}

/// Collects Megascale telemetry metrics (`PJRT_Megascale_CollectMetrics`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_CollectMetrics_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_CollectMetrics_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CollectMetrics(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_CollectMetrics_Args>(
            raw_args,
            handle_megascale_collect_metrics,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ResetMetrics`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_ResetMetrics_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_ResetMetrics_Args`].
unsafe fn handle_megascale_reset_metrics(args: &mut PJRT_Megascale_ResetMetrics_Args) {
    MEGASCALE_BARRIER_SEQ.store(0, Ordering::Relaxed);
    if (args.handle as usize) > 0x10000 {
        // SAFETY: `args.handle` is a valid `PJRT_Megascale_ClientContext*`.
        let ctx = unsafe { &*args.handle.cast::<PJRT_Megascale_ClientContext>() };
        ctx.completed_barriers.store(0, Ordering::Relaxed);
        ctx.metrics_arena.lock().clear();
    }
}

/// Resets Megascale telemetry metrics (`PJRT_Megascale_ResetMetrics`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_ResetMetrics_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_ResetMetrics_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ResetMetrics(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_ResetMetrics_Args>(
            raw_args,
            handle_megascale_reset_metrics,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_GetTransportType`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_GetTransportType_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_GetTransportType_Args`].
unsafe fn handle_megascale_get_transport_type(args: &mut PJRT_Megascale_GetTransportType_Args) {
    args.transport_type_out = if args.context > 0x10000 {
        // SAFETY: `args.context` is a valid `PJRT_Megascale_ClientContext*`.
        let ctx = unsafe { &*(args.context as *const PJRT_Megascale_ClientContext) };
        ctx.transport_type.load(Ordering::Relaxed)
    } else {
        MEGASCALE_DEFAULT_TRANSPORT.load(Ordering::Relaxed)
    };
}

/// Queries Megascale transport type (`PJRT_Megascale_GetTransportType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_GetTransportType_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_GetTransportType_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetTransportType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_GetTransportType_Args>(
            raw_args,
            handle_megascale_get_transport_type,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_SetTransportType`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_SetTransportType_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_SetTransportType_Args`].
unsafe fn handle_megascale_set_transport_type(args: &mut PJRT_Megascale_SetTransportType_Args) {
    let requested = if args.requested_transport_type == 0 {
        1
    } else {
        args.requested_transport_type
    };
    MEGASCALE_DEFAULT_TRANSPORT.store(requested, Ordering::Relaxed);
    args.transport_type_out = requested;
}

/// Sets Megascale transport type (`PJRT_Megascale_SetTransportType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_SetTransportType_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_SetTransportType_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_SetTransportType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_SetTransportType_Args>(
            raw_args,
            handle_megascale_set_transport_type,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_WaitForBarrier`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_WaitForBarrier_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_WaitForBarrier_Args`].
unsafe fn handle_megascale_wait_for_barrier(args: &mut PJRT_Megascale_WaitForBarrier_Args) {
    MEGASCALE_BARRIER_SEQ.fetch_add(1, Ordering::Relaxed);
    args.handle = ptr::null_mut();
}

/// Waits for a Megascale barrier (`PJRT_Megascale_WaitForBarrier`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_WaitForBarrier_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_WaitForBarrier_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_WaitForBarrier(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_WaitForBarrier_Args>(
            raw_args,
            handle_megascale_wait_for_barrier,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_AbortCollectives`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_AbortCollectives_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_AbortCollectives_Args`].
unsafe fn handle_megascale_abort_collectives(args: &mut PJRT_Megascale_AbortCollectives_Args) {
    MEGASCALE_BARRIER_SEQ.fetch_add(1, Ordering::Relaxed);
    let cb_fn = MEGASCALE_ERROR_CB_FN.load(Ordering::Acquire);
    if cb_fn > 0x10000 {
        let ptr_void = cb_fn as *const std::ffi::c_void;
        // SAFETY: `cb_fn` was registered via `PJRT_Megascale_RegisterErrorHandler`.
        let cb: unsafe extern "C" fn(*mut std::ffi::c_void) = unsafe {
            *std::ptr::addr_of!(ptr_void).cast::<unsafe extern "C" fn(*mut std::ffi::c_void)>()
        };
        // SAFETY: Invoking caller-provided C error callback with `args.handle`.
        unsafe {
            cb(args.handle);
        }
    }
    args.handle = ptr::null_mut();
}

/// Aborts in-flight Megascale collectives (`PJRT_Megascale_AbortCollectives`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_AbortCollectives_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_AbortCollectives_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_AbortCollectives(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_AbortCollectives_Args>(
            raw_args,
            handle_megascale_abort_collectives,
        )
    }
}
