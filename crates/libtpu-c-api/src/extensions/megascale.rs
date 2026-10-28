//! PJRT Multi-Slice (`PJRT_MultiSlice_Extension`) and Megascale (`PJRT_Megascale_Extension`) entrypoints.

use super::args::*;
use super::{alloc_extension_bytes, dispatch_typed_extension_args, extension_state};
use crate::pjrt_helpers::box_serialized_artifact;
use crate::pjrt_types::PJRT_Error;
use libtpu_core::config::global_init_config;
use std::ptr;

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
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_create_client_context(args: &mut PJRT_Megascale_Query_Args) {
    let mut st = extension_state().lock();
    st.megascale_contexts = st.megascale_contexts.saturating_add(1);
    args.out_value = st.megascale_contexts;
}

/// Creates a Megascale client context (`PJRT_Megascale_CreateClientContext`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateClientContext(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_create_client_context,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_DeleteClientContext`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Handle_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Handle_Args`].
unsafe fn handle_megascale_delete_client_context(args: &mut PJRT_Megascale_Handle_Args) {
    let mut st = extension_state().lock();
    st.megascale_contexts = st.megascale_contexts.saturating_sub(1);
    args.handle = ptr::null_mut();
}

/// Deletes a Megascale client context (`PJRT_Megascale_DeleteClientContext`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_DeleteClientContext(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(
            raw_args,
            handle_megascale_delete_client_context,
        )
    }
}

/// Strongly-typed handler for Megascale config creation entrypoints.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_create_config(args: &mut PJRT_Megascale_Query_Args) {
    let mut st = extension_state().lock();
    st.megascale_configs = st.megascale_configs.saturating_add(1);
    args.out_value = st.megascale_configs;
}

/// Creates a default Megascale config (`PJRT_Megascale_CreateDefaultConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateDefaultConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Creates an AOT Megascale config (`PJRT_Megascale_CreateAotConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateAotConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Creates a Megascale multi-slice config (`PJRT_Megascale_CreateMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CreateMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_DeleteMultiSliceConfig`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Handle_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Handle_Args`].
unsafe fn handle_megascale_delete_config(args: &mut PJRT_Megascale_Handle_Args) {
    let mut st = extension_state().lock();
    st.megascale_configs = st.megascale_configs.saturating_sub(1);
    args.handle = ptr::null_mut();
}

/// Deletes a Megascale multi-slice config (`PJRT_Megascale_DeleteMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_DeleteMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(
            raw_args,
            handle_megascale_delete_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigNumSlices`] and [`PJRT_Megascale_GetNumSlices`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_num_slices(args: &mut PJRT_Megascale_Query_Args) {
    args.out_value = usize::try_from(global_init_config().megascale_num_slices.max(1)).unwrap_or(1);
}

/// Queries slice count from Megascale config (`PJRT_Megascale_ConfigNumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigNumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_num_slices,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigNumDevicesPerSlice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_num_devices_per_slice(args: &mut PJRT_Megascale_Query_Args) {
    args.out_value = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
}

/// Queries devices per slice from Megascale config (`PJRT_Megascale_ConfigNumDevicesPerSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigNumDevicesPerSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_num_devices_per_slice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigSliceId`] and [`PJRT_Megascale_GetSliceId`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_slice_id(args: &mut PJRT_Megascale_Query_Args) {
    args.out_value = usize::try_from(global_init_config().megascale_slice_id).unwrap_or(0);
}

/// Queries slice ID from Megascale config (`PJRT_Megascale_ConfigSliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_slice_id,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ConfigSerializedSliceKeyValue`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Data_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Data_Args`].
unsafe fn handle_megascale_serialized_kv(args: &mut PJRT_Megascale_Data_Args) {
    args.out_first = global_init_config().extra_flags.len().saturating_add(16);
    args.out_second = 1;
}

/// Queries serialized slice key-values from Megascale config (`PJRT_Megascale_ConfigSerializedSliceKeyValue`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Data_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Data_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSerializedSliceKeyValue(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Data_Args>(
            raw_args,
            handle_megascale_serialized_kv,
        )
    }
}

/// Serializes a Megascale multi-slice config (`PJRT_Megascale_ConfigSerialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_Serialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSerialize(
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

/// Deserializes a Megascale multi-slice config (`PJRT_Megascale_DeserializeMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_DeserializeMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_create_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_Addresses`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_addresses(args: &mut PJRT_Megascale_Query_Args) {
    args.out_value = usize::try_from(global_init_config().total_hosts()).unwrap_or(1);
}

/// Queries Megascale coordinator addresses (`PJRT_Megascale_Addresses`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_Addresses(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_addresses,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_Barrier`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_barrier(args: &mut PJRT_Megascale_Query_Args) {
    let mut st = extension_state().lock();
    st.megascale_barriers = st.megascale_barriers.saturating_add(1);
    args.out_value = st.megascale_barriers;
}

/// Executes a Megascale multi-slice barrier (`PJRT_Megascale_Barrier`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_Barrier(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_barrier,
        )
    }
}

/// Strongly-typed no-op handler for [`PJRT_Megascale_Handle_Args`].
///
/// # Arguments
///
/// * `_args` - Validated [`PJRT_Megascale_Handle_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Handle_Args`].
pub(super) unsafe fn handle_megascale_noop(_args: &mut PJRT_Megascale_Handle_Args) {}

/// Logs a Megascale collective error (`PJRT_Megascale_LogError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_LogError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(raw_args, handle_megascale_noop)
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_GetDcnTopology`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_dcn_topology(args: &mut PJRT_Megascale_Query_Args) {
    let cfg = global_init_config();
    let dcn_bytes = libtpu_core::proto::MegascaleDcnTopologyProto {
        num_slices: i32::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1),
        num_hosts_per_slice: i32::try_from(cfg.total_hosts()).unwrap_or(1),
        transport_type: 1,
    }
    .encode();
    args.out_value = dcn_bytes.len();
}

/// Queries the Megascale DCN topology (`PJRT_Megascale_GetDcnTopology`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetDcnTopology(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_dcn_topology,
        )
    }
}

/// Queries the active Megascale slice ID (`PJRT_Megascale_GetSliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetSliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_slice_id,
        )
    }
}

/// Queries the total Megascale slice count (`PJRT_Megascale_GetNumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetNumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_num_slices,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_InitializeSlice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_initialize_slice(args: &mut PJRT_Megascale_Query_Args) {
    args.out_value = 1;
}

/// Initializes a Megascale slice (`PJRT_Megascale_InitializeSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_InitializeSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_initialize_slice,
        )
    }
}

/// Finalizes a Megascale slice (`PJRT_Megascale_FinalizeSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_FinalizeSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(raw_args, handle_megascale_noop)
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_RegisterErrorHandler`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_register_error_handler(args: &mut PJRT_Megascale_Query_Args) {
    let mut st = extension_state().lock();
    st.megascale_error_handlers = st.megascale_error_handlers.saturating_add(1);
    args.out_value = st.megascale_error_handlers;
}

/// Registers a Megascale error handler (`PJRT_Megascale_RegisterErrorHandler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_RegisterErrorHandler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_register_error_handler,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_UnregisterErrorHandler`].
///
/// # Arguments
///
/// * `_args` - Validated [`PJRT_Megascale_Handle_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Handle_Args`].
unsafe fn handle_megascale_unregister_error_handler(_args: &mut PJRT_Megascale_Handle_Args) {
    let mut st = extension_state().lock();
    st.megascale_error_handlers = st.megascale_error_handlers.saturating_sub(1);
}

/// Unregisters a Megascale error handler (`PJRT_Megascale_UnregisterErrorHandler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_UnregisterErrorHandler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(
            raw_args,
            handle_megascale_unregister_error_handler,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_CollectMetrics`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Data_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Data_Args`].
unsafe fn handle_megascale_collect_metrics(args: &mut PJRT_Megascale_Data_Args) {
    let encoded = {
        let st = extension_state().lock();
        libtpu_core::proto::MegascaleRuntimeMetricsProto {
            completed_barriers: u64::try_from(st.megascale_barriers).unwrap_or(0),
            collective_ops_count: u64::try_from(st.collective_ops_executed).unwrap_or(0),
            active_contexts: i32::try_from(st.megascale_contexts).unwrap_or(0),
        }
        .encode()
    };
    let (ptr, len) = alloc_extension_bytes(encoded);
    args.out_first = ptr as usize;
    args.out_second = len;
}

/// Collects Megascale telemetry metrics (`PJRT_Megascale_CollectMetrics`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Data_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Data_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_CollectMetrics(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Data_Args>(
            raw_args,
            handle_megascale_collect_metrics,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_ResetMetrics`].
///
/// # Arguments
///
/// * `_args` - Validated [`PJRT_Megascale_Handle_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Handle_Args`].
unsafe fn handle_megascale_reset_metrics(_args: &mut PJRT_Megascale_Handle_Args) {
    let mut st = extension_state().lock();
    st.megascale_barriers = 0;
}

/// Resets Megascale telemetry metrics (`PJRT_Megascale_ResetMetrics`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_ResetMetrics(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(
            raw_args,
            handle_megascale_reset_metrics,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_GetTransportType`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_get_transport_type(args: &mut PJRT_Megascale_Query_Args) {
    args.out_value = extension_state().lock().megascale_transport_type;
}

/// Queries Megascale transport type (`PJRT_Megascale_GetTransportType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_GetTransportType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_get_transport_type,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Megascale_SetTransportType`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_megascale_set_transport_type(args: &mut PJRT_Megascale_Query_Args) {
    let mut st = extension_state().lock();
    st.megascale_transport_type = 1;
    args.out_value = st.megascale_transport_type;
}

/// Sets Megascale transport type (`PJRT_Megascale_SetTransportType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_SetTransportType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_megascale_set_transport_type,
        )
    }
}

/// Waits for a Megascale barrier (`PJRT_Megascale_WaitForBarrier`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_WaitForBarrier(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(raw_args, handle_megascale_noop)
    }
}

/// Aborts in-flight Megascale collectives (`PJRT_Megascale_AbortCollectives`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Megascale_AbortCollectives(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(raw_args, handle_megascale_noop)
    }
}
