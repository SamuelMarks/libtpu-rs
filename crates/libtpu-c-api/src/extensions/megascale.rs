//! PJRT Multi-Slice (`PJRT_MultiSlice_Extension`) and Megascale (`PJRT_Megascale_Extension`) entrypoints.

use super::args::*;
use super::{alloc_extension_bytes, dispatch_typed_extension_args, extension_state};
use crate::pjrt_helpers::box_serialized_artifact;
use crate::pjrt_types::{PJRT_Error, PJRT_Megascale_ClientContext, PJRT_MultiSlice_Config};
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
    let mut st = extension_state().lock();
    st.megascale_contexts = st.megascale_contexts.saturating_add(1);
    let ctx = Box::into_raw(Box::new(PJRT_Megascale_ClientContext {
        context_id: u64::try_from(st.megascale_contexts).unwrap_or(1),
        transport_type: st.megascale_transport_type,
    }));
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
    }
    let mut st = extension_state().lock();
    st.megascale_contexts = st.megascale_contexts.saturating_sub(1);
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
    let mut st = extension_state().lock();
    st.megascale_configs = st.megascale_configs.saturating_add(1);
    drop(st);
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
    let mut st = extension_state().lock();
    st.megascale_configs = st.megascale_configs.saturating_sub(1);
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
    }
    .encode();
    let (_ptr, _len) = alloc_extension_bytes(addr_bytes);
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
    let mut st = extension_state().lock();
    st.megascale_barriers = st.megascale_barriers.saturating_add(1);
    args.completed_barriers = st.megascale_barriers;
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
    let cb_fn = {
        let mut st = extension_state().lock();
        st.megascale_error_handlers = st.megascale_error_handlers.saturating_add(1);
        st.megascale_barriers = st.megascale_barriers.saturating_add(1);
        st.megascale_error_callback_fn
    };
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
    let transport =
        i32::try_from(extension_state().lock().megascale_transport_type.max(1)).unwrap_or(1);
    let dcn_bytes = libtpu_core::proto::MegascaleDcnTopologyProto {
        num_slices: i32::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1),
        num_hosts_per_slice: i32::try_from(cfg.total_hosts()).unwrap_or(1),
        transport_type: transport,
    }
    .encode();
    let (_ptr, len) = alloc_extension_bytes(dcn_bytes);
    args.serialized_size = len;
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
    let mut st = extension_state().lock();
    st.megascale_barriers = st.megascale_barriers.saturating_add(1);
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
    let mut st = extension_state().lock();
    st.megascale_error_handlers = st.megascale_error_handlers.saturating_add(1);
    if args.callback_fn > 0x10000 {
        st.megascale_error_callback_fn = args.callback_fn;
    }
    args.handler_id_out = st.megascale_error_handlers;
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
    let mut st = extension_state().lock();
    st.megascale_error_handlers = st.megascale_error_handlers.saturating_sub(1);
    if st.megascale_error_handlers == 0 {
        st.megascale_error_callback_fn = 0;
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
    let encoded = {
        let st = extension_state().lock();
        libtpu_core::proto::MegascaleRuntimeMetricsProto {
            completed_barriers: u64::try_from(st.megascale_barriers).unwrap_or(0),
            collective_ops_count: u64::try_from(st.collective_ops_executed).unwrap_or(0),
            active_contexts: i32::try_from(st.megascale_contexts).unwrap_or(0),
            bytes_transferred: u64::try_from(st.collective_ops_executed)
                .unwrap_or(0)
                .saturating_mul(64),
        }
        .encode()
    };
    let (ptr, len) = alloc_extension_bytes(encoded);
    args.serialized_metrics_ptr = ptr as usize;
    args.serialized_metrics_size = len;
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
/// * `_args` - Validated [`PJRT_Megascale_ResetMetrics_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_ResetMetrics_Args`].
unsafe fn handle_megascale_reset_metrics(_args: &mut PJRT_Megascale_ResetMetrics_Args) {
    let mut st = extension_state().lock();
    st.megascale_barriers = 0;
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
    args.transport_type_out = extension_state().lock().megascale_transport_type;
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
    let mut st = extension_state().lock();
    let requested = if args.requested_transport_type == 0 {
        1
    } else {
        args.requested_transport_type
    };
    st.megascale_transport_type = requested;
    args.transport_type_out = st.megascale_transport_type;
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
    let mut st = extension_state().lock();
    st.megascale_barriers = st.megascale_barriers.saturating_add(1);
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
    let cb_fn = {
        let mut st = extension_state().lock();
        st.megascale_barriers = st.megascale_barriers.saturating_add(1);
        st.megascale_error_callback_fn
    };
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
