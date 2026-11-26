//! PJRT TPU Topology (`PJRT_TpuTopology_Extension`) entrypoints.

use super::args::*;
use super::dispatch_typed_extension_args;
use crate::pjrt_types::{PJRT_Error, PJRT_TopologyDescription};
use libtpu_core::config::global_init_config;
use libtpu_core::hal::TopologySpec;
use std::sync::atomic::Ordering;

/// Resolves a typed `topology` pointer into an optional [`PJRT_TopologyDescription`] reference.
///
/// # Arguments
///
/// * `ptr` - Typed `PJRT_TopologyDescription*` handle from a topology extension args struct.
///
/// # Safety
///
/// If non-null, `ptr` must point to a valid [`PJRT_TopologyDescription`].
unsafe fn resolve_topology_handle<'a>(
    ptr: *const PJRT_TopologyDescription,
) -> Option<&'a PJRT_TopologyDescription> {
    // SAFETY: Caller guarantees non-null `ptr` points to a valid `PJRT_TopologyDescription`.
    unsafe { ptr.as_ref() }
}

/// Resolves a typed `topology` pointer from a `PJRT_TpuTopology_*` argument structure into an
/// optional [`TopologySpec`] reference.
///
/// # Arguments
///
/// * `ptr` - Typed `PJRT_TopologyDescription*` handle from a topology extension args struct.
///
/// # Safety
///
/// If non-null, `ptr` must point to a valid [`PJRT_TopologyDescription`].
unsafe fn resolve_topology_spec<'a>(
    ptr: *const PJRT_TopologyDescription,
) -> Option<&'a TopologySpec> {
    // SAFETY: Caller upholds `ptr` precondition.
    unsafe { resolve_topology_handle(ptr).map(|t| &t.inner.spec) }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_Subslice`] and [`PJRT_TpuTopology_ReplaceHostBounds`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_subslice(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    // SAFETY: `args.topology` is null/sentinel or a valid `PJRT_TopologyDescription*`.
    if let Some(topo) = unsafe { resolve_topology_handle(args.topology) } {
        topo.is_subslice.store(true, Ordering::Release);
    }
    args.out_value = 1;
}

/// Creates a subslice TPU topology (`PJRT_TpuTopology_Subslice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_Subslice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_subslice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_IsSubsliceTopology`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_IsSubsliceTopology_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_IsSubsliceTopology_Args`].
unsafe fn handle_tpu_topology_is_subslice(args: &mut PJRT_TpuTopology_IsSubsliceTopology_Args) {
    // SAFETY: `args.topology` is null/sentinel or a valid `PJRT_TopologyDescription*`.
    let is_sub = unsafe { resolve_topology_handle(args.topology) }
        .is_some_and(|t| t.is_subslice.load(Ordering::Acquire));
    args.is_subslice = usize::from(is_sub);
}

/// Checks whether a topology is a subslice (`PJRT_TpuTopology_IsSubsliceTopology`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_IsSubsliceTopology_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_IsSubsliceTopology_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_IsSubsliceTopology(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_IsSubsliceTopology_Args>(
            raw_args,
            handle_tpu_topology_is_subslice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_GetRoutingStrategy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_GetRoutingStrategy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_GetRoutingStrategy_Args`].
unsafe fn handle_tpu_topology_routing_strategy(
    args: &mut PJRT_TpuTopology_GetRoutingStrategy_Args,
) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let z_dim =
        unsafe { resolve_topology_spec(args.topology) }.map_or(1, |spec| spec.chip_bounds.z);
    args.routing_strategy = usize::from(z_dim > 1);
}

/// Strongly-typed handler for one-returning [`PJRT_TpuTopology_ScalarQuery_Args`] entrypoints.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_one(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    args.out_value = 1;
}

/// Strongly-typed handler for [`PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_subslice_device_id(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let per_host = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || {
            let cfg = global_init_config();
            usize::try_from(
                cfg.chips_per_host().max(1) * cfg.generation.logical_devices_per_chip().max(1),
            )
            .unwrap_or(4)
        },
        |spec| {
            usize::try_from(
                spec.chips_per_host.max(1) * spec.generation.logical_devices_per_chip().max(1),
            )
            .unwrap_or(4)
        },
    );
    args.out_value %= per_host;
}

/// Maps full device ID to subslice device ID (`PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_subslice_device_id,
        )
    }
}

/// Replaces host bounds on a TPU topology (`PJRT_TpuTopology_ReplaceHostBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ReplaceHostBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_subslice,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_IsEnhancedBarrierEnabled`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_enhanced_barrier(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    args.out_value = usize::from(global_init_config().allow_async_allocations);
}

/// Checks whether enhanced ICI barrier is enabled (`PJRT_TpuTopology_IsEnhancedBarrierEnabled`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_IsEnhancedBarrierEnabled(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_enhanced_barrier,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_HasLimitedIciConnectivity`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_HasLimitedIciConnectivity_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_HasLimitedIciConnectivity_Args`].
unsafe fn handle_tpu_topology_has_limited_ici(
    args: &mut PJRT_TpuTopology_HasLimitedIciConnectivity_Args,
) {
    let tpu_gen = global_init_config().generation;
    // SAFETY: `args.topology` is null/sentinel or a valid `PJRT_TopologyDescription*`.
    let is_sub = unsafe { resolve_topology_handle(args.topology) }
        .is_some_and(|t| t.is_subslice.load(Ordering::Acquire));
    args.has_limited_ici = usize::from(
        matches!(
            tpu_gen,
            libtpu_core::types::TpuGeneration::V5e | libtpu_core::types::TpuGeneration::V6e
        ) && is_sub,
    );
}

/// Checks whether topology has limited ICI connectivity (`PJRT_TpuTopology_HasLimitedIciConnectivity`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_HasLimitedIciConnectivity_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_HasLimitedIciConnectivity_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_HasLimitedIciConnectivity(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_HasLimitedIciConnectivity_Args>(
            raw_args,
            handle_tpu_topology_has_limited_ici,
        )
    }
}

/// Checks whether two chips are reachable over limited ICI (`PJRT_TpuTopology_IsReachableOverLimitedIci`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_IsReachableOverLimitedIci(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_one,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcessCount`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ProcessCount_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ProcessCount_Args`].
unsafe fn handle_tpu_topology_process_count(args: &mut PJRT_TpuTopology_ProcessCount_Args) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let hosts = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || global_init_config().total_hosts(),
        |spec| spec.host_bounds.x * spec.host_bounds.y * spec.host_bounds.z,
    );
    args.process_count = usize::try_from(hosts.max(1)).unwrap_or(1);
}

/// Queries process count in TPU topology (`PJRT_TpuTopology_ProcessCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ProcessCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ProcessCount_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ProcessCount_Args>(
            raw_args,
            handle_tpu_topology_process_count,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ChipsPerProcess`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ChipsPerProcess_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ChipsPerProcess_Args`].
unsafe fn handle_tpu_topology_chips_per_process(args: &mut PJRT_TpuTopology_ChipsPerProcess_Args) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let chips = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || global_init_config().chips_per_host(),
        |spec| spec.chips_per_host,
    );
    args.chips_per_process = usize::try_from(chips.max(1)).unwrap_or(4);
}

/// Queries chips per process in TPU topology (`PJRT_TpuTopology_ChipsPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ChipsPerProcess_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ChipsPerProcess_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipsPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ChipsPerProcess_Args>(
            raw_args,
            handle_tpu_topology_chips_per_process,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_CoreCountPerChip`] and [`PJRT_TpuTopology_LogiDeviceCountPerChip`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_CoreCountPerChip_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_CoreCountPerChip_Args`].
unsafe fn handle_tpu_topology_cores_per_chip(args: &mut PJRT_TpuTopology_CoreCountPerChip_Args) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let cores = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || global_init_config().generation.logical_devices_per_chip(),
        |spec| spec.generation.logical_devices_per_chip(),
    );
    args.core_count_per_chip = usize::try_from(cores.max(1)).unwrap_or(1);
}

/// Queries cores per chip in TPU topology (`PJRT_TpuTopology_CoreCountPerChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_CoreCountPerChip_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_CoreCountPerChip_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCountPerChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_CoreCountPerChip_Args>(
            raw_args,
            handle_tpu_topology_cores_per_chip,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ChipCount`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ChipCount_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ChipCount_Args`].
unsafe fn handle_tpu_topology_total_chips(args: &mut PJRT_TpuTopology_ChipCount_Args) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let total_chips = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || {
            let cfg = global_init_config();
            cfg.chips_per_host() * cfg.total_hosts()
        },
        |spec| spec.chip_bounds.x * spec.chip_bounds.y * spec.chip_bounds.z,
    );
    args.chip_count = usize::try_from(total_chips.max(1)).unwrap_or(4);
}

/// Strongly-typed handler for [`PJRT_TpuTopology_CoreCount`] and [`PJRT_TpuTopology_LogiDeviceCount`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_CoreCount_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_CoreCount_Args`].
unsafe fn handle_tpu_topology_total_cores(args: &mut PJRT_TpuTopology_CoreCount_Args) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let total_cores = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || {
            let cfg = global_init_config();
            usize::try_from(
                cfg.chips_per_host()
                    * cfg.total_hosts()
                    * cfg.generation.logical_devices_per_chip(),
            )
            .unwrap_or(4)
        },
        |spec| spec.devices.len().max(1),
    );
    args.core_count = total_cores;
}

/// Strongly-typed handler for [`PJRT_TpuTopology_LogiDeviceCountPerProcess`] and [`PJRT_TpuTopology_CoreCountPerProcess`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_CoreCountPerProcess_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_CoreCountPerProcess_Args`].
unsafe fn handle_tpu_topology_cores_per_process(
    args: &mut PJRT_TpuTopology_CoreCountPerProcess_Args,
) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let cores_per_proc = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || {
            let cfg = global_init_config();
            cfg.chips_per_host() * cfg.generation.logical_devices_per_chip()
        },
        |spec| spec.chips_per_host * spec.generation.logical_devices_per_chip(),
    );
    args.core_count_per_process = usize::try_from(cores_per_proc.max(1)).unwrap_or(4);
}

/// Queries total chip count in TPU topology (`PJRT_TpuTopology_ChipCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ChipCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ChipCount_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ChipCount_Args>(
            raw_args,
            handle_tpu_topology_total_chips,
        )
    }
}

/// Queries total core count in TPU topology (`PJRT_TpuTopology_CoreCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_CoreCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_CoreCount_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_CoreCount_Args>(
            raw_args,
            handle_tpu_topology_total_cores,
        )
    }
}

/// Queries logical devices per process (`PJRT_TpuTopology_LogiDeviceCountPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_CoreCountPerProcess_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_CoreCountPerProcess_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCountPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_CoreCountPerProcess_Args>(
            raw_args,
            handle_tpu_topology_cores_per_process,
        )
    }
}

/// Queries total logical device count (`PJRT_TpuTopology_LogiDeviceCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_CoreCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_CoreCount_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_CoreCount_Args>(
            raw_args,
            handle_tpu_topology_total_cores,
        )
    }
}

/// Queries logical devices per chip (`PJRT_TpuTopology_LogiDeviceCountPerChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_CoreCountPerChip_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_CoreCountPerChip_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCountPerChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_CoreCountPerChip_Args>(
            raw_args,
            handle_tpu_topology_cores_per_chip,
        )
    }
}

/// Queries cores per process (`PJRT_TpuTopology_CoreCountPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_CoreCountPerProcess_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_CoreCountPerProcess_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCountPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_CoreCountPerProcess_Args>(
            raw_args,
            handle_tpu_topology_cores_per_process,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcessIds`], populating a caller `int32_t*` array
/// when `process_ids_out` is non-null.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ProcessIds_Args`] structure.
///
/// # Safety
///
/// If `args.process_ids_out` is non-null, it must point to at least `process_count` `i32` slots.
unsafe fn handle_tpu_topology_process_ids(args: &mut PJRT_TpuTopology_ProcessIds_Args) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let hosts = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || global_init_config().total_hosts(),
        |spec| spec.host_bounds.x * spec.host_bounds.y * spec.host_bounds.z,
    );
    let count = usize::try_from(hosts.max(1)).unwrap_or(1);
    if !args.process_ids_out.is_null() {
        for idx in 0..count {
            // SAFETY: `process_ids_out` points to a caller-supplied `int32_t` buffer of at least `count` elements.
            unsafe {
                *args.process_ids_out.add(idx) = i32::try_from(idx).unwrap_or(0);
            }
        }
    }
}

/// Queries process ID list (`PJRT_TpuTopology_ProcessIds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ProcessIds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ProcessIds_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ProcessIds_Args>(
            raw_args,
            handle_tpu_topology_process_ids,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_LogiDeviceIdsOnProcess`], populating a caller
/// `int32_t*` array when `device_ids_out` is non-null.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args`] structure.
///
/// # Safety
///
/// If `args.device_ids_out` is non-null, it must point to at least `core_count_per_process` `i32` slots.
unsafe fn handle_tpu_topology_logi_device_ids_on_process(
    args: &mut PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args,
) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let spec_opt = unsafe { resolve_topology_spec(args.topology) };
    let cores_per_proc = spec_opt.map_or_else(
        || {
            let cfg = global_init_config();
            cfg.chips_per_host() * cfg.generation.logical_devices_per_chip()
        },
        |spec| spec.chips_per_host * spec.generation.logical_devices_per_chip(),
    );
    let count = usize::try_from(cores_per_proc.max(1)).unwrap_or(4);
    if !args.device_ids_out.is_null() {
        for idx in 0..count {
            let fallback_id = i32::try_from(idx).unwrap_or(0);
            let dev_id = spec_opt
                .and_then(|s| s.devices.get(idx))
                .and_then(|d| i32::try_from(d.device_id.0).ok())
                .unwrap_or(fallback_id);
            // SAFETY: `device_ids_out` points to a caller-supplied `int32_t` buffer of at least `count` elements.
            unsafe {
                *args.device_ids_out.add(idx) = dev_id;
            }
        }
    }
}

/// Queries logical device IDs on a process (`PJRT_TpuTopology_LogiDeviceIdsOnProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceIdsOnProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args>(
            raw_args,
            handle_tpu_topology_logi_device_ids_on_process,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args`].
unsafe fn handle_tpu_topology_proc_for_chip(
    args: &mut PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args,
) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let chips_per_proc = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || usize::try_from(global_init_config().chips_per_host().max(1)).unwrap_or(4),
        |spec| usize::try_from(spec.chips_per_host.max(1)).unwrap_or(4),
    );
    args.out_proc_id = args.chip_id / chips_per_proc;
    args.out_idx_on_proc = args.chip_id % chips_per_proc;
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args`].
unsafe fn handle_tpu_topology_proc_for_device(
    args: &mut PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args,
) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let spec_opt = unsafe { resolve_topology_spec(args.topology) };
    let devs_per_proc = spec_opt.map_or_else(
        || {
            let cfg = global_init_config();
            usize::try_from(cfg.chips_per_host().max(1)).unwrap_or(4)
        },
        |spec| {
            usize::try_from(
                (spec.chips_per_host * spec.generation.logical_devices_per_chip()).max(1),
            )
            .unwrap_or(4)
        },
    );
    if let Some(dev) = spec_opt.and_then(|s| s.devices.get(args.device_id)) {
        args.out_proc_id = usize::try_from(dev.process_index.0.max(0)).unwrap_or(0);
    } else {
        args.out_proc_id = args.device_id / devs_per_proc;
    }
    args.out_idx_on_proc = args.device_id % devs_per_proc;
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcessCoordFromId`], writing `int32_t[3]`
/// `(x, y, z)` coordinates when `coords_out` is non-null.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ProcessCoordFromId_Args`] structure.
///
/// # Safety
///
/// If `args.coords_out` is non-null, it must point to at least 3 `i32` slots.
unsafe fn handle_tpu_topology_process_coord_from_id(
    args: &mut PJRT_TpuTopology_ProcessCoordFromId_Args,
) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let hb = unsafe { resolve_topology_spec(args.topology) }
        .map_or_else(|| global_init_config().host_bounds, |spec| spec.host_bounds);
    let hx = usize::try_from(hb.x.max(1)).unwrap_or(1);
    let hy = usize::try_from(hb.y.max(1)).unwrap_or(1);
    if !args.coords_out.is_null() {
        let x = i32::try_from(args.process_id % hx).unwrap_or(0);
        let y = i32::try_from((args.process_id / hx) % hy).unwrap_or(0);
        let z = i32::try_from(args.process_id / (hx * hy)).unwrap_or(0);
        // SAFETY: `coords_out` points to an `int32_t[3]` coordinate array.
        unsafe {
            *args.coords_out = x;
            *args.coords_out.add(1) = y;
            *args.coords_out.add(2) = z;
        }
    }
    args.num_coords = 3;
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ChipIdFromCoord`], linearizing `const int32_t[3]`
/// chip coordinates when `coords` is non-null.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ChipIdFromCoord_Args`] structure.
///
/// # Safety
///
/// If `args.coords` is non-null, it must point to at least 3 `i32` values.
unsafe fn handle_tpu_topology_chip_id_from_coord(args: &mut PJRT_TpuTopology_ChipIdFromCoord_Args) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let spec_opt = unsafe { resolve_topology_spec(args.topology) };
    let cb = spec_opt.map_or_else(
        || global_init_config().chips_per_host_bounds,
        |spec| spec.chip_bounds,
    );
    let cx = usize::try_from(cb.x.max(1)).unwrap_or(2);
    let cy = usize::try_from(cb.y.max(1)).unwrap_or(2);
    let chips_per_proc = usize::try_from(
        spec_opt
            .map_or_else(
                || global_init_config().chips_per_host(),
                |spec| spec.chips_per_host,
            )
            .max(1),
    )
    .unwrap_or(4);
    let chip_id = if !args.coords.is_null() {
        // SAFETY: `coords` points to `const int32_t[3]`.
        let (x, y, z) = unsafe {
            (
                usize::try_from((*args.coords).max(0)).unwrap_or(0),
                usize::try_from((*args.coords.add(1)).max(0)).unwrap_or(0),
                usize::try_from((*args.coords.add(2)).max(0)).unwrap_or(0),
            )
        };
        z * (cx * cy) + y * cx + x
    } else {
        0
    };
    args.chip_id_out = chip_id;
    args.idx_on_proc_out = chip_id % chips_per_proc;
}

/// Strongly-typed handler for [`PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args`] structure.
///
/// # Safety
///
/// If `args.coords` is non-null, it must point to at least 3 `i32` values.
unsafe fn handle_tpu_topology_logi_device_from_coord(
    args: &mut PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args,
) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let spec_opt = unsafe { resolve_topology_spec(args.topology) };
    let (x, y, z) = if !args.coords.is_null() {
        // SAFETY: `coords` points to `const int32_t[3]`.
        unsafe {
            (
                u32::try_from((*args.coords).max(0)).unwrap_or(0),
                u32::try_from((*args.coords.add(1)).max(0)).unwrap_or(0),
                u32::try_from((*args.coords.add(2)).max(0)).unwrap_or(0),
            )
        }
    } else {
        (0, 0, 0)
    };
    let core_idx = u32::try_from(args.core_idx).unwrap_or(0);
    let found = spec_opt
        .and_then(|s| {
            s.devices
                .iter()
                .find(|d| {
                    (
                        d.chip_coords.x,
                        d.chip_coords.y,
                        d.chip_coords.z,
                        d.core_on_chip.0,
                    ) == (x, y, z, core_idx)
                })
                .map(|d| usize::try_from(d.device_id.0).unwrap_or(0))
        })
        .unwrap_or_else(|| usize::try_from(y * 2 + x + core_idx).unwrap_or(0));
    args.device_id_out = found;
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args`] structure.
///
/// # Safety
///
/// If `args.coords_out` is non-null, it must point to at least 3 `i32` slots.
unsafe fn handle_tpu_topology_coord_for_logi_device(
    args: &mut PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args,
) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let spec_opt = unsafe { resolve_topology_spec(args.topology) };
    let (x, y, z, core_idx) = spec_opt
        .and_then(|s| s.devices.get(args.device_id))
        .map_or_else(
            || {
                let d = global_init_config().chips_per_host_bounds;
                let bx = usize::try_from(d.x.max(1)).unwrap_or(2);
                let by = usize::try_from(d.y.max(1)).unwrap_or(2);
                let cx = i32::try_from(args.device_id % bx).unwrap_or(0);
                let cy = i32::try_from((args.device_id / bx) % by).unwrap_or(0);
                let cz = i32::try_from(args.device_id / (bx * by)).unwrap_or(0);
                (cx, cy, cz, 0)
            },
            |d| {
                (
                    i32::try_from(d.chip_coords.x).unwrap_or(0),
                    i32::try_from(d.chip_coords.y).unwrap_or(0),
                    i32::try_from(d.chip_coords.z).unwrap_or(0),
                    usize::try_from(d.core_on_chip.0).unwrap_or(0),
                )
            },
        );
    if !args.coords_out.is_null() {
        // SAFETY: `coords_out` points to an `int32_t[3]` coordinate array.
        unsafe {
            *args.coords_out = x;
            *args.coords_out.add(1) = y;
            *args.coords_out.add(2) = z;
        }
    }
    args.core_idx_out = core_idx;
}

/// Maps chip ID to `(process_id, index_on_process)` (`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcIdAndIdxOnProcForChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args>(
            raw_args,
            handle_tpu_topology_proc_for_chip,
        )
    }
}

/// Maps logical device ID to `(process_id, index_on_process)` (`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args>(
            raw_args,
            handle_tpu_topology_proc_for_device,
        )
    }
}

/// Maps process ID to 3D process coordinates (`PJRT_TpuTopology_ProcessCoordFromId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ProcessCoordFromId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ProcessCoordFromId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessCoordFromId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ProcessCoordFromId_Args>(
            raw_args,
            handle_tpu_topology_process_coord_from_id,
        )
    }
}

/// Maps 3D chip coordinates to chip ID (`PJRT_TpuTopology_ChipIdFromCoord`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ChipIdFromCoord_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ChipIdFromCoord_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipIdFromCoord(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ChipIdFromCoord_Args>(
            raw_args,
            handle_tpu_topology_chip_id_from_coord,
        )
    }
}

/// Maps `(chip_coord, core_idx)` to logical device ID (`PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args>(
            raw_args,
            handle_tpu_topology_logi_device_from_coord,
        )
    }
}

/// Maps logical device ID to `(chip_coord, core_idx)` (`PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args>(
            raw_args,
            handle_tpu_topology_coord_for_logi_device,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ChipsPerProcessBounds`] and [`PJRT_TpuTopology_ChipBounds`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ChipBounds_Args`] structure.
///
/// # Safety
///
/// If `args.coords_out` is non-null, it must point to at least 3 `i32` slots.
unsafe fn handle_tpu_topology_chip_bounds(args: &mut PJRT_TpuTopology_ChipBounds_Args) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let bounds = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || global_init_config().chips_per_host_bounds,
        |spec| spec.chip_bounds,
    );
    if !args.coords_out.is_null() {
        // SAFETY: `coords_out` points to an `int32_t[3]` coordinate array.
        unsafe {
            *args.coords_out = i32::try_from(bounds.x).unwrap_or(2);
            *args.coords_out.add(1) = i32::try_from(bounds.y).unwrap_or(2);
            *args.coords_out.add(2) = i32::try_from(bounds.z).unwrap_or(1);
        }
    }
    args.num_coords = 3;
}

/// Queries 3D chips-per-process bounds (`PJRT_TpuTopology_ChipsPerProcessBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ChipBounds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ChipBounds_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipsPerProcessBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ChipBounds_Args>(
            raw_args,
            handle_tpu_topology_chip_bounds,
        )
    }
}

/// Queries 3D chip mesh bounds (`PJRT_TpuTopology_ChipBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ChipBounds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ChipBounds_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ChipBounds_Args>(
            raw_args,
            handle_tpu_topology_chip_bounds,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcessBounds`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ProcessBounds_Args`] structure.
///
/// # Safety
///
/// If `args.coords_out` is non-null, it must point to at least 3 `i32` slots.
unsafe fn handle_tpu_topology_process_bounds(args: &mut PJRT_TpuTopology_ProcessBounds_Args) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let bounds = unsafe { resolve_topology_spec(args.topology) }
        .map_or_else(|| global_init_config().host_bounds, |spec| spec.host_bounds);
    if !args.coords_out.is_null() {
        // SAFETY: `coords_out` points to an `int32_t[3]` coordinate array.
        unsafe {
            *args.coords_out = i32::try_from(bounds.x).unwrap_or(1);
            *args.coords_out.add(1) = i32::try_from(bounds.y).unwrap_or(1);
            *args.coords_out.add(2) = i32::try_from(bounds.z).unwrap_or(1);
        }
    }
    args.num_coords = 3;
}

/// Queries 3D process mesh bounds (`PJRT_TpuTopology_ProcessBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ProcessBounds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ProcessBounds_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ProcessBounds_Args>(
            raw_args,
            handle_tpu_topology_process_bounds,
        )
    }
}

/// Queries ICI routing strategy (`PJRT_TpuTopology_GetRoutingStrategy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_GetRoutingStrategy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_GetRoutingStrategy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetRoutingStrategy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_GetRoutingStrategy_Args>(
            raw_args,
            handle_tpu_topology_routing_strategy,
        )
    }
}

/// Builds a generation-authentic [`PJRT_TpuTopology_SliceConfig`] from 3D chip bounds.
///
/// # Arguments
///
/// * `bounds` - 3D chip mesh [`libtpu_core::types::ChipCoordinates`] bounds.
fn build_slice_config_from_bounds(
    bounds: libtpu_core::types::ChipCoordinates,
) -> PJRT_TpuTopology_SliceConfig {
    let dx = i32::try_from(bounds.x.max(1)).unwrap_or(2);
    let dy = i32::try_from(bounds.y.max(1)).unwrap_or(2);
    let dz = i32::try_from(bounds.z.max(1)).unwrap_or(1);
    PJRT_TpuTopology_SliceConfig {
        name: c"2x2x1".as_ptr(),
        name_len: 5,
        dimensions: [dx, dy, dz, 0],
        wraps: [false, false, false, false],
        num_dimensions: 3,
        twist: false,
        pad: [0; 7],
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_GetSliceConfig`], populating a caller
/// [`PJRT_TpuTopology_SliceConfig`] descriptor when `slice_config_out` is non-null.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_GetSliceConfig_Args`] structure.
///
/// # Safety
///
/// If `args.slice_config_out` is non-null, it must point to a writable [`PJRT_TpuTopology_SliceConfig`].
unsafe fn handle_tpu_topology_get_slice_config(args: &mut PJRT_TpuTopology_GetSliceConfig_Args) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let bounds = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || global_init_config().chips_per_host_bounds,
        |spec| spec.chip_bounds,
    );
    if !args.slice_config_out.is_null() {
        // SAFETY: `slice_config_out` points to a caller-supplied `PJRT_TpuTopology_SliceConfig`.
        unsafe {
            *args.slice_config_out = build_slice_config_from_bounds(bounds);
        }
    }
}

/// Queries a named TPU slice configuration (`PJRT_TpuTopology_GetSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_GetSliceConfig_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_GetSliceConfig_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_GetSliceConfig_Args>(
            raw_args,
            handle_tpu_topology_get_slice_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_GetSliceConfigs`], populating a caller
/// [`PJRT_TpuTopology_SliceConfig`] descriptor when `slice_configs_out` is non-null.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_GetSliceConfigs_Args`] structure.
///
/// # Safety
///
/// If `args.slice_configs_out` is non-null, it must point to at least 1 [`PJRT_TpuTopology_SliceConfig`].
unsafe fn handle_tpu_topology_get_slice_configs(args: &mut PJRT_TpuTopology_GetSliceConfigs_Args) {
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let bounds = unsafe { resolve_topology_spec(args.topology) }.map_or_else(
        || global_init_config().chips_per_host_bounds,
        |spec| spec.chip_bounds,
    );
    if !args.slice_configs_out.is_null() {
        // SAFETY: `slice_configs_out` points to a caller-supplied `PJRT_TpuTopology_SliceConfig`.
        unsafe {
            *args.slice_configs_out = build_slice_config_from_bounds(bounds);
        }
    }
}

/// Lists all TPU slice configurations (`PJRT_TpuTopology_GetSliceConfigs`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_GetSliceConfigs_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_GetSliceConfigs_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetSliceConfigs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_GetSliceConfigs_Args>(
            raw_args,
            handle_tpu_topology_get_slice_configs,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_GetDefaultPlatformConfig`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_GetDefaultPlatformConfig_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_GetDefaultPlatformConfig_Args`].
unsafe fn handle_tpu_topology_default_platform_config(
    args: &mut PJRT_TpuTopology_GetDefaultPlatformConfig_Args,
) {
    // SAFETY: `args.topology` is null, a small sentinel, or a valid `PJRT_TopologyDescription*`.
    let tpu_gen = unsafe { resolve_topology_spec(args.topology) }
        .map_or_else(|| global_init_config().generation, |spec| spec.generation);
    args.platform_config = usize::try_from(tpu_gen as i32).unwrap_or(5);
}

/// Queries default TPU platform configuration (`PJRT_TpuTopology_GetDefaultPlatformConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_GetDefaultPlatformConfig_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_GetDefaultPlatformConfig_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetDefaultPlatformConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_GetDefaultPlatformConfig_Args>(
            raw_args,
            handle_tpu_topology_default_platform_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ComputeSubsliceForDevices`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ComputeSubsliceForDevices_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ComputeSubsliceForDevices_Args`].
unsafe fn handle_tpu_topology_compute_subslice_for_devices(
    args: &mut PJRT_TpuTopology_ComputeSubsliceForDevices_Args,
) {
    // SAFETY: `args.topology` is null/sentinel or a valid `PJRT_TopologyDescription*`.
    if let Some(topo) = unsafe { resolve_topology_handle(args.topology) } {
        topo.is_subslice.store(true, Ordering::Release);
    }
    if !args.subslice_origin_out.is_null() {
        let count = args.num_subslice_dims.min(4);
        for idx in 0..count {
            // SAFETY: `subslice_origin_out` has space for `num_subslice_dims` elements.
            unsafe {
                *args.subslice_origin_out.add(idx) = 0;
            }
        }
    }
}

/// Computes the subslice origin coordinates for a subset of TPU devices (`PJRT_TpuTopology_ComputeSubsliceForDevices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ComputeSubsliceForDevices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ComputeSubsliceForDevices_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ComputeSubsliceForDevices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ComputeSubsliceForDevices_Args>(
            raw_args,
            handle_tpu_topology_compute_subslice_for_devices,
        )
    }
}
