//! PJRT TPU Topology (`PJRT_TpuTopology_Extension`) entrypoints.

use super::args::*;
use super::{dispatch_typed_extension_args, extension_state};
use crate::pjrt_types::PJRT_Error;
use libtpu_core::config::global_init_config;

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
    let mut st = extension_state().lock();
    st.is_subslice_topology = true;
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
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_is_subslice(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    let is_sub = extension_state().lock().is_subslice_topology;
    args.out_value = usize::from(is_sub);
}

/// Checks whether a topology is a subslice (`PJRT_TpuTopology_IsSubsliceTopology`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_IsSubsliceTopology(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_is_subslice,
        )
    }
}

/// Strongly-typed handler for zero-returning [`PJRT_TpuTopology_ScalarQuery_Args`] entrypoints.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_zero(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    args.out_value = 0;
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
            handle_tpu_topology_zero,
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

/// Checks whether topology has limited ICI connectivity (`PJRT_TpuTopology_HasLimitedIciConnectivity`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_HasLimitedIciConnectivity(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_zero,
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

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcessCount`] and [`PJRT_TpuTopology_ProcessIds`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_process_count(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    args.out_value = usize::try_from(global_init_config().total_hosts()).unwrap_or(1);
}

/// Queries process count in TPU topology (`PJRT_TpuTopology_ProcessCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_process_count,
        )
    }
}

/// Strongly-typed handler for chips-per-process queries.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_chips_per_process(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    args.out_value = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
}

/// Queries chips per process in TPU topology (`PJRT_TpuTopology_ChipsPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipsPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_chips_per_process,
        )
    }
}

/// Strongly-typed handler for cores-per-chip queries.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_cores_per_chip(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    args.out_value =
        usize::try_from(global_init_config().generation.logical_devices_per_chip()).unwrap_or(1);
}

/// Queries cores per chip in TPU topology (`PJRT_TpuTopology_CoreCountPerChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCountPerChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_cores_per_chip,
        )
    }
}

/// Strongly-typed handler for total chip count queries.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_total_chips(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    let cfg = global_init_config();
    args.out_value = usize::try_from(cfg.chips_per_host() * cfg.total_hosts()).unwrap_or(4);
}

/// Strongly-typed handler for total core / logical device count queries.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_total_cores(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    let cfg = global_init_config();
    args.out_value = usize::try_from(
        cfg.chips_per_host() * cfg.total_hosts() * cfg.generation.logical_devices_per_chip(),
    )
    .unwrap_or(4);
}

/// Strongly-typed handler for per-process core / logical device count queries.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_cores_per_process(args: &mut PJRT_TpuTopology_ScalarQuery_Args) {
    let cfg = global_init_config();
    args.out_value =
        usize::try_from(cfg.chips_per_host() * cfg.generation.logical_devices_per_chip())
            .unwrap_or(4);
}

/// Queries total chip count in TPU topology (`PJRT_TpuTopology_ChipCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_total_chips,
        )
    }
}

/// Queries total core count in TPU topology (`PJRT_TpuTopology_CoreCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_total_cores,
        )
    }
}

/// Queries logical devices per process (`PJRT_TpuTopology_LogiDeviceCountPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCountPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_cores_per_process,
        )
    }
}

/// Queries total logical device count (`PJRT_TpuTopology_LogiDeviceCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_total_cores,
        )
    }
}

/// Queries logical devices per chip (`PJRT_TpuTopology_LogiDeviceCountPerChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCountPerChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_cores_per_chip,
        )
    }
}

/// Queries cores per process (`PJRT_TpuTopology_CoreCountPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCountPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_cores_per_process,
        )
    }
}

/// Queries process ID list (`PJRT_TpuTopology_ProcessIds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_process_count,
        )
    }
}

/// Queries logical device IDs on a process (`PJRT_TpuTopology_LogiDeviceIdsOnProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceIdsOnProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_cores_per_process,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_MappingQuery_Args`] coordinate/index bijections.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_MappingQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_MappingQuery_Args`].
unsafe fn handle_tpu_topology_mapping(args: &mut PJRT_TpuTopology_MappingQuery_Args) {
    let chips_per_proc = usize::try_from(global_init_config().chips_per_host().max(1)).unwrap_or(4);
    args.out_first = args.param0 / chips_per_proc;
    args.out_second = args.param0 % chips_per_proc;
}

/// Maps chip ID to `(process_id, index_on_process)` (`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcIdAndIdxOnProcForChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_MappingQuery_Args>(
            raw_args,
            handle_tpu_topology_mapping,
        )
    }
}

/// Maps logical device ID to `(process_id, index_on_process)` (`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_MappingQuery_Args>(
            raw_args,
            handle_tpu_topology_mapping,
        )
    }
}

/// Maps process ID to 3D process coordinates (`PJRT_TpuTopology_ProcessCoordFromId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessCoordFromId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_MappingQuery_Args>(
            raw_args,
            handle_tpu_topology_mapping,
        )
    }
}

/// Maps 3D chip coordinates to chip ID (`PJRT_TpuTopology_ChipIdFromCoord`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipIdFromCoord(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_MappingQuery_Args>(
            raw_args,
            handle_tpu_topology_mapping,
        )
    }
}

/// Maps `(chip_coord, core_idx)` to logical device ID (`PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_MappingQuery_Args>(
            raw_args,
            handle_tpu_topology_mapping,
        )
    }
}

/// Maps logical device ID to `(chip_coord, core_idx)` (`PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_MappingQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_MappingQuery_Args>(
            raw_args,
            handle_tpu_topology_mapping,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ChipsPerProcessBounds`] and [`PJRT_TpuTopology_ChipBounds`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_BoundsQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_BoundsQuery_Args`].
unsafe fn handle_tpu_topology_chip_bounds(args: &mut PJRT_TpuTopology_BoundsQuery_Args) {
    let bounds = global_init_config().chips_per_host_bounds;
    args.out_first = usize::try_from(bounds.x).unwrap_or(2);
    args.out_second = usize::try_from(bounds.y).unwrap_or(2);
}

/// Queries 3D chips-per-process bounds (`PJRT_TpuTopology_ChipsPerProcessBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_BoundsQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_BoundsQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipsPerProcessBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_BoundsQuery_Args>(
            raw_args,
            handle_tpu_topology_chip_bounds,
        )
    }
}

/// Queries 3D chip mesh bounds (`PJRT_TpuTopology_ChipBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_BoundsQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_BoundsQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_BoundsQuery_Args>(
            raw_args,
            handle_tpu_topology_chip_bounds,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_ProcessBounds`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_BoundsQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_BoundsQuery_Args`].
unsafe fn handle_tpu_topology_process_bounds(args: &mut PJRT_TpuTopology_BoundsQuery_Args) {
    let bounds = global_init_config().host_bounds;
    args.out_first = usize::try_from(bounds.x).unwrap_or(1);
    args.out_second = usize::try_from(bounds.y).unwrap_or(1);
}

/// Queries 3D process mesh bounds (`PJRT_TpuTopology_ProcessBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_BoundsQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_BoundsQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_BoundsQuery_Args>(
            raw_args,
            handle_tpu_topology_process_bounds,
        )
    }
}

/// Queries ICI routing strategy (`PJRT_TpuTopology_GetRoutingStrategy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetRoutingStrategy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_zero,
        )
    }
}

/// Queries a named TPU slice configuration (`PJRT_TpuTopology_GetSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetSliceConfig(
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

/// Lists all TPU slice configurations (`PJRT_TpuTopology_GetSliceConfigs`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetSliceConfigs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_chips_per_process,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuTopology_GetDefaultPlatformConfig`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuTopology_ScalarQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuTopology_ScalarQuery_Args`].
unsafe fn handle_tpu_topology_default_platform_config(
    args: &mut PJRT_TpuTopology_ScalarQuery_Args,
) {
    args.out_value = usize::try_from(global_init_config().generation as i32).unwrap_or(5);
}

/// Queries default TPU platform configuration (`PJRT_TpuTopology_GetDefaultPlatformConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuTopology_ScalarQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuTopology_GetDefaultPlatformConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuTopology_ScalarQuery_Args>(
            raw_args,
            handle_tpu_topology_default_platform_config,
        )
    }
}
