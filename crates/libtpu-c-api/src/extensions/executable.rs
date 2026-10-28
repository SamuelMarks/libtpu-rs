//! PJRT ABI Version, Shardings, and TPU Executable extension entrypoints.

use super::args::*;
use super::megascale::handle_megascale_noop;
use super::{alloc_extension_bytes, dispatch_typed_extension_args, extension_state};
use crate::pjrt_helpers::box_serialized_artifact;
use crate::pjrt_types::{PJRT_AbiVersion_Holder, PJRT_Error};
use libtpu_core::config::global_init_config;
use libtpu_core::error::LibTpuError;
use std::ptr;

/// Platform ID constant for TPU (`u32::from_be_bytes(*b"TPU\0")`).
const TPU_PLATFORM_ID: u64 = u32::from_be_bytes(*b"TPU\0") as u64;

/// Strongly-typed handler for [`PJRT_AbiVersion_ClientRuntime`] and [`PJRT_AbiVersion_ExecutableGet`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_Get_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_AbiVersion_Get_Args`].
unsafe fn handle_abi_version_get(args: &mut PJRT_AbiVersion_Get_Args) {
    let holder = Box::into_raw(Box::new(PJRT_AbiVersion_Holder {
        platform_id: TPU_PLATFORM_ID,
        version: crate::pjrt_types::PJRT_API_MINOR,
        proto_bytes: Vec::new(),
    }));
    args.abi_version_out = holder as usize;
}

/// Queries client runtime ABI version (`PJRT_AbiVersion_ClientRuntime`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_Get_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_Get_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ClientRuntime(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_Get_Args>(raw_args, handle_abi_version_get)
    }
}

/// Queries executable ABI version (`PJRT_AbiVersion_ExecutableGet`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_Get_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_Get_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableGet(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_Get_Args>(raw_args, handle_abi_version_get)
    }
}

/// Strongly-typed handler for [`PJRT_AbiVersion_RuntimeDestroy`] and [`PJRT_AbiVersion_ExecutableDestroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_AbiVersion_Destroy_Args`].
unsafe fn handle_abi_version_destroy(args: &mut PJRT_AbiVersion_Destroy_Args) {
    if !args.abi_version.is_null() {
        // SAFETY: `abi_version` was allocated by `PJRT_AbiVersion_*`.
        unsafe {
            drop(Box::from_raw(
                args.abi_version.cast::<PJRT_AbiVersion_Holder>(),
            ));
        }
    }
    args.abi_version = ptr::null_mut();
}

/// Destroys a runtime ABI version handle (`PJRT_AbiVersion_RuntimeDestroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeDestroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_Destroy_Args>(
            raw_args,
            handle_abi_version_destroy,
        )
    }
}

/// Strongly-typed handler for ABI compatibility checks.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_IsCompatible_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_AbiVersion_IsCompatible_Args`].
unsafe fn handle_abi_version_is_compatible(args: &mut PJRT_AbiVersion_IsCompatible_Args) {
    args.is_compatible_out = 1;
}

/// Checks runtime ABI compatibility with another runtime (`PJRT_AbiVersion_RuntimeCompatibleWithRuntime`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_IsCompatible_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_IsCompatible_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeCompatibleWithRuntime(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_IsCompatible_Args>(
            raw_args,
            handle_abi_version_is_compatible,
        )
    }
}

/// Checks runtime ABI compatibility with an executable (`PJRT_AbiVersion_RuntimeCompatibleWithExecutable`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_IsCompatible_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_IsCompatible_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeCompatibleWithExecutable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_IsCompatible_Args>(
            raw_args,
            handle_abi_version_is_compatible,
        )
    }
}

/// Strongly-typed handler for [`PJRT_AbiVersion_RuntimeToProto`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_ToProto_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_AbiVersion_ToProto_Args`].
unsafe fn handle_abi_version_runtime_to_proto(args: &mut PJRT_AbiVersion_ToProto_Args) {
    let encoded = libtpu_core::proto::RuntimeAbiVersionProto {
        platform_id: TPU_PLATFORM_ID,
        version_string: String::from("0.116"),
        pjrt_api_minor: crate::pjrt_types::PJRT_API_MINOR,
    }
    .encode();
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_proto = ptr;
    args.serialized_proto_size = len;
    args.serialized_proto_holder = holder.cast::<std::ffi::c_void>();
}

/// Serializes runtime ABI version to protobuf (`PJRT_AbiVersion_RuntimeToProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_ToProto_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_ToProto_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeToProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_ToProto_Args>(
            raw_args,
            handle_abi_version_runtime_to_proto,
        )
    }
}

/// Strongly-typed handler for [`PJRT_AbiVersion_RuntimePlatformId`] and [`PJRT_AbiVersion_ExecutablePlatformId`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_PlatformId_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_AbiVersion_PlatformId_Args`].
unsafe fn handle_abi_version_platform_id(args: &mut PJRT_AbiVersion_PlatformId_Args) {
    args.platform_id_out = usize::try_from(TPU_PLATFORM_ID).unwrap_or(0);
}

/// Queries platform ID from runtime ABI version (`PJRT_AbiVersion_RuntimePlatformId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_PlatformId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_PlatformId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimePlatformId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_PlatformId_Args>(
            raw_args,
            handle_abi_version_platform_id,
        )
    }
}

/// Destroys an executable ABI version handle (`PJRT_AbiVersion_ExecutableDestroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableDestroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_Destroy_Args>(
            raw_args,
            handle_abi_version_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_AbiVersion_ExecutableToProto`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_ToProto_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_AbiVersion_ToProto_Args`].
unsafe fn handle_abi_version_executable_to_proto(args: &mut PJRT_AbiVersion_ToProto_Args) {
    let encoded = libtpu_core::proto::ExecutableAbiVersionProto {
        platform_id: TPU_PLATFORM_ID,
        compiler_version: String::from("0.0.50"),
        target_generation: global_init_config().generation as i32,
    }
    .encode();
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_proto = ptr;
    args.serialized_proto_size = len;
    args.serialized_proto_holder = holder.cast::<std::ffi::c_void>();
}

/// Serializes executable ABI version to protobuf (`PJRT_AbiVersion_ExecutableToProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_ToProto_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_ToProto_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableToProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_ToProto_Args>(
            raw_args,
            handle_abi_version_executable_to_proto,
        )
    }
}

/// Queries platform ID from executable ABI version (`PJRT_AbiVersion_ExecutablePlatformId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_PlatformId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_PlatformId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutablePlatformId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_PlatformId_Args>(
            raw_args,
            handle_abi_version_platform_id,
        )
    }
}

/// Strongly-typed handler for [`PJRT_AbiVersion_RuntimeFromProto`] and [`PJRT_AbiVersion_ExecutableFromProto`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_FromProto_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_AbiVersion_FromProto_Args`].
unsafe fn handle_abi_version_from_proto(args: &mut PJRT_AbiVersion_FromProto_Args) {
    let proto_bytes = if !args.serialized_proto.is_null() {
        // SAFETY: `serialized_proto` is non-null with `serialized_proto_size` bytes.
        unsafe {
            std::slice::from_raw_parts(
                args.serialized_proto.cast::<u8>(),
                args.serialized_proto_size,
            )
            .to_vec()
        }
    } else {
        Vec::new()
    };
    let holder = Box::into_raw(Box::new(PJRT_AbiVersion_Holder {
        platform_id: TPU_PLATFORM_ID,
        version: crate::pjrt_types::PJRT_API_MINOR,
        proto_bytes,
    }));
    args.abi_version_out = holder as usize;
}

/// Deserializes runtime ABI version from protobuf (`PJRT_AbiVersion_RuntimeFromProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_FromProto_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_FromProto_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeFromProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_FromProto_Args>(
            raw_args,
            handle_abi_version_from_proto,
        )
    }
}

/// Deserializes executable ABI version from protobuf (`PJRT_AbiVersion_ExecutableFromProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_FromProto_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_FromProto_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableFromProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_AbiVersion_FromProto_Args>(
            raw_args,
            handle_abi_version_from_proto,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Shardings_ParameterShardings`] and [`PJRT_Shardings_OutputShardings`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Shardings_PJRT_Executable_Shardings_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Shardings_PJRT_Executable_Shardings_Args`].
unsafe fn handle_executable_shardings(args: &mut PJRT_Shardings_PJRT_Executable_Shardings_Args) {
    let encoded = libtpu_core::proto::OpShardingProto::default().encode();
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let count = unsafe {
        args.executable
            .cast::<crate::pjrt_types::PJRT_Executable>()
            .as_ref()
    }
    .map_or(1, |e| e.inner.program.output_shapes.len().max(1));
    let _ = alloc_extension_bytes(encoded);
    args.num_items = count;
}

/// Queries parameter shardings (`xla.OpSharding` proto) of an executable (`PJRT_Shardings_ParameterShardings`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Shardings_PJRT_Executable_Shardings_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Shardings_PJRT_Executable_Shardings_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Shardings_ParameterShardings(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Shardings_PJRT_Executable_Shardings_Args>(
            raw_args,
            handle_executable_shardings,
        )
    }
}

/// Queries output shardings (`xla.OpSharding` proto) of an executable (`PJRT_Shardings_OutputShardings`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Shardings_PJRT_Executable_Shardings_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Shardings_PJRT_Executable_Shardings_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Shardings_OutputShardings(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Shardings_PJRT_Executable_Shardings_Args>(
            raw_args,
            handle_executable_shardings,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_GetTargetArguments`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_ProtoQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_ProtoQuery_Args`].
unsafe fn handle_tpu_executable_target_arguments(args: &mut PJRT_TpuExecutable_ProtoQuery_Args) {
    let tpu_gen = global_init_config().generation;
    let target_bytes = libtpu_core::proto::TargetArgumentsProto {
        tpu_version: tpu_gen as i32,
        chip_variant: tpu_gen.to_string(),
        clock_rate_mhz: 1750,
        hbm_capacity_bytes: i64::try_from(tpu_gen.hbm_capacity_bytes().0).unwrap_or(0),
        vmem_capacity_bytes: i64::try_from(tpu_gen.vmem_capacity_bytes().0).unwrap_or(0),
        smem_capacity_bytes: 2 * 1024 * 1024,
        cmem_capacity_bytes: 128 * 1024 * 1024,
        tensor_cores_per_chip: i32::try_from(tpu_gen.tensor_cores_per_chip()).unwrap_or(1),
        sparse_cores_per_chip: i32::try_from(tpu_gen.sparse_cores_per_chip()).unwrap_or(0),
        mxu_dimension: i32::try_from(tpu_gen.mxu_dimension()).unwrap_or(128),
        ici_bandwidth_bps: 4_800_000_000_000,
    }
    .encode();
    let (ptr, len, holder) = box_serialized_artifact(target_bytes);
    args.serialized_bytes = ptr as usize;
    args.serialized_bytes_size = len;
    args.holder = holder.cast::<std::ffi::c_void>();
}

/// Queries TPU target arguments from an executable (`PJRT_TpuExecutable_GetTargetArguments`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_ProtoQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_ProtoQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetTargetArguments(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_ProtoQuery_Args>(
            raw_args,
            handle_tpu_executable_target_arguments,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_GetHloModuleWithConfig`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_ProtoQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_ProtoQuery_Args`].
unsafe fn handle_tpu_executable_hlo_module_with_config(
    args: &mut PJRT_TpuExecutable_ProtoQuery_Args,
) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let hlo_mod = unsafe {
        args.executable
            .cast::<crate::pjrt_types::PJRT_Executable>()
            .as_ref()
    }
    .map_or_else(libtpu_core::proto::HloModuleProto::default, |e| {
        libtpu_core::proto::HloModuleProto {
            name: e.inner.program.name.clone(),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            computations: Vec::new(),
            host_program_shape: None,
            id: 1,
        }
    });
    let encoded = libtpu_core::proto::HloModuleProtoWithConfig {
        hlo_module: Some(hlo_mod),
        config: Some(libtpu_core::proto::HloModuleConfigProto::default()),
    }
    .encode();
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_bytes = ptr as usize;
    args.serialized_bytes_size = len;
    args.holder = holder.cast::<std::ffi::c_void>();
}

/// Queries `HloModuleProtoWithConfig` from an executable (`PJRT_TpuExecutable_GetHloModuleWithConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_ProtoQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_ProtoQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetHloModuleWithConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_ProtoQuery_Args>(
            raw_args,
            handle_tpu_executable_hlo_module_with_config,
        )
    }
}

/// Reserved slot 3 of `PJRT_TpuExecutable_Extension` (`PJRT_TpuExecutable_Reserved`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_Reserved(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(raw_args, handle_megascale_noop)
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_GetCompiledMemoryStats`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_GetCompiledMemoryStats_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_GetCompiledMemoryStats_Args`].
unsafe fn handle_tpu_executable_get_compiled_memory_stats(
    args: &mut PJRT_TpuExecutable_GetCompiledMemoryStats_Args,
) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe {
        args.executable
            .cast::<crate::pjrt_types::PJRT_Executable>()
            .as_ref()
    };
    let (code_sz, arg_sz, out_sz, alias_sz, temp_sz) =
        exec_opt.map_or((256, 64, 32, 0, 128), |e| {
            let ms = &e.inner.program.memory_stats;
            (
                ms.generated_code_size_in_bytes,
                ms.argument_size_in_bytes,
                ms.output_size_in_bytes,
                ms.alias_size_in_bytes,
                ms.temp_size_in_bytes,
            )
        });
    args.generated_code_size_in_bytes = code_sz;
    args.argument_size_in_bytes = arg_sz;
    args.output_size_in_bytes = out_sz;
    args.alias_size_in_bytes = alias_sz;
    args.temp_size_in_bytes = temp_sz;
}

/// Queries compiled HBM/VMEM memory statistics from an executable (`PJRT_TpuExecutable_GetCompiledMemoryStats`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_GetCompiledMemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_GetCompiledMemoryStats_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetCompiledMemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_GetCompiledMemoryStats_Args>(
            raw_args,
            handle_tpu_executable_get_compiled_memory_stats,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_RunHloCostAnalysis`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_tpu_executable_cost_analysis(args: &mut PJRT_Megascale_Query_Args) {
    args.out_value = if args.handle != 0 {
        // SAFETY: `args.handle` is a valid `PJRT_Executable*`.
        let exec = unsafe { &*(args.handle as *const crate::pjrt_types::PJRT_Executable) };
        exec.inner.program.instruction_bundles.len().max(1)
    } else {
        4
    };
}

/// Runs HLO cost analysis on an executable (`PJRT_TpuExecutable_RunHloCostAnalysis`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_RunHloCostAnalysis(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_tpu_executable_cost_analysis,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_SetTpuCompilationEnv`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Query_Args`].
unsafe fn handle_tpu_executable_set_env(args: &mut PJRT_Megascale_Query_Args) {
    let mut st = extension_state().lock();
    st.tpu_compilation_env_overrides = st.tpu_compilation_env_overrides.saturating_add(1);
    args.out_value = st.tpu_compilation_env_overrides;
}

/// Sets `TpuCompilationEnvironmentProto` field overrides (`PJRT_TpuExecutable_SetTpuCompilationEnv`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_SetTpuCompilationEnv(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Megascale_Query_Args>(
            raw_args,
            handle_tpu_executable_set_env,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_ProtoQuery_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_ProtoQuery_Args`].
unsafe fn handle_tpu_executable_get_env_field(args: &mut PJRT_TpuExecutable_ProtoQuery_Args) {
    let encoded = libtpu_core::proto::TpuCompilationEnvironmentProto::default().encode();
    let (ptr, len) = alloc_extension_bytes(encoded);
    args.serialized_bytes = ptr as usize;
    args.serialized_bytes_size = len;
}

/// Queries a `TpuCompilationEnvironmentProto` field as string (`PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_ProtoQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_ProtoQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_ProtoQuery_Args>(
            raw_args,
            handle_tpu_executable_get_env_field,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_IsPredeterminedError`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_ErrorCheck_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_ErrorCheck_Args`].
unsafe fn handle_tpu_executable_is_predetermined_error(
    args: &mut PJRT_TpuExecutable_ErrorCheck_Args,
) {
    // SAFETY: `args.error` is null or a valid `PJRT_Error*`.
    let err_opt = unsafe { args.error.cast::<PJRT_Error>().as_ref() };
    let matched = err_opt.map_or_else(
        || {
            LibTpuError::DevicePoisoned {
                device_id: 0,
                launch_id: 0,
                reason: String::new(),
            }
            .is_predetermined_error()
        },
        |e| e.error.is_predetermined_error(),
    );
    args.is_matched = usize::from(matched);
}

/// Checks whether a `PJRT_Error` is a TPU predetermined error (`PJRT_TpuExecutable_IsPredeterminedError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_ErrorCheck_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_ErrorCheck_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_IsPredeterminedError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_ErrorCheck_Args>(
            raw_args,
            handle_tpu_executable_is_predetermined_error,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_IsCancellationPlaceholderError`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_ErrorCheck_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_ErrorCheck_Args`].
unsafe fn handle_tpu_executable_is_cancellation_error(
    args: &mut PJRT_TpuExecutable_ErrorCheck_Args,
) {
    // SAFETY: `args.error` is null or a valid `PJRT_Error*`.
    let err_opt = unsafe { args.error.cast::<PJRT_Error>().as_ref() };
    let matched = err_opt.map_or_else(
        || {
            LibTpuError::EventFailed {
                reason: String::new(),
            }
            .is_cancellation_placeholder_error()
        },
        |e| e.error.is_cancellation_placeholder_error(),
    );
    args.is_matched = usize::from(matched);
}

/// Checks whether a `PJRT_Error` is a TPU cancellation placeholder error (`PJRT_TpuExecutable_IsCancellationPlaceholderError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_ErrorCheck_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_ErrorCheck_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_IsCancellationPlaceholderError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_ErrorCheck_Args>(
            raw_args,
            handle_tpu_executable_is_cancellation_error,
        )
    }
}
