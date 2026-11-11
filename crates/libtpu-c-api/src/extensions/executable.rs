//! PJRT ABI Version, Shardings, and TPU Executable extension entrypoints.

use super::args::*;
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
        compiler_version: String::from("0.0.52"),
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

/// Populates a [`PJRT_Shardings_PJRT_Executable_Shardings_Args`] structure from a slice of
/// [`libtpu_core::proto::OpShardingProto`] descriptors.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Shardings_PJRT_Executable_Shardings_Args`] structure.
/// * `shardings` - Slice of [`libtpu_core::proto::OpShardingProto`] instances.
fn populate_executable_sharding_args(
    args: &mut PJRT_Shardings_PJRT_Executable_Shardings_Args,
    shardings: &[libtpu_core::proto::OpShardingProto],
) {
    let mut ptrs = Vec::with_capacity(shardings.len());
    let mut sizes = Vec::with_capacity(shardings.len());
    for sh in shardings {
        let encoded = sh.encode();
        let (byte_ptr, byte_len) = alloc_extension_bytes(encoded);
        ptrs.push(byte_ptr as usize);
        sizes.push(byte_len);
    }
    let count = ptrs.len();
    let ptr_vec: Box<[usize]> = ptrs.into_boxed_slice();
    let size_vec: Box<[usize]> = sizes.into_boxed_slice();
    args.shardings = ptr_vec.as_ptr().cast::<*const std::os::raw::c_char>();
    args.sharding_sizes = size_vec.as_ptr();
    let mut st = extension_state().lock();
    st.sharding_ptr_arrays.clear();
    st.sharding_size_arrays.clear();
    st.sharding_ptr_arrays.push(ptr_vec);
    st.sharding_size_arrays.push(size_vec);
    drop(st);
    args.num_items = count;
}

/// Strongly-typed handler for [`PJRT_Shardings_ParameterShardings`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Shardings_PJRT_Executable_Shardings_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Shardings_PJRT_Executable_Shardings_Args`].
unsafe fn handle_executable_parameter_shardings(
    args: &mut PJRT_Shardings_PJRT_Executable_Shardings_Args,
) {
    let default_shardings = [libtpu_core::proto::OpShardingProto::default()];
    let exec_opt = if (args.executable as usize) > 0x10000 {
        // SAFETY: `args.executable` is a valid `PJRT_Executable*`.
        unsafe {
            args.executable
                .cast::<crate::pjrt_types::PJRT_Executable>()
                .as_ref()
        }
    } else {
        None
    };
    let slice = exec_opt.map_or(&default_shardings[..], |e| {
        e.inner.program.parameter_shardings.as_slice()
    });
    populate_executable_sharding_args(args, slice);
}

/// Strongly-typed handler for [`PJRT_Shardings_OutputShardings`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Shardings_PJRT_Executable_Shardings_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Shardings_PJRT_Executable_Shardings_Args`].
unsafe fn handle_executable_output_shardings(
    args: &mut PJRT_Shardings_PJRT_Executable_Shardings_Args,
) {
    let default_shardings = [libtpu_core::proto::OpShardingProto::default()];
    let exec_opt = if (args.executable as usize) > 0x10000 {
        // SAFETY: `args.executable` is a valid `PJRT_Executable*`.
        unsafe {
            args.executable
                .cast::<crate::pjrt_types::PJRT_Executable>()
                .as_ref()
        }
    } else {
        None
    };
    let slice = exec_opt.map_or(&default_shardings[..], |e| {
        e.inner.program.output_shardings.as_slice()
    });
    populate_executable_sharding_args(args, slice);
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
            handle_executable_parameter_shardings,
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
            handle_executable_output_shardings,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_GetTargetArguments`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_TargetArguments_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_TargetArguments_Args`].
unsafe fn handle_tpu_executable_target_arguments(
    args: &mut PJRT_TpuExecutable_TargetArguments_Args,
) {
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
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_TargetArguments_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_TargetArguments_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetTargetArguments(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_TargetArguments_Args>(
            raw_args,
            handle_tpu_executable_target_arguments,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_GetHloModuleWithConfig`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_HloModule_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_HloModule_Args`].
unsafe fn handle_tpu_executable_hlo_module_with_config(
    args: &mut PJRT_TpuExecutable_HloModule_Args,
) {
    let exec_opt = if (args.executable as usize) > 0x10000 {
        // SAFETY: `args.executable` is a valid `PJRT_Executable*`.
        unsafe {
            args.executable
                .cast::<crate::pjrt_types::PJRT_Executable>()
                .as_ref()
        }
    } else {
        None
    };
    let (hlo_mod, cfg_proto) = exec_opt.map_or_else(
        || {
            (
                libtpu_core::proto::HloModuleProto::default(),
                libtpu_core::proto::HloModuleConfigProto::default(),
            )
        },
        |e| {
            let prog = &e.inner.program;
            let cfg = libtpu_core::proto::HloModuleConfigProto {
                replica_count: i64::try_from(prog.num_replicas).unwrap_or(1),
                num_partitions: i64::try_from(prog.num_partitions).unwrap_or(1),
                use_spmd_partitioning: prog.num_partitions > 1,
                ..libtpu_core::proto::HloModuleConfigProto::default()
            };
            (prog.hlo_module_proto.clone(), cfg)
        },
    );
    let encoded = libtpu_core::proto::HloModuleProtoWithConfig {
        hlo_module: Some(hlo_mod),
        config: Some(cfg_proto),
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
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_HloModule_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_HloModule_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetHloModuleWithConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_HloModule_Args>(
            raw_args,
            handle_tpu_executable_hlo_module_with_config,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_Reserved`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Handle_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Megascale_Handle_Args`].
unsafe fn handle_tpu_executable_reserved(args: &mut PJRT_Megascale_Handle_Args) {
    let mut st = extension_state().lock();
    st.tpu_compilation_env_overrides = st.tpu_compilation_env_overrides.saturating_add(1);
    args.handle = ptr::null_mut();
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
        dispatch_typed_extension_args::<PJRT_Megascale_Handle_Args>(
            raw_args,
            handle_tpu_executable_reserved,
        )
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
    args.out_value = if args.handle > 0x10000 {
        // SAFETY: `args.handle` is a valid `PJRT_Executable*`.
        let exec = unsafe { &*(args.handle as *const crate::pjrt_types::PJRT_Executable) };
        let prog = &exec.inner.program;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let flops_cost = prog.cost_analysis.flops.max(1.0) as usize;
        flops_cost.max(prog.instruction_bundles.len().max(1))
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
    let overrides = st.tpu_compilation_env_overrides;
    st.tpu_compilation_env
        .set_field_from_str("xla_tpu_enable_flash_attention", "true");
    args.out_value = overrides;
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
/// * `args` - Validated [`PJRT_TpuExecutable_Serialize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_TpuExecutable_Serialize_Args`].
unsafe fn handle_tpu_executable_get_env_field(args: &mut PJRT_TpuExecutable_Serialize_Args) {
    let encoded = extension_state().lock().tpu_compilation_env.encode();
    let (ptr, len) = alloc_extension_bytes(encoded);
    args.serialized_bytes = ptr as usize;
    args.serialized_bytes_size = len;
}

/// Queries a `TpuCompilationEnvironmentProto` field as string (`PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_Serialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_Serialize_Args>(
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
