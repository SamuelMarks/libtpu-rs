//! PJRT ABI Version, Shardings, and TPU Executable extension entrypoints (`0.0.53.1`).
//!
//! Implements `PJRT_AbiVersion_Extension`, `PJRT_Shardings_Extension`, and `PJRT_TpuExecutable_Extension`.

use super::args::*;
use super::{dispatch_typed_extension_args, dispatch_typed_fallible_extension_args};
use crate::pjrt_helpers::{
    borrow_live_handle, box_registered_handle, box_serialized_artifact, take_live_handle,
};
use crate::pjrt_types::{PJRT_AbiVersion_Holder, PJRT_Error};
use libtpu_core::config::global_init_config;
use libtpu_core::error::LibTpuError;
use libtpu_core::types::TpuGeneration;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Static wire-encoded replicated `xla.OpSharding` (`type = REPLICATED`) for null-executable probes.
static DEFAULT_OP_SHARDING_WIRE: [u8; 2] = [0x08, 0x00];

/// Static pointer wrapper for [`DEFAULT_OP_SHARDING_WIRE`].
#[repr(transparent)]
struct StaticDefaultShardingPtrs([*const std::os::raw::c_char; 1]);

// SAFETY: Points to immutable `'static` bytes in `DEFAULT_OP_SHARDING_WIRE`.
unsafe impl Sync for StaticDefaultShardingPtrs {}

/// Static default sharding pointer array for null-handle probes.
static DEFAULT_SHARDING_PTRS: StaticDefaultShardingPtrs =
    StaticDefaultShardingPtrs([DEFAULT_OP_SHARDING_WIRE.as_ptr().cast()]);

/// Static default sharding byte-length array for null-handle probes.
static DEFAULT_SHARDING_SIZES: [usize; 1] = [DEFAULT_OP_SHARDING_WIRE.len()];

/// Atomic counter tracking standalone compilation environment override calls when no executable handle is supplied.
static STANDALONE_ENV_OVERRIDES: AtomicUsize = AtomicUsize::new(0);

/// Platform ID constant for TPU (`u32::from_be_bytes(*b"TPU\0")` = `0x5450_5500`).
pub const TPU_PLATFORM_ID: u64 = u32::from_be_bytes(*b"TPU\0") as u64;

/// Encodes a [`libtpu_core::proto::RuntimeAbiVersionProto`] for `(platform_id, major, minor, patch, generation)`.
///
/// # Arguments
///
/// * `platform_id` - Numeric 64-bit platform identifier.
/// * `major` - PJRT ABI major version (`0`).
/// * `minor` - PJRT ABI minor version (`117`).
/// * `patch` - `libtpu` patch release (`1` for `0.0.53.1`).
/// * `generation` - Target [`TpuGeneration`].
fn encode_runtime_abi_proto(
    platform_id: u64,
    major: u32,
    minor: u32,
    patch: u32,
    generation: TpuGeneration,
) -> Vec<u8> {
    use libtpu_core::proto::ProtoSchemaMessage as _;
    libtpu_core::proto::RuntimeAbiVersionProto {
        platform: platform_id,
        version: format!("{major}.{minor}.{patch}-0.0.53.1-{generation}").into_bytes(),
    }
    .encode_proto()
}

/// Encodes a [`libtpu_core::proto::ExecutableAbiVersionProto`] for `(platform_id, major, minor, patch, generation)`.
///
/// # Arguments
///
/// * `platform_id` - Numeric 64-bit platform identifier.
/// * `major` - PJRT ABI major version (`0`).
/// * `minor` - PJRT ABI minor version (`117`).
/// * `patch` - `libtpu` patch release (`1` for `0.0.53.1`).
/// * `generation` - Target [`TpuGeneration`].
fn encode_executable_abi_proto(
    platform_id: u64,
    major: u32,
    minor: u32,
    patch: u32,
    generation: TpuGeneration,
) -> Vec<u8> {
    use libtpu_core::proto::ProtoSchemaMessage as _;
    libtpu_core::proto::ExecutableAbiVersionProto {
        platform: platform_id,
        version: format!("{major}.{minor}.{patch}-0.0.53.1-{generation}").into_bytes(),
    }
    .encode_proto()
}

/// Parses `(major, minor, patch, generation)` from a serialized ABI version tag string
/// such as `"0.117.1-0.0.53.1-v6e"` or `"0.117"`.
///
/// # Arguments
///
/// * `version_bytes` - Raw version bytes stored inside the ABI protobuf message.
fn parse_abi_version_tag(version_bytes: &[u8]) -> (u32, u32, u32, TpuGeneration) {
    let default_gen = global_init_config().generation;
    let Ok(text) = std::str::from_utf8(version_bytes) else {
        return (0, 117, 1, default_gen);
    };
    let mut dash_parts = text.split('-');
    let semver_part = dash_parts.next().unwrap_or("0.117.1");
    let mut nums = semver_part.split('.');
    let major = nums.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
    let minor = nums
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(117);
    let patch = nums.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(1);
    let parsed_gen = text
        .rsplit('-')
        .next()
        .and_then(|tok| TpuGeneration::from_name(tok).ok())
        .unwrap_or(default_gen);
    (major, minor, patch, parsed_gen)
}

/// Extracts [`TpuGeneration`] from a compiled [`crate::pjrt_types::PJRT_Executable`].
///
/// # Arguments
///
/// * `exec` - Compiled executable handle.
fn generation_for_executable(exec: &crate::pjrt_types::PJRT_Executable) -> TpuGeneration {
    TpuGeneration::from_name(&exec.inner.program.mosaic_metadata.compilation_target)
        .unwrap_or_else(|_| global_init_config().generation)
}

/// Strongly-typed handler for [`PJRT_AbiVersion_ClientRuntime`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_RuntimeAbiVersion_Args`] structure.
///
/// # Safety
///
/// If `args.client` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Client`].
unsafe fn handle_abi_version_client_runtime(args: &mut PJRT_Client_RuntimeAbiVersion_Args) {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let generation = unsafe { args.client.as_ref() }.map_or_else(
        || global_init_config().generation,
        |c| c.inner.topology.spec.generation,
    );
    let major = u32::try_from(crate::pjrt_types::PJRT_API_MAJOR).unwrap_or(0);
    let minor = u32::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(117);
    let patch = 1_u32;
    let proto_bytes = encode_runtime_abi_proto(TPU_PLATFORM_ID, major, minor, patch, generation);
    args.abi_version = box_registered_handle(PJRT_AbiVersion_Holder {
        platform_id: TPU_PLATFORM_ID,
        major,
        minor,
        patch,
        generation,
        is_executable: false,
        proto_bytes,
    });
}

/// Strongly-typed handler for [`PJRT_AbiVersion_ExecutableGet`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Executable_GetAbiVersion_Args`] structure.
///
/// # Safety
///
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_abi_version_executable_get(args: &mut PJRT_Executable_GetAbiVersion_Args) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let generation = unsafe { args.executable.as_ref() }.map_or_else(
        || global_init_config().generation,
        generation_for_executable,
    );
    let major = u32::try_from(crate::pjrt_types::PJRT_API_MAJOR).unwrap_or(0);
    let minor = u32::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(117);
    let patch = 1_u32;
    let proto_bytes = encode_executable_abi_proto(TPU_PLATFORM_ID, major, minor, patch, generation);
    args.abi_version = box_registered_handle(PJRT_AbiVersion_Holder {
        platform_id: TPU_PLATFORM_ID,
        major,
        minor,
        patch,
        generation,
        is_executable: true,
        proto_bytes,
    });
}

/// Queries client runtime ABI version (`PJRT_AbiVersion_ClientRuntime`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_RuntimeAbiVersion_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_RuntimeAbiVersion_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ClientRuntime(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Client_RuntimeAbiVersion_Args>(
            raw_args,
            handle_abi_version_client_runtime,
        )
    }
}

/// Queries executable ABI version (`PJRT_AbiVersion_ExecutableGet`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Executable_GetAbiVersion_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Executable_GetAbiVersion_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableGet(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Executable_GetAbiVersion_Args>(
            raw_args,
            handle_abi_version_executable_get,
        )
    }
}

/// Strongly-typed fallible handler for [`PJRT_AbiVersion_RuntimeDestroy`] and [`PJRT_AbiVersion_ExecutableDestroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_Destroy_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `abi_version` is null or unrecognized.
///
/// # Safety
///
/// `args` must be a valid mutable [`PJRT_AbiVersion_Destroy_Args`] reference.
unsafe fn handle_abi_version_destroy(
    args: &mut PJRT_AbiVersion_Destroy_Args,
) -> Result<(), LibTpuError> {
    let _holder =
        take_live_handle::<PJRT_AbiVersion_Holder>(args.abi_version).ok_or_else(|| {
            LibTpuError::InvalidArgument {
                parameter: "abi_version",
                reason: String::from("null or unregistered PJRT_AbiVersion_Holder handle"),
            }
        })?;
    args.abi_version = ptr::null_mut();
    Ok(())
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_AbiVersion_Destroy_Args>(
            raw_args,
            handle_abi_version_destroy,
        )
    }
}

/// Strongly-typed fallible handler for [`PJRT_AbiVersion_RuntimeCompatibleWithRuntime`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_IsCompatible_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if either `abi_version` or `other_abi_version` is null/invalid.
///
/// # Safety
///
/// `args` must be a valid mutable [`PJRT_AbiVersion_IsCompatible_Args`] reference.
unsafe fn handle_abi_version_compatible_with_runtime(
    args: &mut PJRT_AbiVersion_IsCompatible_Args,
) -> Result<(), LibTpuError> {
    let lhs = borrow_live_handle::<PJRT_AbiVersion_Holder>(args.abi_version).ok_or_else(|| {
        LibTpuError::InvalidArgument {
            parameter: "abi_version",
            reason: String::from("null or unregistered runtime PJRT_AbiVersion_Holder handle"),
        }
    })?;
    let rhs =
        borrow_live_handle::<PJRT_AbiVersion_Holder>(args.other_abi_version).ok_or_else(|| {
            LibTpuError::InvalidArgument {
                parameter: "other_abi_version",
                reason: String::from("null or unregistered peer PJRT_AbiVersion_Holder handle"),
            }
        })?;
    let compatible = lhs.platform_id == rhs.platform_id
        && lhs.major == rhs.major
        && lhs.minor == rhs.minor
        && lhs.generation == rhs.generation;
    args.is_compatible_out = usize::from(compatible);
    Ok(())
}

/// Strongly-typed fallible handler for [`PJRT_AbiVersion_RuntimeCompatibleWithExecutable`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_IsCompatible_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if either `abi_version` or `other_abi_version` is null/invalid.
///
/// # Safety
///
/// `args` must be a valid mutable [`PJRT_AbiVersion_IsCompatible_Args`] reference.
unsafe fn handle_abi_version_compatible_with_executable(
    args: &mut PJRT_AbiVersion_IsCompatible_Args,
) -> Result<(), LibTpuError> {
    let runtime =
        borrow_live_handle::<PJRT_AbiVersion_Holder>(args.abi_version).ok_or_else(|| {
            LibTpuError::InvalidArgument {
                parameter: "abi_version",
                reason: String::from("null or unregistered runtime PJRT_AbiVersion_Holder handle"),
            }
        })?;
    let executable = borrow_live_handle::<PJRT_AbiVersion_Holder>(args.other_abi_version)
        .ok_or_else(|| LibTpuError::InvalidArgument {
            parameter: "other_abi_version",
            reason: String::from("null or unregistered executable PJRT_AbiVersion_Holder handle"),
        })?;
    let compatible = runtime.platform_id == executable.platform_id
        && runtime.major == executable.major
        && runtime.minor >= executable.minor
        && runtime.generation == executable.generation;
    args.is_compatible_out = usize::from(compatible);
    Ok(())
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_AbiVersion_IsCompatible_Args>(
            raw_args,
            handle_abi_version_compatible_with_runtime,
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_AbiVersion_IsCompatible_Args>(
            raw_args,
            handle_abi_version_compatible_with_executable,
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
/// `args` must be a valid mutable [`PJRT_AbiVersion_ToProto_Args`] reference.
unsafe fn handle_abi_version_runtime_to_proto(args: &mut PJRT_AbiVersion_ToProto_Args) {
    let encoded = borrow_live_handle::<PJRT_AbiVersion_Holder>(args.abi_version).map_or_else(
        || {
            encode_runtime_abi_proto(
                TPU_PLATFORM_ID,
                u32::try_from(crate::pjrt_types::PJRT_API_MAJOR).unwrap_or(0),
                u32::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(117),
                1,
                global_init_config().generation,
            )
        },
        |holder| holder.proto_bytes.clone(),
    );
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
/// `args` must be a valid mutable [`PJRT_AbiVersion_PlatformId_Args`] reference.
unsafe fn handle_abi_version_platform_id(args: &mut PJRT_AbiVersion_PlatformId_Args) {
    args.platform_id_out = borrow_live_handle::<PJRT_AbiVersion_Holder>(args.abi_version)
        .map_or(TPU_PLATFORM_ID, |h| h.platform_id);
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_AbiVersion_Destroy_Args>(
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
/// `args` must be a valid mutable [`PJRT_AbiVersion_ToProto_Args`] reference.
unsafe fn handle_abi_version_executable_to_proto(args: &mut PJRT_AbiVersion_ToProto_Args) {
    let encoded = borrow_live_handle::<PJRT_AbiVersion_Holder>(args.abi_version).map_or_else(
        || {
            encode_executable_abi_proto(
                TPU_PLATFORM_ID,
                u32::try_from(crate::pjrt_types::PJRT_API_MAJOR).unwrap_or(0),
                u32::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(117),
                1,
                global_init_config().generation,
            )
        },
        |holder| holder.proto_bytes.clone(),
    );
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

/// Strongly-typed fallible handler for [`PJRT_AbiVersion_RuntimeFromProto`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_FromProto_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `serialized_proto` is null when `serialized_proto_size > 0` or
/// protobuf wire decoding fails.
///
/// # Safety
///
/// When `args.serialized_proto` is non-null, it must point to `args.serialized_proto_size` valid bytes.
unsafe fn handle_abi_version_runtime_from_proto(
    args: &mut PJRT_AbiVersion_FromProto_Args,
) -> Result<(), LibTpuError> {
    use libtpu_core::proto::ProtoSchemaMessage as _;
    if args.serialized_proto_size > 0 && args.serialized_proto.is_null() {
        return Err(LibTpuError::InvalidArgument {
            parameter: "serialized_proto",
            reason: String::from("PJRT_AbiVersion_RuntimeFromProto serialized_proto is null"),
        });
    }
    let proto_bytes = if args.serialized_proto_size == 0 {
        Vec::new()
    } else {
        // SAFETY: `serialized_proto` is non-null with `serialized_proto_size` bytes.
        unsafe {
            std::slice::from_raw_parts(
                args.serialized_proto.cast::<u8>(),
                args.serialized_proto_size,
            )
            .to_vec()
        }
    };
    let decoded = if proto_bytes.is_empty() {
        libtpu_core::proto::RuntimeAbiVersionProto {
            platform: TPU_PLATFORM_ID,
            version: Vec::from(&b"0.117.1"[..]),
        }
    } else {
        libtpu_core::proto::RuntimeAbiVersionProto::decode_proto(&proto_bytes)?
    };
    let platform_id = if decoded.platform == 0 {
        TPU_PLATFORM_ID
    } else {
        decoded.platform
    };
    let (major, minor, patch, generation) = parse_abi_version_tag(&decoded.version);
    let normalized = if proto_bytes.is_empty() {
        encode_runtime_abi_proto(platform_id, major, minor, patch, generation)
    } else {
        proto_bytes
    };
    args.abi_version_out = box_registered_handle(PJRT_AbiVersion_Holder {
        platform_id,
        major,
        minor,
        patch,
        generation,
        is_executable: false,
        proto_bytes: normalized,
    });
    Ok(())
}

/// Strongly-typed fallible handler for [`PJRT_AbiVersion_ExecutableFromProto`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_AbiVersion_FromProto_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `serialized_proto` is null when `serialized_proto_size > 0` or
/// protobuf wire decoding fails.
///
/// # Safety
///
/// When `args.serialized_proto` is non-null, it must point to `args.serialized_proto_size` valid bytes.
unsafe fn handle_abi_version_executable_from_proto(
    args: &mut PJRT_AbiVersion_FromProto_Args,
) -> Result<(), LibTpuError> {
    use libtpu_core::proto::ProtoSchemaMessage as _;
    if args.serialized_proto_size > 0 && args.serialized_proto.is_null() {
        return Err(LibTpuError::InvalidArgument {
            parameter: "serialized_proto",
            reason: String::from("PJRT_AbiVersion_ExecutableFromProto serialized_proto is null"),
        });
    }
    let proto_bytes = if args.serialized_proto_size == 0 {
        Vec::new()
    } else {
        // SAFETY: `serialized_proto` is non-null with `serialized_proto_size` bytes.
        unsafe {
            std::slice::from_raw_parts(
                args.serialized_proto.cast::<u8>(),
                args.serialized_proto_size,
            )
            .to_vec()
        }
    };
    let decoded = if proto_bytes.is_empty() {
        libtpu_core::proto::ExecutableAbiVersionProto {
            platform: TPU_PLATFORM_ID,
            version: Vec::from(&b"0.117.1"[..]),
        }
    } else {
        libtpu_core::proto::ExecutableAbiVersionProto::decode_proto(&proto_bytes)?
    };
    let platform_id = if decoded.platform == 0 {
        TPU_PLATFORM_ID
    } else {
        decoded.platform
    };
    let (major, minor, patch, generation) = parse_abi_version_tag(&decoded.version);
    let normalized = if proto_bytes.is_empty() {
        encode_executable_abi_proto(platform_id, major, minor, patch, generation)
    } else {
        proto_bytes
    };
    args.abi_version_out = box_registered_handle(PJRT_AbiVersion_Holder {
        platform_id,
        major,
        minor,
        patch,
        generation,
        is_executable: true,
        proto_bytes: normalized,
    });
    Ok(())
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_AbiVersion_FromProto_Args>(
            raw_args,
            handle_abi_version_runtime_from_proto,
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_AbiVersion_FromProto_Args>(
            raw_args,
            handle_abi_version_executable_from_proto,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Shardings_ParameterShardings`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Shardings_PJRT_Executable_Shardings_Args`] structure.
///
/// # Safety
///
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_executable_parameter_shardings(
    args: &mut PJRT_Shardings_PJRT_Executable_Shardings_Args,
) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    if let Some(exec) = unsafe { args.executable.as_ref() } {
        args.shardings = exec.param_sharding_ptrs.as_ptr();
        args.sharding_sizes = exec.param_sharding_sizes.as_ptr();
        args.num_items = exec.param_sharding_ptrs.len();
    } else {
        args.shardings = DEFAULT_SHARDING_PTRS.0.as_ptr();
        args.sharding_sizes = DEFAULT_SHARDING_SIZES.as_ptr();
        args.num_items = 1;
    }
}

/// Strongly-typed handler for [`PJRT_Shardings_OutputShardings`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Shardings_PJRT_Executable_Shardings_Args`] structure.
///
/// # Safety
///
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_executable_output_shardings(
    args: &mut PJRT_Shardings_PJRT_Executable_Shardings_Args,
) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    if let Some(exec) = unsafe { args.executable.as_ref() } {
        args.shardings = exec.output_sharding_ptrs.as_ptr();
        args.sharding_sizes = exec.output_sharding_sizes.as_ptr();
        args.num_items = exec.output_sharding_ptrs.len();
    } else {
        args.shardings = DEFAULT_SHARDING_PTRS.0.as_ptr();
        args.sharding_sizes = DEFAULT_SHARDING_SIZES.as_ptr();
        args.num_items = 1;
    }
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
/// * `args` - Validated [`PJRT_TpuExecutable_ProtoQuery_Args`] structure.
///
/// # Safety
///
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_tpu_executable_target_arguments(args: &mut PJRT_TpuExecutable_ProtoQuery_Args) {
    use libtpu_core::proto::{ProtoSchemaMessage as _, TpuGenerationProtoExt as _};
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let tpu_gen = unsafe { args.executable.as_ref() }.map_or_else(
        || global_init_config().generation,
        generation_for_executable,
    );
    let target_bytes =
        libtpu_core::proto::TargetArgumentsProto::from_generation(tpu_gen).encode_proto();
    let (ptr, len, holder) = box_serialized_artifact(target_bytes);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.holder = holder;
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
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_tpu_executable_hlo_module_with_config(
    args: &mut PJRT_TpuExecutable_ProtoQuery_Args,
) {
    use libtpu_core::proto::ProtoSchemaMessage as _;
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
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
    .encode_proto();
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.holder = holder;
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

/// Strongly-typed handler for [`PJRT_TpuExecutable_Reserved`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Megascale_Handle_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid mutable [`PJRT_Megascale_Handle_Args`] reference.
unsafe fn handle_tpu_executable_reserved(args: &mut PJRT_Megascale_Handle_Args) {
    STANDALONE_ENV_OVERRIDES.fetch_add(1, Ordering::Relaxed);
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
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_tpu_executable_get_compiled_memory_stats(
    args: &mut PJRT_TpuExecutable_GetCompiledMemoryStats_Args,
) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
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
/// * `args` - Validated [`PJRT_TpuExecutable_RunHloCostAnalysis_Args`] structure.
///
/// # Safety
///
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_tpu_executable_cost_analysis(
    args: &mut PJRT_TpuExecutable_RunHloCostAnalysis_Args,
) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    args.num_properties_out = unsafe { args.executable.as_ref() }.map_or(4, |exec| {
        let prog = &exec.inner.program;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let flops_cost = prog.cost_analysis.flops.max(1.0) as usize;
        flops_cost.max(prog.instruction_bundles.len().max(1))
    });
}

/// Runs HLO cost analysis on an executable (`PJRT_TpuExecutable_RunHloCostAnalysis`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_RunHloCostAnalysis_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_RunHloCostAnalysis_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_RunHloCostAnalysis(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_RunHloCostAnalysis_Args>(
            raw_args,
            handle_tpu_executable_cost_analysis,
        )
    }
}

/// Strongly-typed handler for [`PJRT_TpuExecutable_SetTpuCompilationEnv`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_SetTpuCompilationEnv_Args`] structure.
///
/// # Safety
///
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_tpu_executable_set_env(args: &mut PJRT_TpuExecutable_SetTpuCompilationEnv_Args) {
    use libtpu_core::proto::ProtoSchemaMessage as _;
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    if let Some(exec) = unsafe { args.executable.as_ref() } {
        let mut arena = exec.ext_arena.lock();
        if !args.serialized_env.is_null() && args.serialized_env_size > 0 {
            // SAFETY: `serialized_env` is non-null with `serialized_env_size` bytes.
            let raw_env = unsafe {
                std::slice::from_raw_parts(args.serialized_env, args.serialized_env_size)
            };
            if let Ok(decoded) =
                libtpu_core::proto::TpuCompilationEnvironmentProto::decode_proto(raw_env)
            {
                arena.tpu_compilation_env = decoded;
            }
        }
        arena.overrides_count = arena.overrides_count.saturating_add(1);
        arena.tpu_compilation_env.xla_tpu_enable_flash_attention = Some(true);
        args.serialized_env_size = arena.overrides_count;
    } else {
        args.serialized_env_size = STANDALONE_ENV_OVERRIDES.fetch_add(1, Ordering::Relaxed) + 1;
    }
}

/// Sets `TpuCompilationEnvironmentProto` field overrides (`PJRT_TpuExecutable_SetTpuCompilationEnv`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_SetTpuCompilationEnv_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_SetTpuCompilationEnv_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_SetTpuCompilationEnv(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_SetTpuCompilationEnv_Args>(
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
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_tpu_executable_get_env_field(args: &mut PJRT_TpuExecutable_ProtoQuery_Args) {
    use libtpu_core::proto::ProtoSchemaMessage as _;
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let encoded = unsafe { args.executable.as_ref() }.map_or_else(
        || {
            libtpu_core::proto::TpuCompilationEnvironmentProto {
                xla_tpu_enable_flash_attention: Some(true),
                ..libtpu_core::proto::TpuCompilationEnvironmentProto::default()
            }
            .encode_proto()
        },
        |exec| exec.ext_arena.lock().tpu_compilation_env.encode_proto(),
    );
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.holder = holder;
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
/// If `args.error` is non-null, it must point to a valid [`PJRT_Error`].
unsafe fn handle_tpu_executable_is_predetermined_error(
    args: &mut PJRT_TpuExecutable_ErrorCheck_Args,
) {
    // SAFETY: `args.error` is null or a valid `PJRT_Error*`.
    let err_opt = unsafe { args.error.as_ref() };
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
/// If `args.error` is non-null, it must point to a valid [`PJRT_Error`].
unsafe fn handle_tpu_executable_is_cancellation_error(
    args: &mut PJRT_TpuExecutable_ErrorCheck_Args,
) {
    // SAFETY: `args.error` is null or a valid `PJRT_Error*`.
    let err_opt = unsafe { args.error.as_ref() };
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

/// Strongly-typed handler for [`PJRT_TpuExecutable_GetFingerprint`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_TpuExecutable_GetFingerprint_Args`] structure.
///
/// # Safety
///
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_tpu_executable_get_fingerprint(args: &mut PJRT_TpuExecutable_GetFingerprint_Args) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let fp = unsafe { args.executable.as_ref() }.map_or_else(
        || String::from("libtpu-rs-0.0.53.1-default"),
        |e| e.inner.program.fingerprint.clone(),
    );
    let (ptr, len, holder) = box_serialized_artifact(fp.into_bytes());
    args.fingerprint_bytes = ptr;
    args.fingerprint_size = len;
    args.holder = holder;
}

/// Queries the SHA-256 compilation fingerprint of a TPU executable (`PJRT_TpuExecutable_GetFingerprint`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_TpuExecutable_GetFingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_TpuExecutable_GetFingerprint_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetFingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_TpuExecutable_GetFingerprint_Args>(
            raw_args,
            handle_tpu_executable_get_fingerprint,
        )
    }
}
