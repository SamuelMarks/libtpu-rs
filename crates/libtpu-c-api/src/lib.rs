//! C ABI drop-in replacement for `libtpu.so`.
//!
//! Exports `GetPjrtApi`, `GetLibtpuSdkApi`, `TfTpu_Initialize`, `TpuStatus_*`,
//! `TpuTopology_*`, `TpuCoreLocation_*`, `TpuHostLocation_*`, `TpuMeshState_*`,
//! `TpuProfiler_*`, `TfTpuOrdinalSelector_*`, `TpuEmbeddingEngine_*`, `TpuDriver_*`,
//! and clean `libtpu_rs_*` entrypoints for C and C++20 consumers.

#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]
#![allow(
    non_camel_case_types,
    non_snake_case,
    unused_qualifications,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::unnecessary_wraps,
    clippy::wildcard_imports,
    clippy::missing_const_for_fn,
    clippy::significant_drop_tightening,
    clippy::if_not_else
)]

pub mod extensions;
pub mod pjrt_types;
pub mod tpu_driver_types;

use crate::extensions::EXT_01_XLA_TRANSFORM;
use crate::pjrt_types::{
    PJRT_API_MAJOR, PJRT_API_MINOR, PJRT_API_STRUCT_SIZE, PJRT_API_VERSION_STRUCT_SIZE, PJRT_Api,
    PJRT_Api_Version, PJRT_Buffer, PJRT_Client, PJRT_Client_AddressableMemories_Args,
    PJRT_Client_Compile_Args, PJRT_Client_Create_Args, PJRT_Client_Destroy_Args,
    PJRT_Client_Devices_Args, PJRT_Client_LookupDevice_Args, PJRT_Client_PlatformStr_Args,
    PJRT_Client_ProcessIndex_Args, PJRT_Device, PJRT_DeviceDescription, PJRT_Error,
    PJRT_Error_Destroy_Args, PJRT_Error_GetCode_Args, PJRT_Error_Message_Args, PJRT_Executable,
    PJRT_LoadedExecutable, PJRT_Memory, PJRT_Plugin_Initialize_Args, PJRT_TopologyDescription,
};
use crate::tpu_driver_types::{
    LibTpuCStatus, LibtpuSdk_Api, LibtpuSdk_ChipLocationToCoordinates, LibtpuSdk_Client,
    LibtpuSdk_Client_Args, LibtpuSdk_Error, LibtpuSdk_Error_Destroy_Args,
    LibtpuSdk_Error_GetCode_Args, LibtpuSdk_Error_GetMessage_Args,
    LibtpuSdk_GetCartesianCoordinates_Args, LibtpuSdk_GetChipCoordinates_Args,
    LibtpuSdk_GetChipIndex_Args, LibtpuSdk_GetHostName_Args, LibtpuSdk_GetMetric_Args,
    LibtpuSdk_GetMetricDescription_Args, LibtpuSdk_GetMetricValues_Args,
    LibtpuSdk_GetRuntimeStatus_Args, LibtpuSdk_Metric, LibtpuSdk_RegisterDebugCallback_Args,
    LibtpuSdk_RuntimeStatus, LibtpuSdk_RuntimeStatus_Destroy_Args,
    LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args, LibtpuSdk_UnregisterDebugCallback_Args,
    TFNPD_PluginParams, TfTpuOrdinalSelector, TpuCoreLocation, TpuDriver, TpuDriverFn,
    TpuEmbeddingEngine_CollateMemory_Params, TpuEmbeddingEngine_Computation_Params,
    TpuEmbeddingEngine_ConfigureHost_Params, TpuEmbeddingEngine_ConfigureMemory_Params,
    TpuEmbeddingEngine_ConnectHosts_Params, TpuEmbeddingEngine_EnqueueTensorBatch_Params,
    TpuEmbeddingEngine_ExecutePartitioner_Params, TpuEmbeddingEngine_Finalize_Params,
    TpuEmbeddingEngine_IsInitialized_Params, TpuEmbeddingEngine_WriteParameters_Params,
    TpuEmbeddingEngineState, TpuHostLocation, TpuMeshState, TpuPartitionedCall_Params, TpuProfiler,
    TpuSerializedProto, TpuStatus, TpuTopology, XLA_Shape,
};
use libtpu_core::config::TpuInitConfig;
use libtpu_core::driver::{
    LibtpuSdkCoreClient, TfTpuCoreOrdinalSelector, TpuCoreDriver, TpuCoreEmbeddingEngine,
    TpuCoreProfiler, TpuCoreStatus,
};
use libtpu_core::error::LibTpuError;
use libtpu_core::hal::SimulatedTpuHal;
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::types::{
    ByteSize, CallbackId, ChipCoordinates, DeviceAddress, DeviceId, ElementType, Shape,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::OnceLock;

/// Returns `true` if any pointer address in `addrs` is zero (`NULL`).
///
/// # Arguments
///
/// * `addrs` - Slice of pointer addresses cast to `usize`.
fn any_null(addrs: &[usize]) -> bool {
    addrs.contains(&0)
}

/// Helper to convert a Rust string into a sanitized [`CString`] without interior NUL errors.
///
/// # Arguments
///
/// * `text` - Input UTF-8 string.
fn to_c_string(text: &str) -> CString {
    let sanitized: Vec<u8> = text.bytes().filter(|&b| b != 0).collect();
    // SAFETY: `sanitized` contains no zero bytes because `filter(|&b| b != 0)` removed them.
    unsafe { CString::from_vec_unchecked(sanitized) }
}

/// Allocates a heap-backed [`PJRT_Error`] from a [`LibTpuError`].
///
/// # Arguments
///
/// * `error` - The domain error to wrap.
#[must_use]
pub fn box_pjrt_error(error: LibTpuError) -> *mut PJRT_Error {
    let c_message = to_c_string(&error.to_string());
    Box::into_raw(Box::new(PJRT_Error { error, c_message }))
}

/// Allocates a heap-backed [`LibtpuSdk_Error`] from a [`LibTpuError`].
///
/// # Arguments
///
/// * `error` - The domain error to wrap.
#[must_use]
pub fn box_sdk_error(error: &LibTpuError) -> *mut LibtpuSdk_Error {
    let c_message = to_c_string(&error.to_string());
    Box::into_raw(Box::new(LibtpuSdk_Error {
        code: error.sdk_error_code().as_raw(),
        c_message,
    }))
}

/// Monotonic counter recording `PJRT_Api` vtable dispatches.
static PJRT_SLOT_DISPATCH_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Returns the cumulative number of `PJRT_Api` slot dispatches.
#[must_use]
pub fn pjrt_slot_dispatch_count() -> u64 {
    PJRT_SLOT_DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Validates `struct_size` of a PJRT vtable argument structure (`PJRT_CheckMatchingStructSizes`).
///
/// # Arguments
///
/// * `raw_args` - Generic PJRT argument structure pointer whose first field is `size_t struct_size`.
///
/// # Safety
///
/// `raw_args` may be null or a valid pointer to a PJRT args struct beginning with `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CheckMatchingStructSizes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    if raw_args.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `raw_args` is non-null and starts with `usize struct_size`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < size_of::<usize>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Args",
            expected_min: size_of::<usize>(),
            actual: actual_size,
        });
    }
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ptr::null_mut()
}

/// Invokes a strongly-typed `unsafe fn(&mut T)` from a type-erased reference.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the `T` argument structure.
/// * `handler_ref` - Type-erased pointer to the `unsafe fn(&mut T)` callback.
///
/// # Safety
///
/// `raw_args` must point to a valid `T` and `handler_ref` must point to `unsafe fn(&mut T)`.
unsafe fn invoke_typed_pjrt_void_handler<T>(
    raw_args: *mut std::ffi::c_void,
    handler_ref: *const std::ffi::c_void,
) {
    // SAFETY: `handler_ref` points to `handler` on the stack of `dispatch_typed_pjrt_void_args`.
    let handler = unsafe { *handler_ref.cast::<unsafe fn(&mut T)>() };
    // SAFETY: `raw_args` was validated non-null by `dispatch_erased_pjrt_void_args`.
    unsafe {
        handler(&mut *raw_args.cast::<T>());
    }
}

/// Non-generic void dispatcher backing [`dispatch_typed_pjrt_void_args`].
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the `PJRT_Error_*_Args` structure.
/// * `handler_ref` - Type-erased pointer to the strongly-typed callback.
/// * `invoker` - Monomorphized trampoline invoking the strongly-typed callback.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid `T`.
unsafe fn dispatch_erased_pjrt_void_args(
    raw_args: *mut std::ffi::c_void,
    handler_ref: *const std::ffi::c_void,
    invoker: unsafe fn(*mut std::ffi::c_void, *const std::ffi::c_void),
) {
    if !raw_args.is_null() {
        PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: `raw_args` is non-null.
        unsafe {
            invoker(raw_args, handler_ref);
        }
    }
}

/// Validates `raw_args != null` and dispatches to a strongly-typed `&mut T` void handler.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to a `#[repr(C)]` `PJRT_Error_*_Args` structure of type `T`.
/// * `handler` - Strongly-typed callback invoked with `&mut T`.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid `T` structure.
#[inline]
pub(crate) unsafe fn dispatch_typed_pjrt_void_args<T>(
    raw_args: *mut std::ffi::c_void,
    handler: unsafe fn(&mut T),
) {
    let handler_ref = ptr::from_ref(&handler).cast::<std::ffi::c_void>();
    // SAFETY: Caller upholds `raw_args` contract; `handler_ref` points to `handler`.
    unsafe {
        dispatch_erased_pjrt_void_args(raw_args, handler_ref, invoke_typed_pjrt_void_handler::<T>);
    }
}

/// Invokes a strongly-typed `unsafe fn(&mut T) -> Result<(), LibTpuError>` from a type-erased reference.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the validated `T` argument structure.
/// * `handler_ref` - Type-erased pointer to the `unsafe fn(&mut T) -> Result<(), LibTpuError>` callback.
///
/// # Safety
///
/// `raw_args` must point to a valid `T` and `handler_ref` must point to `unsafe fn(&mut T) -> Result<(), LibTpuError>`.
unsafe fn invoke_typed_pjrt_handler<T>(
    raw_args: *mut std::ffi::c_void,
    handler_ref: *const std::ffi::c_void,
) -> Result<(), LibTpuError> {
    // SAFETY: `handler_ref` points to `handler` on the stack of `dispatch_typed_pjrt_args`.
    let handler = unsafe { *handler_ref.cast::<unsafe fn(&mut T) -> Result<(), LibTpuError>>() };
    // SAFETY: `raw_args` was validated non-null and `>= size_of::<T>()` by `dispatch_erased_pjrt_args`.
    unsafe { handler(&mut *raw_args.cast::<T>()) }
}

/// Non-generic validation and error-conversion dispatcher backing [`dispatch_typed_pjrt_args`].
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the `PJRT_*_Args` structure.
/// * `expected_size` - Minimum required `struct_size` in bytes.
/// * `handler_ref` - Type-erased pointer to the strongly-typed callback.
/// * `invoker` - Monomorphized trampoline invoking the strongly-typed callback.
///
/// # Safety
///
/// `raw_args` must be null or point to a `PJRT_*_Args` structure of at least `expected_size` bytes.
unsafe fn dispatch_erased_pjrt_args(
    raw_args: *mut std::ffi::c_void,
    expected_size: usize,
    handler_ref: *const std::ffi::c_void,
    invoker: unsafe fn(*mut std::ffi::c_void, *const std::ffi::c_void) -> Result<(), LibTpuError>,
) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size >= 8`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: `raw_args` is non-null and starts with `struct_size: usize`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < expected_size {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Args",
            expected_min: expected_size,
            actual: actual_size,
        });
    }
    // SAFETY: `raw_args` is non-null and has at least `expected_size` bytes.
    match unsafe { invoker(raw_args, handler_ref) } {
        Ok(()) => ptr::null_mut(),
        Err(e) => box_pjrt_error(e),
    }
}

/// Validates a `PJRT_*_Args` pointer against `T::MIN_STRUCT_SIZE` and invokes a strongly-typed `&mut T` handler.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to a `#[repr(C)]` `PJRT_*_Args` structure of type `T`.
/// * `handler` - Strongly-typed callback invoked with `&mut T` when `raw_args` is valid.
///
/// # Safety
///
/// `raw_args` must be null or point to a `T` structure whose first field is `struct_size: usize`.
#[inline]
pub(crate) unsafe fn dispatch_typed_pjrt_args<T: crate::pjrt_types::PjrtArgsStruct>(
    raw_args: *mut std::ffi::c_void,
    handler: unsafe fn(&mut T) -> Result<(), LibTpuError>,
) -> *mut PJRT_Error {
    let handler_ref = ptr::from_ref(&handler).cast::<std::ffi::c_void>();
    // SAFETY: Caller upholds `raw_args` contract; `handler_ref` points to `handler`.
    unsafe {
        dispatch_erased_pjrt_args(
            raw_args,
            T::MIN_STRUCT_SIZE,
            handler_ref,
            invoke_typed_pjrt_handler::<T>,
        )
    }
}

/// Legacy TensorFlow TPU C API entrypoints.
pub mod legacy_tf_api;
/// PJRT buffer, copy-to-device stream, and async transfer manager C API entrypoints.
pub mod pjrt_buffer_api;
/// PJRT client, topology description, and execution context C API entrypoints.
pub mod pjrt_client_api;
/// PJRT device, device description, and memory space C API entrypoints.
pub mod pjrt_device_api;
/// PJRT error, plugin initialization, and event C API entrypoints.
pub mod pjrt_error_api;
/// PJRT executable and loaded executable C API entrypoints.
pub mod pjrt_executable_api;
/// Ergonomic `libtpu_rs_*` C/C++/Python FFI helper entrypoints.
pub mod rust_ergonomic_api;
/// `LibtpuSdk_Api` (`tpumonitoring`, `tpuz`, `debug`, `slice`) C API entrypoints.
pub mod sdk_api;

pub use legacy_tf_api::*;
pub use pjrt_buffer_api::*;
pub use pjrt_client_api::*;
pub use pjrt_device_api::*;
pub use pjrt_error_api::*;
pub use pjrt_executable_api::*;
pub use rust_ergonomic_api::*;
pub use sdk_api::*;

/// Deleter callback for heap-allocated [`crate::pjrt_types::PjrtSerializedArtifact`] holders.
///
/// Used by `PJRT_SerializedExecutable`, `PJRT_SerializedTopology`,
/// `PJRT_SerializedDeviceAssignment`, and `PJRT_SerializedCompileOptions`.
///
/// # Arguments
///
/// * `holder` - Opaque [`crate::pjrt_types::PjrtSerializedArtifact`] pointer.
#[unsafe(no_mangle)]
pub extern "C" fn pjrt_serialized_artifact_deleter(holder: *mut std::ffi::c_void) {
    if !holder.is_null() {
        // SAFETY: `holder` was allocated via `Box::into_raw(Box::new(PjrtSerializedArtifact { .. }))`.
        unsafe {
            drop(Box::from_raw(
                holder.cast::<crate::pjrt_types::PjrtSerializedArtifact>(),
            ));
        }
    }
}

/// Backwards-compatible alias for [`pjrt_serialized_artifact_deleter`].
///
/// # Arguments
///
/// * `holder` - Opaque serialized artifact holder pointer.
#[unsafe(no_mangle)]
pub extern "C" fn pjrt_serialized_noop_deleter(holder: *mut std::ffi::c_void) {
    pjrt_serialized_artifact_deleter(holder);
}

/// Allocates a heap-owned [`crate::pjrt_types::PjrtSerializedArtifact`] and returns `(bytes_ptr, bytes_len, holder_ptr)`.
///
/// # Arguments
///
/// * `bytes` - Owned serialized byte buffer to wrap.
fn box_serialized_artifact(
    bytes: Vec<u8>,
) -> (
    *const c_char,
    usize,
    *mut crate::pjrt_types::PjrtSerializedArtifact,
) {
    let holder = Box::new(crate::pjrt_types::PjrtSerializedArtifact { bytes });
    let ptr = holder.bytes.as_ptr().cast::<c_char>();
    let len = holder.bytes.len();
    (ptr, len, Box::into_raw(holder))
}

/// Wraps a [`libtpu_core::pjrt::PjrtCoreTopology`] into a heap-allocated [`PJRT_TopologyDescription`].
///
/// # Arguments
///
/// * `core_topo` - Core topology instance to wrap.
fn box_topology_description(
    core_topo: libtpu_core::pjrt::PjrtCoreTopology,
) -> *mut PJRT_TopologyDescription {
    let mut desc_ptrs = Vec::with_capacity(core_topo.device_descriptions.len());
    for desc in &core_topo.device_descriptions {
        desc_ptrs.push(Box::into_raw(Box::new(PJRT_DeviceDescription::new(
            desc.clone(),
        ))));
    }
    Box::into_raw(Box::new(PJRT_TopologyDescription {
        inner: core_topo,
        desc_ptrs,
    }))
}

/// Allocates a boxed [`PJRT_TopologyDescription`] from the global [`libtpu_core::config::TpuInitConfig`].
fn box_default_topology_description() -> *mut PJRT_TopologyDescription {
    let cfg = libtpu_core::config::global_init_config();
    let hal = libtpu_core::hal::SimulatedTpuHal::new(&cfg);
    let spec = hal.topology_spec();
    box_topology_description(libtpu_core::pjrt::PjrtCoreTopology::new(spec))
}

/// Allocates a [`PJRT_Buffer`] on `client_ptr`, `device_ptr`, and `memory_ptr` with the requested shape.
///
/// # Arguments
///
/// * `client_ptr` - Optional owning `PJRT_Client*`.
/// * `device_ptr` - Optional target `PJRT_Device*`.
/// * `memory_ptr` - Optional target `PJRT_Memory*`.
/// * `raw_element_type` - Raw `PJRT_Buffer_Type` (`i32`).
/// * `dims_ptr` - Optional dimensions array pointer.
/// * `num_dims` - Number of dimensions.
fn alloc_shaped_pjrt_buffer(
    client_ptr: *mut PJRT_Client,
    device_ptr: *mut PJRT_Device,
    memory_ptr: *mut PJRT_Memory,
    raw_element_type: i32,
    dims_ptr: *const i64,
    num_dims: usize,
) -> *mut PJRT_Buffer {
    let elem = ElementType::from_raw(raw_element_type).unwrap_or(ElementType::F32);
    let dims = if dims_ptr.is_null() {
        vec![4]
    } else {
        // SAFETY: `dims_ptr` is non-null and points to `num_dims` `i64` elements.
        unsafe { std::slice::from_raw_parts(dims_ptr, num_dims).to_vec() }
    };
    let shape = Shape::new(elem, dims, None, None).unwrap_or_else(|_| Shape {
        element_type: ElementType::F32,
        dims: vec![4],
        dynamic_dims: vec![false],
        layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
    });
    // SAFETY: `client_ptr`, `device_ptr`, and `memory_ptr` are null or valid C handles.
    let effective_dev_ptr = unsafe {
        if !device_ptr.is_null() {
            device_ptr
        } else if let Some(mem_ref) = memory_ptr.as_ref() {
            mem_ref
                .addressable_by_device_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut())
        } else if let Some(client_ref) = client_ptr.as_ref() {
            client_ref
                .addressable_device_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut())
        } else {
            ptr::null_mut()
        }
    };
    // SAFETY: `effective_dev_ptr` is null or a valid `PJRT_Device*`.
    let (resolved_dev_ptr, resolved_mem_ptr, dev_id, mem_id, hal) = unsafe {
        effective_dev_ptr.as_ref().map_or_else(
            || {
                let cfg = libtpu_core::config::global_init_config();
                (
                    effective_dev_ptr,
                    memory_ptr,
                    DeviceId(0),
                    libtpu_core::types::MemorySpaceId(0),
                    libtpu_core::hal::create_default_hal(&cfg),
                )
            },
            |dev_ref| {
                let mem_out = if memory_ptr.is_null() {
                    dev_ref.default_memory_ptr
                } else {
                    memory_ptr
                };
                (
                    effective_dev_ptr,
                    mem_out,
                    dev_ref.inner.description.spec.device_id,
                    dev_ref.inner.default_memory_id,
                    std::sync::Arc::clone(&dev_ref.hal),
                )
            },
        )
    };
    let on_dev = shape.on_device_byte_size();
    let addr = hal
        .allocate_device_memory(dev_id, libtpu_core::types::MemoryKind::DeviceHbm, on_dev)
        .unwrap_or(libtpu_core::types::DeviceAddress(0x1000));
    let core_buf = libtpu_core::pjrt::PjrtCoreBuffer::new(dev_id, mem_id, shape, addr, hal);
    Box::into_raw(Box::new(PJRT_Buffer::new(
        core_buf,
        resolved_dev_ptr,
        resolved_mem_ptr,
    )))
}

/// Allocates a default 4-element `F32` [`PJRT_Buffer`] on `device_ptr` and `memory_ptr`.
///
/// # Arguments
///
/// * `device_ptr` - Optional target `PJRT_Device*`.
/// * `memory_ptr` - Optional target `PJRT_Memory*`.
fn alloc_default_pjrt_buffer(
    device_ptr: *mut PJRT_Device,
    memory_ptr: *mut PJRT_Memory,
) -> *mut PJRT_Buffer {
    alloc_shaped_pjrt_buffer(ptr::null_mut(), device_ptr, memory_ptr, 11, ptr::null(), 0)
}

/// Wrapper allowing static `*const c_char` arrays in `static` items.
#[repr(transparent)]
struct SyncCharPtrArray<const N: usize>([*const c_char; N]);

// SAFETY: Contains only `'static` C-string literal pointers.
unsafe impl<const N: usize> Sync for SyncCharPtrArray<N> {}

/// Static default memory kind string pointer array (`["device"]`).
static DEFAULT_MEMORY_KIND_PTRS: SyncCharPtrArray<1> = SyncCharPtrArray([c"device".as_ptr()]);

/// Static default memory kind byte length array (`[6]`).
static DEFAULT_MEMORY_KIND_SIZES: [usize; 1] = [6];

/// Static default output element types array (`[F32 = 11]`).
static DEFAULT_OUTPUT_ELEMENT_TYPES: [i32; 1] = [11];

/// Static default output dimensions array (`[4]`).
static DEFAULT_OUTPUT_DIMS: [i64; 1] = [4];

/// Static default output dimension rank array (`[1]`).
static DEFAULT_OUTPUT_DIM_SIZES: [usize; 1] = [1];

/// Static default logical device IDs array (`[(0, 0)]`).
static GLOBAL_LOGICAL_DEVICE_IDS: [crate::pjrt_types::PJRT_LogicalDeviceIds; 1] =
    [crate::pjrt_types::PJRT_LogicalDeviceIds {
        replica: 0,
        partition: 0,
    }];

/// Static default memory space kind IDs array (`[0, 1, 2, 3]`).
static GLOBAL_MEMORY_KIND_IDS: [i32; 4] = [0, 1, 2, 3];

/// Static plugin attributes array returned by [`PJRT_Plugin_Attributes`].
static GLOBAL_PLUGIN_ATTRIBUTES: [crate::pjrt_types::PJRT_NamedValue; 3] = [
    crate::pjrt_types::PJRT_NamedValue {
        struct_size: size_of::<crate::pjrt_types::PJRT_NamedValue>(),
        extension_start: ptr::null_mut(),
        name: c"xla_tpu_version".as_ptr(),
        name_size: 15,
        value_type_and_pad: 1,
        value_payload: 50,
        value_size: 1,
    },
    crate::pjrt_types::PJRT_NamedValue {
        struct_size: size_of::<crate::pjrt_types::PJRT_NamedValue>(),
        extension_start: ptr::null_mut(),
        name: c"tpu_generation".as_ptr(),
        name_size: 14,
        value_type_and_pad: 1,
        value_payload: 5,
        value_size: 1,
    },
    crate::pjrt_types::PJRT_NamedValue {
        struct_size: size_of::<crate::pjrt_types::PJRT_NamedValue>(),
        extension_start: ptr::null_mut(),
        name: c"supports_cross_host_transfers".as_ptr(),
        name_size: 29,
        value_type_and_pad: 3,
        value_payload: 1,
        value_size: 1,
    },
];

// SAFETY: `GLOBAL_PLUGIN_ATTRIBUTES` contains only static `'static` C-string pointers and integer constants.
unsafe impl Sync for crate::pjrt_types::PJRT_NamedValue {}

/// Static 1,144-byte `PJRT_Api` vtable returned by [`GetPjrtApi`], with all 138 slots wired to dedicated functions.
static GLOBAL_PJRT_API: PJRT_Api = PJRT_Api {
    struct_size: PJRT_API_STRUCT_SIZE,
    extension_start: ptr::addr_of!(EXT_01_XLA_TRANSFORM.base),
    pjrt_api_version: PJRT_Api_Version {
        struct_size: PJRT_API_VERSION_STRUCT_SIZE,
        extension_start: ptr::null(),
        major_version: PJRT_API_MAJOR,
        minor_version: PJRT_API_MINOR,
    },
    pjrt_error_destroy: PJRT_Error_Destroy,
    pjrt_error_message: PJRT_Error_Message,
    pjrt_error_get_code: PJRT_Error_GetCode,
    pjrt_plugin_initialize: PJRT_Plugin_Initialize,
    pjrt_plugin_attributes: PJRT_Plugin_Attributes,
    pjrt_event_destroy: PJRT_Event_Destroy,
    pjrt_event_is_ready: PJRT_Event_IsReady,
    pjrt_event_error: PJRT_Event_Error,
    pjrt_event_await: PJRT_Event_Await,
    pjrt_event_on_ready: PJRT_Event_OnReady,
    pjrt_client_create: PJRT_Client_Create,
    pjrt_client_destroy: PJRT_Client_Destroy,
    pjrt_client_platform_name: PJRT_Client_PlatformName,
    pjrt_client_process_index: PJRT_Client_ProcessIndex,
    pjrt_client_platform_version: PJRT_Client_PlatformVersion,
    pjrt_client_devices: PJRT_Client_Devices,
    pjrt_client_addressable_devices: PJRT_Client_AddressableDevices,
    pjrt_client_lookup_device: PJRT_Client_LookupDevice,
    pjrt_client_lookup_addressable_device: PJRT_Client_LookupAddressableDevice,
    pjrt_client_addressable_memories: PJRT_Client_AddressableMemories,
    pjrt_client_compile: PJRT_Client_Compile,
    pjrt_client_default_device_assignment: PJRT_Client_DefaultDeviceAssignment,
    pjrt_client_buffer_from_host_buffer: PJRT_Client_BufferFromHostBuffer,
    pjrt_device_description_id: PJRT_DeviceDescription_Id,
    pjrt_device_description_process_index: PJRT_DeviceDescription_ProcessIndex,
    pjrt_device_description_attributes: PJRT_DeviceDescription_Attributes,
    pjrt_device_description_kind: PJRT_DeviceDescription_Kind,
    pjrt_device_description_debug_string: PJRT_DeviceDescription_DebugString,
    pjrt_device_description_to_string: PJRT_DeviceDescription_ToString,
    pjrt_device_get_description: PJRT_Device_GetDescription,
    pjrt_device_is_addressable: PJRT_Device_IsAddressable,
    pjrt_device_local_hardware_id: PJRT_Device_LocalHardwareId,
    pjrt_device_addressable_memories: PJRT_Device_AddressableMemories,
    pjrt_device_default_memory: PJRT_Device_DefaultMemory,
    pjrt_device_memory_stats: PJRT_Device_MemoryStats,
    pjrt_memory_id: PJRT_Memory_Id,
    pjrt_memory_kind: PJRT_Memory_Kind,
    pjrt_memory_debug_string: PJRT_Memory_DebugString,
    pjrt_memory_to_string: PJRT_Memory_ToString,
    pjrt_memory_addressable_by_devices: PJRT_Memory_AddressableByDevices,
    pjrt_executable_destroy: PJRT_Executable_Destroy,
    pjrt_executable_name: PJRT_Executable_Name,
    pjrt_executable_num_replicas: PJRT_Executable_NumReplicas,
    pjrt_executable_num_partitions: PJRT_Executable_NumPartitions,
    pjrt_executable_num_outputs: PJRT_Executable_NumOutputs,
    pjrt_executable_size_of_generated_code_in_bytes: PJRT_Executable_SizeOfGeneratedCodeInBytes,
    pjrt_executable_get_cost_analysis: PJRT_Executable_GetCostAnalysis,
    pjrt_executable_output_memory_kinds: PJRT_Executable_OutputMemoryKinds,
    pjrt_executable_optimized_program: PJRT_Executable_OptimizedProgram,
    pjrt_executable_serialize: PJRT_Executable_Serialize,
    pjrt_loaded_executable_destroy: PJRT_LoadedExecutable_Destroy,
    pjrt_loaded_executable_get_executable: PJRT_LoadedExecutable_GetExecutable,
    pjrt_loaded_executable_addressable_devices: PJRT_LoadedExecutable_AddressableDevices,
    pjrt_loaded_executable_delete: PJRT_LoadedExecutable_Delete,
    pjrt_loaded_executable_is_deleted: PJRT_LoadedExecutable_IsDeleted,
    pjrt_loaded_executable_execute: PJRT_LoadedExecutable_Execute,
    pjrt_executable_deserialize_and_load: PJRT_Executable_DeserializeAndLoad,
    pjrt_loaded_executable_fingerprint: PJRT_LoadedExecutable_Fingerprint,
    pjrt_buffer_destroy: PJRT_Buffer_Destroy,
    pjrt_buffer_element_type: PJRT_Buffer_ElementType,
    pjrt_buffer_dimensions: PJRT_Buffer_Dimensions,
    pjrt_buffer_unpadded_dimensions: PJRT_Buffer_UnpaddedDimensions,
    pjrt_buffer_dynamic_dimension_indices: PJRT_Buffer_DynamicDimensionIndices,
    pjrt_buffer_get_memory_layout: PJRT_Buffer_GetMemoryLayout,
    pjrt_buffer_on_device_size_in_bytes: PJRT_Buffer_OnDeviceSizeInBytes,
    pjrt_buffer_device: PJRT_Buffer_Device,
    pjrt_buffer_memory: PJRT_Buffer_Memory,
    pjrt_buffer_delete: PJRT_Buffer_Delete,
    pjrt_buffer_is_deleted: PJRT_Buffer_IsDeleted,
    pjrt_buffer_copy_to_device: PJRT_Buffer_CopyToDevice,
    pjrt_buffer_to_host_buffer: PJRT_Buffer_ToHostBuffer,
    pjrt_buffer_is_on_cpu: PJRT_Buffer_IsOnCpu,
    pjrt_buffer_ready_event: PJRT_Buffer_ReadyEvent,
    pjrt_buffer_unsafe_pointer: PJRT_Buffer_UnsafePointer,
    pjrt_buffer_increase_external_reference_count: PJRT_Buffer_IncreaseExternalReferenceCount,
    pjrt_buffer_decrease_external_reference_count: PJRT_Buffer_DecreaseExternalReferenceCount,
    pjrt_buffer_opaque_device_memory_data_pointer: PJRT_Buffer_OpaqueDeviceMemoryDataPointer,
    pjrt_copy_to_device_stream_destroy: PJRT_CopyToDeviceStream_Destroy,
    pjrt_copy_to_device_stream_add_chunk: PJRT_CopyToDeviceStream_AddChunk,
    pjrt_copy_to_device_stream_total_bytes: PJRT_CopyToDeviceStream_TotalBytes,
    pjrt_copy_to_device_stream_granule_size: PJRT_CopyToDeviceStream_GranuleSize,
    pjrt_copy_to_device_stream_current_bytes: PJRT_CopyToDeviceStream_CurrentBytes,
    pjrt_topology_description_create: PJRT_TopologyDescription_Create,
    pjrt_topology_description_destroy: PJRT_TopologyDescription_Destroy,
    pjrt_topology_description_platform_name: PJRT_TopologyDescription_PlatformName,
    pjrt_topology_description_platform_version: PJRT_TopologyDescription_PlatformVersion,
    pjrt_topology_description_get_device_descriptions:
        PJRT_TopologyDescription_GetDeviceDescriptions,
    pjrt_topology_description_serialize: PJRT_TopologyDescription_Serialize,
    pjrt_topology_description_attributes: PJRT_TopologyDescription_Attributes,
    pjrt_compile: PJRT_Compile,
    pjrt_executable_output_element_types: PJRT_Executable_OutputElementTypes,
    pjrt_executable_output_dimensions: PJRT_Executable_OutputDimensions,
    pjrt_buffer_copy_to_memory: PJRT_Buffer_CopyToMemory,
    pjrt_client_create_view_of_device_buffer: PJRT_Client_CreateViewOfDeviceBuffer,
    pjrt_executable_fingerprint: PJRT_Executable_Fingerprint,
    pjrt_client_topology_description: PJRT_Client_TopologyDescription,
    pjrt_executable_get_compiled_memory_stats: PJRT_Executable_GetCompiledMemoryStats,
    pjrt_memory_kind_id: PJRT_Memory_Kind_Id,
    pjrt_execute_context_create: PJRT_ExecuteContext_Create,
    pjrt_execute_context_destroy: PJRT_ExecuteContext_Destroy,
    pjrt_buffer_copy_raw_to_host: PJRT_Buffer_CopyRawToHost,
    pjrt_async_transfer_manager_destroy: PJRT_AsyncHostToDeviceTransferManager_Destroy,
    pjrt_async_transfer_manager_transfer_data: PJRT_AsyncHostToDeviceTransferManager_TransferData,
    pjrt_client_create_buffers_for_async_host_to_device:
        PJRT_Client_CreateBuffersForAsyncHostToDevice,
    pjrt_async_transfer_manager_retrieve_buffer:
        PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
    pjrt_async_transfer_manager_device: PJRT_AsyncHostToDeviceTransferManager_Device,
    pjrt_async_transfer_manager_buffer_count: PJRT_AsyncHostToDeviceTransferManager_BufferCount,
    pjrt_async_transfer_manager_buffer_size: PJRT_AsyncHostToDeviceTransferManager_BufferSize,
    pjrt_async_transfer_manager_set_buffer_error:
        PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
    pjrt_async_transfer_manager_add_metadata: PJRT_AsyncHostToDeviceTransferManager_AddMetadata,
    pjrt_client_dma_map: PJRT_Client_DmaMap,
    pjrt_client_dma_unmap: PJRT_Client_DmaUnmap,
    pjrt_client_create_uninitialized_buffer: PJRT_Client_CreateUninitializedBuffer,
    pjrt_client_update_global_process_info: PJRT_Client_UpdateGlobalProcessInfo,
    pjrt_topology_description_deserialize: PJRT_TopologyDescription_Deserialize,
    pjrt_client_create_alias_buffer: PJRT_Client_CreateAliasBuffer,
    pjrt_client_fulfill_alias_buffer: PJRT_Client_FulfillAliasBuffer,
    pjrt_loaded_executable_get_device_assignment: PJRT_LoadedExecutable_GetDeviceAssignment,
    pjrt_client_create_error_buffer: PJRT_Client_CreateErrorBuffer,
    pjrt_async_transfer_manager_transfer_literal:
        PJRT_AsyncHostToDeviceTransferManager_TransferLiteral,
    pjrt_buffer_copy_raw_to_host_future: PJRT_Buffer_CopyRawToHostFuture,
    pjrt_device_poison_execution: PJRT_Device_PoisonExecution,
    pjrt_device_create_async_tracking_event: PJRT_Device_CreateAsyncTrackingEvent,
    pjrt_async_tracking_event_destroy: PJRT_AsyncTrackingEvent_Destroy,
    pjrt_executable_get_compile_options: PJRT_Executable_GetCompileOptions,
    pjrt_buffer_donate_with_control_dependency: PJRT_Buffer_DonateWithControlDependency,
    pjrt_event_create: PJRT_Event_Create,
    pjrt_event_set: PJRT_Event_Set,
    pjrt_device_get_attributes: PJRT_Device_GetAttributes,
    pjrt_client_load: PJRT_Client_Load,
    pjrt_loaded_executable_addressable_device_logical_ids:
        PJRT_LoadedExecutable_AddressableDeviceLogicalIds,
    pjrt_buffer_bitcast: PJRT_Buffer_Bitcast,
    pjrt_error_for_each_payload: PJRT_Error_ForEachPayload,
    pjrt_topology_description_fingerprint: PJRT_TopologyDescription_Fingerprint,
    pjrt_executable_parameter_memory_kinds: PJRT_Executable_ParameterMemoryKinds,
    pjrt_device_clear_memory_stats: PJRT_Device_ClearMemoryStats,
    pjrt_topology_description_make_canonical_shape_for_memory_space:
        PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace,
    pjrt_topology_description_get_memory_space_kind_ids:
        PJRT_TopologyDescription_GetMemorySpaceKindIds,
};

/// Primary PJRT C API entrypoint exported by `libtpu.so` (`GetPjrtApi`).
#[unsafe(no_mangle)]
pub extern "C" fn GetPjrtApi() -> *const PJRT_Api {
    ptr::addr_of!(GLOBAL_PJRT_API)
}

/// Returns the ABI version tuple and core primitive defaults.
#[must_use]
pub const fn abi_version_tuple() -> (i32, i32, DeviceAddress, ByteSize, ChipCoordinates) {
    (
        PJRT_API_MAJOR,
        PJRT_API_MINOR,
        DeviceAddress(0),
        ByteSize(PJRT_API_STRUCT_SIZE),
        ChipCoordinates::new(1, 1, 1),
    )
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_types::{
        PJRT_AsyncHostToDeviceTransferManager, PJRT_CopyToDeviceStream, PJRT_Event,
        PJRT_ExecuteContext, PJRT_Program,
    };
    use crate::tpu_driver_types::LibtpuSdk_Client;
    use libtpu_core::driver::LibtpuSdkCoreClient;
    use libtpu_core::pjrt::{
        PjrtCoreAsyncTransferManager, PjrtCoreCopyToDeviceStream, PjrtCoreEvent,
    };

    #[test]
    fn test_pjrt_api_vtable_and_lifecycle() {
        let tup = abi_version_tuple();
        assert_eq!(tup.0, 0);
        assert_eq!(tup.1, 116);
        let cstr_with_nul = to_c_string("a\0b");
        assert_eq!(cstr_with_nul.as_bytes(), b"ab");

        let api_ptr = GetPjrtApi();
        assert!(!api_ptr.is_null());
        // SAFETY: `api_ptr` points to `GLOBAL_PJRT_API`.
        let api = unsafe { *api_ptr };
        assert_eq!(api.struct_size, 1144);
        assert_eq!(api.pjrt_api_version.struct_size, 24);
        assert_eq!(api.pjrt_api_version.major_version, 0);
        assert_eq!(api.pjrt_api_version.minor_version, 116);
        assert!(!api.extension_start.is_null());
        assert_ne!(format!("{:?}", api.pjrt_api_version), "");

        let mut count = 0;
        let mut curr = api.extension_start;
        while !curr.is_null() {
            count += 1;
            // SAFETY: Static extension array.
            let ext = unsafe { *curr };
            assert_ne!(format!("{ext:?}"), "");
            curr = ext.next;
        }
        assert_eq!(count, 17);

        let ev = PJRT_Event {
            inner: PjrtCoreEvent::ready_ok(),
        };
        let ctx = PJRT_ExecuteContext { context_id: 1 };
        let stm = PJRT_CopyToDeviceStream {
            inner: PjrtCoreCopyToDeviceStream::new(256, 128),
        };
        let sdk_cli = LibtpuSdk_Client {
            inner: LibtpuSdkCoreClient::new(),
            host_names: Vec::new(),
            coords_entries: Vec::new(),
        };
        assert_ne!(format!("{ev:?}{ctx:?}{stm:?}{sdk_cli:?}"), "");
        assert_ne!(
            format!(
                "{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}",
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API,
                crate::extensions::EXT_01_XLA_TRANSFORM,
                crate::extensions::EXT_02_HOST_ALLOCATOR,
                crate::extensions::EXT_03_MULTI_SLICE,
                crate::extensions::EXT_04_COLLECTIVES,
                crate::extensions::EXT_05_ABI_VERSION,
                crate::extensions::EXT_06_SHARDINGS,
                crate::extensions::EXT_07_MEGASCALE,
                crate::extensions::EXT_08_TPU_EXECUTABLE,
                crate::extensions::EXT_09_TPU_TOPOLOGY,
                crate::extensions::EXT_10_CALLBACK,
                crate::extensions::EXT_11_PHASE_COMPILE,
                crate::extensions::EXT_12_CROSS_HOST,
                crate::extensions::EXT_13_EXEC_METADATA,
                crate::extensions::EXT_14_MEM_DESC,
                crate::extensions::EXT_15_LAYOUTS,
                crate::extensions::EXT_16_RAW_BUFFER,
                crate::extensions::EXT_17_PROFILER,
            ),
            ""
        );

        // SAFETY: Testing PJRT C API functions with both null and valid structs.
        unsafe {
            for slot_fn in [
                api.pjrt_plugin_attributes,
                api.pjrt_event_destroy,
                api.pjrt_event_is_ready,
                api.pjrt_event_error,
                api.pjrt_event_await,
                api.pjrt_event_on_ready,
                api.pjrt_client_lookup_addressable_device,
                api.pjrt_client_default_device_assignment,
                api.pjrt_client_buffer_from_host_buffer,
                api.pjrt_device_description_id,
                api.pjrt_device_description_process_index,
                api.pjrt_device_description_attributes,
                api.pjrt_device_description_kind,
                api.pjrt_device_description_debug_string,
                api.pjrt_device_description_to_string,
                api.pjrt_device_get_description,
                api.pjrt_device_is_addressable,
                api.pjrt_device_local_hardware_id,
                api.pjrt_device_addressable_memories,
                api.pjrt_device_default_memory,
                api.pjrt_device_memory_stats,
                api.pjrt_memory_id,
                api.pjrt_memory_kind,
                api.pjrt_memory_debug_string,
                api.pjrt_memory_to_string,
                api.pjrt_memory_addressable_by_devices,
                api.pjrt_executable_destroy,
                api.pjrt_executable_name,
                api.pjrt_executable_num_replicas,
                api.pjrt_executable_num_partitions,
                api.pjrt_executable_num_outputs,
                api.pjrt_executable_size_of_generated_code_in_bytes,
                api.pjrt_executable_get_cost_analysis,
                api.pjrt_executable_output_memory_kinds,
                api.pjrt_executable_optimized_program,
                api.pjrt_executable_serialize,
                api.pjrt_loaded_executable_destroy,
                api.pjrt_loaded_executable_get_executable,
                api.pjrt_loaded_executable_addressable_devices,
                api.pjrt_loaded_executable_delete,
                api.pjrt_loaded_executable_is_deleted,
                api.pjrt_loaded_executable_execute,
                api.pjrt_executable_deserialize_and_load,
                api.pjrt_loaded_executable_fingerprint,
                api.pjrt_buffer_destroy,
                api.pjrt_buffer_element_type,
                api.pjrt_buffer_dimensions,
                api.pjrt_buffer_unpadded_dimensions,
                api.pjrt_buffer_dynamic_dimension_indices,
                api.pjrt_buffer_get_memory_layout,
                api.pjrt_buffer_on_device_size_in_bytes,
                api.pjrt_buffer_device,
                api.pjrt_buffer_memory,
                api.pjrt_buffer_delete,
                api.pjrt_buffer_is_deleted,
                api.pjrt_buffer_copy_to_device,
                api.pjrt_buffer_to_host_buffer,
                api.pjrt_buffer_is_on_cpu,
                api.pjrt_buffer_ready_event,
                api.pjrt_buffer_unsafe_pointer,
                api.pjrt_buffer_increase_external_reference_count,
                api.pjrt_buffer_decrease_external_reference_count,
                api.pjrt_buffer_opaque_device_memory_data_pointer,
                api.pjrt_copy_to_device_stream_destroy,
                api.pjrt_copy_to_device_stream_add_chunk,
                api.pjrt_copy_to_device_stream_total_bytes,
                api.pjrt_copy_to_device_stream_granule_size,
                api.pjrt_copy_to_device_stream_current_bytes,
                api.pjrt_topology_description_create,
                api.pjrt_topology_description_destroy,
                api.pjrt_topology_description_platform_name,
                api.pjrt_topology_description_platform_version,
                api.pjrt_topology_description_get_device_descriptions,
                api.pjrt_topology_description_serialize,
                api.pjrt_topology_description_attributes,
                api.pjrt_compile,
                api.pjrt_executable_output_element_types,
                api.pjrt_executable_output_dimensions,
                api.pjrt_buffer_copy_to_memory,
                api.pjrt_client_create_view_of_device_buffer,
                api.pjrt_executable_fingerprint,
                api.pjrt_client_topology_description,
                api.pjrt_executable_get_compiled_memory_stats,
                api.pjrt_memory_kind_id,
                api.pjrt_execute_context_create,
                api.pjrt_execute_context_destroy,
                api.pjrt_buffer_copy_raw_to_host,
                api.pjrt_async_transfer_manager_destroy,
                api.pjrt_async_transfer_manager_transfer_data,
                api.pjrt_client_create_buffers_for_async_host_to_device,
                api.pjrt_async_transfer_manager_retrieve_buffer,
                api.pjrt_async_transfer_manager_device,
                api.pjrt_async_transfer_manager_buffer_count,
                api.pjrt_async_transfer_manager_buffer_size,
                api.pjrt_async_transfer_manager_set_buffer_error,
                api.pjrt_async_transfer_manager_add_metadata,
                api.pjrt_client_dma_map,
                api.pjrt_client_dma_unmap,
                api.pjrt_client_create_uninitialized_buffer,
                api.pjrt_client_update_global_process_info,
                api.pjrt_topology_description_deserialize,
                api.pjrt_client_create_alias_buffer,
                api.pjrt_client_fulfill_alias_buffer,
                api.pjrt_loaded_executable_get_device_assignment,
                api.pjrt_client_create_error_buffer,
                api.pjrt_async_transfer_manager_transfer_literal,
                api.pjrt_buffer_copy_raw_to_host_future,
                api.pjrt_device_poison_execution,
                api.pjrt_device_create_async_tracking_event,
                api.pjrt_async_tracking_event_destroy,
                api.pjrt_executable_get_compile_options,
                api.pjrt_buffer_donate_with_control_dependency,
                api.pjrt_event_create,
                api.pjrt_event_set,
                api.pjrt_device_get_attributes,
                api.pjrt_client_load,
                api.pjrt_loaded_executable_addressable_device_logical_ids,
                api.pjrt_buffer_bitcast,
                api.pjrt_error_for_each_payload,
                api.pjrt_topology_description_fingerprint,
                api.pjrt_executable_parameter_memory_kinds,
                api.pjrt_device_clear_memory_stats,
                api.pjrt_topology_description_make_canonical_shape_for_memory_space,
                api.pjrt_topology_description_get_memory_space_kind_ids,
            ] {
                let mut dummy_args = [0_usize; 28];
                dummy_args[0] = 224;
                assert!(slot_fn(dummy_args.as_mut_ptr().cast()).is_null());
                if slot_fn as usize == api.pjrt_event_create as usize && dummy_args[2] != 0 {
                    drop(Box::from_raw(
                        dummy_args[2] as *mut crate::pjrt_types::PJRT_Event,
                    ));
                }
                if slot_fn as usize == api.pjrt_execute_context_create as usize
                    && dummy_args[2] != 0
                {
                    drop(Box::from_raw(
                        dummy_args[2] as *mut crate::pjrt_types::PJRT_ExecuteContext,
                    ));
                }
                if slot_fn as usize == api.pjrt_device_create_async_tracking_event as usize
                    && dummy_args[5] != 0
                {
                    drop(Box::from_raw(
                        dummy_args[5] as *mut crate::pjrt_types::PJRT_AsyncTrackingEvent,
                    ));
                }
                if slot_fn as usize == api.pjrt_topology_description_create as usize
                    || slot_fn as usize == api.pjrt_topology_description_deserialize as usize
                {
                    let topo_val =
                        if slot_fn as usize == api.pjrt_topology_description_create as usize {
                            dummy_args[6]
                        } else {
                            dummy_args[5]
                        };
                    if topo_val != 0 {
                        let mut del_tp = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                            struct_size: size_of::<
                                crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args,
                            >(),
                            extension_start: ptr::null_mut(),
                            topology: topo_val as *mut PJRT_TopologyDescription,
                        };
                        let _ = PJRT_TopologyDescription_Destroy(ptr::addr_of_mut!(del_tp).cast());
                    }
                }
                if (slot_fn as usize == api.pjrt_executable_serialize as usize
                    || slot_fn as usize == api.pjrt_topology_description_serialize as usize
                    || slot_fn as usize
                        == api.pjrt_loaded_executable_get_device_assignment as usize
                    || slot_fn as usize == api.pjrt_executable_get_compile_options as usize)
                    && dummy_args[5] != 0
                {
                    pjrt_serialized_artifact_deleter(dummy_args[5] as *mut std::ffi::c_void);
                }
            }
            let mut small_args = [8_usize; 16];
            for strict_fn in [
                PJRT_Client_BufferFromHostBuffer
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_LoadedExecutable_Execute,
                PJRT_Buffer_ToHostBuffer,
                PJRT_Buffer_Dimensions,
                PJRT_Buffer_ElementType,
                PJRT_Buffer_OnDeviceSizeInBytes,
                PJRT_Buffer_Destroy,
                PJRT_LoadedExecutable_Destroy,
            ] {
                drop(Box::from_raw(strict_fn(small_args.as_mut_ptr().cast())));
            }
            let ext_null_err = crate::extensions::PJRT_Extension_ValidateArgs(ptr::null_mut());
            drop(Box::from_raw(ext_null_err));
            drop(Box::from_raw(PJRT_Plugin_Attributes(ptr::null_mut())));
            drop(Box::from_raw(PJRT_DeviceDescription_Id(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Device_GetDescription(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Memory_Id(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Executable_Name(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_IsDeleted(ptr::null_mut())));
            drop(Box::from_raw(
                crate::extensions::PLUGIN_Profiler_Error_Destroy(ptr::null_mut()),
            ));
            let mut zero_sz: usize = 0;
            let ext_sz_err =
                crate::extensions::PJRT_Extension_ValidateArgs(ptr::addr_of_mut!(zero_sz).cast());
            drop(Box::from_raw(ext_sz_err));
            let mut valid_sz: [usize; 12] = [96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            assert!(
                crate::extensions::PJRT_Extension_ValidateArgs(valid_sz.as_mut_ptr().cast())
                    .is_null()
            );

            // Verify all 138 extension and profiler functions are distinct and execute both valid and error paths
            let ext_fns: [crate::pjrt_types::PjrtFn; 138] = [
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.error_destroy,
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.error_message,
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.error_get_code,
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.create,
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.destroy,
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.start,
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.stop,
                crate::extensions::GLOBAL_PLUGIN_PROFILER_API.collect_data,
                crate::extensions::EXT_01_XLA_TRANSFORM.register_xla_transform,
                crate::extensions::EXT_01_XLA_TRANSFORM.wait_for_all_transforms,
                crate::extensions::EXT_01_XLA_TRANSFORM.unregister_xla_transform,
                crate::extensions::EXT_01_XLA_TRANSFORM.list_xla_transforms,
                crate::extensions::EXT_02_HOST_ALLOCATOR.get_host_memory_allocator,
                crate::extensions::EXT_03_MULTI_SLICE.config_destroy,
                crate::extensions::EXT_03_MULTI_SLICE.config_num_slices,
                crate::extensions::EXT_03_MULTI_SLICE.config_num_devices_per_slice,
                crate::extensions::EXT_03_MULTI_SLICE.config_slice_id,
                crate::extensions::EXT_03_MULTI_SLICE.config_serialize,
                crate::extensions::EXT_04_COLLECTIVES.collectives_destroy,
                crate::extensions::EXT_04_COLLECTIVES.create_communicator,
                crate::extensions::EXT_04_COLLECTIVES.communicator_destroy,
                crate::extensions::EXT_04_COLLECTIVES.communicator_all_reduce,
                crate::extensions::EXT_04_COLLECTIVES.communicator_reduce_scatter,
                crate::extensions::EXT_04_COLLECTIVES.communicator_all_gather,
                crate::extensions::EXT_04_COLLECTIVES.communicator_collective_permute,
                crate::extensions::EXT_04_COLLECTIVES.communicator_all_to_all,
                crate::extensions::EXT_04_COLLECTIVES.communicator_to_string,
                crate::extensions::EXT_05_ABI_VERSION.client_runtime_abi_version,
                crate::extensions::EXT_05_ABI_VERSION.executable_get_abi_version,
                crate::extensions::EXT_05_ABI_VERSION.runtime_abi_version_destroy,
                crate::extensions::EXT_05_ABI_VERSION
                    .runtime_abi_version_is_compatible_with_runtime,
                crate::extensions::EXT_05_ABI_VERSION
                    .runtime_abi_version_is_compatible_with_executable,
                crate::extensions::EXT_05_ABI_VERSION.runtime_abi_version_to_proto,
                crate::extensions::EXT_05_ABI_VERSION.runtime_abi_version_platform_id,
                crate::extensions::EXT_05_ABI_VERSION.executable_abi_version_destroy,
                crate::extensions::EXT_05_ABI_VERSION.executable_abi_version_to_proto,
                crate::extensions::EXT_05_ABI_VERSION.executable_abi_version_platform_id,
                crate::extensions::EXT_05_ABI_VERSION.runtime_abi_version_from_proto,
                crate::extensions::EXT_05_ABI_VERSION.executable_abi_version_from_proto,
                crate::extensions::EXT_06_SHARDINGS.parameter_shardings,
                crate::extensions::EXT_06_SHARDINGS.output_shardings,
                crate::extensions::EXT_07_MEGASCALE.create_client_context,
                crate::extensions::EXT_07_MEGASCALE.delete_client_context,
                crate::extensions::EXT_07_MEGASCALE.create_default_config,
                crate::extensions::EXT_07_MEGASCALE.create_aot_config,
                crate::extensions::EXT_07_MEGASCALE.create_multi_slice_config,
                crate::extensions::EXT_07_MEGASCALE.delete_multi_slice_config,
                crate::extensions::EXT_07_MEGASCALE.multi_slice_config_num_slices,
                crate::extensions::EXT_07_MEGASCALE.multi_slice_config_num_devices_per_slice,
                crate::extensions::EXT_07_MEGASCALE.multi_slice_config_slice_id,
                crate::extensions::EXT_07_MEGASCALE.multi_slice_config_serialized_slice_key_value,
                crate::extensions::EXT_07_MEGASCALE.multi_slice_config_serialize,
                crate::extensions::EXT_07_MEGASCALE.deserialize_multi_slice_config,
                crate::extensions::EXT_07_MEGASCALE.megascale_addresses,
                crate::extensions::EXT_07_MEGASCALE.barrier,
                crate::extensions::EXT_07_MEGASCALE.log_error,
                crate::extensions::EXT_07_MEGASCALE.get_megascale_dcn_topology,
                crate::extensions::EXT_07_MEGASCALE.get_slice_id,
                crate::extensions::EXT_07_MEGASCALE.get_num_slices,
                crate::extensions::EXT_07_MEGASCALE.initialize_slice,
                crate::extensions::EXT_07_MEGASCALE.finalize_slice,
                crate::extensions::EXT_07_MEGASCALE.register_error_handler,
                crate::extensions::EXT_07_MEGASCALE.unregister_error_handler,
                crate::extensions::EXT_07_MEGASCALE.collect_metrics,
                crate::extensions::EXT_07_MEGASCALE.reset_metrics,
                crate::extensions::EXT_07_MEGASCALE.get_transport_type,
                crate::extensions::EXT_07_MEGASCALE.set_transport_type,
                crate::extensions::EXT_07_MEGASCALE.wait_for_barrier,
                crate::extensions::EXT_07_MEGASCALE.abort_collectives,
                crate::extensions::EXT_08_TPU_EXECUTABLE.get_target_arguments,
                crate::extensions::EXT_08_TPU_EXECUTABLE.get_hlo_module_with_config,
                crate::extensions::EXT_08_TPU_EXECUTABLE.reserved,
                crate::extensions::EXT_08_TPU_EXECUTABLE.get_compiled_memory_stats,
                crate::extensions::EXT_08_TPU_EXECUTABLE.run_hlo_cost_analysis,
                crate::extensions::EXT_08_TPU_EXECUTABLE.set_tpu_compilation_env,
                crate::extensions::EXT_08_TPU_EXECUTABLE.get_tpu_compilation_env_field_as_string,
                crate::extensions::EXT_08_TPU_EXECUTABLE.is_tpu_predetermined_error,
                crate::extensions::EXT_08_TPU_EXECUTABLE.is_tpu_cancellation_placeholder_error,
                crate::extensions::EXT_09_TPU_TOPOLOGY.subslice,
                crate::extensions::EXT_09_TPU_TOPOLOGY.is_subslice_topology,
                crate::extensions::EXT_09_TPU_TOPOLOGY.subslice_device_id_from_full_device_id,
                crate::extensions::EXT_09_TPU_TOPOLOGY.replace_host_bounds,
                crate::extensions::EXT_09_TPU_TOPOLOGY.is_enhanced_barrier_enabled,
                crate::extensions::EXT_09_TPU_TOPOLOGY.has_limited_ici_connectivity,
                crate::extensions::EXT_09_TPU_TOPOLOGY.is_reachable_over_limited_ici,
                crate::extensions::EXT_09_TPU_TOPOLOGY.process_count,
                crate::extensions::EXT_09_TPU_TOPOLOGY.chips_per_process,
                crate::extensions::EXT_09_TPU_TOPOLOGY.core_count_per_chip,
                crate::extensions::EXT_09_TPU_TOPOLOGY.chip_count,
                crate::extensions::EXT_09_TPU_TOPOLOGY.core_count,
                crate::extensions::EXT_09_TPU_TOPOLOGY.logi_device_count_per_process,
                crate::extensions::EXT_09_TPU_TOPOLOGY.logi_device_count,
                crate::extensions::EXT_09_TPU_TOPOLOGY.logi_device_count_per_chip,
                crate::extensions::EXT_09_TPU_TOPOLOGY.core_count_per_process,
                crate::extensions::EXT_09_TPU_TOPOLOGY.process_ids,
                crate::extensions::EXT_09_TPU_TOPOLOGY.logi_device_ids_on_process,
                crate::extensions::EXT_09_TPU_TOPOLOGY.proc_id_and_idx_on_proc_for_chip,
                crate::extensions::EXT_09_TPU_TOPOLOGY.proc_id_and_idx_on_proc_for_logi_device,
                crate::extensions::EXT_09_TPU_TOPOLOGY.process_coord_from_id,
                crate::extensions::EXT_09_TPU_TOPOLOGY.chip_id_from_coord,
                crate::extensions::EXT_09_TPU_TOPOLOGY.logi_device_id_from_chip_coord_and_idx,
                crate::extensions::EXT_09_TPU_TOPOLOGY.chip_coord_and_idx_for_logi_device,
                crate::extensions::EXT_09_TPU_TOPOLOGY.chips_per_process_bounds,
                crate::extensions::EXT_09_TPU_TOPOLOGY.chip_bounds,
                crate::extensions::EXT_09_TPU_TOPOLOGY.process_bounds,
                crate::extensions::EXT_09_TPU_TOPOLOGY.get_routing_strategy,
                crate::extensions::EXT_09_TPU_TOPOLOGY.get_slice_config,
                crate::extensions::EXT_09_TPU_TOPOLOGY.get_slice_configs,
                crate::extensions::EXT_09_TPU_TOPOLOGY.get_default_platform_config,
                crate::extensions::EXT_10_CALLBACK.register_callback,
                crate::extensions::EXT_10_CALLBACK.invoke_callback,
                crate::extensions::EXT_11_PHASE_COMPILE.phase_compile_get_compiler,
                crate::extensions::EXT_11_PHASE_COMPILE.phase_compile_destroy_compiler,
                crate::extensions::EXT_11_PHASE_COMPILE.phase_compile_run_phase,
                crate::extensions::EXT_11_PHASE_COMPILE.phase_compile_get_phase_names,
                crate::extensions::EXT_11_PHASE_COMPILE.phase_compile_c_buffers_destroy,
                crate::extensions::EXT_12_CROSS_HOST.make_cross_host_receive_buffers,
                crate::extensions::EXT_12_CROSS_HOST.copy_to_remote_device,
                crate::extensions::EXT_12_CROSS_HOST.cross_host_receive_buffers,
                crate::extensions::EXT_12_CROSS_HOST.cross_host_send_buffers,
                crate::extensions::EXT_13_EXEC_METADATA.get_executable_metadata,
                crate::extensions::EXT_13_EXEC_METADATA.destroy_executable_metadata,
                crate::extensions::EXT_14_MEM_DESC.get_memory_descriptions,
                crate::extensions::EXT_14_MEM_DESC.destroy_memory_descriptions,
                crate::extensions::EXT_15_LAYOUTS.memory_layout_destroy,
                crate::extensions::EXT_15_LAYOUTS.memory_layout_serialize,
                crate::extensions::EXT_15_LAYOUTS.client_get_default_layout,
                crate::extensions::EXT_15_LAYOUTS.buffer_memory_layout,
                crate::extensions::EXT_15_LAYOUTS.topology_get_default_layout,
                crate::extensions::EXT_15_LAYOUTS.executable_get_output_layouts,
                crate::extensions::EXT_15_LAYOUTS.executable_get_parameter_layouts,
                crate::extensions::EXT_16_RAW_BUFFER.create_raw_alias_of_buffer,
                crate::extensions::EXT_16_RAW_BUFFER.destroy,
                crate::extensions::EXT_16_RAW_BUFFER.get_host_pointer,
                crate::extensions::EXT_16_RAW_BUFFER.get_on_device_size_in_bytes,
                crate::extensions::EXT_16_RAW_BUFFER.get_memory_space,
                crate::extensions::EXT_16_RAW_BUFFER.copy_raw_host_to_device,
                crate::extensions::EXT_16_RAW_BUFFER.copy_raw_device_to_host,
            ];
            for ext_fn in ext_fns {
                valid_sz = [96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                assert!(ext_fn(valid_sz.as_mut_ptr().cast()).is_null());
            }
            drop(Box::from_raw(
                crate::extensions::PLUGIN_Profiler_CollectData(ptr::null_mut()),
            ));
            let mut prof_collect_null = crate::extensions::PLUGIN_Profiler_CollectData_Args {
                struct_size: size_of::<crate::extensions::PLUGIN_Profiler_CollectData_Args>(),
                extension_start: ptr::null_mut(),
                profiler: 1,
                buffer: ptr::null_mut(),
                size_in_bytes: ptr::null_mut(),
            };
            assert_ne!(format!("{prof_collect_null:?}"), "");
            assert!(
                crate::extensions::PLUGIN_Profiler_CollectData(
                    ptr::addr_of_mut!(prof_collect_null).cast()
                )
                .is_null()
            );
            let mut xspace_sz: usize = 0;
            let mut xspace_buf = vec![0_u8; 512];
            let mut prof_collect_full = crate::extensions::PLUGIN_Profiler_CollectData_Args {
                struct_size: size_of::<crate::extensions::PLUGIN_Profiler_CollectData_Args>(),
                extension_start: ptr::null_mut(),
                profiler: 1,
                buffer: xspace_buf.as_mut_ptr(),
                size_in_bytes: ptr::addr_of_mut!(xspace_sz),
            };
            assert!(
                crate::extensions::PLUGIN_Profiler_CollectData(
                    ptr::addr_of_mut!(prof_collect_full).cast()
                )
                .is_null()
            );
            assert!(xspace_sz > 0);
            assert_ne!(
                format!("{:?}", crate::extensions::ExtensionRuntimeState::default()),
                ""
            );
            let ext_destroy_args = crate::extensions::PJRT_Megascale_Handle_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                handle: ptr::null_mut(),
            };
            let ext_q32_args = crate::extensions::PJRT_Megascale_Query_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: 0,
                out_value: 0,
            };
            let ext_q40_args = crate::extensions::PJRT_Megascale_Data_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                handle: 0,
                out_first: 0,
                out_second: 0,
            };
            let ext_q48_args = crate::extensions::PJRT_TpuExecutable_ProtoQuery_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                executable: ptr::null(),
                serialized_bytes: 0,
                serialized_bytes_size: 0,
                holder: ptr::null_mut(),
            };
            let ext_q56_args = crate::extensions::PJRT_CrossHostTransfers_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                buffers: 0,
                num_buffers: 0,
                remote_device: 0,
                out_event: 0,
            };
            let ext_q64_args = crate::extensions::PJRT_Communicator_CollectivePermute_Args {
                struct_size: 64,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: ptr::null_mut(),
                recv_buffer: ptr::null_mut(),
                target_ranks: ptr::null(),
                count: 0,
                event: ptr::null_mut(),
            };
            let prof_err_del = crate::extensions::PLUGIN_Profiler_Error_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                error: ptr::null_mut(),
            };
            let prof_err_msg = crate::extensions::PLUGIN_Profiler_Error_Message_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                message: ptr::null(),
                message_size: 0,
            };
            let prof_err_code = crate::extensions::PLUGIN_Profiler_Error_GetCode_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                code: 0,
            };
            let prof_create = crate::extensions::PLUGIN_Profiler_Create_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                options: ptr::null(),
                options_size: 0,
                profiler: ptr::null_mut(),
            };
            let prof_del = crate::extensions::PLUGIN_Profiler_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                profiler: ptr::null_mut(),
            };
            let prof_start = crate::extensions::PLUGIN_Profiler_Start_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                profiler: ptr::null_mut(),
            };
            let prof_stop = crate::extensions::PLUGIN_Profiler_Stop_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                profiler: ptr::null_mut(),
            };
            let ms_del = crate::extensions::PJRT_MultiSlice_Config_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                config: ptr::null_mut(),
            };
            let ms_ns = crate::extensions::PJRT_MultiSlice_Config_NumSlices_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                num_slices: 0,
            };
            let ms_ndps = crate::extensions::PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                num_devices_per_slice_map: 0,
                num_slices: 0,
            };
            let ms_sid = crate::extensions::PJRT_MultiSlice_Config_SliceId_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                slice_id: 0,
            };
            let coll_del = crate::extensions::PJRT_Collectives_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                collectives: ptr::null_mut(),
            };
            let coll_cc = crate::extensions::PJRT_Collectives_CreateCommunicator_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                collectives: ptr::null_mut(),
                device_ids: ptr::null(),
                num_devices: 0,
                communicator: ptr::null_mut(),
            };
            let comm_del = crate::extensions::PJRT_Communicator_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
            };
            let comm_ag = crate::extensions::PJRT_Communicator_AllGather_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: ptr::null_mut(),
                recv_buffer: ptr::null_mut(),
                count: 0,
                event: ptr::null_mut(),
            };
            let comm_str = crate::extensions::PJRT_Communicator_ToString_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                communicator: ptr::null(),
                str_out: 0,
                str_size: 0,
            };
            let abi_get = crate::extensions::PJRT_AbiVersion_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: ptr::null(),
                abi_version_out: 0,
            };
            let abi_del = crate::extensions::PJRT_AbiVersion_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null_mut(),
            };
            let abi_cmp = crate::extensions::PJRT_AbiVersion_IsCompatible_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null(),
                other_abi_version: 0,
                is_compatible_out: 0,
            };
            let abi_pid = crate::extensions::PJRT_AbiVersion_PlatformId_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null(),
                platform_id_out: 0,
            };
            let abi_fp = crate::extensions::PJRT_AbiVersion_FromProto_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                serialized_proto: ptr::null(),
                serialized_proto_size: 0,
                abi_version_out: 0,
            };
            let tpu_ec = crate::extensions::PJRT_TpuExecutable_ErrorCheck_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                is_matched: 0,
            };
            let topo_sq = crate::extensions::PJRT_TpuTopology_ScalarQuery_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                topology: ptr::null(),
                out_value: 0,
            };
            let topo_bq = crate::extensions::PJRT_TpuTopology_BoundsQuery_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                topology: ptr::null(),
                out_first: 0,
                out_second: 0,
            };
            let topo_mq = crate::extensions::PJRT_TpuTopology_MappingQuery_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                topology: ptr::null(),
                param0: 0,
                out_first: 0,
                out_second: 0,
            };
            let cb_inv = crate::extensions::PJRT_Callback_Invoke_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                callback_type: 0,
                payload: ptr::null_mut(),
            };
            let pc_gc = crate::extensions::PJRT_PhaseCompile_GetCompiler_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                compiler_out: 0,
            };
            let pc_del = crate::extensions::PJRT_PhaseCompile_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                handle: ptr::null_mut(),
            };
            let md_get = crate::extensions::PJRT_Metadata_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: ptr::null(),
                metadata_out: 0,
            };
            let md_del = crate::extensions::PJRT_Metadata_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                metadata: ptr::null_mut(),
            };
            let rb_q = crate::extensions::PJRT_RawBuffer_Query_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                out_value: 0,
            };
            let rb_del = crate::extensions::PJRT_RawBuffer_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                raw_buffer: ptr::null_mut(),
            };
            assert_ne!(
                format!(
                    "{ext_destroy_args:?}{ext_q32_args:?}{ext_q40_args:?}{ext_q48_args:?}{ext_q56_args:?}{ext_q64_args:?}{prof_err_del:?}{prof_err_msg:?}{prof_err_code:?}{prof_create:?}{prof_del:?}{prof_start:?}{prof_stop:?}{ms_del:?}{ms_ns:?}{ms_ndps:?}{ms_sid:?}{coll_del:?}{coll_cc:?}{comm_del:?}{comm_ag:?}{comm_str:?}{abi_get:?}{abi_del:?}{abi_cmp:?}{abi_pid:?}{abi_fp:?}{tpu_ec:?}{topo_sq:?}{topo_bq:?}{topo_mq:?}{cb_inv:?}{pc_gc:?}{pc_del:?}{md_get:?}{md_del:?}{rb_q:?}{rb_del:?}"
                ),
                ""
            );
            let layout_destroy_args = crate::extensions::PJRT_Layouts_MemoryLayout_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                layout: ptr::null_mut(),
            };
            let layout_ser_args = crate::extensions::PJRT_Layouts_MemoryLayout_Serialize_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                layout: ptr::null(),
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_layout: ptr::null_mut(),
                serialized_layout_deleter: ptr::null(),
            };
            let layout_def_args =
                crate::extensions::PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
                    struct_size: 56,
                    extension_start: ptr::null_mut(),
                    client: ptr::null_mut(),
                    elem_type: 11,
                    pad: 0,
                    dims: ptr::null(),
                    num_dims: 0,
                    layout: ptr::null_mut(),
                };
            let layout_buf_args = crate::extensions::PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                layout: ptr::null_mut(),
            };
            let shardings_args = crate::extensions::PJRT_Shardings_PJRT_Executable_Shardings_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                executable: ptr::null_mut(),
                num_items: 0,
                shardings: ptr::null(),
                sharding_sizes: ptr::null(),
            };
            let phase_names_args = crate::extensions::PJRT_PhaseCompile_GetPhaseNames_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                phase_compiler: ptr::null(),
                phase_names: ptr::null(),
                phase_names_sizes: ptr::null(),
                num_phase_names: 0,
            };
            let triton_args = crate::extensions::PJRT_Triton_Compile_Args {
                struct_size: 88,
                extension_start: ptr::null_mut(),
                module: ptr::null(),
                module_size: 0,
                arch_name: ptr::null(),
                arch_name_size: 0,
                num_warps: 4,
                num_ctas: 1,
                num_stages: 2,
                pad: 0,
                out_asm: ptr::null(),
                out_asm_size: 0,
                out_smem_bytes: 0,
            };
            assert_ne!(
                format!(
                    "{layout_destroy_args:?}{layout_ser_args:?}{layout_def_args:?}{layout_buf_args:?}{shardings_args:?}{phase_names_args:?}{triton_args:?}"
                ),
                ""
            );
            let xla_reg_args = crate::extensions::PJRT_XlaTransform_Register_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                name: ptr::null(),
                name_size: 0,
                transform_fn: 0,
            };
            let xla_wait_args = crate::extensions::PJRT_XlaTransform_WaitForAll_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            let xla_unreg_args = crate::extensions::PJRT_XlaTransform_Unregister_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                name: ptr::null(),
                name_size: 0,
            };
            let xla_list_args = crate::extensions::PJRT_XlaTransform_List_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                names_out: ptr::null(),
                num_names_out: 0,
            };
            let mut host_alloc_args = crate::extensions::PJRT_HostMemoryAllocator_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                allocator_out: 0,
            };
            assert!(
                crate::extensions::PJRT_HostMemoryAllocator_Get(
                    ptr::addr_of_mut!(host_alloc_args).cast()
                )
                .is_null()
            );
            assert!(host_alloc_args.allocator_out > 0);
            let mut md_get32 = crate::extensions::PJRT_Metadata_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: ptr::null(),
                metadata_out: 0,
            };
            assert!(
                crate::extensions::PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(md_get32).cast())
                    .is_null()
            );
            assert_eq!(md_get32.metadata_out, 4);
            let mut md_del24 = crate::extensions::PJRT_Metadata_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                metadata: ptr::dangling_mut(),
            };
            assert!(
                crate::extensions::PJRT_MemoryDescriptions_Destroy(
                    ptr::addr_of_mut!(md_del24).cast()
                )
                .is_null()
            );
            assert!(md_del24.metadata.is_null());
            let ms_ser_args = crate::extensions::PJRT_MultiSlice_Config_Serialize_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_config: ptr::null_mut(),
            };
            let comm_ar_args = crate::extensions::PJRT_Communicator_AllReduce_Args {
                struct_size: 64,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: ptr::null_mut(),
                recv_buffer: ptr::null_mut(),
                reduction_op: 0,
                pad: 0,
                count: 4,
                event: ptr::null_mut(),
            };
            let abi_proto_args = crate::extensions::PJRT_AbiVersion_ToProto_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null(),
                serialized_proto: ptr::null(),
                serialized_proto_size: 0,
                serialized_proto_holder: ptr::null_mut(),
            };
            let tpu_mem_stats_args =
                crate::extensions::PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
                    struct_size: 64,
                    extension_start: ptr::null_mut(),
                    executable: ptr::null_mut(),
                    generated_code_size_in_bytes: 0,
                    argument_size_in_bytes: 0,
                    output_size_in_bytes: 0,
                    alias_size_in_bytes: 0,
                    temp_size_in_bytes: 0,
                };
            let cb_reg_args = crate::extensions::PJRT_Callback_Register_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                callback_type: 1,
                callback_fn: ptr::null(),
                user_arg: ptr::null_mut(),
            };
            let phase_run_args = crate::extensions::PJRT_PhaseCompile_RunPhase_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                phase_compiler: ptr::null_mut(),
                input_bytes: ptr::null(),
                input_size: 0,
                phase_name: ptr::null(),
                output_status: 0,
            };
            let raw_copy_args = crate::extensions::PJRT_RawBuffer_CopyRaw_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                host_ptr: ptr::null_mut(),
                offset: 0,
                transfer_size: 0,
                event: ptr::null_mut(),
            };
            assert_ne!(
                format!(
                    "{xla_reg_args:?}{xla_wait_args:?}{xla_unreg_args:?}{xla_list_args:?}{host_alloc_args:?}{ms_ser_args:?}{comm_ar_args:?}{abi_proto_args:?}{tpu_mem_stats_args:?}{cb_reg_args:?}{phase_run_args:?}{raw_copy_args:?}"
                ),
                ""
            );
            let mut host_alloc64 = crate::extensions::PJRT_HostMemoryAllocator_Allocate_Args {
                struct_size: size_of::<crate::extensions::PJRT_HostMemoryAllocator_Allocate_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                size: 4096,
                numa_node: 0,
                pad: 0,
                ptr: ptr::null_mut(),
                deleter: ptr::null(),
                deleter_arg: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_HostMemoryAllocator_Get(
                    ptr::addr_of_mut!(host_alloc64).cast()
                )
                .is_null()
            );
            assert!(!host_alloc64.ptr.is_null());
            let mut mem_desc48 =
                crate::extensions::PJRT_DeviceDescription_MemoryDescriptions_Args {
                    struct_size: size_of::<
                        crate::extensions::PJRT_DeviceDescription_MemoryDescriptions_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    device_description: ptr::null(),
                    memory_descriptions: ptr::null(),
                    num_memory_descriptions: 0,
                    default_memory_index: -1,
                };
            assert!(
                crate::extensions::PJRT_MemoryDescriptions_Get(
                    ptr::addr_of_mut!(mem_desc48).cast()
                )
                .is_null()
            );
            assert_eq!(mem_desc48.num_memory_descriptions, 4);
            assert_eq!(mem_desc48.default_memory_index, 0);
            let mut mem_kind48 = crate::extensions::PJRT_MemoryDescription_Kind_Args {
                struct_size: size_of::<crate::extensions::PJRT_MemoryDescription_Kind_Args>(),
                extension_start: ptr::null_mut(),
                memory_description: 2_usize as *const std::ffi::c_void,
                kind: ptr::null(),
                kind_size: 0,
                kind_id: -1,
                pad: 0,
            };
            assert!(
                crate::extensions::PJRT_MemoryDescriptions_Destroy(
                    ptr::addr_of_mut!(mem_kind48).cast()
                )
                .is_null()
            );
            assert_eq!(mem_kind48.kind_id, 1);
            assert_eq!(mem_kind48.kind_size, 4);
            let cb_sb = crate::extensions::PJRT_Callback_Tpu_SliceBuilderArgs {
                struct_size: 16,
                failure_type: crate::extensions::PJRT_Callback_Tpu_SliceFailureType::default()
                    as i32,
                pad: 0,
            };
            let cb_pf = crate::extensions::PJRT_Callback_PrefatalArgs {
                struct_size: 32,
                error_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
            };
            let exec_md = crate::extensions::PJRT_ExecutableMetadata {
                serialized_metadata: ptr::null(),
                serialized_metadata_size: 0,
            };
            assert_ne!(
                format!(
                    "{host_alloc64:?}{mem_desc48:?}{mem_kind48:?}{cb_sb:?}{cb_pf:?}{exec_md:?}{:?}{:?}{:?}",
                    crate::extensions::PJRT_Collectives_ReductionKind::default(),
                    crate::extensions::PJRT_Collectives_CollectiveOpKind::default(),
                    crate::extensions::PJRT_Callback_Type::default(),
                ),
                ""
            );
            assert!(crate::extensions::extension_call_count() > 0);
            assert!(pjrt_slot_dispatch_count() > 0);

            PJRT_Error_Destroy(ptr::null_mut());
            let mut destroy_null_err = PJRT_Error_Destroy_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                error: ptr::null_mut(),
            };
            assert_ne!(format!("{destroy_null_err:?}"), "");
            PJRT_Error_Destroy(ptr::addr_of_mut!(destroy_null_err).cast());

            PJRT_Error_Message(ptr::null_mut());
            let mut msg_null_err = PJRT_Error_Message_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                message: ptr::null(),
                message_size: 0,
            };
            assert_ne!(format!("{msg_null_err:?}"), "");
            PJRT_Error_Message(ptr::addr_of_mut!(msg_null_err).cast());

            let null_code_err = PJRT_Error_GetCode(ptr::null_mut());
            assert!(!null_code_err.is_null());
            let mut msg_args = PJRT_Error_Message_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: null_code_err,
                message: ptr::null(),
                message_size: 0,
            };
            PJRT_Error_Message(ptr::addr_of_mut!(msg_args).cast());
            assert!(msg_args.message_size > 0);
            assert_ne!(format!("{:?}", *null_code_err), "");

            let mut code_args = PJRT_Error_GetCode_Args {
                struct_size: size_of::<PJRT_Error_GetCode_Args>(),
                extension_start: ptr::null_mut(),
                error: null_code_err,
                code: 0,
            };
            assert_ne!(format!("{code_args:?}"), "");
            assert!(PJRT_Error_GetCode(ptr::addr_of_mut!(code_args).cast()).is_null());
            assert_eq!(code_args.code, 3);

            let mut code_null_inner = PJRT_Error_GetCode_Args {
                struct_size: size_of::<PJRT_Error_GetCode_Args>(),
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                code: 0,
            };
            let err2 = PJRT_Error_GetCode(ptr::addr_of_mut!(code_null_inner).cast());
            let mut d2 = PJRT_Error_Destroy_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                error: err2,
            };
            PJRT_Error_Destroy(ptr::addr_of_mut!(d2).cast());

            let mut d1 = PJRT_Error_Destroy_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                error: null_code_err,
            };
            PJRT_Error_Destroy(ptr::addr_of_mut!(d1).cast());

            let e_init = PJRT_Plugin_Initialize(ptr::null_mut());
            drop(Box::from_raw(e_init));
            let mut init_args = PJRT_Plugin_Initialize_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
            };
            assert_ne!(format!("{init_args:?}"), "");
            assert!(PJRT_Plugin_Initialize(ptr::addr_of_mut!(init_args).cast()).is_null());

            let e_chk = PJRT_CheckMatchingStructSizes(ptr::null_mut());
            drop(Box::from_raw(e_chk));
            let e_chk_sz = PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(zero_sz).cast());
            drop(Box::from_raw(e_chk_sz));
            assert!(PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(init_args).cast()).is_null());

            let e_cc = PJRT_Client_Create(ptr::null_mut());
            drop(Box::from_raw(e_cc));
            let create_opts = [
                crate::pjrt_types::PJRT_NamedValue {
                    struct_size: size_of::<crate::pjrt_types::PJRT_NamedValue>(),
                    extension_start: ptr::null_mut(),
                    name: c"node_id".as_ptr(),
                    name_size: 7,
                    value_type_and_pad: 1,
                    value_payload: 0,
                    value_size: 1,
                },
                crate::pjrt_types::PJRT_NamedValue {
                    struct_size: size_of::<crate::pjrt_types::PJRT_NamedValue>(),
                    extension_start: ptr::null_mut(),
                    name: c"process_index".as_ptr(),
                    name_size: 13,
                    value_type_and_pad: 0,
                    value_payload: 0,
                    value_size: 1,
                },
                crate::pjrt_types::PJRT_NamedValue {
                    struct_size: size_of::<crate::pjrt_types::PJRT_NamedValue>(),
                    extension_start: ptr::null_mut(),
                    name: c"other".as_ptr(),
                    name_size: 5,
                    value_type_and_pad: 1,
                    value_payload: 0,
                    value_size: 1,
                },
                crate::pjrt_types::PJRT_NamedValue {
                    struct_size: size_of::<crate::pjrt_types::PJRT_NamedValue>(),
                    extension_start: ptr::null_mut(),
                    name: ptr::null(),
                    name_size: 0,
                    value_type_and_pad: 0,
                    value_payload: 0,
                    value_size: 0,
                },
            ];
            /// Test callback for `kv_put_callback` returning null (`OK`).
            ///
            /// # Arguments
            ///
            /// * `_key` - Key string pointer.
            /// * `_key_len` - Key byte length.
            /// * `_val` - Value string pointer.
            /// * `_val_len` - Value byte length.
            /// * `user_arg` - User context pointer.
            unsafe extern "C" fn mock_kv_put_ok(
                _key: *const std::ffi::c_char,
                _key_len: usize,
                _val: *const std::ffi::c_char,
                _val_len: usize,
                user_arg: *mut std::ffi::c_void,
            ) -> *mut PJRT_Error {
                if !user_arg.is_null() {
                    // SAFETY: `user_arg` points to `kv_called` in test stack frame.
                    unsafe {
                        *user_arg.cast::<usize>() += 1;
                    }
                }
                ptr::null_mut()
            }

            /// Test callback for `kv_put_callback` returning a heap-allocated `PJRT_Error`.
            ///
            /// # Arguments
            ///
            /// * `_key` - Key string pointer.
            /// * `_key_len` - Key byte length.
            /// * `_val` - Value string pointer.
            /// * `_val_len` - Value byte length.
            /// * `_user_arg` - User context pointer.
            unsafe extern "C" fn mock_kv_put_err(
                _key: *const std::ffi::c_char,
                _key_len: usize,
                _val: *const std::ffi::c_char,
                _val_len: usize,
                _user_arg: *mut std::ffi::c_void,
            ) -> *mut PJRT_Error {
                box_pjrt_error(LibTpuError::InvalidArgument {
                    parameter: "kv_put",
                    reason: String::from("simulated kv_put failure"),
                })
            }

            let mut kv_called: usize = 0;
            let mut create_args_opts = PJRT_Client_Create_Args {
                struct_size: size_of::<PJRT_Client_Create_Args>(),
                extension_start: ptr::null_mut(),
                create_options: create_opts.as_ptr().cast(),
                num_options: create_opts.len(),
                kv_get_callback: ptr::null(),
                kv_get_user_arg: ptr::null_mut(),
                kv_put_callback: mock_kv_put_ok as *const std::ffi::c_void,
                kv_put_user_arg: ptr::addr_of_mut!(kv_called).cast(),
                kv_try_get_callback: ptr::null(),
                kv_try_get_user_arg: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            assert!(PJRT_Client_Create(ptr::addr_of_mut!(create_args_opts).cast()).is_null());
            assert_eq!(kv_called, 1);
            let mut destroy_opts_client = PJRT_Client_Destroy_Args {
                struct_size: size_of::<PJRT_Client_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                client: create_args_opts.client,
            };
            assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(destroy_opts_client).cast()).is_null());

            create_args_opts.kv_put_callback = mock_kv_put_err as *const std::ffi::c_void;
            create_args_opts.kv_put_user_arg = ptr::null_mut();
            create_args_opts.client = ptr::null_mut();
            assert!(PJRT_Client_Create(ptr::addr_of_mut!(create_args_opts).cast()).is_null());
            destroy_opts_client.client = create_args_opts.client;
            assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(destroy_opts_client).cast()).is_null());

            let mut create_args = PJRT_Client_Create_Args {
                struct_size: size_of::<PJRT_Client_Create_Args>(),
                extension_start: ptr::null_mut(),
                create_options: ptr::null(),
                num_options: 0,
                kv_get_callback: ptr::null(),
                kv_get_user_arg: ptr::null_mut(),
                kv_put_callback: ptr::null(),
                kv_put_user_arg: ptr::null_mut(),
                kv_try_get_callback: ptr::null(),
                kv_try_get_user_arg: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            assert_ne!(format!("{create_args:?}"), "");
            assert!(PJRT_Client_Create(ptr::addr_of_mut!(create_args).cast()).is_null());
            let client = create_args.client;
            assert!(!client.is_null());
            assert_ne!(format!("{:?}", *client), "");
            assert_ne!(format!("{:?}", *(*client).topology_ptr), "");
            for &dev0 in (*client).device_ptrs.first() {
                assert_ne!(format!("{:?}{:?}", *dev0, *(*dev0).desc_ptr), "");
            }
            for &mem0 in (*client).memory_ptrs.first() {
                assert_ne!(format!("{:?}", *mem0), "");
            }

            for mgr in PjrtCoreAsyncTransferManager::create(
                DeviceId(0),
                libtpu_core::types::MemorySpaceId(0),
                Vec::new(),
                &(*client).inner.hal,
            ) {
                let c_mgr = PJRT_AsyncHostToDeviceTransferManager {
                    inner: mgr,
                    device_ptr: ptr::null_mut(),
                    memory_ptr: ptr::null_mut(),
                };
                assert_ne!(format!("{c_mgr:?}"), "");
            }

            drop(Box::from_raw(PJRT_Client_PlatformName(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Client_PlatformVersion(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Client_ProcessIndex(ptr::null_mut())));
            let mut str_args = PJRT_Client_PlatformStr_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                platform_str: ptr::null(),
                platform_str_size: 0,
            };
            assert_ne!(format!("{str_args:?}"), "");
            drop(Box::from_raw(PJRT_Client_PlatformName(
                ptr::addr_of_mut!(str_args).cast(),
            )));
            drop(Box::from_raw(PJRT_Client_PlatformVersion(
                ptr::addr_of_mut!(str_args).cast(),
            )));
            str_args.client = client;
            assert!(PJRT_Client_PlatformName(ptr::addr_of_mut!(str_args).cast()).is_null());
            assert_eq!(str_args.platform_str_size, 3);
            assert!(PJRT_Client_PlatformVersion(ptr::addr_of_mut!(str_args).cast()).is_null());
            assert!(str_args.platform_str_size > 0);

            let mut pi_args = PJRT_Client_ProcessIndex_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                process_index: -1,
            };
            assert_ne!(format!("{pi_args:?}"), "");
            drop(Box::from_raw(PJRT_Client_ProcessIndex(
                ptr::addr_of_mut!(pi_args).cast(),
            )));
            pi_args.client = client;
            assert!(PJRT_Client_ProcessIndex(ptr::addr_of_mut!(pi_args).cast()).is_null());
            assert_eq!(pi_args.process_index, 0);

            drop(Box::from_raw(PJRT_Client_Devices(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Client_AddressableDevices(
                ptr::null_mut(),
            )));
            let mut dev_args = PJRT_Client_Devices_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                devices: ptr::null(),
                num_devices: 0,
            };
            assert_ne!(format!("{dev_args:?}"), "");
            drop(Box::from_raw(PJRT_Client_Devices(
                ptr::addr_of_mut!(dev_args).cast(),
            )));
            drop(Box::from_raw(PJRT_Client_AddressableDevices(
                ptr::addr_of_mut!(dev_args).cast(),
            )));
            dev_args.client = client;
            assert!(PJRT_Client_Devices(ptr::addr_of_mut!(dev_args).cast()).is_null());
            assert_eq!(dev_args.num_devices, 4);
            assert!(PJRT_Client_AddressableDevices(ptr::addr_of_mut!(dev_args).cast()).is_null());
            assert_eq!(dev_args.num_devices, 4);

            drop(Box::from_raw(PJRT_Client_LookupDevice(ptr::null_mut())));
            let mut look_args = PJRT_Client_LookupDevice_Args {
                struct_size: size_of::<PJRT_Client_LookupDevice_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                id: 0,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert_ne!(format!("{look_args:?}"), "");
            drop(Box::from_raw(PJRT_Client_LookupDevice(
                ptr::addr_of_mut!(look_args).cast(),
            )));
            look_args.client = client;
            assert!(PJRT_Client_LookupDevice(ptr::addr_of_mut!(look_args).cast()).is_null());
            assert!(!look_args.device.is_null());
            look_args.id = 999;
            drop(Box::from_raw(PJRT_Client_LookupDevice(
                ptr::addr_of_mut!(look_args).cast(),
            )));

            drop(Box::from_raw(PJRT_Client_AddressableMemories(
                ptr::null_mut(),
            )));
            let mut mem_args = PJRT_Client_AddressableMemories_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                addressable_memories: ptr::null(),
                num_addressable_memories: 0,
            };
            assert_ne!(format!("{mem_args:?}"), "");
            drop(Box::from_raw(PJRT_Client_AddressableMemories(
                ptr::addr_of_mut!(mem_args).cast(),
            )));
            mem_args.client = client;
            assert!(PJRT_Client_AddressableMemories(ptr::addr_of_mut!(mem_args).cast()).is_null());
            assert_eq!(mem_args.num_addressable_memories, 12);

            drop(Box::from_raw(PJRT_Client_Compile(ptr::null_mut())));
            let code_bytes = b"module @test { stablehlo.add }";
            let fmt_bytes = b"mlir";
            let mut prog = PJRT_Program {
                struct_size: size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: ptr::null(),
                code_size: code_bytes.len(),
                format: fmt_bytes.as_ptr().cast(),
                format_size: fmt_bytes.len(),
            };
            assert_ne!(format!("{prog:?}"), "");
            let mut comp_args = PJRT_Client_Compile_Args {
                struct_size: size_of::<PJRT_Client_Compile_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                program: ptr::addr_of!(prog),
                compile_options: ptr::null(),
                compile_options_size: 0,
                executable: ptr::null_mut(),
            };
            assert_ne!(format!("{comp_args:?}"), "");
            drop(Box::from_raw(PJRT_Client_Compile(
                ptr::addr_of_mut!(comp_args).cast(),
            )));
            comp_args.client = client;
            drop(Box::from_raw(PJRT_Client_Compile(
                ptr::addr_of_mut!(comp_args).cast(),
            )));
            prog.code = code_bytes.as_ptr().cast();
            comp_args.program = ptr::addr_of!(prog);
            assert!(PJRT_Client_Compile(ptr::addr_of_mut!(comp_args).cast()).is_null());
            assert_ne!(
                format!(
                    "{:?}{:?}",
                    *comp_args.executable,
                    *(*comp_args.executable).executable_ptr
                ),
                ""
            );

            // Test full PJRT vtable execution flow: BufferFromHostBuffer -> Execute -> ToHostBuffer -> Dimensions/ElementType/OnDeviceSize -> Destroy
            drop(Box::from_raw(PJRT_Client_BufferFromHostBuffer(
                ptr::null_mut(),
            )));
            drop(Box::from_raw(
                PJRT_LoadedExecutable_Execute(ptr::null_mut()),
            ));
            drop(Box::from_raw(PJRT_Buffer_ToHostBuffer(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_Dimensions(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_ElementType(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_OnDeviceSizeInBytes(
                ptr::null_mut(),
            )));
            drop(Box::from_raw(PJRT_Buffer_Destroy(ptr::null_mut())));
            drop(Box::from_raw(
                PJRT_LoadedExecutable_Destroy(ptr::null_mut()),
            ));

            let host_floats = [1.0_f32, 2.0, 3.0, 4.0];
            let dims_arr = [4_i64];
            let mut h2d_args = crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client,
                data: host_floats.as_ptr().cast(),
                element_type: 11,
                pad0: 0,
                dims: dims_arr.as_ptr(),
                num_dims: 1,
                byte_strides: ptr::null(),
                num_byte_strides: 0,
                host_buffer_semantics: 0,
                pad1: 0,
                device: look_args.device,
                memory: ptr::null_mut(),
                device_layout: ptr::null(),
                done_with_host_buffer: ptr::null_mut(),
                out_buffer: ptr::null_mut(),
            };
            assert!(PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_args).cast()).is_null());
            let buf_in = h2d_args.out_buffer;
            assert!(!buf_in.is_null());

            let mut h2d_args_default_dev = h2d_args;
            h2d_args_default_dev.device = ptr::null_mut();
            h2d_args_default_dev.dims = ptr::null();
            h2d_args_default_dev.num_dims = 0;
            h2d_args_default_dev.out_buffer = ptr::null_mut();
            assert!(
                PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_args_default_dev).cast())
                    .is_null()
            );
            let buf_in2 = h2d_args_default_dev.out_buffer;
            assert!(!buf_in2.is_null());

            let mut h2d_args_zero_ndims = h2d_args;
            h2d_args_zero_ndims.num_dims = 0;
            h2d_args_zero_ndims.out_buffer = ptr::null_mut();
            assert!(
                PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_args_zero_ndims).cast())
                    .is_null()
            );
            libtpu_rs_buffer_destroy(h2d_args_zero_ndims.out_buffer);

            // Test error propagation in PJRT_Client_BufferFromHostBuffer
            let mut h2d_bad_et = h2d_args;
            h2d_bad_et.element_type = 999;
            drop(Box::from_raw(PJRT_Client_BufferFromHostBuffer(
                ptr::addr_of_mut!(h2d_bad_et).cast(),
            )));
            let neg_dim_val = [-1_i64];
            let mut h2d_bad_shape = h2d_args;
            h2d_bad_shape.dims = neg_dim_val.as_ptr();
            h2d_bad_shape.num_dims = 1;
            drop(Box::from_raw(PJRT_Client_BufferFromHostBuffer(
                ptr::addr_of_mut!(h2d_bad_shape).cast(),
            )));
            let mut bad_desc = (*look_args.device).inner.clone();
            bad_desc.description.spec.device_id = DeviceId(99);
            let mut fake_dev = PJRT_Device {
                inner: bad_desc,
                desc_ptr: ptr::null_mut(),
                default_memory_ptr: ptr::null_mut(),
                addressable_memory_ptrs: Vec::new(),
                hal: std::sync::Arc::clone(&(*client).inner.hal),
            };
            let mut h2d_bad_dev = h2d_args;
            h2d_bad_dev.device = ptr::addr_of_mut!(fake_dev);
            drop(Box::from_raw(PJRT_Client_BufferFromHostBuffer(
                ptr::addr_of_mut!(h2d_bad_dev).cast(),
            )));

            let mut plug_attr_args = crate::pjrt_types::PJRT_Plugin_Attributes_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                attributes: ptr::null(),
                num_attributes: 0,
            };
            drop(Box::from_raw(PJRT_Plugin_Attributes(
                ptr::addr_of_mut!(plug_attr_args).cast(),
            )));
            plug_attr_args.struct_size =
                size_of::<crate::pjrt_types::PJRT_Plugin_Attributes_Args>();
            assert!(PJRT_Plugin_Attributes(ptr::addr_of_mut!(plug_attr_args).cast()).is_null());
            assert_eq!(plug_attr_args.num_attributes, 3);
            assert!(!plug_attr_args.attributes.is_null());
            assert_ne!(format!("{:?}", *plug_attr_args.attributes), "");

            let mut b_dims_args = crate::pjrt_types::PJRT_Buffer_Dimensions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dims: ptr::null(),
                num_dims: 0,
            };
            assert!(PJRT_Buffer_Dimensions(ptr::addr_of_mut!(b_dims_args).cast()).is_null());
            assert_eq!(b_dims_args.num_dims, 1);

            let mut b_et_args = crate::pjrt_types::PJRT_Buffer_ElementType_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                element_type: 0,
                pad: 0,
            };
            assert!(PJRT_Buffer_ElementType(ptr::addr_of_mut!(b_et_args).cast()).is_null());
            assert_eq!(b_et_args.element_type, 11);

            let mut b_sz_args = crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                on_device_size_in_bytes: 0,
            };
            assert!(PJRT_Buffer_OnDeviceSizeInBytes(ptr::addr_of_mut!(b_sz_args).cast()).is_null());
            assert!(b_sz_args.on_device_size_in_bytes >= 16);

            // Exercise handle-aware dispatchers with real handles and behavioral assertions
            for &dev0 in (*client).device_ptrs.first() {
                let desc0 = (*dev0).desc_ptr;
                let mut dd_id = crate::pjrt_types::PJRT_DeviceDescription_Id_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_Id_Args>(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    id: -1,
                    pad: 0,
                };
                assert!(PJRT_DeviceDescription_Id(ptr::addr_of_mut!(dd_id).cast()).is_null());
                assert_eq!(dd_id.id, 0);

                let mut dd_proc = crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    process_index: -1,
                    pad: 0,
                };
                assert!(
                    PJRT_DeviceDescription_ProcessIndex(ptr::addr_of_mut!(dd_proc).cast())
                        .is_null()
                );
                assert_eq!(dd_proc.process_index, 0);

                let mut dd_attrs = crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    num_attributes: 0,
                    attributes: ptr::null(),
                };
                assert!(
                    PJRT_DeviceDescription_Attributes(ptr::addr_of_mut!(dd_attrs).cast()).is_null()
                );

                let mut dd_kind = crate::pjrt_types::PJRT_DeviceDescription_Kind_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_Kind_Args>(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    device_kind: ptr::null(),
                    device_kind_size: 0,
                };
                assert!(PJRT_DeviceDescription_Kind(ptr::addr_of_mut!(dd_kind).cast()).is_null());
                assert!(dd_kind.device_kind_size > 0);

                let mut dd_dbg = crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    debug_string: ptr::null(),
                    debug_string_size: 0,
                };
                assert!(
                    PJRT_DeviceDescription_DebugString(ptr::addr_of_mut!(dd_dbg).cast()).is_null()
                );
                assert!(dd_dbg.debug_string_size > 0);

                let mut dd_str = crate::pjrt_types::PJRT_DeviceDescription_ToString_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_ToString_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    to_string: ptr::null(),
                    to_string_size: 0,
                };
                assert!(
                    PJRT_DeviceDescription_ToString(ptr::addr_of_mut!(dd_str).cast()).is_null()
                );
                assert!(dd_str.to_string_size > 0);

                let mut dv_desc = crate::pjrt_types::PJRT_Device_GetDescription_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_GetDescription_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    device_description: ptr::null_mut(),
                };
                assert!(PJRT_Device_GetDescription(ptr::addr_of_mut!(dv_desc).cast()).is_null());
                assert_eq!(dv_desc.device_description, desc0);

                let mut dv_addr = crate::pjrt_types::PJRT_Device_IsAddressable_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_IsAddressable_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    is_addressable: false,
                    pad: [0; 7],
                };
                assert!(PJRT_Device_IsAddressable(ptr::addr_of_mut!(dv_addr).cast()).is_null());
                assert!(dv_addr.is_addressable);

                let mut dv_hw = crate::pjrt_types::PJRT_Device_LocalHardwareId_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_LocalHardwareId_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    local_hardware_id: -1,
                    pad: 0,
                };
                assert!(PJRT_Device_LocalHardwareId(ptr::addr_of_mut!(dv_hw).cast()).is_null());
                assert_eq!(dv_hw.local_hardware_id, 0);

                let mut dv_mems = crate::pjrt_types::PJRT_Device_AddressableMemories_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_AddressableMemories_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    memories: ptr::null(),
                    num_memories: 0,
                };
                assert!(
                    PJRT_Device_AddressableMemories(ptr::addr_of_mut!(dv_mems).cast()).is_null()
                );

                let mut dv_defm = crate::pjrt_types::PJRT_Device_DefaultMemory_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_DefaultMemory_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    memory: ptr::null_mut(),
                };
                assert!(PJRT_Device_DefaultMemory(ptr::addr_of_mut!(dv_defm).cast()).is_null());

                let mut dv_mstat = crate::pjrt_types::PJRT_Device_MemoryStats_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_MemoryStats_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    bytes_in_use: 0,
                    peak_bytes_in_use: 0,
                    peak_bytes_in_use_is_set: false,
                    pad0: [0; 7],
                    num_allocs: 0,
                    num_allocs_is_set: false,
                    pad1: [0; 7],
                    largest_alloc_size: 0,
                    largest_alloc_size_is_set: false,
                    pad2: [0; 7],
                    bytes_limit: 0,
                    bytes_limit_is_set: false,
                    pad3: [0; 7],
                    bytes_reserved: 0,
                    bytes_reserved_is_set: false,
                    pad4: [0; 7],
                    peak_bytes_reserved: 0,
                    peak_bytes_reserved_is_set: false,
                    pad5: [0; 7],
                    bytes_reservable_limit: 0,
                    bytes_reservable_limit_is_set: false,
                    pad6: [0; 7],
                    largest_free_block_bytes: 0,
                    largest_free_block_bytes_is_set: false,
                    pad7: [0; 7],
                    pool_bytes: 0,
                    pool_bytes_is_set: false,
                    pad8: [0; 7],
                    peak_pool_bytes: 0,
                    peak_pool_bytes_is_set: false,
                    pad9: [0; 7],
                };
                assert!(PJRT_Device_MemoryStats(ptr::addr_of_mut!(dv_mstat).cast()).is_null());
                assert!(dv_mstat.bytes_in_use > 0);
                assert!(dv_mstat.peak_bytes_in_use_is_set);

                let mut dv_clr = crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                };
                assert!(PJRT_Device_ClearMemoryStats(ptr::addr_of_mut!(dv_clr).cast()).is_null());

                let mut dv_pois = crate::pjrt_types::PJRT_Device_PoisonExecution_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_PoisonExecution_Args>(),
                    extension_start: ptr::null_mut(),
                    device: (*client).device_ptrs.last().copied().unwrap_or(dev0),
                    launch_id: 1,
                    error_code: 3,
                    error_message: ptr::null(),
                    error_message_size: 0,
                    reserved0: 0,
                    reserved1: 0,
                    poisoned: false,
                    pad: [0; 7],
                };
                assert!(PJRT_Device_PoisonExecution(ptr::addr_of_mut!(dv_pois).cast()).is_null());
                assert!(dv_pois.poisoned);
                dv_pois.launch_id = -1;
                assert!(PJRT_Device_PoisonExecution(ptr::addr_of_mut!(dv_pois).cast()).is_null());
            }
            for &mem0 in (*client).memory_ptrs.first() {
                let mut mm_id = crate::pjrt_types::PJRT_Memory_Id_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_Id_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    id: -1,
                    pad: 0,
                };
                assert!(PJRT_Memory_Id(ptr::addr_of_mut!(mm_id).cast()).is_null());
                assert_eq!(mm_id.id, 0);

                let mut mm_kind = crate::pjrt_types::PJRT_Memory_Kind_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_Kind_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    kind: ptr::null(),
                    kind_size: 0,
                };
                assert!(PJRT_Memory_Kind(ptr::addr_of_mut!(mm_kind).cast()).is_null());
                assert!(mm_kind.kind_size > 0);

                let mut mm_dbg = crate::pjrt_types::PJRT_Memory_DebugString_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_DebugString_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    debug_string: ptr::null(),
                    debug_string_size: 0,
                };
                assert!(PJRT_Memory_DebugString(ptr::addr_of_mut!(mm_dbg).cast()).is_null());
                assert!(mm_dbg.debug_string_size > 0);

                let mut mm_str = crate::pjrt_types::PJRT_Memory_ToString_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_ToString_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    to_string: ptr::null(),
                    to_string_size: 0,
                };
                assert!(PJRT_Memory_ToString(ptr::addr_of_mut!(mm_str).cast()).is_null());
                assert!(mm_str.to_string_size > 0);

                let mut mm_devs = crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    devices: ptr::null(),
                    num_devices: 0,
                };
                assert!(
                    PJRT_Memory_AddressableByDevices(ptr::addr_of_mut!(mm_devs).cast()).is_null()
                );

                let mut mm_kid = crate::pjrt_types::PJRT_Memory_Kind_Id_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_Kind_Id_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    kind_id: -1,
                    pad: 0,
                };
                assert!(PJRT_Memory_Kind_Id(ptr::addr_of_mut!(mm_kid).cast()).is_null());
                assert!(mm_kid.kind_id >= 0);
            }
            let exec0 = (*comp_args.executable).executable_ptr;
            let mut ex_name = crate::pjrt_types::PJRT_Executable_Name_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Name_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                executable_name: ptr::null(),
                executable_name_size: 0,
            };
            assert!(PJRT_Executable_Name(ptr::addr_of_mut!(ex_name).cast()).is_null());
            assert!(ex_name.executable_name_size > 0);

            let mut ex_rep = crate::pjrt_types::PJRT_Executable_NumReplicas_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_NumReplicas_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                num_replicas: 0,
            };
            assert!(PJRT_Executable_NumReplicas(ptr::addr_of_mut!(ex_rep).cast()).is_null());
            assert_eq!(ex_rep.num_replicas, 1);

            let mut ex_part = crate::pjrt_types::PJRT_Executable_NumPartitions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_NumPartitions_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                num_partitions: 0,
            };
            assert!(PJRT_Executable_NumPartitions(ptr::addr_of_mut!(ex_part).cast()).is_null());
            assert_eq!(ex_part.num_partitions, 1);

            let mut ex_outs = crate::pjrt_types::PJRT_Executable_NumOutputs_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_NumOutputs_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                num_outputs: 0,
            };
            assert!(PJRT_Executable_NumOutputs(ptr::addr_of_mut!(ex_outs).cast()).is_null());
            assert_eq!(ex_outs.num_outputs, 1);

            let mut ex_code_sz =
                crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    executable: exec0,
                    size_in_bytes: 0,
                };
            assert!(
                PJRT_Executable_SizeOfGeneratedCodeInBytes(ptr::addr_of_mut!(ex_code_sz).cast())
                    .is_null()
            );
            assert!(ex_code_sz.size_in_bytes >= 256);

            let mut ex_cost = crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                num_properties: 0,
                properties: ptr::null(),
            };
            assert!(PJRT_Executable_GetCostAnalysis(ptr::addr_of_mut!(ex_cost).cast()).is_null());
            assert_eq!(ex_cost.num_properties, 3);

            let mut ex_omk = crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                num_outputs: 0,
                memory_kinds: ptr::null(),
                memory_kind_sizes: ptr::null(),
            };
            assert!(PJRT_Executable_OutputMemoryKinds(ptr::addr_of_mut!(ex_omk).cast()).is_null());
            assert_eq!(ex_omk.num_outputs, 1);

            let mut ex_opt = crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                program: ptr::null_mut(),
            };
            assert!(PJRT_Executable_OptimizedProgram(ptr::addr_of_mut!(ex_opt).cast()).is_null());
            let mut opt_prog_buf = [0_u8; 256];
            let mut opt_prog = PJRT_Program {
                struct_size: size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: ptr::null(),
                code_size: 0,
                format: ptr::null(),
                format_size: 0,
            };
            ex_opt.program = ptr::addr_of_mut!(opt_prog);
            assert!(PJRT_Executable_OptimizedProgram(ptr::addr_of_mut!(ex_opt).cast()).is_null());
            assert!(opt_prog.code_size > 0);
            let req_opt_size = opt_prog.code_size;
            opt_prog.code = opt_prog_buf.as_mut_ptr().cast();
            opt_prog.code_size = 1;
            drop(Box::from_raw(PJRT_Executable_OptimizedProgram(
                ptr::addr_of_mut!(ex_opt).cast(),
            )));
            opt_prog.code_size = opt_prog_buf.len();
            assert!(PJRT_Executable_OptimizedProgram(ptr::addr_of_mut!(ex_opt).cast()).is_null());
            assert_eq!(opt_prog.code_size, req_opt_size);
            assert_eq!(
                opt_prog_buf.get(..req_opt_size),
                Some(code_bytes.as_slice())
            );

            let mut ex_ser = crate::pjrt_types::PJRT_Executable_Serialize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Serialize_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_executable: ptr::null_mut(),
                serialized_executable_deleter: None,
            };
            assert!(PJRT_Executable_Serialize(ptr::addr_of_mut!(ex_ser).cast()).is_null());
            assert!(ex_ser.serialized_bytes_size > 0);

            let mut ex_oet = crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args>(
                ),
                extension_start: ptr::null_mut(),
                executable: exec0,
                output_types: ptr::null(),
                num_output_types: 0,
            };
            assert!(PJRT_Executable_OutputElementTypes(ptr::addr_of_mut!(ex_oet).cast()).is_null());
            assert_eq!(ex_oet.num_output_types, 1);

            let mut ex_odim = crate::pjrt_types::PJRT_Executable_OutputDimensions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_OutputDimensions_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                num_outputs: 0,
                dims: ptr::null(),
                dim_sizes: ptr::null(),
            };
            assert!(PJRT_Executable_OutputDimensions(ptr::addr_of_mut!(ex_odim).cast()).is_null());
            assert_eq!(ex_odim.num_outputs, 1);

            let mut ex_fp = crate::pjrt_types::PJRT_Executable_Fingerprint_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Fingerprint_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                executable_fingerprint: ptr::null(),
                executable_fingerprint_size: 0,
            };
            assert!(PJRT_Executable_Fingerprint(ptr::addr_of_mut!(ex_fp).cast()).is_null());
            assert_eq!(ex_fp.executable_fingerprint_size, 32);

            let mut ex_cms = crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                generated_code_size_in_bytes: 0,
                argument_size_in_bytes: 0,
                output_size_in_bytes: 0,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 0,
                host_generated_code_size_in_bytes: 0,
                host_argument_size_in_bytes: 0,
                host_output_size_in_bytes: 0,
                host_alias_size_in_bytes: 0,
                host_temp_size_in_bytes: 0,
                peak_memory_in_bytes: 0,
            };
            assert!(
                PJRT_Executable_GetCompiledMemoryStats(ptr::addr_of_mut!(ex_cms).cast()).is_null()
            );
            assert!(ex_cms.generated_code_size_in_bytes >= 256);

            let mut ex_gco = crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_compile_options: ptr::null_mut(),
                serialized_compile_options_deleter: None,
            };
            assert!(PJRT_Executable_GetCompileOptions(ptr::addr_of_mut!(ex_gco).cast()).is_null());

            let mut ex_pmk = crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args>(
                ),
                extension_start: ptr::null_mut(),
                executable: exec0,
                num_parameters: 0,
                memory_kinds: ptr::null(),
                memory_kind_sizes: ptr::null(),
            };
            assert!(
                PJRT_Executable_ParameterMemoryKinds(ptr::addr_of_mut!(ex_pmk).cast()).is_null()
            );

            let mut ex_del_noop = crate::pjrt_types::PJRT_Executable_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec0,
            };
            assert!(PJRT_Executable_Destroy(ptr::addr_of_mut!(ex_del_noop).cast()).is_null());

            let loaded0 = comp_args.executable;
            let mut le_get = crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args>(
                ),
                extension_start: ptr::null_mut(),
                loaded_executable: loaded0,
                executable: ptr::null_mut(),
            };
            assert!(
                PJRT_LoadedExecutable_GetExecutable(ptr::addr_of_mut!(le_get).cast()).is_null()
            );
            assert_eq!(le_get.executable, exec0);

            let mut le_devs = crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: loaded0,
                addressable_devices: ptr::null(),
                num_addressable_devices: 0,
            };
            assert!(
                PJRT_LoadedExecutable_AddressableDevices(ptr::addr_of_mut!(le_devs).cast())
                    .is_null()
            );
            assert_eq!(le_devs.num_addressable_devices, 4);

            let mut le_isdel = crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args>(),
                extension_start: ptr::null_mut(),
                executable: loaded0,
                is_deleted: true,
                pad: [0; 7],
            };
            assert!(PJRT_LoadedExecutable_IsDeleted(ptr::addr_of_mut!(le_isdel).cast()).is_null());
            assert!(!le_isdel.is_deleted);

            let mut le_fp = crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args>(),
                extension_start: ptr::null_mut(),
                executable: loaded0,
                executable_fingerprint: ptr::null(),
                executable_fingerprint_size: 0,
            };
            assert!(PJRT_LoadedExecutable_Fingerprint(ptr::addr_of_mut!(le_fp).cast()).is_null());
            assert_eq!(le_fp.executable_fingerprint_size, 32);

            let mut le_da = crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: loaded0,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_device_assignment: ptr::null_mut(),
                serialized_device_assignment_deleter: None,
            };
            assert!(
                PJRT_LoadedExecutable_GetDeviceAssignment(ptr::addr_of_mut!(le_da).cast())
                    .is_null()
            );
            assert!(le_da.serialized_bytes_size > 0);

            let mut le_lids =
                crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    executable: loaded0,
                    addressable_device_logical_ids: ptr::null(),
                    num_addressable_device_logical_ids: 0,
                };
            assert!(
                PJRT_LoadedExecutable_AddressableDeviceLogicalIds(
                    ptr::addr_of_mut!(le_lids).cast()
                )
                .is_null()
            );
            assert_eq!(le_lids.num_addressable_device_logical_ids, 4);

            let mut bf_udim = crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                unpadded_dims: ptr::null(),
                num_dims: 0,
            };
            assert!(PJRT_Buffer_UnpaddedDimensions(ptr::addr_of_mut!(bf_udim).cast()).is_null());
            assert_eq!(bf_udim.num_dims, 1);

            let mut bf_ddim = crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args>(
                ),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dynamic_dimension_indices: ptr::null(),
                num_dynamic_dimensions: 0,
            };
            assert!(
                PJRT_Buffer_DynamicDimensionIndices(ptr::addr_of_mut!(bf_ddim).cast()).is_null()
            );

            let mut bf_gml = crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                layout_words: [0; 10],
            };
            assert!(PJRT_Buffer_GetMemoryLayout(ptr::addr_of_mut!(bf_gml).cast()).is_null());
            assert_eq!(bf_gml.layout_words[0], 80);
            assert_eq!(bf_gml.layout_words[3], 1);
            assert_eq!(bf_gml.layout_words[6], 1);
            (*buf_in).inner.shape.layout = libtpu_core::types::MemoryLayout::Strides {
                byte_strides: vec![4],
            };
            let strided_buf = PJRT_Buffer::new(
                libtpu_core::pjrt::PjrtCoreBuffer::new(
                    (*buf_in).inner.device_id,
                    (*buf_in).inner.memory_id,
                    (*buf_in).inner.shape.clone(),
                    (*buf_in).inner.address,
                    std::sync::Arc::clone(&(*buf_in).inner.hal),
                ),
                (*buf_in).device_ptr,
                (*buf_in).memory_ptr,
            );
            assert_eq!(strided_buf.tile_dims_flat, vec![4]);
            assert!(PJRT_Buffer_GetMemoryLayout(ptr::addr_of_mut!(bf_gml).cast()).is_null());
            (*buf_in).inner.shape.layout = libtpu_core::types::MemoryLayout::default_tpu_tiled(1);

            let mut bf_dev = crate::pjrt_types::PJRT_Buffer_Device_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Device_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                device: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_Device(ptr::addr_of_mut!(bf_dev).cast()).is_null());
            assert!(!bf_dev.device.is_null());

            let mut bf_mem = crate::pjrt_types::PJRT_Buffer_Memory_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Memory_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                memory: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_Memory(ptr::addr_of_mut!(bf_mem).cast()).is_null());
            assert!(!bf_mem.memory.is_null());

            let mut bf_isdel = crate::pjrt_types::PJRT_Buffer_IsDeleted_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_IsDeleted_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                is_deleted: true,
                pad: [0; 7],
            };
            assert!(PJRT_Buffer_IsDeleted(ptr::addr_of_mut!(bf_isdel).cast()).is_null());
            assert!(!bf_isdel.is_deleted);

            let mut bf_cpu = crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                is_on_cpu: true,
                pad: [0; 7],
            };
            assert!(PJRT_Buffer_IsOnCpu(ptr::addr_of_mut!(bf_cpu).cast()).is_null());
            assert!(!bf_cpu.is_on_cpu);

            let mut bf_rev = crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                event: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_ReadyEvent(ptr::addr_of_mut!(bf_rev).cast()).is_null());

            let mut bf_uptr = crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                buffer_pointer: 0,
            };
            assert!(PJRT_Buffer_UnsafePointer(ptr::addr_of_mut!(bf_uptr).cast()).is_null());
            assert_ne!(bf_uptr.buffer_pointer, 0);

            let mut bf_inc = crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
            };
            assert!(
                PJRT_Buffer_IncreaseExternalReferenceCount(ptr::addr_of_mut!(bf_inc).cast())
                    .is_null()
            );

            let mut bf_dec = crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
            };
            assert!(
                PJRT_Buffer_DecreaseExternalReferenceCount(ptr::addr_of_mut!(bf_dec).cast())
                    .is_null()
            );

            let mut bf_optr = crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                device_memory_ptr: ptr::null_mut(),
            };
            assert!(
                PJRT_Buffer_OpaqueDeviceMemoryDataPointer(ptr::addr_of_mut!(bf_optr).cast())
                    .is_null()
            );
            assert!(!bf_optr.device_memory_ptr.is_null());

            let mut bf_crhf = crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
                callback_data: ptr::null_mut(),
                future_ready_callback: ptr::null(),
            };
            assert!(PJRT_Buffer_CopyRawToHostFuture(ptr::addr_of_mut!(bf_crhf).cast()).is_null());

            let mut bf_don = crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                callback_data: ptr::null_mut(),
                dependency_callback: ptr::null(),
                out_buffer: ptr::null_mut(),
            };
            assert!(
                PJRT_Buffer_DonateWithControlDependency(ptr::addr_of_mut!(bf_don).cast()).is_null()
            );
            assert_eq!(bf_don.out_buffer, buf_in);
            pjrt_serialized_noop_deleter(ptr::null_mut());

            // Exercise non-null branches of DeserializeAndLoad, CopyToDevice, CopyToMemory, AddChunk, Compile, CreateViewOfDeviceBuffer, CopyRawToHost, ExecuteContext_Destroy, and AsyncTrackingEvent_Destroy
            pjrt_serialized_noop_deleter(ptr::null_mut());
            let mut deser_args = crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args>(
                ),
                extension_start: ptr::null_mut(),
                client,
                serialized_executable: code_bytes.as_ptr().cast(),
                serialized_executable_size: code_bytes.len(),
                loaded_executable: ptr::null_mut(),
                overridden_serialized_compile_options: ptr::null(),
                overridden_serialized_compile_options_size: 0,
            };
            assert!(
                PJRT_Executable_DeserializeAndLoad(ptr::addr_of_mut!(deser_args).cast()).is_null()
            );
            assert!(!deser_args.loaded_executable.is_null());
            libtpu_rs_executable_destroy(deser_args.loaded_executable);
            deser_args.serialized_executable = ptr::null();
            deser_args.serialized_executable_size = 0;
            deser_args.loaded_executable = ptr::null_mut();
            assert!(
                PJRT_Executable_DeserializeAndLoad(ptr::addr_of_mut!(deser_args).cast()).is_null()
            );
            libtpu_rs_executable_destroy(deser_args.loaded_executable);
            deser_args.serialized_executable = c"bad".as_ptr();
            deser_args.serialized_executable_size = 3;
            deser_args.loaded_executable = ptr::null_mut();
            drop(Box::from_raw(PJRT_Executable_DeserializeAndLoad(
                ptr::addr_of_mut!(deser_args).cast(),
            )));
            drop(Box::from_raw(PJRT_Executable_DeserializeAndLoad(
                ptr::null_mut(),
            )));

            let mut c2d_args = crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dst_device: look_args.device,
                dst_buffer: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_CopyToDevice(ptr::addr_of_mut!(c2d_args).cast()).is_null());
            assert!(!c2d_args.dst_buffer.is_null());
            libtpu_rs_buffer_destroy(c2d_args.dst_buffer);

            for &mem0 in (*client).memory_ptrs.first() {
                let mut c2m_args = crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args>(),
                    extension_start: ptr::null_mut(),
                    buffer: buf_in,
                    dst_memory: mem0,
                    dst_buffer: ptr::null_mut(),
                };
                assert!(PJRT_Buffer_CopyToMemory(ptr::addr_of_mut!(c2m_args).cast()).is_null());
                assert!(!c2m_args.dst_buffer.is_null());
                libtpu_rs_buffer_destroy(c2m_args.dst_buffer);
            }

            unsafe extern "C" fn test_chunk_deleter(
                _data: *mut std::ffi::c_void,
                arg: *mut std::ffi::c_void,
            ) {
                // SAFETY: `arg` points to a mutable `u32` counter.
                unsafe {
                    *arg.cast::<u32>() += 1;
                }
            }
            unsafe extern "C" fn test_view_deleter(arg: *mut std::ffi::c_void) {
                // SAFETY: `arg` points to a mutable `u32` counter.
                unsafe {
                    *arg.cast::<u32>() += 1;
                }
            }
            let sample_stream =
                Box::into_raw(Box::new(crate::pjrt_types::PJRT_CopyToDeviceStream {
                    inner: libtpu_core::pjrt::PjrtCoreCopyToDeviceStream::new(256, 128),
                }));
            for st_fn in [
                PJRT_CopyToDeviceStream_AddChunk
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_CopyToDeviceStream_TotalBytes,
                PJRT_CopyToDeviceStream_GranuleSize,
                PJRT_CopyToDeviceStream_CurrentBytes,
            ] {
                let mut st_args = [0_usize; 8];
                st_args[0] = 64;
                st_args[2] = sample_stream as usize;
                assert!(st_fn(st_args.as_mut_ptr().cast()).is_null());
            }
            let mut chunk_del_hits: u32 = 0;
            let mut chunk_payload = [42_u8; 64];
            let mut chunk_val = crate::pjrt_types::PJRT_Chunk {
                data: chunk_payload.as_mut_ptr().cast(),
                size: 64,
                deleter: Some(test_chunk_deleter),
                deleter_arg: ptr::addr_of_mut!(chunk_del_hits).cast(),
            };
            let mut add_chunk_args = crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args>(),
                extension_start: ptr::null_mut(),
                stream: sample_stream,
                chunk: ptr::addr_of_mut!(chunk_val),
                transfer_complete: ptr::null_mut(),
            };
            assert!(
                PJRT_CopyToDeviceStream_AddChunk(ptr::addr_of_mut!(add_chunk_args).cast())
                    .is_null()
            );
            assert_eq!(chunk_del_hits, 1);
            let mut chunk_val_no_del = crate::pjrt_types::PJRT_Chunk {
                data: ptr::null_mut(),
                size: 64,
                deleter: None,
                deleter_arg: ptr::null_mut(),
            };
            add_chunk_args.chunk = ptr::addr_of_mut!(chunk_val_no_del);
            assert!(
                PJRT_CopyToDeviceStream_AddChunk(ptr::addr_of_mut!(add_chunk_args).cast())
                    .is_null()
            );
            add_chunk_args.stream = ptr::null_mut();
            assert!(
                PJRT_CopyToDeviceStream_AddChunk(ptr::addr_of_mut!(add_chunk_args).cast())
                    .is_null()
            );
            let mut st_del_args = crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                stream: sample_stream,
            };
            assert!(
                PJRT_CopyToDeviceStream_Destroy(ptr::addr_of_mut!(st_del_args).cast()).is_null()
            );

            let mut aot_args = crate::pjrt_types::PJRT_Compile_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Compile_Args>(),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr,
                program: ptr::addr_of!(prog),
                compile_options: c"opts".as_ptr(),
                compile_options_size: 4,
                client,
                executable: ptr::null_mut(),
            };
            assert!(PJRT_Compile(ptr::addr_of_mut!(aot_args).cast()).is_null());
            assert!(!aot_args.executable.is_null());
            let mut aot_del = crate::pjrt_types::PJRT_Executable_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: aot_args.executable,
            };
            assert!(PJRT_Executable_Destroy(ptr::addr_of_mut!(aot_del).cast()).is_null());
            let empty_prog = PJRT_Program {
                struct_size: size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: ptr::null(),
                code_size: 0,
                format: ptr::null(),
                format_size: 0,
            };
            aot_args.program = ptr::addr_of!(empty_prog);
            aot_args.compile_options = ptr::null();
            aot_args.compile_options_size = 0;
            aot_args.executable = ptr::null_mut();
            assert!(PJRT_Compile(ptr::addr_of_mut!(aot_args).cast()).is_null());
            aot_del.executable = aot_args.executable;
            assert!(PJRT_Executable_Destroy(ptr::addr_of_mut!(aot_del).cast()).is_null());
            let bad_aot_prog = PJRT_Program {
                struct_size: size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: b"SYNTAX_ERROR".as_ptr().cast(),
                code_size: 12,
                format: b"mlir".as_ptr().cast(),
                format_size: 4,
            };
            aot_args.program = ptr::addr_of!(bad_aot_prog);
            assert!(PJRT_Compile(ptr::addr_of_mut!(aot_args).cast()).is_null());
            assert!(aot_args.executable.is_null());

            let mut view_del_hits: u32 = 0;
            let mut view_args = crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args>(
                ),
                extension_start: ptr::null_mut(),
                client,
                device_buffer_ptr: ptr::addr_of_mut!(view_del_hits).cast(),
                dims: dims_arr.as_ptr(),
                num_dims: 1,
                element_type: 11,
                pad: 0,
                layout: ptr::null(),
                device: look_args.device,
                on_delete_callback: test_view_deleter as *const std::ffi::c_void,
                buffer: ptr::null_mut(),
            };
            assert!(
                PJRT_Client_CreateViewOfDeviceBuffer(ptr::addr_of_mut!(view_args).cast()).is_null()
            );
            assert_eq!(view_del_hits, 1);
            libtpu_rs_buffer_destroy(view_args.buffer);
            view_args.dims = neg_dim_val.as_ptr();
            view_args.buffer = ptr::null_mut();
            assert!(
                PJRT_Client_CreateViewOfDeviceBuffer(ptr::addr_of_mut!(view_args).cast()).is_null()
            );
            libtpu_rs_buffer_destroy(view_args.buffer);

            let mut raw_dst = [0_u8; 16];
            let mut raw_copy_h = crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dst: raw_dst.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_CopyRawToHost(ptr::addr_of_mut!(raw_copy_h).cast()).is_null());
            raw_copy_h.offset = 9999;
            assert!(PJRT_Buffer_CopyRawToHost(ptr::addr_of_mut!(raw_copy_h).cast()).is_null());

            let mut ec_create = crate::pjrt_types::PJRT_ExecuteContext_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_ExecuteContext_Create_Args>(),
                extension_start: ptr::null_mut(),
                context: ptr::null_mut(),
            };
            assert!(PJRT_ExecuteContext_Create(ptr::addr_of_mut!(ec_create).cast()).is_null());
            let mut ec_destroy = crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                context: ec_create.context,
            };
            assert!(PJRT_ExecuteContext_Destroy(ptr::addr_of_mut!(ec_destroy).cast()).is_null());

            let mut ate_create = crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args>(
                ),
                extension_start: ptr::null_mut(),
                device: look_args.device,
                description: c"trace".as_ptr(),
                description_size: 5,
                event: ptr::null_mut(),
            };
            assert!(
                PJRT_Device_CreateAsyncTrackingEvent(ptr::addr_of_mut!(ate_create).cast())
                    .is_null()
            );
            let mut ate_destroy = crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ate_create.event,
            };
            assert!(
                PJRT_AsyncTrackingEvent_Destroy(ptr::addr_of_mut!(ate_destroy).cast()).is_null()
            );

            // Behavioral tests for typed PJRT extension argument structs with real handles
            let pass_name = b"custom_fusion_pass";
            let mut xla_reg = crate::extensions::PJRT_XlaTransform_Register_Args {
                struct_size: size_of::<crate::extensions::PJRT_XlaTransform_Register_Args>(),
                extension_start: ptr::null_mut(),
                name: pass_name.as_ptr().cast(),
                name_size: pass_name.len(),
                transform_fn: 0,
            };
            assert!(
                crate::extensions::PJRT_XlaTransform_Register(ptr::addr_of_mut!(xla_reg).cast())
                    .is_null()
            );
            assert!(xla_reg.transform_fn > 0);
            let mut xla_list = crate::extensions::PJRT_XlaTransform_List_Args {
                struct_size: size_of::<crate::extensions::PJRT_XlaTransform_List_Args>(),
                extension_start: ptr::null_mut(),
                names_out: ptr::null(),
                num_names_out: 0,
            };
            assert!(
                crate::extensions::PJRT_XlaTransform_List(ptr::addr_of_mut!(xla_list).cast())
                    .is_null()
            );
            assert!(xla_list.num_names_out >= 1);
            let mut xla_unreg = crate::extensions::PJRT_XlaTransform_Unregister_Args {
                struct_size: size_of::<crate::extensions::PJRT_XlaTransform_Unregister_Args>(),
                extension_start: ptr::null_mut(),
                name: pass_name.as_ptr().cast(),
                name_size: pass_name.len(),
            };
            assert!(
                crate::extensions::PJRT_XlaTransform_Unregister(
                    ptr::addr_of_mut!(xla_unreg).cast()
                )
                .is_null()
            );

            let mut comm_ar = crate::extensions::PJRT_Communicator_AllReduce_Args {
                struct_size: size_of::<crate::extensions::PJRT_Communicator_AllReduce_Args>(),
                extension_start: ptr::null_mut(),
                communicator: std::ptr::dangling_mut::<std::ffi::c_void>(),
                send_buffer: buf_in.cast(),
                recv_buffer: buf_in.cast(),
                reduction_op: 0,
                pad: 0,
                count: 4,
                event: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_Communicator_AllReduce(ptr::addr_of_mut!(comm_ar).cast())
                    .is_null()
            );
            assert!(!comm_ar.event.is_null());

            let mut tpu_mem_stats =
                crate::extensions::PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
                    struct_size: size_of::<
                        crate::extensions::PJRT_TpuExecutable_GetCompiledMemoryStats_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    executable: (*comp_args.executable).executable_ptr.cast(),
                    generated_code_size_in_bytes: 0,
                    argument_size_in_bytes: 0,
                    output_size_in_bytes: 0,
                    alias_size_in_bytes: 0,
                    temp_size_in_bytes: 0,
                };
            assert!(
                crate::extensions::PJRT_TpuExecutable_GetCompiledMemoryStats(
                    ptr::addr_of_mut!(tpu_mem_stats).cast()
                )
                .is_null()
            );
            assert!(tpu_mem_stats.generated_code_size_in_bytes >= 256);

            let pred_err = box_pjrt_error(LibTpuError::DevicePoisoned {
                device_id: 0,
                launch_id: 1,
                reason: String::from("poisoned"),
            });
            let mut pred_chk = crate::extensions::PJRT_TpuExecutable_ErrorCheck_Args {
                struct_size: size_of::<crate::extensions::PJRT_TpuExecutable_ErrorCheck_Args>(),
                extension_start: ptr::null_mut(),
                error: pred_err.cast(),
                is_matched: 0,
            };
            assert!(
                crate::extensions::PJRT_TpuExecutable_IsPredeterminedError(
                    ptr::addr_of_mut!(pred_chk).cast()
                )
                .is_null()
            );
            assert_eq!(pred_chk.is_matched, 1);
            assert!(
                crate::extensions::PJRT_TpuExecutable_IsCancellationPlaceholderError(
                    ptr::addr_of_mut!(pred_chk).cast()
                )
                .is_null()
            );
            assert_eq!(pred_chk.is_matched, 0);
            drop(Box::from_raw(pred_err));

            let mut topo_map = crate::extensions::PJRT_TpuTopology_MappingQuery_Args {
                struct_size: size_of::<crate::extensions::PJRT_TpuTopology_MappingQuery_Args>(),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr.cast(),
                param0: 5,
                out_first: 0,
                out_second: 0,
            };
            assert!(
                crate::extensions::PJRT_TpuTopology_ProcIdAndIdxOnProcForChip(
                    ptr::addr_of_mut!(topo_map).cast()
                )
                .is_null()
            );
            assert_eq!(topo_map.out_first, 1);
            assert_eq!(topo_map.out_second, 1);

            unsafe extern "C" fn sample_ext_cb(
                payload: *mut std::ffi::c_void,
                user_arg: *mut std::ffi::c_void,
            ) {
                if !user_arg.is_null() {
                    // SAFETY: `user_arg` points to `cb_counter` in the test frame.
                    unsafe {
                        *user_arg.cast::<usize>() += payload as usize;
                    }
                }
            }
            let mut cb_counter: usize = 0;
            let cb_fn_ptr: unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) =
                sample_ext_cb;
            let mut cb_reg = crate::extensions::PJRT_Callback_Register_Args {
                struct_size: size_of::<crate::extensions::PJRT_Callback_Register_Args>(),
                extension_start: ptr::null_mut(),
                client: client.cast(),
                callback_type: 7,
                callback_fn: cb_fn_ptr as *const std::ffi::c_void,
                user_arg: ptr::addr_of_mut!(cb_counter).cast(),
            };
            assert!(
                crate::extensions::PJRT_Callback_RegisterCallback(ptr::addr_of_mut!(cb_reg).cast())
                    .is_null()
            );
            let mut cb_inv_real = crate::extensions::PJRT_Callback_Invoke_Args {
                struct_size: size_of::<crate::extensions::PJRT_Callback_Invoke_Args>(),
                extension_start: ptr::null_mut(),
                client: client.cast(),
                callback_type: 7,
                payload: 42 as *mut std::ffi::c_void,
            };
            assert!(
                crate::extensions::PJRT_Callback_InvokeCallback(
                    ptr::addr_of_mut!(cb_inv_real).cast()
                )
                .is_null()
            );
            assert_eq!(cb_counter, 42);

            let mut raw_host_bytes: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let mut raw_h2d = crate::extensions::PJRT_RawBuffer_CopyRaw_Args {
                struct_size: size_of::<crate::extensions::PJRT_RawBuffer_CopyRaw_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in.cast(),
                host_ptr: raw_host_bytes.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_RawBuffer_CopyRawHostToDevice(
                    ptr::addr_of_mut!(raw_h2d).cast()
                )
                .is_null()
            );
            let mut raw_d2h_out = vec![0_u8; 16];
            let mut raw_d2h = crate::extensions::PJRT_RawBuffer_CopyRaw_Args {
                struct_size: size_of::<crate::extensions::PJRT_RawBuffer_CopyRaw_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in.cast(),
                host_ptr: raw_d2h_out.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_RawBuffer_CopyRawDeviceToHost(
                    ptr::addr_of_mut!(raw_d2h).cast()
                )
                .is_null()
            );
            assert_eq!(raw_d2h_out, raw_host_bytes);

            let triton_src = b"module @triton { stablehlo.add }";
            let mut triton_call = crate::extensions::PJRT_Triton_Compile_Args {
                struct_size: size_of::<crate::extensions::PJRT_Triton_Compile_Args>(),
                extension_start: ptr::null_mut(),
                module: ptr::null(),
                module_size: 0,
                arch_name: ptr::null(),
                arch_name_size: 0,
                num_warps: 4,
                num_ctas: 1,
                num_stages: 2,
                pad: 0,
                out_asm: ptr::null(),
                out_asm_size: 0,
                out_smem_bytes: 0,
            };
            assert!(
                crate::extensions::PJRT_Triton_Compile(ptr::addr_of_mut!(triton_call).cast())
                    .is_null()
            );
            triton_call.module = triton_src.as_ptr().cast();
            triton_call.module_size = triton_src.len();
            assert!(
                crate::extensions::PJRT_Triton_Compile(ptr::addr_of_mut!(triton_call).cast())
                    .is_null()
            );
            assert!(triton_call.out_asm_size > 0);
            assert_eq!(triton_call.out_smem_bytes, 4096);
            drop(Box::from_raw(crate::extensions::PJRT_Triton_Compile(
                ptr::null_mut(),
            )));

            let mut tp_pname = crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args>(
                ),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr,
                platform_name: ptr::null(),
                platform_name_size: 0,
            };
            assert!(
                PJRT_TopologyDescription_PlatformName(ptr::addr_of_mut!(tp_pname).cast()).is_null()
            );
            assert_eq!(tp_pname.platform_name_size, 3);

            let mut tp_pver = crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args,
                >(),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr,
                platform_version: ptr::null(),
                platform_version_size: 0,
            };
            assert!(
                PJRT_TopologyDescription_PlatformVersion(ptr::addr_of_mut!(tp_pver).cast())
                    .is_null()
            );
            assert!(tp_pver.platform_version_size > 0);

            let mut tp_descs =
                crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    topology: (*client).topology_ptr,
                    descriptions: ptr::null(),
                    num_descriptions: 0,
                };
            assert!(
                PJRT_TopologyDescription_GetDeviceDescriptions(ptr::addr_of_mut!(tp_descs).cast())
                    .is_null()
            );
            assert_eq!(tp_descs.num_descriptions, 4);
            assert!(!tp_descs.descriptions.is_null());

            let mut tp_ser = crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args>(
                ),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_topology: ptr::null_mut(),
                serialized_topology_deleter: None,
            };
            assert!(PJRT_TopologyDescription_Serialize(ptr::addr_of_mut!(tp_ser).cast()).is_null());
            assert!(tp_ser.serialized_bytes_size > 0);
            assert!(!tp_ser.serialized_bytes.is_null());
            let mut tp_deser = crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args>(
                ),
                extension_start: ptr::null_mut(),
                serialized_topology: tp_ser.serialized_bytes,
                serialized_topology_size: tp_ser.serialized_bytes_size,
                topology: ptr::null_mut(),
            };
            assert!(
                PJRT_TopologyDescription_Deserialize(ptr::addr_of_mut!(tp_deser).cast()).is_null()
            );
            assert!(!tp_deser.topology.is_null());
            let mut tp_del_deser = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                topology: tp_deser.topology,
            };
            assert!(
                PJRT_TopologyDescription_Destroy(ptr::addr_of_mut!(tp_del_deser).cast()).is_null()
            );
            if let Some(del) = tp_ser.serialized_topology_deleter {
                del(tp_ser.serialized_topology.cast());
            }
            if let Some(del) = ex_ser.serialized_executable_deleter {
                del(ex_ser.serialized_executable.cast());
            }
            if let Some(del) = ex_gco.serialized_compile_options_deleter {
                del(ex_gco.serialized_compile_options.cast());
            }
            if let Some(del) = le_da.serialized_device_assignment_deleter {
                del(le_da.serialized_device_assignment.cast());
            }

            let mut tp_attrs = crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args>(
                ),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr,
                attributes: ptr::null(),
                num_attributes: 0,
            };
            assert!(
                PJRT_TopologyDescription_Attributes(ptr::addr_of_mut!(tp_attrs).cast()).is_null()
            );
            assert_eq!(tp_attrs.num_attributes, 3);

            let mut tp_fp = crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args>(
                ),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr,
                fingerprint: 0,
            };
            assert!(
                PJRT_TopologyDescription_Fingerprint(ptr::addr_of_mut!(tp_fp).cast()).is_null()
            );
            assert_ne!(tp_fp.fingerprint, 0);

            let mut tp_canon =
                crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    topology: (*client).topology_ptr,
                    dims: ptr::null(),
                    num_dims: 0,
                    element_type: 11,
                    memory_space_kind: 0,
                    layout: ptr::null_mut(),
                    reserved: 0,
                    reserved1: 0,
                    reserved2: 0,
                    reserved3: 0,
                };
            assert!(
                PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace(
                    ptr::addr_of_mut!(tp_canon).cast()
                )
                .is_null()
            );

            let mut tp_kinds =
                crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    topology: (*client).topology_ptr,
                    kind_ids: ptr::null(),
                    num_kind_ids: 0,
                };
            assert!(
                PJRT_TopologyDescription_GetMemorySpaceKindIds(ptr::addr_of_mut!(tp_kinds).cast())
                    .is_null()
            );
            assert_eq!(tp_kinds.num_kind_ids, 4);

            let mut tp_create = crate::pjrt_types::PJRT_TopologyDescription_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(),
                extension_start: ptr::null_mut(),
                topology_name: c"tpu".as_ptr(),
                topology_name_size: 3,
                create_options: ptr::null(),
                num_options: 0,
                topology: ptr::null_mut(),
            };
            assert!(PJRT_TopologyDescription_Create(ptr::addr_of_mut!(tp_create).cast()).is_null());
            let mut tp_del_standalone = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                topology: tp_create.topology,
            };
            assert!(
                PJRT_TopologyDescription_Destroy(ptr::addr_of_mut!(tp_del_standalone).cast())
                    .is_null()
            );

            let shape_spec_0 = crate::pjrt_types::PJRT_ShapeSpec {
                struct_size: size_of::<crate::pjrt_types::PJRT_ShapeSpec>(),
                extension_start: ptr::null_mut(),
                dims: dims_arr.as_ptr(),
                num_dims: 1,
                element_type: 11,
                pad: 0,
            };
            let shape_spec_empty = crate::pjrt_types::PJRT_ShapeSpec {
                struct_size: size_of::<crate::pjrt_types::PJRT_ShapeSpec>(),
                extension_start: ptr::null_mut(),
                dims: ptr::null(),
                num_dims: 0,
                element_type: 11,
                pad: 0,
            };
            let neg_dim_spec = crate::pjrt_types::PJRT_ShapeSpec {
                struct_size: size_of::<crate::pjrt_types::PJRT_ShapeSpec>(),
                extension_start: ptr::null_mut(),
                dims: neg_dim_val.as_ptr(),
                num_dims: 1,
                element_type: 11,
                pad: 0,
            };
            let specs_arr = [shape_spec_0, shape_spec_empty, neg_dim_spec];
            let mut h2d_mgr_args =
                crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    client,
                    shape_specs: specs_arr.as_ptr(),
                    num_shape_specs: specs_arr.len(),
                    device_layouts: ptr::null_mut(),
                    num_device_layouts: 0,
                    memory: (*client)
                        .memory_ptrs
                        .first()
                        .copied()
                        .unwrap_or(ptr::null_mut()),
                    transfer_manager: ptr::null_mut(),
                };
            assert!(
                PJRT_Client_CreateBuffersForAsyncHostToDevice(
                    ptr::addr_of_mut!(h2d_mgr_args).cast()
                )
                .is_null()
            );
            assert!(!h2d_mgr_args.transfer_manager.is_null());
            let mut h2d_mgr_del =
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    transfer_manager: h2d_mgr_args.transfer_manager,
                };
            assert!(
                PJRT_AsyncHostToDeviceTransferManager_Destroy(
                    ptr::addr_of_mut!(h2d_mgr_del).cast()
                )
                .is_null()
            );
            h2d_mgr_args.memory = ptr::null_mut();
            h2d_mgr_args.shape_specs = ptr::null();
            h2d_mgr_args.num_shape_specs = 0;
            h2d_mgr_args.transfer_manager = ptr::null_mut();
            assert!(
                PJRT_Client_CreateBuffersForAsyncHostToDevice(
                    ptr::addr_of_mut!(h2d_mgr_args).cast()
                )
                .is_null()
            );
            h2d_mgr_del.transfer_manager = h2d_mgr_args.transfer_manager;
            assert!(
                PJRT_AsyncHostToDeviceTransferManager_Destroy(
                    ptr::addr_of_mut!(h2d_mgr_del).cast()
                )
                .is_null()
            );

            for tm_inner in libtpu_core::pjrt::PjrtCoreAsyncTransferManager::create(
                DeviceId(0),
                libtpu_core::types::MemorySpaceId(0),
                vec![(*buf_in).inner.shape.clone()],
                &(*client).inner.hal,
            ) {
                let sample_tm = Box::into_raw(Box::new(
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager {
                        inner: tm_inner,
                        device_ptr: look_args.device,
                        memory_ptr: (*look_args.device).default_memory_ptr,
                    },
                ));
                let mut tm_bc =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_count: 0,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_BufferCount(
                        ptr::addr_of_mut!(tm_bc).cast()
                    )
                    .is_null()
                );
                assert_eq!(tm_bc.buffer_count, 1);

                let mut tm_bs =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        buffer_size: 0,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_BufferSize(
                        ptr::addr_of_mut!(tm_bs).cast()
                    )
                    .is_null()
                );
                assert!(tm_bs.buffer_size >= 16);
                tm_bs.buffer_index = 99;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_BufferSize(
                        ptr::addr_of_mut!(tm_bs).cast()
                    )
                    .is_null()
                );

                let mut tm_dev =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        device_out: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_Device(ptr::addr_of_mut!(tm_dev).cast())
                        .is_null()
                );

                let mut tm_tx =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        data: raw_host_bytes.as_ptr().cast(),
                        offset: 0,
                        transfer_size: 16,
                        is_last_transfer: true,
                        pad1: [0; 7],
                        done_with_h2d_transfer: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferData(
                        ptr::addr_of_mut!(tm_tx).cast()
                    )
                    .is_null()
                );
                tm_tx.data = ptr::null();
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferData(
                        ptr::addr_of_mut!(tm_tx).cast()
                    )
                    .is_null()
                );

                let mut tm_ret =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        buffer_out: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
                        ptr::addr_of_mut!(tm_ret).cast()
                    )
                    .is_null()
                );
                assert!(!tm_ret.buffer_out.is_null());
                libtpu_rs_buffer_destroy(tm_ret.buffer_out);
                tm_ret.buffer_index = 99;
                tm_ret.buffer_out = ptr::null_mut();
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
                        ptr::addr_of_mut!(tm_ret).cast()
                    )
                    .is_null()
                );
                libtpu_rs_buffer_destroy(tm_ret.buffer_out);

                let mut tm_sbe =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad0: 0,
                        error_code: 3,
                        pad1: 0,
                        error_message: c"transfer failed".as_ptr(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
                        ptr::addr_of_mut!(tm_sbe).cast()
                    )
                    .is_null()
                );

                let sample_meta = crate::pjrt_types::PJRT_TransferMetadata {
                    struct_size: size_of::<crate::pjrt_types::PJRT_TransferMetadata>(),
                    extension_start: ptr::null_mut(),
                    flags: 7,
                };
                let mut tm_meta =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        transfer_metadata: ptr::null(),
                        num_metadata: 0,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
                        ptr::addr_of_mut!(tm_meta).cast()
                    )
                    .is_null()
                );
                tm_meta.transfer_metadata = ptr::addr_of!(sample_meta);
                tm_meta.num_metadata = 1;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
                        ptr::addr_of_mut!(tm_meta).cast()
                    )
                    .is_null()
                );

                let mut tm_lit =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        data: ptr::null(),
                        shape_dims: ptr::null(),
                        shape_num_dims: 0,
                        shape_element_type: 11,
                        pad1: 0,
                        shape_layout: ptr::null(),
                        done_with_h2d_transfer: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
                        ptr::addr_of_mut!(tm_lit).cast()
                    )
                    .is_null()
                );
                tm_lit.data = raw_host_bytes.as_ptr().cast();
                tm_lit.shape_dims = dims_arr.as_ptr();
                tm_lit.shape_num_dims = 1;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
                        ptr::addr_of_mut!(tm_lit).cast()
                    )
                    .is_null()
                );
                tm_lit.shape_dims = ptr::null();
                tm_lit.shape_num_dims = 0;
                tm_lit.shape_element_type = 999;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
                        ptr::addr_of_mut!(tm_lit).cast()
                    )
                    .is_null()
                );

                let mut tm_del =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_Destroy(ptr::addr_of_mut!(tm_del).cast())
                        .is_null()
                );
            }

            let mut cl_topo = crate::pjrt_types::PJRT_Client_TopologyDescription_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_TopologyDescription_Args>(),
                extension_start: ptr::null_mut(),
                client,
                topology: ptr::null_mut(),
            };
            assert!(PJRT_Client_TopologyDescription(ptr::addr_of_mut!(cl_topo).cast()).is_null());
            assert_eq!(cl_topo.topology, (*client).topology_ptr);

            let mut cl_dma_map = crate::pjrt_types::PJRT_Client_DmaMap_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_DmaMap_Args>(),
                extension_start: ptr::null_mut(),
                client,
                data: 0x2000 as *const std::ffi::c_void,
                size: 4096,
            };
            assert!(PJRT_Client_DmaMap(ptr::addr_of_mut!(cl_dma_map).cast()).is_null());
            let mut cl_dma_unmap = crate::pjrt_types::PJRT_Client_DmaUnmap_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_DmaUnmap_Args>(),
                extension_start: ptr::null_mut(),
                client,
                data: 0x2000 as *const std::ffi::c_void,
            };
            assert!(PJRT_Client_DmaUnmap(ptr::addr_of_mut!(cl_dma_unmap).cast()).is_null());

            let mut cl_fulfill = crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client,
                buffer: buf_in,
                status_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
                fulfill_alias_buffer_cb: ptr::null_mut(),
            };
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(cl_fulfill).cast()).is_null());

            let mut cl_up_proc = crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args>(
                ),
                extension_start: ptr::null_mut(),
                client,
                processes: ptr::null(),
                num_processes: 0,
            };
            assert!(
                PJRT_Client_UpdateGlobalProcessInfo(ptr::addr_of_mut!(cl_up_proc).cast()).is_null()
            );

            let dev0_inputs: [*const PJRT_Buffer; 2] = [buf_in, buf_in2];
            let dev0_inputs_ptr: *const *const PJRT_Buffer = dev0_inputs.as_ptr();
            let mut out_buf_ptr: *mut PJRT_Buffer = ptr::null_mut();
            let dev0_outputs_ptr: *mut *mut PJRT_Buffer = ptr::addr_of_mut!(out_buf_ptr);
            let mut exec_vtable_args = crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args>(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
                options: ptr::null(),
                argument_lists: ptr::addr_of!(dev0_inputs_ptr),
                num_devices: 1,
                num_args: 2,
                output_lists: ptr::addr_of!(dev0_outputs_ptr),
                device_complete_events: ptr::null_mut(),
                execute_device: ptr::null_mut(),
            };
            assert!(
                PJRT_LoadedExecutable_Execute(ptr::addr_of_mut!(exec_vtable_args).cast()).is_null()
            );
            assert!(!out_buf_ptr.is_null());
            exec_vtable_args.num_devices = 99;
            drop(Box::from_raw(PJRT_LoadedExecutable_Execute(
                ptr::addr_of_mut!(exec_vtable_args).cast(),
            )));
            exec_vtable_args.num_devices = 1;

            let mut d2h_size_args = crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                src: out_buf_ptr,
                host_layout: ptr::null(),
                dst: ptr::null_mut(),
                dst_size: 0,
                event: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_ToHostBuffer(ptr::addr_of_mut!(d2h_size_args).cast()).is_null());
            assert_eq!(d2h_size_args.dst_size, 16);

            let mut out_floats = [0.0_f32; 4];
            d2h_size_args.dst = out_floats.as_mut_ptr().cast();
            assert!(PJRT_Buffer_ToHostBuffer(ptr::addr_of_mut!(d2h_size_args).cast()).is_null());
            assert_eq!(out_floats, [2.0, 4.0, 6.0, 8.0]);

            // Exercise newly upgraded PJRT Event, Client_Load, Buffer_Bitcast, DefaultDeviceAssignment, LookupAddressableDevice, Device_GetAttributes, and Error_ForEachPayload slots
            drop(Box::from_raw(PJRT_Event_Create(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Set(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_IsReady(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Error(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Await(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_OnReady(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Destroy(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Client_LookupAddressableDevice(
                ptr::null_mut(),
            )));
            drop(Box::from_raw(PJRT_Client_DefaultDeviceAssignment(
                ptr::null_mut(),
            )));
            drop(Box::from_raw(PJRT_Device_GetAttributes(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Client_Load(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_Bitcast(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Error_ForEachPayload(ptr::null_mut())));

            let mut ev_create_args = crate::pjrt_types::PJRT_Event_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Create_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!(PJRT_Event_Create(ptr::addr_of_mut!(ev_create_args).cast()).is_null());
            let created_ev = ev_create_args.event;
            assert!(!created_ev.is_null());

            let mut ev_ready_args = crate::pjrt_types::PJRT_Event_IsReady_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_IsReady_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                is_ready: true,
                pad: [0; 7],
            };
            assert!(PJRT_Event_IsReady(ptr::addr_of_mut!(ev_ready_args).cast()).is_null());
            assert!(!ev_ready_args.is_ready);

            let mut ev_set_args = crate::pjrt_types::PJRT_Event_Set_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Set_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                error_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
            };
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());
            assert!(PJRT_Event_IsReady(ptr::addr_of_mut!(ev_ready_args).cast()).is_null());
            assert!(ev_ready_args.is_ready);

            let mut ev_err_args = crate::pjrt_types::PJRT_Event_Error_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Error_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Error(ptr::addr_of_mut!(ev_err_args).cast()).is_null());
            let mut ev_await_args = crate::pjrt_types::PJRT_Event_Await_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Await_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Await(ptr::addr_of_mut!(ev_await_args).cast()).is_null());

            // Test error event setting (both null and non-null error_message)
            ev_set_args.error_code = 3;
            ev_set_args.error_message = ptr::null();
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());
            drop(Box::from_raw(PJRT_Event_Error(
                ptr::addr_of_mut!(ev_err_args).cast(),
            )));
            drop(Box::from_raw(PJRT_Event_Await(
                ptr::addr_of_mut!(ev_await_args).cast(),
            )));
            ev_set_args.error_message = c"custom event failure".as_ptr();
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());

            unsafe extern "C" fn test_on_ready_cb(
                err: *mut PJRT_Error,
                user_arg: *mut std::ffi::c_void,
            ) {
                if !err.is_null() {
                    // SAFETY: `err` was boxed by `box_pjrt_error`.
                    unsafe {
                        drop(Box::from_raw(err));
                    }
                }
                // SAFETY: `user_arg` points to a mutable `u32` counter in this test.
                unsafe {
                    *user_arg.cast::<u32>() += 1;
                }
            }
            let mut cb_hits: u32 = 0;
            let mut on_ready_args = crate::pjrt_types::PJRT_Event_OnReady_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_OnReady_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                callback: Some(test_on_ready_cb),
                user_arg: ptr::addr_of_mut!(cb_hits).cast(),
            };
            assert!(PJRT_Event_OnReady(ptr::addr_of_mut!(on_ready_args).cast()).is_null());
            assert_eq!(cb_hits, 1);
            on_ready_args.event = ptr::null_mut();
            assert!(PJRT_Event_OnReady(ptr::addr_of_mut!(on_ready_args).cast()).is_null());
            assert_eq!(cb_hits, 2);

            let mut ev_create2 = crate::pjrt_types::PJRT_Event_Create_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Create_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!(PJRT_Event_Create(ptr::addr_of_mut!(ev_create2).cast()).is_null());
            on_ready_args.event = ev_create2.event;
            assert!(PJRT_Event_OnReady(ptr::addr_of_mut!(on_ready_args).cast()).is_null());
            assert_eq!(cb_hits, 2);
            ev_set_args.event = ev_create2.event;
            ev_set_args.error_code = 0;
            ev_set_args.error_message = ptr::null();
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());
            assert_eq!(cb_hits, 3);
            let mut ev_del2 = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ev_create2.event,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del2).cast()).is_null());

            let mut ev_del_args = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del_args).cast()).is_null());

            let mut look_addr_args = PJRT_Client_LookupDevice_Args {
                struct_size: size_of::<PJRT_Client_LookupDevice_Args>(),
                extension_start: ptr::null_mut(),
                client,
                id: 0,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert!(
                PJRT_Client_LookupAddressableDevice(ptr::addr_of_mut!(look_addr_args).cast())
                    .is_null()
            );
            assert!(!look_addr_args.device.is_null());
            look_addr_args.id = 999;
            assert!(
                PJRT_Client_LookupAddressableDevice(ptr::addr_of_mut!(look_addr_args).cast())
                    .is_null()
            );

            let mut assign_buf = [-1_i32; 4];
            let mut dda_args = crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args>(
                ),
                extension_start: ptr::null_mut(),
                client,
                num_replicas: 2,
                num_partitions: 2,
                default_assignment_size: assign_buf.len(),
                default_assignment: assign_buf.as_mut_ptr(),
            };
            assert!(
                PJRT_Client_DefaultDeviceAssignment(ptr::addr_of_mut!(dda_args).cast()).is_null()
            );
            assert_eq!(assign_buf, [0, 1, 2, 3]);

            let mut dev_attr_args = crate::pjrt_types::PJRT_Device_GetAttributes_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                device: look_args.device,
                attributes: ptr::null(),
                num_attributes: 0,
                reserved0: 0,
                reserved1: 0,
            };
            drop(Box::from_raw(PJRT_Device_GetAttributes(
                ptr::addr_of_mut!(dev_attr_args).cast(),
            )));
            dev_attr_args.struct_size =
                size_of::<crate::pjrt_types::PJRT_Device_GetAttributes_Args>();
            assert!(PJRT_Device_GetAttributes(ptr::addr_of_mut!(dev_attr_args).cast()).is_null());
            assert_eq!(dev_attr_args.num_attributes, 3);

            let mut load_args = crate::pjrt_types::PJRT_Client_Load_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_Load_Args>(),
                extension_start: ptr::null_mut(),
                client,
                executable: (*comp_args.executable).executable_ptr,
                compile_options: ptr::null(),
                compile_options_size: 0,
                loaded_executable: ptr::null_mut(),
            };
            assert!(PJRT_Client_Load(ptr::addr_of_mut!(load_args).cast()).is_null());
            assert!(!load_args.loaded_executable.is_null());
            let mut del_loaded2 = crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: load_args.loaded_executable,
            };
            assert!(PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(del_loaded2).cast()).is_null());

            let bc_dims = [2_i64, 2];
            let mut bc_args = crate::pjrt_types::PJRT_Buffer_Bitcast_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Bitcast_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                element_type: 11,
                pad: 0,
                dims: bc_dims.as_ptr(),
                num_dims: bc_dims.len(),
                device_layout: ptr::null(),
                out_buffer: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_Bitcast(ptr::addr_of_mut!(bc_args).cast()).is_null());
            assert!(!bc_args.out_buffer.is_null());
            libtpu_rs_buffer_destroy(bc_args.out_buffer);

            bc_args.dims = ptr::null();
            bc_args.num_dims = 0;
            bc_args.element_type = 999;
            bc_args.out_buffer = ptr::null_mut();
            assert!(PJRT_Buffer_Bitcast(ptr::addr_of_mut!(bc_args).cast()).is_null());
            libtpu_rs_buffer_destroy(bc_args.out_buffer);

            let bad_bc_dims = [-1_i64];
            bc_args.dims = bad_bc_dims.as_ptr();
            bc_args.num_dims = 1;
            bc_args.out_buffer = ptr::null_mut();
            assert!(PJRT_Buffer_Bitcast(ptr::addr_of_mut!(bc_args).cast()).is_null());
            assert!(bc_args.out_buffer.is_null());

            unsafe extern "C" fn test_payload_visitor(
                _k: *const c_char,
                _k_len: usize,
                _v: *const c_char,
                _v_len: usize,
                user_arg: *mut std::ffi::c_void,
            ) {
                // SAFETY: `user_arg` points to a mutable `u32` counter in this test.
                unsafe {
                    *user_arg.cast::<u32>() += 1;
                }
            }
            let sample_err = box_pjrt_error(LibTpuError::NullPointer { parameter: "test" });
            let mut visited_count: u32 = 0;
            let mut fep_args = crate::pjrt_types::PJRT_Error_ForEachPayload_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Error_ForEachPayload_Args>(),
                extension_start: ptr::null_mut(),
                error: sample_err,
                visitor: Some(test_payload_visitor),
                user_arg: ptr::addr_of_mut!(visited_count).cast(),
            };
            assert!(PJRT_Error_ForEachPayload(ptr::addr_of_mut!(fep_args).cast()).is_null());
            assert!(visited_count >= 2);
            drop(Box::from_raw(sample_err));

            // Delete `out_buf_ptr` via PJRT_Buffer_Delete and verify `PJRT_Buffer_IsDeleted`, `PJRT_Buffer_ToHostBuffer`, and `PJRT_LoadedExecutable_Execute`
            let mut b_del_call = crate::pjrt_types::PJRT_Buffer_Delete_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Delete_Args>(),
                extension_start: ptr::null_mut(),
                buffer: out_buf_ptr,
            };
            assert!(PJRT_Buffer_Delete(ptr::addr_of_mut!(b_del_call).cast()).is_null());
            let mut b_is_del_call = crate::pjrt_types::PJRT_Buffer_IsDeleted_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_IsDeleted_Args>(),
                extension_start: ptr::null_mut(),
                buffer: out_buf_ptr,
                is_deleted: false,
                pad: [0; 7],
            };
            assert!(PJRT_Buffer_IsDeleted(ptr::addr_of_mut!(b_is_del_call).cast()).is_null());
            assert!(b_is_del_call.is_deleted);

            drop(Box::from_raw(PJRT_Buffer_ToHostBuffer(
                ptr::addr_of_mut!(d2h_size_args).cast(),
            )));
            let deleted_inputs: [*const PJRT_Buffer; 2] = [out_buf_ptr, buf_in2];
            let deleted_inputs_ptr: *const *const PJRT_Buffer = deleted_inputs.as_ptr();
            exec_vtable_args.argument_lists = ptr::addr_of!(deleted_inputs_ptr);
            drop(Box::from_raw(PJRT_LoadedExecutable_Execute(
                ptr::addr_of_mut!(exec_vtable_args).cast(),
            )));

            let mut le_del_call = crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args>(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
            };
            assert!(PJRT_LoadedExecutable_Delete(ptr::addr_of_mut!(le_del_call).cast()).is_null());
            let mut le_is_del_call = crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args>(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
                is_deleted: false,
                pad: [0; 7],
            };
            assert!(
                PJRT_LoadedExecutable_IsDeleted(ptr::addr_of_mut!(le_is_del_call).cast()).is_null()
            );
            assert!(le_is_del_call.is_deleted);

            for b_del in [buf_in, buf_in2, out_buf_ptr] {
                let mut del_b_args = crate::pjrt_types::PJRT_Buffer_Destroy_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    buffer: b_del,
                };
                assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_b_args).cast()).is_null());
            }

            let mut del_exec_args = crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
            };
            assert!(
                PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(del_exec_args).cast()).is_null()
            );

            let opts = b"opts";
            comp_args.compile_options = opts.as_ptr().cast();
            comp_args.compile_options_size = opts.len();
            prog.format = b"bad".as_ptr().cast();
            prog.format_size = 3;
            comp_args.program = ptr::addr_of!(prog);
            drop(Box::from_raw(PJRT_Client_Compile(
                ptr::addr_of_mut!(comp_args).cast(),
            )));

            drop(Box::from_raw(PJRT_Client_Destroy(ptr::null_mut())));
            let mut destroy_client_args = PJRT_Client_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            assert_ne!(format!("{destroy_client_args:?}"), "");
            drop(Box::from_raw(PJRT_Client_Destroy(
                ptr::addr_of_mut!(destroy_client_args).cast(),
            )));
            destroy_client_args.client = client;
            assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(destroy_client_args).cast()).is_null());
        }
    }

    #[test]
    fn test_sdk_legacy_and_libtpu_rs_entrypoints() {
        let sdk_ptr = GetLibtpuSdkApi();
        assert!(!sdk_ptr.is_null());
        assert_eq!(PyInit_sdk(), sdk_ptr);
        // SAFETY: Testing SDK, legacy TfTpu/TpuTopology/TpuProfiler/TpuEmbeddingEngine, and libtpu_rs_* functions.
        unsafe {
            let sdk = *sdk_ptr;
            assert_eq!(sdk.version_header, 0x0000_0001_0000_0000);

            // Exercise all 17 LibtpuSdk_Api entrypoints (both null/error and success paths)
            LibtpuSdk_Error_GetMessage(ptr::null_mut());
            let mut sdk_msg_args = LibtpuSdk_Error_GetMessage_Args {
                error: ptr::null(),
                message: ptr::null(),
                message_size: 0,
            };
            assert_ne!(format!("{sdk_msg_args:?}"), "");
            LibtpuSdk_Error_GetMessage(ptr::addr_of_mut!(sdk_msg_args).cast());
            let sdk_err = LibtpuSdk_Error_GetCode(ptr::null_mut());
            assert!(!sdk_err.is_null());
            assert_ne!(format!("{:?}", *sdk_err), "");
            sdk_msg_args.error = sdk_err;
            LibtpuSdk_Error_GetMessage(ptr::addr_of_mut!(sdk_msg_args).cast());
            assert!(sdk_msg_args.message_size > 0);

            let mut sdk_code_args = LibtpuSdk_Error_GetCode_Args {
                error: ptr::null(),
                code: 0,
                pad: 0,
            };
            assert_ne!(format!("{sdk_code_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_Error_GetCode(
                ptr::addr_of_mut!(sdk_code_args).cast(),
            )));
            sdk_code_args.error = sdk_err;
            assert!(LibtpuSdk_Error_GetCode(ptr::addr_of_mut!(sdk_code_args).cast()).is_null());
            assert_eq!(sdk_code_args.code, 3);

            LibtpuSdk_Error_Destroy(ptr::null_mut());
            let mut sdk_del_args = LibtpuSdk_Error_Destroy_Args {
                error: ptr::null_mut(),
            };
            assert_ne!(format!("{sdk_del_args:?}"), "");
            LibtpuSdk_Error_Destroy(ptr::addr_of_mut!(sdk_del_args).cast());
            sdk_del_args.error = sdk_err;
            LibtpuSdk_Error_Destroy(ptr::addr_of_mut!(sdk_del_args).cast());

            drop(Box::from_raw(LibtpuSdk_Client_Create(ptr::null_mut())));
            let mut sdk_cli_args = LibtpuSdk_Client_Args {
                client: ptr::null_mut(),
            };
            assert_ne!(format!("{sdk_cli_args:?}"), "");
            assert!(LibtpuSdk_Client_Create(ptr::addr_of_mut!(sdk_cli_args).cast()).is_null());
            let sdk_client = sdk_cli_args.client;
            assert!(!sdk_client.is_null());

            drop(Box::from_raw(LibtpuSdk_GetChipCoordinates(ptr::null_mut())));
            let mut coords_args = LibtpuSdk_GetChipCoordinates_Args {
                client: ptr::null_mut(),
                coords: ptr::null(),
                num_coords: 0,
            };
            assert_ne!(format!("{coords_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetChipCoordinates(
                ptr::addr_of_mut!(coords_args).cast(),
            )));
            coords_args.client = sdk_client;
            assert!(LibtpuSdk_GetChipCoordinates(ptr::addr_of_mut!(coords_args).cast()).is_null());
            assert_eq!(coords_args.num_coords, 4);
            assert_ne!(format!("{:?}", *coords_args.coords), "");

            drop(Box::from_raw(LibtpuSdk_GetHostName(ptr::null_mut())));
            let mut host_args = LibtpuSdk_GetHostName_Args {
                coord: ptr::null(),
                host_name: ptr::null(),
                host_name_size: 0,
            };
            assert_ne!(format!("{host_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetHostName(
                ptr::addr_of_mut!(host_args).cast(),
            )));
            host_args.coord = coords_args.coords;
            assert!(LibtpuSdk_GetHostName(ptr::addr_of_mut!(host_args).cast()).is_null());
            assert!(host_args.host_name_size > 0);

            drop(Box::from_raw(LibtpuSdk_GetChipIndex(ptr::null_mut())));
            let mut cidx_args = LibtpuSdk_GetChipIndex_Args {
                coord: ptr::null(),
                chip_index: -1,
                pad: 0,
            };
            assert_ne!(format!("{cidx_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetChipIndex(
                ptr::addr_of_mut!(cidx_args).cast(),
            )));
            cidx_args.coord = coords_args.coords;
            assert!(LibtpuSdk_GetChipIndex(ptr::addr_of_mut!(cidx_args).cast()).is_null());
            assert_eq!(cidx_args.chip_index, 0);

            drop(Box::from_raw(LibtpuSdk_GetCartesianCoordinates(
                ptr::null_mut(),
            )));
            let mut cart_args = LibtpuSdk_GetCartesianCoordinates_Args {
                coord: ptr::null(),
                coordinates: ptr::null(),
                num_coordinates: 0,
            };
            assert_ne!(format!("{cart_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetCartesianCoordinates(
                ptr::addr_of_mut!(cart_args).cast(),
            )));
            cart_args.coord = coords_args.coords;
            assert!(
                LibtpuSdk_GetCartesianCoordinates(ptr::addr_of_mut!(cart_args).cast()).is_null()
            );
            assert_eq!(cart_args.num_coordinates, 3);

            drop(Box::from_raw(LibtpuSdk_GetMetric(ptr::null_mut())));
            let mut met_args = LibtpuSdk_GetMetric_Args {
                client: ptr::null_mut(),
                metric_name: ptr::null(),
                metric: ptr::null_mut(),
            };
            assert_ne!(format!("{met_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetMetric(
                ptr::addr_of_mut!(met_args).cast(),
            )));
            met_args.client = sdk_client;
            met_args.metric_name = c"unknown_metric".as_ptr();
            drop(Box::from_raw(LibtpuSdk_GetMetric(
                ptr::addr_of_mut!(met_args).cast(),
            )));
            met_args.metric_name = c"tpu.runtime.hbm.usage_bytes".as_ptr();
            assert!(LibtpuSdk_GetMetric(ptr::addr_of_mut!(met_args).cast()).is_null());
            assert!(!met_args.metric.is_null());
            assert_ne!(format!("{:?}", *met_args.metric), "");

            drop(Box::from_raw(LibtpuSdk_GetMetricDescription(
                ptr::null_mut(),
            )));
            let mut mdesc_args = LibtpuSdk_GetMetricDescription_Args {
                metric: ptr::null(),
                description: ptr::null(),
                description_size: 0,
            };
            assert_ne!(format!("{mdesc_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetMetricDescription(
                ptr::addr_of_mut!(mdesc_args).cast(),
            )));
            mdesc_args.metric = met_args.metric;
            assert!(LibtpuSdk_GetMetricDescription(ptr::addr_of_mut!(mdesc_args).cast()).is_null());
            assert!(mdesc_args.description_size > 0);

            drop(Box::from_raw(LibtpuSdk_GetMetricValues(ptr::null_mut())));
            let mut mvals_args = LibtpuSdk_GetMetricValues_Args {
                metric: ptr::null(),
                values: ptr::null(),
                num_values: 0,
            };
            assert_ne!(format!("{mvals_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetMetricValues(
                ptr::addr_of_mut!(mvals_args).cast(),
            )));
            mvals_args.metric = met_args.metric;
            assert!(LibtpuSdk_GetMetricValues(ptr::addr_of_mut!(mvals_args).cast()).is_null());
            assert_eq!(mvals_args.num_values, 4);
            drop(Box::from_raw(met_args.metric));

            drop(Box::from_raw(LibtpuSdk_GetRuntimeStatus(ptr::null_mut())));
            let mut rstat_args = LibtpuSdk_GetRuntimeStatus_Args {
                client: ptr::null_mut(),
                include_hlo_info: true,
                pad: [0; 7],
                status: ptr::null_mut(),
            };
            assert_ne!(format!("{rstat_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetRuntimeStatus(
                ptr::addr_of_mut!(rstat_args).cast(),
            )));
            rstat_args.client = sdk_client;
            assert!(LibtpuSdk_GetRuntimeStatus(ptr::addr_of_mut!(rstat_args).cast()).is_null());
            assert!(!rstat_args.status.is_null());
            assert_ne!(format!("{:?}", *rstat_args.status), "");

            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
                ptr::null_mut(),
            )));
            let mut sum_args = LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args {
                status: ptr::null(),
                summary: ptr::null(),
                summary_size: 0,
            };
            assert_ne!(format!("{sum_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
                ptr::addr_of_mut!(sum_args).cast(),
            )));
            sum_args.status = rstat_args.status;
            assert!(
                LibtpuSdk_RuntimeStatus_GetCoreStateSummary(ptr::addr_of_mut!(sum_args).cast())
                    .is_null()
            );
            assert!(sum_args.summary_size > 0);

            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_Destroy(
                ptr::null_mut(),
            )));
            let mut rdel_args = LibtpuSdk_RuntimeStatus_Destroy_Args {
                status: ptr::null_mut(),
            };
            assert_ne!(format!("{rdel_args:?}"), "");
            assert!(LibtpuSdk_RuntimeStatus_Destroy(ptr::addr_of_mut!(rdel_args).cast()).is_null());
            rdel_args.status = rstat_args.status;
            assert!(LibtpuSdk_RuntimeStatus_Destroy(ptr::addr_of_mut!(rdel_args).cast()).is_null());

            drop(Box::from_raw(LibtpuSdk_RegisterDebugCallback(
                ptr::null_mut(),
            )));
            let mut reg_cb_args = LibtpuSdk_RegisterDebugCallback_Args {
                client: ptr::null_mut(),
                callback: ptr::null(),
                user_data: ptr::null_mut(),
                callback_id: 0,
            };
            assert_ne!(format!("{reg_cb_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_RegisterDebugCallback(
                ptr::addr_of_mut!(reg_cb_args).cast(),
            )));
            reg_cb_args.client = sdk_client;
            assert!(
                LibtpuSdk_RegisterDebugCallback(ptr::addr_of_mut!(reg_cb_args).cast()).is_null()
            );
            assert!(reg_cb_args.callback_id > 0);
            assert_eq!(LibtpuSdk_TriggerDebugCallbacks(ptr::null()), 0);
            assert_eq!(LibtpuSdk_TriggerDebugCallbacks(sdk_client), 1);

            unsafe extern "C" fn rust_debug_cb(user_data: *mut std::ffi::c_void) {
                // SAFETY: `user_data` points to `cb_counter`.
                unsafe {
                    *user_data.cast::<u32>() += 1;
                }
            }
            let mut cb_counter = 0_u32;
            let mut real_cb_args = LibtpuSdk_RegisterDebugCallback_Args {
                client: sdk_client,
                callback: (rust_debug_cb as *const () as usize) as *const std::ffi::c_void,
                user_data: ptr::addr_of_mut!(cb_counter).cast(),
                callback_id: 0,
            };
            assert!(
                LibtpuSdk_RegisterDebugCallback(ptr::addr_of_mut!(real_cb_args).cast()).is_null()
            );
            assert_eq!(LibtpuSdk_TriggerDebugCallbacks(sdk_client), 2);
            assert_eq!(cb_counter, 1);
            let mut unreg_real = LibtpuSdk_UnregisterDebugCallback_Args {
                client: sdk_client,
                callback_id: real_cb_args.callback_id,
            };
            assert!(
                LibtpuSdk_UnregisterDebugCallback(ptr::addr_of_mut!(unreg_real).cast()).is_null()
            );

            drop(Box::from_raw(LibtpuSdk_UnregisterDebugCallback(
                ptr::null_mut(),
            )));
            let mut unreg_cb_args = LibtpuSdk_UnregisterDebugCallback_Args {
                client: ptr::null_mut(),
                callback_id: reg_cb_args.callback_id,
            };
            assert_ne!(format!("{unreg_cb_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_UnregisterDebugCallback(
                ptr::addr_of_mut!(unreg_cb_args).cast(),
            )));
            unreg_cb_args.client = sdk_client;
            assert!(
                LibtpuSdk_UnregisterDebugCallback(ptr::addr_of_mut!(unreg_cb_args).cast())
                    .is_null()
            );
            drop(Box::from_raw(LibtpuSdk_UnregisterDebugCallback(
                ptr::addr_of_mut!(unreg_cb_args).cast(),
            )));

            drop(Box::from_raw(LibtpuSdk_Client_Destroy(ptr::null_mut())));
            let mut null_cli_args = LibtpuSdk_Client_Args {
                client: ptr::null_mut(),
            };
            drop(Box::from_raw(LibtpuSdk_Client_Destroy(
                ptr::addr_of_mut!(null_cli_args).cast(),
            )));
            assert!(LibtpuSdk_Client_Destroy(ptr::addr_of_mut!(sdk_cli_args).cast()).is_null());

            TfTpu_Initialize(false, 0, ptr::null());
            let cli_ptrs: [*const c_char; 2] = [c"--tpu_generation=v5e".as_ptr(), ptr::null()];
            TfTpu_Initialize(true, 2, cli_ptrs.as_ptr());
            TF_InitKernel();
            TFNPD_InitPlugin(ptr::null_mut(), ptr::null_mut());
            let mut tfnpd_params = TFNPD_PluginParams {
                struct_size: size_of::<TFNPD_PluginParams>(),
                priv_data: ptr::null_mut(),
                device_type: ptr::null(),
                compilation_device_name: ptr::null(),
                priority: 0,
                is_pluggable_device: false,
                use_pjrt_on_demand_compile: false,
                pad: 0,
            };
            assert_ne!(format!("{tfnpd_params:?}"), "");
            TFNPD_InitPlugin(ptr::addr_of_mut!(tfnpd_params), ptr::null_mut());
            assert_eq!(tfnpd_params.priority, 55);
            assert!(tfnpd_params.is_pluggable_device);

            assert_eq!(
                SetGlobalTPUArrayOp_DoWork(0, ptr::null(), ptr::null_mut()),
                0
            );
            assert_eq!(
                SetGlobalTPUArrayOp_DoWork(4, ptr::null(), ptr::null_mut()),
                -1
            );
            let topo_proto = libtpu_core::proto::TpuTopologyArgsProto {
                version: 5,
                num_hosts: 1,
                chips_per_host: 4,
                mesh_bounds: vec![2, 2, 1],
                platform_version: String::from("TPU v5 lite"),
                fingerprint: String::from("1"),
            }
            .encode();
            assert_eq!(
                SetGlobalTPUArrayOp_DoWork(
                    topo_proto.len(),
                    topo_proto.as_ptr().cast(),
                    ptr::null_mut()
                ),
                0
            );
            assert_eq!(
                SetGlobalTPUArrayOp_DoWork(1, c"\xff".as_ptr(), ptr::null_mut()),
                0
            );
            assert_eq!(
                InitializeCreateMeshGlobalState(0, ptr::null(), ptr::null_mut()),
                0
            );
            assert_eq!(
                ConfigureDistributedTpuOp_DoWork(
                    2,
                    ptr::null(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                -1
            );
            let cores_arr = [4_i32, 4_i32];
            let mut dist_cfg_len: usize = 0;
            let mut dist_cfg_ptr: *mut c_char = ptr::null_mut();
            assert_eq!(
                ConfigureDistributedTpuOp_DoWork(
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                0
            );
            assert_eq!(
                ConfigureDistributedTpuOp_DoWork(
                    cores_arr.len(),
                    cores_arr.as_ptr(),
                    ptr::addr_of_mut!(dist_cfg_len),
                    ptr::addr_of_mut!(dist_cfg_ptr),
                    ptr::null_mut(),
                ),
                0
            );
            assert!(dist_cfg_len > 0);
            TpuConfigurationApi_FreeCharArray(dist_cfg_ptr);

            assert_eq!(TfTpu_GetTpuPartitionedCallParams(ptr::null_mut()), -1);
            let mut pcall_params = TpuPartitionedCall_Params {
                input_shape_opt: false,
                group_tensors_for_packing: false,
                pad0: 0,
                minimum_input_tensors_packing: 0,
                minimum_output_tensors_packing: 0,
                enable_auto_xla_input_sharding: false,
                pad1: [0; 3],
                auto_xla_input_sharding_dim: -1,
                scoped_allocator_for_tpu_anned_call: false,
                pad2: [0; 3],
            };
            assert_ne!(format!("{pcall_params:?}"), "");
            assert_eq!(
                TfTpu_GetTpuPartitionedCallParams(ptr::addr_of_mut!(pcall_params)),
                0
            );
            assert!(pcall_params.input_shape_opt);
            assert!(pcall_params.enable_variable_deduplication());

            TpuConfigurationApi_FreeCharArray(ptr::null_mut());
            TpuConfigurationApi_FreeCharArray(to_c_string("abc").into_raw());

            assert_eq!(
                TpuTransferManager_GetInfeedLayout(ptr::null(), ptr::null_mut()),
                -1
            );
            let shape_dims = [8_i64, 128];
            let host_shape = XLA_Shape {
                element_type: 11,
                rank: 2,
                dimensions: shape_dims.as_ptr(),
                minor_to_major: [0; 8],
                tile_dims: [0; 2],
            };
            let mut infeed_shape = host_shape;
            assert_ne!(format!("{host_shape:?}"), "");
            assert_eq!(
                TpuTransferManager_GetInfeedLayout(
                    ptr::addr_of!(host_shape),
                    ptr::addr_of_mut!(infeed_shape),
                ),
                0
            );
            assert_eq!(infeed_shape.tile_dims, [8, 128]);
            assert_eq!(infeed_shape.minor_to_major[0], 1);

            let (mut max_ids, mut max_uniques) = (0_i64, 0_i64);
            assert_eq!(
                SparseCore_GetMaxIdsAndUniques(
                    ptr::null(),
                    32,
                    16,
                    ptr::addr_of_mut!(max_ids),
                    ptr::addr_of_mut!(max_uniques),
                    ptr::null_mut(),
                ),
                -1
            );
            assert_eq!(
                SparseCore_GetMaxIdsAndUniques(
                    c"table_0".as_ptr(),
                    32,
                    16,
                    ptr::addr_of_mut!(max_ids),
                    ptr::addr_of_mut!(max_uniques),
                    ptr::null_mut(),
                ),
                128
            );
            assert!(max_ids >= 128 && max_uniques >= 64);

            let st1 = TpuStatus_New();
            assert_ne!(format!("{:?}", *st1), "");
            assert!(TpuStatus_Ok(st1));
            assert_eq!(TpuStatus_Code(st1), 0);
            assert!(!TpuStatus_Message(st1).is_null());
            TpuStatus_Set(ptr::null_mut(), 3, ptr::null(), 0);
            TpuStatus_Set(st1, 3, ptr::null(), 0);
            TpuStatus_Set(st1, 3, c"err".as_ptr(), 3);
            assert!(!TpuStatus_Ok(st1));
            TpuStatus_Free(st1);
            TpuStatus_Free(ptr::null_mut());

            let st2 = TpuStatus_Create(0, ptr::null());
            TpuStatus_Free(st2);
            let st3 = TpuStatus_Create(5, c"not found".as_ptr());
            TpuStatus_Free(st3);
            assert!(TpuStatus_Message(ptr::null()).is_null());
            assert_eq!(TpuStatus_Code(ptr::null()), -1);
            assert!(!TpuStatus_Ok(ptr::null()));

            TfTpuOrdinalSelector_Create(ptr::null_mut(), 2);
            let mut sel: *mut TfTpuOrdinalSelector = ptr::null_mut();
            TfTpuOrdinalSelector_Create(ptr::addr_of_mut!(sel), 2);
            assert_ne!(format!("{:?}", *sel), "");
            let mut ord: i64 = -1;
            let mut seed_val: u64 = 10;
            assert_eq!(
                TfTpuOrdinalSelector_GetOrdinal(
                    ptr::null(),
                    ptr::null_mut(),
                    ptr::addr_of_mut!(ord)
                ),
                -1
            );
            assert_eq!(
                TfTpuOrdinalSelector_GetOrdinal(sel, ptr::null_mut(), ptr::addr_of_mut!(ord)),
                0
            );
            assert_eq!(
                TfTpuOrdinalSelector_GetOrdinal(
                    sel,
                    ptr::addr_of_mut!(seed_val),
                    ptr::addr_of_mut!(ord)
                ),
                1
            );
            assert_eq!(seed_val, 11);
            TfTpuOrdinalSelector_DequeueFromCoreSelector(ptr::null_mut(), 0, 1);
            TfTpuOrdinalSelector_DequeueFromCoreSelector(sel, -1, 1);
            TfTpuOrdinalSelector_DequeueFromCoreSelector(sel, 0, -1);
            TfTpuOrdinalSelector_DequeueFromCoreSelector(sel, 0, 1);
            TfTpuOrdinalSelector_Destroy(sel);
            TfTpuOrdinalSelector_Destroy(ptr::null_mut());

            let topo = TpuUtil_GetTopologyPtr();
            assert!(!topo.is_null());
            assert_ne!(format!("{:?}", *topo), "");
            assert_eq!(TpuUtil_GetXlaPadSizeFromTpuTopology(), 128);
            assert_eq!(TpuTopology_LogicalDevicesPerHost(topo, 0), 4);
            assert_eq!(TpuTopology_LogicalDevicesPerHost(ptr::null(), 0), 4);
            assert_eq!(TpuTopology_LogicalDevicesPerChip(topo, 0), 1);
            assert_eq!(TpuTopology_LogicalDevicesPerChip(topo, 1), 2);
            assert_eq!(TpuTopology_HostCount(topo), 1);
            assert_eq!(TpuTopology_ChipsPerHost(topo), 4);
            assert_eq!(TpuTopology_ChipBounds_X(topo), 2);
            assert_eq!(TpuTopology_ChipBounds_Y(topo), 2);
            assert_eq!(TpuTopology_ChipBounds_Z(topo), 1);
            assert!(TpuTopology_HasChip(topo, 0, 0, 0));
            assert!(!TpuTopology_HasChip(topo, 9, 0, 0));
            assert!(!TpuTopology_HasChip(topo, -1, 0, 0));
            assert!(!TpuTopology_HasChip(topo, 0, -1, 0));
            assert!(!TpuTopology_HasChip(topo, 0, 0, -1));
            assert!(!TpuTopology_HasChip(topo, 0, 9, 0));
            assert!(!TpuTopology_HasChip(topo, 0, 0, 9));

            let c0 = TpuTopology_CoreForId(topo, 0, 0);
            assert!(!c0.is_null());
            assert_ne!(format!("{:?}", *c0), "");
            assert!(TpuTopology_CoreForId(topo, 0, 99).is_null());
            assert!(!TpuTopology_Core(topo, 0, 0, 0, 0, 0).is_null());
            assert!(TpuTopology_Core(topo, 0, 9, 9, 9, 0).is_null());
            assert_eq!(TpuTopology_NumCores(topo, 0), 4);
            assert_eq!(TpuTopology_NumCores(topo, 1), 8);
            TpuTopology_Cores(topo, 0, ptr::null_mut());
            let mut core_arr: [*const TpuCoreLocation; 4] = [ptr::null(); 4];
            TpuTopology_Cores(topo, 0, core_arr.as_mut_ptr());
            assert!(!core_arr[0].is_null());
            assert_eq!(TpuTopology_IdForHost(topo, 0, 0, 0), 0);
            assert_eq!(TpuTopology_Version(topo), 5);
            assert_eq!(TpuTopology_AvailableCoreCount(ptr::null(), 0), 4);
            assert_eq!(TpuTopology_AvailableCoresPerChip(0), 1);
            assert_eq!(TpuTopology_AvailableCoresPerChip(1), 2);
            assert_eq!(TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice(0), 2);

            let (mut x, mut y, mut z) = (-1, -1, -1);
            TpuCoreLocation_ChipCoordinates(
                ptr::null(),
                ptr::addr_of_mut!(x),
                ptr::addr_of_mut!(y),
                ptr::addr_of_mut!(z),
            );
            TpuCoreLocation_ChipCoordinates(
                c0,
                ptr::addr_of_mut!(x),
                ptr::addr_of_mut!(y),
                ptr::addr_of_mut!(z),
            );
            assert_eq!((x, y, z), (0, 0, 0));
            TpuCoreLocation_HostCoordinates(
                ptr::null(),
                ptr::addr_of_mut!(x),
                ptr::addr_of_mut!(y),
                ptr::addr_of_mut!(z),
            );
            TpuCoreLocation_HostCoordinates(
                c0,
                ptr::addr_of_mut!(x),
                ptr::addr_of_mut!(y),
                ptr::addr_of_mut!(z),
            );
            assert_eq!(TpuCoreLocation_Index(ptr::null()), -1);
            assert_eq!(TpuCoreLocation_Index(c0), 0);
            assert_eq!(TpuCoreLocation_Id(ptr::null()), -1);
            assert_eq!(TpuCoreLocation_Id(c0), 0);
            let host_loc = TpuHostLocation {
                host_id: 2,
                host_coords: [0, 0, 0],
                num_cores: 4,
            };
            assert_ne!(format!("{host_loc:?}"), "");
            assert_eq!(TpuHostLocation_Id(ptr::null()), 0);
            assert_eq!(TpuHostLocation_Id(ptr::addr_of!(host_loc)), 2);
            assert_eq!(TpuHostLocation_NumCores(ptr::null(), 0), 4);
            assert_eq!(TpuHostLocation_NumCores(ptr::addr_of!(host_loc), 0), 4);
            let host_loc_zero = TpuHostLocation {
                host_id: 0,
                host_coords: [0, 0, 0],
                num_cores: 0,
            };
            assert_eq!(TpuHostLocation_NumCores(ptr::addr_of!(host_loc_zero), 0), 4);
            TpuHostLocation_Cores(ptr::null(), 0, core_arr.as_mut_ptr());

            let ms = TpuMeshState_Create();
            assert_ne!(format!("{:?}", *ms), "");
            assert!(!TpuMeshState_MeshCommonState(ms).is_null());
            assert!(TpuMeshState_MeshCommonState(ptr::null()).is_null());
            TpuMeshState_Free(ms);
            TpuMeshState_Free(ptr::null_mut());

            let st_prof = TpuStatus_New();
            TpuProfiler_Create(ptr::null_mut(), st_prof);
            let mut prof: *mut TpuProfiler = ptr::null_mut();
            TpuProfiler_Create(ptr::addr_of_mut!(prof), st_prof);
            assert_ne!(format!("{:?}", *prof), "");
            TpuProfiler_Start(ptr::null_mut(), st_prof);
            TpuProfiler_Stop(ptr::null_mut(), st_prof);
            TpuProfiler_Stop(prof, st_prof);
            TpuProfiler_Start(prof, st_prof);
            TpuProfiler_Start(prof, st_prof);
            TpuProfiler_Stop(prof, st_prof);
            let mut sz: usize = 0;
            TpuProfiler_CollectData(
                ptr::null_mut(),
                st_prof,
                ptr::null_mut(),
                ptr::addr_of_mut!(sz),
            );
            TpuProfiler_CollectData(prof, st_prof, ptr::null_mut(), ptr::addr_of_mut!(sz));
            assert!(sz > 0);
            let mut prof_buf = vec![0_u8; sz];
            TpuProfiler_CollectData(prof, st_prof, prof_buf.as_mut_ptr(), ptr::addr_of_mut!(sz));
            TpuProfiler_Destroy(prof);
            TpuProfiler_Destroy(ptr::null_mut());
            TpuStatus_Free(st_prof);

            let emb = TpuEmbeddingEngineState_Create();
            assert_ne!(format!("{:?}", *emb), "");
            assert!(!TpuEmbeddingEngineState_GetState(emb).is_null());
            let mut emb_out_sz: usize = 0;
            let mut emb_out_ptr: *mut c_char = ptr::null_mut();
            let mut emb_params = TpuEmbeddingEngine_ExecutePartitioner_Params {
                struct_size: size_of::<TpuEmbeddingEngine_ExecutePartitioner_Params>(),
                priv_data: ptr::null_mut(),
                tpu_embedding_config_bytes: ptr::null(),
                tpu_embedding_config_size: 0,
                common_config_string_size: ptr::addr_of_mut!(emb_out_sz),
                common_config_string: ptr::addr_of_mut!(emb_out_ptr),
                status: ptr::null_mut(),
            };
            assert_ne!(format!("{emb_params:?}"), "");
            assert_eq!(TpuEmbeddingEngine_ExecutePartitioner(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_ExecutePartitioner(ptr::addr_of_mut!(emb_params)),
                0
            );
            assert!(emb_out_sz > 0 && !emb_out_ptr.is_null());
            TpuConfigurationApi_FreeCharArray(emb_out_ptr);
            emb_params.tpu_embedding_config_bytes = c"tbl".as_ptr();
            emb_params.tpu_embedding_config_size = 3;
            emb_params.common_config_string_size = ptr::null_mut();
            emb_params.common_config_string = ptr::null_mut();
            assert_eq!(
                TpuEmbeddingEngine_ExecutePartitioner(ptr::addr_of_mut!(emb_params)),
                0
            );
            emb_params.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_ExecutePartitioner(ptr::addr_of_mut!(emb_params)),
                -1
            );

            let mut mem_sz: usize = 0;
            let mut mem_ptr: *mut c_char = ptr::null_mut();
            let mut cfg_mem = TpuEmbeddingEngine_ConfigureMemory_Params {
                struct_size: size_of::<TpuEmbeddingEngine_ConfigureMemory_Params>(),
                priv_data: ptr::null_mut(),
                num_inputs: 1,
                pad: 0,
                common_config_string: ptr::null(),
                common_config_string_size: 0,
                task_host_config_size: ptr::addr_of_mut!(mem_sz),
                task_host_config: ptr::addr_of_mut!(mem_ptr),
                status: ptr::null_mut(),
            };
            assert_eq!(TpuEmbeddingEngine_ConfigureMemory(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_ConfigureMemory(ptr::addr_of_mut!(cfg_mem)),
                0
            );
            TpuConfigurationApi_FreeCharArray(mem_ptr);
            cfg_mem.common_config_string = c"custom_mem".as_ptr();
            cfg_mem.common_config_string_size = 10;
            cfg_mem.task_host_config_size = ptr::null_mut();
            cfg_mem.task_host_config = ptr::null_mut();
            assert_eq!(
                TpuEmbeddingEngine_ConfigureMemory(ptr::addr_of_mut!(cfg_mem)),
                0
            );
            cfg_mem.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_ConfigureMemory(ptr::addr_of_mut!(cfg_mem)),
                -1
            );

            let mut col_sz: usize = 0;
            let mut col_mem = TpuEmbeddingEngine_CollateMemory_Params {
                struct_size: size_of::<TpuEmbeddingEngine_CollateMemory_Params>(),
                priv_data: ptr::null_mut(),
                memory_configs_size: 0,
                memory_configs: ptr::null(),
                collated_size: ptr::addr_of_mut!(col_sz),
                merged_memory_config: ptr::null_mut(),
                status: ptr::null_mut(),
            };
            assert_eq!(TpuEmbeddingEngine_CollateMemory(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_CollateMemory(ptr::addr_of_mut!(col_mem)),
                0
            );
            col_mem.collated_size = ptr::null_mut();
            assert_eq!(
                TpuEmbeddingEngine_CollateMemory(ptr::addr_of_mut!(col_mem)),
                0
            );
            col_mem.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_CollateMemory(ptr::addr_of_mut!(col_mem)),
                -1
            );

            let mut net_sz: usize = 0;
            let mut net_ptr: *mut c_char = ptr::null_mut();
            let mut cfg_host = TpuEmbeddingEngine_ConfigureHost_Params {
                struct_size: size_of::<TpuEmbeddingEngine_ConfigureHost_Params>(),
                priv_data: ptr::null_mut(),
                num_inputs: 1,
                pad: 0,
                common_config_string: ptr::null(),
                common_config_string_size: 0,
                task_host_config: ptr::null(),
                task_host_config_size: 0,
                tpu_embedding_config_bytes: ptr::null(),
                tpu_embedding_config_size: 0,
                network_config_size: ptr::addr_of_mut!(net_sz),
                network_config: ptr::addr_of_mut!(net_ptr),
                status: ptr::null_mut(),
            };
            assert_eq!(TpuEmbeddingEngine_ConfigureHost(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_ConfigureHost(ptr::addr_of_mut!(cfg_host)),
                0
            );
            TpuConfigurationApi_FreeCharArray(net_ptr);
            cfg_host.common_config_string = c"host_1".as_ptr();
            cfg_host.common_config_string_size = 6;
            cfg_host.network_config_size = ptr::null_mut();
            cfg_host.network_config = ptr::null_mut();
            assert_eq!(
                TpuEmbeddingEngine_ConfigureHost(ptr::addr_of_mut!(cfg_host)),
                0
            );
            cfg_host.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_ConfigureHost(ptr::addr_of_mut!(cfg_host)),
                -1
            );

            let mut conn_hosts = TpuEmbeddingEngine_ConnectHosts_Params {
                struct_size: size_of::<TpuEmbeddingEngine_ConnectHosts_Params>(),
                priv_data: ptr::null_mut(),
                network_configs: ptr::null(),
                network_configs_size: 0,
                status: ptr::null_mut(),
            };
            assert_eq!(TpuEmbeddingEngine_ConnectHosts(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_ConnectHosts(ptr::addr_of_mut!(conn_hosts)),
                0
            );
            conn_hosts.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_ConnectHosts(ptr::addr_of_mut!(conn_hosts)),
                -1
            );

            let mut fin_params = TpuEmbeddingEngine_Finalize_Params {
                struct_size: size_of::<TpuEmbeddingEngine_Finalize_Params>(),
                priv_data: ptr::null_mut(),
                common_config_string: ptr::null(),
                common_config_string_size: 0,
                task_host_config: ptr::null(),
                task_host_config_size: 0,
                mesh_common_state: ptr::null(),
                status: ptr::null_mut(),
            };
            assert_eq!(TpuEmbeddingEngine_Finalize(ptr::null_mut()), -1);
            assert!(!TpuEmbeddingEngine_IsInitialized(ptr::null()));
            assert_eq!(
                TpuEmbeddingEngine_Finalize(ptr::addr_of_mut!(fin_params)),
                0
            );
            let mut is_init_bool = false;
            let init_params = TpuEmbeddingEngine_IsInitialized_Params {
                struct_size: size_of::<TpuEmbeddingEngine_IsInitialized_Params>(),
                priv_data: ptr::null_mut(),
                config_string: ptr::null(),
                config_string_size: 0,
                is_embedding_initialized: ptr::addr_of_mut!(is_init_bool),
                status: ptr::null_mut(),
            };
            assert!(TpuEmbeddingEngine_IsInitialized(ptr::addr_of!(init_params)));
            assert!(is_init_bool);
            let init_params_null_out = TpuEmbeddingEngine_IsInitialized_Params {
                struct_size: 0,
                priv_data: ptr::null_mut(),
                config_string: ptr::null(),
                config_string_size: 0,
                is_embedding_initialized: ptr::null_mut(),
                status: ptr::null_mut(),
            };
            assert!(TpuEmbeddingEngine_IsInitialized(ptr::addr_of!(
                init_params_null_out
            )));

            let mut wr_buf = [7_u8; 16];
            let mut rd_buf = [0_u8; 16];
            let mut rd_floats = [0.0_f32; 4];
            let mut num_rw: usize = 0;
            let mut wr_params = TpuEmbeddingEngine_WriteParameters_Params {
                struct_size: size_of::<TpuEmbeddingEngine_WriteParameters_Params>(),
                priv_data: ptr::null_mut(),
                table_name: ptr::null(),
                table_name_len: 0,
                parameters: ptr::null(),
                parameters_out: ptr::null_mut(),
                num_parameters: 4,
                num_parameters_written_or_read: ptr::addr_of_mut!(num_rw),
                status: ptr::null_mut(),
            };
            assert_eq!(TpuEmbeddingEngine_WriteParameters(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_WriteParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            assert_eq!(num_rw, 4);
            wr_params.parameters = wr_buf.as_mut_ptr().cast();
            wr_params.num_parameters_written_or_read = ptr::null_mut();
            assert_eq!(
                TpuEmbeddingEngine_WriteParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            wr_params.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_WriteParameters(ptr::addr_of_mut!(wr_params)),
                -1
            );
            wr_params.struct_size = size_of::<TpuEmbeddingEngine_WriteParameters_Params>();
            wr_params.parameters = ptr::null();
            assert_eq!(TpuEmbeddingEngine_ReadParameters(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_ReadParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            wr_params.parameters = rd_buf.as_mut_ptr().cast();
            assert_eq!(
                TpuEmbeddingEngine_ReadParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            assert_eq!(rd_buf, [7_u8; 16]);
            wr_params.priv_data = c"custom_table".as_ptr().cast_mut().cast();
            wr_params.parameters = wr_buf.as_mut_ptr().cast();
            assert_eq!(
                TpuEmbeddingEngine_WriteParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            rd_buf = [0_u8; 16];
            wr_params.parameters = rd_buf.as_mut_ptr().cast();
            assert_eq!(
                TpuEmbeddingEngine_ReadParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            assert_eq!(rd_buf, [7_u8; 16]);
            wr_params.priv_data = ptr::null_mut();
            wr_params.table_name = c"named_table".as_ptr();
            wr_params.table_name_len = 11;
            wr_params.parameters = wr_buf.as_mut_ptr().cast();
            wr_params.num_parameters_written_or_read = ptr::addr_of_mut!(num_rw);
            assert_eq!(
                TpuEmbeddingEngine_WriteParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            wr_params.parameters = ptr::null();
            wr_params.parameters_out = rd_floats.as_mut_ptr();
            assert_eq!(
                TpuEmbeddingEngine_ReadParameters(ptr::addr_of_mut!(wr_params)),
                0
            );
            assert_eq!(num_rw, 4);
            wr_params.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_ReadParameters(ptr::addr_of_mut!(wr_params)),
                -1
            );

            let fixed = TpuEmbeddingTensorBatchFixedState_Create();
            TpuEmbeddingTensorBatchFixedState_Destroy(fixed);
            TpuEmbeddingTensorBatchFixedState_Destroy(ptr::null_mut());

            let dummy_indices = [0_i32; 4];
            let dummy_weights = [1.0_f32; 4];
            let mut enq_params = TpuEmbeddingEngine_EnqueueTensorBatch_Params {
                struct_size: size_of::<TpuEmbeddingEngine_EnqueueTensorBatch_Params>(),
                priv_data: ptr::null_mut(),
                sample_indices: ptr::null(),
                embedding_indices: ptr::null(),
                aggregation_weights: ptr::null(),
                num_samples: 0,
                status: ptr::null_mut(),
            };
            assert_eq!(TpuEmbeddingEngine_EnqueueTensorBatch(ptr::null_mut()), 0);
            assert_eq!(
                TpuEmbeddingEngine_EnqueueTensorBatch(ptr::addr_of_mut!(enq_params)),
                0
            );
            enq_params.sample_indices = dummy_indices.as_ptr().cast();
            enq_params.embedding_indices = dummy_indices.as_ptr().cast();
            enq_params.aggregation_weights = dummy_weights.as_ptr().cast();
            assert_eq!(
                TpuEmbeddingEngine_EnqueueTensorBatch(ptr::addr_of_mut!(enq_params)),
                0
            );
            enq_params.num_samples = 8;
            assert_eq!(
                TpuEmbeddingEngine_EnqueueTensorBatch(ptr::addr_of_mut!(enq_params)),
                0
            );
            enq_params.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_EnqueueTensorBatch(ptr::addr_of_mut!(enq_params)),
                -1
            );

            let mut xla_out = TpuSerializedProto {
                bytes: ptr::null_mut(),
                size: 0,
            };
            let mut comp_params = TpuEmbeddingEngine_Computation_Params {
                struct_size: size_of::<TpuEmbeddingEngine_Computation_Params>(),
                priv_data: ptr::null_mut(),
                tpu_embedding_config_bytes: ptr::null(),
                tpu_embedding_config_size: 0,
                deduplication_tuple_mask_bytes: ptr::null(),
                deduplication_tuple_mask_size: 0,
                hbm_buffers_config_bytes: ptr::null(),
                hbm_buffers_config_size: 0,
                tpu_topology_bytes: ptr::null(),
                tpu_topology_size: 0,
                xla_computation_out: ptr::addr_of_mut!(xla_out),
                status: ptr::null_mut(),
            };
            assert_eq!(
                TpuEmbeddingEngine_RecvActivationsComputation(ptr::null_mut()),
                0
            );
            assert_eq!(
                TpuEmbeddingEngine_RecvActivationsComputation(ptr::addr_of_mut!(comp_params)),
                0
            );
            assert!(!xla_out.bytes.is_null() && xla_out.size > 0);
            TpuConfigurationApi_FreeCharArray(xla_out.bytes);
            comp_params.xla_computation_out = ptr::null_mut();
            assert_eq!(
                TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation(ptr::addr_of_mut!(
                    comp_params
                )),
                0
            );
            assert_eq!(
                TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(ptr::addr_of_mut!(
                    comp_params
                )),
                0
            );
            assert_eq!(
                TpuEmbeddingEngine_DedupDataSizeComputation(ptr::addr_of_mut!(comp_params)),
                0
            );
            assert_eq!(
                TpuEmbeddingEngine_DedupDataTupleMaskComputation(ptr::addr_of_mut!(comp_params)),
                0
            );
            comp_params.struct_size = 0;
            assert_eq!(
                TpuEmbeddingEngine_DedupDataTupleMaskComputation(ptr::addr_of_mut!(comp_params)),
                -1
            );
            TpuEmbeddingEngineState_Free(emb);
            TpuEmbeddingEngineState_Free(ptr::null_mut());

            TpuDriver_Initialize(ptr::null_mut(), false);
            let mut drv_fn_init: TpuDriverFn = std::mem::zeroed();
            TpuDriver_Initialize(ptr::addr_of_mut!(drv_fn_init), true);
            assert!(!drv_fn_init.open.is_null());

            let drv = TpuDriver_Open(c"local://".as_ptr());
            assert!(!(*drv).fn_table.is_null());
            assert_ne!(format!("{:?}{:?}", *drv, *(*drv).fn_table), "");
            assert_eq!(
                TpuDriver_ComputeLinearizedBytesFromShape(ptr::null_mut(), 16),
                0
            );
            assert_eq!(TpuDriver_ComputeLinearizedBytesFromShape(drv, 16), 256);
            assert!(
                TpuDriver_CompileProgramFromText(ptr::null_mut(), ptr::null(), 1, 0, ptr::null())
                    .is_null()
            );
            let cp_text = TpuDriver_CompileProgramFromText(
                drv,
                c"HloModule text_add".as_ptr(),
                1,
                0,
                ptr::null(),
            );
            assert!(!cp_text.is_null());
            assert!(TpuDriver_GetCompiledProgramShape(ptr::null_mut(), cp_text).is_null());
            assert_eq!(TpuDriver_GetCompiledProgramShape(drv, cp_text), cp_text);
            TpuDriver_FreeCompiledProgramHandle(cp_text);
            let cp_text_null =
                TpuDriver_CompileProgramFromText(drv, ptr::null(), 1, 0, ptr::null());
            TpuDriver_FreeCompiledProgramHandle(cp_text_null);
            TpuDriver_AddCallback(drv, ptr::null_mut(), ptr::null(), ptr::null_mut());
            assert!(!TpuDriver_Version().is_null());

            let st_reset_ok = TpuDriver_Reset(drv);
            assert!(TpuStatus_Ok(st_reset_ok));
            TpuStatus_Free(st_reset_ok);
            let st_reset_err = TpuDriver_Reset(ptr::null_mut());
            assert!(!TpuStatus_Ok(st_reset_err));
            TpuStatus_Free(st_reset_err);

            assert!(TpuDriver_QuerySystemInfo(ptr::null_mut()).is_null());
            TpuDriver_FreeSystemInfo(ptr::null_mut());
            let sys_info = TpuDriver_QuerySystemInfo(drv);
            assert!(!sys_info.is_null());
            assert_ne!(format!("{:?}", *sys_info), "");
            TpuDriver_FreeSystemInfo(sys_info);

            assert!(TpuDriver_Compile(ptr::null_mut(), ptr::null(), 1, 0, ptr::null()).is_null());
            TpuDriver_FreeCompiledProgramHandle(ptr::null_mut());
            let hlo_src_bytes = b"module @add { stablehlo.add }";
            let hlo_proto = crate::tpu_driver_types::HloProto {
                buffer: hlo_src_bytes.as_ptr().cast(),
                size: i64::try_from(hlo_src_bytes.len()).unwrap_or(0),
            };
            assert_ne!(format!("{hlo_proto:?}"), "");
            let cp_default = TpuDriver_Compile(drv, ptr::null(), 1, 0, ptr::null());
            assert!(!cp_default.is_null());
            assert_ne!(format!("{:?}", *cp_default), "");
            TpuDriver_FreeCompiledProgramHandle(cp_default);
            let bad_hlo = crate::tpu_driver_types::HloProto {
                buffer: b"".as_ptr().cast(),
                size: 0,
            };
            let cp_bad = TpuDriver_Compile(drv, ptr::addr_of!(bad_hlo), 1, 0, ptr::null());
            TpuDriver_FreeCompiledProgramHandle(cp_bad);
            let cp_handle = TpuDriver_Compile(drv, ptr::addr_of!(hlo_proto), 1, 0, ptr::null());
            assert!(!cp_handle.is_null());

            assert!(TpuDriver_LoadProgram(ptr::null_mut(), 0, cp_handle, 0, ptr::null()).is_null());
            let lp_default = TpuDriver_LoadProgram(drv, 0, ptr::null(), 0, ptr::null());
            let unl_ev0 = TpuDriver_UnloadProgram(drv, lp_default, 0, ptr::null());
            TpuDriver_FreeEvent(unl_ev0);
            let lp_handle = TpuDriver_LoadProgram(drv, 0, cp_handle, 0, ptr::null());
            assert!(!lp_handle.is_null());
            assert_ne!(format!("{:?}", *lp_handle), "");

            assert!(TpuDriver_Allocate(ptr::null_mut(), 0, 0, 16, 0, ptr::null()).is_null());
            let drv_b0 = TpuDriver_Allocate(drv, 0, 0, 16, 0, ptr::null());
            let drv_b1 = TpuDriver_AllocateShape(drv, 0, 0, 16, 0, ptr::null());
            let drv_b_out = TpuDriver_AllocateTuple(drv, 0, 0, 2, ptr::null(), 0, ptr::null());
            assert!(!drv_b0.is_null() && !drv_b1.is_null() && !drv_b_out.is_null());
            assert_ne!(format!("{:?}", *drv_b0), "");

            let h2d_src = [1_u8; 16];
            let mut d2h_dst = [0_u8; 16];
            TpuDriver_FreeEvent(TpuDriver_TransferToDevice(
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::null(),
            ));
            let ev_h2d =
                TpuDriver_TransferToDevice(drv, h2d_src.as_ptr().cast(), drv_b0, 0, ptr::null());
            TpuDriver_FreeEvent(ev_h2d);

            TpuDriver_FreeEvent(TpuDriver_TransferFromDeviceToDevice(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null(),
            ));
            let ev_d2d = TpuDriver_TransferFromDeviceToDevice(drv, drv_b0, drv_b1, 0, ptr::null());
            TpuDriver_FreeEvent(ev_d2d);

            TpuDriver_FreeEvent(TpuDriver_TransferFromDevice(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null(),
            ));
            let ev_d2h = TpuDriver_TransferFromDevice(
                drv,
                drv_b1,
                d2h_dst.as_mut_ptr().cast(),
                0,
                ptr::null(),
            );
            assert_eq!(d2h_dst, h2d_src);

            let da = crate::tpu_driver_types::DeviceAssignment {
                replica_count: 1,
                computation_count: 1,
            };
            assert_ne!(format!("{da:?}"), "");
            let in_bufs = [drv_b0, drv_b1];
            let out_bufs = [drv_b_out];
            TpuDriver_FreeEvent(TpuDriver_ExecuteProgram(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                da,
                0,
                ptr::null(),
            ));
            let ev_exec = TpuDriver_ExecuteProgram(
                drv,
                lp_handle,
                2,
                in_bufs.as_ptr(),
                1,
                out_bufs.as_ptr(),
                da,
                0,
                ptr::null(),
            );
            assert_ne!(format!("{:?}", *ev_exec), "");

            let wait_events = [ev_d2h, ev_exec];
            let st_wait_ok = TpuDriver_WaitForEvents(drv, 2, wait_events.as_ptr(), 1000);
            assert!(TpuStatus_Ok(st_wait_ok));
            TpuStatus_Free(st_wait_ok);
            let st_wait_err = TpuDriver_WaitForEvents(ptr::null_mut(), 0, ptr::null(), 0);
            assert!(!TpuStatus_Ok(st_wait_err));
            TpuStatus_Free(st_wait_err);
            TpuDriver_FreeEvent(ev_d2h);
            TpuDriver_FreeEvent(ev_exec);
            TpuDriver_FreeEvent(ptr::null_mut());

            let mut lin_dst = [0_u8; 16];
            let st_lin_err =
                TpuDriver_LinearizeShape(ptr::null_mut(), ptr::null_mut(), ptr::null(), 16);
            assert!(!TpuStatus_Ok(st_lin_err));
            TpuStatus_Free(st_lin_err);
            let st_lin_ok = TpuDriver_LinearizeShape(
                drv,
                lin_dst.as_mut_ptr().cast(),
                h2d_src.as_ptr().cast(),
                16,
            );
            assert!(TpuStatus_Ok(st_lin_ok));
            TpuStatus_Free(st_lin_ok);
            let st_delin_err =
                TpuDriver_DelinearizeShape(ptr::null_mut(), ptr::null_mut(), ptr::null(), 16);
            assert!(!TpuStatus_Ok(st_delin_err));
            TpuStatus_Free(st_delin_err);
            let st_delin_ok = TpuDriver_DelinearizeShape(
                drv,
                d2h_dst.as_mut_ptr().cast(),
                lin_dst.as_ptr().cast(),
                16,
            );
            assert!(TpuStatus_Ok(st_delin_ok));
            TpuStatus_Free(st_delin_ok);

            TpuDriver_FreeEvent(TpuDriver_Deallocate(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null(),
            ));
            for b_ptr in [drv_b0, drv_b1, drv_b_out] {
                let ev_dealloc = TpuDriver_Deallocate(drv, b_ptr, 0, ptr::null());
                TpuDriver_FreeEvent(ev_dealloc);
            }
            TpuDriver_FreeEvent(TpuDriver_UnloadProgram(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null(),
            ));
            let ev_unload = TpuDriver_UnloadProgram(drv, lp_handle, 0, ptr::null());
            TpuDriver_FreeEvent(ev_unload);
            TpuDriver_FreeCompiledProgramHandle(cp_handle);

            // Exercise all 21 C ABI alias entrypoints
            let st_alias = TpuStatus_New();
            TpuStatus_Destroy(st_alias);
            let st_drv_alias = TpuStatus_New();
            TpuDriver_FreeStatus(st_drv_alias);
            let cp_prog_alias =
                TpuDriver_CompileProgram(drv, ptr::addr_of!(hlo_proto), 1, 0, ptr::null());
            assert!(!cp_prog_alias.is_null());
            TpuDriver_FreeCompiledProgramShape(cp_prog_alias);
            let st_wait_ev_alias = TpuDriver_WaitForEvent(drv, 0, ptr::null(), 100);
            assert!(TpuStatus_Ok(st_wait_ev_alias));
            TpuDriver_FreeStatus(st_wait_ev_alias);

            let topo_ptr = TpuUtil_GetTopologyPtr();
            let avail_core0 = TpuTopology_AnAvailableCore(topo_ptr, 0);
            assert!(!avail_core0.is_null());
            let mut avail_cores: [*const TpuCoreLocation; 16] = [ptr::null(); 16];
            TpuTopology_AvailableCores(topo_ptr, 0, avail_cores.as_mut_ptr());
            assert_eq!(avail_cores[0], avail_core0);
            let avail_cnt = TpuTopology_AvailableCoresCount(topo_ptr, 0);
            assert!(avail_cnt > 0);
            let mut cx = 0;
            let mut cy = 0;
            let mut cz = 0;
            TpuTopology_ChipCoordinates(
                avail_core0,
                ptr::addr_of_mut!(cx),
                ptr::addr_of_mut!(cy),
                ptr::addr_of_mut!(cz),
            );
            assert!(TpuTopology_ChipCount(topo_ptr) > 0);
            assert_eq!(TpuTopology_CoreCount(topo_ptr, 0), avail_cnt);
            assert_eq!(TpuTopology_CoreId(avail_core0), 0);
            assert_eq!(TpuTopology_CoreIndexOnChip(avail_core0), 0);
            TpuTopology_CoreLocations(topo_ptr, 0, avail_cores.as_mut_ptr());
            assert_eq!(TpuTopology_CurrentHost(topo_ptr), 0);
            assert!(TpuTopology_HostBounds_X(topo_ptr) >= 1);
            assert!(TpuTopology_HostBounds_Y(topo_ptr) >= 1);
            assert!(TpuTopology_HostBounds_Z(topo_ptr) >= 1);
            TpuTopology_HostLocations(ptr::null(), 0, avail_cores.as_mut_ptr());

            let st_dist = TpuStatus_New();
            assert_eq!(WaitForDistributedTpuOp_DoWork(0, 1, st_dist), -1);
            assert_eq!(WaitForDistributedTpuOp_DoWork(1, 0, st_dist), -1);
            assert_eq!(WaitForDistributedTpuOp_DoWork(1, 4, st_dist), 0);
            assert_eq!(InitializeHostForDistributedTpuOp_DoWork(-1, st_dist), -1);
            assert_eq!(InitializeHostForDistributedTpuOp_DoWork(0, st_dist), 0);
            assert_eq!(SetGlobalTPUArrayOp(0, ptr::null(), st_dist), 0);
            assert_eq!(DisconnectDistributedTpuChipsOp_DoWork(st_dist), 0);
            assert_eq!(ShutdownDistributedTpuOp_DoWork(st_dist), 0);
            assert_eq!(TpuNodeContext_Initialize(-1, st_dist), -1);
            assert_eq!(TpuNodeContext_Initialize(0, st_dist), 0);
            assert!(TpuNodeContext_Create(-1, st_dist).is_null());
            let node_ctx = TpuNodeContext_Create(0, st_dist);
            assert!(!node_ctx.is_null());
            TpuMeshState_Free(node_ctx);
            assert_eq!(TpuNodeContext_CloseTpuHost(st_dist), 0);
            assert_eq!(TpuNodeContext_StopChipHeartbeats(st_dist), 0);
            assert!(!TpuNodeContext_CompilationsDisabled());
            assert_eq!(TpuCompile_CompileAndBuild(ptr::null(), 0, st_dist), -1);
            assert_eq!(TpuCompile_CompileAndBuild(ptr::null(), 4, st_dist), -1);
            assert_eq!(TpuCompile_CompileAndBuild(c"mlir".as_ptr(), 4, st_dist), 0);
            assert_eq!(
                TpuCompile_XrtCompileAndBuild(c"mlir".as_ptr(), 4, st_dist),
                0
            );
            assert_eq!(TpuExecutable_LoadProgramAndEnqueueToStream(-1, st_dist), -1);
            assert_eq!(TpuExecutable_LoadProgramAndEnqueueToStream(0, st_dist), 0);
            let res_mgr = TpuResourceMgr_Create();
            assert!(!res_mgr.is_null());
            TpuResourceMgr_Destroy(res_mgr);
            TpuStatus_Destroy(st_dist);

            TpuDriver_Close(drv);
            TpuDriver_Close(ptr::null_mut());

            let bad_st = libtpu_rs_client_create(ptr::null_mut());
            assert_ne!(format!("{bad_st:?}"), "");
            assert_ne!(bad_st.code, 0);
            libtpu_rs_status_free(bad_st);
            libtpu_rs_status_free(c_status_ok());

            let mut rs_client: *mut PJRT_Client = ptr::null_mut();
            let st_ok = libtpu_rs_client_create(ptr::addr_of_mut!(rs_client));
            assert_eq!(st_ok.code, 0);
            assert_eq!(libtpu_rs_client_device_count(ptr::null()), -1);
            assert_eq!(libtpu_rs_client_device_count(rs_client), 4);

            let dims = [4_i64];
            let mut in0 = Vec::new();
            let mut in1 = Vec::new();
            for f in [2.0_f32, 4.0, 6.0, 8.0] {
                in0.extend_from_slice(&f.to_le_bytes());
            }
            for f in [1.0_f32, 2.0, 3.0, 4.0] {
                in1.extend_from_slice(&f.to_le_bytes());
            }

            let mut b0: *mut PJRT_Buffer = ptr::null_mut();
            let mut b1: *mut PJRT_Buffer = ptr::null_mut();
            let mut b_scalar: *mut PJRT_Buffer = ptr::null_mut();

            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                ptr::null_mut(),
                0,
                11,
                dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                rs_client,
                0,
                999,
                dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));
            let neg_dims = [-1_i64];
            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                rs_client,
                0,
                11,
                neg_dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                rs_client,
                99,
                11,
                dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));

            assert_eq!(
                libtpu_rs_buffer_from_host(
                    rs_client,
                    0,
                    11,
                    ptr::null(),
                    0,
                    ptr::null(),
                    0,
                    ptr::addr_of_mut!(b_scalar),
                )
                .code,
                0
            );
            assert_ne!(format!("{:?}", *b_scalar), "");
            libtpu_rs_buffer_destroy(b_scalar);

            assert_eq!(
                libtpu_rs_buffer_from_host(
                    rs_client,
                    0,
                    11,
                    dims.as_ptr(),
                    1,
                    in0.as_ptr(),
                    in0.len(),
                    ptr::addr_of_mut!(b0),
                )
                .code,
                0
            );
            assert_eq!(
                libtpu_rs_buffer_from_host(
                    rs_client,
                    0,
                    11,
                    dims.as_ptr(),
                    1,
                    in1.as_ptr(),
                    in1.len(),
                    ptr::addr_of_mut!(b1),
                )
                .code,
                0
            );

            let mut exec: *mut PJRT_LoadedExecutable = ptr::null_mut();
            let code_src = b"module @add { stablehlo.add }";
            libtpu_rs_status_free(libtpu_rs_compile(
                ptr::null_mut(),
                c"mlir".as_ptr(),
                code_src.as_ptr(),
                code_src.len(),
                ptr::addr_of_mut!(exec),
            ));
            libtpu_rs_status_free(libtpu_rs_compile(
                rs_client,
                c"bad".as_ptr(),
                code_src.as_ptr(),
                code_src.len(),
                ptr::addr_of_mut!(exec),
            ));
            assert_eq!(
                libtpu_rs_compile(
                    rs_client,
                    c"mlir".as_ptr(),
                    code_src.as_ptr(),
                    code_src.len(),
                    ptr::addr_of_mut!(exec),
                )
                .code,
                0
            );

            let inputs = [b0.cast_const(), b1.cast_const()];
            let mut out_buf: *mut PJRT_Buffer = ptr::null_mut();
            libtpu_rs_status_free(libtpu_rs_execute(
                ptr::null(),
                0,
                inputs.as_ptr(),
                2,
                ptr::addr_of_mut!(out_buf),
            ));
            let null_inputs: [*const PJRT_Buffer; 1] = [ptr::null()];
            libtpu_rs_status_free(libtpu_rs_execute(
                exec,
                0,
                null_inputs.as_ptr(),
                1,
                ptr::addr_of_mut!(out_buf),
            ));
            libtpu_rs_status_free(libtpu_rs_execute(
                exec,
                99,
                ptr::null(),
                0,
                ptr::addr_of_mut!(out_buf),
            ));
            assert_eq!(
                libtpu_rs_execute(exec, 0, inputs.as_ptr(), 2, ptr::addr_of_mut!(out_buf)).code,
                0
            );

            let mut host_out = [0_u8; 16];
            libtpu_rs_status_free(libtpu_rs_buffer_to_host(
                ptr::null(),
                host_out.as_mut_ptr(),
                16,
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_to_host(
                out_buf,
                host_out.as_mut_ptr(),
                1024,
            ));
            assert_eq!(
                libtpu_rs_buffer_to_host(out_buf, host_out.as_mut_ptr(), 16).code,
                0
            );
            let f0 = f32::from_le_bytes([host_out[0], host_out[1], host_out[2], host_out[3]]);
            assert!((f0 - 3.0).abs() < 1e-5);

            // Test new libtpu_rs_* functions
            let mut rank_out: usize = 0;
            let mut queried_dims = [0_i64; 4];
            libtpu_rs_status_free(libtpu_rs_buffer_dimensions(
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::addr_of_mut!(rank_out),
            ));
            assert_eq!(
                libtpu_rs_buffer_dimensions(b0, ptr::null_mut(), 0, ptr::addr_of_mut!(rank_out))
                    .code,
                0
            );
            assert_eq!(rank_out, 1);
            assert_eq!(
                libtpu_rs_buffer_dimensions(
                    b0,
                    queried_dims.as_mut_ptr(),
                    4,
                    ptr::addr_of_mut!(rank_out),
                )
                .code,
                0
            );
            assert_eq!(queried_dims[0], 4);
            assert_eq!(libtpu_rs_buffer_element_type(ptr::null()), 0);
            assert_eq!(libtpu_rs_buffer_element_type(b0), 11);

            let mut copied_buf: *mut PJRT_Buffer = ptr::null_mut();
            libtpu_rs_status_free(libtpu_rs_buffer_copy_to_device(
                ptr::null_mut(),
                b0,
                1,
                ptr::addr_of_mut!(copied_buf),
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_copy_to_device(
                rs_client,
                b0,
                99,
                ptr::addr_of_mut!(copied_buf),
            ));
            assert_eq!(
                libtpu_rs_buffer_copy_to_device(rs_client, b0, 1, ptr::addr_of_mut!(copied_buf))
                    .code,
                0
            );
            libtpu_rs_buffer_destroy(copied_buf);

            let mut multi_outs: [*mut PJRT_Buffer; 1] = [ptr::null_mut()];
            libtpu_rs_status_free(libtpu_rs_execute_multi(
                ptr::null(),
                0,
                inputs.as_ptr(),
                2,
                multi_outs.as_mut_ptr(),
                1,
            ));
            libtpu_rs_status_free(libtpu_rs_execute_multi(
                exec,
                0,
                null_inputs.as_ptr(),
                1,
                multi_outs.as_mut_ptr(),
                1,
            ));
            libtpu_rs_status_free(libtpu_rs_execute_multi(
                exec,
                99,
                ptr::null(),
                0,
                multi_outs.as_mut_ptr(),
                1,
            ));
            assert_eq!(
                libtpu_rs_execute_multi(exec, 0, inputs.as_ptr(), 2, multi_outs.as_mut_ptr(), 1,)
                    .code,
                0
            );
            libtpu_rs_buffer_destroy(multi_outs[0]);

            let mut ser_len: usize = 0;
            libtpu_rs_status_free(libtpu_rs_executable_serialize(
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::addr_of_mut!(ser_len),
            ));
            assert_eq!(
                libtpu_rs_executable_serialize(
                    exec,
                    ptr::null_mut(),
                    0,
                    ptr::addr_of_mut!(ser_len)
                )
                .code,
                0
            );
            assert!(ser_len > 0);
            let mut ser_bytes = vec![0_u8; ser_len];
            assert_eq!(
                libtpu_rs_executable_serialize(
                    exec,
                    ser_bytes.as_mut_ptr(),
                    ser_len,
                    ptr::addr_of_mut!(ser_len),
                )
                .code,
                0
            );

            let mut deser_exec: *mut PJRT_LoadedExecutable = ptr::null_mut();
            libtpu_rs_status_free(libtpu_rs_executable_deserialize(
                ptr::null_mut(),
                ser_bytes.as_ptr(),
                ser_len,
                ptr::addr_of_mut!(deser_exec),
            ));
            libtpu_rs_status_free(libtpu_rs_executable_deserialize(
                rs_client,
                b"bad".as_ptr(),
                3,
                ptr::addr_of_mut!(deser_exec),
            ));
            assert_eq!(
                libtpu_rs_executable_deserialize(
                    rs_client,
                    ser_bytes.as_ptr(),
                    ser_len,
                    ptr::addr_of_mut!(deser_exec),
                )
                .code,
                0
            );
            libtpu_rs_executable_destroy(deser_exec);

            assert_eq!(
                libtpu_rs_executable_fingerprint(ptr::null(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_executable_fingerprint(exec, ptr::null_mut(), 0),
                32
            );
            let mut fp_buf = [0 as c_char; 40];
            assert_eq!(
                libtpu_rs_executable_fingerprint(exec, fp_buf.as_mut_ptr(), 0),
                32
            );
            assert_eq!(
                libtpu_rs_executable_fingerprint(exec, fp_buf.as_mut_ptr(), fp_buf.len()),
                32
            );
            assert_eq!(libtpu_rs_executable_flops(ptr::null()), 0.0);
            assert!(libtpu_rs_executable_flops(exec) > 0.0);

            (*out_buf).inner.delete().ok();
            libtpu_rs_status_free(libtpu_rs_buffer_to_host(out_buf, host_out.as_mut_ptr(), 16));

            // Test dynamic dimension indices on a buffer with dynamic dimensions
            if let Ok(dyn_shape) =
                Shape::new(ElementType::F32, vec![4, 8], Some(vec![false, true]), None)
            {
                let dyn_core_buf = libtpu_core::pjrt::PjrtCoreBuffer::new(
                    DeviceId(0),
                    libtpu_core::types::MemorySpaceId(0),
                    dyn_shape,
                    libtpu_core::types::DeviceAddress(0x1000),
                    std::sync::Arc::clone(&(*rs_client).inner.hal),
                );
                let dyn_buf = Box::into_raw(Box::new(PJRT_Buffer::new(
                    dyn_core_buf,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )));
                let mut dyn_args = crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    buffer: dyn_buf,
                    dynamic_dimension_indices: ptr::null(),
                    num_dynamic_dimensions: 0,
                };
                assert!(
                    PJRT_Buffer_DynamicDimensionIndices(ptr::addr_of_mut!(dyn_args).cast())
                        .is_null()
                );
                assert_eq!(dyn_args.num_dynamic_dimensions, 1);
                assert!(!dyn_args.dynamic_dimension_indices.is_null());
                libtpu_rs_buffer_destroy(dyn_buf);
            }

            // Test CreateErrorBuffer (with client & memory, with and without error_message)
            let mem0 = (*rs_client)
                .memory_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut());
            let dims_arr: [i64; 2] = [2, 2];
            let err_msg = b"custom buffer error";
            let mut err_buf_args = crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client,
                dims: dims_arr.as_ptr(),
                num_dims: 2,
                element_type: 11,
                error_code: 13,
                error_message: err_msg.as_ptr().cast::<c_char>(),
                error_message_size: err_msg.len(),
                memory: mem0,
                layout: ptr::null(),
                device: ptr::null_mut(),
                reserved: 0,
                reserved1: 0,
                buffer: ptr::null_mut(),
            };
            assert!(
                PJRT_Client_CreateErrorBuffer(ptr::addr_of_mut!(err_buf_args).cast()).is_null()
            );
            let mut ready_ev_args = crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args>(),
                extension_start: ptr::null_mut(),
                buffer: err_buf_args.buffer,
                event: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_ReadyEvent(ptr::addr_of_mut!(ready_ev_args).cast()).is_null());
            let mut ev_err_args = crate::pjrt_types::PJRT_Event_Error_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Error_Args>(),
                extension_start: ptr::null_mut(),
                event: ready_ev_args.event,
            };
            let ev_err = PJRT_Event_Error(ptr::addr_of_mut!(ev_err_args).cast());
            assert!(!ev_err.is_null());
            let mut err_destroy = crate::pjrt_types::PJRT_Error_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Error_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                error: ev_err,
            };
            PJRT_Error_Destroy(ptr::addr_of_mut!(err_destroy).cast());
            let mut ev_destroy = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ready_ev_args.event,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_destroy).cast()).is_null());
            libtpu_rs_buffer_destroy(err_buf_args.buffer);

            // Test CreateAliasBuffer + FulfillAliasBuffer (with client & null memory, then fulfill OK and Error)
            let mut alias_args = crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client,
                memory: ptr::null_mut(),
                dims: dims_arr.as_ptr(),
                num_dims: 2,
                element_type: 11,
                pad: 0,
                layout: ptr::null(),
                alias_buffer: ptr::null_mut(),
                fulfill_alias_buffer_cb: ptr::null_mut(),
            };
            assert!(PJRT_Client_CreateAliasBuffer(ptr::addr_of_mut!(alias_args).cast()).is_null());
            let mut fulfill_ok = crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client,
                buffer: b0,
                status_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
                fulfill_alias_buffer_cb: alias_args.fulfill_alias_buffer_cb,
            };
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            fulfill_ok.buffer = ptr::null_mut();
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            fulfill_ok.status_code = 13;
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            fulfill_ok.error_message = err_msg.as_ptr().cast::<c_char>();
            fulfill_ok.error_message_size = err_msg.len();
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            libtpu_rs_buffer_destroy(alias_args.alias_buffer);

            // Test UpdateGlobalProcessInfo (with client non-null, processes null and non-null)
            let proc_info = crate::pjrt_types::PJRT_ProcessInfo {
                struct_size: size_of::<crate::pjrt_types::PJRT_ProcessInfo>(),
                task_id: 0,
                state: 1,
                incarnation_id: 7,
                error_code: 0,
            };
            let mut proc_args = crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args>(
                ),
                extension_start: ptr::null_mut(),
                client: rs_client,
                processes: ptr::null(),
                num_processes: 0,
            };
            assert!(
                PJRT_Client_UpdateGlobalProcessInfo(ptr::addr_of_mut!(proc_args).cast()).is_null()
            );
            proc_args.processes = ptr::addr_of!(proc_info);
            proc_args.num_processes = 1;
            assert!(
                PJRT_Client_UpdateGlobalProcessInfo(ptr::addr_of_mut!(proc_args).cast()).is_null()
            );

            // Test MakeCanonicalShapeForMemorySpace with dims and layout non-null
            let mut out_layout = crate::pjrt_types::PJRT_Buffer_MemoryLayout {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_MemoryLayout>(),
                extension_start: ptr::null_mut(),
                minor_to_major: ptr::null(),
                minor_to_major_size: 0,
                tile_dims: ptr::null(),
                tile_dim_sizes: ptr::null(),
            };
            let mut canon_args =
                crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    topology: (*rs_client).topology_ptr,
                    dims: dims_arr.as_ptr(),
                    num_dims: 2,
                    element_type: 11,
                    memory_space_kind: 0,
                    layout: ptr::addr_of_mut!(out_layout),
                    reserved: 0,
                    reserved1: 0,
                    reserved2: 0,
                    reserved3: 0,
                };
            assert!(
                PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace(
                    ptr::addr_of_mut!(canon_args).cast()
                )
                .is_null()
            );
            assert!(canon_args.reserved > 0);
            assert_eq!(out_layout.minor_to_major_size, 2);

            // Test RawBuffer, Collectives, Shardings, and TpuExecutable extensions with real handles
            let mut raw_q = crate::extensions::PJRT_RawBuffer_Query_Args {
                struct_size: size_of::<crate::extensions::PJRT_RawBuffer_Query_Args>(),
                extension_start: ptr::null_mut(),
                buffer: b0.cast(),
                out_value: 0,
            };
            assert!(
                crate::extensions::PJRT_RawBuffer_GetHostPointer(ptr::addr_of_mut!(raw_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_RawBuffer_GetOnDeviceSizeInBytes(
                    ptr::addr_of_mut!(raw_q).cast()
                )
                .is_null()
            );
            assert!(
                crate::extensions::PJRT_RawBuffer_GetMemorySpace(ptr::addr_of_mut!(raw_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_RawBuffer_CreateRawAliasOfBuffer(
                    ptr::addr_of_mut!(raw_q).cast()
                )
                .is_null()
            );
            assert!(raw_q.out_value > 0);

            let mut buf_layout_q = crate::extensions::PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
                struct_size: size_of::<crate::extensions::PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>(
                ),
                extension_start: ptr::null_mut(),
                buffer: b0.cast(),
                layout: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_Layouts_Buffer_GetMemoryLayout(
                    ptr::addr_of_mut!(buf_layout_q).cast()
                )
                .is_null()
            );
            assert!(!buf_layout_q.layout.is_null());

            let phase_prog = b"module @phase { stablehlo.add }";
            let mut phase_run = crate::extensions::PJRT_PhaseCompile_RunPhase_Args {
                struct_size: size_of::<crate::extensions::PJRT_PhaseCompile_RunPhase_Args>(),
                extension_start: ptr::null_mut(),
                phase_compiler: ptr::null_mut(),
                input_bytes: phase_prog.as_ptr().cast(),
                input_size: phase_prog.len(),
                phase_name: c"hlo_verification".as_ptr(),
                output_status: 0,
            };
            assert!(
                crate::extensions::PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(phase_run).cast())
                    .is_null()
            );
            assert!(phase_run.output_status >= 6);

            let mut ag_args = crate::extensions::PJRT_Communicator_AllGather_Args {
                struct_size: size_of::<crate::extensions::PJRT_Communicator_AllGather_Args>(),
                extension_start: ptr::null_mut(),
                communicator: std::ptr::dangling_mut::<std::ffi::c_void>(),
                send_buffer: b0.cast(),
                recv_buffer: b1.cast(),
                count: 4,
                event: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag_args).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_Communicator_AllToAll(ptr::addr_of_mut!(ag_args).cast())
                    .is_null()
            );
            let mut cp_args = crate::extensions::PJRT_Communicator_CollectivePermute_Args {
                struct_size: size_of::<crate::extensions::PJRT_Communicator_CollectivePermute_Args>(
                ),
                extension_start: ptr::null_mut(),
                communicator: std::ptr::dangling_mut::<std::ffi::c_void>(),
                send_buffer: b0.cast(),
                recv_buffer: b1.cast(),
                target_ranks: ptr::null(),
                count: 4,
                event: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_Communicator_CollectivePermute(
                    ptr::addr_of_mut!(cp_args).cast()
                )
                .is_null()
            );

            let mut xh_args = crate::extensions::PJRT_CrossHostTransfers_Args {
                struct_size: size_of::<crate::extensions::PJRT_CrossHostTransfers_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client.cast(),
                buffers: b0 as usize,
                num_buffers: 1,
                remote_device: b1 as usize,
                out_event: 0,
            };
            assert!(
                crate::extensions::PJRT_CrossHostTransfers_CopyToRemoteDevice(
                    ptr::addr_of_mut!(xh_args).cast()
                )
                .is_null()
            );
            assert!(xh_args.out_event > 0);

            let exec_inner = (*exec).executable_ptr;
            let mut meta_q = crate::extensions::PJRT_Metadata_Get_Args {
                struct_size: size_of::<crate::extensions::PJRT_Metadata_Get_Args>(),
                extension_start: ptr::null_mut(),
                handle: exec_inner.cast(),
                metadata_out: 0,
            };
            assert!(
                crate::extensions::PJRT_ExecutableMetadata_Get(ptr::addr_of_mut!(meta_q).cast())
                    .is_null()
            );
            assert!(meta_q.metadata_out > 0);
            assert!(
                crate::extensions::PJRT_Layouts_Executable_GetOutputLayouts(
                    ptr::addr_of_mut!(meta_q).cast()
                )
                .is_null()
            );
            assert_eq!(meta_q.metadata_out, 1);

            let mut sh_args = crate::extensions::PJRT_Shardings_PJRT_Executable_Shardings_Args {
                struct_size: size_of::<
                    crate::extensions::PJRT_Shardings_PJRT_Executable_Shardings_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: exec_inner.cast(),
                num_items: 0,
                shardings: ptr::null(),
                sharding_sizes: ptr::null(),
            };
            assert!(
                crate::extensions::PJRT_Shardings_ParameterShardings(
                    ptr::addr_of_mut!(sh_args).cast()
                )
                .is_null()
            );
            assert_eq!(sh_args.num_items, 1);

            let mut hlo_q = crate::extensions::PJRT_TpuExecutable_ProtoQuery_Args {
                struct_size: size_of::<crate::extensions::PJRT_TpuExecutable_ProtoQuery_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec_inner.cast(),
                serialized_bytes: 0,
                serialized_bytes_size: 0,
                holder: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_TpuExecutable_GetHloModuleWithConfig(
                    ptr::addr_of_mut!(hlo_q).cast()
                )
                .is_null()
            );
            assert!(hlo_q.serialized_bytes_size > 0);

            let mut cost_q = crate::extensions::PJRT_Megascale_Query_Args {
                struct_size: size_of::<crate::extensions::PJRT_Megascale_Query_Args>(),
                extension_start: ptr::null_mut(),
                handle: exec_inner as usize,
                out_value: 0,
            };
            assert!(
                crate::extensions::PJRT_TpuExecutable_RunHloCostAnalysis(
                    ptr::addr_of_mut!(cost_q).cast()
                )
                .is_null()
            );
            assert!(cost_q.out_value >= 1);

            libtpu_rs_buffer_destroy(out_buf);
            libtpu_rs_buffer_destroy(b0);
            libtpu_rs_buffer_destroy(b1);
            libtpu_rs_buffer_destroy(ptr::null_mut());
            libtpu_rs_executable_destroy(exec);
            libtpu_rs_executable_destroy(ptr::null_mut());
            libtpu_rs_client_destroy(rs_client);
            libtpu_rs_client_destroy(ptr::null_mut());
        }
    }
}
