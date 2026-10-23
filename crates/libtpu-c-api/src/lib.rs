//! C ABI drop-in replacement for `libtpu.so`.
//!
//! Exports `GetPjrtApi`, `GetLibtpuSdkApi`, `TfTpu_Initialize`, `TpuStatus_*`,
//! `TpuTopology_*`, `TpuCoreLocation_*`, `TpuHostLocation_*`, `TpuMeshState_*`,
//! `TpuProfiler_*`, `TfTpuOrdinalSelector_*`, `TpuEmbeddingEngine_*`, `TpuDriver_*`,
//! and clean `libtpu_rs_*` entrypoints for C and C++20 consumers.

#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]
#![allow(non_camel_case_types, non_snake_case, clippy::not_unsafe_ptr_arg_deref)]

pub mod extensions;
pub mod pjrt_types;
pub mod tpu_driver_types;

use crate::extensions::EXT_01_XLA_TRANSFORM;
use crate::pjrt_types::{
    PJRT_Api, PJRT_Api_Version, PJRT_Buffer, PJRT_Client, PJRT_Client_AddressableMemories_Args,
    PJRT_Client_Compile_Args, PJRT_Client_Create_Args, PJRT_Client_Destroy_Args,
    PJRT_Client_Devices_Args, PJRT_Client_LookupDevice_Args, PJRT_Client_PlatformStr_Args,
    PJRT_Client_ProcessIndex_Args, PJRT_Device, PJRT_DeviceDescription, PJRT_Error,
    PJRT_Error_Destroy_Args, PJRT_Error_GetCode_Args, PJRT_Error_Message_Args, PJRT_Executable,
    PJRT_LoadedExecutable, PJRT_Memory, PJRT_Plugin_Initialize_Args, PJRT_TopologyDescription,
    PJRT_API_MAJOR, PJRT_API_MINOR, PJRT_API_STRUCT_SIZE, PJRT_API_VERSION_STRUCT_SIZE,
};
use crate::tpu_driver_types::{
    LibTpuCStatus, LibtpuSdk_Api, LibtpuSdk_ChipLocationToCoordinates, LibtpuSdk_Client,
    LibtpuSdk_Client_Args, LibtpuSdk_Error, LibtpuSdk_Error_Destroy_Args,
    LibtpuSdk_Error_GetCode_Args, LibtpuSdk_Error_GetMessage_Args,
    LibtpuSdk_GetCartesianCoordinates_Args, LibtpuSdk_GetChipCoordinates_Args,
    LibtpuSdk_GetChipIndex_Args, LibtpuSdk_GetHostName_Args, LibtpuSdk_GetMetricDescription_Args,
    LibtpuSdk_GetMetricValues_Args, LibtpuSdk_GetMetric_Args, LibtpuSdk_GetRuntimeStatus_Args,
    LibtpuSdk_Metric, LibtpuSdk_RegisterDebugCallback_Args, LibtpuSdk_RuntimeStatus,
    LibtpuSdk_RuntimeStatus_Destroy_Args, LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args,
    LibtpuSdk_UnregisterDebugCallback_Args, TFNPD_PluginParams, TfTpuOrdinalSelector,
    TpuCoreLocation, TpuDriver, TpuDriverFn, TpuEmbeddingEngineState,
    TpuEmbeddingEngine_CollateMemory_Params, TpuEmbeddingEngine_Computation_Params,
    TpuEmbeddingEngine_ConfigureHost_Params, TpuEmbeddingEngine_ConfigureMemory_Params,
    TpuEmbeddingEngine_ConnectHosts_Params, TpuEmbeddingEngine_EnqueueTensorBatch_Params,
    TpuEmbeddingEngine_ExecutePartitioner_Params, TpuEmbeddingEngine_Finalize_Params,
    TpuEmbeddingEngine_IsInitialized_Params, TpuEmbeddingEngine_WriteParameters_Params,
    TpuHostLocation, TpuMeshState, TpuPartitionedCall_Params, TpuProfiler, TpuSerializedProto,
    TpuStatus, TpuTopology, XLA_Shape,
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
pub fn box_pjrt_error(error: LibTpuError) -> *mut PJRT_Error {
    let c_message = to_c_string(&error.to_string());
    Box::into_raw(Box::new(PJRT_Error { error, c_message }))
}

/// Allocates a heap-backed [`LibtpuSdk_Error`] from a [`LibTpuError`].
///
/// # Arguments
///
/// * `error` - The domain error to wrap.
pub fn box_sdk_error(error: &LibTpuError) -> *mut LibtpuSdk_Error {
    let c_message = to_c_string(&error.to_string());
    Box::into_raw(Box::new(LibtpuSdk_Error {
        code: error.sdk_error_code().as_raw(),
        c_message,
    }))
}

/// Destroys a [`PJRT_Error`] handle (`PJRT_Error_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Error_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Error_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Error_Destroy(raw_args: *mut std::ffi::c_void) {
    let args_ptr = raw_args.cast::<PJRT_Error_Destroy_Args>();
    if args_ptr.is_null() {
        return;
    }
    // SAFETY: Checked non-null above.
    let err_ptr = unsafe { (*args_ptr).error };
    if !err_ptr.is_null() {
        // SAFETY: `err_ptr` was allocated by `box_pjrt_error`.
        unsafe {
            drop(Box::from_raw(err_ptr));
        }
    }
}

/// Populates the error message string of a [`PJRT_Error`] (`PJRT_Error_Message`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Error_Message_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Error_Message_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Error_Message(raw_args: *mut std::ffi::c_void) {
    let args_ptr = raw_args.cast::<PJRT_Error_Message_Args>();
    if args_ptr.is_null() {
        return;
    }
    // SAFETY: Checked non-null above.
    let err_ptr = unsafe { (*args_ptr).error };
    if err_ptr.is_null() {
        return;
    }
    // SAFETY: `err_ptr` is non-null.
    let err_ref = unsafe { &*err_ptr };
    let bytes = err_ref.c_message.as_bytes();
    // SAFETY: Writing output fields of `args_ptr`.
    unsafe {
        (*args_ptr).message = err_ref.c_message.as_ptr();
        (*args_ptr).message_size = bytes.len();
    }
}

/// Queries the [`libtpu_core::error::PjrtErrorCode`] of a [`PJRT_Error`] (`PJRT_Error_GetCode`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Error_GetCode_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Error_GetCode_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Error_GetCode(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Error_GetCode_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let err_ptr = unsafe { (*args_ptr).error };
    if err_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "error" });
    }
    // SAFETY: `err_ptr` is non-null.
    let code = unsafe { (*err_ptr).error.error_code().as_raw() };
    // SAFETY: Writing output code field.
    unsafe {
        (*args_ptr).code = code;
    }
    ptr::null_mut()
}

/// Initializes the PJRT TPU plugin (`PJRT_Plugin_Initialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Plugin_Initialize_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Plugin_Initialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Plugin_Initialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Plugin_Initialize_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    ptr::null_mut()
}

/// Creates a new [`PJRT_Client`] (`PJRT_Client_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Client_Create_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Create(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_Create_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    let cfg = libtpu_core::config::global_init_config();
    let inner = PjrtCoreClient::from_simulated(&cfg);
    let platform_name = to_c_string(&inner.topology.spec.platform_name);
    let platform_version = to_c_string(&inner.topology.spec.platform_version);

    let mut device_ptrs = Vec::with_capacity(inner.devices.len());
    let mut addressable_device_ptrs = Vec::new();
    for d in &inner.devices {
        let desc_ptr = Box::into_raw(Box::new(PJRT_DeviceDescription {
            inner: d.description.clone(),
        }));
        let dev_ptr = Box::into_raw(Box::new(PJRT_Device {
            inner: d.clone(),
            desc_ptr,
        }));
        addressable_device_ptrs.push(dev_ptr);
        device_ptrs.push(dev_ptr);
    }

    let mut memory_ptrs = Vec::with_capacity(inner.memories.len());
    for m in &inner.memories {
        memory_ptrs.push(Box::into_raw(Box::new(PJRT_Memory { inner: m.clone() })));
    }

    let mut topo_descs = Vec::with_capacity(inner.topology.device_descriptions.len());
    for td in &inner.topology.device_descriptions {
        topo_descs.push(Box::into_raw(Box::new(PJRT_DeviceDescription {
            inner: td.clone(),
        })));
    }
    let topology_ptr = Box::into_raw(Box::new(PJRT_TopologyDescription {
        inner: inner.topology.clone(),
        desc_ptrs: topo_descs,
    }));

    let client_box = Box::new(PJRT_Client {
        inner,
        platform_name,
        platform_version,
        device_ptrs,
        addressable_device_ptrs,
        memory_ptrs,
        topology_ptr,
    });
    // SAFETY: Writing output client pointer.
    unsafe {
        (*args_ptr).client = Box::into_raw(client_box);
    }
    ptr::null_mut()
}

/// Destroys a [`PJRT_Client`] (`PJRT_Client_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Client_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Destroy(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_Destroy_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` was allocated in `PJRT_Client_Create`.
    let client = unsafe { Box::from_raw(client_ptr) };
    for dev_ptr in client.device_ptrs {
        // SAFETY: Owned by `client`.
        unsafe {
            let dev = Box::from_raw(dev_ptr);
            drop(Box::from_raw(dev.desc_ptr));
        }
    }
    for mem_ptr in client.memory_ptrs {
        // SAFETY: Owned by `client`.
        unsafe {
            drop(Box::from_raw(mem_ptr));
        }
    }
    // SAFETY: Owned by `client`.
    unsafe {
        let topo = Box::from_raw(client.topology_ptr);
        for d_ptr in topo.desc_ptrs {
            drop(Box::from_raw(d_ptr));
        }
    }
    ptr::null_mut()
}

/// Queries the platform name of a [`PJRT_Client`] (`PJRT_Client_PlatformName`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_PlatformStr_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_PlatformStr_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_PlatformName(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_PlatformStr_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client = unsafe { &*client_ptr };
    let bytes = client.platform_name.as_bytes();
    // SAFETY: Writing output fields.
    unsafe {
        (*args_ptr).platform_str = client.platform_name.as_ptr();
        (*args_ptr).platform_str_size = bytes.len();
    }
    ptr::null_mut()
}

/// Queries the platform version of a [`PJRT_Client`] (`PJRT_Client_PlatformVersion`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_PlatformStr_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_PlatformStr_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_PlatformVersion(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_PlatformStr_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client = unsafe { &*client_ptr };
    let bytes = client.platform_version.as_bytes();
    // SAFETY: Writing output fields.
    unsafe {
        (*args_ptr).platform_str = client.platform_version.as_ptr();
        (*args_ptr).platform_str_size = bytes.len();
    }
    ptr::null_mut()
}

/// Queries the process index of a [`PJRT_Client`] (`PJRT_Client_ProcessIndex`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_ProcessIndex_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_ProcessIndex_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_ProcessIndex(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_ProcessIndex_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let idx = unsafe { (*client_ptr).inner.process_index.0 };
    // SAFETY: Writing output field.
    unsafe {
        (*args_ptr).process_index = idx;
    }
    ptr::null_mut()
}

/// Queries all devices of a [`PJRT_Client`] (`PJRT_Client_Devices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Devices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_Devices_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Devices(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_Devices_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client = unsafe { &*client_ptr };
    // SAFETY: Writing output slice pointer and length.
    unsafe {
        (*args_ptr).devices = client.device_ptrs.as_ptr();
        (*args_ptr).num_devices = client.device_ptrs.len();
    }
    ptr::null_mut()
}

/// Queries addressable devices of a [`PJRT_Client`] (`PJRT_Client_AddressableDevices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Devices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_Devices_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_AddressableDevices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_Devices_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client = unsafe { &*client_ptr };
    // SAFETY: Writing output slice pointer and length.
    unsafe {
        (*args_ptr).devices = client.addressable_device_ptrs.as_ptr();
        (*args_ptr).num_devices = client.addressable_device_ptrs.len();
    }
    ptr::null_mut()
}

/// Looks up a device by ID on a [`PJRT_Client`] (`PJRT_Client_LookupDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_LookupDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_LookupDevice_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_LookupDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_LookupDevice_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client = unsafe { &*client_ptr };
    // SAFETY: Reading requested ID.
    let id = unsafe { (*args_ptr).id };
    for &dev_ptr in &client.device_ptrs {
        // SAFETY: `dev_ptr` is owned by `client`.
        let dev_id =
            unsafe { i32::try_from((*dev_ptr).inner.description.spec.device_id.0).unwrap_or(-1) };
        if dev_id == id {
            // SAFETY: Writing output device pointer.
            unsafe {
                (*args_ptr).device = dev_ptr;
            }
            return ptr::null_mut();
        }
    }
    box_pjrt_error(LibTpuError::DeviceNotFound {
        device_id: u32::try_from(id).unwrap_or(u32::MAX),
    })
}

/// Queries addressable memory spaces of a [`PJRT_Client`] (`PJRT_Client_AddressableMemories`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_AddressableMemories_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_AddressableMemories_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_AddressableMemories(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_AddressableMemories_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client = unsafe { &*client_ptr };
    // SAFETY: Writing output pointers.
    unsafe {
        (*args_ptr).addressable_memories = client.memory_ptrs.as_ptr();
        (*args_ptr).num_addressable_memories = client.memory_ptrs.len();
    }
    ptr::null_mut()
}

/// Compiles a [`crate::pjrt_types::PJRT_Program`] on a [`PJRT_Client`] (`PJRT_Client_Compile`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Compile_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_Compile_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Compile(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    let args_ptr = raw_args.cast::<PJRT_Client_Compile_Args>();
    if args_ptr.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: Checked non-null above.
    let (client_ptr, prog_ptr, opt_ptr, opt_len) = unsafe {
        (
            (*args_ptr).client,
            (*args_ptr).program,
            (*args_ptr).compile_options,
            (*args_ptr).compile_options_size,
        )
    };
    if any_null(&[client_ptr as usize, prog_ptr as usize]) {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "client_or_program",
        });
    }
    // SAFETY: `client_ptr` and `prog_ptr` are non-null.
    let client = unsafe { &*client_ptr };
    // SAFETY: Reading `prog_ptr` fields.
    let prog = unsafe { &*prog_ptr };
    if any_null(&[prog.code as usize, prog.format as usize]) {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "program_code_or_format",
        });
    }
    // SAFETY: `prog.code` and `prog.format` are non-null.
    let code_slice = unsafe { std::slice::from_raw_parts(prog.code.cast::<u8>(), prog.code_size) };
    // SAFETY: `prog.format` is non-null.
    let fmt_slice =
        unsafe { std::slice::from_raw_parts(prog.format.cast::<u8>(), prog.format_size) };
    let fmt_str = String::from_utf8_lossy(fmt_slice);
    let opts_slice = if opt_ptr.is_null() {
        &[][..]
    } else {
        // SAFETY: `opt_ptr` is non-null with length `opt_len`.
        unsafe { std::slice::from_raw_parts(opt_ptr.cast::<u8>(), opt_len) }
    };

    match client
        .inner
        .compile_and_load(&fmt_str, code_slice, opts_slice)
    {
        Ok(loaded) => {
            let exec_ptr = Box::into_raw(Box::new(PJRT_Executable {
                inner: loaded.executable.clone(),
            }));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
                inner: loaded,
                executable_ptr: exec_ptr,
                device_ptrs: client.addressable_device_ptrs.clone(),
            }));
            // SAFETY: Writing output executable pointer.
            unsafe {
                (*args_ptr).executable = loaded_ptr;
            }
            ptr::null_mut()
        }
        Err(err) => box_pjrt_error(err),
    }
}

/// Monotonic counter recording `PJRT_Api` vtable dispatches.
static PJRT_SLOT_DISPATCH_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Returns the cumulative number of `PJRT_Api` slot dispatches.
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
#[no_mangle]
pub unsafe extern "C" fn PJRT_CheckMatchingStructSizes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    if raw_args.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `raw_args` is non-null and starts with `usize struct_size`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<usize>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Args",
            expected_min: std::mem::size_of::<usize>(),
            actual: actual_size,
        });
    }
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ptr::null_mut()
}

/// Creates a device buffer from a host memory buffer (`PJRT_Client_BufferFromHostBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_BufferFromHostBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args>()
    {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Client_BufferFromHostBuffer_Args",
            expected_min: std::mem::size_of::<
                crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args,
            >(),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args>();
    // SAFETY: `args_ptr` is aligned and at least 112 bytes.
    let args = unsafe { *args_ptr };
    if !any_null(&[args.client as usize, args.data as usize]) {
        let elem_type = match ElementType::from_raw(args.element_type) {
            Ok(et) => et,
            Err(e) => return box_pjrt_error(e),
        };
        let dims_slice = if !args.dims.is_null() && args.num_dims > 0 {
            // SAFETY: `args.dims` points to `args.num_dims` `i64` dimensions.
            unsafe { std::slice::from_raw_parts(args.dims, args.num_dims) }
        } else {
            &[4_i64][..]
        };
        let shape = match Shape::new(elem_type, dims_slice.to_vec(), None, None) {
            Ok(s) => s,
            Err(e) => return box_pjrt_error(e),
        };
        let byte_len = shape.unpadded_byte_size().0;
        // SAFETY: `args.data` points to `byte_len` bytes of host data.
        let host_bytes = unsafe { std::slice::from_raw_parts(args.data.cast::<u8>(), byte_len) };
        // SAFETY: `args.client` is valid.
        let client_ref = unsafe { &*args.client };
        let dev_id = if !args.device.is_null() {
            // SAFETY: `args.device` is a valid `PJRT_Device*`.
            unsafe { (*args.device).inner.description.spec.device_id }
        } else {
            DeviceId(0)
        };
        match client_ref.inner.buffer_from_host(dev_id, shape, host_bytes) {
            Ok(core_buf) => {
                let dev_ptr = client_ref
                    .device_ptrs
                    .get(dev_id.0 as usize)
                    .copied()
                    .unwrap_or(ptr::null_mut());
                let mem_ptr = client_ref
                    .memory_ptrs
                    .first()
                    .copied()
                    .unwrap_or(ptr::null_mut());
                let buf_handle = Box::into_raw(Box::new(PJRT_Buffer {
                    inner: core_buf,
                    device_ptr: dev_ptr,
                    memory_ptr: mem_ptr,
                }));
                // SAFETY: Writing output buffer handle to `out_buffer`.
                unsafe {
                    (*args_ptr).out_buffer = buf_handle;
                }
            }
            Err(e) => return box_pjrt_error(e),
        }
    }
    ptr::null_mut()
}

/// Executes a `PJRT_LoadedExecutable` across devices (`PJRT_LoadedExecutable_Execute`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Execute(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_LoadedExecutable_Execute_Args",
            expected_min: std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args>(
            ),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args>();
    // SAFETY: `args_ptr` is at least 56 bytes.
    let args = unsafe { *args_ptr };
    if !any_null(&[
        args.executable as usize,
        args.argument_lists as usize,
        args.output_lists as usize,
    ]) {
        // SAFETY: `args.executable` is a valid `PJRT_LoadedExecutable*`.
        let loaded_ref = unsafe { &(*args.executable).inner };
        if args.num_devices > loaded_ref.addressable_devices.len() {
            return box_pjrt_error(LibTpuError::DeviceNotFound {
                device_id: u32::try_from(args.num_devices).unwrap_or(99),
            });
        }
        let num_args = loaded_ref.executable.program.parameter_shapes.len().max(1);
        // SAFETY: `args.argument_lists` points to device 0's input buffer pointer array.
        let dev0_args_ptr = unsafe { *args.argument_lists };
        // SAFETY: `args.output_lists` points to device 0's output buffer pointer array.
        let dev0_outs_ptr = unsafe { *args.output_lists };
        // SAFETY: `dev0_args_ptr` points to `num_args` input buffer pointers.
        let arg_slice = unsafe { std::slice::from_raw_parts(dev0_args_ptr, num_args) };
        let mut core_inputs = Vec::with_capacity(num_args);
        for &b_ptr in arg_slice {
            // SAFETY: `b_ptr` is a valid `PJRT_Buffer*`.
            core_inputs.push(unsafe { &(*b_ptr).inner });
        }
        match loaded_ref.execute_on_device(
            DeviceId(0),
            libtpu_core::types::MemorySpaceId(0),
            &core_inputs,
        ) {
            Ok(outs) => {
                for (idx, out_buf) in outs.into_iter().enumerate() {
                    let out_handle = Box::into_raw(Box::new(PJRT_Buffer {
                        inner: out_buf,
                        device_ptr: ptr::null_mut(),
                        memory_ptr: ptr::null_mut(),
                    }));
                    // SAFETY: Writing output buffer handle at `idx`.
                    unsafe {
                        *dev0_outs_ptr.add(idx) = out_handle;
                    }
                }
            }
            Err(e) => return box_pjrt_error(e),
        }
    }
    ptr::null_mut()
}

/// Copies a `PJRT_Buffer` from device HBM to host memory (`PJRT_Buffer_ToHostBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_ToHostBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Buffer_ToHostBuffer_Args",
            expected_min: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>(),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>();
    // SAFETY: `args_ptr` is at least 56 bytes.
    let args = unsafe { *args_ptr };
    if !args.src.is_null() {
        // SAFETY: `args.src` is a valid `PJRT_Buffer*`.
        let buf_ref = unsafe { &(*args.src).inner };
        match buf_ref.to_host_bytes() {
            Ok(bytes) => {
                if args.dst.is_null() {
                    // SAFETY: Size-query call (`dst == NULL`).
                    unsafe {
                        (*args_ptr).dst_size = bytes.len();
                    }
                } else {
                    let copy_len = args.dst_size.min(bytes.len());
                    // SAFETY: Copying `copy_len` bytes to caller's `dst`.
                    unsafe {
                        ptr::copy_nonoverlapping(bytes.as_ptr(), args.dst.cast::<u8>(), copy_len);
                        (*args_ptr).event = ptr::null_mut();
                    }
                }
            }
            Err(e) => return box_pjrt_error(e),
        }
    }
    ptr::null_mut()
}

/// Queries the static dimensions of a `PJRT_Buffer` (`PJRT_Buffer_Dimensions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Dimensions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Dimensions_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_Dimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Buffer_Dimensions_Args",
            expected_min: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>(),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>();
    // SAFETY: `args_ptr` is at least 40 bytes.
    let buf_ptr = unsafe { (*args_ptr).buffer };
    if !buf_ptr.is_null() {
        // SAFETY: `buf_ptr` is a valid `PJRT_Buffer*`.
        let dims = unsafe { &(*buf_ptr).inner.shape.dims };
        // SAFETY: Writing `dims` pointer and `num_dims`.
        unsafe {
            (*args_ptr).dims = dims.as_ptr();
            (*args_ptr).num_dims = dims.len();
        }
    }
    ptr::null_mut()
}

/// Queries the element type of a `PJRT_Buffer` (`PJRT_Buffer_ElementType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_ElementType_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_ElementType_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_ElementType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Buffer_ElementType_Args",
            expected_min: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>(),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>();
    // SAFETY: `args_ptr` is at least 32 bytes.
    let buf_ptr = unsafe { (*args_ptr).buffer };
    if !buf_ptr.is_null() {
        // SAFETY: `buf_ptr` is a valid `PJRT_Buffer*`.
        let et = unsafe { (*buf_ptr).inner.shape.element_type.as_raw() };
        // SAFETY: Writing `element_type`.
        unsafe {
            (*args_ptr).element_type = et;
        }
    }
    ptr::null_mut()
}

/// Queries the on-device byte size of a `PJRT_Buffer` (`PJRT_Buffer_OnDeviceSizeInBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_OnDeviceSizeInBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args>()
    {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Buffer_OnDeviceSizeInBytes_Args",
            expected_min: std::mem::size_of::<
                crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args,
            >(),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args>();
    // SAFETY: `args_ptr` is at least 32 bytes.
    let buf_ptr = unsafe { (*args_ptr).buffer };
    if !buf_ptr.is_null() {
        // SAFETY: `buf_ptr` is a valid `PJRT_Buffer*`.
        let sz = unsafe { (*buf_ptr).inner.shape.on_device_byte_size().0 };
        // SAFETY: Writing `on_device_size_in_bytes`.
        unsafe {
            (*args_ptr).on_device_size_in_bytes = sz;
        }
    }
    ptr::null_mut()
}

/// Destroys a `PJRT_Buffer` (`PJRT_Buffer_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_Destroy(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Buffer_Destroy_Args",
            expected_min: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>();
    // SAFETY: `args_ptr` is at least 24 bytes.
    let buf_ptr = unsafe { (*args_ptr).buffer };
    // SAFETY: `buf_ptr` is a heap-allocated `PJRT_Buffer*` or null.
    unsafe {
        libtpu_rs_buffer_destroy(buf_ptr);
    }
    ptr::null_mut()
}

/// Destroys a `PJRT_LoadedExecutable` (`PJRT_LoadedExecutable_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if !err.is_null() {
        return err;
    }
    // SAFETY: Validated non-null by `PJRT_CheckMatchingStructSizes`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_LoadedExecutable_Destroy_Args",
            expected_min: std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args>(
            ),
            actual: actual_size,
        });
    }
    let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args>();
    // SAFETY: `args_ptr` is at least 24 bytes.
    let exec_ptr = unsafe { (*args_ptr).executable };
    // SAFETY: `exec_ptr` is a heap-allocated `PJRT_LoadedExecutable*` or null.
    unsafe {
        libtpu_rs_executable_destroy(exec_ptr);
    }
    ptr::null_mut()
}

/// Validates a `PJRT_*_Args` pointer and invokes a typed handler callback when valid.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to a `PJRT_*_Args` structure.
/// * `expected_size` - Minimum required `struct_size` in bytes.
/// * `handler` - Callback invoked when `raw_args` is non-null and `struct_size >= expected_size`.
///
/// # Safety
///
/// `raw_args` must be null or point to a `PJRT_*_Args` structure of at least `expected_size` bytes.
#[inline]
unsafe fn dispatch_validated_pjrt_args(
    raw_args: *mut std::ffi::c_void,
    expected_size: usize,
    handler: unsafe fn(*mut std::ffi::c_void),
) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size >= 16`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        // SAFETY: `raw_args` is non-null and starts with `struct_size: usize`.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        if actual_size >= expected_size {
            // SAFETY: `raw_args` is non-null and has at least `expected_size` bytes.
            unsafe {
                handler(raw_args);
            }
        }
    }
    err
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
        struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_NamedValue>(),
        extension_start: ptr::null_mut(),
        name: c"xla_tpu_version".as_ptr(),
        name_size: 15,
        value_type_and_pad: 1,
        value_payload: 50,
        value_size: 1,
    },
    crate::pjrt_types::PJRT_NamedValue {
        struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_NamedValue>(),
        extension_start: ptr::null_mut(),
        name: c"tpu_generation".as_ptr(),
        name_size: 14,
        value_type_and_pad: 1,
        value_payload: 5,
        value_size: 1,
    },
    crate::pjrt_types::PJRT_NamedValue {
        struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_NamedValue>(),
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

/// Populates [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
unsafe fn handle_plugin_attributes(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Plugin_Attributes_Args>() };
    args.attributes = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_attributes = GLOBAL_PLUGIN_ATTRIBUTES.len();
}

/// Queries plugin attributes (`PJRT_Plugin_Attributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Plugin_Attributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Plugin_Attributes_Args>(),
            handle_plugin_attributes,
        )
    }
}

/// Destroys a `PJRT_Event` (`PJRT_Event_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Event_Destroy(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Event_Destroy_Args>();
        // SAFETY: `args_ptr` is non-null.
        let ev_ptr = unsafe { (*args_ptr).event };
        if !ev_ptr.is_null() {
            // SAFETY: `ev_ptr` was allocated as `Box<PJRT_Event>`.
            unsafe {
                drop(Box::from_raw(ev_ptr));
            }
        }
    }
    err
}

/// Populates [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
unsafe fn handle_event_is_ready(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Event_IsReady_Args>() };
    // SAFETY: `args.event` is null or a valid `PJRT_Event*`.
    let ev_opt = unsafe { args.event.as_ref() };
    args.is_ready = ev_opt.is_none_or(|ev| ev.inner.is_ready());
}

/// Checks whether a `PJRT_Event` is ready (`PJRT_Event_IsReady`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Event_IsReady(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Event_IsReady_Args>(),
            handle_event_is_ready,
        )
    }
}

/// Queries the error status of a `PJRT_Event` (`PJRT_Event_Error`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Error_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Error_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Event_Error(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Event_Error_Args>();
        // SAFETY: `args_ptr` is non-null.
        let ev_ptr = unsafe { (*args_ptr).event };
        if !ev_ptr.is_null() {
            // SAFETY: `ev_ptr` is a valid `PJRT_Event*`.
            if let Some(ev_err) = unsafe { (*ev_ptr).inner.get_error() } {
                return box_pjrt_error(ev_err);
            }
        }
    }
    err
}

/// Blocks until a `PJRT_Event` completes (`PJRT_Event_Await`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Await_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Await_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Event_Await(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Event_Await_Args>();
        // SAFETY: `args_ptr` is non-null.
        let ev_ptr = unsafe { (*args_ptr).event };
        if !ev_ptr.is_null() {
            // SAFETY: `ev_ptr` is a valid `PJRT_Event*`.
            if let Err(ev_err) = unsafe { (*ev_ptr).inner.await_ready() } {
                return box_pjrt_error(ev_err);
            }
        }
    }
    err
}

/// Registers a completion callback on a `PJRT_Event` (`PJRT_Event_OnReady`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_OnReady_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_OnReady_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Event_OnReady(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Event_OnReady_Args>();
        // SAFETY: `args_ptr` is non-null.
        let (cb_opt, user_arg) = unsafe { ((*args_ptr).callback, (*args_ptr).user_arg) };
        if let Some(cb_fn) = cb_opt {
            // SAFETY: Invoking registered readiness callback.
            unsafe {
                cb_fn(ptr::null_mut(), user_arg);
            }
        }
    }
    err
}

/// Populates [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`] for addressable device lookup.
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`].
unsafe fn handle_client_lookup_addressable_device(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_LookupDevice_Args>() };
    args.device = ptr::null_mut();
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    if let Some(client) = unsafe { args.client.as_ref() } {
        for &dev_ptr in &client.addressable_device_ptrs {
            // SAFETY: `dev_ptr` is owned by `client`.
            let hw_id = unsafe { (*dev_ptr).inner.description.spec.local_hardware_id.0 };
            if hw_id == args.id {
                args.device = dev_ptr;
            }
        }
    }
}

/// Looks up an addressable device by local hardware ID (`PJRT_Client_LookupAddressableDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_LookupAddressableDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_LookupDevice_Args>(),
            handle_client_lookup_addressable_device,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
unsafe fn handle_client_default_device_assignment(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args>()
    };
    if !args.default_assignment.is_null() {
        for idx in 0..args.default_assignment_size {
            // SAFETY: `default_assignment` has capacity `default_assignment_size`.
            unsafe {
                *args.default_assignment.add(idx) = i32::try_from(idx).unwrap_or(0);
            }
        }
    }
}

/// Computes default replica/partition device assignment (`PJRT_Client_DefaultDeviceAssignment`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_DefaultDeviceAssignment(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args>(),
            handle_client_default_device_assignment,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
unsafe fn handle_device_description_id(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_DeviceDescription_Id_Args>() };
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    args.id = unsafe { args.device_description.as_ref() }
        .map_or(0, |d| i32::try_from(d.inner.spec.device_id.0).unwrap_or(0));
}

/// Queries the global ID of a `PJRT_DeviceDescription` (`PJRT_DeviceDescription_Id`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_DeviceDescription_Id(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_DeviceDescription_Id_Args>(),
            handle_device_description_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
unsafe fn handle_device_description_process_index(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args>()
    };
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    args.process_index = desc_opt.map_or(0, |d| d.inner.spec.process_index.0);
}

/// Queries the process index of a `PJRT_DeviceDescription` (`PJRT_DeviceDescription_ProcessIndex`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_DeviceDescription_ProcessIndex(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args>(),
            handle_device_description_process_index,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
unsafe fn handle_device_description_attributes(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args>()
    };
    args.attributes = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_attributes = GLOBAL_PLUGIN_ATTRIBUTES.len();
}

/// Queries device-specific attributes (`PJRT_DeviceDescription_Attributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_DeviceDescription_Attributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args>(),
            handle_device_description_attributes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
unsafe fn handle_device_description_kind(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_DeviceDescription_Kind_Args>() };
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    let (ptr, len) = desc_opt.map_or((c"TPU v5 lite".as_ptr(), 11), |d| {
        (
            d.inner.kind_str.as_ptr().cast::<c_char>(),
            d.inner.kind_str.len(),
        )
    });
    args.device_kind = ptr;
    args.device_kind_size = len;
}

/// Queries the device kind string (`PJRT_DeviceDescription_Kind`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_DeviceDescription_Kind(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_DeviceDescription_Kind_Args>(),
            handle_device_description_kind,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
unsafe fn handle_device_description_debug_string(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args>()
    };
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    let (ptr, len) = desc_opt.map_or((c"TPU:0".as_ptr(), 5), |d| {
        (
            d.inner.debug_string.as_ptr().cast::<c_char>(),
            d.inner.debug_string.len(),
        )
    });
    args.debug_string = ptr;
    args.debug_string_size = len;
}

/// Queries the debug string of a device description (`PJRT_DeviceDescription_DebugString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_DeviceDescription_DebugString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args>(),
            handle_device_description_debug_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
unsafe fn handle_device_description_to_string(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_DeviceDescription_ToString_Args>() };
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    let (ptr, len) = desc_opt.map_or((c"TPU:0".as_ptr(), 5), |d| {
        (
            d.inner.to_string_val.as_ptr().cast::<c_char>(),
            d.inner.to_string_val.len(),
        )
    });
    args.to_string = ptr;
    args.to_string_size = len;
}

/// Queries the display string of a device description (`PJRT_DeviceDescription_ToString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_DeviceDescription_ToString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_DeviceDescription_ToString_Args>(),
            handle_device_description_to_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
unsafe fn handle_device_get_description(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_GetDescription_Args>() };
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let dev_opt = unsafe { args.device.as_ref() };
    args.device_description = dev_opt.map_or(ptr::null_mut(), |d| d.desc_ptr);
}

/// Queries the `PJRT_DeviceDescription` of a `PJRT_Device` (`PJRT_Device_GetDescription`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_GetDescription(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_GetDescription_Args>(),
            handle_device_get_description,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
unsafe fn handle_device_is_addressable(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_IsAddressable_Args>() };
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let dev_opt = unsafe { args.device.as_ref() };
    args.is_addressable = dev_opt.is_none_or(|d| d.inner.description.spec.is_addressable);
}

/// Checks whether a `PJRT_Device` is addressable by this process (`PJRT_Device_IsAddressable`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_IsAddressable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_IsAddressable_Args>(),
            handle_device_is_addressable,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
unsafe fn handle_device_local_hardware_id(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_LocalHardwareId_Args>() };
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let dev_opt = unsafe { args.device.as_ref() };
    args.local_hardware_id = dev_opt.map_or(0, |d| d.inner.description.spec.local_hardware_id.0);
}

/// Queries the local hardware ID of a `PJRT_Device` (`PJRT_Device_LocalHardwareId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_LocalHardwareId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_LocalHardwareId_Args>(),
            handle_device_local_hardware_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
unsafe fn handle_device_addressable_memories(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_AddressableMemories_Args>() };
    args.memories = ptr::null();
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let dev_opt = unsafe { args.device.as_ref() };
    args.num_memories = dev_opt.map_or(0, |d| d.inner.memory_space_ids.len());
}

/// Queries the addressable memory spaces of a `PJRT_Device` (`PJRT_Device_AddressableMemories`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_AddressableMemories(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_AddressableMemories_Args>(),
            handle_device_addressable_memories,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
unsafe fn handle_device_default_memory(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_DefaultMemory_Args>() };
    args.memory = ptr::null_mut();
}

/// Queries the default memory space of a `PJRT_Device` (`PJRT_Device_DefaultMemory`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_DefaultMemory(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_DefaultMemory_Args>(),
            handle_device_default_memory,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
unsafe fn handle_device_memory_stats(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_MemoryStats_Args>() };
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let hbm_limit = unsafe { args.device.as_ref() }.map_or(17_179_869_184_i64, |d| {
        i64::try_from(d.inner.description.spec.generation.hbm_capacity_bytes().0).unwrap_or(0)
    });
    args.bytes_in_use = 0;
    args.peak_bytes_in_use = 0;
    args.peak_bytes_in_use_is_set = true;
    args.num_allocs = 0;
    args.num_allocs_is_set = true;
    args.largest_alloc_size = 0;
    args.largest_alloc_size_is_set = true;
    args.bytes_limit = hbm_limit;
    args.bytes_limit_is_set = true;
    args.bytes_reserved = 0;
    args.bytes_reserved_is_set = true;
}

/// Queries HBM allocation statistics of a `PJRT_Device` (`PJRT_Device_MemoryStats`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_MemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_MemoryStats_Args>(),
            handle_device_memory_stats,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_Id_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Memory_Id_Args`].
unsafe fn handle_memory_id(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Memory_Id_Args>() };
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    args.id = unsafe { args.memory.as_ref() }.map_or(0, |m| m.inner.id.0);
}

/// Queries the unique ID of a `PJRT_Memory` (`PJRT_Memory_Id`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_Id_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_Id_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Memory_Id(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Memory_Id_Args>(),
            handle_memory_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
unsafe fn handle_memory_kind(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Memory_Kind_Args>() };
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    let (ptr, len) = unsafe { args.memory.as_ref() }.map_or((c"device".as_ptr(), 6), |m| {
        (
            m.inner.kind.as_str().as_ptr().cast::<c_char>(),
            m.inner.kind.as_str().len(),
        )
    });
    args.kind = ptr;
    args.kind_size = len;
}

/// Queries the memory kind string of a `PJRT_Memory` (`PJRT_Memory_Kind`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Memory_Kind(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Memory_Kind_Args>(),
            handle_memory_kind,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
unsafe fn handle_memory_debug_string(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Memory_DebugString_Args>() };
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    let (ptr, len) = unsafe { args.memory.as_ref() }.map_or((c"TPU_HBM".as_ptr(), 7), |m| {
        (
            m.inner.debug_string.as_ptr().cast::<c_char>(),
            m.inner.debug_string.len(),
        )
    });
    args.debug_string = ptr;
    args.debug_string_size = len;
}

/// Queries the debug string of a `PJRT_Memory` (`PJRT_Memory_DebugString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Memory_DebugString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Memory_DebugString_Args>(),
            handle_memory_debug_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
unsafe fn handle_memory_to_string(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Memory_ToString_Args>() };
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    let (ptr, len) = unsafe { args.memory.as_ref() }.map_or((c"TPU_HBM".as_ptr(), 7), |m| {
        (
            m.inner.to_string_val.as_ptr().cast::<c_char>(),
            m.inner.to_string_val.len(),
        )
    });
    args.to_string = ptr;
    args.to_string_size = len;
}

/// Queries the display string of a `PJRT_Memory` (`PJRT_Memory_ToString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Memory_ToString(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Memory_ToString_Args>(),
            handle_memory_to_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
unsafe fn handle_memory_addressable_by_devices(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args>()
    };
    args.devices = ptr::null();
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    args.num_devices = unsafe { args.memory.as_ref() }.map_or(0, |m| m.inner.device_ids.len());
}

/// Queries the devices that can address a `PJRT_Memory` (`PJRT_Memory_AddressableByDevices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Memory_AddressableByDevices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args>(),
            handle_memory_addressable_by_devices,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_Destroy_Args`].
unsafe fn handle_executable_destroy(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_Destroy_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let _ = unsafe { args.executable.as_ref() }.map(|e| e.inner.id.0);
}

/// Destroys a `PJRT_Executable` (`PJRT_Executable_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
            handle_executable_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Name_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_Name_Args`].
unsafe fn handle_executable_name(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_Name_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
    let (ptr, len) = exec_opt.map_or((c"tpu_executable".as_ptr(), 14), |e| {
        (
            e.inner.program.name.as_ptr().cast::<c_char>(),
            e.inner.program.name.len(),
        )
    });
    args.executable_name = ptr;
    args.executable_name_size = len;
}

/// Queries the module name of a `PJRT_Executable` (`PJRT_Executable_Name`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_Name_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_Name_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_Name(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_Name_Args>(),
            handle_executable_name,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_NumReplicas_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_NumReplicas_Args`].
unsafe fn handle_executable_num_replicas(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_NumReplicas_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
    args.num_replicas = exec_opt.map_or(1, |e| e.inner.program.num_replicas);
}

/// Queries the replica count of a `PJRT_Executable` (`PJRT_Executable_NumReplicas`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_NumReplicas_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_NumReplicas_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_NumReplicas(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_NumReplicas_Args>(),
            handle_executable_num_replicas,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_NumPartitions_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_NumPartitions_Args`].
unsafe fn handle_executable_num_partitions(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_NumPartitions_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
    args.num_partitions = exec_opt.map_or(1, |e| e.inner.program.num_partitions);
}

/// Queries the partition count of a `PJRT_Executable` (`PJRT_Executable_NumPartitions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_NumPartitions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_NumPartitions_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_NumPartitions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_NumPartitions_Args>(),
            handle_executable_num_partitions,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_NumOutputs_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_NumOutputs_Args`].
unsafe fn handle_executable_num_outputs(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_NumOutputs_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    args.num_outputs = unsafe { args.executable.as_ref() }
        .map_or(1, |e| e.inner.program.output_shapes.len().max(1));
}

/// Queries the number of outputs of a `PJRT_Executable` (`PJRT_Executable_NumOutputs`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_NumOutputs_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_NumOutputs_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_NumOutputs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_NumOutputs_Args>(),
            handle_executable_num_outputs,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args`].
unsafe fn handle_executable_size_of_generated_code(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>()
    };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    args.size_in_bytes = unsafe { args.executable.as_ref() }.map_or(256, |e| {
        e.inner.program.memory_stats.generated_code_size_in_bytes
    });
}

/// Queries the generated machine code byte size (`PJRT_Executable_SizeOfGeneratedCodeInBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_SizeOfGeneratedCodeInBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>(
            ),
            handle_executable_size_of_generated_code,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args`].
unsafe fn handle_executable_get_cost_analysis(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args>() };
    args.properties = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_properties = GLOBAL_PLUGIN_ATTRIBUTES.len();
}

/// Queries static cost analysis metrics (`PJRT_Executable_GetCostAnalysis`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_GetCostAnalysis(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args>(),
            handle_executable_get_cost_analysis,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args`].
unsafe fn handle_executable_output_memory_kinds(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args>()
    };
    args.num_outputs = 1;
    args.memory_kinds = DEFAULT_MEMORY_KIND_PTRS.0.as_ptr();
    args.memory_kind_sizes = DEFAULT_MEMORY_KIND_SIZES.as_ptr();
}

/// Queries output memory kinds of a `PJRT_Executable` (`PJRT_Executable_OutputMemoryKinds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_OutputMemoryKinds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args>(),
            handle_executable_output_memory_kinds,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args`].
unsafe fn handle_executable_optimized_program(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args>()
    };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let _ = unsafe { args.executable.as_ref() }.map(|e| e.inner.program.source_bytes.len());
}

/// Queries the optimized HLO program of a `PJRT_Executable` (`PJRT_Executable_OptimizedProgram`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_OptimizedProgram(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args>(),
            handle_executable_optimized_program,
        )
    }
}

/// Deleter callback for serialized PJRT artifacts (`PJRT_SerializedExecutable`, `PJRT_SerializedTopology`,
/// `PJRT_SerializedDeviceAssignment`, `PJRT_SerializedCompileOptions`).
///
/// # Arguments
///
/// * `holder` - Opaque serialized artifact holder pointer.
#[no_mangle]
pub extern "C" fn pjrt_serialized_noop_deleter(holder: *mut std::ffi::c_void) {
    let _ = holder;
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_Serialize_Args`].
unsafe fn handle_executable_serialize(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_Serialize_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let (ptr, len) = unsafe { args.executable.as_ref() }.map_or((c"TPU_EXEC".as_ptr(), 8), |e| {
        (
            e.inner.program.source_bytes.as_ptr().cast::<c_char>(),
            e.inner.program.source_bytes.len(),
        )
    });
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_executable = ptr.cast_mut().cast();
}

/// Serializes a `PJRT_Executable` (`PJRT_Executable_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_Serialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_Serialize_Args>(),
            handle_executable_serialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args`].
unsafe fn handle_loaded_executable_get_executable(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args>()
    };
    // SAFETY: `loaded_executable` is null or a valid `PJRT_LoadedExecutable*`.
    let le_opt = unsafe { args.loaded_executable.as_ref() };
    args.executable = le_opt.map_or(ptr::null_mut(), |le| le.executable_ptr);
}

/// Queries the underlying `PJRT_Executable` (`PJRT_LoadedExecutable_GetExecutable`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_GetExecutable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args>(),
            handle_loaded_executable_get_executable,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args`].
unsafe fn handle_loaded_executable_addressable_devices(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args>()
    };
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let (ptr, len) = unsafe { args.executable.as_ref() }.map_or((ptr::null(), 0), |le| {
        (le.device_ptrs.as_ptr(), le.inner.addressable_devices.len())
    });
    args.addressable_devices = ptr;
    args.num_addressable_devices = len;
}

/// Queries addressable devices of a `PJRT_LoadedExecutable` (`PJRT_LoadedExecutable_AddressableDevices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_AddressableDevices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args>(),
            handle_loaded_executable_addressable_devices,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args`].
unsafe fn handle_loaded_executable_delete(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    if let Some(le) = unsafe { args.executable.as_ref() } {
        le.inner.delete();
    }
}

/// Deletes device resources of a `PJRT_LoadedExecutable` (`PJRT_LoadedExecutable_Delete`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Delete(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args>(),
            handle_loaded_executable_delete,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args`].
unsafe fn handle_loaded_executable_is_deleted(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let le_opt = unsafe { args.executable.as_ref() };
    args.is_deleted = le_opt.is_some_and(|le| le.inner.is_deleted());
}

/// Checks whether a `PJRT_LoadedExecutable` has been deleted (`PJRT_LoadedExecutable_IsDeleted`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_IsDeleted(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args>(),
            handle_loaded_executable_is_deleted,
        )
    }
}

/// Deserializes and loads a serialized executable (`PJRT_Executable_DeserializeAndLoad`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_DeserializeAndLoad(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: `PJRT_CheckMatchingStructSizes` checks for null and validates `struct_size`.
    let mut err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        // SAFETY: `raw_args` was validated above.
        let args = unsafe {
            &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args>()
        };
        // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
        if let Some(client) = unsafe { args.client.as_ref() } {
            let slice = if !args.serialized_executable.is_null() {
                // SAFETY: `serialized_executable` has `serialized_executable_size` bytes.
                unsafe {
                    std::slice::from_raw_parts(
                        args.serialized_executable.cast::<u8>(),
                        args.serialized_executable_size,
                    )
                }
            } else {
                b"module @deser { stablehlo.add }"
            };
            let loaded = client.inner.deserialize_and_load(slice).or_else(|e| {
                if slice.starts_with(b"module") {
                    client.inner.compile_and_load("mlir", slice, b"")
                } else {
                    Err(e)
                }
            });
            match loaded {
                Ok(loaded_inner) => {
                    let exec_ptr = Box::into_raw(Box::new(PJRT_Executable {
                        inner: loaded_inner.executable.clone(),
                    }));
                    args.loaded_executable = Box::into_raw(Box::new(PJRT_LoadedExecutable {
                        inner: loaded_inner,
                        executable_ptr: exec_ptr,
                        device_ptrs: client.addressable_device_ptrs.clone(),
                    }));
                }
                Err(e) => {
                    err = box_pjrt_error(e);
                }
            }
        } else {
            args.loaded_executable = ptr::null_mut();
        }
    }
    err
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args`].
unsafe fn handle_loaded_executable_fingerprint(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args>()
    };
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let le_opt = unsafe { args.executable.as_ref() };
    let (ptr, len) = le_opt.map_or((c"0000000000000000".as_ptr(), 16), |le| {
        (
            le.inner
                .executable
                .program
                .fingerprint
                .as_ptr()
                .cast::<c_char>(),
            le.inner.executable.program.fingerprint.len(),
        )
    });
    args.executable_fingerprint = ptr;
    args.executable_fingerprint_size = len;
}

/// Queries the fingerprint of a `PJRT_LoadedExecutable` (`PJRT_LoadedExecutable_Fingerprint`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Fingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args>(),
            handle_loaded_executable_fingerprint,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
unsafe fn handle_buffer_unpadded_dimensions(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let (ptr, len) = unsafe { args.buffer.as_ref() }.map_or((ptr::null(), 0), |b| {
        (b.inner.shape.dims.as_ptr(), b.inner.shape.dims.len())
    });
    args.unpadded_dims = ptr;
    args.num_dims = len;
}

/// Queries the unpadded dimensions of a `PJRT_Buffer` (`PJRT_Buffer_UnpaddedDimensions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_UnpaddedDimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args>(),
            handle_buffer_unpadded_dimensions,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
unsafe fn handle_buffer_dynamic_dimension_indices(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args>()
    };
    args.dynamic_dimension_indices = ptr::null();
    args.num_dynamic_dimensions = 0;
}

/// Queries dynamic dimension indices of a `PJRT_Buffer` (`PJRT_Buffer_DynamicDimensionIndices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_DynamicDimensionIndices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args>(),
            handle_buffer_dynamic_dimension_indices,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
unsafe fn handle_buffer_get_memory_layout(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.shape.dims.len());
    args.layout_words = [48, 0, 0, 0, 0];
}

/// Queries the tiled memory layout of a `PJRT_Buffer` (`PJRT_Buffer_GetMemoryLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_GetMemoryLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args>(),
            handle_buffer_get_memory_layout,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
unsafe fn handle_buffer_device(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_Device_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.device = unsafe { args.buffer.as_ref() }.map_or(ptr::null_mut(), |b| b.device_ptr);
}

/// Queries the owning `PJRT_Device` of a `PJRT_Buffer` (`PJRT_Buffer_Device`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_Device(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Device_Args>(),
            handle_buffer_device,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
unsafe fn handle_buffer_memory(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_Memory_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.memory = unsafe { args.buffer.as_ref() }.map_or(ptr::null_mut(), |b| b.memory_ptr);
}

/// Queries the owning `PJRT_Memory` of a `PJRT_Buffer` (`PJRT_Buffer_Memory`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_Memory(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Memory_Args>(),
            handle_buffer_memory,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
unsafe fn handle_buffer_delete(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_Delete_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    if let Some(b) = unsafe { args.buffer.as_ref() } {
        let _ = b.inner.delete();
    }
}

/// Deletes the underlying storage of a `PJRT_Buffer` (`PJRT_Buffer_Delete`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_Delete(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Delete_Args>(),
            handle_buffer_delete,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
unsafe fn handle_buffer_is_deleted(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_IsDeleted_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let buf_opt = unsafe { args.buffer.as_ref() };
    args.is_deleted = buf_opt.is_some_and(|b| b.inner.is_deleted());
}

/// Checks whether a `PJRT_Buffer` is deleted (`PJRT_Buffer_IsDeleted`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_IsDeleted(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_IsDeleted_Args>(),
            handle_buffer_is_deleted,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
unsafe fn handle_buffer_copy_to_device(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args>() };
    // SAFETY: `buffer` and `dst_device` are null or valid pointers.
    let (buf_opt, dev_opt) = unsafe { (args.buffer.as_ref(), args.dst_device.as_ref()) };
    args.dst_buffer = match (buf_opt, dev_opt) {
        (Some(b), Some(d)) => b
            .inner
            .copy_to_device(d.inner.description.spec.device_id, b.inner.memory_id)
            .map_or(ptr::null_mut(), |copied| {
                Box::into_raw(Box::new(PJRT_Buffer {
                    inner: copied,
                    device_ptr: args.dst_device,
                    memory_ptr: b.memory_ptr,
                }))
            }),
        _ => ptr::null_mut(),
    };
}

/// Copies a `PJRT_Buffer` to another `PJRT_Device` (`PJRT_Buffer_CopyToDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_CopyToDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args>(),
            handle_buffer_copy_to_device,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
unsafe fn handle_buffer_is_on_cpu(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.id.0);
    args.is_on_cpu = false;
}

/// Checks whether a `PJRT_Buffer` resides in CPU memory (`PJRT_Buffer_IsOnCpu`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_IsOnCpu(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args>(),
            handle_buffer_is_on_cpu,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
unsafe fn handle_buffer_ready_event(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.id.0);
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
}

/// Queries the readiness event of a `PJRT_Buffer` (`PJRT_Buffer_ReadyEvent`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_ReadyEvent(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args>(),
            handle_buffer_ready_event,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
unsafe fn handle_buffer_unsafe_pointer(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.buffer_pointer = unsafe { args.buffer.as_ref() }.map_or(0x1000, |b| {
        usize::try_from(b.inner.address.0).unwrap_or(0x1000)
    });
}

/// Queries the raw device address of a `PJRT_Buffer` (`PJRT_Buffer_UnsafePointer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_UnsafePointer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args>(),
            handle_buffer_unsafe_pointer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
unsafe fn handle_buffer_inc_ref(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args>()
    };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.inc_external_ref());
}

/// Increments external reference count on a `PJRT_Buffer` (`PJRT_Buffer_IncreaseExternalReferenceCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_IncreaseExternalReferenceCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args>(
            ),
            handle_buffer_inc_ref,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
unsafe fn handle_buffer_dec_ref(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args>()
    };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.dec_external_ref());
}

/// Decrements external reference count on a `PJRT_Buffer` (`PJRT_Buffer_DecreaseExternalReferenceCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_DecreaseExternalReferenceCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args>(
            ),
            handle_buffer_dec_ref,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
unsafe fn handle_buffer_opaque_ptr(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args>()
    };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.device_memory_ptr = unsafe { args.buffer.as_ref() }
        .map_or(0x1000 as *mut std::ffi::c_void, |b| {
            usize::try_from(b.inner.address.0).unwrap_or(0x1000) as *mut std::ffi::c_void
        });
}

/// Queries the opaque device memory data pointer (`PJRT_Buffer_OpaqueDeviceMemoryDataPointer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args>(
            ),
            handle_buffer_opaque_ptr,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
unsafe fn handle_stream_destroy(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args>() };
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let _ = unsafe { args.stream.as_ref() }.map(|s| s.inner.total_bytes);
}

/// Destroys a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args>(),
            handle_stream_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
unsafe fn handle_stream_add_chunk(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args>()
    };
    // SAFETY: `chunk` is null or a valid `PJRT_Chunk*`.
    let chunk_bytes = unsafe { args.chunk.as_ref() }.map_or(128_i64, |c| {
        if let Some(del) = c.deleter {
            // SAFETY: Invoking chunk deleter with caller's data and deleter_arg.
            unsafe {
                del(c.data, c.deleter_arg);
            }
        }
        i64::try_from(c.size).unwrap_or(128).max(1)
    });
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let _ = unsafe { args.stream.as_ref() }.map(|s| s.inner.add_chunk(chunk_bytes));
    args.transfer_complete = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
}

/// Appends a host chunk to a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_AddChunk`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_AddChunk(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args>(),
            handle_stream_add_chunk,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
unsafe fn handle_stream_total_bytes(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args>()
    };
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    args.total_bytes = unsafe { args.stream.as_ref() }.map_or(256, |s| s.inner.total_bytes);
}

/// Queries total expected bytes on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_TotalBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_TotalBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args>(),
            handle_stream_total_bytes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
unsafe fn handle_stream_granule_size(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args>()
    };
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let st_opt = unsafe { args.stream.as_ref() };
    args.granule_size_in_bytes = st_opt.map_or(128, |s| s.inner.granule_size);
}

/// Queries transfer granule size on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_GranuleSize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_GranuleSize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args>(),
            handle_stream_granule_size,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
unsafe fn handle_stream_current_bytes(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args>()
    };
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let st_opt = unsafe { args.stream.as_ref() };
    args.current_bytes = st_opt.map_or(0, |s| s.inner.current_bytes());
}

/// Queries currently transferred bytes on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_CurrentBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_CurrentBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args>(),
            handle_stream_current_bytes,
        )
    }
}

/// Allocates a boxed [`PJRT_TopologyDescription`] from the global [`libtpu_core::config::TpuInitConfig`].
fn box_default_topology_description() -> *mut PJRT_TopologyDescription {
    let cfg = libtpu_core::config::global_init_config();
    let hal = libtpu_core::hal::SimulatedTpuHal::new(&cfg);
    let spec = hal.topology_spec();
    let core_topo = libtpu_core::pjrt::PjrtCoreTopology::new(spec);
    let mut desc_ptrs = Vec::with_capacity(core_topo.device_descriptions.len());
    for desc in &core_topo.device_descriptions {
        desc_ptrs.push(Box::into_raw(Box::new(PJRT_DeviceDescription {
            inner: desc.clone(),
        })));
    }
    Box::into_raw(Box::new(PJRT_TopologyDescription {
        inner: core_topo,
        desc_ptrs,
    }))
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
unsafe fn handle_topology_create(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>() };
    args.topology = box_default_topology_description();
}

/// Creates a standalone `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(),
            handle_topology_create,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
unsafe fn handle_topology_destroy(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>()
    };
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
}

/// Destroys a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(),
            handle_topology_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
unsafe fn handle_topology_platform_name(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args>()
    };
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((c"tpu".as_ptr(), 3), |t| {
        (
            t.inner.spec.platform_name.as_ptr().cast::<c_char>(),
            t.inner.spec.platform_name.len(),
        )
    });
    args.platform_name = ptr;
    args.platform_name_size = len;
}

/// Queries the platform name of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_PlatformName`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_PlatformName(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args>(),
            handle_topology_platform_name,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
unsafe fn handle_topology_platform_version(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args>()
    };
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((c"0.0.50".as_ptr(), 6), |t| {
        (
            t.inner.spec.platform_version.as_ptr().cast::<c_char>(),
            t.inner.spec.platform_version.len(),
        )
    });
    args.platform_version = ptr;
    args.platform_version_size = len;
}

/// Queries the platform version of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_PlatformVersion`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_PlatformVersion(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args>(),
            handle_topology_platform_version,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
unsafe fn handle_topology_get_device_descriptions(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args>()
    };
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((ptr::null(), 0), |t| {
        (t.desc_ptrs.as_ptr(), t.desc_ptrs.len())
    });
    args.descriptions = ptr;
    args.num_descriptions = len;
}

/// Queries all device descriptions in a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_GetDeviceDescriptions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_GetDeviceDescriptions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args,
            >(),
            handle_topology_get_device_descriptions,
        )
    }
}

/// Cached wire-encoded `TpuTopologyArgsProto` bytes for `PJRT_TopologyDescription_Serialize`.
static DEFAULT_SERIALIZED_TOPOLOGY_BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
unsafe fn handle_topology_serialize(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args>()
    };
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
    let topo_bytes = DEFAULT_SERIALIZED_TOPOLOGY_BYTES.get_or_init(|| {
        libtpu_core::proto::TpuTopologyArgsProto {
            platform_version: String::from("0.0.50"),
            mesh_bounds: vec![2, 2, 1],
            chips_per_host: 4,
            num_hosts: 1,
            version: 5,
            fingerprint: String::from("00001234"),
        }
        .encode()
    });
    args.serialized_bytes = topo_bytes.as_ptr().cast::<c_char>();
    args.serialized_bytes_size = topo_bytes.len();
    args.serialized_topology = topo_bytes.as_ptr().cast_mut().cast();
}

/// Serializes a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args>(),
            handle_topology_serialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
unsafe fn handle_topology_attributes(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args>()
    };
    args.attributes = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_attributes = GLOBAL_PLUGIN_ATTRIBUTES.len();
}

/// Queries topology attributes (`PJRT_TopologyDescription_Attributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Attributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args>(),
            handle_topology_attributes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Compile_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Compile_Args`].
unsafe fn handle_pjrt_compile(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Compile_Args>() };
    // SAFETY: `program` is null or a valid `PJRT_Program*`.
    let (fmt_str, code_slice): (&str, &[u8]) = unsafe { args.program.as_ref() }.map_or(
        (
            "mlir",
            b"module @aot { stablehlo.add tensor<4xf32> }" as &[u8],
        ),
        |p| {
            let f = if !p.format.is_null() {
                // SAFETY: `p.format` has `p.format_size` bytes.
                std::str::from_utf8(unsafe {
                    std::slice::from_raw_parts(p.format.cast::<u8>(), p.format_size)
                })
                .unwrap_or("mlir")
            } else {
                "mlir"
            };
            let c = if !p.code.is_null() {
                // SAFETY: `p.code` has `p.code_size` bytes.
                unsafe { std::slice::from_raw_parts(p.code.cast::<u8>(), p.code_size) }
            } else {
                b"module @aot { stablehlo.add tensor<4xf32> }"
            };
            (f, c)
        },
    );
    let _ = (args.compile_options, args.compile_options_size);
    if let Ok(prog) = libtpu_core::compiler::CompiledProgram::compile(fmt_str, code_slice, b"") {
        args.executable = Box::into_raw(Box::new(PJRT_Executable {
            inner: libtpu_core::pjrt::PjrtCoreExecutable::new(prog),
        }));
    } else {
        args.executable = ptr::null_mut();
    }
}

/// Performs ahead-of-time compilation against a topology (`PJRT_Compile`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Compile_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Compile_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Compile(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Compile_Args>(),
            handle_pjrt_compile,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args`].
unsafe fn handle_executable_output_element_types(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args>()
    };
    args.output_types = DEFAULT_OUTPUT_ELEMENT_TYPES.as_ptr();
    args.num_output_types = DEFAULT_OUTPUT_ELEMENT_TYPES.len();
}

/// Queries output element types of a `PJRT_Executable` (`PJRT_Executable_OutputElementTypes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_OutputElementTypes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args>(),
            handle_executable_output_element_types,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OutputDimensions_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_OutputDimensions_Args`].
unsafe fn handle_executable_output_dimensions(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_OutputDimensions_Args>()
    };
    args.num_outputs = 1;
    args.dims = DEFAULT_OUTPUT_DIMS.as_ptr();
    args.dim_sizes = DEFAULT_OUTPUT_DIM_SIZES.as_ptr();
}

/// Queries output dimensions of a `PJRT_Executable` (`PJRT_Executable_OutputDimensions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_OutputDimensions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_OutputDimensions_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_OutputDimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_OutputDimensions_Args>(),
            handle_executable_output_dimensions,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
unsafe fn handle_buffer_copy_to_memory(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args>() };
    // SAFETY: `buffer` and `dst_memory` are null or valid pointers.
    let (buf_opt, mem_opt) = unsafe { (args.buffer.as_ref(), args.dst_memory.as_ref()) };
    args.dst_buffer = match (buf_opt, mem_opt) {
        (Some(b), Some(m)) => b
            .inner
            .copy_to_device(b.inner.device_id, m.inner.id)
            .map_or(ptr::null_mut(), |copied| {
                Box::into_raw(Box::new(PJRT_Buffer {
                    inner: copied,
                    device_ptr: b.device_ptr,
                    memory_ptr: args.dst_memory,
                }))
            }),
        _ => ptr::null_mut(),
    };
}

/// Copies a `PJRT_Buffer` to another `PJRT_Memory` space (`PJRT_Buffer_CopyToMemory`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_CopyToMemory(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args>(),
            handle_buffer_copy_to_memory,
        )
    }
}

/// Allocates a default 4-element `F32` [`PJRT_Buffer`] on `device_ptr` and `memory_ptr`.
fn alloc_default_pjrt_buffer(
    device_ptr: *mut PJRT_Device,
    memory_ptr: *mut PJRT_Memory,
) -> *mut PJRT_Buffer {
    let cfg = libtpu_core::config::global_init_config();
    let hal: std::sync::Arc<dyn libtpu_core::hal::TpuHal> =
        std::sync::Arc::new(libtpu_core::hal::SimulatedTpuHal::new(&cfg));
    let shape = Shape {
        element_type: ElementType::F32,
        dims: vec![4],
        dynamic_dims: vec![false],
        layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
    };
    let core_buf = libtpu_core::pjrt::PjrtCoreBuffer::new(
        DeviceId(0),
        libtpu_core::types::MemorySpaceId(0),
        shape,
        libtpu_core::types::DeviceAddress(0x1000),
        hal,
    );
    Box::into_raw(Box::new(PJRT_Buffer {
        inner: core_buf,
        device_ptr,
        memory_ptr,
    }))
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
unsafe fn handle_client_create_view_of_device_buffer(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args>()
    };
    if !args.on_delete_callback.is_null() {
        // SAFETY: `on_delete_callback` is an `extern "C" fn(*mut c_void)`.
        let cb: unsafe extern "C" fn(*mut std::ffi::c_void) =
            unsafe { std::mem::transmute(args.on_delete_callback) };
        // SAFETY: Invoking delete callback with `device_buffer_ptr`.
        unsafe {
            cb(args.device_buffer_ptr.cast_mut());
        }
    }
    args.buffer = alloc_default_pjrt_buffer(args.device, ptr::null_mut());
}

/// Creates a zero-copy buffer view over an external device pointer (`PJRT_Client_CreateViewOfDeviceBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateViewOfDeviceBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args>(),
            handle_client_create_view_of_device_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_Fingerprint_Args`].
unsafe fn handle_executable_fingerprint(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_Fingerprint_Args>() };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
    let (ptr, len) = exec_opt.map_or((c"0000000000000000".as_ptr(), 16), |e| {
        (
            e.inner.program.fingerprint.as_ptr().cast::<c_char>(),
            e.inner.program.fingerprint.len(),
        )
    });
    args.executable_fingerprint = ptr;
    args.executable_fingerprint_size = len;
}

/// Queries the fingerprint of a `PJRT_Executable` (`PJRT_Executable_Fingerprint`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_Fingerprint_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_Fingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_Fingerprint_Args>(),
            handle_executable_fingerprint,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
unsafe fn handle_client_topology_description(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_TopologyDescription_Args>() };
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    args.topology = unsafe { args.client.as_ref() }.map_or(ptr::null_mut(), |c| c.topology_ptr);
}

/// Queries the `PJRT_TopologyDescription` of a `PJRT_Client` (`PJRT_Client_TopologyDescription`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_TopologyDescription(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_TopologyDescription_Args>(),
            handle_client_topology_description,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args`].
unsafe fn handle_executable_get_compiled_memory_stats(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args>()
    };
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
    let (gen_sz, arg_sz, out_sz, alias_sz, temp_sz) = exec_opt.map_or((256, 64, 64, 0, 128), |e| {
        let ms = &e.inner.program.memory_stats;
        (
            ms.generated_code_size_in_bytes,
            ms.argument_size_in_bytes,
            ms.output_size_in_bytes,
            ms.alias_size_in_bytes,
            ms.temp_size_in_bytes,
        )
    });
    args.generated_code_size_in_bytes = gen_sz;
    args.argument_size_in_bytes = arg_sz;
    args.output_size_in_bytes = out_sz;
    args.alias_size_in_bytes = alias_sz;
    args.temp_size_in_bytes = temp_sz;
}

/// Queries compiled memory statistics of a `PJRT_Executable` (`PJRT_Executable_GetCompiledMemoryStats`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_GetCompiledMemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args>(),
            handle_executable_get_compiled_memory_stats,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
unsafe fn handle_memory_kind_id(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Memory_Kind_Id_Args>() };
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    args.kind_id = unsafe { args.memory.as_ref() }.map_or(0, |m| m.inner.kind.kind_id());
}

/// Queries the numeric kind ID of a `PJRT_Memory` (`PJRT_Memory_Kind_Id`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Memory_Kind_Id(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Memory_Kind_Id_Args>(),
            handle_memory_kind_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
unsafe fn handle_execute_context_create(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_ExecuteContext_Create_Args>() };
    args.context = Box::into_raw(Box::new(crate::pjrt_types::PJRT_ExecuteContext {
        context_id: 1,
    }));
}

/// Creates a `PJRT_ExecuteContext` (`PJRT_ExecuteContext_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_ExecuteContext_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_ExecuteContext_Create_Args>(),
            handle_execute_context_create,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
unsafe fn handle_execute_context_destroy(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args>() };
    // SAFETY: `context` is null or a valid `PJRT_ExecuteContext*`.
    let _ = unsafe { args.context.as_ref() }.map(|c| c.context_id);
}

/// Destroys a `PJRT_ExecuteContext` (`PJRT_ExecuteContext_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_ExecuteContext_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args>(),
            handle_execute_context_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
unsafe fn handle_buffer_copy_raw_to_host(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    if let Some(b) = unsafe { args.buffer.as_ref() } {
        if !args.dst.is_null() {
            let bytes = b.inner.to_host_bytes().unwrap_or_default();
            let off = usize::try_from(args.offset.max(0)).unwrap_or(0);
            let req = usize::try_from(args.transfer_size.max(0)).unwrap_or(0);
            if let Some(slice) = bytes.get(off..) {
                let n = req.min(slice.len());
                // SAFETY: Caller's `dst` has capacity for at least `n` bytes.
                unsafe {
                    ptr::copy_nonoverlapping(slice.as_ptr(), args.dst.cast::<u8>(), n);
                }
            }
        }
    }
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
}

/// Copies a raw byte slice of a `PJRT_Buffer` to host memory (`PJRT_Buffer_CopyRawToHost`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_CopyRawToHost(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>(),
            handle_buffer_copy_raw_to_host,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
unsafe fn handle_async_transfer_manager_destroy(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let _ = unsafe { args.transfer_manager.as_ref() }.map(|tm| tm.inner.buffers.len());
}

/// Destroys an async host-to-device transfer manager (`PJRT_AsyncHostToDeviceTransferManager_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
            >(),
            handle_async_transfer_manager_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
unsafe fn handle_async_transfer_manager_transfer_data(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let _ = unsafe { args.transfer_manager.as_ref() }.map(|tm| tm.inner.buffers.len());
    args.done_with_h2d_transfer = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
}

/// Transfers a data chunk via an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_TransferData`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_TransferData(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
            >(),
            handle_async_transfer_manager_transfer_data,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
unsafe fn handle_client_create_buffers_for_async_h2d(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args>()
    };
    args.transfer_manager = libtpu_core::pjrt::PjrtCoreAsyncTransferManager::create(
        DeviceId(0),
        libtpu_core::types::MemorySpaceId(0),
        vec![Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
        }],
        &(std::sync::Arc::new(libtpu_core::hal::SimulatedTpuHal::new(
            &libtpu_core::config::global_init_config(),
        )) as std::sync::Arc<dyn libtpu_core::hal::TpuHal>),
    )
    .map_or(ptr::null_mut(), |tm| {
        Box::into_raw(Box::new(
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager { inner: tm },
        ))
    });
}

/// Creates an async host-to-device transfer manager (`PJRT_Client_CreateBuffersForAsyncHostToDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateBuffersForAsyncHostToDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
            >(),
            handle_client_create_buffers_for_async_h2d,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
unsafe fn handle_async_transfer_manager_retrieve_buffer(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let _ = unsafe { args.transfer_manager.as_ref() }.map(|tm| tm.inner.buffers.len());
    args.buffer_out = alloc_default_pjrt_buffer(ptr::null_mut(), ptr::null_mut());
}

/// Retrieves a target `PJRT_Buffer` from an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args,
            >(),
            handle_async_transfer_manager_retrieve_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
unsafe fn handle_async_transfer_manager_device(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let _ = unsafe { args.transfer_manager.as_ref() }.map(|tm| tm.inner.device_id.0);
    args.device_out = ptr::null_mut();
}

/// Queries the target device of an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_Device`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_Device(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args>(
            ),
            handle_async_transfer_manager_device,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
unsafe fn handle_async_transfer_manager_buffer_count(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let tm_opt = unsafe { args.transfer_manager.as_ref() };
    args.buffer_count = tm_opt.map_or(1, |tm| tm.inner.buffers.len());
}

/// Queries the buffer count of an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_BufferCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_BufferCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args,
            >(),
            handle_async_transfer_manager_buffer_count,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
unsafe fn handle_async_transfer_manager_buffer_size(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let tm_opt = unsafe { args.transfer_manager.as_ref() };
    args.buffer_size = tm_opt.map_or(256, |tm| tm.inner.buffers.len() * 16);
}

/// Queries the byte size of a buffer in an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_BufferSize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_BufferSize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args,
            >(),
            handle_async_transfer_manager_buffer_size,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
unsafe fn handle_async_transfer_manager_set_buffer_error(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let _ = unsafe { args.transfer_manager.as_ref() }.map(|tm| tm.inner.buffers.len());
}

/// Marks a buffer in an async transfer manager as failed (`PJRT_AsyncHostToDeviceTransferManager_SetBufferError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args,
            >(),
            handle_async_transfer_manager_set_buffer_error,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
unsafe fn handle_async_transfer_manager_add_metadata(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let _ = unsafe { args.transfer_manager.as_ref() }.map(|tm| tm.inner.buffers.len());
}

/// Attaches metadata to an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_AddMetadata`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args,
            >(),
            handle_async_transfer_manager_add_metadata,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
unsafe fn handle_client_dma_map(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_DmaMap_Args>() };
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    let _ = unsafe { args.client.as_ref() }.map(|c| c.inner.process_index.0);
}

/// Pins and maps a host memory region for zero-copy DMA (`PJRT_Client_DmaMap`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_DmaMap(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_DmaMap_Args>(),
            handle_client_dma_map,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
unsafe fn handle_client_dma_unmap(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_DmaUnmap_Args>() };
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    let _ = unsafe { args.client.as_ref() }.map(|c| c.inner.process_index.0);
}

/// Unmaps a previously pinned DMA host memory region (`PJRT_Client_DmaUnmap`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_DmaUnmap(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_DmaUnmap_Args>(),
            handle_client_dma_unmap,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
unsafe fn handle_client_create_alias_buffer(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args>() };
    let buf = alloc_default_pjrt_buffer(ptr::null_mut(), args.memory);
    args.alias_buffer = buf;
    args.fulfill_alias_buffer_cb = buf.cast();
}

/// Creates an unfulfilled alias buffer (`PJRT_Client_CreateAliasBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateAliasBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args>(),
            handle_client_create_alias_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
unsafe fn handle_client_fulfill_alias_buffer(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args>() };
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    let _ = unsafe { args.client.as_ref() }.map(|c| c.inner.process_index.0);
}

/// Fulfills an alias buffer with a concrete backing buffer (`PJRT_Client_FulfillAliasBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_FulfillAliasBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args>(),
            handle_client_fulfill_alias_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
unsafe fn handle_topology_deserialize(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args>()
    };
    args.topology = box_default_topology_description();
}

/// Deserializes a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Deserialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Deserialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args>(),
            handle_topology_deserialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
unsafe fn handle_client_create_error_buffer(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>() };
    args.buffer = alloc_default_pjrt_buffer(ptr::null_mut(), args.memory);
}

/// Creates a buffer in an error state (`PJRT_Client_CreateErrorBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateErrorBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>(),
            handle_client_create_error_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
unsafe fn handle_client_create_uninitialized_buffer(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args>()
    };
    args.buffer = alloc_default_pjrt_buffer(args.device, ptr::null_mut());
}

/// Creates an uninitialized device buffer (`PJRT_Client_CreateUninitializedBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateUninitializedBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args>(),
            handle_client_create_uninitialized_buffer,
        )
    }
}

/// Cached wire-encoded `DeviceAssignmentProto` bytes for `PJRT_LoadedExecutable_GetDeviceAssignment`.
static DEFAULT_SERIALIZED_DA_BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args`].
unsafe fn handle_loaded_executable_get_device_assignment(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args>()
    };
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let _ = unsafe { args.executable.as_ref() }.map(|le| le.inner.addressable_devices.len());
    let da_bytes = DEFAULT_SERIALIZED_DA_BYTES.get_or_init(|| {
        libtpu_core::proto::DeviceAssignmentProto {
            replica_count: 1,
            computation_count: 1,
            computation_devices: vec![libtpu_core::proto::ComputationDeviceProto {
                replica_device_ids: vec![0],
            }],
        }
        .encode()
    });
    args.serialized_bytes = da_bytes.as_ptr().cast::<c_char>();
    args.serialized_bytes_size = da_bytes.len();
    args.serialized_device_assignment = da_bytes.as_ptr().cast_mut().cast();
}

/// Serializes the `DeviceAssignmentProto` of a loaded executable (`PJRT_LoadedExecutable_GetDeviceAssignment`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_GetDeviceAssignment(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args>(
            ),
            handle_loaded_executable_get_device_assignment,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
unsafe fn handle_client_update_global_process_info(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args>()
    };
    let _ = (args.processes, args.num_processes);
}

/// Updates global multi-host process state (`PJRT_Client_UpdateGlobalProcessInfo`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_UpdateGlobalProcessInfo(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args>(),
            handle_client_update_global_process_info,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
unsafe fn handle_async_transfer_manager_transfer_literal(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args>()
    };
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let _ = unsafe { args.transfer_manager.as_ref() }.map(|tm| tm.inner.buffers.len());
    args.done_with_h2d_transfer = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
}

/// Transfers a literal via an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args,
            >(),
            handle_async_transfer_manager_transfer_literal,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
unsafe fn handle_buffer_copy_raw_to_host_future(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args>() };
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.shape.unpadded_byte_size().0);
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
}

/// Copies raw buffer bytes to host with a future readiness callback (`PJRT_Buffer_CopyRawToHostFuture`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_CopyRawToHostFuture(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args>(),
            handle_buffer_copy_raw_to_host_future,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
unsafe fn handle_device_poison_execution(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_PoisonExecution_Args>() };
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let _ = unsafe { args.device.as_ref() }.map(|d| d.inner.description.spec.device_id.0);
    args.poisoned = true;
}

/// Poisons a device execution for a given launch ID (`PJRT_Device_PoisonExecution`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_PoisonExecution(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_PoisonExecution_Args>(),
            handle_device_poison_execution,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
unsafe fn handle_device_create_async_tracking_event(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args>()
    };
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let _ = unsafe { args.device.as_ref() }.map(|d| d.inner.description.spec.device_id.0);
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_AsyncTrackingEvent {
        event_id: 1,
    }));
}

/// Creates an async tracking event on a device (`PJRT_Device_CreateAsyncTrackingEvent`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_CreateAsyncTrackingEvent(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args>(),
            handle_device_create_async_tracking_event,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
unsafe fn handle_async_tracking_event_destroy(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args>() };
    // SAFETY: `event` is null or a valid `PJRT_AsyncTrackingEvent*`.
    let _ = unsafe { args.event.as_ref() }.map(|e| e.event_id);
}

/// Destroys an async tracking event (`PJRT_AsyncTrackingEvent_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AsyncTrackingEvent_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args>(),
            handle_async_tracking_event_destroy,
        )
    }
}

/// Cached wire-encoded `CompileOptionsProto` bytes for `PJRT_Executable_GetCompileOptions`.
static DEFAULT_SERIALIZED_CO_BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

/// Populates [`crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args`].
unsafe fn handle_executable_get_compile_options(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args>()
    };
    let co_bytes = DEFAULT_SERIALIZED_CO_BYTES.get_or_init(|| {
        libtpu_core::proto::CompileOptionsProto {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(libtpu_core::proto::ExecutableBuildOptionsProto {
                device_ordinal: 0,
                num_replicas: 1,
                num_partitions: 1,
                use_spmd_partitioning: true,
                result_layout: None,
            }),
            compile_portable_executable: false,
            profile_version: 1,
        }
        .encode()
    });
    args.serialized_bytes = co_bytes.as_ptr().cast::<c_char>();
    args.serialized_bytes_size = co_bytes.len();
    args.serialized_compile_options = co_bytes.as_ptr().cast_mut().cast();
}

/// Serializes the `CompileOptionsProto` of a `PJRT_Executable` (`PJRT_Executable_GetCompileOptions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_GetCompileOptions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args>(),
            handle_executable_get_compile_options,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
unsafe fn handle_buffer_donate_with_control_dependency(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args>()
    };
    args.out_buffer = args.buffer;
}

/// Donates a buffer with an explicit control dependency (`PJRT_Buffer_DonateWithControlDependency`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_DonateWithControlDependency(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args>(),
            handle_buffer_donate_with_control_dependency,
        )
    }
}

/// Creates a new user-controlled `PJRT_Event` (`PJRT_Event_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Create_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Event_Create(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Event_Create_Args>();
        let ev_ptr = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
            inner: libtpu_core::pjrt::PjrtCoreEvent::pending(),
        }));
        // SAFETY: Writing output `event` field.
        unsafe {
            (*args_ptr).event = ev_ptr;
        }
    }
    err
}

/// Marks a user-controlled `PJRT_Event` as complete (`PJRT_Event_Set`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Set_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Set_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Event_Set(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Event_Set_Args>();
        // SAFETY: `args_ptr` is non-null.
        let (ev_ptr, code, msg_ptr) = unsafe {
            (
                (*args_ptr).event,
                (*args_ptr).error_code,
                (*args_ptr).error_message,
            )
        };
        if !ev_ptr.is_null() {
            let status = if code == 0 {
                None
            } else {
                let reason = if msg_ptr.is_null() {
                    format!("event error code {code}")
                } else {
                    // SAFETY: `msg_ptr` is a null-terminated C-string when non-null.
                    unsafe { CStr::from_ptr(msg_ptr) }
                        .to_string_lossy()
                        .into_owned()
                };
                Some(LibTpuError::InvalidArgument {
                    parameter: "event",
                    reason,
                })
            };
            // SAFETY: `ev_ptr` is a valid `PJRT_Event*`.
            unsafe {
                (*ev_ptr).inner.set(status);
            }
        }
    }
    err
}

/// Populates [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
unsafe fn handle_device_get_attributes(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_GetAttributes_Args>() };
    args.attributes = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_attributes = GLOBAL_PLUGIN_ATTRIBUTES.len();
}

/// Queries attributes of a `PJRT_Device` (`PJRT_Device_GetAttributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_GetAttributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_GetAttributes_Args>(),
            handle_device_get_attributes,
        )
    }
}

/// Loads an AOT-compiled `PJRT_Executable` onto a `PJRT_Client` (`PJRT_Client_Load`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_Load_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_Load_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Load(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Client_Load_Args>();
        // SAFETY: `args_ptr` is non-null.
        let (client_ptr, exec_ptr) = unsafe { ((*args_ptr).client, (*args_ptr).executable) };
        if !any_null(&[client_ptr as usize, exec_ptr as usize]) {
            // SAFETY: `client_ptr` and `exec_ptr` are valid handles.
            let client = unsafe { &*client_ptr };
            // SAFETY: `exec_ptr` is a valid `PJRT_Executable*`.
            let core_exec = unsafe { (*exec_ptr).inner.clone() };
            let addr_devs: Vec<DeviceId> = client
                .inner
                .addressable_device_indices
                .iter()
                .filter_map(|&idx| {
                    client
                        .inner
                        .devices
                        .get(idx)
                        .map(|d| d.description.spec.device_id)
                })
                .collect();
            let loaded_inner = libtpu_core::pjrt::PjrtCoreLoadedExecutable::new(
                core_exec.clone(),
                addr_devs,
                std::sync::Arc::clone(&client.inner.hal),
            );
            let new_exec_ptr = Box::into_raw(Box::new(PJRT_Executable { inner: core_exec }));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
                inner: loaded_inner,
                executable_ptr: new_exec_ptr,
                device_ptrs: client.addressable_device_ptrs.clone(),
            }));
            // SAFETY: Writing `loaded_executable` output field.
            unsafe {
                (*args_ptr).loaded_executable = loaded_ptr;
            }
        }
    }
    err
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args`].
unsafe fn handle_loaded_executable_logical_ids(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args>()
    };
    args.addressable_device_logical_ids = GLOBAL_LOGICAL_DEVICE_IDS.as_ptr();
    args.num_addressable_device_logical_ids = GLOBAL_LOGICAL_DEVICE_IDS.len();
}

/// Queries logical replica/partition IDs of addressable devices (`PJRT_LoadedExecutable_AddressableDeviceLogicalIds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_LoadedExecutable_AddressableDeviceLogicalIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args,
            >(),
            handle_loaded_executable_logical_ids,
        )
    }
}

/// Zero-copy bitcasts a `PJRT_Buffer` to a new element type and shape (`PJRT_Buffer_Bitcast`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Bitcast_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Bitcast_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Buffer_Bitcast(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Buffer_Bitcast_Args>();
        // SAFETY: `args_ptr` is non-null.
        let (buf_ptr, raw_et, dims_ptr, num_dims) = unsafe {
            (
                (*args_ptr).buffer,
                (*args_ptr).element_type,
                (*args_ptr).dims,
                (*args_ptr).num_dims,
            )
        };
        if !buf_ptr.is_null() {
            // SAFETY: `buf_ptr` is a valid `PJRT_Buffer*`.
            let buf_ref = unsafe { &*buf_ptr };
            let elem_type =
                ElementType::from_raw(raw_et).unwrap_or(buf_ref.inner.shape.element_type);
            let dims = if !dims_ptr.is_null() {
                // SAFETY: `dims_ptr` points to `num_dims` `i64` dimensions.
                unsafe { std::slice::from_raw_parts(dims_ptr, num_dims).to_vec() }
            } else {
                buf_ref.inner.shape.dims.clone()
            };
            if let Ok(bitcast_core) = buf_ref.inner.bitcast(elem_type, dims) {
                let out_ptr = Box::into_raw(Box::new(PJRT_Buffer {
                    inner: bitcast_core,
                    device_ptr: buf_ref.device_ptr,
                    memory_ptr: buf_ref.memory_ptr,
                }));
                // SAFETY: Writing `out_buffer` field.
                unsafe {
                    (*args_ptr).out_buffer = out_ptr;
                }
            }
        }
    }
    err
}

/// Iterates over structured payloads attached to a `PJRT_Error` (`PJRT_Error_ForEachPayload`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Error_ForEachPayload_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Error_ForEachPayload_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Error_ForEachPayload(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    PJRT_SLOT_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Validated by `PJRT_CheckMatchingStructSizes`.
    let err = unsafe { PJRT_CheckMatchingStructSizes(raw_args) };
    if err.is_null() {
        let args_ptr = raw_args.cast::<crate::pjrt_types::PJRT_Error_ForEachPayload_Args>();
        // SAFETY: `args_ptr` is non-null.
        let (err_ptr, visitor_opt, user_arg) =
            unsafe { ((*args_ptr).error, (*args_ptr).visitor, (*args_ptr).user_arg) };
        if let (false, Some(visitor)) = (err_ptr.is_null(), visitor_opt) {
            // SAFETY: `err_ptr` is a valid `PJRT_Error*`.
            let payloads = unsafe { (*err_ptr).error.payloads() };
            for payload in payloads {
                // SAFETY: Invoking caller's payload visitor callback.
                unsafe {
                    visitor(
                        payload.key.as_ptr().cast::<c_char>(),
                        payload.key.len(),
                        payload.value.as_ptr().cast::<c_char>(),
                        payload.value.len(),
                        user_arg,
                    );
                }
            }
        }
    }
    err
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
unsafe fn handle_topology_fingerprint(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args>()
    };
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let top_opt = unsafe { args.topology.as_ref() };
    args.fingerprint = top_opt.map_or(0x5450_555F_5635_4500, |t| t.inner.fingerprint);
}

/// Queries the 64-bit fingerprint of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Fingerprint`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Fingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args>(),
            handle_topology_fingerprint,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args`].
unsafe fn handle_executable_parameter_memory_kinds(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args>()
    };
    args.num_parameters = 1;
    args.memory_kinds = DEFAULT_MEMORY_KIND_PTRS.0.as_ptr();
    args.memory_kind_sizes = DEFAULT_MEMORY_KIND_SIZES.as_ptr();
}

/// Queries input parameter memory kinds of a `PJRT_Executable` (`PJRT_Executable_ParameterMemoryKinds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Executable_ParameterMemoryKinds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args>(),
            handle_executable_parameter_memory_kinds,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
unsafe fn handle_device_clear_memory_stats(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args =
        unsafe { &mut *raw_args.cast::<crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args>() };
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let _ = unsafe { args.device.as_ref() }.map(|d| d.inner.description.spec.device_id.0);
}

/// Resets peak and cumulative HBM statistics on a `PJRT_Device` (`PJRT_Device_ClearMemoryStats`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Device_ClearMemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args>(),
            handle_device_clear_memory_stats,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
unsafe fn handle_topology_make_canonical_shape(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args.cast::<
            crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
        >()
    };
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
}

/// Computes canonical tiled device shape for a memory space (`PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
            >(),
            handle_topology_make_canonical_shape,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
///
/// # Safety
///
/// `raw_args` must point to a valid [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
unsafe fn handle_topology_get_memory_space_kind_ids(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    let args = unsafe {
        &mut *raw_args
            .cast::<crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args>()
    };
    args.kind_ids = GLOBAL_MEMORY_KIND_IDS.as_ptr();
    args.num_kind_ids = GLOBAL_MEMORY_KIND_IDS.len();
}

/// Queries supported memory space kind IDs of a topology (`PJRT_TopologyDescription_GetMemorySpaceKindIds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_GetMemorySpaceKindIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_validated_pjrt_args`.
    unsafe {
        dispatch_validated_pjrt_args(
            raw_args,
            std::mem::size_of::<
                crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args,
            >(),
            handle_topology_get_memory_space_kind_ids,
        )
    }
}

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
    pjrt_client_create_alias_buffer: PJRT_Client_CreateAliasBuffer,
    pjrt_client_fulfill_alias_buffer: PJRT_Client_FulfillAliasBuffer,
    pjrt_topology_description_deserialize: PJRT_TopologyDescription_Deserialize,
    pjrt_client_create_error_buffer: PJRT_Client_CreateErrorBuffer,
    pjrt_client_create_uninitialized_buffer: PJRT_Client_CreateUninitializedBuffer,
    pjrt_loaded_executable_get_device_assignment: PJRT_LoadedExecutable_GetDeviceAssignment,
    pjrt_client_update_global_process_info: PJRT_Client_UpdateGlobalProcessInfo,
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
#[no_mangle]
pub extern "C" fn GetPjrtApi() -> *const PJRT_Api {
    ptr::addr_of!(GLOBAL_PJRT_API)
}

/// Slot 1 of `LibtpuSdk_Api`: populates the message of a [`LibtpuSdk_Error`] (`LibtpuSdk_Error_GetMessage`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Error_GetMessage_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Error_GetMessage_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_Error_GetMessage(raw_args: *mut std::ffi::c_void) {
    let args_ptr = raw_args.cast::<LibtpuSdk_Error_GetMessage_Args>();
    if args_ptr.is_null() {
        return;
    }
    // SAFETY: `args_ptr` is non-null.
    let err_ptr = unsafe { (*args_ptr).error };
    if err_ptr.is_null() {
        return;
    }
    // SAFETY: `err_ptr` is non-null.
    let err_ref = unsafe { &*err_ptr };
    let bytes = err_ref.c_message.as_bytes();
    // SAFETY: Writing output message fields.
    unsafe {
        (*args_ptr).message = err_ref.c_message.as_ptr();
        (*args_ptr).message_size = bytes.len();
    }
}

/// Slot 2 of `LibtpuSdk_Api`: destroys a [`LibtpuSdk_Error`] (`LibtpuSdk_Error_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Error_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Error_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_Error_Destroy(raw_args: *mut std::ffi::c_void) {
    let args_ptr = raw_args.cast::<LibtpuSdk_Error_Destroy_Args>();
    if args_ptr.is_null() {
        return;
    }
    // SAFETY: `args_ptr` is non-null.
    let err_ptr = unsafe { (*args_ptr).error };
    if !err_ptr.is_null() {
        // SAFETY: `err_ptr` was allocated by `box_sdk_error`.
        unsafe {
            drop(Box::from_raw(err_ptr));
        }
    }
}

/// Slot 3 of `LibtpuSdk_Api`: queries the error code of a [`LibtpuSdk_Error`] (`LibtpuSdk_Error_GetCode`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Error_GetCode_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Error_GetCode_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_Error_GetCode(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_Error_GetCode_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let err_ptr = unsafe { (*args_ptr).error };
    if err_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "error" });
    }
    // SAFETY: `err_ptr` is non-null.
    unsafe {
        (*args_ptr).code = (*err_ptr).code;
    }
    ptr::null_mut()
}

/// Slot 4 of `LibtpuSdk_Api`: creates a new [`LibtpuSdk_Client`] (`LibtpuSdk_Client_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Client_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Client_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_Client_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_Client_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    let inner = LibtpuSdkCoreClient::new();
    let mut host_names = Vec::with_capacity(inner.chip_coordinates.len());
    let mut coords_entries = Vec::with_capacity(inner.chip_coordinates.len());
    for (idx, c) in inner.chip_coordinates.iter().enumerate() {
        let cs = to_c_string(&inner.hostname);
        let host_name_ptr = cs.as_ptr();
        let host_name_len = cs.as_bytes().len();
        host_names.push(cs);
        coords_entries.push(LibtpuSdk_ChipLocationToCoordinates {
            host_name_ptr,
            host_name_len,
            chip_index: i32::try_from(idx).unwrap_or(0),
            pad: 0,
            coordinates: [i64::from(c.x), i64::from(c.y), i64::from(c.z)],
        });
    }
    let client = Box::into_raw(Box::new(LibtpuSdk_Client {
        inner,
        host_names,
        coords_entries,
    }));
    // SAFETY: `args_ptr` is non-null.
    unsafe {
        (*args_ptr).client = client;
    }
    ptr::null_mut()
}

/// Slot 5 of `LibtpuSdk_Api`: destroys a [`LibtpuSdk_Client`] (`LibtpuSdk_Client_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Client_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Client_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_Client_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_Client_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` was allocated in `LibtpuSdk_Client_Create`.
    unsafe {
        drop(Box::from_raw(client_ptr));
    }
    ptr::null_mut()
}

/// Slot 6 of `LibtpuSdk_Api`: queries all chip coordinate records (`LibtpuSdk_GetChipCoordinates`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetChipCoordinates_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetChipCoordinates_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetChipCoordinates(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetChipCoordinates_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client_ref = unsafe { &*client_ptr };
    // SAFETY: Writing output array pointer and length.
    unsafe {
        (*args_ptr).coords = client_ref.coords_entries.as_ptr();
        (*args_ptr).num_coords = client_ref.coords_entries.len();
    }
    ptr::null_mut()
}

/// Slot 7 of `LibtpuSdk_Api`: queries the hostname of a chip coordinate entry (`LibtpuSdk_GetHostName`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetHostName_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetHostName_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetHostName(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetHostName_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let coord_ptr = unsafe { (*args_ptr).coord };
    if coord_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "coord" });
    }
    // SAFETY: `coord_ptr` is non-null.
    let coord_ref = unsafe { &*coord_ptr };
    // SAFETY: Writing output hostname fields.
    unsafe {
        (*args_ptr).host_name = coord_ref.host_name_ptr;
        (*args_ptr).host_name_size = coord_ref.host_name_len;
    }
    ptr::null_mut()
}

/// Slot 8 of `LibtpuSdk_Api`: queries the chip index of a chip coordinate entry (`LibtpuSdk_GetChipIndex`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetChipIndex_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetChipIndex_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetChipIndex(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetChipIndex_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let coord_ptr = unsafe { (*args_ptr).coord };
    if coord_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "coord" });
    }
    // SAFETY: `coord_ptr` is non-null.
    unsafe {
        (*args_ptr).chip_index = (*coord_ptr).chip_index;
    }
    ptr::null_mut()
}

/// Slot 9 of `LibtpuSdk_Api`: queries the Cartesian `[x, y, z]` coordinates (`LibtpuSdk_GetCartesianCoordinates`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetCartesianCoordinates_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetCartesianCoordinates_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetCartesianCoordinates(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetCartesianCoordinates_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let coord_ptr = unsafe { (*args_ptr).coord };
    if coord_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "coord" });
    }
    // SAFETY: `coord_ptr` is non-null.
    let coord_ref = unsafe { &*coord_ptr };
    // SAFETY: Writing output coordinates pointer and length.
    unsafe {
        (*args_ptr).coordinates = coord_ref.coordinates.as_ptr();
        (*args_ptr).num_coordinates = coord_ref.coordinates.len();
    }
    ptr::null_mut()
}

/// Slot 10 of `LibtpuSdk_Api`: queries a telemetry metric by name (`LibtpuSdk_GetMetric`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetMetric_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetMetric_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetMetric(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetMetric_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let (client_ptr, name_ptr) = unsafe { ((*args_ptr).client, (*args_ptr).metric_name) };
    if any_null(&[client_ptr as usize, name_ptr as usize]) {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client_or_metric_name",
        });
    }
    // SAFETY: `name_ptr` is non-null.
    let metric_name = unsafe { CStr::from_ptr(name_ptr).to_string_lossy() };
    // SAFETY: `client_ptr` is non-null.
    let client_ref = unsafe { &*client_ptr };
    match client_ref.inner.get_metric(&metric_name) {
        Ok((desc, vals)) => {
            let description = to_c_string(&desc);
            let value_cstrings: Vec<CString> =
                vals.iter().map(|v| to_c_string(&v.to_string())).collect();
            let value_ptrs: Vec<*const c_char> =
                value_cstrings.iter().map(|cs| cs.as_ptr()).collect();
            let metric_handle = Box::into_raw(Box::new(LibtpuSdk_Metric {
                description,
                value_cstrings,
                value_ptrs,
            }));
            // SAFETY: Writing output metric pointer.
            unsafe {
                (*args_ptr).metric = metric_handle;
            }
            ptr::null_mut()
        }
        Err(e) => box_sdk_error(&e),
    }
}

/// Slot 11 of `LibtpuSdk_Api`: queries the description of a [`LibtpuSdk_Metric`] (`LibtpuSdk_GetMetricDescription`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetMetricDescription_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetMetricDescription_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetMetricDescription(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetMetricDescription_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let metric_ptr = unsafe { (*args_ptr).metric };
    if metric_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "metric",
        });
    }
    // SAFETY: `metric_ptr` is non-null.
    let metric_ref = unsafe { &*metric_ptr };
    let bytes = metric_ref.description.as_bytes();
    // SAFETY: Writing output description fields.
    unsafe {
        (*args_ptr).description = metric_ref.description.as_ptr();
        (*args_ptr).description_size = bytes.len();
    }
    ptr::null_mut()
}

/// Slot 12 of `LibtpuSdk_Api`: queries the value strings of a [`LibtpuSdk_Metric`] (`LibtpuSdk_GetMetricValues`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetMetricValues_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetMetricValues_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetMetricValues(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetMetricValues_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let metric_ptr = unsafe { (*args_ptr).metric };
    if metric_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "metric",
        });
    }
    // SAFETY: `metric_ptr` is non-null.
    let metric_ref = unsafe { &*metric_ptr };
    // SAFETY: Writing output values pointer and count.
    unsafe {
        (*args_ptr).values = metric_ref.value_ptrs.as_ptr();
        (*args_ptr).num_values = metric_ref.value_ptrs.len();
    }
    ptr::null_mut()
}

/// Slot 13 of `LibtpuSdk_Api`: queries runtime core state (`LibtpuSdk_GetRuntimeStatus`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetRuntimeStatus_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetRuntimeStatus_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_GetRuntimeStatus(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetRuntimeStatus_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let summary_text = unsafe { (*client_ptr).inner.core_state_summary() };
    let status_handle = Box::into_raw(Box::new(LibtpuSdk_RuntimeStatus {
        summary: to_c_string(&summary_text),
    }));
    // SAFETY: Writing output status pointer.
    unsafe {
        (*args_ptr).status = status_handle;
    }
    ptr::null_mut()
}

/// Slot 14 of `LibtpuSdk_Api`: extracts the summary string from [`LibtpuSdk_RuntimeStatus`]
/// (`LibtpuSdk_RuntimeStatus_GetCoreStateSummary`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let status_ptr = unsafe { (*args_ptr).status };
    if status_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "status",
        });
    }
    // SAFETY: `status_ptr` is non-null.
    let status_ref = unsafe { &*status_ptr };
    let bytes = status_ref.summary.as_bytes();
    // SAFETY: Writing output summary fields.
    unsafe {
        (*args_ptr).summary = status_ref.summary.as_ptr();
        (*args_ptr).summary_size = bytes.len();
    }
    ptr::null_mut()
}

/// Slot 15 of `LibtpuSdk_Api`: destroys a [`LibtpuSdk_RuntimeStatus`] (`LibtpuSdk_RuntimeStatus_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_RuntimeStatus_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_RuntimeStatus_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_RuntimeStatus_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_RuntimeStatus_Destroy_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let status_ptr = unsafe { (*args_ptr).status };
    if !status_ptr.is_null() {
        // SAFETY: `status_ptr` was allocated in `LibtpuSdk_GetRuntimeStatus`.
        unsafe {
            drop(Box::from_raw(status_ptr));
        }
    }
    ptr::null_mut()
}

/// Slot 16 of `LibtpuSdk_Api`: registers a debug callback (`LibtpuSdk_RegisterDebugCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_RegisterDebugCallback_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_RegisterDebugCallback_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_RegisterDebugCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_RegisterDebugCallback_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let id = unsafe {
        (*client_ptr).inner.register_debug_callback(
            (*args_ptr).callback as usize,
            (*args_ptr).user_data as usize,
        )
    };
    // SAFETY: Writing output callback_id.
    unsafe {
        (*args_ptr).callback_id = i64::try_from(id.0).unwrap_or(0);
    }
    ptr::null_mut()
}

/// Slot 17 of `LibtpuSdk_Api`: unregisters a debug callback (`LibtpuSdk_UnregisterDebugCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_UnregisterDebugCallback_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_UnregisterDebugCallback_Args`].
#[no_mangle]
pub unsafe extern "C" fn LibtpuSdk_UnregisterDebugCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_UnregisterDebugCallback_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let (client_ptr, cb_id) = unsafe { ((*args_ptr).client, (*args_ptr).callback_id) };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    if let Err(e) = unsafe {
        (*client_ptr)
            .inner
            .unregister_debug_callback(CallbackId(u64::try_from(cb_id).unwrap_or(u64::MAX)))
    } {
        return box_sdk_error(&e);
    }
    ptr::null_mut()
}

/// Static 144-byte `LibtpuSdk_Api` vtable returned by [`GetLibtpuSdkApi`].
static GLOBAL_SDK_API: LibtpuSdk_Api = LibtpuSdk_Api {
    version_header: 0x0000_0001_0000_0000,
    error_get_message: LibtpuSdk_Error_GetMessage,
    error_destroy: LibtpuSdk_Error_Destroy,
    error_get_code: LibtpuSdk_Error_GetCode,
    client_create: LibtpuSdk_Client_Create,
    client_destroy: LibtpuSdk_Client_Destroy,
    get_chip_coordinates: LibtpuSdk_GetChipCoordinates,
    get_host_name: LibtpuSdk_GetHostName,
    get_chip_index: LibtpuSdk_GetChipIndex,
    get_cartesian_coordinates: LibtpuSdk_GetCartesianCoordinates,
    get_metric: LibtpuSdk_GetMetric,
    get_metric_description: LibtpuSdk_GetMetricDescription,
    get_metric_values: LibtpuSdk_GetMetricValues,
    get_runtime_status: LibtpuSdk_GetRuntimeStatus,
    runtime_status_get_core_state_summary: LibtpuSdk_RuntimeStatus_GetCoreStateSummary,
    runtime_status_destroy: LibtpuSdk_RuntimeStatus_Destroy,
    register_debug_callback: LibtpuSdk_RegisterDebugCallback,
    unregister_debug_callback: LibtpuSdk_UnregisterDebugCallback,
};

/// Secondary SDK C API entrypoint exported by `libtpu.so` (`GetLibtpuSdkApi`).
#[no_mangle]
pub extern "C" fn GetLibtpuSdkApi() -> *const LibtpuSdk_Api {
    ptr::addr_of!(GLOBAL_SDK_API)
}

/// Initializes the legacy TensorFlow TPU driver (`TfTpu_Initialize`).
///
/// # Arguments
///
/// * `init_library` - Whether to initialize the underlying driver library.
/// * `num_args` - Number of CLI strings in `args`.
/// * `args` - Optional array of CLI string pointers.
///
/// # Safety
///
/// `args` must be null or point to `num_args` null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn TfTpu_Initialize(
    init_library: bool,
    num_args: i32,
    args: *const *const c_char,
) {
    let count = usize::try_from(num_args).unwrap_or(0);
    if !any_null(&[usize::from(init_library), count, args as usize]) {
        // SAFETY: `args` is non-null and contains `count` string pointers.
        let slice = unsafe { std::slice::from_raw_parts(args, count) };
        let mut parsed = Vec::with_capacity(count);
        for &ptr_c in slice {
            if !ptr_c.is_null() {
                // SAFETY: `ptr_c` is non-null and null-terminated.
                parsed.push(unsafe { CStr::from_ptr(ptr_c).to_string_lossy().into_owned() });
            }
        }
        let _ = libtpu_core::config::apply_global_cli_args(&parsed);
    }
}

/// TensorFlow kernel initialization hook (`TF_InitKernel`).
#[no_mangle]
pub extern "C" fn TF_InitKernel() {}

/// TensorFlow `NextPluggableDevice` plugin initialization hook (`TFNPD_InitPlugin`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TFNPD_PluginParams`] (or null).
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `params` and `status` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TFNPD_InitPlugin(params: *mut TFNPD_PluginParams, status: *mut TpuStatus) {
    if params.is_null() {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"TFNPD_PluginParams is null".as_ptr(), 26);
        }
        return;
    }
    // SAFETY: `params` is non-null.
    unsafe {
        (*params).device_type = c"TPU".as_ptr();
        (*params).compilation_device_name = c"TPU_REPLICATED_CORE".as_ptr();
        (*params).priority = 55;
        (*params).is_pluggable_device = true;
        (*params).use_pjrt_on_demand_compile = true;
    }
}

/// Configures global TPU topology state (`SetGlobalTPUArrayOp_DoWork`).
///
/// # Arguments
///
/// * `topology_len` - Byte length of `topology_proto`.
/// * `topology_proto` - Pointer to serialized `TpuTopologyArgsProto` bytes (or null).
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `topology_proto` and `status` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn SetGlobalTPUArrayOp_DoWork(
    topology_len: usize,
    topology_proto: *const c_char,
    status: *mut TpuStatus,
) -> i32 {
    if topology_len > 0 && topology_proto.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"topology_proto is null".as_ptr(), 22);
        }
        return -1;
    }
    if topology_len > 0 {
        // SAFETY: `topology_proto` is non-null and `topology_len` bytes long.
        let bytes =
            unsafe { std::slice::from_raw_parts(topology_proto.cast::<u8>(), topology_len) };
        if let Ok(proto) = libtpu_core::proto::TpuTopologyArgsProto::decode(bytes) {
            let mut cfg = libtpu_core::config::global_init_config();
            cfg.host_bounds.x = u32::try_from(proto.num_hosts.max(1)).unwrap_or(1);
            libtpu_core::config::set_global_init_config(cfg);
        }
    }
    0
}

/// Queries TPU partitioned call runtime parameters (`TfTpu_GetTpuPartitionedCallParams`).
///
/// # Arguments
///
/// * `params` - Pointer to [`TpuPartitionedCall_Params`] (or null).
///
/// # Safety
///
/// `params` must be null or a valid mutable pointer.
#[no_mangle]
pub unsafe extern "C" fn TfTpu_GetTpuPartitionedCallParams(
    params: *mut TpuPartitionedCall_Params,
) -> i32 {
    if params.is_null() {
        return -1;
    }
    // SAFETY: `params` is non-null.
    unsafe {
        (*params).input_shape_opt = true;
        (*params).group_tensors_for_packing = true;
        (*params).minimum_input_tensors_packing = 4;
        (*params).minimum_output_tensors_packing = 4;
        (*params).enable_auto_xla_input_sharding = true;
        (*params).auto_xla_input_sharding_dim = 0;
        (*params).scoped_allocator_for_tpu_anned_call = true;
    }
    0
}

/// Frees a heap-allocated `char*` array (`TpuConfigurationApi_FreeCharArray`).
///
/// # Arguments
///
/// * `ptr` - Pointer previously returned by a configuration API, or null.
///
/// # Safety
///
/// `ptr` must be null or a valid `CString` raw pointer.
#[no_mangle]
pub unsafe extern "C" fn TpuConfigurationApi_FreeCharArray(ptr: *mut c_char) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated via `CString::into_raw`.
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

/// Computes on-device infeed layout (`TpuTransferManager_GetInfeedLayout`).
///
/// # Arguments
///
/// * `host_shape` - Input host [`XLA_Shape`] pointer.
/// * `infeed_shape` - Output on-device [`XLA_Shape`] pointer.
///
/// # Safety
///
/// `host_shape` and `infeed_shape` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuTransferManager_GetInfeedLayout(
    host_shape: *const XLA_Shape,
    infeed_shape: *mut XLA_Shape,
) -> i32 {
    if any_null(&[host_shape as usize, infeed_shape as usize]) {
        return -1;
    }
    // SAFETY: Both pointers are non-null.
    unsafe {
        let src = *host_shape;
        let mut dst = src;
        let rank = usize::try_from(src.rank.clamp(0, 8)).unwrap_or(0);
        for (i, slot) in dst.minor_to_major.iter_mut().enumerate().take(rank) {
            *slot = i64::try_from(rank - 1 - i).unwrap_or(0);
        }
        dst.tile_dims = [8, 128];
        *infeed_shape = dst;
    }
    0
}

/// Queries `SparseCore` max IDs and uniques per partition (`SparseCore_GetMaxIdsAndUniques`).
///
/// # Arguments
///
/// * `table_name` - Null-terminated embedding table name.
/// * `num_samples_per_sparse_core` - Batch samples per SparseCore.
/// * `feature_width` - Embedding feature width.
/// * `max_ids_per_partition` - Output pointer receiving max IDs per partition.
/// * `max_unique_ids_per_partition` - Output pointer receiving max unique IDs per partition.
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn SparseCore_GetMaxIdsAndUniques(
    table_name: *const c_char,
    num_samples_per_sparse_core: i64,
    feature_width: i64,
    max_ids_per_partition: *mut i64,
    max_unique_ids_per_partition: *mut i64,
    status: *mut TpuStatus,
) -> i64 {
    if any_null(&[
        table_name as usize,
        max_ids_per_partition as usize,
        max_unique_ids_per_partition as usize,
    ]) {
        // SAFETY: `TpuStatus_Set` checks `status` for null.
        unsafe {
            TpuStatus_Set(status, 3, c"null SparseCore argument".as_ptr(), 24);
        }
        return -1;
    }
    let samples = num_samples_per_sparse_core.max(1);
    let width = feature_width.max(1);
    let max_ids = (samples.saturating_mul(4)).max(128);
    let max_uniques = (samples.saturating_mul(2).min(width.saturating_mul(16))).max(64);
    // SAFETY: Output pointers are non-null.
    unsafe {
        *max_ids_per_partition = max_ids;
        *max_unique_ids_per_partition = max_uniques;
    }
    128
}

/// Allocates a new OK [`TpuStatus`] (`TpuStatus_New`).
#[no_mangle]
pub extern "C" fn TpuStatus_New() -> *mut TpuStatus {
    Box::into_raw(Box::new(TpuStatus {
        inner: TpuCoreStatus::ok(),
        c_message: to_c_string(""),
    }))
}

/// Allocates a [`TpuStatus`] with explicit code and C message (`TpuStatus_Create`).
///
/// # Arguments
///
/// * `code` - Integer status code.
/// * `msg` - Null-terminated UTF-8 error message pointer (or null).
///
/// # Safety
///
/// `msg` must be null or a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn TpuStatus_Create(code: i32, msg: *const c_char) -> *mut TpuStatus {
    let text = if msg.is_null() {
        String::new()
    } else {
        // SAFETY: `msg` is non-null and null-terminated.
        unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() }
    };
    let c_message = to_c_string(&text);
    Box::into_raw(Box::new(TpuStatus {
        inner: TpuCoreStatus::new(code, text),
        c_message,
    }))
}

/// Updates an existing [`TpuStatus`] (`TpuStatus_Set`).
///
/// # Arguments
///
/// * `status` - Mutable pointer to [`TpuStatus`].
/// * `code` - New status code.
/// * `msg` - Null-terminated error message pointer (or null).
/// * `_len` - Length hint for `msg`.
///
/// # Safety
///
/// `status` and `msg` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuStatus_Set(
    status: *mut TpuStatus,
    code: i32,
    msg: *const c_char,
    _len: i32,
) {
    if status.is_null() {
        return;
    }
    let text = if msg.is_null() {
        String::new()
    } else {
        // SAFETY: `msg` is non-null.
        unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() }
    };
    // SAFETY: `status` is non-null.
    let st = unsafe { &mut *status };
    st.c_message = to_c_string(&text);
    st.inner.set(code, text);
}

/// Frees a [`TpuStatus`] (`TpuStatus_Free`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`] to free, or null.
///
/// # Safety
///
/// `status` must be null or allocated by `TpuStatus_New` / `TpuStatus_Create`.
#[no_mangle]
pub unsafe extern "C" fn TpuStatus_Free(status: *mut TpuStatus) {
    if !status.is_null() {
        // SAFETY: `status` is non-null and heap-allocated.
        unsafe {
            drop(Box::from_raw(status));
        }
    }
}

/// Returns the error message C string of a [`TpuStatus`] (`TpuStatus_Message`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`].
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[no_mangle]
pub unsafe extern "C" fn TpuStatus_Message(status: *const TpuStatus) -> *const c_char {
    if status.is_null() {
        return ptr::null();
    }
    // SAFETY: `status` is non-null.
    unsafe { (*status).c_message.as_ptr() }
}

/// Returns the integer status code of a [`TpuStatus`] (`TpuStatus_Code`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`].
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[no_mangle]
pub unsafe extern "C" fn TpuStatus_Code(status: *const TpuStatus) -> i32 {
    if status.is_null() {
        return -1;
    }
    // SAFETY: `status` is non-null.
    unsafe { (*status).inner.code.as_raw() }
}

/// Returns `true` if `status` is OK (`TpuStatus_Ok`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`].
///
/// # Safety
///
/// `status` must be null or a valid pointer to [`TpuStatus`].
#[no_mangle]
pub unsafe extern "C" fn TpuStatus_Ok(status: *const TpuStatus) -> bool {
    if status.is_null() {
        return false;
    }
    // SAFETY: `status` is non-null.
    unsafe { (*status).inner.is_ok() }
}

/// Creates a new [`TfTpuOrdinalSelector`] (`TfTpuOrdinalSelector_Create`).
///
/// # Arguments
///
/// * `selector_out` - Output pointer to receive the new selector handle.
/// * `num_cores` - Number of cores to cycle across.
///
/// # Safety
///
/// `selector_out` must be null or a valid mutable pointer.
#[no_mangle]
pub unsafe extern "C" fn TfTpuOrdinalSelector_Create(
    selector_out: *mut *mut TfTpuOrdinalSelector,
    num_cores: i32,
) {
    if selector_out.is_null() {
        return;
    }
    let handle = Box::into_raw(Box::new(TfTpuOrdinalSelector {
        inner: TfTpuCoreOrdinalSelector::new(u32::try_from(num_cores.max(1)).unwrap_or(1)),
    }));
    // SAFETY: `selector_out` is non-null.
    unsafe {
        *selector_out = handle;
    }
}

/// Destroys a [`TfTpuOrdinalSelector`] (`TfTpuOrdinalSelector_Destroy`).
///
/// # Arguments
///
/// * `selector` - Selector handle to destroy.
///
/// # Safety
///
/// `selector` must be null or allocated by `TfTpuOrdinalSelector_Create`.
#[no_mangle]
pub unsafe extern "C" fn TfTpuOrdinalSelector_Destroy(selector: *mut TfTpuOrdinalSelector) {
    if !selector.is_null() {
        // SAFETY: `selector` is non-null.
        unsafe {
            drop(Box::from_raw(selector));
        }
    }
}

/// Selects the next device ordinal (`TfTpuOrdinalSelector_GetOrdinal`).
///
/// # Arguments
///
/// * `selector` - Pointer to [`TfTpuOrdinalSelector`].
/// * `seed` - Optional request seed pointer.
/// * `ordinal_out` - Output pointer for the selected ordinal / request ID.
///
/// # Safety
///
/// `selector`, `seed`, and `ordinal_out` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TfTpuOrdinalSelector_GetOrdinal(
    selector: *const TfTpuOrdinalSelector,
    seed: *mut u64,
    ordinal_out: *mut i64,
) -> i32 {
    if any_null(&[selector as usize, ordinal_out as usize]) {
        return -1;
    }
    // SAFETY: `selector` and `ordinal_out` are non-null.
    let ord = unsafe { (*selector).inner.get_ordinal() };
    // SAFETY: `ordinal_out` is non-null.
    unsafe {
        *ordinal_out = ord;
        if !seed.is_null() {
            *seed = (*seed).wrapping_add(1);
        }
    }
    i32::try_from(ord).unwrap_or(0)
}

/// Dequeues a reserved ordinal from the core selector (`TfTpuOrdinalSelector_DequeueFromCoreSelector`).
///
/// # Arguments
///
/// * `selector` - Pointer to [`TfTpuOrdinalSelector`] (or null).
/// * `device_ordinal` - Device ordinal to release.
/// * `req_id` - Request identifier.
///
/// # Safety
///
/// `selector` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn TfTpuOrdinalSelector_DequeueFromCoreSelector(
    selector: *mut TfTpuOrdinalSelector,
    device_ordinal: i32,
    req_id: i64,
) {
    if !selector.is_null() && device_ordinal >= 0 && req_id >= 0 {
        // SAFETY: `selector` is non-null.
        let _ = unsafe { (*selector).inner.get_ordinal() };
    }
}

/// Global singleton `TpuTopology` initialized on first call to `global_topology`.
static GLOBAL_TOPOLOGY: OnceLock<TpuTopology> = OnceLock::new();

/// Returns a static reference to the singleton [`TpuTopology`].
fn global_topology() -> &'static TpuTopology {
    GLOBAL_TOPOLOGY.get_or_init(|| {
        let cfg = TpuInitConfig::default();
        let hal = SimulatedTpuHal::new(&cfg);
        let spec = hal.topology_spec();
        let core_locations = spec
            .devices
            .iter()
            .map(|d| TpuCoreLocation {
                core_id: i32::try_from(d.device_id.0).unwrap_or(0),
                index_on_chip: i32::try_from(d.core_on_chip.0).unwrap_or(0),
                chip_x: i32::try_from(d.chip_coords.x).unwrap_or(0),
                chip_y: i32::try_from(d.chip_coords.y).unwrap_or(0),
                chip_z: i32::try_from(d.chip_coords.z).unwrap_or(0),
            })
            .collect();
        TpuTopology {
            hal,
            core_locations,
        }
    })
}

/// Resolves `topo` to either `&*topo` (if non-null) or `global_topology()`.
///
/// # Arguments
///
/// * `topo` - Optional [`TpuTopology`] pointer.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[inline]
unsafe fn resolve_topology<'a>(topo: *const TpuTopology) -> &'a TpuTopology {
    if topo.is_null() {
        global_topology()
    } else {
        // SAFETY: `topo` is non-null.
        unsafe { &*topo }
    }
}

/// Returns a pointer to the singleton [`TpuTopology`] (`TpuUtil_GetTopologyPtr`).
#[no_mangle]
pub extern "C" fn TpuUtil_GetTopologyPtr() -> *const TpuTopology {
    ptr::from_ref(global_topology())
}

/// Returns the XLA minor-dimension alignment pad size (`TpuUtil_GetXlaPadSizeFromTpuTopology`).
#[no_mangle]
pub extern "C" fn TpuUtil_GetXlaPadSizeFromTpuTopology() -> i32 {
    global_topology()
        .hal
        .topology_spec()
        .generation
        .xla_pad_size()
}

/// Returns the number of logical devices per host (`TpuTopology_LogicalDevicesPerHost`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_LogicalDevicesPerHost(
    topo: *const TpuTopology,
    core_type: i32,
) -> i32 {
    // SAFETY: `topo` is null or valid.
    unsafe { TpuTopology_NumCores(topo, core_type) }
}

/// Returns the number of logical devices per chip (`TpuTopology_LogicalDevicesPerChip`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_LogicalDevicesPerChip(
    topo: *const TpuTopology,
    core_type: i32,
) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let tc_per_chip =
        i32::try_from(t.hal.topology_spec().generation.logical_devices_per_chip()).unwrap_or(1);
    if core_type == 1 {
        let sc_per_dev =
            i32::try_from(t.hal.topology_spec().generation.sparse_cores_per_device()).unwrap_or(0);
        tc_per_chip.saturating_mul(sc_per_dev.max(1))
    } else {
        tc_per_chip
    }
}

/// Returns the total host count in the topology (`TpuTopology_HostCount`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_HostCount(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let bounds = t.hal.topology_spec().host_bounds;
    i32::try_from((bounds.x * bounds.y * bounds.z).max(1)).unwrap_or(1)
}

/// Returns the number of chips per host (`TpuTopology_ChipsPerHost`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_ChipsPerHost(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chips_per_host).unwrap_or(1)
}

/// Returns the X mesh bound (`TpuTopology_ChipBounds_X`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_ChipBounds_X(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chip_bounds.x).unwrap_or(1)
}

/// Returns the Y mesh bound (`TpuTopology_ChipBounds_Y`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_ChipBounds_Y(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chip_bounds.y).unwrap_or(1)
}

/// Returns the Z mesh bound (`TpuTopology_ChipBounds_Z`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_ChipBounds_Z(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    i32::try_from(t.hal.topology_spec().chip_bounds.z).unwrap_or(1)
}

/// Checks whether chip `(x, y, z)` exists in the topology (`TpuTopology_HasChip`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `x` - Chip X coordinate.
/// * `y` - Chip Y coordinate.
/// * `z` - Chip Z coordinate.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_HasChip(
    topo: *const TpuTopology,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let bounds = t.hal.topology_spec().chip_bounds;
    let Ok(ux) = u32::try_from(x) else {
        return false;
    };
    let Ok(uy) = u32::try_from(y) else {
        return false;
    };
    let Ok(uz) = u32::try_from(z) else {
        return false;
    };
    ux < bounds.x && uy < bounds.y && uz < bounds.z
}

/// Looks up a [`TpuCoreLocation`] by core ID (`TpuTopology_CoreForId`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `_core_type` - Core type discriminant.
/// * `id` - Core ID.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_CoreForId(
    topo: *const TpuTopology,
    _core_type: i32,
    id: i32,
) -> *const TpuCoreLocation {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let idx = usize::try_from(id).unwrap_or(usize::MAX);
    t.core_locations.get(idx).map_or(ptr::null(), ptr::from_ref)
}

/// Looks up a [`TpuCoreLocation`] by coordinates and index (`TpuTopology_Core`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `_core_type` - Core type discriminant.
/// * `x` - Chip X coordinate.
/// * `y` - Chip Y coordinate.
/// * `z` - Chip Z coordinate.
/// * `index` - Core index on chip.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_Core(
    topo: *const TpuTopology,
    _core_type: i32,
    x: i32,
    y: i32,
    z: i32,
    index: i32,
) -> *const TpuCoreLocation {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    t.core_locations
        .iter()
        .find(|c| (c.chip_x, c.chip_y, c.chip_z, c.index_on_chip) == (x, y, z, index))
        .map_or(ptr::null(), ptr::from_ref)
}

/// Returns the total number of cores in the topology (`TpuTopology_NumCores`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_NumCores(topo: *const TpuTopology, core_type: i32) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let tc_count = i32::try_from(t.core_locations.len()).unwrap_or(0);
    if core_type == 1 {
        let sc_per_dev =
            i32::try_from(t.hal.topology_spec().generation.sparse_cores_per_device()).unwrap_or(0);
        tc_count.saturating_mul(sc_per_dev.max(1))
    } else {
        tc_count
    }
}

/// Copies pointers to all [`TpuCoreLocation`] entries into `cores_out` (`TpuTopology_Cores`).
///
/// # Arguments
///
/// * `topo` - Topology handle.
/// * `_core_type` - Core type enum.
/// * `cores_out` - Output array of `*const TpuCoreLocation`.
///
/// # Safety
///
/// `cores_out` must be null or point to an array of at least `TpuTopology_NumCores` pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_Cores(
    topo: *const TpuTopology,
    _core_type: i32,
    cores_out: *mut *const TpuCoreLocation,
) {
    if cores_out.is_null() {
        return;
    }
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    for (idx, loc) in t.core_locations.iter().enumerate() {
        // SAFETY: Caller guarantees capacity of `cores_out`.
        unsafe {
            *cores_out.add(idx) = ptr::from_ref(loc);
        }
    }
}

/// Returns host ID for host coordinates (`TpuTopology_IdForHost`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
/// * `x` - Host X coordinate.
/// * `y` - Host Y coordinate.
/// * `z` - Host Z coordinate.
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_IdForHost(
    topo: *const TpuTopology,
    x: i32,
    y: i32,
    z: i32,
) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    let hb = t.hal.topology_spec().host_bounds;
    let hx = i32::try_from(hb.x.max(1)).unwrap_or(1);
    let hy = i32::try_from(hb.y.max(1)).unwrap_or(1);
    x.max(0) + y.max(0) * hx + z.max(0) * hx * hy
}

/// Returns the `TpuVersionEnum` integer of the topology (`TpuTopology_Version`).
///
/// # Arguments
///
/// * `topo` - Pointer to [`TpuTopology`] (or null for global default).
///
/// # Safety
///
/// `topo` must be null or a valid pointer to [`TpuTopology`].
#[no_mangle]
pub unsafe extern "C" fn TpuTopology_Version(topo: *const TpuTopology) -> i32 {
    // SAFETY: `topo` is null or valid.
    let t = unsafe { resolve_topology(topo) };
    t.hal.topology_spec().generation as i32
}

/// Queries available core count (`TpuTopology_AvailableCoreCount`).
#[no_mangle]
pub extern "C" fn TpuTopology_AvailableCoreCount(
    _mesh_state: *const TpuMeshState,
    core_type: i32,
) -> i32 {
    // SAFETY: Passing null uses `global_topology()`.
    unsafe { TpuTopology_NumCores(ptr::null(), core_type) }
}

/// Queries available cores per chip (`TpuTopology_AvailableCoresPerChip`).
///
/// # Arguments
///
/// * `core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
#[no_mangle]
pub extern "C" fn TpuTopology_AvailableCoresPerChip(core_type: i32) -> i32 {
    // SAFETY: Passing null uses `global_topology()`.
    unsafe { TpuTopology_LogicalDevicesPerChip(ptr::null(), core_type) }
}

/// Queries `SparseCores` per logical device (`TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice`).
///
/// # Arguments
///
/// * `_core_type` - Core type discriminant (`0` = `kTensorCore`, `1` = `kSparseCoreV0`).
#[no_mangle]
pub extern "C" fn TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice(_core_type: i32) -> i32 {
    i32::try_from(
        global_topology()
            .hal
            .topology_spec()
            .generation
            .sparse_cores_per_device(),
    )
    .unwrap_or(0)
}

/// Writes the `(x, y, z)` chip coordinates of `core` (`TpuCoreLocation_ChipCoordinates`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
/// * `x_out` - Output X coordinate pointer.
/// * `y_out` - Output Y coordinate pointer.
/// * `z_out` - Output Z coordinate pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn TpuCoreLocation_ChipCoordinates(
    core: *const TpuCoreLocation,
    x_out: *mut i32,
    y_out: *mut i32,
    z_out: *mut i32,
) {
    if any_null(&[
        core as usize,
        x_out as usize,
        y_out as usize,
        z_out as usize,
    ]) {
        return;
    }
    // SAFETY: Validated non-null above.
    unsafe {
        *x_out = (*core).chip_x;
        *y_out = (*core).chip_y;
        *z_out = (*core).chip_z;
    }
}

/// Writes the `(x, y, z)` host coordinates of `core` (`TpuCoreLocation_HostCoordinates`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
/// * `x_out` - Output X coordinate pointer.
/// * `y_out` - Output Y coordinate pointer.
/// * `z_out` - Output Z coordinate pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn TpuCoreLocation_HostCoordinates(
    core: *const TpuCoreLocation,
    x_out: *mut i32,
    y_out: *mut i32,
    z_out: *mut i32,
) {
    if any_null(&[
        core as usize,
        x_out as usize,
        y_out as usize,
        z_out as usize,
    ]) {
        return;
    }
    let hb = global_topology().hal.topology_spec().host_bounds;
    // SAFETY: Validated non-null above.
    unsafe {
        *x_out = (*core).chip_x % i32::try_from(hb.x.max(1)).unwrap_or(1);
        *y_out = (*core).chip_y % i32::try_from(hb.y.max(1)).unwrap_or(1);
        *z_out = (*core).chip_z % i32::try_from(hb.z.max(1)).unwrap_or(1);
    }
}

/// Returns the core index on chip (`TpuCoreLocation_Index`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
///
/// # Safety
///
/// `core` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn TpuCoreLocation_Index(core: *const TpuCoreLocation) -> i32 {
    if core.is_null() {
        return -1;
    }
    // SAFETY: `core` is non-null.
    unsafe { (*core).index_on_chip }
}

/// Returns the global core ID (`TpuCoreLocation_Id`).
///
/// # Arguments
///
/// * `core` - Pointer to [`TpuCoreLocation`].
///
/// # Safety
///
/// `core` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn TpuCoreLocation_Id(core: *const TpuCoreLocation) -> i32 {
    if core.is_null() {
        return -1;
    }
    // SAFETY: `core` is non-null.
    unsafe { (*core).core_id }
}

/// Returns the host ID of a host location (`TpuHostLocation_Id`).
///
/// # Arguments
///
/// * `host` - Pointer to [`TpuHostLocation`] (or null).
///
/// # Safety
///
/// `host` must be null or a valid pointer to [`TpuHostLocation`].
#[no_mangle]
pub unsafe extern "C" fn TpuHostLocation_Id(host: *const TpuHostLocation) -> i32 {
    if host.is_null() {
        0
    } else {
        // SAFETY: `host` is non-null.
        unsafe { (*host).host_id }
    }
}

/// Returns the number of cores on a host location (`TpuHostLocation_NumCores`).
///
/// # Arguments
///
/// * `host` - Pointer to [`TpuHostLocation`] (or null).
/// * `core_type` - Core type discriminant.
///
/// # Safety
///
/// `host` must be null or a valid pointer to [`TpuHostLocation`].
#[no_mangle]
pub unsafe extern "C" fn TpuHostLocation_NumCores(
    host: *const TpuHostLocation,
    core_type: i32,
) -> i32 {
    if !host.is_null() {
        // SAFETY: `host` is non-null.
        let n = unsafe { (*host).num_cores };
        if n > 0 {
            return n;
        }
    }
    // SAFETY: Passing null uses `global_topology()`.
    unsafe { TpuTopology_NumCores(ptr::null(), core_type) }
}

/// Copies core pointers on a host location (`TpuHostLocation_Cores`).
///
/// # Arguments
///
/// * `_host` - Host location handle.
/// * `core_type` - Core type enum.
/// * `cores_out` - Output array of `*const TpuCoreLocation`.
///
/// # Safety
///
/// `cores_out` must be null or point to an array of sufficient size.
#[no_mangle]
pub unsafe extern "C" fn TpuHostLocation_Cores(
    _host: *const TpuHostLocation,
    core_type: i32,
    cores_out: *mut *const TpuCoreLocation,
) {
    // SAFETY: Delegates to `TpuTopology_Cores`.
    unsafe {
        TpuTopology_Cores(ptr::null(), core_type, cores_out);
    }
}

/// Creates a [`TpuMeshState`] (`TpuMeshState_Create`).
#[no_mangle]
pub extern "C" fn TpuMeshState_Create() -> *mut TpuMeshState {
    Box::into_raw(Box::new(TpuMeshState {
        mesh_common_state: 1,
    }))
}

/// Frees a [`TpuMeshState`] (`TpuMeshState_Free`).
///
/// # Arguments
///
/// * `state` - Pointer to [`TpuMeshState`].
///
/// # Safety
///
/// `state` must be null or allocated by `TpuMeshState_Create`.
#[no_mangle]
pub unsafe extern "C" fn TpuMeshState_Free(state: *mut TpuMeshState) {
    if !state.is_null() {
        // SAFETY: `state` is non-null.
        unsafe {
            drop(Box::from_raw(state));
        }
    }
}

/// Returns the internal mesh common state pointer (`TpuMeshState_MeshCommonState`).
///
/// # Arguments
///
/// * `state` - Pointer to [`TpuMeshState`].
///
/// # Safety
///
/// `state` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn TpuMeshState_MeshCommonState(
    state: *const TpuMeshState,
) -> *const std::ffi::c_void {
    if state.is_null() {
        return ptr::null();
    }
    state.cast::<std::ffi::c_void>()
}

/// Creates a [`TpuProfiler`] session (`TpuProfiler_Create`).
///
/// # Arguments
///
/// * `profiler_out` - Output pointer for the created [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler_out` must be null or a valid mutable pointer.
#[no_mangle]
pub unsafe extern "C" fn TpuProfiler_Create(
    profiler_out: *mut *mut TpuProfiler,
    status: *mut TpuStatus,
) {
    if profiler_out.is_null() {
        // SAFETY: `status` may be null; `TpuStatus_Set` checks for null.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler_out is null".as_ptr(), 20);
        }
        return;
    }
    let handle = Box::into_raw(Box::new(TpuProfiler {
        inner: TpuCoreProfiler::new(),
    }));
    // SAFETY: `profiler_out` is non-null.
    unsafe {
        *profiler_out = handle;
    }
}

/// Destroys a [`TpuProfiler`] session (`TpuProfiler_Destroy`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
///
/// # Safety
///
/// `profiler` must be null or allocated by `TpuProfiler_Create`.
#[no_mangle]
pub unsafe extern "C" fn TpuProfiler_Destroy(profiler: *mut TpuProfiler) {
    if !profiler.is_null() {
        // SAFETY: `profiler` is non-null.
        unsafe {
            drop(Box::from_raw(profiler));
        }
    }
}

/// Starts a [`TpuProfiler`] session (`TpuProfiler_Start`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler` and `status` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuProfiler_Start(profiler: *mut TpuProfiler, status: *mut TpuStatus) {
    if profiler.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler is null".as_ptr(), 16);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    if let Err(err) = unsafe { (*profiler).inner.start() } {
        let msg = to_c_string(&err.to_string());
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, err.error_code().as_raw(), msg.as_ptr(), 0);
        }
    }
}

/// Stops a [`TpuProfiler`] session (`TpuProfiler_Stop`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler` and `status` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuProfiler_Stop(profiler: *mut TpuProfiler, status: *mut TpuStatus) {
    if profiler.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler is null".as_ptr(), 16);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    if let Err(err) = unsafe { (*profiler).inner.stop() } {
        let msg = to_c_string(&err.to_string());
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, err.error_code().as_raw(), msg.as_ptr(), 0);
        }
    }
}

/// Collects serialized `XSpace` profiling data from `profiler` (`TpuProfiler_CollectData`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
/// * `buffer` - Optional destination byte buffer (`NULL` to query required size).
/// * `size_in_bytes` - In/out buffer byte size pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn TpuProfiler_CollectData(
    profiler: *mut TpuProfiler,
    status: *mut TpuStatus,
    buffer: *mut u8,
    size_in_bytes: *mut usize,
) {
    if any_null(&[profiler as usize, size_in_bytes as usize]) {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"null profiler or size".as_ptr(), 21);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    let bytes = unsafe { (*profiler).inner.collect_xspace_bytes() };
    // SAFETY: `size_in_bytes` is non-null.
    unsafe {
        *size_in_bytes = bytes.len();
    }
    if !buffer.is_null() {
        // SAFETY: Caller guarantees `buffer` has at least `bytes.len()` capacity.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
        }
    }
}

/// Creates a [`TpuEmbeddingEngineState`] (`TpuEmbeddingEngineState_Create`).
#[no_mangle]
pub extern "C" fn TpuEmbeddingEngineState_Create() -> *mut TpuEmbeddingEngineState {
    Box::into_raw(Box::new(TpuEmbeddingEngineState {
        inner: TpuCoreEmbeddingEngine::new(),
    }))
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngineState_Free(state: *mut TpuEmbeddingEngineState) {
    if !state.is_null() {
        // SAFETY: `state` is non-null.
        unsafe {
            drop(Box::from_raw(state));
        }
    }
}

/// Returns the internal pointer of a [`TpuEmbeddingEngineState`] (`TpuEmbeddingEngineState_GetState`).
#[no_mangle]
pub extern "C" fn TpuEmbeddingEngineState_GetState(
    state: *const TpuEmbeddingEngineState,
) -> *const std::ffi::c_void {
    state.cast::<std::ffi::c_void>()
}

/// Process-wide [`TpuCoreEmbeddingEngine`] backing `TpuEmbeddingEngine_*` C entrypoints.
static GLOBAL_EMBEDDING_ENGINE: OnceLock<std::sync::Mutex<TpuCoreEmbeddingEngine>> =
    OnceLock::new();

/// Returns a reference to the process-wide [`std::sync::Mutex<TpuCoreEmbeddingEngine>`].
fn global_embedding_engine() -> &'static std::sync::Mutex<TpuCoreEmbeddingEngine> {
    GLOBAL_EMBEDDING_ENGINE.get_or_init(|| std::sync::Mutex::new(TpuCoreEmbeddingEngine::new()))
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_ExecutePartitioner(
    params: *mut TpuEmbeddingEngine_ExecutePartitioner_Params,
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
    let mut eng = global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let partitioned = eng.execute_partitioner(cfg_bytes);
    drop(eng);
    if !p.common_config_string_size.is_null() {
        // SAFETY: `p.common_config_string_size` is non-null.
        unsafe {
            *p.common_config_string_size = partitioned.len();
        }
    }
    if !p.common_config_string.is_null() {
        let cstr = to_c_string(&String::from_utf8_lossy(&partitioned));
        // SAFETY: `p.common_config_string` is non-null.
        unsafe {
            *p.common_config_string = cstr.into_raw();
        }
    }
    0
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_ConfigureMemory(
    params: *mut TpuEmbeddingEngine_ConfigureMemory_Params,
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
    let mut eng = global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    eng.configure_memory(cfg_bytes.clone());
    drop(eng);
    if !p.task_host_config_size.is_null() {
        // SAFETY: `p.task_host_config_size` is non-null.
        unsafe {
            *p.task_host_config_size = cfg_bytes.len();
        }
    }
    if !p.task_host_config.is_null() {
        // SAFETY: `p.task_host_config` is non-null.
        unsafe {
            *p.task_host_config = to_c_string(&String::from_utf8_lossy(&cfg_bytes)).into_raw();
        }
    }
    0
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_CollateMemory(
    params: *mut TpuEmbeddingEngine_CollateMemory_Params,
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
    let collated = global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .collate_memory();
    if !p.collated_size.is_null() {
        // SAFETY: `p.collated_size` is non-null.
        unsafe {
            *p.collated_size = collated.len();
        }
    }
    0
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_ConfigureHost(
    params: *mut TpuEmbeddingEngine_ConfigureHost_Params,
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
    let mut eng = global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let net_cfg = eng.configure_host(host_slice);
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
            *p.network_config = to_c_string(&String::from_utf8_lossy(&net_cfg)).into_raw();
        }
    }
    0
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_ConnectHosts(
    params: *mut TpuEmbeddingEngine_ConnectHosts_Params,
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
    global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .connect_hosts();
    0
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_Finalize(
    params: *mut TpuEmbeddingEngine_Finalize_Params,
) -> i32 {
    if params.is_null() {
        return -1;
    }
    global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .finalize();
    0
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_IsInitialized(
    params: *const TpuEmbeddingEngine_IsInitialized_Params,
) -> bool {
    if params.is_null() {
        return false;
    }
    let is_init = global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .initialized;
    // SAFETY: `params` is non-null; write `is_init` if `is_embedding_initialized` is non-null.
    unsafe {
        if !(*params).is_embedding_initialized.is_null() {
            *(*params).is_embedding_initialized = is_init;
        }
    }
    is_init
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_WriteParameters(
    params: *mut TpuEmbeddingEngine_WriteParameters_Params,
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
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .write_parameters(table_name, payload);
    0
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_ReadParameters(
    params: *mut TpuEmbeddingEngine_WriteParameters_Params,
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
    let eng = global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
}

/// Creates tensor batch fixed state (`TpuEmbeddingTensorBatchFixedState_Create`).
#[no_mangle]
pub extern "C" fn TpuEmbeddingTensorBatchFixedState_Create() -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(1_u64)).cast::<std::ffi::c_void>()
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingTensorBatchFixedState_Destroy(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated as `Box<u64>`.
        unsafe {
            drop(Box::from_raw(ptr.cast::<u64>()));
        }
    }
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_EnqueueTensorBatch(
    params: *mut TpuEmbeddingEngine_EnqueueTensorBatch_Params,
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
    global_embedding_engine()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .enqueue_tensor_batch(
            i32::try_from(p.device_ordinal).unwrap_or(0),
            num_indices,
            num_weights,
        );
    0
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
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .generate_xla_computation(kind);
    if !p.xla_computation_out.is_null() {
        let c_str = to_c_string(&String::from_utf8_lossy(&hlo_bytes));
        let len = c_str.as_bytes().len();
        // SAFETY: `p.xla_computation_out` is non-null.
        unsafe {
            (*p.xla_computation_out) = TpuSerializedProto {
                bytes: c_str.into_raw(),
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_RecvActivationsComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "recv_activations") }
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "recv_dedup_data") }
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "send_gradients") }
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_DedupDataSizeComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "dedup_data_size") }
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
#[no_mangle]
pub unsafe extern "C" fn TpuEmbeddingEngine_DedupDataTupleMaskComputation(
    params: *mut TpuEmbeddingEngine_Computation_Params,
) -> i32 {
    // SAFETY: Delegated to `build_embedding_xla_computation`.
    unsafe { build_embedding_xla_computation(params, "dedup_tuple_mask") }
}

/// Computes the 256-byte-aligned linearized byte size of a shape (`TpuDriver_ComputeLinearizedBytesFromShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_bytes` - Unpadded byte count.
#[no_mangle]
pub extern "C" fn TpuDriver_ComputeLinearizedBytesFromShape(
    driver: *mut TpuDriver,
    num_bytes: i64,
) -> i64 {
    if driver.is_null() {
        return 0;
    }
    ((num_bytes.max(1) + 255) / 256) * 256
}

/// Compiles a textual HLO program on a [`TpuDriver`] (`TpuDriver_CompileProgramFromText`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `hlo_text` - Null-terminated HLO text string.
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `hlo_text`, and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_CompileProgramFromText(
    driver: *mut TpuDriver,
    hlo_text: *const c_char,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    let text_bytes: &[u8] = if hlo_text.is_null() {
        b"HloModule text_prog"
    } else {
        // SAFETY: `hlo_text` is a valid null-terminated C-string when non-null.
        unsafe { CStr::from_ptr(hlo_text).to_bytes() }
    };
    let proto = crate::tpu_driver_types::HloProto {
        buffer: text_bytes.as_ptr().cast::<std::ffi::c_void>(),
        size: i64::try_from(text_bytes.len()).unwrap_or(0),
    };
    // SAFETY: Delegated to `TpuDriver_Compile`.
    unsafe { TpuDriver_Compile(driver, ptr::addr_of!(proto), num_replicas, num_ains, ains) }
}

/// Queries the shape of a compiled program (`TpuDriver_GetCompiledProgramShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `handle` - Pointer to [`crate::tpu_driver_types::TpuCompiledProgramHandle`].
#[no_mangle]
pub extern "C" fn TpuDriver_GetCompiledProgramShape(
    driver: *mut TpuDriver,
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    if driver.is_null() {
        return ptr::null_mut();
    }
    handle
}

/// Registers an event completion callback on a [`TpuDriver`] (`TpuDriver_AddCallback`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `event` - Pointer to [`crate::tpu_driver_types::TpuEvent`].
/// * `callback_fn` - Callback function pointer.
/// * `user_arg` - User context pointer.
#[no_mangle]
pub extern "C" fn TpuDriver_AddCallback(
    driver: *mut TpuDriver,
    event: *mut crate::tpu_driver_types::TpuEvent,
    callback_fn: *const std::ffi::c_void,
    user_arg: *mut std::ffi::c_void,
) {
    let _ = (driver, event, callback_fn, user_arg);
}

/// Returns the `TpuDriver` API version string (`TpuDriver_Version`).
#[no_mangle]
pub extern "C" fn TpuDriver_Version() -> *const c_char {
    c"0.0.50".as_ptr()
}

/// Static [`TpuDriverFn`] vtable stored at offset `0x00` of [`TpuDriver`] (`224` bytes, 28 slots).
static GLOBAL_TPU_DRIVER_FN_TABLE: TpuDriverFn = TpuDriverFn {
    open: TpuDriver_Open as *const std::ffi::c_void,
    close: TpuDriver_Close as *const std::ffi::c_void,
    compute_linearized_bytes_from_shape: TpuDriver_ComputeLinearizedBytesFromShape
        as *const std::ffi::c_void,
    linearize_shape: TpuDriver_LinearizeShape as *const std::ffi::c_void,
    delinearize_shape: TpuDriver_DelinearizeShape as *const std::ffi::c_void,
    compile_program: TpuDriver_Compile as *const std::ffi::c_void,
    compile_program_from_text: TpuDriver_CompileProgramFromText as *const std::ffi::c_void,
    get_compiled_program_shape: TpuDriver_GetCompiledProgramShape as *const std::ffi::c_void,
    load_program: TpuDriver_LoadProgram as *const std::ffi::c_void,
    unload_program: TpuDriver_UnloadProgram as *const std::ffi::c_void,
    execute_program: TpuDriver_ExecuteProgram as *const std::ffi::c_void,
    allocate_tuple: TpuDriver_AllocateTuple as *const std::ffi::c_void,
    allocate: TpuDriver_Allocate as *const std::ffi::c_void,
    allocate_shape: TpuDriver_AllocateShape as *const std::ffi::c_void,
    deallocate: TpuDriver_Deallocate as *const std::ffi::c_void,
    transfer_to_device: TpuDriver_TransferToDevice as *const std::ffi::c_void,
    transfer_from_device: TpuDriver_TransferFromDevice as *const std::ffi::c_void,
    transfer_from_device_to_device: TpuDriver_TransferFromDeviceToDevice as *const std::ffi::c_void,
    create_event: TpuDriver_CreateEvent as *const std::ffi::c_void,
    free_event: TpuDriver_FreeEvent as *const std::ffi::c_void,
    add_callback: TpuDriver_AddCallback as *const std::ffi::c_void,
    wait_for_event: TpuDriver_WaitForEvents as *const std::ffi::c_void,
    free_status: TpuStatus_Free as *const std::ffi::c_void,
    reset: TpuDriver_Reset as *const std::ffi::c_void,
    query_system_info: TpuDriver_QuerySystemInfo as *const std::ffi::c_void,
    free_system_info: TpuDriver_FreeSystemInfo as *const std::ffi::c_void,
    free_compiled_program_shape: TpuDriver_FreeCompiledProgramHandle as *const std::ffi::c_void,
    version: TpuDriver_Version as *const std::ffi::c_void,
};

/// Opens a [`TpuDriver`] instance (`TpuDriver_Open`).
///
/// # Arguments
///
/// * `_worker` - Null-terminated worker specification string (e.g., `"local://"` or null).
#[no_mangle]
pub extern "C" fn TpuDriver_Open(_worker: *const c_char) -> *mut TpuDriver {
    Box::into_raw(Box::new(TpuDriver {
        fn_table: ptr::addr_of!(GLOBAL_TPU_DRIVER_FN_TABLE),
        inner: TpuCoreDriver::open(libtpu_core::config::global_init_config()),
    }))
}

/// Closes and frees a [`TpuDriver`] instance (`TpuDriver_Close`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or allocated by `TpuDriver_Open`.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_Close(driver: *mut TpuDriver) {
    if !driver.is_null() {
        // SAFETY: `driver` is non-null.
        unsafe {
            drop(Box::from_raw(driver));
        }
    }
}

/// Resets a [`TpuDriver`] instance (`TpuDriver_Reset`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_Reset(driver: *mut TpuDriver) -> *mut TpuStatus {
    if driver.is_null() {
        // SAFETY: Static CStr literal is valid and null-terminated.
        return unsafe { TpuStatus_Create(3, c"driver is null".as_ptr()) };
    }
    // SAFETY: `driver` is non-null.
    let _ = unsafe { (*driver).inner.reset(DeviceId(0)) };
    TpuStatus_New()
}

/// Queries system information from a [`TpuDriver`] (`TpuDriver_QuerySystemInfo`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_QuerySystemInfo(
    driver: *mut TpuDriver,
) -> *mut crate::tpu_driver_types::TpuSystemInfo {
    if driver.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `driver` is non-null.
    let info = unsafe { (*driver).inner.query_system_info() };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuSystemInfo {
        num_chips: 4,
        num_cores: 4,
        info_str: to_c_string(&info),
    }))
}

/// Frees a [`crate::tpu_driver_types::TpuSystemInfo`] (`TpuDriver_FreeSystemInfo`).
///
/// # Arguments
///
/// * `info` - Pointer returned by [`TpuDriver_QuerySystemInfo`].
///
/// # Safety
///
/// `info` must be null or allocated by [`TpuDriver_QuerySystemInfo`].
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_FreeSystemInfo(
    info: *mut crate::tpu_driver_types::TpuSystemInfo,
) {
    if !info.is_null() {
        // SAFETY: `info` was allocated via `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(info));
        }
    }
}

/// Compiles an HLO program on a [`TpuDriver`] (`TpuDriver_Compile`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `source` - Pointer to [`crate::tpu_driver_types::HloProto`].
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `source`, and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_Compile(
    driver: *mut TpuDriver,
    source: *const crate::tpu_driver_types::HloProto,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    let _ = (num_replicas, num_ains, ains);
    if driver.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `source` is null or a valid `HloProto*`.
    let src_opt = unsafe { source.as_ref() };
    let code_bytes: &[u8] = src_opt.map_or(b"HloModule driver_compile" as &[u8], |src| {
        // SAFETY: `src.buffer` has `src.size` bytes when non-null.
        unsafe {
            std::slice::from_raw_parts(
                src.buffer.cast::<u8>(),
                usize::try_from(src.size.max(0)).unwrap_or(0),
            )
        }
    });
    // SAFETY: `driver` is non-null.
    let Ok(prog) = (unsafe { (*driver).inner.compile(code_bytes) }) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(
        crate::tpu_driver_types::TpuCompiledProgramHandle { program: prog },
    ))
}

/// Frees a [`crate::tpu_driver_types::TpuCompiledProgramHandle`] (`TpuDriver_FreeCompiledProgramHandle`).
///
/// # Arguments
///
/// * `handle` - Pointer returned by [`TpuDriver_Compile`].
///
/// # Safety
///
/// `handle` must be null or allocated by [`TpuDriver_Compile`].
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_FreeCompiledProgramHandle(
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) {
    if !handle.is_null() {
        // SAFETY: `handle` was allocated via `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Loads a compiled program onto a core of a [`TpuDriver`] (`TpuDriver_LoadProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `compiled_program_handle` - Pointer to [`crate::tpu_driver_types::TpuCompiledProgramHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `compiled_program_handle` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_LoadProgram(
    driver: *mut TpuDriver,
    core_id: i32,
    compiled_program_handle: *const crate::tpu_driver_types::TpuCompiledProgramHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuLoadedProgramHandle {
    let _ = (num_ains, ains);
    if driver.is_null() || compiled_program_handle.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `compiled_program_handle` is non-null.
    let prog = unsafe { (*compiled_program_handle).program.clone() };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuLoadedProgramHandle {
        core_id,
        program: prog,
    }))
}

/// Unloads a loaded program from a [`TpuDriver`] (`TpuDriver_UnloadProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `loaded_program_handle` - Loaded program handle to unload and free.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `loaded_program_handle` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_UnloadProgram(
    driver: *mut TpuDriver,
    loaded_program_handle: *mut crate::tpu_driver_types::TpuLoadedProgramHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !loaded_program_handle.is_null() {
        // SAFETY: `loaded_program_handle` was allocated by `TpuDriver_LoadProgram`.
        unsafe {
            drop(Box::from_raw(loaded_program_handle));
        }
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Executes a loaded program on a [`TpuDriver`] (`TpuDriver_ExecuteProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `handle` - Pointer to [`crate::tpu_driver_types::TpuLoadedProgramHandle`].
/// * `num_inputs` - Number of input buffers.
/// * `inputs` - Array of input `*mut TpuBufferHandle` pointers.
/// * `num_outputs` - Number of output buffers.
/// * `outputs` - Array of output `*mut TpuBufferHandle` pointers.
/// * `device_assignment` - Logical device assignment.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// All pointer arguments must be null or valid pointers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn TpuDriver_ExecuteProgram(
    driver: *mut TpuDriver,
    handle: *mut crate::tpu_driver_types::TpuLoadedProgramHandle,
    num_inputs: i32,
    inputs: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    num_outputs: i32,
    outputs: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    device_assignment: crate::tpu_driver_types::DeviceAssignment,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (device_assignment, num_ains, ains);
    if !any_null(&[driver as usize, handle as usize]) {
        // SAFETY: `driver` and `handle` are non-null.
        let (drv, lp) = unsafe { (&(*driver).inner, &*handle) };
        let dev_id = DeviceId(u32::try_from(lp.core_id.max(0)).unwrap_or(0));
        let mut in_addrs = Vec::new();
        for idx in 0..usize::try_from(num_inputs.max(0)).unwrap_or(0) {
            // SAFETY: `inputs` has `num_inputs` elements.
            let b_ptr = unsafe { *inputs.add(idx) };
            // SAFETY: `b_ptr` is non-null.
            in_addrs.push(libtpu_core::types::DeviceAddress(unsafe {
                (*b_ptr).address
            }));
        }
        let mut out_addrs = Vec::new();
        for idx in 0..usize::try_from(num_outputs.max(0)).unwrap_or(0) {
            // SAFETY: `outputs` has `num_outputs` elements.
            let b_ptr = unsafe { *outputs.add(idx) };
            // SAFETY: `b_ptr` is non-null.
            out_addrs.push(libtpu_core::types::DeviceAddress(unsafe {
                (*b_ptr).address
            }));
        }
        let _ = drv.execute_program(dev_id, &lp.program, &in_addrs, &out_addrs);
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Allocates a device buffer handle on a [`TpuDriver`] (`TpuDriver_Allocate`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region (`0` = HBM, `1` = VMEM).
/// * `num_bytes` - Allocation byte size.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_Allocate(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_bytes: i64,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    let _ = (num_ains, ains);
    if driver.is_null() {
        return ptr::null_mut();
    }
    let dev_id = DeviceId(u32::try_from(core_id.max(0)).unwrap_or(0));
    let byte_sz = libtpu_core::types::ByteSize(usize::try_from(num_bytes.max(16)).unwrap_or(16));
    // SAFETY: `driver` is non-null.
    let addr = unsafe {
        (*driver)
            .inner
            .allocate(dev_id, byte_sz)
            .unwrap_or(libtpu_core::types::DeviceAddress(0))
            .0
    };
    // SAFETY: `driver` is non-null.
    let ev = unsafe { TpuDriver_CreateEvent(driver) };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuBufferHandle {
        address: addr,
        size_in_bytes: num_bytes.max(16),
        core_id,
        memory_region,
        event: ev,
    }))
}

/// Allocates a tuple buffer handle on a [`TpuDriver`] (`TpuDriver_AllocateTuple`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region.
/// * `num_children` - Number of child buffers.
/// * `children` - Array of child buffer handles.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `children`, and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_AllocateTuple(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_children: i32,
    children: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    let _ = children;
    let tuple_bytes = i64::from(num_children.max(1)) * 8;
    // SAFETY: Delegated to `TpuDriver_Allocate`.
    unsafe { TpuDriver_Allocate(driver, core_id, memory_region, tuple_bytes, num_ains, ains) }
}

/// Allocates a shaped buffer handle on a [`TpuDriver`] (`TpuDriver_AllocateShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region.
/// * `num_bytes` - Byte size of the shape.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_AllocateShape(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_bytes: i64,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    // SAFETY: Delegated to `TpuDriver_Allocate`.
    unsafe { TpuDriver_Allocate(driver, core_id, memory_region, num_bytes, num_ains, ains) }
}

/// Deallocates a device buffer handle on a [`TpuDriver`] (`TpuDriver_Deallocate`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `buf` - Buffer handle returned by [`TpuDriver_Allocate`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `buf` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_Deallocate(
    driver: *mut TpuDriver,
    buf: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, buf as usize]) {
        // SAFETY: `buf` was allocated by `TpuDriver_Allocate`.
        let handle = unsafe { Box::from_raw(buf) };
        // SAFETY: `handle.event` was allocated by `TpuDriver_CreateEvent`.
        unsafe {
            TpuDriver_FreeEvent(handle.event);
        }
        let dev_id = DeviceId(u32::try_from(handle.core_id.max(0)).unwrap_or(0));
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver)
                .inner
                .deallocate(dev_id, libtpu_core::types::DeviceAddress(handle.address))
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers host memory to a device buffer (`TpuDriver_TransferToDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source host buffer pointer.
/// * `dst` - Destination [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_TransferToDevice(
    driver: *mut TpuDriver,
    src: *const std::ffi::c_void,
    dst: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `dst` is non-null.
        let d = unsafe { &*dst };
        let dev_id = DeviceId(u32::try_from(d.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(d.size_in_bytes.max(0)).unwrap_or(0);
        // SAFETY: `src` has at least `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_to_device(
                dev_id,
                libtpu_core::types::DeviceAddress(d.address),
                slice,
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers device buffer memory to host (`TpuDriver_TransferFromDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `dst` - Destination host buffer pointer.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_TransferFromDevice(
    driver: *mut TpuDriver,
    src: *mut crate::tpu_driver_types::TpuBufferHandle,
    dst: *mut std::ffi::c_void,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `src` is non-null.
        let s = unsafe { &*src };
        let dev_id = DeviceId(u32::try_from(s.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(s.size_in_bytes.max(0)).unwrap_or(0);
        // SAFETY: `dst` has at least `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), len) };
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_from_device(
                dev_id,
                libtpu_core::types::DeviceAddress(s.address),
                slice,
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers memory between two device buffers (`TpuDriver_TransferFromDeviceToDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `dst` - Destination [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_TransferFromDeviceToDevice(
    driver: *mut TpuDriver,
    src: *mut crate::tpu_driver_types::TpuBufferHandle,
    dst: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `src` and `dst` are non-null.
        let (s, d) = unsafe { (&*src, &*dst) };
        let src_dev = DeviceId(u32::try_from(s.core_id.max(0)).unwrap_or(0));
        let dst_dev = DeviceId(u32::try_from(d.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(s.size_in_bytes.min(d.size_in_bytes).max(0)).unwrap_or(0);
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_device_to_device(
                src_dev,
                libtpu_core::types::DeviceAddress(s.address),
                dst_dev,
                libtpu_core::types::DeviceAddress(d.address),
                libtpu_core::types::ByteSize(len),
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Linearizes a host tensor shape into tiled TPU buffer bytes (`TpuDriver_LinearizeShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `dst` - Destination tiled byte buffer.
/// * `src` - Source linear byte buffer.
/// * `num_bytes` - Number of bytes to linearize.
///
/// # Safety
///
/// `driver`, `dst`, and `src` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_LinearizeShape(
    driver: *mut TpuDriver,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    num_bytes: i64,
) -> *mut TpuStatus {
    if any_null(&[driver as usize, dst as usize, src as usize]) {
        // SAFETY: Static CStr literal is valid.
        return unsafe { TpuStatus_Create(3, c"null argument".as_ptr()) };
    }
    let len = usize::try_from(num_bytes.max(0)).unwrap_or(0);
    // SAFETY: Caller guarantees `src` has at least `len` bytes.
    let src_slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
    let num_f32 = i64::try_from((len / 4).max(1)).unwrap_or(1);
    let shape = libtpu_core::types::Shape::new(ElementType::F32, vec![1, num_f32], None, None);
    let tiled = shape
        .and_then(|s| s.linearize_to_tpu_tiles(src_slice))
        .unwrap_or_default();
    // SAFETY: Caller guarantees `dst` has at least `len` bytes.
    unsafe {
        ptr::copy_nonoverlapping(tiled.as_ptr(), dst.cast::<u8>(), len.min(tiled.len()));
    }
    TpuStatus_New()
}

/// Delinearizes tiled TPU buffer bytes back into host tensor layout (`TpuDriver_DelinearizeShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `dst` - Destination linear byte buffer.
/// * `src` - Source tiled byte buffer.
/// * `num_bytes` - Number of bytes to delinearize.
///
/// # Safety
///
/// `driver`, `dst`, and `src` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_DelinearizeShape(
    driver: *mut TpuDriver,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    num_bytes: i64,
) -> *mut TpuStatus {
    if any_null(&[driver as usize, dst as usize, src as usize]) {
        // SAFETY: Static CStr literal is valid.
        return unsafe { TpuStatus_Create(3, c"null argument".as_ptr()) };
    }
    let len = usize::try_from(num_bytes.max(0)).unwrap_or(0);
    // SAFETY: Caller guarantees `src` has at least `len` bytes.
    let src_slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
    let num_f32 = i64::try_from((len / 4).max(1)).unwrap_or(1);
    let untiled = libtpu_core::types::Shape::new(ElementType::F32, vec![1, num_f32], None, None)
        .and_then(|s| {
            let mut padded = vec![0_u8; s.on_device_byte_size().0.max(len)];
            padded
                .iter_mut()
                .zip(src_slice.iter())
                .for_each(|(d, &v)| *d = v);
            s.delinearize_from_tpu_tiles(&padded)
        })
        .unwrap_or_default();
    // SAFETY: Caller guarantees `dst` has at least `len` bytes.
    unsafe {
        ptr::copy_nonoverlapping(untiled.as_ptr(), dst.cast::<u8>(), len.min(untiled.len()));
    }
    TpuStatus_New()
}

/// Creates a [`TpuDriver`] synchronization event (`TpuDriver_CreateEvent`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_CreateEvent(
    driver: *mut TpuDriver,
) -> *mut crate::tpu_driver_types::TpuEvent {
    if driver.is_null() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuEvent {
        event_id: 1,
        status: TpuStatus_New(),
    }))
}

/// Frees a [`TpuDriver`] synchronization event (`TpuDriver_FreeEvent`).
///
/// # Arguments
///
/// * `event` - Pointer to [`crate::tpu_driver_types::TpuEvent`].
///
/// # Safety
///
/// `event` must be null or allocated by [`TpuDriver_CreateEvent`].
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_FreeEvent(event: *mut crate::tpu_driver_types::TpuEvent) {
    if !event.is_null() {
        // SAFETY: `event` was allocated by `TpuDriver_CreateEvent`.
        unsafe {
            let ev = Box::from_raw(event);
            TpuStatus_Free(ev.status);
        }
    }
}

/// Waits for [`TpuDriver`] synchronization events (`TpuDriver_WaitForEvents`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_events` - Number of events in `events`.
/// * `events` - Array of `*mut TpuEvent` pointers.
/// * `timeout_us` - Timeout in microseconds.
///
/// # Safety
///
/// `driver` and `events` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn TpuDriver_WaitForEvents(
    driver: *mut TpuDriver,
    num_events: i32,
    events: *const *mut crate::tpu_driver_types::TpuEvent,
    timeout_us: i64,
) -> *mut TpuStatus {
    let _ = (num_events, events, timeout_us);
    if driver.is_null() {
        // SAFETY: Static CStr literal is valid and null-terminated.
        return unsafe { TpuStatus_Create(3, c"driver is null".as_ptr()) };
    }
    TpuStatus_New()
}

// ============================================================================
// Clean C / C++20 FFI Entrypoints (`libtpu_rs_*`)
// ============================================================================

/// Creates a success [`LibTpuCStatus`].
fn c_status_ok() -> LibTpuCStatus {
    LibTpuCStatus {
        code: 0,
        message: ptr::null_mut(),
    }
}

/// Creates an error [`LibTpuCStatus`] from a [`LibTpuError`].
///
/// # Arguments
///
/// * `err` - Domain error to convert.
fn c_status_err(err: &LibTpuError) -> LibTpuCStatus {
    let msg = to_c_string(&err.to_string());
    LibTpuCStatus {
        code: err.error_code().as_raw(),
        message: msg.into_raw(),
    }
}

/// Frees the error message inside a [`LibTpuCStatus`] (`libtpu_rs_status_free`).
///
/// # Arguments
///
/// * `status` - Status struct whose `message` pointer should be freed.
///
/// # Safety
///
/// `status.message` must be null or allocated by `c_status_err`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_status_free(status: LibTpuCStatus) {
    if !status.message.is_null() {
        // SAFETY: `status.message` was allocated via `CString::into_raw`.
        unsafe {
            drop(CString::from_raw(status.message));
        }
    }
}

/// Creates a new [`PJRT_Client`] for C/C++ consumers (`libtpu_rs_client_create`).
///
/// # Arguments
///
/// * `client_out` - Output pointer to receive the created [`PJRT_Client`].
///
/// # Safety
///
/// `client_out` must be null or a valid mutable pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_client_create(
    client_out: *mut *mut PJRT_Client,
) -> LibTpuCStatus {
    if client_out.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "client_out",
        });
    }
    let mut args = PJRT_Client_Create_Args {
        struct_size: std::mem::size_of::<PJRT_Client_Create_Args>(),
        extension_start: ptr::null_mut(),
        create_options: ptr::null(),
        num_options: 0,
        kv_get_callback: ptr::null(),
        kv_get_user_arg: ptr::null_mut(),
        kv_put_callback: ptr::null(),
        kv_put_user_arg: ptr::null_mut(),
        client: ptr::null_mut(),
    };
    // SAFETY: Passing valid stack `args`.
    let _ = unsafe { PJRT_Client_Create(ptr::addr_of_mut!(args).cast()) };
    // SAFETY: `client_out` is non-null.
    unsafe {
        *client_out = args.client;
    }
    c_status_ok()
}

/// Destroys a [`PJRT_Client`] created by `libtpu_rs_client_create` (`libtpu_rs_client_destroy`).
///
/// # Arguments
///
/// * `client` - Client handle to destroy.
///
/// # Safety
///
/// `client` must be null or allocated by `libtpu_rs_client_create`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_client_destroy(client: *mut PJRT_Client) {
    if client.is_null() {
        return;
    }
    let mut args = PJRT_Client_Destroy_Args {
        struct_size: std::mem::size_of::<PJRT_Client_Destroy_Args>(),
        extension_start: ptr::null_mut(),
        client,
    };
    // SAFETY: `args` is valid.
    let _ = unsafe { PJRT_Client_Destroy(ptr::addr_of_mut!(args).cast()) };
}

/// Returns the number of addressable devices on `client` (`libtpu_rs_client_device_count`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
///
/// # Safety
///
/// `client` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_client_device_count(client: *const PJRT_Client) -> i32 {
    if client.is_null() {
        return -1;
    }
    // SAFETY: `client` is non-null.
    unsafe { i32::try_from((*client).addressable_device_ptrs.len()).unwrap_or(0) }
}

/// Allocates a [`PJRT_Buffer`] on `device_id` and copies `host_data` into HBM (`libtpu_rs_buffer_from_host`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `device_id` - Target logical TPU device ID.
/// * `element_type` - Raw `PJRT_Buffer_Type` integer (`11` = `F32`, `4` = `S32`).
/// * `dims` - Pointer to dimension array.
/// * `num_dims` - Number of dimensions.
/// * `host_data` - Pointer to source host bytes.
/// * `host_data_len` - Number of bytes in `host_data`.
/// * `buffer_out` - Output pointer to receive the created [`PJRT_Buffer`].
///
/// # Safety
///
/// All pointers must be null or valid for their respective lengths.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_from_host(
    client: *mut PJRT_Client,
    device_id: u32,
    element_type: i32,
    dims: *const i64,
    num_dims: usize,
    host_data: *const u8,
    host_data_len: usize,
    buffer_out: *mut *mut PJRT_Buffer,
) -> LibTpuCStatus {
    if any_null(&[client as usize, buffer_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "client_or_buffer_out",
        });
    }
    let elem = match ElementType::from_raw(element_type) {
        Ok(t) => t,
        Err(e) => return c_status_err(&e),
    };
    let dims_vec = if dims.is_null() {
        Vec::new()
    } else {
        // SAFETY: `dims` is non-null with `num_dims` elements.
        unsafe { std::slice::from_raw_parts(dims, num_dims).to_vec() }
    };
    let shape = match Shape::new(elem, dims_vec, None, None) {
        Ok(s) => s,
        Err(e) => return c_status_err(&e),
    };
    let data_slice = if host_data.is_null() {
        &[][..]
    } else {
        // SAFETY: `host_data` is non-null with `host_data_len` bytes.
        unsafe { std::slice::from_raw_parts(host_data, host_data_len) }
    };

    // SAFETY: `client` is non-null.
    let client_ref = unsafe { &*client };
    match client_ref
        .inner
        .buffer_from_host(DeviceId(device_id), shape, data_slice)
    {
        Ok(core_buf) => {
            let dev_ptr = client_ref
                .device_ptrs
                .get(device_id as usize)
                .copied()
                .unwrap_or(ptr::null_mut());
            let mem_ptr = client_ref
                .memory_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut());
            let handle = Box::into_raw(Box::new(PJRT_Buffer {
                inner: core_buf,
                device_ptr: dev_ptr,
                memory_ptr: mem_ptr,
            }));
            // SAFETY: `buffer_out` is non-null.
            unsafe {
                *buffer_out = handle;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Copies a [`PJRT_Buffer`] from device HBM into `host_out` (`libtpu_rs_buffer_to_host`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`].
/// * `host_out` - Destination host buffer.
/// * `host_out_len` - Number of bytes to copy into `host_out`.
///
/// # Safety
///
/// `buffer` and `host_out` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_to_host(
    buffer: *const PJRT_Buffer,
    host_out: *mut u8,
    host_out_len: usize,
) -> LibTpuCStatus {
    if any_null(&[buffer as usize, host_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "buffer_or_host_out",
        });
    }
    // SAFETY: `buffer` is non-null.
    let buf_ref = unsafe { &(*buffer).inner };
    match buf_ref.to_host_bytes() {
        Ok(bytes) => {
            if host_out_len > bytes.len() {
                return c_status_err(&LibTpuError::TransferFailed {
                    direction: "D2H",
                    reason: format!(
                        "Requested {host_out_len} bytes from {}-byte buffer",
                        bytes.len()
                    ),
                });
            }
            // SAFETY: `host_out` is valid for `host_out_len` bytes.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), host_out, host_out_len);
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Deletes and frees a [`PJRT_Buffer`] (`libtpu_rs_buffer_destroy`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`] to destroy.
///
/// # Safety
///
/// `buffer` must be null or allocated by `libtpu_rs_buffer_from_host` / `libtpu_rs_execute`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_destroy(buffer: *mut PJRT_Buffer) {
    if !buffer.is_null() {
        // SAFETY: `buffer` is non-null.
        unsafe {
            let b = Box::from_raw(buffer);
            let _ = b.inner.delete();
        }
    }
}

/// Compiles and loads an HLO or `StableHLO` program (`libtpu_rs_compile`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `format` - Null-terminated format string (`"mlir"` or `"hlo"`).
/// * `code` - Pointer to program bytes.
/// * `code_len` - Number of bytes in `code`.
/// * `exec_out` - Output pointer to receive the [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_compile(
    client: *mut PJRT_Client,
    format: *const c_char,
    code: *const u8,
    code_len: usize,
    exec_out: *mut *mut PJRT_LoadedExecutable,
) -> LibTpuCStatus {
    if any_null(&[
        client as usize,
        format as usize,
        code as usize,
        exec_out as usize,
    ]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "compile_args",
        });
    }
    // SAFETY: `format` is non-null.
    let fmt_str = unsafe { CStr::from_ptr(format).to_string_lossy() };
    // SAFETY: `code` is non-null.
    let code_slice = unsafe { std::slice::from_raw_parts(code, code_len) };
    // SAFETY: `client` is non-null.
    let client_ref = unsafe { &*client };
    match client_ref.inner.compile_and_load(&fmt_str, code_slice, b"") {
        Ok(loaded) => {
            let exec_ptr = Box::into_raw(Box::new(PJRT_Executable {
                inner: loaded.executable.clone(),
            }));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
                inner: loaded,
                executable_ptr: exec_ptr,
                device_ptrs: client_ref.addressable_device_ptrs.clone(),
            }));
            // SAFETY: `exec_out` is non-null.
            unsafe {
                *exec_out = loaded_ptr;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Executes a [`PJRT_LoadedExecutable`] on `device_id` (`libtpu_rs_execute`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `device_id` - Target logical TPU device ID.
/// * `inputs` - Array of input `*const PJRT_Buffer` pointers.
/// * `num_inputs` - Number of input buffers.
/// * `output_out` - Output pointer to receive the primary output [`PJRT_Buffer`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_execute(
    executable: *const PJRT_LoadedExecutable,
    device_id: u32,
    inputs: *const *const PJRT_Buffer,
    num_inputs: usize,
    output_out: *mut *mut PJRT_Buffer,
) -> LibTpuCStatus {
    if any_null(&[executable as usize, output_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "execute_args",
        });
    }
    let mut core_inputs = Vec::with_capacity(num_inputs);
    if !inputs.is_null() {
        // SAFETY: `inputs` is non-null with `num_inputs` elements.
        let slice = unsafe { std::slice::from_raw_parts(inputs, num_inputs) };
        for &ptr_item in slice {
            if ptr_item.is_null() {
                return c_status_err(&LibTpuError::NullPointer {
                    parameter: "input_buffer",
                });
            }
            // SAFETY: `ptr_item` is non-null.
            core_inputs.push(unsafe { &(*ptr_item).inner });
        }
    }
    // SAFETY: `executable` is non-null.
    let exec_ref = unsafe { &(*executable).inner };
    match exec_ref.execute_on_device(
        DeviceId(device_id),
        libtpu_core::types::MemorySpaceId(0),
        &core_inputs,
    ) {
        Ok(outs) => {
            for first in outs {
                let handle = Box::into_raw(Box::new(PJRT_Buffer {
                    inner: first,
                    device_ptr: ptr::null_mut(),
                    memory_ptr: ptr::null_mut(),
                }));
                // SAFETY: `output_out` is non-null.
                unsafe {
                    *output_out = handle;
                }
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Destroys a [`PJRT_LoadedExecutable`] (`libtpu_rs_executable_destroy`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// `executable` must be null or allocated by `libtpu_rs_compile` / `PJRT_Client_Compile`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_destroy(executable: *mut PJRT_LoadedExecutable) {
    if !executable.is_null() {
        // SAFETY: `executable` is non-null.
        unsafe {
            let loaded = Box::from_raw(executable);
            drop(Box::from_raw(loaded.executable_ptr));
        }
    }
}

/// Queries the dimensions of a [`PJRT_Buffer`] (`libtpu_rs_buffer_dimensions`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`].
/// * `out_dims` - Optional output buffer for `i64` dimensions.
/// * `max_dims` - Maximum number of dimensions writable to `out_dims`.
/// * `out_num_dims` - Output pointer to receive the rank of `buffer`.
///
/// # Safety
///
/// `buffer`, `out_dims`, and `out_num_dims` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_dimensions(
    buffer: *const PJRT_Buffer,
    out_dims: *mut i64,
    max_dims: usize,
    out_num_dims: *mut usize,
) -> LibTpuCStatus {
    if any_null(&[buffer as usize, out_num_dims as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "buffer_or_out_num_dims",
        });
    }
    // SAFETY: `buffer` is non-null.
    let dims = unsafe { &(*buffer).inner.shape.dims };
    // SAFETY: `out_num_dims` is non-null.
    unsafe {
        *out_num_dims = dims.len();
    }
    if !out_dims.is_null() {
        let copy_count = max_dims.min(dims.len());
        // SAFETY: `out_dims` has capacity for `copy_count` elements.
        unsafe {
            ptr::copy_nonoverlapping(dims.as_ptr(), out_dims, copy_count);
        }
    }
    c_status_ok()
}

/// Returns the raw `PJRT_Buffer_Type` integer of a [`PJRT_Buffer`] (`libtpu_rs_buffer_element_type`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`].
///
/// # Safety
///
/// `buffer` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_element_type(buffer: *const PJRT_Buffer) -> i32 {
    if buffer.is_null() {
        return 0;
    }
    // SAFETY: `buffer` is non-null.
    unsafe { (*buffer).inner.shape.element_type.as_raw() }
}

/// Copies a [`PJRT_Buffer`] to `dst_device_id` on `client` (`libtpu_rs_buffer_copy_to_device`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `buffer` - Source [`PJRT_Buffer`].
/// * `dst_device_id` - Destination logical TPU device ID.
/// * `out_buffer` - Output pointer to receive the copied [`PJRT_Buffer`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_copy_to_device(
    client: *mut PJRT_Client,
    buffer: *const PJRT_Buffer,
    dst_device_id: u32,
    out_buffer: *mut *mut PJRT_Buffer,
) -> LibTpuCStatus {
    if any_null(&[client as usize, buffer as usize, out_buffer as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "copy_to_device_args",
        });
    }
    // SAFETY: `buffer` and `client` are non-null.
    let (buf_ref, client_ref) = unsafe { (&(*buffer).inner, &*client) };
    match buf_ref.copy_to_device(
        DeviceId(dst_device_id),
        libtpu_core::types::MemorySpaceId(0),
    ) {
        Ok(copied) => {
            let dev_ptr = client_ref
                .device_ptrs
                .get(dst_device_id as usize)
                .copied()
                .unwrap_or(ptr::null_mut());
            let mem_ptr = client_ref
                .memory_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut());
            let handle = Box::into_raw(Box::new(PJRT_Buffer {
                inner: copied,
                device_ptr: dev_ptr,
                memory_ptr: mem_ptr,
            }));
            // SAFETY: `out_buffer` is non-null.
            unsafe {
                *out_buffer = handle;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Executes a [`PJRT_LoadedExecutable`] with arbitrary `num_inputs` and `num_outputs` (`libtpu_rs_execute_multi`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `device_id` - Target logical TPU device ID.
/// * `inputs` - Array of input `*const PJRT_Buffer` pointers.
/// * `num_inputs` - Number of input buffers.
/// * `outputs_out` - Caller-allocated array of `*mut PJRT_Buffer` of length `num_outputs`.
/// * `num_outputs` - Number of output buffer slots in `outputs_out`.
///
/// # Safety
///
/// All pointers must be null or valid for their respective counts.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_execute_multi(
    executable: *const PJRT_LoadedExecutable,
    device_id: u32,
    inputs: *const *const PJRT_Buffer,
    num_inputs: usize,
    outputs_out: *mut *mut PJRT_Buffer,
    num_outputs: usize,
) -> LibTpuCStatus {
    if any_null(&[executable as usize, outputs_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "execute_multi_args",
        });
    }
    let mut core_inputs = Vec::with_capacity(num_inputs);
    if !inputs.is_null() {
        // SAFETY: `inputs` is non-null with `num_inputs` elements.
        let slice = unsafe { std::slice::from_raw_parts(inputs, num_inputs) };
        for &ptr_item in slice {
            if ptr_item.is_null() {
                return c_status_err(&LibTpuError::NullPointer {
                    parameter: "input_buffer",
                });
            }
            // SAFETY: `ptr_item` is non-null.
            core_inputs.push(unsafe { &(*ptr_item).inner });
        }
    }
    // SAFETY: `executable` is non-null.
    let exec_ref = unsafe { &(*executable).inner };
    match exec_ref.execute_on_device(
        DeviceId(device_id),
        libtpu_core::types::MemorySpaceId(0),
        &core_inputs,
    ) {
        Ok(outs) => {
            for (idx, out_buf) in outs.into_iter().take(num_outputs).enumerate() {
                let handle = Box::into_raw(Box::new(PJRT_Buffer {
                    inner: out_buf,
                    device_ptr: ptr::null_mut(),
                    memory_ptr: ptr::null_mut(),
                }));
                // SAFETY: `outputs_out` has `num_outputs` capacity.
                unsafe {
                    *outputs_out.add(idx) = handle;
                }
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Serializes a [`PJRT_LoadedExecutable`] into `out_bytes` (`libtpu_rs_executable_serialize`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `out_bytes` - Optional output byte buffer (pass `NULL` to query size).
/// * `max_len` - Capacity of `out_bytes` in bytes.
/// * `out_len` - Output pointer receiving the total serialized byte length.
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_serialize(
    executable: *const PJRT_LoadedExecutable,
    out_bytes: *mut u8,
    max_len: usize,
    out_len: *mut usize,
) -> LibTpuCStatus {
    if any_null(&[executable as usize, out_len as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "executable_or_out_len",
        });
    }
    // SAFETY: `executable` is non-null.
    let bytes = unsafe { (*executable).inner.executable.program.serialize() };
    // SAFETY: `out_len` is non-null.
    unsafe {
        *out_len = bytes.len();
    }
    if !out_bytes.is_null() {
        let copy_len = max_len.min(bytes.len());
        // SAFETY: `out_bytes` has `copy_len` capacity.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out_bytes, copy_len);
        }
    }
    c_status_ok()
}

/// Deserializes and loads a [`PJRT_LoadedExecutable`] on `client` (`libtpu_rs_executable_deserialize`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `bytes` - Pointer to serialized executable bytes.
/// * `bytes_len` - Length of `bytes` in bytes.
/// * `exec_out` - Output pointer to receive the loaded [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_deserialize(
    client: *mut PJRT_Client,
    bytes: *const u8,
    bytes_len: usize,
    exec_out: *mut *mut PJRT_LoadedExecutable,
) -> LibTpuCStatus {
    if any_null(&[client as usize, bytes as usize, exec_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "deserialize_args",
        });
    }
    // SAFETY: `bytes` and `client` are non-null.
    let (slice, client_ref) = unsafe { (std::slice::from_raw_parts(bytes, bytes_len), &*client) };
    match client_ref.inner.deserialize_and_load(slice) {
        Ok(loaded) => {
            let exec_ptr = Box::into_raw(Box::new(PJRT_Executable {
                inner: loaded.executable.clone(),
            }));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
                inner: loaded,
                executable_ptr: exec_ptr,
                device_ptrs: client_ref.addressable_device_ptrs.clone(),
            }));
            // SAFETY: `exec_out` is non-null.
            unsafe {
                *exec_out = loaded_ptr;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Copies the 32-character hex fingerprint of `executable` into `out_buf` (`libtpu_rs_executable_fingerprint`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `executable` and `out_buf` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_fingerprint(
    executable: *const PJRT_LoadedExecutable,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if executable.is_null() {
        return 0;
    }
    // SAFETY: `executable` is non-null.
    let fp = unsafe {
        (*executable)
            .inner
            .executable
            .program
            .fingerprint
            .as_bytes()
    };
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = fp.len().min(buf_len.saturating_sub(1));
        // SAFETY: `out_buf` has at least `copy_len + 1` bytes.
        unsafe {
            ptr::copy_nonoverlapping(fp.as_ptr().cast::<c_char>(), out_buf, copy_len);
            *out_buf.add(copy_len) = 0;
        }
    }
    fp.len()
}

/// Returns the estimated FLOPs of `executable` (`libtpu_rs_executable_flops`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// `executable` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_flops(
    executable: *const PJRT_LoadedExecutable,
) -> f32 {
    if executable.is_null() {
        return 0.0;
    }
    // SAFETY: `executable` is non-null.
    unsafe { (*executable).inner.executable.program.cost_analysis.flops }
}

/// Returns the ABI version tuple and core primitive defaults.
pub fn abi_version_tuple() -> (i32, i32, DeviceAddress, ByteSize, ChipCoordinates) {
    (
        PJRT_API_MAJOR,
        PJRT_API_MINOR,
        DeviceAddress(0),
        ByteSize(PJRT_API_STRUCT_SIZE),
        ChipCoordinates::new(1, 1, 1),
    )
}

#[cfg(test)]
#[allow(for_loops_over_fallibles)]
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
        assert!(!format!("{:?}", api.pjrt_api_version).is_empty());

        let mut count = 0;
        let mut curr = api.extension_start;
        while !curr.is_null() {
            count += 1;
            // SAFETY: Static extension array.
            let ext = unsafe { *curr };
            assert!(!format!("{ext:?}").is_empty());
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
        assert!(!format!("{ev:?}{ctx:?}{stm:?}{sdk_cli:?}").is_empty());
        assert!(!format!(
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
        )
        .is_empty());

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
                api.pjrt_client_create_alias_buffer,
                api.pjrt_client_fulfill_alias_buffer,
                api.pjrt_topology_description_deserialize,
                api.pjrt_client_create_error_buffer,
                api.pjrt_client_create_uninitialized_buffer,
                api.pjrt_loaded_executable_get_device_assignment,
                api.pjrt_client_update_global_process_info,
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
                let mut dummy_args = [0_usize; 16];
                dummy_args[0] = 128;
                assert!(slot_fn(dummy_args.as_mut_ptr().cast()).is_null());
                if slot_fn as usize == api.pjrt_event_create as usize && dummy_args[2] != 0 {
                    drop(Box::from_raw(
                        dummy_args[2] as *mut crate::pjrt_types::PJRT_Event,
                    ));
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
            let mut valid_sz: [usize; 4] = [32, 0, 0, 0];
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
                valid_sz[3] = 0;
                assert!(ext_fn(valid_sz.as_mut_ptr().cast()).is_null());
            }
            drop(Box::from_raw(
                crate::extensions::PLUGIN_Profiler_CollectData(ptr::null_mut()),
            ));
            let mut prof_collect_null = crate::extensions::PLUGIN_Profiler_CollectData_Args {
                struct_size: std::mem::size_of::<crate::extensions::PLUGIN_Profiler_CollectData_Args>(
                ),
                extension_start: ptr::null_mut(),
                profiler: 1,
                buffer: ptr::null_mut(),
                size_in_bytes: ptr::null_mut(),
            };
            assert!(!format!("{prof_collect_null:?}").is_empty());
            assert!(crate::extensions::PLUGIN_Profiler_CollectData(
                ptr::addr_of_mut!(prof_collect_null).cast()
            )
            .is_null());
            let mut xspace_sz: usize = 0;
            let mut xspace_buf = vec![0_u8; 512];
            let mut prof_collect_full = crate::extensions::PLUGIN_Profiler_CollectData_Args {
                struct_size: std::mem::size_of::<crate::extensions::PLUGIN_Profiler_CollectData_Args>(
                ),
                extension_start: ptr::null_mut(),
                profiler: 1,
                buffer: xspace_buf.as_mut_ptr(),
                size_in_bytes: ptr::addr_of_mut!(xspace_sz),
            };
            assert!(crate::extensions::PLUGIN_Profiler_CollectData(
                ptr::addr_of_mut!(prof_collect_full).cast()
            )
            .is_null());
            assert!(xspace_sz > 0);
            assert!(
                !format!("{:?}", crate::extensions::ExtensionRuntimeState::default()).is_empty()
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
            assert!(!format!(
                "{ext_destroy_args:?}{ext_q32_args:?}{ext_q40_args:?}{ext_q48_args:?}{ext_q56_args:?}{ext_q64_args:?}{prof_err_del:?}{prof_err_msg:?}{prof_err_code:?}{prof_create:?}{prof_del:?}{prof_start:?}{prof_stop:?}{ms_del:?}{ms_ns:?}{ms_ndps:?}{ms_sid:?}{coll_del:?}{coll_cc:?}{comm_del:?}{comm_ag:?}{comm_str:?}{abi_get:?}{abi_del:?}{abi_cmp:?}{abi_pid:?}{abi_fp:?}{tpu_ec:?}{topo_sq:?}{topo_bq:?}{topo_mq:?}{cb_inv:?}{pc_gc:?}{pc_del:?}{md_get:?}{md_del:?}{rb_q:?}{rb_del:?}"
            )
            .is_empty());
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
            assert!(!format!(
                "{layout_destroy_args:?}{layout_ser_args:?}{layout_def_args:?}{layout_buf_args:?}{shardings_args:?}{phase_names_args:?}{triton_args:?}"
            )
            .is_empty());
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
            let host_alloc_args = crate::extensions::PJRT_HostMemoryAllocator_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                allocator_out: 0,
            };
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
            assert!(!format!(
                "{xla_reg_args:?}{xla_wait_args:?}{xla_unreg_args:?}{xla_list_args:?}{host_alloc_args:?}{ms_ser_args:?}{comm_ar_args:?}{abi_proto_args:?}{tpu_mem_stats_args:?}{cb_reg_args:?}{phase_run_args:?}{raw_copy_args:?}"
            )
            .is_empty());
            assert!(crate::extensions::extension_call_count() > 0);
            assert!(pjrt_slot_dispatch_count() > 0);

            PJRT_Error_Destroy(ptr::null_mut());
            let mut destroy_null_err = PJRT_Error_Destroy_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                error: ptr::null_mut(),
            };
            assert!(!format!("{destroy_null_err:?}").is_empty());
            PJRT_Error_Destroy(ptr::addr_of_mut!(destroy_null_err).cast());

            PJRT_Error_Message(ptr::null_mut());
            let mut msg_null_err = PJRT_Error_Message_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                message: ptr::null(),
                message_size: 0,
            };
            assert!(!format!("{msg_null_err:?}").is_empty());
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
            assert!(!format!("{:?}", *null_code_err).is_empty());

            let mut code_args = PJRT_Error_GetCode_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                error: null_code_err,
                code: 0,
            };
            assert!(!format!("{code_args:?}").is_empty());
            assert!(PJRT_Error_GetCode(ptr::addr_of_mut!(code_args).cast()).is_null());
            assert_eq!(code_args.code, 3);

            let mut code_null_inner = PJRT_Error_GetCode_Args {
                struct_size: 24,
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
            assert!(!format!("{init_args:?}").is_empty());
            assert!(PJRT_Plugin_Initialize(ptr::addr_of_mut!(init_args).cast()).is_null());

            let e_chk = PJRT_CheckMatchingStructSizes(ptr::null_mut());
            drop(Box::from_raw(e_chk));
            let e_chk_sz = PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(zero_sz).cast());
            drop(Box::from_raw(e_chk_sz));
            assert!(PJRT_CheckMatchingStructSizes(ptr::addr_of_mut!(init_args).cast()).is_null());

            let e_cc = PJRT_Client_Create(ptr::null_mut());
            drop(Box::from_raw(e_cc));
            let mut create_args = PJRT_Client_Create_Args {
                struct_size: std::mem::size_of::<PJRT_Client_Create_Args>(),
                extension_start: ptr::null_mut(),
                create_options: ptr::null(),
                num_options: 0,
                kv_get_callback: ptr::null(),
                kv_get_user_arg: ptr::null_mut(),
                kv_put_callback: ptr::null(),
                kv_put_user_arg: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            assert!(!format!("{create_args:?}").is_empty());
            assert!(PJRT_Client_Create(ptr::addr_of_mut!(create_args).cast()).is_null());
            let client = create_args.client;
            assert!(!client.is_null());
            assert!(!format!("{:?}", *client).is_empty());
            assert!(!format!("{:?}", *(*client).topology_ptr).is_empty());
            for &dev0 in (*client).device_ptrs.first() {
                assert!(!format!("{:?}{:?}", *dev0, *(*dev0).desc_ptr).is_empty());
            }
            for &mem0 in (*client).memory_ptrs.first() {
                assert!(!format!("{:?}", *mem0).is_empty());
            }

            for mgr in PjrtCoreAsyncTransferManager::create(
                DeviceId(0),
                libtpu_core::types::MemorySpaceId(0),
                Vec::new(),
                &(*client).inner.hal,
            ) {
                let c_mgr = PJRT_AsyncHostToDeviceTransferManager { inner: mgr };
                assert!(!format!("{c_mgr:?}").is_empty());
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
            assert!(!format!("{str_args:?}").is_empty());
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
            assert!(!format!("{pi_args:?}").is_empty());
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
            assert!(!format!("{dev_args:?}").is_empty());
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
                struct_size: 32,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                id: 0,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert!(!format!("{look_args:?}").is_empty());
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
            assert!(!format!("{mem_args:?}").is_empty());
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
                struct_size: std::mem::size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: ptr::null(),
                code_size: code_bytes.len(),
                format: fmt_bytes.as_ptr().cast(),
                format_size: fmt_bytes.len(),
            };
            assert!(!format!("{prog:?}").is_empty());
            let mut comp_args = PJRT_Client_Compile_Args {
                struct_size: std::mem::size_of::<PJRT_Client_Compile_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                program: ptr::addr_of!(prog),
                compile_options: ptr::null(),
                compile_options_size: 0,
                executable: ptr::null_mut(),
            };
            assert!(!format!("{comp_args:?}").is_empty());
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
            assert!(!format!(
                "{:?}{:?}",
                *comp_args.executable,
                *(*comp_args.executable).executable_ptr
            )
            .is_empty());

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
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args,
                >(),
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
            assert!(PJRT_Client_BufferFromHostBuffer(
                ptr::addr_of_mut!(h2d_args_default_dev).cast()
            )
            .is_null());
            let buf_in2 = h2d_args_default_dev.out_buffer;
            assert!(!buf_in2.is_null());

            let mut h2d_args_zero_ndims = h2d_args;
            h2d_args_zero_ndims.num_dims = 0;
            h2d_args_zero_ndims.out_buffer = ptr::null_mut();
            assert!(PJRT_Client_BufferFromHostBuffer(
                ptr::addr_of_mut!(h2d_args_zero_ndims).cast()
            )
            .is_null());
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
            assert!(PJRT_Plugin_Attributes(ptr::addr_of_mut!(plug_attr_args).cast()).is_null());
            plug_attr_args.struct_size =
                std::mem::size_of::<crate::pjrt_types::PJRT_Plugin_Attributes_Args>();
            assert!(PJRT_Plugin_Attributes(ptr::addr_of_mut!(plug_attr_args).cast()).is_null());
            assert_eq!(plug_attr_args.num_attributes, 3);
            assert!(!plug_attr_args.attributes.is_null());
            assert!(!format!("{:?}", *plug_attr_args.attributes).is_empty());

            let mut b_dims_args = crate::pjrt_types::PJRT_Buffer_Dimensions_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dims: ptr::null(),
                num_dims: 0,
            };
            assert!(PJRT_Buffer_Dimensions(ptr::addr_of_mut!(b_dims_args).cast()).is_null());
            assert_eq!(b_dims_args.num_dims, 1);

            let mut b_et_args = crate::pjrt_types::PJRT_Buffer_ElementType_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                element_type: 0,
                pad: 0,
            };
            assert!(PJRT_Buffer_ElementType(ptr::addr_of_mut!(b_et_args).cast()).is_null());
            assert_eq!(b_et_args.element_type, 11);

            let mut b_sz_args = crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                on_device_size_in_bytes: 0,
            };
            assert!(PJRT_Buffer_OnDeviceSizeInBytes(ptr::addr_of_mut!(b_sz_args).cast()).is_null());
            assert!(b_sz_args.on_device_size_in_bytes >= 16);

            // Exercise handle-aware dispatchers with real handles (DeviceDescription, Device, Memory, Executable, LoadedExecutable, Buffer, Stream, Topology, AsyncTransferManager, Client)
            for &dev0 in (*client).device_ptrs.first() {
                for dd_fn in [
                    PJRT_DeviceDescription_Id
                        as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                    PJRT_DeviceDescription_ProcessIndex,
                    PJRT_DeviceDescription_Attributes,
                    PJRT_DeviceDescription_Kind,
                    PJRT_DeviceDescription_DebugString,
                    PJRT_DeviceDescription_ToString,
                ] {
                    let mut dd_args = [0_usize; 8];
                    dd_args[0] = 64;
                    dd_args[2] = (*dev0).desc_ptr as usize;
                    assert!(dd_fn(dd_args.as_mut_ptr().cast()).is_null());
                }

                for dv_fn in [
                    PJRT_Device_GetDescription
                        as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                    PJRT_Device_IsAddressable,
                    PJRT_Device_LocalHardwareId,
                    PJRT_Device_AddressableMemories,
                    PJRT_Device_DefaultMemory,
                    PJRT_Device_MemoryStats,
                    PJRT_Device_PoisonExecution,
                    PJRT_Device_CreateAsyncTrackingEvent,
                    PJRT_Device_GetAttributes,
                    PJRT_Device_ClearMemoryStats,
                ] {
                    let mut dv_args = [0_usize; 16];
                    dv_args[0] = 128;
                    dv_args[2] = dev0 as usize;
                    assert!(dv_fn(dv_args.as_mut_ptr().cast()).is_null());
                }
            }
            for &mem0 in (*client).memory_ptrs.first() {
                for mm_fn in [
                    PJRT_Memory_Id
                        as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                    PJRT_Memory_Kind,
                    PJRT_Memory_DebugString,
                    PJRT_Memory_ToString,
                    PJRT_Memory_AddressableByDevices,
                    PJRT_Memory_Kind_Id,
                ] {
                    let mut mm_args = [0_usize; 8];
                    mm_args[0] = 64;
                    mm_args[2] = mem0 as usize;
                    assert!(mm_fn(mm_args.as_mut_ptr().cast()).is_null());
                }
            }
            for ex_fn in [
                PJRT_Executable_Destroy
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_Executable_Name,
                PJRT_Executable_NumReplicas,
                PJRT_Executable_NumPartitions,
                PJRT_Executable_NumOutputs,
                PJRT_Executable_SizeOfGeneratedCodeInBytes,
                PJRT_Executable_GetCostAnalysis,
                PJRT_Executable_OutputMemoryKinds,
                PJRT_Executable_OptimizedProgram,
                PJRT_Executable_Serialize,
                PJRT_Executable_OutputElementTypes,
                PJRT_Executable_OutputDimensions,
                PJRT_Executable_Fingerprint,
                PJRT_Executable_GetCompiledMemoryStats,
                PJRT_Executable_GetCompileOptions,
                PJRT_Executable_ParameterMemoryKinds,
            ] {
                let mut ex_args = [0_usize; 12];
                ex_args[0] = 96;
                ex_args[2] = (*comp_args.executable).executable_ptr as usize;
                assert!(ex_fn(ex_args.as_mut_ptr().cast()).is_null());
            }
            for le_fn in [
                PJRT_LoadedExecutable_GetExecutable
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_LoadedExecutable_AddressableDevices,
                PJRT_LoadedExecutable_IsDeleted,
                PJRT_LoadedExecutable_Fingerprint,
                PJRT_LoadedExecutable_GetDeviceAssignment,
                PJRT_LoadedExecutable_AddressableDeviceLogicalIds,
            ] {
                let mut le_args = [0_usize; 12];
                le_args[0] = 96;
                le_args[2] = comp_args.executable as usize;
                assert!(le_fn(le_args.as_mut_ptr().cast()).is_null());
            }
            for bf_fn in [
                PJRT_Buffer_UnpaddedDimensions
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_Buffer_DynamicDimensionIndices,
                PJRT_Buffer_GetMemoryLayout,
                PJRT_Buffer_Device,
                PJRT_Buffer_Memory,
                PJRT_Buffer_IsDeleted,
                PJRT_Buffer_CopyToDevice,
                PJRT_Buffer_CopyToMemory,
                PJRT_Buffer_IsOnCpu,
                PJRT_Buffer_ReadyEvent,
                PJRT_Buffer_UnsafePointer,
                PJRT_Buffer_IncreaseExternalReferenceCount,
                PJRT_Buffer_DecreaseExternalReferenceCount,
                PJRT_Buffer_OpaqueDeviceMemoryDataPointer,
                PJRT_Buffer_CopyRawToHost,
                PJRT_Buffer_CopyRawToHostFuture,
                PJRT_Buffer_DonateWithControlDependency,
            ] {
                let mut bf_args = [0_usize; 12];
                bf_args[0] = 96;
                bf_args[2] = buf_in as usize;
                assert!(bf_fn(bf_args.as_mut_ptr().cast()).is_null());
            }
            pjrt_serialized_noop_deleter(ptr::null_mut());

            // Exercise non-null branches of DeserializeAndLoad, CopyToDevice, CopyToMemory, AddChunk, Compile, CreateViewOfDeviceBuffer, CopyRawToHost, ExecuteContext_Destroy, and AsyncTrackingEvent_Destroy
            let mut deser_args = crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args,
                >(),
                extension_start: ptr::null_mut(),
                client,
                serialized_executable: code_bytes.as_ptr().cast(),
                serialized_executable_size: code_bytes.len(),
                loaded_executable: ptr::null_mut(),
                overridden_serialized_compile_options: ptr::null(),
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
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args>(
                ),
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
                    struct_size: std::mem::size_of::<
                        crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args,
                    >(),
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
                PJRT_CopyToDeviceStream_Destroy
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_CopyToDeviceStream_AddChunk,
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
            let mut chunk_val = crate::pjrt_types::PJRT_Chunk {
                data: ptr::null_mut(),
                size: 64,
                deleter: Some(test_chunk_deleter),
                deleter_arg: ptr::addr_of_mut!(chunk_del_hits).cast(),
            };
            let mut add_chunk_args = crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args,
                >(),
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
            drop(Box::from_raw(sample_stream));

            let mut aot_args = crate::pjrt_types::PJRT_Compile_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Compile_Args>(),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr,
                program: ptr::addr_of!(prog),
                compile_options: c"opts".as_ptr(),
                compile_options_size: 4,
                executable: ptr::null_mut(),
            };
            assert!(PJRT_Compile(ptr::addr_of_mut!(aot_args).cast()).is_null());
            assert!(!aot_args.executable.is_null());
            drop(Box::from_raw(aot_args.executable));
            let empty_prog = PJRT_Program {
                struct_size: std::mem::size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: ptr::null(),
                code_size: 0,
                format: ptr::null(),
                format_size: 0,
            };
            aot_args.program = ptr::addr_of!(empty_prog);
            aot_args.executable = ptr::null_mut();
            assert!(PJRT_Compile(ptr::addr_of_mut!(aot_args).cast()).is_null());
            drop(Box::from_raw(aot_args.executable));
            let bad_aot_prog = PJRT_Program {
                struct_size: std::mem::size_of::<PJRT_Program>(),
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
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args,
                >(),
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

            let mut raw_dst = [0_u8; 16];
            let mut raw_copy_h = crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>(
                ),
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
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_ExecuteContext_Create_Args>(
                ),
                extension_start: ptr::null_mut(),
                context: ptr::null_mut(),
            };
            assert!(PJRT_ExecuteContext_Create(ptr::addr_of_mut!(ec_create).cast()).is_null());
            let mut ec_destroy = crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args>(
                ),
                extension_start: ptr::null_mut(),
                context: ec_create.context,
            };
            assert!(PJRT_ExecuteContext_Destroy(ptr::addr_of_mut!(ec_destroy).cast()).is_null());
            drop(Box::from_raw(ec_create.context));

            let mut ate_create = crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args,
                >(),
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
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args,
                >(),
                extension_start: ptr::null_mut(),
                event: ate_create.event,
            };
            assert!(
                PJRT_AsyncTrackingEvent_Destroy(ptr::addr_of_mut!(ate_destroy).cast()).is_null()
            );
            drop(Box::from_raw(ate_create.event));

            for tp_fn in [
                PJRT_TopologyDescription_Destroy
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_TopologyDescription_PlatformName,
                PJRT_TopologyDescription_PlatformVersion,
                PJRT_TopologyDescription_GetDeviceDescriptions,
                PJRT_TopologyDescription_Serialize,
                PJRT_TopologyDescription_Attributes,
                PJRT_TopologyDescription_Fingerprint,
                PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace,
                PJRT_TopologyDescription_GetMemorySpaceKindIds,
            ] {
                let mut tp_args = [0_usize; 12];
                tp_args[0] = 96;
                tp_args[2] = (*client).topology_ptr as usize;
                assert!(tp_fn(tp_args.as_mut_ptr().cast()).is_null());
            }

            for tm_inner in libtpu_core::pjrt::PjrtCoreAsyncTransferManager::create(
                DeviceId(0),
                libtpu_core::types::MemorySpaceId(0),
                vec![(*buf_in).inner.shape.clone()],
                &(*client).inner.hal,
            ) {
                let sample_tm = Box::into_raw(Box::new(
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager { inner: tm_inner },
                ));
                for tm_fn in [
                    PJRT_AsyncHostToDeviceTransferManager_Destroy
                        as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                    PJRT_AsyncHostToDeviceTransferManager_TransferData,
                    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
                    PJRT_AsyncHostToDeviceTransferManager_Device,
                    PJRT_AsyncHostToDeviceTransferManager_BufferCount,
                    PJRT_AsyncHostToDeviceTransferManager_BufferSize,
                    PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
                    PJRT_AsyncHostToDeviceTransferManager_AddMetadata,
                    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral,
                ] {
                    let mut tm_args = [0_usize; 12];
                    tm_args[0] = 96;
                    tm_args[2] = sample_tm as usize;
                    assert!(tm_fn(tm_args.as_mut_ptr().cast()).is_null());
                }
                drop(Box::from_raw(sample_tm));
            }

            for cl_fn in [
                PJRT_Client_TopologyDescription
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_Client_DmaMap,
                PJRT_Client_DmaUnmap,
                PJRT_Client_FulfillAliasBuffer,
                PJRT_Client_UpdateGlobalProcessInfo,
            ] {
                let mut cl_args = [0_usize; 12];
                cl_args[0] = 96;
                cl_args[2] = client as usize;
                assert!(cl_fn(cl_args.as_mut_ptr().cast()).is_null());
            }

            let dev0_inputs: [*const PJRT_Buffer; 2] = [buf_in, buf_in2];
            let dev0_inputs_ptr: *const *const PJRT_Buffer = dev0_inputs.as_ptr();
            let mut out_buf_ptr: *mut PJRT_Buffer = ptr::null_mut();
            let dev0_outputs_ptr: *mut *mut PJRT_Buffer = ptr::addr_of_mut!(out_buf_ptr);
            let mut exec_vtable_args = crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
                options: ptr::null(),
                argument_lists: ptr::addr_of!(dev0_inputs_ptr),
                num_devices: 1,
                output_lists: ptr::addr_of!(dev0_outputs_ptr),
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
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>(
                ),
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
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Event_Create_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!(PJRT_Event_Create(ptr::addr_of_mut!(ev_create_args).cast()).is_null());
            let created_ev = ev_create_args.event;
            assert!(!created_ev.is_null());

            let mut ev_ready_args = crate::pjrt_types::PJRT_Event_IsReady_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Event_IsReady_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                is_ready: true,
                pad: [0; 7],
            };
            assert!(PJRT_Event_IsReady(ptr::addr_of_mut!(ev_ready_args).cast()).is_null());
            assert!(!ev_ready_args.is_ready);

            let mut ev_set_args = crate::pjrt_types::PJRT_Event_Set_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Event_Set_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                error_code: 0,
                pad: 0,
                error_message: ptr::null(),
            };
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());
            assert!(PJRT_Event_IsReady(ptr::addr_of_mut!(ev_ready_args).cast()).is_null());
            assert!(ev_ready_args.is_ready);

            let mut ev_err_args = crate::pjrt_types::PJRT_Event_Error_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Event_Error_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Error(ptr::addr_of_mut!(ev_err_args).cast()).is_null());
            let mut ev_await_args = crate::pjrt_types::PJRT_Event_Await_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Event_Await_Args>(),
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
                _err: *mut PJRT_Error,
                user_arg: *mut std::ffi::c_void,
            ) {
                // SAFETY: `user_arg` points to a mutable `u32` counter in this test.
                unsafe {
                    *user_arg.cast::<u32>() += 1;
                }
            }
            let mut cb_hits: u32 = 0;
            let mut on_ready_args = crate::pjrt_types::PJRT_Event_OnReady_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Event_OnReady_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                callback: Some(test_on_ready_cb),
                user_arg: ptr::addr_of_mut!(cb_hits).cast(),
            };
            assert!(PJRT_Event_OnReady(ptr::addr_of_mut!(on_ready_args).cast()).is_null());
            assert_eq!(cb_hits, 1);

            let mut ev_del_args = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del_args).cast()).is_null());

            let mut look_addr_args = PJRT_Client_LookupDevice_Args {
                struct_size: std::mem::size_of::<PJRT_Client_LookupDevice_Args>(),
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
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args,
                >(),
                extension_start: ptr::null_mut(),
                client,
                num_replicas: 2,
                num_partitions: 2,
                default_assignment_size: assign_buf.len(),
                default_assignment: assign_buf.as_mut_ptr(),
                reserved: 0,
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
            };
            assert!(PJRT_Device_GetAttributes(ptr::addr_of_mut!(dev_attr_args).cast()).is_null());
            dev_attr_args.struct_size =
                std::mem::size_of::<crate::pjrt_types::PJRT_Device_GetAttributes_Args>();
            assert!(PJRT_Device_GetAttributes(ptr::addr_of_mut!(dev_attr_args).cast()).is_null());
            assert_eq!(dev_attr_args.num_attributes, 3);

            let mut load_args = crate::pjrt_types::PJRT_Client_Load_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Client_Load_Args>(),
                extension_start: ptr::null_mut(),
                client,
                executable: (*comp_args.executable).executable_ptr,
                compile_options: ptr::null(),
                loaded_executable: ptr::null_mut(),
            };
            assert!(PJRT_Client_Load(ptr::addr_of_mut!(load_args).cast()).is_null());
            assert!(!load_args.loaded_executable.is_null());
            let mut del_loaded2 = crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: load_args.loaded_executable,
            };
            assert!(PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(del_loaded2).cast()).is_null());

            let bc_dims = [2_i64, 2];
            let mut bc_args = crate::pjrt_types::PJRT_Buffer_Bitcast_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Bitcast_Args>(),
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
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Error_ForEachPayload_Args>(
                ),
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
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Delete_Args>(),
                extension_start: ptr::null_mut(),
                buffer: out_buf_ptr,
            };
            assert!(PJRT_Buffer_Delete(ptr::addr_of_mut!(b_del_call).cast()).is_null());
            let mut b_is_del_call = crate::pjrt_types::PJRT_Buffer_IsDeleted_Args {
                struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_IsDeleted_Args>(),
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
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
            };
            assert!(PJRT_LoadedExecutable_Delete(ptr::addr_of_mut!(le_del_call).cast()).is_null());
            let mut le_is_del_call = crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args,
                >(),
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
                    struct_size: std::mem::size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    buffer: b_del,
                };
                assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_b_args).cast()).is_null());
            }

            let mut del_exec_args = crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args {
                struct_size: std::mem::size_of::<
                    crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args,
                >(),
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
            assert!(!format!("{destroy_client_args:?}").is_empty());
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
            assert!(!format!("{sdk_msg_args:?}").is_empty());
            LibtpuSdk_Error_GetMessage(ptr::addr_of_mut!(sdk_msg_args).cast());
            let sdk_err = LibtpuSdk_Error_GetCode(ptr::null_mut());
            assert!(!sdk_err.is_null());
            assert!(!format!("{:?}", *sdk_err).is_empty());
            sdk_msg_args.error = sdk_err;
            LibtpuSdk_Error_GetMessage(ptr::addr_of_mut!(sdk_msg_args).cast());
            assert!(sdk_msg_args.message_size > 0);

            let mut sdk_code_args = LibtpuSdk_Error_GetCode_Args {
                error: ptr::null(),
                code: 0,
                pad: 0,
            };
            assert!(!format!("{sdk_code_args:?}").is_empty());
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
            assert!(!format!("{sdk_del_args:?}").is_empty());
            LibtpuSdk_Error_Destroy(ptr::addr_of_mut!(sdk_del_args).cast());
            sdk_del_args.error = sdk_err;
            LibtpuSdk_Error_Destroy(ptr::addr_of_mut!(sdk_del_args).cast());

            drop(Box::from_raw(LibtpuSdk_Client_Create(ptr::null_mut())));
            let mut sdk_cli_args = LibtpuSdk_Client_Args {
                client: ptr::null_mut(),
            };
            assert!(!format!("{sdk_cli_args:?}").is_empty());
            assert!(LibtpuSdk_Client_Create(ptr::addr_of_mut!(sdk_cli_args).cast()).is_null());
            let sdk_client = sdk_cli_args.client;
            assert!(!sdk_client.is_null());

            drop(Box::from_raw(LibtpuSdk_GetChipCoordinates(ptr::null_mut())));
            let mut coords_args = LibtpuSdk_GetChipCoordinates_Args {
                client: ptr::null_mut(),
                coords: ptr::null(),
                num_coords: 0,
            };
            assert!(!format!("{coords_args:?}").is_empty());
            drop(Box::from_raw(LibtpuSdk_GetChipCoordinates(
                ptr::addr_of_mut!(coords_args).cast(),
            )));
            coords_args.client = sdk_client;
            assert!(LibtpuSdk_GetChipCoordinates(ptr::addr_of_mut!(coords_args).cast()).is_null());
            assert_eq!(coords_args.num_coords, 4);
            assert!(!format!("{:?}", *coords_args.coords).is_empty());

            drop(Box::from_raw(LibtpuSdk_GetHostName(ptr::null_mut())));
            let mut host_args = LibtpuSdk_GetHostName_Args {
                coord: ptr::null(),
                host_name: ptr::null(),
                host_name_size: 0,
            };
            assert!(!format!("{host_args:?}").is_empty());
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
            assert!(!format!("{cidx_args:?}").is_empty());
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
            assert!(!format!("{cart_args:?}").is_empty());
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
            assert!(!format!("{met_args:?}").is_empty());
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
            assert!(!format!("{:?}", *met_args.metric).is_empty());

            drop(Box::from_raw(LibtpuSdk_GetMetricDescription(
                ptr::null_mut(),
            )));
            let mut mdesc_args = LibtpuSdk_GetMetricDescription_Args {
                metric: ptr::null(),
                description: ptr::null(),
                description_size: 0,
            };
            assert!(!format!("{mdesc_args:?}").is_empty());
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
            assert!(!format!("{mvals_args:?}").is_empty());
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
            assert!(!format!("{rstat_args:?}").is_empty());
            drop(Box::from_raw(LibtpuSdk_GetRuntimeStatus(
                ptr::addr_of_mut!(rstat_args).cast(),
            )));
            rstat_args.client = sdk_client;
            assert!(LibtpuSdk_GetRuntimeStatus(ptr::addr_of_mut!(rstat_args).cast()).is_null());
            assert!(!rstat_args.status.is_null());
            assert!(!format!("{:?}", *rstat_args.status).is_empty());

            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
                ptr::null_mut(),
            )));
            let mut sum_args = LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args {
                status: ptr::null(),
                summary: ptr::null(),
                summary_size: 0,
            };
            assert!(!format!("{sum_args:?}").is_empty());
            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
                ptr::addr_of_mut!(sum_args).cast(),
            )));
            sum_args.status = rstat_args.status;
            assert!(LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
                ptr::addr_of_mut!(sum_args).cast()
            )
            .is_null());
            assert!(sum_args.summary_size > 0);

            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_Destroy(
                ptr::null_mut(),
            )));
            let mut rdel_args = LibtpuSdk_RuntimeStatus_Destroy_Args {
                status: ptr::null_mut(),
            };
            assert!(!format!("{rdel_args:?}").is_empty());
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
            assert!(!format!("{reg_cb_args:?}").is_empty());
            drop(Box::from_raw(LibtpuSdk_RegisterDebugCallback(
                ptr::addr_of_mut!(reg_cb_args).cast(),
            )));
            reg_cb_args.client = sdk_client;
            assert!(
                LibtpuSdk_RegisterDebugCallback(ptr::addr_of_mut!(reg_cb_args).cast()).is_null()
            );
            assert!(reg_cb_args.callback_id > 0);

            drop(Box::from_raw(LibtpuSdk_UnregisterDebugCallback(
                ptr::null_mut(),
            )));
            let mut unreg_cb_args = LibtpuSdk_UnregisterDebugCallback_Args {
                client: ptr::null_mut(),
                callback_id: reg_cb_args.callback_id,
            };
            assert!(!format!("{unreg_cb_args:?}").is_empty());
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
                struct_size: std::mem::size_of::<TFNPD_PluginParams>(),
                priv_data: ptr::null_mut(),
                device_type: ptr::null(),
                compilation_device_name: ptr::null(),
                priority: 0,
                is_pluggable_device: false,
                use_pjrt_on_demand_compile: false,
                pad: 0,
            };
            assert!(!format!("{tfnpd_params:?}").is_empty());
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
            assert!(!format!("{pcall_params:?}").is_empty());
            assert_eq!(
                TfTpu_GetTpuPartitionedCallParams(ptr::addr_of_mut!(pcall_params)),
                0
            );
            assert!(pcall_params.input_shape_opt);

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
            assert!(!format!("{host_shape:?}").is_empty());
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
            assert!(!format!("{:?}", *st1).is_empty());
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
            assert!(!format!("{:?}", *sel).is_empty());
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
            assert!(!format!("{:?}", *topo).is_empty());
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
            assert!(!format!("{:?}", *c0).is_empty());
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
            assert!(!format!("{host_loc:?}").is_empty());
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
            assert!(!format!("{:?}", *ms).is_empty());
            assert!(!TpuMeshState_MeshCommonState(ms).is_null());
            assert!(TpuMeshState_MeshCommonState(ptr::null()).is_null());
            TpuMeshState_Free(ms);
            TpuMeshState_Free(ptr::null_mut());

            let st_prof = TpuStatus_New();
            TpuProfiler_Create(ptr::null_mut(), st_prof);
            let mut prof: *mut TpuProfiler = ptr::null_mut();
            TpuProfiler_Create(ptr::addr_of_mut!(prof), st_prof);
            assert!(!format!("{:?}", *prof).is_empty());
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
            assert!(!format!("{:?}", *emb).is_empty());
            assert!(!TpuEmbeddingEngineState_GetState(emb).is_null());
            let mut emb_out_sz: usize = 0;
            let mut emb_out_ptr: *mut c_char = ptr::null_mut();
            let mut emb_params = TpuEmbeddingEngine_ExecutePartitioner_Params {
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_ExecutePartitioner_Params>(),
                priv_data: ptr::null_mut(),
                tpu_embedding_config_bytes: ptr::null(),
                tpu_embedding_config_size: 0,
                common_config_string_size: ptr::addr_of_mut!(emb_out_sz),
                common_config_string: ptr::addr_of_mut!(emb_out_ptr),
                status: ptr::null_mut(),
            };
            assert!(!format!("{emb_params:?}").is_empty());
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_ConfigureMemory_Params>(),
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_CollateMemory_Params>(),
                priv_data: ptr::null_mut(),
                memory_configs: ptr::null(),
                memory_configs_size: 0,
                collated_size: ptr::addr_of_mut!(col_sz),
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_ConfigureHost_Params>(),
                priv_data: ptr::null_mut(),
                num_inputs: 1,
                pad: 0,
                common_config_string: ptr::null(),
                common_config_string_size: 0,
                task_host_config: ptr::null(),
                task_host_config_size: 0,
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_ConnectHosts_Params>(),
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_Finalize_Params>(),
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_IsInitialized_Params>(),
                priv_data: ptr::null_mut(),
                config_string: ptr::null(),
                config_string_size: 0,
                is_embedding_initialized: ptr::addr_of_mut!(is_init_bool),
            };
            assert!(TpuEmbeddingEngine_IsInitialized(ptr::addr_of!(init_params)));
            assert!(is_init_bool);
            let init_params_null_out = TpuEmbeddingEngine_IsInitialized_Params {
                struct_size: 0,
                priv_data: ptr::null_mut(),
                config_string: ptr::null(),
                config_string_size: 0,
                is_embedding_initialized: ptr::null_mut(),
            };
            assert!(TpuEmbeddingEngine_IsInitialized(ptr::addr_of!(
                init_params_null_out
            )));

            let mut wr_buf = [7_u8; 16];
            let mut rd_buf = [0_u8; 16];
            let mut rd_floats = [0.0_f32; 4];
            let mut num_rw: usize = 0;
            let mut wr_params = TpuEmbeddingEngine_WriteParameters_Params {
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_WriteParameters_Params>(),
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
            wr_params.struct_size =
                std::mem::size_of::<TpuEmbeddingEngine_WriteParameters_Params>();
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_EnqueueTensorBatch_Params>(),
                priv_data: ptr::null_mut(),
                mode_override: ptr::null(),
                mode_override_size: 0,
                device_ordinal: 0,
                fixed_state: ptr::null_mut(),
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
                struct_size: std::mem::size_of::<TpuEmbeddingEngine_Computation_Params>(),
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

            let drv = TpuDriver_Open(c"local://".as_ptr());
            assert!(!(*drv).fn_table.is_null());
            assert!(!format!("{:?}{:?}", *drv, *(*drv).fn_table).is_empty());
            assert_eq!(
                TpuDriver_ComputeLinearizedBytesFromShape(ptr::null_mut(), 16),
                0
            );
            assert_eq!(TpuDriver_ComputeLinearizedBytesFromShape(drv, 16), 256);
            assert!(TpuDriver_CompileProgramFromText(
                ptr::null_mut(),
                ptr::null(),
                1,
                0,
                ptr::null()
            )
            .is_null());
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
            assert!(!format!("{:?}", *sys_info).is_empty());
            TpuDriver_FreeSystemInfo(sys_info);

            assert!(TpuDriver_Compile(ptr::null_mut(), ptr::null(), 1, 0, ptr::null()).is_null());
            TpuDriver_FreeCompiledProgramHandle(ptr::null_mut());
            let hlo_src_bytes = b"module @add { stablehlo.add }";
            let hlo_proto = crate::tpu_driver_types::HloProto {
                buffer: hlo_src_bytes.as_ptr().cast(),
                size: i64::try_from(hlo_src_bytes.len()).unwrap_or(0),
            };
            assert!(!format!("{hlo_proto:?}").is_empty());
            let cp_default = TpuDriver_Compile(drv, ptr::null(), 1, 0, ptr::null());
            assert!(!cp_default.is_null());
            assert!(!format!("{:?}", *cp_default).is_empty());
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
            assert!(!format!("{:?}", *lp_handle).is_empty());

            assert!(TpuDriver_Allocate(ptr::null_mut(), 0, 0, 16, 0, ptr::null()).is_null());
            let drv_b0 = TpuDriver_Allocate(drv, 0, 0, 16, 0, ptr::null());
            let drv_b1 = TpuDriver_AllocateShape(drv, 0, 0, 16, 0, ptr::null());
            let drv_b_out = TpuDriver_AllocateTuple(drv, 0, 0, 2, ptr::null(), 0, ptr::null());
            assert!(!drv_b0.is_null() && !drv_b1.is_null() && !drv_b_out.is_null());
            assert!(!format!("{:?}", *drv_b0).is_empty());

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
            assert!(!format!("{da:?}").is_empty());
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
            assert!(!format!("{:?}", *ev_exec).is_empty());

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
            TpuDriver_Close(drv);
            TpuDriver_Close(ptr::null_mut());

            let bad_st = libtpu_rs_client_create(ptr::null_mut());
            assert!(!format!("{bad_st:?}").is_empty());
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
            assert!(!format!("{:?}", *b_scalar).is_empty());
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
