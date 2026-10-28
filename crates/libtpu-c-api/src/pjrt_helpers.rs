//! Shared PJRT and `LibtpuSdk` FFI dispatchers and error/buffer helpers.

use crate::pjrt_types::{
    PJRT_Buffer, PJRT_Client, PJRT_Device, PJRT_DeviceDescription, PJRT_Error, PJRT_Memory,
    PJRT_TopologyDescription,
};
use crate::tpu_driver_types::LibtpuSdk_Error;
use libtpu_core::error::LibTpuError;
use libtpu_core::types::{DeviceId, ElementType, Shape};
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

/// Returns `true` if any pointer address in `addrs` is zero (`NULL`).
///
/// # Arguments
///
/// * `addrs` - Slice of pointer addresses cast to `usize`.
pub(crate) fn any_null(addrs: &[usize]) -> bool {
    addrs.contains(&0)
}

/// Helper to convert a Rust string into a sanitized [`CString`] without interior NUL errors.
///
/// # Arguments
///
/// * `text` - Input UTF-8 string.
pub(crate) fn to_c_string(text: &str) -> CString {
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
        error: error.clone(),
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

/// Validates `struct_size` of a PJRT argument struct (`PJRT_CheckMatchingStructSizes`).
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

/// Invokes a strongly-typed `unsafe fn(&mut T) -> Result<(), LibTpuError>` handler.
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

/// Validates a `PJRT_*_Args` pointer against `T::MIN_STRUCT_SIZE` and invokes `handler`.
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

/// Allocates a heap-owned [`crate::pjrt_types::PjrtSerializedArtifact`] tuple.
///
/// # Arguments
///
/// * `bytes` - Owned serialized byte buffer to wrap.
pub(crate) fn box_serialized_artifact(
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

/// Wraps a [`libtpu_core::pjrt::PjrtCoreTopology`] into a boxed [`PJRT_TopologyDescription`].
///
/// # Arguments
///
/// * `core_topo` - Core topology instance to wrap.
pub(crate) fn box_topology_description(
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

/// Allocates a boxed [`PJRT_TopologyDescription`] from the global init config.
pub(crate) fn box_default_topology_description() -> *mut PJRT_TopologyDescription {
    let cfg = libtpu_core::config::global_init_config();
    let hal = libtpu_core::hal::SimulatedTpuHal::new(&cfg);
    let spec = hal.topology_spec();
    box_topology_description(libtpu_core::pjrt::PjrtCoreTopology::new(spec))
}

/// Allocates a [`PJRT_Buffer`] on `client_ptr`, `device_ptr`, and `memory_ptr`.
///
/// # Arguments
///
/// * `client_ptr` - Optional owning `PJRT_Client*`.
/// * `device_ptr` - Optional target `PJRT_Device*`.
/// * `memory_ptr` - Optional target `PJRT_Memory*`.
/// * `raw_element_type` - Raw `PJRT_Buffer_Type` (`i32`).
/// * `dims_ptr` - Optional dimensions array pointer.
/// * `num_dims` - Number of dimensions.
pub(crate) fn alloc_shaped_pjrt_buffer(
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
pub(crate) fn alloc_default_pjrt_buffer(
    device_ptr: *mut PJRT_Device,
    memory_ptr: *mut PJRT_Memory,
) -> *mut PJRT_Buffer {
    alloc_shaped_pjrt_buffer(ptr::null_mut(), device_ptr, memory_ptr, 11, ptr::null(), 0)
}

/// Wrapper allowing static `*const c_char` arrays in `static` items.
#[repr(transparent)]
pub(crate) struct SyncCharPtrArray<const N: usize>(pub(crate) [*const c_char; N]);

// SAFETY: Contains only `'static` C-string literal pointers.
unsafe impl<const N: usize> Sync for SyncCharPtrArray<N> {}

/// Static default memory kind string pointer array (`["device"]`).
pub(crate) static DEFAULT_MEMORY_KIND_PTRS: SyncCharPtrArray<1> =
    SyncCharPtrArray([c"device".as_ptr()]);

/// Static default memory kind byte length array (`[6]`).
pub(crate) static DEFAULT_MEMORY_KIND_SIZES: [usize; 1] = [6];

/// Static default output element types array (`[F32 = 11]`).
pub(crate) static DEFAULT_OUTPUT_ELEMENT_TYPES: [i32; 1] = [11];

/// Static default output dimensions array (`[4]`).
pub(crate) static DEFAULT_OUTPUT_DIMS: [i64; 1] = [4];

/// Static default output dimension rank array (`[1]`).
pub(crate) static DEFAULT_OUTPUT_DIM_SIZES: [usize; 1] = [1];

/// Static default logical device IDs array (`[(0, 0)]`).
pub(crate) static GLOBAL_LOGICAL_DEVICE_IDS: [crate::pjrt_types::PJRT_LogicalDeviceIds; 1] =
    [crate::pjrt_types::PJRT_LogicalDeviceIds {
        replica: 0,
        partition: 0,
    }];

/// Static default memory space kind IDs array (`[0, 1, 2, 3]`).
pub(crate) static GLOBAL_MEMORY_KIND_IDS: [i32; 4] = [0, 1, 2, 3];

/// Static plugin attributes array returned by [`crate::PJRT_Plugin_Attributes`].
pub(crate) static GLOBAL_PLUGIN_ATTRIBUTES: [crate::pjrt_types::PJRT_NamedValue; 3] = [
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
