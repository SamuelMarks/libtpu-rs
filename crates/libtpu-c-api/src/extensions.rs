//! `#[repr(C)]` PJRT extension structures for `libtpu==0.0.53.1` (`PJRT_Api` `v0.117`).
//!
//! Defines all 18 TPU extensions linked from `PJRT_Api.extension_start` matching the exact byte sizes,
//! type discriminants (`PJRT_Extension_Type`), and named function pointer fields lifted from `libtpu.so`:
//! 1. `PJRT_FFI_Extension` (`48` bytes, `ext_type = 5`)
//! 2. `PJRT_XlaTransform_Extension` (`56` bytes, `ext_type = 24`)
//! 3. `PJRT_HostMemoryAllocator_Extension` (`32` bytes, `ext_type = 23`)
//! 4. `PJRT_MultiSlice_Extension` (`64` bytes, `ext_type = 22`)
//! 5. `PJRT_Collectives_Extension` (`96` bytes, `ext_type = 21`)
//! 6. `PJRT_AbiVersion_Extension` (`120` bytes, `ext_type = 20`)
//! 7. `PJRT_Shardings_Extension` (`40` bytes, `ext_type = 19`)
//! 8. `PJRT_Megascale_Extension` (`288` bytes, `ext_type = 18`)
//! 9. `PJRT_TpuExecutable_Extension` (`96` bytes, `ext_type = 17`)
//! 10. `PJRT_TpuTopology_Extension` (`280` bytes, `ext_type = 16`)
//! 11. `PJRT_Callback_Extension` (`40` bytes, `ext_type = 14`)
//! 12. `PJRT_PhaseCompile_Extension` (`192` bytes, `ext_type = 9`)
//! 13. `PJRT_CrossHostTransfers_Extension` (`56` bytes, `ext_type = 12`)
//! 14. `PJRT_ExecutableMetadata_Extension` (`40` bytes, `ext_type = 13`)
//! 15. `PJRT_MemoryDescriptions_Extension` (`40` bytes, `ext_type = 6`)
//! 16. `PJRT_Layouts_Extension` (`80` bytes, `ext_type = 4`)
//! 17. `PJRT_RawBuffer_Extension` (`80` bytes, `ext_type = 8`)
//! 18. `PJRT_Profiler_Extension` (`40` bytes, `ext_type = 1`)

/// `#[repr(C)]` argument structures, enums, and size assertions for PJRT extensions.
pub mod args;
/// PJRT Collectives and Cross-Host Transfers extension entrypoints.
pub mod collectives;
/// PJRT FFI, XLA Transform, Callback, PhaseCompile/Relocatable, and Triton extension entrypoints.
pub mod compiler;
/// PJRT ABI Version, Shardings, and TPU Executable extension entrypoints.
pub mod executable;
/// PJRT Multi-Slice and Megascale extension entrypoints.
pub mod megascale;
/// PJRT Host Memory Allocator, Metadata, Memory Descriptions, Layouts, and Raw Buffer entrypoints.
pub mod memory;
/// PJRT Profiler and `PLUGIN_Profiler_Api` extension entrypoints.
pub mod profiler;
/// PJRT TPU Topology extension entrypoints.
pub mod topology;

pub use args::*;
pub use collectives::*;
pub use compiler::*;
pub use executable::*;
pub use megascale::*;
pub use memory::*;
pub use profiler::*;
pub use topology::*;

use crate::box_pjrt_error;
use crate::pjrt_helpers::{ffi_catch_unwind, ffi_panic_pjrt_error};
use crate::pjrt_types::{PJRT_Error, PJRT_Extension_Base, PjrtFn};
use libtpu_core::error::LibTpuError;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic counter recording extension invocations across all extension and profiler entrypoints.
static EXTENSION_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns the total number of extension invocations recorded so far.
pub fn extension_call_count() -> u64 {
    EXTENSION_CALL_COUNTER.load(Ordering::Relaxed)
}

/// Validates a PJRT extension argument structure (`PJRT_Extension_ValidateArgs`).
///
/// # Arguments
///
/// * `raw_args` - Opaque pointer to a PJRT extension argument structure whose first field is `size_t struct_size`.
///
/// # Safety
///
/// `raw_args` must be null or point to a struct beginning with `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Extension_ValidateArgs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    ffi_catch_unwind(ptr::null_mut(), || {
        // SAFETY: Caller upholds `raw_args` contract.
        unsafe { validate_extension_args_size(raw_args, size_of::<usize>()) }
    })
}

/// Validates that `raw_args` is non-null and its leading `struct_size` field is at least `min_size`.
///
/// # Arguments
///
/// * `raw_args` - Opaque pointer to a PJRT extension argument structure whose first field is `size_t struct_size`.
/// * `min_size` - Minimum required `struct_size` in bytes.
///
/// # Safety
///
/// `raw_args` must be null or point to a struct beginning with `usize`.
unsafe fn validate_extension_args_size(
    raw_args: *mut std::ffi::c_void,
    min_size: usize,
) -> *mut PJRT_Error {
    if raw_args.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "extension_args",
        });
    }
    // SAFETY: `raw_args` is non-null and starts with `usize struct_size`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < min_size {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Extension_Base",
            expected_min: min_size,
            actual: actual_size,
        });
    }
    EXTENSION_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    ptr::null_mut()
}

/// Allocates an owned C byte buffer handle freed explicitly by [`free_phase_compile_buffer`]
/// (`PJRT_PhaseCompile_CBuffersDestroy`) without any global mutable map or ring buffer.
///
/// # Arguments
///
/// * `bytes` - Phase compilation output bytes.
pub(super) fn alloc_phase_compile_buffer(
    bytes: Vec<u8>,
) -> (*mut std::ffi::c_void, *const u8, usize) {
    let boxed: Box<[u8]> = bytes.into_boxed_slice();
    let ptr = boxed.as_ptr();
    let len = boxed.len();
    let handle = Box::into_raw(Box::new(boxed)).cast::<std::ffi::c_void>();
    (handle, ptr, len)
}

/// Allocates an owned, pointer-aligned `[*mut PJRT_Relocatable]` slice freed by [`free_phase_compile_buffer`].
///
/// # Arguments
///
/// * `relocatables` - Relocatable handles to expose as a contiguous C pointer array.
pub(super) fn alloc_phase_compile_relocatable_array(
    relocatables: Vec<*mut crate::pjrt_types::PJRT_Relocatable>,
) -> (
    *mut std::ffi::c_void,
    *const *mut crate::pjrt_types::PJRT_Relocatable,
    usize,
) {
    let boxed: Box<[*mut crate::pjrt_types::PJRT_Relocatable]> = relocatables.into_boxed_slice();
    let ptr = boxed.as_ptr();
    let len = boxed.len();
    let typed_handle = crate::pjrt_helpers::box_registered_handle(boxed);
    (typed_handle.cast::<std::ffi::c_void>(), ptr, len)
}

/// Frees a phase compilation buffer handle previously allocated by [`alloc_phase_compile_buffer`]
/// or [`alloc_phase_compile_relocatable_array`].
///
/// # Arguments
///
/// * `handle` - Heap handle returned by [`alloc_phase_compile_buffer`] or [`alloc_phase_compile_relocatable_array`].
pub(super) fn free_phase_compile_buffer(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        if crate::pjrt_helpers::take_live_handle::<Box<[*mut crate::pjrt_types::PJRT_Relocatable]>>(
            handle.cast(),
        )
        .is_some()
        {
            return;
        }
        // SAFETY: `handle` was allocated via `Box::into_raw(Box::new(Box<[u8]>))` in `alloc_phase_compile_buffer`.
        unsafe {
            drop(Box::from_raw(handle.cast::<Box<[u8]>>()));
        }
    }
}

/// Invokes a strongly-typed `unsafe fn(&mut T)` from a type-erased reference.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the validated extension `_Args` structure of type `T`.
/// * `handler_ref` - Type-erased pointer to the `unsafe fn(&mut T)` callback.
///
/// # Safety
///
/// `raw_args` must point to a valid `T` and `handler_ref` must point to `unsafe fn(&mut T)`.
unsafe fn invoke_typed_extension_handler<T>(
    raw_args: *mut std::ffi::c_void,
    handler_ref: *const std::ffi::c_void,
) {
    // SAFETY: `handler_ref` points to `handler` on the stack of `dispatch_typed_extension_args`.
    let handler = unsafe { *handler_ref.cast::<unsafe fn(&mut T)>() };
    // SAFETY: `raw_args` was validated non-null by `dispatch_erased_extension_args`.
    unsafe {
        handler(&mut *raw_args.cast::<T>());
    }
}

/// Invokes a strongly-typed fallible `unsafe fn(&mut T) -> Result<(), LibTpuError>` from a type-erased reference.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the validated extension `_Args` structure of type `T`.
/// * `handler_ref` - Type-erased pointer to the `unsafe fn(&mut T) -> Result<(), LibTpuError>` callback.
///
/// # Safety
///
/// `raw_args` must point to a valid `T` and `handler_ref` must point to `unsafe fn(&mut T) -> Result<(), LibTpuError>`.
unsafe fn invoke_typed_fallible_extension_handler<T>(
    raw_args: *mut std::ffi::c_void,
    handler_ref: *const std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: `handler_ref` points to `handler` on the stack of `dispatch_typed_fallible_extension_args`.
    let handler = unsafe { *handler_ref.cast::<unsafe fn(&mut T) -> Result<(), LibTpuError>>() };
    // SAFETY: `raw_args` was validated non-null by `dispatch_erased_fallible_extension_args`.
    match unsafe { handler(&mut *raw_args.cast::<T>()) } {
        Ok(()) => ptr::null_mut(),
        Err(err) => box_pjrt_error(err),
    }
}

/// Non-generic validation and invocation dispatcher backing [`dispatch_typed_extension_args`].
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the extension `_Args` structure.
/// * `min_size` - Minimum required `struct_size` in bytes (`T::MIN_STRUCT_SIZE`).
/// * `handler_ref` - Type-erased pointer to the strongly-typed callback.
/// * `invoker` - Monomorphized trampoline invoking the strongly-typed callback.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid extension `_Args` structure of type `T`.
pub(crate) unsafe fn dispatch_erased_extension_args(
    raw_args: *mut std::ffi::c_void,
    min_size: usize,
    handler_ref: *const std::ffi::c_void,
    invoker: unsafe fn(*mut std::ffi::c_void, *const std::ffi::c_void),
) -> *mut PJRT_Error {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Caller upholds `raw_args` precondition.
        let err = unsafe { validate_extension_args_size(raw_args, min_size) };
        if err.is_null() {
            // SAFETY: `raw_args` was validated non-null and `>= min_size` by `validate_extension_args_size`.
            unsafe {
                invoker(raw_args, handler_ref);
            }
        }
        err
    }))
    .unwrap_or_else(|_| ffi_panic_pjrt_error())
}

/// Non-generic validation and invocation dispatcher backing [`dispatch_typed_fallible_extension_args`].
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the extension `_Args` structure.
/// * `min_size` - Minimum required `struct_size` in bytes (`T::MIN_STRUCT_SIZE`).
/// * `handler_ref` - Type-erased pointer to the strongly-typed fallible callback.
/// * `invoker` - Monomorphized trampoline invoking the strongly-typed fallible callback.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid extension `_Args` structure of type `T`.
pub(crate) unsafe fn dispatch_erased_fallible_extension_args(
    raw_args: *mut std::ffi::c_void,
    min_size: usize,
    handler_ref: *const std::ffi::c_void,
    invoker: unsafe fn(*mut std::ffi::c_void, *const std::ffi::c_void) -> *mut PJRT_Error,
) -> *mut PJRT_Error {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Caller upholds `raw_args` precondition.
        let err = unsafe { validate_extension_args_size(raw_args, min_size) };
        if err.is_null() {
            // SAFETY: `raw_args` was validated non-null and `>= min_size` by `validate_extension_args_size`.
            unsafe { invoker(raw_args, handler_ref) }
        } else {
            err
        }
    }))
    .unwrap_or_else(|_| ffi_panic_pjrt_error())
}

/// Validates a typed `#[repr(C)]` PJRT extension argument structure and invokes `handler(&mut T)`.
///
/// # Arguments
///
/// * `raw_args` - Pointer to the typed extension `_Args` structure of type `T`.
/// * `handler` - Strongly-typed callback operating on `&mut T`.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid extension `_Args` struct of type `T`.
#[inline]
pub(super) unsafe fn dispatch_typed_extension_args<T: ExtensionArgsStruct>(
    raw_args: *mut std::ffi::c_void,
    handler: unsafe fn(&mut T),
) -> *mut PJRT_Error {
    let handler_ref = ptr::from_ref(&handler).cast::<std::ffi::c_void>();
    // SAFETY: Caller upholds `raw_args` contract; `handler_ref` points to `handler`.
    unsafe {
        dispatch_erased_extension_args(
            raw_args,
            T::MIN_STRUCT_SIZE,
            handler_ref,
            invoke_typed_extension_handler::<T>,
        )
    }
}

/// Validates a typed `#[repr(C)]` PJRT extension argument structure and invokes fallible `handler(&mut T)`.
///
/// # Arguments
///
/// * `raw_args` - Pointer to the typed extension `_Args` structure of type `T`.
/// * `handler` - Strongly-typed fallible callback operating on `&mut T`.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid extension `_Args` struct of type `T`.
#[inline]
pub(super) unsafe fn dispatch_typed_fallible_extension_args<T: ExtensionArgsStruct>(
    raw_args: *mut std::ffi::c_void,
    handler: unsafe fn(&mut T) -> Result<(), LibTpuError>,
) -> *mut PJRT_Error {
    let handler_ref = ptr::from_ref(&handler).cast::<std::ffi::c_void>();
    // SAFETY: Caller upholds `raw_args` contract; `handler_ref` points to `handler`.
    unsafe {
        dispatch_erased_fallible_extension_args(
            raw_args,
            T::MIN_STRUCT_SIZE,
            handler_ref,
            invoke_typed_fallible_extension_handler::<T>,
        )
    }
}

/// Node 1 (`ext_type = 5`, `48` bytes): `PJRT_FFI_Extension` for `xla::ffi` custom calls.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_FFI_Extension {
    /// 24-byte `PJRT_Extension_Base` header (`ext_type = 5`).
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `PJRT_FFI_TypeID_Register`.
    pub type_id_register: PjrtFn,
    /// Slot 2 (`0x20`): `PJRT_FFI_UserData_Add`.
    pub user_data_add: PjrtFn,
    /// Slot 3 (`0x28`): `PJRT_FFI_Register_Handler`.
    pub register_handler: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_FFI_Extension {}

/// Node 2 (`ext_type = 24`, `56` bytes): `PJRT_XlaTransform_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `register_xla_transform`.
    pub register_xla_transform: PjrtFn,
    /// Slot 2 (`0x20`): `wait_for_all_transforms`.
    pub wait_for_all_transforms: PjrtFn,
    /// Slot 3 (`0x28`): `unregister_xla_transform`.
    pub unregister_xla_transform: PjrtFn,
    /// Slot 4 (`0x30`): `list_xla_transforms`.
    pub list_xla_transforms: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_XlaTransform_Extension {}

/// Node 3 (`ext_type = 23`, `32` bytes): `PJRT_HostMemoryAllocator_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_HostMemoryAllocator_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `get_host_memory_allocator`.
    pub get_host_memory_allocator: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_HostMemoryAllocator_Extension {}

/// Node 4 (`ext_type = 22`, `64` bytes): `PJRT_MultiSlice_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `config_destroy`.
    pub config_destroy: PjrtFn,
    /// Slot 2 (`0x20`): `config_num_slices`.
    pub config_num_slices: PjrtFn,
    /// Slot 3 (`0x28`): `config_num_devices_per_slice`.
    pub config_num_devices_per_slice: PjrtFn,
    /// Slot 4 (`0x30`): `config_slice_id`.
    pub config_slice_id: PjrtFn,
    /// Slot 5 (`0x38`): `config_serialize`.
    pub config_serialize: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_MultiSlice_Extension {}

/// Node 5 (`ext_type = 21`, `96` bytes): `PJRT_Collectives_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Collectives_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `collectives_destroy`.
    pub collectives_destroy: PjrtFn,
    /// Slot 2 (`0x20`): `create_communicator`.
    pub create_communicator: PjrtFn,
    /// Slot 3 (`0x28`): `communicator_destroy`.
    pub communicator_destroy: PjrtFn,
    /// Slot 4 (`0x30`): `communicator_all_reduce`.
    pub communicator_all_reduce: PjrtFn,
    /// Slot 5 (`0x38`): `communicator_reduce_scatter`.
    pub communicator_reduce_scatter: PjrtFn,
    /// Slot 6 (`0x40`): `communicator_all_gather`.
    pub communicator_all_gather: PjrtFn,
    /// Slot 7 (`0x48`): `communicator_collective_permute`.
    pub communicator_collective_permute: PjrtFn,
    /// Slot 8 (`0x50`): `communicator_all_to_all`.
    pub communicator_all_to_all: PjrtFn,
    /// Slot 9 (`0x58`): `communicator_to_string`.
    pub communicator_to_string: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Collectives_Extension {}

/// Node 6 (`ext_type = 20`, `120` bytes): `PJRT_AbiVersion_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `client_runtime_abi_version`.
    pub client_runtime_abi_version: PjrtFn,
    /// Slot 2 (`0x20`): `executable_get_abi_version`.
    pub executable_get_abi_version: PjrtFn,
    /// Slot 3 (`0x28`): `runtime_abi_version_destroy`.
    pub runtime_abi_version_destroy: PjrtFn,
    /// Slot 4 (`0x30`): `runtime_abi_version_is_compatible_with_runtime`.
    pub runtime_abi_version_is_compatible_with_runtime: PjrtFn,
    /// Slot 5 (`0x38`): `runtime_abi_version_is_compatible_with_executable`.
    pub runtime_abi_version_is_compatible_with_executable: PjrtFn,
    /// Slot 6 (`0x40`): `runtime_abi_version_to_proto`.
    pub runtime_abi_version_to_proto: PjrtFn,
    /// Slot 7 (`0x48`): `runtime_abi_version_platform_id`.
    pub runtime_abi_version_platform_id: PjrtFn,
    /// Slot 8 (`0x50`): `executable_abi_version_destroy`.
    pub executable_abi_version_destroy: PjrtFn,
    /// Slot 9 (`0x58`): `executable_abi_version_to_proto`.
    pub executable_abi_version_to_proto: PjrtFn,
    /// Slot 10 (`0x60`): `executable_abi_version_platform_id`.
    pub executable_abi_version_platform_id: PjrtFn,
    /// Slot 11 (`0x68`): `runtime_abi_version_from_proto`.
    pub runtime_abi_version_from_proto: PjrtFn,
    /// Slot 12 (`0x70`): `executable_abi_version_from_proto`.
    pub executable_abi_version_from_proto: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_AbiVersion_Extension {}

/// Node 7 (`ext_type = 19`, `40` bytes): `PJRT_Shardings_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Shardings_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `PJRT_Shardings_PJRT_Executable_ParameterShardings`.
    pub parameter_shardings: PjrtFn,
    /// Slot 2 (`0x20`): `PJRT_Shardings_PJRT_Executable_OutputShardings`.
    pub output_shardings: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Shardings_Extension {}

/// Node 8 (`ext_type = 18`, `288` bytes): `PJRT_Megascale_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `create_client_context`.
    pub create_client_context: PjrtFn,
    /// Slot 2 (`0x20`): `delete_client_context`.
    pub delete_client_context: PjrtFn,
    /// Slot 3 (`0x28`): `create_default_config`.
    pub create_default_config: PjrtFn,
    /// Slot 4 (`0x30`): `create_aot_config`.
    pub create_aot_config: PjrtFn,
    /// Slot 5 (`0x38`): `create_multi_slice_config`.
    pub create_multi_slice_config: PjrtFn,
    /// Slot 6 (`0x40`): `delete_multi_slice_config`.
    pub delete_multi_slice_config: PjrtFn,
    /// Slot 7 (`0x48`): `multi_slice_config_num_slices`.
    pub multi_slice_config_num_slices: PjrtFn,
    /// Slot 8 (`0x50`): `multi_slice_config_num_devices_per_slice`.
    pub multi_slice_config_num_devices_per_slice: PjrtFn,
    /// Slot 9 (`0x58`): `multi_slice_config_slice_id`.
    pub multi_slice_config_slice_id: PjrtFn,
    /// Slot 10 (`0x60`): `multi_slice_config_serialized_slice_key_value`.
    pub multi_slice_config_serialized_slice_key_value: PjrtFn,
    /// Slot 11 (`0x68`): `multi_slice_config_serialize`.
    pub multi_slice_config_serialize: PjrtFn,
    /// Slot 12 (`0x70`): `deserialize_multi_slice_config`.
    pub deserialize_multi_slice_config: PjrtFn,
    /// Slot 13 (`0x78`): `megascale_addresses`.
    pub megascale_addresses: PjrtFn,
    /// Slot 14 (`0x80`): `barrier`.
    pub barrier: PjrtFn,
    /// Slot 15 (`0x88`): `log_error`.
    pub log_error: PjrtFn,
    /// Slot 16 (`0x90`): `get_megascale_dcn_topology`.
    pub get_megascale_dcn_topology: PjrtFn,
    /// Slot 17 (`0x98`): `get_slice_id`.
    pub get_slice_id: PjrtFn,
    /// Slot 18 (`0xa0`): `get_num_slices`.
    pub get_num_slices: PjrtFn,
    /// Slot 19 (`0xa8`): `initialize_slice`.
    pub initialize_slice: PjrtFn,
    /// Slot 20 (`0xb0`): `finalize_slice`.
    pub finalize_slice: PjrtFn,
    /// Slot 21 (`0xb8`): `register_error_handler`.
    pub register_error_handler: PjrtFn,
    /// Slot 22 (`0xc0`): `unregister_error_handler`.
    pub unregister_error_handler: PjrtFn,
    /// Slot 23 (`0xc8`): `collect_metrics`.
    pub collect_metrics: PjrtFn,
    /// Slot 24 (`0xd0`): `reset_metrics`.
    pub reset_metrics: PjrtFn,
    /// Slot 25 (`0xd8`): `get_transport_type`.
    pub get_transport_type: PjrtFn,
    /// Slot 26 (`0xe0`): `set_transport_type`.
    pub set_transport_type: PjrtFn,
    /// Slot 27 (`0xe8`): `wait_for_barrier`.
    pub wait_for_barrier: PjrtFn,
    /// Slot 28 (`0xf0`): `abort_collectives`.
    pub abort_collectives: PjrtFn,
    /// Slot 29 (`0xf8`): `discover_slices`.
    pub discover_slices: PjrtFn,
    /// Slot 30 (`0x100`): `multi_slice_config_get_incarnation_id`.
    pub multi_slice_config_get_incarnation_id: PjrtFn,
    /// Slot 31 (`0x108`): `multi_slice_config_get_local_host_id`.
    pub multi_slice_config_get_local_host_id: PjrtFn,
    /// Slot 32 (`0x110`): `multi_slice_config_get_num_hosts`.
    pub multi_slice_config_get_num_hosts: PjrtFn,
    /// Slot 33 (`0x118`): `global_barrier`.
    pub global_barrier: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Megascale_Extension {}

/// Node 9 (`ext_type = 17`, `96` bytes): `PJRT_TpuExecutable_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `get_target_arguments`.
    pub get_target_arguments: PjrtFn,
    /// Slot 2 (`0x20`): `get_hlo_module_with_config`.
    pub get_hlo_module_with_config: PjrtFn,
    /// Slot 3 (`0x28`): `reserved`.
    pub reserved: PjrtFn,
    /// Slot 4 (`0x30`): `get_compiled_memory_stats`.
    pub get_compiled_memory_stats: PjrtFn,
    /// Slot 5 (`0x38`): `run_hlo_cost_analysis`.
    pub run_hlo_cost_analysis: PjrtFn,
    /// Slot 6 (`0x40`): `set_tpu_compilation_env`.
    pub set_tpu_compilation_env: PjrtFn,
    /// Slot 7 (`0x48`): `get_tpu_compilation_env_field_as_string`.
    pub get_tpu_compilation_env_field_as_string: PjrtFn,
    /// Slot 8 (`0x50`): `is_tpu_predetermined_error`.
    pub is_tpu_predetermined_error: PjrtFn,
    /// Slot 9 (`0x58`): `is_tpu_cancellation_placeholder_error`.
    pub is_tpu_cancellation_placeholder_error: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_TpuExecutable_Extension {}

/// Node 10 (`ext_type = 16`, `280` bytes): `PJRT_TpuTopology_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `subslice`.
    pub subslice: PjrtFn,
    /// Slot 2 (`0x20`): `is_subslice_topology`.
    pub is_subslice_topology: PjrtFn,
    /// Slot 3 (`0x28`): `subslice_device_id_from_full_device_id`.
    pub subslice_device_id_from_full_device_id: PjrtFn,
    /// Slot 4 (`0x30`): `replace_host_bounds`.
    pub replace_host_bounds: PjrtFn,
    /// Slot 5 (`0x38`): `is_enhanced_barrier_enabled`.
    pub is_enhanced_barrier_enabled: PjrtFn,
    /// Slot 6 (`0x40`): `has_limited_ici_connectivity`.
    pub has_limited_ici_connectivity: PjrtFn,
    /// Slot 7 (`0x48`): `is_reachable_over_limited_ici`.
    pub is_reachable_over_limited_ici: PjrtFn,
    /// Slot 8 (`0x50`): `process_count`.
    pub process_count: PjrtFn,
    /// Slot 9 (`0x58`): `chips_per_process`.
    pub chips_per_process: PjrtFn,
    /// Slot 10 (`0x60`): `core_count_per_chip`.
    pub core_count_per_chip: PjrtFn,
    /// Slot 11 (`0x68`): `chip_count`.
    pub chip_count: PjrtFn,
    /// Slot 12 (`0x70`): `core_count`.
    pub core_count: PjrtFn,
    /// Slot 13 (`0x78`): `logi_device_count_per_process`.
    pub logi_device_count_per_process: PjrtFn,
    /// Slot 14 (`0x80`): `logi_device_count`.
    pub logi_device_count: PjrtFn,
    /// Slot 15 (`0x88`): `logi_device_count_per_chip`.
    pub logi_device_count_per_chip: PjrtFn,
    /// Slot 16 (`0x90`): `core_count_per_process`.
    pub core_count_per_process: PjrtFn,
    /// Slot 17 (`0x98`): `process_ids`.
    pub process_ids: PjrtFn,
    /// Slot 18 (`0xa0`): `logi_device_ids_on_process`.
    pub logi_device_ids_on_process: PjrtFn,
    /// Slot 19 (`0xa8`): `proc_id_and_idx_on_proc_for_chip`.
    pub proc_id_and_idx_on_proc_for_chip: PjrtFn,
    /// Slot 20 (`0xb0`): `proc_id_and_idx_on_proc_for_logi_device`.
    pub proc_id_and_idx_on_proc_for_logi_device: PjrtFn,
    /// Slot 21 (`0xb8`): `process_coord_from_id`.
    pub process_coord_from_id: PjrtFn,
    /// Slot 22 (`0xc0`): `chip_id_from_coord`.
    pub chip_id_from_coord: PjrtFn,
    /// Slot 23 (`0xc8`): `logi_device_id_from_chip_coord_and_idx`.
    pub logi_device_id_from_chip_coord_and_idx: PjrtFn,
    /// Slot 24 (`0xd0`): `chip_coord_and_idx_for_logi_device`.
    pub chip_coord_and_idx_for_logi_device: PjrtFn,
    /// Slot 25 (`0xd8`): `chips_per_process_bounds`.
    pub chips_per_process_bounds: PjrtFn,
    /// Slot 26 (`0xe0`): `chip_bounds`.
    pub chip_bounds: PjrtFn,
    /// Slot 27 (`0xe8`): `process_bounds`.
    pub process_bounds: PjrtFn,
    /// Slot 28 (`0xf0`): `get_routing_strategy`.
    pub get_routing_strategy: PjrtFn,
    /// Slot 29 (`0xf8`): `get_slice_config`.
    pub get_slice_config: PjrtFn,
    /// Slot 30 (`0x100`): `get_slice_configs`.
    pub get_slice_configs: PjrtFn,
    /// Slot 31 (`0x108`): `get_default_platform_config`.
    pub get_default_platform_config: PjrtFn,
    /// Slot 32 (`0x110`): `compute_subslice_for_devices`.
    pub compute_subslice_for_devices: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_TpuTopology_Extension {}

/// Node 11 (`ext_type = 14`, `40` bytes): `PJRT_Callback_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `PJRT_Callback_RegisterCallback`.
    pub register_callback: PjrtFn,
    /// Slot 2 (`0x20`): `PJRT_Callback_InvokeCallback`.
    pub invoke_callback: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Callback_Extension {}

/// Node 12 (`ext_type = 9`, `192` bytes): `PJRT_PhaseCompile_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `phase_compile_get_compiler`.
    pub phase_compile_get_compiler: PjrtFn,
    /// Slot 2 (`0x20`): `phase_compile_destroy_compiler`.
    pub phase_compile_destroy_compiler: PjrtFn,
    /// Slot 3 (`0x28`): `phase_compile_run_phase`.
    pub phase_compile_run_phase: PjrtFn,
    /// Slot 4 (`0x30`): `phase_compile_get_phase_names`.
    pub phase_compile_get_phase_names: PjrtFn,
    /// Slot 5 (`0x38`): `phase_compile_c_buffers_destroy`.
    pub phase_compile_c_buffers_destroy: PjrtFn,
    /// Slot 6 (`0x40`): `relocatable_destroy`.
    pub relocatable_destroy: PjrtFn,
    /// Slot 7 (`0x48`): `relocatable_name`.
    pub relocatable_name: PjrtFn,
    /// Slot 8 (`0x50`): `relocatable_build_id`.
    pub relocatable_build_id: PjrtFn,
    /// Slot 9 (`0x58`): `relocatable_call_backend_config`.
    pub relocatable_call_backend_config: PjrtFn,
    /// Slot 10 (`0x60`): `relocatable_has_side_effects`.
    pub relocatable_has_side_effects: PjrtFn,
    /// Slot 11 (`0x68`): `relocatable_serialize`.
    pub relocatable_serialize: PjrtFn,
    /// Slot 12 (`0x70`): `relocatable_parameter_element_types`.
    pub relocatable_parameter_element_types: PjrtFn,
    /// Slot 13 (`0x78`): `relocatable_output_element_types`.
    pub relocatable_output_element_types: PjrtFn,
    /// Slot 14 (`0x80`): `relocatable_parameter_dimensions`.
    pub relocatable_parameter_dimensions: PjrtFn,
    /// Slot 15 (`0x88`): `relocatable_output_dimensions`.
    pub relocatable_output_dimensions: PjrtFn,
    /// Slot 16 (`0x90`): `relocatable_parameter_layouts`.
    pub relocatable_parameter_layouts: PjrtFn,
    /// Slot 17 (`0x98`): `relocatable_output_layouts`.
    pub relocatable_output_layouts: PjrtFn,
    /// Slot 18 (`0xa0`): `relocatable_parameter_output_aliases`.
    pub relocatable_parameter_output_aliases: PjrtFn,
    /// Slot 19 (`0xa8`): `phase_compile_compile_to_relocatable`.
    pub phase_compile_compile_to_relocatable: PjrtFn,
    /// Slot 20 (`0xb0`): `phase_compile_deserialize_relocatable`.
    pub phase_compile_deserialize_relocatable: PjrtFn,
    /// Slot 21 (`0xb8`): `phase_compile_link_relocatables`.
    pub phase_compile_link_relocatables: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_PhaseCompile_Extension {}

/// Node 13 (`ext_type = 12`, `56` bytes): `PJRT_CrossHostTransfers_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CrossHostTransfers_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `make_cross_host_receive_buffers`.
    pub make_cross_host_receive_buffers: PjrtFn,
    /// Slot 2 (`0x20`): `copy_to_remote_device`.
    pub copy_to_remote_device: PjrtFn,
    /// Slot 3 (`0x28`): `cross_host_receive_buffers`.
    pub cross_host_receive_buffers: PjrtFn,
    /// Slot 4 (`0x30`): `cross_host_send_buffers`.
    pub cross_host_send_buffers: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_CrossHostTransfers_Extension {}

/// Node 14 (`ext_type = 13`, `40` bytes): `PJRT_ExecutableMetadata_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecutableMetadata_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `get_executable_metadata`.
    pub get_executable_metadata: PjrtFn,
    /// Slot 2 (`0x20`): `destroy_executable_metadata`.
    pub destroy_executable_metadata: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_ExecutableMetadata_Extension {}

/// Node 15 (`ext_type = 6`, `40` bytes): `PJRT_MemoryDescriptions_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MemoryDescriptions_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `PJRT_MemoryDescriptions_GetMemoryDescriptions`.
    pub get_memory_descriptions: PjrtFn,
    /// Slot 2 (`0x20`): `PJRT_MemoryDescriptions_Destroy`.
    pub destroy_memory_descriptions: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_MemoryDescriptions_Extension {}

/// Node 16 (`ext_type = 4`, `80` bytes): `PJRT_Layouts_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `memory_layout_destroy`.
    pub memory_layout_destroy: PjrtFn,
    /// Slot 2 (`0x20`): `memory_layout_serialize`.
    pub memory_layout_serialize: PjrtFn,
    /// Slot 3 (`0x28`): `client_get_default_layout`.
    pub client_get_default_layout: PjrtFn,
    /// Slot 4 (`0x30`): `buffer_memory_layout`.
    pub buffer_memory_layout: PjrtFn,
    /// Slot 5 (`0x38`): `topology_get_default_layout`.
    pub topology_get_default_layout: PjrtFn,
    /// Slot 6 (`0x40`): `executable_get_output_layouts`.
    pub executable_get_output_layouts: PjrtFn,
    /// Slot 7 (`0x48`): `executable_get_parameter_layouts`.
    pub executable_get_parameter_layouts: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Layouts_Extension {}

/// Node 17 (`ext_type = 8`, `80` bytes): `PJRT_RawBuffer_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `create_raw_alias_of_buffer`.
    pub create_raw_alias_of_buffer: PjrtFn,
    /// Slot 2 (`0x20`): `destroy`.
    pub destroy: PjrtFn,
    /// Slot 3 (`0x28`): `get_host_pointer`.
    pub get_host_pointer: PjrtFn,
    /// Slot 4 (`0x30`): `get_on_device_size_in_bytes`.
    pub get_on_device_size_in_bytes: PjrtFn,
    /// Slot 5 (`0x38`): `get_memory_space`.
    pub get_memory_space: PjrtFn,
    /// Slot 6 (`0x40`): `copy_raw_host_to_device`.
    pub copy_raw_host_to_device: PjrtFn,
    /// Slot 7 (`0x48`): `copy_raw_device_to_host`.
    pub copy_raw_device_to_host: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_RawBuffer_Extension {}

/// Node 18 (`ext_type = 1`, `40` bytes): `PJRT_Profiler_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Profiler_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): pointer to `PLUGIN_Profiler_Api`.
    pub profiler_api: *const PLUGIN_Profiler_Api,
    /// Slot 2 (`0x20`): `traceme_context_id`.
    pub traceme_context_id: i64,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Profiler_Extension {}

const _: () = assert!(size_of::<PJRT_FFI_Extension>() == 48);
const _: () = assert!(std::mem::offset_of!(PJRT_FFI_Extension, register_handler) == 0x28);
const _: () = assert!(size_of::<PJRT_XlaTransform_Extension>() == 56);
const _: () =
    assert!(std::mem::offset_of!(PJRT_XlaTransform_Extension, register_xla_transform) == 0x18);
const _: () = assert!(size_of::<PJRT_HostMemoryAllocator_Extension>() == 32);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_HostMemoryAllocator_Extension,
        get_host_memory_allocator
    ) == 0x18
);
const _: () = assert!(size_of::<PJRT_MultiSlice_Extension>() == 64);
const _: () = assert!(std::mem::offset_of!(PJRT_MultiSlice_Extension, config_serialize) == 0x38);
const _: () = assert!(size_of::<PJRT_Collectives_Extension>() == 96);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Collectives_Extension, communicator_to_string) == 0x58);
const _: () = assert!(size_of::<PJRT_AbiVersion_Extension>() == 120);
const _: () = assert!(
    std::mem::offset_of!(PJRT_AbiVersion_Extension, executable_abi_version_from_proto) == 0x70
);
const _: () = assert!(size_of::<PJRT_Shardings_Extension>() == 40);
const _: () = assert!(std::mem::offset_of!(PJRT_Shardings_Extension, output_shardings) == 0x20);
const _: () = assert!(size_of::<PJRT_Megascale_Extension>() == 288);
const _: () = assert!(std::mem::offset_of!(PJRT_Megascale_Extension, global_barrier) == 0x118);
const _: () = assert!(size_of::<PJRT_TpuExecutable_Extension>() == 96);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_TpuExecutable_Extension,
        is_tpu_cancellation_placeholder_error
    ) == 0x58
);
const _: () = assert!(size_of::<PJRT_TpuTopology_Extension>() == 280);
const _: () = assert!(
    std::mem::offset_of!(PJRT_TpuTopology_Extension, compute_subslice_for_devices) == 0x110
);
const _: () = assert!(size_of::<PJRT_Callback_Extension>() == 40);
const _: () = assert!(std::mem::offset_of!(PJRT_Callback_Extension, invoke_callback) == 0x20);
const _: () = assert!(size_of::<PJRT_PhaseCompile_Extension>() == 192);
const _: () = assert!(
    std::mem::offset_of!(PJRT_PhaseCompile_Extension, phase_compile_link_relocatables) == 0xb8
);
const _: () = assert!(size_of::<PJRT_CrossHostTransfers_Extension>() == 56);
const _: () = assert!(
    std::mem::offset_of!(PJRT_CrossHostTransfers_Extension, cross_host_send_buffers) == 0x30
);
const _: () = assert!(size_of::<PJRT_ExecutableMetadata_Extension>() == 40);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_ExecutableMetadata_Extension,
        destroy_executable_metadata
    ) == 0x20
);
const _: () = assert!(size_of::<PJRT_MemoryDescriptions_Extension>() == 40);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_MemoryDescriptions_Extension,
        destroy_memory_descriptions
    ) == 0x20
);
const _: () = assert!(size_of::<PJRT_Layouts_Extension>() == 80);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Layouts_Extension, executable_get_parameter_layouts) == 0x48);
const _: () = assert!(size_of::<PJRT_RawBuffer_Extension>() == 80);
const _: () =
    assert!(std::mem::offset_of!(PJRT_RawBuffer_Extension, copy_raw_device_to_host) == 0x48);
const _: () = assert!(size_of::<PJRT_Profiler_Extension>() == 40);
const _: () = assert!(std::mem::offset_of!(PJRT_Profiler_Extension, traceme_context_id) == 0x20);

/// Standalone `PJRT_FFI_Extension` descriptor (`ext_type = 5`, `48` bytes).
pub static GLOBAL_PJRT_FFI_EXTENSION: PJRT_FFI_Extension = PJRT_FFI_Extension {
    base: PJRT_Extension_Base {
        struct_size: 48,
        ext_type: 5,
        pad: 0,
        next: ptr::null(),
    },
    type_id_register: PJRT_FFI_TypeID_Register,
    user_data_add: PJRT_FFI_UserData_Add,
    register_handler: PJRT_FFI_Register_Handler,
};

/// Returns the static [`PJRT_FFI_Extension`] descriptor (`PJRT_FFI_GetExtension`).
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn PJRT_FFI_GetExtension() -> *const PJRT_FFI_Extension {
    ffi_catch_unwind(ptr::null(), || ptr::addr_of!(GLOBAL_PJRT_FFI_EXTENSION))
}

/// Static Node 18 (`PJRT_Profiler_Extension`, `ext_type = 1`, `40` bytes).
pub static EXT_18_PROFILER: PJRT_Profiler_Extension = PJRT_Profiler_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 1,
        pad: 0,
        next: ptr::null(),
    },
    profiler_api: ptr::addr_of!(profiler::NEGOTIATED_PLUGIN_PROFILER_API),
    traceme_context_id: 0,
};

/// Static Node 17 (`PJRT_RawBuffer_Extension`, `ext_type = 8`, `80` bytes).
pub static EXT_17_RAW_BUFFER: PJRT_RawBuffer_Extension = PJRT_RawBuffer_Extension {
    base: PJRT_Extension_Base {
        struct_size: 80,
        ext_type: 8,
        pad: 0,
        next: ptr::addr_of!(EXT_18_PROFILER.base),
    },
    create_raw_alias_of_buffer: PJRT_RawBuffer_CreateRawAliasOfBuffer,
    destroy: PJRT_RawBuffer_Destroy,
    get_host_pointer: PJRT_RawBuffer_GetHostPointer,
    get_on_device_size_in_bytes: PJRT_RawBuffer_GetOnDeviceSizeInBytes,
    get_memory_space: PJRT_RawBuffer_GetMemorySpace,
    copy_raw_host_to_device: PJRT_RawBuffer_CopyRawHostToDevice,
    copy_raw_device_to_host: PJRT_RawBuffer_CopyRawDeviceToHost,
};

/// Static Node 16 (`PJRT_Layouts_Extension`, `ext_type = 4`, `80` bytes).
pub static EXT_16_LAYOUTS: PJRT_Layouts_Extension = PJRT_Layouts_Extension {
    base: PJRT_Extension_Base {
        struct_size: 80,
        ext_type: 4,
        pad: 0,
        next: ptr::addr_of!(EXT_17_RAW_BUFFER.base),
    },
    memory_layout_destroy: PJRT_Layouts_MemoryLayout_Destroy,
    memory_layout_serialize: PJRT_Layouts_MemoryLayout_Serialize,
    client_get_default_layout: PJRT_Layouts_Client_GetDefaultLayout,
    buffer_memory_layout: PJRT_Layouts_Buffer_GetMemoryLayout,
    topology_get_default_layout: PJRT_Layouts_Topology_GetDefaultLayout,
    executable_get_output_layouts: PJRT_Layouts_Executable_GetOutputLayouts,
    executable_get_parameter_layouts: PJRT_Layouts_Executable_GetParameterLayouts,
};

/// Static Node 15 (`PJRT_MemoryDescriptions_Extension`, `ext_type = 6`, `40` bytes).
pub static EXT_15_MEM_DESC: PJRT_MemoryDescriptions_Extension = PJRT_MemoryDescriptions_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 6,
        pad: 0,
        next: ptr::addr_of!(EXT_16_LAYOUTS.base),
    },
    get_memory_descriptions: PJRT_MemoryDescriptions_Get,
    destroy_memory_descriptions: PJRT_MemoryDescriptions_Destroy,
};

/// Static Node 14 (`PJRT_ExecutableMetadata_Extension`, `ext_type = 13`, `40` bytes).
pub static EXT_14_EXEC_METADATA: PJRT_ExecutableMetadata_Extension =
    PJRT_ExecutableMetadata_Extension {
        base: PJRT_Extension_Base {
            struct_size: 40,
            ext_type: 13,
            pad: 0,
            next: ptr::addr_of!(EXT_15_MEM_DESC.base),
        },
        get_executable_metadata: PJRT_ExecutableMetadata_Get,
        destroy_executable_metadata: PJRT_ExecutableMetadata_Destroy,
    };

/// Static Node 13 (`PJRT_CrossHostTransfers_Extension`, `ext_type = 12`, `56` bytes).
pub static EXT_13_CROSS_HOST: PJRT_CrossHostTransfers_Extension =
    PJRT_CrossHostTransfers_Extension {
        base: PJRT_Extension_Base {
            struct_size: 56,
            ext_type: 12,
            pad: 0,
            next: ptr::addr_of!(EXT_14_EXEC_METADATA.base),
        },
        make_cross_host_receive_buffers: PJRT_CrossHostTransfers_MakeReceiveBuffers,
        copy_to_remote_device: PJRT_CrossHostTransfers_CopyToRemoteDevice,
        cross_host_receive_buffers: PJRT_CrossHostTransfers_ReceiveBuffers,
        cross_host_send_buffers: PJRT_CrossHostTransfers_SendBuffers,
    };

/// Static Node 12 (`PJRT_PhaseCompile_Extension`, `ext_type = 9`, `192` bytes).
pub static EXT_12_PHASE_COMPILE: PJRT_PhaseCompile_Extension = PJRT_PhaseCompile_Extension {
    base: PJRT_Extension_Base {
        struct_size: 192,
        ext_type: 9,
        pad: 0,
        next: ptr::addr_of!(EXT_13_CROSS_HOST.base),
    },
    phase_compile_get_compiler: PJRT_PhaseCompile_GetCompiler,
    phase_compile_destroy_compiler: PJRT_PhaseCompile_DestroyCompiler,
    phase_compile_run_phase: PJRT_PhaseCompile_RunPhase,
    phase_compile_get_phase_names: PJRT_PhaseCompile_GetPhaseNames,
    phase_compile_c_buffers_destroy: PJRT_PhaseCompile_CBuffersDestroy,
    relocatable_destroy: PJRT_Relocatable_Destroy,
    relocatable_name: PJRT_Relocatable_Name,
    relocatable_build_id: PJRT_Relocatable_BuildId,
    relocatable_call_backend_config: PJRT_Relocatable_CallBackendConfig,
    relocatable_has_side_effects: PJRT_Relocatable_HasSideEffects,
    relocatable_serialize: PJRT_Relocatable_Serialize,
    relocatable_parameter_element_types: PJRT_Relocatable_ParameterElementTypes,
    relocatable_output_element_types: PJRT_Relocatable_OutputElementTypes,
    relocatable_parameter_dimensions: PJRT_Relocatable_ParameterDimensions,
    relocatable_output_dimensions: PJRT_Relocatable_OutputDimensions,
    relocatable_parameter_layouts: PJRT_Relocatable_ParameterLayouts,
    relocatable_output_layouts: PJRT_Relocatable_OutputLayouts,
    relocatable_parameter_output_aliases: PJRT_Relocatable_ParameterOutputAliases,
    phase_compile_compile_to_relocatable: PJRT_PhaseCompile_CompileToRelocatable,
    phase_compile_deserialize_relocatable: PJRT_PhaseCompile_DeserializeRelocatable,
    phase_compile_link_relocatables: PJRT_PhaseCompile_LinkRelocatables,
};

/// Static Node 11 (`PJRT_Callback_Extension`, `ext_type = 14`, `40` bytes).
pub static EXT_11_CALLBACK: PJRT_Callback_Extension = PJRT_Callback_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 14,
        pad: 0,
        next: ptr::addr_of!(EXT_12_PHASE_COMPILE.base),
    },
    register_callback: PJRT_Callback_RegisterCallback,
    invoke_callback: PJRT_Callback_InvokeCallback,
};

/// Static Node 10 (`PJRT_TpuTopology_Extension`, `ext_type = 16`, `280` bytes).
pub static EXT_10_TPU_TOPOLOGY: PJRT_TpuTopology_Extension = PJRT_TpuTopology_Extension {
    base: PJRT_Extension_Base {
        struct_size: 280,
        ext_type: 16,
        pad: 0,
        next: ptr::addr_of!(EXT_11_CALLBACK.base),
    },
    subslice: PJRT_TpuTopology_Subslice,
    is_subslice_topology: PJRT_TpuTopology_IsSubsliceTopology,
    subslice_device_id_from_full_device_id: PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId,
    replace_host_bounds: PJRT_TpuTopology_ReplaceHostBounds,
    is_enhanced_barrier_enabled: PJRT_TpuTopology_IsEnhancedBarrierEnabled,
    has_limited_ici_connectivity: PJRT_TpuTopology_HasLimitedIciConnectivity,
    is_reachable_over_limited_ici: PJRT_TpuTopology_IsReachableOverLimitedIci,
    process_count: PJRT_TpuTopology_ProcessCount,
    chips_per_process: PJRT_TpuTopology_ChipsPerProcess,
    core_count_per_chip: PJRT_TpuTopology_CoreCountPerChip,
    chip_count: PJRT_TpuTopology_ChipCount,
    core_count: PJRT_TpuTopology_CoreCount,
    logi_device_count_per_process: PJRT_TpuTopology_LogiDeviceCountPerProcess,
    logi_device_count: PJRT_TpuTopology_LogiDeviceCount,
    logi_device_count_per_chip: PJRT_TpuTopology_LogiDeviceCountPerChip,
    core_count_per_process: PJRT_TpuTopology_CoreCountPerProcess,
    process_ids: PJRT_TpuTopology_ProcessIds,
    logi_device_ids_on_process: PJRT_TpuTopology_LogiDeviceIdsOnProcess,
    proc_id_and_idx_on_proc_for_chip: PJRT_TpuTopology_ProcIdAndIdxOnProcForChip,
    proc_id_and_idx_on_proc_for_logi_device: PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice,
    process_coord_from_id: PJRT_TpuTopology_ProcessCoordFromId,
    chip_id_from_coord: PJRT_TpuTopology_ChipIdFromCoord,
    logi_device_id_from_chip_coord_and_idx: PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx,
    chip_coord_and_idx_for_logi_device: PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice,
    chips_per_process_bounds: PJRT_TpuTopology_ChipsPerProcessBounds,
    chip_bounds: PJRT_TpuTopology_ChipBounds,
    process_bounds: PJRT_TpuTopology_ProcessBounds,
    get_routing_strategy: PJRT_TpuTopology_GetRoutingStrategy,
    get_slice_config: PJRT_TpuTopology_GetSliceConfig,
    get_slice_configs: PJRT_TpuTopology_GetSliceConfigs,
    get_default_platform_config: PJRT_TpuTopology_GetDefaultPlatformConfig,
    compute_subslice_for_devices: PJRT_TpuTopology_ComputeSubsliceForDevices,
};

/// Static Node 9 (`PJRT_TpuExecutable_Extension`, `ext_type = 17`, `96` bytes).
pub static EXT_09_TPU_EXECUTABLE: PJRT_TpuExecutable_Extension = PJRT_TpuExecutable_Extension {
    base: PJRT_Extension_Base {
        struct_size: 96,
        ext_type: 17,
        pad: 0,
        next: ptr::addr_of!(EXT_10_TPU_TOPOLOGY.base),
    },
    get_target_arguments: PJRT_TpuExecutable_GetTargetArguments,
    get_hlo_module_with_config: PJRT_TpuExecutable_GetHloModuleWithConfig,
    reserved: PJRT_TpuExecutable_Reserved,
    get_compiled_memory_stats: PJRT_TpuExecutable_GetCompiledMemoryStats,
    run_hlo_cost_analysis: PJRT_TpuExecutable_RunHloCostAnalysis,
    set_tpu_compilation_env: PJRT_TpuExecutable_SetTpuCompilationEnv,
    get_tpu_compilation_env_field_as_string: PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString,
    is_tpu_predetermined_error: PJRT_TpuExecutable_IsPredeterminedError,
    is_tpu_cancellation_placeholder_error: PJRT_TpuExecutable_IsCancellationPlaceholderError,
};

/// Static Node 8 (`PJRT_Megascale_Extension`, `ext_type = 18`, `288` bytes).
pub static EXT_08_MEGASCALE: PJRT_Megascale_Extension = PJRT_Megascale_Extension {
    base: PJRT_Extension_Base {
        struct_size: 288,
        ext_type: 18,
        pad: 0,
        next: ptr::addr_of!(EXT_09_TPU_EXECUTABLE.base),
    },
    create_client_context: PJRT_Megascale_CreateClientContext,
    delete_client_context: PJRT_Megascale_DeleteClientContext,
    create_default_config: PJRT_Megascale_CreateDefaultConfig,
    create_aot_config: PJRT_Megascale_CreateAotConfig,
    create_multi_slice_config: PJRT_Megascale_CreateMultiSliceConfig,
    delete_multi_slice_config: PJRT_Megascale_DeleteMultiSliceConfig,
    multi_slice_config_num_slices: PJRT_Megascale_ConfigNumSlices,
    multi_slice_config_num_devices_per_slice: PJRT_Megascale_ConfigNumDevicesPerSlice,
    multi_slice_config_slice_id: PJRT_Megascale_ConfigSliceId,
    multi_slice_config_serialized_slice_key_value: PJRT_Megascale_ConfigSerializedSliceKeyValue,
    multi_slice_config_serialize: PJRT_Megascale_ConfigSerialize,
    deserialize_multi_slice_config: PJRT_Megascale_DeserializeMultiSliceConfig,
    megascale_addresses: PJRT_Megascale_Addresses,
    barrier: PJRT_Megascale_Barrier,
    log_error: PJRT_Megascale_LogError,
    get_megascale_dcn_topology: PJRT_Megascale_GetDcnTopology,
    get_slice_id: PJRT_Megascale_GetSliceId,
    get_num_slices: PJRT_Megascale_GetNumSlices,
    initialize_slice: PJRT_Megascale_InitializeSlice,
    finalize_slice: PJRT_Megascale_FinalizeSlice,
    register_error_handler: PJRT_Megascale_RegisterErrorHandler,
    unregister_error_handler: PJRT_Megascale_UnregisterErrorHandler,
    collect_metrics: PJRT_Megascale_CollectMetrics,
    reset_metrics: PJRT_Megascale_ResetMetrics,
    get_transport_type: PJRT_Megascale_GetTransportType,
    set_transport_type: PJRT_Megascale_SetTransportType,
    wait_for_barrier: PJRT_Megascale_WaitForBarrier,
    abort_collectives: PJRT_Megascale_AbortCollectives,
    discover_slices: PJRT_Megascale_DiscoverSlices,
    multi_slice_config_get_incarnation_id: PJRT_Megascale_MultiSliceConfig_GetIncarnationId,
    multi_slice_config_get_local_host_id: PJRT_Megascale_MultiSliceConfig_GetLocalHostId,
    multi_slice_config_get_num_hosts: PJRT_Megascale_MultiSliceConfig_GetNumHosts,
    global_barrier: PJRT_Megascale_GlobalBarrier,
};

/// Static Node 7 (`PJRT_Shardings_Extension`, `ext_type = 19`, `40` bytes).
pub static EXT_07_SHARDINGS: PJRT_Shardings_Extension = PJRT_Shardings_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 19,
        pad: 0,
        next: ptr::addr_of!(EXT_08_MEGASCALE.base),
    },
    parameter_shardings: PJRT_Shardings_ParameterShardings,
    output_shardings: PJRT_Shardings_OutputShardings,
};

/// Static Node 6 (`PJRT_AbiVersion_Extension`, `ext_type = 20`, `120` bytes).
pub static EXT_06_ABI_VERSION: PJRT_AbiVersion_Extension = PJRT_AbiVersion_Extension {
    base: PJRT_Extension_Base {
        struct_size: 120,
        ext_type: 20,
        pad: 0,
        next: ptr::addr_of!(EXT_07_SHARDINGS.base),
    },
    client_runtime_abi_version: PJRT_AbiVersion_ClientRuntime,
    executable_get_abi_version: PJRT_AbiVersion_ExecutableGet,
    runtime_abi_version_destroy: PJRT_AbiVersion_RuntimeDestroy,
    runtime_abi_version_is_compatible_with_runtime: PJRT_AbiVersion_RuntimeCompatibleWithRuntime,
    runtime_abi_version_is_compatible_with_executable:
        PJRT_AbiVersion_RuntimeCompatibleWithExecutable,
    runtime_abi_version_to_proto: PJRT_AbiVersion_RuntimeToProto,
    runtime_abi_version_platform_id: PJRT_AbiVersion_RuntimePlatformId,
    executable_abi_version_destroy: PJRT_AbiVersion_ExecutableDestroy,
    executable_abi_version_to_proto: PJRT_AbiVersion_ExecutableToProto,
    executable_abi_version_platform_id: PJRT_AbiVersion_ExecutablePlatformId,
    runtime_abi_version_from_proto: PJRT_AbiVersion_RuntimeFromProto,
    executable_abi_version_from_proto: PJRT_AbiVersion_ExecutableFromProto,
};

/// Static Node 5 (`PJRT_Collectives_Extension`, `ext_type = 21`, `96` bytes).
pub static EXT_05_COLLECTIVES: PJRT_Collectives_Extension = PJRT_Collectives_Extension {
    base: PJRT_Extension_Base {
        struct_size: 96,
        ext_type: 21,
        pad: 0,
        next: ptr::addr_of!(EXT_06_ABI_VERSION.base),
    },
    collectives_destroy: PJRT_Collectives_Destroy,
    create_communicator: PJRT_Collectives_CreateCommunicator,
    communicator_destroy: PJRT_Communicator_Destroy,
    communicator_all_reduce: PJRT_Communicator_AllReduce,
    communicator_reduce_scatter: PJRT_Communicator_ReduceScatter,
    communicator_all_gather: PJRT_Communicator_AllGather,
    communicator_collective_permute: PJRT_Communicator_CollectivePermute,
    communicator_all_to_all: PJRT_Communicator_AllToAll,
    communicator_to_string: PJRT_Communicator_ToString,
};

/// Static Node 4 (`PJRT_MultiSlice_Extension`, `ext_type = 22`, `64` bytes).
pub static EXT_04_MULTI_SLICE: PJRT_MultiSlice_Extension = PJRT_MultiSlice_Extension {
    base: PJRT_Extension_Base {
        struct_size: 64,
        ext_type: 22,
        pad: 0,
        next: ptr::addr_of!(EXT_05_COLLECTIVES.base),
    },
    config_destroy: PJRT_MultiSlice_Config_Destroy,
    config_num_slices: PJRT_MultiSlice_Config_NumSlices,
    config_num_devices_per_slice: PJRT_MultiSlice_Config_NumDevicesPerSlice,
    config_slice_id: PJRT_MultiSlice_Config_SliceId,
    config_serialize: PJRT_MultiSlice_Config_Serialize,
};

/// Static Node 3 (`PJRT_HostMemoryAllocator_Extension`, `ext_type = 23`, `32` bytes).
pub static EXT_03_HOST_ALLOCATOR: PJRT_HostMemoryAllocator_Extension =
    PJRT_HostMemoryAllocator_Extension {
        base: PJRT_Extension_Base {
            struct_size: 32,
            ext_type: 23,
            pad: 0,
            next: ptr::addr_of!(EXT_04_MULTI_SLICE.base),
        },
        get_host_memory_allocator: PJRT_HostMemoryAllocator_Get,
    };

/// Static Node 2 (`PJRT_XlaTransform_Extension`, `ext_type = 24`, `56` bytes).
pub static EXT_02_XLA_TRANSFORM: PJRT_XlaTransform_Extension = PJRT_XlaTransform_Extension {
    base: PJRT_Extension_Base {
        struct_size: 56,
        ext_type: 24,
        pad: 0,
        next: ptr::addr_of!(EXT_03_HOST_ALLOCATOR.base),
    },
    register_xla_transform: PJRT_XlaTransform_Register,
    wait_for_all_transforms: PJRT_XlaTransform_WaitForAll,
    unregister_xla_transform: PJRT_XlaTransform_Unregister,
    list_xla_transforms: PJRT_XlaTransform_List,
};

/// Static Node 1 (`PJRT_FFI_Extension`, `ext_type = 5`, `48` bytes), head of the 18-node TPU extension chain.
pub static EXT_01_FFI: PJRT_FFI_Extension = PJRT_FFI_Extension {
    base: PJRT_Extension_Base {
        struct_size: 48,
        ext_type: 5,
        pad: 0,
        next: ptr::addr_of!(EXT_02_XLA_TRANSFORM.base),
    },
    type_id_register: PJRT_FFI_TypeID_Register,
    user_data_add: PJRT_FFI_UserData_Add,
    register_handler: PJRT_FFI_Register_Handler,
};

#[cfg(test)]
mod tests;
