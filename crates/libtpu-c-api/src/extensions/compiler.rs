//! PJRT XLA Transform, Callback, PhaseCompile, Triton, and `OpenXLA` FFI entrypoints (`0.0.53.1`).

use super::args::*;
use super::memory::build_layout_handles_for_shapes;
use super::{
    alloc_phase_compile_buffer, dispatch_typed_extension_args,
    dispatch_typed_fallible_extension_args, free_phase_compile_buffer,
};
use crate::pjrt_helpers::{
    borrow_live_handle, borrow_live_handle_mut, box_registered_handle, box_serialized_artifact,
    ffi_catch_unwind, pjrt_serialized_artifact_deleter, take_live_handle,
};
use crate::pjrt_types::{
    PJRT_Error, PJRT_Executable, PJRT_PhaseCompile_Compiler, PJRT_Relocatable,
};
use libtpu_core::compiler::{
    COMPILATION_PHASE_NAMES, CompiledProgram, global_xla_transform_registry,
};
use libtpu_core::error::LibTpuError;
use libtpu_core::pjrt::PjrtCoreExecutable;
use libtpu_core::proto::ProtoSchemaMessage;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

/// Monotonic instance counter for `PJRT_PhaseCompile_Compiler`.
static PHASE_COMPILER_SEQ: AtomicUsize = AtomicUsize::new(0);
/// Monotonic type-ID sequence counter for `PJRT_FFI_TypeID_Register`.
static FFI_TYPE_ID_SEQ: AtomicI64 = AtomicI64::new(0);

/// Static array of byte lengths for [`COMPILATION_PHASE_NAMES`].
static STATIC_PHASE_NAME_SIZES: [usize; 5] = [
    COMPILATION_PHASE_NAMES[0].len(),
    COMPILATION_PHASE_NAMES[1].len(),
    COMPILATION_PHASE_NAMES[2].len(),
    COMPILATION_PHASE_NAMES[3].len(),
    COMPILATION_PHASE_NAMES[4].len(),
];

/// Static null-terminated C-string pointers for [`COMPILATION_PHASE_NAMES`].
#[repr(transparent)]
struct StaticPhaseNamePtrs([*const c_char; 5]);

// SAFETY: Contains only immutable static `'static` C-string pointers.
unsafe impl Sync for StaticPhaseNamePtrs {}

/// Static array of phase name C-string pointers.
static STATIC_PHASE_NAME_PTRS: StaticPhaseNamePtrs = StaticPhaseNamePtrs([
    c"hlo_verification".as_ptr(),
    c"tpu_layout_assignment".as_ptr(),
    c"tpu_memory_planning".as_ptr(),
    c"vliw_bundle_emission".as_ptr(),
    c"binary_serialization".as_ptr(),
]);

/// Owned snapshot of registered XLA transform names exposed via [`PJRT_XlaTransform_List`].
#[derive(Default)]
struct XlaTransformListSnapshot {
    /// Owned null-terminated C strings.
    _cstrings: Vec<CString>,
    /// Pointer array into `_cstrings`.
    name_ptrs: Box<[*const c_char]>,
    /// Byte length array matching `_cstrings`.
    name_sizes: Box<[usize]>,
}

// SAFETY: Heap buffers inside `XlaTransformListSnapshot` are immutable once constructed.
unsafe impl Send for XlaTransformListSnapshot {}
// SAFETY: Heap buffers inside `XlaTransformListSnapshot` are immutable once constructed.
unsafe impl Sync for XlaTransformListSnapshot {}

/// Arena retaining [`XlaTransformListSnapshot`]s so C callers can read `names_out` / `name_sizes_out`.
static XLA_TRANSFORM_LIST_ARENA: OnceLock<parking_lot::Mutex<Vec<XlaTransformListSnapshot>>> =
    OnceLock::new();

/// Returns a reference to [`XLA_TRANSFORM_LIST_ARENA`].
fn xla_transform_list_arena() -> &'static parking_lot::Mutex<Vec<XlaTransformListSnapshot>> {
    XLA_TRANSFORM_LIST_ARENA.get_or_init(|| parking_lot::Mutex::new(Vec::new()))
}

/// Typed runtime callback entry registered via [`PJRT_Callback_RegisterCallback`].
#[derive(Clone, Copy)]
struct RuntimeCallbackEntry {
    /// Typed C callback function pointer.
    callback: PjrtRuntimeCallbackFn,
    /// Opaque caller user argument (`usize` address).
    user_arg_addr: usize,
}

/// Process-wide map of `(client_addr, callback_type) -> RuntimeCallbackEntry`.
static CLIENT_RUNTIME_CALLBACKS: OnceLock<
    parking_lot::Mutex<BTreeMap<(usize, i64), RuntimeCallbackEntry>>,
> = OnceLock::new();

/// Returns a reference to [`CLIENT_RUNTIME_CALLBACKS`].
fn client_runtime_callbacks()
-> &'static parking_lot::Mutex<BTreeMap<(usize, i64), RuntimeCallbackEntry>> {
    CLIENT_RUNTIME_CALLBACKS.get_or_init(|| parking_lot::Mutex::new(BTreeMap::new()))
}

/// Extracts a trimmed UTF-8 string slice from `(ptr, len)` or returns [`LibTpuError::InvalidArgument`].
///
/// # Arguments
///
/// * `ptr` - UTF-8 byte pointer.
/// * `len` - Byte length of `ptr`.
/// * `parameter` - Parameter name for error diagnostics.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `ptr` is null, `len == 0`, or invalid UTF-8.
///
/// # Safety
///
/// If `ptr` is non-null, it must reference at least `len` readable bytes.
unsafe fn parse_non_empty_c_utf8<'a>(
    ptr: *const c_char,
    len: usize,
    parameter: &'static str,
) -> Result<&'a str, LibTpuError> {
    if ptr.is_null() || len == 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter,
            reason: format!("{parameter} cannot be null or empty"),
        });
    }
    // SAFETY: `ptr` is non-null and caller guarantees `len` readable bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|e| LibTpuError::InvalidArgument {
        parameter,
        reason: format!("invalid UTF-8 in {parameter}: {e}"),
    })
}

/// Strongly-typed handler for [`PJRT_XlaTransform_Register`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_XlaTransform_Register_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.name` is null or empty.
///
/// # Safety
///
/// If `args.name` is non-null, it must reference `args.name_size` readable bytes.
unsafe fn handle_xla_transform_register(
    args: &mut PJRT_XlaTransform_Register_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.name` validated against `args.name_size`.
    let pass_name = unsafe { parse_non_empty_c_utf8(args.name, args.name_size, "name") }?;
    let registry = global_xla_transform_registry();
    registry.register_transform(pass_name, args.transform_fn, None)?;
    args.transform_fn = registry.len();
    Ok(())
}

/// Registers an XLA compiler transform pass (`PJRT_XlaTransform_Register`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_XlaTransform_Register_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_XlaTransform_Register_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_XlaTransform_Register(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_XlaTransform_Register_Args>(
            raw_args,
            handle_xla_transform_register,
        )
    }
}

/// Strongly-typed handler for [`PJRT_XlaTransform_WaitForAll`].
///
/// # Arguments
///
/// * `_args` - Validated [`PJRT_XlaTransform_WaitForAll_Args`] structure.
///
/// # Safety
///
/// `_args` must be a valid [`PJRT_XlaTransform_WaitForAll_Args`].
unsafe fn handle_xla_transform_wait_for_all(_args: &mut PJRT_XlaTransform_WaitForAll_Args) {
    let _ = global_xla_transform_registry().len();
}

/// Waits for all active XLA transforms (`PJRT_XlaTransform_WaitForAll`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_XlaTransform_WaitForAll_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_XlaTransform_WaitForAll_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_XlaTransform_WaitForAll(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_XlaTransform_WaitForAll_Args>(
            raw_args,
            handle_xla_transform_wait_for_all,
        )
    }
}

/// Strongly-typed handler for [`PJRT_XlaTransform_Unregister`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_XlaTransform_Unregister_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.name` is null, empty, or not registered.
///
/// # Safety
///
/// If `args.name` is non-null, it must reference `args.name_size` readable bytes.
unsafe fn handle_xla_transform_unregister(
    args: &mut PJRT_XlaTransform_Unregister_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.name` validated against `args.name_size`.
    let pass_name = unsafe { parse_non_empty_c_utf8(args.name, args.name_size, "name") }?;
    global_xla_transform_registry().unregister_transform(pass_name)
}

/// Unregisters an XLA compiler transform pass (`PJRT_XlaTransform_Unregister`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_XlaTransform_Unregister_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_XlaTransform_Unregister_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_XlaTransform_Unregister(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_XlaTransform_Unregister_Args>(
            raw_args,
            handle_xla_transform_unregister,
        )
    }
}

/// Lists registered XLA compiler transform passes (`PJRT_XlaTransform_List`), writing owned
/// C-string pointers into `names_out`, `num_names_out`, and optional `name_sizes_out` (`struct_size >= 40`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_XlaTransform_List_Args`] (`32` or `40` bytes).
///
/// # Safety
///
/// `raw_args` must be null or point to at least 32 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_XlaTransform_List(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        let expected_min = PJRT_XlaTransform_List_Args::MIN_STRUCT_SIZE;
        if actual_size < expected_min {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_XlaTransform_List_Args",
                expected_min,
                actual: actual_size,
            });
        }
        let names = global_xla_transform_registry().list_transforms();
        let mut cstrings = Vec::with_capacity(names.len());
        let mut sizes = Vec::with_capacity(names.len());
        for n in &names {
            sizes.push(n.len());
            cstrings.push(crate::pjrt_helpers::to_c_string(n));
        }
        let ptrs: Vec<*const c_char> = cstrings.iter().map(|cs| cs.as_ptr()).collect();
        let snapshot = XlaTransformListSnapshot {
            _cstrings: cstrings,
            name_ptrs: ptrs.into_boxed_slice(),
            name_sizes: sizes.into_boxed_slice(),
        };
        let names_ptr = snapshot.name_ptrs.as_ptr();
        let sizes_ptr = snapshot.name_sizes.as_ptr();
        let count = snapshot.name_ptrs.len();
        let mut arena = xla_transform_list_arena().lock();
        if arena.len() >= 64 {
            arena.drain(..32);
        }
        arena.push(snapshot);
        drop(arena);
        // SAFETY: `actual_size >= 32` covers `names_out` (`+16`) and `num_names_out` (`+24`); `+32` written if `>= 40`.
        unsafe {
            *raw_args.byte_add(16).cast::<*const *const c_char>() = names_ptr;
            *raw_args.byte_add(24).cast::<usize>() = count;
            if actual_size >= 40 {
                *raw_args.byte_add(32).cast::<*const usize>() = sizes_ptr;
            }
        }
        ptr::null_mut()
    })
}

/// Strongly-typed handler for [`PJRT_Callback_RegisterCallback`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Callback_Register_Args`] structure.
///
/// # Safety
///
/// If `args.client` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Client`].
unsafe fn handle_callback_register(args: &mut PJRT_Callback_Register_Args) {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    if let Some(client) = unsafe { args.client.as_ref() } {
        let client_addr = ptr::from_ref(client) as usize;
        if let Some(cb_fn) = args.callback_fn {
            client
                .ext_arena
                .lock()
                .callback_registry
                .insert(args.callback_type, (cb_fn as usize, args.user_arg as usize));
            client_runtime_callbacks().lock().insert(
                (client_addr, args.callback_type),
                RuntimeCallbackEntry {
                    callback: cb_fn,
                    user_arg_addr: args.user_arg as usize,
                },
            );
        } else {
            client
                .ext_arena
                .lock()
                .callback_registry
                .remove(&args.callback_type);
            client_runtime_callbacks()
                .lock()
                .remove(&(client_addr, args.callback_type));
        }
    }
}

/// Registers a runtime callback (`PJRT_Callback_RegisterCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Callback_Register_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Callback_Register_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Callback_RegisterCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Callback_Register_Args>(
            raw_args,
            handle_callback_register,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Callback_InvokeCallback`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Callback_Invoke_Args`] structure.
///
/// # Safety
///
/// If `args.client` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Client`].
unsafe fn handle_callback_invoke(args: &mut PJRT_Callback_Invoke_Args) {
    let cb_key = i64::try_from(args.callback_type).unwrap_or(0);
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let entry = unsafe { args.client.as_ref() }.and_then(|client| {
        client_runtime_callbacks()
            .lock()
            .get(&(ptr::from_ref(client) as usize, cb_key))
            .copied()
    });
    if let Some(cb_entry) = entry {
        // SAFETY: Invoking caller's registered C callback with `payload` and `user_arg`.
        unsafe {
            (cb_entry.callback)(
                args.payload,
                cb_entry.user_arg_addr as *mut std::ffi::c_void,
            );
        }
    }
}

/// Invokes a registered runtime callback (`PJRT_Callback_InvokeCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Callback_Invoke_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Callback_Invoke_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Callback_InvokeCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Callback_Invoke_Args>(raw_args, handle_callback_invoke)
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_GetCompiler`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_GetCompiler_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_PhaseCompile_GetCompiler_Args`].
unsafe fn handle_phase_compile_get_compiler(args: &mut PJRT_PhaseCompile_GetCompiler_Args) {
    let seq = PHASE_COMPILER_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    args.compiler_out = box_registered_handle(PJRT_PhaseCompile_Compiler {
        compiler_id: u64::try_from(seq).unwrap_or(1),
        phases_executed: 0,
        owned_buffers: parking_lot::Mutex::new(Vec::new()),
    });
}

/// Acquires a `PJRT_PhaseCompiler` instance (`PJRT_PhaseCompile_GetCompiler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_GetCompiler_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_GetCompiler_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_GetCompiler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_PhaseCompile_GetCompiler_Args>(
            raw_args,
            handle_phase_compile_get_compiler,
        )
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_DestroyCompiler`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_PhaseCompile_Destroy_Args`].
unsafe fn handle_phase_compile_destroy_compiler(args: &mut PJRT_PhaseCompile_Destroy_Args) {
    drop(take_live_handle(
        args.handle.cast::<PJRT_PhaseCompile_Compiler>(),
    ));
    args.handle = ptr::null_mut();
}

/// Destroys a `PJRT_PhaseCompiler` instance (`PJRT_PhaseCompile_DestroyCompiler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_DestroyCompiler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_PhaseCompile_Destroy_Args>(
            raw_args,
            handle_phase_compile_destroy_compiler,
        )
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_RunPhase`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_RunPhase_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_PhaseCompile_RunPhase_Args`].
unsafe fn handle_phase_compile_run_phase(args: &mut PJRT_PhaseCompile_RunPhase_Args) {
    let input = if !args.input_bytes.is_null() {
        // SAFETY: `input_bytes` is non-null with `input_size` bytes.
        unsafe { std::slice::from_raw_parts(args.input_bytes.cast::<u8>(), args.input_size) }
    } else {
        b"module @phase { stablehlo.add }"
    };
    let phase_name = if !args.phase_name.is_null() {
        // SAFETY: `phase_name` is a non-null C-string pointer.
        unsafe { std::ffi::CStr::from_ptr(args.phase_name) }
            .to_str()
            .unwrap_or(COMPILATION_PHASE_NAMES[0])
    } else {
        COMPILATION_PHASE_NAMES[0]
    };
    args.output_status = CompiledProgram::run_compilation_phase_proto(phase_name, input, b"")
        .map_or(COMPILATION_PHASE_NAMES.len(), |proto| {
            let v = proto.encode_proto();
            let len = v.len().max(COMPILATION_PHASE_NAMES.len());
            if let Some(comp) = borrow_live_handle_mut(args.phase_compiler) {
                comp.phases_executed = comp.phases_executed.saturating_add(1);
                comp.owned_buffers.lock().push(v.into_boxed_slice());
            } else {
                let (buf_handle, _, _) = alloc_phase_compile_buffer(v);
                free_phase_compile_buffer(buf_handle);
            }
            len
        });
}

/// Executes a named TPU compilation phase (`PJRT_PhaseCompile_RunPhase`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_RunPhase_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_RunPhase_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_RunPhase(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_PhaseCompile_RunPhase_Args>(
            raw_args,
            handle_phase_compile_run_phase,
        )
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_GetPhaseNames`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_GetPhaseNames_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_PhaseCompile_GetPhaseNames_Args`].
unsafe fn handle_phase_compile_get_phase_names(args: &mut PJRT_PhaseCompile_GetPhaseNames_Args) {
    args.phase_names = STATIC_PHASE_NAME_PTRS.0.as_ptr();
    args.phase_names_sizes = STATIC_PHASE_NAME_SIZES.as_ptr();
    args.num_phase_names = COMPILATION_PHASE_NAMES.len();
}

/// Queries the list of supported compilation phase names (`PJRT_PhaseCompile_GetPhaseNames`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_GetPhaseNames_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_GetPhaseNames_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_GetPhaseNames(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_PhaseCompile_GetPhaseNames_Args>(
            raw_args,
            handle_phase_compile_get_phase_names,
        )
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_CBuffersDestroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_PhaseCompile_Destroy_Args`].
unsafe fn handle_phase_compile_c_buffers_destroy(args: &mut PJRT_PhaseCompile_Destroy_Args) {
    free_phase_compile_buffer(args.handle);
    args.handle = ptr::null_mut();
}

/// Frees C buffers returned by `PJRT_PhaseCompile_Extension` (`PJRT_PhaseCompile_CBuffersDestroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_CBuffersDestroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_PhaseCompile_Destroy_Args>(
            raw_args,
            handle_phase_compile_c_buffers_destroy,
        )
    }
}

/// Constructs a registered [`PJRT_Relocatable`] handle from a [`CompiledProgram`].
///
/// # Arguments
///
/// * `program` - Lowered [`CompiledProgram`] to package as a relocatable object.
fn build_relocatable_from_program(program: &CompiledProgram) -> *mut PJRT_Relocatable {
    let mut param_types = Vec::with_capacity(program.parameter_shapes.len());
    let mut param_dims = Vec::new();
    let mut param_ranks = Vec::with_capacity(program.parameter_shapes.len());
    for s in &program.parameter_shapes {
        param_types.push(i32::from(s.element_type));
        param_ranks.push(s.dims.len());
        param_dims.extend_from_slice(&s.dims);
    }
    let mut out_types = Vec::with_capacity(program.output_shapes.len());
    let mut out_dims = Vec::new();
    let mut out_ranks = Vec::with_capacity(program.output_shapes.len());
    for s in &program.output_shapes {
        out_types.push(i32::from(s.element_type));
        out_ranks.push(s.dims.len());
        out_dims.extend_from_slice(&s.dims);
    }
    let param_layouts = build_layout_handles_for_shapes(&program.parameter_shapes);
    let output_layouts = build_layout_handles_for_shapes(&program.output_shapes);
    let aliases = vec![-1_i64; param_types.len()];
    let mod_name = program.name.as_bytes().to_vec();
    let build_id = format!("libtpu-0.0.53.1-{}", program.fingerprint).into_bytes();
    let backend_cfg = format!("{{\"opcode\":\"{:?}\"}}", program.opcode).into_bytes();
    box_registered_handle(PJRT_Relocatable {
        name: mod_name,
        build_id,
        call_backend_config: backend_cfg,
        has_side_effects: false,
        param_element_types: param_types,
        output_element_types: out_types,
        param_dims_flat: param_dims,
        param_dim_sizes: param_ranks,
        output_dims_flat: out_dims,
        output_dim_sizes: out_ranks,
        param_layout_ptrs: param_layouts,
        output_layout_ptrs: output_layouts,
        param_output_aliases: aliases,
        serialized_bytes: program.serialize(),
        program: program.clone(),
    })
}

/// Strongly-typed handler for [`PJRT_Relocatable_Destroy`] (`0.0.53.1`).
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_Destroy_Args`].
unsafe fn handle_relocatable_destroy(args: &mut PJRT_Relocatable_Destroy_Args) {
    drop(take_live_handle(args.relocatable));
    args.relocatable = ptr::null_mut();
}

/// Destroys a `PJRT_Relocatable` handle (`PJRT_Relocatable_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Relocatable_Destroy_Args>(
            raw_args,
            handle_relocatable_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_Name`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_StringQuery_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_StringQuery_Args`].
unsafe fn handle_relocatable_name(
    args: &mut PJRT_Relocatable_StringQuery_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.value_ptr = rel.name.as_ptr().cast::<c_char>();
    args.value_size = rel.name.len();
    Ok(())
}

/// Queries the module name of a `PJRT_Relocatable` (`PJRT_Relocatable_Name`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_StringQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_StringQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_Name(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_StringQuery_Args>(
            raw_args,
            handle_relocatable_name,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_BuildId`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_StringQuery_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_StringQuery_Args`].
unsafe fn handle_relocatable_build_id(
    args: &mut PJRT_Relocatable_StringQuery_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.value_ptr = rel.build_id.as_ptr().cast::<c_char>();
    args.value_size = rel.build_id.len();
    Ok(())
}

/// Queries the build identifier of a `PJRT_Relocatable` (`PJRT_Relocatable_BuildId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_StringQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_StringQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_BuildId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_StringQuery_Args>(
            raw_args,
            handle_relocatable_build_id,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_CallBackendConfig`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_StringQuery_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_StringQuery_Args`].
unsafe fn handle_relocatable_backend_config(
    args: &mut PJRT_Relocatable_StringQuery_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.value_ptr = rel.call_backend_config.as_ptr().cast::<c_char>();
    args.value_size = rel.call_backend_config.len();
    Ok(())
}

/// Queries the backend call configuration of a `PJRT_Relocatable` (`PJRT_Relocatable_CallBackendConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_StringQuery_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_StringQuery_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_CallBackendConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_StringQuery_Args>(
            raw_args,
            handle_relocatable_backend_config,
        )
    }
}

/// Queries whether a `PJRT_Relocatable` has side effects (`PJRT_Relocatable_HasSideEffects`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_HasSideEffects_Args`] (`>= 25` bytes).
///
/// # Safety
///
/// `raw_args` must be null or point to at least 25 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_HasSideEffects(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        if actual_size < PJRT_Relocatable_HasSideEffects_Args::MIN_STRUCT_SIZE {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_Relocatable_HasSideEffects_Args",
                expected_min: PJRT_Relocatable_HasSideEffects_Args::MIN_STRUCT_SIZE,
                actual: actual_size,
            });
        }
        // SAFETY: `actual_size >= 25` covers `relocatable` (`+16`) and `has_side_effects` (`+24`).
        let rel_ptr = unsafe { *raw_args.byte_add(16).cast::<*const PJRT_Relocatable>() };
        let Some(rel) = borrow_live_handle(rel_ptr) else {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidArgument {
                parameter: "relocatable",
                reason: String::from("null or unregistered PJRT_Relocatable handle"),
            });
        };
        // SAFETY: Offset `+24` is `bool has_side_effects`.
        unsafe {
            *raw_args.byte_add(24).cast::<bool>() = rel.has_side_effects;
        }
        ptr::null_mut()
    })
}

/// Strongly-typed handler for [`PJRT_Relocatable_Serialize`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_Serialize_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_Serialize_Args`].
unsafe fn handle_relocatable_serialize(
    args: &mut PJRT_Relocatable_Serialize_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    let (ptr, len, holder) = box_serialized_artifact(rel.serialized_bytes.clone());
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_relocatable = holder.cast::<std::ffi::c_void>();
    args.serialized_relocatable_deleter = Some(pjrt_serialized_artifact_deleter);
    Ok(())
}

/// Serializes a `PJRT_Relocatable` module (`PJRT_Relocatable_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_Serialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_Serialize_Args>(
            raw_args,
            handle_relocatable_serialize,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_ParameterElementTypes`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_ElementTypes_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_ElementTypes_Args`].
unsafe fn handle_relocatable_param_types(
    args: &mut PJRT_Relocatable_ElementTypes_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.element_types = rel.param_element_types.as_ptr();
    args.num_types = rel.param_element_types.len();
    Ok(())
}

/// Queries parameter element types of a `PJRT_Relocatable` (`PJRT_Relocatable_ParameterElementTypes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_ElementTypes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_ElementTypes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_ParameterElementTypes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_ElementTypes_Args>(
            raw_args,
            handle_relocatable_param_types,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_OutputElementTypes`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_ElementTypes_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_ElementTypes_Args`].
unsafe fn handle_relocatable_output_types(
    args: &mut PJRT_Relocatable_ElementTypes_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.element_types = rel.output_element_types.as_ptr();
    args.num_types = rel.output_element_types.len();
    Ok(())
}

/// Queries output element types of a `PJRT_Relocatable` (`PJRT_Relocatable_OutputElementTypes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_ElementTypes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_ElementTypes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_OutputElementTypes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_ElementTypes_Args>(
            raw_args,
            handle_relocatable_output_types,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_ParameterDimensions`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_Dimensions_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_Dimensions_Args`].
unsafe fn handle_relocatable_param_dims(
    args: &mut PJRT_Relocatable_Dimensions_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.num_items = rel.param_dim_sizes.len();
    args.dims = rel.param_dims_flat.as_ptr();
    args.dim_sizes = rel.param_dim_sizes.as_ptr();
    Ok(())
}

/// Queries parameter dimensions of a `PJRT_Relocatable` (`PJRT_Relocatable_ParameterDimensions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_Dimensions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_Dimensions_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_ParameterDimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_Dimensions_Args>(
            raw_args,
            handle_relocatable_param_dims,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_OutputDimensions`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_Dimensions_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_Dimensions_Args`].
unsafe fn handle_relocatable_output_dims(
    args: &mut PJRT_Relocatable_Dimensions_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.num_items = rel.output_dim_sizes.len();
    args.dims = rel.output_dims_flat.as_ptr();
    args.dim_sizes = rel.output_dim_sizes.as_ptr();
    Ok(())
}

/// Queries output dimensions of a `PJRT_Relocatable` (`PJRT_Relocatable_OutputDimensions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_Dimensions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_Dimensions_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_OutputDimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_Dimensions_Args>(
            raw_args,
            handle_relocatable_output_dims,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_ParameterLayouts`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_Layouts_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_Layouts_Args`].
unsafe fn handle_relocatable_param_layouts(
    args: &mut PJRT_Relocatable_Layouts_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.num_layouts = rel.param_layout_ptrs.len();
    args.layouts = rel.param_layout_ptrs.as_ptr();
    Ok(())
}

/// Queries parameter layouts of a `PJRT_Relocatable` (`PJRT_Relocatable_ParameterLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_Layouts_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_Layouts_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_ParameterLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_Layouts_Args>(
            raw_args,
            handle_relocatable_param_layouts,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_OutputLayouts`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_Layouts_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_Layouts_Args`].
unsafe fn handle_relocatable_output_layouts(
    args: &mut PJRT_Relocatable_Layouts_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.num_layouts = rel.output_layout_ptrs.len();
    args.layouts = rel.output_layout_ptrs.as_ptr();
    Ok(())
}

/// Queries output layouts of a `PJRT_Relocatable` (`PJRT_Relocatable_OutputLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_Layouts_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_Layouts_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_OutputLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_Layouts_Args>(
            raw_args,
            handle_relocatable_output_layouts,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Relocatable_ParameterOutputAliases`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Relocatable_ParameterOutputAliases_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.relocatable` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Relocatable_ParameterOutputAliases_Args`].
unsafe fn handle_relocatable_aliases(
    args: &mut PJRT_Relocatable_ParameterOutputAliases_Args,
) -> Result<(), LibTpuError> {
    let rel = borrow_live_handle(args.relocatable).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatable",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    args.num_parameters = rel.param_output_aliases.len();
    args.parameter_output_aliases = rel.param_output_aliases.as_ptr();
    Ok(())
}

/// Queries parameter-to-output aliases of a `PJRT_Relocatable` (`PJRT_Relocatable_ParameterOutputAliases`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Relocatable_ParameterOutputAliases_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Relocatable_ParameterOutputAliases_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Relocatable_ParameterOutputAliases(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Relocatable_ParameterOutputAliases_Args>(
            raw_args,
            handle_relocatable_aliases,
        )
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_CompileToRelocatable`] (`0.0.53.1`).
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_CompileToRelocatable_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `args.program` is null or compilation fails.
///
/// # Safety
///
/// `args.program` must be null or point to a valid [`crate::pjrt_types::PJRT_Program`].
unsafe fn handle_compile_to_relocatable(
    args: &mut PJRT_PhaseCompile_CompileToRelocatable_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.program` is null or a valid `PJRT_Program*`.
    let prog = unsafe { args.program.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "program",
    })?;
    if prog.code.is_null() || prog.code_size == 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "program.code",
            reason: String::from("relocatable compilation requires non-empty program code"),
        });
    }
    // SAFETY: `prog.code` is non-null with `prog.code_size` bytes.
    let code = unsafe { std::slice::from_raw_parts(prog.code.cast::<u8>(), prog.code_size) };
    let fmt = if !prog.format.is_null() && prog.format_size > 0 {
        // SAFETY: `prog.format` is non-null with `prog.format_size` bytes.
        let fmt_bytes =
            unsafe { std::slice::from_raw_parts(prog.format.cast::<u8>(), prog.format_size) };
        std::str::from_utf8(fmt_bytes).unwrap_or("mlir")
    } else {
        "mlir"
    };
    let opts = if !args.compile_options.is_null() && args.compile_options_size > 0 {
        // SAFETY: `compile_options` is non-null with `compile_options_size` bytes.
        unsafe {
            std::slice::from_raw_parts(args.compile_options.cast::<u8>(), args.compile_options_size)
        }
    } else {
        &[]
    };
    let compiled = CompiledProgram::compile(fmt, code, opts)?;
    let rel_ptr = build_relocatable_from_program(&compiled);
    let (buf_handle, rel_arr_ptr, rel_count) =
        super::alloc_phase_compile_relocatable_array(vec![rel_ptr]);
    args.relocatables = rel_arr_ptr;
    args.num_relocatables = rel_count;
    args.relocatables_buffer = buf_handle;
    Ok(())
}

/// Compiles a `PJRT_Program` into relocatable object modules (`PJRT_PhaseCompile_CompileToRelocatable`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_CompileToRelocatable_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_CompileToRelocatable_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_CompileToRelocatable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_PhaseCompile_CompileToRelocatable_Args>(
            raw_args,
            handle_compile_to_relocatable,
        )
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_DeserializeRelocatable`] (`0.0.53.1`).
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_DeserializeRelocatable_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if serialized bytes are null/empty or corrupt.
///
/// # Safety
///
/// If `args.serialized_relocatable` is non-null, it must reference `args.serialized_relocatable_size` readable bytes.
unsafe fn handle_deserialize_relocatable(
    args: &mut PJRT_PhaseCompile_DeserializeRelocatable_Args,
) -> Result<(), LibTpuError> {
    if args.serialized_relocatable.is_null() || args.serialized_relocatable_size == 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "serialized_relocatable",
            reason: String::from("serialized_relocatable cannot be null or empty"),
        });
    }
    // SAFETY: `serialized_relocatable` is non-null with `serialized_relocatable_size` bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            args.serialized_relocatable.cast::<u8>(),
            args.serialized_relocatable_size,
        )
    };
    let compiled = CompiledProgram::deserialize(bytes)?;
    args.relocatable = build_relocatable_from_program(&compiled);
    Ok(())
}

/// Deserializes a `PJRT_Relocatable` module (`PJRT_PhaseCompile_DeserializeRelocatable`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_DeserializeRelocatable_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_DeserializeRelocatable_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_DeserializeRelocatable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_PhaseCompile_DeserializeRelocatable_Args>(
            raw_args,
            handle_deserialize_relocatable,
        )
    }
}

/// Strongly-typed handler for [`PJRT_PhaseCompile_LinkRelocatables`] (`0.0.53.1`).
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_PhaseCompile_LinkRelocatables_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `relocatables` is empty/null or contains unregistered handles.
///
/// # Safety
///
/// If `args.relocatables` is non-null, it must reference `args.num_relocatables` pointer slots.
unsafe fn handle_link_relocatables(
    args: &mut PJRT_PhaseCompile_LinkRelocatables_Args,
) -> Result<(), LibTpuError> {
    if args.relocatables.is_null() || args.num_relocatables == 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "relocatables",
            reason: String::from("at least one PJRT_Relocatable is required for linking"),
        });
    }
    // SAFETY: `args.relocatables` is non-null with `args.num_relocatables >= 1` entries.
    let rel_ptrs = unsafe { std::slice::from_raw_parts(args.relocatables, args.num_relocatables) };
    let mut rel_iter = rel_ptrs.iter().copied();
    let first_ptr = rel_iter.next().unwrap_or(ptr::null_mut());
    let first_rel = borrow_live_handle(first_ptr).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "relocatables[0]",
        reason: String::from("null or unregistered PJRT_Relocatable handle"),
    })?;
    for p in rel_iter {
        if borrow_live_handle(p).is_none() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "relocatables",
                reason: String::from("unregistered PJRT_Relocatable handle in link list"),
            });
        }
    }
    args.executable = Box::into_raw(Box::new(PJRT_Executable::new(PjrtCoreExecutable::new(
        first_rel.program.clone(),
    ))));
    Ok(())
}

/// Links one or more `PJRT_Relocatable` modules into a `PJRT_Executable` (`PJRT_PhaseCompile_LinkRelocatables`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_PhaseCompile_LinkRelocatables_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_PhaseCompile_LinkRelocatables_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_PhaseCompile_LinkRelocatables(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_PhaseCompile_LinkRelocatables_Args>(
            raw_args,
            handle_link_relocatables,
        )
    }
}

/// Validates and compiles a Triton MLIR kernel into VLIW/executable wire bytes (`PJRT_Triton_Compile`).
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Triton_Compile_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] when `module` is null/empty or [`LibTpuError::CompilationFailed`] on invalid MLIR.
///
/// # Safety
///
/// If `args.module` is non-null, it must reference `args.module_size` readable bytes.
unsafe fn handle_triton_compile(args: &mut PJRT_Triton_Compile_Args) -> Result<(), LibTpuError> {
    if args.module.is_null() || args.module_size == 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "module",
            reason: String::from("Triton TTIR MLIR module cannot be null or empty"),
        });
    }
    // SAFETY: `args.module` is non-null with `args.module_size` bytes.
    let mlir_bytes =
        unsafe { std::slice::from_raw_parts(args.module.cast::<u8>(), args.module_size) };
    let text = std::str::from_utf8(mlir_bytes).map_err(|e| LibTpuError::CompilationFailed {
        reason: format!("Triton MLIR must be valid UTF-8: {e}"),
    })?;
    let trimmed = text.trim();
    let open_braces = trimmed.bytes().filter(|&b| b == b'{').count();
    let close_braces = trimmed.bytes().filter(|&b| b == b'}').count();
    let has_mlir_root = trimmed.contains("module")
        || trimmed.contains("tt.func")
        || trimmed.contains("func.func")
        || trimmed.contains("stablehlo.");
    if !has_mlir_root || open_braces == 0 || open_braces != close_braces {
        return Err(LibTpuError::CompilationFailed {
            reason: String::from(
                "Invalid Triton MLIR syntax: missing module/func or unbalanced braces",
            ),
        });
    }
    let compiled = CompiledProgram::compile("mlir", mlir_bytes, b"")?;
    let wire_bytes = compiled.to_tpu_executable_proto().encode_proto();
    let boxed = wire_bytes.into_boxed_slice();
    let asm_len = boxed.len();
    let raw_slice = Box::into_raw(boxed);
    args.out_asm = raw_slice.cast::<c_char>();
    args.out_asm_size = asm_len;
    let warps = i64::from(args.num_warps.max(1));
    let stages = i64::from(args.num_stages.max(1));
    let temp_smem = compiled.memory_stats.temp_size_in_bytes.max(0);
    args.out_smem_bytes = (warps * stages * 1024).max(4096).saturating_add(temp_smem);
    Ok(())
}

/// Compiles a Triton MLIR kernel for TPU Mosaic/VLIW execution (`PJRT_Triton_Compile`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Triton_Compile_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Triton_Compile_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Triton_Compile(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Triton_Compile_Args>(
            raw_args,
            handle_triton_compile,
        )
    }
}

/// Frees a heap-allocated compiled Triton assembly payload (`PJRT_Triton_FreeAsm`).
///
/// # Arguments
///
/// * `asm_ptr` - Pointer returned in `PJRT_Triton_Compile_Args::out_asm`.
/// * `asm_size` - Byte length returned in `PJRT_Triton_Compile_Args::out_asm_size`.
///
/// # Safety
///
/// `asm_ptr` must be null or a `(out_asm, out_asm_size)` pair returned by [`PJRT_Triton_Compile`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Triton_FreeAsm(asm_ptr: *const c_char, asm_size: usize) {
    ffi_catch_unwind((), || {
        if !asm_ptr.is_null() && asm_size > 0 {
            let raw_slice =
                ptr::slice_from_raw_parts_mut(asm_ptr.cast_mut().cast::<u8>(), asm_size);
            // SAFETY: `raw_slice` was allocated via `Box::into_raw` in `handle_triton_compile`.
            unsafe {
                drop(Box::from_raw(raw_slice));
            }
        }
    });
}

/// Registers an `OpenXLA` FFI custom user-data type ID (`PJRT_FFI_TypeID_Register`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_FFI_TypeID_Register_Args`] (`40` or `48` bytes).
///
/// # Safety
///
/// `raw_args` must be null or point to at least 40 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_FFI_TypeID_Register(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        if actual_size < PJRT_FFI_TypeID_Register_Args::MIN_STRUCT_SIZE {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_FFI_TypeID_Register_Args",
                expected_min: PJRT_FFI_TypeID_Register_Args::MIN_STRUCT_SIZE,
                actual: actual_size,
            });
        }
        let next_id = FFI_TYPE_ID_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
        // SAFETY: Offset `+32` is `int64_t type_id` in both 40-byte and 48-byte `PJRT_FFI_TypeID_Register_Args`.
        unsafe {
            *raw_args.byte_add(32).cast::<i64>() = next_id;
        }
        ptr::null_mut()
    })
}

/// Attaches typed user data to a `PJRT_ExecuteContext` (`PJRT_FFI_UserData_Add`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_FFI_UserData_Add_Args`] (`40` or `48` bytes).
///
/// # Safety
///
/// `raw_args` must be null or point to at least 40 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_FFI_UserData_Add(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        if actual_size < PJRT_FFI_UserData_Add_Args::MIN_STRUCT_SIZE {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_FFI_UserData_Add_Args",
                expected_min: PJRT_FFI_UserData_Add_Args::MIN_STRUCT_SIZE,
                actual: actual_size,
            });
        }
        if actual_size >= 48 {
            // SAFETY: `actual_size >= 48` covers `data` (`+32`) and `deleter` (`+40`).
            let args = unsafe { &mut *raw_args.cast::<PJRT_FFI_UserData_Add_Args>() };
            if let Some(deleter) = args.deleter
                && !args.data.is_null()
            {
                // SAFETY: Invoking caller-provided deleter on `args.data`.
                unsafe {
                    deleter(args.data);
                }
                args.data = ptr::null_mut();
            }
        }
        ptr::null_mut()
    })
}

/// Registers an `OpenXLA` Typed FFI (`xla::ffi`) custom-call handler (`PJRT_FFI_Register_Handler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_FFI_Register_Handler_Args`] (`>= 52` bytes).
///
/// # Safety
///
/// `raw_args` must be null or point to at least 52 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_FFI_Register_Handler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        if actual_size < PJRT_FFI_Register_Handler_Args::MIN_STRUCT_SIZE {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_FFI_Register_Handler_Args",
                expected_min: PJRT_FFI_Register_Handler_Args::MIN_STRUCT_SIZE,
                actual: actual_size,
            });
        }
        if actual_size >= size_of::<PJRT_FFI_Register_Handler_Args>() {
            // SAFETY: `actual_size >= 64` covers full `PJRT_FFI_Register_Handler_Args`.
            let args = unsafe { &*raw_args.cast::<PJRT_FFI_Register_Handler_Args>() };
            if !args.target_name.is_null()
                && args.target_name_size > 0
                && let Some(typed_cb) = args.handler
            {
                // SAFETY: `target_name` points to `target_name_size` bytes.
                let name_bytes = unsafe {
                    std::slice::from_raw_parts(args.target_name.cast::<u8>(), args.target_name_size)
                };
                let c_target =
                    crate::pjrt_helpers::to_c_string(&String::from_utf8_lossy(name_bytes));
                // SAFETY: `c_target` is a valid null-terminated C string and `typed_cb` is valid.
                let _ = unsafe {
                    crate::rust_ergonomic_api::libtpu_rs_register_ffi_handler(
                        c_target.as_ptr(),
                        Some(typed_cb),
                    )
                };
            }
        }
        ptr::null_mut()
    })
}

/// Static `OpenXLA` Typed FFI v0.2 vtable (`GLOBAL_XLA_FFI_API`, `56` bytes).
pub static GLOBAL_XLA_FFI_API: XLA_FFI_Api = XLA_FFI_Api {
    struct_size: size_of::<XLA_FFI_Api>(),
    extension_start: ptr::null_mut(),
    api_version: XLA_FFI_Api_Version {
        struct_size: size_of::<XLA_FFI_Api_Version>(),
        extension_start: ptr::null_mut(),
        major_version: 0,
        minor_version: 2,
    },
    register_handler: PJRT_FFI_Register_Handler,
    register_type_id: PJRT_FFI_TypeID_Register,
};
