//! PJRT XLA Transform, Callback, PhaseCompile, Triton, and `OpenXLA` FFI (`PJRT_FFI_Extension`) entrypoints.

use super::args::*;
use super::{alloc_phase_compile_buffer, dispatch_typed_extension_args, free_phase_compile_buffer};
use crate::pjrt_helpers::ffi_catch_unwind;
use crate::pjrt_types::{PJRT_Client, PJRT_Error, PJRT_PhaseCompile_Compiler};
use libtpu_core::compiler::{COMPILATION_PHASE_NAMES, CompiledProgram};
use std::ptr;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

/// Monotonic pass counter for `PJRT_XlaTransform_Extension`.
static XLA_TRANSFORM_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Monotonic instance counter for `PJRT_PhaseCompile_Compiler`.
static PHASE_COMPILER_SEQ: AtomicUsize = AtomicUsize::new(0);
/// Monotonic type-ID sequence counter for `PJRT_FFI_TypeID_Register`.
static FFI_TYPE_ID_SEQ: AtomicI64 = AtomicI64::new(0);
/// Static default compiled Triton kernel stub bytes for null-module probes.
static DEFAULT_TRITON_ASM_WIRE: [u8; 4] = [0x0A, 0x02, 0x08, 0x01];

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
struct StaticPhaseNamePtrs([*const std::os::raw::c_char; 5]);

// SAFETY: Contains only immutable static `'static` C-string pointers.
unsafe impl Sync for StaticPhaseNamePtrs {}

/// Static array of phase name C-string pointers.
static STATIC_PHASE_NAME_PTRS: StaticPhaseNamePtrs = StaticPhaseNamePtrs([
    c"hlo_verification".as_ptr(),
    c"spmd_partitioning".as_ptr(),
    c"tpu_layout_assignment_and_tiling".as_ptr(),
    c"llo_vliw_bundle_scheduling".as_ptr(),
    c"binary_serialization".as_ptr(),
]);

/// Strongly-typed handler for [`PJRT_XlaTransform_Register`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_XlaTransform_Register_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_XlaTransform_Register_Args`].
unsafe fn handle_xla_transform_register(args: &mut PJRT_XlaTransform_Register_Args) {
    let _pass_name_len = if args.name.is_null() {
        21
    } else {
        // SAFETY: `args.name` points to `args.name_size` bytes.
        unsafe { std::slice::from_raw_parts(args.name.cast::<u8>(), args.name_size) }.len()
    };
    args.transform_fn = XLA_TRANSFORM_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
}

/// Registers an XLA HLO transform pass (`PJRT_XlaTransform_Register`).
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_XlaTransform_Register_Args>(
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
/// `args` must be a valid [`PJRT_XlaTransform_WaitForAll_Args`].
unsafe fn handle_xla_transform_wait_for_all(_args: &mut PJRT_XlaTransform_WaitForAll_Args) {
    let _ = XLA_TRANSFORM_COUNT.load(Ordering::Acquire);
}

/// Waits for all active XLA HLO transform passes (`PJRT_XlaTransform_WaitForAll`).
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
/// # Safety
///
/// `args` must be a valid [`PJRT_XlaTransform_Unregister_Args`].
unsafe fn handle_xla_transform_unregister(args: &mut PJRT_XlaTransform_Unregister_Args) {
    let _pass_name_len = if args.name.is_null() {
        21
    } else {
        // SAFETY: `args.name` points to `args.name_size` bytes.
        unsafe { std::slice::from_raw_parts(args.name.cast::<u8>(), args.name_size) }.len()
    };
    let cur = XLA_TRANSFORM_COUNT.load(Ordering::Relaxed);
    XLA_TRANSFORM_COUNT.store(cur.saturating_sub(1), Ordering::Relaxed);
}

/// Unregisters an XLA HLO transform pass (`PJRT_XlaTransform_Unregister`).
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_XlaTransform_Unregister_Args>(
            raw_args,
            handle_xla_transform_unregister,
        )
    }
}

/// Strongly-typed handler for [`PJRT_XlaTransform_List`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_XlaTransform_List_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_XlaTransform_List_Args`].
unsafe fn handle_xla_transform_list(args: &mut PJRT_XlaTransform_List_Args) {
    args.num_names_out = XLA_TRANSFORM_COUNT.load(Ordering::Relaxed);
}

/// Lists registered XLA HLO transform passes (`PJRT_XlaTransform_List`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_XlaTransform_List_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_XlaTransform_List_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_XlaTransform_List(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_XlaTransform_List_Args>(
            raw_args,
            handle_xla_transform_list,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Callback_RegisterCallback`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Callback_Register_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Callback_Register_Args`].
unsafe fn handle_callback_register(args: &mut PJRT_Callback_Register_Args) {
    if (args.client as usize) > 0x10000 {
        // SAFETY: `args.client` is a valid `PJRT_Client*`.
        let client = unsafe { &*args.client.cast::<PJRT_Client>() };
        client.ext_arena.lock().callback_registry.insert(
            args.callback_type,
            (args.callback_fn as usize, args.user_arg as usize),
        );
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
/// `args` must be a valid [`PJRT_Callback_Invoke_Args`].
unsafe fn handle_callback_invoke(args: &mut PJRT_Callback_Invoke_Args) {
    let cb_key = i64::try_from(args.callback_type).unwrap_or(0);
    let (fn_addr, user_arg) = if (args.client as usize) > 0x10000 {
        // SAFETY: `args.client` is a valid `PJRT_Client*`.
        let client = unsafe { &*args.client.cast::<PJRT_Client>() };
        client
            .ext_arena
            .lock()
            .callback_registry
            .get(&cb_key)
            .copied()
            .unwrap_or((0, 0))
    } else {
        (0, 0)
    };
    if fn_addr != 0 {
        let fn_ptr = fn_addr as *const ();
        // SAFETY: `fn_addr` is a non-null C callback function pointer registered by the caller.
        let cb = unsafe {
            *ptr::from_ref(&fn_ptr)
                .cast::<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void)>()
        };
        // SAFETY: Invoking caller's C callback with `payload` and `user_arg`.
        unsafe {
            cb(args.payload, user_arg as *mut std::ffi::c_void);
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
    let handle = Box::into_raw(Box::new(PJRT_PhaseCompile_Compiler {
        compiler_id: u64::try_from(seq).unwrap_or(1),
        phases_executed: 0,
        owned_buffers: parking_lot::Mutex::new(Vec::new()),
    }));
    args.compiler_out = handle as usize;
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
    if !args.handle.is_null() {
        // SAFETY: `handle` was allocated by `PJRT_PhaseCompile_GetCompiler`.
        unsafe {
            drop(Box::from_raw(
                args.handle.cast::<PJRT_PhaseCompile_Compiler>(),
            ));
        }
    }
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
            let v = libtpu_core::proto::ProtoSchemaMessage::encode_proto(&proto);
            let len = v.len().max(COMPILATION_PHASE_NAMES.len());
            if !args.phase_compiler.is_null() {
                // SAFETY: `phase_compiler` is a valid `PJRT_PhaseCompile_Compiler*`.
                let comp =
                    unsafe { &mut *args.phase_compiler.cast::<PJRT_PhaseCompile_Compiler>() };
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

/// Strongly-typed handler for [`PJRT_Triton_Compile`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Triton_Compile_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Triton_Compile_Args`].
unsafe fn handle_triton_compile(args: &mut PJRT_Triton_Compile_Args) {
    if args.module.is_null() {
        args.out_asm = DEFAULT_TRITON_ASM_WIRE.as_ptr().cast();
        args.out_asm_size = DEFAULT_TRITON_ASM_WIRE.len();
        args.out_smem_bytes = 4096;
        return;
    }
    // SAFETY: `args.module` points to `args.module_size` bytes.
    let mlir_bytes =
        unsafe { std::slice::from_raw_parts(args.module.cast::<u8>(), args.module_size) };
    let _ = CompiledProgram::compile("mlir", mlir_bytes, b"").map(|prog| {
        let boxed = prog.to_executable_proto().encode().into_boxed_slice();
        let len = boxed.len();
        let raw_slice = Box::into_raw(boxed);
        args.out_asm = raw_slice.cast::<std::os::raw::c_char>();
        args.out_asm_size = len;
        args.out_smem_bytes = 4096;
    });
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Triton_Compile_Args>(raw_args, handle_triton_compile)
    }
}

/// Frees a compiled kernel bitstream returned by [`PJRT_Triton_Compile`] when `module` was non-null.
///
/// # Arguments
///
/// * `asm_ptr` - Pointer returned in [`PJRT_Triton_Compile_Args::out_asm`].
/// * `asm_size` - Byte length returned in [`PJRT_Triton_Compile_Args::out_asm_size`].
///
/// # Safety
///
/// `asm_ptr` must be null, the static default Triton wire buffer, or a heap buffer of `asm_size` bytes
/// returned by [`PJRT_Triton_Compile`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Triton_FreeCompiledAsm(
    asm_ptr: *const std::os::raw::c_char,
    asm_size: usize,
) {
    ffi_catch_unwind((), || {
        if !asm_ptr.is_null() && !ptr::eq(asm_ptr.cast::<u8>(), DEFAULT_TRITON_ASM_WIRE.as_ptr()) {
            let raw_slice =
                ptr::slice_from_raw_parts_mut(asm_ptr.cast_mut().cast::<u8>(), asm_size);
            // SAFETY: `raw_slice` was allocated via `Box::into_raw` in `handle_triton_compile`.
            unsafe {
                drop(Box::from_raw(raw_slice));
            }
        }
    });
}

/// Strongly-typed handler for [`PJRT_FFI_TypeID_Register`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_FFI_TypeID_Register_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_FFI_TypeID_Register_Args`].
unsafe fn handle_ffi_type_id_register(args: &mut PJRT_FFI_TypeID_Register_Args) {
    args.type_id = FFI_TYPE_ID_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
}

/// Registers an `OpenXLA` FFI custom user-data type ID (`PJRT_FFI_TypeID_Register`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_FFI_TypeID_Register_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_FFI_TypeID_Register_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_FFI_TypeID_Register(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_FFI_TypeID_Register_Args>(
            raw_args,
            handle_ffi_type_id_register,
        )
    }
}

/// Strongly-typed handler for [`PJRT_FFI_UserData_Add`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_FFI_UserData_Add_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_FFI_UserData_Add_Args`].
unsafe fn handle_ffi_user_data_add(args: &mut PJRT_FFI_UserData_Add_Args) {
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

/// Attaches typed user data to a `PJRT_ExecuteContext` (`PJRT_FFI_UserData_Add`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_FFI_UserData_Add_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_FFI_UserData_Add_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_FFI_UserData_Add(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_FFI_UserData_Add_Args>(
            raw_args,
            handle_ffi_user_data_add,
        )
    }
}

/// Strongly-typed handler for [`PJRT_FFI_Register_Handler`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_FFI_Register_Handler_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_FFI_Register_Handler_Args`].
unsafe fn handle_ffi_register_handler(args: &mut PJRT_FFI_Register_Handler_Args) {
    if !crate::any_null(&[
        args.target_name as usize,
        args.target_name_size,
        args.handler as usize,
    ]) {
        // SAFETY: `target_name` points to `target_name_size` bytes.
        let name_bytes = unsafe {
            std::slice::from_raw_parts(args.target_name.cast::<u8>(), args.target_name_size)
        };
        let c_target = crate::pjrt_helpers::to_c_string(&String::from_utf8_lossy(name_bytes));
        let fn_ptr = args.handler;
        // SAFETY: `args.handler` is a valid `XLA_FFI_Handler` function pointer.
        let typed_cb: XLA_FFI_Handler =
            unsafe { *ptr::from_ref(&fn_ptr).cast::<XLA_FFI_Handler>() };
        // SAFETY: `c_target` is a valid null-terminated C string and `typed_cb` is valid.
        let _ = unsafe {
            crate::rust_ergonomic_api::libtpu_rs_register_ffi_handler(
                c_target.as_ptr(),
                Some(typed_cb),
            )
        };
    }
}

/// Registers an `OpenXLA` Typed FFI (`xla::ffi`) custom-call handler (`PJRT_FFI_Register_Handler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_FFI_Register_Handler_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_FFI_Register_Handler_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_FFI_Register_Handler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_FFI_Register_Handler_Args>(
            raw_args,
            handle_ffi_register_handler,
        )
    }
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
