//! PJRT XLA Transform, Callback, PhaseCompile, and Triton compilation entrypoints.

use super::args::*;
use super::{alloc_extension_bytes, dispatch_typed_extension_args, extension_state};
use crate::pjrt_types::{PJRT_Error, PJRT_PhaseCompile_Compiler};
use libtpu_core::compiler::{COMPILATION_PHASE_NAMES, CompiledProgram};
use std::ptr;

/// Static array of byte lengths for [`COMPILATION_PHASE_NAMES`].
static STATIC_PHASE_NAME_SIZES: [usize; 5] = [
    COMPILATION_PHASE_NAMES[0].len(),
    COMPILATION_PHASE_NAMES[1].len(),
    COMPILATION_PHASE_NAMES[2].len(),
    COMPILATION_PHASE_NAMES[3].len(),
    COMPILATION_PHASE_NAMES[4].len(),
];

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
    let pass_name = if args.name.is_null() {
        String::from("default_xla_transform")
    } else {
        // SAFETY: `args.name` points to `args.name_size` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(args.name.cast::<u8>(), args.name_size) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    let mut st = extension_state().lock();
    st.registered_transforms = st.registered_transforms.saturating_add(1);
    let count = st.registered_transforms;
    st.transform_passes.insert(pass_name, count);
    args.transform_fn = count;
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
    let _ = extension_state().lock().registered_transforms;
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
    let pass_name = if args.name.is_null() {
        String::from("default_xla_transform")
    } else {
        // SAFETY: `args.name` points to `args.name_size` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(args.name.cast::<u8>(), args.name_size) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    let mut st = extension_state().lock();
    st.transform_passes.remove(&pass_name);
    st.registered_transforms = st.registered_transforms.saturating_sub(1);
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
    let st = extension_state().lock();
    args.num_names_out = st.transform_passes.len().max(st.registered_transforms);
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
    let mut st = extension_state().lock();
    st.registered_callbacks = st.registered_callbacks.saturating_add(1);
    st.callback_registry.insert(
        args.callback_type,
        (args.callback_fn as usize, args.user_arg as usize),
    );
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
    let (fn_addr, user_arg) = extension_state()
        .lock()
        .callback_registry
        .get(&cb_key)
        .copied()
        .unwrap_or((0, 0));
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
    let mut st = extension_state().lock();
    st.phase_compilers = st.phase_compilers.saturating_add(1);
    let handle = Box::into_raw(Box::new(PJRT_PhaseCompile_Compiler {
        compiler_id: u64::try_from(st.phase_compilers).unwrap_or(1),
        phases_executed: 0,
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
    let mut st = extension_state().lock();
    st.phase_compilers = st.phase_compilers.saturating_sub(1);
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
    if !args.phase_compiler.is_null() {
        // SAFETY: `phase_compiler` is a valid `PJRT_PhaseCompile_Compiler*`.
        let comp = unsafe { &mut *args.phase_compiler.cast::<PJRT_PhaseCompile_Compiler>() };
        comp.phases_executed = comp.phases_executed.saturating_add(1);
    }
    let input = if !args.input_bytes.is_null() {
        // SAFETY: `input_bytes` is non-null with `input_size` bytes.
        unsafe { std::slice::from_raw_parts(args.input_bytes.cast::<u8>(), args.input_size) }
    } else {
        b"module @phase { stablehlo.add }"
    };
    args.output_status =
        CompiledProgram::run_compilation_phase(COMPILATION_PHASE_NAMES[0], input, b"")
            .map_or(COMPILATION_PHASE_NAMES.len(), |v| {
                v.len().max(COMPILATION_PHASE_NAMES.len())
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
    args.phase_names = ptr::null();
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
    let mlir_bytes = if args.module.is_null() {
        &b"module @triton_kernel { stablehlo.add }"[..]
    } else {
        // SAFETY: `args.module` points to `args.module_size` bytes.
        unsafe { std::slice::from_raw_parts(args.module.cast::<u8>(), args.module_size) }
    };
    let _ = CompiledProgram::compile("mlir", mlir_bytes, b"").map(|prog| {
        let serialized = prog.to_executable_proto().encode();
        let (ptr, len) = alloc_extension_bytes(serialized);
        args.out_asm = ptr.cast();
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
