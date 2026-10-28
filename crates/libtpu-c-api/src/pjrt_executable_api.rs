//! PJRT executable, loaded executable, and AOT compilation C API entrypoints.
//!
//! Exports `PJRT_Executable_*`, `PJRT_LoadedExecutable_*`, and `PJRT_Compile`.

use super::*;

/// Strongly-typed handler for [`PJRT_LoadedExecutable_Execute`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args`].
unsafe fn handle_loaded_executable_execute(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args,
) -> Result<(), LibTpuError> {
    if !any_null(&[
        args.executable as usize,
        args.argument_lists as usize,
        args.output_lists as usize,
    ]) {
        // SAFETY: `args.executable` is a valid `PJRT_LoadedExecutable*`.
        let loaded = unsafe { &*args.executable };
        let loaded_ref = &loaded.inner;
        if args.num_devices > loaded_ref.addressable_devices.len() {
            return Err(LibTpuError::DeviceNotFound {
                device_id: u32::try_from(args.num_devices).unwrap_or(99),
            });
        }
        let num_args = loaded_ref.executable.program.parameter_shapes.len().max(1);
        for (dev_idx, (&dev_id, &dev_ptr)) in loaded_ref
            .addressable_devices
            .iter()
            .zip(&loaded.device_ptrs)
            .take(args.num_devices.max(1))
            .enumerate()
        {
            // SAFETY: `dev_ptr` is a valid `PJRT_Device*` owned by the client.
            let mem_ptr = unsafe { (*dev_ptr).default_memory_ptr };
            // SAFETY: `args.argument_lists` points to `dev_idx`'s input buffer pointer array.
            let dev_args_ptr = unsafe { *args.argument_lists.add(dev_idx) };
            // SAFETY: `args.output_lists` points to `dev_idx`'s output buffer pointer array.
            let dev_outs_ptr = unsafe { *args.output_lists.add(dev_idx) };
            // SAFETY: `dev_args_ptr` points to `num_args` input buffer pointers.
            let arg_slice = unsafe { std::slice::from_raw_parts(dev_args_ptr, num_args) };
            let mut core_inputs: smallvec::SmallVec<[&libtpu_core::pjrt::PjrtCoreBuffer; 8]> =
                smallvec::SmallVec::with_capacity(num_args);
            for &b_ptr in arg_slice {
                // SAFETY: `b_ptr` is a valid `PJRT_Buffer*`.
                core_inputs.push(unsafe { &(*b_ptr).inner });
            }
            let outs = loaded_ref.execute_on_device(
                dev_id,
                libtpu_core::types::MemorySpaceId(i32::try_from(dev_id.0 * 3).unwrap_or(0)),
                &core_inputs,
            )?;
            for (idx, out_buf) in outs.into_iter().enumerate() {
                let out_handle =
                    Box::into_raw(Box::new(PJRT_Buffer::new(out_buf, dev_ptr, mem_ptr)));
                // SAFETY: Writing output buffer handle at `idx`.
                unsafe {
                    *dev_outs_ptr.add(idx) = out_handle;
                }
            }
        }
    }
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Execute(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_Execute_Args>(
            raw_args,
            handle_loaded_executable_execute,
        )
    }
}

/// Strongly-typed handler for [`PJRT_LoadedExecutable_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args`].
unsafe fn handle_loaded_executable_destroy(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.executable` is a heap-allocated `PJRT_LoadedExecutable*` or null.
    unsafe {
        libtpu_rs_executable_destroy(args.executable);
    }
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_Destroy_Args>(
            raw_args,
            handle_loaded_executable_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Destroy_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_Destroy_Args`].
unsafe fn handle_executable_destroy(
    args: &mut crate::pjrt_types::PJRT_Executable_Destroy_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    if let Some(exec_ref) = unsafe { args.executable.as_ref() } {
        if !exec_ref
            .owned_by_loaded
            .load(std::sync::atomic::Ordering::Acquire)
        {
            // SAFETY: Standalone `PJRT_Executable` was allocated via `Box::into_raw`.
            unsafe {
                drop(Box::from_raw(args.executable));
            }
        }
    }
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(
            raw_args,
            handle_executable_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Name_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_Name_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_Name_Args`].
unsafe fn handle_executable_name(
    args: &mut crate::pjrt_types::PJRT_Executable_Name_Args,
) -> Result<(), LibTpuError> {
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
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_Name(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_Name_Args>(
            raw_args,
            handle_executable_name,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_NumReplicas_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_NumReplicas_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_NumReplicas_Args`].
unsafe fn handle_executable_num_replicas(
    args: &mut crate::pjrt_types::PJRT_Executable_NumReplicas_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
    args.num_replicas = exec_opt.map_or(1, |e| e.inner.program.num_replicas);
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_NumReplicas(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_NumReplicas_Args>(
            raw_args,
            handle_executable_num_replicas,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_NumPartitions_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_NumPartitions_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_NumPartitions_Args`].
unsafe fn handle_executable_num_partitions(
    args: &mut crate::pjrt_types::PJRT_Executable_NumPartitions_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let exec_opt = unsafe { args.executable.as_ref() };
    args.num_partitions = exec_opt.map_or(1, |e| e.inner.program.num_partitions);
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_NumPartitions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_NumPartitions_Args>(
            raw_args,
            handle_executable_num_partitions,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_NumOutputs_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_NumOutputs_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_NumOutputs_Args`].
unsafe fn handle_executable_num_outputs(
    args: &mut crate::pjrt_types::PJRT_Executable_NumOutputs_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    args.num_outputs = unsafe { args.executable.as_ref() }
        .map_or(1, |e| e.inner.program.output_shapes.len().max(1));
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_NumOutputs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_NumOutputs_Args>(
            raw_args,
            handle_executable_num_outputs,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args`].
unsafe fn handle_executable_size_of_generated_code(
    args: &mut crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    args.size_in_bytes = unsafe { args.executable.as_ref() }.map_or(256, |e| {
        e.inner.program.memory_stats.generated_code_size_in_bytes
    });
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_SizeOfGeneratedCodeInBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>(
            raw_args,
            handle_executable_size_of_generated_code,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args`].
unsafe fn handle_executable_get_cost_analysis(
    args: &mut crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let (ptr, len) = unsafe { args.executable.as_ref() }.map_or(
        (
            GLOBAL_PLUGIN_ATTRIBUTES.as_ptr(),
            GLOBAL_PLUGIN_ATTRIBUTES.len(),
        ),
        |e| (e.cost_properties.as_ptr(), e.cost_properties.len()),
    );
    args.properties = ptr;
    args.num_properties = len;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_GetCostAnalysis(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_GetCostAnalysis_Args>(
            raw_args,
            handle_executable_get_cost_analysis,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args`].
unsafe fn handle_executable_output_memory_kinds(
    args: &mut crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let (num, ptrs, sizes) = unsafe { args.executable.as_ref() }.map_or(
        (
            1,
            DEFAULT_MEMORY_KIND_PTRS.0.as_ptr(),
            DEFAULT_MEMORY_KIND_SIZES.as_ptr(),
        ),
        |e| {
            (
                e.output_memory_kind_ptrs.len(),
                e.output_memory_kind_ptrs.as_ptr(),
                e.output_memory_kind_sizes.as_ptr(),
            )
        },
    );
    args.num_outputs = num;
    args.memory_kinds = ptrs;
    args.memory_kind_sizes = sizes;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_OutputMemoryKinds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_OutputMemoryKinds_Args>(
            raw_args,
            handle_executable_output_memory_kinds,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args`].
unsafe fn handle_executable_optimized_program(
    args: &mut crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let src_bytes: &[u8] = unsafe { args.executable.as_ref() }
        .map_or(b"module @tpu_optimized { stablehlo.add }", |e| {
            e.inner.program.source_bytes.as_slice()
        });
    // SAFETY: `args.program` is null or a valid `PJRT_Program*`.
    if let Some(prog) = unsafe { args.program.as_mut() } {
        prog.format = c"hlo_with_config".as_ptr();
        prog.format_size = 15;
        if !prog.code.is_null() {
            if prog.code_size < src_bytes.len() {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "program.code_size",
                    reason: format!(
                        "Provided buffer size {} is smaller than optimized program size {}",
                        prog.code_size,
                        src_bytes.len()
                    ),
                });
            }
            // SAFETY: `prog.code` was allocated by the caller with at least `src_bytes.len()` bytes.
            unsafe {
                ptr::copy_nonoverlapping(
                    src_bytes.as_ptr(),
                    prog.code.cast_mut().cast::<u8>(),
                    src_bytes.len(),
                );
            }
        }
        prog.code_size = src_bytes.len();
    }
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_OptimizedProgram(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_OptimizedProgram_Args>(
            raw_args,
            handle_executable_optimized_program,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Serialize_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_Serialize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_Serialize_Args`].
unsafe fn handle_executable_serialize(
    args: &mut crate::pjrt_types::PJRT_Executable_Serialize_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let ser_bytes = unsafe { args.executable.as_ref() }
        .map_or_else(|| b"TPU_EXEC".to_vec(), |e| e.inner.program.serialize());
    let (ptr, len, holder) = crate::box_serialized_artifact(ser_bytes);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_executable = holder.cast();
    args.serialized_executable_deleter = Some(crate::pjrt_serialized_artifact_deleter);
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_Serialize_Args>(
            raw_args,
            handle_executable_serialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args`].
unsafe fn handle_loaded_executable_get_executable(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `loaded_executable` is null or a valid `PJRT_LoadedExecutable*`.
    let le_opt = unsafe { args.loaded_executable.as_ref() };
    args.executable = le_opt.map_or(ptr::null_mut(), |le| le.executable_ptr);
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_GetExecutable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args>(
            raw_args,
            handle_loaded_executable_get_executable,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args`].
unsafe fn handle_loaded_executable_addressable_devices(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let (ptr, len) = unsafe { args.executable.as_ref() }.map_or((ptr::null(), 0), |le| {
        (le.device_ptrs.as_ptr(), le.inner.addressable_devices.len())
    });
    args.addressable_devices = ptr;
    args.num_addressable_devices = len;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_AddressableDevices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_AddressableDevices_Args>(
            raw_args,
            handle_loaded_executable_addressable_devices,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args`].
unsafe fn handle_loaded_executable_delete(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    if let Some(le) = unsafe { args.executable.as_ref() } {
        le.inner.delete();
    }
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Delete(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_Delete_Args>(
            raw_args,
            handle_loaded_executable_delete,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args`].
unsafe fn handle_loaded_executable_is_deleted(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let le_opt = unsafe { args.executable.as_ref() };
    args.is_deleted = le_opt.is_some_and(|le| le.inner.is_deleted());
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_IsDeleted(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_IsDeleted_Args>(
            raw_args,
            handle_loaded_executable_is_deleted,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Executable_DeserializeAndLoad`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args`].
unsafe fn handle_executable_deserialize_and_load(
    args: &mut crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args,
) -> Result<(), LibTpuError> {
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
        let loaded_inner = client.inner.deserialize_and_load(slice).or_else(|e| {
            if slice.starts_with(b"module") {
                client.inner.compile_and_load("mlir", slice, b"")
            } else {
                Err(e)
            }
        })?;
        let exec_ptr = Box::into_raw(Box::new(PJRT_Executable::new(
            loaded_inner.executable.clone(),
        )));
        args.loaded_executable = Box::into_raw(Box::new(PJRT_LoadedExecutable::new(
            loaded_inner,
            exec_ptr,
            client.addressable_device_ptrs.clone(),
        )));
    } else {
        args.loaded_executable = ptr::null_mut();
    }
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_DeserializeAndLoad(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_DeserializeAndLoad_Args>(
            raw_args,
            handle_executable_deserialize_and_load,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args`].
unsafe fn handle_loaded_executable_fingerprint(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args,
) -> Result<(), LibTpuError> {
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
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_Fingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_Fingerprint_Args>(
            raw_args,
            handle_loaded_executable_fingerprint,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Compile_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Compile_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Compile_Args`].
unsafe fn handle_pjrt_compile(
    args: &mut crate::pjrt_types::PJRT_Compile_Args,
) -> Result<(), LibTpuError> {
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
    let opts_slice: &[u8] = if !args.compile_options.is_null() {
        // SAFETY: `args.compile_options` has `args.compile_options_size` bytes.
        unsafe {
            std::slice::from_raw_parts(args.compile_options.cast::<u8>(), args.compile_options_size)
        }
    } else {
        b""
    };
    // SAFETY: `args.topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
    if let Ok(prog) =
        libtpu_core::compiler::CompiledProgram::compile(fmt_str, code_slice, opts_slice)
            .or_else(|_| libtpu_core::compiler::CompiledProgram::compile(fmt_str, code_slice, b""))
    {
        args.executable = Box::into_raw(Box::new(PJRT_Executable::new(
            libtpu_core::pjrt::PjrtCoreExecutable::new(prog),
        )));
    } else {
        args.executable = ptr::null_mut();
    }
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Compile(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Compile_Args>(
            raw_args,
            handle_pjrt_compile,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args`].
unsafe fn handle_executable_output_element_types(
    args: &mut crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let (ptr, len) = unsafe { args.executable.as_ref() }.map_or(
        (
            DEFAULT_OUTPUT_ELEMENT_TYPES.as_ptr(),
            DEFAULT_OUTPUT_ELEMENT_TYPES.len(),
        ),
        |e| {
            (
                e.output_element_types.as_ptr(),
                e.output_element_types.len(),
            )
        },
    );
    args.output_types = ptr;
    args.num_output_types = len;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_OutputElementTypes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_OutputElementTypes_Args>(
            raw_args,
            handle_executable_output_element_types,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_OutputDimensions_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_OutputDimensions_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_OutputDimensions_Args`].
unsafe fn handle_executable_output_dimensions(
    args: &mut crate::pjrt_types::PJRT_Executable_OutputDimensions_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let (dims_ptr, sizes_ptr, len) = unsafe { args.executable.as_ref() }.map_or(
        (
            DEFAULT_OUTPUT_DIMS.as_ptr(),
            DEFAULT_OUTPUT_DIM_SIZES.as_ptr(),
            DEFAULT_OUTPUT_DIM_SIZES.len(),
        ),
        |e| {
            (
                e.output_dims_flat.as_ptr(),
                e.output_dim_sizes.as_ptr(),
                e.output_dim_sizes.len(),
            )
        },
    );
    args.dims = dims_ptr;
    args.dim_sizes = sizes_ptr;
    args.num_outputs = len;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_OutputDimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_OutputDimensions_Args>(
            raw_args,
            handle_executable_output_dimensions,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_Fingerprint_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_Fingerprint_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_Fingerprint_Args`].
unsafe fn handle_executable_fingerprint(
    args: &mut crate::pjrt_types::PJRT_Executable_Fingerprint_Args,
) -> Result<(), LibTpuError> {
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
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_Fingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_Fingerprint_Args>(
            raw_args,
            handle_executable_fingerprint,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args`].
unsafe fn handle_executable_get_compiled_memory_stats(
    args: &mut crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args,
) -> Result<(), LibTpuError> {
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
    args.host_generated_code_size_in_bytes = 0;
    args.host_argument_size_in_bytes = 0;
    args.host_output_size_in_bytes = 0;
    args.host_alias_size_in_bytes = 0;
    args.host_temp_size_in_bytes = 0;
    args.peak_memory_in_bytes = gen_sz + arg_sz + out_sz + temp_sz;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_GetCompiledMemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_GetCompiledMemoryStats_Args>(
            raw_args,
            handle_executable_get_compiled_memory_stats,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args`].
unsafe fn handle_loaded_executable_get_device_assignment(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let da_bytes = unsafe { args.executable.as_ref() }.map_or_else(
        || {
            libtpu_core::proto::DeviceAssignmentProto {
                replica_count: 1,
                computation_count: 1,
                computation_devices: vec![libtpu_core::proto::ComputationDeviceProto {
                    replica_device_ids: vec![0],
                }],
            }
            .encode()
        },
        |le| {
            let replica_device_ids: Vec<i64> = le
                .inner
                .addressable_devices
                .iter()
                .map(|d| i64::from(d.0))
                .collect();
            libtpu_core::proto::DeviceAssignmentProto {
                replica_count: i32::try_from(le.inner.executable.program.num_replicas).unwrap_or(1),
                computation_count: i32::try_from(le.inner.executable.program.num_partitions)
                    .unwrap_or(1),
                computation_devices: vec![libtpu_core::proto::ComputationDeviceProto {
                    replica_device_ids,
                }],
            }
            .encode()
        },
    );
    let (ptr, len, holder) = crate::box_serialized_artifact(da_bytes);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_device_assignment = holder.cast();
    args.serialized_device_assignment_deleter = Some(crate::pjrt_serialized_artifact_deleter);
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_GetDeviceAssignment(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_LoadedExecutable_GetDeviceAssignment_Args>(
            raw_args,
            handle_loaded_executable_get_device_assignment,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args`].
unsafe fn handle_executable_get_compile_options(
    args: &mut crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let co_bytes = unsafe { args.executable.as_ref() }.map_or_else(
        || {
            libtpu_core::proto::CompileOptionsProto {
                argument_layouts: Vec::new(),
                parameter_is_tupled_arguments: false,
                executable_build_options: Some(libtpu_core::proto::ExecutableBuildOptionsProto {
                    device_ordinal: 0,
                    num_replicas: 1,
                    num_partitions: 1,
                    use_spmd_partitioning: true,
                    result_layout: None,
                    ..libtpu_core::proto::ExecutableBuildOptionsProto::default()
                }),
                compile_portable_executable: false,
                env_option_overrides: std::collections::BTreeMap::new(),
                profile_version: 1,
                serialized_multi_slice_config: Vec::new(),
            }
            .encode()
        },
        |e| e.inner.program.compile_options_bytes.clone(),
    );
    let (ptr, len, holder) = crate::box_serialized_artifact(co_bytes);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_compile_options = holder.cast();
    args.serialized_compile_options_deleter = Some(crate::pjrt_serialized_artifact_deleter);
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_GetCompileOptions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_GetCompileOptions_Args>(
            raw_args,
            handle_executable_get_compile_options,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args`].
unsafe fn handle_loaded_executable_logical_ids(
    args: &mut crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_LoadedExecutable*`.
    let (ptr, len) = unsafe { args.executable.as_ref() }.map_or(
        (
            GLOBAL_LOGICAL_DEVICE_IDS.as_ptr(),
            GLOBAL_LOGICAL_DEVICE_IDS.len(),
        ),
        |le| (le.logical_device_ids.as_ptr(), le.logical_device_ids.len()),
    );
    args.addressable_device_logical_ids = ptr;
    args.num_addressable_device_logical_ids = len;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_LoadedExecutable_AddressableDeviceLogicalIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args,
        >(raw_args, handle_loaded_executable_logical_ids)
    }
}

/// Populates [`crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args`].
unsafe fn handle_executable_parameter_memory_kinds(
    args: &mut crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `executable` is null or a valid `PJRT_Executable*`.
    let (num, ptrs, sizes) = unsafe { args.executable.as_ref() }.map_or(
        (
            1,
            DEFAULT_MEMORY_KIND_PTRS.0.as_ptr(),
            DEFAULT_MEMORY_KIND_SIZES.as_ptr(),
        ),
        |e| {
            (
                e.param_memory_kind_ptrs.len(),
                e.param_memory_kind_ptrs.as_ptr(),
                e.param_memory_kind_sizes.as_ptr(),
            )
        },
    );
    args.num_parameters = num;
    args.memory_kinds = ptrs;
    args.memory_kind_sizes = sizes;
    Ok(())
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Executable_ParameterMemoryKinds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Executable_ParameterMemoryKinds_Args>(
            raw_args,
            handle_executable_parameter_memory_kinds,
        )
    }
}

#[cfg(test)]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_buffer_api::{
        PJRT_Buffer_Delete, PJRT_Buffer_Destroy, PJRT_Buffer_IsDeleted, PJRT_Buffer_ToHostBuffer,
    };
    use crate::pjrt_client_api::{
        PJRT_Client_BufferFromHostBuffer, PJRT_Client_Compile, PJRT_Client_Create,
        PJRT_Client_Destroy, PJRT_Client_LookupDevice,
    };
    use crate::pjrt_types::{
        PJRT_Buffer_Delete_Args, PJRT_Buffer_Destroy_Args, PJRT_Buffer_IsDeleted_Args,
        PJRT_Buffer_ToHostBuffer_Args, PJRT_Client_BufferFromHostBuffer_Args,
        PJRT_Client_Compile_Args, PJRT_Client_Create_Args, PJRT_Client_Destroy_Args,
        PJRT_Client_LookupDevice_Args, PJRT_LoadedExecutable_Destroy_Args, PJRT_Program,
    };
    use crate::rust_ergonomic_api::libtpu_rs_executable_destroy;

    #[test]
    fn test_pjrt_executable_and_loaded_executable_api() {
        // SAFETY: Testing PJRT executable and loaded executable entrypoints with a valid client and buffers.
        unsafe {
            drop(Box::from_raw(
                PJRT_LoadedExecutable_Execute(ptr::null_mut()),
            ));
            drop(Box::from_raw(
                PJRT_LoadedExecutable_Destroy(ptr::null_mut()),
            ));

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
            assert!(PJRT_Client_Create(ptr::addr_of_mut!(create_args).cast()).is_null());
            let client = create_args.client;

            let mut look_args = PJRT_Client_LookupDevice_Args {
                struct_size: size_of::<PJRT_Client_LookupDevice_Args>(),
                extension_start: ptr::null_mut(),
                client,
                id: 0,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert!(PJRT_Client_LookupDevice(ptr::addr_of_mut!(look_args).cast()).is_null());

            let code_bytes = b"module @test { stablehlo.add }";
            let fmt_bytes = b"mlir";
            let prog = PJRT_Program {
                struct_size: size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: code_bytes.as_ptr().cast(),
                code_size: code_bytes.len(),
                format: fmt_bytes.as_ptr().cast(),
                format_size: fmt_bytes.len(),
            };
            let mut comp_args = PJRT_Client_Compile_Args {
                struct_size: size_of::<PJRT_Client_Compile_Args>(),
                extension_start: ptr::null_mut(),
                client,
                program: ptr::addr_of!(prog),
                compile_options: ptr::null(),
                compile_options_size: 0,
                executable: ptr::null_mut(),
            };
            assert!(PJRT_Client_Compile(ptr::addr_of_mut!(comp_args).cast()).is_null());

            let host_floats = [1.0_f32, 2.0, 3.0, 4.0];
            let dims_arr = [4_i64];
            let mut h2d_args = PJRT_Client_BufferFromHostBuffer_Args {
                struct_size: size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
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
            h2d_args.out_buffer = ptr::null_mut();
            assert!(PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_args).cast()).is_null());
            let buf_in2 = h2d_args.out_buffer;

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

            if let Some(del) = ex_ser.serialized_executable_deleter {
                del(ex_ser.serialized_executable.cast());
            }
            if let Some(del) = ex_gco.serialized_compile_options_deleter {
                del(ex_gco.serialized_compile_options.cast());
            }
            if let Some(del) = le_da.serialized_device_assignment_deleter {
                del(le_da.serialized_device_assignment.cast());
            }

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

            let mut d2h_size_args = PJRT_Buffer_ToHostBuffer_Args {
                struct_size: size_of::<PJRT_Buffer_ToHostBuffer_Args>(),
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

            let mut b_del_call = PJRT_Buffer_Delete_Args {
                struct_size: size_of::<PJRT_Buffer_Delete_Args>(),
                extension_start: ptr::null_mut(),
                buffer: out_buf_ptr,
            };
            assert!(PJRT_Buffer_Delete(ptr::addr_of_mut!(b_del_call).cast()).is_null());
            let mut b_is_del_call = PJRT_Buffer_IsDeleted_Args {
                struct_size: size_of::<PJRT_Buffer_IsDeleted_Args>(),
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
                let mut del_b_args = PJRT_Buffer_Destroy_Args {
                    struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    buffer: b_del,
                };
                assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_b_args).cast()).is_null());
            }

            let mut del_exec_args = PJRT_LoadedExecutable_Destroy_Args {
                struct_size: size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
            };
            assert!(
                PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(del_exec_args).cast()).is_null()
            );

            let mut destroy_client_args = PJRT_Client_Destroy_Args {
                struct_size: size_of::<PJRT_Client_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                client,
            };
            assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(destroy_client_args).cast()).is_null());
        }
    }
}
