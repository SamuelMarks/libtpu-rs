//! Layer 1 (negative/error contract) and Layer 2 (real-handle behavioral lifecycle) tests
//! for all 18 `libtpu==0.0.53.1` PJRT extensions (`EXT_01_FFI` .. `EXT_18_PROFILER`).

#![allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]

use super::*;
use crate::pjrt_buffer_api::{PJRT_Buffer_Destroy, PJRT_Buffer_ToHostBuffer};
use crate::pjrt_client_api::{
    PJRT_Client_BufferFromHostBuffer, PJRT_Client_Compile, PJRT_Client_Create, PJRT_Client_Destroy,
};
use crate::pjrt_error_api::{PJRT_Error_Destroy, PJRT_Error_GetCode, PJRT_Event_Destroy};
use crate::pjrt_executable_api::PJRT_LoadedExecutable_Destroy;
use crate::pjrt_helpers::pjrt_serialized_artifact_deleter;
use crate::pjrt_types::{
    PJRT_Buffer, PJRT_Buffer_Destroy_Args, PJRT_Buffer_ToHostBuffer_Args, PJRT_Client,
    PJRT_Client_BufferFromHostBuffer_Args, PJRT_Client_Compile_Args, PJRT_Client_Create_Args,
    PJRT_Client_Destroy_Args, PJRT_Error_Destroy_Args, PJRT_Error_GetCode_Args,
    PJRT_Event_Destroy_Args, PJRT_LoadedExecutable, PJRT_LoadedExecutable_Destroy_Args,
    PJRT_Program,
};

/// Asserts that `err_ptr` is a non-null [`PJRT_Error`] with exact `expected_code`, then frees it.
///
/// # Arguments
///
/// * `err_ptr` - Returned `PJRT_Error*` pointer to validate and free.
/// * `expected_code` - Expected `PJRT_Error_Code` integer (`3` = `INVALID_ARGUMENT`, etc.).
///
/// # Safety
///
/// `err_ptr` must be a valid heap-allocated `PJRT_Error*`.
#[track_caller]
unsafe fn assert_and_free_pjrt_error(err_ptr: *mut PJRT_Error, expected_code: i32) {
    assert!(!err_ptr.is_null());
    let mut code_args = PJRT_Error_GetCode_Args {
        struct_size: size_of::<PJRT_Error_GetCode_Args>(),
        extension_start: ptr::null_mut(),
        error: err_ptr,
        code: 0,
    };
    // SAFETY: `code_args` and `err_ptr` are valid.
    unsafe {
        assert!(PJRT_Error_GetCode(ptr::addr_of_mut!(code_args).cast()).is_null());
    }
    assert_eq!(code_args.code, expected_code);
    let mut del_args = PJRT_Error_Destroy_Args {
        struct_size: size_of::<PJRT_Error_Destroy_Args>(),
        extension_start: ptr::null_mut(),
        error: err_ptr,
    };
    // SAFETY: Destroying `err_ptr`.
    unsafe {
        PJRT_Error_Destroy(ptr::addr_of_mut!(del_args).cast());
    }
}

/// Allocates a 1-D `F32` [`PJRT_Buffer`] initialized from `vals` on device 0 of `client`.
///
/// # Arguments
///
/// * `client` - Valid `PJRT_Client*` handle.
/// * `vals` - Host `f32` values to upload.
///
/// # Safety
///
/// `client` must point to a valid [`PJRT_Client`].
unsafe fn upload_f32_buffer(client: *mut PJRT_Client, vals: &[f32]) -> *mut PJRT_Buffer {
    // SAFETY: `client` is valid.
    let dev0 = unsafe { *(*client).device_ptrs.first().unwrap_or(&ptr::null_mut()) };
    // SAFETY: `client` is valid.
    let mem0 = unsafe { *(*client).memory_ptrs.first().unwrap_or(&ptr::null_mut()) };
    let dims = [i64::try_from(vals.len()).unwrap_or(0)];
    let mut h2d = PJRT_Client_BufferFromHostBuffer_Args {
        struct_size: size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
        extension_start: ptr::null_mut(),
        client,
        data: vals.as_ptr().cast(),
        element_type: 11,
        pad0: 0,
        dims: dims.as_ptr(),
        num_dims: 1,
        byte_strides: ptr::null(),
        num_byte_strides: 0,
        host_buffer_semantics: 0,
        pad1: 0,
        device: dev0,
        memory: mem0,
        device_layout: ptr::null(),
        done_with_host_buffer: ptr::null_mut(),
        out_buffer: ptr::null_mut(),
    };
    // SAFETY: `h2d` is valid.
    unsafe {
        assert!(PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d).cast()).is_null());
        let mut ev_del = PJRT_Event_Destroy_Args {
            struct_size: size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event: h2d.done_with_host_buffer,
        };
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del).cast()).is_null());
    }
    h2d.out_buffer
}

/// Reads `len` `f32` elements from `buf` back to host memory.
///
/// # Arguments
///
/// * `buf` - Valid `PJRT_Buffer*` handle.
/// * `len` - Number of `f32` elements to read.
///
/// # Safety
///
/// `buf` must point to a valid [`PJRT_Buffer`] of `len` `f32` elements.
unsafe fn download_f32_buffer(buf: *mut PJRT_Buffer, len: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; len];
    let mut d2h = PJRT_Buffer_ToHostBuffer_Args {
        struct_size: size_of::<PJRT_Buffer_ToHostBuffer_Args>(),
        extension_start: ptr::null_mut(),
        src: buf,
        host_layout: ptr::null_mut(),
        dst: out.as_mut_ptr().cast(),
        dst_size: len * size_of::<f32>(),
        event: ptr::null_mut(),
    };
    // SAFETY: `d2h` is valid.
    unsafe {
        assert!(PJRT_Buffer_ToHostBuffer(ptr::addr_of_mut!(d2h).cast()).is_null());
        let mut ev_del = PJRT_Event_Destroy_Args {
            struct_size: size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event: d2h.event,
        };
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del).cast()).is_null());
    }
    out
}

/// Verifies the 18-node `0.0.53.1` extension linked list structure and Layer 1 negative error contracts.
#[test]
fn test_extension_chain_layout_and_layer1_negative_contracts() {
    assert_ne!(
        format!(
            "{GLOBAL_PLUGIN_PROFILER_API:?}{EXT_01_FFI:?}{EXT_02_XLA_TRANSFORM:?}{EXT_03_HOST_ALLOCATOR:?}{EXT_04_MULTI_SLICE:?}{EXT_05_COLLECTIVES:?}{EXT_06_ABI_VERSION:?}{EXT_07_SHARDINGS:?}{EXT_08_MEGASCALE:?}{EXT_09_TPU_EXECUTABLE:?}{EXT_10_TPU_TOPOLOGY:?}{EXT_11_CALLBACK:?}{EXT_12_PHASE_COMPILE:?}{EXT_13_CROSS_HOST:?}{EXT_14_EXEC_METADATA:?}{EXT_15_MEM_DESC:?}{EXT_16_LAYOUTS:?}{EXT_17_RAW_BUFFER:?}{EXT_18_PROFILER:?}"
        ),
        ""
    );

    let expected_chain: [(i32, usize); 18] = [
        (5, 48),
        (24, 56),
        (23, 32),
        (22, 64),
        (21, 96),
        (20, 120),
        (19, 40),
        (18, 288),
        (17, 96),
        (16, 280),
        (14, 40),
        (9, 192),
        (12, 56),
        (13, 40),
        (6, 40),
        (4, 80),
        (8, 80),
        (1, 40),
    ];
    let mut actual_chain = Vec::new();
    let mut curr: *const PJRT_Extension_Base = ptr::addr_of!(EXT_01_FFI.base);
    while !curr.is_null() {
        // SAFETY: Walking immutable `'static` extension chain.
        let hdr = unsafe { *curr };
        actual_chain.push((hdr.ext_type, hdr.struct_size));
        curr = hdr.next;
    }
    assert_eq!(actual_chain.as_slice(), &expected_chain);

    // SAFETY: Layer 1 negative tests: null args, undersized struct_size, and null/invalid handles.
    unsafe {
        assert_and_free_pjrt_error(PJRT_Extension_ValidateArgs(ptr::null_mut()), 3);
        let mut zero_sz: usize = 0;
        assert_and_free_pjrt_error(
            PJRT_Extension_ValidateArgs(ptr::addr_of_mut!(zero_sz).cast()),
            3,
        );
        let mut ok_sz: usize = 16;
        assert!(PJRT_Extension_ValidateArgs(ptr::addr_of_mut!(ok_sz).cast()).is_null());

        for null_fn in [
            PJRT_FFI_TypeID_Register as crate::pjrt_types::PjrtFn,
            PJRT_FFI_UserData_Add,
            PJRT_FFI_Register_Handler,
            PJRT_XlaTransform_Register,
            PJRT_XlaTransform_List,
            PJRT_XlaTransform_Unregister,
            PJRT_MultiSlice_Config_Destroy,
            PJRT_Collectives_CreateCommunicator,
            PJRT_Communicator_Destroy,
            PJRT_Communicator_AllReduce,
            PJRT_AbiVersion_RuntimeDestroy,
            PJRT_AbiVersion_RuntimeCompatibleWithRuntime,
            PJRT_PhaseCompile_DestroyCompiler,
            PJRT_Relocatable_Destroy,
            PJRT_Relocatable_HasSideEffects,
            PJRT_CrossHostTransfers_SendBuffers,
            PJRT_MemoryDescriptions_Get,
            PJRT_MemoryDescriptions_Destroy,
            PJRT_Layouts_Executable_GetOutputLayouts,
            PJRT_RawBuffer_CreateRawAliasOfBuffer,
            PJRT_RawBuffer_Destroy,
            PJRT_RawBuffer_GetHostPointer,
            PJRT_RawBuffer_GetOnDeviceSizeInBytes,
            PJRT_RawBuffer_GetMemorySpace,
            PJRT_RawBuffer_CopyRawHostToDevice,
            PJRT_RawBuffer_CopyRawDeviceToHost,
            PJRT_Triton_Compile,
            PLUGIN_Profiler_Error_Destroy,
        ] {
            assert_and_free_pjrt_error(null_fn(ptr::null_mut()), 3);
            let mut small = [8_usize; 16];
            assert_and_free_pjrt_error(null_fn(small.as_mut_ptr().cast()), 3);
        }

        // Null handle checks on RawBuffer, Communicator, AbiVersion, MultiSlice, Relocatable, Triton
        let mut null_raw_hp = PJRT_RawBuffer_GetHostPointer_Args {
            struct_size: size_of::<PJRT_RawBuffer_GetHostPointer_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: ptr::null(),
            host_pointer_out: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_GetHostPointer(ptr::addr_of_mut!(null_raw_hp).cast()),
            3,
        );
        let mut null_raw_del = PJRT_RawBuffer_Destroy_Args {
            struct_size: size_of::<PJRT_RawBuffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(null_raw_del).cast()),
            3,
        );
        let mut null_comm_del = PJRT_Communicator_Destroy_Args {
            struct_size: size_of::<PJRT_Communicator_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            communicator: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_Destroy(ptr::addr_of_mut!(null_comm_del).cast()),
            3,
        );
        let mut null_abi_compat = PJRT_AbiVersion_IsCompatible_Args {
            struct_size: size_of::<PJRT_AbiVersion_IsCompatible_Args>(),
            extension_start: ptr::null_mut(),
            abi_version: ptr::null(),
            other_abi_version: ptr::null(),
            is_compatible_out: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_RuntimeCompatibleWithRuntime(ptr::addr_of_mut!(null_abi_compat).cast()),
            3,
        );
        let mut empty_triton = PJRT_Triton_Compile_Args {
            struct_size: size_of::<PJRT_Triton_Compile_Args>(),
            extension_start: ptr::null_mut(),
            module: ptr::null(),
            module_size: 0,
            arch_name: c"v6e".as_ptr(),
            arch_name_size: 3,
            num_warps: 4,
            num_ctas: 1,
            num_stages: 2,
            pad: 0,
            out_asm: ptr::null(),
            out_asm_size: 0,
            out_smem_bytes: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Triton_Compile(ptr::addr_of_mut!(empty_triton).cast()),
            3,
        );
    }
}

/// Exercises Layer 2 real-handle lifecycles for `RawBuffer`, `Layouts`, `MemoryDescriptions`,
/// Multi-Rank `Collectives`, `CrossHostTransfers`, `MultiSlice`/`Megascale`, `AbiVersion`, `XlaTransform`,
/// `PhaseCompile`/`Relocatable`, `Triton`, `FFI`, `Callback`, `TpuTopology`, `Shardings`, `TpuExecutable`, and `Profiler`.
#[test]
fn test_layer2_real_handle_extension_lifecycles() {
    let prev_count = extension_call_count();
    // SAFETY: Creating real PJRT client, buffers, executables, and extension handles.
    unsafe {
        let mut cli_create = PJRT_Client_Create_Args {
            struct_size: size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: None,
            kv_put_user_arg: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
        };
        assert!(PJRT_Client_Create(ptr::addr_of_mut!(cli_create).cast()).is_null());
        let client = cli_create.client;
        assert!(!client.is_null());

        // 1. PJRT_XlaTransform_Extension: Register -> List -> WaitForAll -> Compile -> Unregister
        let mut xla_reg = PJRT_XlaTransform_Register_Args {
            struct_size: size_of::<PJRT_XlaTransform_Register_Args>(),
            extension_start: ptr::null_mut(),
            name: c"custom_fusion_pass".as_ptr(),
            name_size: 18,
            transform_fn: 0,
        };
        assert!(PJRT_XlaTransform_Register(ptr::addr_of_mut!(xla_reg).cast()).is_null());
        let mut xla_list = PJRT_XlaTransform_List_Args {
            struct_size: size_of::<PJRT_XlaTransform_List_Args>(),
            extension_start: ptr::null_mut(),
            names_out: ptr::null(),
            num_names_out: 0,
            name_sizes_out: ptr::null(),
        };
        assert!(PJRT_XlaTransform_List(ptr::addr_of_mut!(xla_list).cast()).is_null());
        assert!(xla_list.num_names_out >= 1);
        assert!(!xla_list.names_out.is_null() && !xla_list.name_sizes_out.is_null());
        let mut xla_wait = PJRT_XlaTransform_WaitForAll_Args {
            struct_size: size_of::<PJRT_XlaTransform_WaitForAll_Args>(),
            extension_start: ptr::null_mut(),
            client,
        };
        assert!(PJRT_XlaTransform_WaitForAll(ptr::addr_of_mut!(xla_wait).cast()).is_null());

        let mlir_code = b"module @ext_add { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }";
        let mut prog = PJRT_Program {
            struct_size: size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: mlir_code.as_ptr().cast(),
            code_size: mlir_code.len(),
            format: c"mlir".as_ptr(),
            format_size: 4,
        };
        let mut comp = PJRT_Client_Compile_Args {
            struct_size: size_of::<PJRT_Client_Compile_Args>(),
            extension_start: ptr::null_mut(),
            client,
            program: ptr::addr_of_mut!(prog),
            compile_options: ptr::null(),
            compile_options_size: 0,
            executable: ptr::null_mut(),
        };
        assert!(PJRT_Client_Compile(ptr::addr_of_mut!(comp).cast()).is_null());
        let loaded_exec: *mut PJRT_LoadedExecutable = comp.executable;
        let exec_ptr = (*loaded_exec).executable_ptr;

        let mut xla_unreg = PJRT_XlaTransform_Unregister_Args {
            struct_size: size_of::<PJRT_XlaTransform_Unregister_Args>(),
            extension_start: ptr::null_mut(),
            name: c"custom_fusion_pass".as_ptr(),
            name_size: 18,
        };
        assert!(PJRT_XlaTransform_Unregister(ptr::addr_of_mut!(xla_unreg).cast()).is_null());

        // 2. PJRT_RawBuffer_Extension: CreateRawAlias -> HostPointer/Size/MemorySpace -> Offset H2D/D2H -> Out-of-bounds -> Destroy
        let raw_backing = upload_f32_buffer(client, &[1.0, 2.0, 3.0, 4.0]);
        let mut raw_create = PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args {
            struct_size: size_of::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>(),
            extension_start: ptr::null_mut(),
            buffer: raw_backing,
            raw_buffer_out: ptr::null_mut(),
        };
        assert!(
            PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(raw_create).cast()).is_null()
        );
        let raw_buf = raw_create.raw_buffer_out;
        assert!(!raw_buf.is_null());

        let mut raw_sz = PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args {
            struct_size: size_of::<PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: raw_buf,
            on_device_size_in_bytes: 0,
        };
        assert!(PJRT_RawBuffer_GetOnDeviceSizeInBytes(ptr::addr_of_mut!(raw_sz).cast()).is_null());
        assert!(raw_sz.on_device_size_in_bytes >= 16);

        let mut raw_ms = PJRT_RawBuffer_GetMemorySpace_Args {
            struct_size: size_of::<PJRT_RawBuffer_GetMemorySpace_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: raw_buf,
            memory_space_out: ptr::null_mut(),
        };
        assert!(PJRT_RawBuffer_GetMemorySpace(ptr::addr_of_mut!(raw_ms).cast()).is_null());
        assert!(!raw_ms.memory_space_out.is_null());

        let patch_vals = [99.0_f32, 88.0_f32];
        let mut raw_h2d = PJRT_RawBuffer_CopyRaw_Args {
            struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
            extension_start: ptr::null_mut(),
            buffer: raw_buf,
            host_ptr: patch_vals.as_ptr().cast_mut().cast(),
            offset: 4,
            transfer_size: 8,
            event: ptr::null_mut(),
        };
        assert!(PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(raw_h2d).cast()).is_null());
        drop(Box::from_raw(raw_h2d.event));

        let mut read_back = [0.0_f32; 2];
        let mut raw_d2h = PJRT_RawBuffer_CopyRaw_Args {
            struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
            extension_start: ptr::null_mut(),
            buffer: raw_buf,
            host_ptr: read_back.as_mut_ptr().cast(),
            offset: 4,
            transfer_size: 8,
            event: ptr::null_mut(),
        };
        assert!(PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(raw_d2h).cast()).is_null());
        drop(Box::from_raw(raw_d2h.event));
        assert_eq!(read_back, [99.0_f32, 88.0_f32]);

        // Out-of-bounds offset + transfer_size returns InvalidArgument (3)
        let mut oob_copy = PJRT_RawBuffer_CopyRaw_Args {
            struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
            extension_start: ptr::null_mut(),
            buffer: raw_buf,
            host_ptr: read_back.as_mut_ptr().cast(),
            offset: i64::try_from(raw_sz.on_device_size_in_bytes).unwrap_or(256),
            transfer_size: 8,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(oob_copy).cast()),
            3,
        );

        let mut raw_del = PJRT_RawBuffer_Destroy_Args {
            struct_size: size_of::<PJRT_RawBuffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: raw_buf,
        };
        assert!(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(raw_del).cast()).is_null());

        // 3. PJRT_Layouts & PJRT_MemoryDescriptions
        let dev0_ptr = (*client)
            .device_ptrs
            .first()
            .copied()
            .unwrap_or(ptr::null_mut());
        let dev0_desc = (*dev0_ptr).desc_ptr;
        let mut mem_descs = PJRT_DeviceDescription_MemoryDescriptions_Args {
            struct_size: size_of::<PJRT_DeviceDescription_MemoryDescriptions_Args>(),
            extension_start: ptr::null_mut(),
            device_description: dev0_desc,
            memory_descriptions: ptr::null(),
            num_memory_descriptions: 0,
            default_memory_index: -1,
        };
        assert!(PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(mem_descs).cast()).is_null());
        assert!(mem_descs.num_memory_descriptions >= 3);

        let mut param_layouts = PJRT_Layouts_PJRT_Executable_Layouts_Args {
            struct_size: size_of::<PJRT_Layouts_PJRT_Executable_Layouts_Args>(),
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
            num_layouts: 0,
            layouts: ptr::null(),
        };
        assert!(
            PJRT_Layouts_Executable_GetParameterLayouts(ptr::addr_of_mut!(param_layouts).cast())
                .is_null()
        );
        assert_eq!(param_layouts.num_layouts, 2);
        let mut out_layouts = PJRT_Layouts_PJRT_Executable_Layouts_Args {
            struct_size: size_of::<PJRT_Layouts_PJRT_Executable_Layouts_Args>(),
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
            num_layouts: 0,
            layouts: ptr::null(),
        };
        assert!(
            PJRT_Layouts_Executable_GetOutputLayouts(ptr::addr_of_mut!(out_layouts).cast())
                .is_null()
        );
        assert_eq!(out_layouts.num_layouts, 1);

        // 4. Multi-Rank PJRT_Collectives_Extension (2-rank group AllReduce, ReduceScatter, AllGather, AllToAll, CollectivePermute)
        let group_devs = [0_i32, 1_i32];
        let mut c0_args = PJRT_Collectives_CreateCommunicator_Args {
            struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
            extension_start: ptr::null_mut(),
            collectives: ptr::null_mut(),
            device_ids: group_devs.as_ptr(),
            num_devices: 2,
            communicator: ptr::null_mut(),
        };
        let mut c1_args = c0_args;
        assert!(PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(c0_args).cast()).is_null());
        assert!(PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(c1_args).cast()).is_null());
        let comm0 = c0_args.communicator;
        let comm1 = c1_args.communicator;

        let r0_send = upload_f32_buffer(client, &[1.0, 2.0, 3.0, 4.0]);
        let r1_send = upload_f32_buffer(client, &[10.0, 20.0, 30.0, 40.0]);
        let r0_recv4 = upload_f32_buffer(client, &[0.0; 4]);
        let r1_recv4 = upload_f32_buffer(client, &[0.0; 4]);
        let r0_recv2 = upload_f32_buffer(client, &[0.0; 2]);
        let r1_recv2 = upload_f32_buffer(client, &[0.0; 2]);
        let r0_recv8 = upload_f32_buffer(client, &[0.0; 8]);

        // Stage rank 1 then execute AllReduce Sum on rank 0 & rank 1
        (*comm1)
            .inner
            .stage_rank_buffer(1, (*r1_send).inner.to_host_bytes().unwrap_or_default())
            .ok();
        let mut ar0 = PJRT_Communicator_AllReduce_Args {
            struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm0,
            send_buffer: r0_send,
            recv_buffer: r0_recv4,
            reduction_op: 0,
            pad: 0,
            count: 4,
            event: ptr::null_mut(),
        };
        assert!(PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar0).cast()).is_null());
        drop(Box::from_raw(ar0.event));
        assert_eq!(
            download_f32_buffer(r0_recv4, 4),
            vec![11.0, 22.0, 33.0, 44.0]
        );

        // ReduceScatter Sum: rank 0 gets [11, 22], rank 1 gets [33, 44]
        let mut rs0 = PJRT_Communicator_AllReduce_Args {
            struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm0,
            send_buffer: r0_send,
            recv_buffer: r0_recv2,
            reduction_op: 0,
            pad: 0,
            count: 2,
            event: ptr::null_mut(),
        };
        let mut rs1 = PJRT_Communicator_AllReduce_Args {
            communicator: comm1,
            send_buffer: r1_send,
            recv_buffer: r1_recv2,
            ..rs0
        };
        assert!(PJRT_Communicator_ReduceScatter(ptr::addr_of_mut!(rs0).cast()).is_null());
        drop(Box::from_raw(rs0.event));
        assert!(PJRT_Communicator_ReduceScatter(ptr::addr_of_mut!(rs1).cast()).is_null());
        drop(Box::from_raw(rs1.event));
        assert_eq!(download_f32_buffer(r0_recv2, 2), vec![11.0, 22.0]);
        assert_eq!(download_f32_buffer(r1_recv2, 2), vec![33.0, 44.0]);

        // AllGather: rank 0 gathers [1,2,3,4, 10,20,30,40]
        let mut ag0 = PJRT_Communicator_AllGather_Args {
            struct_size: size_of::<PJRT_Communicator_AllGather_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm0,
            send_buffer: r0_send,
            recv_buffer: r0_recv8,
            count: 4,
            event: ptr::null_mut(),
        };
        assert!(PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag0).cast()).is_null());
        drop(Box::from_raw(ag0.event));
        assert_eq!(
            download_f32_buffer(r0_recv8, 8),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
        );

        // AllToAll: rank 0 gets [1, 2, 10, 20], rank 1 gets [3, 4, 30, 40]
        let mut a2a0 = PJRT_Communicator_CollectivePermute_Args {
            struct_size: size_of::<PJRT_Communicator_CollectivePermute_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm0,
            send_buffer: r0_send,
            recv_buffer: r0_recv4,
            target_ranks: ptr::null(),
            count: 4,
            event: ptr::null_mut(),
        };
        let mut a2a1 = PJRT_Communicator_CollectivePermute_Args {
            communicator: comm1,
            send_buffer: r1_send,
            recv_buffer: r1_recv4,
            ..a2a0
        };
        assert!(PJRT_Communicator_AllToAll(ptr::addr_of_mut!(a2a0).cast()).is_null());
        drop(Box::from_raw(a2a0.event));
        assert!(PJRT_Communicator_AllToAll(ptr::addr_of_mut!(a2a1).cast()).is_null());
        drop(Box::from_raw(a2a1.event));
        assert_eq!(download_f32_buffer(r0_recv4, 4), vec![1.0, 2.0, 10.0, 20.0]);
        assert_eq!(download_f32_buffer(r1_recv4, 4), vec![3.0, 4.0, 30.0, 40.0]);

        // CollectivePermute swap [(0 -> 1), (1 -> 0)]
        let swap_pairs = [0_i32, 1_i32, 1_i32, 0_i32];
        let mut cp0 = PJRT_Communicator_CollectivePermute_Args {
            struct_size: size_of::<PJRT_Communicator_CollectivePermute_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm0,
            send_buffer: r0_send,
            recv_buffer: r0_recv4,
            target_ranks: swap_pairs.as_ptr(),
            count: 4,
            event: ptr::null_mut(),
        };
        assert!(PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp0).cast()).is_null());
        drop(Box::from_raw(cp0.event));
        assert_eq!(
            download_f32_buffer(r0_recv4, 4),
            vec![10.0, 20.0, 30.0, 40.0]
        );

        let mut comm_str = PJRT_Communicator_ToString_Args {
            struct_size: size_of::<PJRT_Communicator_ToString_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm0,
            str_out: ptr::null(),
            str_size: 0,
        };
        assert!(PJRT_Communicator_ToString(ptr::addr_of_mut!(comm_str).cast()).is_null());
        assert!(comm_str.str_size > 0);

        for c in [comm0, comm1] {
            let mut cd = PJRT_Communicator_Destroy_Args {
                struct_size: size_of::<PJRT_Communicator_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                communicator: c,
            };
            assert!(PJRT_Communicator_Destroy(ptr::addr_of_mut!(cd).cast()).is_null());
        }

        // 5. PJRT_CrossHostTransfers_Extension: SendBuffers -> ReceiveBuffers actual HBM transfer
        let xh_dst = upload_f32_buffer(client, &[0.0; 4]);
        let send_arr = [r0_send];
        let recv_arr = [xh_dst];
        let mut xh_send = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: send_arr.as_ptr(),
            num_buffers: 1,
            transfer_key: 777,
            out_event: ptr::null_mut(),
        };
        assert!(PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_send).cast()).is_null());
        drop(Box::from_raw(xh_send.out_event));

        let mut xh_recv = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: recv_arr.as_ptr(),
            num_buffers: 1,
            transfer_key: 777,
            out_event: ptr::null_mut(),
        };
        assert!(
            PJRT_CrossHostTransfers_ReceiveBuffers(ptr::addr_of_mut!(xh_recv).cast()).is_null()
        );
        drop(Box::from_raw(xh_recv.out_event));
        assert_eq!(download_f32_buffer(xh_dst, 4), vec![1.0, 2.0, 3.0, 4.0]);

        // 6. PJRT_AbiVersion_Extension: Runtime + Executable + Proto roundtrip + Compatible vs Incompatible
        let mut rt_abi = PJRT_Client_RuntimeAbiVersion_Args {
            struct_size: size_of::<PJRT_Client_RuntimeAbiVersion_Args>(),
            extension_start: ptr::null_mut(),
            client,
            abi_version: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_ClientRuntime(ptr::addr_of_mut!(rt_abi).cast()).is_null());
        let mut ex_abi = PJRT_Executable_GetAbiVersion_Args {
            struct_size: size_of::<PJRT_Executable_GetAbiVersion_Args>(),
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
            abi_version: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_ExecutableGet(ptr::addr_of_mut!(ex_abi).cast()).is_null());

        let mut compat = PJRT_AbiVersion_IsCompatible_Args {
            struct_size: size_of::<PJRT_AbiVersion_IsCompatible_Args>(),
            extension_start: ptr::null_mut(),
            abi_version: rt_abi.abi_version,
            other_abi_version: ex_abi.abi_version,
            is_compatible_out: 0,
        };
        assert!(
            PJRT_AbiVersion_RuntimeCompatibleWithExecutable(ptr::addr_of_mut!(compat).cast())
                .is_null()
        );
        assert_eq!(compat.is_compatible_out, 1);

        let mut rt_proto = PJRT_AbiVersion_ToProto_Args {
            struct_size: size_of::<PJRT_AbiVersion_ToProto_Args>(),
            extension_start: ptr::null_mut(),
            abi_version: rt_abi.abi_version,
            serialized_proto: ptr::null(),
            serialized_proto_size: 0,
            serialized_proto_holder: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_RuntimeToProto(ptr::addr_of_mut!(rt_proto).cast()).is_null());
        assert!(rt_proto.serialized_proto_size > 0);
        let mut rt_from = PJRT_AbiVersion_FromProto_Args {
            struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
            extension_start: ptr::null_mut(),
            serialized_proto: rt_proto.serialized_proto,
            serialized_proto_size: rt_proto.serialized_proto_size,
            abi_version_out: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_RuntimeFromProto(ptr::addr_of_mut!(rt_from).cast()).is_null());
        pjrt_serialized_artifact_deleter(rt_proto.serialized_proto_holder);

        // Incompatible foreign platform ABI check returns 0
        use libtpu_core::proto::ProtoSchemaMessage as _;
        let gpu_proto = libtpu_core::proto::ExecutableAbiVersionProto {
            platform: 0x4750_5500,
            version: Vec::from(&b"0.250.0-v4"[..]),
        }
        .encode_proto();
        let mut bad_ex_from = PJRT_AbiVersion_FromProto_Args {
            struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
            extension_start: ptr::null_mut(),
            serialized_proto: gpu_proto.as_ptr().cast(),
            serialized_proto_size: gpu_proto.len(),
            abi_version_out: ptr::null_mut(),
        };
        assert!(
            PJRT_AbiVersion_ExecutableFromProto(ptr::addr_of_mut!(bad_ex_from).cast()).is_null()
        );
        compat.other_abi_version = bad_ex_from.abi_version_out;
        assert!(
            PJRT_AbiVersion_RuntimeCompatibleWithExecutable(ptr::addr_of_mut!(compat).cast())
                .is_null()
        );
        assert_eq!(compat.is_compatible_out, 0);

        for h in [rt_abi.abi_version, rt_from.abi_version_out] {
            let mut del = PJRT_AbiVersion_Destroy_Args {
                struct_size: size_of::<PJRT_AbiVersion_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: h,
            };
            assert!(PJRT_AbiVersion_RuntimeDestroy(ptr::addr_of_mut!(del).cast()).is_null());
        }
        for h in [ex_abi.abi_version, bad_ex_from.abi_version_out] {
            let mut del = PJRT_AbiVersion_Destroy_Args {
                struct_size: size_of::<PJRT_AbiVersion_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: h,
            };
            assert!(PJRT_AbiVersion_ExecutableDestroy(ptr::addr_of_mut!(del).cast()).is_null());
        }

        // 7. PJRT_MultiSlice & PJRT_Megascale (per-handle state + 0.0.53.1 slots)
        let mut ms_create = PJRT_Megascale_CreateMultiSliceConfig_Args {
            struct_size: size_of::<PJRT_Megascale_CreateMultiSliceConfig_Args>(),
            extension_start: ptr::null_mut(),
            topology: (*client).topology_ptr,
            config_out: ptr::null_mut(),
        };
        assert!(
            PJRT_Megascale_CreateMultiSliceConfig(ptr::addr_of_mut!(ms_create).cast()).is_null()
        );
        let ms_cfg = ms_create.config_out;
        assert!(!ms_cfg.is_null());

        let mut ms_ns = PJRT_MultiSlice_Config_NumSlices_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_NumSlices_Args>(),
            extension_start: ptr::null_mut(),
            config: ms_cfg,
            num_slices: 0,
        };
        assert!(PJRT_MultiSlice_Config_NumSlices(ptr::addr_of_mut!(ms_ns).cast()).is_null());
        assert!(ms_ns.num_slices >= 1);

        let mut ms_sid = PJRT_MultiSlice_Config_SliceId_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_SliceId_Args>(),
            extension_start: ptr::null_mut(),
            config: ms_cfg,
            slice_id: 0,
        };
        assert!(PJRT_MultiSlice_Config_SliceId(ptr::addr_of_mut!(ms_sid).cast()).is_null());

        let mut ms_host = PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args {
            struct_size: size_of::<PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args>(),
            config: ms_cfg,
            local_host_id: -1,
            pad: 0,
        };
        assert!(
            PJRT_Megascale_MultiSliceConfig_GetLocalHostId(ptr::addr_of_mut!(ms_host).cast())
                .is_null()
        );
        assert!(ms_host.local_host_id >= 0);

        let mut ms_del = PJRT_MultiSlice_Config_Destroy_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            config: ms_cfg,
        };
        assert!(PJRT_MultiSlice_Config_Destroy(ptr::addr_of_mut!(ms_del).cast()).is_null());

        // 8. PJRT_PhaseCompile & PJRT_Relocatable (0.0.53.1 CompileToRelocatable -> Serialize -> Deserialize -> Link)
        let mut pc_get = PJRT_PhaseCompile_GetCompiler_Args {
            struct_size: size_of::<PJRT_PhaseCompile_GetCompiler_Args>(),
            extension_start: ptr::null_mut(),
            client,
            compiler_out: ptr::null_mut(),
        };
        assert!(PJRT_PhaseCompile_GetCompiler(ptr::addr_of_mut!(pc_get).cast()).is_null());
        let pc = pc_get.compiler_out;

        let mut comp_rel = PJRT_PhaseCompile_CompileToRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_CompileToRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc,
            client,
            program: ptr::addr_of!(prog),
            compile_options: ptr::null(),
            compile_options_size: 0,
            topology: (*client).topology_ptr,
            relocatables: ptr::null(),
            num_relocatables: 0,
            relocatables_buffer: ptr::null_mut(),
        };
        assert!(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(comp_rel).cast()).is_null()
        );
        assert_eq!(comp_rel.num_relocatables, 1);
        let rel0 = *comp_rel.relocatables;

        let mut rel_side = PJRT_Relocatable_HasSideEffects_Args {
            struct_size: size_of::<PJRT_Relocatable_HasSideEffects_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel0,
            has_side_effects: true,
            pad: [0; 7],
        };
        assert!(PJRT_Relocatable_HasSideEffects(ptr::addr_of_mut!(rel_side).cast()).is_null());
        assert!(!rel_side.has_side_effects);

        let mut rel_ser = PJRT_Relocatable_Serialize_Args {
            struct_size: size_of::<PJRT_Relocatable_Serialize_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel0,
            serialized_bytes: ptr::null(),
            serialized_bytes_size: 0,
            serialized_relocatable: ptr::null_mut(),
            serialized_relocatable_deleter: None,
        };
        assert!(PJRT_Relocatable_Serialize(ptr::addr_of_mut!(rel_ser).cast()).is_null());
        assert!(rel_ser.serialized_bytes_size > 0);

        let mut rel_deser = PJRT_PhaseCompile_DeserializeRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_DeserializeRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc,
            serialized_relocatable: rel_ser.serialized_bytes,
            serialized_relocatable_size: rel_ser.serialized_bytes_size,
            relocatable: ptr::null_mut(),
        };
        assert!(
            PJRT_PhaseCompile_DeserializeRelocatable(ptr::addr_of_mut!(rel_deser).cast()).is_null()
        );
        if let Some(del_fn) = rel_ser.serialized_relocatable_deleter {
            del_fn(rel_ser.serialized_relocatable);
        }

        let rel_inputs = [rel_deser.relocatable];
        let mut rel_link = PJRT_PhaseCompile_LinkRelocatables_Args {
            struct_size: size_of::<PJRT_PhaseCompile_LinkRelocatables_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc,
            relocatables: rel_inputs.as_ptr(),
            num_relocatables: 1,
            program: ptr::null(),
            topology: (*client).topology_ptr,
            compile_options: ptr::null(),
            compile_options_size: 0,
            executable: ptr::null_mut(),
        };
        assert!(PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(rel_link).cast()).is_null());
        assert!(!rel_link.executable.is_null());
        let mut del_linked = crate::pjrt_types::PJRT_Executable_Destroy_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: rel_link.executable,
        };
        assert!(
            crate::pjrt_executable_api::PJRT_Executable_Destroy(
                ptr::addr_of_mut!(del_linked).cast()
            )
            .is_null()
        );

        for rel_to_free in [rel0, rel_deser.relocatable] {
            let mut rel_del = PJRT_Relocatable_Destroy_Args {
                struct_size: size_of::<PJRT_Relocatable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                relocatable: rel_to_free,
            };
            assert!(PJRT_Relocatable_Destroy(ptr::addr_of_mut!(rel_del).cast()).is_null());
        }
        let mut cbuf_del = PJRT_PhaseCompile_Destroy_Args {
            struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            handle: comp_rel.relocatables_buffer,
        };
        assert!(PJRT_PhaseCompile_CBuffersDestroy(ptr::addr_of_mut!(cbuf_del).cast()).is_null());
        let mut pc_del = PJRT_PhaseCompile_Destroy_Args {
            struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            handle: pc.cast(),
        };
        assert!(PJRT_PhaseCompile_DestroyCompiler(ptr::addr_of_mut!(pc_del).cast()).is_null());

        // 9. PJRT_Triton_Compile valid MLIR
        let ttir = b"module @triton_kernel { tt.func @add_kernel() { tt.return } }";
        let mut triton_ok = PJRT_Triton_Compile_Args {
            struct_size: size_of::<PJRT_Triton_Compile_Args>(),
            extension_start: ptr::null_mut(),
            module: ttir.as_ptr().cast(),
            module_size: ttir.len(),
            arch_name: c"v6e".as_ptr(),
            arch_name_size: 3,
            num_warps: 4,
            num_ctas: 1,
            num_stages: 2,
            pad: 0,
            out_asm: ptr::null(),
            out_asm_size: 0,
            out_smem_bytes: 0,
        };
        assert!(PJRT_Triton_Compile(ptr::addr_of_mut!(triton_ok).cast()).is_null());
        assert!(triton_ok.out_asm_size > 0 && triton_ok.out_smem_bytes >= 0);

        // 10. PLUGIN_Profiler_Api: Create -> Start -> Stop -> CollectData -> Destroy (0.0.53.1 & negotiated v0.1)
        let prof_api = &GLOBAL_PLUGIN_PROFILER_API;
        let mut prof_create = PLUGIN_Profiler_Create_Args {
            struct_size: size_of::<PLUGIN_Profiler_Create_Args>(),
            extension_start: ptr::null_mut(),
            options: ptr::null(),
            options_size: 0,
            profiler: ptr::null_mut(),
        };
        assert!((prof_api.create)(ptr::addr_of_mut!(prof_create).cast()).is_null());
        let prof_inst = prof_create.profiler;
        assert!(!prof_inst.is_null());

        let mut prof_start = PLUGIN_Profiler_Start_Args {
            struct_size: size_of::<PLUGIN_Profiler_Start_Args>(),
            extension_start: ptr::null_mut(),
            profiler: prof_inst,
        };
        assert!((prof_api.start)(ptr::addr_of_mut!(prof_start).cast()).is_null());
        let trace_buf = upload_f32_buffer(client, &[5.0, 6.0, 7.0, 8.0]);
        let mut tb_del = PJRT_Buffer_Destroy_Args {
            struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            buffer: trace_buf,
        };
        assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(tb_del).cast()).is_null());

        let mut prof_stop = PLUGIN_Profiler_Stop_Args {
            struct_size: size_of::<PLUGIN_Profiler_Stop_Args>(),
            extension_start: ptr::null_mut(),
            profiler: prof_inst,
        };
        assert!((prof_api.stop)(ptr::addr_of_mut!(prof_stop).cast()).is_null());

        let mut xspace_len: usize = 0;
        let mut prof_collect = PLUGIN_Profiler_CollectData_Args {
            struct_size: size_of::<PLUGIN_Profiler_CollectData_Args>(),
            extension_start: ptr::null_mut(),
            profiler: prof_inst,
            buffer: ptr::null_mut(),
            size_in_bytes: ptr::addr_of_mut!(xspace_len),
        };
        assert!((prof_api.collect_data)(ptr::addr_of_mut!(prof_collect).cast()).is_null());
        assert!(xspace_len > 0);
        let mut xspace_buf = vec![0_u8; xspace_len];
        prof_collect.buffer = xspace_buf.as_mut_ptr();
        assert!((prof_api.collect_data)(ptr::addr_of_mut!(prof_collect).cast()).is_null());

        let mut prof_del = PLUGIN_Profiler_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            profiler: prof_inst,
        };
        assert!((prof_api.destroy)(ptr::addr_of_mut!(prof_del).cast()).is_null());

        // Also test NEGOTIATED_PLUGIN_PROFILER_API (32-byte Create/CollectData, 16-byte Start/Stop/Destroy + null paths)
        let neg_api = &NEGOTIATED_PLUGIN_PROFILER_API;
        assert!((neg_api.create)(ptr::null_mut()).is_null());
        assert!((neg_api.start)(ptr::null_mut()).is_null());
        assert!((neg_api.stop)(ptr::null_mut()).is_null());
        assert!((neg_api.collect_data)(ptr::null_mut()).is_null());
        assert!((neg_api.destroy)(ptr::null_mut()).is_null());
        let mut neg_create = [32_usize, 0, 0, 0];
        assert!((neg_api.create)(neg_create.as_mut_ptr().cast()).is_null());
        let neg_inst = neg_create[3];
        assert_ne!(neg_inst, 0);
        let mut neg_op = [16_usize, neg_inst];
        assert!((neg_api.start)(neg_op.as_mut_ptr().cast()).is_null());
        assert!((neg_api.stop)(neg_op.as_mut_ptr().cast()).is_null());
        let mut neg_cd = [32_usize, neg_inst, 0, 0];
        assert!((neg_api.collect_data)(neg_cd.as_mut_ptr().cast()).is_null());
        assert!(neg_cd[2] != 0 && neg_cd[3] > 0);
        assert!((neg_api.destroy)(neg_op.as_mut_ptr().cast()).is_null());
        // Unregistered handle no-ops on start/stop/collect_data
        assert!((neg_api.start)(neg_op.as_mut_ptr().cast()).is_null());
        assert!((neg_api.stop)(neg_op.as_mut_ptr().cast()).is_null());
        assert!((neg_api.collect_data)(neg_cd.as_mut_ptr().cast()).is_null());

        // Cleanup buffers, executable, and client
        for b in [
            raw_backing,
            r0_send,
            r1_send,
            r0_recv4,
            r1_recv4,
            r0_recv2,
            r1_recv2,
            r0_recv8,
            xh_dst,
        ] {
            let mut bd = PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: b,
            };
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(bd).cast()).is_null());
        }
        let mut lex_del = PJRT_LoadedExecutable_Destroy_Args {
            struct_size: size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: loaded_exec,
        };
        assert!(PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(lex_del).cast()).is_null());
        let mut cli_del = PJRT_Client_Destroy_Args {
            struct_size: size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client,
        };
        assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(cli_del).cast()).is_null());
    }
    assert!(extension_call_count() > prev_count);
}

/// Dummy C error handler invoked during Megascale error handler registration tests.
///
/// # Arguments
///
/// * `_ctx` - Opaque callback context pointer.
///
/// # Safety
///
/// Safe to invoke with any pointer value.
unsafe extern "C" fn dummy_megascale_err_cb(_ctx: *mut std::ffi::c_void) {}

/// Dummy user-data deleter recording invocation count for `PJRT_FFI_UserData_Add`.
///
/// # Arguments
///
/// * `data` - Pointer to an `AtomicUsize` counter.
///
/// # Safety
///
/// `data` must point to a valid `std::sync::atomic::AtomicUsize`.
unsafe extern "C" fn dummy_ffi_user_data_deleter(data: *mut std::ffi::c_void) {
    // SAFETY: `data` points to `AtomicUsize` in the test stack frame.
    unsafe {
        (*data.cast::<std::sync::atomic::AtomicUsize>())
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Dummy `OpenXLA` Typed FFI custom-call handler returning null (`OK`).
///
/// # Arguments
///
/// * `_call_frame` - Pointer to `XLA_FFI_CallFrame`.
///
/// # Safety
///
/// Safe to invoke with any pointer.
unsafe extern "C" fn dummy_typed_ffi_handler(
    _call_frame: *mut XLA_FFI_CallFrame,
) -> *mut PJRT_Error {
    ptr::null_mut()
}

/// Dummy runtime callback invoked via `PJRT_Callback_InvokeCallback`.
///
/// # Arguments
///
/// * `_payload` - Callback payload pointer.
/// * `user_arg` - Pointer to a `usize` counter incremented on invocation.
///
/// # Safety
///
/// `user_arg` must point to a writable `usize`.
unsafe extern "C" fn dummy_runtime_cb(
    _payload: *mut std::ffi::c_void,
    user_arg: *mut std::ffi::c_void,
) {
    // SAFETY: `user_arg` points to `usize` in the test stack frame.
    unsafe {
        *user_arg.cast::<usize>() += 1;
    }
}

/// Dummy cross-host transfer descriptor callback recording invocation.
///
/// # Arguments
///
/// * `user_arg` - Pointer to a `usize` counter.
/// * `_desc` - Serialized descriptor bytes pointer.
/// * `desc_len` - Serialized descriptor byte length.
///
/// # Safety
///
/// `user_arg` must point to a writable `usize`.
unsafe extern "C" fn dummy_xh_desc_cb(
    _desc: *const std::os::raw::c_char,
    desc_len: usize,
    user_arg: *mut std::ffi::c_void,
) {
    // SAFETY: `user_arg` points to `usize` in the test stack frame.
    unsafe {
        *user_arg.cast::<usize>() += desc_len;
    }
}

/// Exercises all 33 `PJRT_Megascale_*` slots and all 32 `PJRT_TpuTopology_*` slots
/// across both valid handles and null/error edge cases.
#[test]
fn test_megascale_and_topology_exhaustive_lifecycles() {
    use crate::pjrt_client_api::{
        PJRT_TopologyDescription_Create, PJRT_TopologyDescription_Destroy,
    };
    use crate::pjrt_types::{
        PJRT_TopologyDescription_Create_Args, PJRT_TopologyDescription_Destroy_Args,
    };

    // SAFETY: All FFI argument structs are stack-allocated with exact `struct_size` and valid pointers.
    unsafe {
        let mut client_args = PJRT_Client_Create_Args {
            struct_size: size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: None,
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
        };
        assert!(PJRT_Client_Create(ptr::addr_of_mut!(client_args).cast()).is_null());
        let client = client_args.client;

        let mut topo_create = PJRT_TopologyDescription_Create_Args {
            struct_size: size_of::<PJRT_TopologyDescription_Create_Args>(),
            extension_start: ptr::null_mut(),
            topology_name: c"v5p".as_ptr(),
            topology_name_size: 3,
            create_options: ptr::null(),
            num_options: 0,
            topology: ptr::null_mut(),
        };
        assert!(PJRT_TopologyDescription_Create(ptr::addr_of_mut!(topo_create).cast()).is_null());
        let topo = topo_create.topology.cast_const();

        // --- Megascale 33-slot lifecycle ---
        let mut set_tr = PJRT_Megascale_SetTransportType_Args {
            struct_size: size_of::<PJRT_Megascale_SetTransportType_Args>(),
            extension_start: ptr::null_mut(),
            requested_transport_type: 2,
            transport_type_out: 0,
        };
        assert!(PJRT_Megascale_SetTransportType(ptr::addr_of_mut!(set_tr).cast()).is_null());
        assert_eq!(set_tr.transport_type_out, 2);
        set_tr.requested_transport_type = 99;
        assert!(PJRT_Megascale_SetTransportType(ptr::addr_of_mut!(set_tr).cast()).is_null());
        assert_eq!(set_tr.transport_type_out, 99);

        let mut ctx_create = PJRT_Megascale_CreateClientContextFromPjRtClient_Args {
            struct_size: size_of::<PJRT_Megascale_CreateClientContextFromPjRtClient_Args>(),
            extension_start: ptr::null_mut(),
            client,
            context_out: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_CreateClientContext(ptr::addr_of_mut!(ctx_create).cast()).is_null());
        let ctx1 = ctx_create.context_out;
        assert!(!ctx1.is_null());

        let mut def_cfg = PJRT_Megascale_CreateAoTMegascaleConfig_Args {
            struct_size: size_of::<PJRT_Megascale_CreateAoTMegascaleConfig_Args>(),
            extension_start: ptr::null_mut(),
            topology: topo,
            config_out: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_CreateDefaultConfig(ptr::addr_of_mut!(def_cfg).cast()).is_null());
        let ms_cfg0 = def_cfg.config_out;
        assert!(!ms_cfg0.is_null());

        let mut aot_cfg = PJRT_Megascale_CreateAoTMegascaleConfig_Args {
            struct_size: size_of::<PJRT_Megascale_CreateAoTMegascaleConfig_Args>(),
            extension_start: ptr::null_mut(),
            topology: topo,
            config_out: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_CreateAotConfig(ptr::addr_of_mut!(aot_cfg).cast()).is_null());
        let ms_cfg1 = aot_cfg.config_out;

        let mut ms_create = PJRT_Megascale_CreateMultiSliceConfig_Args {
            struct_size: size_of::<PJRT_Megascale_CreateMultiSliceConfig_Args>(),
            extension_start: ptr::null_mut(),
            topology: ptr::null(),
            config_out: ptr::null_mut(),
        };
        assert!(
            PJRT_Megascale_CreateMultiSliceConfig(ptr::addr_of_mut!(ms_create).cast()).is_null()
        );
        let ms_cfg2 = ms_create.config_out;

        for cfg_ptr in [ms_cfg1.cast_const(), ptr::null()] {
            let mut ns = PJRT_Megascale_ConfigNumSlices_Args {
                struct_size: size_of::<PJRT_Megascale_ConfigNumSlices_Args>(),
                extension_start: ptr::null_mut(),
                config: cfg_ptr,
                num_slices: 0,
            };
            assert!(PJRT_Megascale_ConfigNumSlices(ptr::addr_of_mut!(ns).cast()).is_null());
            assert!(ns.num_slices >= 1);

            let mut nd = PJRT_Megascale_ConfigNumDevicesPerSlice_Args {
                struct_size: size_of::<PJRT_Megascale_ConfigNumDevicesPerSlice_Args>(),
                extension_start: ptr::null_mut(),
                config: cfg_ptr,
                num_devices_per_slice: 0,
            };
            assert!(
                PJRT_Megascale_ConfigNumDevicesPerSlice(ptr::addr_of_mut!(nd).cast()).is_null()
            );
            assert!(nd.num_devices_per_slice >= 1);

            let mut sid = PJRT_Megascale_ConfigSliceId_Args {
                struct_size: size_of::<PJRT_Megascale_ConfigSliceId_Args>(),
                extension_start: ptr::null_mut(),
                config: cfg_ptr,
                slice_id: 99,
            };
            assert!(PJRT_Megascale_ConfigSliceId(ptr::addr_of_mut!(sid).cast()).is_null());

            let mut kv = PJRT_Megascale_ConfigSerializedSliceKeyValue_Args {
                struct_size: size_of::<PJRT_Megascale_ConfigSerializedSliceKeyValue_Args>(),
                extension_start: ptr::null_mut(),
                config: cfg_ptr,
                serialized_kv_out: 0,
                num_entries_out: 0,
            };
            assert!(
                PJRT_Megascale_ConfigSerializedSliceKeyValue(ptr::addr_of_mut!(kv).cast())
                    .is_null()
            );
            assert!(kv.serialized_kv_out > 0);
        }

        // MultiSlice null-config fallbacks + Megascale ConfigSerialize / DeserializeMultiSliceConfig
        let mut ms_ns_null = PJRT_MultiSlice_Config_NumSlices_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_NumSlices_Args>(),
            extension_start: ptr::null_mut(),
            config: ptr::null(),
            num_slices: 0,
        };
        assert!(PJRT_MultiSlice_Config_NumSlices(ptr::addr_of_mut!(ms_ns_null).cast()).is_null());
        let mut ms_del_null = PJRT_MultiSlice_Config_Destroy_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            config: ptr::null_mut(),
        };
        assert!(PJRT_MultiSlice_Config_Destroy(ptr::addr_of_mut!(ms_del_null).cast()).is_null());

        let mut cfg_ser = PJRT_MultiSlice_Config_Serialize_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_Serialize_Args>(),
            extension_start: ptr::null_mut(),
            config: ms_cfg1.cast_const(),
            serialized_bytes: ptr::null(),
            serialized_bytes_size: 0,
            serialized_config: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_ConfigSerialize(ptr::addr_of_mut!(cfg_ser).cast()).is_null());
        pjrt_serialized_artifact_deleter(cfg_ser.serialized_config);

        let mut cfg_deser = PJRT_Megascale_DeserializeMultiSliceConfig_Args {
            struct_size: size_of::<PJRT_Megascale_DeserializeMultiSliceConfig_Args>(),
            extension_start: ptr::null_mut(),
            serialized_bytes: ptr::null(),
            config_out: ptr::null_mut(),
        };
        assert!(
            PJRT_Megascale_DeserializeMultiSliceConfig(ptr::addr_of_mut!(cfg_deser).cast())
                .is_null()
        );
        let ms_cfg3 = cfg_deser.config_out;

        // 0.0.53.1 new Megascale slots (29..32)
        let mut disc = PJRT_Megascale_DiscoverSlices_Args {
            struct_size: size_of::<PJRT_Megascale_DiscoverSlices_Args>(),
            client_context: ctx1,
            multi_slice_config: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_DiscoverSlices(ptr::addr_of_mut!(disc).cast()).is_null());
        let ms_cfg4 = disc.multi_slice_config;

        for (sz_lh, cfg_p) in [(20_usize, ms_cfg1.cast_const()), (24, ptr::null())] {
            let mut inc = PJRT_Megascale_MultiSliceConfig_GetIncarnationId_Args {
                struct_size: size_of::<PJRT_Megascale_MultiSliceConfig_GetIncarnationId_Args>(),
                config: cfg_p,
                incarnation_id: 0,
            };
            assert!(
                PJRT_Megascale_MultiSliceConfig_GetIncarnationId(ptr::addr_of_mut!(inc).cast())
                    .is_null()
            );
            assert!(inc.incarnation_id >= 1);

            let mut lh = PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args {
                struct_size: sz_lh,
                config: cfg_p,
                local_host_id: -1,
                pad: 0,
            };
            assert!(
                PJRT_Megascale_MultiSliceConfig_GetLocalHostId(ptr::addr_of_mut!(lh).cast())
                    .is_null()
            );

            let mut nh = PJRT_Megascale_MultiSliceConfig_GetNumHosts_Args {
                struct_size: sz_lh,
                config: cfg_p,
                num_hosts: 0,
                pad: 0,
            };
            assert!(
                PJRT_Megascale_MultiSliceConfig_GetNumHosts(ptr::addr_of_mut!(nh).cast()).is_null()
            );
            assert!(nh.num_hosts >= 1);
        }
        assert_and_free_pjrt_error(
            PJRT_Megascale_MultiSliceConfig_GetLocalHostId(ptr::null_mut()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Megascale_MultiSliceConfig_GetNumHosts(ptr::null_mut()),
            3,
        );
        let mut undersized_lh = PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args {
            struct_size: 8,
            config: ms_cfg1.cast_const(),
            local_host_id: 0,
            pad: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_MultiSliceConfig_GetLocalHostId(ptr::addr_of_mut!(undersized_lh).cast()),
            3,
        );
        let mut undersized_nh = PJRT_Megascale_MultiSliceConfig_GetNumHosts_Args {
            struct_size: 8,
            config: ms_cfg1.cast_const(),
            num_hosts: 0,
            pad: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_MultiSliceConfig_GetNumHosts(ptr::addr_of_mut!(undersized_nh).cast()),
            3,
        );

        let mut gb = PJRT_Megascale_GlobalBarrier_Args {
            struct_size: size_of::<PJRT_Megascale_GlobalBarrier_Args>(),
            config: ms_cfg1.cast_const(),
            barrier_id: c"step_barrier".as_ptr(),
            barrier_id_size: 12,
            timeout_ms: 1000,
        };
        assert!(PJRT_Megascale_GlobalBarrier(ptr::addr_of_mut!(gb).cast()).is_null());
        gb.barrier_id = ptr::null();
        assert!(PJRT_Megascale_GlobalBarrier(ptr::addr_of_mut!(gb).cast()).is_null());
        gb.timeout_ms = -5;
        assert_and_free_pjrt_error(
            PJRT_Megascale_GlobalBarrier(ptr::addr_of_mut!(gb).cast()),
            3,
        );

        // Remaining Megascale context & barrier & metrics & error-handler slots
        let mut addrs = PJRT_Megascale_Addresses_Args {
            struct_size: size_of::<PJRT_Megascale_Addresses_Args>(),
            extension_start: ptr::null_mut(),
            context: ctx1,
            num_addresses: 0,
        };
        assert!(PJRT_Megascale_Addresses(ptr::addr_of_mut!(addrs).cast()).is_null());

        for c_opt in [ctx1, ptr::null_mut()] {
            let mut bar = PJRT_Megascale_Barrier_Args {
                struct_size: size_of::<PJRT_Megascale_Barrier_Args>(),
                extension_start: ptr::null_mut(),
                context: c_opt,
                completed_barriers: 0,
            };
            assert!(PJRT_Megascale_Barrier(ptr::addr_of_mut!(bar).cast()).is_null());
            assert!(bar.completed_barriers >= 1);

            let mut cm = PJRT_Megascale_CollectMetrics_Args {
                struct_size: size_of::<PJRT_Megascale_CollectMetrics_Args>(),
                extension_start: ptr::null_mut(),
                context: c_opt,
                serialized_metrics_ptr: ptr::null(),
                serialized_metrics_size: 0,
            };
            assert!(PJRT_Megascale_CollectMetrics(ptr::addr_of_mut!(cm).cast()).is_null());
            assert!(!cm.serialized_metrics_ptr.is_null());

            let mut gtr = PJRT_Megascale_GetTransportType_Args {
                struct_size: size_of::<PJRT_Megascale_GetTransportType_Args>(),
                extension_start: ptr::null_mut(),
                context: c_opt,
                transport_type_out: 99,
            };
            assert!(PJRT_Megascale_GetTransportType(ptr::addr_of_mut!(gtr).cast()).is_null());

            let mut rm = PJRT_Megascale_ResetMetrics_Args {
                struct_size: size_of::<PJRT_Megascale_ResetMetrics_Args>(),
                extension_start: ptr::null_mut(),
                handle: c_opt,
            };
            assert!(PJRT_Megascale_ResetMetrics(ptr::addr_of_mut!(rm).cast()).is_null());

            let mut ac = PJRT_Megascale_AbortCollectives_Args {
                struct_size: size_of::<PJRT_Megascale_AbortCollectives_Args>(),
                extension_start: ptr::null_mut(),
                handle: c_opt,
            };
            assert!(PJRT_Megascale_AbortCollectives(ptr::addr_of_mut!(ac).cast()).is_null());
        }

        let mut wbar = PJRT_Megascale_WaitForBarrier_Args {
            struct_size: size_of::<PJRT_Megascale_WaitForBarrier_Args>(),
            extension_start: ptr::null_mut(),
            handle: ctx1,
        };
        assert!(PJRT_Megascale_WaitForBarrier(ptr::addr_of_mut!(wbar).cast()).is_null());

        let mut dcn = PJRT_Megascale_GetDcnTopology_Args {
            struct_size: size_of::<PJRT_Megascale_GetDcnTopology_Args>(),
            extension_start: ptr::null_mut(),
            context: ctx1,
            serialized_size: 0,
        };
        assert!(PJRT_Megascale_GetDcnTopology(ptr::addr_of_mut!(dcn).cast()).is_null());
        assert!(dcn.serialized_size > 0);

        let mut gsid = PJRT_Megascale_GetSliceId_Args {
            struct_size: size_of::<PJRT_Megascale_GetSliceId_Args>(),
            extension_start: ptr::null_mut(),
            context: ctx1,
            slice_id: 99,
        };
        assert!(PJRT_Megascale_GetSliceId(ptr::addr_of_mut!(gsid).cast()).is_null());

        let mut gns = PJRT_Megascale_GetNumSlices_Args {
            struct_size: size_of::<PJRT_Megascale_GetNumSlices_Args>(),
            extension_start: ptr::null_mut(),
            context: ctx1,
            num_slices: 0,
        };
        assert!(PJRT_Megascale_GetNumSlices(ptr::addr_of_mut!(gns).cast()).is_null());

        let mut islice = PJRT_Megascale_InitializeSlice_Args {
            struct_size: size_of::<PJRT_Megascale_InitializeSlice_Args>(),
            extension_start: ptr::null_mut(),
            context: ctx1,
            status_out: 0,
        };
        assert!(PJRT_Megascale_InitializeSlice(ptr::addr_of_mut!(islice).cast()).is_null());
        assert_eq!(islice.status_out, 1);

        let mut fslice = PJRT_Megascale_FinalizeSlice_Args {
            struct_size: size_of::<PJRT_Megascale_FinalizeSlice_Args>(),
            extension_start: ptr::null_mut(),
            handle: ctx1,
        };
        assert!(PJRT_Megascale_FinalizeSlice(ptr::addr_of_mut!(fslice).cast()).is_null());

        let mut reg_eh = PJRT_Megascale_RegisterErrorHandler_Args {
            struct_size: size_of::<PJRT_Megascale_RegisterErrorHandler_Args>(),
            extension_start: ptr::null_mut(),
            callback_fn: Some(dummy_megascale_err_cb),
            handler_id_out: 0,
        };
        assert!(PJRT_Megascale_RegisterErrorHandler(ptr::addr_of_mut!(reg_eh).cast()).is_null());
        let mut unreg_eh = PJRT_Megascale_UnregisterErrorHandler_Args {
            struct_size: size_of::<PJRT_Megascale_UnregisterErrorHandler_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(
            PJRT_Megascale_UnregisterErrorHandler(ptr::addr_of_mut!(unreg_eh).cast()).is_null()
        );

        let mut log_err = PJRT_Megascale_LogError_Args {
            struct_size: size_of::<PJRT_Megascale_LogError_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_LogError(ptr::addr_of_mut!(log_err).cast()).is_null());

        for ms_del_ptr in [ms_cfg0, ms_cfg1, ms_cfg2, ms_cfg3, ms_cfg4, ptr::null_mut()] {
            let mut del_cfg = PJRT_Megascale_DeleteMultiSliceConfig_Args {
                struct_size: size_of::<PJRT_Megascale_DeleteMultiSliceConfig_Args>(),
                extension_start: ptr::null_mut(),
                config: ms_del_ptr,
            };
            assert!(
                PJRT_Megascale_DeleteMultiSliceConfig(ptr::addr_of_mut!(del_cfg).cast()).is_null()
            );
        }
        let mut del_bad_cfg = PJRT_Megascale_DeleteMultiSliceConfig_Args {
            struct_size: size_of::<PJRT_Megascale_DeleteMultiSliceConfig_Args>(),
            extension_start: ptr::null_mut(),
            config: ms_cfg1,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_DeleteMultiSliceConfig(ptr::addr_of_mut!(del_bad_cfg).cast()),
            3,
        );

        for c_del in [ctx1, ptr::null_mut()] {
            let mut del_ctx = PJRT_Megascale_DeleteClientContext_Args {
                struct_size: size_of::<PJRT_Megascale_DeleteClientContext_Args>(),
                extension_start: ptr::null_mut(),
                context: c_del,
            };
            assert!(
                PJRT_Megascale_DeleteClientContext(ptr::addr_of_mut!(del_ctx).cast()).is_null()
            );
        }

        // --- PJRT_TpuTopology 32-slot lifecycle (with both valid `topo` and `null` fallback) ---
        for t_ptr in [topo, ptr::null()] {
            let mut sc = PJRT_TpuTopology_ScalarQuery_Args {
                struct_size: size_of::<PJRT_TpuTopology_ScalarQuery_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                out_value: 7,
            };
            assert!(PJRT_TpuTopology_Subslice(ptr::addr_of_mut!(sc).cast()).is_null());
            sc.topology = t_ptr;
            assert!(PJRT_TpuTopology_ReplaceHostBounds(ptr::addr_of_mut!(sc).cast()).is_null());
            sc.topology = t_ptr;
            sc.out_value = 7;
            assert!(
                PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId(ptr::addr_of_mut!(sc).cast())
                    .is_null()
            );

            let mut hli = PJRT_TpuTopology_ScalarQuery_Args {
                struct_size: size_of::<PJRT_TpuTopology_ScalarQuery_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                out_value: 0,
            };
            assert!(
                PJRT_TpuTopology_HasLimitedIciConnectivity(ptr::addr_of_mut!(hli).cast()).is_null()
            );
            assert!(
                PJRT_TpuTopology_IsEnhancedBarrierEnabled(ptr::addr_of_mut!(hli).cast()).is_null()
            );
            assert!(
                PJRT_TpuTopology_IsReachableOverLimitedIci(ptr::addr_of_mut!(hli).cast()).is_null()
            );
            assert!(PJRT_TpuTopology_IsSubsliceTopology(ptr::addr_of_mut!(hli).cast()).is_null());

            let mut pc = PJRT_TpuTopology_ProcessCount_Args {
                struct_size: size_of::<PJRT_TpuTopology_ProcessCount_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                process_count: 0,
            };
            assert!(PJRT_TpuTopology_ProcessCount(ptr::addr_of_mut!(pc).cast()).is_null());

            let mut cpp = PJRT_TpuTopology_ChipsPerProcess_Args {
                struct_size: size_of::<PJRT_TpuTopology_ChipsPerProcess_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                chips_per_process: 0,
            };
            assert!(PJRT_TpuTopology_ChipsPerProcess(ptr::addr_of_mut!(cpp).cast()).is_null());

            let mut ccpc = PJRT_TpuTopology_CoreCountPerChip_Args {
                struct_size: size_of::<PJRT_TpuTopology_CoreCountPerChip_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                core_count_per_chip: 0,
            };
            assert!(PJRT_TpuTopology_CoreCountPerChip(ptr::addr_of_mut!(ccpc).cast()).is_null());

            let mut cc = PJRT_TpuTopology_ChipCount_Args {
                struct_size: size_of::<PJRT_TpuTopology_ChipCount_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                chip_count: 0,
            };
            assert!(PJRT_TpuTopology_ChipCount(ptr::addr_of_mut!(cc).cast()).is_null());

            let mut corec = PJRT_TpuTopology_CoreCount_Args {
                struct_size: size_of::<PJRT_TpuTopology_CoreCount_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                core_count: 0,
            };
            assert!(PJRT_TpuTopology_CoreCount(ptr::addr_of_mut!(corec).cast()).is_null());

            let mut ldpp = PJRT_TpuTopology_LogiDeviceCountPerProcess_Args {
                struct_size: size_of::<PJRT_TpuTopology_LogiDeviceCountPerProcess_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                logi_device_count_per_process: 0,
            };
            assert!(
                PJRT_TpuTopology_LogiDeviceCountPerProcess(ptr::addr_of_mut!(ldpp).cast())
                    .is_null()
            );

            let mut ldc = PJRT_TpuTopology_LogiDeviceCount_Args {
                struct_size: size_of::<PJRT_TpuTopology_LogiDeviceCount_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                logi_device_count: 0,
            };
            assert!(PJRT_TpuTopology_LogiDeviceCount(ptr::addr_of_mut!(ldc).cast()).is_null());

            let mut ldpc = PJRT_TpuTopology_LogiDeviceCountPerChip_Args {
                struct_size: size_of::<PJRT_TpuTopology_LogiDeviceCountPerChip_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                logi_device_count_per_chip: 0,
            };
            assert!(
                PJRT_TpuTopology_LogiDeviceCountPerChip(ptr::addr_of_mut!(ldpc).cast()).is_null()
            );

            let mut ccpp = PJRT_TpuTopology_CoreCountPerProcess_Args {
                struct_size: size_of::<PJRT_TpuTopology_CoreCountPerProcess_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                core_count_per_process: 0,
            };
            assert!(PJRT_TpuTopology_CoreCountPerProcess(ptr::addr_of_mut!(ccpp).cast()).is_null());

            let mut coords_buf = [0_i32; 3];
            for c_out in [coords_buf.as_mut_ptr(), ptr::null_mut()] {
                let mut pb = PJRT_TpuTopology_ProcessBounds_Args {
                    struct_size: size_of::<PJRT_TpuTopology_ProcessBounds_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    coords_out: c_out,
                    num_coords: 0,
                };
                assert!(PJRT_TpuTopology_ProcessBounds(ptr::addr_of_mut!(pb).cast()).is_null());
                assert!(
                    PJRT_TpuTopology_ChipsPerProcessBounds(ptr::addr_of_mut!(pb).cast()).is_null()
                );
                assert!(PJRT_TpuTopology_ChipBounds(ptr::addr_of_mut!(pb).cast()).is_null());

                let mut pcfi = PJRT_TpuTopology_ProcessCoordFromId_Args {
                    struct_size: size_of::<PJRT_TpuTopology_ProcessCoordFromId_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    process_id: 0,
                    coords_out: c_out,
                    num_coords: 0,
                };
                assert!(
                    PJRT_TpuTopology_ProcessCoordFromId(ptr::addr_of_mut!(pcfi).cast()).is_null()
                );

                for dev_id in [0_usize, 999] {
                    let mut cc_ld = PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args {
                        struct_size: size_of::<PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice_Args>(
                        ),
                        extension_start: ptr::null_mut(),
                        topology: t_ptr,
                        device_id: dev_id,
                        coords_out: c_out,
                        core_idx_out: 0,
                    };
                    assert!(
                        PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice(
                            ptr::addr_of_mut!(cc_ld).cast()
                        )
                        .is_null()
                    );
                }
            }

            let in_coords = [1_i32, 0, 0];
            for c_in in [in_coords.as_ptr(), ptr::null()] {
                let mut cifc = PJRT_TpuTopology_ChipIdFromCoord_Args {
                    struct_size: size_of::<PJRT_TpuTopology_ChipIdFromCoord_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    coords: c_in,
                    chip_id_out: 0,
                    idx_on_proc_out: 0,
                };
                assert!(PJRT_TpuTopology_ChipIdFromCoord(ptr::addr_of_mut!(cifc).cast()).is_null());

                let mut ldfc = PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args {
                    struct_size: size_of::<PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    coords: c_in,
                    core_idx: 1,
                    device_id_out: 0,
                };
                assert!(
                    PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(
                        ptr::addr_of_mut!(ldfc).cast()
                    )
                    .is_null()
                );
            }

            let mut pfc = PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args {
                struct_size: size_of::<PJRT_TpuTopology_ProcIdAndIdxOnProcForChip_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                chip_id: 1,
                out_proc_id: 0,
                out_idx_on_proc: 0,
            };
            assert!(
                PJRT_TpuTopology_ProcIdAndIdxOnProcForChip(ptr::addr_of_mut!(pfc).cast()).is_null()
            );

            let mut pfld = PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args {
                struct_size: size_of::<PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                device_id: 1,
                out_proc_id: 0,
                out_idx_on_proc: 0,
            };
            assert!(
                PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice(ptr::addr_of_mut!(pfld).cast())
                    .is_null()
            );

            let mut ids_buf = [0_i32; 16];
            for ids_ptr in [ids_buf.as_mut_ptr(), ptr::null_mut()] {
                let mut pids = PJRT_TpuTopology_ProcessIds_Args {
                    struct_size: size_of::<PJRT_TpuTopology_ProcessIds_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    process_ids_out: ids_ptr,
                };
                assert!(PJRT_TpuTopology_ProcessIds(ptr::addr_of_mut!(pids).cast()).is_null());

                let mut dids = PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args {
                    struct_size: size_of::<PJRT_TpuTopology_LogiDeviceIdsOnProcess_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    device_ids_out: ids_ptr,
                };
                assert!(
                    PJRT_TpuTopology_LogiDeviceIdsOnProcess(ptr::addr_of_mut!(dids).cast())
                        .is_null()
                );
            }

            let mut scfg = PJRT_TpuTopology_SliceConfig {
                name: ptr::null(),
                name_len: 0,
                dimensions: [0; 4],
                wraps: [false; 4],
                num_dimensions: 0,
                twist: false,
                pad: [0; 7],
            };
            for sc_ptr in [ptr::addr_of_mut!(scfg), ptr::null_mut()] {
                let mut gsc = PJRT_TpuTopology_GetSliceConfig_Args {
                    struct_size: size_of::<PJRT_TpuTopology_GetSliceConfig_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    slice_config_out: sc_ptr,
                };
                assert!(PJRT_TpuTopology_GetSliceConfig(ptr::addr_of_mut!(gsc).cast()).is_null());
                let mut gscs = PJRT_TpuTopology_GetSliceConfigs_Args {
                    struct_size: size_of::<PJRT_TpuTopology_GetSliceConfigs_Args>(),
                    extension_start: ptr::null_mut(),
                    topology: t_ptr,
                    slice_configs_out: sc_ptr,
                };
                assert!(PJRT_TpuTopology_GetSliceConfigs(ptr::addr_of_mut!(gscs).cast()).is_null());
            }

            let mut dpc = PJRT_TpuTopology_GetDefaultPlatformConfig_Args {
                struct_size: size_of::<PJRT_TpuTopology_GetDefaultPlatformConfig_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                platform_config: 0,
            };
            assert!(
                PJRT_TpuTopology_GetDefaultPlatformConfig(ptr::addr_of_mut!(dpc).cast()).is_null()
            );

            let mut grs = PJRT_TpuTopology_GetRoutingStrategy_Args {
                struct_size: size_of::<PJRT_TpuTopology_GetRoutingStrategy_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                routing_strategy: 99,
            };
            assert!(PJRT_TpuTopology_GetRoutingStrategy(ptr::addr_of_mut!(grs).cast()).is_null());

            let dev_list = [0_i32, 1];
            let sub_dims = [1_i32, 1, 1];
            let mut origin_out = [0_i32; 3];
            let mut csub = PJRT_TpuTopology_ComputeSubsliceForDevices_Args {
                struct_size: size_of::<PJRT_TpuTopology_ComputeSubsliceForDevices_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_ptr,
                subslice_dims: sub_dims.as_ptr(),
                num_subslice_dims: sub_dims.len(),
                device_ids: dev_list.as_ptr(),
                num_devices: dev_list.len(),
                subslice_origin_out: origin_out.as_mut_ptr(),
            };
            assert!(
                PJRT_TpuTopology_ComputeSubsliceForDevices(ptr::addr_of_mut!(csub).cast())
                    .is_null()
            );
            csub.device_ids = ptr::null();
            csub.num_devices = 0;
            csub.subslice_dims = ptr::null();
            csub.num_subslice_dims = 0;
            csub.subslice_origin_out = ptr::null_mut();
            assert!(
                PJRT_TpuTopology_ComputeSubsliceForDevices(ptr::addr_of_mut!(csub).cast())
                    .is_null()
            );
        }

        let mut topo_del = PJRT_TopologyDescription_Destroy_Args {
            struct_size: size_of::<PJRT_TopologyDescription_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            topology: topo.cast_mut(),
        };
        assert!(PJRT_TopologyDescription_Destroy(ptr::addr_of_mut!(topo_del).cast()).is_null());
        let mut cli_del = PJRT_Client_Destroy_Args {
            struct_size: size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client,
        };
        assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(cli_del).cast()).is_null());
    }
}

/// Exercises all remaining functions and branches across `extensions.rs`, `compiler.rs`,
/// `executable.rs`, `memory.rs`, `collectives.rs`, and `profiler.rs`.
#[test]
fn test_compiler_executable_memory_collectives_profiler_exhaustive() {
    use libtpu_core::proto::ProtoSchemaMessage as _;

    // SAFETY: Exercising all remaining PJRT extension C ABI entrypoints with valid stack structs.
    unsafe {
        // 1. `extensions.rs` helpers (`PJRT_Extension_ValidateArgs`, `free_phase_compile_buffer`, negotiated profiler)
        assert_and_free_pjrt_error(PJRT_Extension_ValidateArgs(ptr::null_mut()), 3);
        let mut bad_val_sz = 4_usize;
        assert_and_free_pjrt_error(
            PJRT_Extension_ValidateArgs(ptr::addr_of_mut!(bad_val_sz).cast()),
            3,
        );
        let mut ok_val = [16_usize, 0_usize];
        assert!(PJRT_Extension_ValidateArgs(ok_val.as_mut_ptr().cast()).is_null());
        free_phase_compile_buffer(ptr::null_mut());

        // Client + Executable setup
        let mut client_args = PJRT_Client_Create_Args {
            struct_size: size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: None,
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
        };
        assert!(PJRT_Client_Create(ptr::addr_of_mut!(client_args).cast()).is_null());
        let client = client_args.client;

        let mlir_code = b"module @ext_add { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }";
        let prog = PJRT_Program {
            struct_size: size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: mlir_code.as_ptr().cast_mut().cast(),
            code_size: mlir_code.len(),
            format: c"mlir".as_ptr(),
            format_size: 4,
        };
        let mut c_args = crate::pjrt_types::PJRT_Client_Compile_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Client_Compile_Args>(),
            extension_start: ptr::null_mut(),
            client,
            program: ptr::addr_of!(prog),
            compile_options: ptr::null(),
            compile_options_size: 0,
            executable: ptr::null_mut(),
        };
        assert!(crate::PJRT_Client_Compile(ptr::addr_of_mut!(c_args).cast()).is_null());
        let loaded_exec = c_args.executable;
        let mut get_ex = crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args>(),
            extension_start: ptr::null_mut(),
            loaded_executable: loaded_exec,
            executable: ptr::null_mut(),
        };
        assert!(
            crate::pjrt_executable_api::PJRT_LoadedExecutable_GetExecutable(
                ptr::addr_of_mut!(get_ex).cast()
            )
            .is_null()
        );
        let exec_ptr = get_ex.executable;

        // 2. Compiler / PhaseCompile / Relocatable / Triton / FFI / Callback / XlaTransform
        assert_and_free_pjrt_error(PJRT_XlaTransform_List(ptr::null_mut()), 3);
        let mut bad_list = PJRT_XlaTransform_List_Args {
            struct_size: 8,
            extension_start: ptr::null_mut(),
            names_out: ptr::null(),
            num_names_out: 0,
            name_sizes_out: ptr::null(),
        };
        assert_and_free_pjrt_error(
            PJRT_XlaTransform_List(ptr::addr_of_mut!(bad_list).cast()),
            3,
        );
        let mut wait_all = PJRT_XlaTransform_WaitForAll_Args {
            struct_size: size_of::<PJRT_XlaTransform_WaitForAll_Args>(),
            extension_start: ptr::null_mut(),
            client,
        };
        assert!(PJRT_XlaTransform_WaitForAll(ptr::addr_of_mut!(wait_all).cast()).is_null());

        let bad_utf8 = [0xff_u8, 0xfe];
        let mut reg_bad_utf8 = PJRT_XlaTransform_Register_Args {
            struct_size: size_of::<PJRT_XlaTransform_Register_Args>(),
            extension_start: ptr::null_mut(),
            name: bad_utf8.as_ptr().cast(),
            name_size: 2,
            transform_fn: 1,
        };
        assert_and_free_pjrt_error(
            PJRT_XlaTransform_Register(ptr::addr_of_mut!(reg_bad_utf8).cast()),
            3,
        );
        for i in 0..66_usize {
            let mut l32 = PJRT_XlaTransform_List_Args {
                struct_size: if i % 2 == 0 { 32 } else { 40 },
                extension_start: ptr::null_mut(),
                names_out: ptr::null(),
                num_names_out: 0,
                name_sizes_out: ptr::null(),
            };
            assert!(PJRT_XlaTransform_List(ptr::addr_of_mut!(l32).cast()).is_null());
        }

        // Callback register (`Some` & `None`) + invoke (`Some` & `None` & null client)
        let mut cb_hits = 0_usize;
        let mut cb_reg = PJRT_Callback_Register_Args {
            struct_size: size_of::<PJRT_Callback_Register_Args>(),
            extension_start: ptr::null_mut(),
            client,
            callback_type: 7,
            callback_fn: Some(dummy_runtime_cb),
            user_arg: ptr::addr_of_mut!(cb_hits).cast(),
        };
        assert!(PJRT_Callback_RegisterCallback(ptr::addr_of_mut!(cb_reg).cast()).is_null());
        let mut cb_inv = PJRT_Callback_Invoke_Args {
            struct_size: size_of::<PJRT_Callback_Invoke_Args>(),
            extension_start: ptr::null_mut(),
            client,
            callback_type: 7,
            payload: ptr::null_mut(),
        };
        assert!(PJRT_Callback_InvokeCallback(ptr::addr_of_mut!(cb_inv).cast()).is_null());
        assert_eq!(cb_hits, 1);
        cb_reg.callback_fn = None;
        assert!(PJRT_Callback_RegisterCallback(ptr::addr_of_mut!(cb_reg).cast()).is_null());
        assert!(PJRT_Callback_InvokeCallback(ptr::addr_of_mut!(cb_inv).cast()).is_null());
        assert_eq!(cb_hits, 1);
        cb_reg.client = ptr::null_mut();
        assert!(PJRT_Callback_RegisterCallback(ptr::addr_of_mut!(cb_reg).cast()).is_null());
        cb_inv.client = ptr::null_mut();
        assert!(PJRT_Callback_InvokeCallback(ptr::addr_of_mut!(cb_inv).cast()).is_null());

        // PhaseCompile GetCompiler / GetPhaseNames / RunPhase / CBuffersDestroy / DestroyCompiler
        let mut get_pc = PJRT_PhaseCompile_GetCompiler_Args {
            struct_size: size_of::<PJRT_PhaseCompile_GetCompiler_Args>(),
            extension_start: ptr::null_mut(),
            client,
            compiler_out: ptr::null_mut(),
        };
        assert!(PJRT_PhaseCompile_GetCompiler(ptr::addr_of_mut!(get_pc).cast()).is_null());
        let pc_handle = get_pc.compiler_out;

        let mut pnames = PJRT_PhaseCompile_GetPhaseNames_Args {
            struct_size: size_of::<PJRT_PhaseCompile_GetPhaseNames_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc_handle.cast_const(),
            phase_names: ptr::null(),
            phase_names_sizes: ptr::null(),
            num_phase_names: 0,
        };
        assert!(PJRT_PhaseCompile_GetPhaseNames(ptr::addr_of_mut!(pnames).cast()).is_null());
        assert_eq!(pnames.num_phase_names, 5);

        let hlo_in = b"HloModule p\nENTRY main {\n  p0 = f32[4] parameter(0)\n  ROOT r = f32[4] add(p0, p0)\n}\n";
        let mut run_p = PJRT_PhaseCompile_RunPhase_Args {
            struct_size: size_of::<PJRT_PhaseCompile_RunPhase_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc_handle,
            input_bytes: hlo_in.as_ptr().cast(),
            input_size: hlo_in.len(),
            phase_name: c"stablehlo_to_hlo".as_ptr(),
            output_status: 0,
        };
        assert!(PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(run_p).cast()).is_null());
        assert!(run_p.output_status > 0);
        run_p.phase_compiler = ptr::null_mut();
        run_p.input_bytes = ptr::null();
        run_p.input_size = 0;
        run_p.phase_name = ptr::null();
        assert!(PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(run_p).cast()).is_null());

        // Relocatable remaining queries & error branches
        let rel_prog = PJRT_Program {
            struct_size: size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: hlo_in.as_ptr().cast_mut().cast(),
            code_size: hlo_in.len(),
            format: c"hlo".as_ptr(),
            format_size: 3,
        };
        let mut comp_rel = PJRT_PhaseCompile_CompileToRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_CompileToRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc_handle.cast_const(),
            client,
            program: ptr::addr_of!(rel_prog),
            compile_options: ptr::null(),
            compile_options_size: 0,
            topology: ptr::null(),
            relocatables: ptr::null(),
            num_relocatables: 0,
            relocatables_buffer: ptr::null_mut(),
        };
        assert!(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(comp_rel).cast()).is_null()
        );
        assert_eq!(comp_rel.num_relocatables, 1);
        let rel = *comp_rel.relocatables;

        let mut comp_rel_null_prog = PJRT_PhaseCompile_CompileToRelocatable_Args {
            program: ptr::null(),
            ..comp_rel
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(comp_rel_null_prog).cast()),
            3,
        );

        let mut str_q = PJRT_Relocatable_StringQuery_Args {
            struct_size: size_of::<PJRT_Relocatable_StringQuery_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel.cast_const(),
            value_ptr: ptr::null(),
            value_size: 0,
        };
        assert!(PJRT_Relocatable_BuildId(ptr::addr_of_mut!(str_q).cast()).is_null());
        assert!(PJRT_Relocatable_CallBackendConfig(ptr::addr_of_mut!(str_q).cast()).is_null());
        str_q.relocatable = ptr::null();
        assert_and_free_pjrt_error(PJRT_Relocatable_Name(ptr::addr_of_mut!(str_q).cast()), 3);
        assert_and_free_pjrt_error(PJRT_Relocatable_BuildId(ptr::addr_of_mut!(str_q).cast()), 3);
        assert_and_free_pjrt_error(
            PJRT_Relocatable_CallBackendConfig(ptr::addr_of_mut!(str_q).cast()),
            3,
        );

        let mut side_fx = PJRT_Relocatable_HasSideEffects_Args {
            struct_size: size_of::<PJRT_Relocatable_HasSideEffects_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel.cast_const(),
            has_side_effects: true,
            pad: [0; 7],
        };
        assert!(PJRT_Relocatable_HasSideEffects(ptr::addr_of_mut!(side_fx).cast()).is_null());
        assert!(!side_fx.has_side_effects);
        side_fx.relocatable = ptr::null();
        assert_and_free_pjrt_error(
            PJRT_Relocatable_HasSideEffects(ptr::addr_of_mut!(side_fx).cast()),
            3,
        );
        assert_and_free_pjrt_error(PJRT_Relocatable_HasSideEffects(ptr::null_mut()), 3);
        let mut undersized_sfx = PJRT_Relocatable_HasSideEffects_Args {
            struct_size: 8,
            ..side_fx
        };
        assert_and_free_pjrt_error(
            PJRT_Relocatable_HasSideEffects(ptr::addr_of_mut!(undersized_sfx).cast()),
            3,
        );

        let mut et = PJRT_Relocatable_ElementTypes_Args {
            struct_size: size_of::<PJRT_Relocatable_ElementTypes_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel.cast_const(),
            element_types: ptr::null(),
            num_types: 0,
        };
        assert!(PJRT_Relocatable_ParameterElementTypes(ptr::addr_of_mut!(et).cast()).is_null());
        assert!(PJRT_Relocatable_OutputElementTypes(ptr::addr_of_mut!(et).cast()).is_null());
        et.relocatable = ptr::null();
        assert_and_free_pjrt_error(
            PJRT_Relocatable_ParameterElementTypes(ptr::addr_of_mut!(et).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Relocatable_OutputElementTypes(ptr::addr_of_mut!(et).cast()),
            3,
        );

        let mut dims_q = PJRT_Relocatable_Dimensions_Args {
            struct_size: size_of::<PJRT_Relocatable_Dimensions_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel.cast_const(),
            num_items: 0,
            dims: ptr::null(),
            dim_sizes: ptr::null(),
        };
        assert!(PJRT_Relocatable_ParameterDimensions(ptr::addr_of_mut!(dims_q).cast()).is_null());
        assert!(PJRT_Relocatable_OutputDimensions(ptr::addr_of_mut!(dims_q).cast()).is_null());
        dims_q.relocatable = ptr::null();
        assert_and_free_pjrt_error(
            PJRT_Relocatable_ParameterDimensions(ptr::addr_of_mut!(dims_q).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Relocatable_OutputDimensions(ptr::addr_of_mut!(dims_q).cast()),
            3,
        );

        let mut lay_q = PJRT_Relocatable_Layouts_Args {
            struct_size: size_of::<PJRT_Relocatable_Layouts_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel.cast_const(),
            num_layouts: 0,
            layouts: ptr::null(),
        };
        assert!(PJRT_Relocatable_ParameterLayouts(ptr::addr_of_mut!(lay_q).cast()).is_null());
        assert!(PJRT_Relocatable_OutputLayouts(ptr::addr_of_mut!(lay_q).cast()).is_null());
        lay_q.relocatable = ptr::null();
        assert_and_free_pjrt_error(
            PJRT_Relocatable_ParameterLayouts(ptr::addr_of_mut!(lay_q).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Relocatable_OutputLayouts(ptr::addr_of_mut!(lay_q).cast()),
            3,
        );

        let mut alias_q = PJRT_Relocatable_ParameterOutputAliases_Args {
            struct_size: size_of::<PJRT_Relocatable_ParameterOutputAliases_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel.cast_const(),
            num_parameters: 99,
            parameter_output_aliases: ptr::null(),
        };
        assert!(
            PJRT_Relocatable_ParameterOutputAliases(ptr::addr_of_mut!(alias_q).cast()).is_null()
        );
        alias_q.relocatable = ptr::null();
        assert_and_free_pjrt_error(
            PJRT_Relocatable_ParameterOutputAliases(ptr::addr_of_mut!(alias_q).cast()),
            3,
        );

        let mut rel_ser_bad = PJRT_Relocatable_Serialize_Args {
            struct_size: size_of::<PJRT_Relocatable_Serialize_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: ptr::null(),
            serialized_bytes: ptr::null(),
            serialized_bytes_size: 0,
            serialized_relocatable: ptr::null_mut(),
            serialized_relocatable_deleter: None,
        };
        assert_and_free_pjrt_error(
            PJRT_Relocatable_Serialize(ptr::addr_of_mut!(rel_ser_bad).cast()),
            3,
        );

        let mut deser_bad = PJRT_PhaseCompile_DeserializeRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_DeserializeRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc_handle.cast_const(),
            serialized_relocatable: ptr::null(),
            serialized_relocatable_size: 0,
            relocatable: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_DeserializeRelocatable(ptr::addr_of_mut!(deser_bad).cast()),
            3,
        );

        let mut link_bad = PJRT_PhaseCompile_LinkRelocatables_Args {
            struct_size: size_of::<PJRT_PhaseCompile_LinkRelocatables_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc_handle.cast_const(),
            relocatables: ptr::null(),
            num_relocatables: 0,
            program: ptr::null(),
            compile_options: ptr::null(),
            compile_options_size: 0,
            topology: ptr::null(),
            executable: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(link_bad).cast()),
            3,
        );
        let null_rel_arr: [*mut crate::pjrt_types::PJRT_Relocatable; 1] = [ptr::null_mut()];
        link_bad.relocatables = null_rel_arr.as_ptr();
        link_bad.num_relocatables = 1;
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(link_bad).cast()),
            3,
        );
        let mixed_rel_arr = [rel, ptr::null_mut()];
        link_bad.relocatables = mixed_rel_arr.as_ptr();
        link_bad.num_relocatables = 2;
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(link_bad).cast()),
            3,
        );

        let mut cbuf_del = PJRT_PhaseCompile_Destroy_Args {
            struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            handle: comp_rel.relocatables_buffer,
        };
        assert!(PJRT_PhaseCompile_CBuffersDestroy(ptr::addr_of_mut!(cbuf_del).cast()).is_null());

        let mut rel_del = PJRT_Relocatable_Destroy_Args {
            struct_size: size_of::<PJRT_Relocatable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel,
        };
        assert!(PJRT_Relocatable_Destroy(ptr::addr_of_mut!(rel_del).cast()).is_null());
        assert!(PJRT_Relocatable_Destroy(ptr::addr_of_mut!(rel_del).cast()).is_null());

        let mut pc_del = PJRT_PhaseCompile_Destroy_Args {
            struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            handle: pc_handle.cast(),
        };
        assert!(PJRT_PhaseCompile_DestroyCompiler(ptr::addr_of_mut!(pc_del).cast()).is_null());
        pc_del.handle = ptr::null_mut();
        assert!(PJRT_PhaseCompile_DestroyCompiler(ptr::addr_of_mut!(pc_del).cast()).is_null());

        // Triton invalid UTF-8 & unbalanced braces & FreeAsm(null, 0)
        let mut tr_bad_utf8 = PJRT_Triton_Compile_Args {
            struct_size: size_of::<PJRT_Triton_Compile_Args>(),
            extension_start: ptr::null_mut(),
            module: bad_utf8.as_ptr().cast(),
            module_size: 2,
            arch_name: c"v6e".as_ptr(),
            arch_name_size: 3,
            num_warps: 4,
            num_ctas: 1,
            num_stages: 2,
            pad: 0,
            out_asm: ptr::null(),
            out_asm_size: 0,
            out_smem_bytes: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Triton_Compile(ptr::addr_of_mut!(tr_bad_utf8).cast()),
            13,
        );
        let unbal = b"module { func.func @main() {";
        tr_bad_utf8.module = unbal.as_ptr().cast();
        tr_bad_utf8.module_size = unbal.len();
        assert_and_free_pjrt_error(
            PJRT_Triton_Compile(ptr::addr_of_mut!(tr_bad_utf8).cast()),
            13,
        );
        PJRT_Triton_FreeAsm(ptr::null(), 0);

        // OpenXLA FFI 3-slot error & 40/48/52/64-byte branches
        assert_and_free_pjrt_error(PJRT_FFI_TypeID_Register(ptr::null_mut()), 3);
        assert_and_free_pjrt_error(PJRT_FFI_UserData_Add(ptr::null_mut()), 3);
        assert_and_free_pjrt_error(PJRT_FFI_Register_Handler(ptr::null_mut()), 3);

        let mut ffi_tid_bad = PJRT_FFI_TypeID_Register_Args {
            struct_size: 8,
            extension_start: ptr::null_mut(),
            type_name: c"my_type".as_ptr(),
            type_name_size: 7,
            type_id: 0,
            type_info: ptr::null(),
        };
        assert_and_free_pjrt_error(
            PJRT_FFI_TypeID_Register(ptr::addr_of_mut!(ffi_tid_bad).cast()),
            3,
        );

        let del_counter = std::sync::atomic::AtomicUsize::new(0);
        let mut ffi_ud = PJRT_FFI_UserData_Add_Args {
            struct_size: 8,
            extension_start: ptr::null_mut(),
            context: ptr::null_mut(),
            type_id: 1,
            data: ptr::addr_of!(del_counter).cast_mut().cast(),
            deleter: Some(dummy_ffi_user_data_deleter),
        };
        assert_and_free_pjrt_error(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ffi_ud).cast()), 3);
        ffi_ud.struct_size = 40;
        assert!(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ffi_ud).cast()).is_null());
        ffi_ud.struct_size = 48;
        assert!(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ffi_ud).cast()).is_null());
        assert_eq!(del_counter.load(std::sync::atomic::Ordering::Relaxed), 1);

        let mut ffi_rh = PJRT_FFI_Register_Handler_Args {
            struct_size: 8,
            extension_start: ptr::null_mut(),
            target_name: c"custom_op".as_ptr(),
            target_name_size: 9,
            handler: Some(dummy_typed_ffi_handler),
            platform_name: c"TPU".as_ptr(),
            platform_name_size: 3,
            traits: 0,
            pad: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_FFI_Register_Handler(ptr::addr_of_mut!(ffi_rh).cast()),
            3,
        );
        ffi_rh.struct_size = 52;
        assert!(PJRT_FFI_Register_Handler(ptr::addr_of_mut!(ffi_rh).cast()).is_null());
        ffi_rh.struct_size = 64;
        assert!(PJRT_FFI_Register_Handler(ptr::addr_of_mut!(ffi_rh).cast()).is_null());

        // 3. AbiVersion, Shardings & TpuExecutable exhaustive branches
        let mut r_abi_null = PJRT_Client_RuntimeAbiVersion_Args {
            struct_size: size_of::<PJRT_Client_RuntimeAbiVersion_Args>(),
            extension_start: ptr::null_mut(),
            client: ptr::null(),
            abi_version: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_ClientRuntime(ptr::addr_of_mut!(r_abi_null).cast()).is_null());
        let mut e_abi_null = PJRT_Executable_GetAbiVersion_Args {
            struct_size: size_of::<PJRT_Executable_GetAbiVersion_Args>(),
            extension_start: ptr::null_mut(),
            executable: ptr::null(),
            abi_version: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_ExecutableGet(ptr::addr_of_mut!(e_abi_null).cast()).is_null());

        let mut compat_exec = PJRT_AbiVersion_IsCompatible_Args {
            struct_size: size_of::<PJRT_AbiVersion_IsCompatible_Args>(),
            extension_start: ptr::null_mut(),
            abi_version: r_abi_null.abi_version,
            other_abi_version: e_abi_null.abi_version,
            is_compatible_out: 0,
        };
        assert!(
            PJRT_AbiVersion_RuntimeCompatibleWithExecutable(ptr::addr_of_mut!(compat_exec).cast())
                .is_null()
        );
        assert_eq!(compat_exec.is_compatible_out, 1);
        compat_exec.other_abi_version = ptr::null();
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_RuntimeCompatibleWithExecutable(ptr::addr_of_mut!(compat_exec).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_RuntimeCompatibleWithRuntime(ptr::addr_of_mut!(compat_exec).cast()),
            3,
        );
        compat_exec.abi_version = ptr::null();
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_RuntimeCompatibleWithExecutable(ptr::addr_of_mut!(compat_exec).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_RuntimeCompatibleWithRuntime(ptr::addr_of_mut!(compat_exec).cast()),
            3,
        );

        if let Some(h) = crate::pjrt_helpers::borrow_live_handle_mut(r_abi_null.abi_version) {
            h.proto_bytes.clear();
        }
        if let Some(h) = crate::pjrt_helpers::borrow_live_handle_mut(e_abi_null.abi_version) {
            h.proto_bytes.clear();
        }
        for abi_p in [r_abi_null.abi_version.cast_const(), ptr::null()] {
            let mut rtp = PJRT_AbiVersion_ToProto_Args {
                struct_size: size_of::<PJRT_AbiVersion_ToProto_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: abi_p,
                serialized_proto: ptr::null(),
                serialized_proto_size: 0,
                serialized_proto_holder: ptr::null_mut(),
            };
            assert!(PJRT_AbiVersion_RuntimeToProto(ptr::addr_of_mut!(rtp).cast()).is_null());
            pjrt_serialized_artifact_deleter(rtp.serialized_proto_holder);
            let mut rpid = PJRT_AbiVersion_PlatformId_Args {
                struct_size: size_of::<PJRT_AbiVersion_PlatformId_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: abi_p,
                platform_id_out: 0,
            };
            assert!(PJRT_AbiVersion_RuntimePlatformId(ptr::addr_of_mut!(rpid).cast()).is_null());
        }
        for abi_p in [e_abi_null.abi_version.cast_const(), ptr::null()] {
            let mut etp = PJRT_AbiVersion_ToProto_Args {
                struct_size: size_of::<PJRT_AbiVersion_ToProto_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: abi_p,
                serialized_proto: ptr::null(),
                serialized_proto_size: 0,
                serialized_proto_holder: ptr::null_mut(),
            };
            assert!(PJRT_AbiVersion_ExecutableToProto(ptr::addr_of_mut!(etp).cast()).is_null());
            let valid_exec_proto = std::slice::from_raw_parts(
                etp.serialized_proto.cast::<u8>(),
                etp.serialized_proto_size,
            )
            .to_vec();
            pjrt_serialized_artifact_deleter(etp.serialized_proto_holder);

            let mut epid = PJRT_AbiVersion_PlatformId_Args {
                struct_size: size_of::<PJRT_AbiVersion_PlatformId_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: abi_p,
                platform_id_out: 0,
            };
            assert!(PJRT_AbiVersion_ExecutablePlatformId(ptr::addr_of_mut!(epid).cast()).is_null());

            if !abi_p.is_null() {
                let mut efp = PJRT_AbiVersion_FromProto_Args {
                    struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
                    extension_start: ptr::null_mut(),
                    serialized_proto: valid_exec_proto.as_ptr().cast(),
                    serialized_proto_size: valid_exec_proto.len(),
                    abi_version_out: ptr::null_mut(),
                };
                assert!(
                    PJRT_AbiVersion_ExecutableFromProto(ptr::addr_of_mut!(efp).cast()).is_null()
                );
                let mut edel = PJRT_AbiVersion_Destroy_Args {
                    struct_size: size_of::<PJRT_AbiVersion_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    abi_version: efp.abi_version_out,
                };
                assert!(
                    PJRT_AbiVersion_ExecutableDestroy(ptr::addr_of_mut!(edel).cast()).is_null()
                );
            }
        }
        let non_utf8_abi = libtpu_core::proto::RuntimeAbiVersionProto {
            platform: TPU_PLATFORM_ID,
            version: vec![0xff, 0xfe],
        }
        .encode_proto();
        let mut rfp_non_utf8 = PJRT_AbiVersion_FromProto_Args {
            struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
            extension_start: ptr::null_mut(),
            serialized_proto: non_utf8_abi.as_ptr().cast(),
            serialized_proto_size: non_utf8_abi.len(),
            abi_version_out: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_RuntimeFromProto(ptr::addr_of_mut!(rfp_non_utf8).cast()).is_null());
        let mut rdel = PJRT_AbiVersion_Destroy_Args {
            struct_size: size_of::<PJRT_AbiVersion_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            abi_version: rfp_non_utf8.abi_version_out,
        };
        assert!(PJRT_AbiVersion_RuntimeDestroy(ptr::addr_of_mut!(rdel).cast()).is_null());
        rdel.abi_version = r_abi_null.abi_version;
        assert!(PJRT_AbiVersion_RuntimeDestroy(ptr::addr_of_mut!(rdel).cast()).is_null());
        rdel.abi_version = e_abi_null.abi_version;
        assert!(PJRT_AbiVersion_ExecutableDestroy(ptr::addr_of_mut!(rdel).cast()).is_null());
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_ExecutableDestroy(ptr::addr_of_mut!(rdel).cast()),
            3,
        );

        let corrupt_wire = [0xff_u8; 11];
        let mut empty_fp = PJRT_AbiVersion_FromProto_Args {
            struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
            extension_start: ptr::null_mut(),
            serialized_proto: ptr::null(),
            serialized_proto_size: 0,
            abi_version_out: ptr::null_mut(),
        };
        assert!(PJRT_AbiVersion_RuntimeFromProto(ptr::addr_of_mut!(empty_fp).cast()).is_null());
        rdel.abi_version = empty_fp.abi_version_out;
        assert!(PJRT_AbiVersion_RuntimeDestroy(ptr::addr_of_mut!(rdel).cast()).is_null());

        let mut bad_fp = PJRT_AbiVersion_FromProto_Args {
            struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
            extension_start: ptr::null_mut(),
            serialized_proto: ptr::null(),
            serialized_proto_size: 4,
            abi_version_out: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_RuntimeFromProto(ptr::addr_of_mut!(bad_fp).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_ExecutableFromProto(ptr::addr_of_mut!(bad_fp).cast()),
            3,
        );
        bad_fp.serialized_proto = corrupt_wire.as_ptr().cast();
        bad_fp.serialized_proto_size = corrupt_wire.len();
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_RuntimeFromProto(ptr::addr_of_mut!(bad_fp).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_AbiVersion_ExecutableFromProto(ptr::addr_of_mut!(bad_fp).cast()),
            3,
        );

        // Shardings (real & null executable) + TpuExecutable 9 slots
        for ex_opt in [exec_ptr.cast_const(), ptr::null()] {
            let mut sh_args = PJRT_Shardings_PJRT_Executable_Shardings_Args {
                struct_size: size_of::<PJRT_Shardings_PJRT_Executable_Shardings_Args>(),
                extension_start: ptr::null_mut(),
                executable: ex_opt.cast_mut(),
                num_items: 0,
                shardings: ptr::null(),
                sharding_sizes: ptr::null(),
            };
            assert!(PJRT_Shardings_ParameterShardings(ptr::addr_of_mut!(sh_args).cast()).is_null());
            assert!(PJRT_Shardings_OutputShardings(ptr::addr_of_mut!(sh_args).cast()).is_null());

            let mut pq = PJRT_TpuExecutable_ProtoQuery_Args {
                struct_size: size_of::<PJRT_TpuExecutable_ProtoQuery_Args>(),
                extension_start: ptr::null_mut(),
                executable: ex_opt,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                holder: ptr::null_mut(),
            };
            assert!(PJRT_TpuExecutable_GetTargetArguments(ptr::addr_of_mut!(pq).cast()).is_null());
            pjrt_serialized_artifact_deleter(pq.holder.cast());
            assert!(
                PJRT_TpuExecutable_GetHloModuleWithConfig(ptr::addr_of_mut!(pq).cast()).is_null()
            );
            pjrt_serialized_artifact_deleter(pq.holder.cast());
            assert!(
                PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString(ptr::addr_of_mut!(pq).cast())
                    .is_null()
            );
            pjrt_serialized_artifact_deleter(pq.holder.cast());

            let mut ms_q = PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
                struct_size: size_of::<PJRT_TpuExecutable_GetCompiledMemoryStats_Args>(),
                extension_start: ptr::null_mut(),
                executable: ex_opt.cast_mut(),
                generated_code_size_in_bytes: 0,
                argument_size_in_bytes: 0,
                output_size_in_bytes: 0,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 0,
            };
            assert!(
                PJRT_TpuExecutable_GetCompiledMemoryStats(ptr::addr_of_mut!(ms_q).cast()).is_null()
            );

            let mut ca_q = PJRT_TpuExecutable_RunHloCostAnalysis_Args {
                struct_size: size_of::<PJRT_TpuExecutable_RunHloCostAnalysis_Args>(),
                extension_start: ptr::null_mut(),
                executable: ex_opt,
                num_properties_out: 0,
            };
            assert!(
                PJRT_TpuExecutable_RunHloCostAnalysis(ptr::addr_of_mut!(ca_q).cast()).is_null()
            );

            let mut fp_q = PJRT_TpuExecutable_GetFingerprint_Args {
                struct_size: size_of::<PJRT_TpuExecutable_GetFingerprint_Args>(),
                extension_start: ptr::null_mut(),
                executable: ex_opt,
                fingerprint_bytes: ptr::null(),
                fingerprint_size: 0,
                holder: ptr::null_mut(),
            };
            assert!(PJRT_TpuExecutable_GetFingerprint(ptr::addr_of_mut!(fp_q).cast()).is_null());
            pjrt_serialized_artifact_deleter(fp_q.holder.cast());
        }

        let sample_err = PJRT_Error_From_Code_Profiler_Helper();
        for err_opt in [sample_err.cast_const(), ptr::null()] {
            let mut err_chk = PJRT_TpuExecutable_ErrorCheck_Args {
                struct_size: size_of::<PJRT_TpuExecutable_ErrorCheck_Args>(),
                extension_start: ptr::null_mut(),
                error: err_opt,
                is_matched: 1,
            };
            assert!(
                PJRT_TpuExecutable_IsPredeterminedError(ptr::addr_of_mut!(err_chk).cast())
                    .is_null()
            );
            assert!(
                PJRT_TpuExecutable_IsCancellationPlaceholderError(
                    ptr::addr_of_mut!(err_chk).cast()
                )
                .is_null()
            );
        }
        let mut s_err_del = PJRT_Error_Destroy_Args {
            struct_size: size_of::<PJRT_Error_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            error: sample_err,
        };
        PJRT_Error_Destroy(ptr::addr_of_mut!(s_err_del).cast());

        let valid_env_bytes = libtpu_core::proto::TpuCompilationEnvironmentProto {
            xla_tpu_enable_flash_attention: Some(true),
            ..libtpu_core::proto::TpuCompilationEnvironmentProto::default()
        }
        .encode_proto();
        for ex_m in [exec_ptr, ptr::null_mut()] {
            let mut set_env = PJRT_TpuExecutable_SetTpuCompilationEnv_Args {
                struct_size: size_of::<PJRT_TpuExecutable_SetTpuCompilationEnv_Args>(),
                extension_start: ptr::null_mut(),
                executable: ex_m,
                serialized_env: valid_env_bytes.as_ptr(),
                serialized_env_size: valid_env_bytes.len(),
            };
            assert!(
                PJRT_TpuExecutable_SetTpuCompilationEnv(ptr::addr_of_mut!(set_env).cast())
                    .is_null()
            );
            set_env.serialized_env = ptr::null();
            set_env.serialized_env_size = 0;
            assert!(
                PJRT_TpuExecutable_SetTpuCompilationEnv(ptr::addr_of_mut!(set_env).cast())
                    .is_null()
            );
        }
        let mut res_arg = PJRT_Megascale_Handle_Args {
            struct_size: size_of::<PJRT_Megascale_Handle_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(PJRT_TpuExecutable_Reserved(ptr::addr_of_mut!(res_arg).cast()).is_null());

        // 4. Memory / HostMemoryAllocator / ExecutableMetadata / Layouts / RawBuffer remaining branches
        let mut hma_get = PJRT_HostMemoryAllocator_Get_Args {
            struct_size: size_of::<PJRT_HostMemoryAllocator_Get_Args>(),
            extension_start: ptr::null_mut(),
            client,
            allocator_out: 0,
        };
        assert!(PJRT_HostMemoryAllocator_Get(ptr::addr_of_mut!(hma_get).cast()).is_null());
        let mut hma_alloc = PJRT_HostMemoryAllocator_Allocate_Args {
            struct_size: size_of::<PJRT_HostMemoryAllocator_Allocate_Args>(),
            extension_start: ptr::null_mut(),
            client,
            size: 128,
            numa_node: 0,
            pad: 0,
            ptr: ptr::null_mut(),
            deleter: None,
            deleter_arg: ptr::null_mut(),
        };
        assert!(PJRT_HostMemoryAllocator_Allocate(ptr::addr_of_mut!(hma_alloc).cast()).is_null());
        assert!(!hma_alloc.ptr.is_null());
        memory::pjrt_host_memory_deleter(hma_alloc.deleter_arg);
        memory::pjrt_host_memory_deleter(ptr::null_mut());

        for ex_m in [exec_ptr.cast_const(), ptr::null()] {
            let mut md_get = PJRT_Metadata_Get_Args {
                struct_size: size_of::<PJRT_Metadata_Get_Args>(),
                extension_start: ptr::null_mut(),
                executable: ex_m,
                metadata_out: ptr::null_mut(),
            };
            assert!(PJRT_ExecutableMetadata_Get(ptr::addr_of_mut!(md_get).cast()).is_null());
            let mut md_del = PJRT_Metadata_Destroy_Args {
                struct_size: size_of::<PJRT_Metadata_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                metadata: md_get.metadata_out,
            };
            assert!(PJRT_ExecutableMetadata_Destroy(ptr::addr_of_mut!(md_del).cast()).is_null());
        }

        let mut mdesc_null = PJRT_DeviceDescription_MemoryDescriptions_Args {
            struct_size: size_of::<PJRT_DeviceDescription_MemoryDescriptions_Args>(),
            extension_start: ptr::null_mut(),
            device_description: ptr::null(),
            memory_descriptions: ptr::null(),
            num_memory_descriptions: 0,
            default_memory_index: 0,
        };
        assert!(PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(mdesc_null).cast()).is_null());
        let mut mdesc_kind = PJRT_MemoryDescription_Kind_Args {
            struct_size: size_of::<PJRT_MemoryDescription_Kind_Args>(),
            extension_start: ptr::null_mut(),
            memory_description: ptr::null(),
            kind: ptr::null(),
            kind_size: 0,
            kind_id: 0,
            pad: 0,
        };
        assert!(PJRT_MemoryDescription_Kind(ptr::addr_of_mut!(mdesc_kind).cast()).is_null());
        let mut md_list_del = PJRT_Metadata_Destroy_Args {
            struct_size: size_of::<PJRT_Metadata_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            metadata: ptr::null_mut(),
        };
        assert!(PJRT_MemoryDescriptions_Destroy(ptr::addr_of_mut!(md_list_del).cast()).is_null());

        let test_buf = upload_f32_buffer(client, &[1.0, 2.0, 3.0, 4.0]);
        for b_opt in [test_buf, ptr::null_mut()] {
            let mut bml = PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
                struct_size: size_of::<PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>(),
                extension_start: ptr::null_mut(),
                buffer: b_opt,
                layout: ptr::null_mut(),
            };
            assert!(PJRT_Layouts_Buffer_GetMemoryLayout(ptr::addr_of_mut!(bml).cast()).is_null());
            let mut ldel = PJRT_Layouts_MemoryLayout_Destroy_Args {
                struct_size: size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                layout: bml.layout,
            };
            assert!(PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(ldel).cast()).is_null());
        }
        let stride_layout =
            memory::alloc_registered_memory_layout(libtpu_core::types::MemoryLayout::Strides {
                byte_strides: vec![8, 4],
            });
        let mut lser = PJRT_Layouts_MemoryLayout_Serialize_Args {
            struct_size: size_of::<PJRT_Layouts_MemoryLayout_Serialize_Args>(),
            extension_start: ptr::null_mut(),
            layout: stride_layout,
            serialized_bytes: ptr::null(),
            serialized_bytes_size: 0,
            serialized_layout: ptr::null_mut(),
            serialized_layout_deleter: None,
        };
        assert!(PJRT_Layouts_MemoryLayout_Serialize(ptr::addr_of_mut!(lser).cast()).is_null());
        pjrt_serialized_artifact_deleter(lser.serialized_layout);
        lser.layout = ptr::null();
        assert!(PJRT_Layouts_MemoryLayout_Serialize(ptr::addr_of_mut!(lser).cast()).is_null());
        pjrt_serialized_artifact_deleter(lser.serialized_layout);
        let mut ldel_stride = PJRT_Layouts_MemoryLayout_Destroy_Args {
            struct_size: size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            layout: stride_layout,
        };
        assert!(PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(ldel_stride).cast()).is_null());
        ldel_stride.layout = ptr::null_mut();
        assert!(PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(ldel_stride).cast()).is_null());

        let dims2 = [4_i64, 4];
        let mut def_lay = PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
            struct_size: size_of::<PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args>(),
            extension_start: ptr::null_mut(),
            client,
            elem_type: 11,
            pad: 0,
            dims: dims2.as_ptr(),
            num_dims: 2,
            layout: ptr::null_mut(),
        };
        assert!(PJRT_Layouts_Client_GetDefaultLayout(ptr::addr_of_mut!(def_lay).cast()).is_null());
        ldel_stride.layout = def_lay.layout;
        assert!(PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(ldel_stride).cast()).is_null());

        assert_and_free_pjrt_error(PJRT_Layouts_Executable_GetOutputLayouts(ptr::null_mut()), 3);
        let mut bad_elay = PJRT_Layouts_PJRT_Executable_Layouts_Args {
            struct_size: 8,
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
            num_layouts: 0,
            layouts: ptr::null(),
        };
        assert_and_free_pjrt_error(
            PJRT_Layouts_Executable_GetOutputLayouts(ptr::addr_of_mut!(bad_elay).cast()),
            3,
        );
        bad_elay.struct_size = 32;
        bad_elay.executable = ptr::null_mut();
        assert!(
            PJRT_Layouts_Executable_GetOutputLayouts(ptr::addr_of_mut!(bad_elay).cast()).is_null()
        );

        let mut rb_null = PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args {
            struct_size: size_of::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>(),
            extension_start: ptr::null_mut(),
            buffer: ptr::null_mut(),
            raw_buffer_out: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(rb_null).cast()),
            3,
        );
        rb_null.buffer = test_buf;
        assert!(PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(rb_null).cast()).is_null());
        let rb = rb_null.raw_buffer_out;
        let mut cp_null = PJRT_RawBuffer_CopyRaw_Args {
            struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
            extension_start: ptr::null_mut(),
            buffer: rb,
            host_ptr: ptr::null_mut(),
            offset: 0,
            transfer_size: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(cp_null).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(cp_null).cast()),
            3,
        );
        let mut rb_del = PJRT_RawBuffer_Destroy_Args {
            struct_size: size_of::<PJRT_RawBuffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: rb,
        };
        assert!(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(rb_del).cast()).is_null());

        // 5. Collectives & CrossHostTransfers remaining branches
        let mut col_del = PJRT_Collectives_Destroy_Args {
            struct_size: size_of::<PJRT_Collectives_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            collectives: ptr::null_mut(),
        };
        assert!(PJRT_Collectives_Destroy(ptr::addr_of_mut!(col_del).cast()).is_null());

        let mut cc_single = PJRT_Collectives_CreateCommunicator_Args {
            struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
            extension_start: ptr::null_mut(),
            collectives: ptr::null_mut(),
            device_ids: ptr::null(),
            num_devices: 0,
            communicator: ptr::null_mut(),
        };
        assert!(PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(cc_single).cast()).is_null());
        let scomm = cc_single.communicator;
        let out_b = upload_f32_buffer(client, &[0.0, 0.0, 0.0, 0.0]);
        for op_code in [0_i32, 1, 2, 3, 99] {
            let mut ar_s = PJRT_Communicator_AllReduce_Args {
                struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
                extension_start: ptr::null_mut(),
                communicator: scomm,
                send_buffer: test_buf,
                recv_buffer: out_b,
                reduction_op: op_code,
                pad: 0,
                count: 4,
                event: ptr::null_mut(),
            };
            let err = PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar_s).cast());
            if op_code == 99 {
                assert_and_free_pjrt_error(err, 3);
            } else {
                assert!(err.is_null());
                drop(Box::from_raw(ar_s.event));
            }
        }
        let mut cp_empty = PJRT_Communicator_CollectivePermute_Args {
            struct_size: size_of::<PJRT_Communicator_CollectivePermute_Args>(),
            extension_start: ptr::null_mut(),
            communicator: scomm,
            send_buffer: test_buf,
            recv_buffer: out_b,
            target_ranks: ptr::null(),
            count: 0,
            event: ptr::null_mut(),
        };
        assert!(PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_empty).cast()).is_null());
        drop(Box::from_raw(cp_empty.event));
        let bad_pair = [0_i32, 99];
        cp_empty.target_ranks = bad_pair.as_ptr();
        cp_empty.count = 2;
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_empty).cast()),
            3,
        );
        let mut scomm_del = PJRT_Communicator_Destroy_Args {
            struct_size: size_of::<PJRT_Communicator_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            communicator: scomm,
        };
        assert!(PJRT_Communicator_Destroy(ptr::addr_of_mut!(scomm_del).cast()).is_null());

        let xh_null = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: ptr::null(),
            num_buffers: 0,
            transfer_key: 0,
            out_event: ptr::null_mut(),
        };
        let mut xh_n1 = xh_null;
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_n1).cast()),
            3,
        );
        let mut xh_n2 = xh_null;
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_ReceiveBuffers(ptr::addr_of_mut!(xh_n2).cast()),
            3,
        );

        let recv_bufs = [out_b];
        let mut xh_recv_first = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: recv_bufs.as_ptr(),
            num_buffers: 1,
            transfer_key: 9090,
            out_event: ptr::null_mut(),
        };
        assert!(
            PJRT_CrossHostTransfers_ReceiveBuffers(ptr::addr_of_mut!(xh_recv_first).cast())
                .is_null()
        );
        let recv_ev = xh_recv_first.out_event;
        assert!(!recv_ev.is_null());

        let send_bufs = [test_buf];
        let mut xh_send_second = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: send_bufs.as_ptr(),
            num_buffers: 1,
            transfer_key: 9090,
            out_event: ptr::null_mut(),
        };
        assert!(
            PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_send_second).cast()).is_null()
        );
        drop(Box::from_raw(xh_send_second.out_event));
        let mut ev_del = PJRT_Event_Destroy_Args {
            struct_size: size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event: recv_ev,
        };
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del).cast()).is_null());

        // Use `dummy_xh_desc_cb` via `CrossHostTransferRegistry` callback registration check
        let mut desc_bytes_seen = 0_usize;
        dummy_xh_desc_cb(
            c"tag".as_ptr(),
            3,
            ptr::addr_of_mut!(desc_bytes_seen).cast(),
        );
        assert_eq!(desc_bytes_seen, 3);

        // 6. PLUGIN_Profiler remaining functions (`Error_*`, `Consume`, `ConsumeResult_Destroy`, `Serialize`, negotiated wrappers)
        let prof_err = PJRT_Error_From_Code_Profiler_Helper();
        let mut perr_msg = PLUGIN_Profiler_Error_Message_Args {
            struct_size: size_of::<PLUGIN_Profiler_Error_Message_Args>(),
            extension_start: ptr::null_mut(),
            error: prof_err,
            message: ptr::null(),
            message_size: 0,
        };
        assert!(PLUGIN_Profiler_Error_Message(ptr::addr_of_mut!(perr_msg).cast()).is_null());
        assert!(perr_msg.message_size > 0);
        perr_msg.error = ptr::null();
        assert!(PLUGIN_Profiler_Error_Message(ptr::addr_of_mut!(perr_msg).cast()).is_null());

        let mut perr_code = PLUGIN_Profiler_Error_GetCode_Args {
            struct_size: size_of::<PLUGIN_Profiler_Error_GetCode_Args>(),
            extension_start: ptr::null_mut(),
            error: prof_err,
            code: 0,
            pad: 0,
        };
        assert!(PLUGIN_Profiler_Error_GetCode(ptr::addr_of_mut!(perr_code).cast()).is_null());
        perr_code.error = ptr::null();
        assert!(PLUGIN_Profiler_Error_GetCode(ptr::addr_of_mut!(perr_code).cast()).is_null());

        let mut perr_del = PLUGIN_Profiler_Error_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_Error_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            error: prof_err,
        };
        assert!(PLUGIN_Profiler_Error_Destroy(ptr::addr_of_mut!(perr_del).cast()).is_null());
        perr_del.error = ptr::null_mut();
        assert!(PLUGIN_Profiler_Error_Destroy(ptr::addr_of_mut!(perr_del).cast()).is_null());

        let mut p_cons = PLUGIN_Profiler_Consume_Args {
            struct_size: size_of::<PLUGIN_Profiler_Consume_Args>(),
            profiler: ptr::null_mut(),
            result: ptr::null_mut(),
        };
        assert!(PLUGIN_Profiler_Consume(ptr::addr_of_mut!(p_cons).cast()).is_null());
        let mut p_ser = PLUGIN_Profiler_Serialize_Args {
            struct_size: size_of::<PLUGIN_Profiler_Serialize_Args>(),
            profiler: ptr::null_mut(),
            result: p_cons.result.cast_const(),
            buffer: ptr::null(),
            size_in_bytes: 0,
        };
        assert!(PLUGIN_Profiler_Serialize(ptr::addr_of_mut!(p_ser).cast()).is_null());
        let mut p_cdel = PLUGIN_Profiler_ConsumeResult_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_ConsumeResult_Destroy_Args>(),
            result: p_cons.result,
        };
        assert!(PLUGIN_Profiler_ConsumeResult_Destroy(ptr::addr_of_mut!(p_cdel).cast()).is_null());

        // Exercise negotiated_profiler_* wrappers through `NEGOTIATED_EXT_PROFILER`
        let p_api = &*profiler::NEGOTIATED_EXT_PROFILER.profiler_api;
        let mut pc_words = [32_usize, 0, 0, 0];
        assert!((p_api.create)(pc_words.as_mut_ptr().cast()).is_null());
        let p_inst = pc_words[3];
        let mut ps_words = [16_usize, p_inst];
        assert!((p_api.start)(ps_words.as_mut_ptr().cast()).is_null());
        assert!((p_api.stop)(ps_words.as_mut_ptr().cast()).is_null());
        let mut pcd_words = [32_usize, p_inst, 0, 0];
        assert!((p_api.collect_data)(pcd_words.as_mut_ptr().cast()).is_null());
        assert!((p_api.destroy)(ps_words.as_mut_ptr().cast()).is_null());
        let mut pe_msg = PLUGIN_Profiler_Error_Message_Args {
            struct_size: size_of::<PLUGIN_Profiler_Error_Message_Args>(),
            extension_start: ptr::null_mut(),
            error: ptr::null(),
            message: ptr::null(),
            message_size: 0,
        };
        assert!((p_api.error_message)(ptr::addr_of_mut!(pe_msg).cast()).is_null());
        let mut pe_code = PLUGIN_Profiler_Error_GetCode_Args {
            struct_size: size_of::<PLUGIN_Profiler_Error_GetCode_Args>(),
            extension_start: ptr::null_mut(),
            error: ptr::null(),
            code: 0,
            pad: 0,
        };
        assert!((p_api.error_get_code)(ptr::addr_of_mut!(pe_code).cast()).is_null());
        let mut pe_del = PLUGIN_Profiler_Error_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_Error_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            error: ptr::null_mut(),
        };
        assert!((p_api.error_destroy)(ptr::addr_of_mut!(pe_del).cast()).is_null());

        let mut ex_del = crate::pjrt_types::PJRT_Executable_Destroy_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
        };
        assert!(
            crate::pjrt_executable_api::PJRT_Executable_Destroy(ptr::addr_of_mut!(ex_del).cast())
                .is_null()
        );

        for b in [test_buf, out_b] {
            let mut bd = PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: b,
            };
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(bd).cast()).is_null());
        }
        let mut lex_del = PJRT_LoadedExecutable_Destroy_Args {
            struct_size: size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: loaded_exec,
        };
        assert!(PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(lex_del).cast()).is_null());
        let mut cli_del = PJRT_Client_Destroy_Args {
            struct_size: size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client,
        };
        assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(cli_del).cast()).is_null());
    }
}

/// Helper returning a heap-allocated `PJRT_Error*` via `PLUGIN_Profiler_Create(null)` for profiler error tests.
///
/// # Safety
///
/// Caller must destroy the returned `PJRT_Error*` via `PLUGIN_Profiler_Error_Destroy`.
unsafe fn PJRT_Error_From_Code_Profiler_Helper() -> *mut PJRT_Error {
    // SAFETY: Passing null to `PLUGIN_Profiler_Create` returns a fresh `PJRT_Error*`.
    unsafe { PLUGIN_Profiler_Create(ptr::null_mut()) }
}

/// Dummy unwind invoker exercising `dispatch_erased_extension_args` panic containment without `panic!`.
///
/// # Arguments
///
/// * `_raw_args` - Unused args pointer.
/// * `_handler_ref` - Unused handler pointer.
///
/// # Safety
///
/// Always unwinds via `std::panic::resume_unwind`.
unsafe fn unwinding_ext_invoker(
    _raw_args: *mut std::ffi::c_void,
    _handler_ref: *const std::ffi::c_void,
) {
    std::panic::resume_unwind(Box::new("simulated extension unwind"));
}

/// Dummy fallible unwind invoker exercising `dispatch_erased_fallible_extension_args` panic containment without `panic!`.
///
/// # Arguments
///
/// * `_raw_args` - Unused args pointer.
/// * `_handler_ref` - Unused handler pointer.
///
/// # Safety
///
/// Always unwinds via `std::panic::resume_unwind`.
unsafe fn unwinding_fallible_ext_invoker(
    _raw_args: *mut std::ffi::c_void,
    _handler_ref: *const std::ffi::c_void,
) -> *mut PJRT_Error {
    std::panic::resume_unwind(Box::new("simulated fallible extension unwind"))
}

/// Exercises the remaining error, legacy-size, generation-specific, and unwind branches across all extension modules.
#[test]
fn test_remaining_extension_error_and_legacy_branches() {
    // SAFETY: All FFI argument structs and stack pointers are valid for the duration of each call.
    unsafe {
        assert!(!PJRT_FFI_GetExtension().is_null());
        let mut dummy_sz = [16_usize, 0_usize];
        assert_and_free_pjrt_error(
            dispatch_erased_extension_args(
                dummy_sz.as_mut_ptr().cast(),
                8,
                ptr::null(),
                unwinding_ext_invoker,
            ),
            13,
        );
        assert_and_free_pjrt_error(
            dispatch_erased_fallible_extension_args(
                dummy_sz.as_mut_ptr().cast(),
                8,
                ptr::null(),
                unwinding_fallible_ext_invoker,
            ),
            13,
        );

        let mut client_args = PJRT_Client_Create_Args {
            struct_size: size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: None,
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
        };
        assert!(PJRT_Client_Create(ptr::addr_of_mut!(client_args).cast()).is_null());
        let client = client_args.client;

        // `topology.rs` line 929: out-of-bounds chip coord `[99, 99, 99]` on real topology
        let mut tc_v4 = crate::pjrt_types::PJRT_TopologyDescription_Create_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(),
            extension_start: ptr::null_mut(),
            topology_name: c"v4".as_ptr(),
            topology_name_size: 2,
            create_options: ptr::null(),
            num_options: 0,
            topology: ptr::null_mut(),
        };
        assert!(crate::PJRT_TopologyDescription_Create(ptr::addr_of_mut!(tc_v4).cast()).is_null());
        let mut tc_v5e = crate::pjrt_types::PJRT_TopologyDescription_Create_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(),
            extension_start: ptr::null_mut(),
            topology_name: c"v5e".as_ptr(),
            topology_name_size: 3,
            create_options: ptr::null(),
            num_options: 0,
            topology: ptr::null_mut(),
        };
        assert!(crate::PJRT_TopologyDescription_Create(ptr::addr_of_mut!(tc_v5e).cast()).is_null());

        let oob_coords = [99_i32, 99, 99];
        let mut ldfc_oob = PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args {
            struct_size: size_of::<PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args>(),
            extension_start: ptr::null_mut(),
            topology: tc_v4.topology.cast_const(),
            coords: oob_coords.as_ptr(),
            core_idx: 0,
            device_id_out: 0,
        };
        assert!(
            PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(ptr::addr_of_mut!(ldfc_oob).cast())
                .is_null()
        );

        // `memory.rs`: V4 / V5e / 32-byte `PJRT_MemoryDescriptions_Get`, 48-byte `Destroy`, overflow `Allocate`, non-null `Kind`, unregistered `RawBuffer`
        for topo_p in [tc_v4.topology, tc_v5e.topology] {
            let desc_ptr = (*topo_p)
                .desc_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut());
            let mut mdesc = PJRT_DeviceDescription_MemoryDescriptions_Args {
                struct_size: size_of::<PJRT_DeviceDescription_MemoryDescriptions_Args>(),
                extension_start: ptr::null_mut(),
                device_description: desc_ptr.cast_const(),
                memory_descriptions: ptr::null(),
                num_memory_descriptions: 0,
                default_memory_index: 0,
            };
            assert!(PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(mdesc).cast()).is_null());
            assert!(mdesc.num_memory_descriptions >= 3);
            let first_md = *mdesc.memory_descriptions;
            let mut mkind = PJRT_MemoryDescription_Kind_Args {
                struct_size: size_of::<PJRT_MemoryDescription_Kind_Args>(),
                extension_start: ptr::null_mut(),
                memory_description: first_md,
                kind: ptr::null(),
                kind_size: 0,
                kind_id: -1,
                pad: 0,
            };
            assert!(PJRT_MemoryDescription_Kind(ptr::addr_of_mut!(mkind).cast()).is_null());
            assert!(PJRT_MemoryDescriptions_Destroy(ptr::addr_of_mut!(mkind).cast()).is_null());
        }
        assert_and_free_pjrt_error(PJRT_MemoryDescriptions_Get(ptr::null_mut()), 3);
        assert_and_free_pjrt_error(PJRT_MemoryDescriptions_Destroy(ptr::null_mut()), 3);
        let mut md32 = PJRT_DeviceDescription_MemoryDescriptions_Args {
            struct_size: 8,
            extension_start: ptr::null_mut(),
            device_description: ptr::null(),
            memory_descriptions: ptr::null(),
            num_memory_descriptions: 0,
            default_memory_index: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(md32).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_MemoryDescriptions_Destroy(ptr::addr_of_mut!(md32).cast()),
            3,
        );
        md32.struct_size = 32;
        assert!(PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(md32).cast()).is_null());

        let mut hma_overflow = PJRT_HostMemoryAllocator_Allocate_Args {
            struct_size: size_of::<PJRT_HostMemoryAllocator_Allocate_Args>(),
            extension_start: ptr::null_mut(),
            client,
            size: usize::MAX,
            numa_node: 0,
            pad: 0,
            ptr: ptr::null_mut(),
            deleter: None,
            deleter_arg: ptr::null_mut(),
        };
        assert!(
            PJRT_HostMemoryAllocator_Allocate(ptr::addr_of_mut!(hma_overflow).cast()).is_null()
        );
        assert!(hma_overflow.ptr.is_null());

        let empty_tiled =
            memory::alloc_registered_memory_layout(libtpu_core::types::MemoryLayout::Tiled {
                minor_to_major: Vec::new(),
                tile_dims: Vec::new(),
                tile_dim_sizes: Vec::new(),
            });
        let mut lser_empty = PJRT_Layouts_MemoryLayout_Serialize_Args {
            struct_size: size_of::<PJRT_Layouts_MemoryLayout_Serialize_Args>(),
            extension_start: ptr::null_mut(),
            layout: empty_tiled,
            serialized_bytes: ptr::null(),
            serialized_bytes_size: 0,
            serialized_layout: ptr::null_mut(),
            serialized_layout_deleter: None,
        };
        assert!(
            PJRT_Layouts_MemoryLayout_Serialize(ptr::addr_of_mut!(lser_empty).cast()).is_null()
        );
        pjrt_serialized_artifact_deleter(lser_empty.serialized_layout);
        let mut ldel_e = PJRT_Layouts_MemoryLayout_Destroy_Args {
            struct_size: size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            layout: empty_tiled,
        };
        assert!(PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(ldel_e).cast()).is_null());

        let test_buf = upload_f32_buffer(client, &[1.0, 2.0, 3.0, 4.0]);
        let mut rb_c = PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args {
            struct_size: size_of::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>(),
            extension_start: ptr::null_mut(),
            buffer: test_buf,
            raw_buffer_out: ptr::null_mut(),
        };
        assert!(PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(rb_c).cast()).is_null());
        let stale_rb = rb_c.raw_buffer_out;
        let mut rb_del = PJRT_RawBuffer_Destroy_Args {
            struct_size: size_of::<PJRT_RawBuffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: stale_rb,
        };
        assert!(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(rb_del).cast()).is_null());
        rb_del.raw_buffer = stale_rb;
        assert_and_free_pjrt_error(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(rb_del).cast()), 3);
        let mut rb_hp = PJRT_RawBuffer_GetHostPointer_Args {
            struct_size: size_of::<PJRT_RawBuffer_GetHostPointer_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: stale_rb.cast_const(),
            host_pointer_out: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_GetHostPointer(ptr::addr_of_mut!(rb_hp).cast()),
            3,
        );
        let mut rb_sz = PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args {
            struct_size: size_of::<PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: stale_rb.cast_const(),
            on_device_size_in_bytes: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_GetOnDeviceSizeInBytes(ptr::addr_of_mut!(rb_sz).cast()),
            3,
        );
        let mut rb_ms = PJRT_RawBuffer_GetMemorySpace_Args {
            struct_size: size_of::<PJRT_RawBuffer_GetMemorySpace_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: stale_rb.cast_const(),
            memory_space_out: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_GetMemorySpace(ptr::addr_of_mut!(rb_ms).cast()),
            3,
        );
        let mut host_tmp = [0_u8; 4];
        let mut rb_cp = PJRT_RawBuffer_CopyRaw_Args {
            struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
            extension_start: ptr::null_mut(),
            buffer: stale_rb,
            host_ptr: host_tmp.as_mut_ptr().cast(),
            offset: 0,
            transfer_size: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(rb_cp).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(rb_cp).cast()),
            3,
        );
        assert!(PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(rb_c).cast()).is_null());
        rb_cp.buffer = rb_c.raw_buffer_out;
        rb_cp.offset = -4;
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(rb_cp).cast()),
            3,
        );
        rb_cp.offset = 0;
        rb_cp.transfer_size = -4;
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(rb_cp).cast()),
            3,
        );
        rb_del.raw_buffer = rb_c.raw_buffer_out;
        assert!(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(rb_del).cast()).is_null());

        // `megascale.rs`
        let mut ctx_c = PJRT_Megascale_CreateClientContextFromPjRtClient_Args {
            struct_size: size_of::<PJRT_Megascale_CreateClientContextFromPjRtClient_Args>(),
            extension_start: ptr::null_mut(),
            client,
            context_out: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_CreateClientContext(ptr::addr_of_mut!(ctx_c).cast()).is_null());
        let ctx = ctx_c.context_out;
        for out_init in [0_usize, 2] {
            let mut st = PJRT_Megascale_SetTransportType_Args {
                struct_size: size_of::<PJRT_Megascale_SetTransportType_Args>(),
                extension_start: ptr::null_mut(),
                requested_transport_type: ctx as usize,
                transport_type_out: out_init,
            };
            assert!(PJRT_Megascale_SetTransportType(ptr::addr_of_mut!(st).cast()).is_null());
        }
        let mut st_zero = PJRT_Megascale_SetTransportType_Args {
            struct_size: size_of::<PJRT_Megascale_SetTransportType_Args>(),
            extension_start: ptr::null_mut(),
            requested_transport_type: 0,
            transport_type_out: 0,
        };
        assert!(PJRT_Megascale_SetTransportType(ptr::addr_of_mut!(st_zero).cast()).is_null());
        let mut log_with_cb = PJRT_Megascale_RegisterErrorHandler_Args {
            struct_size: size_of::<PJRT_Megascale_RegisterErrorHandler_Args>(),
            extension_start: ptr::null_mut(),
            callback_fn: Some(dummy_megascale_err_cb),
            handler_id_out: 0,
        };
        assert!(
            PJRT_Megascale_RegisterErrorHandler(ptr::addr_of_mut!(log_with_cb).cast()).is_null()
        );
        let mut le = PJRT_Megascale_LogError_Args {
            struct_size: size_of::<PJRT_Megascale_LogError_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_LogError(ptr::addr_of_mut!(le).cast()).is_null());
        let mut ue = PJRT_Megascale_UnregisterErrorHandler_Args {
            struct_size: size_of::<PJRT_Megascale_UnregisterErrorHandler_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_UnregisterErrorHandler(ptr::addr_of_mut!(ue).cast()).is_null());
        let mut dcn_null = PJRT_Megascale_GetDcnTopology_Args {
            struct_size: size_of::<PJRT_Megascale_GetDcnTopology_Args>(),
            extension_start: ptr::null_mut(),
            context: ptr::null_mut(),
            serialized_size: 0,
        };
        assert!(PJRT_Megascale_GetDcnTopology(ptr::addr_of_mut!(dcn_null).cast()).is_null());
        let mut del_ctx = PJRT_Megascale_DeleteClientContext_Args {
            struct_size: size_of::<PJRT_Megascale_DeleteClientContext_Args>(),
            extension_start: ptr::null_mut(),
            context: ctx,
        };
        assert!(PJRT_Megascale_DeleteClientContext(ptr::addr_of_mut!(del_ctx).cast()).is_null());
        let bogus_cfg = ctx.cast::<crate::pjrt_types::PJRT_MultiSlice_Config>();
        let mut ms_nd = PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_NumDevicesPerSlice_Args>(),
            extension_start: ptr::null_mut(),
            config: ptr::null(),
            num_slices: 0,
            num_devices_per_slice_map: 0,
        };
        assert!(
            PJRT_MultiSlice_Config_NumDevicesPerSlice(ptr::addr_of_mut!(ms_nd).cast()).is_null()
        );
        ms_nd.config = bogus_cfg.cast_const();
        assert_and_free_pjrt_error(
            PJRT_MultiSlice_Config_NumDevicesPerSlice(ptr::addr_of_mut!(ms_nd).cast()),
            3,
        );

        // `collectives.rs`
        let send_arr = [test_buf];
        let mut xh_rem = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: send_arr.as_ptr(),
            num_buffers: 1,
            transfer_key: 4242,
            out_event: ptr::null_mut(),
        };
        assert!(
            PJRT_CrossHostTransfers_CopyToRemoteDevice(ptr::addr_of_mut!(xh_rem).cast()).is_null()
        );
        drop(Box::from_raw(xh_rem.out_event));
        let mut xh_recv = PJRT_CrossHostTransfers_Args {
            transfer_key: 4242,
            out_event: ptr::null_mut(),
            ..xh_rem
        };
        assert!(
            PJRT_CrossHostTransfers_ReceiveBuffers(ptr::addr_of_mut!(xh_recv).cast()).is_null()
        );
        drop(Box::from_raw(xh_recv.out_event));
        xh_rem.client = ptr::null_mut();
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_rem).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_ReceiveBuffers(ptr::addr_of_mut!(xh_rem).cast()),
            3,
        );
        let null_buf_arr: [*mut crate::pjrt_types::PJRT_Buffer; 1] = [ptr::null_mut()];
        xh_rem.client = client;
        xh_rem.buffers = null_buf_arr.as_ptr();
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_rem).cast()),
            3,
        );

        let neg_devs = [-1_i32];
        let mut cc_bad = PJRT_Collectives_CreateCommunicator_Args {
            struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
            extension_start: ptr::null_mut(),
            collectives: ptr::null_mut(),
            device_ids: ptr::null(),
            num_devices: 1,
            communicator: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(cc_bad).cast()),
            3,
        );
        cc_bad.device_ids = neg_devs.as_ptr();
        assert_and_free_pjrt_error(
            PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(cc_bad).cast()),
            3,
        );

        let mut ts_bad = PJRT_Communicator_ToString_Args {
            struct_size: size_of::<PJRT_Communicator_ToString_Args>(),
            extension_start: ptr::null_mut(),
            communicator: ptr::null(),
            str_out: ptr::null(),
            str_size: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_ToString(ptr::addr_of_mut!(ts_bad).cast()),
            3,
        );

        let mut cc_ok = PJRT_Collectives_CreateCommunicator_Args {
            struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
            extension_start: ptr::null_mut(),
            collectives: ptr::null_mut(),
            device_ids: ptr::null(),
            num_devices: 0,
            communicator: ptr::null_mut(),
        };
        assert!(PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(cc_ok).cast()).is_null());
        let comm = cc_ok.communicator;
        for bad_ranks in [&[-1_i32, 0][..], &[0_i32, -1][..], &[0_i32][..]] {
            let mut cp_neg = PJRT_Communicator_CollectivePermute_Args {
                struct_size: size_of::<PJRT_Communicator_CollectivePermute_Args>(),
                extension_start: ptr::null_mut(),
                communicator: comm,
                send_buffer: test_buf,
                recv_buffer: test_buf,
                target_ranks: bad_ranks.as_ptr(),
                count: bad_ranks.len(),
                event: ptr::null_mut(),
            };
            assert_and_free_pjrt_error(
                PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_neg).cast()),
                3,
            );
        }
        let mut rs_bad = PJRT_Communicator_AllReduce_Args {
            struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
            extension_start: ptr::null_mut(),
            communicator: ptr::null_mut(),
            send_buffer: test_buf,
            recv_buffer: test_buf,
            reduction_op: 0,
            pad: 0,
            count: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllReduce(ptr::addr_of_mut!(rs_bad).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_ReduceScatter(ptr::addr_of_mut!(rs_bad).cast()),
            3,
        );
        rs_bad.communicator = comm;
        rs_bad.send_buffer = ptr::null_mut();
        assert_and_free_pjrt_error(
            PJRT_Communicator_ReduceScatter(ptr::addr_of_mut!(rs_bad).cast()),
            3,
        );
        let mut ag_bad = PJRT_Communicator_CollectivePermute_Args {
            struct_size: size_of::<PJRT_Communicator_CollectivePermute_Args>(),
            extension_start: ptr::null_mut(),
            communicator: ptr::null_mut(),
            send_buffer: test_buf,
            recv_buffer: test_buf,
            target_ranks: ptr::null(),
            count: 0,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag_bad).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllToAll(ptr::addr_of_mut!(ag_bad).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(ag_bad).cast()),
            3,
        );
        ag_bad.communicator = comm;
        ag_bad.send_buffer = ptr::null_mut();
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag_bad).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllToAll(ptr::addr_of_mut!(ag_bad).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(ag_bad).cast()),
            3,
        );

        let mut comm_del = PJRT_Communicator_Destroy_Args {
            struct_size: size_of::<PJRT_Communicator_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm,
        };
        assert!(PJRT_Communicator_Destroy(ptr::addr_of_mut!(comm_del).cast()).is_null());

        // `profiler.rs`
        let p_api = &*profiler::NEGOTIATED_EXT_PROFILER.profiler_api;
        assert!((p_api.create)(ptr::null_mut()).is_null());
        assert!((p_api.start)(ptr::null_mut()).is_null());
        assert!((p_api.stop)(ptr::null_mut()).is_null());
        assert!((p_api.collect_data)(ptr::null_mut()).is_null());
        assert!((p_api.destroy)(ptr::null_mut()).is_null());
        assert_and_free_pjrt_error((p_api.error_message)(ptr::null_mut()), 3);
        assert_and_free_pjrt_error((p_api.error_get_code)(ptr::null_mut()), 3);
        assert_and_free_pjrt_error((p_api.error_destroy)(ptr::null_mut()), 3);

        let mut leg_create = [32_usize, 0, 0, 0];
        assert!((p_api.create)(leg_create.as_mut_ptr().cast()).is_null());
        let leg_inst = leg_create[3] as *mut crate::pjrt_types::PLUGIN_Profiler_Instance;
        assert!(!leg_inst.is_null());

        let mut leg_start = [16_usize, leg_inst as usize];
        assert!((p_api.start)(leg_start.as_mut_ptr().cast()).is_null());
        assert!((p_api.stop)(leg_start.as_mut_ptr().cast()).is_null());

        let mut p_cons_live = PLUGIN_Profiler_Consume_Args {
            struct_size: size_of::<PLUGIN_Profiler_Consume_Args>(),
            profiler: leg_inst,
            result: ptr::null_mut(),
        };
        assert!((p_api.consume)(ptr::addr_of_mut!(p_cons_live).cast()).is_null());
        let res_ptr = p_cons_live.result;
        let mut p_ser_bad = PLUGIN_Profiler_Serialize_Args {
            struct_size: size_of::<PLUGIN_Profiler_Serialize_Args>(),
            profiler: ptr::null_mut(),
            result: ptr::null(),
            buffer: ptr::null(),
            size_in_bytes: 0,
        };
        assert_and_free_pjrt_error((p_api.serialize)(ptr::addr_of_mut!(p_ser_bad).cast()), 3);
        let mut p_cdel = PLUGIN_Profiler_ConsumeResult_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_ConsumeResult_Destroy_Args>(),
            result: res_ptr,
        };
        assert!((p_api.consume_result_destroy)(ptr::addr_of_mut!(p_cdel).cast()).is_null());
        p_cdel.result = res_ptr;
        assert_and_free_pjrt_error(
            (p_api.consume_result_destroy)(ptr::addr_of_mut!(p_cdel).cast()),
            3,
        );

        let mut leg_sz = 0_usize;
        let mut leg_cd = [
            24_usize,
            leg_inst as usize,
            0,
            ptr::addr_of_mut!(leg_sz) as usize,
        ];
        assert!((p_api.collect_data)(leg_cd.as_mut_ptr().cast()).is_null());
        let mut xbuf = vec![0_u8; leg_sz.max(1)];
        leg_cd[2] = xbuf.as_mut_ptr() as usize;
        assert!((p_api.collect_data)(leg_cd.as_mut_ptr().cast()).is_null());

        assert!((p_api.destroy)(leg_start.as_mut_ptr().cast()).is_null());
        p_cons_live.profiler = leg_inst;
        assert_and_free_pjrt_error((p_api.consume)(ptr::addr_of_mut!(p_cons_live).cast()), 3);

        let prof_err = PJRT_Error_From_Code_Profiler_Helper();
        let mut pe_del = PLUGIN_Profiler_Error_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_Error_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            error: prof_err,
        };
        assert!((p_api.error_destroy)(ptr::addr_of_mut!(pe_del).cast()).is_null());

        for t_del in [tc_v4.topology, tc_v5e.topology] {
            let mut td = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                topology: t_del,
            };
            assert!(
                crate::PJRT_TopologyDescription_Destroy(ptr::addr_of_mut!(td).cast()).is_null()
            );
        }
        let mut bd = PJRT_Buffer_Destroy_Args {
            struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            buffer: test_buf,
        };
        assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(bd).cast()).is_null());
        let mut cli_del = PJRT_Client_Destroy_Args {
            struct_size: size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client,
        };
        assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(cli_del).cast()).is_null());
    }
}

#[test]
fn test_extensions_exhaustive_edge_and_error_contracts() {
    /// Increments the stack `usize` deleter counter passed via `ptr`.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Optional pointer to a mutable `usize` counter.
    ///
    /// # Safety
    ///
    /// When non-null, `ptr` must point to a valid writable `usize`.
    unsafe extern "C" fn userdata_deleter(ptr: *mut std::ffi::c_void) {
        if !ptr.is_null() {
            // SAFETY: `ptr` points to a mutable `usize` counter on the stack.
            unsafe {
                *ptr.cast::<usize>() += 1;
            }
        }
    }
    /// Frees any passed `PJRT_Error*` and increments the stack `usize` callback counter.
    ///
    /// # Arguments
    ///
    /// * `err` - Optional `PJRT_Error*` status returned by the transfer.
    /// * `user_arg` - Optional pointer to a mutable `usize` counter.
    ///
    /// # Safety
    ///
    /// `err` must be null or a valid `PJRT_Error*`, and `user_arg` must be null or a valid `usize*`.
    unsafe extern "C" fn cross_host_ready_cb(
        err: *mut PJRT_Error,
        user_arg: *mut std::ffi::c_void,
    ) {
        if !err.is_null() {
            let mut ed = PJRT_Error_Destroy_Args {
                struct_size: size_of::<PJRT_Error_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                error: err,
            };
            // SAFETY: Valid `PJRT_Error_Destroy_Args`.
            unsafe {
                PJRT_Error_Destroy(ptr::addr_of_mut!(ed).cast());
            }
        }
        if !user_arg.is_null() {
            // SAFETY: `user_arg` points to a stack `usize` counter.
            unsafe {
                *user_arg.cast::<usize>() += 1;
            }
        }
    }

    // SAFETY: Exercising typed FFI contracts across all PJRT extensions.
    unsafe {
        let mut cc = PJRT_Client_Create_Args {
            struct_size: size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: None,
            kv_put_user_arg: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
        };
        assert!(PJRT_Client_Create(ptr::addr_of_mut!(cc).cast()).is_null());
        let client = cc.client;

        // 1. Collectives: 3-rank group allocation, element_type/op errors, & MakeReceiveBuffers + async CrossHostTransfers
        let dev_ids_3 = [0_i32, 1_i32, 2_i32];
        let mut comms_3 = [ptr::null_mut(); 3];
        for slot in &mut comms_3 {
            let mut c_args = PJRT_Collectives_CreateCommunicator_Args {
                struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
                extension_start: ptr::null_mut(),
                collectives: client.cast(),
                device_ids: dev_ids_3.as_ptr(),
                num_devices: 3,
                communicator: ptr::null_mut(),
            };
            assert!(
                PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(c_args).cast()).is_null()
            );
            *slot = c_args.communicator;
        }
        let mut single_comm_args = PJRT_Collectives_CreateCommunicator_Args {
            struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
            extension_start: ptr::null_mut(),
            collectives: client.cast(),
            device_ids: ptr::null(),
            num_devices: 0,
            communicator: ptr::null_mut(),
        };
        assert!(
            PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(single_comm_args).cast())
                .is_null()
        );
        let comm1 = single_comm_args.communicator;

        let b_in = upload_f32_buffer(client, &[1.0, 2.0, 3.0, 4.0]);
        let b_out = upload_f32_buffer(client, &[0.0; 4]);
        let b_big = upload_f32_buffer(client, &[1.0; 8]);

        let mut ar_bad = PJRT_Communicator_AllReduce_Args {
            struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm1,
            send_buffer: b_in,
            recv_buffer: b_out,
            reduction_op: 99,
            pad: 0,
            count: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar_bad).cast()),
            3,
        );
        ar_bad.reduction_op = 0;
        ar_bad.recv_buffer = b_in;
        assert!(PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar_bad).cast()).is_null());
        let mut ev_del_inp = PJRT_Event_Destroy_Args {
            struct_size: size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event: ar_bad.event,
        };
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del_inp).cast()).is_null());
        ar_bad.communicator = comms_3[0];
        ar_bad.recv_buffer = b_big;
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar_bad).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_ReduceScatter(ptr::addr_of_mut!(ar_bad).cast()),
            3,
        );
        ar_bad.recv_buffer = b_out;
        ar_bad.reduction_op = 99;
        assert_and_free_pjrt_error(
            PJRT_Communicator_ReduceScatter(ptr::addr_of_mut!(ar_bad).cast()),
            3,
        );

        let mut ag_bad = PJRT_Communicator_AllGather_Args {
            struct_size: size_of::<PJRT_Communicator_AllGather_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm1,
            send_buffer: b_in,
            recv_buffer: b_big,
            count: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag_bad).cast()),
            3,
        );

        let mut cp_bad = PJRT_Communicator_CollectivePermute_Args {
            struct_size: size_of::<PJRT_Communicator_CollectivePermute_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm1,
            send_buffer: b_in,
            recv_buffer: b_big,
            target_ranks: ptr::null(),
            count: 0,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllToAll(ptr::addr_of_mut!(cp_bad).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_bad).cast()),
            3,
        );
        let neg_src_ranks = [-1_i32, 0_i32];
        let neg_dst_ranks = [0_i32, -1_i32];
        cp_bad.recv_buffer = b_out;
        cp_bad.target_ranks = neg_src_ranks.as_ptr();
        cp_bad.count = 2;
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_bad).cast()),
            3,
        );
        cp_bad.target_ranks = neg_dst_ranks.as_ptr();
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_bad).cast()),
            3,
        );
        let odd_ranks = [0_i32];
        cp_bad.target_ranks = odd_ranks.as_ptr();
        cp_bad.count = 1;
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_bad).cast()),
            3,
        );
        cp_bad.count = 0;
        assert!(PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_bad).cast()).is_null());
        let mut ev_del_cp = PJRT_Event_Destroy_Args {
            struct_size: size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event: cp_bad.event,
        };
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del_cp).cast()).is_null());
        cp_bad.target_ranks = ptr::null();
        assert!(PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_bad).cast()).is_null());
        ev_del_cp.event = cp_bad.event;
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del_cp).cast()).is_null());

        // MakeReceiveBuffers error branches + auto-allocation + async receive-before-send rendez-vous
        let mut xh_mr_err = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client: ptr::null_mut(),
            buffers: ptr::null(),
            num_buffers: 0,
            transfer_key: 0,
            out_event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_MakeReceiveBuffers(ptr::addr_of_mut!(xh_mr_err).cast()),
            3,
        );
        xh_mr_err.client = client;
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_MakeReceiveBuffers(ptr::addr_of_mut!(xh_mr_err).cast()),
            3,
        );

        let mut mr_slots: [*mut PJRT_Buffer; 2] = [ptr::null_mut(), b_out];
        let mut xh_mrecv = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: mr_slots.as_mut_ptr().cast_const(),
            num_buffers: 2,
            transfer_key: 777_888,
            out_event: ptr::null_mut(),
        };
        assert!(
            PJRT_CrossHostTransfers_MakeReceiveBuffers(ptr::addr_of_mut!(xh_mrecv).cast())
                .is_null()
        );
        assert!(!mr_slots[0].is_null());
        let pending_ev = xh_mrecv.out_event;
        assert!(!pending_ev.is_null());
        let mut rdy_count = 0_usize;
        let mut on_rdy = crate::pjrt_types::PJRT_Event_OnReady_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Event_OnReady_Args>(),
            extension_start: ptr::null_mut(),
            event: pending_ev,
            callback: Some(cross_host_ready_cb),
            user_arg: ptr::addr_of_mut!(rdy_count).cast(),
        };
        assert!(crate::PJRT_Event_OnReady(ptr::addr_of_mut!(on_rdy).cast()).is_null());

        let send_2 = [b_in, b_in];
        let mut xh_send = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: send_2.as_ptr(),
            num_buffers: 2,
            transfer_key: 777_888,
            out_event: ptr::null_mut(),
        };
        assert!(PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_send).cast()).is_null());
        assert_eq!(rdy_count, 1);
        for ev_p in [pending_ev, xh_send.out_event] {
            let mut ed = PJRT_Event_Destroy_Args {
                struct_size: size_of::<PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ev_p,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ed).cast()).is_null());
        }
        let mut bd_mr0 = PJRT_Buffer_Destroy_Args {
            struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            buffer: mr_slots[0],
        };
        assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(bd_mr0).cast()).is_null());

        for c_ptr in [comms_3[0], comms_3[1], comms_3[2], comm1] {
            let mut c_del = PJRT_Communicator_Destroy_Args {
                struct_size: size_of::<PJRT_Communicator_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                communicator: c_ptr,
            };
            assert!(PJRT_Communicator_Destroy(ptr::addr_of_mut!(c_del).cast()).is_null());
        }

        // 2. Compiler: XlaTransform UTF-8/empty errors, PhaseCompile unknown phase, Relocatables, Triton & FFI edge branches
        let bad_utf8 = [0xff_u8, 0xfe_u8];
        let mut xt_reg = PJRT_XlaTransform_Register_Args {
            struct_size: size_of::<PJRT_XlaTransform_Register_Args>(),
            extension_start: ptr::null_mut(),
            name: ptr::null(),
            name_size: 0,
            transform_fn: 1,
        };
        assert_and_free_pjrt_error(
            PJRT_XlaTransform_Register(ptr::addr_of_mut!(xt_reg).cast()),
            3,
        );
        xt_reg.name = c"x".as_ptr();
        xt_reg.name_size = 0;
        assert_and_free_pjrt_error(
            PJRT_XlaTransform_Register(ptr::addr_of_mut!(xt_reg).cast()),
            3,
        );
        xt_reg.name = bad_utf8.as_ptr().cast();
        xt_reg.name_size = bad_utf8.len();
        assert_and_free_pjrt_error(
            PJRT_XlaTransform_Register(ptr::addr_of_mut!(xt_reg).cast()),
            3,
        );
        xt_reg.name = c"   ".as_ptr();
        xt_reg.name_size = 3;
        assert_and_free_pjrt_error(
            PJRT_XlaTransform_Register(ptr::addr_of_mut!(xt_reg).cast()),
            3,
        );

        let bad_phase_cstr = [0xff_u8, 0x00_u8];
        let mut run_unk = PJRT_PhaseCompile_RunPhase_Args {
            struct_size: size_of::<PJRT_PhaseCompile_RunPhase_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: ptr::null_mut(),
            input_bytes: ptr::null(),
            input_size: 0,
            phase_name: c"unknown_phase_xyz".as_ptr(),
            output_status: 0,
        };
        assert!(PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(run_unk).cast()).is_null());
        run_unk.phase_name = bad_phase_cstr.as_ptr().cast();
        assert!(PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(run_unk).cast()).is_null());

        let mut ctr_null = PJRT_PhaseCompile_CompileToRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_CompileToRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: ptr::null(),
            client: ptr::null_mut(),
            program: ptr::null(),
            compile_options: ptr::null(),
            compile_options_size: 0,
            topology: ptr::null(),
            relocatables: ptr::null(),
            num_relocatables: 0,
            relocatables_buffer: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(ctr_null).cast()),
            3,
        );
        let mut empty_prog = PJRT_Program {
            struct_size: size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: ptr::null_mut(),
            code_size: 0,
            format: ptr::null(),
            format_size: 0,
        };
        ctr_null.program = ptr::addr_of!(empty_prog);
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(ctr_null).cast()),
            3,
        );
        let mut mlir_code = b"module @rel { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }".to_vec();
        empty_prog.code = mlir_code.as_mut_ptr().cast();
        empty_prog.code_size = 0;
        ctr_null.program = ptr::addr_of!(empty_prog);
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(ctr_null).cast()),
            3,
        );
        empty_prog.code_size = mlir_code.len();
        empty_prog.format = bad_utf8.as_ptr().cast();
        empty_prog.format_size = bad_utf8.len();
        ctr_null.program = ptr::addr_of!(empty_prog);
        let dummy_opts = [0x1a_u8, 0x00_u8];
        ctr_null.compile_options = dummy_opts.as_ptr().cast();
        ctr_null.compile_options_size = dummy_opts.len();
        assert!(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(ctr_null).cast()).is_null()
        );
        let rel1 = *ctr_null.relocatables;
        let rel1_buf = ctr_null.relocatables_buffer;

        empty_prog.format = ptr::null();
        empty_prog.format_size = 0;
        ctr_null.program = ptr::addr_of!(empty_prog);
        ctr_null.compile_options = ptr::null();
        ctr_null.compile_options_size = 0;
        assert!(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(ctr_null).cast()).is_null()
        );
        let rel2 = *ctr_null.relocatables;
        let rel2_buf = ctr_null.relocatables_buffer;

        // Relocatable Name unregistered error
        let mut r_name_bad = PJRT_Relocatable_StringQuery_Args {
            struct_size: size_of::<PJRT_Relocatable_StringQuery_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: ptr::null(),
            value_ptr: ptr::null(),
            value_size: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Relocatable_Name(ptr::addr_of_mut!(r_name_bad).cast()),
            3,
        );

        let bad_rel_bytes = b"corrupt_relocatable";
        let mut drel_bad = PJRT_PhaseCompile_DeserializeRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_DeserializeRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: ptr::null(),
            serialized_relocatable: bad_rel_bytes.as_ptr().cast(),
            serialized_relocatable_size: bad_rel_bytes.len(),
            relocatable: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_DeserializeRelocatable(ptr::addr_of_mut!(drel_bad).cast()),
            13,
        );

        let link_null0 = [ptr::null_mut::<crate::pjrt_types::PJRT_Relocatable>()];
        let mut lnk_args = PJRT_PhaseCompile_LinkRelocatables_Args {
            struct_size: size_of::<PJRT_PhaseCompile_LinkRelocatables_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: ptr::null(),
            relocatables: link_null0.as_ptr(),
            num_relocatables: 1,
            program: ptr::null(),
            compile_options: ptr::null(),
            compile_options_size: 0,
            topology: ptr::null(),
            executable: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(lnk_args).cast()),
            3,
        );
        let link_null1 = [rel1, ptr::null_mut()];
        lnk_args.relocatables = link_null1.as_ptr();
        lnk_args.num_relocatables = 2;
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(lnk_args).cast()),
            3,
        );
        let link_two = [rel1, rel2];
        lnk_args.relocatables = link_two.as_ptr();
        assert!(PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(lnk_args).cast()).is_null());
        let mut ex_del = crate::pjrt_types::PJRT_Executable_Destroy_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: lnk_args.executable,
        };
        assert!(crate::PJRT_Executable_Destroy(ptr::addr_of_mut!(ex_del).cast()).is_null());

        for r_ptr in [rel1, rel2] {
            let mut r_del = PJRT_Relocatable_Destroy_Args {
                struct_size: size_of::<PJRT_Relocatable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                relocatable: r_ptr,
            };
            assert!(PJRT_Relocatable_Destroy(ptr::addr_of_mut!(r_del).cast()).is_null());
        }
        for b_ptr in [rel1_buf, rel2_buf] {
            let mut b_del = PJRT_PhaseCompile_Destroy_Args {
                struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                handle: b_ptr,
            };
            assert!(PJRT_PhaseCompile_CBuffersDestroy(ptr::addr_of_mut!(b_del).cast()).is_null());
        }

        for (bad_ttir, exp_code) in [
            (bad_utf8.as_slice(), 13_i32),
            (b"module".as_slice(), 13_i32),
            (b"module { func.func @k() {".as_slice(), 13_i32),
            (b"unrecognized_syntax".as_slice(), 13_i32),
        ] {
            let mut tc_bad = PJRT_Triton_Compile_Args {
                struct_size: size_of::<PJRT_Triton_Compile_Args>(),
                extension_start: ptr::null_mut(),
                module: bad_ttir.as_ptr().cast(),
                module_size: bad_ttir.len(),
                arch_name: c"v6e".as_ptr(),
                arch_name_size: 3,
                num_warps: 4,
                num_ctas: 1,
                num_stages: 2,
                pad: 0,
                out_asm: ptr::null(),
                out_asm_size: 0,
                out_smem_bytes: 0,
            };
            assert_and_free_pjrt_error(
                PJRT_Triton_Compile(ptr::addr_of_mut!(tc_bad).cast()),
                exp_code,
            );
        }
        PJRT_Triton_FreeAsm(ptr::null(), 0);
        let valid_ttir = b"module { tt.func @k() { tt.return } }";
        let mut tc_ok = PJRT_Triton_Compile_Args {
            struct_size: size_of::<PJRT_Triton_Compile_Args>(),
            extension_start: ptr::null_mut(),
            module: valid_ttir.as_ptr().cast(),
            module_size: valid_ttir.len(),
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
        assert!(PJRT_Triton_Compile(ptr::addr_of_mut!(tc_ok).cast()).is_null());
        PJRT_Triton_FreeAsm(tc_ok.out_asm, tc_ok.out_asm_size);

        let mut tid_null = [40_usize, 0, 0, 0, 0, 0];
        assert!(PJRT_FFI_TypeID_Register(tid_null.as_mut_ptr().cast()).is_null());

        let mut del_calls = 0_usize;
        let mut ud40 = PJRT_FFI_UserData_Add_Args {
            struct_size: 40,
            extension_start: ptr::null_mut(),
            context: ptr::null_mut(),
            type_id: 1,
            data: ptr::addr_of_mut!(del_calls).cast(),
            deleter: Some(userdata_deleter),
        };
        assert!(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ud40).cast()).is_null());
        assert_eq!(del_calls, 0);
        ud40.struct_size = 48;
        assert!(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ud40).cast()).is_null());
        assert_eq!(del_calls, 1);
        ud40.data = ptr::null_mut();
        assert!(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ud40).cast()).is_null());
        ud40.deleter = None;
        assert!(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ud40).cast()).is_null());
        ud40.struct_size = 8;
        assert_and_free_pjrt_error(PJRT_FFI_UserData_Add(ptr::addr_of_mut!(ud40).cast()), 3);

        let mut rh = PJRT_FFI_Register_Handler_Args {
            struct_size: 52,
            extension_start: ptr::null_mut(),
            target_name: ptr::null(),
            target_name_size: 0,
            handler: None,
            platform_name: ptr::null(),
            platform_name_size: 0,
            traits: 0,
            pad: 0,
        };
        assert!(PJRT_FFI_Register_Handler(ptr::addr_of_mut!(rh).cast()).is_null());
        rh.struct_size = 64;
        rh.target_name = c"op".as_ptr();
        rh.target_name_size = 0;
        assert!(PJRT_FFI_Register_Handler(ptr::addr_of_mut!(rh).cast()).is_null());
        rh.struct_size = 8;
        assert_and_free_pjrt_error(PJRT_FFI_Register_Handler(ptr::addr_of_mut!(rh).cast()), 3);

        // 3. Executable: AbiVersion compatibility permutations & FromProto edge cases + SetTpuCompilationEnv corrupt proto
        let mk_rt_abi = |ver: &[u8], plat: u64| -> *mut crate::pjrt_types::PJRT_AbiVersion_Holder {
            use libtpu_core::proto::ProtoSchemaMessage as _;
            let proto = libtpu_core::proto::RuntimeAbiVersionProto {
                platform: plat,
                version: ver.to_vec(),
            }
            .encode_proto();
            let mut fp = PJRT_AbiVersion_FromProto_Args {
                struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
                extension_start: ptr::null_mut(),
                serialized_proto: if proto.is_empty() {
                    ptr::null()
                } else {
                    proto.as_ptr().cast()
                },
                serialized_proto_size: proto.len(),
                abi_version_out: ptr::null_mut(),
            };
            assert!(PJRT_AbiVersion_RuntimeFromProto(ptr::addr_of_mut!(fp).cast()).is_null());
            fp.abi_version_out
        };
        let mk_ex_abi = |ver: &[u8], plat: u64| -> *mut crate::pjrt_types::PJRT_AbiVersion_Holder {
            use libtpu_core::proto::ProtoSchemaMessage as _;
            let proto = libtpu_core::proto::ExecutableAbiVersionProto {
                platform: plat,
                version: ver.to_vec(),
            }
            .encode_proto();
            let mut fp = PJRT_AbiVersion_FromProto_Args {
                struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
                extension_start: ptr::null_mut(),
                serialized_proto: if proto.is_empty() {
                    ptr::null()
                } else {
                    proto.as_ptr().cast()
                },
                serialized_proto_size: proto.len(),
                abi_version_out: ptr::null_mut(),
            };
            assert!(PJRT_AbiVersion_ExecutableFromProto(ptr::addr_of_mut!(fp).cast()).is_null());
            fp.abi_version_out
        };

        let rt_base = mk_rt_abi(b"0.117.1-v5e", 0);
        let rt_diff_plat = mk_rt_abi(b"0.117.1-v5e", 77);
        let rt_diff_maj = mk_rt_abi(b"1.117.1-v5e", 0);
        let rt_diff_min = mk_rt_abi(b"0.116.1-v5e", 0);
        let rt_diff_gen = mk_rt_abi(b"0.117.1-v4", 0);
        let ex_ok = mk_ex_abi(b"0.116.1-v5e", 0);
        let ex_newer_min = mk_ex_abi(b"0.118.1-v5e", 0);
        let ex_diff_maj = mk_ex_abi(b"1.117.1-v5e", 0);
        let ex_diff_plat = mk_ex_abi(b"0.117.1-v5e", 77);
        let ex_diff_gen = mk_ex_abi(b"0.117.1-v4", 0);

        for peer in [rt_base, rt_diff_plat, rt_diff_maj, rt_diff_min, rt_diff_gen] {
            let mut cmp = PJRT_AbiVersion_IsCompatible_Args {
                struct_size: size_of::<PJRT_AbiVersion_IsCompatible_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: rt_base,
                other_abi_version: peer,
                is_compatible_out: 99,
            };
            assert!(
                PJRT_AbiVersion_RuntimeCompatibleWithRuntime(ptr::addr_of_mut!(cmp).cast())
                    .is_null()
            );
        }
        for ex_peer in [ex_ok, ex_newer_min, ex_diff_maj, ex_diff_plat, ex_diff_gen] {
            let mut cmp = PJRT_AbiVersion_IsCompatible_Args {
                struct_size: size_of::<PJRT_AbiVersion_IsCompatible_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: rt_base,
                other_abi_version: ex_peer,
                is_compatible_out: 99,
            };
            assert!(
                PJRT_AbiVersion_RuntimeCompatibleWithExecutable(ptr::addr_of_mut!(cmp).cast())
                    .is_null()
            );
        }

        let corrupt_wire = [0xff_u8; 12];
        for from_fn in [
            PJRT_AbiVersion_RuntimeFromProto,
            PJRT_AbiVersion_ExecutableFromProto,
        ] {
            let mut fp_null_nonzero = PJRT_AbiVersion_FromProto_Args {
                struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
                extension_start: ptr::null_mut(),
                serialized_proto: ptr::null(),
                serialized_proto_size: 4,
                abi_version_out: ptr::null_mut(),
            };
            assert_and_free_pjrt_error(from_fn(ptr::addr_of_mut!(fp_null_nonzero).cast()), 3);
            fp_null_nonzero.serialized_proto_size = 0;
            assert!(from_fn(ptr::addr_of_mut!(fp_null_nonzero).cast()).is_null());
            let mut del = PJRT_AbiVersion_Destroy_Args {
                struct_size: size_of::<PJRT_AbiVersion_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: fp_null_nonzero.abi_version_out,
            };
            assert!(PJRT_AbiVersion_RuntimeDestroy(ptr::addr_of_mut!(del).cast()).is_null());

            let mut fp_corrupt = PJRT_AbiVersion_FromProto_Args {
                struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
                extension_start: ptr::null_mut(),
                serialized_proto: corrupt_wire.as_ptr().cast(),
                serialized_proto_size: corrupt_wire.len(),
                abi_version_out: ptr::null_mut(),
            };
            assert_and_free_pjrt_error(from_fn(ptr::addr_of_mut!(fp_corrupt).cast()), 3);
        }

        for h in [
            rt_base,
            rt_diff_plat,
            rt_diff_maj,
            rt_diff_min,
            rt_diff_gen,
            ex_ok,
            ex_newer_min,
            ex_diff_maj,
            ex_diff_plat,
            ex_diff_gen,
        ] {
            let mut del = PJRT_AbiVersion_Destroy_Args {
                struct_size: size_of::<PJRT_AbiVersion_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: h,
            };
            assert!(PJRT_AbiVersion_RuntimeDestroy(ptr::addr_of_mut!(del).cast()).is_null());
        }

        let mut comp_code = b"module @m { func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %a, %b : tensor<4xf32> return %0 : tensor<4xf32> } }".to_vec();
        let comp_prog = PJRT_Program {
            struct_size: size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: comp_code.as_mut_ptr().cast(),
            code_size: comp_code.len(),
            format: c"mlir".as_ptr(),
            format_size: 4,
        };
        let mut ccomp = PJRT_Client_Compile_Args {
            struct_size: size_of::<PJRT_Client_Compile_Args>(),
            extension_start: ptr::null_mut(),
            client,
            program: ptr::addr_of!(comp_prog),
            compile_options: ptr::null(),
            compile_options_size: 0,
            executable: ptr::null_mut(),
        };
        assert!(PJRT_Client_Compile(ptr::addr_of_mut!(ccomp).cast()).is_null());
        let mut gex = crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_LoadedExecutable_GetExecutable_Args>(),
            extension_start: ptr::null_mut(),
            loaded_executable: ccomp.executable,
            executable: ptr::null_mut(),
        };
        assert!(
            crate::PJRT_LoadedExecutable_GetExecutable(ptr::addr_of_mut!(gex).cast()).is_null()
        );
        let exec_ptr = gex.executable;

        let mut set_env_bad = PJRT_TpuExecutable_SetTpuCompilationEnv_Args {
            struct_size: size_of::<PJRT_TpuExecutable_SetTpuCompilationEnv_Args>(),
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
            serialized_env: corrupt_wire.as_ptr(),
            serialized_env_size: corrupt_wire.len(),
        };
        assert!(
            PJRT_TpuExecutable_SetTpuCompilationEnv(ptr::addr_of_mut!(set_env_bad).cast())
                .is_null()
        );
        set_env_bad.serialized_env = ptr::null();
        set_env_bad.serialized_env_size = 0;
        assert!(
            PJRT_TpuExecutable_SetTpuCompilationEnv(ptr::addr_of_mut!(set_env_bad).cast())
                .is_null()
        );

        // 4. Memory: V5p SparseCore descriptions, empty shapes layout builder, cached layouts, RawBuffer error/zero-len branches
        let empty_shapes = memory::build_layout_handles_for_shapes(&[]);
        assert_eq!(empty_shapes.len(), 1);
        let mut ld_empty = PJRT_Layouts_MemoryLayout_Destroy_Args {
            struct_size: size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            layout: empty_shapes.first().copied().unwrap_or(ptr::null_mut()),
        };
        assert!(PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(ld_empty).cast()).is_null());

        assert_and_free_pjrt_error(PJRT_Layouts_Executable_GetOutputLayouts(ptr::null_mut()), 3);
        let mut small_lay = [8_usize, 0, 0, 0, 0];
        assert_and_free_pjrt_error(
            PJRT_Layouts_Executable_GetOutputLayouts(small_lay.as_mut_ptr().cast()),
            3,
        );
        for _ in 0..2 {
            let mut ol32 = PJRT_Layouts_PJRT_Executable_Layouts_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                executable: exec_ptr,
                num_layouts: 0,
                layouts: ptr::null(),
            };
            assert!(
                PJRT_Layouts_Executable_GetOutputLayouts(ptr::addr_of_mut!(ol32).cast()).is_null()
            );
            assert!(
                PJRT_Layouts_Executable_GetParameterLayouts(ptr::addr_of_mut!(ol32).cast())
                    .is_null()
            );
        }

        let mut tc_v5p = crate::pjrt_types::PJRT_TopologyDescription_Create_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(),
            extension_start: ptr::null_mut(),
            topology_name: c"tpu_v5p".as_ptr(),
            topology_name_size: 7,
            create_options: ptr::null(),
            num_options: 0,
            topology: ptr::null_mut(),
        };
        assert!(crate::PJRT_TopologyDescription_Create(ptr::addr_of_mut!(tc_v5p).cast()).is_null());
        let mut gdd = crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args {
            struct_size: size_of::<
                crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args,
            >(),
            extension_start: ptr::null_mut(),
            topology: tc_v5p.topology,
            descriptions: ptr::null(),
            num_descriptions: 0,
        };
        assert!(
            crate::PJRT_TopologyDescription_GetDeviceDescriptions(ptr::addr_of_mut!(gdd).cast())
                .is_null()
        );
        let v5p_desc = *gdd.descriptions;
        let mut md_v5p = [48_usize, 0, v5p_desc as usize, 0, 0, 0];
        assert!(PJRT_MemoryDescriptions_Get(md_v5p.as_mut_ptr().cast()).is_null());
        assert_eq!(md_v5p[4], 4);

        let high_coords = [99_i32, 99, 99];
        let mut unmatched_coord = PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args {
            struct_size: size_of::<PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args>(),
            extension_start: ptr::null_mut(),
            topology: tc_v5p.topology,
            coords: high_coords.as_ptr(),
            core_idx: 0,
            device_id_out: 0,
        };
        assert!(
            PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(
                ptr::addr_of_mut!(unmatched_coord).cast()
            )
            .is_null()
        );

        let mut create_rb = PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args {
            struct_size: size_of::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>(),
            extension_start: ptr::null_mut(),
            buffer: b_in,
            raw_buffer_out: ptr::null_mut(),
        };
        assert!(
            PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(create_rb).cast()).is_null()
        );
        let rb = create_rb.raw_buffer_out;
        let dangling_rb = std::ptr::dangling_mut::<crate::pjrt_types::PJRT_RawBuffer>();

        let mut tmp_byte = [0_u8; 4];
        for bad_rb in [ptr::null_mut(), dangling_rb] {
            let mut gs = PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args {
                struct_size: size_of::<PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args>(),
                extension_start: ptr::null_mut(),
                raw_buffer: bad_rb,
                on_device_size_in_bytes: 0,
            };
            assert_and_free_pjrt_error(
                PJRT_RawBuffer_GetOnDeviceSizeInBytes(ptr::addr_of_mut!(gs).cast()),
                3,
            );
            let mut gm = PJRT_RawBuffer_GetMemorySpace_Args {
                struct_size: size_of::<PJRT_RawBuffer_GetMemorySpace_Args>(),
                extension_start: ptr::null_mut(),
                raw_buffer: bad_rb,
                memory_space_out: ptr::null_mut(),
            };
            assert_and_free_pjrt_error(
                PJRT_RawBuffer_GetMemorySpace(ptr::addr_of_mut!(gm).cast()),
                3,
            );
            let mut cp = PJRT_RawBuffer_CopyRaw_Args {
                struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
                extension_start: ptr::null_mut(),
                buffer: bad_rb,
                host_ptr: tmp_byte.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 4,
                event: ptr::null_mut(),
            };
            assert_and_free_pjrt_error(
                PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(cp).cast()),
                3,
            );
            assert_and_free_pjrt_error(
                PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(cp).cast()),
                3,
            );
        }

        for (off, tsz, host_p, exp_code) in [
            (0_i64, 4_i64, ptr::null_mut::<std::ffi::c_void>(), 3_i32),
            (-1_i64, 4_i64, tmp_byte.as_mut_ptr().cast(), 3_i32),
            (0_i64, -1_i64, tmp_byte.as_mut_ptr().cast(), 3_i32),
            (i64::MAX, i64::MAX, tmp_byte.as_mut_ptr().cast(), 3_i32),
            (0_i64, 1024_i64, tmp_byte.as_mut_ptr().cast(), 3_i32),
        ] {
            let mut cp = PJRT_RawBuffer_CopyRaw_Args {
                struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
                extension_start: ptr::null_mut(),
                buffer: rb,
                host_ptr: host_p,
                offset: off,
                transfer_size: tsz,
                event: ptr::null_mut(),
            };
            assert_and_free_pjrt_error(
                PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(cp).cast()),
                exp_code,
            );
            assert_and_free_pjrt_error(
                PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(cp).cast()),
                exp_code,
            );
        }
        for copy_fn in [
            PJRT_RawBuffer_CopyRawHostToDevice,
            PJRT_RawBuffer_CopyRawDeviceToHost,
        ] {
            let mut cp0 = PJRT_RawBuffer_CopyRaw_Args {
                struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
                extension_start: ptr::null_mut(),
                buffer: rb,
                host_ptr: tmp_byte.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 0,
                event: ptr::null_mut(),
            };
            assert!(copy_fn(ptr::addr_of_mut!(cp0).cast()).is_null());
            let mut ed0 = PJRT_Event_Destroy_Args {
                struct_size: size_of::<PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: cp0.event,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ed0).cast()).is_null());
        }
        let mut rb_del = PJRT_RawBuffer_Destroy_Args {
            struct_size: size_of::<PJRT_RawBuffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: rb,
        };
        assert!(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(rb_del).cast()).is_null());

        // 5. Megascale: unregistered MultiSlice_Config error branches & cleared error callback AbortCollectives
        let dangling_ms = std::ptr::dangling_mut::<crate::pjrt_types::PJRT_MultiSlice_Config>();
        let mut ms_ns = PJRT_MultiSlice_Config_NumSlices_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_NumSlices_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
            num_slices: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_MultiSlice_Config_NumSlices(ptr::addr_of_mut!(ms_ns).cast()),
            3,
        );
        let mut ms_nd = PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_NumDevicesPerSlice_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
            num_devices_per_slice_map: 0,
            num_slices: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_MultiSlice_Config_NumDevicesPerSlice(ptr::addr_of_mut!(ms_nd).cast()),
            3,
        );
        let mut ms_sid = PJRT_MultiSlice_Config_SliceId_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_SliceId_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
            slice_id: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_MultiSlice_Config_SliceId(ptr::addr_of_mut!(ms_sid).cast()),
            3,
        );
        let mut ms_ser = PJRT_MultiSlice_Config_Serialize_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_Serialize_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
            serialized_bytes: ptr::null(),
            serialized_bytes_size: 0,
            serialized_config: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_MultiSlice_Config_Serialize(ptr::addr_of_mut!(ms_ser).cast()),
            3,
        );
        let mut mg_ns = PJRT_Megascale_ConfigNumSlices_Args {
            struct_size: size_of::<PJRT_Megascale_ConfigNumSlices_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
            num_slices: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_ConfigNumSlices(ptr::addr_of_mut!(mg_ns).cast()),
            3,
        );
        let mut mg_nd = PJRT_Megascale_ConfigNumDevicesPerSlice_Args {
            struct_size: size_of::<PJRT_Megascale_ConfigNumDevicesPerSlice_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
            num_devices_per_slice: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_ConfigNumDevicesPerSlice(ptr::addr_of_mut!(mg_nd).cast()),
            3,
        );
        let mut mg_sid = PJRT_Megascale_ConfigSliceId_Args {
            struct_size: size_of::<PJRT_Megascale_ConfigSliceId_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
            slice_id: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_ConfigSliceId(ptr::addr_of_mut!(mg_sid).cast()),
            3,
        );
        let mut mg_inc = PJRT_Megascale_MultiSliceConfig_GetIncarnationId_Args {
            struct_size: size_of::<PJRT_Megascale_MultiSliceConfig_GetIncarnationId_Args>(),
            config: dangling_ms,
            incarnation_id: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_MultiSliceConfig_GetIncarnationId(ptr::addr_of_mut!(mg_inc).cast()),
            3,
        );
        let mut mg_gb = PJRT_Megascale_GlobalBarrier_Args {
            struct_size: size_of::<PJRT_Megascale_GlobalBarrier_Args>(),
            config: dangling_ms,
            barrier_id: ptr::null(),
            barrier_id_size: 0,
            timeout_ms: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_GlobalBarrier(ptr::addr_of_mut!(mg_gb).cast()),
            3,
        );

        for host_fn in [
            PJRT_Megascale_MultiSliceConfig_GetLocalHostId,
            PJRT_Megascale_MultiSliceConfig_GetNumHosts,
        ] {
            assert_and_free_pjrt_error(host_fn(ptr::null_mut()), 3);
            let mut small = [8_usize, 0, 0];
            assert_and_free_pjrt_error(host_fn(small.as_mut_ptr().cast()), 3);
            let mut bad_cfg = PJRT_Megascale_MultiSliceConfig_GetLocalHostId_Args {
                struct_size: 20,
                config: dangling_ms,
                local_host_id: 0,
                pad: 0,
            };
            assert_and_free_pjrt_error(host_fn(ptr::addr_of_mut!(bad_cfg).cast()), 3);
        }

        let mut reh_some = PJRT_Megascale_RegisterErrorHandler_Args {
            struct_size: size_of::<PJRT_Megascale_RegisterErrorHandler_Args>(),
            extension_start: ptr::null_mut(),
            callback_fn: Some(userdata_deleter),
            handler_id_out: 0,
        };
        assert!(PJRT_Megascale_RegisterErrorHandler(ptr::addr_of_mut!(reh_some).cast()).is_null());
        let mut ac_some = PJRT_Megascale_AbortCollectives_Args {
            struct_size: size_of::<PJRT_Megascale_AbortCollectives_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_AbortCollectives(ptr::addr_of_mut!(ac_some).cast()).is_null());

        let mut reh_none = PJRT_Megascale_RegisterErrorHandler_Args {
            struct_size: size_of::<PJRT_Megascale_RegisterErrorHandler_Args>(),
            extension_start: ptr::null_mut(),
            callback_fn: None,
            handler_id_out: 0,
        };
        assert!(PJRT_Megascale_RegisterErrorHandler(ptr::addr_of_mut!(reh_none).cast()).is_null());
        let mut ac_none = PJRT_Megascale_AbortCollectives_Args {
            struct_size: size_of::<PJRT_Megascale_AbortCollectives_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_AbortCollectives(ptr::addr_of_mut!(ac_none).cast()).is_null());

        let mut ds_null = PJRT_Megascale_DiscoverSlices_Args {
            struct_size: size_of::<PJRT_Megascale_DiscoverSlices_Args>(),
            client_context: ptr::null_mut(),
            multi_slice_config: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_DiscoverSlices(ptr::addr_of_mut!(ds_null).cast()).is_null());
        let mut msc_del0 = PJRT_MultiSlice_Config_Destroy_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            config: ds_null.multi_slice_config,
        };
        assert!(PJRT_MultiSlice_Config_Destroy(ptr::addr_of_mut!(msc_del0).cast()).is_null());

        // 6. Profiler: null-handle Start/Stop/CollectData, valid Serialize, & negotiated profiler 40B/null branches
        let mut p_start_null = PLUGIN_Profiler_Start_Args {
            struct_size: size_of::<PLUGIN_Profiler_Start_Args>(),
            extension_start: ptr::null_mut(),
            profiler: ptr::null_mut(),
        };
        assert!(PLUGIN_Profiler_Start(ptr::addr_of_mut!(p_start_null).cast()).is_null());
        let mut p_stop_null = PLUGIN_Profiler_Stop_Args {
            struct_size: size_of::<PLUGIN_Profiler_Stop_Args>(),
            extension_start: ptr::null_mut(),
            profiler: ptr::null_mut(),
        };
        assert!(PLUGIN_Profiler_Stop(ptr::addr_of_mut!(p_stop_null).cast()).is_null());
        let mut p_cd_null = PLUGIN_Profiler_CollectData_Args {
            struct_size: size_of::<PLUGIN_Profiler_CollectData_Args>(),
            extension_start: ptr::null_mut(),
            profiler: ptr::null_mut(),
            buffer: ptr::null_mut(),
            size_in_bytes: ptr::null_mut(),
        };
        assert!(PLUGIN_Profiler_CollectData(ptr::addr_of_mut!(p_cd_null).cast()).is_null());

        let mut p_create = PLUGIN_Profiler_Create_Args {
            struct_size: size_of::<PLUGIN_Profiler_Create_Args>(),
            extension_start: ptr::null_mut(),
            options: ptr::null(),
            options_size: 0,
            profiler: ptr::null_mut(),
        };
        assert!(PLUGIN_Profiler_Create(ptr::addr_of_mut!(p_create).cast()).is_null());
        let p_inst = p_create.profiler;
        let mut p_cons = PLUGIN_Profiler_Consume_Args {
            struct_size: size_of::<PLUGIN_Profiler_Consume_Args>(),
            profiler: p_inst,
            result: ptr::null_mut(),
        };
        assert!(PLUGIN_Profiler_Consume(ptr::addr_of_mut!(p_cons).cast()).is_null());
        let mut p_ser_ok = PLUGIN_Profiler_Serialize_Args {
            struct_size: size_of::<PLUGIN_Profiler_Serialize_Args>(),
            profiler: p_inst,
            result: p_cons.result,
            buffer: ptr::null(),
            size_in_bytes: 0,
        };
        assert!(PLUGIN_Profiler_Serialize(ptr::addr_of_mut!(p_ser_ok).cast()).is_null());
        assert!(p_ser_ok.size_in_bytes > 0);
        let mut p_cdel = PLUGIN_Profiler_ConsumeResult_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_ConsumeResult_Destroy_Args>(),
            result: p_cons.result,
        };
        assert!(PLUGIN_Profiler_ConsumeResult_Destroy(ptr::addr_of_mut!(p_cdel).cast()).is_null());
        p_cdel.result = ptr::null_mut();
        assert!(PLUGIN_Profiler_ConsumeResult_Destroy(ptr::addr_of_mut!(p_cdel).cast()).is_null());

        let neg_api = &*NEGOTIATED_EXT_PROFILER.profiler_api;
        for f in [
            neg_api.create,
            neg_api.destroy,
            neg_api.start,
            neg_api.stop,
            neg_api.collect_data,
        ] {
            assert!(f(ptr::null_mut()).is_null());
        }
        let mut neg_cr40 = PLUGIN_Profiler_Create_Args {
            struct_size: size_of::<PLUGIN_Profiler_Create_Args>(),
            extension_start: ptr::null_mut(),
            options: ptr::null(),
            options_size: 0,
            profiler: ptr::null_mut(),
        };
        assert!((neg_api.create)(ptr::addr_of_mut!(neg_cr40).cast()).is_null());
        let mut neg_st24 = PLUGIN_Profiler_Start_Args {
            struct_size: size_of::<PLUGIN_Profiler_Start_Args>(),
            extension_start: ptr::null_mut(),
            profiler: neg_cr40.profiler,
        };
        assert!((neg_api.start)(ptr::addr_of_mut!(neg_st24).cast()).is_null());
        assert!((neg_api.stop)(ptr::addr_of_mut!(neg_st24).cast()).is_null());
        let mut neg_sz = 0_usize;
        let mut neg_cd40 = PLUGIN_Profiler_CollectData_Args {
            struct_size: size_of::<PLUGIN_Profiler_CollectData_Args>(),
            extension_start: ptr::null_mut(),
            profiler: neg_cr40.profiler,
            buffer: ptr::null_mut(),
            size_in_bytes: ptr::addr_of_mut!(neg_sz),
        };
        assert!((neg_api.collect_data)(ptr::addr_of_mut!(neg_cd40).cast()).is_null());
        assert!((neg_api.destroy)(ptr::addr_of_mut!(neg_st24).cast()).is_null());

        let mut leg_null_inst = [16_usize, 0];
        assert!((neg_api.start)(leg_null_inst.as_mut_ptr().cast()).is_null());
        assert!((neg_api.stop)(leg_null_inst.as_mut_ptr().cast()).is_null());
        assert!((neg_api.destroy)(leg_null_inst.as_mut_ptr().cast()).is_null());
        let mut leg_null_cd = [32_usize, 0, 0, 0];
        assert!((neg_api.collect_data)(leg_null_cd.as_mut_ptr().cast()).is_null());

        let mut p_del = PLUGIN_Profiler_Destroy_Args {
            struct_size: size_of::<PLUGIN_Profiler_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            profiler: p_inst,
        };
        assert!(PLUGIN_Profiler_Destroy(ptr::addr_of_mut!(p_del).cast()).is_null());

        // 7. Remaining branch & function closure across collectives, compiler, executable, megascale, memory & topology
        assert!(collectives::allocate_group_communicator(&[]).is_err());
        let neg_devs = [-1_i32];
        let mut dup_comm = PJRT_Collectives_CreateCommunicator_Args {
            struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
            extension_start: ptr::null_mut(),
            collectives: client.cast(),
            device_ids: neg_devs.as_ptr(),
            num_devices: 1,
            communicator: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(dup_comm).cast()),
            3,
        );
        let same_devs = [0_i32, 0_i32];
        dup_comm.device_ids = same_devs.as_ptr();
        dup_comm.num_devices = 2;
        assert_and_free_pjrt_error(
            PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(dup_comm).cast()),
            3,
        );

        // Non-null buffers with num_buffers == 0 + pending event destroyed before SendBuffers
        let mut xh_zero_cnt = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: send_2.as_ptr(),
            num_buffers: 0,
            transfer_key: 777_889,
            out_event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_zero_cnt).cast()),
            3,
        );
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_MakeReceiveBuffers(ptr::addr_of_mut!(xh_zero_cnt).cast()),
            3,
        );

        let mut mr_one: [*mut PJRT_Buffer; 1] = [b_out];
        let mut xh_mrecv2 = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: mr_one.as_mut_ptr().cast_const(),
            num_buffers: 1,
            transfer_key: 777_890,
            out_event: ptr::null_mut(),
        };
        assert!(
            PJRT_CrossHostTransfers_MakeReceiveBuffers(ptr::addr_of_mut!(xh_mrecv2).cast())
                .is_null()
        );
        let mut ed_early = PJRT_Event_Destroy_Args {
            struct_size: size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event: xh_mrecv2.out_event,
        };
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ed_early).cast()).is_null());
        let send_one = [b_in];
        let mut xh_send2 = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: send_one.as_ptr(),
            num_buffers: 1,
            transfer_key: 777_890,
            out_event: ptr::null_mut(),
        };
        assert!(PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_send2).cast()).is_null());
        ed_early.event = xh_send2.out_event;
        assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ed_early).cast()).is_null());

        // Deleted PJRT_Buffer propagation across collective, cross-host & RawBuffer operations
        let b_deleted = upload_f32_buffer(client, &[1.0; 4]);
        let mut create_rb_del = PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args {
            struct_size: size_of::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>(),
            extension_start: ptr::null_mut(),
            buffer: b_deleted,
            raw_buffer_out: ptr::null_mut(),
        };
        assert!(
            PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(create_rb_del).cast())
                .is_null()
        );
        let rb_del_backing = create_rb_del.raw_buffer_out;

        let mut del_buf_args = crate::pjrt_types::PJRT_Buffer_Delete_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Delete_Args>(),
            extension_start: ptr::null_mut(),
            buffer: b_deleted,
        };
        assert!(crate::PJRT_Buffer_Delete(ptr::addr_of_mut!(del_buf_args).cast()).is_null());

        let mut cp_dealloc = PJRT_RawBuffer_CopyRaw_Args {
            struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
            extension_start: ptr::null_mut(),
            buffer: rb_del_backing,
            host_ptr: tmp_byte.as_mut_ptr().cast(),
            offset: 0,
            transfer_size: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(cp_dealloc).cast()),
            13,
        );
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(cp_dealloc).cast()),
            13,
        );
        let mut rb_del_b = PJRT_RawBuffer_Destroy_Args {
            struct_size: size_of::<PJRT_RawBuffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: rb_del_backing,
        };
        assert!(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(rb_del_b).cast()).is_null());

        let mut c_tmp_args = PJRT_Collectives_CreateCommunicator_Args {
            struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
            extension_start: ptr::null_mut(),
            collectives: client.cast(),
            device_ids: ptr::null(),
            num_devices: 0,
            communicator: ptr::null_mut(),
        };
        assert!(
            PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(c_tmp_args).cast()).is_null()
        );
        let comm_del = c_tmp_args.communicator;
        let mut ar_del = PJRT_Communicator_AllReduce_Args {
            struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm_del,
            send_buffer: b_in,
            recv_buffer: ptr::null_mut(),
            reduction_op: 0,
            pad: 0,
            count: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar_del).cast()),
            3,
        );
        ar_del.send_buffer = b_deleted;
        ar_del.recv_buffer = b_out;
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar_del).cast()),
            9,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_ReduceScatter(ptr::addr_of_mut!(ar_del).cast()),
            9,
        );
        ar_del.send_buffer = b_in;
        ar_del.recv_buffer = b_deleted;
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllReduce(ptr::addr_of_mut!(ar_del).cast()),
            9,
        );
        let mut ag_del = PJRT_Communicator_AllGather_Args {
            struct_size: size_of::<PJRT_Communicator_AllGather_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm_del,
            send_buffer: b_deleted,
            recv_buffer: b_out,
            count: 4,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag_del).cast()),
            9,
        );
        ag_del.send_buffer = b_in;
        ag_del.recv_buffer = b_deleted;
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag_del).cast()),
            13,
        );
        let mut cp_del = PJRT_Communicator_CollectivePermute_Args {
            struct_size: size_of::<PJRT_Communicator_CollectivePermute_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm_del,
            send_buffer: b_deleted,
            recv_buffer: b_out,
            target_ranks: ptr::null(),
            count: 0,
            event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Communicator_AllToAll(ptr::addr_of_mut!(cp_del).cast()),
            9,
        );
        assert_and_free_pjrt_error(
            PJRT_Communicator_CollectivePermute(ptr::addr_of_mut!(cp_del).cast()),
            9,
        );
        let del_arr = [b_deleted];
        let mut xh_del_buf = PJRT_CrossHostTransfers_Args {
            struct_size: size_of::<PJRT_CrossHostTransfers_Args>(),
            extension_start: ptr::null_mut(),
            client,
            buffers: del_arr.as_ptr(),
            num_buffers: 1,
            transfer_key: 777_891,
            out_event: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_SendBuffers(ptr::addr_of_mut!(xh_del_buf).cast()),
            9,
        );
        assert_and_free_pjrt_error(
            PJRT_CrossHostTransfers_ReceiveBuffers(ptr::addr_of_mut!(xh_del_buf).cast()),
            9,
        );
        let mut cd_tmp = PJRT_Communicator_Destroy_Args {
            struct_size: size_of::<PJRT_Communicator_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            communicator: comm_del,
        };
        assert!(PJRT_Communicator_Destroy(ptr::addr_of_mut!(cd_tmp).cast()).is_null());
        let mut bd_del = PJRT_Buffer_Destroy_Args {
            struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            buffer: b_deleted,
        };
        assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(bd_del).cast()).is_null());

        // Compiler remaining branches: Unregister null/empty, RunPhase invalid input, CompileToRelocatable format_size=0 / opts_size=0 / invalid MLIR, Relocatable_Name valid, Deserialize/Link size=0, Triton root variants, FreeAsm size=0, FFI Register_Handler handler=None
        let mut unreg_bad = PJRT_XlaTransform_Unregister_Args {
            struct_size: size_of::<PJRT_XlaTransform_Unregister_Args>(),
            extension_start: ptr::null_mut(),
            name: ptr::null(),
            name_size: 0,
        };
        assert_and_free_pjrt_error(
            PJRT_XlaTransform_Unregister(ptr::addr_of_mut!(unreg_bad).cast()),
            3,
        );

        let mut pc_get = PJRT_PhaseCompile_GetCompiler_Args {
            struct_size: size_of::<PJRT_PhaseCompile_GetCompiler_Args>(),
            extension_start: ptr::null_mut(),
            client,
            compiler_out: ptr::null_mut(),
        };
        assert!(PJRT_PhaseCompile_GetCompiler(ptr::addr_of_mut!(pc_get).cast()).is_null());
        let invalid_mlir = b"invalid {{{";
        let mut run_bad_in = PJRT_PhaseCompile_RunPhase_Args {
            struct_size: size_of::<PJRT_PhaseCompile_RunPhase_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: pc_get.compiler_out,
            input_bytes: invalid_mlir.as_ptr().cast(),
            input_size: invalid_mlir.len(),
            phase_name: ptr::null(),
            output_status: 0,
        };
        assert!(PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(run_bad_in).cast()).is_null());
        run_bad_in.phase_compiler = ptr::null_mut();
        assert!(PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(run_bad_in).cast()).is_null());
        let mut pc_del = PJRT_PhaseCompile_Destroy_Args {
            struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            handle: pc_get.compiler_out.cast(),
        };
        assert!(PJRT_PhaseCompile_DestroyCompiler(ptr::addr_of_mut!(pc_del).cast()).is_null());

        let mut valid_code = b"module @rel3 { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }".to_vec();
        let prog_f0 = PJRT_Program {
            struct_size: size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: valid_code.as_mut_ptr().cast(),
            code_size: valid_code.len(),
            format: c"mlir".as_ptr(),
            format_size: 0,
        };
        let dummy_o = [0x1a_u8, 0x00_u8];
        let mut ctr_f0 = PJRT_PhaseCompile_CompileToRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_CompileToRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: ptr::null(),
            client: ptr::null_mut(),
            program: ptr::addr_of!(prog_f0),
            compile_options: dummy_o.as_ptr().cast(),
            compile_options_size: 0,
            topology: ptr::null(),
            relocatables: ptr::null(),
            num_relocatables: 0,
            relocatables_buffer: ptr::null_mut(),
        };
        assert!(PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(ctr_f0).cast()).is_null());
        let rel3 = *ctr_f0.relocatables;
        let rel3_buf = ctr_f0.relocatables_buffer;

        let mut r_name_ok = PJRT_Relocatable_StringQuery_Args {
            struct_size: size_of::<PJRT_Relocatable_StringQuery_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel3,
            value_ptr: ptr::null(),
            value_size: 0,
        };
        assert!(PJRT_Relocatable_Name(ptr::addr_of_mut!(r_name_ok).cast()).is_null());
        assert!(r_name_ok.value_size > 0);

        ctr_f0.compile_options = corrupt_wire.as_ptr().cast();
        ctr_f0.compile_options_size = corrupt_wire.len();
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_CompileToRelocatable(ptr::addr_of_mut!(ctr_f0).cast()),
            3,
        );

        let mut drel_sz0 = PJRT_PhaseCompile_DeserializeRelocatable_Args {
            struct_size: size_of::<PJRT_PhaseCompile_DeserializeRelocatable_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: ptr::null(),
            serialized_relocatable: invalid_mlir.as_ptr().cast(),
            serialized_relocatable_size: 0,
            relocatable: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_DeserializeRelocatable(ptr::addr_of_mut!(drel_sz0).cast()),
            3,
        );

        let rel3_arr = [rel3];
        let mut lnk_sz0 = PJRT_PhaseCompile_LinkRelocatables_Args {
            struct_size: size_of::<PJRT_PhaseCompile_LinkRelocatables_Args>(),
            extension_start: ptr::null_mut(),
            phase_compiler: ptr::null(),
            relocatables: rel3_arr.as_ptr(),
            num_relocatables: 0,
            program: ptr::null(),
            compile_options: ptr::null(),
            compile_options_size: 0,
            topology: ptr::null(),
            executable: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_PhaseCompile_LinkRelocatables(ptr::addr_of_mut!(lnk_sz0).cast()),
            3,
        );

        let mut r3_del = PJRT_Relocatable_Destroy_Args {
            struct_size: size_of::<PJRT_Relocatable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            relocatable: rel3,
        };
        assert!(PJRT_Relocatable_Destroy(ptr::addr_of_mut!(r3_del).cast()).is_null());
        let mut b3_del = PJRT_PhaseCompile_Destroy_Args {
            struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            handle: rel3_buf,
        };
        assert!(PJRT_PhaseCompile_CBuffersDestroy(ptr::addr_of_mut!(b3_del).cast()).is_null());

        let mut tc_sz0 = PJRT_Triton_Compile_Args {
            struct_size: size_of::<PJRT_Triton_Compile_Args>(),
            extension_start: ptr::null_mut(),
            module: valid_ttir.as_ptr().cast(),
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
        assert_and_free_pjrt_error(PJRT_Triton_Compile(ptr::addr_of_mut!(tc_sz0).cast()), 3);

        for root_snip in [
            b"tt.func @k() { bad_op }".as_slice(),
            b"func.func @k() { bad_op }".as_slice(),
            b"stablehlo.custom { bad_op }".as_slice(),
            b"} module {".as_slice(),
        ] {
            tc_sz0.module = root_snip.as_ptr().cast();
            tc_sz0.module_size = root_snip.len();
            let err = PJRT_Triton_Compile(ptr::addr_of_mut!(tc_sz0).cast());
            if err.is_null() {
                PJRT_Triton_FreeAsm(tc_sz0.out_asm, tc_sz0.out_asm_size);
            } else {
                assert_and_free_pjrt_error(err, 13);
            }
        }
        PJRT_Triton_FreeAsm(c"x".as_ptr(), 0);

        let mut rh64_no_cb = PJRT_FFI_Register_Handler_Args {
            struct_size: 64,
            extension_start: ptr::null_mut(),
            target_name: ptr::null(),
            target_name_size: 0,
            handler: None,
            platform_name: ptr::null(),
            platform_name_size: 0,
            traits: 0,
            pad: 0,
        };
        assert!(PJRT_FFI_Register_Handler(ptr::addr_of_mut!(rh64_no_cb).cast()).is_null());
        rh64_no_cb.target_name = c"op2".as_ptr();
        rh64_no_cb.target_name_size = 3;
        assert!(PJRT_FFI_Register_Handler(ptr::addr_of_mut!(rh64_no_cb).cast()).is_null());

        // Executable SetTpuCompilationEnv non-null ptr with size == 0
        let mut set_env_sz0 = PJRT_TpuExecutable_SetTpuCompilationEnv_Args {
            struct_size: size_of::<PJRT_TpuExecutable_SetTpuCompilationEnv_Args>(),
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
            serialized_env: corrupt_wire.as_ptr(),
            serialized_env_size: 0,
        };
        assert!(
            PJRT_TpuExecutable_SetTpuCompilationEnv(ptr::addr_of_mut!(set_env_sz0).cast())
                .is_null()
        );

        // Memory: Topology_GetDefaultLayout + RawBuffer_GetHostPointer unregistered
        let mut tdl = PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
            struct_size: size_of::<PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args>(),
            extension_start: ptr::null_mut(),
            client: tc_v5p.topology.cast(),
            elem_type: 11,
            pad: 0,
            dims: ptr::null(),
            num_dims: 0,
            layout: ptr::null_mut(),
        };
        assert!(PJRT_Layouts_Topology_GetDefaultLayout(ptr::addr_of_mut!(tdl).cast()).is_null());
        let mut tdl_del = PJRT_Layouts_MemoryLayout_Destroy_Args {
            struct_size: size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            layout: tdl.layout,
        };
        assert!(PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(tdl_del).cast()).is_null());

        let mut rb_hp_bad = PJRT_RawBuffer_GetHostPointer_Args {
            struct_size: size_of::<PJRT_RawBuffer_GetHostPointer_Args>(),
            extension_start: ptr::null_mut(),
            raw_buffer: dangling_rb,
            host_pointer_out: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_RawBuffer_GetHostPointer(ptr::addr_of_mut!(rb_hp_bad).cast()),
            3,
        );

        // Topology matching (0, 0, 0) coordinate
        let zero_coords = [0_i32, 0, 0];
        let mut matched_coord = PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args {
            struct_size: size_of::<PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx_Args>(),
            extension_start: ptr::null_mut(),
            topology: tc_v5p.topology,
            coords: zero_coords.as_ptr(),
            core_idx: 0,
            device_id_out: 99,
        };
        assert!(
            PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(
                ptr::addr_of_mut!(matched_coord).cast()
            )
            .is_null()
        );
        assert_eq!(matched_coord.device_id_out, 0);

        // Megascale: Destroy dangling config, LogError when callback is None, UnregisterErrorHandler down to 0, DiscoverSlices dangling & live
        let mut ms_del_bad = PJRT_MultiSlice_Config_Destroy_Args {
            struct_size: size_of::<PJRT_MultiSlice_Config_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            config: dangling_ms,
        };
        assert_and_free_pjrt_error(
            PJRT_MultiSlice_Config_Destroy(ptr::addr_of_mut!(ms_del_bad).cast()),
            3,
        );

        let mut le_none = PJRT_Megascale_LogError_Args {
            struct_size: size_of::<PJRT_Megascale_LogError_Args>(),
            extension_start: ptr::null_mut(),
            handle: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_LogError(ptr::addr_of_mut!(le_none).cast()).is_null());

        for _ in 0..16 {
            let mut ueh = PJRT_Megascale_UnregisterErrorHandler_Args {
                struct_size: size_of::<PJRT_Megascale_UnregisterErrorHandler_Args>(),
                extension_start: ptr::null_mut(),
                handle: ptr::null_mut(),
            };
            assert!(PJRT_Megascale_UnregisterErrorHandler(ptr::addr_of_mut!(ueh).cast()).is_null());
        }

        let dangling_ctx =
            std::ptr::dangling_mut::<crate::pjrt_types::PJRT_Megascale_ClientContext>();
        let mut ds_bad = PJRT_Megascale_DiscoverSlices_Args {
            struct_size: size_of::<PJRT_Megascale_DiscoverSlices_Args>(),
            client_context: dangling_ctx,
            multi_slice_config: ptr::null_mut(),
        };
        assert_and_free_pjrt_error(
            PJRT_Megascale_DiscoverSlices(ptr::addr_of_mut!(ds_bad).cast()),
            3,
        );

        let mut cctx_cr = PJRT_Megascale_CreateDefaultClientContext_Args {
            struct_size: size_of::<PJRT_Megascale_CreateDefaultClientContext_Args>(),
            extension_start: ptr::null_mut(),
            flags: 0,
            context_out: ptr::null_mut(),
        };
        assert!(PJRT_Megascale_CreateClientContext(ptr::addr_of_mut!(cctx_cr).cast()).is_null());
        ds_bad.client_context = cctx_cr.context_out;
        assert!(PJRT_Megascale_DiscoverSlices(ptr::addr_of_mut!(ds_bad).cast()).is_null());
        if !ds_bad.multi_slice_config.is_null() {
            let mut msc_del = PJRT_MultiSlice_Config_Destroy_Args {
                struct_size: size_of::<PJRT_MultiSlice_Config_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                config: ds_bad.multi_slice_config,
            };
            assert!(PJRT_MultiSlice_Config_Destroy(ptr::addr_of_mut!(msc_del).cast()).is_null());
        }
        let mut cctx_del = PJRT_Megascale_DeleteClientContext_Args {
            struct_size: size_of::<PJRT_Megascale_DeleteClientContext_Args>(),
            extension_start: ptr::null_mut(),
            context: cctx_cr.context_out,
        };
        assert!(PJRT_Megascale_DeleteClientContext(ptr::addr_of_mut!(cctx_del).cast()).is_null());

        let mut ex_del2 = crate::pjrt_types::PJRT_Executable_Destroy_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Executable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: exec_ptr,
        };
        assert!(crate::PJRT_Executable_Destroy(ptr::addr_of_mut!(ex_del2).cast()).is_null());
        let mut lex_del = PJRT_LoadedExecutable_Destroy_Args {
            struct_size: size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: ccomp.executable,
        };
        assert!(PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(lex_del).cast()).is_null());
        let mut td_v5p = crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            topology: tc_v5p.topology,
        };
        assert!(
            crate::PJRT_TopologyDescription_Destroy(ptr::addr_of_mut!(td_v5p).cast()).is_null()
        );
        for b_p in [b_in, b_out, b_big] {
            let mut bd = PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: b_p,
            };
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(bd).cast()).is_null());
        }
        let mut cli_del = PJRT_Client_Destroy_Args {
            struct_size: size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client,
        };
        assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(cli_del).cast()).is_null());
    }
}
