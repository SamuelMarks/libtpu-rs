//! PJRT client, topology description, and execution context C API entrypoints (`PJRT_Client_*`, `PJRT_TopologyDescription_*`).

/// `PJRT_Client_*` C API entrypoints for buffer creation, aliases, async H2D, and DMA mapping.
pub mod buffer_creation;
/// `PJRT_Client_*` C API entrypoints for client lifecycle, device/memory queries, and compilation.
pub mod client;
/// `PJRT_ExecuteContext_*` C API entrypoints for execution context creation and destruction.
pub mod execute_context;
/// `PJRT_TopologyDescription_*` C API entrypoints for TPU topology inspection and serialization.
pub mod topology_description;

use super::*;

pub use buffer_creation::*;
pub use client::*;
pub use execute_context::*;
pub use topology_description::*;

#[cfg(test)]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::items_after_statements
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_buffer_api::{
        PJRT_AsyncHostToDeviceTransferManager_Destroy, PJRT_Buffer_Destroy,
    };
    use crate::pjrt_executable_api::{PJRT_Executable_Destroy, PJRT_LoadedExecutable_Destroy};
    use crate::pjrt_types::{
        PJRT_AsyncHostToDeviceTransferManager_Destroy_Args, PJRT_Buffer_Destroy_Args,
        PJRT_Executable_Destroy_Args, PJRT_LoadedExecutable_Destroy_Args, PJRT_Program,
    };
    use crate::rust_ergonomic_api::libtpu_rs_buffer_destroy;

    #[test]
    fn test_pjrt_client_topology_and_execute_context_api() {
        // SAFETY: Testing PJRT client, topology description, compile, and execute context entrypoints.
        unsafe {
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

            drop(Box::from_raw(PJRT_Client_LookupAddressableDevice(
                ptr::null_mut(),
            )));
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

            drop(Box::from_raw(PJRT_Client_DefaultDeviceAssignment(
                ptr::null_mut(),
            )));
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

            drop(Box::from_raw(PJRT_Client_Load(ptr::null_mut())));
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
            let mut del_loaded2 = PJRT_LoadedExecutable_Destroy_Args {
                struct_size: size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: load_args.loaded_executable,
            };
            assert!(PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(del_loaded2).cast()).is_null());

            drop(Box::from_raw(PJRT_Client_BufferFromHostBuffer(
                ptr::null_mut(),
            )));
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
            let mut aot_del = PJRT_Executable_Destroy_Args {
                struct_size: size_of::<PJRT_Executable_Destroy_Args>(),
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

            unsafe extern "C" fn test_view_deleter(arg: *mut std::ffi::c_void) {
                // SAFETY: `arg` points to a mutable `u32` counter.
                unsafe {
                    *arg.cast::<u32>() += 1;
                }
            }
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
            let mut h2d_mgr_del = PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
                struct_size: size_of::<PJRT_AsyncHostToDeviceTransferManager_Destroy_Args>(),
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

            for b_del in [buf_in, buf_in2] {
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
}
