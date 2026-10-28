//! Legacy TensorFlow TPU C API entrypoints.
//!
//! Exports `TfTpu_*`, `TpuTopology_*`, `TpuProfiler_*`, `TpuEmbeddingEngine_*`, and `TpuDriver_*`.

/// Low-level `TpuDriver_*` C API and 28-slot `TpuDriverFn` vtable entrypoints.
pub mod driver;
/// Legacy TensorFlow TPU embedding engine (`TpuEmbeddingEngine*`) C API entrypoints.
pub mod embedding;
/// Legacy TensorFlow TPU profiler (`TpuProfiler_*`) C API entrypoints.
pub mod profiler;
/// Legacy TensorFlow TPU runtime, status, and compilation C API entrypoints.
pub mod runtime;
/// Legacy TensorFlow TPU topology and mesh state C API entrypoints.
pub mod topology;

use super::*;

pub use driver::*;
pub use embedding::*;
pub use profiler::*;
pub use runtime::*;
pub use topology::*;

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

    #[test]
    fn test_legacy_tf_and_tpu_driver_api_lifecycle() {
        // SAFETY: Testing all legacy TensorFlow TPU and TpuDriver C API entrypoints.
        unsafe {
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
        }
    }
}
