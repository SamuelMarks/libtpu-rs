//! Pure-Rust, zero-panic Protobuf wire-format encoder, decoder, and schema reflection layer.
//!
//! Organized into domain-specific submodules synchronized with the canonical `OpenXLA`,
//! TensorFlow, and embedded `libtpu.so` `.proto` schemas in `docs/reverse_engineering/protos/`:
//! - [`descriptor`]: `DescriptorPool` reflection, dynamic transcoding, JSON serialization, SHA-256 fingerprinting, and [`ProtoSchemaMessage`].
//! - [`xla_data`]: `xla.ShapeProto`, `xla.LayoutProto`, `xla.TileProto`, `xla.ProgramShapeProto`, `xla.OpSharding`, `xla.DeviceAssignmentProto`.
//! - [`hlo`]: `xla.HloInstructionProto`, `xla.HloComputationProto`, `xla.HloModuleProto`, `xla.HloModuleConfigProto`, `xla.HloModuleProtoWithConfig`, `xla.BufferAssignmentProto`.
//! - [`pjrt`]: `xla.OptionOverrideProto`, `xla.ExecutableBuildOptionsProto`, `xla.CompileOptionsProto`, `xla.ExecuteOptionsProto`, `xla.CompiledMemoryStatsProto`, `xla.PjRtExecutableMetadata`, `xla.PjRtPartialProgramProto`, `xla.PjRtTopologyDescriptionProto`, `xla.PjRtRuntimeAbiVersionProto`, `xla.PjRtExecutableAbiVersionProto`, `xla.LocalTopologyProto`, `xla.GlobalTopologyProto`.
//! - [`tpu`]: `tpu.TpuTopologyArgsProto`, `tpu.TpuCompilationEnvironmentProto`, `tensorflow.tpu.TopologyProto`, `tensorflow.tpu.TPUEmbeddingConfiguration`, `tensorflow.tpu.SparseCoreTableLayouts`, `tensorflow.tpu.CompilationResultProto`, `tensorflow.tpu.TPUCompileMetadataProto`.
//! - [`deepsea`]: `xla.IsaProgramProto`, `xla.TargetArgumentsProto`, `xla.DeepseaExecutableProto`, `xla.TpuCoreProgramProto`, `tpu.HostCommandEnvelopeProto`, `tpu.TpuSerializedExecutableProto`, `xla.TpuExecutableProto`.
//! - [`megascale`]: `megascale.MultiSliceConfigProto`, `megascale.DCNTopologyProto`, `megascale.RuntimeMetricsProto`, `xla.megascale.runtime.EndpointAddresses`, `xla.megascale.runtime.MegascaleTransportConfigProto`, `xla.megascale.runtime.external.MegaScaleRuntimeError`.
//! - [`xplane`]: `tensorflow.profiler.XSpace`, `XPlane`, `XLine`, `XEvent`, `XEventMetadata`, `XStat`, `XStatMetadata`.

pub mod deepsea;
pub mod descriptor;
pub mod hlo;
pub mod megascale;
pub mod pjrt;
pub mod tpu;
pub mod xla_data;
pub mod xplane;

pub use deepsea::*;
pub use descriptor::*;
pub use hlo::*;
pub use megascale::*;
pub use pjrt::*;
pub use tpu::*;
pub use xla_data::*;
pub use xplane::*;

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_sha256_fingerprint_helpers() {
        let fp_hex = sha256_fingerprint_hex(b"tpu_executable_payload");
        assert_eq!(fp_hex.len(), 32);
        assert!(
            fp_hex
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );

        let fp_u64 = sha256_fingerprint_u64(b"tpu_executable_payload");
        assert_ne!(fp_u64, 0);
        assert_eq!(fp_u64, sha256_fingerprint_u64(b"tpu_executable_payload"));
        assert_ne!(fp_u64, sha256_fingerprint_u64(b"different_payload"));
    }

    #[test]
    fn test_global_descriptor_pool_and_dynamic_reflection() {
        let pool = global_descriptor_pool();
        assert!(
            pool.get_message_by_name("xla.TargetArgumentsProto")
                .is_some()
        );
        assert!(pool.get_message_by_name("xla.IsaProgramProto").is_some());
        assert!(
            pool.get_message_by_name("xla.DeepseaExecutableProto")
                .is_some()
        );
        assert!(
            pool.get_message_by_name("tpu.HostCommandEnvelopeProto")
                .is_some()
        );
        assert!(pool.get_message_by_name("xla.ShapeProto").is_some());
        assert!(pool.get_message_by_name("xla.HloModuleProto").is_some());
        assert!(
            pool.get_message_by_name("xla.CompileOptionsProto")
                .is_some()
        );
        assert!(
            pool.get_message_by_name("tensorflow.profiler.XSpace")
                .is_some()
        );

        let verified_count = verify_all_proto_descriptors().ok();
        assert_eq!(
            verified_count,
            Some(ALL_REGISTERED_PROTO_MESSAGE_NAMES.len())
        );

        assert!(decode_descriptor_pool(EMBEDDED_FILE_DESCRIPTOR_SET_BYTES).is_ok());
        assert!(try_global_descriptor_pool().is_ok());
        assert!(decode_descriptor_pool(&[0xFF, 0xFF, 0xFF]).is_err());

        let target = TargetArgumentsProto {
            tpu_version: 5,
            chip_variant: String::from("v5litepod"),
            clock_rate_mhz: 1050,
            hbm_capacity_bytes: 17_179_869_184,
            vmem_capacity_bytes: 16_777_216,
            smem_capacity_bytes: 1_048_576,
            cmem_capacity_bytes: 0,
            tensor_cores_per_chip: 1,
            sparse_cores_per_chip: 0,
            mxu_dimension: 128,
            ici_bandwidth_bps: 400_000_000_000,
        };
        let encoded = target.encode();
        let dyn_msg = decode_dynamic_proto_message("xla.TargetArgumentsProto", &encoded);
        assert!(dyn_msg.is_ok());

        let json = transcode_proto_bytes_to_json("xla.TargetArgumentsProto", &encoded).ok();
        assert!(json.as_deref().unwrap_or_default().contains("v5litepod"));

        let text = transcode_proto_bytes_to_text("xla.TargetArgumentsProto", &encoded).ok();
        assert!(text.as_deref().unwrap_or_default().contains("v5litepod"));
        if let Some(txt) = &text {
            assert_eq!(
                TargetArgumentsProto::from_proto_text(txt).ok(),
                Some(target.clone())
            );
        }
        assert_eq!(target.to_proto_text().ok(), text);

        assert!(lookup_message_descriptor("nonexistent.MessageProto").is_err());
        assert!(decode_dynamic_proto_message("nonexistent.MessageProto", &encoded).is_err());
        assert!(
            decode_dynamic_proto_message("xla.TargetArgumentsProto", &[0x08, 0xFF, 0xFF, 0xFF])
                .is_err()
        );
        assert!(decode_dynamic_proto_json("nonexistent.MessageProto", "{}").is_err());
        assert!(decode_dynamic_proto_json("xla.TargetArgumentsProto", "{invalid_json").is_err());
        assert!(decode_dynamic_proto_text("nonexistent.MessageProto", "").is_err());
        assert!(decode_dynamic_proto_text("xla.TargetArgumentsProto", ":::bad_textproto").is_err());

        let mut comp_env = TpuCompilationEnvironmentProto::default();
        comp_env.set_field_from_str("--xla_tpu_enable_megacore_fusion", "true");
        comp_env.set_field_from_str("2", "1");
        comp_env.set_field_from_str("xla_tpu_scoped_vmem_limit_kib", "32768");
        comp_env.set_field_from_str("custom_flags", "flag_a=10");
        comp_env.set_field_from_str("--xla_tpu_spmd_rng_bit_generator_unsafe", "true");
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_megacore_fusion"),
            "true"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_async_collective_fusion"),
            "true"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_scoped_vmem_limit_kib"),
            "32768"
        );
        assert!(comp_env.get_field_as_string("4").contains("flag_a=10"));
        assert_eq!(comp_env.get_field_as_string("flag_a"), "10");
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_spmd_rng_bit_generator_unsafe"),
            "true"
        );
        assert_eq!(comp_env.get_field_as_string("unknown_flag"), "default");
        let jellyfish_fields = TpuCompilationEnvironmentProto::list_jellyfish_schema_field_names();
        assert_ne!(jellyfish_fields, [] as [String; 0]);
        if let Some(first_jf) = jellyfish_fields.first() {
            assert_ne!(comp_env.get_field_as_string(first_jf), "default");
        }
        assert_ne!(comp_env.get_field_as_string("model_version"), "default");
        let parsed_cli = TpuCompilationEnvironmentProto::from_cli_flags(
            " --xla_tpu_enable_megacore_fusion=false, --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_flag_bool ",
        );
        assert!(!parsed_cli.xla_tpu_enable_megacore_fusion);
        assert_eq!(parsed_cli.xla_tpu_scoped_vmem_limit_kib, 65_536);
        assert_eq!(parsed_cli.get_field_as_string("xla_tpu_flag_bool"), "true");
        let mut init_cfg = crate::config::TpuInitConfig::default();
        parsed_cli.apply_to_init_config(&mut init_cfg);
        assert_eq!(
            init_cfg
                .extra_flags
                .get("xla_tpu_scoped_vmem_limit_kib")
                .map(String::as_str),
            Some("65536")
        );
        let mut comp_env2 = TpuCompilationEnvironmentProto::default();
        comp_env2.set_field_from_str("flag_only", "val");
        assert_eq!(comp_env2.get_field_as_string("flag_only"), "val");
    }

    #[test]
    fn test_proto_schema_message_trait_methods_and_mismatch_detection() {
        #[derive(Clone, PartialEq, Eq, prost::Message)]
        struct MismatchedTileProto {
            #[prost(int64, tag = "99")]
            pub unmapped_field: i64,
        }
        impl ProtoSchemaMessage for MismatchedTileProto {
            const PROTO_FULL_NAME: &'static str = "xla.TileProto";
        }

        let tile = TileProto {
            dimensions: vec![8, 128],
        };
        assert_eq!(TileProto::PROTO_FULL_NAME, "xla.TileProto");
        assert!(TileProto::descriptor().is_ok());
        let wire = tile.encode_proto();
        assert_eq!(TileProto::decode_proto(&wire).ok(), Some(tile.clone()));
        let dyn_msg = tile.to_dynamic_message();
        assert!(dyn_msg.is_ok());
        if let Ok(dm) = dyn_msg {
            assert_eq!(
                TileProto::from_dynamic_message(&dm).ok(),
                Some(tile.clone())
            );
        }
        let json = tile.to_proto_json();
        assert!(json.is_ok());
        if let Ok(j) = json {
            assert_eq!(TileProto::from_proto_json(&j).ok(), Some(tile.clone()));
        }
        assert!(tile.verify_schema_sync().is_ok());

        let bad = MismatchedTileProto { unmapped_field: 42 };
        assert!(bad.verify_schema_sync().is_err());
        let ok_default = MismatchedTileProto { unmapped_field: 0 };
        assert!(ok_default.verify_schema_sync().is_ok());
    }

    #[test]
    fn test_all_protos_roundtrip_and_schema_sync() {
        let tile = TileProto {
            dimensions: vec![8, 128],
        };
        let mut tile_bytes = tile.encode();
        assert_eq!(TileProto::decode(&tile_bytes).ok(), Some(tile.clone()));
        assert!(tile.verify_schema_sync().is_ok());
        tile_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(TileProto::decode(&tile_bytes).ok(), Some(tile.clone()));

        let layout = LayoutProto {
            minor_to_major: vec![1, 0],
            tiles: vec![tile],
            tail_padding_alignment_in_elements: 128,
            index_primitive_type: 4,
            pointer_primitive_type: 4,
            memory_space: 1,
            element_size_in_bits: 16,
            dynamic_shape_metadata_prefix_bytes: 64,
        };
        let mut layout_bytes = layout.encode();
        assert_eq!(
            LayoutProto::decode(&layout_bytes).ok(),
            Some(layout.clone())
        );
        assert!(layout.verify_schema_sync().is_ok());
        layout_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            LayoutProto::decode(&layout_bytes).ok(),
            Some(layout.clone())
        );

        let child_shape = ShapeProto {
            element_type: 11,
            dimensions: vec![4, 8],
            tuple_shapes: Vec::new(),
            layout: Some(layout.clone()),
            is_dynamic_dimension: vec![false, true],
        };
        let shape = ShapeProto {
            element_type: 13,
            dimensions: vec![16, 32],
            tuple_shapes: vec![child_shape],
            layout: Some(layout),
            is_dynamic_dimension: vec![true, false],
        };
        let mut shape_bytes = shape.encode();
        assert_eq!(ShapeProto::decode(&shape_bytes).ok(), Some(shape.clone()));
        assert!(shape.verify_schema_sync().is_ok());
        shape_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(ShapeProto::decode(&shape_bytes).ok(), Some(shape.clone()));

        let prog_shape = ProgramShapeProto {
            parameters: vec![shape.clone()],
            result: Some(shape.clone()),
            parameter_names: vec![String::from("p0"), String::from("p1")],
        };
        let mut prog_shape_bytes = prog_shape.encode();
        assert_eq!(
            ProgramShapeProto::decode(&prog_shape_bytes).ok(),
            Some(prog_shape.clone())
        );
        assert!(prog_shape.verify_schema_sync().is_ok());
        prog_shape_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            ProgramShapeProto::decode(&prog_shape_bytes).ok(),
            Some(prog_shape.clone())
        );

        let sharding = OpShardingProto {
            sharding_type: 3,
            tuple_shardings_shape: Some(shape.clone()),
            tile_assignment_dimensions: vec![2, 2],
            tile_assignment_devices: vec![0, 1, 2, 3],
            tuple_shardings: vec![OpShardingProto::default()],
            replicate_on_last_tile_dim: true,
            iota_reshape_dims: vec![4],
            iota_transpose_perm: vec![0],
        };
        let mut sharding_bytes = sharding.encode();
        assert_eq!(
            OpShardingProto::decode(&sharding_bytes).ok(),
            Some(sharding.clone())
        );
        assert!(sharding.verify_schema_sync().is_ok());
        sharding_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            OpShardingProto::decode(&sharding_bytes).ok(),
            Some(sharding)
        );

        let inst = HloInstructionProto {
            name: String::from("add.1"),
            opcode: String::from("add"),
            shape: Some(shape.clone()),
            id: 1,
            operand_ids: vec![0, 2],
            parameter_number: 3,
            custom_call_target: String::from("tpu_custom_call"),
            ..HloInstructionProto::default()
        };
        let mut inst_bytes = inst.encode();
        assert_eq!(
            HloInstructionProto::decode(&inst_bytes).ok(),
            Some(inst.clone())
        );
        assert!(inst.verify_schema_sync().is_ok());
        inst_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            HloInstructionProto::decode(&inst_bytes).ok(),
            Some(inst.clone())
        );

        let comp = HloComputationProto {
            name: String::from("main"),
            instructions: vec![inst],
            program_shape: Some(prog_shape.clone()),
            id: 10,
            root_id: 1,
            ..HloComputationProto::default()
        };
        let mut comp_bytes = comp.encode();
        assert_eq!(
            HloComputationProto::decode(&comp_bytes).ok(),
            Some(comp.clone())
        );
        assert!(comp.verify_schema_sync().is_ok());
        comp_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            HloComputationProto::decode(&comp_bytes).ok(),
            Some(comp.clone())
        );

        let module = HloModuleProto {
            name: String::from("test_mod"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 10,
            computations: vec![comp],
            host_program_shape: Some(prog_shape.clone()),
            id: 42,
            ..HloModuleProto::default()
        };
        let mut mod_bytes = module.encode();
        assert_eq!(
            HloModuleProto::decode(&mod_bytes).ok(),
            Some(module.clone())
        );
        assert!(module.verify_schema_sync().is_ok());
        mod_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(HloModuleProto::decode(&mod_bytes).ok(), Some(module));

        let mod_cfg = HloModuleConfigProto {
            entry_computation_layout: Some(prog_shape),
            seed: 999,
            replica_count: 2,
            num_partitions: 4,
            use_spmd_partitioning: true,
            ..HloModuleConfigProto::default()
        };
        let mut mod_cfg_bytes = mod_cfg.encode();
        assert_eq!(
            HloModuleConfigProto::decode(&mod_cfg_bytes).ok(),
            Some(mod_cfg.clone())
        );
        assert!(mod_cfg.verify_schema_sync().is_ok());
        mod_cfg_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            HloModuleConfigProto::decode(&mod_cfg_bytes).ok(),
            Some(mod_cfg)
        );

        let opt_override = OptionOverrideProto {
            string_field: String::from("true"),
            bool_field: true,
            int_field: -42,
            double_field: 1.5,
        };
        let mut opt_override_bytes = opt_override.encode();
        assert_eq!(
            OptionOverrideProto::decode(&opt_override_bytes).ok(),
            Some(opt_override.clone())
        );
        assert!(opt_override.verify_schema_sync().is_ok());
        opt_override_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            OptionOverrideProto::decode(&opt_override_bytes).ok(),
            Some(opt_override.clone())
        );

        let bopts = ExecutableBuildOptionsProto {
            device_ordinal: 1,
            result_layout: Some(shape.clone()),
            num_replicas: 2,
            num_partitions: 4,
            use_spmd_partitioning: true,
            use_auto_spmd_partitioning: true,
            device_assignment: Some(DeviceAssignmentProto {
                replica_count: 2,
                computation_count: 4,
                computation_devices: vec![ComputationDeviceProto {
                    replica_device_ids: vec![0, 1],
                }],
            }),
            allow_spmd_sharding_propagation_to_output: vec![true, false],
            allow_spmd_sharding_propagation_to_parameters: vec![false, true],
            process_index: 1,
            process_count: 4,
            use_shardy_partitioner: true,
        };
        let mut bopts_bytes = bopts.encode();
        assert_eq!(
            ExecutableBuildOptionsProto::decode(&bopts_bytes).ok(),
            Some(bopts.clone())
        );
        assert!(bopts.verify_schema_sync().is_ok());
        bopts_bytes.extend_from_slice(&[0xF8, 0x01, 0x01]);
        assert_eq!(
            ExecutableBuildOptionsProto::decode(&bopts_bytes).ok(),
            Some(bopts.clone())
        );

        let empty_copts = CompileOptionsProto::default();
        assert_eq!(
            CompileOptionsProto::decode(&empty_copts.encode()).ok(),
            Some(empty_copts)
        );

        let mut env_overrides = BTreeMap::new();
        env_overrides.insert(String::from("xla_tpu_flag"), opt_override);
        let copts = CompileOptionsProto {
            argument_layouts: vec![shape.clone()],
            parameter_is_tupled_arguments: true,
            executable_build_options: Some(bopts),
            compile_portable_executable: true,
            env_option_overrides: env_overrides,
            profile_version: 7,
            serialized_multi_slice_config: vec![1, 2, 3],
        };
        let mut copts_bytes = copts.encode();
        assert_eq!(
            CompileOptionsProto::decode(&copts_bytes).ok(),
            Some(copts.clone())
        );
        assert!(copts.verify_schema_sync().is_ok());
        copts_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(CompileOptionsProto::decode(&copts_bytes).ok(), Some(copts));

        assert_eq!(ExecutionModeProto::from(0), ExecutionModeProto::Unspecified);
        assert_eq!(ExecutionModeProto::from(1), ExecutionModeProto::Default);
        assert_eq!(ExecutionModeProto::from(2), ExecutionModeProto::Synchronous);
        assert_eq!(
            ExecutionModeProto::from(3),
            ExecutionModeProto::Asynchronous
        );
        let exec_opts = ExecuteOptionsProto {
            arguments_are_tupled: true,
            untuple_result: true,
            launch_id: 77,
            strict_shape_checking: true,
            execution_mode: i32::from(ExecutionModeProto::Asynchronous),
            non_donatable_input_indices: vec![0, 2],
            seed: 12345,
        };
        let mut exec_opts_bytes = exec_opts.encode();
        assert_eq!(
            ExecuteOptionsProto::decode(&exec_opts_bytes).ok(),
            Some(exec_opts.clone())
        );
        assert!(exec_opts.verify_schema_sync().is_ok());
        exec_opts_bytes.extend_from_slice(&[0x58, 0x01]);
        assert_eq!(
            ExecuteOptionsProto::decode(&exec_opts_bytes).ok(),
            Some(exec_opts)
        );

        let mem_stats = CompiledMemoryStatsProto {
            generated_code_size_in_bytes: 4096,
            argument_size_in_bytes: 1024,
            output_size_in_bytes: 2048,
            alias_size_in_bytes: 512,
            temp_size_in_bytes: 8192,
            host_generated_code_size_in_bytes: 256,
            host_argument_size_in_bytes: 128,
            host_output_size_in_bytes: 64,
            host_alias_size_in_bytes: 32,
            host_temp_size_in_bytes: 16,
            serialized_buffer_assignment: vec![0xDE, 0xAD],
            peak_memory_in_bytes: 16384,
            total_size_in_bytes: 15872,
            total_allocation_bytes: 32768,
            indefinite_allocations: 4096,
            peak_unpadded_heap_bytes: 12288,
        };
        let mut mem_stats_bytes = mem_stats.encode();
        assert_eq!(
            CompiledMemoryStatsProto::decode(&mem_stats_bytes).ok(),
            Some(mem_stats.clone())
        );
        assert!(mem_stats.verify_schema_sync().is_ok());
        mem_stats_bytes.extend_from_slice(&[0x98, 0x01, 0x01]);
        assert_eq!(
            CompiledMemoryStatsProto::decode(&mem_stats_bytes).ok(),
            Some(mem_stats.clone())
        );

        let exec_meta = PjRtExecutableMetadataProto {
            compiled_memory_stats: Some(mem_stats),
            platform_specific_metadata: vec![0xAA, 0xBB],
        };
        let mut exec_meta_bytes = exec_meta.encode();
        assert_eq!(
            PjRtExecutableMetadataProto::decode(&exec_meta_bytes).ok(),
            Some(exec_meta.clone())
        );
        assert!(exec_meta.verify_schema_sync().is_ok());
        exec_meta_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            PjRtExecutableMetadataProto::decode(&exec_meta_bytes).ok(),
            Some(exec_meta)
        );

        let partial_prog = PjRtPartialProgramProto {
            program: b"module @partial {}".to_vec(),
            program_format: String::from("mlir"),
            producer_phase: String::from("hlo_optimization"),
            consumer_phases: vec![String::from("tpu_llo_lowering")],
            version: String::from("0.0.50"),
            program_name: String::from("partial_mod"),
        };
        let mut partial_prog_bytes = partial_prog.encode();
        assert_eq!(
            PjRtPartialProgramProto::decode(&partial_prog_bytes).ok(),
            Some(partial_prog.clone())
        );
        assert!(partial_prog.verify_schema_sync().is_ok());
        partial_prog_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            PjRtPartialProgramProto::decode(&partial_prog_bytes).ok(),
            Some(partial_prog)
        );

        let topo = TpuTopologyArgsProto {
            version: 6,
            num_hosts: 1,
            chips_per_host: 4,
            mesh_bounds: vec![2, 2, 1],
            platform_version: String::from("libtpu-rs 0.0.50"),
            fingerprint: sha256_fingerprint_hex(b"topo"),
        };
        let mut topo_bytes = topo.encode();
        assert_eq!(
            TpuTopologyArgsProto::decode(&topo_bytes).ok(),
            Some(topo.clone())
        );
        assert!(topo.verify_schema_sync().is_ok());
        topo_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            TpuTopologyArgsProto::decode(&topo_bytes).ok(),
            Some(topo.clone())
        );

        let pjrt_topo_desc = PjRtTopologyDescriptionProto {
            platform_id: 0x5450_5500,
            platform_name: String::from("tpu"),
            platform_version: String::from("0.0.50"),
            is_subslice_topology: true,
            platform_specific_topology: topo.encode(),
        };
        let mut pjrt_topo_bytes = pjrt_topo_desc.encode();
        assert_eq!(
            PjRtTopologyDescriptionProto::decode(&pjrt_topo_bytes).ok(),
            Some(pjrt_topo_desc.clone())
        );
        assert!(pjrt_topo_desc.verify_schema_sync().is_ok());
        pjrt_topo_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            PjRtTopologyDescriptionProto::decode(&pjrt_topo_bytes).ok(),
            Some(pjrt_topo_desc)
        );

        let hw_feat = TpuHardwareFeatureProto {
            embedding_feature: 2,
            num_embedding_devices_per_chip: 2,
        };
        let mut hw_feat_bytes = hw_feat.encode();
        assert_eq!(
            TpuHardwareFeatureProto::decode(&hw_feat_bytes).ok(),
            Some(hw_feat.clone())
        );
        assert!(hw_feat.verify_schema_sync().is_ok());
        hw_feat_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            TpuHardwareFeatureProto::decode(&hw_feat_bytes).ok(),
            Some(hw_feat.clone())
        );

        let tf_topo = TopologyProto {
            mesh_shape: vec![2, 2, 1, 1],
            num_tasks: 1,
            num_tpu_devices_per_task: 4,
            device_coordinates: vec![0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            tpu_hardware_feature: Some(hw_feat),
        };
        let mut tf_topo_bytes = tf_topo.encode();
        assert_eq!(
            TopologyProto::decode(&tf_topo_bytes).ok(),
            Some(tf_topo.clone())
        );
        assert!(tf_topo.verify_schema_sync().is_ok());
        tf_topo_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(TopologyProto::decode(&tf_topo_bytes).ok(), Some(tf_topo));

        let exec = TpuSerializedExecutableProto {
            name: String::from("exec_0"),
            fingerprint: sha256_fingerprint_hex(b"exec_0"),
            num_replicas: 1,
            num_partitions: 2,
            instruction_opcodes: vec![1, 2, 11],
            output_shapes: vec![shape],
            compile_options_bytes: vec![0x08, 0x01],
            hlo_module_bytes: b"HloModule exec_0".to_vec(),
        };
        let mut exec_bytes = exec.encode();
        assert_eq!(
            TpuSerializedExecutableProto::decode(&exec_bytes).ok(),
            Some(exec.clone())
        );
        assert!(exec.verify_schema_sync().is_ok());
        exec_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            TpuSerializedExecutableProto::decode(&exec_bytes).ok(),
            Some(exec)
        );

        let stat_meta = XStatMetadataProto {
            id: 1,
            name: String::from("flops"),
            description: String::from("Peak FLOPs"),
        };
        let mut stat_meta_bytes = stat_meta.encode();
        assert_eq!(
            XStatMetadataProto::decode(&stat_meta_bytes).ok(),
            Some(stat_meta.clone())
        );
        assert!(stat_meta.verify_schema_sync().is_ok());
        stat_meta_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            XStatMetadataProto::decode(&stat_meta_bytes).ok(),
            Some(stat_meta.clone())
        );

        let ev_meta = XEventMetadataProto {
            id: 2,
            name: String::from("matmul"),
            display_name: String::from("MatMul_128x128"),
            metadata: String::from("hlo_op=dot.1"),
        };
        let mut ev_meta_bytes = ev_meta.encode();
        assert_eq!(
            XEventMetadataProto::decode(&ev_meta_bytes).ok(),
            Some(ev_meta.clone())
        );
        assert!(ev_meta.verify_schema_sync().is_ok());
        ev_meta_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            XEventMetadataProto::decode(&ev_meta_bytes).ok(),
            Some(ev_meta.clone())
        );

        let stat = XStatProto {
            metadata_id: 1,
            double_value: 99.5,
            uint64_value: 1000,
            int64_value: -50,
            str_value: String::from("ok"),
            bytes_value: vec![0xAA],
            ref_value: 7,
        };
        let mut stat_bytes = stat.encode();
        assert_eq!(XStatProto::decode(&stat_bytes).ok(), Some(stat.clone()));
        assert!(stat.verify_schema_sync().is_ok());
        stat_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(XStatProto::decode(&stat_bytes).ok(), Some(stat.clone()));

        let ev = XEventProto {
            metadata_id: 2,
            offset_ps: 100,
            duration_ps: 500,
            num_occurrences: 3,
            stats: vec![stat.clone()],
        };
        let mut ev_bytes = ev.encode();
        assert_eq!(XEventProto::decode(&ev_bytes).ok(), Some(ev.clone()));
        assert!(ev.verify_schema_sync().is_ok());
        ev_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(XEventProto::decode(&ev_bytes).ok(), Some(ev.clone()));

        let line = XLineProto {
            id: 1,
            display_id: 10,
            name: String::from("TensorCore 0"),
            display_name: String::from("TC0"),
            timestamp_ns: 1_000_000,
            duration_ps: 50_000,
            events: vec![ev],
        };
        let mut line_bytes = line.encode();
        assert_eq!(XLineProto::decode(&line_bytes).ok(), Some(line.clone()));
        assert!(line.verify_schema_sync().is_ok());
        line_bytes.extend_from_slice(&[0x78, 0x01]);
        assert_eq!(XLineProto::decode(&line_bytes).ok(), Some(line.clone()));

        let mut ev_map = BTreeMap::new();
        ev_map.insert(2, ev_meta);
        let mut st_map = BTreeMap::new();
        st_map.insert(1, stat_meta);
        let plane = XPlaneProto {
            id: 1,
            name: String::from("/device:TPU:0"),
            lines: vec![line],
            event_metadata: ev_map,
            stat_metadata: st_map,
            stats: vec![stat],
        };
        let mut plane_bytes = plane.encode();
        assert_eq!(XPlaneProto::decode(&plane_bytes).ok(), Some(plane.clone()));
        assert!(plane.verify_schema_sync().is_ok());
        plane_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(XPlaneProto::decode(&plane_bytes).ok(), Some(plane.clone()));

        let xspace = XSpaceProto {
            planes: vec![plane],
            errors: vec![String::from("err0")],
            warnings: vec![String::from("warn0")],
            hostnames: vec![String::from("localhost")],
        };
        let mut xspace_bytes = xspace.encode();
        assert_eq!(
            XSpaceProto::decode(&xspace_bytes).ok(),
            Some(xspace.clone())
        );
        assert!(xspace.verify_schema_sync().is_ok());
        xspace_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(XSpaceProto::decode(&xspace_bytes).ok(), Some(xspace));

        let tbl = EmbeddingTableDescriptorProto {
            name: String::from("user_vocab"),
            vocabulary_size: 100_000,
            dimension: 128,
            num_features: 4,
        };
        let mut tbl_bytes = tbl.encode();
        assert_eq!(
            EmbeddingTableDescriptorProto::decode(&tbl_bytes).ok(),
            Some(tbl.clone())
        );
        assert!(tbl.verify_schema_sync().is_ok());
        tbl_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            EmbeddingTableDescriptorProto::decode(&tbl_bytes).ok(),
            Some(tbl.clone())
        );

        let feat = EmbeddingFeatureDescriptorProto {
            name: String::from("item_id"),
            table_id: 1,
            input_shape: vec![64, 16],
        };
        let mut feat_bytes = feat.encode();
        assert_eq!(
            EmbeddingFeatureDescriptorProto::decode(&feat_bytes).ok(),
            Some(feat.clone())
        );
        assert!(feat.verify_schema_sync().is_ok());
        feat_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            EmbeddingFeatureDescriptorProto::decode(&feat_bytes).ok(),
            Some(feat.clone())
        );

        let spmd_shard = EmbeddingSpmdShardingProto {
            enabled: true,
            num_cores_per_replica: 4,
        };
        let mut spmd_shard_bytes = spmd_shard.encode();
        assert_eq!(
            EmbeddingSpmdShardingProto::decode(&spmd_shard_bytes).ok(),
            Some(spmd_shard.clone())
        );
        assert!(spmd_shard.verify_schema_sync().is_ok());
        spmd_shard_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            EmbeddingSpmdShardingProto::decode(&spmd_shard_bytes).ok(),
            Some(spmd_shard.clone())
        );

        let emb_cfg = TpuEmbeddingConfigurationProto {
            table_descriptor: vec![tbl],
            mode: 2,
            batch_size_per_tensor_core: 64,
            num_hosts: 2,
            num_tensor_cores: 8,
            sharding_strategy: 1,
            pipeline_execution_with_tensor_core: true,
            profile_data_directory: String::from("/tmp/emb_prof"),
            feature_descriptor: vec![feat],
            spmd_sharding: Some(spmd_shard),
        };
        let mut emb_cfg_bytes = emb_cfg.encode();
        assert_eq!(
            TpuEmbeddingConfigurationProto::decode(&emb_cfg_bytes).ok(),
            Some(emb_cfg.clone())
        );
        assert!(emb_cfg.verify_schema_sync().is_ok());
        emb_cfg_bytes.extend_from_slice(&[0x70, 0x01]);
        assert_eq!(
            TpuEmbeddingConfigurationProto::decode(&emb_cfg_bytes).ok(),
            Some(emb_cfg)
        );

        let sc_layout = SparseCoreTableLayoutProto {
            table_name: String::from("vocab_0"),
            unsharded_shape: vec![100_000, 128],
            unsharded_padded_shape: vec![100_096, 128],
            num_sparse_cores: 4,
            num_partitions: 2,
            total_rows_per_sparse_core_shard: 25_024,
        };
        let mut sc_layout_bytes = sc_layout.encode();
        assert_eq!(
            SparseCoreTableLayoutProto::decode(&sc_layout_bytes).ok(),
            Some(sc_layout.clone())
        );
        assert!(sc_layout.verify_schema_sync().is_ok());
        sc_layout_bytes.extend_from_slice(&[0x70, 0x01]);
        assert_eq!(
            SparseCoreTableLayoutProto::decode(&sc_layout_bytes).ok(),
            Some(sc_layout.clone())
        );

        let sc_multi = SparseCoreTableLayoutsProto {
            tables: vec![sc_layout],
        };
        let mut sc_multi_bytes = sc_multi.encode();
        assert_eq!(
            SparseCoreTableLayoutsProto::decode(&sc_multi_bytes).ok(),
            Some(sc_multi.clone())
        );
        assert!(sc_multi.verify_schema_sync().is_ok());
        sc_multi_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            SparseCoreTableLayoutsProto::decode(&sc_multi_bytes).ok(),
            Some(sc_multi)
        );

        let cd = ComputationDeviceProto {
            replica_device_ids: vec![0, 1, 2, 3],
        };
        let mut cd_bytes = cd.encode();
        assert_eq!(
            ComputationDeviceProto::decode(&cd_bytes).ok(),
            Some(cd.clone())
        );
        assert!(cd.verify_schema_sync().is_ok());
        cd_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            ComputationDeviceProto::decode(&cd_bytes).ok(),
            Some(cd.clone())
        );

        let da = DeviceAssignmentProto {
            replica_count: 4,
            computation_count: 1,
            computation_devices: vec![cd],
        };
        let mut da_bytes = da.encode();
        assert_eq!(
            DeviceAssignmentProto::decode(&da_bytes).ok(),
            Some(da.clone())
        );
        assert!(da.verify_schema_sync().is_ok());
        da_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            DeviceAssignmentProto::decode(&da_bytes).ok(),
            Some(da.clone())
        );

        let compile_meta = TpuCompileMetadataProto {
            num_replicas: 4,
            num_cores_per_replica: 1,
            device_assignment: Some(da),
            use_spmd_for_xla_partitioning: true,
        };
        let mut compile_meta_bytes = compile_meta.encode();
        assert_eq!(
            TpuCompileMetadataProto::decode(&compile_meta_bytes).ok(),
            Some(compile_meta.clone())
        );
        assert!(compile_meta.verify_schema_sync().is_ok());
        compile_meta_bytes.extend_from_slice(&[0x98, 0x01, 0x01]);
        assert_eq!(
            TpuCompileMetadataProto::decode(&compile_meta_bytes).ok(),
            Some(compile_meta)
        );

        let comp_env = TpuCompilationEnvironmentProto {
            xla_tpu_enable_megacore_fusion: true,
            xla_tpu_enable_async_collective_fusion: true,
            xla_tpu_scoped_vmem_limit_kib: 16384,
            custom_flags: String::from("--xla_tpu_flag=1"),
        };
        let mut comp_env_bytes = comp_env.encode();
        assert_eq!(
            TpuCompilationEnvironmentProto::decode(&comp_env_bytes).ok(),
            Some(comp_env.clone())
        );
        assert!(comp_env.verify_schema_sync().is_ok());
        comp_env_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            TpuCompilationEnvironmentProto::decode(&comp_env_bytes).ok(),
            Some(comp_env)
        );

        let hlo_with_cfg = HloModuleProtoWithConfig {
            hlo_module: Some(HloModuleProto::decode(&mod_bytes).unwrap_or_default()),
            config: Some(HloModuleConfigProto::decode(&mod_cfg_bytes).unwrap_or_default()),
        };
        let mut hlo_with_cfg_bytes = hlo_with_cfg.encode();
        assert_eq!(
            HloModuleProtoWithConfig::decode(&hlo_with_cfg_bytes).ok(),
            Some(hlo_with_cfg.clone())
        );
        assert!(hlo_with_cfg.verify_schema_sync().is_ok());
        hlo_with_cfg_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            HloModuleProtoWithConfig::decode(&hlo_with_cfg_bytes).ok(),
            Some(hlo_with_cfg.clone())
        );

        let comp_res = CompilationResultProto {
            status_code: 0,
            status_error_message: String::from("OK"),
            hlo_protos: vec![hlo_with_cfg],
        };
        let mut comp_res_bytes = comp_res.encode();
        assert_eq!(
            CompilationResultProto::decode(&comp_res_bytes).ok(),
            Some(comp_res.clone())
        );
        assert!(comp_res.verify_schema_sync().is_ok());
        comp_res_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            CompilationResultProto::decode(&comp_res_bytes).ok(),
            Some(comp_res)
        );

        let r_abi = RuntimeAbiVersionProto {
            platform_id: 0x5450_5500,
            version_string: String::from("0.0.50"),
            pjrt_api_minor: 56,
        };
        let mut r_abi_bytes = r_abi.encode();
        assert_eq!(
            RuntimeAbiVersionProto::decode(&r_abi_bytes).ok(),
            Some(r_abi.clone())
        );
        assert!(r_abi.verify_schema_sync().is_ok());
        r_abi_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            RuntimeAbiVersionProto::decode(&r_abi_bytes).ok(),
            Some(r_abi)
        );

        let e_abi = ExecutableAbiVersionProto {
            platform_id: 0x5450_5500,
            compiler_version: String::from("libtpu-rs-xla-0.0.50"),
            target_generation: 5,
        };
        let mut e_abi_bytes = e_abi.encode();
        assert_eq!(
            ExecutableAbiVersionProto::decode(&e_abi_bytes).ok(),
            Some(e_abi.clone())
        );
        assert!(e_abi.verify_schema_sync().is_ok());
        e_abi_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            ExecutableAbiVersionProto::decode(&e_abi_bytes).ok(),
            Some(e_abi)
        );

        let part_cfg = TpuEmbeddingPartitionedConfigProto {
            num_partitions: 4,
            table_names: vec![String::from("vocab_0"), String::from("vocab_1")],
            raw_config_bytes: vec![0x08, 0x02],
        };
        let mut part_cfg_bytes = part_cfg.encode();
        assert_eq!(
            TpuEmbeddingPartitionedConfigProto::decode(&part_cfg_bytes).ok(),
            Some(part_cfg.clone())
        );
        assert!(part_cfg.verify_schema_sync().is_ok());
        part_cfg_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            TpuEmbeddingPartitionedConfigProto::decode(&part_cfg_bytes).ok(),
            Some(part_cfg)
        );

        let mem_cfg = TpuEmbeddingMemoryConfigProto {
            hbm_buffers_bytes: 67_108_864,
            num_segments: 2,
            segment_blobs: vec![vec![0x01, 0x02], vec![0x03, 0x04]],
        };
        let mut mem_cfg_bytes = mem_cfg.encode();
        assert_eq!(
            TpuEmbeddingMemoryConfigProto::decode(&mem_cfg_bytes).ok(),
            Some(mem_cfg.clone())
        );
        assert!(mem_cfg.verify_schema_sync().is_ok());
        mem_cfg_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            TpuEmbeddingMemoryConfigProto::decode(&mem_cfg_bytes).ok(),
            Some(mem_cfg)
        );

        let host_net = TpuEmbeddingHostNetworkConfigProto {
            host_id: 1,
            endpoint_address: String::from("127.0.0.1:8471"),
            partition_payload: vec![0xAA, 0xBB],
        };
        let mut host_net_bytes = host_net.encode();
        assert_eq!(
            TpuEmbeddingHostNetworkConfigProto::decode(&host_net_bytes).ok(),
            Some(host_net.clone())
        );
        assert!(host_net.verify_schema_sync().is_ok());
        host_net_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            TpuEmbeddingHostNetworkConfigProto::decode(&host_net_bytes).ok(),
            Some(host_net)
        );

        let ms_cfg = MegascaleMultiSliceConfigProto {
            num_slices: 4,
            slice_id: 1,
            num_devices_per_slice: 8,
            coordinator_address: String::from("10.0.0.1:8888"),
        };
        let mut ms_cfg_bytes = ms_cfg.encode();
        assert_eq!(
            MegascaleMultiSliceConfigProto::decode(&ms_cfg_bytes).ok(),
            Some(ms_cfg.clone())
        );
        assert!(ms_cfg.verify_schema_sync().is_ok());
        ms_cfg_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            MegascaleMultiSliceConfigProto::decode(&ms_cfg_bytes).ok(),
            Some(ms_cfg)
        );

        let dcn_topo = MegascaleDcnTopologyProto {
            num_slices: 4,
            num_hosts_per_slice: 2,
            transport_type: 2,
        };
        let mut dcn_bytes = dcn_topo.encode();
        assert_eq!(
            MegascaleDcnTopologyProto::decode(&dcn_bytes).ok(),
            Some(dcn_topo.clone())
        );
        assert!(dcn_topo.verify_schema_sync().is_ok());
        dcn_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            MegascaleDcnTopologyProto::decode(&dcn_bytes).ok(),
            Some(dcn_topo)
        );

        let ms_metrics = MegascaleRuntimeMetricsProto {
            completed_barriers: 42,
            collective_ops_count: 128,
            active_contexts: 2,
        };
        let mut ms_metrics_bytes = ms_metrics.encode();
        assert_eq!(
            MegascaleRuntimeMetricsProto::decode(&ms_metrics_bytes).ok(),
            Some(ms_metrics.clone())
        );
        assert!(ms_metrics.verify_schema_sync().is_ok());
        ms_metrics_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
        assert_eq!(
            MegascaleRuntimeMetricsProto::decode(&ms_metrics_bytes).ok(),
            Some(ms_metrics)
        );

        let host_addr = HostNetworkAddressProto {
            address: String::from("10.0.0.2:9000"),
            interface_name: String::from("eth0"),
            host_name_for_debugging: String::from("tpu-worker-0"),
            numa_node: 0,
        };
        assert_eq!(
            HostNetworkAddressProto::decode(&host_addr.encode()).ok(),
            Some(host_addr.clone())
        );
        assert!(host_addr.verify_schema_sync().is_ok());

        let net_mapping = NetworkAddressMappingProto {
            slice_id: 0,
            host_id: 1,
            addresses: vec![host_addr],
            incarnation_id: Some(12345),
        };
        assert_eq!(
            NetworkAddressMappingProto::decode(&net_mapping.encode()).ok(),
            Some(net_mapping.clone())
        );
        assert!(net_mapping.verify_schema_sync().is_ok());

        let endpoints = EndpointAddressesProto {
            address_mappings: vec![net_mapping],
        };
        assert_eq!(
            EndpointAddressesProto::decode(&endpoints.encode()).ok(),
            Some(endpoints.clone())
        );
        assert!(endpoints.verify_schema_sync().is_ok());

        let transport_cfg = MegascaleTransportConfigProto {
            num_channels: 8,
            max_inflight_bytes: 67_108_864,
            timeout_ms: 30_000,
            enable_zstd_compression: true,
            zstd_compression_level: 3,
        };
        assert_eq!(
            MegascaleTransportConfigProto::decode(&transport_cfg.encode()).ok(),
            Some(transport_cfg.clone())
        );
        assert!(transport_cfg.verify_schema_sync().is_ok());

        let isa_slot = IsaBundleSlotProto {
            unit_kind: 2,
            opcode: 3,
            dst_reg: 0,
            src0_reg: 1,
            src1_reg: 2,
            predicate_reg: 0,
            immediate: 128,
        };
        assert_eq!(
            IsaBundleSlotProto::decode(&isa_slot.encode()).ok(),
            Some(isa_slot.clone())
        );
        assert!(isa_slot.verify_schema_sync().is_ok());

        let isa_bundle = IsaVliwBundleProto {
            pc_index: 0,
            scalar_s0: Some(isa_slot.clone()),
            scalar_s1: Some(isa_slot.clone()),
            vector_v0: Some(isa_slot.clone()),
            vector_v1: Some(isa_slot.clone()),
            matrix_m0: Some(isa_slot.clone()),
            matrix_m1: Some(isa_slot.clone()),
            unary_xlu: Some(isa_slot.clone()),
            dma_d0: Some(isa_slot),
            llo_annotation: String::from("llo.dot"),
        };
        assert_eq!(
            IsaVliwBundleProto::decode(&isa_bundle.encode()).ok(),
            Some(isa_bundle.clone())
        );
        assert!(isa_bundle.verify_schema_sync().is_ok());

        let isa_core = IsaCoreProgramHeaderProto {
            core_index: 0,
            core_type: 0,
            entry_pc: 0,
            bundle_count: 1,
            raw_vliw_stream: vec![0xAA; 64],
            structured_bundles: vec![isa_bundle],
        };
        assert_eq!(
            IsaCoreProgramHeaderProto::decode(&isa_core.encode()).ok(),
            Some(isa_core.clone())
        );
        assert!(isa_core.verify_schema_sync().is_ok());

        let isa_sym = IsaSymbolTableEntryProto {
            name: String::from("entry_0"),
            pc_offset: 0,
            size_bytes: 64,
        };
        assert_eq!(
            IsaSymbolTableEntryProto::decode(&isa_sym.encode()).ok(),
            Some(isa_sym.clone())
        );
        assert!(isa_sym.verify_schema_sync().is_ok());

        let isa_prog = IsaProgramProto {
            isa_architecture: String::from("vfc"),
            isa_abi_version: 163,
            core_programs: vec![isa_core],
            symbols: vec![isa_sym],
            fingerprint: 123_456_789,
        };
        assert_eq!(
            IsaProgramProto::decode(&isa_prog.encode()).ok(),
            Some(isa_prog.clone())
        );
        assert!(isa_prog.verify_schema_sync().is_ok());

        let target_args = TargetArgumentsProto {
            tpu_version: 5,
            chip_variant: String::from("v5litepod"),
            clock_rate_mhz: 1050,
            hbm_capacity_bytes: 17_179_869_184,
            vmem_capacity_bytes: 16_777_216,
            smem_capacity_bytes: 1_048_576,
            cmem_capacity_bytes: 0,
            tensor_cores_per_chip: 1,
            sparse_cores_per_chip: 0,
            mxu_dimension: 128,
            ici_bandwidth_bps: 400_000_000_000,
        };
        assert_eq!(
            TargetArgumentsProto::decode(&target_args.encode()).ok(),
            Some(target_args.clone())
        );
        assert!(target_args.verify_schema_sync().is_ok());

        let ds_overlay = DeepseaOverlaySectionProto {
            overlay_id: 0,
            symbol_name: String::from(".imem.main"),
            imem_offset_bytes: 0,
            byte_size: 64,
            encoded_bundles: vec![0x11; 64],
            core_type: 0,
        };
        assert_eq!(
            DeepseaOverlaySectionProto::decode(&ds_overlay.encode()).ok(),
            Some(ds_overlay.clone())
        );
        assert!(ds_overlay.verify_schema_sync().is_ok());

        let ds_reloc = DeepseaRelocationEntryProto {
            bundle_offset_bytes: 0,
            slot_index: 1,
            buffer_index: 2,
            addend_bytes: 4096,
            memory_space: 0,
        };
        assert_eq!(
            DeepseaRelocationEntryProto::decode(&ds_reloc.encode()).ok(),
            Some(ds_reloc.clone())
        );
        assert!(ds_reloc.verify_schema_sync().is_ok());

        let ds_const = DeepseaConstantPoolEntryProto {
            constant_id: 0,
            memory_space: 1,
            offset_bytes: 256,
            payload: vec![0x00, 0x00, 0x80, 0x3F],
        };
        assert_eq!(
            DeepseaConstantPoolEntryProto::decode(&ds_const.encode()).ok(),
            Some(ds_const.clone())
        );
        assert!(ds_const.verify_schema_sync().is_ok());

        let ds_sflag = DeepseaSyncFlagInitProto {
            sync_flag_id: 3,
            initial_value: 1,
            debug_label: String::from("dma_done"),
        };
        assert_eq!(
            DeepseaSyncFlagInitProto::decode(&ds_sflag.encode()).ok(),
            Some(ds_sflag.clone())
        );
        assert!(ds_sflag.verify_schema_sync().is_ok());

        let core_abi = TpuCoreProgramAbiProto {
            version: 5,
            variant: String::from("default"),
            chip_config: String::from("default"),
            base_abi_version: 163,
            hardware_abi_version: 65_539,
            required_compiler_hash: String::from("abc123"),
        };
        assert_eq!(
            TpuCoreProgramAbiProto::decode(&core_abi.encode()).ok(),
            Some(core_abi.clone())
        );
        assert!(core_abi.verify_schema_sync().is_ok());

        let seg_range = TpuMemorySegmentRangeProto {
            offset: 0,
            size: 1024,
        };
        assert_eq!(
            TpuMemorySegmentRangeProto::decode(&seg_range.encode()).ok(),
            Some(seg_range.clone())
        );
        assert!(seg_range.verify_schema_sync().is_ok());

        let seg_proto = TpuMemorySegmentProto {
            segment_type: 1,
            memory_type: 1,
            initialized_data_range: Some(seg_range),
            segment_size_in_bytes: 1024,
        };
        assert_eq!(
            TpuMemorySegmentProto::decode(&seg_proto.encode()).ok(),
            Some(seg_proto.clone())
        );
        assert!(seg_proto.verify_schema_sync().is_ok());

        let seg_set = TpuMemorySegmentSetProto {
            memory_type: 1,
            segments: vec![seg_proto],
            initialized_data: vec![0x42; 64],
            hash: vec![0x11; 32],
            compression: 0,
        };
        assert_eq!(
            TpuMemorySegmentSetProto::decode(&seg_set.encode()).ok(),
            Some(seg_set.clone())
        );
        assert!(seg_set.verify_schema_sync().is_ok());

        let core_prog = TpuCoreProgramProto {
            platform_type: 1,
            fingerprint: b"fp_0".to_vec(),
            compilation_id: 99,
            memory_segments: vec![seg_set],
            concurrent_program_hbm_stack_size: 4096,
            has_concurrent_program: false,
            abi: Some(core_abi),
            oom_during_compilation: false,
        };
        assert_eq!(
            TpuCoreProgramProto::decode(&core_prog.encode()).ok(),
            Some(core_prog.clone())
        );
        assert!(core_prog.verify_schema_sync().is_ok());

        let ds_exec = DeepseaExecutableProto {
            module_name: String::from("ds_mod"),
            isa_program: Some(isa_prog),
            overlays: vec![ds_overlay],
            relocations: vec![ds_reloc],
            constants: vec![ds_const],
            sync_flags: vec![ds_sflag],
            target_arguments: Some(target_args.clone()),
            vmem_allocation_bytes: 65536,
            smem_allocation_bytes: 4096,
            cmem_allocation_bytes: 0,
            hbm_scratchpad_bytes: 1_048_576,
            core_program: Some(core_prog.clone()),
            sparse_core_program: Some(core_prog),
        };
        assert_eq!(
            DeepseaExecutableProto::decode(&ds_exec.encode()).ok(),
            Some(ds_exec.clone())
        );
        assert!(ds_exec.verify_schema_sync().is_ok());

        let host_cmd = HostCommandEnvelopeProto {
            opcode: 16,
            continuation_id: 42,
            device_id: 0,
            core_index: 0,
            handler_id: 1001,
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        assert_eq!(
            HostCommandEnvelopeProto::decode(&host_cmd.encode()).ok(),
            Some(host_cmd.clone())
        );
        assert!(host_cmd.verify_schema_sync().is_ok());

        let tpu_exec = TpuExecutableProto {
            ds_executable: Some(ds_exec),
            hlo_module_with_config: None,
            compile_options: None,
            target_arguments: Some(target_args),
            legacy_executable: None,
            source_uri: String::from("tpu://module_0"),
        };
        assert_eq!(
            TpuExecutableProto::decode(&tpu_exec.encode()).ok(),
            Some(tpu_exec.clone())
        );
        assert!(tpu_exec.verify_schema_sync().is_ok());

        let dist_dev = DistributedDeviceProto {
            local_device_ordinal: 0,
            name: String::from("TPU_0"),
            vendor: String::from("Google"),
            global_device_id: 0,
            partition_index: 0,
            compute_capability: String::from("v5e"),
            core_count: 1,
            device_kind: String::from("Cloud TPU v5e"),
            to_string: String::from("TpuDevice(id=0)"),
            debug_string: String::from("TPU:0"),
            fabric_uuid: String::from("slice0/clique0"),
            numa_node: 0,
            device_memory_bytes_limit: 17_179_869_184,
        };
        assert_eq!(
            DistributedDeviceProto::decode(&dist_dev.encode()).ok(),
            Some(dist_dev.clone())
        );
        assert!(dist_dev.verify_schema_sync().is_ok());

        let local_topo = LocalTopologyProto {
            process_id: 0,
            boot_id: String::from("boot-uuid-0"),
            network_nodes: vec![String::from("spine-0"), String::from("rack-0")],
            devices: vec![dist_dev],
            partition_index: 0,
        };
        assert_eq!(
            LocalTopologyProto::decode(&local_topo.encode()).ok(),
            Some(local_topo.clone())
        );
        assert!(local_topo.verify_schema_sync().is_ok());

        let global_topo = GlobalTopologyProto {
            processes: vec![local_topo],
        };
        assert_eq!(
            GlobalTopologyProto::decode(&global_topo.encode()).ok(),
            Some(global_topo.clone())
        );
        assert!(global_topo.verify_schema_sync().is_ok());

        let buf_slice = BufferAllocationSliceProto {
            offset: 0,
            size: 4096,
            buffer_allocation_index: 0,
            element_type: 11,
        };
        assert_eq!(
            BufferAllocationSliceProto::decode(&buf_slice.encode()).ok(),
            Some(buf_slice.clone())
        );
        assert!(buf_slice.verify_schema_sync().is_ok());

        let buf_alloc = BufferAllocationProto {
            index: 0,
            size: 4096,
            is_thread_local: false,
            is_tuple: false,
            is_entry_computation_parameter: true,
            parameter_number: 0,
            is_constant: false,
            maybe_live_out: true,
            color: 0,
            assigned: vec![buf_slice],
        };
        assert_eq!(
            BufferAllocationProto::decode(&buf_alloc.encode()).ok(),
            Some(buf_alloc.clone())
        );
        assert!(buf_alloc.verify_schema_sync().is_ok());

        let buf_assign = BufferAssignmentProto {
            buffer_allocations: vec![buf_alloc],
        };
        assert_eq!(
            BufferAssignmentProto::decode(&buf_assign.encode()).ok(),
            Some(buf_assign.clone())
        );
        assert!(buf_assign.verify_schema_sync().is_ok());

        let ms_err = MegaScaleRuntimeErrorProto {
            error_type: 1,
            error_message: String::from("Watchdog timeout"),
            estimated_start_time_ms: 1_700_000_000,
            triggering_launch_id: 7,
            hostname: String::from("tpu-host-0"),
            job_name: String::from("jax_worker"),
            task_id: 0,
            event_id: 999,
            unrecoverable_error_type: 1,
            runtime_error_id: 888,
            hang_error_type: 1,
        };
        assert_eq!(
            MegaScaleRuntimeErrorProto::decode(&ms_err.encode()).ok(),
            Some(ms_err.clone())
        );
        assert!(ms_err.verify_schema_sync().is_ok());

        let split_cfg = SplitConfigProto {
            dimension: 0,
            split_indices: vec![64, 128],
        };
        assert_eq!(
            SplitConfigProto::decode(&split_cfg.encode()).ok(),
            Some(split_cfg.clone())
        );
        assert!(split_cfg.verify_schema_sync().is_ok());

        let op_meta = OpMetadata {
            op_type: String::from("jit_matmul"),
            op_name: String::from("main/dot_general"),
            source_file: String::from("model.py"),
            source_line: 42,
            deduplicated_name: String::from("fusion.1"),
            scheduling_name: String::from("fusion.1.sched"),
        };
        assert_eq!(
            OpMetadata::decode(&op_meta.encode()).ok(),
            Some(op_meta.clone())
        );
        assert!(op_meta.verify_schema_sync().is_ok());

        let exec_prof = ExecutionProfile {
            compilation_cache_hit: true,
            compile_time_ms: 12,
            compute_cycle_count: 1024,
            compute_time_ns: 5000,
            compute_and_transfer_time_ns: 8000,
            executable_size_in_bytes: 4096,
            profile_cache_hit: false,
            warmup_run_executed: true,
        };
        assert_eq!(
            ExecutionProfile::decode(&exec_prof.encode()).ok(),
            Some(exec_prof.clone())
        );
        assert!(exec_prof.verify_schema_sync().is_ok());

        let exec_h = ExecutionHandle { handle: 1 };
        assert_eq!(
            ExecutionHandle::decode(&exec_h.encode()).ok(),
            Some(exec_h.clone())
        );
        assert!(exec_h.verify_schema_sync().is_ok());

        let gdata_h = GlobalDataHandle { handle: 2 };
        assert_eq!(
            GlobalDataHandle::decode(&gdata_h.encode()).ok(),
            Some(gdata_h.clone())
        );
        assert!(gdata_h.verify_schema_sync().is_ok());

        let dev_h = DeviceHandle {
            handle: 3,
            device_count: 4,
        };
        assert_eq!(
            DeviceHandle::decode(&dev_h.encode()).ok(),
            Some(dev_h.clone())
        );
        assert!(dev_h.verify_schema_sync().is_ok());

        let chan_h = ChannelHandle {
            handle: 5,
            channel_type: 1,
        };
        assert_eq!(
            ChannelHandle::decode(&chan_h.encode()).ok(),
            Some(chan_h.clone())
        );
        assert!(chan_h.verify_schema_sync().is_ok());

        let lit_proto = LiteralProto {
            shape: None,
            preds: vec![true, false],
            u8s: vec![1, 2],
            s32s: vec![-1, 2],
            s64s: vec![-10, 20],
            u32s: vec![10, 20],
            u64s: vec![100, 200],
            f32s: vec![1.5, -2.5],
            f64s: vec![3.5, -4.5],
            tuple_literals: Vec::new(),
            f16s: vec![0, 0],
            c64s: vec![1.0, 0.0],
            bf16s: vec![0, 0],
            s8s: vec![1, 2],
            u16s: vec![0, 1],
            s16s: vec![0, 2],
            c128s: vec![2.0, -1.0],
            f8e5m2s: vec![0x3C],
            f8e4m3fns: vec![0x38],
            s4s: vec![0x12],
            u4s: vec![0x34],
            f8e4m3b11fnuzs: vec![0x56],
        };
        assert_eq!(
            LiteralProto::decode(&lit_proto.encode()).ok(),
            Some(lit_proto.clone())
        );
        assert!(lit_proto.verify_schema_sync().is_ok());

        let param_rep = ParameterReplicationProto {
            replicated_at_leaf_buffers: vec![true, false],
        };
        assert_eq!(
            ParameterReplicationProto::decode(&param_rep.encode()).ok(),
            Some(param_rep.clone())
        );
        assert!(param_rep.verify_schema_sync().is_ok());

        let slice_dims = HloSliceDimensionsProto {
            start: 0,
            limit: 8,
            stride: 1,
        };
        assert_eq!(
            HloSliceDimensionsProto::decode(&slice_dims.encode()).ok(),
            Some(slice_dims.clone())
        );
        assert!(slice_dims.verify_schema_sync().is_ok());

        let alias_entry = HloAliasEntryProto {
            output_shape_index: vec![0],
            parameter_number: 0,
            parameter_shape_index: vec![0],
            kind: 2,
        };
        assert_eq!(
            HloAliasEntryProto::decode(&alias_entry.encode()).ok(),
            Some(alias_entry.clone())
        );
        assert!(alias_entry.verify_schema_sync().is_ok());

        let io_alias = HloInputOutputAliasProto {
            entries: vec![alias_entry],
        };
        assert_eq!(
            HloInputOutputAliasProto::decode(&io_alias.encode()).ok(),
            Some(io_alias.clone())
        );
        assert!(io_alias.verify_schema_sync().is_ok());

        let donor_entry = HloBufferDonorEntryProto {
            parameter_number: 1,
            parameter_shape_index: vec![0],
        };
        assert_eq!(
            HloBufferDonorEntryProto::decode(&donor_entry.encode()).ok(),
            Some(donor_entry.clone())
        );
        assert!(donor_entry.verify_schema_sync().is_ok());

        let buf_donor = HloBufferDonorProto {
            entries: vec![donor_entry],
        };
        assert_eq!(
            HloBufferDonorProto::decode(&buf_donor.encode()).ok(),
            Some(buf_donor.clone())
        );
        assert!(buf_donor.verify_schema_sync().is_ok());

        let cp_prefetch = CrossProgramPrefetchProto {
            parameter: 0,
            index: vec![0],
            offset: 256,
        };
        assert_eq!(
            CrossProgramPrefetchProto::decode(&cp_prefetch.encode()).ok(),
            Some(cp_prefetch.clone())
        );
        assert!(cp_prefetch.verify_schema_sync().is_ok());

        let reg_entry = RegisterTranslatorEntryProto {
            full_path: String::from("vlc_tc_tcs_core_state.RUN_STATUS"),
            byte_offset: 0xA842_2040,
            width_bits: 64,
            access_mode: String::from("RW"),
            reset_value: 0,
        };
        assert_eq!(
            RegisterTranslatorEntryProto::decode(&reg_entry.encode()).ok(),
            Some(reg_entry.clone())
        );
        assert!(reg_entry.verify_schema_sync().is_ok());

        let reg_data = RegisterTranslatorDataProto {
            chip_codename: String::from("viperfish_lite"),
            tpu_version: 5,
            hardware_abi_version: 65_537,
            registers: vec![reg_entry],
        };
        assert_eq!(
            RegisterTranslatorDataProto::decode(&reg_data.encode()).ok(),
            Some(reg_data.clone())
        );
        assert!(reg_data.verify_schema_sync().is_ok());

        let chip_cfg = TpuChipConfigProto {
            chip_name: String::from("viperfish_megacore"),
            tpu_version: 5,
            tensor_cores_per_chip: 2,
            sparse_cores_per_chip: 4,
            hbm_capacity_bytes: 102_005_473_280,
            vmem_capacity_bytes: 67_108_864,
            smem_capacity_bytes: 1_048_576,
            megacore_enabled: true,
        };
        assert_eq!(
            TpuChipConfigProto::decode(&chip_cfg.encode()).ok(),
            Some(chip_cfg.clone())
        );
        assert!(chip_cfg.verify_schema_sync().is_ok());

        let host_offload = HostOffloadingExecutableProto {
            hlo_module: Some(HloModuleProto::default()),
            executable_type: 1,
        };
        assert_eq!(
            HostOffloadingExecutableProto::decode(&host_offload.encode()).ok(),
            Some(host_offload.clone())
        );
        assert!(host_offload.verify_schema_sync().is_ok());

        let ffi_state = FfiExecutionStateProto {
            type_name: String::from("xla.ffi.state.test"),
            state: vec![1, 2, 3, 4],
        };
        assert_eq!(
            FfiExecutionStateProto::decode(&ffi_state.encode()).ok(),
            Some(ffi_state.clone())
        );
        assert!(ffi_state.verify_schema_sync().is_ok());

        let ffi_flat_attr = FfiFlatAttributeProto {
            str: b"attr_val".to_vec(),
        };
        assert_eq!(
            FfiFlatAttributeProto::decode(&ffi_flat_attr.encode()).ok(),
            Some(ffi_flat_attr.clone())
        );
        assert!(ffi_flat_attr.verify_schema_sync().is_ok());

        let mut ffi_map = BTreeMap::new();
        ffi_map.insert(String::from("k"), ffi_flat_attr);
        let ffi_attrs = FfiFlatAttributesMapProto { attrs: ffi_map };
        assert_eq!(
            FfiFlatAttributesMapProto::decode(&ffi_attrs.encode()).ok(),
            Some(ffi_attrs.clone())
        );
        assert!(ffi_attrs.verify_schema_sync().is_ok());

        let win_dim = WindowDimension {
            size: 3,
            stride: 1,
            padding_low: 1,
            padding_high: 1,
            window_dilation: 1,
            base_dilation: 1,
            window_reversal: false,
        };
        assert_eq!(
            WindowDimension::decode(&win_dim.encode()).ok(),
            Some(win_dim.clone())
        );
        assert!(win_dim.verify_schema_sync().is_ok());

        let win = Window {
            dimensions: vec![win_dim],
        };
        assert_eq!(Window::decode(&win.encode()).ok(), Some(win.clone()));
        assert!(win.verify_schema_sync().is_ok());

        let gather_dn = GatherDimensionNumbers {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            operand_batching_dims: Vec::new(),
            start_indices_batching_dims: Vec::new(),
        };
        assert_eq!(
            GatherDimensionNumbers::decode(&gather_dn.encode()).ok(),
            Some(gather_dn.clone())
        );
        assert!(gather_dn.verify_schema_sync().is_ok());

        let scatter_dn = ScatterDimensionNumbers {
            update_window_dims: vec![1],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
            input_batching_dims: Vec::new(),
            scatter_indices_batching_dims: Vec::new(),
        };
        assert_eq!(
            ScatterDimensionNumbers::decode(&scatter_dn.encode()).ok(),
            Some(scatter_dn.clone())
        );
        assert!(scatter_dn.verify_schema_sync().is_ok());

        let conv_dn = ConvolutionDimensionNumbers {
            kernel_input_feature_dimension: 2,
            kernel_output_feature_dimension: 3,
            kernel_spatial_dimensions: vec![0, 1],
            input_batch_dimension: 0,
            input_feature_dimension: 3,
            output_batch_dimension: 0,
            output_feature_dimension: 3,
            input_spatial_dimensions: vec![1, 2],
            output_spatial_dimensions: vec![1, 2],
        };
        assert_eq!(
            ConvolutionDimensionNumbers::decode(&conv_dn.encode()).ok(),
            Some(conv_dn.clone())
        );
        assert!(conv_dn.verify_schema_sync().is_ok());

        let dot_dn = DotDimensionNumbers {
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![0],
            lhs_batch_dimensions: vec![0],
            rhs_batch_dimensions: vec![0],
        };
        assert_eq!(
            DotDimensionNumbers::decode(&dot_dn.encode()).ok(),
            Some(dot_dn.clone())
        );
        assert!(dot_dn.verify_schema_sync().is_ok());

        let pad_dim = PaddingConfigDimension {
            edge_padding_low: 1,
            edge_padding_high: 1,
            interior_padding: 0,
        };
        assert_eq!(
            PaddingConfigDimension::decode(&pad_dim.encode()).ok(),
            Some(pad_dim.clone())
        );
        assert!(pad_dim.verify_schema_sync().is_ok());

        let pad_cfg = PaddingConfig {
            dimensions: vec![pad_dim],
        };
        assert_eq!(
            PaddingConfig::decode(&pad_cfg.encode()).ok(),
            Some(pad_cfg.clone())
        );
        assert!(pad_cfg.verify_schema_sync().is_ok());

        let rep_grp = ReplicaGroup {
            replica_ids: vec![0, 1, 2, 3],
        };
        assert_eq!(
            ReplicaGroup::decode(&rep_grp.encode()).ok(),
            Some(rep_grp.clone())
        );
        assert!(rep_grp.verify_schema_sync().is_ok());

        let src_tgt = SourceTarget {
            source: 0,
            target: 1,
        };
        assert_eq!(
            SourceTarget::decode(&src_tgt.encode()).ok(),
            Some(src_tgt.clone())
        );
        assert!(src_tgt.verify_schema_sync().is_ok());

        let prec_cfg = PrecisionConfig {
            operand_precision: vec![1, 2],
            algorithm: 6,
        };
        assert_eq!(
            PrecisionConfig::decode(&prec_cfg.encode()).ok(),
            Some(prec_cfg.clone())
        );
        assert!(prec_cfg.verify_schema_sync().is_ok());

        let mut fa_map = BTreeMap::new();
        fa_map.insert(String::from("mhlo.sharding"), String::from("{replicated}"));
        let front_attr = FrontendAttributes { map: fa_map };
        assert_eq!(
            FrontendAttributes::decode(&front_attr.encode()).ok(),
            Some(front_attr.clone())
        );
        assert!(front_attr.verify_schema_sync().is_ok());

        let stat = Statistic {
            stat_name: String::from("roofline"),
            stat_val: 92.5,
        };
        assert_eq!(Statistic::decode(&stat.encode()).ok(), Some(stat.clone()));
        assert!(stat.verify_schema_sync().is_ok());

        let stat_viz = StatisticsViz {
            stat_index_to_visualize: 0,
            statistics: vec![stat],
        };
        assert_eq!(
            StatisticsViz::decode(&stat_viz.encode()).ok(),
            Some(stat_viz.clone())
        );
        assert!(stat_viz.verify_schema_sync().is_ok());

        // Test invalid/corrupt wire bytes across all decode methods
        let corrupt = [0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert!(TileProto::decode(&corrupt).is_err());
        assert!(SplitConfigProto::decode(&corrupt).is_err());
        assert!(OpMetadata::decode(&corrupt).is_err());
        assert!(ExecutionProfile::decode(&corrupt).is_err());
        assert!(ExecutionHandle::decode(&corrupt).is_err());
        assert!(GlobalDataHandle::decode(&corrupt).is_err());
        assert!(DeviceHandle::decode(&corrupt).is_err());
        assert!(ChannelHandle::decode(&corrupt).is_err());
        assert!(LiteralProto::decode(&corrupt).is_err());
        assert!(WindowDimension::decode(&corrupt).is_err());
        assert!(Window::decode(&corrupt).is_err());
        assert!(GatherDimensionNumbers::decode(&corrupt).is_err());
        assert!(ScatterDimensionNumbers::decode(&corrupt).is_err());
        assert!(ConvolutionDimensionNumbers::decode(&corrupt).is_err());
        assert!(DotDimensionNumbers::decode(&corrupt).is_err());
        assert!(PaddingConfigDimension::decode(&corrupt).is_err());
        assert!(PaddingConfig::decode(&corrupt).is_err());
        assert!(ReplicaGroup::decode(&corrupt).is_err());
        assert!(SourceTarget::decode(&corrupt).is_err());
        assert!(PrecisionConfig::decode(&corrupt).is_err());
        assert!(FrontendAttributes::decode(&corrupt).is_err());
        assert!(Statistic::decode(&corrupt).is_err());
        assert!(StatisticsViz::decode(&corrupt).is_err());
        assert!(LayoutProto::decode(&corrupt).is_err());
        assert!(ShapeProto::decode(&corrupt).is_err());
        assert!(ProgramShapeProto::decode(&corrupt).is_err());
        assert!(OpShardingProto::decode(&corrupt).is_err());
        assert!(HloInstructionProto::decode(&corrupt).is_err());
        assert!(HloComputationProto::decode(&corrupt).is_err());
        assert!(HloModuleProto::decode(&corrupt).is_err());
        assert!(HloModuleConfigProto::decode(&corrupt).is_err());
        assert!(OptionOverrideProto::decode(&corrupt).is_err());
        assert!(ExecutableBuildOptionsProto::decode(&corrupt).is_err());
        assert!(CompileOptionsProto::decode(&corrupt).is_err());
        assert!(ExecuteOptionsProto::decode(&corrupt).is_err());
        assert!(CompiledMemoryStatsProto::decode(&corrupt).is_err());
        assert!(PjRtExecutableMetadataProto::decode(&corrupt).is_err());
        assert!(PjRtPartialProgramProto::decode(&corrupt).is_err());
        assert!(PjRtTopologyDescriptionProto::decode(&corrupt).is_err());
        assert!(TpuTopologyArgsProto::decode(&corrupt).is_err());
        assert!(TpuHardwareFeatureProto::decode(&corrupt).is_err());
        assert!(TopologyProto::decode(&corrupt).is_err());
        assert!(TpuSerializedExecutableProto::decode(&corrupt).is_err());
        assert!(XStatMetadataProto::decode(&corrupt).is_err());
        assert!(XEventMetadataProto::decode(&corrupt).is_err());
        assert!(XStatProto::decode(&corrupt).is_err());
        assert!(XEventProto::decode(&corrupt).is_err());
        assert!(XLineProto::decode(&corrupt).is_err());
        assert!(XPlaneProto::decode(&corrupt).is_err());
        assert!(XSpaceProto::decode(&corrupt).is_err());
        assert!(EmbeddingTableDescriptorProto::decode(&corrupt).is_err());
        assert!(EmbeddingFeatureDescriptorProto::decode(&corrupt).is_err());
        assert!(EmbeddingSpmdShardingProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingConfigurationProto::decode(&corrupt).is_err());
        assert!(SparseCoreTableLayoutProto::decode(&corrupt).is_err());
        assert!(SparseCoreTableLayoutsProto::decode(&corrupt).is_err());
        assert!(ComputationDeviceProto::decode(&corrupt).is_err());
        assert!(DeviceAssignmentProto::decode(&corrupt).is_err());
        assert!(TpuCompileMetadataProto::decode(&corrupt).is_err());
        assert!(TpuCompilationEnvironmentProto::decode(&corrupt).is_err());
        assert!(HloModuleProtoWithConfig::decode(&corrupt).is_err());
        assert!(CompilationResultProto::decode(&corrupt).is_err());
        assert!(RuntimeAbiVersionProto::decode(&corrupt).is_err());
        assert!(ExecutableAbiVersionProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingPartitionedConfigProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingMemoryConfigProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingHostNetworkConfigProto::decode(&corrupt).is_err());
        assert!(MegascaleMultiSliceConfigProto::decode(&corrupt).is_err());
        assert!(MegascaleDcnTopologyProto::decode(&corrupt).is_err());
        assert!(MegascaleRuntimeMetricsProto::decode(&corrupt).is_err());
        assert!(HostNetworkAddressProto::decode(&corrupt).is_err());
        assert!(NetworkAddressMappingProto::decode(&corrupt).is_err());
        assert!(EndpointAddressesProto::decode(&corrupt).is_err());
        assert!(MegascaleTransportConfigProto::decode(&corrupt).is_err());
        assert!(IsaBundleSlotProto::decode(&corrupt).is_err());
        assert!(IsaVliwBundleProto::decode(&corrupt).is_err());
        assert!(IsaCoreProgramHeaderProto::decode(&corrupt).is_err());
        assert!(IsaSymbolTableEntryProto::decode(&corrupt).is_err());
        assert!(IsaProgramProto::decode(&corrupt).is_err());
        assert!(TargetArgumentsProto::decode(&corrupt).is_err());
        assert!(DeepseaOverlaySectionProto::decode(&corrupt).is_err());
        assert!(DeepseaRelocationEntryProto::decode(&corrupt).is_err());
        assert!(DeepseaConstantPoolEntryProto::decode(&corrupt).is_err());
        assert!(DeepseaSyncFlagInitProto::decode(&corrupt).is_err());
        assert!(TpuCoreProgramAbiProto::decode(&corrupt).is_err());
        assert!(TpuMemorySegmentRangeProto::decode(&corrupt).is_err());
        assert!(TpuMemorySegmentProto::decode(&corrupt).is_err());
        assert!(TpuMemorySegmentSetProto::decode(&corrupt).is_err());
        assert!(TpuCoreProgramProto::decode(&corrupt).is_err());
        assert!(DeepseaExecutableProto::decode(&corrupt).is_err());
        assert!(HostCommandEnvelopeProto::decode(&corrupt).is_err());
        assert!(TpuExecutableProto::decode(&corrupt).is_err());
        assert!(DistributedDeviceProto::decode(&corrupt).is_err());
        assert!(LocalTopologyProto::decode(&corrupt).is_err());
        assert!(GlobalTopologyProto::decode(&corrupt).is_err());
        assert!(BufferAllocationSliceProto::decode(&corrupt).is_err());
        assert!(BufferAllocationProto::decode(&corrupt).is_err());
        assert!(BufferAssignmentProto::decode(&corrupt).is_err());
        assert!(MegaScaleRuntimeErrorProto::decode(&corrupt).is_err());
        assert!(ParameterReplicationProto::decode(&corrupt).is_err());
        assert!(HloSliceDimensionsProto::decode(&corrupt).is_err());
        assert!(HloAliasEntryProto::decode(&corrupt).is_err());
        assert!(HloInputOutputAliasProto::decode(&corrupt).is_err());
        assert!(HloBufferDonorEntryProto::decode(&corrupt).is_err());
        assert!(HloBufferDonorProto::decode(&corrupt).is_err());
        assert!(CrossProgramPrefetchProto::decode(&corrupt).is_err());
        assert!(RegisterTranslatorEntryProto::decode(&corrupt).is_err());
        assert!(RegisterTranslatorDataProto::decode(&corrupt).is_err());
        assert!(TpuChipConfigProto::decode(&corrupt).is_err());
        assert!(HostOffloadingExecutableProto::decode(&corrupt).is_err());
        assert!(FfiExecutionStateProto::decode(&corrupt).is_err());
        assert!(FfiFlatAttributeProto::decode(&corrupt).is_err());
        assert!(FfiFlatAttributesMapProto::decode(&corrupt).is_err());
    }

    proptest! {
        #[test]
        fn prop_shape_and_topology_proto_roundtrip(
            element_type in 0_i32..24_i32,
            dims in prop::collection::vec(1_i64..4096_i64, 0..6),
            minor_to_major in prop::collection::vec(0_i64..6_i64, 0..6),
            memory_space in 0_i64..4_i64,
            version in 1_i32..8_i32,
            num_hosts in 1_i32..64_i32,
            chips_per_host in 1_i32..8_i32,
            mesh_bounds in prop::collection::vec(1_i64..16_i64, 3..=3),
            platform_version in "[a-zA-Z0-9._-]{1,32}",
        ) {
            let shape = ShapeProto {
                element_type,
                dimensions: dims.clone(),
                tuple_shapes: Vec::new(),
                layout: Some(LayoutProto {
                    minor_to_major,
                    tiles: vec![TileProto { dimensions: vec![8, 128] }],
                    tail_padding_alignment_in_elements: 128,
                    index_primitive_type: 0,
                    pointer_primitive_type: 0,
                    memory_space,
                    element_size_in_bits: 32,
                    dynamic_shape_metadata_prefix_bytes: 0,
                }),
                is_dynamic_dimension: vec![false; dims.len()],
            };
            let encoded = shape.encode();
            prop_assert_eq!(ShapeProto::decode(&encoded).ok(), Some(shape.clone()));
            prop_assert!(shape.verify_schema_sync().is_ok());

            let topo = TpuTopologyArgsProto {
                version,
                num_hosts,
                chips_per_host,
                mesh_bounds,
                platform_version: platform_version.clone(),
                fingerprint: sha256_fingerprint_hex(platform_version.as_bytes()),
            };
            let topo_encoded = topo.encode();
            prop_assert_eq!(TpuTopologyArgsProto::decode(&topo_encoded).ok(), Some(topo.clone()));
            prop_assert!(topo.verify_schema_sync().is_ok());
        }
    }
}
