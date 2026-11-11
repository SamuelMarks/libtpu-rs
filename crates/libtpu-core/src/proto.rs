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

        let hh_empty = riegeli_highwayhash64(b"");
        let hh_short = riegeli_highwayhash64(b"short_payload");
        let hh_long = riegeli_highwayhash64(&[0xAB_u8; 96]);
        assert_ne!(hh_empty, hh_short);
        assert_ne!(hh_short, hh_long);
        let hh_state = HighwayHash64State::new(&RIEGELI_HIGHWAYHASH_KEY);
        assert_eq!(hh_state.clone(), hh_state);
        assert_ne!(format!("{hh_state:?}"), "");

        let wrapped = RiegeliRecordCodec::wrap_record(b"riegeli_proto_payload");
        assert_eq!(
            RiegeliRecordCodec::unwrap_record_checked(&wrapped).ok(),
            Some(b"riegeli_proto_payload".as_slice())
        );
        assert_eq!(
            RiegeliRecordCodec::unwrap_record_checked(b"plain_proto").ok(),
            Some(b"plain_proto".as_slice())
        );
        let mut legacy_zero_hash = wrapped.clone();
        if let Some(hash_slice) = legacy_zero_hash.get_mut(48..64) {
            for b in hash_slice {
                *b = 0;
            }
        }
        assert_eq!(
            RiegeliRecordCodec::unwrap_record_checked(&legacy_zero_hash).ok(),
            Some(b"riegeli_proto_payload".as_slice())
        );
        let mut corrupt_header = wrapped.clone();
        if let Some(b48) = corrupt_header.get_mut(48) {
            *b48 ^= 0xFF;
        }
        assert!(RiegeliRecordCodec::unwrap_record_checked(&corrupt_header).is_err());
        let mut corrupt_payload = wrapped;
        if let Some(b64) = corrupt_payload.get_mut(64) {
            *b64 ^= 0xFF;
        }
        assert!(RiegeliRecordCodec::unwrap_record_checked(&corrupt_payload).is_err());

        let small_recs = vec![b"rec_one".to_vec(), b"rec_two".to_vec()];
        let small_stream = RiegeliRecordCodec::wrap_records(&small_recs);
        assert_eq!(
            RiegeliRecordCodec::unwrap_records(&small_stream).ok(),
            Some(small_recs)
        );
        let large_recs = vec![vec![0x11_u8; 40_000], vec![0x22_u8; 40_000]];
        let large_stream = RiegeliRecordCodec::wrap_records(&large_recs);
        assert!(large_stream.len() > RiegeliRecordCodec::BLOCK_BOUNDARY_BYTES);
        assert_eq!(
            RiegeliRecordCodec::unwrap_records(&large_stream).ok(),
            Some(large_recs)
        );
        assert!(RiegeliRecordCodec::unwrap_records(b"short").is_err());
        let mut trunc_stream = RiegeliRecordCodec::wrap_record(b"longer_payload_bytes");
        trunc_stream.truncate(68);
        assert!(RiegeliRecordCodec::unwrap_records(&trunc_stream).is_err());
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
        assert_eq!(protodesc_cold_293_file_count(), 293);
        assert!(lookup_protodesc_cold_file_descriptor("xla/service/hlo.proto").is_some());
        assert!(lookup_protodesc_cold_file_descriptor("hlo.proto").is_some());
        assert!(lookup_protodesc_cold_file_descriptor("unknown_alias_file.proto").is_some());

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
        comp_env.set_field_from_str("5", "true");
        comp_env.set_field_from_str("6", "true");
        comp_env.set_field_from_str("7", "true");
        comp_env.set_field_from_str("8", "true");
        comp_env.set_field_from_str("--xla_tpu_spmd_rng_bit_generator_unsafe", "true");
        comp_env.set_field_from_str("extra_custom_flag", "123");
        assert_eq!(comp_env.get_field_as_string("5"), "true");
        assert_eq!(comp_env.get_field_as_string("6"), "true");
        assert_eq!(comp_env.get_field_as_string("7"), "true");
        assert_eq!(comp_env.get_field_as_string("8"), "true");
        assert_eq!(comp_env.get_field_as_string("extra_custom_flag"), "123");
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
        comp_env2.set_flag_by_name("--xla_tpu_enable_flash_attention", "true");
        assert_eq!(
            comp_env2.get_flag_by_name("--xla_tpu_enable_flash_attention"),
            "true"
        );
        comp_env2.set_flag_by_name("--xla_tpu_use_shardy_partitioner", "true");
        comp_env2.set_flag_by_name("--xla_tpu_nd_short_transfer_max_chunks", "16");
        comp_env2.set_flag_by_name(
            "--xla_tpu_ici_all_reduce_combine_threshold_bytes",
            "1048576",
        );
        comp_env2.set_flag_by_name("--xla_tpu_bf16_emission_mode", "HIGH");
        comp_env2.set_flag_by_name("25", "3");
        comp_env2.set_flag_by_name("10", "ignored_submsg");
        assert!(comp_env2.xla_tpu_use_shardy_partitioner);
        assert_eq!(comp_env2.xla_tpu_nd_short_transfer_max_chunks, 16);
        assert_eq!(
            comp_env2.xla_tpu_ici_all_reduce_combine_threshold_bytes,
            1_048_576
        );
        assert_eq!(comp_env2.xla_tpu_bf16_emission_mode, "HIGH");
        assert_eq!(comp_env2.xla_tpu_autotune_level, 3);
        assert_eq!(
            comp_env2.get_flag_by_name("xla_tpu_use_shardy_partitioner"),
            "true"
        );
        assert_eq!(comp_env2.get_flag_by_name("20"), "16");
        assert_eq!(comp_env2.get_flag_by_name("21"), "1048576");
        assert_eq!(comp_env2.get_flag_by_name("24"), "HIGH");
        assert_ne!(comp_env2.get_flag_by_name("10"), "default");
        assert_ne!(comp_env2.get_flag_by_name("model_version"), "default");
        assert_eq!(
            verify_all_registered_proto_schemas_strict(),
            Ok(ALL_REGISTERED_PROTO_MESSAGE_NAMES.len())
        );
        assert!(verify_registered_proto_schemas_closure(&[]).is_err());
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
            ..LayoutProto::default()
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
            ..OpShardingProto::default()
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
            debug_options: Some(DebugOptionsProto {
                xla_dump_to: String::from("/tmp/hlo"),
                xla_dump_hlo_as_text: true,
                xla_dump_hlo_as_proto: false,
                xla_backend_optimization_level: 3,
                ..DebugOptionsProto::default()
            }),
            num_replicas: 2,
            num_partitions: 4,
            use_spmd_partitioning: true,
            use_auto_spmd_partitioning: true,
            deduplicate_hlo: true,
            device_assignment: Some(DeviceAssignmentProto {
                replica_count: 2,
                computation_count: 4,
                computation_devices: vec![ComputationDeviceProto {
                    replica_device_ids: vec![0, 1],
                }],
            }),
            alias_passthrough_params: false,
            run_backend_only: false,
            allow_spmd_sharding_propagation_to_output: vec![true, false],
            comp_envs: Some(CompilationEnvironmentsProto {
                environments: vec![vec![0x08, 0x01]],
            }),
            fdo_profile: vec![0xAA, 0xBB],
            device_memory_size: 17_179_869_184,
            auto_spmd_partitioning_mesh_shape: vec![2, 2],
            auto_spmd_partitioning_mesh_ids: vec![0, 1],
            allow_spmd_sharding_propagation_to_parameters: vec![false, true],
            use_shardy_partitioner: true,
            process_index: 1,
            process_count: 4,
            optimization_level: 3,
            memory_fitting_level: 2,
            gpu_topology: Some(GpuTopologyProto {
                num_partitions: 1,
                num_hosts_per_partition: 1,
                num_devices_per_host: 4,
            }),
        };
        let mut bopts_bytes = bopts.encode();
        assert_eq!(
            ExecutableBuildOptionsProto::decode(&bopts_bytes).ok(),
            Some(bopts.clone())
        );
        assert!(bopts.verify_schema_sync().is_ok());
        bopts_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
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
            profile_version: 7,
            serialized_multi_slice_config: vec![1, 2, 3],
            env_option_overrides: env_overrides,
            target_config: Some(GpuTargetConfigProto {
                gpu_device_info: None,
                platform_name: String::from("tpu"),
                dnn_version_info: None,
                autotune_results: None,
                device_description_str: String::from("Cloud TPU v5e"),
                runtime_abi_version: None,
            }),
            allow_in_place_mlir_modification: true,
            matrix_unit_operand_precision: 1,
            compiler_variant: Some(String::from("jellyfish")),
            individually_defined_output_indices: vec![0],
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
            use_major_to_minor_data_layout_for_callbacks: true,
            seed: 12345,
            use_output_arena: false,
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
            version: String::from("0.0.52"),
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
            platform_version: String::from("libtpu-rs 0.0.52"),
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
            platform_version: String::from("0.0.52"),
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
            stats: Vec::new(),
            child_id: vec![3],
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
            use_manual_partitioning: true,
            use_shardy_partitioner: true,
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
            stacked_table_name: String::from("stacked_vocab"),
            unsharded_shape: vec![100_000, 128],
            unsharded_padded_shape: vec![100_096, 128],
            num_sparse_cores: 4,
            num_partitions: 2,
            total_rows_per_sparse_core_shard: 25_024,
            sparse_core_shard_row_offset: 0,
            sparse_core_shard_rotation: 1,
            per_sparse_core_batch_size: 16,
            num_features: 4,
            fdo_key: String::from("fdo_0"),
            spanner_fdo_key: String::from("spanner_0"),
        };
        let mut sc_layout_bytes = sc_layout.encode();
        assert_eq!(
            SparseCoreTableLayoutProto::decode(&sc_layout_bytes).ok(),
            Some(sc_layout.clone())
        );
        assert!(sc_layout.verify_schema_sync().is_ok());
        sc_layout_bytes.extend_from_slice(&[0xF8, 0x07, 0x01]);
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
            args: Vec::new(),
            retvals: Vec::new(),
            num_replicas: 4,
            num_cores_per_replica: 1,
            device_assignment: Some(da),
            padding_map: Vec::new(),
            session_handle: 0,
            function_library_fingerprint: String::new(),
            use_spmd_for_xla_partitioning: true,
            allow_separate_sharding_programs: false,
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
            ..TpuCompilationEnvironmentProto::default()
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
            error_code: 0,
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
            version_string: String::from("0.0.52"),
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
            compiler_version: String::from("libtpu-rs-xla-0.0.52"),
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
            slice_devices: Vec::new(),
            endpoint_addresses: None,
            dcn_topology: None,
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
            bytes_transferred: 8192,
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

        let prog_rel = ProgramRelocationProto {
            offset: 12,
            kind: 0,
            target_index: 0,
            addend: 4096,
        };
        assert_eq!(
            ProgramRelocationProto::decode(&prog_rel.encode()).ok(),
            Some(prog_rel.clone())
        );
        let codec = RiegeliRecordCodec;
        assert_eq!(codec.clone(), RiegeliRecordCodec);
        assert_ne!(format!("{codec:?}"), "");
        let wrapped_riegeli = RiegeliRecordCodec::wrap_record(&prog_rel.encode());
        assert_eq!(
            wrapped_riegeli.get(..8),
            Some(RIEGELI_BLOCK_MAGIC.as_slice())
        );
        assert_eq!(
            RiegeliRecordCodec::unwrap_record(&wrapped_riegeli),
            prog_rel.encode().as_slice()
        );
        let non_riegeli_64 = vec![0_u8; 70];
        assert_eq!(RiegeliRecordCodec::unwrap_record(&non_riegeli_64).len(), 70);
        assert_eq!(RiegeliRecordCodec::unwrap_record(b"plain_pb"), b"plain_pb");

        let isa_prog = IsaProgramProto {
            isa_architecture: String::from("vfc"),
            isa_abi_version: 163,
            core_programs: vec![isa_core],
            symbols: vec![isa_sym],
            fingerprint: 123_456_789,
            smem_size_bytes: 1_048_576,
            vmem_size_bytes: 16_777_216,
            cmem_size_bytes: 0,
            imem_size_bytes: 4096,
            stack_frame_size_bytes: 4096,
            sync_flag_count: 512,
            xla_compiler_version: String::from("libtpu-rs-0.0.52"),
            llo_schedule_fingerprint: 987_654_321,
            predication_mode: 0,
            bundle_compression_kind: 0,
            compilation_id: 1,
            scs_start_bundle: 0,
            tile_start_bundle: 0,
            relocations: vec![prog_rel],
            abi: None,
            program_name: String::from("test_prog"),
            oom_during_compilation: false,
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
            options: Some(TpuCoreProgramAbiOptionsProto {
                use_continuations: true,
            }),
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
            relocations: Vec::new(),
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
            ..DeepseaExecutableProto::default()
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
            shared_memory_per_block_optin: 0,
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
            assigned_logical_buffers: Vec::new(),
        };
        assert_eq!(
            BufferAllocationProto::decode(&buf_alloc.encode()).ok(),
            Some(buf_alloc.clone())
        );
        assert!(buf_alloc.verify_schema_sync().is_ok());

        let buf_assign = BufferAssignmentProto {
            logical_buffers: Vec::new(),
            buffer_allocations: vec![buf_alloc],
            heap_simulator_traces: Vec::new(),
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
            runtime_state: Some(RuntimeStateSummaryProto {
                rapid_eye_info: Some(RapidEyeInfoProto {
                    telemetry: Some(ExecutorTelemetryProto {
                        pending_graphs: vec![PendingGraphStateProto {
                            graph: Some(MegaScaleActionGraphProto {
                                id: String::from("g0"),
                                actions: vec![MegaScaleActionProto {
                                    inputs: vec![MegaScaleActionInputProto { action_index: 0 }],
                                    index: 1,
                                }],
                                hlo_module: String::from("main"),
                            }),
                            launch_id: 7,
                            completed_actions: vec![0],
                            pending_actions: vec![1],
                            active_actions: vec![1],
                        }],
                    }),
                    tpu_core_hlos: vec![TpuCoreHloProto {
                        chip_id: 0,
                        core_idx: 0,
                        hlo: String::from("%dot = f32[8,128] dot()"),
                        executable_fingerprint: String::from("deadbeef"),
                        pc: Some(64),
                        tag: Some(1),
                        hostname: String::from("tpu-host-0"),
                        hlo_name: String::from("dot"),
                        module_name: String::from("main"),
                        launch_id: 7,
                        tpu_core_type: 1,
                        computation_name: String::from("entry"),
                        recent_completions: vec![ProgramRunInfoProto {
                            run_id: 1,
                            launch_id: 6,
                            name: String::from("main"),
                            fingerprint: String::from("deadbeef"),
                        }],
                        active: Some(true),
                        errors: vec![StatusProto {
                            code: 4,
                            message: String::from("deadline exceeded"),
                            payload: BTreeMap::new(),
                        }],
                    }],
                    is_truncated: false,
                    num_slices: 2,
                    num_tasks_per_slice: 1,
                    slice_id: 0,
                    task_id: 0,
                    incarnation_id: 42,
                    tpu_version: 5,
                    chip_config_name: String::from("v5litepod"),
                }),
            }),
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
            ..OpMetadata::default()
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
            f8e5m2fnuzs: vec![0x78],
            f8e4m3fnuzs: vec![0x9A],
            f8e4m3s: vec![0xBC],
            f8e3m4s: vec![0xDE],
            f8e8m0fnus: vec![0x7F],
            s2s: vec![0x03],
            u2s: vec![0x02],
            f4e2m1fns: vec![0x11],
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
            scalar: None,
            array: None,
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

        let slice_cfg = SliceConfigProto {
            name: String::from("2x2x1"),
            dimensions: vec![2, 2, 1],
            wraps: vec![false, false, false],
            twist: false,
        };
        assert_eq!(
            SliceConfigProto::decode(&slice_cfg.encode()).ok(),
            Some(slice_cfg.clone())
        );
        assert!(slice_cfg.verify_schema_sync().is_ok());

        let plat_cfg = TpuPlatformConfigProto {
            platform_type_name: String::from("tpuv5e"),
            slice_configs: vec![slice_cfg],
            num_chips_per_tray: 4,
            num_trays: 2,
        };
        assert_eq!(
            TpuPlatformConfigProto::decode(&plat_cfg.encode()).ok(),
            Some(plat_cfg.clone())
        );
        assert!(plat_cfg.verify_schema_sync().is_ok());

        let plat_cfgs = TpuPlatformConfigsProto {
            platform_configs: vec![plat_cfg],
        };
        assert_eq!(
            TpuPlatformConfigsProto::decode(&plat_cfgs.encode()).ok(),
            Some(plat_cfgs.clone())
        );
        assert!(plat_cfgs.verify_schema_sync().is_ok());

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
        assert!(SliceConfigProto::decode(&corrupt).is_err());
        assert!(TpuPlatformConfigProto::decode(&corrupt).is_err());
        assert!(TpuPlatformConfigsProto::decode(&corrupt).is_err());
        assert!(ProgramRelocationProto::decode(&corrupt).is_err());
        assert!(TpuCoreProgramAbiOptionsProto::decode(&corrupt).is_err());
        assert!(TpuChipPartsProto::decode(&corrupt).is_err());
        assert!(TpuRuntimeAbiProto::decode(&corrupt).is_err());
        assert!(HloSnapshotProto::decode(&corrupt).is_err());
        assert!(StatusProto::decode(&corrupt).is_err());
        assert!(MegaScaleActionInputProto::decode(&corrupt).is_err());
        assert!(MegaScaleActionProto::decode(&corrupt).is_err());
        assert!(MegaScaleActionGraphProto::decode(&corrupt).is_err());
        assert!(PendingGraphStateProto::decode(&corrupt).is_err());
        assert!(ExecutorTelemetryProto::decode(&corrupt).is_err());
        assert!(ProgramRunInfoProto::decode(&corrupt).is_err());
        assert!(TpuCoreHloProto::decode(&corrupt).is_err());
        assert!(RapidEyeInfoProto::decode(&corrupt).is_err());
        assert!(RuntimeStateSummaryProto::decode(&corrupt).is_err());
        assert!(CompilationEnvironmentsProto::decode(&corrupt).is_err());
        assert!(DebugOptionsProto::decode(&corrupt).is_err());
        assert!(GpuTopologyProto::decode(&corrupt).is_err());
        assert!(GpuTargetConfigProto::decode(&corrupt).is_err());
        assert!(ExecutableAndOptionsProto::decode(&corrupt).is_err());
    }

    /// Generated `prost-build` modules compiled from `docs/reverse_engineering/protos/*.proto` into `$OUT_DIR`.
    #[allow(
        dead_code,
        unreachable_pub,
        clippy::nursery,
        clippy::doc_markdown,
        clippy::doc_lazy_continuation,
        clippy::doc_overindented_list_items,
        clippy::empty_line_after_doc_comments,
        clippy::enum_variant_names,
        clippy::large_enum_variant,
        clippy::struct_excessive_bools,
        clippy::struct_field_names,
        clippy::trivially_copy_pass_by_ref,
        clippy::derive_partial_eq_without_eq,
        clippy::default_trait_access
    )]
    mod generated_prost {
        /// Generated `jax` protobuf package.
        pub mod jax {
            include!(concat!(env!("OUT_DIR"), "/jax.rs"));
        }
        /// Generated `megascale` protobuf package.
        pub mod megascale {
            include!(concat!(env!("OUT_DIR"), "/megascale.rs"));
        }
        /// Generated `platforms_deepsea` protobuf package.
        pub mod platforms_deepsea {
            /// Generated `platforms_deepsea.jellyfish` protobuf package.
            pub mod jellyfish {
                /// Generated `platforms_deepsea.jellyfish.xdb` protobuf package.
                pub mod xdb {
                    include!(concat!(
                        env!("OUT_DIR"),
                        "/platforms_deepsea.jellyfish.xdb.rs"
                    ));
                }
            }
        }
        /// Generated `stream_executor` protobuf package.
        pub mod stream_executor {
            include!(concat!(env!("OUT_DIR"), "/stream_executor.rs"));
            /// Generated `stream_executor.dnn` protobuf package.
            pub mod dnn {
                include!(concat!(env!("OUT_DIR"), "/stream_executor.dnn.rs"));
            }
        }
        /// Generated `tensorflow` protobuf package.
        pub mod tensorflow {
            include!(concat!(env!("OUT_DIR"), "/tensorflow.rs"));
            /// Generated `tensorflow.error` protobuf package.
            pub mod error {
                include!(concat!(env!("OUT_DIR"), "/tensorflow.error.rs"));
            }
            /// Generated `tensorflow.profiler` protobuf package.
            pub mod profiler {
                include!(concat!(env!("OUT_DIR"), "/tensorflow.profiler.rs"));
            }
            /// Generated `tensorflow.tf2xla` protobuf package.
            pub mod tf2xla {
                include!(concat!(env!("OUT_DIR"), "/tensorflow.tf2xla.rs"));
            }
            /// Generated `tensorflow.tpu` protobuf package.
            pub mod tpu {
                include!(concat!(env!("OUT_DIR"), "/tensorflow.tpu.rs"));
            }
        }
        /// Generated `tpu` protobuf package.
        pub mod tpu {
            include!(concat!(env!("OUT_DIR"), "/tpu.rs"));
        }
        /// Generated `xla` protobuf package.
        pub mod xla {
            include!(concat!(env!("OUT_DIR"), "/xla.rs"));
            /// Generated `xla.autotuner` protobuf package.
            pub mod autotuner {
                include!(concat!(env!("OUT_DIR"), "/xla.autotuner.rs"));
            }
            /// Generated `xla.buffer_assignment` protobuf package.
            pub mod buffer_assignment {
                include!(concat!(env!("OUT_DIR"), "/xla.buffer_assignment.rs"));
            }
            /// Generated `xla.ffi` protobuf package.
            pub mod ffi {
                include!(concat!(env!("OUT_DIR"), "/xla.ffi.rs"));
            }
            /// Generated `xla.jellyfish` protobuf package.
            pub mod jellyfish {
                include!(concat!(env!("OUT_DIR"), "/xla.jellyfish.rs"));
            }
            /// Generated `xla.megascale` protobuf package.
            pub mod megascale {
                /// Generated `xla.megascale.runtime` protobuf package.
                pub mod runtime {
                    include!(concat!(env!("OUT_DIR"), "/xla.megascale.runtime.rs"));
                    /// Generated `xla.megascale.runtime.external` protobuf package.
                    pub mod external {
                        include!(concat!(
                            env!("OUT_DIR"),
                            "/xla.megascale.runtime.external.rs"
                        ));
                    }
                }
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_all_136_bidirectional_proto_struct_equivalence_and_schema_emission() {
        use prost::Message as _;

        #[derive(Clone, PartialEq, Eq, prost::Message)]
        struct IncompleteTileProto {}
        impl ProtoSchemaMessage for IncompleteTileProto {
            const PROTO_FULL_NAME: &'static str = "xla.TileProto";
        }

        // Verify wire compatibility with prost-build generated structs in $OUT_DIR
        let gen_tile = generated_prost::xla::TileProto {
            dimensions: vec![8, 128],
        };
        let decoded_tile = TileProto::decode(&gen_tile.encode_to_vec());
        assert_eq!(
            decoded_tile.ok(),
            Some(TileProto {
                dimensions: vec![8, 128]
            })
        );
        let gen_topo = generated_prost::tpu::TpuTopologyArgsProto {
            version: 6,
            num_hosts: 1,
            chips_per_host: 4,
            mesh_bounds: vec![2, 2, 1],
            platform_version: String::from("0.0.52"),
            fingerprint: String::from("abc"),
        };
        assert_eq!(
            TpuTopologyArgsProto::decode(&gen_topo.encode_to_vec())
                .ok()
                .map(|t| t.encode()),
            Some(gen_topo.encode_to_vec())
        );

        let corrupt = [0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        // Verify 1:1 bidirectional (.proto <-> Rust struct) equivalence across all 317 types
        assert!(AdadeltaParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(AdadeltaParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            AdadeltaParametersProto::decode(&AdadeltaParametersProto::default().encode()).ok(),
            Some(AdadeltaParametersProto::default())
        );
        assert!(AdagradMomentumParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(AdagradMomentumParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            AdagradMomentumParametersProto::decode(
                &AdagradMomentumParametersProto::default().encode()
            )
            .ok(),
            Some(AdagradMomentumParametersProto::default())
        );
        assert!(AdagradParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(AdagradParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            AdagradParametersProto::decode(&AdagradParametersProto::default().encode()).ok(),
            Some(AdagradParametersProto::default())
        );
        assert!(AdamParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(AdamParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            AdamParametersProto::decode(&AdamParametersProto::default().encode()).ok(),
            Some(AdamParametersProto::default())
        );
        assert!(AssignParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(AssignParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            AssignParametersProto::decode(&AssignParametersProto::default().encode()).ok(),
            Some(AssignParametersProto::default())
        );
        assert!(AssignmentAlgorithmForComputationsWithoutOrderingProto::verify_bidirectional_equivalence().is_ok());
        assert!(AssignmentAlgorithmForComputationsWithoutOrderingProto::decode(&corrupt).is_err());
        assert_eq!(
            AssignmentAlgorithmForComputationsWithoutOrderingProto::decode(
                &AssignmentAlgorithmForComputationsWithoutOrderingProto::default().encode()
            )
            .ok(),
            Some(AssignmentAlgorithmForComputationsWithoutOrderingProto::default())
        );
        assert!(AutotuneBackendConfigKeyProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneBackendConfigKeyProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneBackendConfigKeyProto::decode(
                &AutotuneBackendConfigKeyProto::default().encode()
            )
            .ok(),
            Some(AutotuneBackendConfigKeyProto::default())
        );
        assert!(AutotuneConvKeyProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneConvKeyProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneConvKeyProto::decode(&AutotuneConvKeyProto::default().encode()).ok(),
            Some(AutotuneConvKeyProto::default())
        );
        assert!(AutotuneCudaConvPlanKeyProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneCudaConvPlanKeyProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneCudaConvPlanKeyProto::decode(&AutotuneCudaConvPlanKeyProto::default().encode())
                .ok(),
            Some(AutotuneCudaConvPlanKeyProto::default())
        );
        assert!(AutotuneCustomKernelFusionKeyProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneCustomKernelFusionKeyProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneCustomKernelFusionKeyProto::decode(
                &AutotuneCustomKernelFusionKeyProto::default().encode()
            )
            .ok(),
            Some(AutotuneCustomKernelFusionKeyProto::default())
        );
        assert!(AutotuneFailureResultProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneFailureResultProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneFailureResultProto::decode(&AutotuneFailureResultProto::default().encode())
                .ok(),
            Some(AutotuneFailureResultProto::default())
        );
        assert!(AutotuneGemmKeyProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneGemmKeyProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneGemmKeyProto::decode(&AutotuneGemmKeyProto::default().encode()).ok(),
            Some(AutotuneGemmKeyProto::default())
        );
        assert!(AutotuneResultProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneResultProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneResultProto::decode(&AutotuneResultProto::default().encode()).ok(),
            Some(AutotuneResultProto::default())
        );
        assert!(AutotuneResultsEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneResultsEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneResultsEntryProto::decode(&AutotuneResultsEntryProto::default().encode()).ok(),
            Some(AutotuneResultsEntryProto::default())
        );
        assert!(AutotuneResultsProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneResultsProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneResultsProto::decode(&AutotuneResultsProto::default().encode()).ok(),
            Some(AutotuneResultsProto::default())
        );
        assert!(AutotuneTritonGemmKeyProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuneTritonGemmKeyProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuneTritonGemmKeyProto::decode(&AutotuneTritonGemmKeyProto::default().encode())
                .ok(),
            Some(AutotuneTritonGemmKeyProto::default())
        );
        assert!(AutotuningLogProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuningLogProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuningLogProto::decode(&AutotuningLogProto::default().encode()).ok(),
            Some(AutotuningLogProto::default())
        );
        assert!(AutotuningLogsProto::verify_bidirectional_equivalence().is_ok());
        assert!(AutotuningLogsProto::decode(&corrupt).is_err());
        assert_eq!(
            AutotuningLogsProto::decode(&AutotuningLogsProto::default().encode()).ok(),
            Some(AutotuningLogsProto::default())
        );
        assert!(BarrierErrorProto::verify_bidirectional_equivalence().is_ok());
        assert!(BarrierErrorProto::decode(&corrupt).is_err());
        assert_eq!(
            BarrierErrorProto::decode(&BarrierErrorProto::default().encode()).ok(),
            Some(BarrierErrorProto::default())
        );
        assert!(BarrierRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(BarrierRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            BarrierRequestProto::decode(&BarrierRequestProto::default().encode()).ok(),
            Some(BarrierRequestProto::default())
        );
        assert!(BarrierResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(BarrierResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            BarrierResponseProto::decode(&BarrierResponseProto::default().encode()).ok(),
            Some(BarrierResponseProto::default())
        );
        assert!(BoundedAdagradParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(BoundedAdagradParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            BoundedAdagradParametersProto::decode(
                &BoundedAdagradParametersProto::default().encode()
            )
            .ok(),
            Some(BoundedAdagradParametersProto::default())
        );
        assert!(BufferAllocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(BufferAllocationProto::decode(&corrupt).is_err());
        assert_eq!(
            BufferAllocationProto::decode(&BufferAllocationProto::default().encode()).ok(),
            Some(BufferAllocationProto::default())
        );
        assert!(BufferAllocationSliceProto::verify_bidirectional_equivalence().is_ok());
        assert!(BufferAllocationSliceProto::decode(&corrupt).is_err());
        assert_eq!(
            BufferAllocationSliceProto::decode(&BufferAllocationSliceProto::default().encode())
                .ok(),
            Some(BufferAllocationSliceProto::default())
        );
        assert!(BufferAssignmentAlgorithmProto::verify_bidirectional_equivalence().is_ok());
        assert!(BufferAssignmentAlgorithmProto::decode(&corrupt).is_err());
        assert_eq!(
            BufferAssignmentAlgorithmProto::decode(
                &BufferAssignmentAlgorithmProto::default().encode()
            )
            .ok(),
            Some(BufferAssignmentAlgorithmProto::default())
        );
        assert!(BufferAssignmentProto::verify_bidirectional_equivalence().is_ok());
        assert!(BufferAssignmentProto::decode(&corrupt).is_err());
        assert_eq!(
            BufferAssignmentProto::decode(&BufferAssignmentProto::default().encode()).ok(),
            Some(BufferAssignmentProto::default())
        );
        assert!(BufferIsolationConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(BufferIsolationConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            BufferIsolationConfigProto::decode(&BufferIsolationConfigProto::default().encode())
                .ok(),
            Some(BufferIsolationConfigProto::default())
        );
        assert!(CancelBarrierRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(CancelBarrierRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            CancelBarrierRequestProto::decode(&CancelBarrierRequestProto::default().encode()).ok(),
            Some(CancelBarrierRequestProto::default())
        );
        assert!(CancelBarrierResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(CancelBarrierResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            CancelBarrierResponseProto::decode(&CancelBarrierResponseProto::default().encode())
                .ok(),
            Some(CancelBarrierResponseProto::default())
        );
        assert!(CenteredRmsPropParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(CenteredRmsPropParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            CenteredRmsPropParametersProto::decode(
                &CenteredRmsPropParametersProto::default().encode()
            )
            .ok(),
            Some(CenteredRmsPropParametersProto::default())
        );
        assert!(ChannelHandle::verify_bidirectional_equivalence().is_ok());
        assert!(ChannelHandle::decode(&corrupt).is_err());
        assert_eq!(
            ChannelHandle::decode(&ChannelHandle::default().encode()).ok(),
            Some(ChannelHandle::default())
        );
        assert!(ClippingLimitsProto::verify_bidirectional_equivalence().is_ok());
        assert!(ClippingLimitsProto::decode(&corrupt).is_err());
        assert_eq!(
            ClippingLimitsProto::decode(&ClippingLimitsProto::default().encode()).ok(),
            Some(ClippingLimitsProto::default())
        );
        assert!(CompilationEnvironmentsProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilationEnvironmentsProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilationEnvironmentsProto::decode(&CompilationEnvironmentsProto::default().encode())
                .ok(),
            Some(CompilationEnvironmentsProto::default())
        );
        assert!(CompilationLogEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilationLogEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilationLogEntryProto::decode(&CompilationLogEntryProto::default().encode()).ok(),
            Some(CompilationLogEntryProto::default())
        );
        assert!(CompilationResultProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilationResultProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilationResultProto::decode(&CompilationResultProto::default().encode()).ok(),
            Some(CompilationResultProto::default())
        );
        assert!(CompileOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompileOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            CompileOptionsProto::decode(&CompileOptionsProto::default().encode()).ok(),
            Some(CompileOptionsProto::default())
        );
        assert!(CompiledMemoryStatsProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompiledMemoryStatsProto::decode(&corrupt).is_err());
        assert_eq!(
            CompiledMemoryStatsProto::decode(&CompiledMemoryStatsProto::default().encode()).ok(),
            Some(CompiledMemoryStatsProto::default())
        );
        assert!(CompilerBuildInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilerBuildInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilerBuildInfoProto::decode(&CompilerBuildInfoProto::default().encode()).ok(),
            Some(CompilerBuildInfoProto::default())
        );
        assert!(CompilerPerfStatsProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilerPerfStatsProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilerPerfStatsProto::decode(&CompilerPerfStatsProto::default().encode()).ok(),
            Some(CompilerPerfStatsProto::default())
        );
        assert!(ComputationDeviceProto::verify_bidirectional_equivalence().is_ok());
        assert!(ComputationDeviceProto::decode(&corrupt).is_err());
        assert_eq!(
            ComputationDeviceProto::decode(&ComputationDeviceProto::default().encode()).ok(),
            Some(ComputationDeviceProto::default())
        );
        assert!(ComputeCapabilityProto::verify_bidirectional_equivalence().is_ok());
        assert!(ComputeCapabilityProto::decode(&corrupt).is_err());
        assert_eq!(
            ComputeCapabilityProto::decode(&ComputeCapabilityProto::default().encode()).ok(),
            Some(ComputeCapabilityProto::default())
        );
        assert!(ContinuousProfilingResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(ContinuousProfilingResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            ContinuousProfilingResponseProto::decode(
                &ContinuousProfilingResponseProto::default().encode()
            )
            .ok(),
            Some(ContinuousProfilingResponseProto::default())
        );
        assert!(ConvolutionDimensionNumbers::verify_bidirectional_equivalence().is_ok());
        assert!(ConvolutionDimensionNumbers::decode(&corrupt).is_err());
        assert_eq!(
            ConvolutionDimensionNumbers::decode(&ConvolutionDimensionNumbers::default().encode())
                .ok(),
            Some(ConvolutionDimensionNumbers::default())
        );
        assert!(CoordinatedJobProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoordinatedJobProto::decode(&corrupt).is_err());
        assert_eq!(
            CoordinatedJobProto::decode(&CoordinatedJobProto::default().encode()).ok(),
            Some(CoordinatedJobProto::default())
        );
        assert!(CoordinatedTaskProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoordinatedTaskProto::decode(&corrupt).is_err());
        assert_eq!(
            CoordinatedTaskProto::decode(&CoordinatedTaskProto::default().encode()).ok(),
            Some(CoordinatedTaskProto::default())
        );
        assert!(CoordinatedTaskStateInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoordinatedTaskStateInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            CoordinatedTaskStateInfoProto::decode(
                &CoordinatedTaskStateInfoProto::default().encode()
            )
            .ok(),
            Some(CoordinatedTaskStateInfoProto::default())
        );
        assert!(CoordinationDeviceInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoordinationDeviceInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            CoordinationDeviceInfoProto::decode(&CoordinationDeviceInfoProto::default().encode())
                .ok(),
            Some(CoordinationDeviceInfoProto::default())
        );
        assert!(CoordinationServiceConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoordinationServiceConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            CoordinationServiceConfigProto::decode(
                &CoordinationServiceConfigProto::default().encode()
            )
            .ok(),
            Some(CoordinationServiceConfigProto::default())
        );
        assert!(CoordinationServiceErrorProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoordinationServiceErrorProto::decode(&corrupt).is_err());
        assert_eq!(
            CoordinationServiceErrorProto::decode(
                &CoordinationServiceErrorProto::default().encode()
            )
            .ok(),
            Some(CoordinationServiceErrorProto::default())
        );
        assert!(CrossProgramPrefetchProto::verify_bidirectional_equivalence().is_ok());
        assert!(CrossProgramPrefetchProto::decode(&corrupt).is_err());
        assert_eq!(
            CrossProgramPrefetchProto::decode(&CrossProgramPrefetchProto::default().encode()).ok(),
            Some(CrossProgramPrefetchProto::default())
        );
        assert!(CudaComputeCapabilityProto::verify_bidirectional_equivalence().is_ok());
        assert!(CudaComputeCapabilityProto::decode(&corrupt).is_err());
        assert_eq!(
            CudaComputeCapabilityProto::decode(&CudaComputeCapabilityProto::default().encode())
                .ok(),
            Some(CudaComputeCapabilityProto::default())
        );
        assert!(CudaPlatformVersionProto::verify_bidirectional_equivalence().is_ok());
        assert!(CudaPlatformVersionProto::decode(&corrupt).is_err());
        assert_eq!(
            CudaPlatformVersionProto::decode(&CudaPlatformVersionProto::default().encode()).ok(),
            Some(CudaPlatformVersionProto::default())
        );
        assert!(CudnnVersionProto::verify_bidirectional_equivalence().is_ok());
        assert!(CudnnVersionProto::decode(&corrupt).is_err());
        assert_eq!(
            CudnnVersionProto::decode(&CudnnVersionProto::default().encode()).ok(),
            Some(CudnnVersionProto::default())
        );
        assert!(DcnSliceRangeProto::verify_bidirectional_equivalence().is_ok());
        assert!(DcnSliceRangeProto::decode(&corrupt).is_err());
        assert_eq!(
            DcnSliceRangeProto::decode(&DcnSliceRangeProto::default().encode()).ok(),
            Some(DcnSliceRangeProto::default())
        );
        assert!(DcnSymmetricTreeProto::verify_bidirectional_equivalence().is_ok());
        assert!(DcnSymmetricTreeProto::decode(&corrupt).is_err());
        assert_eq!(
            DcnSymmetricTreeProto::decode(&DcnSymmetricTreeProto::default().encode()).ok(),
            Some(DcnSymmetricTreeProto::default())
        );
        assert!(DcnTopologyTreeProto::verify_bidirectional_equivalence().is_ok());
        assert!(DcnTopologyTreeProto::decode(&corrupt).is_err());
        assert_eq!(
            DcnTopologyTreeProto::decode(&DcnTopologyTreeProto::default().encode()).ok(),
            Some(DcnTopologyTreeProto::default())
        );
        assert!(DcnTreeNodeProto::verify_bidirectional_equivalence().is_ok());
        assert!(DcnTreeNodeProto::decode(&corrupt).is_err());
        assert_eq!(
            DcnTreeNodeProto::decode(&DcnTreeNodeProto::default().encode()).ok(),
            Some(DcnTreeNodeProto::default())
        );
        assert!(DebugOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(DebugOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            DebugOptionsProto::decode(&DebugOptionsProto::default().encode()).ok(),
            Some(DebugOptionsProto::default())
        );
        assert!(DeepseaConstantPoolEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeepseaConstantPoolEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            DeepseaConstantPoolEntryProto::decode(
                &DeepseaConstantPoolEntryProto::default().encode()
            )
            .ok(),
            Some(DeepseaConstantPoolEntryProto::default())
        );
        assert!(DeepseaExecutableProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeepseaExecutableProto::decode(&corrupt).is_err());
        assert_eq!(
            DeepseaExecutableProto::decode(&DeepseaExecutableProto::default().encode()).ok(),
            Some(DeepseaExecutableProto::default())
        );
        assert!(DeepseaOverlaySectionProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeepseaOverlaySectionProto::decode(&corrupt).is_err());
        assert_eq!(
            DeepseaOverlaySectionProto::decode(&DeepseaOverlaySectionProto::default().encode())
                .ok(),
            Some(DeepseaOverlaySectionProto::default())
        );
        assert!(DeepseaRelocationEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeepseaRelocationEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            DeepseaRelocationEntryProto::decode(&DeepseaRelocationEntryProto::default().encode())
                .ok(),
            Some(DeepseaRelocationEntryProto::default())
        );
        assert!(DeepseaSyncFlagInitProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeepseaSyncFlagInitProto::decode(&corrupt).is_err());
        assert_eq!(
            DeepseaSyncFlagInitProto::decode(&DeepseaSyncFlagInitProto::default().encode()).ok(),
            Some(DeepseaSyncFlagInitProto::default())
        );
        assert!(DeleteKeyValueRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeleteKeyValueRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            DeleteKeyValueRequestProto::decode(&DeleteKeyValueRequestProto::default().encode())
                .ok(),
            Some(DeleteKeyValueRequestProto::default())
        );
        assert!(DeleteKeyValueResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeleteKeyValueResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            DeleteKeyValueResponseProto::decode(&DeleteKeyValueResponseProto::default().encode())
                .ok(),
            Some(DeleteKeyValueResponseProto::default())
        );
        assert!(DeviceAssignmentProto::verify_bidirectional_equivalence().is_ok());
        assert!(DeviceAssignmentProto::decode(&corrupt).is_err());
        assert_eq!(
            DeviceAssignmentProto::decode(&DeviceAssignmentProto::default().encode()).ok(),
            Some(DeviceAssignmentProto::default())
        );
        assert!(DeviceHandle::verify_bidirectional_equivalence().is_ok());
        assert!(DeviceHandle::decode(&corrupt).is_err());
        assert_eq!(
            DeviceHandle::decode(&DeviceHandle::default().encode()).ok(),
            Some(DeviceHandle::default())
        );
        assert!(DictKeysProto::verify_bidirectional_equivalence().is_ok());
        assert!(DictKeysProto::decode(&corrupt).is_err());
        assert_eq!(
            DictKeysProto::decode(&DictKeysProto::default().encode()).ok(),
            Some(DictKeysProto::default())
        );
        assert!(DistributedDeviceProto::verify_bidirectional_equivalence().is_ok());
        assert!(DistributedDeviceProto::decode(&corrupt).is_err());
        assert_eq!(
            DistributedDeviceProto::decode(&DistributedDeviceProto::default().encode()).ok(),
            Some(DistributedDeviceProto::default())
        );
        assert!(DnnAlgorithmConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(DnnAlgorithmConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            DnnAlgorithmConfigProto::decode(&DnnAlgorithmConfigProto::default().encode()).ok(),
            Some(DnnAlgorithmConfigProto::default())
        );
        assert!(DnnAlgorithmProto::verify_bidirectional_equivalence().is_ok());
        assert!(DnnAlgorithmProto::decode(&corrupt).is_err());
        assert_eq!(
            DnnAlgorithmProto::decode(&DnnAlgorithmProto::default().encode()).ok(),
            Some(DnnAlgorithmProto::default())
        );
        assert!(DnnConvolutionDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(DnnConvolutionDescriptorProto::decode(&corrupt).is_err());
        assert_eq!(
            DnnConvolutionDescriptorProto::decode(
                &DnnConvolutionDescriptorProto::default().encode()
            )
            .ok(),
            Some(DnnConvolutionDescriptorProto::default())
        );
        assert!(DnnTensorDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(DnnTensorDescriptorProto::decode(&corrupt).is_err());
        assert_eq!(
            DnnTensorDescriptorProto::decode(&DnnTensorDescriptorProto::default().encode()).ok(),
            Some(DnnTensorDescriptorProto::default())
        );
        assert!(DnnVersionInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(DnnVersionInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            DnnVersionInfoProto::decode(&DnnVersionInfoProto::default().encode()).ok(),
            Some(DnnVersionInfoProto::default())
        );
        assert!(DotDimensionNumbers::verify_bidirectional_equivalence().is_ok());
        assert!(DotDimensionNumbers::decode(&corrupt).is_err());
        assert_eq!(
            DotDimensionNumbers::decode(&DotDimensionNumbers::default().encode()).ok(),
            Some(DotDimensionNumbers::default())
        );
        assert!(EmbeddingFeatureDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(EmbeddingFeatureDescriptorProto::decode(&corrupt).is_err());
        assert_eq!(
            EmbeddingFeatureDescriptorProto::decode(
                &EmbeddingFeatureDescriptorProto::default().encode()
            )
            .ok(),
            Some(EmbeddingFeatureDescriptorProto::default())
        );
        assert!(EmbeddingSpmdShardingProto::verify_bidirectional_equivalence().is_ok());
        assert!(EmbeddingSpmdShardingProto::decode(&corrupt).is_err());
        assert_eq!(
            EmbeddingSpmdShardingProto::decode(&EmbeddingSpmdShardingProto::default().encode())
                .ok(),
            Some(EmbeddingSpmdShardingProto::default())
        );
        assert!(EmbeddingTableDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(EmbeddingTableDescriptorProto::decode(&corrupt).is_err());
        assert_eq!(
            EmbeddingTableDescriptorProto::decode(
                &EmbeddingTableDescriptorProto::default().encode()
            )
            .ok(),
            Some(EmbeddingTableDescriptorProto::default())
        );
        assert!(EmitterLearnedCostModelOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(EmitterLearnedCostModelOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            EmitterLearnedCostModelOptionsProto::decode(
                &EmitterLearnedCostModelOptionsProto::default().encode()
            )
            .ok(),
            Some(EmitterLearnedCostModelOptionsProto::default())
        );
        assert!(EndpointAddressesProto::verify_bidirectional_equivalence().is_ok());
        assert!(EndpointAddressesProto::decode(&corrupt).is_err());
        assert_eq!(
            EndpointAddressesProto::decode(&EndpointAddressesProto::default().encode()).ok(),
            Some(EndpointAddressesProto::default())
        );
        assert!(ExecutableAbiVersionProto::verify_bidirectional_equivalence().is_ok());
        assert!(ExecutableAbiVersionProto::decode(&corrupt).is_err());
        assert_eq!(
            ExecutableAbiVersionProto::decode(&ExecutableAbiVersionProto::default().encode()).ok(),
            Some(ExecutableAbiVersionProto::default())
        );
        assert!(ExecutableAndOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(ExecutableAndOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            ExecutableAndOptionsProto::decode(&ExecutableAndOptionsProto::default().encode()).ok(),
            Some(ExecutableAndOptionsProto::default())
        );
        assert!(ExecutableBuildOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(ExecutableBuildOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            ExecutableBuildOptionsProto::decode(&ExecutableBuildOptionsProto::default().encode())
                .ok(),
            Some(ExecutableBuildOptionsProto::default())
        );
        assert!(ExecuteOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(ExecuteOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            ExecuteOptionsProto::decode(&ExecuteOptionsProto::default().encode()).ok(),
            Some(ExecuteOptionsProto::default())
        );
        assert!(ExecutionHandle::verify_bidirectional_equivalence().is_ok());
        assert!(ExecutionHandle::decode(&corrupt).is_err());
        assert_eq!(
            ExecutionHandle::decode(&ExecutionHandle::default().encode()).ok(),
            Some(ExecutionHandle::default())
        );
        assert!(ExecutionProfile::verify_bidirectional_equivalence().is_ok());
        assert!(ExecutionProfile::decode(&corrupt).is_err());
        assert_eq!(
            ExecutionProfile::decode(&ExecutionProfile::default().encode()).ok(),
            Some(ExecutionProfile::default())
        );
        assert!(ExecutorTelemetryProto::verify_bidirectional_equivalence().is_ok());
        assert!(ExecutorTelemetryProto::decode(&corrupt).is_err());
        assert_eq!(
            ExecutorTelemetryProto::decode(&ExecutorTelemetryProto::default().encode()).ok(),
            Some(ExecutorTelemetryProto::default())
        );
        assert!(FfiArrayProto::verify_bidirectional_equivalence().is_ok());
        assert!(FfiArrayProto::decode(&corrupt).is_err());
        assert_eq!(
            FfiArrayProto::decode(&FfiArrayProto::default().encode()).ok(),
            Some(FfiArrayProto::default())
        );
        assert!(FfiExecutionStateProto::verify_bidirectional_equivalence().is_ok());
        assert!(FfiExecutionStateProto::decode(&corrupt).is_err());
        assert_eq!(
            FfiExecutionStateProto::decode(&FfiExecutionStateProto::default().encode()).ok(),
            Some(FfiExecutionStateProto::default())
        );
        assert!(FfiFlatAttributeProto::verify_bidirectional_equivalence().is_ok());
        assert!(FfiFlatAttributeProto::decode(&corrupt).is_err());
        assert_eq!(
            FfiFlatAttributeProto::decode(&FfiFlatAttributeProto::default().encode()).ok(),
            Some(FfiFlatAttributeProto::default())
        );
        assert!(FfiFlatAttributesMapProto::verify_bidirectional_equivalence().is_ok());
        assert!(FfiFlatAttributesMapProto::decode(&corrupt).is_err());
        assert_eq!(
            FfiFlatAttributesMapProto::decode(&FfiFlatAttributesMapProto::default().encode()).ok(),
            Some(FfiFlatAttributesMapProto::default())
        );
        assert!(FfiScalarProto::verify_bidirectional_equivalence().is_ok());
        assert!(FfiScalarProto::decode(&corrupt).is_err());
        assert_eq!(
            FfiScalarProto::decode(&FfiScalarProto::default().encode()).ok(),
            Some(FfiScalarProto::default())
        );
        assert!(FrequencyAwareAdagradParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(FrequencyAwareAdagradParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            FrequencyAwareAdagradParametersProto::decode(
                &FrequencyAwareAdagradParametersProto::default().encode()
            )
            .ok(),
            Some(FrequencyAwareAdagradParametersProto::default())
        );
        assert!(FrequencyEstimatorParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(FrequencyEstimatorParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            FrequencyEstimatorParametersProto::decode(
                &FrequencyEstimatorParametersProto::default().encode()
            )
            .ok(),
            Some(FrequencyEstimatorParametersProto::default())
        );
        assert!(FrontendAttributes::verify_bidirectional_equivalence().is_ok());
        assert!(FrontendAttributes::decode(&corrupt).is_err());
        assert_eq!(
            FrontendAttributes::decode(&FrontendAttributes::default().encode()).ok(),
            Some(FrontendAttributes::default())
        );
        assert!(FtrlParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(FtrlParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            FtrlParametersProto::decode(&FtrlParametersProto::default().encode()).ok(),
            Some(FtrlParametersProto::default())
        );
        assert!(GatherDimensionNumbers::verify_bidirectional_equivalence().is_ok());
        assert!(GatherDimensionNumbers::decode(&corrupt).is_err());
        assert_eq!(
            GatherDimensionNumbers::decode(&GatherDimensionNumbers::default().encode()).ok(),
            Some(GatherDimensionNumbers::default())
        );
        assert!(GetAliveTasksRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetAliveTasksRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            GetAliveTasksRequestProto::decode(&GetAliveTasksRequestProto::default().encode()).ok(),
            Some(GetAliveTasksRequestProto::default())
        );
        assert!(GetAliveTasksResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetAliveTasksResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            GetAliveTasksResponseProto::decode(&GetAliveTasksResponseProto::default().encode())
                .ok(),
            Some(GetAliveTasksResponseProto::default())
        );
        assert!(GetKeyValueDirRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetKeyValueDirRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            GetKeyValueDirRequestProto::decode(&GetKeyValueDirRequestProto::default().encode())
                .ok(),
            Some(GetKeyValueDirRequestProto::default())
        );
        assert!(GetKeyValueDirResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetKeyValueDirResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            GetKeyValueDirResponseProto::decode(&GetKeyValueDirResponseProto::default().encode())
                .ok(),
            Some(GetKeyValueDirResponseProto::default())
        );
        assert!(GetKeyValueRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetKeyValueRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            GetKeyValueRequestProto::decode(&GetKeyValueRequestProto::default().encode()).ok(),
            Some(GetKeyValueRequestProto::default())
        );
        assert!(GetKeyValueResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetKeyValueResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            GetKeyValueResponseProto::decode(&GetKeyValueResponseProto::default().encode()).ok(),
            Some(GetKeyValueResponseProto::default())
        );
        assert!(GetSnapshotRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetSnapshotRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            GetSnapshotRequestProto::decode(&GetSnapshotRequestProto::default().encode()).ok(),
            Some(GetSnapshotRequestProto::default())
        );
        assert!(GetTaskStateRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetTaskStateRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            GetTaskStateRequestProto::decode(&GetTaskStateRequestProto::default().encode()).ok(),
            Some(GetTaskStateRequestProto::default())
        );
        assert!(GetTaskStateResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(GetTaskStateResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            GetTaskStateResponseProto::decode(&GetTaskStateResponseProto::default().encode()).ok(),
            Some(GetTaskStateResponseProto::default())
        );
        assert!(GlobalDataHandle::verify_bidirectional_equivalence().is_ok());
        assert!(GlobalDataHandle::decode(&corrupt).is_err());
        assert_eq!(
            GlobalDataHandle::decode(&GlobalDataHandle::default().encode()).ok(),
            Some(GlobalDataHandle::default())
        );
        assert!(GlobalTopologyProto::verify_bidirectional_equivalence().is_ok());
        assert!(GlobalTopologyProto::decode(&corrupt).is_err());
        assert_eq!(
            GlobalTopologyProto::decode(&GlobalTopologyProto::default().encode()).ok(),
            Some(GlobalTopologyProto::default())
        );
        assert!(GpuDeviceInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(GpuDeviceInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            GpuDeviceInfoProto::decode(&GpuDeviceInfoProto::default().encode()).ok(),
            Some(GpuDeviceInfoProto::default())
        );
        assert!(GpuTargetConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(GpuTargetConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            GpuTargetConfigProto::decode(&GpuTargetConfigProto::default().encode()).ok(),
            Some(GpuTargetConfigProto::default())
        );
        assert!(GpuTopologyProto::verify_bidirectional_equivalence().is_ok());
        assert!(GpuTopologyProto::decode(&corrupt).is_err());
        assert_eq!(
            GpuTopologyProto::decode(&GpuTopologyProto::default().encode()).ok(),
            Some(GpuTopologyProto::default())
        );
        assert!(GradientAccumulationStatusProto::verify_bidirectional_equivalence().is_ok());
        assert!(GradientAccumulationStatusProto::decode(&corrupt).is_err());
        assert_eq!(
            GradientAccumulationStatusProto::decode(
                &GradientAccumulationStatusProto::default().encode()
            )
            .ok(),
            Some(GradientAccumulationStatusProto::default())
        );
        assert!(HeartbeatRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(HeartbeatRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            HeartbeatRequestProto::decode(&HeartbeatRequestProto::default().encode()).ok(),
            Some(HeartbeatRequestProto::default())
        );
        assert!(HeartbeatResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(HeartbeatResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            HeartbeatResponseProto::decode(&HeartbeatResponseProto::default().encode()).ok(),
            Some(HeartbeatResponseProto::default())
        );
        assert!(HloAliasEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloAliasEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            HloAliasEntryProto::decode(&HloAliasEntryProto::default().encode()).ok(),
            Some(HloAliasEntryProto::default())
        );
        assert!(HloBufferDonorEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloBufferDonorEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            HloBufferDonorEntryProto::decode(&HloBufferDonorEntryProto::default().encode()).ok(),
            Some(HloBufferDonorEntryProto::default())
        );
        assert!(HloBufferDonorProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloBufferDonorProto::decode(&corrupt).is_err());
        assert_eq!(
            HloBufferDonorProto::decode(&HloBufferDonorProto::default().encode()).ok(),
            Some(HloBufferDonorProto::default())
        );
        assert!(HloComputationProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloComputationProto::decode(&corrupt).is_err());
        assert_eq!(
            HloComputationProto::decode(&HloComputationProto::default().encode()).ok(),
            Some(HloComputationProto::default())
        );
        assert!(HloInputOutputAliasProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloInputOutputAliasProto::decode(&corrupt).is_err());
        assert_eq!(
            HloInputOutputAliasProto::decode(&HloInputOutputAliasProto::default().encode()).ok(),
            Some(HloInputOutputAliasProto::default())
        );
        assert!(HloInstructionProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloInstructionProto::decode(&corrupt).is_err());
        assert_eq!(
            HloInstructionProto::decode(&HloInstructionProto::default().encode()).ok(),
            Some(HloInstructionProto::default())
        );
        assert!(HloModuleConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloModuleConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            HloModuleConfigProto::decode(&HloModuleConfigProto::default().encode()).ok(),
            Some(HloModuleConfigProto::default())
        );
        assert!(HloModuleProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloModuleProto::decode(&corrupt).is_err());
        assert_eq!(
            HloModuleProto::decode(&HloModuleProto::default().encode()).ok(),
            Some(HloModuleProto::default())
        );
        assert!(HloModuleProtoWithConfig::verify_bidirectional_equivalence().is_ok());
        assert!(HloModuleProtoWithConfig::decode(&corrupt).is_err());
        assert_eq!(
            HloModuleProtoWithConfig::decode(&HloModuleProtoWithConfig::default().encode()).ok(),
            Some(HloModuleProtoWithConfig::default())
        );
        assert!(HloSliceDimensionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloSliceDimensionsProto::decode(&corrupt).is_err());
        assert_eq!(
            HloSliceDimensionsProto::decode(&HloSliceDimensionsProto::default().encode()).ok(),
            Some(HloSliceDimensionsProto::default())
        );
        assert!(HloSnapshotProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloSnapshotProto::decode(&corrupt).is_err());
        assert_eq!(
            HloSnapshotProto::decode(&HloSnapshotProto::default().encode()).ok(),
            Some(HloSnapshotProto::default())
        );
        assert!(HostCommandEnvelopeProto::verify_bidirectional_equivalence().is_ok());
        assert!(HostCommandEnvelopeProto::decode(&corrupt).is_err());
        assert_eq!(
            HostCommandEnvelopeProto::decode(&HostCommandEnvelopeProto::default().encode()).ok(),
            Some(HostCommandEnvelopeProto::default())
        );
        assert!(HostComputeMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(HostComputeMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            HostComputeMetadataProto::decode(&HostComputeMetadataProto::default().encode()).ok(),
            Some(HostComputeMetadataProto::default())
        );
        assert!(HostNetworkAddressProto::verify_bidirectional_equivalence().is_ok());
        assert!(HostNetworkAddressProto::decode(&corrupt).is_err());
        assert_eq!(
            HostNetworkAddressProto::decode(&HostNetworkAddressProto::default().encode()).ok(),
            Some(HostNetworkAddressProto::default())
        );
        assert!(HostOffloadingExecutableProto::verify_bidirectional_equivalence().is_ok());
        assert!(HostOffloadingExecutableProto::decode(&corrupt).is_err());
        assert_eq!(
            HostOffloadingExecutableProto::decode(
                &HostOffloadingExecutableProto::default().encode()
            )
            .ok(),
            Some(HostOffloadingExecutableProto::default())
        );
        assert!(HostTransferMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(HostTransferMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            HostTransferMetadataProto::decode(&HostTransferMetadataProto::default().encode()).ok(),
            Some(HostTransferMetadataProto::default())
        );
        assert!(HostTransferTensorMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(HostTransferTensorMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            HostTransferTensorMetadataProto::decode(
                &HostTransferTensorMetadataProto::default().encode()
            )
            .ok(),
            Some(HostTransferTensorMetadataProto::default())
        );
        assert!(HotIdReplicationConfigurationProto::verify_bidirectional_equivalence().is_ok());
        assert!(HotIdReplicationConfigurationProto::decode(&corrupt).is_err());
        assert_eq!(
            HotIdReplicationConfigurationProto::decode(
                &HotIdReplicationConfigurationProto::default().encode()
            )
            .ok(),
            Some(HotIdReplicationConfigurationProto::default())
        );
        assert!(IncrementKeyValueRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(IncrementKeyValueRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            IncrementKeyValueRequestProto::decode(
                &IncrementKeyValueRequestProto::default().encode()
            )
            .ok(),
            Some(IncrementKeyValueRequestProto::default())
        );
        assert!(IncrementKeyValueResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(IncrementKeyValueResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            IncrementKeyValueResponseProto::decode(
                &IncrementKeyValueResponseProto::default().encode()
            )
            .ok(),
            Some(IncrementKeyValueResponseProto::default())
        );
        assert!(InsertKeyValueRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(InsertKeyValueRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            InsertKeyValueRequestProto::decode(&InsertKeyValueRequestProto::default().encode())
                .ok(),
            Some(InsertKeyValueRequestProto::default())
        );
        assert!(InsertKeyValueResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(InsertKeyValueResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            InsertKeyValueResponseProto::decode(&InsertKeyValueResponseProto::default().encode())
                .ok(),
            Some(InsertKeyValueResponseProto::default())
        );
        assert!(IsaBundleSlotProto::verify_bidirectional_equivalence().is_ok());
        assert!(IsaBundleSlotProto::decode(&corrupt).is_err());
        assert_eq!(
            IsaBundleSlotProto::decode(&IsaBundleSlotProto::default().encode()).ok(),
            Some(IsaBundleSlotProto::default())
        );
        assert!(IsaCoreProgramHeaderProto::verify_bidirectional_equivalence().is_ok());
        assert!(IsaCoreProgramHeaderProto::decode(&corrupt).is_err());
        assert_eq!(
            IsaCoreProgramHeaderProto::decode(&IsaCoreProgramHeaderProto::default().encode()).ok(),
            Some(IsaCoreProgramHeaderProto::default())
        );
        assert!(IsaProgramProto::verify_bidirectional_equivalence().is_ok());
        assert!(IsaProgramProto::decode(&corrupt).is_err());
        assert_eq!(
            IsaProgramProto::decode(&IsaProgramProto::default().encode()).ok(),
            Some(IsaProgramProto::default())
        );
        assert!(IsaSymbolTableEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(IsaSymbolTableEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            IsaSymbolTableEntryProto::decode(&IsaSymbolTableEntryProto::default().encode()).ok(),
            Some(IsaSymbolTableEntryProto::default())
        );
        assert!(IsaVliwBundleProto::verify_bidirectional_equivalence().is_ok());
        assert!(IsaVliwBundleProto::decode(&corrupt).is_err());
        assert_eq!(
            IsaVliwBundleProto::decode(&IsaVliwBundleProto::default().encode()).ok(),
            Some(IsaVliwBundleProto::default())
        );
        assert!(JobInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(JobInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            JobInfoProto::decode(&JobInfoProto::default().encode()).ok(),
            Some(JobInfoProto::default())
        );
        assert!(KeyValueEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(KeyValueEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            KeyValueEntryProto::decode(&KeyValueEntryProto::default().encode()).ok(),
            Some(KeyValueEntryProto::default())
        );
        assert!(KeyValueMetricProto::verify_bidirectional_equivalence().is_ok());
        assert!(KeyValueMetricProto::decode(&corrupt).is_err());
        assert_eq!(
            KeyValueMetricProto::decode(&KeyValueMetricProto::default().encode()).ok(),
            Some(KeyValueMetricProto::default())
        );
        assert!(LayoutProto::verify_bidirectional_equivalence().is_ok());
        assert!(LayoutProto::decode(&corrupt).is_err());
        assert_eq!(
            LayoutProto::decode(&LayoutProto::default().encode()).ok(),
            Some(LayoutProto::default())
        );
        assert!(LearningRateProto::verify_bidirectional_equivalence().is_ok());
        assert!(LearningRateProto::decode(&corrupt).is_err());
        assert_eq!(
            LearningRateProto::decode(&LearningRateProto::default().encode()).ok(),
            Some(LearningRateProto::default())
        );
        assert!(LionParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(LionParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            LionParametersProto::decode(&LionParametersProto::default().encode()).ok(),
            Some(LionParametersProto::default())
        );
        assert!(LiteralProto::verify_bidirectional_equivalence().is_ok());
        assert!(LiteralProto::decode(&corrupt).is_err());
        assert_eq!(
            LiteralProto::decode(&LiteralProto::default().encode()).ok(),
            Some(LiteralProto::default())
        );
        assert!(LocalTopologyProto::verify_bidirectional_equivalence().is_ok());
        assert!(LocalTopologyProto::decode(&corrupt).is_err());
        assert_eq!(
            LocalTopologyProto::decode(&LocalTopologyProto::default().encode()).ok(),
            Some(LocalTopologyProto::default())
        );
        assert!(LowDimensionalPackingStatusProto::verify_bidirectional_equivalence().is_ok());
        assert!(LowDimensionalPackingStatusProto::decode(&corrupt).is_err());
        assert_eq!(
            LowDimensionalPackingStatusProto::decode(
                &LowDimensionalPackingStatusProto::default().encode()
            )
            .ok(),
            Some(LowDimensionalPackingStatusProto::default())
        );
        assert!(MdlAdagradLightParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(MdlAdagradLightParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            MdlAdagradLightParametersProto::decode(
                &MdlAdagradLightParametersProto::default().encode()
            )
            .ok(),
            Some(MdlAdagradLightParametersProto::default())
        );
        assert!(MegaScaleActionGraphProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegaScaleActionGraphProto::decode(&corrupt).is_err());
        assert_eq!(
            MegaScaleActionGraphProto::decode(&MegaScaleActionGraphProto::default().encode()).ok(),
            Some(MegaScaleActionGraphProto::default())
        );
        assert!(MegaScaleActionInputProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegaScaleActionInputProto::decode(&corrupt).is_err());
        assert_eq!(
            MegaScaleActionInputProto::decode(&MegaScaleActionInputProto::default().encode()).ok(),
            Some(MegaScaleActionInputProto::default())
        );
        assert!(MegaScaleActionProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegaScaleActionProto::decode(&corrupt).is_err());
        assert_eq!(
            MegaScaleActionProto::decode(&MegaScaleActionProto::default().encode()).ok(),
            Some(MegaScaleActionProto::default())
        );
        assert!(MegaScaleRuntimeErrorProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegaScaleRuntimeErrorProto::decode(&corrupt).is_err());
        assert_eq!(
            MegaScaleRuntimeErrorProto::decode(&MegaScaleRuntimeErrorProto::default().encode())
                .ok(),
            Some(MegaScaleRuntimeErrorProto::default())
        );
        assert!(MegascaleDcnTopologyProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegascaleDcnTopologyProto::decode(&corrupt).is_err());
        assert_eq!(
            MegascaleDcnTopologyProto::decode(&MegascaleDcnTopologyProto::default().encode()).ok(),
            Some(MegascaleDcnTopologyProto::default())
        );
        assert!(MegascaleMultiSliceConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegascaleMultiSliceConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            MegascaleMultiSliceConfigProto::decode(
                &MegascaleMultiSliceConfigProto::default().encode()
            )
            .ok(),
            Some(MegascaleMultiSliceConfigProto::default())
        );
        assert!(MegascaleRuntimeMetricsProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegascaleRuntimeMetricsProto::decode(&corrupt).is_err());
        assert_eq!(
            MegascaleRuntimeMetricsProto::decode(&MegascaleRuntimeMetricsProto::default().encode())
                .ok(),
            Some(MegascaleRuntimeMetricsProto::default())
        );
        assert!(MegascaleTransportConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegascaleTransportConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            MegascaleTransportConfigProto::decode(
                &MegascaleTransportConfigProto::default().encode()
            )
            .ok(),
            Some(MegascaleTransportConfigProto::default())
        );
        assert!(MemoryBoundLoopOptimizerOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(MemoryBoundLoopOptimizerOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            MemoryBoundLoopOptimizerOptionsProto::decode(
                &MemoryBoundLoopOptimizerOptionsProto::default().encode()
            )
            .ok(),
            Some(MemoryBoundLoopOptimizerOptionsProto::default())
        );
        assert!(MemorySchedulerProto::verify_bidirectional_equivalence().is_ok());
        assert!(MemorySchedulerProto::decode(&corrupt).is_err());
        assert_eq!(
            MemorySchedulerProto::decode(&MemorySchedulerProto::default().encode()).ok(),
            Some(MemorySchedulerProto::default())
        );
        assert!(MemorySpaceAssignmentOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(MemorySpaceAssignmentOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            MemorySpaceAssignmentOptionsProto::decode(
                &MemorySpaceAssignmentOptionsProto::default().encode()
            )
            .ok(),
            Some(MemorySpaceAssignmentOptionsProto::default())
        );
        assert!(MomentumParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(MomentumParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            MomentumParametersProto::decode(&MomentumParametersProto::default().encode()).ok(),
            Some(MomentumParametersProto::default())
        );
        assert!(MonitorRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(MonitorRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            MonitorRequestProto::decode(&MonitorRequestProto::default().encode()).ok(),
            Some(MonitorRequestProto::default())
        );
        assert!(MonitorResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(MonitorResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            MonitorResponseProto::decode(&MonitorResponseProto::default().encode()).ok(),
            Some(MonitorResponseProto::default())
        );
        assert!(MsaSortOrderOverridesProto::verify_bidirectional_equivalence().is_ok());
        assert!(MsaSortOrderOverridesProto::decode(&corrupt).is_err());
        assert_eq!(
            MsaSortOrderOverridesProto::decode(&MsaSortOrderOverridesProto::default().encode())
                .ok(),
            Some(MsaSortOrderOverridesProto::default())
        );
        assert!(MsaTensorOverridesProto::verify_bidirectional_equivalence().is_ok());
        assert!(MsaTensorOverridesProto::decode(&corrupt).is_err());
        assert_eq!(
            MsaTensorOverridesProto::decode(&MsaTensorOverridesProto::default().encode()).ok(),
            Some(MsaTensorOverridesProto::default())
        );
        assert!(NetworkAddressMappingProto::verify_bidirectional_equivalence().is_ok());
        assert!(NetworkAddressMappingProto::decode(&corrupt).is_err());
        assert_eq!(
            NetworkAddressMappingProto::decode(&NetworkAddressMappingProto::default().encode())
                .ok(),
            Some(NetworkAddressMappingProto::default())
        );
        assert!(NullableShapedSliceProto::verify_bidirectional_equivalence().is_ok());
        assert!(NullableShapedSliceProto::decode(&corrupt).is_err());
        assert_eq!(
            NullableShapedSliceProto::decode(&NullableShapedSliceProto::default().encode()).ok(),
            Some(NullableShapedSliceProto::default())
        );
        assert!(OneApiComputeCapabilityProto::verify_bidirectional_equivalence().is_ok());
        assert!(OneApiComputeCapabilityProto::decode(&corrupt).is_err());
        assert_eq!(
            OneApiComputeCapabilityProto::decode(&OneApiComputeCapabilityProto::default().encode())
                .ok(),
            Some(OneApiComputeCapabilityProto::default())
        );
        assert!(OnlineYogiParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(OnlineYogiParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            OnlineYogiParametersProto::decode(&OnlineYogiParametersProto::default().encode()).ok(),
            Some(OnlineYogiParametersProto::default())
        );
        assert!(OpMetadata::verify_bidirectional_equivalence().is_ok());
        assert!(OpMetadata::decode(&corrupt).is_err());
        assert_eq!(
            OpMetadata::decode(&OpMetadata::default().encode()).ok(),
            Some(OpMetadata::default())
        );
        assert!(OpShardingProto::verify_bidirectional_equivalence().is_ok());
        assert!(OpShardingProto::decode(&corrupt).is_err());
        assert_eq!(
            OpShardingProto::decode(&OpShardingProto::default().encode()).ok(),
            Some(OpShardingProto::default())
        );
        assert!(OptimizationParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(OptimizationParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            OptimizationParametersProto::decode(&OptimizationParametersProto::default().encode())
                .ok(),
            Some(OptimizationParametersProto::default())
        );
        assert!(OptimizerDynamicInputProto::verify_bidirectional_equivalence().is_ok());
        assert!(OptimizerDynamicInputProto::decode(&corrupt).is_err());
        assert_eq!(
            OptimizerDynamicInputProto::decode(&OptimizerDynamicInputProto::default().encode())
                .ok(),
            Some(OptimizerDynamicInputProto::default())
        );
        assert!(OptionOverrideProto::verify_bidirectional_equivalence().is_ok());
        assert!(OptionOverrideProto::decode(&corrupt).is_err());
        assert_eq!(
            OptionOverrideProto::decode(&OptionOverrideProto::default().encode()).ok(),
            Some(OptionOverrideProto::default())
        );
        assert!(PaddingConfig::verify_bidirectional_equivalence().is_ok());
        assert!(PaddingConfig::decode(&corrupt).is_err());
        assert_eq!(
            PaddingConfig::decode(&PaddingConfig::default().encode()).ok(),
            Some(PaddingConfig::default())
        );
        assert!(PaddingConfigDimension::verify_bidirectional_equivalence().is_ok());
        assert!(PaddingConfigDimension::decode(&corrupt).is_err());
        assert_eq!(
            PaddingConfigDimension::decode(&PaddingConfigDimension::default().encode()).ok(),
            Some(PaddingConfigDimension::default())
        );
        assert!(PaddingMapProto::verify_bidirectional_equivalence().is_ok());
        assert!(PaddingMapProto::decode(&corrupt).is_err());
        assert_eq!(
            PaddingMapProto::decode(&PaddingMapProto::default().encode()).ok(),
            Some(PaddingMapProto::default())
        );
        assert!(ParameterReplicationProto::verify_bidirectional_equivalence().is_ok());
        assert!(ParameterReplicationProto::decode(&corrupt).is_err());
        assert_eq!(
            ParameterReplicationProto::decode(&ParameterReplicationProto::default().encode()).ok(),
            Some(ParameterReplicationProto::default())
        );
        assert!(PassMetricsProto::verify_bidirectional_equivalence().is_ok());
        assert!(PassMetricsProto::decode(&corrupt).is_err());
        assert_eq!(
            PassMetricsProto::decode(&PassMetricsProto::default().encode()).ok(),
            Some(PassMetricsProto::default())
        );
        assert!(PendingGraphStateProto::verify_bidirectional_equivalence().is_ok());
        assert!(PendingGraphStateProto::decode(&corrupt).is_err());
        assert_eq!(
            PendingGraphStateProto::decode(&PendingGraphStateProto::default().encode()).ok(),
            Some(PendingGraphStateProto::default())
        );
        assert!(PerTpuStepDetailsProto::verify_bidirectional_equivalence().is_ok());
        assert!(PerTpuStepDetailsProto::decode(&corrupt).is_err());
        assert_eq!(
            PerTpuStepDetailsProto::decode(&PerTpuStepDetailsProto::default().encode()).ok(),
            Some(PerTpuStepDetailsProto::default())
        );
        assert!(PjRtDeviceDimensionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(PjRtDeviceDimensionsProto::decode(&corrupt).is_err());
        assert_eq!(
            PjRtDeviceDimensionsProto::decode(&PjRtDeviceDimensionsProto::default().encode()).ok(),
            Some(PjRtDeviceDimensionsProto::default())
        );
        assert!(PjRtExecutableMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(PjRtExecutableMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            PjRtExecutableMetadataProto::decode(&PjRtExecutableMetadataProto::default().encode())
                .ok(),
            Some(PjRtExecutableMetadataProto::default())
        );
        assert!(PjRtPartialProgramProto::verify_bidirectional_equivalence().is_ok());
        assert!(PjRtPartialProgramProto::decode(&corrupt).is_err());
        assert_eq!(
            PjRtPartialProgramProto::decode(&PjRtPartialProgramProto::default().encode()).ok(),
            Some(PjRtPartialProgramProto::default())
        );
        assert!(PjRtTopologyDescriptionProto::verify_bidirectional_equivalence().is_ok());
        assert!(PjRtTopologyDescriptionProto::decode(&corrupt).is_err());
        assert_eq!(
            PjRtTopologyDescriptionProto::decode(&PjRtTopologyDescriptionProto::default().encode())
                .ok(),
            Some(PjRtTopologyDescriptionProto::default())
        );
        assert!(PjRtValueIntVectorProto::verify_bidirectional_equivalence().is_ok());
        assert!(PjRtValueIntVectorProto::decode(&corrupt).is_err());
        assert_eq!(
            PjRtValueIntVectorProto::decode(&PjRtValueIntVectorProto::default().encode()).ok(),
            Some(PjRtValueIntVectorProto::default())
        );
        assert!(PjRtValueTypeProto::verify_bidirectional_equivalence().is_ok());
        assert!(PjRtValueTypeProto::decode(&corrupt).is_err());
        assert_eq!(
            PjRtValueTypeProto::decode(&PjRtValueTypeProto::default().encode()).ok(),
            Some(PjRtValueTypeProto::default())
        );
        assert!(PollForErrorRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(PollForErrorRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            PollForErrorRequestProto::decode(&PollForErrorRequestProto::default().encode()).ok(),
            Some(PollForErrorRequestProto::default())
        );
        assert!(PollForErrorResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(PollForErrorResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            PollForErrorResponseProto::decode(&PollForErrorResponseProto::default().encode()).ok(),
            Some(PollForErrorResponseProto::default())
        );
        assert!(PrecisionConfig::verify_bidirectional_equivalence().is_ok());
        assert!(PrecisionConfig::decode(&corrupt).is_err());
        assert_eq!(
            PrecisionConfig::decode(&PrecisionConfig::default().encode()).ok(),
            Some(PrecisionConfig::default())
        );
        assert!(PreferredPrefetchOverridesProto::verify_bidirectional_equivalence().is_ok());
        assert!(PreferredPrefetchOverridesProto::decode(&corrupt).is_err());
        assert_eq!(
            PreferredPrefetchOverridesProto::decode(
                &PreferredPrefetchOverridesProto::default().encode()
            )
            .ok(),
            Some(PreferredPrefetchOverridesProto::default())
        );
        assert!(ProfileAdvancedConfigValueProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfileAdvancedConfigValueProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfileAdvancedConfigValueProto::decode(
                &ProfileAdvancedConfigValueProto::default().encode()
            )
            .ok(),
            Some(ProfileAdvancedConfigValueProto::default())
        );
        assert!(ProfileOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfileOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfileOptionsProto::decode(&ProfileOptionsProto::default().encode()).ok(),
            Some(ProfileOptionsProto::default())
        );
        assert!(ProfileRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfileRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfileRequestProto::decode(&ProfileRequestProto::default().encode()).ok(),
            Some(ProfileRequestProto::default())
        );
        assert!(ProfileResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfileResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfileResponseProto::decode(&ProfileResponseProto::default().encode()).ok(),
            Some(ProfileResponseProto::default())
        );
        assert!(ProfileToolDataProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfileToolDataProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfileToolDataProto::decode(&ProfileToolDataProto::default().encode()).ok(),
            Some(ProfileToolDataProto::default())
        );
        assert!(ProfileTraceOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfileTraceOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfileTraceOptionsProto::decode(&ProfileTraceOptionsProto::default().encode()).ok(),
            Some(ProfileTraceOptionsProto::default())
        );
        assert!(ProfilerServiceMonitorResultProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfilerServiceMonitorResultProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfilerServiceMonitorResultProto::decode(
                &ProfilerServiceMonitorResultProto::default().encode()
            )
            .ok(),
            Some(ProfilerServiceMonitorResultProto::default())
        );
        assert!(ProfilerTopologyLocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfilerTopologyLocationProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfilerTopologyLocationProto::decode(
                &ProfilerTopologyLocationProto::default().encode()
            )
            .ok(),
            Some(ProfilerTopologyLocationProto::default())
        );
        assert!(ProfilerTopologyProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProfilerTopologyProto::decode(&corrupt).is_err());
        assert_eq!(
            ProfilerTopologyProto::decode(&ProfilerTopologyProto::default().encode()).ok(),
            Some(ProfilerTopologyProto::default())
        );
        assert!(ProgramMemoryMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProgramMemoryMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            ProgramMemoryMetadataProto::decode(&ProgramMemoryMetadataProto::default().encode())
                .ok(),
            Some(ProgramMemoryMetadataProto::default())
        );
        assert!(ProgramRelocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProgramRelocationProto::decode(&corrupt).is_err());
        assert_eq!(
            ProgramRelocationProto::decode(&ProgramRelocationProto::default().encode()).ok(),
            Some(ProgramRelocationProto::default())
        );
        assert!(ProgramRunInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProgramRunInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            ProgramRunInfoProto::decode(&ProgramRunInfoProto::default().encode()).ok(),
            Some(ProgramRunInfoProto::default())
        );
        assert!(ProgramShapeProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProgramShapeProto::decode(&corrupt).is_err());
        assert_eq!(
            ProgramShapeProto::decode(&ProgramShapeProto::default().encode()).ok(),
            Some(ProgramShapeProto::default())
        );
        assert!(ProximalAdagradParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProximalAdagradParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            ProximalAdagradParametersProto::decode(
                &ProximalAdagradParametersProto::default().encode()
            )
            .ok(),
            Some(ProximalAdagradParametersProto::default())
        );
        assert!(ProximalYogiParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(ProximalYogiParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            ProximalYogiParametersProto::decode(&ProximalYogiParametersProto::default().encode())
                .ok(),
            Some(ProximalYogiParametersProto::default())
        );
        assert!(PyHostCallbackProto::verify_bidirectional_equivalence().is_ok());
        assert!(PyHostCallbackProto::decode(&corrupt).is_err());
        assert_eq!(
            PyHostCallbackProto::decode(&PyHostCallbackProto::default().encode()).ok(),
            Some(PyHostCallbackProto::default())
        );
        assert!(PyTreeDefProto::verify_bidirectional_equivalence().is_ok());
        assert!(PyTreeDefProto::decode(&corrupt).is_err());
        assert_eq!(
            PyTreeDefProto::decode(&PyTreeDefProto::default().encode()).ok(),
            Some(PyTreeDefProto::default())
        );
        assert!(PyTreeNodeDefProto::verify_bidirectional_equivalence().is_ok());
        assert!(PyTreeNodeDefProto::decode(&corrupt).is_err());
        assert_eq!(
            PyTreeNodeDefProto::decode(&PyTreeNodeDefProto::default().encode()).ok(),
            Some(PyTreeNodeDefProto::default())
        );
        assert!(RapidEyeInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(RapidEyeInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            RapidEyeInfoProto::decode(&RapidEyeInfoProto::default().encode()).ok(),
            Some(RapidEyeInfoProto::default())
        );
        assert!(RegisterTaskRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(RegisterTaskRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            RegisterTaskRequestProto::decode(&RegisterTaskRequestProto::default().encode()).ok(),
            Some(RegisterTaskRequestProto::default())
        );
        assert!(RegisterTaskResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(RegisterTaskResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            RegisterTaskResponseProto::decode(&RegisterTaskResponseProto::default().encode()).ok(),
            Some(RegisterTaskResponseProto::default())
        );
        assert!(RegisterTranslatorDataProto::verify_bidirectional_equivalence().is_ok());
        assert!(RegisterTranslatorDataProto::decode(&corrupt).is_err());
        assert_eq!(
            RegisterTranslatorDataProto::decode(&RegisterTranslatorDataProto::default().encode())
                .ok(),
            Some(RegisterTranslatorDataProto::default())
        );
        assert!(RegisterTranslatorEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(RegisterTranslatorEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            RegisterTranslatorEntryProto::decode(&RegisterTranslatorEntryProto::default().encode())
                .ok(),
            Some(RegisterTranslatorEntryProto::default())
        );
        assert!(
            RemoteProfilerSessionManagerOptionsProto::verify_bidirectional_equivalence().is_ok()
        );
        assert!(RemoteProfilerSessionManagerOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            RemoteProfilerSessionManagerOptionsProto::decode(
                &RemoteProfilerSessionManagerOptionsProto::default().encode()
            )
            .ok(),
            Some(RemoteProfilerSessionManagerOptionsProto::default())
        );
        assert!(ReplicaGroup::verify_bidirectional_equivalence().is_ok());
        assert!(ReplicaGroup::decode(&corrupt).is_err());
        assert_eq!(
            ReplicaGroup::decode(&ReplicaGroup::default().encode()).ok(),
            Some(ReplicaGroup::default())
        );
        assert!(ReportErrorToServiceRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(ReportErrorToServiceRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            ReportErrorToServiceRequestProto::decode(
                &ReportErrorToServiceRequestProto::default().encode()
            )
            .ok(),
            Some(ReportErrorToServiceRequestProto::default())
        );
        assert!(ReportErrorToServiceResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(ReportErrorToServiceResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            ReportErrorToServiceResponseProto::decode(
                &ReportErrorToServiceResponseProto::default().encode()
            )
            .ok(),
            Some(ReportErrorToServiceResponseProto::default())
        );
        assert!(ReportErrorToTaskRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(ReportErrorToTaskRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            ReportErrorToTaskRequestProto::decode(
                &ReportErrorToTaskRequestProto::default().encode()
            )
            .ok(),
            Some(ReportErrorToTaskRequestProto::default())
        );
        assert!(ReportErrorToTaskResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(ReportErrorToTaskResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            ReportErrorToTaskResponseProto::decode(
                &ReportErrorToTaskResponseProto::default().encode()
            )
            .ok(),
            Some(ReportErrorToTaskResponseProto::default())
        );
        assert!(ResetTaskRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(ResetTaskRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            ResetTaskRequestProto::decode(&ResetTaskRequestProto::default().encode()).ok(),
            Some(ResetTaskRequestProto::default())
        );
        assert!(ResetTaskResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(ResetTaskResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            ResetTaskResponseProto::decode(&ResetTaskResponseProto::default().encode()).ok(),
            Some(ResetTaskResponseProto::default())
        );
        assert!(RmsPropParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(RmsPropParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            RmsPropParametersProto::decode(&RmsPropParametersProto::default().encode()).ok(),
            Some(RmsPropParametersProto::default())
        );
        assert!(RuntimeAbiVersionProto::verify_bidirectional_equivalence().is_ok());
        assert!(RuntimeAbiVersionProto::decode(&corrupt).is_err());
        assert_eq!(
            RuntimeAbiVersionProto::decode(&RuntimeAbiVersionProto::default().encode()).ok(),
            Some(RuntimeAbiVersionProto::default())
        );
        assert!(RuntimeStateSummaryProto::verify_bidirectional_equivalence().is_ok());
        assert!(RuntimeStateSummaryProto::decode(&corrupt).is_err());
        assert_eq!(
            RuntimeStateSummaryProto::decode(&RuntimeStateSummaryProto::default().encode()).ok(),
            Some(RuntimeStateSummaryProto::default())
        );
        assert!(ScatterDimensionNumbers::verify_bidirectional_equivalence().is_ok());
        assert!(ScatterDimensionNumbers::decode(&corrupt).is_err());
        assert_eq!(
            ScatterDimensionNumbers::decode(&ScatterDimensionNumbers::default().encode()).ok(),
            Some(ScatterDimensionNumbers::default())
        );
        assert!(ShapeProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShapeProto::decode(&corrupt).is_err());
        assert_eq!(
            ShapeProto::decode(&ShapeProto::default().encode()).ok(),
            Some(ShapeProto::default())
        );
        assert!(ShapedSliceProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShapedSliceProto::decode(&corrupt).is_err());
        assert_eq!(
            ShapedSliceProto::decode(&ShapedSliceProto::default().encode()).ok(),
            Some(ShapedSliceProto::default())
        );
        assert!(ShutdownTaskRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShutdownTaskRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            ShutdownTaskRequestProto::decode(&ShutdownTaskRequestProto::default().encode()).ok(),
            Some(ShutdownTaskRequestProto::default())
        );
        assert!(ShutdownTaskResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShutdownTaskResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            ShutdownTaskResponseProto::decode(&ShutdownTaskResponseProto::default().encode()).ok(),
            Some(ShutdownTaskResponseProto::default())
        );
        assert!(SimulatedQuantizationProto::verify_bidirectional_equivalence().is_ok());
        assert!(SimulatedQuantizationProto::decode(&corrupt).is_err());
        assert_eq!(
            SimulatedQuantizationProto::decode(&SimulatedQuantizationProto::default().encode())
                .ok(),
            Some(SimulatedQuantizationProto::default())
        );
        assert!(SliceConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(SliceConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            SliceConfigProto::decode(&SliceConfigProto::default().encode()).ok(),
            Some(SliceConfigProto::default())
        );
        assert!(SliceDeviceListProto::verify_bidirectional_equivalence().is_ok());
        assert!(SliceDeviceListProto::decode(&corrupt).is_err());
        assert_eq!(
            SliceDeviceListProto::decode(&SliceDeviceListProto::default().encode()).ok(),
            Some(SliceDeviceListProto::default())
        );
        assert!(SourceTarget::verify_bidirectional_equivalence().is_ok());
        assert!(SourceTarget::decode(&corrupt).is_err());
        assert_eq!(
            SourceTarget::decode(&SourceTarget::default().encode()).ok(),
            Some(SourceTarget::default())
        );
        assert!(SparseCoreTableLayoutProto::verify_bidirectional_equivalence().is_ok());
        assert!(SparseCoreTableLayoutProto::decode(&corrupt).is_err());
        assert_eq!(
            SparseCoreTableLayoutProto::decode(&SparseCoreTableLayoutProto::default().encode())
                .ok(),
            Some(SparseCoreTableLayoutProto::default())
        );
        assert!(SparseCoreTableLayoutsProto::verify_bidirectional_equivalence().is_ok());
        assert!(SparseCoreTableLayoutsProto::decode(&corrupt).is_err());
        assert_eq!(
            SparseCoreTableLayoutsProto::decode(&SparseCoreTableLayoutsProto::default().encode())
                .ok(),
            Some(SparseCoreTableLayoutsProto::default())
        );
        assert!(SplitConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(SplitConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            SplitConfigProto::decode(&SplitConfigProto::default().encode()).ok(),
            Some(SplitConfigProto::default())
        );
        assert!(StateVariableFillWithConstantProto::verify_bidirectional_equivalence().is_ok());
        assert!(StateVariableFillWithConstantProto::decode(&corrupt).is_err());
        assert_eq!(
            StateVariableFillWithConstantProto::decode(
                &StateVariableFillWithConstantProto::default().encode()
            )
            .ok(),
            Some(StateVariableFillWithConstantProto::default())
        );
        assert!(StateVariableSpecificationProto::verify_bidirectional_equivalence().is_ok());
        assert!(StateVariableSpecificationProto::decode(&corrupt).is_err());
        assert_eq!(
            StateVariableSpecificationProto::decode(
                &StateVariableSpecificationProto::default().encode()
            )
            .ok(),
            Some(StateVariableSpecificationProto::default())
        );
        assert!(StateVariableUserDefinedProto::verify_bidirectional_equivalence().is_ok());
        assert!(StateVariableUserDefinedProto::decode(&corrupt).is_err());
        assert_eq!(
            StateVariableUserDefinedProto::decode(
                &StateVariableUserDefinedProto::default().encode()
            )
            .ok(),
            Some(StateVariableUserDefinedProto::default())
        );
        assert!(Statistic::verify_bidirectional_equivalence().is_ok());
        assert!(Statistic::decode(&corrupt).is_err());
        assert_eq!(
            Statistic::decode(&Statistic::default().encode()).ok(),
            Some(Statistic::default())
        );
        assert!(StatisticsViz::verify_bidirectional_equivalence().is_ok());
        assert!(StatisticsViz::decode(&corrupt).is_err());
        assert_eq!(
            StatisticsViz::decode(&StatisticsViz::default().encode()).ok(),
            Some(StatisticsViz::default())
        );
        assert!(StatusProto::verify_bidirectional_equivalence().is_ok());
        assert!(StatusProto::decode(&corrupt).is_err());
        assert_eq!(
            StatusProto::decode(&StatusProto::default().encode()).ok(),
            Some(StatusProto::default())
        );
        assert!(
            StochasticGradientDescentParametersProto::verify_bidirectional_equivalence().is_ok()
        );
        assert!(StochasticGradientDescentParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            StochasticGradientDescentParametersProto::decode(
                &StochasticGradientDescentParametersProto::default().encode()
            )
            .ok(),
            Some(StochasticGradientDescentParametersProto::default())
        );
        assert!(StopContinuousProfilingRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(StopContinuousProfilingRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            StopContinuousProfilingRequestProto::decode(
                &StopContinuousProfilingRequestProto::default().encode()
            )
            .ok(),
            Some(StopContinuousProfilingRequestProto::default())
        );
        assert!(StopContinuousProfilingResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(StopContinuousProfilingResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            StopContinuousProfilingResponseProto::decode(
                &StopContinuousProfilingResponseProto::default().encode()
            )
            .ok(),
            Some(StopContinuousProfilingResponseProto::default())
        );
        assert!(
            StreamExecutorExecutableAbiVersionProto::verify_bidirectional_equivalence().is_ok()
        );
        assert!(StreamExecutorExecutableAbiVersionProto::decode(&corrupt).is_err());
        assert_eq!(
            StreamExecutorExecutableAbiVersionProto::decode(
                &StreamExecutorExecutableAbiVersionProto::default().encode()
            )
            .ok(),
            Some(StreamExecutorExecutableAbiVersionProto::default())
        );
        assert!(StreamExecutorRuntimeAbiVersionProto::verify_bidirectional_equivalence().is_ok());
        assert!(StreamExecutorRuntimeAbiVersionProto::decode(&corrupt).is_err());
        assert_eq!(
            StreamExecutorRuntimeAbiVersionProto::decode(
                &StreamExecutorRuntimeAbiVersionProto::default().encode()
            )
            .ok(),
            Some(StreamExecutorRuntimeAbiVersionProto::default())
        );
        assert!(TagMetricProto::verify_bidirectional_equivalence().is_ok());
        assert!(TagMetricProto::decode(&corrupt).is_err());
        assert_eq!(
            TagMetricProto::decode(&TagMetricProto::default().encode()).ok(),
            Some(TagMetricProto::default())
        );
        assert!(TargetArgumentsProto::verify_bidirectional_equivalence().is_ok());
        assert!(TargetArgumentsProto::decode(&corrupt).is_err());
        assert_eq!(
            TargetArgumentsProto::decode(&TargetArgumentsProto::default().encode()).ok(),
            Some(TargetArgumentsProto::default())
        );
        assert!(TensorProto::verify_bidirectional_equivalence().is_ok());
        assert!(TensorProto::decode(&corrupt).is_err());
        assert_eq!(
            TensorProto::decode(&TensorProto::default().encode()).ok(),
            Some(TensorProto::default())
        );
        assert!(TensorShapeDimProto::verify_bidirectional_equivalence().is_ok());
        assert!(TensorShapeDimProto::decode(&corrupt).is_err());
        assert_eq!(
            TensorShapeDimProto::decode(&TensorShapeDimProto::default().encode()).ok(),
            Some(TensorShapeDimProto::default())
        );
        assert!(TensorShapeProto::verify_bidirectional_equivalence().is_ok());
        assert!(TensorShapeProto::decode(&corrupt).is_err());
        assert_eq!(
            TensorShapeProto::decode(&TensorShapeProto::default().encode()).ok(),
            Some(TensorShapeProto::default())
        );
        assert!(TerminateRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(TerminateRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            TerminateRequestProto::decode(&TerminateRequestProto::default().encode()).ok(),
            Some(TerminateRequestProto::default())
        );
        assert!(TerminateResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(TerminateResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            TerminateResponseProto::decode(&TerminateResponseProto::default().encode()).ok(),
            Some(TerminateResponseProto::default())
        );
        assert!(Tf2XlaConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(Tf2XlaConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            Tf2XlaConfigProto::decode(&Tf2XlaConfigProto::default().encode()).ok(),
            Some(Tf2XlaConfigProto::default())
        );
        assert!(Tf2XlaFeedProto::verify_bidirectional_equivalence().is_ok());
        assert!(Tf2XlaFeedProto::decode(&corrupt).is_err());
        assert_eq!(
            Tf2XlaFeedProto::decode(&Tf2XlaFeedProto::default().encode()).ok(),
            Some(Tf2XlaFeedProto::default())
        );
        assert!(Tf2XlaFetchProto::verify_bidirectional_equivalence().is_ok());
        assert!(Tf2XlaFetchProto::decode(&corrupt).is_err());
        assert_eq!(
            Tf2XlaFetchProto::decode(&Tf2XlaFetchProto::default().encode()).ok(),
            Some(Tf2XlaFetchProto::default())
        );
        assert!(Tf2XlaTensorIdProto::verify_bidirectional_equivalence().is_ok());
        assert!(Tf2XlaTensorIdProto::decode(&corrupt).is_err());
        assert_eq!(
            Tf2XlaTensorIdProto::decode(&Tf2XlaTensorIdProto::default().encode()).ok(),
            Some(Tf2XlaTensorIdProto::default())
        );
        assert!(Tf2XlaVariableProto::verify_bidirectional_equivalence().is_ok());
        assert!(Tf2XlaVariableProto::decode(&corrupt).is_err());
        assert_eq!(
            Tf2XlaVariableProto::decode(&Tf2XlaVariableProto::default().encode()).ok(),
            Some(Tf2XlaVariableProto::default())
        );
        assert!(TileProto::verify_bidirectional_equivalence().is_ok());
        assert!(TileProto::decode(&corrupt).is_err());
        assert_eq!(
            TileProto::decode(&TileProto::default().encode()).ok(),
            Some(TileProto::default())
        );
        assert!(ToolRequestOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(ToolRequestOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            ToolRequestOptionsProto::decode(&ToolRequestOptionsProto::default().encode()).ok(),
            Some(ToolRequestOptionsProto::default())
        );
        assert!(TopologyProto::verify_bidirectional_equivalence().is_ok());
        assert!(TopologyProto::decode(&corrupt).is_err());
        assert_eq!(
            TopologyProto::decode(&TopologyProto::default().encode()).ok(),
            Some(TopologyProto::default())
        );
        assert!(TpuBottleneckAnalysisProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuBottleneckAnalysisProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuBottleneckAnalysisProto::decode(&TpuBottleneckAnalysisProto::default().encode())
                .ok(),
            Some(TpuBottleneckAnalysisProto::default())
        );
        assert!(TpuChipConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuChipConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuChipConfigProto::decode(&TpuChipConfigProto::default().encode()).ok(),
            Some(TpuChipConfigProto::default())
        );
        assert!(TpuChipPartsProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuChipPartsProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuChipPartsProto::decode(&TpuChipPartsProto::default().encode()).ok(),
            Some(TpuChipPartsProto::default())
        );
        assert!(TpuChipVariantDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuChipVariantDescriptorProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuChipVariantDescriptorProto::decode(
                &TpuChipVariantDescriptorProto::default().encode()
            )
            .ok(),
            Some(TpuChipVariantDescriptorProto::default())
        );
        assert!(TpuCompilationEnvironmentProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCompilationEnvironmentProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCompilationEnvironmentProto::decode(
                &TpuCompilationEnvironmentProto::default().encode()
            )
            .ok(),
            Some(TpuCompilationEnvironmentProto::default())
        );
        assert!(TpuCompileMetadataArgProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCompileMetadataArgProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCompileMetadataArgProto::decode(&TpuCompileMetadataArgProto::default().encode())
                .ok(),
            Some(TpuCompileMetadataArgProto::default())
        );
        assert!(TpuCompileMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCompileMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCompileMetadataProto::decode(&TpuCompileMetadataProto::default().encode()).ok(),
            Some(TpuCompileMetadataProto::default())
        );
        assert!(TpuCompileMetadataRetvalProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCompileMetadataRetvalProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCompileMetadataRetvalProto::decode(
                &TpuCompileMetadataRetvalProto::default().encode()
            )
            .ok(),
            Some(TpuCompileMetadataRetvalProto::default())
        );
        assert!(TpuCoreHloProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCoreHloProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCoreHloProto::decode(&TpuCoreHloProto::default().encode()).ok(),
            Some(TpuCoreHloProto::default())
        );
        assert!(TpuCoreProgramAbiOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCoreProgramAbiOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCoreProgramAbiOptionsProto::decode(
                &TpuCoreProgramAbiOptionsProto::default().encode()
            )
            .ok(),
            Some(TpuCoreProgramAbiOptionsProto::default())
        );
        assert!(TpuCoreProgramAbiProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCoreProgramAbiProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCoreProgramAbiProto::decode(&TpuCoreProgramAbiProto::default().encode()).ok(),
            Some(TpuCoreProgramAbiProto::default())
        );
        assert!(TpuCoreProgramProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCoreProgramProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCoreProgramProto::decode(&TpuCoreProgramProto::default().encode()).ok(),
            Some(TpuCoreProgramProto::default())
        );
        assert!(TpuEmbeddingConfigurationProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuEmbeddingConfigurationProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuEmbeddingConfigurationProto::decode(
                &TpuEmbeddingConfigurationProto::default().encode()
            )
            .ok(),
            Some(TpuEmbeddingConfigurationProto::default())
        );
        assert!(TpuEmbeddingHostNetworkConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuEmbeddingHostNetworkConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuEmbeddingHostNetworkConfigProto::decode(
                &TpuEmbeddingHostNetworkConfigProto::default().encode()
            )
            .ok(),
            Some(TpuEmbeddingHostNetworkConfigProto::default())
        );
        assert!(TpuEmbeddingMemoryConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuEmbeddingMemoryConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuEmbeddingMemoryConfigProto::decode(
                &TpuEmbeddingMemoryConfigProto::default().encode()
            )
            .ok(),
            Some(TpuEmbeddingMemoryConfigProto::default())
        );
        assert!(TpuEmbeddingPartitionedConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuEmbeddingPartitionedConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuEmbeddingPartitionedConfigProto::decode(
                &TpuEmbeddingPartitionedConfigProto::default().encode()
            )
            .ok(),
            Some(TpuEmbeddingPartitionedConfigProto::default())
        );
        assert!(TpuExecutableInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuExecutableInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuExecutableInfoProto::decode(&TpuExecutableInfoProto::default().encode()).ok(),
            Some(TpuExecutableInfoProto::default())
        );
        assert!(TpuExecutableProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuExecutableProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuExecutableProto::decode(&TpuExecutableProto::default().encode()).ok(),
            Some(TpuExecutableProto::default())
        );
        assert!(TpuHardwareFeatureProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuHardwareFeatureProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuHardwareFeatureProto::decode(&TpuHardwareFeatureProto::default().encode()).ok(),
            Some(TpuHardwareFeatureProto::default())
        );
        assert!(TpuHostTransferInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuHostTransferInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuHostTransferInfoProto::decode(&TpuHostTransferInfoProto::default().encode()).ok(),
            Some(TpuHostTransferInfoProto::default())
        );
        assert!(TpuInputPipelineAnalysisResultProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuInputPipelineAnalysisResultProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuInputPipelineAnalysisResultProto::decode(
                &TpuInputPipelineAnalysisResultProto::default().encode()
            )
            .ok(),
            Some(TpuInputPipelineAnalysisResultProto::default())
        );
        assert!(TpuMemorySegmentProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuMemorySegmentProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuMemorySegmentProto::decode(&TpuMemorySegmentProto::default().encode()).ok(),
            Some(TpuMemorySegmentProto::default())
        );
        assert!(TpuMemorySegmentRangeProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuMemorySegmentRangeProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuMemorySegmentRangeProto::decode(&TpuMemorySegmentRangeProto::default().encode())
                .ok(),
            Some(TpuMemorySegmentRangeProto::default())
        );
        assert!(TpuMemorySegmentSetProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuMemorySegmentSetProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuMemorySegmentSetProto::decode(&TpuMemorySegmentSetProto::default().encode()).ok(),
            Some(TpuMemorySegmentSetProto::default())
        );
        assert!(TpuPlatformConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuPlatformConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuPlatformConfigProto::decode(&TpuPlatformConfigProto::default().encode()).ok(),
            Some(TpuPlatformConfigProto::default())
        );
        assert!(TpuPlatformConfigsProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuPlatformConfigsProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuPlatformConfigsProto::decode(&TpuPlatformConfigsProto::default().encode()).ok(),
            Some(TpuPlatformConfigsProto::default())
        );
        assert!(TpuRuntimeAbiProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuRuntimeAbiProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuRuntimeAbiProto::decode(&TpuRuntimeAbiProto::default().encode()).ok(),
            Some(TpuRuntimeAbiProto::default())
        );
        assert!(TpuSerializedExecutableProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuSerializedExecutableProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuSerializedExecutableProto::decode(&TpuSerializedExecutableProto::default().encode())
                .ok(),
            Some(TpuSerializedExecutableProto::default())
        );
        assert!(TpuShapeIndexProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuShapeIndexProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuShapeIndexProto::decode(&TpuShapeIndexProto::default().encode()).ok(),
            Some(TpuShapeIndexProto::default())
        );
        assert!(TpuTopologyArgsProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuTopologyArgsProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuTopologyArgsProto::decode(&TpuTopologyArgsProto::default().encode()).ok(),
            Some(TpuTopologyArgsProto::default())
        );
        assert!(TpuUpdateIndexPairProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuUpdateIndexPairProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuUpdateIndexPairProto::decode(&TpuUpdateIndexPairProto::default().encode()).ok(),
            Some(TpuUpdateIndexPairProto::default())
        );
        assert!(TritonGemmConfigsProto::verify_bidirectional_equivalence().is_ok());
        assert!(TritonGemmConfigsProto::decode(&corrupt).is_err());
        assert_eq!(
            TritonGemmConfigsProto::decode(&TritonGemmConfigsProto::default().encode()).ok(),
            Some(TritonGemmConfigsProto::default())
        );
        assert!(CholeskyOptions::verify_bidirectional_equivalence().is_ok());
        assert!(CholeskyOptions::decode(&corrupt).is_err());
        assert_eq!(
            CholeskyOptions::decode(&CholeskyOptions::default().encode()).ok(),
            Some(CholeskyOptions::default())
        );
        assert!(DynamicParameterBindingEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(DynamicParameterBindingEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            DynamicParameterBindingEntryProto::decode(
                &DynamicParameterBindingEntryProto::default().encode()
            )
            .ok(),
            Some(DynamicParameterBindingEntryProto::default())
        );
        assert!(DynamicParameterBindingProto::verify_bidirectional_equivalence().is_ok());
        assert!(DynamicParameterBindingProto::decode(&corrupt).is_err());
        assert_eq!(
            DynamicParameterBindingProto::decode(&DynamicParameterBindingProto::default().encode())
                .ok(),
            Some(DynamicParameterBindingProto::default())
        );
        assert!(TriangularSolveOptions::verify_bidirectional_equivalence().is_ok());
        assert!(TriangularSolveOptions::decode(&corrupt).is_err());
        assert_eq!(
            TriangularSolveOptions::decode(&TriangularSolveOptions::default().encode()).ok(),
            Some(TriangularSolveOptions::default())
        );
        assert!(TryGetKeyValueRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(TryGetKeyValueRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            TryGetKeyValueRequestProto::decode(&TryGetKeyValueRequestProto::default().encode())
                .ok(),
            Some(TryGetKeyValueRequestProto::default())
        );
        assert!(TryGetKeyValueResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(TryGetKeyValueResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            TryGetKeyValueResponseProto::decode(&TryGetKeyValueResponseProto::default().encode())
                .ok(),
            Some(TryGetKeyValueResponseProto::default())
        );
        assert!(UserDefinedProgramParametersProto::verify_bidirectional_equivalence().is_ok());
        assert!(UserDefinedProgramParametersProto::decode(&corrupt).is_err());
        assert_eq!(
            UserDefinedProgramParametersProto::decode(
                &UserDefinedProgramParametersProto::default().encode()
            )
            .ok(),
            Some(UserDefinedProgramParametersProto::default())
        );
        assert!(WaitForAllTasksRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(WaitForAllTasksRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            WaitForAllTasksRequestProto::decode(&WaitForAllTasksRequestProto::default().encode())
                .ok(),
            Some(WaitForAllTasksRequestProto::default())
        );
        assert!(WaitForAllTasksResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(WaitForAllTasksResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            WaitForAllTasksResponseProto::decode(&WaitForAllTasksResponseProto::default().encode())
                .ok(),
            Some(WaitForAllTasksResponseProto::default())
        );
        assert!(WatchJobStateRequestProto::verify_bidirectional_equivalence().is_ok());
        assert!(WatchJobStateRequestProto::decode(&corrupt).is_err());
        assert_eq!(
            WatchJobStateRequestProto::decode(&WatchJobStateRequestProto::default().encode()).ok(),
            Some(WatchJobStateRequestProto::default())
        );
        assert!(WatchJobStateResponseProto::verify_bidirectional_equivalence().is_ok());
        assert!(WatchJobStateResponseProto::decode(&corrupt).is_err());
        assert_eq!(
            WatchJobStateResponseProto::decode(&WatchJobStateResponseProto::default().encode())
                .ok(),
            Some(WatchJobStateResponseProto::default())
        );
        assert!(Window::verify_bidirectional_equivalence().is_ok());
        assert!(Window::decode(&corrupt).is_err());
        assert_eq!(
            Window::decode(&Window::default().encode()).ok(),
            Some(Window::default())
        );
        assert!(WindowDimension::verify_bidirectional_equivalence().is_ok());
        assert!(WindowDimension::decode(&corrupt).is_err());
        assert_eq!(
            WindowDimension::decode(&WindowDimension::default().encode()).ok(),
            Some(WindowDimension::default())
        );
        assert!(XEventMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(XEventMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            XEventMetadataProto::decode(&XEventMetadataProto::default().encode()).ok(),
            Some(XEventMetadataProto::default())
        );
        assert!(XEventProto::verify_bidirectional_equivalence().is_ok());
        assert!(XEventProto::decode(&corrupt).is_err());
        assert_eq!(
            XEventProto::decode(&XEventProto::default().encode()).ok(),
            Some(XEventProto::default())
        );
        assert!(XLineProto::verify_bidirectional_equivalence().is_ok());
        assert!(XLineProto::decode(&corrupt).is_err());
        assert_eq!(
            XLineProto::decode(&XLineProto::default().encode()).ok(),
            Some(XLineProto::default())
        );
        assert!(XPlaneProto::verify_bidirectional_equivalence().is_ok());
        assert!(XPlaneProto::decode(&corrupt).is_err());
        assert_eq!(
            XPlaneProto::decode(&XPlaneProto::default().encode()).ok(),
            Some(XPlaneProto::default())
        );
        assert!(XSpaceProto::verify_bidirectional_equivalence().is_ok());
        assert!(XSpaceProto::decode(&corrupt).is_err());
        assert_eq!(
            XSpaceProto::decode(&XSpaceProto::default().encode()).ok(),
            Some(XSpaceProto::default())
        );
        assert!(XStatMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(XStatMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            XStatMetadataProto::decode(&XStatMetadataProto::default().encode()).ok(),
            Some(XStatMetadataProto::default())
        );
        assert!(XStatProto::verify_bidirectional_equivalence().is_ok());
        assert!(XStatProto::decode(&corrupt).is_err());
        assert_eq!(
            XStatProto::decode(&XStatProto::default().encode()).ok(),
            Some(XStatProto::default())
        );
        assert!(XlaArgumentProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaArgumentProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaArgumentProto::decode(&XlaArgumentProto::default().encode()).ok(),
            Some(XlaArgumentProto::default())
        );
        assert!(XlaArgumentShapeProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaArgumentShapeProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaArgumentShapeProto::decode(&XlaArgumentShapeProto::default().encode()).ok(),
            Some(XlaArgumentShapeProto::default())
        );
        assert!(XlaCompilationResultProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaCompilationResultProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaCompilationResultProto::decode(&XlaCompilationResultProto::default().encode()).ok(),
            Some(XlaCompilationResultProto::default())
        );
        assert!(XlaExecutionOptionsProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaExecutionOptionsProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaExecutionOptionsProto::decode(&XlaExecutionOptionsProto::default().encode()).ok(),
            Some(XlaExecutionOptionsProto::default())
        );

        // Verify newly added diagnostic & Mosaic protobuf structs + Riegeli container helpers
        assert!(CoreDumpMachineInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreDumpMachineInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreDumpMachineInfoProto::decode(&CoreDumpMachineInfoProto::default().encode()).ok(),
            Some(CoreDumpMachineInfoProto::default())
        );
        assert!(CoreDumpTimestampInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreDumpTimestampInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreDumpTimestampInfoProto::decode(&CoreDumpTimestampInfoProto::default().encode())
                .ok(),
            Some(CoreDumpTimestampInfoProto::default())
        );
        assert!(CoreDumpHardwareInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreDumpHardwareInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreDumpHardwareInfoProto::decode(&CoreDumpHardwareInfoProto::default().encode()).ok(),
            Some(CoreDumpHardwareInfoProto::default())
        );
        assert!(CoreDumpRegisterDumpProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreDumpRegisterDumpProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreDumpRegisterDumpProto::decode(&CoreDumpRegisterDumpProto::default().encode()).ok(),
            Some(CoreDumpRegisterDumpProto::default())
        );
        assert!(CoreStateDumpProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreStateDumpProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreStateDumpProto::decode(&CoreStateDumpProto::default().encode()).ok(),
            Some(CoreStateDumpProto::default())
        );
        assert!(CoreDumpAnalyzerReportProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreDumpAnalyzerReportProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreDumpAnalyzerReportProto::decode(&CoreDumpAnalyzerReportProto::default().encode())
                .ok(),
            Some(CoreDumpAnalyzerReportProto::default())
        );
        assert!(CoreDumpProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreDumpProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreDumpProto::decode(&CoreDumpProto::default().encode()).ok(),
            Some(CoreDumpProto::default())
        );
        assert!(ActiveSyncFlagWaitProto::verify_bidirectional_equivalence().is_ok());
        assert!(ActiveSyncFlagWaitProto::decode(&corrupt).is_err());
        assert_eq!(
            ActiveSyncFlagWaitProto::decode(&ActiveSyncFlagWaitProto::default().encode()).ok(),
            Some(ActiveSyncFlagWaitProto::default())
        );
        assert!(CoreStallSummaryProto::verify_bidirectional_equivalence().is_ok());
        assert!(CoreStallSummaryProto::decode(&corrupt).is_err());
        assert_eq!(
            CoreStallSummaryProto::decode(&CoreStallSummaryProto::default().encode()).ok(),
            Some(CoreStallSummaryProto::default())
        );
        assert!(AllCoreStateSummariesProto::verify_bidirectional_equivalence().is_ok());
        assert!(AllCoreStateSummariesProto::decode(&corrupt).is_err());
        assert_eq!(
            AllCoreStateSummariesProto::decode(&AllCoreStateSummariesProto::default().encode())
                .ok(),
            Some(AllCoreStateSummariesProto::default())
        );
        assert!(CompilerSectionAllocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilerSectionAllocationProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilerSectionAllocationProto::decode(
                &CompilerSectionAllocationProto::default().encode()
            )
            .ok(),
            Some(CompilerSectionAllocationProto::default())
        );
        assert!(CompilerSymbolTableEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilerSymbolTableEntryProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilerSymbolTableEntryProto::decode(
                &CompilerSymbolTableEntryProto::default().encode()
            )
            .ok(),
            Some(CompilerSymbolTableEntryProto::default())
        );
        assert!(HloToLloMappingProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloToLloMappingProto::decode(&corrupt).is_err());
        assert_eq!(
            HloToLloMappingProto::decode(&HloToLloMappingProto::default().encode()).ok(),
            Some(HloToLloMappingProto::default())
        );
        assert!(CompilerMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(CompilerMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            CompilerMetadataProto::decode(&CompilerMetadataProto::default().encode()).ok(),
            Some(CompilerMetadataProto::default())
        );
        assert!(MosaicDmaDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(MosaicDmaDescriptorProto::decode(&corrupt).is_err());
        assert_eq!(
            MosaicDmaDescriptorProto::decode(&MosaicDmaDescriptorProto::default().encode()).ok(),
            Some(MosaicDmaDescriptorProto::default())
        );
        assert!(MosaicScratchpadLayoutProto::verify_bidirectional_equivalence().is_ok());
        assert!(MosaicScratchpadLayoutProto::decode(&corrupt).is_err());
        assert_eq!(
            MosaicScratchpadLayoutProto::decode(&MosaicScratchpadLayoutProto::default().encode())
                .ok(),
            Some(MosaicScratchpadLayoutProto::default())
        );
        assert!(MosaicBackendConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(MosaicBackendConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            MosaicBackendConfigProto::decode(&MosaicBackendConfigProto::default().encode()).ok(),
            Some(MosaicBackendConfigProto::default())
        );

        assert!(HloInstructionProfileProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloInstructionProfileProto::decode(&corrupt).is_err());
        assert_eq!(
            HloInstructionProfileProto::decode(&HloInstructionProfileProto::default().encode())
                .ok(),
            Some(HloInstructionProfileProto::default())
        );
        assert!(HloComputationProfileProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloComputationProfileProto::decode(&corrupt).is_err());
        assert_eq!(
            HloComputationProfileProto::decode(&HloComputationProfileProto::default().encode())
                .ok(),
            Some(HloComputationProfileProto::default())
        );
        assert!(HloProfilePrinterDataProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloProfilePrinterDataProto::decode(&corrupt).is_err());
        assert_eq!(
            HloProfilePrinterDataProto::decode(&HloProfilePrinterDataProto::default().encode())
                .ok(),
            Some(HloProfilePrinterDataProto::default())
        );
        assert!(XlaFfiHandlerRegistrationProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaFfiHandlerRegistrationProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaFfiHandlerRegistrationProto::decode(
                &XlaFfiHandlerRegistrationProto::default().encode()
            )
            .ok(),
            Some(XlaFfiHandlerRegistrationProto::default())
        );
        assert!(XlaFfiCallFrameMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaFfiCallFrameMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaFfiCallFrameMetadataProto::decode(&XlaFfiCallFrameMetadataProto::default().encode())
                .ok(),
            Some(XlaFfiCallFrameMetadataProto::default())
        );
        assert!(JellyfishHostTransferProto::verify_bidirectional_equivalence().is_ok());
        assert!(JellyfishHostTransferProto::decode(&corrupt).is_err());
        assert_eq!(
            JellyfishHostTransferProto::decode(&JellyfishHostTransferProto::default().encode())
                .ok(),
            Some(JellyfishHostTransferProto::default())
        );
        assert!(JellyfishOverlayConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(JellyfishOverlayConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            JellyfishOverlayConfigProto::decode(&JellyfishOverlayConfigProto::default().encode())
                .ok(),
            Some(JellyfishOverlayConfigProto::default())
        );
        assert!(JellyfishCoreProgramProto::verify_bidirectional_equivalence().is_ok());
        assert!(JellyfishCoreProgramProto::decode(&corrupt).is_err());
        assert_eq!(
            JellyfishCoreProgramProto::decode(&JellyfishCoreProgramProto::default().encode()).ok(),
            Some(JellyfishCoreProgramProto::default())
        );
        assert!(JellyfishExecutableProto::verify_bidirectional_equivalence().is_ok());
        assert!(JellyfishExecutableProto::decode(&corrupt).is_err());
        assert_eq!(
            JellyfishExecutableProto::decode(&JellyfishExecutableProto::default().encode()).ok(),
            Some(JellyfishExecutableProto::default())
        );
        assert!(SparseCoreDmaDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(SparseCoreDmaDescriptorProto::decode(&corrupt).is_err());
        assert_eq!(
            SparseCoreDmaDescriptorProto::decode(&SparseCoreDmaDescriptorProto::default().encode())
                .ok(),
            Some(SparseCoreDmaDescriptorProto::default())
        );
        assert!(BarnaCoreChannelConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(BarnaCoreChannelConfigProto::decode(&corrupt).is_err());
        assert_eq!(
            BarnaCoreChannelConfigProto::decode(&BarnaCoreChannelConfigProto::default().encode())
                .ok(),
            Some(BarnaCoreChannelConfigProto::default())
        );
        assert!(SparseCoreProgramProto::verify_bidirectional_equivalence().is_ok());
        assert!(SparseCoreProgramProto::decode(&corrupt).is_err());
        assert_eq!(
            SparseCoreProgramProto::decode(&SparseCoreProgramProto::default().encode()).ok(),
            Some(SparseCoreProgramProto::default())
        );
        assert!(BarnaCoreExecutableProto::verify_bidirectional_equivalence().is_ok());
        assert!(BarnaCoreExecutableProto::decode(&corrupt).is_err());
        assert_eq!(
            BarnaCoreExecutableProto::decode(&BarnaCoreExecutableProto::default().encode()).ok(),
            Some(BarnaCoreExecutableProto::default())
        );
        assert!(HloModuleFingerprintProto::verify_bidirectional_equivalence().is_ok());
        assert!(HloModuleFingerprintProto::decode(&corrupt).is_err());
        assert_eq!(
            HloModuleFingerprintProto::decode(&HloModuleFingerprintProto::default().encode()).ok(),
            Some(HloModuleFingerprintProto::default())
        );
        assert!(TpuCompilerEnvironmentFingerprintProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCompilerEnvironmentFingerprintProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCompilerEnvironmentFingerprintProto::decode(
                &TpuCompilerEnvironmentFingerprintProto::default().encode()
            )
            .ok(),
            Some(TpuCompilerEnvironmentFingerprintProto::default())
        );
        assert!(TpuCompilationFingerprintProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuCompilationFingerprintProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuCompilationFingerprintProto::decode(
                &TpuCompilationFingerprintProto::default().encode()
            )
            .ok(),
            Some(TpuCompilationFingerprintProto::default())
        );
        assert!(TpuPowerStateProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuPowerStateProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuPowerStateProto::decode(&TpuPowerStateProto::default().encode()).ok(),
            Some(TpuPowerStateProto::default())
        );
        assert!(ChipThermalThrottleSummaryProto::verify_bidirectional_equivalence().is_ok());
        assert!(ChipThermalThrottleSummaryProto::decode(&corrupt).is_err());
        assert_eq!(
            ChipThermalThrottleSummaryProto::decode(
                &ChipThermalThrottleSummaryProto::default().encode()
            )
            .ok(),
            Some(ChipThermalThrottleSummaryProto::default())
        );
        assert!(FrequencyScalingTelemetryProto::verify_bidirectional_equivalence().is_ok());
        assert!(FrequencyScalingTelemetryProto::decode(&corrupt).is_err());
        assert_eq!(
            FrequencyScalingTelemetryProto::decode(
                &FrequencyScalingTelemetryProto::default().encode()
            )
            .ok(),
            Some(FrequencyScalingTelemetryProto::default())
        );
        assert!(VhloDialectVersionProto::verify_bidirectional_equivalence().is_ok());
        assert!(VhloDialectVersionProto::decode(&corrupt).is_err());
        assert_eq!(
            VhloDialectVersionProto::decode(&VhloDialectVersionProto::default().encode()).ok(),
            Some(VhloDialectVersionProto::default())
        );
        assert!(StableHloPortableArtifactMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(StableHloPortableArtifactMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            StableHloPortableArtifactMetadataProto::decode(
                &StableHloPortableArtifactMetadataProto::default().encode()
            )
            .ok(),
            Some(StableHloPortableArtifactMetadataProto::default())
        );
        assert!(ShardyMeshAxisProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShardyMeshAxisProto::decode(&corrupt).is_err());
        assert_eq!(
            ShardyMeshAxisProto::decode(&ShardyMeshAxisProto::default().encode()).ok(),
            Some(ShardyMeshAxisProto::default())
        );
        assert!(ShardySubAxisInfoProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShardySubAxisInfoProto::decode(&corrupt).is_err());
        assert_eq!(
            ShardySubAxisInfoProto::decode(&ShardySubAxisInfoProto::default().encode()).ok(),
            Some(ShardySubAxisInfoProto::default())
        );
        assert!(ShardyAxisRefProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShardyAxisRefProto::decode(&corrupt).is_err());
        assert_eq!(
            ShardyAxisRefProto::decode(&ShardyAxisRefProto::default().encode()).ok(),
            Some(ShardyAxisRefProto::default())
        );
        assert!(ShardyDimensionShardingProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShardyDimensionShardingProto::decode(&corrupt).is_err());
        assert_eq!(
            ShardyDimensionShardingProto::decode(&ShardyDimensionShardingProto::default().encode())
                .ok(),
            Some(ShardyDimensionShardingProto::default())
        );
        assert!(ShardyTensorShardingProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShardyTensorShardingProto::decode(&corrupt).is_err());
        assert_eq!(
            ShardyTensorShardingProto::decode(&ShardyTensorShardingProto::default().encode()).ok(),
            Some(ShardyTensorShardingProto::default())
        );
        assert!(ShardyMeshProto::verify_bidirectional_equivalence().is_ok());
        assert!(ShardyMeshProto::decode(&corrupt).is_err());
        assert_eq!(
            ShardyMeshProto::decode(&ShardyMeshProto::default().encode()).ok(),
            Some(ShardyMeshProto::default())
        );
        assert!(XlaFfiTypeIdRegistrationProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaFfiTypeIdRegistrationProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaFfiTypeIdRegistrationProto::decode(
                &XlaFfiTypeIdRegistrationProto::default().encode()
            )
            .ok(),
            Some(XlaFfiTypeIdRegistrationProto::default())
        );
        assert!(XlaFfiExecutionContextMetadataProto::verify_bidirectional_equivalence().is_ok());
        assert!(XlaFfiExecutionContextMetadataProto::decode(&corrupt).is_err());
        assert_eq!(
            XlaFfiExecutionContextMetadataProto::decode(
                &XlaFfiExecutionContextMetadataProto::default().encode()
            )
            .ok(),
            Some(XlaFfiExecutionContextMetadataProto::default())
        );
        assert!(TpuzActiveHloExecutionTraceProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuzActiveHloExecutionTraceProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuzActiveHloExecutionTraceProto::decode(
                &TpuzActiveHloExecutionTraceProto::default().encode()
            )
            .ok(),
            Some(TpuzActiveHloExecutionTraceProto::default())
        );
        assert!(TpuzLiveBufferAllocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuzLiveBufferAllocationProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuzLiveBufferAllocationProto::decode(
                &TpuzLiveBufferAllocationProto::default().encode()
            )
            .ok(),
            Some(TpuzLiveBufferAllocationProto::default())
        );
        assert!(TpuzCoreTelemetrySnapshotProto::verify_bidirectional_equivalence().is_ok());
        assert!(TpuzCoreTelemetrySnapshotProto::decode(&corrupt).is_err());
        assert_eq!(
            TpuzCoreTelemetrySnapshotProto::decode(
                &TpuzCoreTelemetrySnapshotProto::default().encode()
            )
            .ok(),
            Some(TpuzCoreTelemetrySnapshotProto::default())
        );

        let sample_meta = CompilerMetadataProto {
            program_id: 42,
            replica_count: 2,
            num_cores_per_replica: 4,
            is_haltless: true,
            executable_size: 4096,
            optimized_hlo_instruction_count: 18,
            module_name: String::from("jax_fn"),
            executable_fingerprint: String::from("0123456789abcdef0123456789abcdef"),
            section_allocations: Vec::new(),
            symbol_table: Vec::new(),
            hlo_to_llo_mappings: Vec::new(),
        };
        let riegeli_wrapped = encode_riegeli_record_container(&sample_meta.encode());
        let riegeli_unwrapped = decode_riegeli_record_container(&riegeli_wrapped);
        assert!(riegeli_unwrapped.is_ok());
        if let Ok(bytes) = riegeli_unwrapped {
            assert_eq!(
                CompilerMetadataProto::decode(&bytes).ok(),
                Some(sample_meta)
            );
        }

        // Exercise inherent .schema_spec() and .emit_proto_schema()
        assert!(TileProto::schema_spec().is_ok());
        assert!(TileProto::emit_proto_schema().is_ok());

        // Test drift detection when a Rust struct omits a .proto field or emits an incomplete schema
        assert!(IncompleteTileProto::verify_bidirectional_equivalence().is_err());
        if let Ok(tile_spec) = TileProto::schema_spec() {
            assert!(
                verify_bidirectional_proto_struct_equivalence_with_spec(
                    "xla.TileProto",
                    roundtrip_proto_wire_bytes::<TileProto>,
                    &tile_spec,
                    ""
                )
                .is_err()
            );
        }
        assert!(extract_proto_message_spec("nonexistent.Proto").is_err());
        assert!(emit_proto_message_schema("nonexistent.Proto").is_err());

        // Test unknown wire field preservation across FFI
        if let Ok(desc) = CompileOptionsProto::descriptor() {
            let base = CompileOptionsProto {
                profile_version: 9,
                ..CompileOptionsProto::default()
            }
            .encode();
            let mut with_unknown = base.clone();
            // Tag 99, wire type 0 (varint) = (99 << 3) | 0 = 792 -> [0x98, 0x06, 0x2A]
            with_unknown.extend_from_slice(&[0x98, 0x06, 0x2A]);
            // Tag 100, wire type 1 (64-bit) = (100 << 3) | 1 = 801 -> [0xA1, 0x06, 1..=8]
            with_unknown.extend_from_slice(&[0xA1, 0x06, 1, 2, 3, 4, 5, 6, 7, 8]);
            // Tag 101, wire type 2 (len-delimited) = (101 << 3) | 2 = 810 -> [0xAA, 0x06, 2, 0xEE, 0xFF]
            with_unknown.extend_from_slice(&[0xAA, 0x06, 2, 0xEE, 0xFF]);
            // Tag 102, wire type 5 (32-bit) = (102 << 3) | 5 = 821 -> [0xB5, 0x06, 1, 2, 3, 4]
            with_unknown.extend_from_slice(&[0xB5, 0x06, 1, 2, 3, 4]);
            let preserved = preserve_unknown_wire_fields(&with_unknown, &base, &desc);
            assert_eq!(preserved, with_unknown);

            // Truncated/invalid wire records are safely ignored
            assert_eq!(
                preserve_unknown_wire_fields(
                    &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                    &base,
                    &desc
                ),
                base
            );
            assert_eq!(
                preserve_unknown_wire_fields(&[0x98, 0x06], &base, &desc),
                base
            );
            assert_eq!(
                preserve_unknown_wire_fields(&[0xAA, 0x06], &base, &desc),
                base
            );
            assert_eq!(
                preserve_unknown_wire_fields(&[0xAF, 0x06], &base, &desc),
                base
            );
        }

        // Test format_proto_kind and emit_proto_schema_from_spec edge cases
        for kind in [
            prost_reflect::Kind::Sint32,
            prost_reflect::Kind::Sint64,
            prost_reflect::Kind::Fixed32,
            prost_reflect::Kind::Sfixed32,
            prost_reflect::Kind::Sfixed64,
        ] {
            assert_ne!(format_proto_kind(&kind), "");
            let _ = build_populated_field_value(&kind, 1, 0);
        }
        let no_pkg_spec = ProtoMessageSpec {
            full_name: String::from("RootMsg"),
            package_name: String::new(),
            message_name: String::from("RootMsg"),
            fields: vec![ProtoFieldSpec {
                name: String::from("x"),
                tag: 1,
                cardinality: ProtoFieldCardinality::Optional,
                type_name: String::from("int32"),
            }],
        };
        let emitted_no_pkg = emit_proto_schema_from_spec(&no_pkg_spec);
        assert!(emitted_no_pkg.contains("message RootMsg"));
        assert!(!emitted_no_pkg.contains("package "));
        assert_eq!(no_pkg_spec.clone(), no_pkg_spec);
        assert_ne!(format!("{no_pkg_spec:?}"), "");

        assert_ne!(GENERATED_PROTO_RUST_FILES, &[] as &[&str]);
        let sample_payload = b"riegeli_compressed_test_payload_bytes_1234567890";
        for mode in [
            RiegeliChunkCompression::Uncompressed,
            RiegeliChunkCompression::Brotli,
            RiegeliChunkCompression::Zstd,
            RiegeliChunkCompression::Metadata,
        ] {
            let wrapped = RiegeliRecordCodec::wrap_record_compressed(sample_payload, mode);
            let unwrapped = RiegeliRecordCodec::unwrap_record_decompressed(&wrapped);
            assert_eq!(unwrapped.ok(), Some((mode, sample_payload.to_vec())));
            assert_eq!(mode.clone(), mode);
            assert_ne!(format!("{mode:?}"), "");
        }
        assert_eq!(
            RiegeliRecordCodec::unwrap_record_decompressed(sample_payload).ok(),
            Some((
                RiegeliChunkCompression::Uncompressed,
                sample_payload.to_vec()
            ))
        );
        let mut corrupt_brotli = RiegeliRecordCodec::wrap_record_compressed(
            b"bad_brotli",
            RiegeliChunkCompression::Brotli,
        );
        corrupt_brotli.truncate(64);
        corrupt_brotli.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        let h_hash = riegeli_highwayhash64(corrupt_brotli.get(..48).unwrap_or_default());
        let p_hash = riegeli_highwayhash64(corrupt_brotli.get(64..).unwrap_or_default());
        if let Some(h_slice) = corrupt_brotli.get_mut(48..56) {
            h_slice.copy_from_slice(&h_hash.to_le_bytes());
        }
        if let Some(p_slice) = corrupt_brotli.get_mut(56..64) {
            p_slice.copy_from_slice(&p_hash.to_le_bytes());
        }
        assert!(RiegeliRecordCodec::unwrap_record_decompressed(&corrupt_brotli).is_err());
        let mut bad_hdr = RiegeliRecordCodec::wrap_record(b"x");
        if let Some(b) = bad_hdr.get_mut(48) {
            *b ^= 0xFF;
        }
        assert!(RiegeliRecordCodec::unwrap_record_decompressed(&bad_hdr).is_err());

        let env_proto = TpuCompilationEnvironmentProto::from_cli_flags("");
        assert_eq!(
            env_proto.get_field_as_string("xla_tpu_enable_windowed_einsum"),
            "true"
        );
        assert_eq!(
            env_proto.get_field_as_string("xla_tpu_enable_llo_software_pipelining"),
            "true"
        );
        assert_eq!(
            env_proto.get_field_as_string("xla_tpu_llo_modulo_scheduler_ii_limit"),
            "16"
        );
        assert_eq!(
            env_proto.get_field_as_string("xla_tpu_enable_rs_forward_pipelining"),
            "true"
        );
        assert_eq!(env_proto.get_field_as_string("256"), "true");
        if let Some(spec) = TpuCompilationEnvironmentProto::lookup_jellyfish_field_spec("34") {
            assert_eq!(spec.name, "xla_tpu_enable_llo_software_pipelining");
            assert_eq!(spec.clone(), *spec);
            assert_ne!(format!("{spec:?}"), "");
        }

        assert!(StackFrameFileLocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(StackFrameEntryProto::verify_bidirectional_equivalence().is_ok());
        assert!(StackFrameIndexProto::verify_bidirectional_equivalence().is_ok());
        assert!(LogicalBufferLocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(LogicalBufferProto::verify_bidirectional_equivalence().is_ok());
        assert!(HeapSimulatorTraceEventProto::verify_bidirectional_equivalence().is_ok());
        assert!(HeapSimulatorTraceProto::verify_bidirectional_equivalence().is_ok());
        assert!(BufferAllocationAssignedProto::verify_bidirectional_equivalence().is_ok());
        assert!(IciRouteHopProto::verify_bidirectional_equivalence().is_ok());
        assert!(IciRoutingTableProto::verify_bidirectional_equivalence().is_ok());
        assert!(VliwBundleEncodingSpecProto::verify_bidirectional_equivalence().is_ok());
        assert!(DmaDescriptorRingConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(GasketPageTableConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(GasketIoctlConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(VfioIommuMapConfigProto::verify_bidirectional_equivalence().is_ok());
        assert!(PcieBarRegionProto::verify_bidirectional_equivalence().is_ok());
        assert!(PcieBarLayoutProto::verify_bidirectional_equivalence().is_ok());
        assert!(LloScheduledInstructionProto::verify_bidirectional_equivalence().is_ok());
        assert!(LloBundleScheduleProto::verify_bidirectional_equivalence().is_ok());
        assert!(LloSpillSlotProto::verify_bidirectional_equivalence().is_ok());
        assert!(LloLiveIntervalProto::verify_bidirectional_equivalence().is_ok());
        assert!(LloRegisterAllocationProto::verify_bidirectional_equivalence().is_ok());
        assert!(LloModuloStageProto::verify_bidirectional_equivalence().is_ok());
        assert!(LloSoftwarePipelineScheduleProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegascaleChannelSliceDescriptorProto::verify_bidirectional_equivalence().is_ok());
        assert!(MegascaleCollectiveScheduleProto::verify_bidirectional_equivalence().is_ok());
        for m in [
            IciRouteHopProto::default().encode(),
            IciRoutingTableProto::default().encode(),
            VliwBundleEncodingSpecProto::default().encode(),
            DmaDescriptorRingConfigProto::default().encode(),
            GasketPageTableConfigProto::default().encode(),
            GasketIoctlConfigProto::default().encode(),
            VfioIommuMapConfigProto::default().encode(),
            PcieBarRegionProto::default().encode(),
            PcieBarLayoutProto::default().encode(),
            LloScheduledInstructionProto::default().encode(),
            LloBundleScheduleProto::default().encode(),
            LloSpillSlotProto::default().encode(),
            LloLiveIntervalProto::default().encode(),
            LloRegisterAllocationProto::default().encode(),
            LloModuloStageProto::default().encode(),
            LloSoftwarePipelineScheduleProto::default().encode(),
            MegascaleChannelSliceDescriptorProto::default().encode(),
            MegascaleCollectiveScheduleProto::default().encode(),
        ] {
            assert_eq!(m.len(), 0);
        }
        assert!(IciRouteHopProto::decode(&[]).is_ok());
        assert!(IciRoutingTableProto::decode(&[]).is_ok());
        assert!(VliwBundleEncodingSpecProto::decode(&[]).is_ok());
        assert!(DmaDescriptorRingConfigProto::decode(&[]).is_ok());
        assert!(GasketPageTableConfigProto::decode(&[]).is_ok());
        assert!(GasketIoctlConfigProto::decode(&[]).is_ok());
        assert!(VfioIommuMapConfigProto::decode(&[]).is_ok());
        assert!(PcieBarRegionProto::decode(&[]).is_ok());
        assert!(PcieBarLayoutProto::decode(&[]).is_ok());
        assert!(LloScheduledInstructionProto::decode(&[]).is_ok());
        assert!(LloBundleScheduleProto::decode(&[]).is_ok());
        assert!(LloSpillSlotProto::decode(&[]).is_ok());
        assert!(LloLiveIntervalProto::decode(&[]).is_ok());
        assert!(LloRegisterAllocationProto::decode(&[]).is_ok());
        assert!(LloModuloStageProto::decode(&[]).is_ok());
        assert!(LloSoftwarePipelineScheduleProto::decode(&[]).is_ok());
        assert!(MegascaleChannelSliceDescriptorProto::decode(&[]).is_ok());
        assert!(MegascaleCollectiveScheduleProto::decode(&[]).is_ok());
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
                    ..LayoutProto::default()
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
