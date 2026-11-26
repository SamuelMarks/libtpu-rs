//! Single-source-of-truth Protocol Buffers (`prost-build` + `protox` + `prost-reflect`) schemas
//! for `libtpu-rs`, compiled directly from all 738 reconstructed `libtpu.so` `protodesc_cold` `.proto` files.
//!
//! Organized into focused domain submodules re-exporting `prost-build` generated structs:
//! - [`descriptor`]: Embedded `FileDescriptorSet`, `generated_prost_schemas`, [`ProtoSchemaMessage`], `Riegeli`, and `HighwayHash64`.
//! - [`xla_data`]: `xla.ShapeProto`, `LayoutProto`, `TileProto`, `ProgramShapeProto`, `OpSharding`, `DeviceAssignmentProto`.
//! - [`hlo`]: `xla.HloInstructionProto`, `HloComputationProto`, `HloModuleProto`, `BufferAssignmentProto`.
//! - [`pjrt`]: `xla.CompileOptionsProto`, `ExecuteOptionsProto`, `PjRtTopologyDescriptionProto`, `PjRtRuntimeAbiVersionProto`.
//! - [`tpu`]: `xla.jellyfish.TpuCompilationEnvironment`, `TPUEmbeddingConfiguration`, `CompilerMetadata`, `CoreDump`.
//! - [`deepsea`]: `xla.IsaProgramProto`, `DeepseaExecutableProto`, `TpuExecutableProto`, `tpu.TpuCoreProgramProto`, GCSR & `BarnaCore` models.
//! - [`megascale`]: `xla.megascale.runtime.*`, `tensorflow.StatusProto`, and `CoordinationService` RPC messages.
//! - [`xplane`]: `tensorflow.profiler.XSpace`, `XPlane`, `XLine`, `XEvent`, `XStat`, and `xprof` profiler schemas.

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
pub use prost_types::Any as ProtobufAny;
pub use tpu::*;
pub use xla_data::*;
pub use xplane::*;

#[cfg(test)]
#[allow(clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::config::TpuInitConfig;
    use crate::types::TpuGeneration;
    use proptest::prelude::*;
    use prost::Message as _;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    type TestResult = crate::error::LibTpuResult<()>;

    fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    collect_rs_files(&p, out);
                } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    out.push(p);
                }
            }
        }
    }

    fn collect_proto_files(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    collect_proto_files(&p, out);
                } else if p.extension().and_then(|s| s.to_str()) == Some("proto") {
                    out.push(p);
                }
            }
        }
    }

    #[test]
    fn test_phase1_1_all_738_reconstructed_protos_roundtrip_protox() -> TestResult {
        let expected_proto_count = crate::elf::parse_golden_proto_descriptors(include_str!(
            "../../../docs/reverse_engineering/golden_proto_descriptors.json"
        ))?
        .proto_files_count;
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let proto_root = manifest_dir.join("../../docs/reverse_engineering/protos");
        let mut proto_paths = Vec::new();
        collect_proto_files(&proto_root, &mut proto_paths);
        proto_paths.sort();
        assert_eq!(proto_paths.len(), expected_proto_count);

        let recompiled_fds = protox::compile(&proto_paths, [&proto_root]).map_err(|e| {
            crate::error::LibTpuError::BinaryFormatError {
                reason: e.to_string(),
            }
        })?;
        assert_eq!(recompiled_fds.file.len(), expected_proto_count);

        let embedded_fds = prost_types::FileDescriptorSet::decode(
            EMBEDDED_FILE_DESCRIPTOR_SET_BYTES,
        )
        .map_err(|e| crate::error::LibTpuError::BinaryFormatError {
            reason: e.to_string(),
        })?;
        assert_eq!(embedded_fds.file.len(), expected_proto_count);

        for (lhs, rhs) in recompiled_fds.file.iter().zip(embedded_fds.file.iter()) {
            assert_eq!(lhs.name, rhs.name);
            assert_eq!(lhs.package, rhs.package);
            assert_eq!(lhs.message_type.len(), rhs.message_type.len());
            assert_eq!(lhs.enum_type.len(), rhs.enum_type.len());
            assert_eq!(lhs.service.len(), rhs.service.len());
            assert_eq!(lhs.extension.len(), rhs.extension.len());
        }
        Ok(())
    }

    #[test]
    fn test_phase2_3_ast_zero_manual_prost_message_and_1to1_bijection() -> TestResult {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let crates_dir = manifest_dir.join("..");
        let mut rs_files = Vec::new();
        collect_rs_files(&crates_dir, &mut rs_files);
        assert_ne!(rs_files.len(), 0);

        for rs_path in &rs_files {
            let content = std::fs::read_to_string(rs_path).map_err(|e| {
                crate::error::LibTpuError::BinaryFormatError {
                    reason: e.to_string(),
                }
            })?;
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("#[derive(") {
                    assert!(
                        !trimmed.contains("Message") && !trimmed.contains("Enumeration"),
                        "Forbidden manual #[derive(prost::Message/Enumeration)] outside OUT_DIR in {}: {trimmed}",
                        rs_path.display()
                    );
                }
                if let Some(rest) = trimmed
                    .strip_prefix("pub struct ")
                    .or_else(|| trimmed.strip_prefix("pub enum "))
                {
                    let ident: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    let is_c_abi_slice = rs_path.file_name().and_then(|n| n.to_str())
                        == Some("tpu_driver_types.rs")
                        && matches!(ident.as_str(), "HloProto" | "TpuSerializedProto");
                    assert!(
                        !ident.ends_with("Proto") || is_c_abi_slice,
                        "Forbidden hand-written `{trimmed}` outside OUT_DIR in {}",
                        rs_path.display()
                    );
                }
            }
        }

        let expected_proto_count = crate::elf::parse_golden_proto_descriptors(include_str!(
            "../../../docs/reverse_engineering/golden_proto_descriptors.json"
        ))?
        .proto_files_count;
        assert_eq!(protodesc_cold_293_file_count(), expected_proto_count);
        assert_eq!(
            ALL_REGISTERED_PROTO_MESSAGE_NAMES.len(),
            ALL_GENERATED_PROTO_BIJECTION.len()
        );

        let pool = global_descriptor_pool();
        let pool_non_wkt_msgs: std::collections::BTreeSet<String> = pool
            .all_messages()
            .filter(|m| !m.is_map_entry() && !m.full_name().starts_with("google.protobuf."))
            .map(|m| String::from(m.full_name()))
            .collect();
        assert_eq!(ALL_GENERATED_PROTO_BIJECTION.len(), pool_non_wkt_msgs.len());
        let generated_msgs: std::collections::BTreeSet<String> = ALL_GENERATED_PROTO_BIJECTION
            .iter()
            .map(|(fqn, _)| String::from(*fqn))
            .collect();
        assert_eq!(pool_non_wkt_msgs, generated_msgs);

        let pool_non_wkt_enums = pool
            .all_enums()
            .filter(|e| !e.full_name().starts_with("google.protobuf."))
            .count();
        assert_eq!(pool_non_wkt_enums, 1139);
        assert_eq!(GENERATED_PACKAGE_REPRESENTATIVES.len(), 177);

        for &(_pkg, _full_name, verify_equiv_fn, _roundtrip_fn) in GENERATED_PACKAGE_REPRESENTATIVES
        {
            verify_equiv_fn()?;
        }

        let verified = verify_all_registered_proto_schemas_strict()?;
        assert_eq!(verified, ALL_GENERATED_PROTO_BIJECTION.len());
        assert_eq!(
            verify_all_proto_descriptors()?,
            ALL_GENERATED_PROTO_BIJECTION.len()
        );
        Ok(())
    }

    #[test]
    fn test_descriptor_reflection_riegeli_highwayhash_and_drift_branches() -> TestResult {
        assert!(try_global_descriptor_pool().is_ok());
        assert!(decode_descriptor_pool(&[0xFF, 0xFF, 0xFF]).is_err());

        assert!(
            lookup_protodesc_cold_file_descriptor(
                "third_party/tensorflow/compiler/xla/service/hlo.proto"
            )
            .is_some()
        );
        assert!(lookup_protodesc_cold_file_descriptor("hlo.proto").is_some());
        assert!(lookup_protodesc_cold_file_descriptor("nonexistent_xyz.proto").is_none());
        assert!(lookup_message_descriptor("nonexistent.NoSuchMessage").is_err());
        assert!(decode_dynamic_proto_message("xla.TileProto", &[0xFF, 0xFF]).is_err());
        assert!(decode_dynamic_proto_json("xla.TileProto", "{bad json").is_err());
        assert!(decode_dynamic_proto_json("nonexistent.Msg", "{}").is_err());
        assert!(transcode_proto_bytes_to_json("xla.TileProto", &[0xFF]).is_err());
        assert!(decode_dynamic_proto_text("xla.TileProto", "invalid === text").is_err());
        assert!(decode_dynamic_proto_text("nonexistent.Msg", "").is_err());
        assert!(transcode_proto_bytes_to_text("xla.TileProto", &[0xFF]).is_err());

        let tile = TileProto {
            dimensions: vec![8, 128],
        };
        assert_eq!(TileProto::full_name(), "xla.TileProto");
        let wire = tile.encode_proto();
        assert_eq!(TileProto::decode_proto(&wire)?, tile);
        assert!(TileProto::decode_proto(&[0xFF, 0xFF]).is_err());
        assert_eq!(tile.sha256_fingerprint_hex().len(), 32);
        assert_ne!(tile.sha256_fingerprint_u64(), 0);
        assert_ne!(tile.highway_hash64(), 0);
        assert!(TileProto::descriptor().is_ok());
        let spec = TileProto::schema_spec()?;
        assert_eq!(spec.full_name, "xla.TileProto");
        let schema_str = TileProto::emit_proto_schema()?;
        assert!(schema_str.contains("message TileProto"));
        assert!(TileProto::verify_bidirectional_equivalence().is_ok());
        let dyn_tile = tile.to_dynamic_message()?;
        assert_eq!(TileProto::from_dynamic_message(&dyn_tile)?, tile);
        tile.verify_schema_sync()?;

        let text = tile.to_proto_text()?;
        assert_eq!(TileProto::from_proto_text(&text)?, tile);
        assert!(TileProto::from_proto_text("bad ===").is_err());
        let json = tile.to_proto_json()?;
        assert_eq!(TileProto::from_proto_json(&json)?, tile);
        assert!(TileProto::from_proto_json("{bad").is_err());

        assert!(extract_proto_message_spec("nonexistent.Msg").is_err());
        let root_spec = extract_proto_message_spec("TraceRecord")?;
        let root_emitted = emit_proto_schema_from_spec(&root_spec);
        assert!(root_emitted.contains("message TraceRecord"));

        // Map & Kind formatting coverage
        let comp_spec = extract_proto_message_spec("xla.CompileOptionsProto")?;
        let comp_emitted = emit_proto_schema_from_spec(&comp_spec);
        assert!(comp_emitted.contains("map<"));
        for msg in global_descriptor_pool()
            .all_messages()
            .filter(|m| m.fields().any(|f| f.is_map()))
        {
            let _ = build_populated_dynamic_message(&msg, 1);
        }
        let bool_map_fds = prost_types::FileDescriptorSet {
            file: vec![prost_types::FileDescriptorProto {
                name: Some(String::from("bm.proto")),
                syntax: Some(String::from("proto3")),
                message_type: vec![prost_types::DescriptorProto {
                    name: Some(String::from("BoolMapMsg")),
                    field: vec![prost_types::FieldDescriptorProto {
                        name: Some(String::from("flags")),
                        number: Some(1),
                        label: Some(3),
                        r#type: Some(11),
                        type_name: Some(String::from(".BoolMapMsg.FlagsEntry")),
                        ..prost_types::FieldDescriptorProto::default()
                    }],
                    nested_type: vec![prost_types::DescriptorProto {
                        name: Some(String::from("FlagsEntry")),
                        field: vec![
                            prost_types::FieldDescriptorProto {
                                name: Some(String::from("key")),
                                number: Some(1),
                                label: Some(1),
                                r#type: Some(8),
                                ..prost_types::FieldDescriptorProto::default()
                            },
                            prost_types::FieldDescriptorProto {
                                name: Some(String::from("value")),
                                number: Some(2),
                                label: Some(1),
                                r#type: Some(9),
                                ..prost_types::FieldDescriptorProto::default()
                            },
                        ],
                        options: Some(prost_types::MessageOptions {
                            map_entry: Some(true),
                            ..prost_types::MessageOptions::default()
                        }),
                        ..prost_types::DescriptorProto::default()
                    }],
                    ..prost_types::DescriptorProto::default()
                }],
                ..prost_types::FileDescriptorProto::default()
            }],
        };
        let bm_pool = prost_reflect::DescriptorPool::from_file_descriptor_set(bool_map_fds)
            .map_err(|e| crate::error::LibTpuError::BinaryFormatError {
                reason: e.to_string(),
            })?;
        if let Some(bm_desc) = bm_pool.get_message_by_name("BoolMapMsg") {
            let _ = build_populated_dynamic_message(&bm_desc, 1);
        }
        BarnaCoreModelParameters::default().verify_schema_sync()?;
        let nan_model = BarnaCoreModelParameters {
            core_frequency_ghz: Some(f32::NAN),
            ..BarnaCoreModelParameters::default()
        };
        assert!(nan_model.verify_schema_sync().is_err());
        for k in [
            prost_reflect::Kind::Double,
            prost_reflect::Kind::Float,
            prost_reflect::Kind::Int32,
            prost_reflect::Kind::Int64,
            prost_reflect::Kind::Uint32,
            prost_reflect::Kind::Uint64,
            prost_reflect::Kind::Sint32,
            prost_reflect::Kind::Sint64,
            prost_reflect::Kind::Fixed32,
            prost_reflect::Kind::Fixed64,
            prost_reflect::Kind::Sfixed32,
            prost_reflect::Kind::Sfixed64,
            prost_reflect::Kind::Bool,
            prost_reflect::Kind::String,
            prost_reflect::Kind::Bytes,
        ] {
            assert_ne!(format_proto_kind(&k), "");
            let _ = build_populated_field_value(&k, 3, 0);
        }

        // HighwayHash64 exact 32B + remainder
        assert_ne!(riegeli_highwayhash64(&[]), 0);
        assert_ne!(riegeli_highwayhash64(&[0x42; 32]), 0);
        assert_ne!(riegeli_highwayhash64(&[0x42; 47]), 0);

        // Riegeli Uncompressed, Metadata, Brotli, Zstd + multi-record >64KiB + corruptions
        for mode in [
            RiegeliChunkCompression::Uncompressed,
            RiegeliChunkCompression::Metadata,
            RiegeliChunkCompression::Brotli,
            RiegeliChunkCompression::Zstd,
        ] {
            let framed = RiegeliRecordCodec::wrap_record_compressed(&wire, mode);
            let (dec_mode, dec_bytes) = RiegeliRecordCodec::unwrap_record_decompressed(&framed)?;
            assert_eq!(dec_mode, mode);
            assert_eq!(dec_bytes, wire);
        }
        let plain_unwrap = RiegeliRecordCodec::unwrap_record_decompressed(&wire)?;
        assert_eq!(
            plain_unwrap,
            (RiegeliChunkCompression::Uncompressed, wire.clone())
        );
        assert_eq!(RiegeliRecordCodec::unwrap_record(&wire), wire.as_slice());
        let non_riegeli_64 = [0xAA_u8; 64];
        assert_eq!(
            RiegeliRecordCodec::unwrap_record(&non_riegeli_64),
            non_riegeli_64.as_slice()
        );
        let wrapped = encode_riegeli_record_container(&wire);
        assert_eq!(RiegeliRecordCodec::unwrap_record(&wrapped), wire.as_slice());
        assert_eq!(decode_riegeli_record_container(&wrapped)?, wire);
        assert_eq!(decode_riegeli_record_container(&wire)?, wire);

        let mut bad_brotli =
            RiegeliRecordCodec::wrap_record_compressed(&wire, RiegeliChunkCompression::Brotli);
        // Zero out checksums so unwrap_record_checked passes, then corrupt brotli stream
        if let Some(s) = bad_brotli.get_mut(48..64) {
            s.fill(0);
        }
        if let Some(s) = bad_brotli.get_mut(64..) {
            s.fill(0xFF);
        }
        assert!(RiegeliRecordCodec::unwrap_record_decompressed(&bad_brotli).is_err());

        let mut bad_hash = wrapped.clone();
        if let Some(b) = bad_hash.get_mut(50) {
            *b ^= 0xFF;
        }
        assert!(decode_riegeli_record_container(&bad_hash).is_err());

        let small_stream = RiegeliRecordCodec::wrap_records(&[wire.clone(), wire.clone()]);
        assert_eq!(
            RiegeliRecordCodec::unwrap_records(&small_stream)?,
            vec![wire.clone(), wire.clone()]
        );
        let big_rec = vec![0x5A_u8; 40_000];
        let big_stream = RiegeliRecordCodec::wrap_records(&[big_rec.clone(), big_rec.clone()]);
        assert!(big_stream.len() > RiegeliRecordCodec::BLOCK_BOUNDARY_BYTES);
        assert_eq!(
            RiegeliRecordCodec::unwrap_records(&big_stream)?,
            vec![big_rec.clone(), big_rec]
        );
        assert!(RiegeliRecordCodec::unwrap_records(&[0x01, 0x02, 0x03]).is_err());
        let mut trunc_stream = wrapped;
        trunc_stream.truncate(65);
        if let Some(s) = trunc_stream.get_mut(24..32) {
            s.copy_from_slice(&9999_u64.to_le_bytes());
        }
        assert!(RiegeliRecordCodec::unwrap_records(&trunc_stream).is_err());

        // preserve_unknown_wire_fields across wire types 0, 1, 2, 5, invalid(3), truncated
        let tile_desc = TileProto::descriptor()?;
        let mut extra_wire = wire.clone();
        extra_wire.extend_from_slice(&[(10 << 3), 0x2A]); // wt 0
        extra_wire.extend_from_slice(&[(11 << 3) | 1, 1, 2, 3, 4, 5, 6, 7, 8]); // wt 1
        extra_wire.extend_from_slice(&[(12 << 3) | 2, 2, 0xAA, 0xBB]); // wt 2
        extra_wire.extend_from_slice(&[(13 << 3) | 5, 1, 2, 3, 4]); // wt 5
        let preserved = preserve_unknown_wire_fields(&extra_wire, &wire, &tile_desc);
        assert!(preserved.len() > wire.len());
        let _ = preserve_unknown_wire_fields(&[0xFF; 12], &wire, &tile_desc);
        let _ = preserve_unknown_wire_fields(&[(14 << 3) | 3], &wire, &tile_desc);
        let _ = preserve_unknown_wire_fields(&[(15 << 3) | 1, 1, 2], &wire, &tile_desc);
        let _ = preserve_unknown_wire_fields(&[(15 << 3) | 5, 1, 2], &wire, &tile_desc);

        // Drift error branches in verify_bidirectional_proto_struct_equivalence_with_spec & closure
        let drop_roundtrip: fn(&[u8]) -> crate::error::LibTpuResult<Vec<u8>> = |_| Ok(Vec::new());
        assert!(
            verify_bidirectional_proto_struct_equivalence_with_spec(
                "xla.TileProto",
                drop_roundtrip,
                &spec,
                &schema_str,
            )
            .is_err()
        );
        assert!(
            verify_bidirectional_proto_struct_equivalence_with_spec(
                "xla.TileProto",
                roundtrip_proto_wire_bytes::<TileProto>,
                &spec,
                r#"syntax = "proto3"; message TileProto {}"#,
            )
            .is_err()
        );
        assert!(verify_registered_proto_schemas_closure(&[]).is_err());
        Ok(())
    }

    #[test]
    fn test_tpu_compilation_env_deepsea_pjrt_megascale_and_xplane_helpers() -> TestResult {
        // CompileOptionsProto + Local/GlobalTopologyProto
        assert_eq!(
            CompileOptionsProto::decode_or_empty(&[])?,
            CompileOptionsProto::default()
        );
        let co = CompileOptionsProto {
            profile_version: 7,
            ..CompileOptionsProto::default()
        };
        assert_eq!(
            CompileOptionsProto::decode_or_empty(&co.encode_proto())?,
            co
        );
        assert!(CompileOptionsProto::decode_or_empty(&[0xFF, 0xFF]).is_err());

        let lt = LocalTopologyProto {
            process_id: 1,
            boot_id: String::from("boot-0"),
            ..LocalTopologyProto::default()
        };
        assert_eq!(LocalTopologyProto::decode(&lt.encode())?, lt);
        assert!(LocalTopologyProto::decode(b"{bad").is_err());

        let gt = GlobalTopologyProto {
            processes: vec![lt],
        };
        assert_eq!(GlobalTopologyProto::decode(&gt.encode())?, gt);
        assert!(GlobalTopologyProto::decode(b"{bad").is_err());

        // Megascale pure-Rust wrappers
        assert_eq!(
            MegascaleMultiSliceConfigProto::decode(&[])?,
            MegascaleMultiSliceConfigProto::default()
        );
        let ms_empty = MegascaleMultiSliceConfigProto::default();
        assert_eq!(
            MegascaleMultiSliceConfigProto::decode(&ms_empty.encode())?,
            ms_empty
        );
        let ms_full = MegascaleMultiSliceConfigProto {
            num_slices: 2,
            endpoint_addresses: Some(EndpointAddressesProto::default()),
            dcn_topology: Some(DcnTopologyTreeProto {
                representation: Some(dcn_topology::Representation::SymmetricTree(
                    DcnSymmetricTreeProto {
                        branching_per_layer: vec![2],
                    },
                )),
            }),
            ..MegascaleMultiSliceConfigProto::default()
        };
        let ms_dec = MegascaleMultiSliceConfigProto::decode(&ms_full.encode())?;
        assert_eq!(ms_dec.num_slices, 2);
        assert!(ms_dec.endpoint_addresses.is_some());
        assert!(ms_dec.dcn_topology.is_some());
        assert!(MegascaleMultiSliceConfigProto::decode(&[0xFF, 0xFF]).is_err());

        let dcn_flat = MegascaleDcnTopologyProto {
            num_slices: 4,
            num_hosts_per_slice: 2,
            transport_type: 1,
        };
        assert_eq!(
            MegascaleDcnTopologyProto::decode(&dcn_flat.encode())?,
            dcn_flat
        );
        assert!(MegascaleDcnTopologyProto::decode(&[0xFF, 0xFF]).is_err());

        let metrics = MegascaleRuntimeMetricsProto {
            completed_barriers: 9,
            ..MegascaleRuntimeMetricsProto::default()
        };
        assert_eq!(
            MegascaleRuntimeMetricsProto::decode(&metrics.encode())?,
            metrics
        );
        assert!(MegascaleRuntimeMetricsProto::decode(&[0xFF, 0xFF]).is_err());

        let tcfg = MegascaleTransportConfigProto {
            num_channels: 8,
            ..MegascaleTransportConfigProto::default()
        };
        assert_eq!(MegascaleTransportConfigProto::decode(&tcfg.encode())?, tcfg);
        assert!(MegascaleTransportConfigProto::decode(&[0xFF, 0xFF]).is_err());

        // XPlane / Profiler generated prost wrappers
        assert_eq!(
            ProfileRequestProto::decode(&[])?,
            ProfileRequestProto::default()
        );
        assert_eq!(
            ProfileResponseProto::decode(&[])?,
            ProfileResponseProto::default()
        );
        assert_eq!(
            TerminateResponseProto::decode(&[])?,
            TerminateResponseProto::default()
        );
        assert_eq!(
            StopContinuousProfilingResponseProto::decode(&[])?,
            StopContinuousProfilingResponseProto::default()
        );
        let pr_none = ProfileRequestProto::default();
        assert_eq!(ProfileRequestProto::decode(&pr_none.encode())?, pr_none);
        let pr_some = ProfileRequestProto {
            duration_ms: 50,
            opts: Some(ProfileOptionsProto {
                version: 1,
                ..ProfileOptionsProto::default()
            }),
            ..ProfileRequestProto::default()
        };
        assert!(
            ProfileRequestProto::decode(&pr_some.encode())?
                .opts
                .is_some()
        );
        assert!(ProfileRequestProto::decode(&[0xFF, 0xFF]).is_err());

        let presp_none = ProfileResponseProto::default();
        assert_eq!(
            ProfileResponseProto::decode(&presp_none.encode())?,
            presp_none
        );
        let presp_some = ProfileResponseProto {
            xspace: Some(XSpaceProto {
                hostnames: vec![String::from("h0")],
                ..XSpaceProto::default()
            }),
            ..ProfileResponseProto::default()
        };
        assert!(
            ProfileResponseProto::decode(&presp_some.encode())?
                .xspace
                .is_some()
        );
        assert!(ProfileResponseProto::decode(&[0xFF, 0xFF]).is_err());

        let tr = TerminateRequestProto {
            session_id: String::from("s1"),
        };
        assert_eq!(TerminateRequestProto::decode(&tr.encode())?, tr);
        assert!(TerminateRequestProto::decode(&[0xFF, 0xFF]).is_err());
        let tres = TerminateResponseProto {};
        assert_eq!(TerminateResponseProto::decode(&tres.encode())?, tres);
        assert!(TerminateResponseProto::decode(&[0xFF, 0xFF]).is_err());
        let gs = GetSnapshotRequestProto {
            snapshot_session_id: String::from("sn1"),
        };
        assert_eq!(GetSnapshotRequestProto::decode(&gs.encode())?, gs);
        assert!(GetSnapshotRequestProto::decode(&[0xFF, 0xFF]).is_err());
        let scr = StopContinuousProfilingRequestProto {};
        assert_eq!(
            StopContinuousProfilingRequestProto::decode(&scr.encode())?,
            scr
        );
        assert!(StopContinuousProfilingRequestProto::decode(&[0xFF, 0xFF]).is_err());
        let scres = StopContinuousProfilingResponseProto {};
        assert_eq!(
            StopContinuousProfilingResponseProto::decode(&scres.encode())?,
            scres
        );
        assert!(StopContinuousProfilingResponseProto::decode(&[0xFF, 0xFF]).is_err());
        let mr = MonitorRequestProto {
            duration_ms: 10,
            monitoring_level: 2,
            timestamp: true,
        };
        assert_eq!(MonitorRequestProto::decode(&mr.encode())?, mr);
        assert!(MonitorRequestProto::decode(&[0xFF, 0xFF]).is_err());
        let mres_none = MonitorResponseProto::default();
        assert_eq!(
            MonitorResponseProto::decode(&mres_none.encode())?,
            mres_none
        );
        let mres_some = MonitorResponseProto {
            data: String::from("ok"),
            monitor_result: Some(ProfilerServiceMonitorResultProto {
                step_time_ms_avg: 2.5,
                ..ProfilerServiceMonitorResultProto::default()
            }),
        };
        assert!(
            MonitorResponseProto::decode(&mres_some.encode())?
                .monitor_result
                .is_some()
        );
        assert!(MonitorResponseProto::decode(&[0xFF, 0xFF]).is_err());
        let ipa = TpuInputPipelineAnalysisResultProto {
            hardware_is_tpu: true,
            ..TpuInputPipelineAnalysisResultProto::default()
        };
        assert_eq!(
            TpuInputPipelineAnalysisResultProto::decode(&ipa.encode())?,
            ipa
        );
        assert!(TpuInputPipelineAnalysisResultProto::decode(&[0xFF, 0xFF]).is_err());

        // TpuCompilationEnvironmentProto reflection & CLI flag helpers
        assert!(
            TpuCompilationEnvironmentProto::lookup_jellyfish_field_spec(
                "--xla_tpu_scoped_vmem_limit_kib"
            )
            .is_some()
        );
        assert!(TpuCompilationEnvironmentProto::lookup_jellyfish_field_spec("418").is_some());
        assert!(TpuCompilationEnvironmentProto::lookup_jellyfish_field_spec("nope").is_none());
        assert_ne!(
            TpuCompilationEnvironmentProto::list_jellyfish_schema_field_names().len(),
            0
        );

        let mut empty_env = TpuCompilationEnvironmentProto::default();
        assert!(
            empty_env
                .reflect_as_message_by_name("nonexistent.Msg")
                .is_none()
        );
        assert_eq!(
            empty_env.get_field_as_string_in_schema("nonexistent.Msg", "k"),
            "default"
        );
        empty_env.set_field_from_str_in_schema("nonexistent.Msg", "k", "v");
        let mut dummy_cfg = TpuInitConfig::default();
        empty_env.apply_to_init_config_in_schema("nonexistent.Msg", &mut dummy_cfg);
        let empty_env = TpuCompilationEnvironmentProto::default();
        assert_eq!(empty_env.get_field_as_string("xla_jf_llo_level"), "default");
        assert_eq!(
            empty_env.get_field_as_string("unknown_nonexistent_flag"),
            "default"
        );

        let mut comp_env = TpuCompilationEnvironmentProto::from_cli_flags(
            "-- --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_autotune_windows \
             --xla_jf_llo_level=2 --xla_tpu_random_all_reduce_delay_mask=7 \
             --xla_tpu_keep_hlo_proto_literals_up_to=128 \
             --xla_tpu_embedding_table_oblongness_threshold=1.5 \
             --internal_embedding_emitter_fraction_vmem_available=0.25 \
             --xla_jf_conv_min_limit_vmem_mib=4.5 --xla_vf_vmem_use_ior_algorithm=fast \
             --xla_enable_async_collective_permute=1 \
             --xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true \
             --xla_tpu_custom_extra=hello --xla_tpu_custom_extra2=world",
        );
        comp_env.set_field_from_str("418", "65536");
        comp_env.set_field_from_str("xla_enable_async_collective_permute", "ENABLED");
        comp_env.set_field_from_str("xla_enable_async_collective_permute", "NOT_AN_ENUM");
        comp_env.set_field_from_str("xla_tpu_memory_bound_loop_optimizer_options", "{}");
        comp_env.xla_tpu_memory_bound_loop_optimizer_options =
            Some(MemoryBoundLoopOptimizerOptionsProto::default());
        assert_eq!(comp_env.get_field_as_string("xla_jf_llo_level"), "2");
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_random_all_reduce_delay_mask"),
            "7"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_keep_hlo_proto_literals_up_to"),
            "128"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_embedding_table_oblongness_threshold"),
            "1.5"
        );
        assert_eq!(
            comp_env.get_field_as_string("internal_embedding_emitter_fraction_vmem_available"),
            "0.25"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_enable_async_collective_permute"),
            "2"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_memory_bound_loop_optimizer_options"),
            "set"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_scoped_vmem_limit_kib"),
            "65536"
        );
        assert_eq!(comp_env.get_field_as_string("418"), "65536");
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_autotune_windows"),
            "true"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_msa_reduce_scoped_vmem_limit"),
            "false"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_jf_conv_min_limit_vmem_mib"),
            "4.5"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_vf_vmem_use_ior_algorithm"),
            "fast"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "true"
        );
        comp_env.set_field_from_str(
            "xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments",
            "42",
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "42"
        );
        comp_env.set_field_from_str(
            "xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments",
            "3.25",
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "3.25"
        );
        comp_env.set_field_from_str(
            "xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments",
            "mode_a",
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "mode_a"
        );
        comp_env.xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments =
            Some(AutoProto { value: None });
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "auto"
        );
        comp_env.xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments = Some(AutoProto {
            value: Some(auto_proto::Value::U64(99)),
        });
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "99"
        );
        comp_env.xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments = Some(AutoProto {
            value: Some(auto_proto::Value::I32(-7)),
        });
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "-7"
        );
        comp_env.xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments = Some(AutoProto {
            value: Some(auto_proto::Value::U32(7)),
        });
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "7"
        );
        comp_env.xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments = Some(AutoProto {
            value: Some(auto_proto::Value::F(1.5)),
        });
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "1.5"
        );
        comp_env.xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments = Some(AutoProto {
            value: Some(auto_proto::Value::ShardyOptions(
                JellyfishShardyOptionsProto::default(),
            )),
        });
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments"),
            "set"
        );
        assert_eq!(
            comp_env.get_field_as_string("xla_tpu_custom_extra"),
            "hello"
        );
        assert_eq!(comp_env.get_field_as_string("xla_tpu_autotune_level"), "2");
        comp_env.set_field_from_str(
            "custom_flags",
            "bare_tok;xla_tpu_scoped_vmem_limit_kib=bad;xla_tpu_scoped_vmem_limit_kib=32768",
        );
        assert_eq!(
            comp_env.get_field_as_string("unknown_nonexistent_flag"),
            "default"
        );

        let mut init_cfg = TpuInitConfig::default();
        comp_env.apply_to_init_config(&mut init_cfg);
        assert_eq!(init_cfg.xla_tpu_scoped_vmem_limit_kib, 32768);

        // Embedding pure-Rust wrappers
        let ep = TpuEmbeddingPartitionedConfigProto {
            num_partitions: 2,
            table_names: vec![String::from("t0")],
            raw_config_bytes: vec![1, 2],
        };
        assert_eq!(
            TpuEmbeddingPartitionedConfigProto::decode(&ep.encode())?,
            ep
        );
        assert!(TpuEmbeddingPartitionedConfigProto::decode(b"{bad").is_err());

        let em = TpuEmbeddingMemoryConfigProto {
            num_segments: 1,
            segment_blobs: vec![vec![3]],
        };
        assert_eq!(TpuEmbeddingMemoryConfigProto::decode(&em.encode())?, em);
        assert!(TpuEmbeddingMemoryConfigProto::decode(b"{bad").is_err());

        let eh = TpuEmbeddingHostNetworkConfigProto {
            endpoint_address: String::from("127.0.0.1:8471"),
            partition_payload: vec![4],
        };
        assert_eq!(
            TpuEmbeddingHostNetworkConfigProto::decode(&eh.encode())?,
            eh
        );
        assert!(TpuEmbeddingHostNetworkConfigProto::decode(b"{bad").is_err());

        // DeepSea / Relocation / ChipConfig / TargetArguments / HostCommandEnvelope
        for tpu_gen in [
            TpuGeneration::V2,
            TpuGeneration::V3,
            TpuGeneration::V4,
            TpuGeneration::V4Lite,
            TpuGeneration::V5e,
            TpuGeneration::V5p,
            TpuGeneration::V6e,
            TpuGeneration::V7x,
        ] {
            let ta = TargetArgumentsProto::from_generation(tpu_gen);
            let ta_wire = ta.encode_proto();
            assert_eq!(TargetArgumentsProto::decode_proto(&ta_wire)?, ta);
        }

        let mut imem = vec![0_u8; 128];
        let rel_in = ProgramRelocationProto {
            bundle: 0,
            slot: 3,
            kind: 0,
            memory_space: 0,
            memory_sub_space: 0,
        };
        rel_in.apply_to_imem(&mut imem, &[0xDEAD_BEEF], &[0xCAFE_BABE]);
        assert_eq!(
            imem.get(12..16),
            Some(0xDEAD_BEEF_u32.to_le_bytes().as_slice())
        );
        let rel_out = ProgramRelocationProto {
            bundle: 1,
            slot: 2,
            kind: 1,
            memory_space: 0,
            memory_sub_space: 0,
        };
        rel_out.apply_to_imem(&mut imem, &[0xDEAD_BEEF], &[0xCAFE_BABE]);
        assert_eq!(
            imem.get(72..76),
            Some(0xCAFE_BABE_u32.to_le_bytes().as_slice())
        );
        let rel_oob = ProgramRelocationProto {
            bundle: 99,
            slot: 0,
            kind: 0,
            memory_space: 0,
            memory_sub_space: 99,
        };
        rel_oob.apply_to_imem(&mut imem, &[1], &[2]);

        let sep = TpuSerializedExecutableProto {
            name: String::from("m"),
            output_shapes: vec![ShapeProto {
                element_type: 11,
                dimensions: vec![4],
                ..ShapeProto::default()
            }],
            ..TpuSerializedExecutableProto::default()
        };
        let sep_dec = TpuSerializedExecutableProto::decode(&sep.encode())?;
        assert_eq!(sep_dec.output_shapes.len(), 1);
        assert!(TpuSerializedExecutableProto::decode(&[0xFF, 0xFF]).is_err());

        let hcmd = HostCommandEnvelopeProto {
            opcode: 2,
            channel_id: 9,
            continuation_id: 9,
            handler_id: 3,
            payload: vec![0xAA, 0xBB],
            ..HostCommandEnvelopeProto::default()
        };
        assert_eq!(HostCommandEnvelopeProto::decode(&hcmd.encode())?, hcmd);
        assert!(HostCommandEnvelopeProto::decode(&[0xFF, 0xFF]).is_err());
        let bin_cmd = [0x08, 0x01, 0x10, 0x05, 0x18, 0x07, 0x22, 0x02, 0xDE, 0xAD];
        let bin_dec = HostCommandEnvelopeProto::decode(&bin_cmd)?;
        assert_eq!(bin_dec.opcode, 1);
        assert_eq!(bin_dec.continuation_id, 5);
        assert_eq!(bin_dec.handler_id, 7);
        assert_eq!(bin_dec.payload, vec![0xDE, 0xAD]);
        assert!(HostCommandEnvelopeProto::decode(&[0x80]).is_err());
        assert!(HostCommandEnvelopeProto::decode(&[0x81; 11]).is_err());
        assert!(HostCommandEnvelopeProto::decode(&[0x08]).is_err());
        assert!(HostCommandEnvelopeProto::decode(&[0x10]).is_err());
        assert!(HostCommandEnvelopeProto::decode(&[0x18]).is_err());
        assert!(HostCommandEnvelopeProto::decode(&[0x22]).is_err());
        assert!(HostCommandEnvelopeProto::decode(&[0x22, 0x05, 0x01]).is_err());
        assert!(HostCommandEnvelopeProto::decode(&[0x7D, 0x01, 0x02]).is_err());
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn prop_generated_domain_packages_wire_json_text_roundtrip(
            dim0 in 1_i64..256,
            dim1 in 1_i64..256,
            vmem_kib in 1024_i64..65536,
            slice_cnt in 1_i32..8,
            err_code in 1_i32..16,
            dur_ps in 100_i64..1_000_000,
            pkg_idx in 0_usize..177,
        ) {
            // 0. Randomized wire + JSON + textproto round-trip across all 177 generated packages
            if let Some(&(_pkg, full_name, _verify_fn, roundtrip_fn)) = GENERATED_PACKAGE_REPRESENTATIVES.get(pkg_idx) {
                if let Ok(desc) = lookup_message_descriptor(full_name) {
                    let dyn_msg = build_populated_dynamic_message(&desc, 0);
                    let dyn_wire = dyn_msg.encode_to_vec();
                    let rt_wire = roundtrip_fn(&dyn_wire).unwrap_or_default();
                    prop_assert_eq!(decode_dynamic_proto_message(full_name, &rt_wire).ok(), Some(dyn_msg.clone()));
                    if let Ok(js) = transcode_proto_bytes_to_json(full_name, &rt_wire) {
                        prop_assert!(decode_dynamic_proto_json(full_name, &js).is_ok());
                    }
                    if let Ok(txt) = transcode_proto_bytes_to_text(full_name, &rt_wire) {
                        prop_assert_eq!(decode_dynamic_proto_text(full_name, &txt).ok(), Some(dyn_msg));
                    }
                }
            }

            // 1. xla package (ShapeProto + HloModuleProto + IsaProgramProto + TpuExecutableProto)
            let shape = ShapeProto {
                element_type: 11,
                dimensions: vec![dim0, dim1],
                layout: Some(Box::new(LayoutProto {
                    minor_to_major: vec![1, 0],
                    tiles: vec![TileProto { dimensions: vec![8, 128] }],
                    ..LayoutProto::default()
                })),
                ..ShapeProto::default()
            };
            let shape_wire = shape.encode_proto();
            prop_assert_eq!(ShapeProto::decode_proto(&shape_wire).ok(), Some(shape.clone()));
            let shape_json = shape.to_proto_json().unwrap_or_default();
            prop_assert_eq!(ShapeProto::from_proto_json(&shape_json).ok(), Some(shape.clone()));
            let shape_txt = shape.to_proto_text().unwrap_or_default();
            prop_assert_eq!(ShapeProto::from_proto_text(&shape_txt).ok(), Some(shape));

            // 2. xla.jellyfish package (TpuCompilationEnvironment)
            let env_proto = TpuCompilationEnvironmentProto {
                xla_tpu_scoped_vmem_limit_kib: Some(vmem_kib),
                ..TpuCompilationEnvironmentProto::default()
            };
            let env_wire = env_proto.encode_proto();
            prop_assert_eq!(TpuCompilationEnvironmentProto::decode_proto(&env_wire).ok(), Some(env_proto.clone()));
            let env_txt = env_proto.to_proto_text().unwrap_or_default();
            prop_assert_eq!(TpuCompilationEnvironmentProto::from_proto_text(&env_txt).ok(), Some(env_proto));

            // 3. xla.megascale.runtime package (DCNTopology + MegaScaleRuntimeError)
            let dcn = DcnTopologyTreeProto {
                representation: Some(dcn_topology::Representation::SymmetricTree(
                    DcnSymmetricTreeProto {
                        branching_per_layer: vec![slice_cnt, 2],
                    },
                )),
            };
            let dcn_json = dcn.to_proto_json().unwrap_or_default();
            prop_assert_eq!(DcnTopologyTreeProto::from_proto_json(&dcn_json).ok(), Some(dcn));

            // 4. tensorflow & tensorflow.profiler packages (StatusProto + XSpace)
            let mut payload = BTreeMap::new();
            payload.insert(String::from("k"), vec![1_u8, 2_u8]);
            let status = StatusProto {
                code: err_code,
                message: format!("err_{err_code}"),
                payload,
            };
            let st_wire = status.encode_proto();
            prop_assert_eq!(StatusProto::decode_proto(&st_wire).ok(), Some(status.clone()));
            let st_txt = status.to_proto_text().unwrap_or_default();
            prop_assert_eq!(StatusProto::from_proto_text(&st_txt).ok(), Some(status));

            let xspace = XSpaceProto {
                planes: vec![XPlaneProto {
                    id: 1,
                    name: String::from("/device:TPU:0"),
                    lines: vec![XLineProto {
                        id: 0,
                        events: vec![XEventProto {
                            metadata_id: 1,
                            duration_ps: dur_ps,
                            data: Some(x_event::Data::OffsetPs(0)),
                            stats: Vec::new(),
                        }],
                        ..XLineProto::default()
                    }],
                    ..XPlaneProto::default()
                }],
                ..XSpaceProto::default()
            };
            let xs_json = xspace.to_proto_json().unwrap_or_default();
            prop_assert_eq!(XSpaceProto::from_proto_json(&xs_json).ok(), Some(xspace));

            // 5. tpu & asic_sw packages (TpuChipConfigProto + CoreDumpProto)
            let chip_cfg = TpuChipConfigProto::from_generation(TpuGeneration::V6e);
            let cc_wire = chip_cfg.encode_proto();
            prop_assert_eq!(TpuChipConfigProto::decode_proto(&cc_wire).ok(), Some(chip_cfg));
        }
    }
}
