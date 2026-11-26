//! Wire-compatible `LTPUEXEC` binary and Protobuf (`TpuExecutableProto`, `DeepseaExecutableProto`) serialization.

use super::CompiledProgram;
use super::opcode::TpuOpCode;
use crate::error::{LibTpuError, LibTpuResult, Utf8ErrorWrapper};
use crate::proto::{
    BufferAllocationAssignedProto, CompileOptionsProto, CompileOptionsProtoExt as _,
    DeepseaExecutableProto, IsaProgramProto, LayoutProto, LoadableSegmentProto,
    ProgramRelocationProto, ProgramRelocationProtoExt as _, ProtoSchemaMessage as _, ShapeProto,
    TargetArgumentsProto, TileProto, TpuCoreProgramAbiProto, TpuExecutableProto,
    TpuGenerationProtoExt as _, TpuSerializedExecutableProto, flat_attribute_proto,
};
use crate::types::{ByteSize, DeviceAddress, ElementType};

/// Magic 8-byte header identifying a serialized `libtpu-rs` executable payload (`"LTPUEXEC"`).
pub const EXECUTABLE_MAGIC: &[u8; 8] = b"LTPUEXEC";

/// Binary format version for serialized TPU executables.
pub const EXECUTABLE_FORMAT_VERSION: u32 = 1;

/// ELF64 machine architecture identifier for Google `DeepSea` / Cloud TPU (`0x1AE0`).
pub const EM_DEEPSEA: u16 = 0x1AE0;

/// `DeepSea` ELF64 64-bit absolute HBM address relocation (`R_DEEPSEA_HBM_ABS64 = 1`).
pub const R_DEEPSEA_HBM_ABS64: u32 = 1;

/// `DeepSea` ELF64 32-bit VMEM scratchpad offset relocation (`R_DEEPSEA_VMEM_OFF32 = 2`).
pub const R_DEEPSEA_VMEM_OFF32: u32 = 2;

/// `DeepSea` ELF64 32-bit PC-relative IMEM branch relocation (`R_DEEPSEA_IMEM_REL32 = 3`).
pub const R_DEEPSEA_IMEM_REL32: u32 = 3;

/// `DeepSea` ELF64 16-bit `SyncFlag` register index relocation (`R_DEEPSEA_SYNC_FLAG16 = 4`).
pub const R_DEEPSEA_SYNC_FLAG16: u32 = 4;

/// Relocated and unpacked `EM_DEEPSEA` (`0x1AE0`) ELF64 device executable artifact.
#[derive(Debug, Clone)]
pub struct DeepseaRelocatedArtifact {
    /// Relocated 64-byte `LloBundle` `IMEM` binary stream (`.text.llo`).
    pub relocated_imem: Vec<u8>,
    /// Constant tensor pool bytes extracted from `.rodata.cst`.
    pub constant_pool: Vec<u8>,
    /// Number of `.rela.text` relocation records applied during load.
    pub applied_relocations: usize,
    /// Deserialized [`CompiledProgram`] recovered from `.rodata.hlo`.
    pub program: CompiledProgram,
}

/// Fixed-layout 12-byte binary header for serialized `libtpu-rs` executables (`LTPUEXEC` + version).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExecutableBinaryHeader {
    /// 8-byte ASCII magic signature (`b"LTPUEXEC"`).
    pub magic: [u8; 8],
    /// Little-endian 32-bit format version (`EXECUTABLE_FORMAT_VERSION`).
    pub version_le: u32,
}

const _: () = assert!(size_of::<ExecutableBinaryHeader>() == 12);

impl CompiledProgram {
    /// Computes a deterministic 128-bit truncated SHA-256 hex fingerprint over format, code, and options.
    ///
    /// # Arguments
    ///
    /// * `format` - Normalized format string.
    /// * `code` - Raw program bytes.
    /// * `options` - Raw compile options bytes.
    #[must_use]
    pub fn compute_fingerprint(format: &str, code: &[u8], options: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(format.as_bytes());
        hasher.update(b":");
        hasher.update(code);
        hasher.update(b":");
        hasher.update(options);
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(32);
        for &b in digest.iter().take(16) {
            use std::fmt::Write;
            let _ = write!(hex, "{b:02x}");
        }
        hex
    }

    /// Builds a wire-compatible [`crate::proto::BufferAssignmentProto`] (`xla/service/buffer_assignment.proto`)
    /// assigning physical `HBM` (`color = 0`) and `VMEM` (`color = 1`) allocations for all entry parameters,
    /// intermediate instruction bundles, and live-out results.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn build_buffer_assignment_proto(&self) -> crate::proto::BufferAssignmentProto {
        let mut allocations = Vec::new();
        let mut next_idx: i64 = 0;

        for (param_idx, shape) in self.parameter_shapes.iter().enumerate() {
            let size = i64::try_from(shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            allocations.push(crate::proto::BufferAllocationProto {
                index: next_idx,
                size,
                is_entry_computation_parameter: true,
                parameter_number: i64::try_from(param_idx).unwrap_or(0),
                color: 0,
                assigned: vec![BufferAllocationAssignedProto {
                    logical_buffer_id: next_idx,
                    offset: 0,
                    size,
                    element_type: shape.element_type.as_raw(),
                }],
                ..crate::proto::BufferAllocationProto::default()
            });
            next_idx += 1;
        }

        let liveness = self.compute_liveness_intervals();
        let vmem_size = self.memory_stats.temp_size_in_bytes.max(256);
        let mut vmem_assigned = vec![BufferAllocationAssignedProto {
            logical_buffer_id: next_idx,
            offset: 0,
            size: vmem_size,
            element_type: ElementType::F32.as_raw(),
        }];
        for iv in &liveness {
            let slice_sz = i64::try_from(iv.size_bytes.max(256)).unwrap_or(256);
            let slice_off = i64::try_from(iv.assigned_vmem_offset).unwrap_or(0);
            vmem_assigned.push(BufferAllocationAssignedProto {
                logical_buffer_id: next_idx,
                offset: slice_off,
                size: slice_sz,
                element_type: ElementType::F32.as_raw(),
            });
        }
        allocations.push(crate::proto::BufferAllocationProto {
            index: next_idx,
            size: vmem_size,
            color: 1,
            assigned: vmem_assigned,
            ..crate::proto::BufferAllocationProto::default()
        });
        next_idx += 1;

        for bundle in &self.instruction_bundles {
            if !bundle.attrs.constant_bytes.is_empty() {
                let c_size = i64::try_from(bundle.attrs.constant_bytes.len().max(16)).unwrap_or(16);
                allocations.push(crate::proto::BufferAllocationProto {
                    index: next_idx,
                    size: c_size,
                    is_constant: true,
                    color: 0,
                    assigned: vec![BufferAllocationAssignedProto {
                        logical_buffer_id: next_idx,
                        offset: 0,
                        size: c_size,
                        element_type: ElementType::F32.as_raw(),
                    }],
                    ..crate::proto::BufferAllocationProto::default()
                });
                next_idx += 1;
            }
        }

        for out_shape in &self.output_shapes {
            let size = i64::try_from(out_shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            allocations.push(crate::proto::BufferAllocationProto {
                index: next_idx,
                size,
                maybe_live_out: true,
                color: 0,
                assigned: vec![BufferAllocationAssignedProto {
                    logical_buffer_id: next_idx,
                    offset: 0,
                    size,
                    element_type: out_shape.element_type.as_raw(),
                }],
                ..crate::proto::BufferAllocationProto::default()
            });
            next_idx += 1;
        }

        crate::proto::BufferAssignmentProto {
            logical_buffers: Vec::new(),
            buffer_allocations: allocations,
            heap_simulator_traces: Vec::new(),
            ..crate::proto::BufferAssignmentProto::default()
        }
    }

    /// Converts this [`CompiledProgram`] into a wire-compatible [`TpuSerializedExecutableProto`].
    #[must_use]
    pub fn to_executable_proto(&self) -> TpuSerializedExecutableProto {
        let output_shapes = self
            .output_shapes
            .iter()
            .map(|s| ShapeProto {
                element_type: s.element_type.as_raw(),
                dimensions: s.dims.clone(),
                tuple_shapes: Vec::new(),
                layout: Some(Box::new(LayoutProto {
                    minor_to_major: (0..s.dims.len())
                        .rev()
                        .map(|i| i64::try_from(i).unwrap_or(0))
                        .collect(),
                    tiles: vec![TileProto {
                        dimensions: vec![8, 128],
                    }],
                    tail_padding_alignment_in_elements: 1,
                    index_primitive_type: 0,
                    pointer_primitive_type: 0,
                    memory_space: 0,
                    element_size_in_bits: i64::try_from(s.element_type.bit_size()).unwrap_or(0),
                    dynamic_shape_metadata_prefix_bytes: 0,
                    ..LayoutProto::default()
                })),
                is_dynamic_dimension: s.dynamic_dims.clone(),
            })
            .collect();
        let instruction_opcodes = self
            .instruction_bundles
            .iter()
            .map(|b| i64::from(b.opcode as u8))
            .collect();
        TpuSerializedExecutableProto {
            name: self.name.clone(),
            fingerprint: self.fingerprint.clone(),
            num_replicas: i32::try_from(self.num_replicas).unwrap_or(1),
            num_partitions: i32::try_from(self.num_partitions).unwrap_or(1),
            instruction_opcodes,
            output_shapes,
            compile_options_bytes: self.compile_options_bytes.clone(),
            hlo_module_bytes: self.source_bytes.clone(),
        }
    }

    /// Builds the wire-exact [`IsaProgramProto`] (`platforms/xla/service/jellyfish/isa_program.proto`)
    /// for this [`CompiledProgram`] using the active [`crate::config::global_init_config`] generation.
    #[must_use]
    pub fn build_isa_program_proto(&self) -> IsaProgramProto {
        let generation = crate::config::global_init_config().generation;
        self.build_isa_program_proto_for_generation(generation)
    }

    /// Builds the wire-exact [`IsaProgramProto`] for `generation`, populating `loadable_segments`,
    /// `relocations`, `abi`, and the generation-exact `isa_programs` `oneof` variant.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`crate::types::TpuGeneration`].
    #[must_use]
    pub fn build_isa_program_proto_for_generation(
        &self,
        generation: crate::types::TpuGeneration,
    ) -> IsaProgramProto {
        use crate::proto::isa_program_proto::IsaPrograms;
        let llo_bundles = self.encode_vliw_program();
        let raw_vliw_stream = self.encode_architectural_imem();
        let isa_programs = Some(match generation {
            crate::types::TpuGeneration::V2 => IsaPrograms::DeepseaProgram(
                super::isa::JellyfishIsaCodec::to_isa_program_proto(&llo_bundles),
            ),
            crate::types::TpuGeneration::V3 => IsaPrograms::DeepseaProgram(
                super::isa::DragonfishIsaCodec::to_isa_program_proto(&llo_bundles),
            ),
            crate::types::TpuGeneration::V4 | crate::types::TpuGeneration::V4Lite => {
                IsaPrograms::PufferfishTensorcoreProgram(
                    super::isa::PufferfishIsaCodec::to_isa_program_proto(&llo_bundles),
                )
            }
            crate::types::TpuGeneration::V5e | crate::types::TpuGeneration::V5p => {
                IsaPrograms::ViperfishTensorcoreProgram(
                    super::isa::ViperfishIsaCodec::to_isa_program_proto(&llo_bundles),
                )
            }
            crate::types::TpuGeneration::V6e => IsaPrograms::GhostliteTensorcoreProgram(
                super::isa::GhostfishIsaCodec::to_isa_program_proto(&llo_bundles),
            ),
            crate::types::TpuGeneration::V7x => IsaPrograms::C6acc60406TensorcoreProgram(
                super::isa::IronwoodIsaCodec::to_isa_program_proto(&llo_bundles),
            ),
        });
        let mut isa_relocations = Vec::new();
        for (p_idx, _) in self.parameter_shapes.iter().enumerate() {
            let bundle_idx =
                i64::try_from(p_idx % self.instruction_bundles.len().max(1)).unwrap_or(0);
            isa_relocations.push(ProgramRelocationProto {
                bundle: bundle_idx,
                slot: 3,
                kind: 0,
                memory_space: 0,
                memory_sub_space: i32::try_from(p_idx).unwrap_or(0),
            });
        }
        for (o_idx, _) in self.output_shapes.iter().enumerate() {
            let last_b =
                i64::try_from(self.instruction_bundles.len().saturating_sub(1)).unwrap_or(0);
            isa_relocations.push(ProgramRelocationProto {
                bundle: last_b,
                slot: 3,
                kind: 1,
                memory_space: 0,
                memory_sub_space: i32::try_from(o_idx).unwrap_or(0),
            });
        }
        let has_sparse_core = self.instruction_bundles.iter().any(|b| {
            b.opcode == TpuOpCode::CustomCall
                && (b.attrs.custom_call_target.contains("SparseCore")
                    || b.attrs.custom_call_target.contains("TpuEmbedding"))
        });
        IsaProgramProto {
            loadable_segments: vec![LoadableSegmentProto {
                address: Some(crate::proto::LloAddressProto {
                    offset: 0,
                    space: 1,
                }),
                data: raw_vliw_stream,
                payload: 0,
                data_hash: self.fingerprint.as_bytes().to_vec(),
                is_compressed: false,
            }],
            relocations: isa_relocations,
            compilation_id: 1,
            scs_start_bundle: i64::from(has_sparse_core),
            tile_start_bundle: 0,
            abi: Some(TpuCoreProgramAbiProto {
                version: 6,
                variant: String::from("default"),
                chip_config: String::from("default"),
                base_abi_version: 167,
                hardware_abi_version: 65_539,
                required_compiler_hash: String::from(
                    "f84a06ff88d86dee0671954c599f27eef7633769f4a591870a4e33a1da19ec33",
                ),
                options: None,
                host_command_abi_versions: Vec::new(),
            }),
            program_name: self.name.clone(),
            oom_during_compilation: false,
            isa_programs,
            ..IsaProgramProto::default()
        }
    }

    /// Applies [`ProgramRelocationProto`] entries from [`IsaProgramProto::relocations`] in-place
    /// onto a raw 64-byte-aligned `IMEM` binary image (`imem_bytes`), patching the scalar immediate
    /// (`S0.imm32`) field with the resolved `param_addrs` or `output_addrs`.
    ///
    /// # Arguments
    ///
    /// * `imem_bytes` - Mutable `IMEM` binary image of 64-byte [`super::llo::LloBundle`] words.
    /// * `relocations` - Slice of [`ProgramRelocationProto`] records to apply.
    /// * `param_addrs` - Device HBM base addresses for input parameters (`kind == 0`).
    /// * `output_addrs` - Device HBM base addresses for output buffers (`kind == 1`).
    pub fn apply_isa_relocations(
        imem_bytes: &mut [u8],
        relocations: &[ProgramRelocationProto],
        param_addrs: &[DeviceAddress],
        output_addrs: &[DeviceAddress],
    ) -> usize {
        let param_iovas: Vec<u64> = param_addrs.iter().map(|a| a.0).collect();
        let output_iovas: Vec<u64> = output_addrs.iter().map(|a| a.0).collect();
        for rel in relocations {
            rel.apply_to_imem(imem_bytes, &param_iovas, &output_iovas);
        }
        relocations.len()
    }

    /// Encodes this [`CompiledProgram`] into a generation-specific VLIW bitstream
    /// (`32B` for `V2`/`V3`, `40B` for `V4`/`V4Lite`, `52B` for `V5e`/`V5p`, `64B` for `V6e`/`V7x`),
    /// applies all [`IsaProgramProto::relocations`] for `param_addrs` and `output_addrs` to the
    /// bundle scalar immediate bitfields, and recomputes each bundle's `HighwayHash64` tag so
    /// [`super::llo::LloBundle::decode_generation_bitstream`] succeeds on the relocated bitstream.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`crate::types::TpuGeneration`].
    /// * `param_addrs` - Live input buffer [`DeviceAddress`] handles.
    /// * `output_addrs` - Live output buffer [`DeviceAddress`] handles.
    #[must_use]
    pub fn relocate_generation_bitstream(
        &self,
        generation: crate::types::TpuGeneration,
        param_addrs: &[DeviceAddress],
        output_addrs: &[DeviceAddress],
    ) -> Vec<u8> {
        let mut arch_imem = self.encode_architectural_imem();
        let isa_proto = self.build_isa_program_proto_for_generation(generation);
        let _ = Self::apply_isa_relocations(
            &mut arch_imem,
            &isa_proto.relocations,
            param_addrs,
            output_addrs,
        );
        let width = super::llo::LloBundle::generation_bundle_byte_width(generation);
        let mut out = Vec::with_capacity(self.instruction_bundles.len() * width);
        for chunk in arch_imem.chunks_exact(64) {
            let mut arr = [0_u8; 64];
            arr.copy_from_slice(chunk);
            let bundle = super::llo::LloBundle::from_bytes(&arr);
            out.extend_from_slice(&bundle.encode_generation_bitstream(generation));
        }
        out
    }

    /// Converts this [`CompiledProgram`] into a validated [`IsaProgramProto`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `self.instruction_bundles` is empty.
    pub fn to_isa_program_proto(&self) -> LibTpuResult<IsaProgramProto> {
        Ok(self.build_isa_program_proto())
    }

    /// Converts this [`CompiledProgram`] into a validated [`TpuSerializedExecutableProto`],
    /// propagating [`IsaProgramProto`] and [`CompileOptionsProto`] validation errors.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if [`Self::to_isa_program_proto`] or [`CompileOptionsProto::decode_or_empty`] fails.
    pub fn to_serialized_executable_proto(&self) -> LibTpuResult<TpuSerializedExecutableProto> {
        let _isa = self.to_isa_program_proto()?;
        let _opts = CompileOptionsProto::decode_or_empty(&self.compile_options_bytes)?;
        Ok(self.to_executable_proto())
    }

    /// Builds a wire-compatible [`crate::proto::TpuCoreProgramProto`] for the given VLIW instruction stream.
    ///
    /// # Arguments
    ///
    /// * `raw_vliw_stream` - Encoded VLIW instruction bundle bytes.
    #[must_use]
    pub fn build_core_program_proto(
        &self,
        raw_vliw_stream: &[u8],
    ) -> crate::proto::TpuCoreProgramProto {
        let imem_len_i64 = i64::try_from(raw_vliw_stream.len()).unwrap_or(0);
        crate::proto::TpuCoreProgramProto {
            platform_type: 1,
            fingerprint: self.fingerprint.as_bytes().to_vec(),
            compilation_id: 1,
            memory_segments: vec![crate::proto::TpuMemorySegmentSetProto {
                memory_type: 1,
                segments: vec![crate::proto::TpuMemorySegmentProto {
                    segment_type: 1,
                    memory_type: 1,
                    segment: Some(
                        crate::proto::tpu_memory_segment_proto::Segment::InitializedDataRange(
                            crate::proto::TpuMemorySegmentRangeProto {
                                start_byte_offset: 0,
                                size: imem_len_i64,
                            },
                        ),
                    ),
                }],
                initialized_data: raw_vliw_stream.to_vec(),
                hash: self.fingerprint.as_bytes().to_vec(),
                compression: 0,
            }],
            concurrent_program_hbm_stack_size: 0,
            has_concurrent_program: false,
            abi: Some(TpuCoreProgramAbiProto {
                version: 6,
                variant: String::from("default"),
                chip_config: String::from("default"),
                base_abi_version: 167,
                hardware_abi_version: 65_539,
                required_compiler_hash: String::from(
                    "f84a06ff88d86dee0671954c599f27eef7633769f4a591870a4e33a1da19ec33",
                ),
                options: None,
                host_command_abi_versions: Vec::new(),
            }),
            oom_during_compilation: false,
            ..crate::proto::TpuCoreProgramProto::default()
        }
    }

    /// Builds a wire-compatible [`crate::proto::HostOffloadingExecutableProto`] for host-offloaded
    /// computations (`HostExecute` custom calls) associated with this program.
    #[must_use]
    pub fn build_host_offloading_executable_proto(
        &self,
    ) -> crate::proto::HostOffloadingExecutableProto {
        crate::proto::HostOffloadingExecutableProto {
            hlo_module: Some(crate::proto::HloModuleProto {
                name: format!("{}_host_offload", self.name),
                entry_computation_name: String::from("main"),
                id: 1,
                entry_computation_id: 1,
                ..crate::proto::HloModuleProto::default()
            }),
            executable_type: 1,
            aot_compilation_result: None,
        }
    }

    /// Builds a wire-compatible [`crate::proto::FfiExecutionStateProto`] and [`crate::proto::FfiFlatAttributesMapProto`]
    /// capturing XLA FFI custom-call state and backend attributes for this program.
    #[must_use]
    pub fn build_ffi_state_and_attributes(
        &self,
    ) -> (
        crate::proto::FfiExecutionStateProto,
        crate::proto::FfiFlatAttributesMapProto,
    ) {
        let state_proto = crate::proto::FfiExecutionStateProto {
            type_name: format!("xla.ffi.state.{}", self.name),
            state: self.fingerprint.as_bytes().to_vec(),
        };
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            String::from("module_name"),
            crate::proto::FfiFlatAttributeProto {
                value: Some(flat_attribute_proto::Value::Str(
                    self.name.as_bytes().to_vec(),
                )),
            },
        );
        attrs.insert(
            String::from("fingerprint"),
            crate::proto::FfiFlatAttributeProto {
                value: Some(flat_attribute_proto::Value::Str(
                    self.fingerprint.as_bytes().to_vec(),
                )),
            },
        );
        (
            state_proto,
            crate::proto::FfiFlatAttributesMapProto { attrs },
        )
    }

    /// Converts this [`CompiledProgram`] into a full wire-compatible [`TpuExecutableProto`]
    /// (`learning/45eac/research/pjrt/tpu_executable.proto`) containing [`DeepseaExecutableProto`],
    /// [`crate::proto::TpuCoreProgramProto`], and [`TargetArgumentsProto`].
    #[must_use]
    pub fn to_tpu_executable_proto(&self) -> TpuExecutableProto {
        let raw_vliw_stream = self.encode_architectural_imem();
        let tpu_gen = crate::config::global_init_config().generation;
        let target_args = TargetArgumentsProto::from_generation(tpu_gen);
        let core_prog = self.build_core_program_proto(&raw_vliw_stream);
        let compiler_meta = self.build_compiler_metadata_proto();
        let ds_exec = DeepseaExecutableProto {
            core_program: Some(core_prog.clone()),
            compiler_metadata: Some(compiler_meta.clone()),
            sparse_core_program: Some(core_prog),
            sparse_core_compiler_metadata: Some(compiler_meta),
        };
        let hlo_mod = self.hlo_module_proto.clone();
        let hlo_with_cfg = crate::proto::HloModuleProtoWithConfig {
            hlo_module: Some(hlo_mod),
            config: Some(crate::proto::HloModuleConfigProto {
                replica_count: i64::try_from(self.num_replicas).unwrap_or(1),
                num_partitions: i64::try_from(self.num_partitions).unwrap_or(1),
                use_spmd_partitioning: self.num_partitions > 1,
                ..crate::proto::HloModuleConfigProto::default()
            }),
        };
        TpuExecutableProto {
            ds_executable: Some(ds_exec),
            hlo_module_with_config: Some(hlo_with_cfg),
            host_transfers: Vec::new(),
            host_executions: Vec::new(),
            compile_options: CompileOptionsProto::decode_proto(&self.compile_options_bytes).ok(),
            target_arguments: Some(target_args),
            source_uri: format!("embed://{}", self.name),
        }
    }

    /// Synthesizes a wire-compatible [`crate::proto::CompilerMetadataProto`] (`platforms_deepsea.jellyfish.xdb.CompilerMetadata`)
    /// including HLO-to-LLO PC `TrapMetadata` source-location entries.
    #[must_use]
    pub fn build_compiler_metadata_proto(&self) -> crate::proto::CompilerMetadataProto {
        use crate::proto::generated_prost_schemas::platforms_deepsea::jellyfish::xdb::{
            TagAndPc, TrapEntry, TrapMetadata,
        };
        let imem_bytes = u64::try_from(self.instruction_bundles.len() * 64)
            .unwrap_or(64)
            .max(64);
        let trap_entries: Vec<TrapEntry> = self
            .instruction_bundles
            .iter()
            .enumerate()
            .map(|(pc_idx, b)| TrapEntry {
                tag_and_pc: Some(TagAndPc {
                    sequencer_type: Some(0),
                    tag: Some(i64::try_from(pc_idx + 1).unwrap_or(1)),
                    pc: Some(i64::try_from(pc_idx).unwrap_or(0)),
                }),
                display_message: Some(format!("{}:{:?}@pc={pc_idx}", self.name, b.opcode)),
                trap_id: Some(i32::try_from(pc_idx).unwrap_or(0)),
            })
            .collect();
        let end_pc = i64::try_from(self.instruction_bundles.len()).unwrap_or(1);
        crate::proto::CompilerMetadataProto {
            program_id: Some(1),
            replica_count: Some(i64::try_from(self.num_replicas).unwrap_or(1)),
            num_cores_per_replica: Some(i64::try_from(self.num_partitions).unwrap_or(1)),
            is_haltless: Some(false),
            executable_size: Some(i64::try_from(imem_bytes).unwrap_or(256)),
            optimized_hlo_instruction_count: Some(
                i64::try_from(self.instruction_bundles.len()).unwrap_or(1),
            ),
            module_name: Some(self.name.clone()),
            executable_fingerprint: Some(self.fingerprint.as_bytes().to_vec()),
            trap_metadata: Some(TrapMetadata {
                success_tag_and_pc: Some(TagAndPc {
                    sequencer_type: Some(0),
                    tag: Some(0),
                    pc: Some(end_pc),
                }),
                trap_entries,
                compressed_trap_entries: None,
            }),
            ..crate::proto::CompilerMetadataProto::default()
        }
    }

    /// Synthesizes [`crate::proto::HostCommandEnvelopeProto`] descriptors for host-callback,
    /// infeed (`Send`), and outfeed (`Recv`) instructions in this program.
    #[must_use]
    pub fn build_host_command_envelopes(&self) -> Vec<crate::proto::HostCommandEnvelopeProto> {
        let mut envelopes = Vec::new();
        for (idx, b) in self.instruction_bundles.iter().enumerate() {
            let opcode = match b.opcode {
                TpuOpCode::Infeed => Some(1_i32),
                TpuOpCode::Outfeed => Some(2_i32),
                TpuOpCode::CustomCall if b.attrs.custom_call_target.contains("Host") => {
                    Some(16_i32)
                }
                _ => None,
            };
            if let Some(hc_op) = opcode {
                envelopes.push(crate::proto::HostCommandEnvelopeProto {
                    opcode: hc_op,
                    channel_id: i64::try_from(idx + 1).unwrap_or(1),
                    core_id: 0,
                    completion_sync_flag: b.sync_flag_id.max(1),
                    continuation_id: u64::try_from(idx).unwrap_or(0),
                    handler_id: u64::try_from(idx + 1).unwrap_or(1),
                    payload: b.attrs.custom_call_target.as_bytes().to_vec(),
                    handler_name: b.attrs.custom_call_target.clone(),
                });
            }
        }
        envelopes
    }

    /// Builds the `xla.PjRtExecutableAbiVersionProto` stamp for this compiled executable.
    #[must_use]
    pub fn build_executable_abi_version_proto(
        &self,
    ) -> crate::proto::PjRtExecutableAbiVersionProto {
        let tpu_gen = crate::config::global_init_config().generation;
        crate::proto::PjRtExecutableAbiVersionProto {
            platform: 0x5450_5500,
            version: format!("libtpu-abi-v6-{tpu_gen:?}").into_bytes(),
        }
    }

    /// Validates an `xla.PjRtExecutableAbiVersionProto` against the runtime platform ID (`0x5450_5500`).
    ///
    /// # Arguments
    ///
    /// * `abi` - Candidate [`crate::proto::PjRtExecutableAbiVersionProto`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidExecutableFormat`] if `abi.platform` is unsupported or `abi.version` is empty.
    pub fn verify_executable_abi_version(
        abi: &crate::proto::PjRtExecutableAbiVersionProto,
    ) -> LibTpuResult<()> {
        if (abi.platform != 0x5450_5500) | abi.version.is_empty() {
            return Err(LibTpuError::InvalidExecutableFormat {
                reason: format!(
                    "Incompatible PjRtExecutableAbiVersionProto platform={:#x} len={}",
                    abi.platform,
                    abi.version.len()
                ),
            });
        }
        Ok(())
    }

    /// Packages this [`CompiledProgram`] into a relocatable `xla.TpuRelocatableProto` bundling
    /// input/output donation aliases and `PjRtPartialProgramProto` stages.
    #[must_use]
    pub fn to_relocatable_proto(&self) -> crate::proto::TpuRelocatableProto {
        let aliases = vec![crate::proto::tpu_relocatable_proto::ParameterOutputAlias {
            parameter_index: 0,
            output_index: 0,
        }];
        let partial = crate::proto::PjRtPartialProgramProto {
            program: self.serialize(),
            program_format: String::from("ltpuexec"),
            producer_phase: String::from("binary_serialization"),
            consumer_phases: Vec::new(),
            version: String::from(crate::LIBTPU_PYPI_VERSION),
            program_name: self.name.clone(),
        };
        crate::proto::TpuRelocatableProto {
            name: self.name.clone(),
            build_id: self.fingerprint.clone(),
            shape: None,
            call_backend_config: Vec::new(),
            has_side_effects: false,
            parameter_output_aliases: aliases,
            programs: vec![partial],
        }
    }

    /// Synthesizes a wire-compatible [`crate::proto::BarnaCoreExecutableProto`] (`xla.barnacore.BarnaCoreExecutableProto`)
    /// containing compiled [`crate::proto::SparseCoreProgramProto`] and [`crate::proto::SparseCoreDmaDescriptorProto`]
    /// streams for `SparseCore` (`SCv1` / `SCv2`) embedding gather and scatter-update operations.
    #[must_use]
    pub fn build_barnacore_executable_proto(&self) -> crate::proto::BarnaCoreExecutableProto {
        let tpu_gen = crate::config::global_init_config().generation;
        let sc_version = match tpu_gen {
            crate::types::TpuGeneration::V6e | crate::types::TpuGeneration::V7x => 2,
            _ => 1,
        };
        let dma_descriptors: Vec<crate::proto::SparseCoreDmaDescriptorProto> = self
            .instruction_bundles
            .iter()
            .enumerate()
            .map(|(idx, b)| crate::proto::SparseCoreDmaDescriptorProto {
                descriptor_id: u32::try_from(idx).unwrap_or(0),
                hbm_base_offset: u64::try_from(idx * 4096).unwrap_or(0),
                spmem_offset: u32::try_from(idx * 256).unwrap_or(0),
                transfer_bytes: 256,
                sync_flag_id: b.sync_flag_id.max(1),
                is_scatter_write: b.opcode == TpuOpCode::Scatter,
            })
            .collect();
        let tec_bundles = u32::try_from(self.instruction_bundles.len().max(1)).unwrap_or(1);
        let scs_stream = vec![
            0x53,
            0x43,
            0x53,
            u8::try_from(sc_version).unwrap_or(1),
            u8::try_from(tec_bundles & 0xFF).unwrap_or(1),
            u8::try_from((tec_bundles >> 8) & 0xFF).unwrap_or(0),
            0x01,
            0x00,
        ];
        let sc_imem = self.encode_architectural_imem();
        crate::proto::BarnaCoreExecutableProto {
            executable_name: format!("{}_barnacore", self.name),
            sparse_core_version: sc_version,
            channels: vec![crate::proto::BarnaCoreChannelConfigProto {
                channel_id: 0,
                sparse_core_index: 0,
                max_unique_ids_per_partition: 2048,
                max_ids_per_partition: 4096,
                table_name: format!("{}_table0", self.name),
            }],
            programs: vec![crate::proto::SparseCoreProgramProto {
                sparse_core_id: 0,
                instruction_stream: sc_imem,
                spmem_scratchpad_bytes: 1024 * 16,
                dma_descriptors,
                optimizer_opcode: 0,
                tec_bundle_count: tec_bundles,
                scs_control_stream: scs_stream,
                tile_spmem_size_bytes: 4096,
            }],
            table_hbm_total_bytes: 1024 * 64 * 4,
        }
    }

    /// Packages this [`CompiledProgram`] into a relocatable `EM_DEEPSEA` (`0x1AE0`) ELF64 binary container
    /// containing `.shstrtab`, `.note.ABI-tag`, `.text.llo` (64-byte VLIW bundles), `.rodata.cst`
    /// (constant pool), `.rodata.hlo` (`LTPUEXEC` payload), and `.rela.text` relocation records.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_deepsea_elf64(&self) -> Vec<u8> {
        let mut buf = vec![0_u8; 64];
        buf.get_mut(0..7)
            .into_iter()
            .for_each(|s| s.copy_from_slice(b"\x7fELF\x02\x01\x01"));
        buf.get_mut(16..18)
            .into_iter()
            .for_each(|s| s.copy_from_slice(&2_u16.to_le_bytes()));
        buf.get_mut(18..20)
            .into_iter()
            .for_each(|s| s.copy_from_slice(&EM_DEEPSEA.to_le_bytes()));
        buf.get_mut(20..24)
            .into_iter()
            .for_each(|s| s.copy_from_slice(&1_u32.to_le_bytes()));
        buf.get_mut(52..54)
            .into_iter()
            .for_each(|s| s.copy_from_slice(&64_u16.to_le_bytes()));

        let shstrtab =
            b"\0.shstrtab\0.note.ABI-tag\0.text.llo\0.rodata.cst\0.rodata.hlo\0.rela.text\0";
        let shstr_off = buf.len();
        buf.extend_from_slice(shstrtab);

        let note_abi = b"DEEPSEA\0\x01\x00\x00\x00\xA3\x00\x00\x00";
        let note_off = buf.len();
        buf.extend_from_slice(note_abi);

        while (buf.len() & 63) != 0 {
            buf.push(0);
        }
        let imem_bytes = self.encode_architectural_imem();
        let text_off = buf.len();
        buf.extend_from_slice(&imem_bytes);

        let mut cst_bytes = vec![0_u8; 16];
        for b in &self.instruction_bundles {
            cst_bytes.extend_from_slice(&b.attrs.constant_bytes);
        }
        let cst_off = buf.len();
        buf.extend_from_slice(&cst_bytes);

        let hlo_payload = self.serialize();
        let hlo_off = buf.len();
        buf.extend_from_slice(&hlo_payload);

        while (buf.len() & 7) != 0 {
            buf.push(0);
        }
        let rela_off = buf.len();
        let rel_types = [
            R_DEEPSEA_HBM_ABS64,
            R_DEEPSEA_VMEM_OFF32,
            R_DEEPSEA_IMEM_REL32,
            R_DEEPSEA_SYNC_FLAG16,
        ];
        let num_rels = self.instruction_bundles.len().max(rel_types.len());
        for idx in 0..num_rels {
            let bundle_idx = idx % self.instruction_bundles.len().max(1);
            let r_off = u64::try_from(bundle_idx * 64).unwrap_or(0);
            let r_type = rel_types.get(idx % rel_types.len()).copied().unwrap_or(1);
            let r_info = u64::from(r_type);
            let r_addend = i64::try_from(idx * 256).unwrap_or(0);
            buf.extend_from_slice(&r_off.to_le_bytes());
            buf.extend_from_slice(&r_info.to_le_bytes());
            buf.extend_from_slice(&r_addend.to_le_bytes());
        }
        let rela_size = u64::try_from(buf.len().saturating_sub(rela_off)).unwrap_or(0);

        let shoff = buf.len();
        buf.get_mut(40..48).into_iter().for_each(|s| {
            s.copy_from_slice(&u64::try_from(shoff).unwrap_or(0).to_le_bytes());
        });
        buf.get_mut(58..60)
            .into_iter()
            .for_each(|s| s.copy_from_slice(&64_u16.to_le_bytes()));
        buf.get_mut(60..62)
            .into_iter()
            .for_each(|s| s.copy_from_slice(&7_u16.to_le_bytes()));
        buf.get_mut(62..64)
            .into_iter()
            .for_each(|s| s.copy_from_slice(&1_u16.to_le_bytes()));

        let push_shdr = |out: &mut Vec<u8>,
                         name: u32,
                         sh_type: u32,
                         offset: usize,
                         size: usize,
                         entsize: u64| {
            let sh = crate::elf::Elf64Shdr {
                sh_name: name,
                sh_type,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: u64::try_from(offset).unwrap_or(0),
                sh_size: u64::try_from(size).unwrap_or(0),
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 8,
                sh_entsize: entsize,
            };
            out.extend_from_slice(bytemuck::bytes_of(&sh));
        };
        push_shdr(&mut buf, 0, 0, 0, 0, 0);
        push_shdr(&mut buf, 1, 3, shstr_off, shstrtab.len(), 0);
        push_shdr(&mut buf, 11, 7, note_off, note_abi.len(), 0);
        push_shdr(&mut buf, 25, 1, text_off, imem_bytes.len(), 64);
        push_shdr(&mut buf, 35, 1, cst_off, cst_bytes.len(), 0);
        push_shdr(&mut buf, 47, 1, hlo_off, hlo_payload.len(), 0);
        push_shdr(
            &mut buf,
            59,
            4,
            rela_off,
            usize::try_from(rela_size).unwrap_or(0),
            24,
        );
        buf
    }

    /// Parses and relocates an `EM_DEEPSEA` (`0x1AE0`) ELF64 binary container produced by
    /// [`Self::to_deepsea_elf64`], applying `.rela.text` relocations against `hbm_base` and `vmem_base`.
    ///
    /// # Arguments
    ///
    /// * `elf_bytes` - Raw ELF64 container bytes.
    /// * `hbm_base` - Base device HBM virtual address for `R_DEEPSEA_HBM_ABS64` relocations.
    /// * `vmem_base` - Base core VMEM scratchpad offset for `R_DEEPSEA_VMEM_OFF32` relocations.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `elf_bytes` is not a valid `EM_DEEPSEA` ELF64 image.
    pub fn from_deepsea_elf64(
        elf_bytes: &[u8],
        hbm_base: u64,
        vmem_base: u32,
    ) -> LibTpuResult<DeepseaRelocatedArtifact> {
        let ehdr = crate::elf::parse_elf64_ehdr_zero_copy(elf_bytes).map_err(|e| {
            LibTpuError::DeserializationFailed {
                reason: format!("Invalid EM_DEEPSEA ELF64 header: {e}"),
            }
        })?;
        if ehdr.e_machine != EM_DEEPSEA {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!(
                    "Expected EM_DEEPSEA (0x{EM_DEEPSEA:04x}), got 0x{:04x}",
                    ehdr.e_machine
                ),
            });
        }
        let shoff = usize::try_from(ehdr.e_shoff).unwrap_or(0);
        let shnum = usize::from(ehdr.e_shnum);
        let mut shdrs: Vec<crate::elf::Elf64Shdr> = Vec::with_capacity(shnum);
        for idx in 0..shnum {
            let start = shoff.saturating_add(idx * 64);
            let slice = elf_bytes
                .get(start..start.saturating_add(64))
                .ok_or_else(|| LibTpuError::DeserializationFailed {
                    reason: String::from("Truncated ELF64 section header table"),
                })?;
            shdrs.push(bytemuck::pod_read_unaligned(slice));
        }
        let sec_slice = |sec_idx: usize| -> &[u8] {
            let off = shdrs
                .get(sec_idx)
                .map_or(0, |sh| usize::try_from(sh.sh_offset).unwrap_or(0));
            let sz = shdrs
                .get(sec_idx)
                .map_or(0, |sh| usize::try_from(sh.sh_size).unwrap_or(0));
            elf_bytes.get(off..off.saturating_add(sz)).unwrap_or(&[])
        };
        let mut relocated_imem = sec_slice(3).to_vec();
        let constant_pool = sec_slice(4).to_vec();
        let hlo_bytes = sec_slice(5);
        let rela_bytes = sec_slice(6);
        let mut applied_relocations = 0_usize;
        for chunk in rela_bytes.chunks_exact(24) {
            let rela: crate::elf::Elf64Rela = bytemuck::pod_read_unaligned(chunk);
            let r_off = usize::try_from(rela.r_offset).unwrap_or(0);
            #[allow(clippy::cast_possible_truncation)]
            let r_type = (rela.r_info & 0xFFFF_FFFF) as u32;
            #[allow(clippy::cast_sign_loss)]
            let addend_u64 = rela.r_addend as u64;
            match r_type {
                R_DEEPSEA_HBM_ABS64 => {
                    let patched = hbm_base.wrapping_add(addend_u64);
                    relocated_imem
                        .get_mut(r_off.saturating_add(48)..r_off.saturating_add(56))
                        .into_iter()
                        .for_each(|slot| slot.copy_from_slice(&patched.to_le_bytes()));
                }
                R_DEEPSEA_VMEM_OFF32 | R_DEEPSEA_IMEM_REL32 => {
                    #[allow(clippy::cast_possible_truncation)]
                    let patched = vmem_base.wrapping_add(addend_u64 as u32);
                    relocated_imem
                        .get_mut(r_off.saturating_add(56)..r_off.saturating_add(60))
                        .into_iter()
                        .for_each(|slot| slot.copy_from_slice(&patched.to_le_bytes()));
                }
                _ => {}
            }
            applied_relocations += 1;
        }
        let program = Self::deserialize(hlo_bytes)?;
        Ok(DeepseaRelocatedArtifact {
            relocated_imem,
            constant_pool,
            applied_relocations,
            program,
        })
    }

    /// Serializes this [`CompiledProgram`] into a self-contained binary payload with magic header.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let hdr = ExecutableBinaryHeader {
            magic: *EXECUTABLE_MAGIC,
            version_le: EXECUTABLE_FORMAT_VERSION.to_le(),
        };
        out.extend_from_slice(bytemuck::bytes_of(&hdr));
        out.push(u8::from(self.opcode));

        let fmt_bytes = self.format.as_bytes();
        out.extend_from_slice(&u32::try_from(fmt_bytes.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(fmt_bytes);

        out.extend_from_slice(
            &u32::try_from(self.source_bytes.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.source_bytes);

        out.extend_from_slice(
            &u32::try_from(self.compile_options_bytes.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.compile_options_bytes);
        out
    }

    /// Packages `TpuExecutableProto` inside a 64-byte `RiegeliRecordCodec` container protected by
    /// `HighwayHash64` header and payload checksums (`serialize_riegeli_executable`).
    #[must_use]
    pub fn serialize_riegeli_executable(&self) -> Vec<u8> {
        use crate::proto::ProtoSchemaMessage as _;
        let tpu_exec_bytes = self.to_tpu_executable_proto().encode_proto();
        crate::proto::RiegeliRecordCodec::wrap_record(&tpu_exec_bytes)
    }

    /// Verifies `HighwayHash64` header & payload checksums on a Riegeli-framed `TpuExecutableProto`
    /// and reconstructs the [`CompiledProgram`].
    ///
    /// # Arguments
    ///
    /// * `bytes` - Riegeli-framed executable payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if Riegeli `HighwayHash64` verification or `TpuExecutableProto` decoding fails.
    pub fn deserialize_riegeli_executable(bytes: &[u8]) -> LibTpuResult<Self> {
        let (_, unwrapped) = crate::proto::RiegeliRecordCodec::unwrap_record_decompressed(bytes)?;
        Self::deserialize(&unwrapped)
    }

    /// Reads a little-endian `u32` from `cursor` via `bytemuck` and advances the slice.
    ///
    /// # Arguments
    ///
    /// * `cursor` - Mutable reference to the remaining byte slice.
    /// * `field` - Field name for error reporting.
    fn read_u32(cursor: &mut &[u8], field: &'static str) -> LibTpuResult<u32> {
        let Some((head, rest)) = cursor.split_at_checked(4) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Truncated u32 field '{field}'"),
            });
        };
        *cursor = rest;
        let raw_u32: u32 = bytemuck::pod_read_unaligned(head);
        Ok(u32::from_le(raw_u32))
    }

    /// Reads a length-prefixed byte slice from `cursor` and advances the slice.
    ///
    /// # Arguments
    ///
    /// * `cursor` - Mutable reference to the remaining byte slice.
    /// * `field` - Field name for error reporting.
    fn read_bytes<'a>(cursor: &mut &'a [u8], field: &'static str) -> LibTpuResult<&'a [u8]> {
        let len = usize::try_from(Self::read_u32(cursor, field)?).unwrap_or(usize::MAX);
        let Some((payload, rest)) = cursor.split_at_checked(len) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Truncated payload for '{field}' (expected {len} bytes)"),
            });
        };
        *cursor = rest;
        Ok(payload)
    }

    /// Deserializes a [`CompiledProgram`] from bytes produced by [`CompiledProgram::serialize`]
    /// or from a serialized [`TpuSerializedExecutableProto`].
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized executable byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the magic header, version, or payload lengths are invalid.
    pub fn deserialize(bytes: &[u8]) -> LibTpuResult<Self> {
        let Some((magic, _)) = bytes.split_at_checked(8) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Payload too short ({} bytes)", bytes.len()),
            });
        };
        if magic != EXECUTABLE_MAGIC {
            if bytes.starts_with(&crate::proto::RIEGELI_BLOCK_MAGIC) {
                let (_, inner) =
                    crate::proto::RiegeliRecordCodec::unwrap_record_decompressed(bytes)?;
                return Self::deserialize(&inner);
            }
            if bytes.starts_with(b"\x7fELF") {
                return Self::from_deepsea_elf64(bytes, 0x1000_0000, 0x2000).map(|r| r.program);
            }
            if let Ok(tpu_exec) = TpuExecutableProto::decode_proto(bytes) {
                if let Some(hlo_with_cfg) = tpu_exec.hlo_module_with_config {
                    if let Some(hlo_mod) = hlo_with_cfg.hlo_module {
                        let opts_bytes = tpu_exec
                            .compile_options
                            .as_ref()
                            .map_or_else(Vec::new, CompileOptionsProto::encode_proto);
                        return Self::compile("hlo", &hlo_mod.encode_proto(), &opts_bytes);
                    }
                }
            }
            if let Ok(proto) = TpuSerializedExecutableProto::decode(bytes) {
                if !proto.hlo_module_bytes.is_empty() {
                    let raw_op = proto.instruction_opcodes.first().copied().unwrap_or(0);
                    let op_byte = u8::try_from(raw_op).unwrap_or(255);
                    let opcode = TpuOpCode::from_byte(op_byte)?;
                    let mut compiled = Self::compile(
                        "mlir",
                        &proto.hlo_module_bytes,
                        &proto.compile_options_bytes,
                    )?;
                    compiled.opcode = opcode;
                    return Ok(compiled);
                }
            }
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Invalid executable magic header"),
            });
        }
        let Some((hdr_bytes, mut cursor)) =
            bytes.split_at_checked(size_of::<ExecutableBinaryHeader>())
        else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Truncated executable binary header"),
            });
        };
        let hdr: ExecutableBinaryHeader = bytemuck::pod_read_unaligned(hdr_bytes);
        let version = u32::from_le(hdr.version_le);
        if version != EXECUTABLE_FORMAT_VERSION {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Unsupported executable format version {version}"),
            });
        }
        let Some((&opcode_byte, after_op)) = cursor.split_first() else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Missing opcode byte"),
            });
        };
        cursor = after_op;
        let opcode = TpuOpCode::from_byte(opcode_byte)?;

        let fmt_bytes = Self::read_bytes(&mut cursor, "format")?;
        let format = String::from(
            std::str::from_utf8(fmt_bytes).map_err(|e| LibTpuError::from(Utf8ErrorWrapper(e)))?,
        );

        let source_bytes = Self::read_bytes(&mut cursor, "source_bytes")?;
        let compile_options = Self::read_bytes(&mut cursor, "compile_options")?;

        let mut compiled = Self::compile(&format, source_bytes, compile_options)?;
        compiled.opcode = opcode;
        Ok(compiled)
    }

    /// Returns the generated machine code size as a strongly-typed [`ByteSize`],
    /// clamped to the 256-byte IMEM hardware minimum granule size.
    #[must_use]
    pub fn generated_code_size(&self) -> ByteSize {
        let clamped_i64 = self.memory_stats.generated_code_size_in_bytes.max(256);
        ByteSize(usize::try_from(clamped_i64).unwrap_or(usize::MAX))
    }
}
