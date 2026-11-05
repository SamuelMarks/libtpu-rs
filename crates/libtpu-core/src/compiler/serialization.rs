//! Wire-compatible `LTPUEXEC` binary and Protobuf (`TpuExecutableProto`, `DeepseaExecutableProto`) serialization.

use super::CompiledProgram;
use super::opcode::TpuOpCode;
use crate::error::{LibTpuError, LibTpuResult, Utf8ErrorWrapper};
use crate::proto::{
    CompileOptionsProto, DeepseaConstantPoolEntryProto, DeepseaExecutableProto,
    DeepseaOverlaySectionProto, DeepseaRelocationEntryProto, DeepseaSyncFlagInitProto,
    IsaBundleSlotProto, IsaCoreProgramHeaderProto, IsaProgramProto, IsaSymbolTableEntryProto,
    IsaVliwBundleProto, LayoutProto, ShapeProto, TargetArgumentsProto, TileProto,
    TpuExecutableProto, TpuSerializedExecutableProto, sha256_fingerprint_u64,
};
use crate::types::{ByteSize, ElementType};

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
    /// Relocated 64-byte `TensorCoreVliwBundle` `IMEM` binary stream (`.text.llo`).
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
    pub fn build_buffer_assignment_proto(&self) -> crate::proto::BufferAssignmentProto {
        let mut allocations = Vec::new();
        let mut next_idx: i64 = 0;

        for (param_idx, shape) in self.parameter_shapes.iter().enumerate() {
            let size = i64::try_from(shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            allocations.push(crate::proto::BufferAllocationProto {
                index: next_idx,
                size,
                is_thread_local: false,
                is_tuple: false,
                is_entry_computation_parameter: true,
                parameter_number: i64::try_from(param_idx).unwrap_or(0),
                is_constant: false,
                maybe_live_out: false,
                color: 0,
                assigned: vec![crate::proto::BufferAllocationSliceProto {
                    offset: 0,
                    size,
                    buffer_allocation_index: next_idx,
                    element_type: shape.element_type.as_raw(),
                }],
            });
            next_idx += 1;
        }

        let vmem_size = self.memory_stats.temp_size_in_bytes.max(256);
        allocations.push(crate::proto::BufferAllocationProto {
            index: next_idx,
            size: vmem_size,
            is_thread_local: false,
            is_tuple: false,
            is_entry_computation_parameter: false,
            parameter_number: 0,
            is_constant: false,
            maybe_live_out: false,
            color: 1,
            assigned: vec![crate::proto::BufferAllocationSliceProto {
                offset: 0,
                size: vmem_size,
                buffer_allocation_index: next_idx,
                element_type: ElementType::F32.as_raw(),
            }],
        });
        next_idx += 1;

        for out_shape in &self.output_shapes {
            let size = i64::try_from(out_shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            allocations.push(crate::proto::BufferAllocationProto {
                index: next_idx,
                size,
                is_thread_local: false,
                is_tuple: false,
                is_entry_computation_parameter: false,
                parameter_number: 0,
                is_constant: false,
                maybe_live_out: true,
                color: 0,
                assigned: vec![crate::proto::BufferAllocationSliceProto {
                    offset: 0,
                    size,
                    buffer_allocation_index: next_idx,
                    element_type: out_shape.element_type.as_raw(),
                }],
            });
            next_idx += 1;
        }

        crate::proto::BufferAssignmentProto {
            buffer_allocations: allocations,
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
                layout: Some(LayoutProto {
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
                }),
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

    /// Builds the structured [`IsaProgramProto`] for this [`CompiledProgram`], populating
    /// only the active functional unit slots (`scalar_s0`, `vector_v0`, `matrix_m0`, `unary_xlu`, `dma_d0`)
    /// for each lowered [`super::opcode::TpuInstructionBundle`].
    #[must_use]
    pub fn build_isa_program_proto(&self) -> IsaProgramProto {
        let raw_vliw_stream = self.encode_architectural_imem();
        let mut structured_bundles: Vec<IsaVliwBundleProto> =
            Vec::with_capacity(self.instruction_bundles.len());
        for b in &self.instruction_bundles {
            let src0 = u32::try_from(b.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
            let src1 = u32::try_from(b.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
            let dst_reg = u32::try_from(b.output_slot).unwrap_or(0);
            let op_code = u32::from(u8::from(b.opcode));
            let imm = i64::from(b.sync_flag_id);
            let is_mxu = b.opcode == TpuOpCode::DotGeneralF32;
            let is_xunary = b.opcode.is_transcendental();
            let is_dma = b.opcode.has_side_effects();

            let scalar_s0 = Some(IsaBundleSlotProto {
                unit_kind: 0,
                opcode: 1,
                dst_reg: 0,
                src0_reg: u32::try_from(b.attrs.iota_dim).unwrap_or(0),
                src1_reg: u32::try_from(b.attrs.concat_dim).unwrap_or(0),
                predicate_reg: 0,
                immediate: i64::from(b.pc_offset),
            });
            let vector_v0 = Some(IsaBundleSlotProto {
                unit_kind: 1,
                opcode: op_code,
                dst_reg,
                src0_reg: src0,
                src1_reg: src1,
                predicate_reg: 0,
                immediate: imm,
            });
            let matrix_m0 = is_mxu.then_some(IsaBundleSlotProto {
                unit_kind: 2,
                opcode: op_code,
                dst_reg,
                src0_reg: src0,
                src1_reg: src1,
                predicate_reg: 0,
                immediate: imm,
            });
            let unary_xlu = is_xunary.then_some(IsaBundleSlotProto {
                unit_kind: 3,
                opcode: op_code,
                dst_reg,
                src0_reg: src0,
                src1_reg: 0,
                predicate_reg: 0,
                immediate: imm,
            });
            let dma_d0 = is_dma.then_some(IsaBundleSlotProto {
                unit_kind: 4,
                opcode: op_code,
                dst_reg,
                src0_reg: src0,
                src1_reg: src1,
                predicate_reg: 0,
                immediate: imm,
            });
            structured_bundles.push(IsaVliwBundleProto {
                pc_index: b.pc_offset,
                scalar_s0,
                scalar_s1: None,
                vector_v0,
                vector_v1: None,
                matrix_m0,
                matrix_m1: None,
                unary_xlu,
                dma_d0,
                llo_annotation: format!("llo.{:?}", b.opcode),
            });
        }
        let imem_bytes = u64::try_from(raw_vliw_stream.len()).unwrap_or(0);
        IsaProgramProto {
            isa_architecture: String::from("vfc"),
            isa_abi_version: 163,
            core_programs: vec![IsaCoreProgramHeaderProto {
                core_index: 0,
                core_type: 0,
                entry_pc: 0,
                bundle_count: u64::try_from(structured_bundles.len()).unwrap_or(0),
                raw_vliw_stream,
                structured_bundles,
            }],
            symbols: vec![IsaSymbolTableEntryProto {
                name: self.name.clone(),
                pc_offset: 0,
                size_bytes: imem_bytes,
            }],
            fingerprint: sha256_fingerprint_u64(self.fingerprint.as_bytes()),
        }
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
                    initialized_data_range: Some(crate::proto::TpuMemorySegmentRangeProto {
                        offset: 0,
                        size: imem_len_i64,
                    }),
                    segment_size_in_bytes: imem_len_i64,
                }],
                initialized_data: raw_vliw_stream.to_vec(),
                hash: self.fingerprint.as_bytes().to_vec(),
                compression: 0,
            }],
            concurrent_program_hbm_stack_size: 0,
            has_concurrent_program: false,
            abi: Some(crate::proto::TpuCoreProgramAbiProto {
                version: 6,
                variant: String::from("default"),
                chip_config: String::from("default"),
                base_abi_version: 163,
                hardware_abi_version: 65_539,
                required_compiler_hash: String::from(
                    "00ef0b06b559d6911774f1854abee45cdcb62dc50b6c9535921e52231188d011",
                ),
            }),
            oom_during_compilation: false,
        }
    }

    /// Builds wire-compatible [`DeepseaRelocationEntryProto`] and [`DeepseaConstantPoolEntryProto`]
    /// tables from the lowered instruction bundles.
    #[must_use]
    pub fn build_deepsea_relocations_and_constants(
        &self,
    ) -> (
        Vec<DeepseaRelocationEntryProto>,
        Vec<DeepseaConstantPoolEntryProto>,
    ) {
        let relocations = self
            .instruction_bundles
            .iter()
            .enumerate()
            .map(|(idx, b)| DeepseaRelocationEntryProto {
                bundle_offset_bytes: u64::try_from(idx * 64).unwrap_or(0),
                slot_index: 1,
                buffer_index: u32::try_from(b.output_slot).unwrap_or(0),
                addend_bytes: 0,
                memory_space: 0,
            })
            .collect();
        let constants = self
            .instruction_bundles
            .iter()
            .enumerate()
            .map(|(idx, b)| DeepseaConstantPoolEntryProto {
                constant_id: u32::try_from(idx).unwrap_or(0),
                memory_space: 1,
                offset_bytes: u64::try_from(idx * 256).unwrap_or(0),
                payload: b.attrs.constant_bytes.clone(),
            })
            .collect();
        (relocations, constants)
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
                str: self.name.as_bytes().to_vec(),
            },
        );
        attrs.insert(
            String::from("fingerprint"),
            crate::proto::FfiFlatAttributeProto {
                str: self.fingerprint.as_bytes().to_vec(),
            },
        );
        (
            state_proto,
            crate::proto::FfiFlatAttributesMapProto { attrs },
        )
    }

    /// Converts this [`CompiledProgram`] into a full wire-compatible [`TpuExecutableProto`]
    /// containing [`DeepseaExecutableProto`], [`IsaProgramProto`], and [`TargetArgumentsProto`].
    #[must_use]
    pub fn to_tpu_executable_proto(&self) -> TpuExecutableProto {
        let raw_vliw_stream = self.encode_architectural_imem();
        let imem_bytes = u64::try_from(raw_vliw_stream.len()).unwrap_or(0);
        let isa_prog = self.build_isa_program_proto();
        let tpu_gen = crate::config::global_init_config().generation;
        let tpu_version = i32::from(tpu_gen);
        let chip_variant = tpu_gen.to_string();
        let hbm_cap = i64::try_from(tpu_gen.hbm_capacity_bytes().0).unwrap_or(17_179_869_184);
        let vmem_cap =
            i64::try_from(tpu_gen.vmem_capacity_bytes_per_core().0).unwrap_or(16_777_216);
        let target_args = TargetArgumentsProto {
            tpu_version,
            chip_variant,
            clock_rate_mhz: 1750,
            hbm_capacity_bytes: hbm_cap,
            vmem_capacity_bytes: vmem_cap,
            smem_capacity_bytes: 2 * 1024 * 1024,
            cmem_capacity_bytes: 128 * 1024 * 1024,
            tensor_cores_per_chip: i32::try_from(tpu_gen.logical_devices_per_chip()).unwrap_or(2),
            sparse_cores_per_chip: 4,
            mxu_dimension: 128,
            ici_bandwidth_bps: 4_800_000_000_000,
        };
        let core_prog = self.build_core_program_proto(&raw_vliw_stream);
        let (relocations, constants) = self.build_deepsea_relocations_and_constants();
        let ds_exec = DeepseaExecutableProto {
            module_name: self.name.clone(),
            isa_program: Some(isa_prog),
            overlays: vec![DeepseaOverlaySectionProto {
                overlay_id: 0,
                symbol_name: format!(".text.{}", self.name),
                imem_offset_bytes: 0,
                byte_size: imem_bytes,
                encoded_bundles: raw_vliw_stream,
                core_type: 0,
            }],
            relocations,
            constants,
            sync_flags: vec![DeepseaSyncFlagInitProto::default()],
            target_arguments: Some(target_args.clone()),
            vmem_allocation_bytes: self.memory_stats.temp_size_in_bytes,
            smem_allocation_bytes: 4096,
            cmem_allocation_bytes: 0,
            hbm_scratchpad_bytes: self.memory_stats.generated_code_size_in_bytes,
            core_program: Some(core_prog.clone()),
            sparse_core_program: Some(core_prog),
        };
        TpuExecutableProto {
            ds_executable: Some(ds_exec),
            hlo_module_with_config: None,
            compile_options: CompileOptionsProto::decode(&self.compile_options_bytes).ok(),
            target_arguments: Some(target_args),
            legacy_executable: Some(self.to_executable_proto()),
            source_uri: format!("embed://{}", self.name),
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
            if bytes.starts_with(b"\x7fELF") {
                return Self::from_deepsea_elf64(bytes, 0x1000_0000, 0x2000).map(|r| r.program);
            }
            let candidate = TpuExecutableProto::decode(bytes)
                .ok()
                .and_then(|tpu_exec| tpu_exec.legacy_executable)
                .or_else(|| TpuSerializedExecutableProto::decode(bytes).ok());
            if let Some(proto) = candidate {
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
