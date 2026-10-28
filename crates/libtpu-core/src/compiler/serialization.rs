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

    /// Converts this [`CompiledProgram`] into a full wire-compatible [`TpuExecutableProto`]
    /// containing [`DeepseaExecutableProto`], [`IsaProgramProto`], and [`TargetArgumentsProto`].
    #[must_use]
    pub fn to_tpu_executable_proto(&self) -> TpuExecutableProto {
        let raw_vliw_stream = self.encode_architectural_imem();
        let structured_bundles: Vec<IsaVliwBundleProto> = self
            .instruction_bundles
            .iter()
            .map(|b| {
                let src0 =
                    u32::try_from(b.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
                let src1 = u32::try_from(b.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
                let slot = IsaBundleSlotProto {
                    unit_kind: 2,
                    opcode: u32::from(u8::from(b.opcode)),
                    dst_reg: u32::try_from(b.output_slot).unwrap_or(0),
                    src0_reg: src0,
                    src1_reg: src1,
                    predicate_reg: 0,
                    immediate: i64::from(b.sync_flag_id),
                };
                IsaVliwBundleProto {
                    pc_index: b.pc_offset,
                    scalar_s0: Some(slot.clone()),
                    scalar_s1: None,
                    vector_v0: Some(slot.clone()),
                    vector_v1: None,
                    matrix_m0: Some(slot.clone()),
                    matrix_m1: None,
                    unary_xlu: Some(slot.clone()),
                    dma_d0: Some(slot),
                    llo_annotation: format!("llo.{:?}", b.opcode),
                }
            })
            .collect();
        let imem_bytes = u64::try_from(raw_vliw_stream.len()).unwrap_or(0);
        let isa_prog = IsaProgramProto {
            isa_architecture: String::from("vfc"),
            isa_abi_version: 163,
            core_programs: vec![IsaCoreProgramHeaderProto {
                core_index: 0,
                core_type: 0,
                entry_pc: 0,
                bundle_count: u64::try_from(structured_bundles.len()).unwrap_or(0),
                raw_vliw_stream: raw_vliw_stream.clone(),
                structured_bundles,
            }],
            symbols: vec![IsaSymbolTableEntryProto {
                name: self.name.clone(),
                pc_offset: 0,
                size_bytes: imem_bytes,
            }],
            fingerprint: sha256_fingerprint_u64(self.fingerprint.as_bytes()),
        };
        let target_args = TargetArgumentsProto {
            tpu_version: 6,
            chip_variant: String::from("ghostfish"),
            clock_rate_mhz: 1750,
            hbm_capacity_bytes: 95 * 1024 * 1024 * 1024,
            vmem_capacity_bytes: 64 * 1024 * 1024,
            smem_capacity_bytes: 2 * 1024 * 1024,
            cmem_capacity_bytes: 128 * 1024 * 1024,
            tensor_cores_per_chip: 2,
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

    /// Returns the generated machine code size as a strongly-typed [`ByteSize`].
    #[must_use]
    pub fn generated_code_size(&self) -> ByteSize {
        ByteSize(usize::try_from(self.memory_stats.generated_code_size_in_bytes).unwrap_or(256))
    }
}
