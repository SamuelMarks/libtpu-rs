//! HLO and `StableHLO` compiler, recursive-descent parser, and VLIW bundle emitter.
//!
//! Provides the DAG lowerer, fingerprint generator, cost analyzer, phased compiler, and
//! wire-compatible `TpuSerializedExecutableProto` serializer for `PJRT_Client_Compile`,
//! `PJRT_Compile`, `PJRT_Executable_Serialize`, `PJRT_Executable_DeserializeAndLoad`, and
//! `PJRT_PhaseCompile_Extension`.
//!
//! Organized into focused submodules:
//! - [`opcode`]: [`TpuOpCode`], [`InstructionAttrs`], and [`TpuInstructionBundle`].
//! - [`vliw`]: Architectural 64-byte [`TensorCoreVliwBundle`] and 16-byte [`VliwInstructionWord`] encoders.
//! - [`mlir_bytecode`]: MLIR Bytecode (`ML\xefR`) container decoder, encoder, and `PrefixVarInt` codecs.
//! - [`lexer`]: `winnow`-backed [`HloLexer`], [`HloParser`], and [`MosaicTargetMetadata`] parser.
//! - [`passes`]: Compiler optimization passes, VMEM liveness intervals, cost analysis, and DAG scheduling.
//! - [`serialization`]: `LTPUEXEC` binary format and `DeepSea` / `TpuExecutableProto` wire serialization.

pub mod lexer;
pub mod mlir_bytecode;
pub mod opcode;
pub mod passes;
pub mod serialization;
pub mod vliw;

pub use self::lexer::{HloLexer, HloParser, HloToken, MosaicTargetMetadata};
pub use self::mlir_bytecode::{
    MLIR_BYTECODE_MAGIC, MlirBytecodeHeader, MlirBytecodeSection, decode_mlir_prefix_varint,
    encode_mlir_prefix_varint,
};
pub use self::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
pub use self::passes::{
    CompiledMemoryStats, CostAnalysis, HloDagAnalysisReport, RegisterLivenessInterval,
    analyze_hlo_computation_dag,
};
pub use self::serialization::{
    EXECUTABLE_FORMAT_VERSION, EXECUTABLE_MAGIC, ExecutableBinaryHeader,
};
pub use self::vliw::{
    AsyncDmaSlot, ExtendedUnarySlot, MatrixMxuSlot, ScalarAluSlot, TensorCoreVliwBundle,
    VectorAluSlot, VliwBundleHeader, VliwInstructionWord,
};

use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{CompileOptionsProto, HloModuleProto, HloModuleProtoWithConfig};
use crate::types::{MemoryKind, Shape};

/// Canonical ordered compilation phases supported by `PJRT_PhaseCompile_Extension`.
pub const COMPILATION_PHASE_NAMES: [&str; 5] = [
    "hlo_verification",
    "tpu_layout_assignment",
    "tpu_memory_planning",
    "vliw_bundle_emission",
    "binary_serialization",
];

/// Validated and lowered TPU executable representation.
#[derive(Debug, Clone)]
pub struct CompiledProgram {
    /// Module name extracted from the HLO or `StableHLO` program.
    pub name: String,
    /// Program format (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    pub format: String,
    /// Original program source or bytecode payload.
    pub source_bytes: Vec<u8>,
    /// Number of data-parallel replicas.
    pub num_replicas: usize,
    /// Number of SPMD model-parallel partitions.
    pub num_partitions: usize,
    /// Primary lowered TPU operation opcode.
    pub opcode: TpuOpCode,
    /// Lowered VLIW instruction bundle sequence.
    pub instruction_bundles: Vec<TpuInstructionBundle>,
    /// Register slot indices corresponding to each output tensor in `output_shapes`.
    pub output_slots: Vec<usize>,
    /// Input parameter shapes.
    pub parameter_shapes: Vec<Shape>,
    /// Output tensor shapes.
    pub output_shapes: Vec<Shape>,
    /// Memory kinds for each input parameter.
    pub parameter_memory_kinds: Vec<MemoryKind>,
    /// Memory kinds for each output tensor.
    pub output_memory_kinds: Vec<MemoryKind>,
    /// Deterministic 32-character hexadecimal fingerprint.
    pub fingerprint: String,
    /// Serialized compile options bytes.
    pub compile_options_bytes: Vec<u8>,
    /// Cost analysis metrics.
    pub cost_analysis: CostAnalysis,
    /// Compiled memory statistics.
    pub memory_stats: CompiledMemoryStats,
}

impl CompiledProgram {
    /// Compiles an HLO (`HloModuleProto` or textual HLO) or `StableHLO` MLIR program into a [`CompiledProgram`].
    ///
    /// # Arguments
    ///
    /// * `format` - Program format identifier (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    /// * `code` - Raw program bytes (textual MLIR/HLO or binary `HloModuleProto`).
    /// * `compile_options` - Raw serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if `code` is empty, `format` is unsupported,
    /// `compile_options` is malformed, or an explicit syntax error marker is present.
    #[allow(clippy::too_many_lines)]
    pub fn compile(format: &str, code: &[u8], compile_options: &[u8]) -> LibTpuResult<Self> {
        let fmt_lower = format.trim().to_ascii_lowercase();
        if fmt_lower != "mlir" && fmt_lower != "hlo" && fmt_lower != "hlo_with_config" {
            return Err(LibTpuError::CompilationFailed {
                reason: format!("Unsupported program format '{format}'"),
            });
        }
        if code.is_empty() {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("Program code payload cannot be empty"),
            });
        }

        let parsed_options = if compile_options.is_empty() {
            CompileOptionsProto::default()
        } else {
            CompileOptionsProto::decode(compile_options)?
        };

        let num_replicas = parsed_options
            .executable_build_options
            .as_ref()
            .and_then(|b| usize::try_from(b.num_replicas).ok())
            .unwrap_or(0)
            .max(1);
        let num_partitions = parsed_options
            .executable_build_options
            .as_ref()
            .and_then(|b| usize::try_from(b.num_partitions).ok())
            .unwrap_or(0)
            .max(1);

        let stripped_code = code.strip_prefix(MLIR_BYTECODE_MAGIC).unwrap_or(code);
        let text = if let Some(bc_hdr) = MlirBytecodeHeader::decode(code) {
            if bc_hdr.string_table.is_empty() {
                String::from_utf8_lossy(stripped_code).into_owned()
            } else {
                bc_hdr.to_synthesized_mlir_text()
            }
        } else {
            String::from_utf8_lossy(stripped_code).into_owned()
        };
        if text.contains("SYNTAX_ERROR") {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("Explicit SYNTAX_ERROR marker encountered in program"),
            });
        }

        let decoded_hlo = HloModuleProto::decode(stripped_code)
            .ok()
            .filter(|m| !m.computations.is_empty())
            .or_else(|| {
                HloModuleProtoWithConfig::decode(stripped_code)
                    .ok()
                    .and_then(|w| w.hlo_module)
                    .filter(|m| !m.computations.is_empty())
            });
        let (name, opcode, parameter_shapes, output_shapes, bundles, output_slots) =
            if let Some(hlo_proto) = decoded_hlo {
                Self::lower_from_hlo_proto(&hlo_proto)?
            } else {
                let extracted_name = Self::extract_module_name(&text);
                let (op, p_shapes, o_shapes, b, o_slots) = Self::parse_textual_program(&text);
                (extracted_name, op, p_shapes, o_shapes, b, o_slots)
            };

        let mut arg_bytes: i64 = 0;
        for shape in &parameter_shapes {
            let shape_bytes = i64::try_from(shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            arg_bytes = arg_bytes.saturating_add(shape_bytes);
        }
        let parameter_memory_kinds = vec![MemoryKind::DeviceHbm; parameter_shapes.len()];

        let mut out_bytes: i64 = 0;
        for shape in &output_shapes {
            let shape_bytes = i64::try_from(shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            out_bytes = out_bytes.saturating_add(shape_bytes);
        }
        let output_memory_kinds = vec![MemoryKind::DeviceHbm; output_shapes.len()];

        let fingerprint = Self::compute_fingerprint(&fmt_lower, code, compile_options);
        let aligned_code_len = code.len().max(256).div_ceil(256) * 256;
        let generated_code_size = i64::try_from(aligned_code_len).unwrap_or(256);

        let out_bytes_u16 = u16::try_from(out_bytes.max(256)).unwrap_or(256);
        let arg_plus_out_u16 = u16::try_from((arg_bytes + out_bytes).max(256)).unwrap_or(512);
        let flops = if opcode == TpuOpCode::Identity {
            0.0
        } else if opcode == TpuOpCode::DotGeneralF32 {
            f32::from(out_bytes_u16.saturating_mul(2))
        } else {
            f32::from(out_bytes_u16 / 4)
        };
        let transcendentals = if opcode.is_transcendental() {
            f32::from(out_bytes_u16 / 4)
        } else {
            0.0
        };
        let bytes_accessed = f32::from(arg_plus_out_u16);

        let mut compiled = Self {
            name,
            format: fmt_lower,
            source_bytes: code.to_vec(),
            num_replicas,
            num_partitions,
            opcode,
            instruction_bundles: bundles,
            output_slots,
            parameter_shapes,
            output_shapes,
            parameter_memory_kinds,
            output_memory_kinds,
            fingerprint,
            compile_options_bytes: compile_options.to_vec(),
            cost_analysis: CostAnalysis {
                flops,
                transcendentals,
                bytes_accessed,
                optimal_seconds: 0.000_001,
            },
            memory_stats: CompiledMemoryStats {
                generated_code_size_in_bytes: generated_code_size,
                argument_size_in_bytes: arg_bytes,
                output_size_in_bytes: out_bytes,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 256,
            },
        };
        compiled.run_all_passes();
        Ok(compiled)
    }
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::proto::{
        ExecutableBuildOptionsProto, HloComputationProto, HloInstructionProto,
        PjRtPartialProgramProto, ShapeProto, TpuSerializedExecutableProto,
    };
    use crate::types::ElementType;

    #[test]
    fn test_opcode_and_compile_variants() {
        for b in 0..=78 {
            let op = TpuOpCode::from_byte(b);
            assert!(op.is_ok());
            for o in op {
                let _ = o.arity();
                let _ = o.is_binary();
                let _ = o.is_transcendental();
            }
        }
        assert!(TpuOpCode::from_byte(99).is_err());
        for m in [
            "shift_left",
            "shift_right_arithmetic",
            "shift_right_logical",
            "population_count",
            "count_leading_zeros",
            "convolution",
            "gather",
            "scatter",
            "dynamic_update_slice",
            "dynamic_slice",
            "reduce_window",
            "sort",
            "rng_bit_generator",
            "tpu.memref_slice",
            "tpu.erase_memref_layout",
            "all_to_all",
            "exponential_minus_one",
            "log_plus_one",
            "logistic",
            "erf",
            "round_nearest_even",
            "sign",
            "is_finite",
            "power",
            "atan2",
            "clamp",
            "popcnt",
            "pad",
            "select_and_scatter",
            "batch_norm_inference",
            "fft",
            "cholesky",
            "triangular_solve",
            "expm1",
            "log1p",
            "round_nearest_afz",
            "cbrt",
            "tan(",
            "stablehlo.tan",
            "top_k",
            "uniform_quantize",
            "uniform_dequantize",
            "partition_id",
            "replica_id",
            "after_all",
            "infeed",
            "outfeed",
            "send",
            "recv",
            "get_tuple_element",
            "tuple",
        ] {
            assert!(CompiledProgram::compile("mlir", m.as_bytes(), b"").is_ok());
        }

        for hlo_spec in [
            "f32[2,3]{1,0}",
            "f64[4]",
            "bf16[4]",
            "f16[4]",
            "s32[4]",
            "s64[4]",
            "s16[4]",
            "s8[4]",
            "u32[4]",
            "u64[4]",
            "u16[4]",
            "u8[4]",
            "pred[4]",
            "i1[]",
        ] {
            assert!(CompiledProgram::parse_hlo_shape_token(hlo_spec).is_some());
        }
        assert!(CompiledProgram::parse_hlo_shape_token("unknown[4]").is_none());
        assert!(CompiledProgram::parse_hlo_shape_token("f32[bad]").is_none());
        assert!(CompiledProgram::parse_hlo_shape_token("no_brackets").is_none());

        assert_eq!(
            CompiledProgram::parse_usize_list_attr("permutation = [1, 0]", "permutation"),
            vec![1, 0]
        );
        assert_eq!(
            CompiledProgram::parse_usize_list_attr("permutation = ]1, 0[", "permutation"),
            [] as [usize; 0]
        );
        assert_eq!(
            CompiledProgram::parse_single_usize_attr(
                "dimension = 2, iota_dimension = 1",
                "dimension"
            ),
            Some(2)
        );
        assert!(CompiledProgram::parse_single_usize_attr("dimension : 2", "dimension").is_none());
        assert!(CompiledProgram::parse_single_usize_attr("dimension = bad", "dimension").is_none());
        assert_eq!(
            CompiledProgram::parse_usize_list_attr(
                "permutation = array<i64: 1, bad, 0>",
                "permutation"
            ),
            vec![1, 0]
        );
        let lex_tokens = HloLexer::tokenize("module @m -> -1.5 %a0 _x");
        assert_ne!(format!("{:?}{:?}{lex_tokens:?}", HloLexer, HloParser), "");
        assert_eq!(lex_tokens.clone(), lex_tokens);

        let dense_attrs = CompiledProgram::parse_instruction_attrs(
            "stablehlo.constant dense<2.5> : tensor<4xf32>",
            &[],
            ElementType::F32,
        );
        assert_eq!(dense_attrs.constant_bytes.len(), 16);
        let dense_list_attrs = CompiledProgram::parse_instruction_attrs(
            "stablehlo.constant dense<[1.0, 2.0]>",
            &[],
            ElementType::F32,
        );
        assert_eq!(dense_list_attrs.constant_bytes.len(), 8);
        let hlo_shape_attrs = CompiledProgram::parse_instruction_attrs(
            "f32[2,2] transpose(%p0), dimensions={1,0}, iota_dimension = 1",
            &[],
            ElementType::F32,
        );
        assert!(hlo_shape_attrs.output_shape.is_some());
        assert_eq!(hlo_shape_attrs.iota_dim, 1);
        let concat_attrs = CompiledProgram::parse_instruction_attrs(
            "stablehlo.concatenate %0, %1, dimension = 2 : tensor<4xf32>",
            &[],
            ElementType::F32,
        );
        assert_eq!(concat_attrs.concat_dim, 2);

        for spec in [
            "tensor<4xf32>",
            "tensor<2x2xf64>",
            "tensor<4xbf16>",
            "tensor<4xf16>",
            "tensor<4xs32>",
            "tensor<4xs64>",
            "tensor<4xs16>",
            "tensor<4xs8>",
            "tensor<4xu32>",
            "tensor<4xu64>",
            "tensor<4xu16>",
            "tensor<4xu8>",
            "tensor<4xi1>",
            "tensor<4xc64>",
            "tensor<4xc128>",
            "tensor<4xf8e5m2>",
            "tensor<4xf8e4m3fn>",
            "tensor<4xf8e4m3b11fnuz>",
            "tensor<4xs4>",
            "tensor<4xi4>",
            "tensor<4xu4>",
            "tensor<4xui4>",
            "tensor<4xs2>",
            "tensor<4xi2>",
            "tensor<4xu2>",
            "tensor<4xui2>",
        ] {
            assert!(CompiledProgram::parse_tensor_type(spec).is_some());
        }
        assert!(CompiledProgram::parse_tensor_type("tensor<f32>").is_some());
        assert!(CompiledProgram::parse_tensor_type("tensor<-1xf32>").is_none());
        assert!(CompiledProgram::parse_tensor_type("tensor<badxf32>").is_none());
        assert!(CompiledProgram::parse_tensor_type("tensor<4xunknown>").is_none());
        assert!(CompiledProgram::parse_tensor_type("tensor<unclosed").is_none());
        assert!(CompiledProgram::parse_tensor_type("not_a_tensor").is_none());

        for phase in COMPILATION_PHASE_NAMES {
            let res = CompiledProgram::run_compilation_phase(
                phase,
                b"module @phased { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }",
                b"",
            );
            assert!(res.is_ok());
            for serialized_phase in res {
                assert!(
                    CompiledProgram::run_compilation_phase(phase, &serialized_phase, b"").is_ok()
                );
            }
            let proto_res = CompiledProgram::run_compilation_phase_proto(
                phase,
                b"module @phased_proto { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }",
                b"",
            );
            assert!(proto_res.is_ok());
            for p_proto in proto_res {
                assert_eq!(p_proto.producer_phase, phase);
                let enc_partial = p_proto.encode();
                assert!(CompiledProgram::run_compilation_phase(phase, &enc_partial, b"").is_ok());
            }
        }
        let empty_partial = PjRtPartialProgramProto {
            program: Vec::new(),
            program_format: String::from("mlir"),
            producer_phase: String::from("hlo_verification"),
            consumer_phases: Vec::new(),
            version: String::from("1.0"),
            program_name: String::from("empty"),
        };
        assert!(
            CompiledProgram::run_compilation_phase(
                "hlo_verification",
                &empty_partial.encode(),
                b""
            )
            .is_ok()
        );
        assert!(CompiledProgram::run_compilation_phase("bad_phase", b"return", b"").is_err());
        assert!(CompiledProgram::run_compilation_phase("hlo_verification", b"", b"").is_err());
        assert!(CompiledProgram::run_compilation_phase_proto("bad_phase", b"return", b"").is_err());

        let mosaic_text = r#"module @mosaic_kernel attributes {tpu.compilation_target = "tpu_v5p", tpu.core_type = "tensor_core", mhlo.layout_mode = "{1,0}", mhlo.sharding = "{replicated}"} {
          func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = tpu.matmul %arg0, %arg1 : tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }"#;
        let mosaic_meta = CompiledProgram::parse_mosaic_metadata(mosaic_text);
        assert!(mosaic_meta.has_mosaic_ops);
        assert_eq!(mosaic_meta.compilation_target, "tpu_v5p");
        assert_eq!(mosaic_meta.core_type, "tensor_core");
        assert_eq!(mosaic_meta.layout_mode, "{1,0}");
        assert_eq!(mosaic_meta.sharding, "{replicated}");
        assert_eq!(mosaic_meta.clone(), mosaic_meta);
        assert_ne!(format!("{mosaic_meta:?}"), "");
        let unquoted_meta = CompiledProgram::parse_mosaic_metadata(
            r#"tpu.compilation_target = unquoted tpu.core_type = "unclosed"#,
        );
        assert!(unquoted_meta.has_mosaic_ops);
        for mosaic_op in [
            b"module @m { tpu.matmul }" as &[u8],
            b"module @m { tpu.matmul_push_rhs }",
            b"module @m { tpu.matmul_acc }",
            b"module @m { tpu.matmul_pop }",
            b"module @m { tpu.enqueue_dma }",
            b"module @m { tpu.wait_dma }",
            b"module @m { tpu.dma_copy }",
            b"module @m { tpu.dma_wait }",
            b"module @m { tpu.semaphore_signal }",
            b"module @m { tpu.semaphore_wait }",
            b"module @m { tpu.semaphore_read }",
            b"module @m { tpu.barrier }",
            b"module @m { tpu.vector_load }",
            b"module @m { tpu.vector_store }",
            b"module @m { tpu.load }",
            b"module @m { tpu.store }",
            b"module @m { tpu.iota }",
            b"module @m { tpu.repeat }",
            b"module @m { tpu.rotate }",
            b"module @m { tpu.pack_subelements }",
            b"module @m { tpu.unpack_subelements }",
            b"module @m { tpu.memref_squeeze }",
            b"module @m { tpu.memref_bitcast }",
            b"module @m { tpu.relabel }",
            b"module @m { tpu.bitcast }",
            b"module @m { tpu.roll }",
            b"module @m { tpu.region }",
            b"module @m { collective_permute }",
        ] {
            assert!(CompiledProgram::compile("mlir", mosaic_op, b"").is_ok());
        }

        assert!(CompiledProgram::compile("unknown_fmt", b"code", b"").is_err());
        assert!(CompiledProgram::compile("hlo_with_config", b"return", b"").is_ok());
        assert!(CompiledProgram::compile("mlir", b"", b"").is_err());
        assert!(CompiledProgram::compile("mlir", b"SYNTAX_ERROR", b"").is_err());
        assert!(
            CompiledProgram::compile("mlir", b"return", &[0xff, 0xff, 0xff, 0xff, 0xff, 0x7f])
                .is_err()
        );

        let ssa_dag = br"module @ssa_dag {
          func.func @main(%arg0: tensor<4xs32>, %arg1: tensor<4xs32>) -> (tensor<4xs32>, tensor<4xs32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xs32>
            %1 = stablehlo.multiply %0, %arg1 : tensor<4xs32>
            return %0, %1 : tensor<4xs32>, tensor<4xs32>
          }
        }";
        let dag_prog = CompiledProgram::compile("mlir", ssa_dag, b"");
        assert!(dag_prog.is_ok());
        for mut dp in dag_prog {
            assert_eq!(dp.parameter_shapes.len(), 2);
            assert_eq!(dp.output_shapes.len(), 2);
            assert_eq!(dp.instruction_bundles.len(), 2);
            dp.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 2,
                opcode: TpuOpCode::Constant,
                operand_slots: Vec::new(),
                output_slot: 4,
                sync_flag_id: 0,
                attrs: InstructionAttrs::default(),
            });
            dp.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 3,
                opcode: TpuOpCode::DotGeneralF32,
                operand_slots: Vec::new(),
                output_slot: 5,
                sync_flag_id: 0,
                attrs: InstructionAttrs::default(),
            });
            dp.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 4,
                opcode: TpuOpCode::CustomCall,
                operand_slots: vec![0],
                output_slot: 6,
                sync_flag_id: 0,
                attrs: InstructionAttrs::default(),
            });
            dp.run_all_passes();
            assert_eq!(dp.encode_vliw_program().len(), 5);
            let imem = dp.encode_architectural_imem();
            assert_eq!(imem.len(), 5 * 64);
            let decoded_imem = CompiledProgram::decode_architectural_imem(&imem);
            assert_eq!(decoded_imem, dp.encode_vliw_program());
        }

        let opts = CompileOptionsProto {
            argument_layouts: vec![],
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableBuildOptionsProto {
                device_ordinal: 0,
                result_layout: None,
                num_replicas: 2,
                num_partitions: 4,
                use_spmd_partitioning: true,
                ..ExecutableBuildOptionsProto::default()
            }),
            compile_portable_executable: false,
            env_option_overrides: std::collections::BTreeMap::new(),
            profile_version: 1,
            serialized_multi_slice_config: Vec::new(),
        };
        let opts_bytes = opts.encode();

        let sources: [&[u8]; 20] = [
            b"module @m_add { stablehlo.add tensor<4xf32> }",
            b"module @m_bad_tensor { tensor<bad> }",
            b"module @m_s32 { s32_add }",
            b"module @m_mul { mul_f32 }",
            b"module @m_sub { sub_f32 }",
            b"module @m_div { div_f32 }",
            b"module @m_max { max_f32 }",
            b"module @m_min { min_f32 }",
            b"module @m_dot { dot_general }",
            b"module @m_neg { neg_f32 }",
            b"module @m_exp { exp_f32 }",
            b"module @m_tanh { tanh_f32 }",
            b"module @m_red { reduce_sum }",
            b"module @m_all { all_reduce }",
            b"module @m_cc { custom_call }",
            b"module @m_xor { xor_u8 }",
            b"module @ { return }",
            b"HloModule plain_ident
ENTRY main { ROOT r = f32[4] parameter(0) }",
            b"HloModule !
ENTRY main { ROOT r = f32[4] parameter(0) }",
            b"ENTRY main { ROOT r = f32[4] parameter(0) }",
        ];

        for src in sources {
            let prog_res = CompiledProgram::compile("mlir", src, &opts_bytes);
            assert!(prog_res.is_ok());
            for prog in prog_res {
                assert_eq!(prog.num_replicas, 2);
                assert_eq!(prog.num_partitions, 4);
                assert!(prog.generated_code_size().0 >= 256);
                let _ = format!(
                    "{:?} {:?} {:?} {:?}",
                    prog, prog.instruction_bundles, prog.cost_analysis, prog.memory_stats
                );
                let b_clone = prog.instruction_bundles.clone();
                assert_eq!(prog.instruction_bundles, b_clone);
                for b in &prog.instruction_bundles {
                    let word = b.encode_vliw_word();
                    assert_eq!(bytemuck::bytes_of(&word).len(), 16);
                    assert_ne!(format!("{word:?}"), "");
                    let arch_bundle = b.encode_architectural_bundle();
                    assert_eq!(bytemuck::bytes_of(&arch_bundle).len(), 64);
                    assert_ne!(format!("{arch_bundle:?}"), "");
                }
                let c_clone = prog.cost_analysis;
                let m_clone = prog.memory_stats;
                let p_clone = prog.clone();
                assert_eq!(p_clone.fingerprint, prog.fingerprint);
                assert!(c_clone.optimal_seconds > 0.0);
                assert!(m_clone.generated_code_size_in_bytes >= 256);

                let ser = prog.serialize();
                let deser = CompiledProgram::deserialize(&ser);
                assert!(deser.is_ok());

                let proto = prog.to_executable_proto();
                let proto_bytes = proto.encode();
                let deser_proto = CompiledProgram::deserialize(&proto_bytes);
                assert!(deser_proto.is_ok());

                let tpu_exec_proto = prog.to_tpu_executable_proto();
                let tpu_exec_bytes = tpu_exec_proto.encode();
                let deser_tpu_exec = CompiledProgram::deserialize(&tpu_exec_bytes);
                assert!(deser_tpu_exec.is_ok());
            }
        }
    }

    #[test]
    fn test_binary_hlo_module_proto_lowering() {
        let hlo = HloModuleProto {
            name: String::from("binary_hlo_add"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            host_program_shape: None,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                program_shape: None,
                id: 1,
                root_id: 3,
                instructions: vec![
                    HloInstructionProto {
                        name: String::from("p0"),
                        opcode: String::from("parameter"),
                        shape: Some(ShapeProto {
                            element_type: 11,
                            dimensions: vec![8],
                            tuple_shapes: vec![],
                            layout: None,
                            is_dynamic_dimension: vec![false],
                        }),
                        id: 1,
                        operand_ids: vec![],
                        parameter_number: 0,
                        custom_call_target: String::new(),
                    },
                    HloInstructionProto {
                        name: String::from("p1"),
                        opcode: String::from("parameter"),
                        shape: Some(ShapeProto {
                            element_type: 0,
                            dimensions: vec![],
                            tuple_shapes: vec![],
                            layout: None,
                            is_dynamic_dimension: vec![],
                        }),
                        id: 2,
                        operand_ids: vec![],
                        parameter_number: 1,
                        custom_call_target: String::new(),
                    },
                    HloInstructionProto {
                        name: String::from("add.3"),
                        opcode: String::from("add"),
                        shape: None,
                        id: 3,
                        operand_ids: vec![1, 2],
                        parameter_number: 0,
                        custom_call_target: String::new(),
                    },
                ],
            }],
            id: 1,
        };
        let encoded = hlo.encode();
        let prog = CompiledProgram::compile("hlo", &encoded, b"");
        assert!(prog.is_ok());
        for p in prog {
            assert_eq!(p.name, "binary_hlo_add");
            assert_eq!(p.opcode, TpuOpCode::AddF32);
            assert_eq!(p.parameter_shapes.len(), 2);
        }

        let with_cfg = HloModuleProtoWithConfig {
            hlo_module: Some(hlo),
            config: None,
        };
        let with_cfg_prog = CompiledProgram::compile("hlo_with_config", &with_cfg.encode(), b"");
        assert!(with_cfg_prog.is_ok());
        for p in with_cfg_prog {
            assert_eq!(p.name, "binary_hlo_add");
            assert_eq!(p.opcode, TpuOpCode::AddF32);
        }

        let mut mlir_bc = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        mlir_bc.extend_from_slice(b"module @bc_mod { stablehlo.add }");
        let bc_hdr = MlirBytecodeHeader::decode(&mlir_bc);
        assert!(bc_hdr.is_some());
        for hdr in bc_hdr {
            assert_ne!(format!("{hdr:?}"), "");
            assert_eq!(hdr.clone(), hdr);
            assert_ne!(hdr.to_synthesized_mlir_text(), "");
        }

        // Test encode_mlir_prefix_varint and decode_mlir_prefix_varint across 1-byte, 2-byte, and 9-byte forms
        for v in [0_u64, 1, 42, 127, 128, 1024, 16383, 16384, 0x1234_5678_9ABC] {
            let enc_v = encode_mlir_prefix_varint(v);
            let mut cur: &[u8] = &enc_v;
            assert_eq!(decode_mlir_prefix_varint(&mut cur), Some(v));
            assert_eq!(cur.len(), 0);
        }
        let mut empty_cur: &[u8] = &[];
        assert!(decode_mlir_prefix_varint(&mut empty_cur).is_none());
        let mut trunc_zero: &[u8] = &[0, 1, 2];
        assert!(decode_mlir_prefix_varint(&mut trunc_zero).is_none());
        let mut trunc_two: &[u8] = &[0x02];
        assert!(decode_mlir_prefix_varint(&mut trunc_two).is_none());

        // Build a structured binary MLIR bytecode container with Sections 0, 1, 2, 3, 4 (aligned), 5, and 6
        let mut structured_bc = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(1)); // version = 1
        structured_bc.extend_from_slice(b"MLIR20.0.0\x00");
        let str_payload =
            b"module\x00@bin_mlir\x00stablehlo.multiply\x00tensor<4xf32>\x00stablehlo\x00";
        structured_bc.push(0); // section_id = 0 (kString), unaligned
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(str_payload.len() as u64));
        structured_bc.extend_from_slice(str_payload);
        // Section 1 (kDialect): index 4 ("stablehlo") and out-of-bounds index 99
        let mut dial_payload = Vec::new();
        dial_payload.extend_from_slice(&encode_mlir_prefix_varint(4));
        dial_payload.extend_from_slice(&encode_mlir_prefix_varint(99));
        structured_bc.push(1);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(dial_payload.len() as u64));
        structured_bc.extend_from_slice(&dial_payload);
        // Section 2 (kAttrType)
        let attr_payload = b"tensor<4xf32>\x00";
        structured_bc.push(2);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(attr_payload.len() as u64));
        structured_bc.extend_from_slice(attr_payload);
        // Section 3 (kAttrTypeOffset)
        let off_payload = encode_mlir_prefix_varint(14);
        structured_bc.push(3);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(off_payload.len() as u64));
        structured_bc.extend_from_slice(&off_payload);
        // Section 4 (kIR), aligned bit set (0x80)
        let ir_payload = encode_mlir_prefix_varint(7);
        structured_bc.push(0x84);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(ir_payload.len() as u64));
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(4)); // alignment = 4
        structured_bc.extend_from_slice(&ir_payload);
        // Section 5 (kResource)
        structured_bc.push(5);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(4));
        structured_bc.extend_from_slice(&[1, 2, 3, 4]);
        // Section 6 (kProperties / unknown)
        structured_bc.push(6);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(1));
        structured_bc.push(0xFF);

        let parsed_struct_bc = MlirBytecodeHeader::decode(&structured_bc);
        assert!(parsed_struct_bc.is_some());
        for s_hdr in parsed_struct_bc {
            assert_eq!(s_hdr.sections.len(), 7);
            assert_eq!(s_hdr.dialects.len(), 2);
            assert_eq!(s_hdr.attr_type_entries, vec![String::from("tensor<4xf32>")]);
            assert_eq!(s_hdr.attr_type_offsets, vec![14]);
            assert_eq!(s_hdr.ir_op_records, vec![7]);
            assert_eq!(s_hdr.resources.len(), 1);
            for sec0 in s_hdr.sections.first() {
                assert_eq!(sec0.clone(), *sec0);
                assert_ne!(format!("{sec0:?}"), "");
            }
            assert!(
                s_hdr
                    .to_synthesized_mlir_text()
                    .contains("stablehlo.multiply")
            );
            let reencoded = s_hdr.encode();
            assert!(MlirBytecodeHeader::decode(&reencoded).is_some());
        }
        let compiled_struct_bc = CompiledProgram::compile("mlir", &structured_bc, b"");
        assert!(compiled_struct_bc.is_ok());
        for p in compiled_struct_bc {
            assert_eq!(p.name, "bin_mlir");
            assert_eq!(p.opcode, TpuOpCode::MulF32);
        }

        // Test truncated section length varint and truncated section payload in MlirBytecodeHeader::decode
        let mut trunc_sec_len = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        trunc_sec_len.extend_from_slice(b"\x03MLIR\x00\x00\x02");
        assert!(MlirBytecodeHeader::decode(&trunc_sec_len).is_some());
        let mut trunc_sec_data = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        trunc_sec_data.extend_from_slice(b"\x03MLIR\x00\x00");
        trunc_sec_data.extend_from_slice(&encode_mlir_prefix_varint(50));
        trunc_sec_data.push(1);
        assert!(MlirBytecodeHeader::decode(&trunc_sec_data).is_some());

        let mut mlir_bc_nul = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        mlir_bc_nul.extend_from_slice(b"\x03MLIR20.0.0\x00");
        assert!(MlirBytecodeHeader::decode(&mlir_bc_nul).is_some());
        assert!(MlirBytecodeHeader::decode(MLIR_BYTECODE_MAGIC).is_some());
        assert!(MlirBytecodeHeader::decode(b"not_mlir").is_none());
        let bc_prog = CompiledProgram::compile("mlir", &mlir_bc, b"");
        assert!(bc_prog.is_ok());
        for p in bc_prog {
            assert_eq!(p.name, "bc_mod");
            assert_eq!(p.opcode, TpuOpCode::AddF32);
        }
        assert!(CompiledProgram::compile("mlir", b"-5", b"").is_ok());
        assert!(CompiledProgram::compile("mlir", b"%v.1", b"").is_ok());
        assert!(CompiledProgram::compile("mlir", b"1_0", b"").is_ok());

        // Also test HloModuleProto without explicit parameter instructions (binary, unary, and parameter-only fallback)
        for op_str in ["mul_f32", "neg_f32", "parameter"] {
            let hlo_no_params = HloModuleProto {
                name: String::from("no_params"),
                entry_computation_name: String::from("main"),
                entry_computation_id: 1,
                host_program_shape: None,
                computations: vec![HloComputationProto {
                    name: String::from("main"),
                    program_shape: None,
                    id: 1,
                    root_id: 1,
                    instructions: vec![HloInstructionProto {
                        name: String::from("op.1"),
                        opcode: String::from(op_str),
                        shape: None,
                        id: 1,
                        operand_ids: vec![],
                        parameter_number: 0,
                        custom_call_target: String::new(),
                    }],
                }],
                id: 1,
            };
            let enc2 = hlo_no_params.encode();
            assert!(CompiledProgram::compile("hlo", &enc2, b"").is_ok());
        }

        let bad_et_hlo = HloModuleProto {
            name: String::from("bad_et"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            host_program_shape: None,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                program_shape: None,
                id: 1,
                root_id: 1,
                instructions: vec![HloInstructionProto {
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    shape: Some(ShapeProto {
                        element_type: 99,
                        dimensions: vec![4],
                        tuple_shapes: vec![],
                        layout: None,
                        is_dynamic_dimension: vec![],
                    }),
                    id: 1,
                    operand_ids: vec![],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                }],
            }],
            id: 1,
        };
        assert!(CompiledProgram::compile("hlo", &bad_et_hlo.encode(), b"").is_err());

        // Test TpuSerializedExecutableProto with empty hlo_module_bytes, bad opcode, and SYNTAX_ERROR
        let empty_proto = TpuSerializedExecutableProto {
            name: String::from("long_enough_name"),
            fingerprint: String::new(),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![],
            output_shapes: vec![],
            compile_options_bytes: vec![],
            hlo_module_bytes: vec![],
        };
        assert!(CompiledProgram::deserialize(&empty_proto.encode()).is_err());

        let bad_op_proto = TpuSerializedExecutableProto {
            name: String::from("long_enough_name"),
            fingerprint: String::new(),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![250],
            output_shapes: vec![],
            compile_options_bytes: vec![],
            hlo_module_bytes: b"return".to_vec(),
        };
        assert!(CompiledProgram::deserialize(&bad_op_proto.encode()).is_err());

        let bad_src_proto = TpuSerializedExecutableProto {
            name: String::from("long_enough_name"),
            fingerprint: String::new(),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![0],
            output_shapes: vec![],
            compile_options_bytes: vec![],
            hlo_module_bytes: b"SYNTAX_ERROR".to_vec(),
        };
        assert!(CompiledProgram::deserialize(&bad_src_proto.encode()).is_err());
    }

    #[test]
    fn test_deserialize_error_branches() {
        assert!(CompiledProgram::deserialize(b"short").is_err());
        assert!(CompiledProgram::deserialize(b"BADMAGIC123456789").is_err());
        assert!(CompiledProgram::deserialize(EXECUTABLE_MAGIC).is_err());

        let mut bad_ver = Vec::from(EXECUTABLE_MAGIC.as_slice());
        bad_ver.extend_from_slice(&99_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&bad_ver).is_err());

        let mut missing_op = Vec::from(EXECUTABLE_MAGIC.as_slice());
        missing_op.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        assert!(CompiledProgram::deserialize(&missing_op).is_err());

        let mut bad_op = Vec::from(EXECUTABLE_MAGIC.as_slice());
        bad_op.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        bad_op.push(250);
        assert!(CompiledProgram::deserialize(&bad_op).is_err());

        let mut trunc_fmt_u32 = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_fmt_u32.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_fmt_u32.push(0);
        assert!(CompiledProgram::deserialize(&trunc_fmt_u32).is_err());

        let mut empty_src_payload = Vec::from(EXECUTABLE_MAGIC.as_slice());
        empty_src_payload.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        empty_src_payload.push(0);
        empty_src_payload.extend_from_slice(&4_u32.to_le_bytes());
        empty_src_payload.extend_from_slice(b"mlir");
        empty_src_payload.extend_from_slice(&0_u32.to_le_bytes());
        empty_src_payload.extend_from_slice(&0_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&empty_src_payload).is_err());

        let mut trunc_fmt = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_fmt.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_fmt.push(0);
        trunc_fmt.extend_from_slice(&100_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&trunc_fmt).is_err());

        let mut bad_utf8_fmt = Vec::from(EXECUTABLE_MAGIC.as_slice());
        bad_utf8_fmt.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        bad_utf8_fmt.push(0);
        bad_utf8_fmt.extend_from_slice(&2_u32.to_le_bytes());
        bad_utf8_fmt.extend_from_slice(&[0xff, 0xfe, 0, 0, 0, 0]);
        assert!(CompiledProgram::deserialize(&bad_utf8_fmt).is_err());

        let mut trunc_src = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_src.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_src.push(0);
        trunc_src.extend_from_slice(&4_u32.to_le_bytes());
        trunc_src.extend_from_slice(b"mlir");
        trunc_src.extend_from_slice(&100_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&trunc_src).is_err());

        let trunc_opt_test = b"module @cst_vec { func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> { %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
 func.return %0 : tensor<2xf32> } }";
        assert!(CompiledProgram::compile("mlir", trunc_opt_test, b"").is_ok());
        let cst_single_test = b"module @cst_s1 { func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> { %0 = stablehlo.constant dense<5.0> : tensor<1xf32>
 return %0 : tensor<1xf32> } }";
        assert!(CompiledProgram::compile("mlir", cst_single_test, b"").is_ok());
        let hlo_entry_test = b"HloModule m
ENTRY %main (x: f32[4]) -> f32[4] {
  %x = f32[4] parameter(0)
  ROOT %r = f32[4] negate(%x)
}";
        for prog in CompiledProgram::compile("hlo", hlo_entry_test, b"") {
            let ba = prog.build_buffer_assignment_proto();
            assert_eq!(ba.buffer_allocations.len(), 3);
            assert!(
                crate::proto::BufferAssignmentProto::decode(&ba.encode())
                    .ok()
                    .is_some()
            );
        }

        let mut trunc_opt = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_opt.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_opt.push(0);
        trunc_opt.extend_from_slice(&4_u32.to_le_bytes());
        trunc_opt.extend_from_slice(b"mlir");
        trunc_opt.extend_from_slice(&2_u32.to_le_bytes());
        trunc_opt.extend_from_slice(b"id");
        trunc_opt.extend_from_slice(&100_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&trunc_opt).is_err());

        let hdr = ExecutableBinaryHeader {
            magic: *EXECUTABLE_MAGIC,
            version_le: EXECUTABLE_FORMAT_VERSION,
        };
        assert_eq!(hdr.clone(), hdr);
        assert_ne!(format!("{hdr:?}"), "");

        // Test petgraph + indexmap DAG analysis (valid DAG and cyclic graph)
        let valid_comp = HloComputationProto {
            name: String::from("dag_main"),
            program_shape: None,
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    shape: Some(ShapeProto {
                        element_type: 11,
                        dimensions: vec![16, 128],
                        tuple_shapes: vec![],
                        layout: None,
                        is_dynamic_dimension: vec![],
                    }),
                    id: 1,
                    operand_ids: vec![999], // external/unknown operand ID gracefully ignored
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
                HloInstructionProto {
                    name: String::from("neg"),
                    opcode: String::from("negate"),
                    shape: None,
                    id: 2,
                    operand_ids: vec![1],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
            ],
        };
        for dag_rep in analyze_hlo_computation_dag(&valid_comp) {
            assert_eq!(dag_rep.topological_order, vec![1, 2]);
            assert_eq!(dag_rep.buffer_offsets.len(), 2);
            assert_eq!(dag_rep.total_hbm_bytes, 12288);
            assert_eq!(dag_rep.clone(), dag_rep);
            assert_ne!(format!("{dag_rep:?}"), "");
        }

        let cyclic_comp = HloComputationProto {
            name: String::from("cyclic"),
            program_shape: None,
            id: 2,
            root_id: 2,
            instructions: vec![
                HloInstructionProto {
                    name: String::from("a"),
                    opcode: String::from("negate"),
                    shape: None,
                    id: 1,
                    operand_ids: vec![2],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
                HloInstructionProto {
                    name: String::from("b"),
                    opcode: String::from("negate"),
                    shape: None,
                    id: 2,
                    operand_ids: vec![1],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
            ],
        };
        assert!(analyze_hlo_computation_dag(&cyclic_comp).is_err());
        assert!(CompiledProgram::parse_hlo_shape_token("[2,2]").is_none());

        // Test constant folding (NegF32, AddF32, MulF32 on Constant bundles), CSE, and liveness intervals
        let cst_fold_mlir = b"module @cst_fold {

          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {

            %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>

            %c1 = stablehlo.negate %c0

            %c2 = stablehlo.add %c0, %c1

            %c3 = stablehlo.multiply %c0, %c2

            %x0 = stablehlo.add %arg0, %c3

            %x1 = stablehlo.add %arg0, %c3

            %x2 = stablehlo.negate %x1

            %rng = stablehlo.rng_bit_generator %x2

            return %x1 : tensor<4xf32>

          }

        }";
        for prog in CompiledProgram::compile("mlir", cst_fold_mlir, b"") {
            let intervals = prog.compute_liveness_intervals();
            assert_ne!(intervals, [] as [RegisterLivenessInterval; 0]);
            for iv in intervals.first() {
                assert_eq!(*iv, *iv);
                assert_ne!(format!("{iv:?}"), "");
            }
        }

        // Test TensorCoreVliwBundle::decode and to_instruction_bundle roundtrip across arities 0, 1, 2, 3 and invalid opcodes
        for (op, ops) in [
            (TpuOpCode::Constant, vec![]),
            (TpuOpCode::NegF32, vec![0]),
            (TpuOpCode::AddF32, vec![0, 1]),
            (TpuOpCode::Select, vec![0, 1, 2]),
        ] {
            let bundle = TpuInstructionBundle {
                pc_offset: 3,
                opcode: op,
                operand_slots: ops.clone(),
                output_slot: 4,
                sync_flag_id: 7,
                attrs: InstructionAttrs::default(),
            };
            let vliw = bundle.encode_architectural_bundle();
            let raw_bytes: [u8; 64] = bytemuck::cast(vliw);
            let decoded = TensorCoreVliwBundle::decode(&raw_bytes);
            assert_eq!(decoded.ok(), Some(vliw));
            let roundtripped = vliw.to_instruction_bundle();
            assert_eq!(roundtripped.ok(), Some(bundle));
        }
        let mut bad_vliw = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::AddF32,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        }
        .encode_architectural_bundle();
        bad_vliw.vector_slot.opcode = 300;
        let bad_bytes_300: [u8; 64] = bytemuck::cast(bad_vliw);
        assert!(TensorCoreVliwBundle::decode(&bad_bytes_300).is_err());
        assert!(bad_vliw.to_instruction_bundle().is_err());
        bad_vliw.vector_slot.opcode = 250;
        let bad_bytes_250: [u8; 64] = bytemuck::cast(bad_vliw);
        assert!(TensorCoreVliwBundle::decode(&bad_bytes_250).is_err());
        assert!(bad_vliw.to_instruction_bundle().is_err());
    }

    proptest::proptest! {
        #[test]
        fn proptest_winnow_tensor_shape_and_compiler_roundtrip(
            d0 in 1_i64..32,
            d1 in 1_i64..32,
        ) {
            let mlir_spec = format!("tensor<{d0}x{d1}xf32>");
            let parsed_mlir = CompiledProgram::parse_tensor_type(&mlir_spec);
            proptest::prop_assert!(parsed_mlir.is_some());
            if let Some(sh) = parsed_mlir {
                proptest::prop_assert_eq!(sh.dims, vec![d0, d1]);
                proptest::prop_assert_eq!(sh.element_type, ElementType::F32);
            }

            let hlo_spec = format!("bf16[{d0},{d1}]{{1,0}}");
            let parsed_hlo = CompiledProgram::parse_hlo_shape_token(&hlo_spec);
            proptest::prop_assert!(parsed_hlo.is_some());
            if let Some(sh) = parsed_hlo {
                proptest::prop_assert_eq!(sh.dims, vec![d0, d1]);
                proptest::prop_assert_eq!(sh.element_type, ElementType::Bf16);
            }
        }
    }
}
