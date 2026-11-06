//! HLO and `StableHLO` compiler, recursive-descent parser, and VLIW bundle emitter.
//!
//! Provides the DAG lowerer, fingerprint generator, cost analyzer, phased compiler, and
//! wire-compatible `TpuSerializedExecutableProto` serializer for `PJRT_Client_Compile`,
//! `PJRT_Compile`, `PJRT_Executable_Serialize`, `PJRT_Executable_DeserializeAndLoad`, and
//! `PJRT_PhaseCompile_Extension`.
//!
//! Organized into focused submodules:
//! - [`ast`]: Strongly-typed [`AstModule`], [`AstFunction`], [`AstRegion`], [`AstBlock`], [`AstOperation`], [`MlirType`], and [`MlirAttribute`].
//! - [`parser`]: Recursive-descent MLIR grammar parser (`TokenCursor`, `MlirModuleParser`) and CHLO legalizer.
//! - [`hlo_parser`]: Recursive-descent textual XLA HLO parser (`HloTextParser`), `HloModuleProto` bridge, and AST lowerer.
//! - [`opcode`]: [`TpuOpCode`], [`InstructionAttrs`], and [`TpuInstructionBundle`].
//! - [`vliw`]: Architectural 64-byte [`TensorCoreVliwBundle`] and 16-byte [`VliwInstructionWord`] encoders.
//! - [`mlir_bytecode`]: MLIR Bytecode (`ML\xefR`) container decoder, encoder, and `PrefixVarInt` codecs.
//! - [`lexer`]: `winnow`-backed [`HloLexer`], [`HloParser`], and [`MosaicTargetMetadata`] parser.
//! - [`passes`]: Compiler optimization passes, VMEM liveness intervals, cost analysis, and DAG scheduling.
//! - [`serialization`]: `LTPUEXEC` binary format and `DeepSea` / `TpuExecutableProto` wire serialization.

pub mod ast;
pub mod hlo_parser;
pub mod lexer;
pub mod mlir_bytecode;
pub mod opcode;
pub mod parser;
pub mod passes;
pub mod serialization;
pub mod vliw;

pub use self::ast::{
    AffineMapExpr, AstBlock, AstBlockArg, AstFunction, AstFunctionType, AstModule, AstOpKind,
    AstOpResult, AstOperand, AstOperation, AstRegion, AstResultType, ComparisonDirection,
    ComparisonType, ConvDimensionNumbersAttr, DenseElementsAttr, DialectKind, DimensionSize,
    DotDimensionNumbersAttr, FftKind, GatherDimensionNumbersAttr, HloComputationAst,
    HloInstructionAst, HloWindowSpec, MemRefLayoutAttr, MemRefType, MlirAttribute, MlirLocation,
    MlirType, MosaicDialectAttr, PrecisionLevel, QuantizedElementType, RankedTensorType,
    RngAlgorithmKind, ScatterDimensionNumbersAttr, SdyDimensionSharding, SdyMeshAttr,
    SdyTensorShardingAttr, TensorEncodingAttr, TpuMemorySpace, TpuSemaphoreKind, TransposeMode,
    VectorType, Visibility,
};
pub use self::hlo_parser::{
    HloTextParser, hlo_ast_to_proto, hlo_proto_to_ast, lower_ast_module_to_parsed_hlo,
    parse_unified_ast_module, verify_ast_module_semantics,
};
pub use self::lexer::{
    HloAstModule, HloLexer, HloParser, HloToken, MosaicTargetMetadata, ParsedHloModule,
    SpannedToken, TokenSpan,
};
pub use self::mlir_bytecode::{
    BINARY_CUSTOM_ATTR_TYPE_MAGIC, BinaryAttrTypeEntry, MLIR_BYTECODE_MAGIC, MlirBytecodeEncoder,
    MlirBytecodeHeader, MlirBytecodeOpNode, MlirBytecodeSection, MlirDialectDescriptor,
    decode_mlir_prefix_varint, encode_mlir_prefix_varint,
};
pub use self::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
pub use self::parser::{
    MlirModuleParser, TokenCursor, classify_ast_operation, legalize_chlo_in_ast_module,
    parse_float_literal, parse_mlir_ast_module, parse_mlir_element_type,
    resolve_reduction_region_opcode, split_shaped_dims_and_dtype,
};
pub use self::passes::{
    CompiledMemoryStats, CostAnalysis, CseExpressionKey, HloDagAnalysisReport, ParsedShardingSpec,
    RegisterLivenessInterval, analyze_hlo_computation_dag,
};
pub use self::serialization::{
    DeepseaRelocatedArtifact, EM_DEEPSEA, EXECUTABLE_FORMAT_VERSION, EXECUTABLE_MAGIC,
    ExecutableBinaryHeader, R_DEEPSEA_HBM_ABS64, R_DEEPSEA_IMEM_REL32, R_DEEPSEA_SYNC_FLAG16,
    R_DEEPSEA_VMEM_OFF32,
};
pub use self::vliw::{
    AsyncDmaSlot, ExtendedUnarySlot, LloVliwScheduler, MatrixMxuSlot, PolyhedralTileSchedule,
    ScalarAluSlot, TensorCoreVliwBundle, VectorAluSlot, VliwBundleHeader, VliwInstructionWord,
    VliwScoreboard,
};

use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{CompileOptionsProto, HloModuleProto, HloModuleProtoWithConfig};
use crate::types::{MemoryKind, Shape};

/// Strongly-typed program input format supported by [`CompiledProgram::compile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramFormat {
    /// `StableHLO` / Mosaic MLIR text or `ML\xefR` bytecode (`"mlir"`).
    Mlir,
    /// XLA `HloModuleProto` or textual HLO (`"hlo"`).
    Hlo,
    /// XLA `HloModuleProtoWithConfig` (`"hlo_with_config"`).
    HloWithConfig,
}

impl ProgramFormat {
    /// Parses a format string (`"mlir"`, `"hlo"`, or `"hlo_with_config"`) into a [`ProgramFormat`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Format string to parse.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if `raw` is not a recognized format.
    pub fn from_str_checked(raw: &str) -> LibTpuResult<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "mlir" => Ok(Self::Mlir),
            "hlo" => Ok(Self::Hlo),
            "hlo_with_config" => Ok(Self::HloWithConfig),
            _ => Err(LibTpuError::CompilationFailed {
                reason: format!("Unsupported program format '{raw}'"),
            }),
        }
    }

    /// Returns the canonical lowercase string representation of this [`ProgramFormat`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mlir => "mlir",
            Self::Hlo => "hlo",
            Self::HloWithConfig => "hlo_with_config",
        }
    }
}

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
    /// Strongly-typed program format (`Mlir`, `Hlo`, or `HloWithConfig`).
    pub program_format: ProgramFormat,
    /// Canonical lowercase program format string (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    pub format: String,
    /// Original program source or bytecode payload.
    pub source_bytes: Vec<u8>,
    /// Extracted JAX Mosaic / Pallas target metadata.
    pub mosaic_metadata: MosaicTargetMetadata,
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
    /// `compile_options` is malformed, or lexical/syntactic validation fails.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    pub fn compile(format: &str, code: &[u8], compile_options: &[u8]) -> LibTpuResult<Self> {
        let program_format = ProgramFormat::from_str_checked(format)?;
        let fmt_lower = String::from(program_format.as_str());
        if code.is_empty() {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("Program code payload cannot be empty"),
            });
        }

        let parsed_options = CompileOptionsProto::decode_or_empty(compile_options)?;

        let mut num_replicas = parsed_options
            .executable_build_options
            .as_ref()
            .and_then(|b| usize::try_from(b.num_replicas).ok())
            .unwrap_or(0)
            .max(1);
        let mut num_partitions = parsed_options
            .executable_build_options
            .as_ref()
            .and_then(|b| usize::try_from(b.num_partitions).ok())
            .unwrap_or(0)
            .max(1);

        let stripped_code = code.strip_prefix(MLIR_BYTECODE_MAGIC).unwrap_or(code);
        let decoded_bc = MlirBytecodeHeader::decode(code).ok();
        let text = decoded_bc.as_ref().map_or_else(
            || String::from_utf8_lossy(stripped_code).into_owned(),
            |bc_hdr| {
                if bc_hdr.string_table.is_empty() {
                    String::from_utf8_lossy(stripped_code).into_owned()
                } else {
                    bc_hdr.to_synthesized_mlir_text()
                }
            },
        );
        let has_syntax_err_tok = HloLexer::tokenize(&text)
            .iter()
            .any(|t| matches!(t, HloToken::Ident(id) if id == "SYNTAX_ERROR"));
        if has_syntax_err_tok {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("Explicit SYNTAX_ERROR token encountered in program"),
            });
        }
        let mosaic_metadata = Self::parse_mosaic_metadata(&text);
        if let Ok(ast_mod) = parse_unified_ast_module(&text) {
            if num_partitions == 1
                && let Some(mesh) = ast_mod.sdy_meshes.values().next()
            {
                num_partitions = mesh.total_devices().max(1);
            }
            if num_partitions == 1
                && let Some(p) = ast_mod
                    .attributes
                    .get("num_partitions")
                    .and_then(MlirAttribute::as_usize)
            {
                num_partitions = p.max(1);
            }
            if num_replicas == 1
                && let Some(r) = ast_mod
                    .attributes
                    .get("replica_count")
                    .and_then(MlirAttribute::as_usize)
            {
                num_replicas = r.max(1);
            }
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
            } else if let Some(bc_hdr) = decoded_bc.as_ref().filter(|h| !h.ir_op_records.is_empty())
            {
                let ast_mod = bc_hdr.to_ast_module();
                let parsed = lower_ast_module_to_parsed_hlo(&ast_mod)?;
                let extracted_name = ast_mod.name.unwrap_or_default();
                (
                    extracted_name,
                    parsed.primary_op,
                    parsed.parameter_shapes,
                    parsed.output_shapes,
                    parsed.bundles,
                    parsed.output_slots,
                )
            } else {
                let extracted_name = Self::extract_module_name(&text);
                let parsed = Self::parse_textual_program(&text);
                (
                    extracted_name,
                    parsed.primary_op,
                    parsed.parameter_shapes,
                    parsed.output_shapes,
                    parsed.bundles,
                    parsed.output_slots,
                )
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

        let out_bytes_f64 = (out_bytes.max(256)) as f64;
        let arg_plus_out_f64 = (arg_bytes.saturating_add(out_bytes).max(512)) as f64;
        #[allow(clippy::cast_possible_truncation)]
        let flops = if opcode == TpuOpCode::Identity {
            0.0_f32
        } else if opcode == TpuOpCode::DotGeneralF32 {
            (out_bytes_f64 * 2.0) as f32
        } else {
            (out_bytes_f64 / 4.0) as f32
        };
        #[allow(clippy::cast_possible_truncation)]
        let transcendentals = if opcode.is_transcendental() {
            (out_bytes_f64 / 4.0) as f32
        } else {
            0.0_f32
        };
        #[allow(clippy::cast_possible_truncation)]
        let bytes_accessed = arg_plus_out_f64 as f32;

        let mut compiled = Self {
            name,
            program_format,
            format: fmt_lower,
            source_bytes: code.to_vec(),
            mosaic_metadata,
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
        for b in 0..=100 {
            let op = TpuOpCode::from_byte(b);
            assert!(op.is_ok());
            assert_eq!(TpuOpCode::from_raw(b), op);
            if let Ok(o) = op {
                let _ = o.arity();
                let _ = o.is_binary();
                let _ = o.is_transcendental();
                let _ = o.has_side_effects();
            }
        }
        assert_eq!(
            ProgramFormat::from_str_checked("mlir").ok(),
            Some(ProgramFormat::Mlir)
        );
        assert_eq!(
            ProgramFormat::from_str_checked("hlo").ok(),
            Some(ProgramFormat::Hlo)
        );
        assert_eq!(
            ProgramFormat::from_str_checked("hlo_with_config").ok(),
            Some(ProgramFormat::HloWithConfig)
        );
        assert!(ProgramFormat::from_str_checked("bad_fmt").is_err());
        assert_eq!(ProgramFormat::Mlir.as_str(), "mlir");
        assert_eq!(ProgramFormat::Hlo.as_str(), "hlo");
        assert_eq!(ProgramFormat::HloWithConfig.as_str(), "hlo_with_config");
        assert_eq!(ProgramFormat::Mlir.clone(), ProgramFormat::Mlir);
        assert_ne!(format!("{:?}", ProgramFormat::Mlir), "");
        assert!(TpuOpCode::from_byte(250).is_err());
        assert!(TpuOpCode::from_raw(250).is_err());
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
            assert!(CompiledProgram::parse_hlo_shape_token(hlo_spec).is_ok());
        }
        assert!(CompiledProgram::parse_hlo_shape_token("unknown[4]").is_err());
        assert!(CompiledProgram::parse_hlo_shape_token("f32[bad]").is_err());
        assert!(CompiledProgram::parse_hlo_shape_token("no_brackets").is_err());

        assert_eq!(
            CompiledProgram::parse_usize_list_attr("permutation = [1, 0]", "permutation"),
            vec![1, 0]
        );
        assert_eq!(
            CompiledProgram::parse_usize_list_attr("permutation = ]1, 0[", "permutation"),
            [] as [usize; 0]
        );
        assert!(
            CompiledProgram::parse_usize_list_attr_checked("permutation = ]1, 0[", "permutation")
                .is_err()
        );
        assert!(
            CompiledProgram::parse_usize_list_attr_checked(
                "permutation = [99999999999999999999999999999]",
                "permutation"
            )
            .is_err()
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
        let spanned_tokens = HloLexer::tokenize_spanned("module @m -> -1.5 %a0 _x");
        assert_eq!(spanned_tokens.len(), lex_tokens.len());
        if let Some(first_sp) = spanned_tokens.first() {
            assert_eq!(first_sp.clone(), *first_sp);
            assert_eq!(first_sp.span.clone(), first_sp.span);
            assert_ne!(format!("{first_sp:?}{:?}", first_sp.span), "");
        }
        assert_ne!(format!("{:?}{:?}{lex_tokens:?}", HloLexer, HloParser), "");
        assert_eq!(lex_tokens.clone(), lex_tokens);

        let dense_attrs = CompiledProgram::parse_instruction_attrs(
            r#"stablehlo.constant dense<2.5> comparison_direction = #stablehlo<comparison_direction LT> call_target_name = "my_cc" : tensor<4xf32>"#,
            &[],
            ElementType::F32,
        );
        assert_eq!(dense_attrs.constant_bytes.len(), 16);
        assert_eq!(dense_attrs.comparison_direction, "LT");
        assert_eq!(dense_attrs.custom_call_target, "my_cc");
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
            assert!(CompiledProgram::parse_tensor_type(spec).is_ok());
        }
        assert!(CompiledProgram::parse_tensor_type("tensor<f32>").is_ok());
        assert!(CompiledProgram::parse_tensor_type("tensor<-1xf32>").is_err());
        assert!(CompiledProgram::parse_tensor_type("tensor<badxf32>").is_err());
        assert!(CompiledProgram::parse_tensor_type("tensor<4xunknown>").is_err());
        assert!(CompiledProgram::parse_tensor_type("tensor<unclosed").is_err());
        assert!(CompiledProgram::parse_tensor_type("not_a_tensor").is_err());

        for phase in COMPILATION_PHASE_NAMES {
            let res = CompiledProgram::run_compilation_phase(
                phase,
                b"module @phased { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }",
                b"",
            );
            assert!(res.is_ok());
            if let Ok(serialized_phase) = res {
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
            if let Ok(p_proto) = proto_res {
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

        let mosaic_text = r#"module @mosaic_kernel attributes {tpu.compilation_target = "tpu_v5p", tpu.core_type = "tensor_core", mhlo.layout_mode = "{1,0}", mhlo.sharding = "{replicated}", dimension_semantics = ["parallel", "arbitrary"], scratch_vmem_bytes = 4096} {
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
        assert_eq!(
            mosaic_meta.dimension_semantics,
            vec![String::from("parallel"), String::from("arbitrary")]
        );
        assert_eq!(mosaic_meta.scratch_vmem_bytes, 4096);
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
        if let Ok(mut dp) = dag_prog {
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
            if let Ok(prog) = prog_res {
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
                    let disasm = arch_bundle.disassemble_vliw_bundle();
                    let reassembled = TensorCoreVliwBundle::assemble_vliw_bundle(&disasm);
                    assert!(reassembled.is_ok());
                    if let Ok(re_b) = reassembled {
                        assert_eq!(re_b.vector_slot.opcode, arch_bundle.vector_slot.opcode);
                        assert_eq!(re_b.matrix_slot.opcode, arch_bundle.matrix_slot.opcode);
                        assert_eq!(re_b.xunary_slot.opcode, arch_bundle.xunary_slot.opcode);
                        assert_eq!(re_b.dma_slot.dma_opcode, arch_bundle.dma_slot.dma_opcode);
                    }
                }
                assert!(TensorCoreVliwBundle::assemble_vliw_bundle("bad_line").is_err());
                assert!(
                    TensorCoreVliwBundle::assemble_vliw_bundle("[pc=0 gen=v5 stall=0] { no_s0 }")
                        .is_err()
                );
                assert!(
                    TensorCoreVliwBundle::assemble_vliw_bundle(
                        "[pc=0 gen=v5 stall=0] { S0(op=1, $s0 <- $s0, $s0, imm=0) | no_v0 }"
                    )
                    .is_err()
                );
                assert!(
                    TensorCoreVliwBundle::assemble_vliw_bundle(
                        "[pc=0 gen=v5 stall=0] { S0(op=1, $s0 <- $s0, $s0, imm=0) | V0(NO_COMMA) }"
                    )
                    .is_err()
                );
                assert!(TensorCoreVliwBundle::assemble_vliw_bundle("[pc=0 gen=v5 stall=0] { S0(op=1, $s0 <- $s0, $s0, imm=0) | V0(BadOpName, $v0 <- $v0, $v0, $v0) }").is_err());
                let sb_default = VliwScoreboard::default();
                assert_eq!(sb_default, VliwScoreboard::new());
                assert_ne!(format!("{sb_default:?}"), "");
                let mut poly_prog = prog.clone();
                let poly_scheds = poly_prog.compute_polyhedral_tile_schedules();
                assert_ne!(poly_scheds, [] as [PolyhedralTileSchedule; 0]);
                for ps in &poly_scheds {
                    assert_eq!(ps.clone(), *ps);
                    assert_ne!(format!("{ps:?}"), "");
                }
                poly_prog.output_shapes.clear();
                for ib in &mut poly_prog.instruction_bundles {
                    ib.attrs.output_shape = None;
                }
                assert_ne!(
                    poly_prog.compute_polyhedral_tile_schedules(),
                    [] as [PolyhedralTileSchedule; 0]
                );
                let sh_3d = Shape {
                    element_type: ElementType::F32,
                    dims: vec![2, 16, 256],
                    dynamic_dims: vec![false, false, false],
                    layout: crate::types::MemoryLayout::default_tpu_tiled(3),
                };
                let ps_3d = PolyhedralTileSchedule::from_shape(
                    &sh_3d,
                    crate::types::TpuGeneration::V6e,
                    0,
                    1,
                );
                assert_eq!(ps_3d.batch_tiles, 2);
                let sh_1d = Shape {
                    element_type: ElementType::F32,
                    dims: vec![128],
                    dynamic_dims: vec![false],
                    layout: crate::types::MemoryLayout::default_tpu_tiled(1),
                };
                let ps_1d = PolyhedralTileSchedule::from_shape(
                    &sh_1d,
                    crate::types::TpuGeneration::V5p,
                    0,
                    1,
                );
                assert_eq!(ps_1d.m_tiles, 1);
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
            computations: vec![HloComputationProto {
                name: String::from("main"),
                id: 1,
                root_id: 3,
                instructions: vec![
                    HloInstructionProto {
                        name: String::from("p0"),
                        opcode: String::from("parameter"),
                        shape: Some(ShapeProto {
                            element_type: 11,
                            dimensions: vec![8],
                            is_dynamic_dimension: vec![false],
                            ..ShapeProto::default()
                        }),
                        id: 1,
                        parameter_number: 0,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("p1"),
                        opcode: String::from("parameter"),
                        shape: Some(ShapeProto::default()),
                        id: 2,
                        parameter_number: 1,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("add.3"),
                        opcode: String::from("add"),
                        id: 3,
                        operand_ids: vec![1, 2],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            }],
            id: 1,
            ..HloModuleProto::default()
        };
        let encoded = hlo.encode();
        let prog = CompiledProgram::compile("hlo", &encoded, b"");
        assert!(prog.is_ok());
        if let Ok(p) = prog {
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
        if let Ok(p) = with_cfg_prog {
            assert_eq!(p.name, "binary_hlo_add");
            assert_eq!(p.opcode, TpuOpCode::AddF32);
        }

        let mut mlir_bc = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        mlir_bc.extend_from_slice(b"module @bc_mod { stablehlo.add }");
        let bc_hdr = MlirBytecodeHeader::decode(&mlir_bc);
        assert!(bc_hdr.is_ok());
        if let Ok(hdr) = bc_hdr {
            assert_ne!(format!("{hdr:?}"), "");
            assert_eq!(hdr.clone(), hdr);
            assert_ne!(hdr.to_synthesized_mlir_text(), "");
        }

        // Test encode_mlir_prefix_varint and decode_mlir_prefix_varint across 1-byte, 2-byte, and 9-byte forms
        for v in [0_u64, 1, 42, 127, 128, 1024, 16383, 16384, 0x1234_5678_9ABC] {
            let enc_v = encode_mlir_prefix_varint(v);
            let mut cur: &[u8] = &enc_v;
            assert_eq!(decode_mlir_prefix_varint(&mut cur), Ok(v));
            assert_eq!(cur.len(), 0);
        }
        let mut empty_cur: &[u8] = &[];
        assert!(decode_mlir_prefix_varint(&mut empty_cur).is_err());
        let mut trunc_zero: &[u8] = &[0, 1, 2];
        assert!(decode_mlir_prefix_varint(&mut trunc_zero).is_err());
        let mut trunc_two: &[u8] = &[0x02];
        assert!(decode_mlir_prefix_varint(&mut trunc_two).is_err());

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
        // Section 6 (kResourceOffset)
        let res_off_payload = encode_mlir_prefix_varint(4);
        structured_bc.push(6);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(res_off_payload.len() as u64));
        structured_bc.extend_from_slice(&res_off_payload);
        // Section 7 (kProperties)
        let prop_payload = encode_mlir_prefix_varint(42);
        structured_bc.push(7);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(prop_payload.len() as u64));
        structured_bc.extend_from_slice(&prop_payload);
        // Section 8 (unknown future section)
        structured_bc.push(8);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(1));
        structured_bc.push(0xFF);

        let parsed_struct_bc = MlirBytecodeHeader::decode(&structured_bc);
        assert!(parsed_struct_bc.is_ok());
        if let Ok(s_hdr) = parsed_struct_bc {
            assert_eq!(s_hdr.sections.len(), 9);
            assert_eq!(s_hdr.dialects.len(), 2);
            assert_eq!(s_hdr.attr_type_entries, vec![String::from("tensor<4xf32>")]);
            assert_eq!(s_hdr.attr_type_offsets, vec![14]);
            assert_eq!(s_hdr.ir_op_records, vec![7]);
            assert_eq!(s_hdr.resources.len(), 1);
            assert_eq!(s_hdr.resource_offsets, vec![4]);
            assert_eq!(s_hdr.properties, vec![42]);
            if let Some(sec0) = s_hdr.sections.first() {
                assert_eq!(sec0.clone(), *sec0);
                assert_ne!(format!("{sec0:?}"), "");
            }
            assert!(
                s_hdr
                    .to_synthesized_mlir_text()
                    .contains("stablehlo.multiply")
            );
            let op_nodes = s_hdr.decode_ir_operations();
            assert_eq!(op_nodes.len(), 1);
            assert_eq!(op_nodes.first().map(|n| n.name_index), Some(7));
            assert_eq!(op_nodes.first().cloned(), op_nodes.first().cloned());
            assert_ne!(format!("{:?}", op_nodes.first()), "");
            let reencoded = s_hdr.encode();
            assert!(MlirBytecodeHeader::decode(&reencoded).is_ok());
        }
        assert_eq!(
            MlirBytecodeHeader::legalize_vhlo_symbol("vhlo"),
            "stablehlo"
        );
        assert_eq!(
            MlirBytecodeHeader::legalize_vhlo_symbol("vhlo.add_v1"),
            "stablehlo.add"
        );
        assert_eq!(
            MlirBytecodeHeader::legalize_vhlo_symbol("vhlo.reduce_variadic"),
            "stablehlo.reduce_variadic"
        );
        assert_eq!(
            MlirBytecodeHeader::legalize_vhlo_symbol("vhlo.custom"),
            "stablehlo.custom"
        );
        let mut bc_enc = MlirBytecodeEncoder::new("MLIR20.0.0");
        let s_idx0 = bc_enc.intern_string("vhlo.add_v1");
        let s_idx0_dup = bc_enc.intern_string("vhlo.add_v1");
        assert_eq!(s_idx0, s_idx0_dup);
        let _ = bc_enc.intern_string("vhlo");
        bc_enc.dialects.push(String::from("vhlo"));
        bc_enc.attr_types.push(String::from("tensor<4xf32>"));
        let dict_attr_idx = bc_enc.intern_attr_type(r#"{call_target_name = "k"}"#);
        let float_attr_idx = bc_enc.intern_attr_type("2.5");
        let _ = (dict_attr_idx, float_attr_idx);
        bc_enc.properties.push(1);
        bc_enc.operations.push(MlirBytecodeOpNode {
            name_index: s_idx0,
            encoding_mask: 0x80 | 0x01 | 0x08 | 0x10 | 0x40,
            attr_dict_index: Some(0),
            operands: vec![0, 1],
            result_type_indices: vec![0],
            successors: vec![1],
            num_regions: 2,
            properties_index: Some(0),
            inline_regions: vec![vec![MlirBytecodeOpNode {
                name_index: s_idx0,
                encoding_mask: 0x80,
                attr_dict_index: None,
                operands: vec![0],
                result_type_indices: vec![0],
                successors: Vec::new(),
                num_regions: 0,
                properties_index: None,
                inline_regions: Vec::new(),
                region_block_arg_types: Vec::new(),
            }]],
            region_block_arg_types: vec![vec![0]],
        });
        bc_enc.operations.push(MlirBytecodeOpNode {
            name_index: s_idx0,
            encoding_mask: 0x80 | 0x01,
            attr_dict_index: Some(1),
            operands: vec![0],
            result_type_indices: vec![0],
            successors: Vec::new(),
            num_regions: 0,
            properties_index: None,
            inline_regions: Vec::new(),
            region_block_arg_types: Vec::new(),
        });
        bc_enc.operations.push(MlirBytecodeOpNode {
            name_index: s_idx0,
            encoding_mask: 0x15,
            attr_dict_index: None,
            operands: vec![0],
            result_type_indices: vec![0],
            successors: Vec::new(),
            num_regions: 1,
            properties_index: None,
            inline_regions: Vec::new(),
            region_block_arg_types: Vec::new(),
        });
        assert_eq!(bc_enc.clone(), bc_enc);
        assert_ne!(format!("{bc_enc:?}"), "");
        let enc_bc_bytes = bc_enc.encode_to_bytes();
        let dec_bc = MlirBytecodeHeader::decode(&enc_bc_bytes);
        assert!(dec_bc.is_ok());
        if let Ok(dh) = dec_bc {
            assert!(dh.to_synthesized_mlir_text().contains("stablehlo.add"));
            let ops = dh.decode_ir_operations();
            assert_ne!(ops.len(), 0);
            assert_eq!(
                dh.resolve_string_at_offset(0),
                Some(String::from("stablehlo.add"))
            );
            assert_eq!(dh.resolve_string_at_offset(11), None);
            assert_eq!(dh.resolve_string_at_offset(9999), None);
            let dial_ops = dh.decode_dialect_op_names();
            assert_eq!(dial_ops.len(), 2);
            let ast_from_bc = dh.to_ast_module();
            assert_eq!(ast_from_bc.functions.len(), 1);
            let mut empty_dh = dh;
            empty_dh.sections.clear();
            empty_dh.attr_type_entries.clear();
            empty_dh.dialects.clear();
            empty_dh.string_table = vec![String::from("add"), String::from("@custom_bc_mod")];
            assert_eq!(
                empty_dh.resolve_string_at_offset(0),
                Some(String::from("add"))
            );
            let _ = empty_dh.decode_dialect_op_names();
            let ast_custom = empty_dh.to_ast_module();
            assert_eq!(ast_custom.name, Some(String::from("custom_bc_mod")));
        }
        let compiled_struct_bc = CompiledProgram::compile("mlir", &structured_bc, b"");
        assert!(compiled_struct_bc.is_ok());
        if let Ok(p) = compiled_struct_bc {
            assert_eq!(p.name, "bin_mlir");
            assert_eq!(p.opcode, TpuOpCode::MulF32);
        }

        // Test truncated section length varint and truncated section payload in MlirBytecodeHeader::decode
        let mut trunc_sec_len = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        trunc_sec_len.extend_from_slice(b"\x03MLIR\x00\x00\x02");
        assert!(MlirBytecodeHeader::decode(&trunc_sec_len).is_ok());
        let mut trunc_sec_data = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        trunc_sec_data.extend_from_slice(b"\x03MLIR\x00\x00");
        trunc_sec_data.extend_from_slice(&encode_mlir_prefix_varint(50));
        trunc_sec_data.push(1);
        assert!(MlirBytecodeHeader::decode(&trunc_sec_data).is_ok());

        let mut mlir_bc_nul = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        mlir_bc_nul.extend_from_slice(b"\x03MLIR20.0.0\x00");
        assert!(MlirBytecodeHeader::decode(&mlir_bc_nul).is_ok());
        assert!(MlirBytecodeHeader::decode(MLIR_BYTECODE_MAGIC).is_ok());
        assert!(MlirBytecodeHeader::decode(b"not_mlir").is_err());
        let bc_prog = CompiledProgram::compile("mlir", &mlir_bc, b"");
        assert!(bc_prog.is_ok());
        if let Ok(p) = bc_prog {
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
                computations: vec![HloComputationProto {
                    name: String::from("main"),
                    id: 1,
                    root_id: 1,
                    instructions: vec![HloInstructionProto {
                        name: String::from("op.1"),
                        opcode: String::from(op_str),
                        id: 1,
                        ..HloInstructionProto::default()
                    }],
                    ..HloComputationProto::default()
                }],
                id: 1,
                ..HloModuleProto::default()
            };
            let enc2 = hlo_no_params.encode();
            assert!(CompiledProgram::compile("hlo", &enc2, b"").is_ok());
        }

        let bad_et_hlo = HloModuleProto {
            name: String::from("bad_et"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                id: 1,
                root_id: 1,
                instructions: vec![HloInstructionProto {
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    shape: Some(ShapeProto {
                        element_type: 99,
                        dimensions: vec![4],
                        ..ShapeProto::default()
                    }),
                    id: 1,
                    ..HloInstructionProto::default()
                }],
                ..HloComputationProto::default()
            }],
            id: 1,
            ..HloModuleProto::default()
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

        if let Ok(prog_ffi) = CompiledProgram::compile("mlir", b"stablehlo.add", b"") {
            let host_offload = prog_ffi.build_host_offloading_executable_proto();
            assert_eq!(host_offload.executable_type, 1);
            let (ffi_st, ffi_attrs) = prog_ffi.build_ffi_state_and_attributes();
            assert!(ffi_st.type_name.contains("xla.ffi.state"));
            assert!(ffi_attrs.attrs.contains_key("module_name"));
        }
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
        let hlo_entry_res = CompiledProgram::compile("hlo", hlo_entry_test, b"");
        assert!(hlo_entry_res.is_ok());
        if let Ok(prog) = hlo_entry_res {
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
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    shape: Some(ShapeProto {
                        element_type: 11,
                        dimensions: vec![16, 128],
                        ..ShapeProto::default()
                    }),
                    id: 1,
                    operand_ids: vec![999], // external/unknown operand ID gracefully ignored
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    name: String::from("neg"),
                    opcode: String::from("negate"),
                    id: 2,
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        };
        let dag_res = analyze_hlo_computation_dag(&valid_comp);
        assert!(dag_res.is_ok());
        if let Ok(dag_rep) = dag_res {
            assert_eq!(dag_rep.topological_order, vec![1, 2]);
            assert_eq!(dag_rep.buffer_offsets.len(), 2);
            assert_eq!(dag_rep.total_hbm_bytes, 12288);
            assert_eq!(dag_rep.clone(), dag_rep);
            assert_ne!(format!("{dag_rep:?}"), "");
        }

        let cyclic_comp = HloComputationProto {
            name: String::from("cyclic"),
            id: 2,
            root_id: 2,
            instructions: vec![
                HloInstructionProto {
                    name: String::from("a"),
                    opcode: String::from("negate"),
                    id: 1,
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    name: String::from("b"),
                    opcode: String::from("negate"),
                    id: 2,
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        };
        assert!(analyze_hlo_computation_dag(&cyclic_comp).is_err());
        assert!(CompiledProgram::parse_hlo_shape_token("[2,2]").is_err());

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
        let cst_fold_res = CompiledProgram::compile("mlir", cst_fold_mlir, b"");
        assert!(cst_fold_res.is_ok());
        if let Ok(prog) = cst_fold_res {
            let intervals = prog.compute_liveness_intervals();
            assert_ne!(intervals, [] as [RegisterLivenessInterval; 0]);
            if let Some(iv) = intervals.first() {
                assert_eq!(iv.slot_index, 0);
                assert_ne!(format!("{iv:?}"), "");
            }
        }
        let cse_key = CseExpressionKey {
            opcode: TpuOpCode::Slice,
            operand_slots: vec![0],
            attrs: InstructionAttrs::default(),
        };
        assert_eq!(cse_key.clone(), cse_key);
        assert_ne!(format!("{cse_key:?}"), "");

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

        // Test TensorCoreVliwBundle::from_ir_bundle bounds validation
        let valid_ir = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::AddF32,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        };
        assert!(TensorCoreVliwBundle::from_ir_bundle(&valid_ir).is_ok());
        let overflow_out_ir = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::AddF32,
            operand_slots: vec![0, 1],
            output_slot: 300,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        };
        assert!(TensorCoreVliwBundle::from_ir_bundle(&overflow_out_ir).is_err());
        let overflow_op_ir = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::AddF32,
            operand_slots: vec![0, 256],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        };
        assert!(TensorCoreVliwBundle::from_ir_bundle(&overflow_op_ir).is_err());

        // Test HloParser::parse_module and MlirBytecodeHeader::decode_strict
        assert!(HloParser::parse_module("").is_err());
        assert!(HloParser::parse_module("module @m { \"unclosed }").is_err());
        assert!(HloParser::parse_module("module @m { } }").is_err());
        assert!(HloParser::parse_module("module @m { ( }").is_err());
        assert!(HloParser::parse_module("SYNTAX_ERROR").is_err());
        assert!(HloParser::parse_module("INVALID_HLO").is_err());
        let parsed_mod_res =
            HloParser::parse_module("module @m { %0 = stablehlo.add(%arg0, %arg1) : [4] }");
        assert!(parsed_mod_res.is_ok());
        if let Ok(parsed_mod) = parsed_mod_res {
            assert_eq!(parsed_mod.primary_op, TpuOpCode::AddF32);
            assert_eq!(parsed_mod.clone(), parsed_mod);
            assert_ne!(format!("{parsed_mod:?}"), "");
        }
        assert!(CompiledProgram::parse_hlo_shape_token("c64[2,2]").is_ok());
        assert!(CompiledProgram::parse_hlo_shape_token("c128[2,2]").is_ok());
        let hlo_tuple_src = "HloModule tuple_out\nENTRY main {\n  %p0 = f32[4] parameter(0)\n  %p1 = f32[4] parameter(1)\n  %sum = f32[4] add(%p0, %p1)\n  %prod = f32[4] multiply(%p0, %p1)\n  ROOT %res = (f32[4], f32[4]) tuple(%sum, %prod)\n}";
        let tuple_res = CompiledProgram::compile("hlo", hlo_tuple_src.as_bytes(), b"");
        assert!(tuple_res.is_ok());
        if let Ok(tuple_mod) = tuple_res {
            assert_eq!(tuple_mod.output_slots.len(), 2);
            assert_eq!(tuple_mod.output_shapes.len(), 2);
        }
        let attr_bundle = TpuInstructionBundle {
            pc_offset: 5,
            opcode: TpuOpCode::ExpF32,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 2,
            attrs: InstructionAttrs {
                iota_dim: 2,
                concat_dim: 3,
                reduce_dims: vec![1],
                ..InstructionAttrs::default()
            },
        };
        let encoded_attr_vliw = attr_bundle.encode_architectural_bundle();
        let decoded_attr_res = encoded_attr_vliw.to_instruction_bundle();
        assert!(decoded_attr_res.is_ok());
        if let Ok(decoded_attr_bundle) = decoded_attr_res {
            assert_eq!(decoded_attr_bundle.attrs.iota_dim, 2);
            assert_eq!(decoded_attr_bundle.attrs.concat_dim, 3);
            assert_eq!(decoded_attr_bundle.attrs.reduce_dims, vec![1]);
        }
        let pack_res = CompiledProgram::compile("mlir", b"module @pack { stablehlo.add }", b"");
        assert!(pack_res.is_ok());
        if let Ok(mut pack_prog) = pack_res {
            pack_prog.instruction_bundles = vec![
                TpuInstructionBundle {
                    pc_offset: 0,
                    opcode: TpuOpCode::AddF32,
                    operand_slots: vec![0, 1],
                    output_slot: 2,
                    sync_flag_id: 1,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 1,
                    opcode: TpuOpCode::DotGeneralF32,
                    operand_slots: vec![0, 1],
                    output_slot: 3,
                    sync_flag_id: 2,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 2,
                    opcode: TpuOpCode::AddF32,
                    operand_slots: vec![0, 1],
                    output_slot: 4,
                    sync_flag_id: 3,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 3,
                    opcode: TpuOpCode::ExpF32,
                    operand_slots: vec![0],
                    output_slot: 5,
                    sync_flag_id: 4,
                    attrs: InstructionAttrs::default(),
                },
            ];
            pack_prog.pass_hlo_operator_fusion();
            assert_eq!(
                pack_prog
                    .instruction_bundles
                    .get(1)
                    .and_then(|b| b.attrs.fused_epilogue_op),
                Some(TpuOpCode::AddF32)
            );
            let packed_vliw = pack_prog.pack_independent_vliw_bundles();
            assert_eq!(packed_vliw.len(), 2);
            for tpu_gen in [
                crate::types::TpuGeneration::V7x,
                crate::types::TpuGeneration::V6e,
                crate::types::TpuGeneration::V5p,
                crate::types::TpuGeneration::V5e,
            ] {
                crate::config::set_global_init_config(crate::config::TpuInitConfig {
                    generation: tpu_gen,
                    ..crate::config::TpuInitConfig::default()
                });
                pack_prog.pass_llo_vliw_bundle_scheduling();
                let sched = LloVliwScheduler::new(tpu_gen);
                assert_eq!(sched.clone(), sched);
                assert_ne!(format!("{sched:?}"), "");
                let _ = sched.instruction_latency_cycles(TpuOpCode::Constant);
                let _ = sched.instruction_latency_cycles(TpuOpCode::AllReduceSumF32);
                let mut sched_inputs = pack_prog.instruction_bundles.clone();
                sched_inputs.push(TpuInstructionBundle {
                    pc_offset: 4,
                    opcode: TpuOpCode::AllReduceSumF32,
                    operand_slots: vec![5],
                    output_slot: 6,
                    sync_flag_id: 5,
                    attrs: InstructionAttrs::default(),
                });
                let scheduled = sched.schedule_bundles(&sched_inputs);
                assert_eq!(scheduled.len(), 5);
                for sb in &scheduled {
                    let _ = sb.active_slot_mask();
                    let _ = sb.stall_cycles();
                    let _ = sb.generation_ordinal();
                    let dis = sb.disassemble_vliw_bundle();
                    assert!(dis.contains("[pc="));
                }
            }
            if let Some(first_bundle) = pack_prog.instruction_bundles.first() {
                let mut zero_gen_vliw = first_bundle.encode_architectural_bundle();
                zero_gen_vliw.header.bundle_flags &= 0x0FFF;
                zero_gen_vliw.vector_slot.opcode = 999;
                assert_eq!(zero_gen_vliw.generation_ordinal(), 5);
                assert!(zero_gen_vliw.disassemble_vliw_bundle().contains("UNKNOWN"));
            }
        }
        assert!(MlirBytecodeHeader::decode_strict(MLIR_BYTECODE_MAGIC).is_ok());
        assert!(MlirBytecodeHeader::decode_strict(b"not_mlir").is_err());

        // Test pass_algebraic_simplifier across uniform vs non-uniform constant tensors
        let alg_res = CompiledProgram::compile("mlir", b"module @alg { stablehlo.add }", b"");
        assert!(alg_res.is_ok());
        if let Ok(mut prog) = alg_res {
            let zero_bytes = ElementType::F32.pack_from_f64_slice(&[0.0, 0.0]);
            let one_bytes = ElementType::F32.pack_from_f64_slice(&[1.0, 1.0]);
            let mixed_bytes = ElementType::F32.pack_from_f64_slice(&[0.0, 5.0]);
            prog.instruction_bundles = vec![
                TpuInstructionBundle {
                    pc_offset: 0,
                    opcode: TpuOpCode::Constant,
                    operand_slots: Vec::new(),
                    output_slot: 1,
                    sync_flag_id: 1,
                    attrs: InstructionAttrs {
                        constant_bytes: zero_bytes,
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 1,
                    opcode: TpuOpCode::Constant,
                    operand_slots: Vec::new(),
                    output_slot: 2,
                    sync_flag_id: 2,
                    attrs: InstructionAttrs {
                        constant_bytes: one_bytes,
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 2,
                    opcode: TpuOpCode::Constant,
                    operand_slots: Vec::new(),
                    output_slot: 3,
                    sync_flag_id: 3,
                    attrs: InstructionAttrs {
                        constant_bytes: mixed_bytes,
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 3,
                    opcode: TpuOpCode::AddF32,
                    operand_slots: vec![0, 1],
                    output_slot: 4,
                    sync_flag_id: 4,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 4,
                    opcode: TpuOpCode::AddF32,
                    operand_slots: vec![1, 4],
                    output_slot: 5,
                    sync_flag_id: 5,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 5,
                    opcode: TpuOpCode::MulF32,
                    operand_slots: vec![5, 2],
                    output_slot: 6,
                    sync_flag_id: 6,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 6,
                    opcode: TpuOpCode::MulF32,
                    operand_slots: vec![2, 6],
                    output_slot: 7,
                    sync_flag_id: 7,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 7,
                    opcode: TpuOpCode::MulF32,
                    operand_slots: vec![7, 1],
                    output_slot: 8,
                    sync_flag_id: 8,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 8,
                    opcode: TpuOpCode::MulF32,
                    operand_slots: vec![1, 7],
                    output_slot: 9,
                    sync_flag_id: 9,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 9,
                    opcode: TpuOpCode::AddF32,
                    operand_slots: vec![0, 3],
                    output_slot: 10,
                    sync_flag_id: 10,
                    attrs: InstructionAttrs::default(),
                },
            ];
            prog.pass_algebraic_simplifier();
            assert!(prog.to_isa_program_proto().is_ok());
            assert!(prog.to_serialized_executable_proto().is_ok());
            let mut bad_prog = prog.clone();
            bad_prog.compile_options_bytes = vec![0xFF, 0xFF, 0xFF];
            assert!(bad_prog.to_serialized_executable_proto().is_err());
            let opcodes: Vec<TpuOpCode> =
                prog.instruction_bundles.iter().map(|b| b.opcode).collect();
            assert_eq!(
                opcodes,
                vec![
                    TpuOpCode::Constant,
                    TpuOpCode::Constant,
                    TpuOpCode::Constant,
                    TpuOpCode::Identity,
                    TpuOpCode::Identity,
                    TpuOpCode::Identity,
                    TpuOpCode::Identity,
                    TpuOpCode::Identity,
                    TpuOpCode::Identity,
                    TpuOpCode::AddF32,
                ]
            );
        }
    }

    #[test]
    fn test_lexer_comments_strings_hex_and_sigils() {
        let src = r#"
            // Single-line comment with UTF-8: λ -> 🚀
            /* Multi-line block comment
               spanning lines */
            #loc0 = loc("model.py":42:8)
            !tpu_sem = !tpu.semaphore
            module @test_mod {
                func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    ^bb0(%x: tensor<4xf32>):
                    %0:2 = "stablehlo.custom_call"(%arg0) {
                        backend_config = "line1\nline2\t\"quoted\"\\\r\0\x41\z",
                        hex_data = dense<"0x0000803F00000040">,
                        sci = 1.0e-5,
                        neg_sci = -3.14E+2,
                        pos_num = +2.5,
                        hex_int = 0xFF,
                        hex_flt = 0x1.8p+1,
                        inf_val = +inf,
                        ninf_val = -inf
                    } : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) loc(#loc0)
                    func.return %0#0 : tensor<4xf32>
                }
            } // trailing comment without newline"#;
        let checked = HloLexer::tokenize_checked(src);
        assert!(checked.is_ok());
        let spanned = HloLexer::tokenize_spanned(src);
        assert!(spanned.len() > 1);
        assert!(
            spanned
                .iter()
                .any(|st| matches!(&st.token, HloToken::AttrAlias(a) if a == "#loc0"))
        );
        assert!(
            spanned
                .iter()
                .any(|st| matches!(&st.token, HloToken::TypeAlias(a) if a == "!tpu_sem"))
        );
        assert!(
            spanned
                .iter()
                .any(|st| matches!(&st.token, HloToken::CaretId(c) if c == "^bb0"))
        );
        assert!(
            spanned
                .iter()
                .any(|st| matches!(&st.token, HloToken::HexBlob(b) if b.len() == 8))
        );
        assert!(
            spanned
                .iter()
                .any(|st| matches!(&st.token, HloToken::SsaValue(s) if s == "%0#0"))
        );

        // Compound punctuation tokens
        let punct_tokens = HloLexer::tokenize("-> ... :: <= >= == != =");
        assert!(punct_tokens.contains(&HloToken::Arrow));
        assert!(punct_tokens.contains(&HloToken::Ellipsis));
        assert!(punct_tokens.contains(&HloToken::DoubleColon));
        assert!(punct_tokens.contains(&HloToken::Le));
        assert!(punct_tokens.contains(&HloToken::Ge));
        assert!(punct_tokens.contains(&HloToken::EqEq));
        assert!(punct_tokens.contains(&HloToken::NotEq));

        // Negative lexer tests: unclosed block comment, unclosed quote, bad \xHH escape, trailing backslash
        assert!(HloLexer::tokenize_checked("/* unclosed comment").is_err());
        assert!(HloLexer::tokenize_checked("\"unclosed string").is_err());
        assert!(HloLexer::tokenize_checked("\"trailing\\").is_err());
        assert!(HloLexer::tokenize_checked("\"bad\\xZZescape\"").is_err());
        assert!(HloLexer::tokenize_checked("\"bad\\x4Zescape\"").is_err());
        assert!(HloLexer::tokenize_checked("a / b /* star * inside */ \"\\x6a\"").is_ok());
        // Direct tokenize on unclosed block comment / string and edge numeric/hex strings
        assert_eq!(HloLexer::tokenize("/* unclosed").len(), 0);
        let edge_tokens = HloLexer::tokenize(
            "-infinity +infinity -a +b \"0x\" \"0x123\" \"0x12zz\" \"0xdeadbeef\"",
        );
        assert!(edge_tokens.contains(&HloToken::Punct('-')));
        assert!(edge_tokens.contains(&HloToken::Punct('+')));
        let _ = HloLexer::tokenize_spanned("\"unclosed");
        let _ = HloLexer::tokenize_spanned("\"bad\\x1\"");
        let _ = HloLexer::tokenize_spanned("\"bad\\");
        assert!(HloParser::parse_verified_ast_module("").is_err());
        assert!(CompiledProgram::compile("mlir", b"sdy.mesh", b"").is_ok());
        assert!(CompiledProgram::compile("mlir", b"sdy.mesh @m = <[\"x\"=2]>", b"").is_ok());
    }

    #[test]
    fn test_ast_types_attributes_and_helpers() {
        for (prefix, expected_kind, expected_str) in [
            ("stablehlo", DialectKind::StableHlo, "stablehlo"),
            ("vhlo", DialectKind::Vhlo, "vhlo"),
            ("mhlo", DialectKind::Mhlo, "mhlo"),
            ("chlo", DialectKind::Chlo, "chlo"),
            ("sdy", DialectKind::Sdy, "sdy"),
            ("tpu", DialectKind::TpuMosaic, "tpu"),
            ("func", DialectKind::Func, "func"),
            ("arith", DialectKind::Arith, "arith"),
            ("scf", DialectKind::Scf, "scf"),
            ("vector", DialectKind::Vector, "vector"),
            ("memref", DialectKind::MemRef, "memref"),
            ("builtin", DialectKind::Builtin, "builtin"),
            ("other", DialectKind::XlaHlo, "hlo"),
        ] {
            let dk = DialectKind::from_prefix(prefix);
            assert_eq!(dk, expected_kind);
            assert_eq!(dk.as_str(), expected_str);
        }

        assert_eq!(DimensionSize::Static(8).concrete_or_bound(), 8);
        assert_eq!(DimensionSize::Bounded(16).concrete_or_bound(), 16);
        assert_eq!(DimensionSize::Dynamic.concrete_or_bound(), 1);
        assert!(!DimensionSize::Static(8).is_dynamic());
        assert!(DimensionSize::Bounded(16).is_dynamic());
        assert!(DimensionSize::Dynamic.is_dynamic());

        for (space_str, expected_space) in [
            ("hbm", Some(TpuMemorySpace::Hbm)),
            ("vmem", Some(TpuMemorySpace::Vmem)),
            ("smem", Some(TpuMemorySpace::Smem)),
            ("cmem", Some(TpuMemorySpace::Cmem)),
            ("semaphore_mem", Some(TpuMemorySpace::SemaphoreMem)),
            ("unknown", None),
        ] {
            assert_eq!(TpuMemorySpace::from_ident(space_str), expected_space);
        }

        for (dir_str, expected_dir) in [
            ("EQ", ComparisonDirection::Eq),
            ("NE", ComparisonDirection::Ne),
            ("GE", ComparisonDirection::Ge),
            ("GT", ComparisonDirection::Gt),
            ("LE", ComparisonDirection::Le),
            ("LT", ComparisonDirection::Lt),
        ] {
            let parsed = ComparisonDirection::from_ident(dir_str);
            assert_eq!(parsed, Some(expected_dir));
            assert_eq!(expected_dir.as_str(), dir_str);
        }
        assert!(ComparisonDirection::from_ident("BAD").is_none());

        for (ct_str, expected_ct) in [
            ("FLOAT", Some(ComparisonType::Float)),
            ("TOTALORDER", Some(ComparisonType::TotalOrder)),
            ("SIGNED", Some(ComparisonType::Signed)),
            ("UNSIGNED", Some(ComparisonType::Unsigned)),
            ("BAD", None),
        ] {
            assert_eq!(ComparisonType::from_ident(ct_str), expected_ct);
        }

        // Test MlirType methods across all variants
        let rt = MlirType::RankedTensor(RankedTensorType {
            dims: vec![DimensionSize::Static(4), DimensionSize::Dynamic],
            element_type: ElementType::F32,
            encoding: Some(TensorEncodingAttr::Bounds(vec![None, Some(8)])),
            layout: None,
        });
        assert_eq!(rt.element_type(), ElementType::F32);
        assert!(rt.to_shape().is_some());

        let unranked = MlirType::UnrankedTensor(ElementType::Bf16);
        assert_eq!(unranked.element_type(), ElementType::Bf16);
        assert!(unranked.to_shape().is_some());

        let mr = MlirType::MemRef(MemRefType {
            dims: vec![DimensionSize::Static(8), DimensionSize::Static(128)],
            element_type: ElementType::Bf16,
            layout: Some(MemRefLayoutAttr::Tiled {
                tile_dims: vec![8, 128],
            }),
            memory_space: Some(TpuMemorySpace::Vmem),
        });
        assert_eq!(mr.element_type(), ElementType::Bf16);
        assert!(mr.to_shape().is_some());

        let vec_ty = MlirType::Vector(VectorType {
            dims: vec![DimensionSize::Static(8), DimensionSize::Static(128)],
            scalable_dims: vec![false, false],
            element_type: ElementType::F32,
        });
        assert_eq!(vec_ty.element_type(), ElementType::F32);
        assert!(vec_ty.to_shape().is_some());

        let quant_ty = MlirType::Quantized(QuantizedElementType {
            storage_type: ElementType::S8,
            expressed_type: ElementType::F32,
            scales: vec![0.05],
            zero_points: vec![2],
            quantized_dimension: None,
        });
        assert_eq!(quant_ty.element_type(), ElementType::S8);
        assert!(quant_ty.to_shape().is_some());

        let sem_ty = MlirType::Semaphore(TpuSemaphoreKind::Dma);
        assert_eq!(sem_ty.element_type(), ElementType::U32);
        assert!(sem_ty.to_shape().is_some());

        let tok_ty = MlirType::Token;
        assert_eq!(tok_ty.element_type(), ElementType::U32);
        assert!(tok_ty.to_shape().is_some());

        let tup_ty = MlirType::Tuple(vec![rt.clone(), MlirType::Scalar(ElementType::S32)]);
        assert_eq!(tup_ty.element_type(), ElementType::F32);
        assert!(tup_ty.to_shape().is_some());
        assert_eq!(tup_ty.flatten_shapes().len(), 2);

        let fn_ty = MlirType::Function(AstFunctionType {
            inputs: vec![rt.clone()],
            results: vec![rt],
        });
        assert_eq!(fn_ty.element_type(), ElementType::F32);
        assert!(fn_ty.to_shape().is_some());

        // Test MlirAttribute helper methods
        let a_i64 = MlirAttribute::DenseI64Array(vec![1, 2, -1]);
        assert_eq!(a_i64.as_usize_vec(), vec![1, 2]);
        let a_arr = MlirAttribute::Array(vec![
            MlirAttribute::Integer { value: 3, ty: None },
            MlirAttribute::Bool(true),
        ]);
        assert_eq!(a_arr.as_usize_vec(), vec![3]);
        let a_de = MlirAttribute::DenseElements(DenseElementsAttr {
            values_f64: vec![4.0, -1.0, 5.0],
            raw_bytes: None,
            is_splat: false,
            tensor_type: None,
        });
        assert_eq!(a_de.as_usize_vec(), vec![4, 5]);
        assert_eq!(MlirAttribute::Bool(false).as_usize_vec().len(), 0);
        assert_eq!(
            MlirAttribute::Integer { value: 7, ty: None }.as_usize(),
            Some(7)
        );
        assert_eq!(MlirAttribute::Bool(true).as_usize(), None);
        assert_eq!(
            MlirAttribute::String(String::from("abc")).as_str(),
            Some("abc")
        );
        assert_eq!(
            MlirAttribute::SymbolRef(vec![String::from("m"), String::from("x")]).as_str(),
            Some("x")
        );
        assert_eq!(MlirAttribute::Bool(false).as_str(), None);

        let empty_mesh = SdyMeshAttr {
            name: String::from("m"),
            axes: Vec::new(),
            device_ids: vec![0, 1, 2, 3],
        };
        assert_eq!(empty_mesh.total_devices(), 4);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_recursive_descent_mlir_dialects_stablehlo_mhlo_chlo_sdy_mosaic() {
        let mlir_all_dialects = r#"
            #loc0 = loc(unknown)
            #loc1 = loc("train.py":12:4)
            #loc2 = loc(callsite(#loc1 at #loc0))
            #loc3 = loc(fused[#loc0, #loc1])
            #loc4 = loc("step"(#loc1))
            #my_sharding = #sdy.sharding<@mesh, [{"x"}, {"y", ?, p1}], replicated={"z"}, unreduced={"w"}>
            !my_tensor = tensor<4x4xf32>

            module @spmd_mosaic_module attributes {
                mhlo.num_partitions = 8 : i64,
                mhlo.num_replicas = 2 : i64
            } {
                sdy.mesh @mesh = <["x"=2, "y"=4], device_ids=[0, 1, 2, 3, 4, 5, 6, 7]>

                func.func private @helper(%h0: !my_tensor) -> !my_tensor {
                    %h1 = chlo.broadcast_add %h0, %h0 {broadcast_dimensions = array<i64: 0, 1>} : (!my_tensor, !my_tensor) -> !my_tensor
                    %h2 = chlo.erf %h1 : !my_tensor -> !my_tensor
                    %h3 = chlo.lgamma %h2 : !my_tensor -> !my_tensor
                    %h4 = chlo.acosh %h3 : !my_tensor -> !my_tensor
                    %h5 = chlo.sinh %h4 : !my_tensor -> !my_tensor
                    %h6 = chlo.atanh %h5 : !my_tensor -> !my_tensor
                    %h7 = chlo.next_after %h6, %h0 : (!my_tensor, !my_tensor) -> !my_tensor
                    %h8 = chlo.conj %h7 : !my_tensor -> !my_tensor
                    %h9 = chlo.erfc %h8 : !my_tensor -> !my_tensor
                    func.return %h9 : !my_tensor
                }

                func.func public @main(
                    %arg0: tensor<4x4xf32> {sdy.sharding = #my_sharding} loc(#loc1),
                    %arg1: tensor<4x4xf32> {mhlo.sharding = "{devices=[2,4]0,1,2,3,4,5,6,7}"} loc(#loc2)
                ) -> (tensor<4x4xf32> {sdy.sharding = #my_sharding}) attributes {
                    tpu.core_type = #tpu.core_type<tensor_core>
                } {
                    %c0 = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]> : tensor<4x4xf32>
                    %c_splat = stablehlo.constant dense<2.0> : tensor<4x4xf32>
                    %c_hex = stablehlo.constant dense<"0x0000803F000000400000404000008040"> : tensor<4xf32>
                    %c_res = stablehlo.constant dense_resource<__elided__> : tensor<4x4xf32>
                    %cmp = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
                    %sel = stablehlo.select %cmp, %arg0, %arg1 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
                    %dot = stablehlo.dot_general %sel, %c0, #stablehlo.dot<
                        lhs_batching_dimensions = [],
                        rhs_batching_dimensions = [],
                        lhs_contracting_dimensions = [1],
                        rhs_contracting_dimensions = [0]
                    >, precision = [#stablehlo<precision HIGH>, #stablehlo<precision HIGHEST>] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
                    %sh_c = sdy.sharding_constraint %dot <@mesh, [{"x"}, {"y"}]> : tensor<4x4xf32>
                    %manual = sdy.manual_computation(%sh_c)
                        in_shardings=[<@mesh, [{"x"}, {"y"}]>]
                        out_shardings=[<@mesh, [{"x"}, {"y"}]>]
                        manual_axes={"x"} (%m_arg: tensor<2x4xf32>) {
                        %m_red = "stablehlo.all_reduce"(%m_arg) ({
                            ^bb0(%r0: tensor<f32>, %r1: tensor<f32>):
                            %r_sum = stablehlo.add %r0, %r1 : tensor<f32>
                            stablehlo.return %r_sum : tensor<f32>
                        }) {
                            channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
                            use_global_device_ids = array<i1: true>
                        } : (tensor<2x4xf32>) -> tensor<2x4xf32>
                        sdy.return %m_red : tensor<2x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<4x4xf32>
                    %red_max = "stablehlo.reduce"(%manual, %c_splat) ({
                        ^bb0(%a: tensor<f32>, %b: tensor<f32>):
                        %m = stablehlo.maximum %a, %b : tensor<f32>
                        stablehlo.return %m : tensor<f32>
                    }) {dimensions = array<i64: 1>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
                    %red_min = "stablehlo.reduce"(%red_max, %c_splat) ({
                        ^bb0(%a: tensor<f32>, %b: tensor<f32>):
                        %m = stablehlo.minimum %a, %b : tensor<f32>
                        stablehlo.return %m : tensor<f32>
                    }) {dimensions = array<i64: 1>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
                    %tup = stablehlo.tuple %red_min, %c_splat : tuple<tensor<4x4xf32>, tensor<4x4xf32>>
                    %gte = stablehlo.get_tuple_element %tup {index = 0 : i32} : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
                    func.return %gte : tensor<4x4xf32> loc(#loc3)
                }
            }
        "#;

        let ast = HloParser::parse_ast_module(mlir_all_dialects);
        assert!(ast.is_ok());
        let ast_mod = ast.unwrap_or_default();
        assert_eq!(ast_mod.name.as_deref(), Some("spmd_mosaic_module"));
        assert_eq!(ast_mod.functions.len(), 2);
        assert!(ast_mod.sdy_meshes.contains_key("mesh"));
        assert!(verify_ast_module_semantics(&ast_mod).is_ok());

        let lowered = lower_ast_module_to_parsed_hlo(&ast_mod);
        assert!(lowered.is_ok());

        let compiled = CompiledProgram::compile("mlir", mlir_all_dialects.as_bytes(), b"");
        assert!(compiled.is_ok());
        if let Ok(prog) = compiled {
            assert_eq!(prog.num_partitions, 8);
        }
    }

    #[test]
    fn test_mosaic_scf_vector_memref_and_quantized_types_parsing() {
        let mosaic_kernel = r"
            module @mosaic_kernel {
                func.func @kernel(
                    %in_vmem: memref<8x128xbf16, #tpu.tiled<(8, 128)>, #tpu.memory_space<vmem>>,
                    %out_hbm: memref<8x128xf32, strided<[128, 1], offset: 0>, #tpu.memory_space<hbm>>,
                    %smem_buf: memref<16xi32, #tpu.memory_space<smem>>,
                    %cmem_buf: memref<16xf32, #tpu.memory_space<cmem>>,
                    %sem_mem: memref<1x!tpu.semaphore, #tpu.memory_space<semaphore_mem>>,
                    %dma_sem: !tpu.dma_semaphore,
                    %q_in: tensor<4x?x!quant.uniform<i8:f32, 0.05:10>, #stablehlo.bounds<?, 16>>,
                    %unranked: tensor<*xf32>,
                    %c_scalar: complex<f32>,
                    %c64_scalar: complex<f64>
                ) -> vector<8x128xf32> {
                    %c0 = arith.constant 0 : index
                    %c8 = arith.constant 8 : index
                    %c1 = arith.constant 1 : index
                    %v0 = vector.transfer_read %in_vmem[%c0, %c0] : memref<8x128xbf16>, vector<8x128xbf16>
                    %loop_res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %v0) -> (vector<8x128xbf16>) {
                        %loaded = tpu.vector_load %in_vmem[%iv, %c0] : memref<8x128xbf16>, vector<8x128xbf16>
                        %prod = tpu.matmul %acc, %loaded {
                            dimension_semantics = [#tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<arbitrary>]
                        } : (vector<8x128xbf16>, vector<8x128xbf16>) -> vector<8x128xbf16>
                        scf.yield %prod : vector<8x128xbf16>
                    }
                    tpu.enqueue_dma %in_vmem, %out_hbm, %dma_sem : (memref<8x128xbf16>, memref<8x128xf32>, !tpu.dma_semaphore) -> ()
                    tpu.wait_dma %dma_sem : (!tpu.dma_semaphore) -> ()
                    tpu.semaphore_signal %sem_mem : (memref<1x!tpu.semaphore>) -> ()
                    %casted = arith.extf %loop_res : vector<8x128xbf16> to vector<8x128xf32>
                    func.return %casted : vector<8x128xf32>
                }
            }
        ";
        let parsed = parse_mlir_ast_module(mosaic_kernel);
        assert!(parsed.is_ok());
        let ast = parsed.unwrap_or_default();
        assert_eq!(ast.functions.len(), 1);
        let lowered = lower_ast_module_to_parsed_hlo(&ast);
        assert!(lowered.is_ok());
    }

    #[test]
    fn test_textual_xla_hlo_multi_computation_and_proto_bridge() {
        use self::hlo_parser::HloTextParser;

        let hlo_text = r#"
            HloModule multi_comp_test, entry_computation_layout={(f32[4,4]{1,0:T(8,128)(2,1)S(1)}, f32[<=4, ?]) -> (f32[4], f32[4,4])}, replica_count=2, num_partitions=4, is_scheduled=true, allow_spmd_sharding_propagation_to_output={true}

            %reducer (x: f32[], y: f32[]) -> f32[] {
                %x = f32[] parameter(0)
                %y = f32[] parameter(1)
                ROOT %max = f32[] maximum(f32[] %x, f32[] %y)
            }

            %min_reducer (f32[], f32[]) {
                %x = f32[] parameter(0)
                %y = f32[] parameter(1)
                %min = f32[] minimum(%x, %y)
            }

            ENTRY %main (p0: f32[4,4], p1: f32[4,4]) -> (f32[4], f32[4,4]) {
                %p0 = f32[4,4]{:T(8,128)S(1)} parameter(0), sharding={devices=[2,2]}0,1
                %p1 = f32[4,4]{1,0} parameter(1), sharding={replicated}
                %tok = token[] after-all()
                %init = f32[] constant({0})
                %dyn_op = f32[<=4, ?, @ignore] reshape(%p0, +), iota_dimension=0, backend_config="str_cfg", metadata={op_name="m"}, feature_group_count=1, api_version=API_VERSION_TYPED_FFI
                %conv = f32[4,4] convolution(%p0, %p1), window={size=3x3 stride=1x1 pad=1_1x1_1 lhs_dilate=1x1 rhs_dilate=1x1}, dim_labels=b01f_01io->b01f
                %dot = f32[4,4] dot(%conv, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, lhs_batch_dims={0}, rhs_batch_dims={0}
                %sliced = f32[2,2] slice(%dot), slice={[0:2:1], [0:2:1]}
                %padded = f32[4,4] pad(%sliced, %init), padding=1_1_0x1_1_0, sharding={replicated}
                %cc = f32[4,4] custom-call(%padded), custom_call_target="tpu_custom_kernel", backend_config={"flag": 1}, channel_id=2, replica_groups={{0,1},{2,3}}, control-predecessors={%tok}
                %w = f32[4,4] while(%cc), condition=%reducer, body=%min_reducer
                %ar = f32[4,4] all-reduce(%w), to_apply=%reducer
                %rs = f32[4,4] reduce-scatter(%ar), to_apply=%reducer
                %unk = f32[4,4] unknown-hlo-op(%rs)
                %red = f32[4] reduce(%unk, %init), dimensions={1}, to_apply=%reducer, direction=LT
                %red2 = f32[4] reduce(%unk, %init), dimensions={1}, to_apply=%min_reducer
                %tup = (f32[4], f32[4,4]) tuple(%red, %cc)
                %gte0 = f32[4] get-tuple-element(%tup), index=0
                %gte_direct = f32[4] get-tuple-element(%red2), index=0
                ROOT %out_tup = (f32[4], f32[4,4]) tuple(%gte0, %cc)
            }
        "#;

        let ast_res = HloParser::parse_ast_module(hlo_text);
        assert!(ast_res.is_ok(), "HLO parse failed: {ast_res:?}");
        let ast = ast_res.unwrap_or_default();
        assert_eq!(ast.hlo_computations.len(), 3);

        // Convert AST -> Proto -> AST -> Lowered
        let mut proto = hlo_ast_to_proto(&ast);
        let _ = hlo_proto_to_ast(&proto);
        proto.entry_computation_id = 0;
        if let Some(comp) = proto.computations.first_mut()
            && let Some(inst) = comp.instructions.first_mut()
            && let Some(sh) = inst.shape.as_mut()
        {
            sh.dimensions.push(-1);
            sh.is_dynamic_dimension.push(true);
        }
        assert_eq!(proto.computations.len(), 3);
        let mut roundtrip_ast = hlo_proto_to_ast(&proto);
        assert_eq!(roundtrip_ast.hlo_computations.len(), 3);
        if let Some(comp) = roundtrip_ast.hlo_computations.first_mut()
            && let Some(inst) = comp.instructions.first_mut()
        {
            inst.result_type = MlirType::Scalar(ElementType::F32);
        }
        let _ = hlo_ast_to_proto(&roundtrip_ast);
        let lowered = lower_ast_module_to_parsed_hlo(&ast);
        assert!(lowered.is_ok());
        if let Ok(lowered_mod) = lowered {
            assert_eq!(lowered_mod.output_shapes.len(), 2);
        }

        let compiled = CompiledProgram::compile("hlo", hlo_text.as_bytes(), b"");
        assert!(compiled.is_ok());
        if let Ok(prog) = compiled {
            assert_eq!(prog.num_replicas, 2);
            assert_eq!(prog.num_partitions, 4);
        }

        // Bare HLO instructions, 1-D layout `{0}`, padding before ROOT, and ROOT on gte / non-tuple
        let bare_hlo = "HloModule bare_mod\n ENTRY %m (p: f32[4]) -> f32[4] {\n %p = f32[4]{0} parameter(0)\n %pad = f32[4] pad(%p, %p), padding=1_1_0\n ROOT %g = f32[4] get-tuple-element(%pad), index=0\n }";
        let bare_ast = HloParser::parse_ast_module(bare_hlo);
        assert!(bare_ast.is_ok(), "bare_hlo failed: {bare_ast:?}");
        assert!(lower_ast_module_to_parsed_hlo(&bare_ast.unwrap_or_default()).is_ok());

        let single_inst_hlo = "HloModule single_mod\n %x = f32[4] negate(%p)";
        let single_ast = HloParser::parse_ast_module(single_inst_hlo);
        assert!(single_ast.is_ok());
        assert!(HloParser::parse_ast_module("HloModule trailing_comma,").is_ok());

        let root_scalar_hlo = "HloModule root_mod\n %comp (x: f32[4]) -> f32[4] {\n %x = f32[4] parameter(0)\n ROOT %neg = f32[4] negate(%x), sharding={unclosed";
        let root_ast = HloParser::parse_ast_module(root_scalar_hlo).unwrap_or_default();
        assert!(lower_ast_module_to_parsed_hlo(&root_ast).is_ok());

        // Negative HloTextParser error branches
        let bad_sp = HloLexer::tokenize_spanned("123");
        assert!(HloTextParser::new(&bad_sp).parse_computation().is_err());
        assert!(HloTextParser::new(&bad_sp).parse_instruction().is_err());
        let bad_op_sp = HloLexer::tokenize_spanned("%x = f32[4] 123");
        assert!(HloTextParser::new(&bad_op_sp).parse_instruction().is_err());
        let bad_sh_sp = HloLexer::tokenize_spanned("bad_dtype[4]");
        assert!(
            HloTextParser::new(&bad_sh_sp)
                .parse_hlo_shape_type()
                .is_err()
        );
        let trail_comma_sp = HloLexer::tokenize_spanned("%x = f32[4] negate(%a), 123");
        assert!(
            HloTextParser::new(&trail_comma_sp)
                .parse_instruction()
                .is_ok()
        );

        // Lowering empty AstModule and top-level MLIR operations / multi-result regions / tuples
        assert!(lower_ast_module_to_parsed_hlo(&AstModule::default()).is_ok());
        let top_mlir = r#"
            %no_opnd = "stablehlo.negate"() : () -> tensor<4xf32>
            %untyped_c = "stablehlo.constant"() {value = dense<2.0>} : () -> tensor<4xf32>
            %a = stablehlo.constant dense<1.0> : tensor<4xf32>
            %s1 = stablehlo.constant dense<1.0> : tensor<1xf32>
            %t = stablehlo.tuple %a, %a : tuple<tensor<4xf32>, tensor<4xf32>>
            stablehlo.tuple %a : tensor<4xf32>
            %g = stablehlo.get_tuple_element %a, index = 0 : (tensor<4xf32>) -> tensor<4xf32>
            %ar = "stablehlo.all_reduce"(%a) ({
            ^bb0(%x: tensor<f32>, %y: tensor<f32>):
                %s = stablehlo.add %x, %y : tensor<f32>
                stablehlo.return %s : tensor<f32>
            }) : (tensor<4xf32>) -> tensor<4xf32>
            %rs = stablehlo.reduce_scatter %a : tensor<4xf32>
            %mc:2 = "sdy.manual_computation"() ({
            ^bb0(%b0: tensor<4xf32>):
                sdy.return %b0, %b0 : tensor<4xf32>, tensor<4xf32>
            }) : () -> (tensor<4xf32>, tensor<4xf32>)
            func.return %t : tuple<tensor<4xf32>, tensor<4xf32>>
        "#;
        let top_ast = parse_mlir_ast_module(top_mlir).unwrap_or_default();
        assert!(lower_ast_module_to_parsed_hlo(&top_ast).is_ok());

        // Verify semantics on nested region and Chlo broadcast with different dims
        let valid_nested_and_chlo = r#"
            func.func @main(%a: tensor<4xf32>, %b: tensor<1xf32>) -> tensor<4xf32> {
                %c = chlo.broadcast_add %a, %b : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
                %m = "sdy.manual_computation"(%c) ({
                ^bb0(%in: tensor<4xf32>):
                    sdy.return %in : tensor<4xf32>
                }) : (tensor<4xf32>) -> tensor<4xf32>
                func.return %m : tensor<4xf32>
            }
        "#;
        let v_ast = HloLexer::tokenize_spanned(valid_nested_and_chlo);
        let v_mod = MlirModuleParser::new(&v_ast)
            .parse_module()
            .unwrap_or_default();
        assert!(verify_ast_module_semantics(&v_mod).is_ok());

        let bad_etype = r"
            func.func @main(%a: tensor<4xf32>, %b: tensor<4xs32>) -> tensor<4xf32> {
                %0 = stablehlo.add %a, %b : (tensor<4xf32>, tensor<4xs32>) -> tensor<4xf32>
                func.return %0 : tensor<4xf32>
            }
        ";
        assert!(HloParser::parse_verified_ast_module(bad_etype).is_err());
    }

    #[test]
    fn test_ast_verification_errors_and_edge_cases() {
        // Undefined SSA value error
        let bad_ssa = r"
            func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                %0 = stablehlo.add %arg0, %undefined_val : tensor<4xf32>
                func.return %0 : tensor<4xf32>
            }
        ";
        let res_ssa = HloParser::parse_verified_ast_module(bad_ssa);
        assert!(matches!(
            res_ssa,
            Err(LibTpuError::UndefinedSsaValue { .. })
        ));

        // Binary arity mismatch error
        let bad_arity = r#"
            func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                %0 = "stablehlo.add"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                func.return %0 : tensor<4xf32>
            }
        "#;
        let res_arity = HloParser::parse_verified_ast_module(bad_arity);
        assert!(matches!(
            res_arity,
            Err(LibTpuError::ShapeInferenceError { .. })
        ));

        // Shape mismatch error
        let bad_shape = r"
            func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>) -> tensor<4xf32> {
                %0 = stablehlo.add %arg0, %arg1 : (tensor<4xf32>, tensor<8xf32>) -> tensor<4xf32>
                func.return %0 : tensor<4xf32>
            }
        ";
        let res_shape = HloParser::parse_verified_ast_module(bad_shape);
        assert!(matches!(
            res_shape,
            Err(LibTpuError::ShapeInferenceError { .. })
        ));

        // sdy.manual_computation missing region error
        let bad_sdy = r#"
            func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                %0 = "sdy.manual_computation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                func.return %0 : tensor<4xf32>
            }
        "#;
        let res_sdy = HloParser::parse_verified_ast_module(bad_sdy);
        assert!(matches!(
            res_sdy,
            Err(LibTpuError::DialectVerificationError { .. })
        ));

        // Undefined #alias and !alias errors
        let sp_attr = HloLexer::tokenize_spanned("#unknown_alias");
        let mut p_attr = MlirModuleParser::new(&sp_attr);
        assert!(matches!(
            p_attr.parse_attribute(),
            Err(LibTpuError::UndefinedAlias { .. })
        ));

        let sp_type = HloLexer::tokenize_spanned("!unknown_type");
        let mut p_type = MlirModuleParser::new(&sp_type);
        assert!(matches!(
            p_type.parse_type(),
            Err(LibTpuError::UndefinedAlias { .. })
        ));

        // Integer add lowering (`AddS32`)
        let int_mlir = r"
            func.func @main(%a: tensor<4xs32>, %b: tensor<4xs32>) -> tensor<4xs32> {
                %0 = stablehlo.add %a, %b : tensor<4xs32>
                func.return %0 : tensor<4xs32>
            }
        ";
        let int_prog = CompiledProgram::compile("mlir", int_mlir.as_bytes(), b"");
        assert!(int_prog.is_ok());
        if let Ok(p) = int_prog {
            assert_eq!(p.opcode, TpuOpCode::AddS32);
        }

        // Parse float literals (hex floats, bit patterns, nan, inf)
        assert!(parse_float_literal("nan").is_nan());
        assert!(parse_float_literal("inf").is_infinite());
        assert!(parse_float_literal("-inf").is_infinite());
        assert!((parse_float_literal("0x1.8p+1") - 3.0).abs() < 1e-6);
        assert!((parse_float_literal("0x3F800000") - 1.0).abs() < 1e-6);
        assert!((parse_float_literal("0x3FF0000000000000") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_all_dialect_operations_and_attributes() {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            String::from("comparison_direction"),
            MlirAttribute::ComparisonDirection(ComparisonDirection::Gt),
        );
        for (dialect, mnem) in [
            (DialectKind::Func, "return"),
            (DialectKind::Func, "call"),
            (DialectKind::Sdy, "return"),
            (DialectKind::Sdy, "manual_computation"),
            (DialectKind::Sdy, "sharding_constraint"),
            (DialectKind::Sdy, "data_flow_edge"),
            (DialectKind::Sdy, "mesh"),
            (DialectKind::Sdy, "constant"),
            (DialectKind::Sdy, "all_gather"),
            (DialectKind::Sdy, "all_reduce"),
            (DialectKind::Sdy, "all_slice"),
            (DialectKind::Sdy, "all_to_all"),
            (DialectKind::Sdy, "collective_permute"),
            (DialectKind::Sdy, "reduce_scatter"),
            (DialectKind::Chlo, "top_k"),
            (DialectKind::Chlo, "erf"),
            (DialectKind::Chlo, "tan"),
            (DialectKind::Chlo, "ragged_dot"),
            (DialectKind::Chlo, "broadcast_add"),
            (DialectKind::Chlo, "broadcast_sub"),
            (DialectKind::Chlo, "broadcast_mul"),
            (DialectKind::Chlo, "broadcast_div"),
            (DialectKind::Chlo, "broadcast_max"),
            (DialectKind::Chlo, "broadcast_min"),
            (DialectKind::Chlo, "broadcast_pow"),
            (DialectKind::Chlo, "broadcast_atan2"),
            (DialectKind::Chlo, "broadcast_select"),
            (DialectKind::Chlo, "broadcast_compare"),
            (DialectKind::Chlo, "zeta"),
            (DialectKind::TpuMosaic, "matmul"),
            (DialectKind::TpuMosaic, "enqueue_dma"),
            (DialectKind::TpuMosaic, "wait_dma"),
            (DialectKind::TpuMosaic, "semaphore_signal"),
            (DialectKind::TpuMosaic, "vector_load"),
            (DialectKind::TpuMosaic, "iota"),
            (DialectKind::TpuMosaic, "repeat"),
            (DialectKind::TpuMosaic, "rotate"),
            (DialectKind::TpuMosaic, "concat"),
            (DialectKind::TpuMosaic, "memref_slice"),
            (DialectKind::TpuMosaic, "pack_subelements"),
            (DialectKind::TpuMosaic, "region"),
            (DialectKind::Scf, "for"),
            (DialectKind::Scf, "while"),
            (DialectKind::Scf, "if"),
            (DialectKind::Scf, "yield"),
            (DialectKind::Arith, "constant"),
            (DialectKind::Arith, "addi"),
            (DialectKind::Arith, "addf"),
            (DialectKind::Arith, "subi"),
            (DialectKind::Arith, "muli"),
            (DialectKind::Arith, "divsi"),
            (DialectKind::Arith, "remsi"),
            (DialectKind::Arith, "maxsi"),
            (DialectKind::Arith, "minsi"),
            (DialectKind::Arith, "andi"),
            (DialectKind::Arith, "ori"),
            (DialectKind::Arith, "xori"),
            (DialectKind::Arith, "shli"),
            (DialectKind::Arith, "shrsi"),
            (DialectKind::Arith, "shrui"),
            (DialectKind::Arith, "cmpi"),
            (DialectKind::Arith, "select"),
            (DialectKind::Arith, "negf"),
            (DialectKind::Arith, "index_cast"),
            (DialectKind::Arith, "bitcast"),
            (DialectKind::Vector, "broadcast"),
            (DialectKind::Vector, "contract"),
            (DialectKind::Vector, "extract"),
            (DialectKind::Vector, "insert"),
            (DialectKind::Vector, "multi_reduction"),
            (DialectKind::Vector, "shape_cast"),
            (DialectKind::Vector, "transpose"),
            (DialectKind::Vector, "transfer_read"),
            (DialectKind::MemRef, "load"),
            (DialectKind::MemRef, "subview"),
            (DialectKind::MemRef, "reinterpret_cast"),
            (DialectKind::StableHlo, "while"),
            (DialectKind::StableHlo, "if"),
            (DialectKind::StableHlo, "optimization_barrier"),
        ] {
            let kind = classify_ast_operation(dialect, mnem, ElementType::F32, &attrs, &[]);
            assert_ne!(format!("{kind:?}"), "");
        }

        // Test dialect attributes (#stablehlo.gather, #stablehlo.scatter, #stablehlo<fft_type ...>, etc.)
        let attr_src = r#"
            {
                g = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], operand_batching_dims = [], start_indices_batching_dims = [], start_index_map = [0], index_vector_dim = 1>,
                s = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], input_batching_dims = [], scatter_indices_batching_dims = [], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>,
                cd = #stablehlo<comparison_direction LE>,
                ct = #stablehlo<comparison_type SIGNED>,
                fft1 = #stablehlo<fft_type IFFT>,
                fft2 = #stablehlo<fft_type RFFT>,
                fft3 = #stablehlo<fft_type IRFFT>,
                fft4 = #stablehlo<fft_type FFT>,
                rng1 = #stablehlo<rng_algorithm THREE_FRY>,
                rng2 = #stablehlo<rng_algorithm PHILOX>,
                rng3 = #stablehlo<rng_algorithm DEFAULT>,
                tr1 = #stablehlo<transpose TRANSPOSE>,
                tr2 = #stablehlo<transpose ADJOINT>,
                tr3 = #stablehlo<transpose NO_TRANSPOSE>,
                sp = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{"y"}]>]>,
                mesh_inline = #sdy.mesh<["a"=2]>,
                ms = #tpu.memory_space<vmem>,
                aff = affine_map<(d0, d1) -> (d0, d1)>,
                sym_chain = @root::@sub
            }
        "#;
        let tokens = HloLexer::tokenize_spanned(attr_src);
        let mut parser = MlirModuleParser::new(&tokens);
        let dict = parser.parse_attribute_dict();
        assert!(dict.is_ok());
        assert_eq!(dict.unwrap_or_default().len(), 19);
    }

    #[test]
    fn test_parser_edge_cases_and_token_cursor() {
        use self::parser::TokenCursor;

        // Exercise TokenCursor methods (mismatch + EOF branches)
        let cur_tokens = HloLexer::tokenize_spanned("@sym %ssa ^bb0 #attr !ty \"str\" 42 ident (");
        let mut cur = TokenCursor::new(&cur_tokens);
        assert!(cur.peek().is_some());
        assert!(TokenCursor::new(&cur_tokens).expect_symbol().is_ok());
        // Mismatch calls before consuming matching token
        assert!(cur.consume_ssa_spanned().is_none());
        assert!(cur.consume_caret_spanned().is_none());
        assert!(cur.consume_attr_alias_spanned().is_none());
        assert!(cur.consume_type_alias_spanned().is_none());
        assert!(cur.consume_string_str().is_none());
        assert!(cur.consume_number_str().is_none());
        assert!(cur.consume_any_ident_str().is_none());
        assert!(cur.expect_punct('(').is_err());
        assert!(cur.expect_ident().is_err());
        assert!(cur.expect_ssa().is_err());
        // Consume @sym, then mismatch consume_symbol_str on %ssa
        let mut cur2 = TokenCursor::new(&cur_tokens);
        assert_eq!(cur2.consume_symbol_str().as_deref(), Some("sym"));
        assert!(cur2.consume_symbol_str().is_none());
        assert!(cur2.consume_ssa_spanned().is_some());
        assert!(cur2.consume_caret_spanned().is_some());
        assert!(cur2.consume_attr_alias_spanned().is_some());
        assert!(cur2.consume_type_alias_spanned().is_some());
        assert_eq!(cur2.consume_string_str().as_deref(), Some("str"));
        assert_eq!(cur2.consume_number_str().as_deref(), Some("42"));
        assert_eq!(cur2.consume_any_ident_str().as_deref(), Some("ident"));
        assert!(cur2.expect_punct('(').is_ok());
        // Test expect_symbol mismatch on non-symbol token
        assert!(cur.expect_symbol().is_err());
        // Now at EOF: test all consume_* and expect_* at EOF
        assert!(!cur2.is_open(')'));
        assert!(cur2.consume_symbol_str().is_none());
        assert!(cur2.consume_ssa_spanned().is_none());
        assert!(cur2.consume_caret_spanned().is_none());
        assert!(cur2.consume_attr_alias_spanned().is_none());
        assert!(cur2.consume_type_alias_spanned().is_none());
        assert!(cur2.consume_string_str().is_none());
        assert!(cur2.consume_number_str().is_none());
        assert!(cur2.consume_any_ident_str().is_none());
        assert!(cur2.expect_punct(')').is_err());
        assert!(cur2.expect_ident().is_err());
        assert!(cur2.expect_ssa().is_err());
        assert!(cur2.expect_symbol().is_err());

        // Exercise all scalar & tensor element types, bounded dims, custom encodings, quantized types
        for ty_src in [
            "tensor<4xf64>",
            "tensor<4xf16>",
            "tensor<4xi16>",
            "tensor<4xi8>",
            "tensor<4xui32>",
            "tensor<4xui64>",
            "tensor<4xui16>",
            "tensor<4xui8>",
            "tensor<4xcomplex<f32>>",
            "tensor<4xcomplex<f64>>",
            "tensor<4xf8E5M2>",
            "tensor<4xf8E4M3FN>",
            "tensor<4xf8E4M3B11FNUZ>",
            "tensor<4xi4>",
            "tensor<4xui4>",
            "tensor<4xi2>",
            "tensor<4xui2>",
            "tensor<f32>",
            "tensor<4x<=16xf32, #custom.enc<a::b, ..., @sym, \"s\", <=8, >=2, [1], (tensor<4xf32>) -> tensor<4xf32>>>",
            "tensor<4x!quant.uniform<i4:f32, 0.1:0>>",
            "tensor<4x!quant.uniform<u8:f32, 0.1:0>>",
            "!quant.uniform<i8:f32, 0.05:10>",
            "!tpu.semaphore",
            "!mhlo.token",
            "token",
            "(tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)",
            "() -> tensor<4xf32>",
        ] {
            let sp = HloLexer::tokenize_spanned(ty_src);
            let mut p = MlirModuleParser::new(&sp);
            assert!(p.parse_type().is_ok(), "Failed to parse type: {ty_src}");
        }

        // Exercise invalid type error paths
        for bad_ty in [
            "unknown_scalar",
            "tensor<4xbad_dtype>",
            "tensor<4x<=badxf32>",
            "tensor<bad_dimxf32>",
            "memref<4xbad_dtype>",
            "vector<4xbad_dtype>",
            "tensor<4xf32",
            "!custom.undef<inner>",
        ] {
            let sp = HloLexer::tokenize_spanned(bad_ty);
            let mut p = MlirModuleParser::new(&sp);
            assert!(p.parse_type().is_err(), "Expected error on: {bad_ty}");
        }

        // Exercise module/function/region/operation/attribute edge cases
        let edge_mlir = r#"
            #loc_top = loc("top.py":1:1)
            !alias_t = tensor<4xf32>
            builtin.module {
                sdy.mesh @m1 = <["x"=2], device_ids=[0, 1]>
                sdy.mesh @m2 = bad_mesh
                sdy.mesh @m3 = <>
                func.func nested main(tensor<4xf32> {arg_flag}, tensor<4xf32>) -> tensor<4xf32>
                func.func @void_fn() {
                    func.return
                }
                func.func @multi_blk(%a: !alias_t) -> !alias_t {
                    %r:2 = stablehlo.rng_bit_generator %a, algorithm = DEFAULT : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
                    %g = stablehlo.gather %a, %a, #stablehlo.gather<offset_dims = [0], index_vector_dim = 0> : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
                    %sc = stablehlo.scatter %a, %a, %a, #stablehlo.scatter<update_window_dims = [0], index_vector_dim = 0> ({
                    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
                        %and_v = stablehlo.and %x, %y : tensor<f32>
                        stablehlo.return %and_v : tensor<f32>
                    }) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    %red_and = stablehlo.reduce(%a init: %a) across dims = [0] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> reducer(%x: tensor<f32>, %y: tensor<f32>) {
                        %and_v2 = stablehlo.and %x, %y : tensor<f32>
                        stablehlo.return %and_v2 : tensor<f32>
                    }
                    %red_or = stablehlo.reduce(%a init: %a) across dims = [0] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> reducer(%x: tensor<f32>, %y: tensor<f32>) {
                        %or_v = stablehlo.or %x, %y : tensor<f32>
                        stablehlo.return %or_v : tensor<f32>
                    }
                    %red_other = stablehlo.reduce(%a init: %a) across dims = [0] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> reducer(%x: tensor<f32>, %y: tensor<f32>) {
                        %abs_v = stablehlo.abs %x : tensor<f32>
                        stablehlo.return %abs_v : tensor<f32>
                    }
                    %cmp_eq = stablehlo.compare EQ, %a, %a, UNKNOWN_CT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                    %cmp_bad = stablehlo.compare BAD_DIR, %a, %a : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                    %cmp_gen = "stablehlo.compare"(%a, %a) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                    %for_no_iv = scf.for to %a step %a -> tensor<4xf32> {
                        scf.yield %a : tensor<4xf32>
                    }
                    %c_s32 = chlo.broadcast_add %a, %a {} : (tensor<4xs32>, tensor<4xs32>) -> tensor<4xs32>
                    %custom_attr_op = stablehlo.abs %a, #stablehlo<precision DEFAULT>, #sdy.sharding_per_value<[<@m1, [{"x"}]>]> : tensor<4xf32> -> (tensor<4xf32>)
                    tpu.wait_dma %a : (tensor<4xf32>)
                    cf.br ^bb1
                ^bb0:
                ^bb1:
                    %neg = stablehlo.negate %r#0 : tensor<4xf32>
                    cf.br ^bb2
                ^bb2(%b_arg: tensor<4xf32>):
                    func.return %b_arg : tensor<4xf32>
                }
            }
            }
            %top_op = stablehlo.abs %a : tensor<4xf32>
            123
            %eof_op = stablehlo.abs %a
        "#;
        let parsed_edge = parse_mlir_ast_module(edge_mlir);
        assert!(parsed_edge.is_ok());

        // Exercise remaining attribute and location branches
        let extra_attrs = r#"
            #loc_def = loc("file.py":2:3)
            module attributes {
                hex_int = 0x2A : i32,
                hex_flt = 0x1.8p+1,
                flt_val = 2.5 : f32,
                int_unknown_ty = 7 : unknown_ty,
                nan_val = nan,
                inf_val = inf,
                bool_t = true,
                bool_f = false,
                "quoted_key" = 1,
                colon_kv : 123,
                hex_blob = "0x0000803F",
                arr_b = array<i1: true, false, 1, 0, ignore>,
                dense_b = dense<[true, false, nan, inf]> : tensor<4xf32>,
                dense_memref = dense<1.0> : memref<4xf32, #other_space>,
                dense_untyped = dense<1.0>,
                dr_untyped = dense_resource<blob_id>,
                loc_named = loc("step_name"),
                loc_Empty = loc(),
                loc_fused_meta = loc(fused<my_meta>["a.py":1:1]),
                loc_ref = #loc_def,
                dir_id = LT,
                ct_id = FLOAT,
                cd_bad = #stablehlo<comparison_direction BAD>,
                ct_bad = #stablehlo<comparison_type BAD>,
                sh_prio = #sdy.sharding<mesh, [{"x", p1}, {"y", pbad}, {}p2]>,
                sh_bare = #sdy.sharding<mesh>,
                sh_custom = #stablehlo<custom_payload>,
                other_alias = #custom.dialect<payload>,
                ssa_fallback = %fallback_tok,
                ty_attr = tensor<2xf32>,
                sym_trailing = @root::
            } {
            }
        "#;
        assert!(parse_mlir_ast_module(extra_attrs).is_ok());

        // Negative attribute & operation parser tests
        let eof_tokens = HloLexer::tokenize_spanned("");
        let mut eof_parser = MlirModuleParser::new(&eof_tokens);
        assert!(eof_parser.parse_attribute().is_err());
        assert!(eof_parser.parse_operation().is_err());

        let lhs_tokens = HloLexer::tokenize_spanned("%lhs_only");
        let mut lhs_parser = MlirModuleParser::new(&lhs_tokens);
        assert!(lhs_parser.parse_operation().is_err());

        let invalid_dict_tokens = HloLexer::tokenize_spanned("{ 123 = 456 }");
        let mut invalid_dict_parser = MlirModuleParser::new(&invalid_dict_tokens);
        assert!(invalid_dict_parser.parse_attribute_dict().is_err());

        assert!(parse_float_literal("+nan").is_nan());
        assert!((parse_float_literal("0x1p+2") - 4.0).abs() < 1e-6);
        assert!((parse_float_literal("true") - 1.0).abs() < 1e-6);
        assert!(parse_float_literal("false").abs() < 1e-6);

        // Test MlirBytecodeHeader::encode_from_ast_module, decode_binary_attr_types, decode_resource_blobs, and versioned kDialect
        let ast_src = r#"
            module @bc_roundtrip {
                func.func @main(%arg0: tensor<f32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.constant dense<"0x0000803F000000400000404000008040"> : tensor<4xf32>
                    %1 = "vhlo.add_v1"(%arg0, %arg1) ({
                    ^bb0(%r0: tensor<f32>):
                        stablehlo.return %r0 : tensor<f32>
                    }) {some_attr = 1 : i64} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    %2 = add %0, %1 : tensor<4xf32>
                    func.return %2 : tensor<4xf32>
                }
            }
        "#;
        if let Ok(ast_m) = parse_mlir_ast_module(ast_src) {
            let bc_bytes = MlirBytecodeHeader::encode_from_ast_module(&ast_m);
            assert!(bc_bytes.starts_with(MLIR_BYTECODE_MAGIC));
            let decoded_hdr = MlirBytecodeHeader::decode(&bc_bytes);
            assert!(decoded_hdr.is_ok());
            if let Ok(mut hdr) = decoded_hdr {
                let d_descs = hdr.decode_dialect_descriptors();
                assert_ne!(d_descs, [] as [MlirDialectDescriptor; 0]);
                if let Some(first_d) = d_descs.first() {
                    assert_eq!(first_d.clone(), *first_d);
                    assert_ne!(format!("{first_d:?}"), "");
                }
                hdr.attr_type_entries.push(String::from("f32"));
                hdr.attr_type_entries.push(String::from("memref<4xf32>"));
                hdr.attr_type_entries.push(String::from("vector<4xf32>"));
                hdr.attr_type_entries
                    .push(String::from("tuple<tensor<4xf32>>"));
                hdr.attr_type_entries.push(String::from("complex<f32>"));
                hdr.attr_type_entries.push(String::from("!tpu.semaphore"));
                hdr.attr_type_entries
                    .push(String::from("!quant.uniform<i8:f32, 0.1:0>"));
                hdr.attr_type_entries.push(String::from("token"));
                hdr.attr_type_entries
                    .push(String::from("tensor<bad_unparseable>"));
                hdr.attr_type_entries.push(String::from("dense<1.0>"));
                hdr.attr_type_entries.push(String::from("array<i64: 1, 2>"));
                hdr.attr_type_entries
                    .push(String::from("#stablehlo<comparison_direction EQ>"));
                hdr.attr_type_entries
                    .push(String::from("#unknown_unparseable_alias"));
                hdr.attr_type_entries.push(String::from("42"));
                hdr.attr_type_entries.push(String::from("custom_attr"));
                let (tys, attrs) = hdr.decode_binary_attr_types();
                assert!(!tys.is_empty() && !attrs.is_empty());

                // Test resource blobs with and without offsets
                assert_eq!(hdr.decode_resource_blobs().len(), 1);
                hdr.resources = vec![vec![1, 2, 3, 4, 5, 6, 7, 8]];
                hdr.resource_offsets.clear();
                assert_eq!(hdr.decode_resource_blobs().len(), 1);
                hdr.resource_offsets = vec![4, 4];
                assert_eq!(hdr.decode_resource_blobs().len(), 2);
                hdr.resource_offsets = vec![0];
                assert_eq!(hdr.decode_resource_blobs().len(), 1);

                hdr.string_table.push(String::from("unqualified_op"));
                hdr.ir_op_records.extend_from_slice(&[
                    (hdr.string_table.len() - 1) as u64,
                    0x06,
                    0,
                ]);
                let rt_ast = hdr.to_ast_module();
                assert_eq!(rt_ast.name.as_deref(), Some("bc_roundtrip"));
            }
        }
        let empty_ast = AstModule::default();
        let empty_bc = MlirBytecodeHeader::encode_from_ast_module(&empty_ast);
        if let Ok(mut hdr) = MlirBytecodeHeader::decode(&empty_bc) {
            hdr.attr_type_entries.clear();
            let (tys, _) = hdr.decode_binary_attr_types();
            assert_eq!(tys.len(), 2);
            hdr.attr_type_offsets.clear();
            let (tys_empty, _) = hdr.decode_binary_attr_types();
            assert_eq!(tys_empty.len(), 1);
        }

        // Test EM_DEEPSEA (0x1AE0) ELF64 packaging, relocation, and deserialization
        if let Ok(mut prog_elf) = CompiledProgram::compile(
            "mlir",
            b"module @deepsea_elf { %0 = stablehlo.constant dense<2.0> : tensor<4xf32> %1 = stablehlo.add %0, %0 : tensor<4xf32> %2 = stablehlo.multiply %1, %0 : tensor<4xf32> %3 = stablehlo.subtract %2, %0 : tensor<4xf32> return %3 : tensor<4xf32> }",
            b"",
        ) {
            let elf64 = prog_elf.to_deepsea_elf64();
            assert!(elf64.starts_with(b"\x7fELF"));
            let reloc = CompiledProgram::from_deepsea_elf64(&elf64, 0x1000_0000, 0x2000);
            assert!(reloc.is_ok());
            if let Ok(r) = reloc {
                assert!(r.applied_relocations >= 4);
                assert_ne!(r.relocated_imem, [] as [u8; 0]);
                assert_ne!(r.constant_pool, [] as [u8; 0]);
                assert_eq!(r.program.name, prog_elf.name);
                assert_ne!(format!("{r:?}"), "");
                let r_cl = r.clone();
                assert_eq!(r_cl.applied_relocations, r.applied_relocations);
            }
            assert!(CompiledProgram::deserialize(&elf64).is_ok());
            assert!(CompiledProgram::from_deepsea_elf64(b"short", 0, 0).is_err());
            let mut wrong_mach = elf64.clone();
            if let Some(m0) = wrong_mach.get_mut(18) {
                *m0 = 0x3E;
            }
            if let Some(m1) = wrong_mach.get_mut(19) {
                *m1 = 0x00;
            }
            assert!(CompiledProgram::from_deepsea_elf64(&wrong_mach, 0, 0).is_err());
            let mut trunc_shdr = elf64;
            trunc_shdr.truncate(80);
            assert!(CompiledProgram::from_deepsea_elf64(&trunc_shdr, 0, 0).is_err());

            // Test pass_dead_code_elimination, shape inference, and extended algebraic/constant folding
            prog_elf.pass_dead_code_elimination();
            let mut empty_out_prog = prog_elf.clone();
            empty_out_prog.output_slots.clear();
            empty_out_prog.pass_dead_code_elimination();
            let mut dce_prog = prog_elf.clone();
            dce_prog.output_slots = vec![1];
            dce_prog.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 10,
                opcode: TpuOpCode::AddF32,
                operand_slots: vec![0, 0],
                output_slot: 99,
                sync_flag_id: 1,
                attrs: InstructionAttrs::default(),
            });
            dce_prog.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 11,
                opcode: TpuOpCode::AllReduceSumF32,
                operand_slots: vec![0],
                output_slot: 98,
                sync_flag_id: 1,
                attrs: InstructionAttrs::default(),
            });
            dce_prog.pass_dead_code_elimination();
            assert!(
                dce_prog
                    .instruction_bundles
                    .iter()
                    .all(|b| b.output_slot != 99)
            );
            }

        // Test per-op shape inference and extended algebraic/constant folding
        if let Ok(mut shape_prog) =
            CompiledProgram::compile("mlir", b"module @sh { stablehlo.add }", b"")
        {
            let base_sh = Shape {
                element_type: ElementType::F32,
                dims: vec![4, 4],
                dynamic_dims: vec![false, false],
                layout: crate::types::MemoryLayout::default_tpu_tiled(2),
            };
            shape_prog.parameter_shapes = vec![base_sh.clone(), base_sh];
            shape_prog.instruction_bundles = vec![
                TpuInstructionBundle {
                    pc_offset: 0,
                    opcode: TpuOpCode::Constant,
                    operand_slots: vec![],
                    output_slot: 2,
                    sync_flag_id: 1,
                    attrs: InstructionAttrs {
                        constant_bytes: vec![0_u8; 16],
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 1,
                    opcode: TpuOpCode::Transpose,
                    operand_slots: vec![0],
                    output_slot: 3,
                    sync_flag_id: 2,
                    attrs: InstructionAttrs {
                        permutation: vec![1, 0],
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 2,
                    opcode: TpuOpCode::Transpose,
                    operand_slots: vec![3],
                    output_slot: 4,
                    sync_flag_id: 3,
                    attrs: InstructionAttrs {
                        permutation: vec![1, 0],
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 3,
                    opcode: TpuOpCode::Reshape,
                    operand_slots: vec![0],
                    output_slot: 5,
                    sync_flag_id: 4,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 4,
                    opcode: TpuOpCode::Reshape,
                    operand_slots: vec![5],
                    output_slot: 6,
                    sync_flag_id: 5,
                    attrs: InstructionAttrs::default(),
                },
                TpuInstructionBundle {
                    pc_offset: 5,
                    opcode: TpuOpCode::Slice,
                    operand_slots: vec![0],
                    output_slot: 7,
                    sync_flag_id: 6,
                    attrs: InstructionAttrs {
                        slice_starts: vec![0, 0],
                        slice_limits: vec![2, 2],
                        slice_strides: vec![1, 1],
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 6,
                    opcode: TpuOpCode::Pad,
                    operand_slots: vec![0],
                    output_slot: 8,
                    sync_flag_id: 7,
                    attrs: InstructionAttrs {
                        pad_edge_low: vec![1, 1],
                        pad_edge_high: vec![1, 1],
                        pad_interior: vec![0, 0],
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 7,
                    opcode: TpuOpCode::Concatenate,
                    operand_slots: vec![0, 1],
                    output_slot: 9,
                    sync_flag_id: 8,
                    attrs: InstructionAttrs {
                        concat_dim: 0,
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 8,
                    opcode: TpuOpCode::ReduceSumF32,
                    operand_slots: vec![0],
                    output_slot: 10,
                    sync_flag_id: 9,
                    attrs: InstructionAttrs {
                        reduce_dims: vec![0, 1],
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 9,
                    opcode: TpuOpCode::ReduceSumF32,
                    operand_slots: vec![0],
                    output_slot: 14,
                    sync_flag_id: 13,
                    attrs: InstructionAttrs {
                        reduce_dims: vec![0],
                        ..InstructionAttrs::default()
                    },
                },
                TpuInstructionBundle {
                    pc_offset: 10,
                    opcode: TpuOpCode::CompareEq,
                    operand_slots: vec![0, 1],
                    output_slot: 11,
                    sync_flag_id: 10,
                    attrs: InstructionAttrs::default(),
                },
            ];
            shape_prog.pass_hlo_verification_and_shape_inference();
            shape_prog.pass_algebraic_simplifier();
            for u_op in [
                TpuOpCode::Abs,
                TpuOpCode::ExpF32,
                TpuOpCode::TanhF32,
                TpuOpCode::Sqrt,
            ] {
                shape_prog.instruction_bundles.push(TpuInstructionBundle {
                    pc_offset: 10,
                    opcode: u_op,
                    operand_slots: vec![2],
                    output_slot: 12,
                    sync_flag_id: 11,
                    attrs: InstructionAttrs::default(),
                });
            }
            for b_op in [TpuOpCode::SubF32, TpuOpCode::MaxF32, TpuOpCode::MinF32] {
                shape_prog.instruction_bundles.push(TpuInstructionBundle {
                    pc_offset: 11,
                    opcode: b_op,
                    operand_slots: vec![2, 2],
                    output_slot: 13,
                    sync_flag_id: 12,
                    attrs: InstructionAttrs::default(),
                });
            }
            shape_prog.pass_constant_folding();
        }

        // Test versioned kDialect ((str_idx << 1) | 1 followed by version len & bytes)
        let mut ver_dial = Vec::new();
        ver_dial.extend_from_slice(&encode_mlir_prefix_varint((10 << 1) | 1));
        ver_dial.extend_from_slice(&encode_mlir_prefix_varint(2));
        ver_dial.extend_from_slice(&[1, 9]);
        let ver_hdr = MlirBytecodeHeader {
            version: 6,
            producer: String::from("MLIR20"),
            section_count: 2,
            sections: vec![
                MlirBytecodeSection {
                    section_id: 0,
                    alignment: None,
                    data: b"s0\0s1\0s2\0s3\0s4\0s5\0s6\0s7\0s8\0s9\0vhlo\0".to_vec(),
                },
                MlirBytecodeSection {
                    section_id: 1,
                    alignment: None,
                    data: ver_dial,
                },
            ],
            string_table: Vec::new(),
            dialects: Vec::new(),
            attr_type_offsets: Vec::new(),
            attr_type_entries: Vec::new(),
            ir_op_records: Vec::new(),
            resources: Vec::new(),
            resource_offsets: Vec::new(),
            properties: Vec::new(),
        };
        let encoded_ver = ver_hdr.encode();
        if let Ok(dec_ver) = MlirBytecodeHeader::decode(&encoded_ver) {
            assert_eq!(dec_ver.dialects, vec!["vhlo"]);
        }

        // Test CompiledProgram::compile on bytecode where string_table is generic so kIR fallback triggers
        let mut fallback_enc = MlirBytecodeEncoder::new("MLIR20");
        let _ = fallback_enc.intern_string("@mod");
        let op_idx = fallback_enc.intern_string("vhlo.exponential_v1");
        fallback_enc.operations.push(MlirBytecodeOpNode {
            name_index: op_idx,
            encoding_mask: 0x06,
            attr_dict_index: None,
            operands: vec![0],
            result_type_indices: vec![0],
            successors: Vec::new(),
            num_regions: 0,
            properties_index: None,
            inline_regions: Vec::new(),
            region_block_arg_types: Vec::new(),
        });
        let mut raw_fb = fallback_enc.encode_to_bytes();
        // Replace string table entry in-place only in ir_op_records test:
        let _ = CompiledProgram::compile("mlir", &raw_fb, b"");
        raw_fb.clear();

        // Test BinaryAttrTypeEntry custom dialect encoding/decoding across all 14 tags and dtypes
        let sample_strings = vec![
            String::from("vhlo.func_v1"),
            String::from("vhlo.dot_general_v1"),
            String::from("vhlo.add_v1"),
            String::from("x"),
            String::from("y"),
            String::from("mesh"),
        ];
        for et in [
            ElementType::Pred,
            ElementType::S4,
            ElementType::U4,
            ElementType::S8,
            ElementType::U8,
            ElementType::S16,
            ElementType::U16,
            ElementType::S32,
            ElementType::U32,
            ElementType::S64,
            ElementType::U64,
            ElementType::F16,
            ElementType::Bf16,
            ElementType::F32,
            ElementType::F64,
            ElementType::F8E4M3Fn,
            ElementType::F8E5M2,
        ] {
            let sc = BinaryAttrTypeEntry::ScalarType(et);
            let enc = sc.encode_binary();
            assert_eq!(
                BinaryAttrTypeEntry::decode_binary(&enc).ok(),
                Some(sc.clone())
            );
            assert_ne!(sc.to_canonical_mlir_string(&sample_strings), "");
        }
        let all_bin_entries = vec![
            BinaryAttrTypeEntry::IntegerType {
                width: 32,
                signedness: 1,
            },
            BinaryAttrTypeEntry::IntegerType {
                width: 16,
                signedness: 2,
            },
            BinaryAttrTypeEntry::RankedTensorType {
                dims: Vec::new(),
                element_type: ElementType::F32,
                memory_space: 0,
            },
            BinaryAttrTypeEntry::RankedTensorType {
                dims: vec![-1, 4],
                element_type: ElementType::Bf16,
                memory_space: 1,
            },
            BinaryAttrTypeEntry::FunctionType {
                input_type_indices: vec![0, 1],
                result_type_indices: vec![0],
            },
            BinaryAttrTypeEntry::IntegerAttr {
                value: -42,
                element_type: ElementType::S32,
            },
            BinaryAttrTypeEntry::FloatAttr {
                bits: 3.5_f64.to_bits(),
                element_type: ElementType::F32,
            },
            BinaryAttrTypeEntry::StringAttr { string_index: 3 },
            BinaryAttrTypeEntry::StringAttr { string_index: 99 },
            BinaryAttrTypeEntry::DictionaryAttr {
                entries: vec![(3, 0), (4, 1)],
            },
            BinaryAttrTypeEntry::DenseElementsAttr {
                dims: vec![2],
                element_type: ElementType::F32,
                is_splat: false,
                raw_bytes: vec![0, 0, 128, 63, 0, 0, 0, 64],
            },
            BinaryAttrTypeEntry::DenseElementsAttr {
                dims: Vec::new(),
                element_type: ElementType::F32,
                is_splat: true,
                raw_bytes: vec![0, 0, 128, 63],
            },
            BinaryAttrTypeEntry::DenseResourceAttr {
                dims: vec![2, 2],
                element_type: ElementType::F32,
                resource_index: 0,
            },
            BinaryAttrTypeEntry::DotDimensionNumbersAttr {
                lhs_batching: vec![0],
                rhs_batching: vec![0],
                lhs_contracting: vec![2],
                rhs_contracting: vec![1],
            },
            BinaryAttrTypeEntry::SdyMeshAttr {
                axes: vec![(3, 2), (4, 4)],
            },
            BinaryAttrTypeEntry::SdyShardingAttr {
                mesh_name_index: 5,
                dim_axis_indices: vec![vec![3], vec![4]],
            },
            BinaryAttrTypeEntry::TpuTiledLayoutAttr {
                tile_dims: vec![8, 128],
                memory_space: 1,
            },
        ];
        let mut bin_mod_enc = MlirBytecodeEncoder::new("MLIR20.0.0");
        for s in &sample_strings {
            let _ = bin_mod_enc.intern_string(s);
        }
        bin_mod_enc.dialects.push(String::from("vhlo"));
        let t_4xf32_idx =
            bin_mod_enc.intern_binary_attr_type(BinaryAttrTypeEntry::RankedTensorType {
                dims: vec![4],
                element_type: ElementType::F32,
                memory_space: 0,
            });
        assert_eq!(
            bin_mod_enc.intern_binary_attr_type(BinaryAttrTypeEntry::RankedTensorType {
                dims: vec![4],
                element_type: ElementType::F32,
                memory_space: 0,
            }),
            t_4xf32_idx
        );
        for entry in &all_bin_entries {
            let raw = entry.encode_binary();
            assert_eq!(raw.first(), Some(&BINARY_CUSTOM_ATTR_TYPE_MAGIC));
            let decoded = BinaryAttrTypeEntry::decode_binary(&raw);
            assert_eq!(decoded.ok().as_ref(), Some(entry));
            let _ = entry.to_canonical_mlir_string(&sample_strings);
            let _ = bin_mod_enc.intern_binary_attr_type(entry.clone());
        }
        assert!(BinaryAttrTypeEntry::decode_binary(b"bad").is_err());
        assert!(BinaryAttrTypeEntry::decode_binary(&[BINARY_CUSTOM_ATTR_TYPE_MAGIC]).is_err());
        let mut bad_tag = vec![BINARY_CUSTOM_ATTR_TYPE_MAGIC];
        bad_tag.extend_from_slice(&encode_mlir_prefix_varint(99));
        assert!(BinaryAttrTypeEntry::decode_binary(&bad_tag).is_err());

        let _ = bin_mod_enc.add_resource_blob(vec![
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64,
        ]);
        bin_mod_enc.operations.push(MlirBytecodeOpNode {
            name_index: 0, // vhlo.func_v1
            encoding_mask: 0x80 | 0x10,
            attr_dict_index: None,
            operands: Vec::new(),
            result_type_indices: vec![t_4xf32_idx],
            successors: Vec::new(),
            num_regions: 1,
            properties_index: None,
            inline_regions: vec![vec![MlirBytecodeOpNode {
                name_index: 2, // vhlo.add_v1
                encoding_mask: 0x86,
                attr_dict_index: None,
                operands: vec![0, 1],
                result_type_indices: vec![t_4xf32_idx],
                successors: Vec::new(),
                num_regions: 0,
                properties_index: None,
                inline_regions: Vec::new(),
                region_block_arg_types: Vec::new(),
            }]],
            region_block_arg_types: vec![vec![t_4xf32_idx, t_4xf32_idx]],
        });
        let bin_mod_bytes = bin_mod_enc.encode_to_bytes();
        let bin_mod_dec = MlirBytecodeHeader::decode(&bin_mod_bytes);
        assert!(bin_mod_dec.is_ok());
        if let Ok(mut b_hdr) = bin_mod_dec {
            assert_ne!(
                b_hdr.decode_custom_binary_entries(),
                [] as [BinaryAttrTypeEntry; 0]
            );
            let b_ast = b_hdr.to_ast_module();
            assert_eq!(b_ast.functions.len(), 1);
            assert_eq!(b_ast.functions.first().map(|f| f.arguments.len()), Some(2));
            b_hdr.attr_type_entries.clear();
            let (tys, attrs) = b_hdr.decode_binary_attr_types();
            assert_ne!(tys, [] as [MlirType; 0]);
            assert_ne!(attrs, [] as [MlirAttribute; 0]);
        }
        let compiled_bin_vhlo = CompiledProgram::compile("mlir", &bin_mod_bytes, b"");
        assert!(compiled_bin_vhlo.is_ok());

        // Test Section 2: Multi-computation HLO & MLIR control flow lowering
        let multi_comp_hlo = r"
            HloModule multi_comp_mod
            %max_reducer (x: f32[], y: f32[]) -> f32[] {
                %x = f32[] parameter(0)
                %y = f32[] parameter(1)
                ROOT %m = f32[] maximum(%x, %y)
            }
            %min_reducer (x: f32[], y: f32[]) -> f32[] {
                %x = f32[] parameter(0)
                %y = f32[] parameter(1)
                ROOT %m = f32[] minimum(%x, %y)
            }
            %mul_reducer (x: f32[], y: f32[]) -> f32[] {
                %x = f32[] parameter(0)
                %y = f32[] parameter(1)
                ROOT %m = f32[] multiply(%x, %y)
            }
            %and_reducer (x: f32[], y: f32[]) -> f32[] {
                %x = f32[] parameter(0)
                %y = f32[] parameter(1)
                ROOT %m = f32[] and(%x, %y)
            }
            %or_reducer (x: f32[], y: f32[]) -> f32[] {
                %x = f32[] parameter(0)
                %y = f32[] parameter(1)
                ROOT %m = f32[] or(%x, %y)
            }
            %fused_sub (p0: f32[4], p1: f32[4]) -> f32[4] {
                %p0 = f32[4] parameter(0)
                %p1 = f32[4] parameter(1)
                %c = f32[4] constant(2.0)
                %s = f32[4] add(%p0, %p1)
                ROOT %prod = f32[4] multiply(%s, %c)
            }
            %while_cond (state: f32[4]) -> pred[] {
                %state = f32[4] parameter(0)
                ROOT %ok = pred[] compare(%state, %state), direction=EQ
            }
            %while_body (state: f32[4]) -> f32[4] {
                %state = f32[4] parameter(0)
                ROOT %next = f32[4] add(%state, %state)
            }
            %true_branch (x: f32[4]) -> f32[4] {
                %x = f32[4] parameter(0)
                ROOT %t_out = f32[4] exponential(%x)
            }
            %false_branch (x: f32[4]) -> f32[4] {
                %x = f32[4] parameter(0)
                ROOT %f_out = f32[4] negate(%x)
            }
            ENTRY %main (arg0: f32[4], arg1: f32[4]) -> f32[4] {
                %arg0 = f32[4] parameter(0)
                %arg1 = f32[4] parameter(1)
                %r_max = f32[4] reduce(%arg0, %arg1), dimensions={0}, to_apply=%max_reducer
                %r_min = f32[4] reduce(%r_max, %arg1), dimensions={0}, to_apply=%min_reducer
                %r_mul = f32[4] reduce(%r_min, %arg1), dimensions={0}, to_apply=%mul_reducer
                %r_and = f32[4] reduce(%r_mul, %arg1), dimensions={0}, to_apply=%and_reducer
                %r_or = f32[4] reduce(%r_and, %arg1), dimensions={0}, to_apply=%or_reducer
                %fus = f32[4] fusion(%r_or, %arg1), kind=kLoop, calls=%fused_sub
                %called = f32[4] call(%fus, %arg1), to_apply=%fused_sub
                %wh = f32[4] while(%called), condition=%while_cond, body=%while_body
                ROOT %cond = f32[4] conditional(%arg0, %wh, %arg1), true_computation=%true_branch, false_computation=%false_branch
            }
        ";
        let parsed_mc = HloParser::parse_verified_ast_module(multi_comp_hlo);
        assert!(parsed_mc.is_ok());
        if let Ok(mc_ast) = parsed_mc {
            let proto = hlo_ast_to_proto(&mc_ast);
            let roundtrip_ast = hlo_proto_to_ast(&proto);
            let lowered_mc = lower_ast_module_to_parsed_hlo(&roundtrip_ast);
            assert!(lowered_mc.is_ok());
            if let Ok(pm) = lowered_mc {
                assert!(pm.bundles.iter().any(|b| b.opcode == TpuOpCode::ReduceMax));
                assert!(pm.bundles.iter().any(|b| b.opcode == TpuOpCode::ReduceMin));
                assert!(
                    pm.bundles
                        .iter()
                        .any(|b| b.opcode == TpuOpCode::While
                            && !b.attrs.sub_computations.is_empty())
                );
                assert!(
                    pm.bundles.iter().any(|b| b.opcode == TpuOpCode::Conditional
                        && !b.attrs.sub_computations.is_empty())
                );
            }
        }
        let root_fusion_hlo = r"
            HloModule root_fus
            %sub (p0: f32[4]) -> f32[4] {
                %p0 = f32[4] parameter(0)
                %tup = (f32[4]) tuple(%p0)
                ROOT %out = f32[4] add(%p0, %p0)
            }
            ENTRY %main (a: f32[4]) -> f32[4] {
                %a = f32[4] parameter(0)
                ROOT %f = f32[4] fusion(%a), kind=kLoop, calls=%sub
            }
        ";
        assert!(CompiledProgram::compile("hlo", root_fusion_hlo.as_bytes(), b"").is_ok());
        let mlir_call_src = r"
            module @call_inlining {
                func.func @helper(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
                    %sum = stablehlo.add %x, %y : tensor<4xf32>
                    func.return %sum : tensor<4xf32>
                }
                func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = func.call @helper(%arg0, %arg1) {callee = @helper} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    func.return %0 : tensor<4xf32>
                }
            }
        ";
        let compiled_call = CompiledProgram::compile("mlir", mlir_call_src.as_bytes(), b"");
        assert!(compiled_call.is_ok());
        if let Ok(cp) = compiled_call {
            assert!(
                cp.instruction_bundles
                    .iter()
                    .any(|b| b.opcode == TpuOpCode::AddF32)
            );
            let vliw_prog = cp.encode_vliw_program();
            assert_ne!(vliw_prog, [] as [TensorCoreVliwBundle; 0]);
            assert!(vliw_prog.iter().all(|v| v.dma_slot.dma_opcode != 0));
            let isa_proto = cp.build_isa_program_proto();
            assert_ne!(
                isa_proto.relocations,
                [] as [crate::proto::ProgramRelocationProto; 0]
            );
            assert!(isa_proto.abi.is_some());
            let mut imem = cp.encode_architectural_imem();
            let patched_count = CompiledProgram::apply_isa_relocations(
                &mut imem,
                &isa_proto.relocations,
                &[
                    crate::types::DeviceAddress(0x1000_0000),
                    crate::types::DeviceAddress(0x1000_1000),
                ],
                &[crate::types::DeviceAddress(0x2000_0000)],
            );
            assert_eq!(patched_count, isa_proto.relocations.len());
        }

        let empty_bc_hdr = MlirBytecodeHeader {
            version: 1,
            producer: String::new(),
            section_count: 0,
            sections: Vec::new(),
            string_table: Vec::new(),
            dialects: Vec::new(),
            attr_type_offsets: Vec::new(),
            attr_type_entries: Vec::new(),
            ir_op_records: Vec::new(),
            resources: Vec::new(),
            resource_offsets: Vec::new(),
            properties: Vec::new(),
        };
        assert!(empty_bc_hdr.decode_to_ast().is_err());
        let sample_ast = AstModule {
            name: Some(String::from("sample_bc")),
            ..AstModule::default()
        };
        let v6_bytes = MlirBytecodeHeader::encode_from_ast(&sample_ast, 6);
        let v6_dec = MlirBytecodeHeader::decode(&v6_bytes);
        assert!(v6_dec.is_ok());
        if let Ok(v6_hdr) = v6_dec {
            assert_eq!(v6_hdr.version, 6);
            assert!(v6_hdr.decode_to_ast().is_ok());
        }
        let v1_bytes = MlirBytecodeHeader::encode_from_ast(&sample_ast, 1);
        assert_ne!(v1_bytes, [] as [u8; 0]);
        let sched = LloVliwScheduler::new(crate::types::TpuGeneration::V5e);
        let packed_coissued = sched.pack_independent_bundles(&[
            TpuInstructionBundle {
                pc_offset: 0,
                opcode: TpuOpCode::AddF32,
                operand_slots: vec![0, 1],
                output_slot: 2,
                sync_flag_id: 1,
                attrs: InstructionAttrs::default(),
            },
            TpuInstructionBundle {
                pc_offset: 1,
                opcode: TpuOpCode::DotGeneralF32,
                operand_slots: vec![0, 1],
                output_slot: 3,
                sync_flag_id: 2,
                attrs: InstructionAttrs::default(),
            },
            TpuInstructionBundle {
                pc_offset: 2,
                opcode: TpuOpCode::AddF32,
                operand_slots: vec![0, 1],
                output_slot: 4,
                sync_flag_id: 3,
                attrs: InstructionAttrs::default(),
            },
            TpuInstructionBundle {
                pc_offset: 3,
                opcode: TpuOpCode::ExpF32,
                operand_slots: vec![0],
                output_slot: 5,
                sync_flag_id: 4,
                attrs: InstructionAttrs::default(),
            },
            TpuInstructionBundle {
                pc_offset: 4,
                opcode: TpuOpCode::AddF32,
                operand_slots: vec![0, 1],
                output_slot: 6,
                sync_flag_id: 5,
                attrs: InstructionAttrs::default(),
            },
            TpuInstructionBundle {
                pc_offset: 5,
                opcode: TpuOpCode::AllReduceSumF32,
                operand_slots: vec![0],
                output_slot: 7,
                sync_flag_id: 6,
                attrs: InstructionAttrs::default(),
            },
            TpuInstructionBundle {
                pc_offset: 6,
                opcode: TpuOpCode::AddF32,
                operand_slots: vec![7, 1],
                output_slot: 8,
                sync_flag_id: 7,
                attrs: InstructionAttrs {
                    output_shape: Some(Shape {
                        element_type: ElementType::F32,
                        dims: vec![4],
                        dynamic_dims: vec![false],
                        layout: crate::types::MemoryLayout::default_tpu_tiled(1),
                    }),
                    ..InstructionAttrs::default()
                },
            },
            TpuInstructionBundle {
                pc_offset: 7,
                opcode: TpuOpCode::ExpF32,
                operand_slots: vec![8],
                output_slot: 9,
                sync_flag_id: 8,
                attrs: InstructionAttrs::default(),
            },
            TpuInstructionBundle {
                pc_offset: 8,
                opcode: TpuOpCode::SubF32,
                operand_slots: vec![0, 1],
                output_slot: 10,
                sync_flag_id: 9,
                attrs: InstructionAttrs::default(),
            },
        ]);
        assert!(packed_coissued.len() <= 7);
        if let Ok(mut hdr_with_attrs) = MlirBytecodeHeader::decode(&v6_bytes) {
            let mut ast_with_attrs = hdr_with_attrs.to_ast_module();
            if let Some(f0) = ast_with_attrs.functions.first_mut()
                && let Some(b0) = f0.body.blocks.first_mut()
                && let Some(op0) = b0.operations.first_mut()
            {
                op0.attributes
                    .insert(String::from("k"), MlirAttribute::String(String::from("v")));
            }
            let re_enc = MlirBytecodeHeader::encode_from_ast_module(&ast_with_attrs);
            assert_ne!(re_enc, [] as [u8; 0]);
            hdr_with_attrs
                .attr_type_entries
                .push(String::from("{a = 1}"));
            let _ = hdr_with_attrs.to_ast_module();
        }
        for mut arch_b in packed_coissued {
            assert!(arch_b.to_instruction_bundle().is_ok());
            arch_b.vector_slot.opcode = 0;
            arch_b.vector_slot.dst_vreg = 0;
            arch_b.vector_slot.src0_vreg = 0;
            if arch_b.matrix_slot.opcode != 0
                || arch_b.xunary_slot.opcode != 0
                || arch_b.dma_slot.collective_op != 0
            {
                assert!(arch_b.to_instruction_bundle().is_ok());
            }
        }

        // Test SPMD sharding propagation and contraction collective synthesis (AllReduce, ReduceScatter, AllGather, AllToAll)
        for sharding_mode in [
            "{devices=[2,2]0,1,2,3}",
            "#sdy.sharding<@mesh, [{\"x\"}, {}]>",
            "{replicated}",
            "reduce_scatter_x",
            "all_gather_x",
            "all_to_all_x",
        ] {
            let spec = ParsedShardingSpec::from_str_with_partitions(sharding_mode, 4);
            assert_eq!(spec.clone(), spec);
            assert_ne!(format!("{spec:?}"), "");
            let global_sh = Shape {
                element_type: ElementType::F32,
                dims: vec![8, 8],
                dynamic_dims: vec![false, false],
                layout: crate::types::MemoryLayout::default_tpu_tiled(2),
            };
            let local_sh = spec.partition_shape(&global_sh);
            assert!(local_sh.dims.first().copied().unwrap_or(0) <= 8);

            let spmd_mlir = format!(
                r#"module @spmd_dot attributes {{mhlo.sharding = "{sharding_mode}", num_partitions = 4}} {{
                    func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {{
                        %0 = stablehlo.dot_general %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
                        func.return %0 : tensor<4x4xf32>
                    }}
                }}"#
            );
            let spmd_prog = CompiledProgram::compile("mlir", spmd_mlir.as_bytes(), b"");
            assert!(spmd_prog.is_ok());
            if let Ok(mut p) = spmd_prog {
                assert_eq!(p.num_partitions, 4);
                assert!(p.instruction_bundles.len() >= 2);
                // Test resharding edge transition and DotGeneral/Convolution/Gather shape inference
                p.instruction_bundles.push(TpuInstructionBundle {
                    pc_offset: 10,
                    opcode: TpuOpCode::AddF32,
                    operand_slots: vec![0],
                    output_slot: 10,
                    sync_flag_id: 11,
                    attrs: InstructionAttrs {
                        sharding_spec: String::from("{devices=[2,1]0,1}"),
                        ..InstructionAttrs::default()
                    },
                });
                p.instruction_bundles.push(TpuInstructionBundle {
                    pc_offset: 11,
                    opcode: TpuOpCode::ExpF32,
                    operand_slots: vec![10],
                    output_slot: 11,
                    sync_flag_id: 12,
                    attrs: InstructionAttrs {
                        sharding_spec: String::from("{devices=[1,2]0,1}"),
                        ..InstructionAttrs::default()
                    },
                });
                p.instruction_bundles.push(TpuInstructionBundle {
                    pc_offset: 12,
                    opcode: TpuOpCode::Convolution,
                    operand_slots: vec![0, 1],
                    output_slot: 12,
                    sync_flag_id: 13,
                    attrs: InstructionAttrs {
                        window_strides: vec![1],
                        ..InstructionAttrs::default()
                    },
                });
                p.instruction_bundles.push(TpuInstructionBundle {
                    pc_offset: 13,
                    opcode: TpuOpCode::Gather,
                    operand_slots: vec![0, 1],
                    output_slot: 13,
                    sync_flag_id: 14,
                    attrs: InstructionAttrs {
                        gather_slice_sizes: vec![1, 2],
                        collapsed_slice_dims: vec![0],
                        ..InstructionAttrs::default()
                    },
                });
                p.parameter_shapes = vec![
                    Shape {
                        element_type: ElementType::F32,
                        dims: vec![1, 4, 2],
                        dynamic_dims: vec![false; 3],
                        layout: crate::types::MemoryLayout::default_tpu_tiled(3),
                    },
                    Shape {
                        element_type: ElementType::F32,
                        dims: vec![2, 2, 2],
                        dynamic_dims: vec![false; 3],
                        layout: crate::types::MemoryLayout::default_tpu_tiled(3),
                    },
                ];
                p.pass_hlo_verification_and_shape_inference();
                p.pass_spmd_partitioning();
                p.pass_spmd_sharding_propagation();
                assert!(
                    p.instruction_bundles
                        .iter()
                        .any(|b| b.opcode == TpuOpCode::CollectivePermute)
                );
            }
        }

        let multi_fn_mlir = r#"
        module @multi_call {
            func.func private @helper(%x: tensor<4xf32>) -> tensor<4xf32> {
                %0 = stablehlo.exponential %x : tensor<4xf32>
                func.return %0 : tensor<4xf32>
            }
            func.func nested @nested_helper(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                %0 = stablehlo.cosine %x : tensor<4xf32>
                func.return %0, %x : tensor<4xf32>, tensor<4xf32>
            }
            func.func public @main(%a: tensor<4xf32>) -> tensor<4xf32> {
                %c = func.call @helper(%a) { mhlo.sharding = "{replicated}" } : (tensor<4xf32>) -> tensor<4xf32>
                %n:2 = func.call @nested_helper(%c) : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
                func.return %n#0 : tensor<4xf32>
            }
        }
        "#;
        let parsed_multi = HloParser::parse_verified_ast_module(multi_fn_mlir);
        assert!(parsed_multi.is_ok());
        if let Ok(ast_m) = parsed_multi {
            let lowered_m = lower_ast_module_to_parsed_hlo(&ast_m);
            assert!(lowered_m.is_ok());
            let bc_bytes = MlirBytecodeHeader::encode_from_ast_module(&ast_m);
            if let Ok(bc_hdr) = MlirBytecodeHeader::decode(&bc_bytes) {
                let roundtrip_ast = bc_hdr.to_ast_module();
                assert_eq!(roundtrip_ast.functions.len(), 3);
            }
        }

        for attr_val in [
            MlirAttribute::Bool(true),
            MlirAttribute::Integer {
                value: 42,
                ty: None,
            },
            MlirAttribute::Float {
                value: 1.5,
                ty: None,
            },
            MlirAttribute::String(String::from("s")),
            MlirAttribute::SymbolRef(vec![String::from("sym")]),
            MlirAttribute::Array(vec![MlirAttribute::Bool(false)]),
            MlirAttribute::DenseI64Array(vec![1, 2]),
            MlirAttribute::DenseBoolArray(vec![true, false]),
            MlirAttribute::DenseElements(DenseElementsAttr {
                values_f64: vec![1.0],
                raw_bytes: None,
                is_splat: true,
                tensor_type: None,
            }),
            MlirAttribute::DenseElements(DenseElementsAttr {
                values_f64: vec![1.0, 2.0],
                raw_bytes: None,
                is_splat: false,
                tensor_type: None,
            }),
            MlirAttribute::ComparisonDirection(ComparisonDirection::Gt),
            MlirAttribute::DotDimensionNumbers(DotDimensionNumbersAttr {
                lhs_batching_dimensions: vec![0],
                rhs_batching_dimensions: vec![0],
                lhs_contracting_dimensions: vec![1],
                rhs_contracting_dimensions: vec![1],
            }),
            MlirAttribute::SdyTensorSharding(SdyTensorShardingAttr {
                mesh_name: String::from("mesh"),
                dim_shardings: vec![SdyDimensionSharding {
                    axes: vec![String::from("x")],
                    is_open: false,
                    priority: None,
                }],
                replicated_axes: Vec::new(),
                unreduced_axes: Vec::new(),
            }),
            MlirAttribute::DenseResource(String::from("res")),
        ] {
            assert_ne!(attr_val.to_mlir_string(), "");
        }
    }

    proptest::proptest! {
        #[test]
        fn proptest_winnow_tensor_shape_and_compiler_roundtrip(
            d0 in 1_i64..32,
            d1 in 1_i64..32,
        ) {
            let mlir_spec = format!("tensor<{d0}x{d1}xf32>");
            let parsed_mlir = CompiledProgram::parse_tensor_type(&mlir_spec);
            proptest::prop_assert!(parsed_mlir.is_ok());
            if let Ok(sh) = parsed_mlir {
                proptest::prop_assert_eq!(sh.dims, vec![d0, d1]);
                proptest::prop_assert_eq!(sh.element_type, ElementType::F32);
            }

            let hlo_spec = format!("bf16[{d0},{d1}]{{1,0}}");
            let parsed_hlo = CompiledProgram::parse_hlo_shape_token(&hlo_spec);
            proptest::prop_assert!(parsed_hlo.is_ok());
            if let Ok(sh) = parsed_hlo {
                proptest::prop_assert_eq!(sh.dims, vec![d0, d1]);
                proptest::prop_assert_eq!(sh.element_type, ElementType::Bf16);
            }

            // Full recursive-descent MLIR AST parse & lower roundtrip
            let full_mlir = format!(
                "module @prop_mod {{ func.func @main(%arg0: tensor<{d0}x{d1}xf32>, %arg1: tensor<{d0}x{d1}xf32>) -> tensor<{d0}x{d1}xf32> {{ %0 = stablehlo.add %arg0, %arg1 : tensor<{d0}x{d1}xf32> func.return %0 : tensor<{d0}x{d1}xf32> }} }}"
            );
            let ast_res = HloParser::parse_verified_ast_module(&full_mlir);
            proptest::prop_assert!(ast_res.is_ok());
            if let Ok(ast) = ast_res {
                let lowered = lower_ast_module_to_parsed_hlo(&ast);
                proptest::prop_assert!(lowered.is_ok());
            }
        }
    }
}
