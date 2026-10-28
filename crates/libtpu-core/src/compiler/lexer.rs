//! `StableHLO` MLIR and XLA HLO lexer, recursive-descent parser, and Mosaic metadata extractor.

use super::CompiledProgram;
use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use crate::error::LibTpuResult;
use crate::proto::HloModuleProto;
use crate::types::{ElementType, MemoryLayout, Shape};
use winnow::Parser;
use winnow::ascii::{digit1, multispace0};
use winnow::combinator::{alt, delimited, preceded, repeat, separated};
use winnow::error::ContextError;
use winnow::token::{any, take_till, take_while};

/// Lowered module components: `(name, root_opcode, parameter_shapes, output_shapes, instruction_bundles, output_slots)`.
pub(super) type LoweredModuleTuple = (
    String,
    TpuOpCode,
    Vec<Shape>,
    Vec<Shape>,
    Vec<TpuInstructionBundle>,
    Vec<usize>,
);

/// Token produced by [`HloLexer`] when lexing `StableHLO` MLIR or XLA HLO text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HloToken {
    /// Identifier, keyword, or dotted mnemonic (e.g., `module`, `func.func`, `stablehlo.add`).
    Ident(String),
    /// SSA value identifier starting with `%` (e.g., `%arg0`, `%0`).
    SsaValue(String),
    /// Module or function symbol starting with `@` (e.g., `@main`).
    Symbol(String),
    /// Numeric literal token (e.g., `4`, `2.5`, `-1`).
    Number(String),
    /// Delimiter or punctuation character (`=`, `:`, `,`, `(`, `)`, `{`, `}`, `[`, `]`, `<`, `>`).
    Punct(char),
    /// Arrow token (`->`).
    Arrow,
}

/// Lexer for `StableHLO` MLIR and XLA HLO textual modules powered by `winnow`.
#[derive(Debug, Clone, Default)]
pub struct HloLexer;

/// Parses a single [`HloToken`] from the front of `input` using `winnow` combinators.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` is empty after skipping whitespace.
fn winnow_lex_one_token(input: &mut &str) -> Result<HloToken, ContextError> {
    *input = input.trim_start();
    alt((
        "->".value(HloToken::Arrow),
        preceded(
            '-',
            take_while(0.., |c: char| c.is_ascii_digit() || c == '.'),
        )
        .map(|digits: &str| HloToken::Number(format!("-{digits}"))),
        preceded(
            '%',
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
        .map(|name: &str| HloToken::SsaValue(format!("%{name}"))),
        preceded(
            '@',
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
        .map(|name: &str| HloToken::Symbol(String::from(name))),
        (
            take_while(1.., |c: char| c.is_ascii_digit()),
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
            .take()
            .map(|num: &str| HloToken::Number(String::from(num))),
        (
            take_while(1.., |c: char| matches!(c, 'a'..='z' | 'A'..='Z' | '_')),
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
            .take()
            .map(|ident: &str| HloToken::Ident(String::from(ident))),
        any.map(HloToken::Punct),
    ))
    .parse_next(input)
}

/// Extracts the inner body of a `tensor<...>` specification using `winnow`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` does not begin with `tensor<...>`.
fn winnow_extract_tensor_body<'a>(input: &mut &'a str) -> Result<&'a str, ContextError> {
    *input = input.trim_start();
    let body = delimited("tensor<", take_till(1.., |c: char| c == '>'), '>').parse_next(input)?;
    *input = "";
    Ok(body.trim())
}

/// Parses an XLA HLO `<dtype>[<dims>]` shape specification using `winnow`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` is not a valid `<dtype>[<dims>]` specification.
fn winnow_parse_hlo_shape_parts<'a>(
    input: &mut &'a str,
) -> Result<(&'a str, Vec<i64>), ContextError> {
    *input = input.trim_start();
    let dtype = take_till(1.., |c: char| c == '[').parse_next(input)?;
    let dims: Vec<i64> = delimited(
        '[',
        separated(
            0..,
            delimited(multispace0, digit1.try_map(str::parse::<i64>), multispace0),
            ',',
        ),
        ']',
    )
    .parse_next(input)?;
    *input = "";
    Ok((dtype.trim(), dims))
}

impl HloLexer {
    /// Tokenizes a `StableHLO` MLIR or XLA HLO source string into a sequence of [`HloToken`]s using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `src` - Source text to tokenize.
    #[must_use]
    pub fn tokenize(src: &str) -> Vec<HloToken> {
        let mut parser = terminated_token_stream();
        parser.parse(src).unwrap_or_default()
    }
}

/// Builds a `winnow` parser that collects all [`HloToken`]s and consumes trailing whitespace.
fn terminated_token_stream<'a>() -> impl Parser<&'a str, Vec<HloToken>, ContextError> {
    (repeat(0.., winnow_lex_one_token), multispace0).map(|(tokens, _)| tokens)
}

/// Recursive-descent parser for `StableHLO` MLIR and XLA HLO modules.
#[derive(Debug, Clone, Default)]
pub struct HloParser;

impl HloParser {
    /// Extracts the module name from a token stream (`@name` or `HloModule name`).
    ///
    /// # Arguments
    ///
    /// * `tokens` - Tokenized module slice.
    #[must_use]
    pub fn parse_module_name(tokens: &[HloToken]) -> String {
        for (idx, tok) in tokens.iter().enumerate() {
            if let HloToken::Symbol(sym) = tok {
                if !sym.is_empty() {
                    return sym.clone();
                }
            }
            if let HloToken::Ident(kw) = tok {
                if kw == "HloModule" {
                    if let Some(HloToken::Ident(name)) = tokens.get(idx + 1) {
                        return name.clone();
                    }
                }
            }
        }
        String::from("tpu_module")
    }
}

/// Extracted JAX Mosaic TPU and MHLO target attributes from a textual MLIR module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MosaicTargetMetadata {
    /// Target TPU generation string (`tpu.compilation_target`, e.g., `"tpu_v5p"`).
    pub compilation_target: String,
    /// Target TPU core type (`tpu.core_type`, e.g., `"tensor_core"` or `"sparse_core"`).
    pub core_type: String,
    /// Layout mode annotation (`mhlo.layout_mode`, e.g., `"default"` or `"{1,0}"`).
    pub layout_mode: String,
    /// Sharding annotation (`mhlo.sharding`, e.g., `"{replicated}"`).
    pub sharding: String,
    /// Whether any Mosaic `tpu.*` dialect operations were detected.
    pub has_mosaic_ops: bool,
}

impl CompiledProgram {
    /// Extracts JAX Mosaic TPU (`tpu.*`) and `mhlo.*` target attributes from an MLIR module string.
    ///
    /// # Arguments
    ///
    /// * `text` - Source MLIR module text.
    #[must_use]
    pub fn parse_mosaic_metadata(text: &str) -> MosaicTargetMetadata {
        let mut meta = MosaicTargetMetadata {
            has_mosaic_ops: text.contains("tpu."),
            ..MosaicTargetMetadata::default()
        };
        for (key, dst) in [
            ("tpu.compilation_target", &mut meta.compilation_target),
            ("tpu.core_type", &mut meta.core_type),
            ("mhlo.layout_mode", &mut meta.layout_mode),
            ("mhlo.sharding", &mut meta.sharding),
        ] {
            if let Some(val) = text
                .split_once(key)
                .and_then(|(_, after)| after.split_once('"'))
                .and_then(|(_, quoted)| quoted.split_once('"'))
                .map(|(v, _)| v)
            {
                *dst = String::from(val);
            }
        }
        meta
    }

    /// Parses a `tensor<...>` or HLO shape token (e.g., `tensor<2x4xf32>`, `tensor<8xs32>`) into a [`Shape`] using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `token` - Tensor type specification substring.
    #[must_use]
    pub fn parse_tensor_type(token: &str) -> Option<Shape> {
        let trimmed = winnow_extract_tensor_body.parse(token).ok()?;
        let (dims_part, dtype_str) = match trimmed.rsplit_once('x') {
            Some((d, dt)) => (Some(d), dt),
            None => (None, trimmed),
        };
        let elem_type = match dtype_str {
            "f32" => ElementType::F32,
            "f64" => ElementType::F64,
            "bf16" => ElementType::Bf16,
            "f16" => ElementType::F16,
            "s32" => ElementType::S32,
            "s64" => ElementType::S64,
            "s16" => ElementType::S16,
            "s8" => ElementType::S8,
            "u32" => ElementType::U32,
            "u64" => ElementType::U64,
            "u16" => ElementType::U16,
            "u8" => ElementType::U8,
            "i1" | "pred" => ElementType::Pred,
            "c64" | "complex<f32>" => ElementType::C64,
            "c128" | "complex<f64>" => ElementType::C128,
            "f8e5m2" => ElementType::F8E5M2,
            "f8e4m3fn" => ElementType::F8E4M3Fn,
            "f8e4m3b11fnuz" => ElementType::F8E4M3B11Fnuz,
            "s4" | "i4" => ElementType::S4,
            "u4" | "ui4" => ElementType::U4,
            "s2" | "i2" => ElementType::S2,
            "u2" | "ui2" => ElementType::U2,
            _ => return None,
        };
        let mut dims = Vec::new();
        if let Some(dp) = dims_part {
            for ds in dp.split('x') {
                let parsed = ds.parse::<i64>().ok()?;
                dims.push(parsed);
            }
        }
        Shape::new(elem_type, dims, None, None).ok()
    }

    /// Lowers a decoded binary [`HloModuleProto`] into module name, [`TpuOpCode`], parameter shapes,
    /// output shapes, VLIW instruction bundles, and output register slots.
    ///
    /// # Arguments
    ///
    /// * `proto` - Decoded binary `HloModuleProto`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError`] if an instruction shape specifies an unsupported [`ElementType`].
    #[allow(clippy::too_many_lines)]
    pub(super) fn lower_from_hlo_proto(proto: &HloModuleProto) -> LibTpuResult<LoweredModuleTuple> {
        let mut param_shapes = Vec::new();
        let mut slot_shapes = Vec::new();
        let mut root_op = TpuOpCode::Identity;
        let mut elem_type = ElementType::F32;
        let mut dims = vec![4_i64];
        let mut bundles = Vec::new();
        let mut id_to_slot = std::collections::BTreeMap::new();
        let mut next_slot: usize = 0;
        let mut pc: u32 = 0;

        for comp in &proto.computations {
            for inst in &comp.instructions {
                if let Some(shape) = &inst.shape {
                    if shape.element_type != 0 {
                        elem_type = ElementType::from_raw(shape.element_type)?;
                    }
                    if !shape.dimensions.is_empty() {
                        dims.clone_from(&shape.dimensions);
                    }
                }
                let inst_shape = Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                };
                if inst.opcode == "parameter" {
                    id_to_slot.insert(inst.id, next_slot);
                    next_slot += 1;
                    slot_shapes.push(inst_shape.clone());
                    param_shapes.push(inst_shape);
                } else {
                    let (op, et, _, _) = Self::parse_opcode_mnemonic(&inst.opcode);
                    root_op = op;
                    elem_type = et;
                    let op_slots: Vec<usize> = if inst.operand_ids.is_empty() {
                        (0..op.arity()).collect()
                    } else {
                        inst.operand_ids
                            .iter()
                            .enumerate()
                            .map(|(idx, id)| id_to_slot.get(id).copied().unwrap_or(idx))
                            .collect()
                    };
                    let in_shapes: Vec<Shape> = op_slots
                        .iter()
                        .filter_map(|&s| slot_shapes.get(s).cloned())
                        .collect();
                    let out_slot = next_slot;
                    id_to_slot.insert(inst.id, out_slot);
                    next_slot += 1;
                    slot_shapes.push(inst_shape.clone());
                    bundles.push(TpuInstructionBundle {
                        pc_offset: pc,
                        opcode: op,
                        operand_slots: op_slots,
                        output_slot: out_slot,
                        sync_flag_id: pc + 1,
                        attrs: InstructionAttrs {
                            input_shapes: in_shapes,
                            output_shape: Some(inst_shape),
                            ..InstructionAttrs::default()
                        },
                    });
                    pc += 1;
                }
            }
        }

        if param_shapes.is_empty() {
            let arity = root_op.arity();
            for _ in 0..arity {
                param_shapes.push(Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                });
            }
        }
        let out_shape = Shape {
            element_type: elem_type,
            dynamic_dims: vec![false; dims.len()],
            layout: MemoryLayout::default_tpu_tiled(dims.len()),
            dims,
        };
        if bundles.is_empty() {
            bundles.push(TpuInstructionBundle {
                pc_offset: 0,
                opcode: root_op,
                operand_slots: (0..param_shapes.len()).collect(),
                output_slot: param_shapes.len(),
                sync_flag_id: 1,
                attrs: InstructionAttrs {
                    input_shapes: param_shapes.clone(),
                    output_shape: Some(out_shape.clone()),
                    ..InstructionAttrs::default()
                },
            });
        }
        let last_slot = bundles.last().map_or(param_shapes.len(), |b| b.output_slot);
        Ok((
            proto.name.clone(),
            root_op,
            param_shapes,
            vec![out_shape],
            bundles,
            vec![last_slot],
        ))
    }

    /// Extracts a module name from textual MLIR/HLO (`module @foo` or `HloModule foo`) or returns `"tpu_module"`.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 lossy program text.
    pub(super) fn extract_module_name(text: &str) -> String {
        let tokens = HloLexer::tokenize(text);
        HloParser::parse_module_name(&tokens)
    }

    /// Parses a comma-separated list of `usize` integers following `key` inside `line` (e.g., `permutation = [1, 0]`).
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `key` - Attribute key substring to search for.
    #[must_use]
    pub fn parse_usize_list_attr(line: &str, key: &str) -> Vec<usize> {
        let tokens = HloLexer::tokenize(line);
        let Some(key_pos) = tokens
            .iter()
            .position(|t| matches!(t, HloToken::Ident(id) if id.contains(key)))
        else {
            return Vec::new();
        };
        let tail = tokens.get(key_pos + 1..).unwrap_or(&[]);
        let op = tail
            .iter()
            .position(|t| matches!(t, HloToken::Punct('[' | '{' | '<')))
            .unwrap_or(0);
        let cl = tail
            .iter()
            .position(|t| matches!(t, HloToken::Punct(']' | '}' | '>')))
            .unwrap_or(0);
        if let Some(inner) = tail.get(op + 1..cl) {
            let start = inner
                .iter()
                .rposition(|t| matches!(t, HloToken::Punct(':')))
                .map_or(0, |p| p + 1);
            return inner
                .get(start..)
                .unwrap_or(inner)
                .iter()
                .filter_map(|t| match t {
                    HloToken::Number(n) => n.parse::<usize>().ok(),
                    _ => None,
                })
                .collect();
        }
        Vec::new()
    }

    /// Parses a single `usize` integer attribute following `key` inside `line` (e.g., `dimension = 1`).
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `key` - Attribute key substring to search for.
    #[must_use]
    pub fn parse_single_usize_attr(line: &str, key: &str) -> Option<usize> {
        let tokens = HloLexer::tokenize(line);
        let key_pos = tokens
            .iter()
            .position(|t| matches!(t, HloToken::Ident(id) if id == key))?;
        if tokens.get(key_pos + 1) != Some(&HloToken::Punct('=')) {
            return None;
        }
        match tokens.get(key_pos + 2) {
            Some(HloToken::Number(n)) => n.parse::<usize>().ok(),
            _ => None,
        }
    }

    /// Parses a textual XLA HLO shape token (e.g., `f32[2,3]{1,0}` or `s32[4]`) into a [`Shape`] using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `token` - HLO shape token.
    #[must_use]
    pub fn parse_hlo_shape_token(token: &str) -> Option<Shape> {
        let (dtype_str, dims) = winnow_parse_hlo_shape_parts.parse(token).ok()?;
        let elem_type = match dtype_str {
            "f32" => ElementType::F32,
            "f64" => ElementType::F64,
            "bf16" => ElementType::Bf16,
            "f16" => ElementType::F16,
            "s32" => ElementType::S32,
            "s64" => ElementType::S64,
            "s16" => ElementType::S16,
            "s8" => ElementType::S8,
            "u32" => ElementType::U32,
            "u64" => ElementType::U64,
            "u16" => ElementType::U16,
            "u8" => ElementType::U8,
            "pred" | "i1" => ElementType::Pred,
            _ => return None,
        };
        Shape::new(elem_type, dims, None, None).ok()
    }

    /// Parses instruction attributes (`dot_general` dims, `permutation`, `broadcast_dimensions`,
    /// `slice` bounds, `reduce` dims, `dense<...>` constants, and result shape) from an instruction line.
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `input_shapes` - Shapes of the resolved input operands.
    /// * `default_elem_type` - Fallback element type for constant packing.
    #[must_use]
    pub fn parse_instruction_attrs(
        line: &str,
        input_shapes: &[Shape],
        default_elem_type: ElementType,
    ) -> InstructionAttrs {
        let mut attrs = InstructionAttrs {
            input_shapes: input_shapes.to_vec(),
            ..InstructionAttrs::default()
        };
        attrs.dot_lhs_contracting_dims = Self::parse_usize_list_attr(line, "lhs_contracting_dim");
        attrs.dot_rhs_contracting_dims = Self::parse_usize_list_attr(line, "rhs_contracting_dim");
        attrs.dot_lhs_batch_dims = Self::parse_usize_list_attr(line, "lhs_batching_dim");
        attrs.dot_rhs_batch_dims = Self::parse_usize_list_attr(line, "rhs_batching_dim");

        attrs.permutation = Self::parse_usize_list_attr(line, "permutation");
        if attrs.permutation.is_empty() {
            attrs.permutation = Self::parse_usize_list_attr(line, "dims");
        }
        attrs.broadcast_dims = Self::parse_usize_list_attr(line, "broadcast_dimensions");
        attrs.slice_starts = Self::parse_usize_list_attr(line, "start_indices");
        attrs.slice_limits = Self::parse_usize_list_attr(line, "limit_indices");
        attrs.slice_strides = Self::parse_usize_list_attr(line, "strides");
        attrs.pad_edge_low = Self::parse_usize_list_attr(line, "edge_padding_low");
        attrs.pad_edge_high = Self::parse_usize_list_attr(line, "edge_padding_high");
        attrs.pad_interior = Self::parse_usize_list_attr(line, "interior_padding");
        attrs.reduce_dims = Self::parse_usize_list_attr(line, "dimensions");
        if let Some(cd) = Self::parse_single_usize_attr(line, "dimension") {
            attrs.concat_dim = cd;
        }
        if let Some(id) = Self::parse_single_usize_attr(line, "iota_dimension") {
            attrs.iota_dim = id;
        }

        // Extract output shape from `-> tensor<...>` or `: tensor<...>` or HLO `<type>[<dims>]`
        if let Some((_, seg)) = line.rsplit_once("tensor<") {
            attrs.output_shape = Self::parse_tensor_type(&format!("tensor<{seg}"));
        }
        if attrs.output_shape.is_none() {
            for tok in line.split_whitespace() {
                if let Some(sh) = Self::parse_hlo_shape_token(tok) {
                    attrs.output_shape = Some(sh);
                    break;
                }
            }
        }

        let elem_type = attrs
            .output_shape
            .as_ref()
            .map_or(default_elem_type, |s| s.element_type);
        if let Some((_, after_dense)) = line.split_once("dense<") {
            let inner = after_dense.split('>').next().unwrap_or(after_dense);
            let cleaned: String = inner
                .chars()
                .map(|c| if c == '[' || c == ']' { ' ' } else { c })
                .collect();
            let vals: Vec<f64> = cleaned
                .split(',')
                .filter_map(|t| t.trim().parse::<f64>().ok())
                .collect();
            let count = attrs
                .output_shape
                .as_ref()
                .map_or(vals.len(), Shape::element_count)
                .max(vals.len());
            let expanded = if vals.len() == 1 && count > 1 {
                vec![vals.first().copied().unwrap_or(0.0); count]
            } else {
                vals
            };
            attrs.constant_bytes = elem_type.pack_from_f64_slice(&expanded);
        }
        attrs
    }

    /// Parses an instruction line or mnemonic token into `(TpuOpCode, ElementType, arity, default_dims)`.
    ///
    /// # Arguments
    ///
    /// * `token` - Instruction line or operation mnemonic.
    #[allow(clippy::too_many_lines)]
    pub(super) fn parse_opcode_mnemonic(token: &str) -> (TpuOpCode, ElementType, usize, Vec<i64>) {
        let lower = token.to_ascii_lowercase();
        let table: &[(&str, TpuOpCode, ElementType, usize, &[i64])] = &[
            ("s32_add", TpuOpCode::AddS32, ElementType::S32, 2, &[4]),
            ("mul_f32", TpuOpCode::MulF32, ElementType::F32, 2, &[4]),
            ("multiply", TpuOpCode::MulF32, ElementType::F32, 2, &[4]),
            ("sub_f32", TpuOpCode::SubF32, ElementType::F32, 2, &[4]),
            ("subtract", TpuOpCode::SubF32, ElementType::F32, 2, &[4]),
            ("div_f32", TpuOpCode::DivF32, ElementType::F32, 2, &[4]),
            ("divide", TpuOpCode::DivF32, ElementType::F32, 2, &[4]),
            ("max_f32", TpuOpCode::MaxF32, ElementType::F32, 2, &[4]),
            ("maximum", TpuOpCode::MaxF32, ElementType::F32, 2, &[4]),
            ("min_f32", TpuOpCode::MinF32, ElementType::F32, 2, &[4]),
            ("minimum", TpuOpCode::MinF32, ElementType::F32, 2, &[4]),
            (
                "dot_general",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_push_rhs",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_acc",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_pop",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.enqueue_dma",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.wait_dma",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.dma_copy",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.dma_wait",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_signal",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_wait",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_read",
                TpuOpCode::CustomCall,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "tpu.barrier",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.vector_load",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.vector_store",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.load", TpuOpCode::Identity, ElementType::F32, 1, &[4]),
            ("tpu.store", TpuOpCode::Identity, ElementType::F32, 1, &[4]),
            ("tpu.iota", TpuOpCode::Iota, ElementType::S32, 0, &[4]),
            (
                "tpu.repeat",
                TpuOpCode::BroadcastInDim,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.rotate", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "tpu.pack_subelements",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.unpack_subelements",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.memref_squeeze",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.memref_bitcast",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.relabel",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.bitcast", TpuOpCode::Reshape, ElementType::F32, 1, &[4]),
            ("tpu.roll", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "tpu.memref_slice",
                TpuOpCode::Slice,
                ElementType::F32,
                1,
                &[2],
            ),
            (
                "tpu.erase_memref_layout",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.region",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "collective_permute",
                TpuOpCode::CollectivePermute,
                ElementType::F32,
                1,
                &[4],
            ),
            ("all_to_all", TpuOpCode::AllToAll, ElementType::F32, 1, &[4]),
            (
                "exponential_minus_one",
                TpuOpCode::Expm1,
                ElementType::F32,
                1,
                &[4],
            ),
            ("expm1", TpuOpCode::Expm1, ElementType::F32, 1, &[4]),
            ("log_plus_one", TpuOpCode::Log1p, ElementType::F32, 1, &[4]),
            ("log1p", TpuOpCode::Log1p, ElementType::F32, 1, &[4]),
            ("logistic", TpuOpCode::Logistic, ElementType::F32, 1, &[4]),
            ("erf", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            (
                "round_nearest_even",
                TpuOpCode::RoundNearestEven,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "round_nearest_afz",
                TpuOpCode::RoundNearestAfz,
                ElementType::F32,
                1,
                &[4],
            ),
            ("sign", TpuOpCode::Sign, ElementType::F32, 1, &[4]),
            ("is_finite", TpuOpCode::IsFinite, ElementType::Pred, 1, &[4]),
            ("cbrt", TpuOpCode::Cbrt, ElementType::F32, 1, &[4]),
            ("tan(", TpuOpCode::Tan, ElementType::F32, 1, &[4]),
            ("stablehlo.tan", TpuOpCode::Tan, ElementType::F32, 1, &[4]),
            ("power", TpuOpCode::Pow, ElementType::F32, 2, &[4]),
            ("atan2", TpuOpCode::Atan2, ElementType::F32, 2, &[4]),
            ("clamp", TpuOpCode::Clamp, ElementType::F32, 3, &[4]),
            (
                "popcnt",
                TpuOpCode::PopulationCount,
                ElementType::S32,
                1,
                &[4],
            ),
            ("pad", TpuOpCode::Pad, ElementType::F32, 2, &[4]),
            (
                "select_and_scatter",
                TpuOpCode::Scatter,
                ElementType::F32,
                3,
                &[4],
            ),
            (
                "batch_norm_inference",
                TpuOpCode::MulF32,
                ElementType::F32,
                2,
                &[4],
            ),
            ("fft", TpuOpCode::Fft, ElementType::F32, 1, &[4]),
            (
                "cholesky",
                TpuOpCode::Cholesky,
                ElementType::F32,
                1,
                &[2, 2],
            ),
            (
                "triangular_solve",
                TpuOpCode::TriangularSolve,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            ("top_k", TpuOpCode::TopK, ElementType::F32, 1, &[4]),
            (
                "uniform_quantize",
                TpuOpCode::UniformQuantize,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "uniform_dequantize",
                TpuOpCode::UniformDequantize,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "partition_id",
                TpuOpCode::Constant,
                ElementType::U32,
                0,
                &[1],
            ),
            ("replica_id", TpuOpCode::Constant, ElementType::U32, 0, &[1]),
            (
                "after_all",
                TpuOpCode::Identity,
                ElementType::Token,
                1,
                &[1],
            ),
            ("infeed", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("outfeed", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("send", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("recv", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            (
                "get_tuple_element",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tuple", TpuOpCode::Identity, ElementType::Tuple, 1, &[4]),
            (
                " dot(",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            ("neg_f32", TpuOpCode::NegF32, ElementType::F32, 1, &[4]),
            ("negate", TpuOpCode::NegF32, ElementType::F32, 1, &[4]),
            ("exp_f32", TpuOpCode::ExpF32, ElementType::F32, 1, &[4]),
            ("exponential", TpuOpCode::ExpF32, ElementType::F32, 1, &[4]),
            ("tanh_f32", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            ("tanh", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            ("abs", TpuOpCode::Abs, ElementType::F32, 1, &[4]),
            ("log", TpuOpCode::Log, ElementType::F32, 1, &[4]),
            ("rsqrt", TpuOpCode::Rsqrt, ElementType::F32, 1, &[4]),
            ("sqrt", TpuOpCode::Sqrt, ElementType::F32, 1, &[4]),
            ("cosine", TpuOpCode::Cosine, ElementType::F32, 1, &[4]),
            ("sine", TpuOpCode::Sine, ElementType::F32, 1, &[4]),
            ("floor", TpuOpCode::Floor, ElementType::F32, 1, &[4]),
            ("ceil", TpuOpCode::Ceil, ElementType::F32, 1, &[4]),
            ("not", TpuOpCode::Not, ElementType::Pred, 1, &[4]),
            ("remainder", TpuOpCode::Rem, ElementType::F32, 2, &[4]),
            (
                "compare_lt",
                TpuOpCode::CompareLt,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "compare_gt",
                TpuOpCode::CompareGt,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "compare_eq",
                TpuOpCode::CompareEq,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "shift_left",
                TpuOpCode::ShiftLeft,
                ElementType::S32,
                2,
                &[4],
            ),
            (
                "shift_right_arithmetic",
                TpuOpCode::ShiftRightArithmetic,
                ElementType::S32,
                2,
                &[4],
            ),
            (
                "shift_right_logical",
                TpuOpCode::ShiftRightLogical,
                ElementType::U32,
                2,
                &[4],
            ),
            (
                "population_count",
                TpuOpCode::PopulationCount,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "count_leading_zeros",
                TpuOpCode::CountLeadingZeros,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "convolution",
                TpuOpCode::Convolution,
                ElementType::F32,
                2,
                &[4],
            ),
            ("gather", TpuOpCode::Gather, ElementType::F32, 2, &[4]),
            ("scatter", TpuOpCode::Scatter, ElementType::F32, 3, &[4]),
            (
                "dynamic_update_slice",
                TpuOpCode::DynamicUpdateSlice,
                ElementType::F32,
                3,
                &[4],
            ),
            (
                "dynamic_slice",
                TpuOpCode::DynamicSlice,
                ElementType::F32,
                2,
                &[2],
            ),
            (
                "reduce_window",
                TpuOpCode::ReduceWindow,
                ElementType::F32,
                1,
                &[4],
            ),
            ("sort", TpuOpCode::Sort, ElementType::F32, 1, &[4]),
            (
                "rng_bit_generator",
                TpuOpCode::RngBitGenerator,
                ElementType::U32,
                1,
                &[4],
            ),
            ("select", TpuOpCode::Select, ElementType::F32, 3, &[4]),
            ("iota", TpuOpCode::Iota, ElementType::F32, 1, &[4]),
            ("reverse", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "reduce_max",
                TpuOpCode::ReduceMax,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_min",
                TpuOpCode::ReduceMin,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_sum",
                TpuOpCode::ReduceSumF32,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_scatter",
                TpuOpCode::ReduceScatter,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "all_gather",
                TpuOpCode::AllGather,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "all_reduce",
                TpuOpCode::AllReduceSumF32,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "custom_call",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            ("xor_u8", TpuOpCode::XorBytes, ElementType::U8, 2, &[16]),
            (
                "bitcast_convert",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            ("convert", TpuOpCode::Convert, ElementType::F32, 1, &[4]),
            (
                "transpose",
                TpuOpCode::Transpose,
                ElementType::F32,
                1,
                &[2, 2],
            ),
            (
                "broadcast_in_dim",
                TpuOpCode::BroadcastInDim,
                ElementType::F32,
                1,
                &[4],
            ),
            ("slice", TpuOpCode::Slice, ElementType::F32, 1, &[2]),
            (
                "concatenate",
                TpuOpCode::Concatenate,
                ElementType::F32,
                2,
                &[4],
            ),
            ("constant", TpuOpCode::Constant, ElementType::F32, 0, &[4]),
            ("reshape", TpuOpCode::Reshape, ElementType::F32, 1, &[4]),
            ("and", TpuOpCode::And, ElementType::S32, 2, &[4]),
            ("or", TpuOpCode::Or, ElementType::S32, 2, &[4]),
            ("add", TpuOpCode::AddF32, ElementType::F32, 2, &[4]),
        ];
        for &(pat, op, et, arity, dims) in table {
            if lower.contains(pat) {
                return (op, et, arity, dims.to_vec());
            }
        }
        (TpuOpCode::Identity, ElementType::F32, 1, vec![4])
    }

    /// Parses textual HLO or `StableHLO` MLIR into primary [`TpuOpCode`], parameter [`Shape`]s,
    /// output [`Shape`]s, topological [`TpuInstructionBundle`] SSA DAG, and output slot indices.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 lossy program text or HLO instruction opcode.
    #[allow(
        for_loops_over_fallibles,
        clippy::cognitive_complexity,
        clippy::too_many_lines
    )]
    pub(super) fn parse_textual_program(
        text: &str,
    ) -> (
        TpuOpCode,
        Vec<Shape>,
        Vec<Shape>,
        Vec<TpuInstructionBundle>,
        Vec<usize>,
    ) {
        let (mut primary_op, mut elem_type, default_inputs, mut dims) =
            Self::parse_opcode_mnemonic(text);

        let mut explicit_param_shapes = Vec::new();
        let mut explicit_output_shapes = Vec::new();
        let mut ssa_slots = std::collections::BTreeMap::new();
        let mut slot_shapes: Vec<Shape> = Vec::new();
        let mut next_slot = 0_usize;

        // Extract `func.func @main(...) -> ...` signature shapes and SSA parameter names when present
        for (_, after_func) in text.split_once("func.func") {
            for (_, after_open) in after_func.split_once('(') {
                for (params_str, after_params) in after_open.split_once(')') {
                    for param_decl in params_str.split(',') {
                        for (var_part, type_part) in param_decl.split_once(':') {
                            for sh in Self::parse_tensor_type(type_part.trim()) {
                                elem_type = sh.element_type;
                                dims.clone_from(&sh.dims);
                                ssa_slots.insert(String::from(var_part.trim()), next_slot);
                                next_slot += 1;
                                slot_shapes.push(sh.clone());
                                explicit_param_shapes.push(sh);
                            }
                        }
                    }
                    for (_, after_arrow) in after_params.split_once("->") {
                        let ret_header = after_arrow.split('{').next().unwrap_or(after_arrow);
                        for segment in ret_header.split("tensor<").skip(1) {
                            let candidate = format!("tensor<{segment}");
                            for sh in Self::parse_tensor_type(&candidate) {
                                explicit_output_shapes.push(sh);
                            }
                        }
                    }
                }
            }
        }

        if explicit_param_shapes.is_empty() {
            for segment in text.split("tensor<").skip(1) {
                let candidate = format!("tensor<{segment}");
                if let Some(sh) = Self::parse_tensor_type(&candidate) {
                    elem_type = sh.element_type;
                    dims = sh.dims;
                    break;
                }
            }
        }

        let mut bundles = Vec::new();
        let mut output_slots = Vec::new();
        for line in text.lines() {
            let trimmed_line = line.trim();
            if trimmed_line.starts_with("return ") || trimmed_line.starts_with("func.return ") {
                let before_colon = trimmed_line.split(':').next().unwrap_or(trimmed_line);
                for tok in before_colon.split(|c: char| c.is_whitespace() || c == ',') {
                    if let Some(&slot) = ssa_slots.get(tok.trim()) {
                        output_slots.push(slot);
                    }
                }
                continue;
            }
            if !trimmed_line.contains('=') || trimmed_line.starts_with("HloModule") {
                continue;
            }
            let (lhs, rhs) = trimmed_line.split_once('=').unwrap_or(("", trimmed_line));
            let lhs_var = String::from(lhs.split_whitespace().last().unwrap_or("").trim());
            if rhs.contains("parameter(") {
                for tok in rhs.split_whitespace() {
                    if let Some(sh) = Self::parse_hlo_shape_token(tok) {
                        elem_type = sh.element_type;
                        dims.clone_from(&sh.dims);
                        ssa_slots.insert(lhs_var.clone(), next_slot);
                        next_slot += 1;
                        slot_shapes.push(sh.clone());
                        explicit_param_shapes.push(sh);
                    }
                }
                continue;
            }
            let (line_op, _, line_arity, _) = Self::parse_opcode_mnemonic(rhs);
            primary_op = line_op;

            let mut op_slots = Vec::new();
            let rhs_before_colon = rhs.split(':').next().unwrap_or(rhs);
            for tok in rhs_before_colon
                .split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')')
            {
                let t = tok.trim();
                if !t.is_empty() {
                    if let Some(&slot) = ssa_slots.get(t) {
                        op_slots.push(slot);
                    }
                }
            }
            if op_slots.is_empty() {
                op_slots = (0..line_arity).collect();
            }
            let in_shapes: Vec<Shape> = op_slots
                .iter()
                .filter_map(|&s| slot_shapes.get(s).cloned())
                .collect();
            let attrs = Self::parse_instruction_attrs(rhs, &in_shapes, elem_type);
            let default_bundle_shape = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false; dims.len()],
                layout: MemoryLayout::default_tpu_tiled(dims.len()),
                dims: dims.clone(),
            };
            let bundle_out_shape = attrs.output_shape.clone().unwrap_or(default_bundle_shape);
            let out_slot = next_slot;
            ssa_slots.insert(lhs_var, out_slot);
            next_slot += 1;
            slot_shapes.push(bundle_out_shape);
            let pc = u32::try_from(bundles.len()).unwrap_or(0);
            bundles.push(TpuInstructionBundle {
                pc_offset: pc,
                opcode: line_op,
                operand_slots: op_slots,
                output_slot: out_slot,
                sync_flag_id: pc + 1,
                attrs,
            });
        }

        if explicit_param_shapes.is_empty() {
            for _ in 0..default_inputs {
                explicit_param_shapes.push(Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                });
            }
        }
        if explicit_output_shapes.is_empty() {
            let fallback_out = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false; dims.len()],
                layout: MemoryLayout::default_tpu_tiled(dims.len()),
                dims,
            };
            let inferred_out = bundles
                .last()
                .and_then(|b| b.attrs.output_shape.clone())
                .unwrap_or(fallback_out);
            explicit_output_shapes.push(inferred_out);
        }

        if bundles.is_empty() {
            let operand_slots: Vec<usize> = (0..explicit_param_shapes.len()).collect();
            let out_slot = explicit_param_shapes.len();
            bundles.push(TpuInstructionBundle {
                pc_offset: 0,
                opcode: primary_op,
                operand_slots,
                output_slot: out_slot,
                sync_flag_id: 1,
                attrs: InstructionAttrs {
                    input_shapes: explicit_param_shapes.clone(),
                    output_shape: explicit_output_shapes.first().cloned(),
                    ..InstructionAttrs::default()
                },
            });
        }
        if output_slots.is_empty() {
            let last_slot = bundles
                .last()
                .map_or(explicit_param_shapes.len(), |b| b.output_slot);
            output_slots.push(last_slot);
        }
        (
            primary_op,
            explicit_param_shapes,
            explicit_output_shapes,
            bundles,
            output_slots,
        )
    }
}
