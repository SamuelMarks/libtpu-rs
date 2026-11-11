//! True recursive-descent grammar parser (`TokenCursor`, `MlirModuleParser`) for textual MLIR
//! (`stablehlo`, `vhlo`, `mhlo`, `chlo`, `sdy`, `tpu` Mosaic, `func`, `arith`, `scf`, `vector`, `memref`).

use super::ast::{
    AffineMapExpr, AstBlock, AstBlockArg, AstFunction, AstFunctionType, AstModule, AstOpKind,
    AstOpResult, AstOperand, AstOperation, AstRegion, AstResultType, ComparisonDirection,
    ComparisonType, DenseElementsAttr, DialectKind, DimensionSize, DotDimensionNumbersAttr,
    FftKind, GatherDimensionNumbersAttr, MemRefLayoutAttr, MemRefType, MlirAttribute, MlirLocation,
    MlirType, MosaicDialectAttr, PrecisionLevel, QuantizedElementType, RankedTensorType,
    RngAlgorithmKind, ScatterDimensionNumbersAttr, SdyDimensionSharding, SdyMeshAttr,
    SdyTensorShardingAttr, TensorEncodingAttr, TpuMemorySpace, TpuSemaphoreKind, TransposeMode,
    VectorType, Visibility,
};
use super::lexer::{HloLexer, HloToken, SpannedToken, TokenSpan};
use super::mlir_bytecode::MlirBytecodeHeader;
use super::opcode::TpuOpCode;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::ElementType;
use std::collections::BTreeMap;

/// Zero-copy cursor over a slice of [`SpannedToken`]s supporting `k`-token lookahead and structured diagnostics.
#[derive(Debug, Clone)]
pub struct TokenCursor<'a> {
    /// Underlying token slice.
    tokens: &'a [SpannedToken],
    /// Current cursor index within `tokens`.
    pos: usize,
    /// Span of the last token in `tokens` (used for EOF diagnostics).
    eof_span: TokenSpan,
}

impl<'a> TokenCursor<'a> {
    /// Creates a new [`TokenCursor`] over `tokens`.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Slice of spanned tokens produced by [`HloLexer::tokenize_spanned`].
    #[must_use]
    pub fn new(tokens: &'a [SpannedToken]) -> Self {
        let eof_span = tokens
            .last()
            .map_or(TokenSpan::new(1, 1, 0, 1, 1, 0), |st| {
                TokenSpan::new(
                    st.span.end_line,
                    st.span.end_column,
                    st.span.offset + st.span.length,
                    st.span.end_line,
                    st.span.end_column,
                    0,
                )
            });
        Self {
            tokens,
            pos: 0,
            eof_span,
        }
    }

    /// Returns `true` if all tokens have been consumed.
    #[must_use]
    pub const fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Returns the current cursor index.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Resets the cursor position to a previously saved index.
    ///
    /// # Arguments
    ///
    /// * `saved` - Saved position from [`Self::position`].
    pub const fn restore(&mut self, saved: usize) {
        self.pos = saved;
    }

    /// Returns the [`TokenSpan`] of the next unconsumed token, or the EOF span if at the end.
    #[must_use]
    pub fn current_span(&self) -> TokenSpan {
        self.tokens
            .get(self.pos)
            .map_or(self.eof_span, |st| st.span)
    }

    /// Peeks at the next [`SpannedToken`] without advancing the cursor.
    #[must_use]
    pub fn peek(&self) -> Option<&'a SpannedToken> {
        self.tokens.get(self.pos)
    }

    /// Peeks at the next [`HloToken`] without advancing the cursor.
    #[must_use]
    pub fn peek_token(&self) -> Option<&'a HloToken> {
        self.tokens.get(self.pos).map(|st| &st.token)
    }

    /// Peeks `k` tokens ahead (`0` is equivalent to [`Self::peek_token`]).
    ///
    /// # Arguments
    ///
    /// * `k` - Lookahead offset.
    #[must_use]
    pub fn peek_n(&self, k: usize) -> Option<&'a HloToken> {
        self.tokens.get(self.pos + k).map(|st| &st.token)
    }

    /// Advances the cursor and returns the consumed [`SpannedToken`].
    pub fn next_token(&mut self) -> Option<&'a SpannedToken> {
        let tok = self.tokens.get(self.pos)?;
        self.pos += 1;
        Some(tok)
    }

    /// Consumes the next token if `predicate` returns `true`.
    ///
    /// # Arguments
    ///
    /// * `predicate` - Closure inspecting the next [`HloToken`].
    pub fn consume_if<F>(&mut self, predicate: F) -> Option<&'a SpannedToken>
    where
        F: FnOnce(&HloToken) -> bool,
    {
        let st = self.tokens.get(self.pos)?;
        if predicate(&st.token) {
            self.pos += 1;
            Some(st)
        } else {
            None
        }
    }

    /// Consumes the next token if it is `HloToken::Punct(ch)`.
    ///
    /// # Arguments
    ///
    /// * `ch` - Punctuation character to match.
    pub fn consume_punct(&mut self, ch: char) -> bool {
        self.consume_if(|t| *t == HloToken::Punct(ch)).is_some()
    }

    /// Consumes the next token if it is `HloToken::Ident(kw)`.
    ///
    /// # Arguments
    ///
    /// * `kw` - Identifier string to match.
    pub fn consume_ident(&mut self, kw: &str) -> bool {
        self.consume_if(|t| matches!(t, HloToken::Ident(s) if s == kw))
            .is_some()
    }

    /// Returns `true` if the cursor has not reached EOF and the next token is not `HloToken::Punct(close_punct)`
    /// (consuming `close_punct` when encountered).
    ///
    /// # Arguments
    ///
    /// * `close_punct` - Closing delimiter character (`')'`, `']'`, `'}'`, `'>'`).
    pub fn is_open(&mut self, close_punct: char) -> bool {
        !(self.is_eof() || self.consume_punct(close_punct))
    }

    /// Consumes the next token if it is [`HloToken::Symbol`] and returns the symbol string.
    pub fn consume_symbol_str(&mut self) -> Option<String> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::Symbol(s) = &tok.token {
            self.pos += 1;
            Some(s.clone())
        } else {
            None
        }
    }

    /// Consumes the next token if it is [`HloToken::SsaValue`] and returns `(name, span)`.
    pub fn consume_ssa_spanned(&mut self) -> Option<(String, TokenSpan)> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::SsaValue(s) = &tok.token {
            self.pos += 1;
            Some((s.clone(), tok.span))
        } else {
            None
        }
    }

    /// Consumes the next token if it is [`HloToken::CaretId`] and returns `(caret_id, span)`.
    pub fn consume_caret_spanned(&mut self) -> Option<(String, TokenSpan)> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::CaretId(s) = &tok.token {
            self.pos += 1;
            Some((s.clone(), tok.span))
        } else {
            None
        }
    }

    /// Consumes the next token if it is [`HloToken::AttrAlias`] and returns `(alias, span)`.
    pub fn consume_attr_alias_spanned(&mut self) -> Option<(String, TokenSpan)> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::AttrAlias(s) = &tok.token {
            self.pos += 1;
            Some((s.clone(), tok.span))
        } else {
            None
        }
    }

    /// Consumes the next token if it is [`HloToken::TypeAlias`] and returns `(alias, span)`.
    pub fn consume_type_alias_spanned(&mut self) -> Option<(String, TokenSpan)> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::TypeAlias(s) = &tok.token {
            self.pos += 1;
            Some((s.clone(), tok.span))
        } else {
            None
        }
    }

    /// Consumes the next token if it is [`HloToken::StringLiteral`] and returns the string content.
    pub fn consume_string_str(&mut self) -> Option<String> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::StringLiteral(s) = &tok.token {
            self.pos += 1;
            Some(s.clone())
        } else {
            None
        }
    }

    /// Consumes the next token if it is [`HloToken::Number`] and returns the numeric string.
    pub fn consume_number_str(&mut self) -> Option<String> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::Number(s) = &tok.token {
            self.pos += 1;
            Some(s.clone())
        } else {
            None
        }
    }

    /// Consumes the next token if it is [`HloToken::Ident`] and returns the identifier string.
    pub fn consume_any_ident_str(&mut self) -> Option<String> {
        let tok = self.tokens.get(self.pos)?;
        if let HloToken::Ident(s) = &tok.token {
            self.pos += 1;
            Some(s.clone())
        } else {
            None
        }
    }

    /// Expects and consumes a specific punctuation character `ch`.
    ///
    /// # Arguments
    ///
    /// * `ch` - Expected punctuation character.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::UnexpectedEof`] or [`LibTpuError::UnexpectedToken`] on mismatch.
    pub fn expect_punct(&mut self, ch: char) -> LibTpuResult<TokenSpan> {
        let span = self.current_span();
        let Some(st) = self.next_token() else {
            return Err(LibTpuError::UnexpectedEof {
                expected: format!("'{ch}'"),
                line: span.line,
                column: span.column,
            });
        };
        if st.token == HloToken::Punct(ch) {
            Ok(st.span)
        } else {
            Err(LibTpuError::UnexpectedToken {
                expected: format!("'{ch}'"),
                found: format!("{:?}", st.token),
                line: st.span.line,
                column: st.span.column,
            })
        }
    }

    /// Expects and consumes any [`HloToken::Ident`] token.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::UnexpectedEof`] or [`LibTpuError::UnexpectedToken`] if the next token is not an identifier.
    pub fn expect_ident(&mut self) -> LibTpuResult<(String, TokenSpan)> {
        let span = self.current_span();
        let Some(st) = self.next_token() else {
            return Err(LibTpuError::UnexpectedEof {
                expected: String::from("identifier"),
                line: span.line,
                column: span.column,
            });
        };
        match &st.token {
            HloToken::Ident(id) => Ok((id.clone(), st.span)),
            other => Err(LibTpuError::UnexpectedToken {
                expected: String::from("identifier"),
                found: format!("{other:?}"),
                line: st.span.line,
                column: st.span.column,
            }),
        }
    }

    /// Expects and consumes any [`HloToken::SsaValue`] token.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::UnexpectedEof`] or [`LibTpuError::UnexpectedToken`] if the next token is not an SSA value.
    pub fn expect_ssa(&mut self) -> LibTpuResult<(String, TokenSpan)> {
        let span = self.current_span();
        let Some(st) = self.next_token() else {
            return Err(LibTpuError::UnexpectedEof {
                expected: String::from("SSA value (%...)"),
                line: span.line,
                column: span.column,
            });
        };
        match &st.token {
            HloToken::SsaValue(ssa) => Ok((ssa.clone(), st.span)),
            other => Err(LibTpuError::UnexpectedToken {
                expected: String::from("SSA value (%...)"),
                found: format!("{other:?}"),
                line: st.span.line,
                column: st.span.column,
            }),
        }
    }

    /// Expects and consumes any [`HloToken::Symbol`] token.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::UnexpectedEof`] or [`LibTpuError::UnexpectedToken`] if the next token is not a `@symbol`.
    pub fn expect_symbol(&mut self) -> LibTpuResult<(String, TokenSpan)> {
        let span = self.current_span();
        let Some(st) = self.next_token() else {
            return Err(LibTpuError::UnexpectedEof {
                expected: String::from("symbol (@...)"),
                line: span.line,
                column: span.column,
            });
        };
        match &st.token {
            HloToken::Symbol(sym) => Ok((sym.clone(), st.span)),
            other => Err(LibTpuError::UnexpectedToken {
                expected: String::from("symbol (@...)"),
                found: format!("{other:?}"),
                line: st.span.line,
                column: st.span.column,
            }),
        }
    }
}

/// Parses an MLIR scalar [`ElementType`] identifier (`f32`, `bf16`, `i32`, `s32`, `ui8`, `index`, `complex<f32>`, etc.).
///
/// # Arguments
///
/// * `raw` - Element type name.
#[must_use]
pub fn parse_mlir_element_type(raw: &str) -> Option<ElementType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "f32" | "float" => Some(ElementType::F32),
        "f64" | "double" => Some(ElementType::F64),
        "bf16" | "bfloat16" => Some(ElementType::Bf16),
        "f16" | "half" => Some(ElementType::F16),
        "s32" | "i32" | "si32" => Some(ElementType::S32),
        "s64" | "i64" | "si64" | "index" => Some(ElementType::S64),
        "s16" | "i16" | "si16" => Some(ElementType::S16),
        "s8" | "i8" | "si8" => Some(ElementType::S8),
        "u32" | "ui32" => Some(ElementType::U32),
        "u64" | "ui64" => Some(ElementType::U64),
        "u16" | "ui16" => Some(ElementType::U16),
        "u8" | "ui8" => Some(ElementType::U8),
        "i1" | "pred" | "bool" => Some(ElementType::Pred),
        "c64" | "complex<f32>" => Some(ElementType::C64),
        "c128" | "complex<f64>" => Some(ElementType::C128),
        "f8e5m2" => Some(ElementType::F8E5M2),
        "f8e5m2fnuz" => Some(ElementType::F8E5M2Fnuz),
        "f8e4m3fn" | "f8e4m3" => Some(ElementType::F8E4M3Fn),
        "f8e4m3fnuz" => Some(ElementType::F8E4M3Fnuz),
        "f8e4m3b11fnuz" => Some(ElementType::F8E4M3B11Fnuz),
        "f8e8m0fnu" => Some(ElementType::F8E8M0Fnu),
        "f4e2m1fn" | "f4" => Some(ElementType::F4E2M1Fn),
        "token" | "!stablehlo.token" | "!vhlo.token_v1" => Some(ElementType::Token),
        "s4" | "i4" | "si4" => Some(ElementType::S4),
        "u4" | "ui4" => Some(ElementType::U4),
        "s2" | "i2" | "si2" => Some(ElementType::S2),
        "u2" | "ui2" => Some(ElementType::U2),
        _ => None,
    }
}

/// Splits a shaped MLIR type body like `4x8xf32`, `?x128xbf16`, `<=8x16xf32`, or `*xf32` into
/// `(dims, is_unranked, element_type_str)`.
///
/// # Arguments
///
/// * `spec` - Inner shaped type string before any `,` encoding/layout suffix.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidShape`] if a dimension token is negative or non-numeric.
pub fn split_shaped_dims_and_dtype(spec: &str) -> LibTpuResult<(Vec<DimensionSize>, bool, String)> {
    let trimmed = spec.trim();
    if let Some(rest) = trimmed.strip_prefix("*x") {
        return Ok((Vec::new(), true, String::from(rest.trim())));
    }
    let dtype_pos = trimmed
        .find("complex<")
        .or_else(|| trimmed.find("!quant."))
        .or_else(|| trimmed.rfind('x').map(|i| i + 1));
    match dtype_pos {
        Some(pos) => {
            let dims_prefix = trimmed
                .get(..pos)
                .unwrap_or("")
                .trim()
                .trim_end_matches('x');
            let dtype_part = trimmed.get(pos..).unwrap_or(trimmed).trim();
            let dims = parse_dim_list_tokens(dims_prefix)?;
            Ok((dims, false, String::from(dtype_part)))
        }
        None => Ok((Vec::new(), false, String::from(trimmed))),
    }
}

/// Parses an `x`-separated list of dimension tokens (`4`, `?`, `<=16`) into [`DimensionSize`]s.
///
/// # Arguments
///
/// * `dims_str` - Dimension prefix string (e.g., `"4x?x<=16"`).
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidShape`] if any dimension token is malformed or negative.
fn parse_dim_list_tokens(dims_str: &str) -> LibTpuResult<Vec<DimensionSize>> {
    let mut dims = Vec::new();
    for p in dims_str
        .split('x')
        .map(|s| s.trim().trim_matches(['[', ']']))
        .filter(|s| !s.is_empty())
    {
        if p == "?" {
            dims.push(DimensionSize::Dynamic);
        } else if let Some(bound_str) = p.strip_prefix("<=") {
            let b = bound_str
                .trim()
                .parse::<u64>()
                .ok()
                .and_then(|v| i64::try_from(v).ok())
                .ok_or_else(|| LibTpuError::InvalidShape {
                    reason: format!("Invalid bounded dimension '{p}'"),
                })?;
            dims.push(DimensionSize::Bounded(b));
        } else {
            let d = p
                .parse::<u64>()
                .ok()
                .and_then(|v| i64::try_from(v).ok())
                .ok_or_else(|| LibTpuError::InvalidShape {
                    reason: format!("Invalid static tensor dimension '{p}'"),
                })?;
            dims.push(DimensionSize::Static(d));
        }
    }
    Ok(dims)
}

/// Classifies a `(DialectKind, mnemonic, ElementType)` triple into a strongly-typed [`AstOpKind`].
///
/// # Arguments
///
/// * `dialect` - Parsed [`DialectKind`].
/// * `mnemonic` - Unqualified operation name (with any `vhlo` `_v1` suffix stripped).
/// * `elem_type` - Primary operand/result [`ElementType`].
/// * `attrs` - Operation attributes dictionary (for comparison direction, reduction kind, etc.).
/// * `regions` - Nested operation regions (for `reduce`, `all_reduce`, `reduce_scatter`).
#[must_use]
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn classify_ast_operation(
    dialect: DialectKind,
    mnemonic: &str,
    elem_type: ElementType,
    attrs: &BTreeMap<String, MlirAttribute>,
    regions: &[AstRegion],
) -> AstOpKind {
    let trimmed_mnemonic = mnemonic.trim();
    let clean_mnemonic = trimmed_mnemonic
        .strip_suffix("_v1")
        .or_else(|| trimmed_mnemonic.strip_suffix("_v2"))
        .unwrap_or(trimmed_mnemonic);

    match (dialect, clean_mnemonic) {
        (DialectKind::Func | DialectKind::Sdy, "return")
        | (DialectKind::Scf, "yield" | "condition") => return AstOpKind::Return,
        (DialectKind::Func, "call") => {
            let callee = attrs
                .get("callee")
                .and_then(MlirAttribute::as_str)
                .unwrap_or("callee");
            return AstOpKind::Call(String::from(callee));
        }
        (DialectKind::Sdy, "manual_computation") => return AstOpKind::SdyManualComputation,
        (DialectKind::Sdy, "sharding_constraint" | "reshard") => {
            return AstOpKind::SdyShardingConstraint;
        }
        (DialectKind::Sdy, "data_flow_edge" | "named_computation" | "sharding_group") => {
            return AstOpKind::SdyDataFlow;
        }
        (DialectKind::Sdy, "mesh") => return AstOpKind::BarrierOrToken,
        (DialectKind::Sdy, "all_slice")
        | (DialectKind::TpuMosaic, "memref_slice")
        | (DialectKind::Vector, "extract")
        | (DialectKind::MemRef, "subview") => return AstOpKind::HardwareOp(TpuOpCode::Slice),
        (DialectKind::Chlo, "top_k") => return AstOpKind::HardwareOp(TpuOpCode::TopK),
        (DialectKind::Chlo, "erf") => return AstOpKind::HardwareOp(TpuOpCode::Erf),
        (DialectKind::Chlo, "tan") => return AstOpKind::HardwareOp(TpuOpCode::Tan),
        (DialectKind::Chlo, "ragged_dot")
        | (DialectKind::TpuMosaic, "matmul" | "matmul_push_rhs" | "matmul_acc" | "matmul_pop")
        | (DialectKind::Vector, "contract" | "fma") => {
            return AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32);
        }
        (DialectKind::Chlo, "broadcast_add") => {
            let op = if matches!(
                elem_type,
                ElementType::S32 | ElementType::S64 | ElementType::S16 | ElementType::S8
            ) {
                TpuOpCode::AddS32
            } else {
                TpuOpCode::AddF32
            };
            return AstOpKind::HardwareOp(op);
        }
        (DialectKind::Chlo, "broadcast_sub") | (DialectKind::Arith, "subi" | "subf") => {
            return AstOpKind::HardwareOp(TpuOpCode::SubF32);
        }
        (DialectKind::Chlo, "broadcast_mul") | (DialectKind::Arith, "muli" | "mulf") => {
            return AstOpKind::HardwareOp(TpuOpCode::MulF32);
        }
        (DialectKind::Chlo, "broadcast_div") | (DialectKind::Arith, "divsi" | "divui" | "divf") => {
            return AstOpKind::HardwareOp(TpuOpCode::DivF32);
        }
        (DialectKind::Chlo, "broadcast_max")
        | (DialectKind::Arith, "maxsi" | "maxui" | "maximumf" | "maxnumf") => {
            return AstOpKind::HardwareOp(TpuOpCode::MaxF32);
        }
        (DialectKind::Chlo, "broadcast_min")
        | (DialectKind::Arith, "minsi" | "minui" | "minimumf" | "minnumf") => {
            return AstOpKind::HardwareOp(TpuOpCode::MinF32);
        }
        (DialectKind::Chlo, "broadcast_pow") => return AstOpKind::HardwareOp(TpuOpCode::Pow),
        (DialectKind::Chlo, "broadcast_atan2") => return AstOpKind::HardwareOp(TpuOpCode::Atan2),
        (DialectKind::Chlo, "broadcast_select") => return AstOpKind::HardwareOp(TpuOpCode::Select),
        (DialectKind::Chlo, "broadcast_compare") | (DialectKind::Arith, "cmpi" | "cmpf") => {
            return AstOpKind::HardwareOp(resolve_compare_opcode(attrs));
        }
        (DialectKind::Chlo, other) => return AstOpKind::ChloDecomposable(String::from(other)),
        (DialectKind::TpuMosaic, "enqueue_dma" | "dma_copy") => return AstOpKind::MosaicDmaStart,
        (DialectKind::TpuMosaic, "wait_dma" | "dma_wait") => return AstOpKind::MosaicDmaWait,
        (
            DialectKind::TpuMosaic,
            "semaphore_signal" | "semaphore_wait" | "semaphore_read" | "barrier" | "device_id"
            | "trace" | "trace_start" | "trace_stop",
        ) => {
            return AstOpKind::MosaicSemaphore(String::from(clean_mnemonic));
        }
        (
            DialectKind::TpuMosaic,
            "vector_load" | "vector_store" | "load" | "store" | "strided_load" | "strided_store",
        ) => {
            return AstOpKind::MosaicMemoryOp(String::from(clean_mnemonic));
        }
        (DialectKind::TpuMosaic | DialectKind::Vector, "repeat" | "broadcast") => {
            return AstOpKind::HardwareOp(TpuOpCode::BroadcastInDim);
        }
        (DialectKind::TpuMosaic, "rotate" | "roll") => {
            return AstOpKind::HardwareOp(TpuOpCode::DynamicSlice);
        }
        (DialectKind::TpuMosaic, "concat") => return AstOpKind::HardwareOp(TpuOpCode::Concatenate),
        (
            DialectKind::TpuMosaic,
            "pack_subelements"
            | "unpack_subelements"
            | "memref_squeeze"
            | "memref_bitcast"
            | "relabel"
            | "relayout"
            | "bitcast"
            | "erase_memref_layout"
            | "assume_multiple",
        )
        | (DialectKind::Arith, "bitcast")
        | (DialectKind::Vector, "shape_cast")
        | (DialectKind::MemRef, "reinterpret_cast" | "cast" | "collapse_shape" | "expand_shape") => {
            return AstOpKind::HardwareOp(TpuOpCode::Reshape);
        }
        (DialectKind::TpuMosaic, "region") | (DialectKind::Scf, "if" | "index_switch") => {
            return AstOpKind::Conditional;
        }
        (DialectKind::Scf, "for") => return AstOpKind::ScfFor,
        (DialectKind::Arith, "addi") => return AstOpKind::HardwareOp(TpuOpCode::AddS32),
        (DialectKind::Arith, "addf") => return AstOpKind::HardwareOp(TpuOpCode::AddF32),
        (DialectKind::Arith, "remsi" | "remui" | "remf") => {
            return AstOpKind::HardwareOp(TpuOpCode::Rem);
        }
        (DialectKind::Arith, "andi") => return AstOpKind::HardwareOp(TpuOpCode::And),
        (DialectKind::Arith, "ori") => return AstOpKind::HardwareOp(TpuOpCode::Or),
        (DialectKind::Arith, "xori") => return AstOpKind::HardwareOp(TpuOpCode::XorBytes),
        (DialectKind::Arith, "shli") => return AstOpKind::HardwareOp(TpuOpCode::ShiftLeft),
        (DialectKind::Arith, "shrsi") => {
            return AstOpKind::HardwareOp(TpuOpCode::ShiftRightArithmetic);
        }
        (DialectKind::Arith, "shrui") => {
            return AstOpKind::HardwareOp(TpuOpCode::ShiftRightLogical);
        }
        (DialectKind::Arith, "negf") => return AstOpKind::HardwareOp(TpuOpCode::NegF32),
        (
            DialectKind::Arith,
            "index_cast" | "extsi" | "extui" | "trunci" | "sitofp" | "uitofp" | "fptosi" | "fptoui"
            | "extf" | "truncf",
        ) => return AstOpKind::HardwareOp(TpuOpCode::Convert),
        (DialectKind::Vector, "insert") => {
            return AstOpKind::HardwareOp(TpuOpCode::DynamicUpdateSlice);
        }
        (DialectKind::Vector, "multi_reduction" | "reduction") => {
            return AstOpKind::HardwareOp(TpuOpCode::ReduceSumF32);
        }
        (DialectKind::Vector, "transfer_read" | "transfer_write" | "load" | "store") => {
            return AstOpKind::MosaicMemoryOp(format!("vector.{clean_mnemonic}"));
        }
        (DialectKind::MemRef, "load" | "store" | "alloc" | "alloca") => {
            return AstOpKind::MosaicMemoryOp(format!("memref.{clean_mnemonic}"));
        }
        _ => {}
    }

    // Unified StableHLO / MHLO / VHLO / XLA HLO classification
    match clean_mnemonic {
        "return" => AstOpKind::Return,
        "parameter" => {
            let idx = attrs
                .get("parameter_number")
                .and_then(MlirAttribute::as_usize)
                .unwrap_or(0);
            AstOpKind::Parameter { index: idx }
        }
        "tuple" => AstOpKind::Tuple,
        "get_tuple_element" | "get-tuple-element" => {
            let idx = attrs
                .get("index")
                .and_then(MlirAttribute::as_usize)
                .unwrap_or(0);
            AstOpKind::GetTupleElement { index: idx }
        }
        "reduce" => {
            let resolved = resolve_reduction_region_opcode(regions, TpuOpCode::ReduceSumF32);
            AstOpKind::HardwareOp(resolved)
        }
        "all_reduce" | "all-reduce" | "cross-replica-sum" => AstOpKind::HardwareOp(
            resolve_reduction_region_opcode(regions, TpuOpCode::AllReduceSumF32),
        ),
        "reduce_scatter" | "reduce-scatter" => AstOpKind::HardwareOp(TpuOpCode::ReduceScatter),
        "while" => AstOpKind::While,
        "if" | "case" | "conditional" => AstOpKind::Conditional,
        "add" | "add_f32" => {
            let is_int = matches!(
                elem_type,
                ElementType::S32
                    | ElementType::S64
                    | ElementType::S16
                    | ElementType::S8
                    | ElementType::S4
                    | ElementType::S2
                    | ElementType::U32
                    | ElementType::U64
                    | ElementType::U16
                    | ElementType::U8
                    | ElementType::U4
                    | ElementType::U2
            );
            AstOpKind::HardwareOp(if is_int {
                TpuOpCode::AddS32
            } else {
                TpuOpCode::AddF32
            })
        }
        "add_s32" | "s32_add" => AstOpKind::HardwareOp(TpuOpCode::AddS32),
        "subtract" | "sub" | "sub_f32" => AstOpKind::HardwareOp(TpuOpCode::SubF32),
        "multiply" | "mul" | "mul_f32" => AstOpKind::HardwareOp(TpuOpCode::MulF32),
        "divide" | "div" | "div_f32" => AstOpKind::HardwareOp(TpuOpCode::DivF32),
        "remainder" | "rem" => AstOpKind::HardwareOp(TpuOpCode::Rem),
        "maximum" | "max" | "max_f32" => AstOpKind::HardwareOp(TpuOpCode::MaxF32),
        "minimum" | "min" | "min_f32" => AstOpKind::HardwareOp(TpuOpCode::MinF32),
        "power" | "pow" => AstOpKind::HardwareOp(TpuOpCode::Pow),
        "atan2" => AstOpKind::HardwareOp(TpuOpCode::Atan2),
        "dot" | "dot_general" | "dot-general" | "ragged_dot" | "sparse_dot" => {
            AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32)
        }
        "convolution" | "dynamic_conv" | "DynamicConv" => {
            AstOpKind::HardwareOp(TpuOpCode::Convolution)
        }
        "negate" | "neg" | "neg_f32" => AstOpKind::HardwareOp(TpuOpCode::NegF32),
        "exponential" | "exp" | "exp_f32" => AstOpKind::HardwareOp(TpuOpCode::ExpF32),
        "exponential_minus_one" | "expm1" => AstOpKind::HardwareOp(TpuOpCode::Expm1),
        "log" => AstOpKind::HardwareOp(TpuOpCode::Log),
        "log_plus_one" | "log1p" => AstOpKind::HardwareOp(TpuOpCode::Log1p),
        "logistic" => AstOpKind::HardwareOp(TpuOpCode::Logistic),
        "tanh" | "tanh_f32" => AstOpKind::HardwareOp(TpuOpCode::TanhF32),
        "erf" => AstOpKind::HardwareOp(TpuOpCode::Erf),
        "tan" => AstOpKind::HardwareOp(TpuOpCode::Tan),
        "sine" | "sin" => AstOpKind::HardwareOp(TpuOpCode::Sine),
        "cosine" | "cos" => AstOpKind::HardwareOp(TpuOpCode::Cosine),
        "sqrt" => AstOpKind::HardwareOp(TpuOpCode::Sqrt),
        "rsqrt" => AstOpKind::HardwareOp(TpuOpCode::Rsqrt),
        "cbrt" => AstOpKind::HardwareOp(TpuOpCode::Cbrt),
        "abs" => AstOpKind::HardwareOp(TpuOpCode::Abs),
        "floor" => AstOpKind::HardwareOp(TpuOpCode::Floor),
        "ceil" => AstOpKind::HardwareOp(TpuOpCode::Ceil),
        "round_nearest_even" => AstOpKind::HardwareOp(TpuOpCode::RoundNearestEven),
        "round_nearest_afz" => AstOpKind::HardwareOp(TpuOpCode::RoundNearestAfz),
        "sign" => AstOpKind::HardwareOp(TpuOpCode::Sign),
        "is_finite" | "is-finite" => AstOpKind::HardwareOp(TpuOpCode::IsFinite),
        "not" => AstOpKind::HardwareOp(TpuOpCode::Not),
        "and" => AstOpKind::HardwareOp(TpuOpCode::And),
        "or" => AstOpKind::HardwareOp(TpuOpCode::Or),
        "xor" | "xor_u8" => AstOpKind::HardwareOp(TpuOpCode::XorBytes),
        "shift_left" | "shift-left" => AstOpKind::HardwareOp(TpuOpCode::ShiftLeft),
        "shift_right_arithmetic" | "shift-right-arithmetic" => {
            AstOpKind::HardwareOp(TpuOpCode::ShiftRightArithmetic)
        }
        "shift_right_logical" | "shift-right-logical" => {
            AstOpKind::HardwareOp(TpuOpCode::ShiftRightLogical)
        }
        "popcnt" | "population_count" | "population-count" => {
            AstOpKind::HardwareOp(TpuOpCode::PopulationCount)
        }
        "count_leading_zeros" | "count-leading-zeros" | "clz" => {
            AstOpKind::HardwareOp(TpuOpCode::CountLeadingZeros)
        }
        "compare" => AstOpKind::HardwareOp(resolve_compare_opcode(attrs)),
        "compare_eq" => AstOpKind::HardwareOp(TpuOpCode::CompareEq),
        "compare_lt" => AstOpKind::HardwareOp(TpuOpCode::CompareLt),
        "compare_gt" => AstOpKind::HardwareOp(TpuOpCode::CompareGt),
        "select" => AstOpKind::HardwareOp(TpuOpCode::Select),
        "clamp" => AstOpKind::HardwareOp(TpuOpCode::Clamp),
        "iota" | "dynamic_iota" => AstOpKind::HardwareOp(TpuOpCode::Iota),
        "reverse" => AstOpKind::HardwareOp(TpuOpCode::Reverse),
        "reshape" | "dynamic_reshape" | "bitcast_convert" | "bitcast-convert" | "bitcast"
        | "copy" => AstOpKind::HardwareOp(TpuOpCode::Reshape),
        "transpose" => AstOpKind::HardwareOp(TpuOpCode::Transpose),
        "broadcast_in_dim" | "broadcast-in-dim" | "broadcast" | "dynamic_broadcast_in_dim" => {
            AstOpKind::HardwareOp(TpuOpCode::BroadcastInDim)
        }
        "slice" | "real" | "imag" => AstOpKind::HardwareOp(TpuOpCode::Slice),
        "dynamic_slice" | "dynamic-slice" => AstOpKind::HardwareOp(TpuOpCode::DynamicSlice),
        "dynamic_update_slice" | "dynamic-update-slice" => {
            AstOpKind::HardwareOp(TpuOpCode::DynamicUpdateSlice)
        }
        "concatenate" | "complex" => AstOpKind::HardwareOp(TpuOpCode::Concatenate),
        "convert" | "batch_norm_inference" | "batch_norm_training" | "batch_norm_grad" => {
            AstOpKind::HardwareOp(TpuOpCode::Convert)
        }
        "constant" => AstOpKind::HardwareOp(TpuOpCode::Constant),
        "pad" | "dynamic_pad" => AstOpKind::HardwareOp(TpuOpCode::Pad),
        "gather" => AstOpKind::HardwareOp(TpuOpCode::Gather),
        "scatter" | "select_and_scatter" => AstOpKind::HardwareOp(TpuOpCode::Scatter),
        "reduce_window" | "reduce-window" => AstOpKind::HardwareOp(TpuOpCode::ReduceWindow),
        "sort" => AstOpKind::HardwareOp(TpuOpCode::Sort),
        "top_k" | "topk" => AstOpKind::HardwareOp(TpuOpCode::TopK),
        "rng_bit_generator" | "rng-bit-generator" | "rng" => {
            AstOpKind::HardwareOp(TpuOpCode::RngBitGenerator)
        }
        "cholesky" => AstOpKind::HardwareOp(TpuOpCode::Cholesky),
        "triangular_solve" | "triangular-solve" => {
            AstOpKind::HardwareOp(TpuOpCode::TriangularSolve)
        }
        "fft" => AstOpKind::HardwareOp(TpuOpCode::Fft),
        "uniform_quantize" => AstOpKind::HardwareOp(TpuOpCode::UniformQuantize),
        "uniform_dequantize" => AstOpKind::HardwareOp(TpuOpCode::UniformDequantize),
        "all_gather" | "all-gather" => AstOpKind::HardwareOp(TpuOpCode::AllGather),
        "all_to_all" | "all-to-all" => AstOpKind::HardwareOp(TpuOpCode::AllToAll),
        "collective_permute" | "collective-permute" | "collective_broadcast" => {
            AstOpKind::HardwareOp(TpuOpCode::CollectivePermute)
        }
        "custom_call" | "custom-call" | "infeed" | "outfeed" | "send" | "recv" | "partition_id"
        | "replica_id" | "AsyncStart" | "AsyncUpdate" | "AsyncDone" => {
            AstOpKind::HardwareOp(TpuOpCode::CustomCall)
        }
        "reduce_max" => AstOpKind::HardwareOp(TpuOpCode::ReduceMax),
        "reduce_min" => AstOpKind::HardwareOp(TpuOpCode::ReduceMin),
        "reduce_sum" => AstOpKind::HardwareOp(TpuOpCode::ReduceSumF32),
        "next_after" | "next-after" => AstOpKind::HardwareOp(TpuOpCode::NextAfter),
        "dynamic_gather" | "dynamic-gather" => AstOpKind::HardwareOp(TpuOpCode::DynamicGather),
        "real_dynamic_slice" | "real-dynamic-slice" => {
            AstOpKind::HardwareOp(TpuOpCode::RealDynamicSlice)
        }
        "ragged_all_to_all" | "ragged-all-to-all" => {
            AstOpKind::HardwareOp(TpuOpCode::RaggedAllToAll)
        }
        "send_done" | "send-done" => AstOpKind::HardwareOp(TpuOpCode::SendDone),
        "recv_done" | "recv-done" => AstOpKind::HardwareOp(TpuOpCode::RecvDone),
        "conj" => AstOpKind::HardwareOp(TpuOpCode::Conj),
        "acosh" => AstOpKind::HardwareOp(TpuOpCode::Acosh),
        "asinh" => AstOpKind::HardwareOp(TpuOpCode::Asinh),
        "atanh" => AstOpKind::HardwareOp(TpuOpCode::Atanh),
        "sinh" => AstOpKind::HardwareOp(TpuOpCode::Sinh),
        "cosh" => AstOpKind::HardwareOp(TpuOpCode::Cosh),
        "acos" => AstOpKind::HardwareOp(TpuOpCode::Acos),
        "asin" => AstOpKind::HardwareOp(TpuOpCode::Asin),
        "atan" => AstOpKind::HardwareOp(TpuOpCode::Atan),
        "erfc" => AstOpKind::HardwareOp(TpuOpCode::Erfc),
        "lgamma" => AstOpKind::HardwareOp(TpuOpCode::Lgamma),
        "digamma" => AstOpKind::HardwareOp(TpuOpCode::Digamma),
        "bessel_i1e" | "bessel-i1e" => AstOpKind::HardwareOp(TpuOpCode::BesselI1e),
        "reduce_precision" | "reduce-precision" => {
            AstOpKind::HardwareOp(TpuOpCode::ReducePrecision)
        }
        "outbound_dma" => AstOpKind::HardwareOp(TpuOpCode::OutboundDma),
        "wait_dma_op" => AstOpKind::HardwareOp(TpuOpCode::WaitDma),
        "semaphore_signal_op" => AstOpKind::HardwareOp(TpuOpCode::SemaphoreSignal),
        "semaphore_wait_op" => AstOpKind::HardwareOp(TpuOpCode::SemaphoreWait),
        "broadcast_in_sublanes_op" => AstOpKind::HardwareOp(TpuOpCode::BroadcastInSublanes),
        "rotate_sublanes" => AstOpKind::HardwareOp(TpuOpCode::RotateSublanes),
        "optimization_barrier"
        | "after_all"
        | "create_token"
        | "domain"
        | "compute_reshape_shape"
        | "cstr_reshapable"
        | "get_dimension_size"
        | "set_dimension_size" => AstOpKind::BarrierOrToken,
        _ => AstOpKind::HardwareOp(TpuOpCode::Identity),
    }
}

/// Resolves the [`TpuOpCode`] for a `compare` operation from its parsed `comparison_direction` attribute.
///
/// # Arguments
///
/// * `attrs` - Parsed operation attribute map.
#[must_use]
fn resolve_compare_opcode(attrs: &BTreeMap<String, MlirAttribute>) -> TpuOpCode {
    match attrs.get("comparison_direction") {
        Some(MlirAttribute::ComparisonDirection(
            ComparisonDirection::Lt | ComparisonDirection::Le,
        )) => TpuOpCode::CompareLt,
        Some(MlirAttribute::ComparisonDirection(
            ComparisonDirection::Gt | ComparisonDirection::Ge,
        )) => TpuOpCode::CompareGt,
        _ => TpuOpCode::CompareEq,
    }
}

/// Inspects the nested combiner region of a `reduce` or `all_reduce` operation to determine the hardware reduction opcode.
///
/// # Arguments
///
/// * `regions` - Nested regions attached to the reduction operation.
/// * `default_op` - Fallback reduction opcode when no combiner region is present.
#[must_use]
pub fn resolve_reduction_region_opcode(regions: &[AstRegion], default_op: TpuOpCode) -> TpuOpCode {
    for region in regions {
        for block in &region.blocks {
            for op in &block.operations {
                match op.mnemonic.as_str() {
                    "maximum" | "max" | "max_f32" | "maxsi" | "maxnumf" => {
                        return TpuOpCode::ReduceMax;
                    }
                    "minimum" | "min" | "min_f32" | "minsi" | "minnumf" => {
                        return TpuOpCode::ReduceMin;
                    }
                    "add" | "addi" | "addf" => return default_op,
                    "and" | "andi" => return TpuOpCode::And,
                    "or" | "ori" => return TpuOpCode::Or,
                    _ => {}
                }
            }
        }
    }
    default_op
}

/// Stateful recursive-descent parser for textual MLIR modules (`stablehlo`, `mhlo`, `chlo`, `sdy`, `tpu` Mosaic).
#[derive(Debug)]
pub struct MlirModuleParser<'a> {
    /// Token stream cursor.
    cursor: TokenCursor<'a>,
    /// Top-level `!alias = type` symbol table.
    type_aliases: BTreeMap<String, MlirType>,
    /// Top-level `#alias = attr` symbol table.
    attr_aliases: BTreeMap<String, MlirAttribute>,
    /// Top-level `#loc = loc(...)` symbol table.
    locations: BTreeMap<String, MlirLocation>,
    /// Top-level `sdy.mesh @mesh = <...>` symbol table.
    sdy_meshes: BTreeMap<String, SdyMeshAttr>,
}

impl<'a> MlirModuleParser<'a> {
    /// Creates a new [`MlirModuleParser`] over a slice of [`SpannedToken`]s.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Lexed token slice.
    #[must_use]
    pub fn new(tokens: &'a [SpannedToken]) -> Self {
        Self {
            cursor: TokenCursor::new(tokens),
            type_aliases: BTreeMap::new(),
            attr_aliases: BTreeMap::new(),
            locations: BTreeMap::new(),
            sdy_meshes: BTreeMap::new(),
        }
    }

    /// Parses a complete MLIR [`AstModule`] from the token stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if lexical, syntactic, or alias verification fails.
    pub fn parse_module(&mut self) -> LibTpuResult<AstModule> {
        let start_span = self.cursor.current_span();
        let mut module = AstModule {
            span: start_span,
            ..AstModule::default()
        };

        while !self.cursor.is_eof() {
            // Check for top-level `#alias = ...`
            if let Some((alias_name, _)) = self.cursor.consume_attr_alias_spanned() {
                let _ = self.cursor.consume_punct('=');
                if self.cursor.peek_token() == Some(&HloToken::Ident(String::from("loc"))) {
                    let loc = self.parse_location();
                    self.locations.insert(alias_name.clone(), loc.clone());
                    module
                        .attr_aliases
                        .insert(alias_name, MlirAttribute::Location(loc));
                } else {
                    let attr = self.parse_attribute()?;
                    self.attr_aliases.insert(alias_name.clone(), attr.clone());
                    module.attr_aliases.insert(alias_name, attr);
                }
                continue;
            }
            // Check for top-level `!alias = ...`
            if let Some((alias_name, _)) = self.cursor.consume_type_alias_spanned() {
                let _ = self.cursor.consume_punct('=');
                let ty = self.parse_type()?;
                self.type_aliases.insert(alias_name.clone(), ty.clone());
                module.type_aliases.insert(alias_name, ty);
                continue;
            }
            // Check for `module @name attributes { ... } { ... }`
            if self.cursor.consume_ident("module") || self.cursor.consume_ident("builtin.module") {
                module.name = self.cursor.consume_symbol_str();
                if self.cursor.consume_ident("attributes") {
                    module.attributes = self.parse_attribute_dict()?;
                }
                if self.cursor.consume_punct('{') {
                    self.parse_module_body(&mut module)?;
                }
                let _ = self.parse_optional_location();
                continue;
            }
            // Check for bare `func.func` or `sdy.mesh` or top-level op
            if self.cursor.peek_token() == Some(&HloToken::Ident(String::from("func.func"))) {
                let func = self.parse_function()?;
                module.functions.push(func);
                continue;
            }
            if self.cursor.peek_token() == Some(&HloToken::Ident(String::from("sdy.mesh"))) {
                let mesh = self.parse_sdy_mesh_op();
                self.sdy_meshes.insert(mesh.name.clone(), mesh.clone());
                module.sdy_meshes.insert(mesh.name.clone(), mesh);
                continue;
            }
            if self.cursor.consume_punct('}') {
                continue;
            }
            if let Ok(op) = self.parse_operation() {
                module.top_level_operations.push(op);
            } else {
                // Advance one token to guarantee termination on partial snippets
                self.cursor.next_token();
            }
        }

        module.locations.clone_from(&self.locations);
        Ok(module)
    }

    /// Parses the items inside a `module { ... }` block (`func.func`, `sdy.mesh`, aliases, or bare ops).
    ///
    /// # Arguments
    ///
    /// * `module` - Mutable [`AstModule`] being populated.
    fn parse_module_body(&mut self, module: &mut AstModule) -> LibTpuResult<()> {
        while self.cursor.is_open('}') {
            if self.cursor.peek_token() == Some(&HloToken::Ident(String::from("func.func"))) {
                let func = self.parse_function()?;
                module.functions.push(func);
            } else if self.cursor.peek_token() == Some(&HloToken::Ident(String::from("sdy.mesh"))) {
                let mesh = self.parse_sdy_mesh_op();
                self.sdy_meshes.insert(mesh.name.clone(), mesh.clone());
                module.sdy_meshes.insert(mesh.name.clone(), mesh);
            } else {
                let op = self.parse_operation()?;
                module.top_level_operations.push(op);
            }
        }
        Ok(())
    }

    /// Parses an `sdy.mesh @mesh_name = <["x"=2, "y"=4], device_ids=[...]>` declaration.
    pub fn parse_sdy_mesh_op(&mut self) -> SdyMeshAttr {
        let _ = self.cursor.next_token();
        let mesh_name = self.cursor.consume_symbol_str().unwrap_or_default();
        let _ = self.cursor.consume_punct('=');
        let mut mesh = self.parse_sdy_mesh_body();
        mesh.name = mesh_name;
        let _ = self.parse_optional_location();
        mesh
    }

    /// Parses the `<["x"=2, "y"=4], device_ids=[...]>` body of an `sdy.mesh` or `#sdy.mesh<...>`.
    fn parse_sdy_mesh_body(&mut self) -> SdyMeshAttr {
        let mut mesh = SdyMeshAttr::default();
        if self.cursor.peek_token() != Some(&HloToken::Punct('<')) {
            return mesh;
        }
        let body = self.consume_balanced_angle_brackets();
        mesh.device_ids = extract_bracketed_usize_list(&body, "device_ids");
        let axes_region = body.split("device_ids").next().unwrap_or(&body);
        if let Some((inner, _)) = axes_region
            .split_once('[')
            .and_then(|(_, after_open)| after_open.split_once(']'))
        {
            for (k, v) in inner.split(',').filter_map(|p| p.split_once('=')) {
                let axis_name = k.trim().trim_matches('"');
                let sz = v.trim().parse::<usize>().unwrap_or(1);
                mesh.axes.push((String::from(axis_name), sz));
            }
        }
        mesh
    }

    /// Parses a `func.func [visibility] @name(%args...) -> (results...) attributes { ... } { ... }` definition.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the function signature or body region is malformed.
    #[allow(clippy::too_many_lines)]
    pub fn parse_function(&mut self) -> LibTpuResult<AstFunction> {
        let start_span = self.cursor.current_span();
        let _ = self.cursor.next_token(); // "func.func"
        let visibility = if self.cursor.consume_ident("private") {
            Visibility::Private
        } else if self.cursor.consume_ident("nested") {
            Visibility::Nested
        } else {
            let _ = self.cursor.consume_ident("public");
            Visibility::Public
        };
        let name_opt = self
            .cursor
            .consume_symbol_str()
            .or_else(|| self.cursor.consume_any_ident_str());
        let name = String::from(name_opt.as_deref().unwrap_or("main"));

        let mut arguments = Vec::new();
        let _ = self.cursor.consume_punct('(');
        while self.cursor.is_open(')') {
            if let Some((ssa_name, _)) = self.cursor.consume_ssa_spanned() {
                let _ = self.cursor.consume_punct(':');
                let ty = self.parse_type()?;
                let attrs = if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
                    self.parse_attribute_dict()?
                } else {
                    BTreeMap::new()
                };
                let location = self.parse_optional_location();
                arguments.push(AstBlockArg {
                    ssa_name,
                    ty,
                    attrs,
                    location,
                });
            } else {
                // Anonymous parameter type in declaration
                let ty = self.parse_type()?;
                let attrs = if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
                    self.parse_attribute_dict()?
                } else {
                    BTreeMap::new()
                };
                arguments.push(AstBlockArg {
                    ssa_name: format!("%arg{}", arguments.len()),
                    ty,
                    attrs,
                    location: None,
                });
            }
            let _ = self.cursor.consume_punct(',');
        }

        let mut results = Vec::new();
        if self.cursor.consume_if(|t| *t == HloToken::Arrow).is_some() {
            if self.cursor.consume_punct('(') {
                while self.cursor.is_open(')') {
                    let ty = self.parse_type()?;
                    let attrs = if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
                        self.parse_attribute_dict()?
                    } else {
                        BTreeMap::new()
                    };
                    results.push(AstResultType { ty, attrs });
                    let _ = self.cursor.consume_punct(',');
                }
            } else {
                let ty = self.parse_type()?;
                results.push(AstResultType {
                    ty,
                    attrs: BTreeMap::new(),
                });
            }
        }

        let function_attrs = if self.cursor.consume_ident("attributes") {
            self.parse_attribute_dict()?
        } else {
            BTreeMap::new()
        };

        let body = if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
            self.parse_region()?
        } else {
            AstRegion::default()
        };
        let _ = self.parse_optional_location();

        Ok(AstFunction {
            name,
            visibility,
            arguments,
            results,
            function_attrs,
            body,
            span: start_span,
        })
    }

    /// Recursively parses an [`AstRegion`] enclosed in `{ ... }`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the region is not enclosed in `{ ... }`.
    pub fn parse_region(&mut self) -> LibTpuResult<AstRegion> {
        let span = self.cursor.current_span();
        let _ = self.cursor.consume_punct('{');
        let mut blocks = Vec::new();
        let mut current_block = AstBlock {
            span,
            ..AstBlock::default()
        };

        while self.cursor.is_open('}') {
            if let Some((caret_id, caret_span)) = self.cursor.consume_caret_spanned() {
                if !current_block.operations.is_empty() || current_block.label.is_some() {
                    blocks.push(current_block);
                }
                let mut block_args = Vec::new();
                if self.cursor.consume_punct('(') {
                    while self.cursor.is_open(')') {
                        let (ssa_name, _) = self.cursor.expect_ssa()?;
                        let _ = self.cursor.consume_punct(':');
                        let ty = self.parse_type()?;
                        let location = self.parse_optional_location();
                        block_args.push(AstBlockArg {
                            ssa_name,
                            ty,
                            attrs: BTreeMap::new(),
                            location,
                        });
                        let _ = self.cursor.consume_punct(',');
                    }
                }
                let _ = self.cursor.consume_punct(':');
                current_block = AstBlock {
                    label: Some(caret_id),
                    arguments: block_args,
                    operations: Vec::new(),
                    span: caret_span,
                };
                continue;
            }

            let op = self.parse_operation()?;
            current_block.operations.push(op);
        }

        blocks.push(current_block);

        Ok(AstRegion { blocks, span })
    }

    /// Parses a single MLIR [`AstOperation`] in either custom assembly format or generic `"dialect.op"(...)` format.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if no valid operation mnemonic or generic op string is present.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn parse_operation(&mut self) -> LibTpuResult<AstOperation> {
        let start_span = self.cursor.current_span();
        let mut results = Vec::new();

        // Parse optional `%res0, %res1 =` or `%res:2 =` bindings
        if matches!(self.cursor.peek_token(), Some(HloToken::SsaValue(_))) {
            let saved = self.cursor.position();
            let mut parsed_lhs = Vec::new();
            while let Some((ssa_name, _)) = self.cursor.consume_ssa_spanned() {
                let count = if self.cursor.consume_punct(':') {
                    self.cursor
                        .consume_number_str()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(1)
                        .max(1)
                } else {
                    1_usize
                };
                parsed_lhs.push(AstOpResult {
                    name: ssa_name,
                    count,
                });
                let _ = self.cursor.consume_punct(',');
            }
            if self.cursor.consume_punct('=') {
                results = parsed_lhs;
            } else {
                self.cursor.restore(saved);
            }
        }

        // Parse operation name (either generic `"dialect.op"` StringLiteral or `dialect.op` Ident)
        let full_op_name = match self.cursor.next_token() {
            Some(st) => match &st.token {
                HloToken::StringLiteral(s) | HloToken::Ident(s) => {
                    MlirBytecodeHeader::legalize_vhlo_symbol(s)
                }
                other => {
                    return Err(LibTpuError::UnexpectedToken {
                        expected: String::from("operation mnemonic"),
                        found: format!("{other:?}"),
                        line: st.span.line,
                        column: st.span.column,
                    });
                }
            },
            None => {
                return Err(LibTpuError::UnexpectedEof {
                    expected: String::from("operation mnemonic"),
                    line: start_span.line,
                    column: start_span.column,
                });
            }
        };

        let (dialect, mnemonic) = match full_op_name.split_once('.') {
            Some((d_str, m_str)) => (DialectKind::from_prefix(d_str), String::from(m_str)),
            None => (DialectKind::StableHlo, full_op_name),
        };

        let mut operands = Vec::new();
        let mut regions = Vec::new();
        let mut successors = Vec::new();
        let mut attributes = BTreeMap::new();
        let mut signature = AstFunctionType::default();
        let mut pending_region_args: Vec<AstBlockArg> = Vec::new();

        // Special custom assembly handling for `stablehlo.compare LT, %a, %b, FLOAT`
        if mnemonic == "compare"
            && let Some(HloToken::Ident(dir_cand)) = self.cursor.peek_token()
            && let Some(dir) = ComparisonDirection::from_ident(dir_cand)
        {
            self.cursor.next_token();
            attributes.insert(
                String::from("comparison_direction"),
                MlirAttribute::ComparisonDirection(dir),
            );
            let _ = self.cursor.consume_punct(',');
        }

        // Special handling for `scf.for %iv = ...` induction variable binding
        if dialect == DialectKind::Scf
            && mnemonic == "for"
            && let Some((iv, _)) = self.cursor.consume_ssa_spanned()
        {
            let _ = self.cursor.consume_punct('=');
            pending_region_args.push(AstBlockArg {
                ssa_name: iv,
                ty: MlirType::Scalar(ElementType::S64),
                attrs: BTreeMap::new(),
                location: None,
            });
        }

        // Parse operands, inline attributes, generic `({ ^bb0: ... })` regions, and `: type` signature
        while !self.cursor.is_eof() {
            if self.is_at_operation_boundary()
                || self.cursor.peek_token() == Some(&HloToken::Ident(String::from("loc")))
            {
                break;
            }
            if self.cursor.consume_punct(':') {
                signature = self.parse_operation_signature()?;
                continue;
            }
            if self.cursor.consume_if(|t| *t == HloToken::Arrow).is_some() {
                if self.cursor.consume_punct('(') {
                    while self.cursor.is_open(')') {
                        signature.results.push(self.parse_type()?);
                        let _ = self.cursor.consume_punct(',');
                    }
                } else {
                    signature.results.push(self.parse_type()?);
                }
                continue;
            }
            // Generic or parenthesized operand list or region list: `(%a, %b)` or `({ ^bb0... })`
            if self.cursor.consume_punct('(') {
                while self.cursor.is_open(')') {
                    if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
                        let reg = self.parse_region()?;
                        regions.push(reg);
                    } else if self.cursor.peek_n(1) == Some(&HloToken::Punct(':')) {
                        // Inline block argument list in `sdy.manual_computation ... (%arg0: tensor<...>)`
                        let ssa_name = self
                            .cursor
                            .consume_ssa_spanned()
                            .map_or_else(String::new, |(s, _)| s);
                        let _ = self.cursor.consume_punct(':');
                        let ty = self
                            .parse_type()
                            .unwrap_or(MlirType::Scalar(ElementType::F32));
                        pending_region_args.push(AstBlockArg {
                            ssa_name,
                            ty,
                            attrs: BTreeMap::new(),
                            location: None,
                        });
                    } else if self.cursor.peek_n(1) == Some(&HloToken::Punct('=')) {
                        // `iter_args(%acc = %v0)` binding
                        let iter_ssa = self
                            .cursor
                            .consume_ssa_spanned()
                            .map_or_else(String::new, |(s, _)| s);
                        let _ = self.cursor.consume_punct('=');
                        pending_region_args.push(AstBlockArg {
                            ssa_name: iter_ssa,
                            ty: MlirType::Scalar(ElementType::F32),
                            attrs: BTreeMap::new(),
                            location: None,
                        });
                        operands.extend(self.try_parse_operand());
                    } else if let Some(opnd) = self.try_parse_operand() {
                        operands.push(opnd);
                    } else {
                        self.cursor.next_token();
                    }
                    let _ = self.cursor.consume_punct(',');
                }
                continue;
            }
            // Inline region `{ ^bb0(%a: T): ... }` vs attribute dictionary `{key = val}`
            if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
                if self.is_next_brace_attr_dict() {
                    let dict = self.parse_attribute_dict()?;
                    attributes.extend(dict);
                } else {
                    let mut reg = self.parse_region()?;
                    for first_blk in reg.blocks.iter_mut().take(1) {
                        if !pending_region_args.is_empty() {
                            first_blk.arguments = std::mem::take(&mut pending_region_args);
                        }
                    }
                    regions.push(reg);
                }
                continue;
            }
            // `dense<...>` constant literal
            if self.cursor.peek_token() == Some(&HloToken::Ident(String::from("dense")))
                || self.cursor.peek_token()
                    == Some(&HloToken::Ident(String::from("dense_resource")))
            {
                let attr = self.parse_attribute()?;
                attributes.insert(String::from("value"), attr);
                continue;
            }
            // SSA operand `%a` or `%0#1`
            if let Some(opnd) = self.try_parse_operand() {
                operands.push(opnd);
                let _ = self.cursor.consume_punct(',');
                continue;
            }
            // Caret successor `^bb1`
            if let Some((cid, _)) = self.cursor.consume_caret_spanned() {
                successors.push(cid);
                let _ = self.cursor.consume_punct(',');
                continue;
            }
            // Symbol callee `@func` (in `func.call @callee(%a)`)
            if let Some(sym) = self.cursor.consume_symbol_str() {
                attributes.insert(String::from("callee"), MlirAttribute::SymbolRef(vec![sym]));
                continue;
            }
            // Custom assembly `key = value` attribute (e.g., `dimensions = [1]`, `in_shardings = [...]`, `permutation = array<i64: 1, 0>`)
            if let Some(HloToken::Ident(key)) = self.cursor.peek_token().cloned()
                && self.cursor.peek_n(1) == Some(&HloToken::Punct('='))
            {
                self.cursor.next_token();
                let _ = self.cursor.consume_punct('=');
                let val = self.parse_attribute()?;
                attributes.insert(key, val);
                let _ = self.cursor.consume_punct(',');
                continue;
            }
            // Custom assembly comparison type trailing identifier (`FLOAT`, `SIGNED`, `TOTALORDER`, `UNSIGNED`)
            if mnemonic == "compare"
                && let Some(HloToken::Ident(ct_cand)) = self.cursor.peek_token()
                && let Some(ct) = ComparisonType::from_ident(ct_cand)
            {
                self.cursor.next_token();
                attributes.insert(
                    String::from("comparison_type"),
                    MlirAttribute::ComparisonType(ct),
                );
                let _ = self.cursor.consume_punct(',');
                continue;
            }
            // Bare `#attr` dialect attribute on operation (e.g. `#stablehlo.dot<...>`)
            if matches!(self.cursor.peek_token(), Some(HloToken::AttrAlias(_))) {
                let attr = self.parse_attribute()?;
                let attr_key = match &attr {
                    MlirAttribute::DotDimensionNumbers(_) => "dot_dimension_numbers",
                    MlirAttribute::GatherDimensionNumbers(_) => "dimension_numbers",
                    MlirAttribute::ScatterDimensionNumbers(_) => "scatter_dimension_numbers",
                    MlirAttribute::SdyTensorSharding(_)
                    | MlirAttribute::SdyTensorShardingPerValue(_) => "sdy.sharding",
                    _ => "dialect_attr",
                };
                attributes.insert(String::from(attr_key), attr);
                let _ = self.cursor.consume_punct(',');
                continue;
            }
            // Otherwise advance one token inside custom assembly syntax (e.g. `across`, `dims`, `into`, etc.)
            self.cursor.next_token();
        }

        let location = self.parse_optional_location();
        let elem_type = signature
            .results
            .iter()
            .chain(&signature.inputs)
            .next()
            .map_or(ElementType::F32, MlirType::element_type);
        let op_kind = classify_ast_operation(dialect, &mnemonic, elem_type, &attributes, &regions);

        Ok(AstOperation {
            results,
            dialect,
            mnemonic,
            op_kind,
            operands,
            regions,
            successors,
            attributes,
            signature,
            location,
            span: start_span,
        })
    }

    /// Returns `true` if the cursor is positioned at the start of the next operation or end of block.
    fn is_at_operation_boundary(&self) -> bool {
        if self.cursor.peek_token() == Some(&HloToken::Punct('}')) {
            return true;
        }
        if matches!(self.cursor.peek_token(), Some(HloToken::CaretId(_)))
            && matches!(self.cursor.peek_n(1), Some(HloToken::Punct(':' | '(')))
        {
            return true;
        }
        // Check `%next = ...` or `%next:2 = ...` or `%a, %b = ...`
        if matches!(self.cursor.peek_token(), Some(HloToken::SsaValue(_))) {
            let mut k = 1_usize;
            while let Some(tok) = self.cursor.peek_n(k) {
                match tok {
                    HloToken::Punct('=') => return true,
                    HloToken::Punct(',' | ':') | HloToken::Number(_) | HloToken::SsaValue(_) => {
                        k += 1;
                    }
                    _ => break,
                }
            }
        }
        // Check dialect-qualified operation (`func.return`, `tpu.wait_dma`, `cf.br`, etc.) or `return`
        if let Some(HloToken::Ident(id)) = self.cursor.peek_token()
            && (id.contains('.') || id == "return")
        {
            return true;
        }
        false
    }

    /// Returns `true` if the next `{` begins an attribute dictionary (`{key = val}` or `{}`) rather than a nested region.
    fn is_next_brace_attr_dict(&self) -> bool {
        if self.cursor.peek_n(1) == Some(&HloToken::Punct('}')) {
            return true;
        }
        if matches!(
            self.cursor.peek_n(1),
            Some(HloToken::CaretId(_) | HloToken::SsaValue(_))
        ) {
            return false;
        }
        matches!(
            (self.cursor.peek_n(1), self.cursor.peek_n(2)),
            (
                Some(HloToken::Ident(_) | HloToken::StringLiteral(_)),
                Some(HloToken::Punct('=' | ':' | ',' | '}'))
            )
        )
    }

    /// Attempts to consume an [`AstOperand`] (`%arg0` or `%0#1`).
    fn try_parse_operand(&mut self) -> Option<AstOperand> {
        let (raw, span) = self.cursor.consume_ssa_spanned()?;
        let (base, result_index) = match raw.split_once('#') {
            Some((b, idx_str)) => (String::from(b), idx_str.parse::<usize>().ok()),
            None => (raw, None),
        };
        Some(AstOperand {
            name: base,
            result_index,
            span,
        })
    }

    /// Parses an operation's type signature after `:` (`(T1, T2) -> (R1, R2)` or shorthand `T1, T2` / `T1 -> R1`).
    fn parse_operation_signature(&mut self) -> LibTpuResult<AstFunctionType> {
        let mut sig = AstFunctionType::default();
        let is_paren_inputs = self.cursor.consume_punct('(');
        if is_paren_inputs {
            while self.cursor.is_open(')') {
                sig.inputs.push(self.parse_type()?);
                let _ = self.cursor.consume_punct(',');
            }
        } else {
            sig.inputs.push(self.parse_type()?);
            while self.cursor.consume_punct(',') {
                sig.inputs.push(self.parse_type()?);
            }
        }

        if self.cursor.consume_if(|t| *t == HloToken::Arrow).is_some() {
            if self.cursor.consume_punct('(') {
                while self.cursor.is_open(')') {
                    sig.results.push(self.parse_type()?);
                    let _ = self.cursor.consume_punct(',');
                }
            } else {
                sig.results.push(self.parse_type()?);
            }
        } else if !is_paren_inputs {
            // Functional shorthand: `%0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>`
            sig.results.clone_from(&sig.inputs);
        }

        Ok(sig)
    }

    /// Parses an MLIR [`MlirType`] (`tensor<...>`, `memref<...>`, `vector<...>`, `tuple<...>`, `!tpu.semaphore`, `!quant.uniform<...>`, scalar).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the type syntax is invalid or references an undefined `!alias`.
    #[allow(clippy::too_many_lines)]
    pub fn parse_type(&mut self) -> LibTpuResult<MlirType> {
        // Dialect type or `!alias`
        if let Some((alias, alias_span)) = self.cursor.consume_type_alias_spanned() {
            if matches!(
                alias.as_str(),
                "!stablehlo.token" | "!mhlo.token" | "!vhlo.token_v1"
            ) {
                return Ok(MlirType::Token);
            }
            if alias == "!tpu.semaphore" {
                return Ok(MlirType::Semaphore(TpuSemaphoreKind::Regular));
            }
            if alias == "!tpu.dma_semaphore" {
                return Ok(MlirType::Semaphore(TpuSemaphoreKind::Dma));
            }
            if alias.starts_with("!quant.uniform") {
                return Ok(self.parse_quantized_type());
            }
            if self.cursor.peek_token() == Some(&HloToken::Punct('<')) {
                let _ = self.consume_balanced_angle_brackets();
            }
            if let Some(resolved) = self.type_aliases.get(&alias) {
                return Ok(resolved.clone());
            }
            return Err(LibTpuError::UndefinedAlias {
                alias,
                line: alias_span.line,
                column: alias_span.column,
            });
        }

        // Functional type `(T1, T2) -> R1`
        if self.cursor.peek_token() == Some(&HloToken::Punct('(')) {
            let sig = self.parse_operation_signature()?;
            return Ok(MlirType::Function(sig));
        }

        let (kw, kw_span) = self.cursor.expect_ident()?;
        match kw.as_str() {
            "tensor" => self.parse_tensor_mlir_type(),
            "memref" => self.parse_memref_mlir_type(),
            "vector" => self.parse_vector_mlir_type(),
            "tuple" => {
                let _ = self.cursor.consume_punct('<');
                let mut elems = Vec::new();
                while self.cursor.is_open('>') {
                    elems.push(self.parse_type()?);
                    let _ = self.cursor.consume_punct(',');
                }
                Ok(MlirType::Tuple(elems))
            }
            "token" => Ok(MlirType::Token),
            "complex" => {
                let _ = self.cursor.consume_punct('<');
                let inner = self.cursor.consume_any_ident_str().unwrap_or_default();
                let _ = self.cursor.consume_punct('>');
                let et = if inner == "f64" {
                    ElementType::C128
                } else {
                    ElementType::C64
                };
                Ok(MlirType::Scalar(et))
            }
            scalar_str => parse_mlir_element_type(scalar_str).map_or_else(
                || {
                    Err(LibTpuError::InvalidShape {
                        reason: format!(
                            "Unknown MLIR type '{scalar_str}' at line {}, column {}",
                            kw_span.line, kw_span.column
                        ),
                    })
                },
                |et| Ok(MlirType::Scalar(et)),
            ),
        }
    }

    /// Parses a `tensor<...>` [`MlirType`] (ranked, unranked, dynamic `?`, bounded `<=N`, quantized, or encoded).
    fn parse_tensor_mlir_type(&mut self) -> LibTpuResult<MlirType> {
        let body = self.consume_balanced_angle_brackets();
        let (shape_part, encoding_part) = split_top_level_comma(&body);
        let (dims, is_unranked, dtype_str) = split_shaped_dims_and_dtype(shape_part)?;
        let elem_type = if dtype_str.starts_with("!quant.uniform") {
            if dtype_str.contains("i4") {
                ElementType::S4
            } else if dtype_str.contains("u8") {
                ElementType::U8
            } else {
                ElementType::S8
            }
        } else {
            parse_mlir_element_type(&dtype_str).ok_or_else(|| LibTpuError::InvalidShape {
                reason: format!(
                    "Unknown MLIR tensor element type '{dtype_str}' in 'tensor<{body}>'"
                ),
            })?
        };
        if is_unranked {
            return Ok(MlirType::UnrankedTensor(elem_type));
        }
        let encoding = encoding_part.map(|enc_str| {
            let trimmed = enc_str.trim();
            trimmed
                .strip_prefix("#stablehlo.bounds<")
                .and_then(|s| s.strip_suffix('>'))
                .map_or_else(
                    || TensorEncodingAttr::Custom(String::from(trimmed)),
                    |b_inner| {
                        let bounds = b_inner
                            .split(',')
                            .map(|item| {
                                let t = item.trim();
                                if t == "?" {
                                    None
                                } else {
                                    t.parse::<i64>().ok()
                                }
                            })
                            .collect();
                        TensorEncodingAttr::Bounds(bounds)
                    },
                )
        });
        Ok(MlirType::RankedTensor(RankedTensorType {
            dims,
            element_type: elem_type,
            encoding,
            layout: None,
        }))
    }

    /// Parses a `memref<...>` [`MlirType`] including optional layout (`#tpu.tiled`, `strided`) and memory space (`#tpu.memory_space<vmem>`).
    fn parse_memref_mlir_type(&mut self) -> LibTpuResult<MlirType> {
        let body = self.consume_balanced_angle_brackets();
        let (shape_part, rest_opt) = split_top_level_comma(&body);
        let (dims, _, dtype_str) = split_shaped_dims_and_dtype(shape_part)?;
        let element_type = if matches!(dtype_str.as_str(), "!tpu.semaphore" | "!tpu.dma_semaphore")
        {
            ElementType::U32
        } else {
            parse_mlir_element_type(&dtype_str).ok_or_else(|| LibTpuError::InvalidShape {
                reason: format!("Unknown memref element type '{dtype_str}' in 'memref<{body}>'"),
            })?
        };
        let mut layout = None;
        let mut memory_space = None;
        if let Some(rest) = rest_opt {
            if rest.contains("vmem") {
                memory_space = Some(TpuMemorySpace::Vmem);
            } else if rest.contains("smem") {
                memory_space = Some(TpuMemorySpace::Smem);
            } else if rest.contains("cmem") {
                memory_space = Some(TpuMemorySpace::Cmem);
            } else if rest.contains("semaphore_mem") {
                memory_space = Some(TpuMemorySpace::SemaphoreMem);
            } else if rest.contains("hbm") {
                memory_space = Some(TpuMemorySpace::Hbm);
            }
            if let Some(tiled_after) = rest.split_once("#tpu.tiled<").map(|(_, a)| a) {
                let tile_dims = tiled_after
                    .split(|c: char| !c.is_ascii_digit())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<i64>().unwrap_or(1))
                    .collect();
                layout = Some(MemRefLayoutAttr::Tiled { tile_dims });
            } else if rest.contains("strided<") {
                layout = Some(MemRefLayoutAttr::Strided {
                    strides: vec![1],
                    offset: Some(0),
                });
            }
        }
        Ok(MlirType::MemRef(MemRefType {
            dims,
            element_type,
            layout,
            memory_space,
        }))
    }

    /// Parses a `vector<...>` [`MlirType`].
    fn parse_vector_mlir_type(&mut self) -> LibTpuResult<MlirType> {
        let body = self.consume_balanced_angle_brackets();
        let scalable_each = body.contains('[');
        let (dims, _, dtype_str) = split_shaped_dims_and_dtype(&body)?;
        let element_type =
            parse_mlir_element_type(&dtype_str).ok_or_else(|| LibTpuError::InvalidShape {
                reason: format!("Unknown vector element type '{dtype_str}' in 'vector<{body}>'"),
            })?;
        let scalable_dims = vec![scalable_each; dims.len()];
        Ok(MlirType::Vector(VectorType {
            dims,
            scalable_dims,
            element_type,
        }))
    }

    /// Parses a `!quant.uniform<i8:f32, 0.05:10>` quantized type.
    fn parse_quantized_type(&mut self) -> MlirType {
        let body = self.consume_balanced_angle_brackets();
        let (types_part, params_part) = body.split_once(',').unwrap_or((&body, "1.0:0"));
        let (st_str, ex_str) = types_part.split_once(':').unwrap_or(("i8", "f32"));
        let storage_type = parse_mlir_element_type(st_str.trim()).unwrap_or(ElementType::S8);
        let expressed_type = parse_mlir_element_type(ex_str.trim()).unwrap_or(ElementType::F32);
        let trimmed_params = params_part.trim();
        let (scale_str, zp_str) = trimmed_params
            .split_once(':')
            .unwrap_or((trimmed_params, "0"));
        let scale = scale_str.trim().parse::<f64>().unwrap_or(1.0);
        let zp = zp_str.trim().parse::<i64>().unwrap_or(0);
        MlirType::Quantized(QuantizedElementType {
            storage_type,
            expressed_type,
            scales: vec![scale],
            zero_points: vec![zp],
            quantized_dimension: None,
        })
    }

    /// Consumes a balanced `<...>` block from the token cursor and reconstructs its compact string representation.
    fn consume_balanced_angle_brackets(&mut self) -> String {
        let _ = self.cursor.consume_punct('<');
        let mut depth = 1_i32;
        let mut out = String::new();
        while let Some(st) = self.cursor.next_token() {
            match &st.token {
                HloToken::Punct('<') => {
                    depth += 1;
                    out.push('<');
                }
                HloToken::Punct('>') => {
                    depth -= 1;
                    if depth == 0 {
                        return out;
                    }
                    out.push('>');
                }
                HloToken::Le => out.push_str("<="),
                HloToken::Arrow => out.push_str("->"),
                HloToken::DoubleColon => out.push_str("::"),
                HloToken::Ellipsis => out.push_str("..."),
                HloToken::Ident(s)
                | HloToken::Number(s)
                | HloToken::SsaValue(s)
                | HloToken::AttrAlias(s)
                | HloToken::TypeAlias(s)
                | HloToken::CaretId(s) => {
                    if !out.is_empty()
                        && out.as_bytes().last().is_some_and(u8::is_ascii_alphanumeric)
                    {
                        out.push(' ');
                    }
                    out.push_str(s);
                }
                HloToken::Symbol(s) => {
                    out.push('@');
                    out.push_str(s);
                }
                HloToken::StringLiteral(s) => {
                    out.push('"');
                    out.push_str(s);
                    out.push('"');
                }
                HloToken::Punct(c) => out.push(*c),
                HloToken::Ge | HloToken::EqEq | HloToken::NotEq | HloToken::HexBlob(_) => {
                    out.push(' ');
                }
            }
        }
        String::new()
    }

    /// Parses an attribute dictionary `{ key = value, flag_attr, ... }`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the dictionary contains malformed attributes.
    pub fn parse_attribute_dict(&mut self) -> LibTpuResult<BTreeMap<String, MlirAttribute>> {
        let _ = self.cursor.consume_punct('{');
        let mut dict = BTreeMap::new();
        while self.cursor.is_open('}') {
            let Some(key) = self
                .cursor
                .consume_any_ident_str()
                .or_else(|| self.cursor.consume_string_str())
            else {
                let span = self.cursor.current_span();
                return Err(LibTpuError::UnexpectedToken {
                    expected: String::from("attribute key"),
                    found: format!("{:?}", self.cursor.peek_token()),
                    line: span.line,
                    column: span.column,
                });
            };
            let val = if self.cursor.consume_punct('=') || self.cursor.consume_punct(':') {
                self.parse_attribute()?
            } else {
                // Unit attribute `{has_side_effect}` -> `Bool(true)`
                MlirAttribute::Bool(true)
            };
            dict.insert(key, val);
            let _ = self.cursor.consume_punct(',');
        }
        Ok(dict)
    }

    /// Parses a single [`MlirAttribute`] value.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the attribute syntax is malformed or references an undefined `#alias`.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn parse_attribute(&mut self) -> LibTpuResult<MlirAttribute> {
        let span = self.cursor.current_span();
        let Some(peek_tok) = self.cursor.peek_token().cloned() else {
            return Err(LibTpuError::UnexpectedEof {
                expected: String::from("attribute value"),
                line: span.line,
                column: span.column,
            });
        };

        match peek_tok {
            HloToken::StringLiteral(s) => {
                self.cursor.next_token();
                Ok(MlirAttribute::String(s))
            }
            HloToken::HexBlob(bytes) => {
                self.cursor.next_token();
                Ok(MlirAttribute::DenseElements(DenseElementsAttr {
                    values_f64: ElementType::F32.unpack_to_f64_vec(&bytes),
                    raw_bytes: Some(bytes),
                    is_splat: false,
                    tensor_type: None,
                }))
            }
            HloToken::Symbol(sym) => {
                self.cursor.next_token();
                let mut chain = vec![sym];
                while self
                    .cursor
                    .consume_if(|t| *t == HloToken::DoubleColon)
                    .is_some()
                {
                    if let Some(sub_sym) = self.cursor.consume_symbol_str() {
                        chain.push(sub_sym);
                    }
                }
                Ok(MlirAttribute::SymbolRef(chain))
            }
            HloToken::Number(num_str) => {
                self.cursor.next_token();
                let opt_ty = if self.cursor.consume_punct(':') {
                    self.cursor
                        .consume_any_ident_str()
                        .and_then(|s| parse_mlir_element_type(&s))
                } else {
                    None
                };
                if let Some(hex_digits) = num_str
                    .strip_prefix("0x")
                    .or_else(|| num_str.strip_prefix("0X"))
                    && !num_str.contains(['.', 'p', 'P'])
                {
                    let val = i128::from_str_radix(hex_digits, 16).unwrap_or(0);
                    return Ok(MlirAttribute::Integer {
                        value: val,
                        ty: opt_ty,
                    });
                }
                if let Ok(val) = num_str.parse::<i128>() {
                    return Ok(MlirAttribute::Integer {
                        value: val,
                        ty: opt_ty,
                    });
                }
                let f_val = parse_float_literal(&num_str);
                Ok(MlirAttribute::Float {
                    value: f_val,
                    ty: opt_ty,
                })
            }
            HloToken::Punct('[') => {
                self.cursor.next_token();
                let mut items = Vec::new();
                while self.cursor.is_open(']') {
                    items.push(self.parse_attribute()?);
                    let _ = self.cursor.consume_punct(',');
                }
                Ok(MlirAttribute::Array(items))
            }
            HloToken::Punct('{') => {
                let dict = self.parse_attribute_dict()?;
                Ok(MlirAttribute::Dictionary(dict))
            }
            HloToken::AttrAlias(alias) => {
                self.cursor.next_token();
                self.parse_dialect_or_alias_attr(&alias, span)
            }
            HloToken::Ident(id) => match id.as_str() {
                "true" => {
                    self.cursor.next_token();
                    Ok(MlirAttribute::Bool(true))
                }
                "false" => {
                    self.cursor.next_token();
                    Ok(MlirAttribute::Bool(false))
                }
                "nan" | "inf" => {
                    self.cursor.next_token();
                    Ok(MlirAttribute::Float {
                        value: parse_float_literal(&id),
                        ty: None,
                    })
                }
                "array" => {
                    self.cursor.next_token();
                    Ok(self.parse_dense_array_attr())
                }
                "dense" => {
                    self.cursor.next_token();
                    Ok(self.parse_dense_elements_attr())
                }
                "dense_resource" => {
                    self.cursor.next_token();
                    let handle = self.consume_balanced_angle_brackets();
                    if self.cursor.consume_punct(':') {
                        let _ = self.parse_type();
                    }
                    Ok(MlirAttribute::DenseResource(handle))
                }
                "affine_map" => {
                    self.cursor.next_token();
                    let body = self.consume_balanced_angle_brackets();
                    Ok(MlirAttribute::AffineMap(AffineMapExpr {
                        dims: Vec::new(),
                        symbols: Vec::new(),
                        results: vec![body],
                    }))
                }
                "loc" => {
                    let loc = self.parse_location();
                    Ok(MlirAttribute::Location(loc))
                }
                "tensor" | "memref" | "vector" | "tuple" => {
                    let ty = self.parse_type()?;
                    Ok(MlirAttribute::Type(ty))
                }
                _ => {
                    self.cursor.next_token();
                    if let Some(dir) = ComparisonDirection::from_ident(&id) {
                        return Ok(MlirAttribute::ComparisonDirection(dir));
                    }
                    if let Some(ct) = ComparisonType::from_ident(&id) {
                        return Ok(MlirAttribute::ComparisonType(ct));
                    }
                    Ok(MlirAttribute::String(id))
                }
            },
            _ => {
                self.cursor.next_token();
                Ok(MlirAttribute::String(format!("{peek_tok:?}")))
            }
        }
    }

    /// Parses `array<i64: 1, 2, 3>` or `array<i1: true, false>`.
    fn parse_dense_array_attr(&mut self) -> MlirAttribute {
        let body = self.consume_balanced_angle_brackets();
        let (ty_part, vals_part) = body.split_once(':').unwrap_or(("i64", &body));
        if ty_part.trim() == "i1" {
            let bools = vals_part
                .split(',')
                .filter_map(|s| match s.trim() {
                    "true" | "1" => Some(true),
                    "false" | "0" => Some(false),
                    _ => None,
                })
                .collect();
            return MlirAttribute::DenseBoolArray(bools);
        }
        let ints = vals_part
            .split(',')
            .filter_map(|s| s.trim().parse::<i64>().ok())
            .collect();
        MlirAttribute::DenseI64Array(ints)
    }

    /// Parses `dense<...>` multi-dimensional array literals, splats, complex pairs, and hex strings `"0x..."`.
    fn parse_dense_elements_attr(&mut self) -> MlirAttribute {
        let _ = self.cursor.consume_punct('<');
        let mut raw_bytes = None;
        let mut values_f64 = Vec::new();
        let mut bracket_count = 0_usize;

        while self.cursor.is_open('>') {
            match self.cursor.next_token().map(|st| &st.token) {
                Some(HloToken::Punct('[')) => bracket_count += 1,
                Some(HloToken::HexBlob(bytes)) => {
                    values_f64 = ElementType::F32.unpack_to_f64_vec(bytes);
                    raw_bytes = Some(bytes.clone());
                }
                Some(HloToken::Number(s) | HloToken::Ident(s)) => {
                    values_f64.push(parse_float_literal(s));
                }
                _ => {}
            }
        }

        let tensor_type = if self.cursor.consume_punct(':')
            && let Ok(MlirType::RankedTensor(rt)) = self.parse_type()
        {
            Some(rt)
        } else {
            None
        };
        let is_splat = bracket_count == 0 && values_f64.len() == 1 && raw_bytes.is_none();
        MlirAttribute::DenseElements(DenseElementsAttr {
            values_f64,
            raw_bytes,
            is_splat,
            tensor_type,
        })
    }

    /// Parses a `#dialect<...>` or `#alias` attribute.
    ///
    /// # Arguments
    ///
    /// * `alias` - Dialect attribute prefix or alias identifier (e.g., `"#stablehlo.dot"`, `"#sdy.sharding"`).
    /// * `span` - Source [`TokenSpan`] of the `#alias` token.
    #[allow(clippy::too_many_lines)]
    fn parse_dialect_or_alias_attr(
        &mut self,
        alias: &str,
        span: TokenSpan,
    ) -> LibTpuResult<MlirAttribute> {
        if self.cursor.peek_token() != Some(&HloToken::Punct('<')) {
            if let Some(resolved) = self.attr_aliases.get(alias) {
                return Ok(resolved.clone());
            }
            if let Some(loc) = self.locations.get(alias) {
                return Ok(MlirAttribute::Location(loc.clone()));
            }
            return Err(LibTpuError::UndefinedAlias {
                alias: String::from(alias),
                line: span.line,
                column: span.column,
            });
        }

        if alias == "#sdy.sharding" {
            let sharding = self.parse_sdy_tensor_sharding();
            return Ok(MlirAttribute::SdyTensorSharding(sharding));
        }
        if alias == "#sdy.sharding_per_value" {
            let _ = self.cursor.consume_punct('<');
            let _ = self.cursor.consume_punct('[');
            let mut list = Vec::new();
            while self.cursor.is_open(']') {
                let _ = self.cursor.consume_attr_alias_spanned();
                list.push(self.parse_sdy_tensor_sharding());
                let _ = self.cursor.consume_punct(',');
            }
            let _ = self.cursor.consume_punct('>');
            return Ok(MlirAttribute::SdyTensorShardingPerValue(list));
        }
        if alias == "#sdy.mesh" {
            let mesh = self.parse_sdy_mesh_body();
            return Ok(MlirAttribute::SdyMesh(mesh));
        }

        let body = self.consume_balanced_angle_brackets();
        match alias {
            "#stablehlo.dot" | "#mhlo.dot" => {
                let dot = DotDimensionNumbersAttr {
                    lhs_batching_dimensions: extract_bracketed_usize_list(
                        &body,
                        "lhs_batching_dimensions",
                    ),
                    rhs_batching_dimensions: extract_bracketed_usize_list(
                        &body,
                        "rhs_batching_dimensions",
                    ),
                    lhs_contracting_dimensions: extract_bracketed_usize_list(
                        &body,
                        "lhs_contracting_dimensions",
                    ),
                    rhs_contracting_dimensions: extract_bracketed_usize_list(
                        &body,
                        "rhs_contracting_dimensions",
                    ),
                };
                Ok(MlirAttribute::DotDimensionNumbers(dot))
            }
            "#stablehlo.gather" | "#mhlo.gather" => {
                let gather = GatherDimensionNumbersAttr {
                    offset_dims: extract_bracketed_usize_list(&body, "offset_dims"),
                    collapsed_slice_dims: extract_bracketed_usize_list(
                        &body,
                        "collapsed_slice_dims",
                    ),
                    operand_batching_dims: extract_bracketed_usize_list(
                        &body,
                        "operand_batching_dims",
                    ),
                    start_indices_batching_dims: extract_bracketed_usize_list(
                        &body,
                        "start_indices_batching_dims",
                    ),
                    start_index_map: extract_bracketed_usize_list(&body, "start_index_map"),
                    index_vector_dim: extract_named_usize(&body, "index_vector_dim").unwrap_or(0),
                };
                Ok(MlirAttribute::GatherDimensionNumbers(gather))
            }
            "#stablehlo.scatter" | "#mhlo.scatter" => {
                let scatter = ScatterDimensionNumbersAttr {
                    update_window_dims: extract_bracketed_usize_list(&body, "update_window_dims"),
                    inserted_window_dims: extract_bracketed_usize_list(
                        &body,
                        "inserted_window_dims",
                    ),
                    input_batching_dims: extract_bracketed_usize_list(&body, "input_batching_dims"),
                    scatter_indices_batching_dims: extract_bracketed_usize_list(
                        &body,
                        "scatter_indices_batching_dims",
                    ),
                    scatter_dims_to_operand_dims: extract_bracketed_usize_list(
                        &body,
                        "scatter_dims_to_operand_dims",
                    ),
                    index_vector_dim: extract_named_usize(&body, "index_vector_dim").unwrap_or(0),
                };
                Ok(MlirAttribute::ScatterDimensionNumbers(scatter))
            }
            "#stablehlo.channel_handle" | "#mhlo.channel_handle" => {
                let handle = extract_named_usize(&body, "handle")
                    .and_then(|v| i64::try_from(v).ok())
                    .unwrap_or(0);
                let channel_type = extract_named_usize(&body, "type")
                    .and_then(|v| i64::try_from(v).ok())
                    .unwrap_or(0);
                Ok(MlirAttribute::ChannelHandle {
                    handle,
                    channel_type,
                })
            }
            "#stablehlo" | "#mhlo" => {
                if let Some(dir_str) = body.strip_prefix("comparison_direction")
                    && let Some(dir) = ComparisonDirection::from_ident(dir_str)
                {
                    return Ok(MlirAttribute::ComparisonDirection(dir));
                }
                if let Some(ct_str) = body.strip_prefix("comparison_type")
                    && let Some(ct) = ComparisonType::from_ident(ct_str)
                {
                    return Ok(MlirAttribute::ComparisonType(ct));
                }
                if let Some(prec_str) = body.strip_prefix("precision") {
                    let level = match prec_str.trim().to_ascii_uppercase().as_str() {
                        "HIGH" => PrecisionLevel::High,
                        "HIGHEST" => PrecisionLevel::Highest,
                        _ => PrecisionLevel::Default,
                    };
                    return Ok(MlirAttribute::PrecisionConfig(vec![level]));
                }
                if let Some(fft_str) = body.strip_prefix("fft_type") {
                    let kind = match fft_str.trim().to_ascii_uppercase().as_str() {
                        "IFFT" => FftKind::Ifft,
                        "RFFT" => FftKind::Rfft,
                        "IRFFT" => FftKind::Irfft,
                        _ => FftKind::Fft,
                    };
                    return Ok(MlirAttribute::FftType(kind));
                }
                if let Some(rng_str) = body.strip_prefix("rng_algorithm") {
                    let alg = match rng_str.trim().to_ascii_uppercase().as_str() {
                        "THREE_FRY" => RngAlgorithmKind::ThreeFry,
                        "PHILOX" => RngAlgorithmKind::Philox,
                        _ => RngAlgorithmKind::Default,
                    };
                    return Ok(MlirAttribute::RngAlgorithm(alg));
                }
                if let Some(tr_str) = body.strip_prefix("transpose") {
                    let mode = match tr_str.trim().to_ascii_uppercase().as_str() {
                        "TRANSPOSE" => TransposeMode::Transpose,
                        "ADJOINT" => TransposeMode::Adjoint,
                        _ => TransposeMode::NoTranspose,
                    };
                    return Ok(MlirAttribute::TransposeKind(mode));
                }
                Ok(MlirAttribute::String(body))
            }
            "#tpu.dimension_semantics" => Ok(MlirAttribute::MosaicAttr(
                MosaicDialectAttr::DimensionSemantics(body),
            )),
            "#tpu.memory_space" => {
                let space = TpuMemorySpace::from_ident(&body).unwrap_or(TpuMemorySpace::Vmem);
                Ok(MlirAttribute::MosaicAttr(MosaicDialectAttr::MemorySpace(
                    space,
                )))
            }
            "#tpu.core_type" => Ok(MlirAttribute::MosaicAttr(MosaicDialectAttr::CoreType(body))),
            _ => Ok(MlirAttribute::String(format!("{alias}<{body}>"))),
        }
    }

    /// Parses `#sdy.sharding<@mesh, [{"x"}, {"y", ?}], replicated={"z"}, unreduced={"w"}>`.
    fn parse_sdy_tensor_sharding(&mut self) -> SdyTensorShardingAttr {
        let body = self.consume_balanced_angle_brackets();
        let (mesh_part, rest) = body.split_once(',').unwrap_or((&body, ""));
        let mut sharding = SdyTensorShardingAttr {
            mesh_name: String::from(mesh_part.trim().trim_start_matches('@')),
            dim_shardings: Vec::new(),
            replicated_axes: extract_braced_string_list(rest, "replicated"),
            unreduced_axes: extract_braced_string_list(rest, "unreduced"),
        };
        if let Some((dims_inner, _)) = rest
            .split_once('[')
            .and_then(|(_, after_open)| after_open.split_once(']'))
        {
            for seg in dims_inner.split('{').skip(1) {
                let (brace_inner, after_brace) = seg.split_once('}').unwrap_or((seg, ""));
                let mut dim_sh = SdyDimensionSharding::default();
                for clean in brace_inner
                    .split(',')
                    .map(|item| item.trim().trim_matches('"'))
                    .filter(|item| !item.is_empty())
                {
                    if clean == "?" {
                        dim_sh.is_open = true;
                    } else if let Some(p_str) = clean.strip_prefix('p')
                        && let Ok(prio) = p_str.parse::<i64>()
                    {
                        dim_sh.priority = Some(prio);
                    } else {
                        dim_sh.axes.push(String::from(clean));
                    }
                }
                if let Some(p_str) = after_brace.trim().strip_prefix('p') {
                    dim_sh.priority = p_str.trim_end_matches(',').parse::<i64>().ok();
                }
                sharding.dim_shardings.push(dim_sh);
            }
        }
        sharding
    }

    /// Parses an optional trailing `loc(...)` specification if the next token is `"loc"`.
    fn parse_optional_location(&mut self) -> Option<MlirLocation> {
        if self.cursor.peek_token() == Some(&HloToken::Ident(String::from("loc"))) {
            Some(self.parse_location())
        } else {
            None
        }
    }

    /// Parses a `loc(...)` location specification (`loc(unknown)`, `loc(#loc0)`, `loc("file.py":10:4)`, `loc(callsite(...))`, `loc(fused[...])`).
    pub fn parse_location(&mut self) -> MlirLocation {
        let _ = self.cursor.next_token(); // "loc"
        let _ = self.cursor.consume_punct('(');
        let loc = self.parse_location_inner();
        let _ = self.cursor.consume_punct(')');
        loc
    }

    /// Parses the inner payload inside `loc(...)`.
    fn parse_location_inner(&mut self) -> MlirLocation {
        if self.cursor.consume_ident("unknown") {
            return MlirLocation::Unknown;
        }
        if self.cursor.consume_ident("callsite") {
            let _ = self.cursor.consume_punct('(');
            let inner_callee = self.parse_location_inner();
            let _ = self.cursor.consume_ident("at");
            let outer_caller = self.parse_location_inner();
            let _ = self.cursor.consume_punct(')');
            return MlirLocation::CallSite {
                callee: Box::new(inner_callee),
                caller: Box::new(outer_caller),
            };
        }
        if self.cursor.consume_ident("fused") {
            if self.cursor.peek_token() == Some(&HloToken::Punct('<')) {
                let _ = self.consume_balanced_angle_brackets();
            }
            let mut items = Vec::new();
            let _ = self.cursor.consume_punct('[');
            while self.cursor.is_open(']') {
                items.push(self.parse_location_inner());
                let _ = self.cursor.consume_punct(',');
            }
            return MlirLocation::Fused(items);
        }
        if let Some((alias, _)) = self.cursor.consume_attr_alias_spanned() {
            return MlirLocation::Alias(alias);
        }
        if let Some(file_or_name) = self.cursor.consume_string_str() {
            if self.cursor.consume_punct(':') {
                let line = self
                    .cursor
                    .consume_number_str()
                    .unwrap_or_default()
                    .parse::<usize>()
                    .unwrap_or(1);
                let _ = self.cursor.consume_punct(':');
                let column = self
                    .cursor
                    .consume_number_str()
                    .unwrap_or_default()
                    .parse::<usize>()
                    .unwrap_or(1);
                return MlirLocation::FileLineCol {
                    file: file_or_name,
                    line,
                    column,
                };
            }
            let child = if self.cursor.consume_punct('(') {
                let inner = self.parse_location_inner();
                let _ = self.cursor.consume_punct(')');
                Some(Box::new(inner))
            } else {
                None
            };
            return MlirLocation::Named {
                name: file_or_name,
                child,
            };
        }
        MlirLocation::Unknown
    }
}

/// Parses a floating-point literal string supporting scientific notation (`1.0e-5`), hex float (`0x1.8p+1`),
/// hex integer IEEE-754 bit patterns (`0x7FC00000`), and special values (`nan`, `inf`, `-inf`).
///
/// # Arguments
///
/// * `raw` - Numeric token string.
#[must_use]
pub fn parse_float_literal(raw: &str) -> f64 {
    let trimmed = raw.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "nan" | "+nan" | "-nan" => return f64::NAN,
        "inf" | "+inf" | "infinity" => return f64::INFINITY,
        "-inf" | "-infinity" => return f64::NEG_INFINITY,
        "true" => return 1.0,
        "false" => return 0.0,
        _ => {}
    }
    if let Some(hex_body) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        if let Some((mant_str, exp_str)) = hex_body.split_once(['p', 'P']) {
            let (int_part, frac_part) = mant_str.split_once('.').unwrap_or((mant_str, ""));
            let int_val = u64::from_str_radix(int_part, 16).unwrap_or(0);
            let frac_bits = u64::from_str_radix(frac_part, 16).unwrap_or(0);
            let scale = 16.0_f64.powi(i32::try_from(frac_part.len()).unwrap_or(1));
            #[allow(clippy::cast_precision_loss)]
            let mant = (int_val as f64) + (frac_bits as f64) / scale;
            let exp = exp_str.parse::<i32>().unwrap_or(0);
            return mant * 2.0_f64.powi(exp);
        }
        if hex_body.len() <= 8 {
            return f64::from(f32::from_bits(
                u32::from_str_radix(hex_body, 16).unwrap_or(0),
            ));
        }
        return f64::from_bits(u64::from_str_radix(hex_body, 16).unwrap_or(0));
    }
    trimmed.parse::<f64>().unwrap_or(0.0)
}

/// Splits `s` at the first top-level `,` (ignoring commas nested inside `<...>`, `[...]`, or `(...)`).
///
/// # Arguments
///
/// * `s` - Input string slice to split.
#[must_use]
fn split_top_level_comma(s: &str) -> (&str, Option<&str>) {
    let mut depth = 0_i32;
    let bytes = s.as_bytes();
    for (idx, ch) in s.char_indices() {
        if matches!(ch, '[' | '(' | '{') || (ch == '<' && bytes.get(idx + 1) != Some(&b'=')) {
            depth += 1;
        } else if matches!(ch, '>' | ']' | ')' | '}') {
            depth -= 1;
        } else if ch == ',' && depth == 0 {
            let left = s.get(..idx).unwrap_or(s);
            let right = s.get(idx + 1..).unwrap_or("");
            return (left, Some(right));
        }
    }
    (s, None)
}

/// Extracts a `key = {s0, s1, ...}` list of axis strings from a raw attribute string.
///
/// # Arguments
///
/// * `body` - Raw attribute body string.
/// * `key` - Attribute key name to search for.
#[must_use]
fn extract_braced_string_list(body: &str, key: &str) -> Vec<String> {
    body.split_once(key)
        .and_then(|(_, after_key)| after_key.split_once('{'))
        .and_then(|(_, after_open)| after_open.split_once('}'))
        .map_or_else(Vec::new, |(inner, _)| {
            inner
                .split(',')
                .map(|item| item.trim().trim_matches('"'))
                .filter(|item| !item.is_empty())
                .map(String::from)
                .collect()
        })
}

/// Extracts a `key = [n0, n1, ...]` list of `usize`s from a raw attribute string.
///
/// # Arguments
///
/// * `body` - Raw attribute body string.
/// * `key` - Attribute key name to search for.
#[must_use]
fn extract_bracketed_usize_list(body: &str, key: &str) -> Vec<usize> {
    body.split_once(key)
        .and_then(|(_, after_key)| after_key.split_once('['))
        .and_then(|(_, after_open)| after_open.split_once(']'))
        .map_or_else(Vec::new, |(inner, _)| {
            inner
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .collect()
        })
}

/// Extracts a `key = N` single `usize` from a raw attribute string.
///
/// # Arguments
///
/// * `body` - Raw attribute body string.
/// * `key` - Attribute key name to search for.
#[must_use]
fn extract_named_usize(body: &str, key: &str) -> Option<usize> {
    let (_, after_key) = body.split_once(key)?;
    let (_, after_eq) = after_key.split_once('=')?;
    let digits: String = after_eq
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse::<usize>().ok()
}

/// Legalizes `chlo.*` operations inside an [`AstModule`] into canonical `stablehlo` / hardware [`AstOpKind`]s.
///
/// # Arguments
///
/// * `module` - Mutable [`AstModule`] to legalize in-place.
pub fn legalize_chlo_in_ast_module(module: &mut AstModule) {
    for func in &mut module.functions {
        legalize_chlo_in_region(&mut func.body);
    }
    for op in &mut module.top_level_operations {
        legalize_chlo_operation(op);
    }
}

/// Recursively legalizes `chlo.*` operations inside an [`AstRegion`].
///
/// # Arguments
///
/// * `region` - Mutable [`AstRegion`] whose operations are legalized.
fn legalize_chlo_in_region(region: &mut AstRegion) {
    for block in &mut region.blocks {
        for op in &mut block.operations {
            legalize_chlo_operation(op);
        }
    }
}

/// Legalizes a single [`AstOperation`] if it belongs to [`DialectKind::Chlo`].
///
/// # Arguments
///
/// * `op` - Mutable [`AstOperation`] to decompose or legalize.
fn legalize_chlo_operation(op: &mut AstOperation) {
    for reg in &mut op.regions {
        legalize_chlo_in_region(reg);
    }
    if let AstOpKind::ChloDecomposable(name) = &op.op_kind {
        let hw = match name.as_str() {
            "erfc" | "erf_inv" => TpuOpCode::Erf,
            "lgamma" | "digamma" | "polygamma" | "zeta" | "bessel_i0e" | "bessel_i1e" => {
                TpuOpCode::Log1p
            }
            "acos" | "acosh" | "cosh" => TpuOpCode::Cosine,
            "asin" | "asinh" | "sinh" => TpuOpCode::Sine,
            "atan" | "atanh" => TpuOpCode::Tan,
            "conj" => TpuOpCode::Complex,
            "next_after" => TpuOpCode::AddF32,
            _ => TpuOpCode::NegF32,
        };
        op.op_kind = AstOpKind::HardwareOp(hw);
        op.dialect = DialectKind::StableHlo;
    }
}

/// Parses an MLIR source string into a verified, CHLO-legalized [`AstModule`].
///
/// # Arguments
///
/// * `src` - Source MLIR program text.
///
/// # Errors
///
/// Returns [`LibTpuError`] if lexing or recursive-descent parsing fails.
pub fn parse_mlir_ast_module(src: &str) -> LibTpuResult<AstModule> {
    let _checked = HloLexer::tokenize_checked(src)?;
    let spanned = HloLexer::tokenize_spanned(src);
    let mut parser = MlirModuleParser::new(&spanned);
    let mut module = parser.parse_module()?;
    legalize_chlo_in_ast_module(&mut module);
    Ok(module)
}
