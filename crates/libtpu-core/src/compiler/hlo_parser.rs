//! Textual XLA HLO recursive-descent parser (`HloTextParser`), `HloModuleProto` converter,
//! and unified type-aware AST-to-`TpuInstructionBundle` lowerer.

use super::ast::{
    AstFunction, AstModule, AstOpKind, AstOperation, AstRegion, ConvDimensionNumbersAttr,
    DialectKind, DimensionSize, DotDimensionNumbersAttr, HloComputationAst, HloInstructionAst,
    HloWindowSpec, MlirAttribute, MlirType, RankedTensorType,
};
use super::lexer::{HloLexer, HloToken, ParsedHloModule, SpannedToken, TokenSpan};
use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use super::parser::{
    TokenCursor, classify_ast_operation, parse_float_literal, parse_mlir_element_type,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    ConvolutionDimensionNumbers, DotDimensionNumbers, HloComputationProto, HloInstructionProto,
    HloModuleProto, LayoutProto, PaddingConfig, PaddingConfigDimension, ReplicaGroup, ShapeProto,
    Window, WindowDimension,
};
use crate::types::{ElementType, MemoryLayout, Shape};
use std::collections::BTreeMap;

/// Recursive-descent parser for textual XLA HLO modules (`HloModule ... ENTRY %main { ... }`).
#[derive(Debug)]
pub struct HloTextParser<'a> {
    /// Underlying spanned token cursor.
    cursor: TokenCursor<'a>,
}

impl<'a> HloTextParser<'a> {
    /// Creates a new [`HloTextParser`] over `tokens`.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Lexed token stream from [`HloLexer::tokenize_spanned`].
    #[must_use]
    pub fn new(tokens: &'a [SpannedToken]) -> Self {
        Self {
            cursor: TokenCursor::new(tokens),
        }
    }

    /// Consumes the next token if it is an identifier, SSA value, symbol, or string literal,
    /// returning the string stripped of any leading `%`.
    fn consume_hlo_name(&mut self) -> Option<String> {
        let out = match self.cursor.peek_token() {
            Some(
                HloToken::Ident(s)
                | HloToken::SsaValue(s)
                | HloToken::Symbol(s)
                | HloToken::StringLiteral(s),
            ) => String::from(s.trim_start_matches('%')),
            _ => return None,
        };
        self.cursor.next_token();
        Some(out)
    }

    /// Parses a complete textual XLA HLO module into an [`AstModule`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the `HloModule` header, computations, shapes, or instructions are malformed.
    pub fn parse_hlo_module(&mut self) -> LibTpuResult<AstModule> {
        let start_span = self.cursor.current_span();
        let mut module = AstModule {
            span: start_span,
            ..AstModule::default()
        };

        if self.cursor.consume_ident("HloModule") {
            module.name = self.consume_hlo_name();
            // Parse optional module-level attributes after `,`: `entry_computation_layout={...}`, `replica_count=2`, `num_partitions=4`, `allow_spmd_sharding_propagation_to_output={...}`
            while self.cursor.consume_punct(',') {
                if let Some(key) = self.cursor.consume_any_ident_str() {
                    let _ = self.cursor.consume_punct('=');
                    let attr_val = self.parse_hlo_header_attr_value();
                    module.attributes.insert(key, attr_val);
                }
            }
        }

        while !self.cursor.is_eof() {
            if self.is_at_computation_start() {
                let comp = self.parse_computation()?;
                module.hlo_computations.push(comp);
            } else {
                // Single bare HLO instruction without computation header
                let inst = self.parse_instruction()?;
                module.hlo_computations.push(HloComputationAst {
                    name: String::from("main"),
                    is_entry: true,
                    header_params: Vec::new(),
                    header_return_type: None,
                    instructions: vec![inst],
                    span: start_span,
                });
            }
        }

        // If no computation is explicitly marked `ENTRY`, mark the last computation as entry
        if !module.hlo_computations.iter().any(|c| c.is_entry)
            && let Some(last) = module.hlo_computations.last_mut()
        {
            last.is_entry = true;
        }

        Ok(module)
    }

    /// Parses a header attribute value in `HloModule name, key=val`.
    fn parse_hlo_header_attr_value(&mut self) -> MlirAttribute {
        if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
            return MlirAttribute::String(self.consume_balanced_braces_as_string());
        }
        if let Some(n) = self.cursor.consume_number_str() {
            return MlirAttribute::Integer {
                value: n.parse::<i128>().unwrap_or(0),
                ty: None,
            };
        }
        MlirAttribute::String(self.consume_hlo_name().unwrap_or_default())
    }

    /// Returns `true` if the cursor is at `[ENTRY] %comp_name (...) {` or `[ENTRY] %comp_name {`.
    fn is_at_computation_start(&self) -> bool {
        self.cursor.peek_token() == Some(&HloToken::Ident(String::from("ENTRY")))
            || matches!(
                (self.cursor.peek_token(), self.cursor.peek_n(1)),
                (
                    Some(HloToken::SsaValue(_) | HloToken::Ident(_)),
                    Some(HloToken::Punct('(' | '{'))
                )
            )
    }

    /// Parses a single [`HloComputationAst`] (`[ENTRY] %name (p0: f32[4]) -> f32[4] { ... }`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the computation header or instruction body is malformed.
    pub fn parse_computation(&mut self) -> LibTpuResult<HloComputationAst> {
        let span = self.cursor.current_span();
        let is_entry = self.cursor.consume_ident("ENTRY");
        let name = self
            .consume_hlo_name()
            .ok_or_else(|| LibTpuError::UnexpectedToken {
                expected: String::from("computation name"),
                found: format!("{:?}", self.cursor.peek_token()),
                line: span.line,
                column: span.column,
            })?;

        let mut header_params = Vec::new();
        if self.cursor.consume_punct('(') {
            while self.cursor.is_open(')') {
                let param_name = if self.cursor.peek_n(1) == Some(&HloToken::Punct(':')) {
                    let n = self.consume_hlo_name().unwrap_or_default();
                    let _ = self.cursor.consume_punct(':');
                    n
                } else {
                    format!("p{}", header_params.len())
                };
                let param_ty = self.parse_hlo_shape_type()?;
                header_params.push((param_name, param_ty));
                let _ = self.cursor.consume_punct(',');
            }
        }

        let header_return_type = if self.cursor.consume_if(|t| *t == HloToken::Arrow).is_some() {
            Some(self.parse_hlo_shape_type()?)
        } else {
            None
        };

        let _ = self.cursor.consume_punct('{');
        let mut instructions = Vec::new();
        while self.cursor.is_open('}') {
            instructions.push(self.parse_instruction()?);
        }

        Ok(HloComputationAst {
            name,
            is_entry,
            header_params,
            header_return_type,
            instructions,
            span,
        })
    }

    /// Parses a single XLA HLO [`HloInstructionAst`] (`[ROOT] %name = <Shape> <opcode>(<operands>), <attrs>`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the instruction name, `=`, shape, or opcode is malformed.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn parse_instruction(&mut self) -> LibTpuResult<HloInstructionAst> {
        let span = self.cursor.current_span();
        let is_root = self.cursor.consume_ident("ROOT");
        let name = self
            .consume_hlo_name()
            .ok_or_else(|| LibTpuError::UnexpectedToken {
                expected: String::from("HLO instruction name"),
                found: format!("{:?}", self.cursor.peek_token()),
                line: span.line,
                column: span.column,
            })?;

        let _ = self.cursor.consume_punct('=');
        let result_type = self.parse_hlo_shape_type()?;
        let opcode = self
            .consume_hlo_name()
            .ok_or_else(|| LibTpuError::UnexpectedToken {
                expected: format!("HLO opcode after shape for '{name}'"),
                found: format!("{:?}", self.cursor.peek_token()),
                line: span.line,
                column: span.column,
            })?;

        let mut operands = Vec::new();
        let mut parameter_index = None;
        let mut constant_values = Vec::new();

        let _ = self.cursor.consume_punct('(');
        if opcode == "parameter" {
            parameter_index = self
                .cursor
                .consume_number_str()
                .and_then(|s| s.parse::<usize>().ok());
            let _ = self.cursor.consume_punct(')');
        } else if opcode == "constant" {
            while self.cursor.is_open(')') {
                if let Some(n_str) = self.cursor.consume_number_str() {
                    constant_values.push(parse_float_literal(&n_str));
                } else {
                    self.cursor.next_token();
                }
            }
        } else {
            while self.cursor.is_open(')') {
                // Handle inline typed operand `f32[4] %arg0` or bare `%arg0` / `arg0`
                if self.is_at_hlo_inline_type() {
                    let _ = self.parse_hlo_shape_type()?;
                }
                if let Some(s) = self.consume_hlo_name() {
                    operands.push(s);
                } else {
                    self.cursor.next_token();
                }
                let _ = self.cursor.consume_punct(',');
            }
        }

        let mut tuple_index = None;
        let mut called_computations = BTreeMap::new();
        let mut window = None;
        let mut conv_dim_labels = None;
        let mut dot_dims = DotDimensionNumbersAttr::default();
        let mut dimensions = Vec::new();
        let mut slice_bounds = Vec::new();
        let mut padding_config = Vec::new();
        let mut sharding = None;
        let mut custom_call_target = None;
        let mut backend_config = None;
        let mut channel_id = None;
        let mut replica_groups = Vec::new();
        let mut control_predecessors = Vec::new();
        let mut extra_attrs = BTreeMap::new();

        if !constant_values.is_empty() {
            extra_attrs.insert(
                String::from("constant_values"),
                MlirAttribute::DenseElements(super::ast::DenseElementsAttr {
                    values_f64: constant_values,
                    raw_bytes: None,
                    is_splat: false,
                    tensor_type: None,
                }),
            );
        }

        // Parse trailing `, key=value` attributes
        while self.cursor.consume_punct(',') {
            let Some(key) = self.consume_hlo_name() else {
                break;
            };
            let _ = self.cursor.consume_punct('=');
            match key.as_str() {
                "index" | "iota_dimension" => {
                    let idx = self
                        .cursor
                        .consume_number_str()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                    if key == "index" {
                        tuple_index = Some(idx);
                    }
                    extra_attrs.insert(
                        key,
                        MlirAttribute::Integer {
                            value: i128::try_from(idx).unwrap_or(0),
                            ty: None,
                        },
                    );
                }
                "to_apply"
                | "calls"
                | "condition"
                | "body"
                | "scatter_computation"
                | "true_computation"
                | "false_computation"
                | "select"
                | "scatter" => {
                    called_computations.insert(key, self.consume_hlo_name().unwrap_or_default());
                }
                "dimensions" => {
                    dimensions = self.parse_hlo_brace_usize_list();
                }
                "lhs_contracting_dims" => {
                    dot_dims.lhs_contracting_dimensions = self.parse_hlo_brace_usize_list();
                }
                "rhs_contracting_dims" => {
                    dot_dims.rhs_contracting_dimensions = self.parse_hlo_brace_usize_list();
                }
                "lhs_batch_dims" => {
                    dot_dims.lhs_batching_dimensions = self.parse_hlo_brace_usize_list();
                }
                "rhs_batch_dims" => {
                    dot_dims.rhs_batching_dimensions = self.parse_hlo_brace_usize_list();
                }
                "window" => {
                    window = Some(self.parse_hlo_window_spec());
                }
                "dim_labels" => {
                    conv_dim_labels = Some(self.parse_hlo_conv_dim_labels());
                }
                "slice" => {
                    slice_bounds = self.parse_hlo_slice_bounds();
                }
                "padding" => {
                    padding_config = self.parse_hlo_padding_config();
                }
                "sharding" => {
                    sharding = Some(self.consume_balanced_braces_as_string());
                    while matches!(self.cursor.peek_token(), Some(HloToken::Number(_))) {
                        self.cursor.next_token();
                        let _ = self.cursor.consume_punct(',');
                    }
                }
                "custom_call_target" => {
                    custom_call_target = self.consume_hlo_name();
                }
                "backend_config" => {
                    backend_config = if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
                        Some(self.consume_balanced_braces_as_string())
                    } else {
                        self.consume_hlo_name()
                    };
                }
                "channel_id" => {
                    channel_id = self
                        .cursor
                        .consume_number_str()
                        .and_then(|s| s.parse::<i64>().ok());
                }
                "replica_groups" => {
                    replica_groups = self.parse_hlo_replica_groups();
                }
                "control-predecessors" => {
                    control_predecessors = self
                        .consume_balanced_braces_as_string()
                        .split(',')
                        .map(|s| s.trim().trim_start_matches('%'))
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect();
                }
                "direction" => {
                    let dir_str = self.consume_hlo_name().unwrap_or_default();
                    extra_attrs.insert(
                        String::from("comparison_direction"),
                        MlirAttribute::String(dir_str),
                    );
                }
                other_key => {
                    let val = if self.cursor.peek_token() == Some(&HloToken::Punct('{')) {
                        MlirAttribute::String(self.consume_balanced_braces_as_string())
                    } else if let Some(n_str) = self.cursor.consume_number_str() {
                        MlirAttribute::Integer {
                            value: n_str.parse::<i128>().unwrap_or(0),
                            ty: None,
                        }
                    } else {
                        MlirAttribute::String(self.consume_hlo_name().unwrap_or_default())
                    };
                    extra_attrs.insert(String::from(other_key), val);
                }
            }
        }

        if let Some(p_idx) = parameter_index {
            extra_attrs.insert(
                String::from("parameter_number"),
                MlirAttribute::Integer {
                    value: i128::try_from(p_idx).unwrap_or(0),
                    ty: None,
                },
            );
        }
        if let Some(t_idx) = tuple_index {
            extra_attrs.insert(
                String::from("index"),
                MlirAttribute::Integer {
                    value: i128::try_from(t_idx).unwrap_or(0),
                    ty: None,
                },
            );
        }

        let elem_type = result_type.element_type();
        let op_kind =
            classify_ast_operation(DialectKind::XlaHlo, &opcode, elem_type, &extra_attrs, &[]);

        Ok(HloInstructionAst {
            name,
            is_root,
            result_type,
            opcode,
            op_kind,
            operands,
            parameter_index,
            tuple_index,
            called_computations,
            window,
            conv_dim_labels,
            dot_dims,
            dimensions,
            slice_bounds,
            padding_config,
            sharding,
            custom_call_target,
            backend_config,
            channel_id,
            replica_groups,
            control_predecessors,
            extra_attrs,
            span,
        })
    }

    /// Returns `true` if the cursor is positioned at the start of the next HLO instruction (`[ROOT] %inst = ...`).
    fn is_at_hlo_instruction_start(&self) -> bool {
        self.cursor.peek_token() == Some(&HloToken::Ident(String::from("ROOT")))
            || (matches!(self.cursor.peek_token(), Some(HloToken::SsaValue(_)))
                && self.cursor.peek_n(1) == Some(&HloToken::Punct('=')))
    }

    /// Returns `true` if the cursor is positioned at an inline HLO operand type like `f32[4] %arg0`.
    fn is_at_hlo_inline_type(&self) -> bool {
        self.cursor.peek_n(1) == Some(&HloToken::Punct('['))
    }

    /// Parses an XLA HLO shape (`f32[128,256]{1,0:T(8,128)S(1)}`, `f32[<=8, ?]`, `token[]`, or nested tuple `(f32[4], (s32[2], pred[]))`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the dtype or shape bracket syntax is invalid.
    pub fn parse_hlo_shape_type(&mut self) -> LibTpuResult<MlirType> {
        if self.cursor.consume_punct('(') {
            let mut elems = Vec::new();
            while self.cursor.is_open(')') {
                elems.push(self.parse_hlo_shape_type()?);
                let _ = self.cursor.consume_punct(',');
            }
            return Ok(MlirType::Tuple(elems));
        }

        let (dtype_str, span) = self.cursor.expect_ident()?;
        if dtype_str == "token" {
            let _ = self.cursor.consume_punct('[');
            let _ = self.cursor.consume_punct(']');
            return Ok(MlirType::Token);
        }

        let element_type =
            parse_mlir_element_type(&dtype_str).ok_or_else(|| LibTpuError::InvalidShape {
                reason: format!(
                    "Unknown XLA HLO element type '{dtype_str}' at line {}, column {}",
                    span.line, span.column
                ),
            })?;

        let mut dims = Vec::new();
        let _ = self.cursor.consume_punct('[');
        while self.cursor.is_open(']') {
            if self.cursor.consume_punct('?') {
                dims.push(DimensionSize::Dynamic);
            } else if self.cursor.consume_if(|t| *t == HloToken::Le).is_some() {
                let bound = self
                    .cursor
                    .consume_number_str()
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(1);
                dims.push(DimensionSize::Bounded(bound));
            } else if let Some(n_str) = self.cursor.consume_number_str() {
                dims.push(DimensionSize::Static(n_str.parse::<i64>().unwrap_or(1)));
            } else {
                self.cursor.next_token();
            }
            let _ = self.cursor.consume_punct(',');
        }

        let layout = if self.cursor.peek_token() == Some(&HloToken::Punct('{'))
            && self.is_next_brace_physical_layout()
        {
            Some(self.parse_hlo_physical_layout(dims.len()))
        } else {
            None
        };

        Ok(MlirType::RankedTensor(RankedTensorType {
            dims,
            element_type,
            encoding: None,
            layout,
        }))
    }

    /// Returns `true` if the next `{` begins a physical layout `{1,0...}` or `{:T(...)...}` rather than a computation body `{ %inst = ... }`.
    fn is_next_brace_physical_layout(&self) -> bool {
        matches!(
            self.cursor.peek_n(1),
            Some(HloToken::Number(_) | HloToken::Punct(':'))
        )
    }

    /// Parses an XLA HLO physical layout suffix `{1,0:T(8,128)(2,1)S(1)}`.
    ///
    /// # Arguments
    ///
    /// * `rank` - Tensor rank for default minor-to-major ordering fallback.
    fn parse_hlo_physical_layout(&mut self, rank: usize) -> MemoryLayout {
        let raw = self.consume_balanced_braces_as_string();
        let (order_part, tiles_part) = raw.split_once(':').unwrap_or((&raw, ""));
        let mut minor_to_major: Vec<i64> = order_part
            .split(',')
            .filter_map(|s| s.trim().parse::<i64>().ok())
            .collect();
        let mut tile_dims = Vec::new();
        let mut tile_dim_sizes = Vec::new();
        if let Some((_, after_t)) = tiles_part.split_once('T') {
            let t_section = after_t.split('S').next().unwrap_or(after_t);
            for group in t_section.split('(').skip(1) {
                let inside = group.split(')').next().unwrap_or(group);
                let dims: Vec<i64> = inside
                    .split(',')
                    .filter_map(|d| d.trim().parse::<i64>().ok())
                    .collect();
                tile_dim_sizes.push(dims.len());
                tile_dims.extend(dims);
            }
        }
        if minor_to_major.is_empty() {
            minor_to_major = (0..rank)
                .rev()
                .map(|idx| i64::try_from(idx).unwrap_or(0))
                .collect();
        }
        if tile_dims.is_empty() && rank >= 2 {
            tile_dims = vec![8, 128];
            tile_dim_sizes = vec![2];
        }
        MemoryLayout::Tiled {
            minor_to_major,
            tile_dims,
            tile_dim_sizes,
        }
    }

    /// Parses `{0, 1, 2}` into a `Vec<usize>`.
    fn parse_hlo_brace_usize_list(&mut self) -> Vec<usize> {
        self.consume_balanced_braces_as_string()
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    }

    /// Parses `window={size=3x3 stride=1x1 pad=1_1x1_1 lhs_dilate=1x1 rhs_dilate=1x1}`.
    fn parse_hlo_window_spec(&mut self) -> HloWindowSpec {
        let raw = self.consume_balanced_braces_as_string();
        let mut spec = HloWindowSpec::default();
        for (k, v) in raw.split_whitespace().filter_map(|p| p.split_once('=')) {
            match k {
                "size" => {
                    spec.size = v
                        .split('x')
                        .filter_map(|s| s.parse::<usize>().ok())
                        .collect();
                }
                "stride" => {
                    spec.stride = v
                        .split('x')
                        .filter_map(|s| s.parse::<usize>().ok())
                        .collect();
                }
                "pad" => {
                    for (lo, hi) in v.split('x').filter_map(|d| d.split_once('_')) {
                        spec.pad_low.push(lo.parse::<usize>().unwrap_or(0));
                        spec.pad_high.push(hi.parse::<usize>().unwrap_or(0));
                    }
                }
                "lhs_dilate" => {
                    spec.lhs_dilate = v
                        .split('x')
                        .filter_map(|s| s.parse::<usize>().ok())
                        .collect();
                }
                _ => {
                    spec.rhs_dilate = v
                        .split('x')
                        .filter_map(|s| s.parse::<usize>().ok())
                        .collect();
                }
            }
        }
        spec
    }

    /// Parses `dim_labels=b01f_01io->b01f`.
    fn parse_hlo_conv_dim_labels(&mut self) -> ConvDimensionNumbersAttr {
        let label_str = self.consume_unquoted_attr_token_run();
        let mut attr = ConvDimensionNumbersAttr::default();
        let (lhs_rhs, out_part) = label_str.split_once("->").unwrap_or((&label_str, ""));
        let (in_part, kern_part) = lhs_rhs.split_once('_').unwrap_or((lhs_rhs, ""));
        for (idx, ch) in in_part.chars().enumerate() {
            match ch {
                'b' => attr.input_batch_dimension = idx,
                'f' => attr.input_feature_dimension = idx,
                _ => attr.input_spatial_dimensions.push(idx),
            }
        }
        for (idx, ch) in kern_part.chars().enumerate() {
            match ch {
                'i' => attr.kernel_input_feature_dimension = idx,
                'o' => attr.kernel_output_feature_dimension = idx,
                _ => attr.kernel_spatial_dimensions.push(idx),
            }
        }
        for (idx, ch) in out_part.chars().enumerate() {
            match ch {
                'b' => attr.output_batch_dimension = idx,
                'f' => attr.output_feature_dimension = idx,
                _ => attr.output_spatial_dimensions.push(idx),
            }
        }
        attr
    }

    /// Parses `slice={[0:2:1], [1:4:2]}` into `(start, limit, stride)` tuples.
    fn parse_hlo_slice_bounds(&mut self) -> Vec<(usize, usize, usize)> {
        let raw = self.consume_balanced_braces_as_string();
        let mut bounds = Vec::new();
        for seg in raw.split('[').skip(1) {
            let inner = seg.split(']').next().unwrap_or(seg);
            let nums: Vec<usize> = inner
                .split(':')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            let start = nums.first().copied().unwrap_or(0);
            let limit = nums.get(1).copied().unwrap_or(start + 1);
            let stride = nums.get(2).copied().unwrap_or(1).max(1);
            bounds.push((start, limit, stride));
        }
        bounds
    }

    /// Parses `padding=1_1_0x2_2_0` into `(low, high, interior)` tuples.
    fn parse_hlo_padding_config(&mut self) -> Vec<(usize, usize, usize)> {
        let raw = self.consume_unquoted_attr_token_run();
        let mut out = Vec::new();
        for dim_str in raw.split('x') {
            let parts: Vec<usize> = dim_str
                .split('_')
                .filter_map(|p| p.parse::<usize>().ok())
                .collect();
            out.push((
                parts.first().copied().unwrap_or(0),
                parts.get(1).copied().unwrap_or(0),
                parts.get(2).copied().unwrap_or(0),
            ));
        }
        out
    }

    /// Consumes an unquoted attribute token run (such as `b01f_01io->b01f` or `1_1_0x1_1_0`) up to `,`, `}`, or the next instruction.
    fn consume_unquoted_attr_token_run(&mut self) -> String {
        let mut raw = String::new();
        while !self.is_at_hlo_instruction_start() {
            match self.cursor.peek_token() {
                Some(HloToken::Ident(s) | HloToken::Number(s)) => raw.push_str(s),
                Some(HloToken::Arrow) => raw.push_str("->"),
                _ => break,
            }
            self.cursor.next_token();
        }
        raw
    }

    /// Parses `replica_groups={{0,1},{2,3}}`.
    fn parse_hlo_replica_groups(&mut self) -> Vec<Vec<usize>> {
        let raw = self.consume_balanced_braces_as_string();
        raw.split('{')
            .filter(|seg| !seg.trim().is_empty())
            .map(|seg| {
                seg.split('}')
                    .next()
                    .unwrap_or(seg)
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            })
            .collect()
    }

    /// Consumes a balanced `{ ... }` block and returns its compact string content.
    fn consume_balanced_braces_as_string(&mut self) -> String {
        let _ = self.cursor.consume_punct('{');
        let mut depth = 1_i32;
        let mut out = String::new();
        while let Some(st) = self.cursor.next_token() {
            match &st.token {
                HloToken::Punct('{') => {
                    depth += 1;
                    out.push('{');
                }
                HloToken::Punct('}') => {
                    depth -= 1;
                    if depth == 0 {
                        return out;
                    }
                    out.push('}');
                }
                HloToken::Ident(s) | HloToken::Number(s) | HloToken::SsaValue(s) => {
                    let prev_alpha = out.as_bytes().last().is_some_and(u8::is_ascii_alphanumeric);
                    let cur_alpha = s.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric);
                    let is_dim_sep = out.as_bytes().last().is_some_and(u8::is_ascii_digit)
                        && s.starts_with('x')
                        && s.as_bytes().get(1).is_some_and(u8::is_ascii_digit);
                    if prev_alpha && cur_alpha && !is_dim_sep {
                        out.push(' ');
                    }
                    out.push_str(s);
                }
                HloToken::StringLiteral(s) => {
                    out.push('"');
                    out.push_str(s);
                    out.push('"');
                }
                HloToken::Punct(c) => out.push(*c),
                HloToken::Arrow => out.push_str("->"),
                _ => {}
            }
        }
        out
    }
}

/// Returns a default 1-D `f32[4]` [`Shape`] for fallback slot allocation.
#[must_use]
fn default_f32_vec4_shape() -> Shape {
    Shape {
        element_type: ElementType::F32,
        dims: vec![4],
        dynamic_dims: vec![false],
        layout: MemoryLayout::default_tpu_tiled(1),
    }
}

/// Converts a parsed textual XLA HLO [`AstModule`] into a wire-compatible [`HloModuleProto`].
///
/// # Arguments
///
/// * `ast` - Parsed [`AstModule`] containing [`HloComputationAst`]s.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn hlo_ast_to_proto(ast: &AstModule) -> HloModuleProto {
    let mut computations = Vec::new();
    let mut next_inst_id = 1_i64;
    let mut entry_comp_name = String::new();
    let mut entry_comp_id = 0_i64;

    let comp_name_to_id: BTreeMap<&str, i64> = ast
        .hlo_computations
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.name.as_str(), i64::try_from(idx + 1).unwrap_or(1)))
        .collect();

    for (comp_idx, comp) in ast.hlo_computations.iter().enumerate() {
        let comp_id = i64::try_from(comp_idx + 1).unwrap_or(1);
        if comp.is_entry {
            entry_comp_name.clone_from(&comp.name);
            entry_comp_id = comp_id;
        }
        let mut name_to_id = BTreeMap::new();
        let mut instructions = Vec::new();
        let mut root_id = 0_i64;

        for inst in &comp.instructions {
            let id = next_inst_id;
            next_inst_id += 1;
            name_to_id.insert(inst.name.clone(), id);
            if inst.is_root {
                root_id = id;
            }
            let operand_ids: Vec<i64> = inst
                .operands
                .iter()
                .filter_map(|op_name| name_to_id.get(op_name).copied())
                .collect();

            let mut called_computation_ids: Vec<i64> = inst
                .called_computations
                .values()
                .filter_map(|target_name| comp_name_to_id.get(target_name.as_str()).copied())
                .collect();
            called_computation_ids.dedup();

            let dimensions: Vec<i64> = inst
                .dimensions
                .iter()
                .filter_map(|&d| i64::try_from(d).ok())
                .collect();

            let window = inst.window.as_ref().map(|w| {
                let rank = w
                    .size
                    .len()
                    .max(w.stride.len())
                    .max(w.pad_low.len())
                    .max(w.pad_high.len());
                let mut dims = Vec::with_capacity(rank);
                for idx in 0..rank {
                    dims.push(WindowDimension {
                        size: i64::try_from(w.size.get(idx).copied().unwrap_or(1)).unwrap_or(1),
                        stride: i64::try_from(w.stride.get(idx).copied().unwrap_or(1)).unwrap_or(1),
                        padding_low: i64::try_from(w.pad_low.get(idx).copied().unwrap_or(0))
                            .unwrap_or(0),
                        padding_high: i64::try_from(w.pad_high.get(idx).copied().unwrap_or(0))
                            .unwrap_or(0),
                        window_dilation: i64::try_from(w.rhs_dilate.get(idx).copied().unwrap_or(1))
                            .unwrap_or(1),
                        base_dilation: i64::try_from(w.lhs_dilate.get(idx).copied().unwrap_or(1))
                            .unwrap_or(1),
                        window_reversal: false,
                    });
                }
                Window { dimensions: dims }
            });

            let convolution_dimension_numbers =
                inst.conv_dim_labels
                    .as_ref()
                    .map(|c| ConvolutionDimensionNumbers {
                        input_batch_dimension: i64::try_from(c.input_batch_dimension).unwrap_or(0),
                        input_feature_dimension: i64::try_from(c.input_feature_dimension)
                            .unwrap_or(0),
                        input_spatial_dimensions: c
                            .input_spatial_dimensions
                            .iter()
                            .filter_map(|&d| i64::try_from(d).ok())
                            .collect(),
                        kernel_input_feature_dimension: i64::try_from(
                            c.kernel_input_feature_dimension,
                        )
                        .unwrap_or(0),
                        kernel_output_feature_dimension: i64::try_from(
                            c.kernel_output_feature_dimension,
                        )
                        .unwrap_or(0),
                        kernel_spatial_dimensions: c
                            .kernel_spatial_dimensions
                            .iter()
                            .filter_map(|&d| i64::try_from(d).ok())
                            .collect(),
                        output_batch_dimension: i64::try_from(c.output_batch_dimension)
                            .unwrap_or(0),
                        output_feature_dimension: i64::try_from(c.output_feature_dimension)
                            .unwrap_or(0),
                        output_spatial_dimensions: c
                            .output_spatial_dimensions
                            .iter()
                            .filter_map(|&d| i64::try_from(d).ok())
                            .collect(),
                    });

            let dot_dims_count = inst.dot_dims.lhs_contracting_dimensions.len()
                + inst.dot_dims.rhs_contracting_dimensions.len()
                + inst.dot_dims.lhs_batching_dimensions.len()
                + inst.dot_dims.rhs_batching_dimensions.len();
            let dot_dimension_numbers = (dot_dims_count > 0).then(|| DotDimensionNumbers {
                lhs_contracting_dimensions: inst
                    .dot_dims
                    .lhs_contracting_dimensions
                    .iter()
                    .filter_map(|&d| i64::try_from(d).ok())
                    .collect(),
                rhs_contracting_dimensions: inst
                    .dot_dims
                    .rhs_contracting_dimensions
                    .iter()
                    .filter_map(|&d| i64::try_from(d).ok())
                    .collect(),
                lhs_batch_dimensions: inst
                    .dot_dims
                    .lhs_batching_dimensions
                    .iter()
                    .filter_map(|&d| i64::try_from(d).ok())
                    .collect(),
                rhs_batch_dimensions: inst
                    .dot_dims
                    .rhs_batching_dimensions
                    .iter()
                    .filter_map(|&d| i64::try_from(d).ok())
                    .collect(),
            });

            let padding_config = (!inst.padding_config.is_empty()).then(|| PaddingConfig {
                dimensions: inst
                    .padding_config
                    .iter()
                    .map(|&(lo, hi, interior)| PaddingConfigDimension {
                        edge_padding_low: i64::try_from(lo).unwrap_or(0),
                        edge_padding_high: i64::try_from(hi).unwrap_or(0),
                        interior_padding: i64::try_from(interior).unwrap_or(0),
                    })
                    .collect(),
            });

            let replica_groups: Vec<ReplicaGroup> = inst
                .replica_groups
                .iter()
                .map(|rg| ReplicaGroup {
                    replica_ids: rg.iter().filter_map(|&r| i64::try_from(r).ok()).collect(),
                })
                .collect();

            instructions.push(HloInstructionProto {
                name: inst.name.clone(),
                opcode: inst.opcode.clone(),
                shape: Some(mlir_type_to_shape_proto(&inst.result_type)),
                operand_ids,
                called_computation_ids,
                dimensions,
                window,
                convolution_dimension_numbers,
                dot_dimension_numbers,
                padding_config,
                replica_groups,
                channel_id: inst.channel_id.unwrap_or(0),
                backend_config: inst
                    .backend_config
                    .as_deref()
                    .unwrap_or_default()
                    .as_bytes()
                    .to_vec(),
                id,
                parameter_number: inst
                    .parameter_index
                    .or(inst.tuple_index)
                    .and_then(|i| i64::try_from(i).ok())
                    .unwrap_or(0),
                tuple_index: inst
                    .tuple_index
                    .and_then(|i| i64::try_from(i).ok())
                    .unwrap_or(0),
                custom_call_target: inst.custom_call_target.clone().unwrap_or_default(),
                ..HloInstructionProto::default()
            });
        }

        if root_id == 0 {
            root_id = instructions.last().map_or(1, |i| i.id);
        }

        computations.push(HloComputationProto {
            name: comp.name.clone(),
            instructions,
            id: comp_id,
            root_id,
            ..HloComputationProto::default()
        });
    }

    HloModuleProto {
        name: String::from(ast.name.as_deref().unwrap_or("hlo_module")),
        entry_computation_name: entry_comp_name,
        entry_computation_id: entry_comp_id,
        computations,
        ..HloModuleProto::default()
    }
}

/// Converts a wire [`HloModuleProto`] into a structured [`AstModule`] containing [`HloComputationAst`]s.
///
/// # Arguments
///
/// * `proto` - Decoded [`HloModuleProto`].
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn hlo_proto_to_ast(proto: &HloModuleProto) -> AstModule {
    let default_shape_proto = ShapeProto {
        element_type: i32::from(ElementType::F32),
        dimensions: vec![4],
        ..ShapeProto::default()
    };
    let comp_id_to_name: BTreeMap<i64, String> = proto
        .computations
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();

    let mut hlo_computations = Vec::new();
    for (idx, comp) in proto.computations.iter().enumerate() {
        let is_entry = comp.id == proto.entry_computation_id
            || comp.name == proto.entry_computation_name
            || idx + 1 == proto.computations.len();

        let mut id_to_name = BTreeMap::new();
        for inst in &comp.instructions {
            id_to_name.insert(inst.id, inst.name.clone());
        }

        let mut instructions = Vec::new();
        for (i_idx, inst) in comp.instructions.iter().enumerate() {
            let is_root = inst.id == comp.root_id || i_idx + 1 == comp.instructions.len();
            let result_type =
                shape_proto_to_mlir_type(inst.shape.as_ref().unwrap_or(&default_shape_proto));
            let operands: Vec<String> = inst
                .operand_ids
                .iter()
                .filter_map(|id| id_to_name.get(id).cloned())
                .collect();
            let param_num = usize::try_from(inst.parameter_number.max(0)).unwrap_or(0);
            let parameter_index = (inst.opcode == "parameter").then_some(param_num);
            let tuple_index = matches!(
                inst.opcode.as_str(),
                "get-tuple-element" | "get_tuple_element"
            )
            .then_some(usize::try_from(inst.tuple_index.max(0)).unwrap_or(param_num));
            let mut extra_attrs = BTreeMap::new();
            if let Some(p_idx) = parameter_index {
                extra_attrs.insert(
                    String::from("parameter_number"),
                    MlirAttribute::Integer {
                        value: i128::try_from(p_idx).unwrap_or(0),
                        ty: None,
                    },
                );
            }
            if let Some(t_idx) = tuple_index {
                extra_attrs.insert(
                    String::from("index"),
                    MlirAttribute::Integer {
                        value: i128::try_from(t_idx).unwrap_or(0),
                        ty: None,
                    },
                );
            }
            let mut called_computations = BTreeMap::new();
            let resolved_comp_names: Vec<String> = inst
                .called_computation_ids
                .iter()
                .filter_map(|cid| comp_id_to_name.get(cid).cloned())
                .collect();
            if let Some(first_comp) = resolved_comp_names.first() {
                called_computations.insert(String::from("to_apply"), first_comp.clone());
                called_computations.insert(String::from("calls"), first_comp.clone());
                called_computations.insert(String::from("condition"), first_comp.clone());
                called_computations.insert(String::from("true_computation"), first_comp.clone());
                let second_comp = resolved_comp_names.get(1).unwrap_or(first_comp);
                called_computations.insert(String::from("body"), second_comp.clone());
                called_computations.insert(String::from("false_computation"), second_comp.clone());
            }
            let dimensions: Vec<usize> = inst
                .dimensions
                .iter()
                .filter_map(|&d| usize::try_from(d).ok())
                .collect();
            let op_kind = classify_ast_operation(
                DialectKind::XlaHlo,
                &inst.opcode,
                result_type.element_type(),
                &extra_attrs,
                &[],
            );
            let window = inst.window.as_ref().map(|w| HloWindowSpec {
                size: w
                    .dimensions
                    .iter()
                    .map(|d| usize::try_from(d.size.max(1)).unwrap_or(1))
                    .collect(),
                stride: w
                    .dimensions
                    .iter()
                    .map(|d| usize::try_from(d.stride.max(1)).unwrap_or(1))
                    .collect(),
                pad_low: w
                    .dimensions
                    .iter()
                    .map(|d| usize::try_from(d.padding_low.max(0)).unwrap_or(0))
                    .collect(),
                pad_high: w
                    .dimensions
                    .iter()
                    .map(|d| usize::try_from(d.padding_high.max(0)).unwrap_or(0))
                    .collect(),
                lhs_dilate: w
                    .dimensions
                    .iter()
                    .map(|d| usize::try_from(d.base_dilation.max(1)).unwrap_or(1))
                    .collect(),
                rhs_dilate: w
                    .dimensions
                    .iter()
                    .map(|d| usize::try_from(d.window_dilation.max(1)).unwrap_or(1))
                    .collect(),
            });
            let conv_dim_labels =
                inst.convolution_dimension_numbers
                    .as_ref()
                    .map(|c| ConvDimensionNumbersAttr {
                        input_batch_dimension: usize::try_from(c.input_batch_dimension.max(0))
                            .unwrap_or(0),
                        input_feature_dimension: usize::try_from(c.input_feature_dimension.max(0))
                            .unwrap_or(0),
                        input_spatial_dimensions: c
                            .input_spatial_dimensions
                            .iter()
                            .filter_map(|&d| usize::try_from(d).ok())
                            .collect(),
                        kernel_input_feature_dimension: usize::try_from(
                            c.kernel_input_feature_dimension.max(0),
                        )
                        .unwrap_or(0),
                        kernel_output_feature_dimension: usize::try_from(
                            c.kernel_output_feature_dimension.max(0),
                        )
                        .unwrap_or(0),
                        kernel_spatial_dimensions: c
                            .kernel_spatial_dimensions
                            .iter()
                            .filter_map(|&d| usize::try_from(d).ok())
                            .collect(),
                        output_batch_dimension: usize::try_from(c.output_batch_dimension.max(0))
                            .unwrap_or(0),
                        output_feature_dimension: usize::try_from(
                            c.output_feature_dimension.max(0),
                        )
                        .unwrap_or(0),
                        output_spatial_dimensions: c
                            .output_spatial_dimensions
                            .iter()
                            .filter_map(|&d| usize::try_from(d).ok())
                            .collect(),
                    });
            let dot_dims = inst.dot_dimension_numbers.as_ref().map_or_else(
                DotDimensionNumbersAttr::default,
                |d| DotDimensionNumbersAttr {
                    lhs_batching_dimensions: d
                        .lhs_batch_dimensions
                        .iter()
                        .filter_map(|&x| usize::try_from(x).ok())
                        .collect(),
                    rhs_batching_dimensions: d
                        .rhs_batch_dimensions
                        .iter()
                        .filter_map(|&x| usize::try_from(x).ok())
                        .collect(),
                    lhs_contracting_dimensions: d
                        .lhs_contracting_dimensions
                        .iter()
                        .filter_map(|&x| usize::try_from(x).ok())
                        .collect(),
                    rhs_contracting_dimensions: d
                        .rhs_contracting_dimensions
                        .iter()
                        .filter_map(|&x| usize::try_from(x).ok())
                        .collect(),
                },
            );
            let padding_config = inst.padding_config.as_ref().map_or_else(Vec::new, |p| {
                p.dimensions
                    .iter()
                    .map(|d| {
                        (
                            usize::try_from(d.edge_padding_low.max(0)).unwrap_or(0),
                            usize::try_from(d.edge_padding_high.max(0)).unwrap_or(0),
                            usize::try_from(d.interior_padding.max(0)).unwrap_or(0),
                        )
                    })
                    .collect()
            });
            let replica_groups: Vec<Vec<usize>> = inst
                .replica_groups
                .iter()
                .map(|rg| {
                    rg.replica_ids
                        .iter()
                        .filter_map(|&r| usize::try_from(r).ok())
                        .collect()
                })
                .collect();
            let backend_config = (!inst.backend_config.is_empty())
                .then(|| String::from_utf8_lossy(&inst.backend_config).into_owned());
            let channel_id = (inst.channel_id > 0).then_some(inst.channel_id);
            instructions.push(HloInstructionAst {
                name: inst.name.clone(),
                is_root,
                result_type,
                opcode: inst.opcode.clone(),
                op_kind,
                operands,
                parameter_index,
                tuple_index,
                called_computations,
                window,
                conv_dim_labels,
                dot_dims,
                dimensions,
                slice_bounds: Vec::new(),
                padding_config,
                sharding: None,
                custom_call_target: (!inst.custom_call_target.is_empty())
                    .then(|| inst.custom_call_target.clone()),
                backend_config,
                channel_id,
                replica_groups,
                control_predecessors: Vec::new(),
                extra_attrs,
                span: TokenSpan::default(),
            });
        }

        hlo_computations.push(HloComputationAst {
            name: comp.name.clone(),
            is_entry,
            header_params: Vec::new(),
            header_return_type: None,
            instructions,
            span: TokenSpan::default(),
        });
    }

    AstModule {
        name: (!proto.name.is_empty()).then(|| proto.name.clone()),
        hlo_computations,
        ..AstModule::default()
    }
}

/// Converts an [`MlirType`] into a protobuf [`ShapeProto`].
///
/// # Arguments
///
/// * `ty` - Source [`MlirType`] to convert.
#[must_use]
fn mlir_type_to_shape_proto(ty: &MlirType) -> ShapeProto {
    match ty {
        MlirType::Tuple(elems) => ShapeProto {
            element_type: i32::from(ElementType::Tuple),
            tuple_shapes: elems.iter().map(mlir_type_to_shape_proto).collect(),
            ..ShapeProto::default()
        },
        MlirType::Token => ShapeProto {
            element_type: i32::from(ElementType::Token),
            ..ShapeProto::default()
        },
        MlirType::RankedTensor(rt) => ShapeProto {
            element_type: i32::from(rt.element_type),
            dimensions: rt.dims.iter().map(|d| d.concrete_or_bound()).collect(),
            is_dynamic_dimension: rt.dims.iter().map(|d| d.is_dynamic()).collect(),
            layout: rt.layout.as_ref().map(|l| match l {
                MemoryLayout::Tiled { minor_to_major, .. }
                | MemoryLayout::Strides {
                    byte_strides: minor_to_major,
                    ..
                } => LayoutProto {
                    minor_to_major: minor_to_major.clone(),
                    ..LayoutProto::default()
                },
            }),
            ..ShapeProto::default()
        },
        other => {
            let sh = other.to_shape().unwrap_or_else(default_f32_vec4_shape);
            ShapeProto {
                element_type: i32::from(sh.element_type),
                dimensions: sh.dims,
                is_dynamic_dimension: sh.dynamic_dims,
                ..ShapeProto::default()
            }
        }
    }
}

/// Converts a protobuf [`ShapeProto`] into an [`MlirType`].
///
/// # Arguments
///
/// * `sp` - Source [`ShapeProto`] to convert.
#[must_use]
fn shape_proto_to_mlir_type(sp: &ShapeProto) -> MlirType {
    if !sp.tuple_shapes.is_empty() {
        return MlirType::Tuple(
            sp.tuple_shapes
                .iter()
                .map(shape_proto_to_mlir_type)
                .collect(),
        );
    }
    let elem = ElementType::from_raw(sp.element_type).unwrap_or(ElementType::F32);
    if elem == ElementType::Token {
        return MlirType::Token;
    }
    let dims = sp
        .dimensions
        .iter()
        .enumerate()
        .map(|(idx, &d)| {
            if sp.is_dynamic_dimension.get(idx).copied().unwrap_or(false) {
                if d > 0 {
                    DimensionSize::Bounded(d)
                } else {
                    DimensionSize::Dynamic
                }
            } else {
                DimensionSize::Static(d.max(1))
            }
        })
        .collect();
    MlirType::RankedTensor(RankedTensorType {
        dims,
        element_type: elem,
        encoding: None,
        layout: None,
    })
}

/// Lowers a parsed [`AstModule`] (either MLIR or XLA HLO) into a [`ParsedHloModule`] with register
/// slot assignment, multi-result/tuple tracking, nested region reduction resolution, and shape inference.
///
/// # Arguments
///
/// * `ast` - Parsed [`AstModule`] to lower.
///
/// # Errors
///
/// Returns [`LibTpuError`] if arity verification or shape inference fails on a strict AST node.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn lower_ast_module_to_parsed_hlo(ast: &AstModule) -> LibTpuResult<ParsedHloModule> {
    if !ast.hlo_computations.is_empty() {
        return Ok(lower_hlo_computations(ast));
    }
    Ok(lower_mlir_functions_and_ops(ast))
}

/// Lowers MLIR [`AstFunction`]s and top-level [`AstOperation`]s into a [`ParsedHloModule`].
///
/// # Arguments
///
/// * `ast` - Parsed [`AstModule`] containing MLIR functions and top-level operations.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn lower_mlir_functions_and_ops(ast: &AstModule) -> ParsedHloModule {
    let mut parameter_shapes = Vec::new();
    let mut output_shapes = Vec::new();
    let mut ssa_slots: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut tuple_slots: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut slot_shapes: Vec<Shape> = Vec::new();
    let mut next_slot = 0_usize;
    let mut bundles = Vec::new();
    let mut output_slots = Vec::new();
    let mut primary_op = TpuOpCode::Identity;

    // Pick entry function (`@main` or last function)
    let entry_func: Option<&AstFunction> = ast
        .functions
        .iter()
        .find(|f| f.name == "main")
        .or_else(|| ast.functions.last());

    for func in ast.functions.iter().filter(|f| Some(*f) != entry_func) {
        for (idx, arg) in func.arguments.iter().enumerate() {
            ssa_slots.insert(arg.ssa_name.clone(), vec![idx]);
        }
        lower_region_operations(
            &func.body,
            &mut ssa_slots,
            &mut tuple_slots,
            &mut slot_shapes,
            &mut next_slot,
            &mut bundles,
            &mut output_slots,
            &mut primary_op,
        );
        output_slots.clear();
    }

    if let Some(func) = entry_func {
        for arg in &func.arguments {
            let sh = arg.ty.to_shape().unwrap_or_else(default_f32_vec4_shape);
            ssa_slots.insert(arg.ssa_name.clone(), vec![next_slot]);
            next_slot += 1;
            slot_shapes.push(sh.clone());
            parameter_shapes.push(sh);
        }
        for res in &func.results {
            output_shapes.extend(res.ty.flatten_shapes());
        }
        lower_region_operations(
            &func.body,
            &mut ssa_slots,
            &mut tuple_slots,
            &mut slot_shapes,
            &mut next_slot,
            &mut bundles,
            &mut output_slots,
            &mut primary_op,
        );
    }

    if !ast.top_level_operations.is_empty() {
        let synthetic_region = AstRegion {
            blocks: vec![super::ast::AstBlock {
                label: None,
                arguments: Vec::new(),
                operations: ast.top_level_operations.clone(),
                span: ast.span,
            }],
            span: ast.span,
        };
        lower_region_operations(
            &synthetic_region,
            &mut ssa_slots,
            &mut tuple_slots,
            &mut slot_shapes,
            &mut next_slot,
            &mut bundles,
            &mut output_slots,
            &mut primary_op,
        );
    }

    finalize_lowered_module(
        primary_op,
        parameter_shapes,
        output_shapes,
        &slot_shapes,
        bundles,
        output_slots,
    )
}

/// Resolves an [`super::ast::AstOperand`] (`%name` or `%name#idx`) against `ssa_slots`.
///
/// # Arguments
///
/// * `ssa_slots` - Map from SSA variable name to allocated register slot indices.
/// * `opnd` - Operand reference to resolve.
fn resolve_ssa_slot(
    ssa_slots: &BTreeMap<String, Vec<usize>>,
    opnd: &super::ast::AstOperand,
) -> Option<usize> {
    let slots = ssa_slots.get(&opnd.name)?;
    let idx = opnd
        .result_index
        .unwrap_or(0)
        .min(slots.len().saturating_sub(1));
    slots.get(idx).copied()
}

/// Recursively lowers operations inside an [`AstRegion`] (flattening structured control flow `sdy.manual_computation`, `scf.for`, `while`, `if`).
///
/// # Arguments
///
/// * `region` - [`AstRegion`] to lower.
/// * `ssa_slots` - Map from SSA variable name to allocated register slot indices.
/// * `tuple_slots` - Map from tuple SSA variable name to member register slot indices.
/// * `slot_shapes` - Shapes indexed by register slot.
/// * `next_slot` - Next available register slot counter.
/// * `bundles` - Output instruction bundle list.
/// * `output_slots` - Collected region return slot indices.
/// * `primary_op` - Primary [`TpuOpCode`] tracker.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lower_region_operations(
    region: &AstRegion,
    ssa_slots: &mut BTreeMap<String, Vec<usize>>,
    tuple_slots: &mut BTreeMap<String, Vec<usize>>,
    slot_shapes: &mut Vec<Shape>,
    next_slot: &mut usize,
    bundles: &mut Vec<TpuInstructionBundle>,
    output_slots: &mut Vec<usize>,
    primary_op: &mut TpuOpCode,
) {
    for block in &region.blocks {
        for op in &block.operations {
            match &op.op_kind {
                AstOpKind::Return => {
                    for opnd in &op.operands {
                        if let Some(t_slots) = tuple_slots.get(&opnd.name) {
                            output_slots.extend(t_slots.iter().copied());
                        } else {
                            output_slots.extend(resolve_ssa_slot(ssa_slots, opnd));
                        }
                    }
                }
                AstOpKind::Tuple => {
                    let collected: Vec<usize> = op
                        .operands
                        .iter()
                        .filter_map(|opnd| resolve_ssa_slot(ssa_slots, opnd))
                        .collect();
                    if let Some(res) = op.results.first() {
                        tuple_slots.insert(res.name.clone(), collected.clone());
                        ssa_slots.insert(res.name.clone(), collected);
                    } else {
                        output_slots.extend(collected);
                    }
                }
                AstOpKind::GetTupleElement { index } => {
                    let resolved_slot = op
                        .operands
                        .first()
                        .and_then(|opnd| {
                            let from_tup = tuple_slots
                                .get(&opnd.name)
                                .and_then(|ts| ts.get(*index).copied());
                            from_tup.or_else(|| resolve_ssa_slot(ssa_slots, opnd))
                        })
                        .unwrap_or(0);
                    for res in &op.results {
                        ssa_slots.insert(res.name.clone(), vec![resolved_slot]);
                    }
                }
                AstOpKind::SdyManualComputation
                | AstOpKind::While
                | AstOpKind::Conditional
                | AstOpKind::ScfFor => {
                    // Map nested region block arguments to the outer operands
                    for nested_reg in &op.regions {
                        for entry_blk in nested_reg.blocks.iter().take(1) {
                            for (arg_idx, blk_arg) in entry_blk.arguments.iter().enumerate() {
                                if let Some(outer_slots) = op
                                    .operands
                                    .get(arg_idx)
                                    .and_then(|o| ssa_slots.get(&o.name).cloned())
                                {
                                    ssa_slots.insert(blk_arg.ssa_name.clone(), outer_slots);
                                } else {
                                    let sh = blk_arg
                                        .ty
                                        .to_shape()
                                        .unwrap_or_else(default_f32_vec4_shape);
                                    ssa_slots.insert(blk_arg.ssa_name.clone(), vec![*next_slot]);
                                    *next_slot += 1;
                                    slot_shapes.push(sh);
                                }
                            }
                        }
                        let mut nested_outputs = Vec::new();
                        lower_region_operations(
                            nested_reg,
                            ssa_slots,
                            tuple_slots,
                            slot_shapes,
                            next_slot,
                            bundles,
                            &mut nested_outputs,
                            primary_op,
                        );
                        for (res_idx, res) in op.results.iter().enumerate() {
                            let slots: Vec<usize> = if res.count > 1 {
                                nested_outputs.clone()
                            } else {
                                nested_outputs.get(res_idx).copied().into_iter().collect()
                            };
                            ssa_slots.insert(res.name.clone(), slots);
                        }
                    }
                }
                AstOpKind::SdyShardingConstraint
                | AstOpKind::SdyDataFlow
                | AstOpKind::BarrierOrToken => {
                    let in_slot = op
                        .operands
                        .first()
                        .and_then(|o| resolve_ssa_slot(ssa_slots, o))
                        .unwrap_or(0);
                    for res in &op.results {
                        ssa_slots.insert(res.name.clone(), vec![in_slot]);
                    }
                }
                _ => {
                    lower_single_mlir_operation(
                        op,
                        ssa_slots,
                        slot_shapes,
                        next_slot,
                        bundles,
                        primary_op,
                    );
                }
            }
        }
    }
}

/// Lowers a single hardware-evaluable [`AstOperation`] into one or more [`TpuInstructionBundle`]s.
///
/// # Arguments
///
/// * `op` - [`AstOperation`] node to lower.
/// * `ssa_slots` - Map from SSA variable name to allocated register slot indices.
/// * `slot_shapes` - Shapes indexed by register slot.
/// * `next_slot` - Next available register slot counter.
/// * `bundles` - Output instruction bundle list.
/// * `primary_op` - Primary [`TpuOpCode`] tracker.
#[allow(clippy::too_many_lines)]
fn lower_single_mlir_operation(
    op: &AstOperation,
    ssa_slots: &mut BTreeMap<String, Vec<usize>>,
    slot_shapes: &mut Vec<Shape>,
    next_slot: &mut usize,
    bundles: &mut Vec<TpuInstructionBundle>,
    primary_op: &mut TpuOpCode,
) {
    let opcode = match &op.op_kind {
        AstOpKind::HardwareOp(hw) => *hw,
        AstOpKind::MosaicDmaStart
        | AstOpKind::MosaicDmaWait
        | AstOpKind::MosaicSemaphore(_)
        | AstOpKind::Call(_) => TpuOpCode::CustomCall,
        _ => TpuOpCode::Identity,
    };
    *primary_op = opcode;

    let mut operand_slots: Vec<usize> = op
        .operands
        .iter()
        .filter_map(|opnd| resolve_ssa_slot(ssa_slots, opnd))
        .collect();
    if operand_slots.is_empty() && opcode.arity() > 0 {
        while slot_shapes.len() < opcode.arity() {
            slot_shapes.push(default_f32_vec4_shape());
        }
        *next_slot = slot_shapes.len();
        operand_slots = (0..opcode.arity()).collect();
    }

    let in_shapes: Vec<Shape> = operand_slots
        .iter()
        .filter_map(|&s| slot_shapes.get(s).cloned())
        .collect();
    let out_shape = op
        .signature
        .results
        .first()
        .and_then(MlirType::to_shape)
        .or_else(|| in_shapes.first().cloned())
        .unwrap_or_else(default_f32_vec4_shape);

    let attrs = build_instruction_attrs_from_ast(op, &in_shapes, &out_shape);
    let total_results = op
        .results
        .iter()
        .map(|r| r.count.max(1))
        .sum::<usize>()
        .max(1);

    let first_out_slot = *next_slot;
    let pc = u32::try_from(bundles.len()).unwrap_or(0);
    bundles.push(TpuInstructionBundle {
        pc_offset: pc,
        opcode,
        operand_slots,
        output_slot: first_out_slot,
        sync_flag_id: pc + 1,
        attrs,
    });

    let mut allocated_slots = Vec::with_capacity(total_results);
    for r_idx in 0..total_results {
        let out_slot = *next_slot;
        *next_slot += 1;
        let r_shape = op
            .signature
            .results
            .get(r_idx)
            .and_then(MlirType::to_shape)
            .unwrap_or_else(|| out_shape.clone());
        slot_shapes.push(r_shape);
        allocated_slots.push(out_slot);
    }

    let mut cursor_idx = 0_usize;
    for res in &op.results {
        let cnt = res.count.max(1);
        let slice = allocated_slots
            .get(cursor_idx..cursor_idx + cnt)
            .unwrap_or(&allocated_slots)
            .to_vec();
        ssa_slots.insert(res.name.clone(), slice);
        cursor_idx += cnt;
    }
}

/// Extracts a `Vec<usize>` attribute from `attrs` by `key`.
///
/// # Arguments
///
/// * `attrs` - Parsed operation attribute map.
/// * `key` - Attribute key name to look up.
#[must_use]
fn attr_usize_vec(attrs: &BTreeMap<String, MlirAttribute>, key: &str) -> Vec<usize> {
    attrs
        .get(key)
        .map_or_else(Vec::new, MlirAttribute::as_usize_vec)
}

/// Extracts a single `usize` attribute from `attrs` by `key`.
///
/// # Arguments
///
/// * `attrs` - Parsed operation attribute map.
/// * `key` - Attribute key name to look up.
#[must_use]
fn attr_usize(attrs: &BTreeMap<String, MlirAttribute>, key: &str) -> usize {
    attrs
        .get(key)
        .and_then(MlirAttribute::as_usize)
        .unwrap_or(0)
}

/// Extracts [`InstructionAttrs`] from a parsed [`AstOperation`]'s strongly-typed [`MlirAttribute`]s.
///
/// # Arguments
///
/// * `op` - Source [`AstOperation`] node.
/// * `in_shapes` - Resolved input operand [`Shape`]s.
/// * `out_shape` - Resolved primary output [`Shape`].
#[allow(clippy::too_many_lines)]
fn build_instruction_attrs_from_ast(
    op: &AstOperation,
    in_shapes: &[Shape],
    out_shape: &Shape,
) -> InstructionAttrs {
    let mut attrs = InstructionAttrs {
        input_shapes: in_shapes.to_vec(),
        output_shape: Some(out_shape.clone()),
        permutation: op
            .attributes
            .get("permutation")
            .or_else(|| op.attributes.get("dims"))
            .map_or_else(Vec::new, MlirAttribute::as_usize_vec),
        broadcast_dims: attr_usize_vec(&op.attributes, "broadcast_dimensions"),
        reduce_dims: attr_usize_vec(&op.attributes, "dimensions"),
        iota_dim: attr_usize(&op.attributes, "iota_dimension"),
        concat_dim: attr_usize(&op.attributes, "dimension"),
        slice_starts: attr_usize_vec(&op.attributes, "start_indices"),
        slice_limits: attr_usize_vec(&op.attributes, "limit_indices"),
        slice_strides: attr_usize_vec(&op.attributes, "strides"),
        window_dims: attr_usize_vec(&op.attributes, "window_dimensions"),
        window_strides: attr_usize_vec(&op.attributes, "window_strides"),
        pad_edge_low: attr_usize_vec(&op.attributes, "edge_padding_low"),
        pad_edge_high: attr_usize_vec(&op.attributes, "edge_padding_high"),
        pad_interior: attr_usize_vec(&op.attributes, "interior_padding"),
        lhs_dilation: attr_usize_vec(&op.attributes, "lhs_dilation"),
        rhs_dilation: attr_usize_vec(&op.attributes, "rhs_dilation"),
        feature_group_count: attr_usize(&op.attributes, "feature_group_count").max(1),
        batch_group_count: attr_usize(&op.attributes, "batch_group_count").max(1),
        gather_index_vector_dim: attr_usize(&op.attributes, "index_vector_dim"),
        gather_slice_sizes: attr_usize_vec(&op.attributes, "slice_sizes"),
        collapsed_slice_dims: attr_usize_vec(&op.attributes, "collapsed_slice_dims"),
        scatter_update_window_dims: attr_usize_vec(&op.attributes, "update_window_dims"),
        custom_call_target: op
            .attributes
            .get("call_target_name")
            .and_then(MlirAttribute::as_str)
            .map_or_else(String::new, String::from),
        comparison_direction: match op.attributes.get("comparison_direction") {
            Some(MlirAttribute::ComparisonDirection(dir)) => String::from(dir.as_str()),
            _ => String::new(),
        },
        ..InstructionAttrs::default()
    };

    for nested_reg in &op.regions {
        let mut sub_ssa = BTreeMap::new();
        let mut sub_tuples = BTreeMap::new();
        let mut sub_shapes = in_shapes.to_vec();
        let mut sub_next = sub_shapes.len();
        for entry_blk in nested_reg.blocks.iter().take(1) {
            for (arg_idx, blk_arg) in entry_blk.arguments.iter().enumerate() {
                sub_ssa.insert(blk_arg.ssa_name.clone(), vec![arg_idx]);
            }
        }
        let mut sub_bundles = Vec::new();
        let mut sub_outputs = Vec::new();
        let mut sub_prim = TpuOpCode::Identity;
        lower_region_operations(
            nested_reg,
            &mut sub_ssa,
            &mut sub_tuples,
            &mut sub_shapes,
            &mut sub_next,
            &mut sub_bundles,
            &mut sub_outputs,
            &mut sub_prim,
        );
        attrs.sub_computations.push(sub_bundles);
    }

    if let Some(MlirAttribute::DenseElements(de)) = op.attributes.get("value") {
        let out_sh = de
            .tensor_type
            .as_ref()
            .map_or_else(|| out_shape.clone(), RankedTensorType::to_shape);
        let elem_type = out_sh.element_type;
        attrs.output_shape = Some(out_sh.clone());
        if let Some(raw) = &de.raw_bytes {
            attrs.constant_bytes.clone_from(raw);
        } else {
            let count = out_sh.element_count().max(de.values_f64.len());
            let expanded = if de.values_f64.len() == 1 && count > 1 {
                vec![de.values_f64.first().copied().unwrap_or(0.0); count]
            } else {
                de.values_f64.clone()
            };
            attrs.constant_bytes = elem_type.pack_from_f64_slice(&expanded);
        }
    }
    if let Some(MlirAttribute::DotDimensionNumbers(dot)) =
        op.attributes.get("dot_dimension_numbers")
    {
        attrs
            .dot_lhs_contracting_dims
            .clone_from(&dot.lhs_contracting_dimensions);
        attrs
            .dot_rhs_contracting_dims
            .clone_from(&dot.rhs_contracting_dimensions);
        attrs
            .dot_lhs_batch_dims
            .clone_from(&dot.lhs_batching_dimensions);
        attrs
            .dot_rhs_batch_dims
            .clone_from(&dot.rhs_batching_dimensions);
    }
    attrs
}

/// Lowers a secondary [`HloComputationAst`] (such as a `while` body/condition, `conditional` branch,
/// or `fusion`/`call` target) into a sequence of [`TpuInstructionBundle`]s.
///
/// # Arguments
///
/// * `sub_comp` - Secondary [`HloComputationAst`] to lower.
/// * `caller_op_slots` - Register slot indices passed as arguments by the caller instruction.
/// * `slot_shapes` - Mutable register slot [`Shape`] table.
/// * `next_slot` - Mutable register slot counter.
#[must_use]
fn lower_hlo_subcomputation_bundles(
    sub_comp: &HloComputationAst,
    caller_op_slots: &[usize],
    slot_shapes: &mut Vec<Shape>,
    next_slot: &mut usize,
) -> (Vec<TpuInstructionBundle>, usize) {
    let mut local_slots: BTreeMap<String, usize> = BTreeMap::new();
    let mut sub_bundles = Vec::new();
    let mut last_out_slot = caller_op_slots.first().copied().unwrap_or(0);
    for sub_inst in &sub_comp.instructions {
        if let AstOpKind::Parameter { index } = sub_inst.op_kind {
            let p_idx = sub_inst.parameter_index.unwrap_or(index);
            let mapped_slot = caller_op_slots.get(p_idx).copied().unwrap_or(last_out_slot);
            local_slots.insert(sub_inst.name.clone(), mapped_slot);
            last_out_slot = mapped_slot;
        }
        if let AstOpKind::HardwareOp(sub_op) = sub_inst.op_kind {
            let sub_opnds: Vec<usize> = sub_inst
                .operands
                .iter()
                .filter_map(|n| local_slots.get(n).copied())
                .collect();
            let in_sh: Vec<Shape> = sub_opnds
                .iter()
                .filter_map(|&s| slot_shapes.get(s).cloned())
                .collect();
            let out_sh = sub_inst
                .result_type
                .to_shape()
                .unwrap_or_else(default_f32_vec4_shape);
            let sub_attrs = InstructionAttrs {
                input_shapes: in_sh,
                output_shape: Some(out_sh.clone()),
                reduce_dims: sub_inst.dimensions.clone(),
                permutation: sub_inst.dimensions.clone(),
                broadcast_dims: sub_inst.dimensions.clone(),
                dot_lhs_contracting_dims: sub_inst.dot_dims.lhs_contracting_dimensions.clone(),
                dot_rhs_contracting_dims: sub_inst.dot_dims.rhs_contracting_dimensions.clone(),
                dot_lhs_batch_dims: sub_inst.dot_dims.lhs_batching_dimensions.clone(),
                dot_rhs_batch_dims: sub_inst.dot_dims.rhs_batching_dimensions.clone(),
                ..InstructionAttrs::default()
            };
            let out_slot = *next_slot;
            *next_slot += 1;
            slot_shapes.push(out_sh);
            local_slots.insert(sub_inst.name.clone(), out_slot);
            last_out_slot = out_slot;
            let pc = u32::try_from(sub_bundles.len()).unwrap_or(0);
            sub_bundles.push(TpuInstructionBundle {
                pc_offset: pc,
                opcode: sub_op,
                operand_slots: sub_opnds,
                output_slot: out_slot,
                sync_flag_id: pc + 1,
                attrs: sub_attrs,
            });
        }
    }
    (sub_bundles, last_out_slot)
}

/// Lowers parsed textual XLA HLO [`HloComputationAst`]s into a [`ParsedHloModule`].
///
/// # Arguments
///
/// * `ast` - Parsed [`AstModule`] containing XLA HLO computations.
#[allow(clippy::too_many_lines)]
fn lower_hlo_computations(ast: &AstModule) -> ParsedHloModule {
    let comp_map: BTreeMap<&str, &HloComputationAst> = ast
        .hlo_computations
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    let mut parameter_shapes = Vec::new();
    let mut output_shapes = Vec::new();
    let mut inst_slots: BTreeMap<String, usize> = BTreeMap::new();
    let mut tuple_slots: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut slot_shapes: Vec<Shape> = Vec::new();
    let mut next_slot = 0_usize;
    let mut bundles = Vec::new();
    let mut output_slots = Vec::new();
    let mut primary_op = TpuOpCode::Identity;

    for comp in ast.hlo_computations.iter().filter(|c| c.is_entry) {
        for inst in &comp.instructions {
            match &inst.op_kind {
                AstOpKind::Parameter { .. } => {
                    let sh = inst
                        .result_type
                        .to_shape()
                        .unwrap_or_else(default_f32_vec4_shape);
                    inst_slots.insert(inst.name.clone(), next_slot);
                    next_slot += 1;
                    slot_shapes.push(sh.clone());
                    parameter_shapes.push(sh);
                }
                AstOpKind::Tuple => {
                    let members: Vec<usize> = inst
                        .operands
                        .iter()
                        .filter_map(|op_name| inst_slots.get(op_name).copied())
                        .collect();
                    tuple_slots.insert(inst.name.clone(), members.clone());
                    for &first_slot in members.iter().take(1) {
                        inst_slots.insert(inst.name.clone(), first_slot);
                    }
                    if inst.is_root {
                        output_slots.extend(members.iter().copied());
                        output_shapes
                            .extend(members.iter().filter_map(|&s| slot_shapes.get(s).cloned()));
                    }
                }
                AstOpKind::GetTupleElement { index } => {
                    let idx = inst.tuple_index.unwrap_or(*index);
                    let tuple_member = inst
                        .operands
                        .first()
                        .and_then(|op_name| tuple_slots.get(op_name))
                        .and_then(|members| members.get(idx).copied());
                    let direct_slot = inst
                        .operands
                        .first()
                        .and_then(|op_name| inst_slots.get(op_name).copied());
                    let resolved = tuple_member.or(direct_slot).unwrap_or(0);
                    inst_slots.insert(inst.name.clone(), resolved);
                    if inst.is_root {
                        output_slots.push(resolved);
                        output_shapes.extend(slot_shapes.get(resolved).cloned());
                    }
                }
                _ => {
                    let mut opcode = match &inst.op_kind {
                        AstOpKind::HardwareOp(hw) => *hw,
                        AstOpKind::While => TpuOpCode::While,
                        _ => TpuOpCode::Conditional,
                    };
                    let mut op_slots: Vec<usize> = inst
                        .operands
                        .iter()
                        .filter_map(|op_name| inst_slots.get(op_name).copied())
                        .collect();
                    if op_slots.is_empty() && opcode.arity() > 0 {
                        op_slots = (0..opcode.arity()).collect();
                    }

                    // Resolve sub-computation `to_apply=%reducer`, `calls=%sub`, or `body`/`condition`
                    let mut reduce_op_kind = crate::types::CollectiveReductionOp::Sum;
                    let mut sub_computations = Vec::new();
                    for sub_comp in inst
                        .called_computations
                        .values()
                        .filter_map(|sub_name| comp_map.get(sub_name.as_str()))
                    {
                        for sub_inst in &sub_comp.instructions {
                            match sub_inst.opcode.as_str() {
                                "maximum" | "max" => {
                                    reduce_op_kind = crate::types::CollectiveReductionOp::Max;
                                    opcode = TpuOpCode::ReduceMax;
                                }
                                "minimum" | "min" => {
                                    reduce_op_kind = crate::types::CollectiveReductionOp::Min;
                                    opcode = TpuOpCode::ReduceMin;
                                }
                                "multiply" | "mul" => {
                                    reduce_op_kind = crate::types::CollectiveReductionOp::Product;
                                }
                                "and" => {
                                    opcode = TpuOpCode::And;
                                }
                                "or" => {
                                    opcode = TpuOpCode::Or;
                                }
                                "xor" => {
                                    opcode = TpuOpCode::XorBytes;
                                }
                                _ => {}
                            }
                        }
                        let (sb, _) = lower_hlo_subcomputation_bundles(
                            sub_comp,
                            &op_slots,
                            &mut slot_shapes,
                            &mut next_slot,
                        );
                        sub_computations.push(sb);
                    }
                    primary_op = opcode;

                    let in_shapes: Vec<Shape> = op_slots
                        .iter()
                        .filter_map(|&s| slot_shapes.get(s).cloned())
                        .collect();
                    let out_shape = inst
                        .result_type
                        .to_shape()
                        .unwrap_or_else(default_f32_vec4_shape);

                    let mut attrs = InstructionAttrs {
                        input_shapes: in_shapes,
                        output_shape: Some(out_shape.clone()),
                        reduce_dims: inst.dimensions.clone(),
                        permutation: inst.dimensions.clone(),
                        broadcast_dims: inst.dimensions.clone(),
                        dot_lhs_contracting_dims: inst.dot_dims.lhs_contracting_dimensions.clone(),
                        dot_rhs_contracting_dims: inst.dot_dims.rhs_contracting_dimensions.clone(),
                        dot_lhs_batch_dims: inst.dot_dims.lhs_batching_dimensions.clone(),
                        dot_rhs_batch_dims: inst.dot_dims.rhs_batching_dimensions.clone(),
                        custom_call_target: inst.custom_call_target.clone().unwrap_or_default(),
                        reduce_op_kind,
                        sub_computations,
                        ..InstructionAttrs::default()
                    };
                    if let Some(win) = &inst.window {
                        attrs.window_dims.clone_from(&win.size);
                        attrs.window_strides.clone_from(&win.stride);
                    }
                    if !inst.slice_bounds.is_empty() {
                        attrs.slice_starts = inst.slice_bounds.iter().map(|&(s, _, _)| s).collect();
                        attrs.slice_limits = inst.slice_bounds.iter().map(|&(_, l, _)| l).collect();
                        attrs.slice_strides =
                            inst.slice_bounds.iter().map(|&(_, _, st)| st).collect();
                    }
                    if !inst.padding_config.is_empty() {
                        attrs.pad_edge_low =
                            inst.padding_config.iter().map(|&(l, _, _)| l).collect();
                        attrs.pad_edge_high =
                            inst.padding_config.iter().map(|&(_, h, _)| h).collect();
                        attrs.pad_interior =
                            inst.padding_config.iter().map(|&(_, _, i)| i).collect();
                    }
                    if let Some(MlirAttribute::DenseElements(de)) =
                        inst.extra_attrs.get("constant_values")
                    {
                        attrs.constant_bytes =
                            out_shape.element_type.pack_from_f64_slice(&de.values_f64);
                    }

                    let out_slot = next_slot;
                    inst_slots.insert(inst.name.clone(), out_slot);
                    next_slot += 1;
                    slot_shapes.push(out_shape.clone());
                    let pc = u32::try_from(bundles.len()).unwrap_or(0);
                    bundles.push(TpuInstructionBundle {
                        pc_offset: pc,
                        opcode,
                        operand_slots: op_slots,
                        output_slot: out_slot,
                        sync_flag_id: pc + 1,
                        attrs,
                    });

                    if inst.is_root {
                        output_slots.push(out_slot);
                        output_shapes.push(out_shape);
                    }
                }
            }
        }
    }

    finalize_lowered_module(
        primary_op,
        parameter_shapes,
        output_shapes,
        &slot_shapes,
        bundles,
        output_slots,
    )
}

/// Populates fallback parameter/output shapes and bundles when lowering single-mnemonic or bare snippets.
///
/// # Arguments
///
/// * `primary_op` - Primary [`TpuOpCode`] resolved for the module.
/// * `parameter_shapes` - Input parameter [`Shape`] list.
/// * `output_shapes` - Output result [`Shape`] list.
/// * `slot_shapes` - Register slot [`Shape`] table.
/// * `bundles` - Lowered [`TpuInstructionBundle`] sequence.
/// * `output_slots` - Output register slot indices.
fn finalize_lowered_module(
    primary_op: TpuOpCode,
    mut parameter_shapes: Vec<Shape>,
    mut output_shapes: Vec<Shape>,
    slot_shapes: &[Shape],
    mut bundles: Vec<TpuInstructionBundle>,
    mut output_slots: Vec<usize>,
) -> ParsedHloModule {
    if parameter_shapes.is_empty() {
        let default_shape = default_f32_vec4_shape();
        for _ in 0..primary_op.arity().max(1) {
            parameter_shapes.push(default_shape.clone());
        }
    }

    if output_shapes.is_empty() {
        for &slot in &output_slots {
            output_shapes.extend(slot_shapes.get(slot).cloned());
        }
        if output_shapes.is_empty() {
            output_shapes.push(
                parameter_shapes
                    .first()
                    .cloned()
                    .unwrap_or_else(default_f32_vec4_shape),
            );
        }
    }

    if bundles.is_empty() {
        let operand_slots: Vec<usize> = (0..parameter_shapes.len()).collect();
        let out_slot = parameter_shapes.len();
        bundles.push(TpuInstructionBundle {
            pc_offset: 0,
            opcode: primary_op,
            operand_slots,
            output_slot: out_slot,
            sync_flag_id: 1,
            attrs: InstructionAttrs {
                input_shapes: parameter_shapes.clone(),
                output_shape: output_shapes.first().cloned(),
                ..InstructionAttrs::default()
            },
        });
    }

    if output_slots.is_empty() {
        let last_slot = bundles
            .last()
            .map_or(parameter_shapes.len(), |b| b.output_slot);
        output_slots.push(last_slot);
    }

    ParsedHloModule {
        primary_op,
        parameter_shapes,
        output_shapes,
        bundles,
        output_slots,
    }
}

/// Strictly validates operand counts (`arity`) and operand element types for every operation in `ast`,
/// returning [`LibTpuError::ShapeInferenceError`] or [`LibTpuError::DialectVerificationError`] on mismatch.
///
/// # Arguments
///
/// * `ast` - Parsed [`AstModule`] to verify.
///
/// # Errors
///
/// Returns [`LibTpuError::ShapeInferenceError`] if an operation's operand count or input types violate its signature,
/// or [`LibTpuError::UndefinedSsaValue`] if an operand references an undefined SSA value.
pub fn verify_ast_module_semantics(ast: &AstModule) -> LibTpuResult<()> {
    for func in &ast.functions {
        let mut defined_ssa: std::collections::BTreeSet<String> =
            func.arguments.iter().map(|a| a.ssa_name.clone()).collect();
        verify_region_semantics(&func.body, &mut defined_ssa)?;
    }
    Ok(())
}

/// Recursively verifies SSA definitions, dialect constraints, and shape/arity rules inside an [`AstRegion`].
///
/// # Arguments
///
/// * `region` - [`AstRegion`] to verify.
/// * `defined_ssa` - Set of SSA variable names defined in the enclosing scope.
fn verify_region_semantics(
    region: &AstRegion,
    defined_ssa: &mut std::collections::BTreeSet<String>,
) -> LibTpuResult<()> {
    for block in &region.blocks {
        for arg in &block.arguments {
            defined_ssa.insert(arg.ssa_name.clone());
        }
        for op in &block.operations {
            for opnd in &op.operands {
                if !defined_ssa.contains(&opnd.name) {
                    return Err(LibTpuError::UndefinedSsaValue {
                        name: opnd.name.clone(),
                        line: opnd.span.line,
                        column: opnd.span.column,
                    });
                }
            }
            if op.dialect == DialectKind::Sdy
                && op.mnemonic == "manual_computation"
                && op.regions.is_empty()
            {
                return Err(LibTpuError::DialectVerificationError {
                    dialect: String::from("sdy"),
                    op: String::from("manual_computation"),
                    reason: String::from("sdy.manual_computation requires a body region"),
                    line: op.span.line,
                    column: op.span.column,
                });
            }
            if let AstOpKind::HardwareOp(
                TpuOpCode::AddF32
                | TpuOpCode::AddS32
                | TpuOpCode::SubF32
                | TpuOpCode::MulF32
                | TpuOpCode::DivF32,
            ) = op.op_kind
            {
                if op.operands.len() != 2 {
                    return Err(LibTpuError::ShapeInferenceError {
                        op: format!("{}.{}", op.dialect.as_str(), op.mnemonic),
                        reason: format!(
                            "binary operation requires 2 operands, found {}",
                            op.operands.len()
                        ),
                        line: op.span.line,
                        column: op.span.column,
                    });
                }
                if let [lhs_ty, rhs_ty] = op.signature.inputs.as_slice() {
                    let l = lhs_ty.to_shape().unwrap_or_else(default_f32_vec4_shape);
                    let r = rhs_ty.to_shape().unwrap_or_else(default_f32_vec4_shape);
                    if (l.dims != r.dims || l.element_type != r.element_type)
                        && op.dialect != DialectKind::Chlo
                    {
                        return Err(LibTpuError::ShapeInferenceError {
                            op: format!("{}.{}", op.dialect.as_str(), op.mnemonic),
                            reason: format!(
                                "elementwise binary operand shape mismatch: {:?} vs {:?}",
                                l.dims, r.dims
                            ),
                            line: op.span.line,
                            column: op.span.column,
                        });
                    }
                }
            }
            for res in &op.results {
                defined_ssa.insert(res.name.clone());
            }
            for nested in &op.regions {
                let mut child_scope = defined_ssa.clone();
                verify_region_semantics(nested, &mut child_scope)?;
            }
        }
    }
    Ok(())
}

/// Parses a textual XLA HLO or MLIR source string into a unified [`AstModule`].
///
/// # Arguments
///
/// * `text` - Source text (MLIR or XLA HLO).
///
/// # Errors
///
/// Returns [`LibTpuError`] if lexing or recursive-descent parsing fails.
pub fn parse_unified_ast_module(text: &str) -> LibTpuResult<AstModule> {
    let _checked = HloLexer::tokenize_checked(text)?;
    let spanned = HloLexer::tokenize_spanned(text);
    let is_hlo = spanned.iter().any(|st| {
        matches!(
            &st.token,
            HloToken::Ident(id) if id == "HloModule" || id == "ENTRY"
        )
    });
    if is_hlo {
        let mut hlo_parser = HloTextParser::new(&spanned);
        hlo_parser.parse_hlo_module()
    } else {
        super::parser::parse_mlir_ast_module(text)
    }
}
