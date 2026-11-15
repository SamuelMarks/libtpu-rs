//! Canonical `chlo` (Client HLO) -> `stablehlo` multi-operation graph decomposition pass.
//!
//! Lowers implicit-broadcast `chlo.broadcast_*` ops, `chlo.erf` / `erfc` / `erf_inv`, `chlo.top_k`,
//! `chlo.ragged_dot`, `chlo.conj`, and hyperbolic / inverse-trigonometric ops (`sinh`, `cosh`,
//! `tanh`, `asinh`, `acosh`, `atanh`, `asin`, `acos`, `atan`) into pure `stablehlo` operations.

use super::ast::{
    AstBlock, AstBlockArg, AstFunctionType, AstModule, AstOpKind, AstOpResult, AstOperand,
    AstOperation, AstRegion, ComparisonDirection, ComparisonType, DenseElementsAttr, DialectKind,
    DimensionSize, DotDimensionNumbersAttr, MlirAttribute, MlirLocation, MlirType,
    RankedTensorType,
};
use super::lexer::TokenSpan;
use super::opcode::TpuOpCode;
use super::parser::classify_ast_operation;
use crate::types::ElementType;
use std::collections::BTreeMap;

/// Monotonic SSA temporary counter used during `chlo` -> `stablehlo` multi-op expansion.
#[derive(Debug, Default)]
struct ChloExpansionBuilder {
    /// Next unique synthetic SSA index.
    next_id: usize,
}

impl ChloExpansionBuilder {
    /// Allocates a fresh unique `%__chlo_N` SSA value name.
    fn fresh_ssa(&mut self) -> String {
        let name = format!("%__chlo_{}", self.next_id);
        self.next_id += 1;
        name
    }

    /// Emits a splat scalar/tensor `stablehlo.constant` of `value` matching `ty` and pushes it into `out`.
    ///
    /// # Arguments
    ///
    /// * `out` - Destination operation list.
    /// * `value` - Scalar floating-point constant value.
    /// * `ty` - Result [`MlirType`] of the constant.
    fn emit_splat_constant(
        &mut self,
        out: &mut Vec<AstOperation>,
        value: f64,
        ty: &MlirType,
    ) -> AstOperand {
        let ssa = self.fresh_ssa();
        let elem_ty = ty.element_type();
        let ranked = match ty {
            MlirType::RankedTensor(rt) => Some(rt.clone()),
            _ => None,
        };
        let mut attrs = BTreeMap::new();
        attrs.insert(
            String::from("value"),
            MlirAttribute::DenseElements(DenseElementsAttr {
                values_f64: vec![value],
                raw_bytes: Some(elem_ty.pack_from_f64_slice(&[value])),
                is_splat: true,
                tensor_type: ranked,
            }),
        );
        out.push(AstOperation {
            results: vec![AstOpResult {
                name: ssa.clone(),
                count: 1,
            }],
            dialect: DialectKind::StableHlo,
            mnemonic: String::from("constant"),
            op_kind: AstOpKind::HardwareOp(TpuOpCode::Constant),
            operands: Vec::new(),
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: attrs,
            signature: AstFunctionType {
                inputs: Vec::new(),
                results: vec![ty.clone()],
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        });
        AstOperand {
            name: ssa,
            result_index: None,
            span: TokenSpan::default(),
        }
    }

    /// Emits a unary `stablehlo.<mnemonic>` operation returning `res_name` (or a fresh SSA if `None`).
    ///
    /// # Arguments
    ///
    /// * `out` - Destination operation list.
    /// * `mnemonic` - Canonical `stablehlo` unary mnemonic.
    /// * `x` - Input [`AstOperand`].
    /// * `ty` - Input and result [`MlirType`].
    /// * `res_name` - Optional explicit result SSA name.
    fn emit_unary(
        &mut self,
        out: &mut Vec<AstOperation>,
        mnemonic: &str,
        x: AstOperand,
        ty: &MlirType,
        res_name: Option<String>,
    ) -> AstOperand {
        let ssa = res_name.unwrap_or_else(|| self.fresh_ssa());
        let attrs = BTreeMap::new();
        let op_kind = classify_ast_operation(
            DialectKind::StableHlo,
            mnemonic,
            ty.element_type(),
            &attrs,
            &[],
        );
        out.push(AstOperation {
            results: vec![AstOpResult {
                name: ssa.clone(),
                count: 1,
            }],
            dialect: DialectKind::StableHlo,
            mnemonic: String::from(mnemonic),
            op_kind,
            operands: vec![x],
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: attrs,
            signature: AstFunctionType {
                inputs: vec![ty.clone()],
                results: vec![ty.clone()],
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        });
        AstOperand {
            name: ssa,
            result_index: None,
            span: TokenSpan::default(),
        }
    }

    /// Emits a binary `stablehlo.<mnemonic>` operation returning `res_name` (or a fresh SSA if `None`).
    ///
    /// # Arguments
    ///
    /// * `out` - Destination operation list.
    /// * `mnemonic` - Canonical `stablehlo` binary mnemonic.
    /// * `lhs` - Left-hand [`AstOperand`].
    /// * `rhs` - Right-hand [`AstOperand`].
    /// * `ty` - Broadcasted operand and result [`MlirType`].
    /// * `res_name` - Optional explicit result SSA name.
    fn emit_binary(
        &mut self,
        out: &mut Vec<AstOperation>,
        mnemonic: &str,
        lhs: AstOperand,
        rhs: AstOperand,
        ty: &MlirType,
        res_name: Option<String>,
    ) -> AstOperand {
        let ssa = res_name.unwrap_or_else(|| self.fresh_ssa());
        let attrs = BTreeMap::new();
        let op_kind = classify_ast_operation(
            DialectKind::StableHlo,
            mnemonic,
            ty.element_type(),
            &attrs,
            &[],
        );
        out.push(AstOperation {
            results: vec![AstOpResult {
                name: ssa.clone(),
                count: 1,
            }],
            dialect: DialectKind::StableHlo,
            mnemonic: String::from(mnemonic),
            op_kind,
            operands: vec![lhs, rhs],
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: attrs,
            signature: AstFunctionType {
                inputs: vec![ty.clone(), ty.clone()],
                results: vec![ty.clone()],
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        });
        AstOperand {
            name: ssa,
            result_index: None,
            span: TokenSpan::default(),
        }
    }

    /// Broadcasts `operand` from `in_ty` to `target_ty` via `stablehlo.broadcast_in_dim` when shapes differ.
    ///
    /// # Arguments
    ///
    /// * `out` - Destination operation list.
    /// * `operand` - Source [`AstOperand`].
    /// * `in_ty` - Source operand [`MlirType`].
    /// * `target_ty` - Target broadcasted [`MlirType`].
    /// * `explicit_bcast_dims` - Optional explicit `broadcast_dimensions` axis map.
    fn maybe_broadcast_operand(
        &mut self,
        out: &mut Vec<AstOperation>,
        operand: &AstOperand,
        in_ty: &MlirType,
        target_ty: &MlirType,
        explicit_bcast_dims: Option<&[i64]>,
    ) -> AstOperand {
        let (MlirType::RankedTensor(in_rt), MlirType::RankedTensor(out_rt)) = (in_ty, target_ty)
        else {
            return operand.clone();
        };
        if in_rt.dims == out_rt.dims {
            return operand.clone();
        }
        let bcast_dims: Vec<i64> = if let Some(exp) = explicit_bcast_dims
            && exp.len() == in_rt.dims.len()
        {
            exp.to_vec()
        } else {
            let offset = out_rt.dims.len().saturating_sub(in_rt.dims.len());
            (0..in_rt.dims.len())
                .map(|i| i64::try_from(offset + i).unwrap_or(0))
                .collect()
        };
        let ssa = self.fresh_ssa();
        let mut attrs = BTreeMap::new();
        attrs.insert(
            String::from("broadcast_dimensions"),
            MlirAttribute::DenseI64Array(bcast_dims),
        );
        let target_same_elem = MlirType::RankedTensor(RankedTensorType {
            dims: out_rt.dims.clone(),
            element_type: in_rt.element_type,
            encoding: out_rt.encoding.clone(),
            layout: out_rt.layout.clone(),
        });
        out.push(AstOperation {
            results: vec![AstOpResult {
                name: ssa.clone(),
                count: 1,
            }],
            dialect: DialectKind::StableHlo,
            mnemonic: String::from("broadcast_in_dim"),
            op_kind: AstOpKind::HardwareOp(TpuOpCode::BroadcastInDim),
            operands: vec![operand.clone()],
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: attrs,
            signature: AstFunctionType {
                inputs: vec![in_ty.clone()],
                results: vec![target_same_elem],
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        });
        AstOperand {
            name: ssa,
            result_index: None,
            span: TokenSpan::default(),
        }
    }

    /// Emits Abramowitz & Stegun 7.1.26 rational polynomial sequence for `erf(x)` in pure `stablehlo` ops.
    ///
    /// # Arguments
    ///
    /// * `out` - Destination operation list.
    /// * `x` - Input [`AstOperand`].
    /// * `ty` - Tensor [`MlirType`] of `x`.
    /// * `final_res_name` - Optional explicit result SSA name.
    fn emit_erf_sequence(
        &mut self,
        out: &mut Vec<AstOperation>,
        x: AstOperand,
        ty: &MlirType,
        final_res_name: Option<String>,
    ) -> AstOperand {
        let sign_x = self.emit_unary(out, "sign", x.clone(), ty, None);
        let abs_x = self.emit_unary(out, "abs", x, ty, None);
        let p_c = self.emit_splat_constant(out, 0.327_591_1, ty);
        let one = self.emit_splat_constant(out, 1.0, ty);
        let px = self.emit_binary(out, "multiply", p_c, abs_x.clone(), ty, None);
        let denom = self.emit_binary(out, "add", one.clone(), px, ty, None);
        let t = self.emit_binary(out, "divide", one.clone(), denom, ty, None);

        let a5 = self.emit_splat_constant(out, 1.061_405_429, ty);
        let a4 = self.emit_splat_constant(out, -1.453_152_027, ty);
        let a3 = self.emit_splat_constant(out, 1.421_413_741, ty);
        let a2 = self.emit_splat_constant(out, -0.284_496_736, ty);
        let a1 = self.emit_splat_constant(out, 0.254_829_592, ty);

        let s1 = self.emit_binary(out, "multiply", a5, t.clone(), ty, None);
        let s2 = self.emit_binary(out, "add", s1, a4, ty, None);
        let s3 = self.emit_binary(out, "multiply", s2, t.clone(), ty, None);
        let s4 = self.emit_binary(out, "add", s3, a3, ty, None);
        let s5 = self.emit_binary(out, "multiply", s4, t.clone(), ty, None);
        let s6 = self.emit_binary(out, "add", s5, a2, ty, None);
        let s7 = self.emit_binary(out, "multiply", s6, t.clone(), ty, None);
        let s8 = self.emit_binary(out, "add", s7, a1, ty, None);
        let poly = self.emit_binary(out, "multiply", s8, t, ty, None);

        let x2 = self.emit_binary(out, "multiply", abs_x.clone(), abs_x, ty, None);
        let neg_x2 = self.emit_unary(out, "negate", x2, ty, None);
        let exp_neg_x2 = self.emit_unary(out, "exponential", neg_x2, ty, None);
        let tail = self.emit_binary(out, "multiply", poly, exp_neg_x2, ty, None);
        let erf_abs = self.emit_binary(out, "subtract", one, tail, ty, None);
        self.emit_binary(out, "multiply", sign_x, erf_abs, ty, final_res_name)
    }

    /// Expands a single `chlo` [`AstOperation`] into one or more canonical `stablehlo` [`AstOperation`]s.
    ///
    /// # Arguments
    ///
    /// * `op` - Source [`AstOperation`] to expand.
    /// * `out` - Destination canonical operation list.
    #[allow(clippy::too_many_lines)]
    fn expand_chlo_op(&mut self, op: AstOperation, out: &mut Vec<AstOperation>) {
        if op.dialect != DialectKind::Chlo {
            out.push(op);
            return;
        }
        let default_ty = MlirType::RankedTensor(RankedTensorType {
            dims: vec![DimensionSize::Static(4)],
            element_type: ElementType::F32,
            encoding: None,
            layout: None,
        });
        let res_ty = op
            .signature
            .results
            .first()
            .or_else(|| op.signature.inputs.first())
            .cloned()
            .unwrap_or(default_ty);
        let res_name = op
            .results
            .first()
            .map_or_else(|| self.fresh_ssa(), |r| r.name.clone());
        let x0 = op.operands.first().cloned().unwrap_or_else(|| AstOperand {
            name: String::from("%arg0"),
            result_index: None,
            span: TokenSpan::default(),
        });

        let explicit_bcast: Option<Vec<i64>> =
            op.attributes.get("broadcast_dimensions").map(|attr| {
                attr.as_usize_vec()
                    .into_iter()
                    .map(|u| i64::try_from(u).unwrap_or(0))
                    .collect()
            });

        let map_bcast_binary = |m: &str| -> Option<&'static str> {
            match m {
                "broadcast_add" => Some("add"),
                "broadcast_sub" | "broadcast_subtract" => Some("subtract"),
                "broadcast_mul" | "broadcast_multiply" => Some("multiply"),
                "broadcast_div" | "broadcast_divide" => Some("divide"),
                "broadcast_rem" | "broadcast_remainder" => Some("remainder"),
                "broadcast_pow" | "broadcast_power" => Some("power"),
                "broadcast_atan2" => Some("atan2"),
                "broadcast_max" | "broadcast_maximum" => Some("maximum"),
                "broadcast_min" | "broadcast_minimum" => Some("minimum"),
                "broadcast_and" => Some("and"),
                "broadcast_or" => Some("or"),
                "broadcast_xor" => Some("xor"),
                "broadcast_shift_left" => Some("shift_left"),
                "broadcast_shift_right_arithmetic" => Some("shift_right_arithmetic"),
                "broadcast_shift_right_logical" => Some("shift_right_logical"),
                "broadcast_next_after" => Some("next_after"),
                "broadcast_complex" => Some("complex"),
                _ => None,
            }
        };

        if let Some(target_bin) = map_bcast_binary(&op.mnemonic) {
            let x1 = op.operands.get(1).cloned().unwrap_or_else(|| x0.clone());
            let in0_ty = op.signature.inputs.first().unwrap_or(&res_ty);
            let in1_ty = op.signature.inputs.get(1).unwrap_or(&res_ty);
            let b0 =
                self.maybe_broadcast_operand(out, &x0, in0_ty, &res_ty, explicit_bcast.as_deref());
            let b1 =
                self.maybe_broadcast_operand(out, &x1, in1_ty, &res_ty, explicit_bcast.as_deref());
            let _ = self.emit_binary(out, target_bin, b0, b1, &res_ty, Some(res_name));
            return;
        }

        match op.mnemonic.as_str() {
            "broadcast_compare" => {
                let x1 = op.operands.get(1).cloned().unwrap_or_else(|| x0.clone());
                let bcast_target = op
                    .signature
                    .inputs
                    .iter()
                    .max_by_key(|t| {
                        if let MlirType::RankedTensor(rt) = t {
                            rt.dims.len()
                        } else {
                            0
                        }
                    })
                    .unwrap_or(&res_ty)
                    .clone();
                let in0_ty = op.signature.inputs.first().unwrap_or(&bcast_target);
                let in1_ty = op.signature.inputs.get(1).unwrap_or(&bcast_target);
                let b0 = self.maybe_broadcast_operand(
                    out,
                    &x0,
                    in0_ty,
                    &bcast_target,
                    explicit_bcast.as_deref(),
                );
                let b1 = self.maybe_broadcast_operand(
                    out,
                    &x1,
                    in1_ty,
                    &bcast_target,
                    explicit_bcast.as_deref(),
                );
                let mut attrs = op.attributes.clone();
                attrs.remove("broadcast_dimensions");
                let op_kind = classify_ast_operation(
                    DialectKind::StableHlo,
                    "compare",
                    bcast_target.element_type(),
                    &attrs,
                    &[],
                );
                out.push(AstOperation {
                    results: op.results,
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("compare"),
                    op_kind,
                    operands: vec![b0, b1],
                    regions: Vec::new(),
                    successors: Vec::new(),
                    attributes: attrs,
                    signature: AstFunctionType {
                        inputs: vec![bcast_target.clone(), bcast_target],
                        results: vec![res_ty],
                    },
                    location: op.location,
                    span: op.span,
                });
            }
            "broadcast_select" => {
                let pred = x0;
                let on_true = op.operands.get(1).unwrap_or(&pred).clone();
                let on_false = op.operands.get(2).unwrap_or(&on_true).clone();
                let pred_target = match &res_ty {
                    MlirType::RankedTensor(rt) => MlirType::RankedTensor(RankedTensorType {
                        dims: rt.dims.clone(),
                        element_type: ElementType::Pred,
                        encoding: None,
                        layout: None,
                    }),
                    other => other.clone(),
                };
                let p_ty = op.signature.inputs.first().unwrap_or(&pred_target);
                let t_ty = op.signature.inputs.get(1).unwrap_or(&res_ty);
                let f_ty = op.signature.inputs.get(2).unwrap_or(&res_ty);
                let bp = self.maybe_broadcast_operand(out, &pred, p_ty, &pred_target, None);
                let bt = self.maybe_broadcast_operand(out, &on_true, t_ty, &res_ty, None);
                let bf = self.maybe_broadcast_operand(out, &on_false, f_ty, &res_ty, None);
                out.push(AstOperation {
                    results: op.results,
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("select"),
                    op_kind: AstOpKind::HardwareOp(TpuOpCode::Select),
                    operands: vec![bp, bt, bf],
                    regions: Vec::new(),
                    successors: Vec::new(),
                    attributes: BTreeMap::new(),
                    signature: AstFunctionType {
                        inputs: vec![pred_target, res_ty.clone(), res_ty.clone()],
                        results: vec![res_ty],
                    },
                    location: op.location,
                    span: op.span,
                });
            }
            "erf" => {
                let _ = self.emit_erf_sequence(out, x0, &res_ty, Some(res_name));
            }
            "erfc" => {
                let erf_val = self.emit_erf_sequence(out, x0, &res_ty, None);
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let _ = self.emit_binary(out, "subtract", one, erf_val, &res_ty, Some(res_name));
            }
            "erf_inv" => {
                // Winitzki rational approximation:
                // sgn(x) * sqrt(sqrt((2/(pi*a) + ln(1-x^2)/2)^2 - ln(1-x^2)/a) - (2/(pi*a) + ln(1-x^2)/2))
                let sign_x = self.emit_unary(out, "sign", x0.clone(), &res_ty, None);
                let x2 = self.emit_binary(out, "multiply", x0.clone(), x0, &res_ty, None);
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let omx2 = self.emit_binary(out, "subtract", one, x2, &res_ty, None);
                let ln = self.emit_unary(out, "log", omx2, &res_ty, None);
                let half = self.emit_splat_constant(out, 0.5, &res_ty);
                let a_inv = self.emit_splat_constant(out, 1.0 / 0.147, &res_ty);
                let two_over_pi_a =
                    self.emit_splat_constant(out, 2.0 / (std::f64::consts::PI * 0.147), &res_ty);
                let half_ln = self.emit_binary(out, "multiply", ln.clone(), half, &res_ty, None);
                let term1 = self.emit_binary(out, "add", two_over_pi_a, half_ln, &res_ty, None);
                let term1_sq =
                    self.emit_binary(out, "multiply", term1.clone(), term1.clone(), &res_ty, None);
                let ln_over_a = self.emit_binary(out, "multiply", ln, a_inv, &res_ty, None);
                let inner_diff =
                    self.emit_binary(out, "subtract", term1_sq, ln_over_a, &res_ty, None);
                let inner_sqrt = self.emit_unary(out, "sqrt", inner_diff, &res_ty, None);
                let outer_diff =
                    self.emit_binary(out, "subtract", inner_sqrt, term1, &res_ty, None);
                let mag = self.emit_unary(out, "sqrt", outer_diff, &res_ty, None);
                let _ = self.emit_binary(out, "multiply", sign_x, mag, &res_ty, Some(res_name));
            }
            "sinh" => {
                let neg_x = self.emit_unary(out, "negate", x0.clone(), &res_ty, None);
                let ep = self.emit_unary(out, "exponential", x0, &res_ty, None);
                let en = self.emit_unary(out, "exponential", neg_x, &res_ty, None);
                let diff = self.emit_binary(out, "subtract", ep, en, &res_ty, None);
                let half = self.emit_splat_constant(out, 0.5, &res_ty);
                let _ = self.emit_binary(out, "multiply", diff, half, &res_ty, Some(res_name));
            }
            "cosh" => {
                let neg_x = self.emit_unary(out, "negate", x0.clone(), &res_ty, None);
                let ep = self.emit_unary(out, "exponential", x0, &res_ty, None);
                let en = self.emit_unary(out, "exponential", neg_x, &res_ty, None);
                let sum = self.emit_binary(out, "add", ep, en, &res_ty, None);
                let half = self.emit_splat_constant(out, 0.5, &res_ty);
                let _ = self.emit_binary(out, "multiply", sum, half, &res_ty, Some(res_name));
            }
            "tanh" => {
                let _ = self.emit_unary(out, "tanh", x0, &res_ty, Some(res_name));
            }
            "asinh" => {
                let x2 = self.emit_binary(out, "multiply", x0.clone(), x0.clone(), &res_ty, None);
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let x2p1 = self.emit_binary(out, "add", x2, one, &res_ty, None);
                let root = self.emit_unary(out, "sqrt", x2p1, &res_ty, None);
                let sum = self.emit_binary(out, "add", x0, root, &res_ty, None);
                let _ = self.emit_unary(out, "log", sum, &res_ty, Some(res_name));
            }
            "acosh" => {
                let x2 = self.emit_binary(out, "multiply", x0.clone(), x0.clone(), &res_ty, None);
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let x2m1 = self.emit_binary(out, "subtract", x2, one, &res_ty, None);
                let root = self.emit_unary(out, "sqrt", x2m1, &res_ty, None);
                let sum = self.emit_binary(out, "add", x0, root, &res_ty, None);
                let _ = self.emit_unary(out, "log", sum, &res_ty, Some(res_name));
            }
            "atanh" => {
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let num = self.emit_binary(out, "add", one.clone(), x0.clone(), &res_ty, None);
                let den = self.emit_binary(out, "subtract", one, x0, &res_ty, None);
                let quot = self.emit_binary(out, "divide", num, den, &res_ty, None);
                let ln = self.emit_unary(out, "log", quot, &res_ty, None);
                let half = self.emit_splat_constant(out, 0.5, &res_ty);
                let _ = self.emit_binary(out, "multiply", ln, half, &res_ty, Some(res_name));
            }
            "asin" => {
                let x2 = self.emit_binary(out, "multiply", x0.clone(), x0.clone(), &res_ty, None);
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let omx2 = self.emit_binary(out, "subtract", one, x2, &res_ty, None);
                let root = self.emit_unary(out, "sqrt", omx2, &res_ty, None);
                let _ = self.emit_binary(out, "atan2", x0, root, &res_ty, Some(res_name));
            }
            "acos" => {
                let x2 = self.emit_binary(out, "multiply", x0.clone(), x0.clone(), &res_ty, None);
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let omx2 = self.emit_binary(out, "subtract", one, x2, &res_ty, None);
                let root = self.emit_unary(out, "sqrt", omx2, &res_ty, None);
                let _ = self.emit_binary(out, "atan2", root, x0, &res_ty, Some(res_name));
            }
            "atan" => {
                let one = self.emit_splat_constant(out, 1.0, &res_ty);
                let _ = self.emit_binary(out, "atan2", x0, one, &res_ty, Some(res_name));
            }
            "conj" => {
                let real_ty = MlirType::Scalar(if res_ty.element_type() == ElementType::C128 {
                    ElementType::F64
                } else {
                    ElementType::F32
                });
                let comp_ty = match &res_ty {
                    MlirType::RankedTensor(rt) => MlirType::RankedTensor(RankedTensorType {
                        dims: rt.dims.clone(),
                        element_type: real_ty.element_type(),
                        encoding: None,
                        layout: None,
                    }),
                    _ => real_ty,
                };
                let re = self.emit_unary(out, "real", x0.clone(), &comp_ty, None);
                let im = self.emit_unary(out, "imag", x0, &comp_ty, None);
                let neg_im = self.emit_unary(out, "negate", im, &comp_ty, None);
                let ssa = res_name;
                out.push(AstOperation {
                    results: vec![AstOpResult {
                        name: ssa,
                        count: 1,
                    }],
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("complex"),
                    op_kind: AstOpKind::HardwareOp(TpuOpCode::Complex),
                    operands: vec![re, neg_im],
                    regions: Vec::new(),
                    successors: Vec::new(),
                    attributes: BTreeMap::new(),
                    signature: AstFunctionType {
                        inputs: vec![comp_ty.clone(), comp_ty],
                        results: vec![res_ty],
                    },
                    location: op.location,
                    span: op.span,
                });
            }
            "top_k" => {
                let in_ty = op.signature.inputs.first().unwrap_or(&res_ty).clone();
                let in_dims = match &in_ty {
                    MlirType::RankedTensor(rt) if !rt.dims.is_empty() => rt.dims.clone(),
                    _ => vec![DimensionSize::Static(4)],
                };
                let rank = in_dims.len();
                let last_dim = i64::try_from(rank - 1).unwrap_or(0);
                let k_val = op
                    .attributes
                    .get("k")
                    .and_then(MlirAttribute::as_i64)
                    .unwrap_or(1)
                    .max(1);
                let idx_full_ty = MlirType::RankedTensor(RankedTensorType {
                    dims: in_dims.clone(),
                    element_type: ElementType::S32,
                    encoding: None,
                    layout: None,
                });
                let mut out_dims = in_dims.clone();
                out_dims.pop();
                out_dims.push(DimensionSize::Static(k_val));
                let default_val_out = MlirType::RankedTensor(RankedTensorType {
                    dims: out_dims.clone(),
                    element_type: in_ty.element_type(),
                    encoding: None,
                    layout: None,
                });
                let default_idx_out = MlirType::RankedTensor(RankedTensorType {
                    dims: out_dims,
                    element_type: ElementType::S32,
                    encoding: None,
                    layout: None,
                });
                let val_out_ty = op
                    .signature
                    .results
                    .first()
                    .unwrap_or(&default_val_out)
                    .clone();
                let idx_out_ty = op
                    .signature
                    .results
                    .get(1)
                    .unwrap_or(&default_idx_out)
                    .clone();

                let iota_ssa = self.fresh_ssa();
                let mut iota_attrs = BTreeMap::new();
                iota_attrs.insert(
                    String::from("iota_dimension"),
                    MlirAttribute::Integer {
                        value: i128::from(last_dim),
                        ty: Some(ElementType::S64),
                    },
                );
                out.push(AstOperation {
                    results: vec![AstOpResult {
                        name: iota_ssa.clone(),
                        count: 1,
                    }],
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("iota"),
                    op_kind: AstOpKind::HardwareOp(TpuOpCode::Iota),
                    operands: Vec::new(),
                    regions: Vec::new(),
                    successors: Vec::new(),
                    attributes: iota_attrs,
                    signature: AstFunctionType {
                        inputs: Vec::new(),
                        results: vec![idx_full_ty.clone()],
                    },
                    location: Some(MlirLocation::Unknown),
                    span: TokenSpan::default(),
                });

                let sort_base = self.fresh_ssa();
                let mut sort_attrs = BTreeMap::new();
                sort_attrs.insert(
                    String::from("dimension"),
                    MlirAttribute::Integer {
                        value: i128::from(last_dim),
                        ty: Some(ElementType::S64),
                    },
                );
                sort_attrs.insert(String::from("is_stable"), MlirAttribute::Bool(true));

                let scalar_val_ty = MlirType::RankedTensor(RankedTensorType {
                    dims: Vec::new(),
                    element_type: in_ty.element_type(),
                    encoding: None,
                    layout: None,
                });
                let scalar_idx_ty = MlirType::RankedTensor(RankedTensorType {
                    dims: Vec::new(),
                    element_type: ElementType::S32,
                    encoding: None,
                    layout: None,
                });
                let scalar_pred_ty = MlirType::RankedTensor(RankedTensorType {
                    dims: Vec::new(),
                    element_type: ElementType::Pred,
                    encoding: None,
                    layout: None,
                });
                let mut cmp_attrs = BTreeMap::new();
                cmp_attrs.insert(
                    String::from("comparison_direction"),
                    MlirAttribute::ComparisonDirection(ComparisonDirection::Gt),
                );
                cmp_attrs.insert(
                    String::from("compare_type"),
                    MlirAttribute::ComparisonType(ComparisonType::TotalOrder),
                );
                let cmp_ssa = self.fresh_ssa();
                let comparator_region = AstRegion {
                    blocks: vec![AstBlock {
                        label: Some(String::from("^bb0")),
                        arguments: vec![
                            AstBlockArg {
                                ssa_name: String::from("%lhs_v"),
                                ty: scalar_val_ty.clone(),
                                attrs: BTreeMap::new(),
                                location: None,
                            },
                            AstBlockArg {
                                ssa_name: String::from("%rhs_v"),
                                ty: scalar_val_ty.clone(),
                                attrs: BTreeMap::new(),
                                location: None,
                            },
                            AstBlockArg {
                                ssa_name: String::from("%lhs_i"),
                                ty: scalar_idx_ty.clone(),
                                attrs: BTreeMap::new(),
                                location: None,
                            },
                            AstBlockArg {
                                ssa_name: String::from("%rhs_i"),
                                ty: scalar_idx_ty,
                                attrs: BTreeMap::new(),
                                location: None,
                            },
                        ],
                        operations: vec![
                            AstOperation {
                                results: vec![AstOpResult {
                                    name: cmp_ssa.clone(),
                                    count: 1,
                                }],
                                dialect: DialectKind::StableHlo,
                                mnemonic: String::from("compare"),
                                op_kind: AstOpKind::HardwareOp(TpuOpCode::CompareGt),
                                operands: vec![
                                    AstOperand {
                                        name: String::from("%lhs_v"),
                                        result_index: None,
                                        span: TokenSpan::default(),
                                    },
                                    AstOperand {
                                        name: String::from("%rhs_v"),
                                        result_index: None,
                                        span: TokenSpan::default(),
                                    },
                                ],
                                regions: Vec::new(),
                                successors: Vec::new(),
                                attributes: cmp_attrs,
                                signature: AstFunctionType {
                                    inputs: vec![scalar_val_ty.clone(), scalar_val_ty],
                                    results: vec![scalar_pred_ty.clone()],
                                },
                                location: None,
                                span: TokenSpan::default(),
                            },
                            AstOperation {
                                results: Vec::new(),
                                dialect: DialectKind::StableHlo,
                                mnemonic: String::from("return"),
                                op_kind: AstOpKind::Return,
                                operands: vec![AstOperand {
                                    name: cmp_ssa,
                                    result_index: None,
                                    span: TokenSpan::default(),
                                }],
                                regions: Vec::new(),
                                successors: Vec::new(),
                                attributes: BTreeMap::new(),
                                signature: AstFunctionType {
                                    inputs: vec![scalar_pred_ty],
                                    results: Vec::new(),
                                },
                                location: None,
                                span: TokenSpan::default(),
                            },
                        ],
                        span: TokenSpan::default(),
                    }],
                    span: TokenSpan::default(),
                };

                out.push(AstOperation {
                    results: vec![AstOpResult {
                        name: sort_base.clone(),
                        count: 2,
                    }],
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("sort"),
                    op_kind: AstOpKind::HardwareOp(TpuOpCode::Sort),
                    operands: vec![
                        x0,
                        AstOperand {
                            name: iota_ssa,
                            result_index: None,
                            span: TokenSpan::default(),
                        },
                    ],
                    regions: vec![comparator_region],
                    successors: Vec::new(),
                    attributes: sort_attrs,
                    signature: AstFunctionType {
                        inputs: vec![in_ty.clone(), idx_full_ty.clone()],
                        results: vec![in_ty.clone(), idx_full_ty.clone()],
                    },
                    location: Some(MlirLocation::Unknown),
                    span: TokenSpan::default(),
                });

                let starts = vec![0_i64; rank];
                let mut limits: Vec<i64> = in_dims
                    .iter()
                    .map(|d| d.concrete_or_bound().max(k_val))
                    .collect();
                limits.pop();
                limits.push(k_val);
                let strides = vec![1_i64; rank];
                let mut slice_attrs = BTreeMap::new();
                slice_attrs.insert(
                    String::from("start_indices"),
                    MlirAttribute::DenseI64Array(starts),
                );
                slice_attrs.insert(
                    String::from("limit_indices"),
                    MlirAttribute::DenseI64Array(limits),
                );
                slice_attrs.insert(
                    String::from("strides"),
                    MlirAttribute::DenseI64Array(strides),
                );

                let (val_res_name, idx_res_name) = if let [r0, r1, ..] = op.results.as_slice() {
                    (r0.name.clone(), r1.name.clone())
                } else {
                    (format!("{res_name}#0"), format!("{res_name}#1"))
                };

                if op.results.len() == 1 && op.results.first().is_some_and(|r| r.count >= 2) {
                    let s_val = self.fresh_ssa();
                    let s_idx = self.fresh_ssa();
                    out.push(AstOperation {
                        results: vec![AstOpResult {
                            name: s_val.clone(),
                            count: 1,
                        }],
                        dialect: DialectKind::StableHlo,
                        mnemonic: String::from("slice"),
                        op_kind: AstOpKind::HardwareOp(TpuOpCode::Slice),
                        operands: vec![AstOperand {
                            name: sort_base.clone(),
                            result_index: Some(0),
                            span: TokenSpan::default(),
                        }],
                        regions: Vec::new(),
                        successors: Vec::new(),
                        attributes: slice_attrs.clone(),
                        signature: AstFunctionType {
                            inputs: vec![in_ty],
                            results: vec![val_out_ty.clone()],
                        },
                        location: Some(MlirLocation::Unknown),
                        span: TokenSpan::default(),
                    });
                    out.push(AstOperation {
                        results: vec![AstOpResult {
                            name: s_idx.clone(),
                            count: 1,
                        }],
                        dialect: DialectKind::StableHlo,
                        mnemonic: String::from("slice"),
                        op_kind: AstOpKind::HardwareOp(TpuOpCode::Slice),
                        operands: vec![AstOperand {
                            name: sort_base,
                            result_index: Some(1),
                            span: TokenSpan::default(),
                        }],
                        regions: Vec::new(),
                        successors: Vec::new(),
                        attributes: slice_attrs,
                        signature: AstFunctionType {
                            inputs: vec![idx_full_ty],
                            results: vec![idx_out_ty.clone()],
                        },
                        location: Some(MlirLocation::Unknown),
                        span: TokenSpan::default(),
                    });
                    out.push(AstOperation {
                        results: op.results,
                        dialect: DialectKind::StableHlo,
                        mnemonic: String::from("optimization_barrier"),
                        op_kind: AstOpKind::BarrierOrToken,
                        operands: vec![
                            AstOperand {
                                name: s_val,
                                result_index: None,
                                span: TokenSpan::default(),
                            },
                            AstOperand {
                                name: s_idx,
                                result_index: None,
                                span: TokenSpan::default(),
                            },
                        ],
                        regions: Vec::new(),
                        successors: Vec::new(),
                        attributes: BTreeMap::new(),
                        signature: AstFunctionType {
                            inputs: vec![val_out_ty.clone(), idx_out_ty.clone()],
                            results: vec![val_out_ty, idx_out_ty],
                        },
                        location: op.location,
                        span: op.span,
                    });
                } else {
                    out.push(AstOperation {
                        results: vec![AstOpResult {
                            name: val_res_name,
                            count: 1,
                        }],
                        dialect: DialectKind::StableHlo,
                        mnemonic: String::from("slice"),
                        op_kind: AstOpKind::HardwareOp(TpuOpCode::Slice),
                        operands: vec![AstOperand {
                            name: sort_base.clone(),
                            result_index: Some(0),
                            span: TokenSpan::default(),
                        }],
                        regions: Vec::new(),
                        successors: Vec::new(),
                        attributes: slice_attrs.clone(),
                        signature: AstFunctionType {
                            inputs: vec![in_ty],
                            results: vec![val_out_ty],
                        },
                        location: Some(MlirLocation::Unknown),
                        span: TokenSpan::default(),
                    });
                    out.push(AstOperation {
                        results: vec![AstOpResult {
                            name: idx_res_name,
                            count: 1,
                        }],
                        dialect: DialectKind::StableHlo,
                        mnemonic: String::from("slice"),
                        op_kind: AstOpKind::HardwareOp(TpuOpCode::Slice),
                        operands: vec![AstOperand {
                            name: sort_base,
                            result_index: Some(1),
                            span: TokenSpan::default(),
                        }],
                        regions: Vec::new(),
                        successors: Vec::new(),
                        attributes: slice_attrs,
                        signature: AstFunctionType {
                            inputs: vec![idx_full_ty],
                            results: vec![idx_out_ty],
                        },
                        location: Some(MlirLocation::Unknown),
                        span: TokenSpan::default(),
                    });
                }
            }
            "ragged_dot" => {
                let lhs = x0;
                let rhs = op.operands.get(1).unwrap_or(&lhs).clone();
                let left_ty = op.signature.inputs.first().unwrap_or(&res_ty).clone();
                let right_ty = op.signature.inputs.get(1).unwrap_or(&res_ty).clone();
                let lhs_rank = match &left_ty {
                    MlirType::RankedTensor(rt) => rt.dims.len(),
                    _ => 2,
                };
                let dot_dims = op
                    .attributes
                    .get("ragged_dot_dimension_numbers")
                    .or_else(|| op.attributes.get("dot_dimension_numbers"))
                    .and_then(|a| match a {
                        MlirAttribute::DotDimensionNumbers(d) => Some(d.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| DotDimensionNumbersAttr {
                        lhs_batching_dimensions: Vec::new(),
                        rhs_batching_dimensions: Vec::new(),
                        lhs_contracting_dimensions: vec![lhs_rank.saturating_sub(1)],
                        rhs_contracting_dimensions: vec![usize::from(
                            matches!(&right_ty, MlirType::RankedTensor(rt) if rt.dims.len() >= 3),
                        )],
                    });
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    String::from("dot_dimension_numbers"),
                    MlirAttribute::DotDimensionNumbers(dot_dims),
                );
                out.push(AstOperation {
                    results: op.results,
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("dot_general"),
                    op_kind: AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
                    operands: vec![lhs, rhs],
                    regions: Vec::new(),
                    successors: Vec::new(),
                    attributes: attrs,
                    signature: AstFunctionType {
                        inputs: vec![left_ty, right_ty],
                        results: vec![res_ty],
                    },
                    location: op.location,
                    span: op.span,
                });
            }
            "constant" | "constant_like" => {
                let val = op
                    .attributes
                    .get("value")
                    .and_then(|a| match a {
                        MlirAttribute::Float { value, .. } => Some(*value),
                        MlirAttribute::Integer { value, .. } =>
                        {
                            #[allow(clippy::cast_precision_loss)]
                            Some(*value as f64)
                        }
                        MlirAttribute::DenseElements(de) => de.values_f64.first().copied(),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                let elem_ty = res_ty.element_type();
                let ranked = match &res_ty {
                    MlirType::RankedTensor(rt) => Some(rt.clone()),
                    _ => None,
                };
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    String::from("value"),
                    MlirAttribute::DenseElements(DenseElementsAttr {
                        values_f64: vec![val],
                        raw_bytes: Some(elem_ty.pack_from_f64_slice(&[val])),
                        is_splat: true,
                        tensor_type: ranked,
                    }),
                );
                out.push(AstOperation {
                    results: op.results,
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("constant"),
                    op_kind: AstOpKind::HardwareOp(TpuOpCode::Constant),
                    operands: Vec::new(),
                    regions: Vec::new(),
                    successors: Vec::new(),
                    attributes: attrs,
                    signature: AstFunctionType {
                        inputs: Vec::new(),
                        results: vec![res_ty],
                    },
                    location: op.location,
                    span: op.span,
                });
            }
            other => {
                let mut attrs = op.attributes;
                attrs.insert(
                    String::from("call_target_name"),
                    MlirAttribute::String(format!("chlo.{other}")),
                );
                out.push(AstOperation {
                    results: op.results,
                    dialect: DialectKind::StableHlo,
                    mnemonic: String::from("custom_call"),
                    op_kind: AstOpKind::HardwareOp(TpuOpCode::CustomCall),
                    operands: op.operands,
                    regions: op.regions,
                    successors: op.successors,
                    attributes: attrs,
                    signature: op.signature,
                    location: op.location,
                    span: op.span,
                });
            }
        }
    }

    /// Recursively decomposes all `chlo` operations inside `region`.
    ///
    /// # Arguments
    ///
    /// * `region` - Mutable [`AstRegion`] to rewrite.
    fn decompose_region(&mut self, region: &mut AstRegion) {
        for block in &mut region.blocks {
            let old_ops = std::mem::take(&mut block.operations);
            let mut expanded = Vec::with_capacity(old_ops.len());
            for mut op in old_ops {
                for nested in &mut op.regions {
                    self.decompose_region(nested);
                }
                self.expand_chlo_op(op, &mut expanded);
            }
            block.operations = expanded;
        }
    }
}

/// Decomposes every `chlo.*` operation across all functions, blocks, nested regions, and
/// top-level operations in `module` into canonical `stablehlo` multi-op sequences.
///
/// # Arguments
///
/// * `module` - Mutable [`AstModule`] to legalize in place.
pub fn decompose_chlo_module(module: &mut AstModule) {
    let mut builder = ChloExpansionBuilder::default();
    for func in &mut module.functions {
        builder.decompose_region(&mut func.body);
    }
    if !module.top_level_operations.is_empty() {
        let old_top = std::mem::take(&mut module.top_level_operations);
        let mut expanded = Vec::with_capacity(old_top.len());
        for mut op in old_top {
            for nested in &mut op.regions {
                builder.decompose_region(nested);
            }
            builder.expand_chlo_op(op, &mut expanded);
        }
        module.top_level_operations = expanded;
    }
}
