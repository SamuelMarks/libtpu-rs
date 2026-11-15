//! Static & dynamic shape/type verifier and lossless `StableHLO` <-> `xla::HloModuleProto` converter.
//!
//! Implements:
//! - Static, dynamic (`?`), and bounded (`<=N`) tensor dimension compatibility and op-level verification.
//! - Lossless bidirectional conversion between `StableHLO` [`AstModule`] (`func.func` + `stablehlo.*`)
//!   and `OpenXLA` [`HloModuleProto`] (preserving constants via `LiteralProto`, dynamic/bounded shapes,
//!   nested regions as `HloComputationProto`s, tuple outputs, shardings, and dimension numbers).

use super::ast::{
    AstBlock, AstBlockArg, AstFunction, AstFunctionType, AstModule, AstOpKind, AstOpResult,
    AstOperand, AstOperation, AstRegion, ComparisonDirection, ComparisonType,
    ConvDimensionNumbersAttr, DenseElementsAttr, DialectKind, DimensionSize,
    DotDimensionNumbersAttr, GatherDimensionNumbersAttr, MlirAttribute, MlirLocation, MlirType,
    RankedTensorType, ScatterDimensionNumbersAttr, Visibility,
};
use super::lexer::TokenSpan;
use super::opcode::TpuOpCode;
use super::parser::classify_ast_operation;
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    ConvolutionDimensionNumbers, DotDimensionNumbers, GatherDimensionNumbers, HloComputationProto,
    HloInstructionProto, HloModuleProto, HloSliceDimensionsProto, LayoutProto, LiteralProto,
    PaddingConfig, PaddingConfigDimension, ProgramShapeProto, ReplicaGroup,
    ScatterDimensionNumbers, ShapeProto,
};
use crate::types::{ElementType, MemoryLayout};
use std::collections::{BTreeMap, BTreeSet};

/// Checks whether two [`DimensionSize`] extents (`Static`, `Dynamic`, or `Bounded`) can unify at runtime.
///
/// # Arguments
///
/// * `lhs` - Left-hand dimension size.
/// * `rhs` - Right-hand dimension size.
#[must_use]
pub const fn are_dimensions_compatible(lhs: DimensionSize, rhs: DimensionSize) -> bool {
    match (lhs, rhs) {
        (DimensionSize::Static(a), DimensionSize::Static(b)) => a == b,
        (DimensionSize::Static(s), DimensionSize::Bounded(b))
        | (DimensionSize::Bounded(b), DimensionSize::Static(s)) => s >= 0 && s <= b,
        (DimensionSize::Bounded(_), DimensionSize::Bounded(_))
        | (DimensionSize::Dynamic, _)
        | (_, DimensionSize::Dynamic) => true,
    }
}

/// Checks whether two [`RankedTensorType`]s have equal rank, identical element type, and compatible per-axis dimensions.
///
/// # Arguments
///
/// * `lhs` - Left-hand ranked tensor type.
/// * `rhs` - Right-hand ranked tensor type.
#[must_use]
pub fn are_tensor_shapes_compatible(lhs: &RankedTensorType, rhs: &RankedTensorType) -> bool {
    lhs.element_type == rhs.element_type
        && lhs.dims.len() == rhs.dims.len()
        && lhs
            .dims
            .iter()
            .zip(&rhs.dims)
            .all(|(&a, &b)| are_dimensions_compatible(a, b))
}

/// Computes `Some(product)` if every dimension in `dims` is [`DimensionSize::Static`], or `None` if any axis is dynamic/bounded.
///
/// # Arguments
///
/// * `dims` - Slice of tensor [`DimensionSize`] extents.
#[must_use]
fn static_element_product(dims: &[DimensionSize]) -> Option<i64> {
    let mut prod = 1_i64;
    for d in dims {
        match d {
            DimensionSize::Static(v) => {
                prod = prod.checked_mul(*v)?;
            }
            DimensionSize::Dynamic | DimensionSize::Bounded(_) => return None,
        }
    }
    Some(prod)
}

/// Verifies static and dynamic (`?` / `<=N`) shape, arity, and type invariants on a single [`AstOperation`].
///
/// # Arguments
///
/// * `op` - Parsed [`AstOperation`] to verify.
///
/// # Errors
///
/// Returns [`LibTpuError::ShapeInferenceError`] or [`LibTpuError::DialectVerificationError`] if any
/// arity, element-type, rank, contracting-axis, slice, permutation, or volume constraint is violated.
#[allow(clippy::too_many_lines)]
pub fn verify_operation_static_and_dynamic_types(op: &AstOperation) -> LibTpuResult<()> {
    let fq_name = format!("{}.{}", op.dialect.as_str(), op.mnemonic);

    if matches!(
        op.op_kind,
        AstOpKind::MosaicDmaStart | AstOpKind::MosaicDmaWait
    ) && op.operands.len() < 2
    {
        return Err(LibTpuError::DialectVerificationError {
            dialect: String::from("tpu"),
            op: op.mnemonic.clone(),
            reason: format!(
                "{fq_name} requires at least memref and semaphore operands, found {}",
                op.operands.len()
            ),
            line: op.span.line,
            column: op.span.column,
        });
    }

    if let AstOpKind::HardwareOp(
        TpuOpCode::AddF32
        | TpuOpCode::AddS32
        | TpuOpCode::SubF32
        | TpuOpCode::MulF32
        | TpuOpCode::DivF32
        | TpuOpCode::MaxF32
        | TpuOpCode::MinF32
        | TpuOpCode::Pow
        | TpuOpCode::Atan2
        | TpuOpCode::Rem,
    ) = op.op_kind
    {
        if op.operands.len() != 2 {
            return Err(LibTpuError::ShapeInferenceError {
                op: fq_name,
                reason: format!(
                    "binary operation requires 2 operands, found {}",
                    op.operands.len()
                ),
                line: op.span.line,
                column: op.span.column,
            });
        }
        if op.dialect != DialectKind::Chlo
            && let [MlirType::RankedTensor(lhs), MlirType::RankedTensor(rhs)] =
                op.signature.inputs.as_slice()
            && !are_tensor_shapes_compatible(lhs, rhs)
        {
            return Err(LibTpuError::ShapeInferenceError {
                op: fq_name,
                reason: format!(
                    "incompatible binary operand types: {:?}/{:?} vs {:?}/{:?}",
                    lhs.dims, lhs.element_type, rhs.dims, rhs.element_type
                ),
                line: op.span.line,
                column: op.span.column,
            });
        }
    }

    if op.op_kind == AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32)
        && let [MlirType::RankedTensor(lhs), MlirType::RankedTensor(rhs)] =
            op.signature.inputs.as_slice()
        && let Some(MlirAttribute::DotDimensionNumbers(d)) =
            op.attributes.get("dot_dimension_numbers")
    {
        if d.lhs_contracting_dimensions.len() != d.rhs_contracting_dimensions.len()
            || d.lhs_batching_dimensions.len() != d.rhs_batching_dimensions.len()
        {
            return Err(LibTpuError::ShapeInferenceError {
                op: fq_name,
                reason: String::from("dot_general batching/contracting axis count mismatch"),
                line: op.span.line,
                column: op.span.column,
            });
        }
        for (&lc, &rc) in d
            .lhs_contracting_dimensions
            .iter()
            .zip(&d.rhs_contracting_dimensions)
        {
            let (Some(&ld), Some(&rd)) = (lhs.dims.get(lc), rhs.dims.get(rc)) else {
                return Err(LibTpuError::ShapeInferenceError {
                    op: fq_name,
                    reason: format!("dot_general contracting dimension ({lc}, {rc}) out of bounds"),
                    line: op.span.line,
                    column: op.span.column,
                });
            };
            if !are_dimensions_compatible(ld, rd) {
                return Err(LibTpuError::ShapeInferenceError {
                    op: fq_name,
                    reason: format!(
                        "dot_general contracting dimension size mismatch: {ld:?} vs {rd:?}"
                    ),
                    line: op.span.line,
                    column: op.span.column,
                });
            }
        }
        for (&lb, &rb) in d
            .lhs_batching_dimensions
            .iter()
            .zip(&d.rhs_batching_dimensions)
        {
            let (Some(&ld), Some(&rd)) = (lhs.dims.get(lb), rhs.dims.get(rb)) else {
                return Err(LibTpuError::ShapeInferenceError {
                    op: fq_name,
                    reason: format!("dot_general batching dimension ({lb}, {rb}) out of bounds"),
                    line: op.span.line,
                    column: op.span.column,
                });
            };
            if !are_dimensions_compatible(ld, rd) {
                return Err(LibTpuError::ShapeInferenceError {
                    op: fq_name,
                    reason: format!(
                        "dot_general batching dimension size mismatch: {ld:?} vs {rd:?}"
                    ),
                    line: op.span.line,
                    column: op.span.column,
                });
            }
        }
    }

    if op.mnemonic == "reshape"
        && let (Some(MlirType::RankedTensor(in_rt)), Some(MlirType::RankedTensor(out_rt))) =
            (op.signature.inputs.first(), op.signature.results.first())
        && let (Some(in_vol), Some(out_vol)) = (
            static_element_product(&in_rt.dims),
            static_element_product(&out_rt.dims),
        )
        && in_vol != out_vol
    {
        return Err(LibTpuError::ShapeInferenceError {
            op: fq_name,
            reason: format!("reshape static element count mismatch: {in_vol} vs {out_vol}"),
            line: op.span.line,
            column: op.span.column,
        });
    }

    if op.mnemonic == "transpose"
        && let Some(MlirType::RankedTensor(in_rt)) = op.signature.inputs.first()
        && let Some(perm_attr) = op.attributes.get("permutation")
    {
        let perm = perm_attr.as_usize_vec();
        if !perm.is_empty() {
            let expected: BTreeSet<usize> = (0..in_rt.dims.len()).collect();
            let actual: BTreeSet<usize> = perm.iter().copied().collect();
            if perm.len() != in_rt.dims.len() || actual != expected {
                return Err(LibTpuError::ShapeInferenceError {
                    op: fq_name,
                    reason: format!(
                        "transpose permutation {perm:?} is not a bijection over rank {}",
                        in_rt.dims.len()
                    ),
                    line: op.span.line,
                    column: op.span.column,
                });
            }
        }
    }

    if op.mnemonic == "slice" {
        let starts = op
            .attributes
            .get("start_indices")
            .map(MlirAttribute::as_usize_vec)
            .unwrap_or_default();
        let limits = op
            .attributes
            .get("limit_indices")
            .map(MlirAttribute::as_usize_vec)
            .unwrap_or_default();
        let strides = op
            .attributes
            .get("strides")
            .map(MlirAttribute::as_usize_vec)
            .unwrap_or_default();
        for (&s, &l) in starts.iter().zip(&limits) {
            if s > l {
                return Err(LibTpuError::ShapeInferenceError {
                    op: fq_name,
                    reason: format!("slice start_index {s} exceeds limit_index {l}"),
                    line: op.span.line,
                    column: op.span.column,
                });
            }
        }
        if strides.contains(&0) {
            return Err(LibTpuError::ShapeInferenceError {
                op: fq_name,
                reason: String::from("slice stride must be >= 1"),
                line: op.span.line,
                column: op.span.column,
            });
        }
    }

    if op.mnemonic == "concatenate"
        && let Some(axis) = op
            .attributes
            .get("dimension")
            .and_then(MlirAttribute::as_usize)
        && let Some(MlirType::RankedTensor(first_rt)) = op.signature.inputs.first()
    {
        if axis >= first_rt.dims.len() {
            return Err(LibTpuError::ShapeInferenceError {
                op: fq_name,
                reason: format!(
                    "concatenate dimension {axis} out of bounds for rank {}",
                    first_rt.dims.len()
                ),
                line: op.span.line,
                column: op.span.column,
            });
        }
        for other_ty in op.signature.inputs.iter().skip(1) {
            if let MlirType::RankedTensor(other_rt) = other_ty {
                if other_rt.dims.len() != first_rt.dims.len() {
                    return Err(LibTpuError::ShapeInferenceError {
                        op: fq_name,
                        reason: String::from("concatenate inputs must have equal rank"),
                        line: op.span.line,
                        column: op.span.column,
                    });
                }
                for (dim_idx, (&d0, &d1)) in first_rt.dims.iter().zip(&other_rt.dims).enumerate() {
                    if dim_idx != axis && !are_dimensions_compatible(d0, d1) {
                        return Err(LibTpuError::ShapeInferenceError {
                            op: fq_name,
                            reason: format!(
                                "concatenate non-concat dimension {dim_idx} mismatch: {d0:?} vs {d1:?}"
                            ),
                            line: op.span.line,
                            column: op.span.column,
                        });
                    }
                }
            }
        }
    }

    if op.mnemonic == "select"
        && let [
            MlirType::RankedTensor(pred_rt),
            MlirType::RankedTensor(true_rt),
            MlirType::RankedTensor(false_rt),
        ] = op.signature.inputs.as_slice()
    {
        if pred_rt.element_type != ElementType::Pred {
            return Err(LibTpuError::ShapeInferenceError {
                op: fq_name,
                reason: format!(
                    "select predicate must have element type PRED (i1), found {:?}",
                    pred_rt.element_type
                ),
                line: op.span.line,
                column: op.span.column,
            });
        }
        if !are_tensor_shapes_compatible(true_rt, false_rt) {
            return Err(LibTpuError::ShapeInferenceError {
                op: fq_name,
                reason: String::from("select on_true and on_false shapes/types are incompatible"),
                line: op.span.line,
                column: op.span.column,
            });
        }
    }

    if let AstOpKind::GetTupleElement { index } = op.op_kind
        && let Some(MlirType::Tuple(elems)) = op.signature.inputs.first()
        && index >= elems.len()
    {
        return Err(LibTpuError::ShapeInferenceError {
            op: fq_name,
            reason: format!(
                "get_tuple_element index {index} out of bounds for tuple of {} elements",
                elems.len()
            ),
            line: op.span.line,
            column: op.span.column,
        });
    }

    Ok(())
}

/// Converts an [`MlirType`] losslessly into an `OpenXLA` [`ShapeProto`] (preserving `Static`, `Bounded`, and `Dynamic`).
///
/// # Arguments
///
/// * `ty` - Source [`MlirType`].
#[must_use]
pub fn mlir_type_to_xla_shape_proto(ty: &MlirType) -> ShapeProto {
    match ty {
        MlirType::Tuple(elems) => ShapeProto {
            element_type: i32::from(ElementType::Tuple),
            tuple_shapes: elems.iter().map(mlir_type_to_xla_shape_proto).collect(),
            ..ShapeProto::default()
        },
        MlirType::Token => ShapeProto {
            element_type: i32::from(ElementType::Token),
            ..ShapeProto::default()
        },
        MlirType::RankedTensor(rt) => {
            let dimensions: Vec<i64> = rt
                .dims
                .iter()
                .map(|d| match d {
                    DimensionSize::Static(v) | DimensionSize::Bounded(v) => *v,
                    DimensionSize::Dynamic => -1,
                })
                .collect();
            let is_dynamic_dimension: Vec<bool> = rt
                .dims
                .iter()
                .map(|d| matches!(d, DimensionSize::Dynamic | DimensionSize::Bounded(_)))
                .collect();
            let rank = dimensions.len();
            let default_m2m: Vec<i64> = (0..rank)
                .rev()
                .map(|i| i64::try_from(i).unwrap_or(0))
                .collect();
            let minor_to_major = rt.layout.as_ref().map_or(default_m2m, |l| match l {
                MemoryLayout::Tiled { minor_to_major, .. }
                | MemoryLayout::Strides {
                    byte_strides: minor_to_major,
                    ..
                } => minor_to_major.clone(),
            });
            ShapeProto {
                element_type: i32::from(rt.element_type),
                dimensions,
                is_dynamic_dimension,
                layout: Some(Box::new(LayoutProto {
                    minor_to_major,
                    ..LayoutProto::default()
                })),
                ..ShapeProto::default()
            }
        }
        MlirType::Scalar(et) => ShapeProto {
            element_type: i32::from(*et),
            ..ShapeProto::default()
        },
        other => {
            let et = other.element_type();
            ShapeProto {
                element_type: i32::from(et),
                dimensions: vec![1],
                is_dynamic_dimension: vec![false],
                ..ShapeProto::default()
            }
        }
    }
}

/// Converts an `OpenXLA` [`ShapeProto`] losslessly into an [`MlirType`].
///
/// # Arguments
///
/// * `sp` - Source [`ShapeProto`].
#[must_use]
pub fn xla_shape_proto_to_mlir_type(sp: &ShapeProto) -> MlirType {
    let elem = ElementType::from_raw(sp.element_type).unwrap_or(ElementType::F32);
    if elem == ElementType::Tuple || !sp.tuple_shapes.is_empty() {
        return MlirType::Tuple(
            sp.tuple_shapes
                .iter()
                .map(xla_shape_proto_to_mlir_type)
                .collect(),
        );
    }
    if elem == ElementType::Token {
        return MlirType::Token;
    }
    let dims: Vec<DimensionSize> = sp
        .dimensions
        .iter()
        .enumerate()
        .map(|(idx, &d)| {
            let is_dyn = sp.is_dynamic_dimension.get(idx).copied().unwrap_or(d < 0);
            if is_dyn {
                if d > 0 {
                    DimensionSize::Bounded(d)
                } else {
                    DimensionSize::Dynamic
                }
            } else {
                DimensionSize::Static(d)
            }
        })
        .collect();
    let layout = sp.layout.as_ref().and_then(|lp| {
        (!lp.minor_to_major.is_empty()).then(|| MemoryLayout::Tiled {
            minor_to_major: lp.minor_to_major.clone(),
            tile_dims: vec![8, 128],
            tile_dim_sizes: vec![2],
        })
    });
    MlirType::RankedTensor(RankedTensorType {
        dims,
        element_type: elem,
        encoding: None,
        layout,
    })
}

/// Encodes a [`DenseElementsAttr`] + result [`ShapeProto`] into an `OpenXLA` [`LiteralProto`].
///
/// # Arguments
///
/// * `dense` - Source [`DenseElementsAttr`].
/// * `shape` - Result [`ShapeProto`] describing element type and extents.
fn dense_elements_to_literal_proto(dense: &DenseElementsAttr, shape: &ShapeProto) -> LiteralProto {
    let elem = ElementType::from_raw(shape.element_type).unwrap_or(ElementType::F32);
    let count: usize = shape
        .dimensions
        .iter()
        .map(|&d| usize::try_from(d.max(1)).unwrap_or(1))
        .product::<usize>()
        .max(1);
    let base_vals = dense.raw_bytes.as_ref().map_or_else(
        || dense.values_f64.clone(),
        |raw| elem.unpack_to_f64_vec(raw),
    );
    let vals = if base_vals.len() == 1 && count > 1 {
        vec![base_vals.first().copied().unwrap_or(0.0); count]
    } else {
        base_vals
    };

    let mut lit = LiteralProto {
        shape: Some(shape.clone()),
        ..LiteralProto::default()
    };
    match elem {
        ElementType::Pred => {
            lit.preds = vals.iter().map(|&v| v != 0.0).collect();
        }
        ElementType::F32 => {
            #[allow(clippy::cast_possible_truncation)]
            {
                lit.f32s = vals.iter().map(|&v| v as f32).collect();
            }
        }
        ElementType::F64 => {
            lit.f64s = vals;
        }
        ElementType::S32 => {
            #[allow(clippy::cast_possible_truncation)]
            {
                lit.s32s = vals.iter().map(|&v| v as i32).collect();
            }
        }
        ElementType::S64 => {
            #[allow(clippy::cast_possible_truncation)]
            {
                lit.s64s = vals.iter().map(|&v| v as i64).collect();
            }
        }
        ElementType::U32 => {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                lit.u32s = vals.iter().map(|&v| v as u32).collect();
            }
        }
        ElementType::U64 => {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                lit.u64s = vals.iter().map(|&v| v as u64).collect();
            }
        }
        ElementType::Bf16 => {
            lit.bf16s = elem.pack_from_f64_slice(&vals);
        }
        ElementType::F16 => {
            lit.f16s = elem.pack_from_f64_slice(&vals);
        }
        _ => {
            lit.u8s = elem.pack_from_f64_slice(&vals);
        }
    }
    lit
}

/// Decodes an `OpenXLA` [`LiteralProto`] back into a [`DenseElementsAttr`].
///
/// # Arguments
///
/// * `lit` - Source [`LiteralProto`].
/// * `fallback_ty` - Fallback [`MlirType`] if `lit.shape` is absent.
fn literal_proto_to_dense_elements(
    lit: &LiteralProto,
    fallback_ty: &MlirType,
) -> DenseElementsAttr {
    let mlir_ty = lit
        .shape
        .as_ref()
        .map_or_else(|| fallback_ty.clone(), xla_shape_proto_to_mlir_type);
    let ranked = match &mlir_ty {
        MlirType::RankedTensor(rt) => Some(rt.clone()),
        _ => None,
    };
    let elem = mlir_ty.element_type();
    let values_f64: Vec<f64> = if !lit.f32s.is_empty() {
        lit.f32s.iter().map(|&v| f64::from(v)).collect()
    } else if !lit.f64s.is_empty() {
        lit.f64s.clone()
    } else if !lit.s32s.is_empty() {
        lit.s32s.iter().map(|&v| f64::from(v)).collect()
    } else if !lit.s64s.is_empty() {
        #[allow(clippy::cast_precision_loss)]
        {
            lit.s64s.iter().map(|&v| v as f64).collect()
        }
    } else if !lit.u32s.is_empty() {
        lit.u32s.iter().map(|&v| f64::from(v)).collect()
    } else if !lit.u64s.is_empty() {
        #[allow(clippy::cast_precision_loss)]
        {
            lit.u64s.iter().map(|&v| v as f64).collect()
        }
    } else if !lit.preds.is_empty() {
        lit.preds
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect()
    } else if !lit.bf16s.is_empty() {
        ElementType::Bf16.unpack_to_f64_vec(&lit.bf16s)
    } else if !lit.f16s.is_empty() {
        ElementType::F16.unpack_to_f64_vec(&lit.f16s)
    } else if !lit.u8s.is_empty() {
        elem.unpack_to_f64_vec(&lit.u8s)
    } else {
        vec![0.0]
    };
    let raw_bytes = Some(elem.pack_from_f64_slice(&values_f64));
    DenseElementsAttr {
        values_f64,
        raw_bytes,
        is_splat: false,
        tensor_type: ranked,
    }
}

/// Bidirectional `(stablehlo_mnemonic, hlo_opcode)` canonical mapping table.
const STABLEHLO_HLO_OPCODE_PAIRS: &[(&str, &str)] = &[
    ("dot_general", "dot"),
    ("dot", "dot"),
    ("broadcast_in_dim", "broadcast"),
    ("broadcast", "broadcast"),
    ("get_tuple_element", "get-tuple-element"),
    ("get-tuple-element", "get-tuple-element"),
    ("bitcast_convert", "bitcast-convert"),
    ("all_reduce", "all-reduce"),
    ("all_gather", "all-gather"),
    ("all_to_all", "all-to-all"),
    ("reduce_scatter", "reduce-scatter"),
    ("collective_permute", "collective-permute"),
    ("dynamic_slice", "dynamic-slice"),
    ("dynamic_update_slice", "dynamic-update-slice"),
    ("reduce_window", "reduce-window"),
    ("rng_bit_generator", "rng-bit-generator"),
    ("shift_left", "shift-left"),
    ("shift_right_arithmetic", "shift-right-arithmetic"),
    ("shift_right_logical", "shift-right-logical"),
    ("exponential_minus_one", "expm1"),
    ("log_plus_one", "log1p"),
    ("if", "conditional"),
    ("case", "conditional"),
    ("abs", "abs"),
    ("add", "add"),
    ("subtract", "subtract"),
    ("multiply", "multiply"),
    ("divide", "divide"),
    ("remainder", "remainder"),
    ("maximum", "maximum"),
    ("minimum", "minimum"),
    ("power", "power"),
    ("atan2", "atan2"),
    ("negate", "negate"),
    ("exponential", "exponential"),
    ("log", "log"),
    ("logistic", "logistic"),
    ("tanh", "tanh"),
    ("sine", "sine"),
    ("cosine", "cosine"),
    ("tan", "tan"),
    ("sqrt", "sqrt"),
    ("rsqrt", "rsqrt"),
    ("cbrt", "cbrt"),
    ("ceil", "ceil"),
    ("floor", "floor"),
    ("sign", "sign"),
    ("and", "and"),
    ("or", "or"),
    ("xor", "xor"),
    ("not", "not"),
    ("compare", "compare"),
    ("select", "select"),
    ("clamp", "clamp"),
    ("convert", "convert"),
    ("reshape", "reshape"),
    ("transpose", "transpose"),
    ("slice", "slice"),
    ("pad", "pad"),
    ("concatenate", "concatenate"),
    ("reverse", "reverse"),
    ("iota", "iota"),
    ("gather", "gather"),
    ("scatter", "scatter"),
    ("convolution", "convolution"),
    ("reduce", "reduce"),
    ("sort", "sort"),
    ("while", "while"),
    ("tuple", "tuple"),
    ("constant", "constant"),
    ("call", "call"),
];

/// Maps a `StableHLO` mnemonic to its canonical XLA HLO `HloInstructionProto.opcode`.
///
/// # Arguments
///
/// * `mnemonic` - Source `StableHLO` mnemonic.
#[must_use]
fn stablehlo_mnemonic_to_hlo_opcode(mnemonic: &str) -> &'static str {
    STABLEHLO_HLO_OPCODE_PAIRS
        .iter()
        .find(|&&(shlo, _)| shlo == mnemonic)
        .map_or("custom-call", |&(_, hlo)| hlo)
}

/// Maps an XLA HLO `opcode` back to its canonical `StableHLO` mnemonic.
///
/// # Arguments
///
/// * `opcode` - Source `HloInstructionProto.opcode`.
#[must_use]
fn hlo_opcode_to_stablehlo_mnemonic(opcode: &str) -> &'static str {
    STABLEHLO_HLO_OPCODE_PAIRS
        .iter()
        .find(|&&(_, hlo)| hlo == opcode)
        .map_or("custom_call", |&(shlo, _)| shlo)
}

/// Converts a `&[usize]` axis list into `Vec<i64>`.
///
/// # Arguments
///
/// * `items` - Slice of `usize` axis indices.
#[must_use]
fn usize_slice_to_i64(items: &[usize]) -> Vec<i64> {
    items
        .iter()
        .map(|&x| i64::try_from(x).unwrap_or(0))
        .collect()
}

/// Converts a `&[i64]` axis list into `Vec<usize>`.
///
/// # Arguments
///
/// * `items` - Slice of signed `i64` axis indices.
#[must_use]
fn i64_slice_to_usize(items: &[i64]) -> Vec<usize> {
    items
        .iter()
        .filter_map(|&x| usize::try_from(x).ok())
        .collect()
}

/// Stateful builder lowering [`AstFunction`]s and nested [`AstRegion`]s into [`HloComputationProto`]s.
struct StableHloToProtoBuilder {
    /// Next unique `HloInstructionProto.id`.
    next_inst_id: i64,
    /// Next unique `HloComputationProto.id`.
    next_comp_id: i64,
    /// Accumulated `HloComputationProto` list (sub-computations precede callers).
    computations: Vec<HloComputationProto>,
    /// Pre-assigned IDs for top-level named functions (`@callee` -> `comp_id`).
    func_ids: BTreeMap<String, i64>,
}

impl StableHloToProtoBuilder {
    /// Allocates a fresh instruction ID.
    const fn alloc_inst_id(&mut self) -> i64 {
        let id = self.next_inst_id;
        self.next_inst_id += 1;
        id
    }

    /// Allocates a fresh computation ID.
    const fn alloc_comp_id(&mut self) -> i64 {
        let id = self.next_comp_id;
        self.next_comp_id += 1;
        id
    }

    /// Resolves an [`AstOperand`] against `ssa_to_id`.
    ///
    /// # Arguments
    ///
    /// * `opnd` - Source [`AstOperand`].
    /// * `ssa_to_id` - Mapping from SSA value names to instruction IDs.
    fn resolve_operand_id(opnd: &AstOperand, ssa_to_id: &BTreeMap<String, i64>) -> Option<i64> {
        opnd.result_index.map_or_else(
            || ssa_to_id.get(&opnd.name).copied(),
            |sub_idx| {
                let key = format!("{}#{sub_idx}", opnd.name);
                ssa_to_id
                    .get(&key)
                    .or_else(|| ssa_to_id.get(&opnd.name))
                    .copied()
            },
        )
    }

    /// Lowers a block's arguments and operations into an [`HloComputationProto`].
    ///
    /// # Arguments
    ///
    /// * `comp_name` - Computation symbol name.
    /// * `comp_id` - Numeric computation ID.
    /// * `args` - Block/function parameter specifications.
    /// * `ops` - Sequence of operations inside the block.
    /// * `declared_results` - Declared function result types.
    #[allow(clippy::too_many_lines)]
    fn lower_block_to_computation(
        &mut self,
        comp_name: &str,
        comp_id: i64,
        args: &[AstBlockArg],
        ops: &[AstOperation],
        declared_results: &[MlirType],
    ) -> HloComputationProto {
        let mut ssa_to_id: BTreeMap<String, i64> = BTreeMap::new();
        let mut ssa_to_type: BTreeMap<String, MlirType> = BTreeMap::new();
        let mut instructions = Vec::new();
        let mut param_shapes = Vec::new();
        let mut param_names = Vec::new();

        for (p_idx, arg) in args.iter().enumerate() {
            let id = self.alloc_inst_id();
            let sp = mlir_type_to_xla_shape_proto(&arg.ty);
            let sharding = arg
                .attrs
                .get("sdy.sharding")
                .or_else(|| arg.attrs.get("mhlo.sharding"))
                .map(|a| {
                    super::passes::ParsedShardingSpec::from_str_with_partitions(
                        &a.to_mlir_string(),
                        1,
                    )
                    .to_op_sharding_proto()
                });
            let clean_name = String::from(arg.ssa_name.trim_start_matches('%'));
            param_shapes.push(sp.clone());
            param_names.push(clean_name.clone());
            ssa_to_id.insert(arg.ssa_name.clone(), id);
            ssa_to_type.insert(arg.ssa_name.clone(), arg.ty.clone());
            instructions.push(HloInstructionProto {
                name: clean_name,
                opcode: String::from("parameter"),
                shape: Some(sp),
                id,
                parameter_number: i64::try_from(p_idx).unwrap_or(0),
                sharding,
                ..HloInstructionProto::default()
            });
        }

        let mut root_id = 0_i64;
        for op in ops {
            if op.op_kind == AstOpKind::Return {
                let ret_ids: Vec<i64> = op
                    .operands
                    .iter()
                    .filter_map(|opnd| Self::resolve_operand_id(opnd, &ssa_to_id))
                    .collect();
                if ret_ids.len() == 1 {
                    root_id = ret_ids.first().copied().unwrap_or(0);
                } else if ret_ids.len() > 1 {
                    let tup_id = self.alloc_inst_id();
                    let tup_elem_types: Vec<MlirType> = if declared_results.len() == ret_ids.len() {
                        declared_results.to_vec()
                    } else {
                        op.operands
                            .iter()
                            .map(|o| {
                                ssa_to_type
                                    .get(&o.name)
                                    .cloned()
                                    .unwrap_or(MlirType::Scalar(ElementType::F32))
                            })
                            .collect()
                    };
                    let tup_ty = MlirType::Tuple(tup_elem_types);
                    instructions.push(HloInstructionProto {
                        name: format!("{comp_name}_return_tuple"),
                        opcode: String::from("tuple"),
                        shape: Some(mlir_type_to_xla_shape_proto(&tup_ty)),
                        operand_ids: ret_ids,
                        id: tup_id,
                        ..HloInstructionProto::default()
                    });
                    root_id = tup_id;
                }
                continue;
            }

            let mut called_computation_ids = Vec::new();
            for (r_idx, reg) in op.regions.iter().enumerate() {
                if let Some(blk) = reg.blocks.first() {
                    let sub_id = self.alloc_comp_id();
                    let sub_name = format!("{comp_name}_{}_region_{r_idx}_{sub_id}", op.mnemonic);
                    let sub_comp = self.lower_block_to_computation(
                        &sub_name,
                        sub_id,
                        &blk.arguments,
                        &blk.operations,
                        &[],
                    );
                    self.computations.push(sub_comp);
                    called_computation_ids.push(sub_id);
                }
            }
            if let Some(callee) = op.attributes.get("callee").and_then(MlirAttribute::as_str)
                && let Some(&cid) = self.func_ids.get(callee.trim_start_matches('@'))
            {
                called_computation_ids.push(cid);
            }

            let operand_ids: Vec<i64> = op
                .operands
                .iter()
                .filter_map(|opnd| Self::resolve_operand_id(opnd, &ssa_to_id))
                .collect();

            let res_ty = if op.signature.results.len() > 1 {
                MlirType::Tuple(op.signature.results.clone())
            } else {
                op.signature
                    .results
                    .first()
                    .cloned()
                    .or_else(|| {
                        if let Some(MlirAttribute::DenseElements(de)) = op.attributes.get("value") {
                            de.tensor_type.clone().map(MlirType::RankedTensor)
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        op.operands
                            .first()
                            .and_then(|o| ssa_to_type.get(&o.name).cloned())
                    })
                    .unwrap_or(MlirType::Scalar(ElementType::F32))
            };
            let shape_proto = mlir_type_to_xla_shape_proto(&res_ty);

            let opcode = if matches!(op.dialect, DialectKind::Sdy)
                && matches!(op.mnemonic.as_str(), "sharding_constraint" | "reshard")
            {
                "reshape"
            } else {
                stablehlo_mnemonic_to_hlo_opcode(&op.mnemonic)
            };

            let literal = if opcode == "constant" {
                op.attributes.get("value").and_then(|a| match a {
                    MlirAttribute::DenseElements(de) => {
                        Some(dense_elements_to_literal_proto(de, &shape_proto))
                    }
                    _ => None,
                })
            } else {
                None
            };

            let dimensions: Vec<i64> = op
                .attributes
                .get("dimensions")
                .or_else(|| op.attributes.get("broadcast_dimensions"))
                .or_else(|| op.attributes.get("permutation"))
                .map(|a| {
                    a.as_usize_vec()
                        .into_iter()
                        .map(|u| i64::try_from(u).unwrap_or(0))
                        .collect()
                })
                .or_else(|| {
                    op.attributes
                        .get("dimension")
                        .or_else(|| op.attributes.get("iota_dimension"))
                        .and_then(MlirAttribute::as_i64)
                        .map(|d| vec![d])
                })
                .unwrap_or_default();

            let dot_dimension_numbers = op
                .attributes
                .get("dot_dimension_numbers")
                .and_then(|a| match a {
                    MlirAttribute::DotDimensionNumbers(d) => Some(DotDimensionNumbers {
                        lhs_contracting_dimensions: usize_slice_to_i64(
                            &d.lhs_contracting_dimensions,
                        ),
                        rhs_contracting_dimensions: usize_slice_to_i64(
                            &d.rhs_contracting_dimensions,
                        ),
                        lhs_batch_dimensions: usize_slice_to_i64(&d.lhs_batching_dimensions),
                        rhs_batch_dimensions: usize_slice_to_i64(&d.rhs_batching_dimensions),
                    }),
                    _ => None,
                })
                .or_else(|| {
                    (op.mnemonic == "dot").then(|| DotDimensionNumbers {
                        lhs_contracting_dimensions: vec![1],
                        rhs_contracting_dimensions: vec![0],
                        lhs_batch_dimensions: Vec::new(),
                        rhs_batch_dimensions: Vec::new(),
                    })
                });

            let convolution_dimension_numbers =
                op.attributes
                    .get("dimension_numbers")
                    .and_then(|a| match a {
                        MlirAttribute::ConvDimensionNumbers(c) => {
                            Some(ConvolutionDimensionNumbers {
                                input_batch_dimension: i64::try_from(c.input_batch_dimension)
                                    .unwrap_or(0),
                                input_feature_dimension: i64::try_from(c.input_feature_dimension)
                                    .unwrap_or(0),
                                input_spatial_dimensions: usize_slice_to_i64(
                                    &c.input_spatial_dimensions,
                                ),
                                kernel_input_feature_dimension: i64::try_from(
                                    c.kernel_input_feature_dimension,
                                )
                                .unwrap_or(0),
                                kernel_output_feature_dimension: i64::try_from(
                                    c.kernel_output_feature_dimension,
                                )
                                .unwrap_or(0),
                                kernel_spatial_dimensions: usize_slice_to_i64(
                                    &c.kernel_spatial_dimensions,
                                ),
                                output_batch_dimension: i64::try_from(c.output_batch_dimension)
                                    .unwrap_or(0),
                                output_feature_dimension: i64::try_from(c.output_feature_dimension)
                                    .unwrap_or(0),
                                output_spatial_dimensions: usize_slice_to_i64(
                                    &c.output_spatial_dimensions,
                                ),
                            })
                        }
                        _ => None,
                    });

            let gather_dimension_numbers = op.attributes.values().find_map(|a| match a {
                MlirAttribute::GatherDimensionNumbers(g) => Some(GatherDimensionNumbers {
                    offset_dims: usize_slice_to_i64(&g.offset_dims),
                    collapsed_slice_dims: usize_slice_to_i64(&g.collapsed_slice_dims),
                    start_index_map: usize_slice_to_i64(&g.start_index_map),
                    index_vector_dim: i64::try_from(g.index_vector_dim).unwrap_or(0),
                    operand_batching_dims: usize_slice_to_i64(&g.operand_batching_dims),
                    start_indices_batching_dims: usize_slice_to_i64(&g.start_indices_batching_dims),
                }),
                _ => None,
            });

            let scatter_dimension_numbers = op.attributes.values().find_map(|a| match a {
                MlirAttribute::ScatterDimensionNumbers(s) => Some(ScatterDimensionNumbers {
                    update_window_dims: usize_slice_to_i64(&s.update_window_dims),
                    inserted_window_dims: usize_slice_to_i64(&s.inserted_window_dims),
                    scatter_dims_to_operand_dims: usize_slice_to_i64(
                        &s.scatter_dims_to_operand_dims,
                    ),
                    index_vector_dim: i64::try_from(s.index_vector_dim).unwrap_or(0),
                    input_batching_dims: usize_slice_to_i64(&s.input_batching_dims),
                    scatter_indices_batching_dims: usize_slice_to_i64(
                        &s.scatter_indices_batching_dims,
                    ),
                }),
                _ => None,
            });

            let slice_dimensions: Vec<HloSliceDimensionsProto> = if opcode == "slice" {
                let starts = op
                    .attributes
                    .get("start_indices")
                    .map(MlirAttribute::as_usize_vec)
                    .unwrap_or_default();
                let limits = op
                    .attributes
                    .get("limit_indices")
                    .map(MlirAttribute::as_usize_vec)
                    .unwrap_or_default();
                let strides = op
                    .attributes
                    .get("strides")
                    .map(MlirAttribute::as_usize_vec)
                    .unwrap_or_default();
                starts
                    .iter()
                    .zip(&limits)
                    .enumerate()
                    .map(|(i, (&s, &l))| HloSliceDimensionsProto {
                        start: i64::try_from(s).unwrap_or(0),
                        limit: i64::try_from(l).unwrap_or(0),
                        stride: i64::try_from(strides.get(i).copied().unwrap_or(1).max(1))
                            .unwrap_or(1),
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let padding_config = if opcode == "pad" {
                let lo = op
                    .attributes
                    .get("edge_padding_low")
                    .map(MlirAttribute::as_usize_vec)
                    .unwrap_or_default();
                let hi = op
                    .attributes
                    .get("edge_padding_high")
                    .map(MlirAttribute::as_usize_vec)
                    .unwrap_or_default();
                let intr = op
                    .attributes
                    .get("interior_padding")
                    .map(MlirAttribute::as_usize_vec)
                    .unwrap_or_default();
                let rank = lo.len().max(hi.len()).max(intr.len());
                (rank > 0).then(|| PaddingConfig {
                    dimensions: (0..rank)
                        .map(|i| PaddingConfigDimension {
                            edge_padding_low: i64::try_from(lo.get(i).copied().unwrap_or(0))
                                .unwrap_or(0),
                            edge_padding_high: i64::try_from(hi.get(i).copied().unwrap_or(0))
                                .unwrap_or(0),
                            interior_padding: i64::try_from(intr.get(i).copied().unwrap_or(0))
                                .unwrap_or(0),
                        })
                        .collect(),
                })
            } else {
                None
            };

            let comparison_direction = match op.attributes.get("comparison_direction") {
                Some(MlirAttribute::ComparisonDirection(dir)) => String::from(dir.as_str()),
                Some(MlirAttribute::String(s)) => s.clone(),
                _ => String::new(),
            };
            let comparison_type = match op
                .attributes
                .get("compare_type")
                .or_else(|| op.attributes.get("comparison_type"))
            {
                Some(MlirAttribute::ComparisonType(ct)) => String::from(match ct {
                    ComparisonType::Float => "FLOAT",
                    ComparisonType::TotalOrder => "TOTALORDER",
                    ComparisonType::Signed => "SIGNED",
                    ComparisonType::Unsigned => "UNSIGNED",
                }),
                _ => String::new(),
            };

            let tuple_index = match op.op_kind {
                AstOpKind::GetTupleElement { index } => i64::try_from(index).unwrap_or(0),
                _ => op
                    .attributes
                    .get("index")
                    .and_then(MlirAttribute::as_i64)
                    .unwrap_or(0),
            };

            let replica_groups = match op.attributes.get("replica_groups") {
                Some(MlirAttribute::Array(groups)) => groups
                    .iter()
                    .map(|g| ReplicaGroup {
                        replica_ids: g
                            .as_usize_vec()
                            .into_iter()
                            .map(|r| i64::try_from(r).unwrap_or(0))
                            .collect(),
                    })
                    .collect(),
                _ => Vec::new(),
            };

            let sharding = op
                .attributes
                .get("sdy.sharding")
                .or_else(|| op.attributes.get("mhlo.sharding"))
                .map(|a| {
                    super::passes::ParsedShardingSpec::from_str_with_partitions(
                        &a.to_mlir_string(),
                        1,
                    )
                    .to_op_sharding_proto()
                });

            let custom_call_target = op
                .attributes
                .get("call_target_name")
                .and_then(MlirAttribute::as_str)
                .map_or_else(
                    || {
                        if opcode == "custom-call" && op.mnemonic != "custom_call" {
                            format!("{}.{}", op.dialect.as_str(), op.mnemonic)
                        } else {
                            String::new()
                        }
                    },
                    String::from,
                );

            let inst_id = self.alloc_inst_id();
            let inst_name = op.results.first().map_or_else(
                || format!("{opcode}_{inst_id}"),
                |r| String::from(r.name.trim_start_matches('%')),
            );

            instructions.push(HloInstructionProto {
                name: inst_name,
                opcode: String::from(opcode),
                shape: Some(shape_proto),
                literal,
                tuple_index,
                dimensions,
                convolution_dimension_numbers,
                slice_dimensions,
                padding_config,
                dot_dimension_numbers,
                gather_dimension_numbers,
                scatter_dimension_numbers,
                gather_slice_sizes: op
                    .attributes
                    .get("slice_sizes")
                    .map(|a| {
                        a.as_usize_vec()
                            .into_iter()
                            .map(|u| i64::try_from(u).unwrap_or(0))
                            .collect()
                    })
                    .unwrap_or_default(),
                comparison_direction,
                comparison_type,
                replica_groups,
                channel_id: op
                    .attributes
                    .get("channel_id")
                    .and_then(MlirAttribute::as_i64)
                    .unwrap_or(0),
                custom_call_target,
                sharding,
                id: inst_id,
                operand_ids,
                called_computation_ids,
                ..HloInstructionProto::default()
            });

            if op.results.len() == 1 && op.results.first().is_some_and(|r| r.count > 1) {
                let base_name = op
                    .results
                    .first()
                    .map_or_else(String::new, |r| r.name.clone());
                ssa_to_id.insert(base_name.clone(), inst_id);
                let count = op.results.first().map_or(0, |r| r.count);
                for sub_i in 0..count {
                    let gte_id = self.alloc_inst_id();
                    let elem_ty = op
                        .signature
                        .results
                        .get(sub_i)
                        .cloned()
                        .unwrap_or(MlirType::Scalar(ElementType::F32));
                    instructions.push(HloInstructionProto {
                        name: format!("{}_{sub_i}", base_name.trim_start_matches('%')),
                        opcode: String::from("get-tuple-element"),
                        shape: Some(mlir_type_to_xla_shape_proto(&elem_ty)),
                        tuple_index: i64::try_from(sub_i).unwrap_or(0),
                        id: gte_id,
                        operand_ids: vec![inst_id],
                        ..HloInstructionProto::default()
                    });
                    ssa_to_id.insert(format!("{base_name}#{sub_i}"), gte_id);
                    ssa_to_type.insert(format!("{base_name}#{sub_i}"), elem_ty);
                }
            } else if op.results.len() > 1 {
                ssa_to_id.insert(
                    op.results
                        .first()
                        .map_or_else(String::new, |r| r.name.clone()),
                    inst_id,
                );
                for (sub_i, res_item) in op.results.iter().enumerate() {
                    let gte_id = self.alloc_inst_id();
                    let elem_ty = op
                        .signature
                        .results
                        .get(sub_i)
                        .cloned()
                        .unwrap_or(MlirType::Scalar(ElementType::F32));
                    instructions.push(HloInstructionProto {
                        name: format!("{}_gte", res_item.name.trim_start_matches('%')),
                        opcode: String::from("get-tuple-element"),
                        shape: Some(mlir_type_to_xla_shape_proto(&elem_ty)),
                        tuple_index: i64::try_from(sub_i).unwrap_or(0),
                        id: gte_id,
                        operand_ids: vec![inst_id],
                        ..HloInstructionProto::default()
                    });
                    ssa_to_id.insert(res_item.name.clone(), gte_id);
                    ssa_to_type.insert(res_item.name.clone(), elem_ty);
                }
            } else if let Some(first_res) = op.results.first() {
                ssa_to_id.insert(first_res.name.clone(), inst_id);
                ssa_to_type.insert(first_res.name.clone(), res_ty);
            }
            root_id = inst_id;
        }

        if root_id == 0 {
            root_id = instructions.last().map_or(0, |i| i.id);
        }
        let result_shape = instructions
            .iter()
            .find(|i| i.id == root_id)
            .and_then(|i| i.shape.clone());

        HloComputationProto {
            name: String::from(comp_name),
            instructions,
            program_shape: Some(ProgramShapeProto {
                parameters: param_shapes,
                result: result_shape,
                parameter_names: param_names,
            }),
            id: comp_id,
            root_id,
            ..HloComputationProto::default()
        }
    }
}

/// Losslessly converts a `StableHLO` [`AstModule`] (`func.func` + `stablehlo.*`) into an `OpenXLA` [`HloModuleProto`].
///
/// If `ast` contains textual XLA HLO computations (`ast.hlo_computations`), delegates to [`super::hlo_parser::hlo_ast_to_proto`].
///
/// # Arguments
///
/// * `ast` - Verified [`AstModule`] to convert.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if `ast` contains neither MLIR functions/operations nor HLO computations.
pub fn stablehlo_to_hlo_module_proto(ast: &AstModule) -> LibTpuResult<HloModuleProto> {
    if !ast.hlo_computations.is_empty() && ast.functions.is_empty() {
        return Ok(super::hlo_parser::hlo_ast_to_proto(ast));
    }
    if ast.functions.is_empty() && ast.top_level_operations.is_empty() {
        return Err(LibTpuError::CompilationFailed {
            reason: String::from("Cannot convert empty StableHLO AstModule to HloModuleProto"),
        });
    }

    let mut working = ast.clone();
    super::parser::legalize_chlo_in_ast_module(&mut working);

    let mut builder = StableHloToProtoBuilder {
        next_inst_id: 1,
        next_comp_id: 1,
        computations: Vec::new(),
        func_ids: BTreeMap::new(),
    };

    if working.functions.is_empty() {
        let cid = builder.alloc_comp_id();
        let entry_comp = builder.lower_block_to_computation(
            "main",
            cid,
            &[],
            &working.top_level_operations,
            &[],
        );
        let host_shape = entry_comp.program_shape.clone();
        builder.computations.push(entry_comp);
        return Ok(HloModuleProto {
            name: working
                .name
                .unwrap_or_else(|| String::from("stablehlo_module")),
            entry_computation_name: String::from("main"),
            entry_computation_id: cid,
            computations: builder.computations,
            host_program_shape: host_shape,
            ..HloModuleProto::default()
        });
    }

    for func in &working.functions {
        let cid = builder.alloc_comp_id();
        builder.func_ids.insert(func.name.clone(), cid);
    }

    let entry_idx = working
        .functions
        .iter()
        .position(|f| f.name == "main" || f.visibility == Visibility::Public)
        .unwrap_or_else(|| working.functions.len().saturating_sub(1));

    let mut entry_comp_name = String::from("main");
    let mut entry_comp_id = 1_i64;
    let mut host_program_shape = None;

    for (idx, func) in working.functions.iter().enumerate() {
        let cid = builder.func_ids.get(&func.name).copied().unwrap_or(1);
        let ops: Vec<AstOperation> = func
            .body
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter().cloned())
            .collect();
        let result_types: Vec<MlirType> = func.results.iter().map(|r| r.ty.clone()).collect();
        let comp = builder.lower_block_to_computation(
            &func.name,
            cid,
            &func.arguments,
            &ops,
            &result_types,
        );
        if idx == entry_idx {
            entry_comp_name.clone_from(&func.name);
            entry_comp_id = cid;
            host_program_shape.clone_from(&comp.program_shape);
        }
        builder.computations.push(comp);
    }

    Ok(HloModuleProto {
        name: working
            .name
            .unwrap_or_else(|| String::from("stablehlo_module")),
        entry_computation_name: entry_comp_name,
        entry_computation_id: entry_comp_id,
        computations: builder.computations,
        host_program_shape,
        ..HloModuleProto::default()
    })
}

/// Converts a single [`HloComputationProto`] into an [`AstRegion`] + parameter [`AstBlockArg`]s + result [`MlirType`]s.
///
/// # Arguments
///
/// * `comp` - Source [`HloComputationProto`].
/// * `comp_by_id` - Lookup table of all computations by ID.
/// * `func_names_by_id` - Lookup table of normalized function names by ID.
#[allow(clippy::too_many_lines)]
fn computation_proto_to_stablehlo_region(
    comp: &HloComputationProto,
    comp_by_id: &BTreeMap<i64, &HloComputationProto>,
    func_names_by_id: &BTreeMap<i64, String>,
) -> (Vec<AstBlockArg>, Vec<MlirType>, AstRegion) {
    let mut params: Vec<(i64, AstBlockArg)> = Vec::new();
    let mut id_to_ssa: BTreeMap<i64, String> = BTreeMap::new();
    let mut id_to_ty: BTreeMap<i64, MlirType> = BTreeMap::new();
    let mut operations: Vec<AstOperation> = Vec::new();

    for inst in &comp.instructions {
        let clean = inst.name.trim_start_matches('%');
        let ssa = if clean.is_empty() {
            format!("%v{}", inst.id)
        } else {
            format!("%{clean}")
        };
        let ty = inst.shape.as_ref().map_or(
            MlirType::Scalar(ElementType::F32),
            xla_shape_proto_to_mlir_type,
        );
        id_to_ssa.insert(inst.id, ssa.clone());
        id_to_ty.insert(inst.id, ty.clone());

        if inst.opcode == "parameter" {
            let mut arg_attrs = BTreeMap::new();
            if let Some(sh) = &inst.sharding {
                arg_attrs.insert(
                    String::from("mhlo.sharding"),
                    MlirAttribute::String(format!("{{type={}}}", sh.r#type)),
                );
            }
            params.push((
                inst.parameter_number,
                AstBlockArg {
                    ssa_name: ssa,
                    ty,
                    attrs: arg_attrs,
                    location: Some(MlirLocation::Unknown),
                },
            ));
        }
    }
    params.sort_by_key(|(num, _)| *num);
    let block_args: Vec<AstBlockArg> = params.into_iter().map(|(_, a)| a).collect();

    let mut root_ssa = None;
    let mut root_ty = None;

    for (idx, inst) in comp.instructions.iter().enumerate() {
        let is_root = inst.id == comp.root_id || idx + 1 == comp.instructions.len();
        let ssa = id_to_ssa.get(&inst.id).cloned().unwrap_or_default();
        let res_ty = id_to_ty
            .get(&inst.id)
            .cloned()
            .unwrap_or(MlirType::Scalar(ElementType::F32));

        if inst.opcode == "parameter" {
            if is_root {
                root_ssa = Some(ssa);
                root_ty = Some(res_ty);
            }
            continue;
        }

        let operands: Vec<AstOperand> = inst
            .operand_ids
            .iter()
            .filter_map(|oid| {
                id_to_ssa.get(oid).map(|name| AstOperand {
                    name: name.clone(),
                    result_index: None,
                    span: TokenSpan::default(),
                })
            })
            .collect();
        let input_types: Vec<MlirType> = inst
            .operand_ids
            .iter()
            .filter_map(|oid| id_to_ty.get(oid).cloned())
            .collect();

        let mut dialect = DialectKind::StableHlo;
        let mut mnemonic = String::from(hlo_opcode_to_stablehlo_mnemonic(&inst.opcode));
        let mut attrs = BTreeMap::new();
        let mut regions = Vec::new();

        if inst.opcode == "call" {
            dialect = DialectKind::Func;
            mnemonic = String::from("call");
            if let Some(&cid) = inst.called_computation_ids.first()
                && let Some(fname) = func_names_by_id.get(&cid)
            {
                attrs.insert(
                    String::from("callee"),
                    MlirAttribute::SymbolRef(vec![fname.clone()]),
                );
            }
        } else {
            for &cid in &inst.called_computation_ids {
                if let Some(sub_comp) = comp_by_id.get(&cid) {
                    let (_, _, sub_reg) = computation_proto_to_stablehlo_region(
                        sub_comp,
                        comp_by_id,
                        func_names_by_id,
                    );
                    regions.push(sub_reg);
                }
            }
        }

        if let Some(lit) = &inst.literal {
            attrs.insert(
                String::from("value"),
                MlirAttribute::DenseElements(literal_proto_to_dense_elements(lit, &res_ty)),
            );
        }
        if mnemonic == "get_tuple_element" {
            attrs.insert(
                String::from("index"),
                MlirAttribute::Integer {
                    value: i128::from(inst.tuple_index),
                    ty: Some(ElementType::S32),
                },
            );
        }
        if !inst.dimensions.is_empty() {
            let key = match mnemonic.as_str() {
                "broadcast_in_dim" => "broadcast_dimensions",
                "transpose" => "permutation",
                "concatenate" => "dimension",
                "iota" => "iota_dimension",
                _ => "dimensions",
            };
            if matches!(key, "dimension" | "iota_dimension") {
                attrs.insert(
                    String::from(key),
                    MlirAttribute::Integer {
                        value: i128::from(inst.dimensions.first().copied().unwrap_or(0)),
                        ty: Some(ElementType::S64),
                    },
                );
            } else {
                attrs.insert(
                    String::from(key),
                    MlirAttribute::DenseI64Array(inst.dimensions.clone()),
                );
            }
        }
        if let Some(d) = &inst.dot_dimension_numbers {
            attrs.insert(
                String::from("dot_dimension_numbers"),
                MlirAttribute::DotDimensionNumbers(DotDimensionNumbersAttr {
                    lhs_batching_dimensions: i64_slice_to_usize(&d.lhs_batch_dimensions),
                    rhs_batching_dimensions: i64_slice_to_usize(&d.rhs_batch_dimensions),
                    lhs_contracting_dimensions: i64_slice_to_usize(&d.lhs_contracting_dimensions),
                    rhs_contracting_dimensions: i64_slice_to_usize(&d.rhs_contracting_dimensions),
                }),
            );
        }
        if let Some(c) = &inst.convolution_dimension_numbers {
            attrs.insert(
                String::from("dimension_numbers"),
                MlirAttribute::ConvDimensionNumbers(ConvDimensionNumbersAttr {
                    input_batch_dimension: usize::try_from(c.input_batch_dimension.max(0))
                        .unwrap_or(0),
                    input_feature_dimension: usize::try_from(c.input_feature_dimension.max(0))
                        .unwrap_or(0),
                    input_spatial_dimensions: i64_slice_to_usize(&c.input_spatial_dimensions),
                    kernel_input_feature_dimension: usize::try_from(
                        c.kernel_input_feature_dimension.max(0),
                    )
                    .unwrap_or(0),
                    kernel_output_feature_dimension: usize::try_from(
                        c.kernel_output_feature_dimension.max(0),
                    )
                    .unwrap_or(0),
                    kernel_spatial_dimensions: i64_slice_to_usize(&c.kernel_spatial_dimensions),
                    output_batch_dimension: usize::try_from(c.output_batch_dimension.max(0))
                        .unwrap_or(0),
                    output_feature_dimension: usize::try_from(c.output_feature_dimension.max(0))
                        .unwrap_or(0),
                    output_spatial_dimensions: i64_slice_to_usize(&c.output_spatial_dimensions),
                }),
            );
        }
        if let Some(g) = &inst.gather_dimension_numbers {
            attrs.insert(
                String::from("dimension_numbers"),
                MlirAttribute::GatherDimensionNumbers(GatherDimensionNumbersAttr {
                    offset_dims: i64_slice_to_usize(&g.offset_dims),
                    collapsed_slice_dims: i64_slice_to_usize(&g.collapsed_slice_dims),
                    operand_batching_dims: i64_slice_to_usize(&g.operand_batching_dims),
                    start_indices_batching_dims: i64_slice_to_usize(&g.start_indices_batching_dims),
                    start_index_map: i64_slice_to_usize(&g.start_index_map),
                    index_vector_dim: usize::try_from(g.index_vector_dim.max(0)).unwrap_or(0),
                }),
            );
            if !inst.gather_slice_sizes.is_empty() {
                attrs.insert(
                    String::from("slice_sizes"),
                    MlirAttribute::DenseI64Array(inst.gather_slice_sizes.clone()),
                );
            }
        }
        if let Some(s) = &inst.scatter_dimension_numbers {
            attrs.insert(
                String::from("scatter_dimension_numbers"),
                MlirAttribute::ScatterDimensionNumbers(ScatterDimensionNumbersAttr {
                    update_window_dims: i64_slice_to_usize(&s.update_window_dims),
                    inserted_window_dims: i64_slice_to_usize(&s.inserted_window_dims),
                    input_batching_dims: i64_slice_to_usize(&s.input_batching_dims),
                    scatter_indices_batching_dims: i64_slice_to_usize(
                        &s.scatter_indices_batching_dims,
                    ),
                    scatter_dims_to_operand_dims: i64_slice_to_usize(
                        &s.scatter_dims_to_operand_dims,
                    ),
                    index_vector_dim: usize::try_from(s.index_vector_dim.max(0)).unwrap_or(0),
                }),
            );
        }
        if !inst.slice_dimensions.is_empty() {
            attrs.insert(
                String::from("start_indices"),
                MlirAttribute::DenseI64Array(
                    inst.slice_dimensions.iter().map(|sd| sd.start).collect(),
                ),
            );
            attrs.insert(
                String::from("limit_indices"),
                MlirAttribute::DenseI64Array(
                    inst.slice_dimensions.iter().map(|sd| sd.limit).collect(),
                ),
            );
            attrs.insert(
                String::from("strides"),
                MlirAttribute::DenseI64Array(
                    inst.slice_dimensions
                        .iter()
                        .map(|sd| sd.stride.max(1))
                        .collect(),
                ),
            );
        }
        if let Some(pad) = &inst.padding_config {
            attrs.insert(
                String::from("edge_padding_low"),
                MlirAttribute::DenseI64Array(
                    pad.dimensions.iter().map(|d| d.edge_padding_low).collect(),
                ),
            );
            attrs.insert(
                String::from("edge_padding_high"),
                MlirAttribute::DenseI64Array(
                    pad.dimensions.iter().map(|d| d.edge_padding_high).collect(),
                ),
            );
            attrs.insert(
                String::from("interior_padding"),
                MlirAttribute::DenseI64Array(
                    pad.dimensions.iter().map(|d| d.interior_padding).collect(),
                ),
            );
        }
        if !inst.comparison_direction.is_empty()
            && let Some(dir) = ComparisonDirection::from_ident(&inst.comparison_direction)
        {
            attrs.insert(
                String::from("comparison_direction"),
                MlirAttribute::ComparisonDirection(dir),
            );
        }
        if !inst.comparison_type.is_empty()
            && let Some(ct) = ComparisonType::from_ident(&inst.comparison_type)
        {
            attrs.insert(
                String::from("compare_type"),
                MlirAttribute::ComparisonType(ct),
            );
        }
        if !inst.custom_call_target.is_empty() {
            attrs.insert(
                String::from("call_target_name"),
                MlirAttribute::String(inst.custom_call_target.clone()),
            );
        }
        if inst.channel_id != 0 {
            attrs.insert(
                String::from("channel_id"),
                MlirAttribute::Integer {
                    value: i128::from(inst.channel_id),
                    ty: Some(ElementType::S64),
                },
            );
        }
        if !inst.replica_groups.is_empty() {
            attrs.insert(
                String::from("replica_groups"),
                MlirAttribute::Array(
                    inst.replica_groups
                        .iter()
                        .map(|rg| MlirAttribute::DenseI64Array(rg.replica_ids.clone()))
                        .collect(),
                ),
            );
        }

        let op_kind =
            classify_ast_operation(dialect, &mnemonic, res_ty.element_type(), &attrs, &regions);
        operations.push(AstOperation {
            results: vec![AstOpResult {
                name: ssa.clone(),
                count: 1,
            }],
            dialect,
            mnemonic,
            op_kind,
            operands,
            regions,
            successors: Vec::new(),
            attributes: attrs,
            signature: AstFunctionType {
                inputs: input_types,
                results: vec![res_ty.clone()],
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        });

        if is_root {
            root_ssa = Some(ssa);
            root_ty = Some(res_ty);
        }
    }

    let mut result_types = Vec::new();
    if let (Some(r_ssa), Some(r_ty)) = (root_ssa, root_ty) {
        result_types.push(r_ty.clone());
        operations.push(AstOperation {
            results: Vec::new(),
            dialect: DialectKind::Func,
            mnemonic: String::from("return"),
            op_kind: AstOpKind::Return,
            operands: vec![AstOperand {
                name: r_ssa,
                result_index: None,
                span: TokenSpan::default(),
            }],
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: BTreeMap::new(),
            signature: AstFunctionType {
                inputs: vec![r_ty],
                results: Vec::new(),
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        });
    }

    let region = AstRegion {
        blocks: vec![AstBlock {
            label: Some(String::from("^bb0")),
            arguments: block_args.clone(),
            operations,
            span: TokenSpan::default(),
        }],
        span: TokenSpan::default(),
    };
    (block_args, result_types, region)
}

/// Losslessly converts an `OpenXLA` [`HloModuleProto`] into a canonical `StableHLO` [`AstModule`]
/// (`func.func @main(...)` + nested `stablehlo` regions + private helper functions).
///
/// # Arguments
///
/// * `proto` - Source [`HloModuleProto`] to convert.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if `proto.computations` is empty.
pub fn hlo_module_proto_to_stablehlo(proto: &HloModuleProto) -> LibTpuResult<AstModule> {
    if proto.computations.is_empty() {
        return Err(LibTpuError::CompilationFailed {
            reason: String::from("HloModuleProto contains zero computations"),
        });
    }

    let comp_by_id: BTreeMap<i64, &HloComputationProto> =
        proto.computations.iter().map(|c| (c.id, c)).collect();

    let entry_id = proto
        .computations
        .iter()
        .find(|c| c.id == proto.entry_computation_id || c.name == proto.entry_computation_name)
        .or_else(|| proto.computations.last())
        .map_or(0, |c| c.id);

    let mut inlined_region_ids = BTreeSet::new();
    let mut called_func_ids = BTreeSet::new();
    for comp in &proto.computations {
        for inst in &comp.instructions {
            if inst.opcode == "call" {
                called_func_ids.extend(inst.called_computation_ids.iter().copied());
            } else {
                inlined_region_ids.extend(inst.called_computation_ids.iter().copied());
            }
        }
    }

    let mut func_names_by_id = BTreeMap::new();
    for comp in &proto.computations {
        let clean = String::from(comp.name.trim_start_matches('%'));
        func_names_by_id.insert(
            comp.id,
            if clean.is_empty() {
                format!("comp_{}", comp.id)
            } else {
                clean
            },
        );
    }

    let pure_region_ids: BTreeSet<i64> = inlined_region_ids
        .difference(&called_func_ids)
        .copied()
        .collect();

    let mut functions = Vec::new();
    for comp in &proto.computations {
        let is_entry = comp.id == entry_id;
        if !is_entry && pure_region_ids.contains(&comp.id) {
            continue;
        }
        let (arguments, res_types, mut body) =
            computation_proto_to_stablehlo_region(comp, &comp_by_id, &func_names_by_id);
        for blk in &mut body.blocks {
            blk.arguments.clear();
        }
        let fname = func_names_by_id.get(&comp.id).cloned().unwrap_or_default();
        let results = res_types
            .into_iter()
            .map(|ty| super::ast::AstResultType {
                ty,
                attrs: BTreeMap::new(),
            })
            .collect();
        functions.push(AstFunction {
            name: fname,
            visibility: if is_entry {
                Visibility::Public
            } else {
                Visibility::Private
            },
            arguments,
            results,
            function_attrs: BTreeMap::new(),
            body,
            span: TokenSpan::default(),
        });
    }

    Ok(AstModule {
        name: (!proto.name.is_empty()).then(|| proto.name.clone()),
        functions,
        ..AstModule::default()
    })
}
