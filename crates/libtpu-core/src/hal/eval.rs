//! Tensor operation evaluation routines (`MXU`, `VALU`, `EUP`, and ICI collectives).
//!
//! Implements instruction bundle evaluation and collective execution for [`SimulatedTpuHal`].

use super::SimulatedTpuHal;
use crate::compiler::TpuOpCode;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{DeviceAddress, DeviceId, ElementType, MemoryLayout, Shape};

impl SimulatedTpuHal {
    /// Evaluates a 2x2 or elementwise lower-triangular Cholesky decomposition (`A = L * L^T`).
    ///
    /// # Arguments
    ///
    /// * `vals` - Unpacked input matrix elements.
    pub(super) fn eval_cholesky(vals: &[f64]) -> Vec<f64> {
        if let &[v00, _, v10, v11] = vals {
            let a00 = v00.max(1e-12).sqrt();
            let l10 = v10 / a00;
            let l11 = l10.mul_add(-l10, v11).max(1e-12).sqrt();
            vec![a00, 0.0, l10, l11]
        } else {
            vals.iter().map(|&v| v.max(0.0).sqrt()).collect()
        }
    }

    /// Evaluates a real-part Discrete Fourier Transform projection (`X[k] = sum_n x[n] cos(2*pi*k*n/N)`).
    ///
    /// # Arguments
    ///
    /// * `vals` - Unpacked time-domain signal elements.
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn eval_fft(vals: &[f64]) -> Vec<f64> {
        let n_f = vals.len().max(1) as f64;
        (0..vals.len())
            .map(|k| {
                vals.iter()
                    .enumerate()
                    .map(|(n, &x)| {
                        let angle = -2.0 * std::f64::consts::PI * (k as f64) * (n as f64) / n_f;
                        x * angle.cos()
                    })
                    .sum()
            })
            .collect()
    }

    /// Evaluates a 2x2 or diagonal lower-triangular linear system (`L * X = B`).
    ///
    /// # Arguments
    ///
    /// * `v0` - Unpacked lower-triangular matrix `L`.
    /// * `v1` - Unpacked right-hand side matrix `B`.
    pub(super) fn eval_triangular_solve(v0: &[f64], v1: &[f64]) -> Vec<f64> {
        if let (&[m00, _, l10, m11], &[b00, b01, b10, b11]) = (v0, v1) {
            let l00 = if m00.abs() < 1e-12 { 1.0 } else { m00 };
            let l11 = if m11.abs() < 1e-12 { 1.0 } else { m11 };
            let x00 = b00 / l00;
            let x01 = b01 / l00;
            let x10 = l10.mul_add(-x00, b10) / l11;
            let x11 = l10.mul_add(-x01, b11) / l11;
            return vec![x00, x01, x10, x11];
        }
        v0.iter()
            .zip(v1.iter())
            .map(|(&l, &b)| b / if l.abs() < 1e-12 { 1.0 } else { l })
            .collect()
    }

    /// Evaluates a single unary operation across `input_bytes`.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Unary [`TpuOpCode`].
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `input_bytes` - Source HBM byte buffer.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub(super) fn eval_unary_op(
        opcode: TpuOpCode,
        elem_type: ElementType,
        input_bytes: &[u8],
    ) -> Vec<u8> {
        if matches!(
            opcode,
            TpuOpCode::Identity
                | TpuOpCode::AllReduceSumF32
                | TpuOpCode::AllGather
                | TpuOpCode::ReduceScatter
                | TpuOpCode::CollectivePermute
                | TpuOpCode::AllToAll
                | TpuOpCode::CustomCall
                | TpuOpCode::Reshape
        ) {
            return input_bytes.to_vec();
        }
        let vals = elem_type.unpack_to_f64_vec(input_bytes);
        let mapped: Vec<f64> = match opcode {
            TpuOpCode::NegF32 => vals.iter().map(|&v| -v).collect(),
            TpuOpCode::ExpF32 => vals.iter().map(|&v| v.exp()).collect(),
            TpuOpCode::Expm1 => vals.iter().map(|&v| v.exp_m1()).collect(),
            TpuOpCode::TanhF32 => vals.iter().map(|&v| v.tanh()).collect(),
            TpuOpCode::Tan => vals.iter().map(|&v| v.tan()).collect(),
            TpuOpCode::Cbrt => vals.iter().map(|&v| v.cbrt()).collect(),
            TpuOpCode::Logistic => vals.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
            TpuOpCode::Sign => vals
                .iter()
                .map(|&v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            TpuOpCode::IsFinite => vals
                .iter()
                .map(|&v| f64::from(u8::from(v.is_finite())))
                .collect(),
            TpuOpCode::RoundNearestEven => vals.iter().map(|&v| v.round_ties_even()).collect(),
            TpuOpCode::RoundNearestAfz => vals.iter().map(|&v| v.round()).collect(),
            TpuOpCode::UniformQuantize => vals
                .iter()
                .map(|&v| (v * 127.0).round().clamp(-128.0, 127.0))
                .collect(),
            TpuOpCode::UniformDequantize => vals.iter().map(|&v| v / 127.0).collect(),
            TpuOpCode::Abs => vals.iter().map(|&v| v.abs()).collect(),
            TpuOpCode::Log => vals.iter().map(|&v| v.max(1e-12).ln()).collect(),
            TpuOpCode::Log1p => vals.iter().map(|&v| v.max(-0.999_999).ln_1p()).collect(),
            TpuOpCode::Sqrt => vals.iter().map(|&v| v.max(0.0).sqrt()).collect(),
            TpuOpCode::Rsqrt => vals.iter().map(|&v| 1.0 / v.max(1e-12).sqrt()).collect(),
            TpuOpCode::Sine => vals.iter().map(|&v| v.sin()).collect(),
            TpuOpCode::Cosine => vals.iter().map(|&v| v.cos()).collect(),
            TpuOpCode::Floor => vals.iter().map(|&v| v.floor()).collect(),
            TpuOpCode::Ceil => vals.iter().map(|&v| v.ceil()).collect(),
            TpuOpCode::Not => vals
                .iter()
                .map(|&v| f64::from(u8::from(v == 0.0)))
                .collect(),
            TpuOpCode::PopulationCount => vals
                .iter()
                .map(|&v| f64::from((v as i32 as u32).count_ones()))
                .collect(),
            TpuOpCode::CountLeadingZeros => vals
                .iter()
                .map(|&v| f64::from((v as i32 as u32).leading_zeros()))
                .collect(),
            TpuOpCode::Sort => {
                let mut sorted = vals;
                sorted.sort_by(f64::total_cmp);
                sorted
            }
            TpuOpCode::TopK => {
                let mut desc = vals;
                desc.sort_by(|a, b| b.total_cmp(a));
                desc
            }
            TpuOpCode::Cholesky => Self::eval_cholesky(&vals),
            TpuOpCode::Fft => Self::eval_fft(&vals),
            TpuOpCode::RngBitGenerator => {
                let seed = vals.first().copied().unwrap_or(1.0) as u64;
                (0..vals.len())
                    .map(|i| {
                        let mixed = seed
                            .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                            ^ 0xBF58_476D_1CE4_E5B9;
                        f64::from((mixed & 0xFFFF) as u32)
                    })
                    .collect()
            }
            TpuOpCode::ReduceWindow => vals
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let next = vals.get(i + 1).copied().unwrap_or(v);
                    v.max(next)
                })
                .collect(),
            TpuOpCode::Iota => (0..vals.len()).map(|i| i as f64).collect(),
            TpuOpCode::Reverse => vals.iter().copied().rev().collect(),
            TpuOpCode::ReduceMax => {
                let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                vec![mx; vals.len()]
            }
            TpuOpCode::ReduceMin => {
                let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
                vec![mn; vals.len()]
            }
            _ => {
                let sum: f64 = vals.iter().sum();
                vec![sum; vals.len()]
            }
        };
        elem_type.pack_from_f64_slice(&mapped)
    }

    /// Evaluates a general N-dimensional matrix contraction (`dot_general`) when explicit
    /// contracting axes (`dot_lhs_contracting_dims` / `dot_rhs_contracting_dims`) are provided.
    ///
    /// # Arguments
    ///
    /// * `elem_type` - Output [`ElementType`].
    /// * `v0` - Unpacked LHS elements.
    /// * `v1` - Unpacked RHS elements.
    /// * `lhs_sh` - LHS operand [`Shape`].
    /// * `rhs_sh` - RHS operand [`Shape`].
    /// * `attrs` - [`crate::compiler::InstructionAttrs`] with contraction and batching axes.
    #[allow(for_loops_over_fallibles)]
    pub(super) fn eval_dot_general_nd(
        elem_type: ElementType,
        v0: &[f64],
        v1: &[f64],
        lhs_sh: &Shape,
        rhs_sh: &Shape,
        attrs: &crate::compiler::InstructionAttrs,
    ) -> Vec<u8> {
        let l_batch = &attrs.dot_lhs_batch_dims;
        let r_batch = &attrs.dot_rhs_batch_dims;
        let l_cont = &attrs.dot_lhs_contracting_dims;
        let r_cont = &attrs.dot_rhs_contracting_dims;
        let l_free: Vec<usize> = (0..lhs_sh.rank())
            .filter(|d| !l_batch.contains(d) && !l_cont.contains(d))
            .collect();
        let r_free: Vec<usize> = (0..rhs_sh.rank())
            .filter(|d| !r_batch.contains(d) && !r_cont.contains(d))
            .collect();

        let mut out_dims = Vec::new();
        for &bd in l_batch {
            out_dims.push(lhs_sh.dims.get(bd).copied().unwrap_or(1));
        }
        for &fd in &l_free {
            out_dims.push(lhs_sh.dims.get(fd).copied().unwrap_or(1));
        }
        for &fd in &r_free {
            out_dims.push(rhs_sh.dims.get(fd).copied().unwrap_or(1));
        }
        let out_sh = Shape {
            element_type: elem_type,
            dynamic_dims: vec![false; out_dims.len()],
            layout: MemoryLayout::default_tpu_tiled(out_dims.len()),
            dims: out_dims,
        };
        let mut cont_dims = Vec::with_capacity(l_cont.len());
        for &cd in l_cont {
            cont_dims.push(lhs_sh.dims.get(cd).copied().unwrap_or(1));
        }
        let cont_sh = Shape {
            element_type: elem_type,
            dynamic_dims: vec![false; cont_dims.len()],
            layout: MemoryLayout::default_tpu_tiled(cont_dims.len()),
            dims: cont_dims,
        };

        let out_count = out_sh.element_count();
        let cont_count = cont_sh.element_count();
        let mut out_vals = Vec::with_capacity(out_count);
        for dst_idx in 0..out_count {
            let dst_coords = out_sh.linear_to_coords(dst_idx);
            let mut acc = 0.0_f64;
            for c_idx in 0..cont_count {
                let c_coords = cont_sh.linear_to_coords(c_idx);
                let mut l_coords = vec![0_usize; lhs_sh.rank()];
                let mut r_coords = vec![0_usize; rhs_sh.rank()];
                for (bi, (&lb, &rb)) in l_batch.iter().zip(r_batch.iter()).enumerate() {
                    let b_val = dst_coords.get(bi).copied().unwrap_or(0);
                    for slot in l_coords.get_mut(lb) {
                        *slot = b_val;
                    }
                    for slot in r_coords.get_mut(rb) {
                        *slot = b_val;
                    }
                }
                for (fi, &lf) in l_free.iter().enumerate() {
                    let f_val = dst_coords.get(l_batch.len() + fi).copied().unwrap_or(0);
                    for slot in l_coords.get_mut(lf) {
                        *slot = f_val;
                    }
                }
                for (fi, &rf) in r_free.iter().enumerate() {
                    let f_val = dst_coords
                        .get(l_batch.len() + l_free.len() + fi)
                        .copied()
                        .unwrap_or(0);
                    for slot in r_coords.get_mut(rf) {
                        *slot = f_val;
                    }
                }
                for (ci, (&lc, &rc)) in l_cont.iter().zip(r_cont.iter()).enumerate() {
                    let c_val = c_coords.get(ci).copied().unwrap_or(0);
                    for slot in l_coords.get_mut(lc) {
                        *slot = c_val;
                    }
                    for slot in r_coords.get_mut(rc) {
                        *slot = c_val;
                    }
                }
                let l_lin = lhs_sh.coords_to_linear(&l_coords).unwrap_or(0);
                let r_lin = rhs_sh.coords_to_linear(&r_coords).unwrap_or(0);
                let lv = v0.get(l_lin).copied().unwrap_or(0.0);
                let rv = v1.get(r_lin).copied().unwrap_or(0.0);
                acc = lv.mul_add(rv, acc);
            }
            out_vals.push(acc);
        }
        elem_type.pack_from_f64_slice(&out_vals)
    }

    /// Evaluates a general N-dimensional matrix contraction (`dot_general`) supporting arbitrary
    /// batching dimensions (`dot_lhs_batch_dims`, `dot_rhs_batch_dims`), contracting dimensions
    /// (`dot_lhs_contracting_dims`, `dot_rhs_contracting_dims`), and free dimensions.
    ///
    /// # Arguments
    ///
    /// * `elem_type` - Output [`ElementType`].
    /// * `v0` - Unpacked LHS elements.
    /// * `v1` - Unpacked RHS elements.
    /// * `attrs` - [`crate::compiler::InstructionAttrs`] with input/output shapes and contraction axes.
    pub(super) fn eval_dot_general(
        elem_type: ElementType,
        v0: &[f64],
        v1: &[f64],
        attrs: &crate::compiler::InstructionAttrs,
    ) -> Vec<u8> {
        if let (Some(lhs_sh), Some(rhs_sh)) =
            (attrs.input_shapes.first(), attrs.input_shapes.get(1))
            && !attrs.dot_lhs_contracting_dims.is_empty()
        {
            return Self::eval_dot_general_nd(elem_type, v0, v1, lhs_sh, rhs_sh, attrs);
        }

        let mut batch = 1_usize;
        let mut m = v0.len().min(v1.len());
        let mut k = 1_usize;
        let mut n = 1_usize;
        let mut d = 1_usize;
        while d.saturating_mul(d) < m {
            d += 1;
        }
        if d > 1 && d.saturating_mul(d) == m {
            m = d;
            k = d;
            n = d;
        }
        let lhs_dims = attrs
            .input_shapes
            .first()
            .map_or(&[][..], |s| s.dims.as_slice());
        let rhs_dims = attrs
            .input_shapes
            .get(1)
            .map_or(&[][..], |s| s.dims.as_slice());
        if lhs_dims.len() >= 2 && rhs_dims.len() >= 2 {
            let l_rank = lhs_dims.len();
            let r_rank = rhs_dims.len();
            for &bd in lhs_dims.iter().take(l_rank - 2) {
                batch = batch.saturating_mul(usize::try_from(bd).unwrap_or(1));
            }
            m = usize::try_from(lhs_dims.get(l_rank - 2).copied().unwrap_or(0)).unwrap_or(0);
            k = usize::try_from(lhs_dims.get(l_rank - 1).copied().unwrap_or(0)).unwrap_or(0);
            n = usize::try_from(rhs_dims.get(r_rank - 1).copied().unwrap_or(0)).unwrap_or(0);
        }
        let mut out_vals = Vec::with_capacity(batch * m * n);
        for b_idx in 0..batch {
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0.0_f64;
                    for idx in 0..k {
                        let lv = v0.get((b_idx * m + r) * k + idx).copied().unwrap_or(0.0);
                        let rv = v1.get((b_idx * k + idx) * n + c).copied().unwrap_or(0.0);
                        acc = lv.mul_add(rv, acc);
                    }
                    out_vals.push(acc);
                }
            }
        }
        elem_type.pack_from_f64_slice(&out_vals)
    }

    /// Evaluates multi-dimensional edge and interior tensor padding (`stablehlo.pad`).
    ///
    /// # Arguments
    ///
    /// * `elem_type` - Output [`ElementType`].
    /// * `v0` - Unpacked operand tensor elements.
    /// * `pad_val` - Scalar padding value.
    /// * `attrs` - [`crate::compiler::InstructionAttrs`] with `pad_edge_low`, `pad_edge_high`, and `pad_interior`.
    pub(super) fn eval_pad_with_attrs(
        elem_type: ElementType,
        v0: &[f64],
        pad_val: f64,
        attrs: &crate::compiler::InstructionAttrs,
    ) -> Vec<u8> {
        let Some(in_sh) = attrs.input_shapes.first() else {
            let padded: Vec<f64> = v0.iter().map(|&x| x + pad_val).collect();
            return elem_type.pack_from_f64_slice(&padded);
        };
        let rank = in_sh.rank();
        let mut out_dims = Vec::with_capacity(rank);
        for axis in 0..rank {
            let d = usize::try_from(in_sh.dims.get(axis).copied().unwrap_or(1)).unwrap_or(1);
            let low = attrs.pad_edge_low.get(axis).copied().unwrap_or(0);
            let high = attrs.pad_edge_high.get(axis).copied().unwrap_or(0);
            let inter = attrs.pad_interior.get(axis).copied().unwrap_or(0);
            let span = d + d.saturating_sub(1) * inter + low + high;
            out_dims.push(i64::try_from(span).unwrap_or(1));
        }
        let out_sh = Shape {
            element_type: elem_type,
            dynamic_dims: vec![false; rank],
            layout: MemoryLayout::default_tpu_tiled(rank),
            dims: out_dims,
        };
        let out_count = out_sh.element_count();
        let mut out_vals = Vec::with_capacity(out_count);
        for dst_idx in 0..out_count {
            let dst_coords = out_sh.linear_to_coords(dst_idx);
            let mut src_coords = Vec::with_capacity(rank);
            let mut is_pad = false;
            for (axis, &dc) in dst_coords.iter().enumerate() {
                let d = usize::try_from(in_sh.dims.get(axis).copied().unwrap_or(1)).unwrap_or(1);
                let low = attrs.pad_edge_low.get(axis).copied().unwrap_or(0);
                let inter = attrs.pad_interior.get(axis).copied().unwrap_or(0);
                let stride = inter + 1;
                if dc < low {
                    is_pad = true;
                    break;
                }
                let shifted = dc - low;
                if shifted % stride != 0 {
                    is_pad = true;
                    break;
                }
                if (shifted / stride) >= d {
                    is_pad = true;
                    break;
                }
                src_coords.push(shifted / stride);
            }
            if is_pad {
                out_vals.push(pad_val);
            } else {
                let src_idx = in_sh.coords_to_linear(&src_coords).unwrap_or(0);
                out_vals.push(v0.get(src_idx).copied().unwrap_or(pad_val));
            }
        }
        elem_type.pack_from_f64_slice(&out_vals)
    }

    /// Evaluates axis-selective tensor reduction (`reduce_sum`, `reduce_max`, `reduce_min`) when
    /// `attrs.reduce_dims` and `attrs.input_shapes` are specified.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Reduction [`TpuOpCode`].
    /// * `elem_type` - Output [`ElementType`].
    /// * `vals` - Unpacked operand tensor elements.
    /// * `attrs` - [`crate::compiler::InstructionAttrs`] with `reduce_dims` and input/output shapes.
    #[allow(for_loops_over_fallibles)]
    pub(super) fn eval_reduce_with_attrs(
        opcode: TpuOpCode,
        elem_type: ElementType,
        vals: &[f64],
        attrs: &crate::compiler::InstructionAttrs,
    ) -> Vec<u8> {
        if !attrs.reduce_dims.is_empty() {
            let default_sh = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false],
                layout: MemoryLayout::default_tpu_tiled(1),
                dims: vec![i64::try_from(vals.len().max(1)).unwrap_or(1)],
            };
            let in_sh = attrs.input_shapes.first().unwrap_or(&default_sh);
            let kept_axes: Vec<usize> = (0..in_sh.rank())
                .filter(|a| !attrs.reduce_dims.contains(a))
                .collect();
            let kept_dims: Vec<i64> = kept_axes
                .iter()
                .map(|&a| in_sh.dims.get(a).copied().unwrap_or(1))
                .collect();
            let out_sh = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false; kept_dims.len()],
                layout: MemoryLayout::default_tpu_tiled(kept_dims.len()),
                dims: kept_dims,
            };
            let init = match opcode {
                TpuOpCode::ReduceMax => f64::NEG_INFINITY,
                TpuOpCode::ReduceMin => f64::INFINITY,
                _ => 0.0,
            };
            let mut out_vals = vec![init; out_sh.element_count()];
            for (src_idx, &v) in vals.iter().enumerate().take(in_sh.element_count()) {
                let src_coords = in_sh.linear_to_coords(src_idx);
                let dst_coords: Vec<usize> = kept_axes
                    .iter()
                    .map(|&a| src_coords.get(a).copied().unwrap_or(0))
                    .collect();
                let dst_idx = out_sh.coords_to_linear(&dst_coords).unwrap_or(0);
                for acc in out_vals.get_mut(dst_idx) {
                    *acc = match opcode {
                        TpuOpCode::ReduceMax => acc.max(v),
                        TpuOpCode::ReduceMin => acc.min(v),
                        _ => *acc + v,
                    };
                }
            }
            return elem_type.pack_from_f64_slice(&out_vals);
        }
        let scalar = match opcode {
            TpuOpCode::ReduceMax => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            TpuOpCode::ReduceMin => vals.iter().copied().fold(f64::INFINITY, f64::min),
            _ => vals.iter().sum(),
        };
        let out_len = attrs
            .output_shape
            .as_ref()
            .map_or(vals.len(), |s| s.element_count().max(1));
        elem_type.pack_from_f64_slice(&vec![scalar; out_len])
    }

    /// Evaluates a binary operation across `b0` and `b1`.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Binary [`TpuOpCode`].
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `b0` - First input HBM buffer.
    /// * `b1` - Second input HBM buffer.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(super) fn eval_binary_op(
        opcode: TpuOpCode,
        elem_type: ElementType,
        b0: &[u8],
        b1: &[u8],
    ) -> Vec<u8> {
        if opcode == TpuOpCode::XorBytes {
            return b0.iter().zip(b1.iter()).map(|(&x, &y)| x ^ y).collect();
        }
        let v0 = elem_type.unpack_to_f64_vec(b0);
        let v1 = elem_type.unpack_to_f64_vec(b1);
        if opcode == TpuOpCode::Concatenate {
            let mut joined = Vec::with_capacity(v0.len() + v1.len());
            joined.extend_from_slice(&v0);
            joined.extend_from_slice(&v1);
            return elem_type.pack_from_f64_slice(&joined);
        }
        if opcode == TpuOpCode::Gather {
            let gathered: Vec<f64> = v1
                .iter()
                .map(|&idx| {
                    let pos = (idx.max(0.0) as usize) % v0.len().max(1);
                    v0.get(pos).copied().unwrap_or(0.0)
                })
                .collect();
            return elem_type.pack_from_f64_slice(&gathered);
        }
        if opcode == TpuOpCode::DynamicSlice {
            let start = (v1.first().copied().unwrap_or(0.0).max(0.0) as usize) % v0.len().max(1);
            let rotated: Vec<f64> = (0..v0.len())
                .map(|i| {
                    v0.get((start + i) % v0.len().max(1))
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();
            return elem_type.pack_from_f64_slice(&rotated);
        }
        if opcode == TpuOpCode::Convolution {
            let conv: Vec<f64> = (0..v0.len())
                .map(|i| {
                    let mut sum = 0.0_f64;
                    for (k, &w) in v1.iter().enumerate() {
                        if let Some(&x) = v0.get(i + k) {
                            sum = x.mul_add(w, sum);
                        }
                    }
                    sum
                })
                .collect();
            return elem_type.pack_from_f64_slice(&conv);
        }
        if opcode == TpuOpCode::Pad {
            let pad_val = v1.first().copied().unwrap_or(0.0);
            let padded: Vec<f64> = v0.iter().map(|&x| x + pad_val).collect();
            return elem_type.pack_from_f64_slice(&padded);
        }
        if opcode == TpuOpCode::TriangularSolve {
            return elem_type.pack_from_f64_slice(&Self::eval_triangular_solve(&v0, &v1));
        }
        let out_vals: Vec<f64> = v0
            .iter()
            .zip(v1.iter())
            .map(|(&x, &y)| match opcode {
                TpuOpCode::AddF32 | TpuOpCode::AddS32 => x + y,
                TpuOpCode::SubF32 => x - y,
                TpuOpCode::DivF32 => x / y,
                TpuOpCode::Rem => x % y,
                TpuOpCode::Atan2 => x.atan2(y),
                TpuOpCode::Pow => x.powf(y),
                TpuOpCode::MaxF32 => x.max(y),
                TpuOpCode::MinF32 => x.min(y),
                TpuOpCode::And => f64::from((x as i64 & y as i64) as i32),
                TpuOpCode::Or => f64::from((x as i64 | y as i64) as i32),
                TpuOpCode::ShiftLeft => {
                    let shift = (y.max(0.0) as u32) & 31;
                    f64::from((x as i32).wrapping_shl(shift))
                }
                TpuOpCode::ShiftRightArithmetic => {
                    let shift = (y.max(0.0) as u32) & 31;
                    f64::from((x as i32).wrapping_shr(shift))
                }
                TpuOpCode::ShiftRightLogical => {
                    let shift = (y.max(0.0) as u32) & 31;
                    f64::from((x as i32 as u32).wrapping_shr(shift))
                }
                TpuOpCode::CompareEq => f64::from(u8::from((x - y).abs() < 1e-9)),
                TpuOpCode::CompareLt => f64::from(u8::from(x < y)),
                TpuOpCode::CompareGt => f64::from(u8::from(x > y)),
                _ => x * y,
            })
            .collect();
        elem_type.pack_from_f64_slice(&out_vals)
    }

    /// Evaluates a single [`crate::compiler::TpuInstructionBundle`] with its [`crate::compiler::InstructionAttrs`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Lowered [`crate::compiler::TpuInstructionBundle`].
    /// * `default_elem_type` - Default [`ElementType`] if not overridden by `bundle.attrs`.
    /// * `operands` - Slice of operand byte slices.
    #[allow(clippy::too_many_lines, clippy::or_fun_call)]
    pub(super) fn eval_bundle_with_attrs(
        bundle: &crate::compiler::TpuInstructionBundle,
        default_elem_type: ElementType,
        operands: &[&[u8]],
    ) -> Vec<u8> {
        let out_elem_type = bundle
            .attrs
            .output_shape
            .as_ref()
            .map_or(default_elem_type, |s| s.element_type);
        let in_elem_type = bundle
            .attrs
            .input_shapes
            .first()
            .map_or(out_elem_type, |s| s.element_type);
        let b0 = operands.first().copied().unwrap_or(&[]);

        match bundle.opcode {
            TpuOpCode::Constant => {
                if !bundle.attrs.constant_bytes.is_empty() {
                    return bundle.attrs.constant_bytes.clone();
                }
                return vec![0_u8; 16];
            }
            TpuOpCode::Convert => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                return out_elem_type.pack_from_f64_slice(&vals);
            }
            TpuOpCode::Transpose => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                let in_sh = bundle.attrs.input_shapes.first().cloned().unwrap_or(Shape {
                    element_type: in_elem_type,
                    dims: vec![2, 2],
                    dynamic_dims: vec![false; 2],
                    layout: MemoryLayout::default_tpu_tiled(2),
                });
                let perm = if bundle.attrs.permutation.len() == in_sh.rank() {
                    bundle.attrs.permutation.clone()
                } else {
                    (0..in_sh.rank()).rev().collect()
                };
                let out_sh = in_sh.transpose(&perm).unwrap_or(in_sh.clone());
                let count = out_sh.element_count().min(vals.len());
                let mut out_vals = Vec::with_capacity(count);
                for dst_idx in 0..count {
                    let dst_coords = out_sh.linear_to_coords(dst_idx);
                    let mut src_coords = vec![0_usize; in_sh.rank()];
                    for (src_axis, slot) in src_coords.iter_mut().enumerate() {
                        let dst_axis = perm.iter().position(|&p| p == src_axis).unwrap_or(src_axis);
                        *slot = dst_coords.get(dst_axis).copied().unwrap_or(0);
                    }
                    let src_idx = in_sh.coords_to_linear(&src_coords).unwrap_or(0);
                    out_vals.push(vals.get(src_idx).copied().unwrap_or(0.0));
                }
                return out_elem_type.pack_from_f64_slice(&out_vals);
            }
            TpuOpCode::BroadcastInDim => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                let out_sh = bundle.attrs.output_shape.clone().unwrap_or(Shape {
                    element_type: out_elem_type,
                    dims: vec![4],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                });
                let in_sh = bundle
                    .attrs
                    .input_shapes
                    .first()
                    .cloned()
                    .unwrap_or(out_sh.clone());
                let out_count = out_sh.element_count();
                let mut out_vals = Vec::with_capacity(out_count);
                for dst_idx in 0..out_count {
                    let dst_coords = out_sh.linear_to_coords(dst_idx);
                    let mut src_coords = vec![0_usize; in_sh.rank()];
                    for (src_axis, slot) in src_coords.iter_mut().enumerate() {
                        let dst_axis = bundle
                            .attrs
                            .broadcast_dims
                            .get(src_axis)
                            .copied()
                            .unwrap_or(src_axis);
                        let src_ext =
                            usize::try_from(in_sh.dims.get(src_axis).copied().unwrap_or(1))
                                .unwrap_or(1)
                                .max(1);
                        let dc = dst_coords.get(dst_axis).copied().unwrap_or(0);
                        *slot = dc % src_ext;
                    }
                    let src_idx = in_sh.coords_to_linear(&src_coords).unwrap_or(0);
                    out_vals.push(vals.get(src_idx).copied().unwrap_or(0.0));
                }
                return out_elem_type.pack_from_f64_slice(&out_vals);
            }
            TpuOpCode::Slice => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                let in_sh = bundle.attrs.input_shapes.first().cloned().unwrap_or(Shape {
                    element_type: in_elem_type,
                    dims: vec![4],
                    dynamic_dims: vec![false],
                    layout: MemoryLayout::default_tpu_tiled(1),
                });
                if !bundle.attrs.slice_starts.is_empty() {
                    let mut out_vals = Vec::new();
                    let in_count = in_sh.element_count().min(vals.len());
                    for idx in 0..in_count {
                        let coords = in_sh.linear_to_coords(idx);
                        let mut keep = true;
                        for (axis, &c) in coords.iter().enumerate() {
                            let st = bundle.attrs.slice_starts.get(axis).copied().unwrap_or(0);
                            let lim = bundle
                                .attrs
                                .slice_limits
                                .get(axis)
                                .copied()
                                .unwrap_or(usize::MAX);
                            let stride = bundle
                                .attrs
                                .slice_strides
                                .get(axis)
                                .copied()
                                .unwrap_or(1)
                                .max(1);
                            if c < st || c >= lim || (c - st) % stride != 0 {
                                keep = false;
                                break;
                            }
                        }
                        if keep {
                            out_vals.push(vals.get(idx).copied().unwrap_or(0.0));
                        }
                    }
                    return out_elem_type.pack_from_f64_slice(&out_vals);
                }
                return b0.to_vec();
            }
            TpuOpCode::DotGeneralF32 => {
                let v0 = in_elem_type.unpack_to_f64_vec(b0);
                let v1 = in_elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
                return Self::eval_dot_general(out_elem_type, &v0, &v1, &bundle.attrs);
            }
            TpuOpCode::Iota => {
                if let Some(out_sh) = &bundle.attrs.output_shape {
                    let axis = bundle.attrs.iota_dim.min(out_sh.rank().saturating_sub(1));
                    let count = out_sh.element_count();
                    let mut out_vals = Vec::with_capacity(count);
                    for idx in 0..count {
                        let coords = out_sh.linear_to_coords(idx);
                        let c =
                            u32::try_from(coords.get(axis).copied().unwrap_or(idx)).unwrap_or(0);
                        out_vals.push(f64::from(c));
                    }
                    return out_elem_type.pack_from_f64_slice(&out_vals);
                }
            }
            TpuOpCode::ReduceSumF32 | TpuOpCode::ReduceMax | TpuOpCode::ReduceMin => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                return Self::eval_reduce_with_attrs(
                    bundle.opcode,
                    out_elem_type,
                    &vals,
                    &bundle.attrs,
                );
            }
            TpuOpCode::Pad => {
                let v0 = in_elem_type.unpack_to_f64_vec(b0);
                let v1 = in_elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
                let pad_val = v1.first().copied().unwrap_or(0.0);
                return Self::eval_pad_with_attrs(out_elem_type, &v0, pad_val, &bundle.attrs);
            }
            TpuOpCode::Concatenate if bundle.attrs.concat_dim > 0 => {
                let v0 = in_elem_type.unpack_to_f64_vec(b0);
                let v1 = in_elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
                let mut interleaved = Vec::with_capacity(v0.len() + v1.len());
                for (&x, &y) in v0.iter().zip(v1.iter()) {
                    interleaved.push(x);
                    interleaved.push(y);
                }
                return out_elem_type.pack_from_f64_slice(&interleaved);
            }
            _ => {}
        }
        Self::eval_bundle(bundle.opcode, out_elem_type, operands)
    }

    /// Evaluates a single [`crate::compiler::TpuInstructionBundle`] against a register bank of byte vectors.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Instruction [`TpuOpCode`].
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `operands` - Slice of operand byte slices.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(super) fn eval_bundle(
        opcode: TpuOpCode,
        elem_type: ElementType,
        operands: &[&[u8]],
    ) -> Vec<u8> {
        if opcode == TpuOpCode::Select || opcode == TpuOpCode::Clamp {
            let c = elem_type.unpack_to_f64_vec(operands.first().copied().unwrap_or(&[]));
            let t = elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
            let f = elem_type.unpack_to_f64_vec(operands.get(2).copied().unwrap_or(&[]));
            let sel: Vec<f64> = c
                .iter()
                .zip(t.iter().zip(f.iter()))
                .map(|(&first, (&second, &third))| {
                    if opcode == TpuOpCode::Clamp {
                        second.clamp(first, third)
                    } else if first == 0.0 {
                        third
                    } else {
                        second
                    }
                })
                .collect();
            elem_type.pack_from_f64_slice(&sel)
        } else if opcode == TpuOpCode::Scatter || opcode == TpuOpCode::DynamicUpdateSlice {
            let mut dst = elem_type.unpack_to_f64_vec(operands.first().copied().unwrap_or(&[]));
            let idx_or_upd = elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
            let third = elem_type.unpack_to_f64_vec(operands.get(2).copied().unwrap_or(&[]));
            let len = dst.len().max(1);
            if opcode == TpuOpCode::Scatter {
                for (&idx_f, &upd_val) in idx_or_upd.iter().zip(third.iter()) {
                    let target_pos = (idx_f.max(0.0) as usize) % len;
                    for (pos, slot) in dst.iter_mut().enumerate() {
                        if pos == target_pos {
                            *slot = upd_val;
                        }
                    }
                }
            } else {
                let start = (third.first().copied().unwrap_or(0.0).max(0.0) as usize) % len;
                for (i, &upd_val) in idx_or_upd.iter().enumerate() {
                    let target_pos = (start + i) % len;
                    for (pos, slot) in dst.iter_mut().enumerate() {
                        if pos == target_pos {
                            *slot = upd_val;
                        }
                    }
                }
            }
            elem_type.pack_from_f64_slice(&dst)
        } else if opcode.is_binary() {
            Self::eval_binary_op(
                opcode,
                elem_type,
                operands.first().copied().unwrap_or(&[]),
                operands.get(1).copied().unwrap_or(&[]),
            )
        } else {
            Self::eval_unary_op(opcode, elem_type, operands.first().copied().unwrap_or(&[]))
        }
    }

    /// Executes a cross-device collective (`AllReduce`, `AllGather`, or `ReduceScatter`) across multiple
    /// device buffers on the simulated ICI torus mesh.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Collective [`TpuOpCode`] (`AllReduceSumF32`, `AllGather`, or `ReduceScatter`).
    /// * `elem_type` - Element data type.
    /// * `participants` - Slice of `(DeviceId, input_addr, output_addr)` tuples for each replica.
    /// * `element_count` - Number of elements per replica input buffer.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any participant device or buffer address is invalid.
    #[allow(for_loops_over_fallibles)]
    pub fn execute_collective(
        &self,
        opcode: TpuOpCode,
        elem_type: ElementType,
        participants: &[(DeviceId, DeviceAddress, DeviceAddress)],
        element_count: usize,
    ) -> LibTpuResult<()> {
        let mut guard = self.lock_devices()?;
        let mut replica_vecs: Vec<Vec<f64>> = Vec::with_capacity(participants.len());
        let byte_len = (element_count * elem_type.bit_size()).div_ceil(8);

        for &(dev_id, in_addr, out_addr) in participants {
            let dev = guard.get(&dev_id).ok_or(LibTpuError::DeviceNotFound {
                device_id: dev_id.0,
            })?;
            if !dev.allocations.contains_key(&out_addr) {
                return Err(LibTpuError::ExecutionFailed {
                    device_id: dev_id.0,
                    reason: format!("Collective output {out_addr} not found on device {dev_id}"),
                });
            }
            let buf =
                dev.allocations
                    .get(&in_addr)
                    .ok_or_else(|| LibTpuError::ExecutionFailed {
                        device_id: dev_id.0,
                        reason: format!("Collective input {in_addr} not found on device {dev_id}"),
                    })?;
            let slice = buf.get(..byte_len).unwrap_or(buf.as_slice());
            replica_vecs.push(elem_type.unpack_to_f64_vec(slice));
        }

        let mut reduced = vec![0.0_f64; element_count];
        for r_vec in &replica_vecs {
            for (dst, &val) in reduced.iter_mut().zip(r_vec.iter()) {
                *dst += val;
            }
        }

        let mut gathered = Vec::with_capacity(element_count * replica_vecs.len());
        for r_vec in &replica_vecs {
            gathered.extend_from_slice(r_vec);
        }

        let num_reps = replica_vecs.len().max(1);
        let chunk_len = (element_count / num_reps).max(1);
        for (rep_idx, &(dev_id, _, out_addr)) in participants.iter().enumerate() {
            let payload_f64: Vec<f64> = match opcode {
                TpuOpCode::AllGather => gathered.clone(),
                TpuOpCode::ReduceScatter => {
                    let start = (rep_idx * chunk_len).min(reduced.len());
                    let end = (start + chunk_len).min(reduced.len());
                    reduced.get(start..end).unwrap_or(&[]).to_vec()
                }
                TpuOpCode::CollectivePermute => {
                    let src_rep = (rep_idx + num_reps - 1) % num_reps;
                    replica_vecs.get(src_rep).cloned().unwrap_or_default()
                }
                TpuOpCode::AllToAll => {
                    let mut a2a = Vec::with_capacity(element_count);
                    for src_vec in &replica_vecs {
                        let start = (rep_idx * chunk_len).min(src_vec.len());
                        let end = (start + chunk_len).min(src_vec.len());
                        a2a.extend_from_slice(src_vec.get(start..end).unwrap_or(&[]));
                    }
                    a2a
                }
                _ => reduced.clone(),
            };
            let packed = elem_type.pack_from_f64_slice(&payload_f64);
            for dev in guard.get_mut(&dev_id) {
                for out_buf in dev.allocations.get_mut(&out_addr) {
                    for (dst_b, &src_b) in out_buf.iter_mut().zip(packed.iter()) {
                        *dst_b = src_b;
                    }
                }
            }
        }
        drop(guard);
        Ok(())
    }
}
