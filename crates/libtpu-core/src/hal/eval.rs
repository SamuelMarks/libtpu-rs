//! Tensor operation evaluation routines (`MXU`, `VALU`, `EUP`, and ICI collectives).
//!
//! Implements instruction bundle evaluation and collective execution for [`SimulatedTpuHal`].

use super::SimulatedTpuHal;
use crate::compiler::TpuOpCode;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{DeviceAddress, DeviceId, ElementType, MemoryLayout, Shape};

impl SimulatedTpuHal {
    /// Evaluates a general `N x N` lower-triangular Cholesky decomposition (`A = L * L^T`)
    /// or elementwise square root for non-square buffers.
    ///
    /// # Arguments
    ///
    /// * `vals` - Unpacked input matrix elements.
    pub(super) fn eval_cholesky(vals: &[f64]) -> Vec<f64> {
        let len = vals.len();
        let mut n = 2_usize;
        while n.saturating_mul(n) < len {
            n += 1;
        }
        if n.saturating_mul(n) == len {
            let mut l_mat = vec![0.0_f64; len];
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = 0.0_f64;
                    for k in 0..j {
                        let lik = l_mat.get(i * n + k).copied().unwrap_or(0.0);
                        let ljk = l_mat.get(j * n + k).copied().unwrap_or(0.0);
                        sum = lik.mul_add(ljk, sum);
                    }
                    let a_ij = vals.get(i * n + j).copied().unwrap_or(0.0);
                    let val = if i == j {
                        (a_ij - sum).max(1e-12).sqrt()
                    } else {
                        let l_jj = l_mat.get(j * n + j).copied().unwrap_or(1.0).max(1e-12);
                        (a_ij - sum) / l_jj
                    };
                    l_mat
                        .get_mut(i * n + j)
                        .into_iter()
                        .for_each(|slot| *slot = val);
                }
            }
            l_mat
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

    /// Evaluates full complex 1D/ND `FFT`, `IFFT`, `RFFT`, or `IRFFT` using `num_complex::Complex64`.
    ///
    /// # Arguments
    ///
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `vals` - Unpacked input scalars (interleaved `[re, im]` for `C64`/`C128` or real scalars for `F32`/`F64`).
    /// * `fft_type` - Mode string (`"FFT"`, `"IFFT"`, `"RFFT"`, or `"IRFFT"`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn eval_fft_with_mode(elem_type: ElementType, vals: &[f64], fft_type: &str) -> Vec<u8> {
        let is_inverse =
            fft_type.eq_ignore_ascii_case("IFFT") || fft_type.eq_ignore_ascii_case("IRFFT");
        if matches!(elem_type, ElementType::C64 | ElementType::C128) {
            let complex_in: Vec<num_complex::Complex64> = vals
                .chunks_exact(2)
                .map(|c| {
                    num_complex::Complex64::new(
                        c.first().copied().unwrap_or(0.0),
                        c.get(1).copied().unwrap_or(0.0),
                    )
                })
                .collect();
            let n = complex_in.len().max(1);
            let n_f = n as f64;
            let sign = if is_inverse { 1.0 } else { -1.0 };
            let scale = if is_inverse { 1.0 / n_f } else { 1.0 };
            let mut out_interleaved = Vec::with_capacity(n * 2);
            for k in 0..n {
                let mut sum = num_complex::Complex64::new(0.0, 0.0);
                for (t, &x_t) in complex_in.iter().enumerate() {
                    let theta = sign * 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / n_f;
                    let twiddle = num_complex::Complex64::new(theta.cos(), theta.sin());
                    sum += x_t * twiddle;
                }
                sum *= scale;
                out_interleaved.push(sum.re);
                out_interleaved.push(sum.im);
            }
            return elem_type.pack_from_f64_slice(&out_interleaved);
        }
        let mut real_dft = Self::eval_fft(vals);
        if is_inverse {
            let scale = 1.0 / (vals.len().max(1) as f64);
            for v in &mut real_dft {
                *v *= scale;
            }
        }
        elem_type.pack_from_f64_slice(&real_dft)
    }

    /// Official XLA 20-round **Threefry2x32** counter-based bijection (`threefry2x32`).
    ///
    /// Uses the canonical Skein/Threefry rotation constants `[[13, 15, 26, 6], [17, 29, 16, 24]]`
    /// and key-schedule parity constant `0x1BD1_1BDA`.
    ///
    /// # Arguments
    ///
    /// * `key` - 2x32-bit key words `[k0, k1]`.
    /// * `counter` - 2x32-bit counter words `[c0, c1]`.
    #[must_use]
    #[allow(clippy::indexing_slicing, clippy::manual_is_multiple_of)]
    pub const fn threefry2x32(key: [u32; 2], counter: [u32; 2]) -> [u32; 2] {
        const ROTATIONS: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
        let ks2 = key[0] ^ key[1] ^ 0x1BD1_1BDA_u32;
        let ks = [key[0], key[1], ks2];
        let mut x0 = counter[0].wrapping_add(ks[0]);
        let mut x1 = counter[1].wrapping_add(ks[1]);

        let mut round = 0_usize;
        while round < 20 {
            let rot = ROTATIONS[(round / 4) & 1][round & 3];
            x0 = x0.wrapping_add(x1);
            x1 = x1.rotate_left(rot) ^ x0;
            if (round + 1) % 4 == 0 {
                let inj = (round + 1) / 4;
                x0 = x0.wrapping_add(ks[inj % 3]);
                #[allow(clippy::cast_possible_truncation)]
                let inj_u32 = inj as u32;
                x1 = x1.wrapping_add(ks[(inj + 1) % 3]).wrapping_add(inj_u32);
            }
            round += 1;
        }
        [x0, x1]
    }

    /// Official XLA 10-round **Philox4x32-10** counter-based PRNG bijection (`philox4x32_10`).
    ///
    /// # Arguments
    ///
    /// * `key` - 2x32-bit key words `[k0, k1]`.
    /// * `counter` - 4x32-bit counter words `[c0, c1, c2, c3]`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn philox4x32_10(key: [u32; 2], mut counter: [u32; 4]) -> [u32; 4] {
        const M0: u64 = 0xD251_1F53;
        const M1: u64 = 0xCD9E_8D57;
        const W0: u32 = 0x9E37_79B9;
        const W1: u32 = 0xBB67_AE85;
        let mut k0 = key[0];
        let mut k1 = key[1];
        let mut round = 0_usize;
        while round < 10 {
            let p0 = (counter[0] as u64).wrapping_mul(M0);
            let p1 = (counter[2] as u64).wrapping_mul(M1);
            let hi0 = (p0 >> 32) as u32;
            let lo0 = p0 as u32;
            let hi1 = (p1 >> 32) as u32;
            let lo1 = p1 as u32;
            counter = [hi1 ^ counter[1] ^ k0, lo1, hi0 ^ counter[3] ^ k1, lo0];
            k0 = k0.wrapping_add(W0);
            k1 = k1.wrapping_add(W1);
            round += 1;
        }
        counter
    }

    /// Evaluates `BatchNormInference`, `BatchNormTraining`, or `BatchNormGrad` across `[ operand, scale, offset ]`.
    ///
    /// # Arguments
    ///
    /// * `opcode` - `BatchNorm` [`TpuOpCode`].
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `operands` - Slice of 1 to 5 operand byte buffers.
    #[must_use]
    pub fn eval_batch_norm(
        opcode: TpuOpCode,
        elem_type: ElementType,
        operands: &[&[u8]],
    ) -> Vec<u8> {
        let x = elem_type.unpack_to_f64_vec(operands.first().copied().unwrap_or(&[]));
        let scale = elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
        let offset_or_grad = elem_type.unpack_to_f64_vec(operands.get(2).copied().unwrap_or(&[]));
        #[allow(clippy::cast_precision_loss)]
        let n_f = (x.len().max(1)) as f64;
        let mean: f64 = x.iter().sum::<f64>() / n_f;
        let var: f64 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n_f;
        let inv_std = 1.0 / (var + 1e-5).sqrt();

        let out_vals: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let gamma = scale.get(i % scale.len().max(1)).copied().unwrap_or(1.0);
                let third = offset_or_grad
                    .get(i % offset_or_grad.len().max(1))
                    .copied()
                    .unwrap_or(0.0);
                if opcode == TpuOpCode::BatchNormGrad {
                    third * gamma * inv_std
                } else {
                    ((xi - mean) * inv_std).mul_add(gamma, third)
                }
            })
            .collect();
        elem_type.pack_from_f64_slice(&out_vals)
    }

    /// Evaluates `SelectAndScatter` scattering `source` gradients into the argmax window positions of `operand`.
    ///
    /// # Arguments
    ///
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `operands` - `[operand, source, init_value]` byte slices.
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn eval_select_and_scatter(elem_type: ElementType, operands: &[&[u8]]) -> Vec<u8> {
        let operand = elem_type.unpack_to_f64_vec(operands.first().copied().unwrap_or(&[]));
        let source = elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
        let init_val = elem_type
            .unpack_to_f64_vec(operands.get(2).copied().unwrap_or(&[]))
            .first()
            .copied()
            .unwrap_or(0.0);
        let mut out = vec![init_val; operand.len()];
        let win_size = (operand.len() / source.len().max(1)).max(2);
        for (s_idx, &s_val) in source.iter().enumerate() {
            let start = (s_idx * win_size).min(operand.len().saturating_sub(1));
            let end = (start + win_size).min(operand.len());
            let best_pos = operand
                .iter()
                .enumerate()
                .take(end)
                .skip(start)
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map_or(start, |(idx, _)| idx);
            out[best_pos] += s_val;
        }
        elem_type.pack_from_f64_slice(&out)
    }

    /// Evaluates a general `N x N` lower-triangular linear system (`L * X = B`) via forward substitution,
    /// or diagonal division for non-square buffers.
    ///
    /// # Arguments
    ///
    /// * `v0` - Unpacked lower-triangular matrix `L`.
    /// * `v1` - Unpacked right-hand side matrix `B`.
    pub(super) fn eval_triangular_solve(v0: &[f64], v1: &[f64]) -> Vec<f64> {
        let len = v0.len().min(v1.len());
        let mut n = 2_usize;
        while n.saturating_mul(n) < len {
            n += 1;
        }
        if n.saturating_mul(n) == len {
            let mut x_mat = vec![0.0_f64; len];
            for col in 0..n {
                for row in 0..n {
                    let mut sum = v1.get(row * n + col).copied().unwrap_or(0.0);
                    for k in 0..row {
                        let l_rk = v0.get(row * n + k).copied().unwrap_or(0.0);
                        let x_kc = x_mat.get(k * n + col).copied().unwrap_or(0.0);
                        sum = l_rk.mul_add(-x_kc, sum);
                    }
                    let diag = v0.get(row * n + row).copied().unwrap_or(1.0);
                    let denom = if diag.abs() < 1e-12 { 1.0 } else { diag };
                    x_mat
                        .get_mut(row * n + col)
                        .into_iter()
                        .for_each(|slot| *slot = sum / denom);
                }
            }
            return x_mat;
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
                | TpuOpCode::BitcastConvert
                | TpuOpCode::OptimizationBarrier
                | TpuOpCode::Tuple
                | TpuOpCode::GetTupleElement
                | TpuOpCode::Infeed
                | TpuOpCode::Outfeed
                | TpuOpCode::Send
                | TpuOpCode::Recv
                | TpuOpCode::While
                | TpuOpCode::Conditional
                | TpuOpCode::Call
        ) {
            return input_bytes.to_vec();
        }
        let vals = elem_type.unpack_to_f64_vec(input_bytes);
        let mapped: Vec<f64> = match opcode {
            TpuOpCode::ReplicaId | TpuOpCode::PartitionId | TpuOpCode::AfterAll => {
                vec![0.0; vals.len().max(1)]
            }
            TpuOpCode::Real => vals.iter().step_by(2).copied().collect(),
            TpuOpCode::Imag => vals.iter().skip(1).step_by(2).copied().collect(),
            TpuOpCode::NegF32 => vals.iter().map(|&v| -v).collect(),
            TpuOpCode::ExpF32 => vals.iter().map(|&v| v.exp()).collect(),
            TpuOpCode::Expm1 => vals.iter().map(|&v| v.exp_m1()).collect(),
            TpuOpCode::TanhF32 => vals.iter().map(|&v| v.tanh()).collect(),
            TpuOpCode::Erf => vals
                .iter()
                .map(|&v| {
                    let sign = v.signum();
                    let x = v.abs();
                    let t = 1.0 / 0.327_591_1_f64.mul_add(x, 1.0);
                    let p4 = t.mul_add(1.061_405_429, -1.453_152_027);
                    let p3 = t.mul_add(p4, 1.421_413_741);
                    let p2 = t.mul_add(p3, -0.284_496_736);
                    let poly = t * t.mul_add(p2, 0.254_829_592);
                    sign * poly.mul_add(-(-x * x).exp(), 1.0)
                })
                .collect(),
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
                let seed = vals.first().copied().unwrap_or(1.0) as u32;
                let key = [seed, seed.wrapping_mul(0x9E37_79B9)];
                (0..vals.len())
                    .map(|i| {
                        let pair = Self::threefry2x32(key, [i as u32, 0]);
                        f64::from(pair[0] & 0xFFFF)
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
                    l_coords
                        .get_mut(lb)
                        .into_iter()
                        .for_each(|slot| *slot = b_val);
                    r_coords
                        .get_mut(rb)
                        .into_iter()
                        .for_each(|slot| *slot = b_val);
                }
                for (fi, &lf) in l_free.iter().enumerate() {
                    let f_val = dst_coords.get(l_batch.len() + fi).copied().unwrap_or(0);
                    l_coords
                        .get_mut(lf)
                        .into_iter()
                        .for_each(|slot| *slot = f_val);
                }
                for (fi, &rf) in r_free.iter().enumerate() {
                    let f_val = dst_coords
                        .get(l_batch.len() + l_free.len() + fi)
                        .copied()
                        .unwrap_or(0);
                    r_coords
                        .get_mut(rf)
                        .into_iter()
                        .for_each(|slot| *slot = f_val);
                }
                for (ci, (&lc, &rc)) in l_cont.iter().zip(r_cont.iter()).enumerate() {
                    let c_val = c_coords.get(ci).copied().unwrap_or(0);
                    l_coords
                        .get_mut(lc)
                        .into_iter()
                        .for_each(|slot| *slot = c_val);
                    r_coords
                        .get_mut(rc)
                        .into_iter()
                        .for_each(|slot| *slot = c_val);
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

    /// Evaluates complex matrix multiplication on interleaved `[re, im]` `f64` slices.
    ///
    /// # Arguments
    ///
    /// * `elem_type` - Complex [`ElementType`] (`C64` or `C128`).
    /// * `v0` - Unpacked LHS interleaved complex scalars.
    /// * `v1` - Unpacked RHS interleaved complex scalars.
    fn eval_complex_dot_general(elem_type: ElementType, v0: &[f64], v1: &[f64]) -> Vec<u8> {
        let c0: Vec<num_complex::Complex64> = v0
            .chunks_exact(2)
            .map(|p| {
                num_complex::Complex64::new(
                    p.first().copied().unwrap_or(0.0),
                    p.get(1).copied().unwrap_or(0.0),
                )
            })
            .collect();
        let c1: Vec<num_complex::Complex64> = v1
            .chunks_exact(2)
            .map(|p| {
                num_complex::Complex64::new(
                    p.first().copied().unwrap_or(0.0),
                    p.get(1).copied().unwrap_or(0.0),
                )
            })
            .collect();
        let elem_len = c0.len().min(c1.len());
        let mut dim = 1_usize;
        while dim.saturating_mul(dim) < elem_len {
            dim += 1;
        }
        let (rows, inner, cols) = if dim.saturating_mul(dim) == elem_len {
            (dim, dim, dim)
        } else {
            (elem_len, 1, 1)
        };
        let mut out_complex = Vec::with_capacity(rows * cols * 2);
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = num_complex::Complex64::new(0.0, 0.0);
                for idx in 0..inner {
                    let lv = c0
                        .get(r * inner + idx)
                        .copied()
                        .unwrap_or(num_complex::Complex64::new(0.0, 0.0));
                    let rv = c1
                        .get(idx * cols + c)
                        .copied()
                        .unwrap_or(num_complex::Complex64::new(0.0, 0.0));
                    acc += lv * rv;
                }
                out_complex.push(acc.re);
                out_complex.push(acc.im);
            }
        }
        elem_type.pack_from_f64_slice(&out_complex)
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

        if matches!(elem_type, ElementType::C64 | ElementType::C128) {
            return Self::eval_complex_dot_general(elem_type, v0, v1);
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
                out_vals.get_mut(dst_idx).into_iter().for_each(|acc| {
                    *acc = match opcode {
                        TpuOpCode::ReduceMax => acc.max(v),
                        TpuOpCode::ReduceMin => acc.min(v),
                        _ => *acc + v,
                    };
                });
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

    /// Evaluates complex multiplication or division on interleaved `[re, im]` `f64` slices.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Binary [`TpuOpCode`] (`MulF32` or `DivF32`).
    /// * `elem_type` - Complex [`ElementType`] (`C64` or `C128`).
    /// * `v0` - Unpacked LHS interleaved complex scalars.
    /// * `v1` - Unpacked RHS interleaved complex scalars.
    fn eval_complex_binary_op(
        opcode: TpuOpCode,
        elem_type: ElementType,
        v0: &[f64],
        v1: &[f64],
    ) -> Vec<u8> {
        let mut out_complex = Vec::with_capacity(v0.len().min(v1.len()));
        for (c0, c1) in v0.chunks_exact(2).zip(v1.chunks_exact(2)) {
            let a = num_complex::Complex64::new(
                c0.first().copied().unwrap_or(0.0),
                c0.get(1).copied().unwrap_or(0.0),
            );
            let b = num_complex::Complex64::new(
                c1.first().copied().unwrap_or(0.0),
                c1.get(1).copied().unwrap_or(0.0),
            );
            let res = if opcode == TpuOpCode::DivF32 {
                let denom = b.norm_sqr().max(1e-24);
                num_complex::Complex64::new(
                    a.im.mul_add(b.im, a.re * b.re) / denom,
                    a.re.mul_add(-b.im, a.im * b.re) / denom,
                )
            } else {
                a * b
            };
            out_complex.push(res.re);
            out_complex.push(res.im);
        }
        elem_type.pack_from_f64_slice(&out_complex)
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
        if matches!(elem_type, ElementType::C64 | ElementType::C128)
            && matches!(opcode, TpuOpCode::MulF32 | TpuOpCode::DivF32)
        {
            return Self::eval_complex_binary_op(opcode, elem_type, &v0, &v1);
        }
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
        if opcode == TpuOpCode::Complex {
            let mut interleaved = Vec::with_capacity(v0.len() * 2);
            for (&re, &im) in v0.iter().zip(v1.iter()) {
                interleaved.push(re);
                interleaved.push(im);
            }
            return elem_type.pack_from_f64_slice(&interleaved);
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
            TpuOpCode::Convolution => {
                let v0 = in_elem_type.unpack_to_f64_vec(b0);
                let v1 = in_elem_type.unpack_to_f64_vec(operands.get(1).copied().unwrap_or(&[]));
                let stride = bundle
                    .attrs
                    .window_strides
                    .first()
                    .copied()
                    .unwrap_or(1)
                    .max(1);
                let r_dil = bundle
                    .attrs
                    .rhs_dilation
                    .first()
                    .copied()
                    .unwrap_or(1)
                    .max(1);
                let out_len = v0.len().div_ceil(stride).max(1);
                let conv: Vec<f64> = (0..out_len)
                    .map(|o_idx| {
                        let base = o_idx * stride;
                        let mut sum = 0.0_f64;
                        for (k, &w) in v1.iter().enumerate() {
                            let x = v0.get(base + k * r_dil).copied().unwrap_or(0.0);
                            sum = x.mul_add(w, sum);
                        }
                        sum
                    })
                    .collect();
                return out_elem_type.pack_from_f64_slice(&conv);
            }
            TpuOpCode::Fft => {
                let vals = in_elem_type.unpack_to_f64_vec(b0);
                return Self::eval_fft_with_mode(
                    out_elem_type,
                    &vals,
                    &bundle.attrs.custom_call_target,
                );
            }
            TpuOpCode::BatchNormInference
            | TpuOpCode::BatchNormTraining
            | TpuOpCode::BatchNormGrad => {
                return Self::eval_batch_norm(bundle.opcode, out_elem_type, operands);
            }
            TpuOpCode::SelectAndScatter => {
                return Self::eval_select_and_scatter(out_elem_type, operands);
            }
            TpuOpCode::While => {
                let mut cur = b0.to_vec();
                for sub_prog in &bundle.attrs.sub_computations {
                    for sub_b in sub_prog {
                        cur = Self::eval_bundle_with_attrs(sub_b, out_elem_type, &[&cur]);
                    }
                }
                return cur;
            }
            TpuOpCode::Conditional => {
                let pred = in_elem_type
                    .unpack_to_f64_vec(b0)
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let branch_idx = usize::from(pred == 0.0)
                    .min(bundle.attrs.sub_computations.len().saturating_sub(1));
                let mut cur = operands.get(1).copied().unwrap_or(b0).to_vec();
                for sb in bundle
                    .attrs
                    .sub_computations
                    .get(branch_idx)
                    .iter()
                    .flat_map(|p| p.iter())
                {
                    cur = Self::eval_bundle_with_attrs(sb, out_elem_type, &[&cur]);
                }
                return cur;
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
            guard.get_mut(&dev_id).into_iter().for_each(|dev| {
                dev.allocations
                    .get_mut(&out_addr)
                    .into_iter()
                    .for_each(|out_buf| {
                        for (dst_b, &src_b) in out_buf.iter_mut().zip(packed.iter()) {
                            *dst_b = src_b;
                        }
                    });
            });
        }
        drop(guard);
        Ok(())
    }
}
