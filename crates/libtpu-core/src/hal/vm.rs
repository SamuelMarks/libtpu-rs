//! Architectural `TensorCore` VLIW execution state machine ([`TpuCoreVmState`] / [`DeepSeaVm`]).

use super::SimulatedTpuHal;
use crate::compiler::CompiledProgram;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::ElementType;
use std::collections::BTreeMap;

/// Architectural `TensorCore` VLIW execution state machine (`TpuCoreVmState`).
#[derive(Debug, Clone)]
pub struct TpuCoreVmState {
    /// 32 64-bit scalar general-purpose registers (`SREG[0..32]`).
    pub sreg: [u64; 32],
    /// 16 scalar predicate registers (`PREG[0..16]`).
    pub preg: [bool; 16],
    /// 64 vector register sublane words (`VREG[0..64]`).
    pub vreg_words: [u32; 64],
    /// Full architectural vector register tile bank (`VREG[0..255]`) storing physical tile payloads.
    pub vreg_tiles: BTreeMap<u8, Vec<u8>>,
    /// Addressable on-chip `VMEM` scratchpad regions indexed by linear-scan `assigned_vmem_offset`.
    pub vmem_allocations: BTreeMap<u64, Vec<u8>>,
    /// Staged systolic `MXU` right-hand-side weight tile elements.
    pub mxu_weights: Vec<f64>,
    /// 8 sublane predicate flags (`VMASK[0..8]`).
    pub vmask: [bool; 8],
    /// Systolic matrix unit (`MXU`) accumulator tile summary (`[f32; 16]`).
    pub mxu_accum: [f32; 16],
    /// 256 architectural synchronization flag counters (`SYNC_FLAGS[0..256]`).
    pub sync_flags: [u32; 256],
    /// On-chip Scalar Memory (`SMEM`) scratchpad word bank (`[u64; 64]`).
    pub smem_words: [u64; 64],
    /// On-chip Vector Memory (`VMEM`) scratchpad word bank (`[u32; 256]`).
    pub vmem_words: [u32; 256],
    /// Current VLIW bundle program counter (`PC`).
    pub pc: u32,
    /// Total VLIW instruction bundles executed on this core.
    pub cycles_executed: u64,
}

impl Default for TpuCoreVmState {
    fn default() -> Self {
        Self::new()
    }
}

impl TpuCoreVmState {
    /// Creates a zero-initialized [`TpuCoreVmState`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sreg: [0; 32],
            preg: [true; 16],
            vreg_words: [0; 64],
            vreg_tiles: BTreeMap::new(),
            vmem_allocations: BTreeMap::new(),
            mxu_weights: Vec::new(),
            vmask: [true; 8],
            mxu_accum: [0.0; 16],
            sync_flags: [0; 256],
            smem_words: [0; 64],
            vmem_words: [0; 256],
            pc: 0,
            cycles_executed: 0,
        }
    }

    /// Waits for `SYNC_FLAGS[sync_flag_id]` to reach at least `expected_value`, or returns
    /// [`LibTpuError::SyncFlagTimeout`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - Logical device identifier for diagnostics.
    /// * `sync_flag_id` - Synchronization flag register index (`0..256`).
    /// * `expected_value` - Minimum threshold required for barrier completion.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SyncFlagTimeout`] if the current counter is below `expected_value`.
    #[allow(clippy::indexing_slicing)]
    pub const fn wait_sync_flag(
        &self,
        device_id: u32,
        sync_flag_id: u32,
        expected_value: u32,
    ) -> LibTpuResult<()> {
        let actual_value = self.sync_flags[(sync_flag_id as usize) & 255];
        if actual_value < expected_value {
            return Err(LibTpuError::SyncFlagTimeout {
                device_id,
                sync_flag_id,
                expected_value,
                actual_value,
            });
        }
        Ok(())
    }

    /// Inspects the architectural 2D `[8, 128]` (`8` sublanes $\times$ `128` lanes) `u32` register tile
    /// for vector register `vreg_index` (`VREG[0..32]`), unpacking any populated bytes from `self.vreg_tiles`
    /// or broadcasting `self.vreg_words[vreg_index & 63]` across sublane `0`.
    ///
    /// # Arguments
    ///
    /// * `vreg_index` - Architectural vector register index (`0..32`).
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn inspect_vreg_2d_lanes(&self, vreg_index: u8) -> [[u32; 128]; 8] {
        let mut grid = [[0_u32; 128]; 8];
        if let Some(tile_bytes) = self.vreg_tiles.get(&vreg_index) {
            for (word_idx, chunk) in tile_bytes.chunks_exact(4).take(8 * 128).enumerate() {
                let mut b4 = [0_u8; 4];
                b4.copy_from_slice(chunk);
                let sublane = word_idx / 128;
                let lane = word_idx % 128;
                grid[sublane][lane] = u32::from_le_bytes(b4);
            }
        } else {
            let fallback = self.vreg_words[usize::from(vreg_index) & 63];
            grid[0][0] = fallback;
        }
        grid
    }

    /// Inspects the 16-element summary of matrix accumulator register `GMR[gmr_index]` (`0..4`).
    ///
    /// # Arguments
    ///
    /// * `gmr_index` - Matrix accumulator tile index (`0..4`).
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub const fn inspect_gmr_tile(&self, gmr_index: usize) -> [f32; 4] {
        let base = (gmr_index & 3) * 4;
        [
            self.mxu_accum[base],
            self.mxu_accum[base + 1],
            self.mxu_accum[base + 2],
            self.mxu_accum[base + 3],
        ]
    }

    /// Steps the `TensorCore` VLIW sequencer through a slice of 64-byte [`crate::compiler::TensorCoreVliwBundle`]s,
    /// executing `S0` (Scalar ALU + SMEM + conditional branching `opcode == 6`), `V0` (Vector ALU + VMEM),
    /// `M0` (`MXU` systolic matrix accumulator), `X0` (`EUP` transcendental unit), and `D0` (`DMA`/`SyncFlag` engine)
    /// slots in lockstep.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of 64-byte architectural VLIW instruction bundles.
    #[allow(clippy::indexing_slicing)]
    pub fn execute_vliw_stream(&mut self, bundles: &[crate::compiler::TensorCoreVliwBundle]) {
        for bundle in bundles {
            self.step_vliw_bundle(bundle);
            let preg_idx = usize::from(bundle.scalar_slot.dst_sreg) & 15;
            if bundle.scalar_slot.opcode == 6 && self.preg[preg_idx] {
                self.pc = bundle.scalar_slot.imm32;
            }
        }
    }

    /// Steps a single 64-byte architectural [`crate::compiler::TensorCoreVliwBundle`] across
    /// `S0`, `V0`, `X0`, `M0`, and `D0` functional units.
    ///
    /// # Arguments
    ///
    /// * `bundle` - 64-byte architectural VLIW instruction bundle to execute.
    #[allow(clippy::indexing_slicing)]
    pub fn step_vliw_bundle(&mut self, bundle: &crate::compiler::TensorCoreVliwBundle) {
        self.pc = bundle.header.pc_offset;
        // S0: Scalar ALU & SMEM scratchpad update
        let s_dst = usize::from(bundle.scalar_slot.dst_sreg) & 31;
        let s_src0 = self.sreg[usize::from(bundle.scalar_slot.src0_sreg) & 31];
        let s_src1 = self.sreg[usize::from(bundle.scalar_slot.src1_sreg) & 31];
        let s_imm = u64::from(bundle.scalar_slot.imm32);
        let s_val = match bundle.scalar_slot.opcode {
            2 => s_src0.wrapping_sub(s_src1),
            3 => s_src0 & s_src1,
            4 => s_src0 | s_src1,
            5 => s_src0 ^ s_src1,
            _ => s_src0.wrapping_add(s_imm),
        };
        self.sreg[s_dst] = s_val;
        self.smem_words[s_dst & 63] = s_val;
        self.preg[s_dst & 15] = s_val != 0;

        // V0: Vector ALU & VMEM scratchpad update
        let v_dst = usize::try_from(bundle.vector_slot.dst_vreg).unwrap_or(0) & 63;
        let v_src0 =
            self.vreg_words[usize::try_from(bundle.vector_slot.src0_vreg).unwrap_or(0) & 63];
        let v_src1 =
            self.vreg_words[usize::try_from(bundle.vector_slot.src1_vreg).unwrap_or(0) & 63];
        let v_op_decoded = u8::try_from(bundle.vector_slot.opcode)
            .ok()
            .and_then(|b| crate::compiler::TpuOpCode::try_from(b).ok());
        let v_res = match v_op_decoded {
            Some(crate::compiler::TpuOpCode::AddF32) => {
                (f32::from_bits(v_src0) + f32::from_bits(v_src1)).to_bits()
            }
            Some(crate::compiler::TpuOpCode::SubF32) => {
                (f32::from_bits(v_src0) - f32::from_bits(v_src1)).to_bits()
            }
            Some(crate::compiler::TpuOpCode::MulF32) => {
                (f32::from_bits(v_src0) * f32::from_bits(v_src1)).to_bits()
            }
            Some(crate::compiler::TpuOpCode::DivF32) => {
                (f32::from_bits(v_src0) / f32::from_bits(v_src1)).to_bits()
            }
            Some(crate::compiler::TpuOpCode::MaxF32) => {
                f32::from_bits(v_src0).max(f32::from_bits(v_src1)).to_bits()
            }
            Some(crate::compiler::TpuOpCode::MinF32) => {
                f32::from_bits(v_src0).min(f32::from_bits(v_src1)).to_bits()
            }
            Some(crate::compiler::TpuOpCode::NegF32) => (-f32::from_bits(v_src0)).to_bits(),
            Some(crate::compiler::TpuOpCode::Abs) => f32::from_bits(v_src0).abs().to_bits(),
            _ => v_src0
                .wrapping_add(v_src1)
                .wrapping_add(u32::from(bundle.vector_slot.opcode)),
        };
        self.vreg_words[v_dst] = v_res;
        self.vmem_words[v_dst & 255] = v_res;
        self.vmask[v_dst & 7] = (bundle.vector_slot.lane_mask & 1) != 0;
        self.step_v0_tile(&bundle.vector_slot, v_op_decoded);

        // X0: Extended Unary / Transcendental Pipeline (EUP)
        if bundle.xunary_slot.opcode != 0 {
            let x_dst = usize::from(bundle.xunary_slot.dst_vreg) & 63;
            let x_src_bits = self.vreg_words[usize::from(bundle.xunary_slot.src_vreg) & 63];
            let x_src = f32::from_bits(x_src_bits);
            let x_op_decoded = crate::compiler::TpuOpCode::try_from(bundle.xunary_slot.opcode).ok();
            let x_res = match x_op_decoded {
                Some(crate::compiler::TpuOpCode::ExpF32) => x_src.exp().to_bits(),
                Some(crate::compiler::TpuOpCode::TanhF32) => x_src.tanh().to_bits(),
                Some(crate::compiler::TpuOpCode::Log) => x_src.ln().to_bits(),
                Some(crate::compiler::TpuOpCode::Sqrt) => x_src.sqrt().to_bits(),
                Some(crate::compiler::TpuOpCode::Rsqrt) => (1.0 / x_src.sqrt()).to_bits(),
                Some(crate::compiler::TpuOpCode::Sine) => x_src.sin().to_bits(),
                Some(crate::compiler::TpuOpCode::Cosine) => x_src.cos().to_bits(),
                _ => x_src_bits.wrapping_add(u32::from(bundle.xunary_slot.opcode)),
            };
            self.vreg_words[x_dst] = x_res;
            self.step_x0_tile(bundle.xunary_slot, x_op_decoded);
        }

        // M0: Systolic Matrix Multiply Unit (MXU)
        if bundle.matrix_slot.opcode != 0 {
            let m_idx = usize::from(bundle.matrix_slot.acc_vreg) & 15;
            let k_tiles = f32::from(bundle.matrix_slot.k_tiles.max(1));
            self.mxu_accum[m_idx] += k_tiles;
            self.step_m0_tile(&bundle.matrix_slot);
        }

        // D0: Async DMA & SyncFlag completion barrier
        if bundle.dma_slot.dma_opcode != 0 {
            let vmem_off = u64::from(bundle.dma_slot.src_space) * 32;
            if let Some(tile) = self.vmem_allocations.get(&vmem_off).cloned() {
                #[allow(clippy::cast_possible_truncation)]
                let dst_reg = (bundle.vector_slot.dst_vreg & 255) as u8;
                self.vreg_tiles.insert(dst_reg, tile);
            }
        }
        let sf_idx = usize::try_from(bundle.dma_slot.sync_flag_id).unwrap_or(0) & 255;
        self.sync_flags[sf_idx] = self.sync_flags[sf_idx].wrapping_add(1);
        self.cycles_executed = self.cycles_executed.wrapping_add(1);
    }

    /// Applies the `V0` (`VectorAluSlot`) instruction across all elements of the physical tile
    /// in `vreg_tiles[src0_vreg]`, `vreg_tiles[src1_vreg]`, and `vreg_tiles[src2_vreg]` when populated,
    /// honoring `$vmr` (`vmask`) predication (`Select`, `Clamp`, `Compare*`), integer bitwise ops,
    /// and quantization (`UniformQuantize`, `UniformDequantize`, `BitcastConvert`).
    ///
    /// # Arguments
    ///
    /// * `vslot` - Architectural `VectorAluSlot` instruction fields.
    /// * `op_opt` - Decoded [`crate::compiler::TpuOpCode`], if valid.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn step_v0_tile(
        &mut self,
        vslot: &crate::compiler::VectorAluSlot,
        op_opt: Option<crate::compiler::TpuOpCode>,
    ) {
        let s0_key = (vslot.src0_vreg & 255) as u8;
        let s1_key = (vslot.src1_vreg & 255) as u8;
        let s2_key = (vslot.src2_vreg & 255) as u8;
        let dst_key = (vslot.dst_vreg & 255) as u8;
        let elem_type =
            ElementType::from_raw(i32::from(vslot.element_type)).unwrap_or(ElementType::F32);
        let op = op_opt.unwrap_or(crate::compiler::TpuOpCode::AddF32);
        let empty: &[u8] = &[];
        let t0_bytes = self.vreg_tiles.get(&s0_key).map_or(empty, Vec::as_slice);
        let t1_bytes = self.vreg_tiles.get(&s1_key).map_or(t0_bytes, Vec::as_slice);
        let t2_bytes = self.vreg_tiles.get(&s2_key).map_or(t1_bytes, Vec::as_slice);
        let v0 = elem_type.unpack_to_f64_vec(t0_bytes);
        let v1 = elem_type.unpack_to_f64_vec(t1_bytes);
        let v2 = elem_type.unpack_to_f64_vec(t2_bytes);
        let len = v0.len();
        let mut out_vals = Vec::with_capacity(len);
        for idx in 0..len {
            let a = v0.get(idx).copied().unwrap_or(0.0);
            let b = v1.get(idx).copied().unwrap_or(a);
            let c = v2.get(idx).copied().unwrap_or(b);
            let mask_bit = (vslot.lane_mask & (1 << (idx & 7))) != 0;
            let val = match op {
                crate::compiler::TpuOpCode::SubF32 => a - b,
                crate::compiler::TpuOpCode::MulF32 => a * b,
                crate::compiler::TpuOpCode::DivF32 => a / b,
                crate::compiler::TpuOpCode::MaxF32 => a.max(b),
                crate::compiler::TpuOpCode::MinF32 => a.min(b),
                crate::compiler::TpuOpCode::NegF32 => -a,
                crate::compiler::TpuOpCode::Abs => a.abs(),
                crate::compiler::TpuOpCode::Identity
                | crate::compiler::TpuOpCode::BitcastConvert => a,
                crate::compiler::TpuOpCode::Select => {
                    if a != 0.0 && mask_bit {
                        b
                    } else {
                        c
                    }
                }
                crate::compiler::TpuOpCode::Clamp => b.clamp(a, c),
                crate::compiler::TpuOpCode::CompareEq => {
                    let eq = (a - b).abs() < 1e-9;
                    self.vmask.iter_mut().take(1).for_each(|m| *m = eq);
                    f64::from(u8::from(eq))
                }
                crate::compiler::TpuOpCode::CompareLt => {
                    let lt = a < b;
                    self.vmask.iter_mut().take(1).for_each(|m| *m = lt);
                    f64::from(u8::from(lt))
                }
                crate::compiler::TpuOpCode::CompareGt => {
                    let gt = a > b;
                    self.vmask.iter_mut().take(1).for_each(|m| *m = gt);
                    f64::from(u8::from(gt))
                }
                crate::compiler::TpuOpCode::And => f64::from((a as i32) & (b as i32)),
                crate::compiler::TpuOpCode::Or => f64::from((a as i32) | (b as i32)),
                crate::compiler::TpuOpCode::XorBytes => f64::from((a as i32) ^ (b as i32)),
                crate::compiler::TpuOpCode::Not => f64::from(!(a as i32)),
                crate::compiler::TpuOpCode::ShiftLeft => {
                    f64::from((a as i32).wrapping_shl((b.max(0.0) as u32) & 31))
                }
                crate::compiler::TpuOpCode::ShiftRightLogical => {
                    #[allow(clippy::cast_possible_wrap)]
                    let shifted = ((a as u32) >> ((b.max(0.0) as u32) & 31)) as i32;
                    f64::from(shifted)
                }
                crate::compiler::TpuOpCode::ShiftRightArithmetic => {
                    f64::from((a as i32) >> ((b.max(0.0) as u32) & 31))
                }
                crate::compiler::TpuOpCode::PopulationCount => {
                    f64::from(((a as i32) as u32).count_ones())
                }
                crate::compiler::TpuOpCode::CountLeadingZeros => {
                    f64::from(((a as i32) as u32).leading_zeros())
                }
                crate::compiler::TpuOpCode::UniformQuantize => {
                    (a * 127.0).round().clamp(-128.0, 127.0)
                }
                crate::compiler::TpuOpCode::UniformDequantize => a / 127.0,
                _ => a + b,
            };
            out_vals.push(val);
        }
        let packed = elem_type.pack_from_f64_slice(&out_vals);
        self.vmem_allocations
            .insert(u64::from(dst_key) * 256, packed.clone());
        self.vreg_tiles.insert(dst_key, packed);
    }

    /// Applies the `X0` (`ExtendedUnarySlot`) transcendental and cross-lane `XLU` permute instruction
    /// (`Transpose`, `BroadcastInDim`, `Reverse`, `Iota`, `Roll`, `Concatenate`, `Slice`, `Pad`,
    /// `Gather`, `Scatter`) across all elements of `vreg_tiles[src_vreg]` when populated.
    ///
    /// # Arguments
    ///
    /// * `xslot` - Architectural `ExtendedUnarySlot` instruction fields.
    /// * `op_opt` - Decoded [`crate::compiler::TpuOpCode`], if valid.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn step_x0_tile(
        &mut self,
        xslot: crate::compiler::ExtendedUnarySlot,
        op_opt: Option<crate::compiler::TpuOpCode>,
    ) {
        let src_key = (xslot.src_vreg & 255) as u8;
        let dst_key = (xslot.dst_vreg & 255) as u8;
        let op = op_opt.unwrap_or(crate::compiler::TpuOpCode::ExpF32);
        let empty: &[u8] = &[];
        let t_bytes = self.vreg_tiles.get(&src_key).map_or(empty, Vec::as_slice);
        let vals = ElementType::F32.unpack_to_f64_vec(t_bytes);
        let out_vals: Vec<f64> = match op {
            crate::compiler::TpuOpCode::Transpose => {
                let rows = usize::from(xslot.perm_mode.max(1));
                let cols = vals.len().checked_div(rows).unwrap_or(1).max(1);
                let mut tr = vec![0.0_f64; vals.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        for (&src_v, dst_v) in vals
                            .get(r * cols + c)
                            .into_iter()
                            .zip(tr.get_mut(c * rows + r))
                        {
                            *dst_v = src_v;
                        }
                    }
                }
                tr
            }
            crate::compiler::TpuOpCode::Reverse => vals.iter().copied().rev().collect(),
            crate::compiler::TpuOpCode::Iota => {
                let count = vals.len().max(usize::from(xslot.imm16).max(4));
                (0..count).map(|i| f64::from(i as u32)).collect()
            }
            crate::compiler::TpuOpCode::DynamicSlice => {
                let mut rolled = vals;
                let r_len = rolled.len();
                let shift = usize::from(xslot.imm16.max(1)) % r_len.max(1);
                rolled.rotate_right(shift.min(r_len));
                rolled
            }
            crate::compiler::TpuOpCode::BroadcastInDim => {
                let target_len = usize::from(xslot.imm16).max(vals.len()).max(1);
                let src_len = vals.len().max(1);
                (0..target_len)
                    .map(|i| vals.get(i % src_len).copied().unwrap_or(0.0))
                    .collect()
            }
            crate::compiler::TpuOpCode::Slice => {
                let start = usize::from(xslot.perm_mode).min(vals.len());
                let limit = usize::from(xslot.imm16.max(1)).min(vals.len()).max(start);
                vals.get(start..limit).unwrap_or_default().to_vec()
            }
            crate::compiler::TpuOpCode::Pad => {
                let pad_low = usize::from(xslot.perm_mode);
                let pad_high = usize::from(xslot.imm16);
                let mut padded = vec![0.0_f64; pad_low];
                padded.extend_from_slice(&vals);
                padded.resize(padded.len() + pad_high, 0.0);
                padded
            }
            crate::compiler::TpuOpCode::Concatenate => {
                let aux_key = (xslot.imm16 & 255) as u8;
                let aux_bytes = self.vreg_tiles.get(&aux_key).map_or(empty, Vec::as_slice);
                let mut cat = vals;
                cat.extend(ElementType::F32.unpack_to_f64_vec(aux_bytes));
                cat
            }
            crate::compiler::TpuOpCode::Gather => {
                let idx_key = (xslot.imm16 & 255) as u8;
                let idx_bytes = self.vreg_tiles.get(&idx_key).map_or(t_bytes, Vec::as_slice);
                let indices = ElementType::F32.unpack_to_f64_vec(idx_bytes);
                let n = vals.len().max(1);
                indices
                    .iter()
                    .map(|&idx_f| {
                        vals.get((idx_f.max(0.0) as usize) % n)
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .collect()
            }
            crate::compiler::TpuOpCode::Scatter => {
                let mut scattered = vals;
                let upd_key = (xslot.imm16 & 255) as u8;
                let upd_bytes = self.vreg_tiles.get(&upd_key).map_or(t_bytes, Vec::as_slice);
                let updates = ElementType::F32.unpack_to_f64_vec(upd_bytes);
                let pos = usize::from(xslot.perm_mode);
                for (slot, &u) in scattered.iter_mut().skip(pos).zip(updates.iter()) {
                    *slot = u;
                }
                scattered
            }
            _ => vals
                .iter()
                .map(|&x| match op {
                    crate::compiler::TpuOpCode::TanhF32 => x.tanh(),
                    crate::compiler::TpuOpCode::Log => x.ln(),
                    crate::compiler::TpuOpCode::Sqrt => x.sqrt(),
                    crate::compiler::TpuOpCode::Rsqrt => 1.0 / x.sqrt(),
                    crate::compiler::TpuOpCode::Sine => x.sin(),
                    crate::compiler::TpuOpCode::Cosine => x.cos(),
                    _ => x.exp(),
                })
                .collect(),
        };
        let packed = ElementType::F32.pack_from_f64_slice(&out_vals);
        self.vreg_tiles.insert(dst_key, packed);
    }

    /// Quantizes a single `f64` input operand according to `MatrixMxuSlot::precision_mode`
    /// before accumulating into `f32`/`f64` inside the systolic MXU array:
    /// - `0`: `BF16` round-to-nearest-even (`half::bf16::from_f64(x).to_f64()`)
    /// - `1`: `F32` IEEE-754 (`(x as f32) as f64`)
    /// - `2`: `INT8` / `S8` clamped signed 8-bit integer (`x.round().clamp(-128.0, 127.0)`)
    /// - `3`: `F8E4M3FN` / `F8E5M2` 8-bit float (`ElementType::F8E4M3Fn` round-trip)
    /// - `4`: `S4` / `U4` sub-byte 4-bit integer (`x.round().clamp(-8.0, 7.0)`)
    ///
    /// # Arguments
    ///
    /// * `val` - Raw `f64` matrix operand value.
    /// * `precision_mode` - `MatrixMxuSlot::precision_mode` code (`0..=4`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    fn quantize_mxu_operand(val: f64, precision_mode: u8) -> f64 {
        match precision_mode {
            0 => half::bf16::from_f64(val).to_f64(),
            2 => val.round().clamp(-128.0, 127.0),
            3 => {
                let raw = ElementType::F8E4M3Fn.pack_from_f64_slice(&[val]);
                ElementType::F8E4M3Fn
                    .unpack_to_f64_vec(&raw)
                    .first()
                    .copied()
                    .unwrap_or(val)
            }
            4 => val.round().clamp(-8.0, 7.0),
            _ => f64::from(val as f32),
        }
    }

    /// Applies the `M0` (`MatrixMxuSlot`) systolic contraction across `vreg_tiles[lhs_vreg]` and
    /// `mxu_weights` (or `vreg_tiles[rhs_vreg]`) into `vreg_tiles[acc_vreg]`.
    ///
    /// Supports arbitrary rectangular and batched contractions `[B, M, K] x [B, K, N] -> [B, M, N]`
    /// encoded in `mslot.flags` (`(M << 8) | N`) and `mslot.k_tiles` (`(B << 8) | K`), as well as
    /// automatic square `[N, N] x [N, N]` inference when `flags <= 1`, and honors `mslot.precision_mode`
    /// (`BF16`, `F32`, `INT8`, `FP8`, `S4`).
    ///
    /// # Arguments
    ///
    /// * `mslot` - Architectural `MatrixMxuSlot` instruction fields.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn step_m0_tile(&mut self, mslot: &crate::compiler::MatrixMxuSlot) {
        let lhs_key = (mslot.lhs_vreg & 255) as u8;
        let rhs_key = (mslot.rhs_vreg & 255) as u8;
        let acc_key = (mslot.acc_vreg & 255) as u8;
        let empty: &[u8] = &[];
        let lhs_bytes = self.vreg_tiles.get(&lhs_key).map_or(empty, Vec::as_slice);
        let lhs_vals = ElementType::F32.unpack_to_f64_vec(lhs_bytes);
        let rhs_vals = if self.mxu_weights.is_empty() {
            let rhs_bytes = self
                .vreg_tiles
                .get(&rhs_key)
                .map_or(lhs_bytes, Vec::as_slice);
            ElementType::F32.unpack_to_f64_vec(rhs_bytes)
        } else {
            self.mxu_weights.clone()
        };

        let sq = lhs_vals.len().isqrt().max(1);
        let m_dim = usize::from(mslot.flags >> 8).max(sq);
        let n_dim = usize::from((mslot.flags & 0xFF) & !1).max(m_dim);
        let batch = usize::from(mslot.k_tiles >> 8).max(1);
        let k_dim = usize::from((mslot.k_tiles & 0xFF) & !1).max(m_dim);

        let mut out_vals = Vec::with_capacity(batch * m_dim * n_dim);
        for b in 0..batch {
            let lhs_b_off = b * m_dim * k_dim;
            let rhs_b_off = b * k_dim * n_dim;
            for r in 0..m_dim {
                for c in 0..n_dim {
                    let mut acc = 0.0_f32;
                    for k in 0..k_dim {
                        let raw_l = lhs_vals
                            .get(lhs_b_off + r * k_dim + k)
                            .copied()
                            .unwrap_or(0.0);
                        let raw_r = rhs_vals
                            .get(rhs_b_off + k * n_dim + c)
                            .copied()
                            .unwrap_or(0.0);
                        let ql = Self::quantize_mxu_operand(raw_l, mslot.precision_mode) as f32;
                        let qr = Self::quantize_mxu_operand(raw_r, mslot.precision_mode) as f32;
                        acc = ql.mul_add(qr, acc);
                    }
                    out_vals.push(f64::from(acc));
                }
            }
        }
        let packed = ElementType::F32.pack_from_f64_slice(&out_vals);
        self.vreg_tiles.insert(acc_key, packed);
    }

    /// Decodes a binary `IMEM` image of 64-byte [`crate::compiler::TensorCoreVliwBundle`] words and steps
    /// the [`TpuCoreVmState`] VLIW execution engine through every bundle.
    ///
    /// # Arguments
    ///
    /// * `imem_bytes` - Raw `IMEM` binary image produced by [`CompiledProgram::encode_architectural_imem`].
    pub fn execute_imem_bytes(&mut self, imem_bytes: &[u8]) {
        let bundles = CompiledProgram::decode_architectural_imem(imem_bytes);
        self.execute_vliw_stream(&bundles);
    }

    /// Loads a raw serialized 64-byte VLIW `IMEM` binary stream, reconstructs the architectural
    /// [`crate::compiler::TpuInstructionBundle`] sequence from the decoded 512-bit instruction words,
    /// and executes the program standalone over `input_buffers` to produce `output_count` output buffers.
    ///
    /// # Arguments
    ///
    /// * `imem_bytes` - Raw `IMEM` binary image (must be a non-empty multiple of 64 bytes).
    /// * `input_buffers` - Input HBM tensor byte buffers transferred into VMEM slots `0..inputs.len()`.
    /// * `output_count` - Number of output tensor buffers requested by the caller.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `imem_bytes` is empty, not a multiple of 64 bytes,
    /// or contains an invalid VLIW opcode.
    pub fn execute_imem_binary(
        &mut self,
        imem_bytes: &[u8],
        input_buffers: Vec<Vec<u8>>,
        output_count: usize,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        if imem_bytes.is_empty() || (imem_bytes.len() & 63) != 0 {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!(
                    "IMEM binary size ({}) must be a non-empty multiple of 64 bytes",
                    imem_bytes.len()
                ),
            });
        }
        let vliw_bundles = CompiledProgram::decode_architectural_imem(imem_bytes);
        let mut ir_bundles = Vec::with_capacity(vliw_bundles.len());
        let mut output_slots = Vec::new();
        for vb in &vliw_bundles {
            let ib = vb.to_instruction_bundle()?;
            output_slots.push(ib.output_slot);
            ir_bundles.push(ib);
        }
        let last_out = output_slots.last().copied().unwrap_or(0);
        let first_op = ir_bundles
            .first()
            .map_or(crate::compiler::TpuOpCode::Identity, |b| b.opcode);
        let synthetic_prog = CompiledProgram {
            name: String::from("imem_binary_program"),
            format: String::from("imem"),
            program_format: crate::compiler::ProgramFormat::Mlir,
            opcode: first_op,
            instruction_bundles: ir_bundles,
            output_slots: vec![last_out; output_count.max(1)],
            parameter_shapes: Vec::new(),
            output_shapes: Vec::new(),
            parameter_memory_kinds: Vec::new(),
            output_memory_kinds: Vec::new(),
            fingerprint: String::from("imem"),
            num_replicas: 1,
            num_partitions: 1,
            cost_analysis: crate::compiler::CostAnalysis {
                flops: 1.0,
                transcendentals: 0.0,
                bytes_accessed: 256.0,
                optimal_seconds: 1.0e-6,
            },
            memory_stats: crate::compiler::CompiledMemoryStats {
                generated_code_size_in_bytes: i64::try_from(imem_bytes.len()).unwrap_or(256),
                argument_size_in_bytes: 256,
                output_size_in_bytes: 256,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 256,
            },
            mosaic_metadata: crate::compiler::MosaicTargetMetadata::default(),
            source_bytes: Vec::new(),
            compile_options_bytes: Vec::new(),
        };
        Ok(self.execute_compiled_program(&synthetic_prog, input_buffers, output_count))
    }

    /// Initializes `SyncFlag` counters and executes all `IMEM` overlays in a [`crate::proto::DeepseaExecutableProto`].
    ///
    /// # Arguments
    ///
    /// * `ds_exec` - Compiled [`crate::proto::DeepseaExecutableProto`] payload.
    #[allow(clippy::indexing_slicing)]
    pub fn execute_deepsea_proto(&mut self, ds_exec: &crate::proto::DeepseaExecutableProto) {
        for sf in &ds_exec.sync_flags {
            let idx = (sf.sync_flag_id as usize) & 255;
            self.sync_flags[idx] = sf.initial_value;
        }
        for overlay in &ds_exec.overlays {
            self.execute_imem_bytes(&overlay.encoded_bundles);
        }
    }

    /// Dispatches a `TN_QUEUE_CONTINUATION` [`crate::proto::HostCommandEnvelopeProto`] event,
    /// updating the `SREG[0]` continuation sequence register, processing `IMEM` or `DMA_H2D` payloads
    /// when `opcode == 1` or `opcode == 2`, and incrementing `SYNC_FLAGS[0]`.
    ///
    /// # Arguments
    ///
    /// * `cmd` - Host command envelope from the `TensorCore` continuation queue.
    #[allow(clippy::indexing_slicing)]
    pub fn dispatch_host_command(&mut self, cmd: &crate::proto::HostCommandEnvelopeProto) {
        self.sreg[0] = cmd.continuation_id;
        if cmd.opcode == 1 {
            self.execute_imem_bytes(&cmd.payload);
        } else {
            self.vmem_allocations
                .insert(cmd.continuation_id * 256, cmd.payload.clone());
        }
        self.sync_flags[0] = self.sync_flags[0].wrapping_add(1);
    }

    /// Executes a [`CompiledProgram`] inside this [`TpuCoreVmState`] (`DeepSeaVm`), loading
    /// `input_buffers` into VMEM registers, stepping each architectural 64-byte VLIW instruction
    /// bundle in lockstep with tensor evaluation, and returning the output tensor buffers for
    /// `output_count` outputs.
    ///
    /// # Arguments
    ///
    /// * `program` - Lowered [`CompiledProgram`] containing VLIW instruction bundles and output slot mappings.
    /// * `input_buffers` - Input HBM tensor byte buffers transferred into VMEM slots `0..inputs.len()`.
    /// * `output_count` - Number of output tensor buffers requested by the caller.
    #[allow(clippy::indexing_slicing)]
    pub fn execute_compiled_program(
        &mut self,
        program: &CompiledProgram,
        input_buffers: Vec<Vec<u8>>,
        output_count: usize,
    ) -> Vec<Vec<u8>> {
        let default_elem_type = program
            .output_shapes
            .first()
            .map_or(ElementType::F32, |s| s.element_type);

        for (in_slot, in_buf) in input_buffers.iter().enumerate() {
            let word0 = u32::from_le(bytemuck::pod_read_unaligned::<u32>(
                in_buf.get(0..4).unwrap_or(&[0; 4]),
            ));
            self.vreg_words[in_slot & 63] = word0;
            self.vmem_words[in_slot & 255] = word0;
            #[allow(clippy::cast_possible_truncation)]
            self.vreg_tiles
                .insert((in_slot & 255) as u8, in_buf.clone());
            self.vmem_allocations
                .insert((in_slot as u64) * 256, in_buf.clone());
        }

        let mut registers = input_buffers;
        let max_slot = program
            .instruction_bundles
            .iter()
            .map(|b| b.output_slot)
            .chain(program.output_slots.iter().copied())
            .max()
            .unwrap_or(0);
        let total_slots = (max_slot + 1).max(registers.len());
        registers.resize_with(total_slots, Vec::new);
        let mut last_result = registers.first().cloned().unwrap_or_default();

        for bundle in &program.instruction_bundles {
            let arch_bundle = bundle.encode_architectural_bundle();
            self.step_vliw_bundle(&arch_bundle);
            let mut op_slices: Vec<&[u8]> = Vec::with_capacity(bundle.operand_slots.len());
            for &slot in &bundle.operand_slots {
                let slice = registers.get(slot).map_or(&[][..], Vec::as_slice);
                op_slices.push(slice);
            }
            let bundle_elem_type = bundle
                .attrs
                .output_shape
                .as_ref()
                .map_or(default_elem_type, |s| s.element_type);
            if let Some(&rhs_bytes) = op_slices.last() {
                self.mxu_weights = bundle_elem_type.unpack_to_f64_vec(rhs_bytes);
            }
            let evaluated =
                SimulatedTpuHal::eval_bundle_with_attrs(bundle, bundle_elem_type, &op_slices);
            let word0 = u32::from_le(bytemuck::pod_read_unaligned::<u32>(
                evaluated.get(0..4).unwrap_or(&[0; 4]),
            ));
            self.vreg_words[bundle.output_slot & 63] = word0;
            self.vmem_words[bundle.output_slot & 255] = word0;
            #[allow(clippy::cast_possible_truncation)]
            self.vreg_tiles
                .insert((bundle.output_slot & 255) as u8, evaluated.clone());
            self.vmem_allocations
                .insert((bundle.output_slot as u64) * 256, evaluated.clone());
            last_result.clone_from(&evaluated);
            registers[bundle.output_slot] = evaluated;
        }

        (0..output_count)
            .map(|out_idx| {
                program
                    .output_slots
                    .get(out_idx)
                    .and_then(|&slot| registers.get(slot))
                    .unwrap_or(&last_result)
                    .clone()
            })
            .collect()
    }

    /// Executes a [`CompiledProgram`] with strict output-slot validation, returning
    /// [`LibTpuError::ExecutionFailed`] if any declared `output_slots` index is out of bounds or empty.
    ///
    /// # Arguments
    ///
    /// * `program` - Lowered [`CompiledProgram`] to execute.
    /// * `input_buffers` - Input HBM tensor byte buffers.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ExecutionFailed`] if `program.output_slots` is empty or references an
    /// unpopulated register slot.
    pub fn execute(
        &mut self,
        program: &CompiledProgram,
        input_buffers: Vec<Vec<u8>>,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        if program.output_slots.is_empty() {
            return Err(LibTpuError::ExecutionFailed {
                device_id: 0,
                reason: String::from("CompiledProgram has no declared output_slots"),
            });
        }
        let max_produced_slot = program
            .instruction_bundles
            .iter()
            .map(|b| b.output_slot)
            .max()
            .unwrap_or(0);
        for &slot in &program.output_slots {
            if slot > max_produced_slot.max(input_buffers.len().saturating_sub(1)) {
                return Err(LibTpuError::ExecutionFailed {
                    device_id: 0,
                    reason: format!(
                        "Output slot {slot} exceeds maximum populated VM register slot {max_produced_slot}"
                    ),
                });
            }
        }
        Ok(self.execute_compiled_program(program, input_buffers, program.output_slots.len()))
    }
}

/// Alias for [`TpuCoreVmState`] (`DeepSeaVm` architectural VLIW execution engine).
pub type DeepSeaVm = TpuCoreVmState;

/// `SparseCore` optimizer algorithm for embedding weight updates (`SparseCoreOptimizer`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparseCoreOptimizer {
    /// Stochastic Gradient Descent (`w -= lr * g`).
    Sgd {
        /// Learning rate (`lr`).
        learning_rate: f32,
    },
    /// Adaptive Gradient (`accum += g * g; w -= lr * g / sqrt(accum + eps)`).
    Adagrad {
        /// Learning rate (`lr`).
        learning_rate: f32,
        /// Numerical stability epsilon (`eps`).
        epsilon: f32,
    },
    /// Adaptive Moment Estimation (`m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2; w -= lr * m / (sqrt(v) + eps)`).
    Adam {
        /// Learning rate (`lr`).
        learning_rate: f32,
        /// Exponential decay rate for first moment (`beta1`, e.g. `0.9`).
        beta1: f32,
        /// Exponential decay rate for second moment (`beta2`, e.g. `0.999`).
        beta2: f32,
        /// Numerical stability epsilon (`eps`, e.g. `1e-8`).
        epsilon: f32,
    },
    /// Follow-The-Regularized-Leader (`FTRL-Proximal`) sparse embedding optimizer.
    Ftrl {
        /// Learning rate (`lr`).
        learning_rate: f32,
        /// Learning rate power (`lr_power`, typically `-0.5`).
        learning_rate_power: f32,
        /// L1 regularization strength (`l1`).
        l1_regularization: f32,
        /// L2 regularization strength (`l2`).
        l2_regularization: f32,
    },
    /// Root Mean Square Propagation (`accum = decay * accum + (1 - decay) * g^2; mom = momentum * mom + lr * g / sqrt(accum + eps); w -= mom`).
    RmsProp {
        /// Learning rate (`lr`).
        learning_rate: f32,
        /// Discounting factor (`decay`, e.g. `0.9`).
        decay: f32,
        /// Momentum factor (`momentum`, e.g. `0.0` or `0.9`).
        momentum: f32,
        /// Numerical stability epsilon (`eps`).
        epsilon: f32,
    },
    /// Classical / Nesterov Momentum (`mom = momentum * mom + g; w -= lr * mom`).
    Momentum {
        /// Learning rate (`lr`).
        learning_rate: f32,
        /// Momentum decay factor (`momentum`, e.g. `0.9`).
        momentum: f32,
    },
    /// Evolved Sign Momentum (`Lion`: `c = b1 * m + (1 - b1) * g; w -= lr * (sign(c) + wd * w); m = b2 * m + (1 - b2) * g`).
    Lion {
        /// Learning rate (`lr`).
        learning_rate: f32,
        /// Update interpolation factor (`beta1`, e.g. `0.9`).
        beta1: f32,
        /// EMA momentum factor (`beta2`, e.g. `0.99`).
        beta2: f32,
        /// Decoupled weight decay (`weight_decay`).
        weight_decay: f32,
    },
}

/// Architectural `SparseCore` (`SC0`..`SC3`) embedding gather & scatter-update state machine (`SparseCoreVmState`).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseCoreVmState {
    /// `SparseCore` ordinal index (`0..4`).
    pub core_index: u32,
    /// Flattened `[vocab_size, embedding_dim]` embedding table weights in HBM/VMEM.
    pub table_weights: Vec<f32>,
    /// Flattened `[vocab_size, embedding_dim]` primary accumulator / first-moment (`m` / `accum`) state.
    pub accumulator: Vec<f32>,
    /// Flattened `[vocab_size, embedding_dim]` secondary auxiliary (`v` / `mom` / `linear`) state.
    pub secondary_state: Vec<f32>,
    /// Vocabulary row count.
    pub vocab_size: usize,
    /// Embedding vector width (`dimension`).
    pub embedding_dim: usize,
    /// Total sparse gather/scatter steps executed.
    pub steps_executed: u64,
}

impl SparseCoreVmState {
    /// Creates a new [`SparseCoreVmState`] for a table of shape `[vocab_size, embedding_dim]`.
    ///
    /// # Arguments
    ///
    /// * `core_index` - `SparseCore` ordinal (`0..4`).
    /// * `vocab_size` - Number of rows in the embedding table.
    /// * `embedding_dim` - Width of each embedding row.
    /// * `initial_fill` - Initial weight scalar (`e.g. 0.1`).
    #[must_use]
    pub fn new(
        core_index: u32,
        vocab_size: usize,
        embedding_dim: usize,
        initial_fill: f32,
    ) -> Self {
        let len = vocab_size.saturating_mul(embedding_dim).max(1);
        Self {
            core_index,
            table_weights: vec![initial_fill; len],
            accumulator: vec![0.1_f32; len],
            secondary_state: vec![0.0_f32; len],
            vocab_size: vocab_size.max(1),
            embedding_dim: embedding_dim.max(1),
            steps_executed: 0,
        }
    }

    /// Executes a deduplicated sparse embedding lookup (`Gather`) for `row_ids`.
    ///
    /// # Arguments
    ///
    /// * `row_ids` - Slice of vocabulary row indices to gather.
    #[must_use]
    pub fn sparse_gather(&mut self, row_ids: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(row_ids.len() * self.embedding_dim);
        for &raw_id in row_ids {
            let row = (raw_id as usize) % self.vocab_size;
            let start = row * self.embedding_dim;
            for col in 0..self.embedding_dim {
                let val = self.table_weights.get(start + col).copied().unwrap_or(0.0);
                out.push(val);
            }
        }
        self.steps_executed = self.steps_executed.wrapping_add(1);
        out
    }

    /// Applies sparse gradients to `row_ids` using `optimizer` (`ScatterAccumulate`).
    ///
    /// # Arguments
    ///
    /// * `row_ids` - Vocabulary row indices receiving gradient updates.
    /// * `gradients` - Flattened gradient slice (`row_ids.len() * embedding_dim`).
    /// * `optimizer` - [`SparseCoreOptimizer`] update rule (`Sgd`, `Adagrad`, `Adam`, `Ftrl`, `RmsProp`, `Momentum`, or `Lion`).
    pub fn sparse_scatter_update(
        &mut self,
        row_ids: &[u32],
        gradients: &[f32],
        optimizer: SparseCoreOptimizer,
    ) {
        for (i, &raw_id) in row_ids.iter().enumerate() {
            let row = (raw_id as usize) % self.vocab_size;
            let row_base = row * self.embedding_dim;
            let grad_base = i * self.embedding_dim;
            for col in 0..self.embedding_dim {
                let g = gradients.get(grad_base + col).copied().unwrap_or(0.0);
                let idx = row_base + col;
                self.update_single_element(idx, g, optimizer);
            }
        }
        self.steps_executed = self.steps_executed.wrapping_add(1);
    }

    /// Updates a single embedding parameter element at `idx` with gradient `g` and `optimizer`.
    ///
    /// # Arguments
    ///
    /// * `idx` - Flattened parameter index (`row * embedding_dim + col`).
    /// * `g` - Scalar gradient value.
    /// * `optimizer` - [`SparseCoreOptimizer`] update rule.
    fn update_single_element(&mut self, idx: usize, g: f32, optimizer: SparseCoreOptimizer) {
        match optimizer {
            SparseCoreOptimizer::Sgd { learning_rate } => {
                self.table_weights.get_mut(idx).into_iter().for_each(|w| {
                    *w = learning_rate.mul_add(-g, *w);
                });
            }
            SparseCoreOptimizer::Adagrad {
                learning_rate,
                epsilon,
            } => {
                let mut new_acc = 0.1_f32;
                self.accumulator.get_mut(idx).into_iter().for_each(|acc| {
                    *acc = g.mul_add(g, *acc);
                    new_acc = *acc;
                });
                let delta = learning_rate * g / (new_acc + epsilon).sqrt();
                self.table_weights.get_mut(idx).into_iter().for_each(|w| {
                    *w -= delta;
                });
            }
            SparseCoreOptimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                self.apply_adam_element(idx, g, learning_rate, beta1, beta2, epsilon);
            }
            SparseCoreOptimizer::Ftrl {
                learning_rate,
                learning_rate_power,
                l1_regularization,
                l2_regularization,
            } => {
                self.apply_ftrl_element(
                    idx,
                    g,
                    learning_rate,
                    learning_rate_power,
                    l1_regularization,
                    l2_regularization,
                );
            }
            SparseCoreOptimizer::RmsProp {
                learning_rate,
                decay,
                momentum,
                epsilon,
            } => {
                self.apply_rmsprop_element(idx, g, learning_rate, decay, momentum, epsilon);
            }
            SparseCoreOptimizer::Momentum {
                learning_rate,
                momentum,
            } => {
                let mut mom_val = 0.0_f32;
                self.secondary_state
                    .get_mut(idx)
                    .into_iter()
                    .for_each(|mom| {
                        *mom = momentum.mul_add(*mom, g);
                        mom_val = *mom;
                    });
                self.table_weights.get_mut(idx).into_iter().for_each(|w| {
                    *w = learning_rate.mul_add(-mom_val, *w);
                });
            }
            SparseCoreOptimizer::Lion {
                learning_rate,
                beta1,
                beta2,
                weight_decay,
            } => {
                let m_prev = self.accumulator.get(idx).copied().unwrap_or(0.0);
                let c = beta1.mul_add(m_prev, (1.0 - beta1) * g);
                let sign_c = c.signum();
                self.table_weights.get_mut(idx).into_iter().for_each(|w| {
                    *w = learning_rate.mul_add(-weight_decay.mul_add(*w, sign_c), *w);
                });
                self.accumulator.get_mut(idx).into_iter().for_each(|m| {
                    *m = beta2.mul_add(*m, (1.0 - beta2) * g);
                });
            }
        }
    }

    /// Applies an `FTRL-Proximal` update step to a single parameter element at `idx`.
    ///
    /// # Arguments
    ///
    /// * `idx` - Flattened parameter index.
    /// * `g` - Scalar gradient value.
    /// * `learning_rate` - Learning rate (`lr`).
    /// * `learning_rate_power` - Learning rate power (`lr_power`).
    /// * `l1_regularization` - L1 regularization strength.
    /// * `l2_regularization` - L2 regularization strength.
    fn apply_ftrl_element(
        &mut self,
        idx: usize,
        g: f32,
        learning_rate: f32,
        learning_rate_power: f32,
        l1_regularization: f32,
        l2_regularization: f32,
    ) {
        let old_acc = self.accumulator.get(idx).copied().unwrap_or(0.1).max(1e-8);
        let new_acc = g.mul_add(g, old_acc);
        self.accumulator.get_mut(idx).into_iter().for_each(|acc| {
            *acc = new_acc;
        });
        let sigma = (new_acc.powf(-learning_rate_power) - old_acc.powf(-learning_rate_power))
            / learning_rate.max(1e-8);
        let w_prev = self.table_weights.get(idx).copied().unwrap_or(0.0);
        let mut z_val = 0.0_f32;
        self.secondary_state.get_mut(idx).into_iter().for_each(|z| {
            *z += sigma.mul_add(-w_prev, g);
            z_val = *z;
        });
        let sign_z = z_val.signum();
        let new_w = if z_val.abs() <= l1_regularization {
            0.0_f32
        } else {
            let denom = 2.0_f32.mul_add(
                l2_regularization,
                new_acc.powf(-learning_rate_power) / learning_rate.max(1e-8),
            );
            sign_z.mul_add(l1_regularization, -z_val) / denom.max(1e-8)
        };
        self.table_weights.get_mut(idx).into_iter().for_each(|w| {
            *w = new_w;
        });
    }

    /// Applies an `Adam` update step to a single parameter element at `idx`.
    ///
    /// # Arguments
    ///
    /// * `idx` - Flattened parameter index.
    /// * `g` - Scalar gradient value.
    /// * `learning_rate` - Learning rate (`lr`).
    /// * `beta1` - First-moment decay (`beta1`).
    /// * `beta2` - Second-moment decay (`beta2`).
    /// * `epsilon` - Stability epsilon (`eps`).
    fn apply_adam_element(
        &mut self,
        idx: usize,
        g: f32,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) {
        let mut m_val = 0.0_f32;
        self.accumulator.get_mut(idx).into_iter().for_each(|m| {
            *m = beta1.mul_add(*m, (1.0 - beta1) * g);
            m_val = *m;
        });
        let mut v_val = 0.0_f32;
        self.secondary_state.get_mut(idx).into_iter().for_each(|v| {
            *v = beta2.mul_add(*v, (1.0 - beta2) * g * g);
            v_val = *v;
        });
        let delta = learning_rate * m_val / (v_val.sqrt() + epsilon);
        self.table_weights.get_mut(idx).into_iter().for_each(|w| {
            *w -= delta;
        });
    }

    /// Applies an `RMSProp` update step to a single parameter element at `idx`.
    ///
    /// # Arguments
    ///
    /// * `idx` - Flattened parameter index.
    /// * `g` - Scalar gradient value.
    /// * `learning_rate` - Learning rate (`lr`).
    /// * `decay` - Discounting factor (`decay`).
    /// * `momentum` - Momentum factor (`momentum`).
    /// * `epsilon` - Stability epsilon (`eps`).
    fn apply_rmsprop_element(
        &mut self,
        idx: usize,
        g: f32,
        learning_rate: f32,
        decay: f32,
        momentum: f32,
        epsilon: f32,
    ) {
        let mut acc_val = 0.1_f32;
        self.accumulator.get_mut(idx).into_iter().for_each(|acc| {
            *acc = decay.mul_add(*acc, (1.0 - decay) * g * g);
            acc_val = *acc;
        });
        let mut mom_val = 0.0_f32;
        self.secondary_state
            .get_mut(idx)
            .into_iter()
            .for_each(|mom| {
                *mom = momentum.mul_add(*mom, learning_rate * g / (acc_val + epsilon).sqrt());
                mom_val = *mom;
            });
        self.table_weights.get_mut(idx).into_iter().for_each(|w| {
            *w -= mom_val;
        });
    }
}
