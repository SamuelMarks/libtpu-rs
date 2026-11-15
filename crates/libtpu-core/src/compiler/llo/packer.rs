//! Cycle-accurate LLO VLIW bundle packer (`LloVliwBundlePacker`).
//!
//! Groups independent physical-register-allocated [`LloMicroOp`]s per cycle across
//! `(S0, S1, V0, V1, MXU0..3, XLU0..1, DMA, IMM)` slots with NOP stall compression.

use super::super::vliw::{
    AsyncDmaSlot, ExtendedUnarySlot, MatrixMxuSlot, ScalarAluSlot, VectorAluSlot,
};
use super::ir::{LloBundle, LloFunctionalUnit, LloIr, LloMicroOp, LloRegOperand};
use crate::proto::{BundlePackerInfoProto, LloBundleScheduleProto, LloScheduledInstructionProto};
use crate::types::TpuGeneration;
use std::collections::BTreeMap;

/// Output of [`LloVliwBundlePacker::pack_ir`] (`LloPackedProgram`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LloPackedProgram {
    /// Packed hardware [`LloBundle`] sequence with compressed multi-cycle stall counts.
    pub bundles: Vec<LloBundle>,
    /// Cycle-by-cycle [`LloBundleScheduleProto`] summary.
    pub schedule_proto: LloBundleScheduleProto,
    /// Per-instruction [`BundlePackerInfoProto`] diagnostics.
    pub packer_infos: Vec<BundlePackerInfoProto>,
}

/// Cycle-accurate VLIW bundle packer (`LloVliwBundlePacker`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LloVliwBundlePacker {
    /// Target Cloud TPU hardware generation (`V2..=V7x`).
    pub generation: TpuGeneration,
}

impl LloVliwBundlePacker {
    /// Creates an [`LloVliwBundlePacker`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self { generation }
    }

    /// Computes a deterministic bank-qualified key for scoreboard hazard tracking.
    ///
    /// # Arguments
    ///
    /// * `operand` - Physical or virtual [`LloRegOperand`].
    #[allow(clippy::cast_possible_truncation)]
    const fn scoreboard_key(operand: LloRegOperand) -> Option<(u8, u32)> {
        match operand {
            LloRegOperand::Virtual { id, class } => Some((class.class_code() as u8, id)),
            LloRegOperand::PhysicalSreg(r) => Some((0, r.0 as u32)),
            LloRegOperand::PhysicalVreg(r) => Some((1, r.0 as u32)),
            LloRegOperand::PhysicalMreg(r) => Some((2, r.0 as u32)),
            LloRegOperand::PhysicalPred(r) => Some((3, r.0 as u32)),
            LloRegOperand::PhysicalSyncFlag(r) => Some((4, r.0)),
            LloRegOperand::Immediate(_) => None,
        }
    }

    /// Packs all [`LloMicroOp`]s in `ir` into VLIW [`LloBundle`]s with NOP stall compression (`0..=15` cycles/bundle).
    ///
    /// # Arguments
    ///
    /// * `ir` - Register-allocated [`LloIr`] module.
    #[must_use]
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn pack_ir(&self, ir: &LloIr) -> LloPackedProgram {
        let dual_slots = !matches!(self.generation, TpuGeneration::V2 | TpuGeneration::V3);
        let mut ready_cycle: BTreeMap<(u8, u32), u32> = BTreeMap::new();
        let mut cycle_bundles: BTreeMap<u32, LloBundle> = BTreeMap::new();
        let mut scheduled_insts = Vec::new();
        let mut packer_infos = Vec::new();
        let mut min_cycle_cursor = 0_u32;

        for block in &ir.blocks {
            for op in &block.ops {
                let mut earliest = min_cycle_cursor;
                for &u in &op.uses {
                    if let Some(key) = Self::scoreboard_key(u)
                        && let Some(&r_cyc) = ready_cycle.get(&key)
                    {
                        earliest = earliest.max(r_cyc);
                    }
                }
                for &d in &op.defs {
                    if let Some(key) = Self::scoreboard_key(d)
                        && let Some(&w_cyc) = ready_cycle.get(&key)
                    {
                        earliest = earliest.max(w_cyc);
                    }
                }

                let mut chosen_cycle = earliest;
                loop {
                    let entry = cycle_bundles
                        .entry(chosen_cycle)
                        .or_insert_with(|| LloBundle {
                            generation: self.generation,
                            ..LloBundle::default()
                        });
                    if Self::try_place_micro_op(entry, op, dual_slots) {
                        break;
                    }
                    chosen_cycle = chosen_cycle.saturating_add(1);
                }

                min_cycle_cursor = chosen_cycle;
                let finish_cycle = chosen_cycle.saturating_add(op.latency_cycles.max(1));
                for &d in &op.defs {
                    if let Some(key) = Self::scoreboard_key(d) {
                        ready_cycle.insert(key, finish_cycle);
                    }
                }

                scheduled_insts.push(LloScheduledInstructionProto {
                    bundle_pc: chosen_cycle,
                    functional_unit: op.unit.unit_code(),
                    opcode: op.llo_opcode as u32,
                    predicate_reg: op.predicate.map_or(0, |p| u32::from(p.0)),
                    stall_cycles: 0,
                    hlo_instruction_name: op.hlo_source_name.clone(),
                });
                packer_infos.push(BundlePackerInfoProto {
                    final_bundle_index: chosen_cycle,
                    operand_latency_dep_index: op.latency_cycles,
                    ..BundlePackerInfoProto::default()
                });
            }
        }

        // Compress multi-cycle gaps between occupied cycles into `stall_cycles` (0..=15 per bundle; emit NOP stall bundles for gaps > 15)
        let mut bundles = Vec::with_capacity(cycle_bundles.len().max(1));
        let ordered_cycles: Vec<(u32, LloBundle)> = cycle_bundles.into_iter().collect();
        for (idx, (cyc, mut b)) in ordered_cycles.iter().copied().enumerate() {
            let next_cyc = ordered_cycles.get(idx + 1).map_or(cyc + 1, |(nc, _)| *nc);
            let mut gap = next_cyc.saturating_sub(cyc + 1);
            b.stall_cycles = gap.min(15) as u8;
            bundles.push(b);
            gap = gap.saturating_sub(15);
            while gap > 0 {
                let chunk = gap.min(15) as u8;
                bundles.push(LloBundle {
                    generation: self.generation,
                    bundle_flags: 0x0001,
                    pred_mask: 0x0001,
                    stall_cycles: chunk,
                    ..LloBundle::default()
                });
                gap = gap.saturating_sub(u32::from(chunk));
            }
        }

        if bundles.is_empty() {
            bundles.push(LloBundle {
                generation: self.generation,
                bundle_flags: 0x0001,
                pred_mask: 0x0001,
                ..LloBundle::default()
            });
        }

        let total_cycles = ordered_cycles
            .last()
            .map_or(1, |(c, _)| c.saturating_add(1));

        LloPackedProgram {
            bundles,
            schedule_proto: LloBundleScheduleProto {
                schedule_name: ir.module_name.clone(),
                total_cycles,
                initiation_interval_mii: 1,
                scheduled_instructions: scheduled_insts,
            },
            packer_infos,
        }
    }

    /// Attempts to place `op` into an available hardware slot of `bundle`, returning `true` on success.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Target cycle [`LloBundle`].
    /// * `op` - [`LloMicroOp`] to place.
    /// * `dual_slots` - Whether secondary co-issue slots (`S1`, `V1`, `M1`, `X1`) are available.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn try_place_micro_op(bundle: &mut LloBundle, op: &LloMicroOp, dual_slots: bool) -> bool {
        let dst = op.defs.first().map_or(0, |d| d.raw_index());
        let src0 = op.uses.first().map_or(0, |u| u.raw_index());
        let src1 = op.uses.get(1).map_or(src0, |u| u.raw_index());
        let tpu_byte = op.tpu_opcode as u8;
        let et_u8 = u8::try_from(op.element_type.as_raw()).unwrap_or(11);

        if let Some(pred) = op.predicate {
            bundle.pred_mask |= 1_u16 << (pred.0 & 15);
        }

        match op.unit {
            LloFunctionalUnit::Scalar0
            | LloFunctionalUnit::Scalar1
            | LloFunctionalUnit::ControlPred
            | LloFunctionalUnit::ImmediateExt => {
                if (bundle.bundle_flags & 0x0001) == 0 {
                    bundle.scalar_slots[0] = ScalarAluSlot {
                        opcode: 1,
                        dst_sreg: (dst & 0x3F) as u8,
                        src0_sreg: (src0 & 0x3F) as u8,
                        src1_sreg: (src1 & 0x3F) as u8,
                        imm32: op.imm_value as u32,
                    };
                    bundle.immediates[0] = op.imm_value as u32;
                    bundle.bundle_flags |= 0x0001;
                    true
                } else if dual_slots && ((bundle.bundle_flags & (1 << 5)) == 0) {
                    bundle.scalar_slots[1] = ScalarAluSlot {
                        opcode: 1,
                        dst_sreg: (dst & 0x3F) as u8,
                        src0_sreg: (src0 & 0x3F) as u8,
                        src1_sreg: (src1 & 0x3F) as u8,
                        imm32: op.imm_value as u32,
                    };
                    bundle.immediates[1] = op.imm_value as u32;
                    bundle.bundle_flags |= 1 << 5;
                    true
                } else {
                    false
                }
            }
            LloFunctionalUnit::Vector0 | LloFunctionalUnit::Vector1 => {
                if (bundle.bundle_flags & 0x0002) == 0 {
                    bundle.vector_slots[0] = VectorAluSlot {
                        opcode: u16::from(tpu_byte),
                        element_type: et_u8,
                        lane_mask: 0xFF,
                        dst_vreg: dst,
                        src0_vreg: src0,
                        src1_vreg: src1,
                        src2_vreg: 0,
                    };
                    bundle.bundle_flags |= 0x0002;
                    true
                } else if dual_slots && ((bundle.bundle_flags & (1 << 6)) == 0) {
                    bundle.vector_slots[1] = VectorAluSlot {
                        opcode: u16::from(tpu_byte),
                        element_type: et_u8,
                        lane_mask: 0xFF,
                        dst_vreg: dst,
                        src0_vreg: src0,
                        src1_vreg: src1,
                        src2_vreg: 0,
                    };
                    bundle.bundle_flags |= 1 << 6;
                    true
                } else {
                    false
                }
            }
            LloFunctionalUnit::MatrixMxu0
            | LloFunctionalUnit::MatrixMxu1
            | LloFunctionalUnit::MatrixMxu2
            | LloFunctionalUnit::MatrixMxu3 => {
                if (bundle.bundle_flags & 0x0004) == 0 {
                    bundle.matrix_slots[0] = MatrixMxuSlot {
                        opcode: tpu_byte,
                        precision_mode: 1,
                        flags: 1,
                        acc_vreg: dst as u16,
                        lhs_vreg: src0 as u16,
                        rhs_vreg: src1 as u16,
                        k_tiles: 1,
                    };
                    bundle.bundle_flags |= 0x0004;
                    true
                } else if dual_slots && ((bundle.bundle_flags & (1 << 7)) == 0) {
                    bundle.matrix_slots[1] = MatrixMxuSlot {
                        opcode: tpu_byte,
                        precision_mode: 1,
                        flags: 1,
                        acc_vreg: (dst.max(1)) as u16,
                        lhs_vreg: src0 as u16,
                        rhs_vreg: src1 as u16,
                        k_tiles: 1,
                    };
                    bundle.bundle_flags |= 1 << 7;
                    true
                } else {
                    false
                }
            }
            LloFunctionalUnit::VectorEup
            | LloFunctionalUnit::CrossLaneXlu0
            | LloFunctionalUnit::CrossLaneXlu1 => {
                if (bundle.bundle_flags & 0x0008) == 0 {
                    bundle.xlu_slots[0] = ExtendedUnarySlot {
                        opcode: tpu_byte,
                        perm_mode: (op.imm_value & 0x07) as u8,
                        dst_vreg: dst as u16,
                        src_vreg: src0 as u16,
                        imm16: op.imm_value as u16,
                    };
                    bundle.bundle_flags |= 0x0008;
                    true
                } else {
                    false
                }
            }
            LloFunctionalUnit::AsyncDma => {
                if (bundle.bundle_flags & 0x0010) == 0 {
                    bundle.dma_slot = AsyncDmaSlot {
                        dma_opcode: 1,
                        src_space: op.memory_space,
                        dst_space: 0,
                        collective_op: 0,
                        sync_flag_id: (op.imm_value as u32).max(1),
                    };
                    bundle.bundle_flags |= 0x0010;
                    true
                } else {
                    false
                }
            }
        }
    }
}
