//! Multi-bank LLO register allocator (`LloRegisterAllocator`) across `SREG`, `VREG`, `MREG`,
//! `PREG`, and `SyncFlag` banks with live-range splitting and `VMEM`/`SMEM` spill-slot insertion.

use super::super::opcode::TpuOpCode;
use super::ir::{LloFunctionalUnit, LloIr, LloMicroOp, LloRegClass, LloRegOperand};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    LloLiveIntervalProto, LloOpcodeProto, LloRegisterAllocationProto, LloSpillSlotProto,
};
use crate::types::{ElementType, TpuGeneration};
use std::collections::BTreeMap;

/// Per-bank physical register capacities for [`LloRegisterAllocator`] (`LloBankCapacities`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LloBankCapacities {
    /// Physical scalar register capacity (`SREG`).
    pub sregs: u32,
    /// Physical vector register capacity (`VREG`).
    pub vregs: u32,
    /// Physical matrix accumulator capacity (`MREG`).
    pub mregs: u32,
    /// Physical predicate register capacity (`PREG`).
    pub pregs: u32,
    /// Physical `SyncFlag` counter capacity (`SFLAG`).
    pub sync_flags: u32,
}

impl LloBankCapacities {
    /// Returns default hardware register bank capacities for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn for_generation(generation: TpuGeneration) -> Self {
        Self {
            sregs: LloRegClass::Sreg.hardware_capacity(generation),
            vregs: LloRegClass::Vreg.hardware_capacity(generation),
            mregs: LloRegClass::Mreg.hardware_capacity(generation),
            pregs: LloRegClass::Preg.hardware_capacity(generation),
            sync_flags: LloRegClass::SyncFlag.hardware_capacity(generation),
        }
    }

    /// Returns the configured capacity for `class`.
    ///
    /// # Arguments
    ///
    /// * `class` - Target [`LloRegClass`].
    #[must_use]
    pub const fn capacity_for(&self, class: LloRegClass) -> u32 {
        match class {
            LloRegClass::Sreg => self.sregs,
            LloRegClass::Vreg => self.vregs,
            LloRegClass::Mreg => self.mregs,
            LloRegClass::Preg => self.pregs,
            LloRegClass::SyncFlag => self.sync_flags,
        }
    }
}

/// Multi-bank LLO linear-scan & live-range-splitting register allocator (`LloRegisterAllocator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LloRegisterAllocator {
    /// Target Cloud TPU hardware generation (`V2..=V7x`).
    pub generation: TpuGeneration,
    /// Physical register bank capacities.
    pub capacities: LloBankCapacities,
    /// Base `VMEM` scratchpad byte offset reserved for `VREG`/`MREG` spills.
    pub vmem_spill_base_bytes: u64,
    /// Base `SMEM` scratchpad byte offset reserved for `SREG`/`PREG`/`SyncFlag` spills.
    pub smem_spill_base_bytes: u64,
}

impl LloRegisterAllocator {
    /// Creates an [`LloRegisterAllocator`] using the architectural bank capacities of `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self {
            generation,
            capacities: LloBankCapacities::for_generation(generation),
            vmem_spill_base_bytes: 0x0008_0000,
            smem_spill_base_bytes: 0x0000_1000,
        }
    }

    /// Creates an [`LloRegisterAllocator`] with explicit `capacities` (useful for stress-testing spills).
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `capacities` - Explicit [`LloBankCapacities`].
    #[must_use]
    pub const fn with_capacities(generation: TpuGeneration, capacities: LloBankCapacities) -> Self {
        Self {
            generation,
            capacities,
            vmem_spill_base_bytes: 0x0008_0000,
            smem_spill_base_bytes: 0x0000_1000,
        }
    }

    /// Allocates physical registers across all basic blocks of `ir`, splitting live ranges and
    /// inserting `VMEM`/`SMEM` spill/reload micro-ops when pressure exceeds bank capacity.
    ///
    /// # Arguments
    ///
    /// * `ir` - Mutable [`LloIr`] module whose virtual registers are rewritten into physical registers.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::RegisterAllocationFailed`] if any bank capacity is zero.
    #[allow(clippy::too_many_lines)]
    pub fn allocate(&self, ir: &mut LloIr) -> LibTpuResult<LloRegisterAllocationProto> {
        for cls in [
            LloRegClass::Sreg,
            LloRegClass::Vreg,
            LloRegClass::Mreg,
            LloRegClass::Preg,
            LloRegClass::SyncFlag,
        ] {
            if self.capacities.capacity_for(cls) == 0 {
                return Err(LibTpuError::RegisterAllocationFailed {
                    reg_class: cls.bank_name(),
                    requested: 1,
                    capacity: 0,
                });
            }
        }

        // Flatten instruction PCs to compute exact [def_pc, last_use_pc] intervals per (class, vreg_id)
        let mut intervals: BTreeMap<(LloRegClass, u32), (u32, u32)> = BTreeMap::new();
        let mut global_pc = 0_u32;
        for block in &ir.blocks {
            for op in &block.ops {
                for operand in op.defs.iter().chain(op.uses.iter()) {
                    if let LloRegOperand::Virtual { id, class } = *operand {
                        intervals
                            .entry((class, id))
                            .and_modify(|(start, end)| {
                                *start = (*start).min(global_pc);
                                *end = (*end).max(global_pc);
                            })
                            .or_insert((global_pc, global_pc));
                    }
                }
                global_pc = global_pc.saturating_add(1);
            }
        }

        let mut assigned_map: BTreeMap<(LloRegClass, u32), u32> = BTreeMap::new();
        let mut spilled_map: BTreeMap<(LloRegClass, u32), LloSpillSlotProto> = BTreeMap::new();
        let mut live_interval_protos = Vec::with_capacity(intervals.len());
        let mut spill_slot_protos = Vec::new();
        let mut vmem_cursor = self.vmem_spill_base_bytes;
        let mut smem_cursor = self.smem_spill_base_bytes;
        let mut max_live_vregs = 0_u32;
        let mut peak_scalar_live = 0_u32;

        // Allocate each register bank independently using linear scan + live-range spill eviction
        for cls in [
            LloRegClass::Sreg,
            LloRegClass::Vreg,
            LloRegClass::Mreg,
            LloRegClass::Preg,
            LloRegClass::SyncFlag,
        ] {
            let bank_cap = self.capacities.capacity_for(cls);
            // Reserve highest physical register as dedicated scratch reload/spill register when bank_cap >= 2
            let allocatable_cap = if bank_cap >= 2 { bank_cap - 1 } else { 1 };
            let scratch_phys = bank_cap - 1;

            let mut bank_items: Vec<(u32, u32, u32)> = intervals
                .iter()
                .filter_map(|(&(c, vid), &(s, e))| if c == cls { Some((s, e, vid)) } else { None })
                .collect();
            bank_items.sort_by_key(|&(s, e, vid)| (s, e, vid));

            // Active list: (end_pc, phys_reg, vreg_id, start_pc)
            let mut active: Vec<(u32, u32, u32, u32)> = Vec::new();
            let mut free_regs: Vec<u32> = (0..allocatable_cap).rev().collect();

            for (start_pc, end_pc, vid) in bank_items {
                active.retain(|&(act_end, phys, _, _)| {
                    if act_end < start_pc {
                        free_regs.push(phys);
                        false
                    } else {
                        true
                    }
                });

                let (assigned_phys, was_spilled) = free_regs.pop().map_or_else(
                    || {
                        let (mem_space, byte_size) = cls.spill_layout();
                        let offset = if mem_space == 1 {
                            let o = vmem_cursor;
                            vmem_cursor = vmem_cursor.saturating_add(u64::from(byte_size));
                            o
                        } else {
                            let o = smem_cursor;
                            smem_cursor = smem_cursor.saturating_add(u64::from(byte_size));
                            o
                        };
                        let slot = LloSpillSlotProto {
                            virtual_reg_id: vid,
                            memory_space: mem_space,
                            byte_offset: offset,
                            byte_size,
                        };
                        spilled_map.insert((cls, vid), slot);
                        spill_slot_protos.push(slot);
                        (scratch_phys, true)
                    },
                    |phys| {
                        active.push((end_pc, phys, vid, start_pc));
                        (phys, false)
                    },
                );

                let live_now = u32::try_from(active.len())
                    .unwrap_or(0)
                    .max(u32::from(was_spilled));
                if cls == LloRegClass::Vreg {
                    max_live_vregs = max_live_vregs.max(live_now);
                } else if cls == LloRegClass::Sreg {
                    peak_scalar_live = peak_scalar_live.max(live_now);
                }

                assigned_map.insert((cls, vid), assigned_phys);
                live_interval_protos.push(LloLiveIntervalProto {
                    virtual_reg_id: vid,
                    reg_class: cls.class_code(),
                    def_pc: start_pc,
                    last_use_pc: end_pc,
                    assigned_phys_reg: assigned_phys,
                    spilled: was_spilled,
                });
            }
        }

        // Rewrite virtual operands into physical operands and splice reload/spill micro-ops around spilled intervals
        let mut next_splice_id = u32::try_from(ir.total_micro_ops()).unwrap_or(0) + 1000;
        for block in &mut ir.blocks {
            let mut rewritten = Vec::with_capacity(block.ops.len() + spill_slot_protos.len() * 2);
            for mut op in block.ops.drain(..) {
                let mut post_spills = Vec::new();
                for u in &mut op.uses {
                    if let LloRegOperand::Virtual { id, class } = *u {
                        let phys = assigned_map.get(&(class, id)).copied().unwrap_or(0);
                        let phys_op = LloRegOperand::physical_for_class(class, phys);
                        if let Some(slot) = spilled_map.get(&(class, id)) {
                            let (llo_ld, unit_ld) = if slot.memory_space == 1 {
                                (LloOpcodeProto::OpcodeVectorLoad, LloFunctionalUnit::Vector0)
                            } else {
                                (LloOpcodeProto::OpcodeScalarLoad, LloFunctionalUnit::Scalar0)
                            };
                            rewritten.push(LloMicroOp {
                                id: next_splice_id,
                                llo_opcode: llo_ld,
                                tpu_opcode: TpuOpCode::Identity,
                                unit: unit_ld,
                                element_type: ElementType::F32,
                                predicate: op.predicate,
                                negate_predicate: op.negate_predicate,
                                defs: vec![phys_op],
                                uses: vec![LloRegOperand::Immediate(slot.byte_offset)],
                                imm_value: slot.byte_offset,
                                latency_cycles: 2,
                                memory_space: u8::try_from(slot.memory_space).unwrap_or(1),
                                hlo_source_name: format!("reload_v{id}"),
                            });
                            next_splice_id = next_splice_id.saturating_add(1);
                        }
                        *u = phys_op;
                    }
                }

                for d in &mut op.defs {
                    if let LloRegOperand::Virtual { id, class } = *d {
                        let phys = assigned_map.get(&(class, id)).copied().unwrap_or(0);
                        let phys_op = LloRegOperand::physical_for_class(class, phys);
                        if let Some(slot) = spilled_map.get(&(class, id)) {
                            let (llo_st, unit_st) = if slot.memory_space == 1 {
                                (
                                    LloOpcodeProto::OpcodeVectorStore,
                                    LloFunctionalUnit::Vector0,
                                )
                            } else {
                                (
                                    LloOpcodeProto::OpcodeScalarStore,
                                    LloFunctionalUnit::Scalar0,
                                )
                            };
                            post_spills.push(LloMicroOp {
                                id: next_splice_id,
                                llo_opcode: llo_st,
                                tpu_opcode: TpuOpCode::Identity,
                                unit: unit_st,
                                element_type: ElementType::F32,
                                predicate: op.predicate,
                                negate_predicate: op.negate_predicate,
                                defs: Vec::new(),
                                uses: vec![phys_op, LloRegOperand::Immediate(slot.byte_offset)],
                                imm_value: slot.byte_offset,
                                latency_cycles: 1,
                                memory_space: u8::try_from(slot.memory_space).unwrap_or(1),
                                hlo_source_name: format!("spill_v{id}"),
                            });
                            next_splice_id = next_splice_id.saturating_add(1);
                        }
                        *d = phys_op;
                    }
                }

                rewritten.push(op);
                rewritten.extend(post_spills);
            }
            block.ops = rewritten;
        }

        Ok(LloRegisterAllocationProto {
            computation_name: ir.module_name.clone(),
            max_live_vregs,
            max_live_sregs: peak_scalar_live,
            live_intervals: live_interval_protos,
            spill_slots: spill_slot_protos,
        })
    }
}
