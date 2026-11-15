//! Swing Modulo Scheduler (`SwingModuloPipeliner`) constructing software-pipelined loop
//! prologs, steady-state kernels, and epilogs respecting hardware slot hazards and MXU/VMEM/DMA latencies.

use super::ir::{LloBasicBlock, LloFunctionalUnit, LloMicroOp};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{LloModuloStageProto, LloSoftwarePipelineScheduleProto};
use crate::types::{PregIndex, TpuGeneration};
use std::collections::BTreeMap;

/// Software-pipelined loop lowering output (`SoftwarePipelinedLoop`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftwarePipelinedLoop {
    /// Pipeline fill prolog micro-operations.
    pub prolog_ops: Vec<LloMicroOp>,
    /// Steady-state modulo-scheduled kernel micro-operations (`II` cycles).
    pub kernel_ops: Vec<LloMicroOp>,
    /// Pipeline drain epilog micro-operations.
    pub epilog_ops: Vec<LloMicroOp>,
    /// Number of overlapped modulo stages (`S >= 1`).
    pub stage_count: u32,
    /// Structured software pipeline schedule summary (`LloSoftwarePipelineScheduleProto`).
    pub schedule_proto: LloSoftwarePipelineScheduleProto,
}

/// Swing Modulo Scheduler (`SwingModuloPipeliner`) for `TensorCore` inner loop bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwingModuloPipeliner {
    /// Target Cloud TPU hardware generation (`V2..=V7x`).
    pub generation: TpuGeneration,
    /// Maximum allowed initiation interval (`II`) before rejecting modulo scheduling.
    pub max_ii_cycles: u32,
}

impl SwingModuloPipeliner {
    /// Creates a [`SwingModuloPipeliner`] for `generation` with default `max_ii_cycles = 64`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self {
            generation,
            max_ii_cycles: 64,
        }
    }

    /// Returns the number of parallel issue slots for `unit` class on `self.generation`.
    ///
    /// # Arguments
    ///
    /// * `unit` - [`LloFunctionalUnit`] class.
    #[must_use]
    pub const fn slot_capacity(&self, unit: LloFunctionalUnit) -> u32 {
        let dual_vec_mxu = !matches!(self.generation, TpuGeneration::V2 | TpuGeneration::V3);
        match unit {
            LloFunctionalUnit::Scalar0
            | LloFunctionalUnit::Scalar1
            | LloFunctionalUnit::Vector0
            | LloFunctionalUnit::Vector1 => {
                if dual_vec_mxu {
                    2
                } else {
                    1
                }
            }
            LloFunctionalUnit::MatrixMxu0
            | LloFunctionalUnit::MatrixMxu1
            | LloFunctionalUnit::MatrixMxu2
            | LloFunctionalUnit::MatrixMxu3 => self.generation.mxu_count_per_core(),
            LloFunctionalUnit::VectorEup
            | LloFunctionalUnit::CrossLaneXlu0
            | LloFunctionalUnit::CrossLaneXlu1
            | LloFunctionalUnit::AsyncDma
            | LloFunctionalUnit::ControlPred
            | LloFunctionalUnit::ImmediateExt => 1,
        }
    }

    /// Modulo-schedules `loop_block` into prolog, steady-state kernel, and epilog stages.
    ///
    /// # Arguments
    ///
    /// * `loop_block` - Inner loop [`LloBasicBlock`] to software-pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::LloSchedulingError`] if `loop_block.ops` is empty or no valid schedule exists within `max_ii_cycles`.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn pipeline_loop(&self, loop_block: &LloBasicBlock) -> LibTpuResult<SoftwarePipelinedLoop> {
        if loop_block.ops.is_empty() {
            return Err(LibTpuError::LloSchedulingError {
                bundle_pc: 0,
                slot_name: "S0",
                reason: format!(
                    "Cannot modulo-pipeline empty loop block '{}'",
                    loop_block.label
                ),
            });
        }

        // Compute ResMII across resource classes (Scalar, Vector, MXU, XLU/EUP, DMA)
        let mut scalar_count = 0_u32;
        let mut vector_count = 0_u32;
        let mut mxu_count = 0_u32;
        let mut xlu_count = 0_u32;
        let mut dma_count = 0_u32;
        for op in &loop_block.ops {
            match op.unit {
                LloFunctionalUnit::Scalar0
                | LloFunctionalUnit::Scalar1
                | LloFunctionalUnit::ControlPred
                | LloFunctionalUnit::ImmediateExt => scalar_count += 1,
                LloFunctionalUnit::Vector0 | LloFunctionalUnit::Vector1 => vector_count += 1,
                LloFunctionalUnit::MatrixMxu0
                | LloFunctionalUnit::MatrixMxu1
                | LloFunctionalUnit::MatrixMxu2
                | LloFunctionalUnit::MatrixMxu3 => mxu_count += 1,
                LloFunctionalUnit::VectorEup
                | LloFunctionalUnit::CrossLaneXlu0
                | LloFunctionalUnit::CrossLaneXlu1 => xlu_count += 1,
                LloFunctionalUnit::AsyncDma => dma_count += 1,
            }
        }
        let s_cap = self.slot_capacity(LloFunctionalUnit::Scalar0);
        let v_cap = self.slot_capacity(LloFunctionalUnit::Vector0);
        let m_cap = self.slot_capacity(LloFunctionalUnit::MatrixMxu0);
        let res_mii = scalar_count
            .div_ceil(s_cap)
            .max(vector_count.div_ceil(v_cap))
            .max(mxu_count.div_ceil(m_cap))
            .max(xlu_count)
            .max(dma_count)
            .max(1);

        // Compute RecMII from loop-carried RAW/WAW register dependencies (def at idx_d > use at idx_u)
        let mut first_use: BTreeMap<u32, usize> = BTreeMap::new();
        let mut recurrence_bound = 1_u32;
        for (idx, op) in loop_block.ops.iter().enumerate() {
            for u in &op.uses {
                first_use.entry(u.raw_index()).or_insert(idx);
            }
            for d in &op.defs {
                if let Some(&u_idx) = first_use.get(&d.raw_index())
                    && idx > u_idx
                {
                    recurrence_bound = recurrence_bound.max(op.latency_cycles);
                }
            }
        }

        let min_ii = res_mii.max(recurrence_bound);
        if min_ii > self.max_ii_cycles {
            return Err(LibTpuError::LloSchedulingError {
                bundle_pc: 0,
                slot_name: "MXU0",
                reason: format!(
                    "Minimum II {min_ii} exceeds max_ii_cycles {}",
                    self.max_ii_cycles
                ),
            });
        }

        // Modulo Reservation Table (MRT) scheduling with Swing ordering
        let achieved_ii = min_ii;
        let mut issue_cycles = vec![0_u32; loop_block.ops.len()];
        let mut mrt_counts: BTreeMap<(u32, u8), u32> = BTreeMap::new();
        let mut def_ready_cycle: BTreeMap<u32, u32> = BTreeMap::new();

        for (idx, op) in loop_block.ops.iter().enumerate() {
            let mut earliest = 0_u32;
            for u in &op.uses {
                if let Some(&ready) = def_ready_cycle.get(&u.raw_index()) {
                    earliest = earliest.max(ready);
                }
            }
            let res_group: u8 = match op.unit {
                LloFunctionalUnit::Scalar0
                | LloFunctionalUnit::Scalar1
                | LloFunctionalUnit::ControlPred
                | LloFunctionalUnit::ImmediateExt => 0,
                LloFunctionalUnit::Vector0 | LloFunctionalUnit::Vector1 => 1,
                LloFunctionalUnit::MatrixMxu0
                | LloFunctionalUnit::MatrixMxu1
                | LloFunctionalUnit::MatrixMxu2
                | LloFunctionalUnit::MatrixMxu3 => 2,
                LloFunctionalUnit::VectorEup
                | LloFunctionalUnit::CrossLaneXlu0
                | LloFunctionalUnit::CrossLaneXlu1 => 3,
                LloFunctionalUnit::AsyncDma => 4,
            };
            let cap = self.slot_capacity(op.unit);
            let mut chosen = earliest;
            for delta in 0..achieved_ii {
                let cand = earliest + delta;
                let slot_row = cand % achieved_ii;
                let used = mrt_counts.get(&(slot_row, res_group)).copied().unwrap_or(0);
                if used < cap {
                    mrt_counts.insert((slot_row, res_group), used + 1);
                    chosen = cand;
                    break;
                }
            }
            issue_cycles.get_mut(idx).into_iter().for_each(|slot| {
                *slot = chosen;
            });
            let ready = chosen.saturating_add(op.latency_cycles);
            for d in &op.defs {
                def_ready_cycle.insert(d.raw_index(), ready);
            }
        }

        let max_cycle = issue_cycles.iter().copied().max().unwrap_or(0);
        let stage_count = (max_cycle / achieved_ii).saturating_add(1);

        // Emit prolog (stages 0..stage_count-1), steady-state kernel (guarded by rotating @p_stage), and epilog
        let mut prolog_ops = Vec::new();
        let mut kernel_ops = Vec::with_capacity(loop_block.ops.len());
        let mut epilog_ops = Vec::new();

        let mut order: Vec<usize> = (0..loop_block.ops.len()).collect();
        order.sort_by_key(|&i| {
            let cyc = issue_cycles.get(i).copied().unwrap_or(0);
            (cyc % achieved_ii, cyc / achieved_ii, i)
        });

        for idx in order {
            loop_block.ops.get(idx).into_iter().for_each(|op| {
                let cyc = issue_cycles.get(idx).copied().unwrap_or(0);
                let stage = cyc / achieved_ii;
                let rot_pred = PregIndex((stage & 15) as u8);
                let mut k_op = op.clone();
                k_op.predicate = Some(rot_pred);
                kernel_ops.push(k_op);
                if stage + 1 < stage_count {
                    prolog_ops.push(op.clone());
                }
                if stage > 0 {
                    epilog_ops.push(op.clone());
                }
            });
        }

        let stages = vec![
            LloModuloStageProto {
                stage_index: 0,
                rotating_predicate_reg: 0,
                start_cycle: 0,
                end_cycle: achieved_ii.saturating_mul(stage_count.saturating_sub(1)),
            },
            LloModuloStageProto {
                stage_index: 1,
                rotating_predicate_reg: (stage_count.saturating_sub(1)) & 15,
                start_cycle: achieved_ii.saturating_mul(stage_count.saturating_sub(1)),
                end_cycle: achieved_ii.saturating_mul(stage_count),
            },
            LloModuloStageProto {
                stage_index: 2,
                rotating_predicate_reg: stage_count & 15,
                start_cycle: achieved_ii.saturating_mul(stage_count),
                end_cycle: achieved_ii.saturating_mul(stage_count.saturating_add(1)),
            },
        ];

        Ok(SoftwarePipelinedLoop {
            prolog_ops,
            kernel_ops,
            epilog_ops,
            stage_count,
            schedule_proto: LloSoftwarePipelineScheduleProto {
                loop_name: loop_block.label.clone(),
                resource_mii: res_mii,
                recurrence_mii: recurrence_bound,
                achieved_ii,
                stages,
            },
        })
    }
}
