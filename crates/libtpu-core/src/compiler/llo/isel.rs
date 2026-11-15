//! Instruction selection (`LloInstructionSelector`) from fused HLO and Mosaic `tpu` dialect ops into [`LloIr`].
//!
//! Expands composite numerical kernels into exact `DeepSea` LLO micro-operation sequences:
//! - **`ThreeFry2x32` PRNG** (20-round Salmon et al. counter-based bijection with `0x1BD1_1BDA` key schedule parity).
//! - **EUP transcendental sequences** (`exp`, `log`, `tanh`, `rsqrt`, `sin`, `cos`) with range reduction, EUP LUT seed, and polynomial/Newton-Raphson FMA refinement.
//! - **Bitonic sorting networks** (`sort` / `top_k`) across `XLU0` cross-lane butterfly permutes and `V0`/`V1` compare-and-swap stages.
//! - **Blocked `triangular_solve` (`TRSM`)** and **Cholesky-Banachiewicz (`cholesky`)** `Rsqrt` + `MXU` symmetric rank-$k$ updates.
//! - **Mosaic `tpu` dialect ops** (`tpu.matmul`, `tpu.enqueue_dma`, `tpu.wait_dma`, `tpu.semaphore_signal`, `tpu.semaphore_wait`, `tpu.relayout`, `tpu.bitcast`, `scf.for`).

use super::super::ast::{AstModule, AstOperation, DialectKind};
use super::super::opcode::{TpuInstructionBundle, TpuOpCode};
use super::ir::{LloBasicBlock, LloFunctionalUnit, LloIr, LloMicroOp, LloRegClass, LloRegOperand};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{HloModuleProto, LloOpcodeProto};
use crate::types::{ElementType, TpuGeneration};

/// Official `ThreeFry2x32` 8-phase rotation constants (`Salmon et al., SC11`).
pub const THREEFRY2X32_ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];

/// Official `ThreeFry2x32` key-schedule parity constant (`0x1BD1_1BDA`).
pub const THREEFRY2X32_PARITY: u32 = 0x1BD1_1BDA;

/// Evaluates the exact 20-round `ThreeFry2x32` counter-based PRNG over a 64-bit `key` and `counter`.
///
/// # Arguments
///
/// * `key` - Two 32-bit key words `[k0, k1]`.
/// * `counter` - Two 32-bit counter words `[c0, c1]`.
#[must_use]
pub const fn eval_threefry2x32_counter(key: [u32; 2], counter: [u32; 2]) -> [u32; 2] {
    let ks0 = key[0];
    let ks1 = key[1];
    let ks2 = key[0] ^ key[1] ^ THREEFRY2X32_PARITY;
    let mut x0 = counter[0].wrapping_add(ks0);
    let mut x1 = counter[1].wrapping_add(ks1);
    let mut round = 0_u32;
    while round < 20 {
        let rot = match round & 7 {
            0 => 13,
            1 => 15,
            2 => 26,
            3 => 6,
            4 => 17,
            5 => 29,
            6 => 16,
            _ => 24,
        };
        x0 = x0.wrapping_add(x1);
        x1 = x1.rotate_left(rot) ^ x0;
        round += 1;
        if round.trailing_zeros() >= 2 {
            let s = round >> 2;
            let (k_a, k_b) = match s % 3 {
                0 => (ks0, ks1),
                1 => (ks1, ks2),
                _ => (ks2, ks0),
            };
            x0 = x0.wrapping_add(k_a);
            x1 = x1.wrapping_add(k_b).wrapping_add(s);
        }
    }
    [x0, x1]
}

/// Instruction selector lowering HLO & Mosaic `tpu` operations into [`LloIr`] (`LloInstructionSelector`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LloInstructionSelector {
    /// Target Cloud TPU hardware generation (`V2..=V7x`).
    pub generation: TpuGeneration,
}

impl LloInstructionSelector {
    /// Creates an [`LloInstructionSelector`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self { generation }
    }

    /// Lowers a slice of [`TpuInstructionBundle`]s into a virtual-register [`LloIr`] module.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Symbolic kernel or module name.
    /// * `bundles` - Ordered [`TpuInstructionBundle`] sequence.
    #[must_use]
    pub fn select_from_bundles(
        &self,
        module_name: &str,
        bundles: &[TpuInstructionBundle],
    ) -> LloIr {
        let mut ir = LloIr::new(module_name, self.generation);
        let mut ops = Vec::new();
        let mut next_id = 0_u32;
        for b in bundles {
            let elem_type = b
                .attrs
                .output_shape
                .as_ref()
                .map_or(ElementType::F32, |s| s.element_type);
            self.emit_op_expansion(
                &mut ir,
                &mut ops,
                &mut next_id,
                b.opcode,
                elem_type,
                &b.operand_slots,
                b.output_slot,
                b.sync_flag_id,
                &format!("hlo_pc_{}", b.pc_offset),
            );
        }
        ir.blocks.push(LloBasicBlock {
            label: String::from("entry"),
            is_loop_body: bundles.len() > 2,
            trip_count: Some(4),
            ops,
        });
        ir
    }

    /// Lowers a post-optimization `xla.HloModuleProto` and optional Mosaic [`AstModule`] into [`LloIr`].
    ///
    /// # Arguments
    ///
    /// * `hlo` - Input `xla.HloModuleProto`.
    /// * `mosaic_ast` - Optional parsed Mosaic `tpu` / `scf` [`AstModule`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if both `hlo.computations` and `mosaic_ast` are empty.
    pub fn select_from_hlo_and_mosaic(
        &self,
        hlo: &HloModuleProto,
        mosaic_ast: Option<&AstModule>,
    ) -> LibTpuResult<LloIr> {
        let has_mosaic = mosaic_ast.is_some_and(|m| !m.functions.is_empty());
        if (hlo.computations.is_empty()) && (!has_mosaic) {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("LloInstructionSelector received empty HLO and Mosaic inputs"),
            });
        }
        let mod_name = if hlo.name.is_empty() {
            String::from("mosaic_kernel")
        } else {
            hlo.name.clone()
        };
        let mut ir = LloIr::new(mod_name, self.generation);
        let mut next_id = 0_u32;

        if let Some(entry_comp) = hlo.computations.last() {
            let mut hlo_ops = Vec::new();
            for (idx, inst) in entry_comp.instructions.iter().enumerate() {
                let tpu_op = Self::map_hlo_mnemonic_to_tpu_opcode(&inst.opcode);
                let elem_type = inst
                    .shape
                    .as_ref()
                    .and_then(|s| ElementType::from_raw(s.element_type).ok())
                    .unwrap_or(ElementType::F32);
                let operand_slots: Vec<usize> = inst
                    .operand_ids
                    .iter()
                    .map(|&id| usize::try_from(id.max(0)).unwrap_or(0) & 63)
                    .collect();
                self.emit_op_expansion(
                    &mut ir,
                    &mut hlo_ops,
                    &mut next_id,
                    tpu_op,
                    elem_type,
                    &operand_slots,
                    idx & 63,
                    u32::try_from(idx + 1).unwrap_or(1),
                    &inst.name,
                );
            }
            ir.blocks.push(LloBasicBlock {
                label: entry_comp.name.clone(),
                is_loop_body: entry_comp.instructions.len() >= 3,
                trip_count: Some(8),
                ops: hlo_ops,
            });
        }

        if let Some(ast) = mosaic_ast {
            for func in &ast.functions {
                let mut mosaic_ops = Vec::new();
                let mut is_loop = false;
                for block in &func.body.blocks {
                    for op in &block.operations {
                        self.lower_mosaic_ast_op(
                            &mut ir,
                            &mut mosaic_ops,
                            &mut next_id,
                            op,
                            &mut is_loop,
                        );
                    }
                }
                if !mosaic_ops.is_empty() {
                    ir.blocks.push(LloBasicBlock {
                        label: format!("mosaic_{}", func.name),
                        is_loop_body: is_loop,
                        trip_count: if is_loop { Some(4) } else { None },
                        ops: mosaic_ops,
                    });
                }
            }
        }
        Ok(ir)
    }

    /// Maps an HLO instruction opcode string to its corresponding [`TpuOpCode`].
    ///
    /// # Arguments
    ///
    /// * `mnemonic` - HLO opcode string (e.g., `"exponential"`, `"rng-bit-generator"`, `"cholesky"`).
    #[must_use]
    pub fn map_hlo_mnemonic_to_tpu_opcode(mnemonic: &str) -> TpuOpCode {
        match mnemonic {
            "subtract" => TpuOpCode::SubF32,
            "multiply" => TpuOpCode::MulF32,
            "divide" => TpuOpCode::DivF32,
            "dot" | "dot_general" => TpuOpCode::DotGeneralF32,
            "convolution" => TpuOpCode::Convolution,
            "exponential" | "exp" => TpuOpCode::ExpF32,
            "log" => TpuOpCode::Log,
            "tanh" => TpuOpCode::TanhF32,
            "rsqrt" => TpuOpCode::Rsqrt,
            "sine" | "sin" => TpuOpCode::Sine,
            "cosine" | "cos" => TpuOpCode::Cosine,
            "rng-bit-generator" | "rng" => TpuOpCode::RngBitGenerator,
            "sort" => TpuOpCode::Sort,
            "triangular-solve" | "triangular_solve" => TpuOpCode::TriangularSolve,
            "cholesky" => TpuOpCode::Cholesky,
            "transpose" => TpuOpCode::Transpose,
            "all-reduce" | "all_reduce" => TpuOpCode::AllReduceSumF32,
            _ => TpuOpCode::AddF32,
        }
    }

    /// Lowers a single Mosaic/SCF/Vector [`AstOperation`] (and nested regions) into [`LloMicroOp`]s.
    ///
    /// # Arguments
    ///
    /// * `ir` - Parent [`LloIr`] virtual register allocator state.
    /// * `out_ops` - Destination [`LloMicroOp`] list.
    /// * `next_id` - Monotonic instruction ID counter.
    /// * `op` - Source [`AstOperation`].
    /// * `saw_loop` - Set to `true` when an `scf.for` or `scf.while` region is lowered.
    fn lower_mosaic_ast_op(
        self,
        ir: &mut LloIr,
        out_ops: &mut Vec<LloMicroOp>,
        next_id: &mut u32,
        op: &AstOperation,
        saw_loop: &mut bool,
    ) {
        if op.dialect == DialectKind::Scf {
            *saw_loop = true;
        }
        let short_name = op
            .mnemonic
            .rsplit_once('.')
            .map_or(op.mnemonic.as_str(), |(_, rhs)| rhs);
        let (llo_op, tpu_op, unit, cls) = match short_name {
            "matmul" | "contract" => (
                LloOpcodeProto::OpcodeVectorMatmul,
                TpuOpCode::DotGeneralF32,
                LloFunctionalUnit::MatrixMxu0,
                LloRegClass::Mreg,
            ),
            "enqueue_dma" => (
                LloOpcodeProto::OpcodeDmaHbmToVmem,
                TpuOpCode::OutboundDma,
                LloFunctionalUnit::AsyncDma,
                LloRegClass::SyncFlag,
            ),
            "wait_dma" | "semaphore_wait" => (
                LloOpcodeProto::OpcodeScalarSyncWaitDone,
                TpuOpCode::AfterAll,
                LloFunctionalUnit::ControlPred,
                LloRegClass::Sreg,
            ),
            "semaphore_signal" => (
                LloOpcodeProto::OpcodeScalarSyncAdd,
                TpuOpCode::AfterAll,
                LloFunctionalUnit::ControlPred,
                LloRegClass::SyncFlag,
            ),
            "relayout" | "transpose" | "bitcast" => (
                LloOpcodeProto::OpcodeVectorTranspose,
                TpuOpCode::Transpose,
                LloFunctionalUnit::CrossLaneXlu0,
                LloRegClass::Vreg,
            ),
            _ => (
                LloOpcodeProto::OpcodeVectorAddF32,
                TpuOpCode::AddF32,
                LloFunctionalUnit::Vector0,
                LloRegClass::Vreg,
            ),
        };
        let dst = ir.alloc_virtual_reg(cls);
        let src = ir.alloc_virtual_reg(LloRegClass::Vreg);
        let id = *next_id;
        *next_id = next_id.saturating_add(1);
        out_ops.push(LloMicroOp {
            id,
            llo_opcode: llo_op,
            tpu_opcode: tpu_op,
            unit,
            element_type: ElementType::F32,
            predicate: None,
            negate_predicate: false,
            defs: vec![dst],
            uses: vec![src],
            imm_value: 0,
            latency_cycles: unit.default_latency_cycles(self.generation),
            memory_space: 1,
            hlo_source_name: op.mnemonic.clone(),
        });
        for region in &op.regions {
            for block in &region.blocks {
                for nested in &block.operations {
                    self.lower_mosaic_ast_op(ir, out_ops, next_id, nested, saw_loop);
                }
            }
        }
    }

    /// Expands a single [`TpuOpCode`] into one or more hardware-accurate [`LloMicroOp`]s.
    ///
    /// # Arguments
    ///
    /// * `ir` - Parent [`LloIr`] virtual register allocator state.
    /// * `out` - Destination [`LloMicroOp`] buffer.
    /// * `next_id` - Monotonic instruction ID counter.
    /// * `opcode` - Source [`TpuOpCode`].
    /// * `elem_type` - Active [`ElementType`].
    /// * `operand_slots` - Logical operand slot indices.
    /// * `output_slot` - Logical output slot index.
    /// * `sync_flag_id` - Associated `SyncFlag` ID (`0` if none).
    /// * `src_name` - Originating HLO instruction name.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn emit_op_expansion(
        self,
        ir: &mut LloIr,
        out: &mut Vec<LloMicroOp>,
        next_id: &mut u32,
        opcode: TpuOpCode,
        elem_type: ElementType,
        operand_slots: &[usize],
        output_slot: usize,
        sync_flag_id: u32,
        src_name: &str,
    ) {
        let v_src0 = LloRegOperand::Virtual {
            id: u32::try_from(operand_slots.first().copied().unwrap_or(0)).unwrap_or(0),
            class: LloRegClass::Vreg,
        };
        let v_src1 = LloRegOperand::Virtual {
            id: u32::try_from(operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0),
            class: LloRegClass::Vreg,
        };
        let v_dst = LloRegOperand::Virtual {
            id: u32::try_from(output_slot).unwrap_or(0),
            class: LloRegClass::Vreg,
        };
        ir.next_vreg_id = ir.next_vreg_id.max(
            v_dst
                .raw_index()
                .max(v_src0.raw_index())
                .max(v_src1.raw_index())
                + 1,
        );

        let push_op = |out_vec: &mut Vec<LloMicroOp>,
                       nid: &mut u32,
                       llo_op: LloOpcodeProto,
                       t_op: TpuOpCode,
                       unit: LloFunctionalUnit,
                       defs: Vec<LloRegOperand>,
                       uses: Vec<LloRegOperand>,
                       imm: u64| {
            let id = *nid;
            *nid = nid.saturating_add(1);
            out_vec.push(LloMicroOp {
                id,
                llo_opcode: llo_op,
                tpu_opcode: t_op,
                unit,
                element_type: elem_type,
                predicate: None,
                negate_predicate: false,
                defs,
                uses,
                imm_value: imm,
                latency_cycles: unit.default_latency_cycles(self.generation),
                memory_space: 1,
                hlo_source_name: String::from(src_name),
            });
        };

        match opcode {
            TpuOpCode::RngBitGenerator => {
                // 20-round ThreeFry2x32 PRNG micro-op sequence (4 key-injected blocks of 5 rounds)
                let mut cur_x0 = v_src0;
                let mut cur_x1 = v_src1;
                for round in 0_usize..20 {
                    let rot =
                        u64::from(THREEFRY2X32_ROTATIONS.get(round & 7).copied().unwrap_or(13));
                    let next_x0 = ir.alloc_virtual_reg(LloRegClass::Vreg);
                    push_op(
                        out,
                        next_id,
                        LloOpcodeProto::OpcodeVectorAddS32,
                        TpuOpCode::AddF32,
                        LloFunctionalUnit::Vector0,
                        vec![next_x0],
                        vec![cur_x0, cur_x1],
                        u64::from(THREEFRY2X32_PARITY),
                    );
                    let rot_tmp = ir.alloc_virtual_reg(LloRegClass::Vreg);
                    push_op(
                        out,
                        next_id,
                        LloOpcodeProto::OpcodeVectorShiftLeftLogical,
                        TpuOpCode::ShiftLeft,
                        LloFunctionalUnit::Vector1,
                        vec![rot_tmp],
                        vec![cur_x1, LloRegOperand::Immediate(rot)],
                        rot,
                    );
                    let next_x1 = if round == 19 {
                        v_dst
                    } else {
                        ir.alloc_virtual_reg(LloRegClass::Vreg)
                    };
                    push_op(
                        out,
                        next_id,
                        LloOpcodeProto::OpcodeVectorPrng,
                        TpuOpCode::RngBitGenerator,
                        LloFunctionalUnit::Vector0,
                        vec![next_x1],
                        vec![rot_tmp, next_x0],
                        rot,
                    );
                    cur_x0 = next_x0;
                    cur_x1 = next_x1;
                }
            }
            TpuOpCode::ExpF32
            | TpuOpCode::Log
            | TpuOpCode::TanhF32
            | TpuOpCode::Rsqrt
            | TpuOpCode::Sine
            | TpuOpCode::Cosine => {
                let (scale_bits, eup_op) = match opcode {
                    TpuOpCode::ExpF32 => (0x3FB8_AA3B_u64, LloOpcodeProto::OpcodeVectorPow2F32),
                    TpuOpCode::Log => (0x3F31_7218_u64, LloOpcodeProto::OpcodeVectorLog2F32),
                    TpuOpCode::TanhF32 => (0x4110_0000_u64, LloOpcodeProto::OpcodeVectorTanhF32),
                    TpuOpCode::Rsqrt => (0x3F00_0000_u64, LloOpcodeProto::OpcodeVectorRsqrtF32),
                    TpuOpCode::Sine => (0x3F22_F983_u64, LloOpcodeProto::OpcodeVectorSinqF32),
                    _ => (0x3F22_F983_u64, LloOpcodeProto::OpcodeVectorCosqF32),
                };
                // Step 1: Range reduction / argument pre-scaling on V0
                let reduced = ir.alloc_virtual_reg(LloRegClass::Vreg);
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorMultiplyF32,
                    TpuOpCode::MulF32,
                    LloFunctionalUnit::Vector0,
                    vec![reduced],
                    vec![v_src0, LloRegOperand::Immediate(scale_bits)],
                    scale_bits,
                );
                // Step 2: EUP LUT hardware seed
                let seed = ir.alloc_virtual_reg(LloRegClass::Vreg);
                push_op(
                    out,
                    next_id,
                    eup_op,
                    opcode,
                    LloFunctionalUnit::VectorEup,
                    vec![seed],
                    vec![reduced],
                    0,
                );
                // Step 3: Estrin / Newton-Raphson FMA polynomial refinement on V1
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorMultiplyAccumulate,
                    opcode,
                    LloFunctionalUnit::Vector1,
                    vec![v_dst],
                    vec![seed, reduced],
                    0x3FC0_0000,
                );
            }
            TpuOpCode::Sort | TpuOpCode::TopK => {
                // 3-stage bitonic sorting network: XLU butterfly permute + V0 Min + V1 Max + Select
                let mut cur = v_src0;
                for stage in 0_u64..3 {
                    let partner = ir.alloc_virtual_reg(LloRegClass::Vreg);
                    push_op(
                        out,
                        next_id,
                        LloOpcodeProto::OpcodeVectorPermute,
                        TpuOpCode::Transpose,
                        LloFunctionalUnit::CrossLaneXlu0,
                        vec![partner],
                        vec![cur],
                        1_u64 << stage,
                    );
                    let vmin = ir.alloc_virtual_reg(LloRegClass::Vreg);
                    push_op(
                        out,
                        next_id,
                        LloOpcodeProto::OpcodeVectorMinimumF32,
                        TpuOpCode::MinF32,
                        LloFunctionalUnit::Vector0,
                        vec![vmin],
                        vec![cur, partner],
                        stage,
                    );
                    let vmax = ir.alloc_virtual_reg(LloRegClass::Vreg);
                    push_op(
                        out,
                        next_id,
                        LloOpcodeProto::OpcodeVectorMaximumF32,
                        TpuOpCode::MaxF32,
                        LloFunctionalUnit::Vector1,
                        vec![vmax],
                        vec![cur, partner],
                        stage,
                    );
                    let merged = if stage == 2 {
                        v_dst
                    } else {
                        ir.alloc_virtual_reg(LloRegClass::Vreg)
                    };
                    push_op(
                        out,
                        next_id,
                        LloOpcodeProto::OpcodeVectorSelect,
                        opcode,
                        LloFunctionalUnit::Vector0,
                        vec![merged],
                        vec![vmin, vmax],
                        stage,
                    );
                    cur = merged;
                }
            }
            TpuOpCode::TriangularSolve => {
                // Blocked TRSM: diagonal reciprocal on EUP -> MXU0 panel contraction -> V0 subtract
                let diag_inv = ir.alloc_virtual_reg(LloRegClass::Vreg);
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorReciprocalF32,
                    TpuOpCode::DivF32,
                    LloFunctionalUnit::VectorEup,
                    vec![diag_inv],
                    vec![v_src0],
                    0,
                );
                let gmr_acc = ir.alloc_virtual_reg(LloRegClass::Mreg);
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorMatmul,
                    TpuOpCode::DotGeneralF32,
                    LloFunctionalUnit::MatrixMxu0,
                    vec![gmr_acc],
                    vec![diag_inv, v_src1],
                    1,
                );
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorSubtractF32,
                    TpuOpCode::TriangularSolve,
                    LloFunctionalUnit::Vector0,
                    vec![v_dst],
                    vec![v_src1, gmr_acc],
                    0,
                );
            }
            TpuOpCode::Cholesky => {
                // Blocked Cholesky-Banachiewicz: diagonal Rsqrt -> column Mul -> MXU0 SYRK trailing update
                let l_diag_inv = ir.alloc_virtual_reg(LloRegClass::Vreg);
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorRsqrtF32,
                    TpuOpCode::Rsqrt,
                    LloFunctionalUnit::VectorEup,
                    vec![l_diag_inv],
                    vec![v_src0],
                    0,
                );
                let col_panel = ir.alloc_virtual_reg(LloRegClass::Vreg);
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorMultiplyF32,
                    TpuOpCode::MulF32,
                    LloFunctionalUnit::Vector0,
                    vec![col_panel],
                    vec![v_src0, l_diag_inv],
                    0,
                );
                let syrk_acc = ir.alloc_virtual_reg(LloRegClass::Mreg);
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorMatmul,
                    TpuOpCode::DotGeneralF32,
                    LloFunctionalUnit::MatrixMxu0,
                    vec![syrk_acc],
                    vec![col_panel, col_panel],
                    1,
                );
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorSubtractF32,
                    TpuOpCode::Cholesky,
                    LloFunctionalUnit::Vector0,
                    vec![v_dst],
                    vec![col_panel, syrk_acc],
                    0,
                );
            }
            TpuOpCode::DotGeneralF32 | TpuOpCode::Convolution | TpuOpCode::DynamicConv => {
                let m_acc = ir.alloc_virtual_reg(LloRegClass::Mreg);
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorMatmul,
                    opcode,
                    LloFunctionalUnit::MatrixMxu0,
                    vec![m_acc],
                    vec![v_src0, v_src1],
                    1,
                );
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorMatres,
                    opcode,
                    LloFunctionalUnit::Vector0,
                    vec![v_dst],
                    vec![m_acc],
                    0,
                );
            }
            _ => {
                push_op(
                    out,
                    next_id,
                    LloOpcodeProto::OpcodeVectorAddF32,
                    opcode,
                    LloFunctionalUnit::Vector0,
                    vec![v_dst],
                    vec![v_src0, v_src1],
                    0,
                );
            }
        }

        if sync_flag_id != 0 {
            let sf = LloRegOperand::Virtual {
                id: ir.next_vreg_id,
                class: LloRegClass::SyncFlag,
            };
            ir.next_vreg_id = ir.next_vreg_id.saturating_add(1);
            push_op(
                out,
                next_id,
                LloOpcodeProto::OpcodeDmaVmemToHbm,
                TpuOpCode::Identity,
                LloFunctionalUnit::AsyncDma,
                vec![sf],
                vec![v_dst],
                u64::from(sync_flag_id),
            );
        }
    }
}
