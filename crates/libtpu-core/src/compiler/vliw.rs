//! Architectural 64-byte (`512`-bit) `TensorCore` VLIW instruction bundle layout and encoders.

use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use crate::error::{LibTpuError, LibTpuResult};
use bytemuck::Zeroable;

/// Binary 16-byte VLIW instruction word emitted by [`TpuInstructionBundle::encode_vliw_word`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VliwInstructionWord {
    /// Hardware [`TpuOpCode`] discriminant (`u16`).
    pub opcode: u16,
    /// Execution unit and transcendental/collective flags (`u16`).
    pub unit_flags: u16,
    /// Destination VMEM/HBM register slot index (`u32`).
    pub dst_reg: u32,
    /// First source operand register slot index (`u32`).
    pub src0_reg: u32,
    /// Second source operand register slot index or `SyncFlag` ID (`u32`).
    pub src1_reg: u32,
}

const _: () = assert!(size_of::<VliwInstructionWord>() == 16);

/// 8-byte header of an architectural 64-byte [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VliwBundleHeader {
    /// Program counter bundle index (`u32`).
    pub pc_offset: u32,
    /// Active functional unit slot bitmask (`u16`).
    pub bundle_flags: u16,
    /// Predicate execution mask (`u16`).
    pub pred_mask: u16,
}

/// 8-byte Scalar ALU (`S0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScalarAluSlot {
    /// Scalar opcode (`u8`).
    pub opcode: u8,
    /// Destination scalar register (`u8`).
    pub dst_sreg: u8,
    /// First source scalar register (`u8`).
    pub src0_sreg: u8,
    /// Second source scalar register (`u8`).
    pub src1_sreg: u8,
    /// 32-bit immediate value (`u32`).
    pub imm32: u32,
}

/// 20-byte Vector Processing Unit (`V0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorAluSlot {
    /// Vector ALU opcode (`u16`).
    pub opcode: u16,
    /// Element type code (`u8`).
    pub element_type: u8,
    /// Sublane mask (`u8`).
    pub lane_mask: u8,
    /// Destination VMEM register (`u32`).
    pub dst_vreg: u32,
    /// First source VMEM register (`u32`).
    pub src0_vreg: u32,
    /// Second source VMEM register (`u32`).
    pub src1_vreg: u32,
    /// Third source VMEM register or auxiliary parameter (`u32`).
    pub src2_vreg: u32,
}

/// 12-byte Matrix Multiply Unit (`M0`, `128x128` MXU systolic array) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatrixMxuSlot {
    /// MXU opcode (`u8`).
    pub opcode: u8,
    /// Precision mode (`0` = `BF16`, `1` = `F32`, `2` = `INT8`) (`u8`).
    pub precision_mode: u8,
    /// MXU control flags (`u16`).
    pub flags: u16,
    /// Accumulator/destination register (`u16`).
    pub acc_vreg: u16,
    /// Left-hand activation register (`u16`).
    pub lhs_vreg: u16,
    /// Right-hand weight register (`u16`).
    pub rhs_vreg: u16,
    /// Contracting tile count (`u16`).
    pub k_tiles: u16,
}

/// 8-byte Extended Unary / Transcendental (`X0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExtendedUnarySlot {
    /// Extended unary / cross-lane permute opcode (`u8`).
    pub opcode: u8,
    /// Permutation mode (`u8`).
    pub perm_mode: u8,
    /// Destination vector register (`u16`).
    pub dst_vreg: u16,
    /// Source vector register (`u16`).
    pub src_vreg: u16,
    /// 16-bit immediate parameter (`u16`).
    pub imm16: u16,
}

/// 8-byte Asynchronous DMA / ICI Collective (`D0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AsyncDmaSlot {
    /// DMA / collective opcode (`u8`).
    pub dma_opcode: u8,
    /// Source memory space (`0` = HBM, `1` = VMEM, `2` = SMEM, `3` = CMEM) (`u8`).
    pub src_space: u8,
    /// Destination memory space (`u8`).
    pub dst_space: u8,
    /// Collective reduction mode (`u8`).
    pub collective_op: u8,
    /// Architectural `SyncFlag` ID incremented upon completion (`u32`).
    pub sync_flag_id: u32,
}

/// Architectural 64-byte (`512`-bit) `TensorCore` VLIW instruction bundle (`TensorCoreVliwBundle`).
///
/// Combines parallel issue slots for the Scalar ALU (`S0`), Vector Processing Unit (`V0`),
/// Matrix Multiply Unit (`M0`), Extended Unary/Transcendental Unit (`X0`), and Async DMA/ICI Engine (`D0`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TensorCoreVliwBundle {
    /// 8-byte bundle header (`0x00..0x08`).
    pub header: VliwBundleHeader,
    /// 8-byte Scalar ALU slot (`0x08..0x10`).
    pub scalar_slot: ScalarAluSlot,
    /// 20-byte Vector ALU slot (`0x10..0x24`).
    pub vector_slot: VectorAluSlot,
    /// 12-byte Matrix MXU slot (`0x24..0x30`).
    pub matrix_slot: MatrixMxuSlot,
    /// 8-byte Extended Unary / Transcendental slot (`0x30..0x38`).
    pub xunary_slot: ExtendedUnarySlot,
    /// 8-byte Async DMA / Collective slot (`0x38..0x40`).
    pub dma_slot: AsyncDmaSlot,
}

const _: () = assert!(size_of::<TensorCoreVliwBundle>() == 64);

impl TensorCoreVliwBundle {
    /// Decodes a 64-byte architectural VLIW instruction bundle from raw `IMEM` bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - 64-byte raw `IMEM` instruction word.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the vector slot opcode byte is invalid.
    pub fn decode(bytes: &[u8; 64]) -> LibTpuResult<Self> {
        let bundle: Self = bytemuck::pod_read_unaligned(bytes);
        let raw_op = u8::try_from(bundle.vector_slot.opcode).map_err(|_| {
            LibTpuError::DeserializationFailed {
                reason: format!(
                    "Invalid VLIW vector slot opcode {}",
                    bundle.vector_slot.opcode
                ),
            }
        })?;
        let _ = TpuOpCode::from_byte(raw_op)?;
        Ok(bundle)
    }

    /// Returns the 5-bit active functional unit mask (`S0=1`, `M0=2`, `XLU=4`, `D0=8`, `V0=16`).
    #[must_use]
    pub const fn active_slot_mask(&self) -> u8 {
        #[allow(clippy::cast_possible_truncation)]
        let low = (self.header.bundle_flags & 0x1F) as u8;
        low
    }

    /// Returns the hardware stall cycle count (`0..=15`) encoded in bits `8..=11` of `header.bundle_flags`.
    #[must_use]
    pub const fn stall_cycles(&self) -> u8 {
        #[allow(clippy::cast_possible_truncation)]
        let stalls = ((self.header.bundle_flags >> 8) & 0x0F) as u8;
        stalls
    }

    /// Returns the target [`crate::types::TpuGeneration`] ordinal (`2..=7`) encoded in bits `12..=15` of `header.bundle_flags`.
    #[must_use]
    pub const fn generation_ordinal(&self) -> u8 {
        #[allow(clippy::cast_possible_truncation)]
        let gen_ord = ((self.header.bundle_flags >> 12) & 0x0F) as u8;
        if gen_ord == 0 { 5 } else { gen_ord }
    }

    /// Disassembles this 64-byte [`TensorCoreVliwBundle`] into a human-readable `DeepSea` LLO assembly mnemonic string.
    #[must_use]
    pub fn disassemble_vliw_bundle(&self) -> String {
        let pc = self.header.pc_offset;
        let stalls = self.stall_cycles();
        let gen_ord = self.generation_ordinal();
        let v_op = u8::try_from(self.vector_slot.opcode)
            .ok()
            .and_then(|b| TpuOpCode::from_byte(b).ok())
            .map_or_else(|| String::from("UNKNOWN"), |op| format!("{op:?}"));
        let s_str = format!(
            "S0(op={}, $s{} <- $s{}, $s{}, imm={})",
            self.scalar_slot.opcode,
            self.scalar_slot.dst_sreg,
            self.scalar_slot.src0_sreg,
            self.scalar_slot.src1_sreg,
            self.scalar_slot.imm32
        );
        let v_str = format!(
            "V0({v_op}, $v{} <- $v{}, $v{}, $v{})",
            self.vector_slot.dst_vreg,
            self.vector_slot.src0_vreg,
            self.vector_slot.src1_vreg,
            self.vector_slot.src2_vreg
        );
        let m_str = if self.matrix_slot.opcode != 0 {
            format!(
                " | M0(acc=$v{}, lhs=$v{}, rhs=$v{}, k={})",
                self.matrix_slot.acc_vreg,
                self.matrix_slot.lhs_vreg,
                self.matrix_slot.rhs_vreg,
                self.matrix_slot.k_tiles
            )
        } else {
            String::new()
        };
        let x_str = if self.xunary_slot.opcode != 0 {
            format!(
                " | XLU(op={}, $v{} <- $v{})",
                self.xunary_slot.opcode, self.xunary_slot.dst_vreg, self.xunary_slot.src_vreg
            )
        } else {
            String::new()
        };
        let d_str = if self.dma_slot.dma_opcode != 0 {
            format!(
                " | D0(op={}, flag={})",
                self.dma_slot.dma_opcode, self.dma_slot.sync_flag_id
            )
        } else {
            String::new()
        };
        format!(
            "[pc={pc:04} gen=v{gen_ord} stall={stalls}] {{ {s_str} | {v_str}{m_str}{x_str}{d_str} }}"
        )
    }

    /// Parses a mnemonic string emitted by [`Self::disassemble_vliw_bundle`] back into a
    /// 64-byte [`TensorCoreVliwBundle`], verifying round-trip assembler/disassembler fidelity.
    ///
    /// # Arguments
    ///
    /// * `mnemonic_line` - Single-line `DeepSea` LLO assembly string (e.g., `"[pc=0000 gen=v5 stall=0] { S0(...) | V0(...) }"`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if the assembly syntax or opcode mnemonic is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn assemble_vliw_bundle(mnemonic_line: &str) -> LibTpuResult<Self> {
        let trimmed = mnemonic_line.trim();
        let Some(rest_pc) = trimmed.strip_prefix("[pc=") else {
            return Err(LibTpuError::CompilationFailed {
                reason: format!("Missing '[pc=' header in VLIW assembly line: '{trimmed}'"),
            });
        };
        let extract_between = |haystack: &str, prefix: &str, suffix: &str| -> Option<String> {
            let after = haystack.split_once(prefix)?.1;
            Some(String::from(after.split_once(suffix)?.0.trim()))
        };
        let pc_offset = rest_pc
            .split_once(' ')
            .and_then(|(s, _)| s.parse::<u32>().ok())
            .unwrap_or(0);
        let gen_ord = extract_between(trimmed, "gen=v", " ")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(5)
            & 0x0F;
        let stall_cycles = extract_between(trimmed, "stall=", "]")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0)
            & 0x0F;

        let s0_body =
            extract_between(trimmed, "S0(", ")").ok_or_else(|| LibTpuError::CompilationFailed {
                reason: String::from("Missing S0(...) slot in VLIW assembly line"),
            })?;
        let s_op = extract_between(&s0_body, "op=", ",")
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(1);
        let scalar_dst = extract_between(&s0_body, "$s", "<-")
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(0);
        let after_arrow_s = s0_body.split_once("<-").map_or("", |(_, r)| r);
        let s_parts: Vec<&str> = after_arrow_s.split(',').map(str::trim).collect();
        let scalar_lhs = s_parts
            .first()
            .and_then(|s| s.strip_prefix("$s"))
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(0);
        let scalar_rhs = s_parts
            .get(1)
            .and_then(|s| s.strip_prefix("$s"))
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(0);
        let imm32 = s_parts
            .get(2)
            .and_then(|s| s.strip_prefix("imm="))
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(pc_offset);

        let v0_body =
            extract_between(trimmed, "V0(", ")").ok_or_else(|| LibTpuError::CompilationFailed {
                reason: String::from("Missing V0(...) slot in VLIW assembly line"),
            })?;
        let (v_op_name, v_regs_part) =
            v0_body
                .split_once(',')
                .ok_or_else(|| LibTpuError::CompilationFailed {
                    reason: String::from("Malformed V0(...) slot in VLIW assembly line"),
                })?;
        let resolved_op = (0_u8..=101_u8).find_map(|byte| {
            TpuOpCode::from_byte(byte)
                .ok()
                .filter(|candidate| format!("{candidate:?}") == v_op_name.trim())
        });
        let v_opcode = resolved_op.ok_or_else(|| LibTpuError::CompilationFailed {
            reason: format!("Unknown V0 opcode mnemonic '{}'", v_op_name.trim()),
        })?;
        let vec_out = extract_between(v_regs_part, "$v", "<-")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let after_arrow_v = v_regs_part.split_once("<-").map_or("", |(_, r)| r);
        let v_parts: Vec<&str> = after_arrow_v.split(',').map(str::trim).collect();
        let vec_arg_a = v_parts
            .first()
            .and_then(|s| s.strip_prefix("$v"))
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let vec_arg_b = v_parts
            .get(1)
            .and_then(|s| s.strip_prefix("$v"))
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let vec_arg_c = v_parts
            .get(2)
            .and_then(|s| s.strip_prefix("$v"))
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        let mut matrix_slot = MatrixMxuSlot::zeroed();
        let mut is_mxu = false;
        if let Some(m0_body) = extract_between(trimmed, "M0(", ")") {
            is_mxu = true;
            matrix_slot.opcode = 1;
            matrix_slot.flags = 1;
            matrix_slot.acc_vreg = extract_between(&m0_body, "acc=$v", ",")
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            matrix_slot.lhs_vreg = extract_between(&m0_body, "lhs=$v", ",")
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            matrix_slot.rhs_vreg = extract_between(&m0_body, "rhs=$v", ",")
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            matrix_slot.k_tiles = m0_body
                .rsplit_once("k=")
                .and_then(|(_, s)| s.trim().parse::<u16>().ok())
                .unwrap_or(1);
        }

        let mut xunary_slot = ExtendedUnarySlot::zeroed();
        let mut is_xunary = false;
        if let Some(xlu_body) = extract_between(trimmed, "XLU(", ")") {
            is_xunary = true;
            xunary_slot.opcode = extract_between(&xlu_body, "op=", ",")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(v_opcode as u8);
            xunary_slot.dst_vreg = extract_between(&xlu_body, "$v", "<-")
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            xunary_slot.src_vreg = xlu_body
                .rsplit_once("$v")
                .and_then(|(_, s)| s.trim().parse::<u16>().ok())
                .unwrap_or(0);
        }

        let mut dma_slot = AsyncDmaSlot::zeroed();
        let mut is_dma = false;
        if let Some(d0_body) = extract_between(trimmed, "D0(", ")") {
            is_dma = true;
            dma_slot.dma_opcode = extract_between(&d0_body, "op=", ",")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(1);
            dma_slot.src_space = 1;
            dma_slot.dst_space = 0;
            dma_slot.collective_op = v_opcode as u8;
            dma_slot.sync_flag_id = d0_body
                .rsplit_once("flag=")
                .and_then(|(_, s)| s.trim().parse::<u32>().ok())
                .unwrap_or(0);
        }

        let bundle_flags: u16 = 1
            | (u16::from(is_mxu) << 1)
            | (u16::from(is_xunary) << 2)
            | (u16::from(is_dma) << 3)
            | (1 << 4)
            | (stall_cycles << 8)
            | (gen_ord << 12);

        Ok(Self {
            header: VliwBundleHeader {
                pc_offset,
                bundle_flags,
                pred_mask: 0xFFFF,
            },
            scalar_slot: ScalarAluSlot {
                opcode: s_op,
                dst_sreg: scalar_dst,
                src0_sreg: scalar_lhs,
                src1_sreg: scalar_rhs,
                imm32,
            },
            vector_slot: VectorAluSlot {
                opcode: u16::from(v_opcode as u8),
                element_type: 11,
                lane_mask: 0xFF,
                dst_vreg: vec_out,
                src0_vreg: vec_arg_a,
                src1_vreg: vec_arg_b,
                src2_vreg: vec_arg_c,
            },
            matrix_slot,
            xunary_slot,
            dma_slot,
        })
    }

    /// Converts a logical [`TpuInstructionBundle`] into an architectural 64-byte [`TensorCoreVliwBundle`],
    /// validating that all register slot indices fit within the 8-bit physical register bank (`0..256`).
    ///
    /// # Arguments
    ///
    /// * `bundle` - Logical instruction bundle to encode.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::RegisterIndexOverflow`] if `output_slot` or any operand slot index is `>= 256`.
    pub fn from_ir_bundle(bundle: &TpuInstructionBundle) -> LibTpuResult<Self> {
        const MAX_PHYSICAL_REGISTERS: usize = 255;
        if bundle.output_slot > MAX_PHYSICAL_REGISTERS {
            return Err(LibTpuError::RegisterIndexOverflow {
                slot_index: bundle.output_slot,
                max_registers: MAX_PHYSICAL_REGISTERS,
            });
        }
        for &op_slot in &bundle.operand_slots {
            if op_slot > MAX_PHYSICAL_REGISTERS {
                return Err(LibTpuError::RegisterIndexOverflow {
                    slot_index: op_slot,
                    max_registers: MAX_PHYSICAL_REGISTERS,
                });
            }
        }
        Ok(bundle.encode_architectural_bundle())
    }

    /// Reconstructs a logical [`TpuInstructionBundle`] from this 64-byte [`TensorCoreVliwBundle`],
    /// preserving `vector_slot.element_type`, `matrix_slot`, `xunary_slot`, and `dma_slot` fields.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the functional unit slot opcode is unknown.
    pub fn to_instruction_bundle(&self) -> LibTpuResult<TpuInstructionBundle> {
        let raw_op = if self.matrix_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            TpuOpCode::DotGeneralF32 as u8
        } else if self.xunary_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            self.xunary_slot.opcode
        } else if self.dma_slot.collective_op != 0 && self.vector_slot.opcode == 0 {
            self.dma_slot.collective_op
        } else {
            u8::try_from(self.vector_slot.opcode).map_err(|_| {
                LibTpuError::DeserializationFailed {
                    reason: format!(
                        "Invalid VLIW vector slot opcode {}",
                        self.vector_slot.opcode
                    ),
                }
            })?
        };
        let opcode = TpuOpCode::from_byte(raw_op)?;
        let arity = opcode.arity();
        let mut operand_slots = Vec::with_capacity(arity);
        if self.matrix_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            operand_slots.push(usize::from(self.matrix_slot.lhs_vreg));
            operand_slots.push(usize::from(self.matrix_slot.rhs_vreg));
        } else if self.xunary_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            operand_slots.push(usize::from(self.xunary_slot.src_vreg));
        } else {
            if arity >= 1 {
                operand_slots.push(self.vector_slot.src0_vreg as usize);
            }
            if arity >= 2 {
                operand_slots.push(self.vector_slot.src1_vreg as usize);
            }
            if arity >= 3 {
                operand_slots.push(self.vector_slot.src2_vreg as usize);
            }
        }
        let (input_shapes, output_shape) = if self.vector_slot.element_type != 0 {
            let elem_type =
                crate::types::ElementType::from_raw(i32::from(self.vector_slot.element_type))
                    .unwrap_or(crate::types::ElementType::F32);
            let default_shape = crate::types::Shape {
                element_type: elem_type,
                dims: vec![4],
                dynamic_dims: vec![false],
                layout: crate::types::MemoryLayout::default_tpu_tiled_for_type(1, elem_type),
            };
            (
                vec![default_shape.clone(); operand_slots.len()],
                Some(default_shape),
            )
        } else {
            (Vec::new(), None)
        };
        let mut attrs = InstructionAttrs {
            iota_dim: usize::from(self.scalar_slot.src0_sreg),
            concat_dim: usize::from(self.scalar_slot.src1_sreg),
            input_shapes,
            output_shape,
            ..InstructionAttrs::default()
        };
        if self.xunary_slot.perm_mode != 0 {
            attrs
                .reduce_dims
                .push(usize::from(self.xunary_slot.perm_mode.saturating_sub(1)));
        }
        let output_slot = if self.matrix_slot.opcode != 0 && self.vector_slot.dst_vreg == 0 {
            usize::from(self.matrix_slot.acc_vreg)
        } else if self.xunary_slot.opcode != 0 && self.vector_slot.dst_vreg == 0 {
            usize::from(self.xunary_slot.dst_vreg)
        } else {
            self.vector_slot.dst_vreg as usize
        };
        Ok(TpuInstructionBundle {
            pc_offset: self.header.pc_offset,
            opcode,
            operand_slots,
            output_slot,
            sync_flag_id: self.dma_slot.sync_flag_id,
            attrs,
        })
    }
}

/// Cycle-accurate architectural register & functional-unit hazard scoreboard (`VliwScoreboard`).
///
/// Tracks per-register readiness cycles across `SREG[0..64]`, `PREG[0..16]`, `VREG[0..64]`,
/// `VMASK[0..8]`, `MXU0_ACCUM`, `MXU1_ACCUM`, and `SYNC_FLAGS[0..256]` to compute exact
/// RAW/WAW hazard stall counts during multi-issue VLIW bundle scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VliwScoreboard {
    /// Current schedule cycle counter.
    pub current_cycle: u32,
    /// Cycle at which each scalar register `SREG[0..64]` becomes available for reading.
    pub sreg_ready: [u32; 64],
    /// Cycle at which each predicate register `PREG[0..16]` becomes available for reading.
    pub preg_ready: [u32; 16],
    /// Cycle at which each vector register `VREG[0..64]` becomes available for reading.
    pub vreg_ready: [u32; 64],
    /// Cycle at which each vector mask register `VMASK[0..8]` becomes available for reading.
    pub vmask_ready: [u32; 8],
    /// Cycle at which `MXU0` and `MXU1` systolic accumulators become available.
    pub mxu_accum_ready: [u32; 2],
    /// Cycle at which each hardware `SyncFlag[0..256]` DMA barrier completes.
    pub sync_flag_ready: Vec<u32>,
}

impl Default for VliwScoreboard {
    fn default() -> Self {
        Self::new()
    }
}

impl VliwScoreboard {
    /// Creates a zero-initialized [`VliwScoreboard`] at cycle `0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_cycle: 0,
            sreg_ready: [0; 64],
            preg_ready: [0; 16],
            vreg_ready: [0; 64],
            vmask_ready: [0; 8],
            mxu_accum_ready: [0; 2],
            sync_flag_ready: vec![0; 256],
        }
    }

    /// Computes the number of stall cycles (`0..=15`) required before `bundle` can issue
    /// without RAW/WAW hazards, advances `self.current_cycle`, and records output register
    /// readiness using `latency`.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Logical [`TpuInstructionBundle`] being scheduled.
    /// * `latency` - Functional unit latency in cycles (`1..=16`).
    #[allow(clippy::indexing_slicing)]
    pub fn record_issue(&mut self, bundle: &TpuInstructionBundle, latency: u8) -> u8 {
        let mut required_cycle = self.current_cycle;
        for &src_slot in &bundle.operand_slots {
            required_cycle = required_cycle.max(self.vreg_ready[src_slot & 63]);
        }
        required_cycle = required_cycle.max(self.vreg_ready[bundle.output_slot & 63]);
        if bundle.opcode == TpuOpCode::DotGeneralF32 {
            required_cycle = required_cycle.max(self.mxu_accum_ready[0]);
        }
        if bundle.opcode.has_side_effects() {
            required_cycle =
                required_cycle.max(self.sync_flag_ready[(bundle.sync_flag_id as usize) & 255]);
        }
        let stall_u32 = required_cycle.saturating_sub(self.current_cycle).min(15);
        #[allow(clippy::cast_possible_truncation)]
        let stall_cycles = stall_u32 as u8;
        self.current_cycle = required_cycle.saturating_add(1);
        let ready_at = self.current_cycle.saturating_add(u32::from(latency));
        self.vreg_ready[bundle.output_slot & 63] = ready_at;
        self.sreg_ready[bundle.attrs.iota_dim & 63] = self.current_cycle.saturating_add(1);
        if bundle.opcode == TpuOpCode::DotGeneralF32 {
            self.mxu_accum_ready[0] = ready_at;
        }
        if bundle.opcode.has_side_effects() {
            self.sync_flag_ready[(bundle.sync_flag_id as usize) & 255] = ready_at;
        }
        stall_cycles
    }
}

/// Polyhedral loop tile and double-buffered VMEM DMA prefetch descriptor (`PolyhedralTileSchedule`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyhedralTileSchedule {
    /// Batch dimension tile factor (`>= 1`).
    pub batch_tiles: u32,
    /// Non-contracting row dimension (`M`) tile count in `MXU` sublanes (`128` or `256`).
    pub m_tiles: u32,
    /// Contracting dimension (`K`) tile count in `MXU` lanes (`128` or `256`).
    pub k_tiles: u32,
    /// Non-contracting column dimension (`N`) tile count in `MXU` lanes (`128` or `256`).
    pub n_tiles: u32,
    /// Ping-buffer byte offset in `VMEM` scratchpad.
    pub ping_vmem_offset: u64,
    /// Pong-buffer byte offset in `VMEM` scratchpad for overlapped `D0` DMA prefetching.
    pub pong_vmem_offset: u64,
    /// Architectural `SyncFlag` ID used to synchronize `HBM -> VMEM` prefetch completion.
    pub prefetch_sync_flag_id: u32,
}

impl PolyhedralTileSchedule {
    /// Computes the polyhedral tile schedule and double-buffered `VMEM` ping/pong offsets
    /// for a tensor shape on `generation`.
    ///
    /// # Arguments
    ///
    /// * `shape` - Logical tensor [`crate::types::Shape`] to tile.
    /// * `generation` - Target Cloud TPU generation (`MXU` `128x128` vs `256x256`).
    /// * `base_vmem_offset` - Starting byte offset in `VMEM`.
    /// * `sync_flag_id` - Base `SyncFlag` ID for DMA completion.
    #[must_use]
    pub fn from_shape(
        shape: &crate::types::Shape,
        generation: crate::types::TpuGeneration,
        base_vmem_offset: u64,
        sync_flag_id: u32,
    ) -> Self {
        let mxu_dim = u64::from(generation.mxu_dimension().max(128));
        let rank = shape.dims.len();
        let d_n = u64::try_from(shape.dims.last().copied().unwrap_or(1).max(1)).unwrap_or(1);
        let d_m = if rank >= 2 {
            u64::try_from(shape.dims.get(rank - 2).copied().unwrap_or(1).max(1)).unwrap_or(1)
        } else {
            8
        };
        let d_b: u64 = if rank >= 3 {
            shape.dims.iter().take(rank - 2).fold(1_u64, |acc, &d| {
                acc.saturating_mul(u64::try_from(d.max(1)).unwrap_or(1))
            })
        } else {
            1
        };
        let m_tiles = u32::try_from(d_m.div_ceil(8).max(1)).unwrap_or(1);
        let k_tiles = u32::try_from(d_n.div_ceil(mxu_dim).max(1)).unwrap_or(1);
        let n_tiles = u32::try_from(d_n.div_ceil(mxu_dim).max(1)).unwrap_or(1);
        let tile_bytes = (mxu_dim * 8 * 4).max(4096);
        Self {
            batch_tiles: u32::try_from(d_b).unwrap_or(1),
            m_tiles,
            k_tiles,
            n_tiles,
            ping_vmem_offset: base_vmem_offset,
            pong_vmem_offset: base_vmem_offset.saturating_add(tile_bytes),
            prefetch_sync_flag_id: sync_flag_id.max(1),
        }
    }
}

/// Latency-aware `DeepSea` VLIW instruction list scheduler (`LloVliwScheduler`).
///
/// Models functional unit latencies (Scalar = 1 cycle, Vector ALU = 2 cycles,
/// EUP/XLU = 4 cycles, MXU systolic array = 8 cycles, Async DMA = `SyncFlag` barrier)
/// and packs independent instructions across functional unit slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LloVliwScheduler {
    /// Target Cloud TPU hardware generation.
    pub generation: crate::types::TpuGeneration,
}

impl LloVliwScheduler {
    /// Creates a new [`LloVliwScheduler`] for the specified [`crate::types::TpuGeneration`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation (`V4`, `V5e`, `V5p`, `V6e`, `V7x`).
    #[must_use]
    pub const fn new(generation: crate::types::TpuGeneration) -> Self {
        Self { generation }
    }

    /// Returns the architectural execution latency in cycles for `opcode` on `self.generation`.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Hardware [`TpuOpCode`] to query.
    #[must_use]
    pub const fn instruction_latency_cycles(&self, opcode: TpuOpCode) -> u8 {
        if matches!(opcode, TpuOpCode::DotGeneralF32) {
            match self.generation {
                crate::types::TpuGeneration::V6e | crate::types::TpuGeneration::V7x => 12,
                _ => 8,
            }
        } else if opcode.is_transcendental() {
            4
        } else if opcode.has_side_effects() {
            6
        } else if matches!(opcode, TpuOpCode::Constant | TpuOpCode::Identity) {
            1
        } else {
            2
        }
    }

    /// Schedules a sequence of logical [`TpuInstructionBundle`]s into hazard-annotated,
    /// generation-parameterized 64-byte [`TensorCoreVliwBundle`]s using [`VliwScoreboard`].
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of logical instruction bundles to schedule.
    #[must_use]
    pub fn schedule_bundles(&self, bundles: &[TpuInstructionBundle]) -> Vec<TensorCoreVliwBundle> {
        let mut scoreboard = VliwScoreboard::new();
        let mut out = Vec::with_capacity(bundles.len());

        for bundle in bundles {
            let latency = self.instruction_latency_cycles(bundle.opcode);
            let stall_cycles = scoreboard.record_issue(bundle, latency);
            out.push(bundle.encode_for_generation(self.generation, stall_cycles));
        }

        out
    }

    /// Schedules and co-issues independent [`TpuInstructionBundle`]s across functional unit slots
    /// (`S0`, `V0`, `M0`, `XLU`, `D0`) into consolidated 64-byte [`TensorCoreVliwBundle`]s when
    /// no RAW or WAW register hazard exists between adjacent instructions.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of logical instruction bundles to schedule and pack.
    #[must_use]
    pub fn pack_independent_bundles(
        &self,
        bundles: &[TpuInstructionBundle],
    ) -> Vec<TensorCoreVliwBundle> {
        let scheduled = self.schedule_bundles(bundles);
        let mut packed = Vec::with_capacity(scheduled.len());
        let mut idx = 0_usize;
        while let Some(&first) = scheduled.get(idx) {
            let mut merged = first;
            if let Some(&second) = scheduled.get(idx + 1) {
                let first_dst = first.vector_slot.dst_vreg;
                let second_reads_first = second.vector_slot.src0_vreg == first_dst;
                let can_merge_mxu = second.matrix_slot.opcode != 0 && first.matrix_slot.opcode == 0;
                let can_merge_xlu = second.xunary_slot.opcode != 0 && first.xunary_slot.opcode == 0;
                let can_merge_dma =
                    second.dma_slot.dma_opcode != 0 && first.dma_slot.dma_opcode == 0;
                if !second_reads_first && (can_merge_mxu || can_merge_xlu || can_merge_dma) {
                    if can_merge_mxu {
                        merged.matrix_slot = second.matrix_slot;
                        merged.header.bundle_flags |= 1 << 1;
                    }
                    if can_merge_xlu {
                        merged.xunary_slot = second.xunary_slot;
                        merged.header.bundle_flags |= 1 << 2;
                    }
                    if can_merge_dma {
                        merged.dma_slot = second.dma_slot;
                        merged.header.bundle_flags |= 1 << 3;
                    }
                    idx += 2;
                    packed.push(merged);
                    continue;
                }
            }
            packed.push(merged);
            idx += 1;
        }
        packed
    }
}

impl TpuInstructionBundle {
    /// Encodes this [`TpuInstructionBundle`] into a 16-byte binary [`VliwInstructionWord`].
    #[must_use]
    pub fn encode_vliw_word(&self) -> VliwInstructionWord {
        let mut flags: u16 = 0;
        if self.opcode.is_binary() {
            flags |= 1 << 0;
        }
        if self.opcode.is_transcendental() {
            flags |= 1 << 1;
        }
        let s0 = self.operand_slots.first().copied().unwrap_or(0);
        let s1 = self
            .operand_slots
            .get(1)
            .copied()
            .unwrap_or(self.sync_flag_id as usize);
        VliwInstructionWord {
            opcode: u16::from(self.opcode as u8),
            unit_flags: flags,
            dst_reg: u32::try_from(self.output_slot).unwrap_or(0),
            src0_reg: u32::try_from(s0).unwrap_or(0),
            src1_reg: u32::try_from(s1).unwrap_or(0),
        }
    }

    /// Encodes this [`TpuInstructionBundle`] into a 64-byte architectural [`TensorCoreVliwBundle`].
    #[must_use]
    pub fn encode_architectural_bundle(&self) -> TensorCoreVliwBundle {
        let tpu_gen = crate::config::global_init_config().generation;
        self.encode_for_generation(tpu_gen, 0)
    }

    /// Encodes this [`TpuInstructionBundle`] into a 64-byte architectural [`TensorCoreVliwBundle`]
    /// parameterized by target [`crate::types::TpuGeneration`] and hardware `stall_cycles` (`0..=15`).
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation (`V4`, `V5e`, `V5p`, `V6e`, `V7x`).
    /// * `stall_cycles` - Number of inter-bundle hazard stall cycles (`0..=15`).
    #[must_use]
    pub fn encode_for_generation(
        &self,
        generation: crate::types::TpuGeneration,
        stall_cycles: u8,
    ) -> TensorCoreVliwBundle {
        let s0 = u32::try_from(self.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
        let s1 = u32::try_from(self.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
        let s2 = u32::try_from(self.operand_slots.get(2).copied().unwrap_or(0)).unwrap_or(0);
        let dst = u32::try_from(self.output_slot).unwrap_or(0);
        let et_raw = self.attrs.output_shape.as_ref().map_or(0_u8, |sh| {
            u8::try_from(sh.element_type.as_raw()).unwrap_or(11)
        });
        let iota_u8 = u8::try_from(self.attrs.iota_dim).unwrap_or(0);
        let concat_u8 = u8::try_from(self.attrs.concat_dim).unwrap_or(0);
        let red_mode = self
            .attrs
            .reduce_dims
            .first()
            .map_or(0_u8, |&d| u8::try_from(d.saturating_add(1)).unwrap_or(0));

        let is_mxu = self.opcode == TpuOpCode::DotGeneralF32;
        let is_xunary = self.opcode.is_transcendental();
        let is_collective = matches!(
            self.opcode,
            TpuOpCode::AllReduceSumF32 | TpuOpCode::AllGather | TpuOpCode::ReduceScatter
        );
        let gen_ord = u16::try_from(i32::from(generation)).unwrap_or(5) & 0x0F;
        let stalls_u16 = u16::from(stall_cycles & 0x0F);
        let bundle_flags: u16 = 1
            | (u16::from(is_mxu) << 1)
            | (u16::from(is_xunary) << 2)
            | (u16::from(is_collective) << 3)
            | (1 << 4)
            | (stalls_u16 << 8)
            | (gen_ord << 12);

        let k_tiles = match generation {
            crate::types::TpuGeneration::V6e | crate::types::TpuGeneration::V7x => 2,
            _ => 1,
        };

        TensorCoreVliwBundle {
            header: VliwBundleHeader {
                pc_offset: self.pc_offset,
                bundle_flags,
                pred_mask: 0xFFFF,
            },
            scalar_slot: ScalarAluSlot {
                opcode: 1,
                dst_sreg: 0,
                src0_sreg: iota_u8,
                src1_sreg: concat_u8,
                imm32: self.pc_offset,
            },
            vector_slot: VectorAluSlot {
                opcode: u16::from(self.opcode as u8),
                element_type: et_raw,
                lane_mask: 0xFF,
                dst_vreg: dst,
                src0_vreg: s0,
                src1_vreg: s1,
                src2_vreg: s2,
            },
            matrix_slot: MatrixMxuSlot {
                opcode: u8::from(is_mxu),
                precision_mode: 0,
                flags: u16::from(is_mxu),
                acc_vreg: u16::try_from(dst).unwrap_or(0),
                lhs_vreg: u16::try_from(s0).unwrap_or(0),
                rhs_vreg: u16::try_from(s1).unwrap_or(0),
                k_tiles,
            },
            xunary_slot: ExtendedUnarySlot {
                opcode: if is_xunary { self.opcode as u8 } else { 0 },
                perm_mode: red_mode,
                dst_vreg: u16::try_from(dst).unwrap_or(0),
                src_vreg: u16::try_from(s0).unwrap_or(0),
                imm16: 0,
            },
            dma_slot: AsyncDmaSlot {
                dma_opcode: u8::from(is_collective),
                src_space: 1,
                dst_space: 0,
                collective_op: if is_collective { self.opcode as u8 } else { 0 },
                sync_flag_id: self.sync_flag_id,
            },
        }
    }
}
