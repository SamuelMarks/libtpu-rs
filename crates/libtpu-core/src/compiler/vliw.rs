//! Architectural 64-byte (`512`-bit) `TensorCore` VLIW instruction bundle layout and encoders.

use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use crate::error::{LibTpuError, LibTpuResult};

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

    /// Reconstructs a logical [`TpuInstructionBundle`] from this 64-byte [`TensorCoreVliwBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the vector slot opcode is unknown.
    pub fn to_instruction_bundle(&self) -> LibTpuResult<TpuInstructionBundle> {
        let raw_op = u8::try_from(self.vector_slot.opcode).map_err(|_| {
            LibTpuError::DeserializationFailed {
                reason: format!(
                    "Invalid VLIW vector slot opcode {}",
                    self.vector_slot.opcode
                ),
            }
        })?;
        let opcode = TpuOpCode::from_byte(raw_op)?;
        let arity = opcode.arity();
        let mut operand_slots = Vec::with_capacity(arity);
        if arity >= 1 {
            operand_slots.push(self.vector_slot.src0_vreg as usize);
        }
        if arity >= 2 {
            operand_slots.push(self.vector_slot.src1_vreg as usize);
        }
        if arity >= 3 {
            operand_slots.push(self.vector_slot.src2_vreg as usize);
        }
        Ok(TpuInstructionBundle {
            pc_offset: self.header.pc_offset,
            opcode,
            operand_slots,
            output_slot: self.vector_slot.dst_vreg as usize,
            sync_flag_id: self.dma_slot.sync_flag_id,
            attrs: InstructionAttrs::default(),
        })
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
        let s0 = u32::try_from(self.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
        let s1 = u32::try_from(self.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
        let s2 = u32::try_from(self.operand_slots.get(2).copied().unwrap_or(0)).unwrap_or(0);
        let dst = u32::try_from(self.output_slot).unwrap_or(0);
        let et_raw = self.attrs.output_shape.as_ref().map_or(11_u8, |sh| {
            u8::try_from(sh.element_type.as_raw()).unwrap_or(11)
        });

        let is_mxu = self.opcode == TpuOpCode::DotGeneralF32;
        let is_xunary = self.opcode.is_transcendental();
        let is_collective = matches!(
            self.opcode,
            TpuOpCode::AllReduceSumF32 | TpuOpCode::AllGather | TpuOpCode::ReduceScatter
        );
        let bundle_flags: u16 = 1
            | (u16::from(is_mxu) << 1)
            | (u16::from(is_xunary) << 2)
            | (u16::from(is_collective) << 3);

        TensorCoreVliwBundle {
            header: VliwBundleHeader {
                pc_offset: self.pc_offset,
                bundle_flags,
                pred_mask: 0xFFFF,
            },
            scalar_slot: ScalarAluSlot {
                opcode: 1,
                dst_sreg: 0,
                src0_sreg: 0,
                src1_sreg: 0,
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
                k_tiles: 1,
            },
            xunary_slot: ExtendedUnarySlot {
                opcode: if is_xunary { self.opcode as u8 } else { 0 },
                perm_mode: 0,
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
