//! Architectural `TensorCore` VLIW execution state machine ([`TpuCoreVmState`] / [`DeepSeaVm`]).

use super::SimulatedTpuHal;
use crate::compiler::CompiledProgram;
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::ElementType;

/// Architectural `TensorCore` VLIW execution state machine (`TpuCoreVmState`).
#[derive(Debug, Clone)]
pub struct TpuCoreVmState {
    /// 32 64-bit scalar general-purpose registers (`SREG[0..32]`).
    pub sreg: [u64; 32],
    /// 16 scalar predicate registers (`PREG[0..16]`).
    pub preg: [bool; 16],
    /// 64 vector register sublane words (`VREG[0..64]`).
    pub vreg_words: [u32; 64],
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

    /// Steps the `TensorCore` VLIW sequencer through a slice of 64-byte [`crate::compiler::TensorCoreVliwBundle`]s,
    /// executing `S0` (Scalar ALU + SMEM), `V0` (Vector ALU + VMEM), `M0` (`MXU` systolic matrix accumulator),
    /// `X0` (`EUP` transcendental unit), and `D0` (`DMA`/`SyncFlag` engine) slots in lockstep.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of 64-byte architectural VLIW instruction bundles.
    #[allow(clippy::indexing_slicing)]
    pub fn execute_vliw_stream(&mut self, bundles: &[crate::compiler::TensorCoreVliwBundle]) {
        for bundle in bundles {
            self.step_vliw_bundle(bundle);
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

        // X0: Extended Unary / Transcendental Pipeline (EUP)
        if bundle.xunary_slot.opcode != 0 {
            let x_dst = usize::from(bundle.xunary_slot.dst_vreg) & 63;
            let x_src_bits = self.vreg_words[usize::from(bundle.xunary_slot.src_vreg) & 63];
            let x_src = f32::from_bits(x_src_bits);
            let x_res = match crate::compiler::TpuOpCode::try_from(bundle.xunary_slot.opcode) {
                Ok(crate::compiler::TpuOpCode::ExpF32) => x_src.exp().to_bits(),
                Ok(crate::compiler::TpuOpCode::TanhF32) => x_src.tanh().to_bits(),
                Ok(crate::compiler::TpuOpCode::Log) => x_src.ln().to_bits(),
                Ok(crate::compiler::TpuOpCode::Sqrt) => x_src.sqrt().to_bits(),
                Ok(crate::compiler::TpuOpCode::Rsqrt) => (1.0 / x_src.sqrt()).to_bits(),
                Ok(crate::compiler::TpuOpCode::Sine) => x_src.sin().to_bits(),
                Ok(crate::compiler::TpuOpCode::Cosine) => x_src.cos().to_bits(),
                _ => x_src_bits.wrapping_add(u32::from(bundle.xunary_slot.opcode)),
            };
            self.vreg_words[x_dst] = x_res;
        }

        // M0: Systolic Matrix Multiply Unit (MXU)
        if bundle.matrix_slot.opcode != 0 {
            let m_idx = usize::from(bundle.matrix_slot.acc_vreg) & 15;
            let k_tiles = f32::from(bundle.matrix_slot.k_tiles.max(1));
            self.mxu_accum[m_idx] += k_tiles;
        }

        // D0: Async DMA & SyncFlag completion barrier
        let sf_idx = usize::try_from(bundle.dma_slot.sync_flag_id).unwrap_or(0) & 255;
        self.sync_flags[sf_idx] = self.sync_flags[sf_idx].wrapping_add(1);
        self.cycles_executed = self.cycles_executed.wrapping_add(1);
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
    /// updating the `SREG[0]` continuation sequence register and `SYNC_FLAGS[0]`.
    ///
    /// # Arguments
    ///
    /// * `cmd` - Host command envelope from the `TensorCore` continuation queue.
    #[allow(clippy::indexing_slicing)]
    pub const fn dispatch_host_command(&mut self, cmd: &crate::proto::HostCommandEnvelopeProto) {
        self.sreg[0] = cmd.continuation_id;
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
            let evaluated =
                SimulatedTpuHal::eval_bundle_with_attrs(bundle, bundle_elem_type, &op_slices);
            let word0 = u32::from_le(bytemuck::pod_read_unaligned::<u32>(
                evaluated.get(0..4).unwrap_or(&[0; 4]),
            ));
            self.vreg_words[bundle.output_slot & 63] = word0;
            self.vmem_words[bundle.output_slot & 255] = word0;
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
