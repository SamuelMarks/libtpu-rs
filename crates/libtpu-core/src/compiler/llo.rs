//! Low-Level Optimizer (`LLO`) backend (`crates/libtpu-core/src/compiler/llo/`).
//!
//! Organizes the complete Phase 6.1 LLO pipeline:
//! - [`ir`]: Strongly-typed [`LloIr`], [`LloMicroOp`], [`LloBundle`], [`LloRegClass`], [`LloRegOperand`], and `LloModuleProto` bridge.
//! - [`isel`]: [`LloInstructionSelector`] lowering HLO & Mosaic `tpu` ops (`ThreeFry2x32` PRNG, EUP transcendentals, bitonic `sort`, `triangular_solve`, `cholesky`).
//! - [`pipeliner`]: [`SwingModuloPipeliner`] constructing software-pipelined prolog, steady-state kernel, and epilog blocks.
//! - [`regalloc`]: [`LloRegisterAllocator`] over `SREG`, `VREG`, `MREG`, `PREG`, and `SyncFlag` banks with `VMEM`/`SMEM` spill splicing.
//! - [`packer`]: [`LloVliwBundlePacker`] grouping independent micro-ops into multi-slot [`LloBundle`]s with NOP stall compression.

pub mod ir;
pub mod isel;
pub mod packer;
pub mod pipeliner;
pub mod regalloc;

pub use self::ir::{
    LloBasicBlock, LloBundle, LloFunctionalUnit, LloIr, LloMicroOp, LloRegClass, LloRegOperand,
};
pub use self::isel::{
    LloInstructionSelector, THREEFRY2X32_PARITY, THREEFRY2X32_ROTATIONS, eval_threefry2x32_counter,
};
pub use self::packer::{LloPackedProgram, LloVliwBundlePacker};
pub use self::pipeliner::{SoftwarePipelinedLoop, SwingModuloPipeliner};
pub use self::regalloc::{LloBankCapacities, LloRegisterAllocator};
