//! `SparseCore` / `BarnaCore` V2 & V3 backend (`crates/libtpu-core/src/compiler/sparsecore/`).
//!
//! Organizes the complete Phase 6.3 `SparseCore` compilation and execution stack:
//! - [`codecs`]: [`SparseCoreV2Codec`] (`16` B) & [`SparseCoreV3Codec`] (`24` B) binary encoders, decoders, disassemblers, and `BarnaCore` / `ProgramWrapper` protobuf lowerers.
//! - [`spmem_coo_gather`]: [`SpmemScratchpadTiler`], [`CooCsrSorter`], and [`EmbeddingGatherDedupEngine`] (`Sum`, `Mean`, `Sqrtn` forward gather + backward gradient coalescing).
//! - [`optimizers`]: All 18+1 gradient optimizers (`SGD`, `Adagrad`, `AdagradMomentum`, `BoundedAdagrad`, `MdlAdagradLight`, `FrequencyAwareAdagrad`, `ProximalAdagrad`, `Adam`, `FTRL`, `Lion`, `Momentum`, `RMSProp`, `CenteredRMSProp`, `OnlineYogi`, `ProximalYogi`, `Assign`, `FrequencyEstimator`, `UserDefinedProgram`, and `Adadelta`) mapped 1:1 to `tensorflow.tpu.OptimizationParametersProto`.

pub mod codecs;
pub mod optimizers;
pub mod spmem_coo_gather;

pub use self::codecs::{
    SparseCoreBackend, SparseCoreOp, SparseCoreSubcoreKind, SparseCoreV2Codec,
    SparseCoreV2Instruction, SparseCoreV3Codec, SparseCoreV3Instruction, compute_sparsecore_tag8,
};
pub use self::optimizers::{SparseCoreGradientOptimizer, SparseCoreOptimizerConfig};
pub use self::spmem_coo_gather::{
    CooCsrSorter, CooSparseEntry, CsrSparseBatch, EmbeddingCombiner, EmbeddingGatherDedupEngine,
    SpmemScratchpadTiler,
};
