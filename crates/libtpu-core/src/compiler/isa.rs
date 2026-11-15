//! Multi-generation `TensorCore` VLIW binary ISA codecs (`crates/libtpu-core/src/compiler/isa/`).
//!
//! Provides generation-exact binary encoders, decoders, `prost` protobuf lowerers, and disassemblers:
//! - [`JellyfishIsaCodec`] (`jfc`, Cloud TPU v2, `32` B / `256`-bit)
//! - [`DragonfishIsaCodec`] (`dfc`, Cloud TPU v3, `32` B / `256`-bit)
//! - [`PufferfishIsaCodec`] (`pfc`, Cloud TPU v4 & v4lite, `40` B / `320`-bit, `CMEM` & 3D ICI optical DMA)
//! - [`ViperfishIsaCodec`] (`vfc`, Cloud TPU v5e & v5p, `52` B / `416`-bit, native `INT8`/`FP8` MXU & stochastic rounding)
//! - [`GhostfishIsaCodec`] (`gfc`, Cloud TPU v6e Trillium, `64` B / `512`-bit, `256x256` MXU & `SparseCore` v2/v3 sync)
//! - [`IronwoodIsaCodec`] (`iwc`, Cloud TPU v7x Ironwood, `64` B / `512`-bit, `E8M0FNU`+`E2M1FN` microscaling & 192 GiB `HBM3e`)

pub mod codecs;
pub mod golden;

pub use self::codecs::{
    DragonfishIsaCodec, GenerationIsaCodec, GhostfishIsaCodec, IronwoodIsaCodec,
    IronwoodMicroscalingDescriptor, JellyfishIsaCodec, PufferfishIsaCodec,
    PufferfishOpticalDmaDescriptor, ViperfishIsaCodec, ViperfishMxuMode,
    compute_bundle_highway_tag8, decode_bundle_for_generation, decode_canonical_64b_bundle,
    encode_bundle_for_generation, encode_canonical_64b_bundle, format_generation_bundle_line,
    generation_bundle_byte_width,
};
pub use self::golden::{GoldenIsaParityReport, verify_all_golden_isa_bitstreams};
