//! Standalone Cycle-Accurate Architectural Cloud TPU Hardware Emulator (`VirtualVfioGasketDevice`).
//!
//! Implements Phase 7.4:
//! - [`crate::hal::emulator::core_engine`]: Hardware binary bitstream execution engine (`ArchitecturalBitStreamEngine`)
//!   decoding and executing real generation-specific VLIW (`jfc`, `dfc`, `pfc`, `vfc`, `gfc`, `iwc`)
//!   and `SparseCore` V2/V3 bitstreams directly from virtual `SparseHbmStore`, plus multi-chip
//!   3D twisted-torus `ICI` RDMA fabric (`VirtualIciTorusFabric`).
//! - [`crate::hal::emulator::device`]: Standalone `PCIe` / `VFIO` (`0x3B`) / `Gasket` (`0xDC`) kernel device emulator
//!   (`VirtualVfioGasketDevice`) trapping BAR0/BAR2 MMIO doorbells, translating IOMMU DMA mappings,
//!   recording `XPlane` hardware trace rings in HBM, and signaling real OS `eventfd` descriptors.

/// Hardware binary VLIW & `SparseCore` bitstream engine and 3D torus `ICI` RDMA switch.
pub mod core_engine;
/// Virtual `PCIe` `VFIO`/`Gasket` device backend (`VirtualVfioGasketDevice`).
pub mod device;

pub use core_engine::{
    ArchitecturalBitStreamEngine, BitstreamExecutionSummary, HBM_CMEM_RODATA_BASE,
    HBM_IMEM_SPARSECORE_BASE, HBM_IMEM_TENSORCORE_BASE, HBM_XPLANE_TRACE_RING_BASE,
    HardwareTraceRingRecord, SparseHbmStore, VirtualIciTorusFabric,
};
pub use device::{
    BAR0_CSR_XPLANE_TRACE_CURSOR, BAR2_CMEM_LEN_CSR_OFFSET, BAR2_DMA_RING_OFFSET,
    BAR2_INPUT_ADDR_TABLE_OFFSET, BAR2_INPUT_COUNT_CSR_OFFSET, BAR2_INPUT_LEN_TABLE_OFFSET,
    BAR2_OUTPUT_ADDR_TABLE_OFFSET, BAR2_OUTPUT_COUNT_CSR_OFFSET, BAR2_OUTPUT_LEN_TABLE_OFFSET,
    BAR2_SC_IMEM_LEN_CSR_OFFSET, VirtualVfioGasketDevice, encode_device_hbm_iova,
    unpack_device_hbm_iova,
};
