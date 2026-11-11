//! Strongly-typed domain primitives for TPU devices, memory layouts, shapes, and topologies.
//!
//! All domain identifiers and sizes use strongly-typed newtypes powered by `derive_more`
//! (`Add`, `Display`, `From`, `Into`, `Sub`) to eliminate primitive obsession.
//!
//! Organized into focused semantic submodules:
//! - [`ids`]: Topology, runtime handle, collective, and `TensorCore` register identifiers.
//! - [`address`]: Memory addresses, offsets, byte/granule sizes, alignments, and `PCIe` / kernel primitives.
//! - [`hardware`]: [`TpuGeneration`], [`TpuCoreTypeEnum`], and 3D torus [`ChipCoordinates`].
//! - [`element`]: [`ElementType`] (`PJRT_Buffer_Type`) and low-precision (`f16`, `bf16`, `f8`, complex, sub-byte) scalar codecs.
//! - [`enums`]: Runtime, PJRT, DMA, collective, routing, compilation phase, and memory space enumerations.
//! - [`shape`]: [`MemoryLayout`], [`Shape`] (`TensorShape`) with TPU tile linearization, and [`TypedDeviceBuffer`].

pub mod address;
pub mod element;
pub mod enums;
pub mod hardware;
pub mod ids;
pub mod shape;

pub use self::address::{
    Alignment, AlignmentBytes, BarIndex, BarOffset, BdfAddress, BitWidth, BundlePc, ByteSize,
    CmemAddress, CmemByteOffset, CmemOffset, CsrAddress, DeviceAddress, DimensionIndex,
    DimensionSize, DmaBufFd, EventFdHandle, FileDescriptor, GranuleSize, HbmAddress, HbmByteOffset,
    HbmOffset, HostVirtualAddress, ImemAddress, ImemBundleIndex, ImemBundlePc, InterruptVector,
    IoctlCommand, IommuGroupId, IovaAddr, IovaAddress, LaneCount, MmioOffset, PageTableIndex,
    PciBdfAddress, PciBdfSlot, PciDeviceId, PciRevisionId, PciVendorId, SmOffset, SmemAddress,
    SmemByteOffset, SmemOffset, SpmemAddress, StructSize, SublaneCount, SublaneMask, VmOffset,
    VmemAddress, VmemByteOffset, VmemOffset,
};
pub use self::element::{Bf16Scalar, ElementType, F16Scalar};
pub use self::enums::{
    BufferState, CollectiveOpKind, CollectiveReductionOp, CompilationPhase, CoordinationMethodId,
    CoordinationOpcode, DmaDirection, DmaOpcode, EmbeddingEnginePhase, ExecutableState,
    HostBufferSemantics, HostCommandOpcode, IciRoutingStrategy, MegascaleTransportType, MemoryKind,
    MemoryLayoutType, MemorySpaceColor, MemorySpaceKind, MxuPrecisionMode, PjrtExtensionType,
    PjrtValueType, ProfilerState, ReductionKind, RoutingStrategy, ScalarAluOpcode, TilingKind,
    TilingSpec,
};
pub use self::hardware::{
    ChipCoordinates, CoreType, TileShape, TpuCoreType, TpuCoreTypeEnum, TpuGeneration,
    TpuVersionEnum,
};
pub use self::ids::{
    AbiVersionNumber, BarrierId, BlockId, BufferId, CallbackId, ChannelId, ChipId, CommunicatorId,
    CompilerId, ContextId, ContinuationId, CoreId, CoreIndex, DeviceId, DmaRingIndex, EventId,
    ExecutableId, GlobalChipId, GmrAccumId, GmrIndex, HostId, ImemOffset, IncarnationId, LaunchId,
    LocalHardwareId, LogicalSlotId, MatmulAccIndex, MemorySpaceId, OrdinalSeed, OverlayId,
    PartitionId, PhysicalMregId, PhysicalPregId, PhysicalSregId, PhysicalVregId, PregId, PregIndex,
    ProcessIndex, RegionId, ReplicaId, RequestId, RunId, SemaphoreId, SflagId, SliceId,
    SparseCoreId, SregId, SregIndex, SsaValueId, StreamId, SyncFlagId, SyncFlagIndex,
    SyncFlagValue, TraceMeContextId, TransferManagerId, TransformId, VirtualRegId, VirtualSlotId,
    VmaskBits, VmaskIndex, VmregIndex, VregId, VregIndex, WindowId,
};
pub use self::shape::{
    Allocated, DimVec, DynamicDimVec, MemoryLayout, Shape, TensorShape, TypedDeviceBuffer,
    Unallocated,
};
pub use crate::error::LibtpuSdkErrorCode;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_sdk_error_code_reexport() {
        for sdk_code_raw in [0_i32, 3, 5, 9, 13, 14] {
            let sdk_status = LibtpuSdkErrorCode::from(sdk_code_raw);
            assert_eq!(sdk_status.as_raw(), sdk_code_raw);
            assert_ne!(sdk_status.to_string(), "");
            assert_ne!(format!("{sdk_status:?}"), "");
        }
    }
}
