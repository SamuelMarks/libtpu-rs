//! Runtime, PJRT, DMA, collective, routing, compilation, and memory space enumerations.
//!
//! Defines strongly-typed enumerations for host commands ([`HostCommandOpcode`]), DMA directions
//! ([`DmaDirection`]), tiling specs ([`TilingKind`]), collective reductions and operations
//! ([`CollectiveReductionOp`], [`CollectiveOpKind`]), ICI routing ([`RoutingStrategy`]),
//! Megascale transports ([`MegascaleTransportType`]), compilation stages ([`CompilationPhase`]),
//! PJRT extensions/values/semantics ([`PjrtExtensionType`], [`PjrtValueType`], [`HostBufferSemantics`]),
//! lifecycle states ([`BufferState`], [`ExecutableState`], [`ProfilerState`], [`EmbeddingEnginePhase`]),
//! and device memory spaces ([`MemoryKind`]).

use crate::error::{LibTpuError, LibTpuResult};
use derive_more::Display;
use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Hardware-to-host command opcode (`tpu.HostCommandOpcodeProto` / `TN_QUEUE_CONTINUATION`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, IntoPrimitive, TryFromPrimitive)]
pub enum HostCommandOpcode {
    /// Unspecified / no-op host command (`0`).
    #[display("HOST_COMMAND_UNSPECIFIED")]
    Unspecified = 0,
    /// Invoke a registered Python or C++ host callback (`1`).
    #[display("HOST_CALLBACK")]
    HostCallback = 1,
    /// Dequeue next tensor batch from host infeed queue (`2`).
    #[display("INFEED_DEQUEUE")]
    InfeedDequeue = 2,
    /// Enqueue device tensor payload into host outfeed queue (`3`).
    #[display("OUTFEED_ENQUEUE")]
    OutfeedEnqueue = 3,
    /// Emit a device-side `tpu.trace` or debug print message (`4`).
    #[display("DEVICE_PRINT")]
    DevicePrint = 4,
    /// Signal a device-side assertion trap (`5`).
    #[display("DEVICE_ASSERT")]
    DeviceAssert = 5,
    /// Execute a cross-host barrier synchronization (`6`).
    #[display("RENDEZVOUS_BARRIER")]
    RendezvousBarrier = 6,
    /// Resolve dynamic slice / dynamic update slice offsets on host (`7`).
    #[display("DYNAMIC_SLICE_RESOLVE")]
    DynamicSliceResolve = 7,
    /// Resume `TN_QUEUE_CONTINUATION` after host offloading completion (`8`).
    #[display("QUEUE_CONTINUATION")]
    QueueContinuation = 8,
}

/// Hardware DMA transfer direction (`DmaDirection`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, IntoPrimitive, TryFromPrimitive)]
pub enum DmaDirection {
    /// Host RAM to Device HBM/VMEM (`0`).
    #[display("HostToDevice")]
    HostToDevice = 0,
    /// Device HBM/VMEM to Host RAM (`1`).
    #[display("DeviceToHost")]
    DeviceToHost = 1,
    /// Intra-chip Device-to-Device HBM/VMEM copy (`2`).
    #[display("DeviceToDevice")]
    DeviceToDevice = 2,
    /// Cross-chip Inter-Chip Interconnect (ICI) remote DMA (`3`).
    #[display("IciRemote")]
    IciRemote = 3,
}

/// Hardware tiling layout classification (`TilingKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum TilingKind {
    /// Linear untiled layout (`Linear`).
    #[display("Linear")]
    Linear,
    /// Standard 2D TPU `(8, 128)` tile (`Tile8x128`).
    #[display("Tile8x128")]
    Tile8x128,
    /// Sub-byte 3D TPU `(2, 4, 128)` tile (`Tile2x4x128`).
    #[display("Tile2x4x128")]
    Tile2x4x128,
    /// User-specified custom tile (`CustomTile`).
    #[display("CustomTile")]
    CustomTile,
}

/// Type alias for [`TilingKind`] (`TilingSpec`).
pub type TilingSpec = TilingKind;

/// Collective reduction operation kind (`CollectiveReductionOp`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum CollectiveReductionOp {
    /// Elementwise sum reduction (`0`).
    #[display("SUM")]
    Sum = 0,
    /// Elementwise product reduction (`1`).
    #[display("PRODUCT")]
    Product = 1,
    /// Elementwise minimum reduction (`2`).
    #[display("MIN")]
    Min = 2,
    /// Elementwise maximum reduction (`3`).
    #[display("MAX")]
    Max = 3,
}

/// Type alias for [`CollectiveReductionOp`] (`ReductionKind`).
pub type ReductionKind = CollectiveReductionOp;

impl CollectiveReductionOp {
    /// Converts a raw integer code into a [`CollectiveReductionOp`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw integer discriminant (`0..=3`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `raw` is outside `0..=3`.
    pub fn from_raw(raw: i32) -> LibTpuResult<Self> {
        match raw {
            0 => Ok(Self::Sum),
            1 => Ok(Self::Product),
            2 => Ok(Self::Min),
            3 => Ok(Self::Max),
            _ => Err(LibTpuError::InvalidArgument {
                parameter: "reduction_op",
                reason: format!("Unknown CollectiveReductionOp discriminant {raw}"),
            }),
        }
    }

    /// Returns the raw integer discriminant of this [`CollectiveReductionOp`].
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self as i32
    }
}

/// Memory layout discriminant (`PJRT_Buffer_MemoryLayout_Type`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum MemoryLayoutType {
    /// Tiled layout (`0`).
    #[display("Tiled")]
    Tiled = 0,
    /// Strides layout (`1`).
    #[display("Strides")]
    Strides = 1,
}

/// Host buffer ownership semantics (`PJRT_HostBufferSemantics`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum HostBufferSemantics {
    /// Host buffer is valid only during the `PJRT_Client_BufferFromHostBuffer` call (`0`).
    #[display("ImmutableOnlyDuringCall")]
    ImmutableOnlyDuringCall = 0,
    /// Host buffer remains valid until `done_with_host_buffer` event completes (`1`).
    #[display("ImmutableUntilTransferCompletes")]
    ImmutableUntilTransferCompletes = 1,
    /// Zero-copy immutable host buffer (`2`).
    #[display("ImmutableZeroCopy")]
    ImmutableZeroCopy = 2,
    /// Mutable zero-copy buffer (`3`).
    #[display("MutableZeroCopy")]
    MutableZeroCopy = 3,
}

/// PJRT extension type discriminant (`PJRT_Extension_Type`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum PjrtExtensionType {
    /// `PJRT_Profiler_Extension` (`1`).
    #[display("Profiler")]
    Profiler = 1,
    /// `PJRT_Layouts_Extension` (`6`).
    #[display("Layouts")]
    Layouts = 6,
    /// `PJRT_MemoryDescriptions_Extension` (`7`).
    #[display("MemoryDescriptions")]
    MemoryDescriptions = 7,
    /// `PJRT_RawBuffer_Extension` (`8`).
    #[display("RawBuffer")]
    RawBuffer = 8,
    /// `PJRT_PhaseCompile_Extension` (`9`).
    #[display("PhaseCompile")]
    PhaseCompile = 9,
    /// `PJRT_ExecutableMetadata_Extension` (`11`).
    #[display("ExecutableMetadata")]
    ExecutableMetadata = 11,
    /// `PJRT_CrossHostTransfers_Extension` (`12`).
    #[display("CrossHostTransfers")]
    CrossHostTransfers = 12,
    /// `PJRT_Callback_Extension` (`14`).
    #[display("Callback")]
    Callback = 14,
    /// `PJRT_TpuTopology_Extension` (`16`).
    #[display("TpuTopology")]
    TpuTopology = 16,
    /// `PJRT_TpuExecutable_Extension` (`17`).
    #[display("TpuExecutable")]
    TpuExecutable = 17,
    /// `PJRT_Megascale_Extension` (`18`).
    #[display("Megascale")]
    Megascale = 18,
    /// `PJRT_Shardings_Extension` (`19`).
    #[display("Shardings")]
    Shardings = 19,
    /// `PJRT_AbiVersion_Extension` (`20`).
    #[display("AbiVersion")]
    AbiVersion = 20,
    /// `PJRT_Collectives_Extension` (`25`).
    #[display("Collectives")]
    Collectives = 25,
    /// `PJRT_MultiSlice_Extension` (`26`).
    #[display("MultiSlice")]
    MultiSlice = 26,
    /// `PJRT_HostMemoryAllocator_Extension` (`28`).
    #[display("HostMemoryAllocator")]
    HostMemoryAllocator = 28,
    /// `PJRT_XlaTransform_Extension` (`29`).
    #[display("XlaTransform")]
    XlaTransform = 29,
}

/// PJRT named attribute value type (`PJRT_NamedValue_Type`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum PjrtValueType {
    /// UTF-8 string value (`0`).
    #[display("String")]
    String = 0,
    /// 64-bit signed integer (`1`).
    #[display("Int64")]
    Int64 = 1,
    /// 64-bit signed integer array (`2`).
    #[display("Int64List")]
    Int64List = 2,
    /// 32-bit IEEE float (`3`).
    #[display("Float")]
    Float = 3,
    /// Boolean flag (`4`).
    #[display("Bool")]
    Bool = 4,
}

/// Multi-device collective operation classification (`CollectiveOpKind`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum CollectiveOpKind {
    /// `AllReduce` across replicas/partitions (`0`).
    #[display("AllReduce")]
    AllReduce = 0,
    /// `ReduceScatter` across replicas/partitions (`1`).
    #[display("ReduceScatter")]
    ReduceScatter = 1,
    /// `AllGather` across replicas/partitions (`2`).
    #[display("AllGather")]
    AllGather = 2,
    /// `CollectivePermute` point-to-point permutation (`3`).
    #[display("CollectivePermute")]
    CollectivePermute = 3,
    /// `AllToAll` transpose exchange (`4`).
    #[display("AllToAll")]
    AllToAll = 4,
}

/// ICI torus routing strategy (`RoutingStrategy`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum RoutingStrategy {
    /// Deterministic dimension-ordered X-Y-Z routing (`0`).
    #[display("DimensionOrder")]
    DimensionOrder = 0,
    /// Congestion-aware adaptive routing (`1`).
    #[display("Adaptive")]
    Adaptive = 1,
    /// Minimal hop routing (`2`).
    #[display("Minimal")]
    Minimal = 2,
}

/// Type alias for [`RoutingStrategy`] (`IciRoutingStrategy`).
pub type IciRoutingStrategy = RoutingStrategy;

/// Megascale multi-slice transport layer (`MegascaleTransportType`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum MegascaleTransportType {
    /// Data Center Network TCP/IP (`0`).
    #[display("DcnTcp")]
    DcnTcp = 0,
    /// RDMA over Converged Ethernet v2 (`1`).
    #[display("RoceV2")]
    RoceV2 = 1,
    /// Direct Optical Inter-Chip Interconnect (`2`).
    #[display("IciOptical")]
    IciOptical = 2,
}

/// Phased compilation stage (`CompilationPhase`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum CompilationPhase {
    /// Stage 0: HLO verification (`0`).
    #[display("hlo_verification")]
    HloVerification = 0,
    /// Stage 1: TPU tile layout assignment (`1`).
    #[display("tpu_layout_assignment")]
    TpuLayoutAssignment = 1,
    /// Stage 2: HBM/VMEM memory planning (`2`).
    #[display("tpu_memory_planning")]
    TpuMemoryPlanning = 2,
    /// Stage 3: VLIW instruction bundle emission (`3`).
    #[display("vliw_bundle_emission")]
    VliwBundleEmission = 3,
    /// Stage 4: Binary executable serialization (`4`).
    #[display("binary_serialization")]
    BinarySerialization = 4,
}

/// Lifecycle state of a PJRT buffer (`BufferState`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum BufferState {
    /// Active HBM buffer allocation (`0`).
    #[display("Allocated")]
    Allocated = 0,
    /// Unfulfilled alias placeholder (`1`).
    #[display("AliasPending")]
    AliasPending = 1,
    /// Deleted or donated buffer (`2`).
    #[display("Deleted")]
    Deleted = 2,
}

/// Lifecycle state of a PJRT loaded executable (`ExecutableState`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum ExecutableState {
    /// Loaded and ready to execute (`0`).
    #[display("Loaded")]
    Loaded = 0,
    /// Deleted from device memory (`1`).
    #[display("Deleted")]
    Deleted = 1,
}

/// Lifecycle state of a TPU profiler session (`ProfilerState`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum ProfilerState {
    /// Inactive profiler (`0`).
    #[display("Idle")]
    Idle = 0,
    /// Actively recording trace events (`1`).
    #[display("Recording")]
    Recording = 1,
}

/// Lifecycle phase of the `SparseCore` TPU embedding engine (`EmbeddingEnginePhase`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum EmbeddingEnginePhase {
    /// Uninitialized engine (`0`).
    #[display("Uninitialized")]
    Uninitialized = 0,
    /// Partitioner executed (`1`).
    #[display("Partitioned")]
    Partitioned = 1,
    /// HBM memory configured and collated (`2`).
    #[display("MemoryConfigured")]
    MemoryConfigured = 2,
    /// Hosts connected and engine finalized (`3`).
    #[display("Finalized")]
    Finalized = 3,
}

/// TPU memory space classification (`PJRT_Memory_Kind`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, TryFromPrimitive, IntoPrimitive)]
pub enum MemoryKind {
    /// High-Bandwidth Memory (`"device"`, ID `0`).
    #[display("device")]
    DeviceHbm = 0,
    /// DMA-pinned host memory (`"pinned_host"`, ID `1`).
    #[display("pinned_host")]
    PinnedHost = 1,
    /// Unpinned host system memory (`"unpinned_host"`, ID `2`).
    #[display("unpinned_host")]
    UnpinnedHost = 2,
    /// On-chip Vector Memory (`"vmem"`, ID `3`).
    #[display("vmem")]
    Vmem = 3,
    /// On-chip Scalar Memory (`"smem"`, ID `4`).
    #[display("smem")]
    Smem = 4,
    /// On-chip Collective Memory (`"cmem"`, ID `5`).
    #[display("cmem")]
    Cmem = 5,
}

/// Type alias for [`MemoryKind`] (`MemorySpaceKind`).
pub type MemorySpaceKind = MemoryKind;

impl MemoryKind {
    /// Returns the static string identifier expected by PJRT (`PJRT_Memory_Kind`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeviceHbm => "device",
            Self::PinnedHost => "pinned_host",
            Self::UnpinnedHost => "unpinned_host",
            Self::Vmem => "vmem",
            Self::Smem => "smem",
            Self::Cmem => "cmem",
        }
    }

    /// Returns the numeric kind ID (`PJRT_Memory_Kind_Id`).
    #[must_use]
    pub const fn kind_id(self) -> i32 {
        self as i32
    }
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_and_memory_enums() {
        assert_eq!(ReductionKind::Sum.as_raw(), 0);
        assert_eq!(TilingSpec::Tile8x128, TilingKind::Tile8x128);
        assert_eq!(
            IciRoutingStrategy::DimensionOrder,
            RoutingStrategy::DimensionOrder
        );

        for tk in [
            TilingKind::Linear,
            TilingKind::Tile8x128,
            TilingKind::Tile2x4x128,
            TilingKind::CustomTile,
        ] {
            let tk_copy = tk;
            assert_eq!(tk, tk_copy);
            assert_ne!(tk.to_string(), "");
            assert_ne!(format!("{tk:?}"), "");
        }

        for raw in 0..=3 {
            let op = CollectiveReductionOp::from_raw(raw);
            assert!(op.is_ok());
            for o in op {
                assert_eq!(o.as_raw(), raw);
                assert_ne!(o.to_string(), "");
                assert_ne!(format!("{o:?}"), "");
            }
        }
        assert!(CollectiveReductionOp::from_raw(9).is_err());

        for hcmd_raw in 0..=8_i32 {
            let parsed = HostCommandOpcode::try_from(hcmd_raw);
            assert!(parsed.is_ok());
            for op in parsed {
                assert_eq!(i32::from(op), hcmd_raw);
                assert_ne!(op.to_string(), "");
                assert_ne!(format!("{op:?}"), "");
            }
        }
        assert!(HostCommandOpcode::try_from(999_i32).is_err());

        for raw_dir in 0..=3_u32 {
            let dir = DmaDirection::try_from(raw_dir);
            assert!(dir.is_ok());
            for d in dir {
                assert_eq!(u32::from(d), raw_dir);
                assert_ne!(d.to_string(), "");
                assert_ne!(format!("{d:?}"), "");
            }
        }
        assert!(DmaDirection::try_from(99_u32).is_err());

        let kinds = [
            (MemoryKind::DeviceHbm, "device", 0),
            (MemoryKind::PinnedHost, "pinned_host", 1),
            (MemoryKind::UnpinnedHost, "unpinned_host", 2),
            (MemoryKind::Vmem, "vmem", 3),
            (MemoryKind::Smem, "smem", 4),
            (MemoryKind::Cmem, "cmem", 5),
        ];
        for (k, s, id) in kinds {
            let alias_kind: MemorySpaceKind = k;
            assert_eq!(alias_kind.as_str(), s);
            assert_eq!(k.kind_id(), id);
            assert_eq!(i32::from(k), id);
            assert_eq!(MemoryKind::try_from(id).ok(), Some(k));
            assert_eq!(k.to_string(), s);
            assert_ne!(format!("{k:?}"), "");
        }

        let misc_enums = format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
            MemoryLayoutType::Tiled,
            MemoryLayoutType::Strides,
            HostBufferSemantics::ImmutableOnlyDuringCall,
            PjrtExtensionType::Profiler,
            PjrtValueType::String,
            CollectiveOpKind::AllReduce,
            RoutingStrategy::DimensionOrder,
            MegascaleTransportType::DcnTcp,
            CompilationPhase::HloVerification,
            BufferState::Allocated,
            ExecutableState::Loaded,
            ProfilerState::Idle,
            EmbeddingEnginePhase::Finalized,
            MemoryLayoutType::Tiled,
            HostBufferSemantics::MutableZeroCopy,
            PjrtExtensionType::XlaTransform,
            PjrtValueType::Bool,
            CollectiveOpKind::AllToAll,
            RoutingStrategy::Adaptive,
            MegascaleTransportType::IciOptical,
            CompilationPhase::BinarySerialization,
            BufferState::Deleted,
            ExecutableState::Deleted,
            ProfilerState::Recording,
        );
        assert_ne!(misc_enums, "");
    }
}
