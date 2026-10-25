//! Strongly-typed domain primitives for TPU devices, memory layouts, shapes, and topologies.
//!
//! All domain identifiers and sizes use strongly-typed newtypes powered by `derive_more`
//! (`Add`, `Display`, `From`, `Into`, `Sub`) to eliminate primitive obsession.

use crate::error::{LibTpuError, LibTpuResult};
use derive_more::{Add, Display, From, Into, Sub};
use std::marker::PhantomData;

/// Unique identifier for a logical TPU device (`PJRT_Device_Id`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct DeviceId(pub u32);

/// Unique identifier for a physical TPU chip (`ChipId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ChipId(pub u32);

/// Unique identifier for a `TensorCore` or `SparseCore` on a chip (`CoreId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CoreId(pub u32);

/// Zero-based index of a core on a single physical TPU chip (`CoreIndex`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CoreIndex(pub u8);

/// Unique identifier for a host machine in a TPU pod slice (`HostId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct HostId(pub u32);

/// Process index of a PJRT client in a multi-host TPU pod (`ProcessIndex`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ProcessIndex(pub i32);

/// Identifier for a TPU pod slice in a multi-slice Megascale workload (`SliceId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct SliceId(pub u32);

/// Replica identifier in a data-parallel computation (`ReplicaId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ReplicaId(pub u32);

/// Partition identifier in an SPMD model-parallel computation (`PartitionId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct PartitionId(pub u32);

/// Execution launch identifier used for collective barriers and poisoning (`LaunchId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct LaunchId(pub i32);

/// Local hardware ordinal on the current host (`LocalHardwareId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct LocalHardwareId(pub i32);

/// Unique identifier for a PJRT memory space (`MemorySpaceId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct MemorySpaceId(pub i32);

/// On-device HBM or VMEM virtual memory address (`DeviceAddress`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct DeviceAddress(pub u64);

/// Host virtual memory address (`HostVirtualAddress`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct HostVirtualAddress(pub u64);

/// IOMMU I/O Virtual Address used in VFIO/Gasket page tables (`IovaAddress`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct IovaAddress(pub u64);

/// `PCIe` Base Address Register (BAR) MMIO byte offset (`MmioOffset`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct MmioOffset(pub u64);

/// Size in bytes (`ByteSize`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ByteSize(pub usize);

/// DMA transfer granule size in bytes (`GranuleSize`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct GranuleSize(pub usize);

/// Memory alignment boundary in bytes (`AlignmentBytes`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct AlignmentBytes(pub usize);

/// Size of a tensor dimension (`DimensionSize`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct DimensionSize(pub i64);

/// Unique identifier for a device buffer allocation (`BufferId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct BufferId(pub u64);

/// Unique identifier for a compiled or loaded executable (`ExecutableId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ExecutableId(pub u64);

/// Unique identifier for an asynchronous synchronization event (`EventId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct EventId(pub u64);

/// Unique identifier for a PJRT execution context (`ContextId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ContextId(pub u64);

/// Unique identifier for a host-to-device copy stream (`StreamId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct StreamId(pub u64);

/// Unique identifier for an asynchronous host-to-device transfer manager (`TransferManagerId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct TransferManagerId(pub u64);

/// Unique identifier for a registered host or debug callback (`CallbackId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CallbackId(pub u64);

/// Unique identifier for a collective communicator (`CommunicatorId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CommunicatorId(pub u64);

/// Unique identifier for a phased compiler instance (`CompilerId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CompilerId(pub u64);

/// Unique identifier for a registered XLA transform pass (`TransformId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct TransformId(pub u64);

/// Request identifier used by `TfTpuOrdinalSelector` (`RequestId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct RequestId(pub i64);

/// Seed value used by `TfTpuOrdinalSelector` (`OrdinalSeed`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct OrdinalSeed(pub u64);

/// `PCIe` Base Address Register index (`BarIndex`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct BarIndex(pub u32);

///Validates C ABI structure byte size (`StructSize`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct StructSize(pub usize);

/// Unique identifier for a cross-host or collective communication channel (`ChannelId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ChannelId(pub i64);

/// Unique identifier for an HBM DMA or Megascale collective window (`WindowId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct WindowId(pub u32);

/// Unique incarnation identifier for a distributed PJRT task (`IncarnationId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct IncarnationId(pub i64);

/// Type alias for [`CollectiveReductionOp`] (`ReductionKind`).
pub type ReductionKind = CollectiveReductionOp;

/// Type alias for [`TpuCoreTypeEnum`] (`CoreType`).
pub type CoreType = TpuCoreTypeEnum;

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

/// POSIX file descriptor for `/dev/accel*` or `/dev/vfio/*` (`FileDescriptor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0}")]
pub struct FileDescriptor(pub i32);

/// Linux `eventfd` interrupt notification descriptor (`EventFdHandle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0}")]
pub struct EventFdHandle(pub i32);

/// Kernel `ioctl` request command code (`IoctlCommand`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0:#x}")]
pub struct IoctlCommand(pub u64);

/// `PCIe` Vendor ID (`PciVendorId`, `0x1ae0` for Google LLC).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0:#06x}")]
pub struct PciVendorId(pub u16);

/// `PCIe` Device ID (`PciDeviceId`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0:#06x}")]
pub struct PciDeviceId(pub u16);

/// Linux VFIO IOMMU group identifier (`IommuGroupId`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0}")]
pub struct IommuGroupId(pub u32);

/// TPU hardware core type classification (`TpuCoreTypeEnum`) matching `libtpu.so` topology queries.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum TpuCoreTypeEnum {
    /// Primary dense matrix/vector `TensorCore` (`0`).
    #[display("TensorCore")]
    TensorCore = 0,
    /// Dedicated embedding `SparseCore` (`1`).
    #[display("SparseCore")]
    SparseCore = 1,
}

impl TpuCoreTypeEnum {
    /// Converts a raw 32-bit integer into a [`TpuCoreTypeEnum`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw integer discriminant (`0` = `TensorCore`, `1` = `SparseCore`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `raw` is not `0` or `1`.
    pub fn from_raw(raw: i32) -> LibTpuResult<Self> {
        match raw {
            0 => Ok(Self::TensorCore),
            1 => Ok(Self::SparseCore),
            _ => Err(LibTpuError::InvalidArgument {
                parameter: "core_type",
                reason: format!("Unknown TpuCoreTypeEnum discriminant {raw}"),
            }),
        }
    }

    /// Returns the raw 32-bit integer discriminant of this [`TpuCoreTypeEnum`].
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self as i32
    }
}

/// 3D torus coordinates `(x, y, z)` of a TPU chip or host in a pod slice.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[display("({x}, {y}, {z})")]
pub struct ChipCoordinates {
    /// X coordinate in the torus mesh.
    pub x: u32,
    /// Y coordinate in the torus mesh.
    pub y: u32,
    /// Z coordinate in the torus mesh.
    pub z: u32,
}

impl ChipCoordinates {
    /// Creates a new [`ChipCoordinates`] triple.
    ///
    /// # Arguments
    ///
    /// * `x` - The X mesh coordinate.
    /// * `y` - The Y mesh coordinate.
    /// * `z` - The Z mesh coordinate.
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Checks equality with another [`ChipCoordinates`] instance.
    ///
    /// # Arguments
    ///
    /// * `other` - The other coordinates to compare against.
    #[must_use]
    pub const fn matches(self, other: Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

/// Google Cloud TPU hardware architecture generation (`TpuVersionEnum`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum TpuGeneration {
    /// Cloud TPU v2 (`Jellyfish`, 2 `TensorCores`/chip, 8 GiB HBM/core).
    #[display("TPU v2")]
    V2 = 2,
    /// Cloud TPU v3 (`Dragonfish`, 2 `TensorCores`/chip, 16 GiB HBM/core).
    #[display("TPU v3")]
    V3 = 3,
    /// Cloud TPU v4 (`Pufferfish`, Megacore, 32 GiB HBM/chip, `SparseCore` v1).
    #[display("TPU v4")]
    V4 = 4,
    /// Cloud TPU v5e / v5lite (`Viperfish Lite`, 1 `TensorCore`/chip, 16 GiB HBM/chip).
    #[display("TPU v5 lite")]
    V5e = 5,
    /// Cloud TPU v5p (`Viperfish Performance`, Megacore, 95 GiB HBM/chip, `SparseCore` v1).
    #[display("TPU v5p")]
    V5p = 6,
    /// Cloud TPU v6e / Trillium (`Ghostfish`, 1 `TensorCore`/chip, 32 GiB HBM/chip, 256x256 MXU).
    #[display("TPU v6 lite")]
    V6e = 7,
    /// Cloud `TPU7x` / Ironwood (`Ironwood`, 2 `TensorCores`/chip, 192 GiB HBM3e/chip, 256x256 MXU).
    #[display("TPU7x")]
    V7x = 8,
}

/// Type alias matching the `TpuVersionEnum` C symbol in `libtpu.so`.
pub type TpuVersionEnum = TpuGeneration;

impl TpuGeneration {
    /// Parses a TPU generation from a string identifier (e.g., `"v4"`, `"tpu_v5e"`, `"trillium"`, `"tpu7x"`).
    ///
    /// # Arguments
    ///
    /// * `name` - Case-insensitive architecture string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if the generation string is unrecognized.
    pub fn from_name(name: &str) -> LibTpuResult<Self> {
        let lower = name.trim().to_ascii_lowercase();
        match lower.as_str() {
            "v2" | "tpu_v2" | "jellyfish" => Ok(Self::V2),
            "v3" | "tpu_v3" | "dragonfish" => Ok(Self::V3),
            "v4" | "tpu_v4" | "pufferfish" => Ok(Self::V4),
            "v5e" | "v5lite" | "tpu_v5e" | "tpu_v5lite" | "viperfish_lite" => Ok(Self::V5e),
            "v5p" | "tpu_v5p" | "viperfish" => Ok(Self::V5p),
            "v6e" | "v6lite" | "tpu_v6e" | "trillium" | "ghostfish" => Ok(Self::V6e),
            "v7" | "v7x" | "tpu7x" | "tpu_v7x" | "ironwood" => Ok(Self::V7x),
            _ => Err(LibTpuError::InvalidArgument {
                parameter: "generation",
                reason: format!("Unknown TPU generation '{name}'"),
            }),
        }
    }

    /// Returns the Google LLC `PCIe` Vendor ID (`0x1ae0`).
    #[must_use]
    pub const fn pcie_vendor_id(self) -> PciVendorId {
        let _ = self;
        PciVendorId(0x1ae0)
    }

    /// Returns the `PCIe` Device ID associated with this TPU generation under Google Vendor ID `0x1ae0`.
    #[must_use]
    pub const fn pcie_device_id(self) -> u16 {
        match self {
            Self::V2 | Self::V3 => 0x0027,
            Self::V4 => 0x0056,
            Self::V5e => 0x0063,
            Self::V5p => 0x0062,
            Self::V6e => 0x006f,
            Self::V7x => 0x0076,
        }
    }

    /// Returns the strongly-typed [`PciDeviceId`] for this TPU generation.
    #[must_use]
    pub const fn typed_pcie_device_id(self) -> PciDeviceId {
        PciDeviceId(self.pcie_device_id())
    }

    /// Returns the HBM capacity per logical device in bytes for this TPU generation.
    #[must_use]
    pub const fn hbm_capacity_bytes(self) -> ByteSize {
        match self {
            Self::V2 => ByteSize(8 * 1024 * 1024 * 1024),
            Self::V3 | Self::V5e => ByteSize(16 * 1024 * 1024 * 1024),
            Self::V4 | Self::V6e => ByteSize(32 * 1024 * 1024 * 1024),
            Self::V5p => ByteSize(95 * 1024 * 1024 * 1024),
            Self::V7x => ByteSize(192 * 1024 * 1024 * 1024),
        }
    }

    /// Returns the on-chip Vector Memory (VMEM) capacity per `TensorCore` in bytes.
    #[must_use]
    pub const fn vmem_capacity_bytes(self) -> ByteSize {
        match self {
            Self::V2 | Self::V3 => ByteSize(16 * 1024 * 1024),
            Self::V4 | Self::V5e => ByteSize(32 * 1024 * 1024),
            Self::V5p | Self::V6e | Self::V7x => ByteSize(64 * 1024 * 1024),
        }
    }

    /// Returns the on-chip Scalar Memory (SMEM) capacity per `TensorCore` in bytes.
    #[must_use]
    pub const fn smem_capacity_bytes(self) -> ByteSize {
        match self {
            Self::V2 | Self::V3 | Self::V4 | Self::V5e => ByteSize(1024 * 1024),
            Self::V5p | Self::V6e | Self::V7x => ByteSize(2 * 1024 * 1024),
        }
    }

    /// Returns the Matrix Multiply Unit (MXU) systolic array dimension (`128` or `256`).
    #[must_use]
    pub const fn mxu_dimension(self) -> u32 {
        match self {
            Self::V2 | Self::V3 | Self::V4 | Self::V5e | Self::V5p => 128,
            Self::V6e | Self::V7x => 256,
        }
    }

    /// Returns the HBM memory bandwidth in gigabytes per second (GB/s) per chip.
    #[must_use]
    pub const fn hbm_bandwidth_gbps(self) -> u32 {
        match self {
            Self::V2 => 600,
            Self::V3 => 900,
            Self::V4 => 1200,
            Self::V5e => 820,
            Self::V5p => 2765,
            Self::V6e => 1640,
            Self::V7x => 3280,
        }
    }

    /// Returns the peak `BF16` compute throughput in TFLOPs per chip.
    #[must_use]
    pub const fn peak_bf16_tflops(self) -> u32 {
        match self {
            Self::V2 => 45,
            Self::V3 => 123,
            Self::V4 => 275,
            Self::V5e => 197,
            Self::V5p => 459,
            Self::V6e => 918,
            Self::V7x => 1836,
        }
    }

    /// Returns the number of Inter-Chip Interconnect (ICI) links per chip.
    #[must_use]
    pub const fn ici_link_count(self) -> u32 {
        match self {
            Self::V2 | Self::V3 | Self::V5e | Self::V6e => 4,
            Self::V4 | Self::V5p | Self::V7x => 6,
        }
    }

    /// Returns the number of logical `TensorCore` devices per physical TPU chip.
    #[must_use]
    pub const fn logical_devices_per_chip(self) -> u32 {
        match self {
            Self::V2 | Self::V3 => 2,
            Self::V4 | Self::V5e | Self::V5p | Self::V6e | Self::V7x => 1,
        }
    }

    /// Returns the number of `SparseCores` per logical device.
    #[must_use]
    pub const fn sparse_cores_per_device(self) -> u32 {
        match self {
            Self::V2 | Self::V3 => 0,
            Self::V5e | Self::V6e => 2,
            Self::V4 | Self::V5p | Self::V7x => 4,
        }
    }

    /// Returns the XLA minor-dimension alignment pad size (128 or 256 elements).
    #[must_use]
    pub const fn xla_pad_size(self) -> i32 {
        match self {
            Self::V2 | Self::V3 | Self::V4 | Self::V5e | Self::V5p => 128,
            Self::V6e | Self::V7x => 256,
        }
    }
}

/// Tensor element data type matching `PJRT_Buffer_Type`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum ElementType {
    /// Invalid or unspecified element type (`0`).
    #[display("INVALID")]
    Invalid = 0,
    /// Boolean predicate (`1`).
    #[display("PRED")]
    Pred = 1,
    /// 8-bit signed integer (`2`).
    #[display("S8")]
    S8 = 2,
    /// 16-bit signed integer (`3`).
    #[display("S16")]
    S16 = 3,
    /// 32-bit signed integer (`4`).
    #[display("S32")]
    S32 = 4,
    /// 64-bit signed integer (`5`).
    #[display("S64")]
    S64 = 5,
    /// 8-bit unsigned integer (`6`).
    #[display("U8")]
    U8 = 6,
    /// 16-bit unsigned integer (`7`).
    #[display("U16")]
    U16 = 7,
    /// 32-bit unsigned integer (`8`).
    #[display("U32")]
    U32 = 8,
    /// 64-bit unsigned integer (`9`).
    #[display("U64")]
    U64 = 9,
    /// IEEE 16-bit floating-point (`10`).
    #[display("F16")]
    F16 = 10,
    /// IEEE 32-bit floating-point (`11`).
    #[display("F32")]
    F32 = 11,
    /// IEEE 64-bit floating-point (`12`).
    #[display("F64")]
    F64 = 12,
    /// Brain floating-point 16-bit (`13`).
    #[display("BF16")]
    Bf16 = 13,
    /// 64-bit complex floating-point (`14`).
    #[display("C64")]
    C64 = 14,
    /// 128-bit complex floating-point (`15`).
    #[display("C128")]
    C128 = 15,
    /// 8-bit floating-point E5M2 (`16`).
    #[display("F8E5M2")]
    F8E5M2 = 16,
    /// 8-bit floating-point E4M3FN (`17`).
    #[display("F8E4M3FN")]
    F8E4M3Fn = 17,
    /// 8-bit floating-point E4M3B11FNUZ (`18`).
    #[display("F8E4M3B11FNUZ")]
    F8E4M3B11Fnuz = 18,
    /// 8-bit floating-point E5M2FNUZ (`19`).
    #[display("F8E5M2FNUZ")]
    F8E5M2Fnuz = 19,
    /// 8-bit floating-point E4M3FNUZ (`20`).
    #[display("F8E4M3FNUZ")]
    F8E4M3Fnuz = 20,
    /// 4-bit signed integer (`21`).
    #[display("S4")]
    S4 = 21,
    /// 4-bit unsigned integer (`22`).
    #[display("U4")]
    U4 = 22,
    /// XLA ordering token (`23`).
    #[display("TOKEN")]
    Token = 23,
    /// 2-bit signed integer (`24`).
    #[display("S2")]
    S2 = 24,
    /// 2-bit unsigned integer (`25`).
    #[display("U2")]
    U2 = 25,
    /// 4-bit floating-point E2M1FN (`26`).
    #[display("F4E2M1FN")]
    F4E2M1Fn = 26,
    /// 8-bit block scaling exponent E8M0FNU (`27`).
    #[display("F8E8M0FNU")]
    F8E8M0Fnu = 27,
}

impl ElementType {
    /// Converts a raw `PJRT_Buffer_Type` integer into an [`ElementType`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw integer code from C FFI.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::UnsupportedElementType`] if `raw` is outside `0..=27`.
    pub const fn from_raw(raw: i32) -> LibTpuResult<Self> {
        match raw {
            0 => Ok(Self::Invalid),
            1 => Ok(Self::Pred),
            2 => Ok(Self::S8),
            3 => Ok(Self::S16),
            4 => Ok(Self::S32),
            5 => Ok(Self::S64),
            6 => Ok(Self::U8),
            7 => Ok(Self::U16),
            8 => Ok(Self::U32),
            9 => Ok(Self::U64),
            10 => Ok(Self::F16),
            11 => Ok(Self::F32),
            12 => Ok(Self::F64),
            13 => Ok(Self::Bf16),
            14 => Ok(Self::C64),
            15 => Ok(Self::C128),
            16 => Ok(Self::F8E5M2),
            17 => Ok(Self::F8E4M3Fn),
            18 => Ok(Self::F8E4M3B11Fnuz),
            19 => Ok(Self::F8E5M2Fnuz),
            20 => Ok(Self::F8E4M3Fnuz),
            21 => Ok(Self::S4),
            22 => Ok(Self::U4),
            23 => Ok(Self::Token),
            24 => Ok(Self::S2),
            25 => Ok(Self::U2),
            26 => Ok(Self::F4E2M1Fn),
            27 => Ok(Self::F8E8M0Fnu),
            _ => Err(LibTpuError::UnsupportedElementType { raw_type: raw }),
        }
    }

    /// Returns the raw integer code matching `PJRT_Buffer_Type`.
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self as i32
    }

    /// Returns the size in bits of a single element of this type.
    #[must_use]
    pub const fn bit_size(self) -> usize {
        match self {
            Self::Invalid | Self::Token => 0,
            Self::S2 | Self::U2 => 2,
            Self::S4 | Self::U4 | Self::F4E2M1Fn => 4,
            Self::Pred
            | Self::S8
            | Self::U8
            | Self::F8E5M2
            | Self::F8E4M3Fn
            | Self::F8E4M3B11Fnuz
            | Self::F8E5M2Fnuz
            | Self::F8E4M3Fnuz
            | Self::F8E8M0Fnu => 8,
            Self::S16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::S32 | Self::U32 | Self::F32 => 32,
            Self::S64 | Self::U64 | Self::F64 | Self::C64 => 64,
            Self::C128 => 128,
        }
    }

    /// Returns the size in bytes of a single element (rounded up to at least 1 byte for non-empty types).
    #[must_use]
    pub const fn byte_size(self) -> usize {
        self.bit_size().div_ceil(8)
    }

    /// Converts an IEEE-754 `f32` to 16-bit `bf16` bits using `half::bf16` round-to-nearest-even.
    ///
    /// # Arguments
    ///
    /// * `val` - 32-bit float value.
    #[must_use]
    pub fn f32_to_bf16_bits(val: f32) -> u16 {
        half::bf16::from_f32(val).to_bits()
    }

    /// Converts 16-bit `bf16` bits to an IEEE-754 `f32` using `half::bf16`.
    ///
    /// # Arguments
    ///
    /// * `bits` - 16-bit `bf16` bit pattern.
    #[must_use]
    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        half::bf16::from_bits(bits).to_f32()
    }

    /// Converts an IEEE-754 `f32` to 16-bit `f16` bits using `half::f16` round-to-nearest-even.
    ///
    /// # Arguments
    ///
    /// * `val` - 32-bit float value.
    #[must_use]
    pub fn f32_to_f16_bits(val: f32) -> u16 {
        half::f16::from_f32(val).to_bits()
    }

    /// Converts 16-bit `f16` bits to an IEEE-754 `f32` using `half::f16`.
    ///
    /// # Arguments
    ///
    /// * `bits` - 16-bit `f16` bit pattern.
    #[must_use]
    pub fn f16_bits_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// Unpacks a raw little-endian byte buffer of this [`ElementType`] into `f64` scalar values
    /// (for `C64` and `C128`, unpacks interleaved `[re0, im0, re1, im1, ...]` scalars).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw little-endian tensor payload.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn unpack_to_f64_vec(self, bytes: &[u8]) -> Vec<f64> {
        match self {
            Self::Pred | Self::U8 => bytes
                .iter()
                .map(|&b| {
                    if self == Self::Pred {
                        if b != 0 { 1.0 } else { 0.0 }
                    } else {
                        f64::from(b)
                    }
                })
                .collect(),
            Self::S8 => bytes
                .iter()
                .map(|&b| f64::from(i8::from_le_bytes([b])))
                .collect(),
            Self::S16 => bytes
                .chunks_exact(2)
                .map(|c| {
                    let mut a = [0_u8; 2];
                    a.copy_from_slice(c);
                    f64::from(i16::from_le_bytes(a))
                })
                .collect(),
            Self::U16 => bytes
                .chunks_exact(2)
                .map(|c| {
                    let mut a = [0_u8; 2];
                    a.copy_from_slice(c);
                    f64::from(u16::from_le_bytes(a))
                })
                .collect(),
            Self::F16 => bytes
                .chunks_exact(2)
                .map(|c| {
                    let mut a = [0_u8; 2];
                    a.copy_from_slice(c);
                    f64::from(Self::f16_bits_to_f32(u16::from_le_bytes(a)))
                })
                .collect(),
            Self::Bf16 => bytes
                .chunks_exact(2)
                .map(|c| {
                    let mut a = [0_u8; 2];
                    a.copy_from_slice(c);
                    f64::from(Self::bf16_bits_to_f32(u16::from_le_bytes(a)))
                })
                .collect(),
            Self::S32 => bytes
                .chunks_exact(4)
                .map(|c| {
                    let mut a = [0_u8; 4];
                    a.copy_from_slice(c);
                    f64::from(i32::from_le_bytes(a))
                })
                .collect(),
            Self::U32 => bytes
                .chunks_exact(4)
                .map(|c| {
                    let mut a = [0_u8; 4];
                    a.copy_from_slice(c);
                    f64::from(u32::from_le_bytes(a))
                })
                .collect(),
            Self::F32 | Self::C64 => bytes
                .chunks_exact(4)
                .map(|c| {
                    let mut a = [0_u8; 4];
                    a.copy_from_slice(c);
                    f64::from(f32::from_le_bytes(a))
                })
                .collect(),
            Self::S64 => bytes
                .chunks_exact(8)
                .map(|c| {
                    let mut a = [0_u8; 8];
                    a.copy_from_slice(c);
                    i64::from_le_bytes(a) as f64
                })
                .collect(),
            Self::U64 => bytes
                .chunks_exact(8)
                .map(|c| {
                    let mut a = [0_u8; 8];
                    a.copy_from_slice(c);
                    u64::from_le_bytes(a) as f64
                })
                .collect(),
            Self::F64 | Self::C128 => bytes
                .chunks_exact(8)
                .map(|c| {
                    let mut a = [0_u8; 8];
                    a.copy_from_slice(c);
                    f64::from_le_bytes(a)
                })
                .collect(),
            _ => bytes.iter().map(|&b| f64::from(b)).collect(),
        }
    }

    /// Packs a slice of `f64` scalar values into a little-endian byte vector of this [`ElementType`].
    ///
    /// # Arguments
    ///
    /// * `values` - Scalar values to pack.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn pack_from_f64_slice(self, values: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len().saturating_mul(self.byte_size().max(1)));
        match self {
            Self::Pred => {
                for &v in values {
                    out.push(u8::from(v != 0.0));
                }
            }
            Self::S8 => {
                for &v in values {
                    out.extend_from_slice(&(v as i8).to_le_bytes());
                }
            }
            Self::S16 => {
                for &v in values {
                    out.extend_from_slice(&(v as i16).to_le_bytes());
                }
            }
            Self::U16 => {
                for &v in values {
                    out.extend_from_slice(&(v as u16).to_le_bytes());
                }
            }
            Self::F16 => {
                for &v in values {
                    out.extend_from_slice(&Self::f32_to_f16_bits(v as f32).to_le_bytes());
                }
            }
            Self::Bf16 => {
                for &v in values {
                    out.extend_from_slice(&Self::f32_to_bf16_bits(v as f32).to_le_bytes());
                }
            }
            Self::S32 => {
                for &v in values {
                    out.extend_from_slice(&(v as i32).to_le_bytes());
                }
            }
            Self::U32 => {
                for &v in values {
                    out.extend_from_slice(&(v as u32).to_le_bytes());
                }
            }
            Self::F32 | Self::C64 => {
                for &v in values {
                    out.extend_from_slice(&(v as f32).to_le_bytes());
                }
            }
            Self::S64 => {
                for &v in values {
                    out.extend_from_slice(&(v as i64).to_le_bytes());
                }
            }
            Self::U64 => {
                for &v in values {
                    out.extend_from_slice(&(v as u64).to_le_bytes());
                }
            }
            Self::F64 | Self::C128 => {
                for &v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            _ => {
                for &v in values {
                    out.push(v as u8);
                }
            }
        }
        out
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
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
}

impl MemoryKind {
    /// Returns the static string identifier expected by PJRT (`PJRT_Memory_Kind`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeviceHbm => "device",
            Self::PinnedHost => "pinned_host",
            Self::UnpinnedHost => "unpinned_host",
            Self::Vmem => "vmem",
        }
    }

    /// Returns the numeric kind ID (`PJRT_Memory_Kind_Id`).
    #[must_use]
    pub const fn kind_id(self) -> i32 {
        self as i32
    }
}

/// Memory layout specification for a TPU buffer (`PJRT_Buffer_MemoryLayout`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Tiled layout with minor-to-major ordering and TPU tile dimensions (e.g., `(8, 128)`).
    Tiled {
        /// Dimension permutation from most-minor (fastest varying) to most-major.
        minor_to_major: Vec<i64>,
        /// Flattened tile dimensions.
        tile_dims: Vec<i64>,
        /// Number of dimensions in each tile level.
        tile_dim_sizes: Vec<usize>,
    },
    /// Explicit byte strides per dimension.
    Strides {
        /// Stride in bytes along each dimension.
        byte_strides: Vec<i64>,
    },
}

impl MemoryLayout {
    /// Creates the default TPU tiled layout (`(8, 128)` for rank >= 2) for a given rank.
    ///
    /// # Arguments
    ///
    /// * `rank` - Number of dimensions in the tensor.
    #[must_use]
    pub fn default_tpu_tiled(rank: usize) -> Self {
        let minor_to_major: Vec<i64> = (0..rank)
            .rev()
            .map(|idx| i64::try_from(idx).unwrap_or(0))
            .collect();
        if rank >= 2 {
            Self::Tiled {
                minor_to_major,
                tile_dims: vec![8, 128],
                tile_dim_sizes: vec![2],
            }
        } else if rank == 1 {
            Self::Tiled {
                minor_to_major,
                tile_dims: vec![128],
                tile_dim_sizes: vec![1],
            }
        } else {
            Self::Tiled {
                minor_to_major,
                tile_dims: Vec::new(),
                tile_dim_sizes: Vec::new(),
            }
        }
    }
}

/// Validated tensor shape, element type, dynamic dimension flags, and layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Element data type.
    pub element_type: ElementType,
    /// Logical dimension sizes.
    pub dims: Vec<i64>,
    /// Dynamic dimension flags (`true` if dimension is dynamically bounded).
    pub dynamic_dims: Vec<bool>,
    /// Memory layout on device.
    pub layout: MemoryLayout,
}

impl Shape {
    /// Constructs and validates a new [`Shape`].
    ///
    /// # Arguments
    ///
    /// * `element_type` - Tensor element type.
    /// * `dims` - Logical dimension extents (must all be `>= 0`).
    /// * `dynamic_dims` - Optional dynamic dimension indicators.
    /// * `layout` - Optional memory layout (defaults to [`MemoryLayout::default_tpu_tiled`]).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if any dimension is negative or if `dynamic_dims`
    /// has a non-empty length that differs from `dims.len()`.
    pub fn new(
        element_type: ElementType,
        dims: Vec<i64>,
        dynamic_dims: Option<Vec<bool>>,
        layout: Option<MemoryLayout>,
    ) -> LibTpuResult<Self> {
        if element_type == ElementType::Invalid {
            return Err(LibTpuError::InvalidShape {
                reason: String::from("ElementType::Invalid is not permitted in a valid Shape"),
            });
        }
        for &d in &dims {
            if d < 0 {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Negative dimension size {d} is not allowed"),
                });
            }
        }
        let resolved_dynamic = match dynamic_dims {
            Some(flags) => {
                if flags.len() != dims.len() {
                    return Err(LibTpuError::InvalidShape {
                        reason: format!(
                            "dynamic_dims length ({}) does not match dims length ({})",
                            flags.len(),
                            dims.len()
                        ),
                    });
                }
                flags
            }
            None => vec![false; dims.len()],
        };
        let resolved_layout = layout.unwrap_or_else(|| MemoryLayout::default_tpu_tiled(dims.len()));
        let bits = element_type.bit_size();
        if bits > 0 {
            let mut num_elements: usize = 1;
            for &d in &dims {
                let extent = usize::try_from(d).unwrap_or(usize::MAX);
                num_elements =
                    num_elements
                        .checked_mul(extent)
                        .ok_or_else(|| LibTpuError::InvalidShape {
                            reason: String::from("Tensor element count overflow"),
                        })?;
            }
            let _total_bits =
                num_elements
                    .checked_mul(bits)
                    .ok_or_else(|| LibTpuError::InvalidShape {
                        reason: String::from("Tensor bit size overflow"),
                    })?;
        }
        Ok(Self {
            element_type,
            dims,
            dynamic_dims: resolved_dynamic,
            layout: resolved_layout,
        })
    }

    /// Computes the unpadded host byte size of a tensor with this validated [`Shape`].
    #[must_use]
    pub fn unpadded_byte_size(&self) -> ByteSize {
        let bits = self.element_type.bit_size();
        if bits == 0 {
            return ByteSize(0);
        }
        let mut num_elements: usize = 1;
        for &d in &self.dims {
            let extent = usize::try_from(d).unwrap_or(0);
            num_elements = num_elements.saturating_mul(extent);
        }
        let total_bits = num_elements.saturating_mul(bits);
        ByteSize(total_bits.div_ceil(8))
    }

    /// Computes the on-device HBM byte size aligned to 256-byte TPU DMA granule boundaries.
    #[must_use]
    pub fn on_device_byte_size(&self) -> ByteSize {
        let unpadded = self.unpadded_byte_size().0;
        if unpadded == 0 {
            return ByteSize(256);
        }
        let aligned = unpadded.div_ceil(256) * 256;
        ByteSize(aligned)
    }

    /// Returns the indices of all dynamic dimensions in this shape.
    #[must_use]
    pub fn dynamic_dimension_indices(&self) -> Vec<usize> {
        self.dynamic_dims
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_dyn)| if is_dyn { Some(idx) } else { None })
            .collect()
    }

    /// Returns the number of dimensions (rank) of this shape.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Computes the total number of scalar elements in this shape.
    #[must_use]
    pub fn element_count(&self) -> usize {
        let mut count: usize = 1;
        for &d in &self.dims {
            let extent = usize::try_from(d).unwrap_or(0);
            count = count.saturating_mul(extent);
        }
        count
    }

    /// Converts a flat row-major linear element index into N-dimensional coordinates.
    ///
    /// # Arguments
    ///
    /// * `linear_idx` - Flat row-major element index.
    #[must_use]
    pub fn linear_to_coords(&self, linear_idx: usize) -> Vec<usize> {
        let mut coords = Vec::with_capacity(self.dims.len());
        let mut rem = linear_idx;
        for &d in self.dims.iter().rev() {
            let extent = usize::try_from(d).unwrap_or(0).max(1);
            coords.push(rem % extent);
            rem /= extent;
        }
        coords.reverse();
        coords
    }

    /// Converts N-dimensional coordinates into a flat row-major linear element index.
    ///
    /// # Arguments
    ///
    /// * `coords` - Coordinate slice of length equal to `self.rank()`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if `coords.len() != self.rank()` or any coordinate is out of bounds.
    pub fn coords_to_linear(&self, coords: &[usize]) -> LibTpuResult<usize> {
        if coords.len() != self.dims.len() {
            return Err(LibTpuError::InvalidShape {
                reason: format!(
                    "Coordinate rank {} does not match shape rank {}",
                    coords.len(),
                    self.dims.len()
                ),
            });
        }
        let mut idx: usize = 0;
        for (&c, &d) in coords.iter().zip(self.dims.iter()) {
            let extent = usize::try_from(d).unwrap_or(0);
            if c >= extent {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Coordinate {c} out of bounds for dimension extent {extent}"),
                });
            }
            idx = idx.saturating_mul(extent).saturating_add(c);
        }
        Ok(idx)
    }

    /// Returns a new [`Shape`] with dimensions permuted by `permutation`.
    ///
    /// # Arguments
    ///
    /// * `permutation` - Bijection of `0..self.rank()`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if `permutation` is not a valid bijection of `0..self.rank()`.
    pub fn transpose(&self, permutation: &[usize]) -> LibTpuResult<Self> {
        if permutation.len() != self.dims.len() {
            return Err(LibTpuError::InvalidShape {
                reason: format!(
                    "Permutation length {} does not match rank {}",
                    permutation.len(),
                    self.dims.len()
                ),
            });
        }
        let mut seen = vec![false; self.dims.len()];
        let mut new_dims = Vec::with_capacity(self.dims.len());
        let mut new_dyn = Vec::with_capacity(self.dims.len());
        for &axis in permutation {
            let Some(seen_flag) = seen.get_mut(axis) else {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Permutation axis {axis} out of range"),
                });
            };
            if *seen_flag {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Duplicate axis {axis} in permutation"),
                });
            }
            *seen_flag = true;
            new_dims.push(self.dims.get(axis).copied().unwrap_or(0));
            new_dyn.push(self.dynamic_dims.get(axis).copied().unwrap_or(false));
        }
        Self::new(self.element_type, new_dims, Some(new_dyn), None)
    }

    /// Computes the hardware tile permutation parameters `(pack_factor, sublanes, lanes)` for this shape:
    /// - 32/64/128-bit elements: 2D `(1, 8, 128)` tile (`8x128`)
    /// - 16-bit elements (`F16`, `Bf16`, `S16`, `U16`): 3D `(2, 4, 128)` sub-tile packing
    /// - 8-bit elements (`S8`, `U8`, `Pred`, `F8*`): 3D `(4, 4, 128)` sub-tile packing
    /// - Sub-byte elements (`S4`, `U4`, `F4E2M1Fn`, `S2`, `U2`): 3D `(8, 4, 128)` sub-tile packing
    #[must_use]
    pub const fn tpu_tile_geometry(&self) -> (usize, usize, usize) {
        let bits = self.element_type.bit_size();
        if bits == 16 {
            (2, 4, 128)
        } else if bits == 8 {
            (4, 4, 128)
        } else if bits < 8 {
            (8, 4, 128)
        } else {
            (1, 8, 128)
        }
    }

    /// Computes the bijective element index mapping from TPU tiled order (`(tr, tc, sr, sc, pack)`)
    /// to row-major linear element indices `0..self.element_count()`.
    #[must_use]
    pub fn tile_permutation_indices(&self) -> Vec<usize> {
        let total = self.element_count();
        if self.dims.len() < 2 {
            return (0..total).collect();
        }
        let rank = self.dims.len();
        let rows = usize::try_from(self.dims.get(rank - 2).copied().unwrap_or(1))
            .unwrap_or(1)
            .max(1);
        let cols = usize::try_from(self.dims.get(rank - 1).copied().unwrap_or(1))
            .unwrap_or(1)
            .max(1);
        let plane = (rows * cols).max(1);
        let batch = (total / plane).max(1);
        let (pack, sub_r, tile_c) = self.tpu_tile_geometry();
        let tile_r = pack * sub_r;

        let mut perm = Vec::with_capacity(total);
        for b in 0..batch {
            for tr in 0..rows.div_ceil(tile_r) {
                for tc in 0..cols.div_ceil(tile_c) {
                    for sr in 0..sub_r {
                        for sc in 0..tile_c {
                            for p in 0..pack {
                                let r = tr * tile_r + sr * pack + p;
                                let c = tc * tile_c + sc;
                                if r < rows && c < cols {
                                    perm.push((b * rows + r) * cols + c);
                                }
                            }
                        }
                    }
                }
            }
        }
        perm
    }

    /// Linearizes host row-major bytes into 256-byte-aligned TPU tiled HBM bytes (`TpuDriver_LinearizeShape`),
    /// applying 2D `(8, 128)` and 3D `(2, 4, 128)` / `(4, 4, 128)` / `(8, 4, 128)` hardware tile permutation.
    ///
    /// # Arguments
    ///
    /// * `host_bytes` - Source row-major host byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `host_bytes.len() > self.on_device_byte_size().0`.
    pub fn linearize_to_tpu_tiles(&self, host_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        let on_dev = self.on_device_byte_size().0;
        if host_bytes.len() > on_dev {
            return Err(LibTpuError::InvalidArgument {
                parameter: "host_bytes",
                reason: format!(
                    "Host payload ({} bytes) exceeds on-device size ({on_dev} bytes)",
                    host_bytes.len()
                ),
            });
        }
        let unpadded = self.unpadded_byte_size().0;
        let mut src_padded = Vec::with_capacity(unpadded.max(host_bytes.len()));
        src_padded.extend_from_slice(host_bytes);
        if src_padded.len() < unpadded {
            src_padded.resize(unpadded, 0);
        }
        let bits = self.element_type.bit_size();
        let mut out = vec![0_u8; on_dev];
        if self.dims.len() >= 2 && bits >= 8 {
            let elem_bytes = bits / 8;
            let perm = self.tile_permutation_indices();
            for (tiled_idx, &src_idx) in perm.iter().enumerate() {
                let src_off = src_idx * elem_bytes;
                let dst_off = tiled_idx * elem_bytes;
                for (dst_b, &src_b) in out
                    .iter_mut()
                    .skip(dst_off)
                    .take(elem_bytes)
                    .zip(src_padded.iter().skip(src_off).take(elem_bytes))
                {
                    *dst_b = src_b;
                }
            }
        } else {
            for (dst_b, &src_b) in out.iter_mut().zip(src_padded.iter()) {
                *dst_b = src_b;
            }
        }
        Ok(out)
    }

    /// Delinearizes TPU tiled HBM bytes back into unpadded host row-major bytes (`TpuDriver_DelinearizeShape`),
    /// inverting the 2D `(8, 128)` and 3D `(2, 4, 128)` / `(4, 4, 128)` / `(8, 4, 128)` hardware tile permutation.
    ///
    /// # Arguments
    ///
    /// * `tiled_bytes` - Source TPU tiled HBM byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `tiled_bytes.len() < self.unpadded_byte_size().0`.
    pub fn delinearize_from_tpu_tiles(&self, tiled_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        let unpadded = self.unpadded_byte_size().0;
        if tiled_bytes.len() < unpadded {
            return Err(LibTpuError::InvalidArgument {
                parameter: "tiled_bytes",
                reason: format!(
                    "Tiled buffer ({} bytes) smaller than unpadded size ({unpadded} bytes)",
                    tiled_bytes.len()
                ),
            });
        }
        let bits = self.element_type.bit_size();
        let mut out = vec![0_u8; unpadded];
        if self.dims.len() >= 2 && bits >= 8 {
            let elem_bytes = bits / 8;
            let perm = self.tile_permutation_indices();
            for (tiled_idx, &dst_idx) in perm.iter().enumerate() {
                let src_off = tiled_idx * elem_bytes;
                let dst_off = dst_idx * elem_bytes;
                for (dst_b, &src_b) in out
                    .iter_mut()
                    .skip(dst_off)
                    .take(elem_bytes)
                    .zip(tiled_bytes.iter().skip(src_off).take(elem_bytes))
                {
                    *dst_b = src_b;
                }
            }
        } else {
            for (dst_b, &src_b) in out.iter_mut().zip(tiled_bytes.iter()) {
                *dst_b = src_b;
            }
        }
        Ok(out)
    }

    /// Alias for [`Shape::linearize_to_tpu_tiles`].
    ///
    /// # Arguments
    ///
    /// * `host_bytes` - Source row-major host byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `host_bytes.len() > self.on_device_byte_size().0`.
    pub fn linearize_bytes(&self, host_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.linearize_to_tpu_tiles(host_bytes)
    }

    /// Alias for [`Shape::delinearize_from_tpu_tiles`].
    ///
    /// # Arguments
    ///
    /// * `tiled_bytes` - Source TPU tiled HBM byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `tiled_bytes.len() < self.unpadded_byte_size().0`.
    pub fn delinearize_bytes(&self, tiled_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.delinearize_from_tpu_tiles(tiled_bytes)
    }

    /// Converts linear host row-major bytes into hardware TPU tiled layout (`linear_to_tpu_tiled`).
    ///
    /// # Arguments
    ///
    /// * `linear_bytes` - Source row-major linear byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `linear_bytes.len() > self.on_device_byte_size().0`.
    pub fn linear_to_tpu_tiled(&self, linear_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.linearize_to_tpu_tiles(linear_bytes)
    }

    /// Converts hardware TPU tiled bytes back into linear host row-major layout (`tpu_tiled_to_linear`).
    ///
    /// # Arguments
    ///
    /// * `tiled_bytes` - Source TPU tiled byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `tiled_bytes.len() < self.unpadded_byte_size().0`.
    pub fn tpu_tiled_to_linear(&self, tiled_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.delinearize_from_tpu_tiles(tiled_bytes)
    }
}

/// Type alias for [`Shape`] (`TensorShape`).
pub type TensorShape = Shape;

/// Typestate marker for an unallocated device buffer.
#[derive(Debug, Clone, Copy)]
pub struct Unallocated;

/// Typestate marker for an allocated device buffer backed by HBM.
#[derive(Debug, Clone, Copy)]
pub struct Allocated;

/// Typestate-tracked device buffer handle transitioning from [`Unallocated`] to [`Allocated`].
#[derive(Debug, Clone)]
pub struct TypedDeviceBuffer<State> {
    /// Unique identifier for the buffer.
    pub id: BufferId,
    /// Target device ID.
    pub device_id: DeviceId,
    /// Validated tensor shape.
    pub shape: Shape,
    /// Optional allocated HBM address.
    pub address: Option<DeviceAddress>,
    /// Phantom typestate marker.
    _state: PhantomData<State>,
}

impl TypedDeviceBuffer<Unallocated> {
    /// Creates a new [`TypedDeviceBuffer<Unallocated>`] before HBM allocation.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique buffer identifier.
    /// * `device_id` - Target TPU device identifier.
    /// * `shape` - Validated tensor shape.
    #[must_use]
    pub const fn new_unallocated(id: BufferId, device_id: DeviceId, shape: Shape) -> Self {
        Self {
            id,
            device_id,
            shape,
            address: None,
            _state: PhantomData,
        }
    }

    /// Transitions this buffer into the [`Allocated`] typestate by binding an HBM [`DeviceAddress`].
    ///
    /// # Arguments
    ///
    /// * `address` - Allocated HBM device address.
    #[must_use]
    pub fn bind_allocation(self, address: DeviceAddress) -> TypedDeviceBuffer<Allocated> {
        TypedDeviceBuffer {
            id: self.id,
            device_id: self.device_id,
            shape: self.shape,
            address: Some(address),
            _state: PhantomData,
        }
    }
}

impl TypedDeviceBuffer<Allocated> {
    /// Returns the guaranteed [`DeviceAddress`] for this allocated buffer.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn device_address(&self) -> DeviceAddress {
        self.address.unwrap_or(DeviceAddress(0))
    }
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::float_cmp
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_newtypes_and_coordinates() {
        let dev = DeviceId::from(3_u32) + DeviceId(1) - DeviceId(1);
        assert_eq!(u32::from(dev), 3);
        assert_eq!(dev.to_string(), "3");

        let chip = ChipId::from(1_u32) + ChipId(1) - ChipId(1);
        assert_eq!(u32::from(chip), 1);
        assert_eq!(chip.to_string(), "1");

        let core = CoreId::from(0_u32) + CoreId(1) - CoreId(1);
        assert_eq!(u32::from(core), 0);
        assert_eq!(core.to_string(), "0");

        let core_idx = CoreIndex::from(1_u8) + CoreIndex(1) - CoreIndex(1);
        assert_eq!(u8::from(core_idx), 1);
        assert_eq!(core_idx.to_string(), "1");

        let host = HostId::from(2_u32) + HostId(1) - HostId(1);
        assert_eq!(u32::from(host), 2);
        assert_eq!(host.to_string(), "2");

        let proc_idx = ProcessIndex::from(0_i32) + ProcessIndex(1) - ProcessIndex(1);
        assert_eq!(i32::from(proc_idx), 0);
        assert_eq!(proc_idx.to_string(), "0");

        let slice_id = SliceId::from(0_u32) + SliceId(1) - SliceId(1);
        assert_eq!(u32::from(slice_id), 0);
        assert_eq!(slice_id.to_string(), "0");

        let launch_id = LaunchId::from(7_i32) + LaunchId(1) - LaunchId(1);
        assert_eq!(i32::from(launch_id), 7);
        assert_eq!(launch_id.to_string(), "7");

        let hw_id = LocalHardwareId::from(1_i32) + LocalHardwareId(1) - LocalHardwareId(1);
        assert_eq!(i32::from(hw_id), 1);
        assert_eq!(hw_id.to_string(), "1");

        let mem_id = MemorySpaceId::from(4_i32) + MemorySpaceId(1) - MemorySpaceId(1);
        assert_eq!(i32::from(mem_id), 4);
        assert_eq!(mem_id.to_string(), "4");

        let buf_id = BufferId::from(10_u64) + BufferId(1) - BufferId(1);
        assert_eq!(u64::from(buf_id), 10);
        assert_eq!(buf_id.to_string(), "10");

        let exec_id = ExecutableId::from(11_u64) + ExecutableId(1) - ExecutableId(1);
        assert_eq!(u64::from(exec_id), 11);
        assert_eq!(exec_id.to_string(), "11");

        let ev_id = EventId::from(12_u64) + EventId(1) - EventId(1);
        assert_eq!(u64::from(ev_id), 12);
        assert_eq!(ev_id.to_string(), "12");

        let ctx_id = ContextId::from(13_u64) + ContextId(1) - ContextId(1);
        assert_eq!(u64::from(ctx_id), 13);
        assert_eq!(ctx_id.to_string(), "13");

        let stream_id = StreamId::from(14_u64) + StreamId(1) - StreamId(1);
        assert_eq!(u64::from(stream_id), 14);
        assert_eq!(stream_id.to_string(), "14");

        let tm_id = TransferManagerId::from(15_u64) + TransferManagerId(1) - TransferManagerId(1);
        assert_eq!(u64::from(tm_id), 15);
        assert_eq!(tm_id.to_string(), "15");

        let cb_id = CallbackId::from(16_u64) + CallbackId(1) - CallbackId(1);
        assert_eq!(u64::from(cb_id), 16);
        assert_eq!(cb_id.to_string(), "16");

        let communicator_id = CommunicatorId::from(17_u64) + CommunicatorId(1) - CommunicatorId(1);
        assert_eq!(u64::from(communicator_id), 17);
        assert_eq!(communicator_id.to_string(), "17");

        let compiler_id = CompilerId::from(18_u64) + CompilerId(1) - CompilerId(1);
        assert_eq!(u64::from(compiler_id), 18);
        assert_eq!(compiler_id.to_string(), "18");

        let xform_id = TransformId::from(19_u64) + TransformId(1) - TransformId(1);
        assert_eq!(u64::from(xform_id), 19);
        assert_eq!(xform_id.to_string(), "19");

        let req_id = RequestId::from(20_i64) + RequestId(1) - RequestId(1);
        assert_eq!(i64::from(req_id), 20);
        assert_eq!(req_id.to_string(), "20");

        let seed = OrdinalSeed::from(21_u64) + OrdinalSeed(1) - OrdinalSeed(1);
        assert_eq!(u64::from(seed), 21);
        assert_eq!(seed.to_string(), "21");

        let bar = BarIndex::from(2_u32) + BarIndex(1) - BarIndex(1);
        assert_eq!(u32::from(bar), 2);
        assert_eq!(bar.to_string(), "2");

        let st_sz = StructSize::from(32_usize) + StructSize(8) - StructSize(8);
        assert_eq!(usize::from(st_sz), 32);
        assert_eq!(st_sz.to_string(), "32");

        let chan_id = ChannelId::from(5_i64) + ChannelId(1) - ChannelId(1);
        assert_eq!(i64::from(chan_id), 5);
        assert_eq!(chan_id.to_string(), "5");

        let win_id = WindowId::from(9_u32) + WindowId(1) - WindowId(1);
        assert_eq!(u32::from(win_id), 9);
        assert_eq!(win_id.to_string(), "9");

        let inc_id = IncarnationId::from(42_i64) + IncarnationId(1) - IncarnationId(1);
        assert_eq!(i64::from(inc_id), 42);
        assert_eq!(inc_id.to_string(), "42");

        assert_eq!(ReductionKind::Sum.as_raw(), 0);
        assert_eq!(CoreType::TensorCore.as_raw(), 0);

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

        let addr = DeviceAddress::from(0x1000_u64) + DeviceAddress(0x10) - DeviceAddress(0x10);
        assert_eq!(u64::from(addr), 0x1000);
        assert_eq!(addr.to_string(), "0x1000");

        let hva =
            HostVirtualAddress::from(0x2000_u64) + HostVirtualAddress(8) - HostVirtualAddress(8);
        assert_eq!(u64::from(hva), 0x2000);
        assert_eq!(hva.to_string(), "0x2000");

        let iova = IovaAddress::from(0x3000_u64) + IovaAddress(8) - IovaAddress(8);
        assert_eq!(u64::from(iova), 0x3000);
        assert_eq!(iova.to_string(), "0x3000");

        let mmio = MmioOffset::from(0x400_u64) + MmioOffset(4) - MmioOffset(4);
        assert_eq!(u64::from(mmio), 0x400);
        assert_eq!(mmio.to_string(), "0x400");

        let bsize = ByteSize::from(512_usize) + ByteSize(64) - ByteSize(64);
        assert_eq!(usize::from(bsize), 512);
        assert_eq!(bsize.to_string(), "512");

        let gsize = GranuleSize::from(256_usize) + GranuleSize(16) - GranuleSize(16);
        assert_eq!(usize::from(gsize), 256);
        assert_eq!(gsize.to_string(), "256");

        let align = AlignmentBytes::from(4096_usize) + AlignmentBytes(16) - AlignmentBytes(16);
        assert_eq!(usize::from(align), 4096);
        assert_eq!(align.to_string(), "4096");

        let dim = DimensionSize::from(128_i64) + DimensionSize(1) - DimensionSize(1);
        assert_eq!(i64::from(dim), 128);
        assert_eq!(dim.to_string(), "128");

        let rep = ReplicaId::from(0_u32) + ReplicaId(1) - ReplicaId(1);
        assert_eq!(u32::from(rep), 0);
        assert_eq!(rep.to_string(), "0");

        let part = PartitionId::from(1_u32) + PartitionId(1) - PartitionId(1);
        assert_eq!(u32::from(part), 1);
        assert_eq!(part.to_string(), "1");

        let fd = FileDescriptor::from(3_i32);
        assert_eq!(i32::from(fd), 3);
        assert_eq!(fd.to_string(), "3");

        let efd = EventFdHandle::from(4_i32);
        assert_eq!(i32::from(efd), 4);
        assert_eq!(efd.to_string(), "4");

        let ioctl_cmd = IoctlCommand::from(0x4020_dc08_u64);
        assert_eq!(u64::from(ioctl_cmd), 0x4020_dc08);
        assert_eq!(ioctl_cmd.to_string(), "0x4020dc08");

        let vendor_id = PciVendorId::from(0x1ae0_u16);
        assert_eq!(u16::from(vendor_id), 0x1ae0);
        assert_eq!(vendor_id.to_string(), "0x1ae0");

        let pci_dev_id = PciDeviceId::from(0x006f_u16);
        assert_eq!(u16::from(pci_dev_id), 0x006f);
        assert_eq!(pci_dev_id.to_string(), "0x006f");

        let iommu = IommuGroupId::from(12_u32);
        assert_eq!(u32::from(iommu), 12);
        assert_eq!(iommu.to_string(), "12");

        assert_eq!(
            TpuCoreTypeEnum::from_raw(0).ok(),
            Some(TpuCoreTypeEnum::TensorCore)
        );
        assert_eq!(
            TpuCoreTypeEnum::from_raw(1).ok(),
            Some(TpuCoreTypeEnum::SparseCore)
        );
        assert!(TpuCoreTypeEnum::from_raw(2).is_err());
        assert_eq!(TpuCoreTypeEnum::TensorCore.as_raw(), 0);
        assert_eq!(TpuCoreTypeEnum::SparseCore.as_raw(), 1);
        assert_eq!(TpuCoreTypeEnum::TensorCore.to_string(), "TensorCore");
        assert_eq!(TpuCoreTypeEnum::SparseCore.to_string(), "SparseCore");
        assert_ne!(format!("{:?}", TpuCoreTypeEnum::TensorCore), "");

        let c1 = ChipCoordinates::new(1, 2, 3);
        let c2 = ChipCoordinates::new(1, 2, 3);
        let c3 = ChipCoordinates::new(1, 2, 4);
        let c4 = ChipCoordinates::new(1, 9, 3);
        let c5 = ChipCoordinates::new(9, 2, 3);
        assert!(c1.matches(c2));
        assert!(!c1.matches(c3));
        assert!(!c1.matches(c4));
        assert!(!c1.matches(c5));
        assert_eq!(c1.to_string(), "(1, 2, 3)");
        assert_ne!(format!("{c1:?}"), "");
    }

    #[test]
    fn test_tpu_generation_specs() {
        let gens = [
            (
                "v2",
                TpuGeneration::V2,
                8,
                2,
                0,
                128,
                0x0027,
                16,
                1,
                128,
                600,
                45,
                4,
            ),
            (
                "v3",
                TpuGeneration::V3,
                16,
                2,
                0,
                128,
                0x0027,
                16,
                1,
                128,
                900,
                123,
                4,
            ),
            (
                "v4",
                TpuGeneration::V4,
                32,
                1,
                4,
                128,
                0x0056,
                32,
                1,
                128,
                1200,
                275,
                6,
            ),
            (
                "v5e",
                TpuGeneration::V5e,
                16,
                1,
                2,
                128,
                0x0063,
                32,
                1,
                128,
                820,
                197,
                4,
            ),
            (
                "v5p",
                TpuGeneration::V5p,
                95,
                1,
                4,
                128,
                0x0062,
                64,
                2,
                128,
                2765,
                459,
                6,
            ),
            (
                "v6e",
                TpuGeneration::V6e,
                32,
                1,
                2,
                256,
                0x006f,
                64,
                2,
                256,
                1640,
                918,
                4,
            ),
            (
                "tpu7x",
                TpuGeneration::V7x,
                192,
                1,
                4,
                256,
                0x0076,
                64,
                2,
                256,
                3280,
                1836,
                6,
            ),
        ];
        for (name, expected, gib, devs, sc, pad, pci, vmem_mib, smem_mib, mxu, bw, tflops, ici) in
            gens
        {
            let parsed = TpuGeneration::from_name(name);
            assert!(parsed.is_ok());
            for g in parsed {
                assert_eq!(g, expected);
                assert_eq!(g.pcie_vendor_id(), PciVendorId(0x1ae0));
                assert_eq!(g.pcie_device_id(), pci);
                assert_eq!(g.typed_pcie_device_id(), PciDeviceId(pci));
                assert_eq!(g.hbm_capacity_bytes(), ByteSize(gib * 1024 * 1024 * 1024));
                assert_eq!(g.vmem_capacity_bytes(), ByteSize(vmem_mib * 1024 * 1024));
                assert_eq!(g.smem_capacity_bytes(), ByteSize(smem_mib * 1024 * 1024));
                assert_eq!(g.mxu_dimension(), mxu);
                assert_eq!(g.hbm_bandwidth_gbps(), bw);
                assert_eq!(g.peak_bf16_tflops(), tflops);
                assert_eq!(g.ici_link_count(), ici);
                assert_eq!(g.logical_devices_per_chip(), devs);
                assert_eq!(g.sparse_cores_per_device(), sc);
                assert_eq!(g.xla_pad_size(), pad);
                assert_ne!(g.to_string(), "");
                assert_ne!(format!("{g:?}"), "");
            }
        }
        assert!(TpuGeneration::from_name("unknown_tpu").is_err());
    }

    #[test]
    fn test_element_types_and_memory_kinds() {
        for raw in 0..=27 {
            let et = ElementType::from_raw(raw);
            assert!(et.is_ok());
            for t in et {
                assert_eq!(t.as_raw(), raw);
                let _ = t.bit_size();
                assert_ne!(t.to_string(), "");
                assert_ne!(format!("{t:?}"), "");
            }
        }
        assert!(ElementType::from_raw(99).is_err());

        let kinds = [
            (MemoryKind::DeviceHbm, "device", 0),
            (MemoryKind::PinnedHost, "pinned_host", 1),
            (MemoryKind::UnpinnedHost, "unpinned_host", 2),
            (MemoryKind::Vmem, "vmem", 3),
        ];
        for (k, s, id) in kinds {
            assert_eq!(k.as_str(), s);
            assert_eq!(k.kind_id(), id);
            assert_eq!(k.to_string(), s);
            assert_ne!(format!("{k:?}"), "");
        }

        assert_eq!(ElementType::f16_bits_to_f32(0), 0.0);
        assert!(ElementType::f16_bits_to_f32(0x7c00).is_infinite());
        assert_eq!(ElementType::f32_to_f16_bits(0.0), 0);
        assert_eq!(ElementType::f32_to_f16_bits(100_000.0), 0x7c00);
        let f16_one = ElementType::f32_to_f16_bits(1.5);
        assert!((ElementType::f16_bits_to_f32(f16_one) - 1.5).abs() < 1e-3);
        let bf16_two = ElementType::f32_to_bf16_bits(2.5);
        assert!((ElementType::bf16_bits_to_f32(bf16_two) - 2.5).abs() < 1e-2);

        for et in [
            ElementType::Pred,
            ElementType::S8,
            ElementType::U8,
            ElementType::S16,
            ElementType::U16,
            ElementType::F16,
            ElementType::Bf16,
            ElementType::S32,
            ElementType::U32,
            ElementType::F32,
            ElementType::C64,
            ElementType::S64,
            ElementType::U64,
            ElementType::F64,
            ElementType::C128,
            ElementType::F8E5M2,
        ] {
            let packed = et.pack_from_f64_slice(&[0.0, 2.0, 3.0, 4.0]);
            let unpacked = et.unpack_to_f64_vec(&packed);
            assert_eq!(unpacked.len(), 4);
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

    #[test]
    fn test_shape_and_typestate_buffer() {
        assert!(Shape::new(ElementType::Invalid, vec![1], None, None).is_err());
        assert!(Shape::new(ElementType::F32, vec![-1], None, None).is_err());
        assert!(Shape::new(ElementType::F32, vec![4, 4], Some(vec![true]), None).is_err());

        let scalar = Shape::new(ElementType::Token, vec![], None, None);
        assert!(scalar.is_ok());
        for s in scalar {
            assert_eq!(s.unpadded_byte_size(), ByteSize(0));
            assert_eq!(s.on_device_byte_size(), ByteSize(256));
        }

        let rank1 = Shape::new(
            ElementType::F32,
            vec![10],
            Some(vec![true]),
            Some(MemoryLayout::Strides {
                byte_strides: vec![4],
            }),
        );
        assert!(rank1.is_ok());
        for s in rank1 {
            assert_eq!(s.unpadded_byte_size(), ByteSize(40));
            assert_eq!(s.on_device_byte_size(), ByteSize(256));
            assert_eq!(s.dynamic_dimension_indices(), vec![0]);
            assert_ne!(format!("{s:?}"), "");
        }

        let rank1_default = Shape::new(ElementType::S4, vec![3], None, None);
        assert!(rank1_default.is_ok());
        for s in rank1_default {
            assert_eq!(s.unpadded_byte_size(), ByteSize(2));
        }

        let rank2 = Shape::new(
            ElementType::Bf16,
            vec![8, 128],
            Some(vec![false, true]),
            None,
        );
        assert!(rank2.is_ok());
        for s in rank2 {
            assert_eq!(s.unpadded_byte_size(), ByteSize(2048));
            assert_eq!(s.on_device_byte_size(), ByteSize(2048));
            assert_eq!(s.dynamic_dimension_indices(), vec![1]);

            let unalloc = TypedDeviceBuffer::new_unallocated(BufferId(1), DeviceId(0), s);
            let unalloc_cloned = unalloc.clone();
            assert_ne!(format!("{unalloc_cloned:?}"), "");
            let alloc = unalloc.bind_allocation(DeviceAddress(0x4000));
            let alloc_cloned = alloc.clone();
            assert_eq!(alloc.device_address(), DeviceAddress(0x4000));
            drop(alloc);
            assert_eq!(alloc_cloned.device_address(), DeviceAddress(0x4000));
            assert_ne!(format!("{alloc_cloned:?}"), "");
            assert_ne!(format!("{Unallocated:?}"), "");
            assert_ne!(format!("{Allocated:?}"), "");
        }

        assert!(Shape::new(ElementType::F32, vec![i64::MAX, i64::MAX], None, None).is_err());
        assert!(Shape::new(ElementType::C128, vec![i64::MAX], None, None).is_err());

        for s in Shape::new(ElementType::F32, vec![2, 3], None, None) {
            assert_eq!(s.rank(), 2);
            assert_eq!(s.element_count(), 6);
            assert_eq!(s.linear_to_coords(4), vec![1, 1]);
            assert_eq!(s.coords_to_linear(&[1, 1]).ok(), Some(4));
            assert!(s.coords_to_linear(&[1]).is_err());
            assert!(s.coords_to_linear(&[2, 0]).is_err());

            let tr = s.transpose(&[1, 0]);
            assert!(tr.is_ok());
            for t in tr {
                assert_eq!(t.dims, vec![3, 2]);
            }
            assert!(s.transpose(&[0]).is_err());
            assert!(s.transpose(&[2, 0]).is_err());
            assert!(s.transpose(&[0, 0]).is_err());

            let host_bytes: Vec<u8> = (0_u8..24).collect();
            let tiled = s.linearize_bytes(&host_bytes);
            assert!(tiled.is_ok());
            for tb in tiled {
                assert_eq!(tb.len(), 256);
                let roundtrip = s.delinearize_bytes(&tb);
                assert_eq!(roundtrip.ok(), Some(host_bytes.clone()));
            }
            let short_tiled = s.linearize_to_tpu_tiles(&[1, 2, 3, 4]);
            assert!(short_tiled.is_ok());
            assert!(s.linearize_to_tpu_tiles(&vec![0_u8; 512]).is_err());
            assert!(s.delinearize_from_tpu_tiles(&[0_u8; 4]).is_err());
        }

        for (et, dims, expected_geom) in [
            (ElementType::F32, vec![4_i64], (1, 8, 128)),
            (ElementType::Bf16, vec![2, 3], (2, 4, 128)),
            (ElementType::U8, vec![2, 3], (4, 4, 128)),
            (ElementType::S4, vec![2, 4], (8, 4, 128)),
        ] {
            for sh in TensorShape::new(et, dims, None, None) {
                assert_eq!(sh.tpu_tile_geometry(), expected_geom);
                let perm = sh.tile_permutation_indices();
                assert_eq!(perm.len(), sh.element_count());
                let raw: Vec<u8> = (0..sh.unpadded_byte_size().0)
                    .map(|i| u8::try_from(i % 251).unwrap_or(0))
                    .collect();
                for tb in sh.linear_to_tpu_tiled(&raw) {
                    assert_eq!(sh.tpu_tiled_to_linear(&tb).ok(), Some(raw.clone()));
                }
                for tb in sh.linearize_bytes(&raw) {
                    assert_eq!(sh.delinearize_bytes(&tb).ok(), Some(raw.clone()));
                }
            }
        }
    }
}
