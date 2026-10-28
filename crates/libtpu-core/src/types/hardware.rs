//! TPU hardware generations, core classifications, and 3D torus mesh coordinates.
//!
//! Defines [`TpuGeneration`] (`v2` through `TPU7x` / `Ironwood`), [`TpuCoreTypeEnum`]
//! (`TensorCore`, `SparseCore`, `BarnaCore`), and [`ChipCoordinates`] 3D torus geometry.

use crate::error::{LibTpuError, LibTpuResult};
use crate::types::address::{ByteSize, PciDeviceId, PciVendorId};
use bytemuck::{Pod, Zeroable};
use derive_more::Display;
use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Type alias representing a 2D TPU tile shape (`TileShape`).
pub type TileShape = (i64, i64);

/// TPU hardware core type classification (`TpuCoreTypeEnum`) matching `libtpu.so` topology queries.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, TryFromPrimitive, IntoPrimitive)]
pub enum TpuCoreTypeEnum {
    /// Primary dense matrix/vector `TensorCore` (`0`).
    #[display("TensorCore")]
    TensorCore = 0,
    /// Dedicated embedding `SparseCore` (`1`).
    #[display("SparseCore")]
    SparseCore = 1,
    /// Legacy v2/v3 `BarnaCore` embedding lookup engine (`2`).
    #[display("BarnaCore")]
    BarnaCore = 2,
}

/// Type alias for [`TpuCoreTypeEnum`] (`TpuCoreType`).
pub type TpuCoreType = TpuCoreTypeEnum;

/// Type alias for [`TpuCoreTypeEnum`] (`CoreType`).
pub type CoreType = TpuCoreTypeEnum;

impl TpuCoreTypeEnum {
    /// Converts a raw 32-bit integer into a [`TpuCoreTypeEnum`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw integer discriminant (`0` = `TensorCore`, `1` = `SparseCore`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `raw` is not `0`, `1`, or `2`.
    pub fn from_raw(raw: i32) -> LibTpuResult<Self> {
        Self::try_from(raw).map_err(|_| LibTpuError::InvalidArgument {
            parameter: "core_type",
            reason: format!("Unknown TpuCoreTypeEnum discriminant {raw}"),
        })
    }

    /// Returns the raw 32-bit integer discriminant of this [`TpuCoreTypeEnum`].
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self as i32
    }
}

/// 3D torus coordinates `(x, y, z)` of a TPU chip or host in a pod slice.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, Pod, Zeroable)]
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

    /// Computes the 3D Manhattan distance between `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Target 3D coordinates.
    #[must_use]
    pub const fn manhattan_distance(self, other: Self) -> u32 {
        self.x
            .abs_diff(other.x)
            .saturating_add(self.y.abs_diff(other.y))
            .saturating_add(self.z.abs_diff(other.z))
    }

    /// Computes the 3D torus distance between `self` and `other` given mesh `bounds` and `wrap` flags.
    ///
    /// # Arguments
    ///
    /// * `other` - Target 3D coordinates.
    /// * `bounds` - Mesh dimensions `(bx, by, bz)`.
    /// * `wrap` - Per-axis torus wrap flags `(wx, wy, wz)`.
    #[must_use]
    pub fn torus_distance(self, other: Self, bounds: Self, wrap: (bool, bool, bool)) -> u32 {
        let axis_dist = |a: u32, b: u32, bound: u32, wrapped: bool| -> u32 {
            let d = a.abs_diff(b);
            if wrapped {
                d.min(bound.saturating_sub(d))
            } else {
                d
            }
        };
        axis_dist(self.x, other.x, bounds.x, wrap.0)
            .saturating_add(axis_dist(self.y, other.y, bounds.y, wrap.1))
            .saturating_add(axis_dist(self.z, other.z, bounds.z, wrap.2))
    }

    /// Computes the row-major linear index of `self` within `bounds`.
    ///
    /// # Arguments
    ///
    /// * `bounds` - Mesh dimensions `(bx, by, bz)`.
    #[must_use]
    pub const fn row_major_index(self, bounds: Self) -> u32 {
        self.x
            .saturating_mul(bounds.y)
            .saturating_add(self.y)
            .saturating_mul(bounds.z)
            .saturating_add(self.z)
    }
}

/// Google Cloud TPU hardware architecture generation (`TpuVersionEnum`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, TryFromPrimitive, IntoPrimitive)]
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
    /// Cloud TPU v4 lite (`Pufferfish Lite`, 1 `TensorCore`/chip, 16 GiB HBM/chip).
    #[display("TPU v4 lite")]
    V4Lite = 9,
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
            "v4lite" | "tpu_v4lite" | "pufferfish_lite" => Ok(Self::V4Lite),
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
            Self::V4Lite => 0x005e,
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
            Self::V3 | Self::V4Lite | Self::V5e => ByteSize(16 * 1024 * 1024 * 1024),
            Self::V4 | Self::V6e => ByteSize(32 * 1024 * 1024 * 1024),
            Self::V5p => ByteSize(95 * 1024 * 1024 * 1024),
            Self::V7x => ByteSize(192 * 1024 * 1024 * 1024),
        }
    }

    /// Returns the on-chip Vector Memory (VMEM) capacity per `TensorCore` in bytes.
    #[must_use]
    pub const fn vmem_capacity_bytes(self) -> ByteSize {
        match self {
            Self::V2 | Self::V3 | Self::V4Lite => ByteSize(16 * 1024 * 1024),
            Self::V4 | Self::V5e => ByteSize(32 * 1024 * 1024),
            Self::V5p | Self::V6e | Self::V7x => ByteSize(64 * 1024 * 1024),
        }
    }

    /// Returns the on-chip Scalar Memory (SMEM) capacity per `TensorCore` in bytes.
    #[must_use]
    pub const fn smem_capacity_bytes(self) -> ByteSize {
        match self {
            Self::V2 | Self::V3 | Self::V4 | Self::V4Lite | Self::V5e => ByteSize(1024 * 1024),
            Self::V5p | Self::V6e | Self::V7x => ByteSize(2 * 1024 * 1024),
        }
    }

    /// Returns the on-chip Collective Memory (CMEM) capacity per chip in bytes.
    #[must_use]
    pub const fn cmem_capacity_bytes(self) -> ByteSize {
        match self {
            Self::V4 | Self::V5p | Self::V7x => ByteSize(128 * 1024 * 1024),
            Self::V2 | Self::V3 | Self::V4Lite | Self::V5e | Self::V6e => ByteSize(0),
        }
    }

    /// Returns the Matrix Multiply Unit (MXU) systolic array dimension (`128` or `256`).
    #[must_use]
    pub const fn mxu_dimension(self) -> u32 {
        match self {
            Self::V2 | Self::V3 | Self::V4 | Self::V4Lite | Self::V5e | Self::V5p => 128,
            Self::V6e | Self::V7x => 256,
        }
    }

    /// Returns the HBM memory bandwidth in gigabytes per second (GB/s) per chip.
    #[must_use]
    pub const fn hbm_bandwidth_gbps(self) -> u32 {
        match self {
            Self::V2 => 600,
            Self::V3 | Self::V4Lite => 900,
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
            Self::V4Lite => 138,
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
            Self::V2 | Self::V3 | Self::V4Lite | Self::V5e | Self::V6e => 4,
            Self::V4 | Self::V5p | Self::V7x => 6,
        }
    }

    /// Returns the number of logical `TensorCore` devices per physical TPU chip.
    #[must_use]
    pub const fn logical_devices_per_chip(self) -> u32 {
        match self {
            Self::V2 | Self::V3 => 2,
            Self::V4 | Self::V4Lite | Self::V5e | Self::V5p | Self::V6e | Self::V7x => 1,
        }
    }

    /// Returns the number of `SparseCores` per logical device.
    #[must_use]
    pub const fn sparse_cores_per_device(self) -> u32 {
        match self {
            Self::V2 | Self::V3 | Self::V4Lite => 0,
            Self::V5e | Self::V6e => 2,
            Self::V4 | Self::V5p | Self::V7x => 4,
        }
    }

    /// Returns the XLA minor-dimension alignment pad size (128 or 256 elements).
    #[must_use]
    pub const fn xla_pad_size(self) -> i32 {
        match self {
            Self::V2 | Self::V3 | Self::V4 | Self::V4Lite | Self::V5e | Self::V5p => 128,
            Self::V6e | Self::V7x => 256,
        }
    }

    /// Returns the number of physical `TensorCores` per TPU chip.
    #[must_use]
    pub const fn tensor_cores_per_chip(self) -> u32 {
        match self {
            Self::V4Lite | Self::V5e | Self::V6e => 1,
            Self::V2 | Self::V3 | Self::V4 | Self::V5p | Self::V7x => 2,
        }
    }

    /// Returns the number of physical `SparseCores` per TPU chip.
    #[must_use]
    pub const fn sparse_cores_per_chip(self) -> u32 {
        self.sparse_cores_per_device()
    }

    /// Returns the number of Matrix Multiply Units (MXUs) per `TensorCore`.
    #[must_use]
    pub const fn mxu_count_per_core(self) -> u32 {
        match self {
            Self::V2 => 1,
            Self::V3 | Self::V4 | Self::V4Lite | Self::V6e => 2,
            Self::V5e | Self::V5p | Self::V7x => 4,
        }
    }

    /// Returns the 2D systolic array dimensions `(rows, cols)` of each MXU.
    #[must_use]
    pub const fn mxu_systolic_dim(self) -> (u32, u32) {
        let dim = self.mxu_dimension();
        (dim, dim)
    }

    /// Returns the Vector Processing Unit (VPU) `(lanes, sublanes)` geometry (`(128, 8)`).
    #[must_use]
    pub const fn vpu_lanes_and_sublanes(self) -> (u32, u32) {
        let _ = self;
        (128, 8)
    }

    /// Returns the total HBM capacity per physical TPU chip in bytes.
    #[must_use]
    pub const fn hbm_capacity_bytes_per_chip(self) -> ByteSize {
        match self {
            Self::V2 | Self::V4Lite | Self::V5e => ByteSize(16 * 1024 * 1024 * 1024),
            Self::V3 | Self::V4 | Self::V6e => ByteSize(32 * 1024 * 1024 * 1024),
            Self::V5p => ByteSize(95 * 1024 * 1024 * 1024),
            Self::V7x => ByteSize(192 * 1024 * 1024 * 1024),
        }
    }

    /// Returns the on-chip VMEM capacity per `TensorCore` in bytes.
    #[must_use]
    pub const fn vmem_capacity_bytes_per_core(self) -> ByteSize {
        self.vmem_capacity_bytes()
    }

    /// Returns the on-chip SMEM capacity per `TensorCore` in bytes.
    #[must_use]
    pub const fn smem_capacity_bytes_per_core(self) -> ByteSize {
        self.smem_capacity_bytes()
    }

    /// Returns the default primary 2D minor-to-major tile shape `(second_minor, most_minor)` for this TPU generation.
    #[must_use]
    pub const fn default_minor_to_major_tile(self) -> TileShape {
        match self {
            Self::V2 | Self::V3 | Self::V4 | Self::V4Lite | Self::V5e | Self::V5p => (8, 128),
            Self::V6e | Self::V7x => (16, 256),
        }
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
    fn test_core_type_and_coordinates() {
        assert_eq!(CoreType::TensorCore.as_raw(), 0);
        assert_eq!(TpuCoreType::TensorCore.as_raw(), 0);
        assert_eq!(
            TpuCoreTypeEnum::from_raw(0).ok(),
            Some(TpuCoreTypeEnum::TensorCore)
        );
        assert_eq!(
            TpuCoreTypeEnum::from_raw(1).ok(),
            Some(TpuCoreTypeEnum::SparseCore)
        );
        assert_eq!(
            TpuCoreTypeEnum::from_raw(2).ok(),
            Some(TpuCoreTypeEnum::BarnaCore)
        );
        assert!(TpuCoreTypeEnum::from_raw(3).is_err());
        assert_eq!(TpuCoreTypeEnum::TensorCore.as_raw(), 0);
        assert_eq!(TpuCoreTypeEnum::SparseCore.as_raw(), 1);
        assert_eq!(TpuCoreTypeEnum::BarnaCore.as_raw(), 2);
        assert_eq!(TpuCoreTypeEnum::TensorCore.to_string(), "TensorCore");
        assert_eq!(TpuCoreTypeEnum::SparseCore.to_string(), "SparseCore");
        assert_eq!(TpuCoreTypeEnum::BarnaCore.to_string(), "BarnaCore");
        assert_ne!(format!("{:?}", TpuCoreTypeEnum::TensorCore), "");
        assert_eq!(i32::from(TpuCoreTypeEnum::SparseCore), 1);

        let c1 = ChipCoordinates::new(1, 2, 3);
        let c2 = ChipCoordinates::new(1, 2, 3);
        let c3 = ChipCoordinates::new(1, 2, 4);
        let c4 = ChipCoordinates::new(1, 9, 3);
        let c5 = ChipCoordinates::new(9, 2, 3);
        assert!(c1.matches(c2));
        assert!(!c1.matches(c3));
        assert!(!c1.matches(c4));
        assert!(!c1.matches(c5));
        assert_eq!(c1.manhattan_distance(c3), 1);
        let bounds = ChipCoordinates::new(4, 4, 4);
        assert_eq!(c1.row_major_index(bounds), 27);
        let far = ChipCoordinates::new(3, 3, 3);
        let origin = ChipCoordinates::new(0, 0, 0);
        assert_eq!(origin.torus_distance(far, bounds, (true, true, true)), 3);
        assert_eq!(origin.torus_distance(c1, bounds, (true, true, true)), 4);
        assert_eq!(origin.torus_distance(far, bounds, (false, false, false)), 9);
        assert_eq!(c1.to_string(), "(1, 2, 3)");
        assert_ne!(format!("{c1:?}"), "");
        let coord_bytes = bytemuck::bytes_of(&c1);
        assert_eq!(coord_bytes.len(), 12);
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
                "v4lite",
                TpuGeneration::V4Lite,
                16,
                1,
                0,
                128,
                0x005e,
                16,
                1,
                128,
                900,
                138,
                4,
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
                let _ = g.cmem_capacity_bytes();
                assert_eq!(g.mxu_dimension(), mxu);
                assert_eq!(g.hbm_bandwidth_gbps(), bw);
                assert_eq!(g.peak_bf16_tflops(), tflops);
                assert_eq!(g.ici_link_count(), ici);
                assert_eq!(g.logical_devices_per_chip(), devs);
                assert_eq!(g.sparse_cores_per_device(), sc);
                assert_eq!(g.sparse_cores_per_chip(), sc);
                let _ = g.tensor_cores_per_chip();
                let _ = g.mxu_count_per_core();
                assert_eq!(g.mxu_systolic_dim(), (mxu, mxu));
                assert_eq!(g.vpu_lanes_and_sublanes(), (128, 8));
                let _ = g.hbm_capacity_bytes_per_chip();
                assert_eq!(g.vmem_capacity_bytes_per_core(), g.vmem_capacity_bytes());
                assert_eq!(g.smem_capacity_bytes_per_core(), g.smem_capacity_bytes());
                let _ = g.default_minor_to_major_tile();
                assert_eq!(g.xla_pad_size(), pad);
                assert_ne!(g.to_string(), "");
                assert_ne!(format!("{g:?}"), "");
            }
        }
        assert!(TpuGeneration::from_name("unknown_tpu").is_err());
        assert_eq!(
            TpuVersionEnum::try_from(4_i32).ok(),
            Some(TpuGeneration::V4)
        );
        assert_eq!(i32::from(TpuGeneration::V7x), 8);
    }
}
