//! Strongly-typed memory addresses, byte sizes, alignments, and `PCIe` / kernel driver primitives.
//!
//! Provides type-safe representations for HBM, VMEM, SMEM, CMEM, IMEM, SPMEM, IOVA, and BAR MMIO
//! addresses, byte/granule/alignment sizes, `PCIe` Bus:Device.Function slots, and Linux VFIO/Gasket
//! file descriptors and `ioctl` commands.

use crate::error::{LibTpuError, LibTpuResult};
use derive_more::{Add, Display, From, Into, Sub};

/// Macro implementing checked addition and subtraction returning [`LibTpuResult`] for unsigned/signed newtypes.
macro_rules! impl_checked_newtype_arith {
    ($($ty:ident),+ $(,)?) => {
        $(
            impl $ty {
                /// Checked addition returning [`LibTpuError::NumericOverflow`] on overflow.
                ///
                /// # Arguments
                ///
                /// * `rhs` - Right-hand operand to add.
                ///
                /// # Errors
                ///
                /// Returns [`LibTpuError::NumericOverflow`] if `self.0 + rhs.0` overflows.
                pub const fn checked_add(self, rhs: Self) -> LibTpuResult<Self> {
                    match self.0.checked_add(rhs.0) {
                        Some(v) => Ok(Self(v)),
                        None => Err(LibTpuError::NumericOverflow {
                            context: stringify!($ty),
                        }),
                    }
                }

                /// Checked subtraction returning [`LibTpuError::NumericOverflow`] on underflow.
                ///
                /// # Arguments
                ///
                /// * `rhs` - Right-hand operand to subtract.
                ///
                /// # Errors
                ///
                /// Returns [`LibTpuError::NumericOverflow`] if `self.0 - rhs.0` underflows.
                pub const fn checked_sub(self, rhs: Self) -> LibTpuResult<Self> {
                    match self.0.checked_sub(rhs.0) {
                        Some(v) => Ok(Self(v)),
                        None => Err(LibTpuError::NumericOverflow {
                            context: stringify!($ty),
                        }),
                    }
                }
            }
        )+
    };
}

/// On-device HBM or VMEM virtual memory address (`DeviceAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct DeviceAddress(pub u64);

/// Host virtual memory address (`HostVirtualAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct HostVirtualAddress(pub u64);

/// IOMMU I/O Virtual Address used in VFIO/Gasket page tables (`IovaAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct IovaAddress(pub u64);

/// On-chip Vector Memory (VMEM) byte offset (`VmemAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct VmemAddress(pub u32);

impl VmemAddress {
    /// Aligns `offset` upward to a 256-byte TPU VMEM granule boundary and validates that it fits within `capacity`.
    ///
    /// # Arguments
    ///
    /// * `offset` - Requested VMEM byte offset.
    /// * `capacity` - Total VMEM capacity in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::OutOfDeviceMemory`] if the aligned offset exceeds `capacity`.
    pub const fn new_aligned(offset: u64, capacity: u64) -> LibTpuResult<Self> {
        let aligned = (offset.saturating_add(255)) & !255;
        if aligned > capacity {
            #[allow(clippy::cast_possible_truncation)]
            return Err(LibTpuError::OutOfDeviceMemory {
                requested: aligned as usize,
                available: capacity as usize,
                memory_kind: "vmem",
            });
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok(Self(aligned as u32))
    }
}

/// On-chip Scalar Memory (SMEM) byte offset (`SmemAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct SmemAddress(pub u32);

/// On-chip Collective Memory (CMEM) byte offset (`CmemAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct CmemAddress(pub u32);

/// On-chip Instruction Memory (IMEM) byte offset (`ImemAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct ImemAddress(pub u32);

/// On-chip `SparseCore` Scratchpad Memory (SPMEM) byte offset (`SpmemAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct SpmemAddress(pub u32);

/// On-chip Vector Memory (VMEM) 32-bit word offset (`VmOffset`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct VmOffset(pub u32);

/// On-chip Scalar Memory (SMEM) 32-bit word offset (`SmOffset`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct SmOffset(pub u32);

/// `TensorCore` VLIW instruction bundle Program Counter (`BundlePc`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct BundlePc(pub u32);

/// Zero-based index of a 64-byte VLIW bundle in Instruction Memory (`ImemBundleIndex`).
pub type ImemBundleIndex = BundlePc;

/// Type alias for [`BundlePc`] representing a 64-byte bundle program counter index (`ImemBundlePc`).
pub type ImemBundlePc = BundlePc;

/// 64-bit Vector Memory (`VMEM`) byte offset (`VmemByteOffset`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct VmemByteOffset(pub u64);

impl VmemByteOffset {
    /// Aligns `offset` upward to a 256-byte TPU VMEM granule boundary and validates that it fits within `capacity`.
    ///
    /// # Arguments
    ///
    /// * `offset` - Requested VMEM byte offset.
    /// * `capacity` - Total VMEM capacity in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::OutOfDeviceMemory`] if the aligned offset exceeds `capacity`.
    pub const fn new_aligned(offset: u64, capacity: u64) -> LibTpuResult<Self> {
        let aligned = (offset.saturating_add(255)) & !255;
        if aligned > capacity {
            #[allow(clippy::cast_possible_truncation)]
            return Err(LibTpuError::OutOfDeviceMemory {
                requested: aligned as usize,
                available: capacity as usize,
                memory_kind: "vmem",
            });
        }
        Ok(Self(aligned))
    }
}

/// 64-bit Scalar Memory (`SMEM`) byte offset (`SmemByteOffset`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct SmemByteOffset(pub u64);

/// 64-bit Collective Memory (`CMEM`) byte offset (`CmemByteOffset`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct CmemByteOffset(pub u64);

/// Type alias for [`DeviceAddress`] representing an HBM byte offset (`HbmByteOffset`).
pub type HbmByteOffset = DeviceAddress;

/// 8-bit or 16-bit sublane execution mask (`SublaneMask`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#04x}")]
pub struct SublaneMask(pub u8);

/// Scalar or vector element bit width (`BitWidth`, e.g., `2`, `4`, `8`, `16`, `32`, `64`, `128`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}b")]
pub struct BitWidth(pub u16);

/// Hardware matrix/vector sublane count (`SublaneCount`, typically `8` or `16`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct SublaneCount(pub u16);

/// Hardware matrix/vector lane count (`LaneCount`, typically `128` or `256`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct LaneCount(pub u16);

/// Physical or BAR-relative Control and Status Register byte offset (`CsrAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct CsrAddress(pub u64);

/// `PCIe` Base Address Register (BAR) MMIO byte offset (`MmioOffset`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct MmioOffset(pub u64);

/// Type alias for [`DeviceAddress`] representing an HBM byte offset (`HbmOffset`).
pub type HbmOffset = DeviceAddress;

/// Type alias for [`DeviceAddress`] representing an on-device HBM virtual address (`HbmAddress`).
pub type HbmAddress = DeviceAddress;

/// Type alias for [`IovaAddress`] representing an IOMMU IOVA address (`IovaAddr`).
pub type IovaAddr = IovaAddress;

/// Type alias for [`VmemAddress`] representing a VMEM byte offset (`VmemOffset`).
pub type VmemOffset = VmemAddress;

/// Type alias for [`SmemAddress`] representing an SMEM byte offset (`SmemOffset`).
pub type SmemOffset = SmemAddress;

/// Type alias for [`CmemAddress`] representing a CMEM byte offset (`CmemOffset`).
pub type CmemOffset = CmemAddress;

/// Type alias for [`SpmemAddress`] representing a `SparseCore` scratchpad memory offset (`SpMemAddress`).
pub type SpMemAddress = SpmemAddress;

/// Type alias for [`BundlePc`] representing an Instruction Memory program counter (`ImemPc`).
pub type ImemPc = BundlePc;

/// Type alias for [`MmioOffset`] representing a `PCIe` BAR byte offset (`BarOffset`).
pub type BarOffset = MmioOffset;

/// Size in bytes (`ByteSize`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ByteSize(pub usize);

/// DMA transfer granule size in bytes (`GranuleSize`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct GranuleSize(pub usize);

/// Memory alignment boundary in bytes (`AlignmentBytes`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct AlignmentBytes(pub usize);

/// Type alias for [`AlignmentBytes`] (`Alignment`).
pub type Alignment = AlignmentBytes;

/// Size of a tensor dimension (`DimensionSize`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct DimensionSize(pub i64);

/// Zero-based tensor dimension axis index (`DimensionIndex`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct DimensionIndex(pub usize);

/// Validates C ABI structure byte size (`StructSize`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct StructSize(pub usize);

/// `PCIe` Bus:Device.Function address (`BdfAddress`, e.g., `0000:00:04.0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[display("{domain:04x}:{bus:02x}:{device:02x}.{function}")]
pub struct BdfAddress {
    /// 16-bit PCI domain number (`0x0000`..`0xffff`).
    pub domain: u16,
    /// 8-bit PCI bus number (`0x00`..`0xff`).
    pub bus: u8,
    /// 5-bit PCI device slot number (`0x00`..`0x1f`).
    pub device: u8,
    /// 3-bit PCI function number (`0`..`7`).
    pub function: u8,
}

impl BdfAddress {
    /// Constructs a new [`BdfAddress`] from domain, bus, device, and function numbers.
    ///
    /// # Arguments
    ///
    /// * `domain` - 16-bit PCI domain number.
    /// * `bus` - 8-bit PCI bus number.
    /// * `device` - 5-bit PCI device slot number.
    /// * `function` - 3-bit PCI function number.
    #[must_use]
    pub const fn new(domain: u16, bus: u8, device: u8, function: u8) -> Self {
        Self {
            domain,
            bus,
            device,
            function,
        }
    }

    /// Parses a `sysfs` BDF string such as `"0000:00:04.0"` into a [`BdfAddress`].
    ///
    /// # Arguments
    ///
    /// * `raw` - BDF slot string in `DDDD:BB:DD.F` format.
    #[must_use]
    pub fn parse_bdf(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        let (domain_str, rest) = trimmed.split_once(':')?;
        let (bus_str, dev_fn_str) = rest.split_once(':')?;
        let (dev_str, fn_str) = dev_fn_str.split_once('.')?;
        let domain = u16::from_str_radix(domain_str, 16).ok()?;
        let bus = u8::from_str_radix(bus_str, 16).ok()?;
        let device = u8::from_str_radix(dev_str, 16).ok()?;
        let function = u8::from_str_radix(fn_str, 16).ok()?;
        Some(Self::new(domain, bus, device, function))
    }
}

/// Type alias for [`BdfAddress`] (`PciBdfSlot`).
pub type PciBdfSlot = BdfAddress;

/// Type alias for [`BdfAddress`] (`PciBdfAddress`).
pub type PciBdfAddress = BdfAddress;

/// Packed 32-bit `PCIe` `(domain:16 | bus:8 | device:5 | function:3)` identifier (`PcieDomainBusDevFn`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into, Add, Sub)]
pub struct PcieDomainBusDevFn(pub u32);

impl PcieDomainBusDevFn {
    /// Packs `domain`, `bus`, `device` (`0..=31`), and `function` (`0..=7`) into a 32-bit [`PcieDomainBusDevFn`].
    ///
    /// # Arguments
    ///
    /// * `domain` - 16-bit PCI domain number.
    /// * `bus` - 8-bit PCI bus number.
    /// * `device` - 5-bit PCI device slot (`0..=31`).
    /// * `function` - 3-bit PCI function (`0..=7`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `device > 31` or `function > 7`.
    pub const fn new_checked(domain: u16, bus: u8, device: u8, function: u8) -> LibTpuResult<Self> {
        if device > 31 || function > 7 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "PcieDomainBusDevFn",
                reason: String::new(),
            });
        }
        let packed = ((domain as u32) << 16)
            | ((bus as u32) << 8)
            | (((device & 0x1F) as u32) << 3)
            | ((function & 0x07) as u32);
        Ok(Self(packed))
    }

    /// Parses a `sysfs` `"DDDD:BB:DD.F"` string into a [`PcieDomainBusDevFn`].
    ///
    /// # Arguments
    ///
    /// * `raw` - `sysfs` BDF address string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `raw` is malformed or out of range.
    pub fn parse_bdf(raw: &str) -> LibTpuResult<Self> {
        let bdf = BdfAddress::parse_bdf(raw).ok_or_else(|| LibTpuError::InvalidArgument {
            parameter: "PcieDomainBusDevFn",
            reason: format!("Invalid PCIe BDF '{raw}'"),
        })?;
        Self::new_checked(bdf.domain, bdf.bus, bdf.device, bdf.function)
    }

    /// Returns the 16-bit PCI domain number.
    #[must_use]
    pub const fn domain(self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// Returns the 8-bit PCI bus number.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn bus(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }

    /// Returns the 5-bit PCI device slot (`0..=31`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn device(self) -> u8 {
        ((self.0 >> 3) & 0x1F) as u8
    }

    /// Returns the 3-bit PCI function (`0..=7`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn function(self) -> u8 {
        (self.0 & 0x07) as u8
    }
}

impl std::fmt::Display for PcieDomainBusDevFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{}",
            self.domain(),
            self.bus(),
            self.device(),
            self.function()
        )
    }
}

/// `PCIe` configuration space Revision ID (`PciRevisionId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#04x}")]
pub struct PciRevisionId(pub u8);

/// `PCIe` Base Address Register index (`BarIndex`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct BarIndex(pub u32);

/// Hardware Gasket IOMMU page table index (`PageTableIndex`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct PageTableIndex(pub u32);

/// Hardware MSI-X interrupt vector index (`InterruptVector`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct InterruptVector(pub u32);

/// POSIX file descriptor for `/dev/accel*` or `/dev/vfio/*` (`FileDescriptor`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0}")]
pub struct FileDescriptor(pub i32);

/// Linux `eventfd` interrupt notification descriptor (`EventFdHandle`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0}")]
pub struct EventFdHandle(pub i32);

/// Linux `dma_buf` file descriptor for zero-copy HBM/PCIe peer-to-peer DMA (`DmaBufFd`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0}")]
pub struct DmaBufFd(pub i32);

/// Kernel `ioctl` request command code (`IoctlCommand`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0:#x}")]
pub struct IoctlCommand(pub u64);

/// `PCIe` Vendor ID (`PciVendorId`, `0x1ae0` for Google LLC).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0:#06x}")]
pub struct PciVendorId(pub u16);

/// Type alias for [`PciVendorId`] (`PcieVendorId`).
pub type PcieVendorId = PciVendorId;

/// `PCIe` Device ID (`PciDeviceId`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0:#06x}")]
pub struct PciDeviceId(pub u16);

/// Type alias for [`PciDeviceId`] (`PcieDeviceId`).
pub type PcieDeviceId = PciDeviceId;

/// Linux VFIO IOMMU group identifier (`IommuGroupId`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0}")]
pub struct IommuGroupId(pub u32);

impl_checked_newtype_arith!(
    DeviceAddress,
    HostVirtualAddress,
    IovaAddress,
    VmemAddress,
    SmemAddress,
    CmemAddress,
    ImemAddress,
    SpmemAddress,
    BundlePc,
    SublaneCount,
    LaneCount,
    CsrAddress,
    MmioOffset,
    ByteSize,
    BarIndex,
    PcieDomainBusDevFn,
);

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_addresses_sizes_and_pci_primitives() {
        let addr: HbmOffset =
            DeviceAddress::from(0x1000_u64) + DeviceAddress(0x10) - DeviceAddress(0x10);
        assert_eq!(u64::from(addr), 0x1000);
        assert_eq!(addr.to_string(), "0x1000");

        let hva =
            HostVirtualAddress::from(0x2000_u64) + HostVirtualAddress(8) - HostVirtualAddress(8);
        assert_eq!(u64::from(hva), 0x2000);
        assert_eq!(hva.to_string(), "0x2000");

        let iova: IovaAddr = IovaAddress::from(0x3000_u64) + IovaAddress(8) - IovaAddress(8);
        assert_eq!(u64::from(iova), 0x3000);
        assert_eq!(iova.to_string(), "0x3000");

        let vmem_addr: VmemOffset = VmemAddress::from(0x400_u32) + VmemAddress(4) - VmemAddress(4);
        assert_eq!(u32::from(vmem_addr), 0x400);
        assert_eq!(vmem_addr.to_string(), "0x400");

        let smem_addr: SmemOffset = SmemAddress::from(0x80_u32) + SmemAddress(4) - SmemAddress(4);
        assert_eq!(u32::from(smem_addr), 0x80);
        assert_eq!(smem_addr.to_string(), "0x80");

        let cmem_addr: CmemOffset = CmemAddress::from(0x1000_u32) + CmemAddress(4) - CmemAddress(4);
        assert_eq!(u32::from(cmem_addr), 0x1000);
        assert_eq!(cmem_addr.to_string(), "0x1000");

        let imem_addr = ImemAddress::from(0x2000_u32) + ImemAddress(4) - ImemAddress(4);
        assert_eq!(u32::from(imem_addr), 0x2000);
        assert_eq!(imem_addr.to_string(), "0x2000");

        let scratch_addr = SpmemAddress::from(0x3000_u32) + SpmemAddress(4) - SpmemAddress(4);
        assert_eq!(u32::from(scratch_addr), 0x3000);
        assert_eq!(scratch_addr.to_string(), "0x3000");

        let vm_off = VmOffset::from(0x100_u32) + VmOffset(4) - VmOffset(4);
        assert_eq!(u32::from(vm_off), 0x100);
        assert_eq!(vm_off.to_string(), "0x100");

        let sm_off = SmOffset::from(0x40_u32) + SmOffset(4) - SmOffset(4);
        assert_eq!(u32::from(sm_off), 0x40);
        assert_eq!(sm_off.to_string(), "0x40");

        let b_pc: ImemBundleIndex = BundlePc::from(0x20_u32) + BundlePc(1) - BundlePc(1);
        assert_eq!(u32::from(b_pc), 0x20);
        assert_eq!(b_pc.to_string(), "0x20");

        let bw = BitWidth::from(16_u16) + BitWidth(8) - BitWidth(8);
        assert_eq!(u16::from(bw), 16);
        assert_eq!(bw.to_string(), "16b");

        let sublanes = SublaneCount::from(8_u16) + SublaneCount(2) - SublaneCount(2);
        assert_eq!(u16::from(sublanes), 8);
        assert_eq!(sublanes.to_string(), "8");

        let lanes = LaneCount::from(128_u16) + LaneCount(16) - LaneCount(16);
        assert_eq!(u16::from(lanes), 128);
        assert_eq!(lanes.to_string(), "128");

        let csr_addr = CsrAddress::from(0x3000_u64) + CsrAddress(8) - CsrAddress(8);
        assert_eq!(u64::from(csr_addr), 0x3000);
        assert_eq!(csr_addr.to_string(), "0x3000");

        let mmio: BarOffset = MmioOffset::from(0x400_u64) + MmioOffset(4) - MmioOffset(4);
        assert_eq!(u64::from(mmio), 0x400);
        assert_eq!(mmio.to_string(), "0x400");

        let bsize = ByteSize::from(512_usize) + ByteSize(64) - ByteSize(64);
        assert_eq!(usize::from(bsize), 512);
        assert_eq!(bsize.to_string(), "512");

        let gsize = GranuleSize::from(256_usize) + GranuleSize(16) - GranuleSize(16);
        assert_eq!(usize::from(gsize), 256);
        assert_eq!(gsize.to_string(), "256");

        let align: Alignment =
            AlignmentBytes::from(4096_usize) + AlignmentBytes(16) - AlignmentBytes(16);
        assert_eq!(usize::from(align), 4096);
        assert_eq!(align.to_string(), "4096");

        let dim = DimensionSize::from(128_i64) + DimensionSize(1) - DimensionSize(1);
        assert_eq!(i64::from(dim), 128);
        assert_eq!(dim.to_string(), "128");

        let dim_idx = DimensionIndex::from(2_usize) + DimensionIndex(1) - DimensionIndex(1);
        assert_eq!(usize::from(dim_idx), 2);
        assert_eq!(dim_idx.to_string(), "2");

        let st_sz = StructSize::from(32_usize) + StructSize(8) - StructSize(8);
        assert_eq!(usize::from(st_sz), 32);
        assert_eq!(st_sz.to_string(), "32");

        let bdf = BdfAddress::parse_bdf("0000:00:04.0");
        assert_eq!(bdf, Some(BdfAddress::new(0, 0, 4, 0)));
        assert_eq!(BdfAddress::new(0, 0, 4, 0).to_string(), "0000:00:04.0");
        assert_eq!(BdfAddress::parse_bdf("invalid"), None);
        assert_eq!(BdfAddress::parse_bdf("0000:00"), None);
        assert_eq!(BdfAddress::parse_bdf("0000:00:04"), None);
        assert_eq!(BdfAddress::parse_bdf("zzzz:00:04.0"), None);
        assert_eq!(BdfAddress::parse_bdf("0000:zz:04.0"), None);
        assert_eq!(BdfAddress::parse_bdf("0000:00:zz.0"), None);
        assert_eq!(BdfAddress::parse_bdf("0000:00:04.z"), None);

        let bdf_slot: PciBdfSlot = BdfAddress::new(0, 0, 4, 0);
        let bdf_alias: PciBdfAddress = bdf_slot;
        assert_eq!(bdf_alias.device, 4);

        let pci_revision = PciRevisionId::from(1_u8) + PciRevisionId(1) - PciRevisionId(1);
        assert_eq!(u8::from(pci_revision), 1);
        assert_eq!(pci_revision.to_string(), "0x01");

        let bar = BarIndex::from(2_u32) + BarIndex(1) - BarIndex(1);
        assert_eq!(u32::from(bar), 2);
        assert_eq!(bar.to_string(), "2");

        let pt_idx = PageTableIndex::from(7_u32) + PageTableIndex(1) - PageTableIndex(1);
        assert_eq!(u32::from(pt_idx), 7);
        assert_eq!(pt_idx.to_string(), "7");

        let irq_vec = InterruptVector::from(3_u32) + InterruptVector(1) - InterruptVector(1);
        assert_eq!(u32::from(irq_vec), 3);
        assert_eq!(irq_vec.to_string(), "3");

        let fd = FileDescriptor::from(3_i32);
        assert_eq!(i32::from(fd), 3);
        assert_eq!(fd.to_string(), "3");

        let efd = EventFdHandle::from(4_i32);
        assert_eq!(i32::from(efd), 4);
        assert_eq!(efd.to_string(), "4");

        let dmabuf = DmaBufFd::from(5_i32);
        assert_eq!(i32::from(dmabuf), 5);
        assert_eq!(dmabuf.to_string(), "5");

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

        let hbm_addr: HbmAddress = DeviceAddress::from(0x1000_u64);
        assert_eq!(u64::from(hbm_addr), 0x1000);

        let vbo = VmemByteOffset::from(0x800_u64) + VmemByteOffset(16) - VmemByteOffset(16);
        assert_eq!(u64::from(vbo), 0x800);
        assert_eq!(vbo.to_string(), "0x800");

        let sbo = SmemByteOffset::from(0x100_u64) + SmemByteOffset(8) - SmemByteOffset(8);
        assert_eq!(u64::from(sbo), 0x100);
        assert_eq!(sbo.to_string(), "0x100");

        let cbo = CmemByteOffset::from(0x2000_u64) + CmemByteOffset(32) - CmemByteOffset(32);
        assert_eq!(u64::from(cbo), 0x2000);
        assert_eq!(cbo.to_string(), "0x2000");

        let hbo: HbmByteOffset = DeviceAddress::from(0x4000_u64);
        assert_eq!(u64::from(hbo), 0x4000);

        let ipc: ImemBundlePc = BundlePc::from(7_u32);
        assert_eq!(u32::from(ipc), 7);

        let smask = SublaneMask::from(0xFF_u8) + SublaneMask(0) - SublaneMask(0);
        assert_eq!(u8::from(smask), 0xFF);
        assert_eq!(smask.to_string(), "0xff");

        assert_eq!(VmemAddress::new_aligned(10, 1024), Ok(VmemAddress(256)));
        assert!(VmemAddress::new_aligned(2048, 1024).is_err());
        assert_eq!(
            VmemByteOffset::new_aligned(300, 1024),
            Ok(VmemByteOffset(512))
        );
        assert!(VmemByteOffset::new_aligned(2048, 1024).is_err());

        let pcie_bdf = PcieDomainBusDevFn::parse_bdf("0001:03:04.2");
        assert_eq!(pcie_bdf, PcieDomainBusDevFn::new_checked(1, 3, 4, 2));
        if let Ok(pb) = pcie_bdf {
            assert_eq!(
                (pb.domain(), pb.bus(), pb.device(), pb.function()),
                (1, 3, 4, 2)
            );
            assert_eq!(pb.to_string(), "0001:03:04.2");
            assert_eq!(u32::from(pb), pb.0);
            assert_eq!(
                PcieDomainBusDevFn::from(pb.0) + PcieDomainBusDevFn(0) - PcieDomainBusDevFn(0),
                pb
            );
        }
        assert!(PcieDomainBusDevFn::new_checked(0, 0, 32, 0).is_err());
        assert!(PcieDomainBusDevFn::new_checked(0, 0, 0, 8).is_err());
        assert!(PcieDomainBusDevFn::parse_bdf("bad").is_err());

        let vendor_id: PcieVendorId = PciVendorId(0x1ae0);
        let device_id: PcieDeviceId = PciDeviceId(0x006f);
        let sp_mem: SpMemAddress = SpmemAddress(0x400);
        let imem_pc: ImemPc = BundlePc(12);
        assert_eq!(
            (vendor_id.0, device_id.0, sp_mem.0, imem_pc.0),
            (0x1ae0, 0x006f, 0x400, 12)
        );

        assert_eq!(
            DeviceAddress(10).checked_add(DeviceAddress(5)),
            Ok(DeviceAddress(15))
        );
        assert!(
            DeviceAddress(u64::MAX)
                .checked_add(DeviceAddress(1))
                .is_err()
        );
        assert_eq!(
            DeviceAddress(10).checked_sub(DeviceAddress(5)),
            Ok(DeviceAddress(5))
        );
        assert!(DeviceAddress(0).checked_sub(DeviceAddress(1)).is_err());

        assert_eq!(
            HostVirtualAddress(2).checked_add(HostVirtualAddress(3)),
            Ok(HostVirtualAddress(5))
        );
        assert!(
            HostVirtualAddress(u64::MAX)
                .checked_add(HostVirtualAddress(1))
                .is_err()
        );
        assert_eq!(
            HostVirtualAddress(5).checked_sub(HostVirtualAddress(3)),
            Ok(HostVirtualAddress(2))
        );
        assert!(
            HostVirtualAddress(0)
                .checked_sub(HostVirtualAddress(1))
                .is_err()
        );

        assert_eq!(
            IovaAddress(2).checked_add(IovaAddress(3)),
            Ok(IovaAddress(5))
        );
        assert!(IovaAddress(u64::MAX).checked_add(IovaAddress(1)).is_err());
        assert_eq!(
            IovaAddress(5).checked_sub(IovaAddress(3)),
            Ok(IovaAddress(2))
        );
        assert!(IovaAddress(0).checked_sub(IovaAddress(1)).is_err());

        assert_eq!(
            VmemAddress(2).checked_add(VmemAddress(3)),
            Ok(VmemAddress(5))
        );
        assert!(VmemAddress(u32::MAX).checked_add(VmemAddress(1)).is_err());
        assert_eq!(
            VmemAddress(5).checked_sub(VmemAddress(3)),
            Ok(VmemAddress(2))
        );
        assert!(VmemAddress(0).checked_sub(VmemAddress(1)).is_err());

        assert_eq!(
            SmemAddress(2).checked_add(SmemAddress(3)),
            Ok(SmemAddress(5))
        );
        assert!(SmemAddress(u32::MAX).checked_add(SmemAddress(1)).is_err());
        assert_eq!(
            SmemAddress(5).checked_sub(SmemAddress(3)),
            Ok(SmemAddress(2))
        );
        assert!(SmemAddress(0).checked_sub(SmemAddress(1)).is_err());

        assert_eq!(
            CmemAddress(2).checked_add(CmemAddress(3)),
            Ok(CmemAddress(5))
        );
        assert!(CmemAddress(u32::MAX).checked_add(CmemAddress(1)).is_err());
        assert_eq!(
            CmemAddress(5).checked_sub(CmemAddress(3)),
            Ok(CmemAddress(2))
        );
        assert!(CmemAddress(0).checked_sub(CmemAddress(1)).is_err());

        assert_eq!(
            ImemAddress(2).checked_add(ImemAddress(3)),
            Ok(ImemAddress(5))
        );
        assert!(ImemAddress(u32::MAX).checked_add(ImemAddress(1)).is_err());
        assert_eq!(
            ImemAddress(5).checked_sub(ImemAddress(3)),
            Ok(ImemAddress(2))
        );
        assert!(ImemAddress(0).checked_sub(ImemAddress(1)).is_err());

        assert_eq!(
            SpmemAddress(2).checked_add(SpmemAddress(3)),
            Ok(SpmemAddress(5))
        );
        assert!(SpmemAddress(u32::MAX).checked_add(SpmemAddress(1)).is_err());
        assert_eq!(
            SpmemAddress(5).checked_sub(SpmemAddress(3)),
            Ok(SpmemAddress(2))
        );
        assert!(SpmemAddress(0).checked_sub(SpmemAddress(1)).is_err());

        assert_eq!(BundlePc(2).checked_add(BundlePc(3)), Ok(BundlePc(5)));
        assert!(BundlePc(u32::MAX).checked_add(BundlePc(1)).is_err());
        assert_eq!(BundlePc(5).checked_sub(BundlePc(3)), Ok(BundlePc(2)));
        assert!(BundlePc(0).checked_sub(BundlePc(1)).is_err());

        assert_eq!(
            SublaneCount(2).checked_add(SublaneCount(3)),
            Ok(SublaneCount(5))
        );
        assert!(SublaneCount(u16::MAX).checked_add(SublaneCount(1)).is_err());
        assert_eq!(
            SublaneCount(5).checked_sub(SublaneCount(3)),
            Ok(SublaneCount(2))
        );
        assert!(SublaneCount(0).checked_sub(SublaneCount(1)).is_err());

        assert_eq!(LaneCount(2).checked_add(LaneCount(3)), Ok(LaneCount(5)));
        assert!(LaneCount(u16::MAX).checked_add(LaneCount(1)).is_err());
        assert_eq!(LaneCount(5).checked_sub(LaneCount(3)), Ok(LaneCount(2)));
        assert!(LaneCount(0).checked_sub(LaneCount(1)).is_err());

        assert_eq!(CsrAddress(2).checked_add(CsrAddress(3)), Ok(CsrAddress(5)));
        assert!(CsrAddress(u64::MAX).checked_add(CsrAddress(1)).is_err());
        assert_eq!(CsrAddress(5).checked_sub(CsrAddress(3)), Ok(CsrAddress(2)));
        assert!(CsrAddress(0).checked_sub(CsrAddress(1)).is_err());

        assert_eq!(MmioOffset(2).checked_add(MmioOffset(3)), Ok(MmioOffset(5)));
        assert!(MmioOffset(u64::MAX).checked_add(MmioOffset(1)).is_err());
        assert_eq!(MmioOffset(5).checked_sub(MmioOffset(3)), Ok(MmioOffset(2)));
        assert!(MmioOffset(0).checked_sub(MmioOffset(1)).is_err());

        assert_eq!(ByteSize(2).checked_add(ByteSize(3)), Ok(ByteSize(5)));
        assert!(ByteSize(usize::MAX).checked_add(ByteSize(1)).is_err());
        assert_eq!(ByteSize(5).checked_sub(ByteSize(3)), Ok(ByteSize(2)));
        assert!(ByteSize(0).checked_sub(ByteSize(1)).is_err());

        assert_eq!(BarIndex(2).checked_add(BarIndex(1)), Ok(BarIndex(3)));
        assert!(BarIndex(u32::MAX).checked_add(BarIndex(1)).is_err());
        assert_eq!(BarIndex(3).checked_sub(BarIndex(1)), Ok(BarIndex(2)));
        assert!(BarIndex(0).checked_sub(BarIndex(1)).is_err());

        assert_eq!(
            PcieDomainBusDevFn(2).checked_add(PcieDomainBusDevFn(1)),
            Ok(PcieDomainBusDevFn(3))
        );
        assert!(
            PcieDomainBusDevFn(u32::MAX)
                .checked_add(PcieDomainBusDevFn(1))
                .is_err()
        );
        assert_eq!(
            PcieDomainBusDevFn(3).checked_sub(PcieDomainBusDevFn(1)),
            Ok(PcieDomainBusDevFn(2))
        );
        assert!(
            PcieDomainBusDevFn(0)
                .checked_sub(PcieDomainBusDevFn(1))
                .is_err()
        );
    }
}
