//! Strongly-typed memory addresses, byte sizes, alignments, and `PCIe` / kernel driver primitives.
//!
//! Provides type-safe representations for HBM, VMEM, SMEM, CMEM, IMEM, SPMEM, IOVA, and BAR MMIO
//! addresses, byte/granule/alignment sizes, `PCIe` Bus:Device.Function slots, and Linux VFIO/Gasket
//! file descriptors and `ioctl` commands.

use derive_more::{Add, Display, From, Into, Sub};

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

/// On-chip Vector Memory (VMEM) byte offset (`VmemAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct VmemAddress(pub u32);

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

/// Physical or BAR-relative Control and Status Register byte offset (`CsrAddress`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct CsrAddress(pub u64);

/// `PCIe` Base Address Register (BAR) MMIO byte offset (`MmioOffset`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#x}")]
pub struct MmioOffset(pub u64);

/// Type alias for [`DeviceAddress`] representing an HBM byte offset (`HbmOffset`).
pub type HbmOffset = DeviceAddress;

/// Type alias for [`IovaAddress`] representing an IOMMU IOVA address (`IovaAddr`).
pub type IovaAddr = IovaAddress;

/// Type alias for [`VmemAddress`] representing a VMEM byte offset (`VmemOffset`).
pub type VmemOffset = VmemAddress;

/// Type alias for [`SmemAddress`] representing an SMEM byte offset (`SmemOffset`).
pub type SmemOffset = SmemAddress;

/// Type alias for [`CmemAddress`] representing a CMEM byte offset (`CmemOffset`).
pub type CmemOffset = CmemAddress;

/// Type alias for [`MmioOffset`] representing a `PCIe` BAR byte offset (`BarOffset`).
pub type BarOffset = MmioOffset;

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

/// Type alias for [`AlignmentBytes`] (`Alignment`).
pub type Alignment = AlignmentBytes;

/// Size of a tensor dimension (`DimensionSize`).
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

/// `PCIe` configuration space Revision ID (`PciRevisionId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0:#04x}")]
pub struct PciRevisionId(pub u8);

/// `PCIe` Base Address Register index (`BarIndex`).
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

        let b_pc = BundlePc::from(0x20_u32) + BundlePc(1) - BundlePc(1);
        assert_eq!(u32::from(b_pc), 0x20);
        assert_eq!(b_pc.to_string(), "0x20");

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
    }
}
