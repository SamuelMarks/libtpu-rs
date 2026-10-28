//! Linux `sysfs` `PCIe` hardware discovery and TPU topology descriptors.
//!
//! Provides [`PciDeviceEnumerator`], [`DeviceSpec`], and [`TopologySpec`].

use crate::types::{
    ChipCoordinates, ChipId, CoreId, DeviceId, HostId, LocalHardwareId, PciDeviceId, PciVendorId,
    ProcessIndex, TpuGeneration,
};

/// Kernel driver mode used by [`crate::hal::LinuxDeviceHal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDriverMode {
    /// Legacy `/dev/accel*` Gasket kernel driver.
    Gasket,
    /// Modern `/dev/vfio/vfio` user-space `PCIe` IOMMU driver.
    Vfio,
}

/// Type alias matching the `sysfs` `PCIe` scanner name (`PciSysfsScanner`).
pub type PciSysfsScanner = PciDeviceEnumerator;

/// Discovered `PCIe` Google Cloud TPU device descriptor from `/sys/bus/pci/devices/*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredPciTpuDevice {
    /// `PCIe` BDF slot name (e.g., `"0000:00:04.0"`).
    pub bdf_slot: String,
    /// `PCIe` Vendor ID (`0x1ae0`).
    pub vendor_id: PciVendorId,
    /// `PCIe` Device ID (`0x0027`, `0x0056`, `0x005e`, `0x0062`, `0x0063`, `0x006f`, `0x0076`).
    pub device_id: PciDeviceId,
    /// Detected TPU architecture generation.
    pub generation: TpuGeneration,
    /// Detected kernel driver binding mode (`Gasket` or `Vfio`).
    pub driver_mode: KernelDriverMode,
    /// Device node path (e.g., `"/dev/accel0"` or `"/dev/vfio/0"`).
    pub device_node_path: String,
}

/// Linux `sysfs` `PCIe` hardware enumerator for Google Cloud TPU ASICs (`0x1ae0`).
#[derive(Debug, Clone, Copy, Default)]
pub struct PciDeviceEnumerator;

impl PciDeviceEnumerator {
    /// Maps a Google `PCIe` Device ID (`0x1ae0:<device_id>`) to its [`TpuGeneration`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - 16-bit `PCIe` Device ID.
    #[must_use]
    pub const fn generation_from_pci_device_id(device_id: PciDeviceId) -> Option<TpuGeneration> {
        match device_id.0 {
            0x0027 => Some(TpuGeneration::V3),
            0x0056 => Some(TpuGeneration::V4),
            0x005e | 0x0062 => Some(TpuGeneration::V5p),
            0x0063 => Some(TpuGeneration::V5e),
            0x006f => Some(TpuGeneration::V6e),
            0x0076 => Some(TpuGeneration::V7x),
            _ => None,
        }
    }

    /// Parses a hexadecimal `sysfs` attribute string such as `"0x1ae0"` into a `u16`.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw contents of a `sysfs` `vendor` or `device` file.
    #[must_use]
    pub fn parse_hex_u16(raw: &str) -> Option<u16> {
        let trimmed = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
        u16::from_str_radix(trimmed, 16).ok()
    }

    /// Scans a `sysfs` PCI devices directory (e.g., `/sys/bus/pci/devices`) and `/dev` root to
    /// discover all attached Google Cloud TPU `PCIe` functions.
    ///
    /// # Arguments
    ///
    /// * `sysfs_pci_dir` - Path to `/sys/bus/pci/devices`.
    /// * `dev_root` - Path to `/dev`.
    #[must_use]
    pub fn scan_sysfs(
        sysfs_pci_dir: &std::path::Path,
        dev_root: &std::path::Path,
    ) -> Vec<DiscoveredPciTpuDevice> {
        let Ok(entries) = std::fs::read_dir(sysfs_pci_dir) else {
            return Vec::new();
        };
        let mut slot_paths: Vec<std::path::PathBuf> =
            entries.filter_map(|e| e.ok().map(|d| d.path())).collect();
        slot_paths.sort();

        slot_paths
            .into_iter()
            .enumerate()
            .filter_map(|(idx, path)| {
                let vendor_str = std::fs::read_to_string(path.join("vendor")).unwrap_or_default();
                let device_str = std::fs::read_to_string(path.join("device")).unwrap_or_default();
                let vendor_raw = Self::parse_hex_u16(&vendor_str).filter(|&v| v == 0x1ae0)?;
                let device_raw = Self::parse_hex_u16(&device_str)?;
                let pci_dev_id = PciDeviceId(device_raw);
                let generation = Self::generation_from_pci_device_id(pci_dev_id)?;
                let bdf_slot = path
                    .file_name()
                    .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
                let (driver_mode, device_node_path) = if path.join("vfio").exists() {
                    (
                        KernelDriverMode::Vfio,
                        dev_root
                            .join("vfio")
                            .join(idx.to_string())
                            .to_string_lossy()
                            .into_owned(),
                    )
                } else {
                    (
                        KernelDriverMode::Gasket,
                        dev_root
                            .join(format!("accel{idx}"))
                            .to_string_lossy()
                            .into_owned(),
                    )
                };
                Some(DiscoveredPciTpuDevice {
                    bdf_slot,
                    vendor_id: PciVendorId(vendor_raw),
                    device_id: pci_dev_id,
                    generation,
                    driver_mode,
                    device_node_path,
                })
            })
            .collect()
    }

    /// Convenience wrapper discovering TPU devices from a virtual `sysfs` PCI devices directory.
    ///
    /// # Arguments
    ///
    /// * `sysfs_pci_dir` - Path to `/sys/bus/pci/devices` (or virtual test directory).
    #[must_use]
    pub fn discover_from_sysfs_root(
        sysfs_pci_dir: &std::path::Path,
    ) -> Vec<DiscoveredPciTpuDevice> {
        Self::scan_sysfs(sysfs_pci_dir, std::path::Path::new("/dev"))
    }
}

/// Specification for a single logical TPU device in a discovered topology.
#[derive(Debug, Clone)]
pub struct DeviceSpec {
    /// Global logical device identifier.
    pub device_id: DeviceId,
    /// Physical chip identifier.
    pub chip_id: ChipId,
    /// Core index on the physical chip.
    pub core_on_chip: CoreId,
    /// Host identifier owning this chip.
    pub host_id: HostId,
    /// PJRT process index owning this device.
    pub process_index: ProcessIndex,
    /// Local hardware ordinal on the host.
    pub local_hardware_id: LocalHardwareId,
    /// 3D torus coordinates `(x, y, z)` of the chip.
    pub chip_coords: ChipCoordinates,
    /// 3D torus coordinates `(x, y, z)` of the host.
    pub host_coords: ChipCoordinates,
    /// Whether this device is directly addressable by the current process.
    pub is_addressable: bool,
    /// TPU hardware generation.
    pub generation: TpuGeneration,
}

/// Complete TPU slice topology specification discovered by a [`crate::hal::TpuHal`].
#[derive(Debug, Clone)]
pub struct TopologySpec {
    /// Platform identifier (`"tpu"`).
    pub platform_name: String,
    /// Platform human-readable version string.
    pub platform_version: String,
    /// Hardware generation of chips in the topology.
    pub generation: TpuGeneration,
    /// 3D mesh bounds of chips across the entire slice.
    pub chip_bounds: ChipCoordinates,
    /// 3D mesh bounds of hosts across the slice.
    pub host_bounds: ChipCoordinates,
    /// Number of physical chips per host.
    pub chips_per_host: u32,
    /// Ordered list of all logical devices in the slice.
    pub devices: Vec<DeviceSpec>,
}
