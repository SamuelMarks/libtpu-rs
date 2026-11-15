//! Linux `sysfs` `PCIe` (`0x1ae0`) device enumerator and BAR0/BAR2 hardware CSR prober.
//!
//! Implements Phase 7.1: scans `/sys/bus/pci/devices/*` for Google Cloud TPU `PCIe` vendor
//! `0x1ae0` across all 8 generations (`v2`, `v3`, `v4`, `v4lite`, `v5e`, `v5p`, `v6e`, `v7x`),
//! resolves IOMMU groups (`/dev/vfio/<group>`), `/dev/accel*` character nodes, NUMA CPU affinity,
//! and BAR0/BAR2/BAR4 physical resource apertures, and reads chip SKU, `GLOBAL_CHIP_ID`, 3D
//! board/tray coordinates, and 6-port ICI link status CSRs from BAR0 MMIO.

use super::{ChipRegisterMap, KernelDriverMode};
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ChipCoordinates, MmioOffset, PciBdfAddress, PciDeviceId, TpuGeneration};
use std::path::Path;

/// Official Google LLC `PCI-SIG` vendor identifier (`0x1ae0`).
pub const GOOGLE_PCI_VENDOR_ID_RAW: u16 = 0x1ae0;

/// `PCIe` Base Address Register (`BAR0`, `BAR2`, `BAR4`) physical aperture descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcieBarResource {
    /// BAR index (`0` = CSR/control, `2` = HBM/command-ring aperture, `4` = MSI-X PBA/table).
    pub bar_index: u8,
    /// Physical host bus start address (`0` if unpopulated).
    pub phys_start: u64,
    /// Physical host bus inclusive end address (`0` if unpopulated).
    pub phys_end: u64,
    /// Raw Linux `IORESOURCE_*` flags (`0x0014_220c` = 64-bit prefetchable `MEM`).
    pub flags: u64,
    /// Total aperture byte length (`0` if unpopulated).
    pub size_bytes: u64,
}

impl PcieBarResource {
    /// Returns `true` if this BAR is populated (`size_bytes > 0`) and has `IORESOURCE_MEM` (`0x200`).
    #[must_use]
    pub const fn is_mmio_populated(&self) -> bool {
        (self.size_bytes > 0) && ((self.flags & 0x0200) != 0)
    }

    /// Returns `true` if `IORESOURCE_PREFETCH` (`0x2000`) is set on this BAR.
    #[must_use]
    pub const fn is_prefetchable(&self) -> bool {
        (self.flags & 0x2000) != 0
    }
}

/// Complete `sysfs`-probed Google Cloud TPU `PCIe` function inventory entry (`SysfsTpuPcieFunction`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SysfsTpuPcieFunction {
    /// Parsed `domain:bus:device.function` `PCIe` address.
    pub bdf: PciBdfAddress,
    /// Raw `PCIe` device ID (`0x0027`, `0x0056`, `0x0063`, `0x0062`, `0x006f`, `0x007f`, `0x0076`).
    pub pci_device_id: PciDeviceId,
    /// `PCIe` silicon revision byte (`0x00` = Jellyfish `v2` on `0x0027`, `>= 0x01` = Dragonfish `v3`).
    pub revision_id: u8,
    /// `PCIe` subsystem device ID (`0x004e` = `V4Lite` on `0x0056`).
    pub subsystem_device_id: u16,
    /// Resolved Cloud TPU hardware generation (`V2..=V7x`).
    pub generation: TpuGeneration,
    /// Bound Linux kernel driver mode (`Vfio` when bound to `vfio-pci`, otherwise `Gasket`).
    pub driver_mode: KernelDriverMode,
    /// Resolved Linux IOMMU group ID (`Some(n)` when `iommu_group` symlink exists).
    pub iommu_group: Option<u32>,
    /// Resolved `/dev/vfio/<group>` or `/dev/accel<N>` character device path.
    pub device_node_path: String,
    /// Host NUMA node index (`-1` when uniform / single-socket).
    pub numa_node: i32,
    /// Local NUMA CPU affinity list string (e.g., `"0-55,112-167"`).
    pub local_cpulist: String,
    /// Negotiated `PCIe` link speed string (e.g., `"32.0 GT/s PCIe"`).
    pub link_speed: String,
    /// Negotiated `PCIe` lane width (`16` for `x16`, `8` for `x8`).
    pub link_width: u8,
    /// Physical `BAR0` CSR/doorbell resource descriptor.
    pub bar0: PcieBarResource,
    /// Physical `BAR2` HBM/command-ring aperture resource descriptor.
    pub bar2: PcieBarResource,
    /// Physical `BAR4` MSI-X vector table/PBA resource descriptor.
    pub bar4: PcieBarResource,
}

/// Per-port Inter-Chip Interconnect (`ICI`) link health read from BAR0 router CSRs (`IciPortHardwareStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IciPortHardwareStatus {
    /// ICI optical/electrical port index (`0..6` corresponding to `+X, -X, +Y, -Y, +Z, -Z`).
    pub port_index: u8,
    /// Whether the link PCS/PMA state machine reports `LINK_UP` (`bit 0`).
    pub link_up: bool,
    /// Negotiated lane rate in Gbps (`50`, `100`, `200`, `400`, or `800`).
    pub lane_rate_gbps: u16,
    /// Remote peer `GLOBAL_CHIP_ID` extracted from the link partner handshake (`bits 16..32`).
    pub peer_global_chip_id: u16,
    /// Cumulative corrected FEC / CRC lane error count (`bits 32..64`).
    pub fec_corrected_errors: u32,
}

/// Hardware identity, 3D torus coordinates, and ICI link topology probed from BAR0 MMIO CSRs (`ProbedChipCsrIdentity`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedChipCsrIdentity {
    /// Cloud TPU hardware generation.
    pub generation: TpuGeneration,
    /// 16-bit SKU code read from `GLOBAL_CHIP_ID` (`bits 48..64`).
    pub sku_code: u16,
    /// Global chip ID within the pod slice (`bits 0..16` of `GLOBAL_CHIP_ID`).
    pub global_chip_id: u32,
    /// Physical 3D torus `(x, y, z)` coordinates unpacked from `GLOBAL_CHIP_ID` (`bits 16..40`).
    pub coordinates: ChipCoordinates,
    /// Host tray / board slot index (`bits 40..48` of `GLOBAL_CHIP_ID`).
    pub tray_slot_index: u8,
    /// Per-port ICI link status across the chip's active torus ports (`0..6`).
    pub ici_ports: Vec<IciPortHardwareStatus>,
}

/// BAR0 MMIO read trait used by [`HardwareCsrProber`] (implemented by [`super::MappedBarAperture`],
/// [`super::LinuxDeviceHal`], and [`super::emulator::VirtualVfioGasketDevice`]).
pub trait Bar0CsrReader {
    /// Reads a 64-bit CSR value at `offset` within BAR0/BAR2 MMIO space.
    ///
    /// # Arguments
    ///
    /// * `offset` - Physical CSR byte offset from `csr_register_map.json`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the MMIO read faults or is out of bounds.
    fn read_chip_csr_u64(&self, offset: MmioOffset) -> LibTpuResult<u64>;
}

impl Bar0CsrReader for super::MappedBarAperture {
    fn read_chip_csr_u64(&self, offset: MmioOffset) -> LibTpuResult<u64> {
        self.read_csr_u64(offset)
    }
}

/// `sysfs` `PCIe` (`0x1ae0`) device enumerator covering Cloud TPU `v2` through `v7x` (`SysfsPcieEnumerator`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SysfsPcieEnumerator;

impl SysfsPcieEnumerator {
    /// Classifies `(device_id, revision_id, subsystem_device_id)` under vendor `0x1ae0` into [`TpuGeneration`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - 16-bit `PCIe` device ID.
    /// * `revision_id` - 8-bit `PCIe` revision ID (`0x00` = `V2`, `>= 0x01` = `V3` on `0x0027`).
    /// * `subsystem_device_id` - 16-bit `PCIe` subsystem device ID (`0x004e` = `V4Lite` on `0x0056`).
    #[must_use]
    pub const fn classify_pci_ids(
        device_id: u16,
        revision_id: u8,
        subsystem_device_id: u16,
    ) -> Option<TpuGeneration> {
        match device_id {
            0x0027 => {
                if revision_id == 0 {
                    Some(TpuGeneration::V2)
                } else {
                    Some(TpuGeneration::V3)
                }
            }
            0x0056 => {
                if subsystem_device_id == 0x004e {
                    Some(TpuGeneration::V4Lite)
                } else {
                    Some(TpuGeneration::V4)
                }
            }
            0x005e => Some(TpuGeneration::V4Lite),
            0x0063 => Some(TpuGeneration::V5e),
            0x0062 => Some(TpuGeneration::V5p),
            0x006f => Some(TpuGeneration::V6e),
            0x0073 | 0x0076 | 0x007f => Some(TpuGeneration::V7x),
            _ => None,
        }
    }

    /// Parses a single `BAR` line (`bar_index`) from a Linux `sysfs` `resource` file.
    ///
    /// # Arguments
    ///
    /// * `resource_contents` - Full text of `/sys/bus/pci/devices/<bdf>/resource`.
    /// * `bar_index` - Zero-based BAR line index (`0`, `2`, `4`).
    #[must_use]
    pub fn parse_bar_resource(resource_contents: &str, bar_index: u8) -> PcieBarResource {
        let empty = PcieBarResource {
            bar_index,
            phys_start: 0,
            phys_end: 0,
            flags: 0,
            size_bytes: 0,
        };
        let Some(line) = resource_contents.lines().nth(usize::from(bar_index)) else {
            return empty;
        };
        let mut cols = line.split_whitespace();
        let parse_hex = |opt: Option<&str>| -> Option<u64> {
            let s = opt?;
            let stripped = s
                .strip_prefix("0x")
                .or_else(|| s.strip_prefix("0X"))
                .unwrap_or(s);
            u64::from_str_radix(stripped, 16).ok()
        };
        let (Some(start), Some(end), Some(flags)) = (
            parse_hex(cols.next()),
            parse_hex(cols.next()),
            parse_hex(cols.next()),
        ) else {
            return empty;
        };
        if (start == 0 && end == 0) || end < start {
            return PcieBarResource {
                bar_index,
                phys_start: start,
                phys_end: end,
                flags,
                size_bytes: 0,
            };
        }
        PcieBarResource {
            bar_index,
            phys_start: start,
            phys_end: end,
            flags,
            size_bytes: end.saturating_sub(start).saturating_add(1),
        }
    }

    /// Parses a hexadecimal integer file inside `bdf_dir` (e.g., `vendor`, `device`, `revision`).
    ///
    /// # Arguments
    ///
    /// * `bdf_dir` - Directory `/sys/bus/pci/devices/<bdf>`.
    /// * `leaf` - Attribute filename.
    fn read_hex_u16(bdf_dir: &Path, leaf: &str) -> Option<u16> {
        let raw = std::fs::read_to_string(bdf_dir.join(leaf)).ok()?;
        let trimmed = raw.trim();
        let hex = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
            .unwrap_or(trimmed);
        u16::from_str_radix(hex, 16).ok()
    }

    /// Scans `sysfs_pci_root` (`/sys/bus/pci/devices`) and `dev_root` (`/dev`) for all Google
    /// Cloud TPU `0x1ae0` `PCIe` functions, returning them sorted deterministically by BDF.
    ///
    /// # Arguments
    ///
    /// * `sysfs_pci_root` - Path to `/sys/bus/pci/devices` (or a synthetic test directory).
    /// * `dev_root` - Path to `/dev` (or a synthetic test directory).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HardwareIoError`] if `sysfs_pci_root` cannot be read.
    #[allow(clippy::verbose_file_reads)]
    pub fn enumerate_devices(
        sysfs_pci_root: &Path,
        dev_root: &Path,
    ) -> LibTpuResult<Vec<SysfsTpuPcieFunction>> {
        let dir_iter =
            std::fs::read_dir(sysfs_pci_root).map_err(|err| LibTpuError::HardwareIoError {
                operation: "read_dir(sysfs_pci_root)",
                errno: err.raw_os_error().unwrap_or(2),
                message: err.to_string(),
            })?;

        let mut functions = Vec::new();
        for entry in dir_iter.flatten() {
            let bdf_dir = entry.path();
            let bdf_name = entry.file_name();
            let Some(bdf) = PciBdfAddress::parse_bdf(&bdf_name.to_string_lossy()) else {
                continue;
            };
            if Self::read_hex_u16(&bdf_dir, "vendor") != Some(GOOGLE_PCI_VENDOR_ID_RAW) {
                continue;
            }
            let Some(dev_id) = Self::read_hex_u16(&bdf_dir, "device") else {
                continue;
            };
            #[allow(clippy::cast_possible_truncation)]
            let rev_id = Self::read_hex_u16(&bdf_dir, "revision").unwrap_or(0) as u8;
            let subsys_dev = Self::read_hex_u16(&bdf_dir, "subsystem_device").unwrap_or(0);
            let Some(generation) = Self::classify_pci_ids(dev_id, rev_id, subsys_dev) else {
                continue;
            };

            // Resolve IOMMU group from `iommu_group` symlink target basename or `iommu_group_id` file
            let iommu_group = std::fs::read_link(bdf_dir.join("iommu_group"))
                .ok()
                .and_then(|target| {
                    target
                        .file_name()
                        .and_then(|n| n.to_string_lossy().parse::<u32>().ok())
                })
                .or_else(|| {
                    std::fs::read_to_string(bdf_dir.join("iommu_group_id"))
                        .ok()
                        .and_then(|s| s.trim().parse::<u32>().ok())
                });

            // Determine bound driver (`vfio-pci` symlink or `vfio` marker vs `gasket`/`apex`)
            let driver_is_vfio = std::fs::read_link(bdf_dir.join("driver"))
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .is_some_and(|name| name.contains("vfio"))
                || bdf_dir.join("vfio").exists();

            let accel_idx = functions.len();
            let vfio_grp_num = iommu_group.unwrap_or_else(|| u32::try_from(accel_idx).unwrap_or(0));
            let vfio_candidate = dev_root.join("vfio").join(vfio_grp_num.to_string());
            let accel_candidate = dev_root.join(format!("accel{accel_idx}"));

            let (driver_mode, device_node_path) =
                if driver_is_vfio || (vfio_candidate.exists() && !accel_candidate.exists()) {
                    (
                        KernelDriverMode::Vfio,
                        vfio_candidate.to_string_lossy().into_owned(),
                    )
                } else {
                    (
                        KernelDriverMode::Gasket,
                        accel_candidate.to_string_lossy().into_owned(),
                    )
                };

            let numa_node = std::fs::read_to_string(bdf_dir.join("numa_node"))
                .ok()
                .and_then(|s| s.trim().parse::<i32>().ok())
                .unwrap_or(-1);
            let local_cpulist = std::fs::read_to_string(bdf_dir.join("local_cpulist"))
                .map_or_else(|_| String::from("0"), |s| String::from(s.trim()));
            let link_speed = std::fs::read_to_string(bdf_dir.join("current_link_speed"))
                .map_or_else(
                    |_| String::from("32.0 GT/s PCIe"),
                    |s| String::from(s.trim()),
                );
            let link_width = std::fs::read_to_string(bdf_dir.join("current_link_width"))
                .ok()
                .and_then(|s| s.trim().parse::<u8>().ok())
                .unwrap_or(16);

            let res_txt = std::fs::read_to_string(bdf_dir.join("resource")).unwrap_or_default();
            let bar0 = Self::parse_bar_resource(&res_txt, 0);
            let bar2 = Self::parse_bar_resource(&res_txt, 2);
            let bar4 = Self::parse_bar_resource(&res_txt, 4);

            functions.push(SysfsTpuPcieFunction {
                bdf,
                pci_device_id: PciDeviceId(dev_id),
                revision_id: rev_id,
                subsystem_device_id: subsys_dev,
                generation,
                driver_mode,
                iommu_group,
                device_node_path,
                numa_node,
                local_cpulist,
                link_speed,
                link_width,
                bar0,
                bar2,
                bar4,
            });
        }

        functions.sort_by_key(|f| (f.bdf.domain, f.bdf.bus, f.bdf.device, f.bdf.function));
        Ok(functions)
    }
}

/// Hardware BAR0 CSR identity, coordinate, and ICI link prober (`HardwareCsrProber`).
#[derive(Debug, Clone, Copy, Default)]
pub struct HardwareCsrProber;

impl HardwareCsrProber {
    /// Packs `(global_chip_id, coords, tray_slot, sku_code)` into the canonical 64-bit `GLOBAL_CHIP_ID` CSR word.
    ///
    /// # Arguments
    ///
    /// * `global_chip_id` - 16-bit pod-wide chip identifier.
    /// * `coords` - 3D `(x, y, z)` torus coordinates (`8` bits per axis).
    /// * `tray_slot` - Board/tray slot index (`0..8`).
    /// * `sku_code` - 16-bit hardware SKU code (`e.g. 0x006f`).
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub const fn pack_global_chip_id_csr(
        global_chip_id: u16,
        coords: ChipCoordinates,
        tray_slot: u8,
        sku_code: u16,
    ) -> u64 {
        let x = (coords.x as u64) & 0xFF;
        let y = (coords.y as u64) & 0xFF;
        let z = (coords.z as u64) & 0xFF;
        (global_chip_id as u64)
            | (x << 16)
            | (y << 24)
            | (z << 32)
            | ((tray_slot as u64) << 40)
            | ((sku_code as u64) << 48)
    }

    /// Packs `(link_up, lane_rate_gbps, peer_chip_id, fec_corrected_errors)` into a 64-bit ICI port status CSR word.
    ///
    /// # Arguments
    ///
    /// * `link_up` - Whether PCS/PMA reports link up.
    /// * `lane_rate_gbps` - Per-port bandwidth in Gbps (`50..=800`).
    /// * `peer_chip_id` - Connected neighbor `GLOBAL_CHIP_ID`.
    /// * `fec_corrected_errors` - Corrected FEC codeword count.
    #[must_use]
    pub const fn pack_ici_port_status_csr(
        link_up: bool,
        lane_rate_gbps: u16,
        peer_chip_id: u16,
        fec_corrected_errors: u32,
    ) -> u64 {
        let up_bit = if link_up { 1_u64 } else { 0_u64 };
        let rate_bits = ((lane_rate_gbps as u64) & 0x7FFF) << 1;
        let peer_bits = (peer_chip_id as u64) << 16;
        let fec_bits = (fec_corrected_errors as u64) << 32;
        up_bit | rate_bits | peer_bits | fec_bits
    }

    /// Returns the BAR0 ICI router status CSR base byte offset for `generation` from `csr_register_map.json`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `port_index` - ICI port index (`0..6`).
    #[must_use]
    pub fn ici_port_status_csr_offset(generation: TpuGeneration, port_index: u8) -> MmioOffset {
        let rmap = ChipRegisterMap::for_generation(generation);
        let base = rmap
            .resolve_hierarchical_csr_path("ici.router.link_status_0")
            .map_or(0xA106_7100, |d| d.byte_offset);
        MmioOffset(base.saturating_add(u64::from(port_index).saturating_mul(8)))
    }

    /// Probes chip SKU, `GLOBAL_CHIP_ID`, 3D board/tray coordinates, and active ICI links from `reader`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `reader` - BAR0 MMIO CSR reader.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if reading `GLOBAL_CHIP_ID` or ICI port status CSRs fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn probe_chip_identity(
        generation: TpuGeneration,
        reader: &dyn Bar0CsrReader,
    ) -> LibTpuResult<ProbedChipCsrIdentity> {
        let rmap = ChipRegisterMap::for_generation(generation);
        let raw_id = reader.read_chip_csr_u64(MmioOffset(rmap.global_chip_id_csr))?;
        let global_chip_id = (raw_id & 0xFFFF) as u32;
        let x = u32::from(((raw_id >> 16) & 0xFF) as u8);
        let y = u32::from(((raw_id >> 24) & 0xFF) as u8);
        let z = u32::from(((raw_id >> 32) & 0xFF) as u8);
        let tray_slot_index = ((raw_id >> 40) & 0xFF) as u8;
        let raw_sku = ((raw_id >> 48) & 0xFFFF) as u16;
        let sku_code = if raw_sku == 0 {
            generation.pcie_device_id()
        } else {
            raw_sku
        };

        let port_count: u8 = match generation {
            TpuGeneration::V2 | TpuGeneration::V3 | TpuGeneration::V5e | TpuGeneration::V6e => 4,
            TpuGeneration::V4 | TpuGeneration::V4Lite | TpuGeneration::V5p | TpuGeneration::V7x => {
                6
            }
        };
        let default_rate: u16 = match generation {
            TpuGeneration::V2 => 50,
            TpuGeneration::V3 | TpuGeneration::V4 | TpuGeneration::V4Lite => 100,
            TpuGeneration::V5e => 200,
            TpuGeneration::V5p | TpuGeneration::V6e => 400,
            TpuGeneration::V7x => 800,
        };

        let mut ici_ports = Vec::with_capacity(usize::from(port_count));
        for port_index in 0..port_count {
            let off = Self::ici_port_status_csr_offset(generation, port_index);
            let word = reader.read_chip_csr_u64(off)?;
            let link_up = (word & 1) != 0;
            let parsed_rate = ((word >> 1) & 0x7FFF) as u16;
            let lane_rate_gbps = if parsed_rate == 0 {
                default_rate
            } else {
                parsed_rate
            };
            let peer_global_chip_id = ((word >> 16) & 0xFFFF) as u16;
            let fec_corrected_errors = (word >> 32) as u32;
            ici_ports.push(IciPortHardwareStatus {
                port_index,
                link_up,
                lane_rate_gbps,
                peer_global_chip_id,
                fec_corrected_errors,
            });
        }

        Ok(ProbedChipCsrIdentity {
            generation,
            sku_code,
            global_chip_id,
            coordinates: ChipCoordinates::new(x, y, z),
            tray_slot_index,
            ici_ports,
        })
    }
}
