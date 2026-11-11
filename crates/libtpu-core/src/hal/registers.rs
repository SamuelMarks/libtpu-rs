//! Gasket/VFIO `ioctl` constants, `DeepSea` BAR0 register maps, and CSR bitflags.
//!
//! Defines Gasket (`0xdc`) and VFIO (`0x3b`) `ioctl` command numbers, per-generation `DeepSea` BAR0
//! register maps ([`ChipRegisterMap`]), `#[repr(C)]` kernel `ioctl` payload structures, and
//! architectural CSR bitflags.

use crate::types::TpuGeneration;
use bytemuck::{Pod, Zeroable};

/// Exact Gasket ioctl command for resetting the accelerator (`_IOW(0xdc, 0x00, 8)`).
pub const GASKET_IOCTL_RESET: u64 = 0x4008_dc00;

/// Exact Gasket ioctl command for querying simple page table addresses (`_IOWR(0xdc, 0x05, 32)`).
pub const GASKET_IOCTL_NUMBER_PAGE_TABLES: u64 = 0xc020_dc05;

/// Exact Gasket ioctl command for querying page table size (`_IOWR(0xdc, 0x06, 32)`).
pub const GASKET_IOCTL_PAGE_TABLE_SIZE: u64 = 0xc020_dc06;

/// Exact Gasket ioctl command for querying simple page table size (`_IOW(0xdc, 0x07, 32)`).
pub const GASKET_IOCTL_SIMPLE_PAGE_TABLE_SIZE: u64 = 0x4020_dc07;

/// Exact Gasket ioctl command for partitioning page tables (`_IOW(0xdc, 0x07, 32)`).
pub const GASKET_IOCTL_PARTITION_PAGE_TABLE: u64 = 0x4020_dc07;

/// Exact Gasket ioctl command for mapping host DMA buffers (`_IOW(0xdc, 0x08, 32)`).
pub const GASKET_IOCTL_MAP_BUFFER: u64 = 0x4020_dc08;

/// Exact Gasket ioctl command for unmapping host DMA buffers (`_IOW(0xdc, 0x09, 32)`).
pub const GASKET_IOCTL_UNMAP_BUFFER: u64 = 0x4020_dc09;

/// Exact Gasket ioctl command for clearing interrupt counts (`_IO(0xdc, 0x0a)`).
pub const GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS: u64 = 0x0000_dc0a;

/// Exact Gasket ioctl command for binding an interrupt `eventfd` (`_IOW(0xdc, 0x0b, 32)`).
pub const GASKET_IOCTL_SET_EVENTFD: u64 = 0x4020_dc0b;

/// Exact Gasket ioctl command for clearing an interrupt `eventfd` (`_IOW(0xdc, 0x0c, 8)`).
pub const GASKET_IOCTL_CLEAR_EVENTFD: u64 = 0x4008_dc0c;

/// Exact Gasket ioctl command for mapping a `dma_buf` fd (`_IOW(0xdc, 0x0d, 24)`).
pub const GASKET_IOCTL_MAP_DMABUF: u64 = 0x4018_dc0d;

/// Exact VFIO ioctl command for querying the VFIO API version (`0x3b64`).
pub const VFIO_GET_API_VERSION: u64 = 0x0000_3b64;

/// Exact VFIO ioctl command for checking IOMMU extension support (`0x3b65`).
pub const VFIO_CHECK_EXTENSION: u64 = 0x0000_3b65;

/// Exact VFIO ioctl command for setting the container IOMMU mode (`0x3b66`).
pub const VFIO_SET_IOMMU: u64 = 0x0000_3b66;

/// Exact VFIO ioctl command for querying IOMMU group status (`0x3b67`).
pub const VFIO_GROUP_GET_STATUS: u64 = 0x0000_3b67;

/// Exact VFIO ioctl command for binding a group to a VFIO container (`0x3b68`).
pub const VFIO_GROUP_SET_CONTAINER: u64 = 0x0000_3b68;

/// Exact VFIO ioctl command for opening a device fd from a group (`0x3b6a`).
pub const VFIO_GROUP_GET_DEVICE_FD: u64 = 0x0000_3b6a;

/// Exact VFIO ioctl command for querying device region/IRQ counts (`0x3b6b`).
pub const VFIO_DEVICE_GET_INFO: u64 = 0x0000_3b6b;

/// Exact VFIO ioctl command for querying a `PCIe` BAR region descriptor (`0x3b6c`).
pub const VFIO_DEVICE_GET_REGION_INFO: u64 = 0x0000_3b6c;

/// Exact VFIO ioctl command for querying interrupt vector metadata (`0x3b6d`).
pub const VFIO_DEVICE_GET_IRQ_INFO: u64 = 0x0000_3b6d;

/// Exact VFIO ioctl command for configuring MSI-X interrupt signaling (`0x3b6e`).
pub const VFIO_DEVICE_SET_IRQS_IOCTL: u64 = 0x0000_3b6e;

/// Alias for [`VFIO_DEVICE_SET_IRQS_IOCTL`] (`0x3b6e`).
pub const VFIO_DEVICE_SET_IRQS: u64 = VFIO_DEVICE_SET_IRQS_IOCTL;

/// Exact VFIO ioctl command for resetting a `PCIe` function (`0x3b6f`).
pub const VFIO_DEVICE_RESET: u64 = 0x0000_3b6f;

/// Exact VFIO ioctl command for mapping DMA pages (`0x3b71`).
pub const VFIO_IOMMU_MAP_DMA_IOCTL: u64 = 0x0000_3b71;

/// Alias for [`VFIO_IOMMU_MAP_DMA_IOCTL`] (`0x3b71`).
pub const VFIO_IOMMU_MAP_DMA: u64 = VFIO_IOMMU_MAP_DMA_IOCTL;

/// Exact VFIO ioctl command for unmapping DMA pages (`0x3b72`).
pub const VFIO_IOMMU_UNMAP_DMA_IOCTL: u64 = 0x0000_3b72;

/// Alias for [`VFIO_IOMMU_UNMAP_DMA_IOCTL`] (`0x3b72`).
pub const VFIO_IOMMU_UNMAP_DMA: u64 = VFIO_IOMMU_UNMAP_DMA_IOCTL;

/// Hardware CSR access permission classification from `asic_sw.tools.RegisterTranslatorData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CsrAccessPermission {
    /// Read-Only hardware status/telemetry register (`RO`).
    ReadOnly,
    /// Read-Write hardware control/configuration register (`RW`).
    ReadWrite,
    /// Write-Only hardware doorbell/command register (`WO`).
    WriteOnly,
    /// Write-1-to-Clear sticky interrupt or fault register (`W1C`).
    WriteOneToClear,
}

impl CsrAccessPermission {
    /// Parses a symbolic access permission code (`"RO"`, `"RW"`, `"WO"`, `"W1C"`) into a [`CsrAccessPermission`].
    ///
    /// # Arguments
    ///
    /// * `code` - Access string from `csr_register_map.json`.
    #[must_use]
    pub fn from_str_code(code: &str) -> Self {
        match code {
            "RO" => Self::ReadOnly,
            "WO" => Self::WriteOnly,
            "W1C" => Self::WriteOneToClear,
            _ => Self::ReadWrite,
        }
    }
}

/// Architectural CSR bitfield specification (`CsrBitfieldSpec`) within a 64-bit `DeepSea` CSR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsrBitfieldSpec {
    /// Symbolic bitfield name (e.g., `"RUN_STATUS"`, `"PC"`, `"VWAIT_STATE"`).
    pub name: &'static str,
    /// Least-significant bit index (`0..63`).
    pub lsb: u8,
    /// Most-significant bit index (`lsb..63`).
    pub msb: u8,
    /// Hardware access permission for this bitfield.
    pub access: CsrAccessPermission,
}

impl CsrBitfieldSpec {
    /// Extracts this bitfield's unsigned value from a 64-bit register `reg_value`.
    ///
    /// # Arguments
    ///
    /// * `reg_value` - Raw 64-bit CSR value.
    #[must_use]
    pub const fn unpack(self, reg_value: u64) -> u64 {
        let width = self.msb.saturating_sub(self.lsb).saturating_add(1);
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1_u64 << width).wrapping_sub(1)
        };
        (reg_value >> self.lsb) & mask
    }

    /// Inserts `field_value` into `reg_value` at `[lsb..=msb]` and returns the updated 64-bit word.
    ///
    /// # Arguments
    ///
    /// * `reg_value` - Existing 64-bit CSR value.
    /// * `field_value` - Bitfield value to pack into `[lsb..=msb]`.
    #[must_use]
    pub const fn pack(self, reg_value: u64, field_value: u64) -> u64 {
        let width = self.msb.saturating_sub(self.lsb).saturating_add(1);
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1_u64 << width).wrapping_sub(1)
        };
        let shifted_mask = mask << self.lsb;
        (reg_value & !shifted_mask) | ((field_value & mask) << self.lsb)
    }
}

/// Embedded raw JSON string of `docs/reverse_engineering/csr_register_map.json` containing
/// the full 8-generation `DeepSea` hardware CSR catalog extracted from `*_pf_csrs.br`.
pub const EMBEDDED_CSR_REGISTER_MAP_JSON: &str =
    include_str!("../../../../docs/reverse_engineering/csr_register_map.json");

/// Parsed entry from the global `csr_register_map.json` catalog (`CsrCatalogEntry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsrCatalogEntry {
    /// Cloud TPU hardware generation (`V2..=V7x`).
    pub generation: TpuGeneration,
    /// Full hierarchical register name from `*_pf_csrs.br` (e.g., `"pxc_tc_0_tcs_core_state.PC"`).
    pub full_path: &'static str,
    /// Physical BAR0 MMIO byte offset (`CsrAddress`).
    pub byte_offset: u64,
    /// Register width in bits (`32` or `64`).
    pub width_bits: u32,
    /// Hardware access permission (`RO`, `RW`, `WO`, `W1C`).
    pub access: CsrAccessPermission,
}

impl CsrCatalogEntry {
    /// Converts this [`CsrCatalogEntry`] into a [`CsrRegisterDescriptor`] with architectural bitfields
    /// selected from the register's leaf name and access mode.
    #[must_use]
    pub fn to_descriptor(self) -> CsrRegisterDescriptor {
        let bitfields = ChipRegisterMap::bitfields_for_register_path(self.full_path, self.access);
        let reset_value = u64::from(self.full_path.ends_with("RUN_STATUS"));
        CsrRegisterDescriptor {
            full_path: self.full_path,
            byte_offset: self.byte_offset,
            access: self.access,
            reset_value,
            bitfields,
        }
    }
}

/// Global lazy-initialized catalog of all CSR registers parsed from [`EMBEDDED_CSR_REGISTER_MAP_JSON`].
static GLOBAL_CSR_CATALOG: std::sync::OnceLock<Vec<CsrCatalogEntry>> = std::sync::OnceLock::new();

/// Parses [`EMBEDDED_CSR_REGISTER_MAP_JSON`] into a flat slice of [`CsrCatalogEntry`] records
/// covering all 8 Cloud TPU generations (`V2`, `V3`, `V4`, `V4Lite`, `V5e`, `V5p`, `V6e`, `V7x`).
#[must_use]
pub fn global_csr_register_catalog() -> &'static [CsrCatalogEntry] {
    GLOBAL_CSR_CATALOG.get_or_init(|| {
        let mut entries = Vec::with_capacity(2700);
        let mut current_gen = TpuGeneration::V4;
        let mut pending_name: &'static str = "";
        let mut pending_offset: u64 = 0;
        let mut pending_width: u32 = 64;
        for raw_line in EMBEDDED_CSR_REGISTER_MAP_JSON.lines() {
            let trimmed = raw_line.trim();
            if trimmed.starts_with("\"V2\":") {
                current_gen = TpuGeneration::V2;
            } else if trimmed.starts_with("\"V3\":") {
                current_gen = TpuGeneration::V3;
            } else if trimmed.starts_with("\"V4\":") {
                current_gen = TpuGeneration::V4;
            } else if trimmed.starts_with("\"V4Lite\":") {
                current_gen = TpuGeneration::V4Lite;
            } else if trimmed.starts_with("\"V5e\":") {
                current_gen = TpuGeneration::V5e;
            } else if trimmed.starts_with("\"V5p\":") {
                current_gen = TpuGeneration::V5p;
            } else if trimmed.starts_with("\"V6e\":") {
                current_gen = TpuGeneration::V6e;
            } else if trimmed.starts_with("\"V7x\":") {
                current_gen = TpuGeneration::V7x;
            } else if let Some(rest) = trimmed.strip_prefix("\"name\": \"") {
                pending_name = rest.trim_end_matches("\",");
            } else if let Some(rest) = trimmed.strip_prefix("\"offset\": \"") {
                let val = rest.trim_end_matches("\",");
                let hex = val.strip_prefix("0x").unwrap_or(val);
                pending_offset = u64::from_str_radix(hex, 16).unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("\"width_bits\": ") {
                let num_str = rest.trim_end_matches(',');
                pending_width = num_str.parse::<u32>().unwrap_or(64);
            } else if let Some(rest) = trimmed.strip_prefix("\"access\": \"") {
                let val = rest.trim_end_matches('"');
                let access = CsrAccessPermission::from_str_code(val);
                entries.push(CsrCatalogEntry {
                    generation: current_gen,
                    full_path: pending_name,
                    byte_offset: pending_offset,
                    width_bits: pending_width,
                    access,
                });
            }
        }
        entries
    })
}

/// Complete `asic_sw.tools.RegisterTranslatorData` register node descriptor (`CsrRegisterDescriptor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsrRegisterDescriptor {
    /// Hierarchical path in `RegisterTranslatorData` (e.g., `"tc_0.tcs_core_state.RUN_STATUS"`).
    pub full_path: &'static str,
    /// Physical BAR0 MMIO byte offset (`CsrAddress`).
    pub byte_offset: u64,
    /// Register access permission (`RO`, `RW`, `WO`, `W1C`).
    pub access: CsrAccessPermission,
    /// Reset default value (`0` for status registers).
    pub reset_value: u64,
    /// Architectural bitfields packed inside this 64-bit register.
    pub bitfields: &'static [CsrBitfieldSpec],
}

impl CsrRegisterDescriptor {
    /// Applies a 64-bit software write `write_val` to `current_val` respecting `self.access`
    /// (`ReadOnly`, `ReadWrite`, `WriteOnly`, `WriteOneToClear`).
    ///
    /// # Arguments
    ///
    /// * `current_val` - Current 64-bit CSR value before the write.
    /// * `write_val` - 64-bit value written by the host driver.
    #[must_use]
    pub const fn apply_write_value(&self, current_val: u64, write_val: u64) -> u64 {
        match self.access {
            CsrAccessPermission::ReadOnly => current_val,
            CsrAccessPermission::ReadWrite | CsrAccessPermission::WriteOnly => write_val,
            CsrAccessPermission::WriteOneToClear => current_val & !write_val,
        }
    }

    /// Unpacks the bitfield named `bitfield_name` from `reg_value`, if present.
    ///
    /// # Arguments
    ///
    /// * `reg_value` - Raw 64-bit CSR value.
    /// * `bitfield_name` - Symbolic bitfield name (`"STATE"`, `"FAULT_CODE"`, `"VALUE"`).
    #[must_use]
    pub fn unpack_bitfield(&self, reg_value: u64, bitfield_name: &str) -> Option<u64> {
        self.bitfields
            .iter()
            .find(|bf| bf.name == bitfield_name)
            .map(|bf| bf.unpack(reg_value))
    }

    /// Packs `field_value` into the bitfield named `bitfield_name` within `reg_value`, if present.
    ///
    /// # Arguments
    ///
    /// * `reg_value` - Existing 64-bit CSR value.
    /// * `bitfield_name` - Symbolic bitfield name (`"STATE"`, `"FAULT_CODE"`, `"VALUE"`).
    /// * `field_value` - New bitfield value to pack.
    #[must_use]
    pub fn pack_bitfield(
        &self,
        reg_value: u64,
        bitfield_name: &str,
        field_value: u64,
    ) -> Option<u64> {
        self.bitfields
            .iter()
            .find(|bf| bf.name == bitfield_name)
            .map(|bf| bf.pack(reg_value, field_value))
    }
}

/// Per-generation `asic_sw::deepsea` hardware register and memory layout specification
/// extracted from `libtpu.so` (`*_pf_csrs.br`, `*_chip_configs_*.binarypb`, and
/// `current_tpu_runtime_abi.binarypb`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChipRegisterMap {
    /// Cloud TPU hardware generation.
    pub generation: TpuGeneration,
    /// Internal `DeepSea` ASIC codename (`jellyfish`, `dragonfish`, `pufferfish`, `viperfish_lite`, `viperfish`, `ghostlite`, `6acc60406`).
    pub chip_codename: &'static str,
    /// Hardware ABI version (`65537` or `65539`) from `current_tpu_runtime_abi.binarypb`.
    pub hardware_abi_version: u64,
    /// `TensorCore` 0 `tcs_core_state.RUN_STATUS` BAR0 MMIO byte offset.
    pub tc0_run_status_csr: u64,
    /// `TensorCore` 0 `tcs_core_state.PC` BAR0 MMIO byte offset.
    pub tc0_pc_csr: u64,
    /// `TensorCore` 0 `tcs_core_state.PROGRAM_LOCATION` BAR0 MMIO byte offset.
    pub tc0_program_location_csr: u64,
    /// `TensorCore` 0 `tc_main_status.VWAIT_STATE` BAR0 MMIO byte offset.
    pub tc0_vwait_state_csr: u64,
    /// `TensorCore` 0 `tc_main_status.VWAIT_SYNC_FLAG_STATE` BAR0 MMIO byte offset.
    pub tc0_vwait_sync_flag_state_csr: u64,
    /// `SparseCore` 0 `scs_core_state.RUN_STATUS` BAR0 MMIO byte offset (`0` if absent).
    pub sc0_run_status_csr: u64,
    /// Global chip ID register (`GLOBAL_CHIP_ID = 0xA1067000`).
    pub global_chip_id_csr: u64,
    /// Continuation queue completion `SyncFlag` word offset from `TpuChipConfigProto`.
    pub completion_state_sync_flag_word_offset: u32,
    /// Reserved HBM word offset for runtime continuation state from `TpuChipConfigProto`.
    pub hbm_reserved_word_offset: u64,
}

impl ChipRegisterMap {
    /// Internal `const` constructor for [`ChipRegisterMap`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `chip_codename` - Internal `DeepSea` ASIC codename.
    /// * `hardware_abi_version` - Hardware ABI version (`65537` or `65539`).
    /// * `csrs` - `[tc0_run_status, tc0_pc, tc0_program_location, tc0_vwait_state, tc0_vwait_sync_flag_state, sc0_run_status]`.
    /// * `completion_state_sync_flag_word_offset` - Continuation queue completion `SyncFlag` word offset.
    /// * `hbm_reserved_word_offset` - Reserved HBM word offset.
    const fn build(
        generation: TpuGeneration,
        chip_codename: &'static str,
        hardware_abi_version: u64,
        csrs: [u64; 6],
        completion_state_sync_flag_word_offset: u32,
        hbm_reserved_word_offset: u64,
    ) -> Self {
        Self {
            generation,
            chip_codename,
            hardware_abi_version,
            tc0_run_status_csr: csrs[0],
            tc0_pc_csr: csrs[1],
            tc0_program_location_csr: csrs[2],
            tc0_vwait_state_csr: csrs[3],
            tc0_vwait_sync_flag_state_csr: csrs[4],
            sc0_run_status_csr: csrs[5],
            global_chip_id_csr: 0xA106_7000,
            completion_state_sync_flag_word_offset,
            hbm_reserved_word_offset,
        }
    }

    /// `Jellyfish` / `Dragonfish` (`jxc`) BAR0 CSR offsets.
    const JXC_CSRS: [u64; 6] = [
        0x0042_2018,
        0x0042_2058,
        0x0042_20A0,
        0x0042_30F8,
        0x0042_3100,
        0,
    ];
    /// `Pufferfish` (`pxc`) BAR0 CSR offsets.
    const PXC_CSRS: [u64; 6] = [
        0x0403_A018,
        0x0403_A058,
        0x0403_A0A0,
        0x0403_B1B8,
        0x0403_B1C0,
        0,
    ];
    /// `Viperfish Lite` (`vlc` / `v5e`) BAR0 CSR offsets from `vlc_pf_csrs.br`.
    const VLC_CSRS: [u64; 6] = [
        0xA842_2018,
        0xA842_2058,
        0xA842_20A0,
        0xA842_30F8,
        0xA842_3100,
        0xD184_1018,
    ];
    /// `Viperfish` (`vfc` / `v5p`) BAR0 CSR offsets from `vfc_pf_csrs.br`.
    const VFC_CSRS: [u64; 6] = [
        0xC403_A018,
        0xC403_A058,
        0xC403_A0A0,
        0xC403_B1B8,
        0xC403_B1C0,
        0xD184_1018,
    ];
    /// `Ghostlite` (`glc` / `v6e`) BAR0 CSR offsets from `glc_pf_csrs.br`.
    const GLC_CSRS: [u64; 6] = [
        0xC220_2018,
        0xC220_2058,
        0xC220_20A0,
        0xC220_3048,
        0xC220_3050,
        0xD284_1018,
    ];
    /// `Ghostfish` / `Ironwood` (`gfc` / `6acc60406` / `v7x`) BAR0 CSR offsets from `gfc_pf_csrs.br`.
    const GFC_CSRS: [u64; 6] = [
        0xA667_FA18,
        0xA667_FA58,
        0xA667_FAA0,
        0xA667_F648,
        0xA667_F650,
        0xA7F2_0118,
    ];

    /// Returns the exact extracted `DeepSea` [`ChipRegisterMap`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn for_generation(generation: TpuGeneration) -> Self {
        match generation {
            TpuGeneration::V2 => Self::build(
                generation,
                "jellyfish",
                65_537,
                Self::JXC_CSRS,
                507,
                536_608_768,
            ),
            TpuGeneration::V3 => Self::build(
                generation,
                "dragonfish",
                65_537,
                Self::JXC_CSRS,
                507,
                1_073_479_680,
            ),
            TpuGeneration::V4 => Self::build(
                generation,
                "pufferfish",
                65_537,
                Self::PXC_CSRS,
                507,
                2_147_221_504,
            ),
            TpuGeneration::V4Lite => Self::build(
                generation,
                "pufferfish_lite",
                65_537,
                Self::PXC_CSRS,
                507,
                1_073_479_680,
            ),
            TpuGeneration::V5e => Self::build(
                generation,
                "viperfish_lite",
                65_537,
                Self::VLC_CSRS,
                2041,
                1_073_479_680,
            ),
            TpuGeneration::V5p => Self::build(
                generation,
                "viperfish",
                65_539,
                Self::VFC_CSRS,
                4089,
                3_187_408_896,
            ),
            TpuGeneration::V6e => Self::build(
                generation,
                "ghostlite",
                65_539,
                Self::GLC_CSRS,
                4089,
                2_147_221_504,
            ),
            TpuGeneration::V7x => Self::build(
                generation,
                "6acc60406",
                65_539,
                Self::GFC_CSRS,
                8185,
                3_187_408_896,
            ),
        }
    }

    /// Standard `RUN_STATUS` bitfield slice (`IDLE`, `RUNNING`, `COMPLETED`, `FAULT`).
    const RUN_STATUS_BITFIELDS: [CsrBitfieldSpec; 2] = [
        CsrBitfieldSpec {
            name: "STATE",
            lsb: 0,
            msb: 3,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "FAULT_CODE",
            lsb: 4,
            msb: 15,
            access: CsrAccessPermission::WriteOneToClear,
        },
    ];

    /// Program Counter (`PC` / `PROGRAM_LOCATION`) bitfield slice (`BUNDLE_INDEX`, `SUBCORE_ID`).
    const PC_BITFIELDS: [CsrBitfieldSpec; 2] = [
        CsrBitfieldSpec {
            name: "BUNDLE_INDEX",
            lsb: 0,
            msb: 47,
            access: CsrAccessPermission::ReadWrite,
        },
        CsrBitfieldSpec {
            name: "SUBCORE_ID",
            lsb: 48,
            msb: 63,
            access: CsrAccessPermission::ReadOnly,
        },
    ];

    /// Vector wait / sync flag status (`VWAIT_STATE`) bitfield slice (`WAITING`, `FLAG_ID`, `TARGET_COUNT`).
    const VWAIT_STATE_BITFIELDS: [CsrBitfieldSpec; 3] = [
        CsrBitfieldSpec {
            name: "WAITING",
            lsb: 0,
            msb: 0,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "FLAG_ID",
            lsb: 4,
            msb: 15,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "TARGET_COUNT",
            lsb: 16,
            msb: 31,
            access: CsrAccessPermission::ReadOnly,
        },
    ];

    /// Async DMA ring / queue status (`DMA_STATUS`) bitfield slice (`QUEUE_INDEX`, `RING_FULL`, `ERR_CODE`).
    const DMA_STATUS_BITFIELDS: [CsrBitfieldSpec; 3] = [
        CsrBitfieldSpec {
            name: "QUEUE_INDEX",
            lsb: 0,
            msb: 15,
            access: CsrAccessPermission::ReadWrite,
        },
        CsrBitfieldSpec {
            name: "RING_FULL",
            lsb: 16,
            msb: 16,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "ERR_CODE",
            lsb: 24,
            msb: 31,
            access: CsrAccessPermission::WriteOneToClear,
        },
    ];

    /// Inter-Chip Interconnect (`ICI`) router link status bitfield slice (`LINK_UP_MASK`, `LANE_DEGRADED`, `CRC_FAULT`).
    const ICI_LINK_STATE_BITFIELDS: [CsrBitfieldSpec; 3] = [
        CsrBitfieldSpec {
            name: "LINK_UP_MASK",
            lsb: 0,
            msb: 15,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "LANE_DEGRADED",
            lsb: 16,
            msb: 23,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "CRC_FAULT",
            lsb: 24,
            msb: 31,
            access: CsrAccessPermission::WriteOneToClear,
        },
    ];

    /// HBM memory controller ECC fault / counter bitfield slice (`COUNT`, `CHANNEL_ID`, `OVERFLOW`).
    const HBM_ECC_BITFIELDS: [CsrBitfieldSpec; 3] = [
        CsrBitfieldSpec {
            name: "COUNT",
            lsb: 0,
            msb: 31,
            access: CsrAccessPermission::WriteOneToClear,
        },
        CsrBitfieldSpec {
            name: "CHANNEL_ID",
            lsb: 32,
            msb: 39,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "OVERFLOW",
            lsb: 63,
            msb: 63,
            access: CsrAccessPermission::WriteOneToClear,
        },
    ];

    /// Power / thermal sensor telemetry bitfield slice (`SENSOR_VALUE`, `THROTTLE_ACTIVE`).
    const THERMAL_BITFIELDS: [CsrBitfieldSpec; 2] = [
        CsrBitfieldSpec {
            name: "SENSOR_VALUE",
            lsb: 0,
            msb: 31,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "THROTTLE_ACTIVE",
            lsb: 32,
            msb: 32,
            access: CsrAccessPermission::ReadOnly,
        },
    ];

    /// `SparseCore` (`Barnacore`) status bitfield slice (`STATE`, `ACTIVE_LANES`, `FAULT_CODE`).
    const SPARSECORE_STATUS_BITFIELDS: [CsrBitfieldSpec; 3] = [
        CsrBitfieldSpec {
            name: "STATE",
            lsb: 0,
            msb: 3,
            access: CsrAccessPermission::ReadOnly,
        },
        CsrBitfieldSpec {
            name: "FAULT_CODE",
            lsb: 4,
            msb: 15,
            access: CsrAccessPermission::WriteOneToClear,
        },
        CsrBitfieldSpec {
            name: "ACTIVE_LANES",
            lsb: 16,
            msb: 31,
            access: CsrAccessPermission::ReadOnly,
        },
    ];

    /// Standard 64-bit full-width value bitfield slice.
    const VALUE64_BITFIELDS: [CsrBitfieldSpec; 1] = [CsrBitfieldSpec {
        name: "VALUE",
        lsb: 0,
        msb: 63,
        access: CsrAccessPermission::ReadWrite,
    }];

    /// Standard 64-bit full-width write-only doorbell bitfield slice.
    const DOORBELL64_BITFIELDS: [CsrBitfieldSpec; 1] = [CsrBitfieldSpec {
        name: "VALUE",
        lsb: 0,
        msb: 63,
        access: CsrAccessPermission::WriteOnly,
    }];

    /// Standard 64-bit full-width write-1-to-clear fault bitfield slice.
    const W1C64_BITFIELDS: [CsrBitfieldSpec; 1] = [CsrBitfieldSpec {
        name: "VALUE",
        lsb: 0,
        msb: 63,
        access: CsrAccessPermission::WriteOneToClear,
    }];

    /// Selects the architectural [`CsrBitfieldSpec`] slice for a register path and access mode.
    ///
    /// # Arguments
    ///
    /// * `full_path` - Hierarchical CSR register path from `csr_register_map.json`.
    /// * `access` - Register access permission (`RO`, `RW`, `WO`, `W1C`).
    #[must_use]
    pub fn bitfields_for_register_path(
        full_path: &str,
        access: CsrAccessPermission,
    ) -> &'static [CsrBitfieldSpec] {
        if full_path.contains("scs_core_state") {
            &Self::SPARSECORE_STATUS_BITFIELDS
        } else if full_path.contains("DMA") {
            &Self::DMA_STATUS_BITFIELDS
        } else if full_path.contains("ICI") {
            &Self::ICI_LINK_STATE_BITFIELDS
        } else if full_path.contains("ECC") {
            &Self::HBM_ECC_BITFIELDS
        } else if full_path.contains("THERMAL") {
            &Self::THERMAL_BITFIELDS
        } else if full_path.ends_with("STATUS") {
            &Self::RUN_STATUS_BITFIELDS
        } else if full_path.ends_with("PC") {
            &Self::PC_BITFIELDS
        } else if full_path.contains("VWAIT") {
            &Self::VWAIT_STATE_BITFIELDS
        } else {
            match access {
                CsrAccessPermission::WriteOnly => &Self::DOORBELL64_BITFIELDS,
                CsrAccessPermission::WriteOneToClear => &Self::W1C64_BITFIELDS,
                CsrAccessPermission::ReadOnly | CsrAccessPermission::ReadWrite => {
                    &Self::VALUE64_BITFIELDS
                }
            }
        }
    }

    /// Returns the BAR0 MMIO byte offset of `TensorCore` 1 `tcs_core_state.RUN_STATUS` (`tc_1.tcs_core_state.RUN_STATUS`).
    #[must_use]
    pub const fn tc1_run_status_csr(&self) -> u64 {
        let tc0_base = self.tc0_run_status_csr & !0xFFF_u64;
        tc0_base.wrapping_add(0x0010_0018)
    }

    /// Returns the BAR0 MMIO byte offset of `TensorCore` 1 `tcs_core_state.PC` (`tc_1.tcs_core_state.PC`).
    #[must_use]
    pub const fn tc1_pc_csr(&self) -> u64 {
        let tc0_base = self.tc0_run_status_csr & !0xFFF_u64;
        tc0_base.wrapping_add(0x0010_0058)
    }

    /// Returns the BAR0 MMIO byte offset of `TensorCore` 1 `tcs_core_state.PROGRAM_LOCATION` (`tc_1.tcs_core_state.PROGRAM_LOCATION`).
    #[must_use]
    pub const fn tc1_program_location_csr(&self) -> u64 {
        let tc0_base = self.tc0_run_status_csr & !0xFFF_u64;
        tc0_base.wrapping_add(0x0010_00A0)
    }

    /// Returns the `TensorCore` 0 and 1 [`CsrRegisterDescriptor`] list for this chip generation.
    #[allow(clippy::too_many_lines)]
    fn tensor_core_csr_descriptors(&self) -> Vec<CsrRegisterDescriptor> {
        let tc0_base = self.tc0_run_status_csr & !0xFFF_u64;
        let tc1_base = tc0_base.wrapping_add(0x0010_0000);
        vec![
            CsrRegisterDescriptor {
                full_path: "tc_0.tcs_core_state.CORE_RESET",
                byte_offset: tc0_base,
                access: CsrAccessPermission::WriteOnly,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tcs_core_state.RUN_STATUS",
                byte_offset: self.tc0_run_status_csr,
                access: CsrAccessPermission::ReadOnly,
                reset_value: 1,
                bitfields: &Self::RUN_STATUS_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tcs_core_state.PC",
                byte_offset: self.tc0_pc_csr,
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tcs_core_state.PROGRAM_LOCATION",
                byte_offset: self.tc0_program_location_csr,
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.VWAIT_STATE",
                byte_offset: self.tc0_vwait_state_csr,
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::RUN_STATUS_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.VWAIT_SYNC_FLAG_STATE",
                byte_offset: self.tc0_vwait_sync_flag_state_csr,
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.SCALAR_ALU_STATUS",
                byte_offset: self.tc0_vwait_sync_flag_state_csr.wrapping_add(8),
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::RUN_STATUS_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.MXU0_STATUS",
                byte_offset: self.tc0_vwait_sync_flag_state_csr.wrapping_add(16),
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::RUN_STATUS_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.MXU1_STATUS",
                byte_offset: self.tc0_vwait_sync_flag_state_csr.wrapping_add(24),
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::RUN_STATUS_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.XLU_STATUS",
                byte_offset: self.tc0_vwait_sync_flag_state_csr.wrapping_add(32),
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::RUN_STATUS_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.VMEM_FAULT_ADDR",
                byte_offset: self.tc0_vwait_sync_flag_state_csr.wrapping_add(40),
                access: CsrAccessPermission::WriteOneToClear,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.SMEM_FAULT_ADDR",
                byte_offset: self.tc0_vwait_sync_flag_state_csr.wrapping_add(48),
                access: CsrAccessPermission::WriteOneToClear,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_0.tc_main_status.IMEM_PARITY_ERR",
                byte_offset: self.tc0_vwait_sync_flag_state_csr.wrapping_add(56),
                access: CsrAccessPermission::WriteOneToClear,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_1.tcs_core_state.RUN_STATUS",
                byte_offset: tc1_base.wrapping_add(0x18),
                access: CsrAccessPermission::ReadOnly,
                reset_value: 1,
                bitfields: &Self::RUN_STATUS_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_1.tcs_core_state.PC",
                byte_offset: tc1_base.wrapping_add(0x58),
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "tc_1.tcs_core_state.PROGRAM_LOCATION",
                byte_offset: tc1_base.wrapping_add(0xA0),
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
        ]
    }

    /// Returns the structured `asic_sw.tools.RegisterTranslatorData` [`CsrRegisterDescriptor`] list
    /// for this chip generation.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn csr_descriptors(&self) -> Vec<CsrRegisterDescriptor> {
        let mut list = self.tensor_core_csr_descriptors();
        list.extend([
            CsrRegisterDescriptor {
                full_path: "ici_router.ICI_ROUTER_LINK_STATUS_0",
                byte_offset: BAR0_CSR_ICI_ROUTER_LINK_STATE,
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0x0F,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "ici_router.ICI_BARRIERS_STATE",
                byte_offset: BAR0_CSR_ICI_ROUTER_LINK_STATE.wrapping_add(8),
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "ici_router.ICI_CRC_ERROR_COUNT",
                byte_offset: BAR0_CSR_ICI_ROUTER_LINK_STATE.wrapping_add(16),
                access: CsrAccessPermission::WriteOneToClear,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "hbm_mc.HBM_MC_ECC_CORR_COUNT",
                byte_offset: BAR0_CSR_HBM_DMA_RING_HEAD.wrapping_add(0x100),
                access: CsrAccessPermission::WriteOneToClear,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "hbm_mc.HBM_MC_ECC_UNCORR_COUNT",
                byte_offset: BAR0_CSR_HBM_DMA_RING_HEAD.wrapping_add(0x108),
                access: CsrAccessPermission::WriteOneToClear,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "hbm_mc.HBM_BANDWIDTH_COUNTER",
                byte_offset: BAR0_CSR_HBM_DMA_RING_HEAD.wrapping_add(0x110),
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "chip_id.GLOBAL_CHIP_ID",
                byte_offset: self.global_chip_id_csr,
                access: CsrAccessPermission::ReadOnly,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "gasket_att.ATT_BASE",
                byte_offset: BAR0_CSR_GASKET_ATT_BASE,
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "gasket_att.EXTENDED_ATT_BASE",
                byte_offset: BAR0_CSR_GASKET_EXTENDED_ATT_BASE,
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "gasket_irq.MSI_X_TABLE",
                byte_offset: BAR0_CSR_GASKET_EXTENDED_ATT_BASE.wrapping_add(0x1000),
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "gasket_irq.INTERRUPT_MASK",
                byte_offset: BAR0_CSR_GASKET_EXTENDED_ATT_BASE.wrapping_add(0x1008),
                access: CsrAccessPermission::ReadWrite,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "gasket_irq.INTERRUPT_STATUS",
                byte_offset: BAR0_CSR_GASKET_EXTENDED_ATT_BASE.wrapping_add(0x1010),
                access: CsrAccessPermission::WriteOneToClear,
                reset_value: 0,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "telemetry.THERMAL_MILLICELSIUS",
                byte_offset: BAR0_CSR_THERMAL_MILLICELSIUS,
                access: CsrAccessPermission::ReadOnly,
                reset_value: 42_000,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "telemetry.POWER_MILLIWATTS",
                byte_offset: BAR0_CSR_POWER_MILLIWATTS,
                access: CsrAccessPermission::ReadOnly,
                reset_value: 195_000,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
            CsrRegisterDescriptor {
                full_path: "telemetry.DUTY_CYCLE_PCT",
                byte_offset: BAR0_CSR_POWER_MILLIWATTS.wrapping_add(8),
                access: CsrAccessPermission::ReadOnly,
                reset_value: 95,
                bitfields: &Self::VALUE64_BITFIELDS,
            },
        ]);
        if self.sc0_run_status_csr != 0 {
            let sc0_base = self.sc0_run_status_csr & !0xFF_u64;
            list.extend([
                CsrRegisterDescriptor {
                    full_path: "sc_0.scs_core_state.CORE_RESET",
                    byte_offset: sc0_base,
                    access: CsrAccessPermission::WriteOnly,
                    reset_value: 0,
                    bitfields: &Self::VALUE64_BITFIELDS,
                },
                CsrRegisterDescriptor {
                    full_path: "sc_0.scs_core_state.RUN_STATUS",
                    byte_offset: self.sc0_run_status_csr,
                    access: CsrAccessPermission::ReadOnly,
                    reset_value: 1,
                    bitfields: &Self::RUN_STATUS_BITFIELDS,
                },
                CsrRegisterDescriptor {
                    full_path: "sc_0.scs_core_state.PC",
                    byte_offset: sc0_base.wrapping_add(0x58),
                    access: CsrAccessPermission::ReadWrite,
                    reset_value: 0,
                    bitfields: &Self::VALUE64_BITFIELDS,
                },
                CsrRegisterDescriptor {
                    full_path: "sc_0.scs_core_state.PROGRAM_LOCATION",
                    byte_offset: sc0_base.wrapping_add(0xA0),
                    access: CsrAccessPermission::ReadWrite,
                    reset_value: 0,
                    bitfields: &Self::VALUE64_BITFIELDS,
                },
                CsrRegisterDescriptor {
                    full_path: "sc_0.scs_core_state.DMA_QUEUE_HEAD",
                    byte_offset: sc0_base.wrapping_add(0xB0),
                    access: CsrAccessPermission::ReadWrite,
                    reset_value: 0,
                    bitfields: &Self::VALUE64_BITFIELDS,
                },
                CsrRegisterDescriptor {
                    full_path: "sc_0.scs_core_state.DMA_QUEUE_TAIL",
                    byte_offset: sc0_base.wrapping_add(0xB8),
                    access: CsrAccessPermission::ReadWrite,
                    reset_value: 0,
                    bitfields: &Self::VALUE64_BITFIELDS,
                },
                CsrRegisterDescriptor {
                    full_path: "sc_0.scs_core_state.GATHER_SCATTER_STATUS",
                    byte_offset: sc0_base.wrapping_add(0xC0),
                    access: CsrAccessPermission::ReadOnly,
                    reset_value: 0,
                    bitfields: &Self::RUN_STATUS_BITFIELDS,
                },
            ]);
        }
        list
    }

    /// Converts this [`ChipRegisterMap`]'s full CSR descriptor tree into a wire-compatible
    /// [`crate::proto::RegisterTranslatorDataProto`].
    #[must_use]
    pub fn to_register_translator_proto(&self) -> crate::proto::RegisterTranslatorDataProto {
        let registers = self
            .csr_descriptors()
            .into_iter()
            .map(|d| crate::proto::RegisterTranslatorEntryProto {
                full_path: String::from(d.full_path),
                byte_offset: d.byte_offset,
                width_bits: 64,
                access_mode: String::from(match d.access {
                    CsrAccessPermission::ReadOnly => "RO",
                    CsrAccessPermission::ReadWrite => "RW",
                    CsrAccessPermission::WriteOnly => "WO",
                    CsrAccessPermission::WriteOneToClear => "W1C",
                }),
                reset_value: d.reset_value,
            })
            .collect();
        crate::proto::RegisterTranslatorDataProto {
            chip_codename: String::from(self.chip_codename),
            tpu_version: self.generation as i32,
            hardware_abi_version: self.hardware_abi_version,
            registers,
        }
    }

    /// Returns all [`CsrCatalogEntry`] records for this chip generation from [`global_csr_register_catalog`].
    #[must_use]
    pub fn catalog_entries(&self) -> Vec<CsrCatalogEntry> {
        global_csr_register_catalog()
            .iter()
            .copied()
            .filter(|e| e.generation == self.generation)
            .collect()
    }

    /// Looks up a hardware [`CsrRegisterDescriptor`] on this [`ChipRegisterMap`] by full or leaf symbolic name,
    /// searching both core architectural CSRs and the full [`global_csr_register_catalog`].
    ///
    /// # Arguments
    ///
    /// * `query_name` - Full hierarchical CSR path or leaf register name.
    #[must_use]
    pub fn lookup_by_name(&self, query_name: &str) -> Option<CsrRegisterDescriptor> {
        let trimmed = query_name.trim();
        if let Some(core_desc) = self.csr_descriptors().into_iter().find(|desc| {
            desc.full_path == trimmed
                || desc
                    .full_path
                    .rsplit_once('.')
                    .is_some_and(|(_, leaf)| leaf == trimmed)
        }) {
            return Some(core_desc);
        }
        global_csr_register_catalog()
            .iter()
            .find(|entry| {
                entry.generation == self.generation
                    && (entry.full_path == trimmed
                        || entry
                            .full_path
                            .rsplit_once('.')
                            .is_some_and(|(_, leaf)| leaf == trimmed))
            })
            .map(|e| e.to_descriptor())
    }

    /// Looks up a hardware [`CsrRegisterDescriptor`] on this [`ChipRegisterMap`] by physical BAR0 MMIO byte offset,
    /// searching both core architectural CSRs and the full [`global_csr_register_catalog`].
    ///
    /// # Arguments
    ///
    /// * `byte_offset` - Physical BAR0 MMIO byte offset.
    #[must_use]
    pub fn lookup_by_offset(&self, byte_offset: u64) -> Option<CsrRegisterDescriptor> {
        if let Some(core_desc) = self
            .csr_descriptors()
            .into_iter()
            .find(|desc| desc.byte_offset == byte_offset)
        {
            return Some(core_desc);
        }
        global_csr_register_catalog()
            .iter()
            .find(|entry| entry.generation == self.generation && entry.byte_offset == byte_offset)
            .map(|e| e.to_descriptor())
    }

    /// Looks up a hardware [`CsrRegisterDescriptor`] by full or suffix symbolic name
    /// (e.g., `"tc_0.tcs_core_state.RUN_STATUS"`, `"RUN_STATUS"`, `"PC"`, `"GLOBAL_CHIP_ID"`).
    ///
    /// # Arguments
    ///
    /// * `generation` - Cloud TPU hardware generation (`V2..=V7x`).
    /// * `query_name` - Full hierarchical CSR path or leaf register name.
    #[must_use]
    pub fn lookup_csr_by_name(
        generation: TpuGeneration,
        query_name: &str,
    ) -> Option<CsrRegisterDescriptor> {
        Self::for_generation(generation).lookup_by_name(query_name)
    }

    /// Looks up a hardware [`CsrRegisterDescriptor`] by physical BAR0 MMIO byte offset.
    ///
    /// # Arguments
    ///
    /// * `generation` - Cloud TPU hardware generation (`V2..=V7x`).
    /// * `byte_offset` - Physical BAR0 MMIO byte offset.
    #[must_use]
    pub fn lookup_csr_by_offset(
        generation: TpuGeneration,
        byte_offset: u64,
    ) -> Option<CsrRegisterDescriptor> {
        Self::for_generation(generation).lookup_by_offset(byte_offset)
    }

    /// Resolves a hierarchical `asic_sw` CSR path (including `csr_register_map.json` paths such as
    /// `"tile0.tensor_core0.pxc.run_status"`, `"ici.router.link_status_0"`, `"sparse_core0.scs.status"`,
    /// `"hbm.mc0.ecc_corr_count"`) to its [`CsrRegisterDescriptor`] on this [`ChipRegisterMap`].
    ///
    /// # Arguments
    ///
    /// * `hierarchical_path` - Hierarchical CSR register path string.
    #[must_use]
    pub fn resolve_hierarchical_csr_path(
        &self,
        hierarchical_path: &str,
    ) -> Option<CsrRegisterDescriptor> {
        if let Some(exact) = self.lookup_by_name(hierarchical_path) {
            return Some(exact);
        }
        let lower = hierarchical_path.trim().to_ascii_lowercase();
        if let Some(cat_match) = global_csr_register_catalog().iter().find(|entry| {
            entry.generation == self.generation && entry.full_path.to_ascii_lowercase() == lower
        }) {
            return Some(cat_match.to_descriptor());
        }
        let mapped_name = if lower.contains("tensor_core1") {
            if lower.contains("pc") {
                "tc_1.tcs_core_state.PC"
            } else if lower.contains("program_location") {
                "tc_1.tcs_core_state.PROGRAM_LOCATION"
            } else {
                "tc_1.tcs_core_state.RUN_STATUS"
            }
        } else if lower.contains("sparse_core") {
            if lower.contains("pc") {
                "sc_0.scs_core_state.PC"
            } else {
                "sc_0.scs_core_state.RUN_STATUS"
            }
        } else if lower.contains("ici") {
            "ici_router.ICI_ROUTER_LINK_STATUS_0"
        } else if lower.contains("ecc_corr") {
            "hbm_mc.HBM_MC_ECC_CORR_COUNT"
        } else if lower.contains("ecc_uncorr") {
            "hbm_mc.HBM_MC_ECC_UNCORR_COUNT"
        } else if lower.contains("thermal") {
            "telemetry.THERMAL_MILLICELSIUS"
        } else if lower.contains("power") {
            "telemetry.POWER_MILLIWATTS"
        } else if lower.contains("run_status") {
            "tc_0.tcs_core_state.RUN_STATUS"
        } else {
            return None;
        };
        self.lookup_by_name(mapped_name)
    }
}

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Run Control CSR.
pub const BAR0_CSR_TENSORCORE_RUN_CONTROL: u64 = 0x0000_1000;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 `SyncFlag` Completion Counter CSR.
pub const BAR0_CSR_TENSORCORE_SYNC_FLAG: u64 = 0x0000_1008;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Execution Status CSR.
pub const BAR0_CSR_TENSORCORE_STATUS: u64 = 0x0000_1010;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Program Counter CSR.
pub const BAR0_CSR_TENSORCORE_PC: u64 = 0x0000_1018;

/// `PCIe` BAR0 MMIO register offset for `SyncFlag` table base (`0x2000`).
pub const BAR0_CSR_SYNC_FLAG_BASE: u64 = 0x0000_2000;

/// `PCIe` BAR0 MMIO register offset for `SparseCore` 0 Run Control CSR (`0x4000`).
pub const BAR0_CSR_SPARSECORE_RUN_CONTROL: u64 = 0x0000_4000;

/// `PCIe` BAR0 MMIO register offset for Gasket simple Address Translation Table (ATT) base (`0x5000`).
pub const BAR0_CSR_GASKET_ATT_BASE: u64 = 0x0000_5000;

/// `PCIe` BAR0 MMIO register offset for Gasket 2-level extended Address Translation Table (ATT) base (`0x6000`).
pub const BAR0_CSR_GASKET_EXTENDED_ATT_BASE: u64 = 0x0000_6000;

/// `PCIe` BAR0 MMIO register offset for HBM DMA descriptor ring head pointer (`0x8000`).
pub const BAR0_CSR_HBM_DMA_RING_HEAD: u64 = 0x0000_8000;

/// `PCIe` BAR0 MMIO register offset for HBM DMA descriptor ring tail doorbell (`0x8008`).
pub const BAR0_CSR_HBM_DMA_RING_TAIL: u64 = 0x0000_8008;

/// `PCIe` BAR0 MMIO register offset for ICI optical/electrical router link state (`0x9000`).
pub const BAR0_CSR_ICI_ROUTER_LINK_STATE: u64 = 0x0000_9000;

/// `PCIe` BAR0 MMIO register offset for ASIC junction temperature in millicelsius (`0xA000`).
pub const BAR0_CSR_THERMAL_MILLICELSIUS: u64 = 0x0000_A000;

/// `PCIe` BAR0 MMIO register offset for board power draw in milliwatts (`0xA008`).
pub const BAR0_CSR_POWER_MILLIWATTS: u64 = 0x0000_A008;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` duty cycle percentage (`0xA010`).
pub const BAR0_CSR_DUTY_CYCLE_PERCENT: u64 = 0x0000_A010;

/// Gasket 32-byte DMA page-table mapping payload (`gasket_page_table_ioctl`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct GasketPageTableIoctl {
    /// Page table index (`0` for `TensorCore` 0).
    pub page_table_index: u64,
    /// Size in bytes (page-aligned to 4096 bytes).
    pub size: u64,
    /// Host virtual memory address (`u64`).
    pub host_address: u64,
    /// Device DMA virtual address (`u64`).
    pub device_address: u64,
}

/// Gasket 32-byte interrupt `eventfd` registration payload (`gasket_interrupt_eventfd`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct GasketInterruptEventFd {
    /// Interrupt index (`0..N`).
    pub interrupt: u64,
    /// Linux `eventfd` file descriptor.
    pub event_fd: u64,
    /// Reserved fields (`0`).
    pub reserved: [u64; 2],
}

/// Gasket 24-byte `dma_buf` mapping payload (`gasket_page_table_dmabuf`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct GasketPageTableDmaBuf {
    /// Page table index (`0`).
    pub page_table_index: u64,
    /// Target device IOVA address.
    pub device_address: u64,
    /// Linux `dma_buf` file descriptor.
    pub dma_buf_fd: i32,
    /// Mapping flags.
    pub flags: u32,
}

/// VFIO 32-byte IOMMU DMA mapping payload (`vfio_iommu_type1_dma_map`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct VfioDmaMapIoctl {
    /// Structure size (`32` bytes).
    pub argsz: u32,
    /// Mapping flags (`0x3` = `READ | WRITE`).
    pub flags: u32,
    /// Host user-space virtual address.
    pub vaddr: u64,
    /// I/O virtual address in TPU HBM space.
    pub iova: u64,
    /// Mapping size in bytes.
    pub size: u64,
}

/// VFIO 24-byte IOMMU DMA unmapping payload (`vfio_iommu_type1_dma_unmap`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct VfioDmaUnmapIoctl {
    /// Structure size (`24` bytes).
    pub argsz: u32,
    /// Unmap flags (`0`).
    pub flags: u32,
    /// I/O virtual address to unmap.
    pub iova: u64,
    /// Size in bytes to unmap.
    pub size: u64,
}

/// VFIO 24-byte interrupt signaling header (`vfio_irq_set`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct VfioIrqSetIoctl {
    /// Structure size (`24` bytes).
    pub argsz: u32,
    /// Action and data flags (`VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER`).
    pub flags: u32,
    /// IRQ index (`VFIO_PCI_MSIX_IRQ_INDEX`).
    pub index: u32,
    /// First sub-index to configure.
    pub start: u32,
    /// Number of IRQs to configure.
    pub count: u32,
    /// Eventfd file descriptor payload.
    pub event_fd: i32,
}

/// VFIO 32-byte `PCIe` BAR region descriptor (`vfio_region_info`, `VFIO_DEVICE_GET_REGION_INFO = 0x3b6c`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct VfioRegionInfoIoctl {
    /// Structure size (`32` bytes).
    pub argsz: u32,
    /// Region capability flags (`0x1 = READ`, `0x2 = WRITE`, `0x4 = MMAP`).
    pub flags: u32,
    /// Region BAR index (`0` = BAR0 CSRs, `2` = BAR2 HBM/VMEM aperture).
    pub index: u32,
    /// Offset of extended capability chain (`0`).
    pub cap_offset: u32,
    /// Region size in bytes.
    pub size: u64,
    /// File offset within the VFIO device file descriptor for `mmap`.
    pub offset: u64,
}

const _: () = assert!(size_of::<VfioRegionInfoIoctl>() == 32);
const _: () = assert!(std::mem::offset_of!(VfioRegionInfoIoctl, argsz) == 0);
const _: () = assert!(std::mem::offset_of!(VfioRegionInfoIoctl, flags) == 4);
const _: () = assert!(std::mem::offset_of!(VfioRegionInfoIoctl, index) == 8);
const _: () = assert!(std::mem::offset_of!(VfioRegionInfoIoctl, cap_offset) == 12);
const _: () = assert!(std::mem::offset_of!(VfioRegionInfoIoctl, size) == 16);
const _: () = assert!(std::mem::offset_of!(VfioRegionInfoIoctl, offset) == 24);

const _: () = assert!(size_of::<GasketPageTableIoctl>() == 32);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, page_table_index) == 0);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, size) == 8);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, host_address) == 16);
const _: () = assert!(std::mem::offset_of!(GasketPageTableIoctl, device_address) == 24);
const _: () = assert!(size_of::<GasketInterruptEventFd>() == 32);
const _: () = assert!(std::mem::offset_of!(GasketInterruptEventFd, interrupt) == 0);
const _: () = assert!(std::mem::offset_of!(GasketInterruptEventFd, event_fd) == 8);
const _: () = assert!(std::mem::offset_of!(GasketInterruptEventFd, reserved) == 16);
const _: () = assert!(size_of::<GasketPageTableDmaBuf>() == 24);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, page_table_index) == 0);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, device_address) == 8);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, dma_buf_fd) == 16);
const _: () = assert!(std::mem::offset_of!(GasketPageTableDmaBuf, flags) == 20);
const _: () = assert!(size_of::<VfioDmaMapIoctl>() == 32);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, argsz) == 0);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, flags) == 4);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, vaddr) == 8);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, iova) == 16);
const _: () = assert!(std::mem::offset_of!(VfioDmaMapIoctl, size) == 24);
const _: () = assert!(size_of::<VfioDmaUnmapIoctl>() == 24);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, argsz) == 0);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, flags) == 4);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, iova) == 8);
const _: () = assert!(std::mem::offset_of!(VfioDmaUnmapIoctl, size) == 16);
const _: () = assert!(size_of::<VfioIrqSetIoctl>() == 24);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, argsz) == 0);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, flags) == 4);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, index) == 8);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, start) == 12);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, count) == 16);
const _: () = assert!(std::mem::offset_of!(VfioIrqSetIoctl, event_fd) == 20);

bitflags::bitflags! {
    /// Strongly-typed bitmask flags for `BAR0_CSR_TENSORCORE_RUN_CONTROL`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TensorCoreControlFlags: u64 {
        /// Starts `TensorCore` instruction execution (`1 << 0`).
        const START = 1 << 0;
        /// Halts `TensorCore` instruction execution (`1 << 1`).
        const HALT = 1 << 1;
        /// Advances `TensorCore` by a single VLIW bundle (`1 << 2`).
        const SINGLE_STEP = 1 << 2;
        /// Performs a soft architectural reset of the `TensorCore` (`1 << 3`).
        const SOFT_RESET = 1 << 3;
        /// Clears sticky fault bits in the `TensorCore` status CSR (`1 << 4`).
        const CLEAR_FAULT = 1 << 4;
    }

    /// Strongly-typed bitmask flags for `BAR0_CSR_TENSORCORE_STATUS`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TensorCoreStatusFlags: u64 {
        /// `TensorCore` is idle and ready for launch (`1 << 0`).
        const IDLE = 1 << 0;
        /// `TensorCore` is actively executing bundles (`1 << 1`).
        const RUNNING = 1 << 1;
        /// `TensorCore` has completed the last launched program (`1 << 2`).
        const COMPLETED = 1 << 2;
        /// Uncorrectable HBM ECC error detected (`1 << 3`).
        const HBM_ECC_ERROR = 1 << 3;
        /// Matrix Multiply Unit (MXU) arithmetic fault (`1 << 4`).
        const MXU_OVERFLOW = 1 << 4;
        /// Architectural `SyncFlag` barrier timeout (`1 << 5`).
        const SYNC_FLAG_TIMEOUT = 1 << 5;
        /// Device execution has been poisoned (`1 << 6`).
        const POISONED = 1 << 6;
    }

    /// Strongly-typed bitmask flags for `VFIO_IOMMU_MAP_DMA` (`VfioDmaMapIoctl::flags`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VfioDmaMapFlags: u32 {
        /// Device may read from the mapped DMA region (`1 << 0`).
        const READ = 1 << 0;
        /// Device may write to the mapped DMA region (`1 << 1`).
        const WRITE = 1 << 1;
    }

    /// Strongly-typed bitmask flags for `VFIO_DEVICE_SET_IRQS` (`VfioIrqSetIoctl::flags`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VfioIrqSetFlags: u32 {
        /// No payload data (`1 << 0`).
        const DATA_NONE = 1 << 0;
        /// Boolean payload (`1 << 1`).
        const DATA_BOOL = 1 << 1;
        /// Linux `eventfd` payload (`1 << 2`).
        const DATA_EVENTFD = 1 << 2;
        /// Mask the interrupt (`1 << 3`).
        const ACTION_MASK = 1 << 3;
        /// Unmask the interrupt (`1 << 4`).
        const ACTION_UNMASK = 1 << 4;
        /// Trigger the interrupt (`1 << 5`).
        const ACTION_TRIGGER = 1 << 5;
    }

    /// Strongly-typed bitmask flags for `BAR0_CSR_ICI_ROUTER_LINK_STATE` (`0x9000`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct IciLinkStateFlags: u64 {
        /// Inter-Chip Interconnect (ICI) Port 0 link is up (`1 << 0`).
        const PORT_0_UP = 1 << 0;
        /// Inter-Chip Interconnect (ICI) Port 1 link is up (`1 << 1`).
        const PORT_1_UP = 1 << 1;
        /// Inter-Chip Interconnect (ICI) Port 2 link is up (`1 << 2`).
        const PORT_2_UP = 1 << 2;
        /// Inter-Chip Interconnect (ICI) Port 3 link is up (`1 << 3`).
        const PORT_3_UP = 1 << 3;
        /// All 4 torus optical/electrical ICI links are healthy (`0x0F`).
        const ALL_PORTS_HEALTHY = Self::PORT_0_UP.bits()
            | Self::PORT_1_UP.bits()
            | Self::PORT_2_UP.bits()
            | Self::PORT_3_UP.bits();
    }
}

/// Hardware 32-byte HBM DMA ring descriptor (`HbmDmaRingDescriptor`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct HbmDmaRingDescriptor {
    /// Source IOVA or HBM physical address.
    pub src_iova: u64,
    /// Destination IOVA or HBM physical address.
    pub dst_iova: u64,
    /// Transfer length in bytes.
    pub length_bytes: u32,
    /// DMA direction and attribute flags (`0` = H2D, `1` = D2H, `2` = D2D).
    pub direction_flags: u32,
    /// Architectural `SyncFlag` ID incremented upon DMA completion.
    pub completion_sync_flag: u64,
}

const _: () = assert!(size_of::<HbmDmaRingDescriptor>() == 32);

/// Type alias matching the architectural DMA ring descriptor name (`DmaRingDescriptor`).
pub type DmaRingDescriptor = HbmDmaRingDescriptor;
