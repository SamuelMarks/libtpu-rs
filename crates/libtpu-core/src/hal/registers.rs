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
}

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Run Control CSR.
pub const BAR0_CSR_TENSORCORE_RUN_CONTROL: u64 = 0x0000_1000;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 `SyncFlag` Completion Counter CSR.
pub const BAR0_CSR_TENSORCORE_SYNC_FLAG: u64 = 0x0000_1008;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Program Counter CSR.
pub const BAR0_CSR_TENSORCORE_PC: u64 = 0x0000_1008;

/// `PCIe` BAR0 MMIO register offset for `TensorCore` 0 Execution Status CSR.
pub const BAR0_CSR_TENSORCORE_STATUS: u64 = 0x0000_1010;

/// `PCIe` BAR0 MMIO register offset for `SyncFlag` table base (`0x2000`).
pub const BAR0_CSR_SYNC_FLAG_BASE: u64 = 0x0000_2000;

/// `PCIe` BAR0 MMIO register offset for `SparseCore` 0 Run Control CSR (`0x4000`).
pub const BAR0_CSR_SPARSECORE_RUN_CONTROL: u64 = 0x0000_4000;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
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
