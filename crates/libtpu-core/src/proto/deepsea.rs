//! `DeepSea` / Jellyfish VLIW ISA, core programs, host commands, and TPU executables.
//!
//! Synchronized with `isa_program.proto`, `deepsea_executable.proto`, and `host_command_opcodes.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use crate::proto::hlo::HloModuleProtoWithConfig;
use crate::proto::pjrt::CompileOptionsProto;
use crate::proto::xla_data::ShapeProto;
use prost::Message;

/// Wire-compatible `tpu.TpuSerializedExecutableProto` (`tpu_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuSerializedExecutableProto {
    /// Executable name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// SHA-256 fingerprint hex string (field `2`).
    #[prost(string, tag = "2")]
    pub fingerprint: String,
    /// Number of replicas (field `3`).
    #[prost(int32, tag = "3")]
    pub num_replicas: i32,
    /// Number of partitions (field `4`).
    #[prost(int32, tag = "4")]
    pub num_partitions: i32,
    /// Encoded VLIW instruction bundle opcodes (field `5`).
    #[prost(int64, repeated, packed = "true", tag = "5")]
    pub instruction_opcodes: Vec<i64>,
    /// Output shape protos (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub output_shapes: Vec<ShapeProto>,
    /// Serialized `CompileOptionsProto` (field `7`).
    #[prost(bytes = "vec", tag = "7")]
    pub compile_options_bytes: Vec<u8>,
    /// Serialized HLO / `StableHLO` module text or proto (field `8`).
    #[prost(bytes = "vec", tag = "8")]
    pub hlo_module_bytes: Vec<u8>,
}

/// Type alias for [`TpuSerializedExecutableProto`] (`TpuExecutableSerializedProto`).
pub type TpuExecutableSerializedProto = TpuSerializedExecutableProto;

impl_proto_schema_message!(
    TpuSerializedExecutableProto,
    "tpu.TpuSerializedExecutableProto"
);

/// Wire-compatible `xla.IsaBundleSlotProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaBundleSlotProto {
    /// Slot unit kind (`0` = Scalar/SALU, `1` = Vector/VALU, `2` = Matrix/MXU, `3` = DMA, `4` = Control, field `1`).
    #[prost(uint32, tag = "1")]
    pub unit_kind: u32,
    /// Hardware opcode (`TpuOpCode`, field `2`).
    #[prost(uint32, tag = "2")]
    pub opcode: u32,
    /// Destination register or VMEM slot (field `3`).
    #[prost(uint32, tag = "3")]
    pub dst_reg: u32,
    /// Source 0 register or VMEM slot (field `4`).
    #[prost(uint32, tag = "4")]
    pub src0_reg: u32,
    /// Source 1 register or VMEM slot (field `5`).
    #[prost(uint32, tag = "5")]
    pub src1_reg: u32,
    /// Predicate register ID (field `6`).
    #[prost(uint32, tag = "6")]
    pub predicate_reg: u32,
    /// Immediate operand or `SyncFlag` ID (field `7`).
    #[prost(int64, tag = "7")]
    pub immediate: i64,
}

impl_proto_schema_message!(IsaBundleSlotProto, "xla.IsaBundleSlotProto");

/// Wire-compatible `xla.IsaVliwBundleProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaVliwBundleProto {
    /// Bundle program counter index (field `1`).
    #[prost(uint32, tag = "1")]
    pub pc_index: u32,
    /// Scalar slot 0 instruction (field `2`).
    #[prost(message, optional, tag = "2")]
    pub scalar_s0: Option<IsaBundleSlotProto>,
    /// Scalar slot 1 instruction (field `3`).
    #[prost(message, optional, tag = "3")]
    pub scalar_s1: Option<IsaBundleSlotProto>,
    /// Vector ALU slot 0 instruction (field `4`).
    #[prost(message, optional, tag = "4")]
    pub vector_v0: Option<IsaBundleSlotProto>,
    /// Vector ALU slot 1 instruction (field `5`).
    #[prost(message, optional, tag = "5")]
    pub vector_v1: Option<IsaBundleSlotProto>,
    /// Matrix Multiply Unit (`MXU`) slot 0 instruction (field `6`).
    #[prost(message, optional, tag = "6")]
    pub matrix_m0: Option<IsaBundleSlotProto>,
    /// Matrix Multiply Unit (`MXU`) slot 1 instruction (field `7`).
    #[prost(message, optional, tag = "7")]
    pub matrix_m1: Option<IsaBundleSlotProto>,
    /// Cross-Lane Unit (`XLU`) unary slot instruction (field `8`).
    #[prost(message, optional, tag = "8")]
    pub unary_xlu: Option<IsaBundleSlotProto>,
    /// Async DMA engine slot instruction (field `9`).
    #[prost(message, optional, tag = "9")]
    pub dma_d0: Option<IsaBundleSlotProto>,
    /// Low-Level Optimizer (`LLO`) debug annotation string (field `10`).
    #[prost(string, tag = "10")]
    pub llo_annotation: String,
}

impl_proto_schema_message!(IsaVliwBundleProto, "xla.IsaVliwBundleProto");

/// Wire-compatible `xla.IsaCoreProgramHeaderProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaCoreProgramHeaderProto {
    /// Core index within the chip (field `1`).
    #[prost(uint32, tag = "1")]
    pub core_index: u32,
    /// Core type (`0` = `TensorCore`, `1` = `SparseCore`, `2` = `BarnaCore`, field `2`).
    #[prost(uint32, tag = "2")]
    pub core_type: u32,
    /// Entry program counter offset (field `3`).
    #[prost(uint64, tag = "3")]
    pub entry_pc: u64,
    /// Number of 64-byte VLIW bundles (field `4`).
    #[prost(uint64, tag = "4")]
    pub bundle_count: u64,
    /// Raw encoded VLIW stream bytes (field `5`).
    #[prost(bytes = "vec", tag = "5")]
    pub raw_vliw_stream: Vec<u8>,
    /// Structured VLIW bundles (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub structured_bundles: Vec<IsaVliwBundleProto>,
}

impl_proto_schema_message!(IsaCoreProgramHeaderProto, "xla.IsaCoreProgramHeaderProto");

/// Wire-compatible `xla.IsaSymbolTableEntryProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaSymbolTableEntryProto {
    /// Symbol name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Bundle PC offset (field `2`).
    #[prost(uint64, tag = "2")]
    pub pc_offset: u64,
    /// Symbol size in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub size_bytes: u64,
}

impl_proto_schema_message!(IsaSymbolTableEntryProto, "xla.IsaSymbolTableEntryProto");

/// Wire-compatible `platforms_deepsea.jellyfish.ProgramRelocation` (`isa_program.proto` field `38`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProgramRelocationProto {
    /// Byte offset inside the encoded VLIW instruction stream (`IMEM`, field `1`).
    #[prost(int64, tag = "1")]
    pub offset: i64,
    /// Relocation kind (`0` = HBM parameter, `1` = HBM output, `2` = VMEM ping/pong, `3` = constant pool, field `2`).
    #[prost(int32, tag = "2")]
    pub kind: i32,
    /// Target parameter/output/constant ordinal index (field `3`).
    #[prost(int64, tag = "3")]
    pub target_index: i64,
    /// Signed byte addend (field `4`).
    #[prost(int64, tag = "4")]
    pub addend: i64,
}

impl ProgramRelocationProto {
    /// Patches live parameter and output IOVA addresses (`param_iovas`, `output_iovas`) into `imem_bytes`
    /// at `self.offset` (`32`-bit immediate field of the target VLIW slot).
    ///
    /// # Arguments
    ///
    /// * `imem_bytes` - Mutable `IMEM` VLIW instruction stream.
    /// * `param_iovas` - Live device HBM/IOVA addresses for program input parameters (`kind == 0`).
    /// * `output_iovas` - Live device HBM/IOVA addresses for program output buffers (`kind == 1`).
    pub fn apply_to_imem(&self, imem_bytes: &mut [u8], param_iovas: &[u64], output_iovas: &[u64]) {
        let idx = usize::try_from(self.target_index.max(0)).unwrap_or(0);
        let base_addr = if self.kind == 0 {
            param_iovas.get(idx).copied().unwrap_or(0)
        } else {
            output_iovas.get(idx).copied().unwrap_or(0)
        };
        let patched_u64 = base_addr.wrapping_add_signed(self.addend);
        #[allow(clippy::cast_possible_truncation)]
        let patched_u32 = (patched_u64 & 0xFFFF_FFFF) as u32;
        let off = usize::try_from(self.offset.max(0)).unwrap_or(0);
        imem_bytes
            .get_mut(off..off.saturating_add(4))
            .into_iter()
            .for_each(|dst| dst.copy_from_slice(&patched_u32.to_le_bytes()));
    }
}

impl_proto_schema_message!(ProgramRelocationProto, "xla.ProgramRelocation");

/// Wire-compatible `xla.IsaProgramProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaProgramProto {
    /// Target ISA architecture string (e.g., `"vfc"`, field `1`).
    #[prost(string, tag = "1")]
    pub isa_architecture: String,
    /// ISA ABI version integer (field `2`).
    #[prost(int64, tag = "2")]
    pub isa_abi_version: i64,
    /// Per-core program headers (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub core_programs: Vec<IsaCoreProgramHeaderProto>,
    /// Symbol table entries (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub symbols: Vec<IsaSymbolTableEntryProto>,
    /// Deterministic 64-bit program fingerprint (field `5`).
    #[prost(uint64, tag = "5")]
    pub fingerprint: u64,
    /// Scalar Memory (`SMEM`) footprint in bytes (field `6`).
    #[prost(int64, tag = "6")]
    pub smem_size_bytes: i64,
    /// Vector Memory (`VMEM`) footprint in bytes (field `7`).
    #[prost(int64, tag = "7")]
    pub vmem_size_bytes: i64,
    /// Chip Memory (`CMEM`) footprint in bytes (field `8`).
    #[prost(int64, tag = "8")]
    pub cmem_size_bytes: i64,
    /// Instruction Memory (`IMEM`) footprint in bytes (field `9`).
    #[prost(int64, tag = "9")]
    pub imem_size_bytes: i64,
    /// Scalar stack frame size in bytes (field `10`).
    #[prost(int64, tag = "10")]
    pub stack_frame_size_bytes: i64,
    /// Allocated hardware `SyncFlag` counter count (field `11`).
    #[prost(int32, tag = "11")]
    pub sync_flag_count: i32,
    /// Producing XLA/LLO compiler version string (field `12`).
    #[prost(string, tag = "12")]
    pub xla_compiler_version: String,
    /// Deterministic 64-bit LLO modulo-schedule fingerprint (field `13`).
    #[prost(uint64, tag = "13")]
    pub llo_schedule_fingerprint: u64,
    /// VLIW predication mode (`0` = full, `1` = scalar-only, `2` = unpredicated, field `14`).
    #[prost(int32, tag = "14")]
    pub predication_mode: i32,
    /// Bundle compression mode (`0` = uncompressed, `1` = NOP-elided, `2` = Brotli, field `15`).
    #[prost(int32, tag = "15")]
    pub bundle_compression_kind: i32,
    /// Unique compilation ID (field `20`).
    #[prost(uint64, tag = "20")]
    pub compilation_id: u64,
    /// `SparseCore` sequencer start bundle index (field `23`).
    #[prost(int64, tag = "23")]
    pub scs_start_bundle: i64,
    /// Polyhedral tile loop start bundle index (field `24`).
    #[prost(int64, tag = "24")]
    pub tile_start_bundle: i64,
    /// Program relocations for HBM/VMEM addresses (`platforms_deepsea.jellyfish.ProgramRelocation`, field `38`).
    #[prost(message, repeated, tag = "38")]
    pub relocations: Vec<ProgramRelocationProto>,
    /// Core program ABI version descriptor (`tpu.TpuCoreProgramAbiProto`, field `39`).
    #[prost(message, optional, tag = "39")]
    pub abi: Option<TpuCoreProgramAbiProto>,
    /// Human-readable program name (field `42`).
    #[prost(string, tag = "42")]
    pub program_name: String,
    /// Whether compilation ran out of VMEM/HBM (field `61`).
    #[prost(bool, tag = "61")]
    pub oom_during_compilation: bool,
}

impl_proto_schema_message!(IsaProgramProto, "xla.IsaProgramProto");

/// Wire-compatible `xla.TargetArgumentsProto` (`platforms/xla/service/jellyfish/target_arguments.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TargetArgumentsProto {
    /// TPU generation version (`tpu.TpuVersionProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub tpu_version: i32,
    /// Chip variant string (e.g., `"v5litepod"`, `"v6e"`, field `2`).
    #[prost(string, tag = "2")]
    pub chip_variant: String,
    /// Core clock frequency in MHz (field `3`).
    #[prost(int64, tag = "3")]
    pub clock_rate_mhz: i64,
    /// HBM capacity per chip in bytes (field `4`).
    #[prost(int64, tag = "4")]
    pub hbm_capacity_bytes: i64,
    /// VMEM capacity per core in bytes (field `5`).
    #[prost(int64, tag = "5")]
    pub vmem_capacity_bytes: i64,
    /// SMEM capacity per core in bytes (field `6`).
    #[prost(int64, tag = "6")]
    pub smem_capacity_bytes: i64,
    /// CMEM capacity per chip in bytes (field `7`).
    #[prost(int64, tag = "7")]
    pub cmem_capacity_bytes: i64,
    /// Number of `TensorCores` per chip (field `8`).
    #[prost(int32, tag = "8")]
    pub tensor_cores_per_chip: i32,
    /// Number of `SparseCores` per chip (field `9`).
    #[prost(int32, tag = "9")]
    pub sparse_cores_per_chip: i32,
    /// Matrix Multiply Unit (`MXU`) systolic dimension (e.g., `128`, field `10`).
    #[prost(int32, tag = "10")]
    pub mxu_dimension: i32,
    /// Inter-Chip Interconnect (`ICI`) link bandwidth in bits/sec (field `11`).
    #[prost(int64, tag = "11")]
    pub ici_bandwidth_bps: i64,
}

impl_proto_schema_message!(TargetArgumentsProto, "xla.TargetArgumentsProto");

/// Wire-compatible `xla.DeepseaOverlaySectionProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaOverlaySectionProto {
    /// Overlay ordinal ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub overlay_id: u32,
    /// Symbol name (e.g., `".text.main"`, field `2`).
    #[prost(string, tag = "2")]
    pub symbol_name: String,
    /// Target `IMEM` byte offset (field `3`).
    #[prost(uint64, tag = "3")]
    pub imem_offset_bytes: u64,
    /// Section payload size in bytes (field `4`).
    #[prost(uint64, tag = "4")]
    pub byte_size: u64,
    /// Encoded VLIW instruction bundle bytes (field `5`).
    #[prost(bytes = "vec", tag = "5")]
    pub encoded_bundles: Vec<u8>,
    /// Target core type (`0` = `TensorCore`, `1` = `SparseCore`, field `6`).
    #[prost(uint32, tag = "6")]
    pub core_type: u32,
}

impl_proto_schema_message!(DeepseaOverlaySectionProto, "xla.DeepseaOverlaySectionProto");

/// Wire-compatible `xla.DeepseaRelocationEntryProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaRelocationEntryProto {
    /// Bundle byte offset in `IMEM` where relocation applies (field `1`).
    #[prost(uint64, tag = "1")]
    pub bundle_offset_bytes: u64,
    /// VLIW slot index within the bundle (field `2`).
    #[prost(uint32, tag = "2")]
    pub slot_index: u32,
    /// Parameter or output buffer index being relocated (field `3`).
    #[prost(uint32, tag = "3")]
    pub buffer_index: u32,
    /// Constant byte addend (field `4`).
    #[prost(int64, tag = "4")]
    pub addend_bytes: i64,
    /// Target memory space (`0` = HBM, `1` = VMEM, `2` = SMEM, field `5`).
    #[prost(uint32, tag = "5")]
    pub memory_space: u32,
}

impl_proto_schema_message!(
    DeepseaRelocationEntryProto,
    "xla.DeepseaRelocationEntryProto"
);

/// Wire-compatible `xla.DeepseaConstantPoolEntryProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaConstantPoolEntryProto {
    /// Constant pool ordinal ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub constant_id: u32,
    /// Target memory space (`0` = HBM, `1` = VMEM, `2` = SMEM, field `2`).
    #[prost(uint32, tag = "2")]
    pub memory_space: u32,
    /// Byte offset in the constant pool segment (field `3`).
    #[prost(uint64, tag = "3")]
    pub offset_bytes: u64,
    /// Raw constant data bytes (field `4`).
    #[prost(bytes = "vec", tag = "4")]
    pub payload: Vec<u8>,
}

impl_proto_schema_message!(
    DeepseaConstantPoolEntryProto,
    "xla.DeepseaConstantPoolEntryProto"
);

/// Wire-compatible `xla.DeepseaSyncFlagInitProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaSyncFlagInitProto {
    /// Hardware `SyncFlag` register index (field `1`).
    #[prost(uint32, tag = "1")]
    pub sync_flag_id: u32,
    /// Initial counter value before program launch (field `2`).
    #[prost(uint32, tag = "2")]
    pub initial_value: u32,
    /// Debug label for the synchronization edge (field `3`).
    #[prost(string, tag = "3")]
    pub debug_label: String,
}

impl_proto_schema_message!(DeepseaSyncFlagInitProto, "xla.DeepseaSyncFlagInitProto");

/// Wire-compatible `tpu.TpuCoreProgramAbiOptionsProto` (`learning/45eac/tpu/runtime/tpu_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCoreProgramAbiOptionsProto {
    /// Whether hardware continuation dispatch is enabled (field `1`).
    #[prost(bool, tag = "1")]
    pub use_continuations: bool,
}

impl_proto_schema_message!(
    TpuCoreProgramAbiOptionsProto,
    "tpu.TpuCoreProgramAbiOptionsProto"
);

/// Wire-compatible `tpu.TpuCoreProgramAbiProto` (`learning/45eac/tpu/runtime/tpu_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCoreProgramAbiProto {
    /// Target TPU version enum (`TpuVersionProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub version: i32,
    /// Chip variant string (e.g., `"default"`, field `2`).
    #[prost(string, tag = "2")]
    pub variant: String,
    /// Chip configuration string (field `3`).
    #[prost(string, tag = "3")]
    pub chip_config: String,
    /// Base ABI version integer (field `4`).
    #[prost(int64, tag = "4")]
    pub base_abi_version: i64,
    /// Hardware ABI version integer (field `5`).
    #[prost(int64, tag = "5")]
    pub hardware_abi_version: i64,
    /// Required compiler SHA-256 hash string (field `6`).
    #[prost(string, tag = "6")]
    pub required_compiler_hash: String,
    /// Core program ABI options (field `7`).
    #[prost(message, optional, tag = "7")]
    pub options: Option<TpuCoreProgramAbiOptionsProto>,
}

impl_proto_schema_message!(TpuCoreProgramAbiProto, "tpu.TpuCoreProgramAbiProto");

/// Wire-compatible `xla.TpuMemorySegmentRangeProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuMemorySegmentRangeProto {
    /// Start offset in bytes (field `1`).
    #[prost(int64, tag = "1")]
    pub offset: i64,
    /// Size in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
}

impl_proto_schema_message!(TpuMemorySegmentRangeProto, "xla.TpuMemorySegmentRangeProto");

/// Wire-compatible `xla.TpuMemorySegmentProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuMemorySegmentProto {
    /// Segment type (`tpu.TpuSegmentTypeProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub segment_type: i32,
    /// Segment memory type (`tpu.TpuSegmentMemoryTypeProto`, field `2`).
    #[prost(int32, tag = "2")]
    pub memory_type: i32,
    /// Initialized data range within the segment (field `3`).
    #[prost(message, optional, tag = "3")]
    pub initialized_data_range: Option<TpuMemorySegmentRangeProto>,
    /// Segment size in bytes (field `4`).
    #[prost(int64, tag = "4")]
    pub segment_size_in_bytes: i64,
}

impl_proto_schema_message!(TpuMemorySegmentProto, "xla.TpuMemorySegmentProto");

/// Wire-compatible `xla.TpuMemorySegmentSetProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuMemorySegmentSetProto {
    /// Target memory space (`tpu.TpuSegmentMemoryTypeProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub memory_type: i32,
    /// Memory segments in this address space (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub segments: Vec<TpuMemorySegmentProto>,
    /// Initialized data bytes across the segment set (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub initialized_data: Vec<u8>,
    /// Segment set hash bytes (field `4`).
    #[prost(bytes = "vec", tag = "4")]
    pub hash: Vec<u8>,
    /// Compression algorithm enum (`0` = uncompressed, field `5`).
    #[prost(int32, tag = "5")]
    pub compression: i32,
}

impl_proto_schema_message!(TpuMemorySegmentSetProto, "xla.TpuMemorySegmentSetProto");

/// Wire-compatible `xla.TpuCoreProgramProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCoreProgramProto {
    /// Platform type identifier (field `2`).
    #[prost(int32, tag = "2")]
    pub platform_type: i32,
    /// Program fingerprint bytes (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub fingerprint: Vec<u8>,
    /// Unique compilation ID (field `4`).
    #[prost(uint64, tag = "4")]
    pub compilation_id: u64,
    /// Memory segment sets (`IMEM`, `SMEM`, `VMEM`, `HBM` rodata, field `8`).
    #[prost(message, repeated, tag = "8")]
    pub memory_segments: Vec<TpuMemorySegmentSetProto>,
    /// Concurrent program HBM stack size in bytes (field `9`).
    #[prost(int64, tag = "9")]
    pub concurrent_program_hbm_stack_size: i64,
    /// Whether this program has a concurrent continuation program (field `10`).
    #[prost(bool, tag = "10")]
    pub has_concurrent_program: bool,
    /// Core program ABI descriptor (field `12`).
    #[prost(message, optional, tag = "12")]
    pub abi: Option<TpuCoreProgramAbiProto>,
    /// Whether compilation ran out of VMEM/HBM (field `14`).
    #[prost(bool, tag = "14")]
    pub oom_during_compilation: bool,
    /// Symbol relocations applied to this core program (field `15`).
    #[prost(message, repeated, tag = "15")]
    pub relocations: Vec<ProgramRelocationProto>,
}

impl_proto_schema_message!(TpuCoreProgramProto, "xla.TpuCoreProgramProto");

/// Wire-compatible `xla.DeepseaExecutableProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaExecutableProto {
    /// Executable module name (field `1`).
    #[prost(string, tag = "1")]
    pub module_name: String,
    /// Embedded `IsaProgramProto` (field `2`).
    #[prost(message, optional, tag = "2")]
    pub isa_program: Option<IsaProgramProto>,
    /// Overlay sections to DMA into `IMEM`/`VMEM`/`SMEM` (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub overlays: Vec<DeepseaOverlaySectionProto>,
    /// Relocation entries for parameter and output buffers (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub relocations: Vec<DeepseaRelocationEntryProto>,
    /// Constant pool entries (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub constants: Vec<DeepseaConstantPoolEntryProto>,
    /// Initial `SyncFlag` values (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub sync_flags: Vec<DeepseaSyncFlagInitProto>,
    /// Target hardware arguments (field `7`).
    #[prost(message, optional, tag = "7")]
    pub target_arguments: Option<TargetArgumentsProto>,
    /// Scoped VMEM bytes allocated (field `8`).
    #[prost(int64, tag = "8")]
    pub vmem_allocation_bytes: i64,
    /// Scoped SMEM bytes allocated (field `9`).
    #[prost(int64, tag = "9")]
    pub smem_allocation_bytes: i64,
    /// Scoped CMEM bytes allocated (field `10`).
    #[prost(int64, tag = "10")]
    pub cmem_allocation_bytes: i64,
    /// HBM scratchpad bytes allocated (field `11`).
    #[prost(int64, tag = "11")]
    pub hbm_scratchpad_bytes: i64,
    /// Authentic `tpu.TpuCoreProgramProto` for `TensorCore` (field `12`).
    #[prost(message, optional, tag = "12")]
    pub core_program: Option<TpuCoreProgramProto>,
    /// Authentic `tpu.TpuCoreProgramProto` for `SparseCore` (field `13`).
    #[prost(message, optional, tag = "13")]
    pub sparse_core_program: Option<TpuCoreProgramProto>,
    /// Host command dispatch table embedded in this executable (field `14`).
    #[prost(message, repeated, tag = "14")]
    pub host_command_table: Vec<HostCommandEnvelopeProto>,
    /// ISA program relocations across all cores (field `15`).
    #[prost(message, repeated, tag = "15")]
    pub program_relocations: Vec<ProgramRelocationProto>,
    /// Embedded XDB debug symbol and HLO-to-LLO PC mapping metadata (field `16`).
    #[prost(message, optional, tag = "16")]
    pub xdb_compiler_metadata: Option<crate::proto::tpu::CompilerMetadataProto>,
}

impl_proto_schema_message!(DeepseaExecutableProto, "xla.DeepseaExecutableProto");

/// Wire-compatible `tpu.HostCommandEnvelopeProto` (`learning/45eac/tpu/runtime/host_commands/host_command_opcodes.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HostCommandEnvelopeProto {
    /// Host command opcode (`tpu.HostCommandOpcodeProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub opcode: i32,
    /// Continuation queue sequence ID (`ContinuationId`, field `2`).
    #[prost(uint64, tag = "2")]
    pub continuation_id: u64,
    /// Source device ID (field `3`).
    #[prost(uint32, tag = "3")]
    pub device_id: u32,
    /// Source core index (field `4`).
    #[prost(uint32, tag = "4")]
    pub core_index: u32,
    /// Registered callback or handler ID (field `5`).
    #[prost(uint64, tag = "5")]
    pub handler_id: u64,
    /// Serialized command payload bytes (field `6`).
    #[prost(bytes = "vec", tag = "6")]
    pub payload: Vec<u8>,
}

impl_proto_schema_message!(HostCommandEnvelopeProto, "tpu.HostCommandEnvelopeProto");

/// Wire-compatible `xla.TpuExecutableProto` (`learning/45eac/research/pjrt/tpu_executable.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct TpuExecutableProto {
    /// Embedded `DeepseaExecutableProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub ds_executable: Option<DeepseaExecutableProto>,
    /// Embedded `HloModuleProtoWithConfig` (field `3`).
    #[prost(message, optional, tag = "3")]
    pub hlo_module_with_config: Option<HloModuleProtoWithConfig>,
    /// Embedded `CompileOptionsProto` (field `6`).
    #[prost(message, optional, tag = "6")]
    pub compile_options: Option<CompileOptionsProto>,
    /// Embedded `TargetArgumentsProto` (field `7`).
    #[prost(message, optional, tag = "7")]
    pub target_arguments: Option<TargetArgumentsProto>,
    /// Legacy compatibility `TpuSerializedExecutableProto` payload (field `8`).
    #[prost(message, optional, tag = "8")]
    pub legacy_executable: Option<TpuSerializedExecutableProto>,
    /// Source URI of the compiled module (field `9`).
    #[prost(string, tag = "9")]
    pub source_uri: String,
}

impl_proto_schema_message!(TpuExecutableProto, "xla.TpuExecutableProto");

/// Wire-compatible `tpu.RegisterTranslatorEntryProto` representing a single CSR node in `*_pf_csrs.br`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RegisterTranslatorEntryProto {
    /// Hierarchical CSR path (e.g., `"vlc_tc_tcs_core_state.RUN_STATUS"`, field `1`).
    #[prost(string, tag = "1")]
    pub full_path: String,
    /// Physical BAR0 MMIO byte offset (field `2`).
    #[prost(uint64, tag = "2")]
    pub byte_offset: u64,
    /// Register width in bits (`32` or `64`, field `3`).
    #[prost(uint32, tag = "3")]
    pub width_bits: u32,
    /// Access permission string (`"RO"`, `"RW"`, `"WO"`, `"W1C"`, field `4`).
    #[prost(string, tag = "4")]
    pub access_mode: String,
    /// Hardware reset default value (field `5`).
    #[prost(uint64, tag = "5")]
    pub reset_value: u64,
}

impl_proto_schema_message!(
    RegisterTranslatorEntryProto,
    "tpu.RegisterTranslatorEntryProto"
);

/// Wire-compatible `tpu.RegisterTranslatorDataProto` decoded from Brotli-compressed `*_pf_csrs.br` tables.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RegisterTranslatorDataProto {
    /// Internal `DeepSea` ASIC codename (field `1`).
    #[prost(string, tag = "1")]
    pub chip_codename: String,
    /// Cloud TPU generation ordinal (`2..=7`, field `2`).
    #[prost(int32, tag = "2")]
    pub tpu_version: i32,
    /// Hardware ABI version (`65537` or `65539`, field `3`).
    #[prost(uint64, tag = "3")]
    pub hardware_abi_version: u64,
    /// Decoded CSR register descriptors (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub registers: Vec<RegisterTranslatorEntryProto>,
}

impl_proto_schema_message!(
    RegisterTranslatorDataProto,
    "tpu.RegisterTranslatorDataProto"
);

/// Wire-compatible `tpu.TpuChipConfigProto` decoded from `*_chip_configs_*.binarypb`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuChipConfigProto {
    /// Chip configuration name (e.g., `"pufferfish_megacore_dense"`, field `1`).
    #[prost(string, tag = "1")]
    pub chip_name: String,
    /// Cloud TPU generation ordinal (`2..=7`, field `2`).
    #[prost(int32, tag = "2")]
    pub tpu_version: i32,
    /// Number of `TensorCore`s per chip (`1` or `2`, field `3`).
    #[prost(uint32, tag = "3")]
    pub tensor_cores_per_chip: u32,
    /// Number of `SparseCore`s per chip (`0`, `2`, or `4`, field `4`).
    #[prost(uint32, tag = "4")]
    pub sparse_cores_per_chip: u32,
    /// Total HBM capacity in bytes per chip (field `5`).
    #[prost(uint64, tag = "5")]
    pub hbm_capacity_bytes: u64,
    /// Scoped VMEM capacity in bytes per core (field `6`).
    #[prost(uint64, tag = "6")]
    pub vmem_capacity_bytes: u64,
    /// Scoped SMEM capacity in bytes per core (field `7`).
    #[prost(uint64, tag = "7")]
    pub smem_capacity_bytes: u64,
    /// Whether `MegaCore` mode fuses two `TensorCores` into one logical device (field `8`).
    #[prost(bool, tag = "8")]
    pub megacore_enabled: bool,
}

impl_proto_schema_message!(TpuChipConfigProto, "tpu.TpuChipConfigProto");

/// Wire-compatible `tpu.TpuChipPartsProto` decoded from `*_chip_parts.binarypb` inside `filewrapper_toc`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuChipPartsProto {
    /// Physical ASIC die/part codename (e.g., `"jellyfish"`, `"pufferfish"`, `"viperfish"`, field `1`).
    #[prost(string, tag = "1")]
    pub part_name: String,
    /// Cloud TPU generation ordinal (`2..=9`, field `2`).
    #[prost(int32, tag = "2")]
    pub tpu_version: i32,
    /// Matrix Multiply Unit (`MXU`) systolic array dimension (`128` or `256`, field `3`).
    #[prost(uint32, tag = "3")]
    pub mxu_dimension: u32,
    /// Number of `MXU` systolic units per `TensorCore` (`1`, `2`, or `4`, field `4`).
    #[prost(uint32, tag = "4")]
    pub mxu_count_per_core: u32,
    /// Number of Inter-Chip Interconnect (`ICI`) optical links per chip (`4` or `6`, field `5`).
    #[prost(uint32, tag = "5")]
    pub ici_link_count: u32,
    /// Nominal `TensorCore` clock rate in MHz (field `6`).
    #[prost(int64, tag = "6")]
    pub clock_rate_mhz: i64,
    /// Per-link `ICI` bandwidth in bits per second (field `7`).
    #[prost(int64, tag = "7")]
    pub ici_bandwidth_bps: i64,
    /// On-chip Collective Memory (`CMEM`) capacity per chip in bytes (field `8`).
    #[prost(uint64, tag = "8")]
    pub cmem_capacity_bytes: u64,
}

impl_proto_schema_message!(TpuChipPartsProto, "tpu.TpuChipPartsProto");

/// Wire-compatible `tpu.TpuRuntimeAbiProto` decoded from `current_tpu_runtime_abi.binarypb` inside `filewrapper_toc`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuRuntimeAbiProto {
    /// Base runtime ABI version integer (`1`, field `1`).
    #[prost(int64, tag = "1")]
    pub base_abi_version: i64,
    /// Hardware ISA ABI version (`0x10001` or `0x10003`, field `2`).
    #[prost(int64, tag = "2")]
    pub hardware_abi_version: i64,
    /// Embedded `deepsea_compiler_hash.txt` SHA-256 or build digest (field `3`).
    #[prost(string, tag = "3")]
    pub compiler_hash: String,
    /// Supported ASIC chip configuration names (field `4`).
    #[prost(string, repeated, tag = "4")]
    pub supported_chip_variants: Vec<String>,
}

impl_proto_schema_message!(TpuRuntimeAbiProto, "tpu.TpuRuntimeAbiProto");

/// Wire-compatible `xla.HostOffloadingExecutableProto` (`xla/core/host_offloading/host_offloading_executable.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct HostOffloadingExecutableProto {
    /// Host-offloaded `xla.HloModuleProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub hlo_module: Option<crate::proto::hlo::HloModuleProto>,
    /// Host executable backend type (`0` = Unspecified, `1` = PJRT, `2` = `NanoRT`, field `2`).
    #[prost(int32, tag = "2")]
    pub executable_type: i32,
}

impl_proto_schema_message!(
    HostOffloadingExecutableProto,
    "xla.HostOffloadingExecutableProto"
);

/// Wire-compatible `tpu.TpuChipVariantDescriptorProto` (`tpu_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuChipVariantDescriptorProto {
    /// TPU version enum (`TpuVersionProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub version: i32,
    /// Chip codename string (field `2`).
    #[prost(string, tag = "2")]
    pub codename: String,
    /// `PCIe` vendor ID (`0x1ae0`, field `3`).
    #[prost(uint32, tag = "3")]
    pub pcie_vendor_id: u32,
    /// `PCIe` device ID (field `4`).
    #[prost(uint32, tag = "4")]
    pub pcie_device_id: u32,
    /// Total CSR count in BAR0 (field `5`).
    #[prost(uint32, tag = "5")]
    pub csr_count: u32,
    /// BAR0 MMIO aperture span in bytes (field `6`).
    #[prost(uint64, tag = "6")]
    pub bar0_span_bytes: u64,
}

impl_proto_schema_message!(
    TpuChipVariantDescriptorProto,
    "tpu.TpuChipVariantDescriptorProto"
);

/// Wire-compatible `xla.ffi.ExecutionStateProto` (`xla/ffi/execution_state.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct FfiExecutionStateProto {
    /// Registered XLA FFI state type name (field `1`).
    #[prost(string, tag = "1")]
    pub type_name: String,
    /// Serialized opaque state payload bytes (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub state: Vec<u8>,
}

impl_proto_schema_message!(FfiExecutionStateProto, "xla.ffi.ExecutionStateProto");

/// Wire-compatible `xla.ffi.ScalarProto` (`xla/ffi/attribute_map.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FfiScalarProto {
    /// Boolean scalar value (field `1`).
    #[prost(bool, tag = "1")]
    pub b: bool,
    /// Signed 32-bit integer scalar value (field `4`).
    #[prost(int32, tag = "4")]
    pub i32: i32,
    /// Signed 64-bit integer scalar value (field `5`).
    #[prost(int64, tag = "5")]
    pub i64: i64,
    /// 32-bit float scalar value (field `10`).
    #[prost(float, tag = "10")]
    pub f32: f32,
    /// 64-bit float scalar value (field `11`).
    #[prost(double, tag = "11")]
    pub f64: f64,
}

impl_proto_schema_message!(FfiScalarProto, "xla.ffi.ScalarProto");

/// Wire-compatible `xla.ffi.ArrayProto` (`xla/ffi/attribute_map.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FfiArrayProto {
    /// Signed 32-bit integer array elements (field `3`).
    #[prost(int32, repeated, packed = "true", tag = "3")]
    pub i32_values: Vec<i32>,
    /// Signed 64-bit integer array elements (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub i64_values: Vec<i64>,
    /// 32-bit float array elements (field `9`).
    #[prost(float, repeated, packed = "true", tag = "9")]
    pub f32_values: Vec<f32>,
    /// 64-bit float array elements (field `10`).
    #[prost(double, repeated, packed = "true", tag = "10")]
    pub f64_values: Vec<f64>,
}

impl_proto_schema_message!(FfiArrayProto, "xla.ffi.ArrayProto");

/// Wire-compatible `xla.ffi.FlatAttributeProto` (`xla/ffi/attribute_map.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FfiFlatAttributeProto {
    /// Scalar attribute payload (field `1`).
    #[prost(message, optional, tag = "1")]
    pub scalar: Option<FfiScalarProto>,
    /// Array attribute payload (field `2`).
    #[prost(message, optional, tag = "2")]
    pub array: Option<FfiArrayProto>,
    /// Raw string or byte attribute value (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub str: Vec<u8>,
}

impl_proto_schema_message!(FfiFlatAttributeProto, "xla.ffi.FlatAttributeProto");

/// Wire-compatible `xla.ffi.FlatAttributesMapProto` (`xla/ffi/attribute_map.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FfiFlatAttributesMapProto {
    /// Flat attribute dictionary keyed by attribute name (field `1`).
    #[prost(btree_map = "string, message", tag = "1")]
    pub attrs: std::collections::BTreeMap<String, FfiFlatAttributeProto>,
}

impl_proto_schema_message!(FfiFlatAttributesMapProto, "xla.ffi.FlatAttributesMapProto");

/// Wire-compatible `xla.ffi.XlaFfiHandlerRegistrationProto` (`xla_ffi.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XlaFfiHandlerRegistrationProto {
    /// Custom call target name (field `1`).
    #[prost(string, tag = "1")]
    pub target_name: String,
    /// Target platform name (e.g., `"TPU"`, field `2`).
    #[prost(string, tag = "2")]
    pub platform_name: String,
    /// `OpenXLA` FFI major version (`0`, field `3`).
    #[prost(uint32, tag = "3")]
    pub api_major_version: u32,
    /// `OpenXLA` FFI minor version (`2`, field `4`).
    #[prost(uint32, tag = "4")]
    pub api_minor_version: u32,
    /// Handler traits bitmask (`XLA_FFI_Handler_Traits`, field `5`).
    #[prost(uint32, tag = "5")]
    pub traits_mask: u32,
}

impl_proto_schema_message!(
    XlaFfiHandlerRegistrationProto,
    "xla.ffi.XlaFfiHandlerRegistrationProto"
);

/// Wire-compatible `xla.ffi.XlaFfiCallFrameMetadataProto` (`xla_ffi.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XlaFfiCallFrameMetadataProto {
    /// Invoked custom call target name (field `1`).
    #[prost(string, tag = "1")]
    pub custom_call_target: String,
    /// Number of input buffer arguments (field `2`).
    #[prost(uint32, tag = "2")]
    pub num_args: u32,
    /// Number of output return buffers (field `3`).
    #[prost(uint32, tag = "3")]
    pub num_rets: u32,
    /// Execution stage ordinal (`0 = Instantiate`, `1 = Prepare`, `2 = Initialize`, `3 = Execute`, field `4`).
    #[prost(int32, tag = "4")]
    pub execution_stage: i32,
    /// Target device ordinal (field `5`).
    #[prost(int32, tag = "5")]
    pub device_ordinal: i32,
    /// Wire-serialized `FlatAttributesMapProto` bytes (field `6`).
    #[prost(bytes = "vec", tag = "6")]
    pub serialized_attributes: Vec<u8>,
}

impl_proto_schema_message!(
    XlaFfiCallFrameMetadataProto,
    "xla.ffi.XlaFfiCallFrameMetadataProto"
);

/// Wire-compatible `xla.jellyfish.JellyfishHostTransferProto` (`jellyfish_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct JellyfishHostTransferProto {
    /// Host transfer channel ID (field `1`).
    #[prost(int64, tag = "1")]
    pub channel_id: i64,
    /// Transfer direction (`0 = HostToDevice`, `1 = DeviceToHost`, field `2`).
    #[prost(int32, tag = "2")]
    pub direction: i32,
    /// Transfer payload length in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub byte_size: u64,
    /// Associated hardware `SyncFlag` ID (field `4`).
    #[prost(uint32, tag = "4")]
    pub sync_flag_id: u32,
    /// Host rendezvous key string (field `5`).
    #[prost(string, tag = "5")]
    pub rendezvous_key: String,
}

impl_proto_schema_message!(
    JellyfishHostTransferProto,
    "xla.jellyfish.JellyfishHostTransferProto"
);

/// Wire-compatible `xla.jellyfish.JellyfishOverlayConfigProto` (`jellyfish_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct JellyfishOverlayConfigProto {
    /// Overlay identifier (field `1`).
    #[prost(uint32, tag = "1")]
    pub overlay_id: u32,
    /// Base `IMEM` byte offset (field `2`).
    #[prost(uint64, tag = "2")]
    pub imem_base_offset: u64,
    /// Overlay size in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub imem_size_bytes: u64,
    /// Whether to prefetch the overlay on program launch (field `4`).
    #[prost(bool, tag = "4")]
    pub prefetch_on_launch: bool,
}

impl_proto_schema_message!(
    JellyfishOverlayConfigProto,
    "xla.jellyfish.JellyfishOverlayConfigProto"
);

/// Wire-compatible `xla.jellyfish.JellyfishCoreProgramProto` (`jellyfish_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct JellyfishCoreProgramProto {
    /// Intra-chip core ordinal index (field `1`).
    #[prost(uint32, tag = "1")]
    pub core_index: u32,
    /// Core type (`0 = TensorCore`, `1 = SparseCore`, field `2`).
    #[prost(uint32, tag = "2")]
    pub core_type: u32,
    /// Raw 64-byte VLIW instruction bundle stream (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub encoded_vliw_stream: Vec<u8>,
    /// Entry program counter offset (field `4`).
    #[prost(uint64, tag = "4")]
    pub entry_pc: u64,
    /// Overlay descriptors for this core (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub overlays: Vec<JellyfishOverlayConfigProto>,
    /// Host transfer descriptors for this core (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub host_transfers: Vec<JellyfishHostTransferProto>,
}

impl_proto_schema_message!(
    JellyfishCoreProgramProto,
    "xla.jellyfish.JellyfishCoreProgramProto"
);

/// Wire-compatible `xla.jellyfish.JellyfishExecutableProto` (`jellyfish_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct JellyfishExecutableProto {
    /// Compiled module name (field `1`).
    #[prost(string, tag = "1")]
    pub module_name: String,
    /// Deterministic 64-bit compilation fingerprint (field `2`).
    #[prost(uint64, tag = "2")]
    pub compilation_fingerprint: u64,
    /// Target Cloud TPU generation (`TpuGeneration`, field `3`).
    #[prost(int32, tag = "3")]
    pub tpu_generation: i32,
    /// Per-core compiled programs (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub core_programs: Vec<JellyfishCoreProgramProto>,
    /// HBM scratchpad allocation in bytes (field `5`).
    #[prost(int64, tag = "5")]
    pub hbm_scratchpad_bytes: i64,
    /// Scoped VMEM limit in bytes (field `6`).
    #[prost(int64, tag = "6")]
    pub vmem_limit_bytes: i64,
}

impl_proto_schema_message!(
    JellyfishExecutableProto,
    "xla.jellyfish.JellyfishExecutableProto"
);

/// Wire-compatible `xla.barnacore.SparseCoreDmaDescriptorProto` (`barnacore_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreDmaDescriptorProto {
    /// DMA descriptor ordinal ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub descriptor_id: u32,
    /// Base HBM table byte offset (field `2`).
    #[prost(uint64, tag = "2")]
    pub hbm_base_offset: u64,
    /// Scratchpad memory (`Spmem`) byte offset (field `3`).
    #[prost(uint32, tag = "3")]
    pub spmem_offset: u32,
    /// Transfer size in bytes (field `4`).
    #[prost(uint32, tag = "4")]
    pub transfer_bytes: u32,
    /// Completion `SyncFlag` ID (field `5`).
    #[prost(uint32, tag = "5")]
    pub sync_flag_id: u32,
    /// True for scatter-add HBM writeback, false for gather read (field `6`).
    #[prost(bool, tag = "6")]
    pub is_scatter_write: bool,
}

impl_proto_schema_message!(
    SparseCoreDmaDescriptorProto,
    "xla.barnacore.SparseCoreDmaDescriptorProto"
);

/// Wire-compatible `xla.barnacore.BarnaCoreChannelConfigProto` (`barnacore_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BarnaCoreChannelConfigProto {
    /// Channel identifier (field `1`).
    #[prost(uint32, tag = "1")]
    pub channel_id: u32,
    /// `SparseCore` ordinal index (field `2`).
    #[prost(uint32, tag = "2")]
    pub sparse_core_index: u32,
    /// Maximum unique sparse IDs per partition (field `3`).
    #[prost(uint32, tag = "3")]
    pub max_unique_ids_per_partition: u32,
    /// Maximum total sparse IDs per partition (field `4`).
    #[prost(uint32, tag = "4")]
    pub max_ids_per_partition: u32,
    /// Associated embedding table name (field `5`).
    #[prost(string, tag = "5")]
    pub table_name: String,
}

impl_proto_schema_message!(
    BarnaCoreChannelConfigProto,
    "xla.barnacore.BarnaCoreChannelConfigProto"
);

/// Wire-compatible `xla.barnacore.SparseCoreProgramProto` (`barnacore_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreProgramProto {
    /// Target `SparseCore` ordinal ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub sparse_core_id: u32,
    /// Encoded `SparseCore` instruction bytecode stream (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub instruction_stream: Vec<u8>,
    /// Allocated `Spmem` scratchpad size in bytes (field `3`).
    #[prost(uint32, tag = "3")]
    pub spmem_scratchpad_bytes: u32,
    /// Associated gather/scatter DMA descriptors (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub dma_descriptors: Vec<SparseCoreDmaDescriptorProto>,
    /// Optimizer algorithm opcode (`0 = Adagrad`, `1 = Adam`, `2 = Ftrl`, `3 = Sgd`, `4 = Lion`, field `5`).
    #[prost(int32, tag = "5")]
    pub optimizer_opcode: i32,
    /// Number of Tile Execute Control (`TEC`) VLIW bundles (field `7`).
    #[prost(uint32, tag = "7")]
    pub tec_bundle_count: u32,
    /// Encoded `SparseCore` Sequencer (`SCS`) control stream bytes (field `8`).
    #[prost(bytes = "vec", tag = "8")]
    pub scs_control_stream: Vec<u8>,
    /// Per-tile `Spmem` scratchpad size in bytes (field `9`).
    #[prost(uint32, tag = "9")]
    pub tile_spmem_size_bytes: u32,
}

impl_proto_schema_message!(
    SparseCoreProgramProto,
    "xla.barnacore.SparseCoreProgramProto"
);

/// Wire-compatible `xla.barnacore.BarnaCoreExecutableProto` (`barnacore_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BarnaCoreExecutableProto {
    /// Executable identifier name (field `1`).
    #[prost(string, tag = "1")]
    pub executable_name: String,
    /// Hardware `SparseCore` ISA version (`1` = `SCv1`, `2` = `SCv2`, field `2`).
    #[prost(uint32, tag = "2")]
    pub sparse_core_version: u32,
    /// Configured `BarnaCore` communication channels (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub channels: Vec<BarnaCoreChannelConfigProto>,
    /// Per-`SparseCore` compiled programs (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub programs: Vec<SparseCoreProgramProto>,
    /// Total HBM bytes reserved for embedding tables (field `5`).
    #[prost(uint64, tag = "5")]
    pub table_hbm_total_bytes: u64,
}

impl_proto_schema_message!(
    BarnaCoreExecutableProto,
    "xla.barnacore.BarnaCoreExecutableProto"
);

impl TargetArgumentsProto {
    /// Constructs a canonical [`TargetArgumentsProto`] from a [`TpuChipConfigProto`] and [`TpuChipPartsProto`].
    ///
    /// # Arguments
    ///
    /// * `config` - Decoded chip configuration descriptor (`*_chip_configs_*.binarypb`).
    /// * `parts` - Decoded chip physical parts descriptor (`*_chip_parts.binarypb`).
    #[must_use]
    pub fn from_chip_config_and_parts(
        config: &TpuChipConfigProto,
        parts: &TpuChipPartsProto,
    ) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        Self {
            tpu_version: config.tpu_version,
            chip_variant: config.chip_name.clone(),
            clock_rate_mhz: parts.clock_rate_mhz,
            hbm_capacity_bytes: config.hbm_capacity_bytes as i64,
            vmem_capacity_bytes: config.vmem_capacity_bytes as i64,
            smem_capacity_bytes: config.smem_capacity_bytes as i64,
            cmem_capacity_bytes: parts.cmem_capacity_bytes as i64,
            tensor_cores_per_chip: config.tensor_cores_per_chip as i32,
            sparse_cores_per_chip: config.sparse_cores_per_chip as i32,
            mxu_dimension: parts.mxu_dimension as i32,
            ici_bandwidth_bps: parts.ici_bandwidth_bps,
        }
    }

    /// Constructs a canonical [`TargetArgumentsProto`] for a [`crate::types::TpuGeneration`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU hardware generation.
    #[must_use]
    pub fn from_generation(generation: crate::types::TpuGeneration) -> Self {
        let config = TpuChipConfigProto::from_generation(generation);
        let parts = TpuChipPartsProto::from_generation(generation);
        Self::from_chip_config_and_parts(&config, &parts)
    }
}

impl TpuChipConfigProto {
    /// Returns the embedded `filewrapper_toc` chip configuration descriptor for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Cloud TPU hardware generation.
    #[must_use]
    pub fn from_generation(generation: crate::types::TpuGeneration) -> Self {
        use crate::types::TpuGeneration;
        let (chip_name, megacore_enabled) = match generation {
            TpuGeneration::V2 => ("jellyfish_default", false),
            TpuGeneration::V3 => ("dragonfish_default", false),
            TpuGeneration::V4 => ("pufferfish_megacore_dense", true),
            TpuGeneration::V4Lite => ("pufferfish_lite", false),
            TpuGeneration::V5e => ("viperfish_lite", false),
            TpuGeneration::V5p => ("viperfish_megacore", true),
            TpuGeneration::V6e => ("ghostlite", false),
            TpuGeneration::V7x => ("6acc60406_torch", true),
        };
        Self {
            chip_name: String::from(chip_name),
            tpu_version: generation as i32,
            tensor_cores_per_chip: generation.tensor_cores_per_chip(),
            sparse_cores_per_chip: generation.sparse_cores_per_chip(),
            hbm_capacity_bytes: generation.hbm_capacity_bytes_per_chip().0 as u64,
            vmem_capacity_bytes: generation.vmem_capacity_bytes_per_core().0 as u64,
            smem_capacity_bytes: generation.smem_capacity_bytes_per_core().0 as u64,
            megacore_enabled,
        }
    }
}

impl TpuChipPartsProto {
    /// Returns the embedded `filewrapper_toc` physical die parts descriptor for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Cloud TPU hardware generation.
    #[must_use]
    pub fn from_generation(generation: crate::types::TpuGeneration) -> Self {
        use crate::types::TpuGeneration;
        let (part_name, clock_rate_mhz, ici_bandwidth_bps) = match generation {
            TpuGeneration::V2 => ("jellyfish", 700, 100_000_000_000),
            TpuGeneration::V3 => ("dragonfish", 940, 200_000_000_000),
            TpuGeneration::V4 => ("pufferfish", 1050, 400_000_000_000),
            TpuGeneration::V4Lite => ("pufferfish_lite", 1050, 200_000_000_000),
            TpuGeneration::V5e => ("viperfish_lite", 1050, 400_000_000_000),
            TpuGeneration::V5p => ("viperfish", 1250, 800_000_000_000),
            TpuGeneration::V6e => ("ghostlite", 1500, 800_000_000_000),
            TpuGeneration::V7x => ("6acc60406", 1750, 1_600_000_000_000),
        };
        Self {
            part_name: String::from(part_name),
            tpu_version: generation as i32,
            mxu_dimension: generation.mxu_dimension(),
            mxu_count_per_core: generation.mxu_count_per_core(),
            ici_link_count: generation.ici_link_count(),
            clock_rate_mhz,
            ici_bandwidth_bps,
            cmem_capacity_bytes: generation.cmem_capacity_bytes().0 as u64,
        }
    }
}

/// Wire-compatible `asic_sw.deepsea.IciRouteHopProto` (`docs/reverse_engineering/protos/ici_routing_table.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IciRouteHopProto {
    /// Source chip identifier (field `1`).
    #[prost(uint32, tag = "1")]
    pub src_chip_id: u32,
    /// Destination chip identifier (field `2`).
    #[prost(uint32, tag = "2")]
    pub dst_chip_id: u32,
    /// Outbound optical ICI link port index (field `3`).
    #[prost(uint32, tag = "3")]
    pub egress_link_port: u32,
    /// Virtual channel index for deadlock avoidance (field `4`).
    #[prost(uint32, tag = "4")]
    pub virtual_channel: u32,
    /// Signed hop delta along torus axis (field `5`).
    #[prost(int32, tag = "5")]
    pub axis_delta: i32,
    /// Whether this hop traverses a wrap-around torus optical link (field `6`).
    #[prost(bool, tag = "6")]
    pub is_wrap_around: bool,
}

impl_proto_schema_message!(IciRouteHopProto, "asic_sw.deepsea.IciRouteHopProto");

/// Wire-compatible `asic_sw.deepsea.IciRoutingTableProto` (`docs/reverse_engineering/protos/ici_routing_table.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IciRoutingTableProto {
    /// Topology configuration identifier (field `1`).
    #[prost(string, tag = "1")]
    pub topology_name: String,
    /// Torus dimensionality (`2` or `3`, field `2`).
    #[prost(uint32, tag = "2")]
    pub torus_dimension: u32,
    /// Whether twisted-torus wrap-around routing is enabled (field `3`).
    #[prost(bool, tag = "3")]
    pub is_twisted_torus: bool,
    /// Minimal/DOR route hop entries (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub route_hops: Vec<IciRouteHopProto>,
    /// Hardware ICI semaphore pool size per core (field `5`).
    #[prost(uint32, tag = "5")]
    pub semaphore_pool_size: u32,
}

impl_proto_schema_message!(IciRoutingTableProto, "asic_sw.deepsea.IciRoutingTableProto");

/// Wire-compatible `asic_sw.deepsea.VliwBundleEncodingSpecProto` (`docs/reverse_engineering/protos/vliw_bundle_encoding.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct VliwBundleEncodingSpecProto {
    /// Hardware ISA family name (`jfc`, `pfc`, `vfc`, `vfk`, field `1`).
    #[prost(string, tag = "1")]
    pub isa_family: String,
    /// Encoded bundle byte width (`32`, `40`, `52`, or `64`, field `2`).
    #[prost(uint32, tag = "2")]
    pub bundle_byte_width: u32,
    /// Scalar issue slot count (field `3`).
    #[prost(uint32, tag = "3")]
    pub scalar_slots: u32,
    /// Vector ALU issue slot count (field `4`).
    #[prost(uint32, tag = "4")]
    pub vector_slots: u32,
    /// Systolic MXU issue slot count (field `5`).
    #[prost(uint32, tag = "5")]
    pub matrix_slots: u32,
    /// Cross-lane / EUP issue slot count (field `6`).
    #[prost(uint32, tag = "6")]
    pub xlu_slots: u32,
    /// Asynchronous DMA issue slot count (field `7`).
    #[prost(uint32, tag = "7")]
    pub dma_slots: u32,
    /// Whether dual-word immediate extension prefixes are supported (field `8`).
    #[prost(bool, tag = "8")]
    pub supports_dual_word_imm_ext: bool,
}

impl_proto_schema_message!(
    VliwBundleEncodingSpecProto,
    "asic_sw.deepsea.VliwBundleEncodingSpecProto"
);

/// Wire-compatible `asic_sw.deepsea.DmaDescriptorRingConfigProto` (`docs/reverse_engineering/protos/dma_descriptor_ring.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DmaDescriptorRingConfigProto {
    /// Hardware DMA ring identifier (field `1`).
    #[prost(uint32, tag = "1")]
    pub ring_id: u32,
    /// Number of 32-byte descriptors in the circular ring (field `2`).
    #[prost(uint32, tag = "2")]
    pub ring_capacity_entries: u32,
    /// BAR0 CSR byte offset for consumer head pointer (field `3`).
    #[prost(uint64, tag = "3")]
    pub head_csr_offset: u64,
    /// BAR0 CSR byte offset for producer tail doorbell (field `4`).
    #[prost(uint64, tag = "4")]
    pub tail_csr_offset: u64,
    /// Base `SyncFlag` index signaled on descriptor completion (field `5`).
    #[prost(uint32, tag = "5")]
    pub sync_flag_base_id: u32,
    /// Whether multi-page IOMMU scatter-gather chaining is enabled (field `6`).
    #[prost(bool, tag = "6")]
    pub iommu_scatter_gather_enabled: bool,
}

impl_proto_schema_message!(
    DmaDescriptorRingConfigProto,
    "asic_sw.deepsea.DmaDescriptorRingConfigProto"
);

/// Wire-compatible `asic_sw.driver.GasketPageTableConfigProto` (`docs/reverse_engineering/protos/gasket_ioctl.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GasketPageTableConfigProto {
    /// Hardware page table index (field `1`).
    #[prost(uint32, tag = "1")]
    pub page_table_index: u32,
    /// Number of simple linear page table entries (field `2`).
    #[prost(uint64, tag = "2")]
    pub simple_entries: u64,
    /// Number of extended scatter-gather page table entries (field `3`).
    #[prost(uint64, tag = "3")]
    pub extended_entries: u64,
    /// Base IOVA aperture address in bytes (field `4`).
    #[prost(uint64, tag = "4")]
    pub dma_base_iova: u64,
}

impl_proto_schema_message!(
    GasketPageTableConfigProto,
    "asic_sw.driver.GasketPageTableConfigProto"
);

/// Wire-compatible `asic_sw.driver.GasketIoctlConfigProto` (`docs/reverse_engineering/protos/gasket_ioctl.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GasketIoctlConfigProto {
    /// Character device path (e.g. `/dev/accel0`, field `1`).
    #[prost(string, tag = "1")]
    pub device_node_path: String,
    /// Linux ioctl type magic (`0xDC`, field `2`).
    #[prost(uint32, tag = "2")]
    pub ioctl_magic: u32,
    /// Per-core page table configurations (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub page_tables: Vec<GasketPageTableConfigProto>,
    /// Number of MSI-X interrupt vectors bound to `eventfd`s (field `4`).
    #[prost(uint32, tag = "4")]
    pub num_msix_interrupts: u32,
}

impl_proto_schema_message!(
    GasketIoctlConfigProto,
    "asic_sw.driver.GasketIoctlConfigProto"
);

/// Wire-compatible `asic_sw.driver.VfioIommuMapConfigProto` (`docs/reverse_engineering/protos/vfio_iommu_map.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct VfioIommuMapConfigProto {
    /// VFIO group device path (e.g. `/dev/vfio/0`, field `1`).
    #[prost(string, tag = "1")]
    pub vfio_group_path: String,
    /// VFIO IOMMU driver mode (`VFIO_TYPE1v2_IOMMU = 3`, field `2`).
    #[prost(uint32, tag = "2")]
    pub iommu_type: u32,
    /// Base IOVA mapping address (field `3`).
    #[prost(uint64, tag = "3")]
    pub iova_aperture_base: u64,
    /// Total IOVA aperture size in bytes (field `4`).
    #[prost(uint64, tag = "4")]
    pub iova_aperture_size: u64,
    /// Hugepage mapping granularity (`2_097_152` for 2 MiB, field `5`).
    #[prost(uint64, tag = "5")]
    pub hugepage_granularity_bytes: u64,
}

impl_proto_schema_message!(
    VfioIommuMapConfigProto,
    "asic_sw.driver.VfioIommuMapConfigProto"
);

/// Wire-compatible `asic_sw.driver.PcieBarRegionProto` (`docs/reverse_engineering/protos/pcie_bar_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PcieBarRegionProto {
    /// `PCIe` BAR index (`0` for CSRs, `2` for HBM aperture, field `1`).
    #[prost(uint32, tag = "1")]
    pub bar_index: u32,
    /// Region file offset for `mmap` (field `2`).
    #[prost(uint64, tag = "2")]
    pub region_offset: u64,
    /// BAR aperture size in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub size_bytes: u64,
    /// Whether the BAR supports prefetchable write-combining (field `4`).
    #[prost(bool, tag = "4")]
    pub is_prefetchable: bool,
}

impl_proto_schema_message!(PcieBarRegionProto, "asic_sw.driver.PcieBarRegionProto");

/// Wire-compatible `asic_sw.driver.PcieBarLayoutProto` (`docs/reverse_engineering/protos/pcie_bar_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PcieBarLayoutProto {
    /// `PCIe` vendor ID (`0x1AE0` Google, field `1`).
    #[prost(uint32, tag = "1")]
    pub vendor_id: u32,
    /// `PCIe` device ID (field `2`).
    #[prost(uint32, tag = "2")]
    pub device_id: u32,
    /// Discovered `PCIe` BAR regions (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub bar_regions: Vec<PcieBarRegionProto>,
}

impl_proto_schema_message!(PcieBarLayoutProto, "asic_sw.driver.PcieBarLayoutProto");

/// Wire-compatible `platforms_deepsea.jellyfish.llo.LloScheduledInstructionProto` (`docs/reverse_engineering/protos/llo_bundle_schedule.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LloScheduledInstructionProto {
    /// VLIW bundle program counter index (field `1`).
    #[prost(uint32, tag = "1")]
    pub bundle_pc: u32,
    /// Target functional unit (`0=S0`, `1=V0`, `2=M0`, `3=X0`, `4=D0`, field `2`).
    #[prost(uint32, tag = "2")]
    pub functional_unit: u32,
    /// Hardware `TpuOpCode` byte (field `3`).
    #[prost(uint32, tag = "3")]
    pub opcode: u32,
    /// Guarding predicate register `@p0..@p15` (field `4`).
    #[prost(uint32, tag = "4")]
    pub predicate_reg: u32,
    /// Post-bundle scoreboard stall cycles (field `5`).
    #[prost(uint32, tag = "5")]
    pub stall_cycles: u32,
    /// Originating XLA HLO instruction name (field `6`).
    #[prost(string, tag = "6")]
    pub hlo_instruction_name: String,
}

impl_proto_schema_message!(
    LloScheduledInstructionProto,
    "platforms_deepsea.jellyfish.llo.LloScheduledInstructionProto"
);

/// Wire-compatible `platforms_deepsea.jellyfish.llo.LloBundleScheduleProto` (`docs/reverse_engineering/protos/llo_bundle_schedule.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LloBundleScheduleProto {
    /// Schedule identifier (field `1`).
    #[prost(string, tag = "1")]
    pub schedule_name: String,
    /// Total estimated execution cycles (field `2`).
    #[prost(uint32, tag = "2")]
    pub total_cycles: u32,
    /// Modulo software pipeline initiation interval MII (field `3`).
    #[prost(uint32, tag = "3")]
    pub initiation_interval_mii: u32,
    /// Scheduled VLIW instruction slots (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub scheduled_instructions: Vec<LloScheduledInstructionProto>,
}

impl_proto_schema_message!(
    LloBundleScheduleProto,
    "platforms_deepsea.jellyfish.llo.LloBundleScheduleProto"
);

/// Wire-compatible `platforms_deepsea.jellyfish.llo.LloSpillSlotProto` (`docs/reverse_engineering/protos/llo_register_allocator.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LloSpillSlotProto {
    /// Spilled virtual register index (field `1`).
    #[prost(uint32, tag = "1")]
    pub virtual_reg_id: u32,
    /// Scratchpad VMEM spill byte offset (field `2`).
    #[prost(uint64, tag = "2")]
    pub vmem_spill_offset_bytes: u64,
    /// Bundle PC where the `D0` spill store was spliced (field `3`).
    #[prost(uint32, tag = "3")]
    pub spill_bundle_pc: u32,
    /// Bundle PC where the `D0` reload load was spliced (field `4`).
    #[prost(uint32, tag = "4")]
    pub reload_bundle_pc: u32,
}

impl_proto_schema_message!(
    LloSpillSlotProto,
    "platforms_deepsea.jellyfish.llo.LloSpillSlotProto"
);

/// Wire-compatible `platforms_deepsea.jellyfish.llo.LloLiveIntervalProto` (`docs/reverse_engineering/protos/llo_register_allocator.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LloLiveIntervalProto {
    /// Virtual SSA slot index (field `1`).
    #[prost(uint32, tag = "1")]
    pub virtual_reg_id: u32,
    /// Assigned physical `$v0..$v63` register index (field `2`).
    #[prost(uint32, tag = "2")]
    pub physical_reg_id: u32,
    /// First bundle PC where the value is defined (field `3`).
    #[prost(uint32, tag = "3")]
    pub start_pc: u32,
    /// Last bundle PC where the value is killed (field `4`).
    #[prost(uint32, tag = "4")]
    pub end_pc: u32,
    /// Whether this live range required VMEM spilling (field `5`).
    #[prost(bool, tag = "5")]
    pub is_spilled: bool,
}

impl_proto_schema_message!(
    LloLiveIntervalProto,
    "platforms_deepsea.jellyfish.llo.LloLiveIntervalProto"
);

/// Wire-compatible `platforms_deepsea.jellyfish.llo.LloRegisterAllocationProto` (`docs/reverse_engineering/protos/llo_register_allocator.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LloRegisterAllocationProto {
    /// Peak physical vector registers `$vreg` allocated (field `1`).
    #[prost(uint32, tag = "1")]
    pub max_vregs_used: u32,
    /// Peak physical scalar registers `$sreg` allocated (field `2`).
    #[prost(uint32, tag = "2")]
    pub max_sregs_used: u32,
    /// Peak physical predicate registers `$preg` allocated (field `3`).
    #[prost(uint32, tag = "3")]
    pub max_pregs_used: u32,
    /// Per-slot live interval assignments (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub live_intervals: Vec<LloLiveIntervalProto>,
    /// Recorded VMEM spill/reload slots (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub spill_slots: Vec<LloSpillSlotProto>,
}

impl_proto_schema_message!(
    LloRegisterAllocationProto,
    "platforms_deepsea.jellyfish.llo.LloRegisterAllocationProto"
);

/// Wire-compatible `platforms_deepsea.jellyfish.llo.LloModuloStageProto` (`docs/reverse_engineering/protos/llo_software_pipeliner.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LloModuloStageProto {
    /// Software pipeline stage index `0..K-1` (field `1`).
    #[prost(uint32, tag = "1")]
    pub stage_index: u32,
    /// Rotating stage predicate register `@p0..@p7` (field `2`).
    #[prost(uint32, tag = "2")]
    pub rotating_predicate_reg: u32,
    /// Cycle offset within the modulo kernel (field `3`).
    #[prost(uint32, tag = "3")]
    pub start_cycle: u32,
    /// Completion cycle offset within the modulo kernel (field `4`).
    #[prost(uint32, tag = "4")]
    pub end_cycle: u32,
}

impl_proto_schema_message!(
    LloModuloStageProto,
    "platforms_deepsea.jellyfish.llo.LloModuloStageProto"
);

/// Wire-compatible `platforms_deepsea.jellyfish.llo.LloSoftwarePipelineScheduleProto` (`docs/reverse_engineering/protos/llo_software_pipeliner.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LloSoftwarePipelineScheduleProto {
    /// Loop computation or instruction name (field `1`).
    #[prost(string, tag = "1")]
    pub loop_name: String,
    /// Resource-constrained minimum initiation interval `ResMII` (field `2`).
    #[prost(uint32, tag = "2")]
    pub resource_mii: u32,
    /// Recurrence-constrained minimum initiation interval `RecMII` (field `3`).
    #[prost(uint32, tag = "3")]
    pub recurrence_mii: u32,
    /// Achieved initiation interval `II = max(ResMII, RecMII)` (field `4`).
    #[prost(uint32, tag = "4")]
    pub achieved_ii: u32,
    /// Modulo pipeline stages (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub stages: Vec<LloModuloStageProto>,
}

impl_proto_schema_message!(
    LloSoftwarePipelineScheduleProto,
    "platforms_deepsea.jellyfish.llo.LloSoftwarePipelineScheduleProto"
);
