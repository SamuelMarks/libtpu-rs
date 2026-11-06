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

impl_proto_schema_message!(
    ProgramRelocationProto,
    "platforms_deepsea.jellyfish.ProgramRelocation"
);

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

/// Wire-compatible `xla.ffi.FlatAttributeProto` (`xla/ffi/attribute_map.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct FfiFlatAttributeProto {
    /// Raw string or byte attribute value (`oneof value.str`, field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub str: Vec<u8>,
}

impl_proto_schema_message!(FfiFlatAttributeProto, "xla.ffi.FlatAttributeProto");

/// Wire-compatible `xla.ffi.FlatAttributesMapProto` (`xla/ffi/attribute_map.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct FfiFlatAttributesMapProto {
    /// Flat attribute dictionary keyed by attribute name (field `1`).
    #[prost(btree_map = "string, message", tag = "1")]
    pub attrs: std::collections::BTreeMap<String, FfiFlatAttributeProto>,
}

impl_proto_schema_message!(FfiFlatAttributesMapProto, "xla.ffi.FlatAttributesMapProto");

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
