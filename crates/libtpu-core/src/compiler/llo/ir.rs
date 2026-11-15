//! Strongly-typed Low-Level Optimizer (`LLO`) intermediate representation (`LloIr`, `LloMicroOp`, `LloBundle`).
//!
//! Represents scalar (`S0`/`S1`), vector (`V0`/`V1`/`EUP`), matrix (`MXU0..3`), cross-lane (`XLU0..1`),
//! asynchronous `DMA`, immediate extension (`IMM`), and control/predicate operations, with lossless
//! bidirectional conversion to and from `xla.jellyfish.LloModuleProto`.

use super::super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use super::super::vliw::{
    AsyncDmaSlot, ExtendedUnarySlot, MatrixMxuSlot, ScalarAluSlot, VectorAluSlot,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    BundlePackerInfoProto, LloInstructionProto, LloModuleProto, LloOpcodeProto, LloRegionProto,
    LloRegisterTypeProto, ShapeProto, StaticPerBundleUtilizationProto, llo_instruction_proto,
};
use crate::types::{
    ElementType, MatmulAccIndex, MregIndex, PredRegIndex, PregIndex, SregIndex, SyncFlagId,
    SyncFlagIndex, TpuGeneration, VregIndex,
};

/// Hardware functional unit slot targeted by an [`LloMicroOp`] (`LloFunctionalUnit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LloFunctionalUnit {
    /// Primary scalar ALU & branch sequencer slot (`S0`).
    Scalar0,
    /// Secondary scalar ALU co-issue slot (`S1`).
    Scalar1,
    /// Primary 2D `[8, 128]` vector ALU slot (`V0`).
    Vector0,
    /// Secondary 2D `[8, 128]` vector ALU co-issue slot (`V1`).
    Vector1,
    /// Extended Unary Pipeline (`EUP`) transcendental LUT/polynomial unit.
    VectorEup,
    /// Systolic Matrix Multiply Unit 0 (`MXU0`).
    MatrixMxu0,
    /// Systolic Matrix Multiply Unit 1 (`MXU1`).
    MatrixMxu1,
    /// Systolic Matrix Multiply Unit 2 (`MXU2`).
    MatrixMxu2,
    /// Systolic Matrix Multiply Unit 3 (`MXU3`).
    MatrixMxu3,
    /// Cross-lane / sublane permutation unit 0 (`XLU0`).
    CrossLaneXlu0,
    /// Cross-lane / sublane permutation unit 1 (`XLU1`).
    CrossLaneXlu1,
    /// Asynchronous HBM/VMEM/CMEM/ICI DMA engine (`DMA`).
    AsyncDma,
    /// Predicate & `SyncFlag` control unit (`CTRL`).
    ControlPred,
    /// 32-bit immediate constant extension slot (`IMM`).
    ImmediateExt,
}

impl LloFunctionalUnit {
    /// Returns the numeric functional unit code (`0..=13`) used in [`crate::proto::LloScheduledInstructionProto`].
    #[must_use]
    pub const fn unit_code(self) -> u32 {
        match self {
            Self::Scalar0 => 0,
            Self::Scalar1 => 1,
            Self::MatrixMxu0 => 2,
            Self::CrossLaneXlu0 => 3,
            Self::Vector0 => 4,
            Self::AsyncDma => 5,
            Self::Vector1 => 6,
            Self::VectorEup => 7,
            Self::MatrixMxu1 => 8,
            Self::MatrixMxu2 => 9,
            Self::MatrixMxu3 => 10,
            Self::CrossLaneXlu1 => 11,
            Self::ControlPred => 12,
            Self::ImmediateExt => 13,
        }
    }

    /// Reconstructs an [`LloFunctionalUnit`] from a raw unit code (`0..=13`).
    ///
    /// # Arguments
    ///
    /// * `code` - Numeric functional unit code.
    #[must_use]
    pub const fn from_unit_code(code: u32) -> Self {
        match code {
            0 => Self::Scalar0,
            1 => Self::Scalar1,
            2 => Self::MatrixMxu0,
            3 => Self::CrossLaneXlu0,
            4 => Self::Vector0,
            5 => Self::AsyncDma,
            6 => Self::Vector1,
            7 => Self::VectorEup,
            8 => Self::MatrixMxu1,
            9 => Self::MatrixMxu2,
            10 => Self::MatrixMxu3,
            11 => Self::CrossLaneXlu1,
            12 => Self::ControlPred,
            _ => Self::ImmediateExt,
        }
    }

    /// Returns the canonical hardware slot mnemonic (`"S0"`, `"V0"`, `"MXU0"`, etc.).
    #[must_use]
    pub const fn slot_name(self) -> &'static str {
        match self {
            Self::Scalar0 => "S0",
            Self::Scalar1 => "S1",
            Self::Vector0 => "V0",
            Self::Vector1 => "V1",
            Self::VectorEup => "EUP",
            Self::MatrixMxu0 => "MXU0",
            Self::MatrixMxu1 => "MXU1",
            Self::MatrixMxu2 => "MXU2",
            Self::MatrixMxu3 => "MXU3",
            Self::CrossLaneXlu0 => "XLU0",
            Self::CrossLaneXlu1 => "XLU1",
            Self::AsyncDma => "DMA",
            Self::ControlPred => "CTRL",
            Self::ImmediateExt => "IMM",
        }
    }

    /// Returns the architectural pipeline latency in cycles for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn default_latency_cycles(self, generation: TpuGeneration) -> u32 {
        match self {
            Self::Scalar0 | Self::Scalar1 | Self::ControlPred | Self::ImmediateExt => 1,
            Self::Vector0 | Self::Vector1 => 2,
            Self::VectorEup | Self::CrossLaneXlu0 | Self::CrossLaneXlu1 => 4,
            Self::MatrixMxu0 | Self::MatrixMxu1 | Self::MatrixMxu2 | Self::MatrixMxu3 => {
                if matches!(generation, TpuGeneration::V6e | TpuGeneration::V7x) {
                    10
                } else {
                    8
                }
            }
            Self::AsyncDma => 12,
        }
    }
}

/// Hardware register bank classification (`LloRegClass`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LloRegClass {
    /// 32-bit scalar general-purpose register bank (`SREG`, `$s0..$s63`).
    Sreg,
    /// 2D `[8, 128]` vector register bank (`VREG`, `$v0..$v63`).
    Vreg,
    /// Systolic matrix accumulator register bank (`MREG`, `$gmr0..$gmr7`).
    Mreg,
    /// 1-bit scalar/bundle predicate register bank (`PREG`, `@p0..@p15`).
    Preg,
    /// Hardware DMA/ICI completion counter bank (`SyncFlag`, `$sf0..$sf255`).
    SyncFlag,
}

impl LloRegClass {
    /// Returns the numeric register class discriminant (`0..=4`) used in [`crate::proto::LloLiveIntervalProto`].
    #[must_use]
    pub const fn class_code(self) -> u32 {
        match self {
            Self::Sreg => 0,
            Self::Vreg => 1,
            Self::Mreg => 2,
            Self::Preg => 3,
            Self::SyncFlag => 4,
        }
    }

    /// Returns the canonical register bank name (`"SREG"`, `"VREG"`, `"MREG"`, `"PREG"`, `"SFLAG"`).
    #[must_use]
    pub const fn bank_name(self) -> &'static str {
        match self {
            Self::Sreg => "SREG",
            Self::Vreg => "VREG",
            Self::Mreg => "MREG",
            Self::Preg => "PREG",
            Self::SyncFlag => "SFLAG",
        }
    }

    /// Converts this [`LloRegClass`] into the generated `xla.jellyfish.RegisterTypeProto` enum.
    #[must_use]
    pub const fn to_register_type_proto(self) -> LloRegisterTypeProto {
        match self {
            Self::Sreg => LloRegisterTypeProto::RegisterTypeSreg,
            Self::Vreg => LloRegisterTypeProto::RegisterTypeVreg,
            Self::Mreg => LloRegisterTypeProto::RegisterTypeVmreg,
            Self::Preg => LloRegisterTypeProto::RegisterTypePreg,
            Self::SyncFlag => LloRegisterTypeProto::RegisterTypeDreg,
        }
    }

    /// Converts a generated `xla.jellyfish.RegisterTypeProto` discriminant into an [`LloRegClass`].
    ///
    /// # Arguments
    ///
    /// * `proto` - Generated [`LloRegisterTypeProto`] discriminant.
    #[must_use]
    pub const fn from_register_type_proto(proto: LloRegisterTypeProto) -> Self {
        match proto {
            LloRegisterTypeProto::RegisterTypeSreg => Self::Sreg,
            LloRegisterTypeProto::RegisterTypeVmreg => Self::Mreg,
            LloRegisterTypeProto::RegisterTypePreg | LloRegisterTypeProto::RegisterTypeRpreg => {
                Self::Preg
            }
            LloRegisterTypeProto::RegisterTypeDreg | LloRegisterTypeProto::RegisterTypeCbreg => {
                Self::SyncFlag
            }
            _ => Self::Vreg,
        }
    }

    /// Returns the default physical register count for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn hardware_capacity(self, generation: TpuGeneration) -> u32 {
        let wide_gen = matches!(
            generation,
            TpuGeneration::V5p | TpuGeneration::V6e | TpuGeneration::V7x
        );
        match self {
            Self::Sreg | Self::Vreg => {
                if wide_gen {
                    64
                } else {
                    32
                }
            }
            Self::Mreg => {
                if wide_gen {
                    8
                } else {
                    4
                }
            }
            Self::Preg => 16,
            Self::SyncFlag => 256,
        }
    }

    /// Returns `(memory_space, byte_size)` for spilling a single register of this class (`1 = VMEM`, `2 = SMEM`).
    #[must_use]
    pub const fn spill_layout(self) -> (u32, u32) {
        match self {
            Self::Vreg => (1, 4096),
            Self::Mreg => (1, 65_536),
            Self::Sreg | Self::Preg | Self::SyncFlag => (2, 8),
        }
    }
}

/// Strongly-typed virtual, physical, or immediate operand in [`LloMicroOp`] (`LloRegOperand`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LloRegOperand {
    /// Unallocated virtual SSA register `(virtual_id, reg_class)`.
    Virtual {
        /// Virtual register identifier.
        id: u32,
        /// Target hardware register bank.
        class: LloRegClass,
    },
    /// Physical scalar register (`$sN`).
    PhysicalSreg(SregIndex),
    /// Physical `[8, 128]` vector register (`$vN`).
    PhysicalVreg(VregIndex),
    /// Physical matrix accumulator register (`$gmrN`).
    PhysicalMreg(MregIndex),
    /// Physical 1-bit predicate register (`@pN`).
    PhysicalPred(PredRegIndex),
    /// Physical `SyncFlag` completion counter (`$sfN`).
    PhysicalSyncFlag(SyncFlagIndex),
    /// Inline 64-bit integer/bit-cast immediate constant.
    Immediate(u64),
}

impl LloRegOperand {
    /// Returns the raw register index as `u32` (or truncated immediate value).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn raw_index(self) -> u32 {
        match self {
            Self::Virtual { id, .. } => id,
            Self::PhysicalSreg(r) => r.0 as u32,
            Self::PhysicalVreg(r) => r.0 as u32,
            Self::PhysicalMreg(r) => r.0 as u32,
            Self::PhysicalPred(r) => r.0 as u32,
            Self::PhysicalSyncFlag(r) => r.0,
            Self::Immediate(v) => v as u32,
        }
    }

    /// Constructs a physical [`LloRegOperand`] for `class` and `phys_index`.
    ///
    /// # Arguments
    ///
    /// * `class` - Target [`LloRegClass`].
    /// * `phys_index` - Physical register index within `class`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn physical_for_class(class: LloRegClass, phys_index: u32) -> Self {
        match class {
            LloRegClass::Sreg => Self::PhysicalSreg(SregIndex(phys_index as u8)),
            LloRegClass::Vreg => Self::PhysicalVreg(VregIndex(phys_index as u8)),
            LloRegClass::Mreg => Self::PhysicalMreg(MatmulAccIndex(phys_index as u8)),
            LloRegClass::Preg => Self::PhysicalPred(PregIndex(phys_index as u8)),
            LloRegClass::SyncFlag => Self::PhysicalSyncFlag(SyncFlagId(phys_index)),
        }
    }
}

/// Single LLO micro-operation prior to or after physical register allocation (`LloMicroOp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LloMicroOp {
    /// Dense instruction identifier within the containing [`LloIr`].
    pub id: u32,
    /// Canonical `xla.jellyfish.LloOpcodeProto` opcode.
    pub llo_opcode: LloOpcodeProto,
    /// Lowered [`TpuOpCode`] semantics for functional unit execution.
    pub tpu_opcode: TpuOpCode,
    /// Target [`LloFunctionalUnit`] slot.
    pub unit: LloFunctionalUnit,
    /// Active [`ElementType`] of the operation.
    pub element_type: ElementType,
    /// Optional guarding predicate register (`@p0..@p15`).
    pub predicate: Option<PredRegIndex>,
    /// Whether `predicate` is inverted (`!@pN`).
    pub negate_predicate: bool,
    /// Destination register operands defined by this micro-op.
    pub defs: Vec<LloRegOperand>,
    /// Source register or immediate operands read by this micro-op.
    pub uses: Vec<LloRegOperand>,
    /// Auxiliary 64-bit immediate constant or byte offset.
    pub imm_value: u64,
    /// Pipeline latency in cycles.
    pub latency_cycles: u32,
    /// Memory space (`0 = HBM`, `1 = VMEM`, `2 = SMEM`, `3 = CMEM`).
    pub memory_space: u8,
    /// Originating HLO or Mosaic op name for XDB source mapping.
    pub hlo_source_name: String,
}

impl LloMicroOp {
    /// Serializes this [`LloMicroOp`] into a generated `xla.jellyfish.LloInstructionProto`.
    #[must_use]
    pub fn to_instruction_proto(&self) -> LloInstructionProto {
        let mut operands = Vec::with_capacity(self.uses.len());
        for op in &self.uses {
            let val = match *op {
                LloRegOperand::Immediate(c) => {
                    llo_instruction_proto::llo_operand::Value::ConstantHandle(
                        i32::try_from(c).unwrap_or(i32::MAX),
                    )
                }
                other => llo_instruction_proto::llo_operand::Value::InstructionOrdinal(
                    i32::try_from(other.raw_index()).unwrap_or(0),
                ),
            };
            operands.push(llo_instruction_proto::LloOperand { value: Some(val) });
        }
        let (reg_id, reg_type) = match self.defs.first().copied() {
            Some(LloRegOperand::Virtual { id, class }) => (
                i32::try_from(id).unwrap_or(0),
                class.to_register_type_proto() as i32,
            ),
            Some(LloRegOperand::PhysicalSreg(s)) => (
                i32::from(s.0),
                LloRegisterTypeProto::RegisterTypeSreg as i32,
            ),
            Some(LloRegOperand::PhysicalVreg(v)) => (
                i32::from(v.0),
                LloRegisterTypeProto::RegisterTypeVreg as i32,
            ),
            Some(LloRegOperand::PhysicalMreg(m)) => (
                i32::from(m.0),
                LloRegisterTypeProto::RegisterTypeVmreg as i32,
            ),
            Some(LloRegOperand::PhysicalPred(p)) => (
                i32::from(p.0),
                LloRegisterTypeProto::RegisterTypePreg as i32,
            ),
            Some(LloRegOperand::PhysicalSyncFlag(sf)) => (
                i32::try_from(sf.0).unwrap_or(0),
                LloRegisterTypeProto::RegisterTypeDreg as i32,
            ),
            Some(LloRegOperand::Immediate(_)) | None => {
                (-1, LloRegisterTypeProto::RegisterTypeNone as i32)
            }
        };
        let polarity = match (self.predicate, self.negate_predicate) {
            (None, _) => llo_instruction_proto::PredicationPolarityProto::PolarityNone as i32,
            (Some(_), false) => {
                llo_instruction_proto::PredicationPolarityProto::PolarityPositive as i32
            }
            (Some(_), true) => {
                llo_instruction_proto::PredicationPolarityProto::PolarityNegative as i32
            }
        };
        let pred_count = i32::from(self.predicate.is_some());
        LloInstructionProto {
            ordinal: i32::try_from(self.id).unwrap_or(0),
            opcode: self.llo_opcode as i32,
            operands,
            associated_shape: Some(ShapeProto {
                element_type: self.element_type.as_raw(),
                ..ShapeProto::default()
            }),
            register_id: reg_id,
            register_type: reg_type,
            scalar_constant_value: self.imm_value,
            unit_id: i32::try_from(self.unit.unit_code()).unwrap_or(0),
            source_bus: i32::from(self.tpu_opcode as u8),
            predication_operand_count: pred_count,
            predication_polarity: polarity,
            iar: i64::from(self.id),
            bundle_packer_info: Some(BundlePackerInfoProto {
                final_bundle_index: self.id,
                ..BundlePackerInfoProto::default()
            }),
            ..LloInstructionProto::default()
        }
    }

    /// Reconstructs an [`LloMicroOp`] from a generated `xla.jellyfish.LloInstructionProto`.
    ///
    /// # Arguments
    ///
    /// * `proto` - Source [`LloInstructionProto`].
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_instruction_proto(proto: &LloInstructionProto, generation: TpuGeneration) -> Self {
        let llo_opcode =
            LloOpcodeProto::try_from(proto.opcode).unwrap_or(LloOpcodeProto::OpcodeNone);
        let tpu_opcode = u8::try_from(proto.source_bus)
            .ok()
            .and_then(|b| TpuOpCode::from_byte(b).ok())
            .unwrap_or(TpuOpCode::AddF32);
        let unit = LloFunctionalUnit::from_unit_code(proto.unit_id.max(0) as u32);
        let element_type = proto
            .associated_shape
            .as_ref()
            .and_then(|s| ElementType::from_raw(s.element_type).ok())
            .unwrap_or(ElementType::F32);
        let defs = if proto.register_id >= 0 {
            let reg_proto = LloRegisterTypeProto::try_from(proto.register_type)
                .unwrap_or(LloRegisterTypeProto::RegisterTypeVreg);
            let class = LloRegClass::from_register_type_proto(reg_proto);
            vec![LloRegOperand::physical_for_class(
                class,
                proto.register_id as u32,
            )]
        } else {
            Vec::new()
        };
        let mut uses = Vec::with_capacity(proto.operands.len());
        for op in &proto.operands {
            match op.value {
                Some(llo_instruction_proto::llo_operand::Value::ConstantHandle(c)) => {
                    uses.push(LloRegOperand::Immediate(c.max(0) as u64));
                }
                Some(llo_instruction_proto::llo_operand::Value::InstructionOrdinal(id)) => {
                    uses.push(LloRegOperand::PhysicalVreg(VregIndex(
                        (id.max(0) & 63) as u8,
                    )));
                }
                None => {}
            }
        }
        let predicate = if proto.predication_operand_count > 0 {
            Some(PregIndex(0))
        } else {
            None
        };
        let negate_predicate = proto.predication_polarity
            == (llo_instruction_proto::PredicationPolarityProto::PolarityNegative as i32);
        let id_u32 = proto.ordinal.max(0) as u32;
        Self {
            id: id_u32,
            llo_opcode,
            tpu_opcode,
            unit,
            element_type,
            predicate,
            negate_predicate,
            defs,
            uses,
            imm_value: proto.scalar_constant_value,
            latency_cycles: unit.default_latency_cycles(generation),
            memory_space: 1,
            hlo_source_name: format!("llo_op_{id_u32}"),
        }
    }
}

/// Basic block of [`LloMicroOp`] instructions inside an [`LloIr`] (`LloBasicBlock`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LloBasicBlock {
    /// Symbolic block label (`"entry"`, `"loop.body"`, etc.).
    pub label: String,
    /// Whether this block is an inner software-pipelinable loop body.
    pub is_loop_body: bool,
    /// Static trip count when known.
    pub trip_count: Option<u32>,
    /// Ordered sequence of LLO micro-operations.
    pub ops: Vec<LloMicroOp>,
}

/// Strongly-typed LLO module IR (`LloIr`) across scalar, vector, MXU, XLU, and DMA units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LloIr {
    /// Module or kernel name.
    pub module_name: String,
    /// Target Cloud TPU generation (`V2..=V7x`).
    pub generation: TpuGeneration,
    /// Ordered basic blocks.
    pub blocks: Vec<LloBasicBlock>,
    /// Next available virtual register identifier.
    pub next_vreg_id: u32,
}

impl LloIr {
    /// Creates an empty [`LloIr`] module for `module_name` and `generation`.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Module identifier string.
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub fn new(module_name: impl Into<String>, generation: TpuGeneration) -> Self {
        Self {
            module_name: module_name.into(),
            generation,
            blocks: Vec::new(),
            next_vreg_id: 0,
        }
    }

    /// Allocates a fresh virtual register [`LloRegOperand::Virtual`] in `class`.
    ///
    /// # Arguments
    ///
    /// * `class` - Target [`LloRegClass`].
    pub const fn alloc_virtual_reg(&mut self, class: LloRegClass) -> LloRegOperand {
        let id = self.next_vreg_id;
        self.next_vreg_id = self.next_vreg_id.saturating_add(1);
        LloRegOperand::Virtual { id, class }
    }

    /// Returns the total number of [`LloMicroOp`]s across all basic blocks.
    #[must_use]
    pub fn total_micro_ops(&self) -> usize {
        self.blocks.iter().map(|b| b.ops.len()).sum()
    }

    /// Converts this [`LloIr`] into a wire-exact `xla.jellyfish.LloModuleProto`.
    #[must_use]
    pub fn to_llo_module_proto(&self) -> LloModuleProto {
        use crate::proto::generated_prost_schemas::xla::jellyfish::llo_region_proto::{
            RegionMember, region_member,
        };
        let mut members = Vec::with_capacity(self.total_micro_ops());
        let mut interned_strings = vec![self.module_name.clone()];
        for block in &self.blocks {
            interned_strings.push(block.label.clone());
            for op in &block.ops {
                members.push(RegionMember {
                    value: Some(region_member::Value::Instruction(op.to_instruction_proto())),
                });
            }
        }
        let limit_b = i32::try_from(members.len().max(1)).unwrap_or(1);
        LloModuleProto {
            hlo_instruction_name: self.module_name.clone(),
            hlo_module_name: self.module_name.clone(),
            hlo_module_id: 1,
            top_region: Some(LloRegionProto {
                members,
                name: String::from("entry"),
                start_bundleno: 0,
                limit_bundleno: limit_b,
                ordinal: 0,
            }),
            interned_strings,
            static_utilization: Some(StaticPerBundleUtilizationProto::default()),
            ..LloModuleProto::default()
        }
    }

    /// Reconstructs an [`LloIr`] from a serialized or in-memory `xla.jellyfish.LloModuleProto`.
    ///
    /// # Arguments
    ///
    /// * `proto` - Source [`LloModuleProto`].
    /// * `generation` - Target [`TpuGeneration`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidExecutableFormat`] if `proto.top_region` contains zero instructions.
    pub fn from_llo_module_proto(
        proto: &LloModuleProto,
        generation: TpuGeneration,
    ) -> LibTpuResult<Self> {
        use crate::proto::generated_prost_schemas::xla::jellyfish::llo_region_proto::region_member;
        let mut ops = Vec::new();
        if let Some(region) = &proto.top_region {
            for m in &region.members {
                if let Some(region_member::Value::Instruction(inst)) = &m.value {
                    ops.push(LloMicroOp::from_instruction_proto(inst, generation));
                }
            }
        }
        if ops.is_empty() {
            return Err(LibTpuError::InvalidExecutableFormat {
                reason: String::from("LloModuleProto contains zero LLO instructions"),
            });
        }
        let op_len = u32::try_from(ops.len()).unwrap_or(0);
        let name = if proto.hlo_module_name.is_empty() {
            String::from("llo_module")
        } else {
            proto.hlo_module_name.clone()
        };
        Ok(Self {
            module_name: name,
            generation,
            blocks: vec![LloBasicBlock {
                label: String::from("entry"),
                is_loop_body: false,
                trip_count: None,
                ops,
            }],
            next_vreg_id: op_len,
        })
    }
}

/// Generation-aware architectural LLO VLIW bundle (`LloBundle`) across `(S0, S1, V0, V1, MXU0..3, XLU0..1, DMA, IMM0..5)`.
///
/// Replaces the legacy synthetic `TensorCoreVliwBundle`: contains **zero** embedded `pc_offset` field
/// and **zero** synthetic `0x5A` CRC-8 bytes; program counter addresses are purely positional bundle indices
/// in `IMEM` (`0, 1, 2, ...`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LloBundle {
    /// Target Cloud TPU hardware generation (`V2..=V7x`).
    pub generation: TpuGeneration,
    /// Active functional unit bitmask (`bit 0 = S0`, `bit 1 = V0`, `bit 2 = M0`, `bit 3 = X0`, `bit 4 = D0`, `bit 5 = S1`, `bit 6 = V1`, `bit 7 = M1`).
    pub bundle_flags: u16,
    /// 16-bit bundle predicate execution mask (`@p0..@p15`).
    pub pred_mask: u16,
    /// Compressed post-bundle multi-cycle NOP stall count (`0..=15`).
    pub stall_cycles: u8,
    /// Dual scalar ALU slots `[S0, S1]`.
    pub scalar_slots: [ScalarAluSlot; 2],
    /// Dual vector ALU slots `[V0, V1]`.
    pub vector_slots: [VectorAluSlot; 2],
    /// Up to four systolic matrix multiply slots `[MXU0, MXU1, MXU2, MXU3]`.
    pub matrix_slots: [MatrixMxuSlot; 4],
    /// Dual cross-lane / EUP slots `[XLU0, XLU1]`.
    pub xlu_slots: [ExtendedUnarySlot; 2],
    /// Asynchronous DMA & ICI collective slot `D0`.
    pub dma_slot: AsyncDmaSlot,
    /// Six 32-bit immediate constant slots `IMM0..IMM5`.
    pub immediates: [u32; 6],
}

impl Default for LloBundle {
    fn default() -> Self {
        Self {
            generation: TpuGeneration::V5e,
            bundle_flags: 0,
            pred_mask: 0x0001,
            stall_cycles: 0,
            scalar_slots: [ScalarAluSlot::default(); 2],
            vector_slots: [VectorAluSlot::default(); 2],
            matrix_slots: [MatrixMxuSlot::default(); 4],
            xlu_slots: [ExtendedUnarySlot::default(); 2],
            dma_slot: AsyncDmaSlot::default(),
            immediates: [0_u32; 6],
        }
    }
}

impl LloBundle {
    /// Returns a shared reference to the primary `S0` [`ScalarAluSlot`].
    #[must_use]
    pub const fn scalar_slot(&self) -> &ScalarAluSlot {
        &self.scalar_slots[0]
    }

    /// Returns a shared reference to the primary `V0` [`VectorAluSlot`].
    #[must_use]
    pub const fn vector_slot(&self) -> &VectorAluSlot {
        &self.vector_slots[0]
    }

    /// Returns a shared reference to the primary `MXU0` [`MatrixMxuSlot`].
    #[must_use]
    pub const fn matrix_slot(&self) -> &MatrixMxuSlot {
        &self.matrix_slots[0]
    }

    /// Returns a shared reference to the primary `XLU0`/`EUP` [`ExtendedUnarySlot`].
    #[must_use]
    pub const fn xunary_slot(&self) -> &ExtendedUnarySlot {
        &self.xlu_slots[0]
    }

    /// Encodes secondary co-issue fields `(s1_opcode, v1_opcode, m1_acc_vreg)` into `scalar_slots[1]`,
    /// `vector_slots[1]`, `matrix_slots[1]`, and `bundle_flags` bits `5..=7`.
    ///
    /// # Arguments
    ///
    /// * `s1_opcode` - Secondary scalar ALU `S1` opcode (`0` if idle).
    /// * `v1_opcode` - Secondary vector ALU `V1` opcode (`0` if idle).
    /// * `m1_acc_vreg` - Secondary MXU `M1` accumulator register (`0` if idle).
    pub fn set_dual_issue_slots(&mut self, s1_opcode: u8, v1_opcode: u8, m1_acc_vreg: u8) {
        self.scalar_slots[1].opcode = s1_opcode;
        self.vector_slots[1].opcode = u16::from(v1_opcode);
        self.matrix_slots[1].opcode = u8::from(m1_acc_vreg != 0);
        self.matrix_slots[1].acc_vreg = u16::from(m1_acc_vreg);
        let mut flags = self.bundle_flags & 0x001F;
        if s1_opcode != 0 {
            flags |= 1 << 5;
        }
        if v1_opcode != 0 {
            flags |= 1 << 6;
        }
        if m1_acc_vreg != 0 {
            flags |= 1 << 7;
        }
        self.bundle_flags = flags;
    }

    /// Extracts `(s1_opcode, v1_opcode, m1_acc_vreg)` for the secondary co-issue slots (`S1`, `V1`, `M1`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn dual_issue_slots(&self) -> (u8, u8, u8) {
        let s1 = if (self.bundle_flags & (1 << 5)) != 0 {
            self.scalar_slots[1].opcode
        } else {
            0
        };
        let v1 = if (self.bundle_flags & (1 << 6)) != 0 {
            self.vector_slots[1].opcode as u8
        } else {
            0
        };
        let m1 = if (self.bundle_flags & (1 << 7)) != 0 {
            self.matrix_slots[1].acc_vreg as u8
        } else {
            0
        };
        (s1, v1, m1)
    }

    /// Returns the number of active functional unit slots (`1..=8`) in this bundle.
    #[must_use]
    pub const fn active_slot_count(&self) -> u32 {
        (self.bundle_flags & 0x00FF).count_ones()
    }

    /// Lowers a high-level [`TpuInstructionBundle`] into an [`LloBundle`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `bundle` - High-level [`TpuInstructionBundle`] to lower.
    /// * `generation` - Target [`TpuGeneration`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::RegisterIndexOverflow`] if any register slot index exceeds `255`.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    pub fn from_ir_bundle_for_gen(
        bundle: &TpuInstructionBundle,
        generation: TpuGeneration,
    ) -> LibTpuResult<Self> {
        const MAX_PHYSICAL_REGISTERS: usize = 255;
        if bundle.output_slot > MAX_PHYSICAL_REGISTERS {
            return Err(LibTpuError::RegisterIndexOverflow {
                slot_index: bundle.output_slot,
                max_registers: MAX_PHYSICAL_REGISTERS,
            });
        }
        for &src_slot in &bundle.operand_slots {
            if src_slot > MAX_PHYSICAL_REGISTERS {
                return Err(LibTpuError::RegisterIndexOverflow {
                    slot_index: src_slot,
                    max_registers: MAX_PHYSICAL_REGISTERS,
                });
            }
        }
        let op_byte = bundle.opcode as u8;
        let dst = bundle.output_slot as u32;
        let src0 = bundle.operand_slots.first().copied().unwrap_or(0) as u32;
        let src1 = bundle
            .operand_slots
            .get(1)
            .copied()
            .unwrap_or(src0 as usize) as u32;
        let src2 = bundle.operand_slots.get(2).copied().unwrap_or(0) as u32;

        let is_mxu = matches!(
            bundle.opcode,
            TpuOpCode::DotGeneralF32 | TpuOpCode::Convolution | TpuOpCode::DynamicConv
        );
        let is_xunary = matches!(
            bundle.opcode,
            TpuOpCode::ExpF32
                | TpuOpCode::TanhF32
                | TpuOpCode::Log
                | TpuOpCode::Logistic
                | TpuOpCode::Sqrt
                | TpuOpCode::Rsqrt
                | TpuOpCode::Sine
                | TpuOpCode::Cosine
                | TpuOpCode::Erf
                | TpuOpCode::Transpose
                | TpuOpCode::BroadcastInDim
                | TpuOpCode::Reverse
                | TpuOpCode::Iota
                | TpuOpCode::Gather
                | TpuOpCode::Scatter
        );
        let is_collective = matches!(
            bundle.opcode,
            TpuOpCode::AllReduceSumF32
                | TpuOpCode::AllGather
                | TpuOpCode::ReduceScatter
                | TpuOpCode::AllToAll
                | TpuOpCode::CollectivePermute
                | TpuOpCode::RaggedAllToAll
        );

        let mut flags: u16 = 0x0003;
        if is_mxu {
            flags |= 0x0004;
        }
        if is_xunary {
            flags |= 0x0008;
        }
        if (bundle.sync_flag_id != 0) | is_collective {
            flags |= 0x0010;
        }

        let opt_et = bundle.attrs.output_shape.as_ref().map(|s| s.element_type);
        let elem_code = opt_et.map_or(11_u8, |et| u8::try_from(et.as_raw()).unwrap_or(11));

        let mxu_slot = if is_mxu {
            let prec = match opt_et {
                Some(ElementType::Bf16 | ElementType::F16) => 0,
                Some(
                    ElementType::S8 | ElementType::U8 | ElementType::F8E4M3Fn | ElementType::F8E5M2,
                ) => 2,
                _ => 1,
            };
            MatrixMxuSlot {
                opcode: op_byte,
                precision_mode: prec,
                flags: 1,
                acc_vreg: dst as u16,
                lhs_vreg: src0 as u16,
                rhs_vreg: src1 as u16,
                k_tiles: 1,
            }
        } else {
            MatrixMxuSlot::default()
        };

        let xlu_slot = if is_xunary {
            ExtendedUnarySlot {
                opcode: op_byte,
                perm_mode: u8::from(matches!(
                    bundle.opcode,
                    TpuOpCode::Transpose | TpuOpCode::Reverse
                )),
                dst_vreg: dst as u16,
                src_vreg: src0 as u16,
                imm16: 0,
            }
        } else {
            ExtendedUnarySlot::default()
        };

        let dma_slot = AsyncDmaSlot {
            dma_opcode: if is_collective {
                op_byte
            } else {
                u8::from(bundle.sync_flag_id != 0)
            },
            src_space: 1,
            dst_space: u8::from(!is_collective),
            collective_op: if is_collective { op_byte } else { 0 },
            sync_flag_id: bundle.sync_flag_id,
        };

        let s0 = ScalarAluSlot {
            opcode: 1,
            dst_sreg: (dst & 0x1F) as u8,
            src0_sreg: (src0 & 0x1F) as u8,
            src1_sreg: (src1 & 0x1F) as u8,
            imm32: dst * 256,
        };
        let v0 = VectorAluSlot {
            opcode: u16::from(op_byte),
            element_type: elem_code,
            lane_mask: 0xFF,
            dst_vreg: dst,
            src0_vreg: src0,
            src1_vreg: src1,
            src2_vreg: src2,
        };

        Ok(Self {
            generation,
            bundle_flags: flags,
            pred_mask: 0x0001,
            stall_cycles: 0,
            scalar_slots: [s0, ScalarAluSlot::default()],
            vector_slots: [v0, VectorAluSlot::default()],
            matrix_slots: [
                mxu_slot,
                MatrixMxuSlot::default(),
                MatrixMxuSlot::default(),
                MatrixMxuSlot::default(),
            ],
            xlu_slots: [xlu_slot, ExtendedUnarySlot::default()],
            dma_slot,
            immediates: [s0.imm32, 0, 0, 0, 0, 0],
        })
    }

    /// Lowers a high-level [`TpuInstructionBundle`] into a default (`V5e`) [`LloBundle`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - High-level [`TpuInstructionBundle`] to lower.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::RegisterIndexOverflow`] if any register slot index exceeds `255`.
    pub fn from_ir_bundle(bundle: &TpuInstructionBundle) -> LibTpuResult<Self> {
        Self::from_ir_bundle_for_gen(bundle, TpuGeneration::V5e)
    }

    /// Reconstructs a [`TpuInstructionBundle`] at `pc_offset` from this [`LloBundle`].
    ///
    /// # Arguments
    ///
    /// * `pc_offset` - Positional bundle index in `IMEM`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the encoded vector opcode byte is unknown.
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_ir_bundle(&self, pc_offset: u32) -> LibTpuResult<TpuInstructionBundle> {
        let raw_op =
            u8::try_from(self.vector_slots[0].opcode).map_err(|_| LibTpuError::InvalidOpcode {
                raw_opcode: self.vector_slots[0].opcode,
            })?;
        let opcode = TpuOpCode::from_byte(raw_op)?;
        let v0 = &self.vector_slots[0];
        let mut operand_slots = match opcode.arity() {
            0 => Vec::new(),
            1 => vec![v0.src0_vreg as usize],
            2 => vec![v0.src0_vreg as usize, v0.src1_vreg as usize],
            _ => vec![
                v0.src0_vreg as usize,
                v0.src1_vreg as usize,
                v0.src2_vreg as usize,
            ],
        };
        if (operand_slots.is_empty()) && (v0.src0_vreg != 0) {
            operand_slots.push(v0.src0_vreg as usize);
        }
        let attrs = InstructionAttrs::default();
        let m0 = &self.matrix_slots[0];
        let output_slot = if m0.opcode != 0 {
            usize::from(m0.acc_vreg)
        } else {
            v0.dst_vreg as usize
        };
        Ok(TpuInstructionBundle {
            pc_offset,
            opcode,
            operand_slots,
            output_slot,
            sync_flag_id: self.dma_slot.sync_flag_id,
            attrs,
        })
    }

    /// Encodes this [`LloBundle`] into a canonical 64-byte (`512`-bit) `IMEM` word.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        crate::compiler::isa::encode_canonical_64b_bundle(self)
    }

    /// Decodes a canonical 64-byte (`512`-bit) `IMEM` word into an [`LloBundle`].
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 64-byte `IMEM` word.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        crate::compiler::isa::decode_canonical_64b_bundle(bytes, 0).unwrap_or_default()
    }

    /// Returns the hardware bundle byte width for `generation` (`32`, `40`, `52`, or `64` bytes).
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn generation_bundle_byte_width(generation: TpuGeneration) -> usize {
        crate::compiler::isa::generation_bundle_byte_width(generation)
    }

    /// Encodes this [`LloBundle`] into the exact binary bitstream for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub fn encode_generation_bitstream(&self, generation: TpuGeneration) -> Vec<u8> {
        crate::compiler::isa::encode_bundle_for_generation(generation, self)
    }

    /// Decodes a generation-specific binary bitstream into an [`LloBundle`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `bitstream` - Raw generation-encoded bundle bytes (`32`, `40`, `52`, or `64` bytes).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if length, magic, or `HighwayHash64` tag verification fails.
    pub fn decode_generation_bitstream(
        generation: TpuGeneration,
        bitstream: &[u8],
    ) -> LibTpuResult<Self> {
        crate::compiler::isa::decode_bundle_for_generation(generation, bitstream)
    }
}
