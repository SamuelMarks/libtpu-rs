//! Exhaustive unit and branch-coverage tests for Phase 6 (`llo`, `isa`, `sparsecore`, `vliw`, and `serialization`).

#![allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::float_cmp
)]

use super::isa::*;
use super::llo::*;
use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use super::parser::parse_mlir_ast_module;
use super::sparsecore::*;
use super::vliw::{
    assemble_llo_bundle, disassemble_llo_bundle, extract_slot_bitfield, parse_llo_assembly,
};
use super::{COMPILATION_PHASE_NAMES, CompiledProgram};
use crate::proto::{
    ClippingLimitsProto, HloComputationProto, HloInstructionProto, HloModuleProto,
    LearningRateProto, LloInstructionProto, LloModuleProto, LloOpcodeProto, LloRegionProto,
    LloRegisterTypeProto, OptimizationParametersProto, PjRtExecutableAbiVersionProto,
    PjRtPartialProgramProto, ProtoSchemaMessage as _, ShapeProto, llo_instruction_proto,
};
use crate::types::{
    DeviceAddress, ElementType, MatmulAccIndex, MemoryLayout, PregIndex, Shape, SregIndex,
    SyncFlagId, TpuGeneration,
};

#[test]
fn llo_ir_units_regclasses_operands_and_proto_bridge_cover_all_branches() {
    let all_units = [
        (0, LloFunctionalUnit::Scalar0),
        (1, LloFunctionalUnit::Scalar1),
        (2, LloFunctionalUnit::MatrixMxu0),
        (3, LloFunctionalUnit::CrossLaneXlu0),
        (4, LloFunctionalUnit::Vector0),
        (5, LloFunctionalUnit::AsyncDma),
        (6, LloFunctionalUnit::Vector1),
        (7, LloFunctionalUnit::VectorEup),
        (8, LloFunctionalUnit::MatrixMxu1),
        (9, LloFunctionalUnit::MatrixMxu2),
        (10, LloFunctionalUnit::MatrixMxu3),
        (11, LloFunctionalUnit::CrossLaneXlu1),
        (12, LloFunctionalUnit::ControlPred),
        (13, LloFunctionalUnit::ImmediateExt),
    ];
    for (code, unit) in all_units {
        assert_eq!(unit.unit_code(), code);
        assert_eq!(LloFunctionalUnit::from_unit_code(code), unit);
        assert_ne!(unit.slot_name(), "");
        assert!(unit.default_latency_cycles(TpuGeneration::V4) >= 1);
        assert!(unit.default_latency_cycles(TpuGeneration::V7x) >= 1);
    }
    assert_eq!(
        LloFunctionalUnit::from_unit_code(99),
        LloFunctionalUnit::ImmediateExt
    );

    let all_classes = [
        (0, LloRegClass::Sreg, LloRegisterTypeProto::RegisterTypeSreg),
        (1, LloRegClass::Vreg, LloRegisterTypeProto::RegisterTypeVreg),
        (
            2,
            LloRegClass::Mreg,
            LloRegisterTypeProto::RegisterTypeVmreg,
        ),
        (3, LloRegClass::Preg, LloRegisterTypeProto::RegisterTypePreg),
        (
            4,
            LloRegClass::SyncFlag,
            LloRegisterTypeProto::RegisterTypeDreg,
        ),
    ];
    for (code, cls, proto_ty) in all_classes {
        assert_eq!(cls.class_code(), code);
        assert_ne!(cls.bank_name(), "");
        assert_eq!(cls.to_register_type_proto(), proto_ty);
        assert_eq!(LloRegClass::from_register_type_proto(proto_ty), cls);
        assert!(cls.hardware_capacity(TpuGeneration::V2) >= 4);
        assert!(cls.hardware_capacity(TpuGeneration::V6e) >= 8);
        let (sp, bytes) = cls.spill_layout();
        assert!((sp == 1) || (sp == 2));
        assert!(bytes >= 8);
    }
    assert_eq!(
        LloRegClass::from_register_type_proto(LloRegisterTypeProto::RegisterTypeRpreg),
        LloRegClass::Preg
    );
    assert_eq!(
        LloRegClass::from_register_type_proto(LloRegisterTypeProto::RegisterTypeCbreg),
        LloRegClass::SyncFlag
    );
    assert_eq!(
        LloRegClass::from_register_type_proto(LloRegisterTypeProto::RegisterTypeNone),
        LloRegClass::Vreg
    );

    let ops = [
        LloRegOperand::Virtual {
            id: 9,
            class: LloRegClass::Vreg,
        },
        LloRegOperand::physical_for_class(LloRegClass::Sreg, 3),
        LloRegOperand::physical_for_class(LloRegClass::Vreg, 4),
        LloRegOperand::physical_for_class(LloRegClass::Mreg, 1),
        LloRegOperand::physical_for_class(LloRegClass::Preg, 2),
        LloRegOperand::physical_for_class(LloRegClass::SyncFlag, 7),
        LloRegOperand::Immediate(42),
    ];
    for (idx, op_ref) in ops.iter().copied().enumerate() {
        assert!(op_ref.raw_index() <= 42);
        let pred = match idx % 3 {
            0 => None,
            1 => Some(PregIndex(1)),
            _ => Some(PregIndex(2)),
        };
        let neg = (idx % 3) == 2;
        let defs = if idx == 6 { Vec::new() } else { vec![op_ref] };
        let micro = LloMicroOp {
            id: u32::try_from(idx).unwrap_or(0),
            llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
            tpu_opcode: TpuOpCode::AddF32,
            unit: LloFunctionalUnit::Vector0,
            element_type: ElementType::F32,
            predicate: pred,
            negate_predicate: neg,
            defs,
            uses: vec![LloRegOperand::Immediate(5), ops[1]],
            imm_value: 16,
            latency_cycles: 2,
            memory_space: 1,
            hlo_source_name: format!("op_{idx}"),
        };
        let p = micro.to_instruction_proto();
        let round = LloMicroOp::from_instruction_proto(&p, TpuGeneration::V5e);
        assert_eq!(round.id, micro.id);
    }

    // Also testImmediate def and malformed LloInstructionProto branches
    let imm_def_op = LloMicroOp {
        id: 77,
        llo_opcode: LloOpcodeProto::OpcodeNone,
        tpu_opcode: TpuOpCode::Identity,
        unit: LloFunctionalUnit::Scalar0,
        element_type: ElementType::F32,
        predicate: None,
        negate_predicate: false,
        defs: vec![LloRegOperand::Immediate(1)],
        uses: Vec::new(),
        imm_value: 0,
        latency_cycles: 1,
        memory_space: 2,
        hlo_source_name: String::from("imm_def"),
    };
    let mut bad_inst_proto: LloInstructionProto = imm_def_op.to_instruction_proto();
    bad_inst_proto.opcode = -999;
    bad_inst_proto.source_bus = 999;
    bad_inst_proto.associated_shape = None;
    bad_inst_proto.register_id = 2;
    bad_inst_proto.register_type = -999;
    bad_inst_proto
        .operands
        .push(llo_instruction_proto::LloOperand { value: None });
    let recovered = LloMicroOp::from_instruction_proto(&bad_inst_proto, TpuGeneration::V4);
    assert_eq!(recovered.tpu_opcode, TpuOpCode::AddF32);

    // LloIr <-> LloModuleProto roundtrip + error branches
    assert!(LloIr::from_llo_module_proto(&LloModuleProto::default(), TpuGeneration::V5e).is_err());
    let non_inst_mod = LloModuleProto {
        top_region: Some(LloRegionProto {
            members: vec![
                crate::proto::generated_prost_schemas::xla::jellyfish::llo_region_proto::RegionMember {
                    value: None,
                },
            ],
            ..LloRegionProto::default()
        }),
        ..LloModuleProto::default()
    };
    assert!(LloIr::from_llo_module_proto(&non_inst_mod, TpuGeneration::V5e).is_err());
    let mut ir = LloIr::new("kernel_llo", TpuGeneration::V6e);
    let vr = ir.alloc_virtual_reg(LloRegClass::Vreg);
    ir.blocks.push(LloBasicBlock {
        label: String::from("entry"),
        is_loop_body: false,
        trip_count: None,
        ops: vec![LloMicroOp {
            id: 0,
            llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
            tpu_opcode: TpuOpCode::AddF32,
            unit: LloFunctionalUnit::Vector0,
            element_type: ElementType::F32,
            predicate: None,
            negate_predicate: false,
            defs: vec![vr],
            uses: vec![vr],
            imm_value: 0,
            latency_cycles: 2,
            memory_space: 1,
            hlo_source_name: String::from("add0"),
        }],
    });
    assert_eq!(ir.total_micro_ops(), 1);
    let mut mod_proto = ir.to_llo_module_proto();
    let decoded_ir = LloIr::from_llo_module_proto(&mod_proto, TpuGeneration::V6e);
    assert!(decoded_ir.is_ok());
    mod_proto.hlo_module_name.clear();
    let unnamed_ir = LloIr::from_llo_module_proto(&mod_proto, TpuGeneration::V6e);
    assert!(unnamed_ir.is_ok());
}

#[test]
fn llo_bundle_and_instruction_selector_cover_all_kernels_and_mosaic_ops() {
    // Verify ThreeFry2x32 deterministic bijection
    let tf_a = eval_threefry2x32_counter([0x1234_5678, 0x9ABC_DEF0], [0, 1]);
    let tf_b = eval_threefry2x32_counter([0x1234_5678, 0x9ABC_DEF0], [0, 2]);
    assert_ne!(tf_a, tf_b);
    assert_eq!(
        tf_a,
        eval_threefry2x32_counter([0x1234_5678, 0x9ABC_DEF0], [0, 1])
    );

    // LloBundle dual-issue & precision branches
    let mut lb = LloBundle::default();
    assert_eq!(lb.dual_issue_slots(), (0, 0, 0));
    lb.set_dual_issue_slots(0, 0, 0);
    assert_eq!(lb.dual_issue_slots(), (0, 0, 0));
    lb.set_dual_issue_slots(2, 4, 1);
    assert_eq!(lb.dual_issue_slots(), (2, 4, 1));
    assert_eq!(LloBundle::from_bytes(&[0_u8; 64]), LloBundle::default());

    for et in [ElementType::Bf16, ElementType::F8E4M3Fn, ElementType::F32] {
        let b = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::DotGeneralF32,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 0,
            attrs: InstructionAttrs {
                output_shape: Some(Shape {
                    element_type: et,
                    dims: vec![8, 128],
                    dynamic_dims: vec![false, false],
                    layout: MemoryLayout::default_tpu_tiled(2),
                }),
                ..InstructionAttrs::default()
            },
        };
        assert!(LloBundle::from_ir_bundle_for_gen(&b, TpuGeneration::V5p).is_ok());
    }
    let coll_b = TpuInstructionBundle {
        pc_offset: 0,
        opcode: TpuOpCode::AllReduceSumF32,
        operand_slots: vec![0],
        output_slot: 1,
        sync_flag_id: 0,
        attrs: InstructionAttrs::default(),
    };
    let coll_llo = LloBundle::from_ir_bundle_for_gen(&coll_b, TpuGeneration::V4);
    assert!(coll_llo.is_ok());
    let mut arity0_nonzero = LloBundle::default();
    arity0_nonzero.vector_slots[0].opcode = u16::from(TpuOpCode::Constant as u8);
    arity0_nonzero.vector_slots[0].src0_vreg = 3;
    assert!(arity0_nonzero.to_ir_bundle(0).is_ok());

    // LloInstructionSelector over all composite opcodes
    let sel = LloInstructionSelector::new(TpuGeneration::V6e);
    let mut bundle_list = Vec::new();
    for (idx, op) in [
        TpuOpCode::RngBitGenerator,
        TpuOpCode::ExpF32,
        TpuOpCode::Log,
        TpuOpCode::TanhF32,
        TpuOpCode::Rsqrt,
        TpuOpCode::Sine,
        TpuOpCode::Cosine,
        TpuOpCode::Sort,
        TpuOpCode::TopK,
        TpuOpCode::TriangularSolve,
        TpuOpCode::Cholesky,
        TpuOpCode::DotGeneralF32,
        TpuOpCode::AddF32,
    ]
    .into_iter()
    .enumerate()
    {
        bundle_list.push(TpuInstructionBundle {
            pc_offset: u32::try_from(idx).unwrap_or(0),
            opcode: op,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: u32::try_from(idx % 2).unwrap_or(0),
            attrs: InstructionAttrs::default(),
        });
    }
    let selected_ir = sel.select_from_bundles("all_kernels", &bundle_list);
    assert!(selected_ir.total_micro_ops() >= 80);

    // Also test short bundle_list (len <= 2)
    let short_ir = sel.select_from_bundles("short", bundle_list.get(..1).unwrap_or(&[]));
    assert!(short_ir.blocks.first().is_some_and(|b| !b.is_loop_body));

    // HLO + Mosaic selection (including empty error and nested scf.for)
    assert!(
        sel.select_from_hlo_and_mosaic(&HloModuleProto::default(), None)
            .is_err()
    );
    let mnemonics = [
        "subtract",
        "multiply",
        "divide",
        "dot",
        "convolution",
        "exponential",
        "log",
        "tanh",
        "rsqrt",
        "sine",
        "cosine",
        "rng-bit-generator",
        "sort",
        "triangular-solve",
        "cholesky",
        "transpose",
        "all-reduce",
        "add",
    ];
    let hlo_insts: Vec<HloInstructionProto> = mnemonics
        .iter()
        .enumerate()
        .map(|(i, &m)| HloInstructionProto {
            name: format!("inst_{i}"),
            opcode: String::from(m),
            id: i64::try_from(i).unwrap_or(0),
            operand_ids: if i > 0 { vec![0] } else { Vec::new() },
            shape: Some(ShapeProto {
                element_type: ElementType::F32.as_raw(),
                dimensions: vec![8, 128],
                ..ShapeProto::default()
            }),
            ..HloInstructionProto::default()
        })
        .collect();
    let hlo_mod = HloModuleProto {
        name: String::from("hlo_and_mosaic"),
        computations: vec![HloComputationProto {
            name: String::from("main"),
            instructions: hlo_insts,
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };

    let mosaic_mlir = r#"
        module @mosaic_m {
            func.func @loop_fn(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
                %0 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (tensor<8x128xf32>) {
                    %m = "tpu.matmul"(%acc, %acc) : (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
                    %d = "tpu.enqueue_dma"(%m) : (tensor<8x128xf32>) -> tensor<8x128xf32>
                    %w = "tpu.wait_dma"(%d) : (tensor<8x128xf32>) -> tensor<8x128xf32>
                    %s = "tpu.semaphore_signal"(%w) : (tensor<8x128xf32>) -> tensor<8x128xf32>
                    %r = "tpu.relayout"(%s) : (tensor<8x128xf32>) -> tensor<8x128xf32>
                    %a = arith.addf %r, %r : tensor<8x128xf32>
                    scf.yield %a : tensor<8x128xf32>
                }
                return %0 : tensor<8x128xf32>
            }
            func.func @noloop_fn(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
                %0 = "tpu.matmul"(%arg0, %arg0) : (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
                return %0 : tensor<8x128xf32>
            }
            func.func private @empty_fn(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32>
        }
    "#;
    let parsed_mosaic = parse_mlir_ast_module(mosaic_mlir);
    assert!(parsed_mosaic.is_ok());
    if let Ok(mosaic_ast) = parsed_mosaic {
        let combined = sel.select_from_hlo_and_mosaic(&hlo_mod, Some(&mosaic_ast));
        assert!(combined.is_ok());
        let mosaic_only =
            sel.select_from_hlo_and_mosaic(&HloModuleProto::default(), Some(&mosaic_ast));
        assert!(mosaic_only.is_ok());
    }
}

#[test]
fn llo_modulo_pipeliner_regalloc_and_packer_cover_all_hazards_and_spills() {
    // 1. SwingModuloPipeliner across V2 and V6e + error & MRT bump branches
    let pipe_v2 = SwingModuloPipeliner::new(TpuGeneration::V2);
    let pipe_v6 = SwingModuloPipeliner::new(TpuGeneration::V6e);
    for u in [
        LloFunctionalUnit::Scalar0,
        LloFunctionalUnit::Vector0,
        LloFunctionalUnit::MatrixMxu0,
        LloFunctionalUnit::VectorEup,
        LloFunctionalUnit::AsyncDma,
    ] {
        assert!(pipe_v2.slot_capacity(u) >= 1);
        assert!(pipe_v6.slot_capacity(u) >= 1);
    }
    let empty_blk = LloBasicBlock {
        label: String::from("empty"),
        is_loop_body: true,
        trip_count: Some(4),
        ops: Vec::new(),
    };
    assert!(pipe_v6.pipeline_loop(&empty_blk).is_err());

    // Build a multi-stage dependent loop block with loop-carried recurrence (def of v0 after use of v0)
    let v0 = LloRegOperand::Virtual {
        id: 0,
        class: LloRegClass::Vreg,
    };
    let v1 = LloRegOperand::Virtual {
        id: 1,
        class: LloRegClass::Vreg,
    };
    let m0 = LloRegOperand::Virtual {
        id: 2,
        class: LloRegClass::Mreg,
    };
    let loop_ops = vec![
        LloMicroOp {
            id: 0,
            llo_opcode: LloOpcodeProto::OpcodeDmaHbmToVmem,
            tpu_opcode: TpuOpCode::OutboundDma,
            unit: LloFunctionalUnit::AsyncDma,
            element_type: ElementType::F32,
            predicate: None,
            negate_predicate: false,
            defs: vec![LloRegOperand::Virtual {
                id: 3,
                class: LloRegClass::SyncFlag,
            }],
            uses: vec![v0],
            imm_value: 1,
            latency_cycles: 6,
            memory_space: 1,
            hlo_source_name: String::from("dma0"),
        },
        LloMicroOp {
            id: 1,
            llo_opcode: LloOpcodeProto::OpcodeVectorPow2F32,
            tpu_opcode: TpuOpCode::ExpF32,
            unit: LloFunctionalUnit::VectorEup,
            element_type: ElementType::F32,
            predicate: None,
            negate_predicate: false,
            defs: vec![v1],
            uses: vec![v0],
            imm_value: 0,
            latency_cycles: 4,
            memory_space: 1,
            hlo_source_name: String::from("eup1"),
        },
        LloMicroOp {
            id: 2,
            llo_opcode: LloOpcodeProto::OpcodeVectorPermute,
            tpu_opcode: TpuOpCode::Transpose,
            unit: LloFunctionalUnit::CrossLaneXlu0,
            element_type: ElementType::F32,
            predicate: None,
            negate_predicate: false,
            defs: vec![v1],
            uses: vec![v1],
            imm_value: 0,
            latency_cycles: 4,
            memory_space: 1,
            hlo_source_name: String::from("xlu2"),
        },
        LloMicroOp {
            id: 3,
            llo_opcode: LloOpcodeProto::OpcodeVectorMatmul,
            tpu_opcode: TpuOpCode::DotGeneralF32,
            unit: LloFunctionalUnit::MatrixMxu0,
            element_type: ElementType::F32,
            predicate: None,
            negate_predicate: false,
            defs: vec![m0],
            uses: vec![v1],
            imm_value: 0,
            latency_cycles: 8,
            memory_space: 1,
            hlo_source_name: String::from("mxu3"),
        },
        LloMicroOp {
            id: 4,
            llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
            tpu_opcode: TpuOpCode::AddF32,
            unit: LloFunctionalUnit::Vector0,
            element_type: ElementType::F32,
            predicate: None,
            negate_predicate: false,
            defs: vec![v0],
            uses: vec![m0],
            imm_value: 0,
            latency_cycles: 2,
            memory_space: 1,
            hlo_source_name: String::from("vec4"),
        },
        LloMicroOp {
            id: 5,
            llo_opcode: LloOpcodeProto::OpcodeScalarSyncAdd,
            tpu_opcode: TpuOpCode::AfterAll,
            unit: LloFunctionalUnit::Scalar0,
            element_type: ElementType::S32,
            predicate: None,
            negate_predicate: false,
            defs: vec![LloRegOperand::Virtual {
                id: 4,
                class: LloRegClass::Sreg,
            }],
            uses: Vec::new(),
            imm_value: 1,
            latency_cycles: 1,
            memory_space: 2,
            hlo_source_name: String::from("s5"),
        },
    ];
    let loop_blk = LloBasicBlock {
        label: String::from("pipelined_loop"),
        is_loop_body: true,
        trip_count: Some(16),
        ops: loop_ops,
    };
    let piped = pipe_v6.pipeline_loop(&loop_blk);
    assert!(piped.is_ok());
    if let Ok(pl) = piped {
        assert!(pl.stage_count >= 2);
        assert!(!pl.prolog_ops.len().eq(&0));
        assert!(!pl.kernel_ops.len().eq(&0));
        assert!(!pl.epilog_ops.len().eq(&0));
    }

    // Trigger min_ii > max_ii_cycles error
    let tiny_pipe = SwingModuloPipeliner {
        generation: TpuGeneration::V2,
        max_ii_cycles: 1,
    };
    assert!(tiny_pipe.pipeline_loop(&loop_blk).is_err());

    // Trigger MRT collision retry (two EUP ops becoming ready at cycles 0 and 1 when RecMII=1, ResMII=2 -> achieved_ii=2, both want row 0 when latency makes second op earliest=2!)
    let mrt_bump_blk = LloBasicBlock {
        label: String::from("mrt_collision"),
        is_loop_body: true,
        trip_count: Some(8),
        ops: vec![
            LloMicroOp {
                id: 0,
                llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                tpu_opcode: TpuOpCode::AddF32,
                unit: LloFunctionalUnit::Vector0,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![v0],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 2,
                memory_space: 1,
                hlo_source_name: String::from("def_v0"),
            },
            LloMicroOp {
                id: 1,
                llo_opcode: LloOpcodeProto::OpcodeVectorPow2F32,
                tpu_opcode: TpuOpCode::ExpF32,
                unit: LloFunctionalUnit::VectorEup,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![v1],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("eup_a"),
            },
            LloMicroOp {
                id: 2,
                llo_opcode: LloOpcodeProto::OpcodeVectorLog2F32,
                tpu_opcode: TpuOpCode::Log,
                unit: LloFunctionalUnit::VectorEup,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![v1],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("eup_b"),
            },
            LloMicroOp {
                id: 3,
                llo_opcode: LloOpcodeProto::OpcodeVectorTanhF32,
                tpu_opcode: TpuOpCode::TanhF32,
                unit: LloFunctionalUnit::VectorEup,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![v1],
                uses: vec![v0],
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("eup_c"),
            },
        ],
    };
    // Note: 3 EUP ops -> ResMII = 3; eup_a @ 0 (row 0), eup_b @ 1 (row 1), eup_c earliest=2 (row 2) -> fits;
    // With 2 EUP ops where eup_a @ 0 and eup_b earliest=2 on II=2, delta=0 (row 0) collides, delta=1 (row 1) succeeds;
    // With 3 ops in same resource where two are pinned to row 0 & row 1 and third has earliest=2 on II=2... wait: 3 ops -> ResMII=3!
    // How can MRT fail at `achieved_ii`? Two ops of group 1 (`Vector`, cap=2 on V6e -> ResMII=1!Wait: on V6e, 2 Vector ops have cap=2 so ResMII=1, and 1 EUP op @0 + 1 EUP op? No, 2 EUP ops -> ResMII=2).
    // Wait: what if `loop_blk` has 2 EUP ops on II=2? With cap=1 and `for delta in 0..achieved_ii`, there are `achieved_ii` distinct modulo rows, and if total ops in that group <= `achieved_ii * cap`, by Pigeonhole Principle at least one `delta in 0..achieved_ii` ALWAYS has `used < cap`!
    assert!(pipe_v6.pipeline_loop(&mrt_bump_blk).is_ok());

    // 2. LloRegisterAllocator: zero-capacity error + VMEM & SMEM spilling + interval expiration
    let zero_alloc = LloRegisterAllocator::with_capacities(
        TpuGeneration::V4,
        LloBankCapacities {
            sregs: 0,
            vregs: 4,
            mregs: 2,
            pregs: 2,
            sync_flags: 2,
        },
    );
    let mut dummy_ir = LloIr::new("z", TpuGeneration::V4);
    assert!(zero_alloc.allocate(&mut dummy_ir).is_err());

    let default_alloc = LloRegisterAllocator::new(TpuGeneration::V5p);
    let mut ok_ir = LloIr::new("ok", TpuGeneration::V5p);
    assert!(default_alloc.allocate(&mut ok_ir).is_ok());

    // Tight capacities (1 or 2 per bank) to trigger BOTH VMEM spills (Vreg) and SMEM spills (Sreg) AND interval expiration
    let tight_alloc = LloRegisterAllocator::with_capacities(
        TpuGeneration::V4,
        LloBankCapacities {
            sregs: 1,
            vregs: 2,
            mregs: 2,
            pregs: 2,
            sync_flags: 2,
        },
    );
    let sv0 = LloRegOperand::Virtual {
        id: 10,
        class: LloRegClass::Sreg,
    };
    let sv1 = LloRegOperand::Virtual {
        id: 11,
        class: LloRegClass::Sreg,
    };
    let vv0 = LloRegOperand::Virtual {
        id: 20,
        class: LloRegClass::Vreg,
    };
    let vv1 = LloRegOperand::Virtual {
        id: 21,
        class: LloRegClass::Vreg,
    };
    let vv2 = LloRegOperand::Virtual {
        id: 22,
        class: LloRegClass::Vreg,
    };
    let mut spill_ir = LloIr {
        module_name: String::from("spill_kernel"),
        generation: TpuGeneration::V4,
        next_vreg_id: 30,
        blocks: vec![LloBasicBlock {
            label: String::from("entry"),
            is_loop_body: false,
            trip_count: None,
            ops: vec![
                LloMicroOp {
                    id: 0,
                    llo_opcode: LloOpcodeProto::OpcodeScalarLoad,
                    tpu_opcode: TpuOpCode::Identity,
                    unit: LloFunctionalUnit::Scalar0,
                    element_type: ElementType::S32,
                    predicate: Some(PregIndex(0)),
                    negate_predicate: false,
                    defs: vec![sv0],
                    uses: vec![LloRegOperand::Immediate(0)],
                    imm_value: 0,
                    latency_cycles: 1,
                    memory_space: 2,
                    hlo_source_name: String::from("s_def0"),
                },
                LloMicroOp {
                    id: 1,
                    llo_opcode: LloOpcodeProto::OpcodeScalarLoad,
                    tpu_opcode: TpuOpCode::Identity,
                    unit: LloFunctionalUnit::Scalar1,
                    element_type: ElementType::S32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![sv1],
                    uses: vec![sv0],
                    imm_value: 8,
                    latency_cycles: 1,
                    memory_space: 2,
                    hlo_source_name: String::from("s_def1"),
                },
                LloMicroOp {
                    id: 2,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![vv0],
                    uses: vec![sv1],
                    imm_value: 0,
                    latency_cycles: 2,
                    memory_space: 1,
                    hlo_source_name: String::from("v_def0"),
                },
                LloMicroOp {
                    id: 3,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector1,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![vv1],
                    uses: Vec::new(),
                    imm_value: 0,
                    latency_cycles: 2,
                    memory_space: 1,
                    hlo_source_name: String::from("v_def1"),
                },
                LloMicroOp {
                    id: 4,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![vv2],
                    uses: vec![vv0, vv1],
                    imm_value: 0,
                    latency_cycles: 24,
                    memory_space: 1,
                    hlo_source_name: String::from("v_use_both"),
                },
                LloMicroOp {
                    id: 5,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![LloRegOperand::PhysicalMreg(MatmulAccIndex(0))],
                    uses: vec![
                        vv2,
                        LloRegOperand::PhysicalPred(PregIndex(1)),
                        LloRegOperand::PhysicalSyncFlag(SyncFlagId(2)),
                    ],
                    imm_value: 0,
                    latency_cycles: 2,
                    memory_space: 1,
                    hlo_source_name: String::from("after_long_gap"),
                },
            ],
        }],
    };
    let alloc_proto = tight_alloc.allocate(&mut spill_ir);
    assert!(alloc_proto.is_ok());
    if let Ok(ap) = alloc_proto {
        assert!(!ap.spill_slots.len().eq(&0));
    }

    // 3. LloVliwBundlePacker: empty IR + full multi-slot overflow & >15-cycle NOP stall compression
    let v6_packer = LloVliwBundlePacker::new(TpuGeneration::V6e);
    let empty_packed = v6_packer.pack_ir(&LloIr::new("empty", TpuGeneration::V6e));
    assert_eq!(empty_packed.bundles.len(), 1);

    // Add 3 scalar, 3 vector, 3 MXU, 2 XLU, 2 DMA ops at same earliest cycle to exercise S0->S1->overflow, V0->V1->overflow, M0->M1->overflow, X0->overflow, D0->overflow!
    if let Some(b0) = spill_ir.blocks.first_mut() {
        b0.ops.extend([
            LloMicroOp {
                id: 20,
                llo_opcode: LloOpcodeProto::OpcodeScalarLoad,
                tpu_opcode: TpuOpCode::Identity,
                unit: LloFunctionalUnit::Scalar0,
                element_type: ElementType::S32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::Virtual {
                    id: 99,
                    class: LloRegClass::Sreg,
                }],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 2,
                hlo_source_name: String::from("s_a"),
            },
            LloMicroOp {
                id: 21,
                llo_opcode: LloOpcodeProto::OpcodeScalarLoad,
                tpu_opcode: TpuOpCode::Identity,
                unit: LloFunctionalUnit::Scalar1,
                element_type: ElementType::S32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalSreg(SregIndex(2))],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 2,
                hlo_source_name: String::from("s_b"),
            },
            LloMicroOp {
                id: 22,
                llo_opcode: LloOpcodeProto::OpcodeScalarLoad,
                tpu_opcode: TpuOpCode::Identity,
                unit: LloFunctionalUnit::Scalar0,
                element_type: ElementType::S32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalSreg(SregIndex(3))],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 2,
                hlo_source_name: String::from("s_c"),
            },
            LloMicroOp {
                id: 23,
                llo_opcode: LloOpcodeProto::OpcodeVectorMatmul,
                tpu_opcode: TpuOpCode::DotGeneralF32,
                unit: LloFunctionalUnit::MatrixMxu0,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalMreg(MatmulAccIndex(1))],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("m_a"),
            },
            LloMicroOp {
                id: 24,
                llo_opcode: LloOpcodeProto::OpcodeVectorMatmul,
                tpu_opcode: TpuOpCode::DotGeneralF32,
                unit: LloFunctionalUnit::MatrixMxu1,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalMreg(MatmulAccIndex(2))],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("m_b"),
            },
            LloMicroOp {
                id: 25,
                llo_opcode: LloOpcodeProto::OpcodeVectorMatmul,
                tpu_opcode: TpuOpCode::DotGeneralF32,
                unit: LloFunctionalUnit::MatrixMxu2,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalMreg(MatmulAccIndex(3))],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("m_c"),
            },
            LloMicroOp {
                id: 26,
                llo_opcode: LloOpcodeProto::OpcodeVectorPermute,
                tpu_opcode: TpuOpCode::Transpose,
                unit: LloFunctionalUnit::CrossLaneXlu0,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalPred(PregIndex(3))],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("x_a"),
            },
            LloMicroOp {
                id: 27,
                llo_opcode: LloOpcodeProto::OpcodeVectorPermute,
                tpu_opcode: TpuOpCode::Transpose,
                unit: LloFunctionalUnit::CrossLaneXlu1,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalPred(PregIndex(4))],
                uses: Vec::new(),
                imm_value: 0,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("x_b"),
            },
            LloMicroOp {
                id: 28,
                llo_opcode: LloOpcodeProto::OpcodeDmaHbmToVmem,
                tpu_opcode: TpuOpCode::OutboundDma,
                unit: LloFunctionalUnit::AsyncDma,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalSyncFlag(SyncFlagId(5))],
                uses: Vec::new(),
                imm_value: 5,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("d_a"),
            },
            LloMicroOp {
                id: 29,
                llo_opcode: LloOpcodeProto::OpcodeDmaHbmToVmem,
                tpu_opcode: TpuOpCode::OutboundDma,
                unit: LloFunctionalUnit::AsyncDma,
                element_type: ElementType::F32,
                predicate: None,
                negate_predicate: false,
                defs: vec![LloRegOperand::PhysicalSyncFlag(SyncFlagId(6))],
                uses: Vec::new(),
                imm_value: 6,
                latency_cycles: 1,
                memory_space: 1,
                hlo_source_name: String::from("d_b"),
            },
        ]);
    }
    let res_v6 = v6_packer.pack_ir(&spill_ir);
    assert!(res_v6.bundles.len() >= 4);
    assert!(!res_v6.packer_infos.len().eq(&0));

    let v2_packer = LloVliwBundlePacker::new(TpuGeneration::V2);
    let res_v2 = v2_packer.pack_ir(&spill_ir);
    assert!(res_v2.bundles.len() >= res_v6.bundles.len());
}

#[test]
fn all_six_generation_isa_codecs_and_golden_bitstreams_round_trip_and_reject_corruption() {
    let golden_reports = verify_all_golden_isa_bitstreams();
    assert!(golden_reports.is_ok());
    if let Ok(reps) = golden_reports {
        assert_eq!(reps.len(), 6);
        for r in &reps {
            assert!(r.bitstream_bytes >= 64);
            assert!(r.disassembly.contains("pc=0000"));
            assert_ne!(r.bitstream_highwayhash64, 0);
        }
    }

    let mut active_b = LloBundle {
        generation: TpuGeneration::V7x,
        bundle_flags: 0x001F,
        pred_mask: 0x0003,
        stall_cycles: 2,
        ..LloBundle::default()
    };
    active_b.vector_slots[0].opcode = 250; // invalid byte -> "Nop" branch in format_generation_bundle_line
    active_b.matrix_slots[0].opcode = 1;
    active_b.xlu_slots[0].opcode = 11;
    active_b.xlu_slots[0].imm16 = 7;
    active_b.dma_slot.dma_opcode = 14;
    active_b.dma_slot.collective_op = 14;
    active_b.dma_slot.sync_flag_id = 9;
    let line = format_generation_bundle_line("iwc", 0, &active_b);
    assert!(
        line.contains("Nop")
            && line.contains("M0(")
            && line.contains("X0(")
            && line.contains("D0(")
    );

    let mut idle_b = LloBundle::default();
    idle_b.dma_slot.dma_opcode = 0;
    idle_b.xlu_slots[0].imm16 = 0x1234;
    let idle_line = format_generation_bundle_line("gfc", 1, &idle_b);
    assert!(idle_line.contains("isa=gfc"));

    // Protobuf program lowerers across all 6 codecs
    assert_eq!(
        JellyfishIsaCodec::to_isa_program_proto(&[active_b])
            .bundles
            .len(),
        1
    );
    assert_eq!(
        DragonfishIsaCodec::to_isa_program_proto(&[active_b])
            .bundles
            .len(),
        1
    );
    let mut cmem_b = active_b;
    cmem_b.dma_slot.src_space = 3;
    assert!(
        PufferfishIsaCodec::to_isa_program_proto(&[cmem_b, idle_b])
            .bundles
            .first()
            .is_some_and(|b| b.cmem_load.is_some())
    );
    assert_eq!(
        ViperfishIsaCodec::to_isa_program_proto(&[active_b])
            .bundles
            .len(),
        1
    );
    assert_eq!(
        GhostfishIsaCodec::to_isa_program_proto(&[active_b])
            .bundles
            .len(),
        1
    );
    assert_eq!(
        IronwoodIsaCodec::to_isa_program_proto(&[active_b])
            .bundles
            .len(),
        1
    );

    // Generation-specific descriptors + error branches
    let pfc_bytes = PufferfishIsaCodec::encode_bundle_with_optical(
        &cmem_b,
        PufferfishOpticalDmaDescriptor {
            delta_x: 2,
            delta_y: 0,
            delta_z: -1,
            cmem_enabled: false,
        },
    );
    let opt_desc = PufferfishIsaCodec::decode_optical_descriptor(&pfc_bytes);
    assert!(opt_desc.is_ok());
    assert!(PufferfishIsaCodec::decode_optical_descriptor(&[0_u8; 10]).is_err());

    for nib in 0_u8..=5 {
        let m = ViperfishMxuMode::from_nibble(nib);
        let vfc_bytes = ViperfishIsaCodec::encode_bundle_with_mode(&active_b, m, nib == 3);
        let dec_m = ViperfishIsaCodec::decode_mxu_mode(&vfc_bytes);
        assert_eq!(dec_m.ok(), Some((m, nib == 3)));
    }
    assert!(ViperfishIsaCodec::decode_mxu_mode(&[0_u8; 5]).is_err());

    let gfc_bytes = GhostfishIsaCodec::encode_bundle_trillium(&idle_b, false, 9);
    assert_eq!(
        GhostfishIsaCodec::decode_trillium_descriptors(&gfc_bytes).ok(),
        Some((false, 9))
    );
    assert!(GhostfishIsaCodec::decode_trillium_descriptors(&[0_u8; 4]).is_err());

    let iwc_zero = IronwoodIsaCodec::encode_bundle_ironwood(
        &idle_b,
        IronwoodMicroscalingDescriptor::default(),
    );
    assert!(IronwoodIsaCodec::decode_ironwood_descriptor(&iwc_zero).is_ok());
    assert!(IronwoodIsaCodec::decode_ironwood_descriptor(&[0_u8; 8]).is_err());

    // Corrupt length / magic / HighwayHash64 tag across all 6 codecs
    let check_codec_errors = |mut valid: Vec<u8>, tpu_gen: TpuGeneration| {
        assert!(decode_bundle_for_generation(tpu_gen, valid.get(..4).unwrap_or(&[])).is_err());
        valid.get_mut(7).into_iter().for_each(|b| *b ^= 0x55);
        assert!(decode_bundle_for_generation(tpu_gen, &valid).is_err());
        valid.get_mut(0).into_iter().for_each(|b| *b ^= 0xFF);
        assert!(decode_bundle_for_generation(tpu_gen, &valid).is_err());
    };
    for tpu_gen in [
        TpuGeneration::V2,
        TpuGeneration::V3,
        TpuGeneration::V4,
        TpuGeneration::V5e,
        TpuGeneration::V6e,
        TpuGeneration::V7x,
    ] {
        let enc = encode_bundle_for_generation(tpu_gen, &active_b);
        check_codec_errors(enc, tpu_gen);
    }
    let jfc_good = JellyfishIsaCodec::encode_bundle(&active_b);
    assert!(JellyfishIsaCodec::disassemble_bundle(&jfc_good, 0).is_ok());
    assert!(JellyfishIsaCodec::disassemble_bundle(&[0_u8; 32], 0).is_err());
    assert!(JellyfishIsaCodec::decode_stream(&[]).is_err());
    assert!(JellyfishIsaCodec::disassemble_stream(&[0_u8; 7]).is_err());

    // LLO assembly across all generations and slots in vliw.rs
    for g in [
        TpuGeneration::V2,
        TpuGeneration::V3,
        TpuGeneration::V4,
        TpuGeneration::V5e,
        TpuGeneration::V6e,
        TpuGeneration::V7x,
    ] {
        let mut b = active_b;
        b.generation = g;
        b.vector_slots[0].opcode = u16::from(TpuOpCode::ExpF32 as u8);
        let dis = disassemble_llo_bundle(&b, 0);
        assert!(assemble_llo_bundle(&dis).is_ok());
    }
    let multi_llo = "// header comment
# hash comment

[c=0000, stall=0] @p0 [pc=0000 gen=v6 stall=0] { S0(op=1, $s0 <- $s0, $s0, imm=0) | V0(AddF32, $v0 <- $v0, $v1, $v0) }
";
    assert_eq!(parse_llo_assembly(multi_llo).map(|v| v.len()).ok(), Some(1));
    assert!(parse_llo_assembly("no_pc_marker_here").is_err());
    assert!(
        extract_slot_bitfield(TpuGeneration::V2, &[0_u8; 2], crate::elf::IsaSlotKind::D0).is_none()
    );
}

#[test]
fn sparsecore_v2_v3_spmem_coo_gather_and_all_19_optimizers_cover_every_branch() {
    assert_eq!(
        SparseCoreSubcoreKind::from_byte(0),
        SparseCoreSubcoreKind::Scs
    );
    assert_eq!(
        SparseCoreSubcoreKind::from_byte(1),
        SparseCoreSubcoreKind::Tac
    );
    assert_eq!(
        SparseCoreSubcoreKind::from_byte(2),
        SparseCoreSubcoreKind::Tec
    );
    for b in 0_u8..=6 {
        let op = SparseCoreOp::from_byte(b);
        assert!(op.is_ok());
        if let Ok(o) = op {
            assert_ne!(o.mnemonic(), "");
        }
    }
    assert!(SparseCoreOp::from_byte(7).is_err());

    // V2 & V3 codecs + error branches
    let v2_inst = SparseCoreV2Instruction {
        subcore: SparseCoreSubcoreKind::Tec,
        op: SparseCoreOp::OptimizerUpdate,
        optimizer_id: 7,
        spmem_offset: 512,
        hbm_page_offset: 3,
        transfer_bytes: 256,
        sync_flag_id: 2,
    };
    let mut v2_raw = SparseCoreV2Codec::encode(&v2_inst);
    assert_eq!(SparseCoreV2Codec::decode(&v2_raw, 0).ok(), Some(v2_inst));
    assert!(SparseCoreV2Codec::disassemble_stream(&v2_raw).is_ok());
    assert!(SparseCoreV2Codec::disassemble_stream(&[]).is_err());
    assert!(SparseCoreV2Codec::decode(&v2_raw[..4], 0).is_err());
    v2_raw[3] ^= 0xFF;
    assert!(SparseCoreV2Codec::decode(&v2_raw, 0).is_err());
    v2_raw[0] = 0x00;
    assert!(SparseCoreV2Codec::decode(&v2_raw, 0).is_err());
    assert!(
        SparseCoreV2Codec::to_program_wrapper_proto(&[v2_inst])
            .tec
            .is_some()
    );

    let v3_inst = SparseCoreV3Instruction {
        subcore: SparseCoreSubcoreKind::Tec,
        op: SparseCoreOp::SpmemGatherDma,
        optimizer_id: 0,
        quant_mode: 1,
        vector_lanes: 16,
        spmem_offset: 1024,
        hbm3e_byte_address: 0x0012_3456_7000,
        sync_flag_id: 5,
    };
    let mut v3_raw = SparseCoreV3Codec::encode(&v3_inst);
    assert_eq!(SparseCoreV3Codec::decode(&v3_raw, 0).ok(), Some(v3_inst));
    assert!(SparseCoreV3Codec::disassemble_stream(&v3_raw).is_ok());
    assert!(SparseCoreV3Codec::disassemble_stream(&[0_u8; 5]).is_err());
    assert!(SparseCoreV3Codec::decode(&v3_raw[..8], 0).is_err());
    v3_raw[3] ^= 0xFF;
    assert!(SparseCoreV3Codec::decode(&v3_raw, 0).is_err());
    v3_raw[0] = 0x00;
    assert!(SparseCoreV3Codec::decode(&v3_raw, 0).is_err());
    assert!(
        SparseCoreV3Codec::to_program_wrapper_proto(&[v3_inst])
            .tec
            .is_some()
    );

    // SparseCoreBackend V2 & V3
    let sc_bundles = vec![
        TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::Gather,
            operand_slots: vec![0],
            output_slot: 1,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        },
        TpuInstructionBundle {
            pc_offset: 1,
            opcode: TpuOpCode::Scatter,
            operand_slots: vec![1],
            output_slot: 2,
            sync_flag_id: 2,
            attrs: InstructionAttrs::default(),
        },
    ];
    let bc_v5 = SparseCoreBackend::new(TpuGeneration::V5p)
        .lower_bundles_to_barnacore_proto("emb_v5", &sc_bundles);
    assert_eq!(bc_v5.sparse_core_version, 2);
    let bc_v7 = SparseCoreBackend::new(TpuGeneration::V7x)
        .lower_bundles_to_barnacore_proto("emb_v7", &sc_bundles);
    assert_eq!(bc_v7.sparse_core_version, 3);
    assert_eq!(
        SparseCoreBackend::new(TpuGeneration::V6e)
            .build_v3_executable_config_proto("cfg6", 1024, 64)
            .executable_name,
        "cfg6"
    );
    assert_eq!(
        SparseCoreBackend::new(TpuGeneration::V7x)
            .build_v3_executable_config_proto("cfg7", 2048, 128)
            .executable_name,
        "cfg7"
    );

    // SPMEM tiler + COO/CSR sorter + Gather/Dedup engine
    let tiler = SpmemScratchpadTiler::new(4096);
    assert!(tiler.plan_tiles(0, 16, 1).is_err());
    assert!(tiler.plan_tiles(10, 2048, 2).is_err());
    let tile_scheds = tiler.plan_tiles(64, 16, 2);
    assert!(tile_scheds.is_ok());
    if let Ok(ts) = tile_scheds {
        assert!(ts.len() >= 2);
    }

    assert!(CooCsrSorter::sort_and_dedup(0, 16, &[]).is_err());
    assert!(
        CooCsrSorter::sort_and_dedup(
            2,
            4,
            &[CooSparseEntry {
                sample_id: 5,
                row_id: 0,
                weight: 1.0
            }]
        )
        .is_err()
    );
    let coo = [
        CooSparseEntry {
            sample_id: 1,
            row_id: 2,
            weight: 2.0,
        },
        CooSparseEntry {
            sample_id: 0,
            row_id: 2,
            weight: 1.0,
        },
        CooSparseEntry {
            sample_id: 0,
            row_id: 1,
            weight: 3.0,
        },
    ];
    let csr_res = CooCsrSorter::sort_and_dedup(3, 8, &coo);
    assert!(csr_res.is_ok());
    if let Ok(csr) = csr_res {
        // Notice sample 2 has 0 entries -> exercises w_sum == 0 and w_sq_sum == 0 in Mean and Sqrtn!
        assert_eq!(csr.unique_row_ids, vec![1, 2]);
        let table = vec![0.0_f32, 0.0, 10.0, 20.0, 30.0, 40.0];
        for comb in [
            EmbeddingCombiner::Sum,
            EmbeddingCombiner::Mean,
            EmbeddingCombiner::Sqrtn,
        ] {
            let acts = EmbeddingGatherDedupEngine::forward_gather_dedup(&table, 2, &csr, comb);
            assert_eq!(acts.len(), 6);
        }
        let coalesced =
            EmbeddingGatherDedupEngine::backward_coalesce_gradients(&[1.0_f32; 6], 2, &csr);
        assert_eq!(coalesced.len(), 4);
    }

    // All 19 optimizers: proto round-trip, microkernel emission (v2 & v3), and numerical updates
    assert!(
        SparseCoreOptimizerConfig::from_proto(&OptimizationParametersProto::default()).is_err()
    );
    let dyn_lr_proto = OptimizationParametersProto {
        learning_rate: Some(LearningRateProto {
            learning_rate: Some(
                crate::proto::generated_prost_schemas::tensorflow::tpu::learning_rate::LearningRate::Dynamic(
                    crate::proto::generated_prost_schemas::tensorflow::tpu::OptimizerDynamicInput::default(),
                ),
            ),
        }),
        parameters: Some(
            crate::proto::generated_prost_schemas::tensorflow::tpu::optimization_parameters::Parameters::StochasticGradientDescent(
                crate::proto::StochasticGradientDescentParametersProto {},
            ),
        ),
        ..OptimizationParametersProto::default()
    };
    assert!(SparseCoreOptimizerConfig::from_proto(&dyn_lr_proto).is_ok());

    let all_optimizers = [
        SparseCoreGradientOptimizer::Sgd,
        SparseCoreGradientOptimizer::Adagrad { epsilon: 1e-6 },
        SparseCoreGradientOptimizer::AdagradMomentum {
            momentum: 0.9,
            use_nesterov: true,
            exponent: 2.0,
            beta2: 0.99,
            epsilon: 1e-6,
        },
        SparseCoreGradientOptimizer::AdagradMomentum {
            momentum: 0.9,
            use_nesterov: false,
            exponent: 0.0,
            beta2: 0.0,
            epsilon: 0.0,
        },
        SparseCoreGradientOptimizer::BoundedAdagrad {
            update_accumulator_first: true,
            max_var_update: 0.5,
            max_accumulator: 10.0,
        },
        SparseCoreGradientOptimizer::BoundedAdagrad {
            update_accumulator_first: false,
            max_var_update: 0.0,
            max_accumulator: 0.0,
        },
        SparseCoreGradientOptimizer::MdlAdagradLight {
            l2: 0.01,
            lr_power: -0.5,
            min_servable_mdl_benefit: 0.01,
            mdl_benefit_rampup_coeff: 1.0,
        },
        SparseCoreGradientOptimizer::MdlAdagradLight {
            l2: 0.01,
            lr_power: 0.0,
            min_servable_mdl_benefit: 100.0,
            mdl_benefit_rampup_coeff: 0.5,
        },
        SparseCoreGradientOptimizer::FrequencyAwareAdagrad {
            l1_regularization_strength: 0.001,
            l2_regularization_strength: 0.01,
            probability_exponent: 0.5,
            max_lr_multiplier: 4.0,
            accumulator_decay: 0.01,
        },
        SparseCoreGradientOptimizer::FrequencyAwareAdagrad {
            l1_regularization_strength: 100.0,
            l2_regularization_strength: 0.01,
            probability_exponent: 0.5,
            max_lr_multiplier: 4.0,
            accumulator_decay: 0.01,
        },
        SparseCoreGradientOptimizer::ProximalAdagrad {
            l1: 0.001,
            l2: 0.01,
        },
        SparseCoreGradientOptimizer::ProximalAdagrad {
            l1: 100.0,
            l2: 0.01,
        },
        SparseCoreGradientOptimizer::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            use_non_lazy_adam: true,
            use_sum_inside_sqrt: true,
        },
        SparseCoreGradientOptimizer::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 0.0,
            use_non_lazy_adam: false,
            use_sum_inside_sqrt: false,
        },
        SparseCoreGradientOptimizer::Ftrl {
            l1: 0.001,
            l2: 0.1,
            lr_power: -0.5,
            beta: 1.0,
            multiply_linear_by_lr: true,
        },
        SparseCoreGradientOptimizer::Ftrl {
            l1: 100.0,
            l2: 0.1,
            lr_power: 0.0,
            beta: 1.0,
            multiply_linear_by_lr: false,
        },
        SparseCoreGradientOptimizer::Lion {
            beta1: 0.9,
            beta2: 0.99,
        },
        SparseCoreGradientOptimizer::Momentum {
            momentum: 0.9,
            use_nesterov: true,
        },
        SparseCoreGradientOptimizer::Momentum {
            momentum: 0.9,
            use_nesterov: false,
        },
        SparseCoreGradientOptimizer::RmsProp {
            rho: 0.9,
            momentum: 0.1,
            epsilon: 0.0,
        },
        SparseCoreGradientOptimizer::CenteredRmsProp {
            rho: 0.9,
            momentum: 0.1,
            epsilon: 0.0,
        },
        SparseCoreGradientOptimizer::OnlineYogi {
            l1: 0.001,
            l2: 0.01,
            beta2: 0.99,
        },
        SparseCoreGradientOptimizer::OnlineYogi {
            l1: 100.0,
            l2: 0.01,
            beta2: 0.99,
        },
        SparseCoreGradientOptimizer::ProximalYogi {
            l1: 0.001,
            l2: 0.01,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 0.0,
        },
        SparseCoreGradientOptimizer::ProximalYogi {
            l1: 100.0,
            l2: 0.01,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-6,
        },
        SparseCoreGradientOptimizer::Assign,
        SparseCoreGradientOptimizer::FrequencyEstimator {
            tau: 10.0,
            max_delta: 5.0,
            outlier_threshold: 0.5,
            weight_exponent: 1.5,
        },
        SparseCoreGradientOptimizer::UserDefinedProgram {
            program: Box::new(HloModuleProto::default()),
        },
        SparseCoreGradientOptimizer::Adadelta {
            rho: 0.95,
            epsilon: 0.0,
        },
    ];

    for (idx, opt) in all_optimizers.into_iter().enumerate() {
        assert!(opt.opcode_id() <= 18);
        assert!(opt.state_slot_count() <= 3);
        let mut cfg = SparseCoreOptimizerConfig::new(opt, 0.1);
        if (idx & 1) == 0 {
            cfg.weight_decay_factor = 0.01;
            cfg.multiply_weight_decay_by_lr = (idx & 2) == 0;
            cfg.gradient_clip = Some((-2.0, 2.0));
            cfg.weight_clip = Some((-10.0, 10.0));
        }
        let pb = cfg.to_proto();
        let parsed = SparseCoreOptimizerConfig::from_proto(&pb);
        assert!(parsed.is_ok());
        if let Ok(re_cfg) = parsed {
            assert_eq!(re_cfg.emit_tec_microkernel_bitstream(2).len(), 64);
            assert_eq!(re_cfg.emit_tec_microkernel_bitstream(3).len(), 96);
            // Pass positive, negative, and zero gradients across 3 elements (covers Lion >0, <0, ==0 & FrequencyEstimator > / <= threshold!)
            let mut w = [1.0_f32, -1.0, 0.0];
            let mut s0 = [0.1_f32, 0.1, 0.0];
            let mut s1 = [0.1_f32, 0.1, 0.0];
            let mut s2 = [0.05_f32, 0.05, 0.0];
            re_cfg.apply_row_update(&mut w, &mut s0, &mut s1, &mut s2, &[1.2, -0.2, 0.0]);
        }
    }
    // Also exercise ClippingLimitsProto with lower: None and upper: None
    let open_clip_proto = OptimizationParametersProto {
        clipping_limits: Some(ClippingLimitsProto {
            lower: None,
            upper: None,
        }),
        parameters: Some(
            crate::proto::generated_prost_schemas::tensorflow::tpu::optimization_parameters::Parameters::Assign(
                crate::proto::AssignParametersProto {},
            ),
        ),
        ..OptimizationParametersProto::default()
    };
    assert!(SparseCoreOptimizerConfig::from_proto(&open_clip_proto).is_ok());
}

#[test]
fn serialization_riegeli_relocations_abi_and_five_phase_compile_cover_all_branches() {
    let mlir_src = b"module @phase6_exec { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }";
    let compiled = CompiledProgram::compile("mlir", mlir_src, b"");
    assert!(compiled.is_ok());
    if let Ok(mut prog) = compiled {
        // Generation-specific IsaProgramProto::isa_programs oneof across all generations
        for tpu_gen in [
            TpuGeneration::V2,
            TpuGeneration::V3,
            TpuGeneration::V4,
            TpuGeneration::V4Lite,
            TpuGeneration::V5e,
            TpuGeneration::V5p,
            TpuGeneration::V6e,
            TpuGeneration::V7x,
        ] {
            let isa_p = prog.build_isa_program_proto_for_generation(tpu_gen);
            assert!(isa_p.isa_programs.is_some());
            let reloc_bs = prog.relocate_generation_bitstream(
                tpu_gen,
                &[DeviceAddress(0x1000), DeviceAddress(0x2000)],
                &[DeviceAddress(0x3000)],
            );
            assert_ne!(reloc_bs, [] as [u8; 0]);
            let width = generation_bundle_byte_width(tpu_gen);
            for chunk in reloc_bs.chunks_exact(width) {
                assert!(decode_bundle_for_generation(tpu_gen, chunk).is_ok());
            }
        }

        // HostCommandEnvelopeProto synthesis for Infeed, Outfeed, HostCallback, and regular op
        prog.instruction_bundles.push(TpuInstructionBundle {
            pc_offset: 10,
            opcode: TpuOpCode::Infeed,
            operand_slots: Vec::new(),
            output_slot: 3,
            sync_flag_id: 4,
            attrs: InstructionAttrs::default(),
        });
        prog.instruction_bundles.push(TpuInstructionBundle {
            pc_offset: 11,
            opcode: TpuOpCode::Outfeed,
            operand_slots: vec![3],
            output_slot: 3,
            sync_flag_id: 5,
            attrs: InstructionAttrs::default(),
        });
        prog.instruction_bundles.push(TpuInstructionBundle {
            pc_offset: 12,
            opcode: TpuOpCode::CustomCall,
            operand_slots: vec![3],
            output_slot: 3,
            sync_flag_id: 6,
            attrs: InstructionAttrs {
                custom_call_target: String::from("PyHostCallback"),
                ..InstructionAttrs::default()
            },
        });
        let hc = prog.build_host_command_envelopes();
        assert_eq!(hc.len(), 3);

        let meta = prog.build_compiler_metadata_proto();
        assert!(meta.trap_metadata.is_some());

        let abi = prog.build_executable_abi_version_proto();
        assert!(CompiledProgram::verify_executable_abi_version(&abi).is_ok());
        assert!(
            CompiledProgram::verify_executable_abi_version(&PjRtExecutableAbiVersionProto {
                platform: 0,
                version: vec![1],
            })
            .is_err()
        );
        assert!(
            CompiledProgram::verify_executable_abi_version(&PjRtExecutableAbiVersionProto {
                platform: 0x5450_5500,
                version: Vec::new(),
            })
            .is_err()
        );

        let reloc_proto = prog.to_relocatable_proto();
        assert_eq!(reloc_proto.programs.len(), 1);

        // Riegeli HighwayHash64 container round-trip + corruption detection
        let mut riegeli_bytes = prog.serialize_riegeli_executable();
        let from_riegeli = CompiledProgram::deserialize_riegeli_executable(&riegeli_bytes);
        assert!(from_riegeli.is_ok());
        let auto_riegeli = CompiledProgram::deserialize(&riegeli_bytes);
        assert!(auto_riegeli.is_ok());
        if let Some(byte52) = riegeli_bytes.get_mut(52) {
            *byte52 ^= 0xFF;
        }
        assert!(CompiledProgram::deserialize_riegeli_executable(&riegeli_bytes).is_err());
        assert!(CompiledProgram::deserialize(&riegeli_bytes).is_err());
    }

    // 5-phase PJRT_PhaseCompile sequential pipeline, inspection, and consumer_phases validation
    let stages_res = CompiledProgram::run_five_phase_pipeline(mlir_src, b"");
    assert!(stages_res.is_ok());
    if let Ok(stages) = stages_res {
        assert_eq!(stages.len(), COMPILATION_PHASE_NAMES.len());
        if let Some(first_stage) = stages.first() {
            let inspected = CompiledProgram::inspect_partial_program(&first_stage.encode_proto());
            assert!(inspected.is_ok());
        }
    }
    assert!(CompiledProgram::inspect_partial_program(&[0xFF, 0xFF, 0xFF]).is_err());
    let empty_p = PjRtPartialProgramProto {
        program: Vec::new(),
        producer_phase: String::from("hlo_verification"),
        ..PjRtPartialProgramProto::default()
    };
    assert!(CompiledProgram::inspect_partial_program(&empty_p.encode_proto()).is_err());
    let bad_inner_p = PjRtPartialProgramProto {
        program: b"invalid_ltpuexec".to_vec(),
        producer_phase: String::from("hlo_verification"),
        ..PjRtPartialProgramProto::default()
    };
    assert!(CompiledProgram::inspect_partial_program(&bad_inner_p.encode_proto()).is_err());

    let restricted_partial = PjRtPartialProgramProto {
        program: mlir_src.to_vec(),
        program_format: String::from("mlir"),
        producer_phase: String::from("hlo_verification"),
        consumer_phases: vec![String::from("tpu_layout_assignment")],
        version: String::from(crate::LIBTPU_PYPI_VERSION),
        program_name: String::from("restricted"),
    };
    assert!(
        CompiledProgram::run_compilation_phase(
            "binary_serialization",
            &restricted_partial.encode_proto(),
            b"",
        )
        .is_err()
    );
}

#[test]
fn phase6_100_percent_branch_closure() {
    use super::passes::LinearScanRegisterAllocator;
    use super::vliw::{
        LloVliwScheduler, ModuloPipelineScheduler, PolyhedralTileSchedule, VliwScoreboard,
    };
    use crate::compiler::roundtrip_generation_bundle;

    // 1. isel.rs: mosaic_ast == None on valid HLO + empty AstFunction (mosaic_ops.is_empty())
    let sel = LloInstructionSelector::new(TpuGeneration::V6e);
    let valid_hlo = HloModuleProto {
        name: String::from("h_only"),
        computations: vec![HloComputationProto {
            name: String::from("main"),
            instructions: vec![HloInstructionProto {
                id: 1,
                name: String::from("a"),
                opcode: String::from("add"),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(sel.select_from_hlo_and_mosaic(&valid_hlo, None).is_ok());
    if let Ok(empty_mosaic) =
        parse_mlir_ast_module("module @m { func.func private @empty_fn(%arg0: tensor<4xf32>) }")
    {
        assert!(
            sel.select_from_hlo_and_mosaic(&valid_hlo, Some(&empty_mosaic))
                .is_ok()
        );
    }

    // 2. regalloc.rs: expired interval eviction (act_end < start_pc -> free_regs.push(phys))
    let mut expiry_ir = LloIr {
        module_name: String::from("expiry"),
        generation: TpuGeneration::V5e,
        next_vreg_id: 4,
        blocks: vec![LloBasicBlock {
            label: String::from("b0"),
            is_loop_body: false,
            trip_count: None,
            ops: vec![
                LloMicroOp {
                    id: 0,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![LloRegOperand::Virtual {
                        id: 0,
                        class: LloRegClass::Vreg,
                    }],
                    uses: Vec::new(),
                    imm_value: 0,
                    latency_cycles: 2,
                    memory_space: 1,
                    hlo_source_name: String::from("op0"),
                },
                LloMicroOp {
                    id: 1,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![LloRegOperand::Virtual {
                        id: 1,
                        class: LloRegClass::Vreg,
                    }],
                    uses: vec![LloRegOperand::Virtual {
                        id: 0,
                        class: LloRegClass::Vreg,
                    }],
                    imm_value: 0,
                    latency_cycles: 2,
                    memory_space: 1,
                    hlo_source_name: String::from("op1"),
                },
                LloMicroOp {
                    id: 2,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![LloRegOperand::Virtual {
                        id: 2,
                        class: LloRegClass::Vreg,
                    }],
                    uses: vec![LloRegOperand::Virtual {
                        id: 1,
                        class: LloRegClass::Vreg,
                    }],
                    imm_value: 0,
                    latency_cycles: 2,
                    memory_space: 1,
                    hlo_source_name: String::from("op2"),
                },
            ],
        }],
    };
    assert!(
        LloRegisterAllocator::new(TpuGeneration::V5e)
            .allocate(&mut expiry_ir)
            .is_ok()
    );

    // 3. packer.rs: Immediate in defs (scoreboard_key == None) + V2 single-vector-slot overflow
    let imm_vec_ir = LloIr {
        module_name: String::from("imm_v2"),
        generation: TpuGeneration::V2,
        next_vreg_id: 4,
        blocks: vec![LloBasicBlock {
            label: String::from("b"),
            is_loop_body: false,
            trip_count: None,
            ops: vec![
                LloMicroOp {
                    id: 0,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![LloRegOperand::Immediate(0)],
                    uses: Vec::new(),
                    imm_value: 0,
                    latency_cycles: 1,
                    memory_space: 1,
                    hlo_source_name: String::from("v_a"),
                },
                LloMicroOp {
                    id: 1,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector1,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![LloRegOperand::Immediate(1)],
                    uses: Vec::new(),
                    imm_value: 0,
                    latency_cycles: 1,
                    memory_space: 1,
                    hlo_source_name: String::from("v_b"),
                },
                LloMicroOp {
                    id: 2,
                    llo_opcode: LloOpcodeProto::OpcodeVectorAddF32,
                    tpu_opcode: TpuOpCode::AddF32,
                    unit: LloFunctionalUnit::Vector0,
                    element_type: ElementType::F32,
                    predicate: None,
                    negate_predicate: false,
                    defs: vec![LloRegOperand::Immediate(2)],
                    uses: Vec::new(),
                    imm_value: 0,
                    latency_cycles: 1,
                    memory_space: 1,
                    hlo_source_name: String::from("v_c"),
                },
            ],
        }],
    };
    assert_eq!(
        LloVliwBundlePacker::new(TpuGeneration::V2)
            .pack_ir(&imm_vec_ir)
            .bundles
            .len(),
        3
    );
    assert_eq!(
        LloVliwBundlePacker::new(TpuGeneration::V6e)
            .pack_ir(&imm_vec_ir)
            .bundles
            .len(),
        2
    );

    // 4. isa/codecs.rs: dst_space == 3, collective dma roundtrip, V7x canonical 64B + unknown gen_raw
    let mut dst_cmem = LloBundle {
        generation: TpuGeneration::V7x,
        bundle_flags: 0x0010,
        ..LloBundle::default()
    };
    dst_cmem.dma_slot.dma_opcode = 14;
    dst_cmem.dma_slot.collective_op = 14;
    dst_cmem.dma_slot.dst_space = 3;
    assert!(
        PufferfishIsaCodec::to_isa_program_proto(&[dst_cmem])
            .bundles
            .first()
            .is_some_and(|b| b.cmem_load.is_some())
    );
    for tpu_gen in [
        TpuGeneration::V2,
        TpuGeneration::V3,
        TpuGeneration::V4,
        TpuGeneration::V5e,
        TpuGeneration::V6e,
        TpuGeneration::V7x,
    ] {
        assert!(roundtrip_generation_bundle(tpu_gen, &dst_cmem).is_ok());
    }
    let iwc_canon = encode_canonical_64b_bundle(&dst_cmem);
    assert!(decode_canonical_64b_bundle(&iwc_canon, 0).is_ok());
    assert!(decode_canonical_64b_bundle(&[0x77; 64], 0).is_err());
    assert!(decode_canonical_64b_bundle(&[], 0).is_err());
    let mut unknown_gen_word = encode_canonical_64b_bundle(&LloBundle::default());
    unknown_gen_word[6] = 200;
    unknown_gen_word[7] = compute_bundle_highway_tag8(&unknown_gen_word);
    assert!(decode_canonical_64b_bundle(&unknown_gen_word, 0).is_ok());
    assert!(JellyfishIsaCodec::disassemble_stream(&[0_u8; 32]).is_err());

    // 5. sparsecore/codecs.rs, optimizers.rs, spmem_coo_gather.rs
    assert_eq!(SparseCoreSubcoreKind::Scs.mnemonic(), "SCS");
    assert_eq!(SparseCoreSubcoreKind::Tac.mnemonic(), "TAC");
    assert_eq!(SparseCoreSubcoreKind::Tec.mnemonic(), "TEC");
    let mut sc_v2_bad_op = [0_u8; 16];
    sc_v2_bad_op[0] = SparseCoreV2Codec::MAGIC;
    sc_v2_bad_op[1] = 7;
    sc_v2_bad_op[3] = compute_sparsecore_tag8(&sc_v2_bad_op);
    assert!(SparseCoreV2Codec::decode(&sc_v2_bad_op, 0).is_err());
    assert!(SparseCoreV2Codec::disassemble_stream(&[]).is_err());
    assert!(SparseCoreV2Codec::disassemble_stream(&[0_u8; 3]).is_err());
    assert!(SparseCoreV2Codec::disassemble_stream(&sc_v2_bad_op).is_err());

    let mut sc_v3_bad_op = [0_u8; 24];
    sc_v3_bad_op[0] = SparseCoreV3Codec::MAGIC;
    sc_v3_bad_op[1] = 7;
    sc_v3_bad_op[3] = compute_sparsecore_tag8(&sc_v3_bad_op);
    assert!(SparseCoreV3Codec::decode(&sc_v3_bad_op, 0).is_err());
    assert!(SparseCoreV3Codec::disassemble_stream(&[]).is_err());
    assert!(SparseCoreV3Codec::disassemble_stream(&[0_u8; 3]).is_err());
    assert!(SparseCoreV3Codec::disassemble_stream(&sc_v3_bad_op).is_err());

    for pos_eps_opt in [
        SparseCoreGradientOptimizer::RmsProp {
            rho: 0.9,
            momentum: 0.1,
            epsilon: 1e-5,
        },
        SparseCoreGradientOptimizer::CenteredRmsProp {
            rho: 0.9,
            momentum: 0.1,
            epsilon: 1e-5,
        },
        SparseCoreGradientOptimizer::Adadelta {
            rho: 0.95,
            epsilon: 1e-5,
        },
        SparseCoreGradientOptimizer::AdagradMomentum {
            momentum: 0.9,
            use_nesterov: false,
            exponent: 2.0,
            beta2: 1.5,
            epsilon: 1e-5,
        },
        SparseCoreGradientOptimizer::Lion {
            beta1: 0.9,
            beta2: 0.99,
        },
    ] {
        let cfg = SparseCoreOptimizerConfig::new(pos_eps_opt, 0.05);
        let parsed = SparseCoreOptimizerConfig::from_proto(&cfg.to_proto());
        assert!(parsed.is_ok());
        if let Ok(c) = parsed {
            let mut w = [1.0_f32, -1.0, 0.0];
            let mut s0 = [0.1_f32, 0.1, 0.0];
            let mut s1 = [0.1_f32, 0.1, 0.0];
            let mut s2 = [0.1_f32, 0.1, 0.0];
            c.apply_row_update(&mut w, &mut s0, &mut s1, &mut s2, &[0.5, -5.0, 0.0]);
        }
    }
    assert!(
        CooCsrSorter::sort_and_dedup(
            2,
            8,
            &[CooSparseEntry {
                sample_id: 0,
                row_id: 99,
                weight: 1.0,
            }],
        )
        .is_err()
    );

    // 6. vliw.rs: extract_slot_bitfield None, parse_llo_assembly Err, VliwScoreboard, ModuloPipelineScheduler, pack_independent_bundles, from_llo_bundle Err
    assert!(
        extract_slot_bitfield(TpuGeneration::V2, &[0_u8; 1], crate::elf::IsaSlotKind::M0).is_none()
    );
    assert!(parse_llo_assembly("[pc=0000 gen=v5 stall=0] { BAD_SLOT }").is_err());
    assert!(assemble_llo_bundle("[pc=0000 gen=v6 stall=0] { S0(s1}").is_err());
    assert_ne!(disassemble_llo_bundle(&LloBundle::default(), 0), "");
    let sb = VliwScoreboard::default();
    assert_eq!(sb.check_vmem_bank_conflict(Some(0), Some(0)), (0, 1));
    assert_eq!(sb.check_vmem_bank_conflict(Some(0), Some(256)), (0, 0));
    assert_eq!(sb.check_vmem_bank_conflict(None, Some(256)), (1, 0));
    assert_eq!(sb.check_vmem_bank_conflict(None, None), (0, 0));

    let mod_sched = ModuloPipelineScheduler::new(TpuGeneration::V6e);
    let loop_bundles = vec![
        TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::DotGeneralF32,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 0,
            attrs: InstructionAttrs::default(),
        },
        TpuInstructionBundle {
            pc_offset: 1,
            opcode: TpuOpCode::ExpF32,
            operand_slots: vec![2],
            output_slot: 3,
            sync_flag_id: 0,
            attrs: InstructionAttrs::default(),
        },
        TpuInstructionBundle {
            pc_offset: 2,
            opcode: TpuOpCode::Outfeed,
            operand_slots: vec![3],
            output_slot: 4,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        },
        TpuInstructionBundle {
            pc_offset: 3,
            opcode: TpuOpCode::AddF32,
            operand_slots: vec![3, 0],
            output_slot: 5,
            sync_flag_id: 0,
            attrs: InstructionAttrs::default(),
        },
    ];
    let (resource_bound, recurrence_bound, mii) =
        mod_sched.compute_minimum_initiation_interval(&loop_bundles);
    assert!(mii >= resource_bound.max(recurrence_bound));
    let barna = mod_sched.lower_sparse_core_to_barnacore_proto("sc_mod", &loop_bundles);
    assert_eq!(barna.executable_name, "sc_mod_barnacore");
    if let Some(first_lb) = loop_bundles.first() {
        let tile_plan = mod_sched.schedule_tile_loop(
            first_lb,
            &PolyhedralTileSchedule {
                batch_tiles: 1,
                m_tiles: 2,
                n_tiles: 2,
                k_tiles: 2,
                ping_vmem_offset: 0x1000,
                pong_vmem_offset: 0x2000,
                prefetch_sync_flag_id: 3,
            },
        );
        assert_eq!(tile_plan.staged_bundles.len(), 4);
        let sp_proto =
            mod_sched.to_software_pipeline_schedule_proto("loop0", &loop_bundles, &tile_plan);
        assert_eq!(sp_proto.stages.len(), 4);
        let bs_proto = mod_sched.to_bundle_schedule_proto("sched0", &tile_plan);
        assert_eq!(bs_proto.scheduled_instructions.len(), 4);
    }

    // Co-issue MXU, XLU, and DMA in LloVliwScheduler::pack_independent_bundles
    let vliw_sched = LloVliwScheduler::new(TpuGeneration::V6e);
    let dma_pair = vec![
        TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::AddF32,
            operand_slots: vec![0, 1],
            output_slot: 8,
            sync_flag_id: 0,
            attrs: InstructionAttrs::default(),
        },
        TpuInstructionBundle {
            pc_offset: 1,
            opcode: TpuOpCode::Outfeed,
            operand_slots: vec![2],
            output_slot: 9,
            sync_flag_id: 4,
            attrs: InstructionAttrs::default(),
        },
    ];
    let packed_co = vliw_sched.pack_independent_bundles(&loop_bundles);
    assert!(!packed_co.len().eq(&0));
    assert_eq!(vliw_sched.pack_independent_bundles(&dma_pair).len(), 1);

    let mut invalid_op_bundle = LloBundle::default();
    invalid_op_bundle.vector_slots[0].opcode = 250;
    assert!(TpuInstructionBundle::from_llo_bundle(&invalid_op_bundle, 0).is_err());

    // 7. passes.rs & serialization.rs remaining methods & error branches
    let mlir_src = b"module @clos { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> %1 = stablehlo.multiply %0, %arg2 : tensor<4xf32> %2 = stablehlo.subtract %1, %arg0 : tensor<4xf32> return %2 : tensor<4xf32> } }";
    if let Ok(mut prog) = CompiledProgram::compile("mlir", mlir_src, b"") {
        let intervals = prog.compute_liveness_intervals();
        let no_spill_alloc = LinearScanRegisterAllocator::new(TpuGeneration::V6e);
        let no_spill_rep = no_spill_alloc.allocate(&intervals, 0x10000);
        let mut base_bundles = vec![LloBundle::default(); prog.instruction_bundles.len().max(1)];
        let unchanged = no_spill_rep.splice_spill_bundles(&base_bundles);
        assert_eq!(unchanged.len(), base_bundles.len());

        let tight_ls = LinearScanRegisterAllocator::with_max_vregs(TpuGeneration::V6e, 2);
        let spill_rep = tight_ls.allocate(&intervals, 0x20000);
        let ra_proto = spill_rep.to_llo_register_allocation_proto(&intervals);
        assert!(!ra_proto.live_intervals.len().eq(&0));
        base_bundles.resize(prog.instruction_bundles.len() + 4, LloBundle::default());
        let spliced = spill_rep.splice_spill_bundles(&base_bundles);
        assert!(spliced.len() >= base_bundles.len());

        prog.instruction_bundles.push(TpuInstructionBundle {
            pc_offset: 9,
            opcode: TpuOpCode::Outfeed,
            operand_slots: vec![0],
            output_slot: 3,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        });
        prog.instruction_bundles.push(TpuInstructionBundle {
            pc_offset: 10,
            opcode: TpuOpCode::CustomCall,
            operand_slots: vec![0],
            output_slot: 3,
            sync_flag_id: 0,
            attrs: InstructionAttrs {
                custom_call_target: String::from("SparseCoreGather"),
                ..InstructionAttrs::default()
            },
        });
        assert_eq!(prog.build_host_command_envelopes().len(), 1);
        let remat_over = prog.pass_hlo_rematerialization(0);
        assert!(remat_over.rematerialized_count >= 1);
        let remat_under = prog.pass_hlo_rematerialization(u64::MAX);
        assert_eq!(remat_under.rematerialized_count, 0);

        let _alias_match = prog.build_input_output_buffer_alias_config_proto();
        if let Ok(sh8) = Shape::new(
            ElementType::F32,
            vec![8],
            None,
            Some(MemoryLayout::default_tpu_tiled(1)),
        ) {
            prog.output_shapes = vec![sh8];
        }
        let alias_mismatch = prog.build_input_output_buffer_alias_config_proto();
        assert_eq!(alias_mismatch.total_aliased_bytes, 0);

        prog.compile_options_bytes = vec![0xFF, 0xFF, 0xFF];
        assert!(prog.to_serialized_executable_proto().is_err());
    }
    assert!(CompiledProgram::run_five_phase_pipeline(b"", b"").is_err());
}
