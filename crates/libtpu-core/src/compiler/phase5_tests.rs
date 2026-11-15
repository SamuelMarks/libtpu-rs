//! Exhaustive unit and branch-coverage tests for Phase 5 XLA-TPU Middle-End passes.

#![allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::float_cmp
)]

use super::middle_end::*;
use crate::error::LibTpuResult;
use crate::proto::{
    AlgebraicSimplifierOptionsProto, ConvolutionDimensionNumbers, DotDimensionNumbers,
    EmitterLearnedCostModelOptionsProto, FrontendAttributes, HloAliasEntryProto,
    HloBufferDonorEntryProto, HloBufferDonorProto, HloComputationProto, HloInputOutputAliasProto,
    HloInstructionProto, HloModuleProto, JellyfishLearnedCostModelModeProto,
    JellyfishMemorySpaceProto, JellyfishMlOutputValidationStrategyProto, LayoutProto, LiteralProto,
    MemoryBoundLoopOptimizerOptionsProto, MsaHloOperandFilterProto, MsaHloPositionMatcherProto,
    MsaKeepInDefaultMemoryActionProto, MsaPinInAlternateMemoryActionProto,
    MsaSortOrderOverrideOptionsProto, MsaSortOrderOverrideProto, MsaSortOrderOverridesProto,
    MsaTensorOverrideProto, MsaTensorOverridesProto, OpShardingProto,
    PreferredPrefetchOverrideOptionsProto, PreferredPrefetchOverrideProto,
    PreferredPrefetchOverridesProto, PrimitiveType, ProgramShapeProto, ShapeProto,
    ShardyAxisRefProto, ShardyDimensionShardingProto, ShardyMeshAxisProto, ShardyMeshProto,
    ShardySubAxisInfoProto, ShardyTensorShardingProto, SlicedPrefetchOptionsProto, Window,
    WindowDimension, msa_sort_order_override_options, msa_tensor_override,
    preferred_prefetch_override_options,
};
use crate::types::TpuGeneration;
use std::collections::BTreeMap;

/// Helper constructing an `F32` tensor [`ShapeProto`] with given dimensions.
///
/// # Arguments
///
/// * `dims` - Tensor dimension extents.
fn f32_shape(dims: &[i64]) -> ShapeProto {
    ShapeProto {
        element_type: PrimitiveType::F32 as i32,
        dimensions: dims.to_vec(),
        is_dynamic_dimension: vec![false; dims.len()],
        ..ShapeProto::default()
    }
}

/// Helper serializing a [`ShardyTensorShardingProto`] into [`FrontendAttributes`].
///
/// # Arguments
///
/// * `sharding` - Shardy tensor sharding descriptor.
fn sdy_front_attrs(sharding: &ShardyTensorShardingProto) -> FrontendAttributes {
    use crate::proto::ProtoSchemaMessage as _;
    let mut fa = FrontendAttributes::default();
    if let Ok(s) = sharding.to_proto_json() {
        fa.map.insert(String::from(SDY_SHARDING_ATTR_KEY), s);
    }
    fa
}

#[test]
fn test_shardy_propagation_priorities_subaxes_closed_and_unreduced() -> LibTpuResult<()> {
    let mesh = ShardyMeshProto {
        name: String::from("mesh2d"),
        axes: vec![
            ShardyMeshAxisProto {
                name: String::from("data"),
                size: 4,
            },
            ShardyMeshAxisProto {
                name: String::from("model"),
                size: 2,
            },
        ],
        device_ids: (0..8).collect(),
    };

    // Sub-axis [pre_size=1, size=2] on "data", priority 0, plus closed dim on rhs
    let lhs_sdy = ShardyTensorShardingProto {
        mesh_name: String::from("mesh2d"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("data"),
                    sub_axis_info: Some(ShardySubAxisInfoProto {
                        pre_size: 1,
                        size: 2,
                    }),
                }],
                is_closed: false,
                priority: 0,
            },
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("model"),
                    sub_axis_info: None,
                }],
                is_closed: true,
                priority: 1,
            },
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };

    let rhs_sdy = ShardyTensorShardingProto {
        mesh_name: String::from("mesh2d"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("model"),
                    sub_axis_info: None,
                }],
                is_closed: true,
                priority: 0,
            },
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("data"),
                    sub_axis_info: Some(ShardySubAxisInfoProto {
                        pre_size: 2,
                        size: 2,
                    }),
                }],
                is_closed: false,
                priority: 2,
            },
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };

    let comp = HloComputationProto {
        id: 1,
        name: String::from("main"),
        root_id: 5,
        instructions: vec![
            HloInstructionProto {
                id: 1,
                name: String::from("p0"),
                opcode: String::from("parameter"),
                parameter_number: 0,
                shape: Some(f32_shape(&[16, 8])),
                frontend_attributes: Some(sdy_front_attrs(&lhs_sdy)),
                ..HloInstructionProto::default()
            },
            HloInstructionProto {
                id: 2,
                name: String::from("p1"),
                opcode: String::from("parameter"),
                parameter_number: 1,
                shape: Some(f32_shape(&[8, 16])),
                frontend_attributes: Some(sdy_front_attrs(&rhs_sdy)),
                ..HloInstructionProto::default()
            },
            HloInstructionProto {
                id: 3,
                name: String::from("dot0"),
                opcode: String::from("dot"),
                shape: Some(f32_shape(&[16, 16])),
                operand_ids: vec![1, 2],
                ..HloInstructionProto::default()
            },
            HloInstructionProto {
                id: 4,
                name: String::from("tr0"),
                opcode: String::from("transpose"),
                shape: Some(f32_shape(&[16, 16])),
                operand_ids: vec![3],
                dimensions: vec![1, 0],
                ..HloInstructionProto::default()
            },
            HloInstructionProto {
                id: 5,
                name: String::from("act0"),
                opcode: String::from("tanh"),
                shape: Some(f32_shape(&[16, 16])),
                operand_ids: vec![4],
                ..HloInstructionProto::default()
            },
        ],
        ..HloComputationProto::default()
    };

    let mut module = HloModuleProto {
        id: 1,
        name: String::from("sdy_mod"),
        entry_computation_id: 1,
        computations: vec![comp],
        ..HloModuleProto::default()
    };

    let report = ShardyPropagationPass::new(mesh).run(&mut module)?;
    assert!(report.priority_tiers_evaluated >= 2);
    assert!(report.propagated_edges >= 2);
    assert!(report.unreduced_instructions >= 1);
    assert_eq!(module.spmd_parameters_shardings.len(), 2);
    assert!(module.spmd_output_sharding.is_some());

    // Also test backward propagation from consumer onto open parameter + OpSharding fallback
    let out_sdy = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("data"),
                    sub_axis_info: None,
                }],
                is_closed: false,
                priority: 0,
            },
            ShardyDimensionShardingProto {
                axes: Vec::new(),
                is_closed: true,
                priority: 0,
            },
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let mut back_mod = HloModuleProto {
        id: 2,
        entry_computation_id: 10,
        computations: vec![HloComputationProto {
            id: 10,
            root_id: 13,
            instructions: vec![
                HloInstructionProto {
                    id: 11,
                    name: String::from("in0"),
                    opcode: String::from("parameter"),
                    parameter_number: 0,
                    shape: Some(f32_shape(&[8, 8])),
                    sharding: Some(OpShardingProto {
                        r#type: 3,
                        tile_assignment_dimensions: vec![1, 4],
                        tile_assignment_devices: vec![0, 1, 2, 3],
                        ..OpShardingProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 12,
                    name: String::from("tr_back"),
                    opcode: String::from("transpose"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![11],
                    dimensions: vec![1, 0],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 13,
                    name: String::from("neg_out"),
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![12],
                    frontend_attributes: Some(sdy_front_attrs(&out_sdy)),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let lin_rep = ShardyPropagationPass::with_linear_mesh(4).run(&mut back_mod)?;
    assert!(lin_rep.iterations >= 1);

    // Error paths: invalid mesh axis, unknown axis, invalid sub-axis, malformed JSON
    let bad_mesh = ShardyMeshProto {
        name: String::from("bad"),
        axes: vec![ShardyMeshAxisProto {
            name: String::new(),
            size: 0,
        }],
        device_ids: Vec::new(),
    };
    assert!(
        ShardyPropagationPass::new(bad_mesh)
            .run(&mut back_mod)
            .is_err()
    );

    let unknown_ax = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![ShardyDimensionShardingProto {
            axes: vec![ShardyAxisRefProto {
                name: String::from("missing_axis"),
                sub_axis_info: None,
            }],
            is_closed: false,
            priority: 0,
        }],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let mut err_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                name: String::from("bad_op"),
                opcode: String::from("parameter"),
                shape: Some(f32_shape(&[4, 4])),
                frontend_attributes: Some(sdy_front_attrs(&unknown_ax)),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        ShardyPropagationPass::with_linear_mesh(4)
            .run(&mut err_mod)
            .is_err()
    );

    let bad_sub = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("data"),
                    sub_axis_info: Some(ShardySubAxisInfoProto {
                        pre_size: 4,
                        size: 4,
                    }),
                }],
                is_closed: false,
                priority: 0,
            },
            ShardyDimensionShardingProto::default(),
            ShardyDimensionShardingProto::default(),
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    if let Some(c) = err_mod.computations.first_mut()
        && let Some(i) = c.instructions.first_mut()
    {
        i.frontend_attributes = Some(sdy_front_attrs(&bad_sub));
    }
    assert!(
        ShardyPropagationPass::with_linear_mesh(4)
            .run(&mut err_mod)
            .is_err()
    );

    if let Some(c) = err_mod.computations.first_mut()
        && let Some(i) = c.instructions.first_mut()
    {
        let mut bad_fa = FrontendAttributes::default();
        bad_fa.map.insert(
            String::from(SDY_SHARDING_ATTR_KEY),
            String::from("{not_json"),
        );
        i.frontend_attributes = Some(bad_fa);
    }
    assert!(
        ShardyPropagationPass::with_linear_mesh(4)
            .run(&mut err_mod)
            .is_err()
    );

    // Empty axes mesh validation bypass
    let empty_mesh = ShardyMeshProto::default();
    let mut empty_mod = HloModuleProto::default();
    assert!(
        ShardyPropagationPass::new(empty_mesh)
            .run(&mut empty_mod)
            .is_ok()
    );

    Ok(())
}

#[test]
fn test_gspmd_partitioner_synthesizes_all_six_collectives() -> LibTpuResult<()> {
    let mesh = ShardyMeshProto {
        name: String::from("mesh"),
        axes: vec![
            ShardyMeshAxisProto {
                name: String::from("data"),
                size: 2,
            },
            ShardyMeshAxisProto {
                name: String::from("model"),
                size: 2,
            },
        ],
        device_ids: vec![0, 1, 2, 3],
    };

    let shard_dim0 = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("data"),
                    sub_axis_info: None,
                }],
                is_closed: true,
                priority: 0,
            },
            ShardyDimensionShardingProto::default(),
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let shard_dim1 = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto::default(),
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("data"),
                    sub_axis_info: None,
                }],
                is_closed: true,
                priority: 0,
            },
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let replicated_sdy = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto::default(),
            ShardyDimensionShardingProto::default(),
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let unreduced_all_reduce = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto::default(),
            ShardyDimensionShardingProto::default(),
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: vec![ShardyAxisRefProto {
            name: String::from("model"),
            sub_axis_info: None,
        }],
    };
    let unreduced_reduce_scatter = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("model"),
                    sub_axis_info: None,
                }],
                is_closed: true,
                priority: 0,
            },
            ShardyDimensionShardingProto::default(),
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: vec![ShardyAxisRefProto {
            name: String::from("model"),
            sub_axis_info: None,
        }],
    };

    let mut halo_fa = sdy_front_attrs(&shard_dim0);
    halo_fa
        .map
        .insert(String::from(SPMD_HALO_SHIFT_ATTR_KEY), String::from("1"));

    let mut module = HloModuleProto {
        id: 1,
        name: String::from("gspmd_all6"),
        entry_computation_id: 1,
        host_program_shape: Some(ProgramShapeProto {
            parameters: vec![f32_shape(&[8, 8]), f32_shape(&[8, 8])],
            result: Some(f32_shape(&[8, 8])),
            parameter_names: vec![String::from("p0"), String::from("p1")],
        }),
        computations: vec![HloComputationProto {
            id: 1,
            name: String::from("main"),
            root_id: 6,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    parameter_number: 0,
                    shape: Some(f32_shape(&[8, 8])),
                    frontend_attributes: Some(halo_fa),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("p1"),
                    opcode: String::from("parameter"),
                    parameter_number: 1,
                    shape: Some(f32_shape(&[8, 8])),
                    frontend_attributes: Some(sdy_front_attrs(&shard_dim0)),
                    ..HloInstructionProto::default()
                },
                // Consumer sharded on dim1 -> triggers AllToAll from p1 (dim0 -> dim1)
                HloInstructionProto {
                    id: 3,
                    name: String::from("reshard_a2a"),
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![2],
                    frontend_attributes: Some(sdy_front_attrs(&shard_dim1)),
                    ..HloInstructionProto::default()
                },
                // Consumer replicated -> triggers AllGather from reshard_a2a (dim1 -> replicated)
                HloInstructionProto {
                    id: 4,
                    name: String::from("gather_ag"),
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![3],
                    frontend_attributes: Some(sdy_front_attrs(&replicated_sdy)),
                    ..HloInstructionProto::default()
                },
                // Unreduced + sharded on model -> triggers ReduceScatter
                HloInstructionProto {
                    id: 5,
                    name: String::from("dot_rs"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![1, 4],
                    frontend_attributes: Some(sdy_front_attrs(&unreduced_reduce_scatter)),
                    ..HloInstructionProto::default()
                },
                // Unreduced + replicated + ragged -> triggers AllReduce + RaggedAllToAll
                HloInstructionProto {
                    id: 6,
                    name: String::from("dot_ar_ragged"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![5, 4],
                    frontend_attributes: Some({
                        let mut fa = sdy_front_attrs(&unreduced_all_reduce);
                        fa.map.insert(
                            String::from(SPMD_RAGGED_RESHARD_ATTR_KEY),
                            String::from("true"),
                        );
                        fa
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };

    let rep = GspmdPartitionerPass::new(mesh, 4).run(&mut module)?;
    assert_eq!(rep.all_reduce_count, 1);
    assert_eq!(rep.reduce_scatter_count, 1);
    assert_eq!(rep.all_gather_count, 1);
    assert_eq!(rep.all_to_all_count, 1);
    assert_eq!(rep.collective_permute_count, 1);
    assert_eq!(rep.ragged_all_to_all_count, 1);

    // Check replica group builder when device_ids length differs or axis missing
    let fallback_groups = build_replica_groups_for_axis(
        &ShardyMeshProto {
            name: String::from("m"),
            axes: vec![ShardyMeshAxisProto {
                name: String::from("x"),
                size: 2,
            }],
            device_ids: Vec::new(),
        },
        "x",
        2,
    );
    assert_eq!(fallback_groups.len(), 1);

    Ok(())
}

#[test]
fn test_dot_decomposer_and_convolution_legalizer_passes() -> LibTpuResult<()> {
    // 4D batched dot_general + canonical 2D dot + rank-3 batched canonical dot
    let mut dot_mod = HloModuleProto {
        id: 1,
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 5,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("lhs4d"),
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 3, 4, 5])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("rhs4d"),
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 5, 6, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("dot4d"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[2, 3, 6])),
                    operand_ids: vec![1, 2],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_batch_dimensions: vec![0],
                        rhs_batch_dimensions: vec![0],
                        lhs_contracting_dimensions: vec![2, 3],
                        rhs_contracting_dimensions: vec![3, 1],
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    name: String::from("w2d"),
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[18, 7])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    name: String::from("dot_unbatched"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[2, 7])),
                    operand_ids: vec![3, 4],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_batch_dimensions: Vec::new(),
                        rhs_batch_dimensions: Vec::new(),
                        lhs_contracting_dimensions: vec![1, 2],
                        rhs_contracting_dimensions: vec![0],
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };

    let dot_rep = DotDecomposerPass.run(&mut dot_mod)?;
    assert_eq!(dot_rep.decomposed_dots, 2);
    assert!(dot_rep.inserted_wrappers >= 4);

    // Running DotDecomposerPass a second time sees already canonicalized dots!
    let second_rep = DotDecomposerPass.run(&mut dot_mod)?;
    assert_eq!(second_rep.decomposed_dots, 0);

    // Invalid DotDimensionNumbers error path
    let mut bad_dot_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 1],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_contracting_dimensions: vec![9],
                        rhs_contracting_dimensions: vec![0],
                        lhs_batch_dimensions: Vec::new(),
                        rhs_batch_dimensions: Vec::new(),
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(DotDecomposerPass.run(&mut bad_dot_mod).is_err());

    // ConvolutionLegalizerPass with padding, dilation, and spatial sharding halo exchange
    let mut conv_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("img"),
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 8, 8, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("kern"),
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[3, 3, 4, 16])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("conv0"),
                    opcode: String::from("convolution"),
                    shape: Some(f32_shape(&[2, 8, 8, 16])),
                    operand_ids: vec![1, 2],
                    convolution_dimension_numbers: Some(ConvolutionDimensionNumbers {
                        input_batch_dimension: 0,
                        input_feature_dimension: 3,
                        input_spatial_dimensions: vec![1, 2],
                        kernel_input_feature_dimension: 2,
                        kernel_output_feature_dimension: 3,
                        kernel_spatial_dimensions: vec![0, 1],
                        output_batch_dimension: 0,
                        output_feature_dimension: 3,
                        output_spatial_dimensions: vec![1, 2],
                    }),
                    window: Some(Window {
                        dimensions: vec![
                            WindowDimension {
                                size: 3,
                                stride: 1,
                                padding_low: 1,
                                padding_high: 1,
                                window_dilation: 1,
                                base_dilation: 2,
                                window_reversal: false,
                            },
                            WindowDimension {
                                size: 3,
                                stride: 1,
                                padding_low: 1,
                                padding_high: 1,
                                window_dilation: 1,
                                base_dilation: 1,
                                window_reversal: false,
                            },
                        ],
                    }),
                    sharding: Some(OpShardingProto {
                        r#type: 3,
                        tile_assignment_dimensions: vec![1, 2, 1, 1],
                        tile_assignment_devices: vec![0, 1],
                        ..OpShardingProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };

    let conv_rep = ConvolutionLegalizerPass::new(2).run(&mut conv_mod)?;
    assert_eq!(conv_rep.legalized_convolutions, 1);
    assert_eq!(conv_rep.inserted_pads, 1);
    assert_eq!(conv_rep.inserted_halo_exchanges, 1);

    // Error path: convolution with <2 operands
    let mut bad_conv = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                name: String::from("bad_conv"),
                opcode: String::from("convolution"),
                operand_ids: vec![1],
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(ConvolutionLegalizerPass::new(1).run(&mut bad_conv).is_err());

    Ok(())
}

#[test]
fn test_simplifiers_licm_cse_dce_and_async_collective_latency_scheduler() {
    // 1. ConstantFoldingPass (f32 unary/binary + s32 binary)
    let s32_shape = ShapeProto {
        element_type: PrimitiveType::S32 as i32,
        dimensions: vec![2],
        is_dynamic_dimension: vec![false],
        ..ShapeProto::default()
    };
    let mut cf_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 19,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[2])),
                    literal: Some(LiteralProto {
                        f32s: vec![4.0, 9.0],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[2])),
                    literal: Some(LiteralProto {
                        f32s: vec![2.0, 3.0],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("sqrt"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![5],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("log"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![6],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 8,
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![7],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 9,
                    opcode: String::from("floor"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![7],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 10,
                    opcode: String::from("ceil"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![8],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 11,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 12,
                    opcode: String::from("subtract"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![11, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 13,
                    opcode: String::from("multiply"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![12, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 14,
                    opcode: String::from("divide"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![13, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 15,
                    opcode: String::from("maximum"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![14, 9],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 16,
                    opcode: String::from("minimum"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![15, 10],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 17,
                    opcode: String::from("constant"),
                    shape: Some(s32_shape.clone()),
                    literal: Some(LiteralProto {
                        s32s: vec![5, 7],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 18,
                    opcode: String::from("constant"),
                    shape: Some(s32_shape.clone()),
                    literal: Some(LiteralProto {
                        s32s: vec![2, 3],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 19,
                    opcode: String::from("add"),
                    shape: Some(s32_shape.clone()),
                    operand_ids: vec![17, 18],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 20,
                    opcode: String::from("subtract"),
                    shape: Some(s32_shape.clone()),
                    operand_ids: vec![19, 18],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 21,
                    opcode: String::from("multiply"),
                    shape: Some(s32_shape.clone()),
                    operand_ids: vec![20, 18],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 22,
                    opcode: String::from("maximum"),
                    shape: Some(s32_shape.clone()),
                    operand_ids: vec![21, 17],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 23,
                    opcode: String::from("minimum"),
                    shape: Some(s32_shape),
                    operand_ids: vec![22, 18],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(ConstantFoldingPass.run(&mut cf_mod), 19);

    // 2. AlgebraicSimplifierPass (identities, zero/one rules, inverse cancellation, 1-elem dot, reshape)
    let mut alg_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 18,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[4])),
                    literal: Some(LiteralProto {
                        f32s: vec![0.0; 4],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[4])),
                    literal: Some(LiteralProto {
                        s32s: vec![1; 4],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![2, 4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("subtract"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![5, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("multiply"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![6, 3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 8,
                    opcode: String::from("multiply"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![3, 7],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 9,
                    opcode: String::from("divide"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![8, 3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 10,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![9],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 11,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![10],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 12,
                    opcode: String::from("log"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![11],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 13,
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![12],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 14,
                    opcode: String::from("slice"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![13],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 15,
                    opcode: String::from("reshape"),
                    shape: Some(f32_shape(&[2, 2])),
                    operand_ids: vec![14],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 16,
                    opcode: String::from("reshape"),
                    shape: Some(f32_shape(&[1, 4])),
                    operand_ids: vec![15],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 17,
                    opcode: String::from("subtract"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![1, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 18,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[1, 1])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 19,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[1, 1])),
                    operand_ids: vec![18, 18],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let alg_opts = AlgebraicSimplifierOptionsProto {
        run_to_fixed_point: Some(false),
        enable_dot_to_multiply_rewrite: Some(true),
        ..AlgebraicSimplifierOptionsProto::default()
    };
    assert!(AlgebraicSimplifierPass::new(alg_opts).run(&mut alg_mod) >= 8);

    // 3. BroadcastSimplifierPass + ReshapeMoverPass + TransposeFoldingPass + TupleSimplifierPass
    let mut graph_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 14,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("broadcast"),
                    shape: Some(f32_shape(&[2, 4])),
                    operand_ids: vec![1],
                    dimensions: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("broadcast"),
                    shape: Some(f32_shape(&[3, 2, 4])),
                    operand_ids: vec![2],
                    dimensions: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("broadcast"),
                    shape: Some(f32_shape(&[3, 2, 4])),
                    operand_ids: vec![3],
                    dimensions: vec![0, 1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("reshape"),
                    shape: Some(f32_shape(&[6, 4])),
                    operand_ids: vec![4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("reshape"),
                    shape: Some(f32_shape(&[6, 4])),
                    operand_ids: vec![4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[6, 4])),
                    operand_ids: vec![5, 6],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 8,
                    opcode: String::from("transpose"),
                    shape: Some(f32_shape(&[4, 6])),
                    operand_ids: vec![7],
                    dimensions: vec![1, 0],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 9,
                    opcode: String::from("transpose"),
                    shape: Some(f32_shape(&[6, 4])),
                    operand_ids: vec![8],
                    dimensions: vec![1, 0],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 10,
                    opcode: String::from("transpose"),
                    shape: Some(f32_shape(&[4, 6])),
                    operand_ids: vec![9],
                    dimensions: vec![1, 0],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 11,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![10, 9],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 12,
                    opcode: String::from("tuple"),
                    shape: Some(ShapeProto {
                        element_type: PrimitiveType::Tuple as i32,
                        tuple_shapes: vec![f32_shape(&[4, 4]), f32_shape(&[6, 4])],
                        ..ShapeProto::default()
                    }),
                    operand_ids: vec![11, 9],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 13,
                    opcode: String::from("get-tuple-element"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![12],
                    tuple_index: 0,
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 14,
                    opcode: String::from("get-tuple-element"),
                    shape: Some(f32_shape(&[6, 4])),
                    operand_ids: vec![12],
                    tuple_index: 1,
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(BroadcastSimplifierPass.run(&mut graph_mod), 2);
    assert_eq!(ReshapeMoverPass.run(&mut graph_mod), 1);
    assert!(TransposeFoldingPass.run(&mut graph_mod) >= 2);
    assert_eq!(TupleSimplifierPass.run(&mut graph_mod), 2);

    // 4. WhileLoopInvariantCodeMotionPass + HloCsePass + HloDcePass
    let tup_shape = ShapeProto {
        element_type: PrimitiveType::Tuple as i32,
        tuple_shapes: vec![f32_shape(&[4]), f32_shape(&[4])],
        ..ShapeProto::default()
    };
    let mut licm_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![
            HloComputationProto {
                id: 1,
                root_id: 4,
                instructions: vec![
                    HloInstructionProto {
                        id: 1,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[4])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 2,
                        opcode: String::from("tuple"),
                        shape: Some(tup_shape.clone()),
                        operand_ids: vec![1, 1],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 3,
                        opcode: String::from("abs"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![1],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 30,
                        opcode: String::from("abs"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![1],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 4,
                        opcode: String::from("while"),
                        shape: Some(tup_shape.clone()),
                        operand_ids: vec![2],
                        called_computation_ids: vec![2, 3],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            // Body computation: slot 1 is passed unchanged -> invariant!
            HloComputationProto {
                id: 2,
                root_id: 16,
                instructions: vec![
                    HloInstructionProto {
                        id: 10,
                        opcode: String::from("parameter"),
                        shape: Some(tup_shape.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 11,
                        opcode: String::from("get-tuple-element"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![10],
                        tuple_index: 0,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 12,
                        opcode: String::from("get-tuple-element"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![10],
                        tuple_index: 1,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 13,
                        opcode: String::from("constant"),
                        shape: Some(f32_shape(&[4])),
                        literal: Some(LiteralProto {
                            f32s: vec![2.0; 4],
                            ..LiteralProto::default()
                        }),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 14,
                        name: String::from("inv_mul"),
                        opcode: String::from("multiply"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![12, 13],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 15,
                        opcode: String::from("add"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![11, 14],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 16,
                        opcode: String::from("tuple"),
                        shape: Some(tup_shape.clone()),
                        operand_ids: vec![15, 12],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            HloComputationProto {
                id: 3,
                root_id: 20,
                instructions: vec![HloInstructionProto {
                    id: 20,
                    opcode: String::from("parameter"),
                    shape: Some(tup_shape),
                    ..HloInstructionProto::default()
                }],
                ..HloComputationProto::default()
            },
            // Dead unreachable computation
            HloComputationProto {
                id: 99,
                root_id: 99,
                instructions: vec![HloInstructionProto {
                    id: 99,
                    opcode: String::from("constant"),
                    ..HloInstructionProto::default()
                }],
                ..HloComputationProto::default()
            },
        ],
        ..HloModuleProto::default()
    };
    assert_eq!(WhileLoopInvariantCodeMotionPass.run(&mut licm_mod), 1);
    assert!(HloCsePass.run(&mut licm_mod) >= 1);
    assert!(HloDcePass.run(&mut licm_mod) >= 2);
    assert_eq!(licm_mod.computations.len(), 3);

    // 5. AsyncCollectiveFusionPass + LatencyHidingSchedulerPass
    let mut async_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 6,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[64, 64])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[64, 64])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("ar0"),
                    opcode: String::from("all-reduce"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    name: String::from("ar1"),
                    opcode: String::from("all-reduce"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    name: String::from("indep_dot"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![2, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    name: String::from("sum_end"),
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![3, 5],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let ac_rep = AsyncCollectiveFusionPass::new(1024 * 1024).run(&mut async_mod);
    assert_eq!(ac_rep.converted_pairs, 2);
    let lhs_rep = LatencyHidingSchedulerPass.run(&mut async_mod);
    assert!(lhs_rep.overlapped_compute_ops >= 1);
    assert!(lhs_rep.hidden_collective_bytes > 0);
}

#[test]
fn test_layout_mxu_tiling_fusion_and_rematerialization_passes() -> LibTpuResult<()> {
    // Test sub-32-bit packing tiles (BF16=16b, F8E4M3FN=8b, S4=4b, F64=64b, C128=128b) + (2,128)/(4,128)/(8,128)
    let bf16_2x128 = ShapeProto {
        element_type: PrimitiveType::Bf16 as i32,
        dimensions: vec![2, 129],
        is_dynamic_dimension: vec![false, false],
        layout: Some(Box::new(LayoutProto {
            minor_to_major: vec![0, 1], // non-canonical -> triggers XLU transpose relayout!
            ..LayoutProto::default()
        })),
        ..ShapeProto::default()
    };
    let fp8_4x128 = ShapeProto {
        element_type: PrimitiveType::F8e4m3fn as i32,
        dimensions: vec![4, 128],
        is_dynamic_dimension: vec![false, false],
        ..ShapeProto::default()
    };
    let int4_8x128 = ShapeProto {
        element_type: PrimitiveType::S4 as i32,
        dimensions: vec![8, 128],
        is_dynamic_dimension: vec![false, false],
        ..ShapeProto::default()
    };
    let mut lay_mod = HloModuleProto {
        entry_computation_id: 1,
        host_program_shape: Some(ProgramShapeProto {
            parameters: vec![bf16_2x128.clone(), fp8_4x128.clone()],
            result: Some(int4_8x128.clone()),
            parameter_names: vec![String::from("a"), String::from("b")],
        }),
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("a"),
                    opcode: String::from("parameter"),
                    constrain_layout: true,
                    shape: Some(bf16_2x128),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("b"),
                    opcode: String::from("parameter"),
                    shape: Some(fp8_4x128),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("dot_pack"),
                    opcode: String::from("dot"),
                    shape: Some(int4_8x128),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };

    let lay_rep = TpuLayoutAssignmentPass::new(true, true).run(&mut lay_mod)?;
    assert_eq!(lay_rep.inserted_transposes, 1);
    assert_eq!(lay_rep.inserted_pads, 1);
    assert_eq!(lay_rep.inserted_bitcast_packs, 2);
    assert_eq!(primitive_type_bit_width(PrimitiveType::F64 as i32), 64);
    assert_eq!(primitive_type_bit_width(PrimitiveType::C128 as i32), 128);

    // Invalid constrain_layout error path
    let mut bad_lay = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                opcode: String::from("parameter"),
                constrain_layout: true,
                shape: Some(ShapeProto {
                    element_type: PrimitiveType::F32 as i32,
                    dimensions: vec![4, 4],
                    layout: Some(Box::new(LayoutProto {
                        minor_to_major: vec![0, 0],
                        ..LayoutProto::default()
                    })),
                    ..ShapeProto::default()
                }),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        TpuLayoutAssignmentPass::default()
            .run(&mut bad_lay)
            .is_err()
    );

    // MxuTilingPass across V2 (128, 1 MXU), V3 (128, 2 MXUs), V5p (128, 4 MXUs), V6e/V7x (256, 4 MXUs)
    for (tpu_gen, expected_dim, expected_mxus) in [
        (TpuGeneration::V2, 128, 1),
        (TpuGeneration::V3, 128, 2),
        (TpuGeneration::V5p, 128, 4),
        (TpuGeneration::V6e, 256, 4),
        (TpuGeneration::V7x, 256, 4),
    ] {
        let mut tile_mod = HloModuleProto {
            computations: vec![HloComputationProto {
                id: 1,
                instructions: vec![
                    HloInstructionProto {
                        id: 1,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[512, 512])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 2,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[512, 512])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 3,
                        opcode: String::from("dot"),
                        shape: Some(f32_shape(&[512, 512])),
                        operand_ids: vec![1, 2],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            }],
            ..HloModuleProto::default()
        };
        let mxu_rep = MxuTilingPass::new(tpu_gen).run(&mut tile_mod);
        assert_eq!(mxu_rep.systolic_dim, expected_dim);
        assert_eq!(mxu_rep.mxus_per_core, expected_mxus);
        assert_eq!(mxu_rep.schedules.len(), 1);
        assert!(
            mxu_rep
                .schedules
                .first()
                .is_some_and(|s| s.accumulator_chaining)
        );
    }

    // Producer-Consumer Fusion + Budget rejection + Multi-Output Fusion
    let mut fus_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 5,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[64, 64])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[64, 64])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![3, 4],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    // First test tight budget rejection
    let rej_rep = TpuProducerConsumerFusionPass::new(128, 1).run(&mut fus_mod);
    assert_eq!(rej_rep.rejected_for_budget, 1);
    // Now fuse with V6e hardware budget
    let pc_rep =
        TpuProducerConsumerFusionPass::for_generation(TpuGeneration::V6e).run(&mut fus_mod);
    assert_eq!(pc_rep.fused_producers, 2);
    assert_eq!(pc_rep.created_fusion_computations, 1);

    // MultiOutputFusionPass on sibling ops sharing parameter 1
    let mut mof_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 4,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[32, 32])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[32, 32])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[32, 32])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[32, 32])),
                    operand_ids: vec![2, 3],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let mof_rep = MultiOutputFusionPass::new(1024 * 1024).run(&mut mof_mod);
    assert_eq!(mof_rep.multi_output_fusions, 1);

    // HloRematerializationPass shrinking peak memory across long live range
    let mut remat_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            name: String::from("remat_entry"),
            root_id: 6,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[256, 256])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("cheap_exp"),
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("mid1"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![1, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    name: String::from("mid2"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![3, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    name: String::from("mid3"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![4, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    name: String::from("late_add"),
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![5, 2],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let remat_summary = HloRematerializationPass::new(512 * 1024).run(&mut remat_mod);
    assert!(remat_summary.rematerialized_count >= 1);
    assert!(remat_summary.peak_memory_after_bytes <= remat_summary.peak_memory_before_bytes);

    Ok(())
}

#[test]
fn test_msa_double_buffered_prefetch_heap_simulator_donation_and_cost_model() -> LibTpuResult<()> {
    for ms in [
        TpuMsaMemorySpace::Hbm,
        TpuMsaMemorySpace::Vmem,
        TpuMsaMemorySpace::Smem,
        TpuMsaMemorySpace::Cmem,
    ] {
        assert_ne!(
            ms.to_jellyfish_proto(),
            JellyfishMemorySpaceProto::MemorySpaceNone
        );
        assert!(ms.alignment_bytes() >= 64);
    }

    // Build TPU v4 module with scalar constant (SMEM=2), multi-use tensor (CMEM=3),
    // intermediate tensor (VMEM=1), HBM parameters (HBM=0), ping/pong prefetched dot, and buffer_donor!
    let mut v4_mod = HloModuleProto {
        id: 1,
        name: String::from("v4_msa_mod"),
        entry_computation_id: 1,
        buffer_donor: Some(HloBufferDonorProto {
            entries: vec![HloBufferDonorEntryProto {
                parameter_number: 0,
                parameter_shape_index: Vec::new(),
            }],
        }),
        computations: vec![HloComputationProto {
            id: 1,
            name: String::from("main"),
            root_id: 7,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    parameter_number: 0,
                    shape: Some(f32_shape(&[128, 128])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("p1"),
                    opcode: String::from("parameter"),
                    parameter_number: 1,
                    shape: Some(f32_shape(&[128, 128])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("scalar_c"),
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[])),
                    literal: Some(LiteralProto {
                        f32s: vec![1.0],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    name: String::from("cmem_shared"),
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[128, 128])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    name: String::from("vmem_tmp"),
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[128, 128])),
                    operand_ids: vec![4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    name: String::from("dot_ab"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[128, 128])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 7,
                    name: String::from("out_add"),
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[128, 128])),
                    operand_ids: vec![6, 4],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };

    let mut msa = MemorySpaceAssignmentPass::for_generation(TpuGeneration::V4, 256);
    msa.expanded_scoped_vmem_mode = 2; // Enabled
    msa.sliced_prefetch_options = SlicedPrefetchOptionsProto {
        max_slices: 2,
        min_bytes: 1024,
        fail_on_non_alignment_boundary_slice_proposal: false,
        all_slice_time_permutations_threshold: 4,
        preferred_slice_size: 4 * 1024 * 1024,
    };
    msa.loop_optimizer_options = MemoryBoundLoopOptimizerOptionsProto {
        enabled: Some(true),
        desired_copy_ratio: Some(0.75),
        allow_unsatisfied_fully_pipelined_prefetch: Some(true),
        min_num_iterations: Some(2.0),
    };
    msa.sort_order_overrides = MsaSortOrderOverridesProto {
        overrides: vec![MsaSortOrderOverrideProto {
            hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                instruction_name_regex: Some(String::from("vmem_tmp")),
                size_gte: Some(256),
                size_lte: Some(10 * 1024 * 1024),
                ..MsaHloPositionMatcherProto::default()
            }),
            override_options: Some(MsaSortOrderOverrideOptionsProto {
                options: Some(msa_sort_order_override_options::Options::AssignFirst(true)),
            }),
            apply_to_cross_program_prefetches: Some(false),
        }],
    };
    msa.tensor_overrides = MsaTensorOverridesProto {
        overrides: vec![
            MsaTensorOverrideProto {
                hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                    instruction_name_regex: Some(String::from("vmem_tmp")),
                    ..MsaHloPositionMatcherProto::default()
                }),
                hlo_operand_filter: None,
                action: Some(msa_tensor_override::Action::PinInAlternateMemory(
                    MsaPinInAlternateMemoryActionProto {},
                )),
            },
            MsaTensorOverrideProto {
                hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                    instruction_name_regex: Some(String::from("p1")),
                    ..MsaHloPositionMatcherProto::default()
                }),
                hlo_operand_filter: None,
                action: Some(msa_tensor_override::Action::KeepInDefaultMemory(
                    MsaKeepInDefaultMemoryActionProto {},
                )),
            },
        ],
    };
    msa.preferred_prefetch_overrides = PreferredPrefetchOverridesProto {
        fail_on_unsatisfied_override: Some(true),
        overrides: vec![PreferredPrefetchOverrideProto {
            hlo_operand_filter: Some(MsaHloOperandFilterProto {
                instruction_name_regex: Some(String::from("dot_ab")),
                operand_number: Some(0),
                ..MsaHloOperandFilterProto::default()
            }),
            override_options: Some(PreferredPrefetchOverrideOptionsProto {
                options: Some(preferred_prefetch_override_options::Options::PrefetchEagerness(0.8)),
                strict_timing: Some(true),
            }),
            fail_on_unsatisfied_override: Some(true),
        }],
    };

    let msa_rep = msa.run(&mut v4_mod)?;
    assert!(msa_rep.hbm_count >= 2);
    assert!(msa_rep.vmem_count >= 1);
    assert_eq!(msa_rep.smem_count, 1);
    assert_eq!(msa_rep.cmem_count, 1);
    assert_eq!(msa_rep.scheduled_prefetches, 2);
    assert_eq!(msa_rep.ping_buffers, 1);
    assert_eq!(msa_rep.pong_buffers, 1);

    // Run GlobalDecreasingSizeBestFitHeapSimulator and verify donation + serialized BufferAssignmentProto
    let heap_res = GlobalDecreasingSizeBestFitHeapSimulator::new(8192).run(&mut v4_mod);
    assert!(heap_res.alias_config.total_aliased_bytes > 0);
    assert_eq!(heap_res.buffer_assignment.buffer_aliases.len(), 1);
    assert_ne!(
        heap_res
            .compiled_memory_stats
            .serialized_buffer_assignment
            .len(),
        0
    );
    assert_ne!(
        heap_res
            .memory_usage_report
            .memory_space_allocation_entries
            .len(),
        0
    );

    // Also test explicit HloInputOutputAliasProto (kind=2 must_alias) + spacetime reuse across 3 sequential VMEM temps
    let mut seq_mod = HloModuleProto {
        name: String::from("seq_heap"),
        entry_computation_id: 1,
        input_output_alias: Some(HloInputOutputAliasProto {
            entries: vec![HloAliasEntryProto {
                output_shape_index: Vec::new(),
                parameter_number: 0,
                parameter_shape_index: Vec::new(),
                kind: 2,
            }],
        }),
        computations: vec![HloComputationProto {
            id: 1,
            name: String::from("main"),
            root_id: 5,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    parameter_number: 0,
                    shape: Some(f32_shape(&[64, 64])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("t1"),
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("t2"),
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    name: String::from("t3"),
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    name: String::from("out"),
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[64, 64])),
                    operand_ids: vec![4],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = MemorySpaceAssignmentPass::for_generation(TpuGeneration::V5e, 0).run(&mut seq_mod)?;
    let seq_heap = GlobalDecreasingSizeBestFitHeapSimulator::new(4096).run(&mut seq_mod);
    assert!(
        seq_heap
            .alias_config
            .alias_entries
            .first()
            .is_some_and(|e| e.must_alias)
    );

    // MSA OOM error paths: scoped_vmem_limit_kib > total_vmem & pinned tensor > remaining_vmem & strict override miss
    let oom_scoped = MemorySpaceAssignmentPass::for_generation(TpuGeneration::V2, 64 * 1024);
    assert!(oom_scoped.run(&mut seq_mod).is_err());

    let mut pin_oom = MemorySpaceAssignmentPass::for_generation(TpuGeneration::V2, 15 * 1024 + 512);
    pin_oom.tensor_overrides = MsaTensorOverridesProto {
        overrides: vec![MsaTensorOverrideProto {
            hlo_position_matcher: None,
            hlo_operand_filter: None,
            action: Some(msa_tensor_override::Action::PinInAlternateMemory(
                MsaPinInAlternateMemoryActionProto {},
            )),
        }],
    };
    let mut big_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 2,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[1024, 1024])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[1024, 1024])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(pin_oom.run(&mut big_mod).is_err());

    let mut strict_miss = MemorySpaceAssignmentPass::for_generation(TpuGeneration::V6e, 0);
    strict_miss.preferred_prefetch_overrides = PreferredPrefetchOverridesProto {
        fail_on_unsatisfied_override: Some(true),
        overrides: vec![PreferredPrefetchOverrideProto {
            hlo_operand_filter: Some(MsaHloOperandFilterProto {
                instruction_name_regex: Some(String::from("nonexistent_op_xyz")),
                ..MsaHloOperandFilterProto::default()
            }),
            override_options: None,
            fail_on_unsatisfied_override: None,
        }],
    };
    assert!(strict_miss.run(&mut big_mod).is_err());

    // MiddleEndCostModelEvaluator across all generations, modes, and validation strategies
    for tpu_gen in [
        TpuGeneration::V2,
        TpuGeneration::V3,
        TpuGeneration::V4Lite,
        TpuGeneration::V4,
        TpuGeneration::V5e,
        TpuGeneration::V5p,
        TpuGeneration::V6e,
        TpuGeneration::V7x,
    ] {
        for mode in [
            JellyfishLearnedCostModelModeProto::Invalid,
            JellyfishLearnedCostModelModeProto::OnlyMlPrediction,
            JellyfishLearnedCostModelModeProto::OnlyDb,
            JellyfishLearnedCostModelModeProto::DbWithFallbackToMlPrediction,
            JellyfishLearnedCostModelModeProto::OnlyDataCollection,
        ] {
            for strat in [
                JellyfishMlOutputValidationStrategyProto::None,
                JellyfishMlOutputValidationStrategyProto::NeverTrust,
                JellyfishMlOutputValidationStrategyProto::AlwaysTrust,
                JellyfishMlOutputValidationStrategyProto::NoNegativeCycles,
            ] {
                let mut eval = MiddleEndCostModelEvaluator::new(
                    tpu_gen,
                    EmitterLearnedCostModelOptionsProto {
                        cost_model_mode: mode as i32,
                        ml_output_validation_strategy: strat as i32,
                        ..EmitterLearnedCostModelOptionsProto::default()
                    },
                );
                let mut db = BTreeMap::new();
                db.insert(String::from("v4_msa_mod"), 1234.0);
                eval.profile_db_ns = db;
                let cost = eval.evaluate_module(&v4_mod);
                assert!(cost.computation_stats.flop_count > 0.0);
                assert!(cost.computation_stats.transcendental_count > 0.0);
                assert!(cost.estimated_time_ns >= 0.0);
            }
        }
    }

    Ok(())
}

#[test]
fn test_end_to_end_xla_tpu_middle_end_pipeline() -> LibTpuResult<()> {
    let mut module = HloModuleProto {
        id: 1,
        name: String::from("e2e_middle_end"),
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            name: String::from("main"),
            root_id: 4,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("x"),
                    opcode: String::from("parameter"),
                    parameter_number: 0,
                    shape: Some(f32_shape(&[128, 128])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("w"),
                    opcode: String::from("parameter"),
                    parameter_number: 1,
                    shape: Some(f32_shape(&[128, 128])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("mm"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[128, 128])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    name: String::from("act"),
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[128, 128])),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };

    let pipeline = XlaTpuMiddleEndPipeline::for_generation(TpuGeneration::V6e, 2);
    let artifacts = pipeline.run(&mut module)?;
    assert_eq!(artifacts.mxu_tiling_report.systolic_dim, 256);
    assert_ne!(
        artifacts
            .heap_simulation
            .buffer_assignment
            .buffer_allocations
            .len(),
        0
    );
    assert!(artifacts.cost_report.estimated_time_ns > 0.0);
    Ok(())
}

#[test]
fn test_phase5_exhaustive_edge_cases_and_branch_matrix() -> LibTpuResult<()> {
    // 1. Empty HloModuleProto handling across all passes
    let mut empty_mod = HloModuleProto::default();
    assert_eq!(
        TpuProducerConsumerFusionPass::for_generation(TpuGeneration::V2)
            .run(&mut empty_mod)
            .fused_producers,
        0
    );
    assert_eq!(
        TpuProducerConsumerFusionPass::for_generation(TpuGeneration::V4)
            .run(&mut empty_mod)
            .fused_producers,
        0
    );
    assert_eq!(
        TpuProducerConsumerFusionPass::for_generation(TpuGeneration::V7x)
            .run(&mut empty_mod)
            .fused_producers,
        0
    );
    assert_eq!(
        MultiOutputFusionPass::default()
            .run(&mut empty_mod)
            .multi_output_fusions,
        0
    );
    assert_eq!(
        HloRematerializationPass::new(1024)
            .run(&mut empty_mod)
            .rematerialized_count,
        0
    );
    assert_eq!(
        GlobalDecreasingSizeBestFitHeapSimulator::default()
            .run(&mut empty_mod)
            .buffer_assignment
            .logical_buffers
            .len(),
        0
    );
    assert_eq!(HloDcePass.run(&mut empty_mod), 0);

    // 2. Shardy: overlapping vs non-overlapping sub-axes, multi-axis open dim merge, unreduced propagation through elementwise,
    //    dangling operand IDs, OpSharding factor without exact size match, single-partition GSPMD, root collective_permute
    let mesh2 = ShardyMeshProto {
        name: String::from("m"),
        axes: vec![ShardyMeshAxisProto {
            name: String::from("a"),
            size: 8,
        }],
        device_ids: (0..8).collect(),
    };
    let sdy_sub1 = ShardyTensorShardingProto {
        mesh_name: String::from("m"),
        dim_shardings: vec![ShardyDimensionShardingProto {
            axes: vec![ShardyAxisRefProto {
                name: String::from("a"),
                sub_axis_info: Some(ShardySubAxisInfoProto {
                    pre_size: 1,
                    size: 2,
                }),
            }],
            is_closed: false,
            priority: 0,
        }],
        replicated_axes: Vec::new(),
        unreduced_axes: vec![ShardyAxisRefProto {
            name: String::from("a"),
            sub_axis_info: None,
        }],
    };
    let sdy_sub2 = ShardyTensorShardingProto {
        mesh_name: String::from("m"),
        dim_shardings: vec![ShardyDimensionShardingProto {
            axes: vec![ShardyAxisRefProto {
                name: String::from("a"),
                sub_axis_info: Some(ShardySubAxisInfoProto {
                    pre_size: 2,
                    size: 2,
                }),
            }],
            is_closed: false,
            priority: 0,
        }],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let mut halo_root_fa = sdy_front_attrs(&sdy_sub2);
    halo_root_fa
        .map
        .insert(String::from(SPMD_HALO_SHIFT_ATTR_KEY), String::from("1"));

    let mut sdy_edge_mod = HloModuleProto {
        entry_computation_id: 999, // fallback to first computation
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 999, // fallback to last instruction
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[16])),
                    frontend_attributes: Some(sdy_front_attrs(&sdy_sub1)),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[16])),
                    sharding: Some(OpShardingProto {
                        r#type: 3,
                        tile_assignment_dimensions: vec![2], // != 8 -> fallback to first mesh axis
                        tile_assignment_devices: vec![0, 1],
                        ..OpShardingProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[16])),
                    operand_ids: vec![1, 2, 777], // 777 tests missing operand ID continue
                    frontend_attributes: Some(halo_root_fa),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = ShardyPropagationPass::new(mesh2.clone()).run(&mut sdy_edge_mod)?;
    let _ = GspmdPartitionerPass::new(mesh2.clone(), 1).run(&mut sdy_edge_mod)?;
    let g_rep = GspmdPartitionerPass::new(mesh2, 8).run(&mut sdy_edge_mod)?;
    assert!(g_rep.collective_permute_count >= 1);

    // Sub-axis size <= 0 error branch
    let zero_sub = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![ShardyDimensionShardingProto {
            axes: vec![ShardyAxisRefProto {
                name: String::from("data"),
                sub_axis_info: Some(ShardySubAxisInfoProto {
                    pre_size: 1,
                    size: 0,
                }),
            }],
            is_closed: false,
            priority: 0,
        }],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let mut zero_sub_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                opcode: String::from("parameter"),
                shape: Some(f32_shape(&[4])),
                frontend_attributes: Some(sdy_front_attrs(&zero_sub)),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        ShardyPropagationPass::with_linear_mesh(4)
            .run(&mut zero_sub_mod)
            .is_err()
    );

    // Batched dot forward propagation in Shardy
    let batch_dot_sdy = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("data"),
                    sub_axis_info: None,
                }],
                is_closed: false,
                priority: 0,
            },
            ShardyDimensionShardingProto::default(),
            ShardyDimensionShardingProto::default(),
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let mut bdot_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 8, 8])),
                    frontend_attributes: Some(sdy_front_attrs(&batch_dot_sdy)),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 8, 8])),
                    frontend_attributes: Some(sdy_front_attrs(&batch_dot_sdy)),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[4, 8, 8])),
                    operand_ids: vec![1, 2],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_batch_dimensions: vec![0],
                        rhs_batch_dimensions: vec![0],
                        lhs_contracting_dimensions: vec![2],
                        rhs_contracting_dimensions: vec![1],
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        ShardyPropagationPass::with_linear_mesh(4)
            .run(&mut bdot_mod)
            .is_ok()
    );

    // 3. Legalize & Simplify remaining branches:
    //    - DotDecomposer output reshape when rank-4 non-contracting dims > 1, canonical dot without dnums, rhs invalid dnum
    //    - ConvolutionLegalizer without convolution_dimension_numbers (calls default_conv_dnums!) and without window
    //    - AlgebraicSimplifier: log(exp(x)), 0*x, x*0, non-const literal, identity reshape, dot_rewrite=false, unknown op
    //    - ConstantFolding: unknown unary/binary opcodes
    //    - TupleSimplifier: tuple(gte(t, 0), gte(t, 1)) -> t
    //    - LICM: self-recursive body skip, body without invariant candidates, body without parameter
    //    - CSE with s32 literal + DCE with custom-call side effect & control_predecessor_ids
    //    - AsyncCollectiveFusionPass over all 6 collective opcodes + diverse element types (Pred, BF16, F64, C128, Tuple)
    let mut dot4_out_reshape = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 4,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 3, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 5, 6])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("dot"),
                    shape: Some(ShapeProto {
                        element_type: 0,
                        dimensions: vec![4, 4],
                        ..ShapeProto::default()
                    }),
                    operand_ids: vec![1, 2],
                    dot_dimension_numbers: None,
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[2, 3, 5, 6])),
                    operand_ids: vec![1, 2],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_batch_dimensions: Vec::new(),
                        rhs_batch_dimensions: Vec::new(),
                        lhs_contracting_dimensions: vec![2],
                        rhs_contracting_dimensions: vec![0],
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let d4_rep = DotDecomposerPass.run(&mut dot4_out_reshape)?;
    assert!(d4_rep.decomposed_dots >= 1);

    // Canonical dot with None dot_dimension_numbers + RHS out-of-bounds error
    let mut canon_none_dnums = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 1],
                    dot_dimension_numbers: None,
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(
        DotDecomposerPass
            .run(&mut canon_none_dnums)?
            .decomposed_dots,
        0
    );
    if let Some(c) = canon_none_dnums.computations.first_mut()
        && let Some(d) = c.instructions.get_mut(1)
    {
        d.dot_dimension_numbers = Some(DotDimensionNumbers {
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![7],
            lhs_batch_dimensions: Vec::new(),
            rhs_batch_dimensions: Vec::new(),
        });
    }
    assert!(DotDecomposerPass.run(&mut canon_none_dnums).is_err());

    // Convolution without dnums & without window
    let mut simple_conv = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[1, 4, 4, 2])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 2, 2, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("convolution"),
                    shape: None,
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(
        ConvolutionLegalizerPass::new(1)
            .run(&mut simple_conv)?
            .legalized_convolutions,
        1
    );

    // AlgebraicSimplifier, ConstantFolding unknown ops, TransposeFolding on RHS of dot, TupleSimplifier repack
    let tup2 = ShapeProto {
        element_type: PrimitiveType::Tuple as i32,
        tuple_shapes: vec![
            ShapeProto {
                element_type: PrimitiveType::Pred as i32,
                dimensions: vec![4],
                ..ShapeProto::default()
            },
            ShapeProto {
                element_type: PrimitiveType::C128 as i32,
                dimensions: vec![2],
                ..ShapeProto::default()
            },
        ],
        ..ShapeProto::default()
    };
    let mut misc_simp = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 17,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[4, 4])),
                    literal: None,
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[4, 4])),
                    literal: Some(LiteralProto::default()),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[4, 4])),
                    literal: Some(LiteralProto {
                        f32s: vec![0.0; 16],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("multiply"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![4, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("multiply"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("multiply"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 8,
                    opcode: String::from("subtract"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 9,
                    opcode: String::from("divide"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 10,
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 11,
                    opcode: String::from("log"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![10],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 12,
                    opcode: String::from("reshape"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![11],
                    ..HloInstructionProto::default()
                },
                // Transpose on RHS of dot
                HloInstructionProto {
                    id: 13,
                    opcode: String::from("transpose"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1],
                    dimensions: vec![1, 0],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 14,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 13],
                    ..HloInstructionProto::default()
                },
                // Tuple repack: tuple(gte(p_tup, 0), gte(p_tup, 1)) -> p_tup
                HloInstructionProto {
                    id: 15,
                    opcode: String::from("parameter"),
                    parameter_number: 1,
                    shape: Some(tup2.clone()),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 16,
                    opcode: String::from("get-tuple-element"),
                    shape: tup2.tuple_shapes.first().cloned(),
                    operand_ids: vec![15],
                    tuple_index: 0,
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 160,
                    opcode: String::from("get-tuple-element"),
                    shape: tup2.tuple_shapes.get(1).cloned(),
                    operand_ids: vec![15],
                    tuple_index: 1,
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 17,
                    opcode: String::from("tuple"),
                    shape: Some(tup2),
                    operand_ids: vec![16, 160],
                    ..HloInstructionProto::default()
                },
                // ConstantFolding unsupported unary/binary opcodes
                HloInstructionProto {
                    id: 18,
                    opcode: String::from("sine"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 19,
                    opcode: String::from("atan2"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![4, 4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 20,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[2])),
                    literal: Some(LiteralProto {
                        s32s: vec![10, 20],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 21,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[2])),
                    literal: Some(LiteralProto {
                        s32s: vec![10, 20],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 22,
                    opcode: String::from("divide"),
                    shape: Some(f32_shape(&[2])),
                    operand_ids: vec![20, 21],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 23,
                    opcode: String::from("custom-call"),
                    custom_call_has_side_effect: true,
                    control_predecessor_ids: vec![14],
                    shape: Some(f32_shape(&[4, 4])),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = ConstantFoldingPass.run(&mut misc_simp);
    let _ = AlgebraicSimplifierPass::new(AlgebraicSimplifierOptionsProto {
        run_to_fixed_point: Some(true),
        enable_dot_to_multiply_rewrite: Some(false),
        ..AlgebraicSimplifierOptionsProto::default()
    })
    .run(&mut misc_simp);
    assert!(TransposeFoldingPass.run(&mut misc_simp) >= 1);
    assert!(TupleSimplifierPass.run(&mut misc_simp) >= 1);
    assert!(HloCsePass.run(&mut misc_simp) >= 1);
    let _ = HloDcePass.run(&mut misc_simp);

    // ReshapeMoverPass where reshape is root and where operand shapes mismatch
    let mut rm_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 5,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 3])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[3, 2])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("reshape"),
                    shape: Some(f32_shape(&[6])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("reshape"),
                    shape: Some(f32_shape(&[6])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[6])),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[6])),
                    operand_ids: vec![3, 4],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(ReshapeMoverPass.run(&mut rm_mod), 1);

    // BroadcastSimplifier with empty dimensions identity + missing operand
    let mut bc_edge = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("broadcast"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: Vec::new(),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("broadcast"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![1],
                    dimensions: Vec::new(),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("broadcast"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![999],
                    dimensions: Vec::new(),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(BroadcastSimplifierPass.run(&mut bc_edge), 1);

    // LICM edge cases: self-call body, body without parameter, body with 0 hoistable candidates
    let mut licm_edge = HloModuleProto {
        computations: vec![
            HloComputationProto {
                id: 1,
                root_id: 2,
                instructions: vec![
                    HloInstructionProto {
                        id: 1,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[2])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 2,
                        opcode: String::from("while"),
                        operand_ids: vec![1],
                        called_computation_ids: vec![1],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 3,
                        opcode: String::from("while"),
                        operand_ids: vec![1],
                        called_computation_ids: vec![2],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            HloComputationProto {
                id: 2,
                root_id: 10,
                instructions: vec![HloInstructionProto {
                    id: 10,
                    opcode: String::from("constant"),
                    ..HloInstructionProto::default()
                }],
                ..HloComputationProto::default()
            },
        ],
        ..HloModuleProto::default()
    };
    assert_eq!(WhileLoopInvariantCodeMotionPass.run(&mut licm_edge), 0);

    // AsyncCollectiveFusionPass over all 6 collective opcodes + diverse element types & tuple shape
    let mut all_colls = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 7,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(ShapeProto {
                        element_type: PrimitiveType::U8 as i32,
                        dimensions: vec![8],
                        ..ShapeProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("all-gather"),
                    shape: Some(ShapeProto {
                        element_type: PrimitiveType::Bf16 as i32,
                        dimensions: vec![8],
                        ..ShapeProto::default()
                    }),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("reduce-scatter"),
                    shape: Some(ShapeProto {
                        element_type: PrimitiveType::F64 as i32,
                        dimensions: vec![8],
                        ..ShapeProto::default()
                    }),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("all-to-all"),
                    shape: Some(ShapeProto {
                        element_type: PrimitiveType::C128 as i32,
                        dimensions: vec![8],
                        ..ShapeProto::default()
                    }),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("collective-permute"),
                    shape: Some(ShapeProto {
                        element_type: PrimitiveType::Tuple as i32,
                        tuple_shapes: vec![f32_shape(&[4])],
                        ..ShapeProto::default()
                    }),
                    operand_ids: vec![4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("ragged-all-to-all"),
                    shape: Some(f32_shape(&[8])),
                    operand_ids: vec![5],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("all-reduce"),
                    shape: Some(f32_shape(&[8])),
                    operand_ids: vec![6],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(
        AsyncCollectiveFusionPass::default()
            .run(&mut all_colls)
            .converted_pairs,
        6
    );

    // 4. Layout & Fusion edge cases:
    //    - minor_to_major length mismatch error
    //    - tuple ShapeProto & rank-1 ShapeProto in assign_shape_tpu_layout & shape_bytes_256_aligned
    //    - kLoop producer-consumer fusion + chained root_remap + VREG budget rejection
    //    - MultiOutputFusion where sibling is root
    let mut bad_len_lay = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                opcode: String::from("parameter"),
                constrain_layout: true,
                shape: Some(ShapeProto {
                    element_type: PrimitiveType::F32 as i32,
                    dimensions: vec![4, 4],
                    layout: Some(Box::new(LayoutProto {
                        minor_to_major: vec![0],
                        ..LayoutProto::default()
                    })),
                    ..ShapeProto::default()
                }),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        TpuLayoutAssignmentPass::default()
            .run(&mut bad_len_lay)
            .is_err()
    );

    let mut tup_and_1d = ShapeProto {
        element_type: PrimitiveType::Tuple as i32,
        tuple_shapes: vec![f32_shape(&[128]), f32_shape(&[])],
        ..ShapeProto::default()
    };
    assign_shape_tpu_layout(&mut tup_and_1d, None);
    assert!(shape_bytes_256_aligned(&tup_and_1d) >= 512);

    // kLoop fusion + chained fusion + VREG rejection (huge VMEM budget, 0 VREG budget)
    let mut kloop_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 4,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[16, 16])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![3, 1],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        TpuProducerConsumerFusionPass::new(1024 * 1024, 0)
            .run(&mut kloop_mod)
            .rejected_for_budget
            >= 1
    );
    let kl_rep = TpuProducerConsumerFusionPass::new(1024 * 1024, 32).run(&mut kloop_mod);
    assert!(kl_rep.created_fusion_computations >= 1);

    // MultiOutputFusion where sibling B is root + transitive mid dependency check
    let mut mof_root = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[16, 16])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(
        MultiOutputFusionPass::default()
            .run(&mut mof_root)
            .multi_output_fusions,
        1
    );

    // 5. MSA & HeapSimulator remaining branches:
    //    - V7x capacities, non-strict prefetch override skip, AssignLast & AssignValue &AssignFirst(false) sort overrides
    //    -BeforeInstruction prefetch option, loop_optimizer_options.enabled = Some(false)
    //    - instruction_regex in position matcher & operand filter, size_gte/size_lte false branches
    //    - SMEM non-constant temp buffer (color=2), HloScheduleProto sequence in HeapSimulator, best-fit middle gap, non-matching buffer_donor
    let mut v7x_msa = MemorySpaceAssignmentPass::for_generation(TpuGeneration::V7x, 0);
    assert_eq!(
        v7x_msa.hardware_scratchpad_capacities().0,
        128 * 1024 * 1024
    );
    v7x_msa.loop_optimizer_options.enabled = Some(false);
    v7x_msa.preferred_prefetch_overrides = PreferredPrefetchOverridesProto {
        fail_on_unsatisfied_override: Some(false),
        overrides: vec![PreferredPrefetchOverrideProto {
            hlo_operand_filter: Some(MsaHloOperandFilterProto {
                instruction_regex: Some(String::from("dot")),
                operand_number: Some(99), // out of bounds -> false branch!
                ..MsaHloOperandFilterProto::default()
            }),
            override_options: None,
            fail_on_unsatisfied_override: Some(false),
        }],
    };
    v7x_msa.sort_order_overrides = MsaSortOrderOverridesProto {
        overrides: vec![
            MsaSortOrderOverrideProto {
                hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                    instruction_regex: Some(String::from("negate")),
                    size_gte: Some(999_999_999), // triggers min_ok = false
                    ..MsaHloPositionMatcherProto::default()
                }),
                override_options: Some(MsaSortOrderOverrideOptionsProto {
                    options: Some(msa_sort_order_override_options::Options::AssignLast(true)),
                }),
                apply_to_cross_program_prefetches: None,
            },
            MsaSortOrderOverrideProto {
                hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                    instruction_regex: Some(String::from("negate")),
                    size_lte: Some(1), // triggers max_ok = false
                    ..MsaHloPositionMatcherProto::default()
                }),
                override_options: Some(MsaSortOrderOverrideOptionsProto {
                    options: Some(msa_sort_order_override_options::Options::AssignLast(false)),
                }),
                apply_to_cross_program_prefetches: None,
            },
            MsaSortOrderOverrideProto {
                hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                    instruction_regex: Some(String::from("negate")),
                    ..MsaHloPositionMatcherProto::default()
                }),
                override_options: Some(MsaSortOrderOverrideOptionsProto {
                    options: Some(msa_sort_order_override_options::Options::AssignLast(true)),
                }),
                apply_to_cross_program_prefetches: None,
            },
            MsaSortOrderOverrideProto {
                hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                    instruction_regex: Some(String::from("abs")),
                    ..MsaHloPositionMatcherProto::default()
                }),
                override_options: Some(MsaSortOrderOverrideOptionsProto {
                    options: Some(msa_sort_order_override_options::Options::AssignValue(42)),
                }),
                apply_to_cross_program_prefetches: None,
            },
            MsaSortOrderOverrideProto {
                hlo_position_matcher: Some(MsaHloPositionMatcherProto {
                    instruction_regex: Some(String::from("tanh")),
                    ..MsaHloPositionMatcherProto::default()
                }),
                override_options: Some(MsaSortOrderOverrideOptionsProto {
                    options: Some(msa_sort_order_override_options::Options::AssignFirst(false)),
                }),
                apply_to_cross_program_prefetches: None,
            },
        ],
    };
    v7x_msa.tensor_overrides = MsaTensorOverridesProto {
        overrides: vec![MsaTensorOverrideProto {
            hlo_position_matcher: None,
            hlo_operand_filter: Some(MsaHloOperandFilterProto {
                instruction_regex: Some(String::from("tanh")),
                ..MsaHloOperandFilterProto::default()
            }),
            action: None,
        }],
    };

    let mut v7_mod = HloModuleProto {
        entry_computation_id: 1,
        buffer_donor: Some(HloBufferDonorProto {
            entries: vec![HloBufferDonorEntryProto {
                parameter_number: 99, // non-matching donor -> exercises None return!
                parameter_shape_index: Vec::new(),
            }],
        }),
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 5,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    parameter_number: 0,
                    shape: Some(f32_shape(&[4096, 4096])),
                    ..HloInstructionProto::default()
                },
                // Huge intermediate > vmem_free/2 -> placed in HBM!
                HloInstructionProto {
                    id: 2,
                    name: String::from("big_hbm_neg"),
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[4096, 4096])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    name: String::from("abs_sm"),
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    name: String::from("tanh_sm"),
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    name: String::from("root_dot"),
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![3, 4],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = v7x_msa.run(&mut v7_mod)?;

    // Also test BeforeInstruction prefetch option + SMEM colored non-const instruction in HeapSimulator & CostModel
    v7x_msa.preferred_prefetch_overrides = PreferredPrefetchOverridesProto {
        fail_on_unsatisfied_override: Some(false),
        overrides: vec![PreferredPrefetchOverrideProto {
            hlo_operand_filter: None,
            override_options: Some(PreferredPrefetchOverrideOptionsProto {
                options: Some(
                    preferred_prefetch_override_options::Options::BeforeInstruction(
                        MsaHloPositionMatcherProto {
                            instruction_name_regex: Some(String::from("root_dot")),
                            ..MsaHloPositionMatcherProto::default()
                        },
                    ),
                ),
                strict_timing: None,
            }),
            fail_on_unsatisfied_override: None,
        }],
    };
    let _ = v7x_msa.run(&mut v7_mod)?;

    // Schedule + SMEM scratch + best-fit middle gap in HeapSimulator
    let smem_shape = ShapeProto {
        element_type: PrimitiveType::F32 as i32,
        dimensions: vec![16],
        layout: Some(Box::new(LayoutProto {
            minor_to_major: vec![0],
            memory_space: 2,
            ..LayoutProto::default()
        })),
        ..ShapeProto::default()
    };
    if let Some(c) = v7_mod.computations.first_mut() {
        c.instructions.insert(
            1,
            HloInstructionProto {
                id: 90,
                name: String::from("smem_iota"),
                opcode: String::from("iota"),
                shape: Some(smem_shape),
                ..HloInstructionProto::default()
            },
        );
    }
    let _ = LatencyHidingSchedulerPass.run(&mut v7_mod);
    let hs = GlobalDecreasingSizeBestFitHeapSimulator::new(1024).run(&mut v7_mod);
    assert_ne!(hs.buffer_assignment.buffer_allocations.len(), 0);
    let cost = MiddleEndCostModelEvaluator::new(
        TpuGeneration::V7x,
        EmitterLearnedCostModelOptionsProto::default(),
    )
    .evaluate_module(&v7_mod);
    assert!(cost.smem_bytes_accessed > 0);

    Ok(())
}

#[test]
fn test_phase5_100_percent_branch_closure() -> LibTpuResult<()> {
    // --- A. spmd.rs branch closure ---
    // 1. GspmdPartitionerPass on empty module + invalid mesh error + malformed JSON error
    let mut g_empty = HloModuleProto::default();
    assert_eq!(
        GspmdPartitionerPass::new(ShardyMeshProto::default(), 2)
            .run(&mut g_empty)?
            .partitioned_instructions,
        0
    );
    let bad_mesh = ShardyMeshProto {
        name: String::from("bad"),
        axes: vec![ShardyMeshAxisProto {
            name: String::from("x"),
            size: -1,
        }],
        device_ids: Vec::new(),
    };
    assert!(
        GspmdPartitionerPass::new(bad_mesh, 2)
            .run(&mut g_empty)
            .is_err()
    );
    let mut bad_json_fa = FrontendAttributes::default();
    bad_json_fa
        .map
        .insert(String::from(SDY_SHARDING_ATTR_KEY), String::from("invalid"));
    let mut g_bad_json = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                opcode: String::from("parameter"),
                frontend_attributes: Some(bad_json_fa),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        GspmdPartitionerPass::new(ShardyMeshProto::default(), 2)
            .run(&mut g_bad_json)
            .is_err()
    );

    // 2. GspmdPartitionerPass: halo_shift on exact root_id AND ragged_reshard on non-root instruction
    let mut rag_nonroot_fa = FrontendAttributes::default();
    rag_nonroot_fa.map.insert(
        String::from(SPMD_RAGGED_RESHARD_ATTR_KEY),
        String::from("1"),
    );
    let mut halo_exact_root_fa = FrontendAttributes::default();
    halo_exact_root_fa
        .map
        .insert(String::from(SPMD_HALO_SHIFT_ATTR_KEY), String::from("1"));
    let mut g_root_cases = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 2,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4])),
                    frontend_attributes: Some(rag_nonroot_fa),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![1],
                    frontend_attributes: Some(halo_exact_root_fa),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let gr_rep = GspmdPartitionerPass::new(ShardyMeshProto::default(), 2).run(&mut g_root_cases)?;
    assert_eq!(gr_rep.collective_permute_count, 1);
    assert_eq!(gr_rep.ragged_all_to_all_count, 1);

    // 3. ShardyPropagationPass: empty mesh with instructions + OpSharding type=3 with empty axis_sizes,
    //    sub.pre_size == 0 error, dot & transpose with 0 operands, transpose with 2 operands & negative dim,
    //    compatible_axes empty, dst_ds.is_closed, axes_overlap lhs.name != rhs.name, shared_contract false & not_recorded false
    let mut empty_mesh_with_inst = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 4])),
                    sharding: Some(OpShardingProto {
                        r#type: 3,
                        tile_assignment_dimensions: vec![2, 1],
                        tile_assignment_devices: vec![0, 1],
                        ..OpShardingProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: Vec::new(),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("transpose"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: Vec::new(),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        ShardyPropagationPass::new(ShardyMeshProto::default())
            .run(&mut empty_mesh_with_inst)
            .is_ok()
    );

    let pre_zero_sdy = ShardyTensorShardingProto {
        mesh_name: String::from("mesh"),
        dim_shardings: vec![ShardyDimensionShardingProto {
            axes: vec![ShardyAxisRefProto {
                name: String::from("data"),
                sub_axis_info: Some(ShardySubAxisInfoProto {
                    pre_size: 0,
                    size: 2,
                }),
            }],
            is_closed: false,
            priority: 0,
        }],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let mut pre_zero_mod = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                opcode: String::from("parameter"),
                shape: Some(f32_shape(&[4])),
                frontend_attributes: Some(sdy_front_attrs(&pre_zero_sdy)),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        ShardyPropagationPass::with_linear_mesh(4)
            .run(&mut pre_zero_mod)
            .is_err()
    );

    let mesh_xy = ShardyMeshProto {
        name: String::from("xy"),
        axes: vec![
            ShardyMeshAxisProto {
                name: String::from("x"),
                size: 2,
            },
            ShardyMeshAxisProto {
                name: String::from("y"),
                size: 2,
            },
        ],
        device_ids: vec![0, 1, 2, 3],
    };
    let sdy_x0_y1 = ShardyTensorShardingProto {
        mesh_name: String::from("xy"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("x"),
                    sub_axis_info: None,
                }],
                is_closed: false,
                priority: 0,
            },
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("y"),
                    sub_axis_info: None,
                }],
                is_closed: false,
                priority: 0,
            },
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: Vec::new(),
    };
    let sdy_closed_x1 = ShardyTensorShardingProto {
        mesh_name: String::from("xy"),
        dim_shardings: vec![
            ShardyDimensionShardingProto {
                axes: Vec::new(),
                is_closed: true, // closed -> dst_ds.is_closed true branch!
                priority: 0,
            },
            ShardyDimensionShardingProto {
                axes: vec![ShardyAxisRefProto {
                    name: String::from("x"),
                    sub_axis_info: None,
                }],
                is_closed: false,
                priority: 0,
            },
        ],
        replicated_axes: Vec::new(),
        unreduced_axes: vec![ShardyAxisRefProto {
            name: String::from("x"),
            sub_axis_info: None,
        }],
    };
    let mut sdy_complex = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 4,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 4])),
                    frontend_attributes: Some(sdy_front_attrs(&sdy_x0_y1)),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 4])),
                    frontend_attributes: Some(sdy_front_attrs(&sdy_closed_x1)),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![1, 2],
                    frontend_attributes: Some(sdy_front_attrs(&sdy_closed_x1)), // already has unreduced "x" + mismatched contract!
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("transpose"),
                    shape: Some(f32_shape(&[4, 4])),
                    operand_ids: vec![3, 1], // op_pos == 1 exercises consumer.opcode == "transpose" && op_pos == 0 false!
                    dimensions: vec![-1, 0], // -1 exercises usize::try_from(src_dim) Err!
                    frontend_attributes: Some(sdy_front_attrs(&sdy_x0_y1)),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        ShardyPropagationPass::new(mesh_xy)
            .run(&mut sdy_complex)
            .is_ok()
    );

    // --- B. memory_space_assignment.rs branch closure ---
    // 1. Strict prefetch override with filter=None, smem_limit_bytes=Some(0), sort override with options=None,
    //    tuple instruction in MSA, dot at idx=1 (target_pos == idx), dot with 0 operands in CostModel,
    //    empty computation in HeapSimulator, and 2-gap best-fit slack comparison!
    let mut msa_edge = MemorySpaceAssignmentPass::for_generation(TpuGeneration::V5e, 0);
    msa_edge.smem_limit_bytes = Some(0);
    msa_edge.preferred_prefetch_overrides = PreferredPrefetchOverridesProto {
        fail_on_unsatisfied_override: Some(true),
        overrides: vec![PreferredPrefetchOverrideProto {
            hlo_operand_filter: None,
            override_options: None,
            fail_on_unsatisfied_override: Some(true),
        }],
    };
    msa_edge.sort_order_overrides = MsaSortOrderOverridesProto {
        overrides: vec![MsaSortOrderOverrideProto {
            hlo_position_matcher: None,
            override_options: None,
            apply_to_cross_program_prefetches: None,
        }],
    };

    let mut msa_edge_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 5,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[32, 32])),
                    ..HloInstructionProto::default()
                },
                // Dot right after parameter -> copy-start inserted right after param so target_pos == idx!
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[32, 32])),
                    operand_ids: vec![1, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("constant"),
                    shape: Some(f32_shape(&[])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("tuple"),
                    shape: Some(ShapeProto {
                        element_type: PrimitiveType::Tuple as i32,
                        tuple_shapes: vec![f32_shape(&[32, 32])],
                        ..ShapeProto::default()
                    }),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[32, 32])),
                    operand_ids: Vec::new(),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = msa_edge.run(&mut msa_edge_mod)?;
    let _ = MiddleEndCostModelEvaluator::new(
        TpuGeneration::V5e,
        EmitterLearnedCostModelOptionsProto::default(),
    )
    .evaluate_module(&msa_edge_mod);

    // Empty instructions computation in HeapSimulator
    let mut empty_comp_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 99,
            instructions: Vec::new(),
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = GlobalDecreasingSizeBestFitHeapSimulator::default().run(&mut empty_comp_mod);

    // 2-gap best-fit slack test: create large/medium/small overlapping & sequential VMEM buffers so 2 distinct free gaps appear!
    let vmem_lay = || {
        Some(Box::new(LayoutProto {
            minor_to_major: vec![0],
            memory_space: 1,
            ..LayoutProto::default()
        }))
    };
    let vmem_vec = |n: i64| ShapeProto {
        element_type: PrimitiveType::F32 as i32,
        dimensions: vec![n],
        layout: vmem_lay(),
        ..ShapeProto::default()
    };
    // Sizes sorted decreasing:
    // b_big1 (step 1..2, 1024B) -> offset 0..1024
    // b_big2 (step 1..5, 768B)  -> offset 1024..1792
    // b_big3 (step 1..2, 512B)  -> offset 1792..2304
    // b_big4 (step 1..5, 512B)  -> offset 2304..2816
    // Now at step 3..4, b_big2 [1024..1792] and b_big4 [2304..2816] are STILL LIVE, leaving TWO gaps:
    // Gap 1: [0..1024] (1024B), Gap 2: [1792..2304] (512B)!
    // Candidate b_small1 (step 3..4, 512B): checks Gap 1 (slack=512) -> best=Some(512), then Gap 2 (slack=0 < 512 -> tighter!)!
    // Candidate b_small2 (step 3..4, 256B): Gap 2 is now full, Gap 1 [0..1024] has slack=768, and if another gap [2816..] has larger slack -> tests slack < best_slack false too!
    let mut gap_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 9,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    name: String::from("big1"),
                    opcode: String::from("iota"),
                    shape: Some(vmem_vec(256)),
                    ..HloInstructionProto::default()
                }, // 1024B, dies step 2
                HloInstructionProto {
                    id: 3,
                    name: String::from("big2"),
                    opcode: String::from("iota"),
                    shape: Some(vmem_vec(192)),
                    ..HloInstructionProto::default()
                }, // 768B, dies step 8
                HloInstructionProto {
                    id: 4,
                    name: String::from("big3"),
                    opcode: String::from("iota"),
                    shape: Some(vmem_vec(128)),
                    ..HloInstructionProto::default()
                }, // 512B, dies step 3
                HloInstructionProto {
                    id: 5,
                    name: String::from("big4"),
                    opcode: String::from("iota"),
                    shape: Some(vmem_vec(128)),
                    ..HloInstructionProto::default()
                }, // 512B, dies step 8
                HloInstructionProto {
                    id: 6,
                    name: String::from("use13"),
                    opcode: String::from("add"),
                    shape: Some(vmem_vec(64)),
                    operand_ids: vec![2, 4],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 7,
                    name: String::from("small_tight"),
                    opcode: String::from("iota"),
                    shape: Some(vmem_vec(128)),
                    ..HloInstructionProto::default()
                }, // 512B -> picks Gap 2!
                HloInstructionProto {
                    id: 8,
                    name: String::from("small_wide"),
                    opcode: String::from("iota"),
                    shape: Some(vmem_vec(64)),
                    ..HloInstructionProto::default()
                }, // 256B -> compares remaining gaps!
                HloInstructionProto {
                    id: 9,
                    name: String::from("root"),
                    opcode: String::from("tuple"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![3, 5, 6, 7, 8],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    // Add one more long-lived block at high offset with a wider middle gap so slack < best_slack is ALSO false!
    if let Some(c) = gap_mod.computations.first_mut() {
        c.instructions.insert(
            5,
            HloInstructionProto {
                id: 50,
                name: String::from("big5_dead"),
                opcode: String::from("iota"),
                shape: Some(vmem_vec(384)), // 1536B -> dies early, creating a wider gap than 1024B!
                ..HloInstructionProto::default()
            },
        );
        c.instructions.insert(
            6,
            HloInstructionProto {
                id: 51,
                name: String::from("big6_live"),
                opcode: String::from("iota"),
                shape: Some(vmem_vec(128)),
                ..HloInstructionProto::default()
            },
        );
        if let Some(r) = c.instructions.last_mut() {
            r.operand_ids.push(51);
        }
    }
    let _ = GlobalDecreasingSizeBestFitHeapSimulator::default().run(&mut gap_mod);

    // --- C. layout_and_fusion.rs branch closure ---
    // 1. shape=None, dot with 1D/F32/aligned/missing-shape operands, negative minor_to_major error, batched 3D MxuTilingPass
    let mut lay_edge = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("after-all"),
                    shape: None,
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[16])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[8, 128])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[16])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![3, 3],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[2, 8, 8])),
                    operand_ids: vec![3, 3],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_batch_dimensions: vec![0],
                        rhs_batch_dimensions: vec![0],
                        lhs_contracting_dimensions: vec![2],
                        rhs_contracting_dimensions: vec![1],
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = TpuLayoutAssignmentPass::new(true, true).run(&mut lay_edge)?;
    let _ = MxuTilingPass::new(TpuGeneration::V4Lite).run(&mut lay_edge);

    let mut neg_m2m = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            instructions: vec![HloInstructionProto {
                id: 1,
                opcode: String::from("parameter"),
                constrain_layout: true,
                shape: Some(ShapeProto {
                    element_type: PrimitiveType::F32 as i32,
                    dimensions: vec![4, 4],
                    layout: Some(Box::new(LayoutProto {
                        minor_to_major: vec![-1, 0],
                        ..LayoutProto::default()
                    })),
                    ..ShapeProto::default()
                }),
                ..HloInstructionProto::default()
            }],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert!(
        TpuLayoutAssignmentPass::default()
            .run(&mut neg_m2m)
            .is_err()
    );

    // 2. ProducerConsumerFusion: root_id=0, duplicate operand [2, 2], already-absorbed producer
    let mut pcf_edge = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 0,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[8, 8])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![1, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![2, 1],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = TpuProducerConsumerFusionPass::new(1024 * 1024, 32).run(&mut pcf_edge);

    // 3. MultiOutputFusion: paired j, non-fusable op, direct & transitive dependency, disjoint inputs, budget exceeded, downstream remap
    let mut mof_edge = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 10,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[8, 8])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[8, 8])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![3, 1],
                    ..HloInstructionProto::default()
                }, // non-fusable + mid dep!
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![4, 1],
                    ..HloInstructionProto::default()
                }, // depends transitively via 4!
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                }, // disjoint input!
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                }, // pairs with 3!
                HloInstructionProto {
                    id: 8,
                    opcode: String::from("log"),
                    shape: Some(f32_shape(&[8, 8])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                }, // pairs with 6 (hits paired.contains(&7)!)
                HloInstructionProto {
                    id: 9,
                    opcode: String::from("sqrt"),
                    shape: Some(f32_shape(&[1024, 1024])),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                }, // uses remapped 3 + over budget!
                HloInstructionProto {
                    id: 10,
                    opcode: String::from("rsqrt"),
                    shape: Some(f32_shape(&[1024, 1024])),
                    operand_ids: vec![7],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(
        MultiOutputFusionPass::new(64 * 1024)
            .run(&mut mof_edge)
            .multi_output_fusions,
        2
    );

    // 4. HloRematerializationPass: early break once peak drops <= limit!
    let mut remat_break = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 5,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[256, 256])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![1, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![3, 1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[256, 256])),
                    operand_ids: vec![4, 2],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    // Peak before is 4 * 256KB = 1048576; set limit to 1048576 - 256 so 1 remat drops below or meets limit on next step!
    let _ = HloRematerializationPass::new(786_432).run(&mut remat_break);

    // --- D. legalize_and_simplify.rs branch closure ---
    // 1. DotDecomposer: 1-operand dot, empty-dim output dot, non-canonical rank-2/rank-3 dnums
    let mut dot_edges = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 6,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 4, 4])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("dot"),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[])),
                    operand_ids: vec![1, 1],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_contracting_dimensions: vec![0],
                        rhs_contracting_dimensions: vec![0],
                        lhs_batch_dimensions: Vec::new(),
                        rhs_batch_dimensions: Vec::new(),
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[2, 4, 4])),
                    operand_ids: vec![2, 2],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_batch_dimensions: vec![1],
                        rhs_batch_dimensions: vec![0],
                        lhs_contracting_dimensions: vec![2],
                        rhs_contracting_dimensions: vec![1],
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("dot"),
                    shape: Some(f32_shape(&[2, 4, 4])),
                    operand_ids: vec![2, 2],
                    dot_dimension_numbers: Some(DotDimensionNumbers {
                        lhs_batch_dimensions: vec![0],
                        rhs_batch_dimensions: vec![0],
                        lhs_contracting_dimensions: vec![1],
                        rhs_contracting_dimensions: vec![1],
                    }),
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = DotDecomposerPass.run(&mut dot_edges)?;

    // 2. ConvolutionLegalizer: base_dilation > 1 with 0 padding, invalid spatial dim -1, non-root conv
    let mut conv_edges = HloModuleProto {
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 4,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[1, 4, 2])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[2, 2, 2])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("convolution"),
                    shape: Some(f32_shape(&[1, 4, 2])),
                    operand_ids: vec![1, 2],
                    convolution_dimension_numbers: Some(ConvolutionDimensionNumbers {
                        input_batch_dimension: 0,
                        input_feature_dimension: 2,
                        input_spatial_dimensions: vec![-1],
                        kernel_input_feature_dimension: 1,
                        kernel_output_feature_dimension: 2,
                        kernel_spatial_dimensions: vec![0],
                        output_batch_dimension: 0,
                        output_feature_dimension: 2,
                        output_spatial_dimensions: vec![1],
                    }),
                    window: Some(Window {
                        dimensions: vec![WindowDimension {
                            size: 2,
                            stride: 1,
                            padding_low: 0,
                            padding_high: 0,
                            window_dilation: 1,
                            base_dilation: 2,
                            window_reversal: false,
                        }],
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[1, 4, 2])),
                    operand_ids: vec![3],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = ConvolutionLegalizerPass::new(1).run(&mut conv_edges)?;

    // 3. AlgebraicSimplifier, BroadcastSimplifier, ReshapeMover, TransposeFolding, TupleSimplifier, LICM, DCE, AsyncCollective, LatencyHiding
    let tup2_f32 = ShapeProto {
        element_type: PrimitiveType::Tuple as i32,
        tuple_shapes: vec![f32_shape(&[4]), f32_shape(&[4])],
        ..ShapeProto::default()
    };
    let mut simp_edges = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![
            HloComputationProto {
                id: 1,
                root_id: 25,
                instructions: vec![
                    HloInstructionProto {
                        id: 1,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[4, 4])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 2,
                        opcode: String::from("add"),
                        shape: Some(f32_shape(&[4, 4])),
                        operand_ids: vec![1, 1],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 3,
                        opcode: String::from("subtract"),
                        shape: Some(f32_shape(&[4, 4])),
                        operand_ids: Vec::new(),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 4,
                        opcode: String::from("reshape"),
                        shape: None,
                        operand_ids: vec![1],
                        ..HloInstructionProto::default()
                    },
                    // Broadcast with same shape & non-identity dims [1, 0], plus invalid inner dim 99
                    HloInstructionProto {
                        id: 5,
                        opcode: String::from("broadcast"),
                        shape: Some(f32_shape(&[4, 4])),
                        operand_ids: vec![1],
                        dimensions: vec![1, 0],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 6,
                        opcode: String::from("broadcast"),
                        shape: Some(f32_shape(&[4, 4])),
                        operand_ids: vec![1],
                        dimensions: vec![99],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 7,
                        opcode: String::from("broadcast"),
                        shape: Some(f32_shape(&[4, 4])),
                        operand_ids: vec![6],
                        dimensions: vec![0, 1],
                        ..HloInstructionProto::default()
                    },
                    // ReshapeMover with 0-operand negate
                    HloInstructionProto {
                        id: 8,
                        opcode: String::from("negate"),
                        operand_ids: Vec::new(),
                        ..HloInstructionProto::default()
                    },
                    // TransposeFolding: inner transpose with 0 operands, out-of-range dim, identity transpose with 0 operands, 1-operand dot
                    HloInstructionProto {
                        id: 9,
                        opcode: String::from("transpose"),
                        operand_ids: Vec::new(),
                        dimensions: Vec::new(),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 10,
                        opcode: String::from("transpose"),
                        operand_ids: vec![9],
                        dimensions: vec![0],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 11,
                        opcode: String::from("transpose"),
                        operand_ids: vec![1],
                        dimensions: vec![1, 0],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 12,
                        opcode: String::from("transpose"),
                        operand_ids: vec![11],
                        dimensions: vec![99, 0],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 13,
                        opcode: String::from("dot"),
                        operand_ids: vec![1],
                        ..HloInstructionProto::default()
                    },
                    // TupleSimplifier: empty tuple, arity mismatch tuple(gte(t,0)), non-gte element, mismatched src_tup
                    HloInstructionProto {
                        id: 14,
                        opcode: String::from("tuple"),
                        operand_ids: Vec::new(),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 15,
                        opcode: String::from("parameter"),
                        parameter_number: 1,
                        shape: Some(tup2_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 16,
                        opcode: String::from("parameter"),
                        parameter_number: 2,
                        shape: Some(tup2_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 17,
                        opcode: String::from("get-tuple-element"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![15],
                        tuple_index: 0,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 18,
                        opcode: String::from("get-tuple-element"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![16],
                        tuple_index: 1,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 19,
                        opcode: String::from("tuple"),
                        operand_ids: vec![17],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 20,
                        opcode: String::from("tuple"),
                        operand_ids: vec![17, 1],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 21,
                        opcode: String::from("tuple"),
                        operand_ids: vec![17, 18],
                        ..HloInstructionProto::default()
                    },
                    // Custom-call without side effect + two calls to same subcomp 2
                    HloInstructionProto {
                        id: 22,
                        opcode: String::from("custom-call"),
                        custom_call_has_side_effect: false,
                        called_computation_ids: vec![2],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 23,
                        opcode: String::from("while"),
                        operand_ids: vec![15],
                        called_computation_ids: vec![2],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 24,
                        opcode: String::from("while"),
                        operand_ids: vec![15],
                        called_computation_ids: vec![2],
                        ..HloInstructionProto::default()
                    },
                    // Two adjacent all-reduces with DIFFERENT replica_groups
                    HloInstructionProto {
                        id: 25,
                        opcode: String::from("all-reduce"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![17],
                        replica_groups: vec![crate::proto::ReplicaGroup {
                            replica_ids: vec![0, 1],
                        }],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 26,
                        opcode: String::from("all-reduce"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![17],
                        replica_groups: vec![crate::proto::ReplicaGroup {
                            replica_ids: vec![0, 2],
                        }],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            // Body computation 2: root is tuple with gte of non-param + root=add in comp 3 + empty comp 4
            HloComputationProto {
                id: 2,
                root_id: 34,
                instructions: vec![
                    HloInstructionProto {
                        id: 30,
                        opcode: String::from("parameter"),
                        shape: Some(tup2_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 31,
                        opcode: String::from("tuple"),
                        shape: Some(tup2_f32.clone()),
                        operand_ids: vec![30, 30],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 32,
                        opcode: String::from("get-tuple-element"),
                        operand_ids: vec![31],
                        tuple_index: 0,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 33,
                        opcode: String::from("negate"),
                        operand_ids: Vec::new(),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 34,
                        opcode: String::from("tuple"),
                        shape: Some(tup2_f32),
                        operand_ids: vec![32, 32],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            HloComputationProto {
                id: 3,
                root_id: 40,
                instructions: Vec::new(),
                ..HloComputationProto::default()
            },
        ],
        ..HloModuleProto::default()
    };
    let _ = AlgebraicSimplifierPass::default().run(&mut simp_edges);
    let _ = BroadcastSimplifierPass.run(&mut simp_edges);
    let _ = ReshapeMoverPass.run(&mut simp_edges);
    let _ = TransposeFoldingPass.run(&mut simp_edges);
    let _ = TupleSimplifierPass.run(&mut simp_edges);
    let _ = WhileLoopInvariantCodeMotionPass.run(&mut simp_edges);
    let _ = AsyncCollectiveFusionPass::default().run(&mut simp_edges);
    let _ = LatencyHidingSchedulerPass.run(&mut simp_edges);
    let _ = HloDcePass.run(&mut simp_edges);

    // Cyclic schedule guard in LatencyHidingSchedulerPass (order.len() < n) + zero-between [start, done]
    let mut cyc_mod = HloModuleProto {
        computations: vec![
            HloComputationProto {
                id: 1,
                instructions: vec![
                    HloInstructionProto {
                        id: 1,
                        opcode: String::from("negate"),
                        operand_ids: vec![2],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 2,
                        opcode: String::from("negate"),
                        operand_ids: vec![1],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            HloComputationProto {
                id: 2,
                root_id: 11,
                instructions: vec![
                    HloInstructionProto {
                        id: 10,
                        opcode: String::from("all-reduce-start"),
                        shape: Some(ShapeProto {
                            element_type: PrimitiveType::U8 as i32,
                            dimensions: vec![8],
                            ..ShapeProto::default()
                        }),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 11,
                        opcode: String::from("all-reduce-done"),
                        shape: Some(ShapeProto {
                            element_type: PrimitiveType::U8 as i32,
                            dimensions: vec![8],
                            ..ShapeProto::default()
                        }),
                        operand_ids: vec![10],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
        ],
        ..HloModuleProto::default()
    };
    let _ = LatencyHidingSchedulerPass.run(&mut cyc_mod);

    // Final micro-cases for 100% branch closure across MSA, MOF, LICM, DCE, Broadcast, AlgSimp:
    // (a) True multi-step live intervals creating 2 distinct spatial gaps (one smaller, one larger) in HeapSimulator:
    let vmem_arr = |elems: i64| ShapeProto {
        element_type: PrimitiveType::F32 as i32,
        dimensions: vec![elems],
        layout: Some(Box::new(LayoutProto {
            minor_to_major: vec![0],
            memory_space: 1,
            ..LayoutProto::default()
        })),
        ..ShapeProto::default()
    };
    // Decreasing sizes:
    // id=2 (2048B, born 1, dies 2) -> [0..2048]
    // id=3 (1536B, born 1, dies 8) -> [2048..3584] (LIVE at steps 4..7!)
    // id=4 (1024B, born 2, dies 3) -> [3584..4608]
    // id=5 (1024B, born 2, dies 8) -> [4608..5632] (LIVE at steps 4..7!)
    // At step 4..5, Gap1=[0..2048] (2048B), Gap2=[3584..4608] (1024B):
    //   id=6 (1024B, born 4, dies 5): Gap1 slack=1024 -> Some, Gap2 slack=0 < 1024 (TRUE!) -> [3584..4608]!
    // Also test slack < best_slack FALSE by having GapA (512B) before GapB (1024B):
    //   id=7 (512B, born 4, dies 5) placed at [0..512], leaving [512..2048] (1536B);
    //   Better: separate color=3 (CMEM) with Gap1=512B and Gap2=1024B!
    let cmem_arr = |elems: i64| ShapeProto {
        element_type: PrimitiveType::F32 as i32,
        dimensions: vec![elems],
        layout: Some(Box::new(LayoutProto {
            minor_to_major: vec![0],
            memory_space: 3,
            ..LayoutProto::default()
        })),
        ..ShapeProto::default()
    };
    // Equal-size (1024B = 256 f32s) buffers ordered by decreasing lifetime span:
    // VMEM (color=1): Gap1=2048B [0..2048], L1=[2048..3072], Gap2=1024B [3072..4096], L2=[4096..5120] -> probe (512B) sees slack 1536 then 512 < 1536 (TRUE!)
    // CMEM (color=3): Gap1=1024B [0..1024], L1=[1024..2048], Gap2=2048B [2048..4096], L2=[4096..5120] -> probe (512B) sees slack 512 then 1536 < 512 (FALSE!)
    let mut exact_gap_mod = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 30,
            instructions: vec![
                // step 0
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[4])),
                    ..HloInstructionProto::default()
                },
                // step 1: v_d1a [1..8] span 7 -> [0..1024]
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(vmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // step 2: v_d1b [2..8] span 6 -> [1024..2048]
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("negate"),
                    shape: Some(vmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // step 3: c_d1 [3..10] span 7 -> CMEM [0..1024]
                HloInstructionProto {
                    id: 12,
                    opcode: String::from("abs"),
                    shape: Some(cmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // step 4: v_L1 [4..13] span 9 -> wait: keep span 5 using late root? Use explicit consumer at step 11 (`4..11` span 7? Make v_d1a born 1..9 span 8, v_d1b 2..9 span 7, v_L1 6..12 span 6, v_d2 5..9 span 4? Wait: v_d2 must overlap v_L1 at 6..9 -> born 4..9 span 5!)
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("negate"),
                    shape: Some(vmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // step 5: c_d2a [5..10] span 5
                HloInstructionProto {
                    id: 13,
                    opcode: String::from("abs"),
                    shape: Some(cmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // step 6: c_d2b [6..10] span 4
                HloInstructionProto {
                    id: 14,
                    opcode: String::from("abs"),
                    shape: Some(cmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // step 7: v_L1 [7..13] span 6 & c_L1 [7..13] span 6
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("negate"),
                    shape: Some(vmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 15,
                    opcode: String::from("abs"),
                    shape: Some(cmem_arr(256)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // step 9: kill v_d1a(2), v_d1b(3), v_d2(4) here -> spans: id2=8, id3=7, id4=5! And v_L2 born here [9..13] span 4!
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("tuple"),
                    shape: Some(vmem_arr(256)),
                    operand_ids: vec![2, 3, 4],
                    ..HloInstructionProto::default()
                },
                // step 10: kill c_d1(12), c_d2a(13), c_d2b(14) here -> spans: id12=7, id13=5, id14=4, while id15(c_L1) has [7..13] span 6! And c_L2 born here [10..13] span 3!
                HloInstructionProto {
                    id: 16,
                    opcode: String::from("tuple"),
                    shape: Some(cmem_arr(256)),
                    operand_ids: vec![12, 13, 14],
                    ..HloInstructionProto::default()
                },
                // step 11..12: probes (512B = 128 elems) in VMEM and CMEM!
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("negate"),
                    shape: Some(vmem_arr(128)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 8,
                    opcode: String::from("negate"),
                    shape: Some(vmem_arr(64)),
                    operand_ids: vec![7],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 17,
                    opcode: String::from("abs"),
                    shape: Some(cmem_arr(128)),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 18,
                    opcode: String::from("abs"),
                    shape: Some(cmem_arr(64)),
                    operand_ids: vec![17],
                    ..HloInstructionProto::default()
                },
                // step 15 (root): keep v_L1(5), v_L2(6), c_L1(15), c_L2(16), and probe users (8, 18) live until root!
                HloInstructionProto {
                    id: 30,
                    opcode: String::from("tuple"),
                    shape: Some(f32_shape(&[4])),
                    operand_ids: vec![5, 6, 8, 15, 16, 18],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    let _ = GlobalDecreasingSizeBestFitHeapSimulator::default().run(&mut exact_gap_mod);

    // (b) MultiOutputFusion: direct dependency inst_b uses inst_a, over-budget shared input, and second pair inst_b remapping operand from first pair!
    let mut mof_all = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![HloComputationProto {
            id: 1,
            root_id: 8,
            instructions: vec![
                HloInstructionProto {
                    id: 1,
                    opcode: String::from("parameter"),
                    shape: Some(f32_shape(&[16, 16])),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 2,
                    opcode: String::from("negate"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // Direct dependency on id=2 while also sharing id=1 -> inst_b.operand_ids.contains(&inst_a.id) TRUE!
                HloInstructionProto {
                    id: 3,
                    opcode: String::from("add"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![1, 2],
                    ..HloInstructionProto::default()
                },
                // Pairs with id=2 -> first MOF pair (2, 4)!
                HloInstructionProto {
                    id: 4,
                    opcode: String::from("abs"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                // Second MOF pair (5, 6) both using id=2 (remapped!) -> hits remap.get(op_mut) on inst_b!
                HloInstructionProto {
                    id: 5,
                    opcode: String::from("tanh"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 6,
                    opcode: String::from("exponential"),
                    shape: Some(f32_shape(&[16, 16])),
                    operand_ids: vec![2],
                    ..HloInstructionProto::default()
                },
                // Shared input id=1 but huge shapes exceeding budget -> bytes_ab <= vmem_budget_bytes FALSE!
                HloInstructionProto {
                    id: 7,
                    opcode: String::from("log"),
                    shape: Some(f32_shape(&[512, 512])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 8,
                    opcode: String::from("sqrt"),
                    shape: Some(f32_shape(&[512, 512])),
                    operand_ids: vec![1],
                    ..HloInstructionProto::default()
                },
            ],
            ..HloComputationProto::default()
        }],
        ..HloModuleProto::default()
    };
    assert_eq!(
        MultiOutputFusionPass::new(16 * 1024)
            .run(&mut mof_all)
            .multi_output_fusions,
        2
    );

    // (c) Legalize & Simplify final 6 micro-branches:
    //     - BroadcastSimplifier composed.len() != prod.dimensions.len()
    //     - AlgebraicSimplifier empty LiteralProto::default() in add + convert F32 -> BF16 (same_type_and_dims elem_type mismatch)
    //     - AsyncCollectiveFusionPass U8 shape bytes
    //     - LICM root_id == hoistable add AND gte of non-param in root tuple (run LICM directly without TupleSimplifier!)
    //     - HloDcePass live_comps.insert(cid) == false (two live while calls to comp 2) & non-existent root_id fallback
    let mut final_ls = HloModuleProto {
        entry_computation_id: 1,
        computations: vec![
            HloComputationProto {
                id: 1,
                root_id: 999, // missing root_id -> fallback to comp.instructions.last()!
                instructions: vec![
                    HloInstructionProto {
                        id: 1,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[4])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 2,
                        opcode: String::from("constant"),
                        shape: Some(f32_shape(&[4])),
                        literal: Some(LiteralProto::default()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 3,
                        opcode: String::from("add"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![1, 2],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 4,
                        opcode: String::from("convert"),
                        shape: Some(ShapeProto {
                            element_type: PrimitiveType::Bf16 as i32,
                            dimensions: vec![4],
                            ..ShapeProto::default()
                        }),
                        operand_ids: vec![3],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 5,
                        opcode: String::from("broadcast"),
                        shape: Some(f32_shape(&[2, 4])),
                        operand_ids: vec![1],
                        dimensions: vec![99],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 6,
                        opcode: String::from("broadcast"),
                        shape: Some(f32_shape(&[3, 2, 4])),
                        operand_ids: vec![5],
                        dimensions: vec![1, 2],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 7,
                        opcode: String::from("all-reduce"),
                        shape: Some(ShapeProto {
                            element_type: PrimitiveType::U8 as i32,
                            dimensions: vec![8],
                            ..ShapeProto::default()
                        }),
                        operand_ids: vec![1],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 8,
                        opcode: String::from("while"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![1],
                        called_computation_ids: vec![2],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 9,
                        opcode: String::from("while"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![8],
                        called_computation_ids: vec![3, 2],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            // Comp 2: root is pure `add` on constants -> hits inst.id != body_comp.root_id FALSE!
            HloComputationProto {
                id: 2,
                root_id: 22,
                instructions: vec![
                    HloInstructionProto {
                        id: 20,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[4])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 21,
                        opcode: String::from("constant"),
                        shape: Some(f32_shape(&[4])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 22,
                        opcode: String::from("add"),
                        shape: Some(f32_shape(&[4])),
                        operand_ids: vec![21, 21],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
            // Comp 3: root is `tuple` of `gte(other_tup, 0)` where `other_tup != param` -> hits g.operand_ids.first() == Some(param_id) FALSE!
            HloComputationProto {
                id: 3,
                root_id: 33,
                instructions: vec![
                    HloInstructionProto {
                        id: 30,
                        opcode: String::from("parameter"),
                        shape: Some(f32_shape(&[4])),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 31,
                        opcode: String::from("tuple"),
                        operand_ids: vec![30],
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 32,
                        opcode: String::from("get-tuple-element"),
                        operand_ids: vec![31],
                        tuple_index: 0,
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        id: 33,
                        opcode: String::from("tuple"),
                        operand_ids: vec![32],
                        ..HloInstructionProto::default()
                    },
                ],
                ..HloComputationProto::default()
            },
        ],
        ..HloModuleProto::default()
    };
    let _ = AlgebraicSimplifierPass::default().run(&mut final_ls);
    let _ = BroadcastSimplifierPass.run(&mut final_ls);
    let _ = WhileLoopInvariantCodeMotionPass.run(&mut final_ls);
    let _ = AsyncCollectiveFusionPass::default().run(&mut final_ls);
    let _ = HloDcePass.run(&mut final_ls);

    Ok(())
}
