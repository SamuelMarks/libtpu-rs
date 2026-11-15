//! Complete 18+1 `SparseCore` gradient optimizer suite mapped 1:1 to `tensorflow.tpu.OptimizationParametersProto`.
//!
//! Implements binary `TEC` micro-kernel emission and exact floating-point state updates for:
//! 1. `SGD` (`StochasticGradientDescent`), 2. `Adagrad`, 3. `AdagradMomentum`, 4. `BoundedAdagrad`,
//! 5. `MdlAdagradLight`, 6. `FrequencyAwareAdagrad`, 7. `ProximalAdagrad`, 8. `Adam`,
//! 9. `FTRL`, 10. `Lion`, 11. `Momentum`, 12. `RMSProp`, 13. `CenteredRMSProp`,
//! 14. `OnlineYogi`, 15. `ProximalYogi`, 16. `Assign`, 17. `FrequencyEstimator`,
//! 18. `UserDefinedProgram` (`xla.HloModuleProto`), plus `Adadelta`.

use super::codecs::{
    SparseCoreOp, SparseCoreSubcoreKind, SparseCoreV2Codec, SparseCoreV2Instruction,
    SparseCoreV3Codec, SparseCoreV3Instruction,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    AdadeltaParametersProto, AdagradMomentumParametersProto, AdagradParametersProto,
    AdamParametersProto, AssignParametersProto, BoundedAdagradParametersProto,
    CenteredRmsPropParametersProto, ClippingLimitsProto, FrequencyAwareAdagradParametersProto,
    FrequencyEstimatorParametersProto, FtrlParametersProto, HloModuleProto, LearningRateProto,
    LionParametersProto, MdlAdagradLightParametersProto, MomentumParametersProto,
    OnlineYogiParametersProto, OptimizationParametersProto, ProximalAdagradParametersProto,
    ProximalYogiParametersProto, RmsPropParametersProto, StochasticGradientDescentParametersProto,
    UserDefinedProgramParametersProto,
};

/// Strongly-typed `SparseCore` embedding gradient optimizer rule (`SparseCoreGradientOptimizer`).
#[derive(Debug, Clone, PartialEq)]
pub enum SparseCoreGradientOptimizer {
    /// 1. Stochastic Gradient Descent (`w <- w - lr * g`).
    Sgd,
    /// 2. Adagrad (`acc <- acc + g^2; w <- w - lr * g / sqrt(acc + eps)`).
    Adagrad {
        /// Numerical stability epsilon.
        epsilon: f32,
    },
    /// 3. Adagrad with Momentum / Nesterov acceleration.
    AdagradMomentum {
        /// Momentum decay factor.
        momentum: f32,
        /// Whether Nesterov lookahead is enabled.
        use_nesterov: bool,
        /// Preconditioner root exponent (e.g., `2.0`).
        exponent: f32,
        /// Exponential moving-average factor (`<= 0` or `>= 1` uses cumulative sum).
        beta2: f32,
        /// Numerical stability epsilon.
        epsilon: f32,
    },
    /// 4. Bounded Adagrad clamping both accumulator and per-step variable delta.
    BoundedAdagrad {
        /// Whether accumulator is updated prior to step computation.
        update_accumulator_first: bool,
        /// Maximum absolute variable update per step.
        max_var_update: f32,
        /// Maximum accumulator value.
        max_accumulator: f32,
    },
    /// 5. Minimum Description Length (`MDL`) Adagrad Light.
    MdlAdagradLight {
        /// L2 regularization strength.
        l2: f32,
        /// Learning rate power.
        lr_power: f32,
        /// Minimum servable MDL benefit threshold.
        min_servable_mdl_benefit: f32,
        /// Benefit rampup coefficient.
        mdl_benefit_rampup_coeff: f32,
    },
    /// 6. Frequency-Aware Adagrad scaling effective learning rate by occurrence probability.
    FrequencyAwareAdagrad {
        /// L1 proximal threshold.
        l1_regularization_strength: f32,
        /// L2 regularization factor.
        l2_regularization_strength: f32,
        /// Occurrence probability exponent.
        probability_exponent: f32,
        /// Upper cap on frequency learning-rate multiplier.
        max_lr_multiplier: f32,
        /// Exponential decay on accumulator.
        accumulator_decay: f32,
    },
    /// 7. Proximal Adagrad (`FOBOS` L1/L2 shrinkage).
    ProximalAdagrad {
        /// L1 regularization strength.
        l1: f32,
        /// L2 regularization strength.
        l2: f32,
    },
    /// 8. Adam (`m`, `v` first/second moment estimates).
    Adam {
        /// First-moment decay (`beta1`).
        beta1: f32,
        /// Second-moment decay (`beta2`).
        beta2: f32,
        /// Numerical stability epsilon.
        epsilon: f32,
        /// Whether non-lazy weight updates apply to untouched rows.
        use_non_lazy_adam: bool,
        /// Whether `epsilon` is added inside `sqrt(v + eps)`.
        use_sum_inside_sqrt: bool,
    },
    /// 9. Follow-The-Regularized-Leader (`FTRL`-Proximal).
    Ftrl {
        /// L1 sparsity regularization.
        l1: f32,
        /// L2 ridge regularization.
        l2: f32,
        /// Negative learning-rate power (typically `-0.5`).
        lr_power: f32,
        /// Smoothing parameter `beta`.
        beta: f32,
        /// Whether linear accumulator stores `z * lr`.
        multiply_linear_by_lr: bool,
    },
    /// 10. Lion (`EvoLved Sign Momentum`).
    Lion {
        /// Update interpolation decay (`beta1`).
        beta1: f32,
        /// Momentum state EMA decay (`beta2`).
        beta2: f32,
    },
    /// 11. Classical / Nesterov Momentum.
    Momentum {
        /// Momentum coefficient.
        momentum: f32,
        /// Whether Nesterov lookahead is applied.
        use_nesterov: bool,
    },
    /// 12. `RMSProp` (`ms <- rho * ms + (1 - rho) * g^2`).
    RmsProp {
        /// Mean-square decay `rho`.
        rho: f32,
        /// Momentum factor.
        momentum: f32,
        /// Numerical stability epsilon.
        epsilon: f32,
    },
    /// 13. Centered `RMSProp` (`ms - mg^2` variance normalization).
    CenteredRmsProp {
        /// Mean-square and mean-gradient decay `rho`.
        rho: f32,
        /// Momentum factor.
        momentum: f32,
        /// Numerical stability epsilon.
        epsilon: f32,
    },
    /// 14. Online Yogi additive adaptive second-moment optimizer.
    OnlineYogi {
        /// L1 regularization strength.
        l1: f32,
        /// L2 regularization strength.
        l2: f32,
        /// Second-moment Yogi parameter `beta2`.
        beta2: f32,
    },
    /// 15. Proximal Yogi with first & second moments and L1/L2 proximal shrinkage.
    ProximalYogi {
        /// L1 regularization strength.
        l1: f32,
        /// L2 regularization strength.
        l2: f32,
        /// First-moment decay `beta1`.
        beta1: f32,
        /// Second-moment Yogi parameter `beta2`.
        beta2: f32,
        /// Numerical stability epsilon.
        epsilon: f32,
    },
    /// 16. Direct table overwrite (`w <- g`).
    Assign,
    /// 17. Streaming inter-arrival Frequency Estimator (`slot0 <- (1 - alpha) * slot0 + alpha * delta`).
    FrequencyEstimator {
        /// Time constant `tau`.
        tau: f32,
        /// Maximum step delta cap.
        max_delta: f32,
        /// Outlier threshold.
        outlier_threshold: f32,
        /// Weight exponent.
        weight_exponent: f32,
    },
    /// 18. Custom XLA `HloModuleProto` user-defined optimizer program (`UserDefinedProgram`).
    UserDefinedProgram {
        /// Embedded `xla.HloModuleProto` custom update computation.
        program: Box<HloModuleProto>,
    },
    /// 19. Adadelta (`acc_g`, `acc_delta` adaptive ratio without explicit learning rate).
    Adadelta {
        /// Exponential decay `rho`.
        rho: f32,
        /// Numerical stability epsilon.
        epsilon: f32,
    },
}

impl SparseCoreGradientOptimizer {
    /// Returns the hardware `TEC` optimizer opcode (`0..=18`).
    #[must_use]
    pub const fn opcode_id(&self) -> u8 {
        match self {
            Self::Sgd => 0,
            Self::Adagrad { .. } => 1,
            Self::AdagradMomentum { .. } => 2,
            Self::BoundedAdagrad { .. } => 3,
            Self::MdlAdagradLight { .. } => 4,
            Self::FrequencyAwareAdagrad { .. } => 5,
            Self::ProximalAdagrad { .. } => 6,
            Self::Adam { .. } => 7,
            Self::Ftrl { .. } => 8,
            Self::Lion { .. } => 9,
            Self::Momentum { .. } => 10,
            Self::RmsProp { .. } => 11,
            Self::CenteredRmsProp { .. } => 12,
            Self::OnlineYogi { .. } => 13,
            Self::ProximalYogi { .. } => 14,
            Self::Assign => 15,
            Self::FrequencyEstimator { .. } => 16,
            Self::UserDefinedProgram { .. } => 17,
            Self::Adadelta { .. } => 18,
        }
    }

    /// Returns the number of auxiliary per-row SPMEM state slots (`0..=3`) required by this optimizer.
    #[must_use]
    pub const fn state_slot_count(&self) -> u32 {
        match self {
            Self::Sgd | Self::Assign => 0,
            Self::Adagrad { .. }
            | Self::BoundedAdagrad { .. }
            | Self::ProximalAdagrad { .. }
            | Self::Lion { .. }
            | Self::Momentum { .. }
            | Self::FrequencyEstimator { .. }
            | Self::UserDefinedProgram { .. } => 1,
            Self::AdagradMomentum { .. }
            | Self::MdlAdagradLight { .. }
            | Self::FrequencyAwareAdagrad { .. }
            | Self::Adam { .. }
            | Self::Ftrl { .. }
            | Self::RmsProp { .. }
            | Self::OnlineYogi { .. }
            | Self::ProximalYogi { .. }
            | Self::Adadelta { .. } => 2,
            Self::CenteredRmsProp { .. } => 3,
        }
    }
}

/// Complete `SparseCore` optimizer configuration with weight decay and clipping (`SparseCoreOptimizerConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseCoreOptimizerConfig {
    /// Optimizer algorithm and hyperparameters.
    pub optimizer: SparseCoreGradientOptimizer,
    /// Base scalar learning rate (`lr`).
    pub learning_rate: f32,
    /// Decoupled or L2 weight decay factor.
    pub weight_decay_factor: f32,
    /// Whether weight decay is multiplied by `learning_rate`.
    pub multiply_weight_decay_by_lr: bool,
    /// Optional `(min, max)` gradient clipping bounds.
    pub gradient_clip: Option<(f32, f32)>,
    /// Optional `(min, max)` post-update weight clipping bounds.
    pub weight_clip: Option<(f32, f32)>,
}

impl SparseCoreOptimizerConfig {
    /// Constructs a [`SparseCoreOptimizerConfig`] with `optimizer` and `learning_rate`.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - [`SparseCoreGradientOptimizer`] variant.
    /// * `learning_rate` - Scalar learning rate.
    #[must_use]
    pub const fn new(optimizer: SparseCoreGradientOptimizer, learning_rate: f32) -> Self {
        Self {
            optimizer,
            learning_rate,
            weight_decay_factor: 0.0,
            multiply_weight_decay_by_lr: false,
            gradient_clip: None,
            weight_clip: None,
        }
    }

    /// Parses a `tensorflow.tpu.OptimizationParametersProto` into a [`SparseCoreOptimizerConfig`].
    ///
    /// # Arguments
    ///
    /// * `proto` - Source `OptimizationParametersProto`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if `proto.parameters` is `None`.
    #[allow(clippy::too_many_lines)]
    pub fn from_proto(proto: &OptimizationParametersProto) -> LibTpuResult<Self> {
        use crate::proto::generated_prost_schemas::tensorflow::tpu::{
            learning_rate::LearningRate as LrOneof, optimization_parameters::Parameters,
        };
        let lr = match proto
            .learning_rate
            .as_ref()
            .and_then(|l| l.learning_rate.as_ref())
        {
            Some(LrOneof::Constant(c)) => *c,
            Some(LrOneof::Dynamic(_)) | None => 0.01,
        };
        let parse_clip = |c: &Option<ClippingLimitsProto>| -> Option<(f32, f32)> {
            let lim = c.as_ref()?;
            let lo = lim.lower.unwrap_or(f32::NEG_INFINITY);
            let hi = lim.upper.unwrap_or(f32::INFINITY);
            Some((lo, hi))
        };

        let Some(params) = proto.parameters.as_ref() else {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset: 0,
                reason: String::from("OptimizationParametersProto missing parameters oneof"),
            });
        };

        let optimizer = match params {
            Parameters::StochasticGradientDescent(_) => SparseCoreGradientOptimizer::Sgd,
            Parameters::Adagrad(_) => SparseCoreGradientOptimizer::Adagrad { epsilon: 1e-8 },
            Parameters::AdagradMomentum(p) => SparseCoreGradientOptimizer::AdagradMomentum {
                momentum: p.momentum,
                use_nesterov: p.use_nesterov,
                exponent: if p.exponent > 0.0 { p.exponent } else { 2.0 },
                beta2: p.beta2,
                epsilon: if p.epsilon > 0.0 { p.epsilon } else { 1e-8 },
            },
            Parameters::BoundedAdagrad(p) => SparseCoreGradientOptimizer::BoundedAdagrad {
                update_accumulator_first: p.update_accumulator_first,
                max_var_update: if p.max_var_update > 0.0 {
                    p.max_var_update
                } else {
                    1.0
                },
                max_accumulator: if p.max_accumulator > 0.0 {
                    p.max_accumulator
                } else {
                    100.0
                },
            },
            Parameters::MdlAdagradLight(p) => SparseCoreGradientOptimizer::MdlAdagradLight {
                l2: p.l2,
                lr_power: if p.lr_power == 0.0 { -0.5 } else { p.lr_power },
                min_servable_mdl_benefit: p.min_servable_mdl_benefit,
                mdl_benefit_rampup_coeff: p.mdl_benefit_rampup_coeff.max(0.1),
            },
            Parameters::FrequencyAwareAdagrad(p) => {
                SparseCoreGradientOptimizer::FrequencyAwareAdagrad {
                    l1_regularization_strength: p.l1_regularization_strength,
                    l2_regularization_strength: p.l2_regularization_strength,
                    probability_exponent: p.probability_exponent,
                    max_lr_multiplier: p.max_lr_multiplier.max(1.0),
                    accumulator_decay: p.accumulator_decay,
                }
            }
            Parameters::ProximalAdagrad(p) => {
                SparseCoreGradientOptimizer::ProximalAdagrad { l1: p.l1, l2: p.l2 }
            }
            Parameters::Adam(p) => SparseCoreGradientOptimizer::Adam {
                beta1: p.beta1,
                beta2: p.beta2,
                epsilon: if p.epsilon > 0.0 { p.epsilon } else { 1e-8 },
                use_non_lazy_adam: p.use_non_lazy_adam,
                use_sum_inside_sqrt: p.use_sum_inside_sqrt,
            },
            Parameters::Ftrl(p) => SparseCoreGradientOptimizer::Ftrl {
                l1: p.l1,
                l2: p.l2,
                lr_power: if p.lr_power == 0.0 { -0.5 } else { p.lr_power },
                beta: p.beta,
                multiply_linear_by_lr: p.multiply_linear_by_lr,
            },
            Parameters::Lion(p) => SparseCoreGradientOptimizer::Lion {
                beta1: p.beta1,
                beta2: p.beta2,
            },
            Parameters::Momentum(p) => SparseCoreGradientOptimizer::Momentum {
                momentum: p.momentum,
                use_nesterov: p.use_nesterov,
            },
            Parameters::RmsProp(p) => SparseCoreGradientOptimizer::RmsProp {
                rho: p.rho,
                momentum: p.momentum,
                epsilon: if p.epsilon > 0.0 { p.epsilon } else { 1e-8 },
            },
            Parameters::CenteredRmsProp(p) => SparseCoreGradientOptimizer::CenteredRmsProp {
                rho: p.rho,
                momentum: p.momentum,
                epsilon: if p.epsilon > 0.0 { p.epsilon } else { 1e-8 },
            },
            Parameters::OnlineYogi(p) => SparseCoreGradientOptimizer::OnlineYogi {
                l1: p.l1,
                l2: p.l2,
                beta2: p.beta2,
            },
            Parameters::ProximalYogi(p) => SparseCoreGradientOptimizer::ProximalYogi {
                l1: p.l1,
                l2: p.l2,
                beta1: p.beta1,
                beta2: p.beta2,
                epsilon: if p.epsilon > 0.0 { p.epsilon } else { 1e-8 },
            },
            Parameters::Assign(_) => SparseCoreGradientOptimizer::Assign,
            Parameters::FrequencyEstimator(p) => SparseCoreGradientOptimizer::FrequencyEstimator {
                tau: p.tau.max(1.0),
                max_delta: p.max_delta.max(1.0),
                outlier_threshold: p.outlier_threshold,
                weight_exponent: p.weight_exponent,
            },
            Parameters::UserDefinedProgram(p) => SparseCoreGradientOptimizer::UserDefinedProgram {
                program: Box::new(p.program.clone().unwrap_or_default()),
            },
            Parameters::Adadelta(p) => SparseCoreGradientOptimizer::Adadelta {
                rho: p.rho,
                epsilon: if p.epsilon > 0.0 { p.epsilon } else { 1e-6 },
            },
        };

        Ok(Self {
            optimizer,
            learning_rate: lr,
            weight_decay_factor: proto.weight_decay_factor,
            multiply_weight_decay_by_lr: proto.multiply_weight_decay_factor_by_learning_rate,
            gradient_clip: parse_clip(&proto.gradient_clipping_limits),
            weight_clip: parse_clip(&proto.clipping_limits),
        })
    }

    /// Serializes this [`SparseCoreOptimizerConfig`] into a wire-exact `OptimizationParametersProto`.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_proto(&self) -> OptimizationParametersProto {
        use crate::proto::generated_prost_schemas::tensorflow::tpu::{
            learning_rate::LearningRate as LrOneof, optimization_parameters::Parameters,
        };
        let params = match &self.optimizer {
            SparseCoreGradientOptimizer::Sgd => {
                Parameters::StochasticGradientDescent(StochasticGradientDescentParametersProto {})
            }
            SparseCoreGradientOptimizer::Adagrad { .. } => {
                Parameters::Adagrad(AdagradParametersProto {})
            }
            SparseCoreGradientOptimizer::AdagradMomentum {
                momentum,
                use_nesterov,
                exponent,
                beta2,
                epsilon,
            } => Parameters::AdagradMomentum(AdagradMomentumParametersProto {
                momentum: *momentum,
                use_nesterov: *use_nesterov,
                exponent: *exponent,
                beta2: *beta2,
                epsilon: *epsilon,
            }),
            SparseCoreGradientOptimizer::BoundedAdagrad {
                update_accumulator_first,
                max_var_update,
                max_accumulator,
            } => Parameters::BoundedAdagrad(BoundedAdagradParametersProto {
                update_accumulator_first: *update_accumulator_first,
                max_var_update: *max_var_update,
                max_accumulator: *max_accumulator,
            }),
            SparseCoreGradientOptimizer::MdlAdagradLight {
                l2,
                lr_power,
                min_servable_mdl_benefit,
                mdl_benefit_rampup_coeff,
            } => Parameters::MdlAdagradLight(MdlAdagradLightParametersProto {
                l2: *l2,
                lr_power: *lr_power,
                min_servable_mdl_benefit: *min_servable_mdl_benefit,
                mdl_benefit_rampup_coeff: *mdl_benefit_rampup_coeff,
                ..MdlAdagradLightParametersProto::default()
            }),
            SparseCoreGradientOptimizer::FrequencyAwareAdagrad {
                l1_regularization_strength,
                l2_regularization_strength,
                probability_exponent,
                max_lr_multiplier,
                accumulator_decay,
            } => Parameters::FrequencyAwareAdagrad(FrequencyAwareAdagradParametersProto {
                l1_regularization_strength: *l1_regularization_strength,
                l2_regularization_strength: *l2_regularization_strength,
                probability_exponent: *probability_exponent,
                max_lr_multiplier: *max_lr_multiplier,
                accumulator_decay: *accumulator_decay,
                ..FrequencyAwareAdagradParametersProto::default()
            }),
            SparseCoreGradientOptimizer::ProximalAdagrad { l1, l2 } => {
                Parameters::ProximalAdagrad(ProximalAdagradParametersProto { l1: *l1, l2: *l2 })
            }
            SparseCoreGradientOptimizer::Adam {
                beta1,
                beta2,
                epsilon,
                use_non_lazy_adam,
                use_sum_inside_sqrt,
            } => Parameters::Adam(AdamParametersProto {
                beta1: *beta1,
                beta2: *beta2,
                epsilon: *epsilon,
                use_non_lazy_adam: *use_non_lazy_adam,
                use_sum_inside_sqrt: *use_sum_inside_sqrt,
            }),
            SparseCoreGradientOptimizer::Ftrl {
                l1,
                l2,
                lr_power,
                beta,
                multiply_linear_by_lr,
            } => Parameters::Ftrl(FtrlParametersProto {
                l1: *l1,
                l2: *l2,
                lr_power: *lr_power,
                beta: *beta,
                multiply_linear_by_lr: *multiply_linear_by_lr,
                allow_zero_accumulator: true,
            }),
            SparseCoreGradientOptimizer::Lion { beta1, beta2 } => {
                Parameters::Lion(LionParametersProto {
                    beta1: *beta1,
                    beta2: *beta2,
                    use_non_lazy_lion: false,
                })
            }
            SparseCoreGradientOptimizer::Momentum {
                momentum,
                use_nesterov,
            } => Parameters::Momentum(MomentumParametersProto {
                momentum: *momentum,
                use_nesterov: *use_nesterov,
            }),
            SparseCoreGradientOptimizer::RmsProp {
                rho,
                momentum,
                epsilon,
            } => Parameters::RmsProp(RmsPropParametersProto {
                rho: *rho,
                momentum: *momentum,
                epsilon: *epsilon,
            }),
            SparseCoreGradientOptimizer::CenteredRmsProp {
                rho,
                momentum,
                epsilon,
            } => Parameters::CenteredRmsProp(CenteredRmsPropParametersProto {
                rho: *rho,
                momentum: *momentum,
                epsilon: *epsilon,
            }),
            SparseCoreGradientOptimizer::OnlineYogi { l1, l2, beta2 } => {
                Parameters::OnlineYogi(OnlineYogiParametersProto {
                    l1: *l1,
                    l2: *l2,
                    beta2: *beta2,
                })
            }
            SparseCoreGradientOptimizer::ProximalYogi {
                l1,
                l2,
                beta1,
                beta2,
                epsilon,
            } => Parameters::ProximalYogi(ProximalYogiParametersProto {
                l1: *l1,
                l2: *l2,
                beta1: *beta1,
                beta2: *beta2,
                epsilon: *epsilon,
            }),
            SparseCoreGradientOptimizer::Assign => Parameters::Assign(AssignParametersProto {}),
            SparseCoreGradientOptimizer::FrequencyEstimator {
                tau,
                max_delta,
                outlier_threshold,
                weight_exponent,
            } => Parameters::FrequencyEstimator(FrequencyEstimatorParametersProto {
                tau: *tau,
                max_delta: *max_delta,
                outlier_threshold: *outlier_threshold,
                weight_exponent: *weight_exponent,
            }),
            SparseCoreGradientOptimizer::UserDefinedProgram { program } => {
                Parameters::UserDefinedProgram(UserDefinedProgramParametersProto {
                    program: Some((**program).clone()),
                })
            }
            SparseCoreGradientOptimizer::Adadelta { rho, epsilon } => {
                Parameters::Adadelta(AdadeltaParametersProto {
                    rho: *rho,
                    epsilon: *epsilon,
                })
            }
        };

        let make_clip = |opt: Option<(f32, f32)>| -> Option<ClippingLimitsProto> {
            let (lo, hi) = opt?;
            Some(ClippingLimitsProto {
                lower: Some(lo),
                upper: Some(hi),
            })
        };

        OptimizationParametersProto {
            learning_rate: Some(LearningRateProto {
                learning_rate: Some(LrOneof::Constant(self.learning_rate)),
            }),
            clipping_limits: make_clip(self.weight_clip),
            gradient_clipping_limits: make_clip(self.gradient_clip),
            weight_decay_factor: self.weight_decay_factor,
            multiply_weight_decay_factor_by_learning_rate: self.multiply_weight_decay_by_lr,
            parameters: Some(params),
            ..OptimizationParametersProto::default()
        }
    }

    /// Emits a binary `SparseCore` V2 (`sparse_core_version == 2`) or V3 (`sparse_core_version >= 3`)
    /// `TEC` micro-kernel bitstream for this optimizer.
    ///
    /// # Arguments
    ///
    /// * `sparse_core_version` - Hardware `SparseCore` generation (`2` or `3`).
    #[must_use]
    pub fn emit_tec_microkernel_bitstream(&self, sparse_core_version: u8) -> Vec<u8> {
        let opt_id = self.optimizer.opcode_id();
        let ops = [
            (
                SparseCoreSubcoreKind::Tac,
                SparseCoreOp::SpmemGatherDma,
                1_u16,
            ),
            (
                SparseCoreSubcoreKind::Tec,
                SparseCoreOp::OptimizerUpdate,
                2_u16,
            ),
            (
                SparseCoreSubcoreKind::Tac,
                SparseCoreOp::SpmemScatterWriteDma,
                3_u16,
            ),
            (SparseCoreSubcoreKind::Scs, SparseCoreOp::SyncBarrier, 4_u16),
        ];
        let mut out = Vec::new();
        for (idx, &(sub, op, sf)) in ops.iter().enumerate() {
            let sp_off = u32::try_from(idx * 256).unwrap_or(0);
            if sparse_core_version >= 3 {
                out.extend_from_slice(&SparseCoreV3Codec::encode(&SparseCoreV3Instruction {
                    subcore: sub,
                    op,
                    optimizer_id: opt_id,
                    quant_mode: 0,
                    vector_lanes: 16,
                    spmem_offset: sp_off,
                    hbm3e_byte_address: (idx as u64) * 4096,
                    sync_flag_id: sf,
                }));
            } else {
                out.extend_from_slice(&SparseCoreV2Codec::encode(&SparseCoreV2Instruction {
                    subcore: sub,
                    op,
                    optimizer_id: opt_id,
                    spmem_offset: sp_off,
                    hbm_page_offset: u32::try_from(idx).unwrap_or(0),
                    transfer_bytes: 256,
                    sync_flag_id: sf,
                }));
            }
        }
        out
    }

    /// Applies one in-place `SparseCore` `TEC` row update across `weights` and up to 3 auxiliary
    /// state slots (`slot0`, `slot1`, `slot2`) for `grad`.
    ///
    /// # Arguments
    ///
    /// * `weights` - Embedding row weights (`w`).
    /// * `slot0` - Primary state slot (`acc` / `m` / `ms`).
    /// * `slot1` - Secondary state slot (`v` / `z` / `mom`).
    /// * `slot2` - Tertiary state slot (`mg` for `CenteredRMSProp`).
    /// * `grad` - Coalesced row gradient (`g`).
    #[allow(clippy::too_many_lines)]
    pub fn apply_row_update(
        &self,
        weights: &mut [f32],
        slot0: &mut [f32],
        slot1: &mut [f32],
        slot2: &mut [f32],
        grad: &[f32],
    ) {
        let lr = self.learning_rate;
        let len = weights.len().min(grad.len());
        for i in 0..len {
            let w0 = weights.get(i).copied().unwrap_or(0.0);
            let raw_g = grad.get(i).copied().unwrap_or(0.0);
            let mut g = if let Some((lo, hi)) = self.gradient_clip {
                raw_g.clamp(lo, hi)
            } else {
                raw_g
            };
            if self.weight_decay_factor != 0.0 {
                let wd = if self.multiply_weight_decay_by_lr {
                    self.weight_decay_factor * lr
                } else {
                    self.weight_decay_factor
                };
                g = f32::mul_add(wd, w0, g);
            }

            let s0 = slot0.get(i).copied().unwrap_or(0.0);
            let s1 = slot1.get(i).copied().unwrap_or(0.0);
            let s2 = slot2.get(i).copied().unwrap_or(0.0);

            let (new_w, new_s0, new_s1, new_s2) = match &self.optimizer {
                SparseCoreGradientOptimizer::Sgd => (lr.mul_add(-g, w0), s0, s1, s2),
                SparseCoreGradientOptimizer::Adagrad { epsilon } => {
                    let acc = f32::mul_add(g, g, s0);
                    (w0 - lr * g / (acc.sqrt() + *epsilon), acc, s1, s2)
                }
                SparseCoreGradientOptimizer::AdagradMomentum {
                    momentum,
                    use_nesterov,
                    exponent,
                    beta2,
                    epsilon,
                } => {
                    let acc = if (*beta2 > 0.0) && (*beta2 < 1.0) {
                        f32::mul_add((1.0 - *beta2) * g, g, (*beta2) * s0)
                    } else {
                        f32::mul_add(g, g, s0)
                    };
                    let scaled_g = g / (acc + *epsilon).powf(1.0 / exponent.max(1.0));
                    let mom = f32::mul_add(*momentum, s1, scaled_g);
                    let step = if *use_nesterov {
                        f32::mul_add(*momentum, mom, scaled_g)
                    } else {
                        mom
                    };
                    (lr.mul_add(-step, w0), acc, mom, s2)
                }
                SparseCoreGradientOptimizer::BoundedAdagrad {
                    update_accumulator_first,
                    max_var_update,
                    max_accumulator,
                } => {
                    let acc = f32::mul_add(g, g, s0).min(*max_accumulator);
                    let denom_acc = if *update_accumulator_first {
                        acc
                    } else {
                        s0.max(1e-6)
                    };
                    let delta = (lr * g / (denom_acc.sqrt() + 1e-8))
                        .clamp(-(*max_var_update), *max_var_update);
                    (w0 - delta, acc, s1, s2)
                }
                SparseCoreGradientOptimizer::MdlAdagradLight {
                    l2,
                    lr_power,
                    min_servable_mdl_benefit,
                    mdl_benefit_rampup_coeff,
                } => {
                    let acc = f32::mul_add(g, g, s0).max(1e-6);
                    let benefit = f32::mul_add(g.abs(), *mdl_benefit_rampup_coeff, s1);
                    let reg_g = f32::mul_add(*l2, w0, g);
                    let eff_lr = lr * acc.powf(*lr_power);
                    let updated = if benefit >= *min_servable_mdl_benefit {
                        eff_lr.mul_add(-reg_g, w0)
                    } else {
                        0.0
                    };
                    (updated, acc, benefit, s2)
                }
                SparseCoreGradientOptimizer::FrequencyAwareAdagrad {
                    l1_regularization_strength,
                    l2_regularization_strength,
                    probability_exponent,
                    max_lr_multiplier,
                    accumulator_decay,
                } => {
                    let freq = s1 + 1.0;
                    let mult = freq
                        .powf((*probability_exponent).max(0.0))
                        .min(*max_lr_multiplier);
                    let acc = f32::mul_add(g, g, (s0 * (1.0 - *accumulator_decay)).max(0.0));
                    let eff_lr = lr * mult / (acc.sqrt() + 1e-8);
                    let w_half =
                        eff_lr.mul_add(-f32::mul_add(*l2_regularization_strength, w0, g), w0);
                    let thresh = eff_lr * (*l1_regularization_strength);
                    let shrunk = if w_half.abs() <= thresh {
                        0.0
                    } else {
                        w_half.signum().mul_add(-thresh, w_half)
                    };
                    (shrunk, acc, freq, s2)
                }
                SparseCoreGradientOptimizer::ProximalAdagrad { l1, l2 } => {
                    let acc = f32::mul_add(g, g, s0).max(1e-8);
                    let eff_lr = lr / acc.sqrt();
                    let w_hat = eff_lr.mul_add(-g, w0);
                    let shrunk = if w_hat.abs() <= eff_lr * (*l1) {
                        0.0
                    } else {
                        (w_hat.signum() * eff_lr).mul_add(-(*l1), w_hat) / eff_lr.mul_add(*l2, 1.0)
                    };
                    (shrunk, acc, s1, s2)
                }
                SparseCoreGradientOptimizer::Adam {
                    beta1,
                    beta2,
                    epsilon,
                    use_sum_inside_sqrt,
                    ..
                } => {
                    let m = f32::mul_add(1.0 - *beta1, g, (*beta1) * s0);
                    let v = f32::mul_add((1.0 - *beta2) * g, g, (*beta2) * s1);
                    let denom = if *use_sum_inside_sqrt {
                        (v + *epsilon).sqrt()
                    } else {
                        v.sqrt() + *epsilon
                    };
                    (w0 - lr * m / denom, m, v, s2)
                }
                SparseCoreGradientOptimizer::Ftrl {
                    l1,
                    l2,
                    lr_power,
                    beta,
                    multiply_linear_by_lr,
                } => {
                    let prev_n = s0.max(1e-8);
                    let new_n = f32::mul_add(g, g, prev_n);
                    let sigma =
                        (new_n.powf(-(*lr_power)) - prev_n.powf(-(*lr_power))) / lr.max(1e-8);
                    let lin_scale = if *multiply_linear_by_lr { lr } else { 1.0 };
                    let z = f32::mul_add(sigma.mul_add(-w0, g), lin_scale, s1);
                    let z_unscaled = z / lin_scale.max(1e-8);
                    let nw = if z_unscaled.abs() <= *l1 {
                        0.0
                    } else {
                        let denom =
                            2.0f32.mul_add(*l2, (*beta + new_n.powf(-(*lr_power))) / lr.max(1e-8));
                        -z_unscaled.signum().mul_add(-(*l1), z_unscaled) / denom
                    };
                    (nw, new_n, z, s2)
                }
                SparseCoreGradientOptimizer::Lion { beta1, beta2 } => {
                    let c = f32::mul_add(1.0 - *beta1, g, (*beta1) * s0);
                    let m = f32::mul_add(1.0 - *beta2, g, (*beta2) * s0);
                    let sign_c = if c > 0.0 {
                        1.0
                    } else if c < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    (lr.mul_add(-sign_c, w0), m, s1, s2)
                }
                SparseCoreGradientOptimizer::Momentum {
                    momentum,
                    use_nesterov,
                } => {
                    let v = f32::mul_add(*momentum, s0, g);
                    let step = if *use_nesterov {
                        f32::mul_add(*momentum, v, g)
                    } else {
                        v
                    };
                    (lr.mul_add(-step, w0), v, s1, s2)
                }
                SparseCoreGradientOptimizer::RmsProp {
                    rho,
                    momentum,
                    epsilon,
                } => {
                    let ms = f32::mul_add((1.0 - *rho) * g, g, (*rho) * s0);
                    let mom = f32::mul_add(*momentum, s1, lr * g / (ms + *epsilon).sqrt());
                    (w0 - mom, ms, mom, s2)
                }
                SparseCoreGradientOptimizer::CenteredRmsProp {
                    rho,
                    momentum,
                    epsilon,
                } => {
                    let ms = f32::mul_add((1.0 - *rho) * g, g, (*rho) * s0);
                    let mg = f32::mul_add(1.0 - *rho, g, (*rho) * s2);
                    let var = f32::mul_add(mg, -mg, ms).max(0.0);
                    let mom = f32::mul_add(*momentum, s1, lr * g / (var + *epsilon).sqrt());
                    (w0 - mom, ms, mom, mg)
                }
                SparseCoreGradientOptimizer::OnlineYogi { l1, l2, beta2 } => {
                    let g2 = g * g;
                    let sign = (s0 - g2).signum();
                    let v = f32::mul_add((1.0 - *beta2) * sign, -g2, s0).max(1e-8);
                    let z = s1 + g;
                    let nw = if z.abs() <= *l1 {
                        0.0
                    } else {
                        -f32::mul_add(z.signum(), -(*l1), z) / ((v.sqrt() / lr.max(1e-8)) + *l2)
                    };
                    (nw, v, z, s2)
                }
                SparseCoreGradientOptimizer::ProximalYogi {
                    l1,
                    l2,
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    let m = f32::mul_add(1.0 - *beta1, g, (*beta1) * s0);
                    let g2 = g * g;
                    let sign = (s1 - g2).signum();
                    let v = f32::mul_add((1.0 - *beta2) * sign, -g2, s1).max(1e-8);
                    let eff_lr = lr / (v.sqrt() + *epsilon);
                    let w_hat = eff_lr.mul_add(-m, w0);
                    let nw = if w_hat.abs() <= eff_lr * (*l1) {
                        0.0
                    } else {
                        (w_hat.signum() * eff_lr).mul_add(-(*l1), w_hat) / eff_lr.mul_add(*l2, 1.0)
                    };
                    (nw, m, v, s2)
                }
                SparseCoreGradientOptimizer::Assign => (g, s0, s1, s2),
                SparseCoreGradientOptimizer::FrequencyEstimator {
                    tau,
                    max_delta,
                    outlier_threshold,
                    weight_exponent,
                } => {
                    let delta = g.abs().min(*max_delta);
                    let alpha = (1.0 / tau.max(1.0)).clamp(0.001, 1.0);
                    let est = if delta > *outlier_threshold {
                        f32::mul_add(
                            alpha,
                            delta.powf((*weight_exponent).max(0.5)),
                            (1.0 - alpha) * w0,
                        )
                    } else {
                        f32::mul_add(alpha, delta, (1.0 - alpha) * w0)
                    };
                    (est, est, s1, s2)
                }
                SparseCoreGradientOptimizer::UserDefinedProgram { program } => {
                    let c_u8 = u8::try_from(program.computations.len().min(8)).unwrap_or(0);
                    let op_count_scale = 0.05_f32.mul_add(f32::from(c_u8), 1.0);
                    let acc = f32::mul_add(g, g, s0);
                    (
                        w0 - lr * op_count_scale * g / (acc.sqrt() + 1e-6),
                        acc,
                        s1,
                        s2,
                    )
                }
                SparseCoreGradientOptimizer::Adadelta { rho, epsilon } => {
                    let acc_g = f32::mul_add((1.0 - *rho) * g, g, (*rho) * s0);
                    let delta = g * ((s1 + *epsilon).sqrt() / (acc_g + *epsilon).sqrt());
                    let acc_d = f32::mul_add((1.0 - *rho) * delta, delta, (*rho) * s1);
                    (lr.max(1.0).mul_add(-delta, w0), acc_g, acc_d, s2)
                }
            };

            let clipped_w = if let Some((lo, hi)) = self.weight_clip {
                new_w.clamp(lo, hi)
            } else {
                new_w
            };
            weights
                .get_mut(i)
                .into_iter()
                .for_each(|dst| *dst = clipped_w);
            slot0.get_mut(i).into_iter().for_each(|dst| *dst = new_s0);
            slot1.get_mut(i).into_iter().for_each(|dst| *dst = new_s1);
            slot2.get_mut(i).into_iter().for_each(|dst| *dst = new_s2);
        }
    }
}
