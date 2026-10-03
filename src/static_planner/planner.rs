use super::prodmp::{ComponentCache, ProDMPConfig};
use super::summary::{SUMMARY_LEN, SummaryNormalizer, distance_over_radius, raw_summary62};
use super::tcn::{self, CHANNELS, INPUT_CHANNELS, KERNEL, TcnWeights, TcnWorkspace, WINDOW};
use crate::error::{Error, Result, require};
use crate::io::{Checkpoint, Value};
use crate::model_store;
use crate::nn::NeuralInference;
use crate::rng::head_from_seed;
use std::path::Path;

pub const HEADS: usize = 16;
pub const OUT_DIM: usize = 43;
pub const HORIZON: usize = 1000;
pub const PREFIX_LEN: usize = 160;

pub const CANONICAL_FEATURE_NAMES: [&str; SUMMARY_LEN] = [
    "prefix_duration_ms", "prefix_dx", "prefix_dy", "prefix_distance", "prefix_path_length",
    "prefix_straightness", "prefix_target_axis_progress", "prefix_lateral_error", "target_rel_x",
    "target_rel_y", "target_distance", "target_radius", "distance_over_radius", "target_unit_x",
    "target_unit_y", "progress_context", "inside_target_at_B", "near_target_flag",
    "prefix_crossing_state", "prefix_overshoot_state", "prefix_min_target_distance_over_radius",
    "prefix_near_target_rate", "last_velocity_x", "last_velocity_y", "movement_dir_x",
    "movement_dir_y", "speed_at_B", "mean_prefix_speed", "peak_prefix_speed", "prefix_speed_std",
    "relative_speed_distance", "relative_speed_radius", "velocity_toward_target",
    "velocity_tangential_to_target", "direction_alignment_cos", "direction_alignment_sin",
    "direction_error_deg", "accel_at_B", "accel_toward_target", "accel_lateral",
    "recent_speed_slope", "recent_accel_slope", "deceleration_indicator", "speed_drop_recent",
    "jerk_at_B", "discontinuity_speed_jump", "recent_zero_rate", "recent_sign_flip_rate",
    "recent_direction_change_rate", "active_motion_flag", "stabilization_like_flag",
    "prefix_shape_curvature_signed", "prefix_shape_curvature_abs",
    "prefix_shape_dir_change_mean_deg", "prefix_shape_dir_change_slope",
    "prefix_shape_approach_cos", "prefix_shape_approach_sin", "prefix_shape_speed_bin_0",
    "prefix_shape_speed_bin_1", "prefix_shape_speed_bin_2", "prefix_shape_speed_bin_3",
    "prefix_shape_recent_lateral_drift",
];

#[derive(Clone, Debug)]
pub struct Intent {
    pub smooth_dxdy: Vec<[f32; 2]>,
    pub duration_ms: usize,
    pub head: usize,
}

pub struct FastPlanner {
    weights: TcnWeights,
    workspace: TcnWorkspace,
    normalizer: SummaryNormalizer,
    prefix_mean: [f32; 2],
    prefix_std: [f32; 2],
    y_mean: Vec<f32>,
    y_std: Vec<f32>,
    decoder: ComponentCache,
    backend: NeuralInference,
    #[cfg(feature = "candle")]
    candle: Option<super::candle::CandleTcn>,
    pub seed: u32,
}

fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = values[row * columns + column];
        }
    }
    out
}

impl FastPlanner {
    pub fn from_pretrained(model_seed: u32, model_dir: Option<&Path>, prewarm: bool) -> Result<Self> {
        let files = model_store::resolve_model_files(model_seed, model_dir, true)?;
        let mut planner = Self::open(&files.planner, prewarm)?;
        planner.seed = model_seed;
        Ok(planner)
    }

    pub fn open(path: impl AsRef<Path>, prewarm: bool) -> Result<Self> {
        Self::open_with(path, prewarm, NeuralInference::Native)
    }

    pub fn backend(&self) -> NeuralInference {
        self.backend
    }

    pub fn open_with(
        path: impl AsRef<Path>,
        prewarm: bool,
        backend: NeuralInference,
    ) -> Result<Self> {
        let checkpoint = Checkpoint::open(path)?;
        let root = checkpoint.root();
        require(
            root.entry("schema")?.as_str()? == "abcurves.planner.v2",
            "unsupported planner checkpoint schema",
        )?;
        require(
            root.entry("heads")?.as_i64()? == HEADS as i64,
            "release Planner must have 16 heads",
        )?;
        require(
            root.entry("horizon")?.as_i64()? == HORIZON as i64,
            "release Planner horizon differs",
        )?;
        let config = root.entry("planner_config")?;
        require(
            config.entry("prefix_len")?.as_i64()? == PREFIX_LEN as i64,
            "release Planner prefix differs",
        )?;
        let contract = root.entry("seam_contract")?;
        let trigger = contract.entry("trigger")?;
        let thresholds = trigger.entry("thresholds")?.as_sequence()?;
        require(
            contract.entry("schema")?.as_str()? == "abcurves.causal_onset_b.v1"
                && trigger.entry("reference")?.as_str()? == "edge"
                && thresholds.len() == 1
                && (thresholds[0].as_f64()? - 0.8).abs() < 1e-12,
            "Planner causal B contract differs from the release",
        )?;
        require(
            root.entry("prefix_representation")?.entry("name")?.as_str()? == "raw",
            "release Planner prefix representation differs",
        )?;

        let state = root.entry("model_state_dict")?;
        let tensor = |value: &Value| -> Result<Vec<f32>> { checkpoint.tensor(value.as_tensor()?)?.to_f32() };
        let from_state = |name: &str| -> Result<Vec<f32>> { tensor(state.entry(name)?) };

        let input_weight = from_state("encoder.input.weight")?;
        let mut conv_weight = vec![0.0f32; 6 * KERNEL * CHANNELS * CHANNELS];
        let mut conv_bias = vec![0.0f32; 6 * CHANNELS];
        let mut norm_weight = vec![0.0f32; 6 * CHANNELS];
        let mut norm_bias = vec![0.0f32; 6 * CHANNELS];
        for block in 0..3 {
            for inner in 0..2 {
                let index = block * 2 + inner;
                let suffix = inner + 1;
                let raw = from_state(&format!("encoder.blocks.{block}.conv{suffix}.weight"))?;
                // im2col layout: col[i, kk*C + ci] pairs with conv_w[kk*C + ci, co]
                for outgoing in 0..CHANNELS {
                    for incoming in 0..CHANNELS {
                        for tap in 0..KERNEL {
                            let source = (outgoing * CHANNELS + incoming) * KERNEL + tap;
                            let target = index * KERNEL * CHANNELS * CHANNELS
                                + (tap * CHANNELS + incoming) * CHANNELS
                                + outgoing;
                            conv_weight[target] = raw[source];
                        }
                    }
                }
                conv_bias[index * CHANNELS..(index + 1) * CHANNELS]
                    .copy_from_slice(&from_state(&format!("encoder.blocks.{block}.conv{suffix}.bias"))?);
                norm_weight[index * CHANNELS..(index + 1) * CHANNELS]
                    .copy_from_slice(&from_state(&format!("encoder.blocks.{block}.norm{suffix}.weight"))?);
                norm_bias[index * CHANNELS..(index + 1) * CHANNELS]
                    .copy_from_slice(&from_state(&format!("encoder.blocks.{block}.norm{suffix}.bias"))?);
            }
        }

        let summary_names: Vec<String> = root
            .entry("summary_feature_names")?
            .as_sequence()?
            .iter()
            .map(|value| value.as_str().map(str::to_string))
            .collect::<Result<Vec<_>>>()?;
        let reorder: Result<Vec<usize>> = summary_names
            .iter()
            .map(|name| {
                CANONICAL_FEATURE_NAMES
                    .iter()
                    .position(|known| known == name)
                    .ok_or_else(|| {
                        Error::InferenceContract(format!(
                            "compiled summary does not know checkpoint feature {name}"
                        ))
                    })
            })
            .collect();
        let normalizer = SummaryNormalizer::new(
            reorder?,
            tensor(root.entry("summary_mean")?)?,
            tensor(root.entry("summary_std")?)?,
        );

        let prefix_mean_raw = tensor(root.entry("prefix_mean")?)?;
        let prefix_std_raw = tensor(root.entry("prefix_std")?)?;
        let mut prefix_std = [prefix_std_raw[0], prefix_std_raw[1]];
        for value in prefix_std.iter_mut() {
            if *value < 1e-6 {
                *value = 1.0;
            }
        }

        let prodmp = root.entry("prodmp")?;
        let config = ProDMPConfig {
            n_basis: prodmp.entry("n_basis")?.as_usize()?,
            alpha: prodmp.entry("alpha")?.as_f64()?,
            alpha_phase: prodmp.entry("alpha_phase")?.as_f64()?,
            ridge: prodmp.entry("ridge")?.as_f64()?,
            ..ProDMPConfig::default()
        };
        let mut decoder = ComponentCache::new(config, HORIZON);
        if prewarm {
            decoder.prewarm();
        }

        let summary_dim = root.entry("summary_dim")?.as_usize()?;
        let weights = TcnWeights {
            input_weight: transpose(&input_weight, CHANNELS, INPUT_CHANNELS),
            input_bias: from_state("encoder.input.bias")?,
            conv_weight,
            conv_bias,
            norm_weight,
            norm_bias,
            summary_weight: from_state("encoder.summary.0.weight")?,
            summary_bias: from_state("encoder.summary.0.bias")?,
            trunk_weight: from_state("trunk.0.weight")?,
            trunk_bias: from_state("trunk.0.bias")?,
            head_weight: from_state("out.weight")?,
            head_bias: from_state("out.bias")?,
            heads: HEADS,
            out_dim: OUT_DIM,
            summary_dim,
        };

        #[cfg(feature = "candle")]
        let candle = match backend {
            NeuralInference::Candle => Some(super::candle::CandleTcn::load(&checkpoint, summary_dim)?),
            NeuralInference::Native => None,
        };
        #[cfg(not(feature = "candle"))]
        if backend == NeuralInference::Candle {
            return Err(Error::InferenceContract(
                "the candle backend requires the 'candle' feature".into(),
            ));
        }

        Ok(Self {
            workspace: TcnWorkspace::new(summary_dim),
            weights,
            normalizer,
            prefix_mean: [prefix_mean_raw[0], prefix_mean_raw[1]],
            prefix_std,
            y_mean: tensor(root.entry("y_mean")?)?,
            y_std: tensor(root.entry("y_std")?)?,
            decoder,
            backend,
            #[cfg(feature = "candle")]
            candle,
            seed: root.entry("seed")?.as_i64()? as u32,
        })
    }

    pub fn normalizer(&self) -> &SummaryNormalizer {
        &self.normalizer
    }

    pub fn prefix_len(&self) -> usize {
        PREFIX_LEN
    }

    pub fn raw_summary(
        &self,
        prefix: &[[f32; 2]],
        target_rel_at_b: [f64; 2],
        target_radius: f64,
        progress_center: f64,
    ) -> [f64; SUMMARY_LEN] {
        raw_summary62(
            prefix,
            target_rel_at_b[0] as f32,
            target_rel_at_b[1] as f32,
            target_radius as f32,
            progress_center as f32,
            distance_over_radius(target_rel_at_b, target_radius),
        )
    }

    pub fn plan(
        &mut self,
        prefix_raw_dxdy: &[[f32; 2]],
        target_rel_at_b: [f64; 2],
        target_radius: f64,
        progress_center: f64,
        seed: u128,
        head: Option<usize>,
    ) -> Result<Intent> {
        require(
            !prefix_raw_dxdy.is_empty()
                && prefix_raw_dxdy
                    .iter()
                    .flat_map(|row| row.iter())
                    .all(|value| value.is_finite()),
            "Planner prefix must be a non-empty finite array with shape (P, 2)",
        )?;
        require(
            target_rel_at_b.iter().all(|value| value.is_finite()),
            "target_rel_at_B must contain two finite count-space values",
        )?;
        require(
            target_radius.is_finite() && target_radius > 0.0,
            "target_radius must be positive",
        )?;
        require(
            progress_center.is_finite() && (0.0..=1.0).contains(&progress_center),
            "progress_center must lie in [0, 1]",
        )?;

        let start = prefix_raw_dxdy.len().saturating_sub(PREFIX_LEN);
        let represented = &prefix_raw_dxdy[start..];
        let raw = self.raw_summary(represented, target_rel_at_b, target_radius, progress_center);
        let mut summary = vec![0.0f32; self.weights.summary_dim];
        self.normalizer.apply(&raw, &mut summary);

        // Summaries use the full represented prefix; the TCN only consumes its
        // receptive window, including normalized-space zero padding.
        let mut window = vec![0.0f32; WINDOW * INPUT_CHANNELS];
        let take = WINDOW.min(represented.len());
        for step in 0..take {
            let source = represented[represented.len() - take + step];
            let row = (WINDOW - take + step) * INPUT_CHANNELS;
            window[row] = (source[0] - self.prefix_mean[0]) / self.prefix_std[0];
            window[row + 1] = (source[1] - self.prefix_mean[1]) / self.prefix_std[1];
            window[row + 2] = 1.0;
        }
        let boundary_velocity = [
            f64::from(represented[represented.len() - 1][0]),
            f64::from(represented[represented.len() - 1][1]),
        ];

        let chosen = match head {
            Some(value) => {
                require(value < HEADS, "Planner head is out of range")?;
                value
            }
            None => head_from_seed(seed, HEADS as u32),
        };

        let mut prediction = vec![0.0f32; OUT_DIM];
        #[cfg(feature = "candle")]
        if let Some(engine) = self.candle.as_ref() {
            engine.forward(&window, &summary, chosen, &mut prediction)?;
        }
        if self.backend == NeuralInference::Native {
            let mut head_weight = vec![0.0f32; CHANNELS * OUT_DIM];
            for channel in 0..CHANNELS {
                for slot in 0..OUT_DIM {
                    head_weight[channel * OUT_DIM + slot] =
                        self.weights.head_weight[(chosen * OUT_DIM + slot) * CHANNELS + channel];
                }
            }
            let head_bias = &self.weights.head_bias[chosen * OUT_DIM..(chosen + 1) * OUT_DIM];
            tcn::forward(
                &self.weights,
                &mut self.workspace,
                &window,
                &summary,
                &head_weight,
                head_bias,
                &mut prediction,
            );
        }

        let mut raw_output = vec![0.0f64; OUT_DIM];
        for slot in 0..OUT_DIM {
            raw_output[slot] = f64::from(prediction[slot]) * f64::from(self.y_std[slot])
                + f64::from(self.y_mean[slot]);
        }
        let duration = raw_output[OUT_DIM - 1]
            .clamp(0.0, (HORIZON as f64).ln())
            .exp();
        let duration = (duration.round_ties_even() as i64).clamp(1, HORIZON as i64) as usize;
        let components = self.decoder.components(duration).clone();
        let mut smooth = vec![[0.0f64; 2]; duration];
        self.decoder.prodmp().generate_deltas(
            &components,
            &raw_output[..OUT_DIM - 1],
            boundary_velocity,
            &mut smooth,
        );

        Ok(Intent {
            smooth_dxdy: smooth
                .into_iter()
                .map(|row| [row[0] as f32, row[1] as f32])
                .collect(),
            duration_ms: duration,
            head: chosen,
        })
    }
}
