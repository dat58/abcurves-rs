use super::constants::COMMON_RADIANS_PER_COUNT;
use super::runtime::{AdvanceError, MovementRuntime};
use super::stream::ContinuousOptions;
use crate::error::{Error, Result};
use crate::model_store;
use crate::renderer::{RendererModel, RendererProfile, RendererStream};
use std::fmt;
use std::path::PathBuf;

/// Fixed per-axis conversion between native and common angular counts.
#[derive(Clone, Copy, Debug)]
pub struct CountTransform {
    radians_per_count: [f64; 2],
    pub y_down: bool,
}

impl CountTransform {
    pub fn new(radians_per_count: f64, y_down: bool) -> Result<Self> {
        Self::per_axis([radians_per_count, radians_per_count], y_down)
    }

    pub fn per_axis(radians_per_count: [f64; 2], y_down: bool) -> Result<Self> {
        if !radians_per_count
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(Error::InferenceContract(
                "radians_per_count must be one or two finite positive scales".into(),
            ));
        }
        let transform = Self {
            radians_per_count,
            y_down,
        };
        if !transform
            .common_per_native()
            .iter()
            .all(|value| value.is_finite() && *value != 0.0)
        {
            return Err(Error::Numerical(
                "Angular conversion exceeds finite coordinate range".into(),
            ));
        }
        Ok(transform)
    }

    pub fn radians_per_count(&self) -> [f64; 2] {
        self.radians_per_count
    }

    pub fn common_per_native(&self) -> [f64; 2] {
        let flip = if self.y_down { -1.0 } else { 1.0 };
        [
            self.radians_per_count[0] / COMMON_RADIANS_PER_COUNT,
            self.radians_per_count[1] / COMMON_RADIANS_PER_COUNT * flip,
        ]
    }

    pub fn to_common(&self, report: [i16; 2]) -> [f64; 2] {
        let scale = self.common_per_native();
        [
            f64::from(report[0]) * scale[0],
            f64::from(report[1]) * scale[1],
        ]
    }

    pub fn to_native(&self, displacement: [f64; 2]) -> [f64; 2] {
        let scale = self.common_per_native();
        [displacement[0] / scale[0], displacement[1] / scale[1]]
    }
}

#[derive(Clone, Debug, Default)]
pub struct PipelineAdvance {
    pub time_us: Vec<i64>,
    pub xy: Vec<[f64; 2]>,
    pub reports: Vec<[i16; 2]>,
    pub rendered_xy: Vec<[f64; 2]>,
}

#[derive(Debug)]
pub struct PipelineFailure {
    pub message: String,
    pub partial: PipelineAdvance,
}

#[derive(Debug)]
pub enum PipelineError {
    Contract(Error),
    Failed(Box<PipelineFailure>),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Contract(error) => write!(f, "{error}"),
            PipelineError::Failed(failure) => write!(f, "{}", failure.message),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<Error> for PipelineError {
    fn from(value: Error) -> Self {
        PipelineError::Contract(value)
    }
}

pub struct PipelineOptions {
    pub planner: ContinuousOptions,
    pub transform: CountTransform,
    pub renderer_seed: u64,
    pub model_dir: Option<PathBuf>,
    pub observed_xy: Option<[f64; 2]>,
}

impl PipelineOptions {
    pub fn new(transform: CountTransform) -> Self {
        Self {
            planner: ContinuousOptions::default(),
            transform,
            renderer_seed: 7,
            model_dir: None,
            observed_xy: None,
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.planner = self.planner.seed(seed);
        self
    }

    pub fn renderer_seed(mut self, seed: u64) -> Self {
        self.renderer_seed = seed;
        self
    }

    pub fn initial_xy(mut self, initial_xy: [f64; 2]) -> Self {
        self.planner = self.planner.initial_xy(initial_xy);
        self
    }

    pub fn history(mut self, history: Vec<[f64; 2]>) -> Self {
        self.planner = self.planner.history(history);
        self
    }

    pub fn observed_xy(mut self, observed_xy: [f64; 2]) -> Self {
        self.observed_xy = Some(observed_xy);
        self
    }
}

pub struct ContinuousPipeline {
    pub transform: CountTransform,
    movement: MovementRuntime,
    model: RendererModel,
    profile: RendererProfile,
    renderer: RendererStream,
    renderer_seed: u64,
    previous: [f64; 2],
    observed_offset: [f64; 2],
    rendered: [f64; 2],
    failed: bool,
}

impl ContinuousPipeline {
    pub fn load(profile_reports: &[[i16; 2]], options: PipelineOptions) -> Result<Self> {
        let root = options
            .model_dir
            .clone()
            .unwrap_or_else(model_store::default_model_dir);
        let mut planner_options = options.planner;
        if planner_options.assets.is_none() {
            planner_options.assets = Some(root.join("continuous"));
        }
        let movement = planner_options.load()?;
        let model = RendererModel::from_release(Some(&root))?;
        // Prepare the learned texture state in canonical x-right/y-up counts.
        let canonical: Vec<[i16; 2]> = profile_reports
            .iter()
            .map(|report| {
                if options.transform.y_down {
                    [report[0], -report[1]]
                } else {
                    *report
                }
            })
            .collect();
        let profile = RendererProfile::prepare(&model, &canonical)?;
        let renderer = profile.begin_stream(options.renderer_seed)?;
        let previous = movement.current_xy();
        let observed_offset = match options.observed_xy {
            Some(observed) => [observed[0] - previous[0], observed[1] - previous[1]],
            None => [0.0, 0.0],
        };
        Ok(Self {
            transform: options.transform,
            movement,
            model,
            profile,
            renderer,
            renderer_seed: options.renderer_seed,
            previous,
            observed_offset,
            rendered: [
                previous[0] + observed_offset[0],
                previous[1] + observed_offset[1],
            ],
            failed: false,
        })
    }

    /// Reset both state machines; retain profile, sensitivity and seed defaults.
    pub fn reset(
        &mut self,
        renderer_seed: Option<u64>,
        initial_xy: Option<[f64; 2]>,
        history: Option<&[[f64; 2]]>,
        seed: Option<u64>,
        observed_xy: Option<[f64; 2]>,
    ) -> Result<()> {
        let event_seed = renderer_seed.unwrap_or(self.renderer_seed);
        let stream = self.profile.begin_stream(event_seed)?;
        self.movement.reset(initial_xy, history, seed)?;
        self.renderer = stream;
        self.renderer_seed = event_seed;
        self.previous = self.movement.current_xy();
        if let Some(observed) = observed_xy {
            self.observed_offset = [
                observed[0] - self.previous[0],
                observed[1] - self.previous[1],
            ];
        }
        self.rendered = [
            self.previous[0] + self.observed_offset[0],
            self.previous[1] + self.observed_offset[1],
        ];
        self.failed = false;
        Ok(())
    }

    pub fn update_target(&mut self, xy: [f64; 2], timestamp_us: i64) -> Result<()> {
        if self.failed {
            return Err(Error::Mode("Pipeline failed; reset before updating targets".into()));
        }
        self.movement.update_target(xy, timestamp_us)
    }

    pub fn advance(
        &mut self,
        timestamp_us: i64,
    ) -> std::result::Result<PipelineAdvance, PipelineError> {
        if self.failed {
            return Err(PipelineError::Contract(Error::Mode(
                "Pipeline failed; inspect partial output and reset".into(),
            )));
        }
        let (planned, mut failure) = match self.movement.advance(timestamp_us) {
            Ok(block) => (block, None),
            Err(AdvanceError::Contract(error)) => return Err(PipelineError::Contract(error)),
            Err(AdvanceError::Failed(error)) => {
                let message = error.message.clone();
                (error.partial.clone(), Some(message))
            }
        };

        let scale = self.transform.common_per_native();
        let mut result = PipelineAdvance::default();
        for (index, point) in planned.xy.iter().enumerate() {
            let delta = [
                ((point[0] - self.previous[0]) / scale[0].abs()) as f32,
                ((point[1] - self.previous[1]) / scale[1].abs()) as f32,
            ];
            let mut report = match self.renderer.step(&self.model, delta) {
                Ok(report) => report,
                Err(error) => {
                    failure = Some(error.to_string());
                    break;
                }
            };
            if self.transform.y_down {
                report[1] = -report[1];
            }
            self.previous = *point;
            let common = self.transform.to_common(report);
            self.rendered[0] += common[0];
            self.rendered[1] += common[1];
            result.time_us.push(planned.time_us[index]);
            result.xy.push(*point);
            result.reports.push(report);
            result.rendered_xy.push(self.rendered);
        }

        match failure {
            Some(message) => {
                self.failed = true;
                Err(PipelineError::Failed(Box::new(PipelineFailure {
                    message,
                    partial: result,
                })))
            }
            None => Ok(result),
        }
    }

    pub fn current_xy(&self) -> [f64; 2] {
        self.movement.current_xy()
    }

    pub fn rendered_xy(&self) -> [f64; 2] {
        self.rendered
    }

    pub fn needs_plan(&self) -> bool {
        self.movement.needs_plan()
    }

    pub fn movement(&self) -> &MovementRuntime {
        &self.movement
    }

    pub fn movement_mut(&mut self) -> &mut MovementRuntime {
        &mut self.movement
    }
}
