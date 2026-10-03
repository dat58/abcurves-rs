use super::planner::{FastPlanner, Intent};
use crate::error::{Error, Result, require};
use crate::model_store;
use crate::renderer::{RendererModel, RendererProfile, RendererStream};
use std::path::Path;

pub const CONTEXT_TICKS: usize = 256;

pub struct StaticPipeline {
    planner: FastPlanner,
    model: RendererModel,
    pub model_seed: u32,
}

pub struct PreparedStream {
    renderer: RendererStream,
    smooth: Vec<[f32; 2]>,
    tick: usize,
    pub head: usize,
}

impl StaticPipeline {
    pub fn from_pretrained(
        model_seed: u32,
        model_dir: Option<&Path>,
        prewarm: bool,
    ) -> Result<Self> {
        let files = model_store::resolve_model_files(model_seed, model_dir, true)?;
        let planner = FastPlanner::open(&files.planner, prewarm)?;
        let model = RendererModel::open(&files.renderer)?;
        Ok(Self {
            planner,
            model,
            model_seed,
        })
    }

    pub fn prepare_renderer_profile(&self, raw_dxdy: &[[i16; 2]]) -> Result<RendererProfile> {
        require(
            raw_dxdy.len() == CONTEXT_TICKS,
            "renderer context must contain exactly 256 integer reports",
        )?;
        RendererProfile::prepare(&self.model, raw_dxdy)
    }

    pub fn planner(&self) -> &FastPlanner {
        &self.planner
    }

    pub fn planner_mut(&mut self) -> &mut FastPlanner {
        &mut self.planner
    }

    pub fn model(&self) -> &RendererModel {
        &self.model
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        prefix_raw_dxdy: &[[f32; 2]],
        profile: &RendererProfile,
        target_rel_at_b: [f64; 2],
        target_radius: f64,
        progress_center: f64,
        planner_seed: u128,
        renderer_event_seed: u64,
        planner_head: Option<usize>,
    ) -> Result<PreparedStream> {
        let intent = self.planner.plan(
            prefix_raw_dxdy,
            target_rel_at_b,
            target_radius,
            progress_center,
            planner_seed,
            planner_head,
        )?;
        self.begin(profile, intent, renderer_event_seed)
    }

    pub fn begin(
        &self,
        profile: &RendererProfile,
        intent: Intent,
        renderer_event_seed: u64,
    ) -> Result<PreparedStream> {
        if intent.smooth_dxdy.is_empty() {
            return Err(Error::InferenceContract(
                "planned intent has no active future ticks".into(),
            ));
        }
        Ok(PreparedStream {
            renderer: profile.begin_stream(renderer_event_seed)?,
            smooth: intent.smooth_dxdy,
            tick: 0,
            head: intent.head,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        prefix_raw_dxdy: &[[f32; 2]],
        profile: &RendererProfile,
        target_rel_at_b: [f64; 2],
        target_radius: f64,
        progress_center: f64,
        seed: u64,
        planner_head: Option<usize>,
    ) -> Result<Vec<[i16; 2]>> {
        let mut stream = self.prepare(
            prefix_raw_dxdy,
            profile,
            target_rel_at_b,
            target_radius,
            progress_center,
            u128::from(seed),
            seed,
            planner_head,
        )?;
        stream.render_remaining(&self.model)
    }
}

impl PreparedStream {
    pub fn duration_ms(&self) -> usize {
        self.smooth.len()
    }

    pub fn complete(&self) -> bool {
        self.tick >= self.smooth.len()
    }

    pub fn step(&mut self, model: &RendererModel) -> Result<[i16; 2]> {
        if self.complete() {
            return Err(Error::Mode("renderer event is complete".into()));
        }
        let report = self.renderer.step(model, self.smooth[self.tick])?;
        self.tick += 1;
        Ok(report)
    }

    pub fn render_remaining(&mut self, model: &RendererModel) -> Result<Vec<[i16; 2]>> {
        let mut out = Vec::with_capacity(self.smooth.len() - self.tick);
        while !self.complete() {
            out.push(self.step(model)?);
        }
        Ok(out)
    }
}
