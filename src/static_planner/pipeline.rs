use super::movement::Movement;
use super::planner::{FastPlanner, Intent};
use super::request::{Draw, Handoff};
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

    /// Plan the finish and open a Renderer event for it.
    ///
    /// Step the returned stream once per millisecond, or drain it with
    /// [`PreparedStream::render_remaining`].
    pub fn prepare(
        &mut self,
        prefix_raw_dxdy: &[[f32; 2]],
        profile: &RendererProfile,
        handoff: Handoff,
        draw: Draw,
    ) -> Result<PreparedStream> {
        let intent = self.planner.plan(prefix_raw_dxdy, handoff, draw)?;
        self.begin(profile, intent, draw.renderer_seed)
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

    /// Continue a movement given only A, the target, the radius and the
    /// reports observed so far.
    ///
    /// B is the last sample in the movement. An ineligible handoff is an
    /// error naming the rule it failed and the measured value.
    pub fn continue_movement(
        &mut self,
        movement: &Movement,
        profile: &RendererProfile,
        draw: Draw,
    ) -> Result<Vec<[i16; 2]>> {
        let handoff = movement.handoff()?;
        self.generate(movement.prefix(), profile, handoff, draw)
    }

    /// Every head's finish for the same handoff, in head order, unrendered.
    ///
    /// The sixteen heads share the trunk, so this costs barely more than one
    /// plan. Read [`Intent::duration_ms`] to choose, then hand the winner to
    /// [`begin`](Self::begin) with the renderer seed; the result is identical
    /// to having drawn that head in the first place.
    pub fn plan_movement_all(&mut self, movement: &Movement) -> Result<Vec<Intent>> {
        let handoff = movement.handoff()?;
        self.planner.plan_all(movement.prefix(), handoff)
    }

    /// Every head's finish, rendered.
    ///
    /// Rendering is the expensive half and it is paid sixteen times here, so
    /// prefer [`plan_movement_all`](Self::plan_movement_all) when the choice
    /// can be made from the plan. Each entry renders from its own copy of the
    /// profile, so the streams do not interfere and entry `k` equals what
    /// [`continue_movement`](Self::continue_movement) returns for
    /// `draw.head(k)`.
    pub fn continue_movement_all(
        &mut self,
        movement: &Movement,
        profile: &RendererProfile,
        draw: Draw,
    ) -> Result<Vec<Vec<[i16; 2]>>> {
        self.plan_movement_all(movement)?
            .into_iter()
            .map(|intent| {
                self.begin(profile, intent, draw.renderer_seed)?
                    .render_remaining(&self.model)
            })
            .collect()
    }

    /// Plan the finish and render it to completion in one call.
    pub fn generate(
        &mut self,
        prefix_raw_dxdy: &[[f32; 2]],
        profile: &RendererProfile,
        handoff: Handoff,
        draw: Draw,
    ) -> Result<Vec<[i16; 2]>> {
        let mut stream = self.prepare(prefix_raw_dxdy, profile, handoff, draw)?;
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
