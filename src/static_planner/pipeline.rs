use super::movement::Movement;
use super::planner::{FastPlanner, Intent};
use super::request::{Draw, Handoff};
use crate::error::{Error, Result, require};
use crate::model_store;
use crate::renderer::{RendererModel, RendererProfile, RendererStream};
use std::path::Path;

pub const CONTEXT_TICKS: usize = 256;

/// Hardware Y is downward-positive, so every native count crossing this
/// boundary is reflected into the canonical x-right/y-up frame the models were
/// trained in, and reflected back on the way out. Dot products, distances and
/// the seam rules are invariant under that reflection, so `Movement`,
/// `BTrigger` and `OnsetDetector` work in whichever frame they are handed, as
/// long as it is one frame throughout.
fn canonical_i16(report: [i16; 2], y_down: bool) -> [i16; 2] {
    if y_down {
        [report[0], report[1].wrapping_neg()]
    } else {
        report
    }
}

fn canonical_f32(pair: [f32; 2], y_down: bool) -> [f32; 2] {
    if y_down { [pair[0], -pair[1]] } else { pair }
}

fn canonical_f64(pair: [f64; 2], y_down: bool) -> [f64; 2] {
    if y_down { [pair[0], -pair[1]] } else { pair }
}

fn canonical_handoff(handoff: Handoff, y_down: bool) -> Handoff {
    Handoff {
        target_rel_at_b: canonical_f64(handoff.target_rel_at_b, y_down),
        ..handoff
    }
}

fn canonical_intent(mut intent: Intent, y_down: bool) -> Intent {
    if y_down {
        for step in &mut intent.smooth_dxdy {
            step[1] = -step[1];
        }
    }
    intent
}

pub struct StaticPipeline {
    planner: FastPlanner,
    model: RendererModel,
    pub model_seed: u32,
    y_down: bool,
}

pub struct PreparedStream {
    renderer: RendererStream,
    smooth: Vec<[f32; 2]>,
    tick: usize,
    y_down: bool,
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
            y_down: true,
        })
    }

    /// Whether native counts crossing this pipeline are downward-positive.
    /// Defaults to `true`, matching hardware and [`CountTransform`].
    ///
    /// [`CountTransform`]: crate::continuous::CountTransform
    pub fn y_down(&self) -> bool {
        self.y_down
    }

    pub fn set_y_down(&mut self, y_down: bool) -> &mut Self {
        self.y_down = y_down;
        self
    }

    pub fn prepare_renderer_profile(&self, raw_dxdy: &[[i16; 2]]) -> Result<RendererProfile> {
        require(
            raw_dxdy.len() == CONTEXT_TICKS,
            "renderer context must contain exactly 256 integer reports",
        )?;
        let canonical: Vec<[i16; 2]> = raw_dxdy
            .iter()
            .map(|report| canonical_i16(*report, self.y_down))
            .collect();
        RendererProfile::prepare(&self.model, &canonical)
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
        let prefix: Vec<[f32; 2]> = prefix_raw_dxdy
            .iter()
            .map(|pair| canonical_f32(*pair, self.y_down))
            .collect();
        let intent = self
            .planner
            .plan(&prefix, canonical_handoff(handoff, self.y_down), draw)?;
        self.open(profile, intent, draw.renderer_seed)
    }

    /// Open a stream for an [`Intent`] this pipeline produced.
    pub fn begin(
        &self,
        profile: &RendererProfile,
        intent: Intent,
        renderer_event_seed: u64,
    ) -> Result<PreparedStream> {
        self.open(
            profile,
            canonical_intent(intent, self.y_down),
            renderer_event_seed,
        )
    }

    fn open(
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
            y_down: self.y_down,
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
        let handoff = canonical_handoff(movement.handoff()?, self.y_down);
        let prefix: Vec<[f32; 2]> = movement
            .prefix()
            .iter()
            .map(|pair| canonical_f32(*pair, self.y_down))
            .collect();
        Ok(self
            .planner
            .plan_all(&prefix, handoff)?
            .into_iter()
            .map(|intent| canonical_intent(intent, self.y_down))
            .collect())
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
        Ok(canonical_i16(report, self.y_down))
    }

    pub fn render_remaining(&mut self, model: &RendererModel) -> Result<Vec<[i16; 2]>> {
        let mut out = Vec::with_capacity(self.smooth.len() - self.tick);
        while !self.complete() {
            out.push(self.step(model)?);
        }
        Ok(out)
    }
}
