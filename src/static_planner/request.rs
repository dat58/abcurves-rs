use super::seam::BFire;
use crate::error::{Result, require};

/// The geometry at B, which is everything the planner needs to know about
/// where the handoff happened.
///
/// In normal use all three values arrive together, from the [`BFire`] that
/// [`BTrigger`](super::BTrigger) emits at the handoff, so you never assemble
/// them by hand:
///
/// ```no_run
/// use abcurves::static_planner::{BEvent, BTrigger, Draw, Handoff, StaticPipeline};
///
/// # fn main() -> abcurves::Result<()> {
/// # let (observed, target_rel_at_a, radius): (Vec<[i16; 2]>, [f64; 2], f64) =
/// #     (Vec::new(), [0.0, 0.0], 1.0);
/// # let profile_window: Vec<[i16; 2]> = Vec::new();
/// let mut trigger = BTrigger::recommended();
/// trigger.arm(target_rel_at_a, radius)?;
///
/// let mut moved = [0.0f64; 2];
/// for (index, delta) in observed.iter().enumerate() {
///     moved[0] += f64::from(delta[0]);
///     moved[1] += f64::from(delta[1]);
///     let target_now = [target_rel_at_a[0] - moved[0], target_rel_at_a[1] - moved[1]];
///     if let Some(BEvent::Fire(fire)) =
///         trigger.push_tick(f64::from(delta[0]), f64::from(delta[1]), target_now, radius)?
///     {
///         let prefix: Vec<[f32; 2]> = observed[..=index]
///             .iter()
///             .map(|row| [f32::from(row[0]), f32::from(row[1])])
///             .collect();
///         let mut pipeline = StaticPipeline::from_pretrained(7, None, true)?;
///         let profile = pipeline.prepare_renderer_profile(&profile_window)?;
///         let reports =
///             pipeline.generate(&prefix, &profile, Handoff::from(&fire), Draw::seeded(2026))?;
///         println!("{} integer reports", reports.len());
///         break;
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Handoff {
    /// Vector from B to the target centre, in canonical native counts with
    /// X right and Y up. Not the absolute target position.
    pub target_rel_at_b: [f64; 2],
    /// Target radius in the same counts. Must be positive.
    pub target_radius: f64,
    /// How far the observed movement travelled towards the target *centre*,
    /// as a fraction in `[0, 1]`.
    ///
    /// The trigger fires on *edge* progress, but the model was trained on the
    /// centre-referenced number, and the two differ a lot: a handoff at 0.906
    /// edge progress can be only 0.664 of the way to the centre.
    pub progress_center: f64,
}

impl Handoff {
    pub fn new(
        target_rel_at_b: [f64; 2],
        target_radius: f64,
        progress_center: f64,
    ) -> Result<Self> {
        let handoff = Self {
            target_rel_at_b,
            target_radius,
            progress_center,
        };
        handoff.validate()?;
        Ok(handoff)
    }

    /// Derive the handoff from the geometry at A plus the motion observed
    /// since, for callers that do not drive a [`BTrigger`](super::BTrigger).
    ///
    /// `target_rel_at_a` points from A to the target centre; `moved_since_a` is
    /// the summed displacement up to and including the closed B bin.
    pub fn from_observed(
        target_rel_at_a: [f64; 2],
        target_radius: f64,
        moved_since_a: [f64; 2],
    ) -> Result<Self> {
        let distance = target_rel_at_a[0].hypot(target_rel_at_a[1]);
        require(
            distance > 1e-9,
            "target_rel_at_A must be a nonzero vector towards the target",
        )?;
        let along = (moved_since_a[0] * target_rel_at_a[0] + moved_since_a[1] * target_rel_at_a[1])
            / distance;
        Self::new(
            [
                target_rel_at_a[0] - moved_since_a[0],
                target_rel_at_a[1] - moved_since_a[1],
            ],
            target_radius,
            (along / distance).clamp(0.0, 1.0),
        )
    }

    pub fn validate(&self) -> Result<()> {
        require(
            self.target_rel_at_b.iter().all(|value| value.is_finite()),
            "target_rel_at_B must contain two finite count-space values",
        )?;
        require(
            self.target_radius.is_finite() && self.target_radius > 0.0,
            "target_radius must be positive",
        )?;
        require(
            self.progress_center.is_finite() && (0.0..=1.0).contains(&self.progress_center),
            "progress_center must lie in [0, 1]",
        )
    }
}

impl From<&BFire> for Handoff {
    fn from(fire: &BFire) -> Self {
        Self {
            target_rel_at_b: fire.target_rel_at_b,
            target_radius: fire.target_radius,
            progress_center: fire.progress_center,
        }
    }
}

impl From<BFire> for Handoff {
    fn from(fire: BFire) -> Self {
        Self::from(&fire)
    }
}

/// Which of the sixteen heads to draw, and which texture draw to render with.
///
/// These select one sample, not a model. The trained variant is chosen by the
/// `model_seed` passed when loading the planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Draw {
    /// Seeds the uniform head draw. Ignored when `head` is set.
    pub planner_seed: u128,
    /// Seeds the Renderer's per-event random stream.
    pub renderer_seed: u64,
    /// Pin one head, for tests and inspection. The release rule is to leave
    /// this `None` so the seed picks uniformly among the sixteen.
    pub head: Option<usize>,
}

impl Draw {
    /// One seed for both the head and the texture, as the reference
    /// `generate` does.
    pub fn seeded(seed: u64) -> Self {
        Self {
            planner_seed: u128::from(seed),
            renderer_seed: seed,
            head: None,
        }
    }

    /// Seed the head and the texture independently.
    pub fn split(planner_seed: u128, renderer_seed: u64) -> Self {
        Self {
            planner_seed,
            renderer_seed,
            head: None,
        }
    }

    pub fn head(mut self, head: usize) -> Self {
        self.head = Some(head);
        self
    }
}

impl Default for Draw {
    fn default() -> Self {
        Self::seeded(7)
    }
}
