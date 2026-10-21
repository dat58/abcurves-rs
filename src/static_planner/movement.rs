use super::request::Handoff;
use super::seam::BConfig;
use crate::error::{Error, Result};
use std::ops::RangeInclusive;

/// Which seam policy decides whether B is a usable handoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy {
    /// Prefer the 90 percent recommendation, fall back to the 80 percent
    /// training contract when the geometry cannot satisfy it.
    Auto,
    /// The qualified 90 percent inference setting. Lands better but needs
    /// `|BC| - radius >= 8`, which rules out most short continuations.
    Recommended,
    /// The 80 percent contract the model was trained and measured under.
    Contract,
    Custom(BConfig),
}

impl Default for Policy {
    fn default() -> Self {
        Self::Auto
    }
}

impl Policy {
    fn candidates(self) -> Vec<BConfig> {
        match self {
            Policy::Auto => vec![BConfig::recommended(), BConfig::default()],
            Policy::Recommended => vec![BConfig::recommended()],
            Policy::Contract => vec![BConfig::default()],
            Policy::Custom(config) => vec![config],
        }
    }
}

/// One target-directed movement in absolute coordinates.
///
/// Supply what an application already has: where the movement started, where
/// the target is, how big it is, and the reports observed so far. B is the last
/// sample supplied — the movement from A to B has already physically happened,
/// so the library validates that point rather than choosing its own.
///
/// # Frame
///
/// Every value here is in native counts in whichever frame the capture uses,
/// and the seam rules are reflection-invariant, so `Movement` never needs to
/// know which one it is. [`StaticPipeline`](super::StaticPipeline) does: it
/// reflects into the canonical frame on the way to the model and back on the
/// way out, controlled by
/// [`set_y_down`](super::StaticPipeline::set_y_down), which defaults to the
/// hardware convention of downward-positive Y.
///
/// # Radius
///
/// `radius` is the "counted as a hit" radius in native counts: the target
/// radius on screen divided by pixels per count. It is not decoration. The
/// model reads it through `distance_over_radius`, `prefix_lateral / radius`,
/// `speed_at_B / radius` and the edge-progress denominator, so it tells the
/// model how precise it has to be. Training spans roughly 2 to 90 counts, with
/// radii 2 to 20 deliberately oversampled as tiny targets.
///
/// # The short continuation floor
///
/// Three seam rules depend only on `|BC|` and the radius, whatever path was
/// taken from A: `|BC| >= 8`, `|BC| > radius`, and under the 90 percent policy
/// `|BC| - radius >= 8`. A continuation with less than 8 counts left cannot be
/// planned under any policy; [`Movement::eligible_remaining`] reports the
/// window before you capture, instead of leaving you to discover a refusal.
#[derive(Clone, Debug)]
pub struct Movement {
    pub a: [f64; 2],
    pub target: [f64; 2],
    pub radius: f64,
    prefix: Vec<[f32; 2]>,
    moved: [f64; 2],
    policy: Policy,
}

impl Movement {
    /// Build from integer hardware reports, the usual shape.
    pub fn from_reports(a: [f64; 2], target: [f64; 2], radius: f64, reports: &[[i16; 2]]) -> Self {
        Self::from_counts(
            a,
            target,
            radius,
            &reports
                .iter()
                .map(|row| [f32::from(row[0]), f32::from(row[1])])
                .collect::<Vec<_>>(),
        )
    }

    pub fn from_counts(a: [f64; 2], target: [f64; 2], radius: f64, counts: &[[f32; 2]]) -> Self {
        let mut moved = [0.0f64; 2];
        for row in counts {
            moved[0] += f64::from(row[0]);
            moved[1] += f64::from(row[1]);
        }
        Self {
            a,
            target,
            radius,
            prefix: counts.to_vec(),
            moved,
            policy: Policy::Auto,
        }
    }

    pub fn with_policy(mut self, policy: Policy) -> Self {
        self.policy = policy;
        self
    }

    pub fn prefix(&self) -> &[[f32; 2]] {
        &self.prefix
    }

    /// Summed displacement from A to B.
    pub fn moved(&self) -> [f64; 2] {
        self.moved
    }

    /// Vector from B to the target centre.
    pub fn remaining(&self) -> [f64; 2] {
        [
            self.target[0] - self.a[0] - self.moved[0],
            self.target[1] - self.a[1] - self.moved[1],
        ]
    }

    /// Validate B and derive the geometry the planner needs.
    pub fn handoff(&self) -> Result<Handoff> {
        self.resolve().map(|(handoff, _)| handoff)
    }

    /// Which policy `Auto` settled on.
    pub fn resolved_policy(&self) -> Result<BConfig> {
        self.resolve().map(|(_, config)| config)
    }

    fn resolve(&self) -> Result<(Handoff, BConfig)> {
        let mut reasons: Vec<(f64, String)> = Vec::new();
        for config in self.policy.candidates() {
            match self.check(&config) {
                Ok(handoff) => return Ok((handoff, config)),
                Err(reason) => reasons.push((config.threshold, reason)),
            }
        }
        // When every policy fails the same way, the thresholds are not the story.
        let uniform = reasons.windows(2).all(|pair| pair[0].1 == pair[1].1);
        let mut message = if uniform {
            reasons
                .first()
                .map(|(_, reason)| reason.clone())
                .unwrap_or_else(|| "no handoff policy was offered".into())
        } else {
            reasons
                .iter()
                .map(|(threshold, reason)| format!("at {:.0}%: {reason}", threshold * 100.0))
                .collect::<Vec<_>>()
                .join("; ")
        };
        let remaining = self.remaining();
        let magnitude = remaining[0].hypot(remaining[1]);
        if magnitude < 8.0 {
            message.push_str(
                ". A continuation with under 8 counts left cannot be planned under any \
                 policy; for corrections this small use the Continuous Planner, whose \
                 settling behaviour covers exactly this case",
            );
        } else if let Some(window) =
            Self::eligible_remaining(self.a, self.target, self.radius, self.policy)
        {
            message.push_str(&format!(
                ". For this geometry |BC| must be between {:.2} and {:.2} counts; it is {magnitude:.2}",
                window.start(),
                window.end()
            ));
        }
        Err(Error::InferenceContract(message))
    }

    fn check(&self, config: &BConfig) -> std::result::Result<Handoff, String> {
        let goal = [self.target[0] - self.a[0], self.target[1] - self.a[1]];
        let distance = goal[0].hypot(goal[1]);
        if distance.is_nan() || distance <= 1e-9 {
            return Err("A and the target coincide".into());
        }
        if !self.radius.is_finite() || self.radius <= 0.0 {
            return Err("radius must be positive".into());
        }
        let ticks = self.prefix.len();
        if ticks > config.max_ab_ms {
            return Err(format!(
                "A to B is {ticks} ms, over the {} ms limit",
                config.max_ab_ms
            ));
        }

        // Replay the per-tick regression rule, which needs the path.
        if config.progress_regression_reset {
            let mut running = [0.0f64; 2];
            let mut peak = 0.0f64;
            for (index, delta) in self.prefix.iter().enumerate() {
                running[0] += f64::from(delta[0]);
                running[1] += f64::from(delta[1]);
                let centre = (running[0] * goal[0] + running[1] * goal[1]) / (distance * distance);
                if centre > peak {
                    peak = centre;
                } else if peak - centre > config.progress_regression_threshold {
                    return Err(format!(
                        "progress fell {:.3} from its peak at {} ms, over the {:.2} limit",
                        peak - centre,
                        index + 1,
                        config.progress_regression_threshold
                    ));
                }
            }
        }

        let along = (self.moved[0] * goal[0] + self.moved[1] * goal[1]) / distance;
        let centre = along / distance;
        let edge = along / (distance - self.radius.max(0.0)).max(1e-9);
        if edge < config.threshold {
            let needed = config.threshold * (distance - self.radius) - along;
            return Err(format!(
                "edge progress is {edge:.3} at B, under the {:.2} threshold ({needed:.1} counts short)",
                config.threshold
            ));
        }
        if let Some(limit) = config.max_realized_progress {
            if edge > limit {
                return Err(format!(
                    "edge progress {edge:.3} is over the {limit:.2} limit"
                ));
            }
        }
        if centre > config.max_center_progress {
            return Err(format!(
                "centre progress is {centre:.3} at B, over the {:.2} limit",
                config.max_center_progress
            ));
        }
        let remaining = self.remaining();
        let magnitude = remaining[0].hypot(remaining[1]);
        if magnitude < config.min_remaining_counts {
            return Err(format!(
                "|BC| is {magnitude:.2} counts, under the {:.1} the contract requires",
                config.min_remaining_counts
            ));
        }
        if config.require_outside_target && magnitude <= self.radius {
            return Err(format!(
                "B is already inside the target: |BC| {magnitude:.2} <= radius {:.2}",
                self.radius
            ));
        }
        if ticks < config.minimum_prefix {
            return Err(format!(
                "the observed prefix is {ticks} ms, under the {} ms minimum",
                config.minimum_prefix
            ));
        }
        if magnitude - self.radius < config.minimum_edge_margin {
            return Err(format!(
                "the edge margin is {:.2} counts, under the {:.1} minimum",
                magnitude - self.radius,
                config.minimum_edge_margin
            ));
        }

        Handoff::new(remaining, self.radius, centre.clamp(0.0, 1.0))
            .map_err(|error| error.to_string())
    }

    /// The window of `|BC|` that can yield a valid handoff for this geometry.
    ///
    /// The progress bounds assume a straight movement from A; the rules that
    /// depend only on `|BC|` and the radius hold for any path.
    pub fn eligible_remaining(
        a: [f64; 2],
        target: [f64; 2],
        radius: f64,
        policy: Policy,
    ) -> Option<RangeInclusive<f64>> {
        let goal = [target[0] - a[0], target[1] - a[1]];
        let distance = goal[0].hypot(goal[1]);
        if distance.is_nan() || distance <= 1e-9 || !radius.is_finite() || radius <= 0.0 {
            return None;
        }
        let mut best: Option<RangeInclusive<f64>> = None;
        for config in policy.candidates() {
            let lower = config
                .min_remaining_counts
                .max(radius + config.minimum_edge_margin.max(f64::EPSILON))
                .max((1.0 - config.max_center_progress) * distance);
            let upper = distance - config.threshold * (distance - radius);
            if lower <= upper {
                let widened = match best {
                    Some(current) => current.start().min(lower)..=current.end().max(upper),
                    None => lower..=upper,
                };
                best = Some(widened);
            }
        }
        best
    }
}
