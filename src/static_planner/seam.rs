use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct OnsetConfig {
    pub noise_window_ms: usize,
    pub consecutive_ticks: usize,
    pub backtrack_ms: usize,
    pub speed_floor: f64,
    pub threshold_mad_multiplier: f64,
    pub alignment_min: f64,
    pub quiet_baseline_fallback: bool,
}

impl Default for OnsetConfig {
    fn default() -> Self {
        Self {
            noise_window_ms: 24,
            consecutive_ticks: 12,
            backtrack_ms: 4,
            speed_floor: 0.35,
            threshold_mad_multiplier: 6.0,
            alignment_min: 0.15,
            quiet_baseline_fallback: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnsetEvent {
    pub index: usize,
    pub threshold: f64,
    pub speed_median: f64,
    pub speed_mad: f64,
}

#[derive(Clone, Copy, Debug)]
struct OnsetTick {
    index: usize,
    speed: f64,
    aligned: bool,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let count = values.len();
    if count % 2 == 1 {
        values[count / 2]
    } else {
        (values[count / 2 - 1] + values[count / 2]) / 2.0
    }
}

/// Detect A causally from 1 kHz counts and the current target vector.
pub struct OnsetDetector {
    pub config: OnsetConfig,
    baseline_speeds: Vec<f64>,
    pending: Vec<OnsetTick>,
    threshold: Option<f64>,
    median: f64,
    mad: f64,
    run: usize,
    tick_count: usize,
    done: bool,
}

impl OnsetDetector {
    pub fn new(config: OnsetConfig) -> Self {
        Self {
            config,
            baseline_speeds: Vec::new(),
            pending: Vec::new(),
            threshold: None,
            median: 0.0,
            mad: 0.0,
            run: 0,
            tick_count: 0,
            done: false,
        }
    }

    pub fn fired(&self) -> bool {
        self.done
    }

    /// Consume one closed 1 ms bin; return A once, then remain latched.
    pub fn push(
        &mut self,
        dx: f64,
        dy: f64,
        target_rel: Option<[f64; 2]>,
    ) -> Result<Option<OnsetEvent>> {
        if self.done {
            return Ok(None);
        }
        let speed = dx.hypot(dy);
        let aligned = match target_rel {
            None => true,
            Some(target) => {
                if !target.iter().all(|value| value.is_finite()) {
                    return Err(Error::InferenceContract(
                        "target_rel must contain two finite values".into(),
                    ));
                }
                let target_norm = (target[0] * target[0] + target[1] * target[1]).sqrt();
                let denominator = speed * target_norm + 1e-6;
                (dx * target[0] + dy * target[1]) / denominator >= self.config.alignment_min
            }
        };
        let tick = OnsetTick {
            index: self.tick_count,
            speed,
            aligned,
        };
        self.tick_count += 1;

        if self.threshold.is_none() {
            self.baseline_speeds.push(speed);
            self.pending.push(tick);
            if self.baseline_speeds.len() < self.config.noise_window_ms {
                return Ok(None);
            }
            let mut baseline = self.baseline_speeds.clone();
            self.median = median(&mut baseline);
            let mut deviations: Vec<f64> = self
                .baseline_speeds
                .iter()
                .map(|value| (value - self.median).abs())
                .collect();
            self.mad = median(&mut deviations);
            self.threshold = Some(
                if self.config.quiet_baseline_fallback && self.median > self.config.speed_floor {
                    self.config.speed_floor
                } else {
                    self.config
                        .speed_floor
                        .max(self.median + self.config.threshold_mad_multiplier * self.mad)
                },
            );
            let pending = std::mem::take(&mut self.pending);
            for buffered in pending {
                if let Some(event) = self.scan(buffered) {
                    return Ok(Some(event));
                }
            }
            return Ok(None);
        }
        Ok(self.scan(tick))
    }

    fn scan(&mut self, tick: OnsetTick) -> Option<OnsetEvent> {
        let threshold = self.threshold.unwrap_or(0.0);
        let preferred = tick.speed > threshold && tick.aligned;
        self.run = if preferred { self.run + 1 } else { 0 };
        if self.run < self.config.consecutive_ticks {
            return None;
        }
        self.done = true;
        let onset_start = tick.index + 1 - self.config.consecutive_ticks;
        Some(OnsetEvent {
            index: onset_start.saturating_sub(self.config.backtrack_ms),
            threshold,
            speed_median: self.median,
            speed_mad: self.mad,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BConfig {
    pub threshold: f64,
    pub require_outside_target: bool,
    pub max_ab_ms: usize,
    pub min_remaining_counts: f64,
    pub progress_regression_reset: bool,
    pub progress_regression_threshold: f64,
    pub max_center_progress: f64,
    pub max_realized_progress: Option<f64>,
    pub minimum_prefix: usize,
    pub minimum_edge_margin: f64,
}

impl Default for BConfig {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            require_outside_target: true,
            max_ab_ms: 1500,
            min_remaining_counts: 8.0,
            progress_regression_reset: true,
            progress_regression_threshold: 0.18,
            max_center_progress: 0.92,
            max_realized_progress: None,
            minimum_prefix: 0,
            minimum_edge_margin: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BFire {
    pub t_ms: usize,
    pub progress_edge: f64,
    pub progress_center: f64,
    pub movement_counts: [f64; 2],
    pub target_rel_at_b: [f64; 2],
    pub target_radius: f64,
    pub remaining_counts: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BReject {
    pub t_ms: usize,
    pub reason: &'static str,
    pub progress_center: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BEvent {
    Fire(BFire),
    Reject(BReject),
}

/// Track one A->B movement and fire at the frozen edge-progress seam.
pub struct BTrigger {
    pub config: BConfig,
    armed: bool,
    movement: [f64; 2],
    target_at_a: [f64; 2],
    radius_at_a: f64,
    t_ms: usize,
    max_progress_center: f64,
}

impl BTrigger {
    pub fn new(config: BConfig) -> Self {
        Self {
            config,
            armed: false,
            movement: [0.0; 2],
            target_at_a: [0.0; 2],
            radius_at_a: 0.0,
            t_ms: 0,
            max_progress_center: 0.0,
        }
    }

    /// Prefer a qualified 90% edge-progress handoff: at the first crossing
    /// require 24 observed milliseconds and 8 counts outside the target edge.
    pub fn recommended() -> Self {
        Self::new(BConfig {
            threshold: 0.9,
            minimum_prefix: 24,
            minimum_edge_margin: 8.0,
            ..BConfig::default()
        })
    }

    pub fn from_seam_contract() -> Self {
        Self::new(BConfig::default())
    }

    pub fn armed(&self) -> bool {
        self.armed
    }

    pub fn movement_counts(&self) -> [f64; 2] {
        self.movement
    }

    pub fn arm(&mut self, target_rel_at_a: [f64; 2], target_radius: f64) -> Result<()> {
        if !target_rel_at_a.iter().all(|value| value.is_finite()) {
            return Err(Error::InferenceContract(
                "target_rel_at_A must contain two finite values".into(),
            ));
        }
        if !target_radius.is_finite() || target_radius <= 0.0 {
            return Err(Error::InferenceContract(
                "target_radius must be positive".into(),
            ));
        }
        let distance = (target_rel_at_a[0] * target_rel_at_a[0]
            + target_rel_at_a[1] * target_rel_at_a[1])
            .sqrt();
        if distance <= target_radius {
            return Err(Error::InferenceContract(
                "A must begin outside the target".into(),
            ));
        }
        self.armed = true;
        self.movement = [0.0; 2];
        self.target_at_a = target_rel_at_a;
        self.radius_at_a = target_radius;
        self.t_ms = 0;
        self.max_progress_center = 0.0;
        Ok(())
    }

    pub fn disarm(&mut self) {
        self.armed = false;
    }

    pub fn progress(&self, edge: bool) -> f64 {
        let distance = (self.target_at_a[0] * self.target_at_a[0]
            + self.target_at_a[1] * self.target_at_a[1])
            .sqrt();
        let mut denominator = distance;
        if edge {
            denominator -= self.radius_at_a.max(0.0);
        }
        denominator = denominator.max(1e-9);
        let unit = if distance > 1e-9 {
            [
                self.target_at_a[0] / distance,
                self.target_at_a[1] / distance,
            ]
        } else {
            [0.0, 0.0]
        };
        (self.movement[0] * unit[0] + self.movement[1] * unit[1]) / denominator
    }

    pub fn push_tick(
        &mut self,
        dx: f64,
        dy: f64,
        target_rel_now: [f64; 2],
        target_radius_now: f64,
    ) -> Result<Option<BEvent>> {
        if !self.armed {
            return Ok(None);
        }
        if !target_rel_now.iter().all(|value| value.is_finite()) {
            return Err(Error::InferenceContract(
                "target_rel_now must contain two finite values".into(),
            ));
        }
        if !target_radius_now.is_finite() || target_radius_now <= 0.0 {
            return Err(Error::InferenceContract(
                "target_radius_now must be positive".into(),
            ));
        }
        if !dx.is_finite() || !dy.is_finite() {
            return Err(Error::InferenceContract(
                "motion must contain finite count displacements".into(),
            ));
        }

        self.movement[0] += dx;
        self.movement[1] += dy;
        self.t_ms += 1;
        let progress_center = self.progress(false);
        let progress_edge = self.progress(true);
        let config = self.config;

        if self.t_ms > config.max_ab_ms {
            return Ok(Some(self.reject("max_ab_ms", progress_center)));
        }
        if progress_center > self.max_progress_center {
            self.max_progress_center = progress_center;
        } else if config.progress_regression_reset
            && self.max_progress_center - progress_center > config.progress_regression_threshold
        {
            return Ok(Some(self.reject("progress_regression", progress_center)));
        }
        if progress_edge < config.threshold {
            return Ok(None);
        }
        if let Some(limit) = config.max_realized_progress {
            if progress_edge > limit {
                return Ok(Some(self.reject("max_realized_progress", progress_center)));
            }
        }
        if progress_center > config.max_center_progress {
            return Ok(Some(self.reject("max_center_progress", progress_center)));
        }
        let remaining =
            (target_rel_now[0] * target_rel_now[0] + target_rel_now[1] * target_rel_now[1]).sqrt();
        if remaining < config.min_remaining_counts {
            return Ok(Some(self.reject("min_remaining_counts", progress_center)));
        }
        if config.require_outside_target && remaining <= target_radius_now {
            return Ok(Some(self.reject("inside_target", progress_center)));
        }
        if self.t_ms < config.minimum_prefix {
            return Ok(Some(self.reject("min_prefix_ms", progress_center)));
        }
        if remaining - target_radius_now < config.minimum_edge_margin {
            return Ok(Some(self.reject("min_edge_margin_counts", progress_center)));
        }

        self.armed = false;
        Ok(Some(BEvent::Fire(BFire {
            t_ms: self.t_ms,
            progress_edge,
            progress_center,
            movement_counts: self.movement,
            target_rel_at_b: target_rel_now,
            target_radius: target_radius_now,
            remaining_counts: remaining,
        })))
    }

    fn reject(&mut self, reason: &'static str, progress_center: f64) -> BEvent {
        self.armed = false;
        BEvent::Reject(BReject {
            t_ms: self.t_ms,
            reason,
            progress_center,
        })
    }
}
