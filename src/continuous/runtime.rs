use super::constants::*;
use super::kernels::Window;
use super::planner::Planner;
use crate::error::{Error, Result};
use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct Advance {
    pub time_us: Vec<i64>,
    pub xy: Vec<[f64; 2]>,
}

#[derive(Debug)]
pub struct RuntimeFailure {
    pub message: String,
    pub at_us: i64,
    pub partial: Advance,
}

#[derive(Debug)]
pub enum AdvanceError {
    Contract(Error),
    Failed(Box<RuntimeFailure>),
}

impl fmt::Display for AdvanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdvanceError::Contract(error) => write!(f, "{error}"),
            AdvanceError::Failed(failure) => {
                write!(
                    f,
                    "Policy action failed at {} us: {}",
                    failure.at_us, failure.message
                )
            }
        }
    }
}

impl std::error::Error for AdvanceError {}

impl From<Error> for AdvanceError {
    fn from(value: Error) -> Self {
        AdvanceError::Contract(value)
    }
}

pub struct MovementRuntime {
    planner: Planner,
    initial_xy: [f64; 2],
    initial_history: Vec<[f64; 2]>,
    seed: u64,
    history: Vec<[f64; 2]>,
    target_history: Vec<[f64; 2]>,
    available: Vec<i64>,
    valid: Vec<bool>,
    motion_known: Vec<bool>,
    actions: [[f64; 2]; COMMIT_SAMPLES],
    action_points: [[f64; 2]; COMMIT_SAMPLES],
    position: [f64; 2],
    receipts: VecDeque<(i64, [f64; 2])>,
    clock_origin: Instant,
    now_us: i64,
    requested_us: i64,
    last_receipt_us: i64,
    receipt_us: i64,
    target: Option<[f64; 2]>,
    cursor: usize,
    pending_index: usize,
    observed: bool,
    fresh_decisions: u64,
    failed: bool,
}

fn check_position(value: [f64; 2]) -> Result<[f64; 2]> {
    if !value.iter().all(|component| component.is_finite()) {
        return Err(Error::InferenceContract(
            "Position must be a finite pair in common counts".into(),
        ));
    }
    Ok(value)
}

fn check_history(value: Option<&[[f64; 2]]>) -> Result<Vec<[f64; 2]>> {
    let history = match value {
        None => vec![[0.0; 2]; WARM_HISTORY_SAMPLES],
        Some(rows) => rows.to_vec(),
    };
    let finite = history
        .iter()
        .flat_map(|row| row.iter())
        .all(|component| component.is_finite());
    if history.len() != WARM_HISTORY_SAMPLES || !finite {
        return Err(Error::InferenceContract(
            "History must contain 160 finite displacement pairs".into(),
        ));
    }
    Ok(history)
}

impl MovementRuntime {
    pub fn new(
        planner: Planner,
        initial_xy: [f64; 2],
        history: Option<&[[f64; 2]]>,
        seed: u64,
    ) -> Result<Self> {
        let mut runtime = Self {
            planner,
            initial_xy: check_position(initial_xy)?,
            initial_history: check_history(history)?,
            seed,
            // Mirrored rings make any chronological 640-sample window contiguous.
            // Each tick writes one slot and its mirror, rather than shifting history.
            history: vec![[0.0; 2]; 2 * HISTORY_SAMPLES],
            target_history: vec![[0.0; 2]; 2 * HISTORY_SAMPLES],
            available: vec![0; 2 * HISTORY_SAMPLES],
            valid: vec![false; 2 * HISTORY_SAMPLES],
            motion_known: vec![false; 2 * HISTORY_SAMPLES],
            actions: [[0.0; 2]; COMMIT_SAMPLES],
            action_points: [[0.0; 2]; COMMIT_SAMPLES],
            position: [0.0; 2],
            receipts: VecDeque::new(),
            clock_origin: Instant::now(),
            now_us: 0,
            requested_us: 0,
            last_receipt_us: -1,
            receipt_us: 0,
            target: None,
            cursor: 0,
            pending_index: COMMIT_SAMPLES,
            observed: false,
            fresh_decisions: 0,
            failed: false,
        };
        runtime.reset(None, None, None)?;
        Ok(runtime)
    }

    pub fn reset(
        &mut self,
        initial_xy: Option<[f64; 2]>,
        history: Option<&[[f64; 2]]>,
        seed: Option<u64>,
    ) -> Result<()> {
        let position = match initial_xy {
            Some(value) => check_position(value)?,
            None => self.initial_xy,
        };
        let warm = match history {
            Some(_) => check_history(history)?,
            None => self.initial_history.clone(),
        };
        let reset_seed = seed.unwrap_or(self.seed);
        self.planner.reset(reset_seed);
        self.initial_xy = position;
        self.initial_history = warm.clone();
        self.seed = reset_seed;
        self.clock_origin = Instant::now();
        self.now_us = 0;
        self.requested_us = 0;
        self.last_receipt_us = -1;
        self.receipt_us = 0;
        self.receipts.clear();
        self.target = None;
        self.position = position;
        self.cursor = 0;
        self.history.fill([0.0; 2]);
        self.target_history.fill([0.0; 2]);
        self.available.fill(0);
        self.valid.fill(false);
        self.motion_known.fill(false);
        let start = HISTORY_SAMPLES - WARM_HISTORY_SAMPLES;
        for (step, row) in warm.iter().enumerate() {
            self.history[start + step] = *row;
            self.history[start + step + HISTORY_SAMPLES] = *row;
            self.motion_known[start + step] = true;
            self.motion_known[start + step + HISTORY_SAMPLES] = true;
        }
        self.pending_index = COMMIT_SAMPLES;
        self.observed = false;
        self.fresh_decisions = 0;
        self.failed = false;
        Ok(())
    }

    pub fn elapsed_us(&self) -> i64 {
        self.clock_origin.elapsed().as_micros() as i64
    }

    pub fn update_target(&mut self, xy: [f64; 2], timestamp_us: i64) -> Result<()> {
        if timestamp_us < 0 {
            return Err(Error::InferenceContract(
                "Time must be a nonnegative int64 microsecond offset".into(),
            ));
        }
        if !xy.iter().all(|component| component.is_finite()) {
            return Err(Error::InferenceContract(
                "Target must be a finite pair in common counts".into(),
            ));
        }
        if timestamp_us < self.now_us.max(self.last_receipt_us) {
            return Err(Error::InferenceContract(
                "Target receipts must be monotonic and not precede committed movement".into(),
            ));
        }
        self.receipts.push_back((timestamp_us, xy));
        self.last_receipt_us = timestamp_us;
        Ok(())
    }

    fn arrive(&mut self) {
        match self.receipts.front() {
            Some(&(at, _)) if at <= self.now_us => {}
            _ => return,
        }
        while let Some(&(at, point)) = self.receipts.front() {
            if at > self.now_us {
                break;
            }
            self.receipt_us = at;
            self.target = Some(point);
            self.receipts.pop_front();
        }
        let slot = (self.cursor + HISTORY_SAMPLES - 1) % HISTORY_SAMPLES;
        let target = self.target.unwrap_or([0.0; 2]);
        self.target_history[slot] = target;
        self.target_history[slot + HISTORY_SAMPLES] = target;
        self.available[slot] = self.receipt_us;
        self.available[slot + HISTORY_SAMPLES] = self.receipt_us;
        self.valid[slot] = true;
        self.valid[slot + HISTORY_SAMPLES] = true;
    }

    fn append(&mut self, delta: Option<[f64; 2]>) {
        let slot = self.cursor;
        let mirror = slot + HISTORY_SAMPLES;
        let motion = delta.unwrap_or([0.0; 2]);
        self.history[slot] = motion;
        self.history[mirror] = motion;
        match self.target {
            None => {
                self.target_history[slot] = [0.0; 2];
                self.target_history[mirror] = [0.0; 2];
                self.available[slot] = 0;
                self.available[mirror] = 0;
                self.valid[slot] = false;
                self.valid[mirror] = false;
            }
            Some(target) => {
                self.target_history[slot] = target;
                self.target_history[mirror] = target;
                self.available[slot] = self.receipt_us;
                self.available[mirror] = self.receipt_us;
                self.valid[slot] = true;
                self.valid[mirror] = true;
            }
        }
        self.motion_known[slot] = true;
        self.motion_known[mirror] = true;
        self.cursor = (slot + 1) % HISTORY_SAMPLES;
    }

    fn initialize_observation(&mut self) {
        // The research adapter creates its 640 ms state on the first commanded
        // call, even if an arbitrary uncommanded interval preceded that call.
        for step in 0..HISTORY_SAMPLES - WARM_HISTORY_SAMPLES {
            let slot = (self.cursor + step) % HISTORY_SAMPLES;
            for index in [slot, slot + HISTORY_SAMPLES] {
                self.history[index] = [0.0; 2];
                self.target_history[index] = [0.0; 2];
                self.available[index] = 0;
                self.valid[index] = false;
                self.motion_known[index] = false;
            }
        }
        self.observed = true;
    }

    fn plan(&mut self) -> Result<()> {
        if !self.observed {
            self.initialize_observation();
        }
        let start = self.cursor;
        let end = start + HISTORY_SAMPLES;
        let window = Window {
            history: &self.history[start..end],
            target: &self.target_history[start..end],
            available: &self.available[start..end],
            valid: &self.valid[start..end],
            motion_known: &self.motion_known[start..end],
            position: self.position,
            cut_us: self.now_us,
        };
        let action = self.planner.plan(&window, self.now_us)?;
        self.actions = *action;
        // The reference rebases absolute positions on each 8 ms public slice.
        // Retaining that summation boundary avoids needless numerical drift.
        let mut origin = self.position;
        for offset in (0..COMMIT_SAMPLES).step_by(8) {
            let mut running = origin;
            for step in offset..offset + 8 {
                running = [
                    running[0] + self.actions[step][0],
                    running[1] + self.actions[step][1],
                ];
                self.action_points[step] = running;
            }
            origin = self.action_points[offset + 7];
        }
        let finite = self
            .action_points
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite());
        if !finite {
            return Err(Error::Numerical("Nonfinite absolute action path".into()));
        }
        self.pending_index = 0;
        self.fresh_decisions += 1;
        Ok(())
    }

    pub fn advance(&mut self, timestamp_us: i64) -> std::result::Result<Advance, AdvanceError> {
        if self.failed {
            return Err(AdvanceError::Contract(Error::Mode(
                "Runtime has failed; inspect retained state or reset before advancing".into(),
            )));
        }
        if timestamp_us < 0 {
            return Err(AdvanceError::Contract(Error::InferenceContract(
                "Time must be a nonnegative int64 microsecond offset".into(),
            )));
        }
        if timestamp_us < self.requested_us {
            return Err(AdvanceError::Contract(Error::InferenceContract(
                "Advance clock must be monotonic".into(),
            )));
        }
        self.requested_us = timestamp_us;
        let count = ((timestamp_us - self.now_us) / SAMPLE_US).max(0) as usize;
        let base = self.now_us;
        let time_us: Vec<i64> = (1..=count as i64)
            .map(|step| base + step * SAMPLE_US)
            .collect();
        let mut xy = vec![[0.0f64; 2]; count];
        self.arrive();
        for output_at in 0..count {
            if self.pending_index == COMMIT_SAMPLES {
                if self.target.is_none() {
                    // With no target, preserve numerical position and append a
                    // known zero displacement. Do not consume policy randomness.
                    self.append(None);
                    self.now_us += SAMPLE_US;
                    self.arrive();
                    xy[output_at] = self.position;
                    continue;
                }
                if let Err(error) = self.plan() {
                    self.failed = true;
                    return Err(AdvanceError::Failed(Box::new(RuntimeFailure {
                        message: error.to_string(),
                        at_us: self.now_us,
                        partial: Advance {
                            time_us: time_us[..output_at].to_vec(),
                            xy: xy[..output_at].to_vec(),
                        },
                    })));
                }
            }
            let at = self.pending_index;
            self.position = self.action_points[at];
            let delta = self.actions[at];
            self.append(Some(delta));
            self.pending_index += 1;
            self.now_us += SAMPLE_US;
            self.arrive();
            xy[output_at] = self.position;
        }
        Ok(Advance { time_us, xy })
    }

    pub fn current_xy(&self) -> [f64; 2] {
        self.position
    }

    pub fn now_us(&self) -> i64 {
        self.now_us
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn needs_plan(&self) -> bool {
        let queued = self
            .receipts
            .front()
            .is_some_and(|&(at, _)| at <= self.now_us);
        self.pending_index == COMMIT_SAMPLES && (self.target.is_some() || queued)
    }

    pub fn decisions(&self) -> u64 {
        self.fresh_decisions
    }

    pub fn motor_evaluations(&self) -> u64 {
        self.planner.motor_evaluations
    }

    pub fn planner(&self) -> &Planner {
        &self.planner
    }

    pub fn planner_mut(&mut self) -> &mut Planner {
        &mut self.planner
    }
}
