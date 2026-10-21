use super::constants::*;
use super::kernels::{self, EventFeatures, EventState, MotorFeatures, Window};
use crate::error::{Error, Result, require};
use crate::io::Npz;
use crate::math::log_f32;
use crate::nn::{Engine, EventOutputs, NeuralInference, hazard_probabilities};
use crate::rng::{RandomStream, softmax_into};

const SELECTOR_SALT: u64 = 0x5397A1;
const BRAKE_SALT: u64 = 0x1763B4;

/// The sixteen movement shapes the motor proposes, scored by the choice
/// network, before one is drawn.
///
/// `geometries` holds each head's coarse preview: [`GEOMETRY_STEPS`] steps of
/// `[dx, dy]`, laid out head-major. `sampled` is the head the model's own
/// categorical draw produced, which is what the planner uses when no chooser
/// is installed.
pub struct MotorCandidates<'a> {
    pub probabilities: &'a [f32; HEADS],
    pub geometries: &'a [f32],
    pub sampled: usize,
}

impl MotorCandidates<'_> {
    pub fn geometry(&self, head: usize) -> &[f32] {
        &self.geometries[head * GEOMETRY_STEPS * 2..(head + 1) * GEOMETRY_STEPS * 2]
    }

    /// Where the head's preview ends up, summed over its steps.
    pub fn displacement(&self, head: usize) -> [f64; 2] {
        self.geometry(head)
            .chunks_exact(2)
            .fold([0.0f64; 2], |total, step| {
                [total[0] + f64::from(step[0]), total[1] + f64::from(step[1])]
            })
    }

    pub fn most_likely(&self) -> usize {
        let mut best = 0;
        for head in 1..HEADS {
            if self.probabilities[head] > self.probabilities[best] {
                best = head;
            }
        }
        best
    }
}

/// The sixteen stopping shapes, scored when a movement decides to brake.
///
/// `durations` is each head's braking length in milliseconds, the continuous
/// counterpart of the Static Planner's `duration_ms`.
pub struct BrakeCandidates<'a> {
    pub probabilities: &'a [f32; HEADS],
    pub durations: &'a [f32],
    pub sampled: usize,
}

impl BrakeCandidates<'_> {
    pub fn most_likely(&self) -> usize {
        let mut best = 0;
        for head in 1..HEADS {
            if self.probabilities[head] > self.probabilities[best] {
                best = head;
            }
        }
        best
    }
}

/// Replace the planner's own head draws.
///
/// Both hooks default to `candidates.sampled`, which is the drawn head, so an
/// implementation overrides only the decision it cares about. The draw happens
/// either way: the random words it consumes keep every later decision in this
/// stream aligned with the reference, so returning `sampled` is exactly the
/// untouched planner.
///
/// A returned head outside `0..16` fails the plan rather than being clamped.
pub trait HeadChooser: Send {
    fn motor(&mut self, candidates: &MotorCandidates<'_>) -> usize {
        candidates.sampled
    }

    fn brake(&mut self, candidates: &BrakeCandidates<'_>) -> usize {
        candidates.sampled
    }
}

/// Hold one or both decisions to a fixed head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinnedHeads {
    pub motor: Option<usize>,
    pub brake: Option<usize>,
}

impl PinnedHeads {
    pub fn motor(head: usize) -> Self {
        Self {
            motor: Some(head),
            brake: None,
        }
    }

    pub fn brake(head: usize) -> Self {
        Self {
            motor: None,
            brake: Some(head),
        }
    }

    pub fn with_brake(mut self, head: usize) -> Self {
        self.brake = Some(head);
        self
    }

    pub fn with_motor(mut self, head: usize) -> Self {
        self.motor = Some(head);
        self
    }
}

impl HeadChooser for PinnedHeads {
    fn motor(&mut self, candidates: &MotorCandidates<'_>) -> usize {
        self.motor.unwrap_or(candidates.sampled)
    }

    fn brake(&mut self, candidates: &BrakeCandidates<'_>) -> usize {
        self.brake.unwrap_or(candidates.sampled)
    }
}

pub struct Planner {
    engine: Engine,
    velocity_basis: Vec<f64>,
    carry_velocity: Vec<f64>,
    mean_basis: Vec<f64>,
    mean_carry: Vec<f64>,
    skip_unused: bool,

    motor_rng: RandomStream,
    selector_rng: RandomStream,
    brake_rng: RandomStream,

    mode: Option<u8>,
    age: f64,
    duration: f64,
    brake_velocity: [f64; 2],
    brake_acceleration: [f64; 2],
    coefficients: [[f64; 2]; 5],
    hold_goal: [f64; 2],
    hold_position: [f64; 2],
    hold_error: f64,
    changed: bool,
    innovation_at: f64,
    resume_integral: f64,
    stop_integral: f64,
    resume_budget: f32,
    stop_budget: f32,

    last_head: i32,
    previous: Vec<f32>,
    previous_valid: bool,
    origin: Option<[f64; 2]>,

    motor: MotorFeatures,
    events: EventFeatures,
    outputs: EventOutputs,
    heads: Vec<f32>,
    geometry: Vec<f32>,
    pairs: Vec<f32>,
    probabilities: [f32; HEADS],
    curve: Vec<[f64; 2]>,
    action: [[f64; 2]; COMMIT_SAMPLES],

    pub decision_count: u64,
    pub motor_evaluations: u64,
    pub record_decisions: bool,
    pub decisions: Vec<Decision>,
    chooser: Option<Box<dyn HeadChooser>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub at_ms: f64,
    pub mode: u8,
    pub head: i32,
}

impl Planner {
    pub fn load(bundle: &Npz, skip_unused: bool, backend: NeuralInference) -> Result<Self> {
        let engine = Engine::load(bundle, backend)?;
        let velocity_basis = bundle.f64("motor.velocity_basis")?;
        let carry_velocity = bundle.f64("motor.carry_velocity")?;
        let (mean_basis, mean_carry) = kernels::decoder_geometry(&velocity_basis, &carry_velocity);
        let mut planner = Self {
            engine,
            velocity_basis,
            carry_velocity,
            mean_basis,
            mean_carry,
            skip_unused,
            motor_rng: RandomStream::new(7),
            selector_rng: RandomStream::new(7),
            brake_rng: RandomStream::new(7),
            mode: None,
            age: 160.0,
            duration: 64.0,
            brake_velocity: [0.0; 2],
            brake_acceleration: [0.0; 2],
            coefficients: [[0.0; 2]; 5],
            hold_goal: [0.0; 2],
            hold_position: [0.0; 2],
            hold_error: 0.0,
            changed: false,
            innovation_at: f64::NAN,
            resume_integral: 0.0,
            stop_integral: 0.0,
            resume_budget: 0.0,
            stop_budget: 0.0,
            last_head: -1,
            previous: vec![0.0; GEOMETRY_STEPS * 2],
            previous_valid: false,
            origin: None,
            motor: MotorFeatures::default(),
            events: EventFeatures::default(),
            outputs: EventOutputs::default(),
            heads: vec![0.0; HEADS * GEOMETRY_STEPS * 2],
            geometry: vec![0.0; HEADS * GEOMETRY_STEPS],
            pairs: vec![0.0; HEADS * PAIR_LEN],
            probabilities: [0.0; HEADS],
            curve: vec![[0.0; 2]; COMMIT_SAMPLES + 1],
            action: [[0.0; 2]; COMMIT_SAMPLES],
            decision_count: 0,
            motor_evaluations: 0,
            record_decisions: false,
            decisions: Vec::new(),
            chooser: None,
        };
        planner.reset(7);
        Ok(planner)
    }

    pub fn reset(&mut self, seed: u64) {
        self.motor_rng = RandomStream::new(seed);
        self.selector_rng = RandomStream::new(self.motor_rng.seed() ^ SELECTOR_SALT);
        self.brake_rng = RandomStream::new(self.motor_rng.seed() ^ BRAKE_SALT);
        self.mode = None;
        self.previous_valid = false;
        self.origin = None;
        self.decision_count = 0;
        self.motor_evaluations = 0;
        self.decisions.clear();
    }

    pub fn backend(&self) -> NeuralInference {
        self.engine.backend()
    }

    pub fn mode(&self) -> u8 {
        self.mode.unwrap_or(MODE_HOLD)
    }

    fn initialize(
        &mut self,
        history: &[[f64; 2]],
        position: [f64; 2],
        goal: [f64; 2],
        now_us: i64,
    ) {
        let still = history[HISTORY_SAMPLES - COMMIT_SAMPLES..]
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0f64, |peak, value| peak.max(value.abs()))
            <= 1e-12;
        self.mode = Some(if still { MODE_HOLD } else { MODE_MOVE });
        self.age = 160.0;
        self.duration = 64.0;
        self.brake_velocity = [0.0; 2];
        self.brake_acceleration = [0.0; 2];
        self.coefficients = [[0.0; 2]; 5];
        self.hold_goal = goal;
        self.hold_position = position;
        self.hold_error = norm(goal[0] - position[0], goal[1] - position[1]);
        self.changed = self.hold_error > 1e-7;
        self.innovation_at = if self.changed {
            now_us as f64 / 1000.0
        } else {
            f64::NAN
        };
        self.resume_integral = 0.0;
        self.stop_integral = 0.0;
        // Reference budget arrays are float32, including subsequent assignments.
        self.resume_budget = -log_f32(self.selector_rng.uniform_scalar().max(1e-8));
        self.stop_budget = -log_f32(self.brake_rng.uniform_scalar().max(1e-8));
    }

    fn run_motor(&mut self, window: &Window<'_>) -> Result<()> {
        kernels::motor_features(window, &mut self.motor);
        self.engine
            .motor(&self.motor.coarse, &self.motor.fine, &self.motor.dynamics)
            .expect("motor evaluation");
        let incoming = window.history[HISTORY_SAMPLES - 1];
        kernels::decode_geometry(
            self.engine.coefficients(),
            incoming,
            &self.mean_basis,
            &self.mean_carry,
            &mut self.heads,
        );
        let actual = match self.origin {
            Some(origin) => [
                (window.position[0] - origin[0]) as f32,
                (window.position[1] - origin[1]) as f32,
            ],
            None => [0.0f32, 0.0],
        };
        self.pairs.fill(0.0);
        let previous = if self.previous_valid {
            Some((self.previous.as_slice(), actual))
        } else {
            None
        };
        kernels::selector_inputs(&self.heads, previous, &mut self.geometry, &mut self.pairs);
        self.engine
            .choice(&self.geometry, &self.pairs, self.previous_valid)
            .expect("choice evaluation");
        let logits = self.engine.logits();
        softmax_into(&logits, &mut self.probabilities).expect("choice logits are finite");
        let sampled = self.motor_rng.categorical(&self.probabilities);
        let chosen = match self.chooser.as_mut() {
            None => sampled,
            Some(chooser) => chooser.motor(&MotorCandidates {
                probabilities: &self.probabilities,
                geometries: &self.heads,
                sampled,
            }),
        };
        require(chosen < HEADS, "chosen motor head is out of range")?;
        self.previous.copy_from_slice(
            &self.heads[chosen * GEOMETRY_STEPS * 2..(chosen + 1) * GEOMETRY_STEPS * 2],
        );
        self.last_head = chosen as i32;
        self.origin = Some(window.position);
        self.previous_valid = true;
        self.motor_evaluations += 1;
        let coefficients =
            self.engine.coefficients()[chosen * WEIGHTS * 2..(chosen + 1) * WEIGHTS * 2].to_vec();
        kernels::decode_selected(
            &coefficients,
            incoming,
            &self.velocity_basis[..COMMIT_SAMPLES * WEIGHTS],
            &self.carry_velocity[..COMMIT_SAMPLES],
            &mut self.action,
        );
        Ok(())
    }

    pub fn plan(
        &mut self,
        window: &Window<'_>,
        now_us: i64,
    ) -> Result<&[[f64; 2]; COMMIT_SAMPLES]> {
        let last = HISTORY_SAMPLES - 1;
        let known = window.valid[last] && window.available[last] <= now_us;
        let goal = if known {
            window.target[last]
        } else {
            window.position
        };
        if self.mode.is_none() {
            self.initialize(window.history, window.position, goal, now_us);
        }
        let old = self.mode.unwrap();

        if old != MODE_MOVE {
            self.previous_valid = false;
            for step in 0..COMMIT_SAMPLES {
                let slot = HISTORY_SAMPLES - COMMIT_SAMPLES + step;
                let offset = (step as i64 - 31) * 1000;
                if !(window.valid[slot] && window.available[slot] <= now_us + offset) {
                    continue;
                }
                let moved = norm(
                    window.target[slot][0] - self.hold_goal[0],
                    window.target[slot][1] - self.hold_goal[1],
                ) > 1e-7;
                if moved {
                    self.changed = true;
                    if self.innovation_at.is_nan() {
                        self.innovation_at = now_us as f64 / 1000.0 - 31.0 + step as f64;
                    }
                    break;
                }
            }
        }
        let innovation_age = if self.innovation_at.is_nan() {
            0.0
        } else {
            (now_us as f64 / 1000.0 - self.innovation_at).max(0.0)
        };

        let state = EventState {
            hold_age: self.age,
            hold_target: self.hold_goal,
            hold_position: self.hold_position,
            initial_error: self.hold_error,
            innovation_age,
            mode: old,
        };
        kernels::event_features(window, &state, &mut self.events);

        let innovation = old == MODE_HOLD
            && (self.changed
                || norm(
                    window.position[0] - self.hold_position[0],
                    window.position[1] - self.hold_position[1],
                ) > 1e-5
                || f64::from(self.events.raw[0]) > 10.0);
        let full_events = !self.skip_unused;
        let need_hazard = old == MODE_MOVE || innovation;

        let mut hazard = [0.0f32; 2];
        let mut have_hazard = false;
        if full_events {
            self.engine
                .events(&self.events.context, &mut self.outputs)?;
            hazard = self.outputs.hazard;
            have_hazard = true;
        } else if need_hazard {
            self.engine.hazard(&self.events.context, &mut hazard)?;
            have_hazard = true;
        }
        let probabilities = if have_hazard {
            hazard_probabilities(&hazard)
        } else {
            [0.0f32; 2]
        };
        if !probabilities.iter().all(|value| value.is_finite()) {
            return Err(Error::Numerical("Nonfinite event probabilities".into()));
        }

        let mut draws = [0.0f32; 2];
        self.selector_rng.uniform_into(&mut draws);
        let mut brake_weights = [0.0f32; HEADS];
        let mut brake_head = if full_events {
            softmax_into(&self.outputs.frequency, &mut brake_weights)?;
            Some(self.brake_rng.categorical(&brake_weights))
        } else {
            None
        };

        if old == MODE_MOVE {
            if known {
                self.stop_integral += -(1.0 - f64::from(probabilities[0])).max(1e-8).ln();
            }
            if self.stop_integral >= f64::from(self.stop_budget) && known {
                if !full_events {
                    self.engine.brake(&self.events.context, &mut self.outputs)?;
                    softmax_into(&self.outputs.frequency, &mut brake_weights)?;
                    brake_head = Some(self.brake_rng.categorical(&brake_weights));
                }
                let sampled = brake_head.unwrap();
                let head = match self.chooser.as_mut() {
                    None => sampled,
                    Some(chooser) => chooser.brake(&BrakeCandidates {
                        probabilities: &brake_weights,
                        durations: &self.outputs.duration,
                        sampled,
                    }),
                };
                require(head < HEADS, "chosen brake head is out of range")?;
                self.mode = Some(MODE_BRAKE);
                self.age = 0.0;
                self.brake_velocity = window.history[last];
                let recent = block_mean(window.history, HISTORY_SAMPLES - 8, HISTORY_SAMPLES);
                let earlier = block_mean(window.history, HISTORY_SAMPLES - 16, HISTORY_SAMPLES - 8);
                self.brake_acceleration = [
                    (recent[0] - earlier[0]) / 8.0,
                    (recent[1] - earlier[1]) / 8.0,
                ];
                self.duration = f64::from(self.outputs.duration[head]);
                for row in 0..5 {
                    let local = [
                        f64::from(self.outputs.coefficients[head * 10 + row * 2]),
                        f64::from(self.outputs.coefficients[head * 10 + row * 2 + 1]),
                    ];
                    self.coefficients[row] = [
                        local[0] * self.events.basis[0][0] + local[1] * self.events.basis[0][1],
                        local[0] * self.events.basis[1][0] + local[1] * self.events.basis[1][1],
                    ];
                }
                self.hold_goal = goal;
                self.hold_error = f64::from(self.events.raw[0]);
                self.changed = false;
                self.innovation_at = f64::NAN;
            }
        } else if old == MODE_HOLD {
            if innovation && known {
                self.resume_integral += -(1.0 - f64::from(probabilities[1])).max(1e-8).ln();
            }
            if innovation && self.resume_integral >= f64::from(self.resume_budget) && known {
                self.mode = Some(MODE_MOVE);
                self.age = 0.0;
                self.changed = false;
                self.innovation_at = f64::NAN;
                self.stop_integral = 0.0;
                self.stop_budget = -(f64::from(draws[0]).max(1e-8).ln()) as f32;
            }
        }

        if !full_events && brake_head.is_none() {
            self.brake_rng.discard_categorical(HEADS);
        }

        // Separate RNG streams allow event computation before the motor. The
        // discarded proposal never enters physical state or valid predecessor
        // geometry. Consuming its raw draw words keeps future seeds aligned.
        let moving = self.mode == Some(MODE_MOVE);
        if moving || full_events {
            self.run_motor(window)?;
        } else {
            self.motor_rng.discard_categorical(HEADS);
            self.last_head = -1;
        }

        match self.mode.unwrap() {
            MODE_BRAKE => {
                let times: Vec<f64> = (0..=COMMIT_SAMPLES)
                    .map(|step| self.age + step as f64)
                    .collect();
                kernels::c2_path(
                    self.brake_velocity,
                    self.brake_acceleration,
                    self.duration,
                    &self.coefficients,
                    &times,
                    &mut self.curve,
                );
                let mut total = [0.0f64; 2];
                for step in 0..COMMIT_SAMPLES {
                    for axis in 0..2 {
                        self.action[step][axis] =
                            self.curve[step + 1][axis] - self.curve[step][axis];
                        total[axis] += self.action[step][axis];
                    }
                }
                self.age += 32.0;
                if self.age >= self.duration {
                    self.mode = Some(MODE_HOLD);
                    self.age = (self.age - self.duration).max(0.0);
                    self.hold_position =
                        [window.position[0] + total[0], window.position[1] + total[1]];
                    self.hold_goal = goal;
                    self.hold_error = norm(
                        goal[0] - self.hold_position[0],
                        goal[1] - self.hold_position[1],
                    );
                    self.changed = false;
                    self.innovation_at = f64::NAN;
                    self.resume_integral = 0.0;
                    self.resume_budget = -(f64::from(draws[1]).max(1e-8).ln()) as f32;
                }
            }
            MODE_HOLD => {
                self.action = [[0.0; 2]; COMMIT_SAMPLES];
                self.age += 32.0;
            }
            _ => {
                self.age += 32.0;
            }
        }
        if self.mode != Some(MODE_MOVE) {
            self.previous_valid = false;
        }
        let bounded = self
            .action
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite() && value.abs() <= 1e5);
        if !bounded {
            return Err(Error::Numerical("Invalid finite primitive".into()));
        }
        if self.record_decisions {
            self.decisions.push(Decision {
                at_ms: now_us as f64 / 1000.0,
                mode: self.mode.unwrap(),
                head: self.last_head,
            });
        }
        self.decision_count += 1;
        Ok(&self.action)
    }

    /// Install a [`HeadChooser`], replacing any previous one.
    ///
    /// It survives [`reset`](Self::reset), because it is configuration rather
    /// than state.
    pub fn set_head_chooser(&mut self, chooser: impl HeadChooser + 'static) {
        self.chooser = Some(Box::new(chooser));
    }

    pub fn clear_head_chooser(&mut self) {
        self.chooser = None;
    }

    pub fn has_head_chooser(&self) -> bool {
        self.chooser.is_some()
    }

    pub fn prewarm(&mut self) {
        let history = vec![[0.0f64; 2]; HISTORY_SAMPLES];
        let target = vec![[100.0f64; 2]; HISTORY_SAMPLES];
        let available = vec![-640_000i64; HISTORY_SAMPLES];
        let known = vec![true; HISTORY_SAMPLES];
        let window = Window {
            history: &history,
            target: &target,
            available: &available,
            valid: &known,
            motion_known: &known,
            position: [0.0, 0.0],
            cut_us: 0,
        };
        let _ = self.run_motor(&window);
        let _ = self.plan(&window, 0);
        let context = [0.0f32; EVENT_CONTEXT_LEN];
        let _ = self.engine.events(&context, &mut self.outputs);
        let mut hazard = [0.0f32; 2];
        let _ = self.engine.hazard(&context, &mut hazard);
        let _ = self.engine.brake(&context, &mut self.outputs);
        self.reset(7);
    }
}

#[inline]
fn norm(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

fn block_mean(history: &[[f64; 2]], start: usize, end: usize) -> [f64; 2] {
    let mut total = [0.0f64; 2];
    for row in &history[start..end] {
        total[0] += row[0];
        total[1] += row[1];
    }
    let count = (end - start) as f64;
    [total[0] / count, total[1] / count]
}
