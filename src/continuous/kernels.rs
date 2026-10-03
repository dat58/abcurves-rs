use super::constants::*;
use crate::math::asinh_f32;

pub struct Window<'a> {
    pub history: &'a [[f64; 2]],
    pub target: &'a [[f64; 2]],
    pub available: &'a [i64],
    pub valid: &'a [bool],
    pub motion_known: &'a [bool],
    pub position: [f64; 2],
    pub cut_us: i64,
}

#[derive(Clone)]
pub struct MotorFeatures {
    pub coarse: [f32; COARSE_LEN],
    pub fine: [f32; FINE_LEN],
    pub dynamics: [f32; DYNAMICS_LEN],
}

impl Default for MotorFeatures {
    fn default() -> Self {
        Self {
            coarse: [0.0; COARSE_LEN],
            fine: [0.0; FINE_LEN],
            dynamics: [0.0; DYNAMICS_LEN],
        }
    }
}

#[derive(Clone)]
pub struct EventState {
    pub hold_age: f64,
    pub hold_target: [f64; 2],
    pub hold_position: [f64; 2],
    pub initial_error: f64,
    pub innovation_age: f64,
    pub mode: u8,
}

#[derive(Clone)]
pub struct EventFeatures {
    pub raw: [f32; EVENT_RAW_LEN],
    pub context: [f32; EVENT_CONTEXT_LEN],
    pub basis: [[f64; 2]; 2],
}

impl Default for EventFeatures {
    fn default() -> Self {
        Self {
            raw: [0.0; EVENT_RAW_LEN],
            context: [0.0; EVENT_CONTEXT_LEN],
            basis: [[0.0; 2]; 2],
        }
    }
}

#[inline]
fn norm(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

#[inline]
fn mean(history: &[[f64; 2]], start: usize, end: usize) -> (f64, f64) {
    let mut x = 0.0;
    let mut y = 0.0;
    for row in &history[start..end] {
        x += row[0];
        y += row[1];
    }
    let count = (end - start) as f64;
    (x / count, y / count)
}

pub fn motor_features(window: &Window<'_>, out: &mut MotorFeatures) {
    let Window {
        history,
        target,
        available,
        valid,
        motion_known,
        position,
        cut_us,
    } = *window;

    let mut known = [false; HISTORY_SAMPLES];
    let mut total_x = 0.0;
    let mut total_y = 0.0;
    let mut coverage = 0usize;
    for index in 0..HISTORY_SAMPLES {
        known[index] = valid[index] && available[index] <= cut_us + (index as i64 - 639) * 1000;
        if motion_known[index] {
            total_x += history[index][0];
            total_y += history[index][1];
            coverage += 1;
        }
    }

    let coarse = &mut out.coarse;
    let fine = &mut out.fine;
    let dynamics = &mut out.dynamics;

    let mut cumulative_x = 0.0;
    let mut cumulative_y = 0.0;
    let mut end = 0usize;
    for token in 0..COARSE_TOKENS {
        let width = if token < 30 { 16 } else { 4 };
        let begin = end;
        end += width;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0usize;
        for slot in begin..end {
            if motion_known[slot] {
                sum_x += history[slot][0];
                sum_y += history[slot][1];
                cumulative_x += history[slot][0];
                cumulative_y += history[slot][1];
                count += 1;
            }
        }
        let base = token * COARSE_FEATURES;
        coarse[base] = (sum_x / width as f64 / VELOCITY_SCALE) as f32;
        coarse[base + 1] = (sum_y / width as f64 / VELOCITY_SCALE) as f32;
        let index = end - 1;
        for axis in 0..2 {
            let mut relative = 0.0;
            if known[index] && motion_known[index] {
                let cumulative = if axis == 0 {
                    cumulative_x
                } else {
                    cumulative_y
                };
                let total = if axis == 0 { total_x } else { total_y };
                let point = position[axis] + cumulative - total;
                relative = ((target[index][axis] - point) / POSITION_SCALE).asinh();
            }
            coarse[base + 2 + axis] = relative as f32;
            let mut secant = 0.0;
            if index >= 32 && known[index] && known[index - 32] {
                secant = (target[index][axis] - target[index - 32][axis]) / 32.0;
            }
            coarse[base + 4 + axis] = (secant / VELOCITY_SCALE) as f32;
        }
        coarse[base + 6] = f32::from(u8::from(known[index]));
        coarse[base + 7] = (count as f64 / width as f64) as f32;
        coarse[base + 8] = if token < 30 { 1.0 } else { 0.25 };
    }

    for slot in 0..16 {
        for axis in 0..2 {
            fine[slot * 2 + axis] = if motion_known[624 + slot] {
                (history[624 + slot][axis] / VELOCITY_SCALE) as f32
            } else {
                0.0
            };
        }
        fine[32 + slot] = f32::from(u8::from(motion_known[624 + slot]));
    }
    for axis in 0..2 {
        fine[48 + axis] = if known[639] {
            (((target[639][axis] - position[axis]) / POSITION_SCALE).asinh()) as f32
        } else {
            0.0
        };
        fine[50 + axis] = if known[639] && known[607] {
            ((target[639][axis] - target[607][axis]) / 32.0 / VELOCITY_SCALE) as f32
        } else {
            0.0
        };
    }
    fine[52] = f32::from(u8::from(known[639]));
    fine[53] = (coverage as f64 / 640.0) as f32;

    let mut valid_acceleration = true;
    for slot in 40..48 {
        valid_acceleration = valid_acceleration && fine[slot] > 0.5;
    }
    for axis in 0..2 {
        let mut recent = 0.0f32;
        let mut earlier = 0.0f32;
        for step in 0..4 {
            recent += fine[(12 + step) * 2 + axis];
            earlier += fine[(8 + step) * 2 + axis];
        }
        recent /= 4.0;
        earlier /= 4.0;
        let mut acceleration = recent - earlier;
        acceleration *= VELOCITY_SCALE as f32;
        acceleration /= 4.0;
        acceleration = if valid_acceleration {
            acceleration / 0.25
        } else {
            0.0
        };
        dynamics[axis] = fine[30 + axis];
        dynamics[2 + axis] = acceleration;
    }
    let error_x = f64::from(fine[48]).sinh() * POSITION_SCALE;
    let error_y = f64::from(fine[49]).sinh() * POSITION_SCALE;
    let magnitude = (error_x * error_x + error_y * error_y + 100.0).sqrt();
    let known_target = fine[52] > 0.5;
    let direction_x = if known_target {
        (error_x / magnitude) as f32
    } else {
        0.0
    };
    let direction_y = if known_target {
        (error_y / magnitude) as f32
    } else {
        0.0
    };
    let relative_x = fine[30] - fine[50];
    let relative_y = fine[31] - fine[51];
    let acceleration_x = dynamics[2];
    let acceleration_y = dynamics[3];
    dynamics[4] = relative_x * direction_x + relative_y * direction_y;
    dynamics[5] = relative_x * (-direction_y) + relative_y * direction_x;
    dynamics[6] = acceleration_x * direction_x + acceleration_y * direction_y;
    dynamics[7] = acceleration_x * (-direction_y) + acceleration_y * direction_x;
}

pub fn event_features(window: &Window<'_>, state: &EventState, out: &mut EventFeatures) {
    let Window {
        history,
        target,
        available,
        valid,
        motion_known,
        position,
        cut_us,
    } = *window;

    let mut known = [false; HISTORY_SAMPLES];
    let mut known_context = [false; HISTORY_SAMPLES];
    let mut motion_count = 0usize;
    for index in 0..HISTORY_SAMPLES {
        known_context[index] =
            valid[index] && available[index] <= cut_us + (index as i64 - 639) * 1000;
        known[index] =
            known_context[index] && target[index][0].is_finite() && target[index][1].is_finite();
        motion_count += usize::from(motion_known[index]);
    }

    let latest = known[639];
    let goal_x = if latest { target[639][0] } else { position[0] };
    let goal_y = if latest { target[639][1] } else { position[1] };
    let error_x = goal_x - position[0];
    let error_y = goal_y - position[1];
    let distance = norm(error_x, error_y);
    let mut axis_x = error_x / distance.max(1e-8);
    let mut axis_y = error_y / distance.max(1e-8);

    let velocity_1 = history[639];
    let (velocity_8_x, velocity_8_y) = mean(history, 632, 640);
    let (velocity_32_x, velocity_32_y) = mean(history, 608, 640);
    let (previous_x, previous_y) = mean(history, 624, 632);
    let acceleration_x = (velocity_8_x - previous_x) / 8.0;
    let acceleration_y = (velocity_8_y - previous_y) / 8.0;
    let speed_1 = norm(velocity_1[0], velocity_1[1]);
    let speed_8 = norm(velocity_8_x, velocity_8_y);
    let speed_32 = norm(velocity_32_x, velocity_32_y);

    let mut secants = [[0.0f64; 2]; 3];
    for (slot, period) in [8usize, 32, 128].into_iter().enumerate() {
        if latest && known[639 - period] {
            secants[slot][0] = (goal_x - target[639 - period][0]) / period as f64;
            secants[slot][1] = (goal_y - target[639 - period][1]) / period as f64;
        }
    }
    let target_speed_8 = norm(secants[0][0], secants[0][1]);
    let target_speed_32 = norm(secants[1][0], secants[1][1]);
    let target_speed_128 = norm(secants[2][0], secants[2][1]);

    let mut growth = 0.0;
    if latest && known[607] {
        let old = norm(
            target[607][0] - (position[0] - velocity_32_x * 32.0),
            target[607][1] - (position[1] - velocity_32_y * 32.0),
        );
        growth = (distance - old) / 32.0;
    }

    let mut last_change: i64 = -1;
    let mut excursion = 0.0f64;
    let mut target_count = 0usize;
    for index in 512..640 {
        let offset = norm(target[index][0] - goal_x, target[index][1] - goal_y);
        if offset > 1e-8 || !known[index] {
            last_change = index as i64 - 512;
        }
        if known[index] {
            excursion = excursion.max(offset);
            target_count += 1;
        }
    }

    let raw = &mut out.raw;
    raw[0] = distance as f32;
    raw[1] = speed_1 as f32;
    raw[2] = speed_8 as f32;
    raw[3] = speed_32 as f32;
    raw[4] = (speed_8 / speed_32.max(0.005)) as f32;
    raw[5] = target_speed_32 as f32;
    raw[6] = target_speed_128 as f32;
    raw[7] = (velocity_8_x * axis_x + velocity_8_y * axis_y) as f32;
    raw[8] = (velocity_8_x * axis_y - velocity_8_y * axis_x).abs() as f32;
    raw[9] = growth as f32;
    raw[10] = (-(acceleration_x * velocity_8_x + acceleration_y * velocity_8_y)
        / speed_8.max(0.005)) as f32;
    raw[11] = excursion as f32;
    raw[12] = (127 - last_change) as f32;
    raw[13] = (target_count as f64 / 128.0) as f32;
    raw[14] = if latest { 1.0 } else { 0.0 };
    raw[15] = (motion_count as f64 / 640.0) as f32;
    raw[16] = state.hold_age.min(4096.0) as f32;
    raw[17] = if state.mode != MODE_MOVE {
        norm(goal_x - state.hold_target[0], goal_y - state.hold_target[1]) as f32
    } else {
        0.0
    };
    raw[18] = if state.mode != MODE_MOVE {
        norm(
            position[0] - state.hold_position[0],
            position[1] - state.hold_position[1],
        ) as f32
    } else {
        0.0
    };
    raw[19] = target_speed_8 as f32;
    raw[20] = ((velocity_8_x * secants[1][0] + velocity_8_y * secants[1][1])
        / (speed_8 * target_speed_32).max(1e-8)) as f32;
    raw[21] = if state.mode != MODE_MOVE {
        state.initial_error as f32
    } else {
        0.0
    };

    let context_goal_x = if known_context[639] {
        target[639][0]
    } else {
        position[0]
    };
    let context_goal_y = if known_context[639] {
        target[639][1]
    } else {
        position[1]
    };
    let context_error_x = context_goal_x - position[0];
    let context_error_y = context_goal_y - position[1];
    let context_distance = norm(context_error_x, context_error_y);
    if context_distance > 1e-6 {
        axis_x = context_error_x / context_distance.max(1e-9);
        axis_y = context_error_y / context_distance.max(1e-9);
    } else if speed_8 > 1e-6 {
        axis_x = velocity_8_x / speed_8.max(1e-9);
        axis_y = velocity_8_y / speed_8.max(1e-9);
    } else {
        axis_x = 1.0;
        axis_y = 0.0;
    }
    out.basis = [[axis_x, -axis_y], [axis_y, axis_x]];

    let mut extra = [0.0f64; 10];
    extra[0] = velocity_1[0] * axis_x + velocity_1[1] * axis_y;
    extra[1] = -velocity_1[0] * axis_y + velocity_1[1] * axis_x;
    extra[2] = (acceleration_x * axis_x + acceleration_y * axis_y) * 8.0;
    extra[3] = (-acceleration_x * axis_y + acceleration_y * axis_x) * 8.0;
    for (slot, period) in [32usize, 128].into_iter().enumerate() {
        let mut secant_x = 0.0;
        let mut secant_y = 0.0;
        if known_context[639] && known_context[639 - period] {
            secant_x = (context_goal_x - target[639 - period][0]) / period as f64;
            secant_y = (context_goal_y - target[639 - period][1]) / period as f64;
        }
        extra[4 + slot * 2] = secant_x * axis_x + secant_y * axis_y;
        extra[5 + slot * 2] = -secant_x * axis_y + secant_y * axis_x;
    }
    extra[8] = -velocity_8_x * axis_y + velocity_8_y * axis_x;
    extra[9] = state.innovation_age.min(2048.0) / 128.0;

    for slot in 0..EVENT_RAW_LEN {
        let value = if slot == 16 {
            f64::from(raw[slot]).min(256.0)
        } else {
            f64::from(raw[slot])
        };
        out.context[slot] = (value / EVENT_SCALES[slot]).asinh() as f32;
    }
    for slot in 0..10 {
        out.context[EVENT_RAW_LEN + slot] = extra[slot].asinh() as f32;
    }
}

pub fn c2_path(
    velocity: [f64; 2],
    acceleration: [f64; 2],
    duration: f64,
    coefficients: &[[f64; 2]; 5],
    times: &[f64],
    out: &mut [[f64; 2]],
) {
    for (slot, &time) in times.iter().enumerate() {
        let s = (time / duration).clamp(0.0, 1.0);
        let z = 2.0 * s - 1.0;
        let envelope = 64.0 * s.powi(3) * (1.0 - s).powi(3);
        let carry = s - 6.0 * s.powi(3) + 8.0 * s.powi(4) - 3.0 * s.powi(5);
        let acceleration_weight = 0.5 * s * s * (1.0 - s).powi(3);
        let goal = 10.0 * s.powi(3) - 15.0 * s.powi(4) + 6.0 * s.powi(5);
        let r0 = envelope;
        let r1 = envelope * z;
        let r2 = envelope * (3.0 * z * z - 1.0) / 2.0;
        let r3 = envelope * (5.0 * z * z * z - 3.0 * z) / 2.0;
        for axis in 0..2 {
            let residual = r0 * coefficients[1][axis]
                + r1 * coefficients[2][axis]
                + r2 * coefficients[3][axis]
                + r3 * coefficients[4][axis];
            out[slot][axis] = carry * duration * velocity[axis]
                + acceleration_weight * (duration * duration) * acceleration[axis]
                + goal * coefficients[0][axis]
                + residual;
        }
    }
}

pub fn selector_inputs(
    heads: &[f32],
    previous: Option<(&[f32], [f32; 2])>,
    geometry: &mut [f32],
    pairs: &mut [f32],
) {
    let mut previous_sum = [0.0f32; 2];
    let mut previous_prefix_sum = [0.0f32; 2];
    if let Some((previous, _)) = previous {
        for axis in 0..2 {
            for step in 0..GEOMETRY_STEPS {
                previous_sum[axis] += previous[step * 2 + axis];
                if step < 8 {
                    previous_prefix_sum[axis] += previous[step * 2 + axis];
                }
            }
        }
    }
    for head in 0..HEADS {
        let base = head * GEOMETRY_STEPS * 2;
        for step in 0..8 {
            for axis in 0..2 {
                let value = heads[base + step * 2 + axis];
                geometry[head * GEOMETRY_STEPS + step * 2 + axis] = asinh_f32(value);
                if let Some((previous, _)) = previous {
                    pairs[head * PAIR_LEN + step * 2 + axis] =
                        asinh_f32(previous[(8 + step) * 2 + axis] - value);
                }
            }
        }
        if let Some((_, actual)) = previous {
            for axis in 0..2 {
                let mut total = 0.0f32;
                for step in 0..8 {
                    total += heads[base + step * 2 + axis];
                }
                let mut endpoint = previous_sum[axis] * 4.0 - actual[axis];
                endpoint -= total * 4.0;
                endpoint /= 32.0;
                let feedback = (previous_prefix_sum[axis] * 4.0 - actual[axis]) / 32.0;
                pairs[head * PAIR_LEN + 16 + axis] = asinh_f32(endpoint);
                pairs[head * PAIR_LEN + 18 + axis] = asinh_f32(feedback);
            }
        }
    }
}

pub fn decoder_geometry(velocity_basis: &[f64], carry_velocity: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut basis = vec![0.0f64; GEOMETRY_STEPS * WEIGHTS];
    let mut carry = vec![0.0f64; GEOMETRY_STEPS];
    for step in 0..GEOMETRY_STEPS {
        for weight in 0..WEIGHTS {
            let mut total = 0.0;
            for sub in 0..4 {
                total += velocity_basis[(step * 4 + sub) * WEIGHTS + weight];
            }
            basis[step * WEIGHTS + weight] = total / 4.0;
        }
        let mut total = 0.0;
        for sub in 0..4 {
            total += carry_velocity[step * 4 + sub];
        }
        carry[step] = total / 4.0;
    }
    (basis, carry)
}

pub fn decode_geometry(
    coefficients: &[f32],
    incoming: [f64; 2],
    mean_basis: &[f64],
    mean_carry: &[f64],
    out: &mut [f32],
) {
    for head in 0..HEADS {
        for step in 0..GEOMETRY_STEPS {
            for axis in 0..2 {
                let mut total = 0.0f64;
                for weight in 0..WEIGHTS {
                    total += mean_basis[step * WEIGHTS + weight]
                        * f64::from(coefficients[(head * WEIGHTS + weight) * 2 + axis]);
                }
                total += mean_carry[step] * incoming[axis];
                out[(head * GEOMETRY_STEPS + step) * 2 + axis] = total as f32;
            }
        }
    }
}

pub fn decode_selected(
    coefficients: &[f32],
    incoming: [f64; 2],
    velocity_basis: &[f64],
    carry_velocity: &[f64],
    out: &mut [[f64; 2]],
) {
    for (step, row) in out.iter_mut().enumerate() {
        for axis in 0..2 {
            let mut total = 0.0f64;
            for weight in 0..WEIGHTS {
                total += velocity_basis[step * WEIGHTS + weight]
                    * f64::from(coefficients[weight * 2 + axis]);
            }
            row[axis] = total + carry_velocity[step] * incoming[axis];
        }
    }
}
