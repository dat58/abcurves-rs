use super::model::*;
use crate::error::{Error, Result};

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX2: u64 = 0x94D0_49BB_1331_11EB;
const I32_MIN_VALUE: i64 = -(i32::MAX as i64) - 1;
const LN2_Q14: i32 = 11357;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Observe,
    Generate,
}

#[inline]
pub fn abs_u64(value: i64) -> u64 {
    value.unsigned_abs()
}

#[inline]
pub fn round_div_even_i64(value: i64, divisor: u64) -> i64 {
    let absolute = abs_u64(value);
    let mut quotient = absolute / divisor;
    let remainder = absolute % divisor;
    let twice = remainder.wrapping_mul(2);
    if twice > divisor || (twice == divisor && quotient & 1 == 1) {
        quotient += 1;
    }
    let result = quotient as i64;
    if value < 0 { -result } else { result }
}

#[inline]
pub fn round_shift_even_i64(value: i64, shift: u32) -> i32 {
    round_div_even_i64(value, 1u64 << shift) as i32
}

#[inline]
fn clamp_i16(value: i64, low: i16, high: i16) -> i16 {
    value.clamp(i64::from(low), i64::from(high)) as i16
}

#[inline]
fn lerp_i32(a: i32, b: i32, remainder: u32, interval_q: u32) -> i32 {
    let delta = i64::from(b) - i64::from(a);
    a + round_div_even_i64(delta * i64::from(remainder), 1u64 << interval_q) as i32
}

#[inline]
pub fn requant_q31(accumulator: i32, multiplier: i32) -> i32 {
    round_div_even_i64(i64::from(accumulator) * i64::from(multiplier), 1u64 << 31) as i32
}

#[inline]
pub fn dot_requant(weights: &[i8], vector: &[i16], multiplier: i32) -> i32 {
    let mut accumulator = 0i32;
    for (weight, value) in weights.iter().zip(vector) {
        accumulator += i32::from(*weight) * i32::from(*value);
    }
    requant_q31(accumulator, multiplier)
}

pub fn isqrt_u64(value: u64) -> u64 {
    let mut remaining = value;
    let mut result = 0u64;
    let mut bit = 1u64 << 62;
    while bit > remaining {
        bit >>= 2;
    }
    while bit != 0 {
        if remaining >= result + bit {
            remaining -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    result
}

fn speed_q16(x: i32, y: i32) -> u32 {
    let ax = u64::from(x.unsigned_abs());
    let ay = u64::from(y.unsigned_abs());
    let speed = isqrt_u64(ax * ax + ay * ay);
    speed.min(u64::from(u32::MAX)) as u32
}

fn normalize_q15(x_q16: i32, y_q16: i32) -> ([i16; 2], u32) {
    let speed = speed_q16(x_q16, y_q16);
    if speed == 0 {
        return ([0, 0], speed);
    }
    let tx = round_div_even_i64(i64::from(x_q16) * 32767, u64::from(speed));
    let ty = round_div_even_i64(i64::from(y_q16) * 32767, u64::from(speed));
    (
        [clamp_i16(tx, -32767, 32767), clamp_i16(ty, -32767, 32767)],
        speed,
    )
}

fn ratio_q15(numerator: i64, denominator: u64) -> i32 {
    if denominator == 0 {
        return 0;
    }
    let negative = numerator < 0;
    let mut remainder = abs_u64(numerator);
    let mut quotient = remainder / denominator;
    remainder %= denominator;
    for _ in 0..TANGENT_Q {
        quotient <<= 1;
        if remainder >= denominator - remainder {
            remainder -= denominator - remainder;
            quotient |= 1;
        } else {
            remainder <<= 1;
        }
    }
    let round_up = remainder > denominator - remainder
        || (remainder == denominator - remainder && quotient & 1 == 1);
    quotient += u64::from(round_up);
    let result = quotient as i64;
    (if negative { -result } else { result }) as i32
}

#[inline]
fn mul_q16(value_q14: i32, factor_q16: i32) -> i32 {
    round_shift_even_i64(i64::from(value_q14) * i64::from(factor_q16), 16)
}

fn splitmix64(seed: u64, ordinal: u64) -> u64 {
    let mut value = seed.wrapping_add(GAMMA.wrapping_mul(ordinal));
    value = (value ^ (value >> 30)).wrapping_mul(MIX1);
    value = (value ^ (value >> 27)).wrapping_mul(MIX2);
    value ^ (value >> 31)
}

#[inline]
fn offset_ring(row: usize) -> usize {
    let x = (row / SIDE) as i32 - RADIUS;
    let y = (row % SIDE) as i32 - RADIUS;
    x.unsigned_abs().max(y.unsigned_abs()) as usize
}

#[inline]
fn sign_i32(value: i32) -> i32 {
    i32::from(value > 0) - i32::from(value < 0)
}

impl FixedModel {
    fn sigmoid_q15(&self, value_q14: i32) -> i32 {
        let clamped = value_q14.clamp(-LUT_DOMAIN_Q14, LUT_DOMAIN_Q14);
        let position = (clamped + LUT_DOMAIN_Q14) as u32;
        let index = (position >> (GATE_Q - 4)) as usize;
        if index >= LUT_INTERVALS {
            return i32::from(self.sigmoid[LUT_INTERVALS]);
        }
        let remainder = position & ((1 << (GATE_Q - 4)) - 1);
        lerp_i32(
            i32::from(self.sigmoid[index]),
            i32::from(self.sigmoid[index + 1]),
            remainder,
            GATE_Q - 4,
        )
    }

    fn tanh_q15(&self, value_q14: i32) -> i32 {
        let clamped = value_q14.clamp(-LUT_DOMAIN_Q14, LUT_DOMAIN_Q14);
        let position = (clamped + LUT_DOMAIN_Q14) as u32;
        let index = (position >> (GATE_Q - 4)) as usize;
        if index >= LUT_INTERVALS {
            return i32::from(self.tanh[LUT_INTERVALS]);
        }
        let remainder = position & ((1 << (GATE_Q - 4)) - 1);
        lerp_i32(
            i32::from(self.tanh[index]),
            i32::from(self.tanh[index + 1]),
            remainder,
            GATE_Q - 4,
        )
    }

    fn exp_neg_q20(&self, value_q14: i32) -> u32 {
        let magnitude = (-i64::from(value_q14)).clamp(0, i64::from(EXP_DOMAIN_Q14)) as u32;
        let index = (magnitude >> (GATE_Q - 4)) as usize;
        if index >= LUT_INTERVALS {
            return self.exp[LUT_INTERVALS];
        }
        let remainder = magnitude & ((1 << (GATE_Q - 4)) - 1);
        lerp_i32(
            self.exp[index] as i32,
            self.exp[index + 1] as i32,
            remainder,
            GATE_Q - 4,
        ) as u32
    }

    fn ln_positive_q14(&self, value_q20: u64) -> i32 {
        if value_q20 == 0 {
            return -EXP_DOMAIN_Q14;
        }
        let mut exponent = 0u32;
        while exponent < 63 && (value_q20 >> (exponent + 1)) != 0 {
            exponent += 1;
        }
        let mantissa_q24 = if exponent >= 24 {
            value_q20 >> (exponent - 24)
        } else {
            value_q20 << (24 - exponent)
        };
        let fraction_q24 = (mantissa_q24 - (1u64 << 24)) as u32;
        let index = (fraction_q24 >> 16) as usize;
        let log_mantissa = if index >= LUT_INTERVALS {
            i32::from(self.log[LUT_INTERVALS])
        } else {
            let remainder = fraction_q24 & 0xffff;
            lerp_i32(
                i32::from(self.log[index]),
                i32::from(self.log[index + 1]),
                remainder,
                16,
            )
        };
        log_mantissa + (exponent as i32 - EXP_Q) * LN2_Q14
    }

    fn ln1p_q14_from_q16(&self, value_q16: u32) -> i32 {
        self.ln_positive_q14((u64::from(value_q16) + (1u64 << 16)) << 4)
    }

    #[inline]
    fn bias(&self, index: usize) -> i32 {
        self.biases[index]
    }

    #[inline]
    fn multiplier(&self, index: usize) -> i32 {
        self.multipliers[index]
    }
}

#[derive(Clone)]
pub struct FixedRenderer {
    pub mode: Mode,
    pub hidden: [i16; HIDDEN],
    pub accumulator_q16: [i32; 2],
    pub previous_emit: [i16; 2],
    pub last_nonzero: [i16; 2],
    pub last_axis_nonzero: [i16; 2],
    pub run_active: u8,
    pub run_length: u16,
    pub active_ring: [u8; RECENT_WINDOW],
    pub active_ring_pos: u8,
    pub active_ring_count: u8,
    pub prefix_raw: Vec<[i16; 2]>,
    pub prefix_pos: u16,
    pub prefix_count: u16,
    pub regime_q8: [i16; 5],
    pub tangent_q15: [i16; 2],
    pub last_smooth_q16: [i32; 2],
    pub previous_speed_q16: u32,
    pub event_seed: u64,
    pub generation_tick: u64,
    pub last_emit_probability_q15: u32,
    pub last_joint_index: u16,
    logits: Vec<i32>,
    feature: [i16; FEATURES],
    next_hidden: [i16; HIDDEN],
}

#[derive(Clone, Debug)]
pub struct Boundary {
    pub regime_q8: [i16; 5],
    pub previous_emit: [i16; 2],
    pub last_nonzero: [i16; 2],
    pub last_axis_nonzero: [i16; 2],
    pub last_smooth_q16: [i32; 2],
    pub run_length: u16,
    pub run_active: u8,
    pub active_ring: [u8; RECENT_WINDOW],
    pub active_ring_pos: u8,
    pub active_ring_count: u8,
}

impl Default for Boundary {
    fn default() -> Self {
        Self {
            regime_q8: [0; 5],
            previous_emit: [0; 2],
            last_nonzero: [0; 2],
            last_axis_nonzero: [0; 2],
            last_smooth_q16: [0; 2],
            run_length: 0,
            run_active: 0,
            active_ring: [0; RECENT_WINDOW],
            active_ring_pos: 0,
            active_ring_count: 0,
        }
    }
}

impl Default for FixedRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedRenderer {
    pub fn new() -> Self {
        Self {
            mode: Mode::Observe,
            hidden: [0; HIDDEN],
            accumulator_q16: [0; 2],
            previous_emit: [0; 2],
            last_nonzero: [0; 2],
            last_axis_nonzero: [0; 2],
            run_active: 0,
            run_length: 0,
            active_ring: [0; RECENT_WINDOW],
            active_ring_pos: 0,
            active_ring_count: 0,
            prefix_raw: vec![[0; 2]; PREFIX_WINDOW],
            prefix_pos: 0,
            prefix_count: 0,
            regime_q8: [0; 5],
            tangent_q15: [32767, 0],
            last_smooth_q16: [0; 2],
            previous_speed_q16: 0,
            event_seed: 0,
            generation_tick: 0,
            last_emit_probability_q15: 0,
            last_joint_index: 0,
            logits: vec![0; GRID],
            feature: [0; FEATURES],
            next_hidden: [0; HIDDEN],
        }
    }

    pub fn observe_prefix(&mut self, dx: i16, dy: i16) -> Result<()> {
        if self.mode != Mode::Observe {
            return Err(Error::Mode("renderer is not observing".into()));
        }
        self.prefix_raw[usize::from(self.prefix_pos)] = [dx, dy];
        self.prefix_pos = (self.prefix_pos + 1) % PREFIX_WINDOW as u16;
        if usize::from(self.prefix_count) < PREFIX_WINDOW {
            self.prefix_count += 1;
        }
        Ok(())
    }

    pub fn prefix_value(&self, chronological_index: u16, axis: usize) -> i16 {
        let oldest = if usize::from(self.prefix_count) == PREFIX_WINDOW {
            self.prefix_pos
        } else {
            0
        };
        let physical = (usize::from(oldest) + usize::from(chronological_index)) % PREFIX_WINDOW;
        self.prefix_raw[physical][axis]
    }

    pub fn gru_step(&mut self, model: &FixedModel) {
        let w_ih = &model.weights[..4800];
        let w_hh = &model.weights[4800..24000];
        for index in 0..HIDDEN {
            let reset_row = index;
            let update_row = HIDDEN + index;
            let new_row = 2 * HIDDEN + index;
            let reset_input = dot_requant(
                &w_ih[reset_row * FEATURES..(reset_row + 1) * FEATURES],
                &self.feature,
                model.multiplier(0),
            ) + model.bias(reset_row);
            let reset_hidden = dot_requant(
                &w_hh[reset_row * HIDDEN..(reset_row + 1) * HIDDEN],
                &self.hidden,
                model.multiplier(3),
            ) + model.bias(GATES + reset_row);
            let update_input = dot_requant(
                &w_ih[update_row * FEATURES..(update_row + 1) * FEATURES],
                &self.feature,
                model.multiplier(1),
            ) + model.bias(update_row);
            let update_hidden = dot_requant(
                &w_hh[update_row * HIDDEN..(update_row + 1) * HIDDEN],
                &self.hidden,
                model.multiplier(4),
            ) + model.bias(GATES + update_row);
            let new_input = dot_requant(
                &w_ih[new_row * FEATURES..(new_row + 1) * FEATURES],
                &self.feature,
                model.multiplier(2),
            ) + model.bias(new_row);
            let new_hidden = dot_requant(
                &w_hh[new_row * HIDDEN..(new_row + 1) * HIDDEN],
                &self.hidden,
                model.multiplier(5),
            ) + model.bias(GATES + new_row);
            let r = model.sigmoid_q15(reset_input + reset_hidden);
            let z = model.sigmoid_q15(update_input + update_hidden);
            let reset_hg =
                round_div_even_i64(i64::from(r) * i64::from(new_hidden), 1u64 << PROB_Q) as i32;
            let n = model.tanh_q15(new_input + reset_hg);
            let mixed = i64::from((1i32 << PROB_Q) - z) * i64::from(n)
                + i64::from(z) * i64::from(self.hidden[index]);
            let value = round_div_even_i64(mixed, 1u64 << PROB_Q);
            self.next_hidden[index] = clamp_i16(value, -32767, 32767);
        }
        self.hidden = self.next_hidden;
    }

    pub fn online_gru_step_q8(
        &mut self,
        model: &FixedModel,
        feature: &[i16; FEATURES],
    ) -> Result<()> {
        if self.mode != Mode::Observe {
            return Err(Error::Mode("renderer is not observing".into()));
        }
        self.feature = *feature;
        self.gru_step(model);
        Ok(())
    }

    pub fn install_boundary(&mut self, boundary: &Boundary) -> Result<()> {
        if self.mode != Mode::Observe || usize::from(self.prefix_count) != PREFIX_WINDOW {
            return Err(Error::Mode("renderer cannot install a boundary now".into()));
        }
        self.regime_q8 = boundary.regime_q8;
        self.previous_emit = boundary.previous_emit;
        self.last_nonzero = boundary.last_nonzero;
        self.last_axis_nonzero = boundary.last_axis_nonzero;
        self.last_smooth_q16 = boundary.last_smooth_q16;
        self.run_length = boundary.run_length;
        self.run_active = boundary.run_active;
        self.active_ring = boundary.active_ring;
        self.active_ring_pos = boundary.active_ring_pos;
        self.active_ring_count = boundary.active_ring_count;
        let (tangent, speed) = normalize_q15(self.last_smooth_q16[0], self.last_smooth_q16[1]);
        self.tangent_q15 = tangent;
        self.previous_speed_q16 = speed;
        if self.tangent_q15 == [0, 0] {
            self.tangent_q15[0] = 32767;
        }
        self.accumulator_q16 = [0; 2];
        Ok(())
    }

    pub fn begin(&mut self, event_seed: u64) -> Result<()> {
        if self.mode != Mode::Observe || usize::from(self.prefix_count) != PREFIX_WINDOW {
            return Err(Error::Mode("renderer cannot begin an event now".into()));
        }
        self.accumulator_q16 = [0; 2];
        self.event_seed = event_seed;
        self.generation_tick = 0;
        self.mode = Mode::Generate;
        Ok(())
    }

    fn update_history(&mut self, dx: i16, dy: i16) {
        let active = u8::from(dx != 0 || dy != 0);
        if self.run_length == 0 || active != self.run_active {
            self.run_length = 1;
        } else {
            self.run_length = self.run_length.saturating_add(1);
        }
        self.run_active = active;
        self.previous_emit = [dx, dy];
        if active != 0 {
            self.last_nonzero = [dx, dy];
        }
        if dx != 0 {
            self.last_axis_nonzero[0] = dx;
        }
        if dy != 0 {
            self.last_axis_nonzero[1] = dy;
        }
        self.active_ring[usize::from(self.active_ring_pos)] = active;
        self.active_ring_pos = (self.active_ring_pos + 1) % RECENT_WINDOW as u8;
        if usize::from(self.active_ring_count) < RECENT_WINDOW {
            self.active_ring_count += 1;
        }
    }

    fn build_future_feature(
        &mut self,
        model: &FixedModel,
        smooth_x: i32,
        smooth_y: i32,
        accumulator_q16: &[i64; 2],
    ) -> [i16; 2] {
        let (new_tangent, speed) = normalize_q15(smooth_x, smooth_y);
        if speed != 0 {
            self.tangent_q15 = new_tangent;
        }
        let normal_q15 = [-self.tangent_q15[1], self.tangent_q15[0]];

        let accel = (i64::from(speed) - i64::from(self.previous_speed_q16))
            .clamp(-(3 << SMOOTH_Q), 3 << SMOOTH_Q);
        let mut curvature_q8 = 0i32;
        if self.previous_speed_q16 > 0 && speed > 0 {
            let dot_x = i64::from(self.last_smooth_q16[0]) * i64::from(smooth_x);
            let dot_y = i64::from(self.last_smooth_q16[1]) * i64::from(smooth_y);
            // Two INT32_MIN axes can sum to 2^63; saturation preserves Q15 rounding.
            let dot = if dot_x > 0 && dot_y > i64::MAX - dot_x {
                i64::MAX
            } else {
                dot_x + dot_y
            };
            let denominator = u64::from(self.previous_speed_q16) * u64::from(speed);
            let cosine_q15 = ratio_q15(dot, denominator).clamp(-(1 << TANGENT_Q), 1 << TANGENT_Q);
            curvature_q8 = round_div_even_i64(
                i64::from((1 << TANGENT_Q) - cosine_q15) << FEATURE_Q,
                1u64 << TANGENT_Q,
            ) as i32;
        }

        let tangent_dot = accumulator_q16[0] * i64::from(self.tangent_q15[0])
            + accumulator_q16[1] * i64::from(self.tangent_q15[1]);
        let normal_dot = accumulator_q16[0] * i64::from(normal_q15[0])
            + accumulator_q16[1] * i64::from(normal_q15[1]);
        let previous_tangent_dot = i64::from(self.previous_emit[0])
            * i64::from(self.tangent_q15[0])
            + i64::from(self.previous_emit[1]) * i64::from(self.tangent_q15[1]);
        let previous_normal_dot = i64::from(self.previous_emit[0]) * i64::from(normal_q15[0])
            + i64::from(self.previous_emit[1]) * i64::from(normal_q15[1]);
        let last_tangent_dot = i64::from(self.last_nonzero[0]) * i64::from(self.tangent_q15[0])
            + i64::from(self.last_nonzero[1]) * i64::from(self.tangent_q15[1]);
        let last_normal_dot = i64::from(self.last_nonzero[0]) * i64::from(normal_q15[0])
            + i64::from(self.last_nonzero[1]) * i64::from(normal_q15[1]);
        let mut inactive = 0u32;
        for index in 0..usize::from(self.active_ring_count) {
            inactive += u32::from(self.active_ring[index] == 0);
        }

        let limit = 4 << FEATURE_Q;
        self.feature[0] = clamp_i16(
            round_div_even_i64(i64::from(speed), 1u64 << 10),
            i16::MIN,
            i16::MAX,
        );
        self.feature[1] = clamp_i16(
            i64::from(round_shift_even_i64(accel, SMOOTH_Q - FEATURE_Q)),
            -768,
            768,
        );
        self.feature[2] = clamp_i16(i64::from(curvature_q8), i16::MIN, i16::MAX);
        self.feature[3] =
            round_shift_even_i64(i64::from(self.tangent_q15[0]), TANGENT_Q - FEATURE_Q) as i16;
        self.feature[4] =
            round_shift_even_i64(i64::from(self.tangent_q15[1]), TANGENT_Q - FEATURE_Q) as i16;
        self.feature[5] = clamp_i16(
            i64::from(round_shift_even_i64(
                tangent_dot,
                SMOOTH_Q + TANGENT_Q - FEATURE_Q,
            )),
            -limit,
            limit,
        );
        self.feature[6] = clamp_i16(
            i64::from(round_shift_even_i64(
                normal_dot,
                SMOOTH_Q + TANGENT_Q - FEATURE_Q,
            )),
            -limit,
            limit,
        );
        self.feature[7] = clamp_i16(
            i64::from(round_shift_even_i64(
                previous_tangent_dot,
                TANGENT_Q - FEATURE_Q + 2,
            )),
            -limit,
            limit,
        );
        self.feature[8] = clamp_i16(
            i64::from(round_shift_even_i64(
                previous_normal_dot,
                TANGENT_Q - FEATURE_Q + 2,
            )),
            -limit,
            limit,
        );
        self.feature[9] = i16::from(self.run_active) << FEATURE_Q;
        self.feature[10] = (self.run_length.min(64) * 4) as i16;
        self.feature[11] = if self.active_ring_count == 0 {
            256
        } else {
            round_div_even_i64(
                i64::from(inactive) << FEATURE_Q,
                u64::from(self.active_ring_count),
            ) as i16
        };
        self.feature[12] = clamp_i16(
            i64::from(round_shift_even_i64(
                i64::from(model.ln1p_q14_from_q16(speed)),
                LOG_Q - FEATURE_Q,
            )),
            i16::MIN,
            i16::MAX,
        );
        self.feature[13] = clamp_i16(
            i64::from(round_shift_even_i64(
                last_tangent_dot,
                TANGENT_Q - FEATURE_Q + 2,
            )),
            -limit,
            limit,
        );
        self.feature[14] = clamp_i16(
            i64::from(round_shift_even_i64(
                last_normal_dot,
                TANGENT_Q - FEATURE_Q + 2,
            )),
            -limit,
            limit,
        );
        self.feature[15..20].copy_from_slice(&self.regime_q8);
        self.last_smooth_q16 = [smooth_x, smooth_y];
        self.previous_speed_q16 = speed;
        normal_q15
    }

    pub fn step_q16(
        &mut self,
        model: &FixedModel,
        smooth_dx_q16: i32,
        smooth_dy_q16: i32,
    ) -> Result<[i16; 2]> {
        if self.mode != Mode::Generate {
            return Err(Error::Mode("renderer is not generating".into()));
        }
        // Keep pre-emission headroom without changing the stored-state ABI.
        let accumulator_q16 = [
            i64::from(self.accumulator_q16[0]) + i64::from(smooth_dx_q16),
            i64::from(self.accumulator_q16[1]) + i64::from(smooth_dy_q16),
        ];
        let normal_q15 =
            self.build_future_feature(model, smooth_dx_q16, smooth_dy_q16, &accumulator_q16);
        self.gru_step(model);

        let w_emit = &model.weights[24000..24080];
        let w_off = &model.weights[24080..];
        let emit_logit = dot_requant(w_emit, &self.hidden, model.multiplier(6))
            + model.bias(480)
            + model.config[CFG_EMIT_BIAS_Q14];
        let emit_scaled = mul_q16(emit_logit, model.config[CFG_INV_EMIT_Q16]);
        let emit_probability = model.sigmoid_q15(emit_scaled);
        for row in 0..GRID {
            self.logits[row] = dot_requant(
                &w_off[row * HIDDEN..(row + 1) * HIDDEN],
                &self.hidden,
                model.multiplier(7 + row),
            ) + model.bias(481 + row);
        }

        let rings = RADIUS as usize + 1;
        let mut maxima = vec![I32_MIN_VALUE as i32; rings];
        let mut sums = vec![0u64; rings];
        let mut lse_by_ring = vec![0i32; rings];
        // Interleave the rings while retaining each ring's arithmetic order.
        for row in 0..GRID {
            let ring = offset_ring(row);
            if self.logits[row] > maxima[ring] {
                maxima[ring] = self.logits[row];
            }
        }
        for row in 0..GRID {
            let ring = offset_ring(row);
            sums[ring] += u64::from(model.exp_neg_q20(self.logits[row] - maxima[ring]));
        }
        for ring in 0..rings {
            lse_by_ring[ring] = maxima[ring] + model.ln_positive_q14(sums[ring]);
        }
        for row in 0..GRID {
            let lse = lse_by_ring[offset_ring(row)];
            self.logits[row] = mul_q16(lse, model.config[CFG_INV_MAG_Q16])
                + mul_q16(self.logits[row] - lse, model.config[CFG_INV_DIR_Q16]);
        }
        for row in 0..GRID {
            let offset_x = (row / SIDE) as i32 - RADIUS;
            let offset_y = (row % SIDE) as i32 - RADIUS;
            let lateral = (i64::from(offset_x) * i64::from(normal_q15[0])
                + i64::from(offset_y) * i64::from(normal_q15[1]))
            .abs();
            let excess = lateral as i32 - model.config[CFG_LATERAL_FREE_Q15];
            if excess > 0 {
                self.logits[row] -= round_shift_even_i64(
                    i64::from(excess) * i64::from(model.config[CFG_AF_Q16]),
                    TANGENT_Q + 16 - GATE_Q,
                );
            }
        }
        let mut total = 0u64;
        let maximum = *self.logits.iter().max().unwrap();
        for row in 0..GRID {
            let exponential = model.exp_neg_q20(self.logits[row] - maximum);
            self.logits[row] = exponential as i32;
            total += u64::from(exponential);
        }

        let last_emit_top23 =
            (splitmix64(self.event_seed, 2 * self.generation_tick + 1) >> 41) as u32;
        let last_offset_top23 =
            (splitmix64(self.event_seed, 2 * self.generation_tick + 2) >> 41) as u32;
        let mut emit = (u64::from(2 * last_emit_top23 + 1) << PROB_Q)
            < u64::from(emit_probability as u32) * 2 * (1u64 << 23);
        let force_release = model.config[CFG_FORCE_RELEASE_Q16] as u32;
        let safety = abs_u64(accumulator_q16[0]) >= u64::from(force_release)
            || abs_u64(accumulator_q16[1]) >= u64::from(force_release);
        if safety {
            emit = true;
        } else {
            let zero_intent = u64::from(model.config[CFG_ZERO_INTENT_Q16] as u32);
            let zero_debt = u64::from(model.config[CFG_ZERO_DEBT_Q16] as u32);
            let quiet = abs_u64(i64::from(smooth_dx_q16)) <= zero_intent
                && abs_u64(i64::from(smooth_dy_q16)) <= zero_intent;
            let low_debt =
                abs_u64(accumulator_q16[0]) < zero_debt && abs_u64(accumulator_q16[1]) < zero_debt;
            if quiet && low_debt {
                emit = false;
            }
        }

        let target = u64::from(2 * last_offset_top23 + 1) * total;
        let mut cumulative = 0u64;
        let mut joint = GRID - 1;
        for row in 0..GRID {
            cumulative += u64::from(self.logits[row] as u32);
            if target <= cumulative * 2 * (1u64 << 23) {
                joint = row;
                break;
            }
        }

        let mut mark = [
            round_div_even_i64(accumulator_q16[0], 1u64 << SMOOTH_Q) as i32,
            round_div_even_i64(accumulator_q16[1], 1u64 << SMOOTH_Q) as i32,
        ];
        for axis in 0..2 {
            if self.last_axis_nonzero[axis] != 0
                && mark[axis] != 0
                && sign_i32(i32::from(self.last_axis_nonzero[axis])) != sign_i32(mark[axis])
                && abs_u64(accumulator_q16[axis])
                    < u64::from(model.config[CFG_HYSTERESIS_Q16] as u32)
            {
                mark[axis] = 0;
            }
        }
        mark[0] += (joint / SIDE) as i32 - RADIUS;
        mark[1] += (joint % SIDE) as i32 - RADIUS;
        let limit = model.config[CFG_MAX_ABS_COUNT];
        mark[0] = mark[0].clamp(-limit, limit);
        mark[1] = mark[1].clamp(-limit, limit);
        if !emit {
            mark = [0, 0];
        }
        let next = [
            accumulator_q16[0] - i64::from(mark[0]) * (1i64 << SMOOTH_Q),
            accumulator_q16[1] - i64::from(mark[1]) * (1i64 << SMOOTH_Q),
        ];
        if next
            .iter()
            .any(|&value| value < I32_MIN_VALUE || value > i64::from(i32::MAX))
        {
            return Err(Error::Range(
                "renderer accumulator exceeded its range".into(),
            ));
        }
        self.accumulator_q16 = [next[0] as i32, next[1] as i32];
        self.update_history(mark[0] as i16, mark[1] as i16);
        self.last_emit_probability_q15 = emit_probability as u32;
        self.last_joint_index = joint as u16;
        self.generation_tick += 1;
        Ok([mark[0] as i16, mark[1] as i16])
    }
}
