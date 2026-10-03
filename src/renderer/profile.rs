use super::fixed::{Boundary, FixedRenderer};
use super::model::*;
use super::w5::W5Stream;
use crate::error::{Error, Result};
use crate::math::tanh_f32;

const CORE_FEATURES: usize = 15;
const DELTA_ENERGY_SCALE: f32 = 0.695_087_73;

fn q8_from_double(value: f64) -> i16 {
    let rounded = (f64::from(value as f32) * 256.0).round_ties_even();
    if rounded < -32768.0 {
        return i16::MIN;
    }
    if rounded > 32767.0 {
        return i16::MAX;
    }
    rounded as i16
}

fn q16_from_double(value: f64) -> i32 {
    let rounded = (f64::from(value as f32) * 65536.0).round_ties_even();
    if rounded < -2_147_483_648.0 {
        return i32::MIN;
    }
    if rounded > 2_147_483_647.0 {
        return i32::MAX;
    }
    rounded as i32
}

fn q15_from_float(value: f32) -> i16 {
    let clipped = f64::from(value).clamp(-1.0, 1.0);
    (clipped * 32768.0).round_ties_even().clamp(-32767.0, 32767.0) as i16
}

pub fn q16_pair(displacement: [f32; 2]) -> Result<[i32; 2]> {
    let mut out = [0i32; 2];
    for axis in 0..2 {
        let scaled = (f64::from(displacement[axis]) * 65536.0).round_ties_even();
        if !(-2_147_483_648.0..2_147_483_648.0).contains(&scaled) {
            return Err(Error::Range(
                "smooth intent tick exceeds signed Q16 range".into(),
            ));
        }
        out[axis] = scaled as i32;
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct CoreState {
    cumulative_smooth: [f64; 2],
    cumulative_previous_raw: [f64; 2],
    previous_raw: [f64; 2],
    last_nonzero: [f64; 2],
    last_axis_nonzero: [i16; 2],
    last_direction: [f64; 2],
    previous_smooth: [f64; 2],
    previous_speed: f64,
    run_length: u16,
    run_active: u8,
    active_ring: [u8; RECENT_WINDOW],
    ring_pos: u8,
    ring_count: u8,
}

impl Default for CoreState {
    fn default() -> Self {
        Self {
            cumulative_smooth: [0.0; 2],
            cumulative_previous_raw: [0.0; 2],
            previous_raw: [0.0; 2],
            last_nonzero: [0.0; 2],
            last_axis_nonzero: [0; 2],
            last_direction: [1.0, 0.0],
            previous_smooth: [0.0; 2],
            previous_speed: 0.0,
            run_length: 0,
            run_active: 0,
            active_ring: [0; RECENT_WINDOW],
            ring_pos: 0,
            ring_count: 0,
        }
    }
}

impl CoreState {
    fn build(
        &mut self,
        logical_tick: u16,
        raw_x: i16,
        raw_y: i16,
        smooth: [f64; 2],
        feature: &mut [i16; CORE_FEATURES],
    ) {
        let speed = (smooth[0] * smooth[0] + smooth[1] * smooth[1]).sqrt();
        let accel = if logical_tick == 0 { 0.0 } else { speed - self.previous_speed };
        let mut curvature = 0.0;
        let active = u8::from(raw_x != 0 || raw_y != 0);
        if logical_tick > 0 && self.previous_speed > 1.0e-9 && speed > 1.0e-9 {
            let cosine = (self.previous_smooth[0] * smooth[0] + self.previous_smooth[1] * smooth[1])
                / (self.previous_speed * speed);
            curvature = 1.0 - cosine.clamp(-1.0, 1.0);
        }
        if speed > 1.0e-9 {
            self.last_direction = smooth;
        }
        let mut norm = (self.last_direction[0] * self.last_direction[0]
            + self.last_direction[1] * self.last_direction[1])
            .sqrt();
        if norm <= 1.0e-9 {
            norm = 1.0;
        }
        let tx = self.last_direction[0] / norm;
        let ty = self.last_direction[1] / norm;
        let nx = -ty;
        let ny = tx;
        self.cumulative_smooth[0] += smooth[0];
        self.cumulative_smooth[1] += smooth[1];
        if logical_tick > 0 {
            self.cumulative_previous_raw[0] += self.previous_raw[0];
            self.cumulative_previous_raw[1] += self.previous_raw[1];
        }
        let desired_x = self.cumulative_smooth[0] - self.cumulative_previous_raw[0];
        let desired_y = self.cumulative_smooth[1] - self.cumulative_previous_raw[1];
        let acc_t = desired_x * tx + desired_y * ty;
        let acc_n = desired_x * nx + desired_y * ny;
        let prev_t = self.previous_raw[0] * tx + self.previous_raw[1] * ty;
        let prev_n = self.previous_raw[0] * nx + self.previous_raw[1] * ny;
        let last_t = self.last_nonzero[0] * tx + self.last_nonzero[1] * ty;
        let last_n = self.last_nonzero[0] * nx + self.last_nonzero[1] * ny;
        let mut inactive = 0u32;
        for index in 0..usize::from(self.ring_count) {
            inactive += u32::from(self.active_ring[index] == 0);
        }
        let recent_zero = if self.ring_count == 0 {
            1.0
        } else {
            f64::from(inactive) / f64::from(self.ring_count)
        };
        feature[0] = q8_from_double(speed / 4.0);
        feature[1] = q8_from_double(accel.clamp(-3.0, 3.0));
        feature[2] = q8_from_double(curvature);
        feature[3] = q8_from_double(tx);
        feature[4] = q8_from_double(ty);
        feature[5] = q8_from_double(acc_t.clamp(-4.0, 4.0));
        feature[6] = q8_from_double(acc_n.clamp(-4.0, 4.0));
        feature[7] = q8_from_double((prev_t / 4.0).clamp(-4.0, 4.0));
        feature[8] = q8_from_double((prev_n / 4.0).clamp(-4.0, 4.0));
        feature[9] = i16::from(self.run_active) << 8;
        feature[10] = q8_from_double(f64::from(self.run_length.min(64)) / 64.0);
        feature[11] = q8_from_double(recent_zero);
        feature[12] = q8_from_double(speed.ln_1p());
        feature[13] = q8_from_double((last_t / 4.0).clamp(-4.0, 4.0));
        feature[14] = q8_from_double((last_n / 4.0).clamp(-4.0, 4.0));
        self.run_length = if self.run_length == 0 || active != self.run_active {
            1
        } else {
            self.run_length + 1
        };
        self.run_active = active;
        self.active_ring[usize::from(self.ring_pos)] = active;
        self.ring_pos = (self.ring_pos + 1) % RECENT_WINDOW as u8;
        if usize::from(self.ring_count) < RECENT_WINDOW {
            self.ring_count += 1;
        }
        self.previous_raw = [f64::from(raw_x), f64::from(raw_y)];
        if active != 0 {
            self.last_nonzero = [f64::from(raw_x), f64::from(raw_y)];
        }
        if raw_x != 0 {
            self.last_axis_nonzero[0] = raw_x;
        }
        if raw_y != 0 {
            self.last_axis_nonzero[1] = raw_y;
        }
        self.previous_smooth = smooth;
        self.previous_speed = speed;
    }
}

/// Frozen float32 dot topology: two four-lane FMA accumulators, pairwise lane
/// reduction, then scalar remainder.
fn adapter_dot(quantized: &[i8], scale: f32, values: &[f32]) -> f32 {
    let count = values.len();
    let mut lane0 = [0.0f32; 4];
    let mut lane1 = [0.0f32; 4];
    let mut column = 0usize;
    while column + 8 <= count {
        for lane in 0..4 {
            let weight0 = f32::from(quantized[column + lane]) * scale;
            let weight1 = f32::from(quantized[column + 4 + lane]) * scale;
            lane0[lane] = weight0.mul_add(values[column + lane], lane0[lane]);
            lane1[lane] = weight1.mul_add(values[column + 4 + lane], lane1[lane]);
        }
        column += 8;
    }
    let merged: [f32; 4] = std::array::from_fn(|lane| lane0[lane] + lane1[lane]);
    let mut sum = (merged[0] + merged[1]) + (merged[2] + merged[3]);
    while column < count {
        let weight = f32::from(quantized[column]) * scale;
        sum = weight.mul_add(values[column], sum);
        column += 1;
    }
    sum
}

fn adapter_predict(adapter: &Adapter, input: &[f32; ADAPTER_INPUT]) -> [f32; HIDDEN] {
    let mut normalized = [0.0f32; ADAPTER_INPUT];
    for column in 0..ADAPTER_INPUT {
        normalized[column] = (input[column] - adapter.mean[column]) * adapter.invstd[column];
    }
    let mut rank = [0.0f32; ADAPTER_RANK];
    for row in 0..ADAPTER_RANK {
        let sum = adapter_dot(
            &adapter.v_q[row * ADAPTER_INPUT..(row + 1) * ADAPTER_INPUT],
            adapter.v_scale[row],
            &normalized,
        ) + adapter.v_bias[row];
        rank[row] = tanh_f32(sum);
    }
    let mut output = [0.0f32; HIDDEN];
    for row in 0..HIDDEN {
        let mut sum = adapter_dot(
            &adapter.u_q[row * ADAPTER_RANK..(row + 1) * ADAPTER_RANK],
            adapter.u_scale[row],
            &rank,
        );
        sum = input[row] + sum;
        sum += adapter.u_bias[row];
        output[row] = sum;
    }
    output
}

struct Observer {
    fixed: FixedRenderer,
    smoother: W5Stream,
    target_smoother: W5Stream,
    observed: u16,
    observer_core: CoreState,
    target_core: CoreState,
    canonical_active_tail: [u8; RECENT_WINDOW],
    active_count: u32,
    active_sum: f64,
    running_max: u16,
    previous_magnitude: u16,
    sign_flip_count: u32,
    sign_comparison_count: u32,
    previous_sign: [i8; 2],
    delta_energy: f64,
    sorted_active_magnitudes: Vec<u16>,
    magnitude_sum_squares: f64,
    low_dft_real: [f64; 7],
    low_dft_imag: [f64; 7],
    low_dft_twiddle_real: [f64; 7],
    low_dft_twiddle_imag: [f64; 7],
    low_dft_oscillator_real: [f64; 7],
    low_dft_oscillator_imag: [f64; 7],
    nyquist: f64,
    final_regime_q8: [i16; 5],
    target_tail4_core_q8: [[i16; CORE_FEATURES]; 4],
}

impl Observer {
    fn new() -> Self {
        // Six half-angle steps derive exp(-i*pi/128) from cos(pi/2)=0.
        let mut base_real = 0.0f64;
        let mut base_imag = 0.0f64;
        for _ in 0..6 {
            base_imag = -((1.0 - base_real) * 0.5).sqrt();
            base_real = ((1.0 + base_real) * 0.5).sqrt();
        }
        let mut twiddle_real = [0.0f64; 7];
        let mut twiddle_imag = [0.0f64; 7];
        for index in 0..7 {
            if index == 0 {
                twiddle_real[index] = base_real;
                twiddle_imag[index] = base_imag;
            } else {
                let previous_real = twiddle_real[index - 1];
                let previous_imag = twiddle_imag[index - 1];
                twiddle_real[index] = previous_real * base_real - previous_imag * base_imag;
                twiddle_imag[index] = previous_real * base_imag + previous_imag * base_real;
            }
        }
        Self {
            fixed: FixedRenderer::new(),
            smoother: W5Stream::new(),
            target_smoother: W5Stream::new(),
            observed: 0,
            observer_core: CoreState::default(),
            target_core: CoreState::default(),
            canonical_active_tail: [0; RECENT_WINDOW],
            active_count: 0,
            active_sum: 0.0,
            running_max: 0,
            previous_magnitude: 0,
            sign_flip_count: 0,
            sign_comparison_count: 0,
            previous_sign: [0; 2],
            delta_energy: 0.0,
            sorted_active_magnitudes: vec![0; PREFIX_WINDOW],
            magnitude_sum_squares: 0.0,
            low_dft_real: [0.0; 7],
            low_dft_imag: [0.0; 7],
            low_dft_twiddle_real: twiddle_real,
            low_dft_twiddle_imag: twiddle_imag,
            low_dft_oscillator_real: [1.0; 7],
            low_dft_oscillator_imag: [0.0; 7],
            nyquist: 0.0,
            final_regime_q8: [0; 5],
            target_tail4_core_q8: [[0; CORE_FEATURES]; 4],
        }
    }

    fn update_summaries(&mut self, dx: i16, dy: i16) -> [i16; 5] {
        let magnitude = i32::from(dx).abs().max(i32::from(dy).abs()) as u16;
        let tick = u32::from(self.observed);
        if magnitude > 0 {
            let mut index = self.active_count as usize;
            while index > 0 && self.sorted_active_magnitudes[index - 1] > magnitude {
                self.sorted_active_magnitudes[index] = self.sorted_active_magnitudes[index - 1];
                index -= 1;
            }
            self.sorted_active_magnitudes[index] = magnitude;
            self.active_count += 1;
            self.active_sum += f64::from(magnitude);
            self.magnitude_sum_squares += f64::from(magnitude) * f64::from(magnitude);
            if magnitude > self.running_max {
                self.running_max = magnitude;
            }
        }
        for axis in 0..2 {
            let value = if axis == 0 { dx } else { dy };
            let sign = (i8::from(value > 0)) - (i8::from(value < 0));
            if sign != 0 {
                if self.previous_sign[axis] != 0 {
                    self.sign_comparison_count += 1;
                    if sign != self.previous_sign[axis] {
                        self.sign_flip_count += 1;
                    }
                }
                self.previous_sign[axis] = sign;
            }
        }
        if tick > 0 {
            let delta = f64::from(magnitude) - f64::from(self.previous_magnitude);
            self.delta_energy += delta * delta;
        }
        self.previous_magnitude = magnitude;
        for axis in 0..7 {
            let real = self.low_dft_oscillator_real[axis];
            let imag = self.low_dft_oscillator_imag[axis];
            self.low_dft_real[axis] += f64::from(magnitude) * real;
            self.low_dft_imag[axis] += f64::from(magnitude) * imag;
            self.low_dft_oscillator_real[axis] =
                real * self.low_dft_twiddle_real[axis] - imag * self.low_dft_twiddle_imag[axis];
            self.low_dft_oscillator_imag[axis] =
                real * self.low_dft_twiddle_imag[axis] + imag * self.low_dft_twiddle_real[axis];
        }
        self.nyquist += if tick & 1 == 1 {
            -f64::from(magnitude)
        } else {
            f64::from(magnitude)
        };

        [
            q8_from_double(f64::from(self.active_count) / f64::from(tick + 1)),
            q8_from_double(
                if self.active_count > 0 {
                    self.active_sum / f64::from(self.active_count)
                } else {
                    0.0
                }
                .ln_1p(),
            ),
            q8_from_double(f64::from(self.running_max).ln_1p()),
            q8_from_double(if self.sign_comparison_count > 0 {
                f64::from(self.sign_flip_count) / f64::from(self.sign_comparison_count)
            } else {
                0.0
            }),
            q8_from_double(f64::from(DELTA_ENERGY_SCALE) * self.delta_energy.ln_1p()),
        ]
    }

    fn final_regime(&mut self) {
        let mean = if self.active_count > 0 {
            self.active_sum / f64::from(self.active_count)
        } else {
            0.0
        };
        let mut p95 = 0.0f64;
        if self.active_count == 1 {
            p95 = f64::from(self.sorted_active_magnitudes[0]);
        } else if self.active_count > 1 {
            let position = 0.95 * f64::from(self.active_count - 1);
            let index = position as usize;
            p95 = f64::from(self.sorted_active_magnitudes[index])
                + (f64::from(self.sorted_active_magnitudes[index + 1])
                    - f64::from(self.sorted_active_magnitudes[index]))
                    * (position - index as f64);
        }
        let total_centered =
            256.0 * (self.magnitude_sum_squares - self.active_sum * self.active_sum / 256.0);
        let mut low = 0.0f64;
        for index in 0..7 {
            low += self.low_dft_real[index] * self.low_dft_real[index]
                + self.low_dft_imag[index] * self.low_dft_imag[index];
        }
        let target = (total_centered - self.nyquist * self.nyquist) * 0.5 - low;
        let high_frequency = if target > 0.0 { target / 256.0 } else { 0.0 };
        self.final_regime_q8 = [
            q8_from_double(f64::from(self.active_count) / 256.0),
            q8_from_double(mean.ln_1p()),
            q8_from_double(p95.ln_1p()),
            q8_from_double(if self.sign_comparison_count > 0 {
                f64::from(self.sign_flip_count) / f64::from(self.sign_comparison_count)
            } else {
                0.0
            }),
            q8_from_double(high_frequency.ln_1p()),
        ];
    }

    fn observe(&mut self, model: &RendererModel, dx: i16, dy: i16) -> Result<()> {
        if usize::from(self.observed) >= PREFIX_WINDOW {
            return Err(Error::Mode("renderer profile is already complete".into()));
        }
        let tick = self.observed;
        self.fixed.observe_prefix(dx, dy)?;
        let summary = self.update_summaries(dx, dy);
        let mut feature = [0i16; FEATURES];
        if let Some(smooth) = self.smoother.push_delta(f64::from(dx), f64::from(dy)) {
            let logical = tick - 4;
            let raw = self.fixed.prefix_raw[usize::from(logical)];
            let mut core = [0i16; CORE_FEATURES];
            self.observer_core.build(logical, raw[0], raw[1], smooth, &mut core);
            feature[..CORE_FEATURES].copy_from_slice(&core);
        }
        if tick >= 128 {
            if let Some(smooth) = self.target_smoother.push_delta(f64::from(dx), f64::from(dy)) {
                let logical = tick - 128 - 4;
                let raw = self.fixed.prefix_raw[usize::from(128 + logical)];
                let mut discarded = [0i16; CORE_FEATURES];
                self.target_core.build(logical, raw[0], raw[1], smooth, &mut discarded);
            }
            if tick >= 196 {
                self.canonical_active_tail[usize::from(tick - 196)] =
                    u8::from(dx != 0 || dy != 0);
            }
        }
        feature[CORE_FEATURES..].copy_from_slice(&summary);
        self.fixed.online_gru_step_q8(&model.fixed, &feature)?;
        if usize::from(tick) == PREFIX_WINDOW - 1 {
            self.finalize(model)?;
        }
        self.observed += 1;
        Ok(())
    }

    fn finalize(&mut self, model: &RendererModel) -> Result<()> {
        self.final_regime();
        let tail = self.target_smoother.flush();
        if tail.len() != 4 {
            return Err(Error::Mode("renderer profile tail is incomplete".into()));
        }
        for index in 0..4 {
            let logical = 124 + index as u16;
            let raw = self.fixed.prefix_raw[usize::from(128 + logical)];
            let mut core = [0i16; CORE_FEATURES];
            self.target_core.build(logical, raw[0], raw[1], tail[index], &mut core);
            self.target_tail4_core_q8[index] = core;
        }
        let mut boundary = Boundary {
            regime_q8: self.final_regime_q8,
            ..Boundary::default()
        };
        for axis in 0..2 {
            boundary.previous_emit[axis] = self.target_core.previous_raw[axis] as i16;
            boundary.last_nonzero[axis] = self.target_core.last_nonzero[axis] as i16;
            boundary.last_axis_nonzero[axis] = self.target_core.last_axis_nonzero[axis];
            boundary.last_smooth_q16[axis] =
                q16_from_double(self.target_core.previous_smooth[axis]);
        }
        boundary.run_length = self.target_core.run_length;
        boundary.run_active = self.target_core.run_active;
        boundary.active_ring = self.canonical_active_tail;
        boundary.active_ring_count = RECENT_WINDOW as u8;
        boundary.active_ring_pos = 0;
        self.fixed.install_boundary(&boundary)?;

        let mut input = [0.0f32; ADAPTER_INPUT];
        for index in 0..HIDDEN {
            input[index] = f32::from(self.fixed.hidden[index]) / 32768.0;
        }
        for index in 0..60 {
            input[HIDDEN + index] =
                f32::from(self.target_tail4_core_q8[index / 15][index % 15]) / 256.0;
        }
        for index in 0..5 {
            input[140 + index] = f32::from(self.final_regime_q8[index]) / 256.0;
        }
        let output = adapter_predict(&model.adapter, &input);
        for index in 0..HIDDEN {
            self.fixed.hidden[index] = q15_from_float(output[index]);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct RendererProfile {
    template: FixedRenderer,
}

#[derive(Clone)]
pub struct RendererStream {
    state: FixedRenderer,
    ticks: u64,
    failed: bool,
}

impl RendererProfile {
    /// Prepare one immutable profile from exactly 256 chronological integer reports.
    pub fn prepare(model: &RendererModel, reports: &[[i16; 2]]) -> Result<Self> {
        if reports.len() != PREFIX_WINDOW {
            return Err(Error::InferenceContract(format!(
                "renderer context must contain exactly {PREFIX_WINDOW} integer reports"
            )));
        }
        let mut observer = Observer::new();
        for report in reports {
            observer.observe(model, report[0], report[1])?;
        }
        Ok(Self {
            template: observer.fixed,
        })
    }

    pub fn begin_stream(&self, event_seed: u64) -> Result<RendererStream> {
        let mut state = self.template.clone();
        state.begin(event_seed)?;
        Ok(RendererStream {
            state,
            ticks: 0,
            failed: false,
        })
    }

    pub fn hidden(&self) -> &[i16] {
        &self.template.hidden
    }

    pub fn boundary_state(&self) -> &FixedRenderer {
        &self.template
    }
}

impl RendererStream {
    pub fn step(&mut self, model: &RendererModel, displacement: [f32; 2]) -> Result<[i16; 2]> {
        if self.failed {
            return Err(Error::Mode(
                "Renderer stream failed; begin a new stream".into(),
            ));
        }
        let pair = q16_pair(displacement)?;
        match self.state.step_q16(&model.fixed, pair[0], pair[1]) {
            Ok(report) => {
                self.ticks += 1;
                Ok(report)
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    pub fn step_q16(&mut self, model: &RendererModel, x: i32, y: i32) -> Result<[i16; 2]> {
        if self.failed {
            return Err(Error::Mode(
                "Renderer stream failed; begin a new stream".into(),
            ));
        }
        match self.state.step_q16(&model.fixed, x, y) {
            Ok(report) => {
                self.ticks += 1;
                Ok(report)
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    pub fn accumulator_q16(&self) -> [i32; 2] {
        self.state.accumulator_q16
    }
}
