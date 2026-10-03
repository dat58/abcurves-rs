use super::activation::{silu_exact, silu_pade9};
use super::linear::{matmul, matvec};
use crate::continuous::constants::*;
use crate::error::{Error, Result};
use crate::io::Npz;
use crate::math::exp_f32;

const TOKENS: usize = COARSE_TOKENS;
const GATES: usize = 3 * HIDDEN;
const COMBINED: usize = HIDDEN + FINE_LEN + DYNAMICS_LEN;
const COEFFICIENTS: usize = HEADS * WEIGHTS * 2;
const BRAKE_PARAMETERS: usize = HEADS * 11;

const BRAKE_LOG_MINIMUM: f32 = 1.386_294_4;
const BRAKE_LOG_MAXIMUM: f32 = 5.257_495_4;
const BRAKE_SCALE: f32 = 64.0;

fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = values[row * columns + column];
        }
    }
    out
}

fn take(bundle: &Npz, name: &str) -> Result<Vec<f32>> {
    let array = bundle.array(name)?;
    if array.dtype != crate::io::DType::F32 {
        return Err(Error::ModelIntegrity(format!(
            "Learned weights must remain float32: {name}"
        )));
    }
    array.to_f32()
}

pub struct MotorWeights {
    input_weight: Vec<f32>,
    input_bias: Vec<f32>,
    input_gates: Vec<f32>,
    input_gate_bias: Vec<f32>,
    recurrent_transpose: Vec<f32>,
    recurrent_bias: Vec<f32>,
    trunk_weight: Vec<f32>,
    trunk_bias: Vec<f32>,
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
}

pub struct HeadWeights {
    unary_latent: Vec<f32>,
    unary_geometry: Vec<f32>,
    unary_bias: Vec<f32>,
    unary_output: Vec<f32>,
    unary_output_bias: f32,
    context_weight: Vec<f32>,
    context_bias: Vec<f32>,
    pair_context: Vec<f32>,
    pair_geometry: Vec<f32>,
    pair_bias: Vec<f32>,
    pair_output: Vec<f32>,
    pair_output_bias: f32,
    hazard: [(Vec<f32>, Vec<f32>); 3],
    brake_encoder: [(Vec<f32>, Vec<f32>); 2],
    brake_weight: Vec<f32>,
    brake_bias: Vec<f32>,
    frequency_weight: Vec<f32>,
    frequency_bias: Vec<f32>,
}

struct MotorWorkspace {
    tokens: Vec<f32>,
    projected: Vec<f32>,
    hidden: Vec<f32>,
    recurrent: Vec<f32>,
    combined: Vec<f32>,
}

struct HeadWorkspace {
    shared: Vec<f32>,
    unary: Vec<f32>,
    context: Vec<f32>,
    shared_pair: Vec<f32>,
    pair: Vec<f32>,
    first: Vec<f32>,
    second: Vec<f32>,
    stage: Vec<f32>,
    encoded: Vec<f32>,
    parameters: Vec<f32>,
}

pub struct EventOutputs {
    pub duration: [f32; HEADS],
    pub coefficients: [f32; HEADS * 10],
    pub frequency: [f32; HEADS],
    pub hazard: [f32; 2],
}

impl Default for EventOutputs {
    fn default() -> Self {
        Self {
            duration: [0.0; HEADS],
            coefficients: [0.0; HEADS * 10],
            frequency: [0.0; HEADS],
            hazard: [0.0; 2],
        }
    }
}

pub struct NativeEngine {
    motor: MotorWeights,
    heads: HeadWeights,
    motor_workspace: MotorWorkspace,
    head_workspace: HeadWorkspace,
    pub encoded: Vec<f32>,
    pub coefficients: Vec<f32>,
    pub logits: [f32; HEADS],
}

impl NativeEngine {
    pub fn load(bundle: &Npz) -> Result<Self> {
        let motor = MotorWeights {
            input_weight: transpose(&take(bundle, "motor.input.0.weight")?, HIDDEN, 9),
            input_bias: take(bundle, "motor.input.0.bias")?,
            input_gates: transpose(&take(bundle, "motor.encoder.weight_ih_l0")?, GATES, HIDDEN),
            input_gate_bias: take(bundle, "motor.encoder.bias_ih_l0")?,
            recurrent_transpose: transpose(
                &take(bundle, "motor.encoder.weight_hh_l0")?,
                GATES,
                HIDDEN,
            ),
            recurrent_bias: take(bundle, "motor.encoder.bias_hh_l0")?,
            trunk_weight: take(bundle, "motor.trunk.0.weight")?,
            trunk_bias: take(bundle, "motor.trunk.0.bias")?,
            output_weight: take(bundle, "motor.output.weight")?,
            output_bias: take(bundle, "motor.output.bias")?,
        };

        let unary = take(bundle, "motor.selection.unary.0.weight")?;
        let pair = take(bundle, "motor.selection.pair.0.weight")?;
        let mut unary_latent = vec![0.0f32; 64 * HIDDEN];
        let mut unary_geometry_rows = vec![0.0f32; 64 * 16];
        for row in 0..64 {
            unary_latent[row * HIDDEN..(row + 1) * HIDDEN]
                .copy_from_slice(&unary[row * 112..row * 112 + HIDDEN]);
            unary_geometry_rows[row * 16..(row + 1) * 16]
                .copy_from_slice(&unary[row * 112 + HIDDEN..(row + 1) * 112]);
        }
        let mut pair_context = vec![0.0f32; 32 * 16];
        let mut pair_geometry_rows = vec![0.0f32; 32 * 20];
        for row in 0..32 {
            pair_geometry_rows[row * 20..(row + 1) * 20]
                .copy_from_slice(&pair[row * 36..row * 36 + 20]);
            pair_context[row * 16..(row + 1) * 16]
                .copy_from_slice(&pair[row * 36 + 20..(row + 1) * 36]);
        }

        let heads = HeadWeights {
            unary_latent,
            unary_geometry: transpose(&unary_geometry_rows, 64, 16),
            unary_bias: take(bundle, "motor.selection.unary.0.bias")?,
            unary_output: take(bundle, "motor.selection.unary.2.weight")?,
            unary_output_bias: take(bundle, "motor.selection.unary.2.bias")?[0],
            context_weight: take(bundle, "motor.selection.context.0.weight")?,
            context_bias: take(bundle, "motor.selection.context.0.bias")?,
            pair_context,
            pair_geometry: transpose(&pair_geometry_rows, 32, 20),
            pair_bias: take(bundle, "motor.selection.pair.0.bias")?,
            pair_output: take(bundle, "motor.selection.pair.2.weight")?,
            pair_output_bias: take(bundle, "motor.selection.pair.2.bias")?[0],
            hazard: [
                (
                    take(bundle, "events.hazard.0.weight")?,
                    take(bundle, "events.hazard.0.bias")?,
                ),
                (
                    take(bundle, "events.hazard.2.weight")?,
                    take(bundle, "events.hazard.2.bias")?,
                ),
                (
                    take(bundle, "events.hazard.4.weight")?,
                    take(bundle, "events.hazard.4.bias")?,
                ),
            ],
            brake_encoder: [
                (
                    take(bundle, "events.encoder.0.weight")?,
                    take(bundle, "events.encoder.0.bias")?,
                ),
                (
                    take(bundle, "events.encoder.2.weight")?,
                    take(bundle, "events.encoder.2.bias")?,
                ),
            ],
            brake_weight: take(bundle, "events.brake.weight")?,
            brake_bias: take(bundle, "events.brake.bias")?,
            frequency_weight: take(bundle, "events.frequency.weight")?,
            frequency_bias: take(bundle, "events.frequency.bias")?,
        };

        Ok(Self {
            motor,
            heads,
            motor_workspace: MotorWorkspace {
                tokens: vec![0.0; TOKENS * HIDDEN],
                projected: vec![0.0; TOKENS * GATES],
                hidden: vec![0.0; HIDDEN],
                recurrent: vec![0.0; GATES],
                combined: vec![0.0; COMBINED],
            },
            head_workspace: HeadWorkspace {
                shared: vec![0.0; 64],
                unary: vec![0.0; HEADS * 64],
                context: vec![0.0; 16],
                shared_pair: vec![0.0; 32],
                pair: vec![0.0; HEADS * 32],
                first: vec![0.0; 64],
                second: vec![0.0; 32],
                stage: vec![0.0; HIDDEN],
                encoded: vec![0.0; HIDDEN],
                parameters: vec![0.0; BRAKE_PARAMETERS],
            },
            encoded: vec![0.0; HIDDEN],
            coefficients: vec![0.0; COEFFICIENTS],
            logits: [0.0; HEADS],
        })
    }

    pub fn motor(&mut self, coarse: &[f32], fine: &[f32], dynamics: &[f32]) {
        let weights = &self.motor;
        let space = &mut self.motor_workspace;

        matmul(coarse, TOKENS, 9, &weights.input_weight, HIDDEN, &mut space.tokens);
        for row in 0..TOKENS {
            for column in 0..HIDDEN {
                let value = space.tokens[row * HIDDEN + column] + weights.input_bias[column];
                space.tokens[row * HIDDEN + column] = silu_pade9(value);
            }
        }
        matmul(
            &space.tokens,
            TOKENS,
            HIDDEN,
            &weights.input_gates,
            GATES,
            &mut space.projected,
        );
        for row in 0..TOKENS {
            for column in 0..GATES {
                space.projected[row * GATES + column] += weights.input_gate_bias[column];
            }
        }
        space.hidden.fill(0.0);
        for row in 0..TOKENS {
            space.recurrent.copy_from_slice(&weights.recurrent_bias);
            for incoming in 0..HIDDEN {
                let value = space.hidden[incoming];
                let lane = &weights.recurrent_transpose[incoming * GATES..(incoming + 1) * GATES];
                for column in 0..GATES {
                    space.recurrent[column] += value * lane[column];
                }
            }
            let projected = &space.projected[row * GATES..(row + 1) * GATES];
            for column in 0..HIDDEN {
                let reset = super::activation::sigmoid_pade9(
                    projected[column] + space.recurrent[column],
                );
                let update = super::activation::sigmoid_pade9(
                    projected[HIDDEN + column] + space.recurrent[HIDDEN + column],
                );
                let candidate = super::activation::tanh_pade9(
                    projected[2 * HIDDEN + column] + reset * space.recurrent[2 * HIDDEN + column],
                );
                space.hidden[column] =
                    candidate + (space.hidden[column] - candidate) * update;
            }
        }
        space.combined[..HIDDEN].copy_from_slice(&space.hidden);
        space.combined[HIDDEN..HIDDEN + FINE_LEN].copy_from_slice(fine);
        space.combined[HIDDEN + FINE_LEN..].copy_from_slice(dynamics);
        matvec(
            &weights.trunk_weight,
            HIDDEN,
            COMBINED,
            &space.combined,
            &mut self.encoded,
        );
        for column in 0..HIDDEN {
            let value = self.encoded[column] + weights.trunk_bias[column];
            self.encoded[column] = silu_pade9(value);
        }
        matvec(
            &weights.output_weight,
            COEFFICIENTS,
            HIDDEN,
            &self.encoded,
            &mut self.coefficients,
        );
        for column in 0..COEFFICIENTS {
            self.coefficients[column] += weights.output_bias[column];
        }
    }

    pub fn choice(&mut self, geometry: &[f32], pairs: &[f32], previous_valid: bool) {
        let weights = &self.heads;
        let space = &mut self.head_workspace;

        matvec(&weights.unary_latent, 64, HIDDEN, &self.encoded, &mut space.shared);
        for column in 0..64 {
            space.shared[column] += weights.unary_bias[column];
        }
        matmul(geometry, HEADS, 16, &weights.unary_geometry, 64, &mut space.unary);
        for row in 0..HEADS {
            for column in 0..64 {
                let value = space.shared[column] + space.unary[row * 64 + column];
                space.unary[row * 64 + column] = silu_exact(value);
            }
        }
        for row in 0..HEADS {
            let lane = &space.unary[row * 64..(row + 1) * 64];
            let mut total = 0.0f32;
            for column in 0..64 {
                total += lane[column] * weights.unary_output[column];
            }
            self.logits[row] = total + weights.unary_output_bias;
        }
        if previous_valid {
            matvec(
                &weights.context_weight,
                16,
                HIDDEN,
                &self.encoded,
                &mut space.context,
            );
            for column in 0..16 {
                let value = space.context[column] + weights.context_bias[column];
                space.context[column] = silu_exact(value);
            }
            matvec(
                &weights.pair_context,
                32,
                16,
                &space.context,
                &mut space.shared_pair,
            );
            for column in 0..32 {
                space.shared_pair[column] += weights.pair_bias[column];
            }
            matmul(pairs, HEADS, PAIR_LEN, &weights.pair_geometry, 32, &mut space.pair);
            for row in 0..HEADS {
                for column in 0..32 {
                    let value = space.shared_pair[column] + space.pair[row * 32 + column];
                    space.pair[row * 32 + column] = silu_exact(value);
                }
            }
            for row in 0..HEADS {
                let lane = &space.pair[row * 32..(row + 1) * 32];
                let mut total = 0.0f32;
                for column in 0..32 {
                    total += lane[column] * weights.pair_output[column];
                }
                self.logits[row] += total + weights.pair_output_bias;
            }
        }
    }

    pub fn hazard(&mut self, context: &[f32], out: &mut [f32; 2]) {
        let weights = &self.heads;
        let space = &mut self.head_workspace;
        matvec(&weights.hazard[0].0, 64, EVENT_CONTEXT_LEN, context, &mut space.first);
        for column in 0..64 {
            space.first[column] = silu_exact(space.first[column] + weights.hazard[0].1[column]);
        }
        matvec(&weights.hazard[1].0, 32, 64, &space.first, &mut space.second);
        for column in 0..32 {
            space.second[column] = silu_exact(space.second[column] + weights.hazard[1].1[column]);
        }
        matvec(&weights.hazard[2].0, 2, 32, &space.second, out);
        for column in 0..2 {
            out[column] += weights.hazard[2].1[column];
        }
    }

    pub fn brake(&mut self, context: &[f32], out: &mut EventOutputs) {
        let weights = &self.heads;
        let space = &mut self.head_workspace;
        matvec(
            &weights.brake_encoder[0].0,
            HIDDEN,
            EVENT_CONTEXT_LEN,
            context,
            &mut space.stage,
        );
        for column in 0..HIDDEN {
            space.stage[column] =
                silu_exact(space.stage[column] + weights.brake_encoder[0].1[column]);
        }
        matvec(
            &weights.brake_encoder[1].0,
            HIDDEN,
            HIDDEN,
            &space.stage,
            &mut space.encoded,
        );
        for column in 0..HIDDEN {
            space.encoded[column] =
                silu_exact(space.encoded[column] + weights.brake_encoder[1].1[column]);
        }
        matvec(
            &weights.brake_weight,
            BRAKE_PARAMETERS,
            HIDDEN,
            &space.encoded,
            &mut space.parameters,
        );
        for head in 0..HEADS {
            // The float32 clamp/exp is exactly the learned brake's output
            // transform. It is never replaced by an activation approximation.
            let log_duration = (space.parameters[11 * head] + weights.brake_bias[11 * head])
                .clamp(BRAKE_LOG_MINIMUM, BRAKE_LOG_MAXIMUM);
            out.duration[head] = exp_f32(log_duration);
            for column in 0..10 {
                out.coefficients[10 * head + column] = BRAKE_SCALE
                    * (space.parameters[11 * head + column + 1]
                        + weights.brake_bias[11 * head + column + 1]);
            }
        }
        matvec(
            &weights.frequency_weight,
            HEADS,
            HIDDEN,
            &space.encoded,
            &mut out.frequency,
        );
        for head in 0..HEADS {
            out.frequency[head] += weights.frequency_bias[head];
        }
    }

    pub fn events(&mut self, context: &[f32], out: &mut EventOutputs) {
        self.brake(context, out);
        let mut hazard = [0.0f32; 2];
        self.hazard(context, &mut hazard);
        out.hazard = hazard;
    }
}

pub fn hazard_probabilities(logits: &[f32; 2]) -> [f32; 2] {
    let convert = |value: f32| 1.0 / (1.0 + exp_f32(-value.clamp(-40.0, 40.0)));
    [convert(logits[0]), convert(logits[1])]
}
