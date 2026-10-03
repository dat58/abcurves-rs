use super::native::EventOutputs;
use crate::continuous::constants::*;
use crate::error::{Error, Result};
use crate::io::Npz;
use candle_core::{DType, Device, Tensor};

const TOKENS: usize = COARSE_TOKENS;
const GATES: usize = 3 * HIDDEN;
const COMBINED: usize = HIDDEN + FINE_LEN + DYNAMICS_LEN;
const COEFFICIENTS: usize = HEADS * WEIGHTS * 2;

const BRAKE_LOG_MINIMUM: f32 = 1.386_294_4;
const BRAKE_LOG_MAXIMUM: f32 = 5.257_495_4;
const BRAKE_SCALE: f32 = 64.0;

fn wrap(error: candle_core::Error) -> Error {
    Error::Numerical(format!("candle: {error}"))
}

type Outcome<T> = std::result::Result<T, candle_core::Error>;

fn sigmoid(value: &Tensor) -> Outcome<Tensor> {
    ((value.neg()?.exp()? + 1.0)?).recip()
}

fn silu(value: &Tensor) -> Outcome<Tensor> {
    value.mul(&sigmoid(value)?)
}

fn affine(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Outcome<Tensor> {
    input.matmul(weight)?.broadcast_add(bias)
}

struct Layer {
    weight: Tensor,
    bias: Tensor,
}

impl Layer {
    fn load(bundle: &Npz, weight: &str, bias: &str, rows: usize, columns: usize) -> Result<Self> {
        let raw = bundle.f32(weight)?;
        let transposed = transpose(&raw, rows, columns);
        Ok(Self {
            weight: tensor(&transposed, &[columns, rows])?,
            bias: tensor(&bundle.f32(bias)?, &[rows])?,
        })
    }

    fn apply(&self, input: &Tensor) -> Outcome<Tensor> {
        affine(input, &self.weight, &self.bias)
    }
}

fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = values[row * columns + column];
        }
    }
    out
}

fn tensor(values: &[f32], shape: &[usize]) -> Result<Tensor> {
    Tensor::from_slice(values, shape, &Device::Cpu).map_err(wrap)
}

/// Reference backend: the same learned tensors evaluated with exact float32
/// transcendentals instead of the accepted Pade9 approximations.
pub struct CandleEngine {
    input: Layer,
    input_gates: Layer,
    recurrent: Layer,
    trunk: Layer,
    output: Layer,
    unary_latent: Tensor,
    unary_geometry: Tensor,
    unary_bias: Tensor,
    unary_output: Layer,
    context: Layer,
    pair_context: Tensor,
    pair_geometry: Tensor,
    pair_bias: Tensor,
    pair_output: Layer,
    hazard: [Layer; 3],
    brake_encoder: [Layer; 2],
    brake: Layer,
    frequency: Layer,
    pub encoded: Vec<f32>,
    pub coefficients: Vec<f32>,
    pub logits: [f32; HEADS],
}

impl CandleEngine {
    pub fn load(bundle: &Npz) -> Result<Self> {
        let unary = bundle.f32("motor.selection.unary.0.weight")?;
        let mut latent = vec![0.0f32; 64 * HIDDEN];
        let mut geometry = vec![0.0f32; 64 * 16];
        for row in 0..64 {
            latent[row * HIDDEN..(row + 1) * HIDDEN]
                .copy_from_slice(&unary[row * 112..row * 112 + HIDDEN]);
            geometry[row * 16..(row + 1) * 16]
                .copy_from_slice(&unary[row * 112 + HIDDEN..(row + 1) * 112]);
        }
        let pair = bundle.f32("motor.selection.pair.0.weight")?;
        let mut pair_context = vec![0.0f32; 32 * 16];
        let mut pair_geometry = vec![0.0f32; 32 * 20];
        for row in 0..32 {
            pair_geometry[row * 20..(row + 1) * 20].copy_from_slice(&pair[row * 36..row * 36 + 20]);
            pair_context[row * 16..(row + 1) * 16]
                .copy_from_slice(&pair[row * 36 + 20..(row + 1) * 36]);
        }

        Ok(Self {
            input: Layer::load(
                bundle,
                "motor.input.0.weight",
                "motor.input.0.bias",
                HIDDEN,
                9,
            )?,
            input_gates: Layer::load(
                bundle,
                "motor.encoder.weight_ih_l0",
                "motor.encoder.bias_ih_l0",
                GATES,
                HIDDEN,
            )?,
            recurrent: Layer::load(
                bundle,
                "motor.encoder.weight_hh_l0",
                "motor.encoder.bias_hh_l0",
                GATES,
                HIDDEN,
            )?,
            trunk: Layer::load(
                bundle,
                "motor.trunk.0.weight",
                "motor.trunk.0.bias",
                HIDDEN,
                COMBINED,
            )?,
            output: Layer::load(
                bundle,
                "motor.output.weight",
                "motor.output.bias",
                COEFFICIENTS,
                HIDDEN,
            )?,
            unary_latent: tensor(&transpose(&latent, 64, HIDDEN), &[HIDDEN, 64])?,
            unary_geometry: tensor(&transpose(&geometry, 64, 16), &[16, 64])?,
            unary_bias: tensor(&bundle.f32("motor.selection.unary.0.bias")?, &[64])?,
            unary_output: Layer::load(
                bundle,
                "motor.selection.unary.2.weight",
                "motor.selection.unary.2.bias",
                1,
                64,
            )?,
            context: Layer::load(
                bundle,
                "motor.selection.context.0.weight",
                "motor.selection.context.0.bias",
                16,
                HIDDEN,
            )?,
            pair_context: tensor(&transpose(&pair_context, 32, 16), &[16, 32])?,
            pair_geometry: tensor(&transpose(&pair_geometry, 32, 20), &[20, 32])?,
            pair_bias: tensor(&bundle.f32("motor.selection.pair.0.bias")?, &[32])?,
            pair_output: Layer::load(
                bundle,
                "motor.selection.pair.2.weight",
                "motor.selection.pair.2.bias",
                1,
                32,
            )?,
            hazard: [
                Layer::load(
                    bundle,
                    "events.hazard.0.weight",
                    "events.hazard.0.bias",
                    64,
                    EVENT_CONTEXT_LEN,
                )?,
                Layer::load(
                    bundle,
                    "events.hazard.2.weight",
                    "events.hazard.2.bias",
                    32,
                    64,
                )?,
                Layer::load(
                    bundle,
                    "events.hazard.4.weight",
                    "events.hazard.4.bias",
                    2,
                    32,
                )?,
            ],
            brake_encoder: [
                Layer::load(
                    bundle,
                    "events.encoder.0.weight",
                    "events.encoder.0.bias",
                    HIDDEN,
                    EVENT_CONTEXT_LEN,
                )?,
                Layer::load(
                    bundle,
                    "events.encoder.2.weight",
                    "events.encoder.2.bias",
                    HIDDEN,
                    HIDDEN,
                )?,
            ],
            brake: Layer::load(
                bundle,
                "events.brake.weight",
                "events.brake.bias",
                HEADS * 11,
                HIDDEN,
            )?,
            frequency: Layer::load(
                bundle,
                "events.frequency.weight",
                "events.frequency.bias",
                HEADS,
                HIDDEN,
            )?,
            encoded: vec![0.0; HIDDEN],
            coefficients: vec![0.0; COEFFICIENTS],
            logits: [0.0; HEADS],
        })
    }

    pub fn motor(&mut self, coarse: &[f32], fine: &[f32], dynamics: &[f32]) -> Result<()> {
        self.motor_inner(coarse, fine, dynamics).map_err(wrap)
    }

    fn motor_inner(&mut self, coarse: &[f32], fine: &[f32], dynamics: &[f32]) -> Outcome<()> {
        let tokens = Tensor::from_slice(coarse, (TOKENS, 9), &Device::Cpu)?;
        let tokens = silu(&self.input.apply(&tokens)?)?;
        let projected = self.input_gates.apply(&tokens)?;
        let mut hidden = Tensor::zeros((1, HIDDEN), DType::F32, &Device::Cpu)?;
        for row in 0..TOKENS {
            let recurrent = self.recurrent.apply(&hidden)?;
            let gate = projected.narrow(0, row, 1)?;
            let reset = sigmoid(
                &gate
                    .narrow(1, 0, HIDDEN)?
                    .add(&recurrent.narrow(1, 0, HIDDEN)?)?,
            )?;
            let update = sigmoid(
                &gate
                    .narrow(1, HIDDEN, HIDDEN)?
                    .add(&recurrent.narrow(1, HIDDEN, HIDDEN)?)?,
            )?;
            let candidate = gate
                .narrow(1, 2 * HIDDEN, HIDDEN)?
                .add(&reset.mul(&recurrent.narrow(1, 2 * HIDDEN, HIDDEN)?)?)?
                .tanh()?;
            hidden = candidate.add(&hidden.sub(&candidate)?.mul(&update)?)?;
        }
        let mut combined = vec![0.0f32; COMBINED];
        combined[..HIDDEN].copy_from_slice(&hidden.flatten_all()?.to_vec1::<f32>()?);
        combined[HIDDEN..HIDDEN + FINE_LEN].copy_from_slice(fine);
        combined[HIDDEN + FINE_LEN..].copy_from_slice(dynamics);
        let combined = Tensor::from_slice(&combined, (1, COMBINED), &Device::Cpu)?;
        let encoded = silu(&self.trunk.apply(&combined)?)?;
        self.encoded = encoded.flatten_all()?.to_vec1::<f32>()?;
        self.coefficients = self
            .output
            .apply(&encoded)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(())
    }

    pub fn choice(&mut self, geometry: &[f32], pairs: &[f32], previous_valid: bool) -> Result<()> {
        self.choice_inner(geometry, pairs, previous_valid)
            .map_err(wrap)
    }

    fn choice_inner(
        &mut self,
        geometry: &[f32],
        pairs: &[f32],
        previous_valid: bool,
    ) -> Outcome<()> {
        let encoded = Tensor::from_slice(&self.encoded, (1, HIDDEN), &Device::Cpu)?;
        let shared = encoded
            .matmul(&self.unary_latent)?
            .broadcast_add(&self.unary_bias)?;
        let geometry = Tensor::from_slice(geometry, (HEADS, GEOMETRY_STEPS), &Device::Cpu)?;
        let unary = silu(
            &geometry
                .matmul(&self.unary_geometry)?
                .broadcast_add(&shared)?,
        )?;
        let mut logits = self
            .unary_output
            .apply(&unary)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if previous_valid {
            let context = silu(&self.context.apply(&encoded)?)?;
            let shared_pair = context
                .matmul(&self.pair_context)?
                .broadcast_add(&self.pair_bias)?;
            let pairs = Tensor::from_slice(pairs, (HEADS, PAIR_LEN), &Device::Cpu)?;
            let pair = silu(
                &pairs
                    .matmul(&self.pair_geometry)?
                    .broadcast_add(&shared_pair)?,
            )?;
            let transition = self
                .pair_output
                .apply(&pair)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for head in 0..HEADS {
                logits[head] += transition[head];
            }
        }
        self.logits.copy_from_slice(&logits);
        Ok(())
    }

    pub fn hazard(&mut self, context: &[f32], out: &mut [f32; 2]) -> Result<()> {
        let values = self.hazard_inner(context).map_err(wrap)?;
        out.copy_from_slice(&values);
        Ok(())
    }

    fn hazard_inner(&self, context: &[f32]) -> Outcome<Vec<f32>> {
        let input = Tensor::from_slice(context, (1, EVENT_CONTEXT_LEN), &Device::Cpu)?;
        let first = silu(&self.hazard[0].apply(&input)?)?;
        let second = silu(&self.hazard[1].apply(&first)?)?;
        self.hazard[2]
            .apply(&second)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    pub fn brake(&mut self, context: &[f32], out: &mut EventOutputs) -> Result<()> {
        let (parameters, frequency) = self.brake_inner(context).map_err(wrap)?;
        for head in 0..HEADS {
            let log_duration = parameters[11 * head].clamp(BRAKE_LOG_MINIMUM, BRAKE_LOG_MAXIMUM);
            out.duration[head] = log_duration.exp();
            for column in 0..10 {
                out.coefficients[10 * head + column] =
                    BRAKE_SCALE * parameters[11 * head + column + 1];
            }
        }
        out.frequency.copy_from_slice(&frequency);
        Ok(())
    }

    fn brake_inner(&self, context: &[f32]) -> Outcome<(Vec<f32>, Vec<f32>)> {
        let input = Tensor::from_slice(context, (1, EVENT_CONTEXT_LEN), &Device::Cpu)?;
        let stage = silu(&self.brake_encoder[0].apply(&input)?)?;
        let encoded = silu(&self.brake_encoder[1].apply(&stage)?)?;
        Ok((
            self.brake
                .apply(&encoded)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            self.frequency
                .apply(&encoded)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        ))
    }

    pub fn events(&mut self, context: &[f32], out: &mut EventOutputs) -> Result<()> {
        self.brake(context, out)?;
        let mut hazard = [0.0f32; 2];
        self.hazard(context, &mut hazard)?;
        out.hazard = hazard;
        Ok(())
    }
}
