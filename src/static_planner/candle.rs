use super::planner::{HEADS, OUT_DIM};
use super::tcn::{BLOCKS, CHANNELS, INPUT_CHANNELS, KERNEL, LAYER_NORM_EPS, WINDOW};
use crate::error::{Error, Result};
use crate::io::{Checkpoint, Value};
use candle_core::{D, DType, Device, Tensor};

fn wrap(error: candle_core::Error) -> Error {
    Error::Numerical(format!("candle: {error}"))
}

type Outcome<T> = std::result::Result<T, candle_core::Error>;

fn relu(value: &Tensor) -> Outcome<Tensor> {
    value.relu()
}

struct Block {
    conv: [Tensor; 2],
    conv_bias: [Tensor; 2],
    norm_weight: [Tensor; 2],
    norm_bias: [Tensor; 2],
}

/// Reference backend for the Static Planner encoder: the same learned tensors
/// with float32 reductions instead of the compiled kernel's float64 ladder.
pub struct CandleTcn {
    input_weight: Tensor,
    input_bias: Tensor,
    blocks: Vec<Block>,
    summary_weight: Tensor,
    summary_bias: Tensor,
    trunk_weight: Tensor,
    trunk_bias: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    summary_dim: usize,
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

impl CandleTcn {
    pub fn load(checkpoint: &Checkpoint, summary_dim: usize) -> Result<Self> {
        let state = checkpoint.root().entry("model_state_dict")?;
        let take = |name: &str| -> Result<Vec<f32>> {
            let value: &Value = state.entry(name)?;
            checkpoint.tensor(value.as_tensor()?)?.to_f32()
        };
        let mut blocks = Vec::with_capacity(BLOCKS);
        for block in 0..BLOCKS {
            let mut conv = Vec::new();
            let mut conv_bias = Vec::new();
            let mut norm_weight = Vec::new();
            let mut norm_bias = Vec::new();
            for inner in 1..=2 {
                conv.push(tensor(
                    &take(&format!("encoder.blocks.{block}.conv{inner}.weight"))?,
                    &[CHANNELS, CHANNELS, KERNEL],
                )?);
                conv_bias.push(tensor(
                    &take(&format!("encoder.blocks.{block}.conv{inner}.bias"))?,
                    &[CHANNELS],
                )?);
                norm_weight.push(tensor(
                    &take(&format!("encoder.blocks.{block}.norm{inner}.weight"))?,
                    &[CHANNELS],
                )?);
                norm_bias.push(tensor(
                    &take(&format!("encoder.blocks.{block}.norm{inner}.bias"))?,
                    &[CHANNELS],
                )?);
            }
            blocks.push(Block {
                conv: [conv[0].clone(), conv[1].clone()],
                conv_bias: [conv_bias[0].clone(), conv_bias[1].clone()],
                norm_weight: [norm_weight[0].clone(), norm_weight[1].clone()],
                norm_bias: [norm_bias[0].clone(), norm_bias[1].clone()],
            });
        }
        Ok(Self {
            input_weight: tensor(
                &take("encoder.input.weight")?,
                &[CHANNELS, INPUT_CHANNELS, 1],
            )?,
            input_bias: tensor(&take("encoder.input.bias")?, &[CHANNELS])?,
            blocks,
            summary_weight: tensor(
                &transpose(&take("encoder.summary.0.weight")?, CHANNELS, summary_dim),
                &[summary_dim, CHANNELS],
            )?,
            summary_bias: tensor(&take("encoder.summary.0.bias")?, &[CHANNELS])?,
            trunk_weight: tensor(
                &transpose(&take("trunk.0.weight")?, CHANNELS, 2 * CHANNELS),
                &[2 * CHANNELS, CHANNELS],
            )?,
            trunk_bias: tensor(&take("trunk.0.bias")?, &[CHANNELS])?,
            head_weight: tensor(
                &transpose(&take("out.weight")?, HEADS * OUT_DIM, CHANNELS),
                &[CHANNELS, HEADS * OUT_DIM],
            )?,
            head_bias: tensor(&take("out.bias")?, &[HEADS * OUT_DIM])?,
            summary_dim,
        })
    }

    pub fn forward(
        &self,
        window: &[f32],
        summary: &[f32],
        head: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let values = self.forward_inner(window, summary).map_err(wrap)?;
        out.copy_from_slice(&values[head * OUT_DIM..(head + 1) * OUT_DIM]);
        Ok(())
    }

    fn layer_norm(&self, value: &Tensor, weight: &Tensor, bias: &Tensor) -> Outcome<Tensor> {
        let mean = value.mean_keepdim(D::Minus1)?;
        let centered = value.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        centered
            .broadcast_div(&(variance + LAYER_NORM_EPS)?.sqrt()?)?
            .broadcast_mul(weight)?
            .broadcast_add(bias)
    }

    fn causal(&self, value: &Tensor) -> Outcome<Tensor> {
        let pad = Tensor::zeros((1, CHANNELS, KERNEL - 1), DType::F32, &Device::Cpu)?;
        Tensor::cat(&[&pad, value], 2)
    }

    fn forward_inner(&self, window: &[f32], summary: &[f32]) -> Outcome<Vec<f32>> {
        let input = Tensor::from_slice(window, (1, WINDOW, INPUT_CHANNELS), &Device::Cpu)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut x = input
            .conv1d(&self.input_weight, 0, 1, 1, 1)?
            .broadcast_add(&self.input_bias.reshape((1, CHANNELS, 1))?)?;
        for block in &self.blocks {
            let mut y = x.clone();
            for inner in 0..2 {
                let padded = self.causal(&y)?;
                let convolved = padded
                    .conv1d(&block.conv[inner], 0, 1, 1, 1)?
                    .broadcast_add(&block.conv_bias[inner].reshape((1, CHANNELS, 1))?)?
                    .transpose(1, 2)?
                    .contiguous()?;
                y = relu(&self.layer_norm(
                    &convolved,
                    &block.norm_weight[inner],
                    &block.norm_bias[inner],
                )?)?
                .transpose(1, 2)?
                .contiguous()?;
            }
            x = x.add(&y)?;
        }
        let last = x.transpose(1, 2)?.contiguous()?.narrow(1, WINDOW - 1, 1)?;
        let summary = Tensor::from_slice(summary, (1, self.summary_dim), &Device::Cpu)?;
        let encoded = relu(
            &summary
                .matmul(&self.summary_weight)?
                .broadcast_add(&self.summary_bias)?,
        )?;
        let combined = Tensor::cat(&[&last.reshape((1, CHANNELS))?, &encoded], 1)?;
        let trunk = relu(
            &combined
                .matmul(&self.trunk_weight)?
                .broadcast_add(&self.trunk_bias)?,
        )?;
        trunk
            .matmul(&self.head_weight)?
            .broadcast_add(&self.head_bias)?
            .flatten_all()?
            .to_vec1::<f32>()
    }
}
