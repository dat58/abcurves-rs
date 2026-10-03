use crate::nn::linear::{matmul, matvec};

pub const CHANNELS: usize = 96;
pub const KERNEL: usize = 5;
pub const BLOCKS: usize = 3;
pub const CONVOLUTIONS: usize = 2 * BLOCKS;
pub const WINDOW: usize = 1 + CONVOLUTIONS * (KERNEL - 1);
pub const INPUT_CHANNELS: usize = 3;
pub const LAYER_NORM_EPS: f64 = 1e-5;

pub struct TcnWeights {
    pub input_weight: Vec<f32>,
    pub input_bias: Vec<f32>,
    pub conv_weight: Vec<f32>,
    pub conv_bias: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub summary_weight: Vec<f32>,
    pub summary_bias: Vec<f32>,
    pub trunk_weight: Vec<f32>,
    pub trunk_bias: Vec<f32>,
    pub head_weight: Vec<f32>,
    pub head_bias: Vec<f32>,
    pub heads: usize,
    pub out_dim: usize,
    pub summary_dim: usize,
}

pub struct TcnWorkspace {
    x: Vec<f32>,
    y: Vec<f32>,
    col: Vec<f32>,
    staged: Vec<f32>,
    summary: Vec<f32>,
    combined: Vec<f32>,
    trunk: Vec<f32>,
}

impl TcnWorkspace {
    pub fn new(summary_dim: usize) -> Self {
        let _ = summary_dim;
        Self {
            x: vec![0.0; WINDOW * CHANNELS],
            y: vec![0.0; WINDOW * CHANNELS],
            col: vec![0.0; WINDOW * KERNEL * CHANNELS],
            staged: vec![0.0; WINDOW * CHANNELS],
            summary: vec![0.0; CHANNELS],
            combined: vec![0.0; 2 * CHANNELS],
            trunk: vec![0.0; CHANNELS],
        }
    }
}

#[inline]
fn relu(value: f32) -> f32 {
    if value > 0.0 { value } else { 0.0 }
}

/// Receptive-cone trim: only the last row is consumed downstream, so each
/// convolution only needs output rows inside the cone. Rows below the cone are
/// never read, so this is selection, not approximation.
pub fn forward(
    weights: &TcnWeights,
    space: &mut TcnWorkspace,
    window: &[f32],
    summary: &[f32],
    head_weight: &[f32],
    head_bias: &[f32],
    out: &mut [f32],
) {
    matmul(
        window,
        WINDOW,
        INPUT_CHANNELS,
        &weights.input_weight,
        CHANNELS,
        &mut space.x,
    );
    for row in 0..WINDOW {
        for column in 0..CHANNELS {
            space.x[row * CHANNELS + column] += weights.input_bias[column];
        }
    }

    space.y.fill(0.0);
    space.col.fill(0.0);
    for block in 0..BLOCKS {
        for inner in 0..2 {
            let index = block * 2 + inner;
            let low = (KERNEL - 1) * (index + 1);
            for row in low..WINDOW {
                for tap in 0..KERNEL {
                    let source = row + tap - (KERNEL - 1);
                    for column in 0..CHANNELS {
                        let value = if inner == 0 {
                            space.x[source * CHANNELS + column]
                        } else {
                            space.y[source * CHANNELS + column]
                        };
                        space.col[row * KERNEL * CHANNELS + tap * CHANNELS + column] = value;
                    }
                }
            }
            let rows = WINDOW - low;
            matmul(
                &space.col[low * KERNEL * CHANNELS..],
                rows,
                KERNEL * CHANNELS,
                &weights.conv_weight[index * KERNEL * CHANNELS * CHANNELS..],
                CHANNELS,
                &mut space.staged[..rows * CHANNELS],
            );
            for row in low..WINDOW {
                for column in 0..CHANNELS {
                    space.y[row * CHANNELS + column] = space.staged[(row - low) * CHANNELS + column]
                        + weights.conv_bias[index * CHANNELS + column];
                }
                let lane = &mut space.y[row * CHANNELS..(row + 1) * CHANNELS];
                let mut mean = 0.0f64;
                for column in 0..CHANNELS {
                    mean += f64::from(lane[column]);
                }
                mean /= CHANNELS as f64;
                let mut variance = 0.0f64;
                for column in 0..CHANNELS {
                    let deviation = f64::from(lane[column]) - mean;
                    variance += deviation * deviation;
                }
                variance /= CHANNELS as f64;
                let inverse = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
                for column in 0..CHANNELS {
                    let value = ((f64::from(lane[column]) - mean) * inverse) as f32
                        * weights.norm_weight[index * CHANNELS + column]
                        + weights.norm_bias[index * CHANNELS + column];
                    lane[column] = relu(value);
                }
            }
        }
        for row in 2 * (KERNEL - 1) * (block + 1)..WINDOW {
            for column in 0..CHANNELS {
                space.x[row * CHANNELS + column] += space.y[row * CHANNELS + column];
            }
        }
    }

    matvec(
        &weights.summary_weight,
        CHANNELS,
        weights.summary_dim,
        summary,
        &mut space.summary,
    );
    for column in 0..CHANNELS {
        space.summary[column] = relu(space.summary[column] + weights.summary_bias[column]);
    }
    space.combined[..CHANNELS]
        .copy_from_slice(&space.x[(WINDOW - 1) * CHANNELS..WINDOW * CHANNELS]);
    space.combined[CHANNELS..].copy_from_slice(&space.summary);
    matvec(
        &weights.trunk_weight,
        CHANNELS,
        2 * CHANNELS,
        &space.combined,
        &mut space.trunk,
    );
    for column in 0..CHANNELS {
        space.trunk[column] = relu(space.trunk[column] + weights.trunk_bias[column]);
    }
    for slot in 0..out.len() {
        let mut total = 0.0f32;
        for column in 0..CHANNELS {
            total += space.trunk[column] * head_weight[column * out.len() + slot];
        }
        out[slot] = total + head_bias[slot];
    }
}
