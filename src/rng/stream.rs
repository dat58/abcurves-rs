use super::mt19937::Mt19937;
use crate::error::{Error, Result};

const UNIFORM_SCALE: f32 = 1.0 / 16_777_216.0;
const MANTISSA_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
const MANTISSA_MASK: u64 = (1u64 << 53) - 1;
const INLINE_CATEGORIES: usize = 64;

#[derive(Clone)]
pub struct RandomStream {
    seed: u64,
    generator: Mt19937,
}

impl RandomStream {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            generator: Mt19937::new((seed & 0xffff_ffff) as u32),
        }
    }

    pub fn from_signed(seed: i64) -> Self {
        Self::new(seed as u64)
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[inline]
    fn word(&mut self) -> u32 {
        self.generator.next_u32()
    }

    #[inline]
    pub fn uniform_scalar(&mut self) -> f32 {
        ((self.word() & 0x00ff_ffff) as f32) * UNIFORM_SCALE
    }

    pub fn uniform_into(&mut self, out: &mut [f32]) {
        for slot in out.iter_mut() {
            *slot = self.uniform_scalar();
        }
    }

    pub fn uniform(&mut self, count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count];
        self.uniform_into(&mut out);
        out
    }

    #[inline]
    fn exponential_scalar(&mut self) -> f32 {
        let high = self.word() as u64;
        let low = self.word() as u64;
        let bits = (high << 32) | low;
        let uniform = ((bits & MANTISSA_MASK) as f64) * MANTISSA_SCALE;
        (-((-uniform).ln_1p())) as f32
    }

    pub fn exponential_into(&mut self, out: &mut [f32]) {
        for slot in out.iter_mut() {
            *slot = self.exponential_scalar();
        }
    }

    pub fn exponential(&mut self, count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count];
        self.exponential_into(&mut out);
        out
    }

    pub fn categorical(&mut self, probabilities: &[f32]) -> usize {
        let count = probabilities.len();
        if count <= INLINE_CATEGORIES {
            let mut draws = [0.0f32; INLINE_CATEGORIES];
            self.exponential_into(&mut draws[..count]);
            race(probabilities, &draws[..count])
        } else {
            let draws = self.exponential(count);
            race(probabilities, &draws)
        }
    }

    pub fn discard_categorical(&mut self, size: usize) {
        for _ in 0..2 * size {
            self.word();
        }
    }
}

fn race(weights: &[f32], draws: &[f32]) -> usize {
    let mut best = weights[0] / draws[0];
    let mut index = 0usize;
    if best.is_nan() {
        return index;
    }
    for position in 1..weights.len() {
        let score = weights[position] / draws[position];
        if !(score <= best) {
            best = score;
            index = position;
            if best.is_nan() {
                break;
            }
        }
    }
    index
}

pub fn pairwise_sum(values: &[f32]) -> f32 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        let mut total = 0.0f32;
        for &value in values {
            total += value;
        }
        total
    } else if n <= BLOCK {
        let mut lanes = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let mut index = 8;
        while index < n - (n % 8) {
            for lane in 0..8 {
                lanes[lane] += values[index + lane];
            }
            index += 8;
        }
        let mut total =
            ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while index < n {
            total += values[index];
            index += 1;
        }
        total
    } else {
        let half = (n / 2) - ((n / 2) % 8);
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

pub fn softmax_into(logits: &[f32], out: &mut [f32]) -> Result<()> {
    let mut maximum = f32::NEG_INFINITY;
    for &value in logits {
        if value > maximum {
            maximum = value;
        }
    }
    for (slot, &value) in out.iter_mut().zip(logits) {
        *slot = (value - maximum).exp();
    }
    let total = pairwise_sum(out);
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::Numerical("Invalid categorical probabilities".into()));
    }
    for slot in out.iter_mut() {
        *slot /= total;
    }
    Ok(())
}

pub fn softmax(logits: &[f32]) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; logits.len()];
    softmax_into(logits, &mut out)?;
    Ok(out)
}
