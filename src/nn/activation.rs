use crate::math::exp_f32;

// Bounded Pade approximants. Coefficients below describe activation functions,
// not learned parameters. Inputs are clamped before squaring.
#[inline]
pub fn tanh_pade9(value: f32) -> f32 {
    let x = value.clamp(-7.0, 7.0);
    let z = x * x;
    let numerator = x * (34_459_425.0 + z * (4_729_725.0 + z * (135_135.0 + z * (990.0 + z))));
    let denominator =
        34_459_425.0 + z * (16_216_200.0 + z * (945_945.0 + z * (13_860.0 + 45.0 * z)));
    (numerator / denominator).clamp(-1.0, 1.0)
}

#[inline]
pub fn sigmoid_pade9(value: f32) -> f32 {
    0.5 + 0.5 * tanh_pade9(0.5 * value)
}

#[inline]
pub fn silu_pade9(value: f32) -> f32 {
    value * sigmoid_pade9(value)
}

// Keep the exponential finite even on very negative selector inputs. Fast
// reciprocal code must never receive an intermediate infinity.
#[inline]
pub fn sigmoid_exact(value: f32) -> f32 {
    let z = exp_f32(-value.abs());
    let denominator = 1.0 + z;
    if value >= 0.0 {
        1.0 / denominator
    } else {
        z / denominator
    }
}

#[inline]
pub fn silu_exact(value: f32) -> f32 {
    value * sigmoid_exact(value)
}
