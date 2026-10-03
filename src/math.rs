unsafe extern "C" {
    fn asinhf(value: f32) -> f32;
    fn expf(value: f32) -> f32;
    fn tanhf(value: f32) -> f32;
}

#[inline]
pub fn asinh_f32(value: f32) -> f32 {
    unsafe { asinhf(value) }
}

#[inline]
pub fn exp_f32(value: f32) -> f32 {
    unsafe { expf(value) }
}

#[inline]
pub fn tanh_f32(value: f32) -> f32 {
    unsafe { tanhf(value) }
}

unsafe extern "C" {
    fn logf(value: f32) -> f32;
}

#[inline]
pub fn log_f32(value: f32) -> f32 {
    unsafe { logf(value) }
}
