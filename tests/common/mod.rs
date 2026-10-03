#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn golden(fixture: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(fixture)
}

fn blob(fixture: &str, name: &str, suffix: &str) -> Vec<u8> {
    let path = golden(fixture).join(format!("{name}.{suffix}"));
    std::fs::read(&path).unwrap_or_else(|error| panic!("missing {}: {error}", path.display()))
}

macro_rules! reader {
    ($name:ident, $kind:ty, $suffix:literal, $width:literal) => {
        pub fn $name(fixture: &str, field: &str) -> Vec<$kind> {
            blob(fixture, field, $suffix)
                .chunks_exact($width)
                .map(|chunk| <$kind>::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        }
    };
}

reader!(read_f32, f32, "f32", 4);
reader!(read_f64, f64, "f64", 8);
reader!(read_i16, i16, "i16", 2);
reader!(read_i32, i32, "i32", 4);
reader!(read_i64, i64, "i64", 8);
reader!(read_u64, u64, "u64", 8);
reader!(read_i8, i8, "i8", 1);
reader!(read_u8, u8, "u8", 1);

pub fn models_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("ABCURVES_MODEL_DIR") {
        return PathBuf::from(explicit);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("origin")
        .join("ABCurves")
        .join("models")
}

pub fn legacy_uniforms(seed: u32, count: usize) -> Vec<f64> {
    let mut generator = abcurves::rng::Mt19937::new(seed);
    (0..count)
        .map(|_| {
            let high = f64::from(generator.next_u32() >> 5);
            let low = f64::from(generator.next_u32() >> 6);
            (high * 67_108_864.0 + low) / 9_007_199_254_740_992.0
        })
        .collect()
}

pub const CASE_UNIFORMS: usize = 5376;
pub const CASES: usize = 32;
pub const CASE_CUT_US: i64 = 1_000_000;

pub struct Case {
    pub history: Vec<[f64; 2]>,
    pub target: Vec<[f64; 2]>,
    pub available: Vec<i64>,
    pub valid: Vec<bool>,
    pub motion_known: Vec<bool>,
    pub position: [f64; 2],
    pub hold_age: f64,
    pub hold_target: [f64; 2],
    pub hold_position: [f64; 2],
    pub initial_error: f64,
    pub innovation_age: f64,
    pub mode: u8,
    pub brake_velocity: [f64; 2],
    pub brake_acceleration: [f64; 2],
    pub brake_duration: f64,
    pub brake_age: f64,
    pub coefficients: Vec<f32>,
    pub brake_coefficients: [[f64; 2]; 5],
    pub previous: Vec<f32>,
}

pub fn build_case(seed: u32) -> Case {
    let u = legacy_uniforms(seed, CASE_UNIFORMS);
    let mut history = Vec::with_capacity(640);
    let mut target = Vec::with_capacity(640);
    for index in 0..640 {
        history.push([
            (u[index * 2] - 0.5) * 8.0,
            (u[index * 2 + 1] - 0.5) * 8.0,
        ]);
        target.push([
            (u[1280 + index * 2] - 0.5) * 300.0,
            (u[1280 + index * 2 + 1] - 0.5) * 300.0,
        ]);
    }
    let mut valid: Vec<bool> = (0..640).map(|index| u[2560 + index] > 0.25).collect();
    let mut motion_known: Vec<bool> = (0..640).map(|index| u[3200 + index] > 0.1).collect();
    let available: Vec<i64> = (0..640)
        .map(|index| {
            CASE_CUT_US + (index as i64 - 639) * 1000
                - ((u[3840 + index] - 0.45) * 4000.0).round_ties_even() as i64
        })
        .collect();

    if seed % 3 == 1 {
        target[511][0] = f64::NAN;
    }
    if seed % 3 == 2 {
        target[500][1] = f64::INFINITY;
        valid.iter_mut().for_each(|slot| *slot = true);
    }
    if seed == 0 {
        history.iter_mut().for_each(|row| *row = [0.0, 0.0]);
        valid.iter_mut().for_each(|slot| *slot = false);
        motion_known.iter_mut().for_each(|slot| *slot = false);
    }
    if seed == 1 {
        valid.iter_mut().for_each(|slot| *slot = true);
        motion_known.iter_mut().for_each(|slot| *slot = true);
        history.iter_mut().for_each(|row| *row = [0.0, 0.0]);
    }

    Case {
        history,
        target,
        available,
        valid,
        motion_known,
        position: [(u[4480] - 0.5) * 200.0, (u[4481] - 0.5) * 200.0],
        hold_age: u[4482] * 6000.0,
        hold_target: [(u[4483] - 0.5) * 300.0, (u[4484] - 0.5) * 300.0],
        hold_position: [(u[4485] - 0.5) * 300.0, (u[4486] - 0.5) * 300.0],
        initial_error: u[4487] * 400.0,
        innovation_age: u[4488] * 4000.0,
        mode: (u[4489] * 3.0) as u8,
        brake_velocity: [(u[4490] - 0.5) * 6.0, (u[4491] - 0.5) * 6.0],
        brake_acceleration: [(u[4492] - 0.5) * 0.5, (u[4493] - 0.5) * 0.5],
        brake_duration: 4.0 + u[4494] * 188.0,
        brake_age: u[4495] * 200.0,
        coefficients: (0..672).map(|index| ((u[4608 + index] - 0.5) * 2.0) as f32).collect(),
        brake_coefficients: std::array::from_fn(|row| {
            [
                (u[5280 + row * 2] - 0.5) * 40.0,
                (u[5280 + row * 2 + 1] - 0.5) * 40.0,
            ]
        }),
        previous: (0..32).map(|index| ((u[5290 + index] - 0.5) * 4.0) as f32).collect(),
    }
}

pub fn build_encoded(seed: u32) -> Vec<f32> {
    legacy_uniforms(seed + 1000, 96)
        .into_iter()
        .map(|value| ((value - 0.5) * 4.0) as f32)
        .collect()
}

pub fn relative_drift(actual: &[f32], expected: &[f32], label: &str) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut worst = 0.0f32;
    let scale = expected
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()))
        .max(1e-6);
    for (lane, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(left.is_finite(), "{label} lane {lane} is not finite");
        worst = worst.max((left - right).abs() / scale);
    }
    worst
}
