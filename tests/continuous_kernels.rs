#![allow(clippy::needless_range_loop)]

mod common;

use abcurves::continuous::kernels::{self, EventFeatures, EventState, MotorFeatures, Window};
use abcurves::io::Npz;
use common::{CASE_CUT_US, CASES, build_case, models_root, read_f32, read_f64};

const FIXTURE: &str = "continuous_kernels";

fn same_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (lane, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left == right || (left.is_nan() && right.is_nan()),
            "{label} lane {lane}: {left} vs {right}"
        );
    }
}

fn same_f64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (lane, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left == right || (left.is_nan() && right.is_nan()),
            "{label} lane {lane}: {left} vs {right}"
        );
    }
}

fn slice_f32(data: &[f32], case: usize, width: usize) -> &[f32] {
    &data[case * width..(case + 1) * width]
}

fn slice_f64(data: &[f64], case: usize, width: usize) -> &[f64] {
    &data[case * width..(case + 1) * width]
}

#[test]
fn physical_kernels_match_the_numba_reference() {
    let path = models_root().join("continuous").join("weights.npz");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }
    let bundle = Npz::open(&path).unwrap();
    let velocity_basis = bundle.f64("motor.velocity_basis").unwrap();
    let carry_velocity = bundle.f64("motor.carry_velocity").unwrap();
    let (mean_basis, mean_carry) = kernels::decoder_geometry(&velocity_basis, &carry_velocity);
    assert_eq!(mean_basis, read_f64(FIXTURE, "mean_basis"));
    assert_eq!(mean_carry, read_f64(FIXTURE, "mean_carry"));

    let coarse = read_f32(FIXTURE, "coarse");
    let fine = read_f32(FIXTURE, "fine");
    let dynamics = read_f32(FIXTURE, "dynamics");
    let raw = read_f32(FIXTURE, "raw");
    let context = read_f32(FIXTURE, "context");
    let basis = read_f64(FIXTURE, "basis");
    let heads = read_f32(FIXTURE, "heads");
    let geometry = read_f32(FIXTURE, "geometry");
    let pairs = read_f32(FIXTURE, "pairs");
    let selected = read_f64(FIXTURE, "selected");
    let brake = read_f64(FIXTURE, "brake");

    for case in 0..CASES {
        let built = build_case(case as u32);
        let window = Window {
            history: &built.history,
            target: &built.target,
            available: &built.available,
            valid: &built.valid,
            motion_known: &built.motion_known,
            position: built.position,
            cut_us: CASE_CUT_US,
        };

        let mut motor = MotorFeatures::default();
        kernels::motor_features(&window, &mut motor);
        same_f32(
            &motor.coarse,
            slice_f32(&coarse, case, 630),
            &format!("coarse {case}"),
        );
        same_f32(
            &motor.fine,
            slice_f32(&fine, case, 54),
            &format!("fine {case}"),
        );
        same_f32(
            &motor.dynamics,
            slice_f32(&dynamics, case, 8),
            &format!("dynamics {case}"),
        );

        let state = EventState {
            hold_age: built.hold_age,
            hold_target: built.hold_target,
            hold_position: built.hold_position,
            initial_error: built.initial_error,
            innovation_age: built.innovation_age,
            mode: built.mode,
        };
        let mut events = EventFeatures::default();
        kernels::event_features(&window, &state, &mut events);
        same_f32(
            &events.raw,
            slice_f32(&raw, case, 22),
            &format!("raw {case}"),
        );
        same_f32(
            &events.context,
            slice_f32(&context, case, 32),
            &format!("context {case}"),
        );
        let flat = [
            events.basis[0][0],
            events.basis[0][1],
            events.basis[1][0],
            events.basis[1][1],
        ];
        same_f64(&flat, slice_f64(&basis, case, 4), &format!("basis {case}"));

        let incoming = built.history[639];
        let mut head_geometry = vec![0.0f32; 16 * 16 * 2];
        kernels::decode_geometry(
            &built.coefficients,
            incoming,
            &mean_basis,
            &mean_carry,
            &mut head_geometry,
        );
        same_f32(
            &head_geometry,
            slice_f32(&heads, case, 512),
            &format!("heads {case}"),
        );

        let mut unary = vec![0.0f32; 16 * 16];
        let mut pair = vec![0.0f32; 16 * 20];
        let previous = if case % 2 == 1 {
            Some((built.previous.as_slice(), [0.5f32, -0.25]))
        } else {
            None
        };
        kernels::selector_inputs(&head_geometry, previous, &mut unary, &mut pair);
        same_f32(
            &unary,
            slice_f32(&geometry, case, 256),
            &format!("geometry {case}"),
        );
        same_f32(
            &pair,
            slice_f32(&pairs, case, 320),
            &format!("pairs {case}"),
        );

        let head = case % 16;
        let mut action = vec![[0.0f64; 2]; 32];
        kernels::decode_selected(
            &built.coefficients[head * 42..(head + 1) * 42],
            incoming,
            &velocity_basis[..32 * 21],
            &carry_velocity[..32],
            &mut action,
        );
        let flat: Vec<f64> = action.iter().flat_map(|row| row.iter().copied()).collect();
        let expected = slice_f64(&selected, case, 64);
        for (lane, value) in flat.iter().enumerate() {
            let drift = (value - expected[lane]).abs();
            assert!(
                drift < 1e-12 || (value.is_nan() && expected[lane].is_nan()),
                "selected {case} lane {lane} drifted {drift}; numpy reduces through blas"
            );
        }

        let times: Vec<f64> = (0..33).map(|step| built.brake_age + step as f64).collect();
        let mut curve = vec![[0.0f64; 2]; 33];
        kernels::c2_path(
            built.brake_velocity,
            built.brake_acceleration,
            built.brake_duration,
            &built.brake_coefficients,
            &times,
            &mut curve,
        );
        let flat: Vec<f64> = curve.iter().flat_map(|row| row.iter().copied()).collect();
        same_f64(&flat, slice_f64(&brake, case, 66), &format!("brake {case}"));
    }
}
