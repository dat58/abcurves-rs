mod common;

use abcurves::continuous::kernels::{self, EventFeatures, EventState, MotorFeatures, Window};
use abcurves::io::Npz;
use abcurves::nn::{EventOutputs, NativeEngine};
use common::{CASES, CASE_CUT_US, build_case, build_encoded, models_root, read_f32, relative_drift};

const FIXTURE: &str = "continuous_neural";

fn part(data: &[f32], case: usize, width: usize) -> &[f32] {
    &data[case * width..(case + 1) * width]
}

#[test]
fn neural_kernels_track_the_native_reference() {
    let path = models_root().join("continuous").join("weights.npz");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }
    let bundle = Npz::open(&path).unwrap();
    let mut engine = NativeEngine::load(&bundle).unwrap();
    let velocity_basis = bundle.f64("motor.velocity_basis").unwrap();
    let carry_velocity = bundle.f64("motor.carry_velocity").unwrap();
    let (mean_basis, mean_carry) = kernels::decoder_geometry(&velocity_basis, &carry_velocity);

    let encoded = read_f32(FIXTURE, "encoded");
    let coefficients = read_f32(FIXTURE, "coefficients");
    let logits = read_f32(FIXTURE, "logits");
    let duration = read_f32(FIXTURE, "duration");
    let brake_coefficients = read_f32(FIXTURE, "brake_coefficients");
    let frequency = read_f32(FIXTURE, "frequency");
    let hazard = read_f32(FIXTURE, "hazard");

    let mut worst_encoded = 0.0f32;
    let mut worst_coefficients = 0.0f32;
    let mut worst_logits = 0.0f32;
    let mut worst_events = 0.0f32;

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
        if motor.coarse.iter().any(|value| !value.is_finite()) {
            continue;
        }
        engine.motor(&motor.coarse, &motor.fine, &motor.dynamics);
        worst_encoded =
            worst_encoded.max(relative_drift(&engine.encoded, part(&encoded, case, 96), "encoded"));
        worst_coefficients = worst_coefficients.max(relative_drift(
            &engine.coefficients,
            part(&coefficients, case, 672),
            "coefficients",
        ));

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
        let mut outputs = EventOutputs::default();
        engine.events(&events.context, &mut outputs);
        worst_events = worst_events
            .max(relative_drift(&outputs.duration, part(&duration, case, 16), "duration"))
            .max(relative_drift(
                &outputs.coefficients,
                part(&brake_coefficients, case, 160),
                "brake coefficients",
            ))
            .max(relative_drift(&outputs.frequency, part(&frequency, case, 16), "frequency"))
            .max(relative_drift(&outputs.hazard, part(&hazard, case, 2), "hazard"));

        let mut head_geometry = vec![0.0f32; 16 * 16 * 2];
        kernels::decode_geometry(
            &built.coefficients,
            built.history[639],
            &mean_basis,
            &mean_carry,
            &mut head_geometry,
        );
        let mut geometry = vec![0.0f32; 16 * 16];
        let mut pairs = vec![0.0f32; 16 * 20];
        let previous = if case % 2 == 1 {
            Some((built.previous.as_slice(), [0.5f32, -0.25]))
        } else {
            None
        };
        kernels::selector_inputs(&head_geometry, previous, &mut geometry, &mut pairs);
        engine.encoded.copy_from_slice(&build_encoded(case as u32));
        engine.choice(&geometry, &pairs, previous.is_some());
        worst_logits =
            worst_logits.max(relative_drift(&engine.logits, part(&logits, case, 16), "logits"));
    }

    eprintln!(
        "worst relative drift: encoded {worst_encoded:e}, coefficients {worst_coefficients:e}, \
         logits {worst_logits:e}, events {worst_events:e}"
    );
    assert!(worst_encoded < 1e-5, "encoded drifted {worst_encoded:e}");
    assert!(worst_coefficients < 1e-5, "coefficients drifted {worst_coefficients:e}");
    assert!(worst_logits < 1e-5, "logits drifted {worst_logits:e}");
    assert!(worst_events < 1e-5, "event heads drifted {worst_events:e}");
}
