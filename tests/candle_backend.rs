#![cfg(feature = "candle")]

mod common;

use abcurves::continuous::ContinuousOptions;
use abcurves::nn::NeuralInference;
use common::{models_root, read_f64, read_i64};

const FIXTURE: &str = "continuous_stream";

#[test]
fn candle_backend_reproduces_the_native_policy() {
    let directory = models_root().join("continuous");
    if !directory.join("weights.npz").is_file() {
        eprintln!("skipping: {} is absent", directory.display());
        return;
    }
    let expected_time = read_i64(FIXTURE, "s1_time");
    let expected_xy = read_f64(FIXTURE, "s1_xy");
    let expected_mode = read_i64(FIXTURE, "s1_mode");
    let expected_head = read_i64(FIXTURE, "s1_head");

    let mut runtime = ContinuousOptions::default()
        .seed(7)
        .backend(NeuralInference::Candle)
        .load()
        .unwrap();
    assert_eq!(runtime.planner().backend(), NeuralInference::Candle);
    runtime.planner_mut().record_decisions = true;
    runtime.update_target([800.0, -200.0], 0).unwrap();

    let mut times = Vec::new();
    let mut points = Vec::new();
    for step in 1..=750i64 {
        let block = runtime.advance(step * 8_000).unwrap();
        times.extend(block.time_us);
        points.extend(block.xy);
    }
    runtime.update_target([300.0, 500.0], 6_000_000).unwrap();
    for step in 1..=250i64 {
        let block = runtime.advance(6_000_000 + step * 8_000).unwrap();
        times.extend(block.time_us);
        points.extend(block.xy);
    }

    assert_eq!(times, expected_time, "sample times");
    let decisions = runtime.planner().decisions.clone();
    assert_eq!(decisions.len(), expected_mode.len());
    for (step, decision) in decisions.iter().enumerate() {
        assert_eq!(i64::from(decision.mode), expected_mode[step], "mode {step}");
        if expected_mode[step] == 0 {
            assert_eq!(i64::from(decision.head), expected_head[step], "head {step}");
        }
    }

    let mut worst = 0.0f64;
    for (step, point) in points.iter().enumerate() {
        for axis in 0..2 {
            worst = worst.max((point[axis] - expected_xy[step * 2 + axis]).abs());
        }
    }
    eprintln!("candle maximum position gap {worst:e} common counts");
    assert!(worst < 1.5e-3, "candle position gap {worst:e}");
}

#[test]
fn candle_static_encoder_tracks_the_compiled_kernel() {
    use abcurves::static_planner::FastPlanner;
    use common::read_f32;

    let root = models_root();
    if !root.join("planner_seed7.pt").is_file() {
        eprintln!("skipping: {} is absent", root.display());
        return;
    }
    let path = root.join("planner_seed7.pt");
    let mut native = FastPlanner::open(&path, true).unwrap();
    let mut candle =
        FastPlanner::open_with(&path, true, NeuralInference::Candle).unwrap();
    assert_eq!(candle.backend(), NeuralInference::Candle);

    let prefixes = read_f32("static_planner", "prefixes");
    let lengths = read_i64("static_planner", "prefix_lengths");
    let targets = read_f64("static_planner", "targets");
    let radii = read_f64("static_planner", "radii");
    let progresses = read_f64("static_planner", "progresses");

    let mut cursor = 0usize;
    let mut worst = 0.0f32;
    let mut durations_matched = 0usize;
    for index in 0..lengths.len() {
        let length = lengths[index] as usize;
        let prefix: Vec<[f32; 2]> = prefixes[cursor..cursor + length * 2]
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();
        cursor += length * 2;
        let target = [targets[index * 2], targets[index * 2 + 1]];
        for head in 0..16 {
            let reference = native
                .plan(&prefix, target, radii[index], progresses[index], 0, Some(head))
                .unwrap();
            let actual = candle
                .plan(&prefix, target, radii[index], progresses[index], 0, Some(head))
                .unwrap();
            assert_eq!(actual.head, reference.head);
            assert_eq!(
                actual.duration_ms, reference.duration_ms,
                "duration for event {index} head {head}"
            );
            durations_matched += 1;
            for step in 0..reference.duration_ms {
                for axis in 0..2 {
                    worst = worst
                        .max((actual.smooth_dxdy[step][axis] - reference.smooth_dxdy[step][axis]).abs());
                }
            }
        }
    }
    eprintln!("candle static gap {worst:e} counts over {durations_matched} plans");
    assert!(worst < 1e-5, "candle static gap {worst:e}");
}
