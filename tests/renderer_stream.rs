mod common;

use abcurves::renderer::{RendererModel, RendererProfile};
use common::{models_root, read_i16, renderer_script};

const FIXTURE: &str = "renderer";

#[test]
fn c_runtime_golden_vector_matches() {
    let path = models_root().join("renderer_global_h80.bin");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }
    let model = RendererModel::open(&path).unwrap();
    let context: Vec<[i16; 2]> = (0..256).map(|index| [(index & 1) as i16, 0]).collect();
    let profile = RendererProfile::prepare(&model, &context).unwrap();
    let mut stream = profile.begin_stream(123).unwrap();
    let expected = [
        [1, 0], [1, 1], [1, 0], [0, 0], [2, 2], [1, 0], [1, 0], [1, 1],
        [1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1],
    ];
    for (tick, want) in expected.iter().enumerate() {
        let report = stream.step(&model, [1.0, 0.5]).unwrap();
        assert_eq!(report, *want, "tick {tick}");
    }
}

#[test]
fn streams_match_the_c_runtime() {
    let path = models_root().join("renderer_global_h80.bin");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }
    let model = RendererModel::open(&path).unwrap();
    let human_flat = read_i16(FIXTURE, "human_profile");
    let human: Vec<[i16; 2]> = human_flat.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let synthetic: Vec<[i16; 2]> = (0..256).map(|index| [(index & 1) as i16, 0]).collect();
    let zeros = vec![[0i16; 2]; 256];

    let scenarios: Vec<(&Vec<[i16; 2]>, u64, Vec<[f32; 2]>)> = vec![
        (&synthetic, 123, vec![[1.0f32, 0.5]; 16]),
        (&human, 101, renderer_script(3001, 1500, 6.0)),
        (&human, 29, renderer_script(3002, 600, 6.0)),
        (&zeros, 11, renderer_script(3003, 800, 2000.0)),
        (&human, 2026, renderer_script(3004, 400, 2000.0)),
    ];

    for (index, (context, seed, script)) in scenarios.iter().enumerate() {
        let expected = read_i16(FIXTURE, &format!("r{index}_reports"));
        let profile = RendererProfile::prepare(&model, context).unwrap();
        let mut stream = profile.begin_stream(*seed).unwrap();
        for (tick, displacement) in script.iter().enumerate() {
            let report = stream.step(&model, *displacement).unwrap();
            assert_eq!(
                report,
                [expected[tick * 2], expected[tick * 2 + 1]],
                "scenario {index} tick {tick}"
            );
        }
    }
}

#[test]
fn accumulator_conserves_debt() {
    let path = models_root().join("renderer_global_h80.bin");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }
    let model = RendererModel::open(&path).unwrap();
    let zeros = vec![[0i16; 2]; 256];
    let profile = RendererProfile::prepare(&model, &zeros).unwrap();
    let directions: [(i32, i32); 8] = [
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
    ];
    for seed in [6u64, 11, 23] {
        for (dx, dy) in directions {
            let mut stream = profile.begin_stream(seed).unwrap();
            let x = dx * 32767 * 65536;
            let y = dy * 32767 * 65536;
            for _ in 0..2048 {
                let before = stream.accumulator_q16();
                let report = stream.step_q16(&model, x, y).unwrap();
                assert_ne!(report[0], i16::MIN);
                assert_ne!(report[1], i16::MIN);
                let after = stream.accumulator_q16();
                for axis in 0..2 {
                    let intent = if axis == 0 { x } else { y };
                    let emitted = i64::from(report[axis]) * 65536;
                    assert_eq!(
                        i64::from(before[axis]) + i64::from(intent) - emitted,
                        i64::from(after[axis]),
                        "debt conservation seed {seed} axis {axis}"
                    );
                    assert!(
                        after[axis].abs() <= 6 * 65536,
                        "residual debt {} seed {seed}",
                        after[axis]
                    );
                }
            }
        }
    }
}
