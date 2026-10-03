#![allow(clippy::needless_range_loop)]

mod common;

use abcurves::continuous::{ContinuousOptions, MovementRuntime, prepare_history};
use common::{models_root, read_f64, read_i64};

const FIXTURE: &str = "continuous_stream";

enum Step {
    Target([f64; 2], i64),
    Advance(i64),
}

fn scenario(index: usize, target_xy: [f64; 2]) -> Vec<Step> {
    match index {
        0 => vec![
            Step::Target([100.0, 30.0], 0),
            Step::Advance(32_000),
            Step::Target([125.0, 45.0], 40_000),
            Step::Advance(64_000),
            Step::Advance(1_000_000),
        ],
        1 => {
            let mut script = vec![Step::Target([800.0, -200.0], 0)];
            script.extend((1..=750).map(|step| Step::Advance(step * 8_000)));
            script.push(Step::Target([300.0, 500.0], 6_000_000));
            script.extend((1..=250).map(|step| Step::Advance(6_000_000 + step * 8_000)));
            script
        }
        2 => vec![
            Step::Advance(50_000),
            Step::Target([40.0, 40.0], 50_000),
            Step::Advance(2_000_000),
        ],
        _ => vec![
            Step::Target(target_xy, 0),
            Step::Advance(128_000),
            Step::Advance(1_000_000),
        ],
    }
}

fn run(mut runtime: MovementRuntime, script: &[Step]) -> (Vec<i64>, Vec<[f64; 2]>) {
    runtime.planner_mut().record_decisions = true;
    let mut times = Vec::new();
    let mut points = Vec::new();
    for step in script {
        match step {
            Step::Target(xy, at) => runtime.update_target(*xy, *at).unwrap(),
            Step::Advance(at) => {
                let block = runtime.advance(*at).unwrap();
                times.extend(block.time_us);
                points.extend(block.xy);
            }
        }
    }
    let decisions = runtime.planner().decisions.clone();
    DECISIONS.with(|slot| *slot.borrow_mut() = decisions);
    (times, points)
}

thread_local! {
    static DECISIONS: std::cell::RefCell<Vec<abcurves::continuous::Decision>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[test]
fn prepared_history_matches_the_reference() {
    let raw = read_f64(FIXTURE, "raw_common");
    let observed = read_f64(FIXTURE, "observed_xy");
    let expected_history = read_f64(FIXTURE, "filtered_history");
    let expected_anchor = read_f64(FIXTURE, "filtered_xy");
    let rows: Vec<[f64; 2]> = raw.chunks_exact(2).map(|pair| [pair[0], pair[1]]).collect();
    let start = prepare_history(&rows, [observed[0], observed[1]]).unwrap();
    let flat: Vec<f64> = start
        .history
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    assert_eq!(flat, expected_history);
    assert_eq!(start.initial_xy.to_vec(), expected_anchor);
    assert_eq!(start.observed_xy.to_vec(), observed);
}

#[test]
fn streams_match_the_reference_planner() {
    let directory = models_root().join("continuous");
    if !directory.join("weights.npz").is_file() {
        eprintln!("skipping: {} is absent", directory.display());
        return;
    }
    let raw = read_f64(FIXTURE, "raw_common");
    let observed = read_f64(FIXTURE, "observed_xy");
    let target = read_f64(FIXTURE, "target_xy");
    let rows: Vec<[f64; 2]> = raw.chunks_exact(2).map(|pair| [pair[0], pair[1]]).collect();
    let human = prepare_history(&rows, [observed[0], observed[1]]).unwrap();

    let seeds = [2026u64, 7, 23, 2026];
    let mut worst_rms = 0.0f64;
    let mut worst_gap = 0.0f64;

    for index in 0..4 {
        let mut options = ContinuousOptions::default().seed(seeds[index]);
        if index == 3 {
            options = options
                .initial_xy(human.observed_xy)
                .history(human.history.clone());
        }
        let runtime = options.load().unwrap();
        let (times, points) = run(runtime, &scenario(index, [target[0], target[1]]));

        let expected_time = read_i64(FIXTURE, &format!("s{index}_time"));
        let expected_xy = read_f64(FIXTURE, &format!("s{index}_xy"));
        let expected_mode = read_i64(FIXTURE, &format!("s{index}_mode"));
        let expected_head = read_i64(FIXTURE, &format!("s{index}_head"));
        let expected_at_ms = read_f64(FIXTURE, &format!("s{index}_at_ms"));

        assert_eq!(times, expected_time, "scenario {index} sample times");
        assert_eq!(
            points.len() * 2,
            expected_xy.len(),
            "scenario {index} sample count"
        );

        let mut squared = 0.0f64;
        for (step, point) in points.iter().enumerate() {
            for axis in 0..2 {
                let gap = (point[axis] - expected_xy[step * 2 + axis]).abs();
                squared += gap * gap;
                worst_gap = worst_gap.max(gap);
            }
        }
        worst_rms = worst_rms.max((squared / (points.len() * 2) as f64).sqrt());

        DECISIONS.with(|slot| {
            let decisions = slot.borrow();
            assert_eq!(
                decisions.len(),
                expected_mode.len(),
                "scenario {index} decisions"
            );
            for (step, decision) in decisions.iter().enumerate() {
                assert_eq!(
                    decision.at_ms, expected_at_ms[step],
                    "scenario {index} time {step}"
                );
                assert_eq!(
                    i64::from(decision.mode),
                    expected_mode[step],
                    "scenario {index} mode {step}"
                );
                if expected_mode[step] == 0 {
                    assert_eq!(
                        i64::from(decision.head),
                        expected_head[step],
                        "scenario {index} head {step}"
                    );
                }
            }
        });
    }

    eprintln!("position rms {worst_rms:e}, max gap {worst_gap:e} common counts");
    assert!(worst_rms < 1e-4, "position rms {worst_rms:e}");
    assert!(worst_gap < 1.5e-3, "maximum position gap {worst_gap:e}");
}
