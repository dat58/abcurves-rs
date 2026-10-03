mod common;

use abcurves::static_planner::FastPlanner;
use common::{models_root, read_f32, read_f64, read_i64};

const FIXTURE: &str = "static_planner";

struct Event {
    prefix: Vec<[f32; 2]>,
    target: [f64; 2],
    radius: f64,
    progress: f64,
}

fn events() -> Vec<Event> {
    let prefixes = read_f32(FIXTURE, "prefixes");
    let lengths = read_i64(FIXTURE, "prefix_lengths");
    let targets = read_f64(FIXTURE, "targets");
    let radii = read_f64(FIXTURE, "radii");
    let progresses = read_f64(FIXTURE, "progresses");
    let mut cursor = 0usize;
    lengths
        .iter()
        .enumerate()
        .map(|(index, &length)| {
            let length = length as usize;
            let prefix: Vec<[f32; 2]> = prefixes[cursor..cursor + length * 2]
                .chunks_exact(2)
                .map(|pair| [pair[0], pair[1]])
                .collect();
            cursor += length * 2;
            Event {
                prefix,
                target: [targets[index * 2], targets[index * 2 + 1]],
                radius: radii[index],
                progress: progresses[index],
            }
        })
        .collect()
}

#[test]
fn summary_and_plans_match_the_reference() {
    let root = models_root();
    if !root.join("planner_seed7.pt").is_file() {
        eprintln!("skipping: {} is absent", root.display());
        return;
    }
    let events = events();
    let raw_summaries = read_f64(FIXTURE, "raw_summaries");
    let vectors = read_f32(FIXTURE, "vectors");
    let durations = read_i64(FIXTURE, "durations");
    let heads = read_i64(FIXTURE, "heads");
    let smooth = read_f32(FIXTURE, "smooth");

    let mut plan_index = 0usize;
    let mut smooth_cursor = 0usize;
    let mut worst_summary = 0.0f64;
    let mut worst_vector = 0.0f32;
    let mut worst_smooth = 0.0f32;

    for (model_index, model_seed) in [7u32, 23].into_iter().enumerate() {
        let mut planner = FastPlanner::from_pretrained(model_seed, Some(&root), true).unwrap();
        for (event_index, event) in events.iter().enumerate() {
            if model_index == 0 {
                let start = event.prefix.len().saturating_sub(160);
                let raw = planner.raw_summary(
                    &event.prefix[start..],
                    event.target,
                    event.radius,
                    event.progress,
                );
                for slot in 0..62 {
                    worst_summary =
                        worst_summary.max((raw[slot] - raw_summaries[event_index * 62 + slot]).abs());
                }
                let mut vector = vec![0.0f32; 62];
                planner.normalizer().apply(&raw, &mut vector);
                for slot in 0..62 {
                    worst_vector =
                        worst_vector.max((vector[slot] - vectors[event_index * 62 + slot]).abs());
                }
            }

            let check = |intent: abcurves::static_planner::Intent,
                             plan_index: &mut usize,
                             smooth_cursor: &mut usize,
                             worst_smooth: &mut f32| {
                assert_eq!(
                    intent.duration_ms as i64, durations[*plan_index],
                    "duration for plan {}", *plan_index
                );
                assert_eq!(intent.head as i64, heads[*plan_index], "head for plan {}", *plan_index);
                for step in 0..intent.duration_ms {
                    for axis in 0..2 {
                        *worst_smooth = worst_smooth.max(
                            (intent.smooth_dxdy[step][axis]
                                - smooth[*smooth_cursor + step * 2 + axis])
                                .abs(),
                        );
                    }
                }
                *smooth_cursor += intent.duration_ms * 2;
                *plan_index += 1;
            };

            for head in 0..16 {
                let intent = planner
                    .plan(&event.prefix, event.target, event.radius, event.progress, 2026, Some(head))
                    .unwrap();
                check(intent, &mut plan_index, &mut smooth_cursor, &mut worst_smooth);
            }
            for seed in [0u128, 7, 23, 2026, 12345] {
                let intent = planner
                    .plan(&event.prefix, event.target, event.radius, event.progress, seed, None)
                    .unwrap();
                check(intent, &mut plan_index, &mut smooth_cursor, &mut worst_smooth);
            }
        }
    }

    eprintln!(
        "summary gap {worst_summary:e}, normalized gap {worst_vector:e}, smooth gap {worst_smooth:e}"
    );
    assert_eq!(plan_index, durations.len());
    assert!(worst_summary == 0.0, "raw summary drifted {worst_summary:e}");
    assert!(worst_vector == 0.0, "normalized summary drifted {worst_vector:e}");
    assert!(worst_smooth < 1e-5, "smooth deltas drifted {worst_smooth:e}");
}
