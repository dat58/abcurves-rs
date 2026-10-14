#![allow(clippy::needless_range_loop)]

mod common;

use abcurves::static_planner::{BEvent, BTrigger, Draw, Handoff, StaticPipeline};
use common::{models_root, read_f64, read_i16, read_i64};

const FIXTURE: &str = "static_pipeline";

#[test]
fn seam_trigger_and_continuations_match_the_reference() {
    let root = models_root();
    if !root.join("planner_seed7.pt").is_file() {
        eprintln!("skipping: {} is absent", root.display());
        return;
    }
    let raw_flat = read_i16(FIXTURE, "raw");
    let lengths = read_i64(FIXTURE, "raw_lengths");
    let profiles_flat = read_i16(FIXTURE, "profiles");
    let trigger_t = read_i64(FIXTURE, "trigger_t");
    let trigger_edge = read_f64(FIXTURE, "trigger_edge");
    let trigger_center = read_f64(FIXTURE, "trigger_center");
    let trigger_reason = read_i64(FIXTURE, "trigger_reason");
    let fire_target = read_f64(FIXTURE, "fire_target");
    let fire_radius = read_f64(FIXTURE, "fire_radius");
    let report_counts = read_i64(FIXTURE, "report_counts");
    let reports = read_i16(FIXTURE, "reports");

    // The fixture's armed geometry comes from the recorded events; the trigger
    // is re-armed here from the same target vector and radius.
    let targets: Vec<([f64; 2], f64)> = (0..lengths.len())
        .map(|index| {
            (
                [fire_target[index * 2], fire_target[index * 2 + 1]],
                fire_radius[index],
            )
        })
        .collect();
    let _ = &targets;

    let mut raw_cursor = 0usize;
    let mut continuation = 0usize;
    let mut report_cursor = 0usize;

    for event_index in 0..lengths.len() {
        let length = lengths[event_index] as usize;
        let raw: Vec<[i16; 2]> = raw_flat[raw_cursor..raw_cursor + length * 2]
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();
        raw_cursor += length * 2;
        let profile_window: Vec<[i16; 2]> = profiles_flat
            [event_index * 512..(event_index + 1) * 512]
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();

        let target_a = read_f64("static_pipeline_arm", &format!("target_a_{event_index}"));
        let radius = read_f64("static_pipeline_arm", &format!("radius_{event_index}"))[0];

        let mut trigger = BTrigger::recommended();
        trigger.arm([target_a[0], target_a[1]], radius).unwrap();
        let mut running = [0.0f64; 2];
        let mut fired = None;
        for (tick, delta) in raw.iter().enumerate() {
            running[0] += f64::from(delta[0]);
            running[1] += f64::from(delta[1]);
            let target_now = [target_a[0] - running[0], target_a[1] - running[1]];
            let event = trigger
                .push_tick(f64::from(delta[0]), f64::from(delta[1]), target_now, radius)
                .unwrap();
            if let Some(event) = event {
                fired = Some((tick, event));
                break;
            }
        }
        let (tick, event) = fired.expect("trigger produced no event");
        match event {
            BEvent::Reject(reject) => {
                assert_eq!(trigger_reason[event_index], 1, "event {event_index} kind");
                assert_eq!(reject.t_ms as i64, trigger_t[event_index]);
                assert_eq!(reject.progress_center, trigger_center[event_index]);
                continue;
            }
            BEvent::Fire(fire) => {
                assert_eq!(trigger_reason[event_index], 0, "event {event_index} kind");
                assert_eq!(fire.t_ms as i64, trigger_t[event_index]);
                assert_eq!(fire.progress_edge, trigger_edge[event_index]);
                assert_eq!(fire.progress_center, trigger_center[event_index]);
                assert_eq!(fire.target_rel_at_b, targets[event_index].0);
                assert_eq!(fire.target_radius, targets[event_index].1);

                let prefix: Vec<[f32; 2]> = raw[..tick + 1]
                    .iter()
                    .map(|row| [f32::from(row[0]), f32::from(row[1])])
                    .collect();
                for model_seed in [7u32, 23] {
                    let mut pipeline =
                        StaticPipeline::from_pretrained(model_seed, Some(&root), true).unwrap();
                    let profile = pipeline.prepare_renderer_profile(&profile_window).unwrap();
                    for seed in [7u64, 2026, 12345] {
                        let out = pipeline
                            .generate(&prefix, &profile, Handoff::from(&fire), Draw::seeded(seed))
                            .unwrap();
                        let expected = report_counts[continuation] as usize;
                        assert_eq!(out.len(), expected, "continuation {continuation} length");
                        for (step, report) in out.iter().enumerate() {
                            assert_eq!(
                                *report,
                                [
                                    reports[report_cursor + step * 2],
                                    reports[report_cursor + step * 2 + 1]
                                ],
                                "continuation {continuation} report {step}"
                            );
                        }
                        report_cursor += expected * 2;
                        continuation += 1;
                    }
                }
            }
        }
    }
    assert_eq!(continuation, report_counts.len());
}

#[test]
fn onset_detector_matches_the_reference() {
    let raw_flat = read_i16(FIXTURE, "raw");
    let lengths = read_i64(FIXTURE, "raw_lengths");
    let mut cursor = 0usize;
    for event_index in 0..lengths.len() {
        let length = lengths[event_index] as usize;
        let raw: Vec<[i16; 2]> = raw_flat[cursor..cursor + length * 2]
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();
        cursor += length * 2;
        let target_a = read_f64("static_pipeline_arm", &format!("target_a_{event_index}"));
        let expected_index = read_i64("static_pipeline_arm", &format!("onset_{event_index}"))[0];
        let stats = read_f64("static_pipeline_arm", &format!("onset_stats_{event_index}"));

        let mut detector = abcurves::static_planner::OnsetDetector::new(Default::default());
        let mut running = [0.0f64; 2];
        let mut fired = None;
        for delta in &raw {
            let event = detector
                .push(
                    f64::from(delta[0]),
                    f64::from(delta[1]),
                    Some([target_a[0] - running[0], target_a[1] - running[1]]),
                )
                .unwrap();
            running[0] += f64::from(delta[0]);
            running[1] += f64::from(delta[1]);
            if let Some(event) = event {
                fired = Some(event);
                break;
            }
        }
        match fired {
            None => assert_eq!(expected_index, -1, "event {event_index} onset"),
            Some(event) => {
                assert_eq!(
                    event.index as i64, expected_index,
                    "event {event_index} onset index"
                );
                assert_eq!(event.threshold, stats[0], "event {event_index} threshold");
                assert_eq!(event.speed_median, stats[1], "event {event_index} median");
                assert_eq!(event.speed_mad, stats[2], "event {event_index} mad");
            }
        }
    }
}

fn straight_movement(component: f64, radius: f64) -> abcurves::static_planner::Movement {
    // A straight run that stops with BC = (component, component), at 90 percent
    // of the way to the centre so `max_center_progress` is satisfied.
    let far = 10.0 * component;
    let travelled = 9.0 * component;
    let ticks = 100usize;
    let step = (travelled / ticks as f64) as f32;
    abcurves::static_planner::Movement::from_counts(
        [0.0, 0.0],
        [far, far],
        radius,
        &vec![[step, step]; ticks],
    )
}

#[test]
fn short_continuations_follow_the_seam_floor() {
    use abcurves::static_planner::Movement;

    // |BC| = component * sqrt(2). The contract floor is 8 counts.
    for radius in [2.0f64, 5.0] {
        for component in [1.0f64, 5.0] {
            let movement = straight_movement(component, radius);
            let error = movement.handoff().unwrap_err().to_string();
            assert!(
                error.contains("cannot be planned under any"),
                "|BC| {:.2} radius {radius} should be refused outright: {error}",
                component * 2.0f64.sqrt()
            );
        }
        // 8.49 counts clears the floor but not the 90 percent edge margin.
        let movement = straight_movement(6.0, radius);
        let config = movement.resolved_policy().unwrap_or_else(|error| {
            panic!("|BC| 8.49 radius {radius} should be planable: {error}")
        });
        assert_eq!(
            config.threshold, 0.8,
            "radius {radius} should fall back to the contract"
        );

        for component in [10.0f64, 20.0] {
            let movement = straight_movement(component, radius);
            let config = movement.resolved_policy().unwrap();
            assert_eq!(
                config.threshold,
                0.9,
                "|BC| {:.2} radius {radius} should keep the recommendation",
                component * 2.0f64.sqrt()
            );
        }
    }

    // A geometry with no usable window at all.
    assert!(
        Movement::eligible_remaining([0.0, 0.0], [0.0, 0.0], 5.0, Default::default()).is_none()
    );
}

#[test]
fn eligible_window_agrees_with_the_gate() {
    use abcurves::static_planner::{Movement, Policy};

    let target = [200.0, 0.0];
    let radius = 5.0;
    let window =
        Movement::eligible_remaining([0.0, 0.0], target, radius, Policy::Contract).unwrap();
    for remaining in [10.0f64, 20.0, 40.0, 60.0] {
        let step = ((target[0] - remaining) / 100.0) as f32;
        let movement = Movement::from_counts([0.0, 0.0], target, radius, &vec![[step, 0.0]; 100])
            .with_policy(Policy::Contract);
        let inside = window.contains(&remaining);
        assert_eq!(
            movement.handoff().is_ok(),
            inside,
            "|BC| {remaining} against window {:.2}..={:.2}",
            window.start(),
            window.end()
        );
    }
}

#[test]
fn convenience_path_equals_the_explicit_path() {
    use abcurves::static_planner::{Draw, Handoff, Movement, StaticPipeline};

    let root = models_root();
    if !root.join("planner_seed7.pt").is_file() {
        eprintln!("skipping: {} is absent", root.display());
        return;
    }
    let raw_flat = read_i16(FIXTURE, "raw");
    let lengths = read_i64(FIXTURE, "raw_lengths");
    let raw: Vec<[i16; 2]> = raw_flat[..lengths[0] as usize * 2]
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect();
    let target_a = read_f64("static_pipeline_arm", "target_a_0");
    let radius = read_f64("static_pipeline_arm", "radius_0")[0];
    let ticks = read_i64(FIXTURE, "trigger_t")[0] as usize;
    let fire_target = read_f64(FIXTURE, "fire_target");
    let fire_radius = read_f64(FIXTURE, "fire_radius")[0];
    let centre = read_f64(FIXTURE, "trigger_center")[0];

    let movement = Movement::from_reports(
        [0.0, 0.0],
        [target_a[0], target_a[1]],
        radius,
        &raw[..ticks],
    );
    let derived = movement.handoff().unwrap();
    let expected = Handoff::new([fire_target[0], fire_target[1]], fire_radius, centre).unwrap();
    for axis in 0..2 {
        assert!((derived.target_rel_at_b[axis] - expected.target_rel_at_b[axis]).abs() < 1e-9);
    }
    assert_eq!(derived.target_radius, expected.target_radius);
    assert!((derived.progress_center - expected.progress_center).abs() < 1e-9);

    let profiles = read_i16(FIXTURE, "profiles");
    let profile_window: Vec<[i16; 2]> = profiles[..512]
        .chunks_exact(2)
        .map(|p| [p[0], p[1]])
        .collect();
    let report_counts = read_i64(FIXTURE, "report_counts");
    let reports = read_i16(FIXTURE, "reports");

    let mut pipeline = StaticPipeline::from_pretrained(7, Some(&root), true).unwrap();
    let profile = pipeline.prepare_renderer_profile(&profile_window).unwrap();
    let mut cursor = 0usize;
    for (index, seed) in [7u64, 2026, 12345].into_iter().enumerate() {
        let out = pipeline
            .continue_movement(&movement, &profile, Draw::seeded(seed))
            .unwrap();
        let expected_len = report_counts[index] as usize;
        assert_eq!(out.len(), expected_len, "seed {seed} length");
        for (step, report) in out.iter().enumerate() {
            assert_eq!(
                *report,
                [reports[cursor + step * 2], reports[cursor + step * 2 + 1]],
                "seed {seed} report {step}"
            );
        }
        cursor += expected_len * 2;
    }
}

#[test]
fn ineligible_handoffs_name_their_rule() {
    use abcurves::static_planner::Movement;

    let raw_flat = read_i16(FIXTURE, "raw");
    let lengths = read_i64(FIXTURE, "raw_lengths");
    let raw: Vec<[i16; 2]> = raw_flat[..lengths[0] as usize * 2]
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect();
    let target_a = read_f64("static_pipeline_arm", "target_a_0");
    let radius = read_f64("static_pipeline_arm", "radius_0")[0];
    let target = [target_a[0], target_a[1]];

    let empty = Movement::from_reports([0.0, 0.0], target, radius, &[]);
    assert!(empty.handoff().is_err(), "an empty prefix has no handoff");

    let early = Movement::from_reports([0.0, 0.0], target, radius, &raw[..40]);
    let error = early.handoff().unwrap_err().to_string();
    assert!(error.contains("edge progress"), "cut at 40 ms: {error}");

    let late = Movement::from_reports([0.0, 0.0], target, radius, &raw);
    let error = late.handoff().unwrap_err().to_string();
    assert!(
        error.contains("inside the target"),
        "cut at the end: {error}"
    );
}

#[test]
fn rendering_every_head_matches_rendering_one() {
    use abcurves::static_planner::Movement;

    let root = models_root();
    if !root.join("planner_seed7.pt").is_file() {
        eprintln!("skipping: {} is absent", root.display());
        return;
    }
    let raw_flat = read_i16(FIXTURE, "raw");
    let lengths = read_i64(FIXTURE, "raw_lengths");
    let raw: Vec<[i16; 2]> = raw_flat[..lengths[0] as usize * 2]
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect();
    let target_a = read_f64("static_pipeline_arm", "target_a_0");
    let radius = read_f64("static_pipeline_arm", "radius_0")[0];
    let ticks = read_i64(FIXTURE, "trigger_t")[0] as usize;
    let movement = Movement::from_reports(
        [0.0, 0.0],
        [target_a[0], target_a[1]],
        radius,
        &raw[..ticks],
    );
    let profiles = read_i16(FIXTURE, "profiles");
    let profile_window: Vec<[i16; 2]> = profiles[..512]
        .chunks_exact(2)
        .map(|p| [p[0], p[1]])
        .collect();

    let mut pipeline = StaticPipeline::from_pretrained(7, Some(&root), true).unwrap();
    let profile = pipeline.prepare_renderer_profile(&profile_window).unwrap();
    let batch = pipeline
        .continue_movement_all(&movement, &profile, Draw::seeded(2026))
        .unwrap();
    assert_eq!(batch.len(), 16);
    for head in 0..16 {
        let single = pipeline
            .continue_movement(&movement, &profile, Draw::seeded(2026).head(head))
            .unwrap();
        assert_eq!(batch[head], single, "head {head}");
    }
}
