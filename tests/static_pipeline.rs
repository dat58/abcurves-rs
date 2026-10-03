mod common;

use abcurves::static_planner::{BEvent, BTrigger, StaticPipeline};
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
                            .generate(
                                &prefix,
                                &profile,
                                fire.target_rel_at_b,
                                fire.target_radius,
                                fire.progress_center,
                                seed,
                                None,
                            )
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
                assert_eq!(event.index as i64, expected_index, "event {event_index} onset index");
                assert_eq!(event.threshold, stats[0], "event {event_index} threshold");
                assert_eq!(event.speed_median, stats[1], "event {event_index} median");
                assert_eq!(event.speed_mad, stats[2], "event {event_index} mad");
            }
        }
    }
}
