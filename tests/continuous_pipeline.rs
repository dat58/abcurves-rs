mod common;

use abcurves::continuous::{ContinuousPipeline, CountTransform, PipelineOptions, prepare_history};
use common::{models_root, read_f64, read_i16, read_i64};

const FIXTURE: &str = "continuous_pipeline";
const STREAM: &str = "continuous_stream";

fn human_fixture() -> (Vec<[i16; 2]>, f64, [f64; 2], [f64; 2], Vec<[f64; 2]>) {
    let profile_flat = read_i16(STREAM, "profile_hardware");
    let profile: Vec<[i16; 2]> = profile_flat.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let radians = read_f64(STREAM, "radians_per_count")[0];
    let observed = read_f64(STREAM, "observed_xy");
    let target = read_f64(STREAM, "target_xy");
    let raw = read_f64(STREAM, "raw_common");
    let rows: Vec<[f64; 2]> = raw.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    (
        profile,
        radians,
        [observed[0], observed[1]],
        [target[0], target[1]],
        rows,
    )
}

fn compare(label: &str, prefix: &str, time: &[i64], xy: &[[f64; 2]], reports: &[[i16; 2]], rendered: &[[f64; 2]]) {
    let expected_time = read_i64(FIXTURE, &format!("{prefix}_time"));
    let expected_xy = read_f64(FIXTURE, &format!("{prefix}_xy"));
    let expected_reports = read_i16(FIXTURE, &format!("{prefix}_reports"));
    let expected_rendered = read_f64(FIXTURE, &format!("{prefix}_rendered"));
    assert_eq!(time, expected_time, "{label} times");
    assert_eq!(reports.len() * 2, expected_reports.len(), "{label} report count");
    for (step, report) in reports.iter().enumerate() {
        assert_eq!(
            *report,
            [expected_reports[step * 2], expected_reports[step * 2 + 1]],
            "{label} report {step}"
        );
    }
    let mut worst_plan = 0.0f64;
    let mut worst_rendered = 0.0f64;
    for step in 0..xy.len() {
        for axis in 0..2 {
            worst_plan = worst_plan.max((xy[step][axis] - expected_xy[step * 2 + axis]).abs());
            worst_rendered =
                worst_rendered.max((rendered[step][axis] - expected_rendered[step * 2 + axis]).abs());
        }
    }
    eprintln!("{label}: plan gap {worst_plan:e}, rendered gap {worst_rendered:e}");
    assert!(worst_plan < 1.5e-3, "{label} plan gap {worst_plan:e}");
    assert!(worst_rendered < 1e-9, "{label} rendered gap {worst_rendered:e}");
}

#[test]
fn composed_reports_match_the_reference_pipeline() {
    let directory = models_root();
    if !directory.join("continuous").join("weights.npz").is_file() {
        eprintln!("skipping: {} is absent", directory.display());
        return;
    }
    let (profile, radians, observed, target, raw) = human_fixture();
    let transform = CountTransform::new(radians, true).unwrap();

    let options = PipelineOptions::new(transform)
        .seed(2026)
        .renderer_seed(101)
        .initial_xy([0.0, 0.0]);
    let mut stream = ContinuousPipeline::load(&profile, options).unwrap();
    stream.update_target([100.0, 30.0], 0).unwrap();
    let mut time = Vec::new();
    let mut xy = Vec::new();
    let mut reports = Vec::new();
    let mut rendered = Vec::new();
    for tick in 1..=1000i64 {
        if tick == 501 {
            stream.update_target([160.0, -40.0], 500_000).unwrap();
        }
        let block = stream.advance(tick * 1000).unwrap();
        time.extend(block.time_us);
        xy.extend(block.xy);
        reports.extend(block.reports);
        rendered.extend(block.rendered_xy);
    }
    compare("streaming", "p0", &time, &xy, &reports, &rendered);
    let final_rendered = read_f64(FIXTURE, "p0_final");
    for axis in 0..2 {
        assert!((stream.rendered_xy()[axis] - final_rendered[axis]).abs() < 1e-9);
    }

    let human = prepare_history(&raw, observed).unwrap();
    let options = PipelineOptions::new(transform)
        .seed(2026)
        .renderer_seed(101)
        .initial_xy(human.observed_xy)
        .history(human.history.clone())
        .observed_xy(human.observed_xy);
    let mut assisted = ContinuousPipeline::load(&profile, options).unwrap();
    assisted.update_target(target, 0).unwrap();
    let first = assisted.advance(128_000).unwrap();
    let tail = assisted.advance(1_000_000).unwrap();
    let time: Vec<i64> = first.time_us.iter().chain(&tail.time_us).copied().collect();
    let xy: Vec<[f64; 2]> = first.xy.iter().chain(&tail.xy).copied().collect();
    let reports: Vec<[i16; 2]> = first.reports.iter().chain(&tail.reports).copied().collect();
    let rendered: Vec<[f64; 2]> = first
        .rendered_xy
        .iter()
        .chain(&tail.rendered_xy)
        .copied()
        .collect();
    compare("assisted", "p1", &time, &xy, &reports, &rendered);
}
