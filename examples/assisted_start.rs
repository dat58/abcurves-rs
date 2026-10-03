//! Initialize from real human history, retaining the observed physical anchor.

#[path = "support/mod.rs"]
mod support;

use abcurves::continuous::{ContinuousPipeline, CountTransform, PipelineOptions, prepare_history};

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let raw = support::pairs_f64("human_start", "raw_common");
    let observed = support::f64s("human_start", "observed_xy");
    let target = support::f64s("human_start", "target_xy");
    let profile = support::pairs_i16("human_start", "profile_hardware");
    let radians_per_count = support::f64s("human_start", "radians_per_count")[0];

    let start = prepare_history(&raw, [observed[0], observed[1]])?;

    // The same causal displacement history can start at the actual observed
    // cursor. start.initial_xy also exposes the lagged filtered anchor for exact
    // training-representation studies; it is not the cursor's current position.
    let options = PipelineOptions::new(CountTransform::new(radians_per_count, true)?)
        .seed(2026)
        .renderer_seed(101)
        .initial_xy(start.observed_xy)
        .history(start.history.clone())
        .observed_xy(start.observed_xy);
    let mut stream = ContinuousPipeline::load(&profile, options)?;
    stream.update_target([target[0], target[1]], 0)?;

    let output = stream.advance(128_000)?;
    println!(
        "{} reports after a genuine human prefix",
        output.reports.len()
    );
    println!(
        "filtered anchor [{:.4}, {:.4}] trails the observed cursor [{:.4}, {:.4}]",
        start.initial_xy[0], start.initial_xy[1], start.observed_xy[0], start.observed_xy[1]
    );
    let rendered = stream.rendered_xy();
    println!(
        "rendered position: [{:.4}, {:.4}]",
        rendered[0], rendered[1]
    );
    Ok(())
}
