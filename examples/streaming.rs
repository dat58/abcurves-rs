//! Compose the Continuous Planner with a persistent hardware-count Renderer.

#[path = "support/mod.rs"]
mod support;

use abcurves::continuous::{ContinuousPipeline, CountTransform, PipelineOptions};

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    // An authentic sample is bundled with its source lineage. In an application,
    // provide exactly 256 genuine 1 ms reports representative of its device/setup.
    let profile = support::pairs_i16("human_start", "profile_hardware");
    let radians_per_count = support::f64s("human_start", "radians_per_count")[0];
    let transform = CountTransform::new(radians_per_count, true)?;

    let options = PipelineOptions::new(transform)
        .seed(2026)
        .renderer_seed(101)
        .initial_xy([0.0, 0.0]);
    let mut stream = ContinuousPipeline::load(&profile, options)?;
    stream.update_target([100.0, 30.0], 0)?;

    let mut emitted = 0usize;
    for tick in 1..=1000i64 {
        if tick == 501 {
            stream.update_target([160.0, -40.0], 500_000)?;
        }
        let output = stream.advance(tick * 1000)?;
        let report = output.reports[0];
        if report != [0, 0] {
            emitted += 1;
        }
        // Hand (report[0], report[1]) to your paced output layer here.
    }

    let rendered = stream.rendered_xy();
    println!("1,000 reports, {emitted} of them nonzero");
    println!(
        "rendered position: [{:.4}, {:.4}]",
        rendered[0], rendered[1]
    );
    Ok(())
}
