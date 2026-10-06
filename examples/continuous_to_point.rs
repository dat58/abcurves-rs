//! Acquire a target from a standing start and print every 1 ms report.
//!
//! This is the planner built for the job: the Continuous Planner starts
//! independently, drives to a target and settles on it. The Static Planner in
//! `static_to_point` cannot do this well, because it only ever learned to
//! generate the finish of a movement someone else began.
//!
//! The target is in common angular counts, the Continuous model's own frame.
//! Reports come out in native device counts under the bundled sensitivity.

#[path = "support/mod.rs"]
mod support;

use abcurves::continuous::{ContinuousPipeline, CountTransform, PipelineOptions};
use std::io::{self, BufRead, Write};

const DEFAULT_RADIUS: f64 = 18.0;
const DEFAULT_SEED: u64 = 2026;
const SETTLE_MS: i64 = 250;
const TIMEOUT_MS: i64 = 10_000;

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }

    print!("target as `x y [radius] [seed]` in common counts: ");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    let fields: Vec<&str> = line
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|field| !field.is_empty())
        .collect();
    let (x, y) = match (
        fields.first().and_then(|f| f.parse().ok()),
        fields.get(1).and_then(|f| f.parse().ok()),
    ) {
        (Some(x), Some(y)) => (x, y),
        _ => {
            eprintln!("need two finite numbers, for example: 420 -160");
            return Ok(());
        }
    };
    let radius: f64 = fields
        .get(2)
        .and_then(|f| f.parse().ok())
        .unwrap_or(DEFAULT_RADIUS);
    let seed: u64 = fields
        .get(3)
        .and_then(|f| f.parse().ok())
        .unwrap_or(DEFAULT_SEED);

    let radians_per_count = support::f64s("human_start", "radians_per_count")[0];
    let profile = support::pairs_i16("human_start", "profile_hardware");
    let options = PipelineOptions::new(CountTransform::new(radians_per_count, true)?)
        .seed(seed)
        .renderer_seed(seed)
        .initial_xy([0.0, 0.0]);
    let mut stream = ContinuousPipeline::load(&profile, options)?;
    stream.update_target([x, y], 0)?;

    println!("target ({x}, {y}) common counts, radius {radius}, seed {seed}");
    println!(
        "{:>6}  {:>4} {:>4}  {:>9} {:>9}  {:>7}",
        "ms", "dx", "dy", "x", "y", "gap"
    );

    let mut settled_for = 0i64;
    let mut silent = 0usize;
    let mut last = [0.0f64; 2];
    for tick in 1..=TIMEOUT_MS {
        let block = stream.advance(tick * 1000)?;
        let report = block.reports[0];
        last = block.xy[0];
        let gap = (last[0] - x).hypot(last[1] - y);
        if report == [0, 0] {
            silent += 1;
        }
        println!(
            "{tick:>6}  {:>4} {:>4}  {:>9.2} {:>9.2}  {gap:>7.2}",
            report[0], report[1], last[0], last[1]
        );
        settled_for = if gap <= radius { settled_for + 1 } else { 0 };
        if settled_for >= SETTLE_MS {
            println!(
                "\nsettled after {tick} ms ({silent} silent reports), {gap:.2} counts from target"
            );
            return Ok(());
        }
    }
    let gap = (last[0] - x).hypot(last[1] - y);
    println!("\nstill {gap:.2} counts away after {TIMEOUT_MS} ms ({silent} silent reports)");
    Ok(())
}
