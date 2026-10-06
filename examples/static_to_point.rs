//! Move from a standing start to a target typed on the keyboard, printing every
//! 1 ms report the Renderer emits.
//!
//! A and B coincide at the origin here, so there is no observed human movement
//! before the handoff. That is a deliberate cold start and it sits outside the
//! regime the Static Planner was trained for: the model expects a real A to B
//! prefix carrying the current speed, direction and correction pattern, cut at
//! 78 to 92 percent of the way to the target edge. With a standing start it gets
//! zero speed and zero progress and must extrapolate. It still produces a finite
//! plan, but it lands badly: measured against four targets it stopped 43 to 80
//! percent of the way there, between 56 and 169 counts short.
//!
//!     target        landed        gap
//!     (420, -160)   (334, -112)   98.49
//!     (100, 30)     (43, 7)       61.47
//!     (800, 0)      (639, -51)    168.88
//!     (60, 60)      (28, 14)      56.04
//!
//! That is the expected failure. The model only ever learned to generate the
//! finish of a movement someone else began, at 78 to 92 percent of the way to
//! the target edge, so a full-distance request is far outside what it saw.
//!
//! Use `static_quickstart` for the shape this planner is for, or
//! `continuous_to_point` to actually acquire a target from a standing start.
//!
//! Coordinates are canonical native counts with X right and Y up. Flip Y if the
//! consumer uses screen convention.

#[path = "support/mod.rs"]
mod support;

use abcurves::static_planner::StaticPipeline;
use std::io::{self, BufRead, Write};

const PREFIX_TICKS: usize = 160;
const DEFAULT_RADIUS: f64 = 18.0;
const DEFAULT_SEED: u64 = 2026;

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }

    print!("target as `x y [radius] [seed]`: ");
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
    if !(radius.is_finite() && radius > 0.0) {
        eprintln!("radius must be positive");
        return Ok(());
    }

    // A standing start: the observed window is real but completely still.
    let prefix = vec![[0.0f32, 0.0]; PREFIX_TICKS];
    // B is still at A, so nothing has been travelled towards the target yet.
    let progress_center = 0.0;

    let mut pipeline = StaticPipeline::from_pretrained(7, None, true)?;
    // A genuine 256-report sample from a real device supplies the packet texture.
    let profile = pipeline
        .prepare_renderer_profile(&support::pairs_i16("static_event", "profile_before_a"))?;

    let mut stream = pipeline.prepare(
        &prefix,
        &profile,
        [x, y],
        radius,
        progress_center,
        u128::from(seed),
        seed,
        None,
    )?;

    println!(
        "target ({x}, {y}) counts, radius {radius}, seed {seed} -> head {}, {} ms",
        stream.head,
        stream.duration_ms()
    );
    println!(
        "{:>5}  {:>4} {:>4}  {:>8} {:>8}",
        "tick", "dx", "dy", "x", "y"
    );

    let model = pipeline.model();
    let mut position = [0i32; 2];
    let mut silent = 0usize;
    let mut tick = 0usize;
    while !stream.complete() {
        let report = stream.step(model)?;
        tick += 1;
        position[0] += i32::from(report[0]);
        position[1] += i32::from(report[1]);
        if report == [0, 0] {
            silent += 1;
        }
        println!(
            "{tick:>5}  {:>4} {:>4}  {:>8} {:>8}",
            report[0], report[1], position[0], position[1]
        );
    }

    let gap = (f64::from(position[0]) - x).hypot(f64::from(position[1]) - y);
    println!(
        "\n{tick} reports, {silent} silent, landed at ({}, {}) -> {gap:.2} counts from target, radius {radius}",
        position[0], position[1]
    );
    Ok(())
}
