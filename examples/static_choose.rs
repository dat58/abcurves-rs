//! Plan all sixteen finishes for one handoff, then render only the chosen one.
//!
//! The heads share the whole trunk, so planning all sixteen costs barely more
//! than planning one. Rendering is the expensive half, so the choice is made
//! from the plans and the renderer runs once.

#[path = "support/mod.rs"]
mod support;

use abcurves::static_planner::{Movement, StaticPipeline};
use std::time::Instant;

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let raw = support::screen_pairs_i16("static_event", "raw_dxdy");
    let target_a = support::screen_xy(&support::f64s("static_event", "target_rel_a"));
    let radius = support::f64s("static_event", "target_radius")[0];
    let profile_window = support::screen_pairs_i16("static_event", "profile_before_a");
    let b_index = 168usize;
    let seed = 2026;

    let movement = Movement::from_reports(
        [0.0, 0.0],
        [target_a[0], target_a[1]],
        radius,
        &raw[..=b_index],
    );

    let mut pipeline = StaticPipeline::from_pretrained(7, None, true)?;
    let profile = pipeline.prepare_renderer_profile(&profile_window)?;

    let mut timings = Vec::new();
    let mut plans = Vec::new();
    for _ in 0..21 {
        let start = Instant::now();
        plans = pipeline.plan_movement_all(&movement)?;
        timings.push(start.elapsed());
    }
    timings.sort_unstable();
    let planned = timings[timings.len() / 2];

    println!(
        "sixteen heads planned in {:.0} us",
        planned.as_secs_f64() * 1e6
    );
    for plan in &plans {
        let reach = plan
            .smooth_dxdy
            .iter()
            .scan([0.0f64; 2], |total, step| {
                total[0] += f64::from(step[0]);
                total[1] += f64::from(step[1]);
                Some(total[0].hypot(total[1]))
            })
            .collect::<Vec<f64>>();
        let final_distance = *reach.last().unwrap_or(&0.0);
        let travelled = reach
            .iter()
            .position(|value| *value >= 0.95 * final_distance)
            .map_or(0, |index| index + 1);
        println!(
            "  head {:>2}  {:>4} ms, 95% of {:.1} counts travelled by {:>3} ms",
            plan.head, plan.duration_ms, final_distance, travelled
        );
    }

    // Choosing by hand is a bias: the sixteen are the model's distribution of
    // human finishes, and a seed draws from it uniformly. Pick deliberately
    // only when a deadline makes it necessary.
    let budget_ms = 150;
    let chosen = plans
        .iter()
        .filter(|plan| plan.duration_ms <= budget_ms)
        .max_by_key(|plan| plan.duration_ms)
        .map_or(0, |plan| plan.head);
    let intent = plans.into_iter().nth(chosen).unwrap();
    println!(
        "\nunder a {budget_ms} ms budget: head {} at {} ms",
        intent.head, intent.duration_ms
    );

    let start = Instant::now();
    let mut stream = pipeline.begin(&profile, intent, seed)?;
    let reports = stream.render_remaining(pipeline.model())?;
    let rendered = start.elapsed();
    let moving = reports.iter().filter(|report| *report != &[0, 0]).count();
    println!(
        "rendered {} reports in {:.0} us, {moving} carry motion",
        reports.len(),
        rendered.as_secs_f64() * 1e6
    );
    Ok(())
}
