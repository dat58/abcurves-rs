//! Continue a movement from A, the target, the radius and the reports so far.
//!
//! This is the short form of the static handoff. B is the last report supplied,
//! because A to B is motion the mouse has already made and cannot be chosen
//! after the fact. The library validates that point and refuses, with a reason,
//! when it is not a usable handoff.
//!
//! The sweep uses a synthetic straight approach so BC can be set exactly; see
//! `static_quickstart` for the same call driven by a recorded movement.

#[path = "support/mod.rs"]
mod support;

use abcurves::static_planner::{Draw, Movement, Policy, StaticPipeline};

fn straight(component: f64, radius: f64) -> Movement {
    // Stop at 90 percent of the way to the centre, leaving BC = (c, c).
    let far = 10.0 * component;
    let step = ((9.0 * component) / 100.0) as f32;
    Movement::from_counts([0.0, 0.0], [far, far], radius, &vec![[step, step]; 100])
}

fn motion_end(reports: &[[i16; 2]]) -> usize {
    reports
        .iter()
        .rposition(|report| report != &[0, 0])
        .map_or(0, |index| index + 1)
}

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let radius = 5.0;
    let seed = 2026;
    let mut pipeline = StaticPipeline::from_pretrained(7, None, true)?;
    let profile = pipeline
        .prepare_renderer_profile(&support::pairs_i16("static_event", "profile_before_a"))?;

    if let Some(window) =
        Movement::eligible_remaining([0.0, 0.0], [100.0, 100.0], radius, Policy::Auto)
    {
        println!(
            "for A (0,0) -> target (100,100) with radius {radius}, |BC| must be \
             {:.2} to {:.2} counts\n",
            window.start(),
            window.end()
        );
    }

    println!("{:>8}  {:>7}  {:>6}  outcome", "BC", "|BC|", "policy");
    for component in [4.0f64, 6.0, 8.0, 12.0, 16.0, 20.0] {
        let movement = straight(component, radius);
        let remaining = movement.remaining();
        let magnitude = remaining[0].hypot(remaining[1]);
        let label = format!("({component:.0},{component:.0})");
        match pipeline.continue_movement(&movement, &profile, Draw::seeded(seed)) {
            Ok(reports) => {
                let travelled = reports.iter().fold([0i32; 2], |total, report| {
                    [
                        total[0] + i32::from(report[0]),
                        total[1] + i32::from(report[1]),
                    ]
                });
                let moving = reports.iter().filter(|report| *report != &[0, 0]).count();
                let policy = movement.resolved_policy()?.threshold * 100.0;
                println!(
                    "{label:>8}  {magnitude:>7.2}  {policy:>5.0}%  {} reports, {moving} carry \
                     motion, last at {} ms, travelling {travelled:?}",
                    reports.len(),
                    motion_end(&reports)
                );
            }
            Err(error) => println!("{label:>8}  {magnitude:>7.2}  {:>6}  refused: {error}", "-"),
        }
    }

    // The duration is the finishing style, not the distance: at one BC the
    // sixteen heads span an order of magnitude, and a seed only picks among
    // them. Seed 2026 happens to draw head 13, the longest dweller of the
    // sixteen, so it is the slowest row above rather than a typical one.
    let movement = straight(6.0, radius);
    let handoff = movement.handoff()?;
    let mut durations = Vec::new();
    for head in 0..16 {
        let stream = pipeline.prepare(
            movement.prefix(),
            &profile,
            handoff,
            Draw::seeded(seed).head(head),
        )?;
        durations.push((head, stream.duration_ms()));
    }
    let drawn = pipeline
        .prepare(movement.prefix(), &profile, handoff, Draw::seeded(seed))?
        .head;
    let mut sorted: Vec<usize> = durations.iter().map(|entry| entry.1).collect();
    sorted.sort_unstable();
    let fastest = durations.iter().min_by_key(|entry| entry.1).unwrap();
    let slowest = durations.iter().max_by_key(|entry| entry.1).unwrap();
    println!(
        "\nat BC (6,6) the sixteen heads plan {} ms (head {}) to {} ms (head {}), median {} ms;\n\
         seed {seed} draws head {drawn}",
        fastest.1,
        fastest.0,
        slowest.1,
        slowest.0,
        (sorted[7] + sorted[8]) / 2
    );
    Ok(())
}
