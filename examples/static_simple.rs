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

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let radius = 5.0;
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

    println!("{:>8}  {:>7}  {:>6}  {}", "BC", "|BC|", "policy", "outcome");
    for component in [4.0f64, 6.0, 8.0, 12.0, 16.0, 20.0] {
        let movement = straight(component, radius);
        let remaining = movement.remaining();
        let magnitude = remaining[0].hypot(remaining[1]);
        let label = format!("({component:.0},{component:.0})");
        match pipeline.continue_movement(&movement, &profile, Draw::seeded(2026)) {
            Ok(reports) => {
                let travelled = reports.iter().fold([0i32; 2], |total, report| {
                    [
                        total[0] + i32::from(report[0]),
                        total[1] + i32::from(report[1]),
                    ]
                });
                let policy = movement.resolved_policy()?.threshold * 100.0;
                println!(
                    "{label:>8}  {magnitude:>7.2}  {policy:>5.0}%  {} reports, travelling {travelled:?}",
                    reports.len()
                );
            }
            Err(error) => println!("{label:>8}  {magnitude:>7.2}  {:>6}  refused: {error}", "-"),
        }
    }
    Ok(())
}
