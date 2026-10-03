//! Observe a real A->B movement, then generate one complete B->C finish.

#[path = "support/mod.rs"]
mod support;

use abcurves::static_planner::{BEvent, BTrigger, StaticPipeline};

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let raw = support::pairs_i16("static_event", "raw_dxdy");
    let target_a = support::f64s("static_event", "target_rel_a");
    let radius = support::f64s("static_event", "target_radius")[0];
    let profile_window = support::pairs_i16("static_event", "profile_before_a");

    // A was audited in this recorded example. In live use the causal
    // OnsetDetector establishes A, then this trigger observes only the
    // completed bins since A.
    let mut trigger = BTrigger::recommended();
    trigger.arm([target_a[0], target_a[1]], radius)?;

    let mut moved = [0.0f64; 2];
    let mut handoff = None;
    for (index, delta) in raw.iter().enumerate() {
        moved[0] += f64::from(delta[0]);
        moved[1] += f64::from(delta[1]);
        let target_now = [target_a[0] - moved[0], target_a[1] - moved[1]];
        match trigger.push_tick(f64::from(delta[0]), f64::from(delta[1]), target_now, radius)? {
            Some(BEvent::Reject(reject)) => {
                println!("no eligible handoff: {}", reject.reason);
                return Ok(());
            }
            Some(BEvent::Fire(fire)) => {
                handoff = Some((index, fire));
                break;
            }
            None => {}
        }
    }
    let Some((index, fire)) = handoff else {
        println!("this movement has no eligible handoff");
        return Ok(());
    };
    println!(
        "B at {} ms, edge progress {:.4}, center progress {:.4}",
        fire.t_ms, fire.progress_edge, fire.progress_center
    );

    let prefix: Vec<[f32; 2]> = raw[..=index]
        .iter()
        .map(|row| [f32::from(row[0]), f32::from(row[1])])
        .collect();

    let mut pipeline = StaticPipeline::from_pretrained(7, None, true)?;
    // Do this before the latency-sensitive B handoff. Reuse the returned
    // immutable profile for later events from the same device/setup.
    let profile = pipeline.prepare_renderer_profile(&profile_window)?;
    let reports = pipeline.generate(
        &prefix,
        &profile,
        fire.target_rel_at_b,
        fire.target_radius,
        fire.progress_center,
        2026,
        None,
    )?;

    let travelled: [i32; 2] = reports.iter().fold([0, 0], |total, report| {
        [
            total[0] + i32::from(report[0]),
            total[1] + i32::from(report[1]),
        ]
    });
    println!(
        "{} integer reports, travelling {:?} counts",
        reports.len(),
        travelled
    );
    Ok(())
}
