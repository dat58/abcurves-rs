//! Prepare one reusable profile, then emit one raw report per millisecond.

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

    let mut trigger = BTrigger::recommended();
    trigger.arm([target_a[0], target_a[1]], radius)?;
    let mut moved = [0.0f64; 2];
    let mut handoff = None;
    for (index, delta) in raw.iter().enumerate() {
        moved[0] += f64::from(delta[0]);
        moved[1] += f64::from(delta[1]);
        let target_now = [target_a[0] - moved[0], target_a[1] - moved[1]];
        if let Some(BEvent::Fire(fire)) =
            trigger.push_tick(f64::from(delta[0]), f64::from(delta[1]), target_now, radius)?
        {
            handoff = Some((index, fire));
            break;
        }
    }
    let Some((index, fire)) = handoff else {
        println!("this movement has no eligible handoff");
        return Ok(());
    };

    let prefix: Vec<[f32; 2]> = raw[..=index]
        .iter()
        .map(|row| [f32::from(row[0]), f32::from(row[1])])
        .collect();

    let mut pipeline = StaticPipeline::from_pretrained(7, None, true)?;
    let profile = pipeline.prepare_renderer_profile(&profile_window)?;

    // Bind the exact geometry once the closed B bin is finalized. The planner
    // seed selects the head; the renderer seed selects the texture draw.
    let mut stream = pipeline.prepare(
        &prefix,
        &profile,
        fire.target_rel_at_b,
        fire.target_radius,
        fire.progress_center,
        2026,
        2026,
        None,
    )?;
    println!("head {} plans {} ms", stream.head, stream.duration_ms());

    let model = pipeline.model();
    let mut quiet = 0usize;
    while !stream.complete() {
        let report = stream.step(model)?;
        if report == [0, 0] {
            quiet += 1;
        }
        // Send (report[0], report[1]) to the caller's 1 kHz output layer.
    }
    println!(
        "{} ticks rendered, {quiet} of them silent",
        stream.duration_ms()
    );
    Ok(())
}
