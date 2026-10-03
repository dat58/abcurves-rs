//! Generate independently, then change the target at its actual receipt time.

#[path = "support/mod.rs"]
mod support;

use abcurves::ContinuousOptions;

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let mut movement = ContinuousOptions::default().seed(2026).load()?;

    movement.update_target([100.0, 30.0], 0)?;
    let first = movement.advance(32_000)?;
    movement.update_target([125.0, 45.0], 40_000)?;
    let next_part = movement.advance(64_000)?;

    println!(
        "first block closed at {} us",
        first.time_us[first.time_us.len() - 1]
    );
    println!("newly completed endpoints: {:?}", &next_part.time_us[..8]);

    let tail = movement.advance(1_000_000)?;
    let last = tail.xy[tail.xy.len() - 1];
    println!(
        "absolute position in common angular counts: [{:.4}, {:.4}]",
        last[0], last[1]
    );
    println!(
        "{} decisions, {} motor evaluations",
        movement.decisions(),
        movement.motor_evaluations()
    );

    // Keep this instance alive. Advance sample time as observations arrive; the
    // caller owns real-time pacing. Loading once avoids repeated preparation.
    Ok(())
}
