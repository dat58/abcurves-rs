//! Drive the Continuous Planner from a real-time 1 kHz output loop.
//!
//! `advance(t)` is a *sample clock*, not a wall clock. It commits every 1 ms
//! sample up to `t` and returns the ones it just completed, so the caller owns
//! pacing entirely. Two consequences shape the loop below:
//!
//! * Asking for `now` leaves no slack. A tick that re-runs the motor network
//!   costs about 0.9 ms of the 1 ms available; the other ticks cost about
//!   22 us. Advancing a few milliseconds *ahead* of the output clock puts that
//!   spike into a buffer instead of into a missed report.
//! * Lookahead is paid for in freshness. `update_target` refuses a timestamp
//!   that precedes committed movement, so a target observed while the planner
//!   is 8 ms ahead can only be stamped at the sample clock, and the planner
//!   reacts to it 8 ms late. Buy only as much slack as the loop needs.
//!
//! The target is an absolute point in common angular counts. An application
//! measures a gap in native counts instead, so it converts and adds it to the
//! planner's own position, which is what `target - position` is read against.

#[path = "support/mod.rs"]
mod support;

use abcurves::continuous::{ContinuousPipeline, CountTransform, PipelineOptions};
use std::collections::VecDeque;
use std::time::Instant;

const TICK_US: i64 = 1_000;
const LOOKAHEAD_US: i64 = 8_000;
const DEADLINE_MS: i64 = 600;

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let radians_per_count = support::f64s("human_start", "radians_per_count")[0];
    let profile = support::pairs_i16("human_start", "profile_hardware");
    let transform = CountTransform::new(radians_per_count, true)?;
    let options = PipelineOptions::new(transform)
        .seed(7)
        .renderer_seed(7)
        .initial_xy([0.0, 0.0]);
    let mut stream = ContinuousPipeline::load(&profile, options)?;

    // The gap an application actually has: native device counts, Y downward.
    let gap = [50.0f64, 50.0];
    let radius = 5.0f64;
    let scale = transform.common_per_native();
    let here = stream.current_xy();
    let target = [here[0] + gap[0] * scale[0], here[1] + gap[1] * scale[1]];
    stream.update_target(target, 0)?;
    println!(
        "gap {gap:?} native counts is {:.1} common counts, {:.2} degrees",
        (target[0] - here[0]).hypot(target[1] - here[1]),
        (target[0] - here[0]).hypot(target[1] - here[1])
            * 0.000_350_948_708_328_061_8f64.to_degrees()
    );

    let mut queued: VecDeque<[i16; 2]> = VecDeque::new();
    let mut sample_us = 0i64;
    let mut emitted = [0i32; 2];
    let mut first_motion = None;
    let mut inside = None;
    let mut travelled_90 = None;
    let mut worst_ahead = 0f64;
    let mut thinnest = usize::MAX;

    for tick in 1..=DEADLINE_MS {
        // Stay LOOKAHEAD_US ahead of the report that is due this millisecond.
        let wanted = tick * TICK_US + LOOKAHEAD_US;
        if wanted > sample_us {
            let start = Instant::now();
            let block = stream.advance(wanted)?;
            worst_ahead = worst_ahead.max(start.elapsed().as_secs_f64() * 1e6);
            queued.extend(block.reports);
            sample_us = wanted;
        }

        // The output loop itself only drains the queue.
        thinnest = thinnest.min(queued.len());
        let report = queued.pop_front().unwrap_or([0, 0]);

        emitted[0] += i32::from(report[0]);
        emitted[1] += i32::from(report[1]);
        if report != [0, 0] && first_motion.is_none() {
            first_motion = Some(tick);
        }
        let moved = f64::from(emitted[0]).hypot(f64::from(emitted[1]));
        if travelled_90.is_none() && moved >= 0.9 * gap[0].hypot(gap[1]) {
            travelled_90 = Some(tick);
        }
        let remaining = (gap[0] - f64::from(emitted[0])).hypot(gap[1] - f64::from(emitted[1]));
        if inside.is_none() && remaining <= radius {
            inside = Some(tick);
        }
    }

    println!(
        "{} decisions, {} of them re-ran the motor network",
        stream.movement().decisions(),
        stream.movement().motor_evaluations()
    );
    println!(
        "worst advance {worst_ahead:.0} us, absorbed by the {} ms of lookahead, \
         which never held fewer than {thinnest} reports",
        LOOKAHEAD_US / 1000
    );
    println!(
        "reaction {:?} ms, 90 percent of the way at {:?} ms, inside {radius} counts at {:?} ms",
        first_motion, travelled_90, inside
    );
    println!("emitted {emitted:?} against a gap of {gap:?}");

    // A 50 count correction is about three degrees, and the model answers it
    // the way a hand does. Over 64 seeds it waits 4 to 389 ms before moving at
    // all, covers 90 percent of the distance between 99 and 472 ms, and lands
    // inside five counts between 107 and 562 ms, with five of the 64 still
    // adjusting at 600 ms. Nothing here is tunable: the pause is a learned
    // decision drawn before any head exists, so pinning a head moves the
    // settle, not the start. Under 100 ms is outside what this planner does.
    // That request belongs to the Static Planner, which never models reaction
    // because it takes over a hand that is already moving; see `static_choose`
    // for picking a head by its planned duration.
    Ok(())
}
