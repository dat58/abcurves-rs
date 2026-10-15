//! Drive the Static Planner from a live 1 kHz capture.
//!
//! The three phases a real application runs through, in the order it runs
//! them, with the work that must not happen at B pushed out of the way:
//!
//! 1. *idle* — bin the physical reports into closed milliseconds, keep the
//!    last 256, and prepare the renderer profile from them. That step costs
//!    2.7 ms, so it belongs here and nowhere near B.
//! 2. *approaching* — once a target exists, `OnsetDetector` names A and
//!    `BTrigger` watches for the first eligible handoff.
//! 3. *finishing* — at B, plan once (about 0.3 ms) and then emit one
//!    generated report per millisecond, 18 µs each.
//!
//! What the application injects during phase 3 is the generated report. The
//! hand is still on the mouse, so whatever it produces adds to that in the
//! real system; this example replays a recording and only shows the loop.

#[path = "support/mod.rs"]
mod support;

use abcurves::static_planner::{
    BEvent, BTrigger, Draw, Movement, OnsetConfig, OnsetDetector, StaticPipeline,
};
use std::time::Instant;

const BIN_US: i64 = 1_000;

/// Accumulate physical reports into closed 1 ms bins.
///
/// A 1 kHz device does not deliver exactly one report per millisecond, and a
/// bin with no report is a real `[0, 0]`, not a missing sample. Both planners
/// read one pair per closed bin, so this is where a live feed is made to fit.
struct Binner {
    open_until_us: i64,
    pending: [i32; 2],
}

impl Binner {
    fn new(start_us: i64) -> Self {
        Self {
            open_until_us: start_us + BIN_US,
            pending: [0, 0],
        }
    }

    fn push(&mut self, dx: i32, dy: i32, timestamp_us: i64, closed: &mut Vec<[i16; 2]>) {
        while timestamp_us >= self.open_until_us {
            closed.push([self.pending[0] as i16, self.pending[1] as i16]);
            self.pending = [0, 0];
            self.open_until_us += BIN_US;
        }
        self.pending[0] += dx;
        self.pending[1] += dy;
    }
}

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let idle = support::pairs_i16("static_event", "profile_before_a");
    let moving = support::pairs_i16("static_event", "raw_dxdy");
    let target_a = support::f64s("static_event", "target_rel_a");
    let radius = support::f64s("static_event", "target_radius")[0];
    let seed = 2026;

    let mut pipeline = StaticPipeline::from_pretrained(7, None, true)?;
    let mut binner = Binner::new(0);
    let mut closed: Vec<[i16; 2]> = Vec::new();
    let mut recent: Vec<[i16; 2]> = Vec::new();
    let mut clock_us = 0i64;

    // Phase 1. No target yet, so the capture only feeds the rolling window.
    // One tick is dropped to show that an empty bin is a real [0, 0].
    for (index, report) in idle.iter().enumerate() {
        clock_us += BIN_US;
        if index != 100 {
            binner.push(
                i32::from(report[0]),
                i32::from(report[1]),
                clock_us,
                &mut closed,
            );
        }
    }
    binner.push(0, 0, clock_us + BIN_US, &mut closed);
    recent.extend_from_slice(&closed);
    closed.clear();
    let window = recent.split_off(recent.len() - 256);

    let start = Instant::now();
    let profile = pipeline.prepare_renderer_profile(&window)?;
    println!(
        "profile prepared from 256 idle reports in {:.0} us, before any target existed",
        start.elapsed().as_secs_f64() * 1e6
    );

    // Phase 2. A target appears. A is detected causally, then B is watched for.
    //
    // The detector needs about a dozen milliseconds of evidence before it will
    // name A, and it then points *backwards*: `event.index` is where the
    // movement began, several bins before the call that fired. So every bin
    // since the target appeared is kept, and when A is named the trigger is
    // armed and caught up over the bins it missed. Skipping that catch-up
    // would hand the trigger a movement that starts in the middle.
    let mut detector = OnsetDetector::new(OnsetConfig::default());
    let mut trigger = BTrigger::recommended();
    let mut since_target: Vec<[i16; 2]> = Vec::new();
    let mut seen = [0.0f64; 2];
    let mut since_a = [0.0f64; 2];
    let mut a_index = 0usize;
    let mut handoff_at = None;

    'capture: for report in &moving {
        clock_us += BIN_US;
        binner.push(
            i32::from(report[0]),
            i32::from(report[1]),
            clock_us,
            &mut closed,
        );
        for bin in closed.drain(..) {
            since_target.push(bin);
            let (dx, dy) = (f64::from(bin[0]), f64::from(bin[1]));
            let pending = if trigger.armed() {
                seen[0] += dx;
                seen[1] += dy;
                since_target.len() - 1..since_target.len()
            } else {
                // The detector sees the target vector before this bin moved it.
                let before = [target_a[0] - seen[0], target_a[1] - seen[1]];
                seen[0] += dx;
                seen[1] += dy;
                let Some(event) = detector.push(dx, dy, Some(before))? else {
                    continue;
                };
                a_index = event.index;
                trigger.arm([target_a[0], target_a[1]], radius)?;
                println!(
                    "A named at bin {} after {} bins of evidence; catching the trigger up",
                    a_index,
                    since_target.len() - a_index
                );
                a_index..since_target.len()
            };
            for index in pending {
                let bin = since_target[index];
                let (dx, dy) = (f64::from(bin[0]), f64::from(bin[1]));
                since_a[0] += dx;
                since_a[1] += dy;
                // The trigger sees it after.
                let after = [target_a[0] - since_a[0], target_a[1] - since_a[1]];
                match trigger.push_tick(dx, dy, after, radius)? {
                    Some(BEvent::Reject(reject)) => {
                        println!("no eligible handoff: {}", reject.reason);
                        return Ok(());
                    }
                    Some(BEvent::Fire(fire)) => {
                        handoff_at = Some(fire.t_ms);
                        break 'capture;
                    }
                    None => {}
                }
            }
        }
    }
    let Some(t_ms) = handoff_at else {
        println!("the movement ended with no eligible handoff");
        return Ok(());
    };

    // Phase 3. B is the last bin captured; nothing is chosen retroactively.
    let movement = Movement::from_reports(
        [0.0, 0.0],
        [target_a[0], target_a[1]],
        radius,
        &since_target[a_index..],
    );
    let start = Instant::now();
    let mut stream = pipeline.prepare(
        movement.prefix(),
        &profile,
        movement.handoff()?,
        Draw::seeded(seed),
    )?;
    let planned = start.elapsed();
    println!(
        "B at {t_ms} ms after A, planned {} ms of finish in {:.0} us, cold (head {})",
        stream.duration_ms(),
        planned.as_secs_f64() * 1e6,
        stream.head
    );

    let mut worst = 0u128;
    let mut emitted = [0i32; 2];
    while !stream.complete() {
        let start = Instant::now();
        let report = stream.step(pipeline.model())?;
        worst = worst.max(start.elapsed().as_nanos());
        emitted[0] += i32::from(report[0]);
        emitted[1] += i32::from(report[1]);
    }
    println!(
        "emitted {emitted:?} counts, worst single report {:.1} us of the 1000 available",
        worst as f64 / 1000.0
    );
    Ok(())
}
