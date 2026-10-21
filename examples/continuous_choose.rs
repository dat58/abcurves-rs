//! Choose the motor head instead of letting the planner draw it.
//!
//! The Continuous Planner also proposes sixteen heads, but unlike the Static
//! Planner's uniform draw these are *scored*: a choice network produces a
//! distribution over them and the planner samples it. All sixteen geometries
//! are decoded every motor evaluation anyway, so inspecting them is free.
//!
//! The draw happens whether or not a chooser overrides it. That is deliberate:
//! the random words it consumes keep every later decision in the stream
//! aligned, so an override changes the movement without desynchronising the
//! planner.

#[path = "support/mod.rs"]
mod support;

use abcurves::continuous::{ContinuousOptions, HeadChooser, MotorCandidates, PinnedHeads};

#[derive(Default)]
struct Report {
    calls: usize,
}

impl HeadChooser for Report {
    fn motor(&mut self, candidates: &MotorCandidates<'_>) -> usize {
        if self.calls == 0 {
            println!("{:>5}  {:>7}  preview displacement", "head", "p");
            for head in 0..16 {
                let displacement = candidates.displacement(head);
                let mark = if head == candidates.sampled {
                    " <- drawn"
                } else {
                    ""
                };
                println!(
                    "{head:>5}  {:>7.4}  [{:>8.3}, {:>8.3}]{mark}",
                    candidates.probabilities[head], displacement[0], displacement[1]
                );
            }
            println!(
                "\nmost likely is head {}, the draw took head {}",
                candidates.most_likely(),
                candidates.sampled
            );
        }
        self.calls += 1;
        candidates.sampled
    }
}

const TARGET: [f64; 2] = [400.0, 160.0];

fn travel(chooser: Option<PinnedHeads>, label: &str) -> abcurves::Result<()> {
    let mut runtime = ContinuousOptions::default().seed(2026).load()?;
    if let Some(pinned) = chooser {
        runtime.set_head_chooser(pinned);
    }
    runtime.update_target(TARGET, 0)?;
    let mut last = [0.0f64; 2];
    let mut settled = 0usize;
    for tick in 1..=1_500i64 {
        let block = runtime.advance(tick * 1_000)?;
        for point in &block.xy {
            // The first block replays the stationary warm history, so settling
            // only counts once the movement has actually left the origin.
            if point[0].hypot(point[1]) > 1.0 {
                if (point[0] - last[0]).hypot(point[1] - last[1]) < 1e-6 {
                    settled += 1;
                } else {
                    settled = 0;
                }
            }
            last = *point;
        }
        if settled >= 100 {
            println!(
                "{label:>18}: settled at [{:>6.1}, {:>6.1}] by {tick:>4} ms, {:.1} counts from the target",
                last[0],
                last[1],
                (TARGET[0] - last[0]).hypot(TARGET[1] - last[1])
            );
            return Ok(());
        }
    }
    println!(
        "{label:>18}: still moving at [{:>6.1}, {:>6.1}] after 1500 ms",
        last[0], last[1]
    );
    Ok(())
}

fn main() -> abcurves::Result<()> {
    if !support::models_present() {
        return Ok(());
    }
    let mut runtime = ContinuousOptions::default().seed(2026).load()?;
    runtime.set_head_chooser(Report::default());
    runtime.update_target([400.0, 160.0], 0)?;
    for tick in 1..=256i64 {
        runtime.advance(tick * 1_000)?;
    }

    println!();
    travel(None, "drawn")?;
    for head in [0usize, 7, 15] {
        travel(
            Some(PinnedHeads::motor(head)),
            &format!("motor head {head}"),
        )?;
    }
    Ok(())
}
