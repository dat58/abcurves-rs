"""Time the Python origin on this machine, mirroring benches/runtime.rs."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = ROOT / "origin" / "ABCurves"
sys.path.insert(0, str(ORIGIN))

import abcurves


def report(label, values):
    if not values:
        return
    values = np.sort(np.asarray(values, np.float64) / 1000.0)
    print(f"| {label:<48} | {len(values):>7} | {np.median(values):>10.3f} "
          f"| {values[int(len(values) * 0.95) % len(values)]:>10.3f} | {values[-1]:>10.3f} |")


def main():
    fresh_motor, fresh_skipped, buffered, receipt, resets = [], [], [], [], []
    for seed in range(7, 14):
        movement = abcurves.load(seed=seed)
        for tick in range(1, 8001):
            if tick % 16 == 1:
                target = (400.0 + 120.0 * np.sin(tick / 900.0),
                          -150.0 + 90.0 * np.cos(tick / 700.0))
                start = time.perf_counter_ns()
                movement.update_target(target, timestamp_us=(tick - 1) * 1000)
                receipt.append(time.perf_counter_ns() - start)
            before_fresh = movement.decisions
            before_motor = movement.planner.motor_evaluations
            start = time.perf_counter_ns()
            block = movement.advance(tick * 1000)
            elapsed = time.perf_counter_ns() - start
            if movement.decisions == before_fresh:
                buffered.append(elapsed)
            elif movement.planner.motor_evaluations == before_motor:
                fresh_skipped.append(elapsed)
            else:
                fresh_motor.append(elapsed)
        start = time.perf_counter_ns()
        movement.reset()
        resets.append(time.perf_counter_ns() - start)

    print(f"| {'Public call':<48} | {'Samples':>7} | {'Median us':>10} | {'p95 us':>10} | {'Max us':>10} |")
    print("|" + "-" * 50 + "|" + "-" * 9 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    report("Continuous fresh call, motor evaluated", fresh_motor)
    report("Continuous fresh call, motor skipped", fresh_skipped)
    report("Continuous buffered 1 ms output", buffered)
    report("Continuous target receipt", receipt)
    report("Continuous reset", resets)
    composed()
    static_planner()


def composed():
    from abcurves import ContinuousPipeline, CountTransform

    with np.load(ORIGIN / "examples" / "data" / "human_start.npz", allow_pickle=False) as data:
        profile = np.asarray(data["profile_hardware"], np.int16)
        transform = CountTransform(float(data["radians_per_count"]), y_down=True)

    fresh_motor, fresh_skipped, buffered = [], [], []
    for seed in range(7, 10):
        stream = ContinuousPipeline(profile, transform=transform, seed=seed, renderer_seed=101)
        for tick in range(1, 8001):
            if tick % 16 == 1:
                stream.update_target((400.0 + 120.0 * np.sin(tick / 900.0),
                                      -150.0 + 90.0 * np.cos(tick / 700.0)),
                                     timestamp_us=(tick - 1) * 1000)
            before_fresh = stream.movement.decisions
            before_motor = stream.movement.planner.motor_evaluations
            start = time.perf_counter_ns()
            block = stream.advance(tick * 1000)
            elapsed = time.perf_counter_ns() - start
            del block
            if stream.movement.decisions == before_fresh:
                buffered.append(elapsed)
            elif stream.movement.planner.motor_evaluations == before_motor:
                fresh_skipped.append(elapsed)
            else:
                fresh_motor.append(elapsed)
    report("Continuous + Renderer fresh call, motor evaluated", fresh_motor)
    report("Continuous + Renderer fresh call, motor skipped", fresh_skipped)
    report("Continuous + Renderer buffered 1 ms output", buffered)


def static_planner():
    from abcurves import StaticPipeline

    with np.load(ORIGIN / "examples" / "data" / "static_event.npz", allow_pickle=False) as data:
        event = {key: data[key] for key in data.files}
    raw = np.asarray(event["raw_dxdy"], np.float32)
    prefix = raw[: int(event["b_index"]) + 1]
    target = tuple(float(v) for v in event["target_rel_b"])
    radius = float(event["target_radius"])
    progress = float(event["progress_center"])
    window = np.asarray(event["profile_before_a"], np.int16)

    plans, prepares, completes, steps = [], [], [], []
    with StaticPipeline.from_pretrained(model_seed=7) as pipeline:
        for _ in range(20):
            start = time.perf_counter_ns()
            profile = pipeline.prepare_renderer_profile(window)
            prepares.append(time.perf_counter_ns() - start)
        profile = pipeline.prepare_renderer_profile(window)
        for round_index in range(5):
            for head in range(16):
                start = time.perf_counter_ns()
                planned = pipeline.planner.plan(
                    prefix, target_rel_at_B=target, target_radius=radius,
                    progress_center=progress, seed=0, head=head)
                plans.append(time.perf_counter_ns() - start)
                start = time.perf_counter_ns()
                stream = profile._prepared.begin(
                    planned.intent.smooth_dxdy, planned.intent.mask, event_seed=round_index)
                while not stream.complete:
                    inner = time.perf_counter_ns()
                    stream.step()
                    steps.append(time.perf_counter_ns() - inner)
                completes.append(time.perf_counter_ns() - start)
    report("Static seed 7: fresh sampled-head plan", plans)
    report("Static renderer profile preparation", prepares)
    report("Static complete handoff through last report", completes)
    report("Static renderer 1 ms report", steps)


if __name__ == "__main__":
    main()
