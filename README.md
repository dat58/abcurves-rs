# abcurves-rs

A pure-Rust port of the [ABCurves](https://github.com/optima-manent/ABCurves) inference
runtime: human mouse movement generated one millisecond at a time.

The crate covers the Continuous Planner, the Static Planner (A→B→C) and the
fixed-point Renderer. It loads the official release artifacts unchanged — the same
`weights.npz`, `planner_seed*.pt` and `renderer_global_h80.bin`, verified against the
same code-level size and SHA-256 anchors — and needs no Python, no Numba warm-up and
no native build step.

## Status

The inference runtime is complete and validated against the Python origin.
Training, evaluation and dataset tooling stay in the Python project.

## Using it

```rust
use abcurves::ContinuousOptions;

let mut movement = ContinuousOptions::default().seed(2026).load()?;
movement.update_target([100.0, 30.0], 0)?;
let first = movement.advance(500_000)?;

movement.update_target([125.0, 45.0], 540_000)?;
let second = movement.advance(1_000_000)?;
println!("{:?}", second.xy.last());
# Ok::<(), abcurves::Error>(())
```

Times are microseconds from initialization and `xy` holds absolute positions at closed
1 ms endpoints, in the Continuous model's common angular count space. `ContinuousPipeline`
composes the planner with the Renderer and returns integer hardware reports.

### The static handoff

`StaticPipeline` continues a movement a person has already begun. Give it where the
movement started, where the target is, how big it is, and the reports observed so far:

```rust
use abcurves::static_planner::{Draw, Movement, StaticPipeline};

let movement = Movement::from_reports(a, target, radius, &observed);
let reports = pipeline.continue_movement(&movement, &profile, Draw::seeded(2026))?;
```

B is the **last report supplied**. A to B is motion the mouse has already made, so it
cannot be chosen after the fact; the library validates that point rather than picking
its own. An unusable handoff is an error naming the rule it failed and by how much.

`radius` is the "counted as a hit" radius in native counts, the target radius on screen
divided by pixels per count. It is not decoration: the model reads it through
`distance_over_radius`, `prefix_lateral / radius`, `speed_at_B / radius` and the
edge-progress denominator, so it tells the model how precise it must be.

#### Frames

Native counts cross the API in the hardware frame: **X right, Y down**, which is what a
mouse reports and what `CountTransform` already assumed on the continuous side. The
models are trained in the canonical upward-positive frame, so `StaticPipeline` reflects
every report, prefix, profile window and handoff on the way in and reflects the generated
reports back on the way out. `set_y_down(false)` turns that off for a capture that is
already canonical.

`Movement`, `BTrigger` and `OnsetDetector` need no frame at all: their rules are
distances, dot products and progress fractions, every one of them invariant under the
reflection, so they work in whichever frame you hand them as long as it is one frame
throughout. `tests/static_pipeline.rs` pins the whole thing — the same capture mirrored
and run through the hardware frame returns exactly the mirror of the canonical run, byte
for byte.

The continuous side is unchanged: `CountTransform { y_down }` already reflects native
reports in and out, while common angular counts — what `update_target` and `Advance::xy`
speak — stay canonical, as upstream has them.

#### The short-continuation floor

Three rules in the frozen seam contract depend only on `|BC|`, the distance left at B,
and the radius — whatever path was taken from A:

| Rule | Constraint | 80% | 90% |
| --- | --- | :-: | :-: |
| `min_remaining_counts` | `\|BC\| >= 8` | yes | yes |
| `require_outside_target` | `\|BC\| > radius` | yes | yes |
| `min_edge_margin_counts` | `\|BC\| - radius >= 8` | no | yes |

So a continuation with under 8 counts left cannot be planned at all; use the Continuous
Planner, whose settling behaviour covers exactly that. Between those bounds the policy
is chosen automatically, keeping the 90 percent recommendation when the geometry allows
it and falling back to the 80 percent training contract when it does not.
`Movement::eligible_remaining` reports the usable `|BC|` window up front, so you can
plan the capture instead of discovering a refusal. `examples/static_simple.rs` sweeps
it:

```
      BC     |BC|  policy  outcome
    (4,4)     5.66       -  refused: |BC| is 5.66 counts, under the 8.0 the contract requires
    (6,6)     8.49     80%  393 reports, 41 carry motion to 350 ms, travelling [12, 6], 6.0 off
   (6,-6)     8.49     80%  384 reports, 35 carry motion to 339 ms, travelling [6, -9], 3.0 off
  (20,20)    28.28     90%  418 reports, 81 carry motion to 361 ms, travelling [34, 27], 15.7 off
 (20,-20)    28.28     90%  406 reports, 77 carry motion to 361 ms, travelling [21, -22], 2.2 off
```

The two diagonals differ because that sweep's prefix is a synthetic constant-velocity
crawl, far outside what the model saw in training. On the recorded captures they do not:
the three fixture events land 17, 24 and 11 counts from a 90, 58 and 74 count target, and
mirroring each one gives 21, 24 and 8.

#### Duration is the finishing style, not the distance

A plan that is 384 ms long for 8.49 counts looks wrong and is not. The travel is over in
about 17 ms; the rest is the micro-adjustment tail humans leave behind, which averages
0.05 counts per millisecond and reaches the integer reports as a handful of ones among
zeros — 35 of those 384 reports carry any motion at all, and the last of them lands at
339 ms. Lengthening `|BC|` from 8.49 to 28.28 counts moves the duration only from 384 to
406 ms, because what sets it is the head.

The sixteen heads are sixteen learned finishing styles, and at one fixed geometry they
span an order of magnitude: at BC `(6,6)` they plan 38 ms (head 1) through 384 ms
(head 13), median 92 ms. A seed picks among them, so `Draw::seeded(n)` is a draw from
that distribution, not a speed control. The seed used throughout these examples, 2026,
draws head 13 — the longest dweller — on every event in the fixtures, so every example
output here is the slow end of the range rather than a typical one. Pin a head with
`Draw::seeded(n).head(k)` when the duration has to be bounded, or read
`PreparedStream::duration_ms` before rendering and redraw if it does not fit the budget.

#### Drawing every head at once

Since the heads differ only in a 96 by 43 product on a shared trunk, all sixteen can be
planned for the price of roughly one and a half:

```rust
let plans = pipeline.plan_movement_all(&movement)?;      // 483 us for sixteen
let intent = plans.into_iter().min_by_key(|plan| plan.duration_ms).unwrap();
let reports = pipeline.begin(&profile, intent, seed)?.render_remaining(pipeline.model())?;
```

Entry `k` is bit for bit what `continue_movement` returns for `draw.head(k)`, so nothing
is lost by choosing after the fact; `tests/static_planner.rs` and `tests/static_pipeline.rs`
pin that equality for all sixteen. Rendering is the half that does not batch:
`continue_movement_all` exists, but the sixteen finishes of one handoff total about 2,300
reports at 18 µs each, so it costs roughly 42 ms and belongs nowhere near a 1 ms tick.
Choose from the plans — `duration_ms` and the cumulative `smooth_dxdy` are enough — and
render once. `examples/static_choose.rs` does exactly that.

One caution: the sixteen heads *are* the model's distribution of human finishes, and a
seed draws from it uniformly. Always taking the shortest replaces that distribution with
its own tail, which is a signature. Filter when a deadline forces it, and draw otherwise.

#### When you do not know where B is

A live application often has to be *told* when to take over. Drive a `BTrigger` instead,
which fires at the first eligible crossing and hands you the same geometry:

```rust
let mut trigger = BTrigger::recommended();
trigger.arm(target_rel_at_a, radius)?;

if let Some(BEvent::Fire(fire)) = trigger.push_tick(dx, dy, target_now, radius)? {
    let reports =
        pipeline.generate(&prefix, &profile, Handoff::from(&fire), Draw::seeded(2026))?;
}
```

`Handoff` groups the three numbers describing B and comes straight from `BFire`. `Draw`
groups the two seeds: `Draw::seeded(n)` uses one for both, `Draw::split(a, b)` separates
them, `.head(k)` pins a head instead of sampling. Model choice is separate again: the
`model_seed` argument to `StaticPipeline::from_pretrained` picks between the two
independently trained cells, 7 and 23.

#### From a live 1 kHz capture

`examples/static_live.rs` is the whole loop against a simulated capture, and it ends on
the same numbers as `static_quickstart` — B at 168 ms, 341 reports, `[53, -92]` counts —
which is the check that the live path and the offline path agree.

| Phase | Work | Cost | When |
| --- | --- | ---: | --- |
| idle | bin reports, keep the last 256, `prepare_renderer_profile` | 2.7 ms | before any target |
| approaching | `OnsetDetector::push`, then `BTrigger::push_tick` | < 1 µs per bin | every millisecond |
| at B | `Movement::handoff` + `prepare` | 0.3 ms | once |
| finishing | `PreparedStream::step` | 18 µs | every millisecond |

Four things decide whether this works:

**Feed it native counts, not pixels.** The API speaks the device's own counts in the
hardware frame, X right and **Y down**, the same convention `CountTransform` already uses
on the continuous side. A capture in pixels has to be divided by the pixels per count,
radius included. `StaticPipeline::set_y_down(false)` switches the whole pipeline to the
canonical upward-positive frame if that is what your capture produces.

**Close the bins yourself.** A 1 kHz device does not deliver exactly one report per
millisecond. Accumulate by timestamp into 1 ms bins and emit `[0, 0]` for a bin that
received nothing: an empty millisecond is real data, and dropping it shortens the
movement the model sees. Only the last 160 bins reach the planner, but the trigger needs
all of them from A.

**The onset points backwards.** `OnsetDetector` needs about two dozen milliseconds of
evidence before it will name A, and `OnsetEvent::index` then points at a bin well before
the call that fired. So keep every bin since the target appeared, start the prefix at
`index`, and replay the bins the trigger missed through `push_tick` before resuming live.
Arming at the firing bin instead hands the trigger a movement that begins in the middle.

**Prepare the profile while nothing is happening.** It is 2.7 ms, nine times the plan it
precedes, and it depends only on the device and the hand — not on the target. One
profile serves any number of events, so build it from a rolling 256-bin window during a
quiet moment and refresh it between movements.

### Choosing the continuous head

The Continuous Planner proposes sixteen heads too, but unlike the Static Planner's
uniform draw these are *scored*: a choice network produces a distribution over them and
the planner samples it. All sixteen geometries are decoded on every motor evaluation
anyway, so inspecting them costs nothing:

```rust
struct Deliberate;

impl HeadChooser for Deliberate {
    fn motor(&mut self, candidates: &MotorCandidates<'_>) -> usize {
        candidates.most_likely()          // or weigh candidates.displacement(head)
    }
}

runtime.set_head_chooser(Deliberate);
```

`HeadChooser` has two hooks, both defaulting to the drawn head: `motor` picks the
movement shape, `brake` picks the stopping shape and sees each head's `durations`, the
continuous counterpart of the static `duration_ms`. `PinnedHeads::motor(k)` is the
ready-made chooser that holds one decision fixed.

The draw happens whether or not a chooser overrides it, which is the point: the random
words it consumes keep every later decision in the stream aligned with the reference, so
an override changes the movement without desynchronising the planner. A chooser that
returns `candidates.sampled` reproduces the untouched stream bit for bit, and
`tests/continuous_stream.rs` asserts exactly that.

## Models

The release models are bundled under `models/` (4.7 MB), so a clone runs with nothing
else installed. Lookup order is `ABCURVES_MODEL_DIR`, then `./models`, then
`./origin/ABCurves/models`.

They are byte-for-byte copies of the upstream release, redistributed under its MIT
license; see [`models/LICENSE-ABCurves`](models/LICENSE-ABCurves) and the attribution
below. Their SHA-256 digests are pinned in the loader, which refuses anything that does
not match.

Twelve files are carried: both Planner checkpoints, the Renderer image, the continuous
weight archive, the two manifests and the six ONNX graphs. The ONNX graphs are present
because the continuous manifest declares them and the integrity check hashes every
declared file — this crate never evaluates them. The research float checkpoint and the
export-source `.pt` files are not carried, so a few entries in
[`models/README.md`](models/README.md) refer to files that are only in the upstream
project.

`scripts/fetch_models.sh` refreshes `models/` from an upstream checkout and re-verifies
the anchors, which is useful when upstream publishes a new release.

## Examples

Five runnable examples mirror the ones in the Python project. The recorded inputs they
read are bundled under `examples/data/`, so only the release models are needed.

```bash
cargo run --release --example quickstart        # continuous stream, moving target
cargo run --release --example streaming         # composed planner and renderer
cargo run --release --example assisted_start    # continue from real human history
cargo run --release --example static_quickstart # seam trigger, then one B to C finish
cargo run --release --example static_streaming  # the same finish, one report per tick
cargo run --release --example static_simple     # A, target, radius, reports -> continuation
cargo run --release --example static_choose     # plan all sixteen heads, render one
cargo run --release --example static_live       # the full 1 kHz loop: bin, detect, plan, emit
cargo run --release --example continuous_choose # inspect and override the continuous heads
cargo run --release --example continuous_to_point # type a target, watch it acquire
cargo run --release --example static_to_point     # the same, with the wrong planner
```

`continuous_to_point` and `static_to_point` both read a target from the keyboard and
print every report. They exist as a pair: the first acquires the target and settles on
it, the second shows what happens when the Static Planner is asked to start from
standstill, which is outside the regime it was trained for.

## Numerical agreement

Every component is checked against vectors generated from the Python origin
(`scripts/gen_golden.py`, committed under `tests/golden/`). The acceptance bar is the
project's own: exact integer equality for anything discrete, and the published
native-versus-reference envelope for continuous quantities.

| Component | Agreement with the origin |
| --- | --- |
| MT19937 draw stream, PCG64 head draws | bit exact |
| Continuous physical kernels (coarse, fine, dynamics, events, brake, selector) | bit exact |
| Continuous neural kernels | ≤ 5.6e-7 relative |
| Continuous stream, 12,000 samples / 375 decisions | timestamps and every mode and head exact; position RMS 8.9e-5, max 3.2e-4 counts |
| Renderer artifact, profile and generation | bit exact against the C99 runtime |
| Continuous pipeline, 2,000 integer reports | exact; rendered positions bit identical |
| Static 62-feature summary | bit exact |
| Static plans, 126 draws | duration and head exact; smooth deltas ≤ 9.8e-7 |
| Static pipeline, 4,785 integer reports | exact |
| Seam trigger and onset detector | exact |

For reference, the origin accepts a maximum position difference of 1.4e-3 common counts
between its own native and ONNX backends.

## Neural backends

`NeuralInference` selects the arithmetic that evaluates the learned tensors:

- `Native` (default) — hand-written kernels reproducing the accepted Padé9 motor and
  exact-SiLU event networks. No dependencies.
- `Candle` (`--features candle`) — the same tensors through `candle-core`, playing the
  numerical and portability role ONNX plays upstream. It reproduces every policy
  decision of the native path with a maximum position gap of 5.6e-4 counts.

## Performance

Measured on one Xeon Platinum 8358P at 2.60 GHz, running the identical eight-second
scenario through both runtimes on the same machine. The Python column is the origin
itself (NumPy 2.5.3, Numba 0.68, single-threaded BLAS) via `scripts/bench_reference.py`;
the Rust column is `cargo bench` built with `RUSTFLAGS="-C target-cpu=native"`. All
values are median microseconds per public call.

| Public call | Python | Rust | Speedup |
| --- | ---: | ---: | ---: |
| Continuous buffered 1 ms output | 7.77 | 0.066 | **118×** |
| Continuous target receipt | 5.84 | 0.028 | **209×** |
| Continuous reset | 464.7 | 7.1 | **65×** |
| Continuous fresh call, motor skipped | 133.6 | 6.1 | **22×** |
| Continuous + Renderer fresh call, motor skipped | 202.7 | 24.0 | **8.4×** |
| Continuous + Renderer buffered 1 ms output | 49.5 | 18.3 | **2.7×** |
| Continuous + Renderer fresh call, motor evaluated | 723.8 | 490.3 | 1.5× |
| Static complete handoff through last report | 2908 | 2265 | 1.3× |
| Continuous fresh call, motor evaluated | 602.8 | 474.2 | 1.3× |
| Static renderer profile preparation | 3379 | 2664 | 1.3× |
| Renderer 1 ms report | 21.9 | 18.0 | 1.2× |
| Static fresh sampled-head plan | 361.2 | 319.7 | 1.1× |
| Static plan, all sixteen heads | 5779 (sixteen calls) | 482.9 | **12×** |

The sixteen-head row has no Python counterpart: the reference has no batch entry point,
so the only way to get all sixteen there is sixteen full plans, which is what the 5,779 µs
is. Sharing the trunk is what makes the Rust side 12× cheaper, not the kernels.

The streaming and state-machine boundaries — the calls a 1 kHz output loop actually
makes every millisecond — are one to two orders of magnitude cheaper. The two dense
neural forward passes land near parity, because the Python path already dispatches
those GEMMs to OpenBLAS. The Renderer matches the reference C99 runtime, whose own
receipt reports 18.8 µs per generated report.

The static plan is the latency-critical one: once B closes, it has to finish before the
output loop wants its next report. At 320 µs it fits inside a 1 ms tick with room to
spare, provided the renderer profile was prepared in advance — that step costs 2.7 ms
and must not happen at B.

Without `-C target-cpu=native` the renderer costs 21.6 µs and a fresh motor call
548 µs; every other figure is unchanged.

## Development

```bash
scripts/setup_venv.sh                      # python reference for fixture generation
.venv/bin/python scripts/gen_golden.py --all
cargo test                                  # parity suite
cargo test --features candle
cargo bench
```

## License

This crate is MIT licensed.

The models under `models/` are the ABCurves release artifacts, redistributed unmodified
under that project's MIT license, with its copyright notice preserved in
[`models/LICENSE-ABCurves`](models/LICENSE-ABCurves).

> Release models by Optima Manent, from
> [ABCurves](https://github.com/optima-manent/ABCurves), MIT licensed.
>
> Uses ABCurves movement datasets by Optima Manent, licensed under CC BY 4.0.
> https://github.com/optima-manent/ABCurves

The recorded inputs under `examples/data/` and the reference vectors under
`tests/golden/` likewise derive from that project's CC BY 4.0 datasets.
