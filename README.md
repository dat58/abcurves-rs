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
composes the planner with the Renderer and returns integer hardware reports;
`StaticPipeline` performs one finite B→C continuation.

Models are found through `ABCURVES_MODEL_DIR`, then `./models`, then
`./origin/ABCurves/models`.

## Examples

Five runnable examples mirror the ones in the Python project. The recorded inputs they
read are bundled under `examples/data/`, so only the release models are needed.

```bash
cargo run --release --example quickstart        # continuous stream, moving target
cargo run --release --example streaming         # composed planner and renderer
cargo run --release --example assisted_start    # continue from real human history
cargo run --release --example static_quickstart # seam trigger, then one B to C finish
cargo run --release --example static_streaming  # the same finish, one report per tick
```

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
| Static fresh sampled-head plan | 361.2 | 464.2 | 0.8× |

The streaming and state-machine boundaries — the calls a 1 kHz output loop actually
makes every millisecond — are one to two orders of magnitude cheaper. The two dense
neural forward passes land near parity, because the Python path already dispatches
those GEMMs to OpenBLAS; the Static encoder is currently about 28% slower than that
hand-tuned kernel. The Renderer matches the reference C99 runtime, whose own receipt
reports 18.8 µs per generated report.

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

MIT.
