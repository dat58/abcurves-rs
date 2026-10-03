"""Dump reference vectors from the Python origin for Rust parity tests."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = ROOT / "origin" / "ABCurves"
GOLDEN = ROOT / "tests" / "golden"

GENERATORS = {}


def generator(name):
    def register(function):
        GENERATORS[name] = function
        return function
    return register


DTYPE_SUFFIX = {
    "float32": "f32", "float64": "f64",
    "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
    "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
}


def save(fixture, base=None, **arrays):
    """Write each array as a flat little-endian blob named <key>.<dtype>."""
    folder = (base or GOLDEN) / fixture
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.iterdir():
        stale.unlink()
    total = 0
    for key, value in arrays.items():
        array = np.ascontiguousarray(value)
        suffix = DTYPE_SUFFIX[array.dtype.name]
        path = folder / f"{key}.{suffix}"
        path.write_bytes(array.astype(array.dtype.newbyteorder("<")).tobytes())
        total += path.stat().st_size
    print(f"{folder.relative_to(ROOT)}  {len(arrays)} arrays  {total} bytes")


RNG_SEEDS = [0, 7, 23, 109, 701, 2026, (1 << 40) + 17, (1 << 64) - 1]


@generator("rng_stream")
def rng_stream():
    import torch
    from abcurves._continuous.rng import RandomStream, softmax

    uniform_a, uniform_b, uniform_c = [], [], []
    exponential_a, exponential_b = [], []
    categories = []
    probabilities = []
    data_rng = np.random.default_rng(1402)
    logits = (data_rng.normal(size=(len(RNG_SEEDS) * 48, 16)) * 3.0).astype(np.float32)

    for index, seed in enumerate(RNG_SEEDS):
        stream = RandomStream(seed)
        reference = torch.Generator().manual_seed(seed if seed < (1 << 63) else seed - (1 << 64))
        uniform_a.append(stream.uniform(1))
        np.testing.assert_array_equal(uniform_a[-1], torch.rand(1, generator=reference).numpy())
        exponential_a.append(stream.exponential(16))
        np.testing.assert_array_equal(
            exponential_a[-1], torch.empty(16).exponential_(generator=reference).numpy())
        uniform_b.append(stream.uniform(1027))
        np.testing.assert_array_equal(uniform_b[-1], torch.rand(1027, generator=reference).numpy())
        for step in range(48):
            weights = softmax(logits[index * 48 + step])
            probabilities.append(weights)
            if step % 5 == 4:
                stream.discard_categorical()
                torch.multinomial(torch.from_numpy(weights), 1, generator=reference)
                categories.append(-1)
            else:
                drawn = stream.categorical(weights)
                expected = torch.multinomial(torch.from_numpy(weights), 1, generator=reference).item()
                assert drawn == expected, (seed, step, drawn, expected)
                categories.append(drawn)
        exponential_b.append(stream.exponential(300))
        uniform_c.append(stream.uniform(129))

    save(
        "rng_stream",
        seeds=np.array(RNG_SEEDS, dtype=np.uint64),
        uniform_a=np.concatenate(uniform_a),
        exponential_a=np.concatenate(exponential_a),
        uniform_b=np.concatenate(uniform_b),
        exponential_b=np.concatenate(exponential_b),
        uniform_c=np.concatenate(uniform_c),
        logits=logits.reshape(-1),
        probabilities=np.concatenate(probabilities),
        categories=np.array(categories, dtype=np.int64),
    )


@generator("rng_pcg64")
def rng_pcg64():
    seeds = np.array(
        [0, 1, 2, 3, 7, 23, 42, 100, 2026, 12345, 1 << 31, (1 << 63) - 1,
         (1 << 64) - 1, (1 << 70) + 9, 999_999_937],
        dtype=object,
    )
    heads, words = [], []
    for seed in seeds:
        heads.append(int(np.random.default_rng(int(seed)).integers(0, 16)))
        words.append(np.random.default_rng(int(seed)).integers(
            0, 1 << 64, size=16, dtype=np.uint64))
    save(
        "rng_pcg64",
        seeds=np.array([int(s) & ((1 << 64) - 1) for s in seeds], dtype=np.uint64),
        seeds_high=np.array([int(s) >> 64 for s in seeds], dtype=np.uint64),
        heads=np.array(heads, dtype=np.int64),
        words=np.concatenate(words),
    )


def digest_of(array):
    import hashlib
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(hashlib.sha256(contiguous.tobytes()).digest(), dtype=np.uint8)


def describe(named):
    """Pack name list, shapes and per-array content digests."""
    names, shapes, digests = [], [], []
    for name in sorted(named):
        value = np.asarray(named[name])
        names.append(name)
        shapes.append([value.ndim, *value.shape])
        digests.append(digest_of(value))
    return (
        np.frombuffer("\n".join(names).encode("utf8"), dtype=np.uint8),
        np.array([item for shape in shapes for item in shape], dtype=np.int64),
        np.concatenate(digests) if digests else np.zeros(0, np.uint8),
    )


@generator("artifacts")
def artifacts():
    import torch

    with np.load(ORIGIN / "models" / "continuous" / "weights.npz", allow_pickle=False) as bundle:
        continuous = {name: bundle[name] for name in bundle.files}
    continuous_names, continuous_shapes, continuous_digests = describe(continuous)

    payload = torch.load(ORIGIN / "models" / "planner_seed7.pt", map_location="cpu", weights_only=True)
    tensors = {name: value.numpy() for name, value in payload["model_state_dict"].items()}
    for extra in ("summary_mean", "summary_std", "prefix_mean", "prefix_std", "y_mean", "y_std"):
        tensors[extra] = payload[extra].numpy()
    planner_names, planner_shapes, planner_digests = describe(tensors)

    meta = np.array([
        payload["heads"], payload["horizon"], payload["summary_dim"], payload["target_dim"],
        payload["planner_config"]["prefix_len"], payload["prodmp"]["n_basis"],
    ], dtype=np.int64)
    scalars = np.array([
        payload["prodmp"]["alpha"], payload["prodmp"]["alpha_phase"], payload["prodmp"]["ridge"],
        *payload["hinge_thresholds"],
    ], dtype=np.float64)
    text = "\n".join([
        payload["schema"], payload["release_schema"], payload["release_status"],
        payload["seam_contract"]["schema"], payload["seam_contract"]["trigger"]["reference"],
        payload["prefix_representation"]["name"],
        *payload["summary_feature_names"],
    ])

    save(
        "artifacts",
        continuous_names=continuous_names,
        continuous_shapes=continuous_shapes,
        continuous_digests=continuous_digests,
        planner_names=planner_names,
        planner_shapes=planner_shapes,
        planner_digests=planner_digests,
        meta=meta,
        scalars=scalars,
        text=np.frombuffer(text.encode("utf8"), dtype=np.uint8),
    )


@generator("continuous_kernels")
def continuous_kernels():
    sys.path.insert(0, str(ROOT / "scripts"))
    from _cases import CASES, build_case

    from abcurves._continuous import native
    from abcurves._continuous.kernels import decode_geometry, decode_selected, decoder_geometry

    with np.load(ORIGIN / "models" / "continuous" / "weights.npz", allow_pickle=False) as bundle:
        velocity_basis = bundle["motor.velocity_basis"]
        carry_velocity = bundle["motor.carry_velocity"]
    mean_basis, mean_carry = decoder_geometry(velocity_basis, carry_velocity)

    coarse, fine, dynamics = [], [], []
    raw, context, basis = [], [], []
    heads, geometry, pairs, selected, brake = [], [], [], [], []

    for seed in range(CASES):
        case = build_case(seed)
        c, f, d = native.motor_features(
            case["history"], case["position"], case["target"], case["available"],
            case["valid"], case["motion_known"], case["cut_us"])
        coarse.append(c.reshape(-1))
        fine.append(f.reshape(-1))
        dynamics.append(d.reshape(-1))

        r, x, b = native.event_features(
            case["history"], case["position"], case["target"], case["available"],
            case["valid"], case["motion_known"], case["cut_us"], case["hold_age"],
            case["hold_target"], case["hold_position"], case["initial_error"],
            case["innovation_age"], case["mode"])
        raw.append(r.reshape(-1))
        context.append(x.reshape(-1))
        basis.append(b.reshape(-1))

        incoming = case["history"][-1]
        head_geometry = decode_geometry(case["coefficients"], incoming, mean_basis, mean_carry)
        heads.append(head_geometry.reshape(-1))

        previous = case["previous"] if seed % 2 else None
        actual = np.asarray([0.5, -0.25], np.float32) if seed % 2 else None
        unary, pair = native.selector_inputs(
            np.zeros(96, np.float32), head_geometry,
            np.zeros((16, 2), np.float32) if previous is None else previous,
            np.zeros(2, np.float32) if actual is None else actual,
            previous is not None)
        geometry.append(np.ascontiguousarray(unary[0, :, 96:]).reshape(-1))
        pairs.append(pair.reshape(-1))

        selected.append(decode_selected(
            case["coefficients"][seed % 16], incoming,
            velocity_basis[:32], carry_velocity[:32]).reshape(-1))

        times = case["brake_age"] + np.arange(33, dtype=np.float64)
        brake.append(native.c2_path(
            case["brake_velocity"], case["brake_acceleration"], case["brake_duration"],
            case["brake_coefficients"], times).reshape(-1))

    save(
        "continuous_kernels",
        coarse=np.concatenate(coarse),
        fine=np.concatenate(fine),
        dynamics=np.concatenate(dynamics),
        raw=np.concatenate(raw),
        context=np.concatenate(context),
        basis=np.concatenate(basis),
        heads=np.concatenate(heads),
        geometry=np.concatenate(geometry),
        pairs=np.concatenate(pairs),
        selected=np.concatenate(selected),
        brake=np.concatenate(brake),
        mean_basis=mean_basis.reshape(-1),
        mean_carry=mean_carry.reshape(-1),
    )


@generator("continuous_neural")
def continuous_neural():
    sys.path.insert(0, str(ROOT / "scripts"))
    from _cases import CASES, build_case, build_encoded

    from abcurves._continuous import native
    from abcurves._continuous.kernels import decode_geometry, decoder_geometry
    from abcurves._continuous.neural_native import NativeHeads, NativeMotor

    with np.load(ORIGIN / "models" / "continuous" / "weights.npz", allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    mean_basis, mean_carry = decoder_geometry(
        arrays["motor.velocity_basis"], arrays["motor.carry_velocity"])
    motor = NativeMotor(arrays, "pade9_vector")
    heads = NativeHeads(arrays, activation="exact")

    encoded, coefficients, logits = [], [], []
    duration, brake_coefficients, frequency, hazard = [], [], [], []

    for seed in range(CASES):
        case = build_case(seed)
        coarse, fine, dynamics = native.motor_features(
            case["history"], case["position"], case["target"], case["available"],
            case["valid"], case["motion_known"], case["cut_us"])
        out_encoded, out_coefficients = motor.motor(coarse, fine, dynamics)
        encoded.append(out_encoded.reshape(-1).copy())
        coefficients.append(out_coefficients.reshape(-1).copy())

        _, context, _ = native.event_features(
            case["history"], case["position"], case["target"], case["available"],
            case["valid"], case["motion_known"], case["cut_us"], case["hold_age"],
            case["hold_target"], case["hold_position"], case["initial_error"],
            case["innovation_age"], case["mode"])
        d, c, f, h = heads.events(context)
        duration.append(d.copy())
        brake_coefficients.append(c.reshape(-1).copy())
        frequency.append(f.copy())
        hazard.append(h.copy())

        head_geometry = decode_geometry(
            case["coefficients"], case["history"][-1], mean_basis, mean_carry)
        previous = case["previous"] if seed % 2 else None
        actual = np.asarray([0.5, -0.25], np.float32) if seed % 2 else None
        unary, pairs = native.selector_inputs(
            np.zeros(96, np.float32), head_geometry,
            np.zeros((16, 2), np.float32) if previous is None else previous,
            np.zeros(2, np.float32) if actual is None else actual,
            previous is not None)
        synthetic = build_encoded(seed).reshape(1, 96)
        out_logits = heads.choice(
            synthetic, np.ascontiguousarray(unary[:, :, 96:]), pairs, previous is not None)
        logits.append(out_logits.copy())

    save(
        "continuous_neural",
        encoded=np.concatenate(encoded),
        coefficients=np.concatenate(coefficients),
        logits=np.concatenate(logits),
        duration=np.concatenate(duration),
        brake_coefficients=np.concatenate(brake_coefficients),
        frequency=np.concatenate(frequency),
        hazard=np.concatenate(hazard),
    )


def _run_scenario(abcurves, spec, diagnostics):
    movement = abcurves.load(
        seed=spec["seed"], initial_xy=spec["initial_xy"],
        history=spec.get("history"), diagnostics=diagnostics)
    times, points = [], []
    for operation in spec["script"]:
        if operation[0] == "target":
            movement.update_target(operation[1], timestamp_us=operation[2])
        else:
            block = movement.advance(operation[1])
            times.append(block["time_us"])
            points.append(block["xy"])
    choices = movement.planner.movement_choices if diagnostics else []
    return (
        np.concatenate(times) if times else np.zeros(0, np.int64),
        np.concatenate(points) if points else np.zeros((0, 2), np.float64),
        np.array([int(c["mode"][0]) for c in choices], np.int64),
        np.array([int(c["head"][0]) for c in choices], np.int64),
        np.array([float(c["at_ms"][0]) for c in choices], np.float64),
    )


def _scenarios(start):
    quickstart = [
        ("target", (100.0, 30.0), 0), ("advance", 32_000),
        ("target", (125.0, 45.0), 40_000), ("advance", 64_000),
        ("advance", 1_000_000),
    ]
    acquisition = [("target", (800.0, -200.0), 0)]
    acquisition += [("advance", step * 8_000) for step in range(1, 751)]
    acquisition += [("target", (300.0, 500.0), 6_000_000)]
    acquisition += [("advance", 6_000_000 + step * 8_000) for step in range(1, 251)]
    quiet = [
        ("advance", 50_000),
        ("target", (40.0, 40.0), 50_000), ("advance", 2_000_000),
    ]
    assisted = [
        ("target", tuple(float(v) for v in start["target_xy"]), 0),
        ("advance", 128_000), ("advance", 1_000_000),
    ]
    return [
        {"seed": 2026, "initial_xy": (0.0, 0.0), "script": quickstart},
        {"seed": 7, "initial_xy": (0.0, 0.0), "script": acquisition},
        {"seed": 23, "initial_xy": (0.0, 0.0), "script": quiet},
        {"seed": 2026, "initial_xy": tuple(float(v) for v in start["observed_xy"]),
         "history": start["history"], "script": assisted},
    ]


@generator("continuous_stream")
def continuous_stream():
    import abcurves
    from abcurves import prepare_history

    with np.load(ORIGIN / "examples" / "data" / "human_start.npz", allow_pickle=False) as data:
        fixture = {name: data[name] for name in data.files}
    human = prepare_history(fixture["raw_common"], current_xy=fixture["observed_xy"])
    # The bundled filtered arrays were written by an equivalent formulation.
    np.testing.assert_allclose(human.history, fixture["filtered_history"], atol=1e-9)
    np.testing.assert_allclose(human.initial_xy, fixture["filtered_xy"], atol=1e-9)

    start = {
        "history": human.history,
        "observed_xy": human.observed_xy,
        "target_xy": fixture["target_xy"],
    }
    arrays = {
        "raw_common": fixture["raw_common"].reshape(-1),
        "observed_xy": np.asarray(fixture["observed_xy"], np.float64),
        "target_xy": np.asarray(fixture["target_xy"], np.float64),
        "profile_hardware": fixture["profile_hardware"].reshape(-1),
        "radians_per_count": np.asarray([float(fixture["radians_per_count"])], np.float64),
        "filtered_history": human.history.reshape(-1),
        "filtered_xy": np.asarray(human.initial_xy, np.float64),
    }
    for index, spec in enumerate(_scenarios(start)):
        times, points, mode, head, at_ms = _run_scenario(abcurves, spec, True)
        plain = _run_scenario(abcurves, spec, False)
        np.testing.assert_array_equal(times, plain[0])
        np.testing.assert_array_equal(points, plain[1])
        arrays[f"s{index}_time"] = times
        arrays[f"s{index}_xy"] = points.reshape(-1)
        arrays[f"s{index}_mode"] = mode
        arrays[f"s{index}_head"] = head
        arrays[f"s{index}_at_ms"] = at_ms
        print(f"  scenario {index}: {len(times)} samples, {len(mode)} decisions")
    save("continuous_stream", **arrays)


@generator("renderer")
def renderer():
    sys.path.insert(0, str(ROOT / "scripts"))
    from _cases import renderer_script, renderer_script_wide

    from abcurves.portable_renderer import PortableRendererModel

    model = PortableRendererModel(ORIGIN / "models" / "renderer_global_h80.bin")
    with np.load(ORIGIN / "examples" / "data" / "human_start.npz", allow_pickle=False) as data:
        human = np.asarray(data["profile_hardware"], np.int16)

    synthetic = np.zeros((256, 2), np.int16)
    synthetic[:, 0] = np.arange(256) & 1
    zeros = np.zeros((256, 2), np.int16)

    contexts = {"synthetic": synthetic, "human": human, "zeros": zeros}
    scenarios = [
        ("synthetic", 123, np.tile(np.asarray([1.0, 0.5], np.float32), (16, 1))),
        ("human", 101, renderer_script(3001, 1500)),
        ("human", 29, renderer_script(3002, 600)),
        ("zeros", 11, renderer_script_wide(3003, 800)),
        ("human", 2026, renderer_script_wide(3004, 400)),
    ]

    arrays = {"human_profile": human.reshape(-1)}
    for index, (context, seed, script) in enumerate(scenarios):
        profile = model.prepare_context(contexts[context])
        stream = profile.begin_stream(event_seed=seed)
        reports = np.array([stream.step(row) for row in script], np.int16)
        arrays[f"r{index}_reports"] = reports.reshape(-1)
        print(f"  scenario {index}: {context} seed {seed}, {len(script)} steps, "
              f"{int((reports != 0).any(axis=1).sum())} emitted")
    save("renderer", **arrays)


@generator("continuous_pipeline")
def continuous_pipeline():
    from abcurves import ContinuousPipeline, CountTransform, prepare_history

    with np.load(ORIGIN / "examples" / "data" / "human_start.npz", allow_pickle=False) as data:
        fixture = {name: data[name] for name in data.files}
    transform = CountTransform(float(fixture["radians_per_count"]), y_down=True)
    human = prepare_history(fixture["raw_common"], current_xy=fixture["observed_xy"])

    arrays = {}

    stream = ContinuousPipeline(fixture["profile_hardware"], transform=transform,
                                seed=2026, renderer_seed=101, initial_xy=(0.0, 0.0))
    stream.update_target((100.0, 30.0), timestamp_us=0)
    times, points, reports, rendered = [], [], [], []
    for tick in range(1, 1001):
        if tick == 501:
            stream.update_target((160.0, -40.0), timestamp_us=500_000)
        block = stream.advance(tick * 1000)
        times.append(block["time_us"])
        points.append(block["xy"])
        reports.append(block["reports"])
        rendered.append(block["rendered_xy"])
    arrays["p0_time"] = np.concatenate(times)
    arrays["p0_xy"] = np.concatenate(points).reshape(-1)
    arrays["p0_reports"] = np.concatenate(reports).reshape(-1)
    arrays["p0_rendered"] = np.concatenate(rendered).reshape(-1)
    arrays["p0_final"] = np.asarray(stream.rendered_xy, np.float64)

    assisted = ContinuousPipeline(fixture["profile_hardware"], transform=transform,
                                  initial_xy=human.observed_xy, history=human.history,
                                  seed=2026, renderer_seed=101,
                                  observed_xy=human.observed_xy)
    assisted.update_target(tuple(float(v) for v in fixture["target_xy"]), timestamp_us=0)
    block = assisted.advance(128_000)
    tail = assisted.advance(1_000_000)
    arrays["p1_time"] = np.concatenate([block["time_us"], tail["time_us"]])
    arrays["p1_xy"] = np.concatenate([block["xy"], tail["xy"]]).reshape(-1)
    arrays["p1_reports"] = np.concatenate([block["reports"], tail["reports"]]).reshape(-1)
    arrays["p1_rendered"] = np.concatenate([block["rendered_xy"], tail["rendered_xy"]]).reshape(-1)
    arrays["p1_final"] = np.asarray(assisted.rendered_xy, np.float64)

    print(f"  pipeline samples: {len(arrays['p0_time'])} and {len(arrays['p1_time'])}")
    save("continuous_pipeline", **arrays)


@generator("prodmp")
def prodmp():
    from abcurves.planner import _CachedProDMP
    from abcurves.prodmp import ProDMPConfig

    basis = _CachedProDMP(ProDMPConfig(n_basis=20, alpha=25.0, alpha_phase=3.0, ridge=1e-3))
    durations = [1, 2, 3, 5, 17, 64, 137, 256, 499, 1000]
    xi1, xi2, h, deltas = [], [], [], []
    rs = np.random.RandomState(4242)
    for index, duration in enumerate(durations):
        a, b, c = basis._canonical_components(duration)
        xi1.append(a)
        xi2.append(b)
        h.append(c.reshape(-1))
        weights = ((rs.random_sample(21 * 2) - 0.5) * 2.0e4).reshape(21, 2)
        velocity = (rs.random_sample(2) - 0.5) * 40.0
        deltas.append(basis.generate_deltas(weights, velocity, duration).reshape(-1))
    sample = np.arange(0, 2001, 10)
    save(
        "prodmp",
        s_grid=basis._s_grid[sample],
        phi_grid=basis._phi_grid[sample].reshape(-1),
        dphi_grid=basis._dphi_grid[sample].reshape(-1),
        durations=np.array(durations, np.int64),
        xi1=np.concatenate(xi1),
        xi2=np.concatenate(xi2),
        deltas=np.concatenate(deltas),
    )


@generator("static_planner")
def static_planner():
    from abcurves import StaticPlanner

    events = {}
    for name in ("static_event", "static_event_short", "static_event_long"):
        with np.load(ORIGIN / "examples" / "data" / f"{name}.npz", allow_pickle=False) as data:
            events[name] = {key: data[key] for key in data.files}

    arrays = {}
    raw_summaries, vectors, predictions, smooths, durations, heads = [], [], [], [], [], []
    prefixes, targets, radii, progresses, lengths = [], [], [], [], []

    for model_seed in (7, 23):
        planner = StaticPlanner.from_pretrained(model_seed=model_seed, prewarm=True)
        for name, event in events.items():
            raw = np.asarray(event["raw_dxdy"], np.float32)
            b_index = int(event["b_index"])
            prefix = raw[: b_index + 1]
            target = np.asarray(event["target_rel_b"], np.float64)
            radius = float(event["target_radius"])
            progress = float(event["progress_center"])
            if model_seed == 7:
                prefixes.append(prefix.reshape(-1))
                lengths.append(len(prefix))
                targets.append(target)
                radii.append(radius)
                progresses.append(progress)
                represented = planner.planner.represented_prefix_views(prefix)[0]
                raw_summaries.append(planner.summary.raw(
                    represented, (float(target[0]), float(target[1])), radius, progress,
                    assume_finite_counts=True))
                vectors.append(planner.summary.vector(
                    represented, (float(target[0]), float(target[1])), radius, progress,
                    assume_finite_counts=True).reshape(-1))
            for head in range(16):
                planned = planner.plan(prefix, target_rel_at_B=(float(target[0]), float(target[1])),
                                       target_radius=radius, progress_center=progress,
                                       seed=2026, head=head)
                intent = planned.intent
                durations.append(intent.duration_ms)
                heads.append(intent.head)
                smooths.append(intent.smooth_dxdy[: intent.duration_ms].reshape(-1))
            for seed in (0, 7, 23, 2026, 12345):
                planned = planner.plan(prefix, target_rel_at_B=(float(target[0]), float(target[1])),
                                       target_radius=radius, progress_center=progress, seed=seed)
                heads.append(planned.intent.head)
                durations.append(planned.intent.duration_ms)
                smooths.append(planned.intent.smooth_dxdy[: planned.intent.duration_ms].reshape(-1))

    arrays["prefixes"] = np.concatenate(prefixes)
    arrays["prefix_lengths"] = np.array(lengths, np.int64)
    arrays["targets"] = np.concatenate(targets)
    arrays["radii"] = np.array(radii, np.float64)
    arrays["progresses"] = np.array(progresses, np.float64)
    arrays["raw_summaries"] = np.concatenate(raw_summaries)
    arrays["vectors"] = np.concatenate(vectors)
    arrays["durations"] = np.array(durations, np.int64)
    arrays["heads"] = np.array(heads, np.int64)
    arrays["smooth"] = np.concatenate(smooths)
    del predictions
    print(f"  {len(durations)} plans, {len(arrays['smooth']) // 2} smooth samples")
    save("static_planner", **arrays)


@generator("static_pipeline")
def static_pipeline():
    from abcurves import StaticPipeline
    from abcurves.seam import BFire, BReject, BTrigger

    names = ("static_event", "static_event_short", "static_event_long")
    arrays = {}
    arm_arrays = {}
    raw_all, profile_all, lengths = [], [], []
    trigger_t, trigger_edge, trigger_center, trigger_reason = [], [], [], []
    fire_target, fire_radius = [], []
    report_counts, reports_all = [], []

    for index, name in enumerate(names):
        with np.load(ORIGIN / "examples" / "data" / f"{name}.npz", allow_pickle=False) as data:
            event = {key: data[key] for key in data.files}
        raw = np.asarray(event["raw_dxdy"], np.int16)
        target_a = np.asarray(event["target_rel_a"], np.float64)
        radius = float(event["target_radius"])
        profile_window = np.asarray(event["profile_before_a"], np.int16)
        arm_arrays[f"target_a_{index}"] = target_a
        arm_arrays[f"radius_{index}"] = np.array([radius], np.float64)
        raw_all.append(raw.reshape(-1))
        lengths.append(len(raw))
        profile_all.append(profile_window.reshape(-1))

        from abcurves.seam import OnsetDetector
        detector = OnsetDetector()
        onset = None
        running = np.zeros(2)
        for tick, delta in enumerate(raw):
            event = detector.push(float(delta[0]), float(delta[1]),
                                  target_a - running)
            running = running + delta
            if event is not None:
                onset = event
                break
        arm_arrays[f"onset_{index}"] = np.array(
            [-1 if onset is None else onset.index], np.int64)
        arm_arrays[f"onset_stats_{index}"] = np.array(
            [0.0, 0.0, 0.0] if onset is None
            else [onset.threshold, onset.speed_median, onset.speed_mad], np.float64)

        trigger = BTrigger.recommended()
        trigger.arm(target_a, radius)
        outcome = None
        for tick, delta in enumerate(raw):
            target_now = target_a - raw[: tick + 1].sum(axis=0, dtype=np.float64)
            result = trigger.push_tick(*delta.astype(np.float64),
                                       target_rel_now=target_now, target_radius_now=radius)
            if result is not None:
                outcome = (tick, result)
                break
        assert outcome is not None, name
        tick, result = outcome
        if isinstance(result, BReject):
            trigger_t.append(result.t_ms)
            trigger_edge.append(float("nan"))
            trigger_center.append(result.progress_center)
            trigger_reason.append(1)
            fire_target.append(np.zeros(2))
            fire_radius.append(0.0)
            report_counts.append(0)
            continue
        assert isinstance(result, BFire)
        trigger_t.append(result.t_ms)
        trigger_edge.append(result.progress_edge)
        trigger_center.append(result.progress_center)
        trigger_reason.append(0)
        fire_target.append(np.asarray(result.target_rel_at_B, np.float64))
        fire_radius.append(result.target_radius)

        prefix = raw[: tick + 1].astype(np.float32)
        for model_seed in (7, 23):
            with StaticPipeline.from_pretrained(model_seed=model_seed) as pipeline:
                profile = pipeline.prepare_renderer_profile(profile_window)
                for seed in (7, 2026, 12345):
                    out = pipeline.generate(
                        prefix, renderer_profile=profile,
                        target_rel_at_B=result.target_rel_at_B,
                        target_radius=result.target_radius,
                        progress_center=result.progress_center, seed=seed)
                    reports_all.append(np.asarray(out, np.int16).reshape(-1))
                    report_counts.append(len(out))

    arrays["raw"] = np.concatenate(raw_all)
    arrays["raw_lengths"] = np.array(lengths, np.int64)
    arrays["profiles"] = np.concatenate(profile_all)
    arrays["trigger_t"] = np.array(trigger_t, np.int64)
    arrays["trigger_edge"] = np.array(trigger_edge, np.float64)
    arrays["trigger_center"] = np.array(trigger_center, np.float64)
    arrays["trigger_reason"] = np.array(trigger_reason, np.int64)
    arrays["fire_target"] = np.concatenate(fire_target)
    arrays["fire_radius"] = np.array(fire_radius, np.float64)
    arrays["report_counts"] = np.array(report_counts, np.int64)
    arrays["reports"] = np.concatenate(reports_all)
    print(f"  {len(report_counts)} continuations, {len(arrays['reports']) // 2} reports")
    save("static_pipeline", **arrays)
    save("static_pipeline_arm", **arm_arrays)


@generator("example_data")
def example_data():
    """Bundle the recorded inputs the Rust examples read."""
    folder = ROOT / "examples" / "data"
    with np.load(ORIGIN / "examples" / "data" / "human_start.npz", allow_pickle=False) as data:
        save(
            "human_start", folder,
            raw_common=np.asarray(data["raw_common"], np.float64).reshape(-1),
            observed_xy=np.asarray(data["observed_xy"], np.float64),
            target_xy=np.asarray(data["target_xy"], np.float64),
            profile_hardware=np.asarray(data["profile_hardware"], np.int16).reshape(-1),
            radians_per_count=np.asarray([float(data["radians_per_count"])], np.float64),
        )
    with np.load(ORIGIN / "examples" / "data" / "static_event.npz", allow_pickle=False) as data:
        save(
            "static_event", folder,
            raw_dxdy=np.asarray(data["raw_dxdy"], np.int16).reshape(-1),
            profile_before_a=np.asarray(data["profile_before_a"], np.int16).reshape(-1),
            target_rel_a=np.asarray(data["target_rel_a"], np.float64),
            target_radius=np.asarray([float(data["target_radius"])], np.float64),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args()

    sys.path.insert(0, str(ORIGIN))

    if arguments.list:
        for name in GENERATORS:
            print(name)
        return
    names = list(GENERATORS) if arguments.all else arguments.names
    if not names:
        parser.error("pass fixture names, --all or --list")
    for name in names:
        if name not in GENERATORS:
            parser.error(f"unknown fixture {name}")
        GENERATORS[name]()


if __name__ == "__main__":
    main()
