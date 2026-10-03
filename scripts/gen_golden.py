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


def save(fixture, **arrays):
    """Write each array as a flat little-endian blob named <key>.<dtype>."""
    folder = GOLDEN / fixture
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
