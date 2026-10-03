"""Shared deterministic case builder, mirrored bit for bit by the Rust tests."""
import numpy as np

TOTAL_UNIFORMS = 5376
CASES = 32
CUT_US = 1_000_000


def uniforms(seed):
    return np.random.RandomState(seed).random_sample(TOTAL_UNIFORMS)


def build_case(seed):
    u = uniforms(seed)
    history = ((u[0:1280] - 0.5) * 8.0).reshape(640, 2)
    target = ((u[1280:2560] - 0.5) * 300.0).reshape(640, 2)
    valid = u[2560:3200] > 0.25
    motion_known = u[3200:3840] > 0.1
    available = CUT_US + (np.arange(640) - 639) * 1000 - np.rint((u[3840:4480] - 0.45) * 4000).astype(np.int64)

    if seed % 3 == 1:
        target[511, 0] = np.nan
    if seed % 3 == 2:
        target[500, 1] = np.inf
        valid[:] = True
    if seed == 0:
        history[:] = 0.0
        valid[:] = False
        motion_known[:] = False
    if seed == 1:
        valid[:] = True
        motion_known[:] = True
        history[:] = 0.0

    return {
        "history": np.ascontiguousarray(history),
        "target": np.ascontiguousarray(target),
        "available": np.ascontiguousarray(available),
        "valid": np.ascontiguousarray(valid),
        "motion_known": np.ascontiguousarray(motion_known),
        "position": np.array([(u[4480] - 0.5) * 200.0, (u[4481] - 0.5) * 200.0]),
        "cut_us": CUT_US,
        "hold_age": u[4482] * 6000.0,
        "hold_target": np.array([(u[4483] - 0.5) * 300.0, (u[4484] - 0.5) * 300.0]),
        "hold_position": np.array([(u[4485] - 0.5) * 300.0, (u[4486] - 0.5) * 300.0]),
        "initial_error": u[4487] * 400.0,
        "innovation_age": u[4488] * 4000.0,
        "mode": int(u[4489] * 3.0),
        "brake_velocity": np.array([(u[4490] - 0.5) * 6.0, (u[4491] - 0.5) * 6.0]),
        "brake_acceleration": np.array([(u[4492] - 0.5) * 0.5, (u[4493] - 0.5) * 0.5]),
        "brake_duration": 4.0 + u[4494] * 188.0,
        "brake_age": u[4495] * 200.0,
        "coefficients": ((u[4608:5280] - 0.5) * 2.0).astype(np.float32).reshape(16, 21, 2),
        "brake_coefficients": ((u[5280:5290] - 0.5) * 40.0).reshape(5, 2),
        "previous": ((u[5290:5322] - 0.5) * 4.0).astype(np.float32).reshape(16, 2),
    }


def extra_uniforms(seed, count):
    return np.random.RandomState(seed + 1000).random_sample(count)


def build_encoded(seed):
    return ((extra_uniforms(seed, 96) - 0.5) * 4.0).astype(np.float32)


def renderer_script(seed, count):
    u = np.random.RandomState(seed).random_sample(count * 2)
    return ((u - 0.5) * 6.0).astype(np.float32).reshape(count, 2)


def renderer_script_wide(seed, count):
    u = np.random.RandomState(seed).random_sample(count * 2)
    return ((u - 0.5) * 2000.0).astype(np.float32).reshape(count, 2)
