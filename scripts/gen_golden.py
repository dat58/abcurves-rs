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


def save(name, **arrays):
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / f"{name}.npz"
    np.savez(path, **arrays)
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size} bytes")


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
