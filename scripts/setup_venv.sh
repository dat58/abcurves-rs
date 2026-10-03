#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel
"$VENV/bin/python" -m pip install "numpy>=1.26" "numba>=0.61" "scipy>=1.14"
"$VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"
"$VENV/bin/python" -c "import numpy, numba, torch; print(numpy.__version__, numba.__version__, torch.__version__)"
