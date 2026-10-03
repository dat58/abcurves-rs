#!/usr/bin/env bash
# Populate ./models from a local ABCurves checkout so the crate and its examples
# run without ABCURVES_MODEL_DIR. Nothing here is committed; see .gitignore.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-${ABCURVES_SOURCE_DIR:-$ROOT/origin/ABCurves/models}}"
TARGET="$ROOT/models"

if [ ! -f "$SOURCE/manifest.json" ]; then
    cat >&2 <<MESSAGE
No release models at $SOURCE

Clone the upstream project and point this script at its models directory:

    git clone https://github.com/optima-manent/ABCurves.git
    scripts/fetch_models.sh ABCurves/models

The models are published by the ABCurves authors; this crate only reads them.
MESSAGE
    exit 1
fi

REQUIRED=(
    manifest.json
    planner_seed7.pt
    planner_seed23.pt
    renderer_global_h80.bin
    continuous/manifest.json
    continuous/weights.npz
    continuous/brake.onnx
    continuous/choice.onnx
    continuous/choice_split.onnx
    continuous/events.onnx
    continuous/hazard.onnx
    continuous/motor.onnx
)

mkdir -p "$TARGET/continuous"
for name in "${REQUIRED[@]}"; do
    if [ ! -f "$SOURCE/$name" ]; then
        echo "missing from the source checkout: $name" >&2
        exit 1
    fi
    cp -f "$SOURCE/$name" "$TARGET/$name"
done

# The same code-level anchors the crate checks at load time.
cd "$TARGET"
sha256sum -c --quiet <<'ANCHORS'
d82c93071224f7eb225d1f2bcf46d52669a7270db414431d7622e032439b280d  planner_seed7.pt
d691ba155c4fa9b403c5a3e2ed9c44123fe00d3d1bee15c55ee9226f4531a23e  planner_seed23.pt
405c34bceb55485dfd6bd3c0368bce079feea680a64c6b261904ef5b4713e240  renderer_global_h80.bin
018a096c66213c62b429caf4a8ae6f0470f6e283248a24f476f0fed3e36ea881  continuous/weights.npz
ANCHORS

echo "models ready in $TARGET ($(du -sh "$TARGET" | cut -f1))"
