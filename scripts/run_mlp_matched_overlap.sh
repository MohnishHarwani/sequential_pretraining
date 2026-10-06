#!/bin/sh
# Portable end-to-end A/B/C, three seeds. Data downloads are automatic.
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO"
RESULTS_DIR=${1:-results/mlp_ABC}
PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" src/run.py mechanism --out "$RESULTS_DIR/runs.jsonl"
"$PYTHON_BIN" src/analyze_mlp_configs.py "$RESULTS_DIR/runs.jsonl" --output "$RESULTS_DIR/analysis"
