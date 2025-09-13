#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_train.sh [CONFIG]
CONFIG=${1:-configs/exp_baseline.yaml}

# Pick a python interpreter: prefer .venv, then $PYTHON, then python3
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
else
  PY_BIN=${PYTHON:-python3}
  if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "No suitable Python interpreter found (.venv/bin/python or python3)" >&2
    exit 1
  fi
fi

# Ensure src is importable if not installed
export PYTHONPATH=${PYTHONPATH:-src}

$PY_BIN -m earnings_sentiment.preprocess "$CONFIG"
$PY_BIN -m earnings_sentiment.features "$CONFIG"
$PY_BIN -m earnings_sentiment.train "$CONFIG"
$PY_BIN -m earnings_sentiment.eval "$CONFIG"
echo "Done."
