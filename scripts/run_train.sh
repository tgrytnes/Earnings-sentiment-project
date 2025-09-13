#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/exp_baseline.yaml}
python -m yourproj.preprocess --config $CONFIG
python -m yourproj.features --config $CONFIG
python -m yourproj.train --config $CONFIG
python -m yourproj.eval --config $CONFIG
echo "Done."
