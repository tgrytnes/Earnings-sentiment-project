#!/usr/bin/env bash
set -euo pipefail

# Render a notebook using the Quarto project in notebooks/config
# Usage: scripts/render_notebook.sh notebooks/main.ipynb

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-notebook.ipynb>" >&2
  exit 1
fi

NB_PATH="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CONFIG_DIR="${REPO_ROOT}/notebooks/config"

NB_ABS="$(cd "$(dirname "$NB_PATH")" && pwd)/$(basename "$NB_PATH")"
if [[ ! -f "$NB_ABS" ]]; then
  echo "Notebook not found: $NB_PATH" >&2
  exit 1
fi

cd "$CONFIG_DIR"
echo "[quarto] Rendering with project at: $CONFIG_DIR"
quarto render "$NB_ABS" --quiet
echo "[quarto] Outputs in: ${REPO_ROOT}/notebooks/out"
