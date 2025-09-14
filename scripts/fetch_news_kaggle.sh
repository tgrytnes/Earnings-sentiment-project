#!/usr/bin/env bash
set -euo pipefail

# Download a financial/stock news dataset from Kaggle
# Requires: Kaggle CLI configured (~/.kaggle/kaggle.json)

OUTDIR=${1:-data/raw/kaggle_news}

mkdir -p "$OUTDIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 2
fi

DATASET=${KAGGLE_NEWS_DATASET:-}
if [[ -z "${DATASET}" ]]; then
  echo "Selecting a stock/financial news dataset from Kaggle..."
  # Try publisher-focused queries first; pick the first slug from CSV output
  for q in \
    "yahoo finance news" \
    "seekingalpha" \
    "benzinga" \
    "marketwatch news" \
    "nasdaq news" \
    "investing.com news" \
    "bloomberg news" \
    "reuters financial news" \
    "stock market news" \
    "financial news" \
    "company news"; do
    cand=$(kaggle datasets list -s "$q" --csv | tail -n +2 | head -n 1 | cut -d, -f1 | tr -d '\r')
    if [[ -n "$cand" ]]; then DATASET="$cand"; break; fi
  done
fi

if [[ -z "$DATASET" ]]; then
  echo "Could not auto-select a Kaggle news dataset. Try manually:"
  echo "  kaggle datasets list -s 'stock news' --csv | head -n 5"
  echo "Then set: export KAGGLE_NEWS_DATASET=<owner/slug> and rerun."
  exit 3
fi

echo "Using Kaggle news dataset: $DATASET"
kaggle datasets download -d "$DATASET" -p "$OUTDIR" --unzip

echo "Done news fetch. Files in $OUTDIR"
