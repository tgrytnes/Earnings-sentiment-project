#!/usr/bin/env bash
set -euo pipefail

# Download an earnings call transcripts dataset from Kaggle
# Requires: Kaggle CLI configured (~/.kaggle/kaggle.json)

OUTDIR=${1:-data/raw/kaggle_transcripts}

mkdir -p "$OUTDIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 2
fi

DATASET=${KAGGLE_TRANSCRIPTS_DATASET:-}
if [[ -z "${DATASET}" ]]; then
  echo "Selecting an earnings call transcripts dataset..."
  # Use CSV output and take the first slug (ref) column
  DATASET=$(kaggle datasets list -s "earnings call transcripts" --csv | tail -n +2 | head -n 1 | cut -d, -f1 | tr -d '\r')
fi
if [[ -z "$DATASET" ]]; then
  echo "Could not auto-select a Kaggle dataset. Try manually:"
  echo "  kaggle datasets list -s 'earnings call transcripts' --csv | head -n 5"
  echo "Then set: export KAGGLE_TRANSCRIPTS_DATASET=<slug> and rerun."
  exit 3
fi
echo "Using Kaggle dataset: $DATASET"

# Optionally filter files by year(s) or explicit names to reduce download size
# - Set TRANSCRIPTS_YEARS="2022,2023,2024" to match filenames containing these years
# - Or set KAGGLE_TRANSCRIPTS_FILES to a comma-separated list of filename substrings
if kaggle datasets files -d "$DATASET" --csv > /tmp/_kg_files.csv 2>/dev/null; then
  FILTERED=()
  # Build list of candidate filenames (first CSV column)
  ALL_FILES=$(tail -n +2 /tmp/_kg_files.csv | cut -d, -f1)

  if [[ -n "${KAGGLE_TRANSCRIPTS_FILES:-}" ]]; then
    IFS=, read -r -a WANT_FILES <<< "$KAGGLE_TRANSCRIPTS_FILES"
    while IFS= read -r f; do
      for w in "${WANT_FILES[@]}"; do
        if [[ "$f" == *"$w"* ]]; then FILTERED+=("$f"); break; fi
      done
    done <<< "$ALL_FILES"
  fi

  if [[ -n "${TRANSCRIPTS_YEARS:-}" ]]; then
    IFS=, read -r -a YEARS_ARR <<< "$TRANSCRIPTS_YEARS"
    while IFS= read -r f; do
      for y in "${YEARS_ARR[@]}"; do
        if [[ "$f" == *"$y"* ]]; then FILTERED+=("$f"); break; fi
      done
    done <<< "$ALL_FILES"
  fi

  # De-duplicate filtered list
  if [[ ${#FILTERED[@]} -gt 0 ]]; then
    UNIQUE=()
    for f in "${FILTERED[@]}"; do
      skip=
      for u in "${UNIQUE[@]:-}"; do
        [[ "$u" == "$f" ]] && skip=1 && break
      done
      [[ -n "$skip" ]] || UNIQUE+=("$f")
    done
    echo "Downloading filtered files only: ${UNIQUE[*]}"
    for f in "${UNIQUE[@]}"; do
      kaggle datasets download -d "$DATASET" -f "$f" -p "$OUTDIR" --unzip
    done
    SKIP_FULL=1
  fi
fi

if [[ -z "${SKIP_FULL:-}" ]]; then
  echo "No filters provided or matched; downloading full dataset (may be large)"
  kaggle datasets download -d "$DATASET" -p "$OUTDIR" --unzip
fi

echo "Done transcripts fetch. Files in $OUTDIR"
