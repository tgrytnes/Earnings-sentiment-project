#!/usr/bin/env bash
set -euo pipefail

# One-command data fetch
# - Prices + earnings via yfinance
# - News via Kaggle or Hugging Face
# - Earnings call transcripts via Kaggle

# No external storage sync; Azure env removed

TICKERS=${TICKERS:-"AAPL,MSFT,AMZN,GOOG,META,NVDA,JPM,TSLA"}
START=${START:-"2023-01-01"}
END=${END:-"2024-12-31"}
NEWS_SOURCE=${NEWS_SOURCE:-kaggle}  # kaggle | hf

PY=${PYTHON:-python3}
if [[ -x ".venv/bin/python" ]]; then PY=".venv/bin/python"; fi

# Derive year range for selective Kaggle downloads (e.g., transcripts) if possible
Y0=${START:0:4}
Y1=${END:0:4}
YEARS=""
if [[ $Y0 =~ ^[0-9]{4}$ && $Y1 =~ ^[0-9]{4}$ ]]; then
  y=$Y0
  while [[ $y -le $Y1 ]]; do
    if [[ -z "$YEARS" ]]; then YEARS="$y"; else YEARS="$YEARS,$y"; fi
    y=$((y+1))
  done
  export TRANSCRIPTS_YEARS="$YEARS"
fi

echo "[1/3] Fetching prices + earnings..."
"$PY" scripts/fetch_market_data.py --tickers "$TICKERS" --start "$START" --end "$END"

if [[ "$NEWS_SOURCE" == "kaggle" ]]; then
  echo "[2/3] Fetching news from Kaggle..."
  bash scripts/fetch_news_kaggle.sh data/raw/kaggle_news raw/kaggle_news
  echo "[2.1/3] Normalizing news to CSV..."
  "$PY" scripts/normalize_kaggle_news.py --indir data/raw/kaggle_news --out data/raw/news.csv --tickers "$TICKERS" --start "$START" --end "$END"
else
  echo "[2/3] Fetching news from Hugging Face..."
  "$PY" scripts/fetch_news_hf.py --dataset "${HF_NEWS_DATASET:-ashraq/financial-news}" --tickers "$TICKERS" --start "$START" --end "$END" --out data/raw/news.csv
fi

echo "[3/3] Fetching transcripts from Kaggle..."
bash scripts/fetch_transcripts_kaggle.sh data/raw/kaggle_transcripts raw/kaggle_transcripts
echo "[3.1/3] Normalizing transcripts to CSV..."
"$PY" scripts/normalize_kaggle_transcripts.py --indir data/raw/kaggle_transcripts --out data/raw/transcripts.csv --tickers "$TICKERS" --start "$START" --end "$END"

echo "All data fetched and uploaded."
