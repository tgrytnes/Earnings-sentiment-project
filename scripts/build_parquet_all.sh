#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash scripts/build_parquet_all.sh \
#     --tickers AAPL,MSFT,AMZN,GOOG,META,NVDA,JPM,TSLA \
#     --start 2022-01-01 --end 2024-12-31

TICKERS="AAPL,MSFT,AMZN,GOOG,META,NVDA,JPM,TSLA"
START="2022-01-01"
END="2024-12-31"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tickers) TICKERS="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "[1/5] Fetching market data (prices + earnings)" >&2
python scripts/fetch_market_data.py --tickers "$TICKERS" --start "$START" --end "$END" --outdir data/raw

if [[ -d data/raw/kaggle_news ]]; then
  echo "[2/5] Normalizing Kaggle news from data/raw/kaggle_news" >&2
  python scripts/normalize_kaggle_news.py --indir data/raw/kaggle_news \
    --out data/raw/news.csv --tickers "$TICKERS" --start "${START}T00:00:00Z" --end "${END}T23:59:59Z"
elif [[ -f data/raw/news.csv ]]; then
  echo "[2/5] Found existing data/raw/news.csv; skipping normalization" >&2
else
  echo "[2/5] Kaggle news folder not found. Fetching Yahoo Finance news as fallback" >&2
  python scripts/fetch_news_yf.py --tickers "$TICKERS" --start "$START" --end "$END" --outdir data/raw
fi

echo "[3/5] Filtering news" >&2
python scripts/filter_news.py --inp data/raw/news.csv --out data/raw/news_filtered.csv --min-title-len 12

echo "[4/5] Scoring headlines (FinBERT stub)" >&2
python scripts/score_news_finbert.py --inp data/raw/news_filtered.csv --out data/interim/news_scored.parquet

echo "[5/5] Aggregating daily sentiment and building earnings events" >&2
python scripts/aggregate_daily_sentiment.py --inp data/interim/news_scored.parquet --out data/interim/daily_sentiment.parquet
python scripts/build_earnings_events.py --prices data/raw/prices.csv --earnings data/raw/earnings.csv \
  --daily-sent data/interim/daily_sentiment.parquet --out data/processed/earnings_events.parquet

echo "Done. Outputs:"
echo "  - data/interim/news_scored.parquet"
echo "  - data/interim/daily_sentiment.parquet"
echo "  - data/processed/earnings_events.parquet"
