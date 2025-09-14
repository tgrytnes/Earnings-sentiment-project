#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        print(f"Wrote {path} (rows={len(df)})")
    except Exception as e:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"pyarrow not available? Saved fallback CSV {csv_path} (rows={len(df)}). Error: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate headline-level scores to per-ticker daily sentiment")
    ap.add_argument("--inp", default="data/interim/news_scored.parquet")
    ap.add_argument("--out", default="data/interim/daily_sentiment_by_ticker.parquet")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists() and not inp.with_suffix('.csv').exists():
        print(f"Input not found: {inp}")
        return 2
    try:
        df = pd.read_parquet(inp)
    except Exception:
        df = pd.read_csv(inp.with_suffix('.csv'))

    if df.empty:
        _safe_to_parquet(df, Path(args.out))
        return 0

    # Normalize date and ticker
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    df["date"] = df["published_utc"].dt.tz_convert(None).dt.date
    df["date"] = pd.to_datetime(df["date"])  # normalize to Timestamp
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()

    # Drop rows without ticker symbol
    df = df[df["ticker"] != ""]
    if df.empty:
        _safe_to_parquet(df, Path(args.out))
        return 0

    g = df.groupby(["ticker", "date"], as_index=False).agg(
        sent_count=("headline_id", "count"),
        pos_mean=("finbert_pos", "mean"),
        neg_mean=("finbert_neg", "mean"),
        neu_mean=("finbert_neu", "mean"),
    )
    g["comp_mean"] = g["pos_mean"] - g["neg_mean"]

    _safe_to_parquet(g, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

