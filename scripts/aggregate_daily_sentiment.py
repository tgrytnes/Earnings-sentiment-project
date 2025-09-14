#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
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


def rolling_slope(y: pd.Series, window: int) -> pd.Series:
    # OLS slope of y over time index 0..w-1, per rolling window
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    def _slope(arr: np.ndarray) -> float:
        y = arr
        num = ((x - x_mean) * (y - y.mean())).sum()
        return float(num / denom) if denom != 0 else np.nan
    return y.rolling(window, min_periods=window).apply(lambda a: _slope(a), raw=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate headline-level scores to daily market sentiment")
    ap.add_argument("--inp", default="data/interim/news_scored.parquet")
    ap.add_argument("--out", default="data/interim/daily_sentiment.parquet")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    try:
        df = pd.read_parquet(inp)
    except Exception:
        # Fallback if only CSV exists
        df = pd.read_csv(inp.with_suffix(".csv"))
        if "published_utc" in df:
            df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")

    if "published_utc" not in df.columns:
        print("Missing 'published_utc' column", file=sys.stderr)
        return 2

    # Market-wide aggregate per calendar day UTC
    df["date"] = pd.to_datetime(df["published_utc"], utc=True).dt.tz_convert(None).dt.date
    df["date"] = pd.to_datetime(df["date"])  # normalize to Timestamp date

    g = df.groupby("date", as_index=False).agg(
        sent_count=("headline_id", "count"),
        pos_mean=("finbert_pos", "mean"),
        neg_mean=("finbert_neg", "mean"),
        neu_mean=("finbert_neu", "mean"),
    )
    g["comp_mean"] = g["pos_mean"] - g["neg_mean"]

    g = g.sort_values("date").reset_index(drop=True)
    for w in (7, 14, 30):
        g[f"comp_{w}d_mean"] = g["comp_mean"].rolling(w, min_periods=min(3, w)).mean()

    # 30d slope and zscore
    g["comp_30d_slope"] = rolling_slope(g["comp_mean"].astype(float), 30)
    mu = g["comp_mean"].rolling(30, min_periods=10).mean()
    sd = g["comp_mean"].rolling(30, min_periods=10).std()
    g["comp_30d_zscore"] = (g["comp_mean"] - mu) / sd

    # Provenance
    g["news_version"] = df.get("news_version", "").iloc[0] if not df.empty and "news_version" in df.columns else ""
    g["processed_at"] = pd.Timestamp.utcnow()

    _safe_to_parquet(g, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

