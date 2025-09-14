#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
 
# Prepare src on sys.path if needed
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "JPM", "TSLA"]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2024-12-31"


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        df = df.reset_index().rename(columns=str.lower)
        if df.empty:
            continue
        df["ticker"] = t
        frames.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])


def fetch_earnings(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=16)
        except Exception:
            ed = None
        if ed is None or ed.empty:
            continue
        ed = ed.reset_index().rename(
            columns={
                "Earnings Date": "announce_datetime",
                "Reported EPS": "eps_actual",
                "EPS Estimate": "eps_estimate",
            }
        )
        ed["ticker"] = t
        if "announce_datetime" in ed:
            ed["bmo_amc"] = ed["announce_datetime"].dt.hour.apply(
                lambda h: "BMO" if pd.notnull(h) and h < 12 else "AMC"
            )
        else:
            ed["bmo_amc"] = "AMC"
        rows.append(ed[["ticker", "announce_datetime", "bmo_amc", "eps_actual", "eps_estimate"]])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=["ticker", "announce_datetime", "bmo_amc", "eps_actual", "eps_estimate"]
    )


def maybe_upload(*args, **kwargs):
    # External storage disabled; no-op
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch prices and earnings via yfinance")
    ap.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    ap.add_argument("--start", type=str, default=DEFAULT_START)
    ap.add_argument("--end", type=str, default=DEFAULT_END)
    ap.add_argument("--outdir", type=str, default="data/raw")
    # External upload removed; flags kept for compatibility but ignored
    ap.add_argument("--no-upload", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--container", type=str, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prices = fetch_prices(tickers, args.start, args.end)
    p_csv = outdir / "prices.csv"
    prices.to_csv(p_csv, index=False)
    print(f"prices rows: {len(prices)} -> {p_csv}")
    maybe_upload(p_csv, "raw/prices.csv", args.container, not args.no_upload)

    earnings = fetch_earnings(tickers)
    e_csv = outdir / "earnings.csv"
    earnings.to_csv(e_csv, index=False)
    print(f"earnings rows: {len(earnings)} -> {e_csv}")
    maybe_upload(e_csv, "raw/earnings.csv", args.container, not args.no_upload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
