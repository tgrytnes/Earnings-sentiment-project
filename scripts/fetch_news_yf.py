#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from datetime import timezone
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


def maybe_upload(*args, **kwargs):
    # External storage disabled; no-op
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Yahoo Finance news for tickers in date window")
    ap.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    ap.add_argument("--start", type=str, default=DEFAULT_START)
    ap.add_argument("--end", type=str, default=DEFAULT_END)
    ap.add_argument("--outdir", type=str, default="data/raw")
    # External upload removed; flags kept for compatibility but ignored
    ap.add_argument("--no-upload", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--container", type=str, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start = pd.Timestamp(args.start, tz=timezone.utc)
    end = pd.Timestamp(args.end, tz=timezone.utc)

    rows: list[dict] = []
    for t in tickers:
        try:
            items = yf.Ticker(t).news or []
        except Exception:
            items = []
        for it in items:
            ts = it.get("providerPublishTime") or it.get("published")
            if ts is None:
                continue
            try:
                dt = pd.to_datetime(int(ts), unit="s", utc=True)
            except Exception:
                # Some providers may return ms timestamps
                try:
                    dt = pd.to_datetime(int(ts), unit="ms", utc=True)
                except Exception:
                    continue
            if not (start <= dt <= end):
                continue
            rows.append(
                {
                    "ticker": t,
                    "published_utc": dt.isoformat(),
                    "source": (it.get("publisher") or ""),
                    "title": (it.get("title") or ""),
                    "link": (it.get("link") or it.get("url") or ""),
                }
            )

    news = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_csv = outdir / "news.csv"
    news.to_csv(n_csv, index=False)
    print(f"news rows: {len(news)} -> {n_csv}")
    maybe_upload(n_csv, "raw/news.csv", args.container, not args.no_upload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
