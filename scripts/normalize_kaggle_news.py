#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

# Ensure src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def pick_datetime_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "published_utc",
        "published_at",
        "publish_date",
        "pub_date",
        "datetime",
        "timestamp",
        "time",
        "date",
        "created_at",
    ]
    cols = list(df.columns)
    # Prefer named candidates in order
    for c in candidates:
        if c in cols:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                pass
    # Fallback: try all columns
    for c in cols:
        s = df[c]
        # skip numeric columns that look like prices
        if pd.api.types.is_numeric_dtype(s):
            continue
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.5:
                return c
        except Exception:
            continue
    return None


def first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_single_df(df: pd.DataFrame, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, default_source: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "published_utc", "source", "title", "link", "body"])  # type: ignore

    # Identify core columns
    dt_col = pick_datetime_column(df)
    title_col = first_present(df, ["title", "headline", "news_title"]) or None
    body_col = first_present(df, ["text", "content", "article", "body"]) or None
    source_col = first_present(df, ["source", "publisher", "site"]) or None
    link_col = first_present(df, ["link", "url", "news_url", "article_url", "source_url"]) or None
    ticker_col = first_present(df, ["ticker", "symbol", "stock"]) or None

    if not dt_col:
        return pd.DataFrame(columns=["ticker", "published_utc", "source", "title", "link", "body"])  # type: ignore

    # Parse datetime
    ts = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    # If naive, assume UTC
    if ts.dt.tz is None:
        ts = pd.to_datetime(df[dt_col], errors="coerce").dt.tz_localize("UTC")

    df_n = pd.DataFrame({
        "published_utc": ts,
        "source": df[source_col] if source_col else default_source,
        "title": df[title_col] if title_col else None,
        "link": df[link_col] if link_col else None,
        "body": df[body_col] if body_col else None,
    })

    # Filter date window
    df_n = df_n[df_n["published_utc"].between(start, end, inclusive="both")]
    if df_n.empty:
        return pd.DataFrame(columns=["ticker", "published_utc", "source", "title", "link", "body"])  # type: ignore

    # Attach tickers
    out_rows: list[pd.DataFrame] = []
    if ticker_col and ticker_col in df.columns:
        # Normalize symbols and filter to requested set
        syms = df[ticker_col].astype(str).str.upper().str.strip()
        df_n2 = df_n.copy()
        df_n2["ticker"] = syms
        df_n2 = df_n2[df_n2["ticker"].isin(tickers)]
        out_rows.append(df_n2)
    else:
        # Heuristic: detect tickers in title/body
        tickers_sorted = sorted(set(tickers), key=len, reverse=True)
        # Word boundary for tickers consisting of letters/numbers; handle hyphens by escaping
        pat = re.compile(r"(?<![A-Z0-9])(" + "|".join(re.escape(t) for t in tickers_sorted) + r")(?![A-Z0-9])")
        titles = df_n["title"].fillna("").astype(str)
        bodies = df_n["body"].fillna("").astype(str)
        for idx, (t, b) in enumerate(zip(titles, bodies)):
            text = f"{t} {b}".upper()
            matches = sorted(set(pat.findall(text)))
            if not matches:
                continue
            row = df_n.iloc[[idx]].copy()
            for sym in matches:
                r = row.copy()
                r["ticker"] = sym
                out_rows.append(r)

    if not out_rows:
        return pd.DataFrame(columns=["ticker", "published_utc", "source", "title", "link", "body"])  # type: ignore
    out = pd.concat(out_rows, ignore_index=True)
    # Reorder columns
    cols = ["ticker", "published_utc", "source", "title", "link", "body"]
    return out[cols]


def read_any(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                return pd.DataFrame([data])
        if path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        # Skip others
    except Exception:
        return None
    return None


def maybe_upload(*args, **kwargs):
    # External storage disabled; no-op
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize Kaggle news files to a unified CSV and filter by tickers/date")
    ap.add_argument("--indir", type=str, default="data/raw/kaggle_news")
    ap.add_argument("--out", type=str, default="data/raw/news.csv")
    ap.add_argument("--tickers", type=str, default="AAPL,MSFT,AMZN,GOOG,META,NVDA,JPM,TSLA")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    # External upload removed; keep args for compatibility but ignore
    ap.add_argument("--no-upload", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--container", type=str, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    indir = Path(args.indir)
    if not indir.exists():
        print(f"No such directory: {indir}", file=sys.stderr)
        return 2

    frames: list[pd.DataFrame] = []
    for p in indir.rglob("*"):
        if p.suffix.lower() not in {".csv", ".json"}:
            continue
        df = read_any(p)
        if df is None or df.empty:
            continue
        out = normalize_single_df(df, tickers=tickers, start=start, end=end, default_source=indir.name)
        if not out.empty:
            frames.append(out)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=["ticker", "published_utc", "source", "title", "link", "body"])  # type: ignore

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"news normalized rows: {len(merged)} -> {out_path}")
    # Upload disabled
    _ = maybe_upload
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
