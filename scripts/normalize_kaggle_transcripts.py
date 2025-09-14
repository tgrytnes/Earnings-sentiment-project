#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Any

import pandas as pd

# Ensure src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def coerce_df(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return pd.DataFrame([obj])
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        return pd.DataFrame({"value": obj})
    return pd.DataFrame()


def read_any(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".json", ".jsonl"}:
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                return pd.DataFrame(rows)
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return coerce_df(data)
        if path.suffix.lower() == ".pkl":
            try:
                obj = pd.read_pickle(path)
            except Exception:
                # Some pickles may require encoding
                obj = pd.read_pickle(path, compression=None)
            return coerce_df(obj)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "published_utc", "title", "link", "transcript"])  # type: ignore
    # Column guesses
    dt_col = first_present(df, ["date", "published", "publish_date", "datetime", "time"])
    tick_col = first_present(df, ["ticker", "symbol", "company", "company_ticker"])
    title_col = first_present(df, ["title", "headline"]) or None
    link_col = first_present(df, ["link", "url", "source_url"]) or None
    text_col = first_present(df, ["transcript", "content", "text", "article"]) or None

    out = pd.DataFrame()
    if dt_col is None or text_col is None:
        return out

    out["published_utc"] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    if out["published_utc"].dt.tz is None:
        out["published_utc"] = pd.to_datetime(df[dt_col], errors="coerce").dt.tz_localize("UTC")
    out["title"] = df[title_col] if title_col else None
    out["link"] = df[link_col] if link_col else None
    out["transcript"] = df[text_col].astype(str)
    if tick_col and tick_col in df.columns:
        out["ticker"] = df[tick_col].astype(str).str.upper().str.strip()
    else:
        out["ticker"] = None
    return out


def maybe_upload(*args, **kwargs):
    # External storage disabled; no-op
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize Kaggle transcripts to CSV and filter by tickers/date")
    ap.add_argument("--indir", type=str, default="data/raw/kaggle_transcripts")
    ap.add_argument("--out", type=str, default="data/raw/transcripts.csv")
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
    frames: list[pd.DataFrame] = []
    for p in indir.rglob("*"):
        if p.suffix.lower() not in {".csv", ".json", ".jsonl", ".pkl"}:
            continue
        df = read_any(p)
        if df is None or df.empty:
            continue
        norm = normalize_df(df)
        if norm is None or norm.empty:
            continue
        frames.append(norm)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=["ticker", "published_utc", "title", "link", "transcript"])  # type: ignore

    # Filter date window
    if not merged.empty:
        merged = merged[merged["published_utc"].between(start, end, inclusive="both")]
        if "ticker" in merged.columns and merged["ticker"].notna().any():
            merged = merged[merged["ticker"].isin(tickers)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"transcripts normalized rows: {len(merged)} -> {out_path}")
    # Upload disabled
    _ = maybe_upload
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
