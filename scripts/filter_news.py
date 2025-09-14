#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def compile_regex(patterns: list[str]) -> re.Pattern[str]:
    joined = "|".join(f"({p})" for p in patterns if p)
    return re.compile(joined, flags=re.IGNORECASE) if joined else re.compile(r"^$")


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter low-signal financial news rows by source and title patterns")
    ap.add_argument("--inp", default="data/raw/news.csv", help="Input CSV path")
    ap.add_argument("--out", default="data/raw/news_filtered.csv", help="Output CSV path")
    ap.add_argument("--exclude-sources", default="GuruFocus", help="Comma-separated substrings to exclude from source (case-insensitive)")
    ap.add_argument(
        "--exclude-title-rx",
        default=r"\b(buys?|sells?|bought|sold)\b|analyst\s*blog|research\s*daily|value\s*trader|highlights",
        help="Regex for titles to exclude (case-insensitive)",
    )
    ap.add_argument("--min-title-len", type=int, default=12, help="Drop titles shorter than this length")
    ap.add_argument("--require-ticker", action="store_true", help="Keep only rows with non-empty ticker")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"Input not found: {inp}")
        return 2
    df = pd.read_csv(inp)
    n0 = len(df)

    # Normalize columns that might be missing
    for col in ["source", "title", "ticker"]:
        if col not in df.columns:
            df[col] = ""

    # Exclude by source substring
    excludes = [s.strip().lower() for s in args.exclude_sources.split(",") if s.strip()]
    if excludes:
        mask_src = pd.Series(False, index=df.index)
        s = df["source"].astype(str).str.lower()
        for sub in excludes:
            mask_src |= s.str.contains(re.escape(sub))
        df = df[~mask_src]

    # Exclude by title regex
    rx = re.compile(args.exclude_title_rx, flags=re.IGNORECASE)
    df = df[~df["title"].astype(str).str.contains(rx)]

    # Length filter
    if args.min_title_len > 0:
        df = df[df["title"].astype(str).str.len() >= args.min_title_len]

    # Require ticker if requested
    if args.require_ticker:
        df = df[df["ticker"].astype(str).str.strip() != ""]

    n1 = len(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Filtered {n0} -> {n1} rows. Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

