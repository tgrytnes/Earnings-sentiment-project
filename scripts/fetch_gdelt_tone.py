#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests


def ymd_to_gdelt(dt: str) -> str:
    # Accept YYYY-MM-DD or YYYYMMDD and emit YYYYMMDDhhmmss (midnight)
    s = dt.strip()
    if len(s) == 10 and s[4] == '-' and s[7] == '-':
        d = datetime.strptime(s, "%Y-%m-%d")
    elif len(s) == 8 and s.isdigit():
        d = datetime.strptime(s, "%Y%m%d")
    else:
        raise ValueError(f"Unsupported date format: {dt}")
    return d.strftime("%Y%m%d000000")


def build_query(themes: list[str] | None, include_domains: list[str] | None, exclude_domains: list[str] | None, extra: str | None) -> str:
    parts: list[str] = []
    # Themes like ECONOMY, ECON_STOCKMARKET become theme:ECONOMY etc
    if themes:
        theme_terms = [f"theme:{t.strip()}" for t in themes if t.strip()]
        if theme_terms:
            parts.append("(" + " OR ".join(theme_terms) + ")")
    # Domain filters
    if include_domains:
        inc = [f"domain:{d.strip()}" for d in include_domains if d.strip()]
        if inc:
            parts.append("(" + " OR ".join(inc) + ")")
    if exclude_domains:
        exc = [f"-domain:{d.strip()}" for d in exclude_domains if d.strip()]
        parts.extend(exc)
    if extra and extra.strip():
        parts.append(extra.strip())
    if not parts:
        # Default to a broad finance theme if nothing provided
        parts = ["theme:ECONOMY OR theme:ECON_STOCKMARKET"]
    return " ".join(parts)


def fetch_timeline_tone(query: str, start: str, end: str) -> pd.DataFrame:
    from io import StringIO
    # First try the GKG timeseries (TimelineTone)
    ts_base = "https://api.gdeltproject.org/api/v2/gkg/timeseries"
    params = {
        "query": query,
        "mode": "TimelineTone",
        "format": "CSV",
        "startdatetime": start,
        "enddatetime": end,
    }
    url = f"{ts_base}?{urlencode(params)}"
    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        df = pd.read_csv(StringIO(r.text))
        if {"Date", "Value"}.issubset(df.columns):
            df = df.rename(columns={"Date": "date", "Value": "tone_mean"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
    # Fallback: fetch doc-level CSV and aggregate V2Tone
    doc_base = "https://api.gdeltproject.org/api/v2/gkg/gkg"
    params = {
        "query": query,
        "format": "CSV",
        "startdatetime": start,
        "enddatetime": end,
        "maxrecords": 250000,
    }
    url = f"{doc_base}?{urlencode(params)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    if not {"DATE", "V2Tone"}.issubset(df.columns):
        raise ValueError("GDELT doc CSV missing DATE/V2Tone")
    # Parse
    ts = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", errors="coerce")
    df["date"] = ts.dt.date
    # Split V2Tone (tone is first field)
    tone = df["V2Tone"].astype(str).str.split(",", n=1, expand=True)[0].astype(float)
    df["tone"] = tone
    out = df.groupby("date")["tone"].mean().reset_index().rename(columns={"tone": "tone_mean"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Download GDELT TimelineTone CSV and save as daily index")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD or YYYYMMDD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD or YYYYMMDD")
    ap.add_argument("--themes", default="ECONOMY,ECON_STOCKMARKET", help="Comma-separated GDELT themes (e.g., ECONOMY,ECON_STOCKMARKET)")
    ap.add_argument("--include-domains", default="", help="Comma-separated domains to include (e.g., reuters.com,bloomberg.com)")
    ap.add_argument("--exclude-domains", default="", help="Comma-separated domains to exclude (e.g., zacks.com,gurufocus.com)")
    ap.add_argument("--extra", default="sourcelang:english", help="Extra query terms (e.g., sourcelang:english)")
    ap.add_argument("--out", default="data/raw/gdelt_daily_tone.csv")
    args = ap.parse_args()

    start = ymd_to_gdelt(args.start)
    end = ymd_to_gdelt(args.end)
    themes = [t for t in args.themes.split(",") if t.strip()]
    inc = [d for d in args.include_domains.split(",") if d.strip()]
    exc = [d for d in args.exclude_domains.split(",") if d.strip()]
    q = build_query(themes, inc, exc, args.extra)

    try:
        df = fetch_timeline_tone(q, start, end)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
