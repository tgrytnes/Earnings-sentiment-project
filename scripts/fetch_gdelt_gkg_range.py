#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd
import requests
import zipfile


MASTERLIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GKG_SUFFIX = ".gkg.csv.zip"


def parse_ymd(s: str) -> datetime:
    s = s.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
    raise ValueError(f"Invalid date: {s}")


def iter_masterfile_urls() -> Iterator[str]:
    with requests.get(MASTERLIST, stream=True, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            parts = line.split(" ")
            url = parts[-1]
            if url.endswith(GKG_SUFFIX):
                yield url


def ts_from_url(url: str) -> Optional[datetime]:
    # URLs contain .../YYYYMMDDHHMMSS.gkg.csv.zip
    m = re.search(r"/(\d{14})\.gkg\.csv\.zip$", url)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def within(ts: datetime, start: datetime, end: datetime) -> bool:
    return start <= ts <= end


def themes_regex(themes: list[str]) -> re.Pattern[str]:
    pats = [re.escape(t.strip()) for t in themes if t.strip()]
    return re.compile("|".join(pats), flags=re.IGNORECASE) if pats else re.compile(r"^$")


def read_gkg_from_zip(content: bytes, want_cols: list[int]) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(content))
    members = zf.namelist()
    if not members:
        return pd.DataFrame()
    with zf.open(members[0]) as fh:
        # GKG v2 is tab-delimited, no header
        df = pd.read_csv(fh, sep="\t", header=None, quoting=3, low_memory=False)
    # Subset to columns we use; tolerate missing by clipping
    max_idx = df.shape[1] - 1
    cols = [c for c in want_cols if c <= max_idx]
    return df.iloc[:, cols]


@dataclass
class Counters:
    files: int = 0
    rows: int = 0
    kept: int = 0


def aggregate_range(start: datetime, end: datetime, themes: list[str], allow_domains: list[str], block_domains: list[str], limit_files: int | None = None) -> pd.DataFrame:
    # Prepare filters
    rx_themes = themes_regex(themes)
    allow = [a.strip().lower() for a in allow_domains if a.strip()]
    block = [b.strip().lower() for b in block_domains if b.strip()]

    # Column indices (0-based) seen in GKG v2 public schema examples
    # 1: DATE, 3: SourceCommonName, 4: DocumentIdentifier, 23: V2Themes, 34: V2Tone
    IDX_DATE, IDX_SOURCE, IDX_DOCID, IDX_THEMES, IDX_TONE = 1, 3, 4, 23, 34
    want_cols = sorted({IDX_DATE, IDX_SOURCE, IDX_DOCID, IDX_THEMES, IDX_TONE})

    daily_vals: List[pd.Series] = []
    ctr = Counters()

    for url in iter_masterfile_urls():
        ts = ts_from_url(url)
        if ts is None or not within(ts, start, end):
            continue
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception:
            continue

        try:
            df = read_gkg_from_zip(resp.content, want_cols)
        except Exception:
            continue
        if df.empty:
            continue

        # Map columns present to names by position
        cols = df.columns.tolist()
        name_map = {}
        for c in cols:
            if c == IDX_DATE:
                name_map[c] = "DATE"
            elif c == IDX_SOURCE:
                name_map[c] = "Source"
            elif c == IDX_DOCID:
                name_map[c] = "DocID"
            elif c == IDX_THEMES:
                name_map[c] = "Themes"
            elif c == IDX_TONE:
                name_map[c] = "V2Tone"
        df = df.rename(columns=name_map)

        # Basic filters
        if "Themes" in df:
            df = df[df["Themes"].astype(str).str.contains(rx_themes, na=False)]
        if df.empty:
            continue

        if allow or block:
            # Prefer Source (common name) else extract domain from DocID
            if "Source" in df and df["Source"].notna().any():
                s = df["Source"].astype(str).str.lower()
            elif "DocID" in df:
                s = df["DocID"].astype(str).str.extract(r"https?://([^/]+)/", expand=False).fillna("").str.lower()
            else:
                s = pd.Series([""] * len(df))
            if allow:
                keep = False
                for a in allow:
                    keep |= s.str.contains(re.escape(a), na=False)
                df = df[keep]
            for b in block:
                df = df[~s.str.contains(re.escape(b), na=False)]
            if df.empty:
                continue

        # Parse date
        if "DATE" in df:
            t = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", errors="coerce")
            day = t.dt.date
        else:
            # Fallback to the file timestamp's day
            day = pd.Series([ts.date()] * len(df))

        # Extract tone mean (first field in V2Tone)
        if "V2Tone" not in df:
            continue
        tone = pd.to_numeric(df["V2Tone"].astype(str).str.split(",", n=1, expand=True)[0], errors="coerce")
        per_file = pd.DataFrame({"day": day, "tone": tone})
        per_file = per_file.dropna(subset=["tone"])  # drop rows without tone
        if per_file.empty:
            continue

        ctr.files += 1
        ctr.rows += int(len(df))
        ctr.kept += int(len(per_file))

        daily_vals.append(per_file.groupby("day")["tone"].mean())

        if limit_files and ctr.files >= limit_files:
            break

    if not daily_vals:
        return pd.DataFrame(columns=["date", "tone_mean", "articles"])  # type: ignore

    mean_by_day = pd.concat(daily_vals, axis=0).groupby(level=0).mean().reset_index()
    mean_by_day = mean_by_day.rename(columns={"day": "date", "tone": "tone_mean"})
    mean_by_day["date"] = pd.to_datetime(mean_by_day["date"])  # ensure datetime
    return mean_by_day.sort_values("date")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch GDELT 2.0 GKG files for a date range and aggregate daily tone")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or YYYYMMDD")
    ap.add_argument("--themes", default="ECONOMY,ECON_STOCKMARKET,ECON_INFLATION", help="Comma-separated themes to keep")
    ap.add_argument("--allow-domains", default="", help="Comma-separated domain substrings to include")
    ap.add_argument("--block-domains", default="", help="Comma-separated domain substrings to exclude")
    ap.add_argument("--limit-files", type=int, default=0, help="Process at most this many files (0 = no limit)")
    ap.add_argument("--out", default="data/raw/gdelt_daily_tone.csv")
    args = ap.parse_args()

    start = parse_ymd(args.start)
    end = parse_ymd(args.end)
    if end < start:
        print("end must be >= start", file=sys.stderr)
        return 2

    themes = [t for t in args.themes.split(",") if t.strip()]
    allow = [d for d in args.allow_domains.split(",") if d.strip()]
    block = [d for d in args.block_domains.split(",") if d.strip()]
    limit = args.limit_files or None

    df = aggregate_range(start, end, themes, allow, block, limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

