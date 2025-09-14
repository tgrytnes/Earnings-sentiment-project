#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_v2tone(series: pd.Series) -> pd.DataFrame:
    # V2Tone format: "tone,positive,negative,polarity,activityRefDensity,selfRefDensity"
    parts = series.astype(str).str.split(",", n=5, expand=True)
    cols = [
        parts[0].astype(float).rename("tone"),
        parts[1].astype(float).rename("pos"),
        parts[2].astype(float).rename("neg"),
        parts[3].astype(float).rename("polarity"),
        parts[4].astype(float).rename("act_ref_density"),
        parts[5].astype(float).rename("self_ref_density"),
    ]
    return pd.concat(cols, axis=1)


def pick_domain(df: pd.DataFrame) -> pd.Series:
    # Prefer SourceCommonName if present; otherwise extract domain from DocumentIdentifier
    if "SourceCommonName" in df.columns and df["SourceCommonName"].notna().any():
        return df["SourceCommonName"].astype(str)
    if "DocumentIdentifier" in df.columns:
        return (
            df["DocumentIdentifier"].astype(str)
            .str.extract(r"https?://([^/]+)/", expand=False)
            .fillna("")
        )
    return pd.Series([""] * len(df), index=df.index)


def load_gkg_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    # Required columns: DATE (YYYYMMDDhhmmss), V2Tone
    if "DATE" not in df.columns or "V2Tone" not in df.columns:
        raise ValueError(f"Missing DATE/V2Tone in {path}")
    # Parse timestamp and day
    ts = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    df["published_utc"] = ts
    df["day"] = ts.dt.date
    # Parse tone fields
    df = pd.concat([df, parse_v2tone(df["V2Tone"])], axis=1)
    # Attach domain
    df["domain"] = pick_domain(df)
    return df


def domain_filter(series: pd.Series, allow: list[str] | None, block: list[str] | None) -> pd.Series:
    s = series.astype(str).str.lower()
    keep = pd.Series(True, index=series.index)
    if allow:
        al = [a.strip().lower() for a in allow if a.strip()]
        if al:
            keep &= False
            for a in al:
                keep |= s.str.contains(a, na=False)
    if block:
        bl = [b.strip().lower() for b in block if b.strip()]
        for b in bl:
            keep &= ~s.str.contains(b, na=False)
    return keep


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("day", dropna=False)
    out = grp.agg(
        tone_mean=("tone", "mean"),
        tone_median=("tone", "median"),
        pos_mean=("pos", "mean"),
        neg_mean=("neg", "mean"),
        articles=("tone", "count"),
    ).reset_index()
    out = out.rename(columns={"day": "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date")


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate GDELT GKG CSV(s) into a daily tone index")
    ap.add_argument("--inputs", nargs="+", help="One or more GKG CSV files (exported from GDELT)")
    ap.add_argument("--out", default="data/raw/gdelt_daily_tone.csv")
    ap.add_argument("--allow-domains", default="", help="Comma-separated substrings to include (whitelist)")
    ap.add_argument("--block-domains", default="", help="Comma-separated substrings to exclude (blacklist)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    frames: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            print(f"Skipping missing: {p}")
            continue
        try:
            frames.append(load_gkg_csv(p))
        except Exception as e:
            print(f"Warning: failed to parse {p}: {e}")
            continue
    if not frames:
        print("No valid inputs.")
        return 2

    df = pd.concat(frames, ignore_index=True)
    allow = [s for s in args.allow_domains.split(",") if s.strip()]
    block = [s for s in args.block_domains.split(",") if s.strip()]
    if allow or block:
        mask = domain_filter(df["domain"], allow, block)
        df = df[mask]

    daily = aggregate_daily(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out, index=False)
    print(f"Wrote {out} (rows={len(daily)}, inputs={len(frames)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

