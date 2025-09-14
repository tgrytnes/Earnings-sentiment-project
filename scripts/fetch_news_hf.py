#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Dict

import pandas as pd

# Datasets is optional until used (to let non-network parts work without it)


# Ensure src import for optional Azure upload helper
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def iter_dataset(dataset_id: str, split: str = "train") -> Iterator[dict]:
    from datasets import load_dataset  # lazy import

    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
    except Exception:
        # Try without split
        ds = load_dataset(dataset_id, streaming=True)
    return iter(ds)


def first_present(row: dict, names: Iterable[str]):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def parse_datetime_any(v) -> pd.Timestamp | None:
    if v is None:
        return None
    try:
        # If epoch seconds
        if isinstance(v, (int, float)):
            return pd.to_datetime(v, unit="s", utc=True)
        # Strings
        s = str(v)
        # Normalize common timezone abbreviations by removing them; then localize separately
        s = re.sub(r"\b(EST|EDT|CST|CDT|MST|MDT|PST|PDT)\b", "", s).strip()
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def extract_tickers(row: dict) -> list[str]:
    # Try common ticker fields
    for key in ("tickers", "symbols"):
        if key in row and row[key] is not None:
            v = row[key]
            if isinstance(v, (list, tuple)):
                return [str(x).upper().strip() for x in v if str(x).strip()]
            if isinstance(v, str):
                parts = [s.strip() for s in re.split(r"[,\s]+", v) if s.strip()]
                return [p.upper() for p in parts]
    for key in ("ticker", "symbol"):
        if key in row and row[key] is not None and str(row[key]).strip():
            return [str(row[key]).upper().strip()]
    return []


def match_tickers_by_text(tickers: list[str], title: str, body: str) -> list[str]:
    if not tickers:
        return []
    text = f"{title} {body}".upper()
    pats = [re.escape(t) for t in sorted(set(tickers), key=len, reverse=True)]
    pat = re.compile(r"(?<![A-Z0-9])(" + "|".join(pats) + r")(?![A-Z0-9])")
    return sorted(set(pat.findall(text)))


def build_name_patterns(want_tickers: Iterable[str]) -> Dict[str, re.Pattern[str]]:
    # Basic company-name heuristics for the common 10; extendable
    synonyms = {
        "AAPL": ["APPLE"],
        "MSFT": ["MICROSOFT"],
        "AMZN": ["AMAZON"],
        "GOOG": ["ALPHABET", "GOOGLE"],
        "GOOGL": ["ALPHABET", "GOOGLE"],
        "META": ["META", "FACEBOOK"],
        "NVDA": ["NVIDIA"],
        "TSLA": ["TESLA"],
        "JPM": ["JPMORGAN", "JP MORGAN", "JPMORGAN CHASE"],
        "BAC": ["BANK OF AMERICA"],
        "NFLX": ["NETFLIX"],
    }
    pats: Dict[str, re.Pattern[str]] = {}
    for t in want_tickers:
        names = synonyms.get(t, [])
        if not names:
            continue
        pat = re.compile(r"(?i)(?<![A-Z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Z0-9])")
        pats[t] = pat
    return pats


def match_tickers_by_names(name_pats: Dict[str, re.Pattern[str]], title: str, body: str) -> list[str]:
    if not name_pats:
        return []
    text = f"{title} {body}"
    found = []
    for t, pat in name_pats.items():
        if pat.search(text):
            found.append(t)
    return found


def normalize_row(row: dict, want_tickers: set[str], infer_by_names: bool = False) -> list[dict]:
    # Core fields
    dt_val = first_present(row, ("date", "published", "published_at", "time", "datetime", "created_at"))
    ts = parse_datetime_any(dt_val)
    if ts is None:
        return []

    source = first_present(row, ("source", "publisher", "site"))
    title = first_present(row, ("title", "headline")) or ""
    link = first_present(row, ("link", "url", "news_url", "article_url", "source_url")) or ""
    body = first_present(row, ("content", "text", "article", "body", "description", "summary")) or ""

    syms = [t for t in extract_tickers(row) if t in want_tickers]
    if not syms:
        syms = match_tickers_by_text(list(want_tickers), str(title), str(body))
    if not syms and infer_by_names:
        name_pats = build_name_patterns(want_tickers)
        syms = match_tickers_by_names(name_pats, str(title), str(body))
    if not syms:
        return []

    out = []
    for t in syms:
        out.append(
            {
                "ticker": t,
                "published_utc": ts.isoformat(),
                "source": source or "",
                "title": str(title) if title is not None else "",
                "link": str(link) if link is not None else "",
                "body": str(body) if body is not None else "",
            }
        )
    return out


def normalize_row_loose(row: dict) -> list[dict]:
    """Normalize without requiring ticker/date filters. Produces at least one row per item
    if a parsable datetime exists. If no ticker is present, leaves it empty.
    """
    dt_val = first_present(row, ("date", "published", "published_at", "time", "datetime", "created_at"))
    ts = parse_datetime_any(dt_val)
    if ts is None:
        return []

    source = first_present(row, ("source", "publisher", "site"))
    title = first_present(row, ("title", "headline")) or ""
    link = first_present(row, ("link", "url", "news_url", "article_url", "source_url")) or ""
    body = first_present(row, ("content", "text", "article", "body", "description", "summary")) or ""

    syms = extract_tickers(row)
    if not syms:
        syms = [""]  # write one row with empty ticker

    out = []
    for t in syms:
        out.append(
            {
                "ticker": t,
                "published_utc": ts.isoformat(),
                "source": source or "",
                "title": str(title) if title is not None else "",
                "link": str(link) if link is not None else "",
                "body": str(body) if body is not None else "",
            }
        )
    return out


def maybe_upload(local_path: Path, blob_path: str, container: str | None, do_upload: bool) -> None:
    if not do_upload:
        return
    if not container:
        print("[upload] AZURE_BLOB_CONTAINER not set; skipping upload", file=sys.stderr)
        return
    try:
        from earnings_sentiment.storage import upload_file as azure_upload  # type: ignore

        azure_upload(str(local_path), container=container, blob_name=blob_path, overwrite=True)
        print(f"[upload] uploaded {local_path} -> {container}/{blob_path}")
    except Exception as e:
        print(f"[upload] failed: {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch filtered financial news from Hugging Face dataset")
    ap.add_argument("--dataset", default="ashraq/financial-news", help="HF dataset id")
    ap.add_argument("--split", default="train", help="Dataset split (default: train)")
    ap.add_argument("--tickers", required=False, default="", help="Comma-separated tickers to keep")
    ap.add_argument("--start", required=False, default="", help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=False, default="", help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", default="data/raw/news.csv", help="Output CSV path")
    ap.add_argument("--max-rows", type=int, default=0, help="Stop after writing this many rows (0 = no limit)")
    ap.add_argument("--max-scan", type=int, default=0, help="Stop after scanning this many input rows (0 = no limit)")
    ap.add_argument("--verbose", action="store_true", help="Print periodic progress")
    ap.add_argument("--no-filter", action="store_true", help="Do not filter by tickers/date; write first rows found")
    ap.add_argument("--infer-by-names", action="store_true", help="When filtering by tickers, also infer by company names (e.g., Microsoft -> MSFT)")
    ap.add_argument("--exclude-sources", default="", help="Comma-separated source substrings to exclude (e.g., GuruFocus,Benzinga)")
    ap.add_argument("--include-sources", default="", help="Comma-separated source substrings to include (whitelist). If set, only rows with matching sources are kept.")
    ap.add_argument("--container", default=os.environ.get("AZURE_BLOB_CONTAINER"))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Do not write/output, just count matches")
    args = ap.parse_args()

    want: list[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    want_set: Optional[set[str]] = set(want) if (want and not args.no_filter) else None
    if args.no_filter:
        start = end = None
    else:
        if not args.start or not args.end:
            print("Provide --start and --end or use --no-filter", file=sys.stderr)
            return 2
        start = pd.Timestamp(args.start, tz=timezone.utc)
        end = pd.Timestamp(args.end, tz=timezone.utc)

    rows_written = 0
    matches = 0
    scanned = 0
    out_path = Path(args.out)
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write header
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            fh.write("ticker,published_utc,source,title,link,body\n")

    # Prepare source filters
    exc = [s.strip().lower() for s in args.exclude_sources.split(",") if s.strip()]
    inc = [s.strip().lower() for s in args.include_sources.split(",") if s.strip()]

    # Iterate streaming; filter on-the-fly
    for row in iter_dataset(args.dataset, split=args.split):
        scanned += 1
        if args.verbose and scanned % 500 == 0:
            print(f"scanned: {scanned}, matches: {matches}, written: {rows_written}")
        if args.max_scan and scanned > args.max_scan:
            if args.verbose:
                print(f"stopping after max-scan={args.max_scan}")
            break
        # Early source include/exclude check
        src_val = first_present(row, ("source", "publisher", "site"))
        src_lc = str(src_val).lower() if src_val is not None else ""
        if inc and not any(s in src_lc for s in inc):
            continue
        if exc and any(s in src_lc for s in exc):
            continue
        # Normalize and filter
        if args.no_filter:
            outs = normalize_row_loose(row)
        else:
            outs = normalize_row(row, want_set, infer_by_names=args.infer_by_names)  # type: ignore[arg-type]
        if not outs:
            continue
        # Filter by date if applicable
        if not args.no_filter:
            kept = []
            for o in outs:
                try:
                    ts = pd.Timestamp(o["published_utc"])
                except Exception:
                    continue
                if start <= ts <= end:  # type: ignore[operator]
                    kept.append(o)
            if not kept:
                continue
            outs = kept
        matches += len(outs)

        if not args.dry_run:
            with open(out_path, "a", encoding="utf-8", newline="") as fh:
                for o in outs:
                    # Basic CSV escaping by replacing newlines and quotes
                    def esc(s: str) -> str:
                        s = s.replace("\n", " ").replace("\r", " ")
                        if '"' in s or ',' in s:
                            s = '"' + s.replace('"', '""') + '"'
                        return s

                    fh.write(
                        f"{o['ticker']},{o['published_utc']},{esc(str(o['source'] or ''))},{esc(o['title'])},{esc(o['link'])},{esc(o['body'])}\n"
                    )
                    rows_written += 1
            if args.max_rows and rows_written >= args.max_rows:
                break

    if not args.dry_run:
        print(f"news rows: {rows_written} -> {out_path}")
        maybe_upload(out_path, "raw/news.csv", args.container, not args.no_upload)
    else:
        print(f"dry-run matches: {matches} (scanned: {scanned})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
