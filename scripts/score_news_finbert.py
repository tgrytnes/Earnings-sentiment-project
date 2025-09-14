#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import hashlib
from typing import Optional, List

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


def _model_fingerprint(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]


def score_stub(title: pd.Series, body: Optional[pd.Series] = None) -> pd.DataFrame:
    pos_words = {"beat", "beats", "strong", "surge", "rally", "growth", "optimistic", "upbeat", "gain"}
    neg_words = {"miss", "misses", "weak", "drop", "plunge", "fall", "concern", "downgrade", "loss"}
    t = title.fillna("").astype(str).str.lower()
    b = body.fillna("").astype(str).str.lower() if body is not None else pd.Series([""] * len(t), index=t.index)
    txt = t + " " + b

    pos_hits = txt.apply(lambda s: sum(w in s for w in pos_words)).astype(float)
    neg_hits = txt.apply(lambda s: sum(w in s for w in neg_words)).astype(float)
    total = (pos_hits + neg_hits).replace(0, 1.0)
    pos = (pos_hits / total).clip(0, 1)
    neg = (neg_hits / total).clip(0, 1)
    neu = (1.0 - (pos + neg)).clip(0, 1)

    s = pos + neg + neu
    pos, neg, neu = pos / s, neg / s, neu / s

    df = pd.DataFrame({
        "finbert_pos": pos.astype(np.float32),
        "finbert_neu": neu.astype(np.float32),
        "finbert_neg": neg.astype(np.float32),
    })
    df["sentiment_label"] = df[["finbert_pos", "finbert_neu", "finbert_neg"]].idxmax(axis=1).map(
        {"finbert_pos": 1, "finbert_neu": 0, "finbert_neg": -1}
    )
    return df


def score_finbert(texts: List[str], model_name: str, batch_size: int = 32, device: int | str | None = None) -> pd.DataFrame:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        import torch  # noqa: F401
    except Exception as e:
        print(f"transformers/torch not available ({e}); using heuristic stub.")
        s = score_stub(pd.Series(texts))
        return s

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)

    # Device selection: if CUDA available and device is None, it will auto-place on CPU by default
    outputs: list[list[dict]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = pipe(batch)
        outputs.extend(res)

    # Map labels to standardized keys
    pos_list, neu_list, neg_list = [], [], []
    for scores in outputs:
        d = {s["label"].lower(): float(s["score"]) for s in scores}
        pos_list.append(d.get("positive", np.nan))
        neu_list.append(d.get("neutral", np.nan))
        neg_list.append(d.get("negative", np.nan))

    df = pd.DataFrame({
        "finbert_pos": pd.Series(pos_list, dtype=np.float32),
        "finbert_neu": pd.Series(neu_list, dtype=np.float32),
        "finbert_neg": pd.Series(neg_list, dtype=np.float32),
    })
    # If any NaNs (due to label names), row-normalize as fallback
    probs = df[["finbert_pos", "finbert_neu", "finbert_neg"]].to_numpy(dtype=float)
    row_sum = np.nansum(probs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        probs = np.divide(probs, row_sum, out=np.zeros_like(probs), where=row_sum != 0)
    df[["finbert_pos", "finbert_neu", "finbert_neg"]] = probs.astype(np.float32)
    df["sentiment_label"] = df[["finbert_pos", "finbert_neu", "finbert_neg"]].idxmax(axis=1).map(
        {"finbert_pos": 1, "finbert_neu": 0, "finbert_neg": -1}
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Score news with FinBERT to Parquet")
    ap.add_argument("--inp", default="data/raw/news_filtered.csv", help="Input normalized/filtered news CSV")
    ap.add_argument("--out", default="data/interim/news_scored.parquet", help="Output Parquet path")
    ap.add_argument("--model", default="ProsusAI/finbert", help="HF model id for FinBERT sentiment")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-body", action="store_true", help="Concatenate title + body as model input")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    df = pd.read_csv(inp)
    # Normalize common column variants from curated CSVs
    if "published_utc" in df.columns:
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    else:
        # Accept 'Date' or 'date' as the timestamp column
        if "Date" in df.columns:
            df["published_utc"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        else:
            df["published_utc"] = pd.to_datetime(df.get("date", pd.NaT), utc=True, errors="coerce")

    # Map Title->title if needed
    if "title" not in df.columns and "Title" in df.columns:
        df["title"] = df["Title"].astype(str)

    for col in ["ticker", "source", "title", "link", "body"]:
        if col not in df.columns:
            df[col] = ""
    # Provide a default source label if missing
    if (df.get("source") is not None) and (df["source"].eq("").all()):
        df["source"] = "kaggle_sp500_news"

    # If no rows, emit an empty scored file with schema and exit
    if df.empty:
        out = pd.DataFrame({
            "headline_id": pd.Series(dtype=str),
            "published_utc": pd.Series(dtype="datetime64[ns]"),
            "ticker": pd.Series(dtype=str),
            "source": pd.Series(dtype=str),
            "title": pd.Series(dtype=str),
            "link": pd.Series(dtype=str),
            "finbert_pos": pd.Series(dtype="float32"),
            "finbert_neu": pd.Series(dtype="float32"),
            "finbert_neg": pd.Series(dtype="float32"),
            "sentiment_label": pd.Series(dtype="int8"),
            "news_version": pd.Series(dtype=str),
            "model_name": pd.Series(dtype=str),
            "model_hash": pd.Series(dtype=str),
            "processed_at": pd.Series(dtype="datetime64[ns]"),
        })
        _safe_to_parquet(out, Path(args.out))
        return 0

    base_id = (
        df[["published_utc", "source", "title", "link"]]
        .astype(str)
        .agg("|".join, axis=1)
        .apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])
    )
    df["headline_id"] = base_id

    texts = (
        (df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str))
        if args.use_body else df["title"].fillna("").astype(str)
    ).tolist()

    scores = score_finbert(texts, model_name=args.model, batch_size=args.batch_size)
    out = pd.concat([df[[
        "headline_id", "published_utc", "ticker", "source", "title", "link"
    ]].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    out["news_version"] = "kaggle_or_yf_2022_2024"
    out["model_name"] = args.model
    out["model_hash"] = _model_fingerprint(args.model)
    out["processed_at"] = pd.Timestamp.utcnow()

    _safe_to_parquet(out, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
