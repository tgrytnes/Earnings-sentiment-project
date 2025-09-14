#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute stats for general sentiment features before earnings")
    ap.add_argument("--events", default="data/processed/earnings_events.parquet")
    args = ap.parse_args()

    p = Path(args.events)
    if not p.exists():
        print(f"Events Parquet not found: {p}")
        return 2

    try:
        df = pd.read_parquet(p)
    except Exception:
        df = pd.read_csv(p)

    sent_cols = [
        "gen_comp_7d",
        "gen_comp_14d",
        "gen_comp_30d",
        "gen_sent_count_30d",
        "gen_comp_30d_slope",
        "gen_comp_30d_zscore",
    ]

    avail = df[sent_cols].notna().any(axis=1)
    print(f"Events: total={len(df)}, with_any_sentiment={int(avail.sum())}")

    sub = df.loc[avail, sent_cols]
    print("\nOverall sentiment feature stats (mean, std, p25, median, p75):")
    for c in sent_cols:
        s = sub[c].dropna()
        if s.empty:
            print(f"- {c}: no data")
            continue
        q = s.quantile([0.25, 0.5, 0.75])
        print(
            f"- {c}: mean={s.mean():.4f}, std={s.std():.4f}, "
            f"p25={q.iloc[0]:.4f}, p50={q.iloc[1]:.4f}, p75={q.iloc[2]:.4f}, n={s.shape[0]}"
        )

    if "beat_label" in df.columns:
        print("\nBy beat_label (mean ± std | n):")
        for c in sent_cols:
            rows = []
            for k, g in df.groupby("beat_label", dropna=False):
                s = g[c].dropna()
                if len(s) > 0:
                    rows.append((str(k), s.mean(), s.std(), len(s)))
            if not rows:
                print(f"- {c}: no data by group")
                continue
            row_txt = ", ".join([f"{k}={m:.4f}±{sd:.4f} | n={n}" for k, m, sd, n in rows])
            print(f"- {c}: {row_txt}")

    if "fwd_ret_5d" in df.columns:
        print("\nCorrelation with 5D forward return (Pearson r):")
        valid = df[["fwd_ret_5d"] + sent_cols].dropna()
        if len(valid) >= 5:
            for c in sent_cols:
                r = valid["fwd_ret_5d"].corr(valid[c])
                print(f"- {c}: r={r:.4f} (n={len(valid)})")
        else:
            print("- insufficient overlapping data")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

