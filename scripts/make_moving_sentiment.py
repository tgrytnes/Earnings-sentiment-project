#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_windows(ws: str) -> List[int]:
    return [int(w.strip()) for w in ws.split(",") if w.strip()]


def ensure_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        print(f"Wrote {path} (rows={len(df)})")
    except Exception as e:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        print(f"Parquet engine missing? Saved CSV fallback {csv} (rows={len(df)}). Error: {e}")


def build_daily(inp: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(inp)
    except Exception:
        df = pd.read_csv(inp.with_suffix(".csv"))
    if df.empty:
        return df
    # Normalize timestamp and build UTC date (naive)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_utc"]).copy()
    df["date"] = df["published_utc"].dt.tz_convert(None).dt.normalize()
    g = df.groupby("date", as_index=False).agg(
        sent_count=("headline_id", "count"),
        pos_mean=("finbert_pos", "mean"),
        neu_mean=("finbert_neu", "mean"),
        neg_mean=("finbert_neg", "mean"),
    )
    g["comp_mean"] = g["pos_mean"] - g["neg_mean"]
    return g.sort_values("date").reset_index(drop=True)


def add_moving(df: pd.DataFrame, windows: List[int], smoothing: str, weight_by_count: bool) -> pd.DataFrame:
    out = df.copy()
    cols = ["pos_mean", "neu_mean", "neg_mean", "comp_mean"]
    sdf = out.set_index("date")

    for w in windows:
        for c in cols:
            series = sdf[c]
            if weight_by_count and c.endswith("mean"):
                # Weighted by sent_count for the daily means
                weights = sdf["sent_count"].astype(float)
                if smoothing.lower() == "ema":
                    # EMA weighting inherently handles decay; apply weighted series before EMA
                    ema = (series * weights).ewm(span=w, adjust=False).mean() / weights.ewm(span=w, adjust=False).mean()
                    key = f"{c}_ema{w}"
                    out[key] = ema.values
                else:
                    # SMA weighted using rolling sum of weights
                    num = (series * weights).rolling(w, min_periods=max(2, min(5, w))).sum()
                    den = weights.rolling(w, min_periods=max(2, min(5, w))).sum()
                    key = f"{c}_sma{w}"
                    out[key] = (num / den).values
            else:
                if smoothing.lower() == "ema":
                    key = f"{c}_ema{w}"
                    out[key] = series.ewm(span=w, adjust=False).mean().values
                else:
                    key = f"{c}_sma{w}"
                    out[key] = series.rolling(w, min_periods=max(2, min(5, w))).mean().values
    return out


def normalize_to_unit_swing(
    df: pd.DataFrame,
    cols: list[str],
    scope: str = "per_series",
) -> pd.DataFrame:
    """Normalize selected columns to [-1, 1].

    scope:
      - "per_series": each target column is min-max scaled to [-1,1] using its own (min,max).
      - "global": all target columns are scaled using the raw comp_mean's (min,max).

    If a range is zero or NaN, that series is left unchanged.
    """
    out = df.copy()
    if not cols:
        return out

    def scale(series: pd.Series, mn: float, mx: float) -> pd.Series:
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return series
        rng = mx - mn
        x = 2.0 * (series.astype(float) - mn) / rng - 1.0
        return x.clip(lower=-1.0, upper=1.0)

    if scope == "global":
        if "comp_mean" not in out.columns:
            return out
        s = out["comp_mean"].astype(float)
        mn, mx = s.min(skipna=True), s.max(skipna=True)
        for c in cols:
            if c in out.columns:
                out[c] = scale(out[c], mn, mx)
        return out

    # per_series (default)
    for c in cols:
        if c in out.columns:
            s = out[c].astype(float)
            mn, mx = s.min(skipna=True), s.max(skipna=True)
            out[c] = scale(s, mn, mx)
    return out


def maybe_plot(
    df: pd.DataFrame,
    windows: List[int],
    smoothing: str,
    outdir: Path,
    composite_only: bool = False,
    hide_raw: bool = False,
    normalized: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}. Skipping plot.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    lab = f"{smoothing.upper()}-{','.join(map(str,windows))}"
    # Pick the longest window indicator if multiple
    w = max(windows)
    kcmp = f"comp_mean_{smoothing.lower()}{w}"
    if composite_only:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        if not hide_raw:
            ax.plot(df["date"], df["comp_mean"], alpha=0.25, label="comp (pos-neg)", color="#1f77b4")
        ax.plot(df["date"], df.get(kcmp, pd.Series(index=df.index)), label=f"comp {smoothing}{w}", color="#1f77b4")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_ylabel("Composite")
        ax.set_xlabel("Date (UTC)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        if normalized:
            ax.set_ylim(-1.05, 1.05)
        fig.tight_layout()
        png = outdir / f"market_sentiment_{smoothing.lower()}_{w}_composite.png"
        pdf = outdir / f"market_sentiment_{smoothing.lower()}_{w}_composite.pdf"
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
        print(f"Saved figures: {png}, {pdf}")
    else:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        kpos = f"pos_mean_{smoothing.lower()}{w}"
        kneg = f"neg_mean_{smoothing.lower()}{w}"
        kneu = f"neu_mean_{smoothing.lower()}{w}"
        ax[0].plot(df["date"], df.get(kpos, pd.Series(index=df.index)), label=f"pos {smoothing}{w}")
        ax[0].plot(df["date"], df.get(kneg, pd.Series(index=df.index)), label=f"neg {smoothing}{w}")
        ax[0].plot(df["date"], df.get(kneu, pd.Series(index=df.index)), label=f"neu {smoothing}{w}")
        ax[0].set_ylabel("Probabilities")
        ax[0].legend(loc="upper left")
        ax[0].grid(alpha=0.3)

        ax[1].plot(df["date"], df["comp_mean"], alpha=0.25, label="comp (pos-neg)")
        ax[1].plot(df["date"], df.get(kcmp, pd.Series(index=df.index)), label=f"comp {smoothing}{w}")
        ax[1].axhline(0, color="black", lw=0.8, alpha=0.5)
        ax[1].set_ylabel("Composite")
        ax[1].set_xlabel("Date (UTC)")
        ax[1].legend(loc="upper left")
        ax[1].grid(alpha=0.3)

        fig.tight_layout()
        png = outdir / f"market_sentiment_{smoothing.lower()}_{w}.png"
        pdf = outdir / f"market_sentiment_{smoothing.lower()}_{w}.pdf"
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
        print(f"Saved figures: {png}, {pdf}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build moving-average sentiment indicators from FinBERT-scored headlines")
    ap.add_argument("--inp", default="data/interim/news_scored.parquet", help="Input per-item sentiment Parquet")
    ap.add_argument("--out", default="data/interim/daily_sentiment.parquet", help="Output daily series Parquet (with indicators)")
    ap.add_argument("--windows", default="30", help="Comma-separated windows (e.g., 7,30,90)")
    ap.add_argument("--smoothing", choices=["sma","ema"], default="sma", help="Smoothing type")
    ap.add_argument("--weight-by-count", action="store_true", help="Weight moving averages by daily headline counts")
    ap.add_argument("--resample-daily", action="store_true", help="Resample to continuous daily dates (leave gaps as NaN)")
    ap.add_argument("--plot", action="store_true", help="Emit a figure under artifacts/figures/")
    ap.add_argument("--plot-composite-only", action="store_true", help="When plotting, only render the composite (pos-neg)")
    ap.add_argument("--plot-hide-raw", action="store_true", help="When plotting composite-only, hide the raw comp series")
    ap.add_argument("--normalize", action="store_true", help="Normalize composite series to [-1,1]")
    ap.add_argument(
        "--normalize-mode",
        choices=["smoothed", "all"],
        default="smoothed",
        help="Which composites to normalize: only smoothed (default) or raw+smoothed",
    )
    ap.add_argument(
        "--normalize-scope",
        choices=["per_series", "global"],
        default="per_series",
        help="Normalization basis: per-series min/max (default) or global comp_mean min/max",
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    windows = parse_windows(args.windows)

    daily = build_daily(inp)
    if daily.empty:
        ensure_parquet(daily, out)
        return 0

    if args.resample_daily:
        idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(idx).rename_axis("date").reset_index()

    enriched = add_moving(daily, windows=windows, smoothing=args.smoothing, weight_by_count=args.weight_by_count)

    # Optional normalization to [-1,1] for composite series
    if args.normalize:
        if args.normalize_mode == "smoothed":
            comp_cols = [c for c in enriched.columns if c.startswith("comp_mean_")]
        else:  # all
            comp_cols = ["comp_mean"] + [c for c in enriched.columns if c.startswith("comp_mean_")]
        enriched = normalize_to_unit_swing(enriched, comp_cols, scope=args.normalize_scope)

    ensure_parquet(enriched, out)
    if args.plot:
        maybe_plot(
            enriched,
            windows=windows,
            smoothing=args.smoothing,
            outdir=Path("artifacts/figures"),
            composite_only=args.plot_composite_only,
            hide_raw=args.plot_hide_raw,
            normalized=args.normalize,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
