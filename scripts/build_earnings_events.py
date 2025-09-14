#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
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


def classify_beat(actual: float, estimate: float, tol: float = 1e-6) -> tuple[str, int, float, float]:
    surprise = actual - estimate
    surprise_pct = surprise / (abs(estimate) + tol)
    if surprise > tol:
        return ("beat", 1, surprise, surprise_pct)
    if surprise < -tol:
        return ("miss", -1, surprise, surprise_pct)
    return ("meet", 0, surprise, surprise_pct)


def compute_forward_returns(p: pd.DataFrame, ticker: str, t0: pd.Timestamp) -> dict[str, float | None]:
    p_t = p[p["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    # Find t0 row and subsequent trading days by index (exact match expected)
    idx = p_t.index[p_t["date"] == t0]
    if len(idx) == 0:
        return {k: None for k in ["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d"]}
    i = int(idx[0])
    def ret_at(k: int) -> float | None:
        j = i + k
        if j >= len(p_t):
            return None
        c0 = p_t.loc[i, "close"]
        ck = p_t.loc[j, "close"]
        return float(ck / c0 - 1.0) if c0 and not pd.isna(c0) and not pd.isna(ck) else None
    return {
        "fwd_ret_1d": ret_at(1),
        "fwd_ret_3d": ret_at(3),
        "fwd_ret_5d": ret_at(5),
    }


def first_trading_day(prices_t: pd.DataFrame, announce_dt: pd.Timestamp, when: str) -> pd.Timestamp | None:
    """Return the first trading day for this ticker based on timing.

    when:
      - 'BMO': first price date >= announce calendar date
      - 'AMC': first price date > announce calendar date (next trading day)
    """
    if prices_t.empty or pd.isna(announce_dt):
        return None
    d = pd.to_datetime(announce_dt).normalize()
    dates = prices_t["date"].sort_values()
    if when == "BMO":
        sel = dates[dates >= d]
    else:  # AMC (treat as if after market on prior day)
        sel = dates[dates > d]
    return sel.iloc[0] if len(sel) else None


def pre_event_features(p: pd.DataFrame, ticker: str, t0: pd.Timestamp) -> dict[str, float | None]:
    p_t = p[p["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    # Use last available close strictly before t0
    mask = p_t["date"] < t0
    if not mask.any():
        return {k: None for k in ["px_t-1", "vol_20d", "mom_21d", "rvol_5d"]}
    last_idx = int(mask[mask].index[-1])
    px_t1 = float(p_t.loc[last_idx, "close"]) if not pd.isna(p_t.loc[last_idx, "close"]) else None
    # Volatility: std of daily returns over 20d
    p_t["ret1"] = p_t["close"].pct_change(fill_method=None)
    vol_20d = float(p_t.loc[:last_idx, "ret1"].tail(20).std()) if last_idx >= 1 else None
    # Momentum over 21 trading days (approx 1 month)
    if last_idx >= 21 and not pd.isna(p_t.loc[last_idx - 21, "close"]) and not pd.isna(p_t.loc[last_idx, "close"]):
        mom_21d = float(p_t.loc[last_idx, "close"] / p_t.loc[last_idx - 21, "close"] - 1.0)
    else:
        mom_21d = None
    # Relative volume: avg volume last 5d / last 20d
    v5 = p_t.loc[:last_idx, "volume"].tail(5).mean()
    v20 = p_t.loc[:last_idx, "volume"].tail(20).mean()
    rvol_5d = float(v5 / v20) if v5 and v20 and not pd.isna(v5) and not pd.isna(v20) else None
    return {"px_t-1": px_t1, "vol_20d": vol_20d, "mom_21d": mom_21d, "rvol_5d": rvol_5d}


def join_general_sentiment(daily: pd.DataFrame, t0: pd.Timestamp) -> dict[str, float | None]:
    # Lookback windows end day before t0
    d = daily.copy()
    d = d.sort_values("date")
    end = (pd.Timestamp(t0) - pd.Timedelta(days=1)).normalize()
    d = d[d["date"] <= end]
    if d.empty:
        return {k: None for k in [
            "gen_comp_7d", "gen_comp_14d", "gen_comp_30d",
            "gen_sent_count_30d", "gen_comp_30d_slope", "gen_comp_30d_zscore",
            "gen_comp_ema10_norm_tminus1",
        ]}
    def win(series: pd.Series, days: int) -> pd.Series:
        return series.tail(days)
    comp = d["comp_mean"]
    res: dict[str, float | None] = {}
    for w in (7, 14, 30):
        res[f"gen_comp_{w}d"] = float(win(comp, w).mean()) if len(comp) >= 1 else None
    res["gen_sent_count_30d"] = float(d["sent_count"].tail(30).sum()) if len(d) else None
    res["gen_comp_30d_slope"] = float(d["comp_30d_slope"].tail(1).iloc[0]) if "comp_30d_slope" in d.columns and len(d) else None
    res["gen_comp_30d_zscore"] = float(d["comp_30d_zscore"].tail(1).iloc[0]) if "comp_30d_zscore" in d.columns and len(d) else None
    # Add normalized EMA(10) at t-1 with 7-day max gap carry-forward
    if "comp_mean_ema10" in d.columns:
        last_date = d["date"].max()
        # Ensure last_date is within 7 calendar days of end
        if pd.notna(last_date) and (end - last_date) <= pd.Timedelta(days=7):
            try:
                res["gen_comp_ema10_norm_tminus1"] = float(d.set_index("date").loc[last_date, "comp_mean_ema10"])
            except Exception:
                res["gen_comp_ema10_norm_tminus1"] = float(d["comp_mean_ema10"].iloc[-1])
        else:
            res["gen_comp_ema10_norm_tminus1"] = None
    else:
        res["gen_comp_ema10_norm_tminus1"] = None
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Build earnings events Parquet with labels and general sentiment features")
    ap.add_argument("--prices", default="data/raw/prices.csv")
    ap.add_argument("--earnings", default="data/raw/earnings.csv")
    ap.add_argument("--daily-sent", default="data/interim/daily_sentiment.parquet")
    ap.add_argument("--out", default="data/processed/earnings_events.parquet")
    ap.add_argument("--export-min", default="data/processed/events_minimal.csv", help="Optional CSV with minimal columns for modeling")
    args = ap.parse_args()

    prices = pd.read_csv(Path(args.prices), parse_dates=["date"]) if Path(args.prices).exists() else pd.DataFrame()
    if not prices.empty:
        for col in ["open", "high", "low", "close", "volume"]:
            if col in prices.columns:
                prices[col] = pd.to_numeric(prices[col], errors="coerce")
    earnings = pd.read_csv(Path(args.earnings), parse_dates=["announce_datetime"]) if Path(args.earnings).exists() else pd.DataFrame()
    if prices.empty or earnings.empty:
        print("Missing prices or earnings input", file=sys.stderr)
        return 2

    # Load daily sentiment
    dpath = Path(args.daily_sent)
    if dpath.exists():
        try:
            daily = pd.read_parquet(dpath)
        except Exception:
            daily = pd.read_csv(dpath.with_suffix(".csv"))
            if "date" in daily:
                daily["date"] = pd.to_datetime(daily["date"])  # normalize
    else:
        daily = pd.DataFrame(columns=["date", "sent_count", "pos_mean", "neg_mean", "neu_mean", "comp_mean"])  # type: ignore

    # Normalize times
    earnings = earnings.copy()
    earnings["announce_datetime"] = pd.to_datetime(earnings["announce_datetime"], utc=True, errors="coerce").dt.tz_convert(None)
    # Determine trading-day t0 based on BMO/AMC using prices per ticker
    earnings = earnings.copy()
    earnings["bmo_amc"] = earnings.get("bmo_amc", "AMC").fillna("AMC")
    earnings_rows = []
    prices_sorted = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    for _, er in earnings.iterrows():
        tkr = str(er.get("ticker", ""))
        if not tkr:
            continue
        p_t = prices_sorted[prices_sorted["ticker"] == tkr]
        ann = er.get("announce_datetime")
        bmoflag = str(er.get("bmo_amc", "AMC")).upper()
        t0 = first_trading_day(p_t, ann, when="BMO" if bmoflag == "BMO" else "AMC")
        if t0 is None:
            continue
        er2 = er.copy()
        er2["earnings_dt"] = pd.to_datetime(t0)
        earnings_rows.append(er2)
    if not earnings_rows:
        print("No earnings rows after trading-day alignment", file=sys.stderr)
        return 2
    earnings = pd.DataFrame(earnings_rows)

    # Build rows
    rows = []
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    for _, r in earnings.iterrows():
        ticker = str(r.get("ticker", ""))
        t0 = pd.Timestamp(r.get("earnings_dt"))
        if not ticker or pd.isna(t0):
            continue
        # Market data features
        pre = pre_event_features(prices, ticker, t0)
        fwd = compute_forward_returns(prices, ticker, t0)
        # Beat classification
        label, beat_cls, surprise, surprise_pct = classify_beat(float(r.get("eps_actual", np.nan) or np.nan), float(r.get("eps_estimate", np.nan) or np.nan))
        # General sentiment
        gs = join_general_sentiment(daily, t0)

        # Normalized variants/targets
        surprise_norm = np.tanh(2.0 * surprise_pct) if surprise_pct is not None and not pd.isna(surprise_pct) else np.nan
        fwd5d = fwd.get("fwd_ret_5d")
        fwd5d_norm = np.tanh(5.0 * fwd5d) if fwd5d is not None and not pd.isna(fwd5d) else np.nan

        # Binary 5D up/down label (for classification experiments)
        if fwd.get("fwd_ret_5d") is None or pd.isna(fwd.get("fwd_ret_5d")):
            fwd5d_up = np.nan
        else:
            fwd5d_up = 1 if float(fwd.get("fwd_ret_5d")) > 0 else 0

        row = {
            "event_id": f"{ticker}-{t0.date()}",
            "ticker": ticker,
            "earnings_dt": t0,
            "earnings_dt_utc": pd.to_datetime(r.get("announce_datetime"), utc=True, errors="coerce"),
            "tod_flag": r.get("bmo_amc", "AMC"),
            "fiscal_period": r.get("fiscal_period", ""),
            "actual_eps": r.get("eps_actual", np.nan),
            "consensus_eps": r.get("eps_estimate", np.nan),
            "surprise": surprise,
            "surprise_pct": surprise_pct,
            "surprise_norm": surprise_norm,
            "beat_label": label,
            "beat_cls": beat_cls,
            **pre,
            **fwd,
            "fwd5d_up": fwd5d_up,
            "fwd5d_norm": fwd5d_norm,
            **gs,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    # Deduplicate multiple announce records that collapse to the same trading day
    # Keep rows with more complete EPS fields, then the latest announce timestamp
    if not out.empty:
        out["_eps_score"] = out[["actual_eps", "consensus_eps"]].notna().sum(axis=1)
        out = (
            out.sort_values(["ticker", "earnings_dt", "_eps_score", "earnings_dt_utc"])  # type: ignore[arg-type]
            .drop_duplicates(["ticker", "earnings_dt"], keep="last")
            .drop(columns=["_eps_score"])
            .reset_index(drop=True)
        )
    # Optional analysis window filter based on trading-day t0 (default: 2022-01-01 .. 2024-03-31)
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-03-31")
    out = out[(out["earnings_dt"] >= start) & (out["earnings_dt"] <= end)].reset_index(drop=True)
    _safe_to_parquet(out, Path(args.out))
    # Minimal export for modeling convenience
    try:
        min_cols = [
            "ticker",
            "earnings_dt",
            "gen_comp_ema10_norm_tminus1",
            "surprise_norm",
            "fwd5d_norm",
            "fwd5d_up",
        ]
        minimal = out[min_cols].copy()
        # Drop rows with missing target
        minimal = minimal.dropna(subset=["fwd5d_norm", "fwd5d_up"]).reset_index(drop=True)
        exp = Path(args.export_min)
        exp.parent.mkdir(parents=True, exist_ok=True)
        minimal.to_csv(exp, index=False)
        print(f"Wrote minimal CSV for ML: {exp} (rows={len(minimal)})")
    except Exception as e:
        print(f"Warning: could not write minimal export: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
