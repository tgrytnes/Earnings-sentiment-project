#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path.with_suffix(".csv"))
    else:
        df = pd.read_csv(path)
    # Parse date
    if "earnings_dt" in df.columns:
        df["earnings_dt"] = pd.to_datetime(df["earnings_dt"], errors="coerce")
    return df


def train_eval_lr(df: pd.DataFrame, split_date: str, features: list[str], target: str) -> dict:
    # Drop rows with missing features or target
    d = df.dropna(subset=features + [target]).copy()
    d = d.sort_values("earnings_dt").reset_index(drop=True)

    # Time-based split
    sd = pd.Timestamp(split_date)
    tr = d[d["earnings_dt"] < sd]
    te = d[d["earnings_dt"] >= sd]
    if tr.empty or te.empty:
        raise RuntimeError(f"Empty train/test after split: train={len(tr)}, test={len(te)}")

    Xtr = tr[features].to_numpy(dtype=float)
    ytr = tr[target].astype(int).to_numpy()
    Xte = te[features].to_numpy(dtype=float)
    yte = te[target].astype(int).to_numpy()

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(Xtr, ytr)

    # Predictions
    yhat_tr = pipe.predict(Xtr)
    yhat_te = pipe.predict(Xte)
    proba_tr = pipe.predict_proba(Xtr)[:, 1]
    proba_te = pipe.predict_proba(Xte)[:, 1]

    # Metrics
    def safe_auc(ytrue: np.ndarray, p: np.ndarray) -> float | None:
        try:
            if len(np.unique(ytrue)) < 2:
                return None
            return float(roc_auc_score(ytrue, p))
        except Exception:
            return None

    metrics = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "features": features,
        "acc_train": float(accuracy_score(ytr, yhat_tr)),
        "acc_test": float(accuracy_score(yte, yhat_te)),
        "auc_train": safe_auc(ytr, proba_tr),
        "auc_test": safe_auc(yte, proba_te),
        "confusion_test": confusion_matrix(yte, yhat_te).tolist(),
        "report_test": classification_report(yte, yhat_te, zero_division=0),
    }

    # Attach predictions for test
    pred_df = te[["ticker", "earnings_dt"]].copy()
    for i, f in enumerate(features):
        pred_df[f] = Xte[:, i]
    pred_df[target] = yte
    pred_df["y_pred"] = yhat_te
    pred_df["y_proba"] = proba_te

    return {"metrics": metrics, "pred_df": pred_df}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Logistic Regression baseline on events_minimal.csv")
    ap.add_argument("--data", default="data/processed/events_minimal.csv")
    ap.add_argument("--split-date", default="2024-01-01", help="Test set starts on or after this date")
    ap.add_argument("--outdir", default="artifacts/metrics")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data))
    # Classification target: fwd5d_up
    target = "fwd5d_up"

    runs: list[tuple[str, list[str]]] = [
        ("lr_surprise_only", ["surprise_norm"]),
        ("lr_surprise_plus_sent", ["surprise_norm", "gen_comp_ema10_norm_tminus1"]),
    ]

    summary_rows = []
    for name, feats in runs:
        res = train_eval_lr(df, args.split_date, feats, target)
        # Save metrics JSON
        with (outdir / f"{name}.json").open("w") as f:
            json.dump(res["metrics"], f, indent=2)
        # Save classification report text
        (outdir / f"{name}_report.txt").write_text(res["metrics"]["report_test"])
        # Save predictions CSV
        res["pred_df"].to_csv(outdir / f"{name}_predictions.csv", index=False)

        m = res["metrics"]
        summary_rows.append(
            {
                "run": name,
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "acc_test": m["acc_test"],
                "auc_test": m["auc_test"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(outdir / "lr_summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Wrote metrics to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

