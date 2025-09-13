from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import load_config
from .utils import ensure_dir
from .preprocess import align_events
from .features import build_structured_features
from .labels import make_day_ahead_label

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)
    prices = pd.DataFrame({
        "ticker": ["AAA"]*6,
        "date": pd.date_range("2025-01-01", periods=6, freq="B"),
        "open":[1,1,1,1,1,1],"high":[1,1,1,1,1,1],"low":[1,1,1,1,1,1],
        "close":[10,11,12,11,12,13],"volume":[100]*6
    })
    earnings = pd.DataFrame({
        "ticker":["AAA","AAA"],
        "announce_datetime":[pd.Timestamp("2025-01-01 09:00"), pd.Timestamp("2025-01-03 16:30")],
        "bmo_amc":["BMO","AMC"],
        "eps_actual":[1.2, 1.0],
        "eps_estimate":[1.0, 1.1]
    })
    events = align_events(prices, earnings)
    feats = build_structured_features(make_day_ahead_label(events))
    assert "y_d1" in feats.columns, "Label missing"
    assert "surprise" in feats.columns, "Feature missing"
    out = Path(cfg.paths['artifacts']) / "events_smoke.csv"
    ensure_dir(out.parent)
    feats.to_csv(out, index=False)
    print("Smoke OK — wrote", out)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
