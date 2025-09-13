from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import load_config
from .utils import ensure_dir
from .labels import make_day_ahead_label

def build_structured_features(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    if {'eps_actual','eps_estimate'}.issubset(df.columns):
        df['surprise'] = df['eps_actual'] - df['eps_estimate']
    else:
        df['surprise'] = 0.0
    return df

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)
    inp = Path(cfg.paths['artifacts']) / "events.csv"
    out = Path(cfg.paths['artifacts']) / "features.csv"
    df = pd.read_csv(inp, parse_dates=['t0_date'])
    df = make_day_ahead_label(df)
    feats = build_structured_features(df)
    ensure_dir(out.parent)
    feats.to_csv(out, index=False)
    print(f"Saved features to {out} (rows={len(feats)})")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
