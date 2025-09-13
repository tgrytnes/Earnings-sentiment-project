from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from .config import load_config
from .utils import ensure_dir, save_json
from .trainer import train_and_eval
from .data.tabular import arrays_from_dataframe

def chrono_split(df: pd.DataFrame, time_col: str, test_size=0.25):
    df = df.sort_values(time_col)
    n = len(df)
    n_test = max(1, int(round(n*test_size)))
    return df.iloc[:-n_test], df.iloc[-n_test:]

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)
    feats_path = Path(cfg.paths['artifacts']) / "features.csv"
    feats = pd.read_csv(feats_path, parse_dates=['t0_date'])
    feats = feats.dropna(subset=[cfg.train['target']])
    Xcols = cfg.train['features']
    train, test = chrono_split(feats, "t0_date", cfg.train['test_size'])
    Xtr, ytr = arrays_from_dataframe(train, Xcols, cfg.train['target'])
    Xte, yte = arrays_from_dataframe(test, Xcols, cfg.train['target'])
    result = train_and_eval(Xtr, ytr, Xte, yte, cfg)
    model = result["model"]
    prob = result["prob"]
    pred = result["pred"]
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, prob)) if len(set(yte))>1 else None,
        "n_test": int(len(yte)),
        "features": Xcols,
    }
    out = Path(cfg.paths['artifacts']) / "metrics.json"
    ensure_dir(out.parent)
    save_json(metrics, out)
    # Save model checkpoint (best-effort; different formats per backend)
    ckpt_dir = Path(cfg.paths['artifacts']) / "checkpoints"
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / "model"
    try:
        # Torch uses raw path; TF SavedModel prefers directory; sklearn uses pickle file
        if hasattr(model, "save"):
            # Try a few common extensions/paths
            try:
                model.save(str(ckpt_path))
            except Exception:
                model.save(str(ckpt_path.with_suffix('.bin')))
        else:
            # Fallback: pickle
            import pickle
            with open(ckpt_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f)
    except Exception as e:
        print("Warning: failed to save model checkpoint:", e)
    print("Saved metrics to", out, metrics)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
