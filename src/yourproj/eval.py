from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
from .config import load_config

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)
    metrics_path = Path(cfg.paths['artifacts']) / "metrics.json"
    with open(metrics_path) as f:
        m = json.load(f)
    print("Metrics:", m)
    vals = {k:v for k,v in m.items() if isinstance(v, (int,float)) and k in ["accuracy","auc"] and v is not None}
    if vals:
        plt.bar(list(vals.keys()), list(vals.values()))
        plt.title("Metrics")
        fig_path = Path(cfg.paths['artifacts']) / "figures" / "metrics.pdf"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        print("Saved", fig_path)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
