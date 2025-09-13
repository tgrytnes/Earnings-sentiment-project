from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    seed: int
    paths: dict
    train: dict
    compute: dict | None = None
    trainer: dict | None = None
    task: str | None = None

def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # Backward compatible: allow configs without `compute`
    if "compute" not in d:
        d["compute"] = {"backend": "auto", "mixed_precision": "auto"}
    else:
        d["compute"].setdefault("backend", "auto")
        d["compute"].setdefault("mixed_precision", "auto")
    # sensible defaults for trainer/task
    d.setdefault("task", "tabular")
    d.setdefault("trainer", {"epochs": 1, "batch_size": 1024, "lr": 1e-2})
    return Config(**d)
