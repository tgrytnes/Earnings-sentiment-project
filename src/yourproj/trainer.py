from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .models import build_model


def train_and_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    cfg: Any,
) -> Dict[str, Any]:
    model_name = cfg.train.get("model", "logreg_sklearn")
    model_args = dict(cfg.train.get("model_args", {}))

    # Inject common args
    input_dim = Xtr.shape[1]
    # Trainer params (epochs, batch_size, lr) for DL models
    trainer = cfg.trainer or {}
    if model_name in ("mlp_torch", "mlp_tf"):
        model_args.setdefault("epochs", int(trainer.get("epochs", 10)))
        model_args.setdefault("batch_size", int(trainer.get("batch_size", 256)))
        model_args.setdefault("lr", float(trainer.get("lr", 1e-3)))
        # compute prefs
        comp = cfg.compute or {}
        model_args.setdefault("mixed_precision", comp.get("mixed_precision", "auto"))
        model_args.setdefault("backend", comp.get("backend", "auto"))
        model_args.setdefault("input_dim", input_dim)

    model = build_model(model_name, input_dim=input_dim, **model_args)
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "model": model,
        "pred": pred,
        "prob": prob,
    }

