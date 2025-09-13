"""
Back-compat shim for prior imports.

Prefer using the new registry/trainer:
  from .models import build_model
  from .trainer import train_and_eval

The older train_logreg(X, y, ...) is preserved below for compatibility,
delegating to the sklearn logreg implementation.
"""
from typing import Any
import numpy as np
from .models.registry import build_model


def train_logreg(X: Any, y: Any, backend: str | None = None, mixed_precision: str | bool | None = None):
    X = np.asarray(X)
    y = np.asarray(y)
    model = build_model("logreg_sklearn")
    return model.fit(X, y)
