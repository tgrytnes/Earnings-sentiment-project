from __future__ import annotations
from typing import Any
import pickle
import numpy as np
from .registry import register


class SKLogReg:
    def __init__(self, max_iter: int = 200, random_state: int | None = None):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SKLogReg":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "SKLogReg":
        with open(path, "rb") as f:
            model = pickle.load(f)
        obj = cls()
        obj.model = model
        return obj


@register("logreg_sklearn")
def _build_logreg_sklearn(**kwargs) -> Any:
    return SKLogReg(**kwargs)

