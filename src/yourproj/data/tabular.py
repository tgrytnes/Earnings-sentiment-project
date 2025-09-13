from __future__ import annotations
from typing import Tuple
import numpy as np


def arrays_from_dataframe(df, feature_cols, target_col) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].astype(int).values.astype(np.int32)
    return X, y


def batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]

