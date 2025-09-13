from __future__ import annotations
from typing import List, Optional
import numpy as np
from .registry import register
from ..device import pick_compute_env


class TFMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 256,
        mixed_precision: str | bool | None = "auto",
        backend: str | None = None,
    ) -> None:
        import tensorflow as tf

        env = pick_compute_env(backend or "auto")
        # Mixed precision
        mp_pref = ("auto" if mixed_precision is None else mixed_precision)
        use_mp = False
        if isinstance(mp_pref, bool):
            use_mp = mp_pref
        else:
            use_mp = env.device != "cpu"
        if use_mp:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
            except Exception:
                pass
        print(f"[compute] framework=tensorflow device={env.device} mixed_precision={use_mp}")

        self.tf = tf
        self.epochs = epochs
        self.batch_size = batch_size
        hs = hidden_sizes or [64, 32]
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = inputs
        for h in hs:
            x = tf.keras.layers.Dense(h, activation="relu")(x)
            if dropout > 0:
                x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1)(x)  # logits
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TFMLP":
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.model.predict(X, verbose=0).reshape(-1)
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)

    def save(self, path: str) -> None:
        # Save weights only to a single file path (HDF5 or TensorFlow checkpoint)
        # If path endswith .h5, use HDF5; else, use SavedModel dir
        if path.endswith('.h5'):
            self.model.save(path, include_optimizer=False, save_format='h5')
        else:
            self.model.save(path)


@register("mlp_tf")
def _build_mlp_tf(**kwargs):
    input_dim = kwargs.pop("input_dim")
    return TFMLP(input_dim=input_dim, **kwargs)

