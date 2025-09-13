from __future__ import annotations
from typing import List, Optional
import numpy as np
from .registry import register
from ..device import pick_compute_env


class TorchMLP:
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
        import torch
        import torch.nn as nn

        self.torch = torch
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        env = pick_compute_env(backend or "auto")
        self.device = torch.device(env.device)
        # Decide mixed precision
        mp_pref = ("auto" if mixed_precision is None else mixed_precision)
        self.use_mp = False
        if isinstance(mp_pref, bool):
            self.use_mp = mp_pref
        else:
            self.use_mp = (self.device.type == "cuda")

        hs = hidden_sizes or [64, 32]
        layers = []
        last = input_dim
        for h in hs:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]  # logits
        self.model = nn.Sequential(*layers).to(self.device)
        self.loss = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_mp)

        print(f"[compute] framework=torch device={self.device} mixed_precision={self.use_mp}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLP":
        t = self.torch
        X_t = t.tensor(X, dtype=t.float32, device=self.device)
        y_t = t.tensor(y.reshape(-1, 1), dtype=t.float32, device=self.device)
        n = X_t.shape[0]
        for _ in range(self.epochs):
            idx = t.randperm(n)
            X_t = X_t[idx]
            y_t = y_t[idx]
            for i in range(0, n, self.batch_size):
                xb = X_t[i : i + self.batch_size]
                yb = y_t[i : i + self.batch_size]
                self.opt.zero_grad()
                if self.use_mp:
                    with t.cuda.amp.autocast():
                        logits = self.model(xb)
                        loss = self.loss(logits, yb)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    logits = self.model(xb)
                    loss = self.loss(logits, yb)
                    loss.backward()
                    self.opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        t = self.torch
        self.model.eval()
        with t.no_grad():
            X_t = t.tensor(X, dtype=t.float32, device=self.device)
            logits = self.model(X_t)
            probs_pos = t.sigmoid(logits).squeeze(1).cpu().numpy()
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)

    def save(self, path: str) -> None:
        t = self.torch
        t.save(self.model.state_dict(), path)


@register("mlp_torch")
def _build_mlp_torch(**kwargs):
    input_dim = kwargs.pop("input_dim")
    return TorchMLP(input_dim=input_dim, **kwargs)

