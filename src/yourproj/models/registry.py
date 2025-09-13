from __future__ import annotations
from typing import Any, Dict, Callable


_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str):
    def deco(fn: Callable[..., Any]):
        _REGISTRY[name] = fn
        return fn
    return deco


def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


# Import side-effects to populate registry
from . import sklearn_logreg  # noqa: F401
from . import torch_mlp  # noqa: F401
from . import tf_mlp  # noqa: F401

