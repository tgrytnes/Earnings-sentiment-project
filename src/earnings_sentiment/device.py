from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ComputeEnv:
    framework: str  # "torch" | "tensorflow" | "sklearn"
    device: str     # e.g., "cuda:0", "mps", "gpu:0", "cpu"
    detail: Optional[str] = None  # e.g., "cuda", "rocm", "mps"


def _torch_env() -> Optional[ComputeEnv]:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    # Prefer CUDA/ROCm, then MPS, then CPU
    if torch.cuda.is_available():
        # On ROCm builds, torch.version.hip is not None and device is still "cuda"
        detail = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return ComputeEnv(framework="torch", device="cuda:0", detail=detail)
    # Apple Silicon Metal Performance Shaders
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ComputeEnv(framework="torch", device="mps", detail="mps")
    # CPU fallback for torch
    return ComputeEnv(framework="torch", device="cpu", detail="cpu")


def _tf_env() -> Optional[ComputeEnv]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return None

    gpus = []
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        gpus = []
    # Enable memory growth to avoid TF reserving all GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    if gpus:
        # Covers CUDA, ROCm, and Apple tf-metal plugin; TensorFlow abstracts it as GPU
        return ComputeEnv(framework="tensorflow", device="gpu:0", detail="gpu")
    return ComputeEnv(framework="tensorflow", device="cpu", detail="cpu")


def pick_compute_env(prefer: str = "auto") -> ComputeEnv:
    prefer = (prefer or "auto").lower()

    # Explicit preferences
    if prefer == "torch":
        env = _torch_env()
        if env is not None:
            return env
    if prefer in ("tf", "tensorflow"):
        env = _tf_env()
        if env is not None:
            return env
    if prefer == "cpu":
        # Force CPU with sklearn
        return ComputeEnv(framework="sklearn", device="cpu", detail="cpu")

    # Auto: prefer GPU with PyTorch, then GPU with TensorFlow, then CPU sklearn
    env = _torch_env()
    if env is not None and env.device != "cpu":
        return env
    env_tf = _tf_env()
    if env_tf is not None and env_tf.device != "cpu":
        return env_tf
    # CPU fallback: prefer sklearn to avoid heavy DL deps on CPU unless explicitly chosen
    return ComputeEnv(framework="sklearn", device="cpu", detail="cpu")
