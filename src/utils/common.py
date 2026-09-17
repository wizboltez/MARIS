"""Shared helpers: reproducible seeding and device selection."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch RNGs for reproducible splits/training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "auto") -> str:
    """Resolve the compute device, falling back to CPU when no GPU is present."""
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "0"  # ultralytics expects a CUDA device index/string
    return "cpu"
