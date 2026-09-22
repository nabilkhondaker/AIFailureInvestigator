"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    Seed Python, NumPy, and optionally PyTorch RNGs.

    Parameters
    ----------
    seed:
        Integer seed.
    deterministic_torch:
        If True, request deterministic CuDNN behaviour (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy Generator."""
    return np.random.default_rng(seed)
