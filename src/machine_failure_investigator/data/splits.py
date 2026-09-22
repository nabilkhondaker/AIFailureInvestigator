"""Split utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def train_val_test_indices(
    n: int,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train)
    n_val = int(n * val)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
