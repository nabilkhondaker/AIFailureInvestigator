"""Simple class balancing helpers."""

from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np


def class_weights(y: np.ndarray) -> dict:
    counts = Counter(y.tolist())
    n = len(y)
    n_classes = len(counts)
    return {c: n / (n_classes * cnt) for c, cnt in counts.items()}


def undersample(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        indices.append(idx)
    sel = np.concatenate(indices)
    rng.shuffle(sel)
    return X[sel], y[sel]
