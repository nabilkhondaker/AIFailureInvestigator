"""Normalization helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + eps)


def minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + eps)


def standardize_features(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise standardization.

    Returns transformed array, mean, and std (for reuse at inference).
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std
