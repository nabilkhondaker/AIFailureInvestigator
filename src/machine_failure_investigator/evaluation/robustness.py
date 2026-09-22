"""Helpers for noise-robustness evaluation."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def evaluate_under_noise(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    noise_levels: List[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    seed: int = 42,
) -> Dict[float, float]:
    rng = np.random.default_rng(seed)
    results = {}
    for sigma in noise_levels:
        Xn = X + rng.normal(0.0, sigma, size=X.shape)
        pred = predict_fn(Xn)
        results[sigma] = float(metric_fn(y, pred))
    return results
