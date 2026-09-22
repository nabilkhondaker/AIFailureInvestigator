"""Basic signal cleaning."""

from __future__ import annotations

import numpy as np


def remove_dc(signal: np.ndarray) -> np.ndarray:
    return signal - np.mean(signal)


def clip_outliers(signal: np.ndarray, n_sigma: float = 6.0) -> np.ndarray:
    mu = np.mean(signal)
    sigma = np.std(signal) + 1e-12
    return np.clip(signal, mu - n_sigma * sigma, mu + n_sigma * sigma)


def replace_nan(signal: np.ndarray, value: float = 0.0) -> np.ndarray:
    out = signal.copy()
    out[~np.isfinite(out)] = value
    return out
