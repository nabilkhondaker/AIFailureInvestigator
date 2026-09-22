"""Probability calibration diagnostics."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def reliability_bins(
    y_true: np.ndarray,
    proba_positive: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Simple reliability diagram data for a binary problem."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (proba_positive >= bins[i]) & (proba_positive < bins[i + 1])
        if not np.any(mask):
            continue
        result.append(
            {
                "bin_low": float(bins[i]),
                "bin_high": float(bins[i + 1]),
                "confidence": float(np.mean(proba_positive[mask])),
                "accuracy": float(np.mean(y_true[mask])),
                "count": int(np.sum(mask)),
            }
        )
    return result
