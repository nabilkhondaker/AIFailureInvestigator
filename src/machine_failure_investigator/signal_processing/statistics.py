"""Time-domain statistical features for vibration diagnostics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import kurtosis, skew


def time_stats(x: np.ndarray) -> Dict[str, float]:
    """
    Classical condition-monitoring statistics.

    RMS captures overall energy. Crest factor and kurtosis are sensitive
    to impulsive content (e.g. early bearing defects). Skewness can reflect
    asymmetry from looseness or nonlinear contact.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {k: 0.0 for k in (
            "rms", "peak", "peak_to_peak", "std", "variance",
            "skewness", "kurtosis", "crest_factor", "shape_factor", "impulse_factor",
        )}
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    p2p = float(np.max(x) - np.min(x))
    std = float(np.std(x))
    var = float(np.var(x))
    sk = float(skew(x))
    ku = float(kurtosis(x, fisher=True))
    mean_abs = float(np.mean(np.abs(x))) + 1e-12
    crest = peak / (rms + 1e-12)
    shape = rms / mean_abs
    impulse = peak / mean_abs
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": p2p,
        "std": std,
        "variance": var,
        "skewness": sk,
        "kurtosis": ku,
        "crest_factor": crest,
        "shape_factor": shape,
        "impulse_factor": impulse,
    }
