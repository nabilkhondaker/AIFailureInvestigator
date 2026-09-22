"""Feature importance helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def top_features(importances: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    return items[:k]


def abnormal_features(
    feature_values: Dict[str, float],
    healthy_mean: Dict[str, float],
    healthy_std: Dict[str, float],
    z_threshold: float = 2.5,
) -> List[Tuple[str, float, float]]:
    """
    Return features whose z-score vs healthy baseline exceeds threshold.

    Each item is (name, value, z_score).
    """
    out = []
    for name, val in feature_values.items():
        mu = healthy_mean.get(name)
        sigma = healthy_std.get(name)
        if mu is None or sigma is None or sigma < 1e-12:
            continue
        z = (val - mu) / sigma
        if abs(z) >= z_threshold:
            out.append((name, val, float(z)))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out
