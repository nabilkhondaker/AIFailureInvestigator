"""Confidence and abstention helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def max_proba_confidence(proba: np.ndarray) -> Tuple[int, float]:
    idx = int(np.argmax(proba))
    return idx, float(proba[idx])


def entropy_uncertainty(proba: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(proba, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def should_abstain(
    confidence: float,
    threshold: float = 0.55,
    anomaly_score: Optional[float] = None,
    anomaly_threshold: Optional[float] = None,
) -> bool:
    if confidence < threshold:
        return True
    if anomaly_score is not None and anomaly_threshold is not None:
        if anomaly_score >= anomaly_threshold and confidence < threshold + 0.1:
            return True
    return False
