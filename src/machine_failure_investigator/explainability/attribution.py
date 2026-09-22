"""Simple contribution attribution from feature deviations."""

from __future__ import annotations

from typing import Dict, List, Tuple


def rank_contributions(
    feature_values: Dict[str, float],
    importances: Dict[str, float],
    healthy_mean: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Score = importance * |value - healthy_mean|.
    """
    scores = []
    for name, val in feature_values.items():
        imp = importances.get(name, 0.0)
        mu = healthy_mean.get(name, val)
        scores.append((name, imp * abs(val - mu)))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores
