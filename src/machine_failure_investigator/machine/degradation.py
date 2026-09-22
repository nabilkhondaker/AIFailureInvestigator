"""Progressive degradation trajectory generators."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def linear_degradation(
    n_steps: int,
    start: float = 0.0,
    end: float = 1.0,
) -> np.ndarray:
    """Linear severity trajectory from ``start`` to ``end``."""
    return np.linspace(start, end, n_steps)


def sigmoid_degradation(
    n_steps: int,
    midpoint: float = 0.5,
    steepness: float = 10.0,
    end: float = 1.0,
) -> np.ndarray:
    """Sigmoid-shaped severity increase (slow start, rapid middle, plateau)."""
    x = np.linspace(0.0, 1.0, n_steps)
    s = 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
    s = (s - s.min()) / max(s.max() - s.min(), 1e-12)
    return s * end


def piecewise_degradation(
    n_steps: int,
    milestones: Optional[list[tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Piecewise-linear severity path.

    ``milestones`` is a list of (fraction_of_timeline, severity) pairs.
    """
    if milestones is None:
        milestones = [(0.0, 0.0), (0.4, 0.15), (0.7, 0.55), (1.0, 1.0)]
    fracs = np.array([m[0] for m in milestones])
    vals = np.array([m[1] for m in milestones])
    x = np.linspace(0.0, 1.0, n_steps)
    return np.interp(x, fracs, vals)


def apply_noise_to_trajectory(
    trajectory: np.ndarray,
    noise_std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add small noise and clip to [0, 1]."""
    rng = rng or np.random.default_rng()
    noisy = trajectory + rng.normal(0.0, noise_std, size=trajectory.shape)
    return np.clip(noisy, 0.0, 1.0)
