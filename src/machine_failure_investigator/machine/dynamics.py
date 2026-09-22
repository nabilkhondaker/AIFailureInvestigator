"""Simplified dynamic response helpers for the simulator."""

from __future__ import annotations

import numpy as np


def harmonic_series(
    t: np.ndarray,
    f0: float,
    n_harmonics: int,
    amplitudes: np.ndarray | None = None,
    phases: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sum of sinusoids at integer multiples of ``f0``.

    Parameters
    ----------
    t:
        Time vector (seconds).
    f0:
        Fundamental frequency (Hz).
    n_harmonics:
        Number of harmonics including the fundamental.
    amplitudes, phases:
        Optional per-harmonic amplitudes and phases (radians).
    """
    if amplitudes is None:
        amplitudes = 1.0 / np.arange(1, n_harmonics + 1)
    if phases is None:
        phases = np.zeros(n_harmonics)
    signal = np.zeros_like(t, dtype=float)
    for k in range(n_harmonics):
        signal += amplitudes[k] * np.sin(2 * np.pi * (k + 1) * f0 * t + phases[k])
    return signal


def resonance_gain(freq_hz: float, natural_hz: float, damping: float) -> float:
    """Simple single-DOF magnitude response gain."""
    if natural_hz <= 0:
        return 1.0
    r = freq_hz / natural_hz
    denom = np.sqrt((1 - r**2) ** 2 + (2 * damping * r) ** 2)
    return 1.0 / max(denom, 1e-9)


def damped_transient(t: np.ndarray, f_hz: float, decay: float, amplitude: float) -> np.ndarray:
    """Exponentially decaying sinusoid (impact-like transient)."""
    return amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * f_hz * t)
