"""Sensor noise and disturbance models."""

from __future__ import annotations

from typing import Optional

import numpy as np


def additive_gaussian(
    signal: np.ndarray,
    std: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return signal + rng.normal(0.0, std, size=signal.shape)


def pink_noise(n: int, std: float = 1.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Approximate 1/f noise via frequency-domain shaping."""
    rng = rng or np.random.default_rng()
    white = rng.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = spectrum / np.sqrt(freqs)
    colored = np.fft.irfft(spectrum, n=n)
    colored = colored / (np.std(colored) + 1e-12) * std
    return colored


def drift(
    n: int,
    rate: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Linear drift with optional small random slope perturbation."""
    rng = rng or np.random.default_rng()
    slope = rate * (1.0 + 0.1 * rng.normal())
    return slope * np.linspace(0.0, 1.0, n)
