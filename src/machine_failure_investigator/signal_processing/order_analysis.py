"""Simple order-related features (relative to shaft speed)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from machine_failure_investigator.signal_processing.fft import compute_fft


def order_amplitudes(
    x: np.ndarray,
    sample_rate: float,
    rpm: float,
    orders: tuple[float, ...] = (1.0, 2.0, 3.0, 0.5),
    bandwidth_hz: float = 2.0,
) -> Dict[str, float]:
    """
    Estimate amplitude near integer (or fractional) orders of rotation.

    For each order ``k``, integrate magnitude in a band around ``k * f0``.
    """
    f0 = rpm / 60.0
    freqs, mag = compute_fft(x, sample_rate)
    out: Dict[str, float] = {}
    for k in orders:
        center = k * f0
        mask = (freqs >= center - bandwidth_hz) & (freqs <= center + bandwidth_hz)
        key = f"order_{k:g}"
        out[key] = float(np.sum(mag[mask])) if np.any(mask) else 0.0
    return out
