"""FFT and power spectral density."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal as sp_signal


def compute_fft(
    x: np.ndarray,
    sample_rate: float,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided magnitude spectrum.

    Returns
    -------
    freqs, magnitude
    """
    n = len(x)
    win = sp_signal.get_window(window, n)
    xw = x * win
    spec = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag = np.abs(spec) * 2.0 / np.sum(win)
    return freqs, mag


def compute_psd(
    x: np.ndarray,
    sample_rate: float,
    nperseg: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density."""
    if nperseg is None:
        nperseg = min(len(x), 1024)
    freqs, psd = sp_signal.welch(x, fs=sample_rate, nperseg=nperseg)
    return freqs, psd
