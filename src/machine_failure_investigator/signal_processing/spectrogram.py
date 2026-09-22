"""Spectrogram computation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal as sp_signal


def compute_spectrogram(
    x: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns frequencies, times, spectrogram magnitude (Sxx).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = sp_signal.spectrogram(
        x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, mode="magnitude"
    )
    return f, t, Sxx
