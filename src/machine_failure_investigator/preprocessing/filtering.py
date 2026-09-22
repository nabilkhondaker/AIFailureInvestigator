"""Digital filters for vibration and related signals."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def bandpass(
    x: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * sample_rate
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return x
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, x)


def highpass(x: np.ndarray, sample_rate: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sample_rate
    wn = min(max(cutoff_hz / nyq, 1e-6), 0.999)
    b, a = sp_signal.butter(order, wn, btype="high")
    return sp_signal.filtfilt(b, a, x)


def lowpass(x: np.ndarray, sample_rate: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sample_rate
    wn = min(max(cutoff_hz / nyq, 1e-6), 0.999)
    b, a = sp_signal.butter(order, wn, btype="low")
    return sp_signal.filtfilt(b, a, x)
