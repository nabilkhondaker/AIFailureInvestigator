"""Time-frequency features (summary statistics of spectrogram)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from machine_failure_investigator.signal_processing.spectrogram import compute_spectrogram


def extract_time_frequency(
    vibration: np.ndarray,
    sample_rate: float,
) -> Dict[str, float]:
    f, t, Sxx = compute_spectrogram(vibration, sample_rate, nperseg=min(256, len(vibration)))
    if Sxx.size == 0:
        return {"tf_mean": 0.0, "tf_std": 0.0, "tf_max": 0.0}
    return {
        "tf_mean": float(np.mean(Sxx)),
        "tf_std": float(np.std(Sxx)),
        "tf_max": float(np.max(Sxx)),
        "tf_energy": float(np.sum(Sxx**2)),
    }
