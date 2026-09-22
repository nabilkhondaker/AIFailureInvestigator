"""Spectrum plots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from machine_failure_investigator.signal_processing.fft import compute_fft


def plot_spectrum(
    signal: np.ndarray,
    sample_rate: float,
    title: str = "FFT magnitude spectrum",
    max_freq: Optional[float] = None,
    ax=None,
    save_path: Optional[str | Path] = None,
):
    freqs, mag = compute_fft(signal, sample_rate)
    if max_freq is not None:
        mask = freqs <= max_freq
        freqs, mag = freqs[mask], mag[mask]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure
    ax.plot(freqs, mag, lw=0.9, color="C1")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig, ax
