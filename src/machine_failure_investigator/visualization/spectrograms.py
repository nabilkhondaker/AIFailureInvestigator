"""Spectrogram plots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from machine_failure_investigator.signal_processing.spectrogram import compute_spectrogram


def plot_spectrogram(
    signal,
    sample_rate: float,
    title: str = "Spectrogram",
    save_path: Optional[str | Path] = None,
):
    f, t, Sxx = compute_spectrogram(signal, sample_rate)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(t, f, Sxx, shading="auto", cmap="magma")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Magnitude")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig, ax
