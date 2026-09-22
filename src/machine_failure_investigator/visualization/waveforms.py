"""Waveform plots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(
    signal: np.ndarray,
    sample_rate: float,
    title: str = "Vibration waveform",
    ax=None,
    save_path: Optional[str | Path] = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure
    t = np.arange(len(signal)) / sample_rate
    ax.plot(t, signal, lw=0.8, color="C0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig, ax
