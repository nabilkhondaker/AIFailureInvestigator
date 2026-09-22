"""Composite diagnostic figure."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from machine_failure_investigator.visualization.spectra import plot_spectrum
from machine_failure_investigator.visualization.waveforms import plot_waveform


def plot_diagnostics(
    vibration,
    sample_rate: float,
    title: str = "Diagnostics",
    save_path: Optional[str | Path] = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    plot_waveform(vibration, sample_rate, title=f"{title} — waveform", ax=axes[0])
    plot_spectrum(vibration, sample_rate, title=f"{title} — spectrum", ax=axes[1])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig, axes
