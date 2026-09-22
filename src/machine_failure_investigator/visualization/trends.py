"""Trend plots for RMS, temperature, severity."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_trend(
    values: Sequence[float],
    title: str = "Trend",
    ylabel: str = "Value",
    save_path: Optional[str | Path] = None,
):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(len(values)), values, marker="o", ms=3)
    ax.set_xlabel("Window index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig, ax
