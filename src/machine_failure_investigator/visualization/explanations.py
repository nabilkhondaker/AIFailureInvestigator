"""Evidence bar charts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def plot_evidence_bars(
    items: List[Tuple[str, float]],
    title: str = "Evidence contributions",
    save_path: Optional[str | Path] = None,
):
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    ax.barh(names[::-1], vals[::-1], color="steelblue")
    ax.set_xlabel("Score")
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig, ax
