"""Windowing utilities."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def sliding_windows(
    x: np.ndarray,
    window_size: int,
    hop: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (start_index, window) pairs."""
    n = len(x)
    for start in range(0, n - window_size + 1, hop):
        yield start, x[start : start + window_size]


def hann_window(n: int) -> np.ndarray:
    return np.hanning(n)
