"""Peak picking in spectra."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import find_peaks


def find_spectral_peaks(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    n_peaks: int = 10,
    min_height: float | None = None,
    min_distance_bins: int = 5,
) -> List[Tuple[float, float]]:
    """
    Return list of (frequency_hz, magnitude) for the strongest peaks.
    """
    if min_height is None:
        min_height = np.mean(magnitude) + 1.5 * np.std(magnitude)
    peaks, props = find_peaks(magnitude, height=min_height, distance=min_distance_bins)
    if len(peaks) == 0:
        return []
    heights = props["peak_heights"]
    order = np.argsort(heights)[::-1][:n_peaks]
    result = [(float(freqs[peaks[i]]), float(heights[i])) for i in order]
    return result
