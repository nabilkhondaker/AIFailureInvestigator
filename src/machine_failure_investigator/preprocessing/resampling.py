"""Resampling utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    if abs(orig_fs - target_fs) < 1e-9:
        return x
    n_out = int(len(x) * target_fs / orig_fs)
    return sp_signal.resample(x, n_out)
