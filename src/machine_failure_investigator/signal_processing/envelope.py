"""Hilbert envelope and envelope spectrum (useful for bearing diagnostics)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal as sp_signal

from machine_failure_investigator.signal_processing.fft import compute_fft


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope via Hilbert transform."""
    analytic = sp_signal.hilbert(x)
    return np.abs(analytic)


def envelope_spectrum(
    x: np.ndarray,
    sample_rate: float,
    band: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Envelope spectrum, optionally after band-pass filtering.

    Bearing faults often modulate a high-frequency carrier; the envelope
    spectrum reveals the modulating fault frequencies (BPFO, BPFI, etc.).
    """
    y = x
    if band is not None:
        low, high = band
        nyq = 0.5 * sample_rate
        wn = [max(low / nyq, 1e-6), min(high / nyq, 0.999)]
        if wn[0] < wn[1]:
            b, a = sp_signal.butter(4, wn, btype="band")
            y = sp_signal.filtfilt(b, a, y)
    env = analytic_envelope(y)
    env = env - np.mean(env)
    return compute_fft(env, sample_rate)
