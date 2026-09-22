import numpy as np

from machine_failure_investigator.signal_processing.fft import compute_fft
from machine_failure_investigator.signal_processing.statistics import time_stats


def test_fft_peak_at_known_freq():
    fs = 1000.0
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 50.0 * t)
    freqs, mag = compute_fft(x, fs)
    peak_freq = freqs[np.argmax(mag)]
    assert abs(peak_freq - 50.0) < 2.0


def test_rms():
    x = np.ones(100)
    stats = time_stats(x)
    assert abs(stats["rms"] - 1.0) < 1e-9
