"""Additional transforms (STFT power, spectral moments)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from machine_failure_investigator.signal_processing.fft import compute_fft, compute_psd


def spectral_moments(x: np.ndarray, sample_rate: float) -> Dict[str, float]:
    """
    Spectral centroid, bandwidth, and entropy-like measures.
    """
    freqs, psd = compute_psd(x, sample_rate)
    p = psd + 1e-18
    p = p / p.sum()
    centroid = float(np.sum(freqs * p))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    entropy = float(-np.sum(p * np.log(p)))
    # High-frequency energy ratio (above 1 kHz when possible)
    hf_mask = freqs >= min(1000.0, 0.4 * sample_rate / 2)
    hf_energy = float(np.sum(psd[hf_mask]) / (np.sum(psd) + 1e-18))
    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy": entropy,
        "hf_energy_ratio": hf_energy,
    }


def band_power(x: np.ndarray, sample_rate: float, bands: list[tuple[float, float]]) -> Dict[str, float]:
    freqs, psd = compute_psd(x, sample_rate)
    out: Dict[str, float] = {}
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        out[f"band_power_{i}_{int(lo)}_{int(hi)}"] = (
            float(_trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        )
    return out
