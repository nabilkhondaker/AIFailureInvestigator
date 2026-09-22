"""Frequency-domain feature extraction."""

from __future__ import annotations

from typing import Dict

import numpy as np

from machine_failure_investigator.signal_processing.order_analysis import order_amplitudes
from machine_failure_investigator.signal_processing.transforms import band_power, spectral_moments


def extract_frequency_domain(
    vibration: np.ndarray,
    sample_rate: float,
    rpm: float,
) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    moments = spectral_moments(vibration, sample_rate)
    feats.update({f"fd_{k}": v for k, v in moments.items()})
    orders = order_amplitudes(vibration, sample_rate, rpm)
    feats.update({f"fd_{k}": v for k, v in orders.items()})
    bands = band_power(
        vibration,
        sample_rate,
        bands=[(10, 100), (100, 500), (500, 2000), (2000, min(5000, sample_rate / 2 - 1))],
    )
    feats.update({f"fd_{k}": v for k, v in bands.items()})
    return feats
