"""Time-domain feature extraction."""

from __future__ import annotations

from typing import Dict

import numpy as np

from machine_failure_investigator.signal_processing.statistics import time_stats


def extract_time_domain(vibration: np.ndarray) -> Dict[str, float]:
    stats = time_stats(vibration)
    return {f"td_{k}": v for k, v in stats.items()}
