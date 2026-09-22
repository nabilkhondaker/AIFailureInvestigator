"""Rule-based evidence flags from feature vectors."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_feature_flags(
    features: Dict[str, float],
    healthy_ref: Dict[str, float] | None = None,
) -> Dict[str, bool]:
    """
    Derive boolean diagnostic flags from a feature dictionary.

    Thresholds are project conventions for the synthetic simulator.
    """
    healthy_ref = healthy_ref or {}
    flags = {
        "high_rms": features.get("td_rms", 0.0) > healthy_ref.get("td_rms", 0.25) * 1.4
        if healthy_ref
        else features.get("td_rms", 0.0) > 0.35,
        "high_kurtosis": features.get("td_kurtosis", 0.0) > 3.0,
        "high_crest": features.get("td_crest_factor", 0.0) > 4.0,
        "high_1x": features.get("fd_order_1", 0.0) > healthy_ref.get("fd_order_1", 0.5) * 1.5
        if healthy_ref
        else features.get("fd_order_1", 0.0) > 0.8,
        "high_2x": features.get("fd_order_2", 0.0) > healthy_ref.get("fd_order_2", 0.3) * 1.5
        if healthy_ref
        else features.get("fd_order_2", 0.0) > 0.5,
        "high_hf_energy": features.get("fd_hf_energy_ratio", 0.0) > 0.25,
        "temp_rise": features.get("cs_temp_mean", 40.0) > healthy_ref.get("cs_temp_mean", 50.0) + 8.0
        if healthy_ref
        else features.get("cs_temp_mean", 40.0) > 60.0,
        "near_baseline": False,
    }
    if healthy_ref:
        # crude near-baseline: most z-scores small
        zs = []
        for k, v in features.items():
            if k in healthy_ref and healthy_ref[k] != 0:
                zs.append(abs(v - healthy_ref[k]) / (abs(healthy_ref[k]) + 1e-6))
        flags["near_baseline"] = bool(zs) and float(np.mean(zs)) < 0.35
    return flags
