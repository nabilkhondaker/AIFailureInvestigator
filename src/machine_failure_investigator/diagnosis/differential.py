"""Differential diagnosis utilities."""

from __future__ import annotations

from typing import Dict, List

FEATURE_TO_FAULT_HINTS: Dict[str, List[str]] = {
    "high_1x": ["IMBALANCE", "SHAFT_DEFECT"],
    "high_2x": ["MISALIGNMENT", "LOOSENESS"],
    "high_hf_energy": ["BEARING_DEGRADATION", "BEARING_FAILURE", "LUBRICATION_FAILURE"],
    "high_kurtosis": ["BEARING_DEGRADATION", "BEARING_FAILURE"],
    "temp_rise": ["OVERHEATING", "LUBRICATION_FAILURE", "BEARING_FAILURE"],
    "gear_mesh": ["GEAR_WEAR"],
}


def suggest_from_flags(flags: Dict[str, bool]) -> List[str]:
    scores: Dict[str, int] = {}
    for flag, active in flags.items():
        if not active:
            continue
        for fault in FEATURE_TO_FAULT_HINTS.get(flag, []):
            scores[fault] = scores.get(fault, 0) + 1
    return sorted(scores, key=lambda k: scores[k], reverse=True)
