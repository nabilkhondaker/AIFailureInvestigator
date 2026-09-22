"""Cross-sensor and operating-condition features."""

from __future__ import annotations

from typing import Dict

import numpy as np


def extract_cross_sensor(
    temperature: np.ndarray,
    current: np.ndarray,
    torque: np.ndarray,
    tachometer: np.ndarray,
    rpm: float,
    load: float,
) -> Dict[str, float]:
    return {
        "cs_temp_mean": float(np.mean(temperature)),
        "cs_temp_std": float(np.std(temperature)),
        "cs_temp_trend": float(np.polyfit(np.arange(len(temperature)), temperature, 1)[0])
        if len(temperature) > 2
        else 0.0,
        "cs_current_mean": float(np.mean(current)),
        "cs_current_std": float(np.std(current)),
        "cs_torque_mean": float(np.mean(torque)),
        "cs_rpm_mean": float(np.mean(tachometer)),
        "cs_rpm_cmd": float(rpm),
        "cs_load": float(load),
    }
