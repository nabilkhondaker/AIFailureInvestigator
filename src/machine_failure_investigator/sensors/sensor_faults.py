"""Injected sensor faults: bias, dropout, saturation, spikes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SensorFaultKind(str, Enum):
    NONE = "none"
    BIAS = "bias"
    DROPOUT = "dropout"
    SATURATION = "saturation"
    SPIKES = "spikes"
    NOISE_INCREASE = "noise_increase"
    DRIFT = "drift"


@dataclass
class SensorFaultConfig:
    kind: SensorFaultKind = SensorFaultKind.NONE
    severity: float = 0.0
    bias_value: float = 0.0
    dropout_fraction: float = 0.0
    sat_level: float = 10.0
    spike_rate: float = 0.01
    spike_amplitude: float = 5.0
    noise_multiplier: float = 3.0
    drift_rate: float = 0.5


def apply_sensor_fault(
    signal: np.ndarray,
    config: SensorFaultConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply a sensor fault transformation to a 1-D signal."""
    rng = rng or np.random.default_rng()
    out = signal.copy()
    s = max(0.0, min(1.0, config.severity))
    kind = config.kind

    if kind == SensorFaultKind.NONE or s < 1e-6:
        return out

    if kind == SensorFaultKind.BIAS:
        out = out + config.bias_value * s

    elif kind == SensorFaultKind.DROPOUT:
        mask = rng.random(len(out)) < (config.dropout_fraction * s)
        out = out.copy()
        out[mask] = 0.0

    elif kind == SensorFaultKind.SATURATION:
        level = config.sat_level * (1.0 - 0.5 * s)
        out = np.clip(out, -level, level)

    elif kind == SensorFaultKind.SPIKES:
        n_spikes = int(config.spike_rate * s * len(out))
        if n_spikes > 0:
            idx = rng.choice(len(out), size=n_spikes, replace=False)
            signs = rng.choice([-1.0, 1.0], size=n_spikes)
            out = out.copy()
            out[idx] += signs * config.spike_amplitude * s

    elif kind == SensorFaultKind.NOISE_INCREASE:
        extra = rng.normal(0.0, np.std(out) * (config.noise_multiplier - 1.0) * s, size=out.shape)
        out = out + extra

    elif kind == SensorFaultKind.DRIFT:
        t = np.linspace(0.0, 1.0, len(out))
        out = out + config.drift_rate * s * t

    return out
