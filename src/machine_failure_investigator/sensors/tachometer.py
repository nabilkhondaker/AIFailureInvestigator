"""Tachometer / rotational speed sensor."""

from __future__ import annotations

from typing import Optional

import numpy as np

from machine_failure_investigator.machine.model import MachineState
from machine_failure_investigator.sensors.noise import additive_gaussian
from machine_failure_investigator.sensors.sensor_faults import SensorFaultConfig, apply_sensor_fault


class TachometerSensor:
    """RPM reading with small jitter."""

    def __init__(self, noise_std: float = 0.5) -> None:
        self.noise_std = noise_std

    def sample(
        self,
        state: MachineState,
        n_samples: int,
        sample_rate: float,
        rng: Optional[np.random.Generator] = None,
        sensor_fault: Optional[SensorFaultConfig] = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        series = np.full(n_samples, state.rpm, dtype=float)
        series = additive_gaussian(series, self.noise_std, rng=rng)
        if sensor_fault is not None:
            series = apply_sensor_fault(series, sensor_fault, rng=rng)
        return series.astype(np.float64)
