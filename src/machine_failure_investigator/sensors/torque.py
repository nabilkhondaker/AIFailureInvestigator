"""Torque sensor model."""

from __future__ import annotations

from typing import Optional

import numpy as np

from machine_failure_investigator.machine.faults import FaultType
from machine_failure_investigator.machine.model import MachineState
from machine_failure_investigator.sensors.noise import additive_gaussian
from machine_failure_investigator.sensors.sensor_faults import SensorFaultConfig, apply_sensor_fault


class TorqueSensor:
    """Load-dependent torque with fault-induced variation."""

    def __init__(self, noise_std: float = 0.1) -> None:
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
        base = state.parameters.base_torque_nm * state.operating.load_fraction
        sev = state.fault.severity
        fault = state.fault.primary

        extra = 0.0
        if fault in (FaultType.BEARING_FAILURE, FaultType.LUBRICATION_FAILURE):
            extra = sev * 0.12 * base
        if fault == FaultType.GEAR_WEAR:
            extra = sev * 0.08 * base

        mean_t = base + extra
        series = np.full(n_samples, mean_t, dtype=float)
        # Small ripple at shaft frequency
        t = np.arange(n_samples) / sample_rate
        series = series * (1.0 + 0.015 * np.sin(2 * np.pi * state.rotational_hz * t))
        series = additive_gaussian(series, self.noise_std, rng=rng)

        if sensor_fault is not None:
            series = apply_sensor_fault(series, sensor_fault, rng=rng)

        return series.astype(np.float64)
