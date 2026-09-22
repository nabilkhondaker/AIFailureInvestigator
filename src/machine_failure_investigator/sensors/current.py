"""Motor current sensor model."""

from __future__ import annotations

from typing import Optional

import numpy as np

from machine_failure_investigator.machine.faults import FAULT_SIGNATURES, FaultType
from machine_failure_investigator.machine.model import MachineState
from machine_failure_investigator.sensors.noise import additive_gaussian
from machine_failure_investigator.sensors.sensor_faults import SensorFaultConfig, apply_sensor_fault


class CurrentSensor:
    """Approximate stator current magnitude series."""

    def __init__(self, noise_std: float = 0.05) -> None:
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
        params = state.parameters
        fault = state.fault
        sev = fault.severity
        sig = FAULT_SIGNATURES.get(fault.primary, {})

        base = params.base_current_a * (0.6 + 0.8 * state.operating.load_fraction)
        fault_delta = sig.get("current_delta", 0.0) * sev * base
        if fault.primary in (FaultType.BEARING_FAILURE, FaultType.LUBRICATION_FAILURE):
            fault_delta += sev * 0.15 * base
        if fault.primary == FaultType.OVERHEATING:
            fault_delta += sev * 0.2 * base

        mean_i = base + fault_delta
        t = np.arange(n_samples) / sample_rate
        # Mild 2x line-frequency modulation approximation (simplified)
        series = mean_i * (1.0 + 0.02 * np.sin(2 * np.pi * 2 * 50.0 * t))
        series = additive_gaussian(series, self.noise_std * mean_i, rng=rng)

        if sensor_fault is not None:
            series = apply_sensor_fault(series, sensor_fault, rng=rng)

        return series.astype(np.float64)
