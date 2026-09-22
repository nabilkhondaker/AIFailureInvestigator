"""Temperature sensor model."""

from __future__ import annotations

from typing import Optional

import numpy as np

from machine_failure_investigator.machine.faults import FAULT_SIGNATURES, FaultType
from machine_failure_investigator.machine.model import MachineState
from machine_failure_investigator.sensors.noise import additive_gaussian
from machine_failure_investigator.sensors.sensor_faults import SensorFaultConfig, apply_sensor_fault


class TemperatureSensor:
    """
    Scalar temperature series with load dependence and fault-induced heating.

    Output is a slowly varying series (one value per sample) for alignment
    with vibration windows; in practice temperature changes on a much
    longer timescale than vibration sampling.
    """

    def __init__(self, noise_std: float = 0.3, baseline_c: float = 45.0) -> None:
        self.noise_std = noise_std
        self.baseline_c = baseline_c

    def sample(
        self,
        state: MachineState,
        n_samples: int,
        sample_rate: float,
        rng: Optional[np.random.Generator] = None,
        sensor_fault: Optional[SensorFaultConfig] = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        fault = state.fault
        sev = fault.severity
        sig = FAULT_SIGNATURES.get(fault.primary, {})

        base = self.baseline_c + state.operating.ambient_temp_c * 0.15
        load_term = 8.0 * state.operating.load_fraction
        speed_term = 3.0 * (state.rpm / 3000.0)
        fault_delta = sig.get("temp_delta", 0.0) * sev * 25.0

        if fault.primary == FaultType.OVERHEATING:
            fault_delta += sev * 40.0
        if fault.primary == FaultType.LUBRICATION_FAILURE:
            fault_delta += sev * 18.0

        mean_temp = base + load_term + speed_term + fault_delta
        # Slow drift within the window
        t = np.linspace(0.0, 1.0, n_samples)
        series = mean_temp + 0.5 * sev * t + rng.normal(0.0, 0.05, n_samples)
        series = additive_gaussian(series, self.noise_std, rng=rng)

        if sensor_fault is not None:
            series = apply_sensor_fault(series, sensor_fault, rng=rng)

        return series.astype(np.float64)
