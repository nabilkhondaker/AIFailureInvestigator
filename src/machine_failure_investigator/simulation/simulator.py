"""Core multi-sensor machine simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from machine_failure_investigator.machine.model import MachineState, RotatingMachine
from machine_failure_investigator.sensors import (
    CurrentSensor,
    TachometerSensor,
    TemperatureSensor,
    TorqueSensor,
    VibrationSensor,
)
from machine_failure_investigator.sensors.sensor_faults import SensorFaultConfig


@dataclass
class SensorBundle:
    """Container for one multi-sensor observation window."""

    vibration: np.ndarray
    temperature: np.ndarray
    current: np.ndarray
    torque: np.ndarray
    tachometer: np.ndarray
    sample_rate: float
    state: MachineState
    meta: Dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.vibration)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "vibration": self.vibration,
            "temperature": self.temperature,
            "current": self.current,
            "torque": self.torque,
            "tachometer": self.tachometer,
        }


class MachineSimulator:
    """
    Generate multi-sensor observations from a MachineState.

    This is an engineering-oriented simulator for controlled experiments,
    not a certified digital twin of any specific industrial asset.
    """

    def __init__(
        self,
        machine: Optional[RotatingMachine] = None,
        vibration_noise: float = 0.02,
        temperature_noise: float = 0.3,
        current_noise: float = 0.05,
        torque_noise: float = 0.1,
        tach_noise: float = 0.5,
        temperature_baseline: float = 45.0,
    ) -> None:
        self.machine = machine or RotatingMachine()
        self.vibration = VibrationSensor(noise_std=vibration_noise)
        self.temperature = TemperatureSensor(
            noise_std=temperature_noise, baseline_c=temperature_baseline
        )
        self.current = CurrentSensor(noise_std=current_noise)
        self.torque = TorqueSensor(noise_std=torque_noise)
        self.tachometer = TachometerSensor(noise_std=tach_noise)

    def simulate(
        self,
        state: MachineState,
        duration_s: float = 1.0,
        sample_rate: float = 5120.0,
        rng: Optional[np.random.Generator] = None,
        sensor_fault: Optional[SensorFaultConfig] = None,
    ) -> SensorBundle:
        rng = rng or np.random.default_rng()
        n = int(duration_s * sample_rate)
        if n < 16:
            raise ValueError("duration_s * sample_rate must yield at least 16 samples")

        vib = self.vibration.sample(state, n, sample_rate, rng=rng, sensor_fault=sensor_fault)
        temp = self.temperature.sample(state, n, sample_rate, rng=rng)
        cur = self.current.sample(state, n, sample_rate, rng=rng)
        tq = self.torque.sample(state, n, sample_rate, rng=rng)
        tach = self.tachometer.sample(state, n, sample_rate, rng=rng)

        return SensorBundle(
            vibration=vib,
            temperature=temp,
            current=cur,
            torque=tq,
            tachometer=tach,
            sample_rate=sample_rate,
            state=state,
            meta={
                "duration_s": duration_s,
                "fault": state.fault.label(),
                "severity": state.fault.severity,
                "rpm": state.rpm,
            },
        )
