"""
Vibration sensor model for rotating machinery.

AI Machine Failure Investigator
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from machine_failure_investigator.machine.dynamics import damped_transient, harmonic_series, resonance_gain
from machine_failure_investigator.machine.faults import FAULT_SIGNATURES, FaultType
from machine_failure_investigator.machine.model import MachineState
from machine_failure_investigator.sensors.noise import additive_gaussian, pink_noise
from machine_failure_investigator.sensors.sensor_faults import SensorFaultConfig, apply_sensor_fault


class VibrationSensor:
    """
    Generate a single-channel vibration waveform for a machine state.

    The model combines:
    - shaft harmonics with fault-modulated amplitudes
    - bearing characteristic frequencies (BPFO/BPFI approximations)
    - high-frequency broadband energy for bearing/lubrication faults
    - optional impact transients and resonance amplification
    - measurement noise
    """

    def __init__(self, noise_std: float = 0.02) -> None:
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
        t = np.arange(n_samples) / sample_rate
        f0 = state.rotational_hz
        params = state.parameters
        fault = state.fault
        sev = fault.severity
        sig = FAULT_SIGNATURES.get(fault.primary, {})

        # Base harmonics
        n_h = params.n_harmonics
        amps = params.base_vibration_amp * (params.harmonic_decay ** np.arange(n_h))
        phases = rng.uniform(0, 2 * np.pi, n_h)
        wave = harmonic_series(t, f0, n_h, amplitudes=amps, phases=phases)

        # Load scaling
        wave *= 0.7 + 0.6 * state.operating.load_fraction

        # Fault-specific components
        if fault.primary == FaultType.IMBALANCE:
            wave += sev * sig.get("1x_amp", 1.0) * 0.8 * np.sin(2 * np.pi * f0 * t)

        if fault.primary == FaultType.MISALIGNMENT:
            wave += sev * sig.get("2x_amp", 1.0) * 0.55 * np.sin(2 * np.pi * 2 * f0 * t + 0.3)
            wave += sev * sig.get("3x_amp", 0.35) * 0.25 * np.sin(2 * np.pi * 3 * f0 * t)

        if fault.primary in (FaultType.BEARING_DEGRADATION, FaultType.BEARING_FAILURE):
            bpfo = sig.get("bpfo_factor", 3.5) * f0
            bpfi = sig.get("bpfi_factor", 5.4) * f0
            bsf = sig.get("bsf_factor", 2.2) * f0
            hf = sig.get("hf_energy", 1.0)
            wave += sev * 0.35 * np.sin(2 * np.pi * bpfo * t)
            wave += sev * 0.25 * np.sin(2 * np.pi * bpfi * t)
            wave += sev * 0.15 * np.sin(2 * np.pi * bsf * t)
            # High-frequency band
            hf_noise = pink_noise(n_samples, std=sev * hf * 0.4, rng=rng)
            wave += hf_noise
            if fault.primary == FaultType.BEARING_FAILURE and sev > 0.4:
                # Occasional impact transient
                if rng.random() < 0.4:
                    start = rng.integers(0, max(1, n_samples // 3))
                    length = min(n_samples - start, int(0.05 * sample_rate))
                    transient = damped_transient(
                        np.arange(length) / sample_rate,
                        f_hz=bpfo,
                        decay=80.0,
                        amplitude=sev * 1.2,
                    )
                    wave[start : start + length] += transient

        if fault.primary == FaultType.LOOSENESS:
            wave += sev * 0.3 * np.sin(2 * np.pi * 0.5 * f0 * t)
            wave += sev * 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
            # Soft clipping nonlinearity
            wave = np.tanh(wave * (1.0 + sev))

        if fault.primary == FaultType.GEAR_WEAR:
            mesh = params.gear_teeth * f0
            wave += sev * 0.45 * np.sin(2 * np.pi * mesh * t)
            # Sidebands
            wave += sev * 0.2 * np.sin(2 * np.pi * (mesh - f0) * t)
            wave += sev * 0.2 * np.sin(2 * np.pi * (mesh + f0) * t)

        if fault.primary == FaultType.LUBRICATION_FAILURE:
            wave += pink_noise(n_samples, std=sev * 0.35, rng=rng)

        if fault.primary == FaultType.SHAFT_DEFECT:
            wave += sev * 0.5 * np.sin(2 * np.pi * f0 * t + 0.5)
            wave += sev * 0.3 * np.sin(2 * np.pi * 2 * f0 * t)

        if fault.primary == FaultType.UNKNOWN_ANOMALY:
            wave += pink_noise(n_samples, std=0.3 + 0.5 * sev, rng=rng)

        # Resonance amplification near natural frequencies
        for fn in params.natural_frequencies_hz:
            g = resonance_gain(f0, fn, params.damping_ratio)
            wave *= 1.0 + 0.05 * (g - 1.0)

        # Measurement noise
        wave = additive_gaussian(wave, self.noise_std, rng=rng)

        if sensor_fault is not None:
            wave = apply_sensor_fault(wave, sensor_fault, rng=rng)

        return wave.astype(np.float64)
