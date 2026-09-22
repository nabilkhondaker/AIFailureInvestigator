"""Machine physical and operating parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MachineParameters:
    """
    Parameters of a simplified rotating machine used for synthetic data.

    These are engineering-oriented controls for data generation, not a
    high-fidelity multi-body dynamics model.
    """

    nominal_rpm: float = 1800.0
    shaft_stiffness: float = 1.0
    damping_ratio: float = 0.05
    resonance_hz: float = 120.0
    bearing_stiffness: float = 1.0
    gear_teeth: int = 32
    base_vibration_amp: float = 0.15
    base_temperature_c: float = 45.0
    base_torque_nm: float = 50.0
    base_current_a: float = 12.0
    n_harmonics: int = 5
    harmonic_decay: float = 0.45
    natural_frequencies_hz: List[float] = field(
        default_factory=lambda: [85.0, 120.0, 210.0]
    )

    def rotational_frequency_hz(self, rpm: float | None = None) -> float:
        r = self.nominal_rpm if rpm is None else rpm
        return r / 60.0
