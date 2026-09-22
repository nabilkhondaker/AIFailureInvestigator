"""Operating condition sampling and representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class OperatingCondition:
    """Instantaneous or steady operating point."""

    rpm: float = 1800.0
    load_fraction: float = 0.7
    ambient_temp_c: float = 25.0

    @property
    def rotational_hz(self) -> float:
        return self.rpm / 60.0


def sample_operating_condition(
    rpm_range: Tuple[float, float] = (800.0, 3000.0),
    load_range: Tuple[float, float] = (0.3, 1.0),
    ambient_range: Tuple[float, float] = (15.0, 35.0),
    rng: Optional[np.random.Generator] = None,
) -> OperatingCondition:
    """Draw a random steady operating condition."""
    rng = rng or np.random.default_rng()
    return OperatingCondition(
        rpm=float(rng.uniform(*rpm_range)),
        load_fraction=float(rng.uniform(*load_range)),
        ambient_temp_c=float(rng.uniform(*ambient_range)),
    )


def discrete_rpm_grid(values: Sequence[float] | None = None) -> list[float]:
    """Common discrete RPM set used in experiments."""
    if values is None:
        values = [500, 750, 1000, 1500, 2000, 2500, 3000]
    return [float(v) for v in values]
