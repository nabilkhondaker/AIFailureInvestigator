"""Helpers for sampling fault specifications."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.metadata import FAULT_LABELS


def sample_fault(
    distribution: Optional[Dict[str, float]] = None,
    severity_range: tuple[float, float] = (0.1, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> FaultSpec:
    """Sample a primary fault type and severity from a categorical distribution."""
    rng = rng or np.random.default_rng()
    if distribution is None:
        labels = [l for l in FAULT_LABELS if l not in ("SENSOR_FAULT", "UNKNOWN_ANOMALY")]
        distribution = {l: 1.0 / len(labels) for l in labels}

    labels = list(distribution.keys())
    probs = np.array([distribution[l] for l in labels], dtype=float)
    probs = probs / probs.sum()
    choice = rng.choice(labels, p=probs)
    ft = FaultType(choice)
    if ft == FaultType.HEALTHY:
        sev = float(rng.uniform(0.0, 0.05))
    else:
        sev = float(rng.uniform(*severity_range))
    return FaultSpec(primary=ft, severity=sev)


def sample_fault_from_list(
    allowed: Sequence[str],
    severity_range: tuple[float, float] = (0.15, 0.95),
    rng: Optional[np.random.Generator] = None,
) -> FaultSpec:
    rng = rng or np.random.default_rng()
    label = rng.choice(list(allowed))
    ft = FaultType(label)
    sev = 0.0 if ft == FaultType.HEALTHY else float(rng.uniform(*severity_range))
    return FaultSpec(primary=ft, severity=sev)
