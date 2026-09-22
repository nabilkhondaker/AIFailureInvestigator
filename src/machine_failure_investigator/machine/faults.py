"""Fault type definitions and severity mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FaultType(str, Enum):
    HEALTHY = "HEALTHY"
    BEARING_DEGRADATION = "BEARING_DEGRADATION"
    BEARING_FAILURE = "BEARING_FAILURE"
    IMBALANCE = "IMBALANCE"
    MISALIGNMENT = "MISALIGNMENT"
    LOOSENESS = "LOOSENESS"
    GEAR_WEAR = "GEAR_WEAR"
    LUBRICATION_FAILURE = "LUBRICATION_FAILURE"
    OVERHEATING = "OVERHEATING"
    SHAFT_DEFECT = "SHAFT_DEFECT"
    SENSOR_FAULT = "SENSOR_FAULT"
    UNKNOWN_ANOMALY = "UNKNOWN_ANOMALY"


@dataclass
class FaultSpec:
    """
    Specification of one or more simultaneous faults with severity in [0, 1].

    Severity is a project-defined continuous scale:
    0 = healthy / negligible, 1 = critical for the given fault type.
    """

    primary: FaultType = FaultType.HEALTHY
    severity: float = 0.0
    secondary: Optional[FaultType] = None
    secondary_severity: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.severity = float(max(0.0, min(1.0, self.severity)))
        self.secondary_severity = float(max(0.0, min(1.0, self.secondary_severity)))
        if isinstance(self.primary, str):
            self.primary = FaultType(self.primary)
        if self.secondary is not None and isinstance(self.secondary, str):
            self.secondary = FaultType(self.secondary)

    @property
    def is_healthy(self) -> bool:
        return self.primary == FaultType.HEALTHY and self.severity < 0.05

    def label(self) -> str:
        return self.primary.value


# Characteristic frequency multipliers relative to shaft rotational frequency.
# Used by the simulator to inject spectral signatures.
FAULT_SIGNATURES: Dict[FaultType, Dict[str, float]] = {
    FaultType.HEALTHY: {},
    FaultType.IMBALANCE: {"1x_amp": 1.0, "temp_delta": 0.05},
    FaultType.MISALIGNMENT: {"1x_amp": 0.4, "2x_amp": 1.0, "3x_amp": 0.35, "temp_delta": 0.1},
    FaultType.BEARING_DEGRADATION: {
        "bpfo_factor": 3.5,
        "bpfi_factor": 5.4,
        "bsf_factor": 2.2,
        "hf_energy": 1.0,
        "temp_delta": 0.15,
    },
    FaultType.BEARING_FAILURE: {
        "bpfo_factor": 3.5,
        "bpfi_factor": 5.4,
        "hf_energy": 2.0,
        "temp_delta": 0.35,
        "envelope_mod": 1.0,
    },
    FaultType.LOOSENESS: {"1x_amp": 0.5, "2x_amp": 0.8, "half_x": 0.6, "nonlinear": 1.0},
    FaultType.GEAR_WEAR: {"gear_mesh": 1.0, "sideband": 0.7, "temp_delta": 0.12},
    FaultType.LUBRICATION_FAILURE: {"hf_energy": 0.8, "temp_delta": 0.4, "friction": 1.0},
    FaultType.OVERHEATING: {"temp_delta": 1.0, "current_delta": 0.2},
    FaultType.SHAFT_DEFECT: {"1x_amp": 0.7, "2x_amp": 0.5, "temp_delta": 0.08},
    FaultType.SENSOR_FAULT: {},
    FaultType.UNKNOWN_ANOMALY: {"broadband": 1.0},
}


def severity_label(severity: float) -> str:
    """Map continuous severity to a textual bin (project convention)."""
    if severity < 0.2:
        return "normal"
    if severity < 0.4:
        return "early"
    if severity < 0.6:
        return "moderate"
    if severity < 0.8:
        return "severe"
    return "critical"
