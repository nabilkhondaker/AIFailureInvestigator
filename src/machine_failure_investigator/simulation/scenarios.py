"""Named scenario builders for experiments and demos."""

from __future__ import annotations

from typing import List

from machine_failure_investigator.machine.degradation import linear_degradation, sigmoid_degradation
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import MachineState, RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition


def healthy_scenario(rpm: float = 1800.0, n_windows: int = 10) -> List[MachineState]:
    machine = RotatingMachine()
    op = OperatingCondition(rpm=rpm, load_fraction=0.7)
    return [
        machine.state(operating=op, fault=FaultSpec(FaultType.HEALTHY, 0.0), time_index=i)
        for i in range(n_windows)
    ]


def bearing_degradation_trajectory(
    rpm: float = 1800.0,
    n_windows: int = 40,
    end_severity: float = 0.95,
) -> List[MachineState]:
    """Progressive bearing degradation under constant operating conditions."""
    machine = RotatingMachine()
    op = OperatingCondition(rpm=rpm, load_fraction=0.75)
    severities = sigmoid_degradation(n_windows, midpoint=0.45, steepness=8.0, end=end_severity)
    states = []
    for i, sev in enumerate(severities):
        fault = FaultSpec(FaultType.BEARING_DEGRADATION, float(sev))
        if sev > 0.85:
            fault = FaultSpec(FaultType.BEARING_FAILURE, float(sev))
        states.append(
            machine.state(
                operating=op,
                fault=fault,
                scenario_id="bearing_traj",
                time_index=i,
            )
        )
    return states


def multi_fault_scenario(rpm: float = 2000.0) -> MachineState:
    """Example of combined imbalance and mild bearing degradation."""
    machine = RotatingMachine()
    return machine.state(
        operating=OperatingCondition(rpm=rpm, load_fraction=0.8),
        fault=FaultSpec(
            primary=FaultType.IMBALANCE,
            severity=0.55,
            secondary=FaultType.BEARING_DEGRADATION,
            secondary_severity=0.3,
        ),
        scenario_id="multi_fault",
    )
