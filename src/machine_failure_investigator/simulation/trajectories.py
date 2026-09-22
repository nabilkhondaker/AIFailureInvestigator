"""Time-indexed trajectory helpers."""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from machine_failure_investigator.machine.degradation import linear_degradation
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import MachineState, RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition


def degradation_trajectory(
    fault_type: FaultType,
    n_steps: int = 50,
    rpm: float = 1800.0,
    load: float = 0.7,
    start_sev: float = 0.0,
    end_sev: float = 1.0,
    scenario_id: str = "degradation",
) -> List[MachineState]:
    machine = RotatingMachine()
    op = OperatingCondition(rpm=rpm, load_fraction=load)
    sevs = linear_degradation(n_steps, start_sev, end_sev)
    return [
        machine.state(
            operating=op,
            fault=FaultSpec(fault_type, float(s)),
            scenario_id=scenario_id,
            time_index=i,
        )
        for i, s in enumerate(sevs)
    ]


def iterate_trajectory(states: List[MachineState]) -> Iterator[MachineState]:
    yield from states
