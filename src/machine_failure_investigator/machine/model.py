"""High-level rotating machine state used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.machine.parameters import MachineParameters


@dataclass
class MachineState:
    """
    Snapshot of machine condition for one simulation window.

    Combines operating condition, fault specification, and base parameters.
    """

    parameters: MachineParameters = field(default_factory=MachineParameters)
    operating: OperatingCondition = field(default_factory=OperatingCondition)
    fault: FaultSpec = field(default_factory=FaultSpec)
    scenario_id: str = "default"
    time_index: int = 0

    @property
    def rpm(self) -> float:
        return self.operating.rpm

    @property
    def rotational_hz(self) -> float:
        return self.operating.rotational_hz

    @property
    def is_healthy(self) -> bool:
        return self.fault.is_healthy

    def label(self) -> str:
        return self.fault.label()


class RotatingMachine:
    """
    Lightweight container that holds parameters and can produce MachineState
    instances under different faults and operating conditions.
    """

    def __init__(self, parameters: Optional[MachineParameters] = None) -> None:
        self.parameters = parameters or MachineParameters()

    def state(
        self,
        operating: Optional[OperatingCondition] = None,
        fault: Optional[FaultSpec] = None,
        scenario_id: str = "default",
        time_index: int = 0,
    ) -> MachineState:
        return MachineState(
            parameters=self.parameters,
            operating=operating or OperatingCondition(rpm=self.parameters.nominal_rpm),
            fault=fault or FaultSpec(),
            scenario_id=scenario_id,
            time_index=time_index,
        )
