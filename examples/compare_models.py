#!/usr/bin/env python3
"""Compare RF vs heuristic investigation on a few faults."""

from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.simulation.simulator import MachineSimulator

inv = Investigator.from_config("configs/system/default.yaml")
sim = MachineSimulator()
machine = RotatingMachine()

faults = [
    FaultSpec(FaultType.HEALTHY, 0.0),
    FaultSpec(FaultType.IMBALANCE, 0.7),
    FaultSpec(FaultType.MISALIGNMENT, 0.6),
    FaultSpec(FaultType.BEARING_DEGRADATION, 0.55),
]

for f in faults:
    state = machine.state(operating=OperatingCondition(rpm=1600), fault=f)
    result = inv.run(sim.simulate(state))
    print(f"{f.primary.value:25s} → pred={result.primary_hypothesis:25s} conf={result.confidence:.2f}")
