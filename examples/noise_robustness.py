#!/usr/bin/env python3
"""Demonstrate investigation under increasing vibration noise."""

from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.simulation.simulator import MachineSimulator

inv = Investigator.from_config("configs/system/default.yaml")
machine = RotatingMachine()
state = machine.state(
    operating=OperatingCondition(rpm=1800),
    fault=FaultSpec(FaultType.BEARING_DEGRADATION, 0.5),
)

for noise in [0.01, 0.05, 0.1, 0.25]:
    sim = MachineSimulator(vibration_noise=noise)
    result = inv.run(sim.simulate(state))
    print(f"noise={noise:.2f} → {result.primary_hypothesis} conf={result.confidence:.2f}")
