#!/usr/bin/env python3
"""Minimal inference example without pre-trained models (heuristic path)."""

from machine_failure_investigator import Investigator
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.simulation.simulator import MachineSimulator

inv = Investigator.from_config("configs/system/default.yaml")
machine = RotatingMachine()
state = machine.state(
    operating=OperatingCondition(rpm=1800, load_fraction=0.8),
    fault=FaultSpec(FaultType.IMBALANCE, 0.6),
)
bundle = MachineSimulator().simulate(state)
result = inv.run(bundle)
print(result.primary_hypothesis, result.confidence, result.severity)
for line in result.summary_lines():
    print(line)
