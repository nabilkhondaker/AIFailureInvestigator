#!/usr/bin/env python3
"""Write an engineering report file for a misalignment case."""

from pathlib import Path

from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.reporting.export import write_text_report
from machine_failure_investigator.reporting.report_builder import ReportBuilder
from machine_failure_investigator.simulation.simulator import MachineSimulator

inv = Investigator.from_config("configs/system/default.yaml")
state = RotatingMachine().state(
    operating=OperatingCondition(rpm=2100, load_fraction=0.85),
    fault=FaultSpec(FaultType.MISALIGNMENT, 0.72),
    scenario_id="eng_report",
)
result = inv.run(MachineSimulator().simulate(state))
text = ReportBuilder().build(result, machine_id="Pump Drive Train #7")
path = write_text_report(text, Path("reports/misalignment_case.txt"))
print(path)
print(text)
