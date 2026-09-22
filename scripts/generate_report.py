#!/usr/bin/env python3
"""Generate and save a sample investigation report."""

from __future__ import annotations

from pathlib import Path

from machine_failure_investigator.cli.commands import cmd_investigate
from machine_failure_investigator.reporting.export import write_text_report
from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.reporting.report_builder import ReportBuilder
from machine_failure_investigator.simulation.simulator import MachineSimulator


def main() -> None:
    inv = Investigator.from_config("configs/system/default.yaml")
    machine = RotatingMachine()
    state = machine.state(
        operating=OperatingCondition(rpm=1750, load_fraction=0.7),
        fault=FaultSpec(FaultType.BEARING_DEGRADATION, 0.7),
        scenario_id="report_demo",
    )
    bundle = MachineSimulator().simulate(state)
    result = inv.run(bundle)
    text = ReportBuilder().build(result, machine_id="Report Demo #001")
    path = write_text_report(text, Path("reports/sample_investigation.txt"))
    print(f"Wrote {path}")
    print(text)


if __name__ == "__main__":
    main()
