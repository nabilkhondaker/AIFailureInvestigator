"""Investigation pipeline wrapper."""

from __future__ import annotations

from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.reporting.report_builder import ReportBuilder
from machine_failure_investigator.simulation.simulator import SensorBundle


def run_investigation_pipeline(
    investigator: Investigator,
    bundle: SensorBundle,
    machine_id: str = "Simulated Machine",
) -> tuple:
    result = investigator.run(bundle)
    report = ReportBuilder().build(result, machine_id=machine_id)
    return result, report
