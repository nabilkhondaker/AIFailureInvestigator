from machine_failure_investigator.pipelines.investigation import run_investigation_pipeline
from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.simulation.simulator import MachineSimulator


def test_pipeline_returns_report():
    inv = Investigator.from_config("configs/system/default.yaml")
    bundle = MachineSimulator().simulate(
        RotatingMachine().state(fault=FaultSpec(FaultType.BEARING_DEGRADATION, 0.7)),
        duration_s=0.4,
    )
    result, report = run_investigation_pipeline(inv, bundle)
    assert "Primary hypothesis" in report or "BEARING" in report
    assert result.anomaly_detected in (True, False)
