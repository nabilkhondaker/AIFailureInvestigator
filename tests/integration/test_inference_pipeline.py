from machine_failure_investigator.diagnosis.investigator import Investigator
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.simulation.simulator import MachineSimulator


def test_investigation_runs():
    inv = Investigator.from_config("configs/system/default.yaml")
    bundle = MachineSimulator().simulate(
        RotatingMachine().state(fault=FaultSpec(FaultType.MISALIGNMENT, 0.5)),
        duration_s=0.4,
    )
    result = inv.run(bundle)
    assert result.primary_hypothesis
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.severity <= 1.0
