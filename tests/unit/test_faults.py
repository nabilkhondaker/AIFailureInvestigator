from machine_failure_investigator.machine.faults import FaultSpec, FaultType


def test_fault_spec_clip():
    f = FaultSpec(FaultType.BEARING_FAILURE, severity=1.5)
    assert f.severity == 1.0
    f2 = FaultSpec(FaultType.HEALTHY, severity=-0.2)
    assert f2.severity == 0.0


def test_healthy_flag():
    assert FaultSpec(FaultType.HEALTHY, 0.0).is_healthy
    assert not FaultSpec(FaultType.IMBALANCE, 0.3).is_healthy
