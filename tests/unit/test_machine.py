"""Unit tests for machine model."""

from machine_failure_investigator.machine.faults import FaultSpec, FaultType, severity_label
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.machine.parameters import MachineParameters


def test_parameters_rotational_hz():
    p = MachineParameters(nominal_rpm=1800)
    assert abs(p.rotational_frequency_hz() - 30.0) < 1e-9


def test_machine_state_label():
    m = RotatingMachine()
    state = m.state(
        operating=OperatingCondition(rpm=1500),
        fault=FaultSpec(FaultType.IMBALANCE, 0.4),
    )
    assert state.label() == "IMBALANCE"
    assert not state.is_healthy


def test_severity_label():
    assert severity_label(0.1) == "normal"
    assert severity_label(0.5) == "moderate"
    assert severity_label(0.95) == "critical"
