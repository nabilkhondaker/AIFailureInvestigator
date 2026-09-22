import numpy as np

from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.sensors.vibration import VibrationSensor


def test_vibration_length_and_finite():
    state = RotatingMachine().state(
        operating=OperatingCondition(rpm=1800),
        fault=FaultSpec(FaultType.HEALTHY, 0.0),
    )
    sensor = VibrationSensor(noise_std=0.01)
    rng = np.random.default_rng(0)
    x = sensor.sample(state, n_samples=1024, sample_rate=5120.0, rng=rng)
    assert len(x) == 1024
    assert np.isfinite(x).all()


def test_imbalance_increases_energy():
    machine = RotatingMachine()
    healthy = machine.state(fault=FaultSpec(FaultType.HEALTHY, 0.0))
    imbalance = machine.state(fault=FaultSpec(FaultType.IMBALANCE, 0.9))
    sensor = VibrationSensor(noise_std=0.001)
    rng = np.random.default_rng(1)
    xh = sensor.sample(healthy, 2048, 5120.0, rng=rng)
    xi = sensor.sample(imbalance, 2048, 5120.0, rng=rng)
    assert np.mean(xi**2) > np.mean(xh**2)
