from machine_failure_investigator.features.feature_pipeline import FeatureExtractor
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.simulation.simulator import MachineSimulator


def test_feature_vector_keys():
    bundle = MachineSimulator().simulate(
        RotatingMachine().state(fault=FaultSpec(FaultType.HEALTHY, 0.0)),
        duration_s=0.5,
    )
    fv = FeatureExtractor().transform(bundle)
    assert "td_rms" in fv.values
    assert "fd_spectral_centroid" in fv.values
    assert len(fv.names) == len(fv.values)
