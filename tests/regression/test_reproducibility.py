import numpy as np

from machine_failure_investigator.simulation.dataset_builder import DatasetBuilder
from machine_failure_investigator.utils.seeds import set_seed


def test_seed_reproducibility():
    set_seed(123)
    b1 = DatasetBuilder(seed=123, duration_s=0.2, sample_rate=2048.0)
    bundles1 = b1.generate(n_samples=5)
    set_seed(123)
    b2 = DatasetBuilder(seed=123, duration_s=0.2, sample_rate=2048.0)
    bundles2 = b2.generate(n_samples=5)
    for a, b in zip(bundles1, bundles2):
        np.testing.assert_allclose(a.vibration, b.vibration)
