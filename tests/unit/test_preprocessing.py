import numpy as np

from machine_failure_investigator.preprocessing.cleaning import remove_dc
from machine_failure_investigator.preprocessing.normalization import zscore


def test_remove_dc():
    x = np.array([1.0, 2.0, 3.0])
    y = remove_dc(x)
    assert abs(y.mean()) < 1e-12


def test_zscore():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    z = zscore(x)
    assert abs(z.mean()) < 1e-12
    assert abs(z.std() - 1.0) < 1e-9
