from machine_failure_investigator.explainability.confidence import max_proba_confidence, should_abstain
import numpy as np


def test_max_proba():
    p = np.array([0.1, 0.7, 0.2])
    idx, conf = max_proba_confidence(p)
    assert idx == 1 and abs(conf - 0.7) < 1e-9


def test_abstain():
    assert should_abstain(0.3, threshold=0.55)
    assert not should_abstain(0.9, threshold=0.55)
