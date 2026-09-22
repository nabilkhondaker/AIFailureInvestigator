import numpy as np

from machine_failure_investigator.models.anomaly.isolation_forest import IsolationForestDetector
from machine_failure_investigator.models.baseline import RandomForestClassifierModel


def test_rf_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = np.array(["A"] * 40 + ["B"] * 40)
    model = RandomForestClassifierModel(n_estimators=20, max_depth=4, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert len(pred) == 80
    assert set(pred).issubset({"A", "B"})


def test_iforest():
    rng = np.random.default_rng(0)
    Xh = rng.normal(size=(100, 4))
    det = IsolationForestDetector(n_estimators=50, random_state=0)
    det.fit(Xh)
    scores = det.anomaly_score(Xh)
    assert scores.shape == (100,)
