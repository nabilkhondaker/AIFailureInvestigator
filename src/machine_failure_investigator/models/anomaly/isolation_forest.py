"""Isolation Forest anomaly detector."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from machine_failure_investigator.utils.serialization import load_pickle, save_pickle


class IsolationForestDetector:
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.08,
        random_state: int = 42,
    ) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.threshold_: Optional[float] = None

    def fit(self, X_healthy: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X_healthy)
        scores = -self.model.score_samples(X_healthy)
        self.threshold_ = float(np.quantile(scores, 0.95))
        self.is_fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        return -self.model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly, 0 = normal."""
        scores = self.anomaly_score(X)
        thr = self.threshold_ if self.threshold_ is not None else 0.0
        return (scores >= thr).astype(int)

    def save(self, path: str | Path) -> None:
        save_pickle({"model": self.model, "threshold": self.threshold_}, path)

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        obj = load_pickle(path)
        det = cls()
        det.model = obj["model"]
        det.threshold_ = obj["threshold"]
        det.is_fitted = True
        return det
