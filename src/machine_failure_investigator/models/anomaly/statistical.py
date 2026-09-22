"""Mahalanobis-distance statistical anomaly detector."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from machine_failure_investigator.utils.serialization import load_pickle, save_pickle


class StatisticalAnomalyDetector:
    def __init__(self, threshold_std: float = 3.0) -> None:
        self.threshold_std = threshold_std
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.is_fitted = False

    def fit(self, X_healthy: np.ndarray) -> "StatisticalAnomalyDetector":
        self.mean_ = np.mean(X_healthy, axis=0)
        cov = np.cov(X_healthy, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * 1e-6
        self.cov_inv_ = np.linalg.pinv(cov)
        scores = self.anomaly_score(X_healthy)
        self.threshold_ = float(np.mean(scores) + self.threshold_std * np.std(scores))
        self.is_fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.cov_inv_ is not None
        diff = X - self.mean_
        left = diff @ self.cov_inv_
        return np.sqrt(np.sum(left * diff, axis=1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.anomaly_score(X)
        thr = self.threshold_ if self.threshold_ is not None else 0.0
        return (scores >= thr).astype(int)

    def save(self, path: str | Path) -> None:
        save_pickle(
            {
                "mean": self.mean_,
                "cov_inv": self.cov_inv_,
                "threshold": self.threshold_,
                "threshold_std": self.threshold_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StatisticalAnomalyDetector":
        obj = load_pickle(path)
        det = cls(threshold_std=obj["threshold_std"])
        det.mean_ = obj["mean"]
        det.cov_inv_ = obj["cov_inv"]
        det.threshold_ = obj["threshold"]
        det.is_fitted = True
        return det
