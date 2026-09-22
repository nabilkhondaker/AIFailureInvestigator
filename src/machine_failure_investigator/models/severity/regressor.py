"""Severity regressor (sklearn GradientBoosting or MLP)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from machine_failure_investigator.utils.serialization import load_pickle, save_pickle


class SeverityRegressor:
    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        severity: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "SeverityRegressor":
        self.model.fit(X, severity)
        self.feature_names = list(feature_names or [f"f{i}" for i in range(X.shape[1])])
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.model.predict(X)
        return np.clip(pred, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        save_pickle({"model": self.model, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SeverityRegressor":
        obj = load_pickle(path)
        reg = cls()
        reg.model = obj["model"]
        reg.feature_names = obj["feature_names"]
        reg.is_fitted = True
        return reg
