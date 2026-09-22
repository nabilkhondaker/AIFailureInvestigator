"""Classical ML baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from machine_failure_investigator.utils.serialization import load_pickle, save_pickle


class RandomForestClassifierModel:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 16,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "RandomForestClassifierModel":
        y_enc = self.label_encoder.fit_transform(y)
        self.clf.fit(X, y_enc)
        self.feature_names = list(feature_names or [f"f{i}" for i in range(X.shape[1])])
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_enc = self.clf.predict(X)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def classes(self) -> np.ndarray:
        return self.label_encoder.classes_

    def feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        imp = self.clf.feature_importances_
        return {n: float(v) for n, v in zip(self.feature_names, imp)}

    def save(self, path: str | Path) -> None:
        save_pickle(
            {
                "clf": self.clf,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestClassifierModel":
        obj = load_pickle(path)
        model = cls()
        model.clf = obj["clf"]
        model.label_encoder = obj["label_encoder"]
        model.feature_names = obj["feature_names"]
        model.is_fitted = True
        return model
