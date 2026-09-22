"""Simple ensemble of classifiers via averaged probabilities."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class SimpleEnsemble:
    def __init__(self, models: Sequence) -> None:
        self.models = list(models)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [m.predict_proba(X) for m in self.models]
        # Assume aligned class orders; average
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        classes = self.models[0].classes()
        return classes[idx]
