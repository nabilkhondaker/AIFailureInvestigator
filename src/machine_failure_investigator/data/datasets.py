"""In-memory dataset wrapper for feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FaultDataset:
    X: np.ndarray
    y: np.ndarray
    severity: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    label_names: Optional[List[str]] = None
    meta: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim == 2 else 0

    def subset(self, indices: Sequence[int]) -> "FaultDataset":
        idx = np.asarray(indices)
        return FaultDataset(
            X=self.X[idx],
            y=self.y[idx],
            severity=None if self.severity is None else self.severity[idx],
            feature_names=self.feature_names,
            label_names=self.label_names,
            meta=None
            if self.meta is None
            else {k: v[idx] for k, v in self.meta.items()},
        )
