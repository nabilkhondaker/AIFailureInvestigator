"""Basic dataset validation checks."""

from __future__ import annotations

from typing import Dict

import numpy as np


def validate_feature_matrix(X: np.ndarray, y: np.ndarray) -> Dict[str, bool]:
    checks = {
        "finite_X": bool(np.isfinite(X).all()),
        "matching_length": len(X) == len(y),
        "nonempty": len(X) > 0,
        "2d_X": X.ndim == 2,
    }
    return checks
