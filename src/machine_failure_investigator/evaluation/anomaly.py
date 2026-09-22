"""Anomaly detection metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def anomaly_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, scores: np.ndarray | None = None) -> Dict[str, float]:
    out = {
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
    }
    if scores is not None and len(np.unique(y_true_binary)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true_binary, scores))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out
