"""High-level training helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from machine_failure_investigator.models.anomaly.isolation_forest import IsolationForestDetector
from machine_failure_investigator.models.baseline import RandomForestClassifierModel
from machine_failure_investigator.models.diagnosis.mlp import MLPClassifier
from machine_failure_investigator.models.severity.regressor import SeverityRegressor
from machine_failure_investigator.utils.logging import get_logger
from machine_failure_investigator.utils.paths import ensure_dir

logger = get_logger(__name__)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    feature_names: Optional[list] = None,
    output_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> Any:
    logger.info("Training classifier type=%s n=%d features=%d", model_type, len(X), X.shape[1])
    if model_type in ("random_forest", "baseline"):
        model = RandomForestClassifierModel(**{k: v for k, v in kwargs.items() if k in (
            "n_estimators", "max_depth", "min_samples_leaf", "random_state"
        )})
        model.fit(X, y, feature_names=feature_names)
    elif model_type == "mlp":
        model = MLPClassifier(**{k: v for k, v in kwargs.items() if k in (
            "hidden_dims", "dropout", "learning_rate", "batch_size", "epochs", "weight_decay", "seed"
        )})
        model.fit(X, y, feature_names=feature_names)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    if output_path is not None:
        ensure_dir(Path(output_path).parent)
        model.save(output_path)
        logger.info("Saved classifier to %s", output_path)
    return model


def train_anomaly_detector(
    X_healthy: np.ndarray,
    model_type: str = "isolation_forest",
    output_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> Any:
    logger.info("Training anomaly detector type=%s n_healthy=%d", model_type, len(X_healthy))
    if model_type == "isolation_forest":
        model = IsolationForestDetector(**{k: v for k, v in kwargs.items() if k in (
            "n_estimators", "contamination", "random_state"
        )})
        model.fit(X_healthy)
    elif model_type == "statistical":
        from machine_failure_investigator.models.anomaly.statistical import StatisticalAnomalyDetector
        model = StatisticalAnomalyDetector(**{k: v for k, v in kwargs.items() if k in ("threshold_std",)})
        model.fit(X_healthy)
    else:
        raise ValueError(f"Unknown anomaly model_type: {model_type}")
    if output_path is not None:
        ensure_dir(Path(output_path).parent)
        model.save(output_path)
        logger.info("Saved anomaly detector to %s", output_path)
    return model


def train_severity_regressor(
    X: np.ndarray,
    severity: np.ndarray,
    feature_names: Optional[list] = None,
    output_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> SeverityRegressor:
    logger.info("Training severity regressor n=%d", len(X))
    model = SeverityRegressor(**{k: v for k, v in kwargs.items() if k in (
        "n_estimators", "max_depth", "learning_rate", "random_state"
    )})
    model.fit(X, severity, feature_names=feature_names)
    if output_path is not None:
        ensure_dir(Path(output_path).parent)
        model.save(output_path)
    return model
