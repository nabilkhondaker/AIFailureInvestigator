"""Inference helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from machine_failure_investigator.features.feature_pipeline import FeatureExtractor
from machine_failure_investigator.simulation.simulator import SensorBundle


def predict_label(classifier: Any, bundle: SensorBundle, extractor: FeatureExtractor) -> str:
    fv = extractor.transform(bundle)
    X = fv.to_array().reshape(1, -1)
    return str(classifier.predict(X)[0])
