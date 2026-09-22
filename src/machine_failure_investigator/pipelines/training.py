"""Training pipeline from NPZ data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from machine_failure_investigator.data.loaders import load_npz_split
from machine_failure_investigator.features.feature_pipeline import FeatureExtractor
from machine_failure_investigator.machine.model import MachineState, RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.simulation.simulator import SensorBundle
from machine_failure_investigator.training.trainer import train_anomaly_detector, train_classifier, train_severity_regressor
from machine_failure_investigator.utils.logging import get_logger

logger = get_logger(__name__)


def features_from_npz(path: str | Path, extractor: Optional[FeatureExtractor] = None):
    extractor = extractor or FeatureExtractor()
    data = load_npz_split(path)
    vib = data["vibration"]
    n = len(vib)
    fs = float(data["sample_rate"]) if "sample_rate" in data else 5120.0
    bundles = []
    machine = RotatingMachine()
    for i in range(n):
        label = str(data["labels"][i])
        sev = float(data["severity"][i]) if "severity" in data else 0.0
        rpm = float(data["rpm"][i]) if "rpm" in data else 1800.0
        load = float(data["load"][i]) if "load" in data else 0.7
        state = machine.state(
            operating=OperatingCondition(rpm=rpm, load_fraction=load),
            fault=FaultSpec(FaultType(label) if label in FaultType.__members__ else FaultType.UNKNOWN_ANOMALY, sev),
            scenario_id=str(data["scenario_id"][i]) if "scenario_id" in data else f"i{i}",
        )
        bundles.append(
            SensorBundle(
                vibration=vib[i],
                temperature=data["temperature"][i],
                current=data["current"][i],
                torque=data["torque"][i],
                tachometer=data["tachometer"][i],
                sample_rate=fs,
                state=state,
            )
        )
    X = extractor.transform_many(bundles)
    y = data["labels"]
    severity = data.get("severity")
    return X, y, severity, extractor.feature_names
