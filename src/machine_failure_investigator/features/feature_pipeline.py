"""Unified feature extraction from a SensorBundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from machine_failure_investigator.features.cross_sensor import extract_cross_sensor
from machine_failure_investigator.features.frequency_domain import extract_frequency_domain
from machine_failure_investigator.features.time_domain import extract_time_domain
from machine_failure_investigator.features.time_frequency import extract_time_frequency
from machine_failure_investigator.simulation.simulator import SensorBundle


@dataclass
class FeatureVector:
    values: Dict[str, float]
    names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.names:
            self.names = sorted(self.values.keys())

    def to_array(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        keys = list(names) if names is not None else self.names
        return np.array([self.values.get(k, 0.0) for k in keys], dtype=np.float64)

    @staticmethod
    def stack(vectors: Sequence["FeatureVector"], names: Optional[Sequence[str]] = None) -> np.ndarray:
        if not vectors:
            return np.zeros((0, 0))
        keys = list(names) if names is not None else vectors[0].names
        return np.stack([v.to_array(keys) for v in vectors])


class FeatureExtractor:
    """
    Extract a fixed-length feature vector from a multi-sensor window.

    Feature groups can be toggled for ablation studies.
    """

    def __init__(
        self,
        use_time_domain: bool = True,
        use_frequency_domain: bool = True,
        use_time_frequency: bool = False,
        use_cross_sensor: bool = True,
    ) -> None:
        self.use_time_domain = use_time_domain
        self.use_frequency_domain = use_frequency_domain
        self.use_time_frequency = use_time_frequency
        self.use_cross_sensor = use_cross_sensor
        self._feature_names: Optional[List[str]] = None

    def transform(self, bundle: SensorBundle) -> FeatureVector:
        feats: Dict[str, float] = {}
        vib = bundle.vibration
        fs = bundle.sample_rate
        rpm = float(bundle.state.rpm)
        load = float(bundle.state.operating.load_fraction)

        if self.use_time_domain:
            feats.update(extract_time_domain(vib))
        if self.use_frequency_domain:
            feats.update(extract_frequency_domain(vib, fs, rpm))
        if self.use_time_frequency:
            feats.update(extract_time_frequency(vib, fs))
        if self.use_cross_sensor:
            feats.update(
                extract_cross_sensor(
                    bundle.temperature,
                    bundle.current,
                    bundle.torque,
                    bundle.tachometer,
                    rpm,
                    load,
                )
            )

        fv = FeatureVector(values=feats)
        if self._feature_names is None:
            self._feature_names = fv.names
        else:
            # Align to established name order
            fv.names = self._feature_names
        return fv

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names or [])

    def transform_many(self, bundles: Sequence[SensorBundle]) -> np.ndarray:
        vectors = [self.transform(b) for b in bundles]
        if not vectors:
            return np.zeros((0, 0))
        names = self.feature_names or vectors[0].names
        self._feature_names = list(names)
        return FeatureVector.stack(vectors, names)
