"""Composable preprocessing pipeline for sensor bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from machine_failure_investigator.preprocessing.cleaning import clip_outliers, remove_dc, replace_nan
from machine_failure_investigator.preprocessing.filtering import highpass
from machine_failure_investigator.simulation.simulator import SensorBundle


@dataclass
class PreprocessConfig:
    remove_dc: bool = True
    clip_sigma: float = 6.0
    highpass_hz: Optional[float] = 5.0


class PreprocessPipeline:
    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    def process_vibration(self, x: np.ndarray, sample_rate: float) -> np.ndarray:
        y = replace_nan(x)
        if self.config.remove_dc:
            y = remove_dc(y)
        y = clip_outliers(y, self.config.clip_sigma)
        if self.config.highpass_hz is not None and self.config.highpass_hz > 0:
            y = highpass(y, sample_rate, self.config.highpass_hz)
        return y

    def process_bundle(self, bundle: SensorBundle) -> SensorBundle:
        vib = self.process_vibration(bundle.vibration, bundle.sample_rate)
        return SensorBundle(
            vibration=vib,
            temperature=replace_nan(bundle.temperature),
            current=replace_nan(bundle.current),
            torque=replace_nan(bundle.torque),
            tachometer=replace_nan(bundle.tachometer),
            sample_rate=bundle.sample_rate,
            state=bundle.state,
            meta=dict(bundle.meta),
        )
