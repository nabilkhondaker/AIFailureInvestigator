"""Lightweight configuration schemas (Pydantic where useful)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class SensorNoiseConfig(BaseModel):
    enabled: bool = True
    noise_std: float = 0.02


class MachineConfig(BaseModel):
    nominal_rpm: float = 1800.0
    sampling_rate_hz: float = 5120.0
    duration_s: float = 1.0
    n_harmonics: int = 5


class InvestigationConfig(BaseModel):
    anomaly_threshold: float = 0.65
    confidence_threshold: float = 0.55
    severity_bins: List[Tuple[float, float, str]] = Field(
        default_factory=lambda: [
            (0.0, 0.2, "normal"),
            (0.2, 0.4, "early"),
            (0.4, 0.6, "moderate"),
            (0.6, 0.8, "severe"),
            (0.8, 1.01, "critical"),
        ]
    )
    report_format: str = "text"


class SystemConfig(BaseModel):
    seed: int = 42
    output_dir: str = "outputs"
    log_level: str = "INFO"
    machine: MachineConfig = Field(default_factory=MachineConfig)
    investigation: InvestigationConfig = Field(default_factory=InvestigationConfig)
    paths: Dict[str, str] = Field(
        default_factory=lambda: {
            "models": "models",
            "data": "data/synthetic",
            "reports": "reports",
            "figures": "outputs/figures",
        }
    )
    sensors: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        return cls.model_validate(data)
