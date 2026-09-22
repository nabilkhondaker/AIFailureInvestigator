"""Default configuration values."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_SYSTEM: Dict[str, Any] = {
    "seed": 42,
    "output_dir": "outputs",
    "log_level": "INFO",
    "machine": {
        "nominal_rpm": 1800.0,
        "sampling_rate_hz": 5120.0,
        "duration_s": 1.0,
        "n_harmonics": 5,
    },
    "investigation": {
        "anomaly_threshold": 0.65,
        "confidence_threshold": 0.55,
    },
}
