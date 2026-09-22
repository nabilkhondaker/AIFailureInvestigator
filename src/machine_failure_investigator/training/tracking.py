"""Lightweight experiment tracking to JSON logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from machine_failure_investigator.utils.paths import ensure_dir


def log_experiment(output_dir: str | Path, name: str, payload: Dict[str, Any]) -> Path:
    d = ensure_dir(output_dir)
    path = d / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    payload = {**payload, "logged_at": datetime.now(timezone.utc).isoformat()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
