"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import yaml

from machine_failure_investigator.utils.paths import project_root


PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Relative paths are resolved against the project root when possible.
    """
    p = Path(path)
    if not p.is_absolute():
        candidate = project_root() / p
        if candidate.exists():
            p = candidate
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {p} must be a mapping, got {type(data)}")
    return data


def merge_configs(*configs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Deep-merge mappings; later entries override earlier ones."""
    result: Dict[str, Any] = {}
    for cfg in configs:
        if cfg:
            _deep_update(result, cfg)
    return result


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
