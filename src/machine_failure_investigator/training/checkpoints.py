"""Checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path

from machine_failure_investigator.utils.paths import ensure_dir


def checkpoint_path(output_dir: str | Path, name: str, suffix: str = ".pkl") -> Path:
    d = ensure_dir(output_dir)
    return d / f"{name}{suffix}"
