"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path | str) -> Path:
    """Create directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
