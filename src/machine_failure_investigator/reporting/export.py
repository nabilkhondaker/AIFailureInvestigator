"""Export helpers."""

from __future__ import annotations

from pathlib import Path

from machine_failure_investigator.utils.paths import ensure_dir


def write_text_report(text: str, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p
