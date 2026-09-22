"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger; ensure basic config exists."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name or "machine_failure_investigator")
