"""Minimal training callback interface."""

from __future__ import annotations

from typing import Any, Dict, List


class HistoryCallback:
    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.history.append({"epoch": epoch, **metrics})
