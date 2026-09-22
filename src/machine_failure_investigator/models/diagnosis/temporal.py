"""Temporal model stubs for sequence-based diagnosis (future extension)."""

from __future__ import annotations


class TemporalDiagnosisModel:
    """Reserved for RNN/Transformer experiments on multi-window trajectories."""

    def __init__(self) -> None:
        self.is_fitted = False

    def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Temporal models are planned for a future release.")
