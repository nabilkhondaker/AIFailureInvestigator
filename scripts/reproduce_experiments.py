#!/usr/bin/env python3
"""Reproduce a minimal experiment chain: data → train → evaluate → investigate."""

from __future__ import annotations

from machine_failure_investigator.cli.commands import (
    cmd_evaluate,
    cmd_generate_data,
    cmd_investigate,
    cmd_train,
)
from pathlib import Path


def main() -> None:
    cmd_generate_data(config=Path("configs/data/default.yaml"), n_samples=400, seed=42)
    cmd_train(config=Path("configs/models/fault_classifier.yaml"))
    cmd_train(config=Path("configs/models/anomaly_detector.yaml"))
    cmd_evaluate(config=Path("configs/experiments/baseline.yaml"))
    cmd_investigate(config=Path("configs/system/default.yaml"))


if __name__ == "__main__":
    main()
