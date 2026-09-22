#!/usr/bin/env python3
"""Evaluate trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

from machine_failure_investigator.cli.commands import cmd_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    args = parser.parse_args()
    cmd_evaluate(config=args.config)


if __name__ == "__main__":
    main()
