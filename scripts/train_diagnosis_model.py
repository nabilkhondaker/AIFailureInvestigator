#!/usr/bin/env python3
"""Train fault classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from machine_failure_investigator.cli.commands import cmd_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/models/fault_classifier.yaml"))
    args = parser.parse_args()
    cmd_train(config=args.config)


if __name__ == "__main__":
    main()
