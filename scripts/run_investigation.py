#!/usr/bin/env python3
"""Run a sample investigation."""

from __future__ import annotations

import argparse
from pathlib import Path

from machine_failure_investigator.cli.commands import cmd_investigate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/system/default.yaml"))
    args = parser.parse_args()
    cmd_investigate(config=args.config)


if __name__ == "__main__":
    main()
