#!/usr/bin/env python3
"""Generate synthetic dataset from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from machine_failure_investigator.cli.commands import cmd_generate_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic machinery dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/data/default.yaml"))
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cmd_generate_data(config=args.config, n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
