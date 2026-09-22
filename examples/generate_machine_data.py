#!/usr/bin/env python3
"""Generate a small dataset for local exploration."""

from pathlib import Path

from machine_failure_investigator.simulation.dataset_builder import DatasetBuilder

builder = DatasetBuilder(seed=7, duration_s=0.5, sample_rate=4096.0)
bundles = builder.generate(n_samples=50)
splits = builder.split(bundles)
out = builder.save(splits, Path("data/synthetic/example_small"))
print("Saved to", out)
