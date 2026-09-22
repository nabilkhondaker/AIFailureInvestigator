#!/usr/bin/env python3
"""Benchmark feature extraction and investigation latency."""

from __future__ import annotations

from machine_failure_investigator.features.feature_pipeline import FeatureExtractor
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import OperatingCondition
from machine_failure_investigator.pipelines.benchmarking import benchmark_callable
from machine_failure_investigator.simulation.simulator import MachineSimulator


def main() -> None:
    sim = MachineSimulator()
    machine = RotatingMachine()
    state = machine.state(
        operating=OperatingCondition(rpm=1800),
        fault=FaultSpec(FaultType.BEARING_DEGRADATION, 0.5),
    )
    bundle = sim.simulate(state)
    extractor = FeatureExtractor()

    def once():
        extractor.transform(bundle)

    stats = benchmark_callable(once, n_warmup=3, n_runs=20)
    print("Feature extraction latency (s):", stats)


if __name__ == "__main__":
    main()
