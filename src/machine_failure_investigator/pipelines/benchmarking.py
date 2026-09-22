"""Simple latency benchmarks."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from machine_failure_investigator.utils.timing import timed_call


def benchmark_callable(fn: Callable, n_warmup: int = 2, n_runs: int = 10) -> Dict[str, float]:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        _, t = timed_call(fn)
        times.append(t)
    arr = np.array(times)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "n_runs": n_runs,
    }
