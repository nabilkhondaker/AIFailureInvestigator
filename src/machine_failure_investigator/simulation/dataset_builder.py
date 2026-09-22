"""Reproducible synthetic dataset generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.operating_conditions import sample_operating_condition
from machine_failure_investigator.simulation.fault_injection import sample_fault
from machine_failure_investigator.simulation.simulator import MachineSimulator, SensorBundle
from machine_failure_investigator.utils.logging import get_logger
from machine_failure_investigator.utils.paths import ensure_dir
from machine_failure_investigator.utils.seeds import get_rng

logger = get_logger(__name__)


class DatasetBuilder:
    """
    Build train/val/test splits of multi-sensor windows.

    Splitting is performed by scenario (independent draws) to reduce leakage
    from highly correlated windows of the same physical trajectory.
    """

    def __init__(
        self,
        simulator: Optional[MachineSimulator] = None,
        sample_rate: float = 5120.0,
        duration_s: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.simulator = simulator or MachineSimulator()
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        self.seed = seed
        self.rng = get_rng(seed)

    def generate(
        self,
        n_samples: int,
        fault_distribution: Optional[Dict[str, float]] = None,
        severity_range: Tuple[float, float] = (0.1, 1.0),
        rpm_range: Tuple[float, float] = (800.0, 3000.0),
        load_range: Tuple[float, float] = (0.3, 1.0),
    ) -> List[SensorBundle]:
        bundles: List[SensorBundle] = []
        machine = RotatingMachine()
        for i in range(n_samples):
            op = sample_operating_condition(rpm_range, load_range, rng=self.rng)
            fault = sample_fault(fault_distribution, severity_range, rng=self.rng)
            state = machine.state(
                operating=op,
                fault=fault,
                scenario_id=f"scen_{i:05d}",
                time_index=0,
            )
            bundle = self.simulator.simulate(
                state,
                duration_s=self.duration_s,
                sample_rate=self.sample_rate,
                rng=self.rng,
            )
            bundles.append(bundle)
            if (i + 1) % 200 == 0:
                logger.info("Generated %d / %d windows", i + 1, n_samples)
        return bundles

    def split(
        self,
        bundles: Sequence[SensorBundle],
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
    ) -> Dict[str, List[SensorBundle]]:
        assert abs(train + val + test - 1.0) < 1e-6
        n = len(bundles)
        idx = self.rng.permutation(n)
        n_train = int(n * train)
        n_val = int(n * val)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        return {
            "train": [bundles[i] for i in train_idx],
            "val": [bundles[i] for i in val_idx],
            "test": [bundles[i] for i in test_idx],
        }

    def save(
        self,
        splits: Dict[str, List[SensorBundle]],
        output_dir: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        out = ensure_dir(output_dir)
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "sample_rate_hz": self.sample_rate,
            "duration_s": self.duration_s,
            "n_train": len(splits.get("train", [])),
            "n_val": len(splits.get("val", [])),
            "n_test": len(splits.get("test", [])),
        }
        if metadata:
            meta.update(metadata)

        for split_name, items in splits.items():
            if not items:
                continue
            path = out / f"{split_name}.npz"
            arrays = self._bundles_to_arrays(items)
            np.savez_compressed(path, **arrays)
            logger.info("Wrote %s (%d samples)", path, len(items))

        meta_path = out / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return out

    def _bundles_to_arrays(self, bundles: Sequence[SensorBundle]) -> Dict[str, np.ndarray]:
        vib = np.stack([b.vibration for b in bundles])
        temp = np.stack([b.temperature for b in bundles])
        cur = np.stack([b.current for b in bundles])
        tq = np.stack([b.torque for b in bundles])
        tach = np.stack([b.tachometer for b in bundles])
        labels = np.array([b.state.label() for b in bundles])
        severity = np.array([b.state.fault.severity for b in bundles], dtype=np.float64)
        rpm = np.array([b.state.rpm for b in bundles], dtype=np.float64)
        load = np.array([b.state.operating.load_fraction for b in bundles], dtype=np.float64)
        scenario = np.array([b.state.scenario_id for b in bundles])
        return {
            "vibration": vib,
            "temperature": temp,
            "current": cur,
            "torque": tq,
            "tachometer": tach,
            "labels": labels,
            "severity": severity,
            "rpm": rpm,
            "load": load,
            "scenario_id": scenario,
            "sample_rate": np.array(self.sample_rate),
        }

    @staticmethod
    def load_split(path: str | Path) -> Dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
