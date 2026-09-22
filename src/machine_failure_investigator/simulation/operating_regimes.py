"""Operating regime definitions for generalization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Regime:
    name: str
    rpm_range: Tuple[float, float]
    load_range: Tuple[float, float]


TRAIN_REGIME = Regime("train", (1000.0, 2000.0), (0.4, 0.8))
TEST_REGIME_HIGH_SPEED = Regime("high_speed", (2200.0, 3000.0), (0.4, 0.8))
TEST_REGIME_HIGH_LOAD = Regime("high_load", (1000.0, 2000.0), (0.85, 1.0))
TEST_REGIME_WIDE = Regime("wide", (800.0, 3000.0), (0.3, 1.0))


def all_regimes() -> List[Regime]:
    return [TRAIN_REGIME, TEST_REGIME_HIGH_SPEED, TEST_REGIME_HIGH_LOAD, TEST_REGIME_WIDE]
