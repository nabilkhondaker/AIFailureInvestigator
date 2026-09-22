"""Load synthetic NPZ splits."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def load_npz_split(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
