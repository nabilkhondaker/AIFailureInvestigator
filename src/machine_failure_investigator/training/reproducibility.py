"""Environment snapshot for experiment metadata."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict


def snapshot_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except ImportError:
        pass
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info
