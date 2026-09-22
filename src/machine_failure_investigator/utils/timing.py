"""Simple timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional


@contextmanager
def timer(label: str = "elapsed", logger=None) -> Generator[dict, None, None]:
    """Context manager that records wall-clock time in seconds."""
    result: dict = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        if logger is not None:
            logger.info("%s: %.4f s", label, result["seconds"])


def timed_call(fn, *args, **kwargs) -> tuple:
    """Call ``fn`` and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
