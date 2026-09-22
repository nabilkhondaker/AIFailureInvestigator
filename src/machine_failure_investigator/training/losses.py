"""Custom loss helpers (PyTorch)."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def classification_loss():
    if nn is None:
        raise ImportError("PyTorch required")
    return nn.CrossEntropyLoss()


def reconstruction_loss():
    if nn is None:
        raise ImportError("PyTorch required")
    return nn.MSELoss()
