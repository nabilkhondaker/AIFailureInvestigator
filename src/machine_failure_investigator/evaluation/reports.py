"""Textual evaluation summaries."""

from __future__ import annotations

from typing import Any, Dict


def format_classification_summary(metrics: Dict[str, Any]) -> str:
    lines = [
        f"Accuracy:  {metrics.get('accuracy', float('nan')):.4f}",
        f"Macro F1:  {metrics.get('macro_f1', float('nan')):.4f}",
        f"Weighted F1:{metrics.get('weighted_f1', float('nan')):.4f}",
    ]
    return "\n".join(lines)
