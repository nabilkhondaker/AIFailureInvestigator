"""Evidence items for investigation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidenceItem:
    description: str
    support: float = 0.0  # positive supports hypothesis; negative argues against
    sensor: Optional[str] = None
    feature: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class HypothesisEvidence:
    hypothesis: str
    confidence: float
    supporting: List[EvidenceItem] = field(default_factory=list)
    opposing: List[EvidenceItem] = field(default_factory=list)
