"""Structured investigation result object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from machine_failure_investigator.explainability.evidence import EvidenceItem, HypothesisEvidence


@dataclass
class InvestigationResult:
    anomaly_detected: bool
    anomaly_score: float
    primary_hypothesis: str
    confidence: float
    severity: float
    severity_label: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    alternatives: List[HypothesisEvidence] = field(default_factory=list)
    abstain: bool = False
    recommendations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = [
            f"Anomaly detected: {self.anomaly_detected} (score={self.anomaly_score:.3f})",
            f"Primary hypothesis: {self.primary_hypothesis}",
            f"Confidence: {self.confidence:.3f}",
            f"Severity: {self.severity:.3f} ({self.severity_label})",
        ]
        if self.abstain:
            lines.append("Status: ABSTAIN — confidence below threshold or ambiguous evidence")
        if self.evidence:
            lines.append("Evidence:")
            for e in self.evidence[:8]:
                lines.append(f"  - {e.description}")
        if self.alternatives:
            lines.append("Alternatives:")
            for alt in self.alternatives[:5]:
                lines.append(f"  - {alt.hypothesis} (conf={alt.confidence:.3f})")
        return lines
