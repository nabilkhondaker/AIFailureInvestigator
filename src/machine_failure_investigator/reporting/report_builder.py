"""Build human-readable investigation reports from InvestigationResult."""

from __future__ import annotations

from typing import Optional

from machine_failure_investigator.explainability.investigation import InvestigationResult
from machine_failure_investigator.reporting.templates import REPORT_FOOTER, REPORT_HEADER


class ReportBuilder:
    def build(self, result: InvestigationResult, machine_id: str = "Simulated Rotating Machine") -> str:
        meta = result.meta or {}
        lines = [
            REPORT_HEADER,
            f"Machine:\n{machine_id}",
            "",
            "Operating condition:",
            f"{meta.get('rpm', float('nan')):.0f} RPM",
            f"Load fraction: {meta.get('load', float('nan')):.2f}",
            "",
            "Status:",
            "ANOMALY DETECTED" if result.anomaly_detected else "NO ANOMALY FLAG",
            "",
            "Primary hypothesis:",
            result.primary_hypothesis,
            "",
            f"Confidence:\n{result.confidence:.2f}",
            "",
            f"Severity:\n{result.severity:.2f} ({result.severity_label})",
            "",
        ]
        if result.abstain:
            lines += ["Note: System abstained from a firm diagnosis due to low confidence or ambiguity.", ""]
        if result.evidence:
            lines.append("Evidence:")
            for e in result.evidence:
                lines.append(f"- {e.description}")
            lines.append("")
        if result.alternatives:
            lines.append("Alternative hypotheses:")
            for alt in result.alternatives:
                lines.append(f"- {alt.hypothesis} (confidence={alt.confidence:.2f})")
            lines.append("")
        if result.recommendations:
            lines.append("Recommendation:")
            for r in result.recommendations:
                lines.append(r)
            lines.append("")
        if result.limitations:
            lines.append("Limitations:")
            for lim in result.limitations:
                lines.append(lim)
        lines.append(REPORT_FOOTER)
        return "\n".join(lines)
