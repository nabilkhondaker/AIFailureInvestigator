"""Short text summaries."""

from __future__ import annotations

from machine_failure_investigator.explainability.investigation import InvestigationResult


def one_line_summary(result: InvestigationResult) -> str:
    status = "ABSTAIN" if result.abstain else ("ANOMALY" if result.anomaly_detected else "NOMINAL")
    return (
        f"[{status}] {result.primary_hypothesis} "
        f"conf={result.confidence:.2f} sev={result.severity:.2f} ({result.severity_label})"
    )
