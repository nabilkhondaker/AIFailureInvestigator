from machine_failure_investigator.explainability.evidence import EvidenceItem
from machine_failure_investigator.explainability.investigation import InvestigationResult
from machine_failure_investigator.reporting.report_builder import ReportBuilder


def test_report_contains_hypothesis():
    result = InvestigationResult(
        anomaly_detected=True,
        anomaly_score=0.8,
        primary_hypothesis="BEARING_DEGRADATION",
        confidence=0.77,
        severity=0.6,
        severity_label="severe",
        evidence=[EvidenceItem("Elevated RMS", 0.7)],
    )
    text = ReportBuilder().build(result)
    assert "BEARING_DEGRADATION" in text
    assert "Elevated RMS" in text
