"""
End-to-end investigation orchestrator.

Combines anomaly detection, fault classification, severity estimation,
and evidence generation into a single InvestigationResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from machine_failure_investigator.config.loader import load_config
from machine_failure_investigator.config.schemas import SystemConfig
from machine_failure_investigator.diagnosis.evidence_engine import compute_feature_flags
from machine_failure_investigator.diagnosis.hypotheses import build_hypothesis_evidence, rank_hypotheses
from machine_failure_investigator.explainability.confidence import max_proba_confidence, should_abstain
from machine_failure_investigator.explainability.evidence import EvidenceItem
from machine_failure_investigator.explainability.investigation import InvestigationResult
from machine_failure_investigator.features.feature_pipeline import FeatureExtractor
from machine_failure_investigator.machine.faults import severity_label
from machine_failure_investigator.preprocessing.pipeline import PreprocessPipeline
from machine_failure_investigator.simulation.simulator import SensorBundle
from machine_failure_investigator.utils.logging import get_logger

logger = get_logger(__name__)


class Investigator:
    """
    High-level API for machine failure investigation.

    Example
    -------
    >>> inv = Investigator.from_config("configs/system/default.yaml")
    >>> result = inv.run(sensor_bundle)
    >>> print(result.primary_hypothesis, result.confidence)
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        classifier: Any = None,
        anomaly_detector: Any = None,
        severity_model: Any = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        preprocessor: Optional[PreprocessPipeline] = None,
        healthy_feature_ref: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config or SystemConfig()
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.severity_model = severity_model
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.preprocessor = preprocessor or PreprocessPipeline()
        self.healthy_feature_ref = healthy_feature_ref or {}

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "Investigator":
        raw = load_config(path)
        cfg = SystemConfig.from_dict(raw)
        return cls(config=cfg)

    def attach_models(
        self,
        classifier: Any = None,
        anomaly_detector: Any = None,
        severity_model: Any = None,
    ) -> None:
        if classifier is not None:
            self.classifier = classifier
        if anomaly_detector is not None:
            self.anomaly_detector = anomaly_detector
        if severity_model is not None:
            self.severity_model = severity_model

    def run(self, sensor_data: SensorBundle) -> InvestigationResult:
        logger.info("Running investigation for scenario=%s", sensor_data.state.scenario_id)
        bundle = self.preprocessor.process_bundle(sensor_data)
        fv = self.feature_extractor.transform(bundle)
        X = fv.to_array().reshape(1, -1)

        # Anomaly
        anomaly_score = 0.0
        anomaly_detected = False
        if self.anomaly_detector is not None and getattr(self.anomaly_detector, "is_fitted", False):
            scores = self.anomaly_detector.anomaly_score(X)
            anomaly_score = float(scores[0])
            pred = self.anomaly_detector.predict(X)
            anomaly_detected = bool(pred[0] == 1)
        else:
            # Heuristic: elevated RMS / HF
            flags_tmp = compute_feature_flags(fv.values, self.healthy_feature_ref)
            anomaly_detected = flags_tmp.get("high_rms", False) or flags_tmp.get("high_hf_energy", False)
            anomaly_score = 0.7 if anomaly_detected else 0.2

        # Classification
        primary = "UNKNOWN_ANOMALY"
        confidence = 0.0
        alternatives: List = []
        proba = None
        if self.classifier is not None and getattr(self.classifier, "is_fitted", False):
            proba = self.classifier.predict_proba(X)[0]
            classes = self.classifier.classes()
            idx, confidence = max_proba_confidence(proba)
            primary = str(classes[idx])
            ranked = rank_hypotheses(classes, proba, top_k=5)
            flags = compute_feature_flags(fv.values, self.healthy_feature_ref)
            alternatives = build_hypothesis_evidence(ranked[1:], flags)
        else:
            flags = compute_feature_flags(fv.values, self.healthy_feature_ref)
            if flags.get("high_hf_energy"):
                primary = "BEARING_DEGRADATION"
                confidence = 0.55
            elif flags.get("high_1x"):
                primary = "IMBALANCE"
                confidence = 0.5
            elif flags.get("high_2x"):
                primary = "MISALIGNMENT"
                confidence = 0.5
            elif not anomaly_detected:
                primary = "HEALTHY"
                confidence = 0.6

        # Severity
        severity = 0.0
        if self.severity_model is not None and getattr(self.severity_model, "is_fitted", False):
            severity = float(self.severity_model.predict(X)[0])
        else:
            # Heuristic from RMS and HF
            severity = float(
                np.clip(
                    0.4 * (fv.values.get("td_rms", 0) / 0.6)
                    + 0.4 * fv.values.get("fd_hf_energy_ratio", 0)
                    + 0.2 * (1.0 if flags.get("temp_rise") else 0.0),
                    0.0,
                    1.0,
                )
            )

        sev_label = severity_label(severity)
        abstain = should_abstain(
            confidence,
            threshold=self.config.investigation.confidence_threshold,
            anomaly_score=anomaly_score if anomaly_detected else None,
            anomaly_threshold=self.config.investigation.anomaly_threshold,
        )

        if anomaly_detected and primary == "HEALTHY":
            primary = "UNKNOWN_ANOMALY"
            confidence = min(confidence, 0.45)
            abstain = True

        evidence: List[EvidenceItem] = []
        flags = compute_feature_flags(fv.values, self.healthy_feature_ref)
        if flags.get("high_rms"):
            evidence.append(EvidenceItem("Elevated vibration RMS relative to baseline", 0.7, "vibration", "td_rms"))
        if flags.get("high_hf_energy"):
            evidence.append(EvidenceItem("Increased high-frequency energy ratio", 0.75, "vibration", "fd_hf_energy_ratio"))
        if flags.get("high_1x"):
            evidence.append(EvidenceItem("Strong 1× rotational order amplitude", 0.8, "vibration", "fd_order_1"))
        if flags.get("high_2x"):
            evidence.append(EvidenceItem("Elevated 2× harmonic content", 0.75, "vibration", "fd_order_2"))
        if flags.get("high_kurtosis"):
            evidence.append(EvidenceItem("Elevated kurtosis (impulsive content)", 0.65, "vibration", "td_kurtosis"))
        if flags.get("temp_rise"):
            evidence.append(EvidenceItem("Temperature elevated relative to baseline", 0.7, "temperature", "cs_temp_mean"))

        recommendations = []
        if primary in ("BEARING_DEGRADATION", "BEARING_FAILURE"):
            recommendations.append(
                "Acquire additional vibration data at higher sampling rate and inspect bearing at next maintenance interval."
            )
        if primary == "IMBALANCE":
            recommendations.append("Check rotor balance and mounting; verify residual unbalance against acceptance criteria.")
        if primary == "MISALIGNMENT":
            recommendations.append("Verify shaft alignment and coupling condition.")
        if abstain:
            recommendations.append("Diagnosis uncertain; collect additional windows under stable operating conditions.")
        if not recommendations:
            recommendations.append("Continue monitoring; schedule routine inspection if anomaly persists.")

        limitations = [
            "Diagnosis is based on synthetic or project-defined models and is not a certified industrial decision.",
            "Sensor placement, machine design, and operating regime may differ from training assumptions.",
        ]

        return InvestigationResult(
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            primary_hypothesis=primary,
            confidence=confidence,
            severity=severity,
            severity_label=sev_label,
            evidence=evidence,
            alternatives=alternatives,
            abstain=abstain,
            recommendations=recommendations,
            limitations=limitations,
            meta={
                "rpm": float(bundle.state.rpm),
                "load": float(bundle.state.operating.load_fraction),
                "scenario_id": bundle.state.scenario_id,
                "features": fv.values,
            },
        )
