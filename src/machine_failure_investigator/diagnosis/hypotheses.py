"""Hypothesis ranking from model probabilities and evidence rules."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from machine_failure_investigator.explainability.evidence import EvidenceItem, HypothesisEvidence


def rank_hypotheses(
    class_names: np.ndarray,
    proba: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    order = np.argsort(proba)[::-1][:top_k]
    return [(str(class_names[i]), float(proba[i])) for i in order]


def build_hypothesis_evidence(
    ranked: List[Tuple[str, float]],
    feature_flags: Dict[str, bool],
) -> List[HypothesisEvidence]:
    """Attach simple rule-based supporting/opposing evidence."""
    results: List[HypothesisEvidence] = []
    for name, conf in ranked:
        supporting: List[EvidenceItem] = []
        opposing: List[EvidenceItem] = []
        if name in ("BEARING_DEGRADATION", "BEARING_FAILURE"):
            if feature_flags.get("high_hf_energy"):
                supporting.append(EvidenceItem("Elevated high-frequency vibration energy", 0.8, "vibration"))
            if feature_flags.get("high_kurtosis"):
                supporting.append(EvidenceItem("Increased kurtosis (impulsive content)", 0.6, "vibration"))
            if not feature_flags.get("temp_rise"):
                opposing.append(EvidenceItem("Limited temperature rise", -0.2, "temperature"))
        if name == "IMBALANCE":
            if feature_flags.get("high_1x"):
                supporting.append(EvidenceItem("Strong 1× rotational component", 0.9, "vibration"))
        if name == "MISALIGNMENT":
            if feature_flags.get("high_2x"):
                supporting.append(EvidenceItem("Elevated 2× harmonic", 0.85, "vibration"))
        if name == "HEALTHY":
            if feature_flags.get("near_baseline"):
                supporting.append(EvidenceItem("Features near healthy baseline", 0.7))
        results.append(
            HypothesisEvidence(hypothesis=name, confidence=conf, supporting=supporting, opposing=opposing)
        )
    return results
