# AI Machine Failure Investigator

📢 **Release Notice:** This repository contains the complete codebase for this project, engineered between *June 27, 2026* and *September 22, 2026* alongside my [Neural IK](https://github.com/nabilkhondaker/Neural-IK) project. The files have been uploaded in their entirety to officially publish the project for public viewing and use.

Research software for investigating faults in rotating mechanical machinery from multi-sensor measurements (vibration, temperature, current, torque, rotational speed).

The project combines a configurable synthetic machine simulator, classical signal-processing features, anomaly detection, supervised fault classification, severity estimation, and an evidence-oriented investigation layer that produces structured diagnostic reports.

It is intended for method development, controlled experiments, and teaching—not as a certified industrial diagnostic product.

## Overview

Typical industrial condition-monitoring workflows move from raw sensors to features, detectors, and human review. This repository implements that pipeline end-to-end on **synthetic** data so that:

- fault severity and operating conditions are known and controllable;
- experiments (noise robustness, unseen faults, ablations) are reproducible;
- investigation outputs include evidence and competing hypotheses, not only a single label.

## Motivation

Rotating machines (motors, pumps, gearboxes) fail through mechanisms that leave different footprints in vibration spectra, thermal trends, and electrical signatures. Machine-learning models can map features to labels, but operational usefulness also requires:

- separating **anomaly detection** from **fault identification**;
- estimating **severity** along a degradation trajectory;
- explaining **why** a hypothesis is preferred;
- recognising when the observation may be **out of distribution**.

## Problem statement

Given a multi-sensor window (and optionally a short trajectory), produce:

1. whether behaviour is anomalous relative to a healthy reference;
2. a ranked set of fault hypotheses with confidences;
3. a severity estimate in \([0, 1]\);
4. supporting and opposing evidence tied to measured features;
5. a textual investigation report with explicit limitations.

## Research questions

- Can models distinguish mechanically different failure modes from noisy multi-sensor observations?
- How much do frequency-domain and order-related features improve diagnosis over time-domain statistics alone?
- How robust is diagnosis to changes in RPM, load, and sensor noise?
- Can anomaly detection identify a fault type that was absent from supervised training without forcing a known class?
- How should diagnostic confidence be communicated when multiple faults produce overlapping signatures?

## System architecture

```text
                  ┌─────────────────────┐
                  │   Machine Simulator │
                  └──────────┬──────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Sensor Streams  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Preprocessing   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Time Domain    Frequency       Cross-sensor
           Features      Features        Features
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ ML Models       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Anomaly         Diagnosis      Severity
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Evidence Engine │
                    └────────┬────────┘
                             │
                             ▼
                    Engineering Report
```

## Machine model

A lightweight rotating-machine parameter set controls nominal RPM, damping, approximate natural frequencies, gear tooth count, and baseline amplitudes. Operating conditions (RPM, load fraction, ambient temperature) can vary per sample.

This is an **engineering-oriented simulator** for controlled data generation, not a digital twin of a specific asset.

## Sensor model

| Channel | Content |
|---------|---------|
| Vibration | Harmonics, bearing tones, HF energy, optional impacts |
| Temperature | Load/speed dependence + fault-induced heating |
| Current | Load and friction-related variation |
| Torque | Load and mechanical resistance |
| Tachometer | RPM with measurement jitter |

Optional **sensor faults** (bias, dropout, saturation, spikes, drift) can be injected independently of mechanical faults.

## Fault model

Supported labels include:

`HEALTHY`, `BEARING_DEGRADATION`, `BEARING_FAILURE`, `IMBALANCE`, `MISALIGNMENT`, `LOOSENESS`, `GEAR_WEAR`, `LUBRICATION_FAILURE`, `OVERHEATING`, `SHAFT_DEFECT`, `SENSOR_FAULT`, `UNKNOWN_ANOMALY`.

Severity is continuous in \([0, 1]\) with project-defined bins:

| Range | Label |
|-------|-------|
| 0.00–0.20 | normal |
| 0.20–0.40 | early |
| 0.40–0.60 | moderate |
| 0.60–0.80 | severe |
| 0.80–1.00 | critical |

These thresholds are **simulation conventions**, not universal industrial standards. Progressive degradation trajectories (linear / sigmoid) support early-detection experiments.

## Signal processing

Implemented analysis includes FFT, Welch PSD, spectrogram, Hilbert envelope spectrum, spectral peak picking, order amplitudes, and standard time-domain statistics (RMS, crest factor, kurtosis, …).

$$
\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_i^2},\qquad
\mathrm{CrestFactor} = \frac{\max_i |x_i|}{\mathrm{RMS}}
$$

## Feature engineering

`FeatureExtractor` builds a fixed-length vector from time-domain, frequency-domain, optional time–frequency summaries, and cross-sensor aggregates. Groups can be disabled for ablation studies.

## Anomaly detection

Isolation Forest, Mahalanobis distance, and an optional autoencoder operate on feature space. Anomaly detection answers “is this unusual?” before class assignment. Out-of-distribution behaviour can surface as `UNKNOWN_ANOMALY`.

## Fault diagnosis and severity

Baselines: Random Forest on features. Neural option: MLP (PyTorch). Severity: gradient-boosting regressor clipped to \([0, 1]\).

## Explainability and differential diagnosis

The investigator derives boolean flags (elevated RMS, 1×/2× orders, HF energy, temperature rise, …), ranks hypotheses from model probabilities when available, attaches supporting/opposing evidence, and may **abstain** when confidence is low.

A simple fault tree groups rotational, bearing-related, structural, and thermal causes for structured reporting.

## Dataset generation

```bash
neural-failure generate-data --config configs/data/default.yaml --n-samples 2000 --seed 42
```

Splits are written as compressed NPZ files with `metadata.json` (seed, rates, counts). Splitting is by independent scenarios to reduce leakage from correlated windows of one trajectory.

## Experimental design

Configs under `configs/experiments/` cover baseline diagnosis, noise robustness, severity, unseen faults, dataset scaling, and ablations. **Results are produced by running the scripts; they are not hard-coded in this README.**

## Example investigation

```bash
neural-failure investigate
```

Heuristic or trained models yield a report resembling:

```text
MACHINE FAILURE INVESTIGATION
────────────────────────────────
Primary hypothesis: BEARING_DEGRADATION
Confidence: …
Severity: … (moderate/severe/…)
Evidence:
- Elevated vibration RMS …
- Increased high-frequency energy …
```

Exact numbers depend on the simulated state and trained checkpoints.

## Installation

```bash
git clone <repository-url>
cd ai-machine-failure-investigator
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional dashboard extras: `pip install -e ".[dashboard]"`.

## Quick start

```bash
# Small synthetic dataset
neural-failure generate-data --n-samples 400 --seed 42

# Train classifier and anomaly detector
neural-failure train --config configs/models/fault_classifier.yaml
neural-failure train --config configs/models/anomaly_detector.yaml

# Evaluate if checkpoints and test split exist
neural-failure evaluate

# Sample investigation + report
neural-failure investigate
neural-failure report
```

Makefile targets: `make data`, `make train`, `make investigate`, `make test`.

## CLI

Entry point: `neural-failure` (see `pyproject.toml`).

| Command | Purpose |
|---------|---------|
| `generate-data` | Synthetic multi-sensor dataset |
| `train` | Train model from YAML config |
| `evaluate` | Metrics on test split |
| `investigate` / `report` | End-to-end sample investigation |

## Python API

```python
from machine_failure_investigator import Investigator
from machine_failure_investigator.simulation.simulator import MachineSimulator
from machine_failure_investigator.machine.model import RotatingMachine
from machine_failure_investigator.machine.faults import FaultSpec, FaultType
from machine_failure_investigator.machine.operating_conditions import OperatingCondition

investigator = Investigator.from_config("configs/system/default.yaml")
state = RotatingMachine().state(
    operating=OperatingCondition(rpm=1800, load_fraction=0.75),
    fault=FaultSpec(FaultType.BEARING_DEGRADATION, 0.6),
)
bundle = MachineSimulator().simulate(state)
result = investigator.run(bundle)
print(result.primary_hypothesis, result.confidence, result.evidence)
```

## Project structure

See the repository tree: `src/machine_failure_investigator/` (package), `configs/`, `scripts/`, `examples/`, `tests/`, `docs/`, `experiments/`, `notebooks/`.

## Configuration

YAML under `configs/data`, `configs/models`, `configs/experiments`, `configs/system` controls sampling rates, fault mixes, model hyperparameters, seeds, and paths.

## Reproducibility

Seeds, config files, dataset metadata, and environment snapshots support reruns. See `docs/reproducibility.md`. Residual nondeterminism can arise from threaded numerical libraries and GPU kernels.

## Testing

```bash
pytest
pytest --cov=machine_failure_investigator --cov-report=term-missing
```

Unit tests cover machine, sensors, signal processing, features, models, and reporting. Integration tests cover simulation → features → train/investigate paths.

## Benchmarking

```bash
python scripts/benchmark.py
```

Reports wall-clock latency for feature extraction (extend for full investigation timing as needed). Values must be measured locally; none are claimed here a priori.

## Limitations

- Synthetic data and simplified physics.
- No industrial certification or real-plant validation in this repository.
- Domain shift to field data is expected.
- Overlapping signatures and multi-fault cases remain difficult.
- Confidence scores are not guaranteed calibrated probabilities.
- Sensor placement and machine design assumptions are project-defined.

## Failure cases

The system can fail or abstain when noise is high, severity is very low, faults are unseen, sensors drop out, or operating conditions leave the training envelope. Experiment configs under `unseen_fault` and `noise_robustness` are intended to document these behaviours rather than hide them.

## Future work

- Richer multi-fault physics and gear mesh models
- Real-data loaders and domain-adaptation baselines
- Stronger calibration and conformal prediction
- Full temporal models over degradation trajectories
- Optional interactive Streamlit dashboard

## Research directions

See `README_RESEARCH.md` and `docs/research_notes.md`.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
ruff check src tests
pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the code of conduct.

## Citation

```bibtex
@software{khondaker_machine_failure_investigator_2026,
  author = {Khondaker, Nabil},
  title  = {AI Machine Failure Investigator},
  year   = {2026},
  version = {0.1.0},
  url    = {https://github.com/nabilkhondaker/ai-machine-failure-investigator}
}
```

Also see `CITATION.cff`.

## Author

This AI Machine Failure Investigator was engineered by **Nabil Khondaker** as a project exploring the intersection of machine learning, signal processing, mechanical systems, and failure diagnosis. Further notes can be found in the journal section of [my portfolio](https://nabilkhondaker.github.io).

## License

MIT — see [LICENSE](LICENSE).
