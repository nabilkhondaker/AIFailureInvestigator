# Research notes — AI Machine Failure Investigator

Author: Nabil Khondaker  
Version: 0.1.0

## Problem formulation

Let \(\mathbf{x}_{1:T}\) denote a multi-channel window (vibration and auxiliary sensors) acquired under operating condition \(\mathbf{u}\) (RPM, load). A latent fault state \(f \in \mathcal{F}\) with severity \(s \in [0,1]\) governs the generative process. The investigator estimates:

- anomaly score \(a(\mathbf{x})\);
- posterior over faults \(p(f \mid \mathbf{x}, \mathbf{u})\) (or a point prediction);
- severity \(\hat{s}(\mathbf{x})\);
- an evidence set \(\mathcal{E}\) derived from features and model attributions.

## Machine and signal assumptions

Shaft rotation frequency \(f_0 = \mathrm{RPM}/60\). Vibration is synthesised as a sum of harmonics of \(f_0\), fault-specific tones (e.g. approximate BPFO/BPFI factors), broadband energy, and measurement noise. Temperature and current respond more slowly and track load and frictional heating.

These assumptions are transparent controls for experiments; they are not claims of quantitative agreement with a particular industrial asset.

## Dataset methodology

Independent scenarios are drawn from configurable categorical fault distributions and continuous severity/RPM/load ranges. Train/val/test splits are random over scenarios (not over sliding windows of one long run) to reduce temporal leakage. Metadata records seed, sampling rate, duration, and counts.

## Models

- **Anomaly:** Isolation Forest / Mahalanobis / autoencoder on standardised features.
- **Diagnosis:** Random Forest and MLP classifiers on the same feature space.
- **Severity:** Gradient boosting regression with outputs clipped to \([0,1]\).

Training procedures, losses, and hyperparameters are specified in YAML under `configs/models/`.

## Evaluation methodology

Standard classification and regression metrics; anomaly precision/recall; robustness sweeps over additive noise; held-out fault protocols where supervised labels exclude a fault family and anomaly detectors are scored on detection without forcing a known class.

**No performance numbers are asserted in this document.** Run the experiment scripts to obtain measurements for a given environment and seed.

## Uncertainty and abstention

Confidence is taken as the maximum class probability when a probabilistic classifier is available. Abstention triggers when confidence falls below a configured threshold or when anomaly scores are high while class confidence remains mediocre.

## Limitations

Synthetic physics, limited multi-fault interactions, possible feature leakage if splits are misused, and uncalibrated probabilities. Transfer to real machines requires additional validation that is outside the scope of the initial release.

## Reproducibility

Seed control, config snapshots, dataset metadata, and environment recording are provided. Full bit-level reproducibility across platforms is not guaranteed when multi-threaded BLAS or non-deterministic GPU kernels are used.

## Future research

Domain adaptation, conformal prediction, richer gear and bearing models, multi-window temporal reasoning, and systematic human-factors study of report presentation.
