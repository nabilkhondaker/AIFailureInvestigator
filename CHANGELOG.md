# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- Initial research implementation of the AI Machine Failure Investigator.
- Configurable rotating-machine simulator with progressive fault injection.
- Multi-sensor models: vibration, temperature, current, torque, tachometer.
- Signal-processing pipeline: FFT, PSD, spectrogram, envelope analysis, order tracking.
- Time-domain, frequency-domain, and cross-sensor feature extraction.
- Anomaly detection (statistical, Isolation Forest, autoencoder).
- Fault classification models (MLP, Random Forest, CNN baseline).
- Severity estimation regressor.
- Evidence-based investigation engine with differential diagnosis and fault tree.
- Explainability utilities and investigation report generator.
- CLI (`neural-failure`) and Python API.
- Experiment infrastructure, unit/integration tests, documentation, and example scripts.
