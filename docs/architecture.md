# Architecture

The system is organised as a layered pipeline:

1. **Machine & fault model** — defines operating conditions and progressive fault severity.
2. **Sensors** — map machine state to multi-channel time series (vibration, temperature, current, torque, tachometer).
3. **Preprocessing** — DC removal, outlier clipping, optional high-pass filtering.
4. **Signal processing & features** — time-domain statistics, spectral moments, order amplitudes, cross-sensor aggregates.
5. **Models** — anomaly detectors, fault classifiers, severity regressors.
6. **Investigation engine** — combines model outputs with rule-based evidence flags, differential hypotheses, and confidence/abstention logic.
7. **Reporting** — human-readable investigation reports.

Configuration is YAML-driven. The CLI entry point is `neural-failure`.
