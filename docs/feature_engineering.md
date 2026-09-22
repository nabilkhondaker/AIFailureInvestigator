# Feature engineering

Features are grouped for ablation:

- **Time domain** — RMS, kurtosis, crest factor, …
- **Frequency domain** — spectral centroid/bandwidth/entropy, band powers, order amplitudes
- **Time–frequency** — spectrogram summary statistics (optional)
- **Cross-sensor** — temperature mean/trend, current, torque, RPM, load

The `FeatureExtractor` produces a fixed-length vector aligned across samples for classical ML and feed-forward networks.
