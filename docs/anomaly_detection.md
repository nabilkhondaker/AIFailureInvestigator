# Anomaly detection

Anomaly detection is intentionally separate from multi-class diagnosis.

Supported approaches:

- Isolation Forest on feature vectors (default)
- Mahalanobis distance from a healthy baseline
- Autoencoder reconstruction error (PyTorch)

The investigator can emit `UNKNOWN_ANOMALY` when behaviour is anomalous but does not match trained fault classes confidently.
