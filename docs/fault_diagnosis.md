# Fault diagnosis

Primary classifiers:

- Random Forest on engineered features (baseline)
- MLP on engineered features (PyTorch)
- Experimental CNN path on raw vibration (scaffold)

Severity is estimated separately as a continuous value in \([0, 1]\) with project-defined textual bins (normal / early / moderate / severe / critical).
