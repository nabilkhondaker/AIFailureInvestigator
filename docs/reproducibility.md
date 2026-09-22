# Reproducibility

- Global seeds via `utils.seeds.set_seed`
- YAML configs for data, models, and experiments
- Dataset `metadata.json` with seed, rates, and counts
- Environment snapshot helper in `training.reproducibility`
- Deterministic split indices given a seed

Sources of residual nondeterminism: multi-threaded BLAS, GPU ops if enabled, and library version differences.
