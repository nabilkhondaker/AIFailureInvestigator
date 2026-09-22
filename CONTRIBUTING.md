# Contributing

Thank you for considering a contribution to the AI Machine Failure Investigator.

## Development setup

```bash
git clone <repository-url>
cd ai-machine-failure-investigator
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest
pytest --cov=machine_failure_investigator --cov-report=term-missing
```

## Code style

- Python 3.10+
- Type hints on public APIs
- Ruff for linting and formatting
- Docstrings for public functions and classes (NumPy style preferred)

```bash
ruff check src tests scripts examples
ruff format src tests scripts examples
```

## Pull requests

1. Open an issue first for larger changes.
2. Keep commits focused.
3. Add or update tests for new behaviour.
4. Update documentation when behaviour or public APIs change.
5. Ensure the test suite and lint checks pass.

## Scope notes

This is a research-oriented project focused on synthetic rotating-machinery data and controlled experiments. Contributions that improve physical fidelity of the simulator, add real-data loaders, or strengthen evaluation protocols are especially welcome. Please avoid introducing large binary artefacts or non-reproducible experiment results.
