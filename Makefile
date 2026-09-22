.PHONY: install install-dev test lint format clean data train evaluate investigate report help

PYTHON ?= python
PIP ?= pip

help:
	@echo "Common targets:"
	@echo "  install       Install package"
	@echo "  install-dev   Install with development extras"
	@echo "  test          Run pytest"
	@echo "  lint          Run ruff check"
	@echo "  format        Run ruff format"
	@echo "  data          Generate a small synthetic dataset"
	@echo "  train         Train baseline models on small data"
	@echo "  evaluate      Evaluate trained models"
	@echo "  investigate   Run a sample investigation"
	@echo "  report        Generate a sample report"
	@echo "  clean         Remove caches and build artefacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts examples

format:
	ruff format src tests scripts examples

data:
	$(PYTHON) scripts/generate_dataset.py --config configs/data/default.yaml --n-samples 200 --seed 42

train:
	$(PYTHON) scripts/train_anomaly_model.py --config configs/models/anomaly_detector.yaml
	$(PYTHON) scripts/train_diagnosis_model.py --config configs/models/fault_classifier.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/experiments/baseline.yaml

investigate:
	$(PYTHON) scripts/run_investigation.py --config configs/system/default.yaml

report:
	$(PYTHON) scripts/generate_report.py --config configs/system/default.yaml

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
