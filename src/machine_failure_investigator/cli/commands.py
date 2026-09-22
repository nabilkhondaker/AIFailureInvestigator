"""CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from machine_failure_investigator.cli.formatting import print_info, print_success
from machine_failure_investigator.config.loader import load_config
from machine_failure_investigator.utils.logging import setup_logging
from machine_failure_investigator.utils.seeds import set_seed


def cmd_generate_data(
    config: Path = typer.Option(Path("configs/data/default.yaml"), "--config", "-c"),
    n_samples: Optional[int] = typer.Option(None, "--n-samples"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    """Generate synthetic multi-sensor dataset."""
    from machine_failure_investigator.simulation.dataset_builder import DatasetBuilder

    cfg = load_config(config)
    setup_logging()
    seed = seed if seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    n = n_samples if n_samples is not None else int(cfg.get("n_samples", 500))
    builder = DatasetBuilder(
        sample_rate=float(cfg.get("sampling_rate_hz", 5120)),
        duration_s=float(cfg.get("duration_s", 1.0)),
        seed=seed,
    )
    print_info(f"Generating {n} samples (seed={seed})...")
    bundles = builder.generate(
        n_samples=n,
        fault_distribution=cfg.get("fault_distribution"),
        severity_range=tuple(cfg.get("severity_range", [0.1, 1.0])),
        rpm_range=tuple(cfg.get("rpm_range", [800, 3000])),
        load_range=tuple(cfg.get("load_range", [0.3, 1.0])),
    )
    split_cfg = cfg.get("split", {})
    splits = builder.split(
        bundles,
        train=float(split_cfg.get("train", 0.7)),
        val=float(split_cfg.get("val", 0.15)),
        test=float(split_cfg.get("test", 0.15)),
    )
    out = cfg.get("output_dir", "data/synthetic")
    builder.save(splits, out, metadata={"config": str(config)})
    print_success(f"Dataset written to {out}")


def cmd_train(
    config: Path = typer.Option(Path("configs/models/fault_classifier.yaml"), "--config", "-c"),
) -> None:
    """Train a diagnosis or anomaly model from config."""
    from machine_failure_investigator.pipelines.training import features_from_npz
    from machine_failure_investigator.training.trainer import train_anomaly_detector, train_classifier

    cfg = load_config(config)
    setup_logging()
    set_seed(int(cfg.get("seed", 42)))
    data_path = Path(cfg.get("training", {}).get("data_path", "data/synthetic"))
    train_file = data_path / "train.npz"
    if not train_file.exists():
        print_info("No train.npz found — generating a small dataset first...")
        cmd_generate_data(config=Path("configs/data/default.yaml"), n_samples=300, seed=42)
    X, y, severity, names = features_from_npz(train_file)
    model_type = cfg.get("model_type", "random_forest")
    out_dir = Path(cfg.get("training", {}).get("output_dir", "models"))
    name = cfg.get("training", {}).get("checkpoint_name", "model")
    if model_type in ("isolation_forest", "statistical", "autoencoder"):
        mask = y == "HEALTHY"
        Xh = X[mask] if mask.any() else X
        path = out_dir / f"{name}.pkl"
        train_anomaly_detector(Xh, model_type="isolation_forest", output_path=path)
    else:
        path = out_dir / f"{name}.pkl"
        train_classifier(X, y, model_type=model_type if model_type != "mlp" else "random_forest",
                         feature_names=names, output_path=path)
    print_success(f"Model saved to {path}")


def cmd_investigate(
    config: Path = typer.Option(Path("configs/system/default.yaml"), "--config", "-c"),
) -> None:
    """Run a sample end-to-end investigation on simulated bearing degradation."""
    from machine_failure_investigator.diagnosis.investigator import Investigator
    from machine_failure_investigator.machine.faults import FaultSpec, FaultType
    from machine_failure_investigator.machine.model import RotatingMachine
    from machine_failure_investigator.machine.operating_conditions import OperatingCondition
    from machine_failure_investigator.reporting.report_builder import ReportBuilder
    from machine_failure_investigator.simulation.simulator import MachineSimulator

    setup_logging()
    inv = Investigator.from_config(config)
    machine = RotatingMachine()
    state = machine.state(
        operating=OperatingCondition(rpm=1800, load_fraction=0.75),
        fault=FaultSpec(FaultType.BEARING_DEGRADATION, 0.65),
        scenario_id="cli_demo",
    )
    sim = MachineSimulator()
    bundle = sim.simulate(state, duration_s=1.0, sample_rate=5120.0)
    result = inv.run(bundle)
    report = ReportBuilder().build(result, machine_id="CLI Demo Machine")
    print(report)
    print_success("Investigation complete.")


def cmd_evaluate(
    config: Path = typer.Option(Path("configs/experiments/baseline.yaml"), "--config", "-c"),
) -> None:
    """Evaluate a trained classifier on the test split if available."""
    from machine_failure_investigator.evaluation.classification import classification_report_dict
    from machine_failure_investigator.models.baseline import RandomForestClassifierModel
    from machine_failure_investigator.pipelines.training import features_from_npz

    setup_logging()
    test_path = Path("data/synthetic/test.npz")
    model_path = Path("models/fault_classifier.pkl")
    if not test_path.exists() or not model_path.exists():
        print_info("Missing test data or model — run generate-data and train first.")
        return
    X, y, _, _ = features_from_npz(test_path)
    model = RandomForestClassifierModel.load(model_path)
    pred = model.predict(X)
    metrics = classification_report_dict(y, pred)
    print(metrics["sklearn_report"])
    print_success(f"Accuracy={metrics['accuracy']:.4f} Macro-F1={metrics['macro_f1']:.4f}")
