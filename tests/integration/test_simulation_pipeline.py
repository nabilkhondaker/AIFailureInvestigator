from machine_failure_investigator.simulation.dataset_builder import DatasetBuilder


def test_generate_small_dataset(tmp_path):
    builder = DatasetBuilder(seed=1, duration_s=0.25, sample_rate=2048.0)
    bundles = builder.generate(n_samples=12)
    assert len(bundles) == 12
    splits = builder.split(bundles, train=0.5, val=0.25, test=0.25)
    out = builder.save(splits, tmp_path / "ds")
    assert (out / "train.npz").exists()
    assert (out / "metadata.json").exists()
