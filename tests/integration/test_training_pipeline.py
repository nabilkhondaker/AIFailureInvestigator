import numpy as np

from machine_failure_investigator.models.baseline import RandomForestClassifierModel
from machine_failure_investigator.features.feature_pipeline import FeatureExtractor
from machine_failure_investigator.simulation.dataset_builder import DatasetBuilder


def test_train_on_tiny_set():
    builder = DatasetBuilder(seed=2, duration_s=0.25, sample_rate=2048.0)
    bundles = builder.generate(n_samples=30)
    extractor = FeatureExtractor(use_time_frequency=False)
    X = extractor.transform_many(bundles)
    y = np.array([b.state.label() for b in bundles])
    model = RandomForestClassifierModel(n_estimators=15, max_depth=5, random_state=2)
    model.fit(X, y, feature_names=extractor.feature_names)
    pred = model.predict(X)
    assert len(pred) == len(y)
