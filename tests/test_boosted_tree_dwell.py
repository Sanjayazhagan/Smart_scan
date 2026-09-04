import numpy as np
import pytest
from scheduler.boosted_tree_dwell_scheduler import CompactGBDTClassifier, BoostedTreeDwellScheduler


def test_compact_gbdt_classifier():
    # Simple XOR-like synthetic problem
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(np.float64)

    model = CompactGBDTClassifier(n_estimators=15, max_depth=3, learning_rate=0.1)
    model.fit(X, y)

    preds = model.predict_proba(X)
    assert preds.shape == (100,)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
    # Accuracy check
    binary_preds = (preds > 0.5).astype(np.float64)
    accuracy = np.mean(binary_preds == y)
    assert accuracy > 0.70  # Should learn the decision pattern well


def test_boosted_tree_dwell_scheduler_lifecycle():
    sched = BoostedTreeDwellScheduler(num_bands=20)
    for t in range(50):
        b = sched.select_band()
        assert 0 <= b < 20
        det = (t % 4 == 0)
        obs = {
            "selected_band": b,
            "detected": det,
            "quality": np.array([0.80 if det else 0.0], dtype=np.float32),
        }
        sched.update(b, 1.0 if det else -0.1, obs)

    assert sched.step_count == 50
