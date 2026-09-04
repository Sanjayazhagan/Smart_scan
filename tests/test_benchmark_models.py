"""Test that all models in benchmark_models registry can be instantiated and stepped."""

import numpy as np
import pytest
from benchmark_models import MODEL_REGISTRY, get_model


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_model_registry_lifecycle(model_name):
    sched = get_model(model_name, num_bands=20, seed=42)
    assert sched is not None

    # Step through 5 transitions
    for t in range(5):
        action = sched.select_band()
        assert 0 <= action < 20
        det = (t % 2 == 0)
        obs = {
            "selected_band": action,
            "detected": det,
            "quality": np.array([0.75 if det else 0.0], dtype=np.float32),
            "iq": np.zeros((2, 512), dtype=np.float32),
        }
        sched.update(action, 1.0 if det else -0.1, obs)
