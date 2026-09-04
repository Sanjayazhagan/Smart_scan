import numpy as np
import pytest
from detector.pattern_change_detector import PatternChangeDetector

def test_pattern_change_detector_lifecycle():
    detector = PatternChangeDetector(warmup_steps=5, threshold=1.0)
    assert detector.timestep == 0

    dummy_obs_detected = {
        "selected_band": 0,
        "detected": 1,
        "quality": np.array([0.8], dtype=np.float32),
        "iq": np.zeros((2, 512), dtype=np.float32),
    }
    dummy_obs_silent = {
        "selected_band": 0,
        "detected": 0,
        "quality": np.array([0.0], dtype=np.float32),
        "iq": np.zeros((2, 512), dtype=np.float32),
    }

    for st in range(10):
        obs = dummy_obs_detected if st % 2 == 0 else dummy_obs_silent
        res = detector.step_and_detect(band=0, obs_dict=obs)
        assert "alarm" in res
        assert "prediction_error" in res
        assert "ph_statistic" in res
        assert res["timestep"] == st + 1

    detector.reset()
    assert detector.timestep == 0
    assert len(detector.error_history) == 0
