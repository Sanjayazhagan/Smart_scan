import numpy as np
import pytest
from scheduler.alarm_adaptive_nmf_ucb import AlarmAdaptiveNMFUCBScheduler

def test_alarm_adaptive_nmf_ucb_lifecycle():
    scheduler = AlarmAdaptiveNMFUCBScheduler(num_bands=20, alarm_threshold=0.5, alarm_warmup=2)
    assert scheduler.total_alarms == 0
    assert scheduler.world_model_scale == 0.0  # Zero neural action guidance

    dummy_obs = {
        "selected_band": 0,
        "detected": 1,
        "quality": np.array([0.9], dtype=np.float32),
        "iq": np.zeros((2, 512), dtype=np.float32),
    }

    for st in range(10):
        band = scheduler.select_band()
        assert 0 <= band < 20
        dummy_obs["selected_band"] = band
        scheduler.update(band, 1.0, dummy_obs)
