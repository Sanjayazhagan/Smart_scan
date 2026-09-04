import numpy as np
import pytest

from scheduler.nmf_expectimax import NMFExpectimaxScheduler
from scheduler.track2_runtime import DEFAULT_MODEL_PATH

def test_nmf_expectimax_lifecycle():
    sched = NMFExpectimaxScheduler(num_bands=20, top_k=4, depth=2, model_path=DEFAULT_MODEL_PATH)
    sched.reset()

    for st in range(35):
        action = sched.select_band()
        assert 0 <= action < 20
        det = 1 if st % 2 == 0 else 0
        qual = 0.85 if det else 0.0
        obs = {
            "selected_band": action,
            "detected": det,
            "quality": np.array([qual], dtype=np.float32),
            "iq": np.zeros((2, 512), dtype=np.float32),
        }
        sched.update(action, float(det), obs)

    diag = sched.get_diagnostics()
    assert "selected_band" in diag
    assert "selection_mode" in diag
    assert "nmf_top1" in diag
    assert "top_k_candidates" in diag
    assert diag["step_count"] == 35
