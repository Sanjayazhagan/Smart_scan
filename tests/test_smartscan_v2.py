import numpy as np
import pytest

from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH
from scheduler.smart_discounted_ucb import SmartDiscountedUCB
from scheduler.belief_expectimax import BeliefExpectimaxPlanner, BeliefState, _hash_state
from scheduler.smartscan_v2 import SmartScanScheduler


def test_track2_investigation_priority_and_reliability():
    runtime = Track2Runtime(model_path=DEFAULT_MODEL_PATH)
    inv = runtime.get_investigation_priority()
    rel = runtime.get_prediction_reliability()
    assert inv.shape == (20,)
    assert 0.0 <= rel <= 1.0
    assert np.all((inv >= 0.0) & (inv <= 1.0))

    # Feeding observations
    for t in range(5):
        obs = {
            "selected_band": t,
            "detected": 1,
            "quality": np.array([0.9], dtype=np.float32),
            "iq": np.ones((2, 512), dtype=np.float32) * 0.1,
        }
        runtime.update(obs, timestamp=float(t))

    inv_after = runtime.get_investigation_priority()
    assert inv_after.shape == (20,)
    assert 0.0 <= runtime.get_prediction_reliability() <= 1.0


def test_smart_discounted_ucb():
    ucb = SmartDiscountedUCB(num_bands=20, default_k=5, adaptive_k=True)
    assert ucb.counts.shape == (20,)
    assert ucb.values.shape == (20,)

    # Initial sweep when counts are 0
    b_zeros = np.zeros(20, dtype=np.float32)
    top_k, scores, k = ucb.get_top_k_candidates(b_zeros, b_zeros, b_zeros, b_zeros)
    assert len(top_k) >= 2
    assert 0 <= top_k[0] < 20

    # Simulate updates
    for b in range(20):
        ucb.update(b, detected=True, quality=0.8)

    belief = np.linspace(0.1, 0.9, 20, dtype=np.float32)
    top_k, scores, k = ucb.get_top_k_candidates(belief, b_zeros, b_zeros, b_zeros)
    assert len(top_k) == k
    assert 2 <= k <= 6
    # Top-1 candidate must match argmax of scores
    assert top_k[0] == int(np.argmax(scores))


def test_belief_expectimax_planner():
    planner = BeliefExpectimaxPlanner(num_bands=20, depth=2, branch_k=3, enable_caching=True)
    root = BeliefState(
        band_belief=np.full(20, 0.3, dtype=np.float32),
        band_uncertainty=np.full(20, 0.5, dtype=np.float32),
        scan_age=np.full(20, 0.5, dtype=np.float32),
        ucb_values=np.zeros(20, dtype=np.float32),
        ucb_counts=np.ones(20, dtype=np.float32),
        investigation_priority=np.zeros(20, dtype=np.float32),
        last_band=0,
    )
    # Band 5 has high belief
    root.band_belief[5] = 0.95

    # Test transitions
    hit_state = planner.transition_belief(root, action=5, detected=True)
    miss_state = planner.transition_belief(root, action=5, detected=False)
    assert hit_state.band_belief[5] > root.band_belief[5]
    assert miss_state.band_belief[5] < root.band_belief[5]
    assert hit_state.band_uncertainty[5] < root.band_uncertainty[5]
    assert hit_state.scan_age[5] == 0.0

    # Test lookahead planning
    best_action, exp_val, stats = planner.plan(root, top_k_candidates=[0, 1, 5])
    assert best_action in [0, 1, 5]
    assert stats["nodes_expanded"] > 0
    assert not stats["timed_out"]


def test_smartscan_v2_scheduler_lifecycle():
    scheduler = SmartScanScheduler(num_bands=20, depth=2, top_k=4)
    scheduler.reset()

    for st in range(30):
        band = scheduler.select_band()
        assert 0 <= band < 20
        det = 1 if st % 3 == 0 else 0
        qual = 0.85 if det else 0.0
        obs = {
            "selected_band": band,
            "detected": det,
            "quality": np.array([qual], dtype=np.float32),
            "iq": np.zeros((2, 512), dtype=np.float32),
        }
        scheduler.update(band, float(det), obs)

    diag = scheduler.get_diagnostics()
    assert "selected_band" in diag
    assert "selection_mode" in diag
    assert "override" in diag
    assert "top_k_candidates" in diag
    assert "step_count" in diag
    assert diag["step_count"] == 30
