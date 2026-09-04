import numpy as np
import pytest
from scheduler.dual_policy_uncertainty_scheduler import (
    DualPolicyUncertaintyScheduler,
    EnsembleUncertaintyEstimator,
)

def test_ensemble_uncertainty_estimator_sanity():
    est = EnsembleUncertaintyEstimator(num_bands=20)
    for t in range(40):
        # Band 0 stable
        est.update(0, 1.0)
        # Band 1 volatile
        est.update(1, float(t % 2 == 0))
    # Band 5 unvisited
    unc = est.get_uncertainty()
    assert unc[1] > unc[0], f"Volatile ({unc[1]}) should exceed Stable ({unc[0]})"
    assert unc[5] > unc[0], f"Unvisited ({unc[5]}) should exceed Stable ({unc[0]})"

def test_dual_policy_scheduler_lifecycle():
    sched = DualPolicyUncertaintyScheduler(num_bands=20, switch_penalty=0.05, explore_budget_prob=0.3)
    for _ in range(25):
        a = sched.select_band()
        assert 0 <= a < 20
        dummy_obs = {"selected_band": a, "detected": 1, "quality": [0.8], "iq": np.zeros((2, 512))}
        sched.update(a, 1.0, dummy_obs)
    assert sched.exploit_decisions + sched.explore_decisions > 0
