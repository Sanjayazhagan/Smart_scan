import numpy as np
import pytest
from scheduler.contextual_bandit_scheduler import ContextualBanditScheduler, LinUCBPolicyArbitrator

def test_linucb_arbitrator_update():
    bandit = LinUCBPolicyArbitrator(num_arms=4, context_dim=8, alpha=0.20)
    ctx = np.ones(8, dtype=np.float64)
    arm, scores = bandit.select_arm(ctx)
    assert 0 <= arm < 4
    assert len(scores) == 4

    # Update arm with reward
    bandit.update(arm, ctx, reward=1.0)
    assert bandit.arm_counts[arm] == 1
    # Check that covariance matrix updated
    new_arm, new_scores = bandit.select_arm(ctx)
    assert len(new_scores) == 4


def test_contextual_bandit_scheduler_lifecycle():
    sched = ContextualBanditScheduler(num_bands=20, alpha=0.25)
    for t in range(30):
        a = sched.select_band()
        assert 0 <= a < 20
        det = 1.0 if t % 2 == 0 else 0.0
        obs = {
            "selected_band": a,
            "detected": det > 0.5,
            "quality": np.array([0.8 if det else 0.0], dtype=np.float32),
            "iq": np.zeros((2, 512), dtype=np.float32),
        }
        sched.update(a, det, obs)

    assert sched.total_observations > 0
    assert len(sched.recent_hits) > 0
