import numpy as np
import pytest
from pathlib import Path

from scheduler.rl_value_network import (
    BeliefValueNetwork,
    FastBeliefValueCritic,
    extract_belief_features,
    DEFAULT_VALUE_NET_PATH,
)
from scheduler.nmf_expectimax import NMFExpectimaxScheduler

def test_belief_value_network_forward():
    net = BeliefValueNetwork(input_dim=61)
    x = np.random.randn(61).astype(np.float32)
    val = net.predict_scalar(x)
    assert isinstance(val, float)

def test_fast_belief_value_critic():
    critic = FastBeliefValueCritic(DEFAULT_VALUE_NET_PATH)
    assert critic.loaded is True
    b = np.full(20, 0.5, dtype=np.float32)
    u = np.full(20, 0.2, dtype=np.float32)
    a = np.full(20, 0.8, dtype=np.float32)
    val = critic.evaluate_leaf(b, u, a, last_band=3)
    assert isinstance(val, float)

def test_rl_guided_nmf_expectimax_lifecycle():
    sched = NMFExpectimaxScheduler(num_bands=20, depth=2, top_k=4, use_rl_critic=True)
    assert sched.value_critic is not None
    assert sched.value_critic.loaded is True

    for st in range(25):
        action = sched.select_band()
        assert 0 <= action < 20
        obs = {
            "selected_band": action,
            "detected": 1 if st % 2 == 0 else 0,
            "quality": np.array([0.85], dtype=np.float32),
            "iq": np.zeros((2, 512), dtype=np.float32),
        }
        sched.update(action, 1.0 if st % 2 == 0 else 0.0, obs)

    diag = sched.get_diagnostics()
    assert diag["step_count"] == 25
