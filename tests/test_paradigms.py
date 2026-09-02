"""Tests for the 6 Advanced Comparative Paradigms."""

import pytest
import numpy as np
from scheduler.paradigms import (
    RobustPCAPSRScheduler,
    NMFScheduler,
    WhittleIndexRMABScheduler,
    ThompsonSamplingScheduler,
    Exp3BanditScheduler,
    AdaptiveReceiverSearchScheduler,
    DoubleDQNScheduler,
)


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        RobustPCAPSRScheduler,
        NMFScheduler,
        WhittleIndexRMABScheduler,
        ThompsonSamplingScheduler,
        Exp3BanditScheduler,
        AdaptiveReceiverSearchScheduler,
        DoubleDQNScheduler,
    ],
)
def test_paradigm_lifecycle(scheduler_cls):
    """Verifies that all baseline paradigms initialize, select valid bands, and update."""
    num_bands = 20
    scheduler = scheduler_cls(num_bands=num_bands)

    for step in range(15):
        action = scheduler.select_band()
        assert 0 <= action < num_bands, f"Action {action} out of bounds for {scheduler_cls}"

        # Simulate periodic hits and misses
        is_hit = (step % 3 == 0)
        reward = 1.0 if is_hit else -0.2
        obs_dict = {
            "detected": 1 if is_hit else 0,
            "signal_power": 4.5 if is_hit else 0.05,
        }

        scheduler.update(action, reward, obs_dict)
