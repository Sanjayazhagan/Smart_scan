import numpy as np

from scheduler.lean_observable_moe import LeanObservableMoEScheduler


class FakeRuntime:
    def __init__(self):
        self.state = {
            "band_belief": np.zeros(20, dtype=np.float32),
            "scan_age": np.zeros(20, dtype=np.float32),
        }

    def get_global_belief(self):
        return self.state

    def update(self, obs, timestamp):
        pass


def _scheduler():
    return LeanObservableMoEScheduler(
        runtime=FakeRuntime(), minimum_history=10, switch_evidence=2,
        minimum_mode_dwell=0
    )


def test_observable_router_recognizes_stationary_history():
    scheduler = _scheduler()
    scheduler.history.extend((3, True, 0.9) for _ in range(12))
    assert scheduler._candidate_mode(scheduler._history_features()) == "stationary"


def test_observable_router_recognizes_harsh_history():
    scheduler = _scheduler()
    scheduler.history.extend((i % 4, True, 0.1) for i in range(12))
    assert scheduler._candidate_mode(scheduler._history_features()) == "harsh"


def test_observable_router_recognizes_hopping_history():
    scheduler = _scheduler()
    scheduler.history.extend((i % 5, True, 0.8) for i in range(12))
    assert scheduler._candidate_mode(scheduler._history_features()) == "hopping"


def test_moe_selects_valid_band_and_ignores_simulator_reward():
    scheduler = _scheduler()
    action = scheduler.select_band()
    observation = {
        "detected": True,
        "quality": np.array([0.8], dtype=np.float32),
        "signal_power": np.array([1.0], dtype=np.float32),
    }
    scheduler.update(action, 9999.0, observation)
    assert 0 <= action < 20
    assert scheduler.core.values[action] <= 1.0
