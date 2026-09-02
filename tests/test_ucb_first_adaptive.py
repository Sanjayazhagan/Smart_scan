import numpy as np

from scheduler.adaptive_moe import (
    AdaptiveExplorationConfig,
    AdaptiveMixtureOfExpertsScheduler,
)
from scheduler.emitter_aware_predictive import CONFIRMED
from scheduler.ucb_first_adaptive import UCBFirstAdaptiveScheduler


def _state(uncertainty=0.1):
    rows = np.zeros((4, 48), dtype=np.float32)
    rows[0, 3] = 0.9
    rows[0, 41:47] = [uncertainty, 0.05, 0.05, 0.2, 0.9, 0.9]
    rows[0, CONFIRMED] = 1.0
    return {
        "track_features": rows,
        "track_mask": np.array([1, 0, 0, 0], dtype=np.float32),
        "band_belief": np.asarray([0.9, 0.1] + [0.0] * 18, dtype=np.float32),
        "scan_age": np.full(20, 0.1, dtype=np.float32),
        "band_uncertainty": np.full(20, uncertainty, dtype=np.float32),
        "prediction_error": np.zeros(20, dtype=np.float32),
        "recent_hit": np.zeros(20, dtype=np.float32),
        "recent_miss": np.zeros(20, dtype=np.float32),
        "candidate_summary": {},
        "track_ids": ["track"],
    }


class FakeRuntime:
    def __init__(self):
        self.state = _state()
        self.updates = []

    def get_global_belief(self):
        return self.state

    def get_band_belief(self):
        return self.state["band_belief"].copy()

    def get_scan_age(self, normalized=True):
        return self.state["scan_age"].copy()

    def get_band_uncertainty(self):
        return self.state["band_uncertainty"].copy()

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))


def _scheduler(runtime):
    adaptive = AdaptiveMixtureOfExpertsScheduler(
        20,
        runtime=runtime,
        exploration=AdaptiveExplorationConfig(
            easy_epsilon=0.0,
            intermediate_epsilon=0.0,
            complex_epsilon=0.0,
            dynamic_epsilon=0.0,
            unknown_epsilon=0.0,
        ),
    )
    return UCBFirstAdaptiveScheduler(20, adaptive=adaptive)


def test_easy_state_uses_ucb_primary():
    scheduler = _scheduler(FakeRuntime())
    assert scheduler.select_band() == 0
    assert scheduler.last_decision_trace["controller_mode"] == "ucb_primary"
    assert scheduler.last_decision_trace["expert"] == "ucb"


def test_instantaneous_complex_state_remains_on_ucb():
    runtime = FakeRuntime()
    runtime.state["band_belief"] = np.asarray(
        [0.48, 0.46, 0.44, 0.42, 0.40] + [0.05] * 15, dtype=np.float32
    )
    runtime.state["band_uncertainty"][:] = 0.9
    runtime.state["track_features"][0, 41] = 0.9
    scheduler = _scheduler(runtime)
    assert 0 <= scheduler.select_band() < 20
    assert scheduler.last_decision_trace["controller_mode"] == "ucb_primary"
    assert scheduler.last_decision_trace["expert"] == "ucb"


def test_stationary_history_escalates_and_update_is_counted_once():
    runtime = FakeRuntime()
    scheduler = _scheduler(runtime)
    scheduler.adaptive.regime_monitor.history_regime = "clean_stationary"
    scheduler.select_band()
    assert scheduler.last_decision_trace["controller_mode"] == "adaptive_escalation"
    observation = {"detected": True, "quality": np.array([0.8], dtype=np.float32)}
    scheduler.update(0, 1.0, observation)
    assert len(runtime.updates) == 1
    assert scheduler.ucb_expert.counts[0] > 0.0
