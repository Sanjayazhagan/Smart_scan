import numpy as np

from scheduler.world_model_ucb import WorldModelUCBScheduler


def _state(probability=0.9, uncertainty=0.1):
    rows = np.zeros((4, 48), dtype=np.float32)
    rows[0, 3] = probability
    rows[0, 41] = uncertainty
    rows[0, 45] = 0.9
    rows[0, 47] = 1.0
    return {
        "track_features": rows,
        "track_mask": np.array([1, 0, 0, 0], dtype=np.float32),
        "band_belief": np.asarray([0.05, 0.05, 0.05, probability] + [0.0] * 16),
        "scan_age": np.full(20, 0.1, dtype=np.float32),
        "band_uncertainty": np.full(20, uncertainty, dtype=np.float32),
        "prediction_error": np.zeros(20, dtype=np.float32),
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

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))


def _warm_counts(scheduler):
    scheduler.counts[:] = 10.0
    scheduler.total_observations = 200.0


def test_world_model_ucb_starts_with_full_band_coverage():
    scheduler = WorldModelUCBScheduler(runtime=FakeRuntime())
    assert scheduler.select_band() == 0
    assert scheduler.last_trace["mode"] == "initial_coverage"
    assert scheduler.last_trace["world_model_weight"] == 0.0


def test_world_model_runs_pure_ucb_when_scale_zero():
    scheduler = WorldModelUCBScheduler(runtime=FakeRuntime(), world_model_scale=0.0)
    _warm_counts(scheduler)
    scheduler.select_band()
    assert scheduler.last_trace["world_model_weight"] == 0.0


def test_neural_world_model_directly_guides_ucb():
    scheduler = WorldModelUCBScheduler(runtime=FakeRuntime(), world_model_scale=0.65)
    _warm_counts(scheduler)
    assert scheduler.select_band() == 3
    assert scheduler.last_trace["mode"] == "world_model_guided_ucb"
    assert scheduler.last_trace["world_model_weight"] == 0.65
    assert scheduler.last_trace["world_model_contribution"] > 0.0


def test_update_uses_observation_and_advances_runtime_once():
    runtime = FakeRuntime()
    scheduler = WorldModelUCBScheduler(runtime=runtime)
    scheduler.select_band()
    observation = {"detected": True, "quality": np.array([0.8], dtype=np.float32)}
    scheduler.update(0, 123.0, observation)
    assert len(runtime.updates) == 1
    assert scheduler.counts[0] > 0.0
    assert scheduler.values[0] > 0.0
