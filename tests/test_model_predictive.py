import numpy as np

from scheduler.model_predictive import ModelPredictiveScheduler, PlannerConfig


class FakeRuntime:
    def __init__(self):
        self.belief = np.zeros(20, dtype=np.float32)
        self.belief[5] = 0.9
        self.age = np.linspace(0.0, 1.0, 20, dtype=np.float32)
        self.uncertainty = np.full(20, 0.5, dtype=np.float32)
        self.updates = []

    def get_band_belief(self):
        return self.belief.copy()

    def get_scan_age(self, normalized=True):
        return self.age.copy()

    def get_band_uncertainty(self):
        return self.uncertainty.copy()

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))


def test_depth_three_planner_selects_high_value_band():
    runtime = FakeRuntime()
    scheduler = ModelPredictiveScheduler(
        20,
        runtime=runtime,
        config=PlannerConfig(
            depth=3,
            beam_width=8,
            information_weight=0.0,
            scan_age_weight=0.0,
        ),
    )
    action = scheduler.select_band()
    assert action == 5
    assert len(scheduler.last_plan) == 3
    assert all(0 <= band < 20 for band in scheduler.last_plan)
    assert np.isfinite(scheduler.last_plan_score)


def test_planner_update_uses_only_partial_observation():
    runtime = FakeRuntime()
    scheduler = ModelPredictiveScheduler(20, runtime=runtime)
    observation = {
        "selected_band": 5,
        "detected": 0,
        "signal_power": np.array([0.1], dtype=np.float32),
        "quality": np.array([0.1], dtype=np.float32),
        "iq": np.zeros((2, 512), dtype=np.float32),
    }
    scheduler.update(5, 0.1, observation)
    assert runtime.updates == [(observation, 0.0)]
    assert not any("ground_truth" in key for key in observation)
