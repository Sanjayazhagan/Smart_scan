import numpy as np

from scheduler.calibrated_soft_moe import (
    CalibratedSoftMoEScheduler,
    CalibratedWorldNMFUCBScheduler,
    OnlineBrierCalibrator,
)


class FakeRuntime:
    def __init__(self):
        self.state = {
            "band_belief": np.asarray([0.8] + [0.01] * 19, dtype=np.float32),
            "scan_age": np.zeros(20, dtype=np.float32),
        }
        self.updates = []

    def get_global_belief(self):
        return self.state

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))
        return {"known_identity": True}


def test_brier_calibrator_trusts_skillful_forecasts_and_rejects_bad_ones():
    good = OnlineBrierCalibrator(prior_strength=5.0)
    bad = OnlineBrierCalibrator(prior_strength=5.0)
    outcomes = [1.0, 0.0] * 20
    for outcome in outcomes:
        good.update(0.9 if outcome else 0.1, outcome)
        bad.update(0.1 if outcome else 0.9, outcome)
    assert good.skill > 0.3
    assert bad.skill == 0.0


def test_calibrated_ablation_starts_with_neural_guidance_disabled():
    scheduler = CalibratedWorldNMFUCBScheduler(
        runtime=FakeRuntime(), nmf_scale=1.0, world_model_scale=0.35
    )
    scheduler.counts[:] = 2.0
    scheduler.total_observations = 40.0
    scheduler.select_band()
    assert scheduler.last_trace["world_calibration_skill"] == 0.0
    assert scheduler.last_trace["world_model_weight"] == 0.0


def test_soft_router_weights_are_finite_and_sum_to_one():
    scheduler = CalibratedSoftMoEScheduler(runtime=FakeRuntime(), minimum_history=1)
    scheduler.history.extend((3, True, 0.9) for _ in range(10))
    regime = scheduler._regime_weights(scheduler._features())
    expert = scheduler._expert_weights(regime)
    assert np.isclose(sum(regime.values()), 1.0)
    assert np.isclose(sum(expert.values()), 1.0)
    assert max(regime, key=regime.get) == "stationary"


def test_soft_router_ignores_hidden_reward_and_updates_runtime_once():
    runtime = FakeRuntime()
    scheduler = CalibratedSoftMoEScheduler(runtime=runtime)
    action = scheduler.select_band()
    obs = {
        "detected": True,
        "quality": np.array([0.8], dtype=np.float32),
        "signal_power": np.array([1.0], dtype=np.float32),
    }
    scheduler.update(action, 9999.0, obs)
    assert 0 <= action < 20
    assert len(runtime.updates) == 1
    assert scheduler.core.values[action] <= 1.0
