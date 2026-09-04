import numpy as np

from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler


class FakeRuntime:
    def __init__(self):
        self.state = {
            "band_belief": np.asarray([0.0, 0.0, 0.0, 0.9] + [0.0] * 16),
            "scan_age": np.zeros(20, dtype=np.float32),
        }
        self.updates = []

    def get_global_belief(self):
        return self.state

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))


def _warm(scheduler):
    scheduler.counts[:] = 10.0
    scheduler.total_observations = 200.0


def test_nmf_forecast_can_change_the_fused_ucb_action():
    scheduler = WorldModelNMFUCBScheduler(
        runtime=FakeRuntime(), world_model_scale=0.0, nmf_scale=2.0
    )
    _warm(scheduler)
    scheduler.nmf.predicted_spectrum[:] = 0.0
    scheduler.nmf.predicted_spectrum[5] = 1.0
    assert scheduler.select_band() == 5
    assert scheduler.last_trace["nmf_contribution"] > 0.0


def test_fusion_updates_world_runtime_and_nmf_once():
    runtime = FakeRuntime()
    scheduler = WorldModelNMFUCBScheduler(runtime=runtime, nmf_recompute_every=10)
    scheduler.select_band()
    obs = {
        "detected": True,
        "quality": np.array([0.8], dtype=np.float32),
        "signal_power": np.array([1.0], dtype=np.float32),
    }
    scheduler.update(0, 999.0, obs)
    assert len(runtime.updates) == 1
    assert scheduler.nmf.update_count == 1
    assert scheduler.values[0] > 0.0
