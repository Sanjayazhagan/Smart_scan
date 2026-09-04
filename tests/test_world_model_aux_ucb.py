import numpy as np
import pytest

from scheduler.world_model_aux_ucb import WorldModelAuxUCBScheduler


class FakeRuntime:
    def __init__(self):
        self.state = {
            "band_belief": np.zeros(20, dtype=np.float32),
            "scan_age": np.zeros(20, dtype=np.float32),
        }
        self.updates = []

    def get_global_belief(self):
        return self.state

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))


@pytest.mark.parametrize("auxiliary", ["rpca", "pri", "exp3"])
def test_auxiliary_fusions_select_valid_band_and_update_once(auxiliary):
    runtime = FakeRuntime()
    scheduler = WorldModelAuxUCBScheduler(
        runtime=runtime,
        auxiliary=auxiliary,
        auxiliary_scale=1.0,
        world_model_scale=0.0,
    )
    scheduler.counts[:] = 10.0
    scheduler.total_observations = 200.0
    band = scheduler.select_band()
    assert 0 <= band < 20
    observation = {
        "detected": True,
        "quality": np.array([0.8], dtype=np.float32),
        "signal_power": np.array([1.0], dtype=np.float32),
    }
    scheduler.update(band, 999.0, observation)
    assert len(runtime.updates) == 1


def test_unknown_auxiliary_is_rejected():
    with pytest.raises(ValueError, match="Unknown auxiliary"):
        WorldModelAuxUCBScheduler(runtime=FakeRuntime(), auxiliary="unknown")
