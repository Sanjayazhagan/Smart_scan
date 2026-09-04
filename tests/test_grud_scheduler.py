import numpy as np
import pytest
from scheduler.grud_spectrum_model import GRUDSpectrumCell, NumpyGRUDModel
from scheduler.grud_scheduler import GRUDSpectrumScheduler


def test_grud_cell_shapes():
    cell = GRUDSpectrumCell(num_bands=20, hidden_dim=32)
    numpy_model = NumpyGRUDModel(cell)

    x = np.zeros(20, dtype=np.float64)
    x[3] = 1.0
    m = np.zeros(20, dtype=np.float64)
    m[3] = 1.0
    delta = np.ones(20, dtype=np.float64) * 3.0
    last_x = np.zeros(20, dtype=np.float64)
    h = np.zeros(32, dtype=np.float64)

    probs, h_next, last_x_next = numpy_model.forward_step(x, m, delta, last_x, h)
    assert probs.shape == (20,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert h_next.shape == (32,)
    assert last_x_next[3] == 1.0


def test_grud_scheduler_lifecycle():
    sched = GRUDSpectrumScheduler(num_bands=20, hidden_dim=32)
    sched.reset()

    for t in range(50):
        b = sched.select_band()
        assert 0 <= b < 20
        det = (t % 3 == 0)
        obs = {
            "selected_band": b,
            "detected": det,
            "quality": np.array([0.75 if det else 0.0], dtype=np.float32),
        }
        sched.update(b, 1.0 if det else -0.1, obs)

    assert sched.step_count == 50
