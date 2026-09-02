import numpy as np

from scheduler.emitter_attention import (
    TRACK2_ATTENTION_FEATURES,
    EmitterAttentionEncoder,
)

from tests.test_adaptive_moe import global_state


def test_emitter_attention_is_fixed_width_masked_and_deterministic():
    state = global_state()
    encoder = EmitterAttentionEncoder()
    first = encoder.encode(state)
    state["track_features"][1, :20] = 1.0
    second = encoder.encode(state)
    assert first.shape == (TRACK2_ATTENTION_FEATURES,)
    assert np.array_equal(first, second)
    assert np.isclose(encoder.last_attention[:, 1].sum(), 0.0)
    assert np.all((first >= 0.0) & (first <= 1.0))

