import numpy as np
import torch

from scheduler.track2_core import EmitterStateWorldModel, RuntimeIdentityEncoder
from scheduler.track2_runtime import (
    Track2Runtime,
    TemporalConsistencyGate,
    PerceptionAuthenticator,
)


def make_checkpoint(path):
    torch.manual_seed(3)
    identity = RuntimeIdentityEncoder(
        {"embedding.weight": torch.randn(64, 128)}, embedding_dim=64
    )
    state = EmitterStateWorldModel()
    prototype = torch.nn.functional.normalize(torch.randn(1, 64), dim=1)
    torch.save(
        {
            "identity_model_state_dict": identity.state_dict(),
            "state_model_state_dict": state.state_dict(),
            "embedding_dim": 64,
            "prototype_matrix": prototype,
            "prototype_ids": [0],
            "identity_unknown_threshold": 2.0,
            "num_bands": 20,
            "state_dim": 32,
            "band_embedding_dim": 8,
            "gru_hidden": 128,
            "n_fft": 64,
            "hop_length": 16,
            "confirm_after": 1,
            "candidate_match_threshold": 0.75,
            "smoothing_alpha": 0.7,
            "anomaly_thresholds": {
                "identity_novelty": 0.9,
                "behaviour_change": 0.9,
                "prediction_uncertainty": 0.9,
            },
        },
        path,
    )


def observation(detected):
    return {
        "selected_band": np.int64(7),
        "detected": np.int64(detected),
        "signal_power": np.array([8.0], dtype=np.float32),
        "quality": np.array([0.85], dtype=np.float32),
        "iq": np.random.default_rng(5).normal(size=(2, 512)).astype(np.float32),
    }


def test_runtime_belief_detection_no_detection_and_reset(tmp_path):
    checkpoint = tmp_path / "track2_final_world_model.pt"
    make_checkpoint(checkpoint)
    runtime = Track2Runtime(checkpoint, device="cpu", max_scan_age=10)

    initial = runtime.get_band_belief()
    assert initial.shape == (20,)
    assert np.all(np.isfinite(initial))
    assert np.all((initial >= 0.0) & (initial <= 1.0))
    initial_state = runtime.get_rl_state()
    assert initial_state.shape == (60,)
    assert np.array_equal(initial_state[:20], initial)
    assert np.all(initial_state[20:40] == 1.0)
    assert np.all(initial_state[40:] > 0.0)
    assert np.all(np.isfinite(initial_state))
    assert np.all((initial_state >= 0.0) & (initial_state <= 1.0))
    assert all(
        not parameter.requires_grad
        for model in (runtime.identity_model, runtime.state_model)
        for parameter in model.parameters()
    )

    original_track_ids = set(runtime.manager.tracks)
    runtime.update(observation(0), timestamp=0)
    assert runtime.update_count == 0
    assert set(runtime.manager.tracks) == original_track_ids
    assert all(track.observations == 0 for track in runtime.manager.tracks.values())
    assert runtime.get_scan_age()[7] == 0.0
    assert runtime.recent_miss[7] == 1.0
    assert runtime.get_band_uncertainty()[7] == 0.0

    second_miss = observation(0)
    second_miss["selected_band"] = np.int64(8)
    runtime.update(second_miss, timestamp=1)
    assert runtime.get_scan_age()[8] == 0.0
    assert runtime.get_scan_age()[7] > 0.0
    assert all(track.observations == 0 for track in runtime.manager.tracks.values())

    result = runtime.update(observation(1), timestamp=2)
    assert result is not None
    assert result["previous_next_probabilities"] is None
    assert runtime.update_count == 1
    assert "U001" in runtime.manager.tracks
    global_belief = runtime.get_global_belief()
    assert global_belief["track_mask"].sum() == 1
    assert global_belief["track_diagnostics"].shape == (16, 3)
    assert global_belief["prediction_error"][7] > 0.0
    assert np.isclose(global_belief["recent_hit"][7], 0.85)
    assert global_belief["recent_miss"][7] == 0.0
    assert set(global_belief["candidate_summary"]) == {
        "count",
        "mean_novelty",
        "max_novelty",
        "mean_quality",
        "mean_uncertainty",
    }

    repeated_result = runtime.update(observation(1), timestamp=3)
    assert repeated_result is not None
    previous = repeated_result["previous_next_probabilities"]
    assert previous is not None
    assert tuple(previous.shape) == (21,)

    runtime.reset()
    assert runtime.update_count == 0
    assert runtime.last_result is None
    assert "U001" not in runtime.manager.tracks
    assert all(track.observations == 0 for track in runtime.manager.tracks.values())
    assert np.all(runtime.get_scan_age() == 1.0)
    assert np.all(runtime.recent_miss == 0.0)
    assert np.all(runtime.get_recent_hit() == 0.0)
    assert np.all(runtime.get_prediction_error() == 0.0)
    assert np.all(runtime.get_band_uncertainty() > 0.0)


def test_temporal_consistency_gate():
    gate = TemporalConsistencyGate(m=2)
    # Step 1 on band 3: hit -> should not confirm yet (M=2)
    assert not gate.update(3, True)
    # Step 2 on band 3: consecutive hit -> confirmed!
    assert gate.update(3, True)
    # Step 3 on band 3: miss -> resets
    assert not gate.update(3, False)
    # Step 4 on band 3: hit -> not confirmed yet
    assert not gate.update(3, True)
    # Step 5: retune to band 4 with hit -> resets band 3, band 4 not confirmed yet
    assert not gate.update(4, True)
    # Step 6: retune back to band 3 with hit -> band 3 was reset, so 1st hit -> False
    assert not gate.update(3, True)
    # Step 7 on band 3: 2nd consecutive hit -> True
    assert gate.update(3, True)

    # Test M-of-N sliding window
    gate_mn = TemporalConsistencyGate(m=2, n=3)
    assert not gate_mn.update(1, True)
    assert gate_mn.update(1, True)
    assert gate_mn.update(1, False)  # in window of 3: [True, True, False], sum is 2 >= 2 -> True
    assert not gate_mn.update(1, False)  # in window: [True, False, False], sum is 1 < 2 -> False


def test_perception_authenticator(tmp_path):
    checkpoint = tmp_path / "track2_final_world_model.pt"
    make_checkpoint(checkpoint)
    runtime = Track2Runtime(checkpoint, device="cpu", temporal_m=2)
    authenticator = PerceptionAuthenticator(runtime=runtime, identity_threshold=0.50, temporal_m=2)

    dummy_iq = np.random.default_rng(42).normal(size=(2, 512)).astype(np.float32)

    # Pass 1: raw_detected is False -> not authenticated
    res1 = authenticator.authenticate(band=5, iq=dummy_iq, raw_detected=False)
    assert not res1["authenticated"]
    assert res1["consecutive_hits"] == 0

    # Pass 2: raw_detected is True -> 1st hit, not confirmed yet under M=2
    res2 = authenticator.authenticate(band=5, iq=dummy_iq, raw_detected=True, custom_threshold=-1.0)
    assert not res2["authenticated"]
    assert res2["snapshot_authenticated"]
    assert res2["consecutive_hits"] == 1

    # Pass 3: 2nd consecutive hit -> confirmed under M=2!
    res3 = authenticator.authenticate(band=5, iq=dummy_iq, raw_detected=True, custom_threshold=-1.0)
    assert res3["authenticated"]
    assert res3["consecutive_hits"] == 2

