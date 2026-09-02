from dataclasses import replace

import numpy as np

from scheduler.emitter_aware_predictive import (
    BEHAVIOUR_CHANGE,
    CONFIRMED,
    IDENTITY_NOVELTY,
    OBSERVATION_QUALITY,
    PREDICTION_UNCERTAINTY,
    EmitterAwareModelPredictivePlanner,
    EmitterAwarePlannerConfig,
    derive_emitter_band_features,
)


def track_row(band, probability, quality=0.8, confirmed=1.0):
    row = np.zeros(48, dtype=np.float32)
    row[band] = probability
    row[PREDICTION_UNCERTAINTY] = 0.4
    row[IDENTITY_NOVELTY] = 0.7
    row[BEHAVIOUR_CHANGE] = 0.8
    row[44] = 0.5
    row[OBSERVATION_QUALITY] = quality
    row[46] = 0.9
    row[CONFIRMED] = confirmed
    return row


class FakeRuntime:
    def __init__(self, rows=None, mask=None):
        self.belief = np.zeros(20, dtype=np.float32)
        self.belief[3] = 0.8
        self.age = np.zeros(20, dtype=np.float32)
        self.uncertainty = np.full(20, 0.2, dtype=np.float32)
        self.rows = np.asarray(rows if rows is not None else np.zeros((4, 48)), dtype=np.float32)
        self.mask = np.asarray(mask if mask is not None else np.zeros(4), dtype=np.float32)
        self.updates = []

    def get_band_belief(self):
        return self.belief.copy()

    def get_scan_age(self, normalized=True):
        return self.age.copy()

    def get_band_uncertainty(self):
        return self.uncertainty.copy()

    def get_global_belief(self):
        valid_count = int((self.mask > 0.5).sum())
        return {
            "track_features": self.rows.copy(),
            "track_mask": self.mask.copy(),
            "track_ids": [f"E{i}" for i in range(valid_count)],
            "band_belief": self.belief.copy(),
        }

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))


def test_track_mask_and_confirmation_exclude_padded_rows():
    rows = np.stack(
        [
            track_row(3, 0.6),
            track_row(3, 1.0),
            track_row(3, 0.9, confirmed=0.0),
            np.zeros(48, dtype=np.float32),
        ]
    )
    derived = derive_emitter_band_features(
        {"track_features": rows, "track_mask": np.array([1, 0, 1, 0])}
    )
    assert np.isclose(derived.dominant_emitter_probability[3], 0.6)
    assert derived.supporting_emitter_count[3] == 1


def test_zero_emitter_state_is_safe_and_stale_band_remains_explorable():
    runtime = FakeRuntime()
    runtime.belief[:] = 0.0
    runtime.uncertainty[:] = 0.0
    runtime.age[19] = 1.0
    planner = EmitterAwareModelPredictivePlanner(20, runtime=runtime)
    assert planner.select_band() == 19
    assert 0 <= planner.last_decision_trace["selected_band"] < 20


def test_low_quality_suppresses_anomaly_terms():
    rows = np.stack([track_row(3, 0.8, quality=0.0), np.zeros(48), np.zeros(48), np.zeros(48)])
    runtime = FakeRuntime(rows=rows, mask=np.array([1, 0, 0, 0]))
    planner = EmitterAwareModelPredictivePlanner(20, runtime=runtime)
    planner.plan()
    breakdown = planner._score_breakdown(planner._last_root, 3)
    assert breakdown["behaviour_investigation"] == 0.0
    assert breakdown["novelty_investigation"] == 0.0


def test_repeat_penalty_and_score_decomposition():
    rows = np.stack([track_row(3, 0.8), np.zeros(48), np.zeros(48), np.zeros(48)])
    planner = EmitterAwareModelPredictivePlanner(
        20, runtime=FakeRuntime(rows=rows, mask=np.array([1, 0, 0, 0]))
    )
    planner.plan()
    root = planner._last_root
    normal = planner._score_breakdown(root, 3)
    repeated = planner._score_breakdown(replace(root, sequence=(3,)), 3)
    assert np.isclose(
        repeated["total"], normal["total"] - planner.config.repeat_penalty
    )
    assert np.isclose(
        normal["total"], sum(value for key, value in normal.items() if key != "total")
    )


def test_planner_is_deterministic_explainable_and_uses_no_truth():
    rows = np.stack([track_row(3, 0.8), np.zeros(48), np.zeros(48), np.zeros(48)])
    runtime = FakeRuntime(rows=rows, mask=np.array([1, 0, 0, 0]))
    planner = EmitterAwareModelPredictivePlanner(20, runtime=runtime)
    first = planner.select_band()
    first_trace = planner.last_decision_trace
    second = planner.select_band()
    assert first == second
    assert 0 <= first < 20
    assert first_trace["selected_band"] == first
    assert len(first_trace["planned_sequence"]) == 3
    assert "top_contributing_emitters" in first_trace
    assert all("ground_truth" not in key for key in first_trace)


def test_alternative_plan_exploration_is_seeded_and_reproducible():
    rows = np.stack([track_row(3, 0.8), np.zeros(48), np.zeros(48), np.zeros(48)])
    config = EmitterAwarePlannerConfig(
        exploration_probability=1.0,
        exploration_top_k=3,
        exploration_seed=7,
    )
    planners = [
        EmitterAwareModelPredictivePlanner(
            20,
            runtime=FakeRuntime(rows=rows, mask=np.array([1, 0, 0, 0])),
            config=config,
        )
        for _ in range(2)
    ]
    actions = [planner.select_band() for planner in planners]
    assert actions[0] == actions[1]
    assert planners[0].last_plan == planners[1].last_plan
    trace = planners[0].last_decision_trace
    assert trace["selection_mode"] == "exploratory_alternative"
    assert actions[0] != trace["greedy_planned_sequence"][0]
    assert len(trace["planned_sequence"]) == 3


def test_alternative_plan_exploration_probability_is_validated():
    for probability in (-0.01, 1.01):
        try:
            EmitterAwarePlannerConfig(exploration_probability=probability)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid exploration probability was accepted")
