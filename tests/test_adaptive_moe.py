import numpy as np

from scheduler.adaptive_moe import (
    AdaptiveExplorationConfig,
    AdaptiveExplorationGate,
    AdaptiveMixtureOfExpertsScheduler,
    LearnedRegimeRouter,
    ObservableDiscountedUCBScheduler,
    ObservableRegimeMonitor,
    Regime,
    RouterDataset,
    RuleBasedRegimeRouter,
    extract_complexity_features,
)
from scheduler.emitter_aware_predictive import CONFIRMED


def global_state(belief=None, uncertainty=0.1, candidate=None):
    rows = np.zeros((4, 48), dtype=np.float32)
    rows[0, 3] = 0.9
    rows[0, 41:47] = [uncertainty, 0.05, 0.05, 0.2, 0.9, 0.9]
    rows[0, CONFIRMED] = 1.0
    mask = np.array([1, 0, 0, 0], dtype=np.float32)
    return {
        "track_features": rows,
        "track_mask": mask,
        "band_belief": np.asarray(
            belief if belief is not None else [0.9, 0.1] + [0.0] * 18,
            dtype=np.float32,
        ),
        "scan_age": np.full(20, 0.1, dtype=np.float32),
        "band_uncertainty": np.full(20, uncertainty, dtype=np.float32),
        "prediction_error": np.zeros(20, dtype=np.float32),
        "recent_hit": np.zeros(20, dtype=np.float32),
        "recent_miss": np.zeros(20, dtype=np.float32),
        "candidate_summary": candidate or {},
        "track_ids": ["runtime-track"],
    }


def test_rule_router_separates_easy_complex_and_ood():
    router = RuleBasedRegimeRouter()
    easy = extract_complexity_features(global_state())
    assert router.route(easy) is Regime.EASY

    complex_state = global_state(
        belief=[0.48, 0.46, 0.44, 0.42, 0.40] + [0.05] * 15,
        uncertainty=0.9,
    )
    assert router.route(extract_complexity_features(complex_state)) is Regime.COMPLEX

    dynamic_state = global_state(uncertainty=0.7)
    dynamic_state["prediction_error"][3] = 0.9
    assert router.route(extract_complexity_features(dynamic_state)) is Regime.DYNAMIC

    ood_state = global_state(
        candidate={
            "count": 2.0,
            "max_novelty": 0.98,
            "mean_quality": 0.05,
            "mean_uncertainty": 0.95,
        }
    )
    assert router.route(extract_complexity_features(ood_state)) is Regime.UNKNOWN

    empty_state = global_state(belief=[0.0] * 20)
    empty_state["track_mask"][:] = 0.0
    assert router.route(extract_complexity_features(empty_state)) is Regime.DYNAMIC


def test_adaptive_exploration_prefers_useful_unknown_band():
    features = extract_complexity_features(global_state())
    features.exploration_priority[:] = 0.0
    features.exploration_priority[12] = 1.0
    gate = AdaptiveExplorationGate(
        AdaptiveExplorationConfig(
            easy_epsilon=1.0,
            intermediate_epsilon=1.0,
            complex_epsilon=1.0,
            unknown_epsilon=1.0,
            seed=7,
        )
    )
    assert gate.choose(3, Regime.EASY, features) == 12
    assert gate.last_trace["explored"] is True


def test_learned_router_trains_from_best_expert_outcomes():
    dataset = RouterDataset()
    for label in range(4):
        for offset in (0.0, 0.02, -0.02):
            row = np.zeros(14, dtype=np.float32)
            row[label] = 1.0 + offset
            returns = {name: 0.0 for name in dataset.expert_names}
            returns[dataset.expert_names[label]] = 1.0
            dataset.add(row, returns)
    features, labels, _ = dataset.arrays()
    router = LearnedRegimeRouter().fit(features, labels)
    assert [router.predict_expert(features[i]) for i in (0, 3, 6, 9)] == [
        "fast",
        "beam",
        "tree",
        "safe",
    ]


def test_router_dataset_uses_latency_when_rewards_are_close():
    dataset = RouterDataset(latency_penalty_per_ms=0.02)
    returns = {"fast": 10.0, "beam": 11.0, "tree": 12.0, "safe": 5.0}
    latency = {"fast": 1.0, "beam": 30.0, "tree": 300.0, "safe": 1.0}
    dataset.add(np.zeros(14, dtype=np.float32), returns, latency)
    _, labels, _ = dataset.arrays()
    assert dataset.expert_names[int(labels[0])] == "beam"


class FakeRuntime:
    def __init__(self):
        self.state = global_state()
        self.updates = []

    def get_global_belief(self):
        return self.state

    def get_band_belief(self):
        return self.state["band_belief"].copy()

    def get_scan_age(self, normalized=True):
        return self.state["scan_age"].copy()

    def get_band_uncertainty(self):
        return self.state["band_uncertainty"].copy()

    def get_rl_state(self):
        return np.concatenate(
            [
                self.state["band_belief"],
                self.state["scan_age"],
                self.state["band_uncertainty"],
            ]
        )

    def update(self, obs, timestamp):
        self.updates.append((obs, timestamp))


def test_mixture_routes_stable_state_to_mpp_without_truth():
    runtime = FakeRuntime()
    scheduler = AdaptiveMixtureOfExpertsScheduler(
        20,
        runtime=runtime,
        exploration=AdaptiveExplorationConfig(
            easy_epsilon=0.0,
            intermediate_epsilon=0.0,
            complex_epsilon=0.0,
            unknown_epsilon=0.0,
        ),
    )
    action = scheduler.select_band()
    assert action == 0
    assert scheduler.last_decision_trace["regime"] == "easy"
    assert scheduler.last_decision_trace["expert"] == "mpp"
    assert "ground_truth" not in repr(scheduler.last_decision_trace)


def test_mixture_uses_tree_for_complex_state_without_enabling_ppo():
    runtime = FakeRuntime()
    runtime.state = global_state(
        belief=[0.48, 0.46, 0.44, 0.42, 0.40] + [0.05] * 15,
        uncertainty=0.9,
    )
    scheduler = AdaptiveMixtureOfExpertsScheduler(
        20,
        rl_model=object(),
        runtime=runtime,
        exploration=AdaptiveExplorationConfig(
            easy_epsilon=0.0,
            intermediate_epsilon=0.0,
            complex_epsilon=0.0,
            unknown_epsilon=0.0,
        ),
    )
    scheduler.rl_expert = object()
    action = scheduler.select_band()
    assert 0 <= action < 20
    assert scheduler.last_decision_trace["regime"] == "complex"
    assert scheduler.last_decision_trace["expert"] == "tree"


def test_mixture_routes_unreliable_state_to_observable_ucb():
    runtime = FakeRuntime()
    runtime.state["prediction_error"][5] = 0.9
    scheduler = AdaptiveMixtureOfExpertsScheduler(
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
    assert scheduler.select_band() == 0
    assert scheduler.last_decision_trace["regime"] == "dynamic_unreliable"
    assert scheduler.last_decision_trace["expert"] == "ucb"
    assert scheduler.last_decision_trace["routing_source"] == "rule_dynamic_override"


def test_observable_ucb_does_not_learn_from_simulator_reward():
    runtime1 = FakeRuntime()
    runtime2 = FakeRuntime()
    ucb1 = ObservableDiscountedUCBScheduler(20, runtime1)
    ucb2 = ObservableDiscountedUCBScheduler(20, runtime2)
    observation = {"detected": 1, "quality": np.array([0.8], dtype=np.float32)}
    ucb1.update(3, reward=1.0, obs_dict=observation)
    ucb2.update(3, reward=-1.0, obs_dict=observation)
    assert np.allclose(ucb1.values, ucb2.values)


def test_standalone_observable_ucb_can_manage_track2_runtime():
    runtime = FakeRuntime()
    scheduler = ObservableDiscountedUCBScheduler(
        20, runtime, manage_runtime=True
    )
    observation = {"detected": 0, "quality": np.array([0.0], dtype=np.float32)}

    scheduler.update(2, reward=999.0, obs_dict=observation)

    assert runtime.updates == [(observation, 0.0)]


def test_history_monitor_separates_clean_stationary_and_noisy_observations():
    clean = ObservableRegimeMonitor(20)
    noisy = ObservableRegimeMonitor(20)
    for step in range(60):
        clean.update(
            (2, 5)[step % 2],
            {"detected": True, "quality": np.array([0.9], dtype=np.float32)},
        )
        noisy.update(
            step % 20,
            {"detected": True, "quality": np.array([0.2], dtype=np.float32)},
        )
    assert clean.snapshot()["history_regime"] == "clean_stationary"
    assert noisy.snapshot()["history_regime"] == "noisy"


def test_mixture_uses_tree_after_sustained_low_quality_detections():
    runtime = FakeRuntime()
    scheduler = AdaptiveMixtureOfExpertsScheduler(
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
    noisy_obs = {"detected": True, "quality": np.array([0.2], dtype=np.float32)}
    for step in range(40):
        scheduler.update(step % 20, 999.0, noisy_obs)
    scheduler.select_band()
    assert scheduler.last_decision_trace["expert"] == "tree"
    assert scheduler.last_decision_trace["routing_source"] == "history_noisy_tree_guard"


def test_calibration_locks_sustained_noise_to_tree():
    runtime = FakeRuntime()
    scheduler = AdaptiveMixtureOfExpertsScheduler(20, runtime=runtime)
    noisy_obs = {"detected": True, "quality": np.array([0.2], dtype=np.float32)}
    for step in range(80):
        scheduler.update(step % 20, 999.0, noisy_obs)
    calibration = scheduler.begin_scored_phase()
    scheduler.select_band()
    assert calibration["selected_override"] == "noisy_tree"
    assert scheduler.last_decision_trace["expert"] == "tree"
    assert (
        scheduler.last_decision_trace["routing_source"]
        == "calibrated_noisy_tree_guard"
    )


def test_mixture_uses_ucb_after_agile_hopping_history():
    runtime = FakeRuntime()
    scheduler = AdaptiveMixtureOfExpertsScheduler(20, runtime=runtime)
    clean_hit = {"detected": True, "quality": np.array([0.85], dtype=np.float32)}
    clean_miss = {"detected": False, "quality": np.array([0.0], dtype=np.float32)}
    for step in range(60):
        band = (step // 2) % 6
        obs = clean_hit if (step % 2 == 0) else clean_miss
        scheduler.update(band, 1.0 if obs["detected"] else 0.0, obs)
    scheduler.select_band()
    assert scheduler.last_decision_trace["expert"] == "ucb"
    assert (
        scheduler.last_decision_trace["routing_source"]
        == "history_hopping_ucb_guard"
    )


def test_mixture_uses_tree_after_clean_stationary_history():
    runtime = FakeRuntime()
    scheduler = AdaptiveMixtureOfExpertsScheduler(
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
    clean_obs = {"detected": True, "quality": np.array([0.9], dtype=np.float32)}
    for step in range(60):
        scheduler.update((2, 5)[step % 2], 999.0, clean_obs)
    scheduler.select_band()
    assert scheduler.last_decision_trace["expert"] == "tree"
    assert (
        scheduler.last_decision_trace["routing_source"]
        == "history_stationary_tree_guard"
    )
