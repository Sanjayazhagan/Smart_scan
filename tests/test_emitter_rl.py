import numpy as np

from scheduler.emitter_rl import (
    AttentionRLScheduler,
    BeliefRLScheduler,
    ExplorationBeliefRLScheduler,
    MultiScenarioSmartScanEnv,
    Track2Belief20Wrapper,
    Track2Belief60Wrapper,
    Track2DiverseRLWrapper,
)
from scheduler.emitter_attention import TRACK2_ATTENTION_FEATURES
from simulator.environment import SmartScanEnv


class FakeTrack2Runtime:
    def __init__(self):
        self.reset()

    def reset(self):
        self.belief = np.zeros(20, dtype=np.float32)
        self.age = np.ones(20, dtype=np.float32)
        self.uncertainty = np.ones(20, dtype=np.float32)
        self.updates = 0

    def update(self, obs, timestamp):
        self.age = np.minimum(self.age + 0.1, 1.0)
        self.age[int(obs["selected_band"])] = 0.0
        self.updates += int(bool(obs["detected"]))
        if obs["detected"]:
            self.belief[int(obs["selected_band"])] = 0.9

    def get_band_belief(self):
        return self.belief.copy()

    def get_rl_state(self):
        return np.concatenate([self.belief, self.age, self.uncertainty]).astype(
            np.float32
        )

    def get_global_belief(self):
        rows = np.zeros((16, 48), dtype=np.float32)
        return {
            "track_features": rows,
            "track_mask": np.zeros(16, dtype=np.float32),
            "band_belief": self.belief.copy(),
            "scan_age": self.age.copy(),
            "band_uncertainty": self.uncertainty.copy(),
            "prediction_error": np.zeros(20, dtype=np.float32),
            "recent_hit": np.zeros(20, dtype=np.float32),
            "recent_miss": np.zeros(20, dtype=np.float32),
            "candidate_summary": {},
        }


def test_belief_rl_scheduler_uses_track2_runtime():
    class ArgmaxPolicy:
        def predict(self, state, deterministic=True):
            return np.argmax(state), None

    runtime = FakeTrack2Runtime()
    scheduler = BeliefRLScheduler(20, ArgmaxPolicy(), runtime=runtime)
    assert scheduler.select_band() == 0
    scheduler.update(
        band=19,
        reward=1.0,
        obs_dict={"selected_band": 19, "detected": 1},
    )
    assert scheduler.select_band() == 19
    assert 0 <= scheduler.select_band() < 20


def test_track2_belief_wrapper_exposes_only_belief():
    wrapped = Track2Belief20Wrapper(
        SmartScanEnv(num_bands=20, episode_length=2, false_alarm_prob=0.0),
        runtime=FakeTrack2Runtime(),
    )
    obs, info = wrapped.reset(seed=7)
    assert obs.shape == (20,)
    assert wrapped.observation_space.contains(obs)
    assert not isinstance(obs, dict)

    obs, _, _, _, info = wrapped.step(3)
    assert obs.shape == (20,)
    assert np.all(np.isfinite(obs))
    assert np.all((obs >= 0.0) & (obs <= 1.0))
    assert "ground_truth_active_emitters" not in obs


def test_track2_60_wrapper_schema_and_no_ground_truth_leak():
    runtime = FakeTrack2Runtime()
    wrapped = Track2Belief60Wrapper(
        SmartScanEnv(num_bands=20, episode_length=2, false_alarm_prob=0.0),
        runtime=runtime,
    )
    obs, _ = wrapped.reset(seed=7)
    assert obs.shape == (60,)
    assert wrapped.observation_space.contains(obs)
    assert np.array_equal(obs[:20], runtime.belief)
    assert np.array_equal(obs[20:40], runtime.age)
    assert np.array_equal(obs[40:], runtime.uncertainty)
    assert np.all(np.isfinite(obs))
    assert np.all((obs >= 0.0) & (obs <= 1.0))

    next_obs, _, _, _, _ = wrapped.step(3)
    assert next_obs[23] == 0.0
    assert next_obs.shape == (60,)
    assert not isinstance(next_obs, dict)


def test_exploration_scheduler_action_range():
    class Policy:
        def predict(self, state, deterministic=True):
            assert state.shape == (60,)
            return 19, None

    scheduler = ExplorationBeliefRLScheduler(
        20, Policy(), runtime=FakeTrack2Runtime()
    )
    assert scheduler.select_band() == 19


def test_attention_rl_scheduler_uses_fixed_width_masked_state():
    class Policy:
        def predict(self, state, deterministic=True):
            assert state.shape == (TRACK2_ATTENTION_FEATURES,)
            return 11, None

    scheduler = AttentionRLScheduler(20, Policy(), runtime=FakeTrack2Runtime())
    assert scheduler.select_band() == 11


def test_track2_diverse_wrapper_reward_shaping():
    runtime = FakeTrack2Runtime()
    env = SmartScanEnv(num_bands=20, episode_length=10, false_alarm_prob=0.0)
    wrapped = Track2DiverseRLWrapper(
        env,
        runtime=runtime,
        repeat_penalty_weight=0.25,
        coverage_bonus_weight=0.05,
        uncertainty_bonus_weight=0.10,
    )

    obs, _ = wrapped.reset(seed=42)
    assert obs.shape == (60,)
    assert wrapped.observation_space.contains(obs)

    # Step 1: Scan Band 5 for the first time (should get coverage bonus and 0 repeat penalty)
    obs1, r1, _, _, info1 = wrapped.step(5)
    assert info1["repeat_penalty"] == 0.0
    assert info1["coverage_bonus"] > 0.0
    assert wrapped.repeat_count == 0

    # Step 2: Repeat scan on Band 5 (should incur repeat penalty)
    obs2, r2, _, _, info2 = wrapped.step(5)
    assert info2["repeat_penalty"] == 0.25
    assert wrapped.repeat_count == 1

    # Step 3: Repeat scan on Band 5 again (should incur escalated repeat penalty)
    obs3, r3, _, _, info3 = wrapped.step(5)
    assert info3["repeat_penalty"] == 0.50
    assert wrapped.repeat_count == 2

    # Step 4: Switch to Band 12 (repeat penalty should reset to 0)
    obs4, r4, _, _, info4 = wrapped.step(12)
    assert info4["repeat_penalty"] == 0.0
    assert wrapped.repeat_count == 0


def test_track2_window_frequency_penalty():
    runtime = FakeTrack2Runtime()
    env = SmartScanEnv(num_bands=20, episode_length=20, false_alarm_prob=0.0)
    wrapped = Track2DiverseRLWrapper(
        env,
        runtime=runtime,
        repeat_penalty_weight=0.25,
        window_penalty_weight=0.30,
        window_size=4,
    )
    wrapped.reset(seed=123)

    # Alternate between 1 and 2: consecutive repeat is 0, but window frequency builds up
    wrapped.step(1)  # window: [1]
    wrapped.step(2)  # window: [1, 2]
    _, _, _, _, info3 = wrapped.step(1)  # window: [1, 2, 1] -> count(1) = 1 in prior window of 2
    assert info3["repeat_penalty"] == 0.0  # no consecutive repeat
    assert info3["window_penalty"] > 0.0   # window penalty triggered!


def test_multi_scenario_env_resets():
    base_env = SmartScanEnv(num_bands=20, episode_length=10)
    multi_env = MultiScenarioSmartScanEnv(base_env)

    observed_scenarios = set()
    for s in range(30):
        multi_env.reset(seed=s)
        observed_scenarios.add(multi_env.unwrapped.scenario_name)

    # Should sample multiple distinct scenarios
    assert len(observed_scenarios) >= 3


