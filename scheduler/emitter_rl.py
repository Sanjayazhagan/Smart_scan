"""PPO integrations for the frozen Track 2 persistent emitter model."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scheduler.baselines import BaseScheduler
from scheduler.emitter_attention import EmitterAttentionEncoder, TRACK2_ATTENTION_FEATURES
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import (
    DEFAULT_MAX_SCAN_AGE,
    DEFAULT_MODEL_PATH,
    Track2Runtime,
)

TRACK2_BELIEF_FEATURES = 20
TRACK2_EXPLORATION_FEATURES = 60


class Track2Belief20Wrapper(gym.Wrapper):
    """Original ablation: expose only Track 2's 20-band belief to PPO."""

    def __init__(
        self,
        env,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
    ):
        super().__init__(env)
        if self.env.unwrapped.num_bands != NUM_BANDS:
            raise ValueError(f"Track2Belief20Wrapper requires exactly {NUM_BANDS} bands")
        self.runtime = runtime or Track2Runtime(model_path)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(TRACK2_BELIEF_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_BANDS)
        self.timestamp = 0.0

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.timestamp = 0.0
        self.runtime.reset()
        return self.runtime.get_band_belief(), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.runtime.update(raw_obs, timestamp=self.timestamp)
        self.timestamp += 1.0
        return (
            self.runtime.get_band_belief(),
            reward,
            terminated,
            truncated,
            info,
        )


# Preserve the original public name and compatibility with the saved 20-D PPO.
Track2BeliefWrapper = Track2Belief20Wrapper


class Track2Belief60Wrapper(gym.Wrapper):
    """New state: belief[20] + normalized scan age[20] + uncertainty[20]."""

    def __init__(
        self,
        env,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        super().__init__(env)
        if self.env.unwrapped.num_bands != NUM_BANDS:
            raise ValueError(f"Track2Belief60Wrapper requires exactly {NUM_BANDS} bands")
        self.runtime = runtime or Track2Runtime(
            model_path, max_scan_age=max_scan_age
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(TRACK2_EXPLORATION_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_BANDS)
        self.timestamp = 0.0

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.timestamp = 0.0
        self.runtime.reset()
        return self.runtime.get_rl_state(), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.runtime.update(raw_obs, timestamp=self.timestamp)
        self.timestamp += 1.0
        return (
            self.runtime.get_rl_state(),
            reward,
            terminated,
            truncated,
            info,
        )


Track2ExplorationWrapper = Track2Belief60Wrapper


class Track2DiverseRLWrapper(gym.Wrapper):
    """60-D Track 2 state with diverse reward shaping and anti-camping protocols.

    Observation space:
        belief[20] + normalized scan_age[20] + band_uncertainty[20] (shape: 60)

    Reward shaping:
        Total = R_env + R_coverage + R_uncertainty - R_repeat
        - R_repeat: -0.25 * min(repeat_count, 3) if consecutive repeat
        - R_coverage: +0.05 * prior_scan_age[action]
        - R_uncertainty: +0.10 * max(0.0, prior_uncertainty[action] - current_uncertainty[action])
    """

    def __init__(
        self,
        env,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        repeat_penalty_weight: float = 0.25,
        window_penalty_weight: float = 0.30,
        window_size: int = 8,
        coverage_bonus_weight: float = 0.05,
        uncertainty_bonus_weight: float = 0.10,
    ):
        super().__init__(env)
        if self.env.unwrapped.num_bands != NUM_BANDS:
            raise ValueError(f"Track2DiverseRLWrapper requires exactly {NUM_BANDS} bands")
        self.runtime = runtime or Track2Runtime(
            model_path, max_scan_age=max_scan_age
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(TRACK2_EXPLORATION_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_BANDS)
        self.repeat_penalty_weight = float(repeat_penalty_weight)
        self.window_penalty_weight = float(window_penalty_weight)
        self.window_size = int(window_size)
        self.coverage_bonus_weight = float(coverage_bonus_weight)
        self.uncertainty_bonus_weight = float(uncertainty_bonus_weight)

        self.timestamp = 0.0
        self.last_action: int | None = None
        self.repeat_count = 0
        self.action_window: deque[int] = deque(maxlen=self.window_size)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.timestamp = 0.0
        self.last_action = None
        self.repeat_count = 0
        self.action_window.clear()
        self.runtime.reset()
        return self.runtime.get_rl_state(), info

    def step(self, action: int):
        action = int(action)

        # Pre-scan state for reward shaping
        prior_state = self.runtime.get_rl_state()
        prior_scan_age = prior_state[NUM_BANDS : 2 * NUM_BANDS]
        prior_uncertainty = prior_state[2 * NUM_BANDS : 3 * NUM_BANDS]

        # 1. Consecutive repeat penalty
        if self.last_action is not None and action == self.last_action:
            self.repeat_count += 1
            consecutive_penalty = self.repeat_penalty_weight * min(self.repeat_count, 3)
        else:
            self.repeat_count = 0
            consecutive_penalty = 0.0
        self.last_action = action

        # 2. Sliding-window frequency penalty (closes 2-3 band oscillation loops)
        if len(self.action_window) > 0:
            window_freq = self.action_window.count(action) / len(self.action_window)
            window_penalty = self.window_penalty_weight * window_freq
        else:
            window_penalty = 0.0
        self.action_window.append(action)

        total_penalty = consecutive_penalty + window_penalty

        # Execute in environment
        raw_obs, env_reward, terminated, truncated, info = self.env.step(action)
        self.runtime.update(raw_obs, timestamp=self.timestamp)
        self.timestamp += 1.0

        # Post-scan state
        new_state = self.runtime.get_rl_state()
        new_uncertainty = new_state[2 * NUM_BANDS : 3 * NUM_BANDS]

        # 3. Coverage bonus: reward visiting cold/stale bands
        coverage_bonus = self.coverage_bonus_weight * float(prior_scan_age[action])

        # 4. Uncertainty reduction bonus: reward info gain
        uncertainty_reduction = max(
            0.0, float(prior_uncertainty[action] - new_uncertainty[action])
        )
        uncertainty_bonus = self.uncertainty_bonus_weight * uncertainty_reduction

        # Total shaped reward
        shaped_reward = float(
            env_reward + coverage_bonus + uncertainty_bonus - total_penalty
        )

        if info is None:
            info = {}
        info["raw_env_reward"] = float(env_reward)
        info["repeat_penalty"] = float(consecutive_penalty)
        info["window_penalty"] = float(window_penalty)
        info["coverage_bonus"] = float(coverage_bonus)
        info["uncertainty_bonus"] = float(uncertainty_bonus)

        return (
            new_state,
            shaped_reward,
            terminated,
            truncated,
            info,
        )


class Track2AttentionWrapper(gym.Wrapper):
    """Emitter-aware masked-attention state for a separately trained PPO expert."""

    def __init__(
        self,
        env,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        super().__init__(env)
        if self.env.unwrapped.num_bands != NUM_BANDS:
            raise ValueError(f"Track2AttentionWrapper requires exactly {NUM_BANDS} bands")
        self.runtime = runtime or Track2Runtime(model_path, max_scan_age=max_scan_age)
        self.encoder = EmitterAttentionEncoder()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(TRACK2_ATTENTION_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_BANDS)
        self.timestamp = 0.0

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.timestamp = 0.0
        self.runtime.reset()
        return self.encoder.encode(self.runtime.get_global_belief()), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.runtime.update(raw_obs, timestamp=self.timestamp)
        self.timestamp += 1.0
        return (
            self.encoder.encode(self.runtime.get_global_belief()),
            reward,
            terminated,
            truncated,
            info,
        )


class BeliefRLScheduler(BaseScheduler):
    """Original adapter for the saved 20-feature Track 2 PPO policy."""

    def __init__(
        self,
        num_bands: int,
        model,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
    ):
        if num_bands != NUM_BANDS:
            raise ValueError(f"Track 2 scheduling requires exactly {NUM_BANDS} bands")
        super().__init__(num_bands)
        self.model = model
        self.runtime = runtime or Track2Runtime(model_path)
        self.timestamp = 0.0

    def select_band(self) -> int:
        action, _states = self.model.predict(
            self.runtime.get_band_belief(), deterministic=True
        )
        action = int(action)
        if not 0 <= action < NUM_BANDS:
            raise ValueError(f"Track 2 PPO returned invalid action {action}")
        return action

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=self.timestamp)
        self.timestamp += 1.0


class ExplorationBeliefRLScheduler(BaseScheduler):
    """Adapter for the new 60-feature Track 2 exploration PPO policy."""

    def __init__(
        self,
        num_bands: int,
        model,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        if num_bands != NUM_BANDS:
            raise ValueError(f"Track 2 scheduling requires exactly {NUM_BANDS} bands")
        super().__init__(num_bands)
        self.model = model
        self.runtime = runtime or Track2Runtime(
            model_path, max_scan_age=max_scan_age
        )
        self.timestamp = 0.0

    def select_band(self) -> int:
        action, _states = self.model.predict(
            self.runtime.get_rl_state(), deterministic=True
        )
        action = int(action)
        if not 0 <= action < NUM_BANDS:
            raise ValueError(f"Track 2 exploration PPO returned invalid action {action}")
        return action

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=self.timestamp)
        self.timestamp += 1.0


class AttentionRLScheduler(BaseScheduler):
    """Adapter for a PPO policy trained on the 288-D emitter-attention state."""

    def __init__(
        self,
        num_bands: int,
        model,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        if num_bands != NUM_BANDS:
            raise ValueError(f"Track 2 scheduling requires exactly {NUM_BANDS} bands")
        super().__init__(num_bands)
        self.model = model
        self.runtime = runtime or Track2Runtime(model_path, max_scan_age=max_scan_age)
        self.encoder = EmitterAttentionEncoder()
        self.timestamp = 0.0

    def select_band(self) -> int:
        state = self.encoder.encode(self.runtime.get_global_belief())
        action, _states = self.model.predict(state, deterministic=True)
        action = int(action)
        if not 0 <= action < NUM_BANDS:
            raise ValueError(f"Track 2 attention PPO returned invalid action {action}")
        return action

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=self.timestamp)
        self.timestamp += 1.0


def _train_ppo(
    wrapper_type,
    model_path,
    iq_dataset_path,
    total_timesteps,
    output_path,
    seed,
    device,
    episode_length,
):
    from stable_baselines3 import PPO

    from simulator.environment import DEFAULT_IQ_DATASET_PATH, SmartScanEnv

    dataset_path = iq_dataset_path or DEFAULT_IQ_DATASET_PATH
    base_env = SmartScanEnv(
        num_bands=NUM_BANDS,
        episode_length=episode_length,
        iq_dataset_path=dataset_path,
    )
    wrapper_kwargs = {"model_path": model_path}
    if wrapper_type in (Track2Belief60Wrapper, Track2AttentionWrapper):
        wrapper_kwargs["max_scan_age"] = episode_length
    env = wrapper_type(base_env, **wrapper_kwargs)
    model = PPO("MlpPolicy", env, verbose=1, seed=seed, device=device)
    model.learn(total_timesteps=int(total_timesteps))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    env.close()
    return model


def train_track2_ppo(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    iq_dataset_path: str | Path | None = None,
    total_timesteps: int = 100_000,
    output_path: str | Path = "models/track2_belief_ppo",
    seed: int = 42,
    device: str = "auto",
    episode_length: int = 200,
):
    """Train the preserved 20-feature ablation policy."""
    return _train_ppo(
        Track2Belief20Wrapper,
        model_path,
        iq_dataset_path,
        total_timesteps,
        output_path,
        seed,
        device,
        episode_length,
    )


def train_track2_exploration_ppo(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    iq_dataset_path: str | Path | None = None,
    total_timesteps: int = 100_000,
    output_path: str | Path = "models/track2_exploration_ppo",
    seed: int = 42,
    device: str = "auto",
    episode_length: int = 200,
):
    """Train only PPO on the new 60-feature state; Track 2 remains frozen."""
    return _train_ppo(
        Track2Belief60Wrapper,
        model_path,
        iq_dataset_path,
        total_timesteps,
        output_path,
        seed,
        device,
        episode_length,
    )


def train_track2_attention_ppo(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    iq_dataset_path: str | Path | None = None,
    total_timesteps: int = 100_000,
    output_path: str | Path = "models/track2_attention_ppo",
    seed: int = 42,
    device: str = "auto",
    episode_length: int = 200,
):
    """Train PPO on masked emitter attention while Track 2 remains frozen."""
    return _train_ppo(
        Track2AttentionWrapper,
        model_path,
        iq_dataset_path,
        total_timesteps,
        output_path,
        seed,
        device,
        episode_length,
    )


def train_track2_diverse_ppo(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    iq_dataset_path: str | Path | None = None,
    total_timesteps: int = 50_000,
    output_path: str | Path = "models/track2_diverse_ppo",
    seed: int = 42,
    device: str = "auto",
    episode_length: int = 200,
    ent_coef: float = 0.02,
    repeat_penalty_weight: float = 0.25,
    coverage_bonus_weight: float = 0.05,
    uncertainty_bonus_weight: float = 0.10,
):
    """Train PPO on 60-feature state with anti-camping, coverage shaping, and entropy injection."""
    from stable_baselines3 import PPO

    from simulator.environment import DEFAULT_IQ_DATASET_PATH, SmartScanEnv

    dataset_path = iq_dataset_path or DEFAULT_IQ_DATASET_PATH
    base_env = SmartScanEnv(
        num_bands=NUM_BANDS,
        episode_length=episode_length,
        iq_dataset_path=dataset_path,
    )
    env = Track2DiverseRLWrapper(
        base_env,
        model_path=model_path,
        max_scan_age=episode_length,
        repeat_penalty_weight=repeat_penalty_weight,
        coverage_bonus_weight=coverage_bonus_weight,
        uncertainty_bonus_weight=uncertainty_bonus_weight,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        ent_coef=ent_coef,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
    )
    model.learn(total_timesteps=int(total_timesteps))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    env.close()
    return model


class MultiScenarioSmartScanEnv(gym.Wrapper):
    """SmartScanEnv wrapper that samples a random scenario at each episode reset.

    This ensures PPO is exposed to a rich curriculum of stationary, hopping,
    changing, and harsh noise environments, preventing overfitting to any single pattern.
    """

    def __init__(
        self,
        base_env,
        scenarios: tuple[str, ...] = ("stationary", "hopping", "changing", "harsh"),
        probabilities: tuple[float, ...] = (0.35, 0.35, 0.20, 0.10),
    ):
        super().__init__(base_env)
        self.scenarios = scenarios
        self.probabilities = np.array(probabilities, dtype=np.float64) / sum(probabilities)
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        chosen_scenario = self.rng.choice(self.scenarios, p=self.probabilities)
        self.env.unwrapped.scenario = str(chosen_scenario)
        return self.env.reset(seed=seed, options=options)


def train_track2_generalist_ppo(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    iq_dataset_path: str | Path | None = None,
    total_timesteps: int = 150_000,
    output_path: str | Path = "models/track2_generalist_ppo",
    seed: int = 42,
    device: str = "auto",
    episode_length: int = 200,
    ent_coef: float = 0.02,
    repeat_penalty_weight: float = 0.25,
    window_penalty_weight: float = 0.30,
    window_size: int = 8,
    coverage_bonus_weight: float = 0.05,
    uncertainty_bonus_weight: float = 0.10,
):
    """Train generalist PPO on multi-scenario curriculum with window frequency regulation."""
    from stable_baselines3 import PPO

    from simulator.environment import DEFAULT_IQ_DATASET_PATH, SmartScanEnv

    dataset_path = iq_dataset_path or DEFAULT_IQ_DATASET_PATH
    base_env = SmartScanEnv(
        num_bands=NUM_BANDS,
        episode_length=episode_length,
        iq_dataset_path=dataset_path,
        scenario="stationary",
    )
    multi_env = MultiScenarioSmartScanEnv(base_env)
    env = Track2DiverseRLWrapper(
        multi_env,
        model_path=model_path,
        max_scan_age=episode_length,
        repeat_penalty_weight=repeat_penalty_weight,
        window_penalty_weight=window_penalty_weight,
        window_size=window_size,
        coverage_bonus_weight=coverage_bonus_weight,
        uncertainty_bonus_weight=uncertainty_bonus_weight,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        ent_coef=ent_coef,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
    )
    model.learn(total_timesteps=int(total_timesteps))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    env.close()
    return model


def _main():
    parser = argparse.ArgumentParser(description="Train a Track 2 PPO scheduler")
    parser.add_argument(
        "--state-size", type=int, choices=(20, 60, TRACK2_ATTENTION_FEATURES), default=60
    )
    parser.add_argument("--diverse", action="store_true", help="Train Track2DiverseRLWrapper with anti-camping and coverage rewards")
    parser.add_argument("--generalist", action="store_true", help="Train Track2GeneralistPPO on multi-scenario curriculum with window regulation")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy regularizer")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.generalist:
        output = args.output or "models/track2_generalist_ppo"
        train_track2_generalist_ppo(
            model_path=args.model,
            iq_dataset_path=args.dataset,
            total_timesteps=args.timesteps,
            output_path=output,
            seed=args.seed,
            device=args.device,
            episode_length=args.episode_length,
            ent_coef=args.ent_coef,
        )
        return

    if args.diverse:
        output = args.output or "models/track2_diverse_ppo"
        train_track2_diverse_ppo(
            model_path=args.model,
            iq_dataset_path=args.dataset,
            total_timesteps=args.timesteps,
            output_path=output,
            seed=args.seed,
            device=args.device,
            episode_length=args.episode_length,
            ent_coef=args.ent_coef,
        )
        return

    default_outputs = {
        20: "models/track2_belief_ppo",
        60: "models/track2_exploration_ppo",
        TRACK2_ATTENTION_FEATURES: "models/track2_attention_ppo",
    }
    output = args.output or default_outputs[args.state_size]
    trainers = {
        20: train_track2_ppo,
        60: train_track2_exploration_ppo,
        TRACK2_ATTENTION_FEATURES: train_track2_attention_ppo,
    }
    trainer = trainers[args.state_size]
    trainer(
        model_path=args.model,
        iq_dataset_path=args.dataset,
        total_timesteps=args.timesteps,
        output_path=output,
        seed=args.seed,
        device=args.device,
        episode_length=args.episode_length,
    )


if __name__ == "__main__":
    _main()
