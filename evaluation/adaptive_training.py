"""Offline data collection/training for the optional adaptive models.

The simulator reward may label expert outcomes here. Runtime scheduler inputs
remain restricted to observable Track 2 state.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from scheduler.adaptive_moe import (
    LearnedRegimeRouter,
    ROUTER_FEATURE_NAMES,
    RouterDataset,
    extract_complexity_features,
)
from scheduler.learned_value import PathValueDataset, SmallPathValueModel


def _observable_update(scheduler, observation: dict, reward: float, timestamp: int):
    """Update exactly one runtime without passing simulator info/truth."""
    scheduler.update(int(observation["selected_band"]), reward, observation)
    if hasattr(scheduler, "timestamp"):
        scheduler.timestamp = float(timestamp + 1)


def collect_router_dataset(
    env_fn: Callable,
    expert_factories: Mapping[str, Callable[[int], object]],
    seeds: Sequence[int],
    warmup_steps: Sequence[int] = (10, 20, 40),
    warmup_patterns: Sequence[str] = ("sequential", "reverse", "random"),
    latency_penalty_per_ms: float = 0.003,
    rollout_horizon: int = 40,
) -> RouterDataset:
    """Run every expert from identical seeded, fixed-warmup observable states.

    A common scripted warmup makes the Track 2 state identical before each
    expert takes control. Fixed-horizon return minus a latency charge labels
    the best expert, so expensive planning must provide a meaningful gain.
    """
    expected = ("fast", "beam", "tree", "safe")
    if tuple(expert_factories) != expected:
        raise ValueError(f"expert_factories must be ordered as {expected}")
    allowed_patterns = {"sequential", "reverse", "random"}
    unknown_patterns = set(warmup_patterns).difference(allowed_patterns)
    if unknown_patterns:
        raise ValueError(f"Unknown warmup patterns: {sorted(unknown_patterns)}")
    if rollout_horizon < 1:
        raise ValueError("Router rollout horizon must be at least one")
    dataset = RouterDataset(expected, latency_penalty_per_ms=latency_penalty_per_ms)
    for seed in seeds:
        for warmup in warmup_steps:
            for pattern in warmup_patterns:
                returns: dict[str, float] = {}
                latency_ms: dict[str, float] = {}
                common_features = None
                for expert_name, factory in expert_factories.items():
                    env = env_fn(seed=seed)
                    env.reset(seed=seed)
                    scheduler = factory(env.unwrapped.num_bands)
                    warmup_rng = np.random.default_rng(seed + 104729)
                    terminated = truncated = False
                    timestamp = 0
                    while timestamp < int(warmup) and not (terminated or truncated):
                        if pattern == "sequential":
                            action = timestamp % env.unwrapped.num_bands
                        elif pattern == "reverse":
                            action = (-timestamp - 1) % env.unwrapped.num_bands
                        else:
                            action = int(warmup_rng.integers(env.unwrapped.num_bands))
                        observation, reward, terminated, truncated, _ = env.step(action)
                        _observable_update(scheduler, observation, reward, timestamp)
                        timestamp += 1
                    if not hasattr(scheduler, "runtime"):
                        raise TypeError("Each routing expert must expose its Track 2 runtime")
                    features = extract_complexity_features(
                        scheduler.runtime.get_global_belief()
                    )
                    if common_features is None:
                        common_features = features
                    elif not np.allclose(common_features.vector, features.vector, atol=1e-5):
                        raise RuntimeError(
                            "Experts did not reach the same observable warmup state"
                        )
                    total_return = 0.0
                    planning_seconds = 0.0
                    planning_steps = 0
                    while not (terminated or truncated) and planning_steps < rollout_horizon:
                        started = perf_counter()
                        action = int(scheduler.select_band())
                        planning_seconds += perf_counter() - started
                        planning_steps += 1
                        observation, reward, terminated, truncated, _ = env.step(action)
                        _observable_update(scheduler, observation, reward, timestamp)
                        total_return += float(reward)
                        timestamp += 1
                    returns[expert_name] = total_return
                    latency_ms[expert_name] = (
                        1000.0 * planning_seconds / max(1, planning_steps)
                    )
                    env.close()
                dataset.add(common_features, returns, latency_ms)
    return dataset


def save_router_dataset(dataset: RouterDataset, path: str | Path):
    features, labels, returns = dataset.arrays()
    np.savez(
        Path(path),
        features=features,
        labels=labels,
        returns=returns,
        latency_ms=dataset.latencies(),
        latency_penalty_per_ms=np.asarray(dataset.latency_penalty_per_ms),
        expert_names=np.asarray(dataset.expert_names),
        feature_names=np.asarray(ROUTER_FEATURE_NAMES),
    )


def train_router(dataset: RouterDataset, output_path: str | Path) -> LearnedRegimeRouter:
    features, labels, _ = dataset.arrays()
    router = LearnedRegimeRouter(dataset.expert_names).fit(features, labels)
    router.save(output_path)
    return router


def train_path_value_model(
    dataset: PathValueDataset, output_path: str | Path
) -> SmallPathValueModel:
    features, targets = dataset.arrays()
    model = SmallPathValueModel().fit(features, targets)
    model.save(output_path)
    return model


def collect_path_value_dataset(
    env_fn: Callable,
    planner_factory: Callable[[int], object],
    seeds: Sequence[int],
    discount: float = 0.95,
) -> PathValueDataset:
    """Label selected observable path features with realized discounted return."""
    dataset = PathValueDataset()
    for seed in seeds:
        env = env_fn(seed=seed)
        env.reset(seed=seed)
        planner = planner_factory(env.unwrapped.num_bands)
        feature_rows = []
        rewards = []
        terminated = truncated = False
        timestamp = 0
        while not (terminated or truncated):
            action = int(planner.select_band())
            if getattr(planner, "last_value_features", None) is None:
                raise TypeError(
                    "planner_factory must return a planner exposing last_value_features"
                )
            feature_rows.append(planner.last_value_features.copy())
            observation, reward, terminated, truncated, _ = env.step(action)
            planner.update(action, reward, observation)
            rewards.append(float(reward))
            timestamp += 1
        for row, target in zip(
            feature_rows, discounted_returns(rewards, discount), strict=True
        ):
            dataset.add(row, float(target))
        env.close()
    return dataset


def discounted_returns(rewards: Sequence[float], discount: float = 0.95) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        returns[index] = running
    return returns


def _main():
    import argparse

    from scheduler.adaptive_moe import GreedyBeliefScheduler, SafeExplorationScheduler
    from scheduler.belief_tree import ObservationDependentBeliefTreePlanner
    from scheduler.emitter_aware_predictive import (
        EmitterAwareModelPredictivePlanner,
        EmitterAwarePlannerConfig,
    )
    from scheduler.learned_value import LearnedValueEmitterPlanner
    from scheduler.track2_runtime import DEFAULT_MODEL_PATH, Track2Runtime
    from simulator.environment import DEFAULT_IQ_DATASET_PATH, SmartScanEnv
    from simulator.scenarios import scenario_names

    parser = argparse.ArgumentParser(
        description="Collect and train the adaptive router and path-value models"
    )
    parser.add_argument("--mode", choices=("router", "value", "all"), default="all")
    parser.add_argument(
        "--track2-policy",
        help="Deprecated compatibility option; PPO is no longer a default router expert",
    )
    parser.add_argument("--track2-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_IQ_DATASET_PATH))
    parser.add_argument(
        "--scenario", choices=scenario_names(), default="legacy"
    )
    parser.add_argument("--output-dir", default="models/adaptive")
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    parser.add_argument("--warmup-steps", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument(
        "--warmup-patterns",
        nargs="+",
        choices=("sequential", "reverse", "random"),
        default=["sequential", "reverse", "random"],
    )
    parser.add_argument("--latency-penalty-per-ms", type=float, default=0.003)
    parser.add_argument("--router-rollout-horizon", type=int, default=40)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    planner_config = EmitterAwarePlannerConfig(
        track_recency_weight=0.0,
        identity_weight=0.0,
        exploration_probability=0.0,
    )

    def env_fn(seed):
        return SmartScanEnv(
            episode_length=args.episode_length,
            seed=seed,
            iq_dataset_path=args.dataset,
            scenario=args.scenario,
        )

    def runtime():
        return Track2Runtime(args.track2_model, max_scan_age=args.episode_length)

    if args.mode in ("router", "all"):
        factories = {
            "fast": lambda n: GreedyBeliefScheduler(n, runtime()),
            "beam": lambda n: EmitterAwareModelPredictivePlanner(
                n,
                runtime=runtime(),
                config=planner_config,
                max_scan_age=args.episode_length,
            ),
            "tree": lambda n: ObservationDependentBeliefTreePlanner(
                n,
                runtime=runtime(),
                config=planner_config,
                max_scan_age=args.episode_length,
            ),
            "safe": lambda n: SafeExplorationScheduler(n, runtime()),
        }
        router_dataset = collect_router_dataset(
            env_fn,
            factories,
            args.seeds,
            args.warmup_steps,
            args.warmup_patterns,
            args.latency_penalty_per_ms,
            args.router_rollout_horizon,
        )
        save_router_dataset(router_dataset, output_dir / "router_dataset.npz")
        train_router(router_dataset, output_dir / "router_gate.npz")

    if args.mode in ("value", "all"):
        value_dataset = collect_path_value_dataset(
            env_fn,
            lambda n: LearnedValueEmitterPlanner(
                n,
                runtime=runtime(),
                config=planner_config,
                max_scan_age=args.episode_length,
            ),
            args.seeds,
        )
        features, targets = value_dataset.arrays()
        np.savez(
            output_dir / "path_value_dataset.npz",
            features=features,
            targets=targets,
        )
        train_path_value_model(value_dataset, output_dir / "path_value_model.npz")


if __name__ == "__main__":
    _main()
