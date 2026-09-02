"""Comprehensive Head-to-Head Benchmark: 6 Advanced Paradigms vs Adaptive MoE."""

import sys
import os
import json
import time
import numpy as np
from simulator.environment import SmartScanEnv
from scheduler.adaptive_moe import AdaptiveMixtureOfExpertsScheduler
from scheduler.paradigms import (
    RobustPCAPSRScheduler,
    NMFScheduler,
    WhittleIndexRMABScheduler,
    ThompsonSamplingScheduler,
    Exp3BanditScheduler,
    AdaptiveReceiverSearchScheduler,
    DoubleDQNScheduler,
)


def evaluate_scheduler(name, scheduler_factory, scenario="mixed", episodes=3, episode_length=100):
    rewards = []
    hits = []
    planning_times = []

    for ep in range(episodes):
        seed = 42 + ep * 13
        env = SmartScanEnv(num_bands=20, episode_length=episode_length, seed=seed, scenario=scenario)
        scheduler = scheduler_factory(20, seed=seed)
        obs, info = env.reset()

        ep_reward = 0.0
        ep_hits = 0

        for step in range(episode_length):
            t0 = time.perf_counter()
            action = scheduler.select_band()
            t_plan = (time.perf_counter() - t0) * 1000.0
            planning_times.append(t_plan)

            obs, reward, done, truncated, info = env.step(action)
            scheduler.update(action, reward, obs)

            ep_reward += reward
            true_signal = bool(info.get("true_signal_present", False))
            detected = bool(obs.get("detected", 0) > 0.5)

            if (true_signal and detected) or reward > 0.5:
                ep_hits += 1

            if done or truncated:
                break

        rewards.append(ep_reward)
        hits.append(ep_hits)

    mean_rew = float(np.mean(rewards))
    std_rew = float(np.std(rewards))
    hit_rate = float(np.mean(hits)) / episode_length * 100.0
    avg_latency = float(np.mean(planning_times))

    return {
        "name": name,
        "mean_reward": round(mean_rew, 1),
        "std_reward": round(std_rew, 1),
        "hit_rate_pct": round(hit_rate, 1),
        "total_hits": int(np.mean(hits)),
        "latency_ms": round(avg_latency, 3),
    }


def main():
    models = [
        ("Adaptive MoE (Champion)", lambda n, seed: AdaptiveMixtureOfExpertsScheduler(n, max_scan_age=100)),
        ("Double DQN", lambda n, seed: DoubleDQNScheduler(n, seed=seed)),
        ("Robust PCA + PSR", lambda n, seed: RobustPCAPSRScheduler(n, seed=seed)),
        ("Non-Negative Matrix Factorization (NMF)", lambda n, seed: NMFScheduler(n, seed=seed)),
        ("Whittle Index RMAB", lambda n, seed: WhittleIndexRMABScheduler(n, seed=seed)),
        ("Adaptive Receiver Search (PRI)", lambda n, seed: AdaptiveReceiverSearchScheduler(n, seed=seed)),
        ("Bayesian Thompson Sampling", lambda n, seed: ThompsonSamplingScheduler(n, seed=seed)),
        ("Adversarial Exp3 Bandit", lambda n, seed: Exp3BanditScheduler(n, seed=seed)),
    ]

    print("\n" + "="*85)
    print("      SMART SCAN GRAND BENCHMARK: 6 ADVANCED PARADIGMS vs ADAPTIVE MoE")
    print("      Environment: 20-Channel Gymnasium | Scenario: MIXED | Episodes: 3 x 100 steps")
    print("="*85)

    results = []
    for name, factory in models:
        res = evaluate_scheduler(name, factory, scenario="mixed", episodes=3, episode_length=100)
        results.append(res)
        print(f"[*] Evaluated: {name:<36} -> Mean Reward: {res['mean_reward']:>6.1f} | Hit Rate: {res['hit_rate_pct']:>4.1f}% | Latency: {res['latency_ms']:>6.3f}ms")

    # Sort by mean reward descending
    results.sort(key=lambda x: x["mean_reward"], reverse=True)

    print("\n" + "="*85)
    print(f"{'Rank':<5} | {'Model / Paradigm':<38} | {'Mean Reward':<12} | {'Hit Rate (%)':<12} | {'Latency (ms)':<12}")
    print("-" * 85)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} | {r['name']:<38} | {r['mean_reward']:>+6.1f} ± {r['std_reward']:<3.1f} | {r['hit_rate_pct']:>6.1f}%     | {r['latency_ms']:>6.3f} ms")
    print("="*85 + "\n")

    os.makedirs("results", exist_ok=True)
    with open("results/grand_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
