"""Comprehensive Benchmark & Ranking of Alarm-Adaptive NMF+UCB Architecture.

Compares:
1. Alarm-Adaptive NMF+UCB (The Driver + Watchdog Architecture)
2. Static NMF+UCB (Neural-Off Baseline, No Alarm)
3. Direct NMF (Pure Non-Negative Matrix Factorization)
4. Whittle Index RMAB (Pure Restless Multi-Armed Bandit)
5. Neural-Guided NMF+UCB (Old Direct Action Guidance)
6. Observable Plain UCB (Baseline)
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.alarm_adaptive_nmf_ucb import AlarmAdaptiveNMFUCBScheduler
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.paradigms.mathematical_scheduling import WhittleIndexRMABScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler
from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH

MODELS = {
    "Alarm-Adaptive NMF+UCB (Driver+Watchdog)": lambda n, s: AlarmAdaptiveNMFUCBScheduler(n, nmf_scale=1.0, alarm_threshold=0.80, model_path=DEFAULT_MODEL_PATH),
    "Static NMF+UCB (No Alarm, Neural-Off)": lambda n, s: WorldModelNMFUCBScheduler(n, nmf_scale=1.0, world_model_scale=0.0, model_path=DEFAULT_MODEL_PATH),
    "Direct NMF": lambda n, s: NMFScheduler(n, seed=s),
    "Whittle Index RMAB": lambda n, s: WhittleIndexRMABScheduler(n, seed=s),
    "Neural-Guided NMF+UCB (Old Neural-On)": lambda n, s: WorldModelNMFUCBScheduler(n, nmf_scale=1.0, world_model_scale=0.35, model_path=DEFAULT_MODEL_PATH),
    "Observable Plain UCB": lambda n, s: ObservableDiscountedUCBScheduler(n, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True),
}

SCENARIOS = ["changing", "operational", "hopping", "stationary", "harsh", "crowded"]
SEEDS = [101, 102, 103, 104, 105]  # 5 paired seeds per scenario = 30 episodes per model
STEPS = 150

print("=" * 105)
print("  BENCHMARK & RANKING: ALARM-ADAPTIVE NMF+UCB (DRIVER + WATCHDOG) vs. COMPETING PARADIGMS")
print(f"  Scenarios: {SCENARIOS}")
print(f"  Seeds: {SEEDS} | Steps per episode: {STEPS}")
print(f"  Total evaluations: {len(MODELS) * len(SCENARIOS) * len(SEEDS) * STEPS:,} decisions")
print("=" * 105)

results = {}

for name, builder in MODELS.items():
    results[name] = {
        "reward": [],
        "hits": [],
        "switches": [],
        "latencies": [],
        "scenario_rewards": {sc: [] for sc in SCENARIOS}
    }

    for sc in SCENARIOS:
        for seed in SEEDS:
            env = SmartScanEnv(num_bands=20, episode_length=STEPS, seed=seed, scenario=sc)
            scheduler = builder(20, seed)
            
            obs, info = env.reset(seed=seed)
            ep_reward = 0.0
            ep_hits = 0
            ep_switches = 0
            last_a = -1
            ep_latencies = []

            for st in range(STEPS):
                t0 = time.perf_counter()
                action = scheduler.select_band()
                lat = (time.perf_counter() - t0) * 1000.0
                ep_latencies.append(lat)

                if last_a != -1 and action != last_a:
                    ep_switches += 1
                last_a = action

                obs, reward, done, truncated, info = env.step(action)
                scheduler.update(action, reward, obs)
                ep_reward += reward

                if info.get("true_signal_present", False) and obs.get("detected", 0) > 0.5:
                    ep_hits += 1

                if done or truncated:
                    break

            results[name]["reward"].append(ep_reward)
            results[name]["hits"].append(ep_hits)
            results[name]["switches"].append(ep_switches)
            results[name]["latencies"].extend(ep_latencies)
            results[name]["scenario_rewards"][sc].append(ep_reward)

total_decisions = len(SCENARIOS) * len(SEEDS) * STEPS

summary = []
for name, data in results.items():
    mean_r = float(np.mean(data["reward"]))
    hit_rate = (sum(data["hits"]) / total_decisions) * 100.0
    mean_sw = float(np.mean(data["switches"]))
    mean_lat = float(np.mean(data["latencies"]))
    summary.append((name, mean_r, hit_rate, mean_sw, mean_lat, data["scenario_rewards"]))

summary.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 105)
print(f"{'Rank':<4} | {'Architecture / Model':<42} | {'Mean Reward':>11} | {'Hit Rate':>9} | {'Switches':>9} | {'Latency':>9}")
print("=" * 105)

rank = 1
for name, mean_r, hit_rate, mean_sw, mean_lat, _ in summary:
    print(f"{rank:02d}.  | {name:<42} | {mean_r:>+10.2f}  | {hit_rate:>8.1f}% | {mean_sw:>9.1f} | {mean_lat:>7.3f} ms")
    rank += 1
print("=" * 105)

print("\n--- SCENARIO-BY-SCENARIO REWARD BREAKDOWN ---")
header = f"{'Model':<40}" + "".join([f" | {sc[:7]:>7}" for sc in SCENARIOS])
print(header)
print("-" * len(header))
for name, mean_r, hit_rate, mean_sw, mean_lat, sc_rews in summary:
    row = f"{name[:40]:<40}"
    for sc in SCENARIOS:
        avg_sc_r = float(np.mean(sc_rews[sc]))
        row += f" | {avg_sc_r:>+7.1f}"
    print(row)
print("-" * len(header))
