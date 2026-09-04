"""Complete Official Benchmark Across All 11 Models.

Runs a live, paired evaluation across all 11 candidate paradigms across
all 6 operational Electronic Warfare scenarios.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.world_model_ucb import WorldModelUCBScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler
from scheduler.baselines import (
    FixedScheduler,
    RandomScheduler,
    ThompsonSamplingScheduler,
    UCB1Scheduler,
)
from scheduler.paradigms import (
    NMFScheduler,
    RobustPCAPSRScheduler,
    WhittleIndexRMABScheduler,
    DoubleDQNScheduler,
    Exp3BanditScheduler,
)
from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH

MODELS = {
    "World-Model UCB (Champion)": lambda n, s: WorldModelUCBScheduler(n, model_path=DEFAULT_MODEL_PATH),
    "Observable Discounted UCB (Blind)": lambda n, s: ObservableDiscountedUCBScheduler(n, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True),
    "Standard UCB1 (Simulator Cheat)": lambda n, s: UCB1Scheduler(n),
    "Double DQN (Value RL)": lambda n, s: DoubleDQNScheduler(n, seed=s),
    "Whittle Index RMAB": lambda n, s: WhittleIndexRMABScheduler(n, seed=s),
    "Adversarial Exp3 Bandit": lambda n, s: Exp3BanditScheduler(n, seed=s),
    "Bayesian Thompson Sampling": lambda n, s: ThompsonSamplingScheduler(n, seed=s),
    "Robust PCA + PSR": lambda n, s: RobustPCAPSRScheduler(n, seed=s),
    "NMF Matrix Factorization": lambda n, s: NMFScheduler(n, seed=s),
    "Fixed Sequential Sweep (Open Loop)": lambda n, s: FixedScheduler(n),
    "Random Scan": lambda n, s: RandomScheduler(n, seed=s),
}

SCENARIOS = ["operational", "hopping", "mixed", "stationary", "harsh", "crowded"]
SEEDS = [101, 102, 103]
STEPS = 150

print("=" * 105)
print("  COMPLETE LIVE BENCHMARK ACROSS ALL 11 ALGORITHMS & 6 EW SCENARIOS")
print(f"  Scenarios: {SCENARIOS}")
print(f"  Seeds: {SEEDS} | Steps per episode: {STEPS}")
print(f"  Total evaluations: {len(MODELS) * len(SCENARIOS) * len(SEEDS) * STEPS:,} decisions")
print("=" * 105)

results = {}

for name, builder in MODELS.items():
    results[name] = {
        "reward": [],
        "hits": [],
        "false_alarms": [],
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
            ep_fa = 0
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
                
                if info.get("true_signal_present", False):
                    if obs.get("detected", 0) > 0.5:
                        ep_hits += 1
                else:
                    if obs.get("detected", 0) > 0.5:
                        ep_fa += 1
                        
                if done or truncated:
                    break
                    
            results[name]["reward"].append(ep_reward)
            results[name]["hits"].append(ep_hits)
            results[name]["false_alarms"].append(ep_fa)
            results[name]["switches"].append(ep_switches)
            results[name]["latencies"].extend(ep_latencies)
            results[name]["scenario_rewards"][sc].append(ep_reward)

print("\n" + "=" * 105)
print(f"{'Rank':<4} | {'Model / Algorithm':<36} | {'Mean Reward':>11} | {'Hit Rate':>9} | {'Switches':>9} | {'Latency':>10} | {'Grade':>6}")
print("=" * 105)

summary_list = []
total_decisions = len(SCENARIOS) * len(SEEDS) * STEPS

for name, data in results.items():
    mean_r = float(np.mean(data["reward"]))
    tot_hits = sum(data["hits"])
    hit_rate = (tot_hits / total_decisions) * 100.0
    mean_switches = float(np.mean(data["switches"]))
    mean_lat = float(np.mean(data["latencies"]))
    summary_list.append((name, mean_r, hit_rate, mean_switches, mean_lat, data["scenario_rewards"]))

summary_list.sort(key=lambda x: x[1], reverse=True)

rank = 1
for name, mean_r, hit_rate, mean_switches, mean_lat, sc_rews in summary_list:
    grade = "A+" if rank <= 2 else ("A" if rank <= 4 else ("B" if rank <= 7 else ("C" if rank <= 9 else "D")))
    print(f"{rank:02d}.  | {name:<36} | {mean_r:>+10.2f}  | {hit_rate:>8.1f}% | {mean_switches:>9.1f} | {mean_lat:>8.3f} ms | {grade:>6}")
    rank += 1

print("=" * 105)

print("\n--- PER-SCENARIO REWARD BREAKDOWN ---")
header = f"{'Model':<36}" + "".join([f" | {sc[:7]:>7}" for sc in SCENARIOS])
print(header)
print("-" * len(header))
for name, mean_r, hit_rate, mean_switches, mean_lat, sc_rews in summary_list:
    row = f"{name[:36]:<36}"
    for sc in SCENARIOS:
        avg_sc_r = float(np.mean(sc_rews[sc]))
        row += f" | {avg_sc_r:>+7.1f}"
    print(row)
print("-" * len(header))
