"""Master Benchmark: Evaluating SmartScan V2-Hybrid (NMF Basis + UCB Curiosity Scout).

Evaluates across the untouched benchmark seeds [8001, 8002, 8003, 8004, 8005]
Across all 6 scenarios: stationary, hopping, changing, harsh, operational, crowded.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.nmf_expectimax import NMFExpectimaxScheduler
from scheduler.smartscan_v2 import SmartScanScheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.dual_policy_uncertainty_scheduler import DualPolicyUncertaintyScheduler
from scheduler.dwell_dual_policy_scheduler import DwellDualPolicyScheduler
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.paradigms.mathematical_scheduling import WhittleIndexRMABScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler
from scheduler.baselines import FixedScheduler, RandomScheduler
from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH

MODELS = {
    "SmartScan V2-Hybrid (NEW)": lambda n, s: NMFExpectimaxScheduler(
        n, depth=2, top_k=4, branch_k=3, switch_penalty=0.08, nmf_weight=1.0,
        curiosity_scale=0.50, enable_pruning=True, enable_caching=True, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Direct NMF (Reigning Champion)": lambda n, s: NMFScheduler(n, seed=s),
    "Dual-Policy Uncertainty": lambda n, s: DualPolicyUncertaintyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.05, arbitration_mode="probabilistic",
        explore_budget_prob=0.25, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Static NMF+UCB": lambda n, s: WorldModelNMFUCBScheduler(
        n, nmf_scale=1.0, world_model_scale=0.0, switch_penalty=0.05, model_path=DEFAULT_MODEL_PATH
    ),
    "Dwell-Dual Policy": lambda n, s: DwellDualPolicyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.08, dwell_inertia=1.20,
        explore_budget_prob=0.15, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "SmartScan V2 (UCB-based)": lambda n, s: SmartScanScheduler(
        n, depth=2, top_k=5, branch_k=3, enable_pruning=True, enable_caching=True, model_path=DEFAULT_MODEL_PATH
    ),
    "Robust PCA + PSR": lambda n, s: RobustPCAPSRScheduler(n, seed=s),
    "Whittle Index RMAB": lambda n, s: WhittleIndexRMABScheduler(n, seed=s),
    "Fixed Sequential Sweep": lambda n, s: FixedScheduler(n),
    "Observable Plain UCB": lambda n, s: ObservableDiscountedUCBScheduler(
        n, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True
    ),
}

SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
SEEDS = [8001, 8002, 8003, 8004, 8005]
STEPS = 150

print("=" * 115)
print("BENCHMARKING SMARTSCAN V2-HYBRID (NMF SPECTRAL BASIS + UCB CURIOSITY SCOUT + EXPECTIMAX)")
print(f"Scenarios ({len(SCENARIOS)}): {SCENARIOS}")
print(f"Seeds ({len(SEEDS)}): {SEEDS} | Steps per episode: {STEPS}")
print(f"Total episodes: {len(MODELS) * len(SCENARIOS) * len(SEEDS)} ({len(MODELS) * len(SCENARIOS) * len(SEEDS) * STEPS:,} decisions)")
print("=" * 115)

results = {}

for name, builder in MODELS.items():
    print(f"Evaluating {name}...")
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
            sched = builder(20, seed)
            obs, info = env.reset(seed=seed)

            ep_reward = 0.0
            ep_hits = 0
            ep_switches = 0
            last_a = -1
            ep_latencies = []

            for st in range(STEPS):
                t0 = time.perf_counter()
                action = sched.select_band()
                lat = (time.perf_counter() - t0) * 1000.0
                ep_latencies.append(lat)

                if last_a != -1 and action != last_a:
                    ep_switches += 1
                last_a = action

                obs, reward, done, truncated, info = env.step(action)
                sched.update(action, reward, obs)
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

total_decisions_per_model = len(SCENARIOS) * len(SEEDS) * STEPS

summary = []
for name, data in results.items():
    mean_r = float(np.mean(data["reward"]))
    median_r = float(np.median(data["reward"]))
    hit_rate = (sum(data["hits"]) / total_decisions_per_model) * 100.0
    mean_sw = float(np.mean(data["switches"]))
    mean_lat = float(np.mean(data["latencies"]))
    summary.append((name, mean_r, median_r, hit_rate, mean_sw, mean_lat, data["scenario_rewards"]))

summary.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 115)
print("FINAL OFFICIAL LEADERBOARD (SORTED BY MEAN REWARD)")
print("=" * 115)
print(f"{'Rank':<4} | {'Architecture / Model':<32} | {'Mean Reward':>11} | {'Median':>8} | {'Hit Rate':>8} | {'Switches':>8} | {'Latency':>9}")
print("-" * 115)

rank = 1
for name, mean_r, median_r, hit_rate, mean_sw, mean_lat, _ in summary:
    print(f"{rank:02d}.  | {name:<32} | {mean_r:>+10.2f}  | {median_r:>+7.2f} | {hit_rate:>7.1f}% | {mean_sw:>8.1f} | {mean_lat:>7.3f} ms")
    rank += 1
print("=" * 115)

print("\n--- SCENARIO-BY-SCENARIO DETAILED REWARD BREAKDOWN ---")
header = f"{'Model':<30}" + "".join([f" | {sc[:7]:>7}" for sc in SCENARIOS])
print(header)
print("-" * len(header))
for name, mean_r, median_r, hit_rate, mean_sw, mean_lat, sc_rews in summary:
    row = f"{name[:30]:<30}"
    for sc in SCENARIOS:
        avg_sc_r = float(np.mean(sc_rews[sc]))
        row += f" | {avg_sc_r:>+7.1f}"
    print(row)
print("-" * len(header))

# Statistical Comparison vs Direct NMF
r_hybrid = np.array(results["SmartScan V2-Hybrid (NEW)"]["reward"])
r_direct = np.array(results["Direct NMF (Reigning Champion)"]["reward"])
diff = r_hybrid - r_direct
m_diff = float(np.mean(diff))
se_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
ci = [m_diff - 1.95996 * se_diff, m_diff + 1.95996 * se_diff]

print("\n" + "=" * 90)
print("STATISTICAL COMPARISON: SMARTSCAN V2-HYBRID vs. DIRECT NMF")
print("=" * 90)
print(f"  SmartScan V2-Hybrid Mean   : {np.mean(r_hybrid):+.2f}")
print(f"  Direct NMF Champion Mean   : {np.mean(r_direct):+.2f}")
print(f"  Paired Difference (Delta)  : {m_diff:+.2f}")
print(f"  95% Confidence Interval    : [{ci[0]:+.2f}, {ci[1]:+.2f}]")
print(f"  Interval Excludes Zero?    : {'YES' if ci[0]>0 or ci[1]<0 else 'NO'}")
print("=" * 90)

output_path = Path("results/hybrid_expectimax_benchmark.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "steps": STEPS,
        "comparison_vs_direct_nmf": {"delta": m_diff, "ci_95": ci},
        "models": {
            name: {
                "mean_reward": float(np.mean(d["reward"])),
                "median_reward": float(np.median(d["reward"])),
                "hit_rate_pct": float((sum(d["hits"]) / total_decisions_per_model) * 100.0),
                "mean_switches": float(np.mean(d["switches"])),
                "mean_latency_ms": float(np.mean(d["latencies"])),
                "scenario_means": {sc: float(np.mean(d["scenario_rewards"][sc])) for sc in SCENARIOS},
            }
            for name, d in results.items()
        }
    }, f, indent=2)
print(f"\nReport saved to {output_path}")
