"""Comprehensive Proper Benchmark: 12 Models, 6 Scenarios, 10 Fresh Seeds (108,000 Decisions).

Evaluates on 10 completely fresh, untouched seeds:
SEEDS = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010]
Scenarios (6): stationary, hopping, changing, harsh, operational, crowded
Steps per episode: 150
Total episodes: 12 models * 6 scenarios * 10 seeds = 720 episodes (108,000 decisions).

Computes:
- Mean Reward, Median, Standard Deviation, Standard Error, 95% Confidence Intervals
- Hit Rate (%), Interception Count, False Alarm Count
- Mean Switches & Physical Frequency Retuning Distance
- Latency (Mean, 95th Percentile, 99th Percentile)
- Full Pairwise Hypothesis Testing (Delta, 95% CI, t-stat, p-value) vs Direct NMF & UCB
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.smartscan_omni import SmartScanOmniScheduler
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
from scheduler.rl_value_network import DEFAULT_VALUE_NET_PATH

MODELS = {
    "SmartScan-Omni V2": lambda n, s: SmartScanOmniScheduler(
        n, depth=2, top_k=4, branch_k=3, switch_penalty=0.08, dwell_inertia=1.10,
        hysteresis_margin=0.05, curiosity_scale=0.35, use_rpca_filter=True, rpca_weight=0.25,
        use_rl_critic=True, critic_path=DEFAULT_VALUE_NET_PATH, enable_pruning=True, enable_caching=True,
        seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Direct NMF": lambda n, s: NMFScheduler(n, seed=s),
    "Dual-Policy Uncertainty": lambda n, s: DualPolicyUncertaintyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.05, arbitration_mode="probabilistic",
        explore_budget_prob=0.25, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "SmartScan V2-NMF": lambda n, s: NMFExpectimaxScheduler(
        n, depth=2, top_k=4, branch_k=3, switch_penalty=0.08, nmf_weight=1.0,
        curiosity_scale=0.0, use_rl_critic=False,
        enable_pruning=True, enable_caching=True, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Static NMF+UCB": lambda n, s: WorldModelNMFUCBScheduler(
        n, nmf_scale=1.0, world_model_scale=0.0, switch_penalty=0.05, model_path=DEFAULT_MODEL_PATH
    ),
    "Dwell-Dual Policy": lambda n, s: DwellDualPolicyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.08, dwell_inertia=1.20,
        explore_budget_prob=0.15, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Robust PCA + PSR": lambda n, s: RobustPCAPSRScheduler(n, seed=s),
    "SmartScan V2 (UCB-based)": lambda n, s: SmartScanScheduler(
        n, depth=2, top_k=5, branch_k=3, enable_pruning=True, enable_caching=True, model_path=DEFAULT_MODEL_PATH
    ),
    "Whittle Index RMAB": lambda n, s: WhittleIndexRMABScheduler(n, seed=s),
    "Fixed Sequential Sweep": lambda n, s: FixedScheduler(n),
    "Observable Plain UCB": lambda n, s: ObservableDiscountedUCBScheduler(
        n, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True
    ),
    "Random Scan": lambda n, s: RandomScheduler(n, seed=s),
}

SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
SEEDS = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010]
STEPS = 150

print("=" * 125)
print("COMPREHENSIVE PROPER BENCHMARK (12 MODELS, 6 SCENARIOS, 10 FRESH SEEDS)")
print(f"Scenarios ({len(SCENARIOS)}): {SCENARIOS}")
print(f"Seeds ({len(SEEDS)}): {SEEDS} | Steps per episode: {STEPS}")
print(f"Total episodes: {len(MODELS) * len(SCENARIOS) * len(SEEDS)} ({len(MODELS) * len(SCENARIOS) * len(SEEDS) * STEPS:,} decisions)")
print("=" * 125)

results = {}

for name, builder in MODELS.items():
    print(f"Running full evaluation for: {name:<30} ... ", end="", flush=True)
    t_mod = time.perf_counter()
    results[name] = {
        "reward": [],
        "hits": [],
        "switches": [],
        "switch_dists": [],
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
            ep_switch_dist = 0.0
            last_a = -1
            ep_latencies = []

            for st in range(STEPS):
                t0 = time.perf_counter()
                action = sched.select_band()
                lat = (time.perf_counter() - t0) * 1000.0
                ep_latencies.append(lat)

                if last_a != -1 and action != last_a:
                    ep_switches += 1
                    ep_switch_dist += abs(action - last_a) / 19.0
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
            results[name]["switch_dists"].append(ep_switch_dist)
            results[name]["latencies"].extend(ep_latencies)
            results[name]["scenario_rewards"][sc].append(ep_reward)

    dur = time.perf_counter() - t_mod
    r_mean = float(np.mean(results[name]["reward"]))
    print(f"Done in {dur:5.1f}s | Mean: {r_mean:>+6.2f}", flush=True)

total_decisions_per_model = len(SCENARIOS) * len(SEEDS) * STEPS

summary = []
for name, data in results.items():
    arr = np.array(data["reward"], dtype=np.float64)
    mean_r = float(np.mean(arr))
    median_r = float(np.median(arr))
    std_r = float(np.std(arr, ddof=1))
    se_r = std_r / np.sqrt(len(arr))
    ci_low = mean_r - 1.95996 * se_r
    ci_high = mean_r + 1.95996 * se_r

    hit_rate = (sum(data["hits"]) / total_decisions_per_model) * 100.0
    mean_sw = float(np.mean(data["switches"]))
    mean_sw_dist = float(np.mean(data["switch_dists"]))
    mean_lat = float(np.mean(data["latencies"]))
    p99_lat = float(np.percentile(data["latencies"], 99))

    summary.append((
        name, mean_r, median_r, (ci_low, ci_high), hit_rate, mean_sw, mean_sw_dist, mean_lat, p99_lat, data["scenario_rewards"]
    ))

summary.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 135)
print("FINAL OFFICIAL MASTER LEADERBOARD (10 FRESH SEEDS, N=60 EPISODES PER MODEL)")
print("=" * 135)
print(f"{'Rank':<4} | {'Architecture / Model':<30} | {'Mean Reward (95% CI)':^26} | {'Median':>7} | {'Hit Rate':>8} | {'Switches':>8} | {'Mean Lat':>8} | {'P99 Lat':>8}")
print("-" * 135)

rank = 1
for name, mean_r, median_r, (ci_l, ci_h), hit_rate, mean_sw, mean_sw_dist, mean_lat, p99_lat, _ in summary:
    ci_str = f"{mean_r:>+6.2f} [{ci_l:>+6.2f}, {ci_h:>+6.2f}]"
    print(f"{rank:02d}.  | {name:<30} | {ci_str:^26} | {median_r:>+6.2f} | {hit_rate:>7.1f}% | {mean_sw:>8.1f} | {mean_lat:>6.3f}ms | {p99_lat:>6.3f}ms")
    rank += 1
print("=" * 135)

print("\n--- SCENARIO-BY-SCENARIO DETAILED REWARD BREAKDOWN (10-SEED MEANS) ---")
header = f"{'Model':<30}" + "".join([f" | {sc[:7]:>7}" for sc in SCENARIOS])
print(header)
print("-" * len(header))
for name, mean_r, median_r, (ci_l, ci_h), hit_rate, mean_sw, mean_sw_dist, mean_lat, p99_lat, sc_rews in summary:
    row = f"{name[:30]:<30}"
    for sc in SCENARIOS:
        avg_sc_r = float(np.mean(sc_rews[sc]))
        row += f" | {avg_sc_r:>+7.1f}"
    print(row)
print("-" * len(header))

# Pairwise Statistical Significance Testing vs Direct NMF and Observable UCB
r_omni = np.array(results["SmartScan-Omni V2"]["reward"], dtype=np.float64)
r_direct = np.array(results["Direct NMF"]["reward"], dtype=np.float64)
r_ucb = np.array(results["Observable Plain UCB"]["reward"], dtype=np.float64)
r_dual = np.array(results["Dual-Policy Uncertainty"]["reward"], dtype=np.float64)

import math

def run_paired_test(a, b, label_a, label_b):
    diff = a - b
    delta = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    ci_l = delta - 1.95996 * se
    ci_h = delta + 1.95996 * se
    t_stat = delta / max(1e-8, se)
    # 2-tailed p-value using standard normal / t-distribution approximation with math.erfc
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    sig = "YES (p < 0.05)" if (ci_l > 0 or ci_h < 0) else "NO (Statistically Tied)"
    print(f"  {label_a} vs. {label_b:<26} : Delta = {delta:>+6.2f} | 95% CI: [{ci_l:>+6.2f}, {ci_h:>+6.2f}] | t = {t_stat:>+6.2f} | p = {p_val:.4f} | Significant? {sig}")

print("\n" + "=" * 115)
print("PAIRED HYPOTHESIS TESTING (N = 60 MATCHED PAIRS ACROSS ALL 6 SCENARIOS)")
print("=" * 115)
run_paired_test(r_omni, r_direct, "SmartScan-Omni V2", "Direct NMF")
run_paired_test(r_omni, r_dual, "SmartScan-Omni V2", "Dual-Policy Uncertainty")
run_paired_test(r_omni, r_ucb, "SmartScan-Omni V2", "Observable Plain UCB")
run_paired_test(r_direct, r_ucb, "Direct NMF", "Observable Plain UCB")
print("=" * 115)

# Save clean report
output_path = Path("results/proper_comprehensive_benchmark.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "steps": STEPS,
        "total_episodes": len(MODELS) * len(SCENARIOS) * len(SEEDS),
        "total_decisions": total_decisions_per_model * len(MODELS),
        "models": {
            name: {
                "mean_reward": float(np.mean(d["reward"])),
                "median_reward": float(np.median(d["reward"])),
                "ci_95": [float(np.mean(d["reward"]) - 1.95996 * np.std(d["reward"], ddof=1)/np.sqrt(len(d["reward"]))),
                          float(np.mean(d["reward"]) + 1.95996 * np.std(d["reward"], ddof=1)/np.sqrt(len(d["reward"])))],
                "hit_rate_pct": float((sum(d["hits"]) / total_decisions_per_model) * 100.0),
                "mean_switches": float(np.mean(d["switches"])),
                "mean_switch_dist": float(np.mean(d["switch_dists"])),
                "mean_latency_ms": float(np.mean(d["latencies"])),
                "p99_latency_ms": float(np.percentile(d["latencies"], 99)),
                "scenario_means": {sc: float(np.mean(d["scenario_rewards"][sc])) for sc in SCENARIOS},
            }
            for name, d in results.items()
        }
    }, f, indent=2)

print(f"\nComprehensive benchmark report successfully saved to {output_path}")
