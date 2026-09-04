"""Comparative Benchmark: 3 New Candidates vs. Dwell-Dual Champion.

Candidates Evaluated:
1. Contextual Bandit (LinUCB Policy Meta-Arbitrator)
2. Boosted-Tree Model (Compact GBDT Adaptive Dwell)
3. Missing-Data-Aware Recurrent Model (GRU-D Spectrum Model)
Baselines:
- Dwell-Dual Policy (Calibrated Champion)
- Robust PCA + PSR (Classical Factorization Benchmark)

Scenarios (6): stationary, hopping, changing, harsh, operational, crowded.
Fresh Seeds (10): 30001 - 30010 (strictly disjoint from any prior runs).
150 steps/episode. 5 models * 6 scenarios * 10 seeds = 300 episodes (45,000 decisions).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.contextual_bandit_scheduler import ContextualBanditScheduler
from scheduler.boosted_tree_dwell_scheduler import BoostedTreeDwellScheduler
from scheduler.grud_scheduler import GRUDSpectrumScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.track2_runtime import DEFAULT_MODEL_PATH

SEEDS = [30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010]
SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
STEPS = 150

MODELS = {
    "Dwell-Dual Policy (Champion)": lambda n, s: SmartScanProductionScheduler(
        n, nmf_scale=1.0, switch_penalty=0.08, dwell_inertia=1.30,
        explore_budget_prob=0.12, fading_grace_steps=1, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Candidate 1: LinUCB Bandit": lambda n, s: ContextualBanditScheduler(
        n, alpha=0.25, switch_penalty=0.08, dwell_inertia=1.30, fading_grace_steps=1,
        seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Candidate 2: Boosted-Tree Dwell": lambda n, s: BoostedTreeDwellScheduler(
        n, nmf_scale=1.0, switch_penalty=0.08, dwell_inertia=1.30, dwell_threshold=0.42,
        explore_budget_prob=0.12, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Candidate 3: GRU-D Recurrent": lambda n, s: GRUDSpectrumScheduler(
        n, hidden_dim=32, switch_penalty=0.08, dwell_inertia=1.25,
        uncertainty_bonus=0.15, seed=s
    ),
    "Robust PCA + PSR": lambda n, s: RobustPCAPSRScheduler(n, seed=s),
}


def run_benchmark():
    total_episodes = len(MODELS) * len(SCENARIOS) * len(SEEDS)
    total_decisions = total_episodes * STEPS
    print("=" * 120)
    print("COMPREHENSIVE BENCHMARK: 3 NEW ARCHITECTURAL CANDIDATES vs DWELL-DUAL CHAMPION")
    print(f"Scenarios ({len(SCENARIOS)}): {SCENARIOS}")
    print(f"Fresh Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Total: {len(MODELS)} models * {len(SCENARIOS)} scenarios * {len(SEEDS)} seeds = {total_episodes} episodes ({total_decisions:,} decisions)")
    print("=" * 120)

    results = {}

    for name, builder in MODELS.items():
        print(f"Running: {name:<35} ... ", end="", flush=True)
        t_mod = time.perf_counter()
        results[name] = {
            "reward": [],
            "hits": [],
            "switches": [],
            "switch_dists": [],
            "latencies": [],
            "scenario_rewards": {sc: [] for sc in SCENARIOS},
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

                results[name]["reward"].append(ep_reward)
                results[name]["hits"].append(ep_hits)
                results[name]["switches"].append(ep_switches)
                results[name]["switch_dists"].append(ep_switch_dist / max(1, ep_switches))
                results[name]["latencies"].append(float(np.mean(ep_latencies)))
                results[name]["scenario_rewards"][sc].append(ep_reward)

        elapsed = time.perf_counter() - t_mod
        mean_r = float(np.mean(results[name]["reward"]))
        mean_hit = float(np.mean(results[name]["hits"])) / STEPS * 100.0
        print(f"DONE in {elapsed:5.1f}s | Mean Reward: {mean_r:+6.2f} | Hit Rate: {mean_hit:4.1f}%")

    # Statistical Analysis
    champ_rewards = np.array(results["Dwell-Dual Policy (Champion)"]["reward"])
    summary = []

    for name, data in results.items():
        arr_r = np.array(data["reward"])
        mean_r = float(np.mean(arr_r))
        std_r = float(np.std(arr_r, ddof=1))
        ci_95 = float(1.96 * std_r / math.sqrt(len(arr_r)))
        hit_pct = float(np.mean(data["hits"])) / STEPS * 100.0
        mean_sw = float(np.mean(data["switches"]))
        mean_sw_dist = float(np.mean(data["switch_dists"]))
        mean_lat = float(np.mean(data["latencies"]))

        # Paired differences vs Champion
        diffs = arr_r - champ_rewards
        diff_mean = float(np.mean(diffs))
        diff_std = float(np.std(diffs, ddof=1))
        # Paired t-test p-value approximation
        t_stat = diff_mean / (diff_std / math.sqrt(len(diffs)) + 1e-12)
        # Normal approximation for large df (N=60)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

        summary.append({
            "name": name,
            "mean_reward": mean_r,
            "ci_95": ci_95,
            "std_reward": std_r,
            "hit_rate_pct": hit_pct,
            "switches": mean_sw,
            "switch_dist": mean_sw_dist,
            "latency_ms": mean_lat,
            "diff_vs_champ": diff_mean,
            "t_stat": t_stat,
            "p_val": p_val,
            "scenario_means": {
                sc: float(np.mean(data["scenario_rewards"][sc])) for sc in SCENARIOS
            }
        })

    # Sort by mean reward descending
    summary.sort(key=lambda x: x["mean_reward"], reverse=True)

    print("\n" + "=" * 135)
    print("FINAL CANDIDATE RANKING TABLE (Sorted by Mean Cumulative Reward)")
    print("=" * 135)
    header = f"{'Rank':<5} {'Model':<32} {'Mean Reward':<16} {'Hit %':<8} {'Switches':<10} {'Latency':<10} {'vs Champion':<14} {'p-value':<8}"
    print(header)
    print("-" * 135)

    for r, item in enumerate(summary, 1):
        diff_str = f"{item['diff_vs_champ']:+6.2f}" if item['name'] != "Dwell-Dual Policy (Champion)" else "0.00 (REF)"
        p_str = f"{item['p_val']:.4f}" if item['name'] != "Dwell-Dual Policy (Champion)" else "N/A"
        sig = "*" if item['p_val'] < 0.05 and item['name'] != "Dwell-Dual Policy (Champion)" else " "
        row = (
            f"{r:<5} {item['name']:<32} {item['mean_reward']:+6.2f} +/- {item['ci_95']:4.2f}  "
            f"{item['hit_rate_pct']:5.1f}%  "
            f"{item['switches']:6.1f}    "
            f"{item['latency_ms']:5.2f} ms   "
            f"{diff_str:<14} "
            f"{p_str:<6}{sig}"
        )
        print(row)
    print("=" * 135)

    # Print Scenario Breakdown
    print("\nSCENARIO-BY-SCENARIO REWARD BREAKDOWN:")
    print(f"{'Model':<32} " + " ".join([f"{sc[:9]:>10}" for sc in SCENARIOS]))
    print("-" * 100)
    for item in summary:
        sc_vals = " ".join([f"{item['scenario_means'][sc]:+10.2f}" for sc in SCENARIOS])
        print(f"{item['name']:<32} {sc_vals}")
    print("=" * 100)

    # Save to disk
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_three_new_candidates.json"

    export_data = {
        "seeds": SEEDS,
        "scenarios": SCENARIOS,
        "steps": STEPS,
        "summary": summary,
        "raw_results": {
            name: {
                "reward": [float(v) for v in data["reward"]],
                "hits": [int(v) for v in data["hits"]],
                "switches": [int(v) for v in data["switches"]],
                "latencies": [float(v) for v in data["latencies"]],
                "scenario_rewards": {
                    sc: [float(v) for v in data["scenario_rewards"][sc]] for sc in SCENARIOS
                }
            } for name, data in results.items()
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nSaved full benchmark results to {out_path}")


if __name__ == "__main__":
    run_benchmark()
