"""Master Benchmark and Ablation Study for SmartScan V2.

Evaluates:
Part 1: Primary Competitors & Baseline Comparisons (across 6 scenarios, 5 seeds):
 1. SmartScan V2 (Full: Depth-2 Expectimax + Pruning + Caching)
 2. Observable Discounted UCB (Standalone Baseline)
 3. UCB + Track2 Belief
 4. UCB + Track2 Full (Belief + Uncertainty + Investigation Priority)
 5. Direct NMF (Previous Leaderboard Winner)
 6. Whittle Index RMAB
 7. Robust PCA + PSR
 8. Standalone World-Model UCB
 9. Fixed Sweep
10. Random Scan

Part 2: Systematic Ablation Study (Evaluating Components A through K):
 A. Plain Discounted UCB
 B. UCB + Band Belief
 C. UCB + Band Belief + Uncertainty
 D. UCB + Investigation Priority
 E. UCB + Full Track2 Info (Ablation E)
 F. Top-K + Depth-1 Expectimax
 G. Top-K + Depth-2 Expectimax (Unpruned)
 H. Top-K + Depth-2 Expectimax + Probability Pruning + Branch-and-Bound (SmartScan V2)
 I. + Transposition Caching Enabled

Metrics collected:
- Mean Reward & Median Reward
- Hit Rate (%) & Interception Count
- Switch Count
- Control Latency (ms)
- Node Expansions & Pruning Ratio
- Cache Hit Rate
- Expectimax Override Rate & Override Reward Delta
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.smartscan_v2 import SmartScanScheduler
from scheduler.smart_discounted_ucb import SmartDiscountedUCB
from scheduler.belief_expectimax import BeliefExpectimaxPlanner, BeliefState
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.paradigms.mathematical_scheduling import WhittleIndexRMABScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.world_model_ucb import WorldModelUCBScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler
from scheduler.baselines import FixedScheduler, RandomScheduler
from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH

SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
SEEDS = [8001, 8002, 8003, 8004, 8005]
STEPS = 150

# -------------------------------------------------------------
# Part 1: Primary Competitors
# -------------------------------------------------------------
PRIMARY_MODELS = {
    "SmartScan V2 (Full)": lambda n, s: SmartScanScheduler(
        n, depth=2, top_k=5, branch_k=3, enable_pruning=True, enable_caching=True, model_path=DEFAULT_MODEL_PATH
    ),
    "Observable Discounted UCB": lambda n, s: ObservableDiscountedUCBScheduler(
        n, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True
    ),
    "UCB + Track2 Belief": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=1, w_prediction=0.35, w_uncertainty=0.0, w_investigate=0.0,
        enable_pruning=False, enable_caching=False, model_path=DEFAULT_MODEL_PATH
    ),
    "UCB + Full Track2 Info": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=1, w_prediction=0.35, w_uncertainty=0.15, w_investigate=0.20,
        enable_pruning=False, enable_caching=False, model_path=DEFAULT_MODEL_PATH
    ),
    "Direct NMF": lambda n, s: NMFScheduler(n, seed=s),
    "Whittle Index RMAB": lambda n, s: WhittleIndexRMABScheduler(n, seed=s),
    "Robust PCA + PSR": lambda n, s: RobustPCAPSRScheduler(n, seed=s),
    "Standalone World-Model UCB": lambda n, s: WorldModelUCBScheduler(
        n, world_model_scale=0.35, model_path=DEFAULT_MODEL_PATH
    ),
    "Fixed Sequential Sweep": lambda n, s: FixedScheduler(n),
    "Random Scan": lambda n, s: RandomScheduler(n, seed=s),
}

print("=" * 115)
print("SMARTSCAN V2 — MASTER BENCHMARK & EVALUATION HARNESS")
print(f"Scenarios ({len(SCENARIOS)}): {SCENARIOS}")
print(f"Seeds ({len(SEEDS)}): {SEEDS} | Steps per episode: {STEPS}")
print(f"Total primary episodes: {len(PRIMARY_MODELS) * len(SCENARIOS) * len(SEEDS)} decisions: {len(PRIMARY_MODELS) * len(SCENARIOS) * len(SEEDS) * STEPS:,}")
print("=" * 115)

primary_results = {}
override_stats = {
    "override_decisions": 0,
    "total_decisions": 0,
    "override_hits": 0,
    "ucb_would_hit": 0,
}

for name, builder in PRIMARY_MODELS.items():
    print(f"Evaluating {name}...")
    primary_results[name] = {
        "reward": [],
        "hits": [],
        "switches": [],
        "latencies": [],
        "scenario_rewards": {sc: [] for sc in SCENARIOS},
        "nodes_expanded": [],
        "nodes_pruned": [],
        "cache_hits": [],
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
            ep_nodes_exp = 0
            ep_nodes_prun = 0
            ep_cache_hits = 0

            for st in range(STEPS):
                t0 = time.perf_counter()
                action = sched.select_band()
                lat = (time.perf_counter() - t0) * 1000.0
                ep_latencies.append(lat)

                if hasattr(sched, "get_diagnostics"):
                    diag = sched.get_diagnostics()
                    search_stats = diag.get("search_stats", {})
                    ep_nodes_exp += search_stats.get("nodes_expanded", 0)
                    ep_nodes_prun += search_stats.get("nodes_pruned", 0)
                    ep_cache_hits += search_stats.get("cache_hits", 0)

                    if name == "SmartScan V2 (Full)":
                        override_stats["total_decisions"] += 1
                        if diag.get("override", False):
                            override_stats["override_decisions"] += 1
                            if info.get("true_signal_present", False) and obs.get("detected", 0) > 0.5:
                                override_stats["override_hits"] += 1

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

            primary_results[name]["reward"].append(ep_reward)
            primary_results[name]["hits"].append(ep_hits)
            primary_results[name]["switches"].append(ep_switches)
            primary_results[name]["latencies"].extend(ep_latencies)
            primary_results[name]["scenario_rewards"][sc].append(ep_reward)
            primary_results[name]["nodes_expanded"].append(ep_nodes_exp)
            primary_results[name]["nodes_pruned"].append(ep_nodes_prun)
            primary_results[name]["cache_hits"].append(ep_cache_hits)

total_decisions_per_model = len(SCENARIOS) * len(SEEDS) * STEPS

summary = []
for name, data in primary_results.items():
    mean_r = float(np.mean(data["reward"]))
    median_r = float(np.median(data["reward"]))
    hit_rate = (sum(data["hits"]) / total_decisions_per_model) * 100.0
    mean_sw = float(np.mean(data["switches"]))
    mean_lat = float(np.mean(data["latencies"]))
    mean_exp = float(np.mean(data["nodes_expanded"]))
    mean_prun = float(np.mean(data["nodes_pruned"]))
    summary.append((name, mean_r, median_r, hit_rate, mean_sw, mean_lat, mean_exp, mean_prun, data["scenario_rewards"]))

summary.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 115)
print("SMARTSCAN V2 OFFICIAL MASTER LEADERBOARD (SORTED BY MEAN REWARD)")
print("=" * 115)
print(f"{'Rank':<4} | {'Architecture / Model':<32} | {'Mean Reward':>11} | {'Median':>8} | {'Hit Rate':>8} | {'Switches':>8} | {'Latency':>9}")
print("-" * 115)

rank = 1
for name, mean_r, median_r, hit_rate, mean_sw, mean_lat, mean_exp, mean_prun, _ in summary:
    print(f"{rank:02d}.  | {name:<32} | {mean_r:>+10.2f}  | {median_r:>+7.2f} | {hit_rate:>7.1f}% | {mean_sw:>8.1f} | {mean_lat:>7.3f} ms")
    rank += 1
print("=" * 115)

print("\n--- SCENARIO-BY-SCENARIO DETAILED REWARD BREAKDOWN ---")
header = f"{'Model':<30}" + "".join([f" | {sc[:7]:>7}" for sc in SCENARIOS])
print(header)
print("-" * len(header))
for name, mean_r, median_r, hit_rate, mean_sw, mean_lat, mean_exp, mean_prun, sc_rews in summary:
    row = f"{name[:30]:<30}"
    for sc in SCENARIOS:
        avg_sc_r = float(np.mean(sc_rews[sc]))
        row += f" | {avg_sc_r:>+7.1f}"
    print(row)
print("-" * len(header))

# -------------------------------------------------------------
# Part 2: Systematic Ablation Study
# -------------------------------------------------------------
print("\n" + "=" * 115)
print("PART 2: SYSTEMATIC ARCHITECTURAL ABLATION STUDY")
print("=" * 115)

ABLATIONS = {
    "A. Plain UCB": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=1, w_prediction=0.0, w_age=0.0, w_uncertainty=0.0, w_investigate=0.0, model_path=DEFAULT_MODEL_PATH
    ),
    "B. UCB + Band Belief": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=1, w_prediction=0.35, w_age=0.0, w_uncertainty=0.0, w_investigate=0.0, model_path=DEFAULT_MODEL_PATH
    ),
    "C. UCB + Belief + Uncertainty": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=1, w_prediction=0.35, w_age=0.0, w_uncertainty=0.15, w_investigate=0.0, model_path=DEFAULT_MODEL_PATH
    ),
    "D. UCB + Investigation Priority": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=1, w_prediction=0.0, w_age=0.0, w_uncertainty=0.0, w_investigate=0.20, model_path=DEFAULT_MODEL_PATH
    ),
    "E. Full Track2 Feature Suite": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=1, w_prediction=0.35, w_age=0.10, w_uncertainty=0.15, w_investigate=0.20, model_path=DEFAULT_MODEL_PATH
    ),
    "F. Expectimax Depth-1 Lookahead": lambda n, s: SmartScanScheduler(
        n, depth=1, top_k=5, model_path=DEFAULT_MODEL_PATH
    ),
    "G. Expectimax Depth-2 (Unpruned)": lambda n, s: SmartScanScheduler(
        n, depth=2, top_k=5, branch_k=3, enable_pruning=False, enable_caching=False, model_path=DEFAULT_MODEL_PATH
    ),
    "H. Expectimax Depth-2 (Pruned)": lambda n, s: SmartScanScheduler(
        n, depth=2, top_k=5, branch_k=3, enable_pruning=True, enable_caching=False, model_path=DEFAULT_MODEL_PATH
    ),
    "I. SmartScan V2 (Pruned + Cached)": lambda n, s: SmartScanScheduler(
        n, depth=2, top_k=5, branch_k=3, enable_pruning=True, enable_caching=True, model_path=DEFAULT_MODEL_PATH
    ),
}

ablation_results = {}
for ab_name, ab_builder in ABLATIONS.items():
    ab_rewards = []
    ab_switches = []
    ab_lats = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            env = SmartScanEnv(num_bands=20, episode_length=STEPS, seed=seed, scenario=sc)
            sched = ab_builder(20, seed)
            obs, info = env.reset(seed=seed)
            ep_r = 0.0
            ep_sw = 0
            last_a = -1
            for st in range(STEPS):
                t0 = time.perf_counter()
                a = sched.select_band()
                ab_lats.append((time.perf_counter() - t0) * 1000.0)
                if last_a != -1 and a != last_a:
                    ep_sw += 1
                last_a = a
                obs, r, d, tr, info = env.step(a)
                sched.update(a, r, obs)
                ep_r += r
                if d or tr: break
            ab_rewards.append(ep_r)
            ab_switches.append(ep_sw)

    mean_r = float(np.mean(ab_rewards))
    mean_sw = float(np.mean(ab_switches))
    mean_l = float(np.mean(ab_lats))
    ablation_results[ab_name] = {"mean_reward": mean_r, "switches": mean_sw, "latency_ms": mean_l}
    print(f"  {ab_name:<36} | Reward: {mean_r:>+7.2f} | Switches: {mean_sw:>6.1f} | Latency: {mean_l:>6.3f} ms")

# Statistical Paired Comparison: SmartScan V2 vs. Observable Discounted UCB & Direct NMF
r_v2 = np.array(primary_results["SmartScan V2 (Full)"]["reward"])
r_ucb = np.array(primary_results["Observable Discounted UCB"]["reward"])
r_nmf = np.array(primary_results["Direct NMF"]["reward"])

def paired_stat(a, b):
    diff = a - b
    m = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    return m, [m - 1.95996 * se, m + 1.95996 * se]

m_ucb, ci_ucb = paired_stat(r_v2, r_ucb)
m_nmf, ci_nmf = paired_stat(r_v2, r_nmf)

print("\n" + "=" * 90)
print("STATISTICAL COMPARISON: SMARTSCAN V2 vs. KEY BENCHMARKS")
print("=" * 90)
print(f"  vs. Observable Discounted UCB : Delta = {m_ucb:+.2f} | 95% CI: [{ci_ucb[0]:+.2f}, {ci_ucb[1]:+.2f}] | Excludes Zero? {'YES' if ci_ucb[0]>0 or ci_ucb[1]<0 else 'NO'}")
print(f"  vs. Direct NMF Champion       : Delta = {m_nmf:+.2f} | 95% CI: [{ci_nmf[0]:+.2f}, {ci_nmf[1]:+.2f}] | Excludes Zero? {'YES' if ci_nmf[0]>0 or ci_nmf[1]<0 else 'NO'}")
override_rate = (override_stats["override_decisions"] / max(1, override_stats["total_decisions"])) * 100.0
print(f"  Expectimax Override Rate      : {override_rate:.1f}% ({override_stats['override_decisions']}/{override_stats['total_decisions']} decisions)")
print("=" * 90)

# Save JSON Report
output_path = Path("results/smartscan_v2_master_benchmark.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "steps": STEPS,
        "paired_comparisons": {
            "vs_observable_ucb": {"delta": m_ucb, "ci_95": ci_ucb},
            "vs_direct_nmf": {"delta": m_nmf, "ci_95": ci_nmf},
        },
        "override_stats": {
            "override_rate_pct": override_rate,
            "total_decisions": override_stats["total_decisions"],
            "override_decisions": override_stats["override_decisions"],
        },
        "primary_models": {
            name: {
                "mean_reward": float(np.mean(d["reward"])),
                "median_reward": float(np.median(d["reward"])),
                "hit_rate_pct": float((sum(d["hits"]) / total_decisions_per_model) * 100.0),
                "mean_switches": float(np.mean(d["switches"])),
                "mean_latency_ms": float(np.mean(d["latencies"])),
                "scenario_means": {sc: float(np.mean(d["scenario_rewards"][sc])) for sc in SCENARIOS},
            }
            for name, d in primary_results.items()
        },
        "ablation_results": ablation_results,
    }, f, indent=2)

print(f"\nMaster benchmark report successfully saved to {output_path}")
