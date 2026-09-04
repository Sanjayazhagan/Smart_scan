"""Master Benchmark (12 Competitors): Testing Option 2 Dwell-Dual Policy Scheduler.

Evaluates on the untouched benchmark seeds: [8001, 8002, 8003, 8004, 8005]
Across all 6 scenarios: stationary, hopping, changing, harsh, operational, crowded
Total Decisions: 12 models * 6 scenarios * 5 seeds * 150 steps = 54,000 decisions.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.dwell_dual_policy_scheduler import DwellDualPolicyScheduler
from scheduler.dual_policy_uncertainty_scheduler import DualPolicyUncertaintyScheduler
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.alarm_adaptive_nmf_ucb import AlarmAdaptiveNMFUCBScheduler
from scheduler.world_model_ucb import WorldModelUCBScheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.paradigms.mathematical_scheduling import WhittleIndexRMABScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler
from scheduler.baselines import FixedScheduler, RandomScheduler
from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH

MODELS = {
    "Dwell-Dual Policy (Option 2)": lambda n, s: DwellDualPolicyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.08, dwell_inertia=1.20,
        explore_budget_prob=0.15, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Dual-Policy Uncertainty (Old)": lambda n, s: DualPolicyUncertaintyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.05, arbitration_mode="probabilistic",
        explore_budget_prob=0.25, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Direct NMF": lambda n, s: NMFScheduler(n, seed=s),
    "Static NMF+UCB (Baseline)": lambda n, s: WorldModelNMFUCBScheduler(
        n, nmf_scale=1.0, world_model_scale=0.0, switch_penalty=0.05, model_path=DEFAULT_MODEL_PATH
    ),
    "Neural-Guided NMF+UCB": lambda n, s: WorldModelNMFUCBScheduler(
        n, nmf_scale=1.0, world_model_scale=0.35, switch_penalty=0.05, model_path=DEFAULT_MODEL_PATH
    ),
    "Robust PCA + PSR": lambda n, s: RobustPCAPSRScheduler(n, seed=s),
    "Whittle Index RMAB": lambda n, s: WhittleIndexRMABScheduler(n, seed=s),
    "Fixed Sequential Sweep": lambda n, s: FixedScheduler(n),
    "Standalone World-Model UCB": lambda n, s: WorldModelUCBScheduler(
        n, world_model_scale=0.35, model_path=DEFAULT_MODEL_PATH
    ),
    "Random Scan": lambda n, s: RandomScheduler(n, seed=s),
    "Alarm-Adaptive NMF+UCB": lambda n, s: AlarmAdaptiveNMFUCBScheduler(
        n, nmf_scale=1.0, alarm_threshold=0.80, model_path=DEFAULT_MODEL_PATH
    ),
    "Observable Plain UCB": lambda n, s: ObservableDiscountedUCBScheduler(
        n, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True
    ),
}

SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
SEEDS = [8001, 8002, 8003, 8004, 8005]
STEPS = 150

print("=" * 115)
print("MASTER TRUE BENCHMARK (12 COMPETITORS: EVALUATING OPTION 2 DWELL-DUAL POLICY)")
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

# Statistical Paired Tests vs Direct NMF and vs Static NMF+UCB Baseline
r_dwell = np.array(results["Dwell-Dual Policy (Option 2)"]["reward"])
r_nmf = np.array(results["Direct NMF"]["reward"])
r_base = np.array(results["Static NMF+UCB (Baseline)"]["reward"])

diff_vs_nmf = r_dwell - r_nmf
diff_vs_base = r_dwell - r_base

def calc_ci(diff_arr):
    m = float(np.mean(diff_arr))
    se = float(np.std(diff_arr, ddof=1) / np.sqrt(len(diff_arr)))
    return m, [m - 1.95996 * se, m + 1.95996 * se]

m_nmf, ci_nmf = calc_ci(diff_vs_nmf)
m_base, ci_base = calc_ci(diff_vs_base)

print("\n" + "=" * 90)
print("STATISTICAL COMPARISON: DWELL-DUAL POLICY vs. COMPETITORS")
print("=" * 90)
print(f"  vs. Direct NMF Champion   : Delta = {m_nmf:+.2f} | 95% CI: [{ci_nmf[0]:+.2f}, {ci_nmf[1]:+.2f}] | Excludes Zero? {'YES' if ci_nmf[0]>0 or ci_nmf[1]<0 else 'NO'}")
print(f"  vs. Static NMF+UCB Base   : Delta = {m_base:+.2f} | 95% CI: [{ci_base[0]:+.2f}, {ci_base[1]:+.2f}] | Excludes Zero? {'YES' if ci_base[0]>0 or ci_base[1]<0 else 'NO'}")
print("=" * 90)

# Save clean JSON
output_file = Path("results/final_true_benchmark_12models.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "steps": STEPS,
        "comparisons": {
            "vs_direct_nmf": {"delta": float(m_nmf), "ci_95": [float(ci_nmf[0]), float(ci_nmf[1])]},
            "vs_static_nmf_ucb": {"delta": float(m_base), "ci_95": [float(ci_base[0]), float(ci_base[1])]},
        },
        "models": {
            name: {
                "mean_reward": float(np.mean(d["reward"])),
                "median_reward": float(np.median(d["reward"])),
                "hit_rate_pct": float((sum(d["hits"]) / total_decisions_per_model) * 100.0),
                "mean_switches": float(np.mean(d["switches"])),
                "mean_latency_ms": float(np.mean(d["latencies"])),
                "scenario_means": {sc: float(np.mean(d["scenario_rewards"][sc])) for sc in SCENARIOS},
                "raw_rewards": [float(r) for r in d["reward"]]
            }
            for name, d in results.items()
        }
    }, f, indent=2)

print(f"\nRaw results successfully saved to {output_file}")
