"""Heavy Deep Benchmark across all 14 models, 6 full-variate scenarios, 15 fresh seeds.

Evaluates on 15 completely fresh, untouched seeds:
SEEDS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010, 10011, 10012, 10013, 10014, 10015]
Scenarios (6): stationary, hopping, changing, harsh, operational, crowded
Steps per episode: 150
Total episodes: 14 models * 6 scenarios * 15 seeds = 1,260 episodes (189,000 decisions).
"""

import json
import math
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.dwell_dual_policy_scheduler import DwellDualPolicyScheduler
from scheduler.smartscan_omni import SmartScanOmniScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.nmf_expectimax import NMFExpectimaxScheduler
from scheduler.dual_policy_uncertainty_scheduler import DualPolicyUncertaintyScheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.smartscan_v2 import SmartScanScheduler
from scheduler.paradigms.mathematical_scheduling import WhittleIndexRMABScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler
from scheduler.baselines import FixedScheduler, RandomScheduler
from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH
from scheduler.rl_value_network import DEFAULT_VALUE_NET_PATH

SEEDS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010, 10011, 10012, 10013, 10014, 10015]
SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
STEPS = 150

MODELS = {
    "Dwell-Dual Policy (Calibrated V2)": lambda n, s: DwellDualPolicyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.08, dwell_inertia=1.30,
        explore_budget_prob=0.12, fading_grace_steps=1, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Dwell-Dual Policy (Original)": lambda n, s: DwellDualPolicyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.08, dwell_inertia=1.20,
        explore_budget_prob=0.15, fading_grace_steps=0, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "SmartScan-Omni V2 (Tuned)": lambda n, s: SmartScanOmniScheduler(
        n, depth=2, top_k=6, branch_k=3, switch_penalty=0.08, dwell_inertia=1.20,
        hysteresis_margin=0.05, curiosity_scale=0.15, use_rpca_filter=True, rpca_weight=0.25,
        use_rl_critic=True, critic_path=DEFAULT_VALUE_NET_PATH, enable_pruning=True, enable_caching=True,
        seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "SmartScan-Omni V2 (Original)": lambda n, s: SmartScanOmniScheduler(
        n, depth=2, top_k=4, branch_k=3, switch_penalty=0.08, dwell_inertia=1.10,
        hysteresis_margin=0.05, curiosity_scale=0.35, use_rpca_filter=True, rpca_weight=0.25,
        use_rl_critic=True, critic_path=DEFAULT_VALUE_NET_PATH, enable_pruning=True, enable_caching=True,
        seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Robust PCA + PSR": lambda n, s: RobustPCAPSRScheduler(n, seed=s),
    "SmartScan V2-NMF": lambda n, s: NMFExpectimaxScheduler(
        n, depth=2, top_k=4, branch_k=3, switch_penalty=0.08, nmf_weight=1.0,
        curiosity_scale=0.0, use_rl_critic=False,
        enable_pruning=True, enable_caching=True, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Dual-Policy Uncertainty": lambda n, s: DualPolicyUncertaintyScheduler(
        n, nmf_scale=1.0, switch_penalty=0.05, arbitration_mode="probabilistic",
        explore_budget_prob=0.25, seed=s, model_path=DEFAULT_MODEL_PATH
    ),
    "Direct NMF": lambda n, s: NMFScheduler(n, seed=s),
    "Static NMF+UCB": lambda n, s: WorldModelNMFUCBScheduler(
        n, nmf_scale=1.0, world_model_scale=0.0, switch_penalty=0.05, model_path=DEFAULT_MODEL_PATH
    ),
    "SmartScan V2 (UCB-based)": lambda n, s: SmartScanScheduler(
        n, depth=2, top_k=5, branch_k=3, enable_pruning=True, enable_caching=True, model_path=DEFAULT_MODEL_PATH
    ),
    "Whittle Index RMAB": lambda n, s: WhittleIndexRMABScheduler(n, seed=s),
    "Observable Plain UCB": lambda n, s: ObservableDiscountedUCBScheduler(
        n, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True
    ),
    "Random Scan": lambda n, s: RandomScheduler(n, seed=s),
    "Fixed Sequential Sweep": lambda n, s: FixedScheduler(n),
}

total_episodes_all = len(MODELS) * len(SCENARIOS) * len(SEEDS)
total_decisions_all = total_episodes_all * STEPS
episodes_per_model = len(SCENARIOS) * len(SEEDS)
decisions_per_model = episodes_per_model * STEPS

print("=" * 130)
print("HEAVY DEEP BENCHMARK: 14 MODELS, 6 SCENARIOS, 15 FRESH SEEDS")
print(f"Seeds ({len(SEEDS)}): {SEEDS}")
print(f"Scenarios ({len(SCENARIOS)}): {SCENARIOS} | Steps: {STEPS}")
print(f"Total: {len(MODELS)} models * {len(SCENARIOS)} scenarios * {len(SEEDS)} seeds = {total_episodes_all:,} episodes ({total_decisions_all:,} decisions)")
print("=" * 130)

results = {}

for name, builder in MODELS.items():
    print(f"Evaluating: {name:<35} ... ", end="", flush=True)
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

summary = []
for name, data in results.items():
    arr = np.array(data["reward"], dtype=np.float64)
    mean_r = float(np.mean(arr))
    median_r = float(np.median(arr))
    std_r = float(np.std(arr, ddof=1))
    se_r = std_r / np.sqrt(len(arr))
    ci_low = mean_r - 1.95996 * se_r
    ci_high = mean_r + 1.95996 * se_r

    hit_rate = (sum(data["hits"]) / decisions_per_model) * 100.0
    mean_sw = float(np.mean(data["switches"]))
    mean_sw_dist = float(np.mean(data["switch_dists"]))
    mean_lat = float(np.mean(data["latencies"]))
    p99_lat = float(np.percentile(data["latencies"], 99))

    summary.append((
        name, mean_r, median_r, (ci_low, ci_high), hit_rate, mean_sw, mean_sw_dist, mean_lat, p99_lat, data["scenario_rewards"]
    ))

summary.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 140)
print(f"OFFICIAL HEAVY BENCHMARK MASTER LEADERBOARD (N={episodes_per_model} EPISODES PER MODEL, 15 SEEDS)")
print("=" * 140)
print(f"{'Rank':<4} | {'Architecture / Model':<35} | {'Mean Reward (95% CI)':^26} | {'Median':>7} | {'Hit Rate':>8} | {'Switches':>8} | {'Mean Lat':>8} | {'P99 Lat':>8}")
print("-" * 140)

rank = 1
for name, mean_r, median_r, (ci_l, ci_h), hit_rate, mean_sw, mean_sw_dist, mean_lat, p99_lat, _ in summary:
    ci_str = f"{mean_r:>+6.2f} [{ci_l:>+6.2f}, {ci_h:>+6.2f}]"
    print(f"{rank:02d}.  | {name:<35} | {ci_str:^26} | {median_r:>+6.2f} | {hit_rate:>7.1f}% | {mean_sw:>8.1f} | {mean_lat:>6.3f}ms | {p99_lat:>6.3f}ms")
    rank += 1
print("=" * 140)

print("\n--- SCENARIO-BY-SCENARIO DETAILED REWARD BREAKDOWN (15-SEED MEANS) ---")
header = f"{'Model':<35}" + "".join([f" | {sc[:7]:>7}" for sc in SCENARIOS])
print(header)
print("-" * len(header))
for name, mean_r, median_r, (ci_l, ci_h), hit_rate, mean_sw, mean_sw_dist, mean_lat, p99_lat, sc_rews in summary:
    row = f"{name[:35]:<35}"
    for sc in SCENARIOS:
        avg_sc_r = float(np.mean(sc_rews[sc]))
        row += f" | {avg_sc_r:>+7.1f}"
    print(row)
print("-" * len(header))

# Paired hypothesis testing vs Top Model
top_model_name = summary[0][0]
r_top = np.array(results[top_model_name]["reward"], dtype=np.float64)

def run_paired_test(a, b, label_a, label_b):
    diff = a - b
    delta = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    ci_l = delta - 1.95996 * se
    ci_h = delta + 1.95996 * se
    t_stat = delta / max(1e-8, se)
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    sig = "YES (p < 0.05)" if (ci_l > 0 or ci_h < 0) else "NO (Tied)"
    print(f"  {label_a:<35} vs. {label_b:<35} : Delta = {delta:>+6.2f} | 95% CI: [{ci_l:>+6.2f}, {ci_h:>+6.2f}] | t = {t_stat:>+6.2f} | p = {p_val:.4f} | Sig? {sig}")

print("\n" + "=" * 125)
print(f"PAIRED HYPOTHESIS TESTING VS TOP MODEL ({top_model_name}, N = {episodes_per_model} PAIRS)")
print("=" * 125)
for name, _, _, _, _, _, _, _, _, _ in summary[1:]:
    r_other = np.array(results[name]["reward"], dtype=np.float64)
    run_paired_test(r_top, r_other, top_model_name, name)
print("=" * 125)

# Save JSON report
output_path = Path("results/heavy_deep_benchmark.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "steps": STEPS,
        "total_episodes": total_episodes_all,
        "total_decisions": total_decisions_all,
        "models": {
            name: {
                "mean_reward": float(np.mean(d["reward"])),
                "median_reward": float(np.median(d["reward"])),
                "ci_95": [float(np.mean(d["reward"]) - 1.95996 * np.std(d["reward"], ddof=1)/np.sqrt(len(d["reward"]))),
                          float(np.mean(d["reward"]) + 1.95996 * np.std(d["reward"], ddof=1)/np.sqrt(len(d["reward"])))],
                "hit_rate_pct": float((sum(d["hits"]) / decisions_per_model) * 100.0),
                "mean_switches": float(np.mean(d["switches"])),
                "mean_switch_dist": float(np.mean(d["switch_dists"])),
                "mean_latency_ms": float(np.mean(d["latencies"])),
                "p99_latency_ms": float(np.percentile(d["latencies"], 99)),
                "scenario_means": {sc: float(np.mean(d["scenario_rewards"][sc])) for sc in SCENARIOS},
                "raw_rewards": [float(x) for x in d["reward"]],
            }
            for name, d in results.items()
        }
    }, f, indent=2)

print(f"\nHeavy benchmark results successfully written to {output_path}")
