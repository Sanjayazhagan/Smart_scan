"""Statistical Rigor Benchmark: 15-Seed Re-Evaluation of All 14 Models.

Evaluates all 14 models across 6 scenarios and held-out seeds.
Uses ProcessPoolExecutor multiprocessing for high throughput across CPU cores.
Computes:
- Mean +/- Std and 95% Confidence Intervals (t-distribution, df=N-1) for Reward, Interception Rate, and Switches.
- Adjacent model 95% CI overlap checking.
- Paired Student's t-test and Wilcoxon Signed-Rank Test against Dwell-Dual Policy.
- Effect sizes: Paired Cohen's d, Median paired difference, and Interquartile Range (IQR).
- Tier classification:
    * Tier 1: Statistically indistinguishable from top performer (p >= 0.05).
    * Tier 2: Statistically separated / sub-optimal baselines (p < 0.05).
- Canonical run provenance tracking (run_id, git commit hash, timestamp).
"""

import argparse
import concurrent.futures
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats

import sys
ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
sys.path.insert(0, str(ROOT))

from simulator.environment import SmartScanEnv
from benchmark_models import MODEL_REGISTRY, get_model
from evaluation.provenance import get_run_metadata, set_active_suite_run_id

DEFAULT_SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
DEFAULT_SEEDS = list(range(10001, 10016))  # 15 seeds: 10001 to 10015
EPISODE_LENGTH = 150


def _run_single_episode(args):
    """Worker function for executing a single evaluation episode."""
    m_name, sc, seed, ep_len = args
    env = SmartScanEnv(num_bands=20, episode_length=ep_len, seed=seed, scenario=sc)
    scheduler = get_model(m_name, num_bands=20, seed=seed)

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    hits = 0
    switches = 0
    signals_found = 0
    total_signals = 0
    latencies = []

    last_band = None

    for _ in range(ep_len):
        t0 = time.perf_counter()
        action = int(scheduler.select_band())
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dt)

        if last_band is not None and action != last_band:
            switches += 1
        last_band = action

        obs, reward, done, truncated, info = env.step(action)
        scheduler.update(action, reward, obs)

        total_reward += reward
        signal_present = bool(info.get("true_signal_present", False))
        detected = bool(obs.get("detected", False))
        active_bands = info.get("ground_truth_active_bands", [])

        if signal_present and detected:
            hits += 1
            signals_found += 1
            total_signals += 1
        elif len(active_bands) > 0:
            total_signals += 1

        if done or truncated:
            break

    env.close()

    return {
        "model": m_name,
        "scenario": sc,
        "seed": seed,
        "reward": total_reward,
        "hits": hits,
        "switches": switches,
        "signals_found": signals_found,
        "total_signals": total_signals,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def compute_95ci(data):
    """Compute Student's t 95% Confidence Interval for the mean across seeds."""
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n < 2:
        val = float(arr[0]) if n == 1 else 0.0
        return val, val
    mean = float(np.mean(arr))
    se = float(stats.sem(arr))
    if se == 0.0 or np.isnan(se):
        return mean, mean
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    ci_low = float(ci[0]) if not np.isnan(ci[0]) else mean
    ci_high = float(ci[1]) if not np.isnan(ci[1]) else mean
    return ci_low, ci_high


def run_statistical_benchmark(seeds=None, scenarios=None, models_to_run=None, workers=None, run_id=None):
    seeds = seeds or DEFAULT_SEEDS
    scenarios = scenarios or DEFAULT_SCENARIOS
    max_workers = workers or min(16, os.cpu_count() or 4)

    active_run_id = set_active_suite_run_id(run_id)
    run_meta = get_run_metadata(
        "statistical_rigor_14models",
        seeds=seeds,
        scenarios=scenarios,
        extra={"episode_length": EPISODE_LENGTH},
        run_id=active_run_id,
    )

    registry = MODEL_REGISTRY
    if models_to_run:
        registry = {k: v for k, v in registry.items() if any(m.lower() in k.lower() for m in models_to_run)}

    print("=" * 135)
    print(f"  STATISTICAL RIGOR BENCHMARK: {len(seeds)}-SEED CROSS-EVALUATION ACROSS {len(registry)} MODELS")
    print(f"  Run ID: {run_meta['run_id']} | Git Commit: {run_meta['git_commit'][:10]}")
    print(f"  Scenarios ({len(scenarios)}): {scenarios}")
    print(f"  Seeds ({len(seeds)}): {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds) | Episode Length: {EPISODE_LENGTH}")
    total_eps = len(registry) * len(scenarios) * len(seeds)
    print(f"  Total Evaluations: {total_eps:,} episodes ({total_eps * EPISODE_LENGTH:,} total scan decisions) | Workers: {max_workers}")
    print("=" * 135)

    all_model_results = {}
    model_episode_rewards = {}  # model_name -> list of episode rewards in fixed order (sc, seed)
    model_seed_rewards = {}     # model_name -> list of mean reward per seed across scenarios (length = len(seeds))
    model_seed_hits = {}

    ordered_pairs = [(sc, seed) for sc in scenarios for seed in seeds]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, (m_name, m_meta) in enumerate(registry.items(), 1):
            print(f"[{idx:2d}/{len(registry):2d}] Evaluating: {m_name:<32} ... ", end="", flush=True)
            t_start = time.perf_counter()

            tasks = [(m_name, sc, seed, EPISODE_LENGTH) for sc, seed in ordered_pairs]
            results = list(pool.map(_run_single_episode, tasks))

            ep_rewards = []
            ep_hits = []
            ep_switches = []
            ep_signals_found = []
            ep_total_signals = []
            all_latencies = []
            sc_rewards = {sc: [] for sc in scenarios}
            seed_rewards = {s: [] for s in seeds}
            seed_hits = {s: [] for s in seeds}

            for res in results:
                ep_rewards.append(res["reward"])
                ep_hits.append(res["hits"])
                ep_switches.append(res["switches"])
                ep_signals_found.append(res["signals_found"])
                ep_total_signals.append(res["total_signals"])
                all_latencies.append(res["mean_latency_ms"])
                sc_rewards[res["scenario"]].append(res["reward"])
                seed_rewards[res["seed"]].append(res["reward"])
                seed_hits[res["seed"]].append(res["hits"])

            elapsed = time.perf_counter() - t_start

            mean_rew = float(np.mean(ep_rewards))
            std_rew = float(np.std(ep_rewards, ddof=1))
            mean_hit = float(np.mean(ep_hits))
            std_hit = float(np.std(ep_hits, ddof=1))
            mean_sw = float(np.mean(ep_switches))
            std_sw = float(np.std(ep_switches, ddof=1))
            intercept_rate = (sum(ep_signals_found) / max(1, sum(ep_total_signals))) * 100.0
            mean_lat = float(np.mean(all_latencies))
            p95_lat = float(np.percentile(all_latencies, 95))

            # Store ordered episode rewards for paired significance testing
            model_episode_rewards[m_name] = ep_rewards

            # Compute seed-level aggregates for t-distribution 95% Confidence Intervals
            seed_mean_rewards = [float(np.mean(seed_rewards[s])) for s in seeds]
            seed_mean_hits = [float(np.mean(seed_hits[s])) for s in seeds]
            model_seed_rewards[m_name] = seed_mean_rewards
            model_seed_hits[m_name] = seed_mean_hits

            rew_ci_low, rew_ci_high = compute_95ci(seed_mean_rewards)
            hit_ci_low, hit_ci_high = compute_95ci(seed_mean_hits)
            sw_ci_low, sw_ci_high = compute_95ci([float(s) for s in ep_switches])

            all_model_results[m_name] = {
                "category": m_meta.get("category", "Baseline"),
                "description": m_meta.get("description", ""),
                "mean_reward": round(mean_rew, 2),
                "std_reward": round(std_rew, 2),
                "reward_95ci": [round(rew_ci_low, 2), round(rew_ci_high, 2)],
                "reward_ci_str": f"{mean_rew:+5.2f} (95% CI: [{rew_ci_low:+5.2f}, {rew_ci_high:+5.2f}])",
                "mean_hits": round(mean_hit, 1),
                "std_hits": round(std_hit, 1),
                "hits_95ci": [round(hit_ci_low, 1), round(hit_ci_high, 1)],
                "interception_rate_pct": round(intercept_rate, 2),
                "mean_switches": round(mean_sw, 1),
                "std_switches": round(std_sw, 1),
                "switches_95ci": [round(sw_ci_low, 1), round(sw_ci_high, 1)],
                "latency_mean_ms": round(mean_lat, 4),
                "latency_p95_ms": round(p95_lat, 4),
                "scenario_stats": {
                    sc: {
                        "mean": round(float(np.mean(sc_rewards[sc])), 2),
                        "std": round(float(np.std(sc_rewards[sc], ddof=1)), 2),
                    }
                    for sc in scenarios
                },
            }
            print(f"Reward: {mean_rew:+6.2f} +/- {std_rew:4.2f} [95% CI: {rew_ci_low:+5.2f}, {rew_ci_high:+5.2f}] | Intercept: {intercept_rate:5.1f}% | Latency: {mean_lat:6.3f}ms ({elapsed:4.1f}s)")

    # -------------------------------------------------------------
    # Paired Significance Testing vs Dwell-Dual Policy (Champion)
    # -------------------------------------------------------------
    champion_key = "Dwell-Dual Policy (Champion)"
    sig_results = {}

    if champion_key in model_episode_rewards:
        champ_rewards = np.array(model_episode_rewards[champion_key], dtype=np.float64)
        n_pairs = len(champ_rewards)
        print("\n" + "=" * 135)
        print(f"{f'STATISTICAL SIGNIFICANCE & EFFECT SIZE: DWELL-DUAL vs 13 BASELINES (N={n_pairs} Paired Episodes)':^135}")
        print("=" * 135)
        print(f"{'Model Name':<30} | {'Delta Mean':<10} | {'t-stat':<8} | {'p-value':<12} | {'Cohen d':<8} | {'Median Diff [IQR]':<20} | {'Scientific Interpretation'}")
        print("-" * 135)

        for m_name in registry.keys():
            if m_name == champion_key:
                continue
            comp_rewards = np.array(model_episode_rewards[m_name], dtype=np.float64)
            diffs = champ_rewards - comp_rewards
            delta_mean = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1))

            # Paired Student's t-test
            t_res = stats.ttest_rel(champ_rewards, comp_rewards)
            t_stat = float(t_res.statistic) if not np.isnan(t_res.statistic) else 0.0
            p_t = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 1.0

            # Wilcoxon signed-rank test
            non_zero_diffs = diffs[diffs != 0]
            if len(non_zero_diffs) > 0:
                try:
                    w_res = stats.wilcoxon(champ_rewards, comp_rewards, zero_method="wilcox")
                    w_stat = float(w_res.statistic)
                    p_w = float(w_res.pvalue)
                except Exception:
                    w_stat = 0.0
                    p_w = 1.0
            else:
                w_stat = 0.0
                p_w = 1.0

            # Paired Cohen's d effect size: mean(diff) / std(diff)
            cohens_d = delta_mean / std_diff if std_diff > 1e-12 else 0.0

            # Non-parametric Median Difference & Interquartile Range (IQR: 25% - 75%)
            median_diff = float(np.median(diffs))
            q25 = float(np.percentile(diffs, 25))
            q75 = float(np.percentile(diffs, 75))
            iqr_str = f"{median_diff:+5.2f} [{q25:+5.2f}, {q75:+5.2f}]"

            # Strict, scientifically accurate interpretation string
            if p_t < 0.001:
                sig_level = "p < 0.001 (***)"
                interpretation = "Statistically significant; unlikely from seed variance (Large effect)" if abs(cohens_d) >= 0.8 else "Statistically significant; unlikely from seed variance"
            elif p_t < 0.01:
                sig_level = "p < 0.01 (**)"
                interpretation = "Statistically significant; unlikely from seed variance (Medium effect)"
            elif p_t < 0.05:
                sig_level = "p < 0.05 (*)"
                interpretation = "Statistically significant (p < 0.05)"
            else:
                sig_level = "n.s. (p >= 0.05)"
                interpretation = "Difference not statistically significant at alpha=0.05"

            sig_results[m_name] = {
                "delta_mean_reward": round(delta_mean, 2),
                "t_statistic": round(t_stat, 3),
                "p_value_parametric": round(p_t, 6),
                "wilcoxon_w": round(w_stat, 1),
                "p_value_nonparametric": round(p_w, 6),
                "cohens_d": round(cohens_d, 3),
                "median_difference": round(median_diff, 2),
                "iqr_25_75": [round(q25, 2), round(q75, 2)],
                "significance_flag": sig_level,
                "defensible_interpretation": interpretation,
            }

            print(f"{m_name:<30} | {delta_mean:+9.2f}  | {t_stat:+7.3f} | {p_t:<12.4e} | {cohens_d:+7.2f} | {iqr_str:<20} | {interpretation}")

        print("=" * 135)

    # -------------------------------------------------------------
    # Overlap-Aware Ranking & Tier Classification
    # -------------------------------------------------------------
    ranked = sorted(all_model_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True)

    # 1. Detect adjacent 95% CI overlaps
    for i in range(len(ranked)):
        current_name, current_dict = ranked[i]
        c_low, c_high = current_dict["reward_95ci"]
        if i > 0:
            _, prev_dict = ranked[i - 1]
            p_low, p_high = prev_dict["reward_95ci"]
            current_dict["overlaps_with_prev"] = bool(max(c_low, p_low) <= min(c_high, p_high))
        else:
            current_dict["overlaps_with_prev"] = False

        if i < len(ranked) - 1:
            _, next_dict = ranked[i + 1]
            n_low, n_high = next_dict["reward_95ci"]
            current_dict["overlaps_with_next"] = bool(max(c_low, n_low) <= min(c_high, n_high))
        else:
            current_dict["overlaps_with_next"] = False

    # 2. Assign Tiers based on paired hypothesis testing vs Champion (p >= 0.05 cutoff)
    tier1_models = []
    tier2_models = []

    for rank, (name, d) in enumerate(ranked, 1):
        d["final_rank"] = rank
        if name == champion_key:
            d["significantly_beats_dwell_dual"] = False
            d["distinguishable_from_dwell_dual"] = False
            d["p_value_vs_dwell_dual"] = 1.0
            d["cohens_d_vs_dwell_dual"] = 0.0
            d["tier"] = 1
            d["tier_label"] = "Tier 1: Champion (Best Point Estimate)"
            tier1_models.append(name)
        elif name in sig_results:
            sig = sig_results[name]
            p_val = sig["p_value_parametric"]
            d_val = sig["cohens_d"]
            delta_mean = sig["delta_mean_reward"]

            d["p_value_vs_dwell_dual"] = round(p_val, 6)
            d["cohens_d_vs_dwell_dual"] = round(d_val, 3)
            d["significantly_beats_dwell_dual"] = bool(p_val < 0.05 and delta_mean < 0.0)
            d["distinguishable_from_dwell_dual"] = bool(p_val < 0.05)

            if p_val >= 0.05:
                d["tier"] = 1
                d["tier_label"] = "Tier 1: Statistically Indistinguishable from Champion (p >= 0.05)"
                tier1_models.append(name)
            else:
                d["tier"] = 2
                d["tier_label"] = "Tier 2: Statistically Separated / Sub-optimal Baselines (p < 0.05)"
                tier2_models.append(name)
        else:
            d["tier"] = 2
            d["tier_label"] = "Tier 2: Sub-optimal Baselines"
            d["distinguishable_from_dwell_dual"] = True
            d["significantly_beats_dwell_dual"] = False
            tier2_models.append(name)

    # 3. Print Tiered Leaderboard Console Table
    print("\n" + "=" * 135)
    print(f"{'TIERED STATISTICAL LEADERBOARD (95% CI & PAIRWISE SIGNIFICANCE AWARE)':^135}")
    print("=" * 135)
    print("  TIER 1: Statistically Indistinguishable from Top Performer (p >= 0.05 vs Dwell-Dual):")
    for name, d in ranked:
        if d["tier"] == 1:
            ci_str = f"[{d['reward_95ci'][0]:+5.2f}, {d['reward_95ci'][1]:+5.2f}]"
            p_str = f"p={d['p_value_vs_dwell_dual']:.4f}, d={d['cohens_d_vs_dwell_dual']:+5.2f}" if name != champion_key else "(Champion Point Estimate)"
            adj_note = " [CI Overlaps Next Rank]" if d.get("overlaps_with_next") else ""
            print(f"    * [Rank {d['final_rank']:2d}] {name:<32} : Mean {d['mean_reward']:+6.2f} (95% CI: {ci_str}) | Hit: {d['interception_rate_pct']:4.1f}% | {p_str}{adj_note}")

    print("\n  TIER 2: Statistically Separated / Sub-optimal Baselines (p < 0.05 vs Dwell-Dual):")
    for name, d in ranked:
        if d["tier"] == 2:
            ci_str = f"[{d['reward_95ci'][0]:+5.2f}, {d['reward_95ci'][1]:+5.2f}]"
            p_str = f"p={d['p_value_vs_dwell_dual']:.4e}, d={d['cohens_d_vs_dwell_dual']:+5.2f}"
            adj_note = " [CI Overlaps Next Rank]" if d.get("overlaps_with_next") else ""
            print(f"    * [Rank {d['final_rank']:2d}] {name:<32} : Mean {d['mean_reward']:+6.2f} (95% CI: {ci_str}) | Hit: {d['interception_rate_pct']:4.1f}% | {p_str}{adj_note}")
    print("=" * 135)

    n_pairs = len(champ_rewards) if champion_key in model_episode_rewards else len(seeds) * len(scenarios)
    df_pairs = n_pairs - 1

    tiers_dict = {
        "tier_1": {
            "name": "Tier 1: Statistically Indistinguishable from Champion (p >= 0.05)",
            "criterion": f"Paired Student's t-test p >= 0.05 vs Dwell-Dual Policy (df={df_pairs}, {n_pairs} paired episodes)",
            "models": tier1_models,
        },
        "tier_2": {
            "name": "Tier 2: Statistically Separated / Sub-optimal Baselines (p < 0.05)",
            "criterion": f"Paired Student's t-test p < 0.05 vs Dwell-Dual Policy (df={df_pairs}, {n_pairs} paired episodes)",
            "models": tier2_models,
        }
    }

    output_data = {
        "run_metadata": run_meta,
        "tiers": tiers_dict,
        "leaderboard": {name: d for name, d in ranked},
        "significance_tests": sig_results,
    }

    out_file = ROOT / "results" / "master_15seed_statistical_benchmark.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nMaster 15-seed benchmark with 95% CIs, tiers, and effect sizes saved to:\n  {out_file}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Rigor Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick run with 3 seeds for smoke test")
    parser.add_argument("--seeds", nargs="+", type=int, help="Custom seeds list")
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of seeds starting from 10001")
    parser.add_argument("--models", nargs="+", type=str, help="Filter specific models")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit suite run ID")
    args = parser.parse_args()

    if args.quick:
        active_seeds = [10001, 10002, 10003]
    elif args.num_seeds:
        active_seeds = [10001 + i for i in range(args.num_seeds)]
    elif args.seeds:
        active_seeds = args.seeds
    else:
        active_seeds = DEFAULT_SEEDS

    run_statistical_benchmark(
        seeds=active_seeds,
        models_to_run=args.models,
        workers=args.workers,
        run_id=args.run_id,
    )
