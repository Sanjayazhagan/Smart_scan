"""N=30 Aggregated Seed Statistical Benchmark with Holm-Bonferroni Correction.

Evaluates all 18 models across 30 independent seeds and 6 tactical scenarios:
1. Aggregates the 6 scenario scores per seed -> N=30 independent observations per model (df=29).
2. Computes paired Student's t-test and Wilcoxon signed-rank test against Dwell-Dual.
3. Applies Holm-Bonferroni step-down correction across all 17 baseline comparisons.
4. Computes Seed Win Rate (wins/30, losses/30, ties/30, win rate %).
5. Computes Cohen's d effect sizes, 95% CIs, and per-scenario breakdowns.
6. Saves results to results/master_n30_aggregated_benchmark.json.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
sys.path.insert(0, str(ROOT))

import scheduler.stats_compat as stats

from simulator.environment import SmartScanEnv
from benchmark_models import MODEL_REGISTRY, get_model
from evaluation.provenance import get_run_metadata, set_active_suite_run_id

DEFAULT_SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
DEFAULT_SEEDS = list(range(10001, 10031))  # N=30 seeds: 10001 to 10030
EPISODE_LENGTH = 150
CHAMPION_MODEL_NAME = "Dwell-Dual Policy (Champion)"
CHAMPION_KEY = CHAMPION_MODEL_NAME


def _run_single_episode(args):
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
        dt = (time.perf_counter() - t0) * 1000.0
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
        "reward": float(total_reward),
        "hits": int(hits),
        "switches": int(switches),
        "signals_found": int(signals_found),
        "total_signals": int(total_signals),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def compute_95ci(arr):
    ci = stats.ci95(arr)
    return float(ci[0]), float(ci[1])


def holm_bonferroni(p_dict):
    """Applies Holm-Bonferroni step-down correction to a dict of {model: raw_p_value}."""
    sorted_items = sorted(p_dict.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adj_p = {}
    running_max = 0.0
    for rank_idx, (model_name, p_val) in enumerate(sorted_items):
        multiplier = m - rank_idx
        p_candidate = min(1.0, p_val * multiplier)
        running_max = max(running_max, p_candidate)
        adj_p[model_name] = min(1.0, running_max)
    return adj_p


def run_benchmark(num_seeds=30, workers=None, recompute_champion=False):
    seeds = [10001 + i for i in range(num_seeds)]
    scenarios = DEFAULT_SCENARIOS
    models = list(MODEL_REGISTRY.keys())
    max_workers = workers or min(24, os.cpu_count() or 4)

    print("=" * 135)
    print(f"N={num_seeds} AGGREGATED STATISTICAL BENCHMARK WITH HOLM-BONFERRONI CORRECTION")
    print(f"Seeds: {num_seeds} (10001 to {10000 + num_seeds}) | Scenarios: {len(scenarios)} | Models: {len(models)}")
    print(f"Total Decisions: {len(models) * num_seeds * len(scenarios) * EPISODE_LENGTH:,} | Workers: {max_workers}")
    print("=" * 135)

    # Build work queue
    cache_file = ROOT / "results" / f".cache_n{num_seeds}_m{len(models)}_raw_results.json"
    if cache_file.exists():
        print(f"Loading cached raw simulation results from:\n  {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        raw_results = {
            m: {int(s): sc_dict for s, sc_dict in s_dict.items()}
            for m, s_dict in cached_data["raw_results"].items()
        }
        model_latencies = cached_data["model_latencies"]
        print("Cached episode results loaded successfully!")

        if recompute_champion:
            print(f"Recomputing episodes for Champion ({CHAMPION_MODEL_NAME}) with updated logic...")
            champ_tasks = [(CHAMPION_MODEL_NAME, sc, s, EPISODE_LENGTH) for sc in scenarios for s in seeds]
            raw_results[CHAMPION_MODEL_NAME] = {s: {} for s in seeds}
            model_latencies[CHAMPION_MODEL_NAME] = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for res in executor.map(_run_single_episode, champ_tasks, chunksize=4):
                    s = res["seed"]
                    sc = res["scenario"]
                    raw_results[CHAMPION_MODEL_NAME][s][sc] = res
                    model_latencies[CHAMPION_MODEL_NAME].append(res["mean_latency_ms"])
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"raw_results": raw_results, "model_latencies": model_latencies}, f)
            print(f"Updated cache file with fresh champion evaluations: {cache_file}")

        print("Proceeding to aggregation...")
    else:
        tasks = []
        for m in models:
            for sc in scenarios:
                for s in seeds:
                    tasks.append((m, sc, s, EPISODE_LENGTH))

        t_start = time.time()
        raw_results = {m: {s: {} for s in seeds} for m in models}
        model_latencies = {m: [] for m in models}

        print(f"Executing {len(tasks):,} total episodes across {max_workers} worker processes...")
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(_run_single_episode, tasks, chunksize=4):
                m = res["model"]
                sc = res["scenario"]
                s = res["seed"]
                raw_results[m][s][sc] = res
                model_latencies[m].append(res["mean_latency_ms"])
                completed += 1
                if completed % 360 == 0 or completed == len(tasks):
                    pct = completed / len(tasks) * 100.0
                    elapsed = time.time() - t_start
                    print(f"  [Progress] {completed:4d} / {len(tasks)} episodes ({pct:5.1f}%) | Elapsed: {elapsed:5.1f}s")

        t_eval = time.time() - t_start
        print(f"\nAll episodes completed in {t_eval:.1f}s ({t_eval/60:.2f} min). Caching raw results...")
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"raw_results": raw_results, "model_latencies": model_latencies}, f)
        print(f"Cached raw simulation results to {cache_file}.")

    print(f"\nAggregating N={num_seeds} seed units...")

    # 1. Compute Seed-Aggregated Units (N=num_seeds)
    # AggReward(m, s) = mean over 6 scenarios for model m on seed s
    seed_agg_rewards = {m: [] for m in models}
    seed_agg_hits = {m: [] for m in models}
    seed_agg_switches = {m: [] for m in models}
    scenario_stats = {m: {sc: [] for sc in scenarios} for m in models}

    for m in models:
        for s in seeds:
            sc_rewards = [raw_results[m][s][sc]["reward"] for sc in scenarios]
            sc_hits = [raw_results[m][s][sc]["hits"] for sc in scenarios]
            sc_switches = [raw_results[m][s][sc]["switches"] for sc in scenarios]

            seed_agg_rewards[m].append(float(np.mean(sc_rewards)))
            seed_agg_hits[m].append(float(np.mean(sc_hits)))
            seed_agg_switches[m].append(float(np.mean(sc_switches)))

            for sc in scenarios:
                scenario_stats[m][sc].append(raw_results[m][s][sc]["reward"])

    champ_seed_scores = np.array(seed_agg_rewards[CHAMPION_MODEL_NAME])

    # 2. Pairwise Statistics vs Champion (N=30)
    raw_p_values_t = {}
    raw_p_values_w = {}
    t_stats = {}
    cohens_d = {}
    win_counts = {}
    loss_counts = {}
    tie_counts = {}
    win_rates = {}
    mean_diffs = {}
    median_diffs = {}

    for m in models:
        if m == CHAMPION_MODEL_NAME:
            continue
        m_seed_scores = np.array(seed_agg_rewards[m])
        diff = champ_seed_scores - m_seed_scores  # Positive = Dwell-Dual wins

        # Paired t-test
        t_stat, p_t = stats.ttest_rel(champ_seed_scores, m_seed_scores)
        t_stats[m] = float(t_stat)
        raw_p_values_t[m] = float(p_t)

        # Wilcoxon signed-rank
        try:
            w_res = stats.wilcoxon(diff, alternative="two-sided")
            raw_p_values_w[m] = float(w_res.pvalue)
        except Exception:
            raw_p_values_w[m] = 1.0

        # Cohen's d (paired)
        std_diff = float(np.std(diff, ddof=1))
        d_val = float(np.mean(diff) / std_diff) if std_diff > 1e-12 else 0.0
        cohens_d[m] = d_val
        mean_diffs[m] = float(np.mean(diff))
        median_diffs[m] = float(np.median(diff))

        # Seed Win Rate
        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))
        win_counts[m] = wins
        loss_counts[m] = losses
        tie_counts[m] = ties
        win_rates[m] = (wins / num_seeds) * 100.0

    # 3. Holm-Bonferroni Correction
    holm_p_t = holm_bonferroni(raw_p_values_t)
    holm_p_w = holm_bonferroni(raw_p_values_w)

    # 4. Compile Model Summaries & Rank by Aggregated Mean Reward
    model_summaries = {}
    for m in models:
        arr = seed_agg_rewards[m]
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1))
        ci_low, ci_high = compute_95ci(arr)
        hit_pct = float(np.mean(seed_agg_hits[m])) / (EPISODE_LENGTH) * 100.0

        model_summaries[m] = {
            "model_name": m,
            "category": MODEL_REGISTRY[m].get("category", "Baseline"),
            "mean_reward": round(mean_r, 2),
            "std_reward": round(std_r, 2),
            "reward_95ci": [round(ci_low, 2), round(ci_high, 2)],
            "mean_hits": round(float(np.mean(seed_agg_hits[m])), 1),
            "interception_rate_pct": round(hit_pct, 2),
            "mean_switches": round(float(np.mean(seed_agg_switches[m])), 1),
            "mean_latency_ms": round(float(np.mean(model_latencies[m])), 4),
            "scenario_means": {sc: round(float(np.mean(scenario_stats[m][sc])), 2) for sc in scenarios},
        }

    # Sort descending by mean aggregated reward
    ranked_models = sorted(model_summaries.items(), key=lambda x: x[1]["mean_reward"], reverse=True)
    for rank_idx, (m_name, d) in enumerate(ranked_models, 1):
        d["final_rank"] = rank_idx

    # Print Master N=30 Table
    print("\n" + "=" * 145)
    print(f"MASTER N={num_seeds} AGGREGATED BENCHMARK LEADERBOARD (1 SCORE PER SEED, DF={num_seeds-1})")
    print("=" * 145)
    print(f"{'Rank':<5} | {'Model Name':<30} | {'Mean Reward (95% CI)':<26} | {'Hit %':<7} | {'Win Rate vs DD':<16} | {'p (raw)':<10} | {'p (Holm)':<10} | {'Cohen d':<8} | {'Status'}")
    print("-" * 145)

    for m_name, d in ranked_models:
        rank = d["final_rank"]
        rew_ci = f"{d['mean_reward']:+6.2f} [{d['reward_95ci'][0]:+5.2f}, {d['reward_95ci'][1]:+5.2f}]"
        hit_pct = f"{d['interception_rate_pct']:4.1f}%"

        if m_name == CHAMPION_MODEL_NAME:
            win_str = "--- (Champion)"
            p_raw_str = "---"
            p_holm_str = "---"
            d_str = "---"
            status = "GRAND CHAMPION"
        else:
            wins = win_counts[m_name]
            wr = win_rates[m_name]
            win_str = f"{wins:2d}/{num_seeds} ({wr:4.1f}%)"
            p_raw = raw_p_values_t[m_name]
            p_holm = holm_p_t[m_name]
            d_val = cohens_d[m_name]

            p_raw_str = f"{p_raw:.4f}" if p_raw >= 0.0001 else f"{p_raw:.2e}"
            p_holm_str = f"{p_holm:.4f}" if p_holm >= 0.0001 else f"{p_holm:.2e}"
            d_str = f"{d_val:+5.2f}"

            if p_holm < 0.05:
                status = "Stat. Sub-optimal (p<0.05)"
            else:
                status = "Stat. Equivalent"

        print(f"#{rank:<4} | {m_name:<30} | {rew_ci:<26} | {hit_pct:<7} | {win_str:<16} | {p_raw_str:<10} | {p_holm_str:<10} | {d_str:<8} | {status}")
    print("=" * 145)

    # Compile JSON artifact
    artifact_data = {
        "metadata": {
            "num_seeds": num_seeds,
            "seeds": seeds,
            "scenarios": scenarios,
            "episode_length": EPISODE_LENGTH,
            "methodology": f"Scenario-aggregated scores per seed (N={num_seeds}, df={num_seeds-1}), Holm-Bonferroni correction, Seed Win Rate",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "leaderboard": {m: d for m, d in ranked_models},
        "pairwise_tests_vs_dwell_dual": {
            m: {
                "mean_difference": round(mean_diffs[m], 3),
                "median_difference": round(median_diffs[m], 3),
                "cohens_d": round(cohens_d[m], 3),
                "paired_t_stat": round(t_stats[m], 3),
                "p_value_t_raw": raw_p_values_t[m],
                "p_value_t_holm": holm_p_t[m],
                "p_value_wilcoxon_raw": raw_p_values_w[m],
                "p_value_wilcoxon_holm": holm_p_w[m],
                "dwell_dual_wins": win_counts[m],
                "dwell_dual_losses": loss_counts[m],
                "dwell_dual_ties": tie_counts[m],
                "dwell_dual_win_rate_pct": round(win_rates[m], 2),
                "significant_after_holm": bool(holm_p_t[m] < 0.05),
            }
            for m in models if m != CHAMPION_MODEL_NAME
        },
    }

    out_file = ROOT / "results" / "master_n30_aggregated_benchmark.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(artifact_data, f, indent=2)

    print(f"\nSaved complete N={num_seeds} benchmark with Holm correction and win rates to:\n  {out_file}")
    return artifact_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--recompute-champion", action="store_true", help="Re-evaluate champion episodes with updated logic")
    args = parser.parse_args()
    run_benchmark(num_seeds=args.num_seeds, workers=args.workers, recompute_champion=args.recompute_champion)
