"""Rapid Upgrade Screening Benchmark.

Compares:
1. Dwell-Dual Policy (Champion Baseline)
2. Dwell-Dual + Expectimax (Hierarchical Top-K Depth-2 Lookahead)
3. Dwell-Dual + Thompson (Discounted Beta-Bernoulli Posterior Sampling)
4. Dwell-Dual + Whittle RMAB (Restless Multi-Armed Bandit with Age of Information)

Across 6 Scenarios and 5 Seeds (120 total episodes).
Computes Reward, Hit Rate, Switches, Decision Latency, and Paired Statistical Significance.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
sys.path.insert(0, str(ROOT))

from simulator.environment import SmartScanEnv
from benchmark_models import get_model

MODELS_TO_SCREEN = [
    "Dwell-Dual Policy (Champion)",
    "Dwell-Dual + Expectimax",
    "Dwell-Dual + Thompson",
    "Dwell-Dual + Whittle RMAB",
]

SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
SEEDS = [10001, 10002, 10003, 10004, 10005]
EPISODE_LENGTH = 150


def run_screening():
    print("=" * 115)
    print("  SMARTSCAN RAPID UPGRADE SCREENING BENCHMARK")
    print(f"  Models ({len(MODELS_TO_SCREEN)})   : {MODELS_TO_SCREEN}")
    print(f"  Scenarios ({len(SCENARIOS)}): {SCENARIOS}")
    print(f"  Seeds ({len(SEEDS)})        : {SEEDS} (Total {len(MODELS_TO_SCREEN) * len(SCENARIOS) * len(SEEDS)} episodes)")
    print(f"  Episode Length : {EPISODE_LENGTH} steps")
    print("=" * 115)

    raw_results = {m: {sc: [] for sc in SCENARIOS} for m in MODELS_TO_SCREEN}
    model_stats = {m: {"rewards": [], "hits": [], "total_opps": [], "switches": [], "latencies": []} for m in MODELS_TO_SCREEN}

    t_start = time.perf_counter()

    for sc in SCENARIOS:
        print(f"\nEvaluating Scenario: {sc:<12} ...", end="", flush=True)
        t_sc = time.perf_counter()
        for seed in SEEDS:
            for m_name in MODELS_TO_SCREEN:
                env = SmartScanEnv(num_bands=20, episode_length=EPISODE_LENGTH, seed=seed, scenario=sc)
                scheduler = get_model(m_name, num_bands=20, seed=seed)

                obs, info = env.reset(seed=seed)
                tot_rew = 0.0
                hits = 0
                switches = 0
                opps = 0
                lats = []
                last_band = None

                for _ in range(EPISODE_LENGTH):
                    t0 = time.perf_counter()
                    action = int(scheduler.select_band())
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    lats.append(dt_ms)

                    if last_band is not None and action != last_band:
                        switches += 1
                    last_band = action

                    obs, reward, done, truncated, info = env.step(action)
                    scheduler.update(action, reward, obs)

                    tot_rew += reward
                    signal_present = bool(info.get("true_signal_present", False))
                    detected = bool(obs.get("detected", False))
                    if signal_present:
                        opps += 1
                    if signal_present and detected:
                        hits += 1

                    if done or truncated:
                        break

                raw_results[m_name][sc].append({
                    "reward": tot_rew,
                    "hits": hits,
                    "opps": opps,
                    "switches": switches,
                    "median_lat": float(np.median(lats)),
                    "p95_lat": float(np.percentile(lats, 95)),
                })

                model_stats[m_name]["rewards"].append(tot_rew)
                model_stats[m_name]["hits"].append(hits)
                model_stats[m_name]["total_opps"].append(opps)
                model_stats[m_name]["switches"].append(switches)
                model_stats[m_name]["latencies"].extend(lats)

        print(f" Done ({time.perf_counter() - t_sc:.1f}s)")

    print("\n" + "=" * 115)
    print("                      RAPID SCREENING BENCHMARK: OVERALL LEADERBOARD                        ")
    print("=" * 115)
    print(f"{'Model Architecture':<32} | {'Reward (Mean +/- Std)':<22} | {'Hit Rate %':<11} | {'Switches':<10} | {'Latency (Med / P95)':<20} | {'Delta vs Base':<13}")
    print("-" * 115)

    base_rewards = np.array(model_stats["Dwell-Dual Policy (Champion)"]["rewards"])

    summary_records = []

    for m_name in MODELS_TO_SCREEN:
        rews = np.array(model_stats[m_name]["rewards"])
        mean_r = float(np.mean(rews))
        std_r = float(np.std(rews, ddof=1))
        tot_hits = sum(model_stats[m_name]["hits"])
        tot_opps = sum(model_stats[m_name]["total_opps"])
        hit_pct = (tot_hits / max(1, tot_opps)) * 100.0
        avg_sw = float(np.mean(model_stats[m_name]["switches"]))
        med_lat = float(np.median(model_stats[m_name]["latencies"]))
        p95_lat = float(np.percentile(model_stats[m_name]["latencies"], 95))

        delta_str = "---"
        p_val_str = ""
        delta_val = mean_r - float(np.mean(base_rewards))
        if m_name != "Dwell-Dual Policy (Champion)":
            # Paired t-test
            diff = rews - base_rewards
            t_stat, p_val = stats.ttest_rel(rews, base_rewards)
            delta_str = f"{delta_val:+.2f}"
            p_val_str = f" (p={p_val:.3f})"

        print(f"{m_name:<32} | {mean_r:+6.2f} +/- {std_r:5.2f}          | {hit_pct:5.1f}%     | {avg_sw:5.1f}      | {med_lat:5.3f}ms / {p95_lat:5.3f}ms | {delta_str:<6}{p_val_str}")

        summary_records.append({
            "model": m_name,
            "mean_reward": round(mean_r, 2),
            "std_reward": round(std_r, 2),
            "hit_rate_pct": round(hit_pct, 2),
            "avg_switches": round(avg_sw, 1),
            "median_lat_ms": round(med_lat, 4),
            "p95_lat_ms": round(p95_lat, 4),
            "delta_vs_baseline": round(delta_val, 2),
        })

    print("=" * 115)

    print("\n" + "=" * 115)
    print("                   SCENARIO-BY-SCENARIO DETAILED BREAKDOWN (MEAN REWARD)                    ")
    print("=" * 115)
    header = f"{'Scenario':<16} | " + " | ".join([f"{m[:20]:<20}" for m in MODELS_TO_SCREEN])
    print(header)
    print("-" * 115)

    for sc in SCENARIOS:
        row_str = f"{sc:<16} | "
        vals = []
        for m_name in MODELS_TO_SCREEN:
            sc_rews = [r["reward"] for r in raw_results[m_name][sc]]
            vals.append(f"{np.mean(sc_rews):+6.2f} +/- {np.std(sc_rews, ddof=1):4.1f}")
        print(row_str + " | ".join([f"{v:<20}" for v in vals]))

    print("=" * 115)
    print(f"Total screening time: {time.perf_counter() - t_start:.2f}s")

    # Save results to json
    out_file = ROOT / "results" / "rapid_upgrade_screening_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_records,
            "raw": raw_results,
        }, f, indent=2)
    print(f"\nScreening results saved to:\n  {out_file}")


if __name__ == "__main__":
    run_screening()
