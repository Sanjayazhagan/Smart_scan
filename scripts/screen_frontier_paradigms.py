"""Screening benchmark for Frontier Paradigms vs Grand Champion.

Models compared:
1. Dwell-Dual Policy (Champion)
2. Dwell-Dual + Conformal Sets
3. Dwell-Dual + Hawkes Process
4. Dwell-Dual + PRI Tracker

Across 6 scenarios x 5 seeds (120 episodes total).
Computes reward, hit rate, switches, decision latency, and paired delta vs Champion.
"""
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
sys.path.insert(0, str(ROOT))

from simulator.environment import SmartScanEnv
from benchmark_models import get_model
import scheduler.stats_compat as stats

MODELS = [
    "Dwell-Dual Policy (Champion)",
    "Dwell-Dual + Conformal Sets",
    "Dwell-Dual + Hawkes Process",
    "Dwell-Dual + PRI Tracker",
]
SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
SEEDS = [10001, 10002, 10003, 10004, 10005]
EPISODE_LENGTH = 150


def run_single(args):
    m_name, sc, seed = args
    env = SmartScanEnv(num_bands=20, episode_length=EPISODE_LENGTH, seed=seed, scenario=sc)
    sched = get_model(m_name, num_bands=20, seed=seed)
    obs, info = env.reset(seed=seed)
    tot_rew = 0.0
    hits = 0
    switches = 0
    latencies = []
    last_band = None

    for _ in range(EPISODE_LENGTH):
        t0 = time.perf_counter()
        action = int(sched.select_band())
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        if last_band is not None and action != last_band:
            switches += 1
        last_band = action

        obs, reward, terminated, truncated, info = env.step(action)
        sched.update(action, reward, obs)
        tot_rew += reward
        if obs.get("detected", False):
            hits += 1
        if terminated or truncated:
            break

    env.close()
    return {
        "model": m_name,
        "scenario": sc,
        "seed": seed,
        "reward": float(tot_rew),
        "hits": int(hits),
        "switches": int(switches),
        "mean_latency_ms": float(np.mean(latencies)),
    }


def main():
    tasks = [(m, sc, s) for m in MODELS for sc in SCENARIOS for s in SEEDS]
    print("=" * 110)
    print("FRONTIER PARADIGMS SCREENING BENCHMARK")
    print(f"Models: {len(MODELS)} | Scenarios: {len(SCENARIOS)} | Seeds: {len(SEEDS)} | Total Episodes: {len(tasks)}")
    print("=" * 110)

    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        for r in ex.map(run_single, tasks):
            results.append(r)
    elapsed = time.time() - t0
    print(f"Completed {len(tasks)} episodes in {elapsed:.2f}s\n")

    # Aggregate by model and seed
    # score(m, s) = mean reward over 6 scenarios for seed s
    seed_scores = {m: {s: [] for s in SEEDS} for m in MODELS}
    seed_hits = {m: {s: [] for s in SEEDS} for m in MODELS}
    seed_switches = {m: {s: [] for s in SEEDS} for m in MODELS}
    model_latencies = {m: [] for m in MODELS}
    scenario_scores = {m: {sc: [] for sc in SCENARIOS} for m in MODELS}

    for r in results:
        m = r["model"]
        s = r["seed"]
        sc = r["scenario"]
        seed_scores[m][s].append(r["reward"])
        seed_hits[m][s].append(r["hits"])
        seed_switches[m][s].append(r["switches"])
        scenario_scores[m][sc].append(r["reward"])
        model_latencies[m].append(r["mean_latency_ms"])

    champ_name = "Dwell-Dual Policy (Champion)"
    champ_seed_means = np.array([np.mean(seed_scores[champ_name][s]) for s in SEEDS])

    print("=" * 125)
    print(f"{'Rank':<5} | {'Model Name':<30} | {'Mean Reward (95% CI)':<24} | {'Hit %':<7} | {'Switches':<9} | {'Latency':<9} | {'vs Champion':<22}")
    print("-" * 125)

    summaries = []
    for m in MODELS:
        m_seed_means = np.array([np.mean(seed_scores[m][s]) for s in SEEDS])
        mean_r = float(np.mean(m_seed_means))
        ci = stats.ci95(m_seed_means)
        all_hits = [np.mean(seed_hits[m][s]) for s in SEEDS]
        hit_pct = float(np.mean(all_hits)) / EPISODE_LENGTH * 100.0
        all_sw = [np.mean(seed_switches[m][s]) for s in SEEDS]
        mean_sw = float(np.mean(all_sw))
        mean_lat = float(np.mean(model_latencies[m]))

        if m == champ_name:
            vs_str = "--- (Champion)"
        else:
            diff = m_seed_means - champ_seed_means
            delta = float(np.mean(diff))
            t_res = stats.ttest_rel(m_seed_means, champ_seed_means)
            p_val = t_res.pvalue
            wins = int(np.sum(diff > 0))
            vs_str = f"Delta:{delta:+.2f} (p={p_val:.3f}, {wins}/5)"

        summaries.append({
            "name": m,
            "mean_reward": mean_r,
            "ci": ci,
            "hit_pct": hit_pct,
            "switches": mean_sw,
            "latency_ms": mean_lat,
            "vs_str": vs_str,
            "scenarios": {sc: float(np.mean(scenario_scores[m][sc])) for sc in SCENARIOS},
        })

    summaries.sort(key=lambda x: x["mean_reward"], reverse=True)
    for idx, s in enumerate(summaries, 1):
        ci_str = f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"
        rew_str = f"{s['mean_reward']:+.2f} {ci_str}"
        print(f"#{idx:<4} | {s['name']:<30} | {rew_str:<24} | {s['hit_pct']:5.1f}% | {s['switches']:7.1f} | {s['latency_ms']:6.3f}ms | {s['vs_str']:<22}")
    print("=" * 125)

    print("\nScenario Breakdown (Mean Cumulative Reward):")
    header = f"{'Model':<30} | " + " | ".join(f"{sc:<11}" for sc in SCENARIOS)
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = f"{s['name']:<30} | " + " | ".join(f"{s['scenarios'][sc]:>11.2f}" for sc in SCENARIOS)
        print(row)
    print("=" * len(header))

    out_file = ROOT / "results" / "frontier_paradigms_screening.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved screening results to: {out_file}")


if __name__ == "__main__":
    main()
