"""Definitive Live Benchmark Across All 14 Registered Models.

Evaluates all 14 models from benchmark_models across all 6 EW scenarios
on held-out seeds, logging reward, hit rates, switches, latency, and scenario breakdowns.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.environment import SmartScanEnv
from benchmark_models import MODEL_REGISTRY, get_model

DEFAULT_SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
DEFAULT_SEEDS = [8001, 8002, 8003]
EPISODE_LENGTH = 150


def run_benchmark(
    models_to_run: list[str] | None = None,
    seeds: list[int] | None = None,
    scenarios: list[str] | None = None,
):
    scenarios = scenarios or DEFAULT_SCENARIOS
    seeds = seeds or DEFAULT_SEEDS

    # Filter registry if specific models requested
    target_registry = {}
    if models_to_run:
        for m in models_to_run:
            matched = False
            for k, v in MODEL_REGISTRY.items():
                if m.lower() in k.lower():
                    target_registry[k] = v
                    matched = True
            if not matched:
                print(f"Warning: No registered model matches '{m}'")
    if not target_registry:
        target_registry = MODEL_REGISTRY

    print("=" * 110)
    print(f"  DEFINITIVE BENCHMARK: {len(target_registry)} CANDIDATES & BASELINES ACROSS {len(scenarios)} EW SCENARIOS")
    print(f"  Scenarios ({len(scenarios)}): {scenarios}")
    print(f"  Seeds ({len(seeds)}): {seeds} | Steps per episode: {EPISODE_LENGTH}")
    total_episodes = len(target_registry) * len(scenarios) * len(seeds)
    print(f"  Total Evaluations: {total_episodes} episodes ({total_episodes * EPISODE_LENGTH:,} total decisions)")
    print("=" * 110)

    summary_results = {}

    for model_idx, (model_name, meta) in enumerate(target_registry.items(), 1):
        print(f"[{model_idx:2d}/{len(target_registry):2d}] Evaluating: {model_name:<35} ... ", end="", flush=True)
        t_start = time.perf_counter()

        ep_rewards = []
        ep_hits = []
        ep_switches = []
        ep_signals_found = []
        ep_total_signals = []
        all_latencies = []
        sc_breakdown = {sc: [] for sc in scenarios}

        for sc in scenarios:
            for seed in seeds:
                env = SmartScanEnv(num_bands=20, episode_length=EPISODE_LENGTH, seed=seed, scenario=sc)
                scheduler = get_model(model_name, num_bands=20, seed=seed)

                obs, info = env.reset(seed=seed)
                total_reward = 0.0
                hits = 0
                switches = 0
                signals_found = 0
                total_signals = 0
                last_band = None

                for step_idx in range(EPISODE_LENGTH):
                    t0 = time.perf_counter()
                    action = int(scheduler.select_band())
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    all_latencies.append(lat_ms)

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
                ep_rewards.append(total_reward)
                ep_hits.append(hits)
                ep_switches.append(switches)
                ep_signals_found.append(signals_found)
                ep_total_signals.append(total_signals)
                sc_breakdown[sc].append(total_reward)

        elapsed = time.perf_counter() - t_start
        mean_reward = float(np.mean(ep_rewards))
        std_reward = float(np.std(ep_rewards))
        mean_hits = float(np.mean(ep_hits))
        mean_switches = float(np.mean(ep_switches))
        mean_lat = float(np.mean(all_latencies))
        p95_lat = float(np.percentile(all_latencies, 95))

        tot_found = sum(ep_signals_found)
        tot_emitted = sum(ep_total_signals)
        interception_rate = (100.0 * tot_found / max(1, tot_emitted))

        print(f"Reward: {mean_reward:+6.2f} | Intercept: {interception_rate:5.1f}% | Latency: {mean_lat:6.3f}ms ({elapsed:4.1f}s)")

        summary_results[model_name] = {
            "rank_original": meta.get("rank", 99),
            "category": meta.get("category", "General"),
            "description": meta.get("description", ""),
            "mean_reward": round(mean_reward, 2),
            "std_reward": round(std_reward, 2),
            "mean_hits": round(mean_hits, 1),
            "interception_rate_pct": round(interception_rate, 1),
            "signals_found": tot_found,
            "total_signals": tot_emitted,
            "mean_switches": round(mean_switches, 1),
            "latency_mean_ms": round(mean_lat, 4),
            "latency_p95_ms": round(p95_lat, 4),
            "scenario_means": {sc: round(float(np.mean(sc_breakdown[sc])), 2) for sc in scenarios},
        }

    # Final Rank by Mean Reward
    ranked = sorted(summary_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True)
    for rank, (name, d) in enumerate(ranked, 1):
        d["final_rank"] = rank

    # Save to file
    out_path = ROOT / "results" / "definitive_all_models_benchmark.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 125)
    print(f"{'FINAL LEADERBOARD RANKINGS':^125}")
    print("=" * 125)
    header = f"{'Rank':<5} | {'Model Name':<32} | {'Reward':<15} | {'Intercept %':<12} | {'Hits/Ep':<9} | {'Switches':<9} | {'Latency':<11} | {'Category'}"
    print(header)
    print("-" * 125)

    for rank, (name, d) in enumerate(ranked, 1):
        medal = f"[{rank:2d}]"
        rew_str = f"{d['mean_reward']:+6.2f} ±{d['std_reward']:4.2f}"
        int_str = f"{d['interception_rate_pct']:5.1f}%"
        lat_str = f"{d['latency_mean_ms']:6.3f} ms"
        print(f"{medal:<5} | {name:<32} | {rew_str:<15} | {int_str:<12} | {d['mean_hits']:^9.1f} | {d['mean_switches']:^9.1f} | {lat_str:<11} | {d['category']}")

    print("=" * 125)
    print(f"\nDetailed results saved to: {out_path}")
    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmartScan Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick run with 1 seed across scenarios")
    parser.add_argument("--seeds", nargs="+", type=int, help="Custom seeds (e.g. --seeds 10001 10002)")
    parser.add_argument("--models", nargs="+", type=str, help="Specific model names (e.g. --models Dwell NMF)")
    args = parser.parse_args()

    custom_seeds = [8001] if args.quick else (args.seeds or DEFAULT_SEEDS)
    run_benchmark(models_to_run=args.models, seeds=custom_seeds)

