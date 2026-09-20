"""Evaluate all official HF stare/test_stare missions with model-only algorithms."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from huggingface_hub import list_repo_files, hf_hub_download
from model_only_benchmark import PulseBenchmark
from scheduler.baselines import FixedScheduler, RandomScheduler, ThompsonSamplingScheduler, UCB1Scheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.smartscan_production import SmartScanProductionScheduler

REPO = "alan-turing-institute/turing-synthetic-radar-dataset"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("hf_stare_full_results.json"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 means all 250 missions")
    args = parser.parse_args()
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("stare/test_stare/") and f.endswith(".h5"))
    if args.limit:
        files = files[:args.limit]
    names = ["Sequential Sweep", "Random", "UCB1", "Thompson", "NMF-only", "Smart Scan"]
    configs = {
        "Sequential Sweep": lambda seed: FixedScheduler(args.bands),
        "Random": lambda seed: RandomScheduler(args.bands, seed=seed),
        "UCB1": lambda seed: UCB1Scheduler(args.bands),
        "Thompson": lambda seed: ThompsonSamplingScheduler(args.bands, seed=seed),
        "NMF-only": lambda seed: NMFScheduler(args.bands, recompute_every=2),
        "Smart Scan": lambda seed: SmartScanProductionScheduler(args.bands, seed=seed),
    }
    all_rows = []
    print(f"OFFICIAL HF STARE TEST BENCHMARK | missions={len(files)} | steps={args.steps} | seed={args.seed}")
    print("No mission selection: every stare/test_stare/*.h5 file is evaluated.")
    for index, remote in enumerate(files, 1):
        local = Path(hf_hub_download(repo_id=REPO, filename=remote, repo_type="dataset", local_dir="hf_stare_full"))
        benchmark = PulseBenchmark(local, args.bands, args.steps)
        row = {"file": remote}
        for name in names:
            result = benchmark.run(configs[name], args.seed)
            row[name] = result
        all_rows.append(row)
        print(f"{index:03d}/{len(files)} {remote} | " + " | ".join(f"{name}={row[name]['interception_rate']:.2f}%" for name in names), flush=True)
    summary = {}
    for name in names:
        values = [row[name] for row in all_rows]
        summary[name] = {
            "mean_reward": float(np.mean([v["reward"] for v in values])),
            "mean_interception_rate": float(np.mean([v["interception_rate"] for v in values])),
            "mean_switches": float(np.mean([v["switches"] for v in values])),
            "mean_latency_ms": float(np.mean([v["mean_latency_ms"] for v in values])),
            "p95_latency_ms": float(np.percentile([v["mean_latency_ms"] for v in values], 95)),
            "missions_won_by_reward": int(sum(max(names, key=lambda n: row[n]["reward"]) == name for row in all_rows)),
            "missions_won_by_interception": int(sum(max(names, key=lambda n: row[n]["interception_rate"]) == name for row in all_rows)),
        }
    result = {"dataset": REPO, "split": "stare/test_stare", "missions": len(files), "steps": args.steps, "seed": args.seed, "summary": summary, "per_mission": all_rows}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nFINAL MEAN LEADERBOARD ACROSS ALL OFFICIAL STARE TEST MISSIONS")
    print(f"{'Algorithm':<22} {'Mean reward':>13} {'Mean intercept %':>18} {'Mean switches':>15} {'Reward wins':>12} {'Intercept wins':>16}")
    for name in sorted(names, key=lambda n: summary[n]["mean_reward"], reverse=True):
        s = summary[name]
        print(f"{name:<22} {s['mean_reward']:>13.2f} {s['mean_interception_rate']:>17.2f}% {s['mean_switches']:>15.1f} {s['missions_won_by_reward']:>12d} {s['missions_won_by_interception']:>16d}")
    print(f"\nSaved complete results to {args.output}")

if __name__ == "__main__":
    main()
