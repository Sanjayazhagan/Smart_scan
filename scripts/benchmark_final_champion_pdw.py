"""Paired PDW-only benchmark: interruptible Dual-Dwell baseline vs final champion.

Final champion = interruptible recurring coverage + smart stale-band ordering.

This script is intended for reproducible validation on either:
- the included generic Turing-style synthetic stare HDF5 fixtures, or
- compatible real Turing stare HDF5 files supplied by the user.

No I/Q is generated or consumed by this benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scheduler.stats_compat import ci95, ttest_rel

from scheduler.dwell_dual_lightweight_variants import InterruptibleDwellChampion
from scheduler.smartscan_production import SmartScanProductionScheduler
from simulator.turing_pdw_environment import TuringPDWEnv, TuringPulseTrain

MODES = ("B_interruptible", "FINAL_smart_stale")


def discover_h5(data_root: Path) -> list[Path]:
    files = sorted(data_root.rglob("*.h5"))
    stare = [p for p in files if "stare" in str(p).lower()]
    files = stare or files
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {data_root}")
    return files


def build_scheduler(mode: str, seed: int):
    if mode == "B_interruptible":
        return InterruptibleDwellChampion(num_bands=20, seed=seed)
    if mode == "FINAL_smart_stale":
        return SmartScanProductionScheduler(num_bands=20, seed=seed)
    raise KeyError(mode)


def run_episode(source: TuringPulseTrain, seed: int, mode: str, episode_length: int, step_us: float):
    env = TuringPDWEnv(source, episode_length=episode_length, step_us=step_us)
    sched = build_scheduler(mode, seed)
    env.reset(seed=seed)

    reward = hit_steps = active_steps = captured = total_pulses = switches = 0
    select_ms = 0.0
    last_band = None

    for _ in range(episode_length):
        t0 = perf_counter()
        band = int(sched.select_band())
        select_ms += (perf_counter() - t0) * 1000.0

        if last_band is not None and band != last_band:
            switches += 1
        last_band = band

        obs, r, _, truncated, info = env.step(band)
        if "iq" in obs:
            raise AssertionError("I/Q unexpectedly present in PDW benchmark")
        sched.update(band, r, obs)

        reward += float(r)
        hit_steps += int(bool(obs["detected"]))
        active_steps += int(bool(info["ground_truth_active_bands"]))
        captured += int(info["captured_pulses"])
        total_pulses += int(info["ground_truth_pulses_this_step"])
        if truncated:
            break

    return {
        "reward": reward,
        "hit_steps": hit_steps,
        "active_steps": active_steps,
        "captured_pulses": captured,
        "total_pulses": total_pulses,
        "switches": switches,
        "dwell_decisions": int(sched.dwell_decisions),
        "mean_select_ms": select_ms / episode_length,
    }


def worker(args):
    path_str, seed, episode_length, step_us = args
    source = TuringPulseTrain(Path(path_str))
    return {
        mode: run_episode(source, seed, mode, episode_length, step_us)
        for mode in MODES
    }




def summarize(rows):
    rewards = np.asarray([r["reward"] for r in rows], dtype=float)
    return {
        "episodes": len(rows),
        "mean_reward": float(rewards.mean()),
        "reward_ci95": ci95(rewards),
        "hit_step_rate_pct": 100.0 * sum(r["hit_steps"] for r in rows) / max(1, sum(r["active_steps"] for r in rows)),
        "pulse_capture_rate_pct": 100.0 * sum(r["captured_pulses"] for r in rows) / max(1, sum(r["total_pulses"] for r in rows)),
        "mean_switches": float(np.mean([r["switches"] for r in rows])),
        "mean_dwell_decisions": float(np.mean([r["dwell_decisions"] for r in rows])),
        "mean_select_ms": float(np.mean([r["mean_select_ms"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "tests" / "synthetic_turing_style")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--episode-length", type=int, default=150)
    ap.add_argument("--step-us", type=float, default=1000.0)
    ap.add_argument("--seed-offset", type=int, default=1_500_000)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "final_champion" / "final_champion_fresh_run.json")
    args = ap.parse_args()

    files = discover_h5(args.data_root)
    tasks = []
    for fi, p in enumerate(files):
        for si in range(args.seeds):
            seed = 10001 + args.seed_offset + si + fi * 100_000
            tasks.append((str(p), seed, args.episode_length, args.step_us))

    rows = {m: [] for m in MODES}
    workers = min(16, max(1, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, task) for task in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            for mode in MODES:
                rows[mode].append(result[mode])

    base = np.asarray([r["reward"] for r in rows["B_interruptible"]], dtype=float)
    final = np.asarray([r["reward"] for r in rows["FINAL_smart_stale"]], dtype=float)
    delta = final - base
    test = ttest_rel(final, base)

    out = {
        "dataset": str(args.data_root),
        "uses_iq": False,
        "seeds_per_file": args.seeds,
        "num_files": len(files),
        "paired_sets": len(tasks),
        "scheduler_episodes": len(tasks) * 2,
        "total_decisions": len(tasks) * 2 * args.episode_length,
        "summary": {m: summarize(rows[m]) for m in MODES},
        "paired_final_minus_B": {
            "mean_delta_reward": float(delta.mean()),
            "delta_ci95": ci95(delta),
            "paired_p": float(test.pvalue),
            "better_episode_pct": 100.0 * float(np.mean(delta > 1e-12)),
            "worse_episode_pct": 100.0 * float(np.mean(delta < -1e-12)),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
