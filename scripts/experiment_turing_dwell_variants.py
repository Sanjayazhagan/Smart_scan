"""Tune Dwell-Dual behavior on HF Turing missions, then validate on held-out missions."""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import json
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
from benchmark_models.mathematical_baselines import NMFScheduler


class DummyRuntime:
    def update(self, *args, **kwargs):
        return None
    def reset(self):
        return None
    def get_forecast(self, *args, **kwargs):
        return np.zeros(20)


class Mission:
    def __init__(self, path: Path, steps: int = 300):
        self.steps = steps
        with h5py.File(path, "r") as f:
            data = f["data"][:]
        times, rfs, powers = data[:, 0], data[:, 1], data[:, 4]
        end = times.min() + (times.max() - times.min()) * 0.35
        keep = (times >= times.min()) & (times <= end)
        times, rfs, powers = times[keep], rfs[keep], powers[keep]
        edges = np.linspace(float(rfs.min()), float(rfs.max()) + 1e-3, 21)
        bands = np.clip(np.digitize(rfs, edges) - 1, 0, 19)
        slots = np.clip(np.digitize(times, np.linspace(float(times.min()), float(times.max()) + 1e-3, steps + 1)) - 1, 0, steps - 1)
        self.active = np.zeros((steps, 20), dtype=bool)
        self.power = np.full((steps, 20), -150.0, dtype=np.float32)
        for slot, band, power in zip(slots, bands, powers):
            self.active[slot, band] = True
            self.power[slot, band] = max(self.power[slot, band], power)

    def run(self, factory):
        scheduler = factory()
        last = None
        reward = 0.0
        hits = 0
        switches = 0
        for step in range(self.steps):
            band = scheduler.select_band()
            active = bool(self.active[step, band])
            power = float(self.power[step, band])
            detected = bool(active and power > -125.0)
            quality = float(np.clip((power + 120.0) / 40.0, 0.0, 1.0)) if detected else 0.0
            obs = {
                "selected_band": band,
                "detected": detected,
                "signal_power": np.array([max(0.0, power) if detected else 0.0], dtype=np.float32),
                "quality": np.array([quality], dtype=np.float32),
            }
            step_reward = 1.0 if active and detected else 0.1 if not active and not detected else -1.0
            if last is not None and last != band:
                step_reward -= 0.08 * abs(band - last) / 19.0
                switches += 1
            scheduler.update(band, step_reward, obs)
            reward += step_reward
            hits += int(active and detected)
            last = band
        return {"reward": reward, "hits": hits, "switches": switches}


def main():
    files = sorted(Path("turing_dataset").rglob("*.h5"))
    missions = [(str(path.relative_to("turing_dataset")), Mission(path)) for path in files]
    development = missions[:10]
    heldout = missions[10:]

    base = {
        "nmf_scale": 1.20,
        "nmf_window": 30,
        "nmf_recompute_every": 2,
        "switch_penalty": 0.04,
        "dwell_inertia": 1.50,
        "explore_budget_prob": 0.08,
        "uncertainty_trigger_threshold": 0.35,
        "fading_grace_steps": 1,
    }
    variants = {"baseline": base}
    for name, updates in {
        "lower_dwell": {"dwell_inertia": 0.60},
        "higher_dwell": {"dwell_inertia": 2.40},
        "more_explore": {"explore_budget_prob": 0.20},
        "less_explore": {"explore_budget_prob": 0.03},
        "lower_explore_threshold": {"uncertainty_trigger_threshold": 0.25},
        "higher_explore_threshold": {"uncertainty_trigger_threshold": 0.50},
        "lower_switch_penalty": {"switch_penalty": 0.01},
        "higher_switch_penalty": {"switch_penalty": 0.10},
        "more_nmf": {"nmf_scale": 1.80},
        "less_nmf": {"nmf_scale": 0.60},
        "short_memory": {"nmf_window": 12},
        "long_memory": {"nmf_window": 60},
        "fast_nmf": {"nmf_recompute_every": 1},
        "slow_nmf": {"nmf_recompute_every": 5},
        "no_fading_grace": {"fading_grace_steps": 0},
        "two_step_grace": {"fading_grace_steps": 2},
        "balanced_fast": {"dwell_inertia": 0.90, "explore_budget_prob": 0.16, "nmf_window": 20, "switch_penalty": 0.03},
        "agile_hopper": {"dwell_inertia": 0.55, "explore_budget_prob": 0.24, "uncertainty_trigger_threshold": 0.25, "nmf_window": 12, "fading_grace_steps": 0},
        "stable_emitter": {"dwell_inertia": 2.20, "explore_budget_prob": 0.04, "uncertainty_trigger_threshold": 0.45, "nmf_window": 45, "fading_grace_steps": 2},
    }.items():
        config = dict(base)
        config.update(updates)
        variants[name] = config

    def factory(config):
        return lambda: DwellDualPolicyScheduler(num_bands=20, seed=42, runtime=DummyRuntime(), **config)

    rows = []
    for name, config in variants.items():
        dev = [mission.run(factory(config)) for _, mission in development]
        held = [mission.run(factory(config)) for _, mission in heldout]
        rows.append({
            "name": name,
            "config": config,
            "development_mean_reward": float(np.mean([x["reward"] for x in dev])),
            "development_total_hits": int(np.sum([x["hits"] for x in dev])),
            "heldout_mean_reward": float(np.mean([x["reward"] for x in held])),
            "heldout_total_hits": int(np.sum([x["hits"] for x in held])),
            "heldout_mean_switches": float(np.mean([x["switches"] for x in held])),
        })
    rows.sort(key=lambda row: row["development_mean_reward"], reverse=True)
    print("TURING DWELL-DUAL VARIANT STUDY")
    print("Development missions: first 10 files; held-out missions: last 10 files")
    print("rank | name                 | dev reward | held reward | held hits | held switches")
    for rank, row in enumerate(rows, 1):
        print(f"{rank:4d} | {row['name']:<20} | {row['development_mean_reward']:>10.2f} | {row['heldout_mean_reward']:>11.2f} | {row['heldout_total_hits']:>9d} | {row['heldout_mean_switches']:>13.1f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/turing_dwell_variant_study.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
