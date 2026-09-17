"""Paired multi-seed validation for the Turing Dwell-Dual candidate."""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import h5py
import numpy as np
from scripts.experiment_turing_dwell_variants import Mission, DummyRuntime
from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler


def main():
    files = sorted(Path("turing_dataset").rglob("*.h5"))
    missions = [(str(path.relative_to("turing_dataset")), Mission(path)) for path in files]
    seeds = [7, 19, 31, 43, 59, 71, 83, 97, 109, 127]
    configs = {
        "old_dwell_dual": {"dwell_inertia": 1.50, "explore_budget_prob": 0.08, "nmf_window": 30, "switch_penalty": 0.04, "fading_grace_steps": 1},
        "higher_dwell": {"dwell_inertia": 2.40, "explore_budget_prob": 0.08, "nmf_window": 30, "switch_penalty": 0.04, "fading_grace_steps": 1},
        "stable_emitter": {"dwell_inertia": 2.20, "explore_budget_prob": 0.04, "nmf_window": 45, "switch_penalty": 0.04, "fading_grace_steps": 2},
    }
    results = {name: [] for name in configs}
    for seed in seeds:
        for mission_name, mission in missions:
            for name, config in configs.items():
                scheduler = DwellDualPolicyScheduler(num_bands=20, seed=seed, runtime=DummyRuntime(), **config)
                scheduler_result = mission.run(lambda s=scheduler: s)
                results[name].append(scheduler_result["reward"])
    base = np.asarray(results["old_dwell_dual"])
    print("TURING MULTI-SEED PAIRED VALIDATION")
    print(f"missions={len(missions)} seeds={len(seeds)} paired_runs={len(base)}")
    for name, values in results.items():
        values = np.asarray(values)
        delta = values - base
        print(f"{name:<18} mean={values.mean():.3f} std={values.std(ddof=1):.3f} delta_vs_old={delta.mean():+.3f} wins={int((delta > 1e-9).sum())} losses={int((delta < -1e-9).sum())} ties={int((abs(delta) <= 1e-9).sum())}")
    print("Per-mission old/candidate means:")
    for index, (name, _) in enumerate(missions):
        start = index * len(seeds)
        stop = start + len(seeds)
        print(f"{name:<28} old={base[start:stop].mean():>7.2f} higher={np.asarray(results['higher_dwell'])[start:stop].mean():>7.2f}")


if __name__ == "__main__":
    main()
