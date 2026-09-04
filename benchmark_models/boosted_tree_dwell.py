"""Model 06: Boosted-Tree Adaptive Dwell Scheduler (Compact GBDT).

========================================================================================
PROJECT STATUS: CANDIDATE 2 EVALUATED (+11.66 / OVERLY CONSERVATIVE PERSISTENCE)
========================================================================================
- Mean Cumulative Reward : +11.66 +/- 3.39 (Fresh 10-Seed Benchmark)
- Decision Latency       : 0.07 ms mean (Fast pure-NumPy tree evaluator)
- Switching Count        : 31.4 switches per episode (Ultra-conservative)
- Statistical Finding    : Statistically significantly worse than Champion (p = 0.0002)

THEORY & ARCHITECTURE:
Replaces heuristic dwell inertia and fixed fading grace steps with an empirical
machine-learning classifier:

1. Compact GBDT Persistence Model:
   - 25 regression decision trees of max_depth=3 trained on real SmartScanEnv transitions.
   - Fits negative logistic gradients: g_i = p_i - y_i, h_i = p_i * (1 - p_i).
   - Minimizes regularized split objective:
       Gain = 0.5 * [ G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G^2/(H + lambda) ]

2. Context Features (7-Dimensional Vector):
   - consecutive_hits, consecutive_dwell, consecutive_misses, last_quality,
     dwell_mean_quality, band_empirical_hit_rate, global_empirical_hit_rate.

3. Dynamic Dwell Scaling:
   - Predicts P_persist = P(signal active at t+1 | history) in [0, 1].
   - If P_persist >= 0.42:
       bonus = dwell_inertia * P_persist * [1.20 / (1.0 + 0.10 * consecutive_dwell)]
   - If P_persist < 0.42:
       bonus = 0.0 (Immediate release, switching away to explore).

LESSON LEARNED & ARCHITECTURAL TAKEAWAY:
- Although the GBDT achieved 88.6% offline classification accuracy, in closed-loop
  control it suffered from 'exploration starvation': because it was trained only
  to predict whether staying was safe, it rarely authorized switches (31.4 switches),
  suppressing total signal discoveries (8.8% hit rate vs 15.1% for Champion).
"""

from __future__ import annotations

from typing import Optional
from scheduler.boosted_tree_dwell_scheduler import BoostedTreeDwellScheduler
from scheduler.track2_runtime import DEFAULT_MODEL_PATH

NUM_BANDS = 20


def build_boosted_tree_dwell(num_bands: int = NUM_BANDS, seed: Optional[int] = None) -> BoostedTreeDwellScheduler:
    """Instantiate the Boosted-Tree Adaptive Dwell scheduler."""
    return BoostedTreeDwellScheduler(
        num_bands=num_bands,
        nmf_scale=1.00,
        switch_penalty=0.08,
        dwell_inertia=1.30,
        dwell_threshold=0.42,
        explore_budget_prob=0.12,
        seed=seed,
        model_path=DEFAULT_MODEL_PATH,
    )


if __name__ == "__main__":
    sched = build_boosted_tree_dwell(20)
    print(f"Instantiated {sched.__class__.__name__} successfully.")
    for t in range(5):
        b = sched.select_band()
        sched.update(b, 1.0 if t % 2 == 0 else -0.1, {"detected": t % 2 == 0, "quality": [0.80]})
        print(f"Step {t}: Selected Band {b}, Consecutive Dwell {sched.consecutive_dwell}")
