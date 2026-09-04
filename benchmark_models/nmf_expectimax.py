"""Model 07: SmartScan V2-NMF (Lookahead Expectimax Search).

========================================================================================
PROJECT STATUS: LOOKAHEAD SEARCH CHAMPION (#1 IN HARSH MULTIPATH NOISE +2.3)
========================================================================================
- Mean Cumulative Reward : +18.04 +/- 5.30 (Master 15-Seed Benchmark)
- Harsh Noise Score      : +2.30 (Best-performing model under heavy Rayleigh fading)
- Decision Latency       : 1.90 ms mean (Bounded by branch-and-bound pruning)
- Switching Count        : 50.6 switches per episode

THEORY & ARCHITECTURE:
Combines Non-negative Matrix Factorization (NMF) with Depth-2 Expectimax Lookahead Tree Search:

1. Belief Distribution via Matrix Factorization:
   - Evaluates sliding-window receiver spectral history V ~ W * H.
   - Converts the factorized reconstruction into a normalized prior probability distribution
     over all 20 bands.

2. Expectimax Lookahead Tree:
   - Root (Decision Node): Branch on candidate action a in Top-K (K=4).
   - Chance Nodes: Evaluates both HIT and MISS posterior states weighted by NMF probability P(HIT).
   - Pruning Safeguards:
       - Cumulative path probability threshold: Rejects branches with P_path < 0.005.
       - Optimistic upper bound pruning: Prunes branches that cannot surpass current alpha.
       - Transposition caching: Caches evaluated subtrees based on quantized belief hash.

STRENGTHS:
- Outstanding in high-noise/harsh environments (+2.3) where myopic models make false switches.
- Explicitly models future observation uncertainty across 2-step decision horizons.
"""

from __future__ import annotations

from typing import Optional
from scheduler.nmf_expectimax import NMFExpectimaxScheduler
from scheduler.track2_runtime import DEFAULT_MODEL_PATH

NUM_BANDS = 20


def build_nmf_expectimax(num_bands: int = NUM_BANDS, seed: Optional[int] = None) -> NMFExpectimaxScheduler:
    """Instantiate the SmartScan V2-NMF Expectimax scheduler."""
    return NMFExpectimaxScheduler(
        num_bands=num_bands,
        depth=2,
        top_k=4,
        branch_k=3,
        switch_penalty=0.08,
        nmf_weight=1.0,
        curiosity_scale=0.0,
        use_rl_critic=False,
        enable_pruning=True,
        enable_caching=True,
        seed=seed,
        model_path=DEFAULT_MODEL_PATH,
    )


if __name__ == "__main__":
    sched = build_nmf_expectimax(20)
    print(f"Instantiated {sched.__class__.__name__} successfully.")
    for t in range(5):
        b = sched.select_band()
        sched.update(b, 1.0 if t % 2 == 0 else -0.1, {"detected": t % 2 == 0, "quality": [0.85]})
        print(f"Step {t}: Selected Band {b}")
