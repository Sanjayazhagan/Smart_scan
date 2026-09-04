"""SmartScan Benchmark Models Registry.

Provides cleanly decoupled access to all major RF scan scheduling models,
their theoretical formulations, empirical benchmark stats, and builders.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Optional

from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
from benchmark_models.smartscan_omni import build_smartscan_omni_tuned, SmartScanOmniScheduler
from benchmark_models.robust_pca_psr import build_robust_pca_psr, RobustPCAPSRScheduler
from benchmark_models.contextual_bandit import build_linucb_bandit, ContextualBanditScheduler
from benchmark_models.grud_recurrent import build_grud_scheduler, GRUDSpectrumScheduler
from benchmark_models.boosted_tree_dwell import build_boosted_tree_dwell, BoostedTreeDwellScheduler
from benchmark_models.nmf_expectimax import build_nmf_expectimax, NMFExpectimaxScheduler
from benchmark_models.dual_policy_uncertainty import build_dual_policy_uncertainty, DualPolicyUncertaintyScheduler
from benchmark_models.mathematical_baselines import (
    build_direct_nmf,
    build_static_nmf_ucb,
    build_whittle_rmab,
    build_observable_plain_ucb,
    build_random_scan,
    build_fixed_sweep,
)

CHAMPION_MODEL_NAME = "Dwell-Dual Policy (Champion)"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Dwell-Dual Policy (Champion)": {
        "builder": lambda n, s: DwellDualPolicyScheduler(n, seed=s),
        "rank": 1,
        "mean_reward": 17.72,
        "hit_rate_pct": 14.7,
        "latency_ms": 0.15,
        "category": "Grand Champion",
        "description": "NMF Spectral Discovery + Dwell-Lock Inertia + Agile Exploration",
        "file": "benchmark_models/dwell_dual_policy.py",
    },
    "SmartScan-Omni V2 (Tuned)": {
        "builder": lambda n, s: build_smartscan_omni_tuned(n, s),
        "rank": 2,
        "mean_reward": 16.88,
        "hit_rate_pct": 14.1,
        "latency_ms": 1.85,
        "category": "Lookahead / EW Specialist",
        "description": "Depth-2 Expectimax Lookahead + RL Value Critic + Robust PCA Filter",
        "file": "benchmark_models/smartscan_omni.py",
    },
    "Robust PCA + PSR": {
        "builder": lambda n, s: build_robust_pca_psr(n, s),
        "rank": 3,
        "mean_reward": 16.16,
        "hit_rate_pct": 12.3,
        "latency_ms": 0.012,
        "category": "Embedded Champion",
        "description": "Inexact ALM Low-Rank / Sparse Decomposition + Predictive Spectral Recovery",
        "file": "benchmark_models/robust_pca_psr.py",
    },
    "Candidate 1: LinUCB Bandit": {
        "builder": lambda n, s: build_linucb_bandit(n, s),
        "rank": 4,
        "mean_reward": 15.52,
        "hit_rate_pct": 12.7,
        "latency_ms": 0.09,
        "category": "Contextual Bandit",
        "description": "Sherman-Morrison Rank-1 Inversion + 8-Dim Context Vector + 4 Strategy Arms",
        "file": "benchmark_models/contextual_bandit.py",
    },
    "Candidate 3: GRU-D Recurrent": {
        "builder": lambda n, s: build_grud_scheduler(n, s),
        "rank": 5,
        "mean_reward": 15.17,
        "hit_rate_pct": 11.9,
        "latency_ms": 0.012,
        "category": "Deep Recurrent Network",
        "description": "Che et al. (Nature 2018) Observation & Memory Decay for Missing Data",
        "file": "benchmark_models/grud_recurrent.py",
    },
    "SmartScan V2-NMF": {
        "builder": lambda n, s: build_nmf_expectimax(n, s),
        "rank": 6,
        "mean_reward": 18.04,
        "hit_rate_pct": 14.5,
        "latency_ms": 1.90,
        "category": "Expectimax Search",
        "description": "Depth-2 Expectimax Tree Search over NMF Belief Distributions",
        "file": "benchmark_models/nmf_expectimax.py",
    },
    "Dual-Policy Uncertainty": {
        "builder": lambda n, s: build_dual_policy_uncertainty(n, s),
        "rank": 7,
        "mean_reward": 17.65,
        "hit_rate_pct": 15.1,
        "latency_ms": 0.14,
        "category": "Dual-Mode Bandit",
        "description": "Probabilistic Arbitration between NMF Exploitation and Curiosity Exploration",
        "file": "benchmark_models/dual_policy_uncertainty.py",
    },
    "Candidate 2: Boosted-Tree Dwell": {
        "builder": lambda n, s: build_boosted_tree_dwell(n, s),
        "rank": 8,
        "mean_reward": 11.66,
        "hit_rate_pct": 8.8,
        "latency_ms": 0.07,
        "category": "Boosted Tree",
        "description": "25-Tree GBDT Ensemble Estimating P(active next step | history)",
        "file": "benchmark_models/boosted_tree_dwell.py",
    },
    "Direct NMF": {
        "builder": lambda n, s: build_direct_nmf(n, s),
        "rank": 9,
        "mean_reward": 16.05,
        "hit_rate_pct": 14.1,
        "latency_ms": 0.012,
        "category": "Matrix Factorization",
        "description": "Non-negative Matrix Factorization (#1 in Stationary Radars: +52.3)",
        "file": "benchmark_models/mathematical_baselines.py",
    },
    "Static NMF+UCB": {
        "builder": lambda n, s: build_static_nmf_ucb(n, s),
        "rank": 10,
        "mean_reward": 16.88,
        "hit_rate_pct": 15.0,
        "latency_ms": 1.35,
        "category": "Heuristic Fusion",
        "description": "Additive Fusion of Low-Rank NMF Belief and Non-Stationary Discounted UCB",
        "file": "benchmark_models/mathematical_baselines.py",
    },
    "Whittle Index RMAB": {
        "builder": lambda n, s: build_whittle_rmab(n, s),
        "rank": 11,
        "mean_reward": 13.86,
        "hit_rate_pct": 12.6,
        "latency_ms": 0.027,
        "category": "Mathematical Scheduling",
        "description": "Restless Multi-Armed Bandit with Closed-Form Whittle Index Policy",
        "file": "benchmark_models/mathematical_baselines.py",
    },
    "Observable Plain UCB": {
        "builder": lambda n, s: build_observable_plain_ucb(n, s),
        "rank": 12,
        "mean_reward": 13.09,
        "hit_rate_pct": 10.8,
        "latency_ms": 0.005,
        "category": "Discounted Bandit",
        "description": "Discounted Upper Confidence Bound on Partially Observable Channels",
        "file": "benchmark_models/mathematical_baselines.py",
    },
    "Uniform Random Scan": {
        "builder": lambda n, s: build_random_scan(n, s),
        "rank": 13,
        "mean_reward": 10.56,
        "hit_rate_pct": 8.6,
        "latency_ms": 0.002,
        "category": "Random Baseline",
        "description": "Uniform Random Channel Exploration Baseline",
        "file": "benchmark_models/mathematical_baselines.py",
    },
    "Fixed Sequential Sweep": {
        "builder": lambda n, s: build_fixed_sweep(n, s),
        "rank": 14,
        "mean_reward": 8.99,
        "hit_rate_pct": 8.2,
        "latency_ms": 0.001,
        "category": "Sequential Baseline",
        "description": "Deterministic Round-Robin Raster Scan Baseline",
        "file": "benchmark_models/mathematical_baselines.py",
    },
}


def get_model(name: str, num_bands: int = 20, seed: Optional[int] = None):
    """Instantiate any model by registered name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]["builder"](num_bands, seed)
