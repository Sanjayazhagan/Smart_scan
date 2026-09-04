"""Comparative Research & Benchmark Baseline Schedulers.

This module houses all historical, ablation, and comparative baseline models evaluated
during the development of the SmartScan cognitive spectrum system.
For the primary production champion, import SmartScanProductionScheduler from scheduler.smartscan_production.

Models preserved in this registry:
1. SmartScan-Omni V2 (RL Critic + Expectimax + RPCA Filter + Dwell)
2. SmartScan V2 (UCB Top-K + Expectimax Lookahead)
3. SmartScan V2-NMF (NMF Candidate Generator + Expectimax Lookahead)
4. Dual-Policy Uncertainty (Uncertainty-Arbitrated NMF + UCB Scout)
5. Robust PCA + PSR (Inexact ALM Sparse De-noising + PSR)
6. Direct NMF (Pure Matrix Factorization)
7. Static NMF+UCB (Additive Fusion)
8. Whittle Index RMAB (Restless Bandit)
9. Observable Plain UCB (Classical Bandit)
10. Adaptive Mixture of Experts (Retired MoE)
11. Alarm-Adaptive NMF+UCB (Change-Point Detector + Reset)
"""

from scheduler.smartscan_omni import SmartScanOmniScheduler
from scheduler.smartscan_v2 import SmartScanScheduler
from scheduler.nmf_expectimax import NMFExpectimaxScheduler
from scheduler.dual_policy_uncertainty_scheduler import DualPolicyUncertaintyScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.paradigms.mathematical_scheduling import WhittleIndexRMABScheduler
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.world_model_ucb import WorldModelUCBScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler, AdaptiveMixtureOfExpertsScheduler
from scheduler.alarm_adaptive_nmf_ucb import AlarmAdaptiveNMFUCBScheduler
from scheduler.calibrated_soft_moe import CalibratedSoftMoEScheduler
from scheduler.baselines import FixedScheduler, RandomScheduler

__all__ = [
    "SmartScanOmniScheduler",
    "SmartScanScheduler",
    "NMFExpectimaxScheduler",
    "DualPolicyUncertaintyScheduler",
    "RobustPCAPSRScheduler",
    "NMFScheduler",
    "WhittleIndexRMABScheduler",
    "WorldModelNMFUCBScheduler",
    "WorldModelUCBScheduler",
    "ObservableDiscountedUCBScheduler",
    "AdaptiveMixtureOfExpertsScheduler",
    "AlarmAdaptiveNMFUCBScheduler",
    "CalibratedSoftMoEScheduler",
    "FixedScheduler",
    "RandomScheduler",
]
