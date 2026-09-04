"""Active schedulers for the Smart Scan system.

Primary Architecture (SmartScan V2):
- SmartScanScheduler: Unified production scheduler (Track 2 -> Smart Discounted UCB -> Top-K Expectimax).
- SmartDiscountedUCB: Fast non-stationary multi-factor UCB candidate generator.
- BeliefExpectimaxPlanner: Bounded belief-state stochastic lookahead planner.
- Track2Runtime: Perceptual world model with Conv1D fingerprinting and recurrent GRU.

Research & Comparative Baselines:
- WorldModelNMFUCBScheduler: Static and additive matrix factorization hybrids.
- AdaptiveMixtureOfExpertsScheduler: Retired MoE research baseline.
- WhittleIndexRMABScheduler / NMFScheduler / RobustPCAPSRScheduler: Closed-form baselines.
"""

from scheduler.smartscan_v2 import SmartScanScheduler
from scheduler.smartscan_omni import SmartScanOmniScheduler
from scheduler.nmf_expectimax import NMFExpectimaxScheduler
from scheduler.smart_discounted_ucb import SmartDiscountedUCB
from scheduler.belief_expectimax import BeliefExpectimaxPlanner, BeliefState
from scheduler.track2_runtime import Track2Runtime
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.world_model_ucb import WorldModelUCBScheduler
from scheduler.adaptive_moe import AdaptiveMixtureOfExpertsScheduler
from scheduler.async_worker import AsyncPlanningScheduler
from scheduler.belief_tree import ObservationDependentBeliefTreePlanner
from scheduler.emitter_aware_predictive import EmitterAwareModelPredictivePlanner
from scheduler.alarm_adaptive_nmf_ucb import AlarmAdaptiveNMFUCBScheduler
from scheduler.dual_policy_uncertainty_scheduler import DualPolicyUncertaintyScheduler
from scheduler.dwell_dual_policy_scheduler import DwellDualPolicyScheduler

__all__ = [
    "SmartScanScheduler",
    "SmartScanOmniScheduler",
    "NMFExpectimaxScheduler",
    "SmartDiscountedUCB",
    "BeliefExpectimaxPlanner",
    "BeliefState",
    "Track2Runtime",
    "WorldModelNMFUCBScheduler",
    "WorldModelUCBScheduler",
    "AdaptiveMixtureOfExpertsScheduler",
    "AsyncPlanningScheduler",
    "ObservationDependentBeliefTreePlanner",
    "EmitterAwareModelPredictivePlanner",
    "AlarmAdaptiveNMFUCBScheduler",
    "DualPolicyUncertaintyScheduler",
    "DwellDualPolicyScheduler",
]
