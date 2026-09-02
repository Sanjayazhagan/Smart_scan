"""Active schedulers for the Smart Scan research prototype.

Primary interfaces:
- AdaptiveMixtureOfExpertsScheduler: synchronous observable multi-regime router.
- UCBFirstAdaptiveScheduler: product controller; UCB default, adaptive escalation.
- AsyncPlanningScheduler: experimental background-planning wrapper.
- Track2Runtime: Perceptual world model with Conv1D fingerprinting and recurrent GRU.
"""

from scheduler.adaptive_moe import AdaptiveMixtureOfExpertsScheduler
from scheduler.async_worker import AsyncPlanningScheduler
from scheduler.belief_tree import ObservationDependentBeliefTreePlanner
from scheduler.emitter_aware_predictive import EmitterAwareModelPredictivePlanner
from scheduler.track2_runtime import Track2Runtime
from scheduler.ucb_first_adaptive import UCBFirstAdaptiveScheduler

__all__ = [
    "AdaptiveMixtureOfExpertsScheduler",
    "AsyncPlanningScheduler",
    "ObservationDependentBeliefTreePlanner",
    "EmitterAwareModelPredictivePlanner",
    "Track2Runtime",
    "UCBFirstAdaptiveScheduler",
]
