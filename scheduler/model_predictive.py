"""Depth-limited model-predictive planning over observable Track 2 state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scheduler.baselines import BaseScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime


@dataclass(frozen=True)
class PlannerConfig:
    depth: int = 3
    beam_width: int = 8
    discount: float = 0.95
    information_weight: float = 0.30
    scan_age_weight: float = 0.15
    repeat_penalty: float = 0.05
    miss_prob: float = 0.10
    false_alarm_prob: float = 0.05
    miss_belief_discount: float = 0.75
    belief_persistence: float = 0.99

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("Planner depth must be at least one")
        if self.beam_width < 1:
            raise ValueError("Planner beam width must be at least one")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("Planner discount must be in [0,1]")


@dataclass
class _PlanNode:
    score: float
    sequence: tuple[int, ...]
    belief: np.ndarray
    scan_age: np.ndarray
    uncertainty: np.ndarray


class ModelPredictiveScheduler(BaseScheduler):
    """Receding-horizon scheduler using Track 2 predictions without ground truth.

    The planner expands candidate band sequences with beam search. It estimates
    the existing simulator reward from Track 2 band belief, adds observable
    information/recency value, executes only the first band, and replans after
    every real observation.
    """

    def __init__(
        self,
        num_bands: int,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        config: PlannerConfig | None = None,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        if num_bands != NUM_BANDS:
            raise ValueError(f"Track 2 planning requires exactly {NUM_BANDS} bands")
        super().__init__(num_bands)
        self.runtime = runtime or Track2Runtime(
            model_path, max_scan_age=max_scan_age
        )
        self.config = config or PlannerConfig()
        self.max_scan_age = float(max_scan_age)
        self.timestamp = 0.0
        self.last_plan: tuple[int, ...] = ()
        self.last_plan_score = float("nan")

    def _expected_environment_reward(self, presence_probability: float) -> float:
        config = self.config
        true_signal_reward = (1.0 - config.miss_prob) - config.miss_prob
        empty_band_reward = (
            (1.0 - config.false_alarm_prob) * 0.1
            - config.false_alarm_prob
        )
        probability = float(np.clip(presence_probability, 0.0, 1.0))
        return probability * true_signal_reward + (1.0 - probability) * empty_band_reward

    def _simulate_scan(self, node: _PlanNode, action: int, depth: int) -> _PlanNode:
        config = self.config
        probability = float(node.belief[action])
        uncertainty = float(node.uncertainty[action])
        age = float(node.scan_age[action])
        immediate_score = (
            self._expected_environment_reward(probability)
            + config.information_weight * uncertainty
            + config.scan_age_weight * age
        )
        if node.sequence and node.sequence[-1] == action:
            immediate_score -= config.repeat_penalty

        next_belief = np.clip(
            node.belief * config.belief_persistence, 0.0, 1.0
        ).astype(np.float32)
        detection_probability = float(
            np.clip(
                probability * (1.0 - config.miss_prob)
                + (1.0 - probability) * config.false_alarm_prob,
                0.0,
                1.0,
            )
        )
        no_detection_probability = 1.0 - detection_probability
        next_belief[action] *= (
            1.0 - config.miss_belief_discount * no_detection_probability
        )

        age_step = 1.0 / max(self.max_scan_age, 1.0)
        next_age = np.clip(node.scan_age + age_step, 0.0, 1.0).astype(np.float32)
        next_age[action] = 0.0

        next_uncertainty = np.clip(
            node.uncertainty + age_step, 0.0, 1.0
        ).astype(np.float32)
        # A scan is informative even when it misses. Expected remaining
        # uncertainty is largest only when a detection is likely.
        next_uncertainty[action] *= detection_probability

        return _PlanNode(
            score=node.score + (config.discount**depth) * immediate_score,
            sequence=node.sequence + (action,),
            belief=next_belief,
            scan_age=next_age,
            uncertainty=next_uncertainty,
        )

    def plan(self) -> tuple[tuple[int, ...], float]:
        initial = _PlanNode(
            score=0.0,
            sequence=(),
            belief=self.runtime.get_band_belief(),
            scan_age=self.runtime.get_scan_age(normalized=True),
            uncertainty=self.runtime.get_band_uncertainty(),
        )
        beam = [initial]
        for depth in range(self.config.depth):
            candidates = [
                self._simulate_scan(node, action, depth)
                for node in beam
                for action in range(NUM_BANDS)
            ]
            candidates.sort(
                key=lambda node: (node.score, tuple(-x for x in node.sequence)),
                reverse=True,
            )
            beam = candidates[: self.config.beam_width]
        best = beam[0]
        self.last_plan = best.sequence
        self.last_plan_score = float(best.score)
        return best.sequence, float(best.score)

    def select_band(self) -> int:
        sequence, _ = self.plan()
        action = int(sequence[0])
        if not 0 <= action < NUM_BANDS:
            raise RuntimeError(f"Planner returned invalid action {action}")
        return action

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=self.timestamp)
        self.timestamp += 1.0
