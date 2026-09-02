"""Observation-dependent, beam-limited belief-tree planning.

Unlike a fixed action sequence, each candidate scan branches into observable
HIT and MISS posteriors. The rollout remains approximate: it does not fabricate
future I/Q or advance the frozen GRU without a real observation.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from scheduler.emitter_aware_predictive import (
    EmitterAwareModelPredictivePlanner,
    EmitterAwarePlannerConfig,
    _EmitterPlanNode,
    derive_emitter_band_features,
)
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime


class ObservationDependentBeliefTreePlanner(EmitterAwareModelPredictivePlanner):
    """Budgeted receding-horizon planner with explicit hit/miss branches.

    ``branch_width`` limits only deeper conditional branches.  The root still
    considers ``config.beam_width`` actions, while the final layer is solved
    directly because its hit/miss children have no future value.  This keeps
    the useful observation-dependent part of the tree without paying for
    thousands of equivalent terminal nodes.
    """

    def __init__(
        self,
        num_bands: int,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        config: EmitterAwarePlannerConfig | None = None,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        branch_width: int = 4,
    ):
        super().__init__(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            config=config or EmitterAwarePlannerConfig(exploration_probability=0.0),
            max_scan_age=max_scan_age,
        )
        if branch_width < 1:
            raise ValueError("Belief-tree branch width must be at least one")
        self.branch_width = min(int(branch_width), self.config.beam_width)
        self.last_belief_tree: dict = {}
        self.last_search_stats: dict = {}
        self._evaluated_actions = 0
        self._expanded_branches = 0

    def _detection_probability(self, belief: float) -> float:
        return float(
            np.clip(
                belief * (1.0 - self.config.miss_prob)
                + (1.0 - belief) * self.config.false_alarm_prob,
                0.0,
                1.0,
            )
        )

    def _observation_transition(
        self, node: _EmitterPlanNode, action: int, detected: bool
    ) -> _EmitterPlanNode:
        config = self.config
        prior = float(np.clip(node.belief[action], 0.0, 1.0))
        detection_probability = self._detection_probability(prior)
        if detected:
            posterior = prior * (1.0 - config.miss_prob) / max(
                detection_probability, 1e-8
            )
        else:
            miss_probability = 1.0 - detection_probability
            posterior = prior * config.miss_prob / max(miss_probability, 1e-8)

        next_belief = np.clip(
            node.belief * config.belief_persistence, 0.0, 1.0
        ).astype(np.float32)
        next_belief[action] = float(np.clip(posterior, 0.0, 1.0))
        age_step = 1.0 / max(self.max_scan_age, 1.0)
        next_age = np.clip(node.scan_age + age_step, 0.0, 1.0).astype(np.float32)
        next_age[action] = 0.0
        next_uncertainty = np.clip(
            node.uncertainty + age_step, 0.0, 1.0
        ).astype(np.float32)
        next_uncertainty[action] *= 0.25 if detected else 0.50
        return _EmitterPlanNode(
            score=node.score,
            sequence=node.sequence + (action,),
            belief=next_belief,
            scan_age=next_age,
            uncertainty=next_uncertainty,
            emitter=node.emitter.persisted(
                persistence=config.emitter_feature_persistence,
                anomaly_decay=config.anomaly_decay,
                quality_decay=config.quality_decay,
                recency_step=age_step,
            ),
        )

    def _search(self, node: _EmitterPlanNode, depth: int):
        if depth >= self.config.depth:
            return 0.0, (), {}
        width = self.config.beam_width if depth == 0 else self.branch_width
        ranked = sorted(
            (
                (self._score_breakdown(node, action)["total"], action)
                for action in range(NUM_BANDS)
            ),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )[:width]
        self._evaluated_actions += NUM_BANDS

        # At the last decision layer there is no future utility beneath the
        # HIT/MISS outcomes.  Selecting the best immediate action is exactly
        # equivalent to expanding both terminal children, but much cheaper.
        if depth == self.config.depth - 1:
            immediate, action = ranked[0]
            probability = self._detection_probability(float(node.belief[action]))
            return float(immediate), (int(action),), {
                "band": int(action),
                "expected_value": float(immediate),
                "detection_probability": probability,
                "hit": {
                    "probability": probability,
                    "future_value": 0.0,
                    "next": {},
                },
                "miss": {
                    "probability": 1.0 - probability,
                    "future_value": 0.0,
                    "next": {},
                },
            }

        best_value = -float("inf")
        best_sequence: tuple[int, ...] = ()
        best_tree: dict = {}
        for immediate, action in ranked:
            self._expanded_branches += 1
            probability = self._detection_probability(float(node.belief[action]))
            hit_node = self._observation_transition(node, action, True)
            miss_node = self._observation_transition(node, action, False)
            hit_value, hit_sequence, hit_tree = self._search(hit_node, depth + 1)
            miss_value, miss_sequence, miss_tree = self._search(miss_node, depth + 1)
            expected_future = probability * hit_value + (1.0 - probability) * miss_value
            value = float(immediate + self.config.discount * expected_future)
            if value > best_value or (
                np.isclose(value, best_value) and action < best_sequence[0]
            ):
                representative = hit_sequence if probability >= 0.5 else miss_sequence
                best_value = value
                best_sequence = (action,) + representative
                best_tree = {
                    "band": int(action),
                    "expected_value": value,
                    "detection_probability": probability,
                    "hit": {
                        "probability": probability,
                        "future_value": float(hit_value),
                        "next": hit_tree,
                    },
                    "miss": {
                        "probability": 1.0 - probability,
                        "future_value": float(miss_value),
                        "next": miss_tree,
                    },
                }
        return best_value, best_sequence, best_tree

    def plan(self) -> tuple[tuple[int, ...], float]:
        started = perf_counter()
        self._evaluated_actions = 0
        self._expanded_branches = 0
        global_belief = self.runtime.get_global_belief()
        root = _EmitterPlanNode(
            score=0.0,
            sequence=(),
            belief=self.runtime.get_band_belief(),
            scan_age=self.runtime.get_scan_age(normalized=True),
            uncertainty=self.runtime.get_band_uncertainty(),
            emitter=derive_emitter_band_features(
                global_belief, support_threshold=self.config.support_threshold
            ),
        )
        value, sequence, tree = self._search(root, 0)
        self._last_root = root
        self._last_global_belief = global_belief
        self.last_plan = sequence
        self.last_plan_score = float(value)
        self.last_belief_tree = tree
        self.last_search_stats = {
            "depth": int(self.config.depth),
            "root_width": int(self.config.beam_width),
            "branch_width": int(self.branch_width),
            "evaluated_actions": int(self._evaluated_actions),
            "expanded_branches": int(self._expanded_branches),
            "planning_time_ms": float((perf_counter() - started) * 1000.0),
        }
        return sequence, float(value)

    def select_band(self) -> int:
        sequence, value = self.plan()
        action = int(sequence[0])
        breakdown = self._score_breakdown(self._last_root, action)
        self.last_decision_trace = {
            "selected_band": action,
            "selection_mode": "observation_dependent_belief_tree",
            "representative_sequence": list(sequence),
            "expected_tree_value": value,
            "score": breakdown,
            "belief_tree": self.last_belief_tree,
            "search_stats": dict(self.last_search_stats),
            "top_contributing_emitters": self._top_contributing_emitters(action),
        }
        return action
