"""Emitter-aware model-predictive planning and controlled MPP ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scheduler.model_predictive import ModelPredictiveScheduler, PlannerConfig
from scheduler.track2_core import FEATURE_DIM, NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime


NEXT_BAND_SLICE = slice(0, 20)
PREDICTION_UNCERTAINTY = 41
IDENTITY_NOVELTY = 42
BEHAVIOUR_CHANGE = 43
TRACK_RECENCY = 44
OBSERVATION_QUALITY = 45
IDENTITY_SIMILARITY = 46
CONFIRMED = 47


@dataclass
class EmitterBandFeatures:
    dominant_emitter_probability: np.ndarray
    supporting_emitter_count: np.ndarray
    supporting_emitter_fraction: np.ndarray
    emitter_weighted_uncertainty: np.ndarray
    emitter_weighted_behaviour_change: np.ndarray
    emitter_weighted_identity_novelty: np.ndarray
    emitter_weighted_recency: np.ndarray
    emitter_weighted_quality: np.ndarray
    emitter_weighted_identity_confidence: np.ndarray

    @classmethod
    def zeros(cls):
        zero = lambda: np.zeros(NUM_BANDS, dtype=np.float32)
        return cls(*(zero() for _ in range(9)))

    def persisted(
        self,
        persistence: float,
        anomaly_decay: float,
        quality_decay: float,
        recency_step: float,
    ) -> "EmitterBandFeatures":
        return EmitterBandFeatures(
            dominant_emitter_probability=np.clip(
                self.dominant_emitter_probability * persistence, 0.0, 1.0
            ).astype(np.float32),
            supporting_emitter_count=(
                self.supporting_emitter_count * persistence
            ).astype(np.float32),
            supporting_emitter_fraction=np.clip(
                self.supporting_emitter_fraction * persistence, 0.0, 1.0
            ).astype(np.float32),
            emitter_weighted_uncertainty=np.clip(
                self.emitter_weighted_uncertainty + recency_step, 0.0, 1.0
            ).astype(np.float32),
            emitter_weighted_behaviour_change=np.clip(
                self.emitter_weighted_behaviour_change * anomaly_decay, 0.0, 1.0
            ).astype(np.float32),
            emitter_weighted_identity_novelty=np.clip(
                self.emitter_weighted_identity_novelty * anomaly_decay, 0.0, 1.0
            ).astype(np.float32),
            emitter_weighted_recency=np.clip(
                self.emitter_weighted_recency + recency_step, 0.0, 1.0
            ).astype(np.float32),
            emitter_weighted_quality=np.clip(
                self.emitter_weighted_quality * quality_decay, 0.0, 1.0
            ).astype(np.float32),
            emitter_weighted_identity_confidence=self.emitter_weighted_identity_confidence.copy(),
        )


def derive_emitter_band_features(
    global_belief: dict,
    support_threshold: float = 0.05,
) -> EmitterBandFeatures:
    """Convert confirmed masked track rows into interpretable band quantities."""
    track_features = np.asarray(global_belief["track_features"], dtype=np.float32)
    track_mask = np.asarray(global_belief["track_mask"], dtype=np.float32).reshape(-1)
    if track_features.ndim != 2 or track_features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected track_features [N,{FEATURE_DIM}], got {track_features.shape}"
        )
    if track_features.shape[0] != track_mask.shape[0]:
        raise ValueError("track_features and track_mask row counts do not match")

    valid = (track_mask > 0.5) & (track_features[:, CONFIRMED] > 0.5)
    if not np.any(valid):
        return EmitterBandFeatures.zeros()

    rows = track_features[valid]
    probabilities = np.clip(rows[:, NEXT_BAND_SLICE], 0.0, 1.0)
    probability_mass = probabilities.sum(axis=0)
    safe_mass = np.where(probability_mass > 1e-8, probability_mass, 1.0)

    def weighted(column: int) -> np.ndarray:
        values = np.clip(rows[:, column], 0.0, 1.0)
        result = (probabilities * values[:, None]).sum(axis=0) / safe_mass
        result[probability_mass <= 1e-8] = 0.0
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    support_count = (probabilities >= float(support_threshold)).sum(axis=0)
    return EmitterBandFeatures(
        dominant_emitter_probability=probabilities.max(axis=0).astype(np.float32),
        supporting_emitter_count=support_count.astype(np.float32),
        supporting_emitter_fraction=(
            support_count / max(1, probabilities.shape[0])
        ).astype(np.float32),
        emitter_weighted_uncertainty=weighted(PREDICTION_UNCERTAINTY),
        emitter_weighted_behaviour_change=weighted(BEHAVIOUR_CHANGE),
        emitter_weighted_identity_novelty=weighted(IDENTITY_NOVELTY),
        emitter_weighted_recency=weighted(TRACK_RECENCY),
        emitter_weighted_quality=weighted(OBSERVATION_QUALITY),
        emitter_weighted_identity_confidence=weighted(IDENTITY_SIMILARITY),
    )


@dataclass(frozen=True)
class EmitterAwarePlannerConfig(PlannerConfig):
    detection_weight: float = 1.0
    behaviour_weight: float = 0.05
    novelty_weight: float = 0.0
    track_recency_weight: float = 0.0
    identity_weight: float = 0.0
    dominant_emitter_weight: float = 0.05
    supporting_emitters_weight: float = 0.02
    support_threshold: float = 0.05
    emitter_feature_persistence: float = 0.95
    anomaly_decay: float = 0.85
    quality_decay: float = 0.98
    exploration_probability: float = 0.05
    exploration_top_k: int = 3
    exploration_seed: int = 42

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("Exploration probability must be in [0,1]")
        if self.exploration_top_k < 1:
            raise ValueError("Exploration top-k must be at least one")


@dataclass
class _EmitterPlanNode:
    score: float
    sequence: tuple[int, ...]
    belief: np.ndarray
    scan_age: np.ndarray
    uncertainty: np.ndarray
    emitter: EmitterBandFeatures


class BeliefOnlyModelPredictiveScheduler(ModelPredictiveScheduler):
    """Ablation A: MPP using band belief and expected reward only."""

    def __init__(
        self,
        num_bands: int,
        runtime=None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        super().__init__(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            config=PlannerConfig(
                depth=3,
                beam_width=8,
                information_weight=0.0,
                scan_age_weight=0.0,
                repeat_penalty=0.0,
            ),
            max_scan_age=max_scan_age,
        )


class EmitterAwareModelPredictivePlanner(ModelPredictiveScheduler):
    """MPP-EmitterAware: MPP-60 augmented by masked confirmed-track evidence."""

    def __init__(
        self,
        num_bands: int,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        config: EmitterAwarePlannerConfig | None = None,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        super().__init__(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            config=config or EmitterAwarePlannerConfig(),
            max_scan_age=max_scan_age,
        )
        self.config: EmitterAwarePlannerConfig
        self.last_decision_trace: dict = {}
        self._last_root: _EmitterPlanNode | None = None
        self._last_global_belief: dict | None = None
        self._last_first_step_nodes: list[_EmitterPlanNode] = []
        self._exploration_rng = np.random.default_rng(self.config.exploration_seed)

    def _score_breakdown(
        self,
        node: _EmitterPlanNode,
        action: int,
    ) -> dict[str, float]:
        config = self.config
        belief = float(node.belief[action])
        quality = float(node.emitter.emitter_weighted_quality[action])
        repeated = bool(node.sequence and node.sequence[-1] == action)
        contributions = {
            "expected_detection": config.detection_weight
            * self._expected_environment_reward(belief),
            "information_value": config.information_weight
            * float(node.uncertainty[action]),
            "scan_age_value": config.scan_age_weight
            * float(node.scan_age[action]),
            "behaviour_investigation": config.behaviour_weight
            * belief
            * float(node.emitter.emitter_weighted_behaviour_change[action])
            * quality,
            "novelty_investigation": config.novelty_weight
            * belief
            * float(node.emitter.emitter_weighted_identity_novelty[action])
            * quality,
            "emitter_recency_value": config.track_recency_weight
            * belief
            * float(node.emitter.emitter_weighted_recency[action])
            * quality,
            "identity_confidence_value": config.identity_weight
            * belief
            * float(node.emitter.emitter_weighted_identity_confidence[action])
            * quality,
            "dominant_emitter_value": config.dominant_emitter_weight
            * float(node.emitter.dominant_emitter_probability[action])
            * quality,
            "supporting_emitters_value": config.supporting_emitters_weight
            * float(node.emitter.supporting_emitter_fraction[action])
            * quality,
            "repeat_penalty": -config.repeat_penalty if repeated else 0.0,
        }
        contributions["total"] = float(sum(contributions.values()))
        return {key: float(value) for key, value in contributions.items()}

    def _simulate_scan(
        self, node: _EmitterPlanNode, action: int, depth: int
    ) -> _EmitterPlanNode:
        config = self.config
        breakdown = self._score_breakdown(node, action)
        probability = float(node.belief[action])
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
        next_belief[action] *= 1.0 - config.miss_belief_discount * (
            1.0 - detection_probability
        )

        age_step = 1.0 / max(self.max_scan_age, 1.0)
        next_age = np.clip(node.scan_age + age_step, 0.0, 1.0).astype(np.float32)
        next_age[action] = 0.0
        next_uncertainty = np.clip(
            node.uncertainty + age_step, 0.0, 1.0
        ).astype(np.float32)
        next_uncertainty[action] *= detection_probability

        return _EmitterPlanNode(
            score=node.score + (config.discount**depth) * breakdown["total"],
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

    def plan(self) -> tuple[tuple[int, ...], float]:
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
        first_step_nodes = [
            self._simulate_scan(root, action, 0) for action in range(NUM_BANDS)
        ]
        first_step_nodes.sort(
            key=lambda node: (node.score, tuple(-x for x in node.sequence)),
            reverse=True,
        )
        beam = first_step_nodes[: self.config.beam_width]
        for depth in range(1, self.config.depth):
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
        self._last_root = root
        self._last_global_belief = global_belief
        self._last_first_step_nodes = first_step_nodes
        self.last_plan = best.sequence
        self.last_plan_score = float(best.score)
        return best.sequence, float(best.score)

    def _complete_alternative_plan(
        self, greedy_action: int
    ) -> tuple[tuple[int, ...], float] | None:
        alternatives = [
            node
            for node in self._last_first_step_nodes
            if node.sequence[0] != greedy_action
        ][: self.config.exploration_top_k]
        if not alternatives:
            return None
        selected_index = int(self._exploration_rng.integers(len(alternatives)))
        beam = [alternatives[selected_index]]
        for depth in range(1, self.config.depth):
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
        selected = beam[0]
        return selected.sequence, float(selected.score)

    def _top_contributing_emitters(self, selected_band: int, limit: int = 3):
        global_belief = self._last_global_belief or {}
        rows = np.asarray(global_belief.get("track_features", []), dtype=np.float32)
        mask = np.asarray(global_belief.get("track_mask", []), dtype=np.float32)
        track_ids = list(global_belief.get("track_ids", []))
        if rows.ndim != 2 or rows.shape[0] == 0:
            return []
        contributors = []
        valid_row = 0
        for row_index, row in enumerate(rows):
            if row_index >= len(mask) or mask[row_index] <= 0.5 or row[CONFIRMED] <= 0.5:
                continue
            probability = float(np.clip(row[selected_band], 0.0, 1.0))
            track_id = (
                track_ids[valid_row]
                if valid_row < len(track_ids)
                else f"track_{row_index}"
            )
            valid_row += 1
            if probability <= 0.0:
                continue
            contributors.append(
                {
                    "track_id": track_id,
                    "p_band": probability,
                    "uncertainty": float(np.clip(row[PREDICTION_UNCERTAINTY], 0, 1)),
                    "behaviour_change": float(np.clip(row[BEHAVIOUR_CHANGE], 0, 1)),
                    "identity_novelty": float(np.clip(row[IDENTITY_NOVELTY], 0, 1)),
                    "quality": float(np.clip(row[OBSERVATION_QUALITY], 0, 1)),
                }
            )
        contributors.sort(key=lambda item: item["p_band"], reverse=True)
        return contributors[:limit]

    def select_band(self) -> int:
        greedy_sequence, greedy_plan_score = self.plan()
        sequence = greedy_sequence
        plan_score = greedy_plan_score
        selection_mode = "greedy"
        if (
            self.config.exploration_probability > 0.0
            and self._exploration_rng.random() < self.config.exploration_probability
        ):
            alternative = self._complete_alternative_plan(int(greedy_sequence[0]))
            if alternative is not None:
                sequence, plan_score = alternative
                selection_mode = "exploratory_alternative"
                self.last_plan = sequence
                self.last_plan_score = float(plan_score)
        action = int(sequence[0])
        if not 0 <= action < NUM_BANDS:
            raise RuntimeError(f"Emitter-aware planner returned invalid action {action}")
        assert self._last_root is not None
        breakdown = self._score_breakdown(self._last_root, action)
        emitter = self._last_root.emitter
        self.last_decision_trace = {
            "selected_band": action,
            "selection_mode": selection_mode,
            "exploration_probability": float(self.config.exploration_probability),
            "planned_sequence": list(sequence),
            "greedy_planned_sequence": list(greedy_sequence),
            "score": breakdown,
            "plan_total_score": float(plan_score),
            "greedy_plan_total_score": float(greedy_plan_score),
            "band_features": {
                "belief": float(self._last_root.belief[action]),
                "scan_age": float(self._last_root.scan_age[action]),
                "band_uncertainty": float(self._last_root.uncertainty[action]),
                "dominant_emitter_probability": float(
                    emitter.dominant_emitter_probability[action]
                ),
                "supporting_emitter_count": int(
                    round(float(emitter.supporting_emitter_count[action]))
                ),
            },
            "top_contributing_emitters": self._top_contributing_emitters(action),
        }
        return action
