"""Regime-routed Smart Scan mixture of experts and adaptive exploration.

All runtime decisions use observable Track 2 state. Simulator truth may be used
to build offline expert-outcome labels, but is never accepted by this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scheduler.baselines import BaseScheduler
from scheduler.emitter_aware_predictive import (
    BEHAVIOUR_CHANGE,
    CONFIRMED,
    IDENTITY_NOVELTY,
    OBSERVATION_QUALITY,
    EmitterAwareModelPredictivePlanner,
    EmitterAwarePlannerConfig,
    derive_emitter_band_features,
)
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import (
    DEFAULT_MAX_SCAN_AGE,
    DEFAULT_MODEL_PATH,
    Track2Runtime,
)


class Regime(str, Enum):
    EASY = "easy"
    INTERMEDIATE = "intermediate"
    COMPLEX = "complex"
    DYNAMIC = "dynamic_unreliable"
    UNKNOWN = "unknown_ood"


ROUTER_FEATURE_NAMES = (
    "band_entropy",
    "top1_top2_ambiguity",
    "mean_uncertainty",
    "max_uncertainty",
    "plausible_emitter_fraction",
    "competing_band_fraction",
    "behaviour_change",
    "identity_novelty",
    "prediction_disagreement",
    "mean_observation_quality",
    "prediction_error",
    "scan_staleness",
    "candidate_track_fraction",
    "ood_score",
)


def _safe_array(value, length=NUM_BANDS) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != length:
        return np.zeros(length, dtype=np.float32)
    return np.clip(array, 0.0, 1.0)


@dataclass
class ComplexityFeatures:
    vector: np.ndarray
    band_belief: np.ndarray
    scan_age: np.ndarray
    band_uncertainty: np.ndarray
    band_entropy: np.ndarray
    disagreement: np.ndarray
    behaviour: np.ndarray
    novelty: np.ndarray
    quality: np.ndarray
    prediction_error: np.ndarray
    exploration_priority: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(self.vector[index])
            for index, name in enumerate(ROUTER_FEATURE_NAMES)
        }


def extract_complexity_features(global_belief: dict) -> ComplexityFeatures:
    """Build router and exploration features from observable Track 2 state."""
    belief = _safe_array(global_belief.get("band_belief", np.zeros(NUM_BANDS)))
    age = _safe_array(global_belief.get("scan_age", np.zeros(NUM_BANDS)))
    uncertainty = _safe_array(
        global_belief.get("band_uncertainty", np.zeros(NUM_BANDS))
    )
    prediction_error = _safe_array(
        global_belief.get("prediction_error", np.zeros(NUM_BANDS))
    )
    binary_entropy = -(
        belief * np.log(np.clip(belief, 1e-8, 1.0))
        + (1.0 - belief) * np.log(np.clip(1.0 - belief, 1e-8, 1.0))
    ) / np.log(2.0)

    derived = derive_emitter_band_features(global_belief)
    behaviour = np.clip(
        derived.emitter_weighted_behaviour_change
        * derived.emitter_weighted_quality,
        0.0,
        1.0,
    )
    novelty = np.clip(
        derived.emitter_weighted_identity_novelty
        * derived.emitter_weighted_quality,
        0.0,
        1.0,
    )
    quality = np.clip(derived.emitter_weighted_quality, 0.0, 1.0)

    rows = np.asarray(global_belief.get("track_features", []), dtype=np.float32)
    mask = np.asarray(global_belief.get("track_mask", []), dtype=np.float32).reshape(-1)
    valid_rows = np.empty((0, NUM_BANDS), dtype=np.float32)
    if rows.ndim == 2 and rows.shape[1] >= 48 and rows.shape[0] == mask.size:
        valid = (mask > 0.5) & (rows[:, CONFIRMED] > 0.5)
        valid_rows = np.clip(rows[valid, :NUM_BANDS], 0.0, 1.0)
    disagreement = (
        valid_rows.std(axis=0).astype(np.float32)
        if valid_rows.shape[0] > 1
        else np.zeros(NUM_BANDS, dtype=np.float32)
    )

    sorted_belief = np.sort(belief)[::-1]
    top1 = float(sorted_belief[0])
    top2 = float(sorted_belief[1]) if sorted_belief.size > 1 else 0.0
    probability_mass = float(belief.sum())
    distribution = belief / probability_mass if probability_mass > 1e-8 else None
    normalized_entropy = (
        float(
            -(distribution * np.log(np.clip(distribution, 1e-8, 1.0))).sum()
            / np.log(NUM_BANDS)
        )
        if distribution is not None
        else 1.0
    )
    ambiguity = 1.0 - float(np.clip(top1 - top2, 0.0, 1.0))
    competing = float(np.mean(belief >= max(0.10, 0.65 * top1))) if top1 else 1.0
    plausible_emitters = float(min(valid_rows.shape[0] / 8.0, 1.0))
    mean_quality = float(quality[derived.supporting_emitter_count > 0].mean()) if np.any(
        derived.supporting_emitter_count > 0
    ) else 0.0

    candidate = global_belief.get("candidate_summary", {}) or {}
    candidate_fraction = float(
        np.clip(float(candidate.get("count", 0.0)) / 4.0, 0.0, 1.0)
    )
    candidate_novelty = float(np.clip(candidate.get("max_novelty", 0.0), 0.0, 1.0))
    candidate_uncertainty = float(
        np.clip(candidate.get("mean_uncertainty", 0.0), 0.0, 1.0)
    )
    candidate_quality = float(
        np.clip(candidate.get("mean_quality", 0.0), 0.0, 1.0)
    )
    mean_novelty = float(novelty.max(initial=0.0))
    mean_behaviour = float(behaviour.max(initial=0.0))
    observed_errors = prediction_error[prediction_error > 0.02]
    mean_prediction_error = (
        float(observed_errors.mean()) if observed_errors.size else 0.0
    )
    disagreement_score = float(disagreement.max(initial=0.0))
    ood_score = float(
        np.clip(
            max(
                candidate_novelty * (0.5 + 0.5 * (1.0 - candidate_quality)),
                0.35 * mean_novelty
                + 0.25 * mean_prediction_error
                + 0.20 * disagreement_score
                + 0.20 * candidate_uncertainty,
            ),
            0.0,
            1.0,
        )
    )

    vector = np.asarray(
        [
            normalized_entropy,
            ambiguity,
            float(uncertainty.mean()),
            float(uncertainty.max(initial=0.0)),
            plausible_emitters,
            competing,
            mean_behaviour,
            mean_novelty,
            disagreement_score,
            mean_quality,
            mean_prediction_error,
            float(age.mean()),
            candidate_fraction,
            ood_score,
        ],
        dtype=np.float32,
    )
    exploration_priority = np.clip(
        0.30 * uncertainty
        + 0.28 * age
        + 0.14 * novelty
        + 0.14 * behaviour
        + 0.14 * prediction_error,
        0.0,
        1.0,
    ).astype(np.float32)
    return ComplexityFeatures(
        vector=vector,
        band_belief=belief,
        scan_age=age,
        band_uncertainty=uncertainty,
        band_entropy=binary_entropy.astype(np.float32),
        disagreement=disagreement,
        behaviour=behaviour.astype(np.float32),
        novelty=novelty.astype(np.float32),
        quality=quality.astype(np.float32),
        prediction_error=prediction_error,
        exploration_priority=exploration_priority,
    )


@dataclass(frozen=True)
class RouterThresholds:
    easy_top_probability: float = 0.68
    easy_margin: float = 0.20
    easy_entropy: float = 0.68
    easy_uncertainty: float = 0.45
    complex_score: float = 0.50
    complex_uncertainty: float = 0.50
    complex_ambiguity: float = 0.82
    dynamic_prediction_error: float = 0.22
    dynamic_behaviour_change: float = 0.24
    dynamic_low_quality: float = 0.28
    dynamic_uncertainty: float = 0.50
    tree_min_quality: float = 0.35
    tree_max_prediction_error: float = 0.22
    ood_score: float = 0.78
    ood_candidate_fraction: float = 0.25


class RuleBasedRegimeRouter:
    def __init__(self, thresholds: RouterThresholds | None = None):
        self.thresholds = thresholds or RouterThresholds()
        self.last_trace: dict = {}

    def route(self, features: ComplexityFeatures) -> Regime:
        values = features.as_dict()
        sorted_belief = np.sort(features.band_belief)[::-1]
        top1 = float(sorted_belief[0])
        top2 = float(sorted_belief[1]) if sorted_belief.size > 1 else 0.0
        margin = top1 - top2
        complexity = float(
            0.22 * values["band_entropy"]
            + 0.16 * values["top1_top2_ambiguity"]
            + 0.14 * values["mean_uncertainty"]
            + 0.08 * values["plausible_emitter_fraction"]
            + 0.10 * values["competing_band_fraction"]
            + 0.09 * values["behaviour_change"]
            + 0.07 * values["identity_novelty"]
            + 0.09 * values["prediction_disagreement"]
            + 0.05 * values["prediction_error"]
        )
        no_confirmed_support = (
            values["plausible_emitter_fraction"] <= 0.0 and top1 < 0.05
        )
        strong_candidate_ood = (
            values["candidate_track_fraction"]
            >= self.thresholds.ood_candidate_fraction
            and values["ood_score"] >= self.thresholds.ood_score
        )
        dynamic_signals = {
            "prediction_error": values["prediction_error"]
            >= self.thresholds.dynamic_prediction_error,
            # Track 2's behaviour score is useful context, but benchmarked as
            # too noisy to trigger a planner switch by itself.
            "behaviour_change": False,
            "low_quality_uncertainty": (
                values["mean_observation_quality"]
                <= self.thresholds.dynamic_low_quality
                and values["mean_uncertainty"]
                >= self.thresholds.dynamic_uncertainty
            ),
            "disagreement_uncertainty": (
                values["prediction_disagreement"] >= 0.25
                and values["mean_uncertainty"] >= 0.35
            ),
        }
        dynamic_state = any(dynamic_signals.values())
        if strong_candidate_ood:
            regime = Regime.UNKNOWN
            reason = "a persistent candidate has strong observable OOD evidence"
        elif no_confirmed_support or dynamic_state:
            regime = Regime.DYNAMIC
            reason = (
                "no confirmed model support; use observation-driven UCB discovery"
                if no_confirmed_support
                else "predictions are changing or unreliable; use adaptive UCB"
            )
        elif (
            top1 >= self.thresholds.easy_top_probability
            and margin >= self.thresholds.easy_margin
            and values["band_entropy"] <= self.thresholds.easy_entropy
            and values["mean_uncertainty"] <= self.thresholds.easy_uncertainty
            and values["behaviour_change"] < 0.30
            and values["prediction_error"] < 0.15
        ):
            regime = Regime.EASY
            reason = "one band dominates with low uncertainty and stable behaviour"
        elif (
            complexity >= self.thresholds.complex_score
            and values["plausible_emitter_fraction"] > 0.0
            and values["mean_observation_quality"]
            >= self.thresholds.tree_min_quality
            and values["prediction_error"]
            < self.thresholds.tree_max_prediction_error
            and values["behaviour_change"]
            < self.thresholds.dynamic_behaviour_change
            and (
                values["mean_uncertainty"] >= self.thresholds.complex_uncertainty
                or values["top1_top2_ambiguity"]
                >= self.thresholds.complex_ambiguity
                or values["prediction_disagreement"] >= 0.25
            )
        ):
            regime = Regime.COMPLEX
            reason = "known-state ambiguity is high enough to justify a belief tree"
        else:
            regime = Regime.INTERMEDIATE
            reason = "multiple considerations justify short-horizon planning"
        self.last_trace = {
            "regime": regime.value,
            "reason": reason,
            "complexity_score": complexity,
            "top_probability": top1,
            "top_margin": margin,
            "ood_score": values["ood_score"],
            "dynamic_state": bool(dynamic_state or no_confirmed_support),
            "dynamic_signals": {
                name: bool(active) for name, active in dynamic_signals.items()
            },
            "no_confirmed_support": bool(no_confirmed_support),
            "strong_candidate_ood": bool(strong_candidate_ood),
        }
        return regime


class RouterDataset:
    """Offline state/outcome records for a learned mixture-of-experts gate."""

    def __init__(
        self,
        expert_names: Sequence[str] = ("fast", "beam", "tree", "safe"),
        latency_penalty_per_ms: float = 0.003,
    ):
        self.expert_names = tuple(expert_names)
        self.latency_penalty_per_ms = float(latency_penalty_per_ms)
        self.feature_rows: list[np.ndarray] = []
        self.return_rows: list[np.ndarray] = []
        self.latency_rows: list[np.ndarray] = []

    def add(
        self,
        features: ComplexityFeatures | np.ndarray,
        expert_returns: Mapping[str, float],
        expert_latency_ms: Mapping[str, float] | None = None,
    ):
        vector = features.vector if isinstance(features, ComplexityFeatures) else np.asarray(features)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size != len(ROUTER_FEATURE_NAMES):
            raise ValueError("Router feature row has the wrong width")
        returns = np.asarray(
            [float(expert_returns[name]) for name in self.expert_names],
            dtype=np.float32,
        )
        self.feature_rows.append(vector.copy())
        self.return_rows.append(returns)
        latency = expert_latency_ms or {}
        self.latency_rows.append(
            np.asarray(
                [float(latency.get(name, 0.0)) for name in self.expert_names],
                dtype=np.float32,
            )
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.feature_rows:
            raise ValueError("Router dataset is empty")
        features = np.stack(self.feature_rows)
        returns = np.stack(self.return_rows)
        latencies = np.stack(self.latency_rows)
        cost_adjusted_utility = returns - self.latency_penalty_per_ms * latencies
        labels = cost_adjusted_utility.argmax(axis=1).astype(np.int64)
        return features, labels, returns

    def latencies(self) -> np.ndarray:
        if not self.latency_rows:
            raise ValueError("Router dataset is empty")
        return np.stack(self.latency_rows)


class LearnedRegimeRouter:
    """Small trainable softmax gate: observable state -> best expert."""

    def __init__(self, expert_names: Sequence[str] = ("fast", "beam", "tree", "safe")):
        self.expert_names = tuple(expert_names)
        self.mean = np.zeros(len(ROUTER_FEATURE_NAMES), dtype=np.float32)
        self.scale = np.ones(len(ROUTER_FEATURE_NAMES), dtype=np.float32)
        self.weights = np.zeros(
            (len(ROUTER_FEATURE_NAMES) + 1, len(self.expert_names)), dtype=np.float32
        )
        self.fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 600,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
        balance_classes: bool = True,
    ) -> "LearnedRegimeRouter":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != len(ROUTER_FEATURE_NAMES):
            raise ValueError("Learned router expects [N,14] observable features")
        if x.shape[0] != y.size or x.shape[0] == 0:
            raise ValueError("Router features and labels must be non-empty and aligned")
        if np.any((y < 0) | (y >= len(self.expert_names))):
            raise ValueError("Router label is outside the expert set")
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-5)
        normalized = (x - self.mean) / self.scale
        design = np.concatenate(
            [normalized, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1
        )
        target = np.eye(len(self.expert_names), dtype=np.float32)[y]
        sample_weights = np.ones(x.shape[0], dtype=np.float32)
        if balance_classes:
            counts = np.bincount(y, minlength=len(self.expert_names)).astype(np.float32)
            present = counts > 0
            class_weights = np.ones(len(self.expert_names), dtype=np.float32)
            class_weights[present] = x.shape[0] / (
                max(1, int(present.sum())) * counts[present]
            )
            sample_weights = class_weights[y]
            sample_weights /= max(float(sample_weights.mean()), 1e-8)
        self.weights.fill(0.0)
        for _ in range(int(epochs)):
            logits = design @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = design.T @ (
                (probabilities - target) * sample_weights[:, None]
            ) / x.shape[0]
            gradient[:-1] += l2 * self.weights[:-1]
            self.weights -= learning_rate * gradient
        self.fitted = True
        return self

    def predict_proba(self, features: ComplexityFeatures | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Learned router has not been fitted")
        vector = features.vector if isinstance(features, ComplexityFeatures) else features
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        normalized = (vector - self.mean) / self.scale
        design = np.concatenate([normalized, np.ones(1, dtype=np.float32)])
        logits = design @ self.weights
        logits -= logits.max()
        probabilities = np.exp(logits)
        return (probabilities / probabilities.sum()).astype(np.float32)

    def predict_expert(self, features: ComplexityFeatures | np.ndarray) -> str:
        return self.expert_names[int(self.predict_proba(features).argmax())]

    def save(self, path: str | Path):
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted router")
        np.savez(
            Path(path),
            expert_names=np.asarray(self.expert_names),
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRegimeRouter":
        data = np.load(Path(path), allow_pickle=False)
        router = cls(tuple(str(value) for value in data["expert_names"]))
        router.mean = data["mean"].astype(np.float32)
        router.scale = data["scale"].astype(np.float32)
        router.weights = data["weights"].astype(np.float32)
        router.fitted = True
        return router


class GreedyBeliefScheduler(BaseScheduler):
    """Fast expert for obvious low-complexity states."""

    def __init__(self, num_bands: int, runtime: Track2Runtime):
        super().__init__(num_bands)
        self.runtime = runtime
        self.timestamp = 0.0

    def select_band(self) -> int:
        return int(np.argmax(self.runtime.get_band_belief()))

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=self.timestamp)
        self.timestamp += 1.0


class SafeExplorationScheduler(BaseScheduler):
    """OOD fallback that prioritizes observable uncertainty and coverage."""

    def __init__(self, num_bands: int, runtime: Track2Runtime):
        super().__init__(num_bands)
        self.runtime = runtime
        self.last_priority = np.zeros(num_bands, dtype=np.float32)
        self.timestamp = 0.0

    def select_band(self) -> int:
        features = extract_complexity_features(self.runtime.get_global_belief())
        self.last_priority = np.clip(
            features.exploration_priority + 0.10 * (1.0 - features.band_belief),
            0.0,
            1.0,
        )
        return int(np.argmax(self.last_priority))

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=self.timestamp)
        self.timestamp += 1.0


class ObservableRegimeMonitor:
    """Rolling observable history that separates clean/stable and noisy sensing."""

    def __init__(
        self,
        num_bands: int,
        credible_quality: float = 0.40,
        history_window: int = 80,
    ):
        self.num_bands = int(num_bands)
        self.credible_quality = float(credible_quality)
        self.history_window = int(history_window)
        self.counts = np.zeros(num_bands, dtype=np.int64)
        self.hit_rates = np.zeros(num_bands, dtype=np.float64)
        self.volatility = np.zeros(num_bands, dtype=np.float64)
        self.observation_history = deque(maxlen=self.history_window)
        self.credible_band_history = deque(maxlen=max(20, self.history_window // 2))
        self.band_outcome_history = [deque(maxlen=8) for _ in range(num_bands)]
        self.history_regime = "uncertain"
        self.total_updates = 0
        self.detected_count = 0
        self.credible_detection_count = 0
        self.low_quality_detection_count = 0
        self.detected_quality_sum = 0.0

    def update(self, band: int, obs_dict: dict | None):
        if obs_dict is None:
            return
        index = int(band)
        detected = bool(obs_dict.get("detected", False))
        quality = float(
            np.asarray(obs_dict.get("quality", [0.0]), dtype=np.float32).reshape(-1)[0]
        )
        credible = float(detected and quality >= self.credible_quality)
        self.observation_history.append((index, detected, quality, bool(credible)))
        self.band_outcome_history[index].append(int(credible))
        if credible:
            self.credible_band_history.append(index)
        if self.counts[index] > 0:
            surprise = abs(credible - self.hit_rates[index])
            self.volatility[index] = 0.75 * self.volatility[index] + 0.25 * surprise
        learning_rate = 0.35 if self.counts[index] < 3 else 0.20
        self.hit_rates[index] = (
            (1.0 - learning_rate) * self.hit_rates[index]
            + learning_rate * credible
        )
        self.counts[index] += 1
        self.total_updates += 1
        if detected:
            self.detected_count += 1
            self.detected_quality_sum += quality
            if credible:
                self.credible_detection_count += 1
            else:
                self.low_quality_detection_count += 1
        scores = self._history_scores()
        if self.history_regime == "clean_stationary":
            if (
                scores["clean_stationary_score"] < 0.48
                or scores["hit_band_concentration"] < 0.66
            ):
                self.history_regime = "uncertain"
        elif self.history_regime == "noisy":
            if scores["noisy_score"] < 0.35:
                self.history_regime = "uncertain"
        elif self.history_regime == "agile_hopping":
            if (
                scores["outcome_flip_rate"] < 0.25
                or scores["hit_band_concentration"] >= 0.70
            ):
                self.history_regime = "uncertain"
        elif scores["noisy_score"] >= 0.45:
            self.history_regime = "noisy"
        elif (
            scores["clean_stationary_score"] >= 0.60
            and scores["hit_band_concentration"] >= 0.70
        ):
            self.history_regime = "clean_stationary"
        elif (
            scores["history_samples"] >= 12.0
            and scores["history_detections"] >= 3.0
            and scores["outcome_flip_rate"] >= 0.28
            and scores["hit_band_concentration"] < 0.65
            and scores["recent_mean_detected_quality"] >= 0.45
        ):
            self.history_regime = "agile_hopping"

    def _history_scores(self) -> dict[str, float]:
        history = list(self.observation_history)
        detected_rows = [row for row in history if row[1]]
        recent_mean_quality = (
            float(np.mean([row[2] for row in detected_rows]))
            if detected_rows
            else 0.0
        )
        recent_low_quality_ratio = (
            float(
                np.mean([row[2] < self.credible_quality for row in detected_rows])
            )
            if detected_rows
            else 0.0
        )
        flip_values = []
        for outcomes in self.band_outcome_history:
            values = list(outcomes)
            if len(values) >= 3:
                flip_values.extend(
                    float(left != right)
                    for left, right in zip(values[:-1], values[1:], strict=True)
                )
        outcome_flip_rate = float(np.mean(flip_values)) if flip_values else 0.5
        hit_band_concentration = 0.0
        if self.credible_band_history:
            counts = np.bincount(
                np.asarray(self.credible_band_history, dtype=np.int64),
                minlength=self.num_bands,
            ).astype(np.float64)
            distribution = counts[counts > 0.0] / counts.sum()
            entropy = float(-(distribution * np.log(distribution)).sum())
            hit_band_concentration = 1.0 - entropy / np.log(self.num_bands)
        evidence = min(len(history) / 40.0, 1.0) * min(
            len(detected_rows) / 10.0, 1.0
        )
        quality_clean = float(np.clip((recent_mean_quality - 0.58) / 0.20, 0.0, 1.0))
        alarm_clean = float(
            np.clip((0.44 - recent_low_quality_ratio) / 0.24, 0.0, 1.0)
        )
        band_stability = float(np.clip((0.55 - outcome_flip_rate) / 0.40, 0.0, 1.0))
        concentration = float(
            np.clip((hit_band_concentration - 0.10) / 0.40, 0.0, 1.0)
        )
        clean_stationary_score = evidence * (
            0.35 * quality_clean
            + 0.25 * alarm_clean
            + 0.20 * band_stability
            + 0.20 * concentration
        )
        quality_noise = float(
            np.clip((0.52 - recent_mean_quality) / 0.24, 0.0, 1.0)
        )
        alarm_noise = float(
            np.clip((recent_low_quality_ratio - 0.48) / 0.28, 0.0, 1.0)
        )
        noisy_score = evidence * (0.55 * quality_noise + 0.45 * alarm_noise)
        return {
            "history_samples": float(len(history)),
            "history_detections": float(len(detected_rows)),
            "recent_mean_detected_quality": recent_mean_quality,
            "recent_low_quality_detection_ratio": recent_low_quality_ratio,
            "outcome_flip_rate": outcome_flip_rate,
            "hit_band_concentration": hit_band_concentration,
            "clean_stationary_score": float(clean_stationary_score),
            "noisy_score": float(noisy_score),
        }

    def snapshot(self) -> dict[str, float | str]:
        revisited = self.counts >= 2
        observed = self.counts > 0
        mean_volatility = (
            float(self.volatility[revisited].mean()) if np.any(revisited) else 0.0
        )
        history_scores = self._history_scores()
        return {
            "updates": float(self.total_updates),
            "detected_count": float(self.detected_count),
            "coverage": float(np.mean(observed)),
            "revisited_band_fraction": float(np.mean(revisited)),
            "mean_volatility": mean_volatility,
            "max_volatility": float(self.volatility.max(initial=0.0)),
            "low_quality_detection_ratio": (
                float(self.low_quality_detection_count / self.detected_count)
                if self.detected_count
                else 0.0
            ),
            "mean_detected_quality": (
                float(self.detected_quality_sum / self.detected_count)
                if self.detected_count
                else 0.0
            ),
            "credible_detection_ratio": (
                float(self.credible_detection_count / self.total_updates)
                if self.total_updates
                else 0.0
            ),
            "history_regime": self.history_regime,
            **history_scores,
        }


class ObservableDiscountedUCBScheduler(BaseScheduler):
    """Non-stationary UCB with reliability-gated Track-2 guidance.

    Counts slowly decay so a band that moved or went quiet does not dominate
    forever.  The update deliberately ignores simulator reward/truth: a strong,
    high-quality detection is useful evidence, while a miss supplies only a
    small value. Track-2 future-band belief is fused only when confirmed tracks,
    observation quality, uncertainty, and prediction error make it credible.
    """

    def __init__(
        self,
        num_bands: int,
        runtime: Track2Runtime,
        decay: float = 0.985,
        value_learning_rate: float = 0.20,
        exploration_scale: float = 0.75,
        neural_guidance_scale: float = 0.65,
        manage_runtime: bool = False,
    ):
        super().__init__(num_bands)
        self.runtime = runtime
        self.decay = float(decay)
        self.value_learning_rate = float(value_learning_rate)
        self.exploration_scale = float(exploration_scale)
        self.neural_guidance_scale = float(neural_guidance_scale)
        if self.neural_guidance_scale < 0.0:
            raise ValueError("neural_guidance_scale must be non-negative")
        self.manage_runtime = bool(manage_runtime)
        self.timestamp = 0.0
        self.counts = np.zeros(num_bands, dtype=np.float64)
        self.values = np.zeros(num_bands, dtype=np.float64)
        self.total_observations = 0.0
        self.neural_brier_ema = 0.25
        self.neural_calibration_evidence = 0
        self.neural_positive_evidence = 0
        self.neural_high_skill_streak = 0
        self._last_track2_probability: float | None = None
        self.last_trace: dict = {}

    def select_band(self) -> int:
        global_belief = self.runtime.get_global_belief()
        features = extract_complexity_features(global_belief)
        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            self._last_track2_probability = float(features.band_belief[band])
            self.last_trace = {
                "mode": "initial_coverage",
                "selected_band": band,
                "track2_band_probability": self._last_track2_probability,
                "neural_guidance_weight": 0.0,
            }
            return band
        feature_values = features.as_dict()
        bonus = self.exploration_scale * np.sqrt(
            np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
        )
        support = float(
            np.clip(4.0 * feature_values["plausible_emitter_fraction"], 0.0, 1.0)
        )
        quality = float(feature_values["mean_observation_quality"])
        uncertainty = float(feature_values["mean_uncertainty"])
        prediction_error = float(feature_values["prediction_error"])
        belief_sharpness = float(
            np.clip(1.0 - feature_values["band_entropy"], 0.0, 1.0)
        )
        state_reliability = float(
            np.clip(
                support
                * (0.25 + 0.75 * quality)
                * (1.0 - 0.70 * uncertainty)
                * (1.0 - 0.80 * prediction_error)
                * (0.40 + 0.60 * belief_sharpness),
                0.0,
                1.0,
            )
        )
        calibration_maturity = float(
            np.clip(self.neural_calibration_evidence / 40.0, 0.0, 1.0)
        )
        calibration_skill = float(
            np.clip(1.0 - self.neural_brier_ema / 0.25, 0.0, 1.0)
        )
        positive_maturity = float(
            np.clip(self.neural_positive_evidence / 20.0, 0.0, 1.0)
        )
        calibration_gate = float(
            calibration_skill >= 0.90 and self.neural_high_skill_streak >= 20
        )
        neural_reliability = (
            state_reliability
            * calibration_maturity
            * positive_maturity
            * calibration_gate
        )
        neural_weight = self.neural_guidance_scale * neural_reliability
        neural_guidance = neural_weight * features.band_belief
        scores = (
            self.values
            + bonus
            + neural_guidance
            + 0.10 * features.scan_age
            + 0.05 * features.band_uncertainty
        )
        band = int(np.argmax(scores))
        self._last_track2_probability = float(features.band_belief[band])
        self.last_trace = {
            "mode": "discounted_quality_ucb",
            "selected_band": band,
            "value": float(self.values[band]),
            "exploration_bonus": float(bonus[band]),
            "track2_band_probability": float(features.band_belief[band]),
            "neural_guidance_scale": self.neural_guidance_scale,
            "neural_reliability": neural_reliability,
            "state_reliability": state_reliability,
            "online_calibration_maturity": calibration_maturity,
            "online_calibration_skill": calibration_skill,
            "online_calibration_gate": calibration_gate,
            "positive_evidence_maturity": positive_maturity,
            "high_skill_streak": self.neural_high_skill_streak,
            "online_brier_error": self.neural_brier_ema,
            "neural_guidance_weight": neural_weight,
            "neural_guidance_contribution": float(neural_guidance[band]),
            "reliability_inputs": {
                "confirmed_support": support,
                "mean_quality": quality,
                "mean_uncertainty": uncertainty,
                "prediction_error": prediction_error,
                "belief_sharpness": belief_sharpness,
            },
            "score": float(scores[band]),
        }
        return band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        self.timestamp += 1.0
        self.counts *= self.decay
        self.total_observations = self.total_observations * self.decay + 1.0
        self.counts[int(band)] += 1.0
        if obs_dict is None:
            return
        detected = bool(obs_dict.get("detected", False))
        quality = float(
            np.asarray(obs_dict.get("quality", [0.0]), dtype=np.float32).reshape(-1)[0]
        )
        if self._last_track2_probability is not None:
            observable_target = float(detected and quality >= 0.40)
            self.neural_positive_evidence += int(observable_target > 0.5)
            brier_error = (self._last_track2_probability - observable_target) ** 2
            self.neural_brier_ema = (
                0.90 * self.neural_brier_ema + 0.10 * float(brier_error)
            )
            self.neural_calibration_evidence += 1
            updated_skill = float(
                np.clip(1.0 - self.neural_brier_ema / 0.25, 0.0, 1.0)
            )
            self.neural_high_skill_streak = (
                self.neural_high_skill_streak + 1
                if updated_skill >= 0.90
                else 0
            )
        if self.manage_runtime:
            self.runtime.update(obs_dict, timestamp=self.timestamp - 1.0)
        observable_value = (
            float(np.clip(0.10 + 0.90 * quality, 0.0, 1.0))
            if detected
            else 0.02
        )
        index = int(band)
        self.values[index] = (
            (1.0 - self.value_learning_rate) * self.values[index]
            + self.value_learning_rate * observable_value
        )


@dataclass(frozen=True)
class AdaptiveExplorationConfig:
    easy_epsilon: float = 0.005
    intermediate_epsilon: float = 0.01
    complex_epsilon: float = 0.03
    dynamic_epsilon: float = 0.0
    unknown_epsilon: float = 0.05
    seed: int = 42

    def __post_init__(self):
        values = (
            self.easy_epsilon,
            self.intermediate_epsilon,
            self.complex_epsilon,
            self.dynamic_epsilon,
            self.unknown_epsilon,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Adaptive exploration probabilities must be in [0,1]")


class AdaptiveExplorationGate:
    def __init__(self, config: AdaptiveExplorationConfig | None = None):
        self.config = config or AdaptiveExplorationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.last_trace: dict = {}

    def probability(self, regime: Regime, features: ComplexityFeatures) -> float:
        base = {
            Regime.EASY: self.config.easy_epsilon,
            Regime.INTERMEDIATE: self.config.intermediate_epsilon,
            Regime.COMPLEX: self.config.complex_epsilon,
            Regime.DYNAMIC: self.config.dynamic_epsilon,
            Regime.UNKNOWN: self.config.unknown_epsilon,
        }[regime]
        changing = max(
            float(features.behaviour.max(initial=0.0)),
            float(features.novelty.max(initial=0.0)),
        )
        uncertainty = float(features.band_uncertainty.mean())
        multiplier = 0.75 + 0.25 * max(changing, uncertainty)
        return float(np.clip(base * multiplier, 0.0, 1.0))

    def choose(
        self,
        proposed_band: int,
        regime: Regime,
        features: ComplexityFeatures,
    ) -> int:
        epsilon = self.probability(regime, features)
        exploratory = bool(self.rng.random() < epsilon)
        selected = int(proposed_band)
        if exploratory:
            priority = np.asarray(features.exploration_priority, dtype=np.float64).copy()
            priority[int(proposed_band)] = 0.0
            priority = np.maximum(priority, 1e-6)
            priority[int(proposed_band)] = 0.0
            total = float(priority.sum())
            if total > 0.0:
                selected = int(self.rng.choice(NUM_BANDS, p=priority / total))
            else:
                exploratory = False
        self.last_trace = {
            "epsilon": epsilon,
            "explored": exploratory,
            "proposed_band": int(proposed_band),
            "selected_band": selected,
            "selected_priority": float(features.exploration_priority[selected]),
        }
        return selected


class AdaptiveMixtureOfExpertsScheduler(BaseScheduler):
    """Cost-aware hybrid planner with conservative learned-policy use.

    Strong OOD uses safe discovery. Rolling observation history sends clean,
    stationary sensing to the belief tree and sustained noisy sensing to
    MPP-60. Reliable complex states use the tree, while unsupported/dynamic
    states use quality-aware discounted UCB. RL remains explicit opt-in only.
    """

    def __init__(
        self,
        num_bands: int,
        rl_model=None,
        attention_rl_model=None,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        rule_router: RuleBasedRegimeRouter | None = None,
        learned_router: LearnedRegimeRouter | None = None,
        exploration: AdaptiveExplorationConfig | None = None,
        tree_branch_width: int = 4,
        allow_unvalidated_rl: bool = False,
    ):
        if num_bands != NUM_BANDS:
            raise ValueError(f"Adaptive routing requires exactly {NUM_BANDS} bands")
        super().__init__(num_bands)
        from scheduler.belief_tree import ObservationDependentBeliefTreePlanner
        from scheduler.emitter_rl import AttentionRLScheduler, ExplorationBeliefRLScheduler
        from scheduler.model_predictive import ModelPredictiveScheduler

        self.runtime = runtime or Track2Runtime(
            model_path, max_scan_age=max_scan_age
        )
        self.fast_expert = GreedyBeliefScheduler(num_bands, self.runtime)
        self.mpp_expert = ModelPredictiveScheduler(
            num_bands,
            runtime=self.runtime,
            max_scan_age=max_scan_age,
        )
        beam_config = EmitterAwarePlannerConfig(
            track_recency_weight=0.0,
            identity_weight=0.0,
            exploration_probability=0.0,
        )
        self.tree_expert = ObservationDependentBeliefTreePlanner(
            num_bands,
            runtime=self.runtime,
            config=beam_config,
            max_scan_age=max_scan_age,
            branch_width=tree_branch_width,
        )
        # The ordinary beam expert is intentionally separate from the
        # observation tree.  It is roughly an order of magnitude cheaper and
        # handles the large middle of the state distribution.
        self.beam_expert = EmitterAwareModelPredictivePlanner(
            num_bands,
            runtime=self.runtime,
            config=beam_config,
            max_scan_age=max_scan_age,
        )
        self.rl_expert = (
            AttentionRLScheduler(
                num_bands,
                attention_rl_model,
                runtime=self.runtime,
                max_scan_age=max_scan_age,
            )
            if attention_rl_model is not None
            else ExplorationBeliefRLScheduler(
                num_bands, rl_model, runtime=self.runtime, max_scan_age=max_scan_age
            )
            if rl_model is not None
            else None
        )
        self.safe_expert = SafeExplorationScheduler(num_bands, self.runtime)
        self.ucb_expert = ObservableDiscountedUCBScheduler(
            num_bands, self.runtime
        )
        self.regime_monitor = ObservableRegimeMonitor(num_bands)
        self.allow_unvalidated_rl = bool(allow_unvalidated_rl)
        self.rule_router = rule_router or RuleBasedRegimeRouter()
        self.learned_router = learned_router
        self.exploration_gate = AdaptiveExplorationGate(exploration)
        self.calibration_override: str | None = None
        self.calibration_trace: dict = {}
        self.timestamp = 0.0
        self.last_decision_trace: dict = {}

    def begin_scored_phase(self):
        """Freeze high-confidence noisy calibration into resilient Belief Tree.

        Clean/stationary history remains adaptive because the validated
        MPP/tree mixture outperformed either unconditional handoff.
        """
        monitor = self.regime_monitor.snapshot()
        enough_evidence = (
            float(monitor["history_samples"]) >= 40.0
            and float(monitor["history_detections"]) >= 10.0
        )
        # The online noisy-regime hysteresis enters at 0.45. Calibration uses a
        # slightly more sensitive boundary, but stays conservative because a
        # short changing-world sample can temporarily look noisy too.
        noisy = enough_evidence and float(monitor["noisy_score"]) >= 0.35
        self.calibration_override = "noisy_tree" if noisy else None
        self.calibration_trace = {
            "enough_evidence": bool(enough_evidence),
            "selected_override": self.calibration_override or "adaptive",
            "history_regime": str(monitor["history_regime"]),
            "clean_stationary_score": float(monitor["clean_stationary_score"]),
            "noisy_score": float(monitor["noisy_score"]),
            "hit_band_concentration": float(monitor["hit_band_concentration"]),
        }
        return dict(self.calibration_trace)

    def _select_expert(self, regime: Regime, features: ComplexityFeatures):
        if regime is Regime.UNKNOWN:
            return "safe", self.safe_expert, "rule_ood_override"
        if self.calibration_override in {"noisy_mpp", "noisy_tree"}:
            return "tree", self.tree_expert, "calibrated_noisy_tree_guard"
        monitor = self.regime_monitor.snapshot()
        history_regime = monitor["history_regime"]
        if history_regime == "noisy":
            return "tree", self.tree_expert, "history_noisy_tree_guard"
        if history_regime == "clean_stationary":
            return "tree", self.tree_expert, "history_stationary_tree_guard"
        if history_regime == "agile_hopping":
            return "ucb", self.ucb_expert, "history_hopping_ucb_guard"
        if regime is Regime.DYNAMIC:
            return "ucb", self.ucb_expert, "rule_dynamic_override"
        if regime is Regime.COMPLEX:
            return "tree", self.tree_expert, "rule_reliable_complex_override"
        if self.learned_router is not None and self.learned_router.fitted:
            expert_name = self.learned_router.predict_expert(features)
            probabilities = self.learned_router.predict_proba(features)
            if expert_name == "rl" and not (
                self.allow_unvalidated_rl and self.rl_expert is not None
            ):
                return "tree", self.tree_expert, "unvalidated_rl_replaced_by_tree"
            expert = {
                "fast": self.fast_expert,
                "beam": self.beam_expert,
                "mpp": self.mpp_expert,
                "tree": self.tree_expert,
                "ucb": self.ucb_expert,
                "rl": self.rl_expert if self.allow_unvalidated_rl else None,
                "safe": self.safe_expert,
            }.get(expert_name)
            if expert is not None:
                return expert_name, expert, {
                    "source": "learned_router",
                    "probabilities": {
                        name: float(probabilities[index])
                        for index, name in enumerate(self.learned_router.expert_names)
                    },
                }
        return "mpp", self.mpp_expert, "rule_stable_override"

    def select_band(self) -> int:
        global_belief = self.runtime.get_global_belief()
        features = extract_complexity_features(global_belief)
        regime = self.rule_router.route(features)
        expert_name, expert, routing_source = self._select_expert(regime, features)
        proposed = int(expert.select_band())
        history_guard = routing_source in {
            "history_stationary_tree_guard",
        }
        if history_guard:
            selected = proposed
            self.exploration_gate.last_trace = {
                "epsilon": 0.0,
                "explored": False,
                "proposed_band": proposed,
                "selected_band": proposed,
                "selected_priority": float(features.exploration_priority[proposed]),
                "suppressed_by_history_guard": True,
            }
        else:
            selected = self.exploration_gate.choose(proposed, regime, features)
        if not 0 <= selected < NUM_BANDS:
            raise RuntimeError(f"Adaptive mixture returned invalid band {selected}")
        self.last_decision_trace = {
            "selected_band": selected,
            "proposed_band": proposed,
            "regime": regime.value,
            "expert": expert_name,
            "routing_source": routing_source,
            "router": dict(self.rule_router.last_trace),
            "features": features.as_dict(),
            "adaptive_exploration": dict(self.exploration_gate.last_trace),
            "expert_trace": dict(getattr(expert, "last_decision_trace", {})),
            "regime_monitor": self.regime_monitor.snapshot(),
            "calibration": dict(self.calibration_trace),
        }
        return selected

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        self.ucb_expert.update(band, reward, obs_dict)
        self.regime_monitor.update(band, obs_dict)
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=self.timestamp)
        self.timestamp += 1.0
