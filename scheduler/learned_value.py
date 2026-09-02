"""Optional learned utility estimator inside explainable beam planning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from scheduler.emitter_aware_predictive import (
    EmitterAwareModelPredictivePlanner,
    EmitterAwarePlannerConfig,
    _EmitterPlanNode,
)


PATH_VALUE_FEATURES = (
    "belief",
    "uncertainty",
    "scan_age",
    "dominant_emitter",
    "support_fraction",
    "behaviour",
    "novelty",
    "quality",
    "identity_confidence",
    "prefix_depth",
    "repeat",
    "hand_score",
)


def encode_path_candidate(
    node: _EmitterPlanNode, action: int, breakdown: dict[str, float], max_depth: int
) -> np.ndarray:
    emitter = node.emitter
    repeated = float(bool(node.sequence and node.sequence[-1] == action))
    return np.asarray(
        [
            node.belief[action],
            node.uncertainty[action],
            node.scan_age[action],
            emitter.dominant_emitter_probability[action],
            emitter.supporting_emitter_fraction[action],
            emitter.emitter_weighted_behaviour_change[action],
            emitter.emitter_weighted_identity_novelty[action],
            emitter.emitter_weighted_quality[action],
            emitter.emitter_weighted_identity_confidence[action],
            len(node.sequence) / max(1, max_depth),
            repeated,
            breakdown["total"],
        ],
        dtype=np.float32,
    )


class PathValueDataset:
    def __init__(self):
        self.features: list[np.ndarray] = []
        self.targets: list[float] = []

    def add(self, features: np.ndarray, realized_long_term_utility: float):
        row = np.asarray(features, dtype=np.float32).reshape(-1)
        if row.size != len(PATH_VALUE_FEATURES):
            raise ValueError("Path-value feature row has the wrong width")
        self.features.append(row.copy())
        self.targets.append(float(realized_long_term_utility))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.features:
            raise ValueError("Path-value dataset is empty")
        return np.stack(self.features), np.asarray(self.targets, dtype=np.float32)


class SmallPathValueModel:
    """A small one-hidden-layer value model trained with NumPy."""

    def __init__(self, hidden_size: int = 16, seed: int = 42):
        self.hidden_size = int(hidden_size)
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        rng = np.random.default_rng(seed)
        input_size = len(PATH_VALUE_FEATURES)
        self.mean = np.zeros(input_size, dtype=np.float32)
        self.scale = np.ones(input_size, dtype=np.float32)
        self.w1 = rng.normal(0.0, 0.08, (input_size, self.hidden_size)).astype(
            np.float32
        )
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.08, self.hidden_size).astype(np.float32)
        self.b2 = np.float32(0.0)
        self.fitted = False

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 800,
        learning_rate: float = 0.02,
        l2: float = 1e-4,
    ) -> "SmallPathValueModel":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        if x.ndim != 2 or x.shape[1] != len(PATH_VALUE_FEATURES):
            raise ValueError("Path-value model expects [N,12] features")
        if x.shape[0] != y.size or y.size == 0:
            raise ValueError("Value features and targets must be non-empty and aligned")
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-5)
        x = (x - self.mean) / self.scale
        for _ in range(int(epochs)):
            hidden = np.tanh(x @ self.w1 + self.b1)
            prediction = hidden @ self.w2 + self.b2
            error = (prediction - y) / y.size
            grad_w2 = hidden.T @ error + l2 * self.w2
            grad_b2 = error.sum()
            grad_hidden = error[:, None] * self.w2[None, :]
            grad_pre = grad_hidden * (1.0 - hidden**2)
            grad_w1 = x.T @ grad_pre + l2 * self.w1
            grad_b1 = grad_pre.sum(axis=0)
            self.w2 -= learning_rate * grad_w2
            self.b2 -= learning_rate * grad_b2
            self.w1 -= learning_rate * grad_w1
            self.b1 -= learning_rate * grad_b1
        self.fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        normalized = (x - self.mean) / self.scale
        prediction = np.tanh(normalized @ self.w1 + self.b1) @ self.w2 + self.b2
        return prediction[0] if single else prediction

    def save(self, path: str | Path):
        np.savez(
            Path(path),
            hidden_size=self.hidden_size,
            mean=self.mean,
            scale=self.scale,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            fitted=np.asarray(int(self.fitted)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SmallPathValueModel":
        data = np.load(Path(path), allow_pickle=False)
        model = cls(hidden_size=int(data["hidden_size"]))
        for name in ("mean", "scale", "w1", "b1", "w2"):
            setattr(model, name, data[name].astype(np.float32))
        model.b2 = np.float32(data["b2"])
        model.fitted = bool(int(data["fitted"]))
        return model


class LearnedValueEmitterPlanner(EmitterAwareModelPredictivePlanner):
    """Beam planner whose transparent score can be augmented by learned utility."""

    def __init__(self, *args, value_model=None, learned_value_weight=0.20, **kwargs):
        config = kwargs.get("config") or EmitterAwarePlannerConfig(
            track_recency_weight=0.0,
            identity_weight=0.0,
            exploration_probability=0.0,
        )
        kwargs["config"] = replace(config, exploration_probability=0.0)
        super().__init__(*args, **kwargs)
        self.value_model = value_model or SmallPathValueModel()
        self.learned_value_weight = float(learned_value_weight)
        self.last_value_features: np.ndarray | None = None
        self.last_learned_value = 0.0

    def _simulate_scan(self, node: _EmitterPlanNode, action: int, depth: int):
        breakdown = self._score_breakdown(node, action)
        features = encode_path_candidate(node, action, breakdown, self.config.depth)
        result = super()._simulate_scan(node, action, depth)
        if self.value_model.fitted:
            learned = float(self.value_model.predict(features))
            result.score += (
                self.config.discount**depth
            ) * self.learned_value_weight * learned
        return result

    def select_band(self) -> int:
        action = super().select_band()
        breakdown = self._score_breakdown(self._last_root, action)
        self.last_value_features = encode_path_candidate(
            self._last_root, action, breakdown, self.config.depth
        )
        self.last_learned_value = (
            float(self.value_model.predict(self.last_value_features))
            if self.value_model.fitted
            else 0.0
        )
        self.last_decision_trace["learned_value"] = {
            "enabled": bool(self.value_model.fitted),
            "raw_prediction": self.last_learned_value,
            "weight": self.learned_value_weight,
            "features": {
                name: float(self.last_value_features[index])
                for index, name in enumerate(PATH_VALUE_FEATURES)
            },
        }
        return action

