"""Lean observable MoE built from held-out-screened Smart Scan experts."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from scheduler.baselines import BaseScheduler
from scheduler.paradigms.bandit_suite import Exp3BanditScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.world_model_ucb import _vector


class LeanObservableMoEScheduler(BaseScheduler):
    """Route among four complementary modes using observation history only.

    Modes:
    - stationary: direct NMF
    - hopping: World + observable-feedback Exp3 + UCB
    - harsh: neural-disabled UCB
    - dynamic/default: World + NMF + UCB
    """

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        history_size: int = 60,
        minimum_history: int = 30,
        switch_evidence: int = 6,
        minimum_mode_dwell: int = 12,
        seed: int = 42,
    ):
        super().__init__(num_bands)
        self.core = WorldModelNMFUCBScheduler(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            max_scan_age=max_scan_age,
            nmf_scale=1.0,
            world_model_scale=0.35,
        )
        self.runtime = self.core.runtime
        self.exp3 = Exp3BanditScheduler(num_bands, seed=seed)
        self.history = deque(maxlen=int(history_size))
        self.minimum_history = int(minimum_history)
        self.switch_evidence = int(switch_evidence)
        self.minimum_mode_dwell = int(minimum_mode_dwell)
        self.mode = "dynamic"
        self.pending_mode = self.mode
        self.pending_count = 0
        self.mode_dwell = 0
        self.last_decision_trace: dict = {}

    def _history_features(self) -> dict[str, float]:
        detected_rows = [row for row in self.history if row[1]]
        detected_bands = [row[0] for row in detected_rows]
        qualities = np.asarray([row[2] for row in detected_rows], dtype=np.float64)
        if detected_bands:
            concentration = max(Counter(detected_bands).values()) / len(detected_bands)
            transitions = (
                np.mean(
                    [a != b for a, b in zip(detected_bands[:-1], detected_bands[1:])]
                )
                if len(detected_bands) > 1
                else 0.0
            )
            mean_quality = float(qualities.mean())
            low_quality_ratio = float(np.mean(qualities < 0.35))
        else:
            concentration = transitions = mean_quality = low_quality_ratio = 0.0
        return {
            "history_length": float(len(self.history)),
            "detections": float(len(detected_bands)),
            "detection_ratio": float(len(detected_bands) / max(1, len(self.history))),
            "hit_band_concentration": float(concentration),
            "hit_transition_rate": float(transitions),
            "mean_detected_quality": mean_quality,
            "low_quality_detection_ratio": low_quality_ratio,
        }

    def _candidate_mode(self, features: dict[str, float]) -> str:
        if features["history_length"] < self.minimum_history:
            return "dynamic"
        if (
            features["detections"] >= 4
            and (
                features["mean_detected_quality"] < 0.32
                or features["low_quality_detection_ratio"] >= 0.60
            )
        ):
            return "harsh"
        if (
            features["detections"] >= 6
            and features["hit_band_concentration"] >= 0.62
            and features["hit_transition_rate"] <= 0.35
            and features["mean_detected_quality"] >= 0.40
        ):
            return "stationary"
        if (
            features["detections"] >= 6
            and features["hit_transition_rate"] >= 0.62
            and features["mean_detected_quality"] >= 0.32
        ):
            return "hopping"
        return "dynamic"

    def _update_mode(self, candidate: str) -> None:
        self.mode_dwell += 1
        if candidate == self.mode:
            self.pending_mode = candidate
            self.pending_count = 0
            return
        if candidate != self.pending_mode:
            self.pending_mode = candidate
            self.pending_count = 1
        else:
            self.pending_count += 1
        if (
            self.pending_count >= self.switch_evidence
            and self.mode_dwell >= self.minimum_mode_dwell
        ):
            self.mode = candidate
            self.mode_dwell = 0
            self.pending_count = 0

    def _base_components(self):
        state = self.runtime.get_global_belief()
        belief = _vector(state, "band_belief", self.num_bands)
        age = _vector(state, "scan_age", self.num_bands)
        bonus = self.core.exploration_scale * np.sqrt(
            np.log(self.core.total_observations + 2.0)
            / np.maximum(self.core.counts, 1e-6)
        )
        return belief, age, bonus

    def select_band(self) -> int:
        features = self._history_features()
        candidate = self._candidate_mode(features)
        self._update_mode(candidate)
        never_observed = np.flatnonzero(self.core.counts < 0.5)
        if never_observed.size:
            selected = int(never_observed[0])
            expert = "coverage"
        elif self.mode == "stationary":
            selected = int(self.core.nmf.select_band())
            expert = "nmf"
        elif self.mode == "dynamic":
            selected = int(self.core.select_band())
            expert = "world_nmf_ucb"
        else:
            belief, age, bonus = self._base_components()
            scores = self.core.values + bonus + 0.10 * age
            if self.mode == "hopping":
                weights = np.asarray(self.exp3.weights, dtype=np.float64)
                weights /= max(float(weights.sum()), 1e-12)
                gamma = float(self.exp3.gamma)
                exp3_probability = (1.0 - gamma) * weights + gamma / self.num_bands
                scores = scores + 0.35 * belief + exp3_probability
                expert = "world_exp3_ucb"
            else:
                expert = "safe_ucb"
            selected = int(np.argmax(scores))

        self.last_decision_trace = {
            "selected_band": selected,
            "mode": self.mode,
            "regime": self.mode,
            "expert": expert,
            "candidate_mode": candidate,
            "pending_mode": self.pending_mode,
            "pending_count": self.pending_count,
            "history_features": features,
        }
        return selected

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        self.core.update(band, 0.0, obs_dict)
        detected = bool(obs_dict and obs_dict.get("detected", False))
        quality = float(
            np.asarray((obs_dict or {}).get("quality", [0.0])).reshape(-1)[0]
        )
        observable_value = (0.10 + 0.90 * quality) if detected else 0.02
        self.exp3.update(band, 2.0 * observable_value - 1.0, obs_dict)
        self.history.append((int(band), detected, quality))

    def begin_scored_phase(self) -> dict:
        features = self._history_features()
        return {"history_mode": self.mode, **features}
