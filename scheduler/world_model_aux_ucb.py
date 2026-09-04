"""Experimental truth-free fusions of World-Model UCB with auxiliary experts."""

from __future__ import annotations

import numpy as np

from scheduler.paradigms.adaptive_receiver_search import AdaptiveReceiverSearchScheduler
from scheduler.paradigms.bandit_suite import Exp3BanditScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime
from scheduler.world_model_ucb import WorldModelUCBScheduler, _vector


class WorldModelAuxUCBScheduler(WorldModelUCBScheduler):
    """Fuse one normalized auxiliary forecast into the current World-UCB score."""

    VALID_AUXILIARIES = {"rpca", "pri", "exp3"}

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        auxiliary: str,
        auxiliary_scale: float = 1.0,
        seed: int = 42,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        **kwargs,
    ):
        auxiliary = str(auxiliary).lower()
        if auxiliary not in self.VALID_AUXILIARIES:
            raise ValueError(f"Unknown auxiliary {auxiliary!r}")
        if auxiliary_scale < 0.0:
            raise ValueError("auxiliary_scale must be non-negative")
        super().__init__(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            max_scan_age=max_scan_age,
            **kwargs,
        )
        self.auxiliary_name = auxiliary
        self.auxiliary_scale = float(auxiliary_scale)
        if auxiliary == "rpca":
            self.auxiliary = RobustPCAPSRScheduler(
                num_bands, recompute_every=2, seed=seed
            )
        elif auxiliary == "pri":
            self.auxiliary = AdaptiveReceiverSearchScheduler(num_bands, seed=seed)
        else:
            self.auxiliary = Exp3BanditScheduler(num_bands, seed=seed)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
        total = float(values.sum())
        return values / total if total > 1e-12 else np.zeros_like(values)

    def _auxiliary_forecast(self) -> np.ndarray:
        if self.auxiliary_name == "rpca":
            return self._normalize(self.auxiliary.psr_state)
        if self.auxiliary_name == "exp3":
            weights = np.asarray(self.auxiliary.weights, dtype=np.float64)
            total = float(weights.sum())
            if total <= 0.0 or not np.isfinite(total):
                weights = np.ones(self.num_bands, dtype=np.float64)
                total = float(self.num_bands)
            gamma = float(self.auxiliary.gamma)
            probabilities = (1.0 - gamma) * weights / total + gamma / self.num_bands
            self.auxiliary.probabilities = probabilities
            return self._normalize(probabilities)

        urgency = np.zeros(self.num_bands, dtype=np.float64)
        expert = self.auxiliary
        for band in range(self.num_bands):
            pri = float(expert.estimated_pri[band])
            confidence = float(expert.confidence[band])
            if pri <= 1.0 or confidence <= 0.3 or not expert.pulse_history[band]:
                continue
            steps_since = expert.current_step - expert.pulse_history[band][-1]
            time_to_pulse = (pri - (steps_since % pri)) % pri
            if time_to_pulse <= expert.dwell_window:
                urgency[band] = confidence * (
                    1.0 - time_to_pulse / (expert.dwell_window + 1.0)
                )
        return self._normalize(urgency)

    def select_band(self) -> int:
        state = self.runtime.get_global_belief()
        belief = _vector(state, "band_belief", self.num_bands)
        scan_age = _vector(state, "scan_age", self.num_bands)
        auxiliary_forecast = self._auxiliary_forecast()

        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            trace = {
                "mode": "initial_coverage",
                "selected_band": band,
                "expert": f"world_{self.auxiliary_name}_ucb",
                "regime": "INITIAL_SWEEP",
                "world_model_weight": 0.0,
                "auxiliary_weight": 0.0,
            }
            self.last_trace = trace
            self.last_decision_trace = trace
            return band

        exploration_bonus = self.exploration_scale * np.sqrt(
            np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
        )
        world_guidance = self.world_model_scale * belief
        auxiliary_guidance = self.auxiliary_scale * auxiliary_forecast
        scores = (
            self.values
            + exploration_bonus
            + 0.10 * scan_age
            + world_guidance
            + auxiliary_guidance
        )
        band = int(np.argmax(scores))
        trace = {
            "mode": f"world_{self.auxiliary_name}_guided_ucb",
            "selected_band": band,
            "expert": f"world_{self.auxiliary_name}_ucb",
            "regime": "EXPERIMENTAL_FUSION",
            "world_model_probability": float(belief[band]),
            "world_model_weight": self.world_model_scale,
            "auxiliary": self.auxiliary_name,
            "auxiliary_probability": float(auxiliary_forecast[band]),
            "auxiliary_weight": self.auxiliary_scale,
            "auxiliary_contribution": float(auxiliary_guidance[band]),
            "score": float(scores[band]),
        }
        self.last_trace = trace
        self.last_decision_trace = trace
        return band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        super().update(band, reward, obs_dict)
        if self.auxiliary_name == "exp3":
            detected = bool(obs_dict and obs_dict.get("detected", False))
            quality = float(
                np.asarray((obs_dict or {}).get("quality", [0.0])).reshape(-1)[0]
            )
            observable_value = (0.10 + 0.90 * quality) if detected else 0.02
            auxiliary_reward = 2.0 * observable_value - 1.0
        else:
            auxiliary_reward = 0.0
        self.auxiliary.update(band, auxiliary_reward, obs_dict)
