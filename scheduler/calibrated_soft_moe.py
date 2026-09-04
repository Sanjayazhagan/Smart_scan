"""Observable, confidence-calibrated soft mixture for spectrum monitoring.

The router never reads simulator truth or the simulator reward.  It blends
expert score vectors from recent receiver observations and suppresses neural
guidance when online Brier skill does not beat an observable base-rate model.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from scheduler.baselines import BaseScheduler
from scheduler.paradigms.adaptive_receiver_search import AdaptiveReceiverSearchScheduler
from scheduler.paradigms.bandit_suite import Exp3BanditScheduler
from scheduler.paradigms.robust_pca_psr import RobustPCAPSRScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.world_model_ucb import _vector


def _unit_interval(values: np.ndarray) -> np.ndarray:
    """Map a score vector to [0, 1] while retaining its ordering."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(values.min(initial=0.0))
    high = float(values.max(initial=0.0))
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _probability(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(values.sum())
    return values / total if total > 1e-12 else np.zeros_like(values)


class OnlineBrierCalibrator:
    """Estimate observable out-of-sample skill for the selected-band forecast."""

    def __init__(self, decay: float = 0.97, prior_strength: float = 20.0):
        self.decay = float(decay)
        self.prior_strength = float(prior_strength)
        self.effective_count = 0.0
        self.event_rate = 0.05
        self.model_brier = 0.0
        self.baseline_brier = 0.0

    def update(self, probability: float, outcome: float) -> None:
        probability = float(np.clip(probability, 0.0, 1.0))
        outcome = float(np.clip(outcome, 0.0, 1.0))
        baseline_probability = self.event_rate
        self.effective_count = self.decay * self.effective_count + 1.0
        self.model_brier = (
            self.decay * self.model_brier + (probability - outcome) ** 2
        )
        self.baseline_brier = (
            self.decay * self.baseline_brier + (baseline_probability - outcome) ** 2
        )
        rate_step = 1.0 / max(self.prior_strength, self.effective_count)
        self.event_rate += rate_step * (outcome - self.event_rate)

    @property
    def skill(self) -> float:
        if self.effective_count < 5.0 or self.baseline_brier <= 1e-12:
            return 0.0
        raw = 1.0 - self.model_brier / self.baseline_brier
        evidence = self.effective_count / (self.effective_count + self.prior_strength)
        return float(np.clip(raw, 0.0, 1.0) * evidence)


class CalibratedWorldNMFUCBScheduler(WorldModelNMFUCBScheduler):
    """Exact NMF+UCB ablation with online-gated world-model guidance."""

    def __init__(self, *args, calibration_prior: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibrator = OnlineBrierCalibrator(prior_strength=calibration_prior)
        self._decision_probability = 0.0

    def select_band(self) -> int:
        state = self.runtime.get_global_belief()
        belief = _vector(state, "band_belief", self.num_bands)
        original_scale = self.world_model_scale
        self.world_model_scale = original_scale * self.calibrator.skill
        try:
            band = super().select_band()
        finally:
            self.world_model_scale = original_scale
        self._decision_probability = float(belief[band])
        self.last_trace["world_calibration_skill"] = self.calibrator.skill
        self.last_trace["configured_world_model_weight"] = original_scale
        return band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        detected = bool(obs_dict and obs_dict.get("detected", False))
        self.calibrator.update(self._decision_probability, float(detected))
        super().update(band, 0.0, obs_dict)


class CalibratedSoftMoEScheduler(BaseScheduler):
    """Softly blend UCB, NMF, RPCA, PRI, Exp3 and calibrated neural scores."""

    REGIMES = ("stationary", "hopping", "harsh", "dynamic")

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        history_size: int = 60,
        minimum_history: int = 20,
        router_temperature: float = 0.55,
        guidance_scale: float = 1.0,
        world_model_scale: float = 0.35,
        seed: int = 42,
    ):
        super().__init__(num_bands)
        if router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        self.core = WorldModelNMFUCBScheduler(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            max_scan_age=max_scan_age,
            nmf_scale=1.0,
            world_model_scale=world_model_scale,
        )
        self.runtime = self.core.runtime
        self.rpca = RobustPCAPSRScheduler(num_bands, recompute_every=2, seed=seed)
        self.pri = AdaptiveReceiverSearchScheduler(num_bands, seed=seed)
        self.exp3 = Exp3BanditScheduler(num_bands, seed=seed)
        self.history = deque(maxlen=int(history_size))
        self.minimum_history = int(minimum_history)
        self.router_temperature = float(router_temperature)
        self.guidance_scale = float(guidance_scale)
        self.world_model_scale = float(world_model_scale)
        self.calibrator = OnlineBrierCalibrator()
        self._decision_probability = 0.0
        self.last_decision_trace: dict = {}

    def _features(self) -> dict[str, float]:
        hits = [row for row in self.history if row[1]]
        bands = [row[0] for row in hits]
        qualities = np.asarray([row[2] for row in hits], dtype=np.float64)
        if bands:
            concentration = max(Counter(bands).values()) / len(bands)
            transition = float(np.mean([a != b for a, b in zip(bands[:-1], bands[1:])])) if len(bands) > 1 else 0.0
            quality = float(qualities.mean())
            low_quality = float(np.mean(qualities < 0.35))
        else:
            concentration = transition = quality = low_quality = 0.0
        return {
            "history_length": float(len(self.history)),
            "detections": float(len(hits)),
            "detection_ratio": float(len(hits) / max(1, len(self.history))),
            "concentration": float(concentration),
            "transition_rate": transition,
            "mean_quality": quality,
            "low_quality_ratio": low_quality,
        }

    def _regime_weights(self, f: dict[str, float]) -> dict[str, float]:
        if f["history_length"] < self.minimum_history or f["detections"] < 3:
            return {"stationary": 0.05, "hopping": 0.05, "harsh": 0.10, "dynamic": 0.80}
        logits = np.asarray([
            3.0 * (f["concentration"] - 0.45) + 2.0 * (0.50 - f["transition_rate"]) + f["mean_quality"],
            3.0 * (f["transition_rate"] - 0.45) + f["mean_quality"] - f["concentration"],
            4.0 * (f["low_quality_ratio"] - 0.35) + 2.0 * (0.40 - f["mean_quality"]),
            0.5 + 1.5 * (1.0 - f["concentration"]) + 0.5 * f["transition_rate"],
        ], dtype=np.float64)
        logits = (logits - logits.max()) / self.router_temperature
        weights = np.exp(np.clip(logits, -30.0, 0.0))
        weights /= weights.sum()
        return dict(zip(self.REGIMES, weights.tolist()))

    def _pri_forecast(self) -> np.ndarray:
        urgency = np.zeros(self.num_bands, dtype=np.float64)
        for band in range(self.num_bands):
            period = float(self.pri.estimated_pri[band])
            confidence = float(self.pri.confidence[band])
            if period <= 1.0 or confidence <= 0.3 or not self.pri.pulse_history[band]:
                continue
            elapsed = self.pri.current_step - self.pri.pulse_history[band][-1]
            until = (period - (elapsed % period)) % period
            if until <= self.pri.dwell_window:
                urgency[band] = confidence * (1.0 - until / (self.pri.dwell_window + 1.0))
        return _probability(urgency)

    def _exp3_forecast(self) -> np.ndarray:
        weights = np.asarray(self.exp3.weights, dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1e-12)
        return (1.0 - self.exp3.gamma) * weights + self.exp3.gamma / self.num_bands

    def _expert_weights(self, regime: dict[str, float]) -> dict[str, float]:
        s, h, x, d = (regime[name] for name in self.REGIMES)
        weights = {
            "ucb": 0.15 * s + 0.22 * h + 0.30 * x + 0.25 * d,
            "nmf": 0.75 * s + 0.38 * h + 0.10 * x + 0.40 * d,
            "rpca": 0.03 * s + 0.08 * h + 0.22 * x + 0.13 * d,
            "pri": 0.02 * s + 0.12 * h + 0.05 * x + 0.08 * d,
            "exp3": 0.05 * s + 0.20 * h + 0.33 * x + 0.14 * d,
        }
        total = sum(weights.values())
        return {name: value / total for name, value in weights.items()}

    def select_band(self) -> int:
        state = self.runtime.get_global_belief()
        belief = _vector(state, "band_belief", self.num_bands).astype(np.float64)
        age = _vector(state, "scan_age", self.num_bands).astype(np.float64)
        never = np.flatnonzero(self.core.counts < 0.5)
        if never.size:
            band = int(never[0])
            self._decision_probability = float(belief[band])
            self.last_decision_trace = {"selected_band": band, "expert": "coverage", "regime": "initial"}
            return band

        bonus = self.core.exploration_scale * np.sqrt(
            np.log(self.core.total_observations + 2.0) / np.maximum(self.core.counts, 1e-6)
        )
        ucb = _unit_interval(self.core.values + bonus + 0.10 * age)
        nmf = _unit_interval(_probability(self.core.nmf.predicted_spectrum) + 0.05 * np.sqrt(self.core.nmf.scan_age))
        rpca = _unit_interval(_probability(self.rpca.psr_state) + 0.06 * np.sqrt(self.rpca.scan_age))
        pri = _unit_interval(self._pri_forecast() + 0.05 * np.sqrt(self.pri.scan_age))
        exp3 = _unit_interval(self._exp3_forecast())

        features = self._features()
        regime = self._regime_weights(features)
        expert = self._expert_weights(regime)
        scores = self.guidance_scale * (
            expert["ucb"] * ucb + expert["nmf"] * nmf + expert["rpca"] * rpca
            + expert["pri"] * pri + expert["exp3"] * exp3
        )
        neural_weight = self.world_model_scale * self.calibrator.skill
        scores += neural_weight * _unit_interval(belief)
        scores += 0.03 * age
        band = int(np.argmax(scores))
        self._decision_probability = float(belief[band])
        dominant_regime = max(regime, key=regime.get)
        dominant_expert = max(expert, key=expert.get)
        self.last_decision_trace = {
            "selected_band": band,
            "mode": dominant_regime,
            "regime": dominant_regime,
            "expert": dominant_expert,
            "regime_weights": regime,
            "expert_weights": expert,
            "world_calibration_skill": self.calibrator.skill,
            "world_model_weight": neural_weight,
            "history_features": features,
            "score": float(scores[band]),
        }
        return band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        detected = bool(obs_dict and obs_dict.get("detected", False))
        quality = float(np.asarray((obs_dict or {}).get("quality", [0.0])).reshape(-1)[0])
        self.calibrator.update(self._decision_probability, float(detected))
        self.core.update(band, 0.0, obs_dict)
        self.rpca.update(band, 0.0, obs_dict)
        self.pri.update(band, 0.0, obs_dict)
        observable_value = (0.10 + 0.90 * quality) if detected else 0.02
        self.exp3.update(band, 2.0 * observable_value - 1.0, obs_dict)
        self.history.append((int(band), detected, quality))

    def begin_scored_phase(self) -> dict:
        return {
            "world_calibration_skill": self.calibrator.skill,
            **self._features(),
            **{f"router_{key}": value for key, value in self._regime_weights(self._features()).items()},
        }
