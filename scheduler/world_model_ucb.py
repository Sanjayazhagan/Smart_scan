"""World-model-guided UCB scheduler used by the Smart Scan product.

The controller has one decision rule rather than a mixture of experts.  A
discounted UCB score supplies the reliable exploration/exploitation behaviour,
and Track 2's predicted next-band probability is added only after observable
online outcomes show that the world model is well calibrated.
"""

from __future__ import annotations

import numpy as np

from scheduler.baselines import BaseScheduler
from scheduler.track2_core import FEATURE_DIM, NUM_BANDS


def _vector(state: dict, name: str, length: int) -> np.ndarray:
    value = np.asarray(state.get(name, np.zeros(length)), dtype=np.float32).reshape(-1)
    if value.size != length:
        return np.zeros(length, dtype=np.float32)
    return np.clip(value, 0.0, 1.0)


class WorldModelUCBScheduler(BaseScheduler):
    """Discounted UCB directly augmented by Track 2 1D CNN + 2D Spectrogram GRU."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        runtime = None,
        model_path = None,
        max_scan_age: float = 30.0,
        decay: float = 0.985,
        value_learning_rate: float = 0.20,
        exploration_scale: float = 0.75,
        world_model_scale: float = 0.25,
        staying_inertia: float = 0.12,
    ):
        if num_bands != NUM_BANDS:
            raise ValueError(f"World-Model UCB requires exactly {NUM_BANDS} bands")
        super().__init__(num_bands)
        if runtime is not None:
            self.runtime = runtime
        else:
            try:
                from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH, DEFAULT_MAX_SCAN_AGE
                self.runtime = Track2Runtime(
                    model_path=model_path or DEFAULT_MODEL_PATH,
                    max_scan_age=max_scan_age or DEFAULT_MAX_SCAN_AGE,
                )
            except (ImportError, Exception):
                from scheduler.observation_runtime import ObservationOnlyRuntime
                self.runtime = ObservationOnlyRuntime(num_bands)
        self.decay = float(decay)
        self.value_learning_rate = float(value_learning_rate)
        self.exploration_scale = float(exploration_scale)
        self.world_model_scale = float(world_model_scale)
        self.staying_inertia = float(staying_inertia)
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if min(self.value_learning_rate, self.exploration_scale, self.world_model_scale, self.staying_inertia) < 0:
            raise ValueError("UCB scales, inertia, and learning rate must be non-negative")

        self.timestamp = 0.0
        self.counts = np.zeros(num_bands, dtype=np.float64)
        self.values = np.zeros(num_bands, dtype=np.float64)
        self.total_observations = 0.0
        self.last_band: int | None = None
        self._last_world_probability: float | None = None
        self.last_trace: dict = {}
        self.last_decision_trace: dict = {}

    def select_band(self) -> int:
        state = self.runtime.get_global_belief()
        belief = _vector(state, "band_belief", self.num_bands)
        scan_age = _vector(state, "scan_age", self.num_bands)

        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            self._last_world_probability = float(belief[band])
            trace = {
                "mode": "initial_coverage",
                "selected_band": band,
                "world_model_probability": self._last_world_probability,
                "world_model_weight": 0.0,
                "expert": "world_model_ucb",
                "regime": "INITIAL_SWEEP",
            }
            self.last_trace = trace
            self.last_decision_trace = trace
            return band

        # Direct neural guidance from 2D CNN + GRU world model (ZERO GATES)
        exploration_bonus = self.exploration_scale * np.sqrt(
            np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
        )
        world_guidance = self.world_model_scale * belief
        scores = (
            self.values
            + exploration_bonus
            + world_guidance
            + 0.10 * scan_age
        )
        if self.last_band is not None and self.staying_inertia > 0.0:
            scores[self.last_band] += self.staying_inertia
        band = int(np.argmax(scores))
        self.last_band = band
        self._last_world_probability = float(belief[band])
        trace = {
            "mode": "world_model_guided_ucb",
            "selected_band": band,
            "expert": "world_model_ucb",
            "regime": "NEURAL_GUIDANCE_ACTIVE",
            "value": float(self.values[band]),
            "exploration_bonus": float(exploration_bonus[band]),
            "world_model_probability": float(belief[band]),
            "world_model_scale": self.world_model_scale,
            "world_model_weight": self.world_model_scale,
            "world_model_contribution": float(world_guidance[band]),
            "score": float(scores[band]),
        }
        self.last_trace = trace
        self.last_decision_trace = trace
        return band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        self.timestamp += 1.0
        self.counts *= self.decay
        self.total_observations = self.total_observations * self.decay + 1.0
        self.counts[int(band)] += 1.0
        if obs_dict is None:
            return

        res = self.runtime.update(obs_dict, timestamp=self.timestamp - 1.0)
        detected = bool(obs_dict.get("detected", False))
        quality = float(
            np.asarray(obs_dict.get("quality", [0.0]), dtype=np.float32).reshape(-1)[0]
        )
        index = int(band)

        is_known = bool(res.get("known_identity", True)) if isinstance(res, dict) else True
        if detected and is_known:
            # Verified Authentic Radar: award detection reward bonus
            observable_value = float(np.clip(0.10 + 0.90 * quality, 0.0, 1.0))
        else:
            # Empty scan OR Decoy/Jammer tone: neutral baseline (avoids false-alarm decoy traps)
            observable_value = 0.02

        self.values[index] = (
            (1.0 - self.value_learning_rate) * self.values[index]
            + self.value_learning_rate * observable_value
        )
