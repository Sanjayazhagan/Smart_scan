"""Experimental UCB fusion of Track 2 belief and an NMF spectrum forecast."""

from __future__ import annotations

import numpy as np

from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime
from scheduler.world_model_ucb import WorldModelUCBScheduler, _vector


class WorldModelNMFUCBScheduler(WorldModelUCBScheduler):
    """One UCB score augmented by neural and NMF next-band estimates.

    This is an experimental ablation, not the product default. Both predictors
    consume observation history only; simulator truth and reward are ignored by
    their updates.
    """

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        nmf_scale: float = 0.5,
        nmf_components: int = 4,
        nmf_window: int = 30,
        nmf_recompute_every: int = 2,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        switch_penalty: float = 0.05,
        **kwargs,
    ):
        if nmf_scale < 0.0:
            raise ValueError("nmf_scale must be non-negative")
        super().__init__(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            max_scan_age=max_scan_age,
            **kwargs,
        )
        self.nmf_scale = float(nmf_scale)
        self.switch_penalty = float(switch_penalty)
        self.nmf = NMFScheduler(
            num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
        )

    def select_band(self) -> int:
        state = self.runtime.get_global_belief()
        belief = _vector(state, "band_belief", self.num_bands)
        scan_age = _vector(state, "scan_age", self.num_bands)
        nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
        nmf_forecast = np.clip(nmf_forecast, 0.0, None)
        nmf_total = float(nmf_forecast.sum())
        if nmf_total > 1e-12:
            nmf_forecast = nmf_forecast / nmf_total

        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            trace = {
                "mode": "initial_coverage",
                "selected_band": band,
                "expert": "world_model_nmf_ucb",
                "regime": "INITIAL_SWEEP",
                "world_model_weight": 0.0,
                "nmf_weight": 0.0,
            }
            self.last_trace = trace
            self.last_decision_trace = trace
            return band

        exploration_bonus = self.exploration_scale * np.sqrt(
            np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
        )
        world_guidance = self.world_model_scale * belief
        nmf_guidance = self.nmf_scale * nmf_forecast
        scores = (
            self.values
            + exploration_bonus
            + 0.10 * scan_age
            + world_guidance
            + nmf_guidance
        )
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_dist = abs(b - self.last_band) / max(1, self.num_bands - 1)
                scores[b] -= self.switch_penalty * switch_dist
        band = int(np.argmax(scores))
        trace = {
            "mode": "world_model_nmf_guided_ucb",
            "selected_band": band,
            "expert": "world_model_nmf_ucb",
            "regime": "EXPERIMENTAL_FUSION",
            "value": float(self.values[band]),
            "exploration_bonus": float(exploration_bonus[band]),
            "world_model_probability": float(belief[band]),
            "world_model_weight": self.world_model_scale,
            "world_model_contribution": float(world_guidance[band]),
            "nmf_probability": float(nmf_forecast[band]),
            "nmf_weight": self.nmf_scale,
            "nmf_contribution": float(nmf_guidance[band]),
            "score": float(scores[band]),
        }
        self.last_trace = trace
        self.last_decision_trace = trace
        return band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        super().update(band, reward, obs_dict)
        self.nmf.update(band, 0.0, obs_dict)
