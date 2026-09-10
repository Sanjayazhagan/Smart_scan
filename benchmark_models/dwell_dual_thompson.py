"""Dwell-Dual + Discounted Thompson Sampling Scheduler.

Replaces deterministic Upper Confidence Bound (UCB) optimism with Bayesian posterior sampling.
Maintains Beta-Bernoulli conjugate distributions over channel activity with an exponential
discount factor (gamma_bayes) to track non-stationary agile frequency hoppers.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MODEL_PATH


class DwellDualThompsonScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy with Discounted Thompson Sampling Exploration."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        gamma_bayes: float = 0.985,
        thompson_weight: float = 0.35,
        seed: Optional[int] = None,
        model_path: Optional[str] = DEFAULT_MODEL_PATH,
        **kwargs,
    ):
        super().__init__(num_bands=num_bands, seed=seed, model_path=model_path, **kwargs)
        self.gamma_bayes = float(gamma_bayes)
        self.thompson_weight = float(thompson_weight)

        # Bayesian Beta-Bernoulli conjugate priors (Uniform Beta(1, 1))
        self.alpha_params = np.ones(self.num_bands, dtype=np.float64)
        self.beta_params = np.ones(self.num_bands, dtype=np.float64)

    def select_band(self) -> int:
        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            self.last_band = band
            self.last_governing_policy = "initial_sweep"
            return band

        uncertainty = self.uncertainty_estimator.get_uncertainty()
        max_uncertainty = float(np.max(uncertainty))

        # Switching cost distance vector
        switch_costs = np.zeros(self.num_bands, dtype=np.float32)
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_costs[b] = self.switch_penalty * (abs(b - self.last_band) / max(1, self.num_bands - 1))

        # Active signal check
        is_signal_active = self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and self.consecutive_dwell_steps >= 2
        )

        can_scout = (not is_signal_active) and (max_uncertainty >= self.uncertainty_trigger_threshold)
        is_explore = False
        if can_scout:
            prob = self.explore_budget_prob * max_uncertainty
            is_explore = (self.rng.random() < prob)

        # Thompson Sampling posterior draw
        # Clamp parameters to prevent numerical instability
        safe_a = np.clip(self.alpha_params, 0.1, 500.0)
        safe_b = np.clip(self.beta_params, 0.1, 500.0)
        ts_samples = self.rng.beta(safe_a, safe_b)

        if is_explore:
            # EXPLORE: Thompson Sampling draws steer scout direction
            explore_scores = ts_samples + (0.50 * uncertainty) - switch_costs
            selected_band = int(np.argmax(explore_scores))
            self.explore_decisions += 1
            self.last_governing_policy = "thompson_explore"
            self.consecutive_dwell_steps = 0
        else:
            # EXPLOIT: Fusion of NMF forecast, Thompson sampled belief, and Dwell Inertia
            nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
            nmf_forecast = np.clip(nmf_forecast, 0.0, None)
            nmf_total = float(nmf_forecast.sum())
            if nmf_total > 1e-12:
                nmf_forecast = nmf_forecast / nmf_total

            exploit_scores = (
                (1.0 - self.thompson_weight) * (self.values + self.nmf_scale * nmf_forecast)
                + (self.thompson_weight * ts_samples)
                - switch_costs
            )

            if self.last_band is not None and is_signal_active:
                dwell_bonus = self.dwell_inertia * max(0.4, self.last_quality)
                exploit_scores[self.last_band] += dwell_bonus

            selected_band = int(np.argmax(exploit_scores))
            if selected_band == self.last_band and is_signal_active:
                self.dwell_decisions += 1
                self.consecutive_dwell_steps += 1
                self.last_governing_policy = "dwell_exploit"
            else:
                self.exploit_decisions += 1
                self.consecutive_dwell_steps = 0
                self.last_governing_policy = "thompson_exploit"

        self.last_band = selected_band
        return selected_band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        super().update(band, reward, obs_dict)

        # Discount historical evidence across all bands (non-stationarity tracking)
        self.alpha_params = 1.0 + self.gamma_bayes * (self.alpha_params - 1.0)
        self.beta_params = 1.0 + self.gamma_bayes * (self.beta_params - 1.0)

        # Bayesian update on observed band
        detected = bool(obs_dict.get("detected", False)) if obs_dict else False
        if detected:
            self.alpha_params[band] += 1.0
        else:
            self.beta_params[band] += 1.0


def build_dwell_dual_thompson(
    num_bands: int = NUM_BANDS,
    seed: Optional[int] = None,
    **kwargs,
) -> DwellDualThompsonScheduler:
    return DwellDualThompsonScheduler(num_bands=num_bands, seed=seed, **kwargs)
