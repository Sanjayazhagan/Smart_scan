"""Dwell-Dual + Restless Multi-Armed Bandit (RMAB) Whittle Index Scheduler.

Models each unobserved channel as an evolving 2-state Markov chain (Silent vs Active).
Tracks the Age of Information (AoI) tau_b for unobserved channels, computing the closed-form
Whittle index to quantify the marginal opportunity cost of leaving channels unobserved.
Integrates Whittle opportunity indexing with NMF spectral discovery and physical Dwell Inertia.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MODEL_PATH


class DwellDualWhittleScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy with Restless Bandit Age-of-Information Whittle Indexing."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        p_01: float = 0.08,
        p_11: float = 0.70,
        whittle_scale: float = 0.40,
        aoi_curiosity_weight: float = 0.15,
        seed: Optional[int] = None,
        model_path: Optional[str] = DEFAULT_MODEL_PATH,
        **kwargs,
    ):
        super().__init__(num_bands=num_bands, seed=seed, model_path=model_path, **kwargs)
        self.p_01 = float(p_01)
        self.p_11 = float(p_11)
        self.whittle_scale = float(whittle_scale)
        self.aoi_curiosity_weight = float(aoi_curiosity_weight)

        # Steady-state active probability
        self.pi_active = self.p_01 / (max(1e-6, (1.0 - self.p_11) + self.p_01))
        self.eigen_diff = self.p_11 - self.p_01

        # Restless state tracking
        self.age_of_info = np.zeros(self.num_bands, dtype=np.float64)
        self.last_known_belief = np.full(self.num_bands, self.pi_active, dtype=np.float64)

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

        # Compute Restless State Belief Evolution
        # p_b(t) = pi_active + (p_last - pi_active) * (p11 - p01)^tau
        decay_factor = np.power(self.eigen_diff, self.age_of_info)
        restless_prob = self.pi_active + (self.last_known_belief - self.pi_active) * decay_factor
        restless_prob = np.clip(restless_prob, 0.01, 0.99)

        # Closed-form Whittle opportunity index:
        # Higher for bands that have high evolved restless probability or high Age of Information
        whittle_index = restless_prob + self.aoi_curiosity_weight * np.sqrt(self.age_of_info + 1.0)

        if is_explore:
            # EXPLORE: Restless Whittle Index steers exploration to starved/evolved bands
            explore_scores = whittle_index - switch_costs
            selected_band = int(np.argmax(explore_scores))
            self.explore_decisions += 1
            self.last_governing_policy = "whittle_explore"
            self.consecutive_dwell_steps = 0
        else:
            # EXPLOIT: Fusion of NMF forecast, Whittle restless belief, and Dwell Inertia
            nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
            nmf_forecast = np.clip(nmf_forecast, 0.0, None)
            nmf_total = float(nmf_forecast.sum())
            if nmf_total > 1e-12:
                nmf_forecast = nmf_forecast / nmf_total

            exploit_scores = (
                self.values
                + self.nmf_scale * nmf_forecast
                + self.whittle_scale * whittle_index
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
                self.last_governing_policy = "whittle_exploit"

        self.last_band = selected_band
        return selected_band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        super().update(band, reward, obs_dict)

        # Update Age of Information:
        # Scanned band resets to 0; unobserved bands increment by 1
        self.age_of_info += 1.0
        self.age_of_info[band] = 0.0

        # Update last known belief for scanned band
        detected = bool(obs_dict.get("detected", False)) if obs_dict else False
        self.last_known_belief[band] = 1.0 if detected else 0.0


def build_dwell_dual_whittle(
    num_bands: int = NUM_BANDS,
    seed: Optional[int] = None,
    **kwargs,
) -> DwellDualWhittleScheduler:
    return DwellDualWhittleScheduler(num_bands=num_bands, seed=seed, **kwargs)
