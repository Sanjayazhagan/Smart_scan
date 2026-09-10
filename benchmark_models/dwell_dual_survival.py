"""Model: Dwell-Dual + Hazard Rate Survival Dwell.

Applies Survival Analysis to radar beam dwell duration:
- In real radar systems, an illumination beam has a finite physical dwell duration
  (e.g., beam steering interval or scan dwell of eta steps).
- Memoryless exponential decay assumes the probability of departure is constant.
  In reality, the longer a beam has dwelt on a target, the MORE likely it is about to depart.
- Dwell-Dual + Survival models dwell departure using a Weibull hazard function:
    h(k) = (beta / eta) * (k / eta)^(beta - 1)
  where k is consecutive dwell steps, eta is characteristic beam dwell (e.g. 8-10 steps),
  and beta > 1 models positive aging (wear-out / departure).
- When h(k) exceeds an exit threshold, dwell lock is gracefully suppressed, enabling
  preemptive departure and eliminating dead-time waiting after beam shutoff.
"""
from __future__ import annotations

import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualSurvivalScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy augmented with Weibull Hazard Survival Dwell."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        weibull_eta: float = 8.0,      # Characteristic beam dwell duration (steps)
        weibull_beta: float = 2.2,     # Shape parameter (beta > 1 implies increasing hazard)
        hazard_exit_thresh: float = 0.45,
        nmf_scale: float = 1.20,
        switch_penalty: float = 0.04,
        dwell_inertia: float = 1.50,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            num_bands,
            nmf_scale=nmf_scale,
            switch_penalty=switch_penalty,
            dwell_inertia=dwell_inertia,
            seed=seed,
            **kwargs,
        )
        self.weibull_eta = float(weibull_eta)
        self.weibull_beta = float(weibull_beta)
        self.hazard_exit_thresh = float(hazard_exit_thresh)

    def _compute_departure_hazard(self, k_dwell: int) -> float:
        """Computes Weibull hazard rate h(k) for k consecutive dwell steps."""
        if k_dwell <= 0:
            return 0.0
        normalized_k = k_dwell / max(1.0, self.weibull_eta)
        hazard = (self.weibull_beta / self.weibull_eta) * (normalized_k ** (self.weibull_beta - 1.0))
        return float(hazard)

    def select_band(self) -> int:
        # 1. Stale-band coverage check from Champion
        coverage_candidates = self._coverage_candidates()
        if coverage_candidates.size:
            if self._should_interrupt_coverage():
                self.dwell_decisions += 1
                self.consecutive_dwell_steps += 1
                self.coverage_interrupt_dwells += 1
                self.last_governing_policy = "coverage_interrupt_dwell"
                return int(self.last_band)

            band = self._choose_coverage_candidate(coverage_candidates)
            self.last_band = band
            self.last_governing_policy = "smart_forced_coverage"
            return band

        # 2. Hazard survival calculation
        hazard = self._compute_departure_hazard(self.consecutive_dwell_steps)
        imminent_departure = hazard >= self.hazard_exit_thresh

        # 3. Switching costs
        switch_costs = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            switch_costs = self.switch_penalty * (
                np.abs(np.arange(self.num_bands) - int(self.last_band))
                / max(1, self.num_bands - 1)
            )

        is_signal_active = (self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and self.consecutive_dwell_steps >= 2
        )) and (not imminent_departure)

        uncertainty = self.uncertainty_estimator.get_uncertainty()
        max_uncertainty = float(np.max(uncertainty))

        can_scout = (not is_signal_active) and (
            max_uncertainty >= self.uncertainty_trigger_threshold
        )
        is_explore = False
        if can_scout:
            prob = self.explore_budget_prob * max_uncertainty
            is_explore = self.rng.random() < prob

        if is_explore:
            explore_scores = uncertainty - switch_costs
            selected_band = int(np.argmax(explore_scores))
            self.explore_decisions += 1
            self.last_governing_policy = "explore"
            self.consecutive_dwell_steps = 0
        else:
            nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
            nmf_forecast = np.clip(nmf_forecast, 0.0, None)
            nmf_total = float(nmf_forecast.sum())
            if nmf_total > 1e-12:
                nmf_forecast = nmf_forecast / nmf_total

            exploration_bonus = self.exploration_scale * np.sqrt(
                np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
            )
            exploit_scores = (
                self.values
                + exploration_bonus
                + self.nmf_scale * nmf_forecast
                - switch_costs
            )

            if self.last_band is not None and is_signal_active:
                # Survival penalty gracefully reduces dwell bonus as hazard increases
                survival_multiplier = max(0.1, 1.0 - (hazard / max(1e-3, self.hazard_exit_thresh)))
                dwell_bonus = self.dwell_inertia * max(0.4, self.last_quality) * survival_multiplier
                exploit_scores[self.last_band] += dwell_bonus

            selected_band = int(np.argmax(exploit_scores))
            if selected_band == self.last_band and is_signal_active:
                self.dwell_decisions += 1
                self.consecutive_dwell_steps += 1
                self.last_governing_policy = "dwell_exploit"
            else:
                self.exploit_decisions += 1
                self.consecutive_dwell_steps = 0
                self.last_governing_policy = "exploit"

        self.last_band = selected_band
        return selected_band


def build_dwell_dual_survival(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualSurvivalScheduler:
    return DwellDualSurvivalScheduler(num_bands=num_bands, seed=seed)
