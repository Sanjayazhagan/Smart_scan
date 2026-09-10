"""Model: Dwell-Dual + Markov Hopping Transition Predictor.

Targets agile frequency-hopping emitters:
- In electronic warfare, frequency-hopping radars do not hop purely at random;
  they follow pseudo-random hopping patterns (e.g., Costas arrays, LCG sequences).
- Dwell-Dual + Markov maintains an online empirical transition matrix:
    T[i, j] = Count(active signal was at Band i, then next observed at Band j)
  with Laplace smoothing.
- When an active emitter at Band i abruptly hops away (consecutive_misses == 1),
  the scheduler computes the conditional hopping prior:
    P(hop to j | was at i) = (T[i, j] + alpha) / sum_k (T[i, k] + alpha)
- Rather than blindly exploring by general uncertainty, the receiver immediately probes
  the most probable hopping destination, re-intercepting the emitter in 1 step!
"""
from __future__ import annotations

import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualMarkovHopScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy augmented with Markov Hopping Transition Prediction."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        markov_prior_weight: float = 0.60,
        laplace_alpha: float = 0.20,
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
        self.markov_prior_weight = float(markov_prior_weight)
        self.laplace_alpha = float(laplace_alpha)

        # 20x20 Empirical transition count matrix
        self.transition_counts = np.zeros((num_bands, num_bands), dtype=np.float64)
        self.last_active_band: int | None = None

    def _get_hopping_prior(self, origin_band: int | None) -> np.ndarray:
        """Returns conditional probability vector P(j | origin_band)."""
        if origin_band is None:
            return np.full(self.num_bands, 1.0 / self.num_bands, dtype=np.float64)

        row = self.transition_counts[origin_band] + self.laplace_alpha
        # Zero out self-transition because a hop implies leaving the current band
        row_copy = row.copy()
        row_copy[origin_band] = 0.0
        row_sum = float(row_copy.sum())
        if row_sum > 1e-12:
            return row_copy / row_sum
        return np.full(self.num_bands, 1.0 / self.num_bands, dtype=np.float64)

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

        # 2. Hopping prior
        # If the channel went dark just after being active, an agile hop likely occurred!
        is_fresh_hop = (not self.last_detected) and (self.consecutive_misses == 1) and (self.last_active_band is not None)
        hopping_prior = self._get_hopping_prior(self.last_active_band) if is_fresh_hop else np.zeros(self.num_bands, dtype=np.float64)

        # 3. Switching costs
        switch_costs = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            switch_costs = self.switch_penalty * (
                np.abs(np.arange(self.num_bands) - int(self.last_band))
                / max(1, self.num_bands - 1)
            )

        is_signal_active = self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and self.consecutive_dwell_steps >= 2
        )

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
            # Guide exploration using empirical hopping transition prior
            explore_scores = uncertainty + self.markov_prior_weight * hopping_prior - switch_costs
            selected_band = int(np.argmax(explore_scores))
            self.explore_decisions += 1
            self.last_governing_policy = "markov_explore"
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
                + self.markov_prior_weight * hopping_prior
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
                self.last_governing_policy = "exploit"

        self.last_band = selected_band
        return selected_band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        super().update(band, reward, obs_dict)
        det = bool(obs_dict and obs_dict.get("detected", False))

        if det:
            # If we recently had an active signal at another band, record transition
            if self.last_active_band is not None and self.last_active_band != band:
                self.transition_counts[self.last_active_band, band] += 1.0
            self.last_active_band = band
        else:
            if self.consecutive_misses > 3:
                self.last_active_band = None


def build_dwell_dual_markov_hop(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualMarkovHopScheduler:
    return DwellDualMarkovHopScheduler(num_bands=num_bands, seed=seed)
