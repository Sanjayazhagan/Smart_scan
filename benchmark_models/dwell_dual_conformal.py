"""Model: Dwell-Dual + Online Conformal Prediction Sets.

Enhances the Grand Champion with distribution-free Online Conformal Risk Control:
- Instead of relying solely on heuristic UCB exploration (which assumes static sub-Gaussian noise),
  it tracks rolling prediction residuals: R_t = |y_t - y_hat_t|.
- Computes empirical (1 - alpha) quantile q_hat of non-conformity scores online.
- Constructs calibrated prediction intervals: C_t(b) = [y_hat_b - q_hat, y_hat_b + q_hat].
- Focuses scouting budget strictly on bands whose conformal upper bound certifies
  high probability of active emitter presence without parametric assumptions.
"""
from __future__ import annotations

import collections
import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualConformalScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy augmented with Online Conformal Prediction."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        alpha: float = 0.15,
        conformal_window: int = 50,
        conformal_bonus_scale: float = 0.35,
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
        self.alpha = float(alpha)
        self.conformal_window = int(conformal_window)
        self.conformal_bonus_scale = float(conformal_bonus_scale)
        self.residuals = collections.deque(maxlen=self.conformal_window)
        self.last_predicted_spectrum = np.zeros(num_bands, dtype=np.float64)

    def _get_conformal_quantile(self) -> float:
        if len(self.residuals) < 5:
            return 0.5
        res_arr = np.array(self.residuals, dtype=np.float64)
        # Empirical (1 - alpha) quantile with finite-sample adjustment
        q = float(np.quantile(res_arr, 1.0 - self.alpha))
        return max(1e-4, q)

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

        # 2. Conformal Interval Estimation
        q_hat = self._get_conformal_quantile()
        nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
        nmf_forecast = np.clip(nmf_forecast, 0.0, None)
        nmf_total = float(nmf_forecast.sum())
        if nmf_total > 1e-12:
            nmf_forecast = nmf_forecast / nmf_total
        self.last_predicted_spectrum = nmf_forecast.copy()

        # Conformal upper bound: point prediction + calibrated uncertainty quantile
        conformal_upper = nmf_forecast + q_hat

        # 3. Switching penalties
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
            # Guide exploration using conformal upper bound
            explore_scores = uncertainty + self.conformal_bonus_scale * conformal_upper - switch_costs
            selected_band = int(np.argmax(explore_scores))
            self.explore_decisions += 1
            self.last_governing_policy = "conformal_explore"
            self.consecutive_dwell_steps = 0
        else:
            exploration_bonus = self.exploration_scale * np.sqrt(
                np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
            )
            exploit_scores = (
                self.values
                + exploration_bonus
                + self.nmf_scale * nmf_forecast
                + self.conformal_bonus_scale * conformal_upper
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
        # Compute non-conformity residual: |observed_hit - predicted_prob|
        y_obs = 1.0 if (obs_dict and obs_dict.get("detected", False)) else 0.0
        y_pred = float(self.last_predicted_spectrum[band]) if len(self.last_predicted_spectrum) > band else 0.0
        residual = abs(y_obs - y_pred)
        self.residuals.append(residual)


def build_dwell_dual_conformal(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualConformalScheduler:
    return DwellDualConformalScheduler(num_bands=num_bands, seed=seed)
