"""Model: Dwell-Dual + Multi-Scale Dual-Horizon Hawkes.

Upgrades the single-scale Hawkes process with a two-horizon temporal kernel:
- Real radar emitters operate on two fundamentally different timescales:
  1. Micro-scale (intra-burst coherence, beta_fast ~ 1.0): High pulse density within an
     active illumination dwell.
  2. Macro-scale (inter-scan antenna revisit, beta_slow ~ 0.05): Periodic revisits as
     the mechanical or electronic search volume completes a 360-degree rotation.
- Two-scale intensity formulation:
    lambda_b(t) = mu_b + sum_{t_k < t} [ alpha_fast * exp(-beta_fast * dt) + alpha_slow * exp(-beta_slow * dt) ]
- Fast component provides tight dwell lock during live pulse bursts.
- Slow component biases cognitive scouting towards previously active emitter tracks.
"""
from __future__ import annotations

import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualMultiScaleHawkesScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy augmented with Multi-Scale Dual-Horizon Hawkes."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        alpha_fast: float = 0.60,
        beta_fast: float = 0.90,
        alpha_slow: float = 0.30,
        beta_slow: float = 0.06,
        baseline_mu: float = 0.05,
        hawkes_weight: float = 0.50,
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
        self.alpha_fast = float(alpha_fast)
        self.beta_fast = float(beta_fast)
        self.alpha_slow = float(alpha_slow)
        self.beta_slow = float(beta_slow)
        self.baseline_mu = float(baseline_mu)
        self.hawkes_weight = float(hawkes_weight)

        self.current_step = 0
        self.event_history: list[list[int]] = [[] for _ in range(num_bands)]

    def _compute_multiscale_intensity(self) -> tuple[np.ndarray, np.ndarray]:
        """Computes fast intensity and combined total intensity per band."""
        fast_intensities = np.zeros(self.num_bands, dtype=np.float64)
        total_intensities = np.full(self.num_bands, self.baseline_mu, dtype=np.float64)
        t_now = self.current_step

        for b in range(self.num_bands):
            events = self.event_history[b]
            if not events:
                continue
            events_arr = np.array(events, dtype=np.float64)
            dt = t_now - events_arr
            valid = (dt >= 0) & (dt < 40.0)
            if np.any(valid):
                dt_valid = dt[valid]
                f_comp = np.sum(self.alpha_fast * np.exp(-self.beta_fast * dt_valid))
                s_comp = np.sum(self.alpha_slow * np.exp(-self.beta_slow * dt_valid))
                fast_intensities[b] = float(f_comp)
                total_intensities[b] += float(f_comp + s_comp)

        return fast_intensities, total_intensities

    def select_band(self) -> int:
        self.current_step += 1

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

        # 2. Multi-scale Hawkes Evaluation
        fast_intensities, total_intensities = self._compute_multiscale_intensity()

        # 3. Switching costs
        switch_costs = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            switch_costs = self.switch_penalty * (
                np.abs(np.arange(self.num_bands) - int(self.last_band))
                / max(1, self.num_bands - 1)
            )

        current_fast_intensity = fast_intensities[self.last_band] if self.last_band is not None else 0.0
        is_signal_active = self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and current_fast_intensity > 0.15
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
                + self.hawkes_weight * np.clip(total_intensities, 0.0, 2.0)
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
        if obs_dict and obs_dict.get("detected", False):
            history = self.event_history[band]
            history.append(self.current_step)
            if len(history) > 30:
                self.event_history[band] = history[-30:]


def build_dwell_dual_multiscale_hawkes(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualMultiScaleHawkesScheduler:
    return DwellDualMultiScaleHawkesScheduler(num_bands=num_bands, seed=seed)
