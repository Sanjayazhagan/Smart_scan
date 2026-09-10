"""Model: Dwell-Dual + Hawkes Self-Exciting Point Process.

Models radar pulse emissions as self-exciting point processes:
- In real radar systems, pulses arrive in temporal bursts (e.g. search dwells, track illuminations).
- Standard Markov or Poisson assumptions fail because the arrival of one pulse dramatically
  increases the conditional probability of subsequent pulses in that band.
- A univariate Hawkes process models conditional intensity per band:
    lambda_b(t) = mu_b + sum_{t_k < t} alpha * exp(-beta * (t - t_k))
- The Hawkes intensity dynamically modulates dwell lock:
  1. High lambda_b: Firmly locks receiver on active burst.
  2. Decayed lambda_b: Releases dwell lock early, avoiding dead-time waiting between bursts.
"""
from __future__ import annotations

import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualHawkesScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy augmented with Hawkes Self-Exciting Process."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        hawkes_alpha: float = 0.80,   # Excitation jump size
        hawkes_beta: float = 0.35,    # Exponential decay rate
        baseline_mu: float = 0.05,    # Background base rate
        hawkes_weight: float = 0.50,  # Contribution to selection scores
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
        self.hawkes_alpha = float(hawkes_alpha)
        self.hawkes_beta = float(hawkes_beta)
        self.baseline_mu = float(baseline_mu)
        self.hawkes_weight = float(hawkes_weight)

        self.current_step = 0
        # Store recent detection timestamps per band (up to 30 events)
        self.event_history: list[list[int]] = [[] for _ in range(num_bands)]

    def _compute_hawkes_intensity(self) -> np.ndarray:
        """Computes current intensity lambda_b(t) for all bands."""
        intensities = np.full(self.num_bands, self.baseline_mu, dtype=np.float64)
        t_now = self.current_step

        for b in range(self.num_bands):
            events = self.event_history[b]
            if not events:
                continue
            events_arr = np.array(events, dtype=np.float64)
            dt = t_now - events_arr
            # Only consider events within 5 decay half-lives to maintain speed
            valid = (dt >= 0) & (dt < 25.0)
            if np.any(valid):
                decay = np.exp(-self.hawkes_beta * dt[valid])
                intensities[b] += self.hawkes_alpha * np.sum(decay)

        return intensities

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

        # 2. Hawkes Intensity Evaluation
        intensities = self._compute_hawkes_intensity()

        # 3. Switching penalties
        switch_costs = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            switch_costs = self.switch_penalty * (
                np.abs(np.arange(self.num_bands) - int(self.last_band))
                / max(1, self.num_bands - 1)
            )

        # Hawkes-aware signal activity: active if detected OR if burst intensity is still high
        current_band_intensity = intensities[self.last_band] if self.last_band is not None else 0.0
        is_signal_active = self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and current_band_intensity > (self.baseline_mu + 0.20)
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

            # Integrate Hawkes point-process intensity into exploitation score
            exploit_scores = (
                self.values
                + exploration_bonus
                + self.nmf_scale * nmf_forecast
                + self.hawkes_weight * np.clip(intensities, 0.0, 2.0)
                - switch_costs
            )

            if self.last_band is not None and is_signal_active:
                # Modulate dwell bonus by instantaneous burst intensity
                intensity_factor = min(2.0, max(0.5, current_band_intensity / (self.baseline_mu + 1e-3)))
                dwell_bonus = self.dwell_inertia * max(0.4, self.last_quality) * (0.8 + 0.2 * intensity_factor)
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
            # Record pulse arrival event timestamp
            history = self.event_history[band]
            history.append(self.current_step)
            if len(history) > 30:
                self.event_history[band] = history[-30:]


def build_dwell_dual_hawkes(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualHawkesScheduler:
    return DwellDualHawkesScheduler(num_bands=num_bands, seed=seed)
