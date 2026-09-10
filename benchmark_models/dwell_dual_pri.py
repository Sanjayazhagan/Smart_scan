"""Model: Dwell-Dual + PRI Inter-Arrival Timing Predictor.

Applies Pulse Repetition Interval (PRI) deinterleaving physics to scan scheduling:
- Real radar emitters operate with characteristic pulse periods: T_PRI.
- By tracking consecutive inter-arrival times Delta_t = t_k - t_{k-1}, the scheduler
  estimates the underlying emitter period: T_hat = median(Delta_t).
- Predicts exact future arrival windows: (t_now - t_last) ~ T_hat.
- Provides a preemptive rendezvous boost to bands whose scheduled pulse arrival is imminent,
  minimizing dead-time and intercepting periodic emitters just-in-time.
"""
from __future__ import annotations

import collections
import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualPRIScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy augmented with PRI Inter-Arrival Tracking."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        pri_bonus_scale: float = 0.40,
        pri_tolerance: float = 1.0,
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
        self.pri_bonus_scale = float(pri_bonus_scale)
        self.pri_tolerance = float(pri_tolerance)

        self.current_step = 0
        self.last_hit_step = np.full(num_bands, -999, dtype=int)
        # Store recent Delta_t intervals per band
        self.interval_history: list[collections.deque] = [
            collections.deque(maxlen=15) for _ in range(num_bands)
        ]

    def _get_pri_rendezvous_bonus(self) -> np.ndarray:
        """Computes a rendezvous readiness bonus for bands due for a pulse."""
        bonus = np.zeros(self.num_bands, dtype=np.float64)
        t_now = self.current_step

        for b in range(self.num_bands):
            intervals = self.interval_history[b]
            if len(intervals) < 2:
                continue

            last_t = self.last_hit_step[b]
            if last_t < 0:
                continue

            elapsed = t_now - last_t
            pri_est = float(np.median(intervals))
            if pri_est < 1.0:
                continue

            # Check if elapsed time matches an integer multiple of PRI: elapsed ~ k * pri_est
            # specifically for k = 1 (next consecutive pulse)
            timing_error = abs(elapsed - pri_est)
            if timing_error <= self.pri_tolerance:
                # Gaussian timing readiness curve
                readiness = np.exp(-0.5 * (timing_error / max(0.5, self.pri_tolerance)) ** 2)
                bonus[b] = self.pri_bonus_scale * float(readiness)

        return bonus

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

        # 2. PRI Rendezvous Bonus
        pri_bonuses = self._get_pri_rendezvous_bonus()

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
            explore_scores = uncertainty + pri_bonuses - switch_costs
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
                + pri_bonuses
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
            t_now = self.current_step
            last_t = self.last_hit_step[band]
            if last_t > 0:
                delta_t = t_now - last_t
                if 1 <= delta_t <= 50:
                    self.interval_history[band].append(delta_t)
            self.last_hit_step[band] = t_now


def build_dwell_dual_pri(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualPRIScheduler:
    return DwellDualPRIScheduler(num_bands=num_bands, seed=seed)
