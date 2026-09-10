"""Model: Dwell-Dual + Hawkes-CFAR Integrated Champion.

Synthesizes the two highest-performing frontier mechanisms:
1. Hawkes Self-Exciting Point Process:
   - Captures radar pulse burst excitation: lambda_b(t) = mu_b + sum alpha * exp(-beta * dt).
   - Firmly holds dwell lock during active multi-pulse bursts.
   - Terminates dwell early when burst intensity decays, eliminating idle waiting.
2. Adaptive CFAR Clutter Gate:
   - Filters out Rayleigh clutter spikes and thermal false alarms.
   - Prevents noise spikes from falsely triggering Hawkes burst excitation in harsh environments.
   - Clutter-rejected detections do not pollute event history and do not trigger dwell inertia.
"""
from __future__ import annotations

import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualHawkesCFARScheduler(SmartScanProductionScheduler):
    """Grand Hybrid: Dwell-Dual with Hawkes Burst Dynamics and CFAR Clutter Gating."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        hawkes_alpha: float = 0.80,
        hawkes_beta: float = 0.35,
        baseline_mu: float = 0.05,
        hawkes_weight: float = 0.45,
        cfar_min_quality: float = 0.40,
        noise_adapt_rate: float = 0.05,
        cfar_guard_margin: float = 0.12,
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

        self.cfar_min_quality = float(cfar_min_quality)
        self.noise_adapt_rate = float(noise_adapt_rate)
        self.cfar_guard_margin = float(cfar_guard_margin)

        self.current_step = 0
        self.event_history: list[list[int]] = [[] for _ in range(num_bands)]
        self.estimated_noise_floor = np.full(num_bands, 0.20, dtype=np.float64)
        self.is_cfar_verified = False

    def _compute_hawkes_intensity(self) -> np.ndarray:
        intensities = np.full(self.num_bands, self.baseline_mu, dtype=np.float64)
        t_now = self.current_step

        for b in range(self.num_bands):
            events = self.event_history[b]
            if not events:
                continue
            events_arr = np.array(events, dtype=np.float64)
            dt = t_now - events_arr
            valid = (dt >= 0) & (dt < 25.0)
            if np.any(valid):
                decay = np.exp(-self.hawkes_beta * dt[valid])
                intensities[b] += self.hawkes_alpha * np.sum(decay)

        return intensities

    def select_band(self) -> int:
        self.current_step += 1

        # 1. Stale-band coverage check
        coverage_candidates = self._coverage_candidates()
        if coverage_candidates.size:
            # Only defer coverage if currently on an authentic CFAR-verified signal
            if self._should_interrupt_coverage() and self.is_cfar_verified:
                self.dwell_decisions += 1
                self.consecutive_dwell_steps += 1
                self.coverage_interrupt_dwells += 1
                self.last_governing_policy = "coverage_interrupt_dwell"
                return int(self.last_band)

            band = self._choose_coverage_candidate(coverage_candidates)
            self.last_band = band
            self.last_governing_policy = "smart_forced_coverage"
            return band

        # 2. Hawkes Intensity
        intensities = self._compute_hawkes_intensity()

        # 3. Switching costs
        switch_costs = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            switch_costs = self.switch_penalty * (
                np.abs(np.arange(self.num_bands) - int(self.last_band))
                / max(1, self.num_bands - 1)
            )

        current_band_intensity = intensities[self.last_band] if self.last_band is not None else 0.0
        # Active only if verified by CFAR and burst excitation persists
        is_signal_active = (self.last_detected and self.is_cfar_verified) or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and current_band_intensity > (self.baseline_mu + 0.20)
            and self.is_cfar_verified
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
                + self.hawkes_weight * np.clip(intensities, 0.0, 2.0)
                - switch_costs
            )

            if self.last_band is not None and is_signal_active:
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

        if obs_dict is not None:
            qual = float(np.asarray(obs_dict.get("quality", 0.0)).reshape(-1)[0])
            det = bool(obs_dict.get("detected", False))

            # Recursive noise floor adaptation
            if not det or qual < 0.20:
                self.estimated_noise_floor[band] = (
                    (1.0 - self.noise_adapt_rate) * self.estimated_noise_floor[band]
                    + self.noise_adapt_rate * qual
                )

            # CFAR decision
            cfar_threshold = max(
                self.cfar_min_quality,
                self.estimated_noise_floor[band] + self.cfar_guard_margin
            )
            self.is_cfar_verified = bool(det and (qual >= cfar_threshold))

            # Crucial: Only CFAR-verified authentic hits feed Hawkes self-excitation!
            if self.is_cfar_verified:
                history = self.event_history[band]
                history.append(self.current_step)
                if len(history) > 30:
                    self.event_history[band] = history[-30:]
        else:
            self.is_cfar_verified = False


def build_dwell_dual_hawkes_cfar(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualHawkesCFARScheduler:
    return DwellDualHawkesCFARScheduler(num_bands=num_bands, seed=seed)
