"""Model: Dwell-Dual + Adaptive CFAR Noise Gate.

Specifically targets the harsh clutter / low-SNR false alarm bottleneck:
- In harsh Rayleigh fading environments, thermal noise peaks trigger binary detector hits.
- Standard Dwell-Dual locks onto these false alarms with dwell_inertia, accumulating
  switching penalties and missing genuine signals elsewhere.
- Dwell-Dual + CFAR maintains an online recursive estimate of the clutter/noise floor:
    sigma_b^2 <- (1 - eta) * sigma_b^2 + eta * quality
- Implements an Adaptive Cell-Averaging Constant False Alarm Rate (CFAR) Gate:
    If detected is True, but quality < threshold (or SCR is low),
    the detection is classified as CLUTTER.
- Effect: Dwell lock is suppressed on clutter spikes; receiver immediately resumes scanning.
"""
from __future__ import annotations

import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS


class DwellDualCFARScheduler(SmartScanProductionScheduler):
    """Dwell-Dual Policy augmented with Adaptive CFAR Noise/Clutter Gating."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        cfar_min_quality: float = 0.45,
        noise_adapt_rate: float = 0.05,
        cfar_guard_margin: float = 0.15,
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
        self.cfar_min_quality = float(cfar_min_quality)
        self.noise_adapt_rate = float(noise_adapt_rate)
        self.cfar_guard_margin = float(cfar_guard_margin)

        # Online estimated noise/quality baseline per band
        self.estimated_noise_floor = np.full(num_bands, 0.20, dtype=np.float64)
        self.is_cfar_verified = False

    def select_band(self) -> int:
        # 1. Stale-band coverage check from Champion
        coverage_candidates = self._coverage_candidates()
        if coverage_candidates.size:
            # Only interrupt if signal is verified by CFAR gate
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

        # 2. Switching costs
        switch_costs = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            switch_costs = self.switch_penalty * (
                np.abs(np.arange(self.num_bands) - int(self.last_band))
                / max(1, self.num_bands - 1)
            )

        # 3. CFAR-gated signal activity condition
        # A signal is only treated as active for dwell locking if verified by CFAR
        is_signal_active = (self.last_detected and self.is_cfar_verified) or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and self.consecutive_dwell_steps >= 2
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
                - switch_costs
            )

            if self.last_band is not None and is_signal_active:
                # Dwell bonus only granted if CFAR verifies authentic pulse
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

        if obs_dict is not None:
            qual = float(np.asarray(obs_dict.get("quality", 0.0)).reshape(-1)[0])
            det = bool(obs_dict.get("detected", False))

            # Update noise floor tracker
            if not det or qual < 0.20:
                self.estimated_noise_floor[band] = (
                    (1.0 - self.noise_adapt_rate) * self.estimated_noise_floor[band]
                    + self.noise_adapt_rate * qual
                )

            # CFAR test: Quality must exceed both absolute threshold and adaptive noise floor
            cfar_threshold = max(
                self.cfar_min_quality,
                self.estimated_noise_floor[band] + self.cfar_guard_margin
            )
            self.is_cfar_verified = bool(det and (qual >= cfar_threshold))
        else:
            self.is_cfar_verified = False


def build_dwell_dual_cfar(num_bands: int = NUM_BANDS, seed: int | None = None) -> DwellDualCFARScheduler:
    return DwellDualCFARScheduler(num_bands=num_bands, seed=seed)
