"""SmartScan production champion: Interruptible Dual-Dwell + Smart Stale Ordering.

Final lightweight production policy selected from the September 2026 PDW ablations.

What changed from the original Dwell-Dual policy:
1. Recurring stale-band coverage is retained (discounted counts still decay).
2. A live detection is never abandoned merely because another band became stale.
   The stale-band visit is deferred until the active dwell ends.
3. When multiple stale bands are due, they are ranked by observable urgency rather
   than serviced in fixed array order.

The production policy remains PDW/observation-only.  It does not require I/Q or a
Track-2 neural checkpoint.
"""
from __future__ import annotations

import numpy as np

from scheduler.dual_policy_uncertainty_scheduler import EnsembleUncertaintyEstimator
from scheduler.observation_runtime import ObservationOnlyRuntime
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.world_model_ucb import WorldModelUCBScheduler


class SmartScanProductionScheduler(WorldModelUCBScheduler):
    """Final SmartScan champion: interruptible dwell with smart stale coverage."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        nmf_scale: float = 1.00,
        nmf_components: int = 4,
        nmf_window: int = 30,
        nmf_recompute_every: int = 2,
        switch_penalty: float = 0.08,
        dwell_inertia: float = 1.30,
        explore_budget_prob: float = 0.12,
        uncertainty_trigger_threshold: float = 0.35,
        fading_grace_steps: int = 1,
        stale_threshold: float = 0.5,
        seed: int | None = None,
        **kwargs,
    ):
        # No neural world-model guidance is used by Dwell-Dual.  Avoid creating an
        # unnecessary checkpoint/IQ dependency in the production path.
        kwargs.setdefault("runtime", ObservationOnlyRuntime(num_bands))
        super().__init__(num_bands, world_model_scale=0.0, **kwargs)

        self.nmf_scale = float(nmf_scale)
        self.switch_penalty = float(switch_penalty)
        self.dwell_inertia = float(dwell_inertia)
        self.explore_budget_prob = float(explore_budget_prob)
        self.uncertainty_trigger_threshold = float(uncertainty_trigger_threshold)
        self.fading_grace_steps = int(fading_grace_steps)
        self.stale_threshold = float(stale_threshold)
        self.rng = np.random.default_rng(seed)

        self.nmf = NMFScheduler(
            num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
        )
        self.uncertainty_estimator = EnsembleUncertaintyEstimator(num_bands=num_bands)

        self.consecutive_misses = 0
        self.last_detected = False
        self.last_quality = 0.0
        self.consecutive_dwell_steps = 0
        self.exploit_decisions = 0
        self.explore_decisions = 0
        self.dwell_decisions = 0
        self.coverage_interrupt_dwells = 0
        self.smart_coverage_decisions = 0
        self.last_governing_policy = "none"

    # ------------------------------------------------------------------
    # Final champion addition #1: interruptible recurring coverage
    # ------------------------------------------------------------------
    def _coverage_candidates(self) -> np.ndarray:
        """Bands whose discounted visit count has become stale."""
        return np.flatnonzero(self.counts < self.stale_threshold)

    def _should_interrupt_coverage(self) -> bool:
        """Do not abandon a currently detected signal for a stale-band visit."""
        return bool(self.last_band is not None and self.last_detected)

    # ------------------------------------------------------------------
    # Final champion addition #2: smart stale-band ordering
    # ------------------------------------------------------------------
    def _choose_coverage_candidate(self, candidates: np.ndarray) -> int:
        """Rank stale bands using only observable scheduler state.

        Score terms:
          45% uncertainty
          25% normalized scan age
          20% stale depth below the coverage threshold
          10% prior observed value
          minus retuning distance penalty
        """
        candidates = np.asarray(candidates, dtype=int)
        uncertainty = np.asarray(self.uncertainty_estimator.get_uncertainty(), dtype=np.float64)
        age = np.clip(
            np.asarray(self.uncertainty_estimator.scan_age, dtype=np.float64)
            / max(1e-6, float(self.uncertainty_estimator.max_age)),
            0.0,
            1.0,
        )
        stale_depth = np.clip(
            (self.stale_threshold - self.counts) / max(self.stale_threshold, 1e-6),
            0.0,
            1.0,
        )
        prior_value = np.clip(np.asarray(self.values, dtype=np.float64), 0.0, 1.0)

        switch_cost = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            switch_cost = self.switch_penalty * (
                np.abs(np.arange(self.num_bands) - int(self.last_band))
                / max(1, self.num_bands - 1)
            )

        score = (
            0.45 * uncertainty
            + 0.25 * age
            + 0.20 * stale_depth
            + 0.10 * prior_value
            - switch_cost
        )
        self.smart_coverage_decisions += 1
        return int(candidates[int(np.argmax(score[candidates]))])

    def select_band(self) -> int:
        # Recurring coverage remains part of the champion, but it can no longer
        # abruptly interrupt an active signal.
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

        uncertainty = self.uncertainty_estimator.get_uncertainty()
        max_uncertainty = float(np.max(uncertainty))

        switch_costs = np.zeros(self.num_bands, dtype=np.float32)
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_costs[b] = self.switch_penalty * (
                    abs(b - self.last_band) / max(1, self.num_bands - 1)
                )

        # Normal Dwell-Dual arbitration.
        is_signal_active = self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and self.consecutive_dwell_steps >= 2
        )

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
        self.nmf.update(band, 0.0, obs_dict)

        det = 0.0
        qual = 0.0
        if obs_dict is not None:
            det = 1.0 if obs_dict.get("detected", False) else 0.0
            qual = float(np.asarray(obs_dict.get("quality", 0.0)).reshape(-1)[0])
            self.last_detected = bool(det > 0.5)
            self.last_quality = qual
            if self.last_detected:
                self.consecutive_misses = 0
            else:
                self.consecutive_misses += 1
        else:
            self.last_detected = False
            self.last_quality = 0.0
            self.consecutive_misses += 1

        self.uncertainty_estimator.update(band, det)


# Stable public alias retained for older imports.
DwellDualPolicyScheduler = SmartScanProductionScheduler
