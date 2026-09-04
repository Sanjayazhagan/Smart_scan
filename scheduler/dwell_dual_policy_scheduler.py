"""Dwell-Aware Dual-Policy Scheduler (The Disciplined Scout Architecture).

Combines:
1. Mathematical Exploit Index (Multi-Factor Value Function):
   - Active Lock-On Dwell Term: Heavily anchors receiver on the current band while signal is active.
   - NMF Reconstructed Spectrum Energy: Captures global spectral co-occurrence.
   - UCB Quality & Information Gain: Balanced exploration bonus.
   - Physical Distance Retuning Penalty: Proportional to frequency jump distance.

2. "Lazy Scout" Explore Policy:
   - NEVER interrupts an active transmission (0% exploration while current channel detects signal).
   - Only triggers when current channel goes dark / miss AND system uncertainty is elevated.
   - Evaluates uncertainty-to-switching-cost ratio to pick nearby high-uncertainty bands.
"""

from __future__ import annotations

import numpy as np

from scheduler.dual_policy_uncertainty_scheduler import EnsembleUncertaintyEstimator
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.world_model_ucb import WorldModelUCBScheduler


class DwellDualPolicyScheduler(WorldModelUCBScheduler):
    """Dual-Policy Scheduler with Dwell Lock-On and Conditional Scouting."""

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
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            num_bands,
            world_model_scale=0.0,
            **kwargs,
        )
        self.nmf_scale = float(nmf_scale)
        self.switch_penalty = float(switch_penalty)
        self.dwell_inertia = float(dwell_inertia)
        self.explore_budget_prob = float(explore_budget_prob)
        self.uncertainty_trigger_threshold = float(uncertainty_trigger_threshold)
        self.fading_grace_steps = int(fading_grace_steps)
        self.rng = np.random.default_rng(seed)

        # 1. Exploit Core: NMF Decomposition
        self.nmf = NMFScheduler(
            num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
        )

        # 2. Explore Core: Multi-Horizon Uncertainty Estimator
        self.uncertainty_estimator = EnsembleUncertaintyEstimator(num_bands=num_bands)

        # State tracking
        self.consecutive_misses = 0
        self.last_detected = False
        self.last_quality = 0.0
        self.consecutive_dwell_steps = 0
        self.exploit_decisions = 0
        self.explore_decisions = 0
        self.dwell_decisions = 0
        self.last_governing_policy = "none"

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

        # -------------------------------------------------------------
        # 1. "Lazy Scout" Arbitration:
        # If currently locked on an active pulse or within fading grace window, DO NOT SCOUT.
        # Only consider scouting if current band went silent AND uncertainty is elevated.
        # -------------------------------------------------------------
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

        if is_explore:
            # EXPLORE POLICY: Highest uncertainty, penalized by jump distance
            explore_scores = uncertainty - switch_costs
            selected_band = int(np.argmax(explore_scores))
            self.explore_decisions += 1
            self.last_governing_policy = "explore"
            self.consecutive_dwell_steps = 0
        else:
            # EXPLOIT POLICY: Multi-factor Mathematical Index
            nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
            nmf_forecast = np.clip(nmf_forecast, 0.0, None)
            nmf_total = float(nmf_forecast.sum())
            if nmf_total > 1e-12:
                nmf_forecast = nmf_forecast / nmf_total

            exploration_bonus = self.exploration_scale * np.sqrt(
                np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
            )

            # Mathematical multi-factor score:
            exploit_scores = (
                self.values
                + exploration_bonus
                + self.nmf_scale * nmf_forecast
                - switch_costs
            )

            # Active Lock-On Dwell Term:
            # If we were detecting signal with good quality, apply strong inertia to stay
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
