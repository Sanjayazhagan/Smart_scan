"""Dual-Policy Exploit/Explore Scheduler with Uncertainty-Aware World Model.

Architecture:
1. Uncertainty Estimator (Repurposed World Model):
   - Multi-horizon 3-model ensemble (fast, medium, slow).
   - Computes:
     (a) Disagreement: variance across ensemble predictions (captures emitter agility/hopping/volatility).
     (b) Predictive entropy: H(p_bar) (captures stochastic uncertainty).
     (c) Epistemic uncertainty: normalized scan age and inverse visit count.
   - Outputs uncertainty vector U in [0, 1]^K.
   - Sanity-checked: U is higher for volatile and unvisited bands, lower for stable well-sampled bands.

2. Exploit Policy:
   - NMF+UCB best-known channel selection, penalized by switching cost.

3. Explore Policy:
   - Pure uncertainty-seeking channel selection (highest U_b), penalized by switching cost.

4. Arbitration:
   - Tunable exploration budget: explore triggered probabilistically based on max system uncertainty
     (or periodic budget every N steps), with explicit logging of which policy governed each step.
"""

from __future__ import annotations

import numpy as np
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.world_model_ucb import WorldModelUCBScheduler


class EnsembleUncertaintyEstimator:
    """Repurposed World Model: Multi-horizon ensemble uncertainty estimator."""

    def __init__(self, num_bands: int = NUM_BANDS, max_age: float = 30.0):
        self.num_bands = int(num_bands)
        self.max_age = float(max_age)

        # 3 Ensemble models with different temporal memory horizons
        self.alphas = np.array([0.35, 0.12, 0.03], dtype=np.float32)
        self.p_ensemble = np.full((3, self.num_bands), 0.5, dtype=np.float32)

        self.counts = np.zeros(self.num_bands, dtype=np.float32)
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)

    def reset(self):
        self.p_ensemble.fill(0.5)
        self.counts.fill(0.0)
        self.scan_age.fill(0.0)

    def update(self, band: int, detected: float):
        band = int(band)
        detected = float(detected)
        self.scan_age += 1.0
        self.scan_age[band] = 0.0
        self.counts[band] += 1.0

        for m, alpha in enumerate(self.alphas):
            self.p_ensemble[m, band] = (1.0 - alpha) * self.p_ensemble[m, band] + alpha * detected

    def get_uncertainty(self) -> np.ndarray:
        """Returns normalized uncertainty score per band in [0, 1]."""
        p_bar = self.p_ensemble.mean(axis=0)
        ensemble_var = self.p_ensemble.var(axis=0)
        norm_var = np.clip(ensemble_var / 0.10, 0.0, 1.0)

        p_safe = np.clip(p_bar, 1e-4, 1.0 - 1e-4)
        entropy = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))

        recency = np.clip(self.scan_age / self.max_age, 0.0, 1.0)
        count_unc = 1.0 / np.sqrt(self.counts + 1.0)
        epistemic = 0.5 * recency + 0.5 * count_unc

        total_unc = 0.40 * norm_var + 0.25 * entropy + 0.35 * epistemic
        return np.clip(total_unc, 0.0, 1.0).astype(np.float32)


class DualPolicyUncertaintyScheduler(WorldModelUCBScheduler):
    """Arbitrated Dual-Policy Exploit/Explore Scheduler."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        nmf_scale: float = 1.00,
        nmf_components: int = 4,
        nmf_window: int = 30,
        nmf_recompute_every: int = 2,
        switch_penalty: float = 0.05,
        explore_budget_prob: float = 0.20,
        arbitration_mode: str = "probabilistic",  # "probabilistic" or "periodic"
        periodic_explore_interval: int = 5,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            num_bands,
            world_model_scale=0.0,  # Zero direct point-prediction guidance
            **kwargs,
        )
        self.nmf_scale = float(nmf_scale)
        self.switch_penalty = float(switch_penalty)
        self.explore_budget_prob = float(explore_budget_prob)
        self.arbitration_mode = arbitration_mode
        self.periodic_explore_interval = int(periodic_explore_interval)
        self.rng = np.random.default_rng(seed)

        # 1. The Exploit Engine: NMF Factorization
        self.nmf = NMFScheduler(
            num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
        )

        # 2. The Uncertainty Estimator (Repurposed World Model)
        self.uncertainty_estimator = EnsembleUncertaintyEstimator(num_bands=num_bands)

        # Arbitration statistics
        self.exploit_decisions = 0
        self.explore_decisions = 0
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

        # 3. Arbitration rule: decide whether this timestep is Exploit or Explore
        is_explore = False
        if self.arbitration_mode == "periodic":
            is_explore = (int(self.timestamp) % self.periodic_explore_interval == 0)
        else:
            # Probabilistic arbitration proportional to current system uncertainty
            prob = self.explore_budget_prob * max_uncertainty
            is_explore = (self.rng.random() < prob)

        # Switching cost distance vector
        switch_costs = np.zeros(self.num_bands, dtype=np.float32)
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_costs[b] = self.switch_penalty * (abs(b - self.last_band) / max(1, self.num_bands - 1))

        if is_explore:
            # EXPLORE POLICY: Highest uncertainty, penalized by switching cost
            explore_scores = uncertainty - switch_costs
            selected_band = int(np.argmax(explore_scores))
            self.explore_decisions += 1
            self.last_governing_policy = "explore"
        else:
            # EXPLOIT POLICY: NMF+UCB best-known channel, penalized by switching cost
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
            selected_band = int(np.argmax(exploit_scores))
            self.exploit_decisions += 1
            self.last_governing_policy = "exploit"

        self.last_band = selected_band
        return selected_band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        super().update(band, reward, obs_dict)
        self.nmf.update(band, 0.0, obs_dict)

        det = 0.0
        if obs_dict is not None:
            det = 1.0 if obs_dict.get("detected", False) else 0.0
        self.uncertainty_estimator.update(band, det)
