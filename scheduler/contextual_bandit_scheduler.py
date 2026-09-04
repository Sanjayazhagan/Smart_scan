"""Contextual Bandit (LinUCB) Policy Meta-Arbitrator for Cognitive Spectrum Scanning.

Candidate 1 Architecture:
Instead of static heuristic thresholding (e.g. max_uncertainty >= 0.35), LinUCB dynamically
learns which underlying sub-policy to activate based on real-time channel context:
- Arm 0: Dwell Lock (Hold active transmission on current band)
- Arm 1: NMF Spectral Cluster (Exploit multi-emitter co-occurrence)
- Arm 2: Lazy Uncertainty Scout (Probe highest epistemic/ensemble uncertainty)
- Arm 3: Local Neighbor Search (Shift to nearby channels to minimize PLL retuning)

Context Vector x_t in R^8:
- x[0]: Sliding-window hit rate (recent 20 steps)
- x[1]: Last observed signal quality [0, 1]
- x[2]: Maximum system uncertainty max(U)
- x[3]: Mean system uncertainty mean(U)
- x[4]: Normalized consecutive misses (min(misses, 5) / 5.0)
- x[5]: Normalized dwell duration (min(dwell_steps, 10) / 10.0)
- x[6]: Last signal presence indicator (1.0 if detected else 0.0)
- x[7]: Bias term (1.0)

Online Updates:
Uses Sherman-Morrison rank-1 updates for O(d^2) matrix inversion maintenance (~10 microseconds).
Zero offline pre-training required; adapts in real time to each operational mission seed.
"""

from __future__ import annotations

from collections import deque
import numpy as np

from scheduler.baselines import BaseScheduler
from scheduler.dual_policy_uncertainty_scheduler import EnsembleUncertaintyEstimator
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.track2_core import NUM_BANDS


class LinUCBPolicyArbitrator:
    """Disjoint LinUCB Contextual Bandit with Sherman-Morrison covariance tracking."""

    def __init__(self, num_arms: int = 4, context_dim: int = 8, alpha: float = 0.25, l2_reg: float = 1.0):
        self.num_arms = int(num_arms)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)

        # Inverse covariance matrices A_inv initialized to (1/lambda) * I
        self.A_inv = np.array([np.eye(self.context_dim, dtype=np.float64) / self.l2_reg for _ in range(self.num_arms)])
        self.b = np.zeros((self.num_arms, self.context_dim), dtype=np.float64)
        self.arm_counts = np.zeros(self.num_arms, dtype=np.int32)

    def reset(self):
        for a in range(self.num_arms):
            self.A_inv[a] = np.eye(self.context_dim, dtype=np.float64) / self.l2_reg
        self.b.fill(0.0)
        self.arm_counts.fill(0)

    def select_arm(self, context: np.ndarray) -> tuple[int, np.ndarray]:
        """Returns chosen arm index and UCB scores across all arms."""
        scores = np.zeros(self.num_arms, dtype=np.float64)
        x = context.reshape(-1, 1)  # [d, 1]

        for a in range(self.num_arms):
            A_inv_a = self.A_inv[a]
            theta_a = A_inv_a @ self.b[a].reshape(-1, 1)  # [d, 1]
            mean_est = float((theta_a.T @ x).item())
            var_est = float((x.T @ A_inv_a @ x).item())
            ucb_bonus = self.alpha * np.sqrt(max(1e-8, var_est))
            scores[a] = mean_est + ucb_bonus

        chosen_arm = int(np.argmax(scores))
        return chosen_arm, scores

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Sherman-Morrison rank-1 update to A_inv: O(d^2) complexity."""
        x = context.reshape(-1, 1)  # [d, 1]
        A_inv_a = self.A_inv[arm]

        # Sherman-Morrison: (A + x x^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
        v = A_inv_a @ x  # [d, 1]
        denom = float((1.0 + x.T @ v).item())
        self.A_inv[arm] = A_inv_a - (v @ v.T) / max(1e-8, denom)
        self.b[arm] += float(reward) * context
        self.arm_counts[arm] += 1


class ContextualBanditScheduler(BaseScheduler):
    """SmartScan Candidate 1: Contextual Bandit (LinUCB) Policy Meta-Arbitrator."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        alpha: float = 0.25,
        nmf_scale: float = 1.00,
        nmf_components: int = 4,
        nmf_window: int = 30,
        nmf_recompute_every: int = 2,
        switch_penalty: float = 0.08,
        dwell_inertia: float = 1.30,
        fading_grace_steps: int = 1,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(num_bands)
        self.switch_penalty = float(switch_penalty)
        self.dwell_inertia = float(dwell_inertia)
        self.fading_grace_steps = int(fading_grace_steps)
        self.nmf_scale = float(nmf_scale)
        self.rng = np.random.default_rng(seed)

        # Core Engines
        self.nmf = NMFScheduler(
            num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
        )
        self.uncertainty_estimator = EnsembleUncertaintyEstimator(num_bands=num_bands)
        self.meta_bandit = LinUCBPolicyArbitrator(num_arms=4, context_dim=8, alpha=alpha)

        # Context features & history
        self.history_len = 20
        self.recent_hits = deque(maxlen=self.history_len)
        self.consecutive_misses = 0
        self.consecutive_dwell_steps = 0
        self.last_detected = False
        self.last_quality = 0.0
        self.last_arm = 0
        self.last_context = np.zeros(8, dtype=np.float64)

        # UCB Value Tracking
        self.decay = 0.985
        self.value_lr = 0.20
        self.exploration_scale = 0.75
        self.counts = np.zeros(self.num_bands, dtype=np.float64)
        self.values = np.zeros(self.num_bands, dtype=np.float64)
        self.total_observations = 0.0

    def reset(self):
        self.meta_bandit.reset()
        self.uncertainty_estimator.reset()
        self.recent_hits.clear()
        self.consecutive_misses = 0
        self.consecutive_dwell_steps = 0
        self.last_band = None
        self.last_detected = False
        self.last_quality = 0.0
        self.counts.fill(0.0)
        self.values.fill(0.0)
        self.total_observations = 0.0

    def _build_context(self, max_unc: float, mean_unc: float) -> np.ndarray:
        """Constructs 8-dimensional normalized context vector x_t."""
        hit_rate = float(np.mean(self.recent_hits)) if len(self.recent_hits) > 0 else 0.0
        ctx = np.array([
            float(np.clip(hit_rate, 0.0, 1.0)),
            float(np.clip(self.last_quality, 0.0, 1.0)),
            float(np.clip(max_unc, 0.0, 1.0)),
            float(np.clip(mean_unc, 0.0, 1.0)),
            float(min(self.consecutive_misses, 5) / 5.0),
            float(min(self.consecutive_dwell_steps, 10) / 10.0),
            1.0 if self.last_detected else 0.0,
            1.0,  # Bias
        ], dtype=np.float64)
        return ctx

    def select_band(self) -> int:
        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            self.last_band = band
            return band

        uncertainty = self.uncertainty_estimator.get_uncertainty()
        max_unc = float(np.max(uncertainty))
        mean_unc = float(np.mean(uncertainty))

        # Build context vector
        ctx = self._build_context(max_unc, mean_unc)
        self.last_context = ctx

        # Switching cost vector from last band
        switch_costs = np.zeros(self.num_bands, dtype=np.float64)
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_costs[b] = self.switch_penalty * (abs(b - self.last_band) / max(1, self.num_bands - 1))

        # NMF spectral forecast
        nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
        nmf_forecast = np.clip(nmf_forecast, 0.0, None)
        tot_nmf = float(nmf_forecast.sum())
        if tot_nmf > 1e-12:
            nmf_forecast = nmf_forecast / tot_nmf

        # Sub-policy band recommendations:
        # Arm 0: Dwell Lock
        band_dwell = self.last_band if self.last_band is not None else 0

        # Arm 1: NMF Spectral Cluster (best global co-occurrence penalized by jump distance)
        scores_nmf = self.nmf_scale * nmf_forecast - switch_costs
        band_nmf = int(np.argmax(scores_nmf))

        # Arm 2: Uncertainty Scout (highest epistemic uncertainty penalized by jump distance)
        scores_scout = uncertainty - switch_costs
        band_scout = int(np.argmax(scores_scout))

        # Arm 3: Local Neighbor Search (nearest 3 bands with highest values)
        if self.last_band is not None:
            local_mask = np.zeros(self.num_bands, dtype=bool)
            for offset in (-2, -1, 0, 1, 2):
                nb = self.last_band + offset
                if 0 <= nb < self.num_bands:
                    local_mask[nb] = True
            local_scores = np.where(local_mask, self.values + 0.1 * nmf_forecast - switch_costs, -1e9)
            band_local = int(np.argmax(local_scores))
        else:
            band_local = 0

        arm_candidates = [band_dwell, band_nmf, band_scout, band_local]

        # LinUCB meta-arbitration
        chosen_arm, _ = self.meta_bandit.select_arm(ctx)

        # Force Arm 0 (Dwell) if active pulse is detected or in debounce window (The Discipline Principle)
        is_signal_active = self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and self.consecutive_dwell_steps >= 2
        )
        if is_signal_active and self.last_band is not None:
            chosen_arm = 0

        self.last_arm = chosen_arm
        selected_band = arm_candidates[chosen_arm]

        if selected_band == self.last_band and is_signal_active:
            self.consecutive_dwell_steps += 1
        else:
            self.consecutive_dwell_steps = 0

        self.last_band = selected_band
        return selected_band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        self.counts *= self.decay
        self.total_observations = self.total_observations * self.decay + 1.0
        self.counts[int(band)] += 1.0

        det = 0.0
        qual = 0.0
        if obs_dict is not None:
            det = 1.0 if obs_dict.get("detected", False) else 0.0
            qual = float(np.asarray(obs_dict.get("quality", 0.0)).reshape(-1)[0])
            self.last_detected = bool(det > 0.5)
            self.last_quality = qual
            self.recent_hits.append(int(self.last_detected))
            if self.last_detected:
                self.consecutive_misses = 0
            else:
                self.consecutive_misses += 1
        else:
            self.last_detected = False
            self.last_quality = 0.0
            self.consecutive_misses += 1
            self.recent_hits.append(0)

        # Update NMF and Uncertainty
        self.nmf.update(band, 0.0, obs_dict)
        self.uncertainty_estimator.update(band, det)

        # Update UCB empirical values
        obs_val = float(np.clip(0.10 + 0.90 * qual, 0.0, 1.0)) if det > 0.5 else 0.02
        self.values[int(band)] = (1.0 - self.value_lr) * self.values[int(band)] + self.value_lr * obs_val

        # Update Contextual Bandit with normalized reward
        norm_r = float(np.clip(reward, -1.0, 1.0))
        self.meta_bandit.update(self.last_arm, self.last_context, norm_r)
