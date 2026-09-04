"""Smart Discounted UCB Candidate Generator for SmartScan V2.

Maintains discounted visitation statistics and computes multi-factor candidate scores
across all 20 bands. Returns ranked Top-K candidates for deeper Expectimax lookahead.
"""

from __future__ import annotations

import numpy as np

from scheduler.track2_core import NUM_BANDS


class SmartDiscountedUCB:
    """Discounted Upper Confidence Bound policy and candidate generator."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        discount_factor: float = 0.985,
        value_lr: float = 0.20,
        exploration_scale: float = 0.75,
        w_prediction: float = 0.35,
        w_age: float = 0.10,
        w_uncertainty: float = 0.15,
        w_investigate: float = 0.20,
        switch_penalty: float = 0.05,
        detection_base: float = 0.10,
        quality_scale: float = 0.90,
        miss_value: float = 0.02,
        default_k: int = 5,
        adaptive_k: bool = False,
    ):
        self.num_bands = int(num_bands)
        self.discount_factor = float(np.clip(discount_factor, 0.80, 1.0))
        self.value_lr = float(np.clip(value_lr, 0.01, 1.0))
        self.exploration_scale = float(exploration_scale)
        self.w_prediction = float(w_prediction)
        self.w_age = float(w_age)
        self.w_uncertainty = float(w_uncertainty)
        self.w_investigate = float(w_investigate)
        self.switch_penalty = float(switch_penalty)
        self.detection_base = float(detection_base)
        self.quality_scale = float(quality_scale)
        self.miss_value = float(miss_value)
        self.default_k = int(default_k)
        self.adaptive_k = bool(adaptive_k)

        self.counts = np.zeros(self.num_bands, dtype=np.float32)
        self.values = np.zeros(self.num_bands, dtype=np.float32)
        self.last_band: int | None = None
        self.total_steps = 0

    def reset(self):
        self.counts.fill(0.0)
        self.values.fill(0.0)
        self.last_band = None
        self.total_steps = 0

    def score_bands(
        self,
        band_belief: np.ndarray,
        scan_age: np.ndarray,
        band_uncertainty: np.ndarray,
        investigation_priority: np.ndarray,
        prediction_reliability: float = 1.0,
    ) -> np.ndarray:
        """Scores all 20 bands via multi-factor UCB weighting."""
        never_observed = np.flatnonzero(self.counts < 0.2)
        if never_observed.size:
            scores = np.zeros(self.num_bands, dtype=np.float32)
            scores[never_observed] = 100.0 - np.arange(len(never_observed))
            return scores

        total_counts = float(self.counts.sum())
        exploration_bonus = self.exploration_scale * np.sqrt(
            np.log(total_counts + 2.0) / np.maximum(self.counts, 1e-6)
        )

        effective_pred_weight = self.w_prediction * float(np.clip(prediction_reliability, 0.0, 1.0))

        # Physical distance-dependent retuning penalty
        switch_costs = np.zeros(self.num_bands, dtype=np.float32)
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_costs[b] = self.switch_penalty * (abs(b - self.last_band) / max(1, self.num_bands - 1))

        scores = (
            self.values
            + exploration_bonus
            + effective_pred_weight * band_belief
            + self.w_age * scan_age
            + self.w_uncertainty * band_uncertainty
            + self.w_investigate * investigation_priority
            - switch_costs
        )
        return scores.astype(np.float32)

    def determine_k(self, scores: np.ndarray, mean_uncertainty: float, max_inv: float) -> int:
        """Determines search width K adaptively based on state ambiguity."""
        if not self.adaptive_k:
            return min(self.default_k, self.num_bands)

        sorted_s = np.sort(scores)[::-1]
        margin = float(sorted_s[0] - sorted_s[1]) if len(sorted_s) > 1 else 1.0

        # Obvious state: clear leader and low uncertainty
        if margin > 0.35 and mean_uncertainty < 0.25 and max_inv < 0.20:
            return 2
        # High uncertainty or anomaly
        if mean_uncertainty > 0.60 or max_inv > 0.50:
            return min(6, self.num_bands)
        return min(4, self.num_bands)

    def get_top_k_candidates(
        self,
        band_belief: np.ndarray,
        scan_age: np.ndarray,
        band_uncertainty: np.ndarray,
        investigation_priority: np.ndarray,
        prediction_reliability: float = 1.0,
    ) -> tuple[list[int], np.ndarray, int]:
        """Returns (top_k_bands, full_scores, selected_k)."""
        scores = self.score_bands(
            band_belief=band_belief,
            scan_age=scan_age,
            band_uncertainty=band_uncertainty,
            investigation_priority=investigation_priority,
            prediction_reliability=prediction_reliability,
        )
        mean_unc = float(np.mean(band_uncertainty))
        max_inv = float(np.max(investigation_priority))
        k = self.determine_k(scores, mean_unc, max_inv)

        ranked_bands = list(np.argsort(-scores)[:k])
        return ranked_bands, scores, k

    def update(self, band: int, detected: bool, quality: float):
        """Discounts past history and updates observed empirical value."""
        band = int(band)
        self.last_band = band
        self.total_steps += 1

        # Non-stationary temporal discounting
        self.counts *= self.discount_factor
        self.counts[band] += 1.0

        # Observable value derivation (zero simulator truth)
        if detected:
            observed_val = self.detection_base + self.quality_scale * float(np.clip(quality, 0.0, 1.0))
        else:
            observed_val = self.miss_value

        self.values[band] = (1.0 - self.value_lr) * self.values[band] + self.value_lr * observed_val
