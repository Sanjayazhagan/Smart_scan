"""Dwell-Dual + Selective Shortlist Expectimax Lookahead Planner.

Hierarchical Decision Pipeline:
1. Computes base Dwell-Dual multi-factor scores S(b) across all 20 channels in O(B) time.
2. Fast-Path Bypass: If locked on an active pulse with high dwell inertia (margin >= delta),
   immediately holds lock without tree search overhead (~0.03 ms).
3. Shortlist Selection: When ambiguous, shortlists Top-K candidates (K=3 to 4) plus current band.
4. Depth-2 Expectimax Tree Search:
   - Branch on candidate actions a in Shortlist.
   - Chance nodes evaluate HIT vs MISS probabilistic futures using NMF forecast P(HIT | a).
   - Rollback expected multi-step payoff factoring in future dwell bonus and retuning penalties.
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np

from scheduler.smartscan_production import SmartScanProductionScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MODEL_PATH


class DwellDualExpectimaxScheduler(SmartScanProductionScheduler):
    """Hierarchical Dwell-Dual with Selective Shortlist Expectimax lookahead."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        shortlist_k: int = 4,
        lookahead_depth: int = 2,
        discount_gamma: float = 0.85,
        clear_dominance_margin: float = 0.50,
        seed: Optional[int] = None,
        model_path: Optional[str] = DEFAULT_MODEL_PATH,
        **kwargs,
    ):
        super().__init__(num_bands=num_bands, seed=seed, model_path=model_path, **kwargs)
        self.shortlist_k = int(shortlist_k)
        self.lookahead_depth = int(lookahead_depth)
        self.discount_gamma = float(discount_gamma)
        self.clear_dominance_margin = float(clear_dominance_margin)
        self.expectimax_decisions = 0
        self.fast_path_decisions = 0

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

        # Active signal check (dwell inertia / fading grace)
        is_signal_active = self.last_detected or (
            self.fading_grace_steps > 0
            and self.consecutive_misses <= self.fading_grace_steps
            and self.consecutive_dwell_steps >= 2
        )

        # NMF spectral forecast
        nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
        nmf_forecast = np.clip(nmf_forecast, 0.0, None)
        nmf_total = float(nmf_forecast.sum())
        if nmf_total > 1e-12:
            nmf_forecast = nmf_forecast / nmf_total

        exploration_bonus = self.exploration_scale * np.sqrt(
            np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
        )

        # Base Dwell-Dual Exploit Scores
        base_scores = (
            self.values
            + exploration_bonus
            + self.nmf_scale * nmf_forecast
            - switch_costs
        )
        if self.last_band is not None and is_signal_active:
            dwell_bonus = self.dwell_inertia * max(0.4, self.last_quality)
            base_scores[self.last_band] += dwell_bonus

        # Sort candidate bands
        ranked_bands = np.argsort(base_scores)[::-1]
        top_1 = int(ranked_bands[0])
        top_2 = int(ranked_bands[1])
        score_margin = float(base_scores[top_1] - base_scores[top_2])

        # FAST-PATH BYPASS:
        # If locked on active pulse with clear score dominance, hold without tree search
        if is_signal_active and top_1 == self.last_band and score_margin >= self.clear_dominance_margin:
            self.fast_path_decisions += 1
            self.dwell_decisions += 1
            self.consecutive_dwell_steps += 1
            self.last_governing_policy = "dwell_fast_path"
            self.last_band = top_1
            return top_1

        # SHORTLIST EXPAND:
        # Pick Top-K candidates, guaranteeing last_band is included for dwell continuity
        shortlist = list(ranked_bands[:self.shortlist_k])
        if self.last_band is not None and self.last_band not in shortlist:
            shortlist.append(self.last_band)

        # DEPTH-2 EXPECTIMAX LOOKAHEAD SEARCH OVER SHORTLIST:
        self.expectimax_decisions += 1
        best_band = top_1
        best_val = -1e9

        for a in shortlist:
            # Immediate transition cost
            imm_cost = switch_costs[a]
            p_hit = float(np.clip(nmf_forecast[a], 0.05, 0.95))
            if a == self.last_band and self.last_detected:
                p_hit = max(p_hit, 0.80)

            # --- Branch 1: HIT Future ---
            r_hit = 1.0 - imm_cost
            stay_v = 1.0 + (self.dwell_inertia * 0.75)
            best_future_hit = stay_v
            for a_next in shortlist:
                if a_next != a:
                    sw_cost_next = self.switch_penalty * (abs(a_next - a) / max(1, self.num_bands - 1))
                    sw_v = (nmf_forecast[a_next] * 1.0) - sw_cost_next
                    if sw_v > best_future_hit:
                        best_future_hit = sw_v

            # --- Branch 2: MISS Future ---
            r_miss = -0.1 - imm_cost
            stay_v_miss = 0.0
            if a == self.last_band and self.consecutive_dwell_steps >= 1:
                stay_v_miss = 0.40 * self.dwell_inertia
            best_future_miss = stay_v_miss
            for a_next in shortlist:
                if a_next != a:
                    sw_cost_next = self.switch_penalty * (abs(a_next - a) / max(1, self.num_bands - 1))
                    sw_v = (nmf_forecast[a_next] * 1.0) - sw_cost_next
                    if sw_v > best_future_miss:
                        best_future_miss = sw_v

            q_val = (
                p_hit * (r_hit + self.discount_gamma * best_future_hit)
                + (1.0 - p_hit) * (r_miss + self.discount_gamma * best_future_miss)
            )

            # Add exploration boost if band has high uncertainty
            q_val += 0.20 * uncertainty[a]

            if q_val > best_val:
                best_val = q_val
                best_band = a

        if best_band == self.last_band and is_signal_active:
            self.dwell_decisions += 1
            self.consecutive_dwell_steps += 1
            self.last_governing_policy = "dwell_expectimax"
        else:
            self.exploit_decisions += 1
            self.consecutive_dwell_steps = 0
            self.last_governing_policy = "expectimax_switch"

        self.last_band = best_band
        return best_band


def build_dwell_dual_expectimax(
    num_bands: int = NUM_BANDS,
    seed: Optional[int] = None,
    **kwargs,
) -> DwellDualExpectimaxScheduler:
    return DwellDualExpectimaxScheduler(num_bands=num_bands, seed=seed, **kwargs)
