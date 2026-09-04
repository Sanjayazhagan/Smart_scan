"""NMF + UCB Hybrid Belief-State Expectimax Scheduler (SmartScan V2-Hybrid).

Marries NMF spectral stability with UCB curiosity:
1. NMF Core: Discovers latent spectral co-occurrence basis for stability and zero fidgeting.
2. UCB Curiosity Engine: Tracks non-stationary visitations to identify agile hoppers.
3. Reserved Scout Candidate Generation:
   - Slots 1, 2, 3: Top NMF spectral basis candidates (cluster stability).
   - Slot 4: Top UCB curiosity scout (agile frequency hoppers).
4. Belief-State Expectimax Planner: Reasons through HIT/MISS futures across both NMF and UCB candidates.
5. Fallback Guarantee: Instantaneous fallback to NMF/UCB top-1 if planning budget is exceeded.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from scheduler.belief_expectimax import BeliefExpectimaxPlanner, BeliefState
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.rl_value_network import FastBeliefValueCritic, DEFAULT_VALUE_NET_PATH
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MODEL_PATH, Track2Runtime


class NMFExpectimaxScheduler:
    """NMF + UCB Hybrid Belief-State Expectimax Cognitive Spectrum Scheduler."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        # Track 2 configuration
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        # NMF configuration
        nmf_components: int = 4,
        nmf_window: int = 30,
        nmf_recompute_every: int = 2,
        nmf_weight: float = 1.00,
        # UCB Curiosity configuration
        curiosity_scale: float = 0.50,
        discount_factor: float = 0.985,
        value_lr: float = 0.20,
        # General weights
        w_belief: float = 0.35,
        w_investigate: float = 0.20,
        switch_penalty: float = 0.08,
        top_k: int = 4,
        # Expectimax configuration
        depth: int = 2,
        gamma: float = 0.90,
        branch_k: int = 3,
        path_prob_threshold: float = 0.005,
        max_planning_ms: float = 6.0,
        max_nodes: int = 400,
        enable_pruning: bool = True,
        enable_caching: bool = True,
        use_rl_critic: bool = True,
        critic_path=DEFAULT_VALUE_NET_PATH,
        seed: int | None = None,
    ):
        self.num_bands = int(num_bands)
        self.runtime = runtime or Track2Runtime(model_path=model_path)
        self.value_critic = FastBeliefValueCritic(critic_path) if use_rl_critic else None

        # 1. NMF Spectral Matrix Factorization Core
        self.nmf = NMFScheduler(
            num_bands=self.num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
            seed=seed,
        )

        # 2. UCB Non-Stationary Statistics Engine
        self.curiosity_scale = float(curiosity_scale)
        self.discount_factor = float(np.clip(discount_factor, 0.80, 1.0))
        self.value_lr = float(np.clip(value_lr, 0.01, 1.0))
        self.counts = np.zeros(self.num_bands, dtype=np.float32)
        self.values = np.zeros(self.num_bands, dtype=np.float32)

        self.nmf_weight = float(nmf_weight)
        self.w_belief = float(w_belief)
        self.w_investigate = float(w_investigate)
        self.switch_penalty = float(switch_penalty)
        self.top_k = int(top_k)

        # 3. Stochastic Belief-State Expectimax Planner
        self.planner = BeliefExpectimaxPlanner(
            num_bands=self.num_bands,
            depth=depth,
            gamma=gamma,
            branch_k=branch_k,
            path_prob_threshold=path_prob_threshold,
            w_switch=switch_penalty,
            max_planning_ms=max_planning_ms,
            max_nodes=max_nodes,
            enable_pruning=enable_pruning,
            enable_caching=enable_caching,
            value_critic=self.value_critic,
        )

        self.last_band: int | None = None
        self.step_count = 0
        self.override_count = 0
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self):
        self.runtime.reset()
        self.planner.reset()
        self.counts.fill(0.0)
        self.values.fill(0.0)
        self.last_band = None
        self.step_count = 0
        self.override_count = 0
        self.last_diagnostics.clear()

    def select_band(self, observation: dict | None = None) -> int:
        """Selects band via NMF+UCB Hybrid Candidate Extraction + Expectimax Lookahead."""
        # Initial exploration sweep for first 20 steps
        if self.step_count < self.num_bands and self.last_band is None:
            selected_band = self.step_count
            self.last_band = selected_band
            self.step_count += 1
            self.last_diagnostics = {
                "selected_band": selected_band,
                "selection_mode": "initial_sweep",
                "override": False,
                "nmf_top1": selected_band,
                "top_k_candidates": [selected_band],
            }
            return selected_band

        # 1. Retrieve perceptual world-model state from Track 2
        global_state = self.runtime.get_global_belief()
        band_belief = global_state["band_belief"]
        scan_age = global_state["scan_age"]
        band_uncertainty = global_state["band_uncertainty"]
        investigation_priority = global_state.get(
            "investigation_priority", np.zeros(self.num_bands, dtype=np.float32)
        )
        prediction_reliability = float(global_state.get("prediction_reliability", 0.50))

        # 2. NMF Candidate Generation (Spectral Basis)
        nmf_spectrum = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
        nmf_spectrum = np.clip(nmf_spectrum, 0.0, None)
        tot_nmf = float(nmf_spectrum.sum())
        if tot_nmf > 1e-12:
            nmf_spectrum = nmf_spectrum / tot_nmf

        # Distance-dependent frequency retuning penalty
        switch_costs = np.zeros(self.num_bands, dtype=np.float32)
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_costs[b] = self.switch_penalty * (abs(b - self.last_band) / max(1, self.num_bands - 1))

        effective_belief_w = self.w_belief * prediction_reliability
        nmf_scores = (
            self.nmf_weight * nmf_spectrum
            + effective_belief_w * band_belief
            + self.w_investigate * investigation_priority
            - switch_costs
        )

        # 3. UCB Curiosity Engine (The Agile Hopper Scout)
        total_counts = float(self.counts.sum())
        curiosity_bonus = self.curiosity_scale * np.sqrt(
            np.log(total_counts + 2.0) / np.maximum(self.counts, 1e-6)
        )
        ucb_curiosity_scores = curiosity_bonus + 0.20 * band_uncertainty - switch_costs

        # Extract top (K - 1) NMF spectral candidates
        nmf_ranked = list(np.argsort(-nmf_scores))
        selected_candidates = nmf_ranked[: max(1, self.top_k - 1)]

        # Extract the #1 UCB curiosity scout not already in NMF list
        for b in np.argsort(-ucb_curiosity_scores):
            if b not in selected_candidates:
                selected_candidates.append(int(b))
                break

        nmf_top1 = int(nmf_ranked[0])

        # 4. Construct compact BeliefState for Expectimax lookahead
        root_state = BeliefState(
            band_belief=band_belief,
            band_uncertainty=band_uncertainty,
            scan_age=scan_age,
            ucb_values=self.values.copy(),
            ucb_counts=self.counts.copy(),
            investigation_priority=investigation_priority,
            last_band=self.last_band,
        )

        # 5. Expectimax Lookahead Search over Hybrid Candidates
        try:
            planner_action, exp_val, search_stats = self.planner.plan(
                root_state=root_state,
                top_k_candidates=selected_candidates,
            )
            if search_stats.get("timed_out", False) and planner_action not in selected_candidates:
                selected_band = nmf_top1
                mode = "hybrid_fallback_timeout"
            else:
                selected_band = int(planner_action)
                mode = "hybrid_expectimax"
        except Exception as e:
            selected_band = nmf_top1
            mode = "hybrid_fallback_exception"
            search_stats = {"error": str(e)}

        is_override = bool(selected_band != nmf_top1)
        if is_override:
            self.override_count += 1

        self.last_band = selected_band
        self.step_count += 1

        self.last_diagnostics = {
            "selected_band": selected_band,
            "selection_mode": mode,
            "override": is_override,
            "nmf_top1": nmf_top1,
            "expectimax_action": selected_band,
            "top_k_candidates": selected_candidates,
            "search_stats": search_stats,
            "band_belief": band_belief.tolist(),
            "band_uncertainty": band_uncertainty.tolist(),
            "investigation_priority": investigation_priority.tolist(),
            "prediction_reliability": prediction_reliability,
        }
        return selected_band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Assimilates observation into NMF buffer, UCB statistics, and Track 2 runtime."""
        band = int(band)
        detected = bool(obs_dict.get("detected", False)) if obs_dict else False
        quality = float(np.asarray(obs_dict.get("quality", 0.0)).reshape(-1)[0]) if obs_dict else 0.0

        # Update NMF matrix factorization buffer
        self.nmf.update(band, reward, obs_dict)

        # Update non-stationary UCB statistics
        self.counts *= self.discount_factor
        self.counts[band] += 1.0
        observed_val = (0.10 + 0.90 * quality) if detected else 0.02
        self.values[band] = (1.0 - self.value_lr) * self.values[band] + self.value_lr * observed_val

        # Update Track 2 perceptual world model
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=float(self.step_count))

    def get_diagnostics(self) -> dict[str, Any]:
        diag = dict(self.last_diagnostics)
        diag["step_count"] = self.step_count
        diag["override_count"] = self.override_count
        diag["override_rate"] = float(self.override_count / max(1, self.step_count))
        return diag
