"""SmartScan V2 — Unified Production Spectrum Scheduler.

Architecture:
Track 2 Perception & World Model
           ↓
Smart Discounted UCB Candidate Generator (Ranks all 20 bands, yields Top-K)
           ↓
Belief-State Expectimax Planner (Stochastic lookahead with branch-and-bound pruning)
           ↓
Best First Action (with automatic UCB top-1 fallback)
"""

from __future__ import annotations

from typing import Any
import numpy as np

from scheduler.smart_discounted_ucb import SmartDiscountedUCB
from scheduler.belief_expectimax import BeliefExpectimaxPlanner, BeliefState
from scheduler.track2_runtime import DEFAULT_MODEL_PATH, Track2Runtime
from scheduler.track2_core import NUM_BANDS


class SmartScanScheduler:
    """SmartScan V2: Emitter-Aware Belief-State Expectimax with Discounted UCB Candidate Generation."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        # Track 2 configuration
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        # UCB configuration
        discount_factor: float = 0.985,
        exploration_scale: float = 0.75,
        w_prediction: float = 0.35,
        w_age: float = 0.10,
        w_uncertainty: float = 0.15,
        w_investigate: float = 0.20,
        switch_penalty: float = 0.05,
        top_k: int = 5,
        adaptive_k: bool = False,
        # Expectimax configuration
        depth: int = 2,
        gamma: float = 0.90,
        branch_k: int = 3,
        path_prob_threshold: float = 0.005,
        max_planning_ms: float = 6.0,
        max_nodes: int = 400,
        enable_pruning: bool = True,
        enable_caching: bool = True,
    ):
        self.num_bands = int(num_bands)
        self.runtime = runtime or Track2Runtime(model_path=model_path)

        # 1. Candidate Generator
        self.ucb = SmartDiscountedUCB(
            num_bands=self.num_bands,
            discount_factor=discount_factor,
            exploration_scale=exploration_scale,
            w_prediction=w_prediction,
            w_age=w_age,
            w_uncertainty=w_uncertainty,
            w_investigate=w_investigate,
            switch_penalty=switch_penalty,
            default_k=top_k,
            adaptive_k=adaptive_k,
        )

        # 2. Lookahead Planner
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
        )

        self.last_band: int | None = None
        self.step_count = 0
        self.override_count = 0
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self):
        self.runtime.reset()
        self.ucb.reset()
        self.planner.reset()
        self.last_band = None
        self.step_count = 0
        self.override_count = 0
        self.last_diagnostics.clear()

    def select_band(self, observation: dict | None = None) -> int:
        """Selects the next band via UCB candidate ranking + Expectimax lookahead."""
        # 1. Check initial sweep coverage for never-visited bands
        never_observed = np.flatnonzero(self.ucb.counts < 0.2)
        if never_observed.size:
            selected_band = int(never_observed[0])
            self.last_band = selected_band
            self.step_count += 1
            self.last_diagnostics = {
                "selected_band": selected_band,
                "selection_mode": "initial_sweep",
                "override": False,
                "ucb_top1": selected_band,
                "expectimax_action": selected_band,
                "top_k_candidates": [selected_band],
            }
            return selected_band

        # 2. Retrieve perceptual world-model state from Track 2
        global_state = self.runtime.get_global_belief()
        band_belief = global_state["band_belief"]
        scan_age = global_state["scan_age"]
        band_uncertainty = global_state["band_uncertainty"]
        investigation_priority = global_state.get("investigation_priority", np.zeros(self.num_bands, dtype=np.float32))
        prediction_reliability = float(global_state.get("prediction_reliability", 0.50))

        # 3. UCB Candidate Generation: cheap ranking of all 20 bands
        top_k_candidates, ucb_scores, k = self.ucb.get_top_k_candidates(
            band_belief=band_belief,
            scan_age=scan_age,
            band_uncertainty=band_uncertainty,
            investigation_priority=investigation_priority,
            prediction_reliability=prediction_reliability,
        )
        ucb_top1 = int(top_k_candidates[0])

        # 4. Construct compact belief state for lookahead tree
        root_state = BeliefState(
            band_belief=band_belief,
            band_uncertainty=band_uncertainty,
            scan_age=scan_age,
            ucb_values=self.ucb.values,
            ucb_counts=self.ucb.counts,
            investigation_priority=investigation_priority,
            last_band=self.last_band,
        )

        # 5. Expectimax Lookahead Search
        try:
            planner_action, exp_val, search_stats = self.planner.plan(
                root_state=root_state,
                top_k_candidates=top_k_candidates,
            )
            # Automatic fallback if budget exceeded or search timed out with no result
            if search_stats.get("timed_out", False) and planner_action not in top_k_candidates:
                selected_band = ucb_top1
                mode = "ucb_fallback_timeout"
            else:
                selected_band = int(planner_action)
                mode = "belief_expectimax"
        except Exception as e:
            # Absolute fallback guarantee: planner never crashes the controller
            selected_band = ucb_top1
            mode = "ucb_fallback_exception"
            search_stats = {"error": str(e)}

        is_override = bool(selected_band != ucb_top1)
        if is_override:
            self.override_count += 1

        self.last_band = selected_band
        self.step_count += 1

        # 6. Detailed diagnostics for SIH and evaluation
        self.last_diagnostics = {
            "selected_band": selected_band,
            "selection_mode": mode,
            "override": is_override,
            "ucb_top1": ucb_top1,
            "expectimax_action": selected_band,
            "top_k_candidates": top_k_candidates,
            "ucb_scores": ucb_scores.tolist(),
            "search_stats": search_stats,
            "band_belief": band_belief.tolist(),
            "band_uncertainty": band_uncertainty.tolist(),
            "investigation_priority": investigation_priority.tolist(),
            "prediction_reliability": prediction_reliability,
            "tracks": list(self.runtime.manager.tracks.keys()),
        }
        return selected_band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Assimilates real observation into Track 2 runtime and UCB statistics."""
        band = int(band)
        detected = bool(obs_dict.get("detected", False)) if obs_dict else False
        quality = float(np.asarray(obs_dict.get("quality", 0.0)).reshape(-1)[0]) if obs_dict else 0.0

        # Update candidate generator stats
        self.ucb.update(band=band, detected=detected, quality=quality)

        # Update Track 2 perceptual world model
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=float(self.step_count))

    def get_diagnostics(self) -> dict[str, Any]:
        """Returns comprehensive diagnostic dictionary."""
        diag = dict(self.last_diagnostics)
        diag["step_count"] = self.step_count
        diag["override_count"] = self.override_count
        diag["override_rate"] = float(self.override_count / max(1, self.step_count))
        return diag
