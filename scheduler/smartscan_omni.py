"""SmartScan-Omni V2: Unified Master Cognitive Spectrum Controller.

Integrates:
1. Dwell Anchor with 2-Miss Fading Debounce:
   - Holds through momentary 1-step multipath/Rayleigh fading dips.
   - Releases immediately on 2 consecutive misses.
   - Collapses antenna switches in stationary to near-zero.
2. Robust PCA Sparse Pulse De-Noising Front-End:
   - Inexact ALM RPCA decomposes sliding spectrum M = L (Noise/Jamming) + S (Sparse Pulses).
   - Filters continuous wave jamming and thermal static, feeding clean sparse energy to candidate generation.
3. NMF Spectral Co-Occurrence Basis:
   - Discovers latent multi-band radar clusters in crowded battle groups.
4. Agile UCB Curiosity Scout:
   - Automatically reserved candidate slot on confirmed misses to intercept fast frequency hoppers.
5. RL-Guided Expectimax Lookahead:
   - 2-step stochastic lookahead with the trained RL Value Critic (V_theta) at tree leaf nodes.
"""

from __future__ import annotations

from collections import deque
from typing import Any
import numpy as np

from scheduler.belief_expectimax import BeliefExpectimaxPlanner, BeliefState
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.paradigms.robust_pca_psr import inexact_alm_rpca
from scheduler.rl_value_network import FastBeliefValueCritic, DEFAULT_VALUE_NET_PATH
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MODEL_PATH, Track2Runtime


class SmartScanOmniScheduler:
    """SmartScan-Omni V2: Complete Unified Cognitive Spectrum Controller."""

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
        nmf_weight: float = 0.85,
        # Robust PCA configuration (Noise/Jamming filter)
        use_rpca_filter: bool = True,
        rpca_window: int = 30,
        rpca_weight: float = 0.25,
        rpca_recompute_every: int = 2,
        # Dwell & Debounce parameters (Stationary booster)
        dwell_inertia: float = 1.05,
        hysteresis_margin: float = 0.05,
        # UCB Curiosity parameters (Agile Hopper Scout)
        curiosity_scale: float = 0.35,
        discount_factor: float = 0.985,
        value_lr: float = 0.20,
        # General candidate weights
        w_belief: float = 0.35,
        w_investigate: float = 0.25,
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
        # RL Critic configuration (AlphaZero leaf evaluator)
        use_rl_critic: bool = True,
        critic_path=DEFAULT_VALUE_NET_PATH,
        seed: int | None = None,
    ):
        self.num_bands = int(num_bands)
        self.runtime = runtime or Track2Runtime(model_path=model_path)

        # 1. NMF Spectral Matrix Factorization Core
        self.nmf = NMFScheduler(
            num_bands=self.num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
            seed=seed,
        )

        # 2. Robust PCA De-noising Front-End
        self.use_rpca_filter = bool(use_rpca_filter)
        self.rpca_weight = float(rpca_weight)
        self.rpca_recompute_every = int(rpca_recompute_every)
        self.rpca_history = deque(maxlen=int(rpca_window))
        self.sparse_profile = np.zeros(self.num_bands, dtype=np.float32)

        for _ in range(int(rpca_window)):
            self.rpca_history.append(np.full(self.num_bands, 0.05, dtype=np.float32))

        # 3. Non-Stationary UCB Statistics Engine
        self.curiosity_scale = float(curiosity_scale)
        self.discount_factor = float(np.clip(discount_factor, 0.80, 1.0))
        self.value_lr = float(np.clip(value_lr, 0.01, 1.0))
        self.counts = np.zeros(self.num_bands, dtype=np.float32)
        self.values = np.zeros(self.num_bands, dtype=np.float32)

        # 4. Dwell & Debounce Controls
        self.dwell_inertia = float(dwell_inertia)
        self.hysteresis_margin = float(hysteresis_margin)
        self.nmf_weight = float(nmf_weight)
        self.w_belief = float(w_belief)
        self.w_investigate = float(w_investigate)
        self.switch_penalty = float(switch_penalty)
        self.top_k = int(top_k)

        # 5. RL Value Critic (Leaf Evaluator)
        self.value_critic = FastBeliefValueCritic(critic_path) if use_rl_critic else None

        # 6. Stochastic Expectimax Planner
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

        # State tracking & Debounce Filter
        self.last_band: int | None = None
        self.last_detected = False
        self.last_quality = 0.0
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.was_dwell_locked = False

        self.step_count = 0
        self.override_count = 0
        self.dwell_count = 0
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self):
        self.runtime.reset()
        self.planner.reset()
        self.counts.fill(0.0)
        self.values.fill(0.0)
        self.last_band = None
        self.last_detected = False
        self.last_quality = 0.0
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.was_dwell_locked = False
        self.step_count = 0
        self.override_count = 0
        self.dwell_count = 0
        self.sparse_profile.fill(0.0)
        self.rpca_history.clear()
        for _ in range(self.rpca_history.maxlen or 30):
            self.rpca_history.append(np.full(self.num_bands, 0.05, dtype=np.float32))
        self.last_diagnostics.clear()

    def select_band(self, observation: dict | None = None) -> int:
        """Executes the complete unified cognitive decision cycle."""
        # Initial sweep for first 20 steps
        if self.step_count < self.num_bands and self.last_band is None:
            selected_band = self.step_count
            self.last_band = selected_band
            self.step_count += 1
            self.last_diagnostics = {
                "selected_band": selected_band,
                "selection_mode": "initial_sweep",
                "override": False,
                "top_k_candidates": [selected_band],
            }
            return selected_band

        # -------------------------------------------------------------
        # STEP 1: Perceptual World-Model, NMF, and RPCA State
        # -------------------------------------------------------------
        global_state = self.runtime.get_global_belief()
        band_belief = global_state["band_belief"]
        scan_age = global_state["scan_age"]
        band_uncertainty = global_state["band_uncertainty"]
        investigation_priority = global_state.get(
            "investigation_priority", np.zeros(self.num_bands, dtype=np.float32)
        )
        prediction_reliability = float(global_state.get("prediction_reliability", 0.50))

        # 1. NMF Reconstructed Spectrum Energy
        nmf_spectrum = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
        nmf_spectrum = np.clip(nmf_spectrum, 0.0, None)
        tot_nmf = float(nmf_spectrum.sum())
        if tot_nmf > 1e-12:
            nmf_spectrum = nmf_spectrum / tot_nmf

        # 2. RPCA Denoised Sparse Burst Profile
        sparse_vec = np.zeros(self.num_bands, dtype=np.float64)
        if self.use_rpca_filter:
            sparse_vec = np.asarray(self.sparse_profile, dtype=np.float64)

        # 3. Distance-dependent frequency retuning penalty
        switch_costs = np.zeros(self.num_bands, dtype=np.float32)
        if self.last_band is not None and self.switch_penalty > 0.0:
            for b in range(self.num_bands):
                switch_costs[b] = self.switch_penalty * (abs(b - self.last_band) / max(1, self.num_bands - 1))

        # -------------------------------------------------------------
        # STEP 2: Dwell Anchor with 2-Miss Debounce Filter
        # -------------------------------------------------------------
        is_dwell_locked = False
        if self.last_band is not None:
            # If hit: firmly locked
            if self.last_detected:
                is_dwell_locked = True
            # If single miss, but previously firmly locked: DEBOUNCE HOLD (ignore fading dip!)
            elif self.was_dwell_locked and self.consecutive_misses <= 1:
                is_dwell_locked = True
            else:
                is_dwell_locked = False

        self.was_dwell_locked = is_dwell_locked

        # -------------------------------------------------------------
        # STEP 3: Multi-Factor Candidate Scoring
        # -------------------------------------------------------------
        effective_belief_w = self.w_belief * prediction_reliability
        candidate_scores = (
            self.nmf_weight * nmf_spectrum
            + self.rpca_weight * sparse_vec
            + effective_belief_w * band_belief
            + self.w_investigate * investigation_priority
            - switch_costs
        )

        if is_dwell_locked and self.last_band is not None:
            dwell_bonus = self.dwell_inertia * max(0.5, self.last_quality)
            candidate_scores[self.last_band] += dwell_bonus

        primary_ranked = list(np.argsort(-candidate_scores))
        selected_candidates = primary_ranked[: max(1, self.top_k - 1)]

        # Agile Curiosity Scout: only active when lock is genuinely broken
        if not is_dwell_locked:
            total_counts = float(self.counts.sum())
            curiosity_bonus = self.curiosity_scale * np.sqrt(
                np.log(total_counts + 2.0) / np.maximum(self.counts, 1e-6)
            )
            scout_scores = curiosity_bonus + 0.25 * band_uncertainty - switch_costs
            for b in np.argsort(-scout_scores):
                if b not in selected_candidates:
                    selected_candidates.append(int(b))
                    break
        else:
            # If locked, reserve remaining slot for 2nd best spectral band
            if len(primary_ranked) >= self.top_k:
                selected_candidates.append(int(primary_ranked[self.top_k - 1]))

        primary_top1 = int(primary_ranked[0])

        # -------------------------------------------------------------
        # STEP 4: Expectimax Lookahead with RL Value Critic
        # -------------------------------------------------------------
        root_state = BeliefState(
            band_belief=band_belief,
            band_uncertainty=band_uncertainty,
            scan_age=scan_age,
            ucb_values=candidate_scores.astype(np.float32),
            ucb_counts=self.counts.copy(),
            investigation_priority=investigation_priority,
            last_band=self.last_band,
        )

        try:
            planner_action, exp_val, search_stats = self.planner.plan(
                root_state=root_state,
                top_k_candidates=selected_candidates,
            )
            if search_stats.get("timed_out", False) and planner_action not in selected_candidates:
                candidate_action = primary_top1
                mode = "omni_fallback_timeout"
            else:
                candidate_action = int(planner_action)
                mode = "omni_expectimax_rl"
        except Exception as e:
            candidate_action = primary_top1
            mode = "omni_fallback_exception"
            search_stats = {"error": str(e)}

        # -------------------------------------------------------------
        # STEP 5: Switching Hysteresis Gate
        # -------------------------------------------------------------
        if self.last_band is not None and candidate_action != self.last_band:
            curr_score = candidate_scores[self.last_band]
            cand_score = candidate_scores[candidate_action]
            if cand_score - curr_score < self.hysteresis_margin and is_dwell_locked:
                final_action = self.last_band
                mode = "omni_hysteresis_hold"
            else:
                final_action = candidate_action
        else:
            final_action = candidate_action

        is_override = bool(final_action != primary_top1)
        if is_override:
            self.override_count += 1
        if final_action == self.last_band and (self.last_detected or is_dwell_locked):
            self.dwell_count += 1

        self.last_band = final_action
        self.step_count += 1

        self.last_diagnostics = {
            "selected_band": final_action,
            "selection_mode": mode,
            "override": is_override,
            "is_dwell_locked": is_dwell_locked,
            "consecutive_misses": self.consecutive_misses,
            "primary_top1": primary_top1,
            "expectimax_action": candidate_action,
            "top_k_candidates": selected_candidates,
            "search_stats": search_stats,
            "band_belief": band_belief.tolist(),
            "band_uncertainty": band_uncertainty.tolist(),
            "investigation_priority": investigation_priority.tolist(),
            "prediction_reliability": prediction_reliability,
        }
        return final_action

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Assimilates observations into NMF, RPCA, UCB, and Track 2 runtime."""
        band = int(band)
        detected = bool(obs_dict.get("detected", False)) if obs_dict else False
        quality = float(np.asarray(obs_dict.get("quality", 0.0)).reshape(-1)[0]) if obs_dict else 0.0
        power = float(np.asarray(obs_dict.get("signal_power", 0.05)).reshape(-1)[0]) if obs_dict else 0.05

        self.last_detected = detected
        self.last_quality = quality

        if detected:
            self.consecutive_hits += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            self.consecutive_hits = 0

        # 1. Update NMF spectral buffer
        self.nmf.update(band, reward, obs_dict)

        # 2. Update RPCA sliding matrix & compute sparse burst decomposition
        row = np.full(self.num_bands, 0.02, dtype=np.float32)
        row[band] = float(np.clip(power, 0.05, 2.0)) if detected else 0.01
        self.rpca_history.append(row)

        if self.use_rpca_filter and (self.step_count % self.rpca_recompute_every == 0):
            matrix = np.array(self.rpca_history, dtype=np.float32)
            try:
                l, s = inexact_alm_rpca(matrix, max_iter=15)
                # Extract clean sparse bursts from recent window
                sparse_mean = np.clip(np.mean(np.maximum(s[-6:], 0.0), axis=0), 0.0, None)
                tot_s = float(sparse_mean.sum())
                if tot_s > 1e-12:
                    self.sparse_profile = (sparse_mean / tot_s).astype(np.float32)
                else:
                    self.sparse_profile.fill(0.0)
            except Exception:
                pass

        # 3. Update non-stationary UCB statistics
        self.counts *= self.discount_factor
        self.counts[band] += 1.0
        obs_val = (0.10 + 0.90 * quality) if detected else 0.02
        self.values[band] = (1.0 - self.value_lr) * self.values[band] + self.value_lr * obs_val

        # 4. Update Track 2 perceptual world model
        if obs_dict is not None:
            self.runtime.update(obs_dict, timestamp=float(self.step_count))

    def get_diagnostics(self) -> dict[str, Any]:
        diag = dict(self.last_diagnostics)
        diag["step_count"] = self.step_count
        diag["override_count"] = self.override_count
        diag["dwell_count"] = self.dwell_count
        diag["dwell_rate"] = float(self.dwell_count / max(1, self.step_count))
        diag["override_rate"] = float(self.override_count / max(1, self.step_count))
        return diag
