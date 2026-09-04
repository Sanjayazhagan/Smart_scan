"""Evaluation-only perfect-forecast upper bound for Smart Scan.

This scheduler must never be used as a product controller.  The benchmark
injects the hidden activity for the scan that is about to occur, allowing us
to measure whether a perfect next-band predictor would improve the existing
World-Model UCB decision rule.
"""

from __future__ import annotations

import numpy as np

from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime
from scheduler.world_model_ucb import WorldModelUCBScheduler, _vector


class OracleGuidedUCBScheduler(WorldModelUCBScheduler):
    """World-Model UCB with its neural belief replaced by perfect hidden truth."""

    evaluation_only = True

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        oracle_scale: float = 1.0,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        **kwargs,
    ):
        if oracle_scale < 0.0:
            raise ValueError("oracle_scale must be non-negative")
        super().__init__(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            max_scan_age=max_scan_age,
            world_model_scale=0.0,
            **kwargs,
        )
        self.oracle_scale = float(oracle_scale)
        self._oracle_activity: np.ndarray | None = None

    def set_oracle_activity(self, activity) -> None:
        """Accept truth for the imminent scan from the evaluation harness only."""
        vector = np.asarray(activity, dtype=np.float64).reshape(-1)
        if vector.size != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} oracle bands, received {vector.size}"
            )
        self._oracle_activity = np.clip(vector, 0.0, 1.0)

    def select_band(self) -> int:
        if self._oracle_activity is None:
            raise RuntimeError(
                "Oracle activity was not injected. Oracle-UCB is evaluation-only."
            )
        state = self.runtime.get_global_belief()
        scan_age = _vector(state, "scan_age", self.num_bands)

        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            mode = "initial_coverage"
            exploration = 0.0
            contribution = 0.0
            score = 0.0
        else:
            bonuses = self.exploration_scale * np.sqrt(
                np.log(self.total_observations + 2.0)
                / np.maximum(self.counts, 1e-6)
            )
            oracle_guidance = self.oracle_scale * self._oracle_activity
            scores = self.values + bonuses + 0.10 * scan_age + oracle_guidance
            band = int(np.argmax(scores))
            mode = "oracle_guided_ucb"
            exploration = float(bonuses[band])
            contribution = float(oracle_guidance[band])
            score = float(scores[band])

        trace = {
            "mode": mode,
            "selected_band": band,
            "expert": "oracle_ucb",
            "regime": "EVALUATION_ORACLE",
            "evaluation_only": True,
            "oracle_scale": self.oracle_scale,
            "oracle_active_band_count": int(np.count_nonzero(self._oracle_activity)),
            "oracle_selected_band_active": bool(self._oracle_activity[band] > 0.0),
            "oracle_contribution": contribution,
            "exploration_bonus": exploration,
            "score": score,
        }
        self.last_trace = trace
        self.last_decision_trace = trace
        self._oracle_activity = None
        return band
