"""Alarm-Reset NMF+UCB Controller (Driver + Watchdog Architecture).

Separates concerns:
1. The Driver (NMF + UCB):
   - Fast, non-neural action selection (0.33 ms).
   - Combines empirical values, UCB exploration bonuses, and NMF spectrum factorization.
   - Zero noisy neural action guidance (avoids retuning penalties).

2. The Watchdog (World Model Prediction-Error Alarm):
   - Monitors Brier prediction error via Page-Hinkley sequential change detection.
   - When an emitter abruptly flips frequency, PRF, or mode, prediction error spikes.
   - When the alarm triggers:
     (a) Flushes stale NMF history matrix so it stops tracking obsolete patterns.
     (b) Resets UCB observation counts to force immediate wideband exploration.
"""

from __future__ import annotations

import numpy as np

from detector.pattern_change_detector import PatternChangeDetector
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MAX_SCAN_AGE, DEFAULT_MODEL_PATH, Track2Runtime
from scheduler.world_model_ucb import WorldModelUCBScheduler, _vector


class AlarmAdaptiveNMFUCBScheduler(WorldModelUCBScheduler):
    """NMF+UCB controller augmented with a World-Model change-detection watchdog."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        nmf_scale: float = 1.00,
        nmf_components: int = 4,
        nmf_window: int = 30,
        nmf_recompute_every: int = 2,
        alarm_threshold: float = 0.80,
        alarm_delta: float = 0.04,
        alarm_warmup: int = 20,
        count_reset_factor: float = 0.20,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        **kwargs,
    ):
        # Neural guidance on actions is strictly 0.0 (The Driver is non-neural)
        super().__init__(
            num_bands,
            world_model_scale=0.0,
            runtime=runtime,
            model_path=model_path,
            max_scan_age=max_scan_age,
            **kwargs,
        )
        self.nmf_scale = float(nmf_scale)
        self.count_reset_factor = float(count_reset_factor)

        # 1. The Driver: NMF spectral factorization
        self.nmf = NMFScheduler(
            num_bands,
            n_components=nmf_components,
            window_size=nmf_window,
            recompute_every=nmf_recompute_every,
        )

        # 2. The Watchdog: Page-Hinkley prediction-error detector
        self.detector = PatternChangeDetector(
            runtime=self.runtime,
            threshold=alarm_threshold,
            delta=alarm_delta,
            warmup_steps=alarm_warmup,
        )
        self.total_alarms = 0
        self.last_alarm_step = -1

    def select_band(self) -> int:
        state = self.runtime.get_global_belief()
        scan_age = _vector(state, "scan_age", self.num_bands)

        nmf_forecast = np.asarray(self.nmf.predicted_spectrum, dtype=np.float64)
        nmf_forecast = np.clip(nmf_forecast, 0.0, None)
        nmf_total = float(nmf_forecast.sum())
        if nmf_total > 1e-12:
            nmf_forecast = nmf_forecast / nmf_total

        never_observed = np.flatnonzero(self.counts < 0.5)
        if never_observed.size:
            band = int(never_observed[0])
            self.last_band = band
            return band

        # UCB Exploration Bonus
        exploration_bonus = self.exploration_scale * np.sqrt(
            np.log(self.total_observations + 2.0) / np.maximum(self.counts, 1e-6)
        )
        nmf_guidance = self.nmf_scale * nmf_forecast

        scores = (
            self.values
            + exploration_bonus
            + 0.10 * scan_age
            + nmf_guidance
        )
        if self.last_band is not None and self.staying_inertia > 0.0:
            scores[self.last_band] += self.staying_inertia

        band = int(np.argmax(scores))
        self.last_band = band
        return band

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        # Update base UCB tracking
        super().update(band, reward, obs_dict)

        if obs_dict is None:
            self.nmf.update(band, 0.0, obs_dict)
            return

        # 1. Watchdog evaluates observation prediction error
        det_result = self.detector.step_and_detect(band, obs_dict)

        # 2. If pattern change detected, execute adaptive reset
        if det_result["alarm"]:
            self.total_alarms += 1
            self.last_alarm_step = int(self.timestamp)
            
            # (a) Flush stale NMF history matrix to prevent locking onto dead channels
            self.nmf.history.clear()
            for _ in range(self.nmf.window_size):
                self.nmf.history.append(np.full(self.num_bands, 0.05, dtype=np.float32))
            self.nmf.predicted_spectrum = np.ones(self.num_bands, dtype=np.float32) / self.num_bands

            # (b) Contract UCB counts to trigger immediate wideband re-exploration
            self.counts = np.maximum(0.2, self.counts * self.count_reset_factor)
            self.total_observations = float(self.counts.sum())

        # Update NMF with clean observation
        self.nmf.update(band, 0.0, obs_dict)
