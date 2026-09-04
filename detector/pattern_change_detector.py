"""Pattern-Change Alarm Detector using World Model Prediction Error.

Phase 1 Implementation:
- Freezes the existing Track 2 CNN-GRU world model (no parameter updates).
- Computes instantaneous Brier prediction error on the scanned channel:
  e_t = (y_t - P_model(band_t))^2
- Executes a Page-Hinkley cumulative sum change-point test on the error stream.

Page-Hinkley is chosen because it monitors cumulative deviations from the empirical
noise floor rather than relying on single-step thresholding, suppressing false alarms
from transient RF fading and sensor dropouts while maintaining bounded detection delay.
"""

from __future__ import annotations

import numpy as np
import torch

from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import DEFAULT_MODEL_PATH, Track2Runtime


class PatternChangeDetector:
    """Monitors prediction errors of a frozen world model to trigger pattern-change alarms."""

    def __init__(
        self,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        delta: float = 0.05,
        threshold: float = 4.0,
        warmup_steps: int = 20,
    ):
        # Freeze existing world model runtime
        self.runtime = runtime or Track2Runtime(model_path=model_path)
        for model in (self.runtime.identity_model, self.runtime.state_model):
            model.eval()
            model.requires_grad_(False)

        self.delta = float(delta)
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)

        # Page-Hinkley state
        self.timestep = 0
        self.cum_dev = 0.0
        self.min_cum_dev = 0.0
        self.mean_error = 0.0
        self.error_count = 0
        self.alarm_history: list[int] = []
        self.error_history: list[float] = []

    def reset(self):
        """Resets detector state for a new episode."""
        self.timestep = 0
        self.cum_dev = 0.0
        self.min_cum_dev = 0.0
        self.mean_error = 0.0
        self.error_count = 0
        self.alarm_history.clear()
        self.error_history.clear()
        self.runtime.reset()

    def get_forecast(self, band: int) -> float:
        """Returns the world model's prior belief of emitter presence on the chosen band."""
        belief = self.runtime.get_band_belief()
        return float(np.clip(belief[band], 0.0, 1.0))

    def step_and_detect(self, band: int, obs_dict: dict) -> dict:
        """Processes the observation, calculates prediction error, and evaluates alarm trigger.

        Returns:
            dict containing:
                - alarm (bool): True if Page-Hinkley statistic exceeds threshold.
                - prediction_error (float): instantaneous Brier error.
                - ph_statistic (float): current Page-Hinkley test statistic.
                - mean_error (float): running mean baseline error.
        """
        # 1. Prediction error before model assimilates current observation
        prior_p = self.get_forecast(band)
        detected = 1.0 if bool(obs_dict.get("detected", False)) else 0.0
        brier_error = float((detected - prior_p) ** 2)

        # 2. Update the frozen runtime to advance its tracking state
        res = self.runtime.update(obs_dict, timestamp=float(self.timestep))
        
        # Include waveform novelty if signal is detected
        novelty = 0.0
        if detected > 0.5 and isinstance(res, dict):
            novelty = float(np.clip(res.get("identity_novelty", 0.0), 0.0, 1.0))
        
        total_error = float(brier_error + 0.30 * novelty * detected)
        self.error_history.append(total_error)
        self.timestep += 1

        # 3. Update running baseline during initial burn-in
        self.error_count += 1
        if self.error_count == 1:
            self.mean_error = total_error
        else:
            # Exponentially smoothed baseline error
            alpha = 0.05
            self.mean_error = (1.0 - alpha) * self.mean_error + alpha * total_error

        # 4. Page-Hinkley change-point accumulation
        alarm = False
        ph_stat = 0.0

        if self.timestep > self.warmup_steps:
            self.cum_dev += (total_error - self.mean_error - self.delta)
            if self.cum_dev < self.min_cum_dev:
                self.min_cum_dev = self.cum_dev

            ph_stat = float(self.cum_dev - self.min_cum_dev)

            if ph_stat >= self.threshold:
                alarm = True
                self.alarm_history.append(self.timestep)
                # Reset cumulative statistic after alarm to allow future detection
                self.cum_dev = 0.0
                self.min_cum_dev = 0.0

        return {
            "timestep": self.timestep,
            "alarm": alarm,
            "prediction_error": total_error,
            "ph_statistic": ph_stat,
            "mean_error": self.mean_error,
        }
