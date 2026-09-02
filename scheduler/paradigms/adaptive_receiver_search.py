"""Adaptive Receiver Search Scheduler with Pulse Repetition Interval (PRI) Tracking.

Estimates the pulse repetition period and timing of detected radar emitters.
Schedules targeted dwell intercepts at predicted pulse arrivals while executing
staleness-prioritized sweeps during inter-pulse intervals.
"""

from collections import deque
import numpy as np


class AdaptiveReceiverSearchScheduler:
    """Radar Interceptor Search Strategy with PRI Estimation and Targeted Dwells."""

    def __init__(
        self,
        num_bands: int = 20,
        dwell_window: int = 2,
        sweep_bonus: float = 0.07,
        switch_penalty: float = 0.04,
        seed: int = 42,
    ):
        self.num_bands = int(num_bands)
        self.dwell_window = int(dwell_window)
        self.sweep_bonus = float(sweep_bonus)
        self.switch_penalty = float(switch_penalty)
        self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.last_band = 0
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)

        # Pulse arrival history per channel: deque of step indices
        self.pulse_history: dict[int, deque[int]] = {
            i: deque(maxlen=8) for i in range(self.num_bands)
        }
        self.estimated_pri = np.zeros(self.num_bands, dtype=np.float32)  # PRI in steps
        self.confidence = np.zeros(self.num_bands, dtype=np.float32)

    def select_band(self) -> int:
        """Selects band matching an impending predicted pulse, or sweeps stale bands."""
        urgency_scores = np.zeros(self.num_bands, dtype=np.float32)

        for b in range(self.num_bands):
            pri = self.estimated_pri[b]
            conf = self.confidence[b]

            if pri > 1.0 and conf > 0.3 and len(self.pulse_history[b]) > 0:
                last_hit = self.pulse_history[b][-1]
                steps_since = self.current_step - last_hit
                # Time remaining until expected pulse arrival
                time_to_pulse = (pri - (steps_since % pri)) % pri

                if time_to_pulse <= self.dwell_window:
                    # Impending pulse arrival: high priority target lock
                    urgency_scores[b] = 0.85 * conf * (1.0 - (time_to_pulse / (self.dwell_window + 1.0)))
                else:
                    urgency_scores[b] = 0.05
            else:
                # Default exploration priority
                urgency_scores[b] = 0.05

            # Inter-pulse sweep bonus based on scan age
            urgency_scores[b] += self.sweep_bonus * np.sqrt(max(0.0, self.scan_age[b]))

            # Switching penalty
            switch_dist = abs(b - self.last_band) / max(1, self.num_bands - 1)
            urgency_scores[b] -= self.switch_penalty * switch_dist

        return int(np.argmax(urgency_scores))

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Updates PRI estimates and pulse arrival schedules."""
        band = int(band)
        self.last_band = band
        self.current_step += 1

        self.scan_age += 1.0
        self.scan_age[band] = 0.0

        detected = 0
        if obs_dict is not None:
            detected = int(obs_dict.get("detected", 0))

        if detected:
            history = self.pulse_history[band]
            history.append(self.current_step)

            if len(history) >= 3:
                # Calculate consecutive pulse differences
                diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
                mean_pri = float(np.mean(diffs))
                std_pri = float(np.std(diffs))

                self.estimated_pri[band] = max(1.5, mean_pri)
                # Confidence is high if pulse intervals are consistent (low jitter)
                self.confidence[band] = np.clip(1.0 / (1.0 + std_pri), 0.2, 0.95)
            elif len(history) == 2:
                self.estimated_pri[band] = float(history[1] - history[0])
                self.confidence[band] = 0.5
        else:
            # If an expected pulse was missed during targeted dwell, decay confidence
            if self.confidence[band] > 0.3:
                self.confidence[band] *= 0.85
