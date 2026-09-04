"""SmartScan Candidate 3: Missing-Data-Aware Recurrent Model (GRU-D).

GRU-D (Che et al., Nature Scientific Reports 2018) explicitly models
unobserved cognitive radio channels using time decay (gamma_x, gamma_h)
and missingness masks (m_t), preventing the scheduler from confusing
'unobserved' channels with 'silent' channels.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional
import numpy as np
import torch

from scheduler.baselines import BaseScheduler
from scheduler.grud_spectrum_model import GRUDSpectrumCell, NumpyGRUDModel

NUM_BANDS = 20
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "archive", "models", "grud_spectrum_weights.pt"
)


class GRUDSpectrumScheduler(BaseScheduler):
    """Candidate 3 Scheduler: GRU-D predictive belief with missing-data awareness."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        hidden_dim: int = 32,
        switch_penalty: float = 0.08,
        dwell_inertia: float = 1.25,
        uncertainty_bonus: float = 0.15,
        weights_path: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(num_bands)
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim
        self.switch_penalty = float(switch_penalty)
        self.dwell_inertia = float(dwell_inertia)
        self.uncertainty_bonus = float(uncertainty_bonus)
        self.rng = np.random.default_rng(seed)

        # Instantiate PyTorch cell
        self.torch_cell = GRUDSpectrumCell(num_bands=num_bands, hidden_dim=hidden_dim)
        self.weights_path = weights_path or DEFAULT_WEIGHTS_PATH

        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
                self.torch_cell.load_state_dict(state_dict)
            except Exception:
                pass

        # Use pure NumPy mirror for <0.05ms inference
        self.model = NumpyGRUDModel(self.torch_cell)

        # Internal state
        self.h = np.zeros(hidden_dim, dtype=np.float64)
        self.last_x = np.zeros(num_bands, dtype=np.float64)
        self.delta = np.ones(num_bands, dtype=np.float64)  # steps elapsed since last observation
        self.pred_probs = np.full(num_bands, 0.10, dtype=np.float64)

        self.last_band = 0
        self.last_detected = False
        self.consecutive_dwell = 0
        self.step_count = 0

    def reset(self):
        self.h = np.zeros(self.hidden_dim, dtype=np.float64)
        self.last_x = np.zeros(self.num_bands, dtype=np.float64)
        self.delta = np.ones(self.num_bands, dtype=np.float64)
        self.pred_probs = np.full(self.num_bands, 0.10, dtype=np.float64)
        self.last_band = 0
        self.last_detected = False
        self.consecutive_dwell = 0
        self.step_count = 0

    def select_band(self) -> int:
        """Select next band using GRU-D predictive belief + switching/dwell arbitration."""
        scores = np.copy(self.pred_probs)

        # 1. Uncertainty scout bonus: channels unobserved for many steps have high delta
        uncert = np.clip(self.delta / 25.0, 0.0, 1.0)
        scores += self.uncertainty_bonus * uncert

        # 2. Dwell bonus if currently tracking an active signal
        if self.last_detected:
            # Dampen dwell inertia over time to avoid tracking faded/departed emitters forever
            dwell_factor = self.dwell_inertia / (1.0 + 0.15 * self.consecutive_dwell)
            scores[self.last_band] += dwell_factor

        # 3. Switching cost deduction for all non-current bands
        if self.step_count > 0:
            for b in range(self.num_bands):
                if b != self.last_band:
                    scores[b] -= self.switch_penalty

        chosen_band = int(np.argmax(scores))

        # Tie breaking with small random jitter
        max_v = scores[chosen_band]
        candidates = np.where(np.abs(scores - max_v) < 1e-6)[0]
        if len(candidates) > 1:
            chosen_band = int(self.rng.choice(candidates))

        if chosen_band == self.last_band:
            self.consecutive_dwell += 1
        else:
            self.consecutive_dwell = 0

        self.last_band = chosen_band
        return chosen_band

    def update(self, band: int, reward: float, observation: Dict[str, Any]):
        """Update GRU-D temporal belief with partial observation."""
        self.step_count += 1

        # Extract observation signal
        detected = bool(observation.get("detected", False))
        quality = 0.0
        if "quality" in observation:
            q = observation["quality"]
            quality = float(q[0] if isinstance(q, (np.ndarray, list)) else q)

        self.last_detected = detected

        # Construct single-band partial observation vector
        # x_t is 0 everywhere except the scanned band
        x_t = np.zeros(self.num_bands, dtype=np.float64)
        x_t[band] = quality if detected else 0.0

        # Missingness mask: 1 only for scanned band, 0 for all 19 unobserved bands
        m_t = np.zeros(self.num_bands, dtype=np.float64)
        m_t[band] = 1.0

        # Run GRU-D forward step
        pred_probs, self.h, self.last_x = self.model.forward_step(
            x_t=x_t,
            m_t=m_t,
            delta_t=self.delta,
            last_x=self.last_x,
            h_prev=self.h,
        )
        self.pred_probs = pred_probs

        # Update observation gaps: increment all, reset scanned band
        self.delta += 1.0
        self.delta[band] = 1.0
