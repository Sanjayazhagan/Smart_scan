"""GRU-D: Recurrent Neural Network for Spectrum Sensing with Missing Data.

Reference: Che et al., 2018 ("Recurrent Neural Networks for Multivariate
Time Series with Missing Values", Nature Scientific Reports).

In cognitive radio spectrum scanning:
- Only 1 band is scanned at each step t (k_t in {0, ..., K-1}).
- K-1 bands are unobserved (missingness mask m_t[k] = 0).
- Time gap delta_t[k] measures steps elapsed since band k was last observed.
- Input decay gamma_x and hidden decay gamma_h model temporal fading
  and memory degradation over unobserved intervals.
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDSpectrumCell(nn.Module):
    """GRU-D Cell for spectrum prediction under partial observations."""

    def __init__(self, num_bands: int = 20, hidden_dim: int = 32):
        super().__init__()
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim

        # Input decay parameters: gamma_x = exp(-max(0, W_gamma_x * delta + b_gamma_x))
        # Diagonal weights for input features
        self.w_gamma_x = nn.Parameter(torch.full((num_bands,), 0.1, dtype=torch.float32))
        self.b_gamma_x = nn.Parameter(torch.zeros(num_bands, dtype=torch.float32))

        # Hidden state decay parameters: gamma_h = exp(-max(0, W_gamma_h @ delta + b_gamma_h))
        self.w_gamma_h = nn.Linear(num_bands, hidden_dim, bias=True)

        # Baseline value for missing inputs (learnable or fixed prior)
        self.x_mean = nn.Parameter(torch.full((num_bands,), 0.15, dtype=torch.float32))

        # GRU Gates: input is [x_hat, h_hat, mask] -> dim = num_bands + hidden_dim + num_bands
        gate_input_dim = num_bands + hidden_dim + num_bands
        self.w_z = nn.Linear(gate_input_dim, hidden_dim, bias=True)
        self.w_r = nn.Linear(gate_input_dim, hidden_dim, bias=True)
        self.w_h = nn.Linear(gate_input_dim, hidden_dim, bias=True)

        # Output prediction head: predict next-step activity probability for all bands
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bands),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_gamma_h.weight)
        nn.init.zeros_(self.w_gamma_h.bias)
        nn.init.xavier_uniform_(self.w_z.weight)
        nn.init.zeros_(self.w_z.bias)
        nn.init.xavier_uniform_(self.w_r.weight)
        nn.init.zeros_(self.w_r.bias)
        nn.init.xavier_uniform_(self.w_h.weight)
        nn.init.zeros_(self.w_h.bias)

    def forward_step(
        self,
        x_t: torch.Tensor,       # [B, K]: current observation (only active band non-zero)
        m_t: torch.Tensor,       # [B, K]: mask (1 if band was observed, 0 otherwise)
        delta_t: torch.Tensor,   # [B, K]: steps elapsed since last observation
        last_x: torch.Tensor,    # [B, K]: value of last observation per band
        h_prev: torch.Tensor,    # [B, H]: previous hidden state
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward pass.
        
        Returns:
            pred_probs: [B, K] predicted activity probability for each band next step.
            h_t: [B, H] updated hidden state.
            new_last_x: [B, K] updated last observed values.
        """
        # 1. Decay rates: gamma in (0, 1]
        gamma_x = torch.exp(-torch.clamp(self.w_gamma_x * delta_t + self.b_gamma_x, min=0.0))
        gamma_h = torch.exp(-torch.clamp(self.w_gamma_h(delta_t), min=0.0))

        # 2. Input imputation with temporal decay
        # x_hat = m_t * x_t + (1 - m_t) * (gamma_x * last_x + (1 - gamma_x) * x_mean)
        x_decayed = gamma_x * last_x + (1.0 - gamma_x) * self.x_mean
        x_hat = m_t * x_t + (1.0 - m_t) * x_decayed

        # 3. Hidden state decay
        h_hat = gamma_h * h_prev

        # 4. Concatenate gate inputs: [x_hat, h_hat, m_t]
        concat_input = torch.cat([x_hat, h_hat, m_t], dim=-1)

        # 5. GRU gating
        z = torch.sigmoid(self.w_z(concat_input))
        r = torch.sigmoid(self.w_r(concat_input))

        # Candidate hidden state: uses reset gate on decayed hidden state
        candidate_input = torch.cat([x_hat, r * h_hat, m_t], dim=-1)
        h_tilde = torch.tanh(self.w_h(candidate_input))

        h_t = (1.0 - z) * h_hat + z * h_tilde

        # 6. Predict next-step activity probabilities
        logits = self.fc_out(h_t)
        pred_probs = torch.sigmoid(logits)

        # 7. Update last observed values: replace observed positions with x_t
        new_last_x = m_t * x_t + (1.0 - m_t) * last_x

        return pred_probs, h_t, new_last_x

    def forward_sequence(
        self,
        x_seq: torch.Tensor,      # [B, T, K]
        m_seq: torch.Tensor,      # [B, T, K]
        delta_seq: torch.Tensor,  # [B, T, K]
    ) -> torch.Tensor:
        """Batch sequence forward pass for offline supervised training."""
        B, T, K = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, dtype=torch.float32, device=x_seq.device)
        last_x = torch.zeros(B, K, dtype=torch.float32, device=x_seq.device)
        preds = []

        for t in range(T):
            xt = x_seq[:, t, :]
            mt = m_seq[:, t, :]
            dt = delta_seq[:, t, :]
            pred, h, last_x = self.forward_step(xt, mt, dt, last_x, h)
            preds.append(pred.unsqueeze(1))

        return torch.cat(preds, dim=1)  # [B, T, K]


class NumpyGRUDModel:
    """Ultra-fast pure-NumPy inference mirror of GRUDSpectrumCell.
    
    Runs a single forward step in < 0.04 ms on CPU without PyTorch tensor
    conversion overhead.
    """

    def __init__(self, cell: GRUDSpectrumCell):
        self.num_bands = cell.num_bands
        self.hidden_dim = cell.hidden_dim

        # Extract weights as NumPy arrays
        with torch.no_grad():
            self.w_gamma_x = cell.w_gamma_x.detach().cpu().numpy().astype(np.float64)
            self.b_gamma_x = cell.b_gamma_x.detach().cpu().numpy().astype(np.float64)
            self.w_gamma_h = cell.w_gamma_h.weight.detach().cpu().numpy().astype(np.float64)  # [H, K]
            self.b_gamma_h = cell.w_gamma_h.bias.detach().cpu().numpy().astype(np.float64)    # [H]
            self.x_mean = cell.x_mean.detach().cpu().numpy().astype(np.float64)

            self.w_z = cell.w_z.weight.detach().cpu().numpy().astype(np.float64)              # [H, In]
            self.b_z = cell.w_z.bias.detach().cpu().numpy().astype(np.float64)
            self.w_r = cell.w_r.weight.detach().cpu().numpy().astype(np.float64)
            self.b_r = cell.w_r.bias.detach().cpu().numpy().astype(np.float64)
            self.w_h = cell.w_h.weight.detach().cpu().numpy().astype(np.float64)
            self.b_h = cell.w_h.bias.detach().cpu().numpy().astype(np.float64)

            # FC Out
            fc0 = cell.fc_out[0]
            fc2 = cell.fc_out[2]
            self.w_fc0 = fc0.weight.detach().cpu().numpy().astype(np.float64)
            self.b_fc0 = fc0.bias.detach().cpu().numpy().astype(np.float64)
            self.w_fc2 = fc2.weight.detach().cpu().numpy().astype(np.float64)
            self.b_fc2 = fc2.bias.detach().cpu().numpy().astype(np.float64)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

    def forward_step(
        self,
        x_t: np.ndarray,      # [K]
        m_t: np.ndarray,      # [K]
        delta_t: np.ndarray,  # [K]
        last_x: np.ndarray,   # [K]
        h_prev: np.ndarray,   # [H]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-step pure NumPy inference."""
        # 1. Decay rates
        gamma_x = np.exp(-np.maximum(0.0, self.w_gamma_x * delta_t + self.b_gamma_x))
        gamma_h = np.exp(-np.maximum(0.0, self.w_gamma_h @ delta_t + self.b_gamma_h))

        # 2. Input imputation
        x_decayed = gamma_x * last_x + (1.0 - gamma_x) * self.x_mean
        x_hat = m_t * x_t + (1.0 - m_t) * x_decayed

        # 3. Hidden state decay
        h_hat = gamma_h * h_prev

        # 4. Gate input: [x_hat, h_hat, m_t]
        cat_in = np.concatenate([x_hat, h_hat, m_t])

        # 5. GRU gating
        z = self._sigmoid(self.w_z @ cat_in + self.b_z)
        r = self._sigmoid(self.w_r @ cat_in + self.b_r)

        cat_cand = np.concatenate([x_hat, r * h_hat, m_t])
        h_tilde = np.tanh(self.w_h @ cat_cand + self.b_h)

        h_t = (1.0 - z) * h_hat + z * h_tilde

        # 6. Output prediction
        fc_hidden = np.maximum(0.0, self.w_fc0 @ h_t + self.b_fc0)
        pred_probs = self._sigmoid(self.w_fc2 @ fc_hidden + self.b_fc2)

        # 7. Update last observed
        new_last_x = m_t * x_t + (1.0 - m_t) * last_x

        return pred_probs, h_t, new_last_x
