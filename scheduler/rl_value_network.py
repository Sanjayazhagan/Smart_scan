"""Belief-State Value Network (RL Critic) for Expectimax Leaf Evaluation.

AlphaZero-style leaf evaluator:
Instead of a static heuristic at tree leaf nodes, this network evaluates the true
long-term discounted mission utility of simulated posterior belief states.
Input: 61-dimensional compact belief vector:
- band_belief [20]
- band_uncertainty [20]
- scan_age [20]
- normalized last_band index [1]
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

DEFAULT_VALUE_NET_PATH = Path("archive/models/belief_value_network.pt")


class BeliefValueNetwork(nn.Module):
    """Lightweight 2-layer MLP predicting expected long-term mission reward from belief state."""

    def __init__(self, input_dim: int = 61, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_scalar(self, feature_vector: np.ndarray) -> float:
        """Fast CPU inference for single leaf node in Expectimax tree."""
        t = torch.from_numpy(feature_vector).float().unsqueeze(0)
        return float(self.forward(t).item())


def extract_belief_features(
    band_belief: np.ndarray,
    band_uncertainty: np.ndarray,
    scan_age: np.ndarray,
    last_band: int | None,
) -> np.ndarray:
    """Extracts normalized 61-dimensional feature vector from a BeliefState."""
    b = np.clip(band_belief, 0.0, 1.0).astype(np.float32)
    u = np.clip(band_uncertainty, 0.0, 1.0).astype(np.float32)
    a = np.clip(scan_age, 0.0, 1.0).astype(np.float32)
    last_val = (float(last_band) / 19.0) if last_band is not None else 0.5
    last_feat = np.array([last_val], dtype=np.float32)
    return np.concatenate([b, u, a, last_feat])


class FastBeliefValueCritic:
    """Wrapper that loads trained weights and provides microsecond leaf evaluation."""

    def __init__(self, model_path: Path | str = DEFAULT_VALUE_NET_PATH):
        self.path = Path(model_path)
        self.model = BeliefValueNetwork()
        self.loaded = False
        if self.path.is_file():
            state_dict = torch.load(self.path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.loaded = True

    def evaluate_leaf(
        self,
        band_belief: np.ndarray,
        band_uncertainty: np.ndarray,
        scan_age: np.ndarray,
        last_band: int | None,
    ) -> float:
        if not self.loaded:
            # Fallback heuristic if weights not present
            return float(np.max(band_belief))
        feats = extract_belief_features(band_belief, band_uncertainty, scan_age, last_band)
        return self.model.predict_scalar(feats)
