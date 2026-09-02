"""Robust PCA (RPCA) + Predictive State Representations (PSR) Scheduler.

Decomposes the sliding multi-channel spectrum into:
- Low-rank matrix L: Ambient stationary spectrum and noise floor
- Sparse matrix S: Sporadic emitter bursts, pulses, and agile frequency hops

Uses Predictive State Representations (PSR) to track the future probability of
burst emergence without hidden state assumptions.
"""

from collections import deque
import numpy as np


def _singular_value_thresholding(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Soft-thresholds the singular values of a matrix: D_tau(X) = U S_tau V^T."""
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return u @ np.diag(s_thresh) @ vh


def _soft_thresholding(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Entrywise soft-thresholding operator S_tau(X) = sgn(X) * max(|X| - tau, 0)."""
    return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0.0)


def inexact_alm_rpca(
    matrix: np.ndarray,
    lambda_param: float | None = None,
    tol: float = 1e-4,
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Inexact Augmented Lagrange Multiplier algorithm for Robust PCA.

    Solves: min ||L||_* + lambda ||S||_1  s.t. L + S = M.
    """
    m, n = matrix.shape
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))

    norm_two = np.linalg.norm(matrix, 2)
    norm_inf = np.linalg.norm(matrix.flatten(), np.inf) / lambda_param
    dual_norm = max(norm_two, norm_inf)
    y = matrix / max(dual_norm, 1e-6)

    l = np.zeros_like(matrix)
    s = np.zeros_like(matrix)
    mu = 1.25 / max(norm_two, 1e-6)
    rho = 1.5

    d_norm = np.linalg.norm(matrix, "fro")
    if d_norm == 0:
        return np.zeros_like(matrix), np.zeros_like(matrix)

    for _ in range(max_iter):
        temp_l = matrix - s + (1.0 / mu) * y
        l = _singular_value_thresholding(temp_l, 1.0 / mu)

        temp_s = matrix - l + (1.0 / mu) * y
        s = _soft_thresholding(temp_s, lambda_param / mu)

        z = matrix - l - s
        y = y + mu * z
        mu *= rho

        if np.linalg.norm(z, "fro") / d_norm < tol:
            break

    return l, s


class RobustPCAPSRScheduler:
    """Cognitive Spectrum Scheduler using RPCA burst isolation + Predictive State Representations."""

    def __init__(
        self,
        num_bands: int = 20,
        window_size: int = 30,
        staleness_weight: float = 0.06,
        switch_penalty: float = 0.04,
        recompute_every: int = 1,
        seed: int = 42,
    ):
        self.num_bands = int(num_bands)
        self.window_size = int(window_size)
        self.staleness_weight = float(staleness_weight)
        self.switch_penalty = float(switch_penalty)
        if recompute_every < 1:
            raise ValueError("recompute_every must be at least 1")
        self.recompute_every = int(recompute_every)
        self.rng = np.random.default_rng(seed)

        self.history = deque(maxlen=self.window_size)
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)
        self.last_band = 0

        self.psr_state = np.ones(self.num_bands, dtype=np.float32) / self.num_bands
        self.sparse_burst_profile = np.zeros(self.num_bands, dtype=np.float32)
        self.last_sparse_matrix: np.ndarray | None = None
        self.last_low_rank_matrix: np.ndarray | None = None
        self.update_count = 0
        self.factorization_count = 0

        for _ in range(self.window_size):
            self.history.append(np.full(self.num_bands, 0.05, dtype=np.float32))

    def select_band(self) -> int:
        """Selects band balancing predicted sparse burst arrival with staleness sweep."""
        scores = np.copy(self.psr_state)
        scores += self.staleness_weight * np.sqrt(np.maximum(self.scan_age, 0.0))

        for b in range(self.num_bands):
            switch_dist = abs(b - self.last_band) / max(1, self.num_bands - 1)
            scores[b] -= self.switch_penalty * switch_dist

        selected = int(np.argmax(scores))
        return selected

    def update(
        self,
        band: int,
        reward: float,
        obs_dict: dict | None = None,
        recompute: bool | None = None,
    ):
        """Updates the sliding buffer, performs RPCA, and steps the PSR state."""
        band = int(band)
        self.last_band = band

        self.scan_age += 1.0
        self.scan_age[band] = 0.0

        power = 0.05
        detected = 0
        if obs_dict is not None:
            detected = int(obs_dict.get("detected", 0))
            raw_power = obs_dict.get("signal_power", 0.0)
            if isinstance(raw_power, np.ndarray):
                power = float(raw_power.item()) if raw_power.size == 1 else float(raw_power[0])
            else:
                power = float(raw_power)

        last_slice = np.copy(self.history[-1])
        last_slice *= 0.96
        last_slice[band] = power if detected else 0.02
        self.history.append(last_slice)

        self.update_count += 1
        should_recompute = (
            self.update_count % self.recompute_every == 0
            if recompute is None
            else bool(recompute)
        )
        if not should_recompute:
            return

        mat = np.array(self.history, dtype=np.float32).T  # [num_bands, window_size]
        l_mat, s_mat = inexact_alm_rpca(mat, max_iter=15)
        self.factorization_count += 1
        self.last_low_rank_matrix = l_mat
        self.last_sparse_matrix = s_mat

        latest_sparse = np.maximum(s_mat[:, -1], 0.0)
        sparse_sum = np.sum(latest_sparse)
        if sparse_sum > 1e-5:
            normalized_sparse = latest_sparse / sparse_sum
        else:
            normalized_sparse = np.ones(self.num_bands, dtype=np.float32) / self.num_bands

        self.sparse_burst_profile = latest_sparse

        self.psr_state = 0.82 * self.psr_state + 0.18 * normalized_sparse
        psr_sum = np.sum(self.psr_state)
        if psr_sum > 0:
            self.psr_state /= psr_sum
