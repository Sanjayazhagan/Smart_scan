"""Non-Negative Matrix Factorization (NMF) Spectrum Scheduler.

Decomposes continuous channel power history V ~ W * H into:
- W (20 x K): Spatial emitter frequency basis signatures
- H (K x W): Temporal activation coefficients

Predicts next-step spectrum activation using multiplicative Lee-Seung updates.
"""

from collections import deque
import numpy as np


def nmf_factorize(
    v: np.ndarray,
    n_components: int = 4,
    max_iter: int = 30,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Lee & Seung Multiplicative Update Rules for Non-Negative Matrix Factorization.

    Approximates V ~ W @ H where W >= 0, H >= 0.
    """
    n_features, n_samples = v.shape
    rng = np.random.default_rng(101)

    # Initialize non-negative matrices W and H
    avg = np.sqrt(np.mean(v) / n_components)
    w = np.abs(rng.normal(avg, avg * 0.2, (n_features, n_components))).astype(np.float32)
    h = np.abs(rng.normal(avg, avg * 0.2, (n_components, n_samples))).astype(np.float32)

    v = np.maximum(v, eps)

    for _ in range(max_iter):
        # Update H
        wt = w.T
        h *= (wt @ v) / (wt @ w @ h + eps)
        h = np.maximum(h, eps)

        # Update W
        ht = h.T
        w *= (v @ ht) / (w @ h @ ht + eps)
        w = np.maximum(w, eps)

    # Normalize columns of W
    col_norms = np.linalg.norm(w, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0
    w /= col_norms
    h *= col_norms.T

    return w, h


class NMFScheduler:
    """Cognitive Spectrum Scheduler using Non-Negative Matrix Factorization."""

    def __init__(
        self,
        num_bands: int = 20,
        n_components: int = 4,
        window_size: int = 30,
        staleness_weight: float = 0.05,
        switch_penalty: float = 0.03,
        recompute_every: int = 1,
        seed: int = 42,
    ):
        self.num_bands = int(num_bands)
        self.n_components = int(n_components)
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

        self.w_basis: np.ndarray | None = None
        self.h_activation: np.ndarray | None = None
        self.predicted_spectrum = np.ones(self.num_bands, dtype=np.float32) / self.num_bands
        self.update_count = 0
        self.factorization_count = 0

        for _ in range(self.window_size):
            self.history.append(np.full(self.num_bands, 0.05, dtype=np.float32))

    def select_band(self) -> int:
        """Selects band with highest reconstructed energy + staleness exploration."""
        scores = np.copy(self.predicted_spectrum)
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
        """Updates sliding observation matrix, runs NMF, and forecasts next spectrum."""
        band = int(band)
        self.last_band = band

        self.scan_age += 1.0
        self.scan_age[band] = 0.0

        power = 0.05
        detected = 0
        if obs_dict is not None:
            detected = int(obs_dict.get("detected", 0))
            raw_power = obs_dict.get("signal_power", 0.0)
            if isinstance(raw_power, (np.ndarray, list, tuple)):
                power = float(raw_power[0]) if len(raw_power) > 0 else 0.0
            else:
                power = float(raw_power)

        last_slice = np.copy(self.history[-1])
        last_slice *= 0.95
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

        v_matrix = np.array(self.history, dtype=np.float32).T  # [num_bands, window_size]
        w, h = nmf_factorize(v_matrix, n_components=self.n_components, max_iter=20)
        self.factorization_count += 1
        self.w_basis = w
        self.h_activation = h

        # Autoregress temporal activations: h_next = 0.75 * h_last + 0.25 * mean(h)
        h_last = h[:, -1]
        h_mean = np.mean(h[:, -5:], axis=1)
        h_next = 0.75 * h_last + 0.25 * h_mean

        # Forecast spectrum: v_next = W @ h_next
        reconstructed = w @ h_next
        total_energy = np.sum(reconstructed)
        if total_energy > 1e-6:
            self.predicted_spectrum = reconstructed / total_energy
        else:
            self.predicted_spectrum = np.ones(self.num_bands, dtype=np.float32) / self.num_bands
