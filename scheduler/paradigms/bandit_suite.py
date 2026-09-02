"""Multi-Armed Bandit Suite: Thompson Sampling and Adversarial Exp3.

Implements:
1. ThompsonSamplingScheduler: Bayesian probability matching with Beta-Bernoulli conjugate priors
2. Exp3BanditScheduler: Exponential-weight algorithm for non-stationary / adversarial environments
"""

import numpy as np


class ThompsonSamplingScheduler:
    """Bayesian Thompson Sampling with Beta conjugate priors and switching awareness."""

    def __init__(
        self,
        num_bands: int = 20,
        prior_alpha: float = 1.0,
        prior_beta: float = 5.0,
        staleness_weight: float = 0.05,
        switch_penalty: float = 0.03,
        seed: int = 42,
    ):
        self.num_bands = int(num_bands)
        self.staleness_weight = float(staleness_weight)
        self.switch_penalty = float(switch_penalty)
        self.rng = np.random.default_rng(seed)

        # Beta distributions parameters per arm
        self.alpha = np.full(self.num_bands, prior_alpha, dtype=np.float32)
        self.beta = np.full(self.num_bands, prior_beta, dtype=np.float32)
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)
        self.last_band = 0

    def select_band(self) -> int:
        """Draws posterior samples and selects band with highest sampled reward."""
        samples = self.rng.beta(self.alpha, self.beta)
        scores = samples + self.staleness_weight * np.sqrt(np.maximum(self.scan_age, 0.0))

        for b in range(self.num_bands):
            switch_dist = abs(b - self.last_band) / max(1, self.num_bands - 1)
            scores[b] -= self.switch_penalty * switch_dist

        return int(np.argmax(scores))

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Updates conjugate Beta parameters with observed outcome."""
        band = int(band)
        self.last_band = band

        self.scan_age += 1.0
        self.scan_age[band] = 0.0

        detected = 0
        if obs_dict is not None:
            detected = int(obs_dict.get("detected", 0))

        # Discount old evidence to adapt to non-stationary agility
        self.alpha[band] = 0.94 * self.alpha[band] + (1.5 if detected else 0.0)
        self.beta[band] = 0.94 * self.beta[band] + (0.1 if detected else 1.0)

        # Prevent parameters from collapsing to zero
        self.alpha[band] = max(0.5, self.alpha[band])
        self.beta[band] = max(0.5, self.beta[band])


class Exp3BanditScheduler:
    """Adversarial Exp3 (Exponential-weight algorithm for Exploration and Exploitation)."""

    def __init__(
        self,
        num_bands: int = 20,
        gamma: float = 0.15,
        switch_penalty: float = 0.03,
        seed: int = 42,
    ):
        self.num_bands = int(num_bands)
        self.gamma = float(gamma)
        self.switch_penalty = float(switch_penalty)
        self.rng = np.random.default_rng(seed)

        self.weights = np.ones(self.num_bands, dtype=np.float64)
        self.probabilities = np.ones(self.num_bands, dtype=np.float64) / self.num_bands
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)
        self.last_band = 0

    def select_band(self) -> int:
        """Selects band according to exponential probability distribution."""
        total_w = np.sum(self.weights)
        if total_w <= 0 or np.isnan(total_w):
            self.weights = np.ones(self.num_bands, dtype=np.float64)
            total_w = self.num_bands

        # Exp3 mixture: (1 - gamma) * exploitation + gamma * uniform exploration
        p = (1.0 - self.gamma) * (self.weights / total_w) + (self.gamma / self.num_bands)
        self.probabilities = p

        # Sample from probability distribution
        return int(self.rng.choice(self.num_bands, p=self.probabilities))

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Calculates importance-weighted reward and updates exponential weights."""
        band = int(band)
        self.last_band = band

        self.scan_age += 1.0
        self.scan_age[band] = 0.0

        # Normalize reward from [-1.0, 1.0] to [0.0, 1.0]
        norm_reward = np.clip((reward + 1.0) / 2.0, 0.0, 1.0)

        # Importance-weighted estimated reward
        p_band = max(1e-4, self.probabilities[band])
        estimated_reward = norm_reward / p_band

        # Exponential weight update with numerical stability clip
        growth = np.exp(np.clip((self.gamma * estimated_reward) / self.num_bands, -10.0, 10.0))
        self.weights[band] *= growth

        # Periodic weight normalization to prevent overflow
        max_w = np.max(self.weights)
        if max_w > 1e6:
            self.weights /= max_w
