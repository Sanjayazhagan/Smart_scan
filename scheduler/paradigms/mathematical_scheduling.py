"""Mathematical Scheduling via Restless Multi-Armed Bandits (Whittle Index RMAB).

Treats the 20 RF channels as Restless Bandits where arms evolve state whether
scanned or not. Computes the closed-form Whittle Index under switching costs.
"""

import numpy as np


class WhittleIndexRMABScheduler:
    """Restless Multi-Armed Bandit Scheduler using Closed-Form Whittle Indices."""

    def __init__(
        self,
        num_bands: int = 20,
        discount_factor: float = 0.90,
        switch_penalty: float = 0.05,
        staleness_exploration: float = 0.08,
        seed: int = 42,
    ):
        self.num_bands = int(num_bands)
        self.gamma = float(discount_factor)
        self.switch_penalty = float(switch_penalty)
        self.staleness_exploration = float(staleness_exploration)
        self.rng = np.random.default_rng(seed)

        # Belief probability pi_i = P(Channel i is Active)
        self.belief = np.full(self.num_bands, 0.15, dtype=np.float32)
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)
        self.last_band = 0

        # Estimated 2-state Markov transition matrices per channel
        # p01: P(Silent -> Active), p11: P(Active -> Active)
        self.p01 = np.full(self.num_bands, 0.08, dtype=np.float32)
        self.p11 = np.full(self.num_bands, 0.70, dtype=np.float32)

        # Observation tracking counts for Bayesian transition estimation
        self.counts_01 = np.ones(self.num_bands, dtype=np.float32)
        self.counts_00 = np.full(self.num_bands, 10.0, dtype=np.float32)
        self.counts_11 = np.full(self.num_bands, 5.0, dtype=np.float32)
        self.counts_10 = np.full(self.num_bands, 2.0, dtype=np.float32)
        self.last_known_state = np.zeros(self.num_bands, dtype=np.int32)

    def compute_whittle_indices(self) -> np.ndarray:
        """Computes the closed-form Whittle index for each restless arm."""
        indices = np.zeros(self.num_bands, dtype=np.float32)
        for i in range(self.num_bands):
            pi = self.belief[i]
            p01 = self.p01[i]
            p11 = self.p11[i]

            # Denominator accounts for persistent correlation
            denom = max(0.01, 1.0 - self.gamma * (p11 - p01))
            base_index = (pi - p01) / denom

            # Restless staleness compensation: unobserved arms build uncertainty
            staleness_bonus = self.staleness_exploration * np.sqrt(max(0.0, self.scan_age[i]))

            # Physical antenna distance penalty
            switch_dist = abs(i - self.last_band) / max(1, self.num_bands - 1)
            switch_cost = self.switch_penalty * switch_dist

            indices[i] = base_index + staleness_bonus - switch_cost

        return indices

    def select_band(self) -> int:
        """Selects the channel with the highest Whittle index."""
        indices = self.compute_whittle_indices()
        return int(np.argmax(indices))

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Updates belief states, transition estimates, and restless Markov evolution."""
        band = int(band)
        self.last_band = band

        # Scan age increments for all arms, resets for played arm
        self.scan_age += 1.0
        self.scan_age[band] = 0.0

        detected = 0
        if obs_dict is not None:
            detected = int(obs_dict.get("detected", 0))

        # Update empirical transitions from previous state
        prev_state = self.last_known_state[band]
        curr_state = 1 if detected else 0
        if prev_state == 0 and curr_state == 1:
            self.counts_01[band] += 1.0
        elif prev_state == 0 and curr_state == 0:
            self.counts_00[band] += 1.0
        elif prev_state == 1 and curr_state == 1:
            self.counts_11[band] += 1.0
        elif prev_state == 1 and curr_state == 0:
            self.counts_10[band] += 1.0

        self.last_known_state[band] = curr_state

        # Update Bayesian transition parameters
        self.p01[band] = np.clip(
            self.counts_01[band] / max(1.0, self.counts_01[band] + self.counts_00[band]),
            0.02,
            0.40,
        )
        self.p11[band] = np.clip(
            self.counts_11[band] / max(1.0, self.counts_11[band] + self.counts_10[band]),
            0.40,
            0.95,
        )

        # Direct observation collapse for played arm
        if detected:
            self.belief[band] = 0.92
        else:
            self.belief[band] = 0.03

        # Restless Markov evolution for all passive arms (1-step forward projection)
        for i in range(self.num_bands):
            if i != band:
                p01 = self.p01[i]
                p11 = self.p11[i]
                self.belief[i] = self.belief[i] * p11 + (1.0 - self.belief[i]) * p01
