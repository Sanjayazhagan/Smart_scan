import numpy as np

class BaseScheduler:
    """Base class for all RF scan scheduling baseline policies."""
    def __init__(self, num_bands: int):
        self.num_bands = num_bands
        
    def select_band(self) -> int:
        """Returns the index of the band to scan next."""
        raise NotImplementedError
        
    def update(self, band: int, reward: float, obs_dict: dict = None):
        """Updates internal beliefs or values based on the received reward."""
        pass

class FixedScheduler(BaseScheduler):
    """Scans bands in a sequential, round-robin loop."""
    def __init__(self, num_bands: int):
        super().__init__(num_bands)
        self.current_band = 0
        
    def select_band(self) -> int:
        band = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return band

class RandomScheduler(BaseScheduler):
    """Uniformly randomly selects a band to scan at each step."""
    def __init__(self, num_bands: int, seed: int | None = None):
        super().__init__(num_bands)
        self.rng = np.random.default_rng(seed)
        
    def select_band(self) -> int:
        return int(self.rng.integers(0, self.num_bands))

class UCB1Scheduler(BaseScheduler):
    """
    Upper Confidence Bound (UCB1) algorithm to balance exploration 
    of poorly known bands and exploitation of known high-activity bands.
    """
    def __init__(self, num_bands: int):
        super().__init__(num_bands)
        self.counts = np.zeros(num_bands, dtype=int)
        self.values = np.zeros(num_bands, dtype=float)
        self.total_counts = 0
        
    def select_band(self) -> int:
        # 1. Force exploration of unexplored bands first
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return int(unexplored[0])
            
        # 2. UCB calculation
        exploration_term = np.sqrt(2 * np.log(self.total_counts) / self.counts)
        ucb_values = self.values + exploration_term
        return int(np.argmax(ucb_values))
        
    def update(self, band: int, reward: float, obs_dict: dict = None):
        self.counts[band] += 1
        self.total_counts += 1
        
        # Incremental average update
        n = self.counts[band]
        self.values[band] = ((n - 1) / n) * self.values[band] + (1 / n) * reward

class ThompsonSamplingScheduler(BaseScheduler):
    """
    Thompson Sampling using a Beta distribution conjugate prior.
    Assumes binary-like successes, mapping positive rewards to success 
    and non-positive rewards to failure.
    """
    def __init__(self, num_bands: int, seed: int | None = None):
        super().__init__(num_bands)
        self.rng = np.random.default_rng(seed)
        # Beta parameters (alpha, beta) for each band initialized to 1 (uniform prior)
        self.alphas = np.ones(num_bands)
        self.betas = np.ones(num_bands)
        
    def select_band(self) -> int:
        # Sample from the Beta distribution for each arm
        samples = self.rng.beta(self.alphas, self.betas)
        return int(np.argmax(samples))
        
    def update(self, band: int, reward: float, obs_dict: dict = None):
        # Convert environmental reward structure to success/failure for the Beta update.
        # SmartScanEnv rewards: True Positive (+1), True Negative (+0.1), False (+/-) (-1)
        # We classify any positive reward as a successful decision for tracking activity
        if reward > 0:
            self.alphas[band] += 1
        else:
            self.betas[band] += 1
