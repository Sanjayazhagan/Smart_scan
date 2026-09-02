import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scheduler.baselines import BaseScheduler

class FrequencyHistoryWrapper(gym.ObservationWrapper):
    """
    Wraps the SmartScanEnv to provide a historical state representation 
    for RL algorithms (like DQN/PPO).
    
    The observation becomes a matrix of shape (num_bands, 3):
    - [:, 0]: Time steps since last scan
    - [:, 1]: Last detected status (0 or 1)
    - [:, 2]: Last signal power
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_bands = self.env.unwrapped.num_bands
        
        # State: (num_bands, 3)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_bands, 3), 
            dtype=np.float32
        )
        self.history = np.zeros((self.num_bands, 3), dtype=np.float32)
        
    def reset(self, **kwargs):
        # We don't strictly need the underlying dummy obs here, just info
        _, info = self.env.reset(**kwargs)
        
        self.history = np.zeros((self.num_bands, 3), dtype=np.float32)
        # Initialize time since last scan to a high value (e.g. 100)
        self.history[:, 0] = 100.0
        
        return self.history.copy(), info
        
    def observation(self, obs):
        # Increment time since last scan for all bands
        self.history[:, 0] += 1.0
        
        # Parse the underlying partial observation Dict
        band = int(obs["selected_band"])
        detected = float(obs["detected"])
        power = float(obs["signal_power"][0])
        
        # Reset timer and update features for the band we just scanned
        self.history[band, 0] = 0.0
        self.history[band, 1] = detected
        self.history[band, 2] = power
        
        return self.history.copy()

class RLScheduler(BaseScheduler):
    """
    Wraps a trained RL model (e.g., from Stable Baselines 3) 
    so it can be used interchangeably with other baseline schedulers.
    """
    def __init__(self, num_bands: int, model):
        super().__init__(num_bands)
        self.model = model
        
        # Mirror the wrapper's history mechanism internally
        self.history = np.zeros((num_bands, 3), dtype=np.float32)
        self.history[:, 0] = 100.0
        
    def select_band(self) -> int:
        action, _states = self.model.predict(self.history, deterministic=True)
        return int(action)
        
    def update(self, band: int, reward: float, obs_dict: dict = None):
        """Extended update method to accept observation dict to update history."""
        self.history[:, 0] += 1.0
        if obs_dict is not None:
            self.history[band, 0] = 0.0
            self.history[band, 1] = float(obs_dict["detected"])
            self.history[band, 2] = float(obs_dict["signal_power"][0])

def train_ppo_agent(env_fn, total_timesteps=10000):
    """Utility to quickly train a PPO agent."""
    from stable_baselines3 import PPO
    
    env = env_fn()
    env = FrequencyHistoryWrapper(env)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    
    return model


def _main():
    import argparse
    from pathlib import Path

    from simulator.environment import SmartScanEnv

    parser = argparse.ArgumentParser(description="Train Frequency-History PPO")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", default="models/frequency_history_ppo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = train_ppo_agent(
        lambda: SmartScanEnv(num_bands=20, seed=args.seed),
        total_timesteps=args.timesteps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))


if __name__ == "__main__":
    _main()
