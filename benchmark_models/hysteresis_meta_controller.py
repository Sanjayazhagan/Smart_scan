import numpy as np
from scheduler.baselines import BaseScheduler
from benchmark_models.mathematical_baselines import NMFScheduler, RandomScheduler, FixedScheduler
from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
from benchmark_models.robust_pca_psr import RobustPCAPSRScheduler

class HysteresisMetaController(BaseScheduler):
    """
    A 'Bandit of Bandits' Meta-Controller with Hysteresis Guards.
    It runs multiple expert algorithms simultaneously, updating all their internal brains 
    with every observation. It uses UCB to select which expert gets driving privileges 
    for a locked window of N steps (Hysteresis) to prevent thrashing.
    """
    def __init__(self, num_bands: int, lock_steps: int = 15):
        super().__init__(num_bands)
        self.lock_steps = lock_steps
        
        # 1. Instantiate ALL the Experts
        self.experts = [
            RandomScheduler(num_bands),          # Expert 0: Uniform Random (Baseline)
            FixedScheduler(num_bands),           # Expert 1: Sequential Sweep
            NMFScheduler(num_bands),             # Expert 2: Direct NMF (Stationary Specialist)
            DwellDualPolicyScheduler(num_bands), # Expert 3: Dwell-Dual (Hopping/Crowded Specialist)
            RobustPCAPSRScheduler(num_bands)     # Expert 4: Robust PCA (Noise & Fading Specialist)
        ]
        
        self.num_experts = len(self.experts)
        self.expert_rewards = np.zeros(self.num_experts)
        self.expert_pulls = np.zeros(self.num_experts)
        
        self.current_expert = 0
        self.steps_in_current_lock = 0
        self.current_lock_reward_sum = 0.0
        self.total_meta_steps = 0
        
    def select_band(self) -> int:
        # If we need to pick a new expert (lock expired or start of episode)
        if self.steps_in_current_lock >= self.lock_steps or self.total_meta_steps == 0:
            if self.total_meta_steps > 0:
                # Resolve the previous lock
                avg_reward = self.current_lock_reward_sum / max(1, self.steps_in_current_lock)
                self.expert_rewards[self.current_expert] += avg_reward
                self.expert_pulls[self.current_expert] += 1
                
            # Pick next expert using UCB
            unexplored = np.where(self.expert_pulls == 0)[0]
            if len(unexplored) > 0:
                self.current_expert = unexplored[0]
            else:
                # UCB formula balancing Exploration and Exploitation of the EXPERTS
                avg_rewards = self.expert_rewards / self.expert_pulls
                exploration = 0.5 * np.sqrt(np.log(self.total_meta_steps) / self.expert_pulls)
                ucb_scores = avg_rewards + exploration
                self.current_expert = int(np.argmax(ucb_scores))
                
            # Reset lock metrics
            self.steps_in_current_lock = 0
            self.current_lock_reward_sum = 0.0
            self.total_meta_steps += 1
            
        # Ask the currently active expert for its decision
        return self.experts[self.current_expert].select_band()
        
    def update(self, band: int, reward: float, obs_dict: dict = None):
        # Update our meta-controller tracking
        self.steps_in_current_lock += 1
        self.current_lock_reward_sum += reward
        
        # CRITICAL: We feed the observation to ALL experts so their internal 
        # matrices (like NMF) stay perfectly up-to-date with reality, even if 
        # they didn't make the decision!
        for expert in self.experts:
            expert.update(band, reward, obs_dict)
