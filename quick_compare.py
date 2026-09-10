import numpy as np
import warnings
warnings.filterwarnings("ignore")

from simulator.environment import SmartScanEnv
from evaluation.benchmark import evaluate_policy
from benchmark_models.mathematical_baselines import FixedScheduler, NMFScheduler
from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
from benchmark_models.hysteresis_meta_controller import HysteresisMetaController

def run_comparison():
    print("==========================================================")
    print("LIVE SIMULATION: THE FADING PROBLEM (Harsh Noise Scenario)")
    print("Simulating signal dropouts, thick static, and fading...")
    print("==========================================================\n")
    
    # We will test on the Harsh Noise (Fading) scenario
    def env_fn(seed):
        return SmartScanEnv(num_bands=20, episode_length=150, seed=seed, scenario="harsh")
        
    schedulers = {
        "Dumb Baseline: Sequential Sweep": lambda n: FixedScheduler(n),
        "Pure Math: Direct NMF": lambda n: NMFScheduler(n),
        "Math + Grace Period: Dwell-Dual Policy": lambda n: DwellDualPolicyScheduler(n),
        "Adaptive: Your Hysteresis Meta-AI": lambda n: HysteresisMetaController(n, lock_steps=20)
    }
    
    for name, builder in schedulers.items():
        # Evaluate for 3 episodes to get a stable mean
        res = evaluate_policy(env_fn, builder, seed=42, episodes=3)
        mean_reward = res["mean_reward"]
        tps = res["mean_true_positives"]
        
        # Determine performance color / icon string
        if mean_reward > 0:
            icon = "SURVIVED"
        elif mean_reward > -2:
            icon = "STRUGGLED"
        else:
            icon = "CRASHED "
            
        print(f"[{icon}] {name}")
        print(f"   => Mean Reward    : {mean_reward:+.2f}")
        print(f"   => True Positives : {tps:.1f} signals caught\n")

if __name__ == "__main__":
    run_comparison()
