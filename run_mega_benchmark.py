import numpy as np
import warnings
warnings.filterwarnings("ignore")

from simulator.environment import SmartScanEnv
from evaluation.benchmark import evaluate_policy

# We will dynamically try to load as many advanced models as possible
schedulers = {}

try:
    from benchmark_models.mathematical_baselines import FixedScheduler
    schedulers["Baseline: Sequential Sweep"] = lambda n: FixedScheduler(n)
except: pass

try:
    from benchmark_models.mathematical_baselines import NMFScheduler
    schedulers["Pure Math: Direct NMF"] = lambda n: NMFScheduler(n)
except: pass

try:
    from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
    schedulers["Champion: Dwell-Dual Policy"] = lambda n: DwellDualPolicyScheduler(n)
except: pass

try:
    from benchmark_models.nmf_expectimax import NMFExpectimaxScheduler
    schedulers["Search ML: SmartScan V2-NMF"] = lambda n: NMFExpectimaxScheduler(n)
except: pass

try:
    from benchmark_models.robust_pca_psr import RobustPCAPSRScheduler
    schedulers["Filter ML: Robust PCA + PSR"] = lambda n: RobustPCAPSRScheduler(n)
except: pass

try:
    from benchmark_models.dual_policy_uncertainty import DualPolicyUncertaintyScheduler
    schedulers["Bandit ML: Dual-Policy Uncertainty"] = lambda n: DualPolicyUncertaintyScheduler(n)
except: pass

try:
    from benchmark_models.hysteresis_meta_controller import HysteresisMetaController
    schedulers["OURS: Hysteresis Meta-AI"] = lambda n: HysteresisMetaController(n, lock_steps=20)
except: pass


def run_mega_benchmark():
    print("=========================================================================")
    print("LIVE MEGA-BENCHMARK: THE REAL-WORLD NOISE TEST")
    print(f"Loading {len(schedulers)} advanced algorithms...")
    print("Scenario: 'Harsh' (Simulating thick real-world static and fading)")
    print("=========================================================================\n")
    
    def env_fn(seed):
        return SmartScanEnv(num_bands=20, episode_length=150, seed=seed, scenario="harsh")
    
    results = []
    
    for name, builder in schedulers.items():
        try:
            # We run 5 episodes to get a stable, true mean reward
            res = evaluate_policy(env_fn, builder, seed=8000, episodes=5)
            results.append((res["mean_reward"], res["mean_true_positives"], name))
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            
    # Sort by mean reward descending
    results.sort(reverse=True, key=lambda x: x[0])
    
    print("FINAL NOISE LEADERBOARD (Mean over 5 full episodes)")
    print("-------------------------------------------------------------------------")
    for rank, (reward, tps, name) in enumerate(results, 1):
        if "OURS" in name:
            print(f"#{rank} [WINNER] {name:<31} : {reward:+.2f} Reward ({tps:.1f} signals)")
        else:
            print(f"#{rank}          {name:<31} : {reward:+.2f} Reward ({tps:.1f} signals)")

if __name__ == "__main__":
    run_mega_benchmark()
