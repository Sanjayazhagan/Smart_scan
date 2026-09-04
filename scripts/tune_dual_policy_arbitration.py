"""Development set tuning for Dual-Policy Exploit/Explore Arbitration.

Tuning across 3 dev seeds (5001, 5002, 5003) and all 6 scenarios (18 episodes per config).
Tests:
- Baseline: Static NMF+UCB (with switch_penalty = 0.05)
- Probabilistic arbitration: prob in [0.05, 0.15, 0.25, 0.35]
- Periodic arbitration: interval in [5, 8, 12]
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from scheduler.dual_policy_uncertainty_scheduler import DualPolicyUncertaintyScheduler
from scheduler.track2_runtime import DEFAULT_MODEL_PATH

DEV_SEEDS = [5001, 5002, 5003]
SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
STEPS = 150

def run_eval(scheduler_builder):
    rewards = []
    switches = []
    for sc in SCENARIOS:
        for s in DEV_SEEDS:
            env = SmartScanEnv(num_bands=20, episode_length=STEPS, seed=s, scenario=sc)
            sched = scheduler_builder(20, s)
            obs, info = env.reset(seed=s)
            ep_r = 0.0
            ep_sw = 0
            last_a = -1
            for _ in range(STEPS):
                a = sched.select_band()
                if last_a != -1 and a != last_a:
                    ep_sw += 1
                last_a = a
                obs, r, d, tr, info = env.step(a)
                sched.update(a, r, obs)
                ep_r += r
                if d or tr: break
            rewards.append(ep_r)
            switches.append(ep_sw)
    return float(np.mean(rewards)), float(np.mean(switches))

print("Evaluating Baseline: Static NMF+UCB (switch_penalty=0.05)...")
base_r, base_sw = run_eval(lambda n, s: WorldModelNMFUCBScheduler(n, nmf_scale=1.0, world_model_scale=0.0, switch_penalty=0.05, model_path=DEFAULT_MODEL_PATH))
print(f"  Baseline Mean Reward: {base_r:.2f} | Mean Switches: {base_sw:.1f}")

configs = [
    ("probabilistic", 0.05, 0),
    ("probabilistic", 0.15, 0),
    ("probabilistic", 0.25, 0),
    ("probabilistic", 0.35, 0),
    ("periodic", 0.0, 5),
    ("periodic", 0.0, 8),
    ("periodic", 0.0, 12),
]

best_cfg = None
best_r = -1e9
for mode, prob, interval in configs:
    desc = f"{mode}_p{prob}" if mode == "probabilistic" else f"{mode}_int{interval}"
    builder = lambda n, s, m=mode, p=prob, it=interval: DualPolicyUncertaintyScheduler(
        n, switch_penalty=0.05, arbitration_mode=m, explore_budget_prob=p, periodic_explore_interval=it, model_path=DEFAULT_MODEL_PATH
    )
    r, sw = run_eval(builder)
    print(f"  Config {desc:<20} | Mean Reward: {r:+.2f} (diff vs base: {r - base_r:+.2f}) | Switches: {sw:.1f}")
    if r > best_r:
        best_r = r
        best_cfg = (mode, prob, interval)

print(f"\nOptimal Dev Config: {best_cfg} with Mean Reward: {best_r:.2f}")
