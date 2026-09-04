"""Dev-set tuning for Dwell-Aware Dual-Policy Scheduler.

Tests different dwell_inertia values (0.4, 0.8, 1.2, 1.6) and switch_penalties (0.05, 0.08, 0.12)
across 3 dev seeds [5001, 5002, 5003] and all 6 scenarios (18 episodes per config).
Target: Match Direct NMF's stability while retaining multi-band agility.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.dwell_dual_policy_scheduler import DwellDualPolicyScheduler
from scheduler.dual_policy_uncertainty_scheduler import DualPolicyUncertaintyScheduler

DEV_SEEDS = [5001, 5002, 5003]
SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
STEPS = 150

def run_eval(builder):
    rewards = []
    switches = []
    for sc in SCENARIOS:
        for s in DEV_SEEDS:
            env = SmartScanEnv(num_bands=20, episode_length=STEPS, seed=s, scenario=sc)
            sched = builder(20, s)
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

print("Evaluating Direct NMF on Dev Set...")
nmf_r, nmf_sw = run_eval(lambda n, s: NMFScheduler(n, seed=s))
print(f"  Direct NMF Mean Reward: {nmf_r:+.2f} | Switches: {nmf_sw:.1f}")

configs = [
    # (dwell_inertia, switch_penalty, explore_prob)
    (0.40, 0.05, 0.20),
    (0.80, 0.05, 0.20),
    (0.80, 0.08, 0.15),
    (1.20, 0.08, 0.15),
    (1.20, 0.12, 0.10),
    (1.60, 0.12, 0.10),
]

best_cfg = None
best_r = -1e9
for d_in, sw_pen, exp_p in configs:
    builder = lambda n, s, d=d_in, sp=sw_pen, ep=exp_p: DwellDualPolicyScheduler(
        n, dwell_inertia=d, switch_penalty=sp, explore_budget_prob=ep, seed=s
    )
    r, sw = run_eval(builder)
    print(f"  Dwell={d_in:.2f}, SwPen={sw_pen:.2f}, ExpP={exp_p:.2f} | Mean Reward: {r:+.2f} (diff vs NMF: {r - nmf_r:+.2f}) | Switches: {sw:.1f}")
    if r > best_r:
        best_r = r
        best_cfg = (d_in, sw_pen, exp_p)

print(f"\nOptimal Dev Configuration: {best_cfg} with Mean Reward: {best_r:+.2f}")
