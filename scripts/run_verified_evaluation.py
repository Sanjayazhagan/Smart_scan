import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.world_model_ucb import WorldModelUCBScheduler
from scheduler.adaptive_moe import ObservableDiscountedUCBScheduler
from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH

scenarios = ["operational", "hopping", "mixed", "stationary", "harsh", "crowded"]
SEEDS = [101, 102, 103, 104, 105]
STEPS = 150

print("=" * 85)
print("  OFFICIAL VERIFIED EVALUATION: BLIND UCB vs. WORLD-MODEL UCB")
print(f"  Scenarios: {scenarios} | Steps per episode: {STEPS} | Seeds: {SEEDS}")
print("=" * 85)

results = {
    "Blind UCB": {sc: [] for sc in scenarios},
    "World-Model UCB": {sc: [] for sc in scenarios}
}
latencies = {
    "Blind UCB": [],
    "World-Model UCB": []
}

for sc in scenarios:
    for seed in SEEDS:
        # 1. Blind UCB
        env = SmartScanEnv(num_bands=20, episode_length=STEPS, seed=seed, scenario=sc)
        sb = ObservableDiscountedUCBScheduler(20, Track2Runtime(DEFAULT_MODEL_PATH), neural_guidance_scale=0.0, manage_runtime=True)
        env.reset(seed=seed)
        r_b = 0.0
        for st in range(STEPS):
            t0 = time.perf_counter()
            a = sb.select_band()
            latencies["Blind UCB"].append((time.perf_counter() - t0) * 1000)
            obs, r, d, tr, info = env.step(a)
            sb.update(a, r, obs)
            r_b += r
        results["Blind UCB"][sc].append(r_b)

        # 2. World-Model UCB
        env = SmartScanEnv(num_bands=20, episode_length=STEPS, seed=seed, scenario=sc)
        sw = WorldModelUCBScheduler(20, model_path=DEFAULT_MODEL_PATH)
        env.reset(seed=seed)
        r_w = 0.0
        for st in range(STEPS):
            t0 = time.perf_counter()
            a = sw.select_band()
            latencies["World-Model UCB"].append((time.perf_counter() - t0) * 1000)
            obs, r, d, tr, info = env.step(a)
            sw.update(a, r, obs)
            r_w += r
        results["World-Model UCB"][sc].append(r_w)

print(f"\n{'Scenario':<16} | {'Blind UCB Mean':>16} | {'World-Model Mean':>18} | {'Delta':>10} | {'Winner':>14}")
print("-" * 85)

tot_b = 0.0
tot_w = 0.0

for sc in scenarios:
    mb = float(np.mean(results["Blind UCB"][sc]))
    mw = float(np.mean(results["World-Model UCB"][sc]))
    delta = mw - mb
    tot_b += mb
    tot_w += mw
    winner = "World-Model UCB" if delta > 0.1 else ("Blind UCB" if delta < -0.1 else "Tie")
    print(f"{sc:<16} | {mb:>16.2f} | {mw:>18.2f} | {delta:>+10.2f} | {winner:>14}")

print("=" * 85)
mean_all_b = tot_b / len(scenarios)
mean_all_w = tot_w / len(scenarios)
net_delta = mean_all_w - mean_all_b
print(f"{'OVERALL MEAN':<16} | {mean_all_b:>16.2f} | {mean_all_w:>18.2f} | {net_delta:>+10.2f} | {'World-Model UCB' if net_delta > 0 else 'Blind UCB':>14}")
print(f"Latencies: Blind UCB = {np.mean(latencies['Blind UCB']):.3f} ms | World-Model UCB = {np.mean(latencies['World-Model UCB']):.3f} ms")
print("=" * 85)
