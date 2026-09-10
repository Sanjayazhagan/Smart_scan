import json
from pathlib import Path

ROOT = Path(r"c:\Users\asus\Documents\SMART SCAN")
JSON_PATH = ROOT / "results" / "master_15seed_statistical_benchmark.json"

if not JSON_PATH.exists():
    print(f"File not found: {JSON_PATH}")
    exit(1)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 95)
print("EXTENSIVE 18-MODEL CANONICAL BENCHMARK (15 SEEDS, 6 SCENARIOS, 1,620 EPISODES)")
print("=" * 95)
print(f"{'Rank':<5} | {'Model Name':<30} | {'Mean Reward (95% CI)':<26} | {'Hit Rate %':<10} | {'Latency'}")
print("-" * 95)
for name, s in data["leaderboard"].items():
    ci = f"{s['mean_reward']:+6.2f} [{s['reward_95ci'][0]:+5.2f}, {s['reward_95ci'][1]:+5.2f}]"
    print(f"#{s['final_rank']:<4} | {name:<30} | {ci:<26} | {s['interception_rate_pct']:5.1f}%     | {s['latency_mean_ms']:.3f} ms")
print("=" * 95)

print("\nSCENARIO BREAKDOWN (MEAN REWARDS):")
scenarios = data["run_metadata"]["scenarios"]
hdr = f"{'Rank':<5} | {'Model Name':<28} | " + " | ".join([f"{sc.capitalize():<10}" for sc in scenarios])
print(hdr)
print("-" * len(hdr))
for name, s in data["leaderboard"].items():
    row = [f"{s['scenario_stats'][sc]['mean']:+6.2f}" for sc in scenarios]
    print(f"#{s['final_rank']:<4} | {name:<28} | " + " | ".join([f"{r:<10}" for r in row]))
print("=" * len(hdr))
