"""Rigorous Experimental Evaluation of Pattern-Change Alarm Detector.

Follows Phase 1 through Phase 5 strictly:
- Frozen Track 2 world model.
- Pre-registered targets loaded from detector/detector_spec.json.
- Separate development split (seeds 5001-5030) for threshold tuning.
- Untouched held-out test split (seeds 6001-6050) for final evaluation.
- Calculates exact TPR, FPR, detection delay, and 95% confidence intervals.
"""

import json
import math
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
from detector.pattern_change_detector import PatternChangeDetector


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculates Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, float(centre - margin)), min(1.0, float(centre + margin)))


def bootstrap_ci(data: list[float], n_boot: int = 2000, confidence: float = 0.95) -> tuple[float, float]:
    """Calculates 95% bootstrap confidence interval for the mean."""
    if not data:
        return (0.0, 0.0)
    arr = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return (low, high)


def inject_pattern_change(env, seed: int, episode_length: int = 150) -> int:
    """Injects an abrupt emitter pattern change at a deterministic timestep t*."""
    t_star = int(45 + (seed % 45))  # t* between 45 and 89
    for emitter in env.world.emitters:
        old_band = int(emitter.bands[0])
        new_band = (old_band + 7 + emitter.id * 3) % env.num_bands
        emitter.bands[t_star:] = new_band
        emitter.behaviour_changes[t_star] = True
        if emitter.activity.any():
            emitter.activity[t_star:] = ~emitter.activity[t_star:]
    return t_star


def run_episode(
    seed: int,
    is_change: bool,
    threshold: float,
    delta: float,
    episode_length: int = 150,
    warmup_steps: int = 20,
) -> dict:
    env = SmartScanEnv(num_bands=20, episode_length=episode_length, scenario="stationary", seed=seed)
    obs, info = env.reset(seed=seed)
    
    t_star = None
    if is_change:
        t_star = inject_pattern_change(env, seed=seed, episode_length=episode_length)

    controller = WorldModelNMFUCBScheduler(num_bands=20)
    detector = PatternChangeDetector(delta=delta, threshold=threshold, warmup_steps=warmup_steps)

    alarms = []
    for st in range(episode_length):
        action = controller.select_band()
        obs, reward, done, truncated, info = env.step(action)
        controller.update(action, reward, obs)
        det_res = detector.step_and_detect(action, obs)
        if det_res["alarm"]:
            alarms.append(st + 1)
        if done or truncated:
            break

    return {
        "seed": seed,
        "is_change": is_change,
        "t_star": t_star,
        "alarms": alarms,
        "episode_length": episode_length,
        "warmup_steps": warmup_steps,
    }


def evaluate_run_set(runs: list[dict], change_window: int = 20) -> dict:
    tp_count = 0
    fn_count = 0
    delays = []
    total_false_alarms = 0
    total_stat_steps = 0
    stat_runs_with_alarm = 0
    total_stat_runs = 0
    total_change_runs = 0

    for r in runs:
        alarms = r["alarms"]
        if r["is_change"]:
            total_change_runs += 1
            t_star = r["t_star"]
            window_end = t_star + change_window
            # Did an alarm fire in [t_star, t_star + change_window]?
            tp_alarms = [a for a in alarms if t_star <= a <= window_end]
            if tp_alarms:
                tp_count += 1
                delays.append(tp_alarms[0] - t_star)
            else:
                fn_count += 1
        else:
            total_stat_runs += 1
            stat_steps = r["episode_length"] - r["warmup_steps"]
            total_stat_steps += stat_steps
            fa_in_run = len([a for a in alarms if a > r["warmup_steps"]])
            total_false_alarms += fa_in_run
            if fa_in_run > 0:
                stat_runs_with_alarm += 1

    tpr = tp_count / max(1, total_change_runs)
    tpr_ci = wilson_score_interval(tp_count, total_change_runs)
    
    fpr_per_1000 = (total_false_alarms / max(1, total_stat_steps)) * 1000.0
    run_fpr = stat_runs_with_alarm / max(1, total_stat_runs)
    run_fpr_ci = wilson_score_interval(stat_runs_with_alarm, total_stat_runs)

    mean_delay = float(np.mean(delays)) if delays else float("nan")
    delay_ci = bootstrap_ci(delays) if delays else (float("nan"), float("nan"))

    return {
        "total_change_runs": total_change_runs,
        "tp_count": tp_count,
        "fn_count": fn_count,
        "tpr": tpr,
        "tpr_ci": tpr_ci,
        "total_stat_runs": total_stat_runs,
        "stat_runs_with_alarm": stat_runs_with_alarm,
        "run_fpr": run_fpr,
        "run_fpr_ci": run_fpr_ci,
        "total_false_alarms": total_false_alarms,
        "total_stat_steps": total_stat_steps,
        "fpr_per_1000": fpr_per_1000,
        "mean_delay": mean_delay,
        "delay_ci": delay_ci,
    }


def main():
    # Load Pre-Registered Spec
    spec_path = Path("detector/detector_spec.json")
    with open(spec_path, "r", encoding="utf-8-sig") as f:
        spec = json.load(f)

    targets = spec["pre_registered_targets"]
    target_tpr = targets["min_true_positive_rate"]
    target_fpr_rate = targets["max_false_positive_rate_per_1000_steps"]
    target_delay = targets["max_mean_detection_delay_steps"]
    change_window = targets["true_positive_window_steps"]

    print("=" * 80)
    print("PHASE 2 PRE-REGISTERED SUCCESS CRITERIA (FROM detector_spec.json):")
    print(f"  - Change detection window   : {change_window} steps")
    print(f"  - Minimum True Positive Rate: {target_tpr * 100:.1f}%")
    print(f"  - Maximum FPR per 1000 steps: {target_fpr_rate * 1000:.1f} alarms (<= 10.0% rate)")
    print(f"  - Maximum Mean Delay        : {target_delay:.1f} steps")
    print("=" * 80)

    # ---------------------------------------------------------
    # PHASE 4A: THRESHOLD TUNING ON DEVELOPMENT SPLIT (30 change + 30 stationary)
    # ---------------------------------------------------------
    dev_seeds = list(range(5001, 5031))
    print(f"\n[PHASE 3 & 4A] Tuning on Development Split: {len(dev_seeds)} change + {len(dev_seeds)} stationary runs (Seeds 5001-5030)...")
    
    threshold_candidates = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]
    best_thresh = 1.6
    best_score = -1e9
    dev_results = {}

    for th in threshold_candidates:
        runs = []
        for s in dev_seeds:
            runs.append(run_episode(s, is_change=True, threshold=th, delta=0.04))
            runs.append(run_episode(s, is_change=False, threshold=th, delta=0.04))
        metrics = evaluate_run_set(runs, change_window=change_window)
        dev_results[th] = metrics
        # Objective: Maximize TPR while keeping FPR low
        score = metrics["tpr"] - 2.0 * (metrics["fpr_per_1000"] / 1000.0)
        print(f"  Threshold {th:3.1f} | Dev TPR: {metrics['tpr']*100:5.1f}% | Dev FPR/1k: {metrics['fpr_per_1000']:5.1f} | Delay: {metrics['mean_delay']:4.1f}")
        if score > best_score:
            best_score = score
            best_thresh = th

    print(f"\nSelected Frozen Threshold: lambda* = {best_thresh} (frozen for held-out evaluation)")

    # ---------------------------------------------------------
    # PHASE 4B: FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SPLIT (50 change + 50 stationary)
    # ---------------------------------------------------------
    test_seeds = list(range(6001, 6051))
    print(f"\n[PHASE 4B] Running Final Evaluation on Held-Out Test Split (Seeds 6001-6050)...")
    print(f"  Evaluating {len(test_seeds)} change runs and {len(test_seeds)} stationary runs at lambda* = {best_thresh}...")

    test_runs = []
    for s in test_seeds:
        test_runs.append(run_episode(s, is_change=True, threshold=best_thresh, delta=0.04))
        test_runs.append(run_episode(s, is_change=False, threshold=best_thresh, delta=0.04))

    final = evaluate_run_set(test_runs, change_window=change_window)

    # ---------------------------------------------------------
    # PHASE 5: VERDICT & RESULTS TABLE
    # ---------------------------------------------------------
    tpr_pass = final["tpr"] >= target_tpr
    fpr_pass = (final["fpr_per_1000"] / 1000.0) <= target_fpr_rate
    delay_pass = final["mean_delay"] <= target_delay
    overall_pass = tpr_pass and fpr_pass and delay_pass

    print("\n" + "=" * 90)
    print("  PHASE 5: FINAL EVALUATION RESULTS (HELD-OUT TEST SPLIT)")
    print("=" * 90)
    print(f"{'Metric':<32} | {'Pre-Reg Target':>15} | {'Measured Value':>15} | {'95% Confidence Interval':>23} | {'Pass?':>6}")
    print("-" * 90)
    
    tpr_str = f"{final['tpr']*100:.1f}%"
    tpr_ci_str = f"[{final['tpr_ci'][0]*100:.1f}%, {final['tpr_ci'][1]*100:.1f}%]"
    print(f"{'True Positive Rate (TPR)':<32} | {'>= 80.0%':>15} | {tpr_str:>15} | {tpr_ci_str:>23} | {'YES' if tpr_pass else 'NO':>6}")

    fpr_rate_str = f"{final['fpr_per_1000']:.1f} / 1k"
    target_fpr_str = f"<= {target_fpr_rate*1000:.1f} / 1k"
    print(f"{'False Alarm Rate (/1000 steps)':<32} | {target_fpr_str:>15} | {fpr_rate_str:>15} | {'[Exact Poisson Count]':>23} | {'YES' if fpr_pass else 'NO':>6}")

    run_fpr_str = f"{final['run_fpr']*100:.1f}%"
    run_fpr_ci_str = f"[{final['run_fpr_ci'][0]*100:.1f}%, {final['run_fpr_ci'][1]*100:.1f}%]"
    print(f"{'Stationary Runs with Alarm':<32} | {'N/A (Diagnostic)':>15} | {run_fpr_str:>15} | {run_fpr_ci_str:>23} | {'N/A':>6}")

    delay_str = f"{final['mean_delay']:.2f} steps"
    delay_ci_str = f"[{final['delay_ci'][0]:.2f}, {final['delay_ci'][1]:.2f}]"
    print(f"{'Mean Detection Delay':<32} | {'<= 12.0 steps':>15} | {delay_str:>15} | {delay_ci_str:>23} | {'YES' if delay_pass else 'NO':>6}")
    print("=" * 90)

    print("\n>>> OFFICIAL VERDICT <<<")
    if overall_pass:
        print("VERDICT: SUCCESS (YES). All pre-registered criteria were met on the held-out test split.")
    else:
        failed_criteria = []
        if not tpr_pass: failed_criteria.append(f"TPR ({final['tpr']*100:.1f}% < 80.0%)")
        if not fpr_pass: failed_criteria.append(f"FPR ({final['fpr_per_1000']:.1f}/1k > 100/1k)")
        if not delay_pass: failed_criteria.append(f"Delay ({final['mean_delay']:.2f} > 12.0 steps)")
        print(f"VERDICT: FAILED (NO). Pre-registered targets were NOT met. Failed criteria: {', '.join(failed_criteria)}.")

    # Save exact results to JSON
    output_data = {
        "frozen_threshold": best_thresh,
        "dev_results": {str(k): v for k, v in dev_results.items()},
        "test_results": final,
        "targets": targets,
        "verdict": "PASS" if overall_pass else "FAIL",
    }
    with open("results/pattern_change_detector_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print("\nFull results saved to results/pattern_change_detector_evaluation.json")

if __name__ == "__main__":
    main()
