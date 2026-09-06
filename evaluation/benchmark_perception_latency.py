"""Standalone CPU Perception-Layer Latency Benchmark: 1D-CNN I/Q Prototype Matching.

Measures isolated CPU inference and cosine matching latency over N=5,000 real-time trials:
- 1D-CNN Forward Pass (2 channels x 512 I/Q samples -> 64-dim L2-normalized embedding)
- 10-Class Prototype Cosine Similarity Matching Matrix Vector Product
- Threshold Authentication Decision (tau = 0.7415)

WARM-UP BIAS CORRECTION:
- 50 untimed pre-benchmark forward passes to warm instruction caches and PyTorch runtime.
- The first 10 timed samples in the evaluation loop are discarded as a safety buffer
  to eliminate transient CPU frequency scaling or OS context switch artifacts.
- Both parametric (Mean, Std) and non-parametric (Median, P90, P95, P99, Min, Max)
  latencies are reported alongside canonical run provenance metadata.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import psutil
import torch
import torch.nn.functional as F

import sys
ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
sys.path.insert(0, str(ROOT))

from scheduler.track2_runtime import Track2Runtime, DEFAULT_MODEL_PATH
from evaluation.provenance import get_run_metadata

NUM_TRIALS = 5000
UNTIMED_WARMUP_PASSES = 50
DISCARDED_INITIAL_TIMED_SAMPLES = 10
CPU_IDLE_THRESHOLD_PCT = 15.0


def verify_cpu_idle_state(max_retries=3):
    """Verify system is in an uncontended, idle state before latency measurement."""
    import os
    current_pid = os.getpid()
    parent_pid = os.getppid()
    
    # 1. Audit active Python processes (excluding current runner and its launcher parent)
    other_python_procs = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
        try:
            p_name = (p.info['name'] or '').lower()
            if 'python' in p_name and p.info['pid'] not in (current_pid, parent_pid) and p.info['ppid'] != current_pid:
                other_python_procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    print("\n" + "=" * 105)
    print("  CPU IDLE & PROCESS CONTENTION VERIFICATION AUDIT")
    print("=" * 105)
    print(f"  Current Runner PID: {current_pid}")
    if other_python_procs:
        print(f"  [!] Detected {len(other_python_procs)} concurrent Python process(es):")
        for proc in other_python_procs:
            cmd = " ".join(proc['cmdline'] or [])[:80]
            print(f"      - PID {proc['pid']}: {cmd}")
    else:
        print("  Active Concurrent Python Processes: NONE (Clean isolated environment)")

    # 2. Multi-sample CPU utilization check
    reading_1 = psutil.cpu_percent(interval=1.0)
    reading_2 = psutil.cpu_percent(interval=1.0)
    
    print(f"  CPU Utilization Sample 1 (1.0s window): {reading_1:.1f}%")
    print(f"  CPU Utilization Sample 2 (1.0s window): {reading_2:.1f}%")
    
    is_idle = (reading_1 <= CPU_IDLE_THRESHOLD_PCT) and (reading_2 <= CPU_IDLE_THRESHOLD_PCT) and (len(other_python_procs) == 0)
    
    if is_idle:
        print(f"  CPU Status: [PASS] SYSTEM VERIFIED IDLE (Load <= {CPU_IDLE_THRESHOLD_PCT:.1f}%, uncontended single-core)")
    else:
        print(f"\n{'!' * 105}")
        print(f"  [!] WARNING: SYSTEM NOT FULLY IDLE (Sample 1: {reading_1:.1f}%, Sample 2: {reading_2:.1f}%, Threshold: {CPU_IDLE_THRESHOLD_PCT:.1f}%)")
        print("      Background CPU contention or lingering worker pools will corrupt latency measurements,")
        print("      causing thread starvation and 2x-4x latency inflation.")
        print(f"{'!' * 105}\n")
    print("=" * 105 + "\n")
    
    return {
        "verified_idle": is_idle,
        "sample_1_pct": reading_1,
        "sample_2_pct": reading_2,
        "other_python_pids": [p["pid"] for p in other_python_procs],
    }


def run_perception_benchmark(num_trials=5000, model_path=DEFAULT_MODEL_PATH, run_id=None):
    # Perform strict CPU idle verification
    idle_audit = verify_cpu_idle_state()

    run_meta = get_run_metadata(
        "perception_latency_benchmark",
        extra={
            "num_trials": num_trials,
            "untimed_warmup_passes": UNTIMED_WARMUP_PASSES,
            "discarded_initial_samples": DISCARDED_INITIAL_TIMED_SAMPLES,
            "warmup_correction_applied": True,
            "device": "CPU (Single Thread)",
            "cpu_idle_audit": idle_audit,
        },
        run_id=run_id,
    )

    print("=" * 105)
    print("  PERCEPTION-LAYER LATENCY BENCHMARK: 1D-CNN I/Q PROTOTYPE MATCHING IN ISOLATION")
    print(f"  Device: CPU (Single Core Execution) | Trials: {num_trials:,} | Untimed Warmup: {UNTIMED_WARMUP_PASSES}")
    print(f"  Warm-Up Correction: Active ({UNTIMED_WARMUP_PASSES} untimed passes + {DISCARDED_INITIAL_TIMED_SAMPLES} initial timed samples discarded)")
    print(f"  Run ID: {run_meta['run_id']} | Git: {run_meta['git_commit'][:10]}")
    print(f"  Idle Verification: {'VERIFIED IDLE' if idle_audit['verified_idle'] else 'UNVERIFIED / CONTENDED'}")
    print("=" * 105)

    # Force single-thread CPU execution for realistic embedded DSP / SDR measurement
    torch.set_num_threads(1)

    print("Loading Track 2 Identity Model & Prototype Matrix...")
    rt = Track2Runtime(model_path=model_path)
    id_model = rt.identity_model
    id_model.eval()
    prototype_matrix = rt.prototype_matrix  # shape (10, 64)
    threshold = rt.identity_threshold       # 0.7415

    # Synthesize realistic RF baseband I/Q bursts (N, 2, 512)
    rng = np.random.default_rng(42)
    t_axis = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    env = np.exp(-0.5 * (t_axis / 0.4) ** 2)

    total_needed = UNTIMED_WARMUP_PASSES + DISCARDED_INITIAL_TIMED_SAMPLES + num_trials
    synthetic_bursts = []
    for _ in range(total_needed):
        band = rng.integers(0, 20)
        omega = 2.0 * np.pi * (band - 9.5) * 0.1
        i_sig = env * np.cos(omega * t_axis) + rng.normal(0, 0.08, 512).astype(np.float32)
        q_sig = env * np.sin(omega * t_axis) + rng.normal(0, 0.08, 512).astype(np.float32)
        synthetic_bursts.append(torch.tensor(np.stack([i_sig, q_sig]), dtype=torch.float32).unsqueeze(0))

    # 1. Untimed Warmup Phase
    print(f"Executing {UNTIMED_WARMUP_PASSES} untimed warmup passes...")
    with torch.no_grad():
        for i in range(UNTIMED_WARMUP_PASSES):
            x = synthetic_bursts[i]
            z = id_model(x)
            sims = torch.matmul(prototype_matrix, z.squeeze(0))
            _ = bool((sims.max() >= threshold).item())

    # 2. Timed Execution Phase
    raw_cnn_us = []
    raw_match_us = []
    raw_total_us = []

    print(f"Benchmarking {num_trials + DISCARDED_INITIAL_TIMED_SAMPLES:,} timed iterations...")
    timed_start_idx = UNTIMED_WARMUP_PASSES
    timed_end_idx = timed_start_idx + DISCARDED_INITIAL_TIMED_SAMPLES + num_trials

    with torch.no_grad():
        for i in range(timed_start_idx, timed_end_idx):
            x = synthetic_bursts[i]

            t0 = time.perf_counter()
            z = id_model(x)
            t1 = time.perf_counter()

            sims = torch.matmul(prototype_matrix, z.squeeze(0))
            best_sim, _ = torch.max(sims, dim=0)
            _ = bool((best_sim >= threshold).item())
            t2 = time.perf_counter()

            raw_cnn_us.append((t1 - t0) * 1e6)
            raw_match_us.append((t2 - t1) * 1e6)
            raw_total_us.append((t2 - t0) * 1e6)

    # 3. Discard Initial Timed Samples (Safety Margin for Cache Stabilization)
    cnn_latencies_us = raw_cnn_us[DISCARDED_INITIAL_TIMED_SAMPLES:]
    matching_latencies_us = raw_match_us[DISCARDED_INITIAL_TIMED_SAMPLES:]
    total_latencies_us = raw_total_us[DISCARDED_INITIAL_TIMED_SAMPLES:]

    tot_arr = np.array(total_latencies_us)
    cnn_arr = np.array(cnn_latencies_us)
    mat_arr = np.array(matching_latencies_us)

    stats_dict = {
        "run_metadata": run_meta,
        "warmup_correction": {
            "applied": True,
            "untimed_passes": UNTIMED_WARMUP_PASSES,
            "discarded_initial_timed_samples": DISCARDED_INITIAL_TIMED_SAMPLES,
            "retained_trials": len(tot_arr),
        },
        "total_perception_us": {
            "mean": round(float(np.mean(tot_arr)), 2),
            "std": round(float(np.std(tot_arr, ddof=1)), 2),
            "median_p50": round(float(np.percentile(tot_arr, 50)), 2),
            "p90": round(float(np.percentile(tot_arr, 90)), 2),
            "p95": round(float(np.percentile(tot_arr, 95)), 2),
            "p99": round(float(np.percentile(tot_arr, 99)), 2),
            "min": round(float(np.min(tot_arr)), 2),
            "max": round(float(np.max(tot_arr)), 2),
        },
        "total_perception_ms": {
            "mean": round(float(np.mean(tot_arr)) / 1000.0, 4),
            "median_p50": round(float(np.percentile(tot_arr, 50)) / 1000.0, 4),
            "p95": round(float(np.percentile(tot_arr, 95)) / 1000.0, 4),
            "p99": round(float(np.percentile(tot_arr, 99)) / 1000.0, 4),
            "min": round(float(np.min(tot_arr)) / 1000.0, 4),
            "max": round(float(np.max(tot_arr)) / 1000.0, 4),
        },
        "cnn_encoder_us": {
            "mean": round(float(np.mean(cnn_arr)), 2),
            "median_p50": round(float(np.percentile(cnn_arr, 50)), 2),
            "p95": round(float(np.percentile(cnn_arr, 95)), 2),
            "p99": round(float(np.percentile(cnn_arr, 99)), 2),
            "min": round(float(np.min(cnn_arr)), 2),
            "max": round(float(np.max(cnn_arr)), 2),
        },
        "prototype_matching_us": {
            "mean": round(float(np.mean(mat_arr)), 2),
            "median_p50": round(float(np.percentile(mat_arr, 50)), 2),
            "p95": round(float(np.percentile(mat_arr, 95)), 2),
            "p99": round(float(np.percentile(mat_arr, 99)), 2),
            "min": round(float(np.min(mat_arr)), 2),
            "max": round(float(np.max(mat_arr)), 2),
        },
        "throughput_bursts_per_sec": round(1e6 / float(np.mean(tot_arr)), 1),
        "cpu_idle_verification": idle_audit,
    }

    print("\n" + "=" * 105)
    print(f"{'PERCEPTION-LAYER LATENCY IN ISOLATION (N=5,000, Warmup-Corrected)':^105}")
    print("=" * 105)
    print(f"  Stage                                |  Mean (us)  |  Median (us) |   P95 (us)  |   Min (us)  |   Max (us)  ")
    print("-" * 105)
    print(f"  1. 1D-CNN Feature Extraction (IQ)   | {stats_dict['cnn_encoder_us']['mean']:9.2f}   | {stats_dict['cnn_encoder_us']['median_p50']:10.2f}   | {stats_dict['cnn_encoder_us']['p95']:9.2f}   | {stats_dict['cnn_encoder_us']['min']:9.2f}   | {stats_dict['cnn_encoder_us']['max']:9.2f}")
    print(f"  2. Prototype Cosine Matching (10cls)| {stats_dict['prototype_matching_us']['mean']:9.2f}   | {stats_dict['prototype_matching_us']['median_p50']:10.2f}   | {stats_dict['prototype_matching_us']['p95']:9.2f}   | {stats_dict['prototype_matching_us']['min']:9.2f}   | {stats_dict['prototype_matching_us']['max']:9.2f}")
    print("-" * 105)
    print(f"  TOTAL PERCEPTION PIPELINE            | {stats_dict['total_perception_us']['mean']:9.2f}   | {stats_dict['total_perception_us']['median_p50']:10.2f}   | {stats_dict['total_perception_us']['p95']:9.2f}   | {stats_dict['total_perception_us']['min']:9.2f}   | {stats_dict['total_perception_us']['max']:9.2f}")
    print(f"  TOTAL IN MILLISECONDS (ms)           | {stats_dict['total_perception_ms']['mean']:9.4f} ms| {stats_dict['total_perception_ms']['median_p50']:10.4f} ms| {stats_dict['total_perception_ms']['p95']:9.4f} ms| {stats_dict['total_perception_ms']['min']:9.4f} ms| {stats_dict['total_perception_ms']['max']:9.4f} ms")
    print(f"  THROUGHPUT                           | {stats_dict['throughput_bursts_per_sec']:,.1f} bursts / sec")
    print(f"  WARM-UP CORRECTION APPLIED           | YES ({UNTIMED_WARMUP_PASSES} untimed passes + {DISCARDED_INITIAL_TIMED_SAMPLES} initial samples excluded)")
    print(f"  SYSTEM CPU IDLE VERIFICATION         | Sample 1: {idle_audit['sample_1_pct']:.1f}%, Sample 2: {idle_audit['sample_2_pct']:.1f}% -> {'PASS (VERIFIED IDLE)' if idle_audit['verified_idle'] else 'WARN (CONTENDED)'}")
    print("=" * 105)

    out_file = ROOT / "results" / "perception_layer_latency.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"Saved verified latency results to: {out_file}")
    return stats_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perception Latency Benchmark")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit suite run ID")
    parser.add_argument("--trials", type=int, default=5000, help="Number of benchmark trials")
    args = parser.parse_args()
    run_perception_benchmark(num_trials=args.trials, run_id=args.run_id)
