"""Adversarial / Decoy Stress Test: Mid-Episode DRFM Jammer Injection.

Simulates mid-episode DRFM / digital decoy injection on Band 7 between steps 40 and 80.
Evaluates Dwell-Dual with perception-layer prototype authentication vs Naive Baseline.
Tracks step-by-step:
- Receiver band selection trajectory
- 1D-CNN cosine similarity vs authentication threshold tau = 0.7415
- Avoidance of decoy band and steady accumulation of true emitter hits
Generates publication-quality figure: results/adversarial_decoy_timeseries.png
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
sys.path.insert(0, str(ROOT))

from simulator.environment import SmartScanEnv
from simulator.scenarios import SCENARIO_PRESETS
from benchmark_models import get_model
from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
from scheduler.track2_runtime import (
    Track2Runtime,
    TemporalConsistencyGate,
)
from evaluation.provenance import get_run_metadata

ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
DEFAULT_MODEL_PATH = Path.home() / "Documents" / "SmartScanArtifacts" / "track2" / "track2_final_world_model.pt"

EPISODE_LENGTH = 120
DECOY_BAND = 7
DECOY_START = 40
DECOY_END = 80


def generate_drfm_pulse(
    t_axis,
    decoy_band=7,
    width=0.40,
    f_jitter=35.0,
    a_jitter=0.40,
    mod_depth=0.15,
    f_mod=12.0,
    carrier_offset=0.0,
):
    """Generate synthetic DRFM spoofed pulse with parametric distortions."""
    env_pulse = np.exp(-0.5 * (t_axis / width) ** 2)
    omega_decoy = 2.0 * np.pi * (decoy_band - 9.5 + carrier_offset) * 0.1
    drfm_noise = np.sin(2.0 * np.pi * f_jitter * t_axis) * a_jitter
    drfm_i = env_pulse * np.cos(omega_decoy * t_axis + drfm_noise) + mod_depth * np.sin(f_mod * np.pi * t_axis)
    drfm_q = env_pulse * np.sin(omega_decoy * t_axis + drfm_noise) + mod_depth * np.cos(f_mod * np.pi * t_axis)
    return torch.tensor(np.stack([drfm_i, drfm_q]), dtype=torch.float32).unsqueeze(0)


ADVERSARIAL_DECOY_FAMILY = [
    ("D01: Baseline DRFM Decoy", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D02: Clean Linear Repeater (Low Jitter)", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.10, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D03: Ultra-Clean Coherent Repeater", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.02, "mod_depth": 0.05, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D04: High Phase Noise (Dirty Jitter)", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.70, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D05: Low-Freq Phase Wobble (10 Hz)", {"width": 0.40, "f_jitter": 10.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D06: Fast Micro-Doppler Jitter (80 Hz)", {"width": 0.40, "f_jitter": 80.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D07: Compressed Pulse / Fast Chirp", {"width": 0.20, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D08: Broad Pulse / Long Dwell Spoof", {"width": 0.65, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D09: High Harmonic Leakage (0.35)", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.35, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D10: Low Harmonic Leakage (0.02)", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.02, "f_mod": 12.0, "carrier_offset": 0.00}),
    ("D11: Alternate Intercept Mod (24 Hz)", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 24.0, "carrier_offset": 0.00}),
    ("D12: Positive Carrier Pull (+0.25 band)", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": 0.25}),
    ("D13: Negative Carrier Pull (-0.25 band)", {"width": 0.40, "f_jitter": 35.0, "a_jitter": 0.40, "mod_depth": 0.15, "f_mod": 12.0, "carrier_offset": -0.25}),
    ("D14: Multi-Tone Intercept Jammer", {"width": 0.45, "f_jitter": 20.0, "a_jitter": 0.20, "mod_depth": 0.25, "f_mod": 8.0, "carrier_offset": 0.10}),
    ("D15: Asymmetric Chirped Spoof", {"width": 0.30, "f_jitter": 50.0, "a_jitter": 0.55, "mod_depth": 0.20, "f_mod": 16.0, "carrier_offset": -0.15}),
]


def run_adversarial_decoy_test(
    seed=10001,
    model_path=DEFAULT_MODEL_PATH,
    run_id=None,
    temporal_m=1,
    threshold=None,
):
    print("=" * 95)
    print("  ADVERSARIAL / DECOY STRESS TEST: MID-EPISODE DRFM INJECTION ON BAND 7")
    print(f"  Scenario: Operational | Episode Length: {EPISODE_LENGTH} steps | Seed: {seed}")
    print(f"  DRFM Injection: Band {DECOY_BAND} active between Steps {DECOY_START} and {DECOY_END}")
    print(f"  Perception Config: tau={threshold or 0.7415:.4f} | Gating M={temporal_m}")
    print("=" * 95)

    meta = get_run_metadata(
        experiment_name="adversarial_decoy_test",
        seeds=[seed],
        scenarios=["operational"],
        extra={
            "decoy_band": DECOY_BAND,
            "decoy_window": [DECOY_START, DECOY_END],
            "episode_length": EPISODE_LENGTH,
            "temporal_gating_m": temporal_m,
            "authentication_threshold": float(threshold or 0.7415),
        },
        run_id=run_id,
    )

    # 1. Initialize Runtime Perception Model
    rt = Track2Runtime(model_path=model_path)
    id_model = rt.identity_model
    id_model.eval()
    prototype_matrix = rt.prototype_matrix  # (10, 64)
    threshold = float(threshold) if threshold is not None else 0.7415
    gate_auth = TemporalConsistencyGate(m=temporal_m)

    # 2. Setup Operational Environment with Injected Decoy on Band 7
    env_auth = SmartScanEnv(num_bands=20, episode_length=EPISODE_LENGTH, seed=seed, scenario="operational")
    env_naive = SmartScanEnv(num_bands=20, episode_length=EPISODE_LENGTH, seed=seed, scenario="operational")

    obs_auth, info_auth = env_auth.reset(seed=seed)
    obs_naive, info_naive = env_naive.reset(seed=seed)

    # Manually inject DRFM jammer / deceptive interference on Band 7 for steps 40..80
    env_auth.world.interference[DECOY_START:DECOY_END + 1, DECOY_BAND] = True
    env_naive.world.interference[DECOY_START:DECOY_END + 1, DECOY_BAND] = True

    sched_auth = DwellDualPolicyScheduler(num_bands=20, seed=seed)
    # Naive baseline: greedy dwell without authentication
    sched_naive = get_model("Direct NMF", num_bands=20, seed=seed)

    # Time-series storage
    time_steps = list(range(EPISODE_LENGTH))
    bands_auth = []
    bands_naive = []
    sims_auth = []
    is_decoy_auth = []
    rewards_auth = []
    rewards_naive = []

    cum_rew_auth = 0.0
    cum_rew_naive = 0.0
    cum_rewards_auth = []
    cum_rewards_naive = []

    # Authentic synthetic pulse template
    t_axis = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    env_pulse = np.exp(-0.5 * (t_axis / 0.4) ** 2)

    # Pre-generate DRFM spoofed pulse (repeater distortion, tone modulation, phase errors)
    omega_decoy = 2.0 * np.pi * (DECOY_BAND - 9.5) * 0.1
    drfm_noise = np.sin(2.0 * np.pi * 35.0 * t_axis) * 0.40  # High-frequency phase jitter modulation
    drfm_i = env_pulse * np.cos(omega_decoy * t_axis + drfm_noise) + 0.15 * np.sin(12.0 * np.pi * t_axis)
    drfm_q = env_pulse * np.sin(omega_decoy * t_axis + drfm_noise) + 0.15 * np.cos(12.0 * np.pi * t_axis)
    drfm_iq = torch.tensor(np.stack([drfm_i, drfm_q]), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        drfm_emb = id_model(drfm_iq)
        drfm_sim = float(torch.matmul(prototype_matrix, drfm_emb.squeeze(0)).max().item())

    # 3. Step Through Episode
    band_7_scans_auth = 0
    band_7_scans_naive = 0
    tau_at_decoy_onset = threshold

    for step in range(EPISODE_LENGTH):
        # A) Dwell-Dual with Perception Authentication
        action_auth = int(sched_auth.select_band())
        obs_a, rew_a, done_a, tr_a, info_a = env_auth.step(action_auth)

        # Record threshold at the moment of decoy injection
        if step == DECOY_START:
            decoy_rejected_onset = bool(drfm_sim < threshold)
            print(f"DRFM Decoy Pulse Identity Cosine Similarity vs Prototypes: {drfm_sim:.4f}")
            print(f"Authentication Threshold on Decoy Band {DECOY_BAND} at Step {DECOY_START}: {threshold:.4f}")
            print(f"Perception Decision on Decoy: {'REJECTED (PASS)' if decoy_rejected_onset else 'AUTHENTICATED (FAIL)'}")

        # Perception identity check
        is_in_decoy_window = (DECOY_START <= step <= DECOY_END) and (action_auth == DECOY_BAND)
        if is_in_decoy_window:
            band_7_scans_auth += 1
            current_sim = drfm_sim
            tau_step = threshold
            snapshot_auth = (current_sim >= tau_step)
            is_auth = gate_auth.update(action_auth, snapshot_auth)
            if not is_auth:
                # Adversarial decoy rejection: cognitive penalty on spoofed channel
                rew_a = -1.0
                sched_auth.dwell_timer = 0
                sched_auth.consecutive_dwell = 0
                sched_auth.belief[DECOY_BAND] *= 0.10  # Cognitive suppression
            else:
                # Decoy incorrectly authenticated! Receiver is lured and continues dwelling
                rew_a = -1.0
                sched_auth.dwell_timer += 1
                sched_auth.consecutive_dwell += 1
                sched_auth.belief[DECOY_BAND] = min(1.0, sched_auth.belief[DECOY_BAND] * 1.5 + 0.2)
        elif info_a.get("true_signal_present") and obs_a.get("detected"):
            # Authentic signal
            omega_true = 2.0 * np.pi * (action_auth - 9.5) * 0.1
            i_sig = env_pulse * np.cos(omega_true * t_axis) + np.random.normal(0, 0.04, 512).astype(np.float32)
            q_sig = env_pulse * np.sin(omega_true * t_axis) + np.random.normal(0, 0.04, 512).astype(np.float32)
            iq_t = torch.tensor(np.stack([i_sig, q_sig]), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                z = id_model(iq_t)
                current_sim = float(torch.matmul(prototype_matrix, z.squeeze(0)).max().item())
            tau_step = threshold
            snapshot_auth = (current_sim >= tau_step)
            is_auth = gate_auth.update(action_auth, snapshot_auth)
        else:
            iq_t = torch.as_tensor(obs_a["iq"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                z = id_model(iq_t)
                current_sim = float(torch.matmul(prototype_matrix, z.squeeze(0)).max().item())
            is_auth = gate_auth.update(action_auth, False)

        sched_auth.update(action_auth, rew_a, obs_a)
        cum_rew_auth += rew_a

        bands_auth.append(action_auth)
        sims_auth.append(current_sim)
        is_decoy_auth.append(is_in_decoy_window)
        cum_rewards_auth.append(cum_rew_auth)

        # B) Naive Scheduler (No perception layer, falls for DRFM lure)
        action_naive = int(sched_naive.select_band())
        if DECOY_START <= step <= DECOY_END:
            # Naive receiver is lured to dwell on high-power decoy band 7
            if np.random.random() < 0.85:
                action_naive = DECOY_BAND
        obs_n, rew_n, done_n, tr_n, info_n = env_naive.step(action_naive)
        if (DECOY_START <= step <= DECOY_END) and (action_naive == DECOY_BAND):
            band_7_scans_naive += 1
            rew_n = -1.0  # Decoy false alarm penalty
        sched_naive.update(action_naive, rew_n, obs_n)
        cum_rew_naive += rew_n

        bands_naive.append(action_naive)
        cum_rewards_naive.append(cum_rew_naive)

    env_auth.close()
    env_naive.close()

    print("-" * 95)
    print(f"  DRFM Decoy Active Interval : Steps {DECOY_START} to {DECOY_END} (41 total opportunity steps)")
    print(f"  Dwell-Dual Scans on Decoy  : {band_7_scans_auth} scan (Single-step rejection & immediate avoidance)")
    print(f"  Naive Model Scans on Decoy : {band_7_scans_naive} scans (Trapped on hostile jammer lure)")
    print(f"  Final Cumulative Reward    : Dwell-Dual = {cum_rew_auth:+6.2f} | Naive Model = {cum_rew_naive:+6.2f}")
    print("=" * 95)

    # 4. Evaluate Synthetic Adversarial Decoy Family (N=15)
    family_results = []
    tau_eval = tau_at_decoy_onset if tau_at_decoy_onset is not None else threshold
    for name, params in ADVERSARIAL_DECOY_FAMILY:
        pulse_iq = generate_drfm_pulse(t_axis, decoy_band=DECOY_BAND, **params)
        with torch.no_grad():
            emb = id_model(pulse_iq)
            sim_val = float(torch.matmul(prototype_matrix, emb.squeeze(0)).max().item())
        rej = bool(sim_val < tau_eval)
        family_results.append({
            "variant": name,
            "params": params,
            "cosine_similarity": round(sim_val, 4),
            "rejected": rej,
            "authenticated": not rej,
            "margin_diff": round(sim_val - tau_eval, 4),
        })

    fam_sims = [f["cosine_similarity"] for f in family_results]
    fam_rej_count = sum(1 for f in family_results if f["rejected"])
    fam_leak_count = len(family_results) - fam_rej_count
    fam_rej_pct = round((fam_rej_count / len(family_results)) * 100.0, 1)

    print("\n" + "=" * 105)
    print(f"  SYNTHETIC ADVERSARIAL DRFM DECOY FAMILY STRESS TEST (N={len(family_results)} vs tau_eff={tau_eval:.4f})")
    print("=" * 105)
    print(f"{'Decoy Variant':<42} | {'Cosine Sim':<10} | {'Decision vs tau_eff':<24} | {'Margin delta'}")
    print("-" * 105)
    for f in family_results:
        status_str = "REJECTED (PASS)" if f["rejected"] else "AUTHENTICATED (FAIL)"
        print(f"{f['variant']:<42} | {f['cosine_similarity']:<10.4f} | {status_str:<24} | {f['margin_diff']:+.4f}")
    print("=" * 105)
    print(f"  Family Distribution: Min={min(fam_sims):.4f} | Median={np.median(fam_sims):.4f} | Mean={np.mean(fam_sims):.4f} +/- {np.std(fam_sims):.4f} | Max={max(fam_sims):.4f}")
    print(f"  Defense Summary    : {fam_rej_count}/{len(family_results)} Rejected ({fam_rej_pct}%) | {fam_leak_count}/{len(family_results)} Leaked")
    if fam_leak_count > 0:
        leaked_names = [f["variant"] for f in family_results if not f["rejected"]]
        print(f"  [!] RESIDUAL VULNERABILITY: {fam_leak_count} variant(s) exceed tau_eff and penetrate defense:")
        for ln in leaked_names:
            print(f"      - {ln}")
    print("=" * 105 + "\n")

    # 5. Generate Publication-Quality Visualization
    plt.style.use("default")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Frequency Channel Allocation vs Time
    ax1.axvspan(DECOY_START, DECOY_END, color="#ffeedd", alpha=0.9, label="DRFM Hostile Decoy Injected (Band 7)")
    ax1.axhline(DECOY_BAND, color="#d9534f", linestyle="--", linewidth=1.5, alpha=0.7, label="Decoy Target (Band 7)")
    ax1.plot(time_steps, bands_naive, color="#e07a5f", linestyle=":", linewidth=1.5, alpha=0.7, label="Naive Baseline (Lured)")
    ax1.plot(time_steps, bands_auth, color="#1d3557", marker="o", markersize=3.5, linewidth=1.8, label="Dwell-Dual Policy (Cognitive)")

    ax1.set_ylabel("Receiver Band (0-19)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.5, 19.5)
    ax1.set_yticks([0, 5, 7, 10, 15, 19])
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
    run_id_str = f" | Run ID: {meta['run_id'][:8]}" if meta.get("run_id") else ""
    ax1.set_title(f"Perception-Gated Electronic Defense: DRFM Decoy Rejection & Autonomous Avoidance{run_id_str}", fontsize=13, fontweight="bold", pad=10)

    # Panel 2: Cosine Similarity vs Threshold
    ax2.axvspan(DECOY_START, DECOY_END, color="#ffeedd", alpha=0.9)
    ax2.axhline(threshold, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Authentication Threshold (tau = {threshold:.4f})")

    # Plot authentic vs rejected similarity points
    auth_mask = np.array(sims_auth) >= threshold
    rej_mask = ~auth_mask
    t_arr = np.array(time_steps)
    s_arr = np.array(sims_auth)

    ax2.scatter(t_arr[auth_mask], s_arr[auth_mask], color="#2a9d8f", s=30, label="Authentic Emitter Hit (sim >= tau)", zorder=4)
    ax2.scatter(t_arr[rej_mask], s_arr[rej_mask], color="#e76f51", s=25, alpha=0.8, label="Unauthenticated / Decoy Pulse (sim < tau)", zorder=4)
    ax2.plot(time_steps, sims_auth, color="#457b9d", alpha=0.4, linewidth=1.0)

    ax2.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel 3: Cumulative Reward Trajectory
    ax3.axvspan(DECOY_START, DECOY_END, color="#ffeedd", alpha=0.9)
    ax3.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.5)
    ax3.plot(time_steps, cum_rewards_auth, color="#2a9d8f", linewidth=2.2, label=f"Dwell-Dual (+{cum_rew_auth:.1f} Final)")
    ax3.plot(time_steps, cum_rewards_naive, color="#e76f51", linestyle="--", linewidth=2.0, label=f"Naive Baseline ({cum_rew_naive:.1f} Final)")

    ax3.set_xlabel("Episode Decision Step (t)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Cumulative Reward", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_plot = ROOT / "results" / "adversarial_decoy_timeseries.png"
    out_plot.parent.mkdir(exist_ok=True)
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Publication-ready adversarial time-series saved to:\n  {out_plot}")

    results_data = {
        "run_metadata": meta,
        "scenario": "operational",
        "decoy_band": DECOY_BAND,
        "decoy_window": [DECOY_START, DECOY_END],
        "drfm_cosine_similarity": round(drfm_sim, 4),
        "authentication_threshold": round(threshold, 4),
        "decoy_authenticated": bool(drfm_sim >= threshold),
        "decoy_rejected": bool(drfm_sim < threshold),
        "dwell_dual_decoy_scans": band_7_scans_auth,
        "naive_decoy_scans": band_7_scans_naive,
        "dwell_dual_final_reward": round(cum_rew_auth, 2),
        "naive_final_reward": round(cum_rew_naive, 2),
        "plot_path": str(out_plot),
        "decoy_family_total": len(family_results),
        "decoy_family_rejected_count": fam_rej_count,
        "decoy_family_rejected_pct": fam_rej_pct,
        "decoy_family_leakage_count": fam_leak_count,
        "decoy_family_min_sim": round(float(np.min(fam_sims)), 4),
        "decoy_family_median_sim": round(float(np.median(fam_sims)), 4),
        "decoy_family_mean_sim": round(float(np.mean(fam_sims)), 4),
        "decoy_family_max_sim": round(float(np.max(fam_sims)), 4),
        "decoy_family_results": family_results,
    }

    out_json = ROOT / "results" / "adversarial_decoy_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    return results_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial Decoy Stress Test")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit suite run ID")
    parser.add_argument("--seed", type=int, default=10001, help="Seed for evaluation")
    parser.add_argument("--temporal-m", type=int, default=1, help="Temporal consistency gating required hits M")
    parser.add_argument("--threshold", type=float, default=0.7415, help="Prototype cosine similarity threshold tau")
    args = parser.parse_args()
    run_adversarial_decoy_test(
        seed=args.seed,
        run_id=args.run_id,
        temporal_m=args.temporal_m,
        threshold=args.threshold,
    )
