"""Presentation & Publication Quality Charts Generator for SmartScan EW Results.

Generates 4 slide-ready high-resolution (300 DPI) visual charts:
1. results/scenario_performance_breakdown.png: 6-scenario grouped bar chart comparing Dwell-Dual vs top baselines
2. results/sensitivity_curve_analysis.png: Pd vs SNR curve (-10 dB to +25 dB) with MDS threshold (+0.0 dB)
3. results/perception_latency_benchmark.png: Isolated 1D-CNN perception latency vs 2.0 ms real-time deadline
4. results/master_leaderboard_14models.png: 14-model master benchmark leaderboard with Tier 1/Tier 2 statistical grouping
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
RESULTS_DIR = ROOT / "results"


def plot_scenario_breakdown():
    json_path = RESULTS_DIR / "master_15seed_statistical_benchmark.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leaderboard = data["leaderboard"]
    run_meta = data.get("run_metadata", {})
    git_hash = run_meta.get("git_commit_hash", "") or run_meta.get("git_commit", "")
    git_hash = git_hash[:7] if git_hash else ""

    scenarios = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
    scenario_labels = ["Stationary\n(Radar)", "Hopping\n(Agile)", "Changing\nPattern", "Harsh Noise\n& Jamming", "Operational\nEW Stress", "Crowded\nBattlespace"]

    # Select representative models to highlight
    models_to_plot = [
        ("Dwell-Dual Policy (Champion)", "#1d3557", "Dwell-Dual (Champion)"),
        ("Dual-Policy Uncertainty", "#457b9d", "Dual-Policy Uncertainty"),
        ("Static NMF+UCB", "#2a9d8f", "Static NMF+UCB"),
        ("Candidate 1: LinUCB Bandit", "#e76f51", "LinUCB Bandit"),
        ("Whittle Index RMAB", "#8c532b", "Whittle Index RMAB"),
        ("Fixed Sequential Sweep", "#9ca3af", "Fixed Raster Sweep"),
    ]

    x = np.arange(len(scenarios))
    width = 0.13

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#faf8f5")

    for idx, (m_key, color, label) in enumerate(models_to_plot):
        if m_key in leaderboard:
            means = [leaderboard[m_key]["scenario_stats"][sc]["mean"] for sc in scenarios]
            offset = (idx - len(models_to_plot) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, label=label, color=color, edgecolor="#292524", linewidth=0.8, alpha=0.95, zorder=3)

    ax.axhline(0, color="#78350f", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)
    ax.set_ylabel("Mean Cumulative Reward (15-Seed Evaluation)", fontsize=11, fontweight="bold")
    title_suffix = f" | Git: {git_hash}" if git_hash else ""
    ax.set_title(f"Tactical Performance Across 6 Electronic Warfare Scenarios (15 Seeds, 1,260 Episodes){title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=1)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5, facecolor="#ffffff", edgecolor="#78350f")

    plt.tight_layout()
    out_file = RESULTS_DIR / "scenario_performance_breakdown.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_sensitivity_curve():
    json_path = RESULTS_DIR / "figures_of_merit_results.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sens = data["sensitivity"]
    curve = sens["curve"]
    mds_snr = sens["mds_threshold_snr_db"]
    run_meta = data.get("run_metadata", {})
    git_hash = run_meta.get("git_commit_hash", "") or run_meta.get("git_commit", "")
    git_hash = git_hash[:7] if git_hash else ""

    snrs = [pt["snr_db"] for pt in curve]
    sensor_pds = [pt["sensor_pd_pct"] for pt in curve]
    opp_pds = [pt["opportunity_pd_pct"] for pt in curve]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#faf8f5")

    ax.axhline(50.0, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7, label="Minimum Operational Threshold (Pd = 50%)")
    if mds_snr is not None:
        ax.axvline(mds_snr, color="#b45309", linestyle=":", linewidth=1.8, label=f"MDS Threshold: {mds_snr:+.1f} dB")

    ax.plot(snrs, sensor_pds, marker="o", markersize=6, linewidth=2.4, color="#1d3557", label="Physical Receiver Sensor Pd (%)")
    ax.plot(snrs, opp_pds, marker="s", markersize=5, linewidth=2.0, color="#2a9d8f", linestyle="-.", label="Cognitive ESM Intercept Opportunity Pd (%)")

    ax.set_xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability of Detection (Pd %)", fontsize=11, fontweight="bold")
    title_suffix = f" | Git: {git_hash}" if git_hash else ""
    ax.set_title(f"ESM Sensitivity Curve: Detection Probability vs. Environmental SNR (-10 dB to +25 dB){title_suffix}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-11, 26)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10, facecolor="#ffffff", edgecolor="#78350f")

    # Annotations
    ax.annotate("Linear Dynamic\nOperating Range", xy=(15, 88), xytext=(12, 60),
                arrowprops=dict(facecolor='#1d3557', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight="bold", color="#1d3557")

    plt.tight_layout()
    out_file = RESULTS_DIR / "sensitivity_curve_analysis.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_perception_latency():
    json_path = RESULTS_DIR / "perception_layer_latency.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stages = [
        "1D-CNN Feature\nExtraction (IQ)",
        "Prototype Cosine\nMatching (10 Classes)",
        "Total Isolated\nPerception Pipeline"
    ]
    means_us = [
        data["cnn_encoder_us"]["mean"],
        data["prototype_matching_us"]["mean"],
        data["total_perception_us"]["mean"]
    ]
    p95_us = [
        data["cnn_encoder_us"]["p95"],
        data["prototype_matching_us"]["p95"],
        data["total_perception_us"]["p95"]
    ]
    medians_us = [
        data["cnn_encoder_us"]["median_p50"],
        data["prototype_matching_us"]["median_p50"],
        data["total_perception_us"]["median_p50"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#faf8f5")

    x = np.arange(len(stages))
    width = 0.26

    r1 = ax.bar(x - width, medians_us, width, label="Median (P50)", color="#457b9d", edgecolor="#1d3557", alpha=0.9)
    r2 = ax.bar(x, means_us, width, label="Mean Latency", color="#2a9d8f", edgecolor="#14532d", alpha=0.9)
    r3 = ax.bar(x + width, p95_us, width, label="P95 Tail Latency", color="#e76f51", edgecolor="#991b1b", alpha=0.9)

    ax.axhline(2000.0, color="#dc2626", linestyle="--", linewidth=1.8, label="Real-Time Control Loop Deadline (2.0 ms / 2,000 us)")

    ax.set_ylabel("Execution Time (Microseconds, us)", fontsize=11, fontweight="bold")
    run_meta = data.get("run_metadata", {})
    git_hash = run_meta.get("git_commit_hash", "") or run_meta.get("git_commit", "")
    git_hash = git_hash[:7] if git_hash else ""
    title_suffix = f" (Git: {git_hash})" if git_hash else ""
    ax.set_title(f"Perception-Layer Latency in Isolation (CPU Benchmark, N=5,000 Trials, Warmup Corrected){title_suffix}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9.5, facecolor="#ffffff", edgecolor="#78350f")

    # Add text labels on bars
    for bars in [r1, r2, r3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f} us",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_file = RESULTS_DIR / "perception_latency_benchmark.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_master_leaderboard():
    json_path = RESULTS_DIR / "master_15seed_statistical_benchmark.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leaderboard = data["leaderboard"]
    # Sorted ascending so top performers are plotted at the top of horizontal bar chart
    sorted_models = sorted(leaderboard.items(), key=lambda x: x[1]["mean_reward"])

    names = [k for k, _ in sorted_models]
    rewards = [v["mean_reward"] for _, v in sorted_models]
    intercepts = [v["interception_rate_pct"] for _, v in sorted_models]

    # Calculate asymmetric error bars based on 95% CI
    err_low = []
    err_high = []
    for r, (_, v) in zip(rewards, sorted_models):
        ci = v.get("reward_95ci")
        if ci and len(ci) == 2:
            err_low.append(max(0.0, r - ci[0]))
            err_high.append(max(0.0, ci[1] - r))
        else:
            std = v.get("std_reward", 0.0)
            err_low.append(std)
            err_high.append(std)
    xerr = np.array([err_low, err_high])

    # Distinct Tier 1 vs Tier 2 coloring
    colors = []
    edgecolors = []
    for name, v in sorted_models:
        tier = v.get("tier", 2)
        if "Champion" in name:
            colors.append("#1d3557")       # Dark Navy for Champion (Tier 1)
            edgecolors.append("#0f172a")
        elif tier == 1:
            colors.append("#3b82f6")       # Vibrant Blue for Tier 1 peers (statistically tied)
            edgecolors.append("#1d4ed8")
        elif "Candidate" in name:
            colors.append("#2a9d8f")       # Teal for Candidate baselines
            edgecolors.append("#14532d")
        elif "NMF" in name or "PCA" in name or "Whittle" in name:
            colors.append("#e76f51")       # Terracotta for Classic EW baselines
            edgecolors.append("#991b1b")
        else:
            colors.append("#9ca3af")       # Gray for naive baselines
            edgecolors.append("#475569")

    fig, ax = plt.subplots(figsize=(14.5, 9.0))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#faf8f5")

    y = np.arange(len(names))
    bars = ax.barh(y, rewards, xerr=xerr, capsize=3.5, color=colors, edgecolor=edgecolors, linewidth=1.0, alpha=0.92, height=0.68)

    run_meta = data.get("run_metadata", {})
    seeds_list = run_meta.get("seeds", [])
    n_seeds = len(seeds_list) if seeds_list else 15
    df_seeds = n_seeds - 1
    total_episodes = n_seeds * 6 * len(sorted_models)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="bold")
    ax.set_xlabel(f"Mean Cumulative Reward (Error bars: 95% Confidence Interval across {n_seeds} seeds)", fontsize=11, fontweight="bold")

    # Tier demarcation line
    tier1_indices = [idx for idx, (_, v) in enumerate(sorted_models) if v.get("tier", 2) == 1]
    tier1_count = len(tier1_indices)

    if tier1_indices:
        split_y = min(tier1_indices) - 0.5
        ax.axhline(split_y, color="#b45309", linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)
        if tier1_count == 1:
            ax.text(-57.0, split_y + 0.18, "TIER 1 (Statistically Dominant Champion, p < 0.05 vs All Baselines)",
                    fontsize=9.0, fontweight="bold", color="#1e3a8a", va="bottom")
            ax.text(-57.0, split_y - 0.18, "TIER 2 (Statistically Significant Separation, p < 0.05)",
                    fontsize=9.0, fontweight="bold", color="#b45309", va="top")
        else:
            ax.text(-57.0, split_y + 0.18, f"TIER 1 ({tier1_count} Models Statistically Indistinguishable, p >= 0.05)",
                    fontsize=9.0, fontweight="bold", color="#1e3a8a", va="bottom")
            ax.text(-57.0, split_y - 0.18, "TIER 2 (Statistically Significant Separation, p < 0.05)",
                    fontsize=9.0, fontweight="bold", color="#b45309", va="top")

    if tier1_count == 1:
        subtitle_text = f"Dwell-Dual Policy statistically outperforms all other evaluated approaches at 95% confidence ({n_seeds} seeds)"
    else:
        subtitle_text = f"Models within Tier 1 are not statistically distinguishable from the top performer at 95% confidence ({n_seeds} seeds)"

    ax.set_title(
        f"Master 14-Model Leaderboard with Statistical Tiers (df={df_seeds}, {n_seeds} Seeds, {total_episodes:,} Episodes)\n{subtitle_text}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Set axis range to comfortably host annotations
    max_rew = max(rewards)
    ax.set_xlim(left=-60, right=max_rew + 78)

    for idx, (bar, rew, (name, v)) in enumerate(zip(bars, rewards, sorted_models)):
        ci = v.get("reward_95ci")
        intercept = v.get("interception_rate_pct", 0.0)
        tier = v.get("tier", 2)
        p_val = v.get("p_value_vs_dwell_dual")
        is_champ = "Champion" in name

        if is_champ:
            tier_tag = " [Tier 1: Dominant Champion]" if tier1_count == 1 else " [Tier 1: Champion]"
        elif tier == 1:
            tier_tag = f" [Tier 1 Tied, p={p_val:.3f}]" if p_val is not None else " [Tier 1 Tied]"
        else:
            tier_tag = ""

        if ci and len(ci) == 2:
            label_txt = f"{rew:+5.1f} (95% CI: [{ci[0]:+.1f}, {ci[1]:+.1f}]) | Hit: {intercept:4.1f}%{tier_tag}"
            text_x = ci[1] + 1.5
        else:
            std = v.get("std_reward", 0.0)
            label_txt = f"{rew:+5.1f} | Hit: {intercept:4.1f}%{tier_tag}"
            text_x = rew + std + 1.5

        ax.text(text_x, bar.get_y() + bar.get_height() / 2, label_txt,
                va='center', ha='left', fontsize=8.0, fontweight="bold", color="#1c1917")

    plt.tight_layout()
    out_file = RESULTS_DIR / "master_leaderboard_14models.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


if __name__ == "__main__":
    plot_scenario_breakdown()
    plot_sensitivity_curve()
    plot_perception_latency()
    plot_master_leaderboard()
