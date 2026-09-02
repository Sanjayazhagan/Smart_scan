"""Direct, offline diagnostics for the frozen Track 2 world model.

The scan order is fixed and identical for every run. Simulator truth is used
only to score forecasts and identity association; it is never passed to Track 2.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _binary_auc(targets: list[int], scores: list[float]) -> float | None:
    target = np.asarray(targets, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    positive = score[target == 1]
    negative = score[target == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def _forecast_metrics(targets: list[int], scores: list[float]) -> dict:
    target = np.asarray(targets, dtype=np.float64)
    score = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    positive = target == 1.0
    negative = ~positive
    return {
        "samples": int(target.size),
        "positive_rate": float(target.mean()) if target.size else 0.0,
        "constant_prevalence_brier": float(target.mean() * (1.0 - target.mean()))
        if target.size
        else None,
        "brier_score": float(np.mean((score - target) ** 2)) if target.size else None,
        "auc": _binary_auc(targets, scores),
        "mean_probability_on_positive": float(score[positive].mean())
        if np.any(positive)
        else None,
        "mean_probability_on_negative": float(score[negative].mean())
        if np.any(negative)
        else None,
    }


def _true_emitter_for_detection(info: dict, selected_band: int) -> int | None:
    emitters = list(info.get("ground_truth_active_emitters", []))
    bands = list(info.get("ground_truth_active_bands", []))
    matches = [
        index for index, band in enumerate(bands) if int(band) == int(selected_band)
    ]
    if not matches:
        return None
    snr = list(info.get("ground_truth_active_snr_db", []))
    if len(snr) == len(emitters):
        best = max(matches, key=lambda index: float(snr[index]))
    else:
        best = matches[0]
    return int(emitters[best])


def diagnose_episode(
    scenario: str,
    seed: int,
    model_path: str | Path,
    dataset_path: str | Path,
    warmup_steps: int = 80,
    scored_steps: int = 200,
) -> dict:
    from scheduler.track2_core import NUM_BANDS
    from scheduler.track2_runtime import Track2Runtime
    from simulator.environment import SmartScanEnv

    total_steps = int(warmup_steps + scored_steps)
    env = SmartScanEnv(
        num_bands=NUM_BANDS,
        episode_length=total_steps,
        seed=seed,
        iq_dataset_path=dataset_path,
        scenario=scenario,
    )
    env.reset(seed=seed)
    runtime = Track2Runtime(model_path, max_scan_age=total_steps)

    # Beta(1,1) causal history baseline; it sees the same fixed scans as Track 2.
    history_hits = np.zeros(NUM_BANDS, dtype=np.float64)
    history_trials = np.zeros(NUM_BANDS, dtype=np.float64)
    truth_targets: list[int] = []
    observation_targets: list[int] = []
    track2_scores: list[float] = []
    history_scores: list[float] = []
    gru_probabilities: list[float] = []
    gru_top1: list[int] = []
    gru_entropies: list[float] = []
    assignments: dict[str, Counter] = defaultdict(Counter)
    emitter_tracks: dict[int, set[str]] = defaultdict(set)
    true_detection_updates = 0
    false_alarm_updates = 0

    for step in range(total_steps):
        selected_band = int(step % NUM_BANDS)
        track2_probability = float(runtime.get_band_belief()[selected_band])
        history_probability = float(
            (history_hits[selected_band] + 1.0)
            / (history_trials[selected_band] + 2.0)
        )
        observation, _, _, truncated, info = env.step(selected_band)
        detected = bool(observation["detected"])
        signal_present = bool(info.get("true_signal_present", False))
        result = runtime.update(observation, timestamp=float(step))

        history_trials[selected_band] += 1.0
        history_hits[selected_band] += float(detected)

        if step >= warmup_steps:
            truth_targets.append(int(signal_present))
            observation_targets.append(int(detected))
            track2_scores.append(track2_probability)
            history_scores.append(history_probability)

        if result is not None:
            if signal_present and detected:
                true_emitter = _true_emitter_for_detection(info, selected_band)
                if true_emitter is not None:
                    track_id = str(result["track_id"])
                    assignments[track_id][true_emitter] += 1
                    emitter_tracks[true_emitter].add(track_id)
                    true_detection_updates += 1
                previous = result.get("previous_next_probabilities")
                if previous is not None:
                    probabilities = np.asarray(previous, dtype=np.float64)
                    probability = float(
                        np.clip(probabilities[selected_band], 1e-9, 1.0)
                    )
                    gru_probabilities.append(probability)
                    gru_top1.append(int(np.argmax(probabilities) == selected_band))
                    entropy = float(
                        -np.sum(
                            probabilities
                            * np.log(np.clip(probabilities, 1e-12, 1.0))
                        )
                        / np.log(probabilities.size)
                    )
                    gru_entropies.append(entropy)
            elif detected:
                false_alarm_updates += 1

        if truncated and step + 1 != total_steps:
            raise RuntimeError("Environment ended before the requested diagnostic")

    assigned_total = sum(sum(counts.values()) for counts in assignments.values())
    correct_within_track = sum(max(counts.values()) for counts in assignments.values())
    confirmed_tracks = sum(
        int(track.confirmed) for track in runtime.manager.tracks.values()
    )
    return {
        "scenario": scenario,
        "seed": int(seed),
        "warmup_steps": int(warmup_steps),
        "scored_steps": int(scored_steps),
        "track2_vs_truth": _forecast_metrics(truth_targets, track2_scores),
        "history_vs_truth": _forecast_metrics(truth_targets, history_scores),
        "track2_vs_observed_detection": _forecast_metrics(
            observation_targets, track2_scores
        ),
        "history_vs_observed_detection": _forecast_metrics(
            observation_targets, history_scores
        ),
        "gru_transition_samples": int(len(gru_probabilities)),
        "gru_mean_probability_on_observed_next_band": float(
            np.mean(gru_probabilities)
        )
        if gru_probabilities
        else None,
        "gru_next_band_top1_accuracy": float(np.mean(gru_top1))
        if gru_top1
        else None,
        "gru_mean_normalized_entropy": float(np.mean(gru_entropies))
        if gru_entropies
        else None,
        "identity_assignment_purity": float(correct_within_track / assigned_total)
        if assigned_total
        else None,
        "mean_tracks_per_observed_emitter": float(
            np.mean([len(track_ids) for track_ids in emitter_tracks.values()])
        )
        if emitter_tracks
        else None,
        "true_detection_track_updates": int(true_detection_updates),
        "false_alarm_track_updates": int(false_alarm_updates),
        "total_tracks": int(len(runtime.manager.tracks)),
        "confirmed_tracks": int(confirmed_tracks),
    }


def _mean_optional(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _weighted_optional(
    values: list[float | None], weights: list[int]
) -> float | None:
    pairs = [
        (float(value), int(weight))
        for value, weight in zip(values, weights, strict=True)
        if value is not None and weight > 0
    ]
    if not pairs:
        return None
    return float(
        sum(value * weight for value, weight in pairs)
        / sum(weight for _, weight in pairs)
    )


def aggregate_episodes(episodes: list[dict]) -> dict:
    metric_paths = {
        "track2_truth_brier": ("track2_vs_truth", "brier_score"),
        "history_truth_brier": ("history_vs_truth", "brier_score"),
        "constant_truth_brier": (
            "track2_vs_truth",
            "constant_prevalence_brier",
        ),
        "track2_truth_auc": ("track2_vs_truth", "auc"),
        "history_truth_auc": ("history_vs_truth", "auc"),
        "track2_observed_brier": ("track2_vs_observed_detection", "brier_score"),
        "history_observed_brier": ("history_vs_observed_detection", "brier_score"),
    }
    output = {
        name: _mean_optional([episode[parent][child] for episode in episodes])
        for name, (parent, child) in metric_paths.items()
    }
    transition_weights = [episode["gru_transition_samples"] for episode in episodes]
    for name in (
        "gru_mean_probability_on_observed_next_band",
        "gru_next_band_top1_accuracy",
        "gru_mean_normalized_entropy",
    ):
        output[name] = _weighted_optional(
            [episode[name] for episode in episodes], transition_weights
        )
    output["identity_assignment_purity"] = _weighted_optional(
        [episode["identity_assignment_purity"] for episode in episodes],
        [episode["true_detection_track_updates"] for episode in episodes],
    )
    for name in (
        "mean_tracks_per_observed_emitter",
        "total_tracks",
        "confirmed_tracks",
    ):
        output[name] = _mean_optional([episode[name] for episode in episodes])
    output["episodes"] = int(len(episodes))
    output["gru_transition_samples"] = int(
        sum(episode["gru_transition_samples"] for episode in episodes)
    )
    output["true_detection_track_updates"] = int(
        sum(episode["true_detection_track_updates"] for episode in episodes)
    )
    output["false_alarm_track_updates"] = int(
        sum(episode["false_alarm_track_updates"] for episode in episodes)
    )
    return output


def _main():
    import argparse
    import json

    from scheduler.track2_runtime import DEFAULT_MODEL_PATH
    from simulator.environment import DEFAULT_IQ_DATASET_PATH
    from simulator.scenarios import scenario_names

    parser = argparse.ArgumentParser(description="Diagnose frozen Track 2 forecasts")
    parser.add_argument("--track2-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_IQ_DATASET_PATH))
    parser.add_argument(
        "--scenarios", nargs="+", choices=scenario_names(), required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--scored-steps", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()

    per_scenario = {}
    for scenario in args.scenarios:
        episodes = [
            diagnose_episode(
                scenario,
                seed,
                args.track2_model,
                args.dataset,
                args.warmup_steps,
                args.scored_steps,
            )
            for seed in args.seeds
        ]
        per_scenario[scenario] = {
            "per_seed": {str(item["seed"]): item for item in episodes},
            "aggregate": aggregate_episodes(episodes),
        }
    output = {"scenarios": per_scenario}
    rendered = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    _main()
