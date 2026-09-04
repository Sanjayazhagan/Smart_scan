from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable, Dict

import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _inject_evaluation_oracle(env, scheduler) -> None:
    """Inject imminent hidden activity only into an evaluation-only scheduler."""
    setter = getattr(scheduler, "set_oracle_activity", None)
    if not callable(setter):
        return
    if not bool(getattr(scheduler, "evaluation_only", False)):
        raise RuntimeError("A non-evaluation scheduler requested hidden oracle truth")
    base_env = env.unwrapped
    world = getattr(base_env, "world", None)
    if world is None:
        raise ValueError(
            "Oracle-UCB requires a pre-generated deterministic scenario, not legacy"
        )
    step = int(base_env.current_step)
    activity = np.zeros(base_env.num_bands, dtype=np.float32)
    for emitter in world.emitters:
        if emitter.is_active(step):
            activity[emitter.get_band(step)] = 1.0
    setter(activity)


def evaluate_policy(
    env_fn: Callable,
    scheduler_fn: Callable,
    seed: int,
    episodes: int = 5,
    warmup_steps: int = 0,
) -> Dict[str, float]:
    """Evaluate after an optional shared sequential calibration phase.

    Warm-up observations update the scheduler but are excluded from the main
    metrics. The environment is not reset between warm-up and scoring.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    env = env_fn(seed=seed)
    episode_warmup_rewards = []
    episode_warmup_true_positives = []
    episode_warmup_false_positives = []
    episode_warmup_misses = []
    episode_calibration_enough_evidence = []
    episode_calibration_noisy_scores = []
    episode_calibration_clean_scores = []
    episode_calibration_hit_concentrations = []
    episode_calibrated_noisy_overrides = []
    episode_rewards = []
    episode_true_positives = []
    episode_false_positives = []
    episode_misses = []
    episode_unique_bands = []
    all_revisit_intervals = []
    all_planning_times = []
    all_update_times = []
    all_discovery_delays = []
    all_behaviour_change_delays = []
    all_effective_snr_db = []
    total_steps = 0
    total_wasted_scans = 0
    total_active_opportunities = 0
    total_discovered_emitters = 0
    total_observed_emitters = 0
    total_switch_distance = 0.0
    total_switching_cost = 0.0
    total_switches = 0
    total_sensor_dropouts = 0
    total_interference_observations = 0
    total_negative_scans = 0
    total_deceptive_false_alarms = 0
    total_threat_active_opportunities = 0.0
    total_threat_interceptions = 0.0
    total_late_emitters = 0
    total_late_emitters_discovered = 0
    all_late_emitter_discovery_delays = []
    regime_counts = {
        "easy": 0,
        "intermediate": 0,
        "complex": 0,
        "dynamic_unreliable": 0,
        "unknown_ood": 0,
    }
    adaptive_exploration_count = 0
    calibrated_noisy_override_count = 0
    expert_counts = {
        "fast": 0,
        "beam": 0,
        "mpp": 0,
        "tree": 0,
        "ucb": 0,
        "rl": 0,
        "safe": 0,
        "nmf": 0,
        "rpca": 0,
    }
    dynamic_signal_counts = {
        "prediction_error": 0,
        "behaviour_change": 0,
        "low_quality_uncertainty": 0,
        "disagreement_uncertainty": 0,
        "no_confirmed_support": 0,
    }
    monitor_metric_sums = {
        "mean_volatility": 0.0,
        "low_quality_detection_ratio": 0.0,
        "mean_detected_quality": 0.0,
        "credible_detection_ratio": 0.0,
        "clean_stationary_score": 0.0,
        "noisy_score": 0.0,
        "outcome_flip_rate": 0.0,
        "hit_band_concentration": 0.0,
    }
    history_regime_counts = {
        "uncertain": 0,
        "clean_stationary": 0,
        "noisy": 0,
        "agile_hopping": 0,
    }

    for episode in range(episodes):
        _, _ = env.reset(seed=seed + episode)
        scheduler = scheduler_fn(env.unwrapped.num_bands)
        warmup_reward = 0.0
        warmup_true_positives = 0
        warmup_false_positives = 0
        warmup_misses = 0
        warmup_terminated = warmup_truncated = False
        for warmup_index in range(warmup_steps):
            warmup_action = int(warmup_index % env.unwrapped.num_bands)
            warmup_obs, reward, warmup_terminated, warmup_truncated, warmup_info = (
                env.step(warmup_action)
            )
            scheduler.update(warmup_action, reward, warmup_obs)
            signal_present = bool(warmup_info.get("true_signal_present", False))
            detected = bool(warmup_obs["detected"])
            if signal_present and detected:
                warmup_true_positives += 1
            elif signal_present:
                warmup_misses += 1
            elif detected:
                warmup_false_positives += 1
            warmup_reward += float(reward)
            if warmup_terminated or warmup_truncated:
                raise RuntimeError(
                    "Environment ended during warm-up; construct it with "
                    "episode_length = warmup_steps + scored episode length"
                )
        episode_warmup_rewards.append(warmup_reward)
        episode_warmup_true_positives.append(warmup_true_positives)
        episode_warmup_false_positives.append(warmup_false_positives)
        episode_warmup_misses.append(warmup_misses)
        begin_scored_phase = getattr(scheduler, "begin_scored_phase", None)
        calibration_trace = {}
        if callable(begin_scored_phase):
            calibration_trace = begin_scored_phase() or {}
        episode_calibration_enough_evidence.append(
            float(bool(calibration_trace.get("enough_evidence", False)))
        )
        episode_calibration_noisy_scores.append(
            float(calibration_trace.get("noisy_score", 0.0))
        )
        episode_calibration_clean_scores.append(
            float(calibration_trace.get("clean_stationary_score", 0.0))
        )
        episode_calibration_hit_concentrations.append(
            float(calibration_trace.get("hit_band_concentration", 0.0))
        )
        episode_calibrated_noisy_overrides.append(
            float(
                calibration_trace.get("selected_override")
                in {"noisy_mpp", "noisy_tree", "harsh_ucb"}
            )
        )
        episode_reward = 0.0
        true_positives = 0
        false_positives = 0
        misses = 0
        actions = []
        last_scan_step = {}
        first_active_step = {}
        first_detection_step = {}
        pending_behaviour_changes = {}
        emitter_start_steps = {}
        terminated = truncated = False
        step_index = 0

        while not (terminated or truncated):
            _inject_evaluation_oracle(env, scheduler)
            planning_started = perf_counter()
            action = int(scheduler.select_band())
            all_planning_times.append(perf_counter() - planning_started)
            decision_trace = getattr(scheduler, "last_decision_trace", {}) or {}
            regime = decision_trace.get("regime")
            if regime in regime_counts:
                regime_counts[regime] += 1
            expert_name = decision_trace.get("expert")
            if expert_name in expert_counts:
                expert_counts[expert_name] += 1
            calibrated_noisy_override_count += int(
                decision_trace.get("routing_source")
                in {
                    "calibrated_noisy_mpp_guard",
                    "calibrated_noisy_tree_guard",
                    "calibrated_harsh_ucb_guard",
                }
            )
            router_trace = decision_trace.get("router", {}) or {}
            dynamic_signals = router_trace.get("dynamic_signals", {}) or {}
            for signal_name in dynamic_signal_counts:
                if signal_name == "no_confirmed_support":
                    active = router_trace.get(signal_name, False)
                else:
                    active = dynamic_signals.get(signal_name, False)
                dynamic_signal_counts[signal_name] += int(bool(active))
            monitor_trace = decision_trace.get("regime_monitor", {}) or {}
            for metric_name in monitor_metric_sums:
                monitor_metric_sums[metric_name] += float(
                    monitor_trace.get(metric_name, 0.0)
                )
            history_regime = monitor_trace.get("history_regime", "uncertain")
            if history_regime in history_regime_counts:
                history_regime_counts[history_regime] += 1
            exploration_trace = decision_trace.get("adaptive_exploration", {}) or {}
            adaptive_exploration_count += int(
                bool(exploration_trace.get("explored"))
                or decision_trace.get("selection_mode")
                == "exploratory_alternative"
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            update_started = perf_counter()
            scheduler.update(action, reward, next_obs)
            all_update_times.append(perf_counter() - update_started)

            actions.append(action)
            if action in last_scan_step:
                all_revisit_intervals.append(step_index - last_scan_step[action])
            last_scan_step[action] = step_index

            active_emitters = list(info.get("ground_truth_active_emitters", []))
            active_bands = list(info.get("ground_truth_active_bands", []))
            active_threat_weights = list(
                info.get(
                    "ground_truth_active_threat_weights",
                    [1.0] * len(active_emitters),
                )
            )
            emitter_start_steps.update(
                {
                    int(emitter_id): int(start_step)
                    for emitter_id, start_step in info.get(
                        "ground_truth_emitter_start_steps", {}
                    ).items()
                }
            )
            for emitter_id in info.get("ground_truth_behaviour_changes", []):
                pending_behaviour_changes[int(emitter_id)] = step_index
            total_switch_distance += float(info.get("switch_distance", 0.0))
            total_switching_cost += float(info.get("switching_cost", 0.0))
            total_switches += int(float(info.get("switch_distance", 0.0)) > 0.0)
            total_sensor_dropouts += int(bool(info.get("sensor_dropout", False)))
            total_interference_observations += int(
                bool(info.get("interference_present", False))
            )
            total_deceptive_false_alarms += int(
                bool(info.get("deceptive_false_alarm", False))
            )
            if info.get("effective_snr_db") is not None:
                all_effective_snr_db.append(float(info["effective_snr_db"]))
            total_active_opportunities += len(active_emitters)
            total_threat_active_opportunities += float(sum(active_threat_weights))
            for emitter_id in active_emitters:
                first_active_step.setdefault(int(emitter_id), step_index)

            signal_present = bool(info.get("true_signal_present", False))
            detected = bool(next_obs["detected"])
            if signal_present and detected:
                true_positives += 1
                for emitter_id, band, threat_weight in zip(
                    active_emitters,
                    active_bands,
                    active_threat_weights,
                    strict=True,
                ):
                    if int(band) == action:
                        total_threat_interceptions += float(threat_weight)
                        first_detection_step.setdefault(int(emitter_id), step_index)
                        emitter_id = int(emitter_id)
                        if emitter_id in pending_behaviour_changes:
                            all_behaviour_change_delays.append(
                                step_index - pending_behaviour_changes.pop(emitter_id)
                            )
            elif signal_present:
                misses += 1
            elif detected:
                false_positives += 1

            if not signal_present:
                total_wasted_scans += 1
                total_negative_scans += 1
            episode_reward += reward
            total_steps += 1
            step_index += 1

        observed_ids = set(first_active_step)
        discovered_ids = observed_ids.intersection(first_detection_step)
        total_observed_emitters += len(observed_ids)
        total_discovered_emitters += len(discovered_ids)
        all_discovery_delays.extend(
            first_detection_step[emitter_id] - first_active_step[emitter_id]
            for emitter_id in discovered_ids
        )
        late_emitters = {
            emitter_id
            for emitter_id in observed_ids
            if emitter_start_steps.get(emitter_id, 0) > 0
        }
        late_discovered = late_emitters.intersection(discovered_ids)
        total_late_emitters += len(late_emitters)
        total_late_emitters_discovered += len(late_discovered)
        all_late_emitter_discovery_delays.extend(
            first_detection_step[emitter_id] - first_active_step[emitter_id]
            for emitter_id in late_discovered
        )
        episode_rewards.append(episode_reward)
        episode_true_positives.append(true_positives)
        episode_false_positives.append(false_positives)
        episode_misses.append(misses)
        episode_unique_bands.append(len(set(actions)))

    env.close()
    total_true_positives = int(sum(episode_true_positives))
    total_false_positives = int(sum(episode_false_positives))
    total_misses = int(sum(episode_misses))
    return {
        "warmup_steps": int(warmup_steps),
        "scored_steps": int(total_steps),
        "mean_warmup_reward": float(np.mean(episode_warmup_rewards)),
        "mean_warmup_reward_per_step": (
            float(np.mean(episode_warmup_rewards) / warmup_steps)
            if warmup_steps
            else 0.0
        ),
        "mean_warmup_true_positives": float(
            np.mean(episode_warmup_true_positives)
        ),
        "mean_warmup_false_positives": float(
            np.mean(episode_warmup_false_positives)
        ),
        "mean_warmup_misses": float(np.mean(episode_warmup_misses)),
        "calibration_enough_evidence_ratio": float(
            np.mean(episode_calibration_enough_evidence)
        ),
        "calibration_noisy_score": float(
            np.mean(episode_calibration_noisy_scores)
        ),
        "calibration_clean_stationary_score": float(
            np.mean(episode_calibration_clean_scores)
        ),
        "calibration_hit_band_concentration": float(
            np.mean(episode_calibration_hit_concentrations)
        ),
        "calibrated_noisy_episode_ratio": float(
            np.mean(episode_calibrated_noisy_overrides)
        ),
        "mean_total_reward_including_warmup": float(
            np.mean(
                np.asarray(episode_rewards, dtype=np.float64)
                + np.asarray(episode_warmup_rewards, dtype=np.float64)
            )
        ),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "mean_true_positives": float(np.mean(episode_true_positives)),
        "mean_false_positives": float(np.mean(episode_false_positives)),
        "mean_misses": float(np.mean(episode_misses)),
        "detection_rate": _safe_ratio(
            total_true_positives, total_true_positives + total_misses
        ),
        "interception_rate": _safe_ratio(
            total_true_positives, total_active_opportunities
        ),
        "false_alarm_rate": _safe_ratio(
            total_false_positives, total_negative_scans
        ),
        "threat_weighted_interception_rate": _safe_ratio(
            total_threat_interceptions, total_threat_active_opportunities
        ),
        "wasted_scan_ratio": _safe_ratio(total_wasted_scans, total_steps),
        "mean_unique_bands_scanned": float(np.mean(episode_unique_bands)),
        "average_band_revisit_time": float(np.mean(all_revisit_intervals))
        if all_revisit_intervals
        else 0.0,
        "average_planning_time_ms": float(np.mean(all_planning_times) * 1000.0),
        "average_update_time_ms": float(np.mean(all_update_times) * 1000.0),
        "average_control_time_ms": float(
            (np.mean(all_planning_times) + np.mean(all_update_times)) * 1000.0
        ),
        "mean_new_emitter_discovery_delay": float(np.mean(all_discovery_delays))
        if all_discovery_delays
        else 0.0,
        "emitter_discovery_rate": _safe_ratio(
            total_discovered_emitters, total_observed_emitters
        ),
        "late_emitter_discovery_rate": _safe_ratio(
            total_late_emitters_discovered, total_late_emitters
        ),
        "mean_late_emitter_discovery_delay": (
            float(np.mean(all_late_emitter_discovery_delays))
            if all_late_emitter_discovery_delays
            else 0.0
        ),
        "behaviour_change_adaptation_delay": float(
            np.mean(all_behaviour_change_delays)
        )
        if all_behaviour_change_delays
        else None,
        "mean_effective_snr_db": float(np.mean(all_effective_snr_db))
        if all_effective_snr_db
        else None,
        "switch_rate": _safe_ratio(total_switches, total_steps),
        "mean_switch_distance": _safe_ratio(total_switch_distance, total_steps),
        "mean_switching_cost": _safe_ratio(total_switching_cost, total_steps),
        "sensor_dropout_ratio": _safe_ratio(total_sensor_dropouts, total_steps),
        "observed_interference_ratio": _safe_ratio(
            total_interference_observations, total_steps
        ),
        "deceptive_false_alarm_ratio": _safe_ratio(
            total_deceptive_false_alarms, total_steps
        ),
        "easy_route_ratio": _safe_ratio(regime_counts["easy"], total_steps),
        "intermediate_route_ratio": _safe_ratio(
            regime_counts["intermediate"], total_steps
        ),
        "complex_route_ratio": _safe_ratio(regime_counts["complex"], total_steps),
        "dynamic_route_ratio": _safe_ratio(
            regime_counts["dynamic_unreliable"], total_steps
        ),
        "unknown_ood_route_ratio": _safe_ratio(
            regime_counts["unknown_ood"], total_steps
        ),
        "adaptive_exploration_ratio": _safe_ratio(
            adaptive_exploration_count, total_steps
        ),
        "calibrated_noisy_override_ratio": _safe_ratio(
            calibrated_noisy_override_count, total_steps
        ),
        "fast_expert_ratio": _safe_ratio(expert_counts["fast"], total_steps),
        "beam_expert_ratio": _safe_ratio(expert_counts["beam"], total_steps),
        "mpp_expert_ratio": _safe_ratio(expert_counts["mpp"], total_steps),
        "tree_expert_ratio": _safe_ratio(expert_counts["tree"], total_steps),
        "ucb_expert_ratio": _safe_ratio(expert_counts["ucb"], total_steps),
        "rl_expert_ratio": _safe_ratio(expert_counts["rl"], total_steps),
        "safe_expert_ratio": _safe_ratio(expert_counts["safe"], total_steps),
        "nmf_expert_ratio": _safe_ratio(expert_counts["nmf"], total_steps),
        "rpca_expert_ratio": _safe_ratio(expert_counts["rpca"], total_steps),
        "prediction_error_signal_ratio": _safe_ratio(
            dynamic_signal_counts["prediction_error"], total_steps
        ),
        "behaviour_change_signal_ratio": _safe_ratio(
            dynamic_signal_counts["behaviour_change"], total_steps
        ),
        "low_quality_uncertainty_signal_ratio": _safe_ratio(
            dynamic_signal_counts["low_quality_uncertainty"], total_steps
        ),
        "disagreement_uncertainty_signal_ratio": _safe_ratio(
            dynamic_signal_counts["disagreement_uncertainty"], total_steps
        ),
        "no_confirmed_support_ratio": _safe_ratio(
            dynamic_signal_counts["no_confirmed_support"], total_steps
        ),
        "online_volatility": _safe_ratio(
            monitor_metric_sums["mean_volatility"], total_steps
        ),
        "online_low_quality_detection_ratio": _safe_ratio(
            monitor_metric_sums["low_quality_detection_ratio"], total_steps
        ),
        "online_mean_detected_quality": _safe_ratio(
            monitor_metric_sums["mean_detected_quality"], total_steps
        ),
        "online_credible_detection_ratio": _safe_ratio(
            monitor_metric_sums["credible_detection_ratio"], total_steps
        ),
        "history_uncertain_ratio": _safe_ratio(
            history_regime_counts["uncertain"], total_steps
        ),
        "history_clean_stationary_ratio": _safe_ratio(
            history_regime_counts["clean_stationary"], total_steps
        ),
        "history_noisy_ratio": _safe_ratio(
            history_regime_counts["noisy"], total_steps
        ),
        "history_agile_hopping_ratio": _safe_ratio(
            history_regime_counts["agile_hopping"], total_steps
        ),
        "history_clean_stationary_score": _safe_ratio(
            monitor_metric_sums["clean_stationary_score"], total_steps
        ),
        "history_noisy_score": _safe_ratio(
            monitor_metric_sums["noisy_score"], total_steps
        ),
        "history_outcome_flip_rate": _safe_ratio(
            monitor_metric_sums["outcome_flip_rate"], total_steps
        ),
        "history_hit_band_concentration": _safe_ratio(
            monitor_metric_sums["hit_band_concentration"], total_steps
        ),
    }


def run_benchmark(
    env_fn: Callable,
    schedulers: dict,
    seed: int = 42,
    episodes: int = 5,
    warmup_steps: int = 0,
) -> dict:
    return {
        name: evaluate_policy(
            env_fn,
            scheduler_fn,
            seed,
            episodes,
            warmup_steps=warmup_steps,
        )
        for name, scheduler_fn in schedulers.items()
    }


def aggregate_seed_results(per_seed: dict) -> dict:
    scheduler_names = next(iter(per_seed.values())).keys()
    aggregate = {}
    for scheduler_name in scheduler_names:
        metric_names = next(iter(per_seed.values()))[scheduler_name].keys()
        aggregate[scheduler_name] = {}
        for metric_name in metric_names:
            values = [
                seed_results[scheduler_name][metric_name]
                for seed_results in per_seed.values()
                if seed_results[scheduler_name][metric_name] is not None
            ]
            aggregate[scheduler_name][metric_name] = (
                float(np.mean(values)) if values else None
            )
        aggregate[scheduler_name]["across_seed_std_mean_reward"] = float(
            np.std(
                [
                    seed_results[scheduler_name]["mean_reward"]
                    for seed_results in per_seed.values()
                ]
            )
        )
    return aggregate


def build_full_scheduler_suite(
    frequency_policy_path=None,
    track2_policy_path=None,
    track2_model_path=None,
    track2_exploration_policy_path=None,
    track2_attention_policy_path=None,
    router_model_path=None,
    path_value_model_path=None,
    seed: int = 42,
    max_scan_age: float = 200.0,
):
    """Build baselines, planners, adaptive MoE, and optional learned experts."""
    from stable_baselines3 import PPO

    from scheduler.baselines import (
        FixedScheduler,
        RandomScheduler,
        ThompsonSamplingScheduler,
        UCB1Scheduler,
    )
    from scheduler.adaptive_moe import (
        AdaptiveMixtureOfExpertsScheduler,
        LearnedRegimeRouter,
        ObservableDiscountedUCBScheduler,
    )
    from scheduler.paradigms import (
        RobustPCAPSRScheduler,
        NMFScheduler,
        WhittleIndexRMABScheduler,
        Exp3BanditScheduler,
        AdaptiveReceiverSearchScheduler,
        DoubleDQNScheduler,
    )
    from scheduler.async_worker import AsyncPlanningScheduler
    from scheduler.belief_tree import ObservationDependentBeliefTreePlanner
    from scheduler.emitter_aware_predictive import (
        BeliefOnlyModelPredictiveScheduler,
        EmitterAwareModelPredictivePlanner,
        EmitterAwarePlannerConfig,
    )
    from scheduler.emitter_rl import (
        AttentionRLScheduler,
        BeliefRLScheduler,
        ExplorationBeliefRLScheduler,
    )
    from scheduler.frequency_rl import RLScheduler
    from scheduler.learned_value import LearnedValueEmitterPlanner, SmallPathValueModel
    from scheduler.model_predictive import ModelPredictiveScheduler
    from scheduler.track2_runtime import Track2Runtime
    from scheduler.world_model_ucb import WorldModelUCBScheduler
    from scheduler.world_model_nmf_ucb import WorldModelNMFUCBScheduler
    from scheduler.world_model_aux_ucb import WorldModelAuxUCBScheduler
    from scheduler.lean_observable_moe import LeanObservableMoEScheduler
    from scheduler.calibrated_soft_moe import (
        CalibratedSoftMoEScheduler,
        CalibratedWorldNMFUCBScheduler,
    )
    from evaluation.oracle_ucb import OracleGuidedUCBScheduler

    if track2_model_path is None:
        from scheduler.track2_runtime import DEFAULT_MODEL_PATH

        track2_model_path = DEFAULT_MODEL_PATH

    frequency_model = (
        PPO.load(str(frequency_policy_path))
        if frequency_policy_path is not None
        else None
    )
    track2_model = (
        PPO.load(str(track2_policy_path))
        if track2_policy_path is not None
        else None
    )
    full_emitter_config = EmitterAwarePlannerConfig(
        exploration_probability=0.0, exploration_seed=seed
    )
    no_anomaly_config = replace(
        full_emitter_config, behaviour_weight=0.0, novelty_weight=0.0
    )
    no_identity_recency_config = replace(
        full_emitter_config, track_recency_weight=0.0, identity_weight=0.0
    )
    adaptive_exploration_config = replace(
        no_identity_recency_config,
        exploration_probability=0.05,
        exploration_seed=seed,
    )
    track2_exploration_model = (
        PPO.load(str(track2_exploration_policy_path))
        if track2_exploration_policy_path is not None
        else None
    )
    track2_attention_model = (
        PPO.load(str(track2_attention_policy_path))
        if track2_attention_policy_path is not None
        else None
    )
    learned_router = (
        LearnedRegimeRouter.load(router_model_path)
        if router_model_path is not None
        else None
    )
    schedulers = {
        "Fixed / sequential": lambda n: FixedScheduler(n),
        "Random": lambda n: RandomScheduler(n, seed=seed),
        "Thompson Sampling": lambda n: ThompsonSamplingScheduler(n, seed=seed),
        "UCB": lambda n: UCB1Scheduler(n),
        "Double DQN": lambda n: DoubleDQNScheduler(n, seed=seed),
        "Robust PCA + PSR": lambda n: RobustPCAPSRScheduler(n, seed=seed),
        "Non-Negative Matrix Factorization": lambda n: NMFScheduler(n, seed=seed),
        "Adaptive Receiver Search": lambda n: AdaptiveReceiverSearchScheduler(n, seed=seed),
        "Whittle Index RMAB": lambda n: WhittleIndexRMABScheduler(n, seed=seed),
        "Adversarial Exp3 Bandit": lambda n: Exp3BanditScheduler(n, seed=seed),
        "Observable Discounted UCB": lambda n: ObservableDiscountedUCBScheduler(
            n,
            Track2Runtime(track2_model_path, max_scan_age=max_scan_age),
            neural_guidance_scale=0.0,
            manage_runtime=True,
        ),
        "Neural-Augmented UCB": lambda n: ObservableDiscountedUCBScheduler(
            n,
            Track2Runtime(track2_model_path, max_scan_age=max_scan_age),
            neural_guidance_scale=0.65,
            manage_runtime=True,
        ),
        "World-Model UCB": lambda n: WorldModelUCBScheduler(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
        ),
        "World+NMF UCB[0.25]": lambda n: WorldModelNMFUCBScheduler(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            nmf_scale=0.25,
        ),
        "World+NMF UCB[0.50]": lambda n: WorldModelNMFUCBScheduler(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            nmf_scale=0.50,
        ),
        "World+NMF UCB[1.00]": lambda n: WorldModelNMFUCBScheduler(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            nmf_scale=1.00,
        ),
        "World+RPCA UCB[0.50]": lambda n: WorldModelAuxUCBScheduler(
            n, auxiliary="rpca", auxiliary_scale=0.50, seed=seed,
            model_path=track2_model_path, max_scan_age=max_scan_age,
        ),
        "World+RPCA UCB[1.00]": lambda n: WorldModelAuxUCBScheduler(
            n, auxiliary="rpca", auxiliary_scale=1.00, seed=seed,
            model_path=track2_model_path, max_scan_age=max_scan_age,
        ),
        "World+PRI UCB[0.50]": lambda n: WorldModelAuxUCBScheduler(
            n, auxiliary="pri", auxiliary_scale=0.50, seed=seed,
            model_path=track2_model_path, max_scan_age=max_scan_age,
        ),
        "World+PRI UCB[1.00]": lambda n: WorldModelAuxUCBScheduler(
            n, auxiliary="pri", auxiliary_scale=1.00, seed=seed,
            model_path=track2_model_path, max_scan_age=max_scan_age,
        ),
        "World+Exp3 UCB[0.50]": lambda n: WorldModelAuxUCBScheduler(
            n, auxiliary="exp3", auxiliary_scale=0.50, seed=seed,
            model_path=track2_model_path, max_scan_age=max_scan_age,
        ),
        "World+Exp3 UCB[1.00]": lambda n: WorldModelAuxUCBScheduler(
            n, auxiliary="exp3", auxiliary_scale=1.00, seed=seed,
            model_path=track2_model_path, max_scan_age=max_scan_age,
        ),
        "Lean Observable MoE": lambda n: LeanObservableMoEScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age, seed=seed
        ),
        "NMF+UCB[world off]": lambda n: WorldModelNMFUCBScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age,
            nmf_scale=1.0, world_model_scale=0.0,
        ),
        "NMF+UCB[world calibrated]": lambda n: CalibratedWorldNMFUCBScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age,
            nmf_scale=1.0, world_model_scale=0.35,
        ),
        "Calibrated Soft MoE[T0.35]": lambda n: CalibratedSoftMoEScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age, seed=seed,
            router_temperature=0.35,
        ),
        "Calibrated Soft MoE[T0.55]": lambda n: CalibratedSoftMoEScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age, seed=seed,
            router_temperature=0.55,
        ),
        "Calibrated Soft MoE[T0.85]": lambda n: CalibratedSoftMoEScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age, seed=seed,
            router_temperature=0.85,
        ),
        "World-Model UCB[off]": lambda n: WorldModelUCBScheduler(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            world_model_scale=0.0,
        ),
        "Oracle-UCB[0.10]": lambda n: OracleGuidedUCBScheduler(
            n, oracle_scale=0.10, model_path=track2_model_path, max_scan_age=max_scan_age
        ),
        "Oracle-UCB[0.25]": lambda n: OracleGuidedUCBScheduler(
            n, oracle_scale=0.25, model_path=track2_model_path, max_scan_age=max_scan_age
        ),
        "Oracle-UCB[0.50]": lambda n: OracleGuidedUCBScheduler(
            n, oracle_scale=0.50, model_path=track2_model_path, max_scan_age=max_scan_age
        ),
        "Oracle-UCB[1.00]": lambda n: OracleGuidedUCBScheduler(
            n, oracle_scale=1.00, model_path=track2_model_path, max_scan_age=max_scan_age
        ),
        "Oracle-UCB[2.00]": lambda n: OracleGuidedUCBScheduler(
            n, oracle_scale=2.00, model_path=track2_model_path, max_scan_age=max_scan_age
        ),
        "MPP-BeliefOnly": lambda n: BeliefOnlyModelPredictiveScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age
        ),
        "MPP-60": lambda n: ModelPredictiveScheduler(
            n, model_path=track2_model_path, max_scan_age=max_scan_age
        ),
        "MPP-EmitterAware": lambda n: EmitterAwareModelPredictivePlanner(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            config=full_emitter_config,
        ),
        "MPP-EmitterAware minus anomaly": lambda n: EmitterAwareModelPredictivePlanner(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            config=no_anomaly_config,
        ),
        "MPP-EmitterAware minus identity/recency": lambda n: EmitterAwareModelPredictivePlanner(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            config=no_identity_recency_config,
        ),
        "MPP-EmitterAware adaptive exploration": lambda n: EmitterAwareModelPredictivePlanner(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            config=adaptive_exploration_config,
        ),
        "MPP-ObservationDependent Tree": lambda n: ObservationDependentBeliefTreePlanner(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            config=no_identity_recency_config,
        ),
        "Adaptive MoE rule router": lambda n: AdaptiveMixtureOfExpertsScheduler(
            n,
            rl_model=track2_exploration_model,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
        ),
        "Async Adaptive MoE": lambda n: AsyncPlanningScheduler(
            n,
            planner_factory=lambda: AdaptiveMixtureOfExpertsScheduler(
                n,
                rl_model=track2_exploration_model,
                model_path=track2_model_path,
                max_scan_age=max_scan_age,
            ),
        ),
    }
    if frequency_model is not None:
        schedulers["Frequency-History PPO"] = lambda n: RLScheduler(
            n, frequency_model
        )
    if track2_model is not None:
        schedulers["Track2 PPO[20]"] = lambda n: BeliefRLScheduler(
            n, track2_model, model_path=track2_model_path
        )
    if track2_exploration_model is not None:
        schedulers["Track2 PPO[60]"] = lambda n: ExplorationBeliefRLScheduler(
            n,
            track2_exploration_model,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
        )
    if track2_attention_model is not None:
        schedulers["Track2 PPO[EmitterAttention]"] = lambda n: AttentionRLScheduler(
            n,
            track2_attention_model,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
        )
        schedulers["Adaptive MoE attention expert"] = lambda n: AdaptiveMixtureOfExpertsScheduler(
            n,
            attention_rl_model=track2_attention_model,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
        )
    if path_value_model_path is not None:
        path_value_model = SmallPathValueModel.load(path_value_model_path)
        schedulers["MPP-LearnedPathValue"] = lambda n: LearnedValueEmitterPlanner(
            n,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            config=no_identity_recency_config,
            value_model=path_value_model,
        )
    if learned_router is not None:
        schedulers["Adaptive MoE learned gate"] = lambda n: AdaptiveMixtureOfExpertsScheduler(
            n,
            rl_model=track2_exploration_model,
            model_path=track2_model_path,
            max_scan_age=max_scan_age,
            learned_router=learned_router,
        )
    return schedulers


def _main():
    import argparse
    import json
    from pathlib import Path

    from scheduler.track2_core import NUM_BANDS
    from scheduler.track2_runtime import DEFAULT_MODEL_PATH
    from simulator.environment import DEFAULT_IQ_DATASET_PATH, SmartScanEnv
    from simulator.scenarios import scenario_names

    parser = argparse.ArgumentParser(description="Benchmark all Smart Scan schedulers")
    parser.add_argument("--frequency-policy")
    parser.add_argument("--track2-policy")
    parser.add_argument("--track2-exploration-policy")
    parser.add_argument("--track2-attention-policy")
    parser.add_argument("--router-model")
    parser.add_argument("--path-value-model")
    parser.add_argument("--track2-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_IQ_DATASET_PATH))
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default="legacy",
        help="Hidden-world scenario preset; legacy reproduces the original simulator",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=scenario_names(),
        help="Run a scenario matrix; overrides --scenario",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=200,
        help="Number of scored steps after optional warm-up",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Unscored sequential calibration scans before each scored episode; "
            "the environment and scheduler state are not reset"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only these exact scheduler names (use quotes for names with spaces)",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.episode_length < 1:
        parser.error("--episode-length must be positive")

    seeds = args.seeds or [args.seed]
    requested_scenarios = args.scenarios or [args.scenario]
    total_episode_length = args.warmup_steps + args.episode_length
    scenario_outputs = {}
    for scenario_name in requested_scenarios:
        def env_fn(seed, selected_scenario=scenario_name):
            return SmartScanEnv(
                num_bands=NUM_BANDS,
                episode_length=total_episode_length,
                seed=seed,
                iq_dataset_path=args.dataset,
                scenario=selected_scenario,
            )

        per_seed = {}
        for benchmark_seed in seeds:
            schedulers = build_full_scheduler_suite(
                args.frequency_policy,
                args.track2_policy,
                args.track2_model,
                track2_exploration_policy_path=args.track2_exploration_policy,
                track2_attention_policy_path=args.track2_attention_policy,
                router_model_path=args.router_model,
                path_value_model_path=args.path_value_model,
                seed=benchmark_seed,
                max_scan_age=total_episode_length,
            )
            if args.only:
                unknown = set(args.only).difference(schedulers)
                if unknown:
                    raise ValueError(f"Unknown scheduler names: {sorted(unknown)}")
                schedulers = {name: schedulers[name] for name in args.only}
            per_seed[str(benchmark_seed)] = run_benchmark(
                env_fn,
                schedulers,
                benchmark_seed,
                args.episodes,
                warmup_steps=args.warmup_steps,
            )
        scenario_outputs[scenario_name] = (
            next(iter(per_seed.values()))
            if len(per_seed) == 1
            else {"per_seed": per_seed, "aggregate": aggregate_seed_results(per_seed)}
        )

    output = (
        next(iter(scenario_outputs.values()))
        if len(scenario_outputs) == 1
        else {"scenarios": scenario_outputs}
    )
    rendered = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    _main()
