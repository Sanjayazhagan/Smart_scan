import pytest
from simulator.environment import SmartScanEnv
from scheduler.baselines import FixedScheduler, RandomScheduler
from evaluation.benchmark import evaluate_policy, run_benchmark
from evaluation.oracle_ucb import OracleGuidedUCBScheduler

def test_evaluate_policy_determinism():
    """Verify that identical seeds produce identical evaluation metrics."""
    def env_fn(seed):
        return SmartScanEnv(num_bands=5, episode_length=50, seed=seed)
        
    def scheduler_fn(num_bands):
        # We fix the internal seed of the RandomScheduler for identical outputs
        return RandomScheduler(num_bands=num_bands, seed=42)
        
    res1 = evaluate_policy(env_fn, scheduler_fn, seed=100, episodes=2)
    res2 = evaluate_policy(env_fn, scheduler_fn, seed=100, episodes=2)
    
    assert res1["mean_reward"] == res2["mean_reward"]
    assert res1["mean_true_positives"] == res2["mean_true_positives"]
    assert "median_reward" in res1
    assert "mean_false_positives" in res1
    assert "mean_misses" in res1
    assert "detection_rate" in res1
    assert "wasted_scan_ratio" in res1
    assert "mean_unique_bands_scanned" in res1
    assert "average_band_revisit_time" in res1
    assert "average_planning_time_ms" in res1
    assert "adaptive_exploration_ratio" in res1
    assert "beam_expert_ratio" in res1
    assert "switch_rate" in res1
    assert "sensor_dropout_ratio" in res1
    assert "observed_interference_ratio" in res1

def test_run_benchmark():
    """Verify that the benchmarking suite iterates and collates metrics correctly."""
    def env_fn(seed):
        return SmartScanEnv(num_bands=3, episode_length=20, seed=seed)
        
    schedulers = {
        "Fixed": lambda n: FixedScheduler(n),
        "Random": lambda n: RandomScheduler(n, seed=42)
    }
    
    results = run_benchmark(env_fn, schedulers, seed=123, episodes=1)
    
    assert "Fixed" in results
    assert "Random" in results
    assert "mean_reward" in results["Fixed"]
    assert "mean_true_positives" in results["Random"]


def test_evaluate_policy_keeps_state_across_unscored_warmup():
    def env_fn(seed):
        return SmartScanEnv(num_bands=5, episode_length=30, seed=seed)

    result = evaluate_policy(
        env_fn,
        lambda n: FixedScheduler(n),
        seed=321,
        episodes=2,
        warmup_steps=10,
    )

    assert result["warmup_steps"] == 10
    assert result["scored_steps"] == 40
    assert result["mean_total_reward_including_warmup"] == pytest.approx(
        result["mean_warmup_reward"] + result["mean_reward"]
    )
    assert result["mean_warmup_reward_per_step"] == pytest.approx(
        result["mean_warmup_reward"] / 10
    )
    assert result["calibration_enough_evidence_ratio"] == 0.0
    assert result["calibration_noisy_score"] == 0.0
    assert result["calibrated_noisy_episode_ratio"] == 0.0


def test_oracle_ucb_receives_imminent_activity_only_in_benchmark():
    def env_fn(seed):
        return SmartScanEnv(
            num_bands=20,
            episode_length=45,
            seed=seed,
            scenario="stationary",
        )

    result = evaluate_policy(
        env_fn,
        lambda n: OracleGuidedUCBScheduler(n, oracle_scale=1.0),
        seed=919,
        episodes=1,
        warmup_steps=20,
    )

    assert result["scored_steps"] == 25
    assert 0 <= result["mean_true_positives"] <= 25


def test_oracle_ucb_refuses_to_run_without_benchmark_truth():
    scheduler = OracleGuidedUCBScheduler(20)
    with pytest.raises(RuntimeError, match="not injected"):
        scheduler.select_band()
