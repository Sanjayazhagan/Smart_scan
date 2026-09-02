import pytest
import numpy as np
import gymnasium as gym

from simulator.environment import SmartScanEnv
from simulator.scenarios import scenario_names

def test_observation_space_structure():
    """Test 1: Verify output observation space structure"""
    env = SmartScanEnv(num_bands=10, episode_length=50)
    obs, info = env.reset()
    
    assert "selected_band" in obs
    assert "detected" in obs
    assert "signal_power" in obs
    assert "quality" in obs
    assert "iq" in obs
    assert obs["iq"].shape == (2, 512)
    
    # Assert observation space strictly matches observation
    assert env.observation_space.contains(obs), "Reset observation doesn't match space"
    
    # Take a step
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs), "Step observation doesn't match space"
    

def test_deterministic_seeding():
    """Test 2: Verify deterministic seeding (two runs with seed=42 output identical step-by-step history)"""
    env1 = SmartScanEnv(seed=42)
    env2 = SmartScanEnv(seed=42)
    
    # Reset both envs with the same seed
    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)
    
    # Check if ground truth maps match on step 0
    assert info1["ground_truth_active_emitters"] == info2["ground_truth_active_emitters"]
    assert info1["ground_truth_active_bands"] == info2["ground_truth_active_bands"]
    
    # Run through deterministic actions
    for step in range(20):
        action = step % env1.action_space.n
        
        step_obs1, reward1, _, _, step_info1 = env1.step(action)
        step_obs2, reward2, _, _, step_info2 = env2.step(action)
        
        # Verify strict equivalence in observation
        assert step_obs1["selected_band"] == step_obs2["selected_band"]
        assert step_obs1["detected"] == step_obs2["detected"]
        assert np.isclose(step_obs1["signal_power"], step_obs2["signal_power"])
        assert np.isclose(step_obs1["quality"], step_obs2["quality"])
        assert np.allclose(step_obs1["iq"], step_obs2["iq"])
        
        # Verify strict equivalence in hidden info and reward
        assert reward1 == reward2
        assert step_info1["ground_truth_active_emitters"] == step_info2["ground_truth_active_emitters"]


def test_partial_observability():
    """Test 3: Verify partial observability (observation dict contains no global state leaks)"""
    env = SmartScanEnv(num_bands=5, episode_length=20)
    obs, info = env.reset()
    
    # Verify no leaks in initial observation
    assert "ground_truth_active_emitters" in info
    assert "ground_truth_active_emitters" not in obs
    assert "ground_truth_active_bands" in info
    assert "ground_truth_active_bands" not in obs
    
    obs, reward, terminated, truncated, info = env.step(3)
    
    # Verify no leaks during stepping
    assert "ground_truth_active_emitters" in info
    assert "ground_truth_active_emitters" not in obs
    assert "true_signal_present" in info
    assert "true_signal_present" not in obs
    
    # I/Q is a partial measurement of only the selected band, not ground truth.
    expected_keys = {"selected_band", "detected", "signal_power", "quality", "iq"}
    assert set(obs.keys()) == expected_keys


def test_integrated_default_uses_twenty_bands():
    env = SmartScanEnv()
    assert env.num_bands == 20
    assert env.action_space.n == 20


def test_scenario_world_is_independent_of_scheduler_actions():
    """Different policies must face the same hidden future for the same seed."""
    env1 = SmartScanEnv(seed=314, scenario="mixed", episode_length=80)
    env2 = SmartScanEnv(seed=314, scenario="mixed", episode_length=80)
    env1.reset(seed=314)
    env2.reset(seed=314)
    for step in range(60):
        _, _, _, _, info1 = env1.step(step % env1.num_bands)
        _, _, _, _, info2 = env2.step((19 - step) % env2.num_bands)
        assert info1["ground_truth_active_emitters"] == info2[
            "ground_truth_active_emitters"
        ]
        assert info1["ground_truth_active_bands"] == info2[
            "ground_truth_active_bands"
        ]
        assert np.allclose(
            info1["ground_truth_active_snr_db"],
            info2["ground_truth_active_snr_db"],
        )
        assert info1["ground_truth_behaviour_changes"] == info2[
            "ground_truth_behaviour_changes"
        ]


def test_hopping_scenario_changes_emitter_bands_and_preserves_observation_schema():
    env = SmartScanEnv(seed=2718, scenario="hopping", episode_length=100)
    observation, info = env.reset(seed=2718)
    assert env.observation_space.contains(observation)
    assert "hopping" in scenario_names()
    assert any(len(set(emitter.bands.tolist())) > 1 for emitter in env.emitters)
    observation, _, _, _, info = env.step(3)
    assert env.observation_space.contains(observation)
    assert info["scenario"] == "hopping"
    assert "interference_present" in info
    assert "switching_cost" in info
    assert "ground_truth_active_bands" not in observation
