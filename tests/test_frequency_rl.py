import pytest
import numpy as np
from simulator.environment import SmartScanEnv
from scheduler.frequency_rl import FrequencyHistoryWrapper, RLScheduler

def test_frequency_history_wrapper():
    """Test 1: Verify wrapper alters observation space properly and maintains history."""
    env = SmartScanEnv(num_bands=4, episode_length=10)
    wrapped_env = FrequencyHistoryWrapper(env)
    
    obs, info = wrapped_env.reset()
    assert obs.shape == (4, 3)
    assert np.all(obs[:, 0] == 100.0) # Check initial timers
    
    next_obs, reward, terminated, truncated, info = wrapped_env.step(2)
    
    # Band 2 was scanned, so its timer should reset to 0
    assert next_obs[2, 0] == 0.0
    
    # Other bands should increment timer
    assert next_obs[0, 0] == 101.0
    assert next_obs[1, 0] == 101.0
    assert next_obs[3, 0] == 101.0
    
    # Validate strictly enforced partial observability
    assert "ground_truth_active_emitters" not in next_obs
    assert wrapped_env.observation_space.contains(next_obs)

def test_rl_scheduler_mock_model():
    """Test 2: Verify RLScheduler integrates correctly with the simulation loop."""
    class MockModel:
        def predict(self, observation, deterministic=True):
            # Deterministically choose band 1
            return 1, None
            
    mock_model = MockModel()
    scheduler = RLScheduler(num_bands=4, model=mock_model)
    
    assert scheduler.select_band() == 1
    
    obs_dict = {
        "selected_band": 1,
        "detected": 1,
        "signal_power": [5.5],
        "quality": [0.9]
    }
    
    assert scheduler.history[1, 0] == 100.0
    scheduler.update(band=1, reward=1.0, obs_dict=obs_dict)
    
    # History state mirrors the environment correctly
    assert scheduler.history[1, 0] == 0.0
    assert scheduler.history[1, 1] == 1.0
    assert scheduler.history[1, 2] == 5.5
