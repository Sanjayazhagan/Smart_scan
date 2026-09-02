import pytest
from scheduler.baselines import (
    FixedScheduler, 
    RandomScheduler, 
    UCB1Scheduler, 
    ThompsonSamplingScheduler
)

def test_fixed_scheduler():
    scheduler = FixedScheduler(num_bands=3)
    assert scheduler.select_band() == 0
    assert scheduler.select_band() == 1
    assert scheduler.select_band() == 2
    assert scheduler.select_band() == 0

def test_random_scheduler():
    scheduler = RandomScheduler(num_bands=5, seed=42)
    bands = [scheduler.select_band() for _ in range(100)]
    assert all(0 <= b < 5 for b in bands)
    
def test_ucb1_scheduler():
    scheduler = UCB1Scheduler(num_bands=3)
    
    # Phase 1: Exploring all bands initially
    assert scheduler.select_band() == 0
    scheduler.update(0, 0.0)
    
    assert scheduler.select_band() == 1
    scheduler.update(1, 1.0) # High reward for band 1
    
    assert scheduler.select_band() == 2
    scheduler.update(2, 0.0)
    
    # Phase 2: Exploitation
    # Next should be 1 because it yielded the highest reward
    assert scheduler.select_band() == 1

def test_thompson_sampling_scheduler():
    scheduler = ThompsonSamplingScheduler(num_bands=3, seed=42)
    
    # Artificially inflate band 2's alpha (successes)
    for _ in range(50):
        scheduler.update(2, 1.0)
    
    # Artificially inflate band 0 and 1's beta (failures)
    for _ in range(50):
        scheduler.update(0, -1.0)
        scheduler.update(1, -1.0)
        
    # Most subsequent samples should heavily favor band 2
    selections = [scheduler.select_band() for _ in range(100)]
    assert selections.count(2) > 90 # High probability it picks the successful arm
