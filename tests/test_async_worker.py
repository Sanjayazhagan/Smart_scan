import time
import numpy as np
import pytest

from scheduler.baselines import BaseScheduler
from scheduler.async_worker import AsyncPlanningScheduler


class SlowMockPlanner(BaseScheduler):
    """Mock planner that simulates heavy neural tree search with deliberate delay."""

    def __init__(self, num_bands: int = 20, delay_seconds: float = 0.03):
        super().__init__(num_bands)
        self.delay = delay_seconds
        self.step_count = 0
        self.last_decision_trace = {}

    def select_band(self) -> int:
        if self.delay > 0:
            time.sleep(self.delay)
        action = self.step_count % self.num_bands
        self.step_count += 1
        self.last_decision_trace = {
            "selected_band": action,
            "expert": "slow_mock",
            "regime": "mock",
        }
        return action

    def update(self, band: int, reward: float, observation: dict) -> None:
        pass


def test_async_scheduler_non_blocking_latency():
    # Underlying planner takes 30 ms per step
    slow_planner = lambda: SlowMockPlanner(20, delay_seconds=0.03)
    scheduler = AsyncPlanningScheduler(
        20,
        slow_planner,
        target_horizon=5,
        worker_poll_timeout=0.002,
    )

    try:
        # Give worker a moment to warm up initial queue (5 x 30ms = 150ms)
        time.sleep(0.20)

        # Fast loop executes multiple steps rapidly
        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            band = scheduler.select_band()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            scheduler.update(band, 0.0, {"detected": False})

        # Fast loop should execute in under 2 ms (average typically < 0.1 ms)
        # despite the planner taking 30 ms!
        assert np.mean(latencies) < 2.0
    finally:
        scheduler.close()


def test_async_scheduler_queue_replenishment():
    slow_planner = lambda: SlowMockPlanner(20, delay_seconds=0.01)
    scheduler = AsyncPlanningScheduler(
        20,
        slow_planner,
        target_horizon=4,
        worker_poll_timeout=0.002,
    )

    try:
        time.sleep(0.08)
        # Pop 2 items
        b1 = scheduler.select_band()
        b2 = scheduler.select_band()
        assert 0 <= b1 < 20
        assert 0 <= b2 < 20

        # Wait for worker to replenish
        time.sleep(0.06)
        with scheduler._lock:
            assert len(scheduler._plan_queue) >= 3
    finally:
        scheduler.close()


def test_async_scheduler_reflex_interrupt_on_high_quality_pulse():
    slow_planner = lambda: SlowMockPlanner(20, delay_seconds=0.0)
    scheduler = AsyncPlanningScheduler(
        20,
        slow_planner,
        target_horizon=4,
        reflex_quality_threshold=0.60,
        enable_reflex=True,
    )

    try:
        time.sleep(0.05)
        # Seed plan queue with actions
        with scheduler._lock:
            scheduler._plan_queue.clear()
            scheduler._plan_queue.extend([1, 2, 3])

        # Report a high quality surprise hit on Band 9
        high_q_obs = {"detected": True, "quality": np.array([0.85], dtype=np.float32)}
        scheduler.update(9, 1.0, high_q_obs)

        # Reflex should have injected Band 9 at the front of the queue
        with scheduler._lock:
            assert scheduler._plan_queue[0] == 9
            assert scheduler.reflex_interrupt_count >= 1

        next_band = scheduler.select_band()
        assert next_band == 9
    finally:
        scheduler.close()


def test_async_scheduler_reset_and_lifecycle():
    slow_planner = lambda: SlowMockPlanner(20, delay_seconds=0.005)
    scheduler = AsyncPlanningScheduler(20, slow_planner, target_horizon=3)

    try:
        for step in range(5):
            band = scheduler.select_band()
            scheduler.update(band, 0.0, {"detected": False})

        # Test reset across episodes
        scheduler.reset()
        assert scheduler.timestamp == 0.0
        assert scheduler.total_scans == 0
        assert scheduler.queue_starvation_count == 0

        # Worker still functions after reset
        time.sleep(0.05)
        band = scheduler.select_band()
        assert 0 <= band < 20
    finally:
        scheduler.close()
