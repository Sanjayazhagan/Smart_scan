"""Asynchronous background planning worker for high-throughput RF spectrum scanning.

Decouples the high-frequency antenna dwell loop (microseconds/milliseconds)
from heavy neural deliberative calculations (GRU world model updates and
3-step Beam/Tree rollouts) via a thread-safe multi-step plan queue.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

from scheduler.baselines import BaseScheduler, UCB1Scheduler


class AsyncPlanningScheduler(BaseScheduler):
    """Two-tier asynchronous scheduler.

    - The **Fast Loop** (radio/antenna driver): Calls ``select_band()`` and
      ``update()`` with sub-millisecond execution by popping pre-computed
      actions from a thread-safe horizon queue.
    - The **Background Worker** (deliberative planner): Concurrently ingests
      accumulated RF telemetry, runs heavy Track 2 GRU / Tree / MPP rollouts,
      and replenishes the plan queue.
    """

    def __init__(
        self,
        num_bands: int,
        planner_factory: Callable[[], BaseScheduler],
        target_horizon: int = 4,
        max_queue_size: int = 12,
        reflex_quality_threshold: float = 0.60,
        enable_reflex: bool = True,
        worker_poll_timeout: float = 0.005,
    ):
        super().__init__(num_bands)
        self.target_horizon = int(target_horizon)
        self.max_queue_size = int(max_queue_size)
        self.reflex_quality_threshold = float(reflex_quality_threshold)
        self.enable_reflex = bool(enable_reflex)
        self.worker_poll_timeout = float(worker_poll_timeout)

        # Thread-safe synchronization primitives
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lock = threading.RLock()
        self._plan_queue: deque[int] = deque()
        self._telemetry_queue: queue.Queue[tuple[int, float, dict, float] | None] = (
            queue.Queue()
        )

        # Internal deliberative planner runs inside the worker thread
        self.planner: BaseScheduler = planner_factory()

        # Fast non-blocking fallback if plan queue is ever starved
        self.fallback = UCB1Scheduler(num_bands)

        # Decision trace and metrics
        self.last_decision_trace: dict[str, Any] = {}
        self.queue_starvation_count = 0
        self.reflex_interrupt_count = 0
        self.total_scans = 0
        self.timestamp = 0.0

        # Start the background planner daemon
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="SmartScan-AsyncPlannerWorker",
            daemon=True,
        )
        self._worker_thread.start()
        self._wake_event.set()

    def _worker_loop(self) -> None:
        """Continuously consumes telemetry, updates internal state, and plans ahead."""
        while not self._stop_event.is_set():
            # Ingest all pending telemetry observations in batch
            with self._lock:
                while True:
                    try:
                        item = self._telemetry_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:  # Shutdown sentinel
                        return
                    band, reward, obs, ts = item
                    self.planner.update(band, reward, obs)
                    self._telemetry_queue.task_done()

                current_len = len(self._plan_queue)

                if current_len < self.target_horizon:
                    actions_to_add: list[int] = []
                    needed = self.target_horizon - current_len

                    last_trace = getattr(self.planner, "last_decision_trace", {}) or {}
                    planned_seq = last_trace.get("planned_sequence") or last_trace.get("greedy_planned_sequence")

                    if planned_seq and len(planned_seq) >= needed:
                        actions_to_add = [int(b) for b in planned_seq[:needed]]
                    else:
                        for _ in range(needed):
                            action = int(self.planner.select_band())
                            actions_to_add.append(action)

                    for action in actions_to_add:
                        if len(self._plan_queue) < self.max_queue_size:
                            self._plan_queue.append(action)

            # Wait for wake signal or short poll interval
            self._wake_event.wait(timeout=self.worker_poll_timeout)
            self._wake_event.clear()

    def _generate_horizon(self) -> None:
        """Generate next planned actions from deliberative planner."""
        with self._lock:
            while True:
                try:
                    item = self._telemetry_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    return
                band, reward, obs, ts = item
                self.planner.update(band, reward, obs)
                self._telemetry_queue.task_done()

            action = int(self.planner.select_band())
            self._plan_queue.append(action)

            last_trace = getattr(self.planner, "last_decision_trace", {}) or {}
            planned_seq = last_trace.get("planned_sequence") or last_trace.get("greedy_planned_sequence")
            if planned_seq and len(planned_seq) > 1:
                for b in planned_seq[1:self.target_horizon]:
                    if len(self._plan_queue) < self.max_queue_size:
                        self._plan_queue.append(int(b))
            else:
                while len(self._plan_queue) < self.target_horizon:
                    self._plan_queue.append(int(self.planner.select_band()))

    def select_band(self) -> int:
        """Fast-loop selection. Returns in microseconds from the plan queue."""
        self.total_scans += 1
        with self._lock:
            if not self._plan_queue:
                self._generate_horizon()

            if self._plan_queue:
                action = self._plan_queue.popleft()
                source = "async_plan_queue"
            else:
                # Queue starved - use fast reactive fallback
                action = int(self.fallback.select_band())
                self.queue_starvation_count += 1
                source = "fast_fallback_starvation"

            remaining_depth = len(self._plan_queue)

        # Notify worker to replenish if queue is below target horizon
        if remaining_depth < self.target_horizon:
            self._wake_event.set()

        underlying_trace = getattr(self.planner, "last_decision_trace", {}) or {}
        self.last_decision_trace = {
            "selected_band": action,
            "routing_source": source,
            "async_queue_depth": remaining_depth,
            "expert": underlying_trace.get("expert"),
            "regime": underlying_trace.get("regime"),
            "router": underlying_trace.get("router"),
        }
        return action

    def update(self, band: int, reward: float, observation: dict) -> None:
        """Fast-loop observation ingest. Pushes telemetry to worker non-blocking."""
        self.fallback.update(band, reward, observation)
        self.timestamp += 1.0

        # Reflex Safeguard: High-quality surprise detection triggers immediate re-dwell
        if self.enable_reflex and observation.get("detected"):
            quality_arr = observation.get("quality", 0.0)
            quality = (
                float(quality_arr[0])
                if isinstance(quality_arr, (np.ndarray, list))
                else float(quality_arr)
            )
            if quality >= self.reflex_quality_threshold:
                with self._lock:
                    # Inject immediate re-dwell to confirm track before emitter hops
                    if not self._plan_queue or self._plan_queue[0] != band:
                        self._plan_queue.appendleft(band)
                        self.reflex_interrupt_count += 1

        # Push to background worker
        self._telemetry_queue.put((band, float(reward), observation, self.timestamp))
        self._wake_event.set()

    def begin_scored_phase(self) -> dict[str, Any]:
        """Forward calibration to underlying planner if supported."""
        with self._lock:
            if hasattr(self.planner, "begin_scored_phase"):
                while True:
                    try:
                        item = self._telemetry_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        break
                    band, reward, obs, ts = item
                    self.planner.update(band, reward, obs)
                    self._telemetry_queue.task_done()

                res = self.planner.begin_scored_phase()
                self.calibration_trace = getattr(self.planner, "calibration_trace", {}) or {}
                self._plan_queue.clear()
                self._wake_event.set()
                return res
            return {}

    def reset(self) -> None:
        """Reset fast loop and internal worker state across episodes."""
        with self._lock:
            self.timestamp = 0.0
            self.queue_starvation_count = 0
            self.reflex_interrupt_count = 0
            self.total_scans = 0
            if hasattr(self.fallback, "reset"):
                self.fallback.reset()
            else:
                self.fallback = UCB1Scheduler(self.num_bands)

            self._plan_queue.clear()

            # Drain telemetry queue
            while not self._telemetry_queue.empty():
                try:
                    self._telemetry_queue.get_nowait()
                    self._telemetry_queue.task_done()
                except queue.Empty:
                    break

            if hasattr(self.planner, "reset"):
                self.planner.reset()

            self._wake_event.set()

    def close(self) -> None:
        """Gracefully terminate background thread."""
        self._stop_event.set()
        self._wake_event.set()
        try:
            self._telemetry_queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def __del__(self) -> None:
        self.close()
