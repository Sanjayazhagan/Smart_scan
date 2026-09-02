"""Backward-compatible import for the canonical simulator implementation."""

from simulator.environment import Emitter, IQSampleBank, SmartScanEnv
from simulator.scenarios import ScenarioConfig, scenario_names

__all__ = [
    "Emitter",
    "IQSampleBank",
    "ScenarioConfig",
    "SmartScanEnv",
    "scenario_names",
]
