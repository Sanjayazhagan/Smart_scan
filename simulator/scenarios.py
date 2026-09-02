"""Deterministic hidden-world generation for realistic Smart Scan scenarios.

The generated world is independent of scheduler actions.  A scheduler can
change only what it observes, never the future emitter activity, hopping, SNR,
or interference timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    min_emitters: int = 2
    max_emitters: int = 10
    periodic_fraction: float = 0.50
    bursty_fraction: float = 0.10
    active_probability_range: tuple[float, float] = (0.20, 0.80)
    period_range: tuple[int, int] = (2, 10)
    burst_start_probability: float = 0.08
    burst_end_probability: float = 0.30
    hop_probability: float = 0.0
    behaviour_change_probability: float = 0.0
    force_hop_on_behaviour_change: bool = True
    snr_db_range: tuple[float, float] = (8.0, 22.0)
    snr_drift_std: float = 0.30
    fading_std_db: float = 1.0
    interference_probability: float = 0.0
    interference_penalty_db: float = 8.0
    interference_false_alarm_boost: float = 0.10
    sensor_dropout_probability: float = 0.0
    iq_noise_std: float = 0.05
    iq_imbalance_max: float = 0.0
    iq_clipping_level: float = 0.0
    switching_penalty: float = 0.0

    def __post_init__(self):
        if self.min_emitters < 1 or self.max_emitters < self.min_emitters:
            raise ValueError("Scenario emitter limits are invalid")
        probabilities = (
            self.periodic_fraction,
            self.bursty_fraction,
            self.burst_start_probability,
            self.burst_end_probability,
            self.hop_probability,
            self.behaviour_change_probability,
            self.interference_probability,
            self.interference_false_alarm_boost,
            self.sensor_dropout_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("Scenario probabilities must be in [0,1]")
        if self.periodic_fraction + self.bursty_fraction > 1.0:
            raise ValueError("Periodic and bursty fractions cannot exceed one")
        if self.snr_db_range[1] < self.snr_db_range[0]:
            raise ValueError("Scenario SNR range is invalid")
        if self.period_range[0] < 1 or self.period_range[1] < self.period_range[0]:
            raise ValueError("Scenario period range is invalid")


SCENARIO_PRESETS: dict[str, ScenarioConfig] = {
    "stationary": ScenarioConfig(name="stationary"),
    "hopping": ScenarioConfig(
        name="hopping",
        min_emitters=3,
        max_emitters=10,
        bursty_fraction=0.20,
        hop_probability=0.12,
        behaviour_change_probability=0.01,
        snr_db_range=(4.0, 20.0),
        interference_probability=0.03,
        iq_imbalance_max=0.04,
    ),
    "bursty": ScenarioConfig(
        name="bursty",
        min_emitters=3,
        max_emitters=10,
        periodic_fraction=0.15,
        bursty_fraction=0.70,
        burst_start_probability=0.05,
        burst_end_probability=0.40,
        hop_probability=0.03,
        snr_db_range=(2.0, 18.0),
        sensor_dropout_probability=0.01,
    ),
    "crowded": ScenarioConfig(
        name="crowded",
        min_emitters=8,
        max_emitters=15,
        periodic_fraction=0.30,
        bursty_fraction=0.30,
        active_probability_range=(0.35, 0.85),
        hop_probability=0.05,
        snr_db_range=(0.0, 18.0),
        fading_std_db=2.0,
        interference_probability=0.16,
        interference_penalty_db=10.0,
        interference_false_alarm_boost=0.18,
        iq_imbalance_max=0.06,
        switching_penalty=0.01,
    ),
    "changing": ScenarioConfig(
        name="changing",
        min_emitters=3,
        max_emitters=11,
        periodic_fraction=0.35,
        bursty_fraction=0.30,
        hop_probability=0.06,
        behaviour_change_probability=0.035,
        snr_db_range=(1.0, 20.0),
        snr_drift_std=0.75,
        fading_std_db=2.0,
        interference_probability=0.06,
        sensor_dropout_probability=0.02,
        switching_penalty=0.005,
    ),
    "harsh": ScenarioConfig(
        name="harsh",
        min_emitters=4,
        max_emitters=13,
        periodic_fraction=0.25,
        bursty_fraction=0.45,
        active_probability_range=(0.12, 0.70),
        hop_probability=0.12,
        behaviour_change_probability=0.03,
        snr_db_range=(-6.0, 12.0),
        snr_drift_std=1.0,
        fading_std_db=3.0,
        interference_probability=0.25,
        interference_penalty_db=12.0,
        interference_false_alarm_boost=0.25,
        sensor_dropout_probability=0.05,
        iq_noise_std=0.10,
        iq_imbalance_max=0.10,
        iq_clipping_level=2.5,
        switching_penalty=0.015,
    ),
    "mixed": ScenarioConfig(
        name="mixed",
        min_emitters=2,
        max_emitters=13,
        periodic_fraction=0.30,
        bursty_fraction=0.40,
        active_probability_range=(0.15, 0.80),
        hop_probability=0.07,
        behaviour_change_probability=0.02,
        snr_db_range=(-2.0, 20.0),
        snr_drift_std=0.65,
        fading_std_db=2.0,
        interference_probability=0.10,
        interference_penalty_db=10.0,
        interference_false_alarm_boost=0.15,
        sensor_dropout_probability=0.02,
        iq_noise_std=0.07,
        iq_imbalance_max=0.06,
        iq_clipping_level=3.5,
        switching_penalty=0.008,
    ),
}


def scenario_names() -> tuple[str, ...]:
    return ("legacy", *SCENARIO_PRESETS.keys())


def resolve_scenario(
    scenario: str | ScenarioConfig,
    *,
    min_emitters: int | None = None,
    max_emitters: int | None = None,
) -> ScenarioConfig:
    if isinstance(scenario, ScenarioConfig):
        config = scenario
    else:
        if scenario not in SCENARIO_PRESETS:
            raise ValueError(
                f"Unknown scenario {scenario!r}; choose from {scenario_names()}"
            )
        config = SCENARIO_PRESETS[scenario]
    changes = {}
    if min_emitters is not None:
        changes["min_emitters"] = int(min_emitters)
    if max_emitters is not None:
        changes["max_emitters"] = int(max_emitters)
    return replace(config, **changes) if changes else config


@dataclass
class ScenarioEmitter:
    id: int
    activity: np.ndarray
    bands: np.ndarray
    snr_db: np.ndarray
    behaviour_changes: np.ndarray

    def is_active(self, step_idx: int) -> bool:
        return bool(self.activity[int(step_idx)])

    def get_band(self, step_idx: int) -> int:
        return int(self.bands[int(step_idx)])

    def get_snr_db(self, step_idx: int) -> float:
        return float(self.snr_db[int(step_idx)])


@dataclass
class ScenarioWorld:
    name: str
    emitters: list[ScenarioEmitter]
    interference: np.ndarray
    episode_seed: int


def _different_band(rng: np.random.Generator, current: int, num_bands: int) -> int:
    if num_bands <= 1:
        return 0
    candidate = int(rng.integers(num_bands - 1))
    return candidate + 1 if candidate >= current else candidate


def generate_world(
    config: ScenarioConfig,
    *,
    num_bands: int,
    episode_length: int,
    seed: int,
) -> ScenarioWorld:
    """Generate a complete hidden world before the first scheduler action."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 17041]))
    count = int(rng.integers(config.min_emitters, config.max_emitters + 1))
    emitters: list[ScenarioEmitter] = []
    for emitter_id in range(count):
        mode_draw = float(rng.random())
        if mode_draw < config.periodic_fraction:
            mode = "periodic"
        elif mode_draw < config.periodic_fraction + config.bursty_fraction:
            mode = "bursty"
        else:
            mode = "bernoulli"

        activity = np.zeros(episode_length, dtype=bool)
        bands = np.empty(episode_length, dtype=np.int16)
        snr = np.empty(episode_length, dtype=np.float32)
        changes = np.zeros(episode_length, dtype=bool)
        probability = float(rng.uniform(*config.active_probability_range))
        period = int(rng.integers(config.period_range[0], config.period_range[1] + 1))
        offset = int(rng.integers(period))
        burst_on = bool(rng.random() < probability)
        bands[0] = int(rng.integers(num_bands))
        base_snr = float(rng.uniform(*config.snr_db_range))
        snr[0] = base_snr

        for step in range(episode_length):
            changed = step > 0 and rng.random() < config.behaviour_change_probability
            if changed:
                changes[step] = True
                probability = float(rng.uniform(*config.active_probability_range))
                period = int(
                    rng.integers(config.period_range[0], config.period_range[1] + 1)
                )
                offset = int(rng.integers(period))
                mode = ("periodic", "bursty", "bernoulli")[int(rng.integers(3))]

            if mode == "periodic":
                activity[step] = (step + offset) % period == 0
            elif mode == "bursty":
                if burst_on and rng.random() < config.burst_end_probability:
                    burst_on = False
                elif not burst_on and rng.random() < config.burst_start_probability:
                    burst_on = True
                activity[step] = burst_on
            else:
                activity[step] = rng.random() < probability

            if step > 0:
                bands[step] = bands[step - 1]
                if (
                    rng.random() < config.hop_probability
                    or (changed and config.force_hop_on_behaviour_change)
                ):
                    bands[step] = _different_band(
                        rng, int(bands[step - 1]), num_bands
                    )
                base_snr = float(
                    np.clip(
                        base_snr + rng.normal(0.0, config.snr_drift_std),
                        config.snr_db_range[0],
                        config.snr_db_range[1],
                    )
                )
                snr[step] = base_snr
            snr[step] += float(rng.normal(0.0, config.fading_std_db))

        emitters.append(
            ScenarioEmitter(
                id=emitter_id,
                activity=activity,
                bands=bands,
                snr_db=snr,
                behaviour_changes=changes,
            )
        )

    interference = (
        rng.random((episode_length, num_bands))
        < config.interference_probability
    )
    return ScenarioWorld(
        name=config.name,
        emitters=emitters,
        interference=interference,
        episode_seed=int(seed),
    )
