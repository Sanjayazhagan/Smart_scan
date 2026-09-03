from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator.scenarios import (
    ScenarioConfig,
    ScenarioWorld,
    generate_world,
    resolve_scenario,
)


NUM_BANDS = 20
IQ_SHAPE = (2, 512)
DEFAULT_IQ_DATASET_PATH = (
    Path.home()
    / "Documents"
    / "SmartScanArtifacts"
    / "track2"
    / "track2_synthetic_rf_dataset.npz"
)


class Emitter:
    """A hidden RF emitter with a persistent activity pattern and band."""

    def __init__(self, emitter_id: int, num_bands: int, np_random: np.random.Generator):
        self.id = emitter_id
        self.num_bands = num_bands
        self.np_random = np_random
        self.pattern_type = self.np_random.integers(0, 2)
        self.period = self.np_random.integers(2, 10)
        self.offset = self.np_random.integers(0, self.period)
        self.prob_active = self.np_random.uniform(0.2, 0.8)
        self.band = self.np_random.integers(0, self.num_bands)

    def is_active(self, step_idx: int) -> bool:
        if self.pattern_type == 0:
            return (step_idx + self.offset) % self.period == 0
        return self.np_random.random() < self.prob_active

    def get_band(self, step_idx: int) -> int:
        return int(self.band)


class IQSampleBank:
    """Indexes benign synthetic I/Q by simulator-only emitter id and band."""

    def __init__(self, dataset_path: str | Path | None):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.by_emitter_band: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        self.by_emitter: dict[int, list[np.ndarray]] = defaultdict(list)

        if self.dataset_path is None or not self.dataset_path.is_file():
            return

        with np.load(self.dataset_path, allow_pickle=True) as data:
            required = {"iq", "emitter_id", "band"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(
                    f"I/Q dataset {self.dataset_path} is missing keys: {sorted(missing)}"
                )
            for iq, emitter_id, band in zip(
                data["iq"], data["emitter_id"], data["band"], strict=True
            ):
                sample = self._coerce_iq(iq)
                emitter_id = int(emitter_id)
                band = int(band)
                self.by_emitter_band[(emitter_id, band)].append(sample)
                self.by_emitter[emitter_id].append(sample)

    @staticmethod
    def _coerce_iq(iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq)
        if np.iscomplexobj(iq):
            iq = np.stack((iq.real, iq.imag), axis=0)
        if iq.shape == (IQ_SHAPE[1], IQ_SHAPE[0]):
            iq = iq.T
        if iq.shape != IQ_SHAPE:
            raise ValueError(f"Expected I/Q shape {IQ_SHAPE}, received {iq.shape}")
        return np.asarray(iq, dtype=np.float32)

    @property
    def loaded(self) -> bool:
        return bool(self.by_emitter)

    def sample(
        self, emitter_id: int, band: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        candidates = self.by_emitter_band.get((int(emitter_id), int(band)))
        if not candidates:
            candidates = self.by_emitter.get(int(emitter_id))
        if not candidates:
            return None
        return candidates[int(rng.integers(0, len(candidates)))].copy()


class SmartScanEnv(gym.Env):
    """Strictly partially observable, one-band-per-step RF simulator."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        episode_length: int = 200,
        miss_prob: float = 0.1,
        false_alarm_prob: float = 0.05,
        seed: int | None = None,
        iq_dataset_path: str | Path | None = DEFAULT_IQ_DATASET_PATH,
        scenario: str | ScenarioConfig = "legacy",
    ):
        super().__init__()
        self.num_bands = int(num_bands)
        self.episode_length = int(episode_length)
        self.miss_prob = float(miss_prob)
        self.false_alarm_prob = float(false_alarm_prob)
        self.scenario = scenario
        self.scenario_config: ScenarioConfig | None = None
        self.world: ScenarioWorld | None = None
        self.episode_seed = 0
        self.last_action: int | None = None
        self.iq_sample_bank = IQSampleBank(iq_dataset_path)
        self.current_step = 0
        self.emitters: list[Emitter] = []

        self.action_space = spaces.Discrete(self.num_bands)
        self.observation_space = spaces.Dict(
            {
                "selected_band": spaces.Discrete(self.num_bands),
                "detected": spaces.Discrete(2),
                "signal_power": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "quality": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "iq": spaces.Box(
                    low=-np.inf, high=np.inf, shape=IQ_SHAPE, dtype=np.float32
                ),
            }
        )

        if seed is not None:
            self.reset(seed=seed)

    @property
    def scenario_name(self) -> str:
        if isinstance(self.scenario, ScenarioConfig):
            return self.scenario.name
        return str(self.scenario)

    def _keyed_rng(self, stream: int, step: int, band: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence(
                [int(self.episode_seed), int(stream), int(step), int(band)]
            )
        )

    def _noise_iq(self, step: int | None = None, band: int = 0) -> np.ndarray:
        if self.scenario_name == "legacy":
            return self.np_random.normal(0.0, 0.05, IQ_SHAPE).astype(np.float32)
        config = self.scenario_config
        if config is None:
            raise RuntimeError("Scenario configuration is unavailable")
        step = self.current_step if step is None else int(step)
        rng = self._keyed_rng(2401, step, band)
        return rng.normal(0.0, config.iq_noise_std, IQ_SHAPE).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action = None
        if self.scenario_name == "legacy":
            self.scenario_config = None
            self.world = None
            num_emitters = self.np_random.integers(
                2, max(3, self.num_bands // 2 + 1)
            )
            self.emitters = [
                Emitter(i, self.num_bands, self.np_random) for i in range(num_emitters)
            ]
        else:
            self.episode_seed = (
                int(seed)
                if seed is not None
                else int(self.np_random.integers(0, 2**31 - 1))
            )
            self.scenario_config = resolve_scenario(self.scenario)
            self.world = generate_world(
                self.scenario_config,
                num_bands=self.num_bands,
                episode_length=self.episode_length,
                seed=self.episode_seed,
            )
            self.emitters = list(self.world.emitters)
        obs = {
            "selected_band": np.int64(0),
            "detected": np.int64(0),
            "signal_power": np.array([0.0], dtype=np.float32),
            "quality": np.array([0.0], dtype=np.float32),
            "iq": self._noise_iq(),
        }
        return obs, self._get_ground_truth_info()

    def _get_ground_truth_info(self) -> dict:
        active_emitters: list[int] = []
        true_active_bands: list[int] = []
        active_snr_db: list[float] = []
        active_threat_weights: list[float] = []
        behaviour_changes: list[int] = []
        for emitter in self.emitters:
            if (
                self.scenario_name != "legacy"
                and bool(emitter.behaviour_changes[self.current_step])
            ):
                behaviour_changes.append(int(emitter.id))
            if emitter.is_active(self.current_step):
                active_emitters.append(emitter.id)
                true_active_bands.append(emitter.get_band(self.current_step))
                if self.scenario_name != "legacy":
                    active_snr_db.append(emitter.get_snr_db(self.current_step))
                    active_threat_weights.append(float(emitter.threat_weight))
        info = {
            "ground_truth_active_emitters": active_emitters,
            "ground_truth_active_bands": true_active_bands,
        }
        if self.scenario_name != "legacy":
            info.update(
                {
                    "ground_truth_active_snr_db": active_snr_db,
                    "ground_truth_active_threat_weights": active_threat_weights,
                    "ground_truth_behaviour_changes": behaviour_changes,
                    "ground_truth_emitter_start_steps": {
                        int(emitter.id): int(emitter.start_step)
                        for emitter in self.emitters
                    },
                    "ground_truth_emitter_end_steps": {
                        int(emitter.id): int(emitter.end_step)
                        for emitter in self.emitters
                    },
                    "ground_truth_emitter_threat_weights": {
                        int(emitter.id): float(emitter.threat_weight)
                        for emitter in self.emitters
                    },
                    "scenario": self.scenario_name,
                }
            )
        return info

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Band action {action} is outside 0..{self.num_bands - 1}")

        if self.scenario_name != "legacy":
            return self._step_scenario(action)

        return self._step_legacy(action)

    def _step_legacy(self, action: int):

        ground_truth_info = self._get_ground_truth_info()
        true_active_bands = ground_truth_info["ground_truth_active_bands"]
        is_signal_present = action in true_active_bands

        detected = is_signal_present
        if is_signal_present and self.np_random.random() < self.miss_prob:
            detected = False
        elif not is_signal_present and self.np_random.random() < self.false_alarm_prob:
            detected = True

        if detected and is_signal_present:
            signal_power = self.np_random.normal(10.0, 2.0)
            quality = self.np_random.uniform(0.7, 1.0)
        elif detected:
            signal_power = self.np_random.normal(3.0, 1.0)
            quality = self.np_random.uniform(0.1, 0.4)
        else:
            signal_power = self.np_random.exponential(1.0)
            quality = self.np_random.uniform(0.0, 0.2)

        iq = None
        if detected and is_signal_present:
            matching_ids = [
                emitter_id
                for emitter_id, band in zip(
                    ground_truth_info["ground_truth_active_emitters"],
                    true_active_bands,
                    strict=True,
                )
                if band == action
            ]
            if matching_ids:
                iq = self.iq_sample_bank.sample(matching_ids[0], action, self.np_random)
        if iq is None:
            iq = self._noise_iq()

        obs = {
            "selected_band": np.int64(action),
            "detected": np.int64(bool(detected)),
            "signal_power": np.array([max(0.0, float(signal_power))], dtype=np.float32),
            "quality": np.array([quality], dtype=np.float32),
            "iq": iq,
        }

        if is_signal_present and detected:
            reward = 1.0
        elif not is_signal_present and not detected:
            reward = 0.1
        else:
            reward = -1.0

        info = dict(ground_truth_info)
        info["true_signal_present"] = is_signal_present
        self.current_step += 1
        return obs, reward, False, self.current_step >= self.episode_length, info

    def _step_scenario(self, action: int):
        config = self.scenario_config
        world = self.world
        if config is None or world is None:
            raise RuntimeError("Scenario world has not been initialized")

        step = self.current_step
        ground_truth_info = self._get_ground_truth_info()
        active_emitters = ground_truth_info["ground_truth_active_emitters"]
        active_bands = ground_truth_info["ground_truth_active_bands"]
        active_snr = ground_truth_info["ground_truth_active_snr_db"]
        matching = [
            (int(emitter_id), float(snr))
            for emitter_id, band, snr in zip(
                active_emitters, active_bands, active_snr, strict=True
            )
            if int(band) == action
        ]
        signal_present = bool(matching)
        interference_present = bool(world.interference[step, action])
        strongest_snr = max((snr for _, snr in matching), default=float("nan"))
        effective_snr = strongest_snr
        if signal_present and interference_present:
            effective_snr -= config.interference_penalty_db

        switch_distance = (
            0.0
            if self.last_action is None
            else abs(action - self.last_action) / max(1, self.num_bands - 1)
        )
        retune_factor = float(
            np.clip(1.0 - config.retune_detection_loss * switch_distance, 0.1, 1.0)
        )

        sensor_rng = self._keyed_rng(2101, step, action)
        dropout = bool(sensor_rng.random() < config.sensor_dropout_probability)
        if signal_present:
            snr_factor = 1.0 / (1.0 + np.exp(-effective_snr / 4.0))
            detection_probability = (1.0 - self.miss_prob) * (
                0.15 + 0.85 * snr_factor
            ) * retune_factor
            detected = bool(sensor_rng.random() < detection_probability)
        else:
            detection_probability = float(
                np.clip(
                    self.false_alarm_prob
                    + (
                        config.interference_false_alarm_boost
                        if interference_present
                        else 0.0
                    ),
                    0.0,
                    1.0,
                )
            )
            detected = bool(sensor_rng.random() < detection_probability)
        if dropout:
            detected = False

        measurement_rng = self._keyed_rng(2201, step, action)
        if detected and signal_present:
            signal_power = max(0.0, 10.0 + 0.45 * effective_snr + measurement_rng.normal())
            quality = float(
                np.clip(
                    1.0 / (1.0 + np.exp(-(effective_snr - 1.0) / 5.0))
                    + measurement_rng.normal(0.0, 0.04),
                    0.0,
                    1.0,
                )
            )
        elif detected:
            signal_power = max(0.0, measurement_rng.normal(3.0, 1.0))
            if interference_present and config.deceptive_interference_quality > 0.0:
                centre = config.deceptive_interference_quality
                quality = float(
                    np.clip(measurement_rng.normal(centre, 0.10), 0.08, 0.95)
                )
            else:
                quality = float(measurement_rng.uniform(0.08, 0.38))
        else:
            signal_power = float(measurement_rng.exponential(1.0))
            quality = float(measurement_rng.uniform(0.0, 0.20))

        iq = None
        if detected and signal_present:
            emitter_id = max(matching, key=lambda item: item[1])[0]
            iq_rng = self._keyed_rng(2301, step, action)
            iq = self.iq_sample_bank.sample(emitter_id, action, iq_rng)
            if iq is not None:
                iq = self._apply_channel(iq, effective_snr, iq_rng, config)
        if iq is None:
            iq = self._noise_iq(step, action)
            if interference_present:
                phase = measurement_rng.uniform(0.0, 2.0 * np.pi)
                tone = 0.08 * np.sin(
                    np.linspace(phase, phase + 16.0 * np.pi, IQ_SHAPE[1])
                )
                iq = iq.copy()
                iq[0] += tone.astype(np.float32)

        observation = {
            "selected_band": np.int64(action),
            "detected": np.int64(bool(detected)),
            "signal_power": np.array([signal_power], dtype=np.float32),
            "quality": np.array([quality], dtype=np.float32),
            "iq": np.asarray(iq, dtype=np.float32),
        }

        if signal_present and detected:
            reward = 1.0
        elif not signal_present and not detected:
            reward = 0.1
        else:
            reward = -1.0
        switching_cost = config.switching_penalty * switch_distance
        reward -= switching_cost

        info = dict(ground_truth_info)
        info.update(
            {
                "true_signal_present": signal_present,
                "interference_present": interference_present,
                "sensor_dropout": dropout,
                "detection_probability": float(detection_probability),
                "effective_snr_db": float(effective_snr)
                if signal_present
                else None,
                "switch_distance": float(switch_distance),
                "switching_cost": float(switching_cost),
                "retune_detection_factor": retune_factor,
                "deceptive_false_alarm": bool(
                    detected and not signal_present and interference_present
                ),
            }
        )
        self.last_action = action
        self.current_step += 1
        return (
            observation,
            float(reward),
            False,
            self.current_step >= self.episode_length,
            info,
        )

    @staticmethod
    def _apply_channel(
        iq: np.ndarray,
        effective_snr_db: float,
        rng: np.random.Generator,
        config: ScenarioConfig,
    ) -> np.ndarray:
        sample = np.asarray(iq, dtype=np.float32).copy()
        signal_scale = float(np.clip(10.0 ** ((effective_snr_db - 10.0) / 20.0), 0.2, 2.5))
        sample *= signal_scale
        imbalance = float(rng.uniform(-config.iq_imbalance_max, config.iq_imbalance_max))
        sample[0] *= 1.0 + imbalance
        sample[1] *= 1.0 - imbalance
        sample += rng.normal(0.0, config.iq_noise_std, sample.shape).astype(np.float32)
        if config.iq_clipping_level > 0.0:
            sample = np.clip(
                sample, -config.iq_clipping_level, config.iq_clipping_level
            )
        return sample.astype(np.float32)

    def render(self):
        print(f"Step: {self.current_step}")
