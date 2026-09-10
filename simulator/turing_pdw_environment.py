"""PDW-native benchmark environment for the Turing Synthetic Radar Dataset.

The environment consumes the TSRD HDF5 Pulse Descriptor Word stream directly.
No I/Q waveform is created, sampled, or required.

For scheduler evaluation, use *stare* files when possible: stare provides an
oracle-like full-spectrum record, whereas scan files are already censored by a
particular deterministic receiver sweep and therefore cannot serve as neutral
ground truth for a different scheduling policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

NUM_BANDS = 20


@dataclass(frozen=True)
class TuringPDWColumns:
    toa: int = 0
    frequency: int = 1
    pulse_width: int = 2
    aoa: int = 3
    amplitude: int = 4


class TuringPulseTrain:
    """Loads one TSRD-style HDF5 pulse train into compact NumPy arrays."""

    def __init__(
        self,
        path: str | Path,
        *,
        num_bands: int = NUM_BANDS,
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 18_000.0,
        columns: TuringPDWColumns = TuringPDWColumns(),
    ):
        self.path = Path(path)
        self.num_bands = int(num_bands)
        self.freq_min_mhz = float(freq_min_mhz)
        self.freq_max_mhz = float(freq_max_mhz)
        self.columns = columns

        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.freq_max_mhz <= self.freq_min_mhz:
            raise ValueError("freq_max_mhz must exceed freq_min_mhz")

        with h5py.File(self.path, "r") as f:
            if "data" not in f or "labels" not in f:
                raise ValueError(f"{self.path} must contain HDF5 datasets 'data' and 'labels'")
            raw = np.asarray(f["data"][:])
            labels = np.asarray(f["labels"][:]).reshape(-1)

        if raw.ndim != 2 or raw.shape[1] < 5:
            raise ValueError(f"Expected PDW matrix [N, >=5], got {raw.shape}")
        if len(labels) != len(raw):
            raise ValueError("TSRD data and labels lengths do not match")
        if len(raw) == 0:
            raise ValueError(f"TSRD pulse train {self.path} contains no pulses")

        order = np.argsort(raw[:, columns.toa], kind="stable")
        raw = raw[order]
        labels = labels[order]

        self.toa_us = np.asarray(raw[:, columns.toa], dtype=np.float64)
        self.frequency_mhz = np.asarray(raw[:, columns.frequency], dtype=np.float32)
        self.pulse_width_us = np.asarray(raw[:, columns.pulse_width], dtype=np.float32)
        self.aoa_deg = np.asarray(raw[:, columns.aoa], dtype=np.float32)
        self.amplitude_db = np.asarray(raw[:, columns.amplitude], dtype=np.float32)
        self.labels = labels.astype(np.int64, copy=False)

        span = self.freq_max_mhz - self.freq_min_mhz
        self.bands = np.clip(
            ((self.frequency_mhz - self.freq_min_mhz) / span * self.num_bands).astype(np.int16),
            0,
            self.num_bands - 1,
        )

        # Dataset-level robust calibration only; no future signal content is fed
        # to the scheduler.  It merely maps amplitude dB to a stable [0,1] quality.
        self.amp_lo = float(np.percentile(self.amplitude_db, 5.0))
        self.amp_hi = float(np.percentile(self.amplitude_db, 95.0))
        self.amp_median = float(np.median(self.amplitude_db))
        if self.amp_hi <= self.amp_lo:
            self.amp_hi = self.amp_lo + 1.0

    @property
    def duration_us(self) -> float:
        return float(self.toa_us[-1] - self.toa_us[0])

    def amplitude_quality(self, amp_db: float) -> float:
        return float(np.clip((float(amp_db) - self.amp_lo) / (self.amp_hi - self.amp_lo), 0.0, 1.0))

    def relative_power(self, amp_db: float) -> float:
        # Amplitude is in dB.  Convert relative-to-median dB to a bounded linear
        # amplitude ratio so NMF receives non-negative measured energy.
        return float(np.clip(10.0 ** ((float(amp_db) - self.amp_median) / 20.0), 0.02, 20.0))


class TuringPDWEnv:
    """One-band-per-step environment replaying a contiguous TSRD PDW window."""

    def __init__(
        self,
        source: TuringPulseTrain | str | Path,
        *,
        episode_length: int = 150,
        step_us: float = 1_000.0,
    ):
        self.source = source if isinstance(source, TuringPulseTrain) else TuringPulseTrain(source)
        self.num_bands = self.source.num_bands
        self.episode_length = int(episode_length)
        self.step_us = float(step_us)
        if self.episode_length <= 0 or self.step_us <= 0:
            raise ValueError("episode_length and step_us must be positive")

        self.current_step = 0
        self.start_us = float(self.source.toa_us[0])
        self.end_us = self.start_us + self.episode_length * self.step_us
        self.last_action: int | None = None

    def reset(self, *, seed: int | None = None):
        rng = np.random.default_rng(seed)
        window_us = self.episode_length * self.step_us
        lo = float(self.source.toa_us[0])
        hi = float(self.source.toa_us[-1])
        if hi - lo > window_us:
            self.start_us = float(rng.uniform(lo, hi - window_us))
        else:
            self.start_us = lo
            # If a tiny fixture is shorter than requested, fit the full train.
            self.step_us = max((hi - lo) / max(1, self.episode_length), 1e-6)
        self.end_us = self.start_us + self.episode_length * self.step_us
        self.current_step = 0
        self.last_action = None
        return self._empty_observation(0), {"source": str(self.source.path), "pdw_native": True}

    def _empty_observation(self, band: int) -> dict:
        return {
            "selected_band": np.int64(band),
            "detected": np.int64(0),
            "signal_power": np.array([0.02], dtype=np.float32),
            "quality": np.array([0.0], dtype=np.float32),
            "pdw_count": np.int64(0),
            "pdw_summary": np.zeros(5, dtype=np.float32),
        }

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.num_bands:
            raise ValueError(f"Band action {action} outside 0..{self.num_bands - 1}")

        t0 = self.start_us + self.current_step * self.step_us
        t1 = t0 + self.step_us
        left = int(np.searchsorted(self.source.toa_us, t0, side="left"))
        right = int(np.searchsorted(self.source.toa_us, t1, side="left"))

        bands = self.source.bands[left:right]
        local = np.flatnonzero(bands == action)
        detected = bool(local.size)
        total_pulses = int(right - left)
        active_bands = np.unique(bands).astype(int).tolist() if total_pulses else []

        if detected:
            idx = left + local
            amps = self.source.amplitude_db[idx]
            strongest_local = int(np.argmax(amps))
            strongest_idx = int(idx[strongest_local])
            amp = float(self.source.amplitude_db[strongest_idx])
            quality = self.source.amplitude_quality(amp)
            signal_power = self.source.relative_power(amp)
            pdw_count = int(local.size)
            pdw_summary = np.array(
                [
                    float(self.source.toa_us[strongest_idx] - t0),
                    float(self.source.frequency_mhz[strongest_idx]),
                    float(self.source.pulse_width_us[strongest_idx]),
                    float(self.source.aoa_deg[strongest_idx]),
                    amp,
                ],
                dtype=np.float32,
            )
            captured_emitters = np.unique(self.source.labels[idx]).astype(int).tolist()
        else:
            quality = 0.0
            signal_power = 0.02
            pdw_count = 0
            pdw_summary = np.zeros(5, dtype=np.float32)
            captured_emitters = []

        obs = {
            "selected_band": np.int64(action),
            "detected": np.int64(detected),
            "signal_power": np.array([signal_power], dtype=np.float32),
            "quality": np.array([quality], dtype=np.float32),
            "pdw_count": np.int64(pdw_count),
            "pdw_summary": pdw_summary,
        }

        # Preserve the existing SmartScan reward semantics for apples-to-apples
        # scheduler comparison: +1 for a true selected-band hit, +0.1 for a quiet
        # selected band.  Pulse-capture rate is reported separately and is a more
        # direct TSRD metric.
        reward = 1.0 if detected else 0.1

        info = {
            "source": str(self.source.path),
            "pdw_native": True,
            "true_signal_present": detected,
            "ground_truth_active_bands": active_bands,
            "ground_truth_pulses_this_step": total_pulses,
            "captured_pulses": pdw_count,
            "captured_emitters": captured_emitters,
            "time_window_us": (float(t0), float(t1)),
        }

        self.last_action = action
        self.current_step += 1
        truncated = self.current_step >= self.episode_length
        return obs, float(reward), False, truncated, info
