"""Grand Statistical Benchmark across 20 Alan Turing Institute Radar Datasets.

Evaluates Dwell-Dual against classical baselines across Archive, Scan, and Stare
radar operating modes.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import h5py
import numpy as np

from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
from benchmark_models.turing_calibrated_dwell_dual import TuringCalibratedDwellDualPolicyScheduler
from benchmark_models.mathematical_baselines import FixedScheduler, NMFScheduler, RandomScheduler
from scheduler.track2_runtime import DEFAULT_MODEL_PATH


class DummyRuntime:
    """Mock Track 2 runtime since Dwell-Dual sets world_model_scale=0.0."""
    def update(self, *args, **kwargs): pass
    def reset(self): pass
    def get_forecast(self, *args, **kwargs): return np.zeros(20)


class TuringMissionEnvironment:
    """Discretizes a Turing Institute HDF5 pulse stream into a 20-band tactical EW environment."""

    def __init__(self, h5_path: Path, num_bands: int = 20, episode_length: int = 300):
        self.num_bands = num_bands
        self.episode_length = episode_length
        self.h5_path = h5_path

        with h5py.File(h5_path, "r") as f:
            data = f["data"][:]
            labels = f["labels"][:].flatten()

        utc_time = data[:, 0]
        rf_mhz = data[:, 1]
        pa_dbm = data[:, 4]

        # Use first 35% of time span
        t_span = utc_time.max() - utc_time.min()
        t_max = utc_time.min() + t_span * 0.35 if t_span > 0 else utc_time.max()

        valid_idx = (utc_time >= utc_time.min()) & (utc_time <= t_max)
        if not np.any(valid_idx):
            valid_idx = np.ones(len(utc_time), dtype=bool)

        times = utc_time[valid_idx]
        rfs = rf_mhz[valid_idx]
        pas = pa_dbm[valid_idx]
        emitters = labels[valid_idx]

        # 20 frequency bands
        rf_min, rf_max = float(rfs.min()), float(rfs.max())
        band_edges = np.linspace(rf_min, rf_max + 1e-3, num_bands + 1)
        pulse_bands = np.clip(np.digitize(rfs, band_edges) - 1, 0, num_bands - 1)

        # Time slots
        t_start, t_end = float(times.min()), float(times.max())
        slot_edges = np.linspace(t_start, t_end + 1e-3, episode_length + 1)
        pulse_slots = np.clip(np.digitize(times, slot_edges) - 1, 0, episode_length - 1)

        # Ground truth matrix
        self.ground_truth = np.zeros((episode_length, num_bands), dtype=bool)
        self.power = np.full((episode_length, num_bands), -150.0, dtype=np.float32)

        for slot, band, pwr in zip(pulse_slots, pulse_bands, pas):
            self.ground_truth[slot, band] = True
            if pwr > self.power[slot, band]:
                self.power[slot, band] = pwr

        self.step_idx = 0
        self.last_band = None

    def reset(self):
        self.step_idx = 0
        self.last_band = None

    def step(self, band: int):
        is_active = self.ground_truth[self.step_idx, band]
        pwr = self.power[self.step_idx, band]

        detected = bool(is_active and pwr > -125.0)
        quality = float(np.clip((pwr + 120.0) / 40.0, 0.0, 1.0)) if detected else 0.0

        obs = {
            "selected_band": band,
            "detected": detected,
            # NMF consumes non-negative signal strength; convert dBm-like PDW
            # power into a non-negative level while retaining raw power for quality.
            "signal_power": np.array([max(0.0, pwr) if detected else 0.0], dtype=np.float32),
            "quality": np.array([quality], dtype=np.float32),
        }
        if detected:
            t = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
            envelope = np.exp(-0.5 * (t / 0.45) ** 2)
            phase = 2.0 * np.pi * (band - 9.5) * 0.1 * t
            amplitude = max(0.4, quality)
            obs["iq"] = np.stack(
                [amplitude * envelope * np.cos(phase), amplitude * envelope * np.sin(phase)]
            ).astype(np.float32)

        # Scoring
        if is_active and detected:
            reward = 1.0
        elif not is_active and not detected:
            reward = 0.1
        else:
            reward = -1.0

        if self.last_band is not None and self.last_band != band:
            reward -= 0.08 * (abs(band - self.last_band) / (self.num_bands - 1))

        self.last_band = band
        self.step_idx += 1
        done = self.step_idx >= self.episode_length
        return obs, reward, done, {"is_active": is_active, "detected": detected}


def run_grand_tournament():
    print("=" * 95)
    print("   ALAN TURING INSTITUTE: 20-MISSION GRAND ELECTRONIC WARFARE TOURNAMENT")
    print("=" * 95)

    base_dir = Path("turing_dataset")
    mission_files = sorted(list(base_dir.rglob("*.h5")))
    print(f"Discovered {len(mission_files)} real radar operational mission files across splits.")
    print("Models: Sequential Sweep | Uniform Random | Direct NMF | Dwell-Dual Policy (Champion)\n")

    model_factories = {
        "Sequential Sweep": lambda: FixedScheduler(num_bands=20),
        "Uniform Random": lambda: RandomScheduler(num_bands=20, seed=42),
        "Direct NMF": lambda: NMFScheduler(num_bands=20),
        "Dwell-Dual Policy": lambda: DwellDualPolicyScheduler(num_bands=20, seed=42, runtime=DummyRuntime()),
        "Turing-Calibrated Dwell-Dual": lambda: TuringCalibratedDwellDualPolicyScheduler(num_bands=20, seed=42, runtime=DummyRuntime()),
    }
    if DEFAULT_MODEL_PATH.is_file():
        model_factories["Champion + CNN Gate"] = lambda: DwellDualPolicyScheduler(
            num_bands=20,
            seed=42,
            cnn_signal_gate=True,
            model_path=DEFAULT_MODEL_PATH,
        )

    model_scores = {name: [] for name in model_factories}
    model_hits = {name: [] for name in model_factories}
    model_switches = {name: [] for name in model_factories}

    for idx, fpath in enumerate(mission_files, 1):
        rel = fpath.relative_to(base_dir)
        env = TuringMissionEnvironment(fpath, num_bands=20, episode_length=300)
        total_active = env.ground_truth.sum()

        mission_results = {}
        for name, factory in model_factories.items():
            sched = factory()
            env.reset()
            cum_reward = 0.0
            hits = 0
            switches = 0
            last_b = None

            for s in range(env.episode_length):
                b = sched.select_band()
                obs, r, d, info = env.step(b)
                sched.update(b, r, obs)

                cum_reward += r
                if info["is_active"] and info["detected"]:
                    hits += 1
                if last_b is not None and last_b != b:
                    switches += 1
                last_b = b
                if d: break

            model_scores[name].append(cum_reward)
            model_hits[name].append(hits)
            model_switches[name].append(switches)
            mission_results[name] = (cum_reward, hits)

        champ_rew, champ_hit = mission_results["Dwell-Dual Policy"]
        cnn_rew, cnn_hit = mission_results.get("Champion + CNN Gate", (None, None))
        sweep_rew, sweep_hit = mission_results["Sequential Sweep"]
        cnn_text = "" if cnn_rew is None else f" | CNN Gate: {cnn_rew:>+6.1f} rew ({cnn_hit:>2} hits)"
        print(f"Mission {idx:02d}/20 [{str(rel):<25}] -> Dwell-Dual: {champ_rew:>+6.1f} rew ({champ_hit:>2} hits) | Sweep: {sweep_rew:>+6.1f} rew ({sweep_hit:>2} hits){cnn_text}")

    print("\n" + "=" * 95)
    print("   FINAL GRAND LEADERBOARD ACROSS ALL 20 ALAN TURING MISSIONS")
    print("=" * 95)
    print(f"| {'Rank':<4} | {'Model Name':<25} | {'Mean Reward':<12} | {'Total Hits':<11} | {'Avg Switches':<13} |")
    print("+" + "-" * 93 + "+")

    summary = []
    for name in model_factories:
        m_rew = np.mean(model_scores[name])
        t_hits = int(np.sum(model_hits[name]))
        m_sw = np.mean(model_switches[name])
        summary.append((m_rew, t_hits, m_sw, name))

    summary.sort(reverse=True, key=lambda x: x[0])

    for rank, (m_rew, t_hits, m_sw, name) in enumerate(summary, 1):
        win_str = "#1 [WIN]" if rank == 1 else f"#{rank}     "
        print(f"| {win_str:<8} | {name:<25} | {m_rew:>+11.2f} | {t_hits:>10} | {m_sw:>12.1f} |")

    print("+" + "-" * 93 + "+")


if __name__ == "__main__":
    run_grand_tournament()
