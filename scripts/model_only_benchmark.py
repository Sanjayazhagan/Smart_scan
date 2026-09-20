"""Single-file model-only benchmark for a labeled pulse-stream dataset.

Usage:
    python scripts/model_only_benchmark.py --dataset path/to/tagging_phase.h5

Expected HDF5 keys:
    data: N x 5 numeric pulse descriptors [time, frequency, pulse width, angle, power]
    labels: N emitter labels (used only to describe the dataset; scoring uses pulse presence)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from scheduler.baselines import FixedScheduler, RandomScheduler, ThompsonSamplingScheduler, UCB1Scheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.smartscan_production import SmartScanProductionScheduler


class PulseBenchmark:
    def __init__(self, path: Path, bands: int, steps: int):
        with h5py.File(path, "r") as handle:
            data = np.asarray(handle["data"][:], dtype=np.float64)
            labels = np.asarray(handle["labels"][:]).reshape(-1)

        if data.ndim != 2 or data.shape[1] < 5:
            raise ValueError("data must be an N x 5 pulse descriptor table")
        if len(data) != len(labels):
            raise ValueError("data and labels must have the same number of rows")

        times = data[:, 0]
        frequencies = data[:, 1]
        powers = data[:, 4]
        valid = np.isfinite(times) & np.isfinite(frequencies) & np.isfinite(powers)
        times, frequencies, powers = times[valid], frequencies[valid], powers[valid]
        if len(times) == 0:
            raise ValueError("dataset contains no finite pulses")

        time_edges = np.linspace(times.min(), times.max() + 1e-9, steps + 1)
        freq_edges = np.linspace(frequencies.min(), frequencies.max() + 1e-9, bands + 1)
        pulse_steps = np.clip(np.digitize(times, time_edges) - 1, 0, steps - 1)
        pulse_bands = np.clip(np.digitize(frequencies, freq_edges) - 1, 0, bands - 1)

        self.steps = steps
        self.bands = bands
        self.active = np.zeros((steps, bands), dtype=bool)
        self.power = np.zeros((steps, bands), dtype=np.float32)
        for step, band, power in zip(pulse_steps, pulse_bands, powers):
            self.active[step, band] = True
            self.power[step, band] = max(self.power[step, band], float(max(0.0, power)))
        self.metadata = {
            "file": str(path),
            "pulses": int(len(times)),
            "emitters": int(len(np.unique(labels[valid]))),
            "time_slots": steps,
            "bands": bands,
            "schema": "data[N,5]=time,frequency,pulse_width,angle,power; labels[N]",
        }

    def run(self, factory, seed: int) -> dict:
        scheduler = factory(seed)
        reward = 0.0
        hits = 0
        opportunities = int(self.active.sum())
        switches = 0
        previous_band = None
        latencies = []

        for step in range(self.steps):
            started = time.perf_counter()
            band = int(scheduler.select_band())
            latencies.append((time.perf_counter() - started) * 1000.0)
            active = bool(self.active[step, band])
            detected = active
            signal_power = float(self.power[step, band]) if detected else 0.0
            current_reward = 1.0 if active else 0.0
            if previous_band is not None and previous_band != band:
                switches += 1
                current_reward -= 0.08 * abs(band - previous_band) / max(1, self.bands - 1)
            observation = {
                "selected_band": band,
                "detected": detected,
                "quality": np.array([1.0 if detected else 0.0], dtype=np.float32),
                "signal_power": np.array([signal_power], dtype=np.float32),
            }
            scheduler.update(band, current_reward, observation)
            reward += current_reward
            hits += int(active)
            previous_band = band

        return {
            "reward": reward,
            "hits": hits,
            "opportunities": opportunities,
            "interception_rate": 100.0 * hits / max(1, opportunities),
            "switches": switches,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark = PulseBenchmark(args.dataset, args.bands, args.steps)
    factories = {
        "Sequential Sweep": lambda seed: FixedScheduler(args.bands),
        "Random": lambda seed: RandomScheduler(args.bands, seed=seed),
        "UCB1": lambda seed: UCB1Scheduler(args.bands),
        "Thompson": lambda seed: ThompsonSamplingScheduler(args.bands, seed=seed),
        "NMF-only": lambda seed: NMFScheduler(args.bands, recompute_every=2),
        "Smart Scan": lambda seed: SmartScanProductionScheduler(args.bands, seed=seed),
    }

    print("=" * 112)
    print("SMART SCAN MODEL-ONLY BENCHMARK")
    print("=" * 112)
    print(f"Dataset: {benchmark.metadata['file']}")
    print(f"Schema:  {benchmark.metadata['schema']}")
    print(f"Pulses: {benchmark.metadata['pulses']:,} | Labeled emitters: {benchmark.metadata['emitters']} | Slots: {args.steps} | Bands: {args.bands}")
    print("Scoring: same pulse-derived active-band grid, same seed, same switching cost for every algorithm.")
    print("Note: labels are descriptive; hidden pulse presence is used only for post-action scoring.")
    print("-" * 112)
    print(f"{'Algorithm':<22} {'Reward':>11} {'Hits':>8} {'Opportunities':>14} {'Intercept %':>13} {'Switches':>10} {'Mean ms':>10} {'P95 ms':>10}")
    print("-" * 112)

    results = []
    for name, factory in factories.items():
        result = benchmark.run(factory, args.seed)
        results.append((name, result))
        print(f"{name:<22} {result['reward']:>11.2f} {result['hits']:>8d} {result['opportunities']:>14d} {result['interception_rate']:>12.2f}% {result['switches']:>10d} {result['mean_latency_ms']:>10.3f} {result['p95_latency_ms']:>10.3f}")

    baseline = next(result for name, result in results if name == "Sequential Sweep")
    champion = next(result for name, result in results if name == "Smart Scan")
    print("-" * 112)
    print(f"Smart Scan vs Sequential Sweep: reward {champion['reward'] - baseline['reward']:+.2f}, interception {champion['interception_rate'] - baseline['interception_rate']:+.2f} percentage points, switches {baseline['switches'] - champion['switches']:+d} fewer")
    print("=" * 112)


if __name__ == "__main__":
    main()
