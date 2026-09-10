"""Generate a generic TSRD-schema PDW dataset for software ablation tests.

This is NOT the Alan Turing Institute dataset and must not be reported as TSRD
performance.  It only mirrors the simple HDF5 schema: data[N,5] + labels[N].
Columns: ToA(us), centre frequency(MHz), pulse width(us), AoA(deg), amplitude(dB).
"""
from __future__ import annotations

from pathlib import Path
import json
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "synthetic_turing_style"
OUT.mkdir(parents=True, exist_ok=True)

DURATION_US = 8_000_000.0
NUM_BANDS = 20
BAND_W = 18_000.0 / NUM_BANDS
RNG = np.random.default_rng(20260910)


def band_center(b: int) -> float:
    return (b + 0.5) * BAND_W


def add_pulse(rows, labels, toa, freq, pw, aoa, amp, emitter):
    if 0 <= toa < DURATION_US:
        rows.append([toa, freq, pw, aoa, amp])
        labels.append(emitter)


def add_train(rows, labels, emitter, band, start, end, pri, *, amp=-50, aoa=30, pw=2.0,
              freq_jitter=10.0, pri_jitter=0.03):
    t = float(start)
    while t < end:
        freq = band_center(band) + RNG.normal(0, freq_jitter)
        add_pulse(rows, labels, t, freq, max(0.2, pw + RNG.normal(0, 0.08)),
                  aoa + RNG.normal(0, 1.5), amp + RNG.normal(0, 2.0), emitter)
        t += max(50.0, pri * (1.0 + RNG.normal(0, pri_jitter)))


def add_bursty(rows, labels, emitter, band, pri, burst_on, burst_off, *, phase=0,
               amp=-50, aoa=30, pw=2.0):
    t = float(phase)
    while t < DURATION_US:
        add_train(rows, labels, emitter, band, t, min(t + burst_on, DURATION_US), pri,
                  amp=amp, aoa=aoa, pw=pw)
        t += burst_on + burst_off


def add_hopper(rows, labels, emitter, bands, pri, hop_every_us, *, amp=-50, aoa=30, pw=2.0, phase=0):
    t = float(phase)
    k = 0
    while t < DURATION_US:
        seg_end = min(t + hop_every_us, DURATION_US)
        add_train(rows, labels, emitter, bands[k % len(bands)], t, seg_end, pri,
                  amp=amp, aoa=aoa, pw=pw, freq_jitter=7.0)
        t = seg_end
        k += 1


def add_drifter(rows, labels, emitter, band_start, band_end, pri, *, amp=-53, aoa=90, pw=1.5):
    t = 0.0
    f0, f1 = band_center(band_start), band_center(band_end)
    while t < DURATION_US:
        frac = t / DURATION_US
        freq = f0 + frac * (f1 - f0) + RNG.normal(0, 6.0)
        add_pulse(rows, labels, t, freq, max(0.2, pw + RNG.normal(0, .05)),
                  aoa + RNG.normal(0, 1.0), amp + RNG.normal(0, 1.5), emitter)
        t += max(80.0, pri * (1.0 + RNG.normal(0, 0.02)))


def write_case(name, builder, description):
    rows, labels = [], []
    builder(rows, labels)
    data = np.asarray(rows, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int32)
    order = np.argsort(data[:, 0], kind="stable")
    data, lab = data[order], lab[order]
    path = OUT / f"stare_generic_{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("labels", data=lab, compression="gzip")
        f.attrs["provenance"] = "OpenAI generic synthetic PDW fixture; NOT TSRD"
        f.attrs["description"] = description
    return {"name": name, "path": str(path), "pulses": int(len(data)), "emitters": int(len(np.unique(lab))), "description": description}


cases = []


def steady(rows, labels):
    add_bursty(rows, labels, 0, 2, 700, 120_000, 80_000, phase=0, amp=-46, aoa=20)
    add_bursty(rows, labels, 1, 9, 1100, 180_000, 120_000, phase=35_000, amp=-54, aoa=75)
    add_bursty(rows, labels, 2, 16, 1600, 220_000, 180_000, phase=70_000, amp=-50, aoa=135)

cases.append(write_case("steady", steady, "Three persistent bursty emitters on separated bands."))


def hopping(rows, labels):
    add_hopper(rows, labels, 0, [3, 7, 11, 15], 850, 65_000, amp=-48, aoa=40, phase=0)
    add_bursty(rows, labels, 1, 13, 1350, 140_000, 160_000, phase=20_000, amp=-56, aoa=115)
    add_bursty(rows, labels, 2, 5, 950, 90_000, 210_000, phase=90_000, amp=-51, aoa=165)

cases.append(write_case("hopping", hopping, "One periodic hopper plus two bursty fixed-band emitters."))


def intermittent(rows, labels):
    specs = [(1, 650, 22_000, 180_000), (6, 900, 35_000, 260_000), (10, 1250, 45_000, 330_000), (14, 1500, 55_000, 410_000), (18, 1050, 28_000, 230_000)]
    for e, (b, pri, on, off) in enumerate(specs):
        add_bursty(rows, labels, e, b, pri, on, off, phase=e*37_000, amp=-45-e*2, aoa=20+e*30)

cases.append(write_case("intermittent", intermittent, "Five sparse short-burst emitters with different duty cycles."))


def crowded(rows, labels):
    for e, b in enumerate([1, 3, 5, 8, 10, 12, 15, 18]):
        add_bursty(rows, labels, e, b, 650 + 110*e, 150_000 + 8_000*e, 70_000 + 14_000*(e%3),
                   phase=11_000*e, amp=-46-(e%4)*3, aoa=15+20*e, pw=1.0+0.15*(e%4))

cases.append(write_case("crowded", crowded, "Eight overlapping bursty emitters distributed across the band set."))


def drifting(rows, labels):
    add_drifter(rows, labels, 0, 2, 7, 950, amp=-49, aoa=35)
    add_drifter(rows, labels, 1, 15, 10, 1250, amp=-54, aoa=125)
    add_bursty(rows, labels, 2, 17, 1500, 180_000, 180_000, phase=40_000, amp=-47, aoa=170)

cases.append(write_case("drifting", drifting, "Two slowly frequency-drifting emitters plus one fixed bursty emitter."))


def mixed(rows, labels):
    add_hopper(rows, labels, 0, [2, 6, 10, 14, 18], 780, 48_000, amp=-49, aoa=25)
    add_drifter(rows, labels, 1, 4, 9, 1100, amp=-53, aoa=95)
    add_bursty(rows, labels, 2, 12, 700, 75_000, 125_000, phase=15_000, amp=-45, aoa=145)
    add_bursty(rows, labels, 3, 16, 1450, 240_000, 160_000, phase=95_000, amp=-57, aoa=175)
    add_bursty(rows, labels, 4, 0, 1000, 35_000, 365_000, phase=125_000, amp=-50, aoa=60)

cases.append(write_case("mixed", mixed, "Combined hopping, drifting, persistent, and sparse activity."))

manifest = {
    "warning": "Generic synthetic software-test dataset. NOT Alan Turing Institute TSRD.",
    "schema": ["toa_us", "frequency_mhz", "pulse_width_us", "aoa_deg", "amplitude_db"],
    "duration_us": DURATION_US,
    "num_bands": NUM_BANDS,
    "generator_seed": 20260910,
    "cases": cases,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
