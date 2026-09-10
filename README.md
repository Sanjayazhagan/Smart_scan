# SmartScan Production Champion Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pure NumPy](https://img.shields.io/badge/dependencies-NumPy%20only-brightgreen.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Latency](https://img.shields.io/badge/latency-%3C0.15ms-success.svg)](#performance-figures-of-merit)

**DRDO / IDEX / Department of Defence Production // SIH Problem Statement ID: 26055**  
*Smart Scan Strategy for Electronic Warfare in the absence of prior reliable intelligence of emitters and their operating characteristics.*

---

## 🎯 Executive Overview

In Electronic Warfare (EW), surveillance receivers with high sensitivity have **instantaneous bandwidths at least an order of magnitude lower** than the overall surveillance spectrum, mandating dynamic frequency sweeping. Conventional open-loop sweeping wastes valuable dwell time on nonthreatening or empty channels.

This repository contains the standalone **SmartScan Production Champion Model** (`SmartScanProductionScheduler` / `DwellDualPolicyScheduler`). It is an autonomous, closed-loop cognitive interceptor that rapidly detects, tracks, and classifies agile emitters without requiring pre-mission intelligence.

### 🌟 Key Characteristics
- **Pure NumPy Implementation**: Zero PyTorch, TensorFlow, or GPU dependencies.
- **Ultra-Low Latency**: **~0.08 ms – 0.15 ms** execution per decision (well within the $\le 1.0\text{ ms}$ real-time avionics constraint).
- **Observation-Driven Only**: Operates directly on Pulse Descriptor Word (PDW) feedback (`toa_us`, `frequency_mhz`, `pulse_width_us`, `aoa_deg`, `amplitude_db`) with **zero pre-mission intelligence**.
- **Synthesizer Conservation**: Adaptive dwell lock reduces retuning switching overhead by **38%**.

---

## 🔬 Core Architecture

```text
Incoming PDW Stream
        │
        ▼
   Recurring Stale Coverage?
        │
        ├─ Yes ──► Current band actively detected?
        │             ├─ Yes: DWELL LOCK (Defer stale visit, protect active pulse)
        │             └─ No:  SMART STALE RANKING (Service highest observable urgency)
        │
        └─ No  ──► DWELL-DUAL ARBITRATION
                      ├─ Active Emitter Intercept: Dwell Lock (Zero retuning penalty)
                      ├─ High Uncertainty: Cognitive Scout (Probe unvisited spectrum)
                      └─ Nominal State: NMF Exploit (Spectral co-occurrence forecast)
```

1. **Interruptible Dwell Lock**: Locks receiver antenna position on intercepted signals to eliminate synthesizer retuning penalties. A 1-step fading grace window debounces momentary signal drops.
2. **Smart Stale Coverage**: Prioritizes unvisited channels using an observable multi-factor urgency index ($0.45 \cdot \text{Uncertainty} + 0.25 \cdot \text{Age} + 0.20 \cdot \text{Stale Depth} + 0.10 \cdot \text{Prior Value} - \text{Switch Cost}$). Crucially, active dwells are never prematurely abandoned.
3. **NMF Spectral Co-Activation**: Online Non-Negative Matrix Factorization (Lee & Seung multiplicative updates) discovers correlated multi-channel emitter hopping patterns on the fly.

---

## 📊 Performance Figures of Merit

| Figure of Merit (FoM) | Open-Loop / Baselines | SmartScan Champion | Improvement |
|---|---|---|---|
| **Interception Probability** | 5.8% – 14.2% | **23.0%** (Peak ~28%) | **+62% to +296%** intercept gain |
| **Cumulative Reward** | -42.8 to +9.70 | **+19.34 [17.22, 21.47]** | **100% of 17 baselines sub-optimal ($p < 0.05$)** |
| **Control Latency** | 0.45 – 28.5 ms | **0.08 ms – 0.14 ms** | **12x faster** than 1.0 ms avionics deadline |
| **Synthesizer Switches** | 135 – 150 / ep | **70.9 switches / ep** | **-38%** retuning overhead |
| **Pre-Mission Prior** | Mandatory | **None (Zero Prior)** | **100% autonomous compliance** |

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the model-only branch
git clone -b model-only https://github.com/Sanjayazhagan/Smart_scan.git
cd Smart_scan

# Install minimal NumPy requirements
pip install -r requirements.txt
```

### 2. Run Interactive Demo

```bash
python demo.py
```

### 3. Run Verification Tests

```bash
python -m pytest tests/test_model.py -v
```

---

## 💻 3-Line API Integration

```python
from scheduler import SmartScanProductionScheduler

# 1. Initialize for 20 RF channels
scheduler = SmartScanProductionScheduler(num_bands=20, seed=42)

# 2. Select next frequency band to tune
target_band = scheduler.select_band()

# 3. Feed receiver observation back to scheduler
observation = {
    "detected": True,                # True if pulse intercepted
    "quality": [0.92],               # Pulse SNR / classification confidence
    "signal_power": [2.45],          # Measured channel power
}
scheduler.update(target_band, reward=1.0, obs_dict=observation)
```

---

## 📦 Repository Structure

```text
Smart_scan (branch: model-only)
├── demo.py                             # Standalone interactive quickstart demo
├── requirements.txt                    # Minimal NumPy & PyTest dependencies
├── README.md                           # Model documentation and specifications
│
├── scheduler/                          # Core model package
│   ├── __init__.py                     # Package export
│   ├── smartscan_production.py         # Champion scheduler implementation
│   ├── observation_runtime.py          # Observation-only runtime adapter
│   ├── dual_policy_uncertainty_scheduler.py # Multi-horizon uncertainty estimator
│   ├── world_model_ucb.py              # Discounted UCB base policy
│   ├── baselines.py                    # Base scheduler interfaces
│   ├── track2_core.py                  # RF band constants
│   ├── stats_compat.py                 # Pure NumPy statistics compatibility
│   └── paradigms/
│       └── matrix_factorization.py     # Online NMF decomposition
│
└── tests/
    └── test_model.py                   # Automated unit test suite
```
