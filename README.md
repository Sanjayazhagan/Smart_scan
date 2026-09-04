# SmartScan: Cognitive Radio Dynamic Scan Scheduler

SmartScan is an intelligent, partially observable RF spectrum scan-scheduling engine. A single-channel cognitive radio receiver must select one of 20 frequency bands at each time step (150 steps/episode) to detect agile radar emitters, frequency-hopping communications, and hostile electronic warfare interference while minimizing channel retuning switching penalties ($c_{\text{switch}} = 0.08$).

---

## 🏆 Project Grand Champion: Dwell-Dual Policy

Across an exhaustive 15-seed master benchmark (1,260 episodes, 189,000 live decisions) spanning 6 operational scenarios, **Dwell-Dual Policy** emerged as the decisive #1 architecture:

- **Mean Cumulative Reward**: **+17.72 $\pm$ 4.12**
- **Signal Hit Rate**: **14.7%** (Highest among all architectures)
- **Decision Latency**: **0.15 ms** (Strictly bounded under 2.0 ms control loop deadline)
- **Switching Profile**: 88.4 switches per episode (~41% dwell ratio)
- **Key Files**:
  - Production Entrypoint: [`scheduler/smartscan_production.py`](scheduler/smartscan_production.py)
  - Modular Benchmark Implementation: [`benchmark_models/dwell_dual_policy.py`](benchmark_models/dwell_dual_policy.py)

---

## 📊 Master Benchmark Leaderboard (14 Models Ranked)

| Rank | Architecture / Model | Mean Reward (95% CI) | Hit Rate | Switches | Latency | Category | Code File |
|:---:|---|:---:|:---:|:---:|:---:|---|---|
| **01** | **Dwell-Dual Policy (Champion)** | **+17.72** [$\pm 4.12$] | **14.7%** | 88.4 | 0.150 ms | **Grand Champion** | [`benchmark_models/dwell_dual_policy.py`](benchmark_models/dwell_dual_policy.py) |
| **02** | **SmartScan V2-NMF** | **+18.04** [$\pm 5.30$] | 14.5% | 50.6 | 1.900 ms | Expectimax Search | [`benchmark_models/nmf_expectimax.py`](benchmark_models/nmf_expectimax.py) |
| **03** | **Dual-Policy Uncertainty** | **+17.65** [$\pm 4.08$] | **15.1%** | 103.5 | 0.140 ms | Dual-Mode Bandit | [`benchmark_models/dual_policy_uncertainty.py`](benchmark_models/dual_policy_uncertainty.py) |
| **04** | **SmartScan-Omni V2 (Tuned)** | **+16.88** [$\pm 4.25$] | 14.1% | 22.7 | 1.850 ms | Lookahead / EW | [`benchmark_models/smartscan_omni.py`](benchmark_models/smartscan_omni.py) |
| **05** | **Robust PCA + PSR** | **+16.16** [$\pm 4.67$] | 12.3% | 31.2 | **0.012 ms** | Embedded Champion | [`benchmark_models/robust_pca_psr.py`](benchmark_models/robust_pca_psr.py) |
| **06** | **Candidate 1: LinUCB Bandit** | **+15.52** [$\pm 3.64$] | 12.7% | 73.3 | 0.090 ms | Contextual Bandit | [`benchmark_models/contextual_bandit.py`](benchmark_models/contextual_bandit.py) |
| **07** | **Candidate 3: GRU-D Recurrent** | **+15.17** [$\pm 3.78$] | 11.9% | 91.0 | **0.012 ms** | Deep Recurrent Network | [`benchmark_models/grud_recurrent.py`](benchmark_models/grud_recurrent.py) |
| **08** | **Direct NMF** | **+16.05** [$\pm 5.14$] | 14.1% | **10.2** | **0.012 ms** | Matrix Factorization | [`benchmark_models/mathematical_baselines.py`](benchmark_models/mathematical_baselines.py) |
| **09** | **Static NMF+UCB** | **+16.88** [$\pm 4.16$] | 15.0% | 98.1 | 1.350 ms | Heuristic Fusion | [`benchmark_models/mathematical_baselines.py`](benchmark_models/mathematical_baselines.py) |
| **10** | **Whittle Index RMAB** | **+13.86** [$\pm 3.28$] | 12.6% | 120.3 | 0.027 ms | Mathematical Scheduling | [`benchmark_models/mathematical_baselines.py`](benchmark_models/mathematical_baselines.py) |
| **11** | **Candidate 2: Boosted-Tree Dwell** | **+11.66** [$\pm 3.39$] | 8.8% | 31.4 | 0.070 ms | Boosted Tree | [`benchmark_models/boosted_tree_dwell.py`](benchmark_models/boosted_tree_dwell.py) |
| **12** | **Observable Plain UCB** | **+13.09** [$\pm 3.29$] | 10.8% | 143.5 | 0.005 ms | Discounted Bandit | [`benchmark_models/mathematical_baselines.py`](benchmark_models/mathematical_baselines.py) |
| **13** | **Uniform Random Scan** | **+10.56** [$\pm 2.52$] | 8.6% | 139.4 | 0.002 ms | Uniform Baseline | [`benchmark_models/mathematical_baselines.py`](benchmark_models/mathematical_baselines.py) |
| **14** | **Fixed Sequential Sweep** | **+8.99** [$\pm 2.15$] | 8.2% | 149.0 | 0.001 ms | Raster Baseline | [`benchmark_models/mathematical_baselines.py`](benchmark_models/mathematical_baselines.py) |

---

## 🗂️ Clean Modular Organization for Developers

Every model is isolated in its own documented file under `benchmark_models/` with theoretical documentation, mathematical equations, configuration parameters, and benchmark scores:

```text
SMART SCAN/
├── MASTER_BENCHMARK.py           # Central CLI benchmark runner & model inspector
├── MASTER_BENCHMARK_REPORT.md    # Master documentation report across all architectures
│
├── benchmark_models/             # Decoupled model files sorted by architecture
│   ├── __init__.py               # Unified model registry & factory loader
│   ├── dwell_dual_policy.py      # Model 01: Dwell-Dual Policy (Grand Champion)
│   ├── smartscan_omni.py         # Model 02: SmartScan-Omni V2 (Lookahead & EW Specialist)
│   ├── robust_pca_psr.py         # Model 03: Robust PCA + PSR (Embedded Champion)
│   ├── contextual_bandit.py      # Model 04: LinUCB Policy Meta-Arbitrator
│   ├── grud_recurrent.py         # Model 05: Missing-Data GRU-D Spectrum Model
│   ├── boosted_tree_dwell.py     # Model 06: Boosted-Tree Adaptive Dwell (GBDT)
│   ├── nmf_expectimax.py         # Model 07: SmartScan V2-NMF (Lookahead Search)
│   ├── dual_policy_uncertainty.py# Model 08: Dual-Policy Uncertainty Scheduler
│   └── mathematical_baselines.py # Models 09-14: NMF, RMAB, UCB, Random, Sweep
│
├── scheduler/
│   ├── smartscan_production.py   # Isolated production champion deployment
│   └── track2_runtime.py         # Perceptual world model interface
│
├── simulator/
│   ├── environment.py            # Gymnasium RF environment
│   └── scenarios.py              # Operational defense scenario generator
│
└── tests/                        # Full automated test suite
```

---

## 🚀 Quickstart & Verification

### 1. Installation & Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. View Benchmark Leaderboard
```powershell
python MASTER_BENCHMARK.py
```

### 3. Inspect Architecture of Any Model
```powershell
python MASTER_BENCHMARK.py --info dwell
python MASTER_BENCHMARK.py --info omni
python MASTER_BENCHMARK.py --info linucb
python MASTER_BENCHMARK.py --info grud
```

### 4. Run Grand Champion Live Telemetry Demo
```powershell
python MASTER_BENCHMARK.py --champion
```

### 5. Run Repository Tests
```powershell
python -m pytest tests/ -v
```

---

## 📑 Detailed Documentation

For a comprehensive deep-dive into each model's theory, scenario-by-scenario analysis, and paired statistical hypothesis testing ($p$-values), see [MASTER_BENCHMARK_REPORT.md](MASTER_BENCHMARK_REPORT.md).
