# SmartScan — Final GitHub Package

This repository contains the current SmartScan research code and the **final lightweight Dual-Dwell champion candidate** selected in the September 2026 PDW ablations.

## Current production scheduler

Use:

```python
from scheduler import SmartScanProductionScheduler

scheduler = SmartScanProductionScheduler(num_bands=20, seed=42)
```

The production scheduler is **PDW/observation-only**. It does not require an I/Q dataset or a Track-2 neural checkpoint.

### Final policy

```text
Recurring stale coverage
        ↓
Current band actively detected?
   ├─ yes → keep dwelling; defer stale visit
   └─ no  → rank stale candidates by urgency
                         ↓
                  service best stale band
                         ↓
          otherwise normal Dual-Dwell
          (NMF + UCB + uncertainty scout)
```

The final improvement is intentionally small and lightweight: instead of servicing stale bands in fixed array order, it ranks them using uncertainty, scan age, stale depth, prior observed value, and switching distance.

For details see [FINAL_CHAMPION.md](FINAL_CHAMPION.md).

## Install

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify the final scheduler

```bash
python -m pytest tests/test_smartscan_production_final.py -v
```

## Run the final PDW benchmark

The repository includes six **generic Turing-style synthetic PDW** HDF5 files purely for reproducible software testing. They are not official TSRD data.

```bash
python scripts/benchmark_final_champion_pdw.py --seeds 100
```

To evaluate on compatible genuine Turing `stare` HDF5 files:

```bash
python scripts/benchmark_final_champion_pdw.py --data-root /path/to/turing/stare --seeds 100
```

The benchmark asserts that no `iq` field is present in the observation path.

## Important files

```text
scheduler/smartscan_production.py
    Final champion implementation.

scheduler/observation_runtime.py
    PDW/observation-only runtime adapter.

simulator/turing_pdw_environment.py
    HDF5 PDW replay environment.

scripts/benchmark_final_champion_pdw.py
    Reproducible paired B-vs-final benchmark.

scripts/make_generic_turing_style_dataset.py
    Generator for the included generic PDW fixtures.

tests/test_smartscan_production_final.py
    Focused production-policy unit tests.

tests/synthetic_turing_style/
    Generic PDW fixtures for software validation only.

results/final_champion/
    Two independent 100-seed validation runs and report.
```

## Research code

The repository also retains earlier schedulers, ablations, and benchmark scripts for comparison. They are **not** the production entrypoint. The stable production import is `scheduler.SmartScanProductionScheduler`.

## Data note

The included synthetic HDF5 files mimic a simple Turing-style PDW schema (`ToA`, frequency, pulse width, AoA, amplitude) but are independently generated. Do not label their benchmark results as Alan Turing Synthetic Radar Dataset results.
