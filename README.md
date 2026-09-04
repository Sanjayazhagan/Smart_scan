# Smart Scan

Smart Scan is a partially observable RF scan-scheduling prototype. A frozen
Track 2 perception model converts one-band I/Q observations into emitter
identity tracks, future-band belief, scan age, and uncertainty. Schedulers then
choose one of 20 receiver bands.

## Current model status

- **World-Model UCB** is the single product-facing controller. Discounted UCB
  balances exploitation, exploration, recency, and uncertainty. Track 2's
  predicted next-band probability directly augments the UCB score only after
  observable online calibration proves that the forecast is reliable.
- **Observable Discounted UCB**, the retired **Adaptive MoE**, and **MPP-60** remain
  standalone benchmark components, not separate dashboard products.
- **Neural-Augmented UCB** is implemented as a guarded extension. It adds
  `alpha * Track2 future-band probability` only after at least twenty credible
  positive observations and an online Brier skill of at least 0.90 sustained
  for twenty evaluated forecasts. Otherwise its
  neural weight is exactly zero and it behaves as Observable UCB.
- **Standard UCB1** remains a simulator benchmark only: it updates from the
  simulator reward, which is not available as ground truth on real hardware.
- PPO, learned routing/value models, asynchronous planning, NMF, RPCA, and
  other paradigms remain research comparisons. None is presented as the
  production champion.

Turbo-MoE was removed after it failed to demonstrate a reliable reward gain.
There is currently no claim of industry superiority or real-hardware readiness.

## Architecture

```text
I/Q observation [2, 512]
  -> identity CNN + spectrogram state encoder
  -> per-emitter GRU and persistent TrackManager
  -> future-band belief[20] + scan age[20] + uncertainty[20]
  -> discounted UCB value + exploration + recency
  -> calibrated reliability gate * Track 2 future-band belief
  -> one World-Model UCB score -> next receiver band
```

Simulator ground truth is confined to Gymnasium's `info` dictionary and offline
metrics. It is not passed into Track 2 or World-Model UCB.

## External artifacts

Keep weights and the I/Q dataset outside the source tree:

```text
C:\Users\asus\Documents\SmartScanArtifacts\track2\track2_final_world_model.pt
C:\Users\asus\Documents\SmartScanArtifacts\track2\track2_synthetic_rf_dataset.npz
```

## Setup

```powershell
cd "C:\Users\asus\Documents\SMART SCAN"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Dashboard

```powershell
.\.venv\Scripts\python.exe dashboard\server.py
```

Open `http://localhost:8000`. The dashboard exposes only World-Model UCB. There
is no MoE router or tree-planner escalation in the product decision path. If
the Python backend is unavailable, the page clearly labels itself as a client
visual demo.

## Core benchmark

PPO policy files are no longer required when testing core schedulers:

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmark `
  --warmup-steps 80 --episode-length 200 --episodes 2 `
  --seeds 12001 12011 12021 `
  --scenarios stationary hopping changing harsh `
  --only "Observable Discounted UCB" "World-Model UCB" `
  --output results\current_validation.json
```

The benchmark reports selection time, scheduler update time, full control time,
reward, detections, false positives, misses, coverage, switching, discovery,
and routing ratios. Use multiple untouched seeds; a single favorable episode is
not evidence that one scheduler is universally better.

The direct neural term is intentionally safety-gated. Fresh validation found
that the current GRU did not improve hopping: its retained held-out diagnostic
has 0% next-band top-1 accuracy over 34 hopping transitions. An ungated neural
coefficient therefore reduced reward. Retraining/calibrating Track 2 is required
before the neural term can be presented as an active performance advantage.

## Operational simulator

The `operational` scenario is the neutral real-world stress test. It adds
late-arriving and finite-lifetime emitters, persistent deceptive interference,
non-stationary hopping, unequal hidden threat weights, fading/dropout, receiver
switching cost, and retuning-related detection loss. All hidden timelines are
generated before scheduling and are identical for every policy at a given
seed. Threat labels remain in offline `info` metrics and never enter scheduler
observations.

In addition to the existing metrics, the benchmark reports false-alarm rate,
threat-weighted interception rate, late-emitter discovery rate/delay, and the
deceptive-false-alarm ratio.

## Evidence retained in the repository

- `NEURAL_UCB_RIGOROUS_RESULTS.md`: retained comparison showing why the current
  Track 2 forecast remains safety-gated. Retraining is required before claiming
  a performance advantage over the UCB-only fallback.

- `results/warmup_heldout_validation.json`: held-out warm-up evaluation.
- `results/track2_heldout_diagnostic.json`: scheduler-independent Track 2
  diagnostic.
- `results/grand_benchmark_results.json`: comparative paradigm snapshot.
- `BENCHMARK_RESULTS.md`: current interpretation and limitations.
- `MODEL_DESCRIPTIONS.md`: implementation-level model descriptions.

## Known gaps

- No SDR or over-the-air validation.
- The simulator does not yet model every deceptive/jamming waveform.
- Track 2 identity and next-band prediction require broader held-out hardware
  data.
- The present held-out hopping diagnostic found 0% next-band top-1 accuracy over
  34 transitions, so the world-model coefficient currently remains zero.
