# Smart Scan

Smart Scan is a partially observable RF scan-scheduling prototype. A frozen
Track 2 perception model converts one-band I/Q observations into emitter
identity tracks, future-band belief, scan age, and uncertainty. Schedulers then
choose one of 20 receiver bands.

## Current model status

- **UCB-first Adaptive** is the single product-facing controller. Observable
  discounted UCB makes the normal decision. It escalates to Adaptive MoE's
  belief-tree specialist only after the observation history establishes a
  sustained clean/stationary pattern.
- **Observable Discounted UCB**, **Adaptive MoE**, and **MPP-60** remain
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
  -> frozen identity CNN + spectrogram state encoder
  -> per-emitter GRU and persistent TrackManager
  -> observable belief[20] + scan age[20] + uncertainty[20]
  -> Observable UCB primary path
       -> sustained clean/stationary evidence? Adaptive belief tree : UCB
  -> next receiver band
```

Simulator ground truth is confined to Gymnasium's `info` dictionary and offline
metrics. It is not passed into Track 2 or Adaptive MoE.

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

Open `http://localhost:8000`. The dashboard exposes only the UCB-first Adaptive
controller. If the Python backend is unavailable, the page clearly labels
itself as a client visual demo.

## Core benchmark

PPO policy files are no longer required when testing core schedulers:

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmark `
  --warmup-steps 80 --episode-length 200 --episodes 2 `
  --seeds 12001 12011 12021 `
  --scenarios stationary hopping changing harsh `
  --only "Observable Discounted UCB" "UCB-first Adaptive" `
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

- `ucb_first_stationary_gate_validation.json`: fresh-seed paired validation of
  UCB-first Adaptive against standalone Observable Discounted UCB. Across seven
  scenarios it scored 19.198 versus 18.055 (+6.3%); the gain was concentrated
  in stationary and bursty worlds and is not evidence of universal superiority.

- `results/warmup_heldout_validation.json`: held-out warm-up evaluation.
- `results/track2_heldout_diagnostic.json`: scheduler-independent Track 2
  diagnostic.
- `results/grand_benchmark_results.json`: comparative paradigm snapshot.
- `BENCHMARK_RESULTS.md`: current interpretation and limitations.
- `MODEL_DESCRIPTIONS.md`: implementation-level model descriptions.

## Known gaps

- No SDR or over-the-air validation.
- The simulator does not yet model every deceptive/jamming waveform.
- MPP deeper steps approximate future belief instead of rolling the GRU through
  genuine future I/Q.
- Track 2 identity and next-band prediction require broader held-out hardware
  data.
- The fresh validation improvement is not statistically conclusive: the paired
  95% interval crosses zero, and planning latency averaged 7.05 ms versus
  2.33 ms for standalone Observable UCB.
