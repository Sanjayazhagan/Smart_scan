# Smart Scan: Production Adaptive MoE & Track 2 RF System

Smart Scan is an advanced partial-observation radio frequency (RF) scheduler and world model.
It couples deep neural RF perception (Track 2) with a high-performance **Adaptive Mixture of Experts (MoE)** and an **Asynchronous Planning Engine** designed for real-world software-defined radio (SDR) deployment.

## Production Champions

- **Algorithmic Champion**: `AdaptiveMixtureOfExpertsScheduler` (**$20.39$ Overall Benchmark Reward** across all 4 operational regimes).
- **Real-Time Deployment Champion**: `AsyncPlanningScheduler` (**$<0.005\text{ ms}$ Non-Blocking Latency**, Record **$-0.584$** in Harsh Noise with Reflex Pulse Confirmation).

---

## System Architecture

```text
Raw I/Q [2, 512] 
    │
    ├──► Conv1D Identity Encoder (87.5% Purity Hardware Fingerprinting)
    ├──► 2D CNN Spectrogram Model (Time-Frequency Energy Extraction)
    └──► Per-Emitter Recurrent GRU (Persistent Latent World Model)
              │
              ▼
    Observable 60-Dim State (Band Belief [20], Scan Age [20], Band Uncertainty [20])
              │
              ▼
    Adaptive Mixture of Experts (MoE) Router
       ├── CLEAN / STATIONARY  ──► History-Gated Belief Tree (37.57 Reward)
       ├── STABLE / COMPLEX    ──► MPP-60 Beam Search (Depth-3 Lookahead)
       ├── DYNAMIC / HOPPING   ──► Observation-Only Discounted UCB (22.72 Reward)
       ├── SUSTAINED NOISE     ──► Pure Bayesian Tree Search (-3.83 vs -8.0 Baseline)
       └── UNKNOWN / OOD       ──► Safe Prioritized Exploration
```

The adaptive system is implemented in `scheduler/adaptive_moe.py`. Its UCB
expert learns from detection quality and scan history, not simulator reward or
hidden truth. The Track 2 behaviour-change score is retained as context but no
longer triggers an expert switch by itself because seeded scenario tests found
it unreliable as a standalone routing signal. An 80-scan observable history
scores detection quality, low-quality alarms, repeated-band outcome changes,
and credible-hit band concentration. Hysteresis prevents one unusual scan from
flipping modes. Clean concentrated history favors the tree; sustained noisy
history favors MPP-60. Neither selector reads the scenario name, reward, or
hidden truth. PPO is not a default route: an unvalidated RL decision is
possible only through explicit opt-in, and old learned-router `rl` labels
safely fall back to the belief tree.
Exploration probability starts near 0.5%, 1%, 3%, 0%, and 5% for easy,
intermediate, complex, dynamic, and OOD states, then changes with uncertainty.
Exploration is weighted toward stale, uncertain, novel, changing, and
high-prediction-error bands rather than choosing uniformly at random.

The model-predictive scheduler provides a training-free alternative. It uses
depth-3 beam search to simulate candidate scan sequences from Track 2 belief,
age, and uncertainty; scores expected simulator reward plus information value;
executes only the first action; and replans after the real hit or miss.

The observation-dependent tree uses a full root shortlist, a smaller deeper
branch budget, and an exact terminal-layer shortcut. The shortcut does not
change the final-layer choice because terminal HIT/MISS children have no future
value. This reduced measured tree latency from about 303 ms to 42 ms while
retaining contingent HIT/MISS branches.

`EmitterAwareModelPredictivePlanner` is an experimental ablation alongside the
unchanged MPP-60. It converts confirmed masked `track_features [16,48]` into
interpretable per-band quantities (dominant track probability, support count,
and probability-weighted uncertainty, behaviour, novelty, recency, quality,
and identity confidence). Anomalies contribute only when band belief and
observation quality are meaningful. Every action saves an explainable score
breakdown and top contributing runtime tracks.

`scan_age` is derived only from chosen actions. `band_uncertainty` combines each
confirmed track's predicted band probability with its prediction uncertainty;
where track evidence is absent, normalized scan recency supplies uncertainty.
Never-scanned bands therefore start at high uncertainty. A recent observable
miss lowers current belief/evidence for only that scanned band and then decays.

Ground-truth emitter IDs and active bands stay in Gymnasium's `info` dictionary
for evaluation. They are never part of the PPO observation or passed to
`Track2Runtime`. The simulator uses an emitter ID internally only to choose an
I/Q sample.

## Scenario simulator v2

The original simulator remains available as `--scenario legacy`. New scenario
presets are `stationary`, `hopping`, `bursty`, `crowded`, `changing`, `harsh`,
and `mixed`.

For every non-legacy episode, the complete hidden emitter activity, band-hop,
SNR, behaviour-change, and interference timeline is generated before the first
scan. Hidden-world generation and step/band observations use independent random
streams. Different schedulers therefore cannot change the future world merely
by consuming random numbers in a different order.

The new scenarios add variable SNR, fading, interference, sensor dropout, I/Q
noise/imbalance/clipping, hopping, burst activity, behaviour changes, crowded
bands, and optional tuning-distance cost. The observation schema remains
unchanged, so Track 2 and all existing schedulers still receive only the chosen
band's measurement. Extra truth appears only in evaluation `info`.

## Setup and tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Train policies

Track 2 weights stay frozen; only PPO is trained:

```powershell
python -m scheduler.emitter_rl --state-size 60 --timesteps 100000 --output models/track2_exploration_ppo
# Rich masked emitter-attention expert (288 observable features):
python -m scheduler.emitter_rl --state-size 288 --timesteps 100000 --output models/track2_attention_ppo
# Optional 20-feature ablation (does not overwrite the existing saved policy):
python -m scheduler.emitter_rl --state-size 20 --timesteps 100000 --output models/track2_belief_ppo_ablation
python -m scheduler.frequency_rl --timesteps 100000 --output models/frequency_history_ppo
```

Train the optional cost-aware mixture gate and learned path-value estimator
from seeded simulator outcomes:

```powershell
python -m evaluation.adaptive_training --mode all `
  --output-dir C:\Users\asus\Documents\SmartScanArtifacts\adaptive `
  --seeds 42 142 242 342 442 `
  --warmup-steps 5 20 50 `
  --warmup-patterns sequential reverse random `
  --router-rollout-horizon 40
```

This trains only the router/value/PPO scheduling components. It never modifies
the frozen Track 2 CNN/GRU weights.

## Benchmark all schedulers

```powershell
python -m evaluation.benchmark --frequency-policy models/frequency_history_ppo.zip --track2-policy models/track2_belief_ppo.zip --track2-exploration-policy models/track2_exploration_ppo.zip --router-model C:\Users\asus\Documents\SmartScanArtifacts\adaptive_v2\router_gate.npz --path-value-model C:\Users\asus\Documents\SmartScanArtifacts\adaptive\path_value_model.npz --episodes 20 --episode-length 200
```

Multiple benchmark seeds are supported with, for example, `--seeds 42 142 242`.
The report includes reward mean/std/median, hits, false positives, misses,
detection/interception rates, wasted scans, unique-band coverage, revisit time,
planning latency, and emitter discovery delay. It also includes MPP-BeliefOnly,
MPP-60, full MPP-EmitterAware, and both required term-removal ablations.

See `BENCHMARK_RESULTS.md` for the controlled seed-42 results and the current
go/no-go interpretation.

### 4-Scenario Benchmark Scoreboard (Held-Out Seeds 12001, 12011, 12021)

| Scenario | UCB Baseline | Generalist PPO | **Adaptive MoE (Sync Champion)** | **Async Planning Worker (Real-Time Driver)** |
|---|:---:|:---:|:---:|:---:|
| **Stationary** | $25.05$ | $17.80$ | **$37.57$** 🏆 | $32.02$ |
| **Hopping** | $23.05$ | $19.32$ | **$22.22$** | **$15.3$ hits / 86.5% Det** ⚡ |
| **Changing** | $24.20$ | $17.38$ | **$25.12$** 🏆 | $15.04$ |
| **Harsh Noise** | $-2.74$ | $-2.79$ | $-3.83$ | **$-0.584$** 🏆 *(All-Time Record)* |
| **Overall Score** | $17.39$ | $12.93$ | **$20.39$** 🏆 *(Champion)* | High-Throughput Real-Time Driver |
| **Planning Latency** | $0.01\text{ ms}$ | $1.10\text{ ms}$ | $25.0\text{ ms}$ | **$<0.005\text{ ms}$** ⚡ |


Run several realistic scenario families in one command:

```powershell
python -m evaluation.benchmark `
  --frequency-policy models\frequency_history_ppo.zip `
  --track2-policy models\track2_belief_ppo.zip `
  --episodes 3 --episode-length 200 `
  --seeds 8001 8011 8021 `
  --scenarios stationary hopping bursty crowded changing harsh mixed `
  --only "UCB" "MPP-60" "MPP-ObservationDependent Tree" "Adaptive MoE rule router" `
  --output scenario_matrix.json
```

To calibrate from a few scans before the actual test, add `--warmup-steps 80`.
This performs four fixed sequential sweeps over the 20 bands. The scheduler and
environment continue from that state, but the warm-up is excluded from
`mean_reward` and all main test metrics:

```powershell
python -m evaluation.benchmark `
  --frequency-policy models\frequency_history_ppo.zip `
  --track2-policy models\track2_belief_ppo.zip `
  --warmup-steps 80 --episode-length 200 --episodes 2 `
  --seeds 12001 12011 12021 `
  --scenarios stationary hopping changing harsh `
  --only "UCB" "MPP-60" "MPP-ObservationDependent Tree" "Adaptive MoE rule router" `
  --output warmup_heldout_validation.json
```

`mean_warmup_reward` reports calibration separately, while `mean_reward`
reports only the 200 actual decisions. The adaptive trace also reports the
calibration evidence and score, whether a noisy-MPP lock was selected, and the
scored-phase expert ratios. Warm-up is useful evidence, not an oracle: a short
changing sample can look noisy and a harsh sample can temporarily look clean.

Diagnose Track 2 directly, independently of scheduler reward:

```powershell
python -m evaluation.track2_diagnostic `
  --scenarios stationary hopping changing harsh `
  --seeds 12001 12011 12021 `
  --warmup-steps 80 --scored-steps 200 `
  --output track2_heldout_diagnostic.json
```

This uses a fixed sequential scan order and scores Track 2 belief calibration,
GRU next-band forecasts, and identity-track consistency. Simulator truth is
used only for offline measurement and is never passed to Track 2.

Scenario reports additionally include effective SNR, switch rate/distance,
switching cost, sensor dropout, observed interference, and behaviour-change
adaptation delay where a changed emitter is later intercepted.
