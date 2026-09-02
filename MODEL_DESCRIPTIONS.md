# Smart Scan model and scheduler descriptions

This package contains source code only. It intentionally excludes trained
weights, datasets, virtual environments, caches, and Git metadata.

## External runtime files

The integrated runtime expects these files outside the repository:

```text
C:\Users\asus\Documents\SmartScanArtifacts\track2\track2_final_world_model.pt
C:\Users\asus\Documents\SmartScanArtifacts\track2\track2_synthetic_rf_dataset.npz
```

## Track 2 identity model

`scheduler/track2_core.py` defines `RuntimeIdentityEncoder`.

```text
I/Q [2,512]
  -> Conv1D(2,32,7) + BatchNorm + GELU + MaxPool
  -> Conv1D(32,64,5) + BatchNorm + GELU + MaxPool
  -> Conv1D(64,128,3) + BatchNorm + GELU
  -> global average pooling
  -> saved embedding head
  -> normalized identity embedding
```

The embedding is compared with frozen known-emitter prototypes. Observations
that do not match a known prototype can form persistent unknown tracks.

## Track 2 state and future-band model

`EmitterStateWorldModel` converts I/Q into an STFT spectrogram and processes it
with a 2D CNN. The resulting state vector is combined with band, quality, and
elapsed-time features and passed through a per-emitter GRU.

The GRU returns 21 probabilities: 20 possible next bands plus inactive.

## Persistent tracking

`TrackManager` stores identity, last observation, GRU hidden state, future-band
probabilities, novelty, behaviour surprise, and uncertainty separately for
each emitter. Episode reset keeps frozen weights/prototypes but clears all
episode-specific histories and unknown tracks.

Separate runtime diagnostics preserve the frozen 48-feature schema while adding
observable per-track prediction-error EMA, recent hit evidence, and soft miss
exposure. Band-level prediction error, recent hit/miss vectors, and unconfirmed
candidate summaries are exposed independently.

## Observable scheduling state

`scheduler/track2_runtime.py` exposes:

```text
band_belief[20] + normalized_scan_age[20] + band_uncertainty[20] = 60 values
```

No-detection scans update band-level age/miss state but never pass noise through
identity association or a per-emitter GRU.

## Simulator scenarios

`simulator/scenarios.py` defines deterministic hidden-world presets for fixed,
hopping, bursty, crowded, changing, harsh-channel, and mixed operation. Each
world is generated before scheduling begins. Emitter dynamics are independent
of policy actions, while keyed sensing randomness makes the same step/band
counterfactual reproducible across schedulers. `legacy` retains the original
simulator for historical comparisons.

## Scheduling approaches

- `scheduler/baselines.py`: sequential, random, UCB, and Thompson Sampling.
- `scheduler/frequency_rl.py`: PPO using per-band scan history.
- `scheduler/emitter_rl.py`: preserved 20-feature Track 2 PPO and the newer
  60-feature belief/age/uncertainty PPO.
- `scheduler/model_predictive.py`: depth-3 beam-search planner. It simulates
  candidate scan sequences from current Track 2 belief, age, and uncertainty;
  scores expected reward and information value; executes the first action; and
  replans after the real observation.
- `scheduler/emitter_aware_predictive.py`: experimental emitter-aware extension
  that preserves MPP-60 and derives interpretable per-band values from confirmed
  masked Track 2 rows. It also defines the belief-only and term-removal
  ablations and produces an explainable decision trace.
- `scheduler/belief_tree.py`: depth-3 budgeted belief tree. Proposed scans
  branch into Bayesian HIT and MISS posteriors. The root considers eight
  actions, deeper nodes consider four, and the last action layer is solved
  exactly without constructing zero-value terminal children.
- `scheduler/adaptive_moe.py`: conservative regime guards, MPP-60, an
  observation-only discounted UCB expert, an 80-scan quality/stability history,
  clean-stationary tree and sustained-noise MPP guards, a reliable-complex tree
  guard, safe OOD fallback, adaptive prioritized exploration, and the complete
  mixture-of-experts scheduler. The history uses observations only and applies
  hysteresis. The behaviour-change feature cannot switch experts by itself.
  PPO is opt-in only; OOD rules override learned routing. An optional unscored
  four-sweep calibration phase can lock only a high-confidence noisy sample to
  MPP-60; ambiguous samples continue through the online router.
- `scheduler/emitter_attention.py`: masked attention pooling of variable
  per-emitter rows into a fixed 288-feature RL state.
- `scheduler/learned_value.py`: small trainable path-value network that can
  augment, but does not hide, the hand-designed planner score.
- `evaluation/adaptive_training.py`: same-seeded expert-outcome collection
  across sequential, reverse, and seeded-random warmups; latency-aware,
  class-balanced router training; path-return collection; and value training.
- `evaluation/track2_diagnostic.py`: fixed-order offline scoring of Track 2
  band beliefs versus a causal history baseline, the GRU's previous next-band
  forecast, and identity-track purity/fragmentation. Ground truth is diagnostic
  only and is never supplied to the model.

## Important limitation

The planner uses Track 2's one-step future distribution plus a simplified
band-level transition approximation for deeper search. It does not advance the
GRU through genuine future I/Q samples. Real RF and hardware validation remain
necessary.

The included attention PPO checkpoint was trained for only 2,048 steps as an
interface smoke test and is not a competitive policy. The learned value model
also remains a prototype. The v2 learned router is substantially better
balanced than v1 but loses reward and discovery to the rule router on held-out
seeds, so it remains an optional speed-oriented experiment.
