# SMART SCAN: Archived Experimental Algorithms & Ablations

This directory preserves standalone experimental schedulers, model-free reinforcement learning explorations, and neural value prototypes developed during the research and benchmarking phase of SMART SCAN.

---

## 1. Summary of Archived Experiments

| Module | Description | Key Finding / Outcome |
|---|---|---|
| `experiments/emitter_rl.py` | Track 2 PPO Reinforcement Learning models (20-dim, 60-dim, diverse anti-camping, generalist curriculum). | Solved an early policy-collapse issue, but remained seed-sensitive and below the active rule router in the retained held-out evaluation. |
| `experiments/frequency_rl.py` | Track 1 classical frequency-history RL baseline. | Operates on scalar power observations only; strictly inferior to RF-aware world models. |
| `experiments/emitter_attention.py` | Variable-emitter masked attention pooling into a fixed 288-dim state vector. | Interface prototype; high dimensionality increased policy variance. |
| `experiments/learned_value.py` | Trainable small path-value network intended to replace hand-designed search heuristic. | Scored lower than domain-informed heuristic rollouts on held-out seeds. |

---

## 2. Checkpoints

No checkpoints are bundled in this source repository. Historical PPO weights,
when retained, belong under `C:\Users\asus\Documents\SmartScanArtifacts`.

Older development and tuning JSON reports are stored in
`archive/results_legacy/`. They are retained for auditability but are not the
current evidence set documented in the main README.

---

## 3. Active Production Core

The production system in `scheduler/` centers around:
1. **Adaptive Mixture of Experts (`adaptive_moe.py`)**: Active advanced all-rounder; not universally superior on every seed set.
2. **Asynchronous Background Planning Worker (`async_worker.py`)**: Experimental background-planning wrapper. Queue-pop latency is not reported as end-to-end latency.
3. **Track 2 Perceptual World Model (`track2_core.py`, `track2_runtime.py`)**: Conv1D RF fingerprinting + 2D CNN Spectrogram + Recurrent GRU.
