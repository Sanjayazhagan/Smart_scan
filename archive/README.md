# SMART SCAN: Archived Experimental Algorithms & Ablations

This directory preserves standalone experimental schedulers, model-free reinforcement learning explorations, and neural value prototypes developed during the research and benchmarking phase of SMART SCAN.

---

## 1. Summary of Archived Experiments

| Module | Description | Key Finding / Outcome |
|---|---|---|
| `experiments/emitter_rl.py` | Track 2 PPO Reinforcement Learning models (20-dim, 60-dim, diverse anti-camping, generalist curriculum). | Solved initial policy collapse ($6.25 \to 30.85$ stationary). However, pure feedforward PPO lacks multi-step forward lookahead, scoring $12.93$ overall versus Adaptive MoE's $20.39$. |
| `experiments/frequency_rl.py` | Track 1 classical frequency-history RL baseline. | Operates on scalar power observations only; strictly inferior to RF-aware world models. |
| `experiments/emitter_attention.py` | Variable-emitter masked attention pooling into a fixed 288-dim state vector. | Interface prototype; high dimensionality increased policy variance. |
| `experiments/learned_value.py` | Trainable small path-value network intended to replace hand-designed search heuristic. | Scored lower than domain-informed heuristic rollouts on held-out seeds. |

---

## 2. Archived Checkpoints (`archive/models/`)

- `frequency_history_ppo.zip`: Original Track 1 frequency baseline.
- `track2_belief_ppo.zip`: 20-band belief PPO checkpoint.
- `track2_exploration_ppo.zip`: 60-band initial exploration checkpoint.
- `track2_diverse_ppo.zip`: PPO with anti-camping repeat penalties ($30.85$ stationary).
- `track2_generalist_ppo.zip`: 150k-step multi-scenario window-regulated PPO ($11.3$ unique bands).

---

## 3. Active Production Core

The production system in `scheduler/` centers around:
1. **Adaptive Mixture of Experts (`adaptive_moe.py`)**: The verified champion ($20.39$ overall reward).
2. **Asynchronous Background Planning Worker (`async_worker.py`)**: High-throughput sub-millisecond execution engine with reflex pulse interrupt.
3. **Track 2 Perceptual World Model (`track2_core.py`, `track2_runtime.py`)**: Conv1D RF fingerprinting + 2D CNN Spectrogram + Recurrent GRU.
