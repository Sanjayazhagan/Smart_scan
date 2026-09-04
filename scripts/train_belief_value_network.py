"""Trains the Belief-State Value Network (RL Critic) for Expectimax leaf evaluation.

Generates training rollouts across all 6 scenarios on development seeds [4001, 4002, 4003, 4004],
computes discounted Monte Carlo returns, and trains the BeliefValueNetwork via AdamW.
Saves weights to archive/models/belief_value_network.pt.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.nmf_expectimax import NMFExpectimaxScheduler
from scheduler.paradigms.matrix_factorization import NMFScheduler
from scheduler.rl_value_network import (
    BeliefValueNetwork,
    extract_belief_features,
    DEFAULT_VALUE_NET_PATH,
)
from scheduler.track2_runtime import DEFAULT_MODEL_PATH

TRAIN_SEEDS = [4001, 4002, 4003, 4004]
SCENARIOS = ["stationary", "hopping", "changing", "harsh", "operational", "crowded"]
EPISODE_STEPS = 150
GAMMA = 0.90

print("=" * 80)
print("COLLECTING TRAINING DATA FOR BELIEF-STATE RL VALUE NETWORK")
print(f"Scenarios: {SCENARIOS}")
print(f"Seeds: {TRAIN_SEEDS} | Total episodes: {len(SCENARIOS) * len(TRAIN_SEEDS) * 2}")
print("=" * 80)

all_features = []
all_returns = []

t0 = time.perf_counter()

# Collect rollouts from both NMF and Expectimax controllers
for s in TRAIN_SEEDS:
    for sc in SCENARIOS:
        for sched_type in ["nmf", "expectimax"]:
            env = SmartScanEnv(num_bands=20, episode_length=EPISODE_STEPS, seed=s, scenario=sc)
            if sched_type == "nmf":
                sched = NMFScheduler(20, seed=s)
            else:
                sched = NMFExpectimaxScheduler(20, depth=2, top_k=4, model_path=DEFAULT_MODEL_PATH, seed=s)

            obs, info = env.reset(seed=s)
            ep_features = []
            ep_rewards = []
            last_band = None

            for st in range(EPISODE_STEPS):
                action = sched.select_band()
                # Extract current perceptual belief features
                if hasattr(sched, "runtime"):
                    gb = sched.runtime.get_global_belief()
                    feats = extract_belief_features(
                        gb["band_belief"], gb["band_uncertainty"], gb["scan_age"], last_band
                    )
                else:
                    # Synthetic belief representation for pure NMF rollouts
                    b_synth = np.zeros(20, dtype=np.float32)
                    if hasattr(sched, "predicted_spectrum"):
                        spec = np.clip(sched.predicted_spectrum, 0.0, None)
                        if spec.sum() > 1e-6:
                            b_synth = (spec / spec.sum()).astype(np.float32)
                    feats = extract_belief_features(b_synth, np.full(20, 0.5, dtype=np.float32), np.full(20, 0.5, dtype=np.float32), last_band)

                ep_features.append(feats)
                last_band = action

                obs, reward, done, truncated, info = env.step(action)
                sched.update(action, reward, obs)
                ep_rewards.append(reward)

                if done or truncated:
                    break

            # Compute discounted Monte Carlo returns G_t
            T = len(ep_rewards)
            returns = np.zeros(T, dtype=np.float32)
            running_g = 0.0
            for t_idx in reversed(range(T)):
                running_g = ep_rewards[t_idx] + GAMMA * running_g
                returns[t_idx] = running_g

            all_features.extend(ep_features)
            all_returns.extend(returns.tolist())

features_arr = np.array(all_features, dtype=np.float32)
returns_arr = np.array(all_returns, dtype=np.float32)

print(f"Data collection complete in {time.perf_counter() - t0:.2f}s!")
print(f"Total training samples: {len(features_arr):,} | Mean return: {returns_arr.mean():.2f} (std: {returns_arr.std():.2f})")

# -------------------------------------------------------------
# Train the BeliefValueNetwork
# -------------------------------------------------------------
print("\nTraining BeliefValueNetwork (RL Critic)...")
X = torch.from_numpy(features_arr)
y = torch.from_numpy(returns_arr)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = BeliefValueNetwork(input_dim=61, hidden_dim=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

model.train()
for epoch in range(1, 26):
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)
    mse = total_loss / len(features_arr)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:02d}/25 | MSE Loss: {mse:.4f} (RMSE: {np.sqrt(mse):.3f})")

# Save model weights
DEFAULT_VALUE_NET_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), DEFAULT_VALUE_NET_PATH)
print(f"\nModel successfully saved to {DEFAULT_VALUE_NET_PATH}!")
