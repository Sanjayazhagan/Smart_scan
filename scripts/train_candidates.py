"""Pre-training script for Candidate 2 (Boosted Tree Dwell) and Candidate 3 (GRU-D Spectrum).

Collects partial-observation transitions from SmartScanEnv across multiple scenarios
using training seeds [20001..20005] (strictly separated from evaluation seeds).
Trains:
1. archive/models/dwell_boosted_tree.json (25 trees, depth 3)
2. archive/models/grud_spectrum_weights.pt (GRU-D PyTorch cell)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.environment import SmartScanEnv
from scheduler.boosted_tree_dwell_scheduler import CompactGBDTClassifier
from scheduler.grud_spectrum_model import GRUDSpectrumCell

TRAIN_SEEDS = [20001, 20002, 20003, 20004, 20005]
TRAIN_SCENARIOS = ["stationary", "hopping", "changing", "operational"]
STEPS_PER_EPISODE = 100
NUM_BANDS = 20

MODELS_DIR = Path(__file__).resolve().parent.parent / "archive" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def collect_training_data():
    print(f"Collecting transitions across {len(TRAIN_SCENARIOS)} scenarios and {len(TRAIN_SEEDS)} seeds...")
    
    # Data for Boosted Tree
    tree_X = []
    tree_y = []

    # Data for GRU-D sequences
    # We will store episodes of length STEPS_PER_EPISODE
    grud_x_list = []
    grud_m_list = []
    grud_d_list = []
    grud_y_list = []

    for sc in TRAIN_SCENARIOS:
        for s in TRAIN_SEEDS:
            env = SmartScanEnv(num_bands=NUM_BANDS, episode_length=STEPS_PER_EPISODE, seed=s, scenario=sc)
            obs, info = env.reset(seed=s)

            # Rollout with a mixed exploratory policy
            rng = np.random.default_rng(s)
            
            # State tracking for Tree
            last_band = 0
            dwell_count = 0
            hit_count = 0
            miss_count = 0
            last_q = 0.0
            dwell_qs = []
            band_hits = np.zeros(NUM_BANDS)
            band_scans = np.zeros(NUM_BANDS)
            tot_hits = 0
            tot_scans = 0

            # GRU-D sequence buffers
            ep_x = np.zeros((STEPS_PER_EPISODE, NUM_BANDS), dtype=np.float32)
            ep_m = np.zeros((STEPS_PER_EPISODE, NUM_BANDS), dtype=np.float32)
            ep_d = np.zeros((STEPS_PER_EPISODE, NUM_BANDS), dtype=np.float32)
            ep_y = np.zeros((STEPS_PER_EPISODE, NUM_BANDS), dtype=np.float32)
            delta = np.ones(NUM_BANDS, dtype=np.float32)

            # Pre-collect episode
            for t in range(STEPS_PER_EPISODE):
                # Action choice: 70% dwell/track if active, 30% random explore
                if hit_count > 0 and miss_count <= 1 and rng.random() < 0.70:
                    action = last_band
                else:
                    action = int(rng.integers(0, NUM_BANDS))

                if action == last_band:
                    dwell_count += 1
                else:
                    dwell_count = 1
                    hit_count = 0
                    miss_count = 0
                    dwell_qs = []

                # Ground truth activity at step t
                gt_activity = np.zeros(NUM_BANDS, dtype=np.float32)
                base_env = env.unwrapped
                world = getattr(base_env, "world", None)
                if world is not None:
                    step_idx = min(STEPS_PER_EPISODE - 1, t)
                    for em in world.emitters:
                        if em.is_active(step_idx):
                            gt_activity[em.get_band(step_idx)] = 1.0

                # Step environment
                next_obs, reward, done, truncated, next_info = env.step(action)
                det = float(next_obs.get("detected", 0.0) > 0.5)
                q_arr = next_obs.get("quality", [0.0])
                q = float(q_arr[0] if isinstance(q_arr, (np.ndarray, list)) else q_arr)

                # Record GRU-D step
                ep_x[t, action] = q if det > 0.5 else 0.0
                ep_m[t, action] = 1.0
                ep_d[t] = delta
                ep_y[t] = gt_activity

                # Update delta
                delta += 1.0
                delta[action] = 1.0

                # Record Tree transition if dwelling
                if dwell_count > 1:
                    mean_q = float(np.mean(dwell_qs)) if dwell_qs else 0.0
                    b_rate = band_hits[action] / max(1, band_scans[action])
                    g_rate = tot_hits / max(1, tot_scans)
                    feat = [float(hit_count), float(dwell_count), float(miss_count), last_q, mean_q, b_rate, g_rate]
                    # Target is whether this band is active next step
                    label = 1.0 if gt_activity[action] > 0.5 else 0.0
                    tree_X.append(feat)
                    tree_y.append(label)

                # Update stats
                tot_scans += 1
                band_scans[action] += 1
                if det > 0.5:
                    tot_hits += 1
                    band_hits[action] += 1
                    hit_count += 1
                    miss_count = 0
                    last_q = q
                    dwell_qs.append(q)
                else:
                    miss_count += 1
                    hit_count = 0
                    last_q = 0.0

                last_band = action

            grud_x_list.append(ep_x)
            grud_m_list.append(ep_m)
            grud_d_list.append(ep_d)
            grud_y_list.append(ep_y)

    print(f"Collected {len(tree_X)} dwell transitions for Tree model.")
    print(f"Collected {len(grud_x_list)} sequence episodes ({len(grud_x_list) * STEPS_PER_EPISODE} steps) for GRU-D.")
    return (np.array(tree_X, dtype=np.float64), np.array(tree_y, dtype=np.float64),
            np.array(grud_x_list, dtype=np.float32), np.array(grud_m_list, dtype=np.float32),
            np.array(grud_d_list, dtype=np.float32), np.array(grud_y_list, dtype=np.float32))


def train_boosted_tree(X: np.ndarray, y: np.ndarray):
    print("Training Candidate 2 Compact GBDT (25 trees, depth 3)...")
    model = CompactGBDTClassifier(n_estimators=25, max_depth=3, learning_rate=0.10)
    model.fit(X, y)
    preds = model.predict_proba(X)
    acc = np.mean((preds > 0.5).astype(float) == y)
    print(f"Compact GBDT Train Accuracy: {acc * 100:.2f}% (samples={len(y)})")
    
    save_path = str(MODELS_DIR / "dwell_boosted_tree.json")
    model.save(save_path)
    print(f"Saved boosted tree model to {save_path}")


def train_grud_model(x_seq: np.ndarray, m_seq: np.ndarray, d_seq: np.ndarray, y_seq: np.ndarray):
    print("Training Candidate 3 GRU-D Spectrum Model on CPU...")
    # Convert to PyTorch tensors
    X_t = torch.from_numpy(x_seq)
    M_t = torch.from_numpy(m_seq)
    D_t = torch.from_numpy(d_seq)
    Y_t = torch.from_numpy(y_seq)

    model = GRUDSpectrumCell(num_bands=NUM_BANDS, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)

    epochs = 25
    batch_size = 4
    num_samples = len(X_t)

    for ep in range(epochs):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        batches = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i+batch_size]
            bx = X_t[idx]
            bm = M_t[idx]
            bd = D_t[idx]
            by = Y_t[idx]

            optimizer.zero_grad()
            preds = model.forward_sequence(bx, bm, bd)
            loss = F.binary_cross_entropy(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1:2d}/{epochs:2d} | BCE Loss: {total_loss / batches:.4f}")

    save_path = str(MODELS_DIR / "grud_spectrum_weights.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved GRU-D weights to {save_path}")


if __name__ == "__main__":
    t_X, t_y, g_x, g_m, g_d, g_y = collect_training_data()
    train_boosted_tree(t_X, t_y)
    train_grud_model(g_x, g_m, g_d, g_y)
    print("All candidate models successfully trained and serialized!")
