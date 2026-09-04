"""Multi-Scenario Turing Radar World Model Training Pipeline.

Ingests 8 diverse radar scenarios from the Alan Turing Institute dataset,
extracts multi-emitter pulse trains, trains 1D Identity CNN and 2D Spectrogram
GRU world model, and packages the bundle into SmartScanArtifacts.
"""

import io
import os
import sys
import ssl
import time
import urllib.request
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text(encoding="utf-8-sig").strip()
ctx = ssl._create_unverified_context()

print("=" * 80)
print("  MULTI-SCENARIO TURING RADAR WORLD MODEL TRAINING PIPELINE")
print("=" * 80)

# 1. Download and parse 8 diverse scenarios
SCENARIO_COUNT = 8
all_toas = []
all_freqs = []
all_pws = []
all_amps = []
all_emitters = []
global_emitter_offset = 0

print(f"\n[1/5] Ingesting {SCENARIO_COUNT} diverse scenarios from Turing repository...")

for sc_id in range(SCENARIO_COUNT):
    url = f"https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset/resolve/main/scan/train_scan/config_{sc_id}.h5"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            data_bytes = resp.read()
            
        with h5py.File(io.BytesIO(data_bytes), "r") as f:
            raw_d = f["data"][:]
            raw_l = f["labels"][:].flatten()
            
        del data_bytes  # Clean RAM immediately
        
        # Take the top 4 most frequent emitters from this scenario
        counts = Counter(raw_l)
        sc_top_emitters = [e for e, _ in counts.most_common(4)]
        mask = np.isin(raw_l, sc_top_emitters)
        
        sc_data = raw_d[mask]
        sc_labels = raw_l[mask]
        
        # Subsample up to 4,000 pulses per scenario
        if len(sc_labels) > 4000:
            idx = np.random.RandomState(42 + sc_id).choice(len(sc_labels), 4000, replace=False)
            idx.sort()
            sc_data = sc_data[idx]
            sc_labels = sc_labels[idx]
            
        all_toas.append(sc_data[:, 0])
        all_freqs.append(sc_data[:, 1])
        all_pws.append(sc_data[:, 2])
        all_amps.append(sc_data[:, 4])
        
        # Map labels
        local_map = {old: global_emitter_offset + new for new, old in enumerate(sc_top_emitters)}
        all_emitters.append(np.array([local_map[e] for e in sc_labels], dtype=np.int64))
        global_emitter_offset += len(sc_top_emitters)
        
        print(f"      Loaded config_{sc_id}.h5: {len(sc_labels):,} pulses, {len(sc_top_emitters)} emitters.")
    except Exception as e:
        print(f"      Warning: could not load config_{sc_id}.h5: {e}")

toas = np.concatenate(all_toas)
freqs = np.concatenate(all_freqs)
pws = np.concatenate(all_pws)
amps = np.concatenate(all_amps)
emitters = np.concatenate(all_emitters)

TOTAL_PULSES = len(emitters)
print(f"\n[2/5] Total extracted pulses across all scenarios: {TOTAL_PULSES:,}")
print(f"      Total unique emitter classes across scenarios: {len(np.unique(emitters))}")

# Frequency mapping to 20 receiver channels (Bands 0..19)
f_min, f_max = np.percentile(freqs, 1), np.percentile(freqs, 99)
bands = np.clip(
    ((freqs - f_min) / max(1e-5, f_max - f_min) * 20).astype(np.int64),
    0, 19
)

# Remap to top 10 most prominent global emitter classes for the 10 prototype slots
top_10_emitters = [e for e, _ in Counter(emitters).most_common(10)]
proto_mask = np.isin(emitters, top_10_emitters)
p_map = {old: new for new, old in enumerate(top_10_emitters)}

sub_idx = np.where(proto_mask)[0]
# Subsample 12,000 pulses for deep training
if len(sub_idx) > 12000:
    sub_idx = np.random.RandomState(42).choice(sub_idx, 12000, replace=False)
    sub_idx.sort()

N = len(sub_idx)
toas = toas[sub_idx]
freqs = freqs[sub_idx]
pws = pws[sub_idx]
amps = amps[sub_idx]
bands = bands[sub_idx]
emitters = np.array([p_map[emitters[i]] for i in sub_idx], dtype=np.int64)

# 2. Synthesize physical baseband I/Q waveforms
print(f"\n[3/5] Synthesizing physical baseband I/Q waveforms for {N:,} pulses...")
t = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
iq_samples = np.zeros((N, 2, 512), dtype=np.float32)

amp_norm = np.clip((amps - amps.min()) / max(1e-5, amps.max() - amps.min()), 0.4, 1.5)
pw_norm = np.clip(pws / max(1e-5, np.median(pws)), 0.3, 3.0)

for i in range(N):
    omega = 2.0 * np.pi * (bands[i] - 9.5) * 0.1
    env = np.exp(-0.5 * (t / (0.4 * pw_norm[i])) ** 2)
    noise = np.random.normal(0, 0.06, (2, 512)).astype(np.float32)
    iq_samples[i, 0] = amp_norm[i] * env * np.cos(omega * t) + noise[0]
    iq_samples[i, 1] = amp_norm[i] * env * np.sin(omega * t) + noise[1]

# Calculate next-band targets
next_bands = np.full(N, 20, dtype=np.int64)
for e in range(10):
    e_idx = np.where(emitters == e)[0]
    for k in range(len(e_idx) - 1):
        curr_i = e_idx[k]
        next_i = e_idx[k + 1]
        if toas[next_i] - toas[curr_i] < 100000.0:
            next_bands[curr_i] = bands[next_i]

# -------------------------------------------------------------
# 3. Train Model 1: RuntimeIdentityEncoder (1D CNN)
# -------------------------------------------------------------
print("\n[4/5] Training Model 1: 1D-ConvNet Identity Encoder on 10 Radar Classes...")
device = torch.device("cpu")

class IdentityCNN(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        emb = F.normalize(self.embedding_head(feat), p=2, dim=1)
        logits = self.classifier(emb)
        return emb, logits

id_model = IdentityCNN().to(device)
opt_id = torch.optim.AdamW(id_model.parameters(), lr=1e-3, weight_decay=1e-4)
crit_id = nn.CrossEntropyLoss()

X_iq = torch.tensor(iq_samples, dtype=torch.float32)
Y_emitter = torch.tensor(emitters, dtype=torch.long)

loader_id = DataLoader(TensorDataset(X_iq, Y_emitter), batch_size=64, shuffle=True)

id_model.train()
for epoch in range(16):
    total_loss = 0.0
    for batch_x, batch_y in loader_id:
        opt_id.zero_grad()
        _, logits = id_model(batch_x)
        loss = crit_id(logits, batch_y)
        loss.backward()
        opt_id.step()
        total_loss += loss.item() * len(batch_y)
    if (epoch + 1) % 4 == 0:
        print(f"      Epoch {epoch+1:02d}/16 | Identity Loss: {total_loss / N:.4f}")

# Compute 10 Prototypes
id_model.eval()
with torch.no_grad():
    all_embs, _ = id_model(X_iq)
    prototypes = torch.zeros(10, 64)
    for e in range(10):
        e_mask = emitters == e
        if np.any(e_mask):
            prototypes[e] = all_embs[e_mask].mean(dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1)

# -------------------------------------------------------------
# 4. Train Model 2: EmitterStateWorldModel (2D CNN + Temporal GRU)
# -------------------------------------------------------------
print("\n[5/5] Training Model 2: 2D Spectrogram CNN + Temporal GRU on Multi-Scenario Sequences...")
from scheduler.track2_core import EmitterStateWorldModel

world_model = EmitterStateWorldModel(
    num_bands=20,
    state_dim=32,
    band_embedding_dim=8,
    gru_hidden=128,
    n_fft=64,
    hop_length=16,
).to(device)

opt_wm = torch.optim.AdamW(world_model.parameters(), lr=8e-4, weight_decay=1e-4)

# Dynamic class weighting for active frequency hopping
class_counts = Counter(next_bands)
weights = np.zeros(21, dtype=np.float32)
for c in range(21):
    weights[c] = 1.0 / max(10, class_counts.get(c, 1))
weights[:20] *= 4.0  # Strongly reward predicting active frequency hops!
weights /= weights.sum()
crit_wm = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

# Build sequence windows of length 5
seq_len = 5
X_seq_iq = []
X_seq_band = []
X_seq_qual = []
X_seq_dt = []
Y_seq_next = []

for e in range(10):
    e_idx = np.where(emitters == e)[0]
    for start in range(len(e_idx) - seq_len):
        w = e_idx[start : start + seq_len]
        X_seq_iq.append(iq_samples[w])
        X_seq_band.append(bands[w])
        X_seq_qual.append(amp_norm[w])
        dts = np.diff(toas[w], prepend=toas[w[0]]) / 10000.0
        X_seq_dt.append(dts)
        Y_seq_next.append(next_bands[w[-1]])

X_seq_iq = torch.tensor(np.array(X_seq_iq), dtype=torch.float32)
X_seq_band = torch.tensor(np.array(X_seq_band), dtype=torch.long)
X_seq_qual = torch.tensor(np.array(X_seq_qual), dtype=torch.float32)
X_seq_dt = torch.tensor(np.array(X_seq_dt), dtype=torch.float32)
Y_seq_next = torch.tensor(np.array(Y_seq_next), dtype=torch.long)

SEQ_COUNT = len(Y_seq_next)
print(f"      Built {SEQ_COUNT:,} sequential radar pulse windows across scenarios.")

loader_wm = DataLoader(
    TensorDataset(X_seq_iq, X_seq_band, X_seq_qual, X_seq_dt, Y_seq_next),
    batch_size=64,
    shuffle=True
)

world_model.train()
for epoch in range(20):
    total_loss = 0.0
    correct = 0
    total = 0
    for b_iq, b_band, b_qual, b_dt, b_target in loader_wm:
        opt_wm.zero_grad()
        logits, _, _ = world_model(b_iq, b_band, b_qual, b_dt)
        loss = crit_wm(logits, b_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
        opt_wm.step()
        total_loss += loss.item() * len(b_target)
        preds = logits.argmax(dim=-1)
        correct += (preds == b_target).sum().item()
        total += len(b_target)
        
    if (epoch + 1) % 5 == 0:
        acc = (correct / total) * 100.0
        print(f"      Epoch {epoch+1:02d}/20 | Next-Band Loss: {total_loss / SEQ_COUNT:.4f} | Hop Prediction Accuracy: {acc:.1f}%")

# -------------------------------------------------------------
# 5. Save Final Unified Bundle
# -------------------------------------------------------------
save_dir = Path.home() / "Documents" / "SmartScanArtifacts" / "track2"
save_dir.mkdir(parents=True, exist_ok=True)
target_path = save_dir / "track2_final_world_model.pt"

id_state = {}
for k, v in id_model.state_dict().items():
    if not k.startswith("classifier."):
        id_state[k] = v.cpu()

bundle = {
    "identity_model_state_dict": id_state,
    "state_model_state_dict": {k: v.cpu() for k, v in world_model.state_dict().items()},
    "prototype_matrix": prototypes.cpu(),
    "prototype_ids": list(range(10)),
    "embedding_dim": 64,
    "num_bands": 20,
    "num_classes": 21,
    "inactive_class": 20,
    "state_dim": 32,
    "band_embedding_dim": 8,
    "gru_hidden": 128,
    "n_fft": 64,
    "hop_length": 16,
    "identity_unknown_threshold": 0.72,
    "confirm_after": 3,
    "candidate_match_threshold": 0.75,
    "smoothing_alpha": 0.70,
    "anomaly_thresholds": {
        "identity_novelty": 0.1518,
        "behaviour_change": 0.9699,
        "prediction_uncertainty": 0.6425,
    },
    "anomaly_weights": {
        "identity_novelty": 0.45,
        "behaviour_change": 0.45,
        "prediction_uncertainty": 0.10,
    },
    "training_dataset": "Alan Turing Institute Synthetic Radar Dataset (8 Diverse Scenarios)",
    "timestamp": time.time(),
}

torch.save(bundle, target_path)
print(f"\n==================================================================")
print(f"  SUCCESS! Multi-Scenario Turing World Model trained and deployed:")
print(f"  Path: {target_path}")
print(f"  Size: {target_path.stat().st_size / 1024:.1f} KB")
print(f"==================================================================")
