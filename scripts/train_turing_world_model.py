"""Train Track 2 World Model on Alan Turing Institute Radar Dataset.

Downloads scenario in memory, extracts realistic radar pulses, trains 1D CNN
identity encoder and 2D Spectrogram GRU world model, packages the final bundle,
and saves directly to SmartScanArtifacts/track2/track2_final_world_model.pt.
Zero disk bloat: runs in memory and cleans up immediately.
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
from torch.utils.data import DataLoader, Dataset

# 1. Download scenario directly into RAM (Zero disk storage)
print("[1/5] Downloading Turing Synthetic Radar Dataset scenario into memory...")
token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text(encoding="utf-8-sig").strip()
ctx = ssl._create_unverified_context()

url = "https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset/resolve/main/scan/train_scan/config_0.h5"
req = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(req, context=ctx) as resp:
    data_bytes = resp.read()

print(f"      Downloaded {len(data_bytes) / (1024*1024):.2f} MB into memory.")

# 2. Parse HDF5 in RAM
print("[2/5] Parsing radar pulse descriptor words (PDWs)...")
with h5py.File(io.BytesIO(data_bytes), "r") as f:
    raw_data = f["data"][:]  # [ToA, Frequency, PulseWidth, AoA, Amplitude]
    raw_labels = f["labels"][:].flatten()

del data_bytes  # Free RAM immediately!

print(f"      Total pulses in scenario: {len(raw_labels):,}")
top_emitters = [e for e, count in Counter(raw_labels).most_common(10)]
print(f"      Selected top 10 radar emitters: {top_emitters}")

# Filter for top 10 emitters
mask = np.isin(raw_labels, top_emitters)
filtered_data = raw_data[mask]
filtered_labels_raw = raw_labels[mask]

# Remap emitter labels to 0..9
label_map = {old_id: new_id for new_id, old_id in enumerate(top_emitters)}
emitter_labels = np.array([label_map[e] for e in filtered_labels_raw], dtype=np.int64)

# Frequency mapping to 20 receiver channels (Bands 0..19)
freqs = filtered_data[:, 1]
f_min, f_max = np.percentile(freqs, 2), np.percentile(freqs, 98)
band_indices = np.clip(
    ((freqs - f_min) / max(1e-5, f_max - f_min) * 20).astype(np.int64),
    0, 19
)

# Subsample 5,000 pulses for fast, high-quality local training
N = min(5000, len(emitter_labels))
indices = np.random.RandomState(42).choice(len(emitter_labels), N, replace=False)
indices.sort()

toas = filtered_data[indices, 0]
freqs = filtered_data[indices, 1]
pws = filtered_data[indices, 2]
amps = filtered_data[indices, 4]
bands = band_indices[indices]
emitters = emitter_labels[indices]

# Synthesize physical 512 baseband I/Q waveforms from Turing pulse specs
print("[3/5] Synthesizing physical I/Q pulses from Turing radar specs...")
t = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
iq_samples = np.zeros((N, 2, 512), dtype=np.float32)

amp_norm = np.clip((amps - amps.min()) / max(1e-5, amps.max() - amps.min()), 0.4, 1.5)
pw_norm = np.clip(pws / max(1e-5, np.median(pws)), 0.3, 3.0)

for i in range(N):
    omega = 2.0 * np.pi * (bands[i] - 9.5) * 0.1
    env = np.exp(-0.5 * (t / (0.4 * pw_norm[i])) ** 2)
    noise = np.random.normal(0, 0.08, (2, 512)).astype(np.float32)
    iq_samples[i, 0] = amp_norm[i] * env * np.cos(omega * t) + noise[0]
    iq_samples[i, 1] = amp_norm[i] * env * np.sin(omega * t) + noise[1]

# Calculate next-band hopping target for each emitter
next_bands = np.full(N, 20, dtype=np.int64)  # 20 = inactive
for e in range(10):
    e_idx = np.where(emitters == e)[0]
    for k in range(len(e_idx) - 1):
        curr_i = e_idx[k]
        next_i = e_idx[k + 1]
        if toas[next_i] - toas[curr_i] < 50000.0:  # within pulse train window
            next_bands[curr_i] = bands[next_i]

print("      Pulse synthesis complete. Ready for training.")

# -------------------------------------------------------------
# 4. Train Model 1: RuntimeIdentityEncoder (1D CNN)
# -------------------------------------------------------------
print("[4/5] Training Model 1: 1D-ConvNet Identity Encoder...")
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

dataset_id = torch.utils.data.TensorDataset(X_iq, Y_emitter)
loader_id = DataLoader(dataset_id, batch_size=32, shuffle=True)

id_model.train()
for epoch in range(12):
    total_loss = 0.0
    for batch_x, batch_y in loader_id:
        opt_id.zero_grad()
        _, logits = id_model(batch_x)
        loss = crit_id(logits, batch_y)
        loss.backward()
        opt_id.step()
        total_loss += loss.item() * len(batch_y)
    if (epoch + 1) % 4 == 0:
        print(f"      Epoch {epoch+1:02d}/12 | Identity Loss: {total_loss / N:.4f}")

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
# 5. Train Model 2: EmitterStateWorldModel (2D CNN + Temporal GRU)
# -------------------------------------------------------------
print("[5/5] Training Model 2: 2D Spectrogram CNN + Temporal GRU...")
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

# Balanced class weights to prevent predicting inactive all the time
class_counts = Counter(next_bands)
weights = np.zeros(21, dtype=np.float32)
for c in range(21):
    weights[c] = 1.0 / max(5, class_counts.get(c, 1))
weights[:20] *= 3.0  # Boost hopping transitions!
weights /= weights.sum()
crit_wm = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

# Build sequential windows (sequence length = 4 pulses)
seq_len = 4
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
X_seq_qual = torch.tensor(np.array(X_seq_qual), dtype=torch.float32).unsqueeze(-1)
X_seq_dt = torch.tensor(np.array(X_seq_dt), dtype=torch.float32).unsqueeze(-1)
Y_seq_next = torch.tensor(np.array(Y_seq_next), dtype=torch.long)

print(f"      Constructed {len(Y_seq_next)} sequential pulse windows.")
loader_wm = DataLoader(
    torch.utils.data.TensorDataset(X_seq_iq, X_seq_band, X_seq_qual, X_seq_dt, Y_seq_next),
    batch_size=32,
    shuffle=True
)

world_model.train()
for epoch in range(15):
    total_loss = 0.0
    for b_iq, b_band, b_qual, b_dt, b_target in loader_wm:
        opt_wm.zero_grad()
        logits, _, _ = world_model(b_iq, b_band, b_qual, b_dt)
        loss = crit_wm(logits, b_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
        opt_wm.step()
        total_loss += loss.item() * len(b_target)
    if (epoch + 1) % 5 == 0:
        print(f"      Epoch {epoch+1:02d}/15 | GRU Next-Band Loss: {total_loss / len(Y_seq_next):.4f}")

# -------------------------------------------------------------
# 6. Save Bundle to SmartScanArtifacts/track2/
# -------------------------------------------------------------
save_dir = Path.home() / "Documents" / "SmartScanArtifacts" / "track2"
save_dir.mkdir(parents=True, exist_ok=True)
target_path = save_dir / "track2_final_world_model.pt"

# Backup old model
if target_path.exists():
    backup_path = save_dir / "track2_final_world_model.pt.bak"
    if not backup_path.exists():
        import shutil
        shutil.copy2(target_path, backup_path)
        print(f"      Backed up existing model to {backup_path.name}")

# Prepare identity state dict matching RuntimeIdentityEncoder exactly
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
        "identity_drop": 0.20,
        "unexpected_band": 0.75,
        "timing_jitter": 0.25,
        "high_novelty": 0.60,
    },
    "training_dataset": "Alan Turing Institute Synthetic Radar Dataset (scan/train_scan/config_0.h5)",
}

torch.save(bundle, target_path)
print(f"\nSUCCESS! Saved trained Turing Radar World Model to:")
print(f"  --> {target_path}")
print(f"  --> File size: {target_path.stat().st_size / 1024:.1f} KB")
