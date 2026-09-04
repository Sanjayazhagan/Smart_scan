"""Frozen Track 2 neural models and persistent emitter tracking primitives.

Adapted directly from the supplied ``track2_final.py`` so the runtime uses the
same CNN, STFT, GRU, association, anomaly smoothing, and belief construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_BANDS = 20
NUM_CLASSES = 21
INACTIVE_CLASS = 20
MAX_TRACKS = 16
FEATURE_DIM = 48


class RuntimeIdentityEncoder(nn.Module):
    def __init__(self, saved_state: dict, embedding_dim: int = 64):
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
        if any(key.startswith("embedding_head.") for key in saved_state):
            self.embedding_type = "supcon"
            self.embedding_head = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(128, embedding_dim),
            )
        elif any(key.startswith("embedding.") for key in saved_state):
            self.embedding_type = "simple"
            self.embedding = nn.Linear(128, embedding_dim)
        else:
            raise RuntimeError("Unknown Track 2 identity checkpoint architecture")

        current_state = self.state_dict()
        compatible = {
            key: value
            for key, value in saved_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        current_state.update(compatible)
        self.load_state_dict(current_state)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        features = self.encoder(iq).flatten(1)
        embedding = (
            self.embedding_head(features)
            if self.embedding_type == "supcon"
            else self.embedding(features)
        )
        return F.normalize(embedding, p=2, dim=1)


class EmitterStateWorldModel(nn.Module):
    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        state_dim: int = 32,
        band_embedding_dim: int = 8,
        gru_hidden: int = 128,
        n_fft: int = 64,
        hop_length: int = 16,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("stft_window", torch.hann_window(n_fft))
        self.spec_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.state_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, state_dim),
        )
        self.band_embedding = nn.Embedding(num_bands, band_embedding_dim)
        self.gru = nn.GRU(
            input_size=state_dim + band_embedding_dim + 2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_bands + 1),
        )

    def make_spectrogram(self, iq: torch.Tensor) -> torch.Tensor:
        complex_signal = torch.complex(iq[:, 0, :], iq[:, 1, :])
        stft = torch.stft(
            complex_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.stft_window,
            return_complex=True,
            onesided=False,
        )
        spec = torch.log1p(torch.abs(stft))
        spec = torch.fft.fftshift(spec, dim=1)
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True)
        return ((spec - mean) / (std + 1e-6)).unsqueeze(1)

    def encode_state(self, iq: torch.Tensor) -> torch.Tensor:
        return self.state_head(self.spec_cnn(self.make_spectrogram(iq)).flatten(1))

    def step(self, iq, band, quality, delta_t, hidden=None):
        z_state = self.encode_state(iq)
        band_feature = self.band_embedding(
            torch.tensor([band], dtype=torch.long, device=iq.device)
        )
        quality_feature = torch.tensor([[quality]], dtype=torch.float32, device=iq.device)
        delta_feature = torch.tensor([[delta_t]], dtype=torch.float32, device=iq.device)
        gru_input = torch.cat(
            [z_state, band_feature, quality_feature, delta_feature], dim=-1
        ).unsqueeze(1)
        gru_output, new_hidden = self.gru(gru_input, hidden)
        probabilities = torch.softmax(
            self.prediction_head(gru_output[:, -1, :]), dim=-1
        )
        return z_state.squeeze(0), probabilities.squeeze(0), new_hidden

    def forward(
        self,
        iq_seq: torch.Tensor,
        band_seq: torch.Tensor,
        quality_seq: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ):
        B, L, C, N = iq_seq.shape
        iq_flat = iq_seq.reshape(B * L, C, N)
        z_state = self.encode_state(iq_flat).reshape(B, L, -1)
        band_features = self.band_embedding(band_seq)
        if quality_seq.ndim == 2:
            quality_seq = quality_seq.unsqueeze(-1)
        if delta_t_seq.ndim == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        gru_input = torch.cat(
            [z_state, band_features, quality_seq, delta_t_seq], dim=-1
        )
        gru_output, new_hidden = self.gru(gru_input, hidden)
        logits = self.prediction_head(gru_output[:, -1, :])
        return logits, z_state, new_hidden


@dataclass
class EmitterTrack:
    track_id: str
    prototype: torch.Tensor
    confirmed: bool = False
    known_identity: bool = False
    observations: int = 0
    last_timestamp: float | None = None
    last_band: int | None = None
    last_quality: float = 0.0
    latest_z_state: torch.Tensor | None = None
    gru_hidden: torch.Tensor | None = None
    next_probabilities: torch.Tensor | None = None
    uncertainty: float = 1.0
    novelty: float = 1.0
    behaviour_surprise: float = 0.0
    association_similarity: float = 0.0
    smoothed_novelty: float = 0.0
    smoothed_behaviour: float = 0.0
    smoothed_uncertainty: float = 0.0
    anomaly_updates: int = 0
    anomaly_score: float = 0.0
    anomaly_type: str = "NORMAL"
    prediction_error_ema: float = 0.0
    recent_hit_score: float = 0.0
    recent_miss_exposure: float = 0.0


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.reshape(1, -1), dim=1)
    b = F.normalize(b.reshape(1, -1), dim=1)
    return float((a @ b.T).item())


def normalized_entropy(probabilities: torch.Tensor) -> float:
    probabilities = probabilities.clamp(min=1e-12)
    entropy = -(probabilities * torch.log(probabilities)).sum()
    return float(entropy.item() / math.log(len(probabilities)))


def normalize_delta_t(delta_t: float) -> float:
    return float(np.log1p(np.clip(delta_t, 1, 100)) / np.log(101.0))


class TrackManager:
    def __init__(
        self,
        identity_model,
        state_model,
        prototype_matrix,
        prototype_ids,
        identity_threshold,
        confirm_after=3,
        candidate_threshold=0.75,
        device="cpu",
    ):
        self.identity_model = identity_model
        self.state_model = state_model
        self.identity_threshold = float(identity_threshold)
        self.confirm_after = int(confirm_after)
        self.candidate_threshold = float(candidate_threshold)
        self.device = torch.device(device)
        self.tracks: dict[str, EmitterTrack] = {}
        self.known_prototypes: dict[str, torch.Tensor] = {}
        self.next_unknown_id = 1
        for index, emitter_id in enumerate(prototype_ids):
            prototype = prototype_matrix[index].detach().cpu()
            track_id = f"E{int(emitter_id)}"
            self.known_prototypes[track_id] = prototype.clone()
            self.tracks[track_id] = EmitterTrack(
                track_id, prototype.clone(), confirmed=True, known_identity=True
            )

    def get_identity_embedding(self, iq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.identity_model(iq)[0].detach().cpu()

    def _best_match(self, z_id: torch.Tensor, known: bool):
        matches = [
            (cosine_similarity(z_id, track.prototype), track_id)
            for track_id, track in list(self.tracks.items())
            if track.known_identity is known
        ]
        if not matches:
            return None, -1.0
        similarity, track_id = max(matches)
        return track_id, float(similarity)

    def create_candidate(self, z_id: torch.Tensor) -> str:
        track_id = f"U{self.next_unknown_id:03d}"
        self.next_unknown_id += 1
        self.tracks[track_id] = EmitterTrack(track_id, z_id.clone())
        return track_id

    def update_temporal_model(self, track, iq, band, timestamp, quality):
        if track.next_probabilities is None:
            track.behaviour_surprise = 0.0
        else:
            observed_probability = float(
                track.next_probabilities[band].clamp(min=1e-12).item()
            )
            track.behaviour_surprise = float(-math.log(observed_probability))
            prediction_error = 1.0 - float(np.clip(observed_probability, 0.0, 1.0))
            track.prediction_error_ema = float(
                0.8 * track.prediction_error_ema + 0.2 * prediction_error
            )
        delta_t = (
            1.0
            if track.last_timestamp is None
            else max(1.0, timestamp - track.last_timestamp)
        )
        hidden = track.gru_hidden
        if hidden is not None:
            hidden = hidden.to(self.device)
        with torch.no_grad():
            z_state, probabilities, new_hidden = self.state_model.step(
                iq, band, quality, normalize_delta_t(delta_t), hidden
            )
        track.latest_z_state = z_state.detach().cpu()
        track.next_probabilities = probabilities.detach().cpu()
        track.gru_hidden = new_hidden.detach().cpu()
        track.uncertainty = normalized_entropy(track.next_probabilities)

    def record_miss(self, band: int):
        """Attach an observable empty scan to tracks only as soft exposure.

        A miss cannot be assigned to a specific emitter, so each confirmed
        track receives exposure proportional to its own predicted probability
        for the scanned band. No hidden identity or simulator truth is used.
        """
        band = int(band)
        for track in self.tracks.values():
            if (
                not track.confirmed
                or track.observations <= 0
                or track.next_probabilities is None
            ):
                continue
            exposure = float(
                np.clip(track.next_probabilities[band].item(), 0.0, 1.0)
            )
            track.recent_miss_exposure = float(
                0.8 * track.recent_miss_exposure + 0.2 * exposure
            )
            track.recent_hit_score *= 0.95

    def process_observation(self, iq, band, timestamp, quality):
        iq = torch.as_tensor(iq, dtype=torch.float32)
        if iq.ndim == 2:
            iq = iq.unsqueeze(0)
        iq = iq.to(self.device)
        band, timestamp, quality = int(band), float(timestamp), float(quality)
        z_id = self.get_identity_embedding(iq)
        best_known_id, best_known_similarity = self._best_match(z_id, True)
        novelty = float(np.clip(1.0 - best_known_similarity, 0.0, 1.0))
        created_new = False

        if best_known_id is not None and best_known_similarity >= self.identity_threshold:
            assigned_track, similarity = best_known_id, best_known_similarity
        else:
            unknown_id, unknown_similarity = self._best_match(z_id, False)
            if unknown_id is not None and unknown_similarity >= self.candidate_threshold:
                assigned_track, similarity = unknown_id, unknown_similarity
            else:
                assigned_track = self.create_candidate(z_id)
                similarity = best_known_similarity
                created_new = True

        track = self.tracks[assigned_track]
        previous_next_probabilities = (
            None
            if track.next_probabilities is None
            else track.next_probabilities.detach().cpu().clone()
        )
        track.observations += 1
        track.novelty = novelty
        track.association_similarity = float(similarity)
        if not track.known_identity:
            track.prototype = F.normalize(0.8 * track.prototype + 0.2 * z_id, dim=0)
        confirmed_now = False
        if not track.confirmed and track.observations >= self.confirm_after:
            track.confirmed = True
            confirmed_now = True

        self.update_temporal_model(track, iq, band, timestamp, quality)
        track.recent_hit_score = float(
            0.8 * track.recent_hit_score + 0.2 * np.clip(quality, 0.0, 1.0)
        )
        track.recent_miss_exposure *= 0.8
        track.last_timestamp = timestamp
        track.last_band = band
        track.last_quality = quality
        return {
            "track_id": assigned_track,
            "confirmed": track.confirmed,
            "known_identity": track.known_identity,
            "created_new": created_new,
            "confirmed_now": confirmed_now,
            "similarity": float(similarity),
            "best_known_similarity": float(best_known_similarity),
            "identity_novelty": novelty,
            "uncertainty": float(track.uncertainty),
            "behaviour_surprise": float(track.behaviour_surprise),
            # Diagnostic snapshot of the forecast that was evaluated by this
            # observation. It is never fed back into the model or scheduler.
            "previous_next_probabilities": previous_next_probabilities,
            "next_probabilities": track.next_probabilities,
        }


def update_anomaly(track, result, thresholds, smoothing_alpha=0.70):
    novelty = float(np.clip(result["identity_novelty"], 0.0, 1.0))
    uncertainty = float(np.clip(result["uncertainty"], 0.0, 1.0))
    behaviour = float(1.0 - np.exp(-max(0.0, result["behaviour_surprise"])))
    if track.anomaly_updates == 0:
        track.smoothed_novelty = novelty
        track.smoothed_behaviour = behaviour
        track.smoothed_uncertainty = uncertainty
    else:
        alpha = float(smoothing_alpha)
        track.smoothed_novelty = alpha * track.smoothed_novelty + (1 - alpha) * novelty
        track.smoothed_behaviour = alpha * track.smoothed_behaviour + (1 - alpha) * behaviour
        track.smoothed_uncertainty = alpha * track.smoothed_uncertainty + (1 - alpha) * uncertainty
    track.anomaly_updates += 1

    identity_flag = (
        track.anomaly_updates >= 3
        and track.smoothed_novelty > thresholds["identity_novelty"]
    )
    behaviour_flag = track.smoothed_behaviour > thresholds["behaviour_change"]
    uncertainty_flag = (
        track.smoothed_uncertainty > thresholds["prediction_uncertainty"]
    )
    if identity_flag and behaviour_flag:
        track.anomaly_type = "IDENTITY + BEHAVIOUR ANOMALY"
    elif identity_flag:
        track.anomaly_type = "POSSIBLE NEW EMITTER"
    elif behaviour_flag:
        track.anomaly_type = "BEHAVIOUR CHANGE"
    elif uncertainty_flag:
        track.anomaly_type = "MODEL UNCERTAIN"
    else:
        track.anomaly_type = "NORMAL"
    track.anomaly_score = float(
        np.clip(
            0.45 * track.smoothed_novelty
            + 0.45 * track.smoothed_behaviour
            + 0.10 * track.smoothed_uncertainty,
            0.0,
            1.0,
        )
    )


def build_global_belief_state(manager, current_time, max_tracks=MAX_TRACKS):
    tracks = [
        track
        for track in manager.tracks.values()
        if track.observations > 0 and track.confirmed
    ]
    tracks.sort(key=lambda track: -track.last_timestamp)
    tracks = tracks[:max_tracks]
    track_features = torch.zeros(max_tracks, FEATURE_DIM)
    track_diagnostics = torch.zeros(max_tracks, 3)
    track_mask = torch.zeros(max_tracks)
    band_none_probability = torch.ones(NUM_BANDS)
    track_ids = []
    for row_index, track in enumerate(tracks):
        probabilities = (
            torch.ones(NUM_CLASSES) / NUM_CLASSES
            if track.next_probabilities is None
            else track.next_probabilities.float().clone()
        )
        band_onehot = torch.zeros(NUM_BANDS)
        if track.last_band is not None:
            band_onehot[track.last_band] = 1.0
        recency = normalize_delta_t(max(1.0, current_time - track.last_timestamp))
        similarity = float(
            np.clip((track.association_similarity + 1.0) / 2.0, 0.0, 1.0)
        )
        extras = torch.tensor(
            [
                track.smoothed_uncertainty,
                track.smoothed_novelty,
                track.smoothed_behaviour,
                recency,
                track.last_quality,
                similarity,
                1.0,
            ],
            dtype=torch.float32,
        )
        track_features[row_index] = torch.cat(
            [probabilities, band_onehot, extras]
        )
        track_diagnostics[row_index] = torch.tensor(
            [
                track.prediction_error_ema,
                track.recent_hit_score,
                track.recent_miss_exposure,
            ],
            dtype=torch.float32,
        )
        track_mask[row_index] = 1.0
        track_ids.append(track.track_id)
        band_none_probability *= 1.0 - probabilities[:NUM_BANDS].clamp(0.0, 1.0)
    return {
        "track_features": track_features,
        "track_diagnostics": track_diagnostics,
        "track_mask": track_mask,
        "band_belief": 1.0 - band_none_probability,
        "track_ids": track_ids,
    }
