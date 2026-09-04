"""Reusable online runtime for the frozen Track 2 emitter world model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scheduler.track2_core import (
    NUM_BANDS,
    EmitterStateWorldModel,
    RuntimeIdentityEncoder,
    TrackManager,
    build_global_belief_state,
    update_anomaly,
)


DEFAULT_MODEL_PATH = (
    Path.home()
    / "Documents"
    / "SmartScanArtifacts"
    / "track2"
    / "track2_final_world_model.pt"
)
DEFAULT_MAX_SCAN_AGE = 200.0
DEFAULT_MISS_DECAY = 0.90
DEFAULT_MISS_BELIEF_DISCOUNT = 0.75
DEFAULT_DIAGNOSTIC_DECAY = 0.90


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Track2Runtime:
    """Loads frozen Track 2 weights and maintains episode-local emitter tracks."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device=None,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
        miss_decay: float = DEFAULT_MISS_DECAY,
        miss_belief_discount: float = DEFAULT_MISS_BELIEF_DISCOUNT,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Track 2 checkpoint not found at {self.model_path}. "
                "Upload it to Documents/SmartScanArtifacts/track2/"
                "track2_final_world_model.pt."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if max_scan_age <= 0:
            raise ValueError("max_scan_age must be positive")
        self.max_scan_age = float(max_scan_age)
        self.miss_decay = float(np.clip(miss_decay, 0.0, 1.0))
        self.miss_belief_discount = float(
            np.clip(miss_belief_discount, 0.0, 1.0)
        )
        self.bundle = _torch_load(self.model_path)
        self._load_frozen_models()
        self.current_time = 0.0
        self.last_result: dict[str, Any] | None = None
        self.update_count = 0
        self.reset()

    def _load_frozen_models(self):
        required = {
            "identity_model_state_dict",
            "state_model_state_dict",
            "prototype_matrix",
            "prototype_ids",
        }
        missing = required.difference(self.bundle)
        if missing:
            raise ValueError(
                f"Track 2 checkpoint is missing keys: {sorted(missing)}"
            )
        checkpoint_bands = int(self.bundle.get("num_bands", NUM_BANDS))
        if checkpoint_bands != NUM_BANDS:
            raise ValueError(
                f"Track 2 requires {NUM_BANDS} bands; checkpoint has {checkpoint_bands}"
            )

        identity_state = self.bundle["identity_model_state_dict"]
        self.identity_model = RuntimeIdentityEncoder(
            identity_state,
            embedding_dim=int(self.bundle.get("embedding_dim", 64)),
        ).to(self.device)
        self.state_model = EmitterStateWorldModel(
            num_bands=NUM_BANDS,
            state_dim=int(self.bundle.get("state_dim", 32)),
            band_embedding_dim=int(self.bundle.get("band_embedding_dim", 8)),
            gru_hidden=int(self.bundle.get("gru_hidden", 128)),
            n_fft=int(self.bundle.get("n_fft", 64)),
            hop_length=int(self.bundle.get("hop_length", 16)),
        ).to(self.device)
        self.state_model.load_state_dict(self.bundle["state_model_state_dict"])

        for model in (self.identity_model, self.state_model):
            model.eval()
            model.requires_grad_(False)

        self.prototype_matrix = F.normalize(
            torch.as_tensor(self.bundle["prototype_matrix"], dtype=torch.float32),
            p=2,
            dim=1,
        )
        self.prototype_ids = [
            int(value.item()) if torch.is_tensor(value) else int(value)
            for value in self.bundle["prototype_ids"]
        ]
        self.identity_threshold = float(
            self.bundle.get("identity_unknown_threshold", 0.7415)
        )
        self.confirm_after = int(self.bundle.get("confirm_after", 3))
        self.candidate_threshold = float(
            self.bundle.get("candidate_match_threshold", 0.75)
        )
        self.smoothing_alpha = float(self.bundle.get("smoothing_alpha", 0.70))
        self.anomaly_thresholds = self.bundle.get(
            "anomaly_thresholds",
            {
                "identity_novelty": 1.0,
                "behaviour_change": 1.0,
                "prediction_uncertainty": 1.0,
            },
        )

    def _new_manager(self) -> TrackManager:
        return TrackManager(
            identity_model=self.identity_model,
            state_model=self.state_model,
            prototype_matrix=self.prototype_matrix,
            prototype_ids=self.prototype_ids,
            identity_threshold=self.identity_threshold,
            confirm_after=self.confirm_after,
            candidate_threshold=self.candidate_threshold,
            device=self.device,
        )

    def reset(self):
        """Keep frozen weights/prototypes but clear every episode-local history."""
        self.manager = self._new_manager()
        self.current_time = 0.0
        self.last_result = None
        self.update_count = 0
        # Never-scanned bands begin maximally stale/uncertain, without using truth.
        self.scan_age = np.full(
            NUM_BANDS, self.max_scan_age, dtype=np.float32
        )
        self.recent_miss = np.zeros(NUM_BANDS, dtype=np.float32)
        self.recent_hit = np.zeros(NUM_BANDS, dtype=np.float32)
        self.prediction_error = np.zeros(NUM_BANDS, dtype=np.float32)
        return self.get_band_belief()

    def update(self, obs: dict, timestamp: float):
        self.current_time = float(timestamp)
        selected_band = int(obs["selected_band"])
        if not 0 <= selected_band < NUM_BANDS:
            raise ValueError(f"Selected band must be in 0..{NUM_BANDS - 1}")

        predicted_presence = float(self.get_band_belief()[selected_band])
        detected = bool(obs["detected"])
        observed_presence = 1.0 if detected else 0.0
        self.prediction_error *= DEFAULT_DIAGNOSTIC_DECAY
        self.prediction_error[selected_band] = float(
            0.8 * self.prediction_error[selected_band]
            + 0.2 * abs(observed_presence - predicted_presence)
        )
        self.recent_hit *= DEFAULT_DIAGNOSTIC_DECAY

        self.scan_age = np.minimum(
            self.scan_age + 1.0, self.max_scan_age
        ).astype(np.float32)
        self.scan_age[selected_band] = 0.0
        self.recent_miss *= self.miss_decay

        if not detected:
            self.recent_miss[selected_band] = 1.0
            self.manager.record_miss(selected_band)
            self.last_result = None
            return None
        self.recent_miss[selected_band] = 0.0
        self.recent_hit[selected_band] = float(
            np.clip(np.asarray(obs["quality"]).reshape(-1)[0], 0.0, 1.0)
        )

        iq = np.asarray(obs["iq"], dtype=np.float32)
        if iq.shape != (2, 512):
            raise ValueError(f"Track 2 expects I/Q shape (2, 512), received {iq.shape}")
        result = self.manager.process_observation(
            iq=iq,
            band=selected_band,
            timestamp=self.current_time,
            quality=float(np.asarray(obs["quality"]).reshape(-1)[0]),
        )
        track = self.manager.tracks[result["track_id"]]
        update_anomaly(
            track,
            result,
            self.anomaly_thresholds,
            smoothing_alpha=self.smoothing_alpha,
        )
        self.last_result = result
        self.update_count += 1
        return result

    def get_band_belief(self) -> np.ndarray:
        belief = build_global_belief_state(
            self.manager, current_time=self.current_time
        )["band_belief"]
        belief = belief.detach().cpu().numpy()
        # A recent observable miss discounts, but never fabricates, emitter evidence.
        belief *= 1.0 - self.miss_belief_discount * self.recent_miss
        return np.clip(belief, 0.0, 1.0).astype(np.float32, copy=True)

    def get_scan_age(self, normalized: bool = True) -> np.ndarray:
        age = self.scan_age.copy()
        if normalized:
            age = np.clip(age / self.max_scan_age, 0.0, 1.0)
        return age.astype(np.float32, copy=False)

    def get_band_uncertainty(self) -> np.ndarray:
        """Aggregate track uncertainty by predicted per-band probability mass.

        Confirmed tracks contribute ``P(track in band) * track uncertainty``.
        Where track evidence is weak or absent, observable scan recency supplies
        uncertainty: never-scanned/stale bands are high, while a fresh miss is low.
        """
        probability_mass = np.zeros(NUM_BANDS, dtype=np.float32)
        weighted_uncertainty = np.zeros(NUM_BANDS, dtype=np.float32)
        for track in self.manager.tracks.values():
            if (
                not track.confirmed
                or track.observations <= 0
                or track.next_probabilities is None
            ):
                continue
            probabilities = (
                track.next_probabilities[:NUM_BANDS]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            probability_mass += probabilities
            weighted_uncertainty += probabilities * float(
                np.clip(track.uncertainty, 0.0, 1.0)
            )

        track_uncertainty = np.ones(NUM_BANDS, dtype=np.float32)
        has_evidence = probability_mass > 1e-8
        track_uncertainty[has_evidence] = (
            weighted_uncertainty[has_evidence]
            / probability_mass[has_evidence]
        )

        evidence_strength = np.clip(probability_mass, 0.0, 1.0)
        # A fresh miss is valid negative evidence and temporarily suppresses stale
        # model evidence for that band. Its influence decays automatically.
        evidence_strength *= 1.0 - self.recent_miss
        recency_uncertainty = self.get_scan_age(normalized=True)
        uncertainty = (
            evidence_strength * track_uncertainty
            + (1.0 - evidence_strength) * recency_uncertainty
        )
        return np.clip(uncertainty, 0.0, 1.0).astype(np.float32)

    def get_rl_state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.get_band_belief(),
                self.get_scan_age(normalized=True),
                self.get_band_uncertainty(),
            ]
        ).astype(np.float32)

    def get_prediction_error(self) -> np.ndarray:
        return np.clip(self.prediction_error, 0.0, 1.0).astype(
            np.float32, copy=True
        )

    def get_recent_hit(self) -> np.ndarray:
        return np.clip(self.recent_hit, 0.0, 1.0).astype(np.float32, copy=True)

    def get_recent_miss(self) -> np.ndarray:
        return np.clip(self.recent_miss, 0.0, 1.0).astype(np.float32, copy=True)

    def get_candidate_summary(self) -> dict[str, float]:
        candidates = [
            track
            for track in self.manager.tracks.values()
            if track.observations > 0 and not track.confirmed
        ]
        if not candidates:
            return {
                "count": 0.0,
                "mean_novelty": 0.0,
                "max_novelty": 0.0,
                "mean_quality": 0.0,
                "mean_uncertainty": 0.0,
            }
        novelty = np.asarray([track.novelty for track in candidates], dtype=np.float32)
        quality = np.asarray(
            [track.last_quality for track in candidates], dtype=np.float32
        )
        uncertainty = np.asarray(
            [track.uncertainty for track in candidates], dtype=np.float32
        )
        return {
            "count": float(len(candidates)),
            "mean_novelty": float(np.clip(novelty.mean(), 0.0, 1.0)),
            "max_novelty": float(np.clip(novelty.max(), 0.0, 1.0)),
            "mean_quality": float(np.clip(quality.mean(), 0.0, 1.0)),
            "mean_uncertainty": float(np.clip(uncertainty.mean(), 0.0, 1.0)),
        }

    def get_investigation_priority(self) -> np.ndarray:
        """Translates emitter-level anomalies and new candidates into band-level investigation scores."""
        priority = np.zeros(NUM_BANDS, dtype=np.float32)

        # 1. Confirmed tracks: anomaly weighted by next-band probability
        for track in self.manager.tracks.values():
            if not track.confirmed or track.next_probabilities is None:
                continue
            anomaly = float(getattr(track, "anomaly_score", 0.0))
            if anomaly > 0.01:
                probs = track.next_probabilities[:NUM_BANDS].detach().cpu().numpy().astype(np.float32)
                priority += anomaly * probs

        # 2. Candidate tracks: temporary confirmation boost on recently observed band
        for track in self.manager.tracks.values():
            if track.confirmed or track.observations <= 0:
                continue
            last_band = getattr(track, "last_observed_band", None)
            if last_band is not None and 0 <= last_band < NUM_BANDS:
                age = max(0.0, self.current_time - track.last_observed_time)
                decay = np.exp(-age / 5.0)
                novelty = float(getattr(track, "novelty", 0.5))
                quality = float(getattr(track, "last_quality", 0.5))
                confirmation_boost = novelty * quality * decay
                priority[last_band] += float(confirmation_boost)

        # 3. Disagreement / prediction error signal
        priority += 0.20 * self.get_prediction_error()

        return np.clip(priority, 0.0, 1.0).astype(np.float32)

    def get_prediction_reliability(self) -> float:
        """Online calibration metric reflecting world-model forecast reliability in [0, 1]."""
        if self.update_count < 5:
            return 0.50
        mean_err = float(np.mean(self.prediction_error))
        # High prediction error (> 0.5) drives reliability toward 0
        reliability = np.clip(1.0 - 1.8 * mean_err, 0.0, 1.0)
        return float(reliability)

    def get_global_belief(self) -> dict:
        belief = build_global_belief_state(
            self.manager, current_time=self.current_time
        )
        return {
            "track_features": belief["track_features"].cpu().numpy().astype(np.float32),
            "track_diagnostics": belief["track_diagnostics"].cpu().numpy().astype(
                np.float32
            ),
            "track_mask": belief["track_mask"].cpu().numpy().astype(np.float32),
            "band_belief": self.get_band_belief(),
            "scan_age": self.get_scan_age(normalized=True),
            "band_uncertainty": self.get_band_uncertainty(),
            "investigation_priority": self.get_investigation_priority(),
            "prediction_reliability": self.get_prediction_reliability(),
            "prediction_error": self.get_prediction_error(),
            "recent_hit": self.get_recent_hit(),
            "recent_miss": self.get_recent_miss(),
            "candidate_summary": self.get_candidate_summary(),
            "track_ids": list(belief["track_ids"]),
        }
