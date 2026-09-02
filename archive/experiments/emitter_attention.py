"""Fixed-width masked attention representation for emitter-aware RL experts."""

from __future__ import annotations

import numpy as np

from scheduler.emitter_aware_predictive import (
    BEHAVIOUR_CHANGE,
    CONFIRMED,
    IDENTITY_NOVELTY,
    IDENTITY_SIMILARITY,
    OBSERVATION_QUALITY,
    PREDICTION_UNCERTAINTY,
    TRACK_RECENCY,
)
from scheduler.track2_core import MAX_TRACKS, NUM_BANDS


ATTENTION_FEATURES_PER_BAND = 14
ATTENTION_GLOBAL_FEATURES = 8
TRACK2_ATTENTION_FEATURES = (
    NUM_BANDS * ATTENTION_FEATURES_PER_BAND + ATTENTION_GLOBAL_FEATURES
)


class EmitterAttentionEncoder:
    """Attention-pool variable masked tracks into a stable 288-D state."""

    def __init__(self, temperature: float = 0.25):
        if temperature <= 0:
            raise ValueError("Attention temperature must be positive")
        self.temperature = float(temperature)
        self.last_attention = np.zeros((NUM_BANDS, MAX_TRACKS), dtype=np.float32)

    @staticmethod
    def _band_array(global_belief: dict, name: str) -> np.ndarray:
        value = np.asarray(global_belief.get(name, np.zeros(NUM_BANDS)), dtype=np.float32)
        if value.size != NUM_BANDS:
            return np.zeros(NUM_BANDS, dtype=np.float32)
        return np.clip(value.reshape(NUM_BANDS), 0.0, 1.0)

    def encode(self, global_belief: dict) -> np.ndarray:
        rows = np.asarray(global_belief.get("track_features", []), dtype=np.float32)
        mask = np.asarray(global_belief.get("track_mask", []), dtype=np.float32).reshape(-1)
        if rows.ndim != 2 or rows.shape[0] != mask.size or rows.shape[1] < 48:
            raise ValueError("Emitter attention requires aligned track_features [N,48]")
        valid = (mask > 0.5) & (rows[:, CONFIRMED] > 0.5)
        valid_rows = rows[valid]
        belief = self._band_array(global_belief, "band_belief")
        age = self._band_array(global_belief, "scan_age")
        uncertainty = self._band_array(global_belief, "band_uncertainty")
        prediction_error = self._band_array(global_belief, "prediction_error")
        recent_hit = self._band_array(global_belief, "recent_hit")
        recent_miss = self._band_array(global_belief, "recent_miss")

        pooled = np.zeros((NUM_BANDS, 6), dtype=np.float32)
        dominant = np.zeros(NUM_BANDS, dtype=np.float32)
        support_fraction = np.zeros(NUM_BANDS, dtype=np.float32)
        disagreement = np.zeros(NUM_BANDS, dtype=np.float32)
        self.last_attention = np.zeros((NUM_BANDS, rows.shape[0]), dtype=np.float32)
        if valid_rows.shape[0] > 0:
            probabilities = np.clip(valid_rows[:, :NUM_BANDS], 0.0, 1.0)
            quality = np.clip(valid_rows[:, OBSERVATION_QUALITY], 0.0, 1.0)
            metadata = np.clip(
                valid_rows[
                    :,
                    [
                        PREDICTION_UNCERTAINTY,
                        IDENTITY_NOVELTY,
                        BEHAVIOUR_CHANGE,
                        TRACK_RECENCY,
                        OBSERVATION_QUALITY,
                        IDENTITY_SIMILARITY,
                    ],
                ],
                0.0,
                1.0,
            )
            valid_indices = np.flatnonzero(valid)
            for band in range(NUM_BANDS):
                logits = probabilities[:, band] / self.temperature
                logits -= logits.max(initial=0.0)
                weights = np.exp(logits) * quality * (probabilities[:, band] > 1e-8)
                total = float(weights.sum())
                if total > 1e-8:
                    weights /= total
                    pooled[band] = weights @ metadata
                    self.last_attention[band, valid_indices] = weights
                dominant[band] = probabilities[:, band].max(initial=0.0)
                support_fraction[band] = float(
                    np.mean(probabilities[:, band] >= 0.05)
                )
                disagreement[band] = float(probabilities[:, band].std())

        per_band = np.column_stack(
            [
                belief,
                age,
                uncertainty,
                dominant,
                support_fraction,
                pooled,
                prediction_error,
                recent_hit,
                recent_miss,
            ]
        ).astype(np.float32)
        candidate = global_belief.get("candidate_summary", {}) or {}
        mass = float(belief.sum())
        distribution = belief / mass if mass > 1e-8 else np.full(NUM_BANDS, 1 / NUM_BANDS)
        entropy = float(
            -(distribution * np.log(np.clip(distribution, 1e-8, 1.0))).sum()
            / np.log(NUM_BANDS)
        )
        sorted_belief = np.sort(belief)[::-1]
        global_features = np.asarray(
            [
                min(valid_rows.shape[0] / MAX_TRACKS, 1.0),
                min(float(candidate.get("count", 0.0)) / 4.0, 1.0),
                float(np.clip(candidate.get("max_novelty", 0.0), 0.0, 1.0)),
                float(np.clip(candidate.get("mean_quality", 0.0), 0.0, 1.0)),
                entropy,
                float(np.clip(sorted_belief[0] - sorted_belief[1], 0.0, 1.0)),
                float(disagreement.mean()),
                float(prediction_error.mean()),
            ],
            dtype=np.float32,
        )
        encoded = np.concatenate([per_band.reshape(-1), global_features])
        if encoded.size != TRACK2_ATTENTION_FEATURES:
            raise RuntimeError("Emitter attention produced an invalid feature width")
        return np.clip(encoded, 0.0, 1.0).astype(np.float32)

