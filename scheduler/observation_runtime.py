"""Observation-only runtime used by schedulers that do not consume Track-2 neural features.

This intentionally ignores I/Q.  It exists so heuristic schedulers such as
Dwell-Dual can operate directly on compact observation dictionaries (detected,
quality, signal_power) without requiring a Track-2 checkpoint.
"""
from __future__ import annotations

import numpy as np


class ObservationOnlyRuntime:
    """Minimal runtime interface compatible with WorldModelUCBScheduler.

    The Dwell-Dual policy sets ``world_model_scale=0`` and never reads neural
    beliefs in ``select_band``.  Its parent ``update`` still calls a runtime,
    so this adapter returns neutral metadata without examining I/Q.
    """

    def __init__(self, num_bands: int = 20):
        self.num_bands = int(num_bands)
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)

    def get_global_belief(self) -> dict:
        return {
            "band_belief": np.zeros(self.num_bands, dtype=np.float32),
            "scan_age": np.clip(self.scan_age / 30.0, 0.0, 1.0),
        }

    def update(self, obs_dict: dict | None, timestamp: float | None = None) -> dict:
        self.scan_age += 1.0
        if obs_dict is not None and "selected_band" in obs_dict:
            band = int(np.asarray(obs_dict["selected_band"]).reshape(-1)[0])
            if 0 <= band < self.num_bands:
                self.scan_age[band] = 0.0
        return {"known_identity": True, "pdw_native": True}
