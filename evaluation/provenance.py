"""SmartScan Evaluation Provenance & RF Environmental SNR Definitions.

Provides canonical run metadata tracking (run_id, git commit hash, seeds, scenarios)
and the single source of truth for simulator SNR and noise floor definitions.
"""

from __future__ import annotations

import datetime
import math
import os
import subprocess
from pathlib import Path
from typing import Sequence
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ACTIVE_RUN_ID_FILE = ROOT_DIR / "results" / ".active_run_id"


def get_git_commit_hash(repo_dir: Path | None = None) -> str:
    """Retrieve current git commit hash, falling back gracefully if unavailable."""
    try:
        cwd = str(repo_dir) if repo_dir else str(ROOT_DIR)
        res = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return res
    except Exception:
        return "non_git_or_unversioned"


def set_active_suite_run_id(run_id: str | None = None) -> str:
    """Establish an active suite-wide run ID for cross-script provenance consistency."""
    if not run_id:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        run_id = f"smartscan_suite_{now_utc.strftime('%Y%m%d_%H%M%S')}"
    os.environ["SMARTSCAN_RUN_ID"] = run_id
    try:
        ACTIVE_RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_RUN_ID_FILE.write_text(run_id, encoding="utf-8")
    except Exception:
        pass
    return run_id


def get_active_suite_run_id() -> str | None:
    """Retrieve active suite-wide run ID from environment or disk marker."""
    env_id = os.environ.get("SMARTSCAN_RUN_ID")
    if env_id:
        return env_id.strip()
    if ACTIVE_RUN_ID_FILE.exists():
        try:
            val = ACTIVE_RUN_ID_FILE.read_text(encoding="utf-8").strip()
            if val:
                return val
        except Exception:
            pass
    return None


def get_run_metadata(
    experiment_name: str,
    seeds: Sequence[int] | None = None,
    scenarios: Sequence[str] | None = None,
    extra: dict | None = None,
    run_id: str | None = None,
) -> dict:
    """Generate canonical run provenance metadata dictionary."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    active_id = run_id or get_active_suite_run_id()
    if not active_id:
        active_id = f"{experiment_name}_{now_utc.strftime('%Y%m%d_%H%M%S')}"

    metadata = {
        "run_id": active_id,
        "experiment_name": experiment_name,
        "timestamp_utc": now_utc.isoformat(),
        "git_commit": get_git_commit_hash(),
        "seeds": [int(s) for s in seeds] if seeds is not None else [],
        "scenarios": list(scenarios) if scenarios is not None else [],
    }
    if extra:
        metadata.update(extra)
    return metadata


def compute_simulator_snr_db(
    signal_power: float,
    noise_std: float,
    num_samples: int = 512,
    interference_penalty_db: float = 0.0,
) -> float:
    """Compute effective baseband Signal-to-Noise Ratio (SNR) in decibels (dB).

    NOISE FLOOR DEFINITION IN THE SMARTSCAN SIMULATOR:
    --------------------------------------------------
    1. The simulator synthesizes baseband discrete I/Q samples of shape (2, 512).
    2. Additive Channel Noise is modeled as Zero-Mean Additive White Gaussian Noise (AWGN)
       injected independently onto each orthogonal channel:
           I[t] ~ Normal(0, sigma_n^2)
           Q[t] ~ Normal(0, sigma_n^2)
       where sigma_n = config.iq_noise_std (default: 0.15).
    3. Noise Variance / Noise Power:
       The total expected complex noise variance is:
           P_noise = E[|n[t]|^2] = E[I[t]^2 + Q[t]^2] = 2 * (sigma_n^2)
       For sigma_n = 0.15:
           P_noise = 2 * (0.15^2) = 2 * 0.0225 = 0.0450 (-13.47 dB)
    4. Signal Power:
       For a continuous-wave / pulsed sinusoid with envelope amplitude A (signal_strength):
           s(t) = A * cos(omega * t + phi)
       The average baseband signal power over window N is:
           P_signal = (1/N) * sum(|s[t]|^2) approx 0.5 * A^2
    5. Effective SNR Formulation:
           SNR_linear = P_signal / P_noise = (0.5 * A^2) / (2 * sigma_n^2)
           SNR_dB = 10 * log10(SNR_linear) - interference_penalty_db
    6. Channel Effects:
       Under multi-path fading (Rayleigh envelope r ~ Rayleigh(sigma_r)), the instantaneous
       received power scales as r^2, inducing instantaneous SNR fluctuations. Hostile
       interference (DRFM jamming / tone jamming) injects non-coherent energy, modeled as
       an equivalent SINR degradation penalty.
    """
    total_noise_power = 2.0 * (noise_std ** 2)
    if total_noise_power <= 0.0:
        return float("inf")
    raw_snr_linear = max(signal_power, 1e-12) / total_noise_power
    raw_snr_db = 10.0 * math.log10(raw_snr_linear)
    effective_snr_db = raw_snr_db - interference_penalty_db
    return float(effective_snr_db)
