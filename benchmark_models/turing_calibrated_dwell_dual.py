"""Turing-calibrated Dwell-Dual variant.

This preserves the original champion and changes only dwell inertia. The value
was selected on HF Turing development missions and must be validated separately
before becoming the default policy.
"""

from __future__ import annotations

from typing import Optional

from benchmark_models.dwell_dual_policy import DwellDualPolicyScheduler
from scheduler.track2_runtime import DEFAULT_MODEL_PATH


class TuringCalibratedDwellDualPolicyScheduler(DwellDualPolicyScheduler):
    """Dwell-Dual tuned for the downloaded Turing synthetic radar benchmark."""

    def __init__(
        self,
        num_bands: int = 20,
        *,
        dwell_inertia: float = 2.40,
        seed: Optional[int] = None,
        model_path=DEFAULT_MODEL_PATH,
        **kwargs,
    ):
        super().__init__(
            num_bands=num_bands,
            dwell_inertia=dwell_inertia,
            seed=seed,
            model_path=model_path,
            **kwargs,
        )
