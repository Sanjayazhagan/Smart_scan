"""Ablation variants of the interruptible Dual-Dwell architecture.

These variants were evaluated during the lightweight PDW comparison:
- InterruptibleDwellChampion (B_champion): Interruptible recurring stale coverage,
  servicing stale bands by lowest index order.
- SmartStaleDwellChampion (S_smart_stale): B + observable urgency ranking for stale bands
  (promoted to official champion: SmartScanProductionScheduler).
"""
from __future__ import annotations

import numpy as np
from scheduler.smartscan_production import SmartScanProductionScheduler


class InterruptibleDwellChampion(SmartScanProductionScheduler):
    """Interruptible Dual-Dwell baseline (B_champion).
    
    Services stale bands in first-index order rather than urgency-ranked order.
    """

    def _choose_coverage_candidate(self, candidates: np.ndarray) -> int:
        return int(candidates[0])


class SmartStaleDwellChampion(SmartScanProductionScheduler):
    """S_smart_stale: B + urgency-ranked stale-band ordering.
    
    Identical to SmartScanProductionScheduler.
    """
    pass
