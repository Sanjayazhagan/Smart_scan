"""SmartScan Production Package.

Primary Production Architecture:
- SmartScanProductionScheduler (alias: DwellDualPolicyScheduler):
  The empirically validated #1 architecture across all 14 models, 6 operational
  defense scenarios, and 189,000 real-time decisions (+17.72 mean reward, 0.20 ms latency).

All comparative research, ablation, and baseline architectures (SmartScan-Omni,
Expectimax, MoE, Whittle RMAB, etc.) have been isolated into scheduler.research_models.
"""

from scheduler.smartscan_production import (
    SmartScanProductionScheduler,
    DwellDualPolicyScheduler,
)

__all__ = [
    "SmartScanProductionScheduler",
    "DwellDualPolicyScheduler",
]
