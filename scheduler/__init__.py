"""SmartScan scheduler package.

Production champion:
    SmartScanProductionScheduler / DwellDualPolicyScheduler

The production implementation is PDW/observation-only and uses:
- interruptible recurring stale-band coverage,
- smart urgency ordering of stale bands,
- NMF exploitation,
- uncertainty-driven scouting,
- active-signal dwell inertia.
"""

from scheduler.smartscan_production import (
    SmartScanProductionScheduler,
    DwellDualPolicyScheduler,
)

__all__ = ["SmartScanProductionScheduler", "DwellDualPolicyScheduler"]
