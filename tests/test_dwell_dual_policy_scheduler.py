import numpy as np
import pytest
from scheduler.dwell_dual_policy_scheduler import DwellDualPolicyScheduler

def test_dwell_dual_policy_scheduler_lifecycle():
    sched = DwellDualPolicyScheduler(num_bands=20, switch_penalty=0.08, dwell_inertia=0.85)
    for t in range(30):
        a = sched.select_band()
        assert 0 <= a < 20
        # If even, detect signal on that band with high quality
        det = 1 if t % 2 == 0 else 0
        qual = 0.9 if det else 0.0
        obs = {
            "selected_band": a,
            "detected": det,
            "quality": np.array([qual], dtype=np.float32),
            "iq": np.zeros((2, 512), dtype=np.float32),
        }
        sched.update(a, float(det), obs)

    total_decisions = sched.exploit_decisions + sched.explore_decisions + sched.dwell_decisions
    assert total_decisions > 0
    assert sched.dwell_decisions > 0, "Scheduler should exhibit dwell lock-on decisions"


def test_smartscan_production_package_export():
    from scheduler import SmartScanProductionScheduler, DwellDualPolicyScheduler
    from scheduler.smartscan_production import SmartScanProductionScheduler as ProdClass

    assert SmartScanProductionScheduler is ProdClass
    assert DwellDualPolicyScheduler is ProdClass

    prod = SmartScanProductionScheduler(num_bands=20)
    band = prod.select_band()
    assert 0 <= band < 20
