import numpy as np

from scheduler.smartscan_production import SmartScanProductionScheduler


def _hit(q=0.9):
    return {
        "detected": 1,
        "quality": np.array([q], dtype=np.float32),
        "signal_power": np.array([1.0], dtype=np.float32),
        "pdw_count": 3,
        "pdw_summary": np.zeros(5, dtype=np.float32),
    }


def test_production_is_observation_only():
    s = SmartScanProductionScheduler(seed=1)
    assert s.world_model_scale == 0.0
    assert s.runtime.__class__.__name__ == "ObservationOnlyRuntime"


def test_active_hit_interrupts_forced_coverage():
    s = SmartScanProductionScheduler(seed=1)
    s.counts[:] = 1.0
    s.counts[5] = 0.1
    s.last_band = 2
    s.last_detected = True
    assert s.select_band() == 2
    assert s.last_governing_policy == "coverage_interrupt_dwell"


def test_smart_stale_ordering_can_skip_first_stale_index():
    s = SmartScanProductionScheduler(seed=1)
    s.counts[:] = 1.0
    s.counts[[3, 12]] = 0.1
    s.last_band = 10
    s.last_detected = False
    s.uncertainty_estimator.scan_age[3] = 2
    s.uncertainty_estimator.scan_age[12] = 30
    s.values[3] = 0.0
    s.values[12] = 0.9
    assert s.select_band() == 12
    assert s.last_governing_policy == "smart_forced_coverage"


def test_update_accepts_pdw_observation_without_iq():
    s = SmartScanProductionScheduler(seed=1)
    band = s.select_band()
    s.update(band, 1.0, _hit())
    assert s.last_detected is True
