import numpy as np

from scheduler.belief_tree import ObservationDependentBeliefTreePlanner

from tests.test_emitter_aware_predictive import FakeRuntime, track_row


def test_belief_tree_has_distinct_hit_and_miss_branches():
    rows = np.stack([track_row(3, 0.8), np.zeros(48), np.zeros(48), np.zeros(48)])
    planner = ObservationDependentBeliefTreePlanner(
        20, runtime=FakeRuntime(rows=rows, mask=np.array([1, 0, 0, 0]))
    )
    action = planner.select_band()
    tree = planner.last_decision_trace["belief_tree"]
    assert 0 <= action < 20
    assert tree["band"] == action
    assert 0.0 <= tree["detection_probability"] <= 1.0
    assert tree["hit"]["next"] != tree["miss"]["next"]
    assert len(planner.last_decision_trace["representative_sequence"]) == 3
    stats = planner.last_decision_trace["search_stats"]
    assert stats["branch_width"] == 4
    assert stats["expanded_branches"] < 100
    assert stats["planning_time_ms"] >= 0.0
