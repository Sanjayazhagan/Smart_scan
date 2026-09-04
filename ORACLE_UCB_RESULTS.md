# Oracle-UCB feasibility result

## Purpose

Oracle-UCB is an evaluation-only upper bound. Before each scored simulator
action, the benchmark injects a binary vector identifying bands that will be
active during the imminent scan. This vector replaces the learned world-model
belief in the same UCB decision structure. It is never added to the production
observation or Track 2 runtime.

## Implementation

- `evaluation/oracle_ucb.py`: evaluation-only scheduler. It refuses to select
  an action unless the benchmark has injected the oracle vector.
- `evaluation/benchmark.py`: injects hidden activity only when the scheduler is
  explicitly marked `evaluation_only`.
- `tests/test_benchmark.py`: verifies benchmark injection and direct-use
  refusal.

The oracle is aligned to the action being selected: observations are available
through the preceding scan, while the oracle reveals activity for the imminent
scan. It is binary and does not receive threat weights, detection randomness,
or the numeric simulator reward.

## Development tuning

Weights 0.10, 0.25, 0.50, 1.00, and 2.00 were compared on development seeds
71001, 71011, and 71021 across stationary, hopping, changing, harsh, and
operational scenarios. Weight 2.00 had the highest cross-scenario mean reward
(85.42) and was frozen before held-out evaluation.

## Held-out validation

Held-out seeds: 81001, 81011, 81021, 81031, and 81041. Each configuration used
two episodes, an 80-step unscored warm-up, and 200 scored steps in each of five
scenarios.

| Scenario | UCB | World model off | Current world model | Oracle-UCB |
|---|---:|---:|---:|---:|
| Stationary | 41.97 | 39.19 | 36.45 | 123.62 |
| Hopping | 29.37 | 28.79 | 24.78 | 117.75 |
| Changing | 26.65 | 23.64 | 26.30 | 101.02 |
| Harsh | -1.17 | 1.19 | -2.33 | 6.69 |
| Operational | 4.28 | 1.51 | 2.73 | 36.47 |
| **Mean** | **20.22** | **18.87** | **17.59** | **77.11** |

Oracle-UCB improved over the identical UCB base with world-model guidance off
by 58.24 reward points on average across 25 paired scenario-seed results. The
approximate paired 95% interval was [44.69, 71.80], and the oracle won 24 of 25
pairs. Mean end-to-end control time was 3.60 ms.

## Interpretation

Perfect imminent-band information has very high value in this simulator.
Therefore the poor performance of the current World-Model UCB is not evidence
that predictive scheduling is useless. It indicates that the learned predictor
and its data pipeline are far below the available prediction ceiling.

Oracle-UCB is not deployable and must never be presented as achieved model
performance. The next work item is leakage-free, scenario/emitter-separated
training and held-out calibration of next-band and next-arrival-time forecasts.

Raw results:

- `results/oracle_ucb_development_tuning.json`
- `results/oracle_ucb_heldout_validation.json`
