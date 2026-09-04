# Calibrated World-Model Ablation and Soft-MoE Results

## What was implemented

1. **Exact world-model ablation**
   - `NMF+UCB[world off]`: UCB and NMF with neural guidance set to exactly zero.
   - `NMF+UCB[world calibrated]`: the identical controller with neural guidance multiplied by online Brier skill.
2. **Online confidence calibration**
   - Uses only the probability assigned to the scanned band and its observed detection outcome.
   - Compares the world forecast's exponentially weighted Brier loss with an observable base-rate forecast.
   - Neural guidance remains zero unless the world forecast demonstrates positive skill.
3. **Soft observable router**
   - Blends UCB, NMF, RPCA, PRI and observable-feedback Exp3 scores instead of making one hard expert switch.
   - Computes soft stationary, hopping, harsh and dynamic weights from detection concentration, band transitions and signal quality.
   - Uses no simulator truth and ignores the simulator reward.
4. **Separated development and validation**
   - Router temperatures 0.35, 0.55 and 0.85 were compared only on development seeds.
   - Temperature 0.85 was frozen before the untouched validation run.

## Development screen

Seeds `130001`, `130011`, `130021`; five scenarios; one episode per seed; 80 warm-up and 200 scored steps.

| Rank | Scheduler | Mean reward |
|---:|---|---:|
| 1 | World+NMF UCB (fixed neural weight) | 27.110 |
| 2 | Calibrated Soft MoE (temperature 0.85) | 26.924 |
| 3 | Calibrated Soft MoE (temperature 0.35) | 26.768 |
| 4 | Calibrated Soft MoE (temperature 0.55) | 25.728 |
| 5 | Lean Observable MoE | 25.371 |

Temperature 0.85 was selected without consulting the final validation seeds.

## Untouched validation

Seeds `140001`, `140011`, `140021`, `140031`, `140041`; five scenarios; two episodes per seed; 80 warm-up and 200 scored steps. This is 50 episodes per scheduler and 350 scheduler episodes in total.

| Rank | Scheduler | Overall reward | Stationary | Hopping | Changing | Harsh | Operational | Control time |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Direct NMF | **23.703** | 70.680 | 23.690 | 21.246 | -5.862 | 8.763 | 0.329 ms |
| 2 | NMF+UCB, world off | **23.219** | 46.650 | **31.060** | 31.043 | **-1.813** | **9.153** | 3.512 ms |
| 3 | NMF+UCB, world calibrated | 22.916 | 48.220 | 30.600 | 31.401 | -3.923 | 8.282 | 5.058 ms |
| 4 | Lean Observable MoE | 22.509 | 60.130 | 28.540 | 23.676 | -2.538 | 2.736 | 3.145 ms |
| 5 | Calibrated Soft MoE (temperature 0.85) | 21.909 | 63.780 | 26.530 | 18.455 | -3.581 | 4.358 | 5.192 ms |
| 6 | World+NMF UCB, fixed neural weight | 20.852 | 40.800 | 30.370 | **32.625** | -4.533 | 4.998 | 3.443 ms |
| 7 | UCB | 18.970 | 39.880 | 30.080 | 27.694 | -4.796 | 1.993 | **0.009 ms** |

## Controlled conclusions

- Calibrated neural guidance minus neural-off: **-0.303 reward**, 95% CI **[-2.008, 1.403]**, 5/25 paired-cell wins. The current world model does not demonstrate incremental decision value.
- NMF+UCB world-off minus plain UCB: **+4.248**, 95% CI **[0.892, 7.605]**, 18/25 wins. This improvement is statistically supported in this validation matrix.
- NMF+UCB world-off minus soft MoE: **+1.310**, 95% CI **[-4.566, 7.186]**, 18/25 wins. The soft router did not improve the mean, and the interval does not establish a difference.
- Direct NMF minus NMF+UCB world-off: **+0.485**, 95% CI **[-6.141, 7.111]**. They are statistically tied overall, but NMF+UCB is much more balanced across non-stationary scenarios.

## Decision

The new soft router is retained as an experimental implementation, not promoted as champion. The best defensible balanced controller from this run is **NMF+UCB with world guidance disabled**. Direct NMF has the highest raw average because of its stationary score, while NMF+UCB wins hopping, harsh and operational conditions and significantly beats plain UCB.

Improving the world model requires better held-out forecasting—not a stronger router around its current predictions. The next model-training gate should require positive held-out Brier skill and a statistically supported gain over `NMF+UCB[world off]` before neural guidance is enabled.

## Artifacts

- Development results: `results/calibrated_soft_moe_development.json`
- Untouched validation: `results/calibrated_soft_moe_heldout.json`
- Implementation: `scheduler/calibrated_soft_moe.py`
- Tests: `tests/test_calibrated_soft_moe.py`
- Full test suite: **86 passed**
