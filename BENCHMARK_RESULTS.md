# Smart Scan benchmark status

## Honest conclusion

No scheduler is proven universally best.

- Standard UCB1 is extremely fast and often scores well in the simulator, but
  it learns from simulator reward. That reward relies on hidden truth and is
  therefore a benchmark baseline, not a deployable RF controller.
- Observable Discounted UCB is the fair fast baseline. It uses only detections,
  quality, belief uncertainty, and scan history.
- Adaptive MoE is the most complete architecture and has achieved the best
  cross-condition mean in some held-out evaluations, but it does not beat the
  best specialist on every seed set.
- MPP-60 and the belief tree can win individual stationary or harsh cases but
  cost substantially more compute.
- Turbo-MoE was removed because its routing and algebraic specialists did not
  reproduce the proposed reward or latency claims.

## Retained controlled evidence

The repository keeps three result artifacts:

1. `results/warmup_heldout_validation.json` — multi-scenario evaluation after
   an unscored fixed-order 80-step calibration phase.
2. `results/track2_heldout_diagnostic.json` — Track 2 belief, GRU forecast, and
   identity consistency measured independently of scheduler reward.
3. `results/grand_benchmark_results.json` — compact comparative paradigm
   snapshot.

The warm-up held-out report previously produced these four-condition means:

| Scheduler | Mean reward across conditions |
|---|---:|
| Adaptive MoE | 19.03 |
| Standard UCB1 | 17.39 |
| MPP-60 | 16.82 |
| Observation-dependent belief tree | 16.60 |

These numbers describe that exact seed set and simulator version. Later seed
sets sometimes favored UCB1. Therefore the correct claim is **Adaptive is a
promising all-rounder, while UCB1 is a very strong simulator baseline**.

## Measurement rules

- Hidden worlds are generated before scheduling and do not depend on policy
  actions.
- Truth is available only to offline metrics and the simulator reward.
- Adaptive MoE, Track 2, and Observable UCB never receive hidden truth.
- Report planning time, update time, and their sum. Queue-pop latency alone is
  not end-to-end controller latency.
- Tune thresholds on development seeds and publish results only on untouched
  seeds.
- Use several episodes and confidence intervals before making superiority
  claims.

## Remaining validation gaps

- SDR/over-the-air testing.
- Broader hardware-emitter identity data.
- Explicit deceptive waveform and coordinated jammer models.
- Real retune latency, dwell constraints, and compute/power budgets.
- Statistical comparison on a frozen simulator configuration.

Until those gaps are closed, this repository is a research prototype rather
than an industry-validated product.
