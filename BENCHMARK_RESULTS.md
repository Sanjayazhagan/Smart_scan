# Smart Scan benchmark results

## Controlled benchmark

The following results use the same simulator configuration for every scheduler:

- 20 bands
- 20 episodes
- 200 steps per episode
- seed 42
- frozen Track 2 model
- simulator truth used only for evaluation metrics

| Scheduler | Reward mean | Reward std | Median | TP | FP | Misses | Detection | Interception | Wasted scans | Unique bands | Revisit steps | Planning ms | Discovery delay | Discovery rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed / sequential | 24.835 | 9.708 | 25.450 | 19.00 | 9.00 | 2.15 | 0.898 | 0.042 | 0.894 | 20.00 | 20.000 | 0.000 | 34.824 | 0.791 |
| Random | 27.180 | 11.187 | 24.850 | 21.30 | 8.90 | 2.00 | 0.914 | 0.048 | 0.884 | 20.00 | 17.889 | 0.005 | 43.623 | 0.922 |
| Thompson Sampling | 25.875 | 8.778 | 28.200 | 20.40 | 9.50 | 1.85 | 0.917 | 0.046 | 0.889 | 19.30 | 11.430 | 0.025 | 44.644 | 0.783 |
| UCB | 36.270 | 16.184 | 37.450 | 33.60 | 8.60 | 4.10 | 0.891 | 0.074 | 0.812 | 20.00 | 19.389 | 0.014 | 46.653 | 0.878 |
| Frequency-History PPO | 19.665 | 31.592 | 9.000 | 14.05 | 10.20 | 1.60 | 0.898 | 0.032 | 0.922 | 1.00 | 1.000 | 0.393 | 2.000 | 0.035 |
| Track2 PPO[20] | 36.520 | 34.488 | 21.750 | 34.00 | 8.80 | 4.00 | 0.895 | 0.076 | 0.810 | 2.15 | 1.615 | 1.962 | 34.714 | 0.122 |
| MPP-BeliefOnly | 34.355 | 26.161 | 31.450 | 30.80 | 8.65 | 3.50 | 0.898 | 0.069 | 0.829 | 5.00 | 2.162 | 50.687 | 71.893 | 0.243 |
| MPP-60 | 58.785 | 27.525 | 59.150 | 58.80 | 6.90 | 5.95 | 0.908 | 0.132 | 0.676 | 20.00 | 16.144 | 54.110 | 47.102 | 0.765 |
| MPP-EmitterAware | 60.245 | 29.095 | 60.300 | 61.95 | 7.50 | 6.60 | 0.904 | 0.139 | 0.657 | 20.00 | 14.530 | 43.924 | 45.088 | 0.696 |
| MPP-EmitterAware minus anomaly | 60.505 | 28.137 | 67.350 | 61.75 | 6.85 | 6.85 | 0.900 | 0.138 | 0.657 | 20.00 | 15.093 | 32.973 | 47.518 | 0.739 |
| MPP-EmitterAware minus identity/recency | **61.905** | 27.057 | 60.900 | **64.10** | 6.60 | 7.75 | 0.892 | **0.142** | **0.641** | 20.00 | 14.801 | 32.387 | **44.731** | 0.678 |
| Track2 PPO[60] | 29.160 | 38.196 | 9.550 | 25.70 | 9.25 | 3.45 | 0.882 | 0.057 | 0.854 | 2.00 | 1.554 | 0.664 | 0.167 | 0.052 |

`behaviour_change_adaptation_delay` is currently unavailable because the
simulator does not expose timestamped behaviour-change labels. The benchmark
returns `null` rather than inventing this metric.

## Interpretation

Against unchanged MPP-60, full MPP-EmitterAware improved mean reward by 1.460
(2.5%), mean true positives by 3.15 (5.4%), and interception rate by 0.67
percentage points. It reduced wasted scans by 1.9 percentage points and mean
new-emitter discovery delay by about 2.0 steps. However, detection rate fell by
0.44 percentage points, false positives and misses increased, and distinct
emitter discovery rate fell by 6.96 percentage points.

Both term-removal ablations scored above the full planner. Removing
identity/recency produced the highest reward and interceptions, while removing
anomaly terms also improved reward slightly. This means the emitter-aware
feature path is promising, but the current auxiliary terms and hand-set weights
are not yet supported as a complete design.

The observed reward standard deviations are large and this is only one full
benchmark seed. The measured difference between MPP-60 and full
MPP-EmitterAware is therefore not evidence of a statistically reliable win.
Under the project's design rule, MPP-60 should remain the default system while
MPP-EmitterAware stays experimental until it wins across multiple held-out,
difficult seeds and scenarios.

## Example runtime decision trace

After 60 simulator steps at seed 42, the full emitter-aware planner produced:

```text
selected band: 17
planned sequence: [17, 16, 6]
immediate score total: 0.291937
three-step plan total: 0.858914
band belief: 0.170002
band uncertainty: 0.189410
dominant emitter probability: 0.229293
supporting emitter count: 1
top runtime contributors: E1 (p=0.229293), E0 (p=0.013412)
```

The saved trace also contains the detection, information, age, behaviour,
novelty, recency, identity, dominant-emitter, supporting-emitter, and repeat
penalty contributions. It uses Track 2 runtime state only.

## Adaptive architecture v2

The v2 production candidate fixes the two main v1 routing errors:

- INTERMEDIATE now uses the approximately 30 ms emitter-aware sequence-beam
  planner, not the full observation tree.
- COMPLEX now uses the budgeted observation tree, not PPO. PPO is disabled by
  default and requires explicit opt-in.

The belief tree now avoids constructing terminal HIT/MISS children whose future
value is exactly zero and limits deeper conditional branching to four actions.
It still evaluates the full 20-band action set at every visited belief node and
retains eight root candidates. In the held-out comparison below, standalone
tree latency fell from the earlier 302.7 ms smoke measurement to 41.7 ms.

## Earlier preliminary held-out adaptive comparison (v1)

This comparison used seed 999, which was not used for router/value training,
with 3 episodes × 200 steps. Three episodes are still too few for a final rank.

| Scheduler | Mean reward | Reward std | TP | Unique bands | Discovery rate | Planning ms |
|---|---:|---:|---:|---:|---:|---:|
| UCB | 34.100 | 16.182 | 30.33 | 20.00 | 0.864 | 0.006 |
| Track2 PPO[20] | **79.300** | 64.891 | 79.33 | 1.33 | 0.182 | 0.691 |
| MPP-60 | 58.967 | 36.354 | 60.00 | 20.00 | 0.545 | 12.420 |
| MPP-EmitterAware full | 67.367 | 44.806 | 69.33 | 20.00 | 0.545 | 31.372 |
| MPP-EmitterAware minus anomaly | 64.833 | 40.519 | 63.67 | 19.67 | 0.727 | 30.404 |
| MPP-EmitterAware minus identity/recency | 62.167 | 45.297 | 62.33 | 20.00 | 0.455 | 30.758 |
| MPP-EmitterAware adaptive exploration | 61.667 | 38.121 | 63.00 | 20.00 | 0.591 | 31.373 |
| Observation-dependent belief tree | 74.733 | 46.331 | 74.67 | 18.00 | 0.409 | 302.671 |
| Adaptive MoE rule router | 65.600 | 36.170 | 69.00 | 18.67 | 0.636 | 287.353 |
| Track2 PPO[60] | 73.000 | 57.097 | 76.00 | 2.00 | 0.227 | 0.956 |
| Attention PPO smoke | 9.300 | 0.572 | 0.33 | 4.00 | 0.045 | 1.927 |
| Learned path-value MPP | 28.933 | 11.426 | 25.00 | 20.00 | 0.727 | 32.690 |
| Adaptive MoE learned gate | 64.400 | 50.387 | 64.00 | 15.67 | 0.409 | 91.608 |

Interpretation:

- PPO[20] had the highest raw mean reward but effectively collapsed to one band,
  produced poor emitter discovery, and had extremely high variance. It is not a
  robust overall winner.
- The belief tree was the strongest broad-coverage planner, but was about 24×
  slower than MPP-60.
- The rule MoE spent 95.3% of decisions in the tree expert, so the current rule
  thresholds do not yet achieve the intended compute savings.
- The learned gate reduced planning latency by about 68% versus the rule MoE,
  but used only a tiny training set and lost some reward/coverage.
- The learned path-value and attention PPO smoke artifacts reduced performance;
  both remain optional and disabled unless their model paths are supplied.
- This v1 result is retained for provenance; it is superseded by the v2
  architecture and held-out comparison below.

## Held-out v2 comparison

This run used seeds 1099, 1199, and 1299, with 3 episodes × 200 steps per seed
(9 episodes per scheduler). None of these seeds were used for v2 router
training. Simulator truth was used only for evaluation metrics.

| Scheduler | Mean reward | Across-seed reward std | TP | Unique bands | Discovery rate | Planning ms |
|---|---:|---:|---:|---:|---:|---:|
| UCB | 35.12 | 14.13 | 31.33 | 20.00 | **0.85** | **0.01** |
| MPP-60 | 43.20 | 5.98 | 44.11 | 20.00 | 0.72 | 12.91 |
| MPP-EmitterAware minus identity/recency | 44.48 | **3.62** | 43.22 | 20.00 | 0.74 | 29.57 |
| Budgeted observation tree | 46.36 | 6.50 | 50.33 | 20.00 | 0.60 | 41.73 |
| **Adaptive MoE v2 rule router** | **52.53** | 5.30 | **52.44** | 20.00 | 0.70 | 35.43 |

On this held-out set, adaptive v2 improved mean reward by 13.3% over the
standalone tree, 18.1% over emitter-aware MPP, 21.6% over MPP-60, and 49.6%
over UCB. It routed 3.2% of decisions to fast greedy, 54.9% to sequence beam,
37.8% to the tree, and 4.1% to safe OOD exploration. No decisions used PPO.

This is a substantial preliminary gain, but nine simulated episodes are not
enough to claim general superiority or industry readiness. UCB still produced
the highest emitter discovery rate and negligible latency. Broader scenario
families, confidence intervals, hardware I/Q, and real-time limits remain open.

## Learned router v2 decision

The v2 router dataset contains 45 same-observable-state comparisons from five
training seeds, three warmup depths, and three warmup policies. Cost-aware class
counts are `[16 fast, 13 beam, 15 tree, 1 safe]`, versus v1's
`[7 fast, 1 beam, 1 RL, 0 safe]`. Mean measured expert latencies during data
collection were 0.38, 31.91, 44.13, and 1.52 ms respectively.

On the same nine held-out episodes, the learned gate achieved reward 45.01,
discovery rate 0.58, and latency 8.84 ms. It selected fast greedy 78.7% of the
time. This is a useful low-compute operating point, but it does not replace the
rule router because reward and discovery both fell. The rule router remains the
default production candidate.

## Verification

- Full test suite: 36 passed.
- Multi-seed CLI smoke test: seeds 42 and 142 completed and returned per-seed
  results, aggregate metrics, and across-seed reward standard deviation.
- The multi-seed smoke test used only one 20-step episode per seed and validates
  execution/aggregation only; it must not be used to rank schedulers.
- Focused benchmark filtering (`--only`) and JSON export (`--output`) are
  covered by the exercised CLI path used for the v2 held-out reports.

## Scenario simulator v3 routing check

The final v3 routing check used the pre-generated scenario simulator, seeds
9001 and 9011, two episodes per seed, and 100 steps per episode. Every scheduler
received the same hidden world and keyed step/band sensing randomness. Hidden
truth was used only for metrics. The exact report is
`final_condition_benchmark.json`.

| Scenario | Reward UCB | Reward observable UCB | Reward MPP-60 | Reward tree | Reward Adaptive v3 | Adaptive TP | Adaptive FP | Adaptive expert split |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Stationary | 11.45 | 13.68 | **14.55** | 12.07 | 14.38 | 12.50 | 5.25 | 66% MPP, 33% UCB |
| Hopping | **11.40** | 7.08 | 10.00 | 7.97 | 10.57 | 9.50 | 5.00 | 74% MPP, 26% UCB |
| Changing | 10.77 | **11.97** | 8.09 | 7.07 | 11.52 | 8.25 | 3.50 | 66% MPP, 32% UCB, 1% tree |
| Harsh | -3.18 | **-2.85** | -4.30 | -3.44 | -3.42 | 6.00 | 9.25 | 48% MPP, 29% UCB, 24% tree |

Adaptive v3 was the second-highest-reward scheduler in all four conditions. It
was not the universal winner: the best specialist changed with the condition.
Relative to the immediately preceding adaptive scenario build on these same
seeds, v3 improved stationary reward from 13.68 to 14.38, hopping from 9.58 to
10.57, and changing from 8.04 to 11.52; harsh changed from -3.18 to -3.42.
The harsh result therefore remains a gap, although v3 retained more true
positives than either UCB variant.

This is a routing/tuning check, not statistical proof: four episodes per
condition are too few for a superiority claim. The next legitimate gate is a
larger held-out run with confidence intervals, followed by recorded or hardware
I/Q validation.

- Final full test suite: 41 passed.

## History-selector validation

The accepted history selector was evaluated on five held-out seeds
(10001–14001), five 200-step episodes per seed, and 5,000 decisions per
condition. It uses an 80-scan rolling observation history and never reads the
scenario label or simulator reward.

| Scenario | Before history | Tuned history | Change | Planning ms | MPP / UCB / tree |
|---|---:|---:|---:|---:|---|
| Stationary | 43.844 | **45.188** | +1.344 | 34.23 | 33.9% / 14.0% / 52.0% |
| Hopping | 24.276 | **24.612** | +0.336 | 19.80 | 73.7% / 12.4% / 13.9% |
| Changing | 21.553 | **22.101** | +0.548 | 17.27 | 82.7% / 10.7% / 6.6% |
| Harsh | -3.013 | **-2.603** | +0.410 | 16.15 | 87.0% / 10.9% / 2.1% |

The first history attempt was rejected because it labelled 37% of hopping
decisions as stationary. Held-out diagnostics showed credible-hit band
concentration of 0.74–0.82 for stationary seeds, versus at most 0.67 for
hopping and 0.65 for changing. Requiring concentration of at least 0.70
reduced stationary-history activation to 11.8% in hopping and 4.8% in
changing, while retaining 52.0% in stationary. Noisy history activated on
54.3% of harsh decisions and sent them to MPP-60 instead of the tree.

The tuned selector improved all four conditions, but specialists retain some
advantages. The standalone tree remains higher-reward in stationary operation
(47.94), and MPP-60 remains better in harsh operation (-2.00). The selector is
the stronger all-condition default, not a universal specialist replacement.

- Full test suite after history-selector integration: 43 passed.

## Sequential calibration and held-out test

The benchmark now supports an unscored calibration period that does not reset
the environment or scheduler before scoring. The accepted configuration uses
80 sequential scans (four complete 20-band sweeps), followed by 200 scored
decisions. Calibration reward is reported separately and is not included in
the main reward below.

After a small development pilot, the exploration defaults were reduced to
0.5% / 1% / 3% / 0% / 5% for easy, intermediate, complex, dynamic, and unknown
states. A separate held-out run then used seeds 12001, 12011, and 12021, with
two episodes per seed and no further tuning on those results.

| Scenario | UCB | MPP-60 | Belief tree | Adaptive | Adaptive gap to best specialist | Adaptive planning ms |
|---|---:|---:|---:|---:|---:|---:|
| Stationary | 25.050 | 27.483 | 31.133 | **32.200** | +1.067 | 23.97 |
| Hopping | **23.050** | 21.417 | 18.217 | 21.783 | -1.267 | 14.29 |
| Changing | 24.202 | 23.082 | 19.197 | **26.473** | +2.271 | 14.73 |
| Harsh | -2.740 | -4.721 | **-2.156** | -4.321 | -2.165 | 15.33 |

Mean reward across the four conditions was 19.03 for Adaptive, 17.39 for UCB,
16.82 for MPP-60, and 16.60 for the belief tree. Adaptive therefore had the
best cross-condition mean and won two conditions, but it was not the universal
winner. It routed 99.8% of harsh decisions to MPP and lost to the tree there,
showing that the noisy selector can still overcommit. With only six episodes
per condition, this is a held-out engineering check rather than statistical or
real-RF proof. The exact report is `warmup_heldout_validation.json`.

- Full test suite after calibration integration: 45 passed.

## Direct Track 2 diagnosis

The held-out scheduler seeds were also evaluated with a fixed sequential scan
order so Track 2 could be measured independently of planning decisions. Each
run used 80 warm-up scans followed by 200 scored forecasts. The constant
baseline is an offline prevalence reference; the causal history predictor uses
only earlier detections on each band.

| Scenario | Constant Brier | Track 2 Brier | History Brier | Track 2 AUC | GRU transitions | GRU P(actual band) | GRU top-1 | Identity purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stationary | **0.0408** | 0.0575 | 0.0489 | 0.530 | 21 | 0.148 | 0.238 | 0.848 |
| Hopping | **0.0689** | 0.0854 | 0.1017 | 0.539 | 34 | 0.033 | 0.000 | 0.875 |
| Changing | **0.0913** | 0.1260 | 0.1142 | 0.489 | 33 | 0.026 | 0.000 | 0.833 |
| Harsh | **0.0754** | 0.0996 | 0.1102 | 0.436 | 25 | 0.019 | 0.000 | 0.744 |

Lower Brier and higher AUC are better. Track 2 was worse than the constant
reference in every scenario. Across the 92 hopping/changing/harsh GRU
transitions, its highest-probability class never matched the next observed
emitter band. Harsh sensing also produced 84 false-alarm track updates versus
39 true-detection updates, and identity purity fell to 0.744.

This identifies Track 2 generalization and false-track handling as real
limitations, but not the only limitations: the tree and Adaptive can still
differ materially while using the same Track 2 runtime, so routing and planner
approximations also contribute. Raw GRU hidden state should therefore be tested
as a frozen representation for a confidence-aware expert-return head, not
given direct control or fine-tuned on these evaluation seeds. Exact results are
in `track2_heldout_diagnostic.json`.
