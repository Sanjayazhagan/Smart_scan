# Neural-Augmented UCB rigorous validation

## Method

- Compared `Observable Discounted UCB` with `Neural-Augmented UCB`.
- Eight scenarios: stationary, hopping, bursty, crowded, changing, harsh,
  mixed, and operational.
- Ten untouched seeds per scenario, one 300-step scored episode per seed, with
  40 common warm-up scans.
- 80 paired hidden worlds per scheduler. Each pair used the same pre-generated
  emitter, interference, SNR, dropout, and retuning timeline.
- Neural-UCB used no simulator truth. Its Track 2 coefficient required twenty
  credible positive observations plus online Brier skill >= 0.90 sustained for
  twenty evaluated forecasts.
- Confidence intervals are paired non-parametric bootstrap intervals using
  20,000 resamples and a fixed analysis seed.

## Scenario results

| Scenario | Blind UCB reward | Neural-UCB reward | Delta | Blind interception | Neural interception |
|---|---:|---:|---:|---:|---:|
| Stationary | 52.0600 | 52.2800 | +0.2200 | 0.07524 | 0.07524 |
| Hopping | 37.0200 | 37.0200 | 0.0000 | 0.05713 | 0.05713 |
| Bursty | 27.6900 | 27.6900 | 0.0000 | 0.05851 | 0.05851 |
| Crowded | 40.4209 | 40.4209 | 0.0000 | 0.04548 | 0.04548 |
| Changing | 41.0197 | 41.0197 | 0.0000 | 0.05810 | 0.05810 |
| Harsh | -0.6344 | -1.6338 | -0.9994 | 0.03091 | 0.03079 |
| Mixed | 34.0475 | 34.0475 | 0.0000 | 0.05023 | 0.05023 |
| Operational | 1.1453 | 1.2904 | +0.1452 | 0.03341 | 0.03322 |

## Paired overall results

| Metric | Mean Neural minus UCB | 95% paired bootstrap CI | Wins / losses / ties |
|---|---:|---:|---:|
| Reward | -0.079280 | [-0.307144, +0.082500] | 3 / 3 / 74 |
| Interception rate | -0.000038 | [-0.000193, +0.000096] | 2 / 3 / 75 |
| Threat-weighted interception | -0.000033 | [-0.000226, +0.000142] | 2 / 3 / 75 |
| False-alarm rate | +0.000475 | [-0.000237, +0.001682] | 3 / 3 / 74 |
| Control latency | +0.006894 ms | [-0.096851, +0.107156] | 39 / 41 / 0 |

## Conclusion

The guarded implementation is working: it normally falls back to UCB because
the current Track 2 GRU does not satisfy the sustained accuracy requirement.
There is no statistically credible performance difference and no basis to
claim that the present Neural-UCB beats blind UCB.

Earlier forced/soft neural fusion did produce larger behavioral differences,
but it reduced hopping reward on fresh seeds. The next required improvement is
retraining and recalibrating Track 2's next-band GRU. The neural coefficient
should not be relaxed merely to create a visible difference.
