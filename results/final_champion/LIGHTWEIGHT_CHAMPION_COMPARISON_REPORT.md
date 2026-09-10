# Lightweight Dual-Dwell Champion Comparison

Dataset: generic Turing-style synthetic PDW HDF5 (not official TSRD). No I/Q used.

Policies:
- **B_champion** — interruptible recurring stale-band coverage.
- **H_adaptive** — B + confidence-aware dwell hysteresis.
- **S_smart_stale** — B + urgency-ranked stale-band ordering.
- **HS_combined** — B + both modifications.

Each confirmation run used 100 held-out seeds per scenario × 6 scenarios = 600 matched episode sets, 2,400 scheduler episodes, and 360,000 decisions. Two independent seed offsets were run, for 720,000 scheduling decisions total.

## Replication 1

| Policy | Mean reward | Hit-step rate | Pulse capture | Switches | Select latency |
|---|---:|---:|---:|---:|---:|
| B champion | 113.308 | 82.39% | 35.48% | 38.28 | 0.0871 ms |
| H adaptive | 114.117 | 83.06% | 34.70% | 33.88 | 0.1029 ms |
| S smart stale | **114.145** | **83.09%** | **35.49%** | 37.30 | 0.1138 ms |
| HS combined | 114.127 | 83.07% | 34.42% | 33.90 | 0.1383 ms |

Versus B:
- H: +0.809 reward, 95% CI [+0.148, +1.469], p=0.0166, dz=0.098.
- S: **+0.837 reward**, 95% CI [+0.401, +1.273], p=0.000182, dz=0.154.
- HS: +0.819 reward, 95% CI [+0.091, +1.547], p=0.0276, dz=0.090.

## Independent replication 2

| Policy | Mean reward | Hit-step rate | Pulse capture | Switches | Select latency |
|---|---:|---:|---:|---:|---:|
| B champion | 113.047 | 82.29% | 35.38% | 38.57 | 0.0835 ms |
| H adaptive | 113.116 | 82.35% | 34.23% | 34.67 | 0.1090 ms |
| S smart stale | **113.778** | **82.90%** | **35.48%** | 37.80 | 0.1148 ms |
| HS combined | 113.359 | 82.55% | 34.11% | 34.64 | 0.1400 ms |

Versus B:
- H: +0.069 reward, 95% CI [-0.556, +0.694], p=0.828 — did not replicate.
- S: **+0.731 reward**, 95% CI [+0.253, +1.208], p=0.00278, dz=0.123 — replicated.
- HS: +0.312 reward, 95% CI [-0.402, +1.026], p=0.391 — did not replicate.

## Conclusion

**S_smart_stale is the only modification that replicated cleanly.** Across the two independent runs its reward gain was about +0.78 on average, hit-step rate improved by roughly +0.65 percentage points, pulse capture stayed effectively unchanged/slightly higher, and switching fell by roughly 0.9 switches per 150-step episode. The effect is small, but it is consistent and statistically supported in both runs.

Adaptive hysteresis is not recommended as currently implemented: its reward gain was unstable and it reduced pulse capture by roughly 0.8–1.2 percentage points. The combined policy inherited that weakness and did not outperform smart stale ordering.

### What S changes

It does **not** remove decay, recurring stale coverage, startup coverage, or interruptible dwell. It changes only the ordering of stale bands. Instead of always servicing the first stale band by array index, it ranks stale candidates using observable urgency:

- uncertainty,
- scan age,
- depth of staleness,
- small prior-usefulness term,
- switching-distance penalty.

So B's core behavior remains intact; only the next stale-band choice becomes smarter.
