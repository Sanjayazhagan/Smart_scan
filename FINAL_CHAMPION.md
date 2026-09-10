# Final Champion Candidate

## Production policy

`SmartScanProductionScheduler` in `scheduler/smartscan_production.py`

The final lightweight candidate is the previously strongest **interruptible Dual-Dwell** policy plus **smart stale-band ordering**.

### Kept unchanged

- discounted visit counts and recurring stale-band coverage
- active-signal dwell
- NMF exploitation
- uncertainty-driven exploration
- switching penalty
- fading grace
- PDW/observation-only production path

### Change 1 — interruptible stale coverage

If another band becomes stale while the currently tuned band is actively detected, stale coverage is deferred until the active dwell ends. The scheduler does not abruptly abandon the live observation.

### Change 2 — smart stale-band ordering

When several bands require stale coverage, the scheduler no longer chooses the lowest array index. It ranks stale candidates using observable urgency:

```text
0.45 * uncertainty
+ 0.25 * normalized scan age
+ 0.20 * stale depth
+ 0.10 * prior observed value
- switching distance cost
```

No future-tree or additional neural model is required.

## Validation status

The final change was tested twice on independently seeded runs of the included generic Turing-style synthetic PDW data. These fixtures are **not the official Alan Turing Synthetic Radar Dataset** and the numbers must not be presented as TSRD results.

Across both replications, smart stale ordering improved the interruptible Dual-Dwell baseline while retaining essentially the same pulse-capture rate and reducing switching slightly. See:

- `results/final_champion/LIGHTWEIGHT_CHAMPION_COMPARISON_REPORT.md`
- `results/final_champion/champion_lightweight_100seed_heldout.json`
- `results/final_champion/champion_lightweight_100seed_replication.json`

The appropriate next external validation is the same paired benchmark on genuine Turing `stare` HDF5 files.
