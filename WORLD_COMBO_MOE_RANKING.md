# World-Model Fusion and MoE Held-Out Ranking

## Test protocol

- Fresh seeds: `120001`, `120011`, `120021`, `120031`, `120041`
- Scenarios: stationary, hopping, changing, harsh, operational
- Two episodes per seed and scenario
- 80 unscored warm-up steps followed by 200 scored steps
- 50 scored episodes per scheduler; 700 total scheduler episodes
- Every scheduler saw the same paired hidden worlds
- The MoE and fusion controllers use observations only; no simulator truth reward is passed into their decision logic

Raw results: `results/world_combo_moe_heldout.json`

## Overall raw-reward ranking

| Rank | Scheduler | Mean reward | Mean control time |
|---:|---|---:|---:|
| 1 | Non-Negative Matrix Factorization | 25.675 | 0.326 ms |
| 2 | World+NMF UCB (weight 1.00) | 25.541 | 3.574 ms |
| 3 | Lean Observable MoE | 23.247 | 3.092 ms |
| 4 | World+NMF UCB (weight 0.50) | 20.452 | 3.689 ms |
| 5 | World+RPCA UCB (weight 1.00) | 19.643 | 4.442 ms |
| 6 | World+NMF UCB (weight 0.25) | 18.472 | 3.521 ms |
| 7 | World+RPCA UCB (weight 0.50) | 18.403 | 4.588 ms |
| 8 | UCB | 16.388 | 0.009 ms |
| 9 | World-Model UCB, neural guidance off | 16.280 | 2.960 ms |
| 10 | World+Exp3 UCB (weight 0.50) | 16.102 | 2.962 ms |
| 11 | World-Model UCB | 15.922 | 2.885 ms |
| 12 | World+Exp3 UCB (weight 1.00) | 15.838 | 2.971 ms |
| 13 | World+PRI UCB (weight 0.50) | 15.480 | 2.947 ms |
| 14 | World+PRI UCB (weight 1.00) | 15.059 | 3.051 ms |

## Scenario winners

| Scenario | Winner | Reward | Best MoE/fusion result |
|---|---|---:|---|
| Stationary | Direct NMF | 88.750 | MoE: 69.630 (rank 2) |
| Hopping | World+NMF UCB (1.00) | 29.960 | same |
| Changing | World+NMF UCB (1.00) | 32.150 | same |
| Harsh | World+Exp3 UCB (1.00) | -0.114 | same |
| Operational | World+RPCA UCB (0.50) | 6.102 | same |

## Paired statistical checks

Each comparison uses the 25 paired seed/scenario cells; each cell averages its two episodes.

- World+NMF (1.00) versus direct NMF: -0.134 mean reward, 95% CI [-6.388, 6.120], 15/25 cell wins. They are statistically tied in this test.
- MoE versus UCB: +6.859, 95% CI [0.703, 13.014], 16/25 wins.
- MoE versus standalone World-Model UCB: +7.325, 95% CI [1.360, 13.290], 16/25 wins.
- Direct NMF versus MoE: +2.428, 95% CI [-2.379, 7.236], 14/25 wins. The apparent raw-score lead is not statistically decisive.

## Interpretation

The most reliable fixed fusion is World+NMF UCB at weight 1.00. It gives up stationary reward relative to direct NMF but wins hopping and changing conditions and scans more broadly. Direct NMF's overall lead is mostly created by its very large stationary score; it is not the best general controller in every environment.

The MoE successfully learned enough observable structure to route direct NMF frequently in stationary episodes and reached second place there. It also significantly beat plain UCB and standalone World-Model UCB across the paired matrix. It did not beat the simpler NMF approaches, so it should remain experimental rather than becoming the default champion.

The largest routing gap is harsh noise. Development seeds favored disabling neural guidance, but fresh held-out seeds favored the observable Exp3 fusion. This means the present harsh/noisy classifier and expert choice are not stable enough. Changing the branch after seeing this held-out set would contaminate the test; it should be tuned on a new development split and confirmed once on another untouched split.

## Defensible current choice

- Best simple all-round candidate: **World+NMF UCB (weight 1.00)**.
- Fastest high-scoring specialist: **direct NMF**, especially for confirmed stationary emitters.
- Best experimental adaptive architecture: **Lean Observable MoE**, but it needs a better harsh/noisy gate.
- Current world model alone is not competitive; its value appears when combined with NMF, not as the sole guide.
