# World-Model + NMF UCB experiment

## Architecture

The experimental scheduler keeps one UCB decision score and adds two
observation-only forecasts:

```text
score = observable value
      + UCB exploration
      + scan recency
      + 0.35 * Track 2 band belief
      + 1.00 * NMF spectrum forecast
```

NMF factorizes a rolling band-power history and does not receive simulator
truth or reward. The implementation is intentionally separate from the product
default.

## Weight selection

NMF weights 0.25, 0.50, and 1.00 were compared on development seeds 91001,
91011, and 91021. Weight 1.00 had the highest cross-scenario mean reward (23.27)
and was frozen before held-out evaluation.

## Held-out result

Held-out seeds 92001, 92011, 92021, 92031, and 92041 were evaluated with two
episodes, an 80-step warm-up, and 200 scored steps across five scenarios.

| Scenario | UCB | NMF | World UCB off | World UCB | World + NMF UCB |
|---|---:|---:|---:|---:|---:|
| Stationary | 30.85 | 67.87 | 30.79 | 29.69 | 43.63 |
| Hopping | 32.14 | 21.54 | 28.41 | 26.40 | 30.91 |
| Changing | 21.65 | 20.01 | 23.26 | 21.25 | 29.60 |
| Harsh | -2.45 | -1.52 | -0.63 | -0.94 | -0.96 |
| Operational | 6.70 | 3.64 | 5.45 | 6.11 | 11.13 |
| **Mean** | **17.78** | **22.31** | **17.45** | **16.50** | **22.86** |

The hybrid improved over World-Model UCB by 6.36 reward points across 25 paired
scenario-seed results; its approximate 95% interval was [2.71, 10.01]. It also
improved over standard UCB by 5.08 points, interval [1.39, 8.78].

The hybrid exceeded NMF alone by only 0.55 points, interval [-6.32, 7.42], so
superiority over standalone NMF is not established. NMF dominated stationary
cases, while the hybrid was substantially more balanced in changing and
operational cases. Mean hybrid control time was 3.53 ms.

## Honest conclusion

The fusion is a promising experimental all-rounder on this held-out seed set
and clearly repairs much of the current neural-only loss. It is not yet proven
better than NMF alone, and therefore has not replaced the product default.

Raw results:

- `results/world_nmf_development_tuning.json`
- `results/world_nmf_heldout_validation.json`
