# Smart Scan: Complete Plain-Language Research Report

**Date:** 2026-09-17
**Purpose:** Explain what Smart Scan does, how it developed, what made Dwell-Dual strong in the main simulator benchmark, and what happened when the CNN was connected and tested on the Alan Turing synthetic radar data.

## 1. The short answer

Smart Scan is a program that chooses which one of 20 radio-frequency bands to listen to next. It does not get to see the hidden answer in advance. It only learns from the signals it has already observed.

The strongest model in the repository's main controlled simulator benchmark is **Dwell-Dual**. It performs well because it combines three simple ideas:

1. It learns which bands tend to appear together.
2. When it finds a useful signal, it stays there for a while instead of jumping away immediately.
3. When confidence is low, it explores another band, but it avoids unnecessary exploration during an active signal.

However, Dwell-Dual is not best in every test. On the separate Hugging Face Turing synthetic radar test, Direct NMF beat Dwell-Dual on the single `test_0.h5` mission. Across 20 downloaded Turing missions, Dwell-Dual won overall, but its advantage was smaller than the repository's main benchmark suggests.

The CNN noise/signal gate also did **not** improve the overall Turing result. It helped on some missions and hurt on others. The honest conclusion is:

> Dwell-Dual is the current champion for the repository's controlled benchmark and the 20-mission Turing tournament, but it is not universally superior. The CNN gate is useful as an experiment and a possible safety layer, but the current evidence does not justify making it the default decision rule.

This is a research prototype, not a field-ready electronic-warfare system and not proof that it beats real operational systems.

### Important verification correction

After checking the current source code, one label in the older benchmark documentation needs correction. The current Dwell-Dual decision path is **not a reinforcement-learning model**. It does not train a neural policy or update a learned RL network during operation. Its main decision rule is a transparent combination of:

- recent NMF pattern estimates;
- UCB-style band values and under-testing bonuses;
- uncertainty-based exploration;
- switching penalties; and
- a dwell bonus for the recently active band.

The neural Track 2 runtime can still process observations and provide perception or belief information, but the current Dwell-Dual scheduler sets direct world-model guidance to zero. Therefore, calling the champion “Hybrid NMF+RL” is misleading. The more accurate name is **NMF + UCB + uncertainty-guided dwell/explore control**.

There are also two Dwell-Dual implementations in the repository: one under `scheduler/` used by the product-style path, and one under `benchmark_models/` used by benchmark scripts. Their default tuning values are not identical. Results should always be interpreted together with the exact script, commit, and model class that produced them.

## 2. What problem is being solved?

Imagine a receiver with 20 possible frequency bands. At each moment it can listen carefully to only one of them. It must decide:

- Which band should be checked now?
- Should it stay on the current band or move?
- Is a detected signal likely to be a real emitter or noise/deception?
- How can it find signals that move between bands?

The difficult part is that the receiver does not begin with reliable information about the emitters. It must learn while operating.

### Simple explanations of the main terms

- **Band:** One of the 20 frequency areas the receiver can monitor.
- **Emitter:** A device or source transmitting radio energy.
- **Signal:** A meaningful transmission from an emitter.
- **Noise:** Unwanted energy or random disturbance that can look like a signal.
- **Jammer:** A source that deliberately makes detection harder.
- **Decoy:** A misleading signal intended to make the receiver spend attention on the wrong band.
- **Episode:** One complete simulated mission or test run.
- **Seed:** A number used to create a repeatable random test world. Many seeds are needed so that one lucky run does not decide the result.
- **Reward:** The score assigned by the test environment for good and bad decisions.
- **Hit:** Selecting a band where an active transmission is present.
- **False alarm:** Treating noise or a non-target event as meaningful.
- **Latency:** How long the software takes to make a decision.

## 3. How the project developed

The repository shows a progression from simple rules to a carefully constrained hybrid controller.

### Stage 1: Simple scanning methods

The early comparison included methods such as:

- Sequential sweep: visit band 0, then 1, then 2, and so on.
- Random scan: choose a band randomly.
- UCB: give more attention to bands that have produced useful results, while still checking less-tested bands.
- NMF: look for repeated relationships between bands.

These methods are useful baselines. A baseline is a reference method used to determine whether a more complicated method is actually helping.

### Stage 2: NMF and adaptive scanning

NMF became the mathematical foundation of the stronger policies. NMF means **Non-Negative Matrix Factorization**. In plain language, it breaks a table of recent observations into a few recurring patterns. For Smart Scan, those patterns represent bands that tend to be active together or appear in related situations.

NMF helps the receiver make an educated guess about bands it has not just observed. It does not need the hidden simulator answer; it uses the receiver's own observation history.

### Stage 3: Dwell-Dual

Dwell-Dual added a second behavior around the NMF estimate:

- **Dwell:** Stay on a promising active band instead of switching too quickly.
- **Explore:** Visit uncertain or stale bands when the current band no longer looks useful.

The model also includes a short fading grace period. That means one brief missed observation does not immediately cause it to abandon a band. This is intended to handle a signal that temporarily becomes weak.

The name Dwell-Dual refers to the two modes: exploiting a promising band and exploring for new information.

### Stage 4: CNN and GRU perception

The project then added a neural perception system called Track 2:

- A **CNN**, or Convolutional Neural Network, looks at the shape of an I/Q waveform. I/Q means two measurements describing the received radio waveform.
- A **GRU**, or Gated Recurrent Unit, remembers recent observations and tries to estimate what band may appear next.
- A prototype is a stored example representing a known signal identity.
- Cosine similarity is a comparison score between two waveform descriptions. A higher score means the shapes look more alike.

The CNN can answer a different question from the scheduler:

> Does this waveform resemble a known signal, or should it be treated as suspicious or untrusted?

The GRU tries to answer:

> Which band may be active next?

Testing showed that the GRU's next-band predictions were not reliable enough to control the scheduler directly, especially for hopping signals. Therefore, neural action guidance was disabled or safety-gated in the product path.

### Stage 5: CNN signal/noise gate

For this investigation, the CNN was connected as an optional gate. When enabled, a detected signal had to pass the identity check before it was allowed to influence the scheduler's learning state.

In practical terms, an unauthenticated detection could no longer fully:

- Increase the value of that band.
- Strengthen the NMF pattern for that band.
- Extend the dwell lock.
- Increase the uncertainty model's confidence in that observation.

This was an opt-in experiment. The normal champion remained unchanged.

## 4. What made Dwell-Dual strong in the main benchmark?

The main repository benchmark used 50 random seeds, six scenario types, 14 models, 4,200 episodes, and 630,000 time steps. Its reported Dwell-Dual result was:

- Mean reward: **+18.17**
- 95% confidence interval: **+16.60 to +19.74**
- Intercept rate: **18.0%**
- Average decision time: **0.371 ms**

A confidence interval is a range showing the uncertainty around an average. It is not a guarantee that every future run will fall inside the range.

In that benchmark Dwell-Dual was ahead of:

- Dual-Policy Uncertainty: **+16.57**
- Static NMF+UCB: **+16.54**
- Direct NMF: **+16.25**
- Fixed sequential sweep: **+9.84**
- Uniform random scan: **+9.40**

The likely reasons for this result are concrete rather than mysterious:

1. **It uses history.** Recent observations are turned into recurring band patterns.
2. **It avoids needless switching.** Switching bands can cost time and may interrupt a useful observation.
3. **It protects active detections.** The dwell rule gives an active signal priority.
4. **It still explores.** It does not remain stuck forever on one band.
5. **It is lightweight.** The scheduling calculation is mostly small numerical operations rather than a large neural decision system.
6. **It uses only visible observations.** The benchmark policy does not read the hidden simulator truth to choose actions.

These are the reasons it performed well in that test environment. They are not proof that the same ranking must hold on real radio data.

One further precision matters: the scheduler's exploration rule avoids entering its explicit uncertainty-seeking mode while the current signal is active. It still calculates a normal exploit score, and that score can select another band if another band's value, NMF estimate, switching cost, or other terms win. So “protects active detections” is a design intention and a strong tendency, not an absolute mathematical guarantee that the receiver can never switch during a detection.

## 5. Where Dwell-Dual was not best

A fair report must include the negative results.

### Main simulator limitation

Dwell-Dual had the best average reward in the main simulator benchmark, but its advantage over the closest models was modest in statistical terms. The reported effect sizes were small:

- Versus Dual-Policy Uncertainty: Cohen's d about **0.16**.
- Versus Static NMF+UCB: about **0.15**.
- Versus Direct NMF: about **0.12**.

Cohen's d is a standardized way to describe how far apart two averages are. Values near 0 mean the distributions overlap heavily. Therefore, Dwell-Dual's win is real in the reported paired test, but it is not a huge separation from the strongest alternatives.

### Single-mission Turing test

On the external Hugging Face Turing dataset's `test_0.h5` mission, the measured result was:

| Model | Reward | Hits | Hit rate |
|---|---:|---:|---:|
| Direct NMF | +59.88 | 19 | 4.27% |
| Sequential sweep | +58.78 | 19 | 4.27% |
| Dwell-Dual | +58.58 | 20 | 4.49% |
| Random scan | +51.93 | 22 | 4.94% |

On this mission, **Direct NMF beat Dwell-Dual on reward**. Dwell-Dual found one more hit than Direct NMF, but the total reward also includes the cost of misses, non-events, and switching. This is an important distinction: the model with the most hits is not automatically the model with the highest reward.

### Twenty-mission Turing tournament

The downloaded external test set contained 20 missions: archive, scan, and stare modes.

| Rank | Model | Mean reward | Total hits | Average switches |
|---:|---|---:|---:|---:|
| 1 | Dwell-Dual | +112.19 | 1,920 | 196.3 |
| 2 | Direct NMF | +71.36 | 1,044 | 299.0 |
| 3 | Sequential sweep | +71.32 | 1,043 | 299.0 |
| 4 | Random scan | +65.06 | 1,044 | 283.0 |

Dwell-Dual won this combined tournament, but this test has several limitations:

- It uses a custom conversion from pulse descriptor data into 20 bands.
- It evaluates only the first part of each mission's time span.
- It uses a simplified reward definition.
- It is not a direct comparison with raw captured I/Q signals.
- The missions are not necessarily independent in the same way as separate real-world engagements.

The Turing result supports Dwell-Dual as a promising method, but it should not be described as final proof of general superiority.

## 6. What happened when the CNN gate was added?

The CNN gate was trained from the Alan Turing synthetic radar data. Since that dataset contains pulse descriptors rather than native I/Q recordings, the project created approximate 512-sample I/Q waveforms from the descriptors. This is called **synthetic waveform generation**: making a model input from summary measurements rather than recording the waveform directly.

That makes the experiment useful, but weaker than a test using real captured I/Q data.

### Single-mission result with the CNN gate

| Model | Reward | Hits | Hit rate |
|---|---:|---:|---:|
| Direct NMF | +59.88 | 19 | 4.27% |
| Dwell-Dual | +58.58 | 20 | 4.49% |
| Dwell-Dual + CNN gate | +54.08 | 14 | 3.15% |
| Random scan | +51.93 | 22 | 4.94% |

On this mission the CNN gate was worse than the ungated champion.

### Twenty-mission result with the CNN gate

| Rank | Model | Mean reward | Total hits | Average switches |
|---:|---|---:|---:|---:|
| 1 | Dwell-Dual | +112.19 | 1,920 | 196.3 |
| 2 | Dwell-Dual + CNN gate | +109.69 | 1,866 | 203.6 |
| 3 | Direct NMF | +71.36 | 1,044 | 299.0 |
| 4 | Sequential sweep | +71.32 | 1,043 | 299.0 |
| 5 | Random scan | +65.06 | 1,044 | 283.0 |

The CNN gate improved some individual missions, including archive tests 4, 7, 8, and 9, but reduced performance on other missions. Overall it:

- Reduced mean reward by **2.50 points**.
- Reduced hits by **54**.
- Increased average switching from **196.3** to **203.6**.

The correct conclusion is not that CNNs are bad. The correct conclusion is that **this particular CNN gate, trained and tested in this particular way, was not reliable enough to replace the ungated champion**.

## 7. Why the CNN gate lost overall

The gate is making a stricter decision: a signal must look sufficiently similar to a stored prototype before it is trusted. That can reject real signals when:

- The signal shape changes naturally.
- The signal is weak.
- The waveform conversion does not match the training conversion.
- The emitter is not represented well by the stored prototypes.
- A new emitter is real but unfamiliar.

This creates a tradeoff:

- A stricter gate may reject more noise and deception.
- The same strict gate may also reject real but unfamiliar signals.

The current results show that the second problem is large enough to reduce overall reward. Therefore, the gate should remain optional until it is retrained and tested on broader, untouched waveform data.

## 8. Neural guidance: why it is not controlling decisions

The project also tested using the GRU's predicted next band to guide scheduling. A forecast is an estimate of what may happen next.

The guarded validation compared ordinary UCB with Neural-UCB across eight scenarios and 80 paired episodes. The overall result was:

- Reward difference, Neural-UCB minus UCB: **-0.0793**.
- 95% interval: **-0.3071 to +0.0825**.
- Wins/losses/ties: **3 / 3 / 74**.

Because the interval crosses zero and the result is very close to zero, there is no credible evidence that the neural guidance improves performance. In the harsh scenario it was worse by about **1.0 reward point**.

This is why the product path keeps neural action guidance closed unless online calibration proves that the forecasts are genuinely accurate.

## 9. Noise and deception results

The existing perception tests report strong results against a particular family of synthetic decoys:

- 15 of 15 tested decoy variants were rejected.
- The baseline decoy had similarity 0.5190, below the canonical threshold 0.7415.
- A naive energy-only method was trapped on the decoy for much of the attack window.

Those results are encouraging, but they must be interpreted carefully:

- The decoys are synthetic.
- The test family was designed within the project.
- Passing these tests does not prove protection against every real jammer or deception technique.
- The repository also reports that false alarms rise from 4.57% in pure noise to 28.05% under active jamming.

That last result is a major limitation, not a footnote. The system still has difficulty distinguishing some active jamming from useful signals.

## 10. Speed and practical operation

The main reported benchmark measured:

- Dwell-Dual scheduler decision: mean **0.371 ms**.
- CNN perception pipeline: mean **1.305 ms**, median **1.202 ms**, 95th-percentile **1.805 ms**.
- Combined serialized 95th-percentile estimate: about **2.48 ms**.

A millisecond is one thousandth of a second. The 95th percentile means 95 out of 100 measurements were at or below that value; the remaining 5% took longer.

The median combined path appears compatible with a 2 ms target, but the reported tail estimate exceeds 2 ms. Therefore, the honest statement is:

> Typical operation is fast enough for the stated target, but worst-case or high-percentile behavior needs more engineering and should not be described as unconditionally compliant.

The project suggests pipelining, meaning processing the previous waveform while preparing the next scheduling decision. That idea still needs hardware-level verification.

## 11. What is currently trusted

The most defensible current architecture is:

1. Use the observable receiver history.
2. Use NMF to identify recurring frequency patterns.
3. Use discounted UCB to balance known useful bands and under-checked bands.
4. Use Dwell-Dual's stay/explore behavior, while treating its active-band preference as a bias rather than an absolute lock.
5. Keep GRU next-band guidance disabled unless it passes a strict held-out calibration test.
6. Keep the CNN gate optional rather than assuming it improves performance.

This design is conservative because the experiments show that simpler, transparent behavior is more reliable than forcing neural predictions into the control loop.

## 12. What should happen next

The next experiments should focus on evidence quality rather than adding complexity:

1. Test with real or genuinely recorded I/Q waveforms instead of only synthetic waveforms made from pulse descriptors.
2. Train the CNN with many emitter types, unfamiliar emitters, weak signals, noise, and multiple jammer styles.
3. Keep a completely untouched test set that is not used for training or threshold selection.
4. Report reward, hits, false alarms, missed signals, switching, and latency together.
5. Run enough independent missions to produce confidence intervals for the CNN gate.
6. Test whether a soft confidence score is better than an all-or-nothing rejection gate.
7. Test the CNN as a warning signal or secondary feature before allowing it to block scheduler learning.
8. Revoke the exposed Hugging Face token and use local login or environment-based authentication only.

## 13. Final unbiased conclusion

Dwell-Dual is a strong and well-motivated research prototype. It clearly beats simple scanning methods in the repository's main benchmark and wins the combined 20-mission Turing tournament. Its advantage comes from sensible behavior: learn recurring patterns, stay on useful signals, and explore when uncertain.

But Dwell-Dual is not unbeatable. Direct NMF won the single Turing `test_0.h5` mission. The CNN gate reduced the overall Turing score. Neural next-band guidance has not shown a reliable improvement. False alarms become much worse under active jamming, and the combined high-percentile timing estimate needs more work.

The most accurate claim is therefore:

> Dwell-Dual is the current best-performing controller in the tested Smart Scan suite, not a universally proven winner. The current evidence supports using it as the default research baseline while continuing to test simpler alternatives and treating CNN-based filtering as experimental.

## 14. New Turing-calibrated experiment

After this report was first written, a controlled behavior study was run using the downloaded Hugging Face Turing missions. The study tested changes to dwell strength, exploration probability, uncertainty threshold, switching penalty, NMF memory length, NMF update frequency, and fading grace.

Before comparing variants, the Turing adapter was corrected in two ways:

- Turing pulse power is dBm-like and can be negative, while NMF requires non-negative input. The adapter now converts it to a non-negative signal-strength value.
- NMF now protects its update calculations from invalid numeric values.

The winning change was simple: increase `dwell_inertia` from **1.50** to **2.40**. In plain language, the receiver stays with a useful detected band more strongly before moving away.

The variant is named **Turing-Calibrated Dwell-Dual** and is implemented in [benchmark_models/turing_calibrated_dwell_dual.py](benchmark_models/turing_calibrated_dwell_dual.py). It changes no architecture and adds no neural decision rule.

### Corrected 20-mission tournament

| Rank | Model | Mean reward | Total hits | Average switches |
|---:|---|---:|---:|---:|
| 1 | Turing-Calibrated Dwell-Dual | **+125.02** | **2,193** | **171.4** |
| 2 | Original Dwell-Dual | +119.20 | 2,061 | 182.3 |
| 3 | Original Dwell-Dual + CNN gate | +115.91 | 1,996 | 191.4 |
| 4 | Direct NMF | +77.90 | 1,202 | 256.9 |
| 5 | Sequential sweep | +71.32 | 1,043 | 299.0 |
| 6 | Random scan | +65.06 | 1,044 | 283.0 |

The calibrated variant therefore improved average reward by **+5.82** over the original Dwell-Dual in this corrected 20-mission tournament, increased total hits by **132**, and reduced average switching by **10.9**.

A separate corrected paired validation used 20 missions and 10 seeds, producing **200 paired runs**:

- Original Dwell-Dual: mean reward **+117.42**.
- Turing-Calibrated Dwell-Dual: mean reward **+122.23**.
- Mean paired improvement: **+4.81**.
- Wins/losses/ties: **101 / 34 / 65**.

This is a meaningful and promising result, but it is not proof of universal superiority. The calibrated variant lost on one mission, `archive_test/test_7.h5`, and the Turing missions are synthetic. The improvement should currently be described as:

> A validated improvement for this downloaded Turing synthetic radar benchmark, not yet a universal replacement for the original Dwell-Dual policy.

### Why “Dwell-Dual” is the name

The name describes two competing jobs:

- **Dwell:** stay on a band when recent evidence says the signal is useful, reducing unnecessary retuning and preserving observation time.
- **Dual:** maintain a second behavior that explores other bands when the current band becomes weak, stale, or uncertain.

The Turing-calibrated version keeps this same two-part design. It does not become a different kind of model; it simply gives the dwell side more weight because the Turing missions rewarded maintaining a useful band longer.

### What is and is not promoted

The original Dwell-Dual remains available as the legacy benchmark baseline. The Turing-Calibrated `2.40` setting is now promoted in the production-style scheduler and dashboard, but it remains a research configuration. It still needs validation on the original simulator suite and independent RF data that was not used to select the value before any field deployment claim is made.
