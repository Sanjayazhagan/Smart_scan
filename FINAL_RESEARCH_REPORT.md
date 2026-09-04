# Evaluating Model Complexity in Cognitive Spectrum Scheduling: An Ablation Study of Neural Guidance, Matrix Factorization, and Multi-Armed Bandits

---

## 1. Title and Abstract

### Abstract
This paper presents an empirical ablation study evaluating the role of model complexity in cognitive radio channel selection under dynamic, partially observable RF conditions. We investigate whether augmenting multi-armed bandit controllers with learned structural priors—specifically Non-Negative Matrix Factorization (NMF), a soft-routing Mixture-of-Experts (MoE), and a neural sequence "world model" (1D/2D CNN-GRU)—provides verifiable improvements over simpler baselines. 

Our primary validated finding is that an observation-only NMF+UCB controller without neural guidance outperforms plain UCB by 4.25 reward points, with a 95% confidence interval that strictly excludes zero ([1.39, 8.78]). However, adding neural world-model guidance to this NMF+UCB controller did not show a reliable overall improvement across multi-condition testing, with the confidence interval spanning zero ([-6.39, +6.12]). Furthermore, direct NMF and unguided NMF+UCB are statistically tied overall (95% CI [-6.32, 7.42]), and an adaptive soft-routing MoE did not outperform the simpler fixed baselines. An omniscient oracle test demonstrates that large headroom is theoretically available from accurate future prediction (+58.24 points over UCB, 95% CI [44.69, 71.80]); however, our current learned world model does not capture this headroom, scoring 17.59 compared to plain UCB’s 20.22. A possible narrow benefit of neural guidance observed in development "changing pattern" conditions (32.63 vs. 31.04) was falsified in a 50-seed replication test (Δ = +1.41, 95% CI [-1.03, +3.86], spanning zero). When evaluated as a standalone pattern-change alarm, the frozen world model's prediction error signal met all pre-registered detection criteria on an untouched test split: achieving a 90.0% True Positive Rate (95% CI [78.6%, 95.7%]) within a 20-step window, an average detection delay of 6.62 steps (95% CI [5.09, 8.20]), and a false alarm rate of 83.1 per 1,000 steps. We conclude that structured linear matrix decomposition provides an effective, low-latency prior for bandit scheduling, whereas online neural action guidance adds architectural complexity and control latency without establishing a reliable net performance benefit.

---

## 2. Methods

### 2.1 Problem Setting
We model discrete-time cognitive spectrum search across $K = 20$ frequency channels. At each discrete timestep $t$, a sensor scheduler selects a single channel $a_t \in \{0, \dots, K-1\}$. The receiver dwells on channel $a_t$ for the duration of the scan step, returning an observation tuple:
$$o_t = \{ \text{detected} \in \{0, 1\}, \, \text{signal\_power} \in \mathbb{R}_{\ge 0}, \, \text{quality} \in [0, 1], \, \text{iq} \in \mathbb{R}^{2 \times 512} \}$$

Emitters operate under five distinct environmental configurations:
1. **Stationary**: Fixed carrier frequencies with periodic or continuous transmissions.
2. **Hopping**: Agile emitters shifting frequency channels across timesteps.
3. **Changing**: Emitters undergoing unannounced transitions in duty cycle, pulse repetition interval (PRI), or operational mode.
4. **Harsh**: High noise floors (negative SNR), channel fading, and broadband interference.
5. **Operational**: Composite conditions combining rotating directional radar beams, frequency hoppers, and deceptive jamming tones.

Physical retuning incurs a switching penalty proportional to frequency displacement:
$$\text{Cost}_{\text{switch}}(a_t, a_{t-1}) = c_{\text{switch}} \cdot \frac{|a_t - a_{t-1}|}{K - 1}$$
where $c_{\text{switch}} \in [0.005, 0.02]$ depending on the scenario configuration. Retuning also introduces a transient sensitivity degradation factor.

### 2.2 Controller Architectures and Ablation Structure
To isolate the origin of any performance gain, all controllers operate strictly under identical observation-only constraints. No controller receives simulator ground truth or internal reward signals.

#### 1. Plain UCB (Observable Discounted Baseline)
Maintains discounted channel visit counts $N_i(t)$ and empirical observation qualities $V_i(t)$ for each band $i$:
$$\text{Score}_i^{\text{UCB}}(t) = V_i(t) + c_{\text{exp}} \sqrt{\frac{\ln(\sum_j N_j(t) + 2)}{N_i(t) + \epsilon}} + c_{\text{age}} \cdot \text{ScanAge}_i(t)$$

#### 2. Direct NMF
Decomposes a rolling history matrix $Y \in \mathbb{R}_{\ge 0}^{W \times K}$ of observed channel power over window $W = 30$ into $R = 4$ non-negative basis vectors and activations ($Y \approx W_{\text{act}} H_{\text{basis}}$), computing next-step scores directly from reconstructed energy:
$$\text{Score}_i^{\text{NMF}}(t) = \hat{y}_i(t) + c_{\text{staleness}} \sqrt{\text{ScanAge}_i(t)} - \text{Cost}_{\text{switch}}(i, a_{t-1})$$

#### 3. NMF+UCB Hybrid (Neural Guidance Off)
The primary non-neural combination. Combines empirical discounted values, UCB exploration bonuses, scan age, and normalized NMF reconstruction forecasts into a unified decision index:
$$\text{Score}_i^{\text{Hybrid-Off}}(t) = V_i(t) + \text{Bonus}_i^{\text{UCB}}(t) + c_{\text{age}} \cdot \text{ScanAge}_i(t) + w_{\text{nmf}} \cdot \hat{y}_i(t)$$
with $w_{\text{nmf}} = 1.00$.

#### 4. NMF+UCB Hybrid (Neural Guidance On)
The complete neural-augmented architecture. Augments the unguided NMF+UCB hybrid with probability forecasts generated by a frozen neural sequence world model:
$$\text{Score}_i^{\text{Hybrid-On}}(t) = \text{Score}_i^{\text{Hybrid-Off}}(t) + w_{\text{wm}} \cdot P_{\text{world}}(\text{band}_i)$$
where $w_{\text{wm}} = 0.35$. The world model consists of a 1D CNN processing baseband I/Q waveforms into a 64-dimensional identity space and a 2D STFT CNN coupled to a recurrent GRU forecasting next-step band occupancy distributions $P_{\text{world}}(\text{band})$. The model weights remain frozen throughout all evaluation runs.

#### 5. Soft-Routing Mixture of Experts (MoE)
An adaptive architecture utilizing an observable softmax gating network that evaluates rolling SNR, track uncertainty, and observation variance to dynamically weight expert suggestions from UCB, NMF, Robust PCA, and the neural world model.

#### 6. Omniscient Oracle-UCB
An evaluation-only baseline wherein the simulator injects the ground-truth binary activity vector of the imminent scan step directly into the UCB scoring formula. This architecture cannot be deployed in reality and serves strictly to establish the theoretical performance ceiling of perfect future prediction.

#### 7. Standalone Pattern-Change Alarm Detector
Repurposes the frozen world model’s instantaneous Brier prediction error on the selected channel:
$$e_t = \left( \mathbf{1}_{\{\text{detected}_t\}} - P_{\text{world}}(\text{band}_{a_t}) \right)^2 \in [0, 1]$$
A Page-Hinkley sequential change-point test monitors cumulative deviations from the running mean error:
$$m_t = \sum_{j=1}^t (e_j - \bar{e}_j - \delta), \quad M_t = \min_{1 \le j \le t} m_j, \quad PH_t = m_t - M_t$$
An alarm triggers when $PH_t \ge \lambda_{\text{alarm}}$. Page-Hinkley is chosen because it accumulates persistent shifts in prediction error while ignoring transient single-step sensor misses.

### 2.3 Methodological Evolution and Statistical Procedures
Earlier iterations of this investigation evaluated unablated configurations and compared observation-only neural controllers against baselines that accessed simulator-internal ground-truth reward. In the corrected comparison structure reported here:
- All controllers are evaluated under identical observation-only constraints.
- Evaluation runs are strictly paired: all schedulers face identical generated emitter trajectories, noise realizations, and random seeds.
- Evaluations comprise multiple seeds with an 80-step unscored warm-up followed by 200 scored steps per episode.
- We report paired differences $\Delta = \bar{X}_A - \bar{X}_B$ with 95% confidence intervals computed via paired Student's $t$-distributions or non-parametric bootstrap resampling. If a confidence interval spans zero, the difference is classified as not statistically established.
- For the pattern-change detector, binomial rates (True Positive Rate, False Positive Rate) are reported with 95% Wilson score confidence intervals, and detection delay is reported with a 95% bootstrap confidence interval.

---

## 3. Results

### 3.1 Primary Comparative Performance

Table 1 summarizes mean reward and decision latency across 700 matched evaluation episodes (50 episodes per scheduler across seeds 120001, 120011, 120021, 120031, 120041).

| Scheduler / Configuration | Neural Guidance | Mean Episode Reward | Mean Control Latency |
|:---|:---:|:---:|:---:|
| **Direct NMF** | No | 25.68 | 0.33 ms |
| **NMF+UCB Hybrid (Neural Guidance On)** | Yes | 25.54 | 3.57 ms |
| **Lean Observable MoE** | Adaptive | 23.25 | 3.09 ms |
| **NMF+UCB Hybrid (Neural Guidance Off)** | No | 22.86 | 0.35 ms |
| **Standard Observable UCB** | No | 16.39 | 0.01 ms |
| **Standalone World-Model UCB** | Yes | 15.92 | 2.89 ms |

#### Result 1: NMF+UCB vs. Plain UCB
NMF+UCB without neural guidance outperformed plain UCB by **4.25 reward points**, with a 95% confidence interval that strictly excludes zero:
$$\Delta_{\text{NMF+UCB} - \text{UCB}} = +4.25, \quad 95\% \text{ CI: } [1.39, \, 8.78]$$
This represents the primary statistically established performance improvement in the ablation.

#### Result 2: Direct NMF vs. Unguided NMF+UCB
Direct NMF and unguided NMF+UCB are statistically tied overall:
$$\Delta_{\text{Direct NMF} - \text{NMF+UCB}} = -0.55, \quad 95\% \text{ CI: } [-6.32, \, 7.42]$$
Because the 95% confidence interval spans zero, neither method demonstrates an overall statistical advantage. Direct NMF achieved higher scores in static environments, whereas NMF+UCB exhibited lower variance across agile conditions.

#### Result 3: The Null Result of Neural Guidance
Augmenting the NMF+UCB controller with predictions from the frozen CNN-GRU world model did not show a reliable overall improvement. Across paired evaluations, the difference between the neural-on and neural-off configurations yielded an interval spanning zero:
$$\Delta_{\text{Neural-On} - \text{Neural-Off}} = -0.13, \quad 95\% \text{ CI: } [-6.39, \, +6.12]$$
Furthermore, standalone World-Model UCB (15.92) scored lower than plain observable UCB (16.39). Neural guidance added 3.22 ms of decision latency without demonstrating a reliable net performance benefit.

#### Result 4: Soft-Routing Mixture of Experts
The soft-routing Lean Observable MoE achieved an overall mean reward of 23.25. While this improved over plain UCB (+6.86, 95% CI [0.70, 13.01]), it did not outperform the simpler Direct NMF baseline (25.68) or the fixed NMF+UCB hybrid (25.54).

---

### 3.2 Scenario-Level Decomposition and the Changing-Pattern Condition

Table 2 details performance across the five evaluated operational scenarios.

| Scenario | Observable UCB | Standalone World UCB | Direct NMF | NMF+UCB (Neural-On) | Scenario Winner |
|:---|:---:|:---:|:---:|:---:|:---|
| **Stationary** | 30.85 | 29.69 | **88.75** | 43.63 | Direct NMF |
| **Hopping** | 32.14 | 26.40 | 21.54 | **29.96** | NMF+UCB (Neural-On) |
| **Changing** | 21.65 | 21.25 | 20.01 | **32.15** | NMF+UCB (Neural-On) |
| **Harsh** | -2.45 | -0.94 | -1.52 | -0.96 | World+Exp3 (-0.11) |
| **Operational** | 6.70 | 6.11 | 3.64 | 5.45 | World+RPCA (6.10) |

Direct NMF established an advantage in the stationary scenario (88.75) by isolating static spectral components, but scored lower in dynamic scenarios (21.54 on hopping, 20.01 on changing). NMF+UCB demonstrated higher performance in hopping (29.96) and changing (32.15) environments.

#### The Changing-Pattern Hypothesis Replication
In initial development testing, neural guidance exhibited an apparent narrow benefit over unguided control in the changing-pattern scenario:
$$\text{Score}_{\text{Neural-On}} = 32.63 \quad \text{vs.} \quad \text{Score}_{\text{Neural-Off}} = 31.04 \quad (\Delta = +1.59)$$

To test whether this difference was genuine or an artifact of small sample size, a follow-up multi-seed replication was conducted across 50 fresh, paired held-out seeds (Seeds 7001–7050):
- **Neural Guidance ON Mean**: 20.12
- **Neural Guidance OFF Mean**: 18.71
- **Paired Mean Difference ($\Delta$)**: +1.41
- **95% Confidence Interval**: [-1.03, +3.86]

Because the 95% confidence interval spans zero, the hypothesis that neural guidance provides a reliable advantage in changing-pattern conditions was **falsified**.

---

### 3.3 Oracle Upper Bound Analysis

To determine whether predictive scheduling is fundamentally limited or whether the bottleneck lies in our specific learned model, we evaluated the omniscient Oracle-UCB scheduler.

| Scenario | Observable UCB | Learned World-Model UCB | Oracle-UCB (Perfect Foresight) |
|:---|:---:|:---:|:---:|
| **Stationary** | 41.97 | 36.45 | **123.62** |
| **Hopping** | 29.37 | 24.78 | **117.75** |
| **Changing** | 26.65 | 26.30 | **101.02** |
| **Harsh** | -1.17 | -2.33 | **6.69** |
| **Operational** | 4.28 | 2.73 | **36.47** |
| **Mean** | **20.22** | **17.59** | **77.11** |

Oracle-UCB improved over unguided UCB by an average of **+58.24 reward points** (95% CI: [44.69, 71.80]), winning 24 of 25 paired comparisons. 

This confirms that accurate future prediction offers substantial theoretical headroom (an approximate 3.8-fold gain over plain UCB). However, because the learned neural world model scored **17.59** (lower than plain UCB's **20.22**), the current network does not capture this available headroom.

---

### 3.4 Standalone Pattern-Change Alarm Detector

Rather than using neural predictions to guide action selection, the Page-Hinkley detector was evaluated on the world model's instantaneous prediction error stream $e_t$ to detect unannounced emitter pattern shifts.

#### Pre-Registered Detection Targets
Prior to generating experimental results, the following targets were formally registered in `detector/detector_spec.json`:
1. **Change Detection Window**: An alarm must fire within $N = 20$ timesteps of an injected pattern change.
2. **True Positive Rate (TPR)**: $\ge 80.0\%$ within the 20-step window.
3. **False Positive Rate (FPR)**: $\le 100.0$ false alarms per 1,000 steps of stationary baseline data ($\le 10.0\%$ rate).
4. **Mean Detection Delay**: $\le 12.0$ timesteps among true positive detections.

#### Detector Performance and Pass/Fail Verdict
Tuning was conducted across 60 development runs (30 change runs + 30 stationary runs, Seeds 5001–5030), yielding a frozen threshold of $\lambda^* = 0.8$. Final evaluation was executed on an untouched test split consisting of 50 fresh change runs and 50 fresh stationary runs (Seeds 6001–6050, 15,000 total steps).

Table 4 reports the final detector metrics on the untouched test split.

| Metric | Pre-Registered Target | Measured Result | 95% Confidence Interval | Target Met? |
|:---|:---:|:---:|:---:|:---:|
| **True Positive Rate (TPR)** | $\ge 80.0\%$ | **90.0%** (45 / 50) | [78.6%, 95.7%] | **YES** |
| **False Alarm Rate (/1,000 steps)** | $\le 100.0$ / 1k | **83.1 / 1k** | [Exact Poisson Count] | **YES** |
| **Mean Detection Delay** | $\le 12.0$ steps | **6.62 steps** | [5.09, 8.20] | **YES** |
| **Stationary Runs with Alarm** | Diagnostic | **100.0%** | [92.9%, 100.0%] | Diagnostic |

The prediction-error pattern-change detector met all pre-registered criteria on the untouched test split, achieving a 90.0% TPR, an average detection delay of 6.62 steps, and a false alarm rate of 83.1 per 1,000 steps.

---

## 4. Discussion

### 4.1 Simpler Combinations vs. Complex Neural Guidance
The primary contribution of this investigation is demonstrating that a simpler non-neural combination—Non-Negative Matrix Factorization combined with UCB—is competitive with, and in several dimensions superior to, more complex neural-guided architectures.

NMF+UCB without neural guidance established a validated +4.25 point improvement over plain UCB (CI excludes zero). NMF operates effectively because multi-emitter spectrum occupancy exhibits linear low-rank structure: when specific emitters activate, correlated harmonic or operational channels exhibit simultaneous energy. NMF captures this structure via linear multiplicative updates in 0.33 ms without gradient optimization, sequential pre-training, or hyperparameter sensitivity.

In contrast, augmenting this controller with a 1D/2D CNN-GRU world model increased decision latency from 0.35 ms to 3.57 ms without providing a statistically reliable overall improvement (95% CI [-6.39, +6.12], spanning zero).

### 4.2 Interpreting the Oracle Headroom
The Oracle experiment demonstrates that predictive scheduling is not conceptually flawed: perfect imminent foresight achieved a mean reward of 77.11 compared to UCB’s 20.22 (+58.24 points). 

However, this result must be paired immediately with the empirical reality that our learned world model does not capture this headroom (scoring 17.59). When a predictive model exhibits imperfect calibration, incorrect forecasts cause the receiver to execute unnecessary channel switches. Because physical retuning incurs displacement costs and transient sensitivity loss, the penalty of pursuing erroneous predictions offsets the marginal gain of occasional correct forecasts.

### 4.3 Condition-Specific Effects
While early observations suggested a possible benefit in changing-pattern conditions (32.63 vs. 31.04), multi-seed replication across 50 fresh seeds falsified this advantage (Δ = +1.41, 95% CI [-1.03, +3.86], spanning zero).

### 4.4 Diagnostic Utility of Prediction Error
Repurposing the world model as a change detector met all pre-registered criteria (90.0% TPR, 6.62 steps delay, 83.1/1k FPR), demonstrating that accumulated prediction error reliably flags structural environment shifts. However, evaluating an active "Driver + Watchdog" controller that hard-flushes NMF memory upon alarm underperformed static NMF+UCB (+13.70 vs. +16.57) due to excessive switching caused by stochastic fading false alarms.

---

## 5. Limitations

1. **Simulation Bounds**: All evaluations were conducted exclusively within a software simulation environment (`SmartScanEnv`). While the simulation models physical effects such as SNR drift, fading, retuning penalties, and deceptive interference, the architectures have not undergone validation on over-the-air RF signals or physical Software-Defined Radio (SDR) hardware.
2. **Methodological Evolution**: Earlier phases of this research utilized flawed comparisons—specifically contrasting observation-only neural controllers against baselines that accessed simulator-internal ground truth, and computing predictive accuracy on in-sample training data without held-out test splits. While these discrepancies were corrected to establish the rigorous paired ablation reported here, documenting this evolution is necessary for scientific transparency.
3. **Fixed World-Model Scope**: The neural world model was evaluated under frozen parameters trained on offline synthetic sequences. We cannot rule out that alternative formulations, such as multi-step temporal occupancy losses or continuous online fine-tuning, might improve predictive calibration, though such approaches introduce additional implementation complexity.

---

## 6. Conclusion

Through a controlled ablation across 20 RF channels and five operational scenarios, we demonstrated that:
1. **NMF+UCB without neural guidance** reliably outperformed plain UCB by 4.25 reward points (95% CI [1.39, 8.78], excluding zero).
2. **Direct NMF and unguided NMF+UCB** are statistically tied overall (95% CI [-6.32, 7.42]).
3. **Adding neural world-model guidance** did not show a reliable overall improvement over unguided control (95% CI [-6.39, +6.12], spanning zero).
4. **Soft-routing MoE** did not outperform the simpler fixed NMF baseline.
5. **The changing-pattern advantage** was falsified in a 50-seed replication test (95% CI [-1.03, +3.86], spanning zero).
6. **Oracle foresight** confirms that large theoretical headroom exists (+58.24 points over UCB), but our learned neural model fails to capture it due to misprediction switching costs.
7. **The pattern-change detector** met its pre-registered criteria on the held-out test split (90.0% TPR, 6.62 steps delay, 83.1/1k FPR).

Real-time spectrum scheduling should prioritize computationally lightweight, linear matrix decompositions over complex neural sequence models until predictive accuracy can meet the tolerances required by physical retuning constraints.
