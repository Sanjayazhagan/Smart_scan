# Smart Scan: Start Here

This package contains the source code and plain-language documentation needed
to understand, test, and run the current Smart Scan prototype. Large model
weights, datasets, virtual environments, caches, and generated benchmark JSON
are deliberately excluded.

## What the prototype does

Smart Scan chooses which of 20 RF frequency bands a narrow-band receiver should
observe at each time step. It learns from past hits and misses while trying to
find stationary, intermittent, and frequency-hopping emitters without being
given the simulator's hidden truth.

The product-facing controller is **World-Model UCB**:

1. Track 2 converts the current I/Q sample into emitter tracks, a future-band
   belief, scan ages, and uncertainty.
2. Discounted UCB is the reliable scheduler. It balances
   bands that have worked with bands that have not been checked recently.
3. Track 2 future probabilities augment that same UCB score only after online calibration
   proves that those forecasts are accurate. With the current GRU, that safety
   gate stays closed, so the neural term has zero weight.

This is a research prototype, not a field-ready electronic-warfare system and
not proof of superiority over industry systems. The current GRU requires
retraining and broader held-out validation.

## Read the code in this order

1. `README.md` - status, setup, commands, evidence, and limitations.
2. `scheduler/world_model_ucb.py` - the complete product-facing controller.
3. `scheduler/track2_runtime.py` - converts I/Q observations into scheduler
   features and manages persistent emitter tracks.
4. `scheduler/track2_core.py` - CNN/GRU world-model definitions.
5. `simulator/scenarios.py` - emitter worlds and operational stress scenario.
6. `simulator/environment.py` - receiver interaction, observations, and reward.
7. `evaluation/benchmark.py` - fair policy comparison and reported metrics.
8. `tests/` - executable examples of the expected behavior.

`MODEL_DESCRIPTIONS.md` explains every major model in more detail.
`NEURAL_UCB_RIGOROUS_RESULTS.md` records why neural guidance is currently
safety-gated. `BENCHMARK_RESULTS.md` explains the retained comparisons.

## External files and where to put them

The source runs without bundling large artifacts, but Track 2's trained state
and its synthetic RF data belong at:

```text
C:\Users\asus\Documents\SmartScanArtifacts\track2\track2_final_world_model.pt
C:\Users\asus\Documents\SmartScanArtifacts\track2\track2_synthetic_rf_dataset.npz
```

Do not put a Hugging Face token in the source code or in this package. If a
future training script downloads a gated dataset, supply the token through the
Hugging Face login or an environment variable.

## Install and test on Windows

Extract this ZIP to a normal folder, open PowerShell there, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

The last verified full suite in the development project passed 69 tests. A
different machine may take several minutes to install PyTorch and the other
dependencies before the tests themselves run.

## Run the dashboard

```powershell
.\.venv\Scripts\python.exe dashboard\server.py
```

Then open `http://localhost:8000/`. The dashboard shows only World-Model UCB.

## Run a fair core benchmark

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmark `
  --warmup-steps 80 --episode-length 200 --episodes 2 `
  --seeds 12001 12011 12021 `
  --scenarios stationary hopping changing harsh `
  --only "Observable Discounted UCB" "World-Model UCB" `
  --output current_validation.json
```

For the more realistic stress case, replace the scenario list with:

```powershell
--scenarios operational
```

Compare reward together with detection rate, false-alarm rate, threat-weighted
interception, late-emitter discovery, switching cost, and decision latency.
Never decide from one seed or only one favorable scenario.

## Current evidence in plain language

- UCB is the strong, fast baseline and the normal decision path.
- The MoE/router/tree path has been removed from the product architecture. The
  controller is now one UCB formula with optional world-model guidance.
- The current Track 2 GRU did not correctly predict the retained hopping
  transitions. Ungated neural guidance hurt reward, so the implementation now
  refuses to trust it until live forecast scores demonstrate high skill.
- The right next experiment is to retrain and calibrate the CNN/GRU on diverse
  RF data, then evaluate Neural-Augmented UCB on untouched scenarios and seeds.

## Package boundaries

Included: Python source, dashboard source, tests, dependency list, and model and
benchmark explanations.

Excluded: `.venv`, Git history, caches, trained `.pt` weights, datasets, logs,
generated JSON results, and old ZIP/archive copies.
