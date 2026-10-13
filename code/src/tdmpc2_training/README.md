# One-drone TD-MPC2 hover baseline

This experiment uses the existing MJX Crazyflie physics, 31-number observation,
four controls, and reward. It starts at (0, 0, 0.2) metres, targets (0, 0, 1)
metres, and ends on the existing crash/out-of-bounds rule or after 500 steps.
The Gymnasium adapter exposes the physical thrust range [0, 0.35]; the
vendored TD-MPC2 trainer uses Gymnasium's RescaleAction so its agent sees
normalized actions in [-1, 1]. This adapter transfers each step between the
Python Gymnasium loop and JAX/MJX, so it does not preserve fully vectorized
MJX throughput.

The `tdmpc2_jax/` package is copied from
https://github.com/ShaneFlandermeyer/tdmpc2-jax at commit
`5b05ff452424896d709848e1f249bd67e269b8a1`, under its MIT license
(`UPSTREAM_LICENSE`). Local changes: remove TensorFlow/TFP imports, use JAX
for Normal log probabilities, use tensorboardX, load DM Control lazily, and
make initial data collection/update counts configurable for a small run.

## Ubuntu/WSL setup

From the repository root, with your existing Python 3.11 environment:

```bash
~/.local/bin/uv pip install --python .venv/bin/python -r code/src/tdmpc2_training/requirements.txt
```

This can add packages to the existing GPU environment. Do not install into
the Windows `.venv`; the MJX GPU dependencies are for Linux.

## Test the environment without training

```bash
.venv/bin/python code/src/tdmpc2_training/check_hover.py
```

This prints observation/action dimensions, converted thrust, reward and height.
It does not create a trained model.

## Training

Smoke test (collects random experience, checks imports and replay buffer):

```bash
.venv/bin/python code/src/tdmpc2_training/train_hover.py max_steps=10 seed_steps=500
```

First learning test (includes planner and model updates):

```bash
.venv/bin/python code/src/tdmpc2_training/train_hover.py max_steps=700
```

Full initial experiment:

```bash
.venv/bin/python code/src/tdmpc2_training/train_hover.py
```

The default config is `tdmpc2_jax/config_hover.yaml`: 20,000 steps, one
environment, 500 random seed steps, 100 initial updates. The model is an
experimental baseline; completing training does not demonstrate hovering.
Hydra creates a timestamped `outputs/crazyflie_hover_s0/...` folder with a
TensorBoard log, the checkpoint, and `learning_curve.csv`: every 5,000 steps
the agent flies three test flights from the training start and the results
are appended there.

Note: these settings are smaller than the published TD-MPC2 defaults (latent
size 128 instead of 512, 128 planning samples x 3 iterations instead of
512 x 6, batch size 64 instead of 256).

## Testing a saved agent

```bash
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run folder>
```

Rebuilds the agent from the run's config, loads its newest checkpoint, and
flies three flights from each start in `hover_eval.TEST_STARTS` (the training
start, 0.5 m to each side, and above the target). Nothing is trained. Writes
`evaluation/summary.csv` and `evaluation/trajectories.csv` in the run folder.

## Several seeds

```bash
.venv/bin/python code/src/tdmpc2_training/run_seeds.py --seeds 0-10 --parallel 2
```

Trains and tests every seed with identical settings into
`outputs/batch1_fixed_start/seed_XX/`, then runs `analyze_seeds.py`, which
writes plots (mean +/- standard error, and every seed with the median seed
highlighted) and `analysis/summary.md`. Finished seeds are skipped, so the
same command resumes an interrupted batch. On the RTX 3060, one seed trains
at about 44 steps/s alone, two at once at about 34 steps/s each.

| File | Purpose |
| --- | --- |
| `crazyflie_hover.py` | Gymnasium adapter; `place_drone` moves the start/target |
| `hover_eval.py` | Test flights (all flights batched on the GPU) and metric definitions |
| `eval_hover.py` | Load a checkpoint and test it |
| `run_seeds.py` | Train + test a list of seeds |
| `analyze_seeds.py` | Plots and summary for a batch |
| `tdmpc2_jax/build.py` | Builds the agent; shared by training and testing |
