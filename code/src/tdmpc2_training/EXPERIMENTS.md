# TD-MPC2 Crazyflie hover: experiments

Mounir Zouhari, honours research with Prof. Colin Bellinger. This file lists
every experiment: the question it answers, exactly how it was run, where the
results are, and how it was checked. `CONTEXT.md` has the full project
background; `README.md` has the commands.

## Research question

Does a controller that learns a model of the world (TD-MPC2) cope better with
situations it never trained on than one that learns by practice alone, and
when and why? Situations: new start positions, external disturbances (wind,
extra weight, weaker motors, delays). Alongside: is TD-MPC2 fast enough to
fly a real Crazyflie, and can a cheaper version of it be accurate enough?

The key comparison inside TD-MPC2: the same trained agent flown **with**
planning (it imagines action sequences with its world model before acting)
and **without** planning (its policy network alone). Same weights, so any
difference comes from planning with the model.

## The task (all experiments)

The lab's MJX Crazyflie simulator, one drone, unchanged physics, reward and
sensor noise (`gpu_training/crazyflie_env.py`). Every 0.02 s (50 Hz) the agent
sends thrust and three body moments. A flight lasts 500 steps (10 s) and ends
early on a crash (height below 5 cm) or when the drone strays more than 2 m
beyond its starting distance. The target is always (0, 0, 1) m.

- **Batch 1:** every training flight starts at (0, 0, 0.2).
- **Batch 2:** every training flight starts at a new random place in the
  training area (`crazyflie_hover.py`), seen from above:

```
        y (m)
   +0.75 ┌───────────┬─────┐
         │           │ HELD│  held-out block: x, y in [0.25, 0.75],
         │  TRAINING │ OUT │  all heights; never used for training
         │   AREA    ├─────┘
         │     ● target (0, 0, 1)
         │           │        heights 0.2-1.5 m; starts at least
   -0.75 └───────────┘        0.3 m from the target
       -0.75       +0.75  x (m)
```

  Batch 2 also gives the agent the offset to the target **in metres**. In
  Batch 1 the offset was divided by the distance at the start of the flight,
  which is a constant for one start but would make the same position look
  different from flight to flight with random starts.

## How every agent is tested (`hover_eval.py`)

Test starts are drawn once from a fixed seed (2026), so every agent of every
experiment flies from exactly the same places:

| Group | Starts | Meaning |
| --- | --- | --- |
| `batch1` | the 6 Batch 1 starts x 3 flights | continuity with Batch 1 |
| `train_area` | 50 | same distribution as Batch 2 training |
| `held_out` | 30 | inside the held-out block: within the trained range, never seen |
| `outside` | 30 | 0.25-0.75 m beyond the edge of the training area |

- **During training:** every 25,000 steps, 21 test flights (the training start
  plus 8 / 6 / 6 starts from the three Batch 2 groups) go to
  `learning_curve.csv`, with sensor-noise seed = training seed + 1000.
- **After training:** all groups (128 flights), noise seed = training seed +
  2000, flown three ways: planning as trained (`evaluation/`), policy network
  only (`evaluation_policy_only/`), and a small 64x2 planner
  (`evaluation_plan64x2/`).
- **Metrics** (defined before any Batch 2 result): crash; late error = mean
  distance to the target over the last 2 s flown; time within 10 cm after the
  first 2 s. A crashed flight's late error is not comparable with a surviving
  one, so crashes are always reported next to it.
- Seed k of every Batch 2 experiment practises the same sequence of random
  starts (the start sampler is seeded with the training seed).

## Experiments

### Batch 1 (done, Oct 12-13; details in CONTEXT.md section 7)

| ID | Experiment | Seeds x steps | Result (training start) | Folder |
| --- | --- | --- | --- | --- |
| B | small TD-MPC2, fixed start, home PC | 11 x 50k | 0.54 +/- 0.19 m | `batch1_fixed_start` |
| C | small TD-MPC2, fixed start, Narval | 11 x 50k | 0.32 +/- 0.09 m | `batch1_small_narval` |
| D | published TD-MPC2, fixed start | 11 x 50k | 1.06 +/- 0.14 m | `batch1_standard` |
| G | published TD-MPC2, fixed start | 3 x 1M | 0.057 +/- 0.005 m, but crashed 36/36 from sideways starts | `batch1_standard_1M` |
| E | PPO (lab code), fixed start | 11 x 78.6M | median 0.028 m, 4/11 degrade | `ppo_fixed_start` |

### Batch 2 (launched Nov 1, 2026)

All on Narval, one 2/7 A100 slice (`a100_2g.10gb`) per run, account
`def-cbelling-ab`. "Published TD-MPC2" = `config_hover_standard.yaml`
(TD-MPC2 paper defaults); no hyperparameter was tuned.

| ID | Question | Config | Seeds x steps | Folder (`$SCRATCH/tdmpc2/` on Narval) |
| --- | --- | --- | --- | --- |
| R1 | Main: does TD-MPC2 learn to fly from anywhere, and does it generalise to unseen starts? | `config_batch2_random_start` | 11 x 1M | `batch2/random_start_1M/` |
| C1 | Control: identical to R1 but always the Batch 1 start. R1 vs C1 isolates the effect of random starts. | `config_batch2_fixed_start` | 3 x 1M | `batch2/fixed_start_1M/` |
| R4 | Before vs after: the Batch 1 agents (C, D, G) flown on the Batch 2 test starts, with and without planning | (each run's own) | 25 runs, no training | `evaluation_batch2_tests*/` inside each Batch 1 run |
| P | Speed: where a training step spends its time, GPU use, time per control decision on the cluster | n/a | n/a | `batch2/profiling/` |

What each answers from Prof. Bellinger's Oct 12 feedback:

| Feedback | Answered by |
| --- | --- |
| Variance between seeds: train longer? | R1 learning curves: spread of 11 seeds every 25k steps up to 1M |
| Was the Narval comparison PPO? GPU utilisation? | It was small TD-MPC2 with one simulated drone. P, plus the home measurements below |
| Random starts from an area | R1, C1 |
| Does the model help out of distribution? When and why? | R1 on held-out and outside starts, with vs without planning; disturbance tests next |
| Inference time of the big model | `bench_latency.py` (home) and P (Narval) |
| Can a smaller model be accurate enough? | R1 agents flown with the policy network only and with a 64x2 planner |
| PPO learning rate | not in this batch (PPO bugs found on Nov 1 are fixed first; see CONTEXT.md) |

## Measurements already made (home PC, Nov 1)

RTX 3060 under WSL2, Ryzen 5 5500. `outputs/latency_benchmark/`.

Time per control decision for one drone (median / 99th percentile). The drone
needs a decision every 20 ms:

| Network | Decision rule | GPU | CPU |
| --- | --- | --- | --- |
| published (4.95M parameters) | planner 512x6 | 10.7 / 12.4 ms | 589 / 630 ms |
| published | policy network only | 0.8 / 1.9 ms | 0.26 / 0.39 ms |
| small (0.39M) | planner 128x3 | 4.2 / 8.4 ms | 17.0 / 22.4 ms |

Where one training step spends its time (`profile_training.py`), one
simulated drone: simulating 0.02 s of flight about 10 ms, planning 3 ms
(small) or 14 ms (published), learning update about 7 ms. The GPU was busy
25-28% of the time: one drone gives it many tiny jobs, and it mostly waits for
Python.

## Validation before launch

All on Nov 1, 2026. Evidence is in `outputs/batch2_validation/` on the home PC.

- `check_batch2.py`: 23 automatic checks of the start sampler, the metric
  observation, the unchanged Batch 1 behaviour and the test starts. All pass.
- **Regression, testing:** two Batch 1 agents (small seed 4, published seed 2)
  tested with the code from before Batch 2 (git `00c31d3`) and with the new
  code: summaries and every step of every flight are identical (9,000 and
  4,488 trajectory rows). The new code only adds columns.
- **Regression, training:** 900-step Batch 1 training on the CPU (which is
  bit-repeatable) gives identical learning curves with the old and the new
  code, including after 400 learning updates. On the GPU, two runs of the
  *old* code already differ once learning starts (GPU arithmetic is not
  bit-repeatable), so the GPU cannot be used for this comparison.
- **End-to-end, home GPU:** 4,000-step runs with the R1 and C1 settings.
  - Starts read back from the replay buffer: R1's 79 training flights started
    all over the area (x -0.76 to 0.74, y -0.75 to 0.76, height 0.22 to 1.48 m
    including 1 cm sensor noise), none in the held-out block; C1's all at
    (0, 0, 0.2).
  - The target offset the agent sees equals target minus start within 2 mm.
  - Learning curve: 21 flights per test from every group; snapshots and
    checkpoints at the right steps; all three final tests (128 flights each)
    and snapshot testing work; existing results are refused overwriting.
  - A Batch 1 agent tested on the Batch 2 starts (the R4 path) works with its
    own (Batch 1) observation.
  - Timing on the RTX 3060: the final test of a published-size agent (128
    flights with 512x6 planning) takes about 8 min.
- Short test job on Narval before the full launch: see the launch log below.

## Reproduce

From the repository root (home):

```bash
.venv/bin/python code/src/tdmpc2_training/check_batch2.py
.venv/bin/python code/src/tdmpc2_training/train_hover.py --config-name=config_batch2_random_start seed=0
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir>             # all test groups
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --no-mpc    # policy network only
```

On Narval: `cluster/seed_job.sh`, `cluster/eval_job.sh` and
`cluster/profile_job.sh`; the exact submit commands are in their headers.
