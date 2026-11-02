% Learning to hover from anywhere
% TD-MPC2 on a simulated Crazyflie: a complete guide to Batch 2, its results and its code
% Mounir Zouhari, honours research, University of Ottawa
% Supervisor: Prof. Colin Bellinger
% November 2, 2026

[[TOC]]

# 1. How to read this guide

This guide explains everything that was done in the project up to November 2, 2026: what each experiment asked, why it was designed the way it was, what came out, what the result means, and what can be done next. It also describes every file of code, so that any question about how the project is organised can be answered from it. It assumes no background in reinforcement learning beyond what is explained in Chapter 2.

The chapters build on each other. Chapter 2 explains the ideas: reinforcement learning, the simulated drone, PPO, TD-MPC2, planning, and generalisation. Chapter 3 summarises where the project stood after the first batch of experiments and what Professor Bellinger asked for. Chapter 4 explains how the second batch was designed and how the code was checked before anything ran. Chapter 5 goes through every experiment and its results. Chapter 6 answers the professor's questions one by one. Chapter 7 is a tour of the code, file by file, with the commands to reproduce each result. Chapter 8 lists the limitations, and Chapter 9 the next steps. Appendix A is a glossary of every technical term used, and Appendix B lists the cluster job numbers and code versions.

A few conventions are used throughout. A **flight** is one test of a trained drone controller, lasting 10 simulated seconds. A flight is a **success** when the drone flies the full 10 seconds and its average distance to the target over the last 2 seconds (the **late error**) is below 10 cm. A flight is a **crash** when it ends early, either because the drone touched the ground or because it flew too far away. Most results are given as a **mean ± standard error** across several independently trained controllers (**seeds**); the standard error says how precisely the mean is known. Where a difference between two methods matters, a **95% confidence interval** is given in square brackets: if the interval does not contain zero, the difference is very unlikely to be due to chance alone.

# 2. The ideas behind the project

## 2.1 Reinforcement learning in one page

Reinforcement learning (RL) is a way of teaching a computer program to make decisions by trial and error. The program, called the **agent**, interacts with a world, called the **environment**, one small time step at a time. At every step the agent receives an **observation** (here: 31 numbers describing where the drone is, how it moves and how it is tilted), chooses an **action** (here: 4 numbers, the commands sent to the drone), and receives a **reward**, a single number saying how good the result of that action was. Nobody tells the agent which action is correct. It has to discover, from the rewards, which behaviour leads to a high total reward over a whole flight.

The rule the agent uses to choose actions is called its **policy**. In modern RL the policy is a neural network: a function with many adjustable numbers (its **parameters** or **weights**) that maps an observation to an action. Learning means adjusting these weights so that the total reward grows. Many methods also learn a **value function**, which estimates how much reward can still be collected from a given situation; it lets the agent judge the long-term consequences of an action without waiting for the end of the flight.

Learning needs experience. Each simulated step the agent takes is one **environment step** (here 0.02 seconds of simulated flight). How many environment steps a method needs to learn a task is its **sample efficiency**; how many seconds of computer time it needs is its **wall-clock** cost. Both matter: sample efficiency matters when experience is expensive (a real drone), wall-clock time matters when compute is the limit.

Training involves randomness: the starting weights, the exploration, the order in which experience is reused. The **seed** of a run fixes all of this randomness. Two runs with the same settings but different seeds can end up quite different, so a single run proves little. That is why the professor asked for 11 seeds per experiment, and why results are reported as averages with uncertainty.

## 2.2 The simulated Crazyflie

The Crazyflie 2 is a 27-gram quadrotor. The lab's simulator models it with **MuJoCo**, a physics engine, running on the graphics card through **MJX** (MuJoCo written in JAX, a library for fast numerical computing on GPUs). The physics advances in steps of 2 ms; the agent makes a decision every 20 ms, so the drone is controlled **50 times per second (50 Hz)**.

The **action** has four numbers. The first is the collective **thrust**, between 0 and 0.35 newtons. Hovering needs 0.265 N, which is 76% of the maximum, so there is little spare thrust. The other three are **body moments**, twisting forces around the drone's three axes, each between -1 and 1 (scaled to very small torques). The agent's networks work with all four numbers scaled to [-1, 1]; they are converted to these physical units before reaching the simulator.

The **observation** has 31 numbers: the position (3), the velocity (3), the angular velocity (3), the orientation as a 3×3 rotation matrix (9), the offset to the target expressed in the drone's own axes (3), the velocity in the drone's own axes (3), the offset to other drones (3, always zero with one drone), and the previous action (4). Small random noise is added to the measurements, as real sensors would (about 1 cm on the position).

The **reward** of every step is the sum of a tiny bonus for staying in the air (0.001), a bonus for being close to the target (up to 0.01, shrinking with distance), and the reduction of the distance to the target since the previous step. A crash costs -1. A flight ends early (a **crash**) when the drone drops below 5 cm or strays more than 2 m further from the target than where it started; otherwise it lasts 500 steps (10 s). The target is always the point (0, 0, 1), one metre above the origin.

## 2.3 PPO, the model-free method of the previous students

**PPO** (Proximal Policy Optimization) is a widely used **model-free** method: it learns the policy directly from experience, without learning how the world works. It collects a large batch of experience with the current policy, then improves the policy with a few gradient steps that are kept small on purpose (the "proximal" part), and repeats. The lab's PPO simulates **2,048 drones at once** on the GPU, so it collects experience extremely fast: tens of millions of environment steps in minutes. It needs that much, because model-free learning is sample-hungry.

## 2.4 TD-MPC2, the model-based method studied here

**TD-MPC2** (Hansen, Su and Wang, ICLR 2024) is a **model-based** method: besides a policy, it learns a **world model**, a set of networks that predict what will happen. It has five parts:

- an **encoder** that compresses the 31 observation numbers into an internal description, the **latent state** (128 or 512 numbers depending on the size);
- a **dynamics** network that predicts the next latent state from the current one and an action;
- a **reward** network that predicts the reward of an action in a latent state;
- a **value** network (an ensemble of five) that estimates the long-term reward of an action in a latent state;
- a **policy** network that proposes an action for a latent state.

All five are trained together from a **replay buffer**, a memory of all past experience, with one learning update for every environment step (an **updates-to-data ratio** of 1).

The distinctive part is how TD-MPC2 acts. Before every single decision it **plans**: it imagines many possible action sequences a few steps into the future with its world model, scores each one by the predicted rewards plus the value at the end, and executes only the first action of the best one. At the next step it plans again from the new situation. This is called **model predictive control** (MPC), and the planning algorithm is **MPPI**: it samples candidate plans (some proposed by the policy network, most drawn randomly around its current best guess), keeps the best ones (the **elites**), refits its sampling distribution to them, and repeats for several rounds.

Two sizes were used. The **published** settings (from the TD-MPC2 paper) have a latent state of 512 numbers, about 5 million learned weights, and plan with **512 candidate plans for 6 rounds**. The **small** settings, used in September, have 128 latent numbers, about 0.4 million weights, and plan with **128 plans for 3 rounds**. Both plan **3 steps ahead**, which is 0.06 seconds; everything after that is judged by the value network.

## 2.5 Planning versus the policy network alone

Because TD-MPC2 trains a policy network as part of its learning, a trained agent can be flown in two ways from exactly the same weights: **with planning** (the normal TD-MPC2) or with **its policy network only** (take the policy's proposed action, no imagination). Comparing the two isolates what planning with the world model contributes. This comparison is used throughout the project, because it answers the central question directly: does using the learned world model at decision time help or hurt?

## 2.6 Generalisation and distribution shift

A controller **generalises** when it works in situations it did not see during training. The set of situations seen in training is the **training distribution**. Tests can be inside it (new random samples of the same kind), in a deliberately **held-out** region inside its range that training never visited (**interpolation**), or **outside** its range (**extrapolation**, also called **out-of-distribution**). **Disturbances**, such as wind or extra weight, change the physics itself and are another kind of shift. Keeping these cases apart matters, because a method can be good at one and bad at another.

# 3. Where the project stood, and what was asked

## 3.1 Batch 1 (October 12-13)

In the first batch, every controller was trained from one fixed start, (0, 0, 0.2), to hover at (0, 0, 1). The main results were:

| Experiment | Seeds × steps | Result from the training start | From 5 other starts |
|---|---|---|---|
| small TD-MPC2, home PC | 11 × 50k | 0.54 ± 0.19 m | some crashes |
| small TD-MPC2, Narval | 11 × 50k | 0.32 ± 0.09 m (seeds from 6.5 cm to 1.05 m) | few crashes |
| published TD-MPC2 | 11 × 50k | 1.06 ± 0.14 m, mostly crashing | mostly crashing |
| published TD-MPC2 | 3 × 1M | 5.7 ± 0.5 cm | crashed 36 of 36 flights |
| PPO (lab code) | 11 × 78.6M | median 2.8 cm, 4 of 11 runs degraded | mostly crashing |

TD-MPC2 reached 10 cm with about 100 times fewer environment steps than PPO, but PPO was faster in real time. Both methods memorised the single route they were trained on.

## 3.2 Professor Bellinger's feedback

His reply raised the following points, which shaped everything in Batch 2:

- The large differences between seeds are unusual for TD-MPC2; the runs were probably too short.
- Was the cluster comparison done with PPO? How many parallel environments were used, and was the GPU utilisation monitored?
- Three starting positions are not enough; starting positions should be drawn at random from a region.
- An interesting question: does the world model help with situations outside the training distribution, such as new starts or external disturbances, and when and why is it most useful?
- How long does the big model take to decide, and what does that imply for a real drone? Is a smaller model enough, and how can it be made accurate?
- For PPO's late degradation, a lower or decaying learning rate might help.

These were turned into ten research questions (Chapter 6).

## 3.3 What the code audit found (November 1)

Before designing Batch 2, the code was audited. Five findings changed how earlier results must be read:

- **PPO's learning rate never decreased.** The schedule was written in environment steps (60 million) but the optimiser counts its own updates (about 41,000), so it only reached 0.07% of the planned decay. PPO trained at a constant, high learning rate, the most likely cause of the four degrading runs.
- **PPO's automatic reset leaked information.** When a training flight ended, only the drone's position and observation were reset; the "previous position" and "previous action" carried over, so the first reward of every new flight was wrong.
- **The target offset was divided by the starting distance.** With one start this is a constant, but with random starts the same situation would look different from flight to flight. Batch 2 measures the offset in metres.
- **Only the final checkpoint was kept**, so the best moment of a run could not be studied afterwards. Batch 2 keeps a snapshot every 100k steps.
- **The lab's real-drone code had two blocking problems:** it fed the PPO network unnormalised observations, and it converted the body moments into turn-rate commands capped at 1 degree per second, so only thrust was really controlled.

Since the project's contribution is about TD-MPC2, PPO was not repaired or re-run; its bugs are documented and its results kept as a historical baseline.

# 4. Designing Batch 2

## 4.1 Random starts from a region, with a held-out block

Every Batch 2 training flight starts at a new random position, drawn uniformly from a box: x and y between -0.75 and +0.75 m, height between 0.2 and 1.5 m, at least 0.3 m from the target. One block, x and y both between 0.25 and 0.75 m (all heights), is never used for training; it is kept to test interpolation. The drone always starts level and at rest. The figure below shows the area together with every fixed test start (Section 4.3) and validation start.

![The Batch 2 start area seen from above (left) and from the side (right). Blue box: training area; orange square: held-out block. Dots: the fixed test starts by group; grey crosses: validation starts; star: the target. In the side view, some outside starts appear inside the box because they are outside in the y direction.](code/src/tdmpc2_training/reports/figures/start_map.png)

The region was chosen large enough to require real control in all directions (up to about 1 m from the target) but not so large that learning would need many millions of steps.

## 4.2 The target offset in metres

As explained in Section 3.3, Batch 2 gives the agent the offset to the target in metres instead of dividing it by the starting distance. The change is an option (`metric_target`) that is off for every Batch 1 experiment, which therefore remain reproducible exactly.

## 4.3 Fixed test starts

Every agent of every experiment is tested from exactly the same starting positions, drawn once from a fixed random seed (2026) and stored in the code:

| Group | Starts | What it measures |
|---|---|---|
| batch1 | the six Batch 1 starts, 3 flights each | continuity with Batch 1 |
| train_area | 50 | new samples from the training distribution |
| held_out | 30 | the held-out block: interpolation to never-seen starts |
| outside | 30 | 0.25 to 0.75 m beyond the edge of the area: extrapolation |

A separate set of 40 **validation** starts (seed 4242) is used only to choose checkpoints, so that choosing the "best" checkpoint can never use the test results.

## 4.4 What is measured

The measures were fixed before any Batch 2 result was seen: **success** (full flight and late error below 10 cm), **crash**, **late error**, its horizontal and vertical parts, and **time within 10 cm**. Each test is repeated with planning, with the policy network only, and with a small 64×2 planner, all from the same starts and with the same sensor noise, so that differences can only come from the decision rule.

## 4.5 The experiments

| ID | What | Seeds × steps | Question |
|---|---|---|---|
| R1 | published TD-MPC2, random starts | 11 × 1M | does it learn to fly from anywhere, and generalise? |
| C1 | published TD-MPC2, single start, same observation | 3 × 1M | control: the effect of random starts alone |
| D | small TD-MPC2, random starts | 11 × 1M | does a smaller, faster model do as well? |
| H2 | small TD-MPC2, random starts and randomised physics | 3 × 1M | does planning help when the model has seen varied physics? |
| R4 | Batch 1 agents on the Batch 2 test starts | 25 agents | before vs after |
| A1 | Batch 1 agents re-flown 20 times | 33 agents | was the seed spread real or evaluation noise? |
| A2 | learning curves vs training losses | 3 runs | do the losses explain the swings within a run? |
| P | speed and decision time on Narval | none | GPU use, inference time |
| E | parallel-drone projection | none | can TD-MPC2 use many drones at once? |
| H1 | world-model error and planner variants | all agents | why does planning hurt single-start agents? |
| Dist. | wind, gusts, mass, motors, delay | all agents | robustness, with and without planning |
| Snap. | every 100k-step snapshot | all Batch 2 runs | swings within a run, best vs final checkpoint |

## 4.6 How the code was checked before anything ran

Long cluster runs are expensive to repeat, so the code was checked in four layers before launch:

- **Automatic checks** (`check_batch2.py`, 23 checks): starts always inside the area and never in the held-out block; the observation with both options off identical to the lab's original; the metric offset correct in metres; physics and reward unaffected by the observation option; new starts every flight, reproducibly; test starts fixed and in the right places.
- **Regression tests**: the old code (from git, before Batch 2) and the new code tested the same saved agents and produced identical results, down to every step of every flight. For training, a run on the CPU gave identical learning curves with old and new code. (On the GPU, even two runs of the old code differ slightly once learning starts, because the GPU adds numbers in an order that is not always the same; the CPU is exactly repeatable, so it is the right referee.)
- **End-to-end runs** of the full pipeline at home, reading back from the agent's own memory that its training flights really started all over the area.
- **A short test job on Narval** before the real launch.

The same discipline was applied to every later tool: each has its own check script or validation run, recorded in the research log.

## 4.7 How the night was run on Narval

The experiments ran on the Digital Research Alliance cluster **Narval**, each on a 2/7 slice of an NVIDIA A100 GPU. Each training job records the exact code version (git commit) it ran. Jobs longer than 3 hours wait much longer in the queue, so long runs were split into pieces of at most 3 hours, each resuming from the previous one's checkpoint (with a fix so a resumed run does not replay the random starts it already used). Follow-up tests were chained with job dependencies, so each started by itself the moment its training finished. More than 150 jobs ran overnight without a failure.

# 5. Results, experiment by experiment

*This chapter is completed once all results are in.*

# 6. Answers to Professor Bellinger's questions

*This chapter is completed once all results are in.*

# 7. The code, file by file

## 7.1 The repository at a glance

The repository (`uo-mirl-lab/multi-agentCrazyflieLift`, branch `tdmpc2-hover`) contains the lab's earlier work and this project's work side by side:

| Folder | Contents | Who wrote it |
|---|---|---|
| `code/src/gpu_training/` | the MJX drone simulator, PPO and multi-drone MAPPO training | previous students |
| `code/src/hardware_model_testing/` | code that flew trained policies on real Crazyflies | previous students |
| `code/src/initial_cpu_training/` | the lab's first CPU-based experiments | previous students |
| `code/src/tdmpc2_training/` | everything about TD-MPC2: this project | Mounir |
| `outputs/` | all results on the home PC (not in git, large) | generated |
| `results/tdmpc2_batch1/` | a curated copy of the Batch 1 results (in git) | generated |
| `docs/` | the lab's reports and slides | previous students |

From the lab's folders, this project uses only the simulator: `gpu_training/crazyflie_env.py` (the drone environment: reset, step, observation, reward), `constants.py` (physical constants, reward weights), `math_utils.py` (noise and quaternion helpers) and the MuJoCo drone model in `assets_mjx/bitcraze_crazyflie_2/` (`cf2.xml` for the drone, `scene.xml` for the world). None of these files was changed. The lab's `ppo_training.py` and `training_plots.py` are used only by the historical PPO script, and `hardware_model_testing/run_hardware.py` was audited but not used.

## 7.2 How the pieces fit together

A **training run** flows through the files like this:

```
cluster/seed_job.sh            (on Narval; run_seeds.py at home)
  -> train_hover.py            registers the environments, starts the trainer
     -> tdmpc2_jax/train.py    the training loop, configured by a YAML file in tdmpc2_jax/
        -> crazyflie_hover.CrazyflieHoverEnv   one simulated drone (random starts, metric offset)
           -> crazyflie_hover.HoverDrone -> gpu_training/crazyflie_env.CrazyflieEnv  (MJX physics)
        -> tdmpc2_jax/build.py -> tdmpc2.TDMPC2 + world_model.WorldModel             (the agent)
        -> every 25k steps: hover_eval.FlightTester -> learning_curve.csv
        -> checkpoint/ (final agent + replay buffer), snapshots/ (agent every 100k steps)
  -> eval_hover.py             final tests -> evaluation/, evaluation_policy_only/, evaluation_plan64x2/
```

A **test** of a saved agent (any of the `eval_*.py` scripts) rebuilds the agent from the run's own saved configuration (`.hydra/config.yaml`) and loads its weights with `eval_hover.load_agent`, then flies it with `hover_eval.FlightTester` or one of its variants. **Analysis** scripts (`analyze_*.py`) read the CSV files the tests wrote and produce plots and tables. On Narval, each kind of job has a small shell script in `cluster/`, and `cluster/pull_results.sh` copies the results home.

## 7.3 The task and the environment

- **`crazyflie_hover.py`**: the bridge between the lab's simulator and the Gymnasium interface the trainer expects. It defines the action limits (`ACTION_LOW`, `ACTION_HIGH`), the Batch 2 start area (`START_AREA_LOW/HIGH`, `HELD_OUT_XY_LOW/HIGH`, `MIN_GOAL_DISTANCE`) and its sampler `sample_training_start()`. `HoverDrone` wraps the lab's environment and adds the option of the target offset in metres. `place_drone()` moves a freshly reset drone to any start. `CrazyflieHoverEnv` is the environment the trainer uses: `reset()` starts a flight (a random start when `random_starts` is on) and `step()` applies one action and returns the observation, reward and end flags.
- **`crazyflie_hover_dr.py`**: `RandomizedHoverEnv`, the same environment with randomised physics in every flight (mass, motor strength, wind), for the H2 experiment.

## 7.4 Training

- **`train_hover.py`**: the entry point. It registers the environments with Gymnasium (`CrazyflieHover-v0`, and `CrazyflieHoverRandomized-v0` for H2) and starts the trainer with a configuration chosen by `--config-name`.
- **`tdmpc2_jax/`**: TD-MPC2 in JAX, copied from Shane Flandermeyer's open-source implementation (commit 5b05ff4, MIT licence in `UPSTREAM_LICENSE`) and adapted.
  - `train.py`: the training loop. It collects experience, stores it in the replay buffer, runs one learning update per step, records test flights every 25k steps, saves checkpoints and snapshots, and can resume an interrupted run (with fresh randomness).
  - `build.py`: builds an untrained agent from a configuration. Training and testing both use it, so saved weights always fit.
  - `tdmpc2.py`: the TD-MPC2 algorithm. `act()` encodes the observation and calls `plan()` (MPPI planning) or the policy network; `update()` performs one learning step of the world model, the value ensemble and the policy; `estimate_value()` scores imagined plans.
  - `world_model.py`: the networks (encoder, dynamics, reward, policy, value ensemble and its slowly updated target copy) and their optimisers.
  - `networks/`, `common/`, `data/`: building blocks (normalised linear layers, ensembles, the activation functions, the two-hot reward and value encoding, the replay buffer in `data/sequential_buffer.py`). `envs/` holds upstream code for other benchmarks that this project does not use.
- **Configurations** (`tdmpc2_jax/*.yaml`, read by Hydra): `config_hover.yaml` (small settings, the base of all others), `config_hover_standard.yaml` (published settings), `config_batch2_random_start.yaml` (R1), `config_batch2_fixed_start.yaml` (C1), `config_batch2_random_start_small.yaml` (D), `config_batch2_randomized_small.yaml` (H2). `config.yaml` is the upstream default and is not used.
- **`run_seeds.py`**: trains and tests a list of seeds on the home PC (used for Batch 1).

## 7.5 Testing trained agents

- **`hover_eval.py`**: the test-flight engine. It defines the test starts (`TEST_STARTS`, `TEST_SETS`, `CURVE_STARTS`), the measures (`Flight.summary()`), and `FlightTester`, which flies many flights side by side inside one compiled GPU loop, with planning or with the policy network only.
- **`eval_hover.py`**: tests a saved run on any test group, with planning, the policy network only (`--no-mpc`) or another planner size (`--plan 64x2`), at the final checkpoint or a snapshot (`--snapshot`). It refuses to overwrite existing results.
- **`eval_disturbances.py`**: flies an agent in wind, gusts, with extra mass, weaker motors or delayed commands (`WindyDrone`, `DisturbedFlightTester`) and records how each flight ended (`end_cause`).
- **`eval_snapshots.py`**: flies every kept snapshot from the validation starts and all test groups.
- **`eval_planner_variants.py`**: flies the same agent with the policy only, the trained planner, a narrower planner, and a planner with half its candidates from the policy network.
- **`eval_horizon.py`**: flies the same agent with planning horizons of 1, 3, 5 and 10 steps.
- **`model_error.py`**: records flights and measures how wrong the agent's own world model is along them.

## 7.6 Analysis

- **`analyze_batch2.py`**: the main Batch 2 analysis: learning curves per test group, final results per group and decision rule, comparisons between experiments with bootstrap confidence intervals, disturbance results and crash causes.
- **`analyze_snapshots.py`**, **`analyze_model_error.py`**, **`analyze_planner_variants.py`**: summaries of the corresponding tests across seeds.
- **`variance_decomposition.py`** (A1) and **`fluctuation_analysis.py`** (A2): the two analyses of Batch 1 variability.
- **`analyze_seeds.py`** and **`compare_methods.py`**: the Batch 1 plots and the TD-MPC2 vs PPO comparison.

## 7.7 Checks (automated tests)

- **`check_batch2.py`**: 23 checks of the Batch 2 task code (Section 4.6).
- **`check_disturbances.py`**: 10 checks of the disturbance physics with a scripted pilot.
- **`check_deployment.py`**: checks that the deployment adapter builds exactly the simulator's observation.
- **`check_hover.py`**: the original Batch 1 sanity check of the environment.

## 7.8 Measurements of speed

- **`bench_latency.py`**: time per control decision for three network sizes and five planning budgets.
- **`profile_training.py`**: where one training step spends its time.
- **`bench_parallel.py`**: timing of simulation, planning and updates with many drones at once, and the projected training speed.

## 7.9 Deployment preparation

- **`deploy_controller.py`**: a simulation-only adapter for a future real flight. `DroneState.from_crazyflie_log()` converts a state estimate in the Crazyflie's own units and conventions; `build_observation()` reproduces the training observation exactly; `TDMPC2Controller` runs a trained agent, times each decision and must be warmed up before use. `to_crazyflie_command()` deliberately refuses to produce a drone command, because the action space does not yet match what the firmware accepts.

## 7.10 Historical PPO

- **`ppo_hover.py`**: trains the lab's PPO on exactly the Batch 1 task and tests it with the same test flights. It inherits the two bugs described in Section 3.3; its results are kept as a historical baseline.

## 7.11 Cluster scripts (`cluster/`)

- `python.sh`: runs Python inside the container that provides the exact package versions (`requirements-lock.txt`), installed once by `setup_env.sh`.
- `seed_job.sh`: trains and tests one seed per array task, optionally in pieces (`SKIP_TESTS`) and with the extra tests without planning (`DEPLOYMENT_TESTS`).
- `eval_job.sh`, `disturb_job.sh`, `snapshot_job.sh`, `model_error_job.sh`, `variants_job.sh`: run the corresponding test on a list of runs, one per array task.
- `profile_job.sh`, `parallel_job.sh`, `test_job.sh`: speed measurements.
- `ppo_job.sh`: the historical PPO runs.
- `pull_results.sh`: copies all results from Narval to the home PC, without the large checkpoints.

## 7.12 Documents

- `README.md`: how to install and run everything.
- `CONTEXT.md`: the project's history and status, machines and environments.
- `EXPERIMENTS.md`: the list of experiments, their validation and the launch log.
- `reports/`: `RESEARCH_LOG.md` (chronological log), `experiment_registry.json` (machine-readable list of every experiment), `RESULTS_SUMMARY.md`, `NEXT_STEPS.md`, `DEPLOYMENT_READINESS.md`, `LITERATURE.md`, and this guide (`GUIDE.md`, built into a PDF by `make_guide.py`).

## 7.13 Where the results are

Every training run has its own folder (`outputs/batch2/<experiment>/seed_XX/` at home, `$SCRATCH/tdmpc2/batch2/<experiment>/seed_XX/` on Narval) with the same layout:

| Item | Contents |
|---|---|
| `.hydra/config.yaml` | the exact configuration of the run |
| `learning_curve.csv` | test flights during training, every 25k steps |
| `checkpoint/` | the final agent and its replay buffer (Narval only) |
| `snapshots/` | the agent every 100k steps (Narval only) |
| `tensorboard/` | training statistics (losses) |
| `evaluation*/` | final tests: `summary.csv` (one row per flight), `trajectories.csv` (every step) |
| `disturbances*/`, `snapshot_evaluation*/`, `model_error/`, `planner_variants/`, `horizon_test/` | the follow-up tests |
| `gpu_usage.csv`, `*.log` | GPU samples and the training log |

Each experiment folder also has `batch_info.json`, which records the settings and the code version. The old Narval results were backed up to `/project/def-cbelling-ab/mounirz/tdmpc2_scratch_backup_2026-11-01/` before the scratch purge.

## 7.14 Reproducing the results

All commands run from the repository root. At home, Python is `.venv/bin/python`; on Narval, `bash code/src/tdmpc2_training/cluster/python.sh`.

```
# checks
.venv/bin/python code/src/tdmpc2_training/check_batch2.py
.venv/bin/python code/src/tdmpc2_training/check_disturbances.py
.venv/bin/python code/src/tdmpc2_training/check_deployment.py <batch 1 run> <batch 2 run>

# train one seed of R1 (about 7 hours on a GPU slice)
.venv/bin/python code/src/tdmpc2_training/train_hover.py --config-name=config_batch2_random_start seed=0

# test it
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run>             # with planning
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run> --no-mpc    # policy network only
.venv/bin/python code/src/tdmpc2_training/eval_disturbances.py <run>
.venv/bin/python code/src/tdmpc2_training/eval_snapshots.py <run>

# on Narval: a whole experiment, then copy the results home
sbatch --array=0-10 --time=10:00:00 --export=ALL,CONFIG=config_batch2_random_start,\
BATCH=batch2/random_start_1M,STEPS=1000000,DEPLOYMENT_TESTS=1 code/src/tdmpc2_training/cluster/seed_job.sh
bash code/src/tdmpc2_training/cluster/pull_results.sh

# analysis
.venv/bin/python code/src/tdmpc2_training/analyze_batch2.py outputs/batch2/random_start_1M \
    --compare outputs/batch2/random_start_small_1M outputs/batch2/fixed_start_1M outputs/batch1_standard_1M

# this guide
~/.local/bin/uv run --no-project --with reportlab python code/src/tdmpc2_training/reports/make_guide.py
```

# 8. Limitations

*This chapter is completed once all results are in.*

# 9. What to do next

*This chapter is completed once all results are in.*

# Appendix A. Glossary

*Completed with the final version.*

# Appendix B. Jobs and code versions

*Completed with the final version.*
