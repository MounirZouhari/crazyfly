% Crazyflie + TD-MPC2: complete project context
% Every document, result table and key source file, in one place
% Mounir Zouhari, University of Ottawa (supervisor Prof. Colin Bellinger)
% Compiled 2026-11-09

[[TOC]]

# Part 1. Status on November 9 (newest; read first)

Source: `code/src/tdmpc2_training/reports/STATUS_2026-11-09.md`, last modified 2026-11-09.

## Status on November 9, 2026

This note is the newest document in the project. Where it disagrees with an older document, this note is the correct one. It covers what happened after the Batch 2 documents of November 2: a meeting with Prof. Bellinger, the decision to continue with the small model, a diagnosis of why the domain-randomisation pilot (H2) failed, and the experiment proposed to fix it. Nothing new has been trained since November 2.

### The project in brief

- **People.** Mounir Zouhari, honours research student at the University of Ottawa. Supervisor: Prof. Colin Bellinger. The lab repository (`uo-mirl-lab/multi-agentCrazyflieLift`) also holds other students' multi-agent PPO (MAPPO) work, which this project reuses but does not change.
- **Goal.** Train a controller that flies a simulated Crazyflie 2 quadrotor (27 g) from anywhere in a region to a target point and hovers there, using TD-MPC2. The long-term goal is to run it on the real drone.
- **Simulator.** The lab's MuJoCo MJX environment (`code/src/gpu_training/crazyflie_env.py`), used with one drone. Control runs at 50 Hz (one step = 0.02 s) and the physics at 500 Hz. A flight lasts at most 500 steps (10 s). It ends early on a crash: hitting the ground or leaving the allowed box.
- **Observation.** 31 numbers, in this order:
  - world position (3), world velocity (3) and angular velocity (3);
  - rotation matrix (9);
  - target offset in the body frame (3) and velocity in the body frame (3);
  - offset to the other drones (3, always zero with one drone);
  - the previous command (4).

  Only **one** previous command is included (`N_PREV_ACTIONS = 1`), and there is no other history. The observation has small sensor noise. Batch 2 measures the target offset in metres (option `metric_target`). The lab's original version divides it by the distance at the start of the flight.
- **Actions.** 4 numbers: collective thrust in [0, 0.35] N, and roll, pitch and yaw moment commands in [-1, 1] (each scaled by 1e-5 N·m in the model file). Hovering needs 0.265 N, which is 76% of the maximum thrust.
- **Reward (the lab's).** At each step the reward is the sum of four terms: the decrease in distance to the target; 0.01 × (1 − tanh(distance)); a survival bonus of 0.001; and −1 on a crash. There is no penalty on command changes and none on effort.
- **Algorithm.** TD-MPC2 (Hansen et al.), JAX implementation by Shane Flandermeyer, vendored into `code/src/tdmpc2_training/tdmpc2_jax/`. It learns a latent world model, a reward predictor, a value function and a policy network. At every step it plans with MPPI over the learned model, warm-started by the policy network. The **published** size has 5.0M weights and a 512×6 planner. The **small** size has 0.39M weights, latent 128 and a 128×3 planner. "Policy network only" means acting with the policy network, without planning.
- **Success metric.** A flight succeeds when the drone flies the full 10 s and its mean distance to the target over the last 2 s (the *late error*) is below 10 cm. Results are given as success / crash percentages for three groups of test starts:
  - inside the training area;
  - a held-out block of the area that training never uses;
  - outside the training area.

### Experiment names

| Name | Model | Training starts | Seeds | Steps | Folder in `outputs/batch2/` |
|---|---|---|---:|---:|---|
| Batch 1 | both | one fixed start (0, 0, 0.2) to target (0, 0, 1) | 11 | 50k (3 published seeds to 1M) | `outputs/batch1_*` |
| R1 | published | a new random start every flight | 11 | 1M | `random_start_1M` |
| C1 | published | the Batch 1 start only | 3 | 1M | `fixed_start_1M` |
| D | small | random starts | 11 | 1M | `random_start_small_1M` |
| H2 | small | random starts, plus random mass, motor strength and wind | 3 | 1M | `randomized_small_1M` |
| Reward pilot ("sharp") | small | random starts, extra reward within about 5 cm | 3 | 1M | `sharp_small_1M` |
| R1 to 2M | published | random starts (R1 seeds 0-2 continued) | 3 | 2M | `random_start_2M` |

### Headline results (from the November 2 documents)

Success rates, given as training area / held-out block / outside:

| Agent | Result |
|---|---|
| D with planning | **78 / 83 / 59%**, crashes 0 / 0 / 3% |
| D at the snapshot chosen on validation starts | **97 / 91 / 70%** (the best controller so far) |
| R1 with planning | 46 / 39 / 20%, crashes 0 / 2 / 26% |
| R1 policy network only | 60 / 64 / 40% |
| R1 policy network only, at its best snapshot | 90 / 81 / 59%, 0.3 ms per decision |
| C1 (single start) | 26 / 4 / 0%, crashes 35 / 82 / 92% |

- Random starts cause the generalisation: R1 − C1 = +20 / +34 / +20 points.
- R1 at 2M (seeds 0-2): planning success rose from 66 to 72% (training area) and from 65 to 75% (held-out). But crashes from outside starts rose from 22 to 58%. The hypothesis, not yet checked, is that the 1M-transition replay buffer discards the early, wide-ranging flights.
- H2: 13 / 7 / 4% (negative; diagnosed below).
- Reward pilot: 37 / 66 / 23%, best snapshot 91 / 78 / 47%. The value function now peaks at the target, but the slow precision drift remained.

### Meeting with Prof. Bellinger (between November 2 and 9)

Mounir's raw notes, verbatim:

```
h step wise training reward… x axis is step and y axis is reward at that time step over im we
should se emore rearewd over step .. ad dpenalty in reward that penalties the big variable of
time steps .
Use domain randomization . use small range now my mistake .. ad domain rndomaxtion . in
sumialtion you can try add a leg in in practice thyere gonn ae slightly. Add
```

Interpretation (agreed with Mounir):

- 1. **Reward per time step plot.** Plot the reward received at each step of a flight: the step (0 to 500) on the x axis, the reward at that step on the y axis. Overlay curves from several training stages (the 100k-step snapshots). A learning agent should climb sooner and stay higher at later stages. This needs no training: test flights already record the reward at every step (`hover_eval.py`, `trajectories.csv`), and every run kept snapshots.
- 2. **Penalty on large command changes between steps.** Subtract w·‖a_t − a_{t−1}‖² from the reward. The flight-quality analysis showed the planner's commands jump between steps. So far this was fixed after the fact with a 40 ms command filter or the averaged plan. A reward penalty would make the world model predict the cost, so the planner itself would prefer smooth plans. The weight w needs a small sweep: too large makes the drone sluggish.
- 3. **Domain randomisation, with small ranges, plus a command lag.** On the real drone, commands arrive late (radio) and motors respond with delay, so the simulator should add a lag during training. Code for a command delay already exists in the disturbance tester (`eval_disturbances.py`), but it was only used for testing.

### Decision: continue with the small model

| | Small (D) | Published (R1) |
|---|---|---|
| Success with planning at 1M | 78 / 83 / 59% | 46 / 39 / 20% |
| Crashes from outside starts | 3% | 26% (58% at 2M) |
| Time per decision, 2 CPU cores | 11.4 ms (4.4 ms with a 64×2 planner) | 516 ms |
| Training speed (Narval A100 slice) | 88 steps/s, about 4 h for 1M | 49 steps/s, about 14 h for 2M |

The control period is 20 ms, so only the small planner runs in real time without a GPU. Caveat: the small model needs its planner. Its policy network alone reaches only 9%.

### Why H2 failed: diagnosis of November 9

H2 trained the small model with random starts plus, in each flight, a mass and inertia factor drawn from U(1.0, 1.15), a motor strength factor from U(0.9, 1.0), and a steady wind of 0-3 m/s from a random horizontal direction (`crazyflie_hover_dr.py`). It scored 13 / 7 / 4% against D's 78 / 83 / 59%, even in undisturbed tests. Domain randomisation is expected to work, so the setup was examined.

**Checked and correct.** Mass, inertia and motor strength reach the physics: the lab's `step` rebuilds the MJX model from `state.info` every step. The wind reaches it too: `WindyDrone` swaps `opt.wind` in. Each reset starts from nominal values, so the factors do not compound. H2 and D both trained to 1,000,000 steps. The "steps: 500000" in `batch_info.json` is the length of each split Slurm job, not the total.

**Finding 1: the wind was not too strong.** This corrects an earlier claim that it was. The wind force was measured in MuJoCo with the drone at rest, at air density 1.225:

| Wind | Push on the drone | Sideways force available at full thrust |
|---|---|---|
| 0.5 m/s | 0.8 mN | 229 mN (nominal); 80 mN (mass ×1.15, motor ×0.9) |
| 1 m/s | 3.1 mN | same |
| 2 m/s | 12.2 mN | same |
| 3 m/s | 27.5 mN | same |

At 3 m/s, holding position needs a lean of about 6°. Yet every agent, H2 included, fails at 2 m/s.

**Finding 2 (main cause): the agent cannot tell which conditions it is flying in.** The observation is a single snapshot plus one previous command. Inferring mass, motor strength or wind requires comparing commands with the motion they produced over several steps. TD-MPC2's world model is deterministic and sees only the current observation. With hidden physics it can only learn the average response, and the planner then plans with a model that is wrong for any particular flight. The evidence: planning adds 69 points over the policy network for D but *loses* 21 points for H2. Randomisation damaged the world model most of all. A steady wind also calls for a constant lean, a kind of integral action, which a controller without memory cannot learn to adjust per flight.

**Finding 3: the ranges were one-sided, not centred on the real drone.** The training drone was on average heavier and weaker than the nominal one:
- it needed on average 86% of maximum thrust to hover, against 76% nominal;
- 55% of flights needed more than 85%, and 18% needed more than 90%, which leaves little thrust for steering;
- the nominal drone sat at the edge of the training range.

This matches Prof. Bellinger's "use a small range".

**Finding 4: three randomisations at once, with only 3 seeds.** The effect of each cannot be separated.

Status of the diagnosis: findings 1, 3 and 4 and the planning-versus-policy numbers are measured facts. That missing memory is *the* main cause is a hypothesis, which the next experiment tests.

### Proposed next experiment (not launched; waiting for Mounir's go-ahead)

| | H2 (done) | H3a | H3b |
|---|---|---|---|
| Mass and inertia | ×1.0-1.15 | ×0.95-1.05 | ×0.95-1.05 |
| Motor strength | ×0.9-1.0 | ×0.95-1.05 | ×0.95-1.05 |
| Wind | 0-3 m/s | 0-1 m/s | 0-1 m/s |
| Command lag | none | 0 or 1 step (0-20 ms), drawn per flight | same |
| Memory | 1 previous command | 1 previous command | last ~5 observations and commands |

- **Setup.** Small model, random starts, 1M steps, 3 seeds each, on Narval (about 4 h in parallel).
- **Baseline.** Compare with D on the same test starts and the same disturbance tests. Success means staying near D's undisturbed results while beating D under wind, mass, motor and delay disturbances.
- **Code.** The changes go in a new variant of the TD-MPC2 adapter, so the lab's code and the H2 file stay unchanged. Before launching, automatic checks will confirm that the ranges are centred, that the lag delays commands by exactly the drawn number of steps, and that the memory holds the right past steps.
- **Afterwards.** The smoothness penalty (2-3 weights, 3 seeds), and the reward-per-step plot from item 1.

### Open questions

- Why does R1 at 2M crash more from outside starts? To check: the replay-buffer contents at 1M vs 2M, and the 1.1M-2M snapshots.
- What causes the precision drift: planner sampling or continual learning updates?
- How much history H3b needs, and whether to feed raw history or an estimate of mass and wind.
- Which smoothness weight keeps precision while removing command jumps.
- Hardware: firmware-compatible commands (collective thrust plus body rates), state estimation, radio delay and motor response. Every hardware step needs Prof. Bellinger's approval and the lab's safety rules.

### How the rest of this document is organised

The parts that follow are the project's existing documents, unchanged, newest and most important first:
- the complete guide (November 2);
- the results summary, the handoff, the next steps and the supervisor update;
- the plain-language summary;
- the experiment log and the code README;
- the deployment readiness note, the literature notes and the research log;
- the Batch 1 context (October 13, historical);
- the raw result tables from every analysis;
- the key source code.

Each part's heading gives its source file and date. Where an older part disagrees with this note, this note is correct. In particular, older parts do not contain the H2 diagnosis above.

# Part 2. The complete project guide

Source: `code/src/tdmpc2_training/reports/GUIDE.md`, last modified 2026-11-02.



## 1. How to read this guide

This guide explains everything that was done in the project up to November 2, 2026: what each experiment asked, why it was designed the way it was, what came out, what the result means, and what can be done next. It also describes every file of code, so that any question about how the project is organised can be answered from it. It assumes no background in reinforcement learning beyond what is explained in Chapter 2.

The chapters build on each other. Chapter 2 explains the ideas: reinforcement learning, the simulated drone, PPO, TD-MPC2, planning, and generalisation. Chapter 3 summarises where the project stood after the first batch of experiments and what Professor Bellinger asked for. Chapter 4 explains how the second batch was designed and how the code was checked before anything ran. Chapter 5 goes through every experiment and its results. Chapter 6 answers the professor's questions one by one. Chapter 7 is a tour of the code, file by file, with the commands to reproduce each result. Chapter 8 lists the limitations, and Chapter 9 the next steps. Appendix A is a glossary of every technical term used, and Appendix B lists the cluster job numbers and code versions.

A few conventions are used throughout. A **flight** is one test of a trained drone controller, lasting 10 simulated seconds. A flight is a **success** when the drone flies the full 10 seconds and its average distance to the target over the last 2 seconds (the **late error**) is below 10 cm. A flight is a **crash** when it ends early, either because the drone touched the ground or because it flew too far away. Most results are given as a **mean ± standard error** across several independently trained controllers (**seeds**); the standard error says how precisely the mean is known. Where a difference between two methods matters, a **95% confidence interval** is given in square brackets: if the interval does not contain zero, the difference is very unlikely to be due to chance alone.

### 1.1 The experiments and their names

Every Batch 2 experiment has a short name, used throughout the guide. A **step** is one decision of the drone (0.02 s of simulated flight); **1M** means one million steps of training, about 5.5 hours of simulated flying. The **published** model uses TD-MPC2's settings from its paper (5.0 million weights); the **small** model is a reduced version (0.39 million weights) that trains and plans faster.

| Name | Model | Where training flights start | Seeds | Steps | Purpose | Folder in `outputs/batch2/` |
|---|---|---|---:|---:|---|---|
| Batch 1 | both | always the same start, (0, 0, 0.2) | 11 (3 published at 1M) | 50k (1M) | the "before" picture | `outputs/batch1_*` |
| **R1** | published | a new random start every flight | 11 | 1M | the main experiment | `random_start_1M` |
| **C1** | published | always the Batch 1 start | 3 | 1M | control for R1: is it the random starts that matter? | `fixed_start_1M` |
| **D** | small | random starts | 11 | 1M | R1 with the small model | `random_start_small_1M` |
| **H2** | small | random starts, plus random mass, motor strength and wind | 3 | 1M | pilot: does varied physics help planning? | `randomized_small_1M` |
| reward pilot | small | random starts, extra reward near the target | 3 | 1M | pilot: does a sharper reward steady the precision? | `sharp_small_1M` |
| R1 to 2M | published | random starts | 3 | 2M | is the published model just slower? | `random_start_2M` |

### 1.2 The results in one page

- **Random starts work** (Sections 5.5 to 5.7). Trained from one start, TD-MPC2 crashes in most flights that start elsewhere (35-92% for the published single-start agents). Trained from random starts, it almost never crashes inside the area and does as well from a block of starts it never trained from. The control C1 shows that the random starts are what makes the difference.
- **The best controllers** (Section 5.12), choosing each run's checkpoint on separate validation starts: the small model with planning succeeds in 97% of the flights from the training area, 91% from the held-out block and 70% from outside; the published model's policy network alone, without any planning, in 90%, 81% and 59%, with no crash inside the area, at 0.3 ms per decision.
- **At 1M steps the small model beats the published one** (78% against 46% with planning from the training area), but the published model is still improving. Continued to 2M steps (3 seeds, Section 5.18), it becomes about as good as the small model inside the area (72-75% with planning) but crashes far more from starts outside it (58% against 22% at 1M): more training made it specialise.
- **When planning helps** (Sections 5.8 and 5.9, question Q7): planning turns the small model's imprecise policy network (21 cm) into a precise controller (7 cm), because its world model is accurate where it is used. It hurts where the planner can find the world model's errors: single-start agents away from their start, and the published model's hard search, which loses to its own policy network.
- **Disturbances** (Section 5.10, Q8): the small agents with planning shrug off extra mass, weaker motors and command delay (72-77% success); wind of 2 m/s or more defeats every agent, and in strong wind planning crashes more than the policy network.
- **Two pilots gave negative results** (Sections 5.11 and 5.13): training on varied physics cost most of the precision, and a sharper reward near the target, although learned, did not stop the precision from drifting between checkpoints.
- **For a real drone** (Sections 5.15, 5.16 and Q10): the planner's commands jump from step to step; a 40 ms filter (small model) or acting on the planner's averaged plan (both models) fixes it in simulation. The agents' commands are not ones the Crazyflie firmware accepts, so retraining is needed before any flight; nothing was flown.

## 2. The ideas behind the project

### 2.1 Reinforcement learning in one page

Reinforcement learning (RL) is a way of teaching a computer program to make decisions by trial and error. The program, called the **agent**, interacts with a world, called the **environment**, one small time step at a time. At every step the agent receives an **observation** (here: 31 numbers describing where the drone is, how it moves and how it is tilted), chooses an **action** (here: 4 numbers, the commands sent to the drone), and receives a **reward**, a single number saying how good the result of that action was. Nobody tells the agent which action is correct. It has to discover, from the rewards, which behaviour leads to a high total reward over a whole flight.

The rule the agent uses to choose actions is called its **policy**. In modern RL the policy is a neural network: a function with many adjustable numbers (its **parameters** or **weights**) that maps an observation to an action. Learning means adjusting these weights so that the total reward grows. Many methods also learn a **value function**, which estimates how much reward can still be collected from a given situation; it lets the agent judge the long-term consequences of an action without waiting for the end of the flight.

Learning needs experience. Each simulated step the agent takes is one **environment step** (here 0.02 seconds of simulated flight). How many environment steps a method needs to learn a task is its **sample efficiency**; how many seconds of computer time it needs is its **wall-clock** cost. Both matter: sample efficiency matters when experience is expensive (a real drone), wall-clock time matters when compute is the limit.

Training involves randomness: the starting weights, the exploration, the order in which experience is reused. The **seed** of a run fixes all of this randomness. Two runs with the same settings but different seeds can end up quite different, so a single run proves little. That is why the professor asked for 11 seeds per experiment, and why results are reported as averages with uncertainty.

### 2.2 The simulated Crazyflie

The Crazyflie 2 is a 27-gram quadrotor. The lab's simulator models it with **MuJoCo**, a physics engine, running on the graphics card through **MJX** (MuJoCo written in JAX, a library for fast numerical computing on GPUs). The physics advances in steps of 2 ms; the agent makes a decision every 20 ms, so the drone is controlled **50 times per second (50 Hz)**.

The **action** has four numbers. The first is the collective **thrust**, between 0 and 0.35 newtons. Hovering needs 0.265 N, which is 76% of the maximum, so there is little spare thrust. The other three are **body moments**, twisting forces around the drone's three axes, each between -1 and 1 (scaled to very small torques). The agent's networks work with all four numbers scaled to [-1, 1]; they are converted to these physical units before reaching the simulator.

The **observation** has 31 numbers: the position (3), the velocity (3), the angular velocity (3), the orientation as a 3×3 rotation matrix (9), the offset to the target expressed in the drone's own axes (3), the velocity in the drone's own axes (3), the offset to other drones (3, always zero with one drone), and the previous action (4). Small random noise is added to the measurements, as real sensors would (about 1 cm on the position).

The **reward** of every step is the sum of a tiny bonus for staying in the air (0.001), a bonus for being close to the target (up to 0.01, shrinking with distance), and the reduction of the distance to the target since the previous step. A crash costs -1. A flight ends early (a **crash**) when the drone drops below 5 cm or strays more than 2 m further from the target than where it started; otherwise it lasts 500 steps (10 s). The target is always the point (0, 0, 1), one metre above the origin.

### 2.3 PPO, the model-free method of the previous students

**PPO** (Proximal Policy Optimization) is a widely used **model-free** method: it learns the policy directly from experience, without learning how the world works. It collects a large batch of experience with the current policy, then improves the policy with a few gradient steps that are kept small on purpose (the "proximal" part), and repeats. The lab's PPO simulates **2,048 drones at once** on the GPU, so it collects experience extremely fast: tens of millions of environment steps in minutes. It needs that much, because model-free learning is sample-hungry.

### 2.4 TD-MPC2, the model-based method studied here

**TD-MPC2** (Hansen, Su and Wang, ICLR 2024) is a **model-based** method: besides a policy, it learns a **world model**, a set of networks that predict what will happen. It has five parts:

- an **encoder** that compresses the 31 observation numbers into an internal description, the **latent state** (128 or 512 numbers depending on the size);
- a **dynamics** network that predicts the next latent state from the current one and an action;
- a **reward** network that predicts the reward of an action in a latent state;
- a **value** network (an ensemble of five) that estimates the long-term reward of an action in a latent state;
- a **policy** network that proposes an action for a latent state.

All five are trained together from a **replay buffer**, a memory of all past experience, with one learning update for every environment step (an **updates-to-data ratio** of 1).

The distinctive part is how TD-MPC2 acts. Before every single decision it **plans**: it imagines many possible action sequences a few steps into the future with its world model, scores each one by the predicted rewards plus the value at the end, and executes only the first action of the best one. At the next step it plans again from the new situation. This is called **model predictive control** (MPC), and the planning algorithm is **MPPI**: it samples candidate plans (some proposed by the policy network, most drawn randomly around its current best guess), keeps the best ones (the **elites**), refits its sampling distribution to them, and repeats for several rounds.

Two sizes were used. The **published** settings (from the TD-MPC2 paper) have a latent state of 512 numbers, about 5 million learned weights, and plan with **512 candidate plans for 6 rounds**. The **small** settings, used in September, have 128 latent numbers, about 0.4 million weights, and plan with **128 plans for 3 rounds**. Both plan **3 steps ahead**, which is 0.06 seconds; everything after that is judged by the value network.

### 2.5 Planning versus the policy network alone

Because TD-MPC2 trains a policy network as part of its learning, a trained agent can be flown in two ways from exactly the same weights: **with planning** (the normal TD-MPC2) or with **its policy network only** (take the policy's proposed action, no imagination). Comparing the two isolates what planning with the world model contributes. This comparison is used throughout the project, because it answers the central question directly: does using the learned world model at decision time help or hurt?

### 2.6 Generalisation and distribution shift

A controller **generalises** when it works in situations it did not see during training. The set of situations seen in training is the **training distribution**. Tests can be inside it (new random samples of the same kind), in a deliberately **held-out** region inside its range that training never visited (**interpolation**), or **outside** its range (**extrapolation**, also called **out-of-distribution**). **Disturbances**, such as wind or extra weight, change the physics itself and are another kind of shift. Keeping these cases apart matters, because a method can be good at one and bad at another.

## 3. Where the project stood, and what was asked

### 3.1 Batch 1 (October 12-13)

In the first batch, every controller was trained from one fixed start, (0, 0, 0.2), to hover at (0, 0, 1). The main results were:

| Experiment | Seeds × steps | Result from the training start | From 5 other starts |
|---|---|---|---|
| small TD-MPC2, home PC | 11 × 50k | 0.54 ± 0.19 m | some crashes |
| small TD-MPC2, Narval | 11 × 50k | 0.32 ± 0.09 m (seeds from 6.5 cm to 1.05 m) | few crashes |
| published TD-MPC2 | 11 × 50k | 1.06 ± 0.14 m, mostly crashing | mostly crashing |
| published TD-MPC2 | 3 × 1M | 5.7 ± 0.5 cm | crashed 36 of 36 flights |
| PPO (lab code) | 11 × 78.6M | median 2.8 cm, 4 of 11 runs degraded | mostly crashing |

TD-MPC2 reached 10 cm with about 100 times fewer environment steps than PPO, but PPO was faster in real time. Both methods memorised the single route they were trained on.

### 3.2 Professor Bellinger's feedback

His reply raised the following points, which shaped everything in Batch 2:

- The large differences between seeds are unusual for TD-MPC2; the runs were probably too short.
- Was the cluster comparison done with PPO? How many parallel environments were used, and was the GPU utilisation monitored?
- Three starting positions are not enough; starting positions should be drawn at random from a region.
- An interesting question: does the world model help with situations outside the training distribution, such as new starts or external disturbances, and when and why is it most useful?
- How long does the big model take to decide, and what does that imply for a real drone? Is a smaller model enough, and how can it be made accurate?
- For PPO's late degradation, a lower or decaying learning rate might help.

These were turned into ten research questions (Chapter 6).

### 3.3 What the code audit found (November 1)

Before designing Batch 2, the code was audited. Five findings changed how earlier results must be read:

- **PPO's learning rate never decreased.** The schedule was written in environment steps (60 million) but the optimiser counts its own updates (about 41,000), so it only reached 0.07% of the planned decay. PPO trained at a constant, high learning rate, the most likely cause of the four degrading runs.
- **PPO's automatic reset leaked information.** When a training flight ended, only the drone's position and observation were reset; the "previous position" and "previous action" carried over, so the first reward of every new flight was wrong.
- **The target offset was divided by the starting distance.** With one start this is a constant, but with random starts the same situation would look different from flight to flight. Batch 2 measures the offset in metres.
- **Only the final checkpoint was kept**, so the best moment of a run could not be studied afterwards. Batch 2 keeps a snapshot every 100k steps.
- **The lab's real-drone code had two blocking problems:** it fed the PPO network unnormalised observations, and it converted the body moments into turn-rate commands capped at 1 degree per second, so only thrust was really controlled.

Since the project's contribution is about TD-MPC2, PPO was not repaired or re-run; its bugs are documented and its results kept as a historical baseline.

## 4. Designing Batch 2

### 4.1 Random starts from a region, with a held-out block

Every Batch 2 training flight starts at a new random position, drawn uniformly from a box: x and y between -0.75 and +0.75 m, height between 0.2 and 1.5 m, at least 0.3 m from the target. One block, x and y both between 0.25 and 0.75 m (all heights), is never used for training; it is kept to test interpolation. The drone always starts level and at rest. The figure below shows the area together with every fixed test start (Section 4.3) and validation start.

![The Batch 2 start area seen from above (left) and from the side (right). Blue box: training area; orange square: held-out block. Dots: the fixed test starts by group; grey crosses: validation starts; star: the target. In the side view, some outside starts appear inside the box because they are outside in the y direction.](code/src/tdmpc2_training/reports/figures/start_map.png)

The region was chosen large enough to require real control in all directions (up to about 1 m from the target) but not so large that learning would need many millions of steps.

### 4.2 The target offset in metres

As explained in Section 3.3, Batch 2 gives the agent the offset to the target in metres instead of dividing it by the starting distance. The change is an option (`metric_target`) that is off for every Batch 1 experiment, which therefore remain reproducible exactly.

### 4.3 Fixed test starts

Every agent of every experiment is tested from exactly the same starting positions, drawn once from a fixed random seed (2026) and stored in the code:

| Group | Starts | What it measures |
|---|---|---|
| batch1 | the six Batch 1 starts, 3 flights each | continuity with Batch 1 |
| train_area | 50 | new samples from the training distribution |
| held_out | 30 | the held-out block: interpolation to never-seen starts |
| outside | 30 | 0.25 to 0.75 m beyond the edge of the area: extrapolation |

A separate set of 40 **validation** starts (seed 4242) is used only to choose checkpoints, so that choosing the "best" checkpoint can never use the test results.

### 4.4 What is measured

The measures were fixed before any Batch 2 result was seen: **success** (full flight and late error below 10 cm), **crash**, **late error**, its horizontal and vertical parts, and **time within 10 cm**. Each test is repeated with planning, with the policy network only, and with a small 64×2 planner, all from the same starts and with the same sensor noise, so that differences can only come from the decision rule.

### 4.5 The experiments

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
| Hor. | planning horizon 1, 3, 5, 10 at test time | all Batch 2 agents | would looking further ahead help? |
| Smooth. | smoother ways of acting, slower motors | R1 and D agents | does the planner need its jumpy commands? |
| Sharp | small model with an extra reward near the target | 3 × 1M | is the drift of precision caused by the reward? |

The last three rows were added during the night, after the first results had raised the questions they answer (Sections 5.13, 5.14 and 5.16).

### 4.6 How the code was checked before anything ran

Long cluster runs are expensive to repeat, so the code was checked in four layers before launch:

- **Automatic checks** (`check_batch2.py`, 23 checks): starts always inside the area and never in the held-out block; the observation with both options off identical to the lab's original; the metric offset correct in metres; physics and reward unaffected by the observation option; new starts every flight, reproducibly; test starts fixed and in the right places.
- **Regression tests**: the old code (from git, before Batch 2) and the new code tested the same saved agents and produced identical results, down to every step of every flight. For training, a run on the CPU gave identical learning curves with old and new code. (On the GPU, even two runs of the old code differ slightly once learning starts, because the GPU adds numbers in an order that is not always the same; the CPU is exactly repeatable, so it is the right referee.)
- **End-to-end runs** of the full pipeline at home, reading back from the agent's own memory that its training flights really started all over the area.
- **A short test job on Narval** before the real launch.

The same discipline was applied to every later tool: each has its own check script or validation run, recorded in the research log.

### 4.7 How the night was run on Narval

The experiments ran on the Digital Research Alliance cluster **Narval**, each on a 2/7 slice of an NVIDIA A100 GPU. Each training job records the exact code version (git commit) it ran. The R1 and C1 runs were submitted as single jobs of up to 10 hours, and waited about 50 minutes in the queue before starting. Jobs of at most 3 hours start almost at once, so the later runs (D, H2 and the reward pilot) were split into pieces of at most 3 hours, each resuming from the previous one's checkpoint (with a fix so a resumed run does not replay the random starts it already used). Follow-up tests were chained with job dependencies, so each started by itself the moment its training finished. By noon the next day, all 300 job tasks had completed and none had failed (Appendix B).

## 5. Results, experiment by experiment

### 5.1 How to read this chapter

Every section follows the same pattern: the question, how it was tested, what came out, and what it means. Tables give **success / crash** shares as mean ± standard error across seeds; each seed is first averaged over its own flights, so every seed counts equally. Differences between two decision rules are **paired**: the same agents, the same starts and the same sensor noise, so that only the decision rule differs. Where a difference matters, its 95% bootstrap confidence interval is given in square brackets.

The sections go from the past to the present. Sections 5.2 to 5.4 look back at Batch 1 with new tests. Sections 5.5 to 5.8 present the main Batch 2 results: learning to hover from anywhere. Sections 5.9 to 5.14 explain when and why the world model helps or hurts, and what limits the precision. Sections 5.15 and 5.16 look at how the drone flies, beyond success and crashes, which matters for a real drone. Section 5.17 covers speed, and Section 5.18 the continuation of the published model to 2M steps.

### 5.2 Was the Batch 1 spread between seeds real? (A1)

**The question.** In Batch 1, the 11 small agents trained on Narval ended anywhere between 6.5 cm and 1.05 m from the target. Each agent had been tested with only 3 flights. There were two possible explanations. Either the agents really were different, because training with different seeds had produced controllers of different quality; or they were similar, and each test flight was so noisy (because of the sensor noise) that 3 flights could not tell them apart. The professor found the spread unusual for TD-MPC2, so it was important to know which explanation was right.

**How it was tested.** Every one of the 33 Batch 1 final agents (11 small agents from Narval, 11 small agents from the home PC, 11 published-size agents at 50k steps) flew 20 more flights from its training start: 5 different sensor-noise seeds with 4 flights each, the same noises for every agent. The spread of the late errors was then split into two parts with a one-way analysis of variance: the part between seeds (how different the agents are) and the part within seeds (how much the same agent varies from flight to flight). The **intraclass correlation (ICC)** is the share of the first part in the total.

| Agents | Between-seed SD | Within-seed SD | ICC |
|---|---:|---:|---:|
| small, Narval (11) | 0.32 m | 0.09 m | 0.93 |
| small, home (11) | 0.63 m | 0.16 m | 0.94 |
| published, 50k (11) | 0.38 m | 0.36 m | 0.52 |

![Each grey dot is one of the 20 re-flights of an agent, each blue dot that agent's mean. For the small agents (left and middle), the dots of one agent sit tightly together while the agents differ widely: the spread is a property of the agents. The published-size 50k agents (right) crash so often that each agent's flights also vary a lot.](code/src/tdmpc2_training/reports/figures/a1_seed_variance.png)

**What it means.** For the small agents, 93-94% of the spread comes from the agents themselves. The test was not the problem; the training was. One agent from the home PC (seed 0) ends 2.3 m from the target in all 20 re-flights: it is a stable failure of that agent, not bad luck. Together with the learning curves, which were still improving at 50k steps, and with the three 1M-step runs, which all converged to between 5.7 and 8.6 cm, this supports the professor's suspicion: at 50k steps, training had not finished, and each seed was caught at a different point of its learning. This is why every Batch 2 run trains for 1M steps.

### 5.3 Do the training losses explain the swings within a run? (A2)

**The question.** Even the 1M-step runs did not improve smoothly. After a run first hovered within 10 cm, between 13% and 39% of its later test points were worse than 10 cm again. TD-MPC2 logs several training statistics (the losses of the world model, the value and the policy). If these statistics rose whenever the flying got worse, they could be used to detect bad moments and stop training at a good one.

**How it was tested.** For each of the three 1M-step runs, every test point after the run first hovered (165 to 180 points per run) was paired with the average of each logged statistic over the 5,000 steps before it. A **Spearman correlation** then measured whether changes in the test error go together with changes in the statistic, and a **permutation test** checked whether the correlation could be due to chance.

**The result.** None of the statistics tracks the swings: all correlations between changes lie between -0.14 and +0.10, and none is statistically significant (the smallest permutation p-value is 0.075, even before correcting for having tested 18 correlations).

![Test error of the three 1M-step runs during training, with the logged training statistics. The swings in the test error have no counterpart in the losses.](code/src/tdmpc2_training/reports/figures/a2_fluctuations.png)

**What it means.** The losses measure how well the networks predict, not how well the drone flies. A world model can predict its training data slightly better or worse without the controller changing in any visible way, and the controller can get worse without the losses noticing. This is known as **objective mismatch** (Lambert et al., 2020). The practical consequence is that the only reliable way to know how good an agent is, is to fly it. Batch 2 therefore keeps a snapshot every 100k steps and has a separate set of validation starts for choosing a checkpoint (Section 5.12).

### 5.4 The Batch 1 agents on the new test starts (R4)

**The question.** How do controllers trained from a single start fly from the new Batch 2 test starts? This is the "before" picture against which Batch 2 is judged.

**How it was tested.** All 25 Batch 1 TD-MPC2 agents flew the 128 Batch 2 test flights (Section 4.3), once with planning and once with the policy network only, with the same sensor noise.

| Agents | Decision rule | Training area | Held-out block | Outside | Batch 1 starts |
|---|---|---:|---:|---:|---:|
| published, 1M (3) | planning | 20 ± 6% / 66 ± 3% | 2 ± 2% / 91 ± 9% | 2 ± 1% / 87 ± 5% | 20 ± 7% / 59 ± 4% |
| | policy only | 46 ± 12% / 15 ± 7% | 22 ± 17% / 54 ± 14% | 0 ± 0% / 77 ± 11% | 33 ± 17% / 6 ± 6% |
| published, 50k (11) | planning | 0 ± 0% / 98 ± 1% | 0 ± 0% / 98 ± 1% | 0 ± 0% / 100 ± 0% | 0 ± 0% / 87 ± 3% |
| | policy only | 1 ± 1% / 76 ± 8% | 0 ± 0% / 87 ± 4% | 0 ± 0% / 82 ± 6% | 0 ± 0% / 67 ± 11% |
| small, 50k (11) | planning | 19 ± 7% / 9 ± 3% | 11 ± 5% / 15 ± 7% | 4 ± 1% / 31 ± 6% | 18 ± 7% / 6 ± 3% |
| | policy only | 4 ± 4% / 6 ± 4% | 4 ± 4% / 1 ± 1% | 0 ± 0% / 17 ± 5% | 2 ± 2% / 2 ± 2% |

**What it means.** Controllers trained from one start do not generalise: even the published-size agents trained for 1M steps succeed in only 20% of the flights from the training area and almost never from the held-out block or outside. The surprise is in the second row. With the policy network alone, the same agents crash far less from the training area (15% instead of 66%), and they succeed more often: planning costs them 26 points of success there (paired difference -26 [-46, -8]). In other words, for these agents, the world model makes things worse away from the start they trained on. For the small 50k agents the picture is different: planning neither clearly helps nor clearly hurts (+14 [-2, +31] points from the training area), because their policy networks alone are imprecise. Section 5.9 investigates why planning hurts the well-trained single-start agents.

### 5.5 Learning to hover from anywhere: the published model (R1)

**The question.** Can TD-MPC2 learn to reach the target from any start in a region, rather than memorising one route? And does what it learns carry over to starts it never trained from: the held-out block inside the region (interpolation) and the starts outside it (extrapolation)? This is the professor's central suggestion and the main experiment of Batch 2.

**How it was tested.** Eleven seeds of TD-MPC2 with the published settings were trained for 1M steps each, every training flight starting at a new random position in the training area (Section 4.1). During training, 21 test flights every 25k steps recorded the learning curves. At the end, every agent flew the 128 fixed test flights (Section 4.3) three times: with planning as trained, with the policy network only, and with a small 64×2 planner, all with the same sensor noise.

**The result.** All 11 runs finished (about 7 hours each). Success / crashes from each group of test starts (mean ± standard error across the 11 seeds):

| Decision rule | Training area | Held-out block | Outside | Batch 1 starts |
|---|---:|---:|---:|---:|
| with planning (as trained) | 46 ± 8% / 0 ± 0% | 39 ± 9% / 2 ± 1% | 20 ± 6% / 26 ± 5% | 39 ± 10% / 0 ± 0% |
| policy network only | 60 ± 11% / 0 ± 0% | 64 ± 13% / 0 ± 0% | 40 ± 10% / 6 ± 2% | 65 ± 11% / 0 ± 0% |
| small 64×2 planner | 58 ± 9% / 0 ± 0% | 46 ± 12% / 0 ± 0% | 32 ± 7% / 9 ± 2% | 57 ± 8% / 0 ± 0% |
| single start, C1 (3 seeds), planning | 26 ± 7% / 35 ± 10% | 4 ± 3% / 82 ± 11% | 0 ± 0% / 92 ± 4% | 41 ± 5% / 52 ± 5% |
| Batch 1 published 1M (3 seeds), planning | 20 ± 6% / 66 ± 3% | 2 ± 2% / 91 ± 9% | 2 ± 1% / 87 ± 5% | 20 ± 7% / 59 ± 4% |

![Success of the final agents with planning, on the same test starts for every experiment (mean ± standard error across seeds). R1 (blue) against its single-start control C1 (orange), the small random-start agents D (green), the randomised-physics pilot H2 (red) and the Batch 1 published 1M agents (purple).](code/src/tdmpc2_training/reports/figures/r1_comparison.png)

Compared with the single-start control C1, random starts raise success with planning by 20 points from the training area [+1, +39], 34 from the held-out block [+17, +51] and 20 from outside [+9, +32], and remove the crashes inside the area (0-2% against 35-82%). The map below shows the same thing start by start.

![Every test start seen from above, coloured by the share of seeds that succeeded from it (green: most seeds; red: none; black ring: most seeds crashed). Top: R1, trained from random starts; bottom: C1, trained from the single Batch 1 start. Left: with planning; right: policy network only. Blue square: training area; orange corner: held-out block; star: target.](code/src/tdmpc2_training/reports/figures/start_outcomes_R1_C1.png)

Three further results were not expected:

- **R1 is much less precise than the small model.** With planning, the small random-start agents (Section 5.7) do better by 32 points from the training area [+5, +57], 44 from the held-out block and 39 from outside. Most R1 seeds end 9 to 19 cm from the target, just outside the 10 cm line (median late error 6.6 to 18.7 cm across seeds).
- **R1 is still learning at 1M steps.** Its snapshots (Section 5.12) rise steadily: average test success from the training area 26% at 600k, 34% at 900k and 42% at 1M, with the typical final error falling from about 20 to 13 cm. The small model, by contrast, reached its level by about 400k steps. This is why seeds 0 to 2 of R1 were continued to 2M steps (Section 5.18).
- **For R1, planning is worse than the policy network alone.** The paired differences are -15 points from the training area [-37, +10], -25 from the held-out block [-47, -1] and -20 from outside [-40, 0]; outside the area, the planner crashes in 26% of the flights and the policy network in 6%. The small 64×2 search is also better than the full 512×6 search (+12 points from the training area and outside). A stronger search doing worse is the signature of model exploitation (Section 5.9).

![The learning curves of R1: test flights every 25k steps during training (left: mean across seeds; right: every seed). Success is still rising at 1M steps in every group.](code/src/tdmpc2_training/reports/figures/r1_learning_curves.png)

**What it means.** The professor's suggestion works for the published TD-MPC2 too: trained from random starts, it no longer crashes inside the area and it carries over to the held-out block. But at 1M steps it is neither converged nor precise, and its own planner holds it back. The policy network alone is the better way to use these agents, and, at the right checkpoint, a very good one (Section 5.12: 90% from the training area with no crashes).

### 5.6 The single-start control (C1)

**The question.** R1 differs from the Batch 1 agents in two ways: random starts, and the target offset given in metres. Which of the two makes the difference? C1 answers this: it is identical to R1 (same settings, same metric offset, same 1M steps) except that every training flight starts at the Batch 1 start (0, 0, 0.2).

**How it was tested.** Three seeds, trained and tested exactly like R1.

**The result.**

| Decision rule | Training area | Held-out block | Outside | Batch 1 starts |
|---|---:|---:|---:|---:|
| with planning (as trained) | 26 ± 7% / 35 ± 10% | 4 ± 3% / 82 ± 11% | 0 ± 0% / 92 ± 4% | 41 ± 5% / 52 ± 5% |
| policy network only | 47 ± 21% / 15 ± 1% | 18 ± 11% / 54 ± 18% | 0 ± 0% / 98 ± 1% | 39 ± 15% / 28 ± 6% |
| small 64×2 planner | 41 ± 12% / 30 ± 13% | 12 ± 11% / 69 ± 16% | 2 ± 1% / 90 ± 5% | 39 ± 14% / 44 ± 3% |

C1 is indistinguishable from the Batch 1 published 1M agents: with planning, the differences are +6 points from the training area [-9, +21], +2 from the held-out block and -2 from outside. It also shows the same pattern: planning loses to the policy network alone from the training area (-21 points) and crashes more, the planner's flights go where the world model is more wrong (paired ratio 1.68 ± 0.35), and restraining the search reduces crashes (Section 5.9).

**What it means.** Giving the target offset in metres changes nothing; the whole improvement of R1 over Batch 1 comes from the random starts. C1 also repeats the Batch 1 result with a new set of seeds and the new tests, which makes that result more solid.

### 5.7 The small model (D)

**The question.** The published model needs a GPU to plan in real time (6.1 ms per decision on a GPU slice, 516 ms on a CPU). The small model plans in 1.4 ms on a GPU slice and 11.4 ms on a CPU, which would fit a laptop next to the drone. Does it learn random starts as well as the published model?

**How it was tested.** Eleven seeds of the small settings (Section 2.4), trained and tested exactly like R1. Because the small model trains faster, each run was split into two jobs of 3 hours, the second resuming from the first's checkpoint. Three seeds were run first as a pilot; the other eight were launched after the pilot had shown at 200k steps that learning worked.

**The result.** All 11 runs finished. Success / crashes from each group of test starts (mean ± standard error across the 11 seeds):

| Decision rule | Training area | Held-out block | Outside | Batch 1 starts |
|---|---:|---:|---:|---:|
| with planning (as trained) | 78 ± 10% / 0 ± 0% | 83 ± 10% / 0 ± 0% | 59 ± 9% / 3 ± 2% | 80 ± 10% / 0 ± 0% |
| policy network only | 9 ± 9% / 0 ± 0% | 9 ± 9% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 11 ± 9% / 0 ± 0% |
| small 64×2 planner | 76 ± 11% / 0 ± 0% | 77 ± 10% / 0 ± 0% | 60 ± 10% / 2 ± 2% | 79 ± 10% / 0 ± 0% |
| Batch 1 small agents, planning | 19 ± 7% / 9 ± 3% | 11 ± 5% / 15 ± 7% | 4 ± 1% / 31 ± 6% | 18 ± 7% / 6 ± 3% |

![The 11 small random-start agents at the end of training. Left: success per group of test starts with planning (blue), with the policy network only (orange) and with the small 64×2 planner (green); grey dots are single seeds. Right: crashes, almost all zero.](code/src/tdmpc2_training/reports/figures/d_final_by_group.png)

Compared with the Batch 1 small agents trained from one start, random starts raise success by 59 points from the training area [+35, +81], 72 points from the held-out block [+48, +91] and 55 points from outside [+38, +72], and remove almost all crashes. Planning beats the policy network alone by 69 points from the training area [+30, +96], 74 from the held-out block [+36, +98] and 48 from outside [+18, +72]. The small 64×2 planner, which needs less than half the time on a CPU (4.4 ms against 11.4 ms per decision), does as well as the full planner (+2 points [-2, +7]).

The seeds differ in one way only. Eight of the eleven succeed in 86-100% of the flights from the training area. Three (seeds 3, 8 and 10) succeed in only 8-54%, but none of them crashes: their typical final distance to the target is 9 to 13 cm, just above the 10 cm line, against 4 to 8 cm for the others.

**What it means.** The small model learns to fly from anywhere in the training area, and it carries this over to the held-out block it never trained in: success and crashes there are the same as in the training area, so interpolation works. Starts up to 0.75 m outside the area are harder (59% success) but still almost never end in a crash. The policy network alone never crashes either, but it rarely ends within 10 cm; the precision comes from planning (Section 5.8). Finally, the spread between seeds that worried the professor in Batch 1 has changed nature: at 1M steps, no seed is unsafe, and what still varies is a few centimetres of precision around the 10 cm threshold (Section 5.12 shows that this also varies within a run, from one checkpoint to the next).

### 5.8 With and without planning: where the precision comes from

**The question.** Sections 5.5 to 5.7 compare planning with the policy network alone in terms of success and crashes. This section looks at the flights themselves: when the drone does not crash, how close to the target does it end, in which direction is it off, and is the conclusion tied to the 10 cm threshold?

**How it was tested.** The script `analyze_precision.py` takes the full flights (no crash) from the training-area and held-out starts and splits each one's late error into its horizontal part and its vertical part, and computes the height offset (above or below the target). It also plots, for every threshold from 0 to 50 cm, the share of flights that end within it, counting crashes as failures. This last figure shows the whole picture at once, without choosing a threshold.

**The result.** Median late error of the full flights, in centimetres (mean ± standard error across seeds):

| Agents | Decision rule | Late error | Horizontal | Vertical | Height offset |
|---|---|---:|---:|---:|---:|
| small, random starts (11) | planning | 7.4 ± 0.9 | 7.1 ± 0.9 | 1.6 ± 0.3 | +1.1 ± 0.4 |
| | policy network only | 20.6 ± 3.0 | 15.1 ± 1.8 | 11.9 ± 3.1 | +7.9 ± 4.2 |
| | 64×2 planner | 7.8 ± 0.9 | 7.5 ± 0.9 | 1.7 ± 0.3 | +1.2 ± 0.4 |
| published, single start, 1M (3) | planning | 10.2 ± 3.2 | 9.8 ± 2.8 | 1.6 ± 0.6 | +0.2 ± 0.4 |
| | policy network only | 9.3 ± 2.4 | 8.9 ± 2.6 | 2.2 ± 0.4 | -1.1 ± 1.1 |

| published, random starts, R1 (11) | planning | 12.7 ± 1.5 | 12.4 ± 1.4 | 1.6 ± 0.3 | +0.4 ± 0.2 |
| | policy network only | 9.3 ± 1.4 | 8.5 ± 1.6 | 2.7 ± 0.4 | -0.2 ± 0.9 |
| | 64×2 planner | 9.9 ± 1.1 | 9.8 ± 1.1 | 1.2 ± 0.2 | +0.1 ± 0.2 |
| published, single start, C1 (3) | planning | 15.1 ± 1.7 | 14.4 ± 1.4 | 2.5 ± 0.4 | +0.2 ± 0.5 |
| | policy network only | 9.3 ± 2.0 | 6.2 ± 1.5 | 5.7 ± 2.0 | -5.5 ± 2.2 |

For the published-size agents the picture is reversed: the policy network alone is the more precise (9.3 cm against 12.7 cm for R1), and the planner's extra error is almost entirely horizontal (12.4 cm against 1.6 cm vertically). The smaller 64×2 search sits in between.

![Share of flights ending within any distance of the target, from 0 to 50 cm, with crashes counted as failures (training-area and held-out starts; mean ± standard error across seeds). The dotted line is the 10 cm used for success. A curve that rises early and reaches 100% means precise and safe; a curve that stays low means crashes.](code/src/tdmpc2_training/reports/figures/thresholds.png)

**What it means.** For the random-start agents, planning cuts the typical final error from about 21 cm to about 7 cm, and it does so in both directions: the policy network alone is off by 15 cm horizontally and 12 cm vertically, the planner by 7 and 2. The policy network alone also holds a wrong height, and the snapshots show that this offset changes from one checkpoint to the next, sometimes above the target and sometimes below, so it is not a fixed bias. A plausible explanation, not tested here, is that in TD-MPC2 the policy network's main job is to propose candidates for the planner: it learns roughly where the good actions are, and the planner's search against the world model's predicted rewards supplies the last few centimetres. The threshold figure shows that the conclusion does not depend on the 10 cm threshold: for the random-start agents, the planner's curve is above the policy network's at every threshold until both reach 100%, which they do because neither crashes. For the single-start agents the planner's curve stops at about 20 to 40%, because the other flights crashed. For R1, the policy network's curve and the 64×2 planner's are above the full planner's at every threshold above about 5 cm: the published model's full search costs precision, not safety.

### 5.9 Why planning hurts single-start agents and helps random-start ones (H1)

**The question.** Section 5.4 showed that planning makes the well-trained single-start agents crash more off their training start. The suspected mechanism is **model exploitation**: the planner searches for the action sequence that the world model rates best, and away from the training data the model is wrong in places; the search is drawn precisely to the actions for which the model is wrong in an optimistic direction, because those look best. The policy network does not search, so it is not drawn to these errors. Two tests checked this explanation.

**Test 1: how wrong is the world model along the flights?** The script `model_error.py` records every test flight and then asks the agent's own world model to predict, from each recorded state, the next three latent states and rewards given the actions that were actually taken. The difference between prediction and reality is the **model error** along that flight. If the planner exploits model errors, its flights should visit states where the model is more wrong than along the policy network's flights from the same starts, and the error should be largest just before crashes.

| Agents | Starts | Planner / policy model error (paired) | Last 0.5 s before a crash vs completed flights |
|---|---|---:|---:|
| published, 1M (3) | training area | 2.8 ± 0.7 | 4.0 times |
| | held-out block | 2.4 ± 0.2 | |
| small, 50k (11) | training area | 1.4 ± 0.3 | 7.1 times |
| published, 50k (11) | training area | 1.3 ± 0.1 | 1.1 times |

![The world model's 3-step prediction error along test flights, with planning (blue) and with the policy network only (orange). For the published 1M agents (left), the planner's flights go where the model is 2-3 times more wrong.](code/src/tdmpc2_training/reports/figures/h1_model_error_batch1.png)

The prediction holds for the agents where planning hurts: the planner's flights go where the model is 2.4 to 2.8 times more wrong, and the error rises sharply in the half second before a crash. The published 50k agents fit the same picture from the other side: they crash almost always, with or without planning, and their model is about equally poor along both kinds of flights.

**Test 2: does keeping the planner closer to the policy reduce crashes?** If exploitation causes the crashes, restricting how far the planner can search away from what the policy network proposes should reduce them. The script `eval_planner_variants.py` flies the same agents, from the same starts and with the same noise, with four decision rules that differ only at test time: the policy network alone, the planner as trained, the planner with a narrower search (`max_plan_std` 0.5 instead of 2), and the planner with half of its 512 candidates proposed by the policy network instead of 24.

| Published 1M agents, training-area starts | Success | Crashes | Change vs planner |
|---|---:|---:|---|
| planner as trained | 24 ± 8% | 63 ± 6% | |
| narrower search | 33 ± 9% | 50 ± 7% | +9 ± 2 / -13 ± 1 points |
| half the candidates from the policy | 39 ± 6% | 45 ± 6% | +15 ± 2 / -19 ± 1 points |
| policy network only | 45 ± 10% | 15 ± 7% | +21 ± 8 / -49 ± 12 points |

All three seeds improved in the same way with both restrictions, which is why the standard errors of the changes are so small. For the small 50k agents, taking half the candidates from the policy reduced crashes by 3 to 7 points depending on the start group, and the narrower search by 0 to 4 points.

**What it means.** Both tests support model exploitation as one real mechanism: the planner's flights go where the model is wrong, crashes follow large model errors, and restraining the search reduces crashes. But restraining the search does not bring the planner down to the policy network's crash rate (45% against 15%), so exploitation is not the whole story. The next suspect is the value function, which judges everything beyond the 3-step horizon and has also only been trained near the single start.

**The same tests when the data covers the situation (Batch 2).** If the explanation is right, agents trained from random starts should show none of this: their world model has seen the situations it is used in. That is exactly what the small random-start agents show (11 seeds):

- Along the planner's flights, their world model is *more* accurate than along the policy network's (paired ratio 0.68 ± 0.09 from the training area, 0.67 ± 0.06 from the held-out block), and its error is 15 to 30 times smaller than that of the Batch 1 small agents.
- Restraining the planner no longer prevents anything; it only costs precision: the narrower search loses 7 ± 6 points of success from the training area, and half the candidates from the policy network 5 ± 5 points.
- The few crashes that remain, from starts outside the area (13 of 330 flights), are again preceded by a sharp rise of the model error (about 35 times its level in completed flights): where the model is wrong, the planner still fails.

**The published model trained on random starts (R1) sits in between.** Its training data covers the area, but its planner still behaves as if it found something to exploit:

| R1 (11 seeds) | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| planner / policy model error (paired) | 1.17 ± 0.07 | 1.49 ± 0.08 | 2.00 ± 0.14 |
| narrower search, change in success / crashes | +10 ± 3 / -1 ± 0 | +11 ± 6 / -3 ± 2 | +8 ± 4 / -11 ± 3 |
| half the candidates from the policy | +13 ± 8 / -1 ± 0 | +8 ± 9 / -3 ± 2 | +12 ± 4 / -16 ± 4 |

The model error along the planner's flights is slightly higher than along the policy network's inside the area and twice as high outside, it rises 4.4 times in the last half second before a crash, and both restraints of the search improve success and reduce the crashes outside (from 26% to 15% and 10%). The world model errors are measured in each agent's own latent space, so their size cannot be compared between the published and the small model; the reward predictions can, and they are about as accurate for R1 as for the small agents inside the area (mean error 0.0005 against 0.0004 per step) but almost three times worse outside (0.0027 against 0.0010).

![The world model's 3-step prediction error along the Batch 2 test flights, with planning (blue) and with the policy network only (orange). Small random-start agents (third panel): the planner's flights are where the model is most accurate. Single-start agents (second and fourth): the planner goes where the model is more wrong. R1 (first): in between, with the largest gap outside the area. Errors are in each agent's own latent space, so compare the two lines within a panel, not across panels.](code/src/tdmpc2_training/reports/figures/model_error_batch2.png)

Why would the published model's planner exploit errors where the small model's does not, on the same data? The tests here do not settle it, but two differences are candidates. The published planner searches much harder (512 candidates refined over 6 rounds, against 128 over 3), and a harder search is more likely to find a spot where the model is optimistically wrong; the 64×2 search doing better than the full one fits this. And the published model had not finished learning (Section 5.5), so its world model may simply be less accurate at 1M steps. The 2M continuation (Section 5.18) supports both: at 2M the full planner beats the policy network inside the area, but the smaller 64×2 search is still much safer outside it.

**What it means, overall.** The world model is not good or bad in itself; what matters is whether it is used where it was trained. Off its data, the planner seeks out the model's mistakes and crashes; on its data, the planner turns an accurate model into precise control. This is the answer to the professor's question of when and why the model helps (Chapter 6, Q7).

### 5.10 Disturbances

**The question.** A real drone meets things that never happened in training: a draft from a fan, a gust from a door, a payload, a weaker battery, radio lag. How do the controllers cope, and does planning with the world model make them more robust than the policy network alone? This is the professor's question about external disturbances.

**How it was tested.** The script `eval_disturbances.py` flies every agent from the 50 training-area starts under each disturbance at several strengths, with planning and with the policy network only, always with the same sensor noise, so that the only difference between conditions is the disturbance:

| Disturbance | Strengths | Real-life equivalent |
|---|---|---|
| steady wind | 1, 2, 3, 4 m/s | a fan or an air-conditioning draft is about 1-3 m/s |
| gust | 2, 4, 6 m/s for 1 s, from 3 s into the flight | a door opening, a person walking past |
| extra mass and inertia | +5, +10, +15% | an expansion deck, a marker, a small camera |
| weaker motors | -5, -10% thrust and moments | battery voltage dropping, worn propellers |
| command delay | 1 or 2 control steps (20 or 40 ms) | radio and processing lag |

The wind acts through the simulator's air model, so it pushes on the drone the way moving air would. Hovering already needs 76% of the maximum thrust, which is why the extra mass and the motor weakening stay small: with 30% more mass, the drone would need 99% of its maximum thrust just to stay in the air. For each crashed flight the test also records how it ended: hitting the ground, or being carried out of bounds.

**The result.** Success with planning / with the policy network only, from the 50 training-area starts (mean across seeds):

| Condition | Small, random starts (D, 11) | Published, random starts (R1, 11) | Published, single start (C1, 3) |
|---|---:|---:|---:|
| none | 77% / 9% | 43% / 60% | 29% / 45% |
| wind 1 m/s | 60% / 18% | 36% / 63% | 38% / 33% |
| wind 2 m/s | 1% / 0% | 1% / 1% | 0% / 0% |
| gust 2 m/s | 66% / 9% | 37% / 57% | 23% / 40% |
| gust 4 m/s | 15% / 0% | 7% / 23% | 1% / 1% |
| mass ×1.05 / ×1.10 / ×1.15 | 76 / 74 / 72% against 9-10% | 32 / 20 / 15% against 57 / 54 / 51% | 23 / 21 / 15% against 44 / 43 / 41% |
| motors ×0.95 / ×0.90 | 76 / 73% against 9 / 10% | 33 / 25% against 57 / 54% | 24 / 23% against 44 / 41% |
| delay 20 / 40 ms | 77 / 76% against 9% | 33 / 25% against 60% | 28 / 17% against 46 / 44% |

Crashes, with planning / policy network only:

| Condition | D | R1 | C1 |
|---|---:|---:|---:|
| none | 0% / 0% | 0% / 0% | 39% / 16% |
| wind 2 m/s | 0% / 0% | 4% / 0% | 48% / 27% |
| wind 3 m/s | 1% / 0% | 32% / 0% | 94% / 87% |
| wind 4 m/s | 41% / 29% | 87% / 37% | 100% / 99% |
| gust 6 m/s | 67% / 21% | 80% / 67% | 100% / 100% |
| mass ×1.15 | 2% / 0% | 3% / 0% | 62% / 19% |

![Success under each disturbance, from the 50 training-area starts (mean ± standard error across seeds). Solid lines: with planning; dashed: policy network only. Green: small random-start agents (D); blue: R1; orange: C1; red: the randomised-physics pilot H2 (Section 5.11); purple: Batch 1 published 1M.](code/src/tdmpc2_training/reports/figures/disturbances_main.png)

**What it means.**

- **Mass, motors and delay are easy for the small random-start agents.** With planning they keep 72-77% success under every change of mass, motor strength and command delay tested, 61-68 points above their policy network alone. These are the disturbances a real Crazyflie meets every day (a deck, a tired battery, the radio), so this is good news.
- **The published agents lose more.** R1's planner drops from 43% to 15-33% under the same changes, while its policy network barely moves (51-60%). The planner of the published model is the fragile part, consistent with Section 5.9.
- **Wind is the real limit.** At 1 m/s the small agents still succeed in 60% of the flights; from 2 m/s on, no agent stays within 10 cm, because none of them has ever felt wind and none can tell from one observation that it is being pushed. Up to 3 m/s the small agents are pushed off target but stay in the air (at most 1% crashes); the published planner already crashes in a third of the flights at 3 m/s, its policy network in none.
- **Beyond the limit, planning crashes more.** At 4 m/s wind and in 6 m/s gusts, every agent's planner crashes more often than its policy network (for D: 41% against 29%, and 67% against 21%). Strong wind puts the drone in states the world model never saw, which is exactly where Section 5.9 says the planner goes wrong.
- **Single-start agents** crash often even without disturbance, and every disturbance makes it worse.

So, to the professor's question: planning makes the small random-start agents more robust than their policy network to every moderate disturbance, because it keeps their precision; it does not protect any agent against strong wind, and there it makes crashes more likely.

### 5.11 Training on varied physics (H2)

**The question.** Section 5.9 suggests that planning hurts when the world model is used outside the data it was trained on. The opposite hypothesis, called **H2** here, is that planning should help when the world model has been trained on the relevant variation. Disturbances offer a direct test: if the training flights already contain varied mass, motor strength and wind, the world model learns how these change the drone's motion, and planning with it might then cope better with them than the policy network alone.

**How it was tested.** Three seeds of the small model were trained like D, except that every training flight also drew, at random, a mass and inertia between ×1.0 and ×1.15, a motor strength between ×0.9 and ×1.0, and a steady wind of 0 to 3 m/s from a random direction (`crazyflie_hover_dr.py`). The agent is never told these values; it can only infer them from how the drone moves. This is known as **domain randomisation** (Peng et al., 2018). The agents were then tested with the same disturbance tests as D. Disturbances within these ranges test interpolation; 4 m/s wind, 6 m/s gusts and command delays lie outside them. With 3 seeds this is a pilot.

**The result.** Without disturbance, from the final test starts (success / crashes):

| Agents | Decision rule | Training area | Held-out block | Outside | Late error of full flights |
|---|---|---:|---:|---:|---:|
| varied physics, H2 (3) | planning | 13 ± 7% / 0% | 7 ± 4% / 0% | 4 ± 4% / 0% | 14.3 ± 2.0 cm |
| | policy network only | 34 ± 31% / 0% | 34 ± 33% / 0% | 27 ± 23% / 1% | 12.5 ± 2.9 cm |
| nominal physics, D (11) | planning | 78 ± 10% / 0% | 83 ± 10% / 0% | 59 ± 9% / 3% | 7.4 ± 0.9 cm |

Under disturbances, crashes with planning / late error of the full flights (training-area starts):

| Condition | H2 | D |
|---|---:|---:|
| none | 0% / 16 cm | 0% / 7.5 cm |
| wind 2 m/s | 0% / 22 cm | 0% / 25 cm |
| wind 3 m/s | 0% / 47 cm | 1% / 67 cm |
| wind 4 m/s | 34% / 162 cm | 41% / 169 cm |
| gust 6 m/s | 55% / 76 cm | 67% / 82 cm |

![Success under each disturbance for the varied-physics pilot H2 (blue) and the nominal small agents D (orange); solid with planning, dashed policy network only.](code/src/tdmpc2_training/reports/figures/h2_disturbances.png)

**What it means.** The hypothesis is not supported by this pilot. Training on varied physics bought a little robustness in strong wind (fewer crashes at 4 m/s and in 6 m/s gusts, and the drone held 20 cm closer in 3 m/s wind), but it cost most of the precision in calm air: 13% success against 78%. And the planner does not beat the policy network for these agents (-21 points, with a very wide interval over 3 seeds), so the world model trained on varied physics is not used to advantage either.

A likely reason is in the observation. The agent sees only its current state and its previous command, not a history. In a given flight it cannot tell whether it is heavy, weak or pushed by wind, so its world model can only learn the average effect over all the variations it was trained on, which is wrong for any particular flight; the planner then plans with a model that is wrong in every flight, and the precision suffers. This is a known requirement of domain randomisation: the controller must be able to infer the hidden conditions, usually from a short history of observations, or be given an estimate of them. Testing this would need a change of the observation (for example the last few observations and commands), not only of the physics.

### 5.12 Snapshots: change during training, and the best versus the final checkpoint

**The question.** The learning curves of Batch 1 and of Batch 2 swing from one test point to the next. Two questions follow. Are these swings real changes of the agent, or noise of the test flights? And would it be better to keep the best checkpoint of a run than the last one?

**How it was tested.** Every Batch 2 run kept its agent every 100k steps (10 snapshots). The script `eval_snapshots.py` flies every snapshot from the 40 validation starts and the 110 test starts, with planning and with the policy network only, always with the same sensor noise, so that differences between snapshots can only come from the agents. The best snapshot of each run is chosen on the validation starts alone; its test results are then compared with those of the final snapshot. Choosing on separate validation starts matters: choosing on the test starts would make the "best" snapshot look better than it really is.

**The result for the small agents (11 seeds).**

- **The swings are real.** From 500k steps on, the success of a seed from the training-area starts changes from one snapshot to the next with a standard deviation of 0.34 ± 0.03, while the noise of 50 test flights alone would give about 0.04 to 0.07. Since every snapshot flies the same starts with the same sensor noise, these are changes of the agent itself.
- **What swings is mostly precision.** From one snapshot to the next, the typical final error of the full flights moves between about 4 and 15 cm, sometimes 20 to 60 cm. Crashes are rare but not absent: one seed's 500k snapshot crashed in 44% of the flights from the training area, and the next one did not.
- **The best snapshot beats the final one.** Choosing each run's snapshot on the 40 validation starts alone gives an average test success of 97% from the training area, against 76% for the final snapshots (+21 ± 11 points; +9 ± 10 from the held-out block, +7 ± 8 from outside). It rescues exactly the weak seeds: seed 8 goes from 14% to 100% and seed 10 from 2% to 100%.

![Every 100k-step snapshot of every run, with planning, flown from the same starts with the same sensor noise (thin lines: one seed each; thick line: mean). Rows: R1, C1, the small agents D and the reward pilot (Section 5.13). The first four panels of a row show success per group of starts, the last one the typical final error from the training area, with the 10 cm threshold dotted. R1 rises steadily; D swings from snapshot to snapshot.](code/src/tdmpc2_training/reports/figures/snapshots.png)

**The result for the published agents.** The same test for R1 and C1, with planning and with the policy network alone. Test success at the validation-chosen snapshot and at the final one (training area / held-out block / outside, mean ± standard error across seeds):

| Agents | Decision rule | Best snapshot (validation) | Final snapshot |
|---|---|---|---|
| small, random starts, D (11) | planning | 97 ± 2% / 91 ± 8% / 70 ± 6% | 76 ± 11% / 82 ± 10% / 62 ± 9% |
| | policy network only | 47 ± 14% / 48 ± 15% / 37 ± 11% | 9 ± 9% / 9 ± 9% / 11 ± 8% |
| published, random starts, R1 (11) | planning | 49 ± 7% / 46 ± 7% / 26 ± 5% | 42 ± 8% / 39 ± 9% / 24 ± 6% |
| | **policy network only** | **90 ± 6% / 81 ± 11% / 59 ± 8%** | 60 ± 11% / 62 ± 13% / 39 ± 10% |
| published, single start, C1 (3) | planning | 28 ± 2% / 17 ± 12% / 0 ± 0% | 29 ± 5% / 7 ± 7% / 1 ± 1% |
| | policy network only | 59 ± 14% / 35 ± 14% / 3 ± 3% | 45 ± 22% / 18 ± 11% / 0 ± 0% |

(The final-snapshot numbers differ by a few points from Sections 5.5 to 5.7 because the snapshot test uses its own sensor-noise seed and leaves out the Batch 1 starts.)

- **R1's planner is not drifting; it is still learning.** Its snapshots improve almost monotonically (the best snapshot is the final one for 6 of the 11 seeds), and choosing on validation gains only 7 ± 4 points.
- **R1's policy network alone, at its validation-chosen snapshot, is one of the two best controllers of the project:** 90% success from the training area and 81% from the held-out block, with no crash inside the area (6% outside) and a typical final error of 7.2 cm. Its success also drifts between snapshots (within-run standard deviation 0.24 ± 0.04 from 500k on), so choosing the snapshot matters (+30 ± 9 points over the final one).
- **For C1** choosing a snapshot changes little: single-start agents are limited by their crashes, not by drift.

![The same snapshots flown with the policy network only. For R1 (first row) the policy network alone reaches 80-100% success at several snapshots of most seeds.](code/src/tdmpc2_training/reports/figures/snapshots_policy_only.png)

**What it means.** Training longer does not make an agent settle: even after 500k steps, its precision keeps moving by several centimetres from one checkpoint to the next, while it stays safe. Because success is a 10 cm threshold and the typical error sits close to it, these small moves turn into large jumps of success, which is also why the final results of Section 5.7 differ so much between seeds. The cure that works now is simple: keep snapshots, test them on validation starts, and keep the best. With it, the two best controllers of the project are the small model with planning (97% from the training area) and the published model's policy network alone (90%). Whether a sharper reward could cure the drift at the root is the subject of Section 5.13.

### 5.13 A sharper reward near the target (pilot)

**The question.** Section 5.12 shows that the precision of an agent drifts between checkpoints, while it never crashes. A look at the reward suggests why. The lab's reward per step is 0.001 for staying in the air, plus 0.01 × (1 - tanh d) for being at a distance d from the target, plus the reduction of the distance since the previous step, and -1 for a crash. Hovering at 5 cm from the target earns 0.0105 per step; hovering at 15 cm earns 0.0095. The difference, 0.001 per step, is about a tenth of the value of hovering at all, and in TD-MPC2's value encoding (101 bins over a symmetric logarithmic range from -10 to 10, so bins of width 0.2) it is a quarter of a single bin. Within the 3-step planning horizon, the two rewards differ by only 0.003. The agent therefore has very little reason to prefer 5 cm over 15 cm, which could explain why its precision wanders. This is a hypothesis; the experiment below tests it.

**A direct look at the value network.** The script `probe_value.py` asks a trained agent how good it thinks it is to hover still at points around the target: the drone is placed at rest on a grid of points within 30 cm of the target, its exact observation is built, and the agent's value networks give their estimate. Nine snapshots were probed, four precise ones (3-7 cm in flight) and five imprecise ones (10-56 cm). In all nine, the value is almost flat: it changes by less than 0.005 within 10 cm of the target and by about 0.01 within 20 cm, on a value of about 1.1. For comparison, a value that followed the rewards closely would change by about 0.1 per 10 cm near the target, because a drone that is still 10 cm away can collect 0.1 of distance-improvement reward by flying in. In other words, the value network does not see the difference between 0 and 20 cm. The precise and the imprecise snapshots look the same, so this does not explain which checkpoints drift; it explains where the precision comes from. Since the value is flat, the precision must come from the rewards that the world model predicts over the planning horizon. This fits three earlier results: the policy network, trained against the flat value, is imprecise (Section 5.8); planning only one step ahead, which leans almost entirely on the value, is much worse (Section 5.14); and five steps is slightly better than three.

**How it was tested.** Three seeds of the small model were trained exactly like D, with one change: an extra reward term concentrated near the target, 0.01 × exp(-d² / (2 × 0.05²)), worth 0.01 at the target, 0.006 at 5 cm and almost nothing beyond 15 cm (`crazyflie_hover_sharp.py`). Hovering at 2 cm instead of 15 cm now earns about 0.010 more per step instead of 0.001. Physics, observations, starts and tests are unchanged, and the test flights measure distances, so the results compare directly with D. The comparison of interest is the drift between snapshots and the final precision. This is a change of the task's reward, not of TD-MPC2, and with 3 seeds it is a pilot.

**The result.**

| Small agents, with planning | Sharp reward (3 seeds) | Lab reward, D (11 seeds) |
|---|---:|---:|
| final snapshot: training area / held-out / outside | 37 ± 19% / 66 ± 33% / 23 ± 17% | 78 ± 10% / 83 ± 10% / 59 ± 9% |
| best snapshot (validation) | 91 ± 5% / 78 ± 19% / 47 ± 3% | 97 ± 2% / 91 ± 8% / 70 ± 6% |
| crashes, final, training area / outside | 0% / 2% | 0% / 3% |
| within-run SD of training-area success, from 500k | 0.25 ± 0.04 | 0.34 ± 0.03 |
| typical final error per snapshot, from 500k (mean of the seeds' medians) | 9.3 cm | about 10 cm |
| snapshots with a typical final error above 20 cm, from 500k | 0 of 18 | 9 of 66 |
| value drop from the target to 10 cm / 20 cm (probe) | 0.03-0.05 / 0.07-0.13 | about 0.00 / 0.01 |

The probe gives the clearest answer: with the sharp reward, the value network does see the target. Its value is highest within 2-3 cm of the target in three of the six probed snapshots, and it drops by 0.03 to 0.05 at 10 cm and 0.07 to 0.13 at 20 cm, against practically nothing for the lab's reward. The agents learned the extra reward.

Yet the precision still drifts. Success still swings from snapshot to snapshot four times more than test noise allows (0.25 against about 0.06), and the typical final error per snapshot is about the same as with the lab's reward (9.3 against about 10 cm). What disappeared are the large excursions: none of the 18 snapshots from 500k on had a typical error above 20 cm, against 9 of 66 for D (a difference that, with 3 seeds, is suggestive rather than proven). The final snapshots happened to be weak (one seed's final snapshot succeeded in 0% of the flights from the training area while its 400k snapshot succeeded in 84%), which is why the final results look worse than D's; at the validation-chosen snapshots the two are close (91% against 97%).

![Success of the final agents with planning: the sharp-reward pilot against the small agents with the lab's reward.](code/src/tdmpc2_training/reports/figures/sharp_comparison.png)

**What it means.** The hypothesis is rejected in its simple form: making precision worth ten times more did change what the agent values, but it did not stop the precision from drifting around the 10 cm line. The flat value function is therefore not the cause of the drift. What remains as candidates is the planner itself (each decision is the best of a random sample, so the hover position wanders with the sampling) and the continual updates of the world model and policy, which keep moving the agent's behaviour by a few centimetres even after it has learned the task. The practical conclusion does not change: choose checkpoints on validation starts. The sharp reward may be worth keeping for its smaller excursions, but this needs more seeds before it is a finding.

### 5.14 The planning horizon

**The question.** TD-MPC2 plans only 3 steps (0.06 s) ahead and leaves everything after that to its value network. Would looking further ahead help? Or would the world model's errors pile up over a longer horizon, since it was trained to predict only 3 steps?

**How it was tested.** The script `eval_horizon.py` flies the same agents with planning horizons of 1, 3 (as trained), 5 and 10 steps, from the 50 training-area and 30 outside starts, without disturbance, in a 3 m/s steady wind and with 15% extra mass, always with the same sensor noise. Only the horizon used at test time changes; the agents are not retrained.

**The result for the small agents (11 seeds).** Success / crashes, with the paired change in success relative to the trained horizon of 3:

| Horizon | Training area | Outside | Training area, +15% mass |
|---:|---:|---:|---:|
| 1 step | 43 ± 14% / 0% (-35 ± 13) | 31 ± 10% / 4% (-31 ± 9) | 43 ± 14% / 1% (-29 ± 12) |
| 3 steps (as trained) | 78 ± 10% / 0% | 62 ± 10% / 4% | 72 ± 12% / 2% |
| 5 steps | 89 ± 8% / 0% (+11 ± 7) | 69 ± 9% / 5% (+8 ± 4) | 83 ± 9% / 2% (+11 ± 6) |
| 10 steps | 72 ± 9% / 0% (-6 ± 11) | 49 ± 10% / 11% (-12 ± 11) | 59 ± 9% / 3% (-14 ± 13) |

In a 3 m/s steady wind, no horizon gives any success, and the crashes from outside starts grow with the horizon (7% at 1 or 3 steps, 18% at 10).

**The result for the published random-start agents (R1, 11 seeds).**

| Horizon | Training area | Outside | Training area, 3 m/s wind (crashes) |
|---:|---:|---:|---:|
| 1 step | 37 ± 10% / 0% (-5 ± 6) | 27 ± 6% / 7% (+8 ± 6) | 11% |
| 3 steps (as trained) | 43 ± 8% / 0% | 19 ± 6% / 25% | 32% |
| 5 steps | 43 ± 10% / 1% (-0 ± 6) | 19 ± 4% / 32% (+0 ± 4) | 41% |
| 10 steps | 29 ± 8% / 4% (-13 ± 7) | 8 ± 3% / 44% (-12 ± 5) | 49% |

For R1, looking further ahead never helps, and every extra step of horizon adds crashes, outside the area and in wind: from 1 to 10 steps, crashes from outside starts grow from 7% to 44% and in 3 m/s wind from 11% to 49%. A horizon of 1 step, which relies mostly on the value network and least on the world model, is as good as 3 inside the area and crashes least everywhere. This is again what model exploitation predicts: the further the planner searches through an imperfect model, the more model errors it can find.

![Success against the planning horizon used at test time, without disturbance (blue), in 3 m/s wind (red) and with 15% extra mass (orange), for R1, C1, the small agents D and H2. Left: training-area starts; right: outside starts. Dotted line: the trained horizon of 3.](code/src/tdmpc2_training/reports/figures/horizon.png)

**What it means.** Looking only one step ahead is clearly worse, probably because the planner then relies almost entirely on the value network, which judges a situation less precisely than a few steps of the world model's predictions do. Five steps is slightly better than the trained three (by about 10 points, but only about 1.5 standard errors, so this is a hint rather than a finding), and ten steps is no better in the training area while it doubles to triples the crashes from outside starts, where the world model's errors pile up over the longer horizon. For the published R1 agents, whose planner already shows signs of exploitation (Section 5.9), longer horizons only add crashes, and for the single-start C1 agents no horizon is clearly better. The horizon that helps is therefore limited by how accurate the world model is where it is used: the small random-start agents can look a little further ahead, the others cannot. For practical use, the trained horizon of 3 is a sound choice, and 5 could be tried for the small model.

### 5.15 How the drone flies: settling, smoothness and the worst flights

**The question.** Success and crashes say where a flight ends, not how it got there. For a real drone, three more things matter: how long it takes to settle near the target, how smooth its commands are (motors cannot change their speed instantly, and jumpy commands heat them and shake the frame), and how bad the worst flights are. The mission for this project asked for all three; none of them needed new flights, because every test already saved every step of every flight (`trajectories.csv`).

**How it was measured.** The script `analyze_flight_quality.py` reads the step-by-step records of the final tests from the training-area and held-out starts and computes, for each agent:

- the **settling time**: the moment after which the drone stays within 10 cm of the target until the end of the flight;
- the **command jitter**: how much each command changes from one step to the next during the last 5 seconds of a full flight, when the drone should simply be holding its position, in percent of the command's full range;
- the **saturation**: the share of steps at which a command is at its limit (no thrust, full thrust, or a body moment at its maximum);
- the **tail**: the 90th percentile and the worst late error of the full flights.

**The result.** For the 11 small random-start agents (training-area and held-out starts):

| Decision rule | Within 20 cm after | Stays within 10 cm after | Thrust jitter | Moment jitter | Steps at a limit | 90th percentile of late error |
|---|---:|---:|---:|---:|---:|---:|
| with planning | 1.75 s | 6.7 s (79% of flights) | 29% | 33% | 49% | 10 cm |
| policy network only | 2.15 s | 8.2 s (10% of flights) | 10% | 8% | 2% | 22 cm |
| 64×2 planner | 1.75 s | 6.6 s (75% of flights) | 32% | 37% | 65% | 11 cm |

For the published agents (training-area and held-out starts):

| Agents | Decision rule | Within 20 cm after | Stays within 10 cm after | Thrust jitter | Moment jitter | Steps at a limit | 90th percentile of late error | Worst late error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1 (11) | with planning | 1.80 s | 8.6 s (48% of flights) | 23% | 22% | 46% | 21 cm | 51 cm |
| | policy network only | 1.73 s | 5.3 s (57% of flights) | 9.6% | 6.0% | 1% | 14 cm | 18 cm |
| | 64×2 planner | 1.72 s | 8.1 s (57% of flights) | 25% | 28% | 54% | 15 cm | 21 cm |
| C1 (3) | with planning | 2.27 s | 7.1 s (44% of flights) | 23% | 22% | 43% | 50 cm | 173 cm |
| | policy network only | 2.05 s | 5.9 s (45% of flights) | 11% | 7.6% | 12% | 12 cm | 26 cm |

The published planner is as jumpy as the small one, and its worst flights are worse than its policy network's (a worst late error of 51 against 18 cm). R1's policy network alone settles faster than any planner (5.3 s) and has the tightest tail of the whole project (90% of its flights end within 14 cm).

**What it means.** The drone gets close quickly: within 20 cm in under 2 seconds, with or without planning. The last few centimetres take much longer, because the error hovers around the 10 cm line before settling, which is the same precision issue seen in Sections 5.7 and 5.12. The striking number is the jitter. In steady hover, the planner changes its thrust command by 29% of its full range from one step to the next, and half of all its commands sit at a limit (zero thrust, full thrust, or a maximal moment). The policy network alone is three to four times smoother. The tests use the planner exactly as it should be used for testing (the best candidate, no added noise), so this jumpiness is not a testing artefact; Section 5.16 explains where it comes from and tests whether it can be avoided.

### 5.16 Smoother commands and slower motors

**The question.** Section 5.15 shows that the planner's commands jump from step to step. Why, and does it matter? The reason is in how TD-MPC2 acts. At each step, its planner samples many candidate plans, scores them with the world model, and executes the first action of the single best one. Because each step's best candidate is a new random sample, and because many samples are clipped at the command limits, the executed action jumps around even when the drone is hovering calmly. The planner also computes a smoother quantity, the weighted average of its best candidates (its **averaged plan**), but only uses it to start the next step's search. Two questions follow. Does the planner need its jumpy actions to be precise, or would a smoother way of acting do as well? And does the precision survive if the motors cannot follow the commands instantly, as real motors cannot?

**How it was tested.** The script `eval_smoothing.py` flies the same agents, from the same 80 training-area and held-out starts with the same sensor noise, in four ways: the planner as trained; the first action of the planner's averaged plan; the planner's commands passed through a low-pass filter with a time constant of 40 ms (each command moves only part of the way towards the new one); and the policy network alone. Each is flown with instant motors, as in training, and with motors that follow the command with a first-order lag of 30 or 60 ms. These lags were chosen to test sensitivity; the real Crazyflie's motor response has not been measured in this project, so they are assumptions, not identified values. Nothing is retrained; only the way the trained agent acts, or the simulated motors, change.

Before the test was run, it was checked against the existing test code: with the policy network alone, its flights are identical to the disturbance tester's, step for step; with the planner as trained they are statistically identical (same success and crash rates over four noise seeds) but not bit-identical, because the two programs are compiled differently and round the last digit of some numbers differently, which the planner's choice between near-equal candidates then amplifies. Within the smoothing test, all variants run in the same compiled program, so their comparisons are exactly paired.

**The result for the small agents (11 seeds).** Success / crashes from the training-area and held-out starts, with the paired change in success relative to the planner as trained at the same motor lag, the thrust jitter and the share of steps at a limit:

| Motor lag | Decision rule | Success / crashes | Change vs planner | Thrust jitter | At a limit |
|---|---|---:|---:|---:|---:|
| none | planner as trained | 79 ± 10% / 0% | | 29% | 49% |
| none | averaged plan | 80 ± 10% / 0% | +1 ± 1 | 23% | 1% |
| none | planner, 40 ms filter | 79 ± 10% / 0% | +0 ± 1 | 8.5% | 2% |
| none | policy network only | 9 ± 9% / 0% | -69 ± 18 | 10% | 2% |
| 30 ms | planner as trained | 79 ± 10% / 0% | | 25% | 50% |
| 30 ms | planner, 40 ms filter | 76 ± 11% / 1% | -2 ± 1 | 8.1% | 3% |
| 60 ms | planner as trained | 77 ± 11% / 1% | | 23% | 52% |
| 60 ms | averaged plan | 78 ± 10% / 1% | +1 ± 1 | 18% | 3% |
| 60 ms | planner, 40 ms filter | 66 ± 12% / 3% | -11 ± 4 | 6.6% | 15% |

**The result for the published random-start agents (R1, 11 seeds).**

| Motor lag | Decision rule | Success / crashes | Change vs planner | Thrust jitter | At a limit |
|---|---|---:|---:|---:|---:|
| none | planner as trained | 40 ± 8% / 1% | | 23% | 46% |
| none | averaged plan | 43 ± 8% / 1% | +2 ± 2 | 17% | 1% |
| none | planner, 40 ms filter | 29 ± 6% / 3% | -11 ± 3 | 6.6% | 6% |
| none | policy network only | 61 ± 12% / 0% | +21 ± 11 | 9.6% | 1% |
| 30 ms | planner as trained | 32 ± 7% / 2% | | 19% | 47% |
| 30 ms | policy network only | 61 ± 11% / 0% | +29 ± 12 | 8.4% | 1% |
| 60 ms | planner as trained | 21 ± 6% / 6% | | 17% | 52% |
| 60 ms | averaged plan | 22 ± 6% / 5% | +1 ± 2 | 11% | 3% |
| 60 ms | planner, 40 ms filter | 8 ± 3% / 22% | -14 ± 4 | 5.6% | 15% |
| 60 ms | policy network only | 59 ± 11% / 0% | +37 ± 12 | 7.9% | 2% |

![Success against the motor lag for each way of acting (training-area and held-out starts). Left: R1; right: the small agents D. The published planner loses precision with slower motors and with the filter; the small planner keeps it; R1's policy network alone is unaffected.](code/src/tdmpc2_training/reports/figures/smoothing.png)

The published model behaves differently: its planner is sensitive to both the filter and the motor lag (a 60 ms lag halves its success and brings 6% crashes), while its policy network alone does not notice the lag at all (61, 61 and 59%). Acting on the averaged plan is the only change that is free for both models.

**What it means.** The small model's planner does not need its jumpy commands. Passing them through a 40 ms filter keeps exactly the same precision (+0 ± 1 points) while making the commands as smooth as the policy network's and removing almost all saturation; acting on the averaged plan also keeps the precision and removes the saturation, though the commands still change more from step to step. The agents also tolerate slow motors on their own: a 30 or 60 ms motor lag costs the planner at most 2 points. What does not work is stacking the two lags: the filter on top of 60 ms motors costs 11 points, because the delays add up. The practical rule is that the command filter should be sized together with the real motors, once their response has been measured. For the published model the filter is not free (-11 points), and acting on the averaged plan is the safer choice; its policy network alone needs no smoothing at all. This is the most useful deployment result of the night: a simple, cheap change at decision time removes a problem that would otherwise make the planner unsuitable for real motors.

### 5.17 Speed: decision time, training time and parallel drones

**Decision time.** A real controller must decide within the 20 ms control period. The script `bench_latency.py` timed 1,000 decisions (300 on the CPU) for each network size and planning budget, on a Narval GPU slice and on 2 CPU cores of the same machine (median times):

| Network | Decision rule | GPU slice | 2 CPU cores |
|---|---|---:|---:|
| published (5.0M weights) | planner 512×6 | 6.1 ms | 516 ms |
| published | planner 64×2 | 1.9 ms | 28 ms |
| small (0.39M weights) | planner 128×3 | 1.4 ms | 11.4 ms |
| small | planner 64×2 | 1.5 ms | 4.4 ms |
| either | policy network only | 0.3-0.4 ms | 0.2 ms |

The published planner fits the 20 ms budget easily on a GPU but is about 25 times too slow on a CPU; the small planner fits on a CPU with some margin; the policy network alone is negligible anywhere. On a GPU, the time grows much more slowly than the amount of arithmetic: the 512×6 planner of the published model computes 13 times more than its 64×2 planner but takes only about 3 times longer. Most of the time goes into launching many small operations one after the other, not into computing.

**Training time.** The script `profile_training.py` measured where one training step goes on a Narval GPU slice. For the published model: simulating the one drone takes 6.8 ms, planning its action 10.2 ms, and the learning update 2.6 ms, which together give 49 steps per second, or about 5.6 hours of pure training for 1M steps (the real runs took about 7 hours, with the test flights every 25k steps and the checkpoints). For the small model: 6.6, 1.4 and 2.7 ms, 88 steps per second. The GPU is mostly waiting: each operation is small (one drone), so the time goes into starting operations rather than computing. This answers the professor's question about GPU use: TD-MPC2 with one drone cannot keep a GPU busy (at home the GPU was busy only 25-28% of the time; on Narval's slices the usage counter is not available).

**Parallel drones (E).** PPO is fast in real time because it simulates 2,048 drones at once. Could TD-MPC2 do the same? The script `bench_parallel.py` timed simulation, planning and updates with 1 to 128 drones and projected the training speed:

![Projected training speed of the published TD-MPC2 with many drones on a Narval GPU slice. Simulation is free (the same 6 ms for 1 or 128 drones), but planning grows with the number of drones, and one learning update per collected step caps the speed.](code/src/tdmpc2_training/reports/figures/e_parallel_published.png)

| Drones | Published, 1 update per step | Small, 1 update per step | Small, 1 update per 2 steps |
|---:|---:|---:|---:|
| 1 | 45 steps/s | 93 steps/s | 111 steps/s |
| 8 | 68 | 223 | 368 |
| 128 | 74 | 271 | 517 |

The 1-drone projections match the measured training speeds, so the projection can be trusted. Simulation parallelises perfectly, but TD-MPC2 must still plan for every drone and, by default, make one learning update for every step collected. Parallel drones therefore speed it up by only 1.6 times (published) to 2.9 times (small). Going further requires fewer updates per step, which changes the algorithm and would have to be checked for its effect on learning.

### 5.18 Does the published model just need more training? (R1 continued to 2M steps)

**The question.** Sections 5.5 and 5.12 show that R1 was still improving at 1M steps, while the small model had levelled off by about 400k. Is the published model simply slower to learn, so that with more steps it would catch up with (or pass) the small one? This matters for the professor's question about model size: a larger model that needs twice the training is a different trade-off from a larger model that is worse.

**How it is tested.** Seeds 0, 1 and 2 of R1 were copied after their 1M-step training, with their replay buffer, and continued to 2M steps in three pieces of 3 hours (`cluster/copy_run_job.sh`, then `seed_job.sh` resuming), keeping snapshots every 100k steps. At 2M they fly the same final tests as every other agent. Nothing else changes, so the 1M and 2M agents of each seed can be compared directly.

**The result.** All three runs finished at about 12:00 on November 2 (jobs 7669542 to 7669544, every task exit 0). The fair comparison is paired: the same three seeds at 1M and at 2M steps. Success / crashes (mean over seeds 0-2):

| Seeds 0-2 of R1 | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| planning, 1M | 66% / 0% | 65% / 2% | 33% / 22% |
| planning, 2M | 72% / 3% | 75% / 7% | 24% / 58% |
| 64×2 planner, 1M | 51% / 0% | 33% / 0% | 39% / 6% |
| 64×2 planner, 2M | 70% / 0% | 70% / 0% | 57% / 22% |
| policy network only, 1M | 36% / 0% | 53% / 0% | 20% / 3% |
| policy network only, 2M | 29% / 0% | 30% / 0% | 18% / 34% |
| small model, 1M (all 11 seeds), planning | 78% / 0% | 83% / 0% | 59% / 3% |

Per seed, with planning from the training area: seed 0 went from 36% to 78% (typical final error 12.3 to 7.2 cm), seed 1 from 82% to 48% (8.8 to 10.3 cm), seed 2 from 80% to 90% (7.7 to 3.1 cm, the most precise agent of the project).

![Success of the final agents with planning: R1 seeds 0-2 at 2M steps against all R1 seeds at 1M and the small model at 1M.](code/src/tdmpc2_training/reports/figures/r2_comparison.png)

**What it means.**

- **Inside the area, the second million helped, but less than the first.** With planning, success rose by 6 points from the training area and 10 from the held-out block, and precision improved (two of the three seeds now end 3-7 cm from the target). The first million took the same seeds from crashing in most flights to 66%; the second added a few points. The learning curves of these runs show the same: about 55% at 1M and 65% at 1.6-1.8M from the training area.
- **Outside the area, it got worse.** Crashes from the outside starts rose from 22% to 58% with planning and from 3% to 34% with the policy network alone. With more training on the same area, the agents fly better inside it and recover less well from beyond it: they specialise. This is the most important result of the continuation, and with 3 seeds it is an indication, not a firm finding.
- **A likely reason: the replay buffer starts forgetting at 1M steps.** TD-MPC2 learns from a replay buffer that holds the last 1,000,000 transitions (`buffer_size` in the configuration). Up to 1M steps nothing is ever removed, so the buffer still contains the clumsy early flights, in which the drone drifted far from the target and had to recover from unusual states. Between 1M and 2M, every new transition replaces the oldest one; by 2M the early flights are gone, and the buffer holds a million transitions of a competent drone flying from the training area to the target. The world model, the value and the policy are then trained only on that narrower data, and states far from the target, which are exactly what the outside starts produce, are no longer represented. That both the planner and the policy network crash more from outside starts (22% to 58%, and 3% to 34%) fits a change in the data rather than in the planner. This is a hypothesis; it can be checked by comparing the distances to the target stored in the 1M and 2M buffers (both are saved in the checkpoints on Narval), and avoided by a larger buffer, a larger training area, or a validation set that includes far starts.
- **Planning now helps the published model** inside the area (+43 points over the policy network, against -15 at 1M), but mainly because the policy network's precision drifted at the last checkpoint (seed 2's policy network fell from 40% to 0% from the held-out block while its planner rose); the 64×2 search is again as good as the full one inside the area and much better outside.
- **Against the small model:** after twice the training (about 14 hours against 4), the published model with planning is about level with the small one inside the area (72-75% against 78-83%; the differences are within the uncertainty of 3 seeds) and far less safe outside (58% crashes against 3%). So the published model was partly just slower, but more training does not make it the better choice for this task.

The snapshots of the continuation (every 100k steps from 1.1M to 2M) are kept on Narval and in the backup but were not flown; testing them would show whether a validation-chosen checkpoint avoids the loss outside the area.

## 6. Answers to Professor Bellinger's questions

This chapter answers the ten questions that were drawn from Professor Bellinger's feedback. Each answer starts with the short version, then gives the evidence (with the section of Chapter 5 where it is shown) and says how certain it is: **established** means several seeds, the right controls and a repeated measurement; **preliminary** means few seeds or a single setting, so the direction is clear but the size is uncertain.

### Q1. Why did short TD-MPC2 runs vary so much between seeds?

**Short answer.** Because at 50k steps the runs had not finished learning, and each seed was caught at a different stage. The differences were real properties of the trained agents, not noise of the tests.

**Evidence (established, Sections 5.2 and 5.3).** Re-flying all 33 Batch 1 agents 20 times each showed that 93-94% of the variation between flights of the small agents comes from differences between the agents (intraclass correlation 0.93 and 0.94); re-testing the same agent changes its result by much less than changing the seed. The learning curves of these runs were still improving at 50k steps, and the three published-size runs trained to 1M steps all converged to between 5.7 and 8.6 cm from their training start. The professor's suspicion that the runs were too short was right.

**What to say if asked more.** The training losses do not show when a run is good or bad (Section 5.3), so the seeds cannot be told apart from their logs; they have to be flown. This is why Batch 2 keeps snapshots and chooses checkpoints on separate validation starts.

### Q2. Does longer training resolve the variability?

**Short answer.** It resolves the variability in safety, not in precision. After 1M steps, no small random-start agent crashes from the training area or the held-out block, but how precisely each one hovers still varies between seeds and drifts within each run.

**Evidence (established for the small model, Sections 5.7 and 5.12).** All 11 small agents avoid crashes, but their final success from the training area ranges from 8% to 100%, because their steady-state error sits a few centimetres on either side of the 10 cm threshold. The snapshots show that the same happens within a run: from one 100k-step snapshot to the next, success changes five to seven times more than test noise can explain, mostly because the steady-state error moves between about 4 and 15 cm. Choosing each run's snapshot on separate validation starts recovers most of the loss (Section 5.12). For the published model, 1M steps is not even enough to finish learning: R1 improves almost steadily up to the last snapshot (Sections 5.5 and 5.12), so for that model "longer" is still part of the answer. Continuing three seeds to 2M steps (Section 5.18) gained 6-10 points of success inside the area and better precision, much less than the first million did, and it raised the crashes from starts outside the area from 22% to 58%. Longer training sharpens what the agent does in its training area and can cost it elsewhere.

**Why, and what to do.** The first suspect was the reward, which barely distinguishes 5 cm from 15 cm. The reward pilot (Section 5.13) rejected it in its simple form: with a reward ten times sharper near the target, the value network did learn to prefer the target, the large excursions disappeared, but the precision still drifted around the 10 cm line. The remaining suspects are the planner's sampling and the continual updates of the networks. In practice, a run should be judged by its best validated snapshot, not by its last one, and the success threshold should be reported together with the full distribution of errors.

### Q3. How do TD-MPC2 and PPO compare in simulator steps and wall-clock time?

**Short answer.** TD-MPC2 needs about 100 times fewer simulator steps than PPO, but PPO is faster in real time, because it simulates 2,048 drones at once while TD-MPC2 simulates one.

**Evidence (Batch 1, single start, established).** TD-MPC2 first reached 10 cm after 105k to 175k steps; PPO after 7.9M to 19.7M steps. In real time, PPO got there in about 2.6 minutes and TD-MPC2 in about 47 minutes. **Caveat:** the PPO runs had the two bugs described in Section 3.3 (a learning rate that never decreased, and a reset that leaked the previous flight's position and action). They are a historical baseline, not a tuned comparison, so the factor of 100 is indicative rather than exact.

**What to say if asked more.** Which matters depends on the setting. On a real drone, every step is real flight time, so sample efficiency matters most and TD-MPC2's advantage is decisive. In simulation, where steps are almost free, PPO's parallelism wins.

### Q4. How much of the GPU does TD-MPC2 use, and why is it slower in wall-clock time?

**Short answer.** Very little. With one drone, each training step consists of many small operations performed one after the other, and the time goes into starting them rather than into computing.

**Evidence (established, Section 5.17).** On a Narval GPU slice, one training step of the published model takes 6.8 ms to simulate the drone, 10.2 ms to plan its action and 2.6 ms for the learning update: 49 steps per second. At home the GPU was busy only 25-28% of the time; on Narval's GPU slices the usage counter is not available, so the measured times are the evidence instead. The planner is a good example: 13 times more arithmetic costs only about 3 times more time, which means the arithmetic is not the bottleneck.

### Q5. Can TD-MPC2 benefit from parallel environments?

**Short answer.** Only modestly in its standard form: 1.6 times faster for the published model and 2.9 times for the small one with 128 drones, according to a projection from measured timings.

**Evidence (established as a projection, Section 5.17).** Simulating 128 drones costs the same as one, but TD-MPC2 must still plan for every drone at every step, and it makes one learning update for every step collected. These two costs grow with the number of drones and cap the speed (74 steps per second for the published model, 271 for the small one). The projection reproduces the measured one-drone speeds, so it is reliable. Faster training would require fewer learning updates per collected step; that is a different algorithm, and its effect on sample efficiency would have to be measured before it could be called an optimisation.

### Q6. Can TD-MPC2 generalise from continuously randomised starting positions?

**Short answer.** Yes. Trained on random starts, TD-MPC2 learns to reach the target from anywhere in the region, does equally well from a block of starts it never trained from, and still mostly succeeds from starts up to 0.75 m outside the region. Trained from one start, it does none of this.

**Evidence (established for both model sizes, Sections 5.4 to 5.7).** Success / crashes on the same 110 test starts, with planning:

| Agents | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| published, single start (Batch 1 1M and C1) | 20-26% / 35-66% | 2-4% / 82-91% | 0-2% / 87-92% |
| **published, random starts (R1, 11 seeds)** | 46% / 0% | 39% / 2% | 20% / 26% |
| R1, policy network only, best validated snapshot | 90% / 0% | 81% / 0% | 59% / 6% |
| small, single start (Batch 1, 50k) | 19% / 9% | 11% / 15% | 4% / 31% |
| **small, random starts (D, 11 seeds)** | 78% / 0% | 83% / 0% | 59% / 3% |
| D, best validated snapshot | 97% / 0% | 91% / 0% | 70% / 2% |

Random starts beat single-start training for both sizes: +20 to +34 points for the published model against its control C1, +55 to +72 points for the small model against Batch 1, with the crashes inside the area gone. The held-out block, never used for training, does as well as the training area (83% against 78% for the small model; 39% against 46% for R1 with planning, 64% against 60% for its policy network), so interpolation works. Starts outside the area are harder (59% for the small model, 20% for R1 with planning and 40% for its policy network) but rarely end in a crash for the small model. The control C1 shows that the improvement comes from the random starts and not from the other change made at the same time (the target offset in metres). The size of the success also depends on the model and on how it is used: at 1M steps the small model with planning and the published model's policy network alone are the best, both at 90% or more from the training area when the checkpoint is chosen on validation starts.

**What to say if asked more.** The held-out block is inside the range of the training starts, so it tests interpolation; the outside starts test moderate extrapolation, and that is where the remaining failures are. Section 9.2 lists ways to reduce them.

### Q7. When and why does the learned world model improve generalisation?

**Short answer.** The world model helps when it is used in situations its training data covers, and hurts when it is not. On covered situations, planning with the model turns a safe but imprecise policy network into a precise controller. Off its data, the planner seeks out the model's mistakes and crashes.

**Evidence (Sections 5.4, 5.8 and 5.9).**

- **Single-start agents, used away from their start:** planning costs the published 1M agents 26 points of success [-46, -8] and multiplies their crashes by four (66% against 15% from training-area starts). The planner's flights go where the world model is 2.4 to 2.8 times more wrong than along the policy network's flights, the model error rises sharply just before crashes, and restraining the planner reduces crashes by 19 points. This is model exploitation, the known weakness of planning with learned models (Janner et al., 2019).
- **Random-start agents, used where they trained:** planning adds 69 points of success [+30, +96] for the small agents. Their policy network alone never crashes but typically ends 21 cm from the target; the planner brings this to 7 cm. Along the planner's flights the model is as accurate or more accurate than along the policy's, and restraining the planner only loses precision.
- **The published random-start agents (R1) show both sides at once:** their data covers the area, but their harder search (512 candidates over 6 rounds) still finds model errors. The planner loses to the policy network alone (-15 to -25 points), its flights go where the model is 1.2 to 2 times more wrong, and every way of restraining it helps: a narrower search (+8 to +11 points), more candidates from the policy network (+8 to +13), a smaller 64×2 search (+8 to +12) or a shorter horizon (fewer crashes). So "covered by the data" is necessary but not sufficient: how hard the planner searches also matters, and so, probably, does how far the world model has learned (R1 was still improving at 1M steps).
- **In between:** for random-start agents pushed by strong wind into states they never saw, the planner again crashes more than the policy network alone (41% against 29% at 4 m/s, Section 5.10).

**What to say if asked more.** The answer to "when" is "where the model's training data covers the situation", and the answer to "why" has two parts: on covered data, planning adds precision the policy network lacks; off it, optimising against an imperfect model finds its errors. The practical consequences are to train on the variation that will be met (random starts, varied physics), and to keep the planner away from situations the model does not know (for example by penalising uncertain predictions).

### Q8. Does planning improve robustness compared with the policy network alone?

**Short answer.** For the small random-start agents, yes against moderate disturbances, because planning keeps their precision; no agent's planner protects it against strong wind, and there planning makes crashes more likely. For the published agents, the policy network alone is the more robust under every disturbance tested.

**Evidence (established for D and R1, preliminary for C1 and H2; Sections 5.10 and 5.11).**

- **Small random-start agents (11 seeds):** with planning, 72-77% success under extra mass (up to +15%), weaker motors (down to -10%) and command delay (up to 40 ms), 61-68 points above the policy network alone. In wind, 60% at 1 m/s and none from 2 m/s on, but with no crashes up to 3 m/s. Beyond what any agent can handle, the planner crashes more than the policy network (41% against 29% at 4 m/s, 67% against 21% in 6 m/s gusts).
- **Published random-start agents (11 seeds):** the planner falls from 43% to 15-33% under mass, motor and delay changes, the policy network alone stays at 51-60%; in 3 m/s wind the planner crashes in 32% of the flights, the policy network in none.
- **Single-start agents:** the planner is worse than the policy network under every disturbance, and both crash often.
- **Training on varied physics (H2 pilot, 3 seeds):** slightly fewer crashes in strong wind (34% against 41% at 4 m/s), but much less precision in calm air (13% success against 78%), and the planner does not beat the policy network. Not a good trade in this form; the agent probably needs a history of observations to tell which conditions it is flying in.

**What to say if asked more.** Robustness here has two parts: staying precise, and not crashing. Planning helps the first whenever the world model is accurate, which for the small agents includes every moderate disturbance tested. It hurts the second whenever a disturbance pushes the drone into states the model never saw, which is the same mechanism as in Q7. Wind of 2 m/s or more is the limit for every agent trained without wind.

### Q9. How does model size affect inference speed and deployability?

**Short answer.** The small model is faster in every way and, at 1M steps, also more precise with planning: it plans in 11 ms on a laptop-class CPU, against 516 ms for the published model, and it succeeds in 78% of the flights from the training area against 46%. The published model's strength is its policy network: used alone, it is as good as the small model's planner at the right checkpoint and decides in 0.3 ms.

**Evidence (Sections 5.5, 5.7, 5.12 and 5.17).**

| | Published model (5.0M weights) | Small model (0.39M weights) |
|---|---|---|
| decision time with planning, GPU slice / 2 CPU cores | 6.1 ms / 516 ms | 1.4 ms / 11.4 ms |
| decision time, 64×2 planner (CPU) | 28 ms | 4.4 ms |
| decision time, policy network only (CPU) | 0.2 ms | 0.2 ms |
| training speed (Narval GPU slice) | 49 steps/s, about 7 h per 1M | 88 steps/s, about 4 h per 1M |
| success / crashes from the training area at 1M, with planning | 46% / 0% | 78% / 0% |
| same, policy network only | 60% / 0% | 9% / 0% |
| best validated snapshot, best decision rule | 90% / 0% (policy network only) | 97% / 0% (planning) |
| still improving at 1M steps? | yes | no, level since about 400k |

**What it means for deployment.** A real Crazyflie needs a decision every 20 ms. The published planner fits only with a GPU next to the drone; the small planner fits on a CPU with some margin, and its 64×2 version with a lot of margin and no loss of success (Section 5.7). The published model's policy network alone fits on any computer next to the drone, and is the most robust option under disturbances (Q8), but it has to be chosen at the right checkpoint. With twice the training (2M steps, 3 seeds, Section 5.18), the published planner draws level with the small one inside the area (72-75% against 78-83%) but crashes in 58% of the flights from outside starts against 3%, after about 14 hours of training against 4. For this task, the larger model does not pay for itself.

### Q10. What prevents safe transfer to a real Crazyflie?

**Short answer.** Two blockers and several risks, all identified in simulation and in the code; nothing has been tested on hardware.

**Blockers (established by reading the firmware interface and the lab's code).**

- **The commands do not match.** The agents output a collective thrust and three body moments, which MuJoCo applies directly. The Crazyflie firmware accepts attitude or body-rate setpoints with a thrust, positions or velocities, or motor powers; none of these is a body moment. The lab's earlier hardware code sent the moments as body-rate setpoints capped at 1 degree per second, which in practice controlled only the thrust. The agents have to be retrained with commands the firmware accepts, with the firmware's rate controller modelled in the simulator.
- **Thrust calibration.** The lab's notes report that the simulator's hover thrust sent a real drone to the ceiling and had to be scaled by 0.72. The mapping from the simulator's newtons to the firmware's thrust units has to be identified.

**Risks (measured in simulation).**

- **Jumpy commands.** The planner's commands change by about 30% of their range from one step to the next, and half of them sit at a limit (Section 5.15). The smoothing test (Section 5.16) shows that a 40 ms filter, or acting on the planner's averaged plan, keeps the precision with commands as smooth as the policy network's; the filter must then be sized together with the real motors' response.
- **Decision time.** The published planner needs a GPU next to the drone (6 ms per decision on a GPU slice, 516 ms on a CPU). The small planner nearly fits on a laptop CPU (11 ms) and the policy network alone fits anywhere, but it is much less precise.
- **Wind.** Agents trained without wind lose their precision in a steady wind of 2 m/s or more (Section 5.10); training with varied wind and mass did not fix this in the H2 pilot (Section 5.11): it gave slightly fewer crashes in strong wind but much less precision in calm air.
- **Things the simulator does not have:** the state estimate of a real drone (the lab's final report found the Flow deck unreliable on dark netted mats), radio delay and its jitter, take-off and landing (every training flight starts in the air, at rest), battery sag and the propellers' own dynamics.

**What is already solved in simulation.** The deployment adapter (`deploy_controller.py`) builds the agents' observation from Crazyflie-style state estimates (units, frames, quaternion order, the previous-action convention) and was checked to be identical to the simulator's over 300 closed-loop steps; two mistakes of the lab's earlier hardware path (an observation normaliser that was not applied, and a different previous-action convention) are avoided. A supervised test sequence is proposed in `reports/DEPLOYMENT_READINESS.md` and in Section 9.3.

## 7. The code, file by file

### 7.1 The repository at a glance

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

### 7.2 How the pieces fit together

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
  -> eval_hover.py             final tests -> evaluation/, evaluation_policy_only/, 
      evaluation_plan64x2/
```

A **test** of a saved agent (any of the `eval_*.py` scripts) rebuilds the agent from the run's own saved configuration (`.hydra/config.yaml`) and loads its weights with `eval_hover.load_agent`, then flies it with `hover_eval.FlightTester` or one of its variants. **Analysis** scripts (`analyze_*.py`) read the CSV files the tests wrote and produce plots and tables. On Narval, each kind of job has a small shell script in `cluster/`, and `cluster/pull_results.sh` copies the results home.

### 7.3 The task and the environment

- **`crazyflie_hover.py`**: the bridge between the lab's simulator and the Gymnasium interface the trainer expects. It defines the action limits (`ACTION_LOW`, `ACTION_HIGH`), the Batch 2 start area (`START_AREA_LOW/HIGH`, `HELD_OUT_XY_LOW/HIGH`, `MIN_GOAL_DISTANCE`) and its sampler `sample_training_start()`. `HoverDrone` wraps the lab's environment and adds the option of the target offset in metres. `place_drone()` moves a freshly reset drone to any start. `CrazyflieHoverEnv` is the environment the trainer uses: `reset()` starts a flight (a random start when `random_starts` is on) and `step()` applies one action and returns the observation, reward and end flags.
- **`crazyflie_hover_dr.py`**: `RandomizedHoverEnv`, the same environment with randomised physics in every flight (mass, motor strength, wind), for the H2 experiment.
- **`crazyflie_hover_sharp.py`**: `SharpHoverEnv`, the same environment with one extra reward term concentrated within about 5 cm of the target (`sharp_bonus()`), for the reward pilot of Section 5.13.

### 7.4 Training

- **`train_hover.py`**: the entry point. It registers the environments with Gymnasium (`CrazyflieHover-v0`, `CrazyflieHoverRandomized-v0` for H2 and `CrazyflieHoverSharp-v0` for the reward pilot) and starts the trainer with a configuration chosen by `--config-name`.
- **`tdmpc2_jax/`**: TD-MPC2 in JAX, copied from Shane Flandermeyer's open-source implementation (commit 5b05ff4, MIT licence in `UPSTREAM_LICENSE`) and adapted.
  - `train.py`: the training loop. It collects experience, stores it in the replay buffer, runs one learning update per step, records test flights every 25k steps, saves checkpoints and snapshots, and can resume an interrupted run (with fresh randomness).
  - `build.py`: builds an untrained agent from a configuration. Training and testing both use it, so saved weights always fit.
  - `tdmpc2.py`: the TD-MPC2 algorithm. `act()` encodes the observation and calls `plan()` (MPPI planning) or the policy network; `update()` performs one learning step of the world model, the value ensemble and the policy; `estimate_value()` scores imagined plans.
  - `world_model.py`: the networks (encoder, dynamics, reward, policy, value ensemble and its slowly updated target copy) and their optimisers.
  - `networks/`, `common/`, `data/`: building blocks (normalised linear layers, ensembles, the activation functions, the two-hot reward and value encoding, the replay buffer in `data/sequential_buffer.py`). `envs/` holds upstream code for other benchmarks that this project does not use.
- **Configurations** (`tdmpc2_jax/*.yaml`, read by Hydra): `config_hover.yaml` (small settings, the base of all others), `config_hover_standard.yaml` (published settings), `config_batch2_random_start.yaml` (R1), `config_batch2_fixed_start.yaml` (C1), `config_batch2_random_start_small.yaml` (D), `config_batch2_randomized_small.yaml` (H2), `config_batch2_sharp_small.yaml` (reward pilot). `config.yaml` is the upstream default and is not used.
- **`run_seeds.py`**: trains and tests a list of seeds on the home PC (used for Batch 1).

### 7.5 Testing trained agents

- **`hover_eval.py`**: the test-flight engine. It defines the test starts (`TEST_STARTS`, `TEST_SETS`, `CURVE_STARTS`), the measures (`Flight.summary()`), and `FlightTester`, which flies many flights side by side inside one compiled GPU loop, with planning or with the policy network only.
- **`eval_hover.py`**: tests a saved run on any test group, with planning, the policy network only (`--no-mpc`) or another planner size (`--plan 64x2`), at the final checkpoint or a snapshot (`--snapshot`). It refuses to overwrite existing results.
- **`eval_disturbances.py`**: flies an agent in wind, gusts, with extra mass, weaker motors or delayed commands (`WindyDrone`, `DisturbedFlightTester`) and records how each flight ended (`end_cause`).
- **`eval_snapshots.py`**: flies every kept snapshot from the validation starts and all test groups.
- **`eval_planner_variants.py`**: flies the same agent with the policy only, the trained planner, a narrower planner, and a planner with half its candidates from the policy network.
- **`eval_horizon.py`**: flies the same agent with planning horizons of 1, 3, 5 and 10 steps.
- **`eval_smoothing.py`**: flies the same agent with the planner as trained, with the first action of the planner's averaged plan, with the planner's commands through a low-pass filter, and with the policy network alone, each with instant motors and with slower motors (`SmoothedFlightTester`).
- **`model_error.py`**: records flights and measures how wrong the agent's own world model is along them.

### 7.6 Analysis

- **`analyze_batch2.py`**: the main Batch 2 analysis: learning curves per test group, final results per group and decision rule, comparisons between experiments with bootstrap confidence intervals, disturbance results and crash causes.
- **`analyze_snapshots.py`**, **`analyze_model_error.py`**, **`analyze_planner_variants.py`**, **`analyze_horizon.py`**, **`analyze_smoothing.py`**: summaries of the corresponding tests across seeds.
- **`analyze_precision.py`**: splits the late error of full flights into its horizontal and vertical parts and the height offset, per decision rule, and plots the share of flights ending within any distance from 0 to 50 cm, so that no conclusion depends on the 10 cm threshold.
- **`analyze_flight_quality.py`**: from the step-by-step trajectories, the time to stabilise, the jitter of the commands, how often a command is at its limit, and the worst flights.
- **`analyze_start_outcomes.py`**: the map of the test starts coloured by how many seeds succeeded from each.
- **`probe_value.py`**: the agent's own value of hovering still at points around the target (Section 5.13); nothing is flown.
- **`render_flights.py`**: a video of a trained agent flying up to four test flights side by side, replayed in MuJoCo's renderer with its path, the target and the 10 cm zone (`RecordingTester` records the full pose of every step). The videos in `reports/videos/` were made with it: the small model with and without planning, the single-start agent C1 and R1.
- **`analyze_all.sh`**: runs every analysis above on the results copied home, skipping whatever is not finished yet.
- **`variance_decomposition.py`** (A1) and **`fluctuation_analysis.py`** (A2): the two analyses of Batch 1 variability.
- **`analyze_seeds.py`** and **`compare_methods.py`**: the Batch 1 plots and the TD-MPC2 vs PPO comparison.

### 7.7 Checks (automated tests)

- **`check_batch2.py`**: 23 checks of the Batch 2 task code (Section 4.6).
- **`check_disturbances.py`**: 10 checks of the disturbance physics with a scripted pilot.
- **`check_deployment.py`**: checks that the deployment adapter builds exactly the simulator's observation.
- **`check_hover.py`**: the original Batch 1 sanity check of the environment.

### 7.8 Measurements of speed

- **`bench_latency.py`**: time per control decision for three network sizes and five planning budgets.
- **`profile_training.py`**: where one training step spends its time.
- **`bench_parallel.py`**: timing of simulation, planning and updates with many drones at once, and the projected training speed.

### 7.9 Deployment preparation

- **`deploy_controller.py`**: a simulation-only adapter for a future real flight. `DroneState.from_crazyflie_log()` converts a state estimate in the Crazyflie's own units and conventions; `build_observation()` reproduces the training observation exactly; `TDMPC2Controller` runs a trained agent, times each decision and must be warmed up before use. `to_crazyflie_command()` deliberately refuses to produce a drone command, because the action space does not yet match what the firmware accepts.

### 7.10 Historical PPO

- **`ppo_hover.py`**: trains the lab's PPO on exactly the Batch 1 task and tests it with the same test flights. It inherits the two bugs described in Section 3.3; its results are kept as a historical baseline.

### 7.11 Cluster scripts (`cluster/`)

- `python.sh`: runs Python inside the container that provides the exact package versions (`requirements-lock.txt`), installed once by `setup_env.sh`.
- `seed_job.sh`: trains and tests one seed per array task, optionally in pieces (`SKIP_TESTS`) and with the extra tests without planning (`DEPLOYMENT_TESTS`).
- `eval_job.sh`, `disturb_job.sh`, `snapshot_job.sh`, `model_error_job.sh`, `variants_job.sh`, `horizon_job.sh`, `smoothing_job.sh`: run the corresponding test on a list of runs, one per array task.
- `profile_job.sh`, `parallel_job.sh`, `test_job.sh`: speed measurements.
- `ppo_job.sh`: the historical PPO runs.
- `pull_results.sh`: copies all results from Narval to the home PC, without the large checkpoints.

### 7.12 Documents

- `README.md`: how to install and run everything.
- `CONTEXT.md`: the project's history and status, machines and environments.
- `EXPERIMENTS.md`: the list of experiments, their validation and the launch log.
- `reports/`: `RESEARCH_LOG.md` (chronological log), `experiment_registry.json` (machine-readable list of every experiment), `RESULTS_SUMMARY.md`, `NEXT_STEPS.md`, `DEPLOYMENT_READINESS.md`, `LITERATURE.md`, this guide (`GUIDE.md`, built into a PDF by `make_guide.py`), a short plain-language version (`BATCH2_EXPLAINED.md` and `.pdf`, built the same way), the one-page update for the supervisor (`SUPERVISOR_UPDATE.md`) and the handoff notes of the night (`HANDOFF.md`).

### 7.13 Where the results are

Every training run has its own folder (`outputs/batch2/<experiment>/seed_XX/` at home, `$SCRATCH/tdmpc2/batch2/<experiment>/seed_XX/` on Narval) with the same layout:

| Item | Contents |
|---|---|
| `.hydra/config.yaml` | the exact configuration of the run |
| `learning_curve.csv` | test flights during training, every 25k steps |
| `checkpoint/` | the final agent and its replay buffer (Narval only) |
| `snapshots/` | the agent every 100k steps (Narval only) |
| `tensorboard/` | training statistics (losses) |
| `evaluation*/` | final tests: `summary.csv` (one row per flight), `trajectories.csv` (every step) |
| `disturbances*/`, `snapshot_evaluation*/`, `model_error/`, `planner_variants/`, `horizon_test/`, `smoothing_test/` | the follow-up tests |
| `gpu_usage.csv`, `*.log` | GPU samples and the training log |

Each experiment folder also has `batch_info.json`, which records the settings and the code version, and an `analysis/` folder with `summary.md`, the one file to read first: every table of that experiment (final tests, comparisons with the other experiments, disturbances) in plain text, with its figures next to it. The comparisons across experiments are in `outputs/batch2/*_summary/` (snapshots, model error, planner variants, horizon, smoothing, precision) and `outputs/batch2/flight_quality/`. A few snapshots were copied home for the value probe (Section 5.13). The old Narval results were backed up to `/project/def-cbelling-ab/mounirz/tdmpc2_scratch_backup_2026-11-01/` before the scratch purge.

### 7.14 Reproducing the results

All commands run from the repository root. At home, Python is `.venv/bin/python`; on Narval, `bash code/src/tdmpc2_training/cluster/python.sh`.

```
# checks
.venv/bin/python code/src/tdmpc2_training/check_batch2.py
.venv/bin/python code/src/tdmpc2_training/check_disturbances.py
.venv/bin/python code/src/tdmpc2_training/check_deployment.py <batch 1 run> <batch 2 run>

# train one seed of R1 (about 7 hours on a GPU slice)
.venv/bin/python code/src/tdmpc2_training/train_hover.py --config-name=config_batch2_random_start 
    seed=0

# test it
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run>             # with planning
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run> --no-mpc    # policy network only
.venv/bin/python code/src/tdmpc2_training/eval_disturbances.py <run>
.venv/bin/python code/src/tdmpc2_training/eval_snapshots.py <run>

# on Narval: a whole experiment, then copy the results home
sbatch --array=0-10 --time=10:00:00 --export=ALL,CONFIG=config_batch2_random_start,\
BATCH=batch2/random_start_1M,STEPS=1000000,DEPLOYMENT_TESTS=1 
    code/src/tdmpc2_training/cluster/seed_job.sh
bash code/src/tdmpc2_training/cluster/pull_results.sh

# analysis: everything at once, or one analysis at a time
bash code/src/tdmpc2_training/analyze_all.sh
.venv/bin/python code/src/tdmpc2_training/analyze_batch2.py outputs/batch2/random_start_1M \
    --compare outputs/batch2/random_start_small_1M outputs/batch2/fixed_start_1M 
        outputs/batch1_standard_1M

# this guide
~/.local/bin/uv run --no-project --with reportlab python 
    code/src/tdmpc2_training/reports/make_guide.py
```

### 7.15 How the code works: one step of the drone, one decision, one learning step

This section follows the code through the three things that happen all the time during training: the simulated drone moves for one step, the agent decides on its next action, and the agent learns from what it has seen. The line numbers refer to the files as of November 2, 2026.

**One step of the drone: `CrazyflieHoverEnv.step()` (`crazyflie_hover.py` line 138).**

- 1. The four commands (thrust in newtons, three body moments) are checked and clipped to their limits, then passed to the lab's compiled step function, `CrazyflieEnv.step()` (`gpu_training/crazyflie_env.py` line 259).
- 2. The lab's step builds the physics model with this flight's mass, inertia and motor strength (lines 266-274; this is how the disturbance tests and H2 change the physics), clips the commands again, and advances MuJoCo by 10 sub-steps of 2 ms, that is one control step of 20 ms (line 283).
- 3. The observation is built by `_get_obs()` (line 366). Small random errors are added to the measured position (standard deviation 1 cm), velocity (0.5 cm/s), angular velocity and orientation, as a real sensor would make. From these noisy values come the rotation matrix, the offset to the target in the drone's own axes (in metres for Batch 2; `HoverDrone` in `crazyflie_hover.py` handles this), the velocity in the drone's axes, and the previous command.
- 4. The reward is computed from the true positions (lines 287-327): the bonus for staying in the air, the proximity bonus and the distance improvement, minus 1 if the flight ends (Section 2.2). The flight ends if the drone is below 5 cm or more than 2 m further from the target than where it started (lines 313-318).
- 5. Back in `CrazyflieHoverEnv.step()`, all results are copied from the GPU in one transfer, the flight is cut off after 500 steps, and the trainer receives the observation, the reward, whether the flight ended, and the position.

The sections below follow the agent's side, in `tdmpc2_jax/tdmpc2.py` and `tdmpc2_jax/world_model.py`.

**One decision: `TDMPC2.act()` (line 88).**

- 1. The 31 observation numbers go through the encoder (`WorldModel.encode()`, `world_model.py` line 232), which returns the latent state: 128 numbers for the small model, 512 for the published one.
- 2. Without planning, the policy network turns the latent state into an action directly (`WorldModel.sample_actions()`, line 258), and the decision is over.
- 3. With planning, `TDMPC2.plan()` (line 125) runs MPPI:
   - It first rolls the policy network forward with the dynamics network (`WorldModel.next()`, line 239) to make 16 (small) or 24 (published) candidate plans of 3 actions each (`plan()`, lines 147-176).
   - The other candidates (112 or 488) are drawn at random around the current best guess, which starts as the previous decision's plan shifted by one step (lines 181-201), and are clipped to the command limits.
   - Every candidate is scored by `estimate_value()` (line 480): the rewards that the reward network predicts for its 3 actions, discounted by 0.99 per step, plus the discounted value of the latent state reached at the end, taken as the average of the value networks.
   - The best 16 (small) or 64 (published) candidates are the elites. The sampling distribution is refitted to them, weighted by their scores, and the process repeats: 3 rounds for the small model, 6 for the published one (lines 203-226).
   - When testing, the first action of the best elite is executed (lines 229-230). During training, an elite is drawn at random according to its score and noise is added to its first action (lines 232-244), which is how TD-MPC2 explores. This choice of a single candidate is why the planner's commands jump from step to step (Section 5.15).
   - The weighted average of the elites (the averaged plan) is returned to start the next decision's search.
- 4. The action, between -1 and 1, is converted to thrust in newtons and body moments (Gymnasium's `RescaleAction` in training, `FlightTester` in the tests).

**One learning step: `TDMPC2.update()` (line 251).** After every environment step, the training loop (`train.py` line 277) draws a batch of short sequences of 3 consecutive steps (observation, action, reward, next observation) from the replay buffer (64 sequences for the small model, 256 for the published one) and updates every network once:

- 1. **World model (consistency loss).** The first observation of each sequence is encoded, and the dynamics network predicts the next three latent states from the recorded actions. Each prediction is compared with the encoding of the observation that actually followed; the difference is the consistency loss (lines 290-304). This is what teaches the world model to predict, and it is the same quantity that `model_error.py` measures along test flights.
- 2. **Reward (reward loss).** The reward network predicts the recorded rewards (lines 309-320). Rewards and values are not predicted as plain numbers: they are encoded as a probability over 101 bins on a logarithmic scale (a "two-hot" encoding), which makes learning robust to their size.
- 3. **Values (value loss).** The target for each value network is the recorded reward plus 0.99 times the value of the next state, taken from slowly updated copies of the value networks (the smaller of two chosen at random, to avoid overestimating), with an action from the policy (lines 328-362). This is temporal-difference learning, the "TD" of TD-MPC2. The slow copies follow the trained networks by 1% per step (line 412).
- 4. **Weighting.** Within each sequence, later steps count less (a factor 0.5 per step), because predictions further ahead are harder.
- 5. **Policy.** The policy network is trained to choose actions with a high average value along the predicted latent states, with the values rescaled by a running estimate of their size, plus a very small bonus for randomness (entropy coefficient 0.0001) that keeps it exploring (lines 430-455).

Everything except the policy is trained on the sum of the consistency, reward and value losses; the policy is trained separately against the updated value networks. The losses reported in the training logs (`tensorboard/`) are these quantities, which is why Section 5.3 could compare them with the test flights.

## 8. Limitations

Every result in this guide comes with conditions. This chapter collects them in one place, so that no result is stretched beyond what it supports. They fall into four groups: the simulation, the task, the statistics, and the comparisons.

### 8.1 The simulation

All results are from simulation; no real drone was flown. The simulator is a rigid-body model: the agent's thrust and body moments are applied directly to the drone's body, so the motors respond instantly and perfectly (only the delay test adds a lag). There is no model of the individual propellers, no ground effect near the floor, and no battery that runs down during a flight. Air resistance comes from MuJoCo's simple fluid model of a box-shaped body; wind is a uniform movement of the whole air mass, steady or as one 1-second gust, not the turbulent, changing wind of a real room or outdoors. Sensor noise is simple random noise added to perfect measurements; a real Crazyflie estimates its state with a filter, whose errors are correlated in time and can drift. Each of these differences could change how a controller behaves on a real drone, which is why Chapter 6 treats real-world readiness separately (question 10).

### 8.2 The task

The task is deliberately narrow: one drone, one target point one metre above the origin, starts at rest and level. The held-out block tests interpolation only in the horizontal plane (it spans all training heights). The outside starts are at most 0.75 m beyond the training area, so they test moderate extrapolation, not arbitrarily distant starts. A controller that does well here has learned to recover from a displacement, not yet to follow a path, take off, land, or fly with other drones.

### 8.3 The statistics

- **Seeds.** The main experiments (R1 and D) have 11 seeds each, enough for the averages and confidence intervals reported. C1, H2 and the reward pilot have 3 seeds; their results show a direction, and their sizes should be read as preliminary.
- **Test starts.** Each agent is flown once from each of 50, 30 and 30 fixed starts. A share from one agent therefore moves in steps of 2-3 percentage points, and the reported standard errors (across seeds) do not include the uncertainty from the particular choice of starts. Using the same starts for every agent makes comparisons paired and precise, at the price of this small blind spot.
- **Thresholds.** Success means a late error below 10 cm. This threshold is a choice, and it matters more than expected: the agents' steady-state error often sits between 5 and 15 cm, so a few centimetres decide between success and failure (Section 5.12). The figure of success for every threshold (Section 5.8) shows each comparison without choosing a threshold, and crashes, which do not depend on it, are always reported next to success.
- **Repeatability.** Training on a GPU is not exactly repeatable: two runs with the same seed drift apart slightly once learning starts (Section 4.6). Results are therefore statements about the distribution of agents a configuration produces, which is what the seeds measure, not about one exact agent.

### 8.4 The comparisons

- **Test-time changes.** The planner variants, the horizon test and the smoothing test change the planner only when testing agents trained with the default settings. An agent trained with a longer horizon, with more candidates from its policy network or with a command filter might behave differently; these tests show what the trained agents' models can support, not what the best training settings are.
- **Assumed motor lags.** The 30 and 60 ms motor lags of the smoothing test were chosen to probe sensitivity. They are not measured values for the Crazyflie, and the real motor response must be identified before the filter can be sized.
- **The reward pilot changes the task.** The extra reward near the target (Section 5.13) makes it a different task from the lab's, even though the tests measure the same distances. If it is adopted, it should be reported as a change of the reward, not as a property of TD-MPC2.
- **Model sizes at equal steps.** The published and small models are compared after the same 1M steps, but the published model had not finished learning (Section 5.5). "The small model is better" holds at 1M steps; the 2M continuation has only three seeds, so its result (level inside the area, worse outside) is an indication.
- **H2 tested one design of domain randomisation.** Its negative result concerns randomised physics without any memory in the observation; it does not show that domain randomisation cannot help TD-MPC2.
- **Hyperparameters.** TD-MPC2 was used with its published settings (and the small settings from September), without tuning. A tuned configuration might change the sizes of the effects, especially for the small model.
- **PPO.** The Batch 1 comparison with PPO used the lab's PPO with the two bugs described in Section 3.3. It was not repaired or re-run, so it is a historical baseline, not a fair tuned comparison. In particular, PPO's late degradation is likely caused by its learning rate never decreasing.
- **Speed.** Times were measured on a 2/7 slice of an A100 shared with other users' jobs on the same card, and at home on an RTX 3060; other hardware will give other numbers. The parallel-drone speeds are projections from measured timings, not full training runs, and they do not measure how many environment steps a parallel version would need to learn.
- **World-model error.** The model error is measured in the agent's own internal (latent) space. It is meaningful for comparing flights of the same agent (planner versus policy network, before a crash versus a completed flight), less so for comparing different agents.

## 9. What to do next

### 9.1 For the next meeting with Professor Bellinger

The results answer most of the questions he asked; Chapter 6 gives the answers in the order of his feedback, and `reports/SUPERVISOR_UPDATE.md` is a one-page version to send. Four figures carry most of the story: the map of outcomes per start (random against single-start training), the success-for-every-threshold figure (planning against the policy network alone), the disturbance figure, and the smoothing figure. It is worth asking him four questions, because the answers decide where the next weeks go:

- Should the project move toward a real flight now (which first requires retraining with commands the Crazyflie firmware accepts, Section 9.3), or deepen the simulation study first?
- Which controller should the project carry forward: the small model with planning (97% at the best checkpoint, needs 11 ms per decision on a CPU) or the published model's policy network alone (90%, 0.3 ms, the most robust under disturbances)? The answer decides what a first hardware test would use.
- Is a fair PPO comparison needed for the thesis? It would mean fixing the two PPO bugs and rerunning it on random starts.
- Is the extra compute for longer runs of the published model worth it? The 2M continuation (Section 5.18) suggests not for this task: level with the small model inside the area, much less safe outside, at three to four times the training time.

### 9.2 Next experiments in simulation

In order of expected value for the effort:

- 1. **Choose checkpoints on the validation starts** as the standard procedure (Section 5.12), so that a run's final result does not depend on where its precision happened to be at the last step. This costs nothing new: every run already keeps snapshots.
- 2. **Check why longer training hurt outside the area** (Section 5.18). Compare the distances to the target stored in the replay buffers at 1M and at 2M steps (both are in the checkpoints on Narval) and fly the continuation's snapshots. If the buffer explanation holds, long runs need a buffer at least as long as the run.
- 3. **Find the source of the precision drift.** The reward pilot ruled out the flat value function (Section 5.13). Two cheap tests remain: fly one snapshot with several planner random seeds (if the hover position changes with the seed, the planner's sampling is the cause), and lower the learning rate late in training (if the drift shrinks, the continual updates are the cause).
- 4. **Train with smoothing in the loop.** The smoothing test shows what a trained agent tolerates at test time; training with the command filter (or a penalty on command changes) would let the agent adapt to it, and would also give a smoother policy network.
- 5. **Randomised physics with a memory.** The H2 pilot (Section 5.11) lost precision, probably because the agent cannot tell which conditions it is flying in. Repeat it with a short history of observations and commands in the observation (or an estimate of the wind and mass), 3 seeds first, and add the motor lag to the randomised quantities.
- 6. **Reduce crashes from far starts**, the remaining weakness: the outside starts crash far more often than the others, especially for the published planner (26%). Options are a larger training area, or planning that avoids states where the world model is uncertain (penalising disagreement between the value networks).
- 7. **Tame the published planner.** Its full search loses to its own policy network (Sections 5.5 and 5.9). At 2M the 64×2 search is still much safer outside the area than the full one (22% against 58% crashes). Test whether a smaller search (64×2), more candidates from the policy network or a shorter horizon used during training (not only at test time) closes the gap.
- 8. **Training speed**: parallel drones with fewer learning updates per collected step, measuring how many more steps learning then needs. This must be reported as a variant of TD-MPC2, not as TD-MPC2.

### 9.3 Toward a real Crazyflie

Nothing in this guide authorises or prepares a flight on its own; every hardware step needs Professor Bellinger's approval, the lab's safety training and a written test plan (`reports/DEPLOYMENT_READINESS.md`). In order:

- 1. **Firmware-compatible commands.** Retrain with collective thrust plus body-rate setpoints, with a model of the firmware's rate controller in the simulator, and repeat the Batch 2 tests. This is the largest single task, and it is required before anything else.
- 2. **Identify the motors.** Measure, on a bench and under supervision, how fast the real motors follow a thrust command, then size the command filter of Section 5.16 against that measured lag instead of the assumed 30 and 60 ms.
- 3. **Radio-only test.** With the motors never armed, read the real state estimate at 50 Hz, run the deployment adapter (`deploy_controller.py`), and record the decisions and the end-to-end delay and its jitter. Send nothing to the drone.
- 4. **Tethered or caged hover** with a safety pilot, an emergency stop and low thrust limits, taking over from the firmware's own take-off and handing back for landing.

## Appendix A. Glossary

**Action.** What the agent sends to the drone at each step: collective thrust and three body moments (Section 2.2).

**Agent.** The learning program, here a TD-MPC2 controller with its networks.

**Array job.** A Slurm job that runs the same script many times with a different task number (here: one seed or one run per task).

**Bootstrap confidence interval.** An interval for a mean or a difference, obtained by recomputing it many times on seeds resampled with replacement. A 95% interval that does not contain zero means the difference is very unlikely to be due to chance.

**Checkpoint.** The saved state of a training run: the agent's weights, and on Narval the replay buffer, so that training can be resumed.

**Crash.** A flight that ends before 10 s, because the drone went below 5 cm (it hit the **ground**) or strayed more than 2 m beyond its starting distance (**out of bounds**).

**Domain randomisation.** Training in many variants of the physics (here: mass, motor strength, wind) so that the controller cannot rely on one exact set of values.

**Elites.** In MPPI, the best-scoring candidate plans of a round, used to choose the next round's candidates.

**Encoder.** The network that turns the 31 observation numbers into the latent state.

**Ensemble.** Several networks of the same kind trained side by side (here five value networks); their disagreement indicates uncertainty.

**Environment step.** One decision of the agent and 0.02 s of simulated flight.

**Extrapolation (out-of-distribution).** Testing in situations beyond the range seen in training, such as the outside starts.

**Flight.** One test of 500 steps (10 s) from one start.

**Generalisation.** Working well in situations not seen in training.

**GPU, MIG slice.** A graphics card used for computation. Narval's A100 cards are divided into isolated slices (multi-instance GPU); each job here used a slice of 2/7 of a card with 10 GB of memory.

**Held-out block.** The square x, y in [0.25, 0.75] m that is inside the range of the training area but never used for training starts; it tests interpolation.

**Horizon.** How many steps ahead the planner imagines (3 steps, 0.06 s, in TD-MPC2's default).

**Intraclass correlation (ICC).** The share of the total variance that comes from differences between groups (here between seeds) rather than within them (between repeated flights of the same agent).

**Interpolation.** Testing inside the range of the training distribution, in a part that was not visited (the held-out block).

**JAX.** A Python library for numerical computing that compiles programs for GPUs. TD-MPC2 and the simulator both run in JAX here.

**Late error.** The average distance to the target over the last 2 s of a flight.

**Latent state.** The agent's internal description of the situation, produced by the encoder (128 or 512 numbers).

**Model-based / model-free.** A model-based method learns how the world works (a world model) and uses it; a model-free method learns behaviour directly from experience.

**Model exploitation.** When a planner optimises against a learned model, it can pick actions for which the model is wrong in an optimistic way, because those look best to the model. The actions then perform worse in reality than predicted.

**MPC (model predictive control).** Re-planning at every step a few steps into the future and executing only the first action.

**MPPI (model predictive path integral).** The sampling-based planning algorithm of TD-MPC2: sample candidate plans, keep the elites, refit, repeat.

**MuJoCo, MJX.** The physics engine used by the simulator, and its JAX version that runs on GPUs.

**Narval.** A computing cluster of the Digital Research Alliance of Canada, run by Calcul Québec.

**Observation.** What the agent measures at each step: 31 numbers (Section 2.2).

**Paired comparison.** Comparing two methods on the same agents, starts and sensor noise, so that only the method differs; much more precise than comparing unrelated runs.

**Permutation test.** A test of whether a correlation could arise by chance, by shuffling one of the two variables many times and counting how often the shuffled correlation is as strong as the real one.

**Policy, policy network.** The rule that maps an observation to an action; TD-MPC2's policy network proposes actions from the latent state.

**Policy prior samples.** The candidate plans in MPPI that are proposed by the policy network rather than drawn at random (24 of 512 for the published settings, 16 of 128 for the small ones).

**PPO.** Proximal Policy Optimization, the model-free method used by the previous students (Section 2.3).

**Replay buffer.** The memory of all past transitions (observation, action, reward, next observation) from which TD-MPC2 learns.

**Reward.** The number the environment returns after each action, saying how good it was (Section 2.2).

**Seed.** The number that fixes all the randomness of a training run; different seeds give independently trained agents.

**Slurm.** The job scheduler of Narval: it queues jobs and starts them when resources are free.

**Snapshot.** The agent's weights saved every 100k steps during a Batch 2 run (without the replay buffer), so that the run's history can be tested afterwards.

**Spearman correlation.** A correlation between the ranks of two variables; it detects any monotonic relationship.

**Standard error (SE).** The uncertainty of a mean: the standard deviation divided by the square root of the number of seeds.

**Success.** A flight that lasts the full 10 s with a late error below 10 cm.

**TD learning.** Temporal-difference learning: improving the value function by comparing its estimate with the reward actually received plus its own estimate at the next step. It is the "TD" in TD-MPC2.

**Updates-to-data ratio (UTD).** The number of learning updates per environment step (1 for TD-MPC2's default).

**Validation starts.** 40 starts used only to choose checkpoints, kept separate from the test starts.

**Value function.** The estimate of the long-term reward from a situation (and an action); TD-MPC2 uses it to judge everything beyond its 3-step horizon.

**Wall-clock time.** Real elapsed time, as opposed to the number of environment steps.

**World model.** The networks that predict what happens next (dynamics) and the reward, used for planning.

## Appendix B. Jobs and code versions

Every Batch 2 job ran on Narval under the account `def-cbelling-ab`, on a 2/7 slice of an A100 GPU (except the copy job, which needs no GPU). Each job prints the git commit it ran in the first line of its log (`$SCRATCH/tdmpc2/batch2/logs/`), and each training batch records its settings and commit in `batch_info.json`. All job numbers are also listed in `$SCRATCH/tdmpc2/batch2/job_ids.txt`. The table below comes directly from the scheduler's records (`sacct`).

| Job | Name | Tasks | What it did | State (09:00) | Longest task |
|---|---|---:|---|---|---:|
| 7659264 | b2-narval-test | 1 | end-to-end check on Narval before launch | completed | 0:15 |
| 7659438 | b2-random-start | 11 | **R1** training, 1M steps, and final tests | completed | 6:54 |
| 7659439 | b2-fixed-start | 3 | **C1** training and final tests | completed | 6:45 |
| 7659440 | b2-retest | 25 | Batch 1 agents re-flown (A1, R4) | completed | 0:07 |
| 7659441 | b2-profile | 1 | training-time profile (Section 5.17) | completed | 0:30 |
| 7660609 | b2-disturb-batch1 | 3 | disturbances, Batch 1 published 1M | completed | 0:30 |
| 7660720 | b2-disturb-R1 | 11 | disturbances, R1 | completed | 0:30 |
| 7660721 | b2-disturb-C1 | 3 | disturbances, C1 | completed | 0:30 |
| 7661493, 7661496 | b2-small-p1, -p2 | 3 + 3 | **D** pilot seeds 0-2, two pieces of 3 h | completed | 1:54 |
| 7661497 | b2-disturb-small | 3 | disturbances, D seeds 0-2 | completed | 0:06 |
| 7662436, 7662437 | b2-small-p1, -p2 | 8 + 8 | **D** seeds 3-10 | completed | 2:31 |
| 7662438 | b2-disturb-small | 8 | disturbances, D seeds 3-10 | completed | 0:07 |
| 7662439, 7662440 | b2-snap-R1, -C1 | 11 + 3 | snapshots, R1 and C1 | completed | 1:03 |
| 7662441, 7662442 | b2-snap-small | 3 + 8 | snapshots, D | completed | 0:17 |
| 7663279 | tdmpc2-parallel | 1 | parallel-drone timing (E) | completed | 0:36 |
| 7663280-7663284 | tdmpc2-model-error | 50 | model error: Batch 1, R1, C1, D | completed | 0:06 |
| 7663755-7663758, 7663773 | tdmpc2-variants | 50 | planner variants: Batch 1, R1, C1, D | completed | 0:15 |
| 7664888, 7664889 | b2-dr-p1, -p2 | 3 + 3 | **H2** training | completed | 2:01 |
| 7664892, 7664893 | b2-disturb-dr, tdmpc2-variants | 3 + 3 | disturbances and planner variants, H2 | completed | 0:07 |
| 7668108-7668110, 7668130, 7668133 | tdmpc2-horizon | 28 | horizon test: D, R1, C1, H2 | completed | 0:45 |
| 7668917, 7668921, 7668922 | tdmpc2-smoothing | 22 | smoothing test: D, R1 | completed | 0:25 |
| 7668941, 7668942 | b2-sharp-p1, -p2 | 3 + 3 | reward pilot training | completed | 2:04 |
| 7668947 | b2-snap-sharp | 3 | snapshots, reward pilot | completed | 0:16 |
| 7669541 | tdmpc2-copy-run | 3 | copy R1 seeds 0-2 for the continuation (CPU only) | completed | 0:00 |
| 7669542, 7669543, 7669544 | b2-r1x-p1, -p2, -p3 | 3 + 3 + 3 | R1 seeds 0-2 continued to 2M (Section 5.18) | completed (12:00) | 2:22 |

In total, 300 tasks completed between the launch on the evening of November 1 and 12:00 on November 2, and none failed, timed out or was cancelled.

The code versions, in order: `8edab5b` (Batch 2 launch: random starts, test starts, cluster jobs), `52873af` (resume with fresh randomness, used from R1 task 10 and C1 on), `785bfb2` (H2), `b86b66b` (horizon test), `3f9f21b` (smoothing test and reward pilot), `622b694` (R1 extension). Later commits changed only analysis scripts and documents. The training loop (`tdmpc2_jax/train.py`) last changed in `52873af`, the TD-MPC2 algorithm itself (`tdmpc2.py`, `world_model.py`) was not changed during Batch 2, and `train_hover.py` only gained the registrations of the H2 and reward-pilot environments (`785bfb2`, `3f9f21b`). The task and test code (`crazyflie_hover.py`, `hover_eval.py`, `eval_hover.py`) last changed at the launch, `8edab5b`.

# Part 3. Results summary: one answer per question of Prof. Bellinger

Source: `code/src/tdmpc2_training/reports/RESULTS_SUMMARY.md`, last modified 2026-11-02.

## Results summary: TD-MPC2 Crazyflie hover (Nov 2, 2026)

Each answer is marked:
- **Established:** several seeds, the right controls, repeated measurement.
- **Preliminary:** few seeds or a single setting; the direction is clear, the size uncertain.
- **Pending:** nothing; every experiment has finished.

Success = a full 10 s flight whose late error (mean distance to the target
over the last 2 s) is below 10 cm. Numbers are mean ± standard error across
seeds; brackets are 95% bootstrap intervals over seeds. The full explanation
of every result is in `GUIDE.md` / `GUIDE.pdf`.

### Q1. Why did short TD-MPC2 runs vary so much between seeds?

**Established.** The runs had not finished learning at 50k steps. The spread
was real, not test noise: re-flying all 33 Batch 1 agents 20 times, 93-94% of
the variance of the small agents comes from the agents themselves (ICC 0.93
and 0.94). The three published-size runs trained to 1M steps all converged
(5.7-8.6 cm). Training losses do not track the swings within a run (all
correlations of changes between -0.14 and +0.10).

### Q2. Does longer training resolve the variability?

**Established for safety; not for precision.**
- At 1M steps, no small random-start agent crashes from the training area or
  the held-out block (11 of 11 seeds).
- Precision still varies: final success from the training area ranges from 8%
  to 100% across the 11 seeds, because the steady-state error sits a few cm
  around the 10 cm threshold.
- It also drifts within a run: success from one 100k snapshot to the next
  varies 5-7 times more than test noise allows, mostly as precision changes.
- Choosing the snapshot on separate validation starts recovers most of it:
  97% test success from the training area at the validation-chosen snapshot
  vs 76% at the final one (11 seeds; +21 ± 11 points).
- The published model (R1) had not finished learning at 1M: its snapshots
  rise almost steadily (26% at 600k, 42% at 1M). Seeds 0-2 continued to 2M
  (preliminary, 3 seeds): with planning 66 -> 72% (training area), 65 -> 75%
  (held-out), and more precise, but crashes from outside starts 22 -> 58%.
  Likely cause: the 1M-transition replay buffer starts discarding the early,
  wide-ranging flights after 1M steps, so the agent specialises.
- **Reward hypothesis rejected in its simple form (pilot, 3 seeds):** with a
  reward ten times sharper within 5 cm of the target, the value network does
  learn to prefer the target (it drops by 0.03-0.05 at 10 cm, against about
  0.00 with the lab's reward), and no snapshot strayed above 20 cm (0 of 18,
  vs 9 of 66 for D), but success still swings between snapshots (SD 0.25 vs
  0.34) and the typical error is unchanged (about 9-10 cm).

### Q3. TD-MPC2 vs PPO: simulator steps and wall-clock time

**Established (Batch 1, single start).** TD-MPC2 reached 10 cm after 105k-175k
steps, PPO after 7.9M-19.7M (about 100x more). In wall-clock time PPO was
faster (about 2.6 vs 47 min) because it simulates 2,048 drones at once.
Caveat: the historical PPO had two bugs (constant learning rate; leaky
automatic reset); it is a baseline, not a tuned comparison.

### Q4. GPU use, and why TD-MPC2 is slower in wall-clock time

**Established.** One training step of the published model on a Narval A100
slice: simulate 6.8 ms, plan 10.2 ms, update 2.6 ms (49 steps/s). Many small
sequential operations for one drone leave the GPU mostly idle (25-28% busy at
home; not measurable on MIG slices).

### Q5. Parallel environments

**Established as a measured projection.** Simulating 128 drones costs the same
as one, but planning and the one-update-per-step rule scale with the number of
drones: 45 -> 74 steps/s (1.6x) for the published model, 93 -> 271 (2.9x)
for the small one. More needs fewer updates per step, a different algorithm.

### Q6. Learning from continuously randomised starts, and generalisation

**Established for both model sizes.** Success / crashes with planning,
training area / held-out block (never trained on) / outside:
- small random-start agents (D, 11 seeds): 78/0, 83/0, 59/3%; at the
  validation-chosen snapshot 97/0, 91/0, 70/2%;
- published random-start agents (R1, 11 seeds): 46/0, 39/2, 20/26%; their
  policy network alone 60/0, 64/0, 40/6%, and at its validation-chosen
  snapshot 90/0, 81/0, 59/6%;
- published single-start control (C1, 3 seeds): 26/35, 4/82, 0/92%.
- R1 - C1: +20 [+1, +39], +34 [+17, +51], +20 [+9, +32] points; D - Batch 1
  small: +59 [+35, +81], +72 [+48, +91], +55 [+38, +72].
- C1 is indistinguishable from Batch 1 (+6, +2, -2 points), so the gain comes
  from the random starts, not from the metric target offset.
- Interpolation works (held-out = training area for both sizes);
  extrapolation outside the area works partly, best for the small model.
- At 1M steps the small model beats the published one with planning (-32 to
  -44 points for R1), but R1 is still improving.

### Q7. When and why does the learned world model help?

**Established on three contrasting cases.**
- Single-start agents (Batch 1 1M and C1): planning hurts off the training
  start (-21 to -26 points; 35-66% crashes vs 15% for the policy from
  training-area starts). The planner's flights go where the world model is
  1.7-2.8x more wrong, errors peak before crashes, and restraining the planner
  reduces crashes: model exploitation.
- Small random-start agents (D): planning adds +69 [+30, +96] points over the
  policy network alone (median error 7 vs 21 cm). The model is more accurate
  along the planner's flights (ratio 0.68 ± 0.09), and restraining the
  planner only costs precision.
- Published random-start agents (R1): in between. Planning loses to the
  policy network (-15 [-37, +10], -25 [-47, -1], -20 [-40, 0]); model error
  ratio 1.17 / 1.49 / 2.00 (training area / held-out / outside); a narrower
  search, more policy candidates, a 64x2 search or a shorter horizon all help.
- So the model helps where its training data covers the situation and the
  search does not over-optimise it; a harder search (512x6) and a model still
  learning (R1 at 1M) bring exploitation back.
- The value network is flat within about 20 cm of the target for D and R1
  (value probe): the precision of planning comes from the world model's
  predicted rewards over the 3-step horizon.

### Q8. Does planning improve robustness to disturbances?

**Established for D and R1; preliminary for C1 and H2.**
- D: with planning, 72-77% success under +5-15% mass, -5-10% motor strength
  and 20-40 ms command delay, 61-68 points above the policy network alone.
  Wind: 60% at 1 m/s, about 0% from 2 m/s, no crashes up to 3 m/s. Beyond:
  the planner crashes more (41 vs 29% at 4 m/s, 67 vs 21% in 6 m/s gusts).
- R1: the planner is below its policy network under every disturbance
  (-17 to -37 points); in 3 m/s wind it crashes in 32% of flights, the policy
  network in 0%.
- Single-start agents: the planner is worse under every disturbance.
- H2 pilot (training with random mass, motors and wind): slightly fewer
  crashes in strong wind (34 vs 41% at 4 m/s) but 13% success without
  disturbance (D 78%), and the planner does not beat the policy. H2 is not
  supported in this form; the agent likely needs a history of observations to
  infer the conditions.

### Q9. Model size: speed, deployability, accuracy

**Established at 1M steps; preliminary at 2M (3 seeds).**
- Per decision, GPU slice / 2 CPU cores: published planner 6.1 / 516 ms;
  small planner 1.4 / 11.4 ms; small 64x2 planner 1.5 / 4.4 ms; policy only
  0.2-0.4 ms. Control period 20 ms.
- The small model trains 1.8x faster per step (88 vs 49 steps/s).
- Accuracy at 1M: small planner 78% from the training area vs published
  planner 46%; but the published policy network alone reaches 60% (90% at the
  validation-chosen snapshot), while the small one reaches 9%.
- Best deployable options: small model with planning (CPU, 11 ms, or 64x2 at
  4.4 ms with the same success), or the published policy network alone
  (0.3 ms, most robust to disturbances, needs checkpoint selection).
- At 2M steps (seeds 0-2) the published planner draws level with the small
  one inside the area (72-75% vs 78-83%) but crashes in 58% of flights from
  outside starts (small at 1M: 3%), after about 14 h of training vs 4 h.

### Q10. What prevents safe sim-to-real transfer?

**Established in simulation and code (nothing tested on hardware).**
- Blockers: the agents output thrust + body moments, which the firmware does
  not accept; thrust calibration.
- Measured risk: the planner's commands jump (thrust changes by 29% of its
  range per step in steady hover, 49% of steps at a limit; policy alone 10%
  and 2%). A 40 ms command filter keeps the precision (+0 ± 1 points) with
  commands as smooth as the policy's; the agents tolerate 30-60 ms motor lag,
  but filter + 60 ms lag costs 11 ± 4 points.
- Solved in simulation: a deployment adapter that rebuilds the exact
  observation from Crazyflie-style state estimates.
- Open: state estimation, radio delay and jitter, take-off and landing,
  motor identification.

# Part 4. Handoff after the Batch 2 overnight run

Source: `code/src/tdmpc2_training/reports/HANDOFF.md`, last modified 2026-11-02.

## Handoff: Batch 2 overnight run (Nov 1-2, 2026)

Status as of 12:30 on Nov 2 (all experiments finished). Everything is on the branch `tdmpc2-hover` of the
local repository (WSL: `~/Projects/multi-agentCrazyflieLift`); nothing was
pushed (origin returns 403) and nothing was merged into `main`.

Read first: `BATCH2_EXPLAINED.pdf` (14 pages, plain language), then
`GUIDE.pdf` (66 pages, everything) or `RESULTS_SUMMARY.md` (one answer per
question). `SUPERVISOR_UPDATE.md` is the one-page update for Prof. Bellinger.

### 1. What was completed

- Batch 2 design and code: random starts from a region with a held-out
  block, fixed test and validation starts, metric target offset (option),
  snapshots every 100k steps, resumable split jobs, chained follow-up tests.
- Main experiments, 1M steps each: R1 (published, random starts, 11 seeds),
  C1 (published, single start, 3), D (small, random starts, 11).
- Pilots: H2 (randomised mass, motors and wind, 3 seeds), sharp-reward pilot
  (3 seeds).
- Re-tests of Batch 1: variance decomposition (A1), loss-vs-performance
  fluctuations (A2), Batch 1 agents on the new test starts (R4).
- Follow-up tests on every Batch 2 agent: disturbances, snapshots (best on
  validation vs final), world-model error, planner variants, planning horizon,
  smoothing and motor lag, precision, flight quality, start-outcome maps.
- Speed: decision latency (GPU and CPU), training profile, parallel-drone
  projection.
- Extra tests after the main results: value-network probe (small, sharp and
  R1 agents).
- Documents: guide (PDF), plain-language summary (PDF), results summary,
  supervisor update, research log, experiment registry, deployment readiness,
  next steps, this handoff.

### 2. Code changed

72 files under `code/src/tdmpc2_training/` since the Batch 2 launch commit
`8edab5b` (about 7,000 lines added, documents included), all committed. Modified existing files:
`crazyflie_hover.py` (random starts and the metric option, off by default),
`hover_eval.py` and `eval_hover.py` (test starts, decision rules),
`tdmpc2_jax/train.py` (snapshots, resume with fresh randomness),
`tdmpc2_jax/config_hover.yaml`, `train_hover.py` (environment
registrations), `cluster/seed_job.sh`, `README.md`. The TD-MPC2 algorithm
(`tdmpc2.py`, `world_model.py`) was not changed. New: 5 Batch 2 configs, 2
environment variants (`crazyflie_hover_dr.py`, `crazyflie_hover_sharp.py`),
8 test scripts, 12 analysis scripts, 3 check scripts, the deployment adapter,
14 cluster scripts, `probe_value.py`. The full list is in Chapter 7 of the
guide; `git log 8edab5b^..HEAD` shows every commit.

### 3. Tests passed

- `check_batch2.py` (23 checks): starts, held-out exclusion, observation
  identical to the lab's with options off, metric offset, test starts.
- `check_disturbances.py` (10 checks); `check_deployment.py` (observation
  identical to the simulator over 300 closed-loop steps).
- Regression: old and new code give identical test flights for saved agents;
  identical CPU training curves.
- Validation runs: resume with fresh randomness; copy-and-continue of a
  finished run; horizon test reproduces the trained horizon; smoothing test's
  policy flights bit-identical to the disturbance tester's; sharp-reward
  environment equals the plain one plus the bonus at every step.
- A Narval test job before launch (7659264).

### 4. Tests failed

- The smoothing test's planner flights are not bit-identical to the
  disturbance tester's (first difference about 1e-7 after 2-72 steps). Cause:
  the two programs are compiled differently and round differently, which the
  planner's argmax amplifies. Statistically equivalent over 4 noise seeds;
  documented in the guide (Section 5.16). No other failures.

### 5. Experiments finished

All jobs in `GUIDE.md` Appendix B: 294 Slurm tasks completed, 0 failed,
timed out or cancelled (300 by noon, with the 2M run). Main job numbers: R1 7659438, C1 7659439, D 7661493/
7661496 and 7662436/7662437, H2 7664888/7664889, sharp pilot 7668941/7668942.

### 6. Jobs running or queued

- None. The last run, R1 continued to 2M steps (seeds 0-2; 7669542 to
  7669544), finished at about 12:00 on Nov 2, every task exit 0. Results in
  `outputs/batch2/random_start_2M/` and in the guide, Section 5.18.

### 7. Exact paths

- Results at home: `outputs/batch2/<experiment>/` (`analysis/summary.md` is
  the file to read) and `outputs/batch2/*_summary/`.
- Narval: `/scratch/mounirz/tdmpc2/batch2/<experiment>/seed_XX/`
  (checkpoints with replay buffers, snapshots, logs in `batch2/logs/`).
- Backup: `/project/def-cbelling-ab/mounirz/tdmpc2_scratch_backup_2026-11-01/`
  (`batch2/` added on Nov 2; scratch is purged after 60 days of no access).
- Documents: `code/src/tdmpc2_training/reports/`.

Experiment folders: R1 `random_start_1M`, C1 `fixed_start_1M`,
D `random_start_small_1M`, H2 `randomized_small_1M`, sharp pilot
`sharp_small_1M`, continuation `random_start_2M`.

### 8. Key findings

- Random starts work: R1 - C1 = +20/+34/+20 points (training area/held-out/
  outside), D - Batch 1 small = +59/+72/+55; no crashes inside the area.
- Best controllers (validation-chosen checkpoint): small model with planning
  97/91/70%; published policy network alone 90/81/59%, 0.3 ms per decision.
- At 1M the small planner beats the published one (78 vs 46% from the
  training area); the published model is still improving.
- Planning adds 69 points for the small agents and loses 15-25 points for R1;
  restraining R1's search helps (exploitation).
- Moderate disturbances barely affect the small planner (72-77%); wind of
  2 m/s or more defeats every agent.
- H2 and the sharp reward: negative results (see 9).
- The planner's commands jump; a 40 ms filter (small) or the averaged plan
  (both) fixes it in simulation.

### 9. Hypotheses supported or rejected

- Seed variance in Batch 1 came from short training (supported, ICC 0.93-0.94).
- Random starts give generalisation (supported, both sizes).
- H1, model exploitation explains when planning hurts (supported as one
  mechanism; restraining the search reduces but does not remove the crashes
  of single-start agents).
- H2, planning helps when the model is trained on varied physics (not
  supported in the 3-seed pilot).
- The weak reward gradient causes the precision drift (rejected in its simple
  form: the sharp reward was learned but the drift remained).
- A longer horizon helps (only for the small model, +11 ± 7 at 5 steps; never
  for R1).
- The planner needs its jumpy commands (rejected for the small model).

### 10. Professor questions answered

Q1, Q3, Q4, Q5, Q6, Q7, Q8 and Q10 (blockers and risks identified) are
answered; Q2 and Q9 are answered at 1M steps. See `RESULTS_SUMMARY.md`.

### 11. Questions still open

- Why does R1 at 2M crash more from outside starts (22% -> 58%)? Hypothesis:
  the replay buffer discards the early flights after 1M steps (to check).
- What causes the precision drift (planner sampling or continual updates)?
- Q10: everything that needs hardware (state estimation, radio delay, motor
  response, firmware-compatible commands).

### 12. Follow-ups launched and why

- Horizon test: does looking further ahead help, or do model errors compound?
- Smoothing test: the flight-quality analysis showed jumpy planner commands.
- Sharp-reward pilot: the reward barely separates 5 cm from 15 cm.
- R1 to 2M: R1 was still improving at 850k-1M.
- Value probe (home): to explain where the precision comes from, then to check
  whether the sharp reward changed the value.

### 13. What next

`NEXT_STEPS.md` and Chapter 9 of the guide. In short: check the replay-buffer
explanation of the 2M result, adopt validation-based checkpoint selection, test the source of the drift,
then firmware-compatible commands before any hardware step.

### 14. What needs Mounir

- Read `BATCH2_EXPLAINED.pdf`, then send or adapt `SUPERVISOR_UPDATE.md`.
- Decide with Prof. Bellinger: hardware path now or more simulation; which
  controller to carry forward; whether a fair PPO comparison is needed.
- Pushing the branch (origin returns 403: needs his GitHub access or a fork).

### 15. Safety and reproducibility concerns

- No hardware was touched: no flight, no firmware, no motor command. Every
  hardware step in the documents is conditional on Prof. Bellinger's approval
  and the lab's safety rules.
- GPU training is not bit-for-bit repeatable; results are distributions over
  seeds. The CPU is the exact referee for code changes.
- C1, H2 and the reward pilot have 3 seeds: directions, not sizes.
- The 10 cm success threshold matters; success for every threshold is
  reported alongside.
- Narval scratch is purged after 60 days without access: checkpoints are
  backed up to `/project` (all of Batch 2, including the 2M run, verified file by file).
- The historical PPO results were not changed.

# Part 5. Next research plan (as of November 2)

Source: `code/src/tdmpc2_training/reports/NEXT_STEPS.md`, last modified 2026-11-02.

## Next research plan (after Batch 2, Nov 2, 2026)

Ordered by expected research value per unit of effort. Each step says which
result would change the plan. The reasoning behind each is in `GUIDE.md`,
Chapter 9.

### 0. Check why longer training hurt outside the area (about 1 h, no training)

R1 at 2M (seeds 0-2) improved inside the area but crashed far more from
outside starts (22% -> 58%). Hypothesis: the 1M-transition replay buffer
starts discarding the early, wide-ranging flights after 1M steps. Compare the
distances to the target stored in the 1M and 2M buffers (in the checkpoints
on Narval), and fly the 1.1M-2M snapshots.
- **Decision:** if confirmed, use a buffer of at least the run length for
  long runs; either way, carry the small model forward (it matches R1 at 2M
  inside the area and is far safer outside, at a quarter of the training time).

### 1. Make checkpoint selection on validation starts the standard (no training)

Every run already keeps snapshots. Report each run at its validation-chosen
snapshot as well as at its final one. This took the small model from 76% to
97% and the published policy network from 60% to 90%.

### 2. Find the source of the precision drift (about 2 h, no training)

The reward pilot ruled out the flat value function. Two cheap tests:
- fly one snapshot with several planner random seeds: if the hover position
  changes with the seed, the planner's sampling is the cause;
- lower the learning rate for the last 200k steps of one run: if the drift
  shrinks, the continual updates are the cause.

### 3. Tame the published planner (about 8 h on Narval per variant, 3 seeds)

Its 512x6 search loses to its own policy network and to a 64x2 search. Train
with a smaller search, or with more candidates from the policy network, and
compare with R1 at the same steps. Only after step 0.

### 4. Randomised physics with a memory (about 8 h, 3 seeds)

H2 lost precision, probably because the agent cannot infer which mass or wind
it is flying in. Add the last few observations and commands to the
observation (or an estimate of the wind and mass), then repeat H2 and its
disturbance tests. This is also what a real drone needs against wind.

### 5. Firmware-compatible commands (the gate to any hardware work)

Retrain with collective thrust plus body-rate setpoints, with the firmware's
rate controller modelled in the simulator, and repeat the Batch 2 tests. Then
the supervised steps of `DEPLOYMENT_READINESS.md` (motor identification,
radio-only test, tethered hover), each with Prof. Bellinger's approval.

### 6. Optional

- A fair PPO comparison (bugs fixed, random starts), if the thesis needs it.
- Parallel drones with fewer updates per collected step, reported as a
  variant of TD-MPC2.

# Part 6. Supervisor update (November 2)

Source: `code/src/tdmpc2_training/reports/SUPERVISOR_UPDATE.md`, last modified 2026-11-02.

## Batch 2 update for Prof. Bellinger

Hi Professor Bellinger,

Here is where the project stands after Batch 2. Following your feedback, every agent now trains from random starting positions drawn from a region (with one block held out for testing), runs are 1M steps, and the main experiments have 11 seeds (3 for the controls and pilots). Every agent is tested on the same 128 flights: the training area, the held-out block, starts up to 0.75 m outside the area, and the old Batch 1 starts. Everything is in simulation; nothing has been flown.

### What I found

- 1. **Seed variance in Batch 1 was real, and the runs were too short.** Re-flying every Batch 1 agent 20 times, 93-94% of the variance of the small agents comes from the agents themselves (ICC 0.93-0.94), not from test noise. At 1M steps the random-start seeds agree on safety (no crashes inside the area), but their precision still varies by a few centimetres around the 10 cm threshold, between seeds and from one checkpoint to the next.

- 2. **Random starts work, for both model sizes.** With planning, success / crashes from the training area, held-out block and outside:
   - published TD-MPC2, random starts (R1, 11 seeds): 46/0%, 39/2%, 20/26%;
   - the same from the single Batch 1 start (C1, 3 seeds): 26/35%, 4/82%, 0/92%;
   - small TD-MPC2, random starts (D, 11 seeds): 78/0%, 83/0%, 59/3%.

   C1 reproduces Batch 1, so the gain comes from the random starts. The held-out block does as well as the training area (interpolation works); outside the area is harder.

- 3. **Checkpoint selection matters.** Choosing each run's checkpoint on separate validation starts gives 97% / 91% / 70% success for the small model with planning, and 90% / 81% / 59% (no crashes inside the area) for the published model's policy network used alone, at 0.3 ms per decision.

- 4. **When the world model helps.** Planning adds 69 points for the small random-start agents, whose world model is accurate along the planner's flights. It hurts where the planner can find the model's errors: single-start agents away from their start (the planner's flights go where the model is 2-3 times more wrong, and restraining the search reduces crashes), and, to a lesser degree, the published random-start agents, whose 512×6 search loses to their own policy network (-15 to -25 points) and to a smaller 64×2 search. A probe of the value network shows it is almost flat within 20 cm of the target, so the planner's precision comes from the world model's predicted rewards over the short horizon.

- 5. **Disturbances.** The small agents with planning keep 72-77% success under extra mass (up to 15%), weaker motors (up to 10%) and 20-40 ms command delay. Wind of 2 m/s or more defeats every agent trained without wind, and in strong wind the planner crashes more than the policy network alone. Training with randomised mass, motors and wind (3-seed pilot) cost most of the precision (13% success) for slightly fewer crashes in strong wind; I think the agent needs a short history in its observation to infer the conditions.

- 6. **Precision drift.** I tested whether the lab's reward, which pays almost the same for 5 cm and 15 cm, causes the drift: with a reward ten times sharper near the target (3-seed pilot), the value network learned to prefer the target, but the precision still drifted between checkpoints. So the reward is not the main cause; checkpoint selection on validation starts is the practical fix.

- 7. **Model size and speed.** Per decision on a GPU slice / 2 CPU cores: published planner 6.1 / 516 ms, small planner 1.4 / 11.4 ms, policy network alone 0.2-0.4 ms (the control period is 20 ms). The small model also trains 1.8 times faster. At 1M steps it is the more precise planner. Because the published model was still improving, I continued three R1 seeds to 2M steps: inside the area they improved (66% to 72% success with planning, and more precise), but crashes from starts outside the area rose from 22% to 58%. My best explanation is the replay buffer, which holds 1M transitions: after 1M steps it starts discarding the early, wide-ranging flights, so the agent specialises on its training area. So the published model was partly just slower, but even at 2M it does not beat the small model trained for 1M.

- 8. **GPU use and parallel environments.** With one drone, a training step is many small sequential operations (simulate 6.8 ms, plan 10.2 ms, update 2.6 ms), so the GPU is mostly idle. Simulating 128 drones costs the same as one, but planning and one update per step scale with the number of drones, so the projected speed-up is only 1.6 times (published) to 2.9 times (small).

- 9. **Toward a real drone.** Two blockers: the agents output thrust and body moments, which the Crazyflie firmware does not accept (they would have to be retrained with rate setpoints), and the thrust calibration. In simulation, the planner's commands jump by about 30% of their range per step; a 40 ms command filter (small model) or acting on the planner's averaged plan (both models) removes this without losing precision.

### Limitations

All results are from simulation with a rigid-body model and simple sensor noise. C1, H2 and the reward pilot have 3 seeds each. The 10 cm success threshold matters because the agents' error often sits between 5 and 15 cm, so I also report success for every threshold. TD-MPC2 was used with its published settings, without tuning. The PPO comparison from Batch 1 used the lab's PPO, which has two bugs I found (a learning rate that never decreases and a leaky reset); I kept it as a historical baseline.

### Questions for you

- Should I move toward a real flight now (which first means retraining with firmware-compatible commands) or deepen the simulation study first?
- Which controller should I carry forward: the small model with planning, or the published model's policy network alone?
- Do you want a fair PPO comparison (bugs fixed, random starts) for the thesis?

The full write-up, with every figure and a walkthrough of the code, is in `code/src/tdmpc2_training/reports/GUIDE.pdf`, and a short version in `BATCH2_EXPLAINED.pdf`, on the `tdmpc2-hover` branch.

Mounir

# Part 7. Batch 2 in plain language

Source: `code/src/tdmpc2_training/reports/BATCH2_EXPLAINED.md`, last modified 2026-11-02.



## 1. The story in one page

**The goal of the project.** A Crazyflie drone in simulation must fly to a target point 1 m above the ground and hover there, within 10 cm. The controller is learned with **TD-MPC2**, a reinforcement learning method that learns a small model of how the drone moves (a "world model") and uses it to plan.

**What went wrong before (Batch 1).** In Batch 1, every training flight started from the **same spot**, (0, 0, 0.2). Most runs were short (50,000 steps). Professor Bellinger pointed out two problems: the results varied a lot between seeds, and a controller that always starts from one spot may just memorise one route instead of learning to fly.

**What Batch 2 changed.** Following his feedback:

- every training flight starts from a **new random spot** in a 1.5 m by 1.5 m area around the target;
- every run trains for **1 million steps** (20 times longer than most of Batch 1);
- every experiment has **11 seeds** (3 for the small control and pilot experiments);
- every agent is tested on the **same 128 test flights**, including spots it never trained from.

**The main answer.** Training from random spots works. The old single-spot agents crashed in most flights away from their one spot; the new agents almost never crash inside the area, and they also fly well from a block of spots they never trained from. The best agents reach the target (within 10 cm) in 90-97% of the flights from the training area.

**Two surprises.**

- The **small** version of TD-MPC2 learned better than the big (published) one in 1M steps. The big one was still improving at 1M, so three of its runs were continued to 2M. At 2M it is about as good as the small model inside the training area, but it crashes much more from spots outside it (see Section 4.9).
- For the big model, the **policy network alone** (no planning) flies better than with planning, and at its best checkpoint it is as good as the best small agents, while needing only 0.3 ms per decision.

## 2. The words you need

### 2.1 Basic terms

- **Step.** One decision of the drone: it looks at its sensors, chooses a thrust and three turning forces, and the simulator moves it forward 0.02 s. **1M steps = 1 million decisions**, about 5.5 hours of simulated flying, made of thousands of practice flights. One run of 1M steps took about 7 hours of computing on Narval.
- **Seed.** One independent training run. The same experiment is repeated with 11 different seeds (random number choices) to see how much the result depends on luck. Results are averages over seeds.
- **Flight / test flight.** One 10-second flight from one starting spot, used only to measure, not to learn.
- **Success.** The drone flies the full 10 s and, over its last 2 s, stays on average **within 10 cm** of the target.
- **Crash.** The drone hits the ground or flies far away before the 10 s are over.
- **Checkpoint / snapshot.** A saved copy of the agent during training. Batch 2 saved one every 100,000 steps, so we can test the agent at different moments of its training, not only at the end.

### 2.2 The two ways a TD-MPC2 agent can fly

This distinction appears everywhere in the results:

- **With planning** (the normal TD-MPC2 way). At every step the agent imagines hundreds of possible short futures with its world model and picks the best one. Slower (11 ms per decision for the small model on a CPU, 516 ms for the big one), but can be more precise.
- **Policy network only.** The agent simply uses its policy network, which outputs an action directly, with no imagining. Very fast (0.2-0.4 ms per decision). TD-MPC2 trains this network anyway; we test it alone to see what planning adds.

There is also a third way, the **64×2 planner**: planning with a much smaller search (64 candidates, 2 rounds), which is faster.

### 2.3 The four groups of test flights

Every agent flies the same 128 test flights, from four groups of starting spots:

| Group | Number of flights | What it checks |
|---|---:|---|
| **training area** | 50 | random new spots of the kind it trained on |
| **held-out block** | 30 | spots inside the area but in a corner it was **never** allowed to train from: can it fill in the gap? |
| **outside** | 30 | spots 0.25-0.75 m **outside** the area: can it go beyond what it saw? |
| **Batch 1 spots** | 18 | the six old Batch 1 starting spots, to compare with before |

Every result table below gives **success / crash** shares for these groups.

### 2.4 Model sizes

- **Published (big) model:** the TD-MPC2 settings from the paper (5 million weights).
- **Small model:** a reduced version (0.39 million weights) with a smaller planning search. It trains about twice as fast and can plan on a laptop CPU.

## 3. The experiments and their names

Every experiment has a short name. This table is the key to everything else:

| Name | Model | Where training flights start | Seeds | Steps | Why it was run |
|---|---|---|---:|---:|---|
| **B1** (Batch 1, old) | big and small | always the same spot | 11 (+3 big at 1M) | 50k (1M for 3) | the "before" picture |
| **R1** | big | random spots | 11 | 1M | **the main experiment**: the professor's suggestion with the published TD-MPC2 |
| **C1** | big | always the Batch 1 spot | 3 | 1M | a **control** for R1 (see below) |
| **D** | small | random spots | 11 | 1M | the same as R1 with the small model |
| **H2** | small | random spots, plus random weight, motor strength and wind | 3 | 1M | pilot: does training with varied physics make it more robust? |
| **SH** | small | random spots, with an extra reward near the target | 3 | 1M | pilot: does a sharper reward make it more precise? |
| **R1 to 2M** | big | random spots | 3 | 2M | is the big model just slower? |

**Why C1 exists.** Besides the random spots, R1 has one small technical difference from Batch 1 (the target position is given to the agent in metres). If R1 is better than Batch 1, is that because of the random spots, or because of this small difference? C1 has the small difference but trains from the single Batch 1 spot. If C1 looks like Batch 1, the improvement in R1 comes from the random spots. (It does: C1 is within 2-6 points of Batch 1.)

**Folder names.** On disk the experiments have longer names:

| Name | Folder in `outputs/batch2/` |
|---|---|
| R1 | `random_start_1M` |
| C1 | `fixed_start_1M` |
| D | `random_start_small_1M` |
| H2 | `randomized_small_1M` |
| SH | `sharp_small_1M` |
| R1 to 2M | `random_start_2M` |

## 4. The results, one question at a time

All numbers are **success / crash** shares, averaged over the seeds, for the final agents (after 1M steps) unless said otherwise.

### 4.1 Do random starts help? Yes, clearly.

Big model, with planning:

| | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| B1, big, one spot, 1M (old) | 20% / 66% | 2% / 91% | 2% / 87% |
| C1, big, one spot, 1M (control) | 26% / 35% | 4% / 82% | 0% / 92% |
| **R1, big, random spots** | **46% / 0%** | **39% / 2%** | **20% / 26%** |

Small model, with planning:

| | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| B1, small, one spot, 50k (old) | 19% / 9% | 11% / 15% | 4% / 31% |
| **D, small, random spots** | **78% / 0%** | **83% / 0%** | **59% / 3%** |

**In words.** Agents trained from one spot crash most of the time when they start somewhere else. Agents trained from random spots almost never crash inside the area, and they do as well in the corner they never trained from (the held-out block) as in the rest of the area. Outside the area is harder, but still much better than before.

![Success of the final agents with planning, on the same test flights for every experiment (bars: average over seeds; black lines: uncertainty). Blue = R1, orange = C1, green = D (small model), red = H2, purple = old Batch 1 big agents.](code/src/tdmpc2_training/reports/figures/r1_comparison.png)

![Every test starting spot seen from above, coloured by how many seeds succeeded from it (green = most seeds succeed, red = none; a black ring = most seeds crashed). Top: R1, trained from random spots. Bottom: C1, trained from one spot. Left: with planning; right: policy network only. The blue square is the training area, the orange corner the held-out block, the star the target.](code/src/tdmpc2_training/reports/figures/start_outcomes_R1_C1.png)

### 4.2 Which model is better? The small one, for now.

| With planning | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| R1 (big) | 46% / 0% | 39% / 2% | 20% / 26% |
| D (small) | 78% / 0% | 83% / 0% | 59% / 3% |

The small model is 32-44 points better. But the big model's results were **still going up** at the end of training (its test success rose from about 26% at 600k steps to 42% at 1M), so it may simply need more steps. That is why three R1 runs were continued to 2M steps; Section 4.9 gives the answer.

### 4.3 Does planning help? It depends on the model.

| | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| D (small), with planning | 78% / 0% | 83% / 0% | 59% / 3% |
| D (small), policy network only | 9% / 0% | 9% / 0% | 12% / 0% |
| R1 (big), with planning | 46% / 0% | 39% / 2% | 20% / 26% |
| R1 (big), policy network only | 60% / 0% | 64% / 0% | 40% / 6% |

- **Small model:** planning is essential. The policy network alone never crashes but usually stops about 20 cm from the target; planning brings it to about 7 cm.
- **Big model:** planning makes it **worse**, especially outside the area, where it crashes 26% of the time against 6% without planning. A smaller search (64×2) is also better than the full one. This fits a known weakness of planning with a learned model, called **model exploitation**: a strong search finds the places where the model is wrong in an optimistic way, and goes there. Measurements support it: along the planner's flights the big model's predictions are 1.2-2 times more wrong than along the policy network's flights, and restraining the search adds 8-13 points.

### 4.4 Keeping the best checkpoint instead of the last one

During training, precision drifts by a few centimetres from one checkpoint to the next, and since the success line is at 10 cm, a few centimetres can flip many flights. So each run's checkpoints were tested on 40 separate **validation** flights, and the best one was kept. Its results on the test flights:

| Best checkpoint (chosen on validation) | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| D (small), with planning | **97% / 0%** | **91% / 0%** | **70% / 2%** |
| R1 (big), policy network only | **90% / 0%** | **81% / 0%** | **59% / 6%** |
| R1 (big), with planning | 49% / 0% | 46% / 1% | 26% / 32% |

These are the two best controllers of the project so far. The second one is notable for a real drone: it needs no planning at all, so it decides in 0.3 ms on any computer.

### 4.5 Disturbances (wind, weight, weaker motors, delay)

Tested on the 50 training-area flights:

- **Extra weight (+5 to +15%), weaker motors (-5 to -10%), radio delay (20-40 ms):** the small model with planning barely changes (72-77% success, against 77% without disturbance).
- **Wind:** 1 m/s still works (60%). From 2 m/s on, no agent stays within 10 cm, but up to 3 m/s the small model is pushed away without crashing. At 4 m/s many flights crash.
- **Big model (R1):** the planner is worse than its policy network under every disturbance; in 3 m/s wind the planner crashes in 32% of the flights, the policy network in 0%.

### 4.6 The two pilots (3 seeds each, so only indications)

**H2, training with varied physics.** Every training flight had a random weight, motor strength and wind. Result: a little more robust in strong wind (fewer crashes at 4 m/s: 34% against 41%), but much less precise in calm air (13% success against 78% for D). The likely reason: the agent sees only its current state, so it cannot tell which weight or wind it is flying in, and its world model learns an average that is wrong in every particular case. Not a good trade in its current form.

**SH, a sharper reward near the target.** The lab's reward pays almost the same for hovering 5 cm or 15 cm from the target, which could explain why precision drifts. SH adds a reward that is concentrated within about 5 cm. Results: the agent did learn it (its value network now clearly prefers the target, which the normal agents do not), and the big precision swings disappeared, but the precision still drifts around the 10 cm line, and the final checkpoints happened to be worse (37% from the training area; 91% at the best checkpoint). So the reward was **not** the main cause of the drift. Useful to know, but not a fix.

### 4.7 Smooth commands for a real drone

With planning, the commands jump a lot from one step to the next (thrust changes by about 30% of its range every step), which real motors cannot follow. Two simple fixes were tested:

- For the **small model**, smoothing the commands over 40 ms keeps exactly the same success (79%) and makes them as smooth as the policy network's.
- Using the planner's **averaged plan** instead of its single best candidate keeps the success for both models (+1 to +2 points) and removes the commands stuck at their limits.

### 4.8 Speed

| Decision time | GPU (Narval) | CPU (2 cores) |
|---|---:|---:|
| big model, planning | 6.1 ms | 516 ms (too slow for 20 ms control) |
| small model, planning | 1.4 ms | 11.4 ms |
| policy network only | 0.3 ms | 0.2 ms |

A real drone needs a decision every 20 ms. The big model can only plan in time with a GPU; the small model can on a laptop; the policy network alone can anywhere.

### 4.9 Training the big model twice as long (R1 to 2M)

Three R1 runs (seeds 0, 1, 2) were continued from 1M to 2M steps. Same three seeds, before and after:

| Big model, seeds 0-2 | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| with planning, 1M | 66% / 0% | 65% / 2% | 33% / 22% |
| with planning, 2M | **72% / 3%** | **75% / 7%** | 24% / **58%** |
| small model at 1M (for comparison) | 78% / 0% | 83% / 0% | 59% / 3% |

**Better where it trains, worse where it does not.** Inside the area it gained 6-10 points and became more precise (one seed now hovers 3 cm from the target, the best of the project). From spots outside the area, crashes went from 22% to 58%.

**Why more training can make it worse outside.** TD-MPC2 learns from a memory (the **replay buffer**) of its last 1 million steps; older steps are deleted to make room. During the first million steps nothing is deleted, so the memory still holds the clumsy early flights, where the drone drifted far away and had to recover. During the second million, those early flights are pushed out and replaced by flights of a good drone going neatly from the area to the target. The agent then trains only on "nice" flights and forgets how the drone behaves far away, which is exactly what the outside spots require. It became a specialist. (This explanation is a hypothesis that fits the results; it can be checked by looking at what is stored in the memory at 1M and at 2M.)

**What it means for you.** The big model is not broken: inside the area, at 2M, it is one of the best agents. But it is not worth it for this task: the small model at 1M is about as good inside the area, far safer outside, and took about 4 hours to train instead of about 14. If longer training is ever needed, use a larger memory, a larger training area, or choose checkpoints with validation flights that include far spots. With 3 seeds this is a strong hint, not a proven result.

## 5. What it means, and what to tell Professor Bellinger

- 1. **His suggestion worked.** Training from random spots turns a controller that memorised one route (and crashed elsewhere) into one that flies from anywhere in the area, including a corner it never trained from. C1 proves it is the random spots that matter.
- 2. **Longer training fixed safety, not all of the precision.** At 1M steps no good agent crashes inside the area, but precision still drifts by a few centimetres between checkpoints. Keeping the best checkpoint (chosen on separate validation flights) solves this in practice: 97% success for the small model.
- 3. **When planning helps.** Planning helps when the world model is accurate where it is used (small model, inside the area: +69 points). It hurts when the search can find the model's mistakes (single-spot agents away from their spot; the big model outside the area).
- 4. **The small model is the better choice.** Better results in 1M steps, twice as fast to train, and it can plan on a laptop. Trained to 2M steps, the big model draws level with it inside the area but crashes far more from outside it, after three to four times the training time.
- 5. **For a real drone:** the commands must be smoothed (a 40 ms filter or the averaged plan), and the agents must be retrained with commands the Crazyflie firmware accepts; nothing was flown.
- 6. **Two negative pilot results** (varied physics, sharper reward) are reported as they are; both say where to look next.

## 6. Where everything is

### 6.1 Opening the right folder

All the Batch 2 work is on your computer, inside WSL, on the git branch **`tdmpc2-hover`**. It is **not on GitHub**: pushing was refused (error 403), so GitHub and the `main` branch only show the old students' code. In VS Code: **File, Open Folder**, then paste

```
\\wsl.localhost\Ubuntu\home\mounirzouhari\Projects\multi-agentCrazyflieLift
```

The bottom-left corner of VS Code must show the branch `tdmpc2-hover`.

### 6.2 The folders

| What | Where |
|---|---|
| **The TD-MPC2 code (new)** | `code/src/tdmpc2_training/` |
| The documents (this one, the full guide, summaries, log) | `code/src/tdmpc2_training/reports/` |
| The figures | `code/src/tdmpc2_training/reports/figures/` |
| The Batch 2 results (numbers, flight records, plots) | `outputs/batch2/` |
| The old students' code (untouched) | `code/src/gpu_training/`, `code/src/hardware_model_testing/`, `code/src/initial_cpu_training/` |
| The trained agents (too big to copy home) | on Narval: `/scratch/mounirz/tdmpc2/batch2/` |

### 6.3 How a results folder is organised

Each experiment folder in `outputs/batch2/` (for example `random_start_1M/` for R1) contains:

- `analysis/summary.md`: **the one file to read**: all tables for that experiment (final results, comparisons, disturbances) in plain text;
- `analysis/*.png`: its plots;
- `seed_00/` to `seed_10/`: one folder per seed, with the raw records of every test (`evaluation/` with planning, `evaluation_policy_only/`, `disturbances/`, `snapshot_evaluation/`, and so on). Each has a `summary.csv` (one line per test flight) and sometimes `trajectories.csv` (every step of every flight). You do not need to open these; the summaries are built from them.

The summaries that compare experiments are in `outputs/batch2/*_summary/` (for example `snapshot_summary/` for the best-checkpoint results, `smoothing_summary/` for the smooth commands).

### 6.4 The documents

| Document | What it is |
|---|---|
| `BATCH2_EXPLAINED.pdf` | this document: the short, plain version |
| `GUIDE.pdf` | the full textbook-style guide: every experiment in detail, every code file explained, the answers to the professor's ten questions (being finished now) |
| `RESULTS_SUMMARY.md` | one short answer per professor question |
| `RESEARCH_LOG.md` | what was done when, in order, with every number |
| `DEPLOYMENT_READINESS.md` | what stands between these agents and a real Crazyflie |

## 7. Status

Everything is finished: 300 jobs on Narval, none failed. All results are backed up on Narval's permanent storage (`/project/def-cbelling-ab/mounirz/tdmpc2_scratch_backup_2026-11-01/batch2/`, because `/scratch` is cleaned automatically). The full guide (`GUIDE.pdf`), the one-page update for Professor Bellinger (`SUPERVISOR_UPDATE.md`) and the handoff notes (`HANDOFF.md`) are complete.

# Part 8. Experiment log: task, tests, launches

Source: `code/src/tdmpc2_training/EXPERIMENTS.md`, last modified 2026-11-02.

## TD-MPC2 Crazyflie hover: experiments

Mounir Zouhari, honours research with Prof. Colin Bellinger. This file lists
every experiment: the question it answers, exactly how it was run, where the
results are, and how it was checked. `CONTEXT.md` has the full project
background; `README.md` has the commands.

### Research question

Does a controller that learns a model of the world (TD-MPC2) cope better with
situations it never trained on than one that learns by practice alone, and
when and why? Situations: new start positions, external disturbances (wind,
extra weight, weaker motors, delays). Alongside: is TD-MPC2 fast enough to
fly a real Crazyflie, and can a cheaper version of it be accurate enough?

The key comparison inside TD-MPC2: the same trained agent flown **with**
planning (it imagines action sequences with its world model before acting)
and **without** planning (its policy network alone). Same weights, so any
difference comes from planning with the model.

### The task (all experiments)

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

### How every agent is tested (`hover_eval.py`)

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

### Experiments

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
| Can a smaller model be accurate enough? | D (small model, random starts, 11 seeds); R1 agents flown with the policy network only and with a 64x2 planner |
| PPO learning rate | not tested: PPO is kept as a historical baseline (the project is about TD-MPC2). The audit found that its learning rate never decayed (schedule in environment steps, optimiser counting updates), the likely cause of the late degradation; see CONTEXT.md |

### Measurements already made (home PC, Nov 1)

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

### Validation before launch

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

### Launch log

**Nov 1, 2026.**
- **Backup:** Narval scratch (all Batch 1 runs, 916 files) copied to
  `/project/def-cbelling-ab/mounirz/tdmpc2_scratch_backup_2026-11-01/`, with
  contents verified identical, before the 60-day scratch purge.
- **Code:** the Narval copy was synced with its git history; it reports
  commit `8edab5b` with no uncommitted changes.
- **Test job 7659264** (R1 settings, 4,000 steps, seed 0,
  `batch2_validation/narval_test/`): completed in 15 min, exit 0.
  - It recorded commit `8edab5b`, clean.
  - Checkpoint, snapshots, the 21-flight learning curve and the three final
    tests (128 flights each) were all produced.
  - Training 4,000 steps took 6.5 min including two learning-curve tests; the
    three final tests took 8.5 min together.
- **GPU utilisation on Narval:** `nvidia-smi` cannot report utilisation for
  an A100 slice (MIG); it prints `[N/A]`, and `memory.used` is the whole
  A100's. The time breakdown of experiment P is the measure used instead.
- **Submitted about 20:45 EST:**
  - R1 = job 7659438 (tasks 0-10) and C1 = job 7659439 (tasks 0-2): 10 h
    limit, started only after the test job succeeded (`afterok`);
  - R4 = job 7659440 (tasks 0-24, one per Batch 1 run);
  - P = job 7659441.

  Logs: `$SCRATCH/tdmpc2/batch2/logs/`. Job IDs:
  `$SCRATCH/tdmpc2/batch2/job_ids.txt`.
- **Start times:**
  - R4 finished by 21:20 (25 runs, about 2 min each).
  - R1 tasks 0-9 started at 21:38 with commit `8edab5b`.
  - R1 task 10 and C1 started at 21:44 with commit `52873af`, which was synced in between.

  `52873af` differs only by new scripts and by reseeding a run that is
  *resumed* from a checkpoint. That is not used by a fresh run, and on the
  CPU fresh runs were checked to be identical. Each run records its commit
  in its log, and each batch in `batch_info.json`.
- **Disturbance tests**, queued so each starts when its training finishes:
  - job 7660609: the three Batch 1 1M-step agents;
  - job 7660720: R1, each task waiting for the same R1 task (`aftercorr`);
  - job 7660721: C1, likewise.
- **P (job 7659441, 30 min)**, A100 2/7 slice, native Linux
  (`outputs/batch2/profiling_narval/`):
  - **Time per control decision**, one drone, median / 99th percentile /
    worst: published network with the 512x6 planner 6.1 / 7.0 / 11.7 ms on
    the GPU slice and 516 ms on 2 CPU cores; policy network only 0.37 ms
    (GPU) and 0.20 ms (CPU); small network with the 128x3 planner 1.4 ms
    (GPU) and 11.4 ms (CPU).
  - The home RTX 3060 under WSL2 measured 10.7 ms for the published
    planner, so WSL adds overhead.
  - **One training step, published settings:** simulating the drone
    6.8 ms, planning 10.2 ms, learning update 2.6 ms (overlapping), 49
    steps/s. Small settings: 88 steps/s.
  - These speeds match Batch 1 (89 and 42-49 steps/s).

### First results: Batch 1 agents on the Batch 2 test starts (R4)

Success = full 10 s flight with late error below 10 cm; mean across seeds,
success / crashes (`outputs/batch1_standard_1M/analysis/summary.md`):

| Agents | Decision rule | train_area | held_out | outside | batch1 starts |
| --- | --- | --- | --- | --- | --- |
| published, 1M steps (3 seeds) | with planning | 20% / 66% | 2% / 91% | 2% / 87% | 20% / 59% |
| published, 1M steps | policy network only | 46% / 15% | 22% / 54% | 0% / 77% | 33% / 6% |
| published, 50k steps (11 seeds) | with planning | 0% / 98% | 0% / 98% | 0% / 100% | 0% / 87% |
| small, 50k steps (11 seeds) | with planning | 19% / 9% | 11% / 15% | 4% / 31% | 18% / 6% |
| small, 50k steps | policy network only | 4% / 6% | 4% / 1% | 0% / 17% | 2% / 2% |

Agents trained from a single start mostly fail from other starts, as Batch 1
showed. The new observation is that the 1M-step agents crash far more often
**with** planning than with their policy network alone on starts they never
trained from (66% vs 15% crashes on training-area starts).

A hypothesis to test with R1: the world model is wrong in unfamiliar states,
and the planner then picks actions that the model wrongly predicts to be good
("model exploitation"). The small 50k agents show the opposite on precision:
planning gives more successes. So whether planning helps depends on the
agent and on the situation, which is what R1, C1 and the disturbance tests
measure.

### Next: disturbance tests (D)

`eval_disturbances.py` flies finished agents (R1, C1 and the Batch 1
1M-step agents) from the 50 `train_area` starts under disturbances they never
trained with. Each is flown at several strengths, with planning and with the
policy network only. Every condition uses the same starts and sensor noise
(noise seed = training seed + 3000), so only the disturbance changes:

| Disturbance | Strengths | Real-life equivalent |
| --- | --- | --- |
| steady wind along +x (simulator air model, air density 1.225 kg/m3) | 1, 2, 3, 4 m/s | a fan or an air-conditioning draft |
| gust: the same wind for 1 s, 3 s into the flight | 2, 4, 6 m/s | someone walking past, a door |
| extra mass and inertia | x1.05, 1.10, 1.15 | an expansion deck, a marker, a camera |
| weaker motors (thrust and moments) | x0.95, 0.90 | battery voltage drop, worn propellers |
| commands applied late | 1, 2 steps (20, 40 ms) | radio lag |

The simulated drone needs 76% of its maximum thrust to hover, so the mass and
motor changes are kept small. `check_disturbances.py` checks the physics with
a scripted pilot:

| Check | Result |
| --- | --- |
| no disturbance | flights identical to the normal tests, with the scripted pilot and with a trained agent with its planner |
| plain hover command | holds the drone exactly |
| wind | pushes the drone 0.12 m in 1 s at 1.5 m/s and 0.42 m at 3 m/s (drag grows about with speed squared) |
| +15% mass | sinks 0.62 m in 1 s (physics predicts 0.64 m) |
| -10% motors | sinks 0.48 m in 1 s (0.49 m predicted) |
| delays | exact |

The air model treats the drone as a box of similar size, not a real
quadrotor with propellers, so the wind force is the right order of magnitude
but not exact. That is fine for comparing agents, which all face the same
wind; a fan and a wind meter on the real drone could calibrate it later.

### Reproduce

From the repository root (home):

```
.venv/bin/python code/src/tdmpc2_training/check_batch2.py
.venv/bin/python code/src/tdmpc2_training/train_hover.py --config-name=config_batch2_random_start 
    seed=0
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir>             # all test groups
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --no-mpc    # policy network 
    only
```

On Narval: `cluster/seed_job.sh`, `cluster/eval_job.sh` and
`cluster/profile_job.sh`; the exact submit commands are in their headers.

# Part 9. Code README: commands

Source: `code/src/tdmpc2_training/README.md`, last modified 2026-11-02.

## One-drone TD-MPC2 hover baseline

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

### Ubuntu/WSL setup

From the repository root, with your existing Python 3.11 environment:

```
~/.local/bin/uv pip install --python .venv/bin/python -r code/src/tdmpc2_training/requirements.txt
```

This can add packages to the existing GPU environment. Do not install into
the Windows `.venv`; the MJX GPU dependencies are for Linux.

### Test the environment without training

```
.venv/bin/python code/src/tdmpc2_training/check_hover.py
```

This prints observation/action dimensions, converted thrust, reward and height.
It does not create a trained model.

### Training

Smoke test (collects random experience, checks imports and replay buffer):

```
.venv/bin/python code/src/tdmpc2_training/train_hover.py max_steps=10 seed_steps=500
```

First learning test (includes planner and model updates):

```
.venv/bin/python code/src/tdmpc2_training/train_hover.py max_steps=700
```

Full initial experiment:

```
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

### Testing a saved agent

```
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run folder>
```

Rebuilds the agent from the run's config, loads its newest checkpoint, and
flies three flights from each start in `hover_eval.TEST_STARTS` (the training
start, 0.5 m to each side, and above the target). Nothing is trained. Writes
`evaluation/summary.csv` and `evaluation/trajectories.csv` in the run folder.

### Several seeds

```
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
| `crazyflie_hover.py` | Gymnasium adapter; `place_drone` moves the start/target; Batch 2 start area and sampler |
| `hover_eval.py` | Test flights (all flights batched on the GPU), metric definitions, fixed test starts |
| `eval_hover.py` | Load a checkpoint and test it (any test group, with or without planning) |
| `run_seeds.py` | Train + test a list of seeds |
| `analyze_seeds.py` | Plots and summary for a batch |
| `tdmpc2_jax/build.py` | Builds the agent; shared by training and testing |
| `check_batch2.py` | Automatic checks of the Batch 2 task code |
| `bench_latency.py` | Time per control decision (deployment), any network size and planning budget |
| `profile_training.py` | Where one training step spends its time |
| `EXPERIMENTS.md` | Every experiment: question, settings, folders, validation |

### Batch 2: random starts

Every training flight starts at a new random place (x, y within 0.75 m of
the target axis, height 0.2-1.5 m, a held-out block never used), and the agent
sees the target offset in metres. Both are options of `CrazyflieHoverEnv`
(`random_starts`, `metric_target`), off by default so Batch 1 runs are
unchanged. Check the task code first:

```
.venv/bin/python code/src/tdmpc2_training/check_batch2.py
```

Train (published TD-MPC2 settings, 1M steps; on Narval see `cluster/seed_job.sh`):

```
.venv/bin/python code/src/tdmpc2_training/train_hover.py --config-name=config_batch2_random_start 
    seed=0
```

`config_batch2_fixed_start` is the same with the Batch 1 start. Every 25k
steps 21 test flights go to `learning_curve.csv`; the agent is kept every
100k steps in `snapshots/`. Test a finished run on all test groups, then
without planning and with a small planner:

```
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir>
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --no-mpc
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --plan 64x2
.venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --snapshot 300000
```

Test groups and metrics are described in `hover_eval.py` and `EXPERIMENTS.md`.
Existing results are never overwritten unless `--overwrite` is given.

Follow-up tests of a finished run (nothing is trained; each writes a folder
inside the run):

| Script | Folder | What it flies |
| --- | --- | --- |
| `eval_disturbances.py` | `disturbances[_policy_only]/` | wind, gusts, extra mass, weaker motors, command delay |
| `eval_snapshots.py` | `snapshot_evaluation[_policy_only]/` | every 100k-step snapshot, validation and test starts |
| `model_error.py` | `model_error/` | records flights and measures the world model's prediction error along them |
| `eval_planner_variants.py` | `planner_variants/` | policy only, planner, narrower search, half the candidates from the policy |
| `eval_horizon.py` | `horizon_test/` | planning horizons 1, 3, 5 and 10 |
| `eval_smoothing.py` | `smoothing_test/` | averaged-plan action, filtered commands, slower motors |

After copying the results home (`cluster/pull_results.sh`), every analysis
runs with one command; each writes a `summary.md` and its figures:

```
bash code/src/tdmpc2_training/analyze_all.sh
```

Two further training setups exist for follow-up questions:
`config_batch2_randomized_small` (randomised mass, motors and wind,
`crazyflie_hover_dr.py`) and `config_batch2_sharp_small` (an extra reward
term within about 5 cm of the target, `crazyflie_hover_sharp.py`).

# Part 10. Deployment readiness

Source: `code/src/tdmpc2_training/reports/DEPLOYMENT_READINESS.md`, last modified 2026-11-02.

## Deployment readiness: TD-MPC2 on a real Crazyflie

**Status (Nov 2, 2026): not ready. Do not fly.** Everything below was checked
in simulation or by reading code. No drone was powered, armed or commanded.
Physical tests need Prof. Bellinger's approval, the lab safety training, and
the blockers below fixed first.

### Summary

| # | Item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Action semantics: what the agent outputs vs what the firmware accepts | **Blocker** | see 1 below |
| 2 | Thrust calibration | **Blocker** | the lab had to scale thrust by 0.72; the simulator's hover thrust sent a drone into the ceiling |
| 3 | Observation built from real state estimates | Solved in simulation | `deploy_controller.py`, `check_deployment.py` |
| 4 | Units, frames, quaternion order, previous-action convention | Solved in simulation | same |
| 5 | Decision latency | OK with a GPU computer; not on a CPU or on board | experiment P |
| 6 | Robustness to delays, mass, motor strength, wind | Measured in simulation; results pending for Batch 2 | disturbance tests |
| 7 | State estimation (Flow deck) | Risk | lab Final Report: unreliable on dark netted mats |
| 8 | Take-off and landing | Not covered by the agent | agents start in the air, at rest |
| 9 | Failsafes | Must keep the lab's, plus a watchdog | `run_hardware.py` safety latch |
| 10 | Smoothness of the planner's commands | **Risk** (measured) | see 10 below |

### 1. Action semantics (blocker)

- **What the agent was trained to output:** a collective thrust in newtons
  ([0, 0.35]) and three body moments ([-1, 1] x 1e-5 N·m), applied directly by
  MuJoCo motors (`gpu_training/assets_mjx/bitcraze_crazyflie_2/cf2.xml`, lines
  67-70) at 50 Hz.
- **What the Crazyflie firmware accepts:** attitude or body-rate setpoints
  with a thrust value, or position, velocity or full-state setpoints, or motor
  powers. None of these is a body moment.
- **What the lab did before:** `run_hardware.py:278-281` sent the moments as
  body-rate setpoints clamped to ±1 deg/s. That is not physically equivalent,
  and in practice only thrust was controlled.

**Way forward.** Retrain with an action space the firmware accepts, for
example collective thrust plus body-rate setpoints, with the firmware's rate
controller modelled in the simulator. Then repeat the Batch 2 tests. This
changes the task, so the results so far would need repeating. It is the
single biggest sim-to-real task.

`deploy_controller.TDMPC2Controller.to_crazyflie_command` deliberately
raises an error, so the current agents cannot be wired to a drone by mistake.

### 2-4. Observation, units and conventions (solved in simulation)

`deploy_controller.py` builds the agent's 31-number observation from a state
estimate given in the Crazyflie's own log conventions:
- `stateEstimate.x/y/z` in m;
- `stateEstimate.vx/vy/vz` in m/s, world frame;
- `stateEstimate.qx/qy/qz/qw`, reordered to (w, x, y, z);
- `gyro.x/y/z` in deg/s, body frame, converted to rad/s.

It reproduces the simulator's conventions exactly:
- MuJoCo free joint: linear velocity in the world frame, angular velocity in
  the body frame;
- the target offset scaling (metres for Batch 2 agents; divided by the start
  distance for Batch 1 agents);
- the previous-action feature, which in the simulator is the command sent
  **two** decisions earlier.

`check_deployment.py` switches the simulator's sensor noise off and lets the
adapter fly the simulated drone in closed loop. At every step the adapter's
observation must equal the simulator's. Results are recorded in the research
log.

Mistakes found in the lab's PPO hardware path, avoided here:
- the observation normaliser was not applied (`run_hardware.py:352-357`);
- the previous action was taken one step later than in training, and was not
  clipped.

### 5. Latency

Time per decision for one drone, median / 99th percentile / worst:
- **A100 2/7 slice, native Linux:** published planner 6.1 / 7.0 / 11.7 ms;
  policy network only 0.37 ms.
- **2 CPU cores:** published planner 516 ms; small planner 11.4 ms; policy
  only 0.20 ms.
- **Home RTX 3060 under WSL2:** published planner 10.7 ms.

The control period is 20 ms. Planning therefore needs a GPU computer
(off-board, over the radio), unless the policy network alone is used. The
first call compiles for several seconds, so the controller must be warmed up
before take-off (`TDMPC2Controller.warm_up`).

The radio round trip and the state-log period (10-30 ms in `run_hardware.py`)
come on top. The simulator has no such delay. The disturbance tests measure
the effect of a 20 or 40 ms delay. For the Batch 1 1M agents, a 40 ms delay
reduced success from 46% to 41% without planning and from 21% to 12% with
planning.

On board the Crazyflie (STM32F405, 192 KB RAM), the planner is impossible.
Even the policy network would have to be ported and probably quantised.
Eschmann et al. (2024) ran policies on microcontrollers, but those were much
smaller networks.

### 6. Robustness in simulation

`eval_disturbances.py` covers wind, gusts, +5-15% mass, -5-10% motor
strength and 20-40 ms delays. The Batch 1 1M agents fail at 2 m/s steady
wind, with and without planning. Results for the Batch 2 agents are due on
Nov 2.

Real drones will differ by more than these single factors (drone-to-drone
thrust differences: `docs/Crazyflie_Hardware.md`). Dynamics randomisation in
training (Peng et al. 2018; Molchanov et al. 2019) is the standard remedy,
and the lab environment already supports it.

### 7-9. State estimation, take-off, failsafes

- **State estimation:** the Flow deck gives position relative to the power-on
  point and failed on the lab's dark netted mat. For first tests, use
  Lighthouse or motion capture if available, and test the agent's sensitivity
  to position noise and drift in simulation first.
- **Take-off and landing:** every training flight starts in the air (0.2 m
  or higher) and at rest, and a height below 5 cm ends the episode. Let the
  firmware's high-level commander take off to the agent's start height and
  land; hand control to the agent only in between.
- **Failsafes:** keep the lab's latched safety trigger (2 m bound, final
  descent) and the firmware's own commander timeout. Add a check that falls
  back to a hover command when a decision is late.

### 10. Smoothness of the planner's commands (risk)

`analyze_flight_quality.py` measures, from the step-by-step test records, how
much each command changes between consecutive decisions in steady hover (the
last 5 s of full flights). Small random-start agents (8 seeds, Nov 2):
- with planning, the thrust changes by about 29% of its range per step and
  the body moments by about 33%, and about half of all steps have a command
  at its limit;
- the policy network alone: about 8% for both, and 2% of steps at a limit.

The tests use the deterministic planner (best candidate's first action, no
added noise), so this is how TD-MPC2's planner acts: it executes one sampled
candidate per step, and many candidates are clipped at the limits. Real
motors cannot follow a command that jumps by a third of its range every
20 ms; their response would filter the commands, which changes the closed
loop the agent was trained in. `eval_smoothing.py` tests whether a smoother
way of acting (the planner's averaged plan, or a 40 ms command filter) keeps
the precision, and whether the agents tolerate a first-order motor lag of 30
or 60 ms (assumed values; the real motor response must be identified).

### Proposed supervised test sequence (not started)

- 0. Lab safety training, supervisor approval, written test plan.
- 1. **Software in the loop:** the adapter flies the simulator. Done.
- 2. **Radio-only, no thrust:** drone on the bench, motors never armed. Read the
   state at 50 Hz, run the adapter, log the decisions, the end-to-end latency
   and the jitter. Send nothing.
- 3. Retrain with a firmware-compatible action space (item 1). Repeat the
   Batch 2 tests and step 1.
- 4. Tethered or caged hover with a safety pilot and an emergency stop, at low
   thrust limits.

# Part 11. Literature notes

Source: `code/src/tdmpc2_training/reports/LITERATURE.md`, last modified 2026-11-01.

## Literature used in this project

Each entry was checked on Nov 1, 2026 against the arXiv or publisher listing.
"Establishes" is what the source itself reports. "For us" is a hypothesis to
test, not a fact about the Crazyflie task.

### World models and planning

**Hansen, N., Su, H., Wang, X. (2024). TD-MPC2: Scalable, Robust World
Models for Continuous Control. ICLR 2024.** arXiv:2310.16828,
https://arxiv.org/abs/2310.16828
- **Establishes:** decoder-free latent world model with local MPPI planning;
  one fixed set of hyperparameters across 104 continuous-control tasks in four
  domains; agents up to 317M parameters.
- **For us:** we use its published hyperparameters unchanged, so a failure
  here is not a tuning failure.
- **Does not transfer automatically:** those benchmarks are not
  quadrotor-like. Our planning horizon is 3 steps at 50 Hz, i.e. 60 ms of
  look-ahead, short for a drone. Our implementation is the JAX port by
  S. Flandermeyer, with documented differences (CONTEXT.md section 9).

**Hansen, N., Wang, X., Su, H. (2022). Temporal Difference Learning for
Model Predictive Control. ICML 2022, PMLR 162:8387-8406.** arXiv:2203.04955,
https://arxiv.org/abs/2203.04955
- **Establishes:** TD-MPC combines a short-horizon latent model for
  trajectory optimisation with a learned terminal value function, both
  trained by TD learning.
- **For us:** with a 3-step horizon, the value function, not the model
  rollout, carries most of the long-term judgement. So planning-vs-policy
  differences may mostly reflect how MPPI uses the value function off the
  states seen in training.

**Janner, M., Fu, J., Zhang, M., Levine, S. (2019). When to Trust Your
Model: Model-Based Policy Optimization. NeurIPS 2019.** arXiv:1906.08253,
https://arxiv.org/abs/1906.08253
- **Establishes:** the benefit of model-generated data must be weighed
  against its bias. Short model rollouts branched from real data work best.
- **For us, hypothesis H1 (model exploitation):** where the world model has
  little data (unseen starts, disturbances), its errors are largest. A
  planner that optimises against the model then picks actions the model
  wrongly rates highly. The Batch 1 agents crashing more with planning than
  without (R4) fit this.
- **Test:** R1 vs C1 (more diverse data should shrink the gap); measure model
  prediction error along failing flights (next step).

**Lambert, N., Amos, B., Yadan, O., Calandra, R. (2020). Objective Mismatch
in Model-based Reinforcement Learning. L4DC 2020, PMLR 120:761-770.**
arXiv:2002.04523, https://arxiv.org/abs/2002.04523
- **Establishes:** one-step prediction accuracy of a dynamics model does not
  necessarily correlate with control performance.
- **For us:** low TD-MPC2 training losses need not mean good flights. Check
  whether learning-curve dips (within-run fluctuation) line up with the
  logged losses at all.

### Evaluating RL reliably

**Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.
(2018). Deep Reinforcement Learning That Matters. AAAI 2018.**
doi:10.1609/aaai.v32i1.11694, arXiv:1709.06560
- **Establishes:** reported deep-RL results vary strongly with random seeds
  and implementation details; it gives reporting guidelines.
- **For us:** the 6.5 cm to 1.05 m spread of the Batch 1 small runs is
  expected behaviour for short runs. A1 measures how much of it is
  evaluation noise.

**Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., Bellemare, M. G.
(2021). Deep Reinforcement Learning at the Edge of the Statistical
Precipice. NeurIPS 2021.** arXiv:2108.13264, https://arxiv.org/abs/2108.13264
- **Establishes:** with few runs, point estimates mislead. It recommends
  interval estimates (bootstrap), performance profiles and the interquartile
  mean (rliable library).
- **For us:** report mean ± SE across seeds, every seed, and paired
  differences; avoid claims from 3 seeds.

### Sim-to-real for quadrotors

**Molchanov, A., et al. (incl. Hönig, W., Preiss, J. A., Ayanian, N.)
(2019). Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies
to Multiple Quadrotors. IROS 2019.** arXiv:1903.04628,
https://arxiv.org/abs/1903.04628
- **Establishes:** low-level policies (state to motor outputs) trained only
  in simulation transfer to several real quadrotors without a pre-tuned
  inner controller. They withstand disturbances and harsh (thrown) starts.
  The paper studies which randomised parameters matter.
- **For us:** robustness comes from training on varied starts and
  randomised dynamics. That supports Batch 2's random starts and suggests
  testing dynamics randomisation next.
- **Does not transfer automatically:** their action space (motor outputs)
  differs from ours (thrust + body moments); see DEPLOYMENT_READINESS.md.

**Eschmann, J., Albani, D., Loianno, G. (2024). Learning to Fly in Seconds.
IEEE RA-L.** arXiv:2311.13081, https://arxiv.org/abs/2311.13081
- **Establishes:** sim-to-real transfer for direct RPM control after 18 s of
  training on a laptop, with the policy running on microcontrollers under
  real-time guarantees.
- **For us:** a small policy network on board is feasible. This is relevant
  because TD-MPC2's policy network alone decides in 0.2-0.4 ms (P), whereas
  its planner needs a GPU.

**Peng, X. B., Andrychowicz, M., Zaremba, W., Abbeel, P. (2018).
Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA
2018.** doi:10.1109/ICRA.2018.8460528, arXiv:1710.06537
- **Establishes:** randomising simulator dynamics in training gives policies
  that work on the real robot without real-world training, robust to
  calibration error (robot-arm pushing task).
- **For us:** the lab environment already supports mass, inertia and motor
  randomisation (crazyflie_env.reset with random_reset). Batch 2 keeps it
  off so that start generalisation and disturbance robustness stay separate.
  Turning it on is a candidate follow-up.

# Part 12. Research log

Source: `code/src/tdmpc2_training/reports/RESEARCH_LOG.md`, last modified 2026-11-02.

## Research log: TD-MPC2 Crazyflie hover

Chronological record of hypotheses, experiments, results, decisions and
failures. Times are EST. The experiment registry is `experiment_registry.json`;
the full experiment descriptions are in `../EXPERIMENTS.md`.

### 2026-11-01 (Sunday)

**Afternoon: audit after Prof. Bellinger's Oct 12 feedback.**
- Verified in code:
  - The PPO learning-rate schedule never anneals: it is counted in environment steps but stepped per optimizer update, so PPO ran at a constant 3e-3.
  - The PPO auto-reset carries the previous position and previous action into the next episode.
  - The observation divides the target offset by the start distance.
  - Only the final TD-MPC2 checkpoint is kept.
  - The real-drone code skips PPO observation normalisation and caps body rates at 1 deg/s.
- Decision: TD-MPC2 is the focus. PPO stays a historical baseline, its bugs are documented, and it is not rerun.

**Afternoon: measurements that need no training.**
- Decision latency (home, RTX 3060 under WSL2): published planner 10.7 ms median, policy network only under 1 ms; on the CPU 589 ms and 0.26 ms.
- Planner vs policy only on the 11 small 50k agents: planning gave 0.32 m late error vs 0.48 m (better in 7 of 11 seeds, not significant), but 11 vs 3 crashes over 198 flights. A bigger 512x6 planner added nothing (0.31 m).
- Training-step profile (home): one simulated drone about 10 ms per step, GPU busy 25-28% of the time.
- Learning curves of the 1M runs: after first reaching 10 cm, 13-39% of later test points were worse again. Typical late error 6-8 cm; best single points 1.2-1.8 cm.

**Evening: Batch 2 built and validated (commits 8edab5b, 8612881).**
- New code:
  - random starts in an area with a held-out block, and the target offset in metres (both off by default);
  - fixed test groups;
  - snapshots every 100k steps;
  - tests with and without planning;
  - cluster jobs.
- Validation:
  - 23 automatic checks pass.
  - Testing old agents with the old and new code gives identical results, step by step.
  - Training on the CPU gives identical results with old and new code. The GPU is not bit-repeatable: two runs of the old code also differ once learning starts.
  - End-to-end runs at home and on Narval pass.
- Backup of Narval scratch to /project verified.

**20:41-21:44: launch.**
- Test job 7659264 passed (15 min).
- R1 job 7659438 and C1 job 7659439 were submitted with a 10 h limit. They waited in the queue about 50 min: low fairshare, and 10 h jobs are hard to backfill. Plan B was prepared and validated: 3 h pieces that resume from checkpoints, with a fix so a resumed run draws fresh starts and noise (commit 52873af). It was not needed: R1 started 21:38-21:44 and C1 at 21:44.
- R4 job 7659440 finished: Batch 1 agents on the Batch 2 test starts.
  - The 1M single-start agents crash far more often **with** planning than with their policy network only: 66% vs 15% on training-area starts, 91% vs 54% in the held-out block, 3 seeds.
  - Hypothesis H1 (model exploitation): out of the states seen in training, the world model is wrong and MPPI selects actions it wrongly predicts as good, while the policy network is smoother. R1 tests whether training on diverse starts removes this.
- P job 7659441, Narval A100 2/7 slice, native Linux:
  - Published planner 6.1 ms median, 11.7 ms worst; policy only 0.37 ms; 516 ms on 2 CPU cores.
  - One training step: simulate 6.8 ms, plan 10.2 ms, update 2.6 ms (overlapped), 49 steps/s.
  - nvidia-smi reports no utilisation on MIG slices.
- Disturbance tests: Batch 1 1M agents (job 7660609); R1 and C1 queued to start after each seed (jobs 7660720, 7660721, `aftercorr`).
  - `check_disturbances.py` validated the physics: wind drag grows about with speed squared; +15% mass sinks 0.62 m in 1 s (0.64 m predicted).

**21:50: overnight program started** (priorities in `../EXPERIMENTS.md` and the handoff).
- A1 started: re-fly 33 Batch 1 agents with 5 noise seeds x 4 flights each, to split the seed-to-seed spread into evaluation noise vs real agent differences (Q1).

**22:00-22:40.**
- **D pilot launched:** small TD-MPC2 with random starts, 3 seeds x 1M (`config_batch2_random_start_small`), in two pieces of at most 3 h (jobs 7661493, then 7661496 with `aftercorr`), disturbance tests chained (job 7661497). The pieces started immediately: jobs of 3 h or less are scheduled at once on Narval, while 10 h jobs waited about 50 min.
- **Disturbance tests of the Batch 1 1M agents done** (job 7660609, 30 min per seed). Success from 50 training-area starts, planning vs policy only:
  - none: 21% vs 46%
  - wind 1 m/s: 15% vs 15%; wind 2 m/s and above: 0% vs 0%
  - gust 2 m/s: 17% vs 41%; gust 4 m/s and above: 0% vs 0%
  - mass x1.15: 9% vs 36%
  - motors x0.9: 12% vs 40%
  - delay 40 ms: 12% vs 41%

  So for single-start agents, planning is worse under every disturbance. Caveat: these starts are already new for these agents, so this is a combined shift. "Crash" includes leaving the bounds; `eval_disturbances.py` now also records `end_cause` (ground / out_of_bounds) and final height, checked by `check_disturbances.py`.
- **R1 learning check at 75k steps:** crash share falling (home 91% to 64%, training area 99% to 83%). Training speed 46-49 steps/s, expected end about 04:20 plus final tests. C1 learns its single start faster (0% home crashes at 75k).
- **A2 (within-run fluctuations vs training logs), 3 Batch 1 1M runs:** step-to-step changes in test late error do not correlate with changes in any logged training statistic (consistency, reward, value and policy losses, policy log std, value scale). Spearman rho between -0.14 and +0.10, all permutation p > 0.07 before correcting for 18 tests.
  - Level correlations up to ±0.34 in two seeds mostly reflect shared slow trends.
  - Conclusion: the training losses do not reveal when the controller gets worse, consistent with objective mismatch (Lambert et al. 2020).
  - Open question: each learning-curve point is only 3 flights with one noise seed, so part of the fluctuation may be evaluation noise. This motivates the snapshot evaluation (many flights per snapshot).
- **Literature checked** (`LITERATURE.md`): TD-MPC2, TD-MPC, MBPO (model exploitation), objective mismatch, Henderson 2018, Agarwal 2021, Molchanov 2019, Eschmann 2024, Peng 2018.
- **Deployment readiness** written (`DEPLOYMENT_READINESS.md`). New `deploy_controller.py` builds the observation from Crazyflie-style state estimates and refuses to produce a firmware command; `check_deployment.py` tests it in closed loop against the simulator.

**23:00-23:20.**
- **200k check of R1 (11 seeds), pilot criterion "learning occurs": passed.** Crash share at 100k vs 200k:
  - home 64% to 9%; training area 77% to 14%; held-out block 76% to 29%; outside 92% to 74%.
  - Precision not reached yet (median late error about 0.35 m, almost no successes).
- **C1 (single start) specialises.** Home error 0.12 m at 200k. Its held-out crash share went from 56% (150k) to 100% (200k), so generalisation peaks early and then declines. This is the "memorising" effect, now seen during training.
- **Small model with random starts (3 seeds) learns much faster.** About 0% crashes in all groups by 100k; outside crash share 0-6% at 200-225k (big R1 at 200k: 74%); success 17-42%.
  - Decision: scale the small model to 11 seeds (seeds 3-10, jobs 7662436 then 7662437, disturbance tests 7662438), for a credible small vs big comparison (Q9).
  - Snapshot evaluation chained to every Batch 2 seed (jobs 7662439-7662442).
- **Deployment checks pass** (`check_deployment.py`): the adapter's observation equals the simulator's over 300 closed-loop steps for a Batch 1 agent (offset divided by the start distance) and a Batch 2 agent (metres), with planning and without (largest difference 1.9e-6). Commands stay in the actuator limits. The first decision compiles for 20-44 s, so warm-up is required. The latencies in this run are inflated because the GPU was shared.
- **A1 result (Q1): the Batch 1 seed spread was real, not evaluation noise.** 33 agents x 20 flights from the training start:
  - small 50k (Narval): between-seed SD 0.32 m, within-seed SD 0.09 m, ICC 0.93;
  - small 50k (home): ICC 0.94;
  - published-size 50k: ICC 0.52 (many crashes, so flights vary a lot).
  - Batch 1's 3-flight estimate per seed had an SE of about 0.05 m, small next to the 0.32 m spread.
  - Home seed 0 is at 2.3 m on all 20 flights: its Batch 1 "collapse" is a property of the final agent.
  - With the 50k learning curves still improving and the 1M runs converging, this supports "short runs had not finished training".

**23:20-23:45: E, parallel drones (Q5), home RTX 3060, published settings (`bench_parallel.py`).**
- Simulating N drones in one batch costs 8-10 ms for any N from 1 to 128, so the physics parallelises perfectly.
- Planning grows with N: 10.6 ms for 1 drone, 57 ms for 8, 195 ms for 32, 735 ms for 128.
- One learning update (batch 256) takes 12.4 ms.
- Projected training speed with TD-MPC2's default of one update per simulated step: 32 steps/s with 1 drone, 49 with 8, 55 with 128, so at most about 1.7x faster.
  - With 1 update per 8 steps (a different algorithm): up to 136 steps/s.
  - The 1-drone projection (32) matches the measured home training speed (31 steps/s), which validates the cost model.
- Conclusion: TD-MPC2's per-sample costs (planning for every drone, one update per step) scale with the number of drones, unlike PPO, whose updates are shared over large batches. Larger speedups need fewer updates per sample or cheaper planning during data collection. Both change the algorithm and need their own sample-efficiency check.
- A Narval run of the same benchmark is next.

**23:45-00:05.**
- **Model error, validation run:** `model_error.py` on one Batch 1 small agent (50k, seed 4), 3-step latent error:
  - planner flights: training area 0.0003, held-out block 0.0005, outside 0.0009;
  - policy-only flights from the same starts: 0.0003, 0.0003, 0.0006;
  - reward error outside: 0.0055 planner vs 0.0022 policy;
  - crashes outside: 21/30 planner vs 2/30 policy.
  - This fits H1: the model is less accurate on new starts and the planner steers into its weak regions. One agent only.
- **Submitted:** model error for all 25 Batch 1 runs (job 7663280); chained after R1, C1 and small (7663281-7663284); parallel benchmark on Narval (7663279). All 11 small seeds are in piece 1.
- **Mid-run learning curves** (`outputs/batch2/midrun/random_start_1M/analysis/learning_curves_comparison.png`), R1 at about 325k, C1 300k, small 150k-450k:
  - Random-start training generalises to the held-out block: R1 held-out crash share about 10% at 300k, close to its training-area level. The single-start control C1 crashes 40-60% from training-area starts and 70-100% from the held-out block.
  - The small model leads at equal steps: about 0% crashes in the training area and held-out block from 50k, 0-20% outside. R1 is at about 50% outside at 325k.
  - Small-model success is rising, but the points after 175k come from only 1-3 seeds, so they are noisy.
  - To be confirmed by the final 128-flight evaluations.

**00:00-00:15: model error of all 25 Batch 1 runs (job 7663280; `outputs/batch2/model_error_summary_batch1/`).**
- 1M single-start agents (3 seeds):
  - The planner's flights have 2.8 ± 0.7x (training area) and 2.4 ± 0.2x (held-out block) the 3-step latent prediction error of the policy-only flights from the same starts, paired per seed. Outside, 1.1x: both fail there.
  - Crashes, planner vs policy: training area 99/150 vs 22/150; held-out 83/90 vs 48/90.
  - In the last 0.5 s before a crash, the model error is 4.0x that of completed flights.
- Small 50k agents (11 seeds):
  - Planner / policy error ratio 1.4, 1.9 and 1.3; error 7.0x higher before crashes.
  - Error on outside starts is 2.5x that on training-area starts.
- Published-size 50k agents: almost every flight crashes, so this is uninformative.
- **H1 (model exploitation) supported, correlationally:** the planner drives the drone into states where its world model is wrong, and crashes follow rising model error.
  - Alternative explanation: unusual states near a crash are hard to predict whatever caused them.
  - Causal test started: `eval_planner_variants.py` keeps the trained weights but plans more conservatively (sampling spread 0.5 instead of 2, or half the candidates from the policy network). If exploitation is the cause, these should crash less than the default planner.

**00:05: planner variants, validation run.** Small 50k agent, seed 4, same starts and noise for every variant. Success / crash share in training area, held-out block, outside:
- policy only: 0/0%, 0/0%, 0/7%
- planner as trained: 70/6%, 50/23%, 10/67%
- narrower sampling (max_plan_std 0.5): 64/8%, 63/20%, 10/67%, no change
- half the candidates from the policy network: 76/0%, 90/0%, 13/43%

So anchoring the planner to the policy network's proposals removes most of its extra crashes and keeps (even raises) its precision; only narrowing the search does not. One agent only. Submitted for all 25 Batch 1 runs (job 7663755) and chained after R1, C1 and small (7663756-7663773).
- If it holds across seeds, it is a test-time, no-retraining improvement, and causal evidence for exploitation: the crashes come from candidates far from the policy prior.

**00:35: planner variants on all 25 Batch 1 runs** (job 7663755, `outputs/batch2/planner_variants_summary_batch1/`). Paired change vs the planner as trained, mean ± SE across seeds:
- **1M agents (3 seeds), training-area starts:**
  - half the candidates from the policy network: success +15 ± 2 points, crashes -19 ± 1;
  - narrower sampling: +9 ± 2 / -13 ± 1;
  - policy network only: +21 ± 8 / -49 ± 12.
  - In the held-out block and outside, all changes have the same direction but are smaller and noisier.
- **Small 50k agents (11 seeds):** half-policy candidates reduce crashes by 3-7 points (SE 2-4); success changes are within noise.
- **Published-size 50k agents:** crash 95-100% whatever the planner setting.
- **Revised conclusion:** more conservative planning reliably removes part of the planner's extra crashes, consistent in all 3 1M seeds. So exploitation of model errors is one mechanism. It is not the whole story: for single-start 1M agents the policy network alone remains far more robust (63% vs 15% crashes).
  - The seed-4 result (23% to 0% crashes) was an extreme case.
  - Remaining candidates: the value function is wrong off the training data (MPPI ranks plans by model reward plus value); 256 free candidates still allow exploitation.

**00:35: parallel projection on Narval** (job 7663279, 36 min, `outputs/batch2/parallel_projection_narval/`), with 1 update per simulated step:
- **Published size:** simulate 6.0 ms for any N; plan 6.4 ms (1 drone) to 475 ms (128); update 9.7 ms. Projected 45 steps/s with 1 drone, 68 with 8, 74 with 128 (1.6x).
- **Small size:** plan 1.2 to 16.6 ms; update 3.5 ms. Projected 93, 223 and 271 steps/s (2.9x).
- The 1-drone projections match the measured training speeds (46-49 and 88 steps/s), so the cost model holds.

**00:35: health check.** 84 jobs completed, 25 running, none failed. Small pilot piece 1 took 104-112 min per 500k steps; piece 2 is running.

**00:40-00:45: H2 pilot launched.**
- **Hypothesis H2:** planning helps when the world model has trained on the relevant variation (interpolation) and hurts when it has not (H1, extrapolation).
- **Setup:** `crazyflie_hover_dr.py` (new env id `CrazyflieHoverRandomized-v0`, registered lazily so existing runs are untouched). The small model with random starts, plus in every training flight: mass and inertia x1.0-1.15, motor strength x0.9-1.0, steady wind 0-3 m/s from a random direction. The agent does not observe these values.
- **Validated:**
  - the draws stay in range;
  - the flight state matches the draws;
  - the wind in training pushes the drone exactly as in the disturbance tests (0.417 m in 1 s at 3 m/s, both);
  - a 1,500-step training run works.
- **Launched:** 3 seeds x 1M in two 3 h pieces (7664888 then 7664889), with disturbance tests (7664892) and planner variants (7664893) chained.
- **Comparison:** the same tests on the nominal-trained small agents. In-range disturbances: mass, motors, wind up to 3 m/s. Beyond the range: wind 4 m/s, gust 6 m/s, delays.

**02:07: first fully trained Batch 2 agents.** Small model, random starts, seeds 0-2, 1M steps (job 7661496); final tests of 128 flights each (`outputs/batch2/random_start_small_1M/analysis/`). Success / crash share:

| Decision rule | training area | held-out block | outside | Batch 1 starts |
| --- | --- | --- | --- | --- |
| with planning | 95 ± 4% / 0% | 97 ± 2% / 0% | 68 ± 10% / 2% | 94 ± 6% / 0% |
| policy network only | 1% / 0% | 0% / 0% | 8% / 0% | 6% / 0% |
| 64x2 planner | 94% / 0% | 88% / 0% | 69% / 2% | 94% / 0% |

- Median late error from training-area starts: 5.0-6.9 cm per seed.
- With planning vs the Batch 1 1M single-start agents: +75 to +94 points of success (95% bootstrap intervals all far above 0).
- **Planning vs policy only, paired:** +94 [+86, +100] (training area), +97 [+93, +100] (held-out), +60 [+27, +83] (outside). For the single-start 1M agents it was -26 [-46, -8].
- **Interpretation, for when and why the model helps (Q7):**
  - Trained on data covering the test situations, the policy network alone is safe but imprecise (no crashes, about 1% success), and planning with the world model adds the precision.
  - Trained on one start, planning off that data exploits model errors and crashes.
  - So the world model helps where its training data covers the situation, and hurts beyond it.
- 3 seeds; seeds 3-10 due about 03:30.
- **Where the precision comes from** (median late error of full flights from training-area and held-out starts, per seed):
  - planner: 5.1-7.4 cm (horizontal 4.1-7.4 cm, vertical 0.9-2.9 cm);
  - policy only: 15-41 cm (horizontal 13-16 cm, vertical 8-38 cm);
  - 64x2 planner: 5.1-8.7 cm.
- The policy network alone holds the wrong height. Planning corrects it almost entirely, mostly the thrust (hovering needs 76% of maximum thrust, so small biases matter).
- Hypothesis: the policy network, trained to maximise the value with an entropy bonus, has a biased mean thrust, and short-horizon planning against the model's predicted rewards removes the bias.

**02:19-02:27: horizon test (added after the main experiments, eval only).**
- **Question:** TD-MPC2 plans 3 steps (0.06 s) ahead. Would a longer horizon help, or do world-model errors compound beyond the horizon it was trained on?
- **Setup:** `eval_horizon.py` flies the same agent with horizons 1, 3, 5 and 10 (changed only at test time), from the 50 training-area and 30 outside starts, nominal, in 3 m/s steady wind and with 15% extra mass; same sensor noise for every horizon.
- **Validated:** the planner sizes its plan and warm start from the horizon setting (`tdmpc2.py` lines 196-201, `hover_eval.py` line 203), including horizon 1; the pilot ran without errors.
- **Pilot (Batch 1 small agent, seed 4, single start; 380 s at home).** Success / crash from training-area starts, nominal: horizon 1: 0% / 54%; 3: 64% / 10%; 5: 22% / 18%; 10: 2% / 42%. Same ranking with extra mass; everything fails in 3 m/s wind.
- **Reading:** the trained horizon is best for this agent; both shorter and longer horizons are worse. One agent only.
- **Launched:** `cluster/horizon_job.sh` on every Batch 2 agent (small 0-2 now; small 3-10, R1, C1 and randomized chained on their training). Commit b86b66b. `analyze_horizon.py` summarises across seeds.

**02:40-03:00: what the within-run swings consist of, and how the drone flies.** Small random-start agents (D), first finished seeds, all from existing test files.
- **Snapshots (seeds 0-2):** with planning, train-area success swings between about 0% and 100% from one 100k snapshot to the next (within-run SD 0.23-0.45 against 0.05-0.07 expected from 50 flights), but no snapshot of any seed crashed. What swings is the steady-state precision: the median late error of full flights moves between about 3 and 15 cm (once 55 cm), around the 10 cm success threshold, so a shift of a few cm flips almost every flight at once. Success is a threshold measure and amplifies small changes; the continuous late error is now reported next to it (`analyze_snapshots.py`).
- **Policy network alone across snapshots:** no crashes either, but its height offset changes sign between snapshots (-23 to +51 cm). One seed was precise with the policy alone at 300k-400k (100% success) and lost it later. So the policy's imprecision is an unstable offset, not a fixed bias.
- **Command jitter (`analyze_flight_quality.py`, new):** in steady hover (last 5 s), the planner's thrust changes by 31 ± 2% of its range per step and its moments by 34 ± 1%; 49 ± 2% of all steps have a command at its limit. Policy network alone: 7 ± 2%, 8 ± 0.3%, 2 ± 0.3%. The tests use the deterministic planner (`hover_eval.py` line 210: best elite's first action, no added noise), so the jitter is built into how the planner acts: it executes one sampled candidate, and many candidates sit at the limits after clipping. Same pattern for the Batch 1 agents (planner 24-27% thrust jitter, 47-51% saturated).
- **Why it matters:** a real Crazyflie's motors cannot follow commands that jump by a third of their range every 20 ms; the motor dynamics would filter them, which changes the closed loop. This is a deployment blocker not visible in success rates (Q10), and a cost of planning not visible in Q8.
- **Follow-up (eval only), `eval_smoothing.py`:** the same agents with (a) the planner as trained, (b) the first action of the planner's averaged plan, (c) the planner's commands through a 40 ms low-pass filter, (d) the policy alone; each with instant motors and with first-order motor lags of 30 and 60 ms (assumed values to probe sensitivity, not identified Crazyflie values).
- **Validation of `eval_smoothing.py` (Batch 1 small agent, seed 4, 50 starts):**
  - policy network alone: bit-identical to `DisturbedFlightTester`;
  - planner as trained: NOT bit-identical. The same program run twice is bit-identical (50/50 flights), but the new program first differs at steps 2-72 by 1e-8 to 2e-7 (one float32 rounding step) with identical positions before. Differently compiled GPU programs round differently, and the planner's choice between near-tied candidates amplifies it;
  - statistically equivalent over 4 noise seeds: success 66/66, 68/68, 66/68, 64/64%; crashes 10/8, 10/10, 10/10, 10/12%;
  - all variants of the smoothing test run in one compiled program (the variant, filter and lag are inputs), so their comparisons stay exactly paired;
  - the 40 ms filter reaches 63.2% of a step after 40 ms, as it should;
  - jitter on this agent: planner 30.7%, averaged plan 24.4%, filtered 8.8%.

**03:00: a hypothesis for the precision drift (from the reward, not yet tested).** The reward per step (`gpu_training/crazyflie_env.py` lines 293, 321-327) is 0.001 + 0.01 (1 - tanh d) + (previous d - d) - 1 on a crash. Hovering still at 5 cm earns 0.0105 per step, at 15 cm 0.0095: a difference of 0.001 per step. With discount 0.99 the value of hovering is about 1.05, and the difference between 5 and 15 cm about 0.1. In TD-MPC2's value encoding (101 bins over symlog -10..10, bin width 0.2) that is 0.05, a quarter of a bin. Within the 3-step planning horizon the rewards differ by only 0.003. So precision within about 15 cm is weakly anchored by the reward, which may explain why the steady-state error drifts between 3 and 15 cm from checkpoint to checkpoint while crashes stay at zero. Testable next: retrain with a sharper proximity term near the target (for example a Gaussian of width 5 cm) and compare the snapshot-to-snapshot drift.

**03:06: two follow-ups launched (commit 3f9f21b, home and Narval identical, no uncommitted changes).**
- **Smoothing test** (`eval_smoothing.py`, eval only) on the 11 small and 11 R1 agents: jobs 7668917 (small 0-2), 7668921 (small 3-10, after 7662437), 7668922 (R1, after 7659438).
- **Sharp-reward pilot** (`config_batch2_sharp_small`, `crazyflie_hover_sharp.py`): 3 seeds x 1M of the small model, identical to D except an extra reward 0.01 exp(-d^2 / (2 x 0.05^2)) near the target. Tests the 03:00 hypothesis. Jobs 7668941 -> 7668942 (two 3 h pieces), snapshot evaluation 7668947. Validation: bonus values exact; observations, termination and physics identical to the plain task over 600 steps; reward = plain + bonus at every step; a 1,500-step training run with snapshots works. Prediction if the hypothesis holds: smaller final late error and less snapshot-to-snapshot drift than D, with crashes still near zero. Expected done about 07:15 (training) and 07:45 (snapshots).

**03:13: first smoothing-test results (small model, 4 seeds, preliminary).** Training-area and held-out starts; success, then thrust jitter and steps at a limit:
- instant motors: planner as trained 72%, 29.5%, 50%; averaged plan 72%, 23.6%, 1%; planner + 40 ms filter 73% (+2 ± 1 paired), 8.7%, 2%; policy only 25%, 7.2%, 2%.
- 30 ms motor lag: no change for any variant.
- 60 ms motor lag: planner 71%, averaged plan 72%, but planner + 40 ms filter 54% (-17 ± 8 paired): the filter's and the motors' lags add up.
- **Reading so far:** the planner does not need its jumpy commands. Filtering them (or acting on the averaged plan) keeps the precision while making the commands as smooth as the policy network's, and the agents tolerate slow motors. The filter must be sized together with the real motor lag. Waiting for all 11 small seeds and the R1 agents.

**03:20: learning curves at about 850k steps (R1, C1, D, H2; `random_start_1M/analysis/learning_curves_comparison.png`).**
- R1 (published, random starts): crashes from training-area starts near 0 after about 300k, from the held-out block after about 400k; from outside starts still about 30% at 850k, falling slowly. Success (10 cm) only 20-30%.
- C1 (published, single start): 40-60% crashes from training-area starts, 80-100% from the held-out block and outside, throughout.
- D (small, random starts): crashes near 0 everywhere by 50-100k (about 5% outside); success 60-75%.
- H2 (small, randomised physics): crashes like D; success more variable.
- **Reading (mid-run):** random starts remove crashes for both sizes, but the small model learns much faster and is more precise at equal steps. The published model may need more than 1M steps, or a different learning rate, to match it. Final tests will tell.

**03:25: follow-up tests on 8 of the 11 small random-start agents (D).**
- **Model error (H1 check):** along the planner's flights the world model is as accurate or more accurate than along the policy network's (paired ratio 0.73 ± 0.10 training area, 0.70 ± 0.07 held-out, 1.4 ± 0.8 outside), and its 3-step latent error is about 1e-5, 15-30 times below the Batch 1 small agents (3e-4). No sign of exploitation where the data covers the situation.
- **Planner variants:** restraining the planner now costs success (narrower search -10 ± 8, half the candidates from the policy -7 ± 6 points from the training area) instead of preventing crashes. Consistent with the above: nothing to exploit, so the full search is best.
- **Horizon:** 5 steps beats the trained 3 (+15 ± 9 points from the training area, +9 ± 6 outside); 10 steps is no better in the area (-2 ± 15) and crashes more outside (14% vs 5%); 1 step is much worse (-44 ± 16). With an accurate model a slightly longer look-ahead helps; for the inaccurate Batch 1 pilot agent every change from 3 was worse.
- 3 m/s steady wind: no horizon gives any success (0%), and crashes stay low (0-3% in the area): the drone is pushed off target but kept in the air.

**03:24: continuing a finished run elsewhere (prepared, not launched).** Question raised by the curves: is the published model just slower to learn, so that it would catch up with the small one given more steps? `cluster/copy_run_job.sh` copies a finished run (checkpoint with replay buffer, snapshots, history; not the tests) to a new batch folder, where `seed_job.sh` resumes it. Validated at home: a 1,500-step run copied and resumed to 3,000 steps continues from 1,500, extends its learning curve and adds snapshots next to the copied ones. Plan if R1 is still improving at 1M: seeds 0-2 to 2M in three 3 h pieces (1.35M, 1.7M, 2M), about 9 h per seed; it would finish around 13:30, after the handoff, and be reported as still running.

**03:26: R1 extension launched (commit 622b694).** At 850-875k the R1 curves were still improving (success rising from about 20% at 600k to 30% at 800k; outside crashes falling from about 50% to 35%), so seeds 0-2 continue to 2M: copy 7669541 (after R1), then 7669542 -> 7669543 -> 7669544 (to 1.35M, 1.7M, 2M; final tests in the last piece) in `batch2/random_start_2M`. Cost about 3 x 9 h on GPU slices. Expected done about 13:30, after the handoff; to be cancelled before it starts if R1's final tests make it pointless.

**03:30: experiment D final (small model, random starts, 11 seeds; job 7662437 all exit 0).** Success / crashes:
- with planning: training area 78 ± 10% / 0%, held-out 83 ± 10% / 0%, outside 59 ± 9% / 3 ± 2%, Batch 1 starts 80 ± 10% / 0%;
- policy network only: 9-12% success everywhere, no crashes;
- 64x2 planner: as good as the full planner (+2 [-2, +7] points, paired).
- vs Batch 1 small single-start agents: +59 [+35, +81] (training area), +72 [+48, +91] (held-out), +55 [+38, +72] (outside).
- planner - policy (paired): +69 [+30, +96], +74 [+36, +98], +48 [+18, +72].
- Seeds: 8 of 11 at 86-100% from the training area (median late error 4-8 cm); seeds 3, 8, 10 at 54%, 20%, 8% with no crashes (9-13 cm). The first three pilot seeds (95%) were among the good ones; the full set is lower, which is why the scale-up mattered.
- Precision of full flights (training area + held-out): planner 7.4 ± 0.9 cm (horizontal 7.1, vertical 1.6), policy 20.6 ± 3.0 cm (horizontal 15.1, vertical 11.9; height offset +7.9 ± 4.2 cm, but its sign changes between snapshots), 64x2 7.8 ± 0.9 cm.

**03:35: snapshots of 10 small seeds (seed 9 pending).**
- Within-run SD of training-area success from 500k on: 0.10-0.48 per seed, against 0.04-0.07 expected from 50 flights. The swings are real changes of the agent.
- Mostly precision: the median late error of full flights moves between about 4 and 15 cm, sometimes 20-60 cm. Crashes are rare, but not absent: seed 6's 500k snapshot crashed in 44% of the training-area flights (median error 89 cm), then recovered.
- **Best snapshot chosen on the 40 validation starts vs the final one** (test starts): training area +23 ± 12 points, held-out +10 ± 11, outside +8 ± 9. Mean test success from the training area: 96% at the validation-chosen snapshots vs 74% at the final ones. It rescues exactly the weak seeds: seed 8 14% -> 100%, seed 10 2% -> 100%, seed 3 44% -> 76%.
- Policy network alone: best vs final +32 ± 14 points, but from a low level (its error averages 25 cm from 500k on).
- **Reading:** longer training does not remove the within-run drift; choosing the checkpoint on validation starts does most of the job, at the cost of testing every snapshot.

**03:40: disturbances, small random-start agents (11 seeds, 50 training-area starts).**
- Success with planning: none 77%; wind 1 / 2 / 3 / 4 m/s: 60 / 1 / 0 / 0%; gusts 2 / 4 / 6 m/s: 66 / 15 / 0%; mass x1.05 / 1.10 / 1.15: 76 / 74 / 72%; motors x0.95 / 0.90: 76 / 73%; delay 20 / 40 ms: 77 / 76%. Policy alone: 9-10% everywhere (18% at 1 m/s wind).
- Planning minus policy (paired): +61 to +68 points under mass, motor and delay changes, +42 [+14, +69] at 1 m/s wind, +57 [+16, +91] for 2 m/s gusts.
- Crashes: none for either rule up to 3 m/s wind and 2 m/s gusts (0-1%; the drone is pushed off target but stays up). Beyond: planner 41% vs policy 29% at 4 m/s wind, 67% vs 21% for 6 m/s gusts; 71% of the planner's wind crashes are out of bounds.
- **Reading:** for random-start agents, planning keeps the precision under every moderate disturbance, but under strong wind (states never seen in training) it crashes more than the policy network alone, as H1 predicts. Compare the single-start 1M agents: planner 66-100% crashes in every wind condition vs policy 14-100%.

**03:47: snapshots, all 11 small seeds.** Best snapshot chosen on validation vs final, test success: training area +21 ± 11 points (97% vs 76% on average), held-out +9 ± 10, outside +7 ± 8. Within-run SD of training-area success from 500k: 0.34 ± 0.03 (test noise alone: 0.04-0.07). Seed 9 behaves like the good seeds (final = best = 100%). Experiment D is complete except seed 9's policy-only snapshot test.

**03:55: probing the value network near the target (`probe_value.py`, new; no flights).** The drone at rest and level on a grid of points within 30 cm of the target (three planes, 2 cm steps), exact observation (deploy adapter), previous command a plain hover; value = mean of the value networks for the policy's action. 9 snapshots of small random-start agents: precise ones (seed 10 at 400k, seed 8 at 700k, seed 2 at 400k, seed 5 at 1M; flown errors 3-7 cm) and imprecise ones (seed 10 and seed 8 at 1M, seed 2 at 500k, seed 3 at 500k and 1M; 10-56 cm).
- In every snapshot the value is almost flat around the target: it changes by 0.000-0.004 within 10 cm and 0.003-0.012 within 20 cm, on a value of about 1.06-1.11. It is even slightly higher away from the target (the reward's improvement term lets a drone that is away still collect that distance), but a value that matched the rewards would rise by about 0.1 per 10 cm; the learned one rises by a few thousandths.
- Precise and imprecise snapshots look the same, so the value's shape does not explain which checkpoints drift.
- **Reading:** the value network does not resolve positions within about 20 cm of the target. The precision must come from the world model's predicted rewards summed over the planning horizon (the improvement term gives about 1 per metre over the 3 steps). This fits three results at once: the policy network alone, trained against the value, is imprecise (21 cm); a horizon of 1 step, which leans on the value, is much worse (-35 points); a horizon of 5 is slightly better (+11). It also supports the sharp-reward pilot: a reward that is steep within 5 cm gives both the predicted rewards and the value something to resolve.

**08:45: final results of R1, C1, H2 and the reward pilot (pull + `analyze_all.sh`, 242 s, no errors).** 294 Slurm tasks completed since the launch, none failed; only the R1 extension (7669543 running, 7669544 queued) is left.

**R1 final (published model, random starts, 11 seeds; 7659438 all exit 0).** Success / crashes:
- with planning: training area 46 ± 8% / 0%, held-out 39 ± 9% / 2 ± 1%, outside 20 ± 6% / 26 ± 5%, Batch 1 starts 39 ± 10% / 0%;
- policy network only: 60 ± 11% / 0%, 64 ± 13% / 0%, 40 ± 10% / 6 ± 2%, 65 ± 11% / 0%;
- 64x2 planner: 58 ± 9% / 0%, 46 ± 12% / 0%, 32 ± 7% / 9 ± 2%.
- vs C1 (with planning): +20 [+1, +39], +34 [+17, +51], +20 [+9, +32]; vs Batch 1 published 1M: +26 [+7, +45], +37 [+20, +52], +17 [+7, +30].
- vs D (small model): -32 [-57, -5], -44 [-68, -18], -39 [-59, -18].
- planner - policy (paired): -15 [-37, +10], -25 [-47, -1], -20 [-40, +0]; planner - 64x2: -12 [-24, +3], -8 [-24, +10], -12 [-18, -6].
- Precision of full flights: planner 12.7 ± 1.5 cm (horizontal 12.4, vertical 1.6), policy 9.3 ± 1.4 cm, 64x2 9.9 ± 1.1 cm. Seeds: median late error 6.6-18.7 cm; 4 seeds above 15 cm.
- The learning curves and the snapshots still rise at 1M (training-area success of the snapshots 0.26 at 600k, 0.34 at 900k, 0.42 at 1M; late error falling from about 20 to 13 cm): R1 has not finished learning. The extension to 2M (seeds 0-2) is the right test and keeps running.
- **Reading:** random starts help the published model too (no crashes inside the area, +20 to +34 points over C1), but at 1M it is much less precise than the small model, and its full planner is worse than its own policy network and than a smaller search.

**R1 follow-ups.**
- Model error, planner / policy (paired): 1.17 ± 0.07 (training area), 1.49 ± 0.08 (held-out), 2.00 ± 0.14 (outside), and 4.4 times higher in the last 0.5 s before a crash. Between the small random-start agents (0.68) and the single-start agents (2.4-2.8).
- Planner variants: narrower search +10 ± 3 / +11 ± 6 / +8 ± 4 points, half the candidates from the policy +13 ± 8 / +8 ± 9 / +12 ± 4; outside crashes fall from 26% to 15% and 10%. Restraining the search helps: some exploitation even inside the training distribution.
- Horizon: 1 step -5 ± 6, 5 steps -0 ± 6, 10 steps -13 ± 7 (training area); in 3 m/s wind crashes grow with the horizon (11, 32, 41, 49% for 1, 3, 5, 10 steps).
- Smoothing: averaged plan +2 ± 2 (keeps precision, saturation 46% -> 1%); 40 ms filter -11 ± 3 (unlike D, where it is free); motor lag 30 / 60 ms costs the planner 8 / 19 points but the policy network nothing (61, 61, 59%).
- Disturbances: the planner is below the policy network under every condition (-17 to -37 points); in 3 m/s wind the planner crashes in 32% of the flights, the policy network in 0%.
- Snapshots: best (validation) vs final, planner +7 ± 4 points (49% vs 42% from the training area). **Policy network alone at its validation-chosen snapshot: 90 ± 6% / 81 ± 11% / 59 ± 8% test success (training area / held-out / outside), 0 / 0 / 6% crashes, median late error 7.2 cm**, vs 60 / 62 / 39% at the final snapshot (+30 ± 9 points). This is as good as the small model's planner at its final snapshot, at 0.3 ms per decision instead of 11 ms on a CPU.
- Value probe (seeds 7 and 3 at 1M): flat like the small model's (0.001-0.002 within 10 cm, 0.005-0.007 within 20 cm).

**C1 final (published model, single start, metric target; 3 seeds).** Planning: 26 ± 7% / 35 ± 10% (training area), 4 ± 3% / 82 ± 11% (held-out), 0% / 92 ± 4% (outside). Policy network: 47 / 15%, 18 / 54%, 0 / 98%. C1 - Batch 1 published 1M: +6 [-9, +21], +2 [-3, +8], -2 [-3, +0]: the metric target offset changes nothing; R1's improvement comes from the random starts. Same exploitation signature as Batch 1 (planner / policy model error 1.68 ± 0.35; narrower search -10 points of crashes).

**H2 final (small model, random starts + randomised mass, motors, wind; 3 seeds).**
- Without disturbance: planning 13 ± 7% / 0% (training area), 7 ± 4% / 0% (held-out), 4 ± 4% / 0% (outside); policy network 34 ± 31% / 0%. Precision of full flights 14.3 ± 2.0 cm (planner), 12.5 ± 2.9 cm (policy), vs 7.4 cm for D.
- H2 - D (planning): -65 [-86, -42], -76 [-94, -55], -55 [-73, -35].
- In wind: crashes at 4 m/s 34% (D 41%), 6 m/s gusts 55% (D 67%); late error of full flights at 3 m/s 47 cm (D 67 cm). At 2-3 m/s no success for either.
- Planner - policy: -21 [-80, +16] (3 seeds, wide).
- **Reading:** H2 not supported in this pilot. Training on varied physics bought a little robustness in strong wind but cost most of the precision in calm air, and the planner does not beat the policy network. A likely reason: the agent sees no history, so it cannot tell which mass or wind it is flying in, and its world model predicts an average; that average is wrong in every particular case. Testing this needs a history or an estimate of the disturbance in the observation.

**Reward pilot final (sharp reward near the target, small model, 3 seeds).**
- Final snapshots: planning 37 ± 19% / 0% (training area), 66 ± 33% / 0% (held-out), 23 ± 17% / 2 ± 2% (outside); D - pilot +41 [+7, +78] from the training area. Seed 2's final snapshot is at 0%.
- Best snapshot on validation: 91 ± 5% / 78 ± 19% / 47 ± 3% (D: 97 / 91 / 70%).
- Drift from 500k: success SD 0.25 ± 0.04 (D 0.34 ± 0.03; test noise 0.06-0.07); median late error per snapshot 9.3 cm on average (D about 10 cm); late-error SD 2.6 ± 0.6 cm (D 11.0 ± 3.2 cm, driven by excursions); snapshots with a median error above 20 cm: 0 of 18 (D 9 of 66).
- Value probe (6 snapshots): the value now peaks at the target: it drops by 0.03-0.05 at 10 cm and 0.07-0.13 at 20 cm (D: about 0.00), and its maximum is 2-3 cm from the target in 3 of 6 snapshots.
- **Reading:** the extra reward is learned (the value now sees the target), and the large excursions of precision disappeared in these 3 seeds, but the typical precision and the swings of success around the 10 cm threshold are unchanged. So the flat value was not the cause of the drift. Not a cure; a pilot that narrows the search.

**09:30: documents finished.** `GUIDE.pdf` complete with every final result (66 pages; Section 5.18 waits for the 2M run), `BATCH2_EXPLAINED.pdf` (a 14-page plain-language version: the experiment names, what each one is for, the results by question, where every file is), `RESULTS_SUMMARY.md`, `SUPERVISOR_UPDATE.md`, `NEXT_STEPS.md`, `HANDOFF.md`, registry statuses. Batch 2 copied to `/project` (scratch purge). Still running: R1 to 2M (7669543, then 7669544).

**12:00: R1 continued to 2M steps, final (seeds 0-2; 7669542-7669544 all exit 0; backed up to /project, 901 files, identical).** Paired, same seeds at 1M -> 2M, success / crashes:
- planning: training area 66/0 -> 72/3%, held-out 65/2 -> 75/7%, outside 33/22 -> 24/58%; typical final error per seed 12.3, 8.8, 7.7 -> 7.2, 10.3, 3.1 cm;
- 64x2 planner: 51/0 -> 70/0%, 33/0 -> 70/0%, 39/6 -> 57/22%;
- policy network only: 36/0 -> 29/0%, 53/0 -> 30/0%, 20/3 -> 18/34%.
- **Reading:** the second million adds a few points and precision inside the area, much less than the first million, and makes the agents crash far more from outside starts. Hypothesis: the replay buffer holds 1M transitions, so from 1M on it discards the early, wide-ranging flights; the agent then trains only on competent flights inside the area and specialises. Both the planner and the policy network crash more outside, which points at the data. To check: distance distributions in the 1M and 2M buffers. At 2M the published planner is level with the small model at 1M inside the area and far less safe outside, after about 14 h of training against 4 h.
- 300 Slurm tasks completed for Batch 2 in total, none failed. Guide Section 5.18, the summaries and both PDFs updated.

# Part 13. Batch 1 context (historical, October 13)

Source: `code/src/tdmpc2_training/CONTEXT.md`, last modified 2026-10-13.

## TD-MPC2 Crazyflie hover: project context

Last updated: October 13, 2026. Picks up where `CLAUDE_PROJECT_CONTEXT.md` (Sept 28 handoff, kept local
only) left off. Read this first when continuing the project on any machine.

### 1. Where things stand in one paragraph

TD-MPC2 (JAX version by Shane Flandermeyer) is integrated with the lab's MJX Crazyflie environment and
was trained on a one-drone hover task: start (0, 0, 0.2) m, target (0, 0, 1) m, 500-step (10 s)
flights. Batch 1 is finished: 11-seed experiments for TD-MPC2 (small and published settings), PPO on the
identical task, and a 3-seed 1-million-step TD-MPC2 run, all tested from 6 start positions and written
up in a PDF report. Most experiments ran on the Digital Research Alliance cluster Narval. Main result:
TD-MPC2 hovers within about 6 cm with roughly 100x fewer environment steps than PPO, but every method
trained from a single start only learns that start. Next is Batch 2 (random start and goal each episode).

### 2. What Professor Bellinger asked for

From his reply to the Sept 28 slides:

- 1. Keep working on understanding TD-MPC2.
- 2. Test varied starts, then compare with the existing PPO work, before investigating parallel scaling.
- 3. Do this with multiple seeds (e.g. 11). Plots: mean + standard error, and all seeds as thin lines with
   the median seed as a thick line.
- 4. Second batch: randomized start and goal locations in every episode.

Status: 1 ongoing, 2 and 3 done (Batch 1), 4 not started.

### 3. Machines and environments

**Home PC (WSL Ubuntu)**: repo at `~/Projects/multi-agentCrazyflieLift`, Python at `.venv/bin/python`
(3.11.16, made with uv), RTX 3060 12 GB. Key versions: jax 0.7.2, mujoco/mujoco-mjx 3.4.0, flax 0.10.7,
brax 0.13.0, playground 0.0.5, gymnasium 1.2.0, hydra-core 1.3.2, orbax-checkpoint 0.12.5. The Windows
copy at `C:\Users\Mounir\Projects\...` is old and separate; always work in WSL.

**Narval (Digital Research Alliance)**: see section 6.

### 4. Git state

- Branch `tdmpc2-hover`, commit `5730489` "Add TD-MPC2 hover experiments, cluster setup and Batch 1
  results" (364 files, about 38 MB). `main` untouched.
- Git identity for this repo only: Mounir Zouhari <zouharimounir1@gmail.com>.
- Remotes:
  - `origin` = `uo-mirl-lab/multi-agentCrazyflieLift` (lab repo). Push was refused (403): the GitHub
    account MounirZouhari has no write access. Ask Prof. Bellinger to add it, then
    `git push -u origin tdmpc2-hover`.
  - `mine` = `https://github.com/MounirZouhari/crazyfly.git` (private). Push with
    `git push -u mine tdmpc2-hover` from the Ubuntu terminal (GitHub sign-in only works interactively).
- WSL git uses the Windows Git Credential Manager (set in `~/.gitconfig`).
- Kept out of git on purpose via `.git/info/exclude` (local only): `CLAUDE_PROJECT_CONTEXT.md`, the
  weekly-update `.pptx`, `*:Zone.Identifier` files, `outputs.zip`. `outputs/` is in `.gitignore`.
- Results that are in git live in `results/tdmpc2_batch1/` (curated copy of `outputs/`, see its README).

### 5. Code map (code/src/tdmpc2_training)

| File | What it does |
| --- | --- |
| `crazyflie_hover.py` | Gymnasium adapter around the MJX env. `place_drone()` sets any start/target. Reset is jit-compiled and each step does one GPU to CPU copy (about 44 steps/s at home, was about 30). |
| `train_hover.py` | Entry point; registers `CrazyflieHover-v0`, uses `config_hover` unless `--config-name` is given. |
| `tdmpc2_jax/train.py` | Vendored trainer. Changes: agent built by `build.py`; test flights every `eval_interval_steps` appended to `learning_curve.csv` (with `train_seconds`, test time excluded); final checkpoint always saved; old end-of-run gym evaluation removed. |
| `tdmpc2_jax/build.py` | Builds the agent from a Hydra config; shared by training and evaluation so checkpoints always restore into matching shapes. |
| `tdmpc2_jax/config_hover.yaml` | Small settings (latent 128, 128 plans x 3 iterations, batch 64, 500 seed steps, 100 seed updates). Adds `eval_interval_steps: 5000`, `eval_flights: 3`, `eval_flight_steps: 500`. |
| `tdmpc2_jax/config_hover_standard.yaml` | Published TD-MPC2 defaults on top of `config_hover`: encoder 256, latent 512, 512 plans x 6 iterations, 64 elites, 24 policy samples, temperature 0.5, batch 256, 2500 seed steps and updates, encoder lr 9e-5. |
| `hover_eval.py` | Test flights. All flights in one call fly in parallel inside one compiled `lax.scan`. Defines `TEST_STARTS` and the metrics (see section 8). |
| `eval_hover.py` | Reloads a run's newest checkpoint from its own `.hydra/config.yaml` and flies 3 flights from each of the 6 starts. Writes `evaluation/summary.csv` and `trajectories.csv`. |
| `run_seeds.py` | Runs a seed list locally (`--parallel 2` works on the 3060), skips finished seeds, records `batch_info.json`, then calls `analyze_seeds.py`. |
| `analyze_seeds.py` | Per-batch plots (mean + SE, all seeds + median seed) and `analysis/summary.md`. Median seed = median late error from the training start. Flight curves stop at the first crash. |
| `compare_methods.py` | Side-by-side plots: by environment steps (log), by training time, by start position, plus a table. |
| `ppo_hover.py` | Trains the lab's Brax PPO (gpu_training/ppo_training.py defaults) on the exact same task: fixed reset, 500-step episodes, same action rescaling as TD-MPC2. `--load ... --raw-actions` tests an existing PPO checkpoint the same way. |
| `cluster/` | Narval scripts, see section 6. |

Run everything from the repo root, e.g.
`.venv/bin/python code/src/tdmpc2_training/run_seeds.py --seeds 0-10 --parallel 2`.

### 6. Digital Research Alliance (Narval)

### Account and login
- CCDB user `mounirz` (CCI bwd-893), group `def-cbelling-ab` (sponsor Colin Bellinger). Access on Fir,
  Narval, Nibi, Rorqual, Trillium, "opportunistic use" (default priority, no RAC). Project storage 1 TB.
- MFA (Duo) and per-cluster access activation (ccdb.alliancecan.ca/me/access_systems) were done on Oct 12.
- `~/.ssh/config` in WSL defines `Host narval` with ControlMaster (`ControlPersist 4h`). The user logs in
  once with `ssh narval` (password + Duo push) in the Ubuntu terminal; other commands can then reuse that
  connection with `ssh -o BatchMode=yes narval ...`. The shared login does not survive overnight; if it
  is stale, `ssh -O exit narval`, remove `~/.ssh/cm-*`, and log in again.
- Host key fingerprint (ED25519): `SHA256:pTKCWpDC142truNtohGm10+lB8gVyrp3Daz4iR5tT1M`.

### Storage on Narval
- Code: `~/multi-agentCrazyflieLift` (copied with rsync, verified byte-identical to home before every
  submission).
- Results: `/scratch/mounirz/tdmpc2/<batch>/seed_XX/` and job logs in `/scratch/mounirz/*.out`.
  **Scratch is purged after about 60 days.** Checkpoints exist only there (and in `outputs/` at home for
  the home runs). Copy anything worth keeping to `/project/def-cbelling-ab/` (backed up, not done yet).

### Python on Narval (why a container)
The Alliance wheelhouse has no MuJoCo 3.4.0 (MJX only up to 3.3.0) and different flax/orbax/brax
versions, and its Python refuses normal PyPI Linux wheels. To keep the physics identical to home, Python
runs inside a plain `python:3.11-slim` Apptainer image:
- `cluster/setup_env.sh` (login node, once): pulls `~/containers/python311.sif`, creates venv
  `~/envs/tdmpc2-ct`, installs `cluster/requirements-lock.txt` (exact `uv pip freeze` of the home venv).
  Verified: all 99 packages identical to home.
- `mujoco_playground` downloads MuJoCo Menagerie on import, which fails on offline GPU nodes. The home
  copy (commit 14ceccf, 1704 files, no .git) was copied into
  `~/envs/tdmpc2-ct/lib/python3.11/site-packages/mujoco_playground/external_deps/mujoco_menagerie`.
- `cluster/python.sh` runs the venv's Python in the container with `--nv --cleanenv`.

### Jobs
- GPU type: 2/7 A100 slices, `--gres=gpu:a100_2g.10gb:1`, 2 CPUs, 8 GB. Full A100s had queues of
  hundreds of jobs (estimated start hours away); slices usually started within minutes.
- `cluster/test_job.sh`: setup check + 3000-step speed test of both configs. Note: its printed speed for
  the standard config is misleading because the 1000-3000 window contains the 2500 pre-training updates.
- `cluster/seed_job.sh`: one seed per array task. Env vars `CONFIG` (default `config_hover_standard`),
  `BATCH`, `STEPS`, `EXTRA` (extra Hydra overrides). Submit from `$SCRATCH`, e.g.
  `cd $SCRATCH && sbatch --array=0-10 ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/seed_job.sh`
- `cluster/ppo_job.sh`: PPO, one seed per task, `STEPS` default 20M (Batch 1 used 60M).
- Useful: `sq`, `squeue -u $USER --start`, `sacct -j <id> -X -o JobID,State,Elapsed`, `partition-stats`.
  The CCDB "group usage" web page updates with a delay of about a day.

### Measured speeds (environment steps per second, steady state)
| Run | Home RTX 3060 | Narval 2/7 slice |
| --- | --- | --- |
| TD-MPC2 small | 44 (34 each with 2 in parallel) | 89 |
| TD-MPC2 standard | not measured | 42-49 |
| PPO, 2048 parallel envs | not measured | 79M steps in about 11.4 min |

### Jobs run on Oct 12-13
| Job ID | What | Outcome |
| --- | --- | --- |
| 5418119 | test, full A100 | cancelled (queue estimate 2 a.m.) |
| 5418330 | test, slice | failed: Menagerie download (fixed) |
| 5418777 | test, slice | completed |
| 5420951 | TD-MPC2 standard, 11 seeds x 50k | completed |
| 5422314 | TD-MPC2 standard, 3 seeds x 1M (8 h limit, about 6.4-7 h each) | completed |
| 5425800 | PPO smoke test + lab checkpoint test | completed |
| 5425898 | PPO, 11 seeds x 60M (Brax rounded to 78.6M) | completed |
| 5425899 | TD-MPC2 small, 11 seeds x 50k (same-hardware rerun) | completed |

About 17.6 slice-hours used by the evening of Oct 12, plus about 20 for the 1M runs.

### 7. Batch 1 results

All runs: same task, reward, observations and action meaning. Every trained model flew 3 test flights
from each of 6 starts (home (0,0,0.2) = training start; plus_x, minus_x, plus_y, minus_y at 0.5 m to the
side, height 0.2; above (0,0,1.5)), target (0,0,1). Values are mean ± standard error across seeds.

| ID | Experiment | Where | Seeds | Late error, training start | Crashes, all 6 starts | Folder |
| --- | --- | --- | --- | --- | --- | --- |
| A | September run, re-tested | home | 1 | 0.30 m (other starts 0.18-0.38 m) | 0/18 | `crazyflie_hover_s0` |
| B | TD-MPC2 small, 50k | home | 11 | 0.54 ± 0.19 m (median 0.34) | 28/198 | `batch1_fixed_start` |
| C | TD-MPC2 small, 50k | Narval | 11 | 0.32 ± 0.09 m (median 0.16) | 11/198 | `batch1_small_narval` |
| D | TD-MPC2 standard, 50k | Narval | 11 | 1.06 ± 0.14 m (median 1.02) | 182/198 | `batch1_standard` |
| G | TD-MPC2 standard, 1M | Narval | 3 | 0.057 ± 0.005 m, 92% of time within 10 cm | 36/54 (0/9 from home, 36/36 from the 4 sideways starts) | `batch1_standard_1M` |
| E | PPO, same task, 78.6M steps | Narval | 11 | 0.17 ± 0.09 m (median 0.028; 7 of 11 within 10 cm all flight) | 157/198 | `ppo_fixed_start` |
| F | Lab's existing PPO model (trained on random tasks, 60M) | tested on Narval | 1 | 0.064 m (3.5-8.8 cm from 5 starts, 21 cm from above) | 0/18 | `ppo_existing_checkpoint` |

Training time per seed on a Narval slice: small 14.0 min, standard 20.2 min, PPO 11.4 min, 1M about 6 h.
B took 26 min per seed at home with 2 seeds sharing the GPU (not comparable).

### 8. Interpretation and things learned

- **Seed variation is large.** Same settings gave 0.08 m to 2.35 m at 50k steps. B and C agree within
  their standard errors (22 runs of the small settings in total).
- **Height is learned quickly, sideways position is not.** At 50k steps the drone reaches 1 m in about
  0.8 s and holds it, then drifts about 3 cm/s sideways in a random direction per seed.
- **The drift was mainly lack of practice.** With 1M steps (G) all 3 seeds hover within about 5-6 cm with
  no crashes from the training start; they first got under 10 cm at 105k, 130k and 175k steps.
- **Sample efficiency vs wall clock.** TD-MPC2 reaches a precise hover after about 150-200k steps; PPO
  needs about 15-20M (roughly 100x more). PPO is far faster on the clock (2048 parallel envs): best
  hover after about 4 min versus about an hour for TD-MPC2.
- **Both methods memorise the single training start.** PPO and the 1M TD-MPC2 crash from almost all
  sideways starts. The 50k TD-MPC2 looked more robust only because it had not specialised yet (this
  corrects a claim made on Oct 12). The lab's PPO trained on random starts works everywhere, which
  motivates Batch 2.
- **Late training instability.** B seed 0 was at 0.27 m with no crashes at 45k steps, then collapsed to
  2.3 m by 50k (confirmed through two independent test paths). PPO peaks around 20-30M steps, then
  degrades. Final-checkpoint scores include this.
- **Not bit-for-bit repeatable.** GPU arithmetic plus the faster reset mean seed 0 today is not seed 0 of
  September (0.30 m then, 2.35 m now).
- **Bigger settings are slow starters**: worst at 50k, best TD-MPC2 result with long training.

### 9. Known caveats in the setup

- The "previous action" in the observation lags one step (built before the action history is updated in
  `crazyflie_env._get_obs`). Same for every method.
- The target direction in the observation is divided by the distance at the start of the flight (0.8 m
  from home, 0.94 m from the sideways starts), so the same physical state looks different depending on
  the start. Relevant for single-start generalisation.
- JAX TD-MPC2 vs official: dropout stays on in planning, Q averaged over all 5 heads, AdamW per network,
  entropy not scaled by action dim, `deterministic=True` takes the best elite (tested: sampling elites
  instead made no difference).
- Late-flight error (mean distance over the last 2 s flown) is not comparable between crashed and
  surviving flights; read it with the crash count.
- Each seed is tested with only 3 flights per start.
- Lab PPO code: hidden sizes passed to `build_ppo_train_fn` are ignored, so stock Brax networks are used
  (policy 4x32, value 5x256). Brax rounds training steps up in chunks of about 1M per curve point.
- `crazyflie_env.reset(random_reset=True)` draws target x/y and z from the same key (and likewise for the
  start), and also randomises orientation, velocity, mass, inertia and motor gain. Batch 2 should use its
  own start/goal sampling through `place_drone`, not that mode.

### 10. Report

`outputs/report_batch1/Crazyflie_TD-MPC2_vs_PPO_Batch1_Report.pdf` (copy in
`results/tdmpc2_batch1/report_batch1/`), 20 pages, written for a reader new to RL. Rebuild with
`make_report.py` in the same folder; reportlab is not in the project venv, so run it with
`~/.local/bin/uv run --no-project --with reportlab --with matplotlib --with numpy python make_report.py`.
It reads `outputs/`.

### 11. Next steps

- 1. Push `tdmpc2-hover` to `mine` (and to `origin` once write access is granted).
- 2. Back up `/scratch/mounirz/tdmpc2` (checkpoints) to `/project/def-cbelling-ab/` before scratch is purged.
- 3. Confirm experiment G properly: 11 seeds x about 200k steps, for both small and standard settings
   (about 1.2 h per seed on a slice).
- 4. Batch 2: random start and goal every episode, 11 seeds per method, same plots. Proposed ranges (to
   confirm with Prof. Bellinger): x and y in ±1 m, height 0.3-1.5 m, start and goal at least 0.3 m apart,
   positions only (drone level and at rest, no physics randomisation). Needs a new environment option and
   test set; likely more steps than 50k per run.
- 5. Later: the research question on parallel scaling (TD-MPC2 with many simulated drones), with the
   pipeline profiled first.

### 12. Working with Mounir

New to RL, TD-MPC2 and JAX. Wants explanations from the basics, tied to real files and numbers, in small
steps, with clear separation between "it runs" and "it is validated". Prefers asking before long
training runs, no changes to existing code without saying so, honest corrections when a conclusion turns
out wrong, and no long dashes or AI attribution in commit messages.

# Part 14. Raw result tables

Every analysis writes a `summary.md`; these are copied unchanged, in the order R1, C1, D, H2, reward pilot, R1 to 2M, then the follow-up tests and Batch 1. Plot file names refer to images next to each summary, not included here.


## outputs/batch2/random_start_1M/analysis/summary.md

Last modified 2026-11-02.

### random_start_1M: TD-MPC2 hover, Batch 2, 11 seeds

Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Steps: 1000000. Settings: --config-name=config_batch2_random_start. Code: 8edab5b44dd4b68ffcc8f5898657dfbb8e4631ee.

Success = full 10 s flight with late error below 10 cm. Values: mean ± standard error across seeds (each seed first averaged over its flights). Cells: success / crashes.

### Final agents

| batch | decision rule | training area (seen kind of start) | held-out block (never trained on) | outside the training area | the six Batch 1 starts |
|---|---|---:|---:|---:|---:|
| random_start_1M | with planning (as trained) | 46 ± 8% / 0 ± 0% | 39 ± 9% / 2 ± 1% | 20 ± 6% / 26 ± 5% | 39 ± 10% / 0 ± 0% |
| random_start_1M | policy network only | 60 ± 11% / 0 ± 0% | 64 ± 13% / 0 ± 0% | 40 ± 10% / 6 ± 2% | 65 ± 11% / 0 ± 0% |
| random_start_1M | small planner 64x2 | 58 ± 9% / 0 ± 0% | 46 ± 12% / 0 ± 0% | 32 ± 7% / 9 ± 2% | 57 ± 8% / 0 ± 0% |
| fixed_start_1M | with planning (as trained) | 26 ± 7% / 35 ± 10% | 4 ± 3% / 82 ± 11% | 0 ± 0% / 92 ± 4% | 41 ± 5% / 52 ± 5% |
| fixed_start_1M | policy network only | 47 ± 21% / 15 ± 1% | 18 ± 11% / 54 ± 18% | 0 ± 0% / 98 ± 1% | 39 ± 15% / 28 ± 6% |
| fixed_start_1M | small planner 64x2 | 41 ± 12% / 30 ± 13% | 12 ± 11% / 69 ± 16% | 2 ± 1% / 90 ± 5% | 39 ± 14% / 44 ± 3% |
| random_start_small_1M | with planning (as trained) | 78 ± 10% / 0 ± 0% | 83 ± 10% / 0 ± 0% | 59 ± 9% / 3 ± 2% | 80 ± 10% / 0 ± 0% |
| random_start_small_1M | policy network only | 9 ± 9% / 0 ± 0% | 9 ± 9% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 11 ± 9% / 0 ± 0% |
| random_start_small_1M | small planner 64x2 | 76 ± 11% / 0 ± 0% | 77 ± 10% / 0 ± 0% | 60 ± 10% / 2 ± 2% | 79 ± 10% / 0 ± 0% |
| randomized_small_1M | with planning (as trained) | 13 ± 7% / 0 ± 0% | 7 ± 4% / 0 ± 0% | 4 ± 4% / 0 ± 0% | 6 ± 6% / 0 ± 0% |
| randomized_small_1M | policy network only | 34 ± 31% / 0 ± 0% | 34 ± 33% / 0 ± 0% | 27 ± 23% / 1 ± 1% | 33 ± 33% / 0 ± 0% |
| randomized_small_1M | small planner 64x2 | 11 ± 6% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 8 ± 8% / 1 ± 1% | 7 ± 7% / 0 ± 0% |
| batch1_standard_1M | with planning (as trained) | 20 ± 6% / 66 ± 3% | 2 ± 2% / 91 ± 9% | 2 ± 1% / 87 ± 5% | 20 ± 7% / 59 ± 4% |
| batch1_standard_1M | policy network only | 46 ± 12% / 15 ± 7% | 22 ± 17% / 54 ± 14% | 0 ± 0% / 77 ± 11% | 33 ± 17% / 6 ± 6% |

### random_start_1M, each seed (with planning)

| seed | train_area success | held_out success | outside success | batch1 success | train_area median late error, full flights (m) |
|---:|---:|---:|---:|---:|---:|
| 0 | 36% | 60% | 23% | 44% | 0.123 |
| 1 | 82% | 73% | 57% | 44% | 0.088 |
| 2 | 80% | 63% | 20% | 78% | 0.077 |
| 3 | 20% | 0% | 3% | 17% | 0.155 |
| 4 | 16% | 3% | 0% | 0% | 0.163 |
| 5 | 12% | 3% | 3% | 6% | 0.187 |
| 6 | 60% | 50% | 17% | 89% | 0.066 |
| 7 | 72% | 63% | 10% | 67% | 0.085 |
| 8 (median) | 38% | 10% | 17% | 22% | 0.112 |
| 9 | 72% | 60% | 60% | 67% | 0.067 |
| 10 | 14% | 40% | 7% | 0% | 0.182 |

### Statistical comparisons (95% bootstrap intervals over seeds)

Differences in success share (percentage points). Intervals from fewer than about 5 seeds are unreliable.

| comparison | train_area | held_out | outside | batch1 |
|---|---:|---:|---:|---:|
| random_start_1M - fixed_start_1M (with planning) | +20 [+1, +39] | +34 [+17, +51] | +20 [+9, +32] | -1 [-21, +19] |
| random_start_1M - random_start_small_1M (with planning) | -32 [-57, -5] | -44 [-68, -18] | -39 [-59, -18] | -40 [-66, -13] |
| random_start_1M - randomized_small_1M (with planning) | +33 [+15, +52] | +32 [+14, +49] | +15 [+2, +29] | +34 [+14, +55] |
| random_start_1M - batch1_standard_1M (with planning) | +26 [+7, +45] | +37 [+20, +52] | +17 [+7, +30] | +19 [-2, +40] |
| random_start_1M: planner - policy network only (paired) | -15 [-37, +10] | -25 [-47, -1] | -20 [-40, +0] | -26 [-50, +2] |
| random_start_1M: planner - small planner 64x2 (paired) | -12 [-24, +3] | -8 [-24, +10] | -12 [-18, -6] | -18 [-29, -7] |
| fixed_start_1M: planner - policy network only (paired) | -21 [-60, +0] | -13 [-37, +3] | +0 [+0, +0] | +2 [-28, +17] |
| fixed_start_1M: planner - small planner 64x2 (paired) | -15 [-32, -4] | -8 [-23, +3] | -2 [-3, +0] | +2 [-17, +22] |
| random_start_small_1M: planner - policy network only (paired) | +69 [+30, +96] | +74 [+36, +98] | +48 [+18, +72] | +69 [+32, +95] |
| random_start_small_1M: planner - small planner 64x2 (paired) | +2 [-2, +7] | +6 [-1, +15] | -1 [-4, +2] | +1 [-4, +4] |
| randomized_small_1M: planner - policy network only (paired) | -21 [-80, +16] | -28 [-87, +3] | -22 [-60, +0] | -28 [-83, +0] |
| randomized_small_1M: planner - small planner 64x2 (paired) | +2 [-4, +10] | -6 [-13, +0] | -3 [-10, +0] | -2 [-6, +0] |
| batch1_standard_1M: planner - policy network only (paired) | -26 [-46, -8] | -20 [-50, -3] | +2 [+0, +3] | -13 [-50, +17] |

### Disturbances (success share on the 50 training-area starts, mean across seeds)

| batch | decision rule | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_1M | with planning (as trained) | 43% | 36% | 1% | 0% | 0% | 37% | 7% | 0% | 32% | 20% | 15% | 33% | 25% | 33% | 25% |
| random_start_1M | policy network only | 60% | 63% | 1% | 0% | 0% | 57% | 23% | 2% | 57% | 54% | 51% | 57% | 54% | 60% | 60% |
| fixed_start_1M | with planning (as trained) | 29% | 38% | 0% | 0% | 0% | 23% | 1% | 0% | 23% | 21% | 15% | 24% | 23% | 28% | 17% |
| fixed_start_1M | policy network only | 45% | 33% | 0% | 0% | 0% | 40% | 1% | 0% | 44% | 43% | 41% | 44% | 41% | 46% | 44% |
| random_start_small_1M | with planning (as trained) | 77% | 60% | 1% | 0% | 0% | 66% | 15% | 0% | 76% | 74% | 72% | 76% | 73% | 77% | 76% |
| random_start_small_1M | policy network only | 9% | 18% | 0% | 0% | 0% | 9% | 0% | 0% | 9% | 10% | 10% | 9% | 10% | 9% | 9% |
| randomized_small_1M | with planning (as trained) | 10% | 14% | 0% | 0% | 0% | 6% | 0% | 0% | 8% | 10% | 10% | 8% | 9% | 13% | 11% |
| randomized_small_1M | policy network only | 35% | 35% | 0% | 0% | 0% | 34% | 0% | 0% | 33% | 31% | 28% | 33% | 31% | 35% | 26% |
| batch1_standard_1M | with planning (as trained) | 21% | 15% | 0% | 0% | 0% | 17% | 0% | 0% | 15% | 10% | 9% | 15% | 12% | 16% | 12% |
| batch1_standard_1M | policy network only | 46% | 15% | 0% | 0% | 0% | 41% | 0% | 0% | 45% | 41% | 36% | 44% | 40% | 45% | 41% |

Planning minus policy network only, success points, paired by seed (95% bootstrap interval):

| batch | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_1M (11 seeds) | -17 [-38, +6] | -27 [-43, -9] | +0 [+0, +1] | +0 [+0, +0] | +0 [+0, +0] | -20 [-40, +3] | -17 [-36, +0] | -2 [-6, +0] | -25 [-47, -1] | -34 [-53, -14] | -37 [-57, -16] | -24 [-46, +0] | -29 [-49, -7] | -27 [-50, -3] | -35 [-55, -13] |
| fixed_start_1M (3 seeds) | -16 [-64, +8] | +5 [-32, +24] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -17 [-60, +14] | +0 [+0, +0] | +0 [+0, +0] | -21 [-60, +0] | -22 [-66, +12] | -27 [-64, -8] | -20 [-66, +4] | -18 [-54, +2] | -18 [-58, +8] | -27 [-72, +4] |
| random_start_small_1M (11 seeds) | +67 [+29, +95] | +42 [+14, +69] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +57 [+16, +91] | +15 [+0, +34] | +0 [+0, +0] | +67 [+28, +95] | +64 [+25, +93] | +61 [+23, +91] | +66 [+28, +94] | +63 [+23, +93] | +68 [+30, +95] | +67 [+30, +95] |
| randomized_small_1M (3 seeds) | -25 [-82, +8] | -21 [-80, +16] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -28 [-62, +0] | +0 [+0, +0] | +0 [+0, +0] | -25 [-82, +6] | -21 [-64, +0] | -18 [-52, +2] | -25 [-80, +4] | -22 [-64, +0] | -22 [-74, +8] | -15 [-34, +0] |
| batch1_standard_1M (3 seeds) | -25 [-42, -4] | -1 [-12, +10] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -24 [-38, -6] | +0 [+0, +0] | +0 [+0, +0] | -31 [-44, -18] | -31 [-60, -10] | -27 [-54, -10] | -29 [-50, -10] | -28 [-50, -14] | -29 [-46, -14] | -29 [-42, -22] |

random_start_1M minus each other batch, success points (95% bootstrap interval over seeds):

| comparison | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_1M - fixed_start_1M, with planning (as trained) | +13 [-7, +33] | -2 [-16, +14] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +14 [-4, +32] | +6 [+3, +10] | +0 [+0, +1] | +8 [-6, +24] | -2 [-21, +16] | -0 [-13, +14] | +9 [-8, +26] | +2 [-14, +19] | +5 [-15, +24] | +8 [-10, +25] |
| random_start_1M - fixed_start_1M, policy network only | +14 [-28, +52] | +31 [-14, +69] | +1 [+0, +2] | +0 [+0, +0] | +0 [+0, +0] | +17 [-22, +51] | +23 [+5, +43] | +2 [+0, +7] | +13 [-27, +49] | +11 [-30, +46] | +10 [-30, +46] | +13 [-28, +49] | +13 [-28, +50] | +14 [-25, +48] | +16 [-22, +49] |
| random_start_1M - random_start_small_1M, with planning (as trained) | -34 [-59, -7] | -24 [-47, +2] | +0 [-2, +3] | +0 [+0, +0] | +0 [+0, +0] | -29 [-57, +1] | -8 [-27, +7] | +0 [+0, +1] | -44 [-67, -18] | -54 [-76, -29] | -57 [-80, -31] | -43 [-67, -17] | -49 [-73, -22] | -45 [-68, -19] | -51 [-73, -26] |
| random_start_1M - random_start_small_1M, policy network only | +51 [+22, +75] | +45 [+14, +73] | +1 [+0, +2] | +0 [+0, +0] | +0 [+0, +0] | +48 [+18, +75] | +23 [+5, +44] | +2 [+0, +7] | +48 [+19, +73] | +44 [+15, +70] | +41 [+13, +66] | +47 [+19, +73] | +44 [+15, +70] | +51 [+22, +76] | +51 [+22, +75] |
| random_start_1M - randomized_small_1M, with planning (as trained) | +33 [+16, +51] | +22 [+7, +39] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +31 [+15, +48] | +7 [+4, +10] | +0 [+0, +1] | +24 [+10, +39] | +10 [-2, +22] | +5 [-7, +17] | +25 [+10, +40] | +16 [+2, +31] | +20 [+3, +38] | +15 [+1, +29] |
| random_start_1M - randomized_small_1M, policy network only | +25 [-35, +72] | +28 [-33, +74] | +1 [+0, +2] | +0 [+0, +0] | +0 [+0, +0] | +23 [-23, +65] | +23 [+5, +43] | +2 [+0, +7] | +24 [-34, +69] | +23 [-27, +64] | +23 [-21, +61] | +23 [-35, +69] | +23 [-25, +64] | +26 [-33, +71] | +34 [+2, +66] |
| random_start_1M - batch1_standard_1M, with planning (as trained) | +21 [+3, +41] | +22 [+3, +39] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +20 [+2, +40] | +7 [+4, +10] | +0 [+0, +1] | +17 [+1, +34] | +10 [-1, +22] | +6 [-6, +18] | +18 [+2, +35] | +13 [-1, +27] | +17 [+0, +35] | +13 [-1, +28] |
| random_start_1M - batch1_standard_1M, policy network only | +14 [-15, +40] | +48 [+16, +76] | +1 [+0, +2] | +0 [+0, +0] | +0 [+0, +0] | +16 [-13, +43] | +23 [+5, +43] | +2 [+0, +7] | +12 [-19, +40] | +14 [-20, +44] | +15 [-17, +45] | +13 [-17, +41] | +14 [-19, +44] | +16 [-14, +43] | +18 [-10, +45] |

How the crashed flights ended:

| batch | decision rule | conditions | crashed flights | out of bounds | hit the ground |
|---|---|---|---:|---:|---:|
| random_start_1M | with planning (as trained) | no disturbance | 2 | 50% | 50% |
| random_start_1M | with planning (as trained) | wind and gusts | 1171 | 57% | 43% |
| random_start_1M | with planning (as trained) | mass, motors, delay | 43 | 16% | 84% |
| random_start_1M | policy network only | wind and gusts | 570 | 40% | 60% |
| random_start_1M | policy network only | mass, motors, delay | 1 | 0% | 100% |
| fixed_start_1M | with planning (as trained) | no disturbance | 58 | 17% | 83% |
| fixed_start_1M | with planning (as trained) | wind and gusts | 759 | 38% | 62% |
| fixed_start_1M | with planning (as trained) | mass, motors, delay | 486 | 16% | 84% |
| fixed_start_1M | policy network only | no disturbance | 24 | 21% | 79% |
| fixed_start_1M | policy network only | wind and gusts | 650 | 24% | 76% |
| fixed_start_1M | policy network only | mass, motors, delay | 183 | 8% | 92% |
| random_start_small_1M | with planning (as trained) | no disturbance | 1 | 100% | 0% |
| random_start_small_1M | with planning (as trained) | wind and gusts | 635 | 71% | 29% |
| random_start_small_1M | with planning (as trained) | mass, motors, delay | 40 | 65% | 35% |
| random_start_small_1M | policy network only | wind and gusts | 277 | 80% | 20% |
| random_start_small_1M | policy network only | mass, motors, delay | 2 | 0% | 100% |
| randomized_small_1M | with planning (as trained) | wind and gusts | 133 | 80% | 20% |
| randomized_small_1M | policy network only | wind and gusts | 176 | 82% | 18% |

Plots: learning_curves_by_group.png, final_by_group.png, comparison.png, disturbances.png

## outputs/batch2/fixed_start_1M/analysis/summary.md

Last modified 2026-11-02.

### fixed_start_1M: TD-MPC2 hover, Batch 2, 3 seeds

Seeds: 0, 1, 2. Steps: 1000000. Settings: --config-name=config_batch2_fixed_start. Code: 52873af48bf637e2a66007a72505d7fbe3da221a.

Success = full 10 s flight with late error below 10 cm. Values: mean ± standard error across seeds (each seed first averaged over its flights). Cells: success / crashes.

### Final agents

| batch | decision rule | training area (seen kind of start) | held-out block (never trained on) | outside the training area | the six Batch 1 starts |
|---|---|---:|---:|---:|---:|
| fixed_start_1M | with planning (as trained) | 26 ± 7% / 35 ± 10% | 4 ± 3% / 82 ± 11% | 0 ± 0% / 92 ± 4% | 41 ± 5% / 52 ± 5% |
| fixed_start_1M | policy network only | 47 ± 21% / 15 ± 1% | 18 ± 11% / 54 ± 18% | 0 ± 0% / 98 ± 1% | 39 ± 15% / 28 ± 6% |
| fixed_start_1M | small planner 64x2 | 41 ± 12% / 30 ± 13% | 12 ± 11% / 69 ± 16% | 2 ± 1% / 90 ± 5% | 39 ± 14% / 44 ± 3% |
| batch1_standard_1M | with planning (as trained) | 20 ± 6% / 66 ± 3% | 2 ± 2% / 91 ± 9% | 2 ± 1% / 87 ± 5% | 20 ± 7% / 59 ± 4% |
| batch1_standard_1M | policy network only | 46 ± 12% / 15 ± 7% | 22 ± 17% / 54 ± 14% | 0 ± 0% / 77 ± 11% | 33 ± 17% / 6 ± 6% |

### fixed_start_1M, each seed (with planning)

| seed | train_area success | held_out success | outside success | batch1 success | train_area median late error, full flights (m) |
|---:|---:|---:|---:|---:|---:|
| 0 | 14% | 3% | 0% | 33% | 0.205 |
| 1 (median) | 26% | 0% | 0% | 39% | 0.121 |
| 2 | 38% | 10% | 0% | 50% | 0.109 |

### Statistical comparisons (95% bootstrap intervals over seeds)

Differences in success share (percentage points). Intervals from fewer than about 5 seeds are unreliable.

| comparison | train_area | held_out | outside | batch1 |
|---|---:|---:|---:|---:|
| fixed_start_1M - batch1_standard_1M (with planning) | +6 [-9, +21] | +2 [-3, +8] | -2 [-3, +0] | +20 [+7, +33] |
| fixed_start_1M: planner - policy network only (paired) | -21 [-60, +0] | -13 [-37, +3] | +0 [+0, +0] | +2 [-28, +17] |
| fixed_start_1M: planner - small planner 64x2 (paired) | -15 [-32, -4] | -8 [-23, +3] | -2 [-3, +0] | +2 [-17, +22] |
| batch1_standard_1M: planner - policy network only (paired) | -26 [-46, -8] | -20 [-50, -3] | +2 [+0, +3] | -13 [-50, +17] |

### Disturbances (success share on the 50 training-area starts, mean across seeds)

| batch | decision rule | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_start_1M | with planning (as trained) | 29% | 38% | 0% | 0% | 0% | 23% | 1% | 0% | 23% | 21% | 15% | 24% | 23% | 28% | 17% |
| fixed_start_1M | policy network only | 45% | 33% | 0% | 0% | 0% | 40% | 1% | 0% | 44% | 43% | 41% | 44% | 41% | 46% | 44% |
| batch1_standard_1M | with planning (as trained) | 21% | 15% | 0% | 0% | 0% | 17% | 0% | 0% | 15% | 10% | 9% | 15% | 12% | 16% | 12% |
| batch1_standard_1M | policy network only | 46% | 15% | 0% | 0% | 0% | 41% | 0% | 0% | 45% | 41% | 36% | 44% | 40% | 45% | 41% |

Planning minus policy network only, success points, paired by seed (95% bootstrap interval):

| batch | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_start_1M (3 seeds) | -16 [-64, +8] | +5 [-32, +24] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -17 [-60, +14] | +0 [+0, +0] | +0 [+0, +0] | -21 [-60, +0] | -22 [-66, +12] | -27 [-64, -8] | -20 [-66, +4] | -18 [-54, +2] | -18 [-58, +8] | -27 [-72, +4] |
| batch1_standard_1M (3 seeds) | -25 [-42, -4] | -1 [-12, +10] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -24 [-38, -6] | +0 [+0, +0] | +0 [+0, +0] | -31 [-44, -18] | -31 [-60, -10] | -27 [-54, -10] | -29 [-50, -10] | -28 [-50, -14] | -29 [-46, -14] | -29 [-42, -22] |

fixed_start_1M minus each other batch, success points (95% bootstrap interval over seeds):

| comparison | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_start_1M - batch1_standard_1M, with planning (as trained) | +8 [-7, +23] | +23 [+6, +40] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +6 [-10, +22] | +1 [+0, +2] | +0 [+0, +0] | +9 [-4, +21] | +11 [-6, +31] | +6 [-7, +19] | +9 [-3, +22] | +11 [-2, +23] | +12 [-3, +28] | +5 [-10, +22] |
| fixed_start_1M - batch1_standard_1M, policy network only | -1 [-36, +40] | +17 [-23, +62] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -1 [-30, +35] | +1 [+0, +2] | +0 [+0, +0] | -1 [-36, +37] | +3 [-34, +41] | +5 [-31, +42] | +0 [-34, +38] | +1 [-37, +40] | +1 [-32, +37] | +3 [-29, +39] |

How the crashed flights ended:

| batch | decision rule | conditions | crashed flights | out of bounds | hit the ground |
|---|---|---|---:|---:|---:|
| fixed_start_1M | with planning (as trained) | no disturbance | 58 | 17% | 83% |
| fixed_start_1M | with planning (as trained) | wind and gusts | 759 | 38% | 62% |
| fixed_start_1M | with planning (as trained) | mass, motors, delay | 486 | 16% | 84% |
| fixed_start_1M | policy network only | no disturbance | 24 | 21% | 79% |
| fixed_start_1M | policy network only | wind and gusts | 650 | 24% | 76% |
| fixed_start_1M | policy network only | mass, motors, delay | 183 | 8% | 92% |

Plots: learning_curves_by_group.png, final_by_group.png, comparison.png, disturbances.png

## outputs/batch2/random_start_small_1M/analysis/summary.md

Last modified 2026-11-02.

### random_start_small_1M: TD-MPC2 hover, Batch 2, 11 seeds

Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Steps: 500000. Settings: --config-name=config_batch2_random_start_small. Code: 6465634b0d16468dde2c28154f102aa4aeda15ad.

Success = full 10 s flight with late error below 10 cm. Values: mean ± standard error across seeds (each seed first averaged over its flights). Cells: success / crashes.

### Final agents

| batch | decision rule | training area (seen kind of start) | held-out block (never trained on) | outside the training area | the six Batch 1 starts |
|---|---|---:|---:|---:|---:|
| random_start_small_1M | with planning (as trained) | 78 ± 10% / 0 ± 0% | 83 ± 10% / 0 ± 0% | 59 ± 9% / 3 ± 2% | 80 ± 10% / 0 ± 0% |
| random_start_small_1M | policy network only | 9 ± 9% / 0 ± 0% | 9 ± 9% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 11 ± 9% / 0 ± 0% |
| random_start_small_1M | small planner 64x2 | 76 ± 11% / 0 ± 0% | 77 ± 10% / 0 ± 0% | 60 ± 10% / 2 ± 2% | 79 ± 10% / 0 ± 0% |
| randomized_small_1M | with planning (as trained) | 13 ± 7% / 0 ± 0% | 7 ± 4% / 0 ± 0% | 4 ± 4% / 0 ± 0% | 6 ± 6% / 0 ± 0% |
| randomized_small_1M | policy network only | 34 ± 31% / 0 ± 0% | 34 ± 33% / 0 ± 0% | 27 ± 23% / 1 ± 1% | 33 ± 33% / 0 ± 0% |
| randomized_small_1M | small planner 64x2 | 11 ± 6% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 8 ± 8% / 1 ± 1% | 7 ± 7% / 0 ± 0% |
| sharp_small_1M | with planning (as trained) | 37 ± 19% / 0 ± 0% | 66 ± 33% / 0 ± 0% | 23 ± 17% / 2 ± 2% | 31 ± 18% / 0 ± 0% |
| sharp_small_1M | policy network only | 11 ± 11% / 0 ± 0% | 12 ± 12% / 0 ± 0% | 6 ± 4% / 3 ± 3% | 11 ± 11% / 0 ± 0% |
| sharp_small_1M | small planner 64x2 | 36 ± 20% / 0 ± 0% | 64 ± 32% / 0 ± 0% | 19 ± 10% / 2 ± 2% | 24 ± 16% / 0 ± 0% |
| batch1_small_narval | with planning (as trained) | 19 ± 7% / 9 ± 3% | 11 ± 5% / 15 ± 7% | 4 ± 1% / 31 ± 6% | 18 ± 7% / 6 ± 3% |
| batch1_small_narval | policy network only | 4 ± 4% / 6 ± 4% | 4 ± 4% / 1 ± 1% | 0 ± 0% / 17 ± 5% | 2 ± 2% / 2 ± 2% |

### random_start_small_1M, each seed (with planning)

| seed | train_area success | held_out success | outside success | batch1 success | train_area median late error, full flights (m) |
|---:|---:|---:|---:|---:|---:|
| 0 | 86% | 93% | 70% | 83% | 0.069 |
| 1 | 100% | 100% | 83% | 100% | 0.050 |
| 2 (median) | 98% | 97% | 50% | 100% | 0.052 |
| 3 | 54% | 33% | 23% | 61% | 0.094 |
| 4 | 100% | 100% | 93% | 100% | 0.054 |
| 5 | 100% | 100% | 90% | 100% | 0.052 |
| 6 | 92% | 100% | 53% | 100% | 0.076 |
| 7 | 100% | 100% | 73% | 100% | 0.073 |
| 8 | 20% | 87% | 7% | 17% | 0.134 |
| 9 | 100% | 100% | 87% | 100% | 0.042 |
| 10 | 8% | 0% | 20% | 17% | 0.129 |

### Statistical comparisons (95% bootstrap intervals over seeds)

Differences in success share (percentage points). Intervals from fewer than about 5 seeds are unreliable.

| comparison | train_area | held_out | outside | batch1 |
|---|---:|---:|---:|---:|
| random_start_small_1M - randomized_small_1M (with planning) | +65 [+42, +86] | +76 [+55, +94] | +55 [+35, +73] | +74 [+52, +93] |
| random_start_small_1M - sharp_small_1M (with planning) | +41 [+7, +78] | +17 [-29, +79] | +36 [+1, +66] | +48 [+14, +82] |
| random_start_small_1M - batch1_small_narval (with planning) | +59 [+35, +81] | +72 [+48, +91] | +55 [+38, +72] | +62 [+38, +83] |
| random_start_small_1M: planner - policy network only (paired) | +69 [+30, +96] | +74 [+36, +98] | +48 [+18, +72] | +69 [+32, +95] |
| random_start_small_1M: planner - small planner 64x2 (paired) | +2 [-2, +7] | +6 [-1, +15] | -1 [-4, +2] | +1 [-4, +4] |
| randomized_small_1M: planner - policy network only (paired) | -21 [-80, +16] | -28 [-87, +3] | -22 [-60, +0] | -28 [-83, +0] |
| randomized_small_1M: planner - small planner 64x2 (paired) | +2 [-4, +10] | -6 [-13, +0] | -3 [-10, +0] | -2 [-6, +0] |
| sharp_small_1M: planner - policy network only (paired) | +25 [+0, +48] | +53 [+0, +100] | +18 [-3, +57] | +20 [+0, +33] |
| sharp_small_1M: planner - small planner 64x2 (paired) | +1 [-6, +8] | +1 [+0, +3] | +4 [-3, +20] | +7 [+0, +17] |
| batch1_small_narval: planner - policy network only (paired) | +14 [-2, +31] | +7 [-6, +20] | +4 [+1, +6] | +16 [+4, +31] |

### Disturbances (success share on the 50 training-area starts, mean across seeds)

| batch | decision rule | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_small_1M | with planning (as trained) | 77% | 60% | 1% | 0% | 0% | 66% | 15% | 0% | 76% | 74% | 72% | 76% | 73% | 77% | 76% |
| random_start_small_1M | policy network only | 9% | 18% | 0% | 0% | 0% | 9% | 0% | 0% | 9% | 10% | 10% | 9% | 10% | 9% | 9% |
| randomized_small_1M | with planning (as trained) | 10% | 14% | 0% | 0% | 0% | 6% | 0% | 0% | 8% | 10% | 10% | 8% | 9% | 13% | 11% |
| randomized_small_1M | policy network only | 35% | 35% | 0% | 0% | 0% | 34% | 0% | 0% | 33% | 31% | 28% | 33% | 31% | 35% | 26% |

Planning minus policy network only, success points, paired by seed (95% bootstrap interval):

| batch | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_small_1M (11 seeds) | +67 [+29, +95] | +42 [+14, +69] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +57 [+16, +91] | +15 [+0, +34] | +0 [+0, +0] | +67 [+28, +95] | +64 [+25, +93] | +61 [+23, +91] | +66 [+28, +94] | +63 [+23, +93] | +68 [+30, +95] | +67 [+30, +95] |
| randomized_small_1M (3 seeds) | -25 [-82, +8] | -21 [-80, +16] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -28 [-62, +0] | +0 [+0, +0] | +0 [+0, +0] | -25 [-82, +6] | -21 [-64, +0] | -18 [-52, +2] | -25 [-80, +4] | -22 [-64, +0] | -22 [-74, +8] | -15 [-34, +0] |

random_start_small_1M minus each other batch, success points (95% bootstrap interval over seeds):

| comparison | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_small_1M - randomized_small_1M, with planning (as trained) | +67 [+44, +87] | +46 [+20, +70] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +60 [+31, +85] | +15 [+0, +33] | +0 [+0, +0] | +68 [+45, +88] | +64 [+39, +86] | +62 [+36, +84] | +68 [+44, +88] | +65 [+39, +87] | +65 [+41, +86] | +66 [+42, +86] |
| random_start_small_1M - randomized_small_1M, policy network only | -25 [-89, +18] | -17 [-82, +32] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -25 [-71, +13] | +0 [+0, +0] | +0 [+0, +0] | -24 [-85, +18] | -22 [-72, +18] | -18 [-62, +17] | -24 [-85, +18] | -21 [-70, +18] | -25 [-87, +18] | -17 [-43, +10] |

How the crashed flights ended:

| batch | decision rule | conditions | crashed flights | out of bounds | hit the ground |
|---|---|---|---:|---:|---:|
| random_start_small_1M | with planning (as trained) | no disturbance | 1 | 100% | 0% |
| random_start_small_1M | with planning (as trained) | wind and gusts | 635 | 71% | 29% |
| random_start_small_1M | with planning (as trained) | mass, motors, delay | 40 | 65% | 35% |
| random_start_small_1M | policy network only | wind and gusts | 277 | 80% | 20% |
| random_start_small_1M | policy network only | mass, motors, delay | 2 | 0% | 100% |
| randomized_small_1M | with planning (as trained) | wind and gusts | 133 | 80% | 20% |
| randomized_small_1M | policy network only | wind and gusts | 176 | 82% | 18% |

Plots: learning_curves_by_group.png, final_by_group.png, comparison.png, disturbances.png

## outputs/batch2/randomized_small_1M/analysis/summary.md

Last modified 2026-11-02.

### randomized_small_1M: TD-MPC2 hover, Batch 2, 3 seeds

Seeds: 0, 1, 2. Steps: 500000. Settings: --config-name=config_batch2_randomized_small. Code: 785bfb21d634866d374af7791bc3af308c8f1999.

Success = full 10 s flight with late error below 10 cm. Values: mean ± standard error across seeds (each seed first averaged over its flights). Cells: success / crashes.

### Final agents

| batch | decision rule | training area (seen kind of start) | held-out block (never trained on) | outside the training area | the six Batch 1 starts |
|---|---|---:|---:|---:|---:|
| randomized_small_1M | with planning (as trained) | 13 ± 7% / 0 ± 0% | 7 ± 4% / 0 ± 0% | 4 ± 4% / 0 ± 0% | 6 ± 6% / 0 ± 0% |
| randomized_small_1M | policy network only | 34 ± 31% / 0 ± 0% | 34 ± 33% / 0 ± 0% | 27 ± 23% / 1 ± 1% | 33 ± 33% / 0 ± 0% |
| randomized_small_1M | small planner 64x2 | 11 ± 6% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 8 ± 8% / 1 ± 1% | 7 ± 7% / 0 ± 0% |
| random_start_small_1M | with planning (as trained) | 78 ± 10% / 0 ± 0% | 83 ± 10% / 0 ± 0% | 59 ± 9% / 3 ± 2% | 80 ± 10% / 0 ± 0% |
| random_start_small_1M | policy network only | 9 ± 9% / 0 ± 0% | 9 ± 9% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 11 ± 9% / 0 ± 0% |
| random_start_small_1M | small planner 64x2 | 76 ± 11% / 0 ± 0% | 77 ± 10% / 0 ± 0% | 60 ± 10% / 2 ± 2% | 79 ± 10% / 0 ± 0% |

### randomized_small_1M, each seed (with planning)

| seed | train_area success | held_out success | outside success | batch1 success | train_area median late error, full flights (m) |
|---:|---:|---:|---:|---:|---:|
| 0 | 22% | 7% | 0% | 0% | 0.145 |
| 1 (median) | 16% | 13% | 13% | 17% | 0.125 |
| 2 | 0% | 0% | 0% | 0% | 0.201 |

### Statistical comparisons (95% bootstrap intervals over seeds)

Differences in success share (percentage points). Intervals from fewer than about 5 seeds are unreliable.

| comparison | train_area | held_out | outside | batch1 |
|---|---:|---:|---:|---:|
| randomized_small_1M - random_start_small_1M (with planning) | -65 [-86, -42] | -76 [-93, -54] | -55 [-73, -35] | -74 [-93, -52] |
| randomized_small_1M: planner - policy network only (paired) | -21 [-80, +16] | -28 [-87, +3] | -22 [-60, +0] | -28 [-83, +0] |
| randomized_small_1M: planner - small planner 64x2 (paired) | +2 [-4, +10] | -6 [-13, +0] | -3 [-10, +0] | -2 [-6, +0] |
| random_start_small_1M: planner - policy network only (paired) | +69 [+30, +96] | +74 [+36, +98] | +48 [+18, +72] | +69 [+32, +95] |
| random_start_small_1M: planner - small planner 64x2 (paired) | +2 [-2, +7] | +6 [-1, +15] | -1 [-4, +2] | +1 [-4, +4] |

### Disturbances (success share on the 50 training-area starts, mean across seeds)

| batch | decision rule | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| randomized_small_1M | with planning (as trained) | 10% | 14% | 0% | 0% | 0% | 6% | 0% | 0% | 8% | 10% | 10% | 8% | 9% | 13% | 11% |
| randomized_small_1M | policy network only | 35% | 35% | 0% | 0% | 0% | 34% | 0% | 0% | 33% | 31% | 28% | 33% | 31% | 35% | 26% |
| random_start_small_1M | with planning (as trained) | 77% | 60% | 1% | 0% | 0% | 66% | 15% | 0% | 76% | 74% | 72% | 76% | 73% | 77% | 76% |
| random_start_small_1M | policy network only | 9% | 18% | 0% | 0% | 0% | 9% | 0% | 0% | 9% | 10% | 10% | 9% | 10% | 9% | 9% |

Planning minus policy network only, success points, paired by seed (95% bootstrap interval):

| batch | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| randomized_small_1M (3 seeds) | -25 [-82, +8] | -21 [-80, +16] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -28 [-62, +0] | +0 [+0, +0] | +0 [+0, +0] | -25 [-82, +6] | -21 [-64, +0] | -18 [-52, +2] | -25 [-80, +4] | -22 [-64, +0] | -22 [-74, +8] | -15 [-34, +0] |
| random_start_small_1M (11 seeds) | +67 [+29, +95] | +42 [+14, +69] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +57 [+16, +91] | +15 [+0, +34] | +0 [+0, +0] | +67 [+28, +95] | +64 [+25, +93] | +61 [+23, +91] | +66 [+28, +94] | +63 [+23, +93] | +68 [+30, +95] | +67 [+30, +95] |

randomized_small_1M minus each other batch, success points (95% bootstrap interval over seeds):

| comparison | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| randomized_small_1M - random_start_small_1M, with planning (as trained) | -67 [-87, -44] | -46 [-70, -21] | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | -60 [-85, -31] | -15 [-33, -0] | +0 [+0, +0] | -68 [-88, -45] | -64 [-86, -39] | -62 [-84, -36] | -68 [-88, -45] | -65 [-87, -40] | -65 [-85, -41] | -66 [-86, -42] |
| randomized_small_1M - random_start_small_1M, policy network only | +25 [-18, +89] | +17 [-32, +81] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +25 [-11, +71] | +0 [+0, +0] | +0 [+0, +0] | +24 [-18, +85] | +22 [-16, +71] | +18 [-17, +60] | +24 [-18, +85] | +21 [-17, +69] | +25 [-18, +87] | +17 [-10, +43] |

How the crashed flights ended:

| batch | decision rule | conditions | crashed flights | out of bounds | hit the ground |
|---|---|---|---:|---:|---:|
| randomized_small_1M | with planning (as trained) | wind and gusts | 133 | 80% | 20% |
| randomized_small_1M | policy network only | wind and gusts | 176 | 82% | 18% |
| random_start_small_1M | with planning (as trained) | no disturbance | 1 | 100% | 0% |
| random_start_small_1M | with planning (as trained) | wind and gusts | 635 | 71% | 29% |
| random_start_small_1M | with planning (as trained) | mass, motors, delay | 40 | 65% | 35% |
| random_start_small_1M | policy network only | wind and gusts | 277 | 80% | 20% |
| random_start_small_1M | policy network only | mass, motors, delay | 2 | 0% | 100% |

Plots: learning_curves_by_group.png, final_by_group.png, comparison.png, disturbances.png

## outputs/batch2/sharp_small_1M/analysis/summary.md

Last modified 2026-11-02.

### sharp_small_1M: TD-MPC2 hover, Batch 2, 3 seeds

Seeds: 0, 1, 2. Steps: 500000. Settings: --config-name=config_batch2_sharp_small. Code: 3f9f21bbb0db622f37a9b58e2d6d174874c22721.

Success = full 10 s flight with late error below 10 cm. Values: mean ± standard error across seeds (each seed first averaged over its flights). Cells: success / crashes.

### Final agents

| batch | decision rule | training area (seen kind of start) | held-out block (never trained on) | outside the training area | the six Batch 1 starts |
|---|---|---:|---:|---:|---:|
| sharp_small_1M | with planning (as trained) | 37 ± 19% / 0 ± 0% | 66 ± 33% / 0 ± 0% | 23 ± 17% / 2 ± 2% | 31 ± 18% / 0 ± 0% |
| sharp_small_1M | policy network only | 11 ± 11% / 0 ± 0% | 12 ± 12% / 0 ± 0% | 6 ± 4% / 3 ± 3% | 11 ± 11% / 0 ± 0% |
| sharp_small_1M | small planner 64x2 | 36 ± 20% / 0 ± 0% | 64 ± 32% / 0 ± 0% | 19 ± 10% / 2 ± 2% | 24 ± 16% / 0 ± 0% |
| random_start_small_1M | with planning (as trained) | 78 ± 10% / 0 ± 0% | 83 ± 10% / 0 ± 0% | 59 ± 9% / 3 ± 2% | 80 ± 10% / 0 ± 0% |
| random_start_small_1M | policy network only | 9 ± 9% / 0 ± 0% | 9 ± 9% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 11 ± 9% / 0 ± 0% |
| random_start_small_1M | small planner 64x2 | 76 ± 11% / 0 ± 0% | 77 ± 10% / 0 ± 0% | 60 ± 10% / 2 ± 2% | 79 ± 10% / 0 ± 0% |

### sharp_small_1M, each seed (with planning)

| seed | train_area success | held_out success | outside success | batch1 success | train_area median late error, full flights (m) |
|---:|---:|---:|---:|---:|---:|
| 0 (median) | 48% | 100% | 57% | 33% | 0.101 |
| 1 | 62% | 97% | 13% | 61% | 0.083 |
| 2 | 0% | 0% | 0% | 0% | 0.161 |

### Statistical comparisons (95% bootstrap intervals over seeds)

Differences in success share (percentage points). Intervals from fewer than about 5 seeds are unreliable.

| comparison | train_area | held_out | outside | batch1 |
|---|---:|---:|---:|---:|
| sharp_small_1M - random_start_small_1M (with planning) | -41 [-77, -7] | -17 [-76, +29] | -36 [-66, -2] | -48 [-81, -14] |
| sharp_small_1M: planner - policy network only (paired) | +25 [+0, +48] | +53 [+0, +100] | +18 [-3, +57] | +20 [+0, +33] |
| sharp_small_1M: planner - small planner 64x2 (paired) | +1 [-6, +8] | +1 [+0, +3] | +4 [-3, +20] | +7 [+0, +17] |
| random_start_small_1M: planner - policy network only (paired) | +69 [+30, +96] | +74 [+36, +98] | +48 [+18, +72] | +69 [+32, +95] |
| random_start_small_1M: planner - small planner 64x2 (paired) | +2 [-2, +7] | +6 [-1, +15] | -1 [-4, +2] | +1 [-4, +4] |

### Disturbances (success share on the 50 training-area starts, mean across seeds)

| batch | decision rule | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_small_1M | with planning (as trained) | 77% | 60% | 1% | 0% | 0% | 66% | 15% | 0% | 76% | 74% | 72% | 76% | 73% | 77% | 76% |
| random_start_small_1M | policy network only | 9% | 18% | 0% | 0% | 0% | 9% | 0% | 0% | 9% | 10% | 10% | 9% | 10% | 9% | 9% |

Planning minus policy network only, success points, paired by seed (95% bootstrap interval):

| batch | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_small_1M (11 seeds) | +67 [+29, +95] | +42 [+14, +69] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +57 [+16, +91] | +15 [+0, +34] | +0 [+0, +0] | +67 [+28, +95] | +64 [+25, +93] | +61 [+23, +91] | +66 [+28, +94] | +63 [+23, +93] | +68 [+30, +95] | +67 [+30, +95] |

How the crashed flights ended:

| batch | decision rule | conditions | crashed flights | out of bounds | hit the ground |
|---|---|---|---:|---:|---:|
| random_start_small_1M | with planning (as trained) | no disturbance | 1 | 100% | 0% |
| random_start_small_1M | with planning (as trained) | wind and gusts | 635 | 71% | 29% |
| random_start_small_1M | with planning (as trained) | mass, motors, delay | 40 | 65% | 35% |
| random_start_small_1M | policy network only | wind and gusts | 277 | 80% | 20% |
| random_start_small_1M | policy network only | mass, motors, delay | 2 | 0% | 100% |

Plots: learning_curves_by_group.png, final_by_group.png, comparison.png, disturbances.png

## outputs/batch2/random_start_2M/analysis/summary.md

Last modified 2026-11-02.

### random_start_2M: TD-MPC2 hover, Batch 2, 3 seeds

Seeds: 0, 1, 2. Steps: 1350000. Settings: --config-name=config_batch2_random_start. Code: 622b694bd05f966aef584718ce04edcd16c5a4e4.

Success = full 10 s flight with late error below 10 cm. Values: mean ± standard error across seeds (each seed first averaged over its flights). Cells: success / crashes.

### Final agents

| batch | decision rule | training area (seen kind of start) | held-out block (never trained on) | outside the training area | the six Batch 1 starts |
|---|---|---:|---:|---:|---:|
| random_start_2M | with planning (as trained) | 72 ± 12% / 3 ± 2% | 74 ± 12% / 7 ± 5% | 24 ± 6% / 58 ± 11% | 78 ± 8% / 4 ± 4% |
| random_start_2M | policy network only | 29 ± 28% / 0 ± 0% | 30 ± 30% / 2 ± 2% | 18 ± 10% / 33 ± 12% | 33 ± 33% / 0 ± 0% |
| random_start_2M | small planner 64x2 | 70 ± 21% / 0 ± 0% | 70 ± 28% / 0 ± 0% | 57 ± 12% / 22 ± 1% | 69 ± 29% / 0 ± 0% |
| random_start_1M | with planning (as trained) | 46 ± 8% / 0 ± 0% | 39 ± 9% / 2 ± 1% | 20 ± 6% / 26 ± 5% | 39 ± 10% / 0 ± 0% |
| random_start_1M | policy network only | 60 ± 11% / 0 ± 0% | 64 ± 13% / 0 ± 0% | 40 ± 10% / 6 ± 2% | 65 ± 11% / 0 ± 0% |
| random_start_1M | small planner 64x2 | 58 ± 9% / 0 ± 0% | 46 ± 12% / 0 ± 0% | 32 ± 7% / 9 ± 2% | 57 ± 8% / 0 ± 0% |
| random_start_small_1M | with planning (as trained) | 78 ± 10% / 0 ± 0% | 83 ± 10% / 0 ± 0% | 59 ± 9% / 3 ± 2% | 80 ± 10% / 0 ± 0% |
| random_start_small_1M | policy network only | 9 ± 9% / 0 ± 0% | 9 ± 9% / 0 ± 0% | 12 ± 8% / 0 ± 0% | 11 ± 9% / 0 ± 0% |
| random_start_small_1M | small planner 64x2 | 76 ± 11% / 0 ± 0% | 77 ± 10% / 0 ± 0% | 60 ± 10% / 2 ± 2% | 79 ± 10% / 0 ± 0% |

### random_start_2M, each seed (with planning)

| seed | train_area success | held_out success | outside success | batch1 success | train_area median late error, full flights (m) |
|---:|---:|---:|---:|---:|---:|
| 0 (median) | 78% | 97% | 30% | 83% | 0.072 |
| 1 | 48% | 57% | 30% | 61% | 0.103 |
| 2 | 90% | 70% | 13% | 89% | 0.031 |

### Statistical comparisons (95% bootstrap intervals over seeds)

Differences in success share (percentage points). Intervals from fewer than about 5 seeds are unreliable.

| comparison | train_area | held_out | outside | batch1 |
|---|---:|---:|---:|---:|
| random_start_2M - random_start_1M (with planning) | +26 [+0, +51] | +36 [+11, +61] | +5 [-11, +18] | +38 [+15, +61] |
| random_start_2M - random_start_small_1M (with planning) | -6 [-34, +23] | -8 [-33, +20] | -35 [-54, -15] | -2 [-24, +22] |
| random_start_2M: planner - policy network only (paired) | +43 [-8, +90] | +44 [+7, +70] | +7 [-3, +13] | +44 [-17, +89] |
| random_start_2M: planner - small planner 64x2 (paired) | +2 [-10, +18] | +4 [-27, +43] | -32 [-57, -3] | +9 [-11, +50] |
| random_start_1M: planner - policy network only (paired) | -15 [-37, +10] | -25 [-47, -1] | -20 [-40, +0] | -26 [-50, +2] |
| random_start_1M: planner - small planner 64x2 (paired) | -12 [-24, +3] | -8 [-24, +10] | -12 [-18, -6] | -18 [-29, -7] |
| random_start_small_1M: planner - policy network only (paired) | +69 [+30, +96] | +74 [+36, +98] | +48 [+18, +72] | +69 [+32, +95] |
| random_start_small_1M: planner - small planner 64x2 (paired) | +2 [-2, +7] | +6 [-1, +15] | -1 [-4, +2] | +1 [-4, +4] |

### Disturbances (success share on the 50 training-area starts, mean across seeds)

| batch | decision rule | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_1M | with planning (as trained) | 43% | 36% | 1% | 0% | 0% | 37% | 7% | 0% | 32% | 20% | 15% | 33% | 25% | 33% | 25% |
| random_start_1M | policy network only | 60% | 63% | 1% | 0% | 0% | 57% | 23% | 2% | 57% | 54% | 51% | 57% | 54% | 60% | 60% |
| random_start_small_1M | with planning (as trained) | 77% | 60% | 1% | 0% | 0% | 66% | 15% | 0% | 76% | 74% | 72% | 76% | 73% | 77% | 76% |
| random_start_small_1M | policy network only | 9% | 18% | 0% | 0% | 0% | 9% | 0% | 0% | 9% | 10% | 10% | 9% | 10% | 9% | 9% |

Planning minus policy network only, success points, paired by seed (95% bootstrap interval):

| batch | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_1M (11 seeds) | -17 [-38, +6] | -27 [-43, -9] | +0 [+0, +1] | +0 [+0, +0] | +0 [+0, +0] | -20 [-40, +3] | -17 [-36, +0] | -2 [-6, +0] | -25 [-47, -1] | -34 [-53, -14] | -37 [-57, -16] | -24 [-46, +0] | -29 [-49, -7] | -27 [-50, -3] | -35 [-55, -13] |
| random_start_small_1M (11 seeds) | +67 [+29, +95] | +42 [+14, +69] | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | +57 [+16, +91] | +15 [+0, +34] | +0 [+0, +0] | +67 [+28, +95] | +64 [+25, +93] | +61 [+23, +91] | +66 [+28, +94] | +63 [+23, +93] | +68 [+30, +95] | +67 [+30, +95] |

random_start_1M minus each other batch, success points (95% bootstrap interval over seeds):

| comparison | none | wind 1 | wind 2 | wind 3 | wind 4 | gust 2 | gust 4 | gust 6 | mass 1.05 | mass 1.1 | mass 1.15 | motor 0.95 | motor 0.9 | delay 1 | delay 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_1M - random_start_small_1M, with planning (as trained) | -34 [-59, -7] | -24 [-47, +2] | +0 [-2, +3] | +0 [+0, +0] | +0 [+0, +0] | -29 [-57, +1] | -8 [-27, +7] | +0 [+0, +1] | -44 [-67, -18] | -54 [-76, -29] | -57 [-80, -31] | -43 [-67, -17] | -49 [-73, -22] | -45 [-68, -19] | -51 [-73, -26] |
| random_start_1M - random_start_small_1M, policy network only | +51 [+22, +75] | +45 [+14, +73] | +1 [+0, +2] | +0 [+0, +0] | +0 [+0, +0] | +48 [+18, +75] | +23 [+5, +44] | +2 [+0, +7] | +48 [+19, +73] | +44 [+15, +70] | +41 [+13, +66] | +47 [+19, +73] | +44 [+15, +70] | +51 [+22, +76] | +51 [+22, +75] |

How the crashed flights ended:

| batch | decision rule | conditions | crashed flights | out of bounds | hit the ground |
|---|---|---|---:|---:|---:|
| random_start_1M | with planning (as trained) | no disturbance | 2 | 50% | 50% |
| random_start_1M | with planning (as trained) | wind and gusts | 1171 | 57% | 43% |
| random_start_1M | with planning (as trained) | mass, motors, delay | 43 | 16% | 84% |
| random_start_1M | policy network only | wind and gusts | 570 | 40% | 60% |
| random_start_1M | policy network only | mass, motors, delay | 1 | 0% | 100% |
| random_start_small_1M | with planning (as trained) | no disturbance | 1 | 100% | 0% |
| random_start_small_1M | with planning (as trained) | wind and gusts | 635 | 71% | 29% |
| random_start_small_1M | with planning (as trained) | mass, motors, delay | 40 | 65% | 35% |
| random_start_small_1M | policy network only | wind and gusts | 277 | 80% | 20% |
| random_start_small_1M | policy network only | mass, motors, delay | 2 | 0% | 100% |

Plots: learning_curves_by_group.png, final_by_group.png, comparison.png, disturbances.png

## outputs/batch2/snapshot_summary/snapshot_evaluation.md

Last modified 2026-11-02.

### Snapshots: best (chosen on validation) vs final (snapshot_evaluation)

Success = full 10 s flight with late error below 10 cm.

### random_start_1M (11 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 700k | 34% / 60% / 27% | 34% / 57% / 17% | 0.158 | 0.064 | 17 12 12 15 10 12 | 4% |
| seed_01 | 1000k | 72% / 80% / 63% | 72% / 80% / 63% | 0.214 | 0.071 | 20 10 10 7 10 9 | 2% |
| seed_02 | 1000k | 74% / 67% / 33% | 74% / 67% / 33% | 0.311 | 0.065 | 20 12 13 9 14 9 | 8% |
| seed_03 | 900k | 30% / 20% / 3% | 10% / 3% / 7% | 0.114 | 0.039 | 23 24 16 19 13 14 | 4% |
| seed_04 | 900k | 52% / 30% / 23% | 12% / 0% / 10% | 0.148 | 0.066 | 14 11 13 11 10 15 | 2% |
| seed_05 | 600k | 30% / 13% / 10% | 10% / 3% / 7% | 0.113 | 0.057 | 17 17 17 16 11 18 | 2% |
| seed_06 | 1000k | 64% / 40% / 37% | 64% / 40% / 37% | 0.211 | 0.063 | 14 23 18 12 15 6 | 6% |
| seed_07 | 1000k | 70% / 63% / 10% | 70% / 63% / 10% | 0.083 | 0.068 | 10 9 6 8 8 8 | 0% |
| seed_08 | 1000k | 24% / 20% / 17% | 24% / 20% / 17% | 0.081 | 0.045 | 19 16 17 20 14 13 | 2% |
| seed_09 | 1000k | 70% / 57% / 50% | 70% / 57% / 50% | 0.234 | 0.060 | 18 19 20 17 14 8 | 4% |
| seed_10 | 800k | 16% / 57% / 17% | 18% / 43% / 10% | 0.059 | 0.049 | 23 16 19 15 20 16 | 4% |

Best minus final test success, mean ± SE across seeds: train_area +7% ± 4%, held_out +7% ± 3%, outside +3% ± 2%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.16 ± 0.02, late error SD (cm) 3.36 ± 0.41, late error mean (cm) 14.30 ± 0.95

### fixed_start_1M (3 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 500k | 26% / 10% / 0% | 24% / 0% / 0% | 0.069 | 0.061 | 9 17 5 7 8 8 | 68% |
| seed_01 | 800k | 26% / 0% / 0% | 24% / 0% / 3% | 0.091 | 0.054 | 10 21 19 10 13 17 | 90% |
| seed_02 | 900k | 32% / 40% / 0% | 40% / 20% / 0% | 0.112 | 0.065 | 19 16 8 5 12 7 | 50% |

Best minus final test success, mean ± SE across seeds: train_area -1% ± 3%, held_out +10% ± 6%, outside -1% ± 1%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.09 ± 0.01, late error SD (cm) 4.78 ± 0.42, late error mean (cm) 11.78 ± 1.70

### random_start_small_1M (11 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 800k | 92% / 100% / 53% | 88% / 97% / 77% | 0.313 | 0.068 | 11 13 4 7 10 7 | 0% |
| seed_01 | 1000k | 100% / 100% / 80% | 100% / 100% / 80% | 0.226 | 0.050 | 7 3 7 7 10 5 | 0% |
| seed_02 | 1000k | 94% / 93% / 53% | 94% / 93% / 53% | 0.447 | 0.071 | 55 8 19 12 9 5 | 0% |
| seed_03 | 500k | 76% / 13% / 40% | 44% / 43% / 33% | 0.337 | 0.070 | 9 11 7 24 59 10 | 2% |
| seed_04 | 1000k | 100% / 100% / 97% | 100% / 100% / 97% | 0.234 | 0.063 | 6 11 9 8 6 5 | 0% |
| seed_05 | 1000k | 100% / 100% / 97% | 100% / 100% / 97% | 0.100 | 0.037 | 7 4 9 7 7 5 | 0% |
| seed_06 | 700k | 100% / 100% / 47% | 94% / 100% / 63% | 0.407 | 0.063 | 89 5 5 7 12 8 | 44% |
| seed_07 | 1000k | 100% / 100% / 77% | 100% / 100% / 77% | 0.478 | 0.065 | 33 59 28 8 14 7 | 0% |
| seed_08 | 700k | 100% / 100% / 60% | 14% / 70% / 7% | 0.398 | 0.069 | 7 19 7 12 15 12 | 0% |
| seed_09 | 1000k | 100% / 100% / 83% | 100% / 100% / 83% | 0.373 | 0.068 | 13 7 9 29 5 4 | 0% |
| seed_10 | 400k | 100% / 100% / 80% | 2% / 0% / 20% | 0.410 | 0.070 | 21 10 7 6 12 13 | 0% |

Best minus final test success, mean ± SE across seeds: train_area +21% ± 11%, held_out +9% ± 10%, outside +7% ± 8%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.34 ± 0.03, late error SD (cm) 11.02 ± 3.16, late error mean (cm) 13.47 ± 1.93

### sharp_small_1M (3 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 700k | 88% / 100% / 50% | 50% / 100% / 50% | 0.295 | 0.071 | 9 12 6 14 9 10 | 0% |
| seed_01 | 600k | 100% / 93% / 40% | 62% / 97% / 10% | 0.183 | 0.055 | 7 5 8 7 9 8 | 0% |
| seed_02 | 400k | 84% / 40% / 50% | 0% / 0% / 10% | 0.280 | 0.067 | 7 10 15 11 11 16 | 0% |

Best minus final test success, mean ± SE across seeds: train_area +53% ± 15%, held_out +12% ± 14%, outside +23% ± 12%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.25 ± 0.04, late error SD (cm) 2.56 ± 0.63, late error mean (cm) 9.70 ± 1.23


## outputs/batch2/snapshot_summary_policy_only/snapshot_evaluation_policy_only.md

Last modified 2026-11-02.

### Snapshots: best (chosen on validation) vs final (snapshot_evaluation_policy_only)

Success = full 10 s flight with late error below 10 cm.

### random_start_1M (11 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 500k | 94% / 100% / 57% | 22% / 3% / 43% | 0.404 | 0.069 | 8 8 11 15 13 13 | 0% |
| seed_01 | 900k | 100% / 100% / 83% | 74% / 100% / 7% | 0.454 | 0.070 | 6 11 15 8 8 6 | 0% |
| seed_02 | 600k | 34% / 3% / 0% | 8% / 30% / 3% | 0.133 | 0.047 | 12 11 18 13 18 14 | 0% |
| seed_03 | 800k | 84% / 23% / 43% | 0% / 0% / 0% | 0.334 | 0.053 | 15 14 14 9 23 19 | 0% |
| seed_04 | 800k | 98% / 100% / 37% | 54% / 23% / 0% | 0.185 | 0.039 | 7 6 4 5 3 10 | 0% |
| seed_05 | 1000k | 94% / 100% / 87% | 94% / 100% / 87% | 0.259 | 0.069 | 8 9 14 10 16 6 | 0% |
| seed_06 | 500k | 100% / 70% / 57% | 92% / 93% / 47% | 0.339 | 0.058 | 7 10 16 5 8 5 | 0% |
| seed_07 | 1000k | 100% / 100% / 60% | 100% / 100% / 60% | 0.000 | 0.000 | 8 6 7 7 7 4 | 0% |
| seed_08 | 200k | 100% / 100% / 53% | 38% / 40% / 33% | 0.120 | 0.067 | 15 12 9 13 13 12 | 0% |
| seed_09 | 1000k | 100% / 100% / 93% | 100% / 100% / 93% | 0.294 | 0.059 | 10 9 11 9 5 9 | 0% |
| seed_10 | 800k | 86% / 100% / 77% | 78% / 97% / 57% | 0.147 | 0.058 | 7 8 8 5 9 5 | 0% |

Best minus final test success, mean ± SE across seeds: train_area +30% ± 9%, held_out +19% ± 12%, outside +20% ± 7%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.24 ± 0.04, late error SD (cm) 2.87 ± 0.32, late error mean (cm) 9.97 ± 0.96

### fixed_start_1M (3 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 800k | 44% / 10% / 0% | 12% / 0% / 0% | 0.106 | 0.064 | 10 12 12 8 13 11 | 30% |
| seed_01 | 1000k | 86% / 37% / 0% | 86% / 37% / 0% | 0.286 | 0.071 | 12 9 10 7 9 5 | 68% |
| seed_02 | 900k | 46% / 57% / 10% | 38% / 17% / 0% | 0.177 | 0.063 | 15 10 18 13 10 11 | 28% |

Best minus final test success, mean ± SE across seeds: train_area +13% ± 10%, held_out +17% ± 12%, outside +3% ± 3%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.19 ± 0.05, late error SD (cm) 2.39 ± 0.38, late error mean (cm) 10.79 ± 1.21

### random_start_small_1M (11 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 800k | 2% / 13% / 20% | 0% / 0% / 0% | 0.008 | 0.008 | 26 19 16 14 36 19 | 0% |
| seed_01 | 400k | 100% / 100% / 60% | 0% / 0% / 0% | 0.000 | 0.000 | 80 11 19 15 14 41 | 0% |
| seed_02 | 400k | 6% / 0% / 0% | 2% / 0% / 17% | 0.008 | 0.008 | 41 17 24 22 43 14 | 0% |
| seed_03 | 1000k | 0% / 0% / 0% | 0% / 0% / 0% | 0.000 | 0.000 | 18 30 33 30 36 19 | 0% |
| seed_04 | 700k | 6% / 0% / 0% | 0% / 0% / 0% | 0.024 | 0.014 | 15 25 16 74 48 19 | 2% |
| seed_05 | 600k | 100% / 100% / 70% | 0% / 0% / 13% | 0.401 | 0.055 | 13 7 25 16 23 14 | 0% |
| seed_06 | 300k | 20% / 17% / 0% | 0% / 0% / 0% | 0.000 | 0.000 | 119 16 23 26 50 19 | 0% |
| seed_07 | 300k | 2% / 0% / 33% | 0% / 0% / 0% | 0.000 | 0.000 | 18 51 32 24 14 34 | 0% |
| seed_08 | 400k | 86% / 100% / 47% | 0% / 0% / 0% | 0.254 | 0.050 | 9 17 12 22 16 21 | 0% |
| seed_09 | 500k | 100% / 100% / 87% | 0% / 0% / 0% | 0.408 | 0.053 | 6 17 45 17 15 15 | 0% |
| seed_10 | 1000k | 100% / 100% / 90% | 100% / 100% / 90% | 0.463 | 0.069 | 16 10 13 8 12 4 | 0% |

Best minus final test success, mean ± SE across seeds: train_area +38% ± 14%, held_out +39% ± 15%, outside +26% ± 10%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.14 ± 0.06, late error SD (cm) 14.53 ± 3.34, late error mean (cm) 24.69 ± 2.73

### sharp_small_1M (3 seeds)

| seed | best step (validation) | test success at best: train_area / held_out / outside | at final | within-run SD of train_area success (>= 500k) | binomial SD expected | train_area median late error of full flights, >= 500k (cm) | highest crash share, >= 500k |
|---|---:|---|---|---:|---:|---|---:|
| seed_00 | 900k | 16% / 0% / 10% | 0% / 0% / 0% | 0.078 | 0.031 | 14 21 18 17 16 28 | 0% |
| seed_01 | 1000k | 34% / 37% / 13% | 34% / 37% / 13% | 0.139 | 0.033 | 29 23 17 19 17 11 | 0% |
| seed_02 | 300k | 10% / 0% / 0% | 0% / 0% / 3% | 0.017 | 0.014 | 21 19 14 30 19 27 | 0% |

Best minus final test success, mean ± SE across seeds: train_area +9% ± 5%, held_out +0% ± 0%, outside +2% ± 4%

Drift from 500k on (train_area, per seed, then mean ± SE across seeds): success SD 0.08 ± 0.04, late error SD (cm) 5.68 ± 0.33, late error mean (cm) 20.05 ± 0.92


## outputs/batch2/variance_decomposition/summary.md

Last modified 2026-11-01.

### Evaluation noise vs real differences between seeds (Batch 1 agents)

Every final agent flew 20 flights from the Batch 1 training start: 5 sensor-noise seeds x 4 flights, the same noise seeds for every agent.

| batch | seeds | late error: between-seed SD (m) | within-seed SD (m) | ICC | SE of one seed's mean with Batch 1's 3 flights (m) | crash rate: range over seeds |
|---|---:|---:|---:|---:|---:|---|
| batch1_small_narval | 11 | 0.319 | 0.087 | 0.93 | 0.050 | 0% to 0% |
| batch1_fixed_start | 11 | 0.627 | 0.161 | 0.94 | 0.093 | 0% to 100% |
| batch1_standard | 11 | 0.376 | 0.362 | 0.52 | 0.209 | 10% to 100% |

ICC near 1: the spread comes from the agents themselves (training), not from evaluation noise. ICC near 0: re-testing the same agent varies as much as changing seed.

## outputs/batch2/fluctuation_analysis/summary.md

Last modified 2026-11-01.

### Within-run fluctuations vs training statistics

Runs: outputs/batch1_standard_1M. Test point = mean late error of that point's test flights. Statistic = mean of the training log over the 5k steps before the test point. Only points after the run first averaged under 10 cm are used.

| seed | statistic | level: rho (n) | change: rho (n) | change: permutation p |
|---:|---|---:|---:|---:|
| seed_00 | consistency_loss | -0.24 (166) | +0.04 (165) | 0.576 |
| seed_00 | reward_loss | -0.21 (166) | +0.07 (165) | 0.341 |
| seed_00 | value_loss | -0.30 (166) | +0.04 (165) | 0.581 |
| seed_00 | policy_loss | +0.29 (166) | -0.03 (165) | 0.711 |
| seed_00 | policy_log_std | +0.09 (166) | +0.01 (165) | 0.949 |
| seed_00 | value_scale | +0.31 (166) | +0.10 (165) | 0.204 |
| seed_01 | consistency_loss | -0.30 (175) | -0.11 (174) | 0.139 |
| seed_01 | reward_loss | -0.18 (175) | +0.02 (174) | 0.803 |
| seed_01 | value_loss | -0.34 (175) | +0.06 (174) | 0.400 |
| seed_01 | policy_loss | +0.34 (175) | +0.02 (174) | 0.831 |
| seed_01 | policy_log_std | +0.32 (175) | -0.14 (174) | 0.075 |
| seed_01 | value_scale | +0.34 (175) | +0.10 (174) | 0.180 |
| seed_02 | consistency_loss | -0.05 (180) | -0.05 (179) | 0.531 |
| seed_02 | reward_loss | +0.06 (180) | +0.04 (179) | 0.620 |
| seed_02 | value_loss | -0.04 (180) | -0.12 (179) | 0.126 |
| seed_02 | policy_loss | +0.02 (180) | -0.03 (179) | 0.697 |
| seed_02 | policy_log_std | +0.04 (180) | +0.05 (179) | 0.554 |
| seed_02 | value_scale | +0.02 (180) | -0.09 (179) | 0.247 |

Permutation p-values ignore autocorrelation in the series and are not corrected for testing several statistics; treat them as screening only.

## outputs/batch2/model_error_summary/summary.md

Last modified 2026-11-02.

### World-model prediction error along test flights

Latent error: mean squared error of the 3-step latent prediction from the actions actually taken (TD-MPC2's consistency loss), averaged over each flight's steps, then per seed, then mean ± SE across seeds. Crashes: total over all seeds.

| batch | seeds | decision | starts | crashes | latent error k=3 | reward error | planner / policy (paired, per seed) |
|---|---:|---|---|---:|---:|---:|---:|
| random_start_1M | 11 | planner | train_area | 2/550 | 0.00024 ± 0.00001 | 0.0005 ± 0.0000 | 1.17 ± 0.07 |
| random_start_1M | 11 | planner | held_out | 5/330 | 0.00023 ± 0.00002 | 0.0006 ± 0.0000 | 1.49 ± 0.08 |
| random_start_1M | 11 | planner | outside | 84/330 | 0.00060 ± 0.00005 | 0.0027 ± 0.0003 | 2.00 ± 0.14 |
| random_start_1M | 11 | policy_only | train_area | 0/550 | 0.00021 ± 0.00002 | 0.0004 ± 0.0000 |  |
| random_start_1M | 11 | policy_only | held_out | 0/330 | 0.00015 ± 0.00001 | 0.0004 ± 0.0000 |  |
| random_start_1M | 11 | policy_only | outside | 16/330 | 0.00030 ± 0.00002 | 0.0011 ± 0.0002 |  |
| random_start_1M | 11 | planner | last 0.5 s before crashes vs completed flights | | 0.00115 vs 0.00026 | | 4.36x |
| fixed_start_1M | 3 | planner | train_area | 56/150 | 0.00065 ± 0.00015 | 0.0034 ± 0.0010 | 1.68 ± 0.35 |
| fixed_start_1M | 3 | planner | held_out | 72/90 | 0.00111 ± 0.00019 | 0.0093 ± 0.0014 | 1.34 ± 0.05 |
| fixed_start_1M | 3 | planner | outside | 86/90 | 0.00116 ± 0.00008 | 0.0125 ± 0.0004 | 0.88 ± 0.02 |
| fixed_start_1M | 3 | policy_only | train_area | 22/150 | 0.00038 ± 0.00001 | 0.0037 ± 0.0010 |  |
| fixed_start_1M | 3 | policy_only | held_out | 49/90 | 0.00082 ± 0.00011 | 0.0087 ± 0.0004 |  |
| fixed_start_1M | 3 | policy_only | outside | 88/90 | 0.00131 ± 0.00010 | 0.0345 ± 0.0088 |  |
| fixed_start_1M | 3 | planner | last 0.5 s before crashes vs completed flights | | 0.00113 vs 0.00040 | | 2.95x |
| random_start_small_1M | 11 | planner | train_area | 0/550 | 0.00001 ± 0.00000 | 0.0004 ± 0.0000 | 0.68 ± 0.09 |
| random_start_small_1M | 11 | planner | held_out | 0/330 | 0.00002 ± 0.00000 | 0.0004 ± 0.0000 | 0.67 ± 0.06 |
| random_start_small_1M | 11 | planner | outside | 13/330 | 0.00008 ± 0.00003 | 0.0010 ± 0.0002 | 1.32 ± 0.55 |
| random_start_small_1M | 11 | policy_only | train_area | 0/550 | 0.00002 ± 0.00000 | 0.0005 ± 0.0001 |  |
| random_start_small_1M | 11 | policy_only | held_out | 0/330 | 0.00003 ± 0.00001 | 0.0005 ± 0.0001 |  |
| random_start_small_1M | 11 | policy_only | outside | 0/330 | 0.00009 ± 0.00002 | 0.0008 ± 0.0001 |  |
| random_start_small_1M | 4 | planner | last 0.5 s before crashes vs completed flights | | 0.00101 vs 0.00003 | | 34.60x |
| batch1_standard_1M | 3 | planner | train_area | 99/150 | 0.00131 ± 0.00016 | 0.0075 ± 0.0010 | 2.84 ± 0.71 |
| batch1_standard_1M | 3 | planner | held_out | 83/90 | 0.00201 ± 0.00068 | 0.0132 ± 0.0035 | 2.39 ± 0.22 |
| batch1_standard_1M | 3 | planner | outside | 81/90 | 0.00090 ± 0.00015 | 0.0110 ± 0.0011 | 1.10 ± 0.22 |
| batch1_standard_1M | 3 | policy_only | train_area | 22/150 | 0.00049 ± 0.00006 | 0.0043 ± 0.0022 |  |
| batch1_standard_1M | 3 | policy_only | held_out | 48/90 | 0.00081 ± 0.00023 | 0.0149 ± 0.0097 |  |
| batch1_standard_1M | 3 | policy_only | outside | 70/90 | 0.00083 ± 0.00003 | 0.0176 ± 0.0044 |  |
| batch1_standard_1M | 3 | planner | last 0.5 s before crashes vs completed flights | | 0.00159 vs 0.00039 | | 4.03x |

## outputs/batch2/model_error_summary_batch1/summary.md

Last modified 2026-11-01.

### World-model prediction error along test flights

Latent error: mean squared error of the 3-step latent prediction from the actions actually taken (TD-MPC2's consistency loss), averaged over each flight's steps, then per seed, then mean ± SE across seeds. Crashes: total over all seeds.

| batch | seeds | decision | starts | crashes | latent error k=3 | reward error | planner / policy (paired, per seed) |
|---|---:|---|---|---:|---:|---:|---:|
| batch1_standard_1M | 3 | planner | train_area | 99/150 | 0.00131 ± 0.00016 | 0.0075 ± 0.0010 | 2.84 ± 0.71 |
| batch1_standard_1M | 3 | planner | held_out | 83/90 | 0.00201 ± 0.00068 | 0.0132 ± 0.0035 | 2.39 ± 0.22 |
| batch1_standard_1M | 3 | planner | outside | 81/90 | 0.00090 ± 0.00015 | 0.0110 ± 0.0011 | 1.10 ± 0.22 |
| batch1_standard_1M | 3 | policy_only | train_area | 22/150 | 0.00049 ± 0.00006 | 0.0043 ± 0.0022 |  |
| batch1_standard_1M | 3 | policy_only | held_out | 48/90 | 0.00081 ± 0.00023 | 0.0149 ± 0.0097 |  |
| batch1_standard_1M | 3 | policy_only | outside | 70/90 | 0.00083 ± 0.00003 | 0.0176 ± 0.0044 |  |
| batch1_standard_1M | 3 | planner | last 0.5 s before crashes vs completed flights | | 0.00159 vs 0.00039 | | 4.03x |
| batch1_standard | 11 | planner | train_area | 537/550 | 0.00145 ± 0.00007 | 0.0119 ± 0.0005 | 1.29 ± 0.08 |
| batch1_standard | 11 | planner | held_out | 323/330 | 0.00115 ± 0.00009 | 0.0105 ± 0.0013 | 0.96 ± 0.08 |
| batch1_standard | 11 | planner | outside | 329/330 | 0.00104 ± 0.00005 | 0.0146 ± 0.0007 | 1.02 ± 0.06 |
| batch1_standard | 11 | policy_only | train_area | 424/550 | 0.00115 ± 0.00006 | 0.0252 ± 0.0040 |  |
| batch1_standard | 11 | policy_only | held_out | 288/330 | 0.00123 ± 0.00008 | 0.0257 ± 0.0034 |  |
| batch1_standard | 11 | policy_only | outside | 270/330 | 0.00104 ± 0.00004 | 0.0258 ± 0.0046 |  |
| batch1_standard | 7 | planner | last 0.5 s before crashes vs completed flights | | 0.00124 vs 0.00114 | | 1.12x |
| batch1_small_narval | 11 | planner | train_area | 53/550 | 0.00031 ± 0.00004 | 0.0018 ± 0.0002 | 1.39 ± 0.29 |
| batch1_small_narval | 11 | planner | held_out | 49/330 | 0.00035 ± 0.00005 | 0.0020 ± 0.0004 | 1.93 ± 0.35 |
| batch1_small_narval | 11 | planner | outside | 109/330 | 0.00077 ± 0.00009 | 0.0042 ± 0.0005 | 1.30 ± 0.11 |
| batch1_small_narval | 11 | policy_only | train_area | 32/550 | 0.00028 ± 0.00005 | 0.0021 ± 0.0006 |  |
| batch1_small_narval | 11 | policy_only | held_out | 4/330 | 0.00020 ± 0.00002 | 0.0011 ± 0.0002 |  |
| batch1_small_narval | 11 | policy_only | outside | 56/330 | 0.00066 ± 0.00013 | 0.0041 ± 0.0010 |  |
| batch1_small_narval | 11 | planner | last 0.5 s before crashes vs completed flights | | 0.00193 vs 0.00035 | | 7.05x |

## outputs/batch2/planner_variants_summary/summary.md

Last modified 2026-11-02.

### Test-time planner variants (same agents, starts and sensor noise)

Success = full 10 s flight with late error below 10 cm. Cells: success / crash share, mean ± SE across seeds; in brackets the paired change vs the planner as trained.

| batch | seeds | variant | train_area | held_out | outside |
|---|---:|---|---|---|---|
| random_start_1M | 11 | policy_only | 60% ± 11% / 0% ± 0% (+18% ± 12% / -1% ± 0%) | 64% ± 13% / 0% ± 0% (+25% ± 12% / -3% ± 2%) | 41% ± 10% / 5% ± 2% (+20% ± 11% / -21% ± 5%) |
| random_start_1M | 11 | planner | 42% ± 9% / 1% ± 0% | 38% ± 9% / 3% ± 2% | 21% ± 6% / 26% ± 5% |
| random_start_1M | 11 | narrow_std | 52% ± 9% / 0% ± 0% (+10% ± 3% / -1% ± 0%) | 49% ± 11% / 0% ± 0% (+11% ± 6% / -3% ± 2%) | 29% ± 7% / 15% ± 4% (+8% ± 4% / -11% ± 3%) |
| random_start_1M | 11 | more_prior | 55% ± 10% / 0% ± 0% (+13% ± 8% / -1% ± 0%) | 47% ± 12% / 0% ± 0% (+8% ± 9% / -3% ± 2%) | 33% ± 6% / 10% ± 2% (+12% ± 4% / -16% ± 4%) |
| fixed_start_1M | 3 | policy_only | 45% ± 22% / 15% ± 1% (+19% ± 23% / -21% ± 11%) | 19% ± 11% / 54% ± 18% (+13% ± 10% / -18% ± 6%) | 0% ± 0% / 98% ± 1% (+0% ± 0% / +3% ± 2%) |
| fixed_start_1M | 3 | planner | 26% ± 6% / 36% ± 12% | 6% ± 4% / 72% ± 12% | 0% ± 0% / 94% ± 3% |
| fixed_start_1M | 3 | narrow_std | 39% ± 7% / 26% ± 9% (+13% ± 7% / -10% ± 3%) | 9% ± 9% / 71% ± 14% (+3% ± 5% / -1% ± 3%) | 0% ± 0% / 96% ± 3% (+0% ± 0% / +1% ± 1%) |
| fixed_start_1M | 3 | more_prior | 44% ± 7% / 29% ± 10% (+18% ± 6% / -7% ± 2%) | 19% ± 16% / 64% ± 21% (+13% ± 12% / -8% ± 10%) | 1% ± 1% / 91% ± 4% (+1% ± 1% / -3% ± 2%) |
| random_start_small_1M | 11 | policy_only | 9% ± 9% / 0% ± 0% (-68% ± 18% / -0% ± 0%) | 9% ± 9% / 0% ± 0% (-73% ± 18% / +0% ± 0%) | 12% ± 8% / 0% ± 0% (-50% ± 15% / -3% ± 2%) |
| random_start_small_1M | 11 | planner | 77% ± 10% / 0% ± 0% | 82% ± 10% / 0% ± 0% | 62% ± 9% / 3% ± 2% |
| random_start_small_1M | 11 | narrow_std | 70% ± 11% / 0% ± 0% (-7% ± 6% / -0% ± 0%) | 75% ± 12% / 0% ± 0% (-7% ± 4% / +0% ± 0%) | 55% ± 10% / 3% ± 3% (-6% ± 3% / +1% ± 1%) |
| random_start_small_1M | 11 | more_prior | 73% ± 11% / 0% ± 0% (-5% ± 5% / -0% ± 0%) | 74% ± 12% / 0% ± 0% (-8% ± 4% / +0% ± 0%) | 62% ± 9% / 2% ± 1% (+0% ± 2% / -1% ± 1%) |
| randomized_small_1M | 3 | policy_only | 35% ± 32% / 0% ± 0% (+22% ± 27% / +0% ± 0%) | 34% ± 33% / 0% ± 0% (+26% ± 29% / +0% ± 0%) | 27% ± 23% / 1% ± 1% (+21% ± 18% / +1% ± 1%) |
| randomized_small_1M | 3 | planner | 13% ± 7% / 0% ± 0% | 9% ± 5% / 0% ± 0% | 6% ± 6% / 0% ± 0% |
| randomized_small_1M | 3 | narrow_std | 16% ± 12% / 0% ± 0% (+3% ± 8% / +0% ± 0%) | 16% ± 12% / 0% ± 0% (+7% ± 10% / +0% ± 0%) | 10% ± 7% / 0% ± 0% (+4% ± 1% / +0% ± 0%) |
| randomized_small_1M | 3 | more_prior | 17% ± 9% / 0% ± 0% (+5% ± 2% / +0% ± 0%) | 26% ± 12% / 0% ± 0% (+17% ± 7% / +0% ± 0%) | 10% ± 7% / 0% ± 0% (+4% ± 2% / +0% ± 0%) |
| batch1_standard_1M | 3 | policy_only | 45% ± 10% / 15% ± 7% (+21% ± 8% / -49% ± 12%) | 23% ± 17% / 54% ± 14% (+20% ± 13% / -38% ± 7%) | 1% ± 1% / 77% ± 11% (+0% ± 0% / -12% ± 13%) |
| batch1_standard_1M | 3 | planner | 24% ± 8% / 63% ± 6% | 3% ± 3% / 92% ± 8% | 1% ± 1% / 89% ± 5% |
| batch1_standard_1M | 3 | narrow_std | 33% ± 9% / 50% ± 7% (+9% ± 2% / -13% ± 1%) | 6% ± 6% / 88% ± 12% (+2% ± 2% / -4% ± 4%) | 4% ± 1% / 81% ± 9% (+3% ± 2% / -8% ± 4%) |
| batch1_standard_1M | 3 | more_prior | 39% ± 6% / 45% ± 6% (+15% ± 2% / -19% ± 1%) | 9% ± 9% / 86% ± 14% (+6% ± 6% / -7% ± 7%) | 4% ± 1% / 80% ± 12% (+3% ± 0% / -9% ± 7%) |

## outputs/batch2/planner_variants_summary_batch1/summary.md

Last modified 2026-11-02.

### Test-time planner variants (same agents, starts and sensor noise)

Success = full 10 s flight with late error below 10 cm. Cells: success / crash share, mean ± SE across seeds; in brackets the paired change vs the planner as trained.

| batch | seeds | variant | train_area | held_out | outside |
|---|---:|---|---|---|---|
| batch1_standard_1M | 3 | policy_only | 45% ± 10% / 15% ± 7% (+21% ± 8% / -49% ± 12%) | 23% ± 17% / 54% ± 14% (+20% ± 13% / -38% ± 7%) | 1% ± 1% / 77% ± 11% (+0% ± 0% / -12% ± 13%) |
| batch1_standard_1M | 3 | planner | 24% ± 8% / 63% ± 6% | 3% ± 3% / 92% ± 8% | 1% ± 1% / 89% ± 5% |
| batch1_standard_1M | 3 | narrow_std | 33% ± 9% / 50% ± 7% (+9% ± 2% / -13% ± 1%) | 6% ± 6% / 88% ± 12% (+2% ± 2% / -4% ± 4%) | 4% ± 1% / 81% ± 9% (+3% ± 2% / -8% ± 4%) |
| batch1_standard_1M | 3 | more_prior | 39% ± 6% / 45% ± 6% (+15% ± 2% / -19% ± 1%) | 9% ± 9% / 86% ± 14% (+6% ± 6% / -7% ± 7%) | 4% ± 1% / 80% ± 12% (+3% ± 0% / -9% ± 7%) |
| batch1_small_narval | 11 | policy_only | 4% ± 4% / 6% ± 4% (-15% ± 9% / -3% ± 5%) | 3% ± 3% / 2% ± 1% (-10% ± 7% / -12% ± 7%) | 0% ± 0% / 17% ± 5% (-4% ± 2% / -15% ± 8%) |
| batch1_small_narval | 11 | planner | 19% ± 7% / 8% ± 4% | 13% ± 6% / 14% ± 7% | 4% ± 2% / 31% ± 5% |
| batch1_small_narval | 11 | narrow_std | 18% ± 8% / 7% ± 3% (-1% ± 3% / -2% ± 1%) | 18% ± 8% / 10% ± 6% (+5% ± 3% / -4% ± 2%) | 3% ± 2% / 32% ± 6% (-1% ± 1% / +0% ± 2%) |
| batch1_small_narval | 11 | more_prior | 22% ± 9% / 6% ± 3% (+3% ± 5% / -3% ± 2%) | 19% ± 10% / 8% ± 5% (+6% ± 5% / -6% ± 4%) | 5% ± 2% / 24% ± 5% (+1% ± 2% / -7% ± 3%) |
| batch1_standard | 11 | policy_only | 1% ± 1% / 77% ± 8% (+1% ± 1% / -20% ± 8%) | 0% ± 0% / 87% ± 4% (+0% ± 0% / -9% ± 5%) | 0% ± 0% / 82% ± 6% (+0% ± 0% / -18% ± 6%) |
| batch1_standard | 11 | planner | 0% ± 0% / 97% ± 1% | 0% ± 0% / 96% ± 2% | 0% ± 0% / 100% ± 0% |
| batch1_standard | 11 | narrow_std | 0% ± 0% / 96% ± 1% (+0% ± 0% / -1% ± 1%) | 0% ± 0% / 97% ± 1% (+0% ± 0% / +1% ± 2%) | 0% ± 0% / 99% ± 1% (+0% ± 0% / -1% ± 0%) |
| batch1_standard | 11 | more_prior | 0% ± 0% / 96% ± 1% (+0% ± 0% / -1% ± 1%) | 0% ± 0% / 95% ± 2% (+0% ± 0% / -2% ± 1%) | 0% ± 0% / 99% ± 0% (+0% ± 0% / -1% ± 1%) |

## outputs/batch2/horizon_summary/summary.md

Last modified 2026-11-02.

### Planning horizon changed at test time (same agents, starts and sensor noise)

Trained horizon: 3 steps (0.06 s). Success = full 10 s flight with late error below 10 cm. Cells: success / crash share, mean ± SE across seeds; in brackets the paired change in success vs horizon 3.

| batch | seeds | condition | horizon | train_area | outside |
|---|---:|---|---:|---|---|
| random_start_1M | 11 | none | 1 | 37% ± 10% / 0% ± 0% (-5% ± 6%) | 27% ± 6% / 7% ± 2% (+8% ± 6%) |
| random_start_1M | 11 | none | 3 | 43% ± 8% / 0% ± 0% | 19% ± 6% / 25% ± 4% |
| random_start_1M | 11 | none | 5 | 43% ± 10% / 1% ± 0% (-0% ± 6%) | 19% ± 4% / 32% ± 4% (+0% ± 4%) |
| random_start_1M | 11 | none | 10 | 29% ± 8% / 4% ± 1% (-13% ± 7%) | 8% ± 3% / 44% ± 5% (-12% ± 5%) |
| random_start_1M | 11 | wind 3 m/s | 1 | 0% ± 0% / 11% ± 8% (+0% ± 0%) | 0% ± 0% / 26% ± 4% (+0% ± 0%) |
| random_start_1M | 11 | wind 3 m/s | 3 | 0% ± 0% / 32% ± 8% | 0% ± 0% / 52% ± 6% |
| random_start_1M | 11 | wind 3 m/s | 5 | 0% ± 0% / 41% ± 10% (+0% ± 0%) | 0% ± 0% / 55% ± 7% (+0% ± 0%) |
| random_start_1M | 11 | wind 3 m/s | 10 | 0% ± 0% / 49% ± 9% (+0% ± 0%) | 0% ± 0% / 69% ± 6% (+0% ± 0%) |
| random_start_1M | 11 | mass x1.15 | 1 | 18% ± 6% / 0% ± 0% (+4% ± 5%) | 8% ± 2% / 22% ± 3% (+5% ± 3%) |
| random_start_1M | 11 | mass x1.15 | 3 | 14% ± 4% / 3% ± 1% | 3% ± 1% / 52% ± 4% |
| random_start_1M | 11 | mass x1.15 | 5 | 11% ± 4% / 6% ± 1% (-3% ± 3%) | 2% ± 1% / 58% ± 4% (-1% ± 1%) |
| random_start_1M | 11 | mass x1.15 | 10 | 5% ± 2% / 23% ± 3% (-9% ± 4%) | 0% ± 0% / 75% ± 3% (-3% ± 1%) |
| fixed_start_1M | 3 | none | 1 | 45% ± 10% / 27% ± 4% (+15% ± 2%) | 1% ± 1% / 94% ± 3% (+0% ± 2%) |
| fixed_start_1M | 3 | none | 3 | 30% ± 8% / 37% ± 11% | 1% ± 1% / 98% ± 2% |
| fixed_start_1M | 3 | none | 5 | 41% ± 7% / 39% ± 11% (+11% ± 1%) | 0% ± 0% / 97% ± 2% (-1% ± 1%) |
| fixed_start_1M | 3 | none | 10 | 29% ± 9% / 57% ± 10% (-1% ± 2%) | 1% ± 1% / 99% ± 1% (+0% ± 0%) |
| fixed_start_1M | 3 | wind 3 m/s | 1 | 0% ± 0% / 87% ± 13% (+0% ± 0%) | 0% ± 0% / 100% ± 0% (+0% ± 0%) |
| fixed_start_1M | 3 | wind 3 m/s | 3 | 0% ± 0% / 93% ± 7% | 0% ± 0% / 99% ± 1% |
| fixed_start_1M | 3 | wind 3 m/s | 5 | 0% ± 0% / 88% ± 12% (+0% ± 0%) | 0% ± 0% / 100% ± 0% (+0% ± 0%) |
| fixed_start_1M | 3 | wind 3 m/s | 10 | 0% ± 0% / 95% ± 5% (+0% ± 0%) | 0% ± 0% / 100% ± 0% (+0% ± 0%) |
| fixed_start_1M | 3 | mass x1.15 | 1 | 29% ± 13% / 40% ± 8% (+12% ± 8%) | 0% ± 0% / 99% ± 1% (+0% ± 0%) |
| fixed_start_1M | 3 | mass x1.15 | 3 | 17% ± 5% / 57% ± 11% | 0% ± 0% / 99% ± 1% |
| fixed_start_1M | 3 | mass x1.15 | 5 | 19% ± 7% / 61% ± 8% (+2% ± 3%) | 1% ± 1% / 99% ± 1% (+1% ± 1%) |
| fixed_start_1M | 3 | mass x1.15 | 10 | 13% ± 4% / 75% ± 8% (-3% ± 1%) | 0% ± 0% / 100% ± 0% (+0% ± 0%) |
| random_start_small_1M | 11 | none | 1 | 43% ± 14% / 0% ± 0% (-35% ± 13%) | 31% ± 10% / 4% ± 2% (-31% ± 9%) |
| random_start_small_1M | 11 | none | 3 | 78% ± 10% / 0% ± 0% | 62% ± 10% / 4% ± 3% |
| random_start_small_1M | 11 | none | 5 | 89% ± 8% / 0% ± 0% (+11% ± 7%) | 69% ± 9% / 5% ± 4% (+8% ± 4%) |
| random_start_small_1M | 11 | none | 10 | 72% ± 9% / 0% ± 0% (-6% ± 11%) | 49% ± 10% / 11% ± 4% (-12% ± 11%) |
| random_start_small_1M | 11 | wind 3 m/s | 1 | 0% ± 0% / 0% ± 0% (+0% ± 0%) | 0% ± 0% / 7% ± 4% (+0% ± 0%) |
| random_start_small_1M | 11 | wind 3 m/s | 3 | 0% ± 0% / 1% ± 1% | 0% ± 0% / 7% ± 3% |
| random_start_small_1M | 11 | wind 3 m/s | 5 | 0% ± 0% / 3% ± 3% (+0% ± 0%) | 0% ± 0% / 8% ± 4% (+0% ± 0%) |
| random_start_small_1M | 11 | wind 3 m/s | 10 | 0% ± 0% / 8% ± 6% (+0% ± 0%) | 0% ± 0% / 18% ± 6% (+0% ± 0%) |
| random_start_small_1M | 11 | mass x1.15 | 1 | 43% ± 14% / 1% ± 1% (-29% ± 12%) | 25% ± 8% / 12% ± 4% (-21% ± 6%) |
| random_start_small_1M | 11 | mass x1.15 | 3 | 72% ± 12% / 2% ± 2% | 46% ± 9% / 14% ± 6% |
| random_start_small_1M | 11 | mass x1.15 | 5 | 83% ± 9% / 2% ± 2% (+11% ± 6%) | 46% ± 9% / 16% ± 7% (-0% ± 3%) |
| random_start_small_1M | 11 | mass x1.15 | 10 | 59% ± 9% / 3% ± 3% (-14% ± 13%) | 29% ± 8% / 24% ± 6% (-17% ± 9%) |
| randomized_small_1M | 3 | none | 1 | 39% ± 26% / 0% ± 0% (+25% ± 32%) | 16% ± 9% / 1% ± 1% (+11% ± 6%) |
| randomized_small_1M | 3 | none | 3 | 14% ± 7% / 0% ± 0% | 4% ± 4% / 0% ± 0% |
| randomized_small_1M | 3 | none | 5 | 11% ± 11% / 0% ± 0% (-3% ± 10%) | 3% ± 3% / 0% ± 0% (-1% ± 1%) |
| randomized_small_1M | 3 | none | 10 | 15% ± 11% / 0% ± 0% (+1% ± 7%) | 1% ± 1% / 10% ± 6% (-3% ± 3%) |
| randomized_small_1M | 3 | wind 3 m/s | 1 | 0% ± 0% / 0% ± 0% (+0% ± 0%) | 0% ± 0% / 0% ± 0% (+0% ± 0%) |
| randomized_small_1M | 3 | wind 3 m/s | 3 | 0% ± 0% / 0% ± 0% | 0% ± 0% / 3% ± 3% |
| randomized_small_1M | 3 | wind 3 m/s | 5 | 0% ± 0% / 1% ± 1% (+0% ± 0%) | 0% ± 0% / 1% ± 1% (+0% ± 0%) |
| randomized_small_1M | 3 | wind 3 m/s | 10 | 0% ± 0% / 0% ± 0% (+0% ± 0%) | 0% ± 0% / 12% ± 3% (+0% ± 0%) |
| randomized_small_1M | 3 | mass x1.15 | 1 | 39% ± 27% / 0% ± 0% (+30% ± 31%) | 6% ± 3% / 0% ± 0% (+0% ± 4%) |
| randomized_small_1M | 3 | mass x1.15 | 3 | 9% ± 5% / 0% ± 0% | 6% ± 6% / 1% ± 1% |
| randomized_small_1M | 3 | mass x1.15 | 5 | 7% ± 6% / 0% ± 0% (-2% ± 6%) | 4% ± 2% / 6% ± 3% (-1% ± 5%) |
| randomized_small_1M | 3 | mass x1.15 | 10 | 15% ± 7% / 0% ± 0% (+7% ± 2%) | 3% ± 2% / 18% ± 10% (-2% ± 4%) |

## outputs/batch2/smoothing_summary/summary.md

Last modified 2026-11-02.

### Smoother ways of acting, and slow motors (training-area and held-out starts)

Cells: success / crash share, mean ± SE across seeds; in brackets the paired change in success vs the planner as trained at the same motor lag. Jitter: mean step-to-step change of the thrust command in the last 5 s of full flights (% of its range). Saturated: % of steps with a command at its limit.

| batch | seeds | motor lag | variant | success / crash | thrust jitter (%) | saturated (%) |
|---|---:|---:|---|---|---:|---:|
| random_start_1M | 11 | 0 ms | planner as trained | 40% ± 8% / 1% ± 0% | 23.0 ± 0.6 | 46 ± 1 |
| random_start_1M | 11 | 30 ms | planner as trained | 32% ± 7% / 2% ± 1% | 19.4 ± 0.5 | 47 ± 1 |
| random_start_1M | 11 | 60 ms | planner as trained | 21% ± 6% / 6% ± 1% | 17.2 ± 0.4 | 52 ± 1 |
| random_start_1M | 11 | 0 ms | averaged plan | 43% ± 8% / 1% ± 0% (+2% ± 2%) | 16.7 ± 0.6 | 1 ± 0 |
| random_start_1M | 11 | 30 ms | averaged plan | 35% ± 8% / 2% ± 1% (+3% ± 1%) | 13.7 ± 0.6 | 2 ± 0 |
| random_start_1M | 11 | 60 ms | averaged plan | 22% ± 6% / 5% ± 2% (+1% ± 2%) | 11.4 ± 0.5 | 3 ± 1 |
| random_start_1M | 11 | 0 ms | planner, 40 ms filter | 29% ± 6% / 3% ± 1% (-11% ± 3%) | 6.6 ± 0.2 | 6 ± 1 |
| random_start_1M | 11 | 30 ms | planner, 40 ms filter | 19% ± 5% / 6% ± 1% (-13% ± 3%) | 6.3 ± 0.1 | 9 ± 1 |
| random_start_1M | 11 | 60 ms | planner, 40 ms filter | 8% ± 3% / 22% ± 3% (-14% ± 4%) | 5.6 ± 0.1 | 15 ± 1 |
| random_start_1M | 11 | 0 ms | policy network only | 61% ± 12% / 0% ± 0% (+21% ± 11%) | 9.6 ± 0.6 | 1 ± 0 |
| random_start_1M | 11 | 30 ms | policy network only | 61% ± 11% / 0% ± 0% (+29% ± 12%) | 8.4 ± 0.5 | 1 ± 0 |
| random_start_1M | 11 | 60 ms | policy network only | 59% ± 11% / 0% ± 0% (+37% ± 12%) | 7.9 ± 0.5 | 2 ± 0 |
| random_start_small_1M | 11 | 0 ms | planner as trained | 79% ± 10% / 0% ± 0% | 29.1 ± 0.9 | 49 ± 2 |
| random_start_small_1M | 11 | 30 ms | planner as trained | 79% ± 10% / 0% ± 0% | 25.0 ± 0.6 | 50 ± 2 |
| random_start_small_1M | 11 | 60 ms | planner as trained | 77% ± 11% / 1% ± 1% | 22.7 ± 0.5 | 52 ± 2 |
| random_start_small_1M | 11 | 0 ms | averaged plan | 80% ± 10% / 0% ± 0% (+1% ± 1%) | 22.6 ± 1.3 | 1 ± 0 |
| random_start_small_1M | 11 | 30 ms | averaged plan | 80% ± 10% / 0% ± 0% (+2% ± 1%) | 19.4 ± 1.0 | 1 ± 0 |
| random_start_small_1M | 11 | 60 ms | averaged plan | 78% ± 10% / 1% ± 1% (+1% ± 1%) | 17.7 ± 0.8 | 3 ± 0 |
| random_start_small_1M | 11 | 0 ms | planner, 40 ms filter | 79% ± 10% / 0% ± 0% (+0% ± 1%) | 8.5 ± 0.2 | 2 ± 0 |
| random_start_small_1M | 11 | 30 ms | planner, 40 ms filter | 76% ± 11% / 1% ± 1% (-2% ± 1%) | 8.1 ± 0.2 | 3 ± 1 |
| random_start_small_1M | 11 | 60 ms | planner, 40 ms filter | 66% ± 12% / 3% ± 2% (-11% ± 4%) | 6.6 ± 0.2 | 15 ± 3 |
| random_start_small_1M | 11 | 0 ms | policy network only | 9% ± 9% / 0% ± 0% (-69% ± 18%) | 10.0 ± 2.1 | 2 ± 0 |
| random_start_small_1M | 11 | 30 ms | policy network only | 9% ± 9% / 0% ± 0% (-69% ± 18%) | 8.4 ± 1.6 | 2 ± 0 |
| random_start_small_1M | 11 | 60 ms | policy network only | 9% ± 9% / 0% ± 0% (-67% ± 18%) | 7.8 ± 1.4 | 3 ± 0 |

## outputs/batch2/precision_summary/summary.md

Last modified 2026-11-02.

### Where the precision comes from (full flights from the training-area and held-out starts)

Medians per seed, then mean ± SE across seeds, in cm. Height offset = late height minus 1 m (negative: too low).

| batch | decision rule | seeds | full flights | late error | horizontal | vertical | height offset |
|---|---|---:|---:|---:|---:|---:|---:|
| random_start_1M | with planning | 11 | 870 | 12.7 ± 1.5 | 12.4 ± 1.4 | 1.6 ± 0.3 | 0.4 ± 0.2 |
| random_start_1M | policy network only | 11 | 880 | 9.3 ± 1.4 | 8.5 ± 1.6 | 2.7 ± 0.4 | -0.2 ± 0.9 |
| random_start_1M | 64x2 planner | 11 | 880 | 9.9 ± 1.1 | 9.8 ± 1.1 | 1.2 ± 0.2 | 0.1 ± 0.2 |
| fixed_start_1M | with planning | 3 | 113 | 15.1 ± 1.7 | 14.4 ± 1.4 | 2.5 ± 0.4 | 0.2 ± 0.5 |
| fixed_start_1M | policy network only | 3 | 169 | 9.3 ± 2.0 | 6.2 ± 1.5 | 5.7 ± 2.0 | -5.5 ± 2.2 |
| fixed_start_1M | 64x2 planner | 3 | 133 | 9.4 ± 1.1 | 9.2 ± 1.1 | 1.5 ± 0.2 | -0.7 ± 0.3 |
| random_start_small_1M | with planning | 11 | 880 | 7.4 ± 0.9 | 7.1 ± 0.9 | 1.6 ± 0.3 | 1.1 ± 0.4 |
| random_start_small_1M | policy network only | 11 | 880 | 20.6 ± 3.0 | 15.1 ± 1.8 | 11.9 ± 3.1 | 7.9 ± 4.2 |
| random_start_small_1M | 64x2 planner | 11 | 880 | 7.8 ± 0.9 | 7.5 ± 0.9 | 1.7 ± 0.3 | 1.2 ± 0.4 |
| randomized_small_1M | with planning | 3 | 240 | 14.3 ± 2.0 | 14.0 ± 2.1 | 2.9 ± 0.4 | 2.3 ± 0.6 |
| randomized_small_1M | policy network only | 3 | 240 | 12.5 ± 2.9 | 10.3 ± 1.7 | 6.4 ± 2.9 | 6.1 ± 3.2 |
| randomized_small_1M | 64x2 planner | 3 | 240 | 14.5 ± 2.2 | 14.1 ± 2.3 | 2.8 ± 0.4 | 2.6 ± 0.6 |
| sharp_small_1M | with planning | 3 | 240 | 10.3 ± 3.0 | 10.1 ± 3.0 | 1.4 ± 0.5 | 1.2 ± 0.6 |
| sharp_small_1M | policy network only | 3 | 240 | 22.3 ± 5.5 | 22.1 ± 5.4 | 2.9 ± 1.5 | -1.1 ± 2.4 |
| sharp_small_1M | 64x2 planner | 3 | 240 | 10.4 ± 3.0 | 10.3 ± 3.0 | 1.4 ± 0.4 | 1.2 ± 0.6 |
| batch1_standard_1M | with planning | 3 | 59 | 10.2 ± 3.2 | 9.8 ± 2.8 | 1.6 ± 0.6 | 0.2 ± 0.4 |
| batch1_standard_1M | policy network only | 3 | 169 | 9.3 ± 2.4 | 8.9 ± 2.6 | 2.2 ± 0.4 | -1.1 ± 1.1 |

## outputs/batch2/flight_quality/summary.md

Last modified 2026-11-02.

### Flight quality (training-area and held-out starts, final agents)

Mean ± SE across seeds of each seed's value. reach: median time to first come within 20 cm (s); settle: median time to stay within 10 cm until the end (s); settled: share of full flights that do; jitter: mean step-to-step change of the command in the last 5 s, % of its range; saturated: % of steps with a command at its limit; p90 and worst: late error of full flights (cm).

| batch | decision rule | seeds | reach 20 cm (s) | settle (s) | settled | thrust jitter (%) | moment jitter (%) | saturated (%) | p90 late error (cm) | worst late error (cm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_start_1M | with planning | 11 | 1.80 ± 0.03 | 8.62 ± 0.33 | 48% ± 6% | 23.03 ± 0.64 | 21.80 ± 0.88 | 45.8 ± 0.9 | 21.3 ± 2.9 | 50.6 ± 12.8 |
| random_start_1M | policy network only | 11 | 1.73 ± 0.02 | 5.28 ± 0.86 | 57% ± 12% | 9.58 ± 0.62 | 5.95 ± 0.24 | 1.0 ± 0.1 | 13.9 ± 1.6 | 18.2 ± 2.3 |
| random_start_1M | 64x2 planner | 11 | 1.72 ± 0.03 | 8.13 ± 0.50 | 57% ± 8% | 25.18 ± 0.61 | 27.94 ± 0.61 | 54.1 ± 2.0 | 14.9 ± 1.6 | 21.1 ± 2.6 |
| fixed_start_1M | with planning | 3 | 2.27 ± 0.11 | 7.05 ± 1.49 | 44% ± 4% | 22.58 ± 0.85 | 22.21 ± 1.21 | 43.2 ± 3.6 | 50.1 ± 3.8 | 172.8 ± 35.6 |
| fixed_start_1M | policy network only | 3 | 2.05 ± 0.09 | 5.85 ± 1.21 | 45% ± 25% | 10.62 ± 1.92 | 7.64 ± 0.43 | 11.5 ± 3.8 | 11.8 ± 2.0 | 25.8 ± 12.3 |
| fixed_start_1M | 64x2 planner | 3 | 2.17 ± 0.19 | 6.21 ± 1.59 | 47% ± 8% | 27.25 ± 0.71 | 29.45 ± 0.76 | 67.6 ± 4.1 | 51.4 ± 15.5 | 142.8 ± 23.8 |
| random_start_small_1M | with planning | 11 | 1.75 ± 0.03 | 6.74 ± 0.63 | 79% ± 9% | 29.06 ± 0.85 | 32.93 ± 0.86 | 49.3 ± 1.8 | 10.0 ± 1.3 | 27.5 ± 15.4 |
| random_start_small_1M | policy network only | 11 | 2.15 ± 0.20 | 8.19 ± 1.68 | 10% ± 9% | 10.05 ± 2.14 | 8.24 ± 0.50 | 2.0 ± 0.2 | 22.0 ± 3.1 | 22.9 ± 3.2 |
| random_start_small_1M | 64x2 planner | 11 | 1.75 ± 0.03 | 6.62 ± 0.72 | 75% ± 10% | 32.03 ± 0.77 | 37.28 ± 0.72 | 65.0 ± 1.9 | 10.6 ± 1.3 | 21.5 ± 9.0 |
| randomized_small_1M | with planning | 3 | 2.06 ± 0.24 | 9.49 ± 0.07 | 32% ± 15% | 26.93 ± 1.91 | 31.57 ± 1.18 | 50.5 ± 1.7 | 25.5 ± 6.9 | 42.9 ± 21.8 |
| randomized_small_1M | policy network only | 3 | 1.95 ± 0.21 | 8.41 ± 1.33 | 36% ± 27% | 27.08 ± 3.53 | 10.55 ± 0.80 | 2.6 ± 0.9 | 15.9 ± 3.2 | 20.6 ± 5.5 |
| randomized_small_1M | 64x2 planner | 3 | 2.03 ± 0.24 | 9.53 ± 0.09 | 25% ± 11% | 31.09 ± 1.81 | 35.12 ± 1.58 | 66.9 ± 2.4 | 19.6 ± 2.5 | 34.2 ± 13.4 |
| sharp_small_1M | with planning | 3 | 1.87 ± 0.09 | 6.84 ± 0.81 | 45% ± 23% | 25.05 ± 1.77 | 33.12 ± 1.61 | 44.5 ± 2.5 | 15.0 ± 1.3 | 16.3 ± 1.5 |
| sharp_small_1M | policy network only | 3 | 2.27 ± 0.68 | 8.94 ± 0.00 | 16% ± 16% | 17.70 ± 2.83 | 13.37 ± 1.41 | 1.9 ± 0.3 | 24.8 ± 3.5 | 25.6 ± 3.0 |
| sharp_small_1M | 64x2 planner | 3 | 1.85 ± 0.09 | 7.59 ± 0.05 | 47% ± 24% | 28.53 ± 1.65 | 37.82 ± 1.44 | 61.0 ± 3.1 | 15.3 ± 1.2 | 16.4 ± 1.2 |
| batch1_standard_1M | with planning | 3 | 2.59 ± 0.51 | 6.09 ± 0.93 | 56% ± 16% | 24.28 ± 1.19 | 24.02 ± 3.75 | 46.5 ± 1.1 | 79.7 ± 32.0 | 115.2 ± 36.9 |
| batch1_standard_1M | policy network only | 3 | 2.25 ± 0.32 | 6.23 ± 0.84 | 51% ± 18% | 14.68 ± 4.21 | 9.72 ± 1.56 | 12.0 ± 2.3 | 15.1 ± 1.6 | 21.3 ± 1.2 |

## results/tdmpc2_batch1/README.md

Last modified 2026-10-13.

### TD-MPC2 vs PPO, Crazyflie hover: Batch 1 results

Start with **`report_batch1/Crazyflie_TD-MPC2_vs_PPO_Batch1_Report.pdf`**. It explains the task, both
methods, every experiment below, and what the results mean.

Task for every run: one drone starts at (0, 0, 0.2) m and hovers at (0, 0, 1) m for 10 s. After training,
each model flew 3 test flights from 6 start positions (the training start plus 5 it never practised from).

| Folder | Experiment | Runs |
| --- | --- | --- |
| `crazyflie_hover_s0/` | September runs (700, 20k, 50k steps), re-tested | 1 each |
| `batch1_fixed_start/` | TD-MPC2 small settings, 50k steps, home PC | 11 seeds |
| `batch1_small_narval/` | Same as above, repeated on Narval | 11 seeds |
| `batch1_standard/` | TD-MPC2 published settings, 50k steps, Narval | 11 seeds |
| `batch1_standard_1M/` | TD-MPC2 published settings, 1M steps, Narval | 3 seeds |
| `ppo_fixed_start/` | Lab's Brax PPO trained on the same task, Narval | 11 seeds |
| `ppo_existing_checkpoint/` | Lab's earlier PPO model, tested the same way | 1 |
| `comparison_batch1/`, `comparison_with_1M/` | Side-by-side plots and tables | |

Inside each batch folder:

- `analysis/summary.md`: per-seed table, mean ± standard error, results per start position
- `analysis/*.png`: learning curves, flights from the training start, results per start position
- `seed_XX/learning_curve.csv`: test flights every 5,000 training steps
- `seed_XX/evaluation/summary.csv`: one row per final test flight
- `seed_XX/evaluation/trajectories.csv`: every step of every final test flight
- `seed_XX/.hydra/config.yaml`: exact settings of that run

Saved model checkpoints and raw training logs are not in git because of their size. The models are on
the Narval cluster under `/scratch/mounirz/tdmpc2/` (scratch is cleaned after about 60 days).

The plots and summaries were made with `code/src/tdmpc2_training/analyze_seeds.py` and
`compare_methods.py`. `report_batch1/make_report.py` rebuilds the PDF; it reads from `outputs/`, so
point its `OUT` path at this folder if `outputs/` is not present.

## results/tdmpc2_batch1/comparison_batch1/comparison.md

Last modified 2026-10-12.

### Method comparison: one-drone hover, start (0, 0, 0.2), target (0, 0, 1)

Mean ± standard error across seeds. Late error = mean distance to target over the last 2 s flown.

| method | seeds | training steps | training time (min) | late error, training start (m) | late error, all starts (m) | crashes, all starts |
|---|---:|---:|---:|---:|---:|---:|
| TD-MPC2 small | 11 | 50,000 | 14.0 ± 0.3 | 0.320 ± 0.092 | 0.454 ± 0.121 | 11/198 |
| TD-MPC2 standard | 11 | 50,000 | 20.2 ± 0.2 | 1.061 ± 0.140 | 1.122 ± 0.052 | 182/198 |
| PPO | 11 | 78,643,200 | 11.4 ± 0.0 | 0.172 ± 0.085 | 0.711 ± 0.032 | 157/198 |

## results/tdmpc2_batch1/comparison_with_1M/comparison.md

Last modified 2026-10-13.

### Method comparison: one-drone hover, start (0, 0, 0.2), target (0, 0, 1)

Mean ± standard error across seeds. Late error = mean distance to target over the last 2 s flown.

| method | seeds | training steps | training time (min) | late error, training start (m) | late error, all starts (m) | crashes, all starts |
|---|---:|---:|---:|---:|---:|---:|
| TD-MPC2 small 50k | 11 | 50,000 | 14.0 ± 0.3 | 0.320 ± 0.092 | 0.454 ± 0.121 | 11/198 |
| TD-MPC2 standard 50k | 11 | 50,000 | 20.2 ± 0.2 | 1.061 ± 0.140 | 1.122 ± 0.052 | 182/198 |
| TD-MPC2 standard 1M | 3 | 1,000,000 | 359.2 ± 11.5 | 0.057 ± 0.005 | 0.788 ± 0.047 | 36/54 |
| PPO | 11 | 78,643,200 | 11.4 ± 0.0 | 0.172 ± 0.085 | 0.711 ± 0.032 | 157/198 |

# Part 15. Key source code

Unchanged copies of the files that define the task, the experiments and the algorithm. Long lines are wrapped for printing.


## code/src/gpu_training/constants.py

Lab constants: hover thrust, reward weights, ranges. Last modified 2026-09-23.

```
"""Physical, training, and reward/observation constants shared by the GPU
(MJX/Brax) Crazyflie training pipeline."""
import os

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Scene paths
# ---------------------------------------------------------------------------
# Assumes assets_mjx lives alongside this file, as it does in this repo.
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets_mjx", "bitcraze_crazyflie_2")

SCENE_PATH_1_DRONE = os.path.join(_ASSETS_DIR, "scene.xml")
SCENE_PATH_2_DRONES = os.path.join(_ASSETS_DIR, "scene_multi.xml")

# ---------------------------------------------------------------------------
# Base target position and training range
# ---------------------------------------------------------------------------
TARGET_POS = jnp.array([0.0, 0.0, 1.0])
TRAINING_POS_RANGE = 2.0

# ---------------------------------------------------------------------------
# Basic drone / action / observation sizing
# ---------------------------------------------------------------------------
BASE_HOVER_THRUST = 0.26487
QPOS_PER_DRONE = 7   # pos (3) + quat (4)
QVEL_PER_DRONE = 6   # linear vel (3) + angular vel (3)
ACTION_SIZE_PER_DRONE = 4  # thrust, roll, pitch, yaw
N_PREV_ACTIONS = 1   # number of previous actions stored in the observation

# ---------------------------------------------------------------------------
# Random starting state sampled from distribution
# ---------------------------------------------------------------------------
TRAINING_QUAT_RANGE = np.pi / 8
TRAINING_VEL_RANGE = 0.2
TRAINING_ANG_VEL_RANGE = 0.1

# Out of bounds if this much further from target compared to starting distance
OUT_OF_BOUNDS_RANGE = 2.0

# ---------------------------------------------------------------------------
# Reward and observation constants
# ---------------------------------------------------------------------------
TERMINATION_WEIGHT = 1.0
SURVIVAL_WEIGHT = TERMINATION_WEIGHT / 1000
DRONE_PROXIMITY_WEIGHT = 0.01
TARGET_PROXIMITY_WEIGHT = 0.01

DRONE_PROXIMITY_THRESHOLD = 0.2
DRONE_COLLISION_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Model / checkpoint save locations (local training runs)
# ---------------------------------------------------------------------------
_RL_MODELS_GPU_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "rl_models", "gpu")

PPO_MODEL_SAVE_PATH = os.path.join(_RL_MODELS_GPU_DIR, "PPO_brax_model_checkpoint.pkl")
MAPPO_CHECKPOINT_DIR = os.path.join(_RL_MODELS_GPU_DIR, "MAPPO_model_checkpoint")
TRANSLATOR_MODEL_SAVE_PATH = os.path.join(_RL_MODELS_GPU_DIR, 
    "translator_brax_model_checkpoint.pkl")
```

## code/src/gpu_training/crazyflie_env.py

Lab MJX environment: reset, step, reward, observation. Last modified 2026-09-23.

```
"""
CrazyflieEnv: MJX (MuJoCo playground) environment for the Crazyflie drone,
supporting multiple drones and MARL rewards.
"""
from ml_collections import config_dict
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from constants import (
    TARGET_POS,
    QPOS_PER_DRONE,
    QVEL_PER_DRONE,
    ACTION_SIZE_PER_DRONE,
    N_PREV_ACTIONS,
    TRAINING_POS_RANGE,
    TRAINING_QUAT_RANGE,
    TRAINING_VEL_RANGE,
    TRAINING_ANG_VEL_RANGE,
    OUT_OF_BOUNDS_RANGE,
    TERMINATION_WEIGHT,
    SURVIVAL_WEIGHT,
    DRONE_PROXIMITY_WEIGHT,
    TARGET_PROXIMITY_WEIGHT,
    DRONE_PROXIMITY_THRESHOLD,
    DRONE_COLLISION_THRESHOLD,
    SCENE_PATH_2_DRONES,
)
from math_utils import clipped_normal, quat_to_rotation_matrix, add_quat_noise


# MuJoCo playground: Extending mjx_env
# https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/mjx_env.py
class CrazyflieEnv(mjx_env.MjxEnv):
    """MJX environment using the Crazyflie drone model."""

    def __init__(self, scene_path = SCENE_PATH_2_DRONES, num_drones = 2):
        # Define environment configuration (control and sim time steps)
        cfg = config_dict.ConfigDict()
        cfg.ctrl_dt = 0.02    # control step (50Hz)
        cfg.sim_dt = 0.002    # simulation step (500Hz)
        super().__init__(cfg)

        # Load MuJoCo model and convert to MJX
        self.scene_path = scene_path
        self._mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self._mjx_model = mjx.put_model(self._mj_model)

        self.num_drones = num_drones

        # Define control/action size as number of actuators (e.g. 8 for 2 drones)
        self._action_size = self._mj_model.nu


    # Expected env properties for Brax PPO training
    @property
    def xml_path(self) -> str:
        return self.scene_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


    def reset(self, rng: jax.Array, random_reset: bool=True, target: jax.Array=TARGET_POS) -> 
        mjx_env.State:
        """Vectorized reset for Crazyflie drones."""

        # Random keys for state initialization and physics
        (rng,
        rng_target,
        rng_pos,
        rng_quat,
        rng_vel,
        rng_ang,
        rng_mass,
        rng_inertia,
        rng_actuator) = jax.random.split(rng, 9)


        # Random target position within TRAINING_POS_RANGE (Z >= 0.1)
        xy = jax.random.uniform(
            rng_target, (2,),
            minval=-TRAINING_POS_RANGE,
            maxval= TRAINING_POS_RANGE
        )
        z = jax.random.uniform(
            rng_target, (1,),
            minval=0.1,
            maxval=TRAINING_POS_RANGE
        )
        target_pos = jnp.concatenate([xy, z])

        # Random starting positions for drone(s) within TRAINING_POS_RANGE (Z >= 0.1)
        xy_drone = jax.random.uniform(
            rng_pos, (self.num_drones, 2),
            minval=-TRAINING_POS_RANGE,
            maxval= TRAINING_POS_RANGE
        )
        z_drone = jax.random.uniform(
            rng_pos, (self.num_drones, 1),
            minval=0.1,
            maxval=TRAINING_POS_RANGE
        )
        # Concatenate along last dimension to support batch operations
        start_pos = jnp.concatenate([xy_drone, z_drone], axis=-1)

        def enforce_min_spacing(pos, min_dist):
            """
            Ensure drones don't start too close together.

            Parameters
            ----------
            pos: Drone starting positions, shape (NUM_DRONES, 3)
            min_dist: Minimum starting distance between drones
            """
            # Pairwise differences for drone positions --> Add axis to compare drones
            # E.g. diffs[drone_1, drone_2, :] = pos[drone_1, :] - pos[drone_2, :]
            diffs = pos[:, None, :] - pos[None, :, :]      # (N, N, 3)

            # Compute magnitude of each pairwise difference, "ignore" self-difference
            # Set self-difference (e.g. drone 1 - drone 1) to large value
            dists = jnp.linalg.norm(diffs + jnp.eye(self.num_drones)[..., None] * 1e6, axis=-1)

            # Compute pair distance violation, 0 if outside of violation
            violation = jnp.maximum(0.0, min_dist - dists)  # (N, N)

            # Move each drone outside of violation along the direction of separation
            direction = diffs / (dists[..., None] + 1e-9)
            # Ensure we don't push drones downward (to avoid pushing into floor)
            direction = direction.at[..., 2].set(jnp.maximum(direction[..., 2], 0.0))
            adjust = jnp.sum(direction * violation[..., None] * 0.5, axis=1)

            return pos + adjust

        # Ensure drone spacing (don't start too close together)
        start_pos = enforce_min_spacing(start_pos, DRONE_PROXIMITY_THRESHOLD)

        # Random quaternion (different rng for axis and angle)
        rng_quat_axes, rng_quat_angles = jax.random.split(rng_quat, 2)
        std_angle = TRAINING_QUAT_RANGE / 2.0
        axes = jax.random.normal(rng_quat_axes, (self.num_drones, 3))
        axes = axes / jnp.linalg.norm(axes, axis=1, keepdims=True)
        angles = jax.random.normal(rng_quat_angles, (self.num_drones,)) * std_angle
        w = jnp.cos(angles)
        quats = jnp.concatenate([w[:, None], axes * jnp.sin(angles)[:, None]], axis=1)

        # Linear and angular velocities
        lin_vel = (TRAINING_VEL_RANGE / 2.0) * jax.random.normal(rng_vel, (self.num_drones, 3))
        ang_vel = (TRAINING_ANG_VEL_RANGE / 2.0) * jax.random.normal(rng_ang, (self.num_drones, 
            3))

        # Domain randomization for drone physics (clipped standard normal distribution)
        mass_noise = clipped_normal(
            rng_mass,
            self._mjx_model.body_mass.shape,
            std=1.0,
            std_clip=3.0
        )
        inertia_noise = clipped_normal(
            rng_inertia,
            self._mjx_model.body_inertia.shape,
            std=1.0,
            std_clip=3.0
        )
        actuator_noise = clipped_normal(
            rng_actuator,
            self._mjx_model.actuator_gainprm.shape,
            std=1.0,
            std_clip=3.0
        )

        # Scale noise based on property magnitude and apply it
        body_mass = self._mjx_model.body_mass * (1 + 0.05 * mass_noise)
        body_inertia = self._mjx_model.body_inertia * (1 + 0.05 * inertia_noise)
        actuator_gainprm = self._mjx_model.actuator_gainprm * (1 + 0.05 * actuator_noise)

        # Option to disable/override random stateful noise
        if not random_reset:
            # Line drones on X axis, start near floor
            x_offsets = jnp.linspace(
                -(self.num_drones - 1) / 2,
                (self.num_drones - 1) / 2,
                self.num_drones
            )
            y_offsets = jnp.zeros_like(x_offsets)
            z_offsets = jnp.full_like(x_offsets, 0.2)

            start_pos = jnp.stack([x_offsets, y_offsets, z_offsets], axis=1)

            lin_vel = jnp.zeros((self.num_drones, 3))
            quats = jnp.array([[1., 0., 0., 0.]] * self.num_drones)
            ang_vel = jnp.zeros((self.num_drones, 3))
            target_pos = target

            body_mass = self._mjx_model.body_mass
            body_inertia = self._mjx_model.body_inertia
            actuator_gainprm = self._mjx_model.actuator_gainprm

        # Assemble qpos/qvel in batch
        qpos = jnp.zeros((self.num_drones, QPOS_PER_DRONE), dtype=jnp.float32)
        qpos = qpos.at[:, 0:3].set(start_pos)
        qpos = qpos.at[:, 3:7].set(quats)
        qpos = qpos.reshape(-1)  # flatten for mjx_env

        qvel = jnp.zeros((self.num_drones, QVEL_PER_DRONE), dtype=jnp.float32)
        qvel = qvel.at[:, 0:3].set(lin_vel)
        qvel = qvel.at[:, 3:6].set(ang_vel)
        qvel = qvel.reshape(-1)  # flatten for mjx_env

        # Initialize state
        action_history = jnp.zeros((N_PREV_ACTIONS, self.num_drones, ACTION_SIZE_PER_DRONE))
        initial_distance = jnp.linalg.norm(start_pos - target_pos, axis=1)

        data = mjx_env.init(self._mjx_model, qpos=qpos, qvel=qvel)

        # Metrics
        metrics = {
            "reward/distance_improvement": jnp.zeros(()),
            "reward/proximity_bonus": jnp.zeros(()),
            "reward/survival_bonus": jnp.zeros(()),
            "reward/termination": jnp.zeros(()),
            "reward/drone_proximity_penalty": jnp.zeros(()),
            "reward": jnp.zeros(())
        }

        # State info
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "initial_distance": initial_distance,
            "prev_pos": start_pos,
            "action_history": action_history, # (N_prev_actions, N_drones, Action)
            # Pass physics noise in info to step (entire mjx_model is too large)
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "actuator_gainprm": actuator_gainprm,
            # Nominal gravity by default; callers wanting per-episode gravity
            # randomization (e.g. cf_translator_env.py) can overwrite this key.
            "gravity": self._mjx_model.opt.gravity,
        }

        reward = jnp.zeros(())
        done = jnp.zeros(())

        obs = self._get_obs(data, info)

        return mjx_env.State(data=data, obs=obs, reward=reward, done=done, metrics=metrics, 
            info=info)


    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Vectorized Crazyflie step for multiple drones with MARL rewards."""

        # Retrieve stateful info for drone(s)
        prev_pos = state.info["prev_pos"]
        init_dist = state.info["initial_distance"]
        target_pos = state.info["target_pos"]
        mjx_model = self._mjx_model.replace(
            body_mass = state.info["body_mass"],
            body_inertia = state.info["body_inertia"],
            actuator_gainprm = state.info["actuator_gainprm"],
            # Optional per-episode gravity override (see reset() above)
            opt = self._mjx_model.opt.replace(
                gravity = state.info.get("gravity", self._mjx_model.opt.gravity)
            ),
        )

        # Clip flat action (by default done via XML actuators but we are also recording action 
            history)
        action_clipped = action.at[0::4].set(jnp.clip(action[0::4], 0.0, 0.35))   # thrusts
        action_clipped = action_clipped.at[1::4].set(jnp.clip(action[1::4], -1.0, 1.0))  # rolls
        action_clipped = action_clipped.at[2::4].set(jnp.clip(action[2::4], -1.0, 1.0))  # pitches
        action_clipped = action_clipped.at[3::4].set(jnp.clip(action[3::4], -1.0, 1.0))  # yaws

        # self.n_substeps defined in mjx_env as self.dt / self.sim_dt (10 in this case)
        data = mjx_env.step(mjx_model, state.data, action_clipped, n_substeps=self.n_substeps)
        obs = self._get_obs(data, state.info)
        qpos = data.qpos.reshape(self.num_drones, QPOS_PER_DRONE)

        # Dynamic step rewards: Distance to target improvement and target proximity
        pos = qpos[:, :3]
        distance_to_target = jnp.linalg.norm(target_pos - pos, axis=1)
        distance_to_target_old = jnp.linalg.norm(target_pos - prev_pos, axis=1)
        distance_improvement = distance_to_target_old - distance_to_target

        proximity_bonus = TARGET_PROXIMITY_WEIGHT * (1 - jnp.tanh(distance_to_target))

        # Compute pairwise distances between drones
        diffs = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        dists = jnp.linalg.norm(diffs + jnp.eye(self.num_drones)[..., None] * 1e9, axis=-1)  # (N,
             N)

        # Proximity penalty for being closer than threshold to another drone
        proximity_penalty_matrix = jnp.where(
            dists < DRONE_PROXIMITY_THRESHOLD,
            DRONE_PROXIMITY_WEIGHT * (1.0 - dists / DRONE_PROXIMITY_THRESHOLD),
            0.0,
        )
        # Zero out self-distances
        proximity_penalty_matrix = proximity_penalty_matrix * (1.0 - jnp.eye(self.num_drones))
        drone_proximity_penalty = jnp.sum(proximity_penalty_matrix, axis=1)

        # Collision check
        collision_matrix = (dists < DRONE_COLLISION_THRESHOLD).astype(jnp.float32)
        collision_any = jnp.clip(jnp.sum(collision_matrix * (1 - jnp.eye(self.num_drones)), 
            axis=1), 0, 1)

        # Termination conditions: Crash, stray from target, drone collision
        done_per_drone = (
            (pos[:, 2] < 0.05)
            | (distance_to_target > init_dist + OUT_OF_BOUNDS_RANGE)
            | (collision_any > 0)
        )

        # Reward per drone, 1D jax arrays of shape (self.num_drones,)
        reward_per_drone = (
            SURVIVAL_WEIGHT
            + proximity_bonus
            + distance_improvement
            - drone_proximity_penalty
            - done_per_drone.astype(jnp.float32) * TERMINATION_WEIGHT
        )

        # Aggregate reward for the environment
        reward = jnp.sum(reward_per_drone)
        done_all = jnp.any(done_per_drone).astype(jnp.float32)

        # Aggregate per-drone metrics to scalars per environment (match Brax format)
        new_metrics = {
            "reward/distance_improvement": jnp.sum(distance_improvement),
            "reward/proximity_bonus": jnp.sum(proximity_bonus),
            "reward/survival_bonus": SURVIVAL_WEIGHT * self.num_drones,
            "reward/termination": - done_all * TERMINATION_WEIGHT,
            "reward/drone_proximity_penalty": - jnp.sum(drone_proximity_penalty),
            "reward": reward,
        }

        # Reshape flat action back to per drone, update action history
        actions_per_drone = action_clipped.reshape(self.num_drones, ACTION_SIZE_PER_DRONE)
        new_action_history = jnp.concatenate(
            [state.info["action_history"][1:],
            actions_per_drone[jnp.newaxis, ...]],
            axis=0
        )

        # Update info (prev pos and action history)
        new_info = dict(state.info)
        new_info["prev_pos"] = pos
        new_info["action_history"] = new_action_history

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done_all,
            metrics=new_metrics,
            info=new_info,
        )


    def _get_obs(self, data, info):
        qpos = data.qpos.reshape(self.num_drones, QPOS_PER_DRONE)
        qvel = data.qvel.reshape(self.num_drones, QVEL_PER_DRONE)

        ##################################
        # Apply noise to base observations
        ##################################

        new_rng, pos_rng, quat_rng, vel_rng, ang_vel_rng = jax.random.split(info["rng"], 5)
        pos_noise_std = 0.01
        vel_noise_std = 0.005
        ang_vel_noise_std = 0.001
        quat_noise_std = 0.001

        # Add clipped noise (within n std) to base observations
        pos = qpos[:, :3] + clipped_normal(pos_rng, qpos[:, :3].shape, pos_noise_std, 
            std_clip=3.0)
        quat, _ = add_quat_noise(
            quat_rng,
            qpos[:, 3:7],
            quat_noise_std,
            3.0
        )
        vel = qvel[:, :3] + clipped_normal(vel_rng, qvel[:, :3].shape, vel_noise_std, 
            std_clip=3.0)
        ang_vel = qvel[:, 3:6] + clipped_normal(ang_vel_rng, qvel[:, 3:6].shape, 
            ang_vel_noise_std, std_clip=3.0)

        # Update next-step rng so that it does not use the same noise
        info["rng"] = new_rng

        #########################
        # Add engineered features
        #########################

        # Convert drone(s) quat to rotation matrix
        rot_mats = jax.vmap(quat_to_rotation_matrix)(quat) # (N, 3, 3)

        # Get relative position to target in drone body coordinates (normalize by initial dist)
        initial_dist = info["initial_distance"][:, None] # (self.num_drones,) --> (N, 1)
        target_pos = info["target_pos"]
        pos_error = target_pos - pos # (N, 3)
        pos_error_norm = pos_error / (initial_dist + 1e-9)
        rel_pos_body = jnp.einsum('nij,nj->ni', rot_mats.transpose((0, 2, 1)), pos_error_norm)

        # Get linear velocity in drone body coordinates
        linear_vel_body = jnp.einsum('nij,nj->ni', rot_mats.transpose((0, 2, 1)), vel)

        # Compute relative positions to other drones in body coordinates
        # Currently includes self-relative pos (always [0, 0, 0]) - maybe remove with mask or 
            similar
        rel_pos_all = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        rel_pos_all_body = jnp.einsum('nij,nkj->nki', rot_mats.transpose((0, 2, 1)), rel_pos_all)
              # (N, N, 3)
        rel_pos_all_body_flat = rel_pos_all_body.reshape(self.num_drones, -1)

        # Base state (per drone)
        core = jnp.concatenate(
            [
                pos, vel, ang_vel, rot_mats.reshape(self.num_drones, -1),
                rel_pos_body, linear_vel_body, rel_pos_all_body_flat
            ],
            axis=1,
        )

        # Append action history per drone
        def obs_for_drone(i):
            act_hist_flat = info["action_history"][:, i, :].reshape(-1)
            return jnp.concatenate([core[i], act_hist_flat])

        obs_per_drone = jax.vmap(obs_for_drone)(jnp.arange(self.num_drones))

        # Final flattened observation
        return obs_per_drone.reshape(-1)
```

## code/src/tdmpc2_training/crazyflie_hover.py

TD-MPC2 adapter: one drone, random starts, metric target. Last modified 2026-11-01.

```
"""Gymnasium interface for the existing one-drone MJX Crazyflie task."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

from constants import SCENE_PATH_1_DRONE, TARGET_POS
from crazyflie_env import CrazyflieEnv

# Physical control limits: thrust (N) and the three body moments.
ACTION_LOW = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([0.35, 1.0, 1.0, 1.0], dtype=np.float32)

# Batch 2 training starts (metres): drawn uniformly from this box for every
# training flight, except the held-out block (all heights) and anything
# closer than MIN_GOAL_DISTANCE to the target. The held-out block is never
# trained on and is only used for testing (hover_eval.TEST_SETS).
START_AREA_LOW = np.array([-0.75, -0.75, 0.2])
START_AREA_HIGH = np.array([0.75, 0.75, 1.5])
HELD_OUT_XY_LOW = np.array([0.25, 0.25])
HELD_OUT_XY_HIGH = np.array([0.75, 0.75])
MIN_GOAL_DISTANCE = 0.3


def in_held_out_block(position):
    xy = np.asarray(position)[:2]
    return bool(np.all(xy >= HELD_OUT_XY_LOW) and np.all(xy <= HELD_OUT_XY_HIGH))


def sample_training_start(rng, target=TARGET_POS):
    """One start from the Batch 2 training distribution (numpy Generator)."""
    target = np.asarray(target, dtype=np.float64)
    while True:
        start = rng.uniform(START_AREA_LOW, START_AREA_HIGH)
        if not in_held_out_block(start) and np.linalg.norm(start - target) >= MIN_GOAL_DISTANCE:
            return start


class HoverDrone(CrazyflieEnv):
    """The lab's one-drone environment, optionally with the target offset in metres.

    The lab's observation divides the offset to the target by the distance at
    the start of the flight (crazyflie_env._get_obs). With one fixed start
    that is a constant, but with random starts the same position relative to
    the target would look different from flight to flight. metric_target=True
    divides by 1 m instead. The reward and the out-of-bounds rule still use
    the real starting distance.
    """

    def __init__(self, metric_target=False):
        super().__init__(scene_path=SCENE_PATH_1_DRONE, num_drones=1)
        self.metric_target = bool(metric_target)

    def _get_obs(self, data, info):
        if not self.metric_target:
            return super()._get_obs(data, info)
        scaled = dict(info, initial_distance=jnp.ones_like(info["initial_distance"]))
        obs = super()._get_obs(data, scaled)
        info["rng"] = scaled["rng"]  # keep the sensor-noise stream advancing
        return obs


def place_drone(drone, state, start, target):
    """Move a freshly reset drone to `start`, level and at rest, flying to `target`.

    The original environment always starts one drone at (0, 0, 0.2). This
    keeps everything else about its reset and only changes the two positions.
    """
    start = jnp.asarray(start, dtype=jnp.float32).reshape(1, 3)
    target = jnp.asarray(target, dtype=jnp.float32)

    qpos = state.data.qpos.at[0:3].set(start[0])
    data = mjx_env.init(drone.mjx_model, qpos=qpos, qvel=jnp.zeros_like(state.data.qvel))

    info = dict(state.info)
    info["target_pos"] = target
    info["prev_pos"] = start
    # Used by the reward, the out-of-bounds check and the observation scaling.
    info["initial_distance"] = jnp.linalg.norm(start - target, axis=1)
    obs = drone._get_obs(data, info)
    return state.replace(data=data, obs=obs, info=info)


class CrazyflieHoverEnv(gym.Env):
    """Expose MJX's functional state API to Gymnasium's stateful API.

    The action space uses the original drone units. TD-MPC2's Gymnasium
    training code wraps it with RescaleAction, so its policy sees [-1, 1].

    random_starts=True draws a new start from the Batch 2 training area for
    every flight; metric_target=True gives the target offset in metres (see
    HoverDrone). Both off is the Batch 1 task.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_length=500, random_starts=False, metric_target=False):
        super().__init__()
        self.drone = HoverDrone(metric_target=metric_target)
        self.random_starts = bool(random_starts)
        self.observation_space = gym.spaces.Box(
            low=np.full(31, -np.inf, dtype=np.float32),
            high=np.full(31, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self.episode_length = episode_length
        self._state = None
        self._steps = 0
        # Compiled once; running these op by op is slow on the GPU.
        self._reset_fn = jax.jit(lambda key: self.drone.reset(key, random_reset=False, 
            target=TARGET_POS))
        self._place_fn = jax.jit(lambda state, start, target: place_drone(self.drone, state, 
            start, target))
        self._step_fn = jax.jit(self.drone.step)

    def reset(self, *, seed=None, options=None):
        """Start a new flight.

        By default this is the training setup: start (0, 0, 0.2), target
        (0, 0, 1). Pass options={"start": (x, y, z), "target": (x, y, z)} to
        change either. The observation still contains the environment's noise.
        """
        super().reset(seed=seed)
        self._steps = 0
        key = jax.random.PRNGKey(int(self.np_random.integers(0, 2**31 - 1)))
        self._state = self._reset_fn(key)

        options = options or {}
        if self.random_starts and "start" not in options:
            options = {**options, "start": sample_training_start(self.np_random)}
        if "start" in options or "target" in options:
            start = options.get("start", self._state.info["prev_pos"][0])
            target = options.get("target", TARGET_POS)
            self._state = self._place_fn(self._state, jnp.asarray(start), jnp.asarray(target))
        return np.asarray(self._state.obs, dtype=np.float32), {}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError(f"Expected four drone controls, got {action.shape}")
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)
        self._state = self._step_fn(self._state, jnp.asarray(action))
        self._steps += 1

        # One GPU -> CPU copy per step instead of one per value.
        obs, reward, done, qpos = jax.device_get(
            (self._state.obs, self._state.reward, self._state.done, self._state.data.qpos)
        )
        terminated = bool(done)
        truncated = self._steps >= self.episode_length and not terminated
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            {"x": float(qpos[0]), "y": float(qpos[1]), "height": float(qpos[2])},
        )

    def close(self):
        self._state = None
```

## code/src/tdmpc2_training/crazyflie_hover_dr.py

H2: randomised mass, motors and wind. Last modified 2026-11-02.

```
"""Hover task with randomised physics in every training flight (dynamics randomisation).

The same task as CrazyflieHoverEnv (random starts, target offset in metres),
but each flight also draws, uniformly:
  mass and inertia   x 1.0 to 1.15   (an expansion deck, a payload)
  motor strength     x 0.9 to 1.0    (battery drop, worn propellers)
  steady wind        0 to 3 m/s from a random horizontal direction
These cover the in-range part of the disturbance tests (eval_disturbances.py);
4 m/s wind, 6 m/s gusts and command delays stay outside training. The agent
does not observe any of these values. Used by config_batch2_randomized_small.
"""
import jax
import jax.numpy as jnp
import numpy as np

from crazyflie_hover import CrazyflieHoverEnv
from eval_disturbances import WindyDrone

MASS_RANGE = (1.0, 1.15)
MOTOR_RANGE = (0.9, 1.0)
WIND_MAX = 3.0


class RandomizedHoverEnv(CrazyflieHoverEnv):
    """CrazyflieHoverEnv whose mass, motor strength and wind change every flight."""

    def __init__(self, episode_length=500, random_starts=True, metric_target=True):
        super().__init__(episode_length=episode_length, random_starts=random_starts, 
            metric_target=metric_target)
        self.drone = WindyDrone(metric_target=metric_target)
        # The parent compiled its step for the plain drone; reset and place read
        # self.drone when they are first compiled, so only step is rebuilt.
        self._step_fn = jax.jit(self.drone.step)
        self.dynamics = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        mass = self.np_random.uniform(*MASS_RANGE)
        motor = self.np_random.uniform(*MOTOR_RANGE)
        speed = self.np_random.uniform(0.0, WIND_MAX)
        heading = self.np_random.uniform(0.0, 2 * np.pi)
        wind = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0], dtype=np.float32)
        state_info = dict(self._state.info)
        state_info["body_mass"] = state_info["body_mass"] * mass
        state_info["body_inertia"] = state_info["body_inertia"] * mass
        state_info["actuator_gainprm"] = state_info["actuator_gainprm"] * motor
        state_info["wind"] = jnp.asarray(wind)
        self._state = self._state.replace(info=state_info)
        self.dynamics = {"mass": float(mass), "motor": float(motor), "wind": wind}
        return obs, info
```

## code/src/tdmpc2_training/crazyflie_hover_sharp.py

Reward pilot: extra reward near the target. Last modified 2026-11-02.

```
"""Hover task with a sharper reward for holding position close to the target (a reward-design 
    experiment).

The same task as CrazyflieHoverEnv with random starts and the target offset
in metres, plus one reward term concentrated within a few centimetres of the
target:

    bonus = 0.01 * exp(-d^2 / (2 * (0.05 m)^2)),   d = distance to the target

The lab's reward per step is 0.001 + 0.01 (1 - tanh d) + the distance
improvement, and -1 on a crash. Hovering at 5 cm instead of 15 cm earns only
0.001 more per step with it, about a tenth of the value of hovering, so the
precision of the small random-start agents may be weakly anchored. With the
bonus, hovering at 2 cm instead of 15 cm earns about 0.010 more per step.
Physics, observations, starts and termination are unchanged. The test flights
(hover_eval.py) measure distances and are flown with the lab's reward, so
their results compare directly with config_batch2_random_start_small.
Used by config_batch2_sharp_small.
"""
import jax
import numpy as np

from crazyflie_hover import CrazyflieHoverEnv

SHARP_WEIGHT = 0.01
SHARP_WIDTH_M = 0.05


def sharp_bonus(distance_m):
    return SHARP_WEIGHT * np.exp(-np.square(distance_m) / (2 * SHARP_WIDTH_M ** 2))


class SharpHoverEnv(CrazyflieHoverEnv):
    """CrazyflieHoverEnv whose reward has an extra term concentrated near the target."""

    def __init__(self, episode_length=500, random_starts=True, metric_target=True):
        super().__init__(episode_length=episode_length, random_starts=random_starts, 
            metric_target=metric_target)
        self._target = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._target = np.asarray(jax.device_get(self._state.info["target_pos"]), 
            dtype=np.float64).reshape(-1)[:3]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        distance = np.linalg.norm(np.array([info["x"], info["y"], info["height"]]) - self._target)
        return obs, reward + float(sharp_bonus(distance)), terminated, truncated, info
```

## code/src/tdmpc2_training/train_hover.py

Training entry point. Last modified 2026-11-02.

```
"""Start the vendored TD-MPC2 trainer on the one-drone hover task."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import gymnasium as gym
from crazyflie_hover import CrazyflieHoverEnv

gym.register("CrazyflieHover-v0", entry_point=CrazyflieHoverEnv)
# Randomised physics in every flight (crazyflie_hover_dr.py); imported only when used.
gym.register("CrazyflieHoverRandomized-v0", entry_point="crazyflie_hover_dr:RandomizedHoverEnv")
# An extra reward term near the target (crazyflie_hover_sharp.py); imported only when used.
gym.register("CrazyflieHoverSharp-v0", entry_point="crazyflie_hover_sharp:SharpHoverEnv")

from tdmpc2_jax.train import train

if __name__ == "__main__":
    if not any(arg.startswith("--config-name") for arg in sys.argv[1:]):
        sys.argv.append("--config-name=config_hover")
    train()
```

## code/src/tdmpc2_training/tdmpc2_jax/config_batch2_random_start_small.yaml

Config D. Last modified 2026-11-01.

```
# Batch 2, experiment D (pilot): the small TD-MPC2 of Batch 1 (config_hover:
# latent and hidden width 128, 128 plans x 3 rounds, batch 64) trained exactly
# like R1 (config_batch2_random_start): a new random start every training
# flight, target offset in metres, 1M steps, the same test starts.
# Question: does the smaller, faster model (1.4 ms per decision on a GPU slice,
# 11 ms on a CPU) learn and generalise as well as the published size?
#
#   train_hover.py --config-name=config_batch2_random_start_small seed=0
defaults:
  - config_hover
  - _self_

max_steps: 1_000_000
buffer_size: 1_000_000
save_interval_steps: 50_000
snapshot_interval_steps: 100_000
eval_interval_steps: 25_000
curve_set: batch2

env:
  random_starts: True
  metric_target: True
```

## code/src/tdmpc2_training/tdmpc2_jax/config_batch2_random_start.yaml

Config R1. Last modified 2026-11-01.

```
# Batch 2, experiment R1: TD-MPC2 with the published settings
# (config_hover_standard) learning to fly to the target from a new random
# start in every training flight.
#
#   train_hover.py --config-name=config_batch2_random_start seed=0
#
# Starts are drawn from the Batch 2 training area (crazyflie_hover.py:
# x, y in [-0.75, 0.75] m, height 0.2-1.5 m, at least 0.3 m from the target,
# held-out block x, y in [0.25, 0.75] m excluded). The target stays at
# (0, 0, 1). The target offset in the observation is in metres. Everything
# else (reward, physics, sensor noise, 10 s flights) is as in Batch 1.
defaults:
  - config_hover_standard
  - _self_

max_steps: 1_000_000
buffer_size: 1_000_000          # official TD-MPC2 default: keeps every transition
save_interval_steps: 50_000     # resume point if a cluster job is interrupted
snapshot_interval_steps: 100_000
eval_interval_steps: 25_000
curve_set: batch2

env:
  random_starts: True
  metric_target: True
```

## code/src/tdmpc2_training/tdmpc2_jax/config_batch2_fixed_start.yaml

Config C1. Last modified 2026-11-01.

```
# Batch 2, control C1: identical to config_batch2_random_start except that
# every training flight starts from the Batch 1 start (0, 0, 0.2). Comparing
# the two isolates the effect of training from random starts; the old 1M-step
# runs (batch1_standard_1M) also differ in the observation scaling.
#
#   train_hover.py --config-name=config_batch2_fixed_start seed=0
defaults:
  - config_hover_standard
  - _self_

max_steps: 1_000_000
buffer_size: 1_000_000
save_interval_steps: 50_000
snapshot_interval_steps: 100_000
eval_interval_steps: 25_000
curve_set: batch2

env:
  random_starts: False
  metric_target: True
```

## code/src/tdmpc2_training/tdmpc2_jax/config_batch2_randomized_small.yaml

Config H2. Last modified 2026-11-02.

```
# Batch 2, experiment H2 (pilot): the small TD-MPC2 trained like the small
# random-start runs (config_batch2_random_start_small), but every training
# flight also has randomised physics: mass and inertia x1.0-1.15, motor
# strength x0.9-1.0, steady wind 0-3 m/s from a random direction
# (crazyflie_hover_dr.py). Tests use nominal physics and the disturbance tests.
# Question: does planning with a world model trained on varied physics help
# under disturbances inside that range, and how far beyond it?
#
#   train_hover.py --config-name=config_batch2_randomized_small seed=0
defaults:
  - config_hover
  - _self_

max_steps: 1_000_000
buffer_size: 1_000_000
save_interval_steps: 50_000
snapshot_interval_steps: 100_000
eval_interval_steps: 25_000
curve_set: batch2

env:
  env_id: CrazyflieHoverRandomized-v0
  random_starts: True
  metric_target: True
```

## code/src/tdmpc2_training/tdmpc2_jax/config_batch2_sharp_small.yaml

Config reward pilot. Last modified 2026-11-02.

```
# Batch 2 follow-up (pilot): the small TD-MPC2 trained exactly like experiment
# D (config_batch2_random_start_small), but with an extra reward term
# concentrated within about 5 cm of the target (crazyflie_hover_sharp.py).
# Question: is the drift of the small agents' hover precision between
# checkpoints (3-15 cm) caused by a reward that barely distinguishes 5 cm from
# 15 cm? Compare the snapshot-to-snapshot drift and the final precision with D.
#
#   train_hover.py --config-name=config_batch2_sharp_small seed=0
defaults:
  - config_hover
  - _self_

max_steps: 1_000_000
buffer_size: 1_000_000
save_interval_steps: 50_000
snapshot_interval_steps: 100_000
eval_interval_steps: 25_000
curve_set: batch2

env:
  env_id: CrazyflieHoverSharp-v0
  random_starts: True
  metric_target: True
```

## code/src/tdmpc2_training/tdmpc2_jax/config_hover.yaml

Config Batch 1 small. Last modified 2026-11-01.

```
hydra:
  job:
    name: crazyflie_hover_s${seed}
  run:
    dir: outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}

seed: 0
max_steps: 20_000
buffer_size: 50_000
seed_steps: 500
seed_updates: 100
save_interval_steps: 2_000
log_interval_steps: 500

# Test flights from the training start every eval_interval_steps, written to
# learning_curve.csv. Full tests from several starts: eval_hover.py.
eval_interval_steps: 5_000
eval_flights: 3
eval_flight_steps: 500

# Batch 2 additions, all off here so Batch 1 runs are unchanged.
# curve_set 'batch2': learning-curve flights from hover_eval.CURVE_STARTS
# (21 starts from every test group) instead of eval_flights x 'home'.
# snapshot_interval_steps: keep the agent every N steps in snapshots/.
curve_set: home
snapshot_interval_steps: null

env:
  backend: gymnasium
  env_id: CrazyflieHover-v0
  num_envs: 1
  utd_ratio: 1.0
  asynchronous: False
  random_starts: False   # True: a new random start for every training flight
  metric_target: False   # True: target offset in metres (crazyflie_hover.HoverDrone)

encoder:
  encoder_dim: 128
  num_encoder_layers: 2
  learning_rate: 1e-4
  tabulate: False

world_model:
  latent_dim: 128
  value_dropout: 0.01
  num_value_nets: 5
  num_bins: 101
  symlog_min: -10
  symlog_max: 10
  simnorm_dim: 8
  learning_rate: 3e-4
  predict_continues: False
  dtype: float32
  max_grad_norm: 20
  tabulate: False

tdmpc2:
  horizon: 3
  mppi_iterations: 3
  population_size: 128
  policy_prior_samples: 16
  num_elites: 16
  min_plan_std: 0.05
  max_plan_std: 2
  temperature: 1.0
  batch_size: 64
  discount: 0.99
  rho: 0.5
  consistency_loss_scale: 20
  reward_loss_scale: 0.1
  value_loss_scale: 0.1
  continue_loss_scale: 0.0
  entropy_coef: 1e-4
  tau: 0.01
```

## code/src/tdmpc2_training/tdmpc2_jax/config_hover_standard.yaml

Config Batch 1 published. Last modified 2026-10-12.

```
# Same task, training budget and test flights as config_hover.yaml, but with
# the network sizes and planning budget of the published TD-MPC2 defaults
# (github.com/nicklashansen/tdmpc2, config.yaml). Only the values below differ.
#
#   train_hover.py --config-name=config_hover_standard
defaults:
  - config_hover
  - _self_

seed_steps: 2_500     # official: max(1000, 5 x episode length)
seed_updates: 2_500   # official: one update per seed step

encoder:
  encoder_dim: 256
  learning_rate: 9e-5   # official: lr 3e-4 x enc_lr_scale 0.3

world_model:
  latent_dim: 512     # this implementation also uses it as the hidden width

tdmpc2:
  mppi_iterations: 6
  population_size: 512
  policy_prior_samples: 24
  num_elites: 64
  temperature: 0.5
  batch_size: 256
```

## code/src/tdmpc2_training/tdmpc2_jax/config.yaml

Upstream default config. Last modified 2026-09-28.

```
hydra:
  job:
    name: ${env.env_id}_s${seed}
  run:
    dir: outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}

seed: 0
max_steps: 2_000_000
buffer_size: 2_000_000
save_interval_steps: 100000
log_interval_steps: 10000

env:
  backend: dmc
  env_id: dog-run
  num_envs: 4
  utd_ratio: 1.0
  asynchronous: True

  dmc: { obs_type: state }

encoder:
  encoder_dim: 256
  num_encoder_layers: 2
  learning_rate: 1e-4
  tabulate: False

world_model:
  latent_dim: 512
  value_dropout: 0.01
  num_value_nets: 5
  num_bins: 101
  symlog_min: -10
  symlog_max: 10
  simnorm_dim: 8
  learning_rate: 3e-4
  predict_continues: False
  dtype: float32
  max_grad_norm: 20
  tabulate: False

tdmpc2:
  # Planning
  horizon: 3
  mppi_iterations: 6
  population_size: 512
  policy_prior_samples: 24
  num_elites: 64
  min_plan_std: 0.05
  max_plan_std: 2
  temperature: 1.0
  # Optimization
  batch_size: 256
  discount: 0.99
  rho: 0.5
  consistency_loss_scale: 20
  reward_loss_scale: 0.1
  value_loss_scale: 0.1
  continue_loss_scale: 0.0
  entropy_coef: 1e-4
  tau: 0.01
```

## code/src/tdmpc2_training/hover_eval.py

Test flights, test starts and metrics. Last modified 2026-11-01.

```
"""Test flights for a trained TD-MPC2 hover agent.

The agent plans exactly as it does in training (MPC, deterministic=True,
train=False) but never learns here. All requested flights fly side by side on
the GPU, and each whole flight is one compiled loop, so the only CPU<->GPU copy
is the finished trajectory. Testing 18 flights costs about the same as one.

Metrics are defined here, before looking at any multi-seed results:
  late error      mean distance to the target over the last 100 steps flown (2 s)
  time in 10 cm   fraction of the steps after the first 2 s spent within 0.10 m
                  of the target; steps after a crash count as outside

Test starts. TEST_STARTS are the six Batch 1 starts. TEST_SETS adds the Batch 2
groups, drawn once from a fixed seed so that every model, in every run, is
tested from exactly the same places:
  batch1      the six Batch 1 starts (continuity with the Batch 1 results)
  train_area  50 starts from the same distribution as Batch 2 training
  held_out    30 starts in the held-out block (inside the area, never trained on)
  outside     30 starts outside the training area (0.25-0.75 m beyond its edge)
"""
import csv
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from constants import TARGET_POS
from crazyflie_hover import (ACTION_HIGH, ACTION_LOW, HELD_OUT_XY_HIGH, HELD_OUT_XY_LOW,
                             START_AREA_HIGH, START_AREA_LOW, HoverDrone, place_drone,
                             sample_training_start)

CONTROL_DT = 0.02         # seconds per control step (50 Hz)
LATE_WINDOW_STEPS = 100   # last 2 s of a flight
SETTLE_STEPS = 100        # first 2 s are take-off, not hovering
TOLERANCE_M = 0.10

# Fixed before the seed runs. "home" is the start used in training; the others
# are 0.5 m to each side at the same height, and one start above the target.
TEST_STARTS = {
    "home": (0.0, 0.0, 0.2),
    "plus_x": (0.5, 0.0, 0.2),
    "minus_x": (-0.5, 0.0, 0.2),
    "plus_y": (0.0, 0.5, 0.2),
    "minus_y": (0.0, -0.5, 0.2),
    "above": (0.0, 0.0, 1.5),
}

TEST_SET_SEED = 2026          # never change: it defines the Batch 2 test starts
OUTSIDE_MIN, OUTSIDE_MAX = 1.0, 1.5   # max(|x|, |y|) range of the outside starts


def _make_test_sets():
    rng = np.random.default_rng(TEST_SET_SEED)

    def held_out():
        low = np.array([*HELD_OUT_XY_LOW, START_AREA_LOW[2]])
        high = np.array([*HELD_OUT_XY_HIGH, START_AREA_HIGH[2]])
        return rng.uniform(low, high)

    def outside():
        while True:
            xyz = rng.uniform([-OUTSIDE_MAX, -OUTSIDE_MAX, START_AREA_LOW[2]],
                              [OUTSIDE_MAX, OUTSIDE_MAX, START_AREA_HIGH[2]])
            if np.max(np.abs(xyz[:2])) >= OUTSIDE_MIN:
                return xyz

    def labelled(name, draw, n):
        return [(f"{name}/{i:02d}", tuple(float(v) for v in draw())) for i in range(n)]

    return {
        "batch1": list(TEST_STARTS.items()),
        "train_area": labelled("train_area", lambda: sample_training_start(rng), 50),
        "held_out": labelled("held_out", held_out, 30),
        "outside": labelled("outside", outside, 30),
    }


TEST_SETS = _make_test_sets()

# Learning-curve flights during Batch 2 training (curve_set: batch2): the
# training start plus the first few starts of each group, 21 flights in all.
CURVE_STARTS = ([("home", TEST_STARTS["home"])] + TEST_SETS["train_area"][:8]
                + TEST_SETS["held_out"][:6] + TEST_SETS["outside"][:6])


def test_set_of(label):
    """'held_out/03' -> 'held_out'; Batch 1 names like 'home' -> 'batch1'."""
    return label.split("/")[0] if "/" in label else "batch1"


def resolve_starts(starts):
    """Names from TEST_STARTS, or (label, (x, y, z)) pairs -> labels, (n, 3) array."""
    labels, coords = [], []
    for entry in starts:
        if isinstance(entry, str):
            labels.append(entry)
            coords.append(TEST_STARTS[entry])
        else:
            label, xyz = entry
            labels.append(label)
            coords.append(xyz)
    return labels, np.asarray(coords, dtype=np.float32)


@dataclass
class Flight:
    start_name: str
    start: np.ndarray       # (3,)
    target: np.ndarray      # (3,)
    position: np.ndarray    # (steps, 3) true position after each control step
    command: np.ndarray     # (steps, 4) thrust, roll, pitch, yaw sent to the drone
    reward: np.ndarray      # (steps,)
    steps_flown: int
    crashed: bool

    def errors(self):
        """3D, horizontal and vertical distance to the target for the steps flown."""
        offset = self.position[:self.steps_flown] - self.target
        return (np.linalg.norm(offset, axis=1),
                np.linalg.norm(offset[:, :2], axis=1),
                np.abs(offset[:, 2]))

    def summary(self):
        error_3d, error_xy, error_z = self.errors()
        late = slice(max(0, self.steps_flown - LATE_WINDOW_STEPS), self.steps_flown)
        steps_after_settling = len(self.reward) - SETTLE_STEPS
        inside = np.sum(error_3d[SETTLE_STEPS:] < TOLERANCE_M)
        return {
            "start": self.start_name,
            "steps_flown": self.steps_flown,
            "crashed": self.crashed,
            "return": float(self.reward[:self.steps_flown].sum()),
            "final_error_m": float(error_3d[-1]),
            "late_error_m": float(error_3d[late].mean()),
            "late_error_xy_m": float(error_xy[late].mean()),
            "late_error_z_m": float(error_z[late].mean()),
            "late_height_m": float(self.position[late, 2].mean()),
            "time_in_10cm": float(inside / steps_after_settling) if steps_after_settling > 0 else 
                float("nan"),
            # Added for Batch 2; Batch 1 files simply lack these columns.
            "test_set": test_set_of(self.start_name),
            "start_x": float(self.start[0]),
            "start_y": float(self.start[1]),
            "start_z": float(self.start[2]),
        }


class FlightTester:
    """Flies a batch of test flights with a given agent.

    mpc=False flies TD-MPC2's learned policy network alone, without planning.
    metric_target must match the observation the agent was trained with.
    """

    def __init__(self, flight_steps=500, mpc=True, metric_target=False):
        self.drone = HoverDrone(metric_target=metric_target)
        self.flight_steps = int(flight_steps)
        self.mpc = bool(mpc)
        self._fly = jax.jit(self._fly_batch)

    def fly(self, agent, starts, seed, targets=None):
        """Fly one flight per entry in `starts`: names from TEST_STARTS, or
        (label, (x, y, z)) pairs such as the entries of TEST_SETS.

        The same `seed` gives the same sensor noise and planner randomness, so
        two agents tested with one seed face identical conditions.
        """
        start_names, starts = resolve_starts(starts)
        if targets is None:
            targets = np.tile(np.asarray(TARGET_POS, dtype=np.float32), (len(starts), 1))
        reset_key, plan_key = jax.random.split(jax.random.PRNGKey(seed))
        reset_keys = jax.random.split(reset_key, len(starts))

        trajectory = jax.device_get(self._fly(agent, reset_keys, starts, targets, plan_key))

        flights = []
        for i, name in enumerate(start_names):
            alive = trajectory["alive"][:, i]
            steps_flown = int(alive.sum())
            flights.append(Flight(
                start_name=name,
                start=starts[i],
                target=np.asarray(targets[i]),
                position=trajectory["position"][:, i],
                command=trajectory["command"][:, i],
                reward=trajectory["reward"][:, i],
                steps_flown=steps_flown,
                crashed=bool(trajectory["done"][:steps_flown, i].any()),
            ))
        return flights

    def _fly_batch(self, agent, reset_keys, starts, targets, plan_key):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            return place_drone(self.drone, state, start, target)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)

        # An empty previous plan, same as prev_plan=None at the start of a flight.
        n, horizon, action_dim = len(starts), agent.horizon, agent.model.action_dim
        plan = (jnp.zeros((n, horizon, action_dim)),
                jnp.full((n, horizon, action_dim), agent.max_plan_std))

        def control_step(carry, key):
            states, plan, alive = carry
            if self.mpc:
                action, plan = agent.act(states.obs, prev_plan=plan,
                                         deterministic=True, train=False, key=key)
            else:
                action, _ = agent.act(states.obs, mpc=False,
                                      deterministic=True, train=False, key=key)
            # Same mapping as Gymnasium's RescaleAction: [-1, 1] -> drone units.
            command = jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)
            states = step_all(states, command)
            done = states.done > 0
            record = {"position": states.data.qpos[:, :3], "command": command,
                      "reward": states.reward, "done": done, "alive": alive}
            # A crashed drone keeps being simulated, but its steps are masked out.
            return (states, plan, alive & ~done), record

        keys = jax.random.split(plan_key, self.flight_steps)
        carry = (states, plan, jnp.ones(n, dtype=bool))
        _, trajectory = jax.lax.scan(control_step, carry, keys)
        return trajectory


def write_summary(path, flights, **extra_columns):
    """One row per flight. Extra columns (e.g. seed=3) are written first."""
    rows = [{**extra_columns, "flight": i, **f.summary()} for i, f in enumerate(flights)]
    _write_rows(path, rows, append=False)
    return rows


def append_learning_curve(path, flights, **extra_columns):
    """Add one row per flight to a learning curve file, creating it if needed."""
    rows = [{**extra_columns, "flight": i, **f.summary()} for i, f in enumerate(flights)]
    _write_rows(path, rows, append=True)
    return rows


def write_trajectories(path, flights):
    """Step-by-step position, error and controls for every flight."""
    rows = []
    for i, f in enumerate(flights):
        error_3d, _, _ = f.errors()
        for t in range(f.steps_flown):
            x, y, z = f.position[t]
            thrust, roll, pitch, yaw = f.command[t]
            rows.append({
                "flight": i, "start": f.start_name, "step": t,
                "time_s": round((t + 1) * CONTROL_DT, 4),
                "x_m": x, "y_m": y, "z_m": z, "error_m": error_3d[t],
                "reward": f.reward[t],
                "thrust": thrust, "roll": roll, "pitch": pitch, "yaw": yaw,
            })
    _write_rows(path, rows, append=False)


def _write_rows(path, rows, append):
    if not rows:
        return
    new_file = not append or not os.path.exists(path)
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _tidy(v) for k, v in row.items()})


def _tidy(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, np.bool_):
        return bool(value)
    return value
```

## code/src/tdmpc2_training/eval_hover.py

Testing a saved agent. Last modified 2026-11-01.

```
"""Load a saved TD-MPC2 hover agent from disk and test it. Nothing is trained.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_hover.py outputs/batch1_fixed_start/seed_00
    .venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --starts home --flights 3

The agent is rebuilt from the run's own .hydra/config.yaml and its newest
checkpoint is restored, so this also checks that a saved model can be reused
in a fresh process. Results go to <run_dir>/evaluation/:

    summary.csv       one row per flight (errors, crash, return, start position)
    trajectories.csv  every step of every flight (position, error, controls)

Which starts are flown (hover_eval.TEST_SETS): Batch 1 runs default to the six
Batch 1 starts (3 flights each); Batch 2 runs default to all groups. Choose
explicitly with --test-sets, e.g. to test an old run on the Batch 2 starts:

    eval_hover.py <run_dir> --test-sets all --out-name evaluation_batch2_tests

The same trained agent can also be flown with a different decision rule, to
see how much accuracy planning buys (results go to their own folder):

    eval_hover.py <run_dir> --no-mpc       # policy network only -> evaluation_policy_only/
    eval_hover.py <run_dir> --plan 128x3   # smaller planner -> evaluation_plan128x3/

--snapshot 300000 tests the agent kept at 300k steps (Batch 2 runs keep one
every 100k steps in snapshots/). Existing results are never overwritten
unless --overwrite is given.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import numpy as np
import orbax.checkpoint as ocp
from omegaconf import OmegaConf

from hover_eval import TEST_SETS, TEST_STARTS, FlightTester, write_summary, write_trajectories
from tdmpc2_jax.build import build_agent

OBSERVATION_SIZE = 31
ACTION_SIZE = 4


def load_agent(run_dir, step=None, snapshot=None):
    """Return (agent, step, config) for a training run.

    By default the newest checkpoint/ is loaded; snapshot=N loads the agent
    kept at N steps in snapshots/ instead.
    """
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model_key, encoder_key = jax.random.split(jax.random.PRNGKey(0))
    template = build_agent(
        cfg,
        observation=np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
        action_dim=ACTION_SIZE,
        model_key=model_key,
        encoder_key=encoder_key,
    )

    if snapshot is None:
        directory, items = run_dir / "checkpoint", ("agent", "global_step", "buffer_state")
    else:
        directory, items, step = run_dir / "snapshots", ("agent",), snapshot
    options = ocp.CheckpointManagerOptions(read_only=True)
    with ocp.CheckpointManager(directory.resolve(), options=options, item_names=items) as manager:
        step = manager.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"No checkpoint in {directory}")
        # Only the agent is needed; the replay buffer is left on disk.
        restored = manager.restore(
            step, args=ocp.args.Composite(agent=ocp.args.StandardRestore(template))
        )
    return restored.agent, step, cfg


def is_batch2_run(cfg):
    return cfg.get("curve_set", "home") == "batch2"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="training output folder")
    parser.add_argument("--test-sets", default="auto",
                        help=f"'auto', 'all' or comma-separated from: {', '.join(TEST_SETS)}")
    parser.add_argument("--starts", default="all",
                        help=f"Batch 1 starts to fly: 'all' or comma-separated from: {', 
                            '.join(TEST_STARTS)}")
    parser.add_argument("--flights", type=int, default=3, help="flights per Batch 1 start")
    parser.add_argument("--flight-steps", type=int, default=500,
                        help="control steps per flight (500 = 10 simulated seconds)")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: 
        newest)")
    parser.add_argument("--snapshot", type=int, default=None,
                        help="test the agent kept at this many steps in snapshots/")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="noise seed for the test flights (default: training seed + 2000)")
    parser.add_argument("--no-mpc", action="store_true",
                        help="fly the learned policy network alone, without planning")
    parser.add_argument("--plan", default=None,
                        help="planning budget PLANSxROUNDS to use instead of the trained one, e.g.
                             128x3")
    parser.add_argument("--out-name", default=None,
                        help="output folder name inside run_dir (default: evaluation[_variant])")
    parser.add_argument("--overwrite", action="store_true", help="replace existing results")
    args = parser.parse_args()

    run_dir = args.run_dir
    if args.no_mpc and args.plan:
        parser.error("--no-mpc and --plan exclude each other")
    if args.step is not None and args.snapshot is not None:
        parser.error("--step and --snapshot exclude each other")
    batch1_names = list(TEST_STARTS) if args.starts == "all" else args.starts.split(",")
    unknown = [s for s in batch1_names if s not in TEST_STARTS]
    if unknown:
        parser.error(f"unknown start(s): {unknown}")

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    test_sets = args.test_sets
    if test_sets == "auto":
        test_sets = "all" if is_batch2_run(cfg) else "batch1"
    test_sets = list(TEST_SETS) if test_sets == "all" else test_sets.split(",")
    unknown = [s for s in test_sets if s not in TEST_SETS]
    if unknown:
        parser.error(f"unknown test set(s): {unknown}")

    # Results of a changed decision rule go to their own folder, so the
    # standard evaluation/ is never overwritten by a variant.
    variant = ""
    if args.no_mpc:
        variant = "_policy_only"
    elif args.plan:
        plans, rounds = (int(n) for n in args.plan.lower().split("x"))
        variant = f"_plan{plans}x{rounds}"
    if args.snapshot is not None:
        variant = f"_snapshot{args.snapshot}{variant}"
    out_dir = run_dir / (args.out_name or f"evaluation{variant}")
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")

    agent, step, cfg = load_agent(run_dir, args.step, args.snapshot)
    print(f"Loaded {run_dir} at {step} training steps (seed {cfg.seed}).")
    if args.plan:
        # Elites and policy-prior plans scale with the number of plans, as in
        # the two configs (128 plans: 16 elites; 512 plans: 64 elites).
        agent = agent.replace(population_size=plans, mppi_iterations=rounds,
                              num_elites=max(1, plans // 8),
                              policy_prior_samples=min(agent.policy_prior_samples, plans // 4))
        print(f"Planning with {plans} plans x {rounds} rounds instead of the trained budget.")
    decision = ("policy_only" if args.no_mpc
                else f"planner_{agent.population_size}x{agent.mppi_iterations}")

    starts = []
    for name in test_sets:
        if name == "batch1":
            starts += [(n, TEST_STARTS[n]) for n in batch1_names for _ in range(args.flights)]
        else:
            starts += TEST_SETS[name]

    # Training-time test flights use seed + 1000; use different noise here.
    test_seed = cfg.seed + 2000 if args.test_seed is None else args.test_seed
    metric_target = bool(cfg.env.get("metric_target", False))
    tester = FlightTester(flight_steps=args.flight_steps, mpc=not args.no_mpc,
                          metric_target=metric_target)
    flights = tester.fly(agent, starts, seed=test_seed)

    os.makedirs(out_dir, exist_ok=True)
    rows = write_summary(out_dir / "summary.csv", flights, seed=cfg.seed,
                         checkpoint_step=step, decision=decision)
    write_trajectories(out_dir / "trajectories.csv", flights)

    if len(rows) <= 30:
        print(f"\n{'start':<14} {'flight':>6} {'steps':>6} {'crash':>6} {'late err':>9} "
              f"{'xy err':>7} {'z err':>7} {'in 10cm':>8} {'return':>8}")
        for r in rows:
            print(f"{r['start']:<14} {r['flight']:>6} {r['steps_flown']:>6} {str(r['crashed']):>6}
                 "
                  f"{r['late_error_m']:>8.3f}m {r['late_error_xy_m']:>6.3f}m 
                      {r['late_error_z_m']:>6.3f}m "
                  f"{r['time_in_10cm']:>7.0%} {r['return']:>8.3f}")
    print(f"\n{'test set':<11} {'flights':>7} {'crashes':>7} {'late err mean':>13} {'median':>7} 
        {'in 10cm':>8}")
    for name in test_sets:
        mine = [r for r in rows if r["test_set"] == name]
        late = np.array([r["late_error_m"] for r in mine])
        inside = np.mean([r["time_in_10cm"] for r in mine])
        print(f"{name:<11} {len(mine):>7} {sum(r['crashed'] for r in mine):>7} "
              f"{late.mean():>12.3f}m {np.median(late):>6.3f}m {inside:>7.0%}")
    print(f"\nDecision rule: {decision}, obs target offset in "
          f"{'metres' if metric_target else 'start distances'}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
```

## code/src/tdmpc2_training/eval_disturbances.py

Disturbance tests (wind, gusts, mass, motors, delay). Last modified 2026-11-01.

```
"""Test a trained TD-MPC2 hover agent under disturbances it never trained with.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_disturbances.py <run_dir>
    .venv/bin/python code/src/tdmpc2_training/eval_disturbances.py <run_dir> --no-mpc

Every disturbance is flown at several strengths from the same test starts
(hover_eval.TEST_SETS, default the 50 train_area starts) with the same sensor
noise, so the only thing that changes between conditions is the disturbance.
The undisturbed flights are flown too, as the paired baseline. Nothing is
trained. Writes <run_dir>/disturbances[_policy_only]/summary.csv, one row per
flight, with the disturbance and its strength as the first columns.

Disturbances (none of them happens in training):
  wind    steady wind along +x, in m/s, acting through the simulator's air
          model (air density 1.225 kg/m3 in cf2.xml). Indoors: a fan or an
          air-conditioning draft is about 1-3 m/s.
  gust    the same wind, but only for 1 s, starting 3 s into the flight
  mass    extra weight (an expansion deck, a marker, a small camera): body
          mass and inertia times the strength. The real drone weighs 27 g.
  motor   weaker motors (battery voltage dropping, worn propellers): thrust
          and moments times the strength
  delay   every command reaches the drone 1 or 2 control steps late (20 or
          40 ms), like radio lag on a real drone; the first commands are a
          plain hover
The simulated drone needs 76% of its maximum thrust to hover, so extra mass
and weaker motors are kept small.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np

from constants import BASE_HOVER_THRUST, TARGET_POS
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, HoverDrone, place_drone
from eval_hover import load_agent
from hover_eval import CONTROL_DT, TEST_SETS, Flight, FlightTester, _write_rows, resolve_starts

# Strengths per disturbance. The undisturbed baseline is flown separately.
DISTURBANCES = {
    "wind": [1.0, 2.0, 3.0, 4.0],   # m/s
    "gust": [2.0, 4.0, 6.0],        # m/s for 1 s
    "mass": [1.05, 1.10, 1.15],     # x mass and inertia
    "motor": [0.95, 0.90],          # x thrust and moments
    "delay": [1, 2],                # control steps late
}
GUST_START_S, GUST_LENGTH_S = 3.0, 1.0


class WindyDrone(HoverDrone):
    """HoverDrone flying in the wind given by state.info['wind'] (m/s, world axes)."""

    def step(self, state, action):
        # The lab's step builds its model from self._mjx_model and keeps its
        # wind, so swap in this flight's wind while the step is traced.
        base = self._mjx_model
        self._mjx_model = base.replace(opt=base.opt.replace(wind=state.info["wind"]))
        try:
            return super().step(state, action)
        finally:
            self._mjx_model = base


class DisturbedFlightTester(FlightTester):
    """FlightTester whose flights can have wind, extra mass, weaker motors and a command delay.

    Wind, mass and motor strength are inputs of the compiled flight, so one
    compilation serves all their strengths; each delay length compiles once.
    """

    def __init__(self, flight_steps=500, mpc=True, metric_target=False, delay_steps=0):
        super().__init__(flight_steps=flight_steps, mpc=mpc, metric_target=metric_target)
        self.drone = WindyDrone(metric_target=metric_target)
        self.delay_steps = int(delay_steps)
        self._fly = jax.jit(self._fly_disturbed)

    def fly(self, agent, starts, seed, wind=None, mass_factor=1.0, motor_factor=1.0):
        """Like FlightTester.fly; wind is (flight_steps, 3) m/s, one row per control step."""
        start_names, starts = resolve_starts(starts)
        targets = np.tile(np.asarray(TARGET_POS, dtype=np.float32), (len(starts), 1))
        if wind is None:
            wind = np.zeros((self.flight_steps, 3), dtype=np.float32)
        reset_key, plan_key = jax.random.split(jax.random.PRNGKey(seed))
        reset_keys = jax.random.split(reset_key, len(starts))
        trajectory = jax.device_get(self._fly(
            agent, reset_keys, starts, targets, plan_key,
            jnp.asarray(wind, dtype=jnp.float32), jnp.float32(mass_factor), 
                jnp.float32(motor_factor)))

        flights = []
        for i, name in enumerate(start_names):
            alive = trajectory["alive"][:, i]
            steps_flown = int(alive.sum())
            flights.append(Flight(
                start_name=name, start=starts[i], target=np.asarray(targets[i]),
                position=trajectory["position"][:, i], command=trajectory["command"][:, i],
                reward=trajectory["reward"][:, i], steps_flown=steps_flown,
                crashed=bool(trajectory["done"][:steps_flown, i].any()),
            ))
        return flights

    def _fly_disturbed(self, agent, reset_keys, starts, targets, plan_key, wind, mass_factor, 
        motor_factor):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            state = place_drone(self.drone, state, start, target)
            info = dict(state.info)
            info["body_mass"] = info["body_mass"] * mass_factor
            info["body_inertia"] = info["body_inertia"] * mass_factor
            info["actuator_gainprm"] = info["actuator_gainprm"] * motor_factor
            info["wind"] = wind[0]
            return state.replace(info=info)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)
        n, horizon, action_dim = len(starts), agent.horizon, agent.model.action_dim
        plan = (jnp.zeros((n, horizon, action_dim)),
                jnp.full((n, horizon, action_dim), agent.max_plan_std))
        # Commands sent but not yet received (delay); a plain hover to begin with.
        hover = jnp.array([BASE_HOVER_THRUST, 0.0, 0.0, 0.0])
        queue = jnp.tile(hover, (n, max(self.delay_steps, 1), 1))

        def control_step(carry, inputs):
            states, plan, alive, queue = carry
            key, wind_now = inputs
            if self.mpc:
                action, plan = agent.act(states.obs, prev_plan=plan,
                                         deterministic=True, train=False, key=key)
            else:
                action, _ = agent.act(states.obs, mpc=False,
                                      deterministic=True, train=False, key=key)
            command = jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)
            if self.delay_steps:
                received = queue[:, 0]
                queue = jnp.concatenate([queue[:, 1:], command[:, None]], axis=1)
            else:
                received = command
            info = dict(states.info)
            info["wind"] = jnp.broadcast_to(wind_now, (n, 3))
            states = step_all(states.replace(info=info), received)
            done = states.done > 0
            record = {"position": states.data.qpos[:, :3], "command": received,
                      "reward": states.reward, "done": done, "alive": alive}
            return (states, plan, alive & ~done, queue), record

        keys = jax.random.split(plan_key, self.flight_steps)
        carry = (states, plan, jnp.ones(n, dtype=bool), queue)
        _, trajectory = jax.lax.scan(control_step, carry, (keys, wind))
        return trajectory


def end_cause(flight):
    """completed, ground (fell below 5 cm) or out_of_bounds (strayed 2 m beyond its start 
        distance)."""
    if not flight.crashed:
        return "completed"
    height = flight.position[flight.steps_flown - 1, 2]
    return "ground" if height < 0.05 else "out_of_bounds"


def wind_profile(kind, strength, steps):
    wind = np.zeros((steps, 3), dtype=np.float32)
    if kind == "wind":
        wind[:, 0] = strength
    elif kind == "gust":
        t = np.arange(steps) * CONTROL_DT
        wind[(t >= GUST_START_S) & (t < GUST_START_S + GUST_LENGTH_S), 0] = strength
    return wind


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="training output folder")
    parser.add_argument("--test-sets", default="train_area",
                        help=f"comma-separated from: {', '.join(TEST_SETS)}")
    parser.add_argument("--only", default="all", help=f"'all' or comma-separated from: {', 
        '.join(DISTURBANCES)}")
    parser.add_argument("--no-mpc", action="store_true", help="fly the policy network alone, 
        without planning")
    parser.add_argument("--snapshot", type=int, default=None, help="test the agent kept at this 
        many steps")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="noise seed (default: training seed + 3000, the same for every 
                            condition)")
    parser.add_argument("--flight-steps", type=int, default=500)
    parser.add_argument("--out-name", default=None, help="output folder (default: 
        disturbances[_policy_only])")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    kinds = list(DISTURBANCES) if args.only == "all" else args.only.split(",")
    unknown = [k for k in kinds if k not in DISTURBANCES]
    if unknown:
        parser.error(f"unknown disturbance(s): {unknown}")
    variant = ("" if args.snapshot is None else f"_snapshot{args.snapshot}") + ("_policy_only" if 
        args.no_mpc else "")
    out_dir = args.run_dir / (args.out_name or f"disturbances{variant}")
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")

    agent, step, cfg = load_agent(args.run_dir, snapshot=args.snapshot)
    metric_target = bool(cfg.env.get("metric_target", False))
    seed = cfg.seed + 3000 if args.test_seed is None else args.test_seed
    starts = [entry for name in args.test_sets.split(",") for entry in TEST_SETS[name]]
    decision = "policy_only" if args.no_mpc else 
        f"planner_{agent.population_size}x{agent.mppi_iterations}"
    print(f"Loaded {args.run_dir} at {step} steps; {len(starts)} starts per condition; 
        {decision}.")

    testers = {0: DisturbedFlightTester(args.flight_steps, not args.no_mpc, metric_target)}
    conditions = [("none", 0.0)] + [(kind, s) for kind in kinds for s in DISTURBANCES[kind]]
    rows = []
    for kind, strength in conditions:
        delay = int(strength) if kind == "delay" else 0
        if delay not in testers:
            testers[delay] = DisturbedFlightTester(args.flight_steps, not args.no_mpc, 
                metric_target, delay)
        flights = testers[delay].fly(
            agent, starts, seed,
            wind=wind_profile(kind, strength, args.flight_steps),
            mass_factor=strength if kind == "mass" else 1.0,
            motor_factor=strength if kind == "motor" else 1.0)
        mine = [{"disturbance": kind, "strength": strength, "seed": cfg.seed, "checkpoint_step": 
            step,
                 "decision": decision, "flight": i, **f.summary(), "end_cause": end_cause(f),
                 "final_height_m": float(f.position[f.steps_flown - 1, 2])}
                for i, f in enumerate(flights)]
        rows += mine
        late = np.array([r["late_error_m"] for r in mine])
        print(f"{kind:<6} {strength:>5}: crashes {sum(r['crashed'] for r in mine):>3}/{len(mine)},
             "
              f"late error mean {late.mean():.3f} m, median {np.median(late):.3f} m", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _write_rows(out_dir / "summary.csv", rows, append=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
```

## code/src/tdmpc2_training/tdmpc2_jax/build.py

Building the agent. Last modified 2026-10-12.

```
"""Build an untrained TD-MPC2 agent from a Hydra config.

Training and evaluation both call this, so a checkpoint is always restored
into exactly the network shapes it was saved from.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from tdmpc2_jax.common.activations import mish
from tdmpc2_jax.networks import NormedLinear
from tdmpc2_jax.tdmpc2 import TDMPC2
from tdmpc2_jax.world_model import WorldModel


def build_agent(cfg, observation, action_dim, model_key, encoder_key):
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = dict(cfg['tdmpc2'])
  dtype = jnp.dtype(model_config.dtype)

  encoder_module = nn.Sequential(
      [
          NormedLinear(
              encoder_config.encoder_dim, activation=mish, dtype=dtype
          )
          for _ in range(encoder_config.num_encoder_layers-1)
      ] + [
          NormedLinear(
              model_config.latent_dim, activation=None, dtype=dtype
          )
      ]
  )

  if encoder_config.tabulate:
    print("Encoder")
    print("--------------")
    print(
        encoder_module.tabulate(
            jax.random.key(0), observation, compute_flops=True
        )
    )

  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, observation)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.clip_by_global_norm(model_config.max_grad_norm),
          optax.adam(encoder_config.learning_rate),
      )
  )

  model = WorldModel.create(
      action_dim=int(action_dim),
      encoder=encoder,
      **model_config,
      key=model_key
  )
  if model.action_dim >= 20:
    tdmpc_config['mppi_iterations'] += 2

  return TDMPC2.create(world_model=model, **tdmpc_config)
```

## code/src/tdmpc2_training/tdmpc2_jax/train.py

Training loop. Last modified 2026-11-01.

```
import os
import time
from collections import defaultdict
from functools import partial

import gymnasium as gym
import hydra
import jax
import numpy as np
import orbax.checkpoint as ocp
import tqdm
from tensorboardX import SummaryWriter

from tdmpc2_jax.build import build_agent
from tdmpc2_jax.data import SequentialReplayBuffer

@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

  ##############################
  # Environment setup
  ##############################
  # Options of the hover task (crazyflie_hover.CrazyflieHoverEnv). Configs
  # without them build the environment exactly as before.
  env_kwargs = {k: bool(env_config[k]) for k in ('random_starts', 'metric_target')
                if env_config.get(k) is not None}

  def make_env(env_config, seed):
    def make_gym_env(env_id, seed):
      env = gym.make(env_id, **env_kwargs)
      env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    if env_config.backend == "gymnasium":
      return make_gym_env(env_config.env_id, seed)
    elif env_config.backend == "dmc":
      from tdmpc2_jax.envs.dmcontrol import make_dmc_env
      env = make_dmc_env(env_config.env_id, seed, env_config.dmc.obs_type)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
    else:
      raise ValueError("Environment not supported:", env_config)

  if env_config.asynchronous:
    vector_env_cls = gym.vector.AsyncVectorEnv
  else:
    vector_env_cls = gym.vector.SyncVectorEnv
  env = vector_env_cls(
      [
          partial(make_env, env_config, seed)
          for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
      ]
  )
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  rng, model_key, encoder_key = jax.random.split(rng, 3)

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(
      dummy_action
  )
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.buffer_size,
      vectorized=True,
      num_envs=env_config.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc
      )
  )

  ##############################
  # Agent setup
  ##############################
  agent = build_agent(
      cfg,
      observation=dummy_obs,
      action_dim=np.prod(env.single_action_space.shape),
      model_key=model_key,
      encoder_key=encoder_key,
  )
  global_step = 0

  # Hover test flights during training (learning curve). Imported here because
  # the rest of this trainer is not specific to the drone.
  tester = None
  if cfg.get('eval_interval_steps'):
    from hover_eval import CURVE_STARTS, FlightTester, append_learning_curve
    tester = FlightTester(flight_steps=cfg.eval_flight_steps,
                          metric_target=env_kwargs.get('metric_target', False))
    curve_path = os.path.join(output_dir, 'learning_curve.csv')
    # Batch 1: a few flights from the training start. Batch 2: the same 21
    # starts from every test group in every run (hover_eval.CURVE_STARTS).
    if cfg.get('curve_set', 'home') == 'batch2':
      test_starts = CURVE_STARTS
    else:
      test_starts = ['home'] * int(cfg.eval_flights)

  def record_learning_curve(env_steps, train_seconds):
    flights = tester.fly(agent, test_starts, seed=cfg.seed + 1000)
    rows = append_learning_curve(
        curve_path, flights,
        env_steps=env_steps, train_seconds=round(train_seconds, 1)
    )
    late_error = np.mean([r['late_error_m'] for r in rows])
    writer.add_scalar('test/late_error_m', late_error, env_steps)
    writer.add_scalar('test/return', np.mean([r['return'] for r in rows]), env_steps)
    for test_set in sorted({r['test_set'] for r in rows}):
      mine = [r for r in rows if r['test_set'] == test_set]
      writer.add_scalar(f'test_{test_set}/late_error_m',
                        np.mean([r['late_error_m'] for r in mine]), env_steps)
      writer.add_scalar(f'test_{test_set}/crash_fraction',
                        np.mean([r['crashed'] for r in mine]), env_steps)
    tqdm.tqdm.write(f'Test flights at {env_steps} steps: late error {late_error:.3f} m')

  # Batch 2 keeps the agent (no replay buffer) every snapshot_interval_steps
  # in snapshots/, so any point of training can be tested later.
  snapshot_interval = cfg.get('snapshot_interval_steps')
  snapshots = None
  if snapshot_interval:
    snapshots = ocp.CheckpointManager(
        os.path.join(output_dir, 'snapshots'),
        options=ocp.CheckpointManagerOptions(create=True),
        item_names=('agent',)
    )

  def save_snapshot(env_steps):
    if env_steps not in snapshots.all_steps():  # a resumed run may already have it
      snapshots.save(env_steps, args=ocp.args.Composite(agent=ocp.args.StandardSave(agent)))

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg['save_interval_steps']
  )
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path,
      options=options,
      item_names=('agent', 'global_step', 'buffer_state')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
      )
      restored = mngr.restore(
          mngr.latest_step(),
          args=ocp.args.Composite(
              agent=ocp.args.StandardRestore(agent),
              global_step=ocp.args.JsonRestore(),
              buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
          )
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################
    ep_count = np.zeros(env_config.num_envs, dtype=int)
    prev_logged_step = global_step
    plan = None
    env_seed = cfg.seed
    if global_step > 0:
      # A resumed run (e.g. the next piece of a run split into cluster jobs)
      # must not replay the starts, sensor noise, planner noise and replay
      # samples it already used, so these depend on the step it resumes at.
      env_seed = cfg.seed * 10_000_000 + global_step
      rng = jax.random.fold_in(rng, global_step)
      replay_buffer.np_random = np.random.default_rng([cfg.seed, global_step])
    observation, _ = env.reset(seed=env_seed)

    seed_steps = int(cfg.get('seed_steps', 2500 * env_config.num_envs * env_config.utd_ratio))
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    done = np.zeros(env_config.num_envs, dtype=bool)
    # Wall-clock time spent training, with the test flights taken out.
    loop_start = time.perf_counter()
    test_seconds = 0.0
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
      if global_step <= seed_steps:
        action = env.action_space.sample()
      else:
        rng, action_key = jax.random.split(rng)
        action, plan = agent.act(
            observation,
            prev_plan=plan,
            deterministic=False,
            train=True,
            key=action_key
        )
        action = np.array(action)

      next_observation, reward, terminated, truncated, info = env.step(action)

      if np.any(~done):
        replay_buffer.insert(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated
            ),
            mask=~done
        )
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        if plan is not None:
          plan = (
              plan[0].at[done].set(0),
              plan[1].at[done].set(agent.max_plan_std)
          )
        for ienv in range(env_config.num_envs):
          if done[ienv]:
            r = info['episode']['r'][ienv]
            l = info['episode']['l'][ienv]
            print(
                f"Episode {ep_count[ienv]}: r = {r:.2f}, l = {l}"
            )
            writer.add_scalar('episode/return', r, global_step + ienv)
            writer.add_scalar('episode/length', l, global_step + ienv)
            ep_count[ienv] += 1

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = int(cfg.get('seed_updates', seed_steps))
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + \
            cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step

        for iupdate in range(num_updates):
          batch = replay_buffer.sample(agent.batch_size, agent.horizon)
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate]
          )

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            writer.add_scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.add_scalar(f'train/{k}_std', np.std(v), global_step)

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

      pbar.update(env_config.num_envs)

      env_steps = global_step + env_config.num_envs
      if tester is not None and env_steps % cfg.eval_interval_steps < env_config.num_envs:
        test_start = time.perf_counter()
        record_learning_curve(env_steps, test_start - loop_start - test_seconds)
        test_seconds += time.perf_counter() - test_start
      if snapshots is not None and env_steps % snapshot_interval < env_config.num_envs:
        save_snapshot(env_steps)
    pbar.close()
    if snapshots is not None:
      snapshots.wait_until_finished()
      snapshots.close()
    # The upstream interval can exceed a short experiment. Always keep the
    # final learned state so smoke runs can be resumed and inspected too.
    final_step = global_step + env_config.num_envs
    if final_step > 0 and final_step != mngr.latest_step():
      mngr.save(
          final_step, force=True,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(final_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    # Make sure the learning curve ends at the final agent.
    if tester is not None and final_step % cfg.eval_interval_steps >= env_config.num_envs:
      test_start = time.perf_counter()
      record_learning_curve(final_step, test_start - loop_start - test_seconds)
    print(f'Training took {(time.perf_counter() - loop_start) / 60:.1f} min. '
          f'Full test flights: eval_hover.py {output_dir}')
    writer.close()


if __name__ == '__main__':
  train()
```

## code/src/tdmpc2_training/tdmpc2_jax/tdmpc2.py

TD-MPC2 agent: planning (MPPI) and updates. Last modified 2026-09-28.

```
from __future__ import annotations
from functools import partial
from flax import struct
import flax
import jax
from jaxtyping import PRNGKeyArray, PyTree
import optax

from tdmpc2_jax.world_model import WorldModel
import jax.numpy as jnp
from tdmpc2_jax.common.loss import soft_crossentropy
import numpy as np
from typing import Any, Dict, Optional, Tuple
from tdmpc2_jax.common.scale import percentile_normalization
from tdmpc2_jax.common.util import sg


class TDMPC2(struct.PyTreeNode):
  model: WorldModel
  value_scale: jax.Array

  # Planning
  horizon: int = struct.field(pytree_node=False)
  mppi_iterations: int = struct.field(pytree_node=False)
  population_size: int = struct.field(pytree_node=False)
  policy_prior_samples: int = struct.field(pytree_node=False)
  num_elites: int = struct.field(pytree_node=False)
  min_plan_std: float
  max_plan_std: float
  temperature: float
  # Optimization
  batch_size: int = struct.field(pytree_node=False)
  discount: float
  rho: float
  consistency_loss_scale: float
  reward_loss_scale: float
  value_loss_scale: float
  continue_loss_scale: float
  entropy_coef: float
  tau: float

  @classmethod
  def create(cls,
             world_model: WorldModel,
             # Planning
             horizon: int,
             mppi_iterations: int,
             population_size: int,
             policy_prior_samples: int,
             num_elites: int,
             min_plan_std: float,
             max_plan_std: float,
             temperature: float,
             # Optimization
             discount: float,
             batch_size: int,
             rho: float,
             consistency_loss_scale: float,
             reward_loss_scale: float,
             value_loss_scale: float,
             continue_loss_scale: float,
             entropy_coef: float,
             tau: float
             ) -> TDMPC2:

    return cls(model=world_model,
               horizon=horizon,
               mppi_iterations=mppi_iterations,
               population_size=population_size,
               policy_prior_samples=policy_prior_samples,
               num_elites=num_elites,
               min_plan_std=min_plan_std,
               max_plan_std=max_plan_std,
               temperature=temperature,
               discount=discount,
               batch_size=batch_size,
               rho=rho,
               consistency_loss_scale=consistency_loss_scale,
               reward_loss_scale=reward_loss_scale,
               value_loss_scale=value_loss_scale,
               continue_loss_scale=continue_loss_scale,
               entropy_coef=entropy_coef,
               tau=tau,
               value_scale=jnp.array([1.0]),
               )

  @partial(jax.jit, static_argnames=('mpc', 'deterministic', 'train'))
  def act(self,
          obs: PyTree,
          prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
          mpc: bool = True,
          deterministic: bool = False,
          train: bool = False,
          *,
          key: PRNGKeyArray
          ) -> Tuple[np.ndarray, Optional[Tuple[jax.Array]]]:
    encoder_key, action_key = jax.random.split(key, 2)
    z = self.model.encode(
        obs=obs,
        params=self.model.encoder.params,
        key=encoder_key
    )

    if mpc:
      action, plan = self.plan(
          z=z,
          horizon=self.horizon,
          prev_plan=prev_plan,
          deterministic=deterministic,
          train=train,
          key=action_key
      )
    else:
      action, _, _, _ = self.model.sample_actions(
          z=z,
          deterministic=deterministic,
          params=self.model.policy_model.params,
          key=action_key
      )
      plan = None

    return action, plan

  @partial(jax.jit, static_argnames=('horizon', 'deterministic', 'train'))
  def plan(self,
           z: jax.Array,
           horizon: int,
           prev_plan: Tuple[jax.Array, jax.Array] = None,
           deterministic: bool = False,
           train: bool = False,
           *,
           key: PRNGKeyArray,
           ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    batch_shape = z.shape[:-1]
    actions = jnp.zeros(
        (
            *batch_shape,
            self.population_size,
            horizon,
            self.model.action_dim
        )
    )

    ###########################################################
    # Policy prior samples
    ###########################################################
    if self.policy_prior_samples > 0:
      key, *prior_noise_keys = jax.random.split(key, 1+horizon)
      policy_actions = jnp.zeros(
          (
              *batch_shape,
              self.policy_prior_samples,
              horizon,
              self.model.action_dim
          )
      )
      z_t = z[..., None, :].repeat(self.policy_prior_samples, axis=-2)
      for t in range(horizon):
        policy_actions = policy_actions.at[..., t, :].set(
            self.model.sample_actions(
                z=z_t,
                deterministic=False,
                params=self.model.policy_model.params,
                key=prior_noise_keys[t]
            )[0]
        )
        if t < horizon-1:  # Don't need for the last time step
          z_t = self.model.next(
              z=z_t,
              a=policy_actions[..., t, :],
              params=self.model.dynamics_model.params
          )

      actions = actions.at[..., :self.policy_prior_samples, :, :].set(
          policy_actions
      )

    ###########################################################
    # MPPI planning
    ###########################################################
    z_t = z[..., None, :].repeat(self.population_size, axis=-2)
    key, mppi_noise_key, *value_keys = jax.random.split(
        key, 2+self.mppi_iterations
    )
    noise = jax.random.normal(
        mppi_noise_key,
        shape=(
            *batch_shape,
            self.population_size - self.policy_prior_samples,
            self.mppi_iterations,
            horizon,
            self.model.action_dim
        )
    )
    # Initialize population state
    mean = jnp.zeros((*batch_shape, horizon, self.model.action_dim))
    std = jnp.full(
        (*batch_shape, horizon, self.model.action_dim), self.max_plan_std
    )
    if prev_plan is not None:
      mean = mean.at[..., :-1, :].set(prev_plan[0][..., 1:, :])

    for i in range(self.mppi_iterations):
      actions = actions.at[..., self.policy_prior_samples:, :, :].set(
          mean[..., None, :, :] + std[..., None, :, :] * noise[..., i, :, :]
      ).clip(-1, 1)

      # Compute elites
      values = self.estimate_value(
          z=z_t, actions=actions, horizon=horizon, key=value_keys[i]
      )
      elite_values, elite_inds = jax.lax.top_k(values, self.num_elites)
      elite_actions = jnp.take_along_axis(
          actions, elite_inds[..., None, None], axis=-3
      )

      # Update population distribution
      score = jax.nn.softmax(self.temperature * elite_values)
      mean = jnp.sum(score[..., None, None] * elite_actions, axis=-3)
      std = jnp.sqrt(
          jnp.sum(
              score[..., None, None] *
              (elite_actions - mean[..., None, :, :])**2,
              axis=-3
          ) + 1e-6
      ).clip(self.min_plan_std, self.max_plan_std)

    # Sample final action
    if deterministic:  # Use best trajectory
      action_ind = jnp.argmax(elite_values, axis=-1)
    else:  # Sample from elites
      key, final_action_key = jax.random.split(key)
      action_ind = jax.random.categorical(
          final_action_key, logits=jnp.log(score), shape=batch_shape
      )
    action = jnp.take_along_axis(
        elite_actions, action_ind[..., None, None, None], axis=-3
    ).squeeze(-3)
    if train:
      key, final_noise_key = jax.random.split(key)
      final_action = action[..., 0, :] + std[..., 0, :] * \
          jax.random.normal(
              final_noise_key, shape=batch_shape + (self.model.action_dim,)
      )
    else:
      final_action = action[..., 0, :]

    return final_action.clip(-1, 1), (mean, std)

  @jax.jit
  def update(self,
             observations: PyTree,
             actions: jax.Array,
             rewards: jax.Array,
             next_observations: PyTree,
             terminated: jax.Array,
             truncated: jax.Array,
             *,
             key: PRNGKeyArray
             ) -> Tuple[TDMPC2, Dict[str, Any]]:

    world_model_key, policy_key = jax.random.split(key, 2)

    def world_model_loss_fn(encoder_params: flax.core.FrozenDict,
                            dynamics_params: flax.core.FrozenDict,
                            value_params: flax.core.FrozenDict,
                            reward_params: flax.core.FrozenDict,
                            continue_params: flax.core.FrozenDict,
                            ) -> Tuple[jax.Array, Dict[str, Any]]:
      encoder_key, value_key = jax.random.split(world_model_key, 2)
      lam = self.rho**jnp.arange(self.horizon)
      lam /= jnp.sum(lam)

      ###########################################################
      # Encoder forward pass
      ###########################################################
      all_obs = jax.tree.map(
          lambda x, y: jnp.stack([x, y], axis=0),
          observations, next_observations
      )
      all_zs = self.model.encode(
          obs=all_obs, params=encoder_params, key=encoder_key
      )
      encoder_zs = jax.tree.map(lambda x: x[0], all_zs)
      next_zs = jax.tree.map(lambda x: x[1], all_zs)

      ###########################################################
      # Latent rollout (dynamics + consistency loss)
      ###########################################################
      done = jnp.logical_or(terminated, truncated)
      finished = jnp.zeros((self.horizon+1, self.batch_size), dtype=bool)
      latent_zs = jnp.zeros(
          (self.horizon+1, self.batch_size, self.model.latent_dim)
      )
      latent_zs = latent_zs.at[0].set(encoder_zs[0])
      consistency_loss = 0
      for t in range(self.horizon):
        z = self.model.next(
            z=latent_zs[t], a=actions[t], params=dynamics_params
        )
        consistency_loss += lam[t] * \
            jnp.mean((z - sg(next_zs[t]))**2, where=~finished[t][:, None])
        latent_zs = latent_zs.at[t+1].set(z)
        finished = finished.at[t+1].set(jnp.logical_or(finished[t], done[t]))

      ###########################################################
      # Reward loss
      ###########################################################
      _, reward_logits = self.model.reward(
          z=latent_zs[:-1], a=actions, params=reward_params,
      )
      reward_loss = jnp.sum(
          lam[:, None] * soft_crossentropy(
              pred_logits=reward_logits,
              target=rewards,
              low=self.model.symlog_min,
              high=self.model.symlog_max,
              num_bins=self.model.num_bins,
          ), axis=0, where=~finished[:-1]
      ).mean()

      ###########################################################
      # Value loss
      ###########################################################
      next_action_key, value_target_key, ensemble_key, value_key = \
          jax.random.split(value_key, 4)

      # TD targets
      next_action = self.model.sample_actions(
          z=next_zs,
          deterministic=False,
          params=self.model.policy_model.params,
          key=next_action_key
      )[0]
      Qs, _ = self.model.Q(
          z=next_zs,
          a=next_action,
          params=self.model.target_value_model.params,
          key=value_target_key
      )
      # Subsample value networks
      inds = jax.random.choice(
          ensemble_key,
          jnp.arange(0, self.model.num_value_nets),
          shape=(2, ),
          replace=False
      )
      Q = Qs[inds].min(axis=0)
      td_targets = rewards + (1 - terminated) * self.discount * Q

      _, Q_logits = self.model.Q(
          z=latent_zs[:-1], a=actions, params=value_params, key=value_key
      )
      value_loss = jnp.sum(
          lam[:, None] * soft_crossentropy(
              pred_logits=Q_logits,
              target=sg(td_targets),
              low=self.model.symlog_min,
              high=self.model.symlog_max,
              num_bins=self.model.num_bins
          ), axis=1, where=~finished[:-1]
      ).mean()

      ###########################################################
      # Continue loss
      ###########################################################
      if self.model.predict_continues:
        continue_logits = self.model.continue_model.apply_fn(
            {'params': continue_params}, latent_zs[:-1]
        ).squeeze(-1)
        continue_loss = optax.sigmoid_binary_cross_entropy(
            continue_logits, 1 - terminated
        ).mean()
      else:
        continue_loss = 0.0

      total_loss = (
          self.consistency_loss_scale * consistency_loss +
          self.reward_loss_scale * reward_loss +
          self.value_loss_scale * value_loss +
          self.continue_loss_scale * continue_loss
      )

      return total_loss, {
          'consistency_loss': consistency_loss,
          'reward_loss': reward_loss,
          'value_loss': value_loss,
          'continue_loss': continue_loss,
          'total_loss': total_loss,
          'latent_zs': latent_zs,
          'finished': finished,
      }

    # Update world model
    (encoder_grads, dynamics_grads, value_grads, reward_grads, continue_grads), model_info = 
        jax.grad(
        world_model_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            self.model.encoder.params,
            self.model.dynamics_model.params,
            self.model.value_model.params,
            self.model.reward_model.params,
            self.model.continue_model.params if self.model.predict_continues else None
    )
    new_encoder = self.model.encoder.apply_gradients(grads=encoder_grads)
    new_dynamics_model = self.model.dynamics_model.apply_gradients(
        grads=dynamics_grads
    )
    new_reward_model = self.model.reward_model.apply_gradients(
        grads=reward_grads
    )
    new_value_model = self.model.value_model.apply_gradients(
        grads=value_grads)
    new_target_value_model = self.model.target_value_model.replace(
        params=optax.incremental_update(
            new_value_model.params,
            self.model.target_value_model.params,
            self.tau
        )
    )
    if self.model.predict_continues:
      new_continue_model = self.model.continue_model.apply_gradients(
          grads=continue_grads
      )
    else:
      new_continue_model = self.model.continue_model

    # Update policy
    latent_zs = model_info.pop('latent_zs')
    finished = model_info.pop('finished')

    def policy_loss_fn(actor_params: flax.core.FrozenDict):
      action_key, Q_key = jax.random.split(policy_key, 2)
      actions, _, log_std, log_probs = self.model.sample_actions(
          z=latent_zs,
          deterministic=False,
          params=actor_params,
          key=action_key
      )

      # Compute policy objective (equation 4)
      lam = self.rho**jnp.arange(self.horizon+1)
      lam /= jnp.sum(lam)
      Qs, _ = self.model.Q(
          z=latent_zs, a=actions, params=new_value_model.params, key=Q_key
      )
      Q = Qs.mean(axis=0)
      Q_scale = percentile_normalization(Q[0], self.value_scale).clip(1, None)
      policy_loss = jnp.sum(
          lam[:, None] * (self.entropy_coef * log_probs - Q / sg(Q_scale)),
          axis=0, where=~finished
      ).mean()
      return policy_loss, {
          'policy_loss': policy_loss,
          'policy_log_std': log_std,
          'value_scale': Q_scale
      }

    policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
        self.model.policy_model.params
    )
    new_policy = self.model.policy_model.apply_gradients(grads=policy_grads)

    # Update model
    new_agent = self.replace(
        model=self.model.replace(
            encoder=new_encoder,
            dynamics_model=new_dynamics_model,
            reward_model=new_reward_model,
            value_model=new_value_model,
            policy_model=new_policy,
            target_value_model=new_target_value_model,
            continue_model=new_continue_model
        ),
        value_scale=policy_info['value_scale']
    )
    info = {**model_info, **policy_info}

    return new_agent, info

  @partial(jax.jit, static_argnames=('horizon'))
  def estimate_value(self,
                     z: jax.Array,
                     actions: jax.Array,
                     horizon: int,
                     key: PRNGKeyArray
                     ) -> jax.Array:
    G, discount = 0.0, 1.0
    for t in range(horizon):
      reward, _ = self.model.reward(
          z=z, a=actions[..., t, :], params=self.model.reward_model.params
      )
      z = self.model.next(
          z=z, a=actions[..., t, :], params=self.model.dynamics_model.params
      )
      G += discount * reward
      discount *= self.discount

      if self.model.predict_continues:
        continues = jax.nn.sigmoid(
            self.model.continue_model.apply_fn(
                {'params': self.model.continue_model.params}, z
            )
        ).squeeze(-1) > 0.5
        discount *= continues

    action_key, Q_key = jax.random.split(key, 2)
    next_action = self.model.sample_actions(
        z=z,
        deterministic=False,
        params=self.model.policy_model.params,
        key=action_key
    )[0]

    Qs, _ = self.model.Q(
        z=z,
        a=next_action,
        params=self.model.value_model.params,
        key=Q_key
    )
    Q = Qs.mean(axis=0)
    return G + discount * Q
```

## code/src/tdmpc2_training/tdmpc2_jax/world_model.py

TD-MPC2 world model. Last modified 2026-09-28.

```
import copy
from functools import partial
from typing import Dict, Tuple, Callable
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from numpy.typing import ArrayLike
from tdmpc2_jax.networks import NormedLinear
from tdmpc2_jax.common.activations import mish, simnorm
from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp
import optax
from tdmpc2_jax.networks import Ensemble
from tdmpc2_jax.common.util import symlog, two_hot_inv


class WorldModel(struct.PyTreeNode):
  # Models
  encoder: TrainState
  dynamics_model: TrainState
  reward_model: TrainState
  policy_model: TrainState
  value_model: TrainState
  target_value_model: TrainState
  continue_model: TrainState
  # Spaces
  action_dim: int = struct.field(pytree_node=False)
  # Architecture
  latent_dim: int = struct.field(pytree_node=False)
  simnorm_dim: int = struct.field(pytree_node=False)
  num_value_nets: int = struct.field(pytree_node=False)
  num_bins: int = struct.field(pytree_node=False)
  symlog_min: float
  symlog_max: float
  predict_continues: bool = struct.field(pytree_node=False)

  @classmethod
  def create(cls,
             # Spaces
             action_dim: int,
             # Encoder module
             encoder: TrainState,
             # World model
             latent_dim: int,
             value_dropout: float,
             num_value_nets: int,
             num_bins: int,
             symlog_min: float,
             symlog_max: float,
             simnorm_dim: int,
             predict_continues: bool,
             # Optimization
             learning_rate: float,
             max_grad_norm: float = 20,
             # Misc
             tabulate: bool = False,
             dtype: jnp.dtype = jnp.float32,
             *,
             key: PRNGKeyArray,
             ):
    dynamics_key, reward_key, value_key, policy_key, continue_key = jax.random.split(
        key, 5)

    # Latent forward dynamics model
    dynamics_module = nn.Sequential([
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=None, dtype=dtype)
    ])
    dynamics_model = TrainState.create(
        apply_fn=dynamics_module.apply,
        params=dynamics_module.init(
            dynamics_key, jnp.zeros(latent_dim + action_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )

    # Transition reward model
    reward_module = nn.Sequential([
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        nn.Dense(num_bins, kernel_init=nn.initializers.zeros)
    ])
    reward_model = TrainState.create(
        apply_fn=reward_module.apply,
        params=reward_module.init(
            reward_key, jnp.zeros(latent_dim + action_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )

    # Policy model
    policy_module = nn.Sequential([
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        nn.Dense(2*action_dim,
                 kernel_init=nn.initializers.truncated_normal(0.02))
    ])
    policy_model = TrainState.create(
        apply_fn=policy_module.apply,
        params=policy_module.init(policy_key, jnp.zeros(latent_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )

    # Return/value model (ensemble)
    value_param_key, value_dropout_key = jax.random.split(value_key)
    value_base = partial(nn.Sequential, [
        NormedLinear(latent_dim, activation=mish,
                     dropout_rate=value_dropout, dtype=dtype),
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        nn.Dense(num_bins, kernel_init=nn.initializers.zeros)
    ])
    value_ensemble = Ensemble(value_base, num=num_value_nets)
    value_model = TrainState.create(
        apply_fn=value_ensemble.apply,
        params=value_ensemble.init(
            {'params': value_param_key, 'dropout': value_dropout_key},
            jnp.zeros(latent_dim + action_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )
    target_value_model = TrainState.create(
        apply_fn=value_ensemble.apply,
        params=copy.deepcopy(value_model.params),
        tx=optax.GradientTransformation(lambda _: None, lambda _: None))

    if predict_continues:
      continue_module = nn.Sequential([
          NormedLinear(latent_dim, activation=mish, dtype=dtype),
          NormedLinear(latent_dim, activation=mish, dtype=dtype),
          nn.Dense(1, kernel_init=nn.initializers.zeros)
      ])
      continue_model = TrainState.create(
          apply_fn=continue_module.apply,
          params=continue_module.init(
              continue_key, jnp.zeros(latent_dim))['params'],
          tx=optax.chain(
              optax.zero_nans(),
              optax.clip_by_global_norm(max_grad_norm),
              optax.adamw(learning_rate),
          )
      )
    else:
      continue_model = None

    if tabulate:
      print("Dynamics Model")
      print("--------------")
      print(
          dynamics_module.tabulate(
              jax.random.key(0),
              jnp.ones(latent_dim + action_dim),
              compute_flops=True
          )
      )

      print("Reward Model")
      print("------------")
      print(
          reward_module.tabulate(
              jax.random.key(0),
              jnp.ones(latent_dim + action_dim),
              compute_flops=True
          )
      )

      print("Policy Model")
      print("------------")
      print(
          policy_module.tabulate(
              jax.random.key(0), jnp.ones(latent_dim), compute_flops=True
          )
      )

      print("Value Model")
      print("-----------")
      value_param_key, value_dropout_key = jax.random.split(value_key)
      print(
          value_ensemble.tabulate(
              {'params': value_param_key, 'dropout': value_dropout_key},
              jnp.ones(latent_dim + action_dim),
              compute_flops=True
          )
      )

      if predict_continues:
        print("Continue Model")
        print("--------------")
        print(
            continue_module.tabulate(
                jax.random.key(0), jnp.ones(latent_dim), compute_flops=True
            )
        )

    return cls(
        # Spaces
        action_dim=action_dim,
        # Models
        encoder=encoder,
        dynamics_model=dynamics_model,
        reward_model=reward_model,
        policy_model=policy_model,
        value_model=value_model,
        target_value_model=target_value_model,
        continue_model=continue_model,
        # Architecture
        latent_dim=latent_dim,
        simnorm_dim=simnorm_dim,
        num_value_nets=num_value_nets,
        num_bins=num_bins,
        symlog_min=float(symlog_min),
        symlog_max=float(symlog_max),
        predict_continues=predict_continues,
    )

  @jax.jit
  def encode(self, obs: np.ndarray, params: Dict, key: PRNGKeyArray) -> jax.Array:
    z = self.encoder.apply_fn(
        {'params': params}, obs, rngs={'dropout': key}
    ).astype(jnp.float32)
    return simnorm(z, simplex_dim=self.simnorm_dim)

  @jax.jit
  def next(self, z: jax.Array, a: jax.Array, params: Dict) -> jax.Array:
    z = self.dynamics_model.apply_fn(
        {'params': params}, jnp.concatenate([z, a], axis=-1)
    ).astype(jnp.float32)
    return simnorm(z, simplex_dim=self.simnorm_dim)

  @jax.jit
  def reward(self, z: jax.Array, a: jax.Array, params: Dict
             ) -> Tuple[jax.Array, jax.Array]:
    z = jnp.concatenate([z, a], axis=-1)
    logits = self.reward_model.apply_fn(
        {'params': params}, z
    ).astype(jnp.float32)
    reward = two_hot_inv(
        logits, self.symlog_min, self.symlog_max, self.num_bins
    )
    return reward, logits

  @partial(jax.jit, static_argnames=('deterministic',))
  def sample_actions(self,
                     z: jax.Array,
                     params: Dict,
                     deterministic: bool = False,
                     min_log_std: float = -10,
                     max_log_std: float = 2,
                     *,
                     key: PRNGKeyArray
                     ) -> Tuple[jax.Array, ...]:
    # Chunk the policy model output to get mean and logstd
    mean, log_std = jnp.split(
        self.policy_model.apply_fn({'params': params}, z).astype(jnp.float32), 2, axis=-1
    )
    log_std = min_log_std + 0.5 * \
        (max_log_std - min_log_std) * (jnp.tanh(log_std) + 1)

    if deterministic:
      action = mean
    else:
      action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    log_probs = -0.5 * jnp.sum(
        ((action - mean) / jnp.exp(log_std)) ** 2
        + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )

    # Squash tanh
    log_probs -= jnp.sum(
        (2 * (jnp.log(2) - action - jax.nn.softplus(-2 * action))), axis=-1
    )
    mean = jnp.tanh(mean)
    action = jnp.tanh(action)
    return action, mean, log_std, log_probs

  @jax.jit
  def Q(self, z: jax.Array, a: jax.Array, params: Dict, key: PRNGKeyArray
        ) -> Tuple[jax.Array, jax.Array]:
    z = jnp.concatenate([z, a], axis=-1)
    logits = self.value_model.apply_fn(
        {'params': params}, z, rngs={'dropout': key}
    ).astype(jnp.float32)

    Q = two_hot_inv(logits, self.symlog_min, self.symlog_max, self.num_bins)
    return Q, logits
```

## code/src/tdmpc2_training/reports/experiment_registry.json

Experiment registry. Last modified 2026-11-02.

```
{
  "description": "Every TD-MPC2 Crazyflie experiment: question, settings, code version, where it 
      ran, status and outputs. Paths starting with outputs/ are on the home PC (repository root); 
      $SCRATCH paths are on Narval (/scratch/mounirz). Times are EST.",
  "updated": "2026-11-02T12:30",
  "experiments": [
    {"id": "B1-B", "title": "Small TD-MPC2, fixed start, home PC", "question": "Seed variability 
        of short runs", "algorithm": "TD-MPC2 (config_hover, small)", "seeds": "0-10", 
        "env_steps": 50000, "code": "62b44e2 + uncommitted changes", "hardware": "RTX 3060 (WSL2),
         2 runs in parallel", "status": "completed 2026-10-12", "outputs": 
        ["outputs/batch1_fixed_start"]},
    {"id": "B1-C", "title": "Small TD-MPC2, fixed start, Narval", "question": "Seed variability; 
        home vs cluster", "algorithm": "TD-MPC2 (config_hover, small)", "seeds": "0-10", 
        "env_steps": 50000, "code": "62b44e2 + uncommitted changes", "hardware": "A100 2/7 slice",
         "scheduler_job": "5425899", "status": "completed 2026-10-12", "outputs": 
        ["outputs/batch1_small_narval", "$SCRATCH/tdmpc2/batch1_small_narval"]},
    {"id": "B1-D", "title": "Published TD-MPC2, fixed start, 50k", "question": "Model size at a 
        short budget", "algorithm": "TD-MPC2 (config_hover_standard)", "seeds": "0-10", 
        "env_steps": 50000, "code": "62b44e2 + uncommitted changes", "hardware": "A100 2/7 slice",
         "scheduler_job": "5420951", "status": "completed 2026-10-12", "outputs": 
        ["outputs/batch1_standard", "$SCRATCH/tdmpc2/batch1_standard"]},
    {"id": "B1-G", "title": "Published TD-MPC2, fixed start, 1M", "question": "Does long training 
        fix the variance and the drift?", "algorithm": "TD-MPC2 (config_hover_standard, buffer 
        1M)", "seeds": "0-2", "env_steps": 1000000, "code": "62b44e2 + uncommitted changes", 
        "hardware": "A100 2/7 slice", "scheduler_job": "5422314", "status": "completed 2026-10-13 
        (checkpoints only on Narval)", "outputs": ["outputs/batch1_standard_1M", 
        "$SCRATCH/tdmpc2/batch1_standard_1M"]},
    {"id": "B1-E", "title": "PPO (lab code), fixed start (historical baseline)", "question": 
        "Model-free comparison", "algorithm": "Brax PPO, 2048 envs", "seeds": "0-10", "env_steps":
         78643200, "code": "62b44e2 + uncommitted changes", "hardware": "A100 2/7 slice", 
        "scheduler_job": "5425898", "status": "completed 2026-10-12", "notes": "Known bugs found 
        2026-11-01: LR schedule never anneals (constant 3e-3); auto-reset carries previous 
        position/action into the next episode. Results kept as historical; not rerun.", "outputs":
         ["outputs/ppo_fixed_start"]},
    {"id": "B1-F", "title": "Lab's existing PPO checkpoint re-tested", "question": "Model-free 
        agent trained on random tasks", "algorithm": "Brax PPO (lab, 60M steps, random resets)", 
        "status": "completed 2026-10-12", "outputs": ["outputs/ppo_existing_checkpoint"]},
    {"id": "M-LAT-HOME", "title": "Decision latency benchmark, home", "question": "Q9 inference 
        speed vs model size", "script": "bench_latency.py", "hardware": "RTX 3060 (WSL2) and Ryzen
         5 5500", "status": "completed 2026-11-01", "outputs": ["outputs/latency_benchmark"]},
    {"id": "M-ABL-50K", "title": "Planner vs policy only vs 512x6, small 50k agents", "question": 
        "Q8 does planning help", "script": "eval_hover.py --no-mpc / --plan 512x6", "seeds": "0-10
         (batch1_small_narval)", "status": "completed 2026-11-01", "outputs": 
        ["outputs/batch1_small_narval/seed_*/evaluation_policy_only", 
        "outputs/batch1_small_narval/seed_*/evaluation_plan512x6"]},
    {"id": "B2-TEST", "title": "Narval test job, R1 settings shortened", "question": "Pipeline 
        check before launch", "config": "config_batch2_random_start + short intervals", "seeds": 
        "0", "env_steps": 4000, "code": "8edab5b", "scheduler_job": "7659264", "status": 
        "completed 2026-11-01 20:52 (15 min, exit 0)", "outputs": 
        ["$SCRATCH/tdmpc2/batch2_validation/narval_test"]},
    {"id": "B2-R1", "title": "Published TD-MPC2, random starts, 1M", "question": "Q6/Q7: learn 
        from a continuous start distribution; generalisation to held-out and outside starts; 
        planning vs policy under shift", "config": "config_batch2_random_start", "seeds": "0-10", 
        "env_steps": 1000000, "code": "8edab5b (tasks 0-9), 52873af (task 10)", "hardware": "A100 
        2/7 slice, 10 h limit", "scheduler_job": "7659438", "status": "completed 2026-11-02 04:33 
        (11/11 exit 0, 6.6-6.9 h): planning 46/39/20% success (training area/held-out/outside), 
        0/2/26% crashes; policy only 60/64/40%; still improving at 1M", "outputs": 
        ["outputs/batch2/random_start_1M", "$SCRATCH/tdmpc2/batch2/random_start_1M"]},
    {"id": "B2-C1", "title": "Published TD-MPC2, fixed start, 1M, metric observation", "question":
         "Control for R1: isolates the effect of random starts", "config": 
        "config_batch2_fixed_start", "seeds": "0-2", "env_steps": 1000000, "code": "52873af", 
        "hardware": "A100 2/7 slice, 10 h limit", "scheduler_job": "7659439", "status": "completed
         2026-11-02 04:30 (3/3 exit 0): planning 26/4/0% success, 35/82/92% crashes; same as Batch
         1 published 1M (+6/+2/-2 points)", "outputs": ["outputs/batch2/fixed_start_1M", 
        "$SCRATCH/tdmpc2/batch2/fixed_start_1M"]},
    {"id": "B2-R4", "title": "Batch 1 agents on the Batch 2 test starts", "question": 
        "Before/after baseline; planning vs policy for single-start agents", "script": 
        "cluster/eval_job.sh", "runs": "batch1_standard_1M (3), batch1_standard (11), 
        batch1_small_narval (11)", "code": "8edab5b", "scheduler_job": "7659440", "status": 
        "completed 2026-11-01 ~21:20", "outputs": 
        ["outputs/batch1_*/seed_*/evaluation_batch2_tests", 
        "outputs/batch1_*/seed_*/evaluation_batch2_tests_policy_only", 
        "outputs/batch1_standard_1M/analysis"]},
    {"id": "B2-P", "title": "Training-step profile and decision latency on Narval", "question": 
        "Q4 GPU use and wall-clock; Q9 latency", "script": "cluster/profile_job.sh", "code": 
        "8edab5b", "scheduler_job": "7659441", "status": "completed 2026-11-01 21:28 (30 min)", 
        "outputs": ["outputs/batch2/profiling_narval", "$SCRATCH/tdmpc2/batch2/profiling"]},
    {"id": "B2-D-G", "title": "Disturbance tests of the Batch 1 1M agents", "question": "Q8 
        robustness, with vs without planning, single-start agents", "script": 
        "cluster/disturb_job.sh", "code": "52873af", "scheduler_job": "7660609", "status": 
        "completed 2026-11-01", "outputs": 
        ["$SCRATCH/tdmpc2/batch1_standard_1M/seed_*/disturbances*"]},
    {"id": "B2-D-R1", "title": "Disturbance tests of R1 agents", "question": "Q8 robustness after 
        random-start training", "script": "cluster/disturb_job.sh", "code": "52873af", 
        "scheduler_job": "7660720 (aftercorr:7659438)", "status": "completed 2026-11-02 05:24", 
        "outputs": ["outputs/batch2/random_start_1M/seed_*/disturbances*", 
        "$SCRATCH/tdmpc2/batch2/random_start_1M/seed_*/disturbances*"]},
    {"id": "B2-D-C1", "title": "Disturbance tests of C1 agents", "question": "Q8 robustness, 
        fixed-start control", "script": "cluster/disturb_job.sh", "code": "52873af", 
        "scheduler_job": "7660721 (aftercorr:7659439)", "status": "completed 2026-11-02 05:14", 
        "outputs": ["outputs/batch2/fixed_start_1M/seed_*/disturbances*", 
        "$SCRATCH/tdmpc2/batch2/fixed_start_1M/seed_*/disturbances*"]},
    {"id": "A1-VAR", "title": "Evaluation noise vs seed differences (Batch 1 agents)", "question":
         "Q1 why short runs varied", "script": "variance_decomposition.py", "runs": 
        "batch1_small_narval, batch1_fixed_start, batch1_standard (33 agents x 20 flights)", 
        "hardware": "RTX 3060 (home)", "status": "completed 2026-11-01: ICC 0.93 / 0.94 (small), 
        0.52 (published 50k)", "outputs": ["outputs/batch2/variance_decomposition"]},
    {"id": "A2-FLUCT", "title": "Within-run fluctuations vs training statistics (Batch 1 1M 
        runs)", "question": "Q1/Q2 why long runs still swing", "script": 
        "fluctuation_analysis.py", "status": "completed 2026-11-01 ~22:35: no correlation between 
        changes in test error and changes in any logged loss", "outputs": 
        ["outputs/batch2/fluctuation_analysis"]},
    {"id": "B2-D-SMALL", "title": "Small TD-MPC2, random starts, 1M (experiment D)", "question": 
        "Q9 model size: accuracy, generalisation, robustness vs latency", "config": 
        "config_batch2_random_start_small", "seeds": "0-2 (pilot), then 3-10 (scale-up after the 
        200k check)", "env_steps": 1000000, "code": "pilot: commit of 'Small-model random-start 
        pilot config...'; seeds 3-10: commit of 'Snapshot evaluation on a validation set...' 
        (training code identical)", "hardware": "A100 2/7 slice, two pieces of <= 3 h each", 
        "scheduler_job": "pilot 7661493 -> 7661496 (aftercorr); seeds 3-10 7662436 -> 7662437 
        (aftercorr)", "status": "completed 2026-11-02 03:28 (11/11 exit 0): planning 78/83/59% 
        success, 0/0/3% crashes; policy only 9/9/12%", "outputs": 
        ["outputs/batch2/random_start_small_1M", "$SCRATCH/tdmpc2/batch2/random_start_small_1M"]},
    {"id": "B2-D-SMALL-DIST", "title": "Disturbance tests of the small agents", "script": 
        "cluster/disturb_job.sh", "scheduler_job": "7661497 (seeds 0-2), 7662438 (seeds 3-10), 
        aftercorr on piece 2", "status": "completed 2026-11-02", "outputs": 
        ["outputs/batch2/random_start_small_1M/seed_*/disturbances*", 
        "$SCRATCH/tdmpc2/batch2/random_start_small_1M/seed_*/disturbances*"]},
    {"id": "B2-SNAP", "title": "Snapshot evaluation: every 100k-step snapshot on 40 validation 
        starts + 110 test starts, with and without planning", "question": "Q1/Q2 within-run change
         vs evaluation noise; best vs final checkpoint (chosen on validation only); generalisation
         over training", "script": "eval_snapshots.py via cluster/snapshot_job.sh", 
        "scheduler_job": "R1 7662439, C1 7662440, small 0-2 7662441, small 3-10 7662442 (all 
        aftercorr on their training job)", "status": "completed 2026-11-02 05:58 (all batches 
        incl. reward pilot 7668947): best (validation) vs final, training area: D planner 97 vs 
        76%; R1 planner 49 vs 42%; R1 policy only 90 vs 60%", "outputs": 
        ["outputs/batch2/snapshot_summary", "outputs/batch2/snapshot_summary_policy_only", 
        "$SCRATCH/tdmpc2/batch2/*/seed_*/snapshot_evaluation*"]},
    {"id": "DEPLOY-SIL", "title": "Deployment adapter, software in the loop", "question": "Q10 
        sim-to-real interface", "script": "deploy_controller.py, check_deployment.py", "status": 
        "completed 2026-11-01 ~23:10: all checks pass (observation identical to the simulator over
         300 closed-loop steps for both observation types)", "outputs": 
        ["reports/DEPLOYMENT_READINESS.md"]},
    {"id": "E-PAR-HOME", "title": "Parallel-drone speed projection, home RTX 3060", "question": 
        "Q5", "script": "bench_parallel.py", "status": "completed 2026-11-01 ~23:40", "outputs": 
        ["outputs/batch2/parallel_projection_home"]},
    {"id": "E-PAR-NARVAL", "title": "Parallel-drone speed projection, Narval A100 slice, published
         and small sizes", "question": "Q5", "script": "cluster/parallel_job.sh", "scheduler_job":
         "7663279", "status": "completed 2026-11-02 00:35: 1.6x (published) and 2.9x (small) at 
        128 drones with 1 update per step", "outputs": 
        ["outputs/batch2/parallel_projection_narval"]},
    {"id": "H1-MODEL-ERR", "title": "World-model prediction error along test flights", "question":
         "Q7: is the planner exploiting model errors?", "script": "model_error.py via 
        cluster/model_error_job.sh", "scheduler_job": "7663280 (25 Batch 1 runs, completed 
        2026-11-01 23:56); R1 7663281, C1 7663282, small 7663283/7663284 (aftercorr)", "status": 
        "completed 2026-11-02 05:00: planner/policy model error (training area) Batch 1 1M 2.84, 
        C1 1.68, R1 1.17, D 0.68", "outputs": ["outputs/batch2/model_error_summary", 
        "outputs/batch1_*/seed_*/model_error", "outputs/batch2/model_error_summary_batch1"]},
    {"id": "H1-VARIANTS", "title": "Test-time planner variants (policy only, planner, narrower 
        sampling, half candidates from the policy)", "question": "Q7/Q8 causal test of model 
        exploitation", "script": "eval_planner_variants.py via cluster/variants_job.sh", 
        "scheduler_job": "7663755 (25 Batch 1 runs, completed 2026-11-02 00:32); R1 7663756, C1 
        7663757, small 7663758/7663773, randomized 7664893 (aftercorr)", "status": "completed 
        2026-11-02: restraining the planner helps single-start agents and R1 (+8 to +18 points), 
        costs D 5-7 points", "outputs": ["outputs/batch2/planner_variants_summary", 
        "outputs/batch1_*/seed_*/planner_variants", 
        "outputs/batch2/planner_variants_summary_batch1"]},
    {"id": "H2-DR-SMALL", "title": "Small TD-MPC2, random starts + randomised mass, motor strength
         and wind in every training flight (pilot)", "question": "H2: does planning help when the 
        world model has trained on varied physics?", "config": "config_batch2_randomized_small 
        (crazyflie_hover_dr.py: mass x1.0-1.15, motor x0.9-1.0, wind 0-3 m/s)", "seeds": "0-2", 
        "env_steps": 1000000, "hardware": "A100 2/7 slice, two pieces of <= 3 h", "scheduler_job":
         "7664888 -> 7664889 (aftercorr); disturbances 7664892, variants 7664893 (aftercorr)", 
        "status": "completed 2026-11-02 ~04:40 (3/3 exit 0): planning 13/7/4% success, 0% crashes 
        (D: 78/83/59%); fewer crashes in strong wind (34 vs 41% at 4 m/s); planner not better than
         policy. H2 not supported in this pilot", "outputs": 
        ["outputs/batch2/randomized_small_1M", "$SCRATCH/tdmpc2/batch2/randomized_small_1M"]},
    {"id": "HOR-TEST", "title": "Planning horizon changed at test time (1, 3 as trained, 5, 10 
        steps), nominal, 3 m/s wind, mass x1.15", "question": "Would looking further ahead help? 
        Does model error compound beyond the trained horizon?", "script": "eval_horizon.py via 
        cluster/horizon_job.sh; analyze_horizon.py", "seeds": "pilot: batch1_small_narval seed 4 
        (home, 2026-11-02 02:19); then all Batch 2 agents", "scheduler_job": "small 0-2 7668108, 
        small 3-10 7668109 (aftercorr:7662437), R1 7668110 (aftercorr:7659438), C1 7668130 
        (aftercorr:7659439), randomized 7668133 (aftercorr:7664889)", "code": "b86b66b", "status":
         "completed 2026-11-02: D best at 5 steps (+11 +- 7), R1 never better than 3 and crashes 
        grow with the horizon", "outputs": ["outputs/batch2/horizon_summary", 
        "outputs/batch1_small_narval/seed_04/horizon_test", 
        "$SCRATCH/tdmpc2/batch2/*/seed_*/horizon_test"]},
    {"id": "QUALITY", "title": "Flight quality from the saved trajectories: settling time, command
         jitter, saturation, worst flights", "question": "Mission experiment B metrics (action 
        smoothness, time to stabilise, tail performance); deployment (Q10)", "script": 
        "analyze_flight_quality.py (no new flights)", "status": "completed 2026-11-02 08:47 on all
         batches", "outputs": ["outputs/batch2/flight_quality"]},
    {"id": "SMOOTH-TEST", "title": "Smoother ways of acting and slower motors: planner as trained,
         averaged-plan action, 40 ms command filter, policy only; motor lag 0, 30, 60 ms", 
        "question": "Does the planner need its jumpy commands? Does precision survive slow motors 
        (deployment)?", "script": "eval_smoothing.py via cluster/smoothing_job.sh; 
        analyze_smoothing.py", "seeds": "D 0-10, R1 0-10", "code": "3f9f21b", "scheduler_job": 
        "small 0-2 7668917, small 3-10 7668921 (aftercorr:7662437), R1 7668922 
        (aftercorr:7659438)", "validation": "policy-only flights bit-identical to 
        DisturbedFlightTester; planner flights statistically equivalent (differ from step 2-72 by 
        one float32 rounding step, amplified by the planner's argmax; 4 noise seeds: success 
        66/66, 68/68, 66/68, 64/64%); filter step response correct", "status": "completed 
        2026-11-02 ~05:30: D: 40 ms filter +0 +- 1, averaged plan +1 +- 1; R1: filter -11 +- 3, 
        averaged plan +2 +- 2; R1 policy only unaffected by 60 ms motor lag", "outputs": 
        ["outputs/batch2/smoothing_summary", "$SCRATCH/tdmpc2/batch2/*/seed_*/smoothing_test"]},
    {"id": "SHARP-PILOT", "title": "Small TD-MPC2, random starts, extra reward 0.01 exp(-d^2 / (2 
        * 0.05^2)) near the target (pilot)", "question": "Is the checkpoint-to-checkpoint drift of
         hover precision (3-15 cm) caused by a reward that barely distinguishes 5 cm from 15 cm?",
         "config": "config_batch2_sharp_small (crazyflie_hover_sharp.py)", "seeds": "0-2", 
        "env_steps": 1000000, "code": "3f9f21b", "hardware": "A100 2/7 slice, two pieces of <= 3 
        h", "scheduler_job": "7668941 -> 7668942 (aftercorr), snapshots 7668947 (aftercorr)", 
        "validation": "bonus values exact; observations, termination and physics identical to the 
        plain task over 600 steps; reward = plain + bonus at every step; 1,500-step training run 
        with snapshots works", "status": "completed 2026-11-02 ~07:45 (3/3 exit 0): final planning
         37/66/23% success (D 78/83/59%), best-validated 91/78/47%; value probe: the value now 
        peaks at the target, but success still drifts between snapshots (SD 0.25 vs 0.34). 
        Hypothesis rejected in its simple form", "outputs": ["outputs/batch2/sharp_small_1M", 
        "$SCRATCH/tdmpc2/batch2/sharp_small_1M"]},
    {"id": "R1-2M", "title": "R1 seeds 0-2 continued from 1M to 2M steps (published TD-MPC2, 
        random starts)", "question": "Is the published model only slower to learn than the small 
        one, so that it catches up with more steps (Q9 accuracy, Q2 longer training)?", "config": 
        "config_batch2_random_start, resumed from the copied 1M checkpoints", "seeds": "0-2", 
        "env_steps": 2000000, "code": "622b694", "hardware": "A100 2/7 slice, three pieces of <= 3
         h (1.35M, 1.7M, 2M)", "scheduler_job": "copy 7669541 (aftercorr:7659438) -> 7669542 -> 
        7669543 -> 7669544 (aftercorr chain); final tests with DEPLOYMENT_TESTS in the last 
        piece", "validation": "copy and resume checked at home: a 1,500-step run copied and 
        resumed to 3,000 steps continues from 1,500, extends its learning curve and adds 
        snapshots", "status": "completed 2026-11-02 ~12:00 (3/3 exit 0). Paired 1M -> 2M, 
        planning: training area 66 -> 72%, held-out 65 -> 75%, outside 33 -> 24% with crashes 22 
        -> 58%. Likely cause: the 1M replay buffer discards early wide-ranging flights after 1M 
        steps (to check)", "outputs": ["outputs/batch2/random_start_2M", 
        "$SCRATCH/tdmpc2/batch2/random_start_2M", "/project/def-cbelling-ab/mounirz/tdmpc2_scratch
        _backup_2026-11-01/batch2/random_start_2M"]},
    {"id": "VALUE-PROBE", "title": "The agent's own value of hovering still at points within 30 cm
         of the target", "question": "Does the value network resolve small distances to the 
        target? (precision drift, reward pilot)", "script": "probe_value.py (no flights)", "runs":
         "D: 9 snapshots of seeds 2, 3, 5, 8, 10; reward pilot: 6 snapshots of seeds 0-2; R1: 
        seeds 3 and 7 at 1M", "hardware": "home CPU", "status": "completed 2026-11-02 08:50: flat 
        for D and R1 (drop <= 0.004 within 10 cm, on a value of about 1); peaked for the reward 
        pilot (0.03-0.05 at 10 cm, 0.07-0.13 at 20 cm)", "outputs": ["reports/RESEARCH_LOG.md 
        (03:55 and 08:45 entries)"]}
  ]
}
```
