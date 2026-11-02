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
| Hor. | planning horizon 1, 3, 5, 10 at test time | all Batch 2 agents | would looking further ahead help? |
| Smooth. | smoother ways of acting, slower motors | R1 and D agents | does the planner need its jumpy commands? |
| Sharp | small model with an extra reward near the target | 3 × 1M | is the drift of precision caused by the reward? |

The last three rows were added during the night, after the first results had raised the questions they answer (Sections 5.13, 5.14 and 5.16).

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

## 5.1 How to read this chapter

Every section follows the same pattern: the question, how it was tested, what came out, and what it means. Tables give **success / crash** shares as mean ± standard error across seeds; each seed is first averaged over its own flights, so every seed counts equally. Differences between two decision rules are **paired**: the same agents, the same starts and the same sensor noise, so that only the decision rule differs. Where a difference matters, its 95% bootstrap confidence interval is given in square brackets.

The sections go from the past to the present. Sections 5.2 to 5.4 look back at Batch 1 with new tests. Sections 5.5 to 5.8 present the main Batch 2 results: learning to hover from anywhere. Sections 5.9 to 5.14 explain when and why the world model helps or hurts, and what limits the precision. Sections 5.15 and 5.16 look at how the drone flies, beyond success and crashes, which matters for a real drone. Section 5.17 covers speed.

## 5.2 Was the Batch 1 spread between seeds real? (A1)

**The question.** In Batch 1, the 11 small agents trained on Narval ended anywhere between 6.5 cm and 1.05 m from the target. Each agent had been tested with only 3 flights. There were two possible explanations. Either the agents really were different, because training with different seeds had produced controllers of different quality; or they were similar, and each test flight was so noisy (because of the sensor noise) that 3 flights could not tell them apart. The professor found the spread unusual for TD-MPC2, so it was important to know which explanation was right.

**How it was tested.** Every one of the 33 Batch 1 final agents (11 small agents from Narval, 11 small agents from the home PC, 11 published-size agents at 50k steps) flew 20 more flights from its training start: 5 different sensor-noise seeds with 4 flights each, the same noises for every agent. The spread of the late errors was then split into two parts with a one-way analysis of variance: the part between seeds (how different the agents are) and the part within seeds (how much the same agent varies from flight to flight). The **intraclass correlation (ICC)** is the share of the first part in the total.

| Agents | Between-seed SD | Within-seed SD | ICC |
|---|---:|---:|---:|
| small, Narval (11) | 0.32 m | 0.09 m | 0.93 |
| small, home (11) | 0.63 m | 0.16 m | 0.94 |
| published, 50k (11) | 0.38 m | 0.36 m | 0.52 |

![Each grey dot is one of the 20 re-flights of an agent, each blue dot that agent's mean. For the small agents (left and middle), the dots of one agent sit tightly together while the agents differ widely: the spread is a property of the agents. The published-size 50k agents (right) crash so often that each agent's flights also vary a lot.](code/src/tdmpc2_training/reports/figures/a1_seed_variance.png)

**What it means.** For the small agents, 93-94% of the spread comes from the agents themselves. The test was not the problem; the training was. One agent from the home PC (seed 0) ends 2.3 m from the target in all 20 re-flights: it is a stable failure of that agent, not bad luck. Together with the learning curves, which were still improving at 50k steps, and with the three 1M-step runs, which all converged to between 5.7 and 8.6 cm, this supports the professor's suspicion: at 50k steps, training had not finished, and each seed was caught at a different point of its learning. This is why every Batch 2 run trains for 1M steps.

## 5.3 Do the training losses explain the swings within a run? (A2)

**The question.** Even the 1M-step runs did not improve smoothly. After a run first hovered within 10 cm, between 13% and 39% of its later test points were worse than 10 cm again. TD-MPC2 logs several training statistics (the losses of the world model, the value and the policy). If these statistics rose whenever the flying got worse, they could be used to detect bad moments and stop training at a good one.

**How it was tested.** For each of the three 1M-step runs, every test point after the run first hovered (165 to 180 points per run) was paired with the average of each logged statistic over the 5,000 steps before it. A **Spearman correlation** then measured whether changes in the test error go together with changes in the statistic, and a **permutation test** checked whether the correlation could be due to chance.

**The result.** None of the statistics tracks the swings: all correlations between changes lie between -0.14 and +0.10, and none is statistically significant (the smallest permutation p-value is 0.075, even before correcting for having tested 18 correlations).

![Test error of the three 1M-step runs during training, with the logged training statistics. The swings in the test error have no counterpart in the losses.](code/src/tdmpc2_training/reports/figures/a2_fluctuations.png)

**What it means.** The losses measure how well the networks predict, not how well the drone flies. A world model can predict its training data slightly better or worse without the controller changing in any visible way, and the controller can get worse without the losses noticing. This is known as **objective mismatch** (Lambert et al., 2020). The practical consequence is that the only reliable way to know how good an agent is, is to fly it. Batch 2 therefore keeps a snapshot every 100k steps and has a separate set of validation starts for choosing a checkpoint (Section 5.12).

## 5.4 The Batch 1 agents on the new test starts (R4)

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

## 5.5 Learning to hover from anywhere: the published model (R1)

**The question.** Can TD-MPC2 learn to reach the target from any start in a region, rather than memorising one route? And does what it learns carry over to starts it never trained from: the held-out block inside the region (interpolation) and the starts outside it (extrapolation)? This is the professor's central suggestion and the main experiment of Batch 2.

**How it was tested.** Eleven seeds of TD-MPC2 with the published settings were trained for 1M steps each, every training flight starting at a new random position in the training area (Section 4.1). During training, 21 test flights every 25k steps recorded the learning curves. At the end, every agent flew the 128 fixed test flights (Section 4.3) three times: with planning as trained, with the policy network only, and with a small 64×2 planner, all with the same sensor noise.

*Results pending.*

## 5.6 The single-start control (C1)

**The question.** R1 differs from the Batch 1 agents in two ways: random starts, and the target offset given in metres. Which of the two makes the difference? C1 answers this: it is identical to R1 (same settings, same metric offset, same 1M steps) except that every training flight starts at the Batch 1 start (0, 0, 0.2).

**How it was tested.** Three seeds, trained and tested exactly like R1.

*Results pending.*

## 5.7 The small model (D)

**The question.** The published model needs a GPU to plan in real time (6.1 ms per decision on a GPU slice, 516 ms on a CPU). The small model plans in 1.4 ms on a GPU slice and 11.4 ms on a CPU, which would fit a laptop next to the drone. Does it learn random starts as well as the published model?

**How it was tested.** Eleven seeds of the small settings (Section 2.4), trained and tested exactly like R1. Because the small model trains faster, each run was split into two jobs of 3 hours, the second resuming from the first's checkpoint. Three seeds were run first as a pilot; the other eight were launched after the pilot had shown at 200k steps that learning worked.

*Results pending.*

## 5.8 With and without planning: where the precision comes from

**The question.** Sections 5.5 to 5.7 compare planning with the policy network alone in terms of success and crashes. This section looks at the flights themselves: when the drone does not crash, how close to the target does it end, in which direction is it off, and is the conclusion tied to the 10 cm threshold?

**How it was tested.** The script `analyze_precision.py` takes the full flights (no crash) from the training-area and held-out starts and splits each one's late error into its horizontal part and its vertical part, and computes the height offset (above or below the target). It also plots, for every threshold from 0 to 50 cm, the share of flights that end within it, counting crashes as failures. This last figure shows the whole picture at once, without choosing a threshold.

*Results pending.*

## 5.9 Why planning hurts single-start agents (H1)

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

**What it means.** Both tests support model exploitation as one real mechanism: the planner's flights go where the model is wrong, crashes follow large model errors, and restraining the search reduces crashes. But restraining the search does not bring the planner down to the policy network's crash rate (45% against 15%), so exploitation is not the whole story. The next suspect is the value function, which judges everything beyond the 3-step horizon and has also only been trained near the single start. The decisive question is what happens when the training data covers the test situations, which is what Batch 2 provides (Section 5.8).

## 5.10 Disturbances

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

*Results pending.*

## 5.11 Training on varied physics (H2)

**The question.** Section 5.9 suggests that planning hurts when the world model is used outside the data it was trained on. The opposite hypothesis, called **H2** here, is that planning should help when the world model has been trained on the relevant variation. Disturbances offer a direct test: if the training flights already contain varied mass, motor strength and wind, the world model learns how these change the drone's motion, and planning with it might then cope better with them than the policy network alone.

**How it was tested.** Three seeds of the small model were trained like D, except that every training flight also drew, at random, a mass and inertia between ×1.0 and ×1.15, a motor strength between ×0.9 and ×1.0, and a steady wind of 0 to 3 m/s from a random direction (`crazyflie_hover_dr.py`). The agent is never told these values; it can only infer them from how the drone moves. This is known as **domain randomisation** (Peng et al., 2018). The agents were then tested with the same disturbance tests as D. Disturbances within these ranges test interpolation; 4 m/s wind, 6 m/s gusts and command delays lie outside them. With 3 seeds this is a pilot.

*Results pending.*

## 5.12 Snapshots: change during training, and the best versus the final checkpoint

**The question.** The learning curves of Batch 1 and of Batch 2 swing from one test point to the next. Two questions follow. Are these swings real changes of the agent, or noise of the test flights? And would it be better to keep the best checkpoint of a run than the last one?

**How it was tested.** Every Batch 2 run kept its agent every 100k steps (10 snapshots). The script `eval_snapshots.py` flies every snapshot from the 40 validation starts and the 110 test starts, with planning and with the policy network only, always with the same sensor noise, so that differences between snapshots can only come from the agents. The best snapshot of each run is chosen on the validation starts alone; its test results are then compared with those of the final snapshot. Choosing on separate validation starts matters: choosing on the test starts would make the "best" snapshot look better than it really is.

*Results pending.*

## 5.13 A sharper reward near the target (pilot)

**The question.** Section 5.12 shows that the precision of an agent drifts between checkpoints, while it never crashes. A look at the reward suggests why. The lab's reward per step is 0.001 for staying in the air, plus 0.01 × (1 - tanh d) for being at a distance d from the target, plus the reduction of the distance since the previous step, and -1 for a crash. Hovering at 5 cm from the target earns 0.0105 per step; hovering at 15 cm earns 0.0095. The difference, 0.001 per step, is about a tenth of the value of hovering at all, and in TD-MPC2's value encoding (101 bins over a symmetric logarithmic range from -10 to 10, so bins of width 0.2) it is a quarter of a single bin. Within the 3-step planning horizon, the two rewards differ by only 0.003. The agent therefore has very little reason to prefer 5 cm over 15 cm, which could explain why its precision wanders. This is a hypothesis; the experiment below tests it.

**How it was tested.** Three seeds of the small model were trained exactly like D, with one change: an extra reward term concentrated near the target, 0.01 × exp(-d² / (2 × 0.05²)), worth 0.01 at the target, 0.006 at 5 cm and almost nothing beyond 15 cm (`crazyflie_hover_sharp.py`). Hovering at 2 cm instead of 15 cm now earns about 0.010 more per step instead of 0.001. Physics, observations, starts and tests are unchanged, and the test flights measure distances, so the results compare directly with D. The comparison of interest is the drift between snapshots and the final precision. This is a change of the task's reward, not of TD-MPC2, and with 3 seeds it is a pilot.

*Results pending.*

## 5.14 The planning horizon

**The question.** TD-MPC2 plans only 3 steps (0.06 s) ahead and leaves everything after that to its value network. Would looking further ahead help? Or would the world model's errors pile up over a longer horizon, since it was trained to predict only 3 steps?

**How it was tested.** The script `eval_horizon.py` flies the same agents with planning horizons of 1, 3 (as trained), 5 and 10 steps, from the 50 training-area and 30 outside starts, without disturbance, in a 3 m/s steady wind and with 15% extra mass, always with the same sensor noise. Only the horizon used at test time changes; the agents are not retrained.

*Results pending.*

## 5.15 How the drone flies: settling, smoothness and the worst flights

**The question.** Success and crashes say where a flight ends, not how it got there. For a real drone, three more things matter: how long it takes to settle near the target, how smooth its commands are (motors cannot change their speed instantly, and jumpy commands heat them and shake the frame), and how bad the worst flights are. The mission for this project asked for all three; none of them needed new flights, because every test already saved every step of every flight (`trajectories.csv`).

**How it was measured.** The script `analyze_flight_quality.py` reads the step-by-step records of the final tests from the training-area and held-out starts and computes, for each agent:

- the **settling time**: the moment after which the drone stays within 10 cm of the target until the end of the flight;
- the **command jitter**: how much each command changes from one step to the next during the last 5 seconds of a full flight, when the drone should simply be holding its position, in percent of the command's full range;
- the **saturation**: the share of steps at which a command is at its limit (no thrust, full thrust, or a body moment at its maximum);
- the **tail**: the 90th percentile and the worst late error of the full flights.

*Results pending.*

## 5.16 Smoother commands and slower motors

**The question.** Section 5.15 shows that the planner's commands jump from step to step. Why, and does it matter? The reason is in how TD-MPC2 acts. At each step, its planner samples many candidate plans, scores them with the world model, and executes the first action of the single best one. Because each step's best candidate is a new random sample, and because many samples are clipped at the command limits, the executed action jumps around even when the drone is hovering calmly. The planner also computes a smoother quantity, the weighted average of its best candidates (its **averaged plan**), but only uses it to start the next step's search. Two questions follow. Does the planner need its jumpy actions to be precise, or would a smoother way of acting do as well? And does the precision survive if the motors cannot follow the commands instantly, as real motors cannot?

**How it was tested.** The script `eval_smoothing.py` flies the same agents, from the same 80 training-area and held-out starts with the same sensor noise, in four ways: the planner as trained; the first action of the planner's averaged plan; the planner's commands passed through a low-pass filter with a time constant of 40 ms (each command moves only part of the way towards the new one); and the policy network alone. Each is flown with instant motors, as in training, and with motors that follow the command with a first-order lag of 30 or 60 ms. These lags were chosen to test sensitivity; the real Crazyflie's motor response has not been measured in this project, so they are assumptions, not identified values. Nothing is retrained; only the way the trained agent acts, or the simulated motors, change.

*Results pending.*

## 5.17 Speed: decision time, training time and parallel drones

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

# 6. Answers to Professor Bellinger's questions

This chapter answers the ten questions that were drawn from Professor Bellinger's feedback. Each answer starts with the short version, then gives the evidence (with the section of Chapter 5 where it is shown) and says how certain it is: **established** means several seeds, the right controls and a repeated measurement; **preliminary** means few seeds or a single setting, so the direction is clear but the size is uncertain.

## Q1. Why did short TD-MPC2 runs vary so much between seeds?

*Completed with the final results.*

## Q2. Does longer training resolve the variability?

*Completed with the final results.*

## Q3. How do TD-MPC2 and PPO compare in simulator steps and wall-clock time?

**Short answer.** TD-MPC2 needs about 100 times fewer simulator steps than PPO, but PPO is faster in real time, because it simulates 2,048 drones at once while TD-MPC2 simulates one.

**Evidence (Batch 1, single start, established).** TD-MPC2 first reached 10 cm after 105k to 175k steps; PPO after 7.9M to 19.7M steps. In real time, PPO got there in about 2.6 minutes and TD-MPC2 in about 47 minutes. **Caveat:** the PPO runs had the two bugs described in Section 3.3 (a learning rate that never decreased, and a reset that leaked the previous flight's position and action). They are a historical baseline, not a tuned comparison, so the factor of 100 is indicative rather than exact.

**What to say if asked more.** Which matters depends on the setting. On a real drone, every step is real flight time, so sample efficiency matters most and TD-MPC2's advantage is decisive. In simulation, where steps are almost free, PPO's parallelism wins.

## Q4. How much of the GPU does TD-MPC2 use, and why is it slower in wall-clock time?

**Short answer.** Very little. With one drone, each training step consists of many small operations performed one after the other, and the time goes into starting them rather than into computing.

**Evidence (established, Section 5.17).** On a Narval GPU slice, one training step of the published model takes 6.8 ms to simulate the drone, 10.2 ms to plan its action and 2.6 ms for the learning update: 49 steps per second. At home the GPU was busy only 25-28% of the time; on Narval's GPU slices the usage counter is not available, so the measured times are the evidence instead. The planner is a good example: 13 times more arithmetic costs only about 3 times more time, which means the arithmetic is not the bottleneck.

## Q5. Can TD-MPC2 benefit from parallel environments?

**Short answer.** Only modestly in its standard form: 1.6 times faster for the published model and 2.9 times for the small one with 128 drones, according to a projection from measured timings.

**Evidence (established as a projection, Section 5.17).** Simulating 128 drones costs the same as one, but TD-MPC2 must still plan for every drone at every step, and it makes one learning update for every step collected. These two costs grow with the number of drones and cap the speed (74 steps per second for the published model, 271 for the small one). The projection reproduces the measured one-drone speeds, so it is reliable. Faster training would require fewer learning updates per collected step; that is a different algorithm, and its effect on sample efficiency would have to be measured before it could be called an optimisation.

## Q6. Can TD-MPC2 generalise from continuously randomised starting positions?

*Completed with the final results.*

## Q7. When and why does the learned world model improve generalisation?

*Completed with the final results.*

## Q8. Does planning improve robustness compared with the policy network alone?

*Completed with the final results.*

## Q9. How does model size affect inference speed and deployability?

*Completed with the final results.*

## Q10. What prevents safe transfer to a real Crazyflie?

*Completed with the final results.*

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
- **`eval_smoothing.py`**: flies the same agent with the planner as trained, with the first action of the planner's averaged plan, with the planner's commands through a low-pass filter, and with the policy network alone, each with instant motors and with slower motors (`SmoothedFlightTester`).
- **`model_error.py`**: records flights and measures how wrong the agent's own world model is along them.

## 7.6 Analysis

- **`analyze_batch2.py`**: the main Batch 2 analysis: learning curves per test group, final results per group and decision rule, comparisons between experiments with bootstrap confidence intervals, disturbance results and crash causes.
- **`analyze_snapshots.py`**, **`analyze_model_error.py`**, **`analyze_planner_variants.py`**, **`analyze_horizon.py`**, **`analyze_smoothing.py`**: summaries of the corresponding tests across seeds.
- **`analyze_precision.py`**: splits the late error of full flights into its horizontal and vertical parts and the height offset, per decision rule, and plots the share of flights ending within any distance from 0 to 50 cm, so that no conclusion depends on the 10 cm threshold.
- **`analyze_flight_quality.py`**: from the step-by-step trajectories, the time to stabilise, the jitter of the commands, how often a command is at its limit, and the worst flights.
- **`analyze_start_outcomes.py`**: the map of the test starts coloured by how many seeds succeeded from each.
- **`analyze_all.sh`**: runs every analysis above on the results copied home, skipping whatever is not finished yet.
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
- `eval_job.sh`, `disturb_job.sh`, `snapshot_job.sh`, `model_error_job.sh`, `variants_job.sh`, `horizon_job.sh`, `smoothing_job.sh`: run the corresponding test on a list of runs, one per array task.
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
| `disturbances*/`, `snapshot_evaluation*/`, `model_error/`, `planner_variants/`, `horizon_test/`, `smoothing_test/` | the follow-up tests |
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

# analysis: everything at once, or one analysis at a time
bash code/src/tdmpc2_training/analyze_all.sh
.venv/bin/python code/src/tdmpc2_training/analyze_batch2.py outputs/batch2/random_start_1M \
    --compare outputs/batch2/random_start_small_1M outputs/batch2/fixed_start_1M outputs/batch1_standard_1M

# this guide
~/.local/bin/uv run --no-project --with reportlab python code/src/tdmpc2_training/reports/make_guide.py
```

# 8. Limitations

Every result in this guide comes with conditions. This chapter collects them in one place, so that no result is stretched beyond what it supports. They fall into four groups: the simulation, the task, the statistics, and the comparisons.

## 8.1 The simulation

All results are from simulation; no real drone was flown. The simulator is a rigid-body model: the agent's thrust and body moments are applied directly to the drone's body, so the motors respond instantly and perfectly (only the delay test adds a lag). There is no model of the individual propellers, no ground effect near the floor, and no battery that runs down during a flight. Air resistance comes from MuJoCo's simple fluid model of a box-shaped body; wind is a uniform movement of the whole air mass, steady or as one 1-second gust, not the turbulent, changing wind of a real room or outdoors. Sensor noise is simple random noise added to perfect measurements; a real Crazyflie estimates its state with a filter, whose errors are correlated in time and can drift. Each of these differences could change how a controller behaves on a real drone, which is why Chapter 6 treats real-world readiness separately (question 10).

## 8.2 The task

The task is deliberately narrow: one drone, one target point one metre above the origin, starts at rest and level. The held-out block tests interpolation only in the horizontal plane (it spans all training heights). The outside starts are at most 0.75 m beyond the training area, so they test moderate extrapolation, not arbitrarily distant starts. A controller that does well here has learned to recover from a displacement, not yet to follow a path, take off, land, or fly with other drones.

## 8.3 The statistics

- **Seeds.** The main experiments (R1 and D) have 11 seeds each, enough for the averages and confidence intervals reported. C1 and H2 have 3 seeds; their results show a direction, and their sizes should be read as preliminary.
- **Test starts.** Each agent is flown once from each of 50, 30 and 30 fixed starts. A share from one agent therefore moves in steps of 2-3 percentage points, and the reported standard errors (across seeds) do not include the uncertainty from the particular choice of starts. Using the same starts for every agent makes comparisons paired and precise, at the price of this small blind spot.
- **Thresholds.** Success means a late error below 10 cm. This threshold is a choice; the full distributions of late errors are saved, and conclusions that would change with a different threshold are pointed out where they occur.
- **Repeatability.** Training on a GPU is not exactly repeatable: two runs with the same seed drift apart slightly once learning starts (Section 4.6). Results are therefore statements about the distribution of agents a configuration produces, which is what the seeds measure, not about one exact agent.

## 8.4 The comparisons

- **Test-time changes.** The planner variants and the horizon test change the planner only when testing agents trained with the default settings. An agent trained with a longer horizon, or with more candidates from its policy network, might behave differently; these tests show what the trained agents' models can support, not what the best training settings are.
- **Hyperparameters.** TD-MPC2 was used with its published settings (and the small settings from September), without tuning. A tuned configuration might change the sizes of the effects, especially for the small model.
- **PPO.** The Batch 1 comparison with PPO used the lab's PPO with the two bugs described in Section 3.3. It was not repaired or re-run, so it is a historical baseline, not a fair tuned comparison. In particular, PPO's late degradation is likely caused by its learning rate never decreasing.
- **Speed.** Times were measured on a 2/7 slice of an A100 shared with other users' jobs on the same card, and at home on an RTX 3060; other hardware will give other numbers. The parallel-drone speeds are projections from measured timings, not full training runs, and they do not measure how many environment steps a parallel version would need to learn.
- **World-model error.** The model error is measured in the agent's own internal (latent) space. It is meaningful for comparing flights of the same agent (planner versus policy network, before a crash versus a completed flight), less so for comparing different agents.

# 9. What to do next

*This chapter is completed once all results are in.*

# Appendix A. Glossary

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

# Appendix B. Jobs and code versions

*Completed with the final version.*
