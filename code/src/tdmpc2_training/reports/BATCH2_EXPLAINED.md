% Batch 2 in plain words
% What was run overnight, what the names mean, what came out, and where everything is
% Mounir Zouhari, honours research, supervisor Prof. Colin Bellinger
% November 2, 2026 (results as of 09:00; one extra run still going)

[[TOC]]

# 1. The story in one page

**The goal of the project.** A Crazyflie drone in simulation must fly to a target point 1 m above the ground and hover there, within 10 cm. The controller is learned with **TD-MPC2**, a reinforcement learning method that learns a small model of how the drone moves (a "world model") and uses it to plan.

**What went wrong before (Batch 1).** In Batch 1, every training flight started from the **same spot**, (0, 0, 0.2). Most runs were short (50,000 steps). Professor Bellinger pointed out two problems: the results varied a lot between seeds, and a controller that always starts from one spot may just memorise one route instead of learning to fly.

**What Batch 2 changed.** Following his feedback:

- every training flight starts from a **new random spot** in a 1.5 m by 1.5 m area around the target;
- every run trains for **1 million steps** (20 times longer than most of Batch 1);
- every experiment has **11 seeds** (3 for the small control and pilot experiments);
- every agent is tested on the **same 128 test flights**, including spots it never trained from.

**The main answer.** Training from random spots works. The old single-spot agents crashed in most flights away from their one spot; the new agents almost never crash inside the area, and they also fly well from a block of spots they never trained from. The best agents reach the target (within 10 cm) in 90-97% of the flights from the training area.

**Two surprises.**

- The **small** version of TD-MPC2 learned better than the big (published) one in 1M steps. The big one is still improving at 1M, so three of its runs are being continued to 2M (finishing this afternoon).
- For the big model, the **policy network alone** (no planning) flies better than with planning, and at its best checkpoint it is as good as the best small agents, while needing only 0.3 ms per decision.

# 2. The words you need

## 2.1 Basic terms

- **Step.** One decision of the drone: it looks at its sensors, chooses a thrust and three turning forces, and the simulator moves it forward 0.02 s. **1M steps = 1 million decisions**, about 5.5 hours of simulated flying, made of thousands of practice flights. One run of 1M steps took about 7 hours of computing on Narval.
- **Seed.** One independent training run. The same experiment is repeated with 11 different seeds (random number choices) to see how much the result depends on luck. Results are averages over seeds.
- **Flight / test flight.** One 10-second flight from one starting spot, used only to measure, not to learn.
- **Success.** The drone flies the full 10 s and, over its last 2 s, stays on average **within 10 cm** of the target.
- **Crash.** The drone hits the ground or flies far away before the 10 s are over.
- **Checkpoint / snapshot.** A saved copy of the agent during training. Batch 2 saved one every 100,000 steps, so we can test the agent at different moments of its training, not only at the end.

## 2.2 The two ways a TD-MPC2 agent can fly

This distinction appears everywhere in the results:

- **With planning** (the normal TD-MPC2 way). At every step the agent imagines hundreds of possible short futures with its world model and picks the best one. Slower (11 ms per decision for the small model on a CPU, 516 ms for the big one), but can be more precise.
- **Policy network only.** The agent simply uses its policy network, which outputs an action directly, with no imagining. Very fast (0.2-0.4 ms per decision). TD-MPC2 trains this network anyway; we test it alone to see what planning adds.

There is also a third way, the **64×2 planner**: planning with a much smaller search (64 candidates, 2 rounds), which is faster.

## 2.3 The four groups of test flights

Every agent flies the same 128 test flights, from four groups of starting spots:

| Group | Number of flights | What it checks |
|---|---:|---|
| **training area** | 50 | random new spots of the kind it trained on |
| **held-out block** | 30 | spots inside the area but in a corner it was **never** allowed to train from: can it fill in the gap? |
| **outside** | 30 | spots 0.25-0.75 m **outside** the area: can it go beyond what it saw? |
| **Batch 1 spots** | 18 | the six old Batch 1 starting spots, to compare with before |

Every result table below gives **success / crash** shares for these groups.

## 2.4 Model sizes

- **Published (big) model:** the TD-MPC2 settings from the paper (5 million weights).
- **Small model:** a reduced version (0.39 million weights) with a smaller planning search. It trains about twice as fast and can plan on a laptop CPU.

# 3. The experiments and their names

Every experiment has a short name. This table is the key to everything else:

| Name | Model | Where training flights start | Seeds | Steps | Why it was run |
|---|---|---|---:|---:|---|
| **B1** (Batch 1, old) | big and small | always the same spot | 11 (+3 big at 1M) | 50k (1M for 3) | the "before" picture |
| **R1** | big | random spots | 11 | 1M | **the main experiment**: the professor's suggestion with the published TD-MPC2 |
| **C1** | big | always the Batch 1 spot | 3 | 1M | a **control** for R1 (see below) |
| **D** | small | random spots | 11 | 1M | the same as R1 with the small model |
| **H2** | small | random spots, plus random weight, motor strength and wind | 3 | 1M | pilot: does training with varied physics make it more robust? |
| **SH** | small | random spots, with an extra reward near the target | 3 | 1M | pilot: does a sharper reward make it more precise? |
| **R1 to 2M** | big | random spots | 3 | 2M | is the big model just slower? **still running** |

**Why C1 exists.** Besides the random spots, R1 has one small technical difference from Batch 1 (the target position is given to the agent in metres). If R1 is better than Batch 1, is that because of the random spots, or because of this small difference? C1 has the small difference but trains from the single Batch 1 spot. If C1 looks like Batch 1, the improvement in R1 comes from the random spots. (It does: C1 is within 2-6 points of Batch 1.)

**Folder names.** On disk the experiments have longer names:

| Name | Folder in `outputs/batch2/` |
|---|---|
| R1 | `random_start_1M` |
| C1 | `fixed_start_1M` |
| D | `random_start_small_1M` |
| H2 | `randomized_small_1M` |
| SH | `sharp_small_1M` |
| R1 to 2M | `random_start_2M` (on Narval until it finishes) |

# 4. The results, one question at a time

All numbers are **success / crash** shares, averaged over the seeds, for the final agents (after 1M steps) unless said otherwise.

## 4.1 Do random starts help? Yes, clearly.

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

## 4.2 Which model is better? The small one, for now.

| With planning | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| R1 (big) | 46% / 0% | 39% / 2% | 20% / 26% |
| D (small) | 78% / 0% | 83% / 0% | 59% / 3% |

The small model is 32-44 points better. But the big model's results were **still going up** at the end of training (its test success rose from about 26% at 600k steps to 42% at 1M), so it may simply need more steps. That is why three R1 runs are being continued to 2M steps.

## 4.3 Does planning help? It depends on the model.

| | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| D (small), with planning | 78% / 0% | 83% / 0% | 59% / 3% |
| D (small), policy network only | 9% / 0% | 9% / 0% | 12% / 0% |
| R1 (big), with planning | 46% / 0% | 39% / 2% | 20% / 26% |
| R1 (big), policy network only | 60% / 0% | 64% / 0% | 40% / 6% |

- **Small model:** planning is essential. The policy network alone never crashes but usually stops about 20 cm from the target; planning brings it to about 7 cm.
- **Big model:** planning makes it **worse**, especially outside the area, where it crashes 26% of the time against 6% without planning. A smaller search (64×2) is also better than the full one. This fits a known weakness of planning with a learned model, called **model exploitation**: a strong search finds the places where the model is wrong in an optimistic way, and goes there. Measurements support it: along the planner's flights the big model's predictions are 1.2-2 times more wrong than along the policy network's flights, and restraining the search adds 8-13 points.

## 4.4 Keeping the best checkpoint instead of the last one

During training, precision drifts by a few centimetres from one checkpoint to the next, and since the success line is at 10 cm, a few centimetres can flip many flights. So each run's checkpoints were tested on 40 separate **validation** flights, and the best one was kept. Its results on the test flights:

| Best checkpoint (chosen on validation) | Training area | Held-out block | Outside |
|---|---:|---:|---:|
| D (small), with planning | **97% / 0%** | **91% / 0%** | **70% / 2%** |
| R1 (big), policy network only | **90% / 0%** | **81% / 0%** | **59% / 6%** |
| R1 (big), with planning | 49% / 0% | 46% / 1% | 26% / 32% |

These are the two best controllers of the project so far. The second one is notable for a real drone: it needs no planning at all, so it decides in 0.3 ms on any computer.

## 4.5 Disturbances (wind, weight, weaker motors, delay)

Tested on the 50 training-area flights:

- **Extra weight (+5 to +15%), weaker motors (-5 to -10%), radio delay (20-40 ms):** the small model with planning barely changes (72-77% success, against 77% without disturbance).
- **Wind:** 1 m/s still works (60%). From 2 m/s on, no agent stays within 10 cm, but up to 3 m/s the small model is pushed away without crashing. At 4 m/s many flights crash.
- **Big model (R1):** the planner is worse than its policy network under every disturbance; in 3 m/s wind the planner crashes in 32% of the flights, the policy network in 0%.

## 4.6 The two pilots (3 seeds each, so only indications)

**H2, training with varied physics.** Every training flight had a random weight, motor strength and wind. Result: a little more robust in strong wind (fewer crashes at 4 m/s: 34% against 41%), but much less precise in calm air (13% success against 78% for D). The likely reason: the agent sees only its current state, so it cannot tell which weight or wind it is flying in, and its world model learns an average that is wrong in every particular case. Not a good trade in its current form.

**SH, a sharper reward near the target.** The lab's reward pays almost the same for hovering 5 cm or 15 cm from the target, which could explain why precision drifts. SH adds a reward that is concentrated within about 5 cm. Results: the agent did learn it (its value network now clearly prefers the target, which the normal agents do not), and the big precision swings disappeared, but the precision still drifts around the 10 cm line, and the final checkpoints happened to be worse (37% from the training area; 91% at the best checkpoint). So the reward was **not** the main cause of the drift. Useful to know, but not a fix.

## 4.7 Smooth commands for a real drone

With planning, the commands jump a lot from one step to the next (thrust changes by about 30% of its range every step), which real motors cannot follow. Two simple fixes were tested:

- For the **small model**, smoothing the commands over 40 ms keeps exactly the same success (79%) and makes them as smooth as the policy network's.
- Using the planner's **averaged plan** instead of its single best candidate keeps the success for both models (+1 to +2 points) and removes the commands stuck at their limits.

## 4.8 Speed

| Decision time | GPU (Narval) | CPU (2 cores) |
|---|---:|---:|
| big model, planning | 6.1 ms | 516 ms (too slow for 20 ms control) |
| small model, planning | 1.4 ms | 11.4 ms |
| policy network only | 0.3 ms | 0.2 ms |

A real drone needs a decision every 20 ms. The big model can only plan in time with a GPU; the small model can on a laptop; the policy network alone can anywhere.

# 5. What it means, and what to tell Professor Bellinger

1. **His suggestion worked.** Training from random spots turns a controller that memorised one route (and crashed elsewhere) into one that flies from anywhere in the area, including a corner it never trained from. C1 proves it is the random spots that matter.
2. **Longer training fixed safety, not all of the precision.** At 1M steps no good agent crashes inside the area, but precision still drifts by a few centimetres between checkpoints. Keeping the best checkpoint (chosen on separate validation flights) solves this in practice: 97% success for the small model.
3. **When planning helps.** Planning helps when the world model is accurate where it is used (small model, inside the area: +69 points). It hurts when the search can find the model's mistakes (single-spot agents away from their spot; the big model outside the area).
4. **The small model is the better choice for now.** Better results in 1M steps, twice as fast to train, and it can plan on a laptop. Whether the big model catches up with more training is being tested (R1 to 2M).
5. **For a real drone:** the commands must be smoothed (a 40 ms filter or the averaged plan), and the agents must be retrained with commands the Crazyflie firmware accepts; nothing was flown.
6. **Two negative pilot results** (varied physics, sharper reward) are reported as they are; both say where to look next.

# 6. Where everything is

## 6.1 Opening the right folder

All the Batch 2 work is on your computer, inside WSL, on the git branch **`tdmpc2-hover`**. It is **not on GitHub**: pushing was refused (error 403), so GitHub and the `main` branch only show the old students' code. In VS Code: **File, Open Folder**, then paste

```
\\wsl.localhost\Ubuntu\home\mounirzouhari\Projects\multi-agentCrazyflieLift
```

The bottom-left corner of VS Code must show the branch `tdmpc2-hover`.

## 6.2 The folders

| What | Where |
|---|---|
| **The TD-MPC2 code (new)** | `code/src/tdmpc2_training/` |
| The documents (this one, the full guide, summaries, log) | `code/src/tdmpc2_training/reports/` |
| The figures | `code/src/tdmpc2_training/reports/figures/` |
| The Batch 2 results (numbers, flight records, plots) | `outputs/batch2/` |
| The old students' code (untouched) | `code/src/gpu_training/`, `code/src/hardware_model_testing/`, `code/src/initial_cpu_training/` |
| The trained agents (too big to copy home) | on Narval: `/scratch/mounirz/tdmpc2/batch2/` |

## 6.3 How a results folder is organised

Each experiment folder in `outputs/batch2/` (for example `random_start_1M/` for R1) contains:

- `analysis/summary.md`: **the one file to read**: all tables for that experiment (final results, comparisons, disturbances) in plain text;
- `analysis/*.png`: its plots;
- `seed_00/` to `seed_10/`: one folder per seed, with the raw records of every test (`evaluation/` with planning, `evaluation_policy_only/`, `disturbances/`, `snapshot_evaluation/`, and so on). Each has a `summary.csv` (one line per test flight) and sometimes `trajectories.csv` (every step of every flight). You do not need to open these; the summaries are built from them.

The summaries that compare experiments are in `outputs/batch2/*_summary/` (for example `snapshot_summary/` for the best-checkpoint results, `smoothing_summary/` for the smooth commands).

## 6.4 The documents

| Document | What it is |
|---|---|
| `BATCH2_EXPLAINED.pdf` | this document: the short, plain version |
| `GUIDE.pdf` | the full textbook-style guide: every experiment in detail, every code file explained, the answers to the professor's ten questions (being finished now) |
| `RESULTS_SUMMARY.md` | one short answer per professor question |
| `RESEARCH_LOG.md` | what was done when, in order, with every number |
| `DEPLOYMENT_READINESS.md` | what stands between these agents and a real Crazyflie |

# 7. What is still running and what is left

- **Running on Narval:** R1 to 2M (3 seeds), done around 12:30-13:30. It answers whether the big model just needs more training.
- **Being written:** the full `GUIDE.pdf` with all the final results (about 11:00-11:30), a one-page update for Professor Bellinger, and the handoff notes.
- **Then:** back up all results to Narval's permanent storage (`/project`, because `/scratch` is cleaned automatically), and add the 2M result when it lands.
