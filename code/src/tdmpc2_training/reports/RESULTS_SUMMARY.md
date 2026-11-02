# Results summary: TD-MPC2 Crazyflie hover (status Nov 2, 2026)

Each answer is marked:
- **Established:** several seeds, the right controls, repeated measurement.
- **Preliminary:** few seeds or a single agent; direction clear, size uncertain.
- **Pending:** an experiment is running; see the handoff for job IDs.
- **Open:** not yet addressed.

Success = a full 10 s flight with a late error (mean distance over the last
2 s) below 10 cm. Numbers are mean ± standard error across seeds unless
stated.

## Q1. Why did short TD-MPC2 runs vary so much between seeds?

**Established.**
- The spread was real, not evaluation noise. Re-flying all 33 Batch 1 agents
  (50k steps) 20 times each shows that 93-94% of the flight-to-flight variance
  of the small agents comes from the agent itself (intraclass correlation
  0.93 and 0.94; between-seed SD 0.32 m vs within-seed SD 0.09 m). For the
  published-size 50k agents it is 52%, because they crash so often that
  single flights vary a lot (`outputs/batch2/variance_decomposition/`).
- The seeds had simply not finished learning at 50k steps: their learning
  curves were still improving, and the 3 published-size runs trained to 1M
  steps all converged to 5-8 cm.
- One Batch 1 "collapse" (home seed 0, 2.3 m) is a stable property of that
  final agent: all 20 re-flights end at 2.3 m.

## Q2. Does longer training resolve the variability?

**Established for the spread between seeds, not for the swings within a
run.**
- At 1M steps the three published-size runs agree (5.7-8.6 cm final).
- Within each run, test points kept swinging: 13-39% of the points after the
  run first hovered were worse than 10 cm.
- Those swings do not line up with any logged training loss: the correlation
  of step-to-step changes is between -0.14 and +0.10, n = 165-179 per run
  (`outputs/batch2/fluctuation_analysis/`). Low losses do not guarantee good
  flights (cf. Lambert et al. 2020).
- **Pending:** the snapshot evaluation flies every 100k-step snapshot of the
  Batch 2 runs from 150 starts with fixed noise. It will separate real
  checkpoint-to-checkpoint change from evaluation noise, and test whether the
  best checkpoint chosen on a separate validation set beats the final one.

## Q3. TD-MPC2 vs PPO: simulator steps and wall-clock time

**Established (Batch 1, single start).**
- TD-MPC2 first reached 10 cm after 105k-175k steps; PPO after 7.9M-19.7M
  steps, i.e. about 100x more.
- In wall-clock time PPO was faster (about 2.6 min vs 47 min to first reach
  10 cm) because it simulates 2,048 drones at once.
- Caveat: the historical PPO runs had two bugs (the learning rate never
  decayed; the automatic reset leaked the previous position and action into
  the next episode). They are kept as a historical baseline, not as a tuned
  comparison.

## Q4. How much GPU does TD-MPC2 use, and why is it slower in wall-clock time?

**Established.**
- One training step of the published TD-MPC2 on a Narval A100 2/7 slice:
  simulate one drone 6.8 ms, plan 10.2 ms, learning update 2.6 ms
  (overlapped), i.e. 49 steps/s.
- At home the GPU was busy 25-28% of the time. On Narval's slices the
  utilisation counter is not available.
- The cause is many small, sequential operations for a single drone, where
  launch overhead dominates the arithmetic.

## Q5. Can TD-MPC2 use parallel environments?

**Established as a measured projection (home GPU), Narval run pending.**
- Simulation parallelises perfectly: 128 drones cost as much as one (8-10 ms).
- But TD-MPC2 plans for every drone (10.6 ms for 1 drone, 735 ms for 128)
  and by default makes one learning update per simulated step (12.4 ms each).
- With 128 drones it would therefore train only about 1.7x faster (55 vs 32
  steps/s). With 1 update per 8 steps, 4.2x, but that is a different
  algorithm and needs its own sample-efficiency check.
- The projection for one drone (32 steps/s) matches the measured training
  speed (31).

## Q6. Can TD-MPC2 learn from continuously randomised starts and generalise?

**Preliminary (learning curves); final tests pending.**
- R1 (published size, 11 seeds) learns random starts.
  - Crash share from training-area starts fell from 77% at 100k to 1-14%
    from 200k to 325k.
  - The held-out block, never trained on, followed closely: about 10% at
    300k. Training on a region generalises inside it.
  - Starts outside the region remained hard: about 50% crashes at 325k.
- The single-start control C1 kept failing elsewhere: 40-60% crashes from
  training-area starts, 70-100% from the held-out block.
- **Pending:** final tests of R1 and C1 (128 flights per seed, with and
  without planning, small planner), at about 04:35.

## Q7. When and why does the learned world model help or hurt generalisation?

**Preliminary but consistent across 25 single-start agents. Batch 2 agents
pending.**
- For agents trained from one start, planning with the world model
  **hurts** off the training start.
  - 1M agents, crashes on training-area starts: planner 66% vs policy
    network 15%; held-out block 91% vs 54%.
  - The planner's flights visit states where the agent's own world model
    predicts 2.4-2.8x worse than along the policy network's flights from the
    same starts.
  - Model error in the last 0.5 s before a crash is 4-7x that of completed
    flights.
- This matches **model exploitation** (cf. Janner et al. 2019): optimising
  against a model off its data picks actions the model wrongly rates highly.
- Causal test on all 25 Batch 1 agents (same weights, starts and noise;
  only the planner settings change at test time):
  - Giving the planner half its candidates from the policy network
    reduced crashes on training-area starts by 19 ± 1 points and raised
    success by 15 ± 2 points for the 1M agents (all 3 seeds improved alike).
    For the small 50k agents (11 seeds) crashes fell by 3-7 points.
  - A narrower search helped the 1M agents less (-13 ± 1 crash points).
  - The policy network alone was still far more robust for the 1M agents
    (-49 ± 12 crash points vs the default planner).
  - So exploitation of model errors is one mechanism, not the only one; the
    value function's errors off the training data are the next suspect.
- **Pending:** the same tests on every Batch 2 agent (trained on random
  starts).

## Q8. Does planning improve robustness to disturbances compared with the policy alone?

**Preliminary (Batch 1 1M agents, 3 seeds): no.** Success from 50
training-area starts, planner vs policy only:

| Condition | Planner | Policy only |
| --- | --- | --- |
| none | 21% | 46% |
| gust 2 m/s | 17% | 41% |
| mass x1.15 | 9% | 36% |
| motors x0.9 | 12% | 40% |
| delay 40 ms | 12% | 41% |

Both fail completely at 2 m/s steady wind or 4 m/s gusts. These agents
never trained on these starts, so this mixes two shifts.

**Pending:** the same tests on the R1, C1 and small agents, which do know
the training area. Crashes are now split into hitting the ground vs being
blown out of bounds.

## Q9. Model size: speed, deployability, accuracy

**Speed established. Accuracy and generalisation preliminary, pending
11-seed results.**
- Time per decision (Narval A100 slice / 2 CPU cores):
  - published planner (512x6): 6.1 ms / 516 ms;
  - small planner (128x3): 1.4 ms / 11.4 ms;
  - policy network only: 0.37 ms / 0.20 ms (any size).
- The control period is 20 ms. The published planner therefore needs a GPU
  computer; the small planner nearly fits on a CPU.
- With random starts, the small model learned much faster than the
  published one at equal steps: 0% crashes in the training area and held-out
  block by 50k-100k, 0-20% outside, against about 50% for the published
  model at 325k.
- **Pending:** 11-seed final results of both sizes.

## Q10. What prevents safe sim-to-real transfer?

**Established, simulation and code audit only** (`DEPLOYMENT_READINESS.md`).
- **Blocker:** the agent outputs collective thrust plus body moments, which
  the Crazyflie firmware does not accept. The lab's old mapping (moments as
  rate setpoints clamped to 1 deg/s) is not equivalent.
- **Blocker:** thrust calibration.
- **Solved in simulation:** a deployment adapter builds the exact training
  observation from Crazyflie-style state estimates and was verified equal
  to the simulator over 300 closed-loop steps. The previous PPO hardware path
  skipped observation normalisation and used a different previous-action
  convention.
- **Open:** state estimation (Flow deck), take-off and landing hand-over,
  latency and jitter of the real radio loop.
