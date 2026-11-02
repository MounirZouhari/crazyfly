# Next research plan (after Batch 2, Nov 2, 2026)

Ordered by expected research value per unit of effort. Each step says which
result would change the plan. The reasoning behind each is in `GUIDE.md`,
Chapter 9.

## 0. Finish Batch 2 (today, no new work)

- Add the R1 continuation to 2M steps (seeds 0-2, `random_start_2M`) when it
  lands: pull with `cluster/pull_results.sh`, run `analyze_all.sh`, fill
  Section 5.18 of the guide.
- **Decision:** if R1 at 2M catches up with the small model (planning at 70%
  or more from the training area), the published model is just slower and
  longer runs are worth their cost; if not, carry the small model forward.

## 1. Make checkpoint selection on validation starts the standard (no training)

Every run already keeps snapshots. Report each run at its validation-chosen
snapshot as well as at its final one. This took the small model from 76% to
97% and the published policy network from 60% to 90%.

## 2. Find the source of the precision drift (about 2 h, no training)

The reward pilot ruled out the flat value function. Two cheap tests:
- fly one snapshot with several planner random seeds: if the hover position
  changes with the seed, the planner's sampling is the cause;
- lower the learning rate for the last 200k steps of one run: if the drift
  shrinks, the continual updates are the cause.

## 3. Tame the published planner (about 8 h on Narval per variant, 3 seeds)

Its 512x6 search loses to its own policy network and to a 64x2 search. Train
with a smaller search, or with more candidates from the policy network, and
compare with R1 at the same steps. Only after step 0.

## 4. Randomised physics with a memory (about 8 h, 3 seeds)

H2 lost precision, probably because the agent cannot infer which mass or wind
it is flying in. Add the last few observations and commands to the
observation (or an estimate of the wind and mass), then repeat H2 and its
disturbance tests. This is also what a real drone needs against wind.

## 5. Firmware-compatible commands (the gate to any hardware work)

Retrain with collective thrust plus body-rate setpoints, with the firmware's
rate controller modelled in the simulator, and repeat the Batch 2 tests. Then
the supervised steps of `DEPLOYMENT_READINESS.md` (motor identification,
radio-only test, tethered hover), each with Prof. Bellinger's approval.

## 6. Optional

- A fair PPO comparison (bugs fixed, random starts), if the thesis needs it.
- Parallel drones with fewer updates per collected step, reported as a
  variant of TD-MPC2.
