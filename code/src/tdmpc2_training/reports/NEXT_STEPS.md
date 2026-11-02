# Next research plan (after Batch 2, Nov 2, 2026)

Ordered by expected research value per unit of effort. Each step says which
result would change the plan. The reasoning behind each is in `GUIDE.md`,
Chapter 9.

## 0. Check why longer training hurt outside the area (about 1 h, no training)

R1 at 2M (seeds 0-2) improved inside the area but crashed far more from
outside starts (22% -> 58%). Hypothesis: the 1M-transition replay buffer
starts discarding the early, wide-ranging flights after 1M steps. Compare the
distances to the target stored in the 1M and 2M buffers (in the checkpoints
on Narval), and fly the 1.1M-2M snapshots.
- **Decision:** if confirmed, use a buffer of at least the run length for
  long runs; either way, carry the small model forward (it matches R1 at 2M
  inside the area and is far safer outside, at a quarter of the training time).

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
