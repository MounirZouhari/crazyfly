# Results summary: TD-MPC2 Crazyflie hover (Nov 2, 2026)

Each answer is marked:
- **Established:** several seeds, the right controls, repeated measurement.
- **Preliminary:** few seeds or a single setting; the direction is clear, the size uncertain.
- **Pending:** nothing; every experiment has finished.

Success = a full 10 s flight whose late error (mean distance to the target
over the last 2 s) is below 10 cm. Numbers are mean ± standard error across
seeds; brackets are 95% bootstrap intervals over seeds. The full explanation
of every result is in `GUIDE.md` / `GUIDE.pdf`.

## Q1. Why did short TD-MPC2 runs vary so much between seeds?

**Established.** The runs had not finished learning at 50k steps. The spread
was real, not test noise: re-flying all 33 Batch 1 agents 20 times, 93-94% of
the variance of the small agents comes from the agents themselves (ICC 0.93
and 0.94). The three published-size runs trained to 1M steps all converged
(5.7-8.6 cm). Training losses do not track the swings within a run (all
correlations of changes between -0.14 and +0.10).

## Q2. Does longer training resolve the variability?

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

## Q3. TD-MPC2 vs PPO: simulator steps and wall-clock time

**Established (Batch 1, single start).** TD-MPC2 reached 10 cm after 105k-175k
steps, PPO after 7.9M-19.7M (about 100x more). In wall-clock time PPO was
faster (about 2.6 vs 47 min) because it simulates 2,048 drones at once.
Caveat: the historical PPO had two bugs (constant learning rate; leaky
automatic reset); it is a baseline, not a tuned comparison.

## Q4. GPU use, and why TD-MPC2 is slower in wall-clock time

**Established.** One training step of the published model on a Narval A100
slice: simulate 6.8 ms, plan 10.2 ms, update 2.6 ms (49 steps/s). Many small
sequential operations for one drone leave the GPU mostly idle (25-28% busy at
home; not measurable on MIG slices).

## Q5. Parallel environments

**Established as a measured projection.** Simulating 128 drones costs the same
as one, but planning and the one-update-per-step rule scale with the number of
drones: 45 -> 74 steps/s (1.6x) for the published model, 93 -> 271 (2.9x)
for the small one. More needs fewer updates per step, a different algorithm.

## Q6. Learning from continuously randomised starts, and generalisation

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

## Q7. When and why does the learned world model help?

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

## Q8. Does planning improve robustness to disturbances?

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

## Q9. Model size: speed, deployability, accuracy

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

## Q10. What prevents safe sim-to-real transfer?

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
