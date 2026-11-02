# Results summary: TD-MPC2 Crazyflie hover (Nov 2, 2026)

Each answer is marked:
- **Established:** several seeds, the right controls, repeated measurement.
- **Preliminary:** few seeds or a single setting; the direction is clear, the size uncertain.
- **Pending:** an experiment is still running (see `HANDOFF.md`).

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
  around the 10 cm threshold (4-8 cm for 8 seeds, 9-13 cm for 3).
- It also drifts within a run: success from one 100k snapshot to the next
  varies 5-7 times more than test noise allows, mostly as precision changes
  (one seed had a snapshot with 44% crashes).
- Choosing the snapshot on separate validation starts recovers most of it:
  97% test success from the training area at the validation-chosen snapshot
  vs 76% at the final one (11 seeds; +21 ± 11 points).
- Hypothesis (from the reward): hovering at 5 cm instead of 15 cm earns only
  0.001 more per step, so precision is weakly anchored. **Pending:** a 3-seed
  pilot with an extra reward near the target.

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

**Established for the small model; pending for the published one.**
- Small model, 11 seeds, with planning: success / crashes 78 ± 10% / 0%
  (training area), 83 ± 10% / 0% (held-out block, never trained on),
  59 ± 9% / 3 ± 2% (0.25-0.75 m outside the area).
- vs the Batch 1 single-start small agents: +59 [+35, +81], +72 [+48, +91],
  +55 [+38, +72] points.
- Interpolation works (held-out = training area); moderate extrapolation
  mostly works, with lower precision.
- **Pending:** R1 (published model, 11 seeds) and its single-start control C1.

## Q7. When and why does the learned world model help?

**Established on two contrasting cases.**
- Single-start agents (published, 1M): planning hurts off the training start
  (-26 [-46, -8] points; 66% vs 15% crashes from training-area starts). The
  planner's flights go where the world model is 2.4-2.8x more wrong, errors
  peak before crashes, and restraining the planner reduces crashes
  (-19 ± 1 points) but not to the policy's level: model exploitation is one
  mechanism, not the whole story.
- Random-start agents (small, 11 seeds): planning adds +69 [+30, +96] points
  over the policy network alone, which never crashes but rarely ends within
  10 cm (median 21 cm vs 7 cm). The model is as accurate or more accurate
  along the planner's flights (ratio 0.68 ± 0.09), and restraining the
  planner only costs precision (-5 to -7 points).
- So the model helps where its training data covers the situation (it turns
  a safe but imprecise policy into precise control) and hurts where it does
  not (the planner seeks out its errors).
- **Pending:** the same tests on R1 and C1.

## Q8. Does planning improve robustness to disturbances?

**Established for the small random-start agents; pending for R1, C1, H2.**
- With planning, success holds under +5-15% mass (76-72%), -5-10% motor
  strength (76-73%) and 20-40 ms command delay (77-76%), 61-68 points above
  the policy network alone.
- Wind: 60% success at 1 m/s, about 0% from 2 m/s; but no crashes up to
  3 m/s (the drone is pushed off target and stays up).
- Beyond: the planner crashes more than the policy at 4 m/s wind (41% vs 29%)
  and 6 m/s gusts (67% vs 21%), consistent with Q7.
- Single-start agents: the planner is worse than the policy under every
  disturbance.

## Q9. Model size: speed, deployability, accuracy

**Speed established; accuracy pending (R1).**
- Per decision, GPU slice / 2 CPU cores: published planner 6.1 / 516 ms;
  small planner 1.4 / 11.4 ms; small 64x2 planner 1.5 / 4.4 ms; policy only
  0.2-0.4 ms. Control period 20 ms.
- The small model trains 1.8x faster per step (88 vs 49 steps/s).
- The 64x2 planner is as good as the full small planner (+2 [-2, +7] points).
- Mid-run (850k), the small model was more precise than the published one at
  equal steps. **Pending:** R1 final; R1 seeds 0-2 continued to 2M.

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
