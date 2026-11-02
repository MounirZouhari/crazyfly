# Next research plan (based on the Nov 1-2 results)

Ordered by expected research value per unit of effort. Each step names the
result that would change the plan.

## 1. Confirm or reject model exploitation across seeds (1-2 h, no training)

**Why.** It is the clearest "when and why does the model hurt" finding so
far. On single-start agents, planning crashes far more than the policy
network alone; the planner visits states where the world model is 2-3x more
wrong; and on one agent, anchoring half the planner's candidates to the
policy network removed the extra crashes.

**What.** Analyse the planner-variant, model-error and disturbance jobs
already submitted for 25 Batch 1 agents and all Batch 2 agents:

```bash
.venv/bin/python code/src/tdmpc2_training/analyze_planner_variants.py <batches...>
.venv/bin/python code/src/tdmpc2_training/analyze_model_error.py <batches...>
```

**Decision.**
- If anchored planning is better than both the trained planner and the
  policy network across seeds, make it the default for deployment, and test
  it at training time (step 4).
- If it only helps single-start agents, the effect is about data coverage,
  which supports random starts as the main remedy.

## 2. Small vs published model at 1M steps with random starts (pending, ~04:00)

**Why.** Mid-run, the small model learned random starts much faster and
generalised further. It also decides 4x faster (1.4 ms vs 6.1 ms on a GPU
slice; 11 ms vs 516 ms on a CPU).

**Decision.**
- If the small model matches the published one in precision and is at
  least as robust, use it for deployment work.
- If the published model is more precise but less robust, report the
  trade-off and consider a medium size (latent 256: 4.2 ms per decision).

## 3. Planning under disturbances the model was trained on (new training, ~8 h on Narval)

**Hypothesis H2.** Planning helps when the world model has seen the relevant
variation in training (interpolation), and hurts when it has not
(extrapolation, H1).

**Experiment.** Train R1-like agents with dynamics randomisation: mass and
motor strength drawn per episode (the lab environment already supports this)
and, separately, random wind gusts. Test planning vs policy network on
disturbances inside and beyond the training ranges. 3 seeds as a pilot,
then 11.

**Decision.** This is the experiment most likely to give a positive "the
model helps" result, and it is also what sim-to-real needs (Peng et al.
2018; Molchanov et al. 2019).

## 4. Train with policy-anchored planning (~8 h on Narval)

Only if step 1 confirms exploitation. Use `policy_prior_samples = 256`
during data collection too, and compare sample efficiency and robustness
with R1.

## 5. Firmware-compatible action space (engineering, about a week)

The current agents cannot fly a real Crazyflie (see
`DEPLOYMENT_READINESS.md`). Change the simulator's action to collective
thrust plus body-rate setpoints with a model of the firmware's rate
controller, retrain, and repeat the Batch 2 tests and the software-in-the-
loop check. Required before any hardware test. Needs Prof. Bellinger's
agreement on scope.

## 6. Training throughput (optional)

Parallel drones give TD-MPC2 at most about 1.7x at its default of one update
per step. To go further, test fewer updates per step with N drones (for
example 16 drones, 1 update per 4 steps) and measure the cost in sample
efficiency. This must be reported as a different algorithm from TD-MPC2.

## 7. Smaller questions

- **Longer planning horizon at test time** (5 or 10 steps instead of 3):
  does looking further ahead help under disturbances, or do model errors
  compound?
- **Best checkpoint:** if the snapshot evaluation shows large swings between
  checkpoints, add validation-based checkpoint selection to the standard
  evaluation.
- **PPO:** fix the two bugs (learning-rate units; full reset on automatic
  reset) and rerun on random starts, only if a model-free comparison under
  the same conditions is wanted for the thesis.
