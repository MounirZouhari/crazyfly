# Batch 2 update for Prof. Bellinger

Hi Professor Bellinger,

Here is where the project stands after Batch 2. Following your feedback, every agent now trains from random starting positions drawn from a region (with one block held out for testing), runs are 1M steps, and the main experiments have 11 seeds (3 for the controls and pilots). Every agent is tested on the same 128 flights: the training area, the held-out block, starts up to 0.75 m outside the area, and the old Batch 1 starts. Everything is in simulation; nothing has been flown.

## What I found

1. **Seed variance in Batch 1 was real, and the runs were too short.** Re-flying every Batch 1 agent 20 times, 93-94% of the variance of the small agents comes from the agents themselves (ICC 0.93-0.94), not from test noise. At 1M steps the random-start seeds agree on safety (no crashes inside the area), but their precision still varies by a few centimetres around the 10 cm threshold, between seeds and from one checkpoint to the next.

2. **Random starts work, for both model sizes.** With planning, success / crashes from the training area, held-out block and outside:
   - published TD-MPC2, random starts (R1, 11 seeds): 46/0%, 39/2%, 20/26%;
   - the same from the single Batch 1 start (C1, 3 seeds): 26/35%, 4/82%, 0/92%;
   - small TD-MPC2, random starts (D, 11 seeds): 78/0%, 83/0%, 59/3%.

   C1 reproduces Batch 1, so the gain comes from the random starts. The held-out block does as well as the training area (interpolation works); outside the area is harder.

3. **Checkpoint selection matters.** Choosing each run's checkpoint on separate validation starts gives 97% / 91% / 70% success for the small model with planning, and 90% / 81% / 59% (no crashes inside the area) for the published model's policy network used alone, at 0.3 ms per decision.

4. **When the world model helps.** Planning adds 69 points for the small random-start agents, whose world model is accurate along the planner's flights. It hurts where the planner can find the model's errors: single-start agents away from their start (the planner's flights go where the model is 2-3 times more wrong, and restraining the search reduces crashes), and, to a lesser degree, the published random-start agents, whose 512×6 search loses to their own policy network (-15 to -25 points) and to a smaller 64×2 search. A probe of the value network shows it is almost flat within 20 cm of the target, so the planner's precision comes from the world model's predicted rewards over the short horizon.

5. **Disturbances.** The small agents with planning keep 72-77% success under extra mass (up to 15%), weaker motors (up to 10%) and 20-40 ms command delay. Wind of 2 m/s or more defeats every agent trained without wind, and in strong wind the planner crashes more than the policy network alone. Training with randomised mass, motors and wind (3-seed pilot) cost most of the precision (13% success) for slightly fewer crashes in strong wind; I think the agent needs a short history in its observation to infer the conditions.

6. **Precision drift.** I tested whether the lab's reward, which pays almost the same for 5 cm and 15 cm, causes the drift: with a reward ten times sharper near the target (3-seed pilot), the value network learned to prefer the target, but the precision still drifted between checkpoints. So the reward is not the main cause; checkpoint selection on validation starts is the practical fix.

7. **Model size and speed.** Per decision on a GPU slice / 2 CPU cores: published planner 6.1 / 516 ms, small planner 1.4 / 11.4 ms, policy network alone 0.2-0.4 ms (the control period is 20 ms). The small model also trains 1.8 times faster. At 1M steps it is the more precise planner. Because the published model was still improving, I continued three R1 seeds to 2M steps: inside the area they improved (66% to 72% success with planning, and more precise), but crashes from starts outside the area rose from 22% to 58%. My best explanation is the replay buffer, which holds 1M transitions: after 1M steps it starts discarding the early, wide-ranging flights, so the agent specialises on its training area. So the published model was partly just slower, but even at 2M it does not beat the small model trained for 1M.

8. **GPU use and parallel environments.** With one drone, a training step is many small sequential operations (simulate 6.8 ms, plan 10.2 ms, update 2.6 ms), so the GPU is mostly idle. Simulating 128 drones costs the same as one, but planning and one update per step scale with the number of drones, so the projected speed-up is only 1.6 times (published) to 2.9 times (small).

9. **Toward a real drone.** Two blockers: the agents output thrust and body moments, which the Crazyflie firmware does not accept (they would have to be retrained with rate setpoints), and the thrust calibration. In simulation, the planner's commands jump by about 30% of their range per step; a 40 ms command filter (small model) or acting on the planner's averaged plan (both models) removes this without losing precision.

## Limitations

All results are from simulation with a rigid-body model and simple sensor noise. C1, H2 and the reward pilot have 3 seeds each. The 10 cm success threshold matters because the agents' error often sits between 5 and 15 cm, so I also report success for every threshold. TD-MPC2 was used with its published settings, without tuning. The PPO comparison from Batch 1 used the lab's PPO, which has two bugs I found (a learning rate that never decreases and a leaky reset); I kept it as a historical baseline.

## Questions for you

- Should I move toward a real flight now (which first means retraining with firmware-compatible commands) or deepen the simulation study first?
- Which controller should I carry forward: the small model with planning, or the published model's policy network alone?
- Do you want a fair PPO comparison (bugs fixed, random starts) for the thesis?

The full write-up, with every figure and a walkthrough of the code, is in `code/src/tdmpc2_training/reports/GUIDE.pdf`, and a short version in `BATCH2_EXPLAINED.pdf`, on the `tdmpc2-hover` branch.

Mounir
