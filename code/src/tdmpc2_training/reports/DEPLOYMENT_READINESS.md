# Deployment readiness: TD-MPC2 on a real Crazyflie

**Status (Nov 2, 2026): not ready. Do not fly.** Everything below was checked
in simulation or by reading code. No drone was powered, armed or commanded.
Physical tests need Prof. Bellinger's approval, the lab safety training, and
the blockers below fixed first.

## Summary

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

## 1. Action semantics (blocker)

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

## 2-4. Observation, units and conventions (solved in simulation)

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

## 5. Latency

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

## 6. Robustness in simulation

`eval_disturbances.py` covers wind, gusts, +5-15% mass, -5-10% motor
strength and 20-40 ms delays. The Batch 1 1M agents fail at 2 m/s steady
wind, with and without planning. Results for the Batch 2 agents are due on
Nov 2.

Real drones will differ by more than these single factors (drone-to-drone
thrust differences: `docs/Crazyflie_Hardware.md`). Dynamics randomisation in
training (Peng et al. 2018; Molchanov et al. 2019) is the standard remedy,
and the lab environment already supports it.

## 7-9. State estimation, take-off, failsafes

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

## Proposed supervised test sequence (not started)

0. Lab safety training, supervisor approval, written test plan.
1. **Software in the loop:** the adapter flies the simulator. Done.
2. **Radio-only, no thrust:** drone on the bench, motors never armed. Read the
   state at 50 Hz, run the adapter, log the decisions, the end-to-end latency
   and the jitter. Send nothing.
3. Retrain with a firmware-compatible action space (item 1). Repeat the
   Batch 2 tests and step 1.
4. Tethered or caged hover with a safety pilot and an emergency stop, at low
   thrust limits.
