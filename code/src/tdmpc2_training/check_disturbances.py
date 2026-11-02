"""Check the disturbance tests (eval_disturbances.py) before using them. Nothing is trained.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/check_disturbances.py [<run_dir>]

A scripted pilot that always sends a plain hover command (and can switch to
another command at a chosen step) stands in for a trained agent, so the
physics can be checked directly:
  1. without disturbance the disturbed tester flies exactly like the normal one
  2. wind pushes the drone downwind, more for stronger wind
  3. extra mass and weaker motors make a hovering drone sink
  4. a delay of k steps makes the drone receive every command exactly k steps late
With <run_dir>, also checks 1 with that run's trained agent and its planner.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from constants import BASE_HOVER_THRUST
from crazyflie_hover import ACTION_HIGH, ACTION_LOW
from eval_disturbances import DisturbedFlightTester, wind_profile
from hover_eval import TEST_SETS, FlightTester

STEPS = 50                               # 1 s of flight
HOVER = 2 * BASE_HOVER_THRUST / ACTION_HIGH[0] - 1   # hover thrust in the agent's [-1, 1] units
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}")
    if not ok:
        failures.append(name)


class _Model(struct.PyTreeNode):
    action_dim: int = struct.field(pytree_node=False, default=4)


class ScriptedPilot(struct.PyTreeNode):
    """Stands in for a TD-MPC2 agent: `first` until `switch_step`, then `later`.

    The step is counted in the first number of the plan the tester carries.
    """
    first: jax.Array
    later: jax.Array
    switch_step: int = struct.field(pytree_node=False, default=10**9)
    horizon: int = struct.field(pytree_node=False, default=3)
    max_plan_std: float = struct.field(pytree_node=False, default=2.0)
    model: _Model = struct.field(default_factory=_Model)

    def act(self, obs, prev_plan=None, mpc=True, deterministic=False, train=False, *, key=None):
        step = prev_plan[0][:, 0, 0]
        action = jnp.where((step >= self.switch_step)[:, None], self.later, self.first)
        return action, (prev_plan[0].at[:, 0, 0].add(1.0), prev_plan[1])


hover_command = jnp.array([HOVER, 0.0, 0.0, 0.0])
pilot = ScriptedPilot(first=hover_command, later=hover_command)
at_target = [("check/0", (0.0, 0.0, 1.0))]
seed = 7

# 1. No disturbance = normal tester ----------------------------------------
starts = TEST_SETS["batch1"]
normal = FlightTester(flight_steps=STEPS).fly(pilot, starts, seed)
disturbed = DisturbedFlightTester(flight_steps=STEPS).fly(pilot, starts, seed)
same = all(np.array_equal(a.position, b.position) and np.array_equal(a.reward, b.reward)
           for a, b in zip(normal, disturbed))
check("no disturbance: identical to the normal tester (scripted pilot)", same)

tester = DisturbedFlightTester(flight_steps=STEPS)
nominal = tester.fly(pilot, at_target, seed)[0].position
drift = np.abs(nominal[-1] - [0.0, 0.0, 1.0]).max()
check("plain hover command holds the drone for 1 s", drift < 0.02, f"moved {drift:.3f} m")

# 2. Wind --------------------------------------------------------------------
x_after = {}
for speed in (1.5, 3.0):
    flight = tester.fly(pilot, at_target, seed, wind=wind_profile("wind", speed, STEPS))[0]
    x_after[speed] = float(flight.position[-1, 0])
check("wind pushes the drone downwind (+x)", x_after[1.5] > 0.01 and x_after[3.0] > x_after[1.5],
      f"after 1 s: {x_after[1.5]:.3f} m at 1.5 m/s, {x_after[3.0]:.3f} m at 3 m/s")
check("drag grows faster than wind speed (about with its square)", x_after[3.0] > 2.5 * x_after[1.5],
      f"ratio {x_after[3.0] / max(x_after[1.5], 1e-9):.1f}")
gust_tester = DisturbedFlightTester(flight_steps=250)
gust_flight = gust_tester.fly(pilot, at_target, seed, wind=wind_profile("gust", 3.0, 250))[0]
before = np.abs(gust_flight.position[:149, 0]).max()   # gust starts at 3.0 s = step 150
check("gust: no push before 3 s, push during it",
      before < 0.01 and gust_flight.position[-1, 0] > 0.05,
      f"x before {before:.3f} m, at 5 s {gust_flight.position[-1, 0]:.3f} m")

# 3. Mass and motors -----------------------------------------------------------
heavy = tester.fly(pilot, at_target, seed, mass_factor=1.15)[0].position[-1, 2]
weak = tester.fly(pilot, at_target, seed, motor_factor=0.90)[0].position[-1, 2]
check("15% heavier drone sinks with the hover command", heavy < nominal[-1, 2] - 0.1, f"height after 1 s {heavy:.2f} m")
check("10% weaker motors sink with the hover command", weak < nominal[-1, 2] - 0.1, f"height after 1 s {weak:.2f} m")

# 4. Delay ---------------------------------------------------------------------
climb = jnp.array([HOVER + 0.2, 0.0, 0.0, 0.0])
switching = ScriptedPilot(first=hover_command, later=climb, switch_step=10)
received_switch = {}
for delay in (0, 1, 2):
    flight = DisturbedFlightTester(flight_steps=STEPS, delay_steps=delay).fly(switching, at_target, seed)[0]
    thrust = flight.command[:, 0]
    received_switch[delay] = int(np.argmax(thrust > BASE_HOVER_THRUST + 1e-4))
check("delay: the drone receives the new command exactly k steps late",
      received_switch == {0: 10, 1: 11, 2: 12}, str(received_switch))

# 5. How a flight ends -----------------------------------------------------------
from eval_disturbances import end_cause
long_tester = DisturbedFlightTester(flight_steps=200)
fell = long_tester.fly(pilot, at_target, seed, mass_factor=1.15)[0]
blown = long_tester.fly(pilot, at_target, seed, wind=wind_profile("wind", 4.0, 200))[0]
held = long_tester.fly(pilot, at_target, seed)[0]
causes = (end_cause(fell), end_cause(blown), end_cause(held))
check("end cause: heavy drone hits the ground, strong wind blows it out of bounds, hover completes",
      causes == ("ground", "out_of_bounds", "completed"), str(causes))

# 6. Optional: a trained agent with its planner ---------------------------------
if len(sys.argv) > 1:
    from eval_hover import load_agent
    run_dir = Path(sys.argv[1])
    agent, step, cfg = load_agent(run_dir)
    metric = bool(cfg.env.get("metric_target", False))
    normal = FlightTester(flight_steps=200, metric_target=metric).fly(agent, starts, seed)
    disturbed = DisturbedFlightTester(flight_steps=200, metric_target=metric).fly(agent, starts, seed)
    same = all(np.array_equal(a.position, b.position) for a, b in zip(normal, disturbed))
    check(f"no disturbance: identical to the normal tester ({run_dir.name}, planner)", same)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("All disturbance checks passed.")
