"""Test flights for a trained TD-MPC2 hover agent.

The agent plans exactly as it does in training (MPC, deterministic=True,
train=False) but never learns here. All requested flights fly side by side on
the GPU, and each whole flight is one compiled loop, so the only CPU<->GPU copy
is the finished trajectory. Testing 18 flights costs about the same as one.

Metrics are defined here, before looking at any multi-seed results:
  late error      mean distance to the target over the last 100 steps flown (2 s)
  time in 10 cm   fraction of the steps after the first 2 s spent within 0.10 m
                  of the target; steps after a crash count as outside
"""
import csv
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from constants import SCENE_PATH_1_DRONE, TARGET_POS
from crazyflie_env import CrazyflieEnv
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, place_drone

CONTROL_DT = 0.02         # seconds per control step (50 Hz)
LATE_WINDOW_STEPS = 100   # last 2 s of a flight
SETTLE_STEPS = 100        # first 2 s are take-off, not hovering
TOLERANCE_M = 0.10

# Fixed before the seed runs. "home" is the start used in training; the others
# are 0.5 m to each side at the same height, and one start above the target.
TEST_STARTS = {
    "home": (0.0, 0.0, 0.2),
    "plus_x": (0.5, 0.0, 0.2),
    "minus_x": (-0.5, 0.0, 0.2),
    "plus_y": (0.0, 0.5, 0.2),
    "minus_y": (0.0, -0.5, 0.2),
    "above": (0.0, 0.0, 1.5),
}


@dataclass
class Flight:
    start_name: str
    start: np.ndarray       # (3,)
    target: np.ndarray      # (3,)
    position: np.ndarray    # (steps, 3) true position after each control step
    command: np.ndarray     # (steps, 4) thrust, roll, pitch, yaw sent to the drone
    reward: np.ndarray      # (steps,)
    steps_flown: int
    crashed: bool

    def errors(self):
        """3D, horizontal and vertical distance to the target for the steps flown."""
        offset = self.position[:self.steps_flown] - self.target
        return (np.linalg.norm(offset, axis=1),
                np.linalg.norm(offset[:, :2], axis=1),
                np.abs(offset[:, 2]))

    def summary(self):
        error_3d, error_xy, error_z = self.errors()
        late = slice(max(0, self.steps_flown - LATE_WINDOW_STEPS), self.steps_flown)
        steps_after_settling = len(self.reward) - SETTLE_STEPS
        inside = np.sum(error_3d[SETTLE_STEPS:] < TOLERANCE_M)
        return {
            "start": self.start_name,
            "steps_flown": self.steps_flown,
            "crashed": self.crashed,
            "return": float(self.reward[:self.steps_flown].sum()),
            "final_error_m": float(error_3d[-1]),
            "late_error_m": float(error_3d[late].mean()),
            "late_error_xy_m": float(error_xy[late].mean()),
            "late_error_z_m": float(error_z[late].mean()),
            "late_height_m": float(self.position[late, 2].mean()),
            "time_in_10cm": float(inside / steps_after_settling) if steps_after_settling > 0 else float("nan"),
        }


class FlightTester:
    """Flies a batch of test flights with a given agent."""

    def __init__(self, flight_steps=500):
        self.drone = CrazyflieEnv(scene_path=SCENE_PATH_1_DRONE, num_drones=1)
        self.flight_steps = int(flight_steps)
        self._fly = jax.jit(self._fly_batch)

    def fly(self, agent, start_names, seed, targets=None):
        """Fly one flight per entry in `start_names` (names from TEST_STARTS).

        The same `seed` gives the same sensor noise and planner randomness, so
        two agents tested with one seed face identical conditions.
        """
        starts = np.array([TEST_STARTS[name] for name in start_names], dtype=np.float32)
        if targets is None:
            targets = np.tile(np.asarray(TARGET_POS, dtype=np.float32), (len(starts), 1))
        reset_key, plan_key = jax.random.split(jax.random.PRNGKey(seed))
        reset_keys = jax.random.split(reset_key, len(starts))

        trajectory = jax.device_get(self._fly(agent, reset_keys, starts, targets, plan_key))

        flights = []
        for i, name in enumerate(start_names):
            alive = trajectory["alive"][:, i]
            steps_flown = int(alive.sum())
            flights.append(Flight(
                start_name=name,
                start=starts[i],
                target=np.asarray(targets[i]),
                position=trajectory["position"][:, i],
                command=trajectory["command"][:, i],
                reward=trajectory["reward"][:, i],
                steps_flown=steps_flown,
                crashed=bool(trajectory["done"][:steps_flown, i].any()),
            ))
        return flights

    def _fly_batch(self, agent, reset_keys, starts, targets, plan_key):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            return place_drone(self.drone, state, start, target)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)

        # An empty previous plan, same as prev_plan=None at the start of a flight.
        n, horizon, action_dim = len(starts), agent.horizon, agent.model.action_dim
        plan = (jnp.zeros((n, horizon, action_dim)),
                jnp.full((n, horizon, action_dim), agent.max_plan_std))

        def control_step(carry, key):
            states, plan, alive = carry
            action, plan = agent.act(states.obs, prev_plan=plan,
                                     deterministic=True, train=False, key=key)
            # Same mapping as Gymnasium's RescaleAction: [-1, 1] -> drone units.
            command = jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)
            states = step_all(states, command)
            done = states.done > 0
            record = {"position": states.data.qpos[:, :3], "command": command,
                      "reward": states.reward, "done": done, "alive": alive}
            # A crashed drone keeps being simulated, but its steps are masked out.
            return (states, plan, alive & ~done), record

        keys = jax.random.split(plan_key, self.flight_steps)
        carry = (states, plan, jnp.ones(n, dtype=bool))
        _, trajectory = jax.lax.scan(control_step, carry, keys)
        return trajectory


def write_summary(path, flights, **extra_columns):
    """One row per flight. Extra columns (e.g. seed=3) are written first."""
    rows = [{**extra_columns, "flight": i, **f.summary()} for i, f in enumerate(flights)]
    _write_rows(path, rows, append=False)
    return rows


def append_learning_curve(path, flights, **extra_columns):
    """Add one row per flight to a learning curve file, creating it if needed."""
    rows = [{**extra_columns, "flight": i, **f.summary()} for i, f in enumerate(flights)]
    _write_rows(path, rows, append=True)
    return rows


def write_trajectories(path, flights):
    """Step-by-step position, error and controls for every flight."""
    rows = []
    for i, f in enumerate(flights):
        error_3d, _, _ = f.errors()
        for t in range(f.steps_flown):
            x, y, z = f.position[t]
            thrust, roll, pitch, yaw = f.command[t]
            rows.append({
                "flight": i, "start": f.start_name, "step": t,
                "time_s": round((t + 1) * CONTROL_DT, 4),
                "x_m": x, "y_m": y, "z_m": z, "error_m": error_3d[t],
                "reward": f.reward[t],
                "thrust": thrust, "roll": roll, "pitch": pitch, "yaw": yaw,
            })
    _write_rows(path, rows, append=False)


def _write_rows(path, rows, append):
    if not rows:
        return
    new_file = not append or not os.path.exists(path)
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _tidy(v) for k, v in row.items()})


def _tidy(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, np.bool_):
        return bool(value)
    return value
