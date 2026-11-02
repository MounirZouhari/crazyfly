"""Simulation-only deployment adapter for a trained TD-MPC2 hover agent. It never talks to a drone.

It turns a drone state given in the units and conventions of the Crazyflie's
own state estimate into exactly the 31-number observation the agent was
trained on, runs the agent (with its planner or its policy network alone),
returns the command in simulator units and times every decision.
check_deployment.py tests it against the simulator; see
reports/DEPLOYMENT_READINESS.md for what still blocks a real flight.

The simulator's command (collective thrust in N plus three body moments) is
not a command the Crazyflie firmware accepts, so to_crazyflie_command()
deliberately refuses to produce one.

Conventions, matching gpu_training/crazyflie_env.py and MuJoCo's free joint:
  position          m, world frame, z up
  velocity          m/s, world frame        (Crazyflie stateEstimate.vx/vy/vz)
  quaternion        w, x, y, z, body to world (Crazyflie logs qx, qy, qz, qw)
  angular velocity  rad/s, body frame        (Crazyflie gyro.x/y/z is in deg/s)
  previous action   the observation built for decision t carries the command
                    sent at decision t-2 (the simulator records an action only
                    after building the next observation), zeros at the start
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np

from crazyflie_hover import ACTION_HIGH, ACTION_LOW
from eval_hover import load_agent

OBSERVATION_SIZE = 31


@dataclass
class DroneState:
    """One state estimate in SI units and the simulator's conventions."""
    position: np.ndarray           # (3,) m
    velocity: np.ndarray           # (3,) m/s, world frame
    quaternion: np.ndarray         # (4,) w, x, y, z
    angular_velocity: np.ndarray   # (3,) rad/s, body frame

    @classmethod
    def from_crazyflie_log(cls, x, y, z, vx, vy, vz, qx, qy, qz, qw, gyro_x, gyro_y, gyro_z):
        """Values as the Crazyflie logs them: stateEstimate.x/y/z (m), stateEstimate.vx/vy/vz
        (m/s, world frame), stateEstimate.qx/qy/qz/qw, gyro.x/y/z (deg/s, body frame)."""
        quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
        return cls(position=np.array([x, y, z], dtype=np.float64),
                   velocity=np.array([vx, vy, vz], dtype=np.float64),
                   quaternion=quaternion / np.linalg.norm(quaternion),
                   angular_velocity=np.radians([gyro_x, gyro_y, gyro_z]))

    @classmethod
    def from_simulator(cls, qpos, qvel):
        """True state of the one-drone simulator (MuJoCo free joint)."""
        qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
        return cls(position=qpos[0:3], velocity=qvel[0:3], quaternion=qpos[3:7], angular_velocity=qvel[3:6])


def rotation_matrix(q):
    """Same formula as gpu_training/math_utils.quat_to_rotation_matrix, quaternion w, x, y, z."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])


def build_observation(state, target, previous_command, metric_target, initial_distance):
    """The simulator's 31-number observation for one drone, without its sensor noise."""
    rot = rotation_matrix(state.quaternion)
    scale = 1.0 if metric_target else initial_distance
    offset = (np.asarray(target, dtype=np.float64) - state.position) / (scale + 1e-9)
    return np.concatenate([
        state.position, state.velocity, state.angular_velocity, rot.reshape(-1),
        rot.T @ offset, rot.T @ state.velocity,
        np.zeros(3),                        # offset to itself (the multi-drone feature)
        np.asarray(previous_command, dtype=np.float64),
    ]).astype(np.float32)


class TDMPC2Controller:
    """Runs a trained agent on hardware-style state estimates (simulation only)."""

    def __init__(self, run_dir, use_planner=True, plan=None, snapshot=None, target=(0.0, 0.0, 1.0), seed=0):
        self.agent, self.checkpoint_step, cfg = load_agent(Path(run_dir), snapshot=snapshot)
        if plan:
            plans, rounds = (int(n) for n in plan.lower().split("x"))
            self.agent = self.agent.replace(population_size=plans, mppi_iterations=rounds,
                                            num_elites=max(1, plans // 8),
                                            policy_prior_samples=min(self.agent.policy_prior_samples, plans // 4))
        self.use_planner = use_planner
        self.metric_target = bool(cfg.env.get("metric_target", False))
        self.target = np.asarray(target, dtype=np.float64)
        self.key = jax.random.PRNGKey(seed)
        self.latencies_ms = []
        self.reset(start_position=(0.0, 0.0, 0.2))

    def reset(self, start_position):
        """Call at the start of every flight, with the position the flight starts from."""
        self.initial_distance = float(np.linalg.norm(np.asarray(start_position) - self.target))
        self.sent = [np.zeros(4), np.zeros(4)]          # commands sent at decisions t-2 and t-1
        horizon = self.agent.horizon
        self.plan = (jnp.zeros((1, horizon, 4)), jnp.full((1, horizon, 4), self.agent.max_plan_std))
        self.step = 0

    def observation(self, state):
        return build_observation(state, self.target, self.sent[0], self.metric_target, self.initial_distance)

    def decide(self, state):
        """One control decision: state estimate in, command (thrust N, 3 moments) out."""
        obs = self.observation(state)[None]
        key = jax.random.fold_in(self.key, self.step)
        start = time.perf_counter()
        if self.use_planner:
            action, self.plan = self.agent.act(obs, prev_plan=self.plan, deterministic=True, train=False, key=key)
        else:
            action, _ = self.agent.act(obs, mpc=False, deterministic=True, train=False, key=key)
        action = np.asarray(action)[0]                  # waits for the device
        self.latencies_ms.append(1000 * (time.perf_counter() - start))
        # The same mapping as Gymnasium's RescaleAction in training: [-1, 1] -> drone units.
        command = np.clip(ACTION_LOW + (action + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW), ACTION_LOW, ACTION_HIGH)
        self.sent = [self.sent[1], command.astype(np.float64)]
        self.step += 1
        return command

    def warm_up(self, calls=5):
        """Compile before take-off: the first decision takes seconds, not milliseconds."""
        dummy = DroneState(np.array([0.0, 0.0, 0.2]), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3))
        for _ in range(calls):
            self.decide(dummy)
        compile_ms = self.latencies_ms[0]
        self.latencies_ms = []
        self.reset(start_position=(0.0, 0.0, 0.2))
        return compile_ms

    @staticmethod
    def to_crazyflie_command(command):
        raise NotImplementedError(
            "The agent outputs collective thrust (N) and three body moments, the simulator's actuators. "
            "The Crazyflie firmware takes attitude or body-rate setpoints with a thrust value, or motor "
            "powers; none of these is the same quantity, and the lab's earlier mapping (moments sent as "
            "rate setpoints clamped to 1 deg/s) is not physically equivalent. Train with an action space "
            "the firmware accepts first. See reports/DEPLOYMENT_READINESS.md.")
