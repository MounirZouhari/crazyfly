"""Gymnasium interface for the existing one-drone MJX Crazyflie task."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

from constants import SCENE_PATH_1_DRONE, TARGET_POS
from crazyflie_env import CrazyflieEnv

# Physical control limits: thrust (N) and the three body moments.
ACTION_LOW = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([0.35, 1.0, 1.0, 1.0], dtype=np.float32)


def place_drone(drone, state, start, target):
    """Move a freshly reset drone to `start`, level and at rest, flying to `target`.

    The original environment always starts one drone at (0, 0, 0.2). This
    keeps everything else about its reset and only changes the two positions.
    """
    start = jnp.asarray(start, dtype=jnp.float32).reshape(1, 3)
    target = jnp.asarray(target, dtype=jnp.float32)

    qpos = state.data.qpos.at[0:3].set(start[0])
    data = mjx_env.init(drone.mjx_model, qpos=qpos, qvel=jnp.zeros_like(state.data.qvel))

    info = dict(state.info)
    info["target_pos"] = target
    info["prev_pos"] = start
    # Used by the reward, the out-of-bounds check and the observation scaling.
    info["initial_distance"] = jnp.linalg.norm(start - target, axis=1)
    obs = drone._get_obs(data, info)
    return state.replace(data=data, obs=obs, info=info)


class CrazyflieHoverEnv(gym.Env):
    """Expose MJX's functional state API to Gymnasium's stateful API.

    The action space uses the original drone units. TD-MPC2's Gymnasium
    training code wraps it with RescaleAction, so its policy sees [-1, 1].
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_length=500):
        super().__init__()
        self.drone = CrazyflieEnv(scene_path=SCENE_PATH_1_DRONE, num_drones=1)
        self.observation_space = gym.spaces.Box(
            low=np.full(31, -np.inf, dtype=np.float32),
            high=np.full(31, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self.episode_length = episode_length
        self._state = None
        self._steps = 0
        # Compiled once; running these op by op is slow on the GPU.
        self._reset_fn = jax.jit(lambda key: self.drone.reset(key, random_reset=False, target=TARGET_POS))
        self._place_fn = jax.jit(lambda state, start, target: place_drone(self.drone, state, start, target))
        self._step_fn = jax.jit(self.drone.step)

    def reset(self, *, seed=None, options=None):
        """Start a new flight.

        By default this is the training setup: start (0, 0, 0.2), target
        (0, 0, 1). Pass options={"start": (x, y, z), "target": (x, y, z)} to
        change either. The observation still contains the environment's noise.
        """
        super().reset(seed=seed)
        self._steps = 0
        key = jax.random.PRNGKey(int(self.np_random.integers(0, 2**31 - 1)))
        self._state = self._reset_fn(key)

        options = options or {}
        if "start" in options or "target" in options:
            start = options.get("start", self._state.info["prev_pos"][0])
            target = options.get("target", TARGET_POS)
            self._state = self._place_fn(self._state, jnp.asarray(start), jnp.asarray(target))
        return np.asarray(self._state.obs, dtype=np.float32), {}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError(f"Expected four drone controls, got {action.shape}")
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)
        self._state = self._step_fn(self._state, jnp.asarray(action))
        self._steps += 1

        # One GPU -> CPU copy per step instead of one per value.
        obs, reward, done, qpos = jax.device_get(
            (self._state.obs, self._state.reward, self._state.done, self._state.data.qpos)
        )
        terminated = bool(done)
        truncated = self._steps >= self.episode_length and not terminated
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            {"x": float(qpos[0]), "y": float(qpos[1]), "height": float(qpos[2])},
        )

    def close(self):
        self._state = None
