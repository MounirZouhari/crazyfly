"""Hover task with randomised physics in every training flight (dynamics randomisation).

The same task as CrazyflieHoverEnv (random starts, target offset in metres),
but each flight also draws, uniformly:
  mass and inertia   x 1.0 to 1.15   (an expansion deck, a payload)
  motor strength     x 0.9 to 1.0    (battery drop, worn propellers)
  steady wind        0 to 3 m/s from a random horizontal direction
These cover the in-range part of the disturbance tests (eval_disturbances.py);
4 m/s wind, 6 m/s gusts and command delays stay outside training. The agent
does not observe any of these values. Used by config_batch2_randomized_small.
"""
import jax
import jax.numpy as jnp
import numpy as np

from crazyflie_hover import CrazyflieHoverEnv
from eval_disturbances import WindyDrone

MASS_RANGE = (1.0, 1.15)
MOTOR_RANGE = (0.9, 1.0)
WIND_MAX = 3.0


class RandomizedHoverEnv(CrazyflieHoverEnv):
    """CrazyflieHoverEnv whose mass, motor strength and wind change every flight."""

    def __init__(self, episode_length=500, random_starts=True, metric_target=True):
        super().__init__(episode_length=episode_length, random_starts=random_starts, metric_target=metric_target)
        self.drone = WindyDrone(metric_target=metric_target)
        # The parent compiled its step for the plain drone; reset and place read
        # self.drone when they are first compiled, so only step is rebuilt.
        self._step_fn = jax.jit(self.drone.step)
        self.dynamics = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        mass = self.np_random.uniform(*MASS_RANGE)
        motor = self.np_random.uniform(*MOTOR_RANGE)
        speed = self.np_random.uniform(0.0, WIND_MAX)
        heading = self.np_random.uniform(0.0, 2 * np.pi)
        wind = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0], dtype=np.float32)
        state_info = dict(self._state.info)
        state_info["body_mass"] = state_info["body_mass"] * mass
        state_info["body_inertia"] = state_info["body_inertia"] * mass
        state_info["actuator_gainprm"] = state_info["actuator_gainprm"] * motor
        state_info["wind"] = jnp.asarray(wind)
        self._state = self._state.replace(info=state_info)
        self.dynamics = {"mass": float(mass), "motor": float(motor), "wind": wind}
        return obs, info
