"""Hover task with a sharper reward for holding position close to the target (a reward-design experiment).

The same task as CrazyflieHoverEnv with random starts and the target offset
in metres, plus one reward term concentrated within a few centimetres of the
target:

    bonus = 0.01 * exp(-d^2 / (2 * (0.05 m)^2)),   d = distance to the target

The lab's reward per step is 0.001 + 0.01 (1 - tanh d) + the distance
improvement, and -1 on a crash. Hovering at 5 cm instead of 15 cm earns only
0.001 more per step with it, about a tenth of the value of hovering, so the
precision of the small random-start agents may be weakly anchored. With the
bonus, hovering at 2 cm instead of 15 cm earns about 0.010 more per step.
Physics, observations, starts and termination are unchanged. The test flights
(hover_eval.py) measure distances and are flown with the lab's reward, so
their results compare directly with config_batch2_random_start_small.
Used by config_batch2_sharp_small.
"""
import jax
import numpy as np

from crazyflie_hover import CrazyflieHoverEnv

SHARP_WEIGHT = 0.01
SHARP_WIDTH_M = 0.05


def sharp_bonus(distance_m):
    return SHARP_WEIGHT * np.exp(-np.square(distance_m) / (2 * SHARP_WIDTH_M ** 2))


class SharpHoverEnv(CrazyflieHoverEnv):
    """CrazyflieHoverEnv whose reward has an extra term concentrated near the target."""

    def __init__(self, episode_length=500, random_starts=True, metric_target=True):
        super().__init__(episode_length=episode_length, random_starts=random_starts, metric_target=metric_target)
        self._target = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._target = np.asarray(jax.device_get(self._state.info["target_pos"]), dtype=np.float64).reshape(-1)[:3]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        distance = np.linalg.norm(np.array([info["x"], info["y"], info["height"]]) - self._target)
        return obs, reward + float(sharp_bonus(distance)), terminated, truncated, info
