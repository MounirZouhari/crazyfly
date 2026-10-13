"""Check the Gymnasium bridge and normalized thrust without training."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import gymnasium as gym
import numpy as np
from crazyflie_hover import CrazyflieHoverEnv

env = gym.wrappers.RescaleAction(CrazyflieHoverEnv(), -1.0, 1.0)
obs, _ = env.reset(seed=42)
assert obs.shape == (31,) and obs.dtype == np.float32
assert env.action_space.shape == (4,)
action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
assert obs.shape == (31,) and np.isfinite(obs).all()
assert np.isfinite(reward) and not (terminated and truncated)
actual_thrust = float(env.unwrapped._state.info["action_history"][-1, 0, 0])
assert np.isclose(actual_thrust, 0.175), actual_thrust
print("Observation:", obs.shape, "normalized action:", action)
print("Normalized thrust 0 mapped to drone thrust:", actual_thrust)
print("Reward:", reward, "height:", info["height"])
print("terminated:", terminated, "truncated:", truncated)
env.close()
