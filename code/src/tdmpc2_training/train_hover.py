"""Start the vendored TD-MPC2 trainer on the one-drone hover task."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import gymnasium as gym
from crazyflie_hover import CrazyflieHoverEnv

gym.register("CrazyflieHover-v0", entry_point=CrazyflieHoverEnv)

from tdmpc2_jax.train import train

if __name__ == "__main__":
    if not any(arg.startswith("--config-name") for arg in sys.argv[1:]):
        sys.argv.append("--config-name=config_hover")
    train()
