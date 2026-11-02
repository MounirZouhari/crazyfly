"""Where does the agent think it is best to hover? Its own value of hovering still at points around the target.

For a run and a snapshot, the drone is placed at rest and level at every
point of a grid around the target (x, y and z each from -30 to +30 cm in
steps of 2 cm, on the three planes through the target), with a plain hover
as the previous command and no sensor noise; the observation is built
exactly as the simulator builds it (deploy_controller.build_observation).
The agent's value of each point is the average of its value networks for the
action its policy network proposes there. Nothing is trained or flown.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/probe_value.py <run_dir> --snapshots 400000 1000000

Prints, for each snapshot: where on the grid the value is highest (offset
from the target in cm), and how much the value drops from the target to
points 5, 10 and 20 cm away (mean over the grid points at that distance).

What to compare with: the reward's distance-improvement term (previous
distance minus new distance) means that a drone d metres from the target can
still collect about d of reward by flying to it, so a value function that
matched the rewards would grow by roughly 0.1 per 10 cm near the target. A
value that changes much less than that does not see small displacements.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np

from constants import BASE_HOVER_THRUST, TARGET_POS
from deploy_controller import DroneState, build_observation
from eval_hover import load_agent

OFFSETS_M = np.round(np.arange(-0.30, 0.3001, 0.02), 4)


def grid_points(target):
    """Points on the three planes through the target (x-z, y-z, x-y), without duplicates."""
    points = set()
    for a in OFFSETS_M:
        for b in OFFSETS_M:
            points.add((a, 0.0, b))
            points.add((0.0, a, b))
            points.add((a, b, 0.0))
    offsets = np.array(sorted(points))
    return offsets, target + offsets


def values(agent, cfg, positions, target):
    metric = bool(cfg.env.get("metric_target", False))
    hover = np.array([BASE_HOVER_THRUST, 0.0, 0.0, 0.0])
    obs = np.stack([build_observation(DroneState(p, np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3)),
                                      target, hover, metric, initial_distance=1.0) for p in positions])
    key = jax.random.PRNGKey(0)
    z = agent.model.encode(obs=jnp.asarray(obs), params=agent.model.encoder.params, key=key)
    action = agent.model.sample_actions(z=z, deterministic=True, params=agent.model.policy_model.params, key=key)[0]
    q, _ = agent.model.Q(z=z, a=action, params=agent.model.value_model.params, key=key)
    return np.asarray(q).mean(axis=0).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--snapshots", type=int, nargs="+", required=True)
    args = parser.parse_args()

    target = np.asarray(TARGET_POS, dtype=np.float64)
    offsets, positions = grid_points(target)
    distance = np.linalg.norm(offsets, axis=1)
    at_target = np.flatnonzero(distance == 0)[0]
    for step in args.snapshots:
        agent, _, cfg = load_agent(args.run_dir, snapshot=step)
        v = values(agent, cfg, positions, target)
        best = int(np.argmax(v))
        drops = []
        for r in (0.05, 0.10, 0.20):
            ring = np.abs(distance - r) < 0.011
            drops.append(f"{r * 100:.0f} cm: {v[at_target] - v[ring].mean():+.3f}")
        print(f"{args.run_dir.name} snapshot {step // 1000}k: value at target {v[at_target]:.3f}; highest "
              f"{v[best]:.3f} at offset ({', '.join(f'{100 * o:+.0f}' for o in offsets[best])}) cm, "
              f"{100 * distance[best]:.0f} cm from the target; drop from the target to " + ", ".join(drops),
              flush=True)


if __name__ == "__main__":
    main()
