"""Load a saved TD-MPC2 hover agent from disk and test it. Nothing is trained.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_hover.py outputs/batch1_fixed_start/seed_00
    .venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --starts home --flights 3

The agent is rebuilt from the run's own .hydra/config.yaml and its newest
checkpoint is restored, so this also checks that a saved model can be reused
in a fresh process. Results go to <run_dir>/evaluation/:

    summary.csv       one row per flight (errors, crash, return)
    trajectories.csv  every step of every flight (position, error, controls)
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import numpy as np
import orbax.checkpoint as ocp
from omegaconf import OmegaConf

from hover_eval import TEST_STARTS, FlightTester, write_summary, write_trajectories
from tdmpc2_jax.build import build_agent

OBSERVATION_SIZE = 31
ACTION_SIZE = 4


def load_agent(run_dir, step=None):
    """Return (agent, checkpoint_step, config) for a finished training run."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model_key, encoder_key = jax.random.split(jax.random.PRNGKey(0))
    template = build_agent(
        cfg,
        observation=np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
        action_dim=ACTION_SIZE,
        model_key=model_key,
        encoder_key=encoder_key,
    )

    options = ocp.CheckpointManagerOptions(read_only=True)
    with ocp.CheckpointManager(
        (run_dir / "checkpoint").resolve(), options=options,
        item_names=("agent", "global_step", "buffer_state"),
    ) as manager:
        step = manager.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"No checkpoint in {run_dir / 'checkpoint'}")
        # Only the agent is needed; the replay buffer is left on disk.
        restored = manager.restore(
            step, args=ocp.args.Composite(agent=ocp.args.StandardRestore(template))
        )
    return restored.agent, step, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="training output folder")
    parser.add_argument("--starts", default="all",
                        help=f"'all' or comma-separated names from: {', '.join(TEST_STARTS)}")
    parser.add_argument("--flights", type=int, default=3, help="flights per start")
    parser.add_argument("--flight-steps", type=int, default=500,
                        help="control steps per flight (500 = 10 simulated seconds)")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: newest)")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="noise seed for the test flights (default: training seed + 2000)")
    args = parser.parse_args()

    run_dir = args.run_dir
    start_names = list(TEST_STARTS) if args.starts == "all" else args.starts.split(",")
    unknown = [s for s in start_names if s not in TEST_STARTS]
    if unknown:
        parser.error(f"unknown start(s): {unknown}")

    agent, step, cfg = load_agent(run_dir, args.step)
    print(f"Loaded {run_dir} at {step} training steps (seed {cfg.seed}).")

    # Training-time test flights use seed + 1000; use different noise here.
    test_seed = cfg.seed + 2000 if args.test_seed is None else args.test_seed
    flight_starts = [name for name in start_names for _ in range(args.flights)]
    tester = FlightTester(flight_steps=args.flight_steps)
    flights = tester.fly(agent, flight_starts, seed=test_seed)

    out_dir = run_dir / "evaluation"
    os.makedirs(out_dir, exist_ok=True)
    rows = write_summary(out_dir / "summary.csv", flights, seed=cfg.seed, checkpoint_step=step)
    write_trajectories(out_dir / "trajectories.csv", flights)

    print(f"\n{'start':<9} {'flight':>6} {'steps':>6} {'crash':>6} {'late err':>9} "
          f"{'xy err':>7} {'z err':>7} {'in 10cm':>8} {'return':>8}")
    for r in rows:
        print(f"{r['start']:<9} {r['flight']:>6} {r['steps_flown']:>6} {str(r['crashed']):>6} "
              f"{r['late_error_m']:>8.3f}m {r['late_error_xy_m']:>6.3f}m {r['late_error_z_m']:>6.3f}m "
              f"{r['time_in_10cm']:>7.0%} {r['return']:>8.3f}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
