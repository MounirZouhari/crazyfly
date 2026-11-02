"""Load a saved TD-MPC2 hover agent from disk and test it. Nothing is trained.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_hover.py outputs/batch1_fixed_start/seed_00
    .venv/bin/python code/src/tdmpc2_training/eval_hover.py <run_dir> --starts home --flights 3

The agent is rebuilt from the run's own .hydra/config.yaml and its newest
checkpoint is restored, so this also checks that a saved model can be reused
in a fresh process. Results go to <run_dir>/evaluation/:

    summary.csv       one row per flight (errors, crash, return, start position)
    trajectories.csv  every step of every flight (position, error, controls)

Which starts are flown (hover_eval.TEST_SETS): Batch 1 runs default to the six
Batch 1 starts (3 flights each); Batch 2 runs default to all groups. Choose
explicitly with --test-sets, e.g. to test an old run on the Batch 2 starts:

    eval_hover.py <run_dir> --test-sets all --out-name evaluation_batch2_tests

The same trained agent can also be flown with a different decision rule, to
see how much accuracy planning buys (results go to their own folder):

    eval_hover.py <run_dir> --no-mpc       # policy network only -> evaluation_policy_only/
    eval_hover.py <run_dir> --plan 128x3   # smaller planner -> evaluation_plan128x3/

--snapshot 300000 tests the agent kept at 300k steps (Batch 2 runs keep one
every 100k steps in snapshots/). Existing results are never overwritten
unless --overwrite is given.
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

from hover_eval import TEST_SETS, TEST_STARTS, FlightTester, write_summary, write_trajectories
from tdmpc2_jax.build import build_agent

OBSERVATION_SIZE = 31
ACTION_SIZE = 4


def load_agent(run_dir, step=None, snapshot=None):
    """Return (agent, step, config) for a training run.

    By default the newest checkpoint/ is loaded; snapshot=N loads the agent
    kept at N steps in snapshots/ instead.
    """
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model_key, encoder_key = jax.random.split(jax.random.PRNGKey(0))
    template = build_agent(
        cfg,
        observation=np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
        action_dim=ACTION_SIZE,
        model_key=model_key,
        encoder_key=encoder_key,
    )

    if snapshot is None:
        directory, items = run_dir / "checkpoint", ("agent", "global_step", "buffer_state")
    else:
        directory, items, step = run_dir / "snapshots", ("agent",), snapshot
    options = ocp.CheckpointManagerOptions(read_only=True)
    with ocp.CheckpointManager(directory.resolve(), options=options, item_names=items) as manager:
        step = manager.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"No checkpoint in {directory}")
        # Only the agent is needed; the replay buffer is left on disk.
        restored = manager.restore(
            step, args=ocp.args.Composite(agent=ocp.args.StandardRestore(template))
        )
    return restored.agent, step, cfg


def is_batch2_run(cfg):
    return cfg.get("curve_set", "home") == "batch2"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="training output folder")
    parser.add_argument("--test-sets", default="auto",
                        help=f"'auto', 'all' or comma-separated from: {', '.join(TEST_SETS)}")
    parser.add_argument("--starts", default="all",
                        help=f"Batch 1 starts to fly: 'all' or comma-separated from: {', '.join(TEST_STARTS)}")
    parser.add_argument("--flights", type=int, default=3, help="flights per Batch 1 start")
    parser.add_argument("--flight-steps", type=int, default=500,
                        help="control steps per flight (500 = 10 simulated seconds)")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: newest)")
    parser.add_argument("--snapshot", type=int, default=None,
                        help="test the agent kept at this many steps in snapshots/")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="noise seed for the test flights (default: training seed + 2000)")
    parser.add_argument("--no-mpc", action="store_true",
                        help="fly the learned policy network alone, without planning")
    parser.add_argument("--plan", default=None,
                        help="planning budget PLANSxROUNDS to use instead of the trained one, e.g. 128x3")
    parser.add_argument("--out-name", default=None,
                        help="output folder name inside run_dir (default: evaluation[_variant])")
    parser.add_argument("--overwrite", action="store_true", help="replace existing results")
    args = parser.parse_args()

    run_dir = args.run_dir
    if args.no_mpc and args.plan:
        parser.error("--no-mpc and --plan exclude each other")
    if args.step is not None and args.snapshot is not None:
        parser.error("--step and --snapshot exclude each other")
    batch1_names = list(TEST_STARTS) if args.starts == "all" else args.starts.split(",")
    unknown = [s for s in batch1_names if s not in TEST_STARTS]
    if unknown:
        parser.error(f"unknown start(s): {unknown}")

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    test_sets = args.test_sets
    if test_sets == "auto":
        test_sets = "all" if is_batch2_run(cfg) else "batch1"
    test_sets = list(TEST_SETS) if test_sets == "all" else test_sets.split(",")
    unknown = [s for s in test_sets if s not in TEST_SETS]
    if unknown:
        parser.error(f"unknown test set(s): {unknown}")

    # Results of a changed decision rule go to their own folder, so the
    # standard evaluation/ is never overwritten by a variant.
    variant = ""
    if args.no_mpc:
        variant = "_policy_only"
    elif args.plan:
        plans, rounds = (int(n) for n in args.plan.lower().split("x"))
        variant = f"_plan{plans}x{rounds}"
    if args.snapshot is not None:
        variant = f"_snapshot{args.snapshot}{variant}"
    out_dir = run_dir / (args.out_name or f"evaluation{variant}")
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")

    agent, step, cfg = load_agent(run_dir, args.step, args.snapshot)
    print(f"Loaded {run_dir} at {step} training steps (seed {cfg.seed}).")
    if args.plan:
        # Elites and policy-prior plans scale with the number of plans, as in
        # the two configs (128 plans: 16 elites; 512 plans: 64 elites).
        agent = agent.replace(population_size=plans, mppi_iterations=rounds,
                              num_elites=max(1, plans // 8),
                              policy_prior_samples=min(agent.policy_prior_samples, plans // 4))
        print(f"Planning with {plans} plans x {rounds} rounds instead of the trained budget.")
    decision = ("policy_only" if args.no_mpc
                else f"planner_{agent.population_size}x{agent.mppi_iterations}")

    starts = []
    for name in test_sets:
        if name == "batch1":
            starts += [(n, TEST_STARTS[n]) for n in batch1_names for _ in range(args.flights)]
        else:
            starts += TEST_SETS[name]

    # Training-time test flights use seed + 1000; use different noise here.
    test_seed = cfg.seed + 2000 if args.test_seed is None else args.test_seed
    metric_target = bool(cfg.env.get("metric_target", False))
    tester = FlightTester(flight_steps=args.flight_steps, mpc=not args.no_mpc,
                          metric_target=metric_target)
    flights = tester.fly(agent, starts, seed=test_seed)

    os.makedirs(out_dir, exist_ok=True)
    rows = write_summary(out_dir / "summary.csv", flights, seed=cfg.seed,
                         checkpoint_step=step, decision=decision)
    write_trajectories(out_dir / "trajectories.csv", flights)

    if len(rows) <= 30:
        print(f"\n{'start':<14} {'flight':>6} {'steps':>6} {'crash':>6} {'late err':>9} "
              f"{'xy err':>7} {'z err':>7} {'in 10cm':>8} {'return':>8}")
        for r in rows:
            print(f"{r['start']:<14} {r['flight']:>6} {r['steps_flown']:>6} {str(r['crashed']):>6} "
                  f"{r['late_error_m']:>8.3f}m {r['late_error_xy_m']:>6.3f}m {r['late_error_z_m']:>6.3f}m "
                  f"{r['time_in_10cm']:>7.0%} {r['return']:>8.3f}")
    print(f"\n{'test set':<11} {'flights':>7} {'crashes':>7} {'late err mean':>13} {'median':>7} {'in 10cm':>8}")
    for name in test_sets:
        mine = [r for r in rows if r["test_set"] == name]
        late = np.array([r["late_error_m"] for r in mine])
        inside = np.mean([r["time_in_10cm"] for r in mine])
        print(f"{name:<11} {len(mine):>7} {sum(r['crashed'] for r in mine):>7} "
              f"{late.mean():>12.3f}m {np.median(late):>6.3f}m {inside:>7.0%}")
    print(f"\nDecision rule: {decision}, obs target offset in "
          f"{'metres' if metric_target else 'start distances'}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
