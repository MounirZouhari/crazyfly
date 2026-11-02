"""Test every kept snapshot of a training run on a validation set and the test starts. Nothing is trained.

Batch 2 runs keep the agent every 100k steps (snapshots/). Flying each one
from many starts, with the same sensor noise for every snapshot, shows:
  - how much a run really changes between checkpoints (enough flights per
    snapshot that evaluation noise cannot pass for training swings)
  - whether the best checkpoint beats the final one, chosen on a validation
    set that is never used for testing
  - whether performance on unseen starts improves or degrades with training
Validation starts: 40 starts from the Batch 2 training distribution drawn with
their own seed (4242), separate from the test starts (hover_eval.TEST_SETS).

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_snapshots.py <run_dir>
    .venv/bin/python code/src/tdmpc2_training/eval_snapshots.py <run_dir> --no-mpc

Writes <run_dir>/snapshot_evaluation[_policy_only]/summary.csv, one row per
flight, with checkpoint_step and test_set (validation, train_area, held_out,
outside).
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import numpy as np

from crazyflie_hover import sample_training_start
from eval_hover import load_agent
from hover_eval import TEST_SETS, FlightTester, _write_rows

VALIDATION_SEED = 4242     # never change: it defines the validation starts
VALIDATION_SIZE = 40


def validation_starts():
    rng = np.random.default_rng(VALIDATION_SEED)
    return [(f"validation/{i:02d}", tuple(float(v) for v in sample_training_start(rng)))
            for i in range(VALIDATION_SIZE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-mpc", action="store_true", help="fly the policy network alone, without planning")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.run_dir / ("snapshot_evaluation_policy_only" if args.no_mpc else "snapshot_evaluation")
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")
    steps = sorted(int(p.name) for p in (args.run_dir / "snapshots").iterdir() if p.name.isdigit())
    if not steps:
        parser.error(f"no snapshots in {args.run_dir / 'snapshots'}")

    starts = validation_starts() + TEST_SETS["train_area"] + TEST_SETS["held_out"] + TEST_SETS["outside"]
    rows, tester = [], None
    for step in steps:
        agent, _, cfg = load_agent(args.run_dir, snapshot=step)
        if tester is None:
            tester = FlightTester(flight_steps=500, mpc=not args.no_mpc,
                                  metric_target=bool(cfg.env.get("metric_target", False)))
        decision = "policy_only" if args.no_mpc else f"planner_{agent.population_size}x{agent.mppi_iterations}"
        flights = tester.fly(agent, starts, seed=cfg.seed + 4000)   # same noise for every snapshot
        mine = [{"seed": cfg.seed, "checkpoint_step": step, "decision": decision, "flight": i, **f.summary()}
                for i, f in enumerate(flights)]
        rows += mine
        summary = []
        for name in ["validation", "train_area", "held_out", "outside"]:
            group = [r for r in mine if r["test_set"] == name]
            success = np.mean([(not r["crashed"]) and r["late_error_m"] < 0.10 for r in group])
            summary.append(f"{name} success {success:.0%} crash {np.mean([r['crashed'] for r in group]):.0%}")
        print(f"{step:>8}: " + ", ".join(summary), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _write_rows(out_dir / "summary.csv", rows, append=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
